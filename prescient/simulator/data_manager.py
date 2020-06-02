#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from .manager import _Manager
import os.path
from pyomo.core import value

class DataManager(_Manager):
    def initialize(self, options):

        model_filename = os.path.join(options.model_directory, "ReferenceModel.py")
        if not os.path.exists(model_filename):
            raise RuntimeError("The model %s either does not exist or cannot be read" % model_filename)
        
        from pyutilib.misc import import_file
        self._reference_model_module = import_file(model_filename)
        # make sure all utility methods required by the simulator are defined in the reference model module.
        self.validate_reference_model(self._reference_model_module)
        self._ruc_model = self._reference_model_module.load_model_parameters()
        self._sced_model = self._reference_model_module.model
        self._prior_sced_instance = None
        self._current_sced_instance = None
        self._scenario_instances_for_this_period = None
        self._scenario_instances_for_next_period = None
        self._scenario_tree_for_this_period = None
        self._scenario_tree_for_next_period = None
        self._ruc_instance_to_simulate_this_period = None
        self._deterministic_ruc_instance_for_next_period = None
        self._deterministic_ruc_instance_for_this_period = None
        self._prior_sced_instance = None

    def update_time(self, time):
        '''This takes a Time object and makes the appropiate updates to the data'''
        self._current_time = time

    def set_forecast_errors_for_new_ruc_instance(self, options):
        ''' Generate new forecast errors from current ruc instances '''
        if options.run_deterministic_ruc:
            print("")
            # print("NOTE: Positive forecast errors indicate projected values higher than actuals")
            demand_forecast_error = {}  # maps (bus,time-period) pairs to an error, defined as forecast minus actual
            for b in self.deterministic_ruc_instance_for_this_period.Buses:
                for t in range(1, value(self.deterministic_ruc_instance_for_this_period.NumTimePeriods) + 1):  # TBD - not sure how to option-drive the upper bound on the time period value.
                    demand_forecast_error[b, t] = value(self.deterministic_ruc_instance_for_this_period.Demand[b, t]) - \
                                                  value(self.ruc_instance_to_simulate_this_period.Demand[b, t])
                    # print("Demand forecast error for bus=%s at t=%2d: %12.2f" % (b, t, demand_forecast_error[b,t]))

            print("")

            renewables_forecast_error = {}
            # maps (generator,time-period) pairs to an error, defined as forecast minus actual
            for g in self.deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators:
                for t in range(1, value(self.deterministic_ruc_instance_for_this_period.NumTimePeriods) + 1):
                    renewables_forecast_error[g, t] = \
                        value(self.deterministic_ruc_instance_for_this_period.MaxNondispatchablePower[g, t]) - \
                        value(self.ruc_instance_to_simulate_this_period.MaxNondispatchablePower[g, t])
                    # print("Renewables forecast error for generator=%s at t=%2d: %12.2f" % (g, t, renewables_forecast_error[g, t]))
        else:
            demand_forecast_error = None
            renewables_forecast_error = None

        self._renewables_forecast_error = renewables_forecast_error
        self._demand_forecast_error = demand_forecast_error


    def update_forecast_errors_for_delayed_ruc(self, options):
        ''' update the demand and renewables forecast error dictionaries, using the recently released forecasts '''
        if options.run_deterministic_ruc:
            print("")
            print("Updating forecast errors")
            print("")
            for b in sorted(self._deterministic_ruc_instance_for_this_period.Buses):
                for t in range(1, 1 + options.ruc_every_hours):
                    self._demand_forecast_error[b, t + options.ruc_every_hours] = \
                        value(self._deterministic_ruc_instance_for_next_period.Demand[b, t]) - \
                        value(self._ruc_instance_to_simulate_next_period.Demand[b, t])
                    # print("Demand forecast error for bus=%s at t=%2d: %12.2f"
                    #      % (b, t, demand_forecast_error[b, t+options.ruc_every_hours]))

            print("")

            for g in sorted(self._deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators):
                for t in range(1, 1 + options.ruc_every_hours):
                    self._renewables_forecast_error[g, t + options.ruc_every_hours] = \
                        value(self._deterministic_ruc_instance_for_next_period.MaxNondispatchablePower[g, t]) - \
                        value(self._ruc_instance_to_simulate_next_period.MaxNondispatchablePower[g, t])
                    # print("Renewables forecast error for generator=%s at t=%2d: %12.2f"
                    #      % (g, t, renewables_forecast_error[g, t+options.ruc_every_hours]))

    def set_actuals_for_new_ruc_instance(self):
        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        self._actual_demand = dict(((b, t), value(self._ruc_instance_to_simulate_this_period.Demand[b, t]))
                                  for b in self._ruc_instance_to_simulate_this_period.Buses
                                  for t in self._ruc_instance_to_simulate_this_period.TimePeriods)
        self._actual_min_renewables = dict(((g, t), value(self._ruc_instance_to_simulate_this_period.MinNondispatchablePower[g, t]))
                                          for g in self._ruc_instance_to_simulate_this_period.AllNondispatchableGenerators
                                          for t in self._ruc_instance_to_simulate_this_period.TimePeriods)
        self._actual_max_renewables = dict(((g, t), value(self._ruc_instance_to_simulate_this_period.MaxNondispatchablePower[g, t]))
                                          for g in self._ruc_instance_to_simulate_this_period.AllNondispatchableGenerators
                                          for t in self._ruc_instance_to_simulate_this_period.TimePeriods)

    def update_actuals_for_delayed_ruc(self, options):
        # update the second 'ruc_every_hours' hours of the current actual demand/renewables vectors
        for t in range(1,1+options.ruc_every_hours):
            for b in sorted(self.ruc_instance_to_simulate_next_period.Buses):
                self._actual_demand[b, t+options.ruc_every_hours] = value(self.ruc_instance_to_simulate_next_period.Demand[b,t])
            for g in sorted(self.ruc_instance_to_simulate_next_period.AllNondispatchableGenerators):
                self._actual_min_renewables[g, t+options.ruc_every_hours] = \
                                    value(self.ruc_instance_to_simulate_next_period.MinNondispatchablePower[g, t])
                self._actual_max_renewables[g, t+options.ruc_every_hours] = \
                                    value(self.ruc_instance_to_simulate_next_period.MaxNondispatchablePower[g, t])

    def clear_instances_for_next_period(self):
        self._ruc_instance_to_simulate_next_period = None
        self._scenario_tree_for_next_period = None
        self._scenario_instances_for_next_period = None
        self._deterministic_ruc_instance_for_next_period = None

    def validate_reference_model(self, module):
        required_methods = ["fix_binary_variables", "free_binary_variables", "status_var_generator", "define_suffixes",
                            "load_model_parameters"]
        for method in required_methods:
            if not hasattr(module, method):
                raise RuntimeError("Reference model module does not have required method=%s" % method)


    ##########
    # Getters and Setters
    ##########
    @property
    def current_time(self):
        return self._current_time

    @property
    def deterministic_ruc_instance_for_this_period(self):
        return self._deterministic_ruc_instance_for_this_period

    @deterministic_ruc_instance_for_this_period.setter
    def deterministic_ruc_instance_for_this_period(self, value):
        self._deterministic_ruc_instance_for_this_period = value

    @property
    def deterministic_ruc_instance_for_next_period(self):
        ### TODO add rollover to set next to this, etc
        return self._deterministic_ruc_instance_for_next_period

    @deterministic_ruc_instance_for_next_period.setter
    def deterministic_ruc_instance_for_next_period(self, value):
        self._deterministic_ruc_instance_for_next_period = value

    @property
    def scenario_tree_for_next_period(self):
        ### TODO add rollover to set next to this, etc
        return self._scenario_tree_for_next_period

    @scenario_tree_for_next_period.setter
    def scenario_tree_for_next_period(self, value):
        self._scenario_tree_for_next_period = value

    @property
    def scenario_tree_for_this_period(self):
        return self._scenario_tree_for_this_period

    @scenario_tree_for_this_period.setter
    def scenario_tree_for_this_period(self, value):
        self._scenario_tree_for_this_period = value

    @property
    def scenario_instances_for_this_period(self):
        return self._scenario_instances_for_this_period

    @property
    def scenario_instances_for_next_period(self):
        return self._scenario_instances_for_next_period

    @property
    def sced_model(self):
        return self._sced_model

    @scenario_instances_for_next_period.setter
    def scenario_instances_for_next_period(self, value):
        self._scenario_instances_for_next_period = value

    @property
    def actual_demand(self):
        return self._actual_demand

    @property
    def actual_min_renewables(self):
        return self._actual_min_renewables

    @property
    def actual_max_renewables(self):
        return self._actual_max_renewables

    @property
    def ruc_instance_to_simulate_this_period(self):
        return self._ruc_instance_to_simulate_this_period

    @ruc_instance_to_simulate_this_period.setter
    def ruc_instance_to_simulate_this_period(self, value):
        self._ruc_instance_to_simulate_this_period = value

    @property
    def ruc_instance_to_simulate_next_period(self):
        return self._ruc_instance_to_simulate_next_period

    @ruc_instance_to_simulate_next_period.setter
    def ruc_instance_to_simulate_next_period(self, value):
        self._ruc_instance_to_simulate_next_period = value

    @property
    def prior_sced_instance(self):
        return self._prior_sced_instance

    @prior_sced_instance.setter
    def prior_sced_instance(self, value):
        self._prior_sced_instance = value

    @property
    def current_sced_instance(self):
        return self._current_sced_instance

    @current_sced_instance.setter
    def current_sced_instance(self, value):
        self._current_sced_instance = value

    @property
    def demand_forecast_error(self):
        return self._demand_forecast_error

    @property
    def renewables_forecast_error(self):
        return self._renewables_forecast_error

    @property
    def reference_model_module(self):
        return self._reference_model_module

    @property
    def ruc_model(self):
        return self._ruc_model

    def set_current_ruc_plan(self, current_ruc_plan):
        self.ruc_instance_to_simulate_next_period = current_ruc_plan.ruc_instance_to_simulate_next_period
        self.scenario_tree_for_next_period = current_ruc_plan.scenario_tree_for_next_period
        self.scenario_instances_for_next_period = current_ruc_plan.scenario_instances_for_next_period
        self.deterministic_ruc_instance_for_next_period =current_ruc_plan.deterministic_ruc_instance_for_next_period
