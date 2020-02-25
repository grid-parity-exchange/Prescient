#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
from .manager import _Manager
import os.path

from typing import NamedTuple

class RucPlan(NamedTuple):
   ruc_instance_to_simulate: RucModel
   scenario_tree: ScenarioTree
   deterministic_ruc_instance: RucModel

class DataManager(_Manager):
    def initialize(self, engine, options):
        self._ruc_stats_extractor = engine.ruc_data_extractor
        self.prior_sced_instance = None
        self.scenario_tree_for_this_period = None
        self.scenario_tree_for_next_period = None
        self.ruc_instance_to_simulate_this_period = None
        self.ruc_instance_to_simulate_next_period = None
        self.deterministic_ruc_instance_for_this_period = None
        self.deterministic_ruc_instance_for_next_period = None

    def update_time(self, time):
        '''This takes a Time object and makes the appropiate updates to the data'''
        self._current_time = time

    def set_current_ruc_plan(self, current_ruc_plan: RucPlan):
        self.ruc_instance_to_simulate_next_period = current_ruc_plan.ruc_instance_to_simulate
        self.scenario_tree_for_next_period = current_ruc_plan.scenario_tree
        self.deterministic_ruc_instance_for_next_period =current_ruc_plan.deterministic_ruc_instance

    def set_forecast_errors_for_new_ruc_instance(self, options) -> None:
        ''' Generate new forecast errors from current ruc instances '''
        forecast_ruc = self.deterministic_ruc_instance_for_this_period
        actuals_ruc = self.ruc_instance_to_simulate_this_period
        extractor = self._ruc_stats_extractor

        # print("NOTE: Positive forecast errors indicate projected values higher than actuals")
        self._demand_forecast_error = {}
        for b in extractor.get_buses(forecast_ruc):
            for t in range(1, extractor.get_num_time_periods(forecast_ruc) + 1):
                self._demand_forecast_error[b, t] = \
                    extractor.get_bus_demand(forecast_ruc, b, t) - \
                    extractor.get_bus_demand(actuals_ruc, b, t)

        self._renewables_forecast_error = {}
        for g in extractor.get_nondispatchable_generators(forecast_ruc):
            for t in range(1, extractor.get_num_time_periods(forecast_ruc) + 1):
                self._renewables_forecast_error[g, t] = \
                    extractor.get_max_nondispatchable_power(forecast_ruc, g, t) - \
                    extractor.get_max_nondispatchable_power(actuals_ruc, g, t)


    def update_forecast_errors_for_delayed_ruc(self, options):
        ''' update the demand and renewables forecast error dictionaries, using recently released forecasts '''
        actuals_ruc = self.ruc_instance_to_simulate_next_period
        forecast_ruc = self.deterministic_ruc_instance_for_next_period
        extractor = self._ruc_stats_extractor

        for b in extractor.get_buses(forecast_ruc):
            for t in range(1, options.ruc_every_hours + 1):
                self._demand_forecast_error[b, t + options.ruc_every_hours] = \
                    extractor.get_bus_demand(forecast_ruc, b, t) - \
                    extractor.get_bus_demand(actuals_ruc, b, t)

        for g in extractor.get_nondispatchable_generators(forecast_ruc):
            for t in range(1, options.ruc_every_hours + 1):
                self._renewables_forecast_error[g, t + options.ruc_every_hours] = \
                    extractor.get_max_nondispatchable_power(forecast_ruc, g, t) - \
                    extractor.get_max_nondispatchable_power(actuals_ruc, g, t)


    def set_actuals_for_new_ruc_instance(self) -> None:
        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        ruc = self.ruc_instance_to_simulate_this_period
        extractor = self._ruc_stats_extractor
        times = range(1, extractor.get_num_time_periods(ruc) + 1)

        self._actual_demand = {(b, t): extractor.get_bus_demand(ruc, b, t)
                               for b in extractor.get_buses(ruc)
                               for t in times}

        self._actual_min_renewables = {}
        self._actual_max_renewables = {}
        for g in extractor.get_nondispatchable_generators(ruc):
            for t in times:
                self._actual_min_renewables[g, t] = extractor.get_min_nondispatchable_power(ruc, g, t)
                self._actual_max_renewables[g, t] = extractor.get_max_nondispatchable_power(ruc, g, t)

    def update_actuals_for_delayed_ruc(self, options):
        # update the second 'ruc_every_hours' hours of the current actual demand/renewables vectors
        extractor = self._ruc_stats_extractor
        actuals_ruc = self.ruc_instance_to_simulate_next_period

        for t in range(1, options.ruc_every_hours+1):
            for b in extractor.get_buses(actuals_ruc):
                self._actual_demand[b, t+options.ruc_every_hours] = extractor.get_bus_demand(actuals_ruc, b, t)
            for g in extractor.get_nondispatchable_generators(actuals_ruc):
                self._actual_min_renewables[g, t+options.ruc_every_hours] = extractor.get_min_nondispatchable_power(actuals_ruc, g, t)
                self._actual_max_renewables[g, t+options.ruc_every_hours] = extractor.get_max_nondispatchable_power(actuals_ruc, g, t)

    def clear_instances_for_next_period(self):
        self.ruc_instance_to_simulate_next_period = None
        self.scenario_tree_for_next_period = None
        self.deterministic_ruc_instance_for_next_period = None


    ##########
    # Properties
    ##########
    @property
    def current_time(self):
        return self._current_time

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
    def demand_forecast_error(self):
        return self._demand_forecast_error

    @property
    def renewables_forecast_error(self):
        return self._renewables_forecast_error
