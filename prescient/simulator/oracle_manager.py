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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from .options import Options
    from prescient.engine.modeling_engine import ModelingEngine
    from .data_manager import DataManager
    from .time_manager import PrescientTime
    from prescient.stats.operations_stats import OperationsStats


import os

from .manager import _Manager
from .data_manager import RucPlan

class OracleManager(_Manager):

    def initialize(self,
                  engine: ModelingEngine, 
                  datamanager: DataManager,
                  options: Options) -> None:

        self.data_manager = datamanager
        self.engine = engine

    def _get_ruc_delay(self, options):
        ''' The number of hours between the generation of a RUC plan and when it is activated '''
        return -(options.ruc_execution_hour % (-options.ruc_every_hours))

    def _get_uc_activation_time(self, options, time_step):
        ''' Get the hour, date, and next_date that a RUC generated at the given time will be activated '''
        ruc_delay = self._get_ruc_delay(options)
        activation_hour = time_step.hour + ruc_delay

        if (activation_hour < 24):
            # the RUC will go into effect the same day it is generated.
            uc_date = time_step.date
            next_uc_date = time_step.next_date
        else:
            # This is the final RUC of the day, which will go into effect at the beginning of the next day
            uc_date = time_step.next_date
            next_uc_date = time_step.next_next_date

        uc_hour = activation_hour % 24

        return (uc_hour, uc_date, next_uc_date)


    def _get_projected_sced_instance(self, options: Options, time_step: PrescientTime) -> Tuple[OperationsModel, int]:
        ''' Get the sced instance with initial conditions, and which hour of that sced that has the initial conditions'''
        
        ruc_delay = self._get_ruc_delay(options)

        # If there is no RUC delay, use the beginning of the current sced
        if ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:", time_step.date, "hour:", time_step.hour, "from prior SCED instance.")
            return (self.data_manager.prior_sced_instance, 1)

        uc_hour, uc_date, uc_next_date = self._get_uc_activation_time(options, time_step)

        # if this is the first hour of the day, we might (often) want to establish initial conditions from
        # something other than the prior sced instance. these reasons are not for purposes of realism, but
        # rather pragmatism. for example, there may be discontinuities in the "projected" initial condition
        # at this time (based on the stochastic RUC schedule being executed during the day) and that of the
        # actual sced instance.
        # if we're not in the first day, we should always simulate from the
        # state of the prior SCED.
        #
        # If this is the first date to simulate, we don't have anything
        # else to go off of when it comes to initial conditions - use those
        # found in the RUC.
        initialize_from_ruc = self.data_manager.prior_sced_instance is None

        print("")
        print("Creating and solving SCED to determine UC initial conditions for date:", uc_date, "hour:", uc_hour)

        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        use_prescient_forecast_error_in_sced = True  # always the default
        use_persistent_forecast_error_in_sced = False
        if options.run_sced_with_persistent_forecast_errors:
            print("Using persistent forecast error model when projecting demand and renewables in SCED")
            use_persistent_forecast_error_in_sced = True
            use_prescient_forecast_error_in_sced = False
        else:
            print("Using prescient forecast error model when projecting demand and renewables in SCED")
        print("")

        # NOTE: the projected sced probably doesn't have to be run for a full 24 hours - just enough
        #       to get you to midnight and a few hours beyond (to avoid end-of-horizon effects).
        projected_sced_instance, solve_time = self.engine.create_and_solve_sced_instance(
            self.data_manager.deterministic_ruc_instance_for_this_period,
            self.data_manager.scenario_tree_for_this_period,
            None, None, 
            self.data_manager.ruc_instance_to_simulate_this_period,
            self.data_manager.prior_sced_instance,
            self.data_manager.actual_demand,
            self.data_manager.demand_forecast_error,
            self.data_manager.actual_min_renewables,
            self.data_manager.actual_max_renewables,
            self.data_manager.renewables_forecast_error,
            time_step.hour % options.ruc_every_hours,
            options.reserve_factor,
            options,
            ## BK -- I'm not sure this was what it was. Don't we just want to
            ##       consider the next 24 hours or whatever?
            ## BK -- in the case of shorter ruc_horizons we may not have solutions
            ##       a full 24 hours out!
            hours_in_objective=min(24, options.ruc_horizon - time_step.hour % options.ruc_every_hours),
            sced_horizon=min(24, options.ruc_horizon - time_step.hour % options.ruc_every_hours),
            ruc_every_hours=options.ruc_every_hours,
            initialize_from_ruc=initialize_from_ruc,
            use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
            use_persistent_forecast_error=use_persistent_forecast_error_in_sced
           )

        return (projected_sced_instance, ruc_delay)


    def call_initialization_oracle(self, options: Options, time_step: PrescientTime):
        '''Calls the oracle to kick off the simulation, before rolling-horizon starts'''
        ########################################################################################
        # we need to create the "yesterday" deterministic or stochastic ruc instance, to kick  #
        # off the simulation process. for now, simply solve RUC for the first day, as          #
        # specified in the instance files. in practice, this may not be the best idea but it   #
        # will at least get us a solution to start from.                                       #
        ########################################################################################

        if options.run_ruc_with_next_day_data:
            print("Using next day data for 48 hour RUC solves")
        else:
            print("Using only the next 24 hours of data for RUC solves")

        #################################################################
        # construct the simulation data associated with the first date #
        #################################################################

        ruc_plan = self._generate_ruc(options, time_step.hour, time_step.date, time_step.next_date, None, None)

        self.data_manager.prior_sced_instance = None
        self.data_manager.set_current_ruc_plan(ruc_plan)
        self.data_manager.ruc_instance_to_simulate_this_period = ruc_plan.ruc_instance_to_simulate
        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        self.data_manager.set_actuals_for_new_ruc_instance()

        self.data_manager.deterministic_ruc_instance_for_this_period = ruc_plan.deterministic_ruc_instance
        self.data_manager.scenario_tree_for_this_period = ruc_plan.scenario_tree
        self.data_manager.set_forecast_errors_for_new_ruc_instance(options)

        self.data_manager.ruc_market_active = ruc_plan.ruc_market

        return ruc_plan

    def call_planning_oracle(self, options, time_step):
        projected_sced_instance, sced_schedule_hour = self._get_projected_sced_instance(options, time_step)

        uc_hour, uc_date, next_uc_date = self._get_uc_activation_time(options, time_step)

        return self._generate_ruc(options, uc_hour, uc_date, next_uc_date, projected_sced_instance, sced_schedule_hour)

    def _generate_ruc(self, options, uc_hour, uc_date, next_uc_date, projected_sced_instance, sced_schedule_hour):
        '''Creates a RUC plan by calling the oracle for the long-term plan based on forecast'''

        deterministic_ruc_instance, scenario_tree = self.engine.create_and_solve_deterministic_ruc(
                options,
                uc_date,
                uc_hour,
                next_uc_date,
                self.data_manager.deterministic_ruc_instance_for_this_period,
                self.data_manager.scenario_tree_for_this_period,
                options.output_ruc_initial_conditions,
                projected_sced_instance,
                sced_schedule_hour,
                options.ruc_horizon,
                options.run_ruc_with_next_day_data,
               )


        if options.compute_market_settlements:
            print("Solving day-ahead market")
            ruc_market = self.engine.create_and_solve_day_ahead_pricing(deterministic_ruc_instance, options)
        else:
            ruc_market = None
        # the RUC instance to simulate only exists to store the actual demand and renewables outputs
        # to be realized during the course of a day. it also serves to provide a concrete instance,
        # from which static data and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
        #            economic dispatch instances, as that would enable those instances to be
        #            prescient.
        print("")
        print("Creating RUC instance to simulate")

        ruc_instance_to_simulate = self.engine.create_ruc_instance_to_simulate_next_period(
            options,
            uc_date,
            uc_hour,
            next_uc_date,
           )

        return RucPlan(ruc_instance_to_simulate, scenario_tree, deterministic_ruc_instance, ruc_market)

    def call_operation_oracle(self, options: Options, time_step: PrescientTime, is_first_time_step:bool):
        # if this is the first hour of the day, we might (often) want to establish initial conditions from
        # something other than the prior sced instance. these reasons are not for purposes of realism, but
        # rather pragmatism. for example, there may be discontinuities in the "projected" initial condition
        # at this time (based on the stochastic RUC schedule being executed during the day) and that of the
        # actual sced instance.
        #
        # if this is the first date to simulate, we don't have anything
        # else to go off of when it comes to initial conditions, so use those
        # found in the RUC.  Otherwise we always simulate from the state of
        # the prior sced.
        initialize_from_ruc = is_first_time_step

        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        use_persistent_forecast_error_in_sced = options.run_sced_with_persistent_forecast_errors
        use_prescient_forecast_error_in_sced = not use_persistent_forecast_error_in_sced
        if use_persistent_forecast_error_in_sced:
            print("Using persistent forecast error model when projecting demand and renewables in SCED")
        else:
            print("Using prescient forecast error model when projecting demand and renewables in SCED")
        print("")

        lp_filename = None
        if options.write_sced_instances:
            lp_filename = options.output_directory + os.sep + str(time_step.date) + \
                os.sep + "sced_hour_" + str(time_step.hour) + ".lp"

        print("")
        print("Solving SCED instance")

        current_sced_instance, solve_time = self.engine.create_and_solve_sced_instance(
            self.data_manager.deterministic_ruc_instance_for_this_period,
            self.data_manager.scenario_tree_for_this_period,
            self.data_manager.deterministic_ruc_instance_for_next_period,
            self.data_manager.scenario_tree_for_next_period,
            self.data_manager.ruc_instance_to_simulate_this_period,
            self.data_manager.prior_sced_instance,
            self.data_manager.actual_demand,
            self.data_manager.demand_forecast_error,
            self.data_manager.actual_min_renewables,
            self.data_manager.actual_max_renewables,
            self.data_manager.renewables_forecast_error,
            time_step.hour % options.ruc_every_hours,
            options.reserve_factor,
            options,
            sced_horizon=options.sced_horizon,
            ruc_every_hours=options.ruc_every_hours,
            initialize_from_ruc=initialize_from_ruc,
            use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
            use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
            write_sced_instance = options.write_sced_instances,
            lp_filename=lp_filename,
            output_initial_conditions=options.output_sced_initial_conditions,
            output_demands=options.output_sced_demands
            )

        pre_quickstart_cache = None

        if options.enable_quick_start_generator_commitment:
            # Determine whether we are going to run a quickstart optimization
            # TODO: Why the "if True" here?
            if True or engine.operations_data_extractor.has_load_shedding(current_sced_instance):
                # Yep, we're doing it.  Cache data we can use to compare results with and without quickstart
                pre_quickstart_cache = engine.operations_data_extractor.get_pre_quickstart_data(current_sced_instance)

                # TODO: report solution/load shedding before unfixing Quick Start Generators
                # print("")
                # print("SCED Solution before unfixing Quick Start Generators")
                # print("")
                # self.report_sced_stats()

                # Set up the quickstart run, allowing quickstart generators to turn on
                print("Re-solving SCED after unfixing Quick Start Generators")
                current_sced_instance = self.engine.enable_quickstart_and_solve(sced_instance, options)


        print("Solving for LMPs")
        lmp_sced = self.engine.create_and_solve_lmp(current_sced_instance, options)

        ops_stats = self.simulator.stats_manager.collect_operations(current_sced_instance,
                                                                    solve_time,
                                                                    lmp_sced,
                                                                    pre_quickstart_cache,
                                                                    self.engine.operations_data_extractor)

        self._report_sced_stats(ops_stats)

        if options.compute_market_settlements:
            self.simulator.stats_manager.collect_market_settlement(current_sced_instance,
                    self.engine.operations_data_extractor,
                    self.simulator.data_manager.ruc_market_active,
                    time_step.hour % options.ruc_every_hours)


        return  current_sced_instance


    #####################################################################
    # utility functions for pretty-printing solutions, for SCED and RUC #
    #####################################################################


    def _report_sced_stats(self, ops_stats: OperationsStats):
        print("Fixed costs:    %12.2f" % ops_stats.fixed_costs)
        print("Variable costs: %12.2f" % ops_stats.variable_costs)
        print("")

        if ops_stats.load_shedding != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f" % (1, ops_stats.load_shedding))
        if ops_stats.over_generation!= 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f" % (1, ops_stats.over_generation))

        if ops_stats.reserve_shortfall != 0.0:
            print("Reserve shortfall reported at t=%2d: %12.2f" % (1, ops_stats.reserve_shortfall))
            print("Quick start generation capacity available at t=%2d: %12.2f" % (1, ops_stats.available_quickstart))
            print("")

        if ops_stats.renewables_curtailment > 0:
            print("Renewables curtailment reported at t=%d - total=%12.2f" % (1, ops_stats.renewables_curtailment))
            print("")

        print("Number on/offs:       %12d" % ops_stats.on_offs)
        print("Sum on/off ramps:     %12.2f" % ops_stats.sum_on_off_ramps)
        print("Sum nominal ramps:    %12.2f" % ops_stats.sum_nominal_ramps)
        print("")
