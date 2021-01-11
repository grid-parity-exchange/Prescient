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
from datetime import timedelta

from .manager import _Manager
from .data_manager import RucPlan
from prescient.engine.modeling_engine import ForecastErrorMethod
from prescient.data.simulation_state import SimulationState, StateWithOffset

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
        ''' Get the hour and date that a RUC generated at the given time will be activated '''
        ruc_delay = self._get_ruc_delay(options)
        activation_time = time_step.datetime + timedelta(hours=ruc_delay)

        return (activation_time.hour, activation_time.date())

    def _get_projected_state(self, options: Options, time_step: PrescientTime) -> SimulationState:
        ''' Get the simulation state as we project it will appear after the RUC delay '''
        
        ruc_delay = self._get_ruc_delay(options)

        # If there is no RUC delay, use the current state as is
        if ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:", time_step.date, "hour:", time_step.hour, "from prior SCED instance.")
            return self.data_manager.current_state

        uc_hour, uc_date = self._get_uc_activation_time(options, time_step)

        print("")
        print("Creating and solving SCED to determine UC initial conditions for date:", str(uc_date), "hour:", uc_hour)

        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        sced_forecast_error_method = ForecastErrorMethod.PRESCIENT # always the default
        if options.run_sced_with_persistent_forecast_errors:
            print("Using persistent forecast error model when projecting demand and renewables in SCED")
            sced_forecast_error_method = ForecastErrorMethod.PERSISTENT
        else:
            print("Using prescient forecast error model when projecting demand and renewables in SCED")
        print("")

        # NOTE: the projected sced probably doesn't have to be run for a full 24 hours - just enough
        #       to get you to midnight and a few hours beyond (to avoid end-of-horizon effects).
        #       But for now we run for 24 hours.
        current_state = self.data_manager.current_state.get_state_with_step_length(60)
        projected_sced_instance = self.engine.create_sced_instance(
            options,
            current_state,
            hours_in_objective=min(24, current_state.timestep_count),
            sced_horizon=min(24, current_state.timestep_count),
            forecast_error_method=sced_forecast_error_method
           )

        projected_sced_instance, solve_time = self.engine.solve_sced_instance(options, projected_sced_instance)

        future_state = StateWithOffset(current_state, projected_sced_instance, ruc_delay)

        return future_state


    def call_initialization_oracle(self, options: Options, time_step: PrescientTime):
        '''Calls the oracle to kick off the simulation, before rolling-horizon starts'''
        ########################################################################################
        # we need to create the "yesterday" deterministic or stochastic ruc instance, to kick  #
        # off the simulation process. for now, simply solve RUC for the first day, as          #
        # specified in the instance files. in practice, this may not be the best idea but it   #
        # will at least get us a solution to start from.                                       #
        ########################################################################################

        # Print one-time information about simulation options
        if options.run_ruc_with_next_day_data:
            print("Using next day forecasts for 48 hour RUC solves")
        else:
            print("Using current day's forecasts for RUC solves")
        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        if options.run_sced_with_persistent_forecast_errors:
            print("Using persistent forecast error model when projecting demand and renewables in SCED")
        else:
            print("Using prescient forecast error model when projecting demand and renewables in SCED")
        print("")

        #################################################################
        # construct the simulation data associated with the first date #
        #################################################################

        ruc_plan = self._generate_ruc(options, time_step.date, time_step.hour, None)

        self.data_manager.set_pending_ruc_plan(options, ruc_plan)
        self.data_manager.activate_pending_ruc(options)

        self.simulator.plugin_manager.invoke_after_ruc_activation_callbacks(options, self.simulator)
        return ruc_plan

    def call_planning_oracle(self, options: Options, time_step: PrescientTime):
        ''' Create a new RUC and make it the pending RUC
        '''
        projected_state = self._get_projected_state(options, time_step)

        uc_hour, uc_date = self._get_uc_activation_time(options, time_step)

        ruc = self._generate_ruc(options, uc_date, uc_hour, projected_state)
        self.data_manager.set_pending_ruc_plan(options, ruc)

        return ruc

    def _generate_ruc(self, options, uc_date, uc_hour, sim_state_for_ruc):
        '''Creates a RUC plan by calling the oracle for the long-term plan based on forecast'''

        deterministic_ruc_instance = self.engine.create_deterministic_ruc(
                options,
                uc_date,
                uc_hour,
                sim_state_for_ruc,
                options.output_ruc_initial_conditions,
                options.ruc_horizon,
                options.run_ruc_with_next_day_data,
               )

        self.simulator.plugin_manager.invoke_before_ruc_solve_callbacks(options, self.simulator, deterministic_ruc_instance, uc_date, uc_hour)

        deterministic_ruc_instance = self.engine.solve_deterministic_ruc(
                options,
                deterministic_ruc_instance,
                uc_date,
                uc_hour
               )

        if options.compute_market_settlements:
            print("Solving day-ahead market")
            ruc_market = self.engine.create_and_solve_day_ahead_pricing(options, deterministic_ruc_instance)
        else:
            ruc_market = None
        # the RUC instance to simulate only exists to store the actual demand and renewables outputs
        # to be realized during the course of a day. it also serves to provide a concrete instance,
        # from which static data and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
        #            economic dispatch instances, as that would enable those instances to be
        #            prescient.
        print("")
        print("Extracting scenario to simulate")

        simulation_actuals = self.engine.create_simulation_actuals(
            options,
            uc_date,
            uc_hour,
           )

        result = RucPlan(simulation_actuals, deterministic_ruc_instance, ruc_market)
        self.simulator.plugin_manager.invoke_after_ruc_generation_callbacks(options, self.simulator, result, uc_date, uc_hour)
        return result

    def activate_pending_ruc(self, options: Options):
        self.data_manager.activate_pending_ruc(options)
        self.simulator.plugin_manager.invoke_after_ruc_activation_callbacks(options, self.simulator)

    def call_operation_oracle(self, options: Options, time_step: PrescientTime):
        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        if options.run_sced_with_persistent_forecast_errors:
            forecast_error_method = ForecastErrorMethod.PERSISTENT
        else:
            forecast_error_method = ForecastErrorMethod.PRESCIENT

        lp_filename = None
        if options.write_sced_instances:
            lp_filename = options.output_directory + os.sep + str(time_step.date) + \
                os.sep + "sced_hour_" + str(time_step.hour) + ".lp"

        print("")
        print("Solving SCED instance")

        sced_horizon_timesteps = options.sced_horizon * 60 // options.sced_frequency_minutes
        current_sced_instance = self.engine.create_sced_instance(
            options,
            self.data_manager.current_state.get_state_with_step_length(options.sced_frequency_minutes),
            hours_in_objective=1,
            sced_horizon=sced_horizon_timesteps,
            forecast_error_method=forecast_error_method,
            write_sced_instance = options.write_sced_instances,
            lp_filename=lp_filename
            )

        self.simulator.plugin_manager.invoke_before_operations_solve_callbacks(options, self.simulator, current_sced_instance)

        current_sced_instance, solve_time = self.engine.solve_sced_instance(options, current_sced_instance, 
                                                                            options.output_sced_initial_conditions,
                                                                            options.output_sced_demands,
                                                                            lp_filename)

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
                current_sced_instance = self.engine.enable_quickstart_and_solve(options, current_sced_instance)

        print("Solving for LMPs")
        lmp_sced = self.engine.create_and_solve_lmp(options, current_sced_instance)

        self.data_manager.apply_sced(options, current_sced_instance)
        self.simulator.plugin_manager.invoke_after_operations_callbacks(options, self.simulator, current_sced_instance)

        ops_stats = self.simulator.stats_manager.collect_operations(current_sced_instance,
                                                                    solve_time,
                                                                    lmp_sced,
                                                                    pre_quickstart_cache,
                                                                    self.engine.operations_data_extractor)

        self.simulator.plugin_manager.invoke_update_operations_stats_callbacks(options, self.simulator, ops_stats)
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
