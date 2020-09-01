#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## Abstract classes which define a simulation
from __future__ import annotations

import os

from .time_manager import TimeManager
from .data_manager import DataManager
from prescient.engine.modeling_engine import ModelingEngine
from .oracle_manager import OracleManager
from .reporting_manager import ReportingManager
from .stats_manager import StatsManager
import prescient.plugins

## simulator class
class Simulator:

    def __init__(self, model_engine: ModelingEngine,
                       time_manager: TimeManager, 
                       data_manager: DataManager, 
                       oracle_manager: OracleManager,
                       stats_manager: StatsManager,
                       reporting_manager: ReportingManager
                       ):

        print("Initializing simulation...")

        import time
        self.simulation_start_time = time.time()

        if not isinstance(time_manager, TimeManager):
            raise RuntimeError("The time_manager must be an instance of class simulator.TimeManager")
        if not isinstance(data_manager, DataManager):
            raise RuntimeError("The data_manager must be an instance of class simulator.DataManager")
        if not isinstance(oracle_manager, OracleManager):
            raise RuntimeError("The oracle_manager must be an instance of class simulator.OracleManager")
        if not isinstance(stats_manager, StatsManager):
            raise RuntimeError("The stats_manager must be an instance of class simulator.StatsManager")
        if not isinstance(model_engine, ModelingEngine):
            raise RuntimeError("The model_engine must be an instance of class simulator.ModelingEngine")
        
        time_manager.set_simulator(self)
        data_manager.set_simulator(self)
        oracle_manager.set_simulator(self)
        stats_manager.set_simulator(self)
        reporting_manager.set_simulator(self)

        self.engine = model_engine
        self.time_manager = time_manager
        self.data_manager = data_manager
        self.oracle_manager = oracle_manager
        self.stats_manager = stats_manager
        self.reporting_manager = reporting_manager

        self.plugin_manager = prescient.plugins.get_active_plugin_manager()


    def simulate(self, options):

        engine = self.engine
        time_manager = self.time_manager
        data_manager = self.data_manager
        oracle_manager = self.oracle_manager
        stats_manager = self.stats_manager
        reporting_manager = self.reporting_manager

        self.plugin_manager.invoke_options_preview_callbacks(options)

        engine.initialize(options)
        time_manager.initialize(options)
        data_manager.initialize(engine, options)
        oracle_manager.initialize(engine, data_manager, options)
        stats_manager.initialize(options)
        reporting_manager.initialize(options, stats_manager)

        self.plugin_manager.invoke_initialization_callbacks(options, self)

        first_time_step = time_manager.get_first_time_step()

        current_ruc_plan = oracle_manager.call_initialization_oracle(options, first_time_step)

        for time_step in time_manager.time_steps():
            print("Simulating time_step " + time_step.date + " " + str(time_step.hour+1))

            stats_manager.begin_timestep(time_step)

            is_first_time_step = time_manager.is_first_time_step(time_step)

            data_manager.update_time(time_step)

            if time_step.is_planning_time and not is_first_time_step:
                current_ruc_plan = oracle_manager.call_planning_oracle(options, time_step)
                data_manager.deterministic_ruc_instance_for_next_period = current_ruc_plan.deterministic_ruc_instance
                data_manager.scenario_tree_for_next_period = current_ruc_plan.scenario_tree
                data_manager.ruc_instance_to_simulate_next_period = current_ruc_plan.ruc_instance_to_simulate
                data_manager.ruc_market_pending = current_ruc_plan.ruc_market

                # If there is a RUC delay...
                if options.ruc_execution_hour % options.ruc_every_hours > 0:
                    data_manager.update_actuals_for_delayed_ruc(options)
                    data_manager.update_forecast_errors_for_delayed_ruc(options)


            #normally, we want to wait for the ruc execution hour
            # but for now, we're running with ruc_execution hour = 0
            # so we can do this hand-over right away
            #normally we want to look at the ruc_start_hours
            if time_step.is_ruc_start_hour and not is_first_time_step:
                # establish the stochastic ruc instance for this_period - we use this instance to track,
                # for better or worse, the projected and actual UnitOn states through the day.
                data_manager.ruc_instance_to_simulate_this_period = current_ruc_plan.ruc_instance_to_simulate

                # initialize the actual demand and renewables vectors - these will be incrementally
                # updated when new forecasts are released, e.g., when the next RUC is computed.
                data_manager.set_actuals_for_new_ruc_instance()

                data_manager.deterministic_ruc_instance_for_this_period = current_ruc_plan.deterministic_ruc_instance
                data_manager.scenario_tree_for_this_period = current_ruc_plan.scenario_tree
                data_manager.set_forecast_errors_for_new_ruc_instance(options)

                data_manager.ruc_market_active = data_manager.ruc_market_pending

                data_manager.clear_instances_for_next_period()

            # We call the operations oracle at all time steps
            current_sced_instance = oracle_manager.call_operation_oracle(options, time_step, is_first_time_step)
            data_manager.prior_sced_instance = current_sced_instance

            stats_manager.end_timestep(time_step)

        stats_manager.end_simulation()

        print("Simulation Complete")
        import time
        self.simulation_start_time = time.time()
