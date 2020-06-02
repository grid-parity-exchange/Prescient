#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from .manager import _Manager
import os
from pyutilib.misc import import_file
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
from pyomo.core import value
from collections import namedtuple
from pyomo.opt import SolverFactory
from prescient.stats.operations_stats import OperationsStats
from prescient.stats.stats_extractors import OperationsStatsExtractor as StatsExtractor
import pyutilib
import math
from prescient.util import DEFAULT_MAX_LABEL_LENGTH

class OracleManager(_Manager):

    def initialize(self, options, datamanager):

        self.data_manager = datamanager
        self._setup_plugin_methods(options)

        # NOTE: We eventually want specific solver options for the various types below.
        if options.python_io:
            self._deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type, solver_io="python")
            self._ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type, solver_io="python")
            self._sced_solver = SolverFactory(options.sced_solver_type, solver_io="python")
        else:
            self._deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type)
            self._ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type)
            self._sced_solver = SolverFactory(options.sced_solver_type)

        solve_options = {}
        solve_options[self._sced_solver] = {}
        solve_options[self._deterministic_ruc_solver] = {}
        solve_options[self._ef_ruc_solver] = {}

        for s in solve_options:
            solve_options[s]['symbolic_solver_labels'] = options.symbolic_solver_labels

        ##TODO: work out why we get an error from ProblemWriter_cpxlp when this is False
        if options.warmstart_ruc:
            solve_options[self._deterministic_ruc_solver]['warmstart'] = options.warmstart_ruc
            solve_options[self._ef_ruc_solver]['warmstart'] = options.warmstart_ruc

        if options.deterministic_ruc_solver_type == "cplex" or options.deterministic_ruc_solver_type == "cplex_persistent":
            self._deterministic_ruc_solver.options.mip_tolerances_mipgap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "gurobi" or options.deterministic_ruc_solver_type == "gurobi_persistent":
            self._deterministic_ruc_solver.options.MIPGap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "cbc":
            self._deterministic_ruc_solver.options.ratioGap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "glpk":
            self._deterministic_ruc_solver.options.mipgap = options.ruc_mipgap
        else:
            raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)

        if options.ef_ruc_solver_type == "cplex" or options.ef_ruc_solver_type == "cplex_persistent":
            self._ef_ruc_solver.options.mip_tolerances_mipgap = options.ef_mipgap
        elif options.ef_ruc_solver_type == "gurobi" or options.ef_ruc_solver_type == "gurobi_persistent":
            self._ef_ruc_solver.options.MIPGap = options.ef_mipgap
        elif options.ef_ruc_solver == "cbc":
            self._ef_ruc_solver.options.ratioGap = options.ruc_mipgap
        elif options.ef_ruc_solver == "glpk":
            self._ef_ruc_solver.options.mipgap = options.ruc_mipgap
        else:
            raise RuntimeError("Unknown solver type=%s specified" % options.ef_ruc_solver_type)

        self._ef_ruc_solver.set_options("".join(options.stochastic_ruc_ef_solver_options))
        self._deterministic_ruc_solver.set_options("".join(options.deterministic_ruc_solver_options))
        self._sced_solver.set_options("".join(options.sced_solver_options))

        self._solve_options = solve_options

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


    def _get_projected_sced_instance(self, options, time_step):
        ''' Get the sced instance with initial conditions and the hour of that sced '''
        
        ruc_delay = self._get_ruc_delay(options)

        # If there is no RUC delay, use the beginning of the current sced
        if ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:", time_step.date, "hour:", time_step.hour, "from prior SCED instance.")
            return (self.data_manager.current_sced_instance, 1)

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
        projected_sced_instance = self.create_sced_instance(
            self.data_manager.sced_model,
            self.data_manager.reference_model_module,
            self.data_manager.deterministic_ruc_instance_for_this_period,
            self.data_manager.scenario_instances_for_this_period,
            self.data_manager.scenario_tree_for_this_period,
            None, None, None,
            # we're setting up for next_period - it's not here yet!
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
            hours_in_objective=min(24,
            options.ruc_horizon - time_step.hour % options.ruc_every_hours),
            sced_horizon=min(24, options.ruc_horizon - time_step.hour % options.ruc_every_hours),
            ruc_every_hours=options.ruc_every_hours,
            initialize_from_ruc=initialize_from_ruc,
            use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
            use_persistent_forecast_error=use_persistent_forecast_error_in_sced
           )

        sced_results = self.call_solver(self._sced_solver,
                                        projected_sced_instance, 
                                        tee=options.output_solver_logs,
                                        keepfiles=options.keep_solver_files,
                                        **self._solve_options[self._sced_solver])

        projected_sced_instance.solutions.load_from(sced_results)

        return (projected_sced_instance, ruc_delay)


    def call_initialization_oracle(self, options, time_step):
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

        self.data_manager.current_sced_instance = None
        self.data_manager.prior_sced_instance = None
        self.data_manager.set_current_ruc_plan(ruc_plan)
        self.data_manager.ruc_instance_to_simulate_this_period = ruc_plan.ruc_instance_to_simulate_next_period
        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        self.data_manager.set_actuals_for_new_ruc_instance()

        # TODO: to simplify we may just get rid of this option for now... and then add back in after the refactor.
        if options.run_deterministic_ruc:
            self.data_manager.deterministic_ruc_instance_for_this_period = ruc_plan.deterministic_ruc_instance_for_next_period
            self.data_manager.scenario_tree_for_this_period = ruc_plan.scenario_tree_for_next_period
        else:
            self.data_manager.scenario_instances_for_this_period = ruc_plan.scenario_instances_for_next_period
            self.data_manager.scenario_tree_for_this_period = ruc_plan.scenario_tree_for_next_period

        self.data_manager.set_forecast_errors_for_new_ruc_instance(options)

        return ruc_plan

    def call_planning_oracle(self, options, time_step):
        projected_sced_instance, sced_schedule_hour = self._get_projected_sced_instance(options, time_step)


        uc_hour, uc_date, next_uc_date = self._get_uc_activation_time(options, time_step)

        return self._generate_ruc(options, uc_hour, uc_date, next_uc_date, projected_sced_instance, sced_schedule_hour)

    def _generate_ruc(self, options, uc_hour, uc_date, next_uc_date, projected_sced_instance, sced_schedule_hour):
        '''Creates a RUC plan by calling the oracle for the long-term plan based on forecast'''

        scenario_instances_for_next_period = None
        scenario_tree_for_next_period = None
        deterministic_ruc_instance_for_next_period = None

        if options.run_deterministic_ruc:
            deterministic_ruc_instance_for_next_period, \
            scenario_tree_for_next_period = self.create_and_solve_deterministic_ruc(
                self._deterministic_ruc_solver,
                self._solve_options,
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

        else:
            ##SOLVER CALL##
            if options.solve_with_ph == False:
                scenario_instances_for_next_period, \
                scenario_tree_for_next_period = self.create_and_solve_stochastic_ruc_via_ef(
                    self._ef_ruc_solver,
                    self._solve_options,
                    options,
                    uc_date,
                    uc_hour,
                    next_uc_date,
                    self.data_manager.scenario_instances_for_this_period,
                    self.data_manager.scenario_tree_for_this_period,
                    options.output_ruc_initial_conditions,
                    projected_sced_instance,
                    self.data_manager.sced_schedule_hour,
                    options.ruc_horizon,
                    options.run_ruc_with_next_day_data,
                   )
            else:
                scenario_instances_for_next_period, \
                scenario_tree_for_next_period = self.create_and_solve_stochastic_ruc_via_ph(
                    self.solver,
                    None,
                    options,
                    uc_date,
                    uc_hour,
                    next_uc_date,
                    self.data_manager.scenario_instances_for_this_period,
                    self.data_manager.scenario_tree_for_this_period,
                    options.output_ruc_initial_conditions,
                    projected_sced_instance,
                    sced_schedule_hour,
                    options.ruc_horizon,
                    options.run_ruc_with_next_day_data,
                   )

        # identify the .dat file from which (simulated) actual load data will be drawn.
        simulated_dat_filename = self.compute_simulation_filename_for_date(uc_date, options)

        print("")
        print("Actual simulation data drawn from file=" + simulated_dat_filename)

        if not os.path.exists(simulated_dat_filename):
            raise RuntimeError("The file " + simulated_dat_filename + " does not exist or cannot be read.")

        # the RUC instance to simulate only exists to store the actual demand and renewables outputs
        # to be realized during the course of a day. it also serves to provide a concrete instance,
        # from which static data and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
        #            economic dispatch instances, as that would enable those instances to be
        #            prescient.
        print("")
        print("Creating RUC instance to simulate")

        ruc_instance_to_simulate_next_period = self.create_ruc_instance_to_simulate_next_period(
            self.data_manager.ruc_model,
            options,
            uc_date,
            uc_hour,
            next_uc_date,
           )

        ruc_plan = namedtuple('ruc_plan', 'ruc_instance_to_simulate_next_period scenario_tree_for_next_period scenario_instances_for_next_period deterministic_ruc_instance_for_next_period')

        return ruc_plan(ruc_instance_to_simulate_next_period, scenario_tree_for_next_period, scenario_instances_for_next_period, deterministic_ruc_instance_for_next_period)

    def call_operation_oracle(self, options, time_step):
        '''Calls the oracle for the here-and-now plan based on actual data'''
        current_sced_instance = self.data_manager.current_sced_instance

        # for pretty-printing purposes, compute the maximum bus and generator label lengths.
        max_bus_label_length = max((len(this_bus) for this_bus in current_sced_instance.Buses))

        # we often want to write SCED instances, for diagnostics.
        lp_writer = ProblemWriter_cpxlp()

        if len(current_sced_instance.TransmissionLines) == 0:
            max_line_label_length = None
        else:
            max_line_label_length = max((len(this_line) for this_line in current_sced_instance.TransmissionLines))

        if len(current_sced_instance.ThermalGenerators) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max(
                (len(this_generator) for this_generator in current_sced_instance.ThermalGenerators))

        if len(current_sced_instance.AllNondispatchableGenerators) == 0:
            max_nondispatchable_generator_label_length = None
        else:
            max_nondispatchable_generator_label_length = max(
                (len(this_generator) for this_generator in current_sced_instance.AllNondispatchableGenerators))

        if options.write_sced_instances:
            current_sced_filename = options.output_directory + os.sep + str(time_step.date) + \
                                    os.sep + "sced_hour_" + str(time_step.hour) + ".lp"
            lp_writer(current_sced_instance, current_sced_filename, lambda x: True,
                      {"symbolic_solver_labels": True})
            print("SCED instance written to file=" + current_sced_filename)

        if options.output_sced_initial_conditions:
            print("")
            self.output_sced_initial_condition(current_sced_instance,
                                               max_thermal_generator_label_length=max_thermal_generator_label_length)

        if options.output_sced_demands:
            print("")
            self.output_sced_demand(current_sced_instance,
                                    max_bus_label_length=max_bus_label_length)

        print("")
        print("Solving SCED instance")
        infeasibilities_detected_and_corrected = False

        if options.output_solver_logs:
            print("")
            print("------------------------------------------------------------------------------")

        sced_results = self.call_solver(self._sced_solver,
                                        current_sced_instance,
                                        tee=options.output_solver_logs,
                                        keepfiles=options.keep_solver_files,
                                        **self._solve_options[self._sced_solver])

        current_sced_instance.solutions.load_from(sced_results)

        pre_quickstart_cache = None

        if options.enable_quick_start_generator_commitment:
            # Determine whether we are going to run a quickstart optimization
            if True or StatsExtractor.has_load_shedding(current_sced_instance):
                # Yep, we're doing it.  Cache data we can use to compare results with and without quickstart
                pre_quickstart_cache = StatsExtractor.get_pre_quickstart_data(current_sced_instance)

                # TODO: report solution/load shedding before unfixing Quick Start Generators
                # print("")
                # print("SCED Solution before unfixing Quick Start Generators")
                # print("")
                # self.report_sced_stats()

                # Set up the quickstart run, allowing quickstart generators to turn on
                print("Re-solving SCED after unfixing Quick Start Generators")
                for t in sorted(current_sced_instance.TimePeriods):
                    for g in current_sced_instance.QuickStartGenerators:
                        if current_sced_instance.UnitOn[g, t]==0:
                            sced_instance.UnitOn[g,t].unfix()
                # Run the modified sced
                load_shed_sced_results = self.call_solver(self._sced_solver,
                                                          current_sced_instance,
                                                          tee=options.output_solver_logs,
                                                          keepfiles=options.keep_solver_files,
                                                          **self._solve_options[self._sced_solver])
                current_sced_instance.solutions.load_from(load_shed_sced_results)

        if options.output_solver_logs:
            print("")
            print("------------------------------------------------------------------------------")
            print("")

        if sced_results.solution.status.key != "optimal":
            print("SCED RESULTS STATUS=", sced_results.solution.status.key)
            print("")
            print("Failed to solve SCED optimization instance - no feasible solution exists!")
            print("SCED RESULTS:", sced_results)

            # for diagnostic purposes, save the failed SCED instance.
            infeasible_sced_filename = options.output_directory + os.sep + str(time_step.date) + os.sep + \
                                       "failed_sced_hour_" + str(time_step.hour) + ".lp"
            lp_writer(current_sced_instance, infeasible_sced_filename, lambda x: True,
                      {"symbolic_solver_labels": True})
            print("Infeasible SCED instance written to file=" + infeasible_sced_filename)

        lmp_sced = self.load_lmp_results(options)


        ops_stats = self.simulator().stats_manager.collect_operations(current_sced_instance,
                                                                      sced_results.solver.time,
                                                                      lmp_sced,
                                                                      pre_quickstart_cache)

        self.report_sced_stats(ops_stats)



    def load_lmp_results(self, options):
        print("Fixing binaries and solving for LMPs")
        lmp_sced_instance = self.data_manager.current_sced_instance.clone()

        # In case of demand shortfall, the price skyrockets, so we threshold the value.
        if value(lmp_sced_instance.LoadMismatchPenalty) > options.price_threshold:
            lmp_sced_instance.LoadMismatchPenalty = options.price_threshold

        # In case of reserve shortfall, the price skyrockets, so we threshold the value.
        if value(lmp_sced_instance.ReserveShortfallPenalty) > options.reserve_price_threshold:
            lmp_sced_instance.ReserveShortfallPenalty = options.reserve_price_threshold

        self.data_manager.reference_model_module.fix_binary_variables(lmp_sced_instance)

        self.data_manager.reference_model_module.define_suffixes(lmp_sced_instance)

        lmp_sced_results = self.call_solver(self._sced_solver, lmp_sced_instance,
                                            tee=options.output_solver_logs,
                                            keepfiles=options.keep_solver_files,
                                            **self._solve_options[self._sced_solver])

        if lmp_sced_results.solution.status.key != "optimal":
            raise RuntimeError("Failed to solve LMP SCED")

        lmp_sced_instance.solutions.load_from(lmp_sced_results)

        return lmp_sced_instance

    def initialize_operation_oracle(self, options, time_step, is_first_time_step):
        # if this is the first hour of the day, we might (often) want to establish initial conditions from
        # something other than the prior sced instance. these reasons are not for purposes of realism, but
        # rather pragmatism. for example, there may be discontinuities in the "projected" initial condition
        # at this time (based on the stochastic RUC schedule being executed during the day) and that of the
        # actual sced instance.
        if is_first_time_step:
            # if this is the first date to simulate, we don't have anything
            # else to go off of when it comes to initial conditions - use those
            # found in the RUC.
            initialize_from_ruc = True
        else:
            # if we're not in the first day, we should always simulate from the
            # state of the prior SCED.
            initialize_from_ruc = False

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

        if options.run_deterministic_ruc:
            current_sced_instance = self.create_sced_instance(
                self.data_manager.sced_model,
                self.data_manager.reference_model_module,
                self.data_manager.deterministic_ruc_instance_for_this_period,
                None,
                self.data_manager.scenario_tree_for_this_period,
                self.data_manager.deterministic_ruc_instance_for_next_period,
                None,
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
               )

        else:
            current_sced_instance = self.create_sced_instance(
                self.data_manager.sced_model,
                self.data_manager.reference_model_module(),
                None,
                self.data_manager.scenario_instances_for_this_period,
                self.data_manager.scenario_tree_for_this_period,
                None,
                self.data_manager.scenario_instances_for_next_period,
                self.data_manager.scenario_tree_for_next_period,
                self.data_manager.ruc_instance_to_simulate_this_period,
                self.data_manager.prior_sced_instance,
                self.data_manager.actual_demand,
                None,
                self.data_manager.actual_min_renewables,
                self.data_manager.actual_max_renewables,
                None,
                time_step.hour % options.ruc_every_hours,
                options.reserve_factor,
                options,
                sced_horizon=options.sced_horizon,
                ruc_every_hours=options.ruc_every_hours,
                initialize_from_ruc=initialize_from_ruc,
                use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
               )

        return current_sced_instance

    def available_quick_start_for_deterministic_sced(self, instance):
        """Given a SCED instance with commitments from the RUC,
        determine how much quick start capacity is available
        """
        available_quick_start_capacity = 0.0
        for g in instance.QuickStartGenerators:
            available = True  # until proven otherwise
            if int(round(value(instance.UnitOn[g, 1]))) == 1:
                available = False  # unit was already committed in the RUC
            elif instance.MinimumDownTime[g] > 1:
                # minimum downtime should be 1 or less, by definition of a quick start
                available = False
            elif (value(instance.UnitOnT0[g]) - int(round(value(instance.UnitOn[g, 1])))) == 1:
                # there cannot have been a a shutdown in the previous hour
                available = False

            if available:  # add the amount of power that can be accessed in the first hour
                # use the min() because scaled startup ramps can be larger than the generator limit
                available_quick_start_capacity += min(value(instance.ScaledStartupRampLimit[g]),
                                                      value(instance.MaximumPowerOutput[g]))

        return available_quick_start_capacity

    def report_prices_for_deterministic_ruc(day_ahead_prices, day_ahead_reserve_prices, instance, options,
                                            max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

        pricing_type = options.day_ahead_pricing
        print("")
        print(("%-" + str(max_bus_label_length) + "s %5s %14s") % ("Bus", "Time", "Computed " + pricing_type))
        for t in range(0, options.ruc_every_hours):
            for bus in instance.Buses:
                print(("%-" + str(max_bus_label_length) + "s %5d %14.6f") % (bus, t, day_ahead_prices[bus, t]))

        print("")
        print(("Reserves %5s %14s") % ("Time", "Computed " + pricing_type + " reserve price"))
        for t in range(0, options.ruc_every_hours):
            print(("         %5d %14.6f") % (t, day_ahead_reserve_prices[t]))

    #####################################################################
    # utility functions for pretty-printing solutions, for SCED and RUC #
    #####################################################################

    def output_sced_initial_condition(self, sced_instance, hour=1,
                                      max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

        print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated t1-unit-on must-run):")
        for g in sorted(sced_instance.ThermalGenerators):
            if hour == 1:
                print(("%-" + str(max_thermal_generator_label_length) + "s %5d %12.2f %5d %6d") %
                      (g,
                       value(sced_instance.UnitOnT0[g]),
                       value(sced_instance.PowerGeneratedT0[g]),
                       value(sced_instance.UnitOn[g, hour]),
                       value(sced_instance.MustRun[g])))
            else:
                print(("%-" + str(max_thermal_generator_label_length) + "s %5d %12.2f %5d %6d") %
                      (g,
                       value(sced_instance.UnitOn[g, hour - 1]),
                       value(sced_instance.PowerGenerated[g, hour - 1]),
                       value(sced_instance.UnitOn[g, hour]),
                       value(sced_instance.MustRun[g])))

    def output_sced_demand(self, sced_instance, max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

        print("Demand detail:")
        for b in sorted(sced_instance.Buses):
            print(("%-" + str(max_bus_label_length) + "s %12.2f") %
                  (b,
                   value(sced_instance.Demand[b, 1])))

        print("")
        print(("%-" + str(max_bus_label_length) + "s %12.2f") %
              ("Reserve requirement:",
               value(sced_instance.ReserveRequirement[1])))

        print("")
        print("Maximum non-dispatachable power available:")
        for b in sorted(sced_instance.Buses):
            total_max_nondispatchable_power = sum(value(sced_instance.MaxNondispatchablePower[g, 1])
                                                  for g in sced_instance.NondispatchableGeneratorsAtBus[b])
            print("%-30s %12.2f" % (b, total_max_nondispatchable_power))

        print("")
        print("Minimum non-dispatachable power available:")
        for b in sorted(sced_instance.Buses):
            total_min_nondispatchable_power = sum(value(sced_instance.MinNondispatchablePower[g, 1])
                                                  for g in sced_instance.NondispatchableGeneratorsAtBus[b])
            print("%-30s %12.2f" % (b, total_min_nondispatchable_power))

    def output_sced_solution(self, sced_instance, max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

        print("Solution detail:")
        print("")
        print("Dispatch Levels (unit-on, power-generated, reserve-headroom)")
        for g in sorted(sced_instance.ThermalGenerators):
            unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
            print(("%-" + str(max_thermal_generator_label_length) + "s %2d %12.2f %12.2f") %
                  (g,
                   unit_on,
                   value(sced_instance.PowerGenerated[g, 1]),
                   math.fabs(value(sced_instance.MaximumPowerAvailable[g, 1]) -
                             value(sced_instance.PowerGenerated[g, 1]))),
                  end=' ')
            if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                             value(sced_instance.MaximumPowerOutput[g])) <= 1e-5):
                print(" << At max output", end=' ')
            elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                               value(sced_instance.MinimumPowerOutput[g])) <= 1e-5):
                print(" << At min output", end=' ')
            if value(sced_instance.MustRun[g]):
                print(" ***", end=' ')
            print("")

        print("")
        print("Total power dispatched      = %12.2f"
              % sum(value(sced_instance.PowerGenerated[g, 1]) for g in sced_instance.ThermalGenerators))
        print("Total reserve available     = %12.2f"
              % sum(value(sced_instance.MaximumPowerAvailable[g, 1]) - value(sced_instance.PowerGenerated[g, 1])
                    for g in sced_instance.ThermalGenerators))
        print("Total quick start available = %12.2f"
              % self.available_quick_start_for_deterministic_sced(sced_instance))
        print("")

        print("Cost Summary (unit-on production-cost no-load-cost startup-cost)")
        total_startup_costs = 0.0
        for g in sorted(sced_instance.ThermalGenerators):
            unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
            unit_on_t0 = int(round(value(sced_instance.UnitOnT0[g])))
            startup_cost = 0.0
            if unit_on_t0 == 0 and unit_on == 1:
                startup_cost = value(sced_instance.StartupCost[g, 1])
            total_startup_costs += startup_cost
            print(("%-" + str(max_thermal_generator_label_length) + "s %2d %12.2f %12.2f %12.2f") %
                  (g,
                   unit_on,
                   value(sced_instance.ProductionCost[g, 1]),
                   unit_on * value(sced_instance.MinimumProductionCost[g]),
                   startup_cost),
                  end=' ')
            if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                             value(sced_instance.MaximumPowerOutput[g])) <= 1e-5):
                print(" << At max output", end=' ')
            elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                               value(sced_instance.MinimumPowerOutput[
                                                         g])) <= 1e-5):  # TBD - still need a tolerance parameter
                print(" << At min output", end=' ')
            print("")

        print("")
        print("Total cost = %12.2f" % (
                value(sced_instance.TotalNoLoadCost[1]) + value(sced_instance.TotalProductionCost[1]) +
                total_startup_costs))

    # useful in cases where ramp rate constraints have been violated.
    # ramp rates are taken from the original sced instance. unit on
    # values can be taken from either instance, as they should be the
    # same. power generated values must be taken from the relaxed instance.

    def output_sced_ramp_violations(self, original_sced_instance, relaxed_sced_instance):

        # we are assuming that there are only a handful of violations - if that
        # is not the case, we should shift to some kind of table output format.
        for g in original_sced_instance.ThermalGenerators:
            for t in original_sced_instance.TimePeriods:

                current_unit_on = int(round(value(original_sced_instance.UnitOn[g, t])))

                if t == 1:
                    previous_unit_on = int(round(value(original_sced_instance.UnitOnT0[g])))
                else:
                    previous_unit_on = int(round(value(original_sced_instance.UnitOn[g, t - 1])))

                if current_unit_on == 0:
                    if previous_unit_on == 0:
                        # nothing is on, nothing to worry about!
                        pass
                    else:
                        # the unit is switching off.
                        # TBD - eventually deal with shutdown ramp limits
                        pass

                else:
                    if previous_unit_on == 0:
                        # the unit is switching on.
                        # TBD - eventually deal with startup ramp limits
                        pass
                    else:
                        # the unit is remaining on.
                        if t == 1:
                            delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(
                                relaxed_sced_instance.PowerGeneratedT0[g])
                        else:
                            delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(
                                relaxed_sced_instance.PowerGenerated[g, t - 1])
                        if delta_power > 0.0:
                            # the unit is ramping up
                            if delta_power > value(original_sced_instance.NominalRampUpLimit[g]):
                                print(
                                        "Thermal unit=%s violated nominal ramp up limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                        % (g, t - 1, t, delta_power,
                                           value(original_sced_instance.NominalRampUpLimit[g])))
                        else:
                            # the unit is ramping down
                            if math.fabs(delta_power) > value(original_sced_instance.NominalRampDownLimit[g]):
                                print(
                                        "Thermal unit=%s violated nominal ramp down limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                        % (g, t - 1, t, math.fabs(delta_power),
                                           value(original_sced_instance.NominalRampDownLimit[g])))


    def report_sced_stats(self, ops_stats: OperationsStats):
        print("Fixed costs:    %12.2f" % ops_stats.fixed_costs)
        print("Variable costs: %12.2f" % ops_stats.variable_costs)
        print("")

        if ops_stats.load_shedding != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f" % (1, ops_stats.load_shedding))
        if ops_stats.over_generation!= 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f" % (1, ops_stats.over_generation))

        if ops_stats.reserve_shortfall != 0.0:
            print("Reserve shortfall reported at t=%2d: %12.2f" % (1, ops_stats.reserve_shortfall))
            print("Quick start generation capacity available at t=%2d: %12.2f" % (1, ops_stats.available_quick_start))
            print("")

        if ops_stats.renewables_curtailment > 0:
            print("Renewables curtailment reported at t=%d - total=%12.2f" % (1, ops_stats.renewables_curtailment))
            print("")

        print("Number on/offs:       %12d" % ops_stats.on_offs)
        print("Sum on/off ramps:     %12.2f" % ops_stats.sum_on_off_ramps)
        print("Sum nominal ramps:    %12.2f" % ops_stats.sum_nominal_ramps)
        print("")

    def _setup_plugin_methods(self, options):
        ''' Set method instances to either the default, or those provided by a plugin identified in the options '''

        # Setup default methods from default plugin; they'll be overwritten by 
        # methods from another plugin if one was specified
        from prescient.plugins import default_plugin
        self.call_solver = default_plugin.call_solver
        self.create_sced_instance = default_plugin.create_sced_instance
        self.create_and_solve_deterministic_ruc = default_plugin.create_and_solve_deterministic_ruc
        self.create_and_solve_stochastic_ruc_via_ef = default_plugin.create_and_solve_stochastic_ruc_via_ef
        self.create_and_solve_stochastic_ruc_via_ph = default_plugin.create_and_solve_stochastic_ruc_via_ph
        self.compute_simulation_filename_for_date = default_plugin.compute_simulation_filename_for_date
        self.create_ruc_instance_to_simulate_next_period = default_plugin.create_ruc_instance_to_simulate_next_period

        if options.simulator_plugin != None:
            try:
                simulator_plugin_module = pyutilib.misc.import_file(options.simulator_plugin)
            except:
                raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

            method_names = ["call_solver",
                            "create_sced_instance",
                            "create_and_solve_deterministic_ruc",
                            "create_and_solve_stochastic_ruc_via_ef",
                            "create_and_solve_stochastic_ruc_via_ph",
                            "compute_simulation_filename_for_date",
                            "create_ruc_instance_to_simulate_next_period"]

            for method_name in method_names:
                method = getattr(simulator_plugin_module, method_name, None)
                if method is None:
                    print(f"***WARNING***: Could not find function '{method_name}' in simulator plugin module={options.simulator_plugin}, using default!")
                else:
                    print(f"Loaded simulator plugin function '{method_name}' from simulator plugin module={options.simulator_plugin}")
                    setattr(self, method_name, method)


        elif options.deterministic_ruc_solver_plugin != None:
            try:
                solver_plugin_module = pyutilib.misc.import_file(options.deterministic_ruc_solver_plugin)
            except:
                raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

            solve_function = getattr(solver_plugin_module, "solve_deterministic_ruc", None)
            if solve_function is None:
                raise RuntimeError("Could not find function 'solve_deterministic_ruc' in simulator plugin module=%s, using default!" % options.deterministic_ruc_solver_plugin)
            else:
                print("Loaded deterministic ruc solver plugin function 'solve_deterministic_ruc' from simulator plugin module=%s."% options.deterministic_ruc_solver_plugin)
                from prescient.plugins.default_plugin import create_create_and_solve_deterministic_ruc
                self.create_and_solve_deterministic_ruc = create_create_and_solve_deterministic_ruc(solve_function)
