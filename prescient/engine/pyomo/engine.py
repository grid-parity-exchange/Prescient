#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.simulator import Options
    from prescient.engine.abstract_types import *

import os

from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
from pyomo.core import value

from prescient.engine.modeling_engine import ModelingEngine

from .data_extractors import ScedDataExtractor, RucDataExtractor

## termination conditions which are acceptable
safe_termination_conditions = [
                               TerminationCondition.maxTimeLimit,
                               TerminationCondition.maxIterations,
                               TerminationCondition.minFunctionValue,
                               TerminationCondition.minStepLength,
                               TerminationCondition.globallyOptimal,
                               TerminationCondition.locallyOptimal,
                               TerminationCondition.feasible,
                               TerminationCondition.optimal,
                               TerminationCondition.maxEvaluations,
                               TerminationCondition.other,
                              ]

class PyomoEngine(ModelingEngine):

    def initialize(self, options:Options) -> None:
        self._sced_extractor = ScedDataExtractor()
        self._ruc_extractor = RucDataExtractor()
        self._setup_reference_models(options)
        self._setup_solvers(options)
        self._p = PyomoEngine._PluginMethods(options)

    def create_and_solve_deterministic_ruc(self,
            options: Options,
            uc_date:str,
            uc_hour: int,
            next_uc_date: Optional[str],
            prior_ruc_instance: RucModel,
            prior_scenario_tree: ScenarioTree,
            output_ruc_initial_conditions: bool,
            projected_sced_instance: OperationsModel,
            sced_schedule_hour: int,
            ruc_horizon: int,
            run_ruc_with_next_day_data: bool
           ) -> Tuple[RucModel, ScenarioTree]:
        return self._p.create_and_solve_deterministic_ruc(self._ruc_solver, self._solve_options,
                                                    options, uc_date, uc_hour, next_uc_date,
                                                    prior_ruc_instance, prior_scenario_tree,
                                                    output_ruc_initial_conditions,
                                                    projected_sced_instance, 
                                                    sced_schedule_hour, ruc_horizon,
                                                    run_ruc_with_next_day_data)


    def create_ruc_instance_to_simulate_next_period(
            self,
            options: Options,
            uc_date: str,
            uc_hour: int,
            next_uc_date: Optional[str]
           ) -> RucModel:
        return self._p.create_ruc_instance_to_simulate_next_period(self._ruc_model, options, uc_date, uc_hour, next_uc_date)


    def create_and_solve_sced_instance(self,
            deterministic_ruc_instance_for_this_period: RucModel,
            scenario_tree_for_this_period: ScenarioTree,
            deterministic_ruc_instance_for_next_period: RucModel,
            scenario_tree_for_next_period: ScenarioTree,
            ruc_instance_to_simulate_this_period: RucModel,
            prior_sced_instance: OperationsModel,
            actual_demand: Mapping[Tuple[Bus, int], float],
            demand_forecast_error: Mapping[Tuple[Bus, int], float],
            actual_min_renewables: Mapping[Tuple[Generator, int], float],
            actual_max_renewables: Mapping[Tuple[Generator, int], float],
            renewables_forecast_error: Mapping[Tuple[Generator, int], float],
            hour_to_simulate: int,
            reserve_factor: float,
            options: Options,
            hours_in_objective: int=1,
            sced_horizon: int=24,
            ruc_every_hours: int=24,
            initialize_from_ruc: bool=True,
            use_prescient_forecast_error: bool=True,
            use_persistent_forecast_error: bool=False,
            write_sced_instance: bool = False,
            lp_filename: str = None,
            output_initial_conditions: bool = False,
            output_demands: bool = False
           ) -> Tuple[OperationsModel, float]:

        current_sced_instance = self._p.create_sced_instance(
            self._sced_model, self._reference_model_module, 
            deterministic_ruc_instance_for_this_period, None, scenario_tree_for_this_period, 
            deterministic_ruc_instance_for_next_period, None, scenario_tree_for_next_period,
            ruc_instance_to_simulate_this_period,
            prior_sced_instance,
            actual_demand,
            demand_forecast_error,
            actual_min_renewables,
            actual_max_renewables,
            renewables_forecast_error,
            hour_to_simulate,
            reserve_factor,
            options,
            hours_in_objective,
            sced_horizon,
            ruc_every_hours,
            initialize_from_ruc,
            use_prescient_forecast_error,
            use_persistent_forecast_error)

        if write_sced_instance:
            lp_writer = ProblemWriter_cpxlp()
            lp_writer(current_sced_instance, lp_filename, lambda x: True,
                      {"symbolic_solver_labels": True})
            print("SCED instance written to file=" + lp_filename)

        self._print_sced_info(current_sced_instance, output_initial_conditions, output_demands)

        if options.output_solver_logs:
            print("")
            print("------------------------------------------------------------------------------")

        sced_results = self._p.call_solver(self._sced_solver,
                                     current_sced_instance,
                                     tee=options.output_solver_logs,
                                     keepfiles=options.keep_solver_files,
                                     **self._solve_options[self._sced_solver])

        if sced_results.solver.termination_condition not in safe_termination_conditions:
            print("SCED RESULTS STATUS=", sced_results.solver.termination_condition)
            print("")
            print("Failed to solve SCED optimization instance - no feasible solution exists!")
            print("SCED RESULTS:", sced_results)

            # for diagnostic purposes, save the failed SCED instance.
            if lp_filename is not None:
                if lp_filename.endswith(".lp"):
                    infeasible_sced_filename = lp_filename[:-3] + ".FAILED.lp"
                else:
                    infeasible_sced_filename = lp_filename + ".FAILED"
            else:
                infeasible_sced_filename = options.output_directory + os.sep + "FAILED_SCED.lp"
            lp_writer = ProblemWriter_cpxlp()
            lp_writer(current_sced_instance, infeasible_sced_filename, lambda x: True,
                      {"symbolic_solver_labels": True})
            print("Infeasible SCED instance written to file=" + infeasible_sced_filename)
            return None


        current_sced_instance.solutions.load_from(sced_results)
        return current_sced_instance, sced_results.solver.time


    def enable_quickstart_and_solve(self,
            sced_instance: OperationsModel,
            options: Options
           ) -> OperationsModel:
        # Set up the quickstart run, allowing quickstart generators to turn on
        for t in sorted(sced_instance.TimePeriods):
            for g in sced_instance.QuickStartGenerators:
                if sced_instance.UnitOn[g, t]==0:
                    sced_instance.UnitOn[g,t].unfix()
        # Run the modified sced
        load_shed_sced_results = self._p.call_solver(self._sced_solver,
                                               sced_instance,
                                               tee=options.output_solver_logs,
                                               keepfiles=options.keep_solver_files,
                                               **self._solve_options[self._sced_solver])
        sced_instance.solutions.load_from(load_shed_sced_results)
        return sced_instance

    def create_and_solve_lmp(self,
            sced_instance: OperationsModel,
            options:Options,
           ) -> OperationsModel:

        lmp_sced_instance = sced_instance.clone()

        # In case of demand shortfall, the price skyrockets, so we threshold the value.
        if value(lmp_sced_instance.LoadMismatchPenalty) > options.price_threshold:
            lmp_sced_instance.LoadMismatchPenalty = options.price_threshold

        # In case of reserve shortfall, the price skyrockets, so we threshold the value.
        if value(lmp_sced_instance.ReserveShortfallPenalty) > options.reserve_price_threshold:
            lmp_sced_instance.ReserveShortfallPenalty = options.reserve_price_threshold

        self._reference_model_module.fix_binary_variables(lmp_sced_instance)

        self._reference_model_module.define_suffixes(lmp_sced_instance)

        lmp_sced_results = self._p.call_solver(self._sced_solver, lmp_sced_instance,
                                         tee=options.output_solver_logs,
                                         keepfiles=options.keep_solver_files,
                                         **self._solve_options[self._sced_solver])

        if lmp_sced_results.solver.termination_condition not in safe_termination_conditions:
            raise RuntimeError("Failed to solve LMP SCED")

        lmp_sced_instance.solutions.load_from(lmp_sced_results)

        return lmp_sced_instance


    def _print_sced_info(self,
                         sced_instance: OperationsSced,
                         output_initial_conditions: bool,
                         output_demands: bool):
        if not output_initial_conditions and not output_demands:
            return

        # for pretty-printing purposes, compute the maximum bus and generator label lengths.
        max_bus_label_length = max((len(this_bus) for this_bus in sced_instance.Buses))

        if len(sced_instance.TransmissionLines) == 0:
            max_line_label_length = None
        else:
            max_line_label_length = max((len(this_line) for this_line in sced_instance.TransmissionLines))

        if len(sced_instance.ThermalGenerators) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max(
                (len(this_generator) for this_generator in sced_instance.ThermalGenerators))

        if len(sced_instance.AllNondispatchableGenerators) == 0:
            max_nondispatchable_generator_label_length = None
        else:
            max_nondispatchable_generator_label_length = max(
                (len(this_generator) for this_generator in sced_instance.AllNondispatchableGenerators))

        if output_initial_conditions:
            print("Initial condition detail (gen-name t0-unit-on t0-power-generated t1-unit-on must-run):")
            for g in sorted(sced_instance.ThermalGenerators):
                print(("%-" + str(max_thermal_generator_label_length) + "s %5d %12.2f %5d %6d") %
                        (g,
                        value(sced_instance.UnitOnT0[g]),
                        value(sced_instance.PowerGeneratedT0[g]),
                        value(sced_instance.UnitOn[g, 1]),
                        value(sced_instance.MustRun[g])))

        if output_demands:
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

    def _setup_reference_models(self, options: Options):
        model_filename = os.path.join(options.model_directory, "ReferenceModel.py")
        if not os.path.exists(model_filename):
            raise RuntimeError("The model %s either does not exist or cannot be read" % model_filename)
        
        from pyutilib.misc import import_file
        self._reference_model_module = import_file(model_filename)

        # validate reference model
        required_methods = ["fix_binary_variables", "free_binary_variables", "status_var_generator", "define_suffixes", "load_model_parameters"]
        for method in required_methods:
            if not hasattr(self._reference_model_module, method):
                raise RuntimeError("Reference model module does not have required method=%s" % method)

        self._ruc_model = self._reference_model_module.load_model_parameters()
        self._sced_model = self._reference_model_module.model


    def _setup_solvers(self, options: Options):
        if options.python_io:
            self._ruc_solver = SolverFactory(options.deterministic_ruc_solver_type, solver_io="python")
            self._sced_solver = SolverFactory(options.sced_solver_type, solver_io="python")
        else:
            self._ruc_solver = SolverFactory(options.deterministic_ruc_solver_type)
            self._sced_solver = SolverFactory(options.sced_solver_type)

        solve_options = {}
        solve_options[self._sced_solver] = {}
        solve_options[self._ruc_solver] = {}

        for s in solve_options:
            solve_options[s]['symbolic_solver_labels'] = options.symbolic_solver_labels

        ##TODO: work out why we get an error from ProblemWriter_cpxlp when this is False
        if options.warmstart_ruc:
            solve_options[self._ruc_solver]['warmstart'] = options.warmstart_ruc

        if options.deterministic_ruc_solver_type == "cplex" or options.deterministic_ruc_solver_type == "cplex_persistent":
            self._ruc_solver.options.mip_tolerances_mipgap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "gurobi" or options.deterministic_ruc_solver_type == "gurobi_persistent":
            self._ruc_solver.options.MIPGap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "cbc":
            self._ruc_solver.options.ratioGap = options.ruc_mipgap
        elif options.deterministic_ruc_solver_type == "glpk":
            self._ruc_solver.options.mipgap = options.ruc_mipgap
        else:
            raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)

        self._ruc_solver.set_options("".join(options.deterministic_ruc_solver_options))
        self._sced_solver.set_options("".join(options.sced_solver_options))

        self._solve_options = solve_options

    @property
    def ruc_data_extractor(self) -> RucDataExtractor:
        ''' An object that extracts statistics from a RUC model '''
        return self._ruc_extractor

    @property
    def operations_data_extractor(self) -> ScedDataExtractor:
        ''' An object that extracts statistics from a solved operations model '''
        return self._sced_extractor


    class _PluginMethods():
        def __init__(self, options: Options):
            ''' Set method instances to either the default, or those provided by a plugin identified in the options '''

            # Setup default methods from default plugin; they'll be overwritten by 
            # methods from another plugin if one was specified
            from prescient.plugins import default_plugin
            self.call_solver = default_plugin.call_solver
            self.create_sced_instance = default_plugin.create_sced_instance
            self.create_and_solve_deterministic_ruc = default_plugin.create_and_solve_deterministic_ruc
            self.create_ruc_instance_to_simulate_next_period = default_plugin.create_ruc_instance_to_simulate_next_period

            if options.simulator_plugin != None:
                try:
                    simulator_plugin_module = pyutilib.misc.import_file(options.simulator_plugin)
                except:
                    raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

                method_names = ["call_solver",
                                "create_sced_instance",
                                "create_and_solve_deterministic_ruc",
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

