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
    import datetime
    from prescient.simulator import Options
    from prescient.engine.abstract_types import *
    from prescient.data.simulation_state import SimulationState

import os
import pyomo.environ as pe

from prescient.engine.modeling_engine import ModelingEngine, ForecastErrorMethod
from prescient.simulator.data_manager import RucMarket
from prescient.data.providers.dat_data_provider import DatDataProvider

from .data_extractors import ScedDataExtractor, RucDataExtractor
from .ptdf_manager import PTDFManager
from .reporting import report_initial_conditions_for_deterministic_ruc, \
                       report_demand_for_deterministic_ruc

from egret.models.unit_commitment import _get_uc_model, create_tight_unit_commitment_model
from egret.common.lazy_ptdf_utils import uc_instance_binary_relaxer

def create_sced_uc_model(model_data,
                         network_constraints='ptdf_power_flow',
                         relaxed=False,
                         **kwargs):
    '''
    Create a model appropriate for the SCED and pricing
    '''
    formulation_list = [
                        'garver_3bin_vars',
                        'garver_power_vars',
                        'MLR_reserve_vars',
                        'MLR_generation_limits',
                        'damcikurt_ramping',
                        'CA_production_costs',
                        'rajan_takriti_UT_DT',
                        'MLR_startup_costs',
                         network_constraints,
                       ]
    return _get_uc_model(model_data, formulation_list, relaxed, **kwargs)

class EgretEngine(ModelingEngine):

    def initialize(self, options:Options) -> None:
        self._sced_extractor = ScedDataExtractor()
        self._ruc_extractor = RucDataExtractor()
        self._setup_solvers(options)
        self._p = EgretEngine._PluginMethods(options)
        self._ptdf_manager = PTDFManager()
        self._last_sced_pyo_model = None
        self._last_sced_pyo_solver = None
        self._data_provider = DatDataProvider()
        self._data_provider.initialize(options)
        self._actuals_step_frequency = 60 if not options.simulate_out_of_sample \
                                       else self._data_provider.negotiate_data_frequency(options.sced_frequency_minutes)

    def create_deterministic_ruc(self, 
            options: Options,
            uc_date:datetime.date,
            uc_hour: int,
            current_state: SimulationState,
            output_ruc_initial_conditions: bool,
            ruc_horizon: int,
            use_next_day_data: bool
           ) -> RucModel:
        ruc = self._p.create_deterministic_ruc(options, self._data_provider, 
                                               uc_date, uc_hour,
                                               current_state,
                                               ruc_horizon,
                                               use_next_day_data)
        if output_ruc_initial_conditions:
            report_initial_conditions_for_deterministic_ruc(ruc)
            report_demand_for_deterministic_ruc(ruc, options.ruc_every_hours)

        return ruc


    def solve_deterministic_ruc(self, options, ruc_instance, uc_date, uc_hour):
        return self._p.solve_deterministic_ruc(self._ruc_solver, options, ruc_instance, uc_date, uc_hour, self._ptdf_manager)

    def create_and_solve_day_ahead_pricing(self,
            options: Options,
            deterministic_ruc_instance: RucModel,
            ) -> RucMarket:
        return self._p.solve_deterministic_day_ahead_pricing_problem(self._ruc_solver,
                                                                deterministic_ruc_instance,
                                                                options,
                                                                self._ptdf_manager)

    def create_simulation_actuals(
            self,
            options: Options,
            uc_date: datetime.date,
            uc_hour: int
           ) -> RucModel:
        return self._p.create_simulation_actuals(options, self._data_provider, 
                                                 uc_date, uc_hour,
                                                 self._actuals_step_frequency)


    def create_sced_instance(self,
            options: Options,
            current_state: SimulationState,
            hours_in_objective: int,
            sced_horizon: int,
            forecast_error_method = ForecastErrorMethod.PRESCIENT,
            write_sced_instance: bool = False,
            lp_filename: str = None
           ) -> OperationsModel:

        current_sced_instance = self._p.create_sced_instance(
            self._data_provider,
            current_state,
            options,
            sced_horizon,
            forecast_error_method)

        if write_sced_instance:
            current_sced_instance.write(lp_filename)
            print("SCED instance written to file=" + lp_filename)

        self._hours_in_objective = hours_in_objective

        return current_sced_instance

    def solve_sced_instance(self,
                            options,
                            sced_instance,
                            output_initial_conditions = False,
                            output_demands = False,
                            lp_filename: str = None):

        ptdf_manager = self._ptdf_manager
        if self._hours_in_objective > 10:
            ptdf_options = ptdf_manager.look_ahead_sced_ptdf_options
        else:
            ptdf_options = ptdf_manager.sced_ptdf_options

        ptdf_manager.mark_active(sced_instance)
        pyo_model = create_sced_uc_model(sced_instance,
                                         ptdf_options = ptdf_options,
                                         PTDF_matrix_dict=ptdf_manager.PTDF_matrix_dict)

        # update in case lines were taken out
        ptdf_manager.PTDF_matrix_dict = pyo_model._PTDFs

        self._p._zero_out_costs(pyo_model, self._hours_in_objective)

        self._print_sced_info(sced_instance, output_initial_conditions, output_demands)
        if options.output_solver_logs:
            print("")
            print("------------------------------------------------------------------------------")

        try:
            sced_results, sced_time, pyo_solver = self._p.call_solver(self._sced_solver,
                                                                      pyo_model,
                                                                      options,
                                                                      options.sced_solver_options)
        except:
            print("Some isssue with SCED, writing instance")
            print("Problematic SCED from to file")
            # for diagnostic purposes, save the failed SCED instance.
            if lp_filename is not None:
                if lp_filename.endswith(".json"):
                    infeasible_sced_filename = lp_filename[:-5] + ".FAILED.json"
                else:
                    infeasible_sced_filename = lp_filename + ".FAILED.json"
            else:
                infeasible_sced_filename = options.output_directory + os.sep + "FAILED_SCED.json"
            sced_instance.write(infeasible_sced_filename)
            print("Problematic SCED instance written to file=" + infeasible_sced_filename)
            raise

        ptdf_manager.update_active(sced_results)
        self._last_sced_pyo_model = pyo_model
        self._last_sced_pyo_solver = pyo_solver

        return sced_results, sced_time


    def enable_quickstart_and_solve(self,
            options: Options,
            sced_instance: OperationsModel
           ) -> OperationsModel:
        # Set up the quickstart run, allowing quickstart generators to turn on
        for g, g_dict in sced_instance.elements(element_type='generator', fast_start=True):
            del g_dict['fixed_commitment']

        self._ptdf_manager.mark_active(sced_instance)
        pyo_model = create_tight_unit_commitment_model(sced_instance,
                                                       ptdf_options = self._ptdf_manager.sced_ptdf_options,
                                                       PTDF_matrix_dict=self._ptdf_manager.PTDF_matrix_dict)

        try:
            sced_results, _ , pyo_solver = self._p.call_solver(self._sced_solver,
                                                               pyo_model,
                                                               options,
                                                               options.sced_solver_options)
        except:
            print("Some issue with quickstart UC, writing instance")
            quickstart_uc_filename = options.output_directory+os.sep+"FAILED_QS.json"
            sced_instance.write(quickstart_uc_filename)
            print(f"Problematic quickstart UC written to {quickstart_uc_filename}")
            raise

        self._ptdf_manager.update_active(sced_results)
        self._last_sced_pyo_model = pyo_model
        self._last_sced_pyo_solver = pyo_solver

        return sced_results

    def create_and_solve_lmp(self,
            options:Options,
            sced_instance: OperationsModel,
           ) -> OperationsModel:

        lmp_sced_instance = sced_instance.clone()

        # In case of demand shortfall, the price skyrockets, so we threshold the value.
        if 'load_mismatch_cost' not in lmp_sced_instance.data['system'] or \
                lmp_sced_instance.data['system']['load_mismatch_cost'] > \
                    options.price_threshold:
            lmp_sced_instance.data['system']['load_mismatch_cost'] = options.price_threshold

        # In case of reserve shortfall, the price skyrockets, so we threshold the value.
        if 'reserve_shortfall_cost' not in lmp_sced_instance.data['system'] or \
                lmp_sced_instance.data['system']['reserve_shortfall_cost'] > \
                    options.reserve_price_threshold:
            lmp_sced_instance.data['system']['reserve_shortfall_cost'] = \
                    options.reserve_price_threshold

        if self._last_sced_pyo_model is None:
            self._ptdf_manager.mark_active(lmp_sced_instance)
            pyo_model = create_sced_uc_model(lmp_sced_instance, relaxed=True,
                                             ptdf_options = self._ptdf_manager.lmpsced_ptdf_options,
                                             PTDF_matrix_dict=self._ptdf_manager.PTDF_matrix_dict)
            pyo_solver = self._sced_solver
            self._p._zero_out_costs(pyo_model, self._hours_in_objective)
        else:
            pyo_model = self._last_sced_pyo_model
            pyo_solver = self._last_sced_pyo_solver
            self._transform_for_lmp(pyo_model, pyo_solver, lmp_sced_instance)

        pyo_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        try:
            lmp_sced_results, _, _ = self._p.call_solver(pyo_solver,
                                                         pyo_model,
                                                         options,
                                                         options.sced_solver_options,
                                                         relaxed=True,
                                                         set_instance=(self._last_sced_pyo_model is None))
        except:
            print("Some issue with LMP SCED, writing instance")
            quickstart_uc_filename = options.output_directory+os.sep+"FAILED_LMP_SCED.json"
            lmp_sced_instance.write(quickstart_uc_filename)
            print(f"Problematic LMP SCED written to {quickstart_uc_filename}")
            raise

        return lmp_sced_results

    def _transform_for_lmp(self, pyo_model, pyo_solver, lmp_sced_instance):
        from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
        import math

        uc_instance_binary_relaxer(pyo_model, pyo_solver)

        ## reset the penalites
        system = lmp_sced_instance.data['system']

        update_obj = False

        new_load_penalty = system['baseMVA'] * system['load_mismatch_cost']
        if not math.isclose(new_load_penalty, pyo_model.LoadMismatchPenalty.value):
            pyo_model.LoadMismatchPenalty.value = new_load_penalty
            update_obj = True

        new_reserve_penalty =  system['baseMVA'] * system['reserve_shortfall_cost']
        if not math.isclose(new_reserve_penalty, pyo_model.ReserveShortfallPenalty.value):
            pyo_model.ReserveShortfallPenalty.value = new_reserve_penalty
            update_obj = True

        pyo_model.model_data = lmp_sced_instance

        if update_obj and isinstance(pyo_solver, PersistentSolver):
            pyo_solver.set_objective(pyo_model.TotalCostObjective)

    def _print_sced_info(self,
                         sced_instance: OperationsSced,
                         output_initial_conditions: bool,
                         output_demands: bool):
        if not output_initial_conditions and not output_demands:
            return

        sced_data_extractor = self.operations_data_extractor

        # for pretty-printing purposes, compute the maximum bus and generator label lengths.
        buses = list(sced_data_extractor.get_buses(sced_instance))
        max_bus_label_length = max((len(this_bus) for this_bus in buses))

        lines = list(sced_data_extractor.get_transmission_lines(sced_instance))
        if len(lines) == 0:
            max_line_label_length = None
        else:
            max_line_label_length = max((len(this_line) for this_line in lines))

        thermal_gens = list(sced_data_extractor.get_thermal_generators(sced_instance))
        if len(thermal_gens) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max(
                (len(this_generator) for this_generator in thermal_gens))

        nondispatchable_gens = list(sced_data_extractor.get_nondispatchable_generators(sced_instance))
        if len(nondispatchable_gens) == 0:
            max_nondispatchable_generator_label_length = None
        else:
            max_nondispatchable_generator_label_length = max(
                (len(this_generator) for this_generator in nondispatchable_gens))

        if output_initial_conditions:
            print("Initial condition detail (gen-name t0-unit-on t0-power-generated t1-unit-on ):")
            for g in thermal_gens: 
                print(("%-" + str(max_thermal_generator_label_length) + "s %5d %12.2f %5d") %
                        (g,
                        sced_data_extractor.generator_was_on(sced_instance, g),
                        sced_data_extractor.get_power_generated_T0(sced_instance, g),
                        sced_data_extractor.is_generator_on(sced_instance, g),
                        ))

        if output_demands:
            print("Demand detail:")
            for b in buses: 
                print(("%-" + str(max_bus_label_length) + "s %12.2f") %
                      (b,
                       sced_data_extractor.get_bus_demand(sced_instance,b)))

            print("")
            print(("%-" + str(max_bus_label_length) + "s %12.2f") %
                  ("Reserve requirement:",
                    sced_data_extractor.get_reserve_requirement(sced_instance)))

            total_max_nondispatchable_power = {b : 0. for b in buses}
            total_min_nondispatchable_power = {b : 0. for b in buses}
            for g in nondispatchable_gens:
                b = sced_data_extractor.get_generator_bus(sced_instance,g)

                total_max_nondispatchable_power[b] += sced_data_extractor.get_max_nondispatchable_power(sced_instance, g)

                total_min_nondispatchable_power[b] += sced_data_extractor.get_min_nondispatchable_power(sced_instance, g)

            print("")
            print("Maximum non-dispatachable power available:")
            for b in buses: 
                print("%-30s %12.2f" % (b, total_max_nondispatchable_power[b]))

            print("")
            print("Minimum non-dispatachable power available:")
            for b in buses:
                print("%-30s %12.2f" % (b, total_min_nondispatchable_power[b]))

    def _setup_solvers(self, options: Options):
        assert options.python_io is False

        def _get_solver_list(name):
            return [ name+s for s in ['', '_direct', '_persistent']]

        supported_solvers = _get_solver_list('xpress') + \
                            _get_solver_list('gurobi') + \
                            _get_solver_list('cplex') + \
                            ['cbc', 'glpk']

        supported_persistent_solvers = ('xpress', 'gurobi', 'cplex')

        if options.deterministic_ruc_solver_type not in supported_solvers:
            raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)
        if options.sced_solver_type not in supported_solvers:
            raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)

        self._ruc_solver = options.deterministic_ruc_solver_type
        self._sced_solver = options.sced_solver_type

        if self._ruc_solver in supported_persistent_solvers:
            try:
                available = pe.SolverFactory(self._ruc_solver+'_persistent').available()
                if not available:
                    self._print_persistence_warning(self._ruc_solver)
                else:
                    self._ruc_solver = self._ruc_solver+'_persistent'
            except:
                self._print_persistence_warning(self._ruc_solver)

        if not pe.SolverFactory(self._ruc_solver).available():
            raise RuntimeError(f"Solver {self._ruc_solver} is not available to Pyomo")

        if self._sced_solver in supported_persistent_solvers:
            try:
                available = pe.SolverFactory(self._sced_solver+'_persistent').available()
                if not available:
                    self._print_persistence_warning(self._sced_solver)
                else:
                    self._sced_solver = self._sced_solver+'_persistent'
            except:
                self._print_persistence_warning(self._sced_solver)

        if not pe.SolverFactory(self._sced_solver).available():
            raise RuntimeError(f"Solver {self._sced_solver} is not available to Pyomo")


    def _print_persistence_warning(self, solver):
        print(f"WARNING: Solver {solver} supports persistence, which "
               "improves the performance of Prescient. Consider installing the "
              f"python bindings for {solver}.")

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
            from prescient.engine.egret import egret_plugin
            self.call_solver = egret_plugin.call_solver
            self.create_sced_instance = egret_plugin.create_sced_instance
            self.create_deterministic_ruc = egret_plugin.create_deterministic_ruc
            self.solve_deterministic_ruc = egret_plugin.solve_deterministic_ruc
            self.create_simulation_actuals = egret_plugin.create_simulation_actuals
            self.solve_deterministic_day_ahead_pricing_problem = egret_plugin.solve_deterministic_day_ahead_pricing_problem
            self._zero_out_costs = egret_plugin._zero_out_costs

            if options.simulator_plugin != None:
                try:
                    simulator_plugin_module = pyutilib.misc.import_file(options.simulator_plugin)
                except:
                    raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

                method_names = ["call_solver",
                                "create_sced_instance",
                                "create_deterministic_ruc",
                                "solve_deterministic_ruc",
                                "create_simulation_actuals",
                                "solve_deterministic_day_ahead_pricing_problem"]

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
                    from prescient.engine.egret_plugin import create_solve_deterministic_ruc
                    self.solve_deterministic_ruc = create_solve_deterministic_ruc(solve_function)
