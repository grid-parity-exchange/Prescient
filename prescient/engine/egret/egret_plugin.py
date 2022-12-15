#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
import os
import math
import logging
import datetime
import dateutil
import time

from pyomo.environ import value, Suffix

from egret.common.log import logger as egret_logger
from egret.data.model_data import ModelData
from egret.parsers.prescient_dat_parser import get_uc_model, create_model_data_dict_params
from egret.models.unit_commitment import _time_series_dict, _preallocated_list, _solve_unit_commitment, \
                                        _save_uc_results, create_tight_unit_commitment_model, \
                                        _get_uc_model
from egret.model_library.transmission.tx_calc import construct_connection_graph, get_N_minus_1_branches

from prescient.util import DEFAULT_MAX_LABEL_LENGTH
from prescient.util.math_utils import round_small_values
from prescient.simulator.data_manager import RucMarket
from ..modeling_engine import ForecastErrorMethod, PricingType, NetworkType as EngineNetworkType
from ..forecast_helper import get_forecastables, get_forecastables_with_inferral_method, InferralType
from .data_extractors import ScedDataExtractor
from . import reporting

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.data.data_provider import DataProvider
    from prescient.data.simulation_state import SimulationState
    from typing import Optional
    from egret.data.model_data import ModelData as EgretModel


def call_solver(solver, instance, options, solver_options, relaxed=False, set_instance=True):
    tee = options.output_solver_logs
    if not tee:
        egret_logger.setLevel(logging.WARNING)
    symbolic_solver_labels = options.symbolic_solver_labels
    mipgap = options.ruc_mipgap

    m, results, solver = _solve_unit_commitment(instance, solver, mipgap, None,
                                                tee, symbolic_solver_labels, 
                                                solver_options, None, relaxed, set_instance=set_instance)

    md = _save_uc_results(m, relaxed)

    if hasattr(results, 'egret_metasolver_status'):
        time = results.egret_metasolver_status['time']
    else:
        time = results.solver.wallclock_time

    return md, time, solver


def _zero_out_costs(sced_model, hours_in_objective):
    ''' zero out certain costs in a sced model '''
    # establish the objective function for the hour to simulate - which is simply to 
    # minimize production costs during this time period. no fixed costs to worry about.
    # however, we do need to penalize load shedding across all time periods - otherwise,
    # very bad things happen.

    m = sced_model

    hours_in = set()
    for idx,t in enumerate(m.TimePeriods):
        if idx < hours_in_objective:
            hours_in.add(t)
        else:
            break

    hours_out = set(m.TimePeriods) - hours_in

    for t in hours_out:
        for g in m.SingleFuelGenerators:
            m.ProductionCostConstr[g,t].deactivate()
            m.ProductionCost[g,t].value = 0.
            m.ProductionCost[g,t].fix()
        for g in m.DualFuelGenerators:
            m.DualFuelProductionCost[g,t].expr = 0.
        if m.regulation_service:
            for g in m.AGC_Generators:
                m.RegulationCostGeneration[g,t].expr = 0.
        if m.spinning_reserve:
            for g in m.ThermalGenerators:
                m.SpinningReserveCostGeneration[g,t].expr = 0.
        if m.non_spinning_reserve:
            for g in m.ThermalGenerators:
                m.NonSpinningReserveCostGeneration[g,t].expr = 0.
        if m.supplemental_reserve:
            for g in m.ThermalGenerators:
                m.SupplementalReserveCostGeneration[g,t].expr = 0.

    return

# TBD - we probably want to grab topology from somewhere, even if the stochastic 
#       RUC is not solved with the topology.
def create_sced_instance(data_provider:DataProvider,
                         current_state:SimulationState,
                         options,
                         sced_horizon,
                         forecast_error_method = ForecastErrorMethod.PRESCIENT
                         ):
    ''' Create a deterministic economic dispatch instance, given current forecasts and commitments.
    '''
    assert current_state is not None

    sced_md = data_provider.get_initial_actuals_model(options, sced_horizon, current_state.minutes_per_step)
    options.plugin_context.callback_manager.invoke_after_get_initial_actuals_model_for_sced_callbacks(
            options, sced_md)

    # Set initial state
    _copy_initial_state_into_model(options, current_state, sced_md)

    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if forecast_error_method is ForecastErrorMethod.PRESCIENT:
        # Warning: This method can see into the future!
        for forecastable, sced_data in get_forecastables(sced_md):
            future = current_state.get_future_actuals(forecastable)
            for t in range(sced_horizon):
                sced_data[t] = future[t]

    else:  # persistent forecast error:
        # Go through each time series that can be forecasted
        for forecastable, sced_data in get_forecastables(sced_md):
            forecast = current_state.get_forecasts(forecastable)
            # the first value is, by definition, the actual.
            sced_data[0] = current_state.get_current_actuals(forecastable)

            # Find how much the first forecast was off from the actual, as a fraction of 
            # the forecast. For all subsequent times, adjust the forecast by the same fraction.
            if forecast[0] == 0.0:
                forecast_error_ratio = 0.0
            else:
                forecast_error_ratio = sced_data[0] / forecast[0]

            for t in range(1, sced_horizon):
                sced_data[t] = forecast[t] * forecast_error_ratio

    _ensure_reserve_factor_honored(options, sced_md, range(sced_horizon))
    _ensure_contingencies_monitored(options, sced_md)

    # ensure reserves dispatch is consistent with pricing
    system = sced_md.data['system']
    for system_key, threshold_value in get_attrs_to_price_option(options):
        if threshold_value is not None and system_key not in system:
            system[system_key] = 1000.*threshold_value

    # Set generator commitments & future state
    for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
        # Start by preparing an empty array of the correct size for each generator
        fixed_commitment = [None]*sced_horizon
        g_dict['fixed_commitment'] = _time_series_dict(fixed_commitment)

        # Now fill it in with data
        for t in range(sced_horizon):
            fixed_commitment[t] = current_state.get_generator_commitment(g,t)

        # Look as far into the future as we can for future startups / shutdowns
        last_commitment = fixed_commitment[-1]
        for t in range(sced_horizon, current_state.timestep_count):
            this_commitment = current_state.get_generator_commitment(g,t)
            if (this_commitment - last_commitment) > 0.5:
                # future startup
                future_status_time_steps = ( t - sced_horizon + 1 )
                break
            elif (last_commitment - this_commitment) > 0.5:
                # future shutdown
                future_status_time_steps = -( t - sced_horizon + 1 )
                break
        else: # no break
            future_status_time_steps = 0
        g_dict['future_status'] = (current_state.minutes_per_step/60.) * future_status_time_steps

    if not options.no_startup_shutdown_curves:
        minutes_per_step = current_state.minutes_per_step
        for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
            if 'startup_curve' in g_dict:
                continue
            ramp_up_rate_sced = g_dict['ramp_up_60min'] * minutes_per_step/60.
            # this rarely happens, e.g., synchronous condenser
            if ramp_up_rate_sced == 0:
                continue
            if 'startup_capacity' not in g_dict:
                sced_startup_capacity = _calculate_sced_startup_shutdown_capacity_from_none(
                                            g_dict['p_min'], ramp_up_rate_sced)
            else:
                sced_startup_capacity = _calculate_sced_startup_shutdown_capacity_from_existing(
                                            g_dict['startup_capacity'], g_dict['p_min'], minutes_per_step)

            g_dict['startup_curve'] = [ sced_startup_capacity - i*ramp_up_rate_sced \
                                        for i in range(1,int(math.ceil(sced_startup_capacity/ramp_up_rate_sced))) ]

        for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
            if 'shutdown_curve' in g_dict:
                continue

            ramp_down_rate_sced = g_dict['ramp_down_60min'] * minutes_per_step/60.
            # this rarely happens, e.g., synchronous condenser
            if ramp_down_rate_sced == 0:
                continue
            # compute a new shutdown curve if we go from "on" to "off"
            if g_dict['initial_status'] > 0 and g_dict['fixed_commitment']['values'][0] == 0:
                power_t0 = g_dict['initial_p_output']
                # if we end up using a historical curve, it's important
                # for the time-horizons to match, particularly since this
                # function is also used to create long-horizon look-ahead
                # SCEDs for the unit commitment process
                create_sced_instance.shutdown_curves[g, minutes_per_step] = \
                        [ power_t0 - i*ramp_down_rate_sced for i in range(1,int(math.ceil(power_t0/ramp_down_rate_sced))) ]

            if (g,minutes_per_step) in create_sced_instance.shutdown_curves:
                g_dict['shutdown_curve'] = create_sced_instance.shutdown_curves[g,minutes_per_step]
            else:
                if 'shutdown_capacity' not in g_dict:
                    sced_shutdown_capacity = _calculate_sced_startup_shutdown_capacity_from_none(
                                                g_dict['p_min'], ramp_down_rate_sced)
                else:
                    sced_shutdown_capacity = _calculate_sced_startup_shutdown_capacity_from_existing(
                                                g_dict['shutdown_capacity'], g_dict['p_min'], minutes_per_step)

                g_dict['shutdown_curve'] = [ sced_shutdown_capacity - i*ramp_down_rate_sced \
                                             for i in range(1,int(math.ceil(sced_shutdown_capacity/ramp_down_rate_sced))) ]

    if not options.enforce_sced_shutdown_ramprate:
        for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
            # make sure the generator can immediately turn off
            g_dict['shutdown_capacity'] = max(g_dict['shutdown_capacity'], (60./current_state.minutes_per_step)*g_dict['initial_p_output'] + 1.)

    return sced_md

# cache for shutdown curves
create_sced_instance.shutdown_curves = dict()

def _calculate_sced_startup_shutdown_capacity_from_none(p_min, ramp_rate_sced):
    if isinstance(p_min, dict):
        sced_susd_capacity = [pm+ramp_rate_sced/2. for pm in p_min['values']]
        return sum(sced_susd_capacity)/len(sced_susd_capacity)
    else:
        return p_min + ramp_rate_sced/2.

def _calculate_sced_startup_shutdown_capacity_from_existing(startup_shutdown, p_min, minutes_per_step):
    susd_capacity_time_varying = isinstance(startup_shutdown, dict)
    p_min_time_varying = isinstance(p_min, dict)
    if p_min_time_varying and susd_capacity_time_varying:
        sced_susd_capacity = [ (susd - pm)*(minutes_per_step/60.) + pm \
                                    for pm, susd in zip(p_min['values'], startup_shutdown['values']) ]
        return sum(sced_susd_capacity)/len(sced_susd_capacity)
    elif p_min_time_varying:
        sced_susd_capacity = [ (startup_shutdown - pm)*(minutes_per_step/60.) + pm \
                                    for pm in p_min['values'] ]
        return sum(sced_susd_capacity)/len(sced_susd_capacity)
    elif susd_capacity_time_varying:
        sced_susd_capacity = [ (susd - p_min)*(minutes_per_step/60.) + p_min \
                                    for susd in startup_shutdown['values'] ]
        return sum(sced_susd_capacity)/len(sced_susd_capacity)
    else:
        return (startup_shutdown - p_min)*(minutes_per_step/60.) + p_min


##### BEGIN Deterministic RUC solvers and helper functions ########
###################################################################
# utility functions for computing various aspects                 #
# of a deterministic unit commitment solution.                    #
###################################################################

## NOTE: in closure for deterministic_ruc_solver_plugin
def create_solve_deterministic_ruc(deterministic_ruc_solver):

    def solve_deterministic_ruc(solver, options,
                                ruc_instance_for_this_period,
                                this_date,
                                this_hour,
                                network_type,
                                slack_type,
                                ptdf_manager):
        ruc_instance_for_this_period = deterministic_ruc_solver(ruc_instance_for_this_period, solver, options, network_type, slack_type, ptdf_manager)

        if options.write_deterministic_ruc_instances:
            current_ruc_filename = options.output_directory + os.sep + str(this_date) + \
                                                    os.sep + "ruc_hour_" + str(this_hour) + ".json"
            ruc_instance_for_this_period.write(current_ruc_filename)
            print("RUC instance written to file=" + current_ruc_filename)


        
        total_cost = ruc_instance_for_this_period.data['system']['total_cost']
        print("")
        print("Deterministic RUC Cost: {0:.2f}".format(total_cost))
    
        if options.output_ruc_solutions:
            print("")
            reporting.output_solution_for_deterministic_ruc(
                ruc_instance_for_this_period, 
                options.ruc_every_hours)
    
        print("")
        reporting.report_fixed_costs_for_deterministic_ruc(ruc_instance_for_this_period)
        reporting.report_generation_costs_for_deterministic_ruc(ruc_instance_for_this_period)
        print("")
        reporting.report_load_generation_mismatch_for_deterministic_ruc(ruc_instance_for_this_period)                        
        print("")
        reporting.report_curtailment_for_deterministic_ruc(ruc_instance_for_this_period)                        
    
        return ruc_instance_for_this_period
    return solve_deterministic_ruc

def _solve_deterministic_ruc(deterministic_ruc_data,
                             solver, 
                             options,
                             network_type,
                             slack_type,
                             ptdf_manager):

    # set RUC penalites high enough to drive commitment
    system = deterministic_ruc_data.data['system']
    for system_key, threshold_value in get_attrs_to_price_option(options):
        if threshold_value is not None and system_key not in system:
            system[system_key] = 1000.*threshold_value

    if options.ruc_network_type == EngineNetworkType.PTDF:
        ptdf_manager.mark_active(deterministic_ruc_data)

    st = time.time()
    pyo_model = create_tight_unit_commitment_model(deterministic_ruc_data,
                                                   ptdf_options=ptdf_manager.ruc_ptdf_options,
                                                   PTDF_matrix_dict=ptdf_manager.PTDF_matrix_dict,
                                                   network_constraints=network_type,
                                                   slack_type=slack_type)

    print("\nPyomo model construction time: %12.2f\n" % (time.time()-st))

    if options.ruc_network_type == EngineNetworkType.PTDF:
        # update in case lines were taken out
        ptdf_manager.PTDF_matrix_dict = pyo_model._PTDFs

    try:
        st = time.time()
        ruc_results, pyo_results, _  = call_solver(solver,
                                                   pyo_model,
                                                   options,
                                                   options.deterministic_ruc_solver_options)
        print("Pyomo model solve time:",time.time()-st)
    except:
        print("Failed to solve deterministic RUC instance - likely because no feasible solution exists!")        
        output_filename = "bad_ruc.json"
        deterministic_ruc_data.write(output_filename)
        print("Wrote failed RUC data to file=" + output_filename)
        raise

    if options.ruc_network_type == EngineNetworkType.PTDF:
        ptdf_manager.update_active(ruc_results)

    return ruc_results

## create this function with default solver
solve_deterministic_ruc = create_solve_deterministic_ruc(_solve_deterministic_ruc)

# utilities for creating a deterministic RUC instance, and a standard way to solve them.
def create_deterministic_ruc(options,
                             data_provider:DataProvider,
                             this_date, 
                             this_hour,
                             current_state:SimulationState,
                             ruc_horizon,
                             use_next_day_in_ruc):

    ruc_every_hours = options.ruc_every_hours

    start_day = this_date
    start_time = datetime.datetime.combine(start_day, datetime.time(hour=this_hour))

    # Create a new model
    md = data_provider.get_initial_forecast_model(options, ruc_horizon, 60)
    options.plugin_context.callback_manager.invoke_after_get_initial_forecast_model_for_ruc_callbacks(
            options, md)

    initial_ruc = current_state is None or current_state.timestep_count == 0

    # Populate the T0 data
    if initial_ruc:
        data_provider.populate_initial_state_data(options, md)
    else:
        _copy_initial_state_into_model(options, current_state, md)

    # Populate forecasts
    infer_second_day = (not use_next_day_in_ruc)
    forecast_request_count = 24 if infer_second_day else ruc_horizon 
    data_provider.populate_with_forecast_data(options, start_time, forecast_request_count, 
                                              60, md)

    # Make some near-term forecasts more accurate
    ruc_delay = -(options.ruc_execution_hour%(-options.ruc_every_hours))
    if options.ruc_prescience_hour > ruc_delay:
        improved_hour_count = options.ruc_prescience_hour - ruc_delay
        for forecastable, forecast in get_forecastables(md):
            actuals = current_state.get_future_actuals(forecastable)
            for t in range(0, improved_hour_count):
                forecast_portion = (ruc_delay+t)/options.ruc_prescience_hour
                actuals_portion = 1-forecast_portion
                forecast[t] = forecast_portion*forecast[t] + actuals_portion*actuals[t]

    if infer_second_day:
        for infer_type, vals in get_forecastables_with_inferral_method(md):
            for t in range(24, ruc_horizon):
                if infer_type == InferralType.COPY_FIRST_DAY:
                    # Copy from first 24 to second 24
                    vals[t] = vals[t-24]
                else:
                    # Repeat the final value from day 1
                    vals[t] = vals[23]

    # Ensure the reserve requirement is satisfied
    _ensure_reserve_factor_honored(options, md, range(ruc_horizon))

    _ensure_contingencies_monitored(options, md, initial_ruc)

    return md




###### END Deterministic RUC solvers and helper functions #########


def create_pricing_model(model_data,
                         network_constraints='ptdf_power_flow',
                         relaxed=True,
                         **kwargs):
    '''
    Create a model appropriate for pricing
    '''
    formulation_list = [
                        'garver_3bin_vars',
                        'garver_power_vars',
                        'MLR_reserve_vars',
                        'pan_guan_gentile_KOW_generation_limits',
                        'damcikurt_ramping',
                        'KOW_production_costs_tightened',
                        'rajan_takriti_UT_DT',
                        'KOW_startup_costs',
                         network_constraints,
                       ]
    return _get_uc_model(model_data, formulation_list, relaxed, **kwargs)

def _get_fixed_if_off(cur_commit, cur_fixed):
    cur_commit = cur_commit['values']
    if cur_fixed is None:
        cur_fixed = [None for _ in cur_commit]
    elif isinstance(cur_fixed, dict):
        cur_fixed = cur_fixed['values']
    else:
        cur_fixed = [cur_fixed for _ in cur_commit]

    new_fixed = [None]*len(cur_commit)
    for idx, (fixed, committed) in enumerate(zip(cur_fixed, cur_commit)):
        if fixed is None:
            new_fixed[idx] = None if committed == 1 else 0
        else:
            new_fixed[idx] = fixed
    return {'data_type':'time_series', 'values':new_fixed}

def solve_deterministic_day_ahead_pricing_problem(solver, ruc_results, options, ptdf_manager):

    ## create a copy because we want to maintain the solution data
    ## in ruc_results
    pricing_type = options.day_ahead_pricing
    print("Computing day-ahead prices using method "+pricing_type.name+".")
    
    pricing_instance = ruc_results.clone()
    if pricing_type == PricingType.LMP:
        for g, g_dict in pricing_instance.elements(element_type='generator', generator_type='thermal'):
            g_dict['fixed_commitment'] = g_dict['commitment']
            if 'reg_provider' in g_dict:
                g_dict['fixed_regulation'] = g_dict['reg_provider']
        ## TODO: add fixings for storage; need hooks in EGRET
    elif pricing_type == PricingType.ELMP:
        ## for ELMP we fix all commitment binaries that were 0 in the RUC solve
        time_periods = pricing_instance.data['system']['time_keys']
        for g, g_dict in pricing_instance.elements(element_type='generator', generator_type='thermal'):
            g_dict['fixed_commitment'] = _get_fixed_if_off(g_dict['commitment'], 
                                                           g_dict.get('fixed_commitment', None))
            if 'reg_provider' in g_dict:
                g_dict['fixed_regulation'] = _get_fixed_if_off(g_dict['reg_provider'],
                                                               g_dict.get('fixed_regulation', None))
        ## TODO: add fixings for storage; need hooks in EGRET
    elif pricing_type == PricingType.ACHP:
        # don't do anything
        pass
    else:
        raise RuntimeError("Unknown pricing type "+pricing_type+".")

    system = pricing_instance.data['system']

    # In case of shortfall, the price skyrockets, so we threshold the value.
    for system_key, threshold_value in get_attrs_to_price_option(options):
        if threshold_value is not None and ((system_key not in system) or
                (system[system_key] > threshold_value)):
            system[system_key] = threshold_value

    if options.ruc_network_type == EngineNetworkType.PTDF:
        ptdf_manager.mark_active(pricing_instance)

    pyo_model = create_pricing_model(pricing_instance, relaxed=True,
                                     ptdf_options=ptdf_manager.damarket_ptdf_options,
                                     PTDF_matrix_dict=ptdf_manager.PTDF_matrix_dict)

    pyo_model.dual = Suffix(direction=Suffix.IMPORT)

    try:
        ## TODO: Should there be separate options for this run?
        pricing_results, _, _ = call_solver(solver,
                                            pyo_model,
                                            options,
                                            options.deterministic_ruc_solver_options,
                                            relaxed=True)
    except:
        print("Failed to solve pricing instance - likely because no feasible solution exists!")        
        output_filename = "bad_pricing.json"
        pricing_instance.write(output_filename)
        print("Wrote failed RUC model to file=" + output_filename)
        raise

    if options.ruc_network_type == EngineNetworkType.PTDF:
        ptdf_manager.update_active(pricing_results)

    day_ahead_prices = {}
    for b, b_dict in pricing_results.elements(element_type='bus'):
        for t,lmp in enumerate(b_dict['lmp']['values']):
            day_ahead_prices[b,t] = lmp

    ## change the penalty prices to the caps, if necessary
    thermal_reserve_cleared_DA = {}
    DA_reserve_requirements = {}
    DA_reserve_shortfalls = {}
    DA_reserve_prices = {}
    for reserve_prod in ScedDataExtractor.get_reserve_products(pricing_results):
        reserve_data = ScedDataExtractor._get_reserve_parent(pricing_results, reserve_prod)

        DA_reserve_prices[reserve_prod] = list(reserve_data[f'{reserve_prod.reserve_name}_price']['values'])


        res_supplied_name = f'{reserve_prod.reserve_name}_supplied'
        g_reserve_values = { g : g_dict[res_supplied_name]['values'] 
                            for g, g_dict in ruc_results.elements(element_type='generator', generator_type='thermal')
                            if ScedDataExtractor.generator_is_in_scope(g_dict, reserve_prod.region_type, reserve_prod.region_name)}
        reserve_shortfall = reserve_data[f'{reserve_prod.reserve_name}_shortfall']['values']
        reserve_requirement = reserve_data[f'{reserve_prod.reserve_name}_requirement']
        if isinstance(reserve_requirement, dict):
            reserve_requirement = reserve_requirement['values']
        else:
            reserve_requirement = [reserve_requirement]*len(reserve_shortfall)
        DA_reserve_shortfalls[reserve_prod] = list(reserve_shortfall)
        DA_reserve_requirements[reserve_prod] = list(reserve_requirement)

        thermal_reserve_cleared_DA[reserve_prod] = {(g,t): reserve_vals[t]
                                                    for g, reserve_vals in g_reserve_values.items()
                                                    for t in range(0,options.ruc_every_hours)}

    thermal_gen_cleared_DA = {}
    renewable_gen_cleared_DA = {}
    virtual_gen_cleared_DA = {}

    for g, g_dict in ruc_results.elements(element_type='generator'):
        pg = g_dict['pg']['values']
        if g_dict['generator_type'] == 'thermal':
            store_dict = thermal_gen_cleared_DA
        elif g_dict['generator_type'] == 'renewable':
            store_dict = renewable_gen_cleared_DA
        elif g_dict['generator_type'] == 'virtual':
            store_dict = virtual_gen_cleared_DA
        else:
            raise RuntimeError(f"Unrecognized generator type {g_dict['generator_type']}")
        for t in range(0,options.ruc_every_hours):
            store_dict[g,t] = pg[t]

    return RucMarket(day_ahead_prices=day_ahead_prices,
                    DA_reserve_prices=DA_reserve_prices,
                    DA_reserve_requirements=DA_reserve_requirements,
                    DA_reserve_shortfalls=DA_reserve_shortfalls,
                    thermal_gen_cleared_DA=thermal_gen_cleared_DA,
                    thermal_reserve_cleared_DA=thermal_reserve_cleared_DA,
                    renewable_gen_cleared_DA=renewable_gen_cleared_DA,
                    virtual_gen_cleared_DA=virtual_gen_cleared_DA)


def create_simulation_actuals(
        options:Options, data_provider:DataProvider, 
        this_date:datetime.date, this_hour:int,
        step_size_minutes:int) -> EgretModel:
    ''' Get an Egret model consisting of data to be treated as actuals, starting at a given time.

    Parameters
    ----------
    options:Options
        Global option values
    data_provider: DataProvider
        An object that can provide actual and/or forecast data for the requested days
    this_date: date
        The date of the first time step for which data should be retrieved
    this_hour: int
        0-based index of the first hour of the day for which data should be retrieved
    step_size_minutes: int
        The number of minutes between each time step
    ''' 
    # Convert time string to time
    start_time = datetime.datetime.combine(this_date, 
                                           datetime.time(hour=this_hour))

    # Pick whether we're getting actuals or forecasts
    if options.simulate_out_of_sample:
        get_data_func = data_provider.populate_with_actuals
    else:
        print("")
        print("***WARNING: Simulating the forecast scenario when running deterministic RUC - "
              "time consistency across midnight boundaries is not guaranteed, and may lead to threshold events.")
        get_data_func = data_provider.populate_with_forecast_data

    # Get a new model
    total_step_count = options.ruc_horizon * 60 // step_size_minutes
    md = data_provider.get_initial_actuals_model(options, total_step_count, step_size_minutes)
    options.plugin_context.callback_manager.invoke_after_get_initial_actuals_model_for_simulation_actuals_callbacks(
            options, md)

    # Fill it in with data
    if this_hour == 0:
        get_data_func(options, start_time, total_step_count, step_size_minutes, md)
    else:
        # only get up to 24 hours of data, then copy it
        timesteps_per_day = 24 * 60 / step_size_minutes
        steps_to_request = min(timesteps_per_day, total_step_count)
        get_data_func(options, start_time, steps_to_request, step_size_minutes, md)
        for _, vals in get_forecastables(md):
            for t in range(timesteps_per_day, total_step_count):
                vals[t] = vals[t-timesteps_per_day]

    return md

def _ensure_reserve_factor_honored(options:Options, md:EgretModel, time_periods:Iterable[int]) -> None:
    ''' Adjust reserve requirements to satisfy the reserve factor.

    For each time period in time_periods, ensure that the reserve requirement is no less than
    the total load for that time period multiplied by the reserve factor.  If the reserve 
    requirement for a time is too low it is raised, otherwise it is left alone.

    '''
    if options.reserve_factor > 0:
        reserve_factor = options.reserve_factor
        if 'reserve_requirement' not in md.data['system']:
            md.data['system']['reserve_requirement'] = 0.
        if not isinstance(md.data['system']['reserve_requirement'], dict):
            fixed_requirement = md.data['system']['reserve_requirement']
            md.data['system']['reserve_requirement'] = \
                    { 'data_type' : 'time_series',
                      'values' : [fixed_requirement for _ in time_periods] }
        reserve_reqs = md.data['system']['reserve_requirement']['values']
        for t in time_periods:
            total_load = sum(bdata['p_load']['values'][t]
                             for bus, bdata in md.elements('load'))
            min_reserve = reserve_factor*total_load
            if reserve_reqs[t] < min_reserve:
                reserve_reqs[t] = min_reserve

def _ensure_contingencies_monitored(options:Options, md:EgretModel, initial_ruc:bool = False) -> None:
    ''' Add contingency screening, if that option is enabled '''
    if initial_ruc:
        _ensure_contingencies_monitored.contingency_dicts = {}

    for bn, b in md.elements('branch'): 
        if not b.get('in_service', True):
            raise RuntimeError(f"Remove branches from service by setting the `planned_outage` attribute. "
                    f"Branch {bn} has `in_service`:False")
    for bn, b in md.elements('dc_branch'): 
        if not b.get('in_service', True):
            raise RuntimeError(f"Remove branches from service by setting the `planned_outage` attribute. "
                    f"DC Branch {bn} has `in_service`:False")
    for bn, b in md.elements('bus'): 
        if not b.get('in_service', True):
            raise RuntimeError(f"Buses cannot be removed from service in Prescient")

    if options.monitor_all_contingencies:
        key = []
        for bn, b in md.elements('branch'):
            if 'planned_outage' in b:
                if isinstance(b['planned_outage'], dict):
                    if any(b['planned_outage']['values']):
                        key.append(b)
                elif b['planned_outage']:
                    key.append(b)
        key = tuple(key)
        if key not in _ensure_contingencies_monitored.contingency_dicts:
            mapping_bus_to_idx = { k : i for i,k in enumerate(md.data['elements']['bus'].keys())}
            graph = construct_connection_graph(md.data['elements']['branch'], mapping_bus_to_idx)
            contingency_list = get_N_minus_1_branches(graph, md.data['elements']['branch'], mapping_bus_to_idx)
            contingency_dict = { cn : {'branch_contingency':cn} for cn in contingency_list} 
            _ensure_contingencies_monitored.contingency_dicts[key] = contingency_dict

        md.data['elements']['contingency'] = _ensure_contingencies_monitored.contingency_dicts[key]

def _copy_initial_state_into_model(options:Options, 
                                   current_state:SimulationState, 
                                   md:EgretModel):
    for g, g_dict in md.elements('generator', generator_type='thermal'):
        g_dict['initial_status'] = current_state.get_initial_generator_state(g)
        g_dict['initial_p_output']  = current_state.get_initial_power_generated(g)
    for s,s_dict in md.elements('storage'):
        s_dict['initial_state_of_charge'] = current_state.get_initial_state_of_charge(s)

def get_attrs_to_price_option(options:Options):
    '''
    Create a map from internal attributes to various price thresholds
    for the LMP SCED
    '''
    return {
            'load_mismatch_cost' : options.price_threshold,
            'contingency_flow_violation_cost' : options.contingency_price_threshold,
            'transmission_flow_violation_cost' : options.transmission_price_threshold,
            'interface_flow_violation_cost' : options.interface_price_threshold,
            'reserve_shortfall_cost' : options.reserve_price_threshold,
            'regulation_penalty_price' : options.regulation_price_threshold,
            'spinning_reserve_penalty_price' : options.spinning_reserve_price_threshold,
            'non_spinning_reserve_penalty_price' : options.non_spinning_reserve_price_threshold,
            'supplemental_reserve_penalty_price' : options.supplemental_reserve_price_threshold,
            'flexible_ramp_penalty_price' : options.flex_ramp_price_threshold,
            }.items()
