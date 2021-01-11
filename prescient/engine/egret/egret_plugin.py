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

from pyomo.environ import value, Suffix

from egret.common.log import logger as egret_logger
from egret.data.model_data import ModelData
from egret.parsers.prescient_dat_parser import get_uc_model, create_model_data_dict_params
from egret.models.unit_commitment import _time_series_dict, _preallocated_list, _solve_unit_commitment, \
                                        _save_uc_results, create_tight_unit_commitment_model, \
                                        _get_uc_model

from prescient.util import DEFAULT_MAX_LABEL_LENGTH
from prescient.util.math_utils import round_small_values
from prescient.simulator.data_manager import RucMarket
from ..modeling_engine import ForecastErrorMethod
from ..forecast_helper import get_forecastables
from . import reporting

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.data.data_provider import DataProvider
    from prescient.data.simulation_state import SimulationState
    from typing import Optional
    from egret.data.model_data import ModelData as EgretModel

uc_abstract_data_model = get_uc_model()

########################################################################################
# a utility to find the "nearest" - quantified via Euclidean distance - scenario among #
# a candidate set relative to the input scenario, up through and including the         #
# specified simulation hour.                                                           #
########################################################################################

def call_solver(solver,instance,options,solver_options,relaxed=False, set_instance=True):
    tee = options.output_solver_logs
    if not tee:
        egret_logger.setLevel(logging.WARNING)
    symbolic_solver_labels = options.symbolic_solver_labels
    mipgap = options.ruc_mipgap

    solver_options_dict = dict()
    for s in solver_options:
        opts = s.split(' ')
        for opt in opts:
            option, val = opt.split('=')
            try:
                val = float(val)
            except:
                pass
            solver_options_dict[option] = val

    m, results, solver = _solve_unit_commitment(instance, solver, mipgap, None,
                                                tee, symbolic_solver_labels, 
                                                solver_options_dict, None, relaxed, set_instance=set_instance)

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
    ''' Create an hourly deterministic economic dispatch instance, given current forecasts and commitments.
    '''
    assert current_state != None

    sced_md = data_provider.get_initial_model(options, sced_horizon, current_state.minutes_per_step)

    # Set initial state
    _copy_initial_state_into_model(options, current_state, sced_md)

    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if forecast_error_method is ForecastErrorMethod.PRESCIENT:
        # Warning: This method can see into the future!
        future_actuals = current_state.get_future_actuals()
        sced_forecastables, = get_forecastables(sced_md)
        for future,sced_data in zip(future_actuals, sced_actuals):
            for t in range(sced_horizon):
                sced_data[t] = future[t]

    else:  # persistent forecast error:
        current_actuals = current_state.get_current_actuals()
        forecasts = current_state.get_forecasts()
        sced_forecastables = get_forecastables(sced_md)
        # Go through each time series that can be forecasted
        for current_actual, forecast, (sced_data,) in zip(current_actuals, forecasts, sced_forecastables):
            # the first value is, by definition, the actual.
            sced_data[0] = current_actual

            # Find how much the first forecast was off from the actual, as a fraction of 
            # the forecast. For all subsequent times, adjust the forecast by the same fraction.
            current_forecast = forecast[0]
            if current_forecast == 0.0:
                forecast_error_ratio = 0.0
            else:
                forecast_error_ratio = current_actual / forecast[0]

            for t in range(1, sced_horizon):
                sced_data[t] = forecast[t] * forecast_error_ratio

    _ensure_reserve_factor_honored(options, sced_md, range(sced_horizon))

    ## TODO: propogate relax_t0_ramping_initial_day into this function
    ## if relaxing initial ramping, we need to relax it in the first SCED as well
    assert options.relax_t0_ramping_initial_day is False

    # Set generator commitments
    for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
        # Start by preparing an empty array of the correct size for each generator
        fixed_commitment = [None]*sced_horizon
        g_dict['fixed_commitment'] = _time_series_dict(fixed_commitment)

        # Now fill it in with data
        for t in range(sced_horizon):
            fixed_commitment[t] = current_state.get_generator_commitment(g,t)

    return sced_md


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
                                ptdf_manager):
        ruc_instance_for_this_period = deterministic_ruc_solver(ruc_instance_for_this_period, solver, options, ptdf_manager)

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

def _solve_deterministic_ruc(deterministic_ruc_instance,
                            solver, 
                            options,
                            ptdf_manager):

    ptdf_manager.mark_active(deterministic_ruc_instance)
    pyo_model = create_tight_unit_commitment_model(deterministic_ruc_instance,
                                                   ptdf_options=ptdf_manager.ruc_ptdf_options,
                                                   PTDF_matrix_dict=ptdf_manager.PTDF_matrix_dict)

    # update in case lines were taken out
    ptdf_manager.PTDF_matrix_dict = pyo_model._PTDFs

    try:
        ruc_results, pyo_results, _  = call_solver(solver,
                                                   pyo_model,
                                                   options,
                                                   options.deterministic_ruc_solver_options)
    except:
        print("Failed to solve deterministic RUC instance - likely because no feasible solution exists!")        
        output_filename = "bad_ruc.json"
        deterministic_ruc_instance.write(output_filename)
        print("Wrote failed RUC model to file=" + output_filename)
        raise

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
    md = data_provider.get_initial_model(options, ruc_horizon, 60)

    # Populate the T0 data
    if current_state is None or current_state.timestep_count == 0:
        data_provider.populate_initial_state_data(options, start_day, md)
    else:
        _copy_initial_state_into_model(options, current_state, md)

    # Populate forecasts
    copy_first_day = (not use_next_day_in_ruc) and (this_hour != 0)
    forecast_request_count = 24 if copy_first_day else ruc_horizon 
    data_provider.populate_with_forecast_data(options, start_time, forecast_request_count, 
                                              60, md)

    # Make some near-term forecasts more accurate
    ruc_delay = -(options.ruc_execution_hour%(-options.ruc_every_hours))
    if options.ruc_prescience_hour > ruc_delay + 1:
        improved_hour_count = options.ruc_prescience_hour - ruc_delay - 1
        for forecast, actuals in zip(get_forecastables(md),
                                     current_state.get_future_actuals()):
            for t in range(0, improved_hour_count):
                forecast_portion = (ruc_delay+t)/options.ruc_prescience_hour
                actuals_portion = 1-forecast_portion
                forecast[t] = forecast_portion*forecast[t] + actuals_portion*actuals[t]


    # Ensure the reserve requirement is satisfied
    _ensure_reserve_factor_honored(options, md, range(forecast_request_count))

    # Copy from first 24 to second 24, if necessary
    if copy_first_day:
        for vals, in get_forecastables(md):
            for t in range(24, ruc_horizon):
                vals[t] = vals[t-24]

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
    print("Computing day-ahead prices using method "+pricing_type+".")
    
    pricing_instance = ruc_results.clone()
    if pricing_type == "LMP":
        for g, g_dict in pricing_instance.elements(element_type='generator', generator_type='thermal'):
            g_dict['fixed_commitment'] = g_dict['commitment']
            if 'reg_provider' in g_dict:
                g_dict['fixed_regulation'] = g_dict['reg_provider']
        ## TODO: add fixings for storage; need hooks in EGRET
    elif pricing_type == "ELMP":
        ## for ELMP we fix all commitment binaries that were 0 in the RUC solve
        time_periods = pricing_instance.data['system']['time_keys']
        for g, g_dict in pricing_instance.elements(element_type='generator', generator_type='thermal'):
            g_dict['fixed_commitment'] = _get_fixed_if_off(g_dict['commitment'], 
                                                           g_dict.get('fixed_commitment', None))
            if 'reg_provider' in g_dict:
                g_dict['fixed_regulation'] = _get_fixed_if_off(g_dict['reg_provider'],
                                                               g_dict.get('fixed_regulation', None))
        ## TODO: add fixings for storage; need hooks in EGRET
    elif pricing_type == "aCHP":
        # don't do anything
        pass
    else:
        raise RuntimeError("Unknown pricing type "+pricing_type+".")

    ## change the penalty prices to the caps, if necessary
    reserve_requirement = ('reserve_requirement' in pricing_instance.data['system'])

    # In case of demand shortfall, the price skyrockets, so we threshold the value.
    if pricing_instance.data['system']['load_mismatch_cost'] > options.price_threshold:
        pricing_instance.data['system']['load_mismatch_cost'] = options.price_threshold

    # In case of reserve shortfall, the price skyrockets, so we threshold the value.
    if reserve_requirement:
        if pricing_instance.data['system']['reserve_shortfall_cost'] > options.reserve_price_threshold:
            pricing_instance.data['system']['reserve_shortfall_cost'] = options.reserve_price_threshold

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

    ptdf_manager.update_active(pricing_results)

    ## Debugging
    if pricing_results.data['system']['total_cost'] > ruc_results.data['system']['total_cost']*(1.+1.e-06):
        print("The pricing run had a higher objective value than the MIP run. This is indicative of a bug.")
        print("Writing LP pricing_problem.json")
        output_filename = 'pricing_instance.json'
        pricing_results.write(output_filename)

        output_filename = 'ruc_results.json'
        ruc_results.write(output_filename)

        raise RuntimeError("Halting due to bug in pricing.")

    day_ahead_prices = {}
    for b, b_dict in pricing_results.elements(element_type='bus'):
        for t,lmp in enumerate(b_dict['lmp']['values']):
            day_ahead_prices[b,t] = lmp

    if reserve_requirement:
        day_ahead_reserve_prices = {}
        for t,price in enumerate(pricing_results.data['system']['reserve_price']['values']):
            # Thresholding the value of the reserve price to the passed in option
            day_ahead_reserve_prices[t] = price

        print("Recalculating RUC reserve procurement")

        ## scale the provided reserves by the amount over we are
        thermal_reserve_cleared_DA = {}

        g_reserve_values = { g : g_dict['rg']['values'] for g, g_dict in ruc_results.elements(element_type='generator', generator_type='thermal') }
        reserve_shortfall = ruc_results.data['system']['reserve_shortfall']['values']
        reserve_requirement = ruc_results.data['system']['reserve_requirement']['values']

        for t in range(0,options.ruc_every_hours):
            reserve_provided_t = sum(reserve_vals[t] for reserve_vals in g_reserve_values.values()) 
            reserve_shortfall_t = reserve_shortfall[t]
            reserve_requirement_t = reserve_requirement[t]

            surplus_reserves_t = reserve_provided_t + reserve_shortfall_t - reserve_requirement_t

            ## if there's a shortfall, grab the full amount from the RUC solve
            ## or if there's no provided reserves this can safely be set to 1.
            if round_small_values(reserve_shortfall_t) > 0 or reserve_provided_t == 0:
                surplus_multiple_t = 1.
            else:
                ## scale the reserves from the RUC down by the same fraction
                ## so that they exactly meed the needed reserves
                surplus_multiple_t = reserve_requirement_t/reserve_provided_t
            for g, reserve_vals in g_reserve_values.items():
                thermal_reserve_cleared_DA[g,t] = reserve_vals[t]*surplus_multiple_t
    else:
        day_ahead_reserve_prices = { t : 0. for t in enumerate(ruc_results.data['system']['time_keys']) } 
        thermal_reserve_cleared_DA = { (g,t) : 0. \
                for t in enumerate(ruc_results.data['system']['time_keys']) \
                for g,_ in ruc_results.elements(element_type='generator', generator_type='thermal') }
               
    thermal_gen_cleared_DA = {}
    renewable_gen_cleared_DA = {}

    for g, g_dict in ruc_results.elements(element_type='generator'):
        pg = g_dict['pg']['values']
        if g_dict['generator_type'] == 'thermal':
            store_dict = thermal_gen_cleared_DA
        elif g_dict['generator_type'] == 'renewable':
            store_dict = renewable_gen_cleared_DA
        else:
            raise RuntimeError(f"Unrecognized generator type {g_dict['generator_type']}")
        for t in range(0,options.ruc_every_hours):
            store_dict[g,t] = pg[t]

    return RucMarket(day_ahead_prices=day_ahead_prices,
                    day_ahead_reserve_prices=day_ahead_reserve_prices,
                    thermal_gen_cleared_DA=thermal_gen_cleared_DA,
                    thermal_reserve_cleared_DA=thermal_reserve_cleared_DA,
                    renewable_gen_cleared_DA=renewable_gen_cleared_DA)


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
        if options.run_deterministic_ruc:
            print("")
            print("***WARNING: Simulating the forecast scenario when running deterministic RUC - "
                  "time consistency across midnight boundaries is not guaranteed, and may lead to threshold events.")
        get_data_func = data_provider.populate_with_forecast_data

    # Get a new model
    total_step_count = options.ruc_horizon * 60 // step_size_minutes
    md = data_provider.get_initial_model(options, total_step_count, step_size_minutes)

    # Fill it in with data
    data_provider.populate_initial_state_data(options, start_time.date(), md)
    if this_hour == 0:
        get_data_func(options, start_time, total_step_count, step_size_minutes, md)
    else:
        # only get up to 24 hours of data, then copy it
        timesteps_per_day = 24 * 60 / step_size_minutes
        steps_to_request = math.min(timesteps_per_day, total_step_count)
        get_data_func(options, start_time, steps_to_request, step_size_minutes, md)
        for vals, in get_forecastables(md):
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
        reserve_reqs = md.data['system']['reserve_requirement']['values']
        for t in time_periods:
            total_load = sum(bdata['p_load']['values'][t]
                             for bus, bdata in md.elements('load'))
            min_reserve = reserve_factor*total_load
            if reserve_reqs[t] < min_reserve:
                reserve_reqs[t] = min_reserve

def _copy_initial_state_into_model(options:Options, 
                                   current_state:SimulationState, 
                                   md:EgretModel):
    for g, g_dict in md.elements('generator', generator_type='thermal'):
        g_dict['initial_status'] = current_state.get_initial_generator_state(g)
        g_dict['initial_p_output']  = current_state.get_initial_power_generated(g)
    for s,s_dict in md.elements('storage'):
        s_dict['initial_state_of_charge'] = current_state.get_initial_state_of_charge(s)
