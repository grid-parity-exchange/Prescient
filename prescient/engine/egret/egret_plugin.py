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
from prescient.engine.modeling_engine import ForecastErrorMethod
from prescient.simulator.data_manager import RucMarket

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.data.data_provider import DataProvider
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


## utility for constructing pyomo data dictionary from the passed in parameters to use
def _get_data_dict( data_model, time_horizon, demand_dict, reserve_dict, reserve_factor, \
                    min_nondispatch_dict, max_nondispatch_dict, \
                    UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict):

    ## data_model is now an EGERT ModelData object

    # cut off the remaining time steps
    md = data_model.clone_at_time_indices(range(time_horizon))

    load = md.data['elements']['load']
    thermal_gens = dict(md.elements(element_type='generator', generator_type='thermal'))
    renewable_gens = dict(md.elements(element_type='generator', generator_type='renewable'))
    storage = md.data['elements']['storage']

    time_periods = list(range(1,time_horizon+1))

    calculate_reserve_factor = reserve_factor > 0

    if calculate_reserve_factor:
        total_demand = [0. for _ in range(time_horizon)]

    for l, l_dict in load.items():
        l_values = l_dict['p_load']['values']
        for idx,t in enumerate(time_periods):
            l_values[idx] = demand_dict[l,t]
            if calculate_reserve_factor:
                total_demand[idx] += demand_dict[l,t]

    for r, r_dict in renewable_gens.items():
        p_min = r_dict['p_min']['values']
        p_max = r_dict['p_max']['values']
        for idx,t in enumerate(time_periods):
            p_min[idx] = min_nondispatch_dict[r,t]
            p_max[idx] = max_nondispatch_dict[r,t]

    for g, g_dict in thermal_gens.items():
        g_dict['initial_status'] = UnitOnT0StateDict[g]
        g_dict['initial_p_output'] = PowerGeneratedT0Dict[g]
    
    # TODO: storage should really have an initial output/input as well
    for s, s_dict in storage.items():
        s_dict['initial_state_of_charge'] = StorageSocOnT0Dict[s]
    
    if reserve_dict is not None:
        reserve_vals = md.data['system']['reserve_requirement']['values']
        for idx,t in enumerate(time_periods):
            reserve_vals[idx] = reserve_dict[t]

    if calculate_reserve_factor:
        reserve_vals = md.data['system']['reserve_requirement']['values']
        for idx, val in enumerate(reserve_vals):
            if val < reserve_factor*total_demand[idx]:
                reserve_vals[idx] = reserve_factor*total_demand[idx]

    return md

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
#
# a utility to create the hourly (deterministic) economic dispatch instance, given
# the prior day's RUC solution, the basic ruc model, and the ruc instance to simulate.
# TBD - we probably want to grab topology from somewhere, even if the stochastic 
#       RUC is not solved with the topology.
# a description of the input arguments:
#
# 1 - sced_model: an uninstantiated (abstract) unit commitment model. 
#                 used as the basis for the constructed and returned model.
#
# 2 - stochastic_scenario_instances: a map from scenario name to scenario instance, for the stochastic RUC.
#                              these instances are solved, with results loaded.
#
# 3 - ruc_instance_to_simulate: a single scenario / deterministic RUC instance. 
#                               (projected) realized demands and renewables output for the SCED are extracted 
#                               from this instance, as are static / topological features of the network.
#
# 4 - prior_sced_instance: the sced instance from the immediately preceding time period.
#                          provides the initial generator (T0) state information.
#                          can be None, in which case - TBD?!
#
# 5 - hour_to_simulate: the hour of the day to simulate, 0-based.
#

# NOTES: 
# 1) It is critical that the SCED be multi-period, to manage ramp-down / shut-off constraints. 
# 2) As a result, we define SCED as a weird-ish version of the SCUC - namely one in which
#    the initial conditions are taken from the prior period SCED instance time period 1 (the
#    first real period), and the binaries for all remaining subsequent time periods in the day
#    are taken from (and fixed to) the values in the stochastic RUC - as these are the commitments 
#    that must be satisfied.
# 3) We are presently taking the demand in future time periods from the instance to simulate. this
#    may sound a bit like cheating, but we can argue that the one would use the scenario from the
#    RUC that is closest to that observed. this brings to mind a matching / selection scheme, but
#    that is presently not conducted.

## TODO: propogate relax_t0_ramping_initial_day into this function
def create_sced_instance(today_ruc_instance, 
                         tomorrow_ruc_instance,
                         ruc_instance_to_simulate,  # provides actuals and an instance to query
                         prior_sced_instance,  # used for initial conditions if initial_from_ruc=False
                         actual_demand,  # native Python dictionary containing the actuals
                         demand_forecast_error, 
                         actual_min_renewables,  # native Python dictionary containing the actuals
                         actual_max_renewables,  # native Python dictionary containing the actuals
                         renewables_forecast_error,
                         hour_to_simulate,
                         reserve_factor, 
                         options,
                         hours_in_objective=1, # by default, just worry about cost minimization for the hour to simulate.
                         sced_horizon=24, # by default simulate a SCED for 24 hours, to pass through the midnight boundary successfully
                         ruc_every_hours=24,
                         initialize_from_ruc=True,
                         forecast_error_method = ForecastErrorMethod.PRESCIENT
                         ):

    assert ruc_instance_to_simulate != None
    assert hour_to_simulate >= 0
    assert reserve_factor >= 0.0

    if prior_sced_instance is None:
        assert hour_to_simulate == 0

    # NOTE: if the prior SCED instance is None, then we extract the unit T0 state information from the
    #       stochastic RUC instance (solved for the preceding day).

    # the input hour is 0-based, but the time periods in our UC and SCED optimization models are one-based.
    hour_to_simulate += 1 

    #################################################################
    # compute the T0 state parameters, based on the input instances #
    #################################################################

    UnitOnT0Dict         = {}
    UnitOnT0StateDict    = {}
    PowerGeneratedT0Dict = {}

    # there are a variety of ways to accomplish this, which can result in
    # radically different simulator behavior. these are driven by keyword
    # arguments - we don't perform checking for specification of multiple
    # alternatives. 

    if initialize_from_ruc:
        # the simplest initialization method is to set the initial conditions 
        # to those found in the RUC instance. this has 
        # the advantage that simulating from an in-sample scenario will yield
        # a feasible sced at each simulated hour.

        # Use the deterministic RUC directly.
        print("")
        print("Drawing initial conditions from deterministic RUC initial conditions")
        for g, g_dict in today_ruc_instance.elements(element_type='generator', generator_type='thermal'):
            UnitOnT0Dict[g] = int(g_dict['initial_status'] > 0)
            UnitOnT0StateDict[g] = g_dict['initial_status']
            PowerGeneratedT0Dict[g] = g_dict['initial_p_output']

    else:
        # TBD: Clean code below up - if we get this far, shouldn't we always have a prior sched instance?
        print("")
        print("Drawing initial conditions from prior SCED solution, if available")
        assert prior_sced_instance is not None
        #for g in sorted(ruc_instance_to_simulate.ThermalGenerators):
        for g, g_dict in prior_sced_instance.elements(element_type='generator', generator_type='thermal'):

            unit_on = int(round(g_dict['commitment']['values'][0]))
            UnitOnT0Dict[g] = unit_on

            # since we are dealing with a single time period model, propagate the
            # UnitOn state (which will be fixed in any case) backward into the past.
            # NOTE: These should be longer than any reasonable min up/down time, and for
            #       some steam units these can be longer than a day (FERC set as a maximum
            #       of 168, so I've doubled that here)
            if unit_on == 1:
                UnitOnT0StateDict[g] = 336
            else:
                UnitOnT0StateDict[g] = -336

            #PowerGeneratedT0Dict[g] = value(prior_sced_instance.PowerGenerated[g, 1])
            candidate_power_generated = g_dict['pg']['values'][0]

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.

            if isinstance(g_dict['p_min'], dict):
                min_power_output = g_dict['p_min']['values'][0]
            else:
                min_power_output = g_dict['p_min']
            if isinstance(g_dict['p_max'], dict):
                max_power_output = g_dict['p_max']['values'][0]
            else:
                max_power_output = g_dict['p_max']

                
            # TBD: Eventually make the 1e-5 an user-settable option.
            if unit_on == 0:
                # if the unit is off, then the power generated at t0 must be equal to 0 -
                # no tolerances allowed.
                PowerGeneratedT0Dict[g] = 0.0
            elif math.fabs(min_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = min_power_output
            elif math.fabs(max_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = max_power_output
            else:
                PowerGeneratedT0Dict[g] = candidate_power_generated
            
    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if forecast_error_method is ForecastErrorMethod.PRESCIENT:

        demand_dict = dict(((b, t+1), actual_demand[b, hour_to_simulate + t])
                           for b,_ in ruc_instance_to_simulate.elements(element_type='bus') for t in range(0, sced_horizon))
        min_nondispatch_dict = dict(((g, t+1), actual_min_renewables[g, hour_to_simulate + t])
                                    for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable')
                                    for t in range(0, sced_horizon))
        max_nondispatch_dict = dict(((g, t+1), actual_max_renewables[g, hour_to_simulate + t])
                                    for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable')
                                    for t in range(0, sced_horizon))

    else:  # persistent forecast error:

        demand_dict = {}
        min_nondispatch_dict = {}
        max_nondispatch_dict = {}

        # we're running in deterministic mode
        # which is all that is currently supported
        # the current hour is necessarily (by definition) the actual.
        for b,_ in ruc_instance_to_simulate.elements(element_type='bus'):
            demand_dict[(b,1)] = actual_demand[b, hour_to_simulate]

        # for each subsequent hour, apply a simple persistence forecast error model to account for deviations.
        for b,_ in ruc_instance_to_simulate.elements(element_type='bus'):
            forecast_error_now = demand_forecast_error[(b, hour_to_simulate)]
            actual_now = actual_demand[b, hour_to_simulate]
            forecast_now = actual_now + forecast_error_now

            for t in range(1, sced_horizon):
                # IMPT: forecast errors (and therefore forecasts) are relative to actual demand, 
                #       which is the only reason that the latter appears below - to reconstruct
                #       the forecast. thus, no presicence is involved.
                forecast_error_later = demand_forecast_error[(b,hour_to_simulate + t)]
                actual_later = actual_demand[b,hour_to_simulate + t]
                forecast_later = actual_later + forecast_error_later
                # 0 demand can happen, in some odd circumstances (not at the ISO level!).
                if forecast_now != 0.0:
                    demand_dict[(b, t+1)] = (forecast_later/forecast_now)*actual_now
                else:
                    demand_dict[(b, t+1)] = 0.0

        # repeat the above for renewables.
        for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable'):
            min_nondispatch_dict[(g, 1)] = actual_min_renewables[g, hour_to_simulate]
            max_nondispatch_dict[(g, 1)] = actual_max_renewables[g, hour_to_simulate]
            
        for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable'):
            forecast_error_now = renewables_forecast_error[(g, hour_to_simulate)]
            actual_now = actual_max_renewables[g, hour_to_simulate]
            forecast_now = actual_now + forecast_error_now

            for t in range(1, sced_horizon):
                # forecast errors are with respect to the maximum - that is the actual maximum power available.
                forecast_error_later = renewables_forecast_error[(g, hour_to_simulate + t)]
                actual_later = actual_max_renewables[g, hour_to_simulate + t]
                forecast_later = actual_later + forecast_error_later

                if forecast_now != 0.0:
                    max_nondispatch_dict[(g, t+1)] = (forecast_later/forecast_now)*actual_now
                else:
                    max_nondispatch_dict[(g, t+1)] = 0.0
                # TBD - fix this - it should be non-zero!
                min_nondispatch_dict[(g, t+1)] = 0.0

    ##########################################################################
    # construct the data dictionary for instance initialization from scratch #
    ##########################################################################

    if prior_sced_instance is not None:
        #StorageSocOnT0Dict = dict((s, value(prior_sced_instance.SocStorage[s, 1]))
        #                                     for s in sorted(ruc_instance_to_simulate.Storage))
        StorageSocOnT0Dict = { s : s_dict['state_of_charge']['values'][0] for s,s_dict in prior_sced_instance.elements('storage') }
    else:
        StorageSocOnT0Dict = { s : s_dict['state_of_charge']['values'][0] for s,s_dict in ruc_instance_to_simulate.elements('storage') }

    # TBD - for now, we are ignoring the ReserveRequirement parameters for the economic dispatch
    # we do handle the ReserveFactor, below.
    sced_md = _get_data_dict( ruc_instance_to_simulate, sced_horizon, demand_dict, None, options.reserve_factor,\
                              min_nondispatch_dict, max_nondispatch_dict,
                              UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict)


    #######################
    # create the instance #
    #######################

    ## if relaxing initial ramping, we need to relax it in the first SCED as well
    assert options.relax_t0_ramping_initial_day is False

    ##################################################################
    # set the unit on variables in the sced instance to those values #
    # found in the stochastic ruc instance for the input time period #
    ##################################################################

    # NOTE: the values coming back from the RUC solves can obviously
    #       be fractional, due to numerical tolerances on integrality.
    #       we could enforce integrality at the solver level, but are
    #       not presently. instead, we round, to force integrality.
    #       this has the disadvantage of imposing a disconnect between
    #       the stochastic RUC solution and the SCED, but for now,
    #       we will live with it.
    #for t in sorted(sced_instance.TimePeriods):

    for g, g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
        fixed_commitment = _preallocated_list(sced_md.data['system']['time_keys'])
        g_dict['fixed_commitment'] = _time_series_dict(fixed_commitment)

    for idx,t in enumerate(sced_md.data['system']['time_keys']):
        # the input t and hour_to_simulate are both 1-based => so is the translated_t
        t = int(t)
        translated_t = t + hour_to_simulate - 1
        translated_idx = translated_t - 1

        for g,g_dict in sced_md.elements(element_type='generator', generator_type='thermal'):
            # CRITICAL: today's ruc instance and tomorrow's ruc instance are not guaranteed
            #           to be consistent, in terms of the value of the binaries in the time 
            #           periods in which they overlap, nor the projected power levels for 
            #           time units in which they overlap. originally, we were trying to be
            #           clever and using today's ruc instance for hours <= 24, and tomorrow's
            #           ruc instance for hours > 24, but we didn't think this carefully 
            #           enough through. this issue should be revisited. at the moment, we 
            #           are relying on the consistency between our projections for the unit
            #           states at midnight and the values actually observed. these should not
            #           be too disparate, given that the projection is only 3 hours out.
            # NEW: this is actually a problem 3 hours out - especially if the initial state
            #      projections involving PowerGeneratedT0 is incorrect. and these are often
            #      divergent. 
            if translated_t > ruc_every_hours and \
                tomorrow_ruc_instance is not None:
                    #new_value = int(round(value(tomorrow_ruc_instance.UnitOn[g, translated_t - ruc_every_hours])
                    new_value = tomorrow_ruc_instance.data['elements']['generator'][g]['commitment']['values'][translated_idx-ruc_every_hours]
                #else:
                #    new_value = int(round(value(today_ruc_instance.UnitOn[g, translated_t])))
            else:
                new_value = today_ruc_instance.data['elements']['generator'][g]['commitment']['values'][translated_idx]
            #sced_instance.UnitOn[g, t] = new_value
            g_dict['fixed_commitment']['values'][idx] = int(round(new_value))

    return sced_md


##### BEGIN Deterministic RUC solvers and helper functions ########
###################################################################
# utility functions for computing and reporting various aspects   #
# of a deterministic unit commitment solution.                    #
###################################################################

def _output_solution_for_deterministic_ruc(ruc_instance, 
                                          this_date,
                                          this_hour,
                                          ruc_every_hours,
                                          max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    last_time_period = min(len(ruc_instance.data['system']['time_keys']), 36)
    last_time_period_storage = min(len(ruc_instance.data['system']['time_keys']), 26)

    thermal_gens = dict(ruc_instance.elements(element_type='generator', generator_type='thermal'))

    print("Generator Commitments:")
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            print("%2d"% int(round(gdict['commitment']['values'][t])), end=' ')
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Dispatch Levels:")
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            print("%7.2f"% gdict['pg']['values'][t], end=' ')
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Reserve Headroom:")
    total_headroom = [0.0 for i in range(0, last_time_period)]  # add the 0 in for simplicity of indexing
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            headroom = gdict['headroom']['values'][t]
            print("%7.2f" % headroom, end=' ')
            total_headroom[t] += headroom
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")
    print(("%-"+str(max_thermal_generator_label_length)+"s: ") % "Total", end=' ')
    for t in range(last_time_period):
        print("%7.2f" % total_headroom[t], end=' ')
        if t+1 == ruc_every_hours: 
            print(" |", end=' ')
    print("")

    storage = dict(ruc_instance.elements(element_type='storage'))
    if len(storage) > 0:
        
        print("Storage Input levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['p_charge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")

        print("Storage Output levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['p_discharge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")

        print("Storage SOC levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['state_of_charge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")


def _report_fixed_costs_for_deterministic_ruc(deterministic_instance):

    costs = sum( sum(g_dict['commitment_cost']['values']) for _,g_dict in deterministic_instance.elements(element_type='generator', generator_type='thermal'))

    print("Fixed costs:    %12.2f" % costs)

def _report_generation_costs_for_deterministic_ruc(deterministic_instance):
    costs = sum( sum(g_dict['production_cost']['values']) for _,g_dict in deterministic_instance.elements(element_type='generator', generator_type='thermal'))

    print("Variable costs: %12.2f" % costs)
    
def _report_load_generation_mismatch_for_deterministic_ruc(ruc_instance):

    time_periods = ruc_instance.data['system']['time_keys']

    buses = ruc_instance.data['elements']['bus']

    for i,t in enumerate(time_periods):
        mismatch_reported = False
        sum_mismatch = round_small_values(sum(bdict['p_balance_violation']['values'][i]
                                              for bdict in buses.values()))
        if sum_mismatch != 0.0:
            posLoadGenerateMismatch = round_small_values(sum(max(bdict['p_balance_violation']['values'][i],0.)
                                                            for bdict in buses.values()))
            negLoadGenerateMismatch = round_small_values(sum(min(bdict['p_balance_violation']['values'][i],0.)
                                                            for bdict in buses.values()))
            if negLoadGenerateMismatch != 0.0:
                print("Projected over-generation reported at t=%s -   total=%12.2f" % (t, negLoadGenerateMismatch))
            if posLoadGenerateMismatch != 0.0:
                print("Projected load shedding reported at t=%s -     total=%12.2f" % (t, posLoadGenerateMismatch))

        if 'reserve_shortfall' in ruc_instance.data['system']:
            reserve_shortfall_value = round_small_values(ruc_instance.data['system']['reserve_shortfall']['values'][i])
            if reserve_shortfall_value != 0.0:
                print("Projected reserve shortfall reported at t=%s - total=%12.2f" % (t, reserve_shortfall_value))

def _report_curtailment_for_deterministic_ruc(deterministic_instance):
    
    rn_gens = dict(deterministic_instance.elements(element_type='generator', generator_type='renewable'))
    time_periods = deterministic_instance.data['system']['time_keys']

    curtailment_in_some_period = False
    for i,t in enumerate(time_periods):
        quantity_curtailed_this_period = sum(gdict['p_max']['values'][i] - gdict['pg']['values'][i] \
                                            for gdict in rn_gens.values())
        if quantity_curtailed_this_period > 0.0:
            if curtailment_in_some_period == False:
                print("Renewables curtailment summary (time-period, aggregate_quantity):")
                curtailment_in_some_period = True
            print("%s %12.2f" % (t, quantity_curtailed_this_period))

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


        thermal_gens = dict(ruc_instance_for_this_period.elements(element_type='generator', generator_type='thermal'))
        if len(thermal_gens) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max((len(this_generator) for this_generator in thermal_gens))
        
        total_cost = ruc_instance_for_this_period.data['system']['total_cost']
        print("")
        print("Deterministic RUC Cost: {0:.2f}".format(total_cost))
    
        if options.output_ruc_solutions:
    
            print("")
            _output_solution_for_deterministic_ruc(ruc_instance_for_this_period, 
                                                   this_date, 
                                                   this_hour, 
                                                   options.ruc_every_hours,
                                                   max_thermal_generator_label_length=max_thermal_generator_label_length,
                                                   )
    
        print("")
        _report_fixed_costs_for_deterministic_ruc(ruc_instance_for_this_period)
        _report_generation_costs_for_deterministic_ruc(ruc_instance_for_this_period)
        print("")
        _report_load_generation_mismatch_for_deterministic_ruc(ruc_instance_for_this_period)                        
        print("")
        _report_curtailment_for_deterministic_ruc(ruc_instance_for_this_period)                        
    
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
                             prior_deterministic_ruc, # UnitOn T0 state should come from here
                             projected_sced, # PowerGenerated T0 state should come from here
                             output_initial_conditions,
                             sced_schedule_hour=4,
                             ruc_horizon=48,
                             use_next_day_in_ruc=False):

    ruc_every_hours = options.ruc_every_hours

    start_day = dateutil.parser.parse(this_date).date()
    start_time = datetime.datetime.combine(start_day, datetime.time(hour=this_hour))

    # Create a new model
    md = data_provider.get_initial_model(options, ruc_horizon)

    # Populate the T0 data
    if prior_deterministic_ruc is None or projected_sced is None:
        data_provider.populate_initial_state_data(options, start_day, md)
    else:
        for s, sdict in md.elements('storage'):
            s_dict['initial_state_of_charge'] = projected_sced['elements']['storage'][s]['state_of_charge']['values'][sced_schedule_hour-1]

        for g, g_dict in prior_deterministic_ruc.elements(element_type='generator', generator_type='thermal'):
            # this generator's commitments
            g_commit = g_dict['commitment']['values']
            final_unit_on_state = int(round(g_commit[ruc_every_hours-1]))

            state_duration = 1

            hours = list(range(ruc_every_hours-1))
            hours.reverse()
            for i in hours:
                this_unit_on_state = int(round(g_commit[i]))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration
            ## get hours before prior_deterministic_ruc
            prior_UnitOnT0State = int(g_dict['initial_status'])
            # if we have the same state at the beginning and the end of the horizon,
            # AND it agrees with the state from the previous, we can add the prior state.
            # Important for doing more than daily commitments, and long horizon generators
            # like nuclear and some coal units.
            if abs(state_duration) == ruc_every_hours and \
               ((prior_UnitOnT0State < 0) == (state_duration < 0)):
                state_duration += prior_UnitOnT0State
            md.data['elements']['generator'][g]['initial_status'] = state_duration

            # power generated is the projected output at t0
            #power_generated_at_t0 = value(projected_sced.PowerGenerated[g, sced_schedule_hour])
            power_generated_at_t0 = projected_sced.data['elements']['generator'][g]['pg']['values'][sced_schedule_hour-1]

            # on occasion, the average power generated across scenarios for a single generator
            # can be a very small negative number, due to MIP tolerances allowing it. if this
            # is the case, simply threshold it to 0.0. similarly, the instance validator will
            # fail if the average power generated is small-but-positive (e.g., 1e-14) and the
            # UnitOnT0 state is Off. in the latter case, just set the average power to 0.0.
            if power_generated_at_t0 < 0.0:
                power_generated_at_t0 = 0.0
            elif final_unit_on_state == 0:
                power_generated_at_t0 = 0.0

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
            if isinstance(g_dict['p_min'], dict):
                min_power_output = g_dict['p_min']['values'][sced_schedule_hour-1]
            else:
                min_power_output = g_dict['p_min']
            if isinstance(g_dict['p_max'], dict):
                max_power_output = g_dict['p_max']['values'][sced_schedule_hour-1]
            else:
                max_power_output = g_dict['p_max']

            # Ensure power generated is within limits
            # TBD: Eventually make the 1e-5 an user-settable option.
            if math.isclose(min_power_output, power_generated_at_t0, rel_tol=0, abs_tol=1e-5):
                power_generated_at_t0 = min_power_output
            elif math.isclose(max_power_output, power_generated_at_t0, rel_tol=0, abs_tol=1e-5):
                power_generated_at_t0 = max_power_output

            md.data['elements']['generator'][g]['initial_p_output'] = power_generated_at_t0

    # Populate forecasts
    copy_first_day = (not use_next_day_in_ruc) and (this_hour != 0)
    forecast_request_count = 24 if copy_first_day else ruc_horizon 
    data_provider.populate_with_forecast_data(options, start_time, forecast_request_count, 
                                              60, md)

    # Ensure the reserve requirement is satisfied
    if options.reserve_factor > 0:
        reserve_factor = options.reserve_factor
        reserve_reqs = md.data['system']['reserve_requirement']['values']
        for t in range(0, forecast_request_count):
            total_load = sum(bdata['p_load']['values'][t]
                             for bus, bdata in md.elements('load'))
            min_reserve = reserve_factor*total_demand
            if reserve_reqs[t] < min_reserve:
                reserve_reqs[t] = min_reserve

    # Copy from first 24 to second 24, if necessary
    if copy_first_day:
        _copy_time_period_data(options, range(0,24), range(24, ruc_horizon-24), md)

    return md

def _report_initial_conditions_for_deterministic_ruc(deterministic_instance,
                                                    max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    tgens = dict(deterministic_instance.elements(element_type='generator', generator_type='thermal'))
    print("")
    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated):")
    #assert(len(deterministic_instance.PowerGeneratedT0) == len(deterministic_instance.UnitOnT0State))
    for g,gdict in tgens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s %5d %7d %12.2f" ) % 
              (g, 
               int(gdict['initial_status']>0),
               gdict['initial_status'],
               gdict['initial_p_output'],
               ))

    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g,gdict in tgens.items():
        total_t0_power_output += gdict['initial_p_output']
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g,gdict in tgens.items():
        t0_state = gdict['initial_status']
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= gdict['min_down_time']:
                if isinstance(gdict['p_max'], dict):
                    p_max = gdict['p_max']['values'][0]
                else:
                    p_max = gdict['p_max']
                total_new_online_capacity += min(gdict['startup_capacity'], p_max)
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g,gdict in tgens.items():
        t0_state = gdict['initial_status']
        if t0_state > 0: # the unit has been on
            if t0_state >= gdict['min_up_time']:
                if gdict['initial_p_output'] <= gdict['shutdown_capacity']:
                    total_new_offline_capacity += gdict['initial_p_output']
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)

def _report_demand_for_deterministic_ruc(ruc_instance,
                                         ruc_every_hours,
                                         max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    load = ruc_instance.data['elements']['load']
    times = ruc_instance.data['system']['time_keys']
    print("")
    print("Projected Demand:")
    for b, ldict in load.items():
        print(("%-"+str(max_bus_label_length)+"s: ") % b, end=' ')
        for i in range(min(len(times), 36)):
            print("%8.2f"% ldict['p_load']['values'][i], end=' ')
            if i+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")


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
        options: Options, data_provider: DataProvider, 
        this_date: string, this_hour:int) -> EgretModel:
    ''' Get an Egret model consisting of data to be treated as actuals, starting at a given time.

    Parameters
    ----------
    options:Options
        Global option values
    data_provider: DataProvider
        An object that can provide actual and/or forecast data for the requested days
    this_date: string
        A string that can be parsed as a date
    this_hour: int
        0-based index of the first hour of the day for which data should be retrieved
    ''' 
    # Convert time string to time
    start_time = dateutil.parser.parse(this_date)

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
    # TODO: Avoid hard-coding the number of values
    md = data_provider.get_initial_model(options, 48)

    # Fill it in with data
    data_provider.populate_initial_state_data(options, start_time.date(), md)
    if this_hour == 0:
        get_data_func(options, start_time, 48, 60, md)
    else:
        get_data_func(options, start_time, 24, 60, md)
        _copy_time_period_data(options, range(0,24), range(24,48), md)

    return md

def _copy_time_period_data(options, copy_from_periods, copy_to_periods, model):
    ''' Copy forecastable data between time periods
    '''
    # copy renewables limits
    for gen, gdata in model.elements('generator', generator_type='renewable'):
        for t_from, t_to in zip(copy_from_periods, copy_to_periods):
            gdata['p_min']['values'][t_to] = gdata['p_min']['values'][t_from]
            gdata['p_max']['values'][t_to] = gdata['p_max']['values'][t_from]

    # Copy loads
    for bus, bdata in model.elements('load'):
        for t_from, t_to in zip(copy_from_periods, copy_to_periods):
            bdata['p_load']['values'][t_to] = bdata['p_load']['values'][t_from]

    # Copy reserve requirement
    reserve_reqs = model.data['system']['reserve_requirement']['values']
    for t_from, t_to in zip(copy_from_periods, copy_to_periods):
        reserve_reqs[t_to] = reserve_reqs[t_from]
