#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os
import math
import logging

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

uc_abstract_data_model = get_uc_model()

########################################################################################
# a utility to find the "nearest" - quantified via Euclidean distance - scenario among #
# a candidate set relative to the input scenario, up through and including the         #
# specified simulation hour.                                                           #
########################################################################################

def call_solver(solver,instance,options,solver_options,relaxed=False):
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
                                                solver_options_dict, None, relaxed) 

    md = _save_uc_results(m, relaxed)

    if hasattr(results, 'egret_metasolver_status'):
        time = results.egret_metasolver_status['time']
    else:
        time = results.solver.wallclock_time

    return md,time


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
def create_sced_instance(today_ruc_instance, today_stochastic_scenario_instances, today_scenario_tree,
                         tomorrow_ruc_instance, tomorrow_stochastic_scenario_instances, tomorrow_scenario_tree,
                         ruc_instance_to_simulate,  # providies actuals and an instance to query
                         prior_sced_instance,  # used for initial conditions if initial_from_ruc=False
                         actual_demand,  # native Python dictionary containing the actuals
                         demand_forecast_error, 
                         actual_min_renewables,  # native Python dictionary containing the actuals
                         actual_max_renewables,  # native Python dictionary containing the actuals
                         renewables_forecast_error,
                         hour_to_simulate,
                         reserve_factor, 
                         options,
                         hours_in_objective=1,
                         # by default, just worry about cost minimization for the hour to simulate.
                         sced_horizon=24,
                         # by default simulate a SCED for 24 hours, to pass through the midnight boundary successfully
                         ruc_every_hours=24,
                         initialize_from_ruc=True,
                         use_prescient_forecast_error=True,
                         use_persistent_forecast_error=False,
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
        # to those found in the RUC (deterministic or stochastic) instance. this has 
        # the advantage that simulating from an in-sample scenario will yield
        # a feasible sced at each simulated hour.

        # if there isn't a set of stochastic scenario instances, then 
        # we're dealing with a deterministic RUC - use it directly.
        assert today_stochastic_scenario_instances is None
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

            #unit_on = int(round(value(prior_sced_instance.UnitOn[g, 1])))
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
            if math.fabs(min_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = min_power_output
            elif math.fabs(max_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = max_power_output
            else:
                PowerGeneratedT0Dict[g] = candidate_power_generated
            
            # related to the above (and this is a case that is not caught by the above),
            # if the unit is off, then the power generated at t0 must be equal to 0 -
            # no tolerances allowed.
            if unit_on == 0:
                PowerGeneratedT0Dict[g] = 0.0

    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if use_prescient_forecast_error:

        demand_dict = dict(((b, t+1), actual_demand[b, hour_to_simulate + t])
                           for b,_ in ruc_instance_to_simulate.elements(element_type='bus') for t in range(0, sced_horizon))
        min_nondispatch_dict = dict(((g, t+1), actual_min_renewables[g, hour_to_simulate + t])
                                    for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable')
                                    for t in range(0, sced_horizon))
        max_nondispatch_dict = dict(((g, t+1), actual_max_renewables[g, hour_to_simulate + t])
                                    for g,_ in ruc_instance_to_simulate.elements(element_type='generator', generator_type='renewable')
                                    for t in range(0, sced_horizon))

    else:  # use_persistent_forecast_error:

        # there is redundancy between the code for processing the two cases below. 
        # for now, leaving this alone for clarity / debug.
        assert today_stochastic_scenario_instances is None

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
        ruc_results, pyo_results = call_solver(solver,
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
                             this_date, 
                             this_hour,
                             next_date,
                             prior_deterministic_ruc, # UnitOn T0 state should come from here
                             projected_sced, # PowerGenerated T0 state should come from here
                             output_initial_conditions,
                             sced_schedule_hour=4,
                             ruc_horizon=48,
                             use_next_day_in_ruc=False,
                             max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH,
                             max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH,
                             scenario=None,
                             prior_root_node=None):

    ruc_every_hours = options.ruc_every_hours

    # the (1-based) time period in the sced corresponding to midnight

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage")

    # next, construct the RUC instance. for data, we always look in the pysp directory
    # for the forecasted value instance.
    reference_model_filename = os.path.expanduser(options.model_directory) + os.sep + "ReferenceModel.py"

    if (options.ruc_prescience_hour > abs(-options.ruc_execution_hour%options.ruc_every_hours)) and \
            (options.ruc_prescience_hour > 0) and \
            (options.run_deterministic_ruc):
        print("NOTE: Loading actuals data to blend into first hours of deterministic RUC.")
        load_actuals = True
    else:
        load_actuals = False

    if scenario is None:
        scenario_filename = "Scenario_forecasts.dat"
        actuals_filename = "Scenario_actuals.dat"
    else:
        scenario_filename = scenario+".dat"

    today_data = uc_abstract_data_model.create_instance(os.path.join(os.path.expanduser(instance_directory_name),
                                                                     this_date,
                                                                     scenario_filename)
                                                       )
    if load_actuals:
        today_actuals = uc_abstract_data_model.create_instance(os.path.join(os.path.expanduser(instance_directory_name),
                                                                            this_date,
                                                                            actuals_filename)
                                                              )
    else:
        today_actuals = None

    if (next_date is None) or ((this_hour == 0) and (not use_next_day_in_ruc)):
        next_day_data = None
        next_day_actuals = None
    else:
        next_day_data = uc_abstract_data_model.create_instance(os.path.join(os.path.expanduser(instance_directory_name),
                                                                             next_date,
                                                                             scenario_filename )
                                                              )
        if load_actuals:
            next_day_actuals = uc_abstract_data_model.create_instance(os.path.join(os.path.expanduser(instance_directory_name),
                                                                                   next_date,
                                                                                   actuals_filename)
                                                                     )
        else:
            next_day_actuals = None


    if max_thermal_generator_label_length == DEFAULT_MAX_LABEL_LENGTH:
        if len(today_data.ThermalGenerators) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max((len(this_generator) for this_generator in today_data.ThermalGenerators))

    if max_bus_label_length == DEFAULT_MAX_LABEL_LENGTH:
        max_bus_label_length = max((len(this_bus) for this_bus in today_data.Buses))

    # set up the initial conditions for the deterministic RUC, based on all information
    # available as to the conditions at the initial day. if this is the first simulation
    # day, then we have nothing to go on other than the initial conditions already
    # specified in the forecasted value instance. however, if this is not the first simulation
    # day, use the projected solution state embodied (either explicitly or implicitly)
    # in the prior date RUC to establish the initial conditions.
    if (prior_deterministic_ruc is not None) and (projected_sced is not None):
        UnitOnT0Dict = {}
        UnitOnT0StateDict = {}
        PowerGeneratedT0Dict = {}

        #for g in sorted(today_data.ThermalGenerators):
        for g, g_dict in prior_deterministic_ruc.elements(element_type='generator', generator_type='thermal'):

            # this generator's commitments
            g_commit = g_dict['commitment']['values']
            # no stochastic
            assert prior_root_node is None
            #final_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g, ruc_every_hours])))
            final_unit_on_state = int(round(g_commit[ruc_every_hours-1]))

            state_duration = 1

            #hours = list(range(1, ruc_every_hours))
            hours = list(range(ruc_every_hours-1))
            hours.reverse()
            for i in hours:
                #this_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g, i])))
                this_unit_on_state = int(round(g_commit[i]))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration
            ## get hours before prior_deterministic_ruc
            #prior_UnitOnT0State = int(value(prior_deterministic_ruc.UnitOnT0State[g]))
            prior_UnitOnT0State = int(g_dict['initial_status'])
            # if we have the same state at the beginning and the end of the horizon,
            # AND it agrees with the state from the previous, we can add the prior state.
            # Important for doing more than daily commitments, and long horizon generators
            # like nuclear and some coal units.
            if abs(state_duration) == ruc_every_hours and (\
                    ( prior_UnitOnT0State < 0 and state_duration < 0 ) or \
                    ( prior_UnitOnT0State > 0 and state_duration > 0 )
                    ):
                state_duration += prior_UnitOnT0State

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

            # propagate the initial conditions to the deterministic ruc being constructed.

            UnitOnT0Dict[g] = final_unit_on_state
            UnitOnT0StateDict[g] = state_duration

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
            if isinstance(g_dict['p_min'], dict):
                min_power_output = g_dict['p_min']['values'][sced_schedule_hour]
            else:
                min_power_output = g_dict['p_min']
            if isinstance(g_dict['p_max'], dict):
                max_power_output = g_dict['p_max']['values'][sced_schedule_hour]
            else:
                max_power_output = g_dict['p_max']

            # TBD: Eventually make the 1e-5 an user-settable option.
            if math.fabs(min_power_output - power_generated_at_t0) <= 1e-5:
                PowerGeneratedT0Dict[g] = min_power_output
            elif math.fabs(max_power_output - power_generated_at_t0) <= 1e-5:
                PowerGeneratedT0Dict[g] = max_power_output
            else:
                PowerGeneratedT0Dict[g] = power_generated_at_t0

    else:
        UnitOnT0Dict = dict((gen, today_data.UnitOnT0[gen]) for gen in sorted(today_data.ThermalGenerators))
        UnitOnT0StateDict = dict((gen, today_data.UnitOnT0State[gen]) for gen in sorted(today_data.ThermalGenerators))
        PowerGeneratedT0Dict = dict((gen, today_data.PowerGeneratedT0[gen]) for gen in sorted(today_data.ThermalGenerators))

    new_ruc_md = _get_ruc_data(options, today_data, this_hour, next_day_data, UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, projected_sced, ruc_horizon, use_next_day_in_ruc, today_actuals, next_day_actuals)

    ## if relaxing the initial ramping, set this in module
    #if (prior_deterministic_ruc is None) and options.relax_t0_ramping_initial_day:
    #    reference_model.enforce_t1_ramp_rates = False
    assert options.relax_t0_ramping_initial_day is False

    ## reset this value for all subsequent runs
    #if (prior_deterministic_ruc is None) and options.relax_t0_ramping_initial_day:
    #    reference_model.enforce_t1_ramp_rates = True

    if output_initial_conditions:
        _report_initial_conditions_for_deterministic_ruc(new_ruc_md,
                                                         max_thermal_generator_label_length=max_thermal_generator_label_length)

        _report_demand_for_deterministic_ruc(new_ruc_md,
                                             options.ruc_every_hours,
                                             max_bus_label_length=max_bus_label_length)

    return new_ruc_md, None

    
def _get_ruc_data(options,
                  today_data, this_hour, next_day_data,
                  UnitOnT0Dict=None, UnitOnT0StateDict=None, PowerGeneratedT0Dict=None,
                  projected_sced = None,
                  ruc_horizon = 48,
                  use_next_day_in_ruc=False,
                  today_actuals=None, next_day_actuals=None):

    max_ruc_horizon = ruc_horizon

    def _load_from_whole_day(day_data):
        day_data_Buses = sorted(day_data.Buses)
        day_data_AllNondispatchableGenerators = sorted(day_data.AllNondispatchableGenerators)

        ruc_horizon = min(max_ruc_horizon,value(day_data.NumTimePeriods)-this_hour)
        data_time_periods = range(this_hour+1,this_hour+ruc_horizon+1)
        
        demand_dict = dict(((b, t-this_hour), value(day_data.Demand[b,t])) for b in day_data_Buses for t in data_time_periods)
        reserve_dict = dict((t-this_hour, value(day_data.ReserveRequirement[t])) for t in data_time_periods)

        min_nondispatch_dict = dict(((n,t-this_hour), value(day_data.MinNondispatchablePower[n,t])) for n in day_data_AllNondispatchableGenerators for t in data_time_periods)
        max_nondispatch_dict = dict(((n,t-this_hour), value(day_data.MaxNondispatchablePower[n,t])) for n in day_data_AllNondispatchableGenerators for t in data_time_periods)
        return ruc_horizon, demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict

    def _load_from_partial_days(day_data, day_data_next, ruc_horizon, day_data_time_periods, day_data_next_time_periods):
        day_data_Buses = sorted(day_data.Buses)
       	day_data_AllNondispatchableGenerators = sorted(day_data.AllNondispatchableGenerators)

        day_data_next_Buses = sorted(day_data_next.Buses)
        day_data_next_AllNondispatchableGenerators = sorted(day_data_next.AllNondispatchableGenerators)

        ## add for the rest of this file's data
        demand_dict = dict(((b, t-this_hour), value(day_data.Demand[b,t])) for b in day_data_Buses for t in day_data_time_periods)
        reserve_dict = dict((t-this_hour, value(day_data.ReserveRequirement[t])) for t in day_data_time_periods)

        min_nondispatch_dict = dict(((n,t-this_hour), value(day_data.MinNondispatchablePower[n,t])) for n in day_data_AllNondispatchableGenerators for t in day_data_time_periods)
        max_nondispatch_dict = dict(((n,t-this_hour), value(day_data.MaxNondispatchablePower[n,t])) for n in day_data_AllNondispatchableGenerators for t in day_data_time_periods)
        ## get the next hours from the next day, up to 24
        demand_dict.update(dict(((b, t+24-this_hour), value(day_data_next.Demand[b,t])) for b in day_data_next_Buses for t in day_data_next_time_periods))
        reserve_dict.update(dict((t+24-this_hour, value(day_data_next.ReserveRequirement[t])) for t in day_data_next_time_periods))

        min_nondispatch_dict.update(dict(((n,t+24-this_hour), value(day_data_next.MinNondispatchablePower[n,t])) for n in day_data_next_AllNondispatchableGenerators for t in day_data_next_time_periods))
        max_nondispatch_dict.update(dict(((n,t+24-this_hour), value(day_data_next.MaxNondispatchablePower[n,t])) for n in day_data_next_AllNondispatchableGenerators for t in day_data_next_time_periods))

        return demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict

    if use_next_day_in_ruc:
        if next_day_data is None:
            ruc_horizon, demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict = _load_from_whole_day(today_data)
            if today_actuals is not None:
                _, actual_demand_dict, actual_reserve_dict, actual_min_nondispatch_dict, actual_max_nondispatch_dict = \
                        _load_from_whole_day(today_actuals)
        else:
            ruc_horizon = min(value(today_data.NumTimePeriods), max_ruc_horizon)
            today_data_time_periods = range(this_hour+1,24+1)
            next_day_data_time_periods = range(1, 1+this_hour+(ruc_horizon-24))

            demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict = \
                    _load_from_partial_days(today_data, next_day_data, ruc_horizon, today_data_time_periods, next_day_data_time_periods)
            if today_actuals is not None:
                actual_demand_dict, actual_reserve_dict, actual_min_nondispatch_dict, actual_max_nondispatch_dict = \
                    _load_from_partial_days(today_actuals, next_day_actuals, ruc_horizon, today_data_time_periods, next_day_data_time_periods)

    else:
        ## In both cases we'll just get all the data from the current
        ## day, because that covers the next 24 hours or we do not have
        ## better data
        if (next_day_data is None) or (this_hour == 0):
            ruc_horizon, demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict = _load_from_whole_day(today_data)
            if today_actuals is not None:
                _, actual_demand_dict, actual_reserve_dict, actual_min_nondispatch_dict, actual_max_nondispatch_dict = \
                        _load_from_whole_day(today_actuals)
        else:
            ## NOTE: this will get the next few hours from tomorrow, and 
            ##       the repeat that for the next 24 to mirror what the 
            ##       current populator does
            ruc_horizon = min(value(today_data.NumTimePeriods), max_ruc_horizon)
            today_data_time_periods = range(this_hour+1,24+1)
                                               ## in case ruc_hozizon != 48 
            next_day_data_time_periods = range(1, 1+this_hour)

            demand_dict, reserve_dict, min_nondispatch_dict, max_nondispatch_dict = \
                    _load_from_partial_days(today_data, next_day_data, ruc_horizon, today_data_time_periods, next_day_data_time_periods)
            ## get the next 24 hours as a copy of the first
            for t in range(1,ruc_horizon-24+1):
                for b in sorted(today_data.Buses):
                    demand_dict[b,t+24] = demand_dict[b,t]
                reserve_dict[t+24] = reserve_dict[t]
                for n in sorted(today_data.AllNondispatchableGenerators):
                    min_nondispatch_dict[n,t+24] = min_nondispatch_dict[n,t]
                    max_nondispatch_dict[n,t+24] = max_nondispatch_dict[n,t]

            if today_actuals is not None:
                actual_demand_dict, actual_reserve_dict, actual_min_nondispatch_dict, actual_max_nondispatch_dict = \
                    _load_from_partial_days(today_actuals, next_day_actuals, ruc_horizon, today_data_time_periods, next_day_data_time_periods)
                ## get the next 24 hours as a copy of the first
                for t in range(1,ruc_horizon-24+1):
                    for b in sorted(today_data.Buses):
                        actual_demand_dict[b,t+24] = actual_demand_dict[b,t]
                    actual_reserve_dict[t+24] = actual_reserve_dict[t]
                    for n in sorted(today_data.AllNondispatchableGenerators):
                        actual_min_nondispatch_dict[n,t+24] = actual_min_nondispatch_dict[n,t]
                        actual_max_nondispatch_dict[n,t+24] = actual_max_nondispatch_dict[n,t]

    if today_actuals is not None:
        print("NOTE: Computing linear interpolation using actuals data!")

        ## We make a few assumptions here.
        ## 1. The RUC is run before the SCED for a given hour, so even having
        ##    a RUC run hour of 0 doesn't give perfect information for that hour
        ## 2. Consquently, we add "1" to the distance between the RUC run and the 0
        ##    hour, so as to be consistent and somewhat conservative
        ## 3. Note that the time indexing for the dictionaries starts a "1", so 2. 
        ##    is essentially done for us already 
        ## 4. This is a simple linear interpolation with a receeding time horizon.
        ##    Something smarter should be done!

        before_ruc_hour = -(options.ruc_execution_hour%(-options.ruc_every_hours))
        ruc_prescience_hour = options.ruc_prescience_hour

        for t in range(1,ruc_prescience_hour-before_ruc_hour):
            forecast_portion = (before_ruc_hour+t)/float(ruc_prescience_hour)
            actuals_portion = (ruc_prescience_hour-(before_ruc_hour+t))/float(ruc_prescience_hour)
            for b in sorted(today_data.Buses):
                demand_dict[b,t] = forecast_portion*demand_dict[b,t] + \
                                   actuals_portion*actual_demand_dict[b,t]
            reserve_dict[t] = forecast_portion*reserve_dict[t] + actuals_portion*actual_reserve_dict[t]
            for n in sorted(today_data.AllNondispatchableGenerators):
                min_nondispatch_dict[n,t] = forecast_portion*min_nondispatch_dict[n,t] + \
                                            actuals_portion*actual_min_nondispatch_dict[n,t]

                max_nondispatch_dict[n,t] = forecast_portion*max_nondispatch_dict[n,t] + \
                                            actuals_portion*actual_max_nondispatch_dict[n,t]

    ## assume this don't matter if they're not passed in
    if UnitOnT0Dict is None:
        UnitOnT0Dict = dict((g, today_data.UnitOnT0[g]) for g in sorted(today_data.ThermalGenerators))
    if UnitOnT0StateDict is None:
        UnitOnT0StateDict = dict((g, today_data.UnitOnT0State[g]) for g in sorted(today_data.ThermalGenerators))
    if PowerGeneratedT0Dict is None:
        PowerGeneratedT0Dict = dict((g, today_data.PowerGeneratedT0[g]) for g in sorted(today_data.ThermalGenerators))

    ##########################################################################
    # construct the data dictionary for instance initialization from scratch #
    ##########################################################################

    if projected_sced is not None:
        StorageSocOnT0Dict = dict((s, value(projected_sced.SocStorage[s, sced_schedule_hour]))
                                             for s in sorted(today_data.Storage))
    else:
        StorageSocOnT0Dict = dict((s, value(today_data.StorageSocOnT0[s])) for s in sorted(today_data.Storage))

    today_data_md = ModelData(create_model_data_dict_params(today_data, keep_names=True))

    ## preprocess the data coming from *.dat files
    ruc_data = _get_data_dict(today_data_md, ruc_horizon, demand_dict, reserve_dict, options.reserve_factor, \
                              min_nondispatch_dict, max_nondispatch_dict, \
                              UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict)

    return ruc_data

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


#######################################################################
# a utility to determine the name of the file in which simulated data #
# is to be drawn, given a specific input date and run-time options.   #
#######################################################################


def compute_simulation_filename_for_date(a_date, options):

    simulated_dat_filename = ""

    if options.simulate_out_of_sample > 0:
        # assume the input directories have been set up with a Scenario_actuals.dat file in each day.
        simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date), "Scenario_actuals.dat")
    else:
        if options.run_deterministic_ruc:
            print("")
            print("***WARNING: Simulating the forecast scenario when running deterministic RUC - "
                  "time consistency across midnight boundaries is not guaranteed, and may lead to threshold events.")
            simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date),
                                                  "Scenario_forecasts.dat")
    return simulated_dat_filename

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
        pricing_results, _ = call_solver(solver,
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

def create_ruc_instance_to_simulate_next_period(options, this_date, this_hour, next_date):

    simulated_dat_filename_this = compute_simulation_filename_for_date(this_date, options)
    print("")
    print("Actual simulation data for date=" + str(this_date) + " drawn from file=" +
          simulated_dat_filename_this)

    if not os.path.exists(simulated_dat_filename_this):
        raise RuntimeError("The file " + simulated_dat_filename_this + " does not exist or cannot be read.")

    ruc_instance_to_simulate_this_period = uc_abstract_data_model.create_instance(simulated_dat_filename_this)

    if next_date is not None and this_hour != 0:
        simulated_dat_filename_next = compute_simulation_filename_for_date(next_date, options)
        print("")
        print("Actual simulation data for date=" + str(next_date) + " drawn from file=" +
              simulated_dat_filename_next)

        if not os.path.exists(simulated_dat_filename_next):
            raise RuntimeError("The file " + simulated_dat_filename_next + " does not exist or cannot be read.")

        ruc_instance_to_simulate_next_period = uc_abstract_data_model.create_instance(simulated_dat_filename_next)
    else:
        ruc_instance_to_simulate_next_period = None

    print("")
    print("Creating RUC instance to simulate")
    new_ruc_to_simulate_md = _get_ruc_data(options, ruc_instance_to_simulate_this_period, this_hour, ruc_instance_to_simulate_next_period )

    return new_ruc_to_simulate_md
