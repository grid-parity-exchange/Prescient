#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import sys
import os
import shutil
import random
import traceback
import csv
import subprocess
import math

from six import iterkeys, itervalues, iteritems
from pyomo.core import *
from pyomo.opt import *
from pyomo.pysp.ef import create_ef_instance
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory, ScenarioTree
from pyomo.pysp.phutils import find_active_objective, cull_constraints_from_instance
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
import pyutilib

from prescient.util import DEFAULT_MAX_LABEL_LENGTH
from prescient.util.math_utils import round_small_values

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

########################################################################################
# a utility to find the "nearest" - quantified via Euclidean distance - scenario among #
# a candidate set relative to the input scenario, up through and including the         #
# specified simulation hour.                                                           #
########################################################################################

def call_solver(solver,instance,**kwargs):
    return solver.solve(instance, load_solutions=False, **kwargs)


def _find_nearest_scenario(reference_scenario, candidate_scenarios, simulation_hour):

    min_distance_scenario = None
    min_distance = 0.0
    alternative_min_distance_scenarios = [] # if there is more than one at the minimum distance

    # NOTE: because the units of demand and renewables are identical (e.g., MWs) there is
    #       no immediately obvious first-order reason to weight the quantities differently.

    # look through these in sorted order, to maintain sanity when tie-breaking - 
    # we always return the first in this case.
    for candidate_scenario_name in sorted(candidate_scenarios.keys()):
        candidate_scenario = candidate_scenarios[candidate_scenario_name]

        this_distance = 0.0
        
        for t in reference_scenario.TimePeriods:
            if t <= simulation_hour:
                for b in candidate_scenario.Buses:
                    reference_demand = value(reference_scenario.Demand[b, t])
                    candidate_demand = value(candidate_scenario.Demand[b, t])
                    diff = reference_demand - candidate_demand
                    this_distance += diff * diff

                for g in candidate_scenario.AllNondispatchableGenerators:
                    reference_power = value(reference_scenario.MaxNondispatchablePower[g, t])
                    candidate_power = value(candidate_scenario.MaxNondispatchablePower[g, t])
                    diff = reference_power - candidate_power
                    this_distance += diff * diff

        this_distance = math.sqrt(this_distance)
        this_distance /= (simulation_hour * len(reference_scenario.Buses)) # normalize to per-hour, per-bus

        if min_distance_scenario is None:
            min_distance_scenario = candidate_scenario
            min_distance = this_distance
        elif this_distance < min_distance:
            min_distance_scenario = candidate_scenario
            min_distance = this_distance        
            alternative_min_distance_scenarios = []
        elif this_distance == min_distance: # eventually put a tolerance on this
            alternative_min_distance_scenarios.append(candidate_scenario)

    if len(alternative_min_distance_scenarios) > 0:
        print("")
        print("***WARNING: Multiple scenarios exist at the minimum distance="+str(min_distance)+" - additional candidates include:")
        for scenario in alternative_min_distance_scenarios:
            print(scenario.name)

    return min_distance_scenario

## utility for constructing pyomo data dictionary from the passed in parameters to use
def _get_data_dict( data_model, time_horizon, demand_dict, reserve_dict, reserve_factor, \
                    min_nondispatch_dict, max_nondispatch_dict, \
                    UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict):

    ## get some useful generators
    Buses = sorted(data_model.Buses)
    TransmissionLines = sorted(data_model.TransmissionLines)
    Interfaces = sorted(data_model.Interfaces)
    ThermalGenerators = sorted(data_model.ThermalGenerators)
    AllNondispatchableGenerators = sorted(data_model.AllNondispatchableGenerators)
    Storage = sorted(data_model.Storage)

    data = {None: { 'Buses': {None: [b for b in Buses]},
                    'StageSet': {None: ["Stage_1", "Stage_2"]},
                    'TimePeriodLength': {None: 1.0},
                    'NumTimePeriods': {None: time_horizon},
                    'CommitmentTimeInStage': {"Stage_1": list(range(1, time_horizon+1)), "Stage_2": []},
                    'GenerationTimeInStage': {"Stage_1": [], "Stage_2": list(range(1, time_horizon+1))},
                    'TransmissionLines': {None : list(TransmissionLines)},
                    'BusFrom': dict((line, data_model.BusFrom[line])
                                    for line in TransmissionLines),
                    'BusTo': dict((line, data_model.BusTo[line])
                                  for line in TransmissionLines),
                    'Impedence': dict((line, data_model.Impedence[line])
                                      for line in TransmissionLines),
                    'ThermalLimit': dict((line, data_model.ThermalLimit[line])
                                         for line in TransmissionLines),
                    'Interfaces': {None : list(Interfaces)},
                    'InterfaceLines': dict((interface, [line for line in data_model.InterfaceLines[interface]])
                                             for interface in Interfaces),
                    'InterfaceFromLimit': dict((interface, data_model.InterfaceFromLimit[interface])
                                             for interface in Interfaces),
                    'InterfaceToLimit': dict((interface, data_model.InterfaceToLimit[interface])
                                             for interface in Interfaces),
                    'ThermalGenerators': {None: [gen for gen in data_model.ThermalGenerators]},
                    'ThermalGeneratorType': dict((gen, data_model.ThermalGeneratorType[gen])
                                                 for gen in ThermalGenerators),
                    'ThermalGeneratorsAtBus': dict((b, [gen for gen in sorted(data_model.ThermalGeneratorsAtBus[b])])
                                                   for b in Buses),
                    'QuickStart': dict((g, value(data_model.QuickStart[g])) for g in ThermalGenerators),
                    'QuickStartGenerators': {None: [g for g in sorted(data_model.QuickStartGenerators)]},
                    'AllNondispatchableGenerators': {None: [g for g in AllNondispatchableGenerators]},
                    'NondispatchableGeneratorType': dict((gen, data_model.NondispatchableGeneratorType[gen])
                                                         for gen in AllNondispatchableGenerators),
                    'MustRunGenerators': {None: [g for g in sorted(data_model.MustRunGenerators)]},
                    'NondispatchableGeneratorsAtBus': dict((b, [gen for gen in sorted(data_model.NondispatchableGeneratorsAtBus[b])])
                                                           for b in Buses),
                    'Demand': demand_dict,
                    'ReserveRequirement': reserve_dict,
                    'MinimumPowerOutput': dict((g, value(data_model.MinimumPowerOutput[g]))
                                               for g in ThermalGenerators),
                    'MaximumPowerOutput': dict((g, value(data_model.MaximumPowerOutput[g]))
                                               for g in ThermalGenerators),
                    'MinNondispatchablePower': min_nondispatch_dict,
                    'MaxNondispatchablePower': max_nondispatch_dict,
                    'NominalRampUpLimit': dict((g, value(data_model.NominalRampUpLimit[g]))
                                               for g in ThermalGenerators),
                    'NominalRampDownLimit': dict((g, value(data_model.NominalRampDownLimit[g]))
                                                 for g in ThermalGenerators),
                    'StartupRampLimit': dict((g, value(data_model.StartupRampLimit[g]))
                                             for g in ThermalGenerators),
                    'ShutdownRampLimit': dict((g, value(data_model.ShutdownRampLimit[g]))
                                              for g in ThermalGenerators),
                    'MinimumUpTime': dict((g, value(data_model.MinimumUpTime[g]))
                                          for g in ThermalGenerators),
                    'MinimumDownTime': dict((g, value(data_model.MinimumDownTime[g]))
                                            for g in ThermalGenerators),
                    'UnitOnT0': UnitOnT0Dict,
                    'UnitOnT0State': UnitOnT0StateDict,
                    'PowerGeneratedT0': PowerGeneratedT0Dict,
                    'ProductionCostA0': dict((g, value(data_model.ProductionCostA0[g]))
                                             for g in ThermalGenerators),
                    'ProductionCostA1': dict((g, value(data_model.ProductionCostA1[g]))
                                             for g in ThermalGenerators),
                    'ProductionCostA2': dict((g, value(data_model.ProductionCostA2[g]))
                                             for g in ThermalGenerators),
                    'CostPiecewisePoints': dict((g, [point for point in data_model.CostPiecewisePoints[g]])
                                                for g in ThermalGenerators),
                    'CostPiecewiseValues': dict((g, [value for value in data_model.CostPiecewiseValues[g]])
                                                for g in ThermalGenerators),
                    'FuelCost': dict((g, value(data_model.FuelCost[g]))
                                     for g in ThermalGenerators),
                    'NumGeneratorCostCurvePieces': {None:value(data_model.NumGeneratorCostCurvePieces)},
                    'StartupLags': dict((g, [point for point in data_model.StartupLags[g]])
                                        for g in ThermalGenerators),
                    'StartupCosts': dict((g, [point for point in data_model.StartupCosts[g]])
                                         for g in ThermalGenerators),
                    'ShutdownFixedCost': dict((g, value(data_model.ShutdownFixedCost[g]))
                                              for g in ThermalGenerators),
                    'Storage': {None: [s for s in Storage]},
                    'StorageAtBus': dict((b, [s for s in sorted(data_model.StorageAtBus[b])])
                                         for b in Buses),
                    'MinimumPowerOutputStorage': dict((s, value(data_model.MinimumPowerOutputStorage[s]))
                                                      for s in Storage),
                    'MaximumPowerOutputStorage': dict((s, value(data_model.MaximumPowerOutputStorage[s]))
                                                      for s in Storage),
                    'MinimumPowerInputStorage': dict((s, value(data_model.MinimumPowerInputStorage[s]))
                                                     for s in Storage),
                    'MaximumPowerInputStorage': dict((s, value(data_model.MaximumPowerInputStorage[s]))
                                                     for s in Storage),
                    'NominalRampUpLimitStorageOutput': dict((s, value(data_model.NominalRampUpLimitStorageOutput[s]))
                                                            for s in Storage),
                    'NominalRampDownLimitStorageOutput': dict((s, value(data_model.NominalRampDownLimitStorageOutput[s]))
                                                              for s in Storage),
                    'NominalRampUpLimitStorageInput': dict((s, value(data_model.NominalRampUpLimitStorageInput[s]))
                                                           for s in Storage),
                    'NominalRampDownLimitStorageInput': dict((s, value(data_model.NominalRampDownLimitStorageInput[s]))
                                                             for s in Storage),
                    'MaximumEnergyStorage': dict((s, value(data_model.MaximumEnergyStorage[s]))
                                                 for s in Storage),
                    'MinimumSocStorage': dict((s, value(data_model.MinimumSocStorage[s]))
                                              for s in Storage),
                    'InputEfficiencyEnergy': dict((s, value(data_model.InputEfficiencyEnergy[s]))
                                                  for s in Storage),
                    'OutputEfficiencyEnergy': dict((s, value(data_model.OutputEfficiencyEnergy[s]))
                                                   for s in Storage),
                    'RetentionRate': dict((s, value(data_model.RetentionRate[s]))
                                          for s in Storage),
                    'EndPointSocStorage': dict((s, value(data_model.EndPointSocStorage[s]))
                                               for s in Storage),
                    'StoragePowerOutputOnT0': dict((s, value(data_model.StoragePowerOutputOnT0[s]))
                                                   for s in Storage),
                    'StoragePowerInputOnT0': dict((s, value(data_model.StoragePowerInputOnT0[s]))
                                                  for s in Storage),
                    'StorageSocOnT0': StorageSocOnT0Dict,
                    'LoadMismatchPenalty': {None: value(data_model.LoadMismatchPenalty)},
                    'ReserveShortfallPenalty': {None: value(data_model.ReserveShortfallPenalty)}
                  }
           }
    if reserve_factor > 0.0:
        data[None]["ReserveFactor"] = {None: reserve_factor}

    return data

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
def create_sced_instance(sced_model, reference_model_module, 
                         today_ruc_instance, today_stochastic_scenario_instances, today_scenario_tree,
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

    assert sced_model != None
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
        if today_stochastic_scenario_instances == None:
            print("")
            print("Drawing initial conditions from deterministic RUC initial conditions")
            for g in sorted(today_ruc_instance.ThermalGenerators):
                UnitOnT0Dict[g] = value(today_ruc_instance.UnitOnT0[g])
                UnitOnT0StateDict[g] = value(today_ruc_instance.UnitOnT0State[g])
                PowerGeneratedT0Dict[g] = value(today_ruc_instance.PowerGeneratedT0[g])
        else:
            print("")
            print("Drawing initial conditions from stochastic RUC initial conditions")
            arbitrary_scenario_instance = today_stochastic_scenario_instances[list(today_stochastic_scenario_instances.keys())[0]]
            for g in sorted(ruc_instance_to_simulate.ThermalGenerators):
                UnitOnT0Dict[g] = value(arbitrary_scenario_instance.UnitOnT0[g])
                UnitOnT0StateDict[g] = value(arbitrary_scenario_instance.UnitOnT0State[g])
                PowerGeneratedT0Dict[g] = value(arbitrary_scenario_instance.PowerGeneratedT0[g])

    else:
        # TBD: Clean code below up - if we get this far, shouldn't we always have a prior sched instance?
        print("")
        print("Drawing initial conditions from prior SCED solution, if available")
        today_root_node = today_scenario_tree.findRootNode()
        for g in sorted(ruc_instance_to_simulate.ThermalGenerators):

            if prior_sced_instance is None:
                # if there is no prior sced instance, then 
                # let the T0 state be equal to the unit state
                # in the initial time period of the stochastic RUC.
                unit_on = int(round(today_root_node.get_variable_value("UnitOn", (g, "Stage_1", hour_to_simulate))))
                UnitOnT0Dict[g] = unit_on
            else:
                unit_on = int(round(value(prior_sced_instance.UnitOn[g, 1])))
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

            if prior_sced_instance is None:
                # TBD - this is the problem - we need to not compute the expected power generated, but rather
                #       the actual T0 level specified in the stochastic RUC (same for all scenarios) - this
                #       really should be option-driven, so we can retain the older variant.
                total_power_generated = 0.0
                for instance in today_stochastic_scenario_instances.values():
                    total_power_generated += value(instance.PowerGenerated[g, 1])
                PowerGeneratedT0Dict[g] = total_power_generated / float(len(today_stochastic_scenario_instances))
            else:
                PowerGeneratedT0Dict[g] = value(prior_sced_instance.PowerGenerated[g, 1])

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization 
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
            min_power_output = value(ruc_instance_to_simulate.MinimumPowerOutput[g])
            max_power_output = value(ruc_instance_to_simulate.MaximumPowerOutput[g])

            candidate_power_generated = PowerGeneratedT0Dict[g]
                
            # TBD: Eventually make the 1e-5 an user-settable option.
            if math.fabs(min_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = min_power_output
            elif math.fabs(max_power_output - candidate_power_generated) <= 1e-5: 
                PowerGeneratedT0Dict[g] = max_power_output
            
            # related to the above (and this is a case that is not caught by the above),
            # if the unit is off, then the power generated at t0 must be equal to 0 -
            # no tolerances allowed.
            if unit_on == 0:
                PowerGeneratedT0Dict[g] = 0.0

    ################################################################################
    # initialize the demand and renewables data, based on the forecast error model #
    ################################################################################

    if use_prescient_forecast_error:

        demand_dict = dict(((b, t+1), value(actual_demand[b, hour_to_simulate + t]))
                           for b in sorted(ruc_instance_to_simulate.Buses) for t in range(0, sced_horizon))
        min_nondispatch_dict = dict(((g, t+1), value(actual_min_renewables[g, hour_to_simulate + t]))
                                    for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators)
                                    for t in range(0, sced_horizon))
        max_nondispatch_dict = dict(((g, t+1), value(actual_max_renewables[g, hour_to_simulate + t]))
                                    for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators)
                                    for t in range(0, sced_horizon))

    else:  # use_persistent_forecast_error:

        # there is redundancy between the code for processing the two cases below. 
        # for now, leaving this alone for clarity / debug.

        demand_dict = {}
        min_nondispatch_dict = {}
        max_nondispatch_dict = {}

        if today_stochastic_scenario_instances == None:  # we're running in deterministic mode

            # the current hour is necessarily (by definition) the actual.
            for b in sorted(ruc_instance_to_simulate.Buses):
                demand_dict[(b,1)] = value(actual_demand[b, hour_to_simulate])

            # for each subsequent hour, apply a simple persistence forecast error model to account for deviations.
            for b in sorted(ruc_instance_to_simulate.Buses):
                forecast_error_now = demand_forecast_error[(b, hour_to_simulate)]
                actual_now = value(actual_demand[b, hour_to_simulate])
                forecast_now = actual_now + forecast_error_now

                for t in range(1, sced_horizon):
                    # IMPT: forecast errors (and therefore forecasts) are relative to actual demand, 
                    #       which is the only reason that the latter appears below - to reconstruct
                    #       the forecast. thus, no presicence is involved.
                    forecast_error_later = demand_forecast_error[(b,hour_to_simulate + t)]
                    actual_later = value(actual_demand[b,hour_to_simulate + t])
                    forecast_later = actual_later + forecast_error_later
                    # 0 demand can happen, in some odd circumstances (not at the ISO level!).
                    if forecast_now != 0.0:
                        demand_dict[(b, t+1)] = (forecast_later/forecast_now)*actual_now
                    else:
                        demand_dict[(b, t+1)] = 0.0

            # repeat the above for renewables.
            for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators):
                min_nondispatch_dict[(g, 1)] = value(actual_min_renewables[g, hour_to_simulate])
                max_nondispatch_dict[(g, 1)] = value(actual_max_renewables[g, hour_to_simulate])
                
            for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators):
                forecast_error_now = renewables_forecast_error[(g, hour_to_simulate)]
                actual_now = value(actual_max_renewables[g, hour_to_simulate])
                forecast_now = actual_now + forecast_error_now

                for t in range(1, sced_horizon):
                    # forecast errors are with respect to the maximum - that is the actual maximum power available.
                    forecast_error_later = renewables_forecast_error[(g, hour_to_simulate + t)]
                    actual_later = value(actual_max_renewables[g, hour_to_simulate + t])
                    forecast_later = actual_later + forecast_error_later

                    if forecast_now != 0.0:
                        max_nondispatch_dict[(g, t+1)] = (forecast_later/forecast_now)*actual_now
                    else:
                        max_nondispatch_dict[(g, t+1)] = 0.0
                    # TBD - fix this - it should be non-zero!
                    min_nondispatch_dict[(g, t+1)] = 0.0

        else:  # we're running in stochastic mode

            # find the nearest scenario instance, from the current day's first time period through now.
            # this scenario will be used to extract the (point) forecast quantities for demand and renewables. 
            # this process can be viewed as identifying a dynamic forecast.

            nearest_scenario_instance = _find_nearest_scenario(ruc_instance_to_simulate, today_stochastic_scenario_instances, hour_to_simulate)
            print("Nearest scenario to observations for purposes of persistence-based forecast adjustment=" + nearest_scenario_instance.name)

            # the current hour is necessarily (by definition) the actual.
            for b in sorted(ruc_instance_to_simulate.Buses):
                demand_dict[(b, 1)] = value(actual_demand[b, hour_to_simulate])

            # for each subsequent hour, apply a simple persistence forecast error model to account for deviations.
            for b in sorted(ruc_instance_to_simulate.Buses):
                actual_now = value(actual_demand[b, hour_to_simulate])
                forecast_now = value(nearest_scenario_instance.Demand[b, hour_to_simulate])

                for t in range(1, sced_horizon):
                    # the forecast later is simply the value projected by the nearest scenario.
                    forecast_later = value(nearest_scenario_instance.Demand[b, hour_to_simulate+t])
                    demand_dict[(b, t+1)] = (forecast_later/forecast_now) * actual_now

            # repeat the above for renewables.
            for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators):
                min_nondispatch_dict[(g, 1)] = value(actual_min_renewables[g, hour_to_simulate])
                max_nondispatch_dict[(g, 1)] = value(actual_max_renewables[g, hour_to_simulate])
                
            for g in sorted(ruc_instance_to_simulate.AllNondispatchableGenerators):
                actual_now = value(actual_max_renewables[g, hour_to_simulate])
                forecast_now = value(nearest_scenario_instance.MaxNondispatchablePower[g, hour_to_simulate])

                for t in range(1, sced_horizon):
                    forecast_later = value(nearest_scenario_instance.MaxNondispatchablePower[g, hour_to_simulate+t])

                    if forecast_now != 0.0:
                        max_nondispatch_dict[(g, t+1)] = (forecast_later/forecast_now) * actual_now
                    else:
                        max_nondispatch_dict[(g, t+1)] = 0.0
                    # TBD - fix this - it should be non-zero!
                    min_nondispatch_dict[(g, t+1)] = 0.0

    ##########################################################################
    # construct the data dictionary for instance initialization from scratch #
    ##########################################################################

    if prior_sced_instance!=None:
        StorageSocOnT0Dict=dict((s, value(prior_sced_instance.SocStorage[s, 1]))
                                             for s in sorted(ruc_instance_to_simulate.Storage))
    else:
        StorageSocOnT0Dict=dict((s, value(ruc_instance_to_simulate.StorageSocOnT0[s])) for s in sorted(ruc_instance_to_simulate.Storage))

    # TBD - for now, we are ignoring the ReserveRequirement parameters for the economic dispatch
    # we do handle the ReserveFactor, below.
    ed_data = _get_data_dict( ruc_instance_to_simulate, sced_horizon, demand_dict, None, options.reserve_factor,\
                              min_nondispatch_dict, max_nondispatch_dict,
                              UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict)


    #######################
    # create the instance #
    #######################

    ## if relaxing initial ramping, we need to relax it in the first SCED as well
    if (prior_sced_instance is None) and options.relax_t0_ramping_initial_day:
        sced_model.enforce_t1_ramp_rates = False

    sced_instance = sced_model.create_instance(data=ed_data)

    ## reset this value for all subsequent runs
    if (prior_sced_instance is None) and options.relax_t0_ramping_initial_day:
        sced_model.enforce_t1_ramp_rates = True

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
    for t in sorted(sced_instance.TimePeriods):
        # the input t and hour_to_simulate are both 1-based => so is the translated_t
        translated_t = t + hour_to_simulate - 1
#        print "T=",t
#        print "TRANSLATED T=",translated_t

        for g in sorted(sced_instance.ThermalGenerators):
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
            if translated_t > ruc_every_hours:
                if tomorrow_scenario_tree != None:
                    if tomorrow_stochastic_scenario_instances != None:
                        new_value = int(round(tomorrow_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t - ruc_every_hours))))
                    else:
                        new_value = int(round(value(tomorrow_ruc_instance.UnitOn[g, translated_t - ruc_every_hours])))
#                    print "T=",t,"G=",g," TAKING UNIT ON FROM TOMORROW RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t - 24
                else:
                    if today_stochastic_scenario_instances != None:
                        new_value = int(round(today_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t))))
                    else:
                        new_value = int(round(value(today_ruc_instance.UnitOn[g, translated_t])))
#                    print "T=",t,"G=",g," TAKING UNIT ON FROM TODAY RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t
            else:
                if today_stochastic_scenario_instances != None:
                    new_value = int(round(today_scenario_tree.findRootNode().get_variable_value("UnitOn", (g, translated_t))))
                else:
                    new_value = int(round(value(today_ruc_instance.UnitOn[g, translated_t])))
#                print "T=",t,"G=",g," TAKING UNIT ON FROM TODAY RUC - VALUE=",new_value,"HOUR TO TAKE=",translated_t
            sced_instance.UnitOn[g, t] = new_value


    # before fixing all of the UnitOn variables, make sure they
    # have legimitate values - otherwise, an unintelligible 
    # error from preprocessing will result.

    # all models should have UnitOn variables. some models have
    # other binaries, e.g., UnitStart and UnitStop, but presolve
    # (at least CPLEX and Gurobi's) should be able to eliminate 
    # those easily enough.
    for var_data in itervalues(sced_instance.UnitOn):
        if value(var_data) is None:
            raise RuntimeError("The index=" + str(var_data.index()) +
                               " of the UnitOn variable in the SCED instance is None - "
                               "fixing and subsequent preprocessing will fail")
        var_data.fix()

    # establish the objective function for the hour to simulate - which is simply to 
    # minimize production costs during this time period. no fixed costs to worry about.
    # however, we do need to penalize load shedding across all time periods - otherwise,
    # very bad things happen.
    objective = find_active_objective(sced_instance)    
    expr = sum(sced_instance.ProductionCost[g, i]
               for g in sced_instance.ThermalGenerators
               for i in range(1,hours_in_objective+1)) \
           + (sced_instance.LoadMismatchPenalty *
              sum(sced_instance.posLoadGenerateMismatch[b, t] + sced_instance.negLoadGenerateMismatch[b, t]
                  for b in sced_instance.Buses for t in sced_instance.TimePeriods)) + \
           (sced_instance.ReserveShortfallPenalty * sum(sced_instance.ReserveShortfall[t]
                                                        for t in sced_instance.TimePeriods))

    objective.expr = expr

    reference_model_module.reconstruct_instance_for_pricing(sced_instance, preprocess=True)

    # preprocess after all the mucking around.
    sced_instance.preprocess()

    return sced_instance

#
# a utility to inflate the ramp rates for those units in the input instance with violations.
# re-preprocesses the instance as necessary, so it is ready to solve. the input
# factor must be >= 0 and < 1. The new limits are obtained via the scale factor 1 + scale_factor.
# NOTE: We could and probably should just relax the limits by the degree to which they are 
#       violated - the present situation is more out of legacy, and the fact that we would have
#       to restructure what is returned in the reporting / computation method above.
#

def relax_sced_ramp_rates(sced_instance, scale_factor): 
    #                          nominal_up_gens_violated, nominal_down_gens_violated,
    #                          startup_gens_violated, shutdown_gens_violated):

    #    for g in nominal_up_gens_violated:
    #        sced_instance.NominalRampUpLimit[g] = value(sced_instance.NominalRampUpLimit[g]) * (1.0 + scale_factor)

    #    for g in nominal_down_gens_violated:
    #        sced_instance.NominalRampDownLimit[g] = value(sced_instance.NominalRampDownLimit[g]) * (1.0 + scale_factor)

    #    for g in startup_gens_violated:
    #        sced_instance.StartupRampLimit[g] = value(sced_instance.StartupRampLimit[g]) * (1.0 + scale_factor)

    #    for g in shutdown_gens_violated:
    #        sced_instance.ShutdownRampLimit[g] = value(sced_instance.ShutdownRampLimit[g]) * (1.0 + scale_factor)

    # COMMENT ON THE FACT THAT RAMP-DOWN AND SHUT-DOWN ARE THE ONLY ISSUES
    for g in sced_instance.ThermalGenerators:
        # sced_instance.NominalRampUpLimit[g] = value(sced_instance.NominalRampUpLimit[g]) * (1.0 + scale_factor)
        sced_instance.NominalRampDownLimit[g] = value(sced_instance.NominalRampDownLimit[g]) * (1.0 + scale_factor)        
        # sced_instance.StartupRampLimit[g] = value(sced_instance.StartupRampLimit[g]) * (1.0 + scale_factor)
        sced_instance.ShutdownRampLimit[g] = value(sced_instance.ShutdownRampLimit[g]) * (1.0 + scale_factor)

        # doing more work than strictly necessary here, but SCED isn't a bottleneck.
    # sced_instance.ScaledNominalRampUpLimit.reconstruct()
    sced_instance.ScaledNominalRampDownLimit.reconstruct()
    # sced_instance.ScaledStartupRampLimit.reconstruct()
    sced_instance.ScaledShutdownRampLimit.reconstruct()

    # sced_instance.EnforceMaxAvailableRampUpRates.reconstruct()
    # sced_instance.EnforceMaxAvailableRampDownRates.reconstruct()
    sced_instance.EnforceScaledNominalRampDownLimits.reconstruct()
    
    sced_instance.preprocess()


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

    last_time_period = min(value(ruc_instance.NumTimePeriods)+1, 37)
    last_time_period_storage = min(value(ruc_instance.NumTimePeriods)+1, 27)
    print("Generator Commitments:")
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, last_time_period):
            print("%2d"% int(round(value(ruc_instance.UnitOn[g, t]))), end=' ')
            if t == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Dispatch Levels:")
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, last_time_period):
            print("%7.2f"% value(ruc_instance.PowerGenerated[g,t]), end=' ')
            if t == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Reserve Headroom:")
    total_headroom = [0.0 for i in range(0, last_time_period)]  # add the 0 in for simplicity of indexing
    for g in sorted(ruc_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(1, last_time_period):
            headroom = math.fabs(value(ruc_instance.MaximumPowerAvailable[g, t]) -
                                 value(ruc_instance.PowerGenerated[g, t]))
            print("%7.2f" % headroom, end=' ')
            total_headroom[t] += headroom
            if t == ruc_every_hours: 
                print(" |", end=' ')
        print("")
    print(("%-"+str(max_thermal_generator_label_length)+"s: ") % "Total", end=' ')
    for t in range(1, last_time_period):
        print("%7.2f" % total_headroom[t], end=' ')
    print("")

    if len(ruc_instance.Storage) > 0:
        
        directory = "./deterministic_simple_storage"
        if not os.path.exists(directory):
            os.makedirs(directory)
            csv_hourly_output_filename = os.path.join(directory, "Storage_summary_for_" + str(this_date) + ".csv")
        else:
            csv_hourly_output_filename = os.path.join(directory, "Storage_summary_for_" + str(this_date) + ".csv")

        csv_hourly_output_file = open(csv_hourly_output_filename, "w")
        
        print("Storage Input levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage Input:", file=csv_hourly_output_file)
            for t in range(1, last_time_period_storage):
                print("%7.2f"% value(ruc_instance.PowerInputStorage[s,t]), end=' ')
                print("%7.2f"% value(ruc_instance.PowerInputStorage[s,t]), end=' ', file=csv_hourly_output_file)
                if t == ruc_every_hours: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")

        print("Storage Output levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage Output:", file=csv_hourly_output_file)
            for t in range(1,last_time_period_storage):
                print("%7.2f"% value(ruc_instance.PowerOutputStorage[s,t]), end=' ', file=csv_hourly_output_file)
                print("%7.2f"% value(ruc_instance.PowerOutputStorage[s,t]), end=' ')
                if t == ruc_every_hours: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")

        print("Storage SOC levels")
        for s in sorted(ruc_instance.Storage):
            print("%30s: " % s, end=' ')
            print(s, end=' ', file=csv_hourly_output_file)
            print("Storage SOC:", file=csv_hourly_output_file)
            for t in range(1,last_time_period_storage):
                print("%7.2f"% value(ruc_instance.SocStorage[s,t]), end=' ', file=csv_hourly_output_file)
                print("%7.2f"% value(ruc_instance.SocStorage[s,t]), end=' ')
                if t == ruc_every_hours: 
                    print(" |", end=' ')
                    print(" |", end=' ', file=csv_hourly_output_file)
            print("", file=csv_hourly_output_file)
            print("")


def _report_fixed_costs_for_deterministic_ruc(deterministic_instance):

    second_stage = "Stage_1" # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Fixed costs:    %12.2f" % value(deterministic_instance.CommitmentStageCost[second_stage]))

def _report_generation_costs_for_deterministic_ruc(deterministic_instance):

    # only worry about two-stage models for now..
    second_stage = "Stage_2" # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Variable costs: %12.2f" % value(deterministic_instance.GenerationStageCost[second_stage]))
    
def _report_load_generation_mismatch_for_deterministic_ruc(ruc_instance):

    for t in sorted(ruc_instance.TimePeriods):
        mismatch_reported = False
        sum_mismatch = round_small_values(sum(value(ruc_instance.LoadGenerateMismatch[b, t])
                                              for b in ruc_instance.Buses))
        if sum_mismatch != 0.0:
            posLoadGenerateMismatch = round_small_values(sum(value(ruc_instance.posLoadGenerateMismatch[b, t])
                                                             for b in ruc_instance.Buses))
            negLoadGenerateMismatch = round_small_values(sum(value(ruc_instance.negLoadGenerateMismatch[b, t])
                                                             for b in ruc_instance.Buses))
            if negLoadGenerateMismatch != 0.0:
                print("Projected over-generation reported at t=%d -   total=%12.2f" % (t, negLoadGenerateMismatch))
                mismatch_reported = True
            if posLoadGenerateMismatch != 0.0:
                print("Projected load shedding reported at t=%d -     total=%12.2f" % (t, posLoadGenerateMismatch))
                mismatch_reported = True

        reserve_shortfall_value = round_small_values(value(ruc_instance.ReserveShortfall[t]))
        if reserve_shortfall_value != 0.0:
            print("Projected reserve shortfall reported at t=%d - total=%12.2f" % (t, reserve_shortfall_value))
            mismatch_reported = True

        if mismatch_reported:

            print("")
            print("Dispatch detail for time period=%d" % t)
            total_generated = 0.0
            for g in sorted(ruc_instance.ThermalGenerators):
                unit_on = int(round(value(ruc_instance.UnitOn[g, t])))
                print("%-30s %2d %12.2f %12.2f" % (g, 
                                                   unit_on, 
                                                   value(ruc_instance.PowerGenerated[g,t]),
                                                   value(ruc_instance.MaximumPowerAvailable[g,t]) - value(ruc_instance.PowerGenerated[g,t])), end=' ')
                if (unit_on == 1) and (math.fabs(value(ruc_instance.PowerGenerated[g,t]) -
                                                 value(ruc_instance.MaximumPowerOutput[g])) <= 1e-5): 
                    print(" << At max output", end=' ')
                elif (unit_on == 1) and (math.fabs(value(ruc_instance.PowerGenerated[g,t]) -
                                                   value(ruc_instance.MinimumPowerOutput[g])) <= 1e-5): 
                    print(" << At min output", end=' ')
                if value(ruc_instance.MustRun[g]):
                    print(" ***", end=' ')
                print("")
                if unit_on == 1:
                    total_generated += value(ruc_instance.PowerGenerated[g,t])
            print("")
            print("Total power dispatched=%7.2f" % total_generated)

def _report_curtailment_for_deterministic_ruc(deterministic_instance):

    curtailment_in_some_period = False
    for t in deterministic_instance.TimePeriods:
        quantity_curtailed_this_period = sum(value(deterministic_instance.MaxNondispatchablePower[g,t]) - \
                                    value(deterministic_instance.NondispatchablePowerUsed[g,t]) \
                                    for g in deterministic_instance.AllNondispatchableGenerators)
        if quantity_curtailed_this_period > 0.0:
            if curtailment_in_some_period == False:
                print("Renewables curtailment summary (time-period, aggregate_quantity):")
                curtailment_in_some_period = True
            print("%2d %12.2f" % (t, quantity_curtailed_this_period))

## NOTE: in closure for deterministic_ruc_solver_plugin
def create_create_and_solve_deterministic_ruc(deterministic_ruc_solver):
    def create_and_solve_deterministic_ruc(solver, solve_options, options,
                                           this_date,
                                           this_hour,
                                           next_date,
                                           last_ruc_instance,
                                           last_ruc_scenario_tree,
                                           output_initial_conditions,
                                           projected_sced_instance,
                                           sced_schedule_hour,
                                           ruc_horizon,
                                           use_next_day_in_ruc):
        print("")
        print("Creating and solving deterministic RUC instance for date:", this_date, " start hour:", this_hour)
    
        ruc_instance_for_this_period, scenario_tree_for_this_period = _create_deterministic_ruc(options,
                                                                                                this_date,
                                                                                                this_hour,
                                                                                                next_date,
                                                                                                last_ruc_instance,
                                                                                                projected_sced_instance,
                                                                                                output_initial_conditions,
                                                                                                sced_schedule_hour,
                                                                                                ruc_horizon,
                                                                                                use_next_day_in_ruc,
                                                                                                )
        deterministic_ruc_solver(ruc_instance_for_this_period, solver, options, solve_options)

        if options.write_deterministic_ruc_instances:
            lp_writer = ProblemWriter_cpxlp()
            current_ruc_filename = options.output_directory + os.sep + str(this_date) + \
                                                    os.sep + "ruc_hour_" + str(this_hour) + ".lp"
            lp_writer(ruc_instance_for_this_period, current_ruc_filename, lambda x: True,
                      {"symbolic_solver_labels" : options.symbolic_solver_labels})
            print("RUC instance written to file=" + current_ruc_filename)


        if len(ruc_instance_for_this_period.ThermalGenerators) == 0:
            max_thermal_generator_label_length = None
        else:
            max_thermal_generator_label_length = max((len(this_generator) for this_generator in ruc_instance_for_this_period.ThermalGenerators))
        
        total_cost = ruc_instance_for_this_period.TotalCostObjective()
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
    
        return ruc_instance_for_this_period, scenario_tree_for_this_period
    return create_and_solve_deterministic_ruc

def _solve_deterministic_ruc(deterministic_ruc_instance,
                            solver, 
                            options, 
                            solve_options):


    results = call_solver(solver,
                          deterministic_ruc_instance, 
                          tee=options.output_solver_logs,
                          **solve_options[solver])

    if results.solver.termination_condition not in safe_termination_conditions:
        print("Failed to solve deterministic RUC instance - likely because no feasible solution exists!")        
        print("Solution status:", results.solution.status.key)
        print("Solver termination condition:", results.solver.termination_condition)
        output_filename = "bad_ruc.lp"
        print("Writing failed RUC model to file=" + output_filename)
        lp_writer = ProblemWriter_cpxlp()            
        lp_writer(deterministic_ruc_instance, output_filename, 
                  lambda x: True, {"symbolic_solver_labels" : True})

        if options.error_if_infeasible:
            raise RuntimeError("Halting due to infeasibility")

    deterministic_ruc_instance.solutions.load_from(results)

## create this function with default solver
create_and_solve_deterministic_ruc = create_create_and_solve_deterministic_ruc(_solve_deterministic_ruc)

# utilities for creating a deterministic RUC instance, and a standard way to solve them.
def _create_deterministic_ruc(options, 
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

    # first, create the scenario tree - this is still needed, for various reasons, in the
    # master simulation routine.

    # IMPORTANT: "scenario_tree_model" is a *function* that returns an abstract model - the
    #            naming convention does not necessarily reflect this fact...
    from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
    scenario_tree_instance_filename = os.path.join(os.path.expanduser(instance_directory_name),
                                                   this_date,
                                                   "ScenarioStructure.dat")

    # ignoring bundles for now - for the deterministic case, we don't care.
    scenario_tree_instance = CreateAbstractScenarioTreeModel().create_instance(filename=scenario_tree_instance_filename)

    new_scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_instance)

    # next, construct the RUC instance. for data, we always look in the pysp directory
    # for the forecasted value instance.
    reference_model_filename = os.path.expanduser(options.model_directory) + os.sep + "ReferenceModel.py"
    reference_model_module = pyutilib.misc.import_file(reference_model_filename)

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

    today_data = reference_model_module.load_model_parameters().create_instance(
                                                                os.path.join(os.path.expanduser(instance_directory_name),
                                                                this_date,
                                                                scenario_filename)
                                                                )
    if load_actuals:
        today_actuals = reference_model_module.load_model_parameters().create_instance(
                                                                os.path.join(os.path.expanduser(instance_directory_name),
                                                                this_date,
                                                                actuals_filename)
                                                                )
    else:
        today_actuals = None

    if (next_date is None) or ((this_hour == 0) and (not use_next_day_in_ruc)):
        next_day_data = None
        next_day_actuals = None
    else:
        next_day_data = reference_model_module.load_model_parameters().create_instance(
                                                                os.path.join(os.path.expanduser(instance_directory_name),
                                                                next_date,
                                                                scenario_filename )
                                                                )
        if load_actuals:
            next_day_actuals = reference_model_module.load_model_parameters().create_instance(
                                                                os.path.join(os.path.expanduser(instance_directory_name),
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
    if (prior_deterministic_ruc != None) and (projected_sced != None):
        UnitOnT0Dict = {}
        UnitOnT0StateDict = {}
        PowerGeneratedT0Dict = {}

        for g in sorted(today_data.ThermalGenerators):

            if prior_root_node is None:
                final_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g, ruc_every_hours])))
            else:
                final_unit_on_state = int(round(prior_root_node.get_variable_value("UnitOn", (g, ruc_every_hours))))
            state_duration = 1

            hours = list(range(1, ruc_every_hours))
            hours.reverse()
            for i in hours:
                if prior_root_node is None:
                    this_unit_on_state = int(round(value(prior_deterministic_ruc.UnitOn[g, i])))
                else:
                    this_unit_on_state = int(round(prior_root_node.get_variable_value("UnitOn", (g, i))))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration
            ## get hours before prior_deterministic_ruc
            prior_UnitOnT0State = int(value(prior_deterministic_ruc.UnitOnT0State[g]))
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
            power_generated_at_t0 = value(projected_sced.PowerGenerated[g, sced_schedule_hour])

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
            min_power_output = value(today_data.MinimumPowerOutput[g])
            max_power_output = value(today_data.MaximumPowerOutput[g])

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


    ruc_data = _get_ruc_data(options, today_data, this_hour, next_day_data, UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, projected_sced, ruc_horizon, use_next_day_in_ruc, today_actuals, next_day_actuals)

    reference_model = reference_model_module.model

    ## if relaxing the initial ramping, set this in module
    if (prior_deterministic_ruc is None) and options.relax_t0_ramping_initial_day:
        reference_model.enforce_t1_ramp_rates = False

    new_deterministic_ruc = reference_model.create_instance(data=ruc_data)

    ## reset this value for all subsequent runs
    if (prior_deterministic_ruc is None) and options.relax_t0_ramping_initial_day:
        reference_model.enforce_t1_ramp_rates = True

    if output_initial_conditions:
        _report_initial_conditions_for_deterministic_ruc(new_deterministic_ruc,
                                                        max_thermal_generator_label_length=max_thermal_generator_label_length)

        _report_demand_for_deterministic_ruc(new_deterministic_ruc,
                                            options.ruc_every_hours,
                                            max_bus_label_length=max_bus_label_length)

    ## if available, use the tailing hours from
    ## previous RUC to give a partial warmstart for this one
    ## this additionally requires the warmstart option to be set for determinisitic ruc
    if prior_deterministic_ruc is not None:
        ruc_horizon = value(prior_deterministic_ruc.NumTimePeriods)
        prior_start_hour = options.ruc_every_hours
        prior_ruc_time_periods = range(prior_start_hour+1,ruc_horizon+1)
        for g in prior_deterministic_ruc.ThermalGenerators:
            for t in prior_ruc_time_periods:
                if prior_root_node is None:
                    new_deterministic_ruc.UnitOn[g,t-prior_start_hour].value = int(round(value(prior_deterministic_ruc.UnitOn[g,t])))
                else:

                    new_deterministic_ruc.UnitOn[g,t-prior_start_hour].value = int(round(prior_root_node.get_variable_value("UnitOn", (g, t))))

    return new_deterministic_ruc, new_scenario_tree

    
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

    if projected_sced != None:
        StorageSocOnT0Dict = dict((s, value(projected_sced.SocStorage[s, sced_schedule_hour]))
                                             for s in sorted(today_data.Storage))
    else:
        StorageSocOnT0Dict = dict((s, value(today_data.StorageSocOnT0[s])) for s in sorted(today_data.Storage))


    ruc_data = _get_data_dict(today_data, ruc_horizon, demand_dict, reserve_dict, options.reserve_factor, \
                              min_nondispatch_dict, max_nondispatch_dict, \
                              UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict)

    return ruc_data

def _report_initial_conditions_for_deterministic_ruc(deterministic_instance,
                                                    max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated must-run):")
    deterministic_instance.PowerGeneratedT0.pprint()
    deterministic_instance.UnitOnT0State.pprint()
    #assert(len(deterministic_instance.PowerGeneratedT0) == len(deterministic_instance.UnitOnT0State))
    for g in sorted(deterministic_instance.ThermalGenerators):
        print(("%-"+str(max_thermal_generator_label_length)+"s %5d %7d %12.2f %6d") % 
              (g, 
               value(deterministic_instance.UnitOnT0[g]),
               value(deterministic_instance.UnitOnT0State[g]), 
               value(deterministic_instance.PowerGeneratedT0[g]),
               value(deterministic_instance.MustRun[g])))

    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):    
        total_t0_power_output += value(deterministic_instance.PowerGeneratedT0[g])
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):
        t0_state = value(deterministic_instance.UnitOnT0State[g])
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= value(deterministic_instance.MinimumDownTime[g]):
                total_new_online_capacity += min(value(deterministic_instance.StartupRampLimit[g]), value(deterministic_instance.MaximumPowerOutput[g]))
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g in sorted(deterministic_instance.ThermalGenerators):
        t0_state = value(deterministic_instance.UnitOnT0State[g])
        if t0_state > 0: # the unit has been on
            if t0_state >= value(deterministic_instance.MinimumUpTime[g]):
                if value(deterministic_instance.PowerGeneratedT0[g]) <= value(deterministic_instance.ShutdownRampLimit[g]):
                    total_new_offline_capacity += value(deterministic_instance.PowerGeneratedT0[g])
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)
    assert (len(deterministic_instance.PowerGeneratedT0) == len(deterministic_instance.UnitOnT0State))

def _report_demand_for_deterministic_ruc(ruc_instance,
                                         ruc_every_hours,
                                         max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print("Projected Demand:")
    for b in sorted(ruc_instance.Buses):
        print(("%-"+str(max_bus_label_length)+"s: ") % b, end=' ')
        for t in range(1,min(value(ruc_instance.NumTimePeriods)+1, 37)):
            print("%8.2f"% value(ruc_instance.Demand[b,t]), end=' ')
            if t == ruc_every_hours: 
                print(" |", end=' ')
        print("")


###### END Deterministic RUC solvers and helper functions #########

# utility to resolve the scenarios in the extensive form by fixing the unit on binaries. 
# mirrors what ISOs do when solving the RUC. snapshots the solution into the scenario tree.
# NOTE: unused
def resolve_stochastic_ruc_with_fixed_binaries(stochastic_ruc_instance, scenario_instances, scenario_tree,
                                               options, solver, solver_options):

    # the scenarios are independent once we fix the unit on binaries.
    for scenario_name, scenario_instance in iteritems(scenario_instances):
        print("Processing scenario name=",scenario_name)
        for index, var_data in iteritems(scenario_instance.UnitOn):
            var_data.fix()
        scenario_instance.preprocess()

        results = call_solver(solver,scenario_instance, tee=options.output_solver_logs,
                              keepfiles=options.keep_solver_files,**solve_options[solver])
        if results.solver.termination_condition not in safe_termination_conditions: 
            print("Writing failed scenario=" + scenario_name + " to file=" + output_filename)
            lp_writer = ProblemWriter_cpxlp()            
            output_filename = "bad_resolve_scenario.lp"
            lp_writer(scenario_instance, output_filename, lambda x: True, True)
            raise RuntimeError("Halting - failed to solve scenario=" + scenario_name +
                               " when re-solving stochastic RUC with fixed binaries")

        for index, var_data in iteritems(scenario_instance.UnitOn):
            var_data.unfix()
        scenario_instance.preprocess()

    scenario_tree.pullScenarioSolutionsFromInstances()
    scenario_tree.snapshotSolutionFromScenarios()

##### BEGIN Stochastic RUC solvers and helper functions #######
def _report_begin_stochastic_ruc(this_date, this_hour):
    print("")
    print("Solving stochastic RUC for date:", this_date, "beginning hour:", this_hour)
    print()
    print("Creating and solving stochastic RUC instance for date:", this_date, "beginning hour:", this_hour)

def _report_end_stochastic_ruc(scenario_instances, scenario_tree, options):
    # now report the solution that we have generated.
    expected_cost = scenario_tree.findRootNode().computeExpectedNodeCost()
    if expected_cost == None:
        scenario_tree.pprintCosts()
        raise RuntimeError("Could not computed expected cost - one or more stage"
                           " costs is undefined. This is likely due to not all"
                           " second stage variables being specified in the"
                           " scenario tree structure definition file.")
    print("Stochastic RUC expected cost: %8.2f" % expected_cost)

    if options.output_ruc_solutions:
        print("")
        _output_solution_for_stochastic_ruc(scenario_instances, 
                                           scenario_tree, 
                                           options.ruc_every_hours,
                                           output_scenario_dispatches=options.output_ruc_dispatches)

    print("")
    _report_fixed_costs_for_stochastic_ruc(scenario_instances)
    print("")
    _report_generation_costs_for_stochastic_ruc(scenario_instances)
    print("")
    _report_load_generation_mismatch_for_stochastic_ruc(scenario_instances)


###################################################################
# utility functions for computing and reporting various aspects   #
# of a stochastic extensive form unit commitment solution.        #
###################################################################

def _output_solution_for_stochastic_ruc(scenario_instances, scenario_tree, ruc_every_hours, output_scenario_dispatches=False):

    # we can grab solution information either from the scenario tree, or 
    # from the instances themselves. by convention, we are grabbing values
    # from the scenario tree in cases where multiple instances are involved
    # (e.g., at a root node, where variables are blended), and otherwise
    # from the instances themselves.

    print("Generator Commitments:")
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]
    root_node = scenario_tree.findRootNode()
    last_time_period = min(value(arbitrary_scenario_instance.NumTimePeriods)+1, 37)
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        print("%30s: " % g, end=' ')
        for t in range(1, last_time_period):
            raw_value = root_node.get_variable_value("UnitOn", (g, t))
            if raw_value is None:
                raise RuntimeError("***Failed to extract value for variable UnitOn, index=" + str((g, t)) +
                                   " from scenario tree.")
            print("%2d"% int(round(raw_value)), end=' ')
            if t == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    if output_scenario_dispatches:
        scenario_names = sorted(scenario_instances.keys())
        for scenario_name in scenario_names:
            scenario_instance = scenario_instances[scenario_name]
            print("")
            print("Generator Dispatch Levels for Scenario=" + scenario_name)
            for g in sorted(scenario_instance.ThermalGenerators):
                print("%-30s: " % g, end=' ')
                for t in range(1, ruc_every_hours+1):
                    print("%6.2f"% value(scenario_instance.PowerGenerated[g, t]), end=' ')
                print("")

            print("")
            print("Generator Production Costs Scenario=" + scenario_name)
            for g in sorted(scenario_instance.ThermalGenerators):
                print("%30s: " % g, end=' ')
                for t in range(1, ruc_every_hours+1):
                    print("%8.2f"% value(scenario_instance.ProductionCost[g,t]), end=' ')
                print("")
            print("%30s: " % "Total", end=' ')
            for t in range(1, ruc_every_hours+1):
                sum_production_costs = sum(value(scenario_instance.ProductionCost[g, t])
                                           for g in scenario_instance.ThermalGenerators)
                print("%8.2f"% sum_production_costs, end=' ')
            print("")


def _report_fixed_costs_for_stochastic_ruc(scenario_instances):

    # we query on an arbitrary scenario instance, because the 
    # input instance / solution is assumed to be non-anticipative.
    instance = scenario_instances[list(scenario_instances.keys())[0]]

    # no fixed costs to output for stage 2 as of yet.
    stage = "Stage_1"
    print("Fixed cost for stage %s:     %10.2f" % (stage, value(instance.CommitmentStageCost[stage])))

    # break out the startup, shutdown, and minimum-production costs, for reporting purposes.
    startup_costs = sum(value(instance.StartupCost[g,t])
                        for g in instance.ThermalGenerators for t in instance.CommitmentTimeInStage[stage])
    shutdown_costs = sum(value(instance.ShutdownCost[g,t])
                         for g in instance.ThermalGenerators for t in instance.CommitmentTimeInStage[stage])
    minimum_generation_costs = sum(sum(value(instance.UnitOn[g,t])
                                       for t in instance.CommitmentTimeInStage[stage]) *
                                   value(instance.MinimumProductionCost[g]) *
                                   value(instance.TimePeriodLength) for g in instance.ThermalGenerators)

    print("   Startup costs:                 %10.2f" % startup_costs)
    print("   Shutdown costs:                %10.2f" % shutdown_costs)
    print("   Minimum generation costs:      %10.2f" % minimum_generation_costs)


def _report_generation_costs_for_stochastic_ruc(scenario_instances):

    # load shedding is scenario-dependent, so we have to loop through each scenario instance.
    # but we will in any case use an arbitrary scenario instance to access index sets.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    # only worry about two-stage models for now..
    second_stage = "Stage_2"  # TBD - data-drive this - maybe get it from the scenario tree? StageSet should also be ordered in the UC models.

    print("Generation costs (Scenario, Cost):")
    for scenario_name in sorted(scenario_instances.keys()):
        scenario_instance = scenario_instances[scenario_name]
        # print("%-15s %12.2f" % (scenario_name, scenario_instance.GenerationStageCost[second_stage]))
        print("%-15s %12.2f" % (scenario_name, value(scenario_instance.GenerationStageCost[second_stage])))


def _report_load_generation_mismatch_for_stochastic_ruc(scenario_instances):

    # load shedding is scenario-dependent, so we have to loop through each scenario instance.
    # but we will in any case use an arbitrary scenario instance to access index sets.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    for t in sorted(arbitrary_scenario_instance.TimePeriods):
        for scenario_name, scenario_instance in scenario_instances.items():

            sum_mismatch = round_small_values(sum(value(scenario_instance.LoadGenerateMismatch[b,t])
                                                  for b in scenario_instance.Buses))
            if sum_mismatch != 0.0:
                posLoadGenerateMismatch = round_small_values(sum(value(scenario_instance.posLoadGenerateMismatch[b, t])
                                                                 for b in scenario_instance.Buses))
                negLoadGenerateMismatch = round_small_values(sum(value(scenario_instance.negLoadGenerateMismatch[b, t])
                                                                 for b in scenario_instance.Buses))
                if posLoadGenerateMismatch != 0.0:
                    print("Projected load shedding reported at t=%d -     total=%12.2f - scenario=%s"
                          % (t, posLoadGenerateMismatch, scenario_name))
                if negLoadGenerateMismatch != 0.0:
                    print("Projected over-generation reported at t=%d -   total=%12.2f - scenario=%s"
                          % (t, negLoadGenerateMismatch, scenario_name))

            reserve_shortfall_value = round_small_values(value(scenario_instance.ReserveShortfall[t]))
            if reserve_shortfall_value != 0.0:
                print("Projected reserve shortfall reported at t=%d - total=%12.2f - scenario=%s"
                      % (t, reserve_shortfall_value, scenario_name))

# a utility to construct pysp_instance_creation_callback
def make_instance_creation_callback(options, this_date, this_hour, next_date, prior_ruc, projected_sced, sced_schedule_hour, prior_root_node, ruc_horizon, use_next_day_in_ruc):
    def pysp_instance_creation_callback(scenario_name, node_name_list):
        return _create_deterministic_ruc(options, this_date, this_hour, next_date, prior_ruc, projected_sced, False, sced_schedule_hour, ruc_horizon, use_next_day_in_ruc, scenario=scenario_name, prior_root_node=prior_root_node)[0]
    return pysp_instance_creation_callback

# a utility to create a stochastic RUC instance and solve it via the extensive form.
def create_and_solve_stochastic_ruc_via_ef(solver, solve_options, options,
                                           this_date,
                                           this_hour,
                                           next_date,
                                           yesterday_stochastic_ruc_scenarios,
                                           yesterday_scenario_tree,
                                           output_initial_conditions,
                                           projected_sced_instance,
                                           sced_schedule_hour,
                                           ruc_horizon,
                                           use_next_day_in_ruc):

    # NOTE: the "yesterday" technically is relative to whatever day this stochastic RUC is being constructed/solved for.
    #       so in the context of the simulator, yesterday actually means the RUC executing for today.

    _report_begin_stochastic_ruc(this_date, this_hour)

    print("Constructing scenario tree and scenario instances...")

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage",str(this_date)) 
    if not os.path.exists(instance_directory_name):
        raise RuntimeError("Stochastic RUC instance data directory=%s either does not exist or cannot be read"
                           % instance_directory_name)

    if yesterday_stochastic_ruc_scenarios != None:

        yesterday_root_node = yesterday_scenario_tree.findRootNode()
        arbitrary_scenario_instance = yesterday_stochastic_ruc_scenarios[list(yesterday_stochastic_ruc_scenarios.keys())[0]]

    else:
        arbitrary_scenario_instance = None
        yesterday_root_node = None

    instance_creator = make_instance_creation_callback(options, this_date, this_hour, next_date, arbitrary_scenario_instance, projected_sced_instance, sced_schedule_hour, yesterday_root_node, ruc_horizon, use_next_day_in_ruc)

    scenario_tree_instance_factory = ScenarioTreeInstanceFactory( instance_creator,
                                                                 os.path.expanduser(instance_directory_name))

    scenario_tree = scenario_tree_instance_factory.generate_scenario_tree()

    print("")
    print("Number of scenarios=%d" % len(scenario_tree._scenarios))
    print("")

    scenario_instances = scenario_tree_instance_factory.construct_instances_for_scenario_tree(scenario_tree)
    
    print("Done constructing scenario instances")
    
    scenario_tree_instance_factory.close()

    scenario_tree.linkInInstances(scenario_instances)

    stochastic_ruc_instance = create_ef_instance(scenario_tree,
                                                 verbose_output=options.verbose)

    if output_initial_conditions:
        print("")
        _report_initial_conditions_for_stochastic_ruc(scenario_instances)

    # solve the extensive form - which includes loading of results
    # (but not into the scenario tree)
    _solve_extensive_form(solver, stochastic_ruc_instance, options,solve_options, tee=options.output_solver_logs)

    # snapshot the solution into the scenario tree.
    scenario_tree.pullScenarioSolutionsFromInstances()
    scenario_tree.snapshotSolutionFromScenarios()
    print('create_and_solve complete')

    _report_end_stochastic_ruc(scenario_instances, scenario_tree, options)

    return scenario_instances, scenario_tree


def _solve_extensive_form(solver, stochastic_instance, options, solve_options, tee=False):

    results = call_solver(solver, stochastic_instance, tee=tee, **solve_options[solver])
    stochastic_instance.solutions.load_from(results)


# a utility to create a stochastic RUC instance and solve it via progressive hedging.
                                #NOTE: solver and solve_options is not used currently by this funciton
# TODO: ?? Make this funcion aware of sub-daily commitments!
# TODO: ?? doesn't work with options ruc_horizon or use_next_day_in_ruc
def create_and_solve_stochastic_ruc_via_ph(solver, solve_options, options,
                                           this_date, 
                                           this_hour,
                                           next_date,
                                           yesterday_stochastic_ruc_scenarios,
                                           yesterday_scenario_tree,
                                           output_initial_conditions,
                                           projected_sced_instance,
                                           sced_schedule_hour,
                                           ruc_horizon,
                                           use_next_day_in_ruc):

    if this_hour != 0:
        raise Exception("Solving stochastic RUC via PH currently does not support sub-daily commitments")

    _report_begin_stochastic_ruc(this_date, this_hour)

    print("Constructing scenario tree and scenario instances...")

    instance_directory_name = os.path.join(options.data_directory, "pyspdir_twostage", this_date)
    if not os.path.exists(instance_directory_name):
        raise RuntimeError("Stochastic RUC instance data directory=%s either does not exist or cannot be read"
                           % instance_directory_name)

    scenario_tree_instance_factory = ScenarioTreeInstanceFactory(os.path.expanduser(options.model_directory),
                                                                 os.path.expanduser(instance_directory_name))

    scenario_tree = scenario_tree_instance_factory.generate_scenario_tree()

    print("")
    print("Number of scenarios=%d" % len(scenario_tree._scenarios))
    print("")

    # before we construct the instances, cull the constraints from the model - PH is solved outside of 
    # the simulator process, and we only need to instantiate variables in order to store solutions.
    cull_constraints_from_instance(scenario_tree_instance_factory._model_object, ["BuildBusZone","CreatePowerGenerationPiecewisePoints"])

    scenario_instances = scenario_tree_instance_factory.construct_instances_for_scenario_tree(scenario_tree)

    scenario_tree_instance_factory.close()

    scenario_tree.linkInInstances(scenario_instances)

    ## when we do daily commits (which we assume when using PH)
    ## this is a valid transformation
    if sced_schedule_hour is None:
        yesterday_hour_invoked = None
    else:
        yesterday_hour_invoked = 24-sced_schedule_hour

    _write_stochastic_initial_conditions(yesterday_hour_invoked, 
                                        yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                        scenario_instances, 
                                        output_initial_conditions, options,
                                        projected_sced_instance)

    ph_output_filename = options.output_directory + os.sep + str(this_date) + os.sep + "ph.out"

    # TBD: We currently don't link from the option --symbolic-solver-labels to propagate to the below,
    #      when constructing PH command-lines. This is a major omission. For now, it is enabled always -
    #      infeasibilities are impossible to diangose when numeric labels are employed.
    if options.ph_mode == "serial":
        command_line = ("runph -m %s -i %s --max-iterations=%d --solver=%s --solver-manager=serial "
                        "--solution-writer=pyomo.pysp.plugins.csvsolutionwriter "
                        "--user-defined-extension=initialstatesetter.py %s >& %s"
                        % (options.model_directory, instance_directory_name, options.ph_max_iterations,
                           options.ph_ruc_subproblem_solver_type, options.ph_options, ph_output_filename))
    elif options.ph_mode == "localmpi":
        if options.pyro_port == None:
            raise RuntimeError("A pyro port must be specified when running PH in localmpi mode")
        command_line = ("runph -m %s -i %s "
                        "--disable-gc "
                        "--max-iterations=%d --solver-manager=phpyro "
                        "--solver=%s --pyro-host=%s --pyro-port=%d "
                        " --scenario-solver-options=\"threads=2\" "
                        "--solution-writer=pyomo.pysp.plugins.csvsolutionwriter "
                        "--user-defined-extension=initialstatesetter.py %s >& %s"
                        % (options.model_directory, instance_directory_name,
                           options.ph_max_iterations, options.ph_ruc_subproblem_solver_type,
                           options.pyro_host,
                           options.pyro_port, 
                           options.ph_options, ph_output_filename))
    else:
        raise RuntimeError("Unknown PH execution mode=" + str(options.ph_mode) +
                           " specified - legal values are: 'serial' and 'localmpi'")
    
    print("")
    print("Executing PH to solve stochastic RUC - command line=" + command_line)
    sys.stdout.flush()
    
    process = subprocess.Popen(command_line,shell=True, executable='/bin/bash',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # os.system(command_line)
    process.wait()

    # validate that all expected output file were generated.
    if not os.path.exists("ph.csv"):
        raise RuntimeError("No results file=ph.csv generated.")
    if not os.path.exists("ph_StageCostDetail.csv"):
        raise RuntimeError("No result file=ph_StageCostDetail.csv")
    
    # move any files generated by PH to the appropriate output directory for that day.
    shutil.move("ph.csv", options.output_directory + os.sep + str(this_date))
    shutil.move("ph_StageCostDetail.csv", options.output_directory + os.sep + str(this_date))
    shutil.move("ic.txt", options.output_directory + os.sep + str(this_date))

    # load the PH solution into the corresponding scenario instances. 
    _load_stochastic_ruc_solution_from_csv(scenario_instances, 
                                          scenario_tree, 
                                          options.output_directory + os.sep + str(this_date) + os.sep + "ph.csv")

    # snapshot the solution into the scenario tree.
    scenario_tree.pullScenarioSolutionsFromInstances()

    scenario_tree.snapshotSolutionFromScenarios()

    _report_end_stochastic_ruc(scenario_instances, scenario_tree, options)

    return scenario_instances, scenario_tree

#
# a utility function to write generator initial state information to a text file, for use with solver
# callbacks when exeucting runph. presently writes the file to "ic.txt", for "initial conditions".
#

# NOTE: this function is only called by solve_stochastic_ruc_via_ph, so it doesn't have any sub-daily information
def _write_stochastic_initial_conditions(yesterday_hour_invoked, 
                                        yesterday_stochastic_ruc_scenarios, yesterday_scenario_tree,
                                        today_stochastic_ruc_scenarios, 
                                        output_initial_conditions, options,
                                        projected_sced_instance):

    # the sced model and the prior sced instance are used to construct the SCED for projections
    # of generator output at time period t=24. they are only used if there is a stochastic RUC
    # from yesterday. the h hour is the hour at which the RUC is invoked.

    output_filename = "ic.txt"
    output_file = open(output_filename, "w")

    # set up the initial conditions for the stochastic RUC, based on all information
    # available as to the conditions on the initial day. if this is the first simulation
    # day, then we have nothing to go on other than the initial conditions already
    # specified in the scenario instances. however, if this is not the first simulation
    # day, use the projected solution state embodied (either explicitly or implicitly)
    # in the prior date stochastic RUC to establish the initial conditions.
    if yesterday_stochastic_ruc_scenarios != None:

        yesterday_root_node = yesterday_scenario_tree.findRootNode()
        arbitrary_scenario_instance = yesterday_stochastic_ruc_scenarios[
            list(yesterday_stochastic_ruc_scenarios.keys())[0]]

        for g in sorted(arbitrary_scenario_instance.ThermalGenerators):

            final_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, 24))))
            state_duration = 1
            hours = list(range(1, 24))
            hours.reverse()
            for i in hours:
                this_unit_on_state = int(round(yesterday_root_node.get_variable_value("UnitOn", (g, i))))
                if this_unit_on_state != final_unit_on_state:
                    break
                state_duration += 1
            if final_unit_on_state == 0:
                state_duration = -state_duration

            ## get hours before prior_deterministic_ruc
            prior_UnitOnT0State = int(value(arbitrary_scenario_instance.UnitOnT0State[g]))
            # if we have the same state at the beginning and the end of the horizon,
            # AND it agrees with the state from the previous, we can add the prior state.
            # Important for doing more than daily commitments, and long horizon generators
            # like nuclear and some coal units.
            if abs(state_duration) == 24 and (\
                ( prior_UnitOnT0State < 0 and state_duration < 0 ) or \
                ( prior_UnitOnT0State > 0 and state_duration > 0 )
                ):
                state_duration += prior_UnitOnT0State

            projected_power_generated = value(projected_sced_instance.PowerGenerated[
                                                  g, 23 - yesterday_hour_invoked + 1])
            # the input h hours are 0-based - 23 is the last hour of the day

            # on occasion, the average power generated across scenarios for a single generator
            # can be a very small negative number, due to MIP tolerances allowing it. if this
            # is the case, simply threshold it to 0.0. similarly, the instance validator will
            # fail if the average power generated is small-but-positive (e.g., 1e-14) and the
            # UnitOnT0 state is Off. in the latter case, just set the average power to 0.0.
            if projected_power_generated < 0.0:
                projected_power_generated = 0.0
            elif final_unit_on_state == 0:
                projected_power_generated = 0.0                

            print(g, state_duration, projected_power_generated, file=output_file)

            # propagate the initial conditions to each scenario - the initial conditions
            # must obviously be non-anticipative. 
            for instance in today_stochastic_ruc_scenarios.values():
                instance.UnitOnT0[g] = final_unit_on_state
                instance.UnitOnT0State[g] = state_duration

                # the validators are rather picky, in that tolerances are not acceptable.
                # given that the average power generated comes from an optimization 
                # problem solve, the average power generated can wind up being less
                # than or greater than the bounds by a small epsilon. touch-up in this
                # case.
                min_power_output = value(instance.MinimumPowerOutput[g])
                max_power_output = value(instance.MaximumPowerOutput[g])
                
                # TBD: Eventually make the 1e-5 an user-settable option.
                if math.fabs(min_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = min_power_output
                elif math.fabs(max_power_output - projected_power_generated) <= 1e-5: 
                    instance.PowerGeneratedT0[g] = max_power_output
                else:
                    instance.PowerGeneratedT0[g] = projected_power_generated

    else:
        print("")
        print("***WARNING: No prior stochastic RUC instance available "
              "=> There is no solution from which to draw initial conditions from; running with defaults.")

        arbitrary_scenario_instance = today_stochastic_ruc_scenarios[list(today_stochastic_ruc_scenarios.keys())[0]]

        for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
            unit_on_t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
            t0_power_generated = value(arbitrary_scenario_instance.PowerGeneratedT0[g])
            print(g, unit_on_t0_state, t0_power_generated, file=output_file)

    # always output the reserve factor.
    print(options.reserve_factor, file=output_file)

    if output_initial_conditions:
        print("")
        _report_initial_conditions_for_stochastic_ruc(today_stochastic_ruc_scenarios)
        
    print("")
    print('Finished writing initial conditions (stochastic)')

    output_file.close()

# 
# a utility to load a solution from a PH or EF-generated CSV solution file into
# a stochastic RUC instance.
#

# NOTE: this function is only called by solve_stochastic_ruc_via_ph, so it doesn't have any sub-daily information
def _load_stochastic_ruc_solution_from_csv(scenario_instances, scenario_tree, csv_filename):

    csv_file = open(csv_filename, "r")
    csv_reader = csv.reader(csv_file, delimiter=",", quotechar='|')
    for line in csv_reader:
        stage_name = line[0].strip()
        node_name = line[1].strip()
        variable_name = line[2].strip()
        index = line[3].strip().split(":")
        index = tuple(i.strip() for i in index)
        quantity = float(line[4])

        if len(index) == 1:
            index = index[0]
            try:
                index = int(index)
            except ValueError:
                pass
        else:
            transformed_index = ()
            for piece in index:
                piece = piece.strip()
                piece = piece.lstrip('\'')
                piece = piece.rstrip('\'')
                transformed_component = None
                try:
                    transformed_component = int(piece)
                except ValueError:
                    transformed_component = piece
                transformed_index = transformed_index + (transformed_component,)
            index = transformed_index

        tree_node = scenario_tree.get_node(node_name)
        try:
            variable_id = tree_node._name_index_to_id[(variable_name, index)]
            for var_data, probability in tree_node._variable_datas[variable_id]:
                var_data.stale = False
                var_data.value = quantity
        except:
            # we will assume that we're dealing with a stage cost variable -
            # which is admittedly dangerous.
            for cost_vardata, probability in tree_node._cost_variable_datas:
                cost_vardata.stale = False
                cost_vardata.value = quantity

    csv_file.close()


def _report_initial_conditions_for_stochastic_ruc(scenario_instances):

    # we query on an arbitrary scenario instance, because the 
    # initial conditions are assumed to be non-anticipative.
    arbitrary_scenario_instance = scenario_instances[list(scenario_instances.keys())[0]]

    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated):")
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        print("%-30s %5d %7d %12.2f" % (g, 
                                        value(arbitrary_scenario_instance.UnitOnT0[g]), 
                                        value(arbitrary_scenario_instance.UnitOnT0State[g]), 
                                        value(arbitrary_scenario_instance.PowerGeneratedT0[g]))) 


    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):    
        total_t0_power_output += value(arbitrary_scenario_instance.PowerGeneratedT0[g])
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= value(arbitrary_scenario_instance.MinimumDownTime[g]):
                total_new_online_capacity += min(value(arbitrary_scenario_instance.StartupRampLimit[g]),
                                                 value(arbitrary_scenario_instance.MaximumPowerOutput[g]))
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g in sorted(arbitrary_scenario_instance.ThermalGenerators):
        t0_state = value(arbitrary_scenario_instance.UnitOnT0State[g])
        if t0_state > 0: # the unit has been on
            if t0_state >= value(arbitrary_scenario_instance.MinimumUpTime[g]):
                if value(arbitrary_scenario_instance.PowerGeneratedT0[g]) <= \
                        value(arbitrary_scenario_instance.ShutdownRampLimit[g]):
                    total_new_offline_capacity += value(arbitrary_scenario_instance.PowerGeneratedT0[g])
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)

################################################################################
# a simple utility to construct the scenario tree representing the data in the #
# given instance input directory.
################################################################################


def _construct_scenario_tree(instance_directory):
    # create and populate the scenario tree model

    from pyomo.pysp.util.scenariomodels import scenario_tree_model
    scenario_tree_instance = None

    try:
        scenario_tree_instance_filename = os.path.expanduser(instance_directory) + os.sep + "ScenarioStructure.dat"
        scenario_tree_instance = scenario_tree_model.clone()
        instance_data = DataPortal(model=scenario_tree_instance)
        instance_data.load(filename=scenario_tree_instance_filename)
        scenario_tree_instance.load(instance_data)
    except IOError:
        exception = sys.exc_info()[1]
        print(("***ERROR: Failed to load scenario tree instance data from file=" + scenario_tree_instance_filename +
               "; Source error="+str(exception)))
        return None

    # construct the scenario tree
    scenario_tree = ScenarioTree(scenariotreeinstance=scenario_tree_instance)

    return scenario_tree

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
        else:
            simulation_data_directory_name = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date))
            scenario_tree = _construct_scenario_tree(simulation_data_directory_name)

            selected_index = random.randint(0, len(scenario_tree._scenarios)-1)
            selected_scenario = scenario_tree._scenarios[selected_index]
            simulated_dat_filename = os.path.join(options.data_directory, "pyspdir_twostage", str(a_date),
                                                  selected_scenario._name + ".dat")

    return simulated_dat_filename

def solve_deterministic_day_ahead_pricing_problem(solver, solve_options, deterministic_ruc_instance, options, reference_model_module):

    ## create a copy because we want to maintain the solution data
    ## in deterministic_ruc_instance
    from pyomo.core.plugins.transform.relax_integrality import RelaxIntegrality
    relax_integrality = RelaxIntegrality()

    pricing_type = options.day_ahead_pricing
    print("Computing day-ahead prices using method "+pricing_type+".")
    
    pricing_instance = relax_integrality.create_using(deterministic_ruc_instance)
    if pricing_type == "LMP":
        reference_model_module.fix_binary_variables(pricing_instance)
    elif pricing_type == "ELMP":
        ## for ELMP we fix all commitment binaries that were 0 in the RUC solve
        pricing_var_generator = reference_model_module.status_var_generator(pricing_instance)
        ruc_var_generator = reference_model_module.status_var_generator(deterministic_ruc_instance)
        for pricing_status_var, ruc_status_var in zip(pricing_var_generator, ruc_var_generator):
            if int(round(value(ruc_status_var))) == 0:
                pricing_status_var.value = 0
                pricing_status_var.fix()
    elif pricing_type == "aCHP":
        pass
    else:
        raise RuntimeError("Unknown pricing type "+pricing_type+".")

    reference_model_module.reconstruct_instance_for_pricing(pricing_instance)

    reference_model_module.define_suffixes(pricing_instance)

    ## change the penalty prices to the caps, if necessary

    # In case of demand shortfall, the price skyrockets, so we threshold the value.
    if value(pricing_instance.LoadMismatchPenalty) > options.price_threshold:
        pricing_instance.LoadMismatchPenalty = options.price_threshold

    # In case of reserve shortfall, the price skyrockets, so we threshold the value.
    if value(pricing_instance.ReserveShortfallPenalty) > options.reserve_price_threshold:
        pricing_instance.ReserveShortfallPenalty = options.reserve_price_threshold

    pricing_results = call_solver(solver,
                                   pricing_instance, 
                                   tee=options.output_solver_logs,
                                   **solve_options[solver])
                                                  
    if pricing_results.solver.termination_condition not in safe_termination_conditions:
        raise RuntimeError("Failed to solve day-ahead pricing problem!")

    pricing_instance.solutions.load_from(pricing_results)

    ## Debugging
    if pricing_instance.TotalCostObjective() > deterministic_ruc_instance.TotalCostObjective()*(1.+1.e-06):
        print("The pricing run had a higher objective value than the MIP run. This is indicative of a bug.")
        print("Writing LP pricing_problem.lp")
        output_filename = 'pricing_instance.lp'
        lp_writer = ProblemWriter_cpxlp()            
        lp_writer(pricing_instance, output_filename, 
                  lambda x: True, {"symbolic_solver_labels" : True})

        output_filename = 'deterministic_ruc_instance.lp'
        lp_writer = ProblemWriter_cpxlp()            
        lp_writer(deterministic_ruc_instance, output_filename, 
                  lambda x: True, {"symbolic_solver_labels" : True})

        raise RuntimeError("Halting due to bug in pricing.")
        

    day_ahead_prices = {}
    for b in pricing_instance.Buses:
        for t in pricing_instance.TimePeriods:
            balance_price = value(pricing_instance.dual[pricing_instance.PowerBalance[b,t]])
            day_ahead_prices[b,t-1] = balance_price

    day_ahead_reserve_prices = {}
    for t in pricing_instance.TimePeriods:
        reserve_price = value(pricing_instance.dual[pricing_instance.EnforceReserveRequirements[t]])
        # Thresholding the value of the reserve price to the passed in option
        day_ahead_reserve_prices[t-1] = reserve_price

    print("Recalculating RUC reserve procurement")

    ## scale the provided reserves by the amount over we are
    cleared_reserves= {}
    for t in range(0,options.ruc_every_hours):
        reserve_provided_t = sum(value(deterministic_ruc_instance.ReserveProvided[g,t+1]) for g in deterministic_ruc_instance.ThermalGenerators) 
        reserve_shortfall_t = value(deterministic_ruc_instance.ReserveShortfall[t+1])
        reserve_requirement_t = value(deterministic_ruc_instance.ReserveRequirement[t+1])


        surplus_reserves_t = reserve_provided_t + reserve_shortfall_t - reserve_requirement_t

        ## if there's a shortfall, grab the full amount from the RUC solve
        ## or if there's no provided reserves this can safely be set to 1.
        if round_small_values(reserve_shortfall_t) > 0 or reserve_provided_t == 0:
            surplus_multiple_t = 1.
        else:
            ## scale the reserves from the RUC down by the same fraction
            ## so that they exactly meed the needed reserves
            surplus_multiple_t = reserve_requirement_t/reserve_provided_t
        for g in deterministic_ruc_instance.ThermalGenerators:
            cleared_reserves[g,t] = value(deterministic_ruc_instance.ReserveProvided[g,t+1])*surplus_multiple_t
               
    thermal_gen_cleared_DA = {}
    thermal_reserve_cleared_DA = {}
    renewable_gen_cleared_DA = {}

    for t in range(0,options.ruc_every_hours):
        for g in deterministic_ruc_instance.ThermalGenerators:
            thermal_gen_cleared_DA[g,t] = value(deterministic_ruc_instance.PowerGenerated[g,t+1])
            thermal_reserve_cleared_DA[g,t] = cleared_reserves[g,t]
        for g in deterministic_ruc_instance.AllNondispatchableGenerators:
            renewable_gen_cleared_DA[g,t] = value(deterministic_ruc_instance.NondispatchablePowerUsed[g,t+1])

    return day_ahead_prices, day_ahead_reserve_prices, thermal_gen_cleared_DA, thermal_reserve_cleared_DA, renewable_gen_cleared_DA

def compute_market_settlements(ruc_instance, ## just for generators
                               thermal_gen_cleared_DA, thermal_reserve_cleared_DA, renewable_gen_cleared_DA,
                               day_ahead_prices, day_ahead_reserve_prices,
                               observed_thermal_dispatch_levels, observed_thermal_headroom_levels,
                               observed_renewables_levels, observed_bus_LMPs, reserve_RT_price_by_hour,
                               ):
    ## NOTE: This clears the market like ISO-NE seems to do it. We've assumed that the renewables
    #        bid in their expectation in the DA market -- this perhaps is not realistic
    ## TODO: Storage??
    ## TODO: Implicity assumes hourly SCED

    print("")
    print("Computing market settlements")

    thermal_gen_payment = {}
    thermal_reserve_payment = {}
    renewable_gen_payment = {}

    for t in range(0,24):
        for b in ruc_instance.Buses:
            price_DA = day_ahead_prices[b,t]
            price_RT = observed_bus_LMPs[b][t]
            for g in ruc_instance.ThermalGeneratorsAtBus[b]:
                thermal_gen_payment[g,t] = thermal_gen_cleared_DA[g,t]*price_DA \
                                            + (observed_thermal_dispatch_levels[g][t] - thermal_gen_cleared_DA[g,t])*price_RT
            for g in ruc_instance.NondispatchableGeneratorsAtBus[b]:
                renewable_gen_payment[g,t] = renewable_gen_cleared_DA[g,t]*price_DA \
                                            + (observed_renewables_levels[g][t] - renewable_gen_cleared_DA[g,t])*price_RT

    for t in range(0,24):
        r_price_DA = day_ahead_reserve_prices[t]
        r_price_RT = reserve_RT_price_by_hour[t]
        for g in ruc_instance.ThermalGenerators:
            thermal_reserve_payment[g,t] = thermal_reserve_cleared_DA[g,t]*r_price_DA \
                                            + (observed_thermal_headroom_levels[g][t] - thermal_reserve_cleared_DA[g,t])*r_price_RT

    print("Settlements computed")
    print("")
    
    return thermal_gen_cleared_DA, thermal_gen_payment, thermal_reserve_cleared_DA, thermal_reserve_payment, renewable_gen_cleared_DA, renewable_gen_payment

def create_ruc_instance_to_simulate_next_period(ruc_model, options, this_date, this_hour, next_date):

    simulated_dat_filename_this = compute_simulation_filename_for_date(this_date, options)
    print("")
    print("Actual simulation data for date=" + str(this_date) + " drawn from file=" +
          simulated_dat_filename_this)

    if not os.path.exists(simulated_dat_filename_this):
        raise RuntimeError("The file " + simulated_dat_filename_this + " does not exist or cannot be read.")

    ruc_instance_to_simulate_this_period = ruc_model.create_instance(simulated_dat_filename_this)

    if next_date is not None and this_hour != 0:
        simulated_dat_filename_next = compute_simulation_filename_for_date(next_date, options)
        print("")
        print("Actual simulation data for date=" + str(next_date) + " drawn from file=" +
              simulated_dat_filename_next)

        if not os.path.exists(simulated_dat_filename_next):
            raise RuntimeError("The file " + simulated_dat_filename_next + " does not exist or cannot be read.")

        ruc_instance_to_simulate_next_period = ruc_model.create_instance(simulated_dat_filename_next)
    else:
        ruc_instance_to_simulate_next_period = None

    ruc_data = _get_ruc_data(options, ruc_instance_to_simulate_this_period, this_hour, ruc_instance_to_simulate_next_period )

    print("")
    print("Creating RUC instance to simulate")
    ruc_instance_to_simulate = ruc_model.create_instance(data=ruc_data)

    return ruc_instance_to_simulate
