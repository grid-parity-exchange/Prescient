#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from prescient.util.math_utils import round_small_values
import numpy as np
from pyomo.core import value
import math
from typing import NamedTuple, Set, Any, TypeVar, Tuple, Dict, Iterable


G = TypeVar('G')
B = TypeVar('B')

class PreQuickstartCache(NamedTuple):
    ''' Stores select results before quickstart is activated, used to compare after quickstart is enabled '''
    quickstart_generators_off: Set[G]
    total_cost: float
    power_generated: float

class OperationsStatsExtractor:
    """
    Extracts information from an operations schedule model containing results
    """

    @staticmethod
    def get_pre_quickstart_data(sced_instance) -> PreQuickstartCache:
        ''' Extract select data from an optimization run without quickstart enabled, used to 
            detect changes when the same model is run with quickstart enabled
        '''
        quickstart_generators_off = {g for g in sced_instance.QuickStartGenerators
                                     if sced_instance.UnitOn[g, 1]==0}
        total_cost = OperationsStatsExtractor.get_total_costs(sced_instance)
        power_generated = OperationsStatsExtractor.get_power_generated(sced_instance)
        return PreQuickstartCache(quickstart_generators_off, total_cost, power_generated)

    @staticmethod
    def get_generators_used_as_quickstart(cache: PreQuickstartCache, sced_instance) -> Set[G]:
        ''' Get generators that are turned on, but would be turned off if quickstart were not enabled '''
        return {} if cache is None else {g for g in cache.quickstart_generators_off if sced_instance.UnitOn[g, 1]==1}

    @staticmethod
    def get_generator_quickstart_usage(cache: PreQuickstartCache, sced_instance) -> Dict[G, int]:
        ''' Get a dictionary which maps quick start generators to a 0 or 1, with a 1 indicating the generator 
            was used as a quickstart generator, and a 0 indicating it was not.

            A generator was used as a quickstart generator if it was on with quickstart enabled, but would have
            been off if quickstart was not enabled.
        '''
        if cache is None:
            return {g:0 for g in sced_instance.QuickStartGenerators}
        else:
            return {g: 1 if g in cache.quickstart_generators_off and sced_instance.UnitOn[g, 1]==1 else 0
                    for g in sced_instance.QuickStartGenerators}

    @staticmethod
    def get_additional_quickstart_power_generated(cache: PreQuickstartCache, sced_instance) -> float:
        return 0 if cache is None else OperationsStatsExtractor.get_power_generated(sced_instance) - cache.power_generated

    @staticmethod
    def get_additional_quickstart_costs(cache: PreQuickstartCache, sced_instance) -> float:
        return 0 if cache is None else OperationsStatsExtractor.get_total_costs(sced_instance) - cache.total_cost

    @staticmethod
    def get_costs_for_deterministic_sced(sced_instance) -> Tuple[float, float]:
        # only worry about two-stage models for now..
        fixed_cost = value(sum(sced_instance.StartupCost[g,1] + sced_instance.ShutdownCost[g,1] for g in sced_instance.ThermalGenerators) + 
                           sum(sced_instance.UnitOn[g,1] * sced_instance.MinimumProductionCost[g] * sced_instance.TimePeriodLength for g in sced_instance.ThermalGenerators))
        variable_cost = value(sced_instance.TotalProductionCost[1])
        
        return fixed_cost, variable_cost

    @staticmethod
    def get_fixed_costs(sced_instance):
        return value(sum(sced_instance.StartupCost[g,1] + sced_instance.ShutdownCost[g,1] 
                         for g in sced_instance.ThermalGenerators) + 
                     sum(sced_instance.UnitOn[g,1] * sced_instance.MinimumProductionCost[g] * sced_instance.TimePeriodLength 
                         for g in sced_instance.ThermalGenerators))

    @staticmethod
    def get_variable_costs(sced_instance):
        return value(sced_instance.TotalProductionCost[1])

    @staticmethod
    def get_total_costs(sced_instance) -> float:
        return OperationsStatsExtractor.get_fixed_costs(sced_instance) + \
               OperationsStatsExtractor.get_variable_costs(sced_instance)
    
    @staticmethod
    def get_load_mismatches(sced_instance):
        load_generation_mismatch_value = round_small_values(sum(value(sced_instance.LoadGenerateMismatch[b, 1])
                                                                for b in sced_instance.Buses))
        if load_generation_mismatch_value != 0.0:
            load_shedding_value = round_small_values(sum(value(sced_instance.posLoadGenerateMismatch[b, 1])
                                                         for b in sced_instance.Buses))
            over_generation_value = round_small_values(sum(value(sced_instance.negLoadGenerateMismatch[b, 1])
                                                           for b in sced_instance.Buses))
        else:
            load_shedding_value = 0.0
            over_generation_value = 0.0

        return load_shedding_value, over_generation_value

    @staticmethod
    def has_load_shedding(sced_instance) -> bool:
        return any(round_small_values(value(sced_instance.posLoadGenerateMismatch[b, 1])) > 0.0
                   for b in sced_instance.Buses)

    @staticmethod
    def get_available_reserve(sced_instance):
        available_reserve = sum(value(sced_instance.MaximumPowerAvailable[g, 1]) - value(sced_instance.PowerGenerated[g, 1])
                                for g in sced_instance.ThermalGenerators)
        return available_reserve

    @staticmethod
    def get_available_quick_start(sced_instance):
        """Given a SCED sced_instance with commitments from the RUC,
           determine how much quick start capacity is available 
        """
        available_quick_start_capacity = 0.0 
        for g in sced_instance.QuickStartGenerators:
            available = True  # until proven otherwise
            if int(round(value(sced_instance.UnitOn[g, 1]))) == 1:
                available = False  # unit was already committed in the RUC
            elif sced_instance.MinimumDownTime[g] > 1:
                # minimum downtime should be 1 or less, by definition of a quick start
                available = False
            elif (value(sced_instance.UnitOnT0[g]) - int(round(value(sced_instance.UnitOn[g, 1])))) == 1:
                # there cannot have been a a shutdown in the previous hour 
                available = False
 
            if available:  # add the amount of power that can be accessed in the first hour
                # use the min() because scaled startup ramps can be larger than the generator limit
                available_quick_start_capacity += min(value(sced_instance.ScaledStartupRampLimit[g]), value(sced_instance.MaximumPowerOutput[g]))
        
        return available_quick_start_capacity

    @staticmethod
    def get_reserve_shortfall(sced_instance):
       return round_small_values(value(sced_instance.ReserveShortfall[1]))

    @staticmethod
    def get_renewables_curtailment(sced_instance):
        # we only extract curtailment statistics for time period 1
        total_curtailment = round_small_values(sum((value(sced_instance.MaxNondispatchablePower[g, 1]) -
                                                    value(sced_instance.NondispatchablePowerUsed[g, 1]))
                                                   for g in sced_instance.AllNondispatchableGenerators))
        return total_curtailment

    @staticmethod
    def get_renewables_available(sced_instance):
        renewables_available = sum(value(sced_instance.MaxNondispatchablePower[g, 1]) 
                                   for g in sced_instance.AllNondispatchableGenerators)
        return renewables_available

    @staticmethod
    def get_renewables_used(sced_instance):
        renewables_used = sum(value(sced_instance.NondispatchablePowerUsed[g, 1])
                              for g in sced_instance.AllNondispatchableGenerators)
        return renewables_used

    @staticmethod
    def get_total_demand(sced_instance):
        return value(sced_instance.TotalDemand[1])

    @staticmethod
    def get_price(demand, fixed_costs, variable_costs):
        # 0 demand can happen, in some odd circumstances (not at the ISO level!).
        return 0.0 if demand == 0 else (fixed_costs + variable_costs) / demand


    @staticmethod
    def get_power_generated(sced_instance):
        return sum(value(sced_instance.PowerGenerated[g,1]) for g in sced_instance.ThermalGenerators)


    @staticmethod
    def get_on_off_and_ramps(sced_instance):
        """
        Get information about generators that turned on or off this time period

        Returns
        -------
        tuple
           a tuple with the following values:
              * num_on_offs: The number of generators that turned on or turned off this time period
              * sum_on_off_ramps: Amount of power generated by generators that turned on or off
              * sum_nominal_ramps: Change in amount of power generated by generators that stayed on or stayed off
        """
        num_on_offs = 0
        sum_on_off_ramps = 0.0
        sum_nominal_ramps = 0.0  # this is the total ramp change for units not switching on/off
    
        for g in sced_instance.ThermalGenerators:
            unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
            power_generated = value(sced_instance.PowerGenerated[g, 1])
            if value(sced_instance.UnitOnT0State[g]) > 0:
                # unit was on in previous time period
                if unit_on:
                    # on->on (no change)
                    sum_nominal_ramps += math.fabs(power_generated - value(sced_instance.PowerGeneratedT0[g]))
                else:
                    # on->off (unit turned on)
                    num_on_offs += 1
                    sum_on_off_ramps += power_generated
            else: # value(sced_instance.UnitOnT0State[g]) <= 0)
                # unit was off in previous time period
                if not unit_on:
                    # off->off (no change)
                    sum_nominal_ramps += math.fabs(power_generated - value(sced_instance.PowerGeneratedT0[g]))
                else:
                    # off->on (unit turned off)
                    num_on_offs += 1
                    sum_on_off_ramps += power_generated

        return num_on_offs, sum_on_off_ramps, sum_nominal_ramps

    @staticmethod
    def get_reserve_requirement(sced_instance):
        return value(sced_instance.ReserveRequirement[1])

    @staticmethod
    def get_observed_thermal_dispatch_level(sced_instance, thermal_generator):
        return value(sced_instance.PowerGenerated[thermal_generator, 1])

    @staticmethod
    def get_all_observed_thermal_dispatch_levels(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_thermal_dispatch_level(sced_instance, g)
                for g in sced_instance.ThermalGenerators}

    @staticmethod
    def get_observed_thermal_headroom_level(sced_instance, thermal_generator):
        g = thermal_generator
        headroom = value(sced_instance.MaximumPowerAvailable[g,1]) - value(sced_instance.PowerGenerated[g,1])
        return max(headroom, 0.0)

    @staticmethod
    def get_all_observed_thermal_headroom_levels(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_thermal_headroom_level(sced_instance, g)
                for g in sced_instance.ThermalGenerators}

    @staticmethod
    def get_observed_thermal_state(sced_instance, thermal_generator):
        return value(sced_instance.UnitOn[thermal_generator, 1])

    @staticmethod
    def get_all_observed_thermal_states(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_thermal_state(sced_instance, g)
                for g in sced_instance.ThermalGenerators}

    @staticmethod
    def get_observed_cost(sced_instance, thermal_generator):
        g = thermal_generator
        return value(sced_instance.StartupCost[g,1] 
                   + sced_instance.ShutdownCost[g,1]
                   + sced_instance.UnitOn[g,1] * sced_instance.MinimumProductionCost[g] * sced_instance.TimePeriodLength
                   + sced_instance.ProductionCost[g,1])

    @staticmethod
    def get_all_observed_costs(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_cost(sced_instance, g)
                for g in sced_instance.ThermalGenerators}

    @staticmethod
    def get_observed_renewables_level(sced_instance, non_dispatchable_generator):
        return value(sced_instance.NondispatchablePowerUsed[non_dispatchable_generator, 1])

    @staticmethod
    def get_all_observed_renewables_levels(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_renewables_level(sced_instance, g)
                for g in sced_instance.AllNondispatchableGenerators}

    @staticmethod
    def get_observed_renewables_curtailment(sced_instance, non_dispatchable_generator):
        g = non_dispatchable_generator
        return value(sced_instance.MaxNondispatchablePower[g, 1]) - value(sced_instance.NondispatchablePowerUsed[g, 1])

    @staticmethod
    def get_all_observed_renewables_curtailment(sced_instance):
        return {g: OperationsStatsExtractor.get_observed_renewables_curtailment(sced_instance, g)
                for g in sced_instance.AllNondispatchableGenerators}

    @staticmethod
    def get_observed_flow_level(sced_instance, line):
        return value(sced_instance.LinePower[line, 1])

    @staticmethod
    def get_all_observed_flow_levels(sced_instance):
        return {l: OperationsStatsExtractor.get_observed_flow_level(sced_instance, l)
                for l in sced_instance.TransmissionLines}

    @staticmethod
    def observed_bus_mismatch(sced_instance, bus):
        if value(sced_instance.LoadGenerateMismatch[bus, 1]) >= 0.0:
            return value(sced_instance.posLoadGenerateMismatch[bus, 1])
        else:
            return -1.0 * value(sced_instance.negLoadGenerateMismatch[bus, 1])

    @staticmethod
    def get_all_observed_bus_mismatches(sced_instance):
        return {b: OperationsStatsExtractor.observed_bus_mismatch(sced_instance, b)
                for b in sced_instance.Buses}

    @staticmethod
    def get_storage_input_dispatch_levels(sced_instance, storage):
       return np.array([value(sced_instance.PowerInputStorage[storage, 1])])

    @staticmethod
    def get_all_storage_input_dispatch_levels(sced_instance):
        return {s: OperationsStatsExtractor.get_storage_input_dispatch_levels(sced_instance, s)
                for s in sced_instance.Storage}

    @staticmethod
    def get_storage_output_dispatch_levels(sced_instance, storage):
        return np.array([value(sced_instance.PowerOutputStorage[storage, 1])])

    @staticmethod
    def get_all_storage_output_dispatch_levels(sced_instance):
        return {s: OperationsStatsExtractor.get_storage_output_dispatch_levels(sced_instance, s)
                for s in sced_instance.Storage}

    @staticmethod
    def get_storage_soc_dispatch_levels(sced_instance, storage):
        return np.array([value(sced_instance.SocStorage[s, 1])])

    @staticmethod
    def get_all_storage_soc_dispatch_levels(sced_instance):
        return {s: OperationsStatsExtractor.get_storage_soc_dispatch_levels(sced_instance, s)
                for s in sced_instance.Storage}

    @staticmethod
    def get_fleet_thermal_capacity(sced_instance):
        return sum(value(sced_instance.MaximumPowerOutput[g]) 
                   for g in sced_instance.ThermalGenerators)


class LoadMismatchStatsExtractor:
    """
    Extracts data from a schedule optimization model that has been configured and solved
    for LMPs (load mismatch penalties)
    """

    @staticmethod
    def get_reserve_RT_price(lmp_sced_instance):
        """
        Gets the reserve RT price

        Parameters
        ----------
        lmp_sced_instance: schedule optimization model
            A schedule that has been solved for load mismatches
        """
        return value(lmp_sced_instance.dual[lmp_sced_instance.EnforceReserveRequirements[1]])

    @staticmethod
    def get_observed_bus_LMP(lmp_sced_instance, bus):
        """
        Gets the LMP for a single bus

        Parameters
        ----------
        lmp_sced_instance: schedule optimization model
            A schedule that has been solved for load mismatches

        bus: bus entity
            The bus whose LMP is wanted
        """
        return value(lmp_sced_instance.dual[lmp_sced_instance.PowerBalance[bus, 1]])

    @staticmethod
    def get_all_observed_bus_LMPs(lmp_sced_instance):
        """
        Gets a dictionary where keys are buses and values are the LMP for the bus

        There is one entry for each bus in the passed in schedule

        Parameters
        ----------
        lmp_sced_instance: schedule optimization model
            A schedule that has been solved for load mismatches
        """
        return {b: LoadMismatchStatsExtractor.get_observed_bus_LMP(lmp_sced_instance, b)
                for b in lmp_sced_instance.Buses}


class RucStatsExtractor:
    """
    Extracts information from RUC instances
    """

    @staticmethod
    def get_num_time_periods(ruc) -> int:
        ''' Get the number of time periods for which data is available.
            
            Time periods are numbered 1..N, where N is the value returned by this method.
        '''
        return value(ruc.NumTimePeriods)

    @staticmethod
    def get_buses(ruc) -> Iterable[B]:
        return ruc.Buses

    @staticmethod
    def get_bus_demand(ruc, bus: B, time: int) -> float:
        return value(ruc.Demand[bus,time])

    @staticmethod
    def get_nondispatchable_generators(ruc) -> Iterable[G]:
        return ruc.AllNondispatchableGenerators

    @staticmethod
    def get_min_nondispatchable_power(ruc, gen: G, time: int) -> float:
        return value(ruc.MinNondispatchablePower[gen,time])

    @staticmethod
    def get_max_nondispatchable_power(ruc, gen: G, time: int) -> float:
        return value(ruc.MaxNondispatchablePower[gen,time])

