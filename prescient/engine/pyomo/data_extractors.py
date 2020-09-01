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
    from typing import Iterable
    from prescient.engine.abstract_types import *

import numpy as np
from pyomo.core import value

from prescient.util.math_utils import round_small_values

from prescient.engine.data_extractors import ScedDataExtractor as BaseScedExtractor
from prescient.engine.data_extractors import RucDataExtractor as BaseRucExtractor


class ScedDataExtractor(BaseScedExtractor):

    def get_buses(self, sced: OperationsModel) -> Iterable[B]:
        return sced.Buses

    def get_transmission_lines(self, sced: OperationsModel) -> Iterable[L]:
        return sced.TransmissionLines

    def get_all_storage(self, sced: OperationsModel) -> Iterable[S]:
        return sced.Storage

    def get_thermal_generators(self, sced: OperationsModel) -> Iterable[G]:
        return sced.ThermalGenerators

    def get_nondispatchable_generators(self, sced: OperationsModel) -> Iterable[G]:
        return sced.AllNondispatchableGenerators

    def get_thermal_generators_at_bus(self, sced: OperationsModel, b: B) -> Iterable[G]:
        return sced.ThermalGeneratorsAtBus[b]

    def get_nondispatchable_generators_at_bus(self, sced: OperationsModel, b: B) -> Iterable[G]:
        return sced.NondispatchableGeneratorsAtBus[b]

    def get_quickstart_generators(self, sced: OperationsModel) -> Iterable[G]:
        return sced.QuickStartGenerators

    def is_generator_on(self, sced: OperationsModel, g: G) -> bool:
        return value(sced.UnitOn[g, 1]) > 0

    def generator_was_on(self, sced: OperationsModel, g: G) -> bool:
        return value(sced.UnitOnT0State[g]) > 0

    def get_fixed_costs(self, sced: OperationsModel) -> float:
        return value(sum(sced.StartupCost[g,1] + sced.ShutdownCost[g,1] 
                         for g in sced.ThermalGenerators) + 
                     sum(sced.UnitOn[g,1] * sced.MinimumProductionCost[g] * sced.TimePeriodLength 
                         for g in sced.ThermalGenerators))

    def get_variable_costs(self, sced: OperationsModel) -> float:
        return value(sced.TotalProductionCost[1])

    def get_power_generated(self, sced: OperationsModel, g: G) -> float:
        return value(sced.PowerGenerated[g,1])

    def get_power_generated_T0(self, sced: OperationsModel, g: G) -> float:
        return value(sced.PowerGeneratedT0[g])

    def get_load_mismatch(self, sced: OperationsModel, b: B) -> float:
        return value(sced.LoadGenerateMismatch[b, 1])

    def get_positive_load_mismatch(self, sced: OperationsModel, b: B):
        return value(sced.posLoadGenerateMismatch[b, 1])

    def get_negative_load_mismatch(self, sced: OperationsModel, b: B):
        return value(sced.negLoadGenerateMismatch[b, 1])

    def get_max_power_output(self, sced: OperationsModel, g: G) -> float:
        return value(sced.MaximumPowerOutput[g])

    def get_max_power_available(self, sced: OperationsModel, g: G) -> float:
        return value(sced.MaximumPowerAvailable[g,1])

    def get_min_downtime(self, sced: OperationsModel, g: G) -> float:
        return sced.MinimumDownTime[g]

    def get_scaled_startup_ramp_limit(self, sced: OperationsModel, g: G) -> float:
        return value(sced.ScaledStartupRampLimit[g])

    def get_reserve_shortfall(self, sced: OperationsModel) -> float:
       return round_small_values(value(sced.ReserveShortfall[1]))

    def get_max_nondispatchable_power(self, sced: OperationsModel, g: G) -> float:
       return value(sced.MaxNondispatchablePower[g, 1])

    def get_nondispatchable_power_used(self, sced: OperationsModel, g: G) -> float:
       return value(sced.NondispatchablePowerUsed[g, 1])

    def get_total_demand(self, sced: OperationsModel) -> float:
        return value(sced.TotalDemand[1])

    def get_reserve_requirement(self, sced: OperationsModel) -> float:
        return value(sced.ReserveRequirement[1])

    def get_generator_cost(self, sced: OperationsModel, g: G) -> float:
        return value(sced.StartupCost[g,1] 
                   + sced.ShutdownCost[g,1]
                   + sced.UnitOn[g,1] * sced.MinimumProductionCost[g] * sced.TimePeriodLength
                   + sced.ProductionCost[g,1])

    def get_flow_level(self, sced: OperationsModel, line: L) -> float:
        return value(sced.LinePower[line, 1])

    def get_bus_mismatch(self, sced: OperationsModel, bus: B) -> float:
        if value(sced.LoadGenerateMismatch[bus, 1]) >= 0.0:
            return value(sced.posLoadGenerateMismatch[bus, 1])
        else:
            return -1.0 * value(sced.negLoadGenerateMismatch[bus, 1])

    def get_storage_input_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
       return value(sced.PowerInputStorage[storage, 1])

    def get_storage_output_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
        return value(sced.PowerOutputStorage[storage, 1])

    def get_storage_soc_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
        return value(sced.SocStorage[storage, 1])

    def get_reserve_RT_price(self, lmp_sced: OperationsModel) -> float:
        return value(lmp_sced.dual[lmp_sced.EnforceReserveRequirements[1]])

    def get_bus_LMP(self, lmp_sced: OperationsModel, bus: B) -> float:
        return value(lmp_sced.dual[lmp_sced.PowerBalance[bus, 1]])

class RucDataExtractor(BaseRucExtractor):
    """
    Extracts information from RUC instances
    """

    def get_num_time_periods(self, ruc: RucModel) -> int:
        ''' Get the number of time periods for which data is available.
            
            Time periods are numbered 1..N, where N is the value returned by this method.
        '''
        return value(ruc.NumTimePeriods)

    def get_buses(self, ruc: RucModel) -> Iterable[B]:
        ''' Get all buses in the model '''
        return ruc.Buses

    def get_bus_demand(self, ruc: RucModel, bus: B, time: int) -> float:
        ''' get the demand on a bus in a given time period '''
        return value(ruc.Demand[bus,time])

    def get_nondispatchable_generators(self, ruc: RucModel) -> Iterable[G]:
        return ruc.AllNondispatchableGenerators

    def get_min_nondispatchable_power(self, ruc: RucModel, gen: G, time: int) -> float:
        return value(ruc.MinNondispatchablePower[gen,time])

    def get_max_nondispatchable_power(self, ruc: RucModel, gen: G, time: int) -> float:
        return value(ruc.MaxNondispatchablePower[gen,time])
