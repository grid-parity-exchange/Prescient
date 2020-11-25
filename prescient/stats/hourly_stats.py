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
    from typing import Dict, Sequence, TypeVar, Any
    from prescient.engine.data_extractors import ScedDataExtractor
    from prescient.engine.abstract_types import OperationsModel, G, L, B, S
    from .operations_stats import OperationsStats

from dataclasses import dataclass, field
from datetime import date

@dataclass(init=False)
class HourlyStats:
    """Statistics for one hour of simulation"""

    date: date
    hour: int

    # Stats for sceds within this hour (starts out empty, grows as more SCEDs are completed in simulation)
    operations_stats: Sequence[OperationsStats]

    sced_count: int # read-only property

    # This is constant throughout the simulation, but we calculate and store it every hour.
    # We can/should move it somewhere else eventually
    thermal_fleet_capacity: float = 0.0

    sced_runtime: float = 0.0
    average_sced_runtime: float = 0.0

    total_demand: float = 0.0
    fixed_costs: float = 0.0
    variable_costs: float = 0.0
    total_costs: float

    power_generated: float = 0.0
    load_shedding: float = 0.0
    over_generation: float = 0.0
    reserve_shortfall: float = 0.0
    available_reserve: float = 0.0
    available_quickstart: float = 0.0

    renewables_available: float = 0.0
    renewables_used: float = 0.0
    renewables_curtailment: float = 0.0

    on_offs: int = 0
    sum_on_off_ramps: float = 0.0
    sum_nominal_ramps: float = 0.0

    price: float = 0.0

    quick_start_additional_costs: float = 0.0
    quick_start_additional_power_generated: float = 0.0
    used_as_quickstart: Dict[G, int]

    event_annotations: Sequence[str]

    observed_thermal_dispatch_levels: Dict[G, float]
    observed_thermal_headroom_levels: Dict[G, float]
    observed_thermal_states: Dict[G, float]
    observed_costs: Dict[G, float]
    observed_renewables_levels: Dict[G, float] 
    observed_renewables_curtailment: Dict[G, float]

    observed_flow_levels: Dict[L, float]

    bus_demands: Dict[B, float]

    observed_bus_mismatches: Dict[B, float]
    observed_bus_LMPs: Dict[B, float]

    storage_input_dispatch_levels: Dict[S, Sequence[float]]
    storage_output_dispatch_levels: Dict[S, Sequence[float]]
    storage_soc_dispatch_levels: Dict[S, Sequence[float]]

    reserve_requirement: float = 0.0
    reserve_RT_price: float = 0.0

    planning_reserve_price: float = 0.0
    planning_energy_prices: Dict[B, float]

    thermal_gen_cleared_DA: Dict[G, float]
    thermal_gen_revenue: Dict[G, float]
    thermal_reserve_cleared_DA: Dict[G, float]
    thermal_reserve_revenue: Dict[G, float]
    thermal_uplift: Dict[G, float]

    renewable_gen_cleared_DA: Dict[G, float]
    renewable_gen_revenue: Dict[G, float]
    renewable_uplift: Dict[G, float]

    thermal_energy_payments: float #read-only property
    renewable_energy_payments: float #read-only property
    thermal_uplift_payments: float #read-only property
    renewable_uplift_payments: float #read-only property
    reserve_payments: float #read-only property

    extensions: Dict[Any, Any]

    @property 
    def sced_count(self) -> int:
        return len(self.operations_stats)

    @property
    def total_costs(self) -> float:
        return self.fixed_costs + self.variable_costs

    @property
    def thermal_energy_payments(self) -> float:
        if self._options.compute_market_settlements:
            return sum(self.thermal_gen_revenue.values())
        return 0.

    @property
    def renewable_energy_payments(self) -> float:
        if self._options.compute_market_settlements:
            return sum(self.renewable_gen_revenue.values())
        return 0.

    @property
    def thermal_uplift_payments(self) -> float:
        if self._options.compute_market_settlements:
            return sum(self.thermal_uplift.values())
        return 0.

    @property
    def renewable_uplift_payments(self) -> float:
        if self._options.compute_market_settlements:
            return sum(self.renewable_uplift.values())
        return 0.

    @property
    def reserve_payments(self) -> float:
        if self._options.compute_market_settlements:
            return sum(self.thermal_reserve_revenue.values())
        return 0.

    def __init__(self, options, day: date, hour: int):
        self._options = options
        self.date = day
        self.hour = hour
        self.operations_stats = []
        self.event_annotations = []
        self.extensions = {}

    def incorporate_operations_stats(self, ops_stats: OperationsStats):
        # This is a constant, doesn't need to be recalculated over and over, keep it here until
        # we decide on a better place to keep it.
        self.thermal_fleet_capacity = ops_stats.thermal_fleet_capacity

        # Add scalar fields
        summing_fields = [
            'sced_runtime',
            'fixed_costs',
            'variable_costs',
            'on_offs',
            'sum_on_off_ramps',
            'sum_nominal_ramps',
            'quick_start_additional_costs',
           ]
        for field in summing_fields:
            val = getattr(ops_stats, field)
            add_to = getattr(self, field)
            setattr(self, field, add_to+val)

        # Add values indexed by a model entity
        keyed_summing_fields = [
            'observed_costs',
            'thermal_gen_revenue',
            'thermal_reserve_revenue',
            'thermal_uplift',
            'renewable_gen_revenue',
            'renewable_uplift',
           ]
        for field in keyed_summing_fields:
            if not hasattr(ops_stats, field):
                continue
            their_dict = getattr(ops_stats, field)

            if hasattr(self, field):
                my_dict = getattr(self, field)
            else:
                my_dict = {}
                setattr(self, field, my_dict)

            for k,val in their_dict.items():
                if k in my_dict:
                    add_to = my_dict[k]
                    my_dict[k] = add_to + val
                else:
                    my_dict[k] = val

        averaging_fields = [
            'total_demand',
            'power_generated',
            'load_shedding',
            'over_generation',
            'reserve_shortfall',
            'available_reserve',
            'available_quickstart',
            'renewables_available',
            'renewables_used',
            'renewables_curtailment',
            'quick_start_additional_power_generated',
            'reserve_requirement',
            'price',
            'reserve_RT_price',
            'planning_reserve_price'
           ]
        for field in averaging_fields:
            val = getattr(ops_stats, field)
            old_sum = getattr(self, field)*self.sced_count
            setattr(self, field, (old_sum+val)/(self.sced_count+1))

        self.average_sced_runtime = \
            (self.average_sced_runtime*self.sced_count + ops_stats.sced_runtime) / (self.sced_count+1)

        keyed_averaging_fields = [
            'observed_thermal_dispatch_levels',
            'observed_thermal_headroom_levels',
            'observed_thermal_states',
            'observed_renewables_levels',
            'observed_renewables_curtailment',
            'observed_flow_levels',
            'bus_demands',
            'observed_bus_mismatches',
            'observed_bus_LMPs',
            'storage_input_dispatch_levels',
            'storage_output_dispatch_levels',
            'storage_soc_dispatch_levels',
            'planning_energy_prices',
            'thermal_gen_cleared_DA',
            'thermal_reserve_cleared_DA',
            'renewable_gen_cleared_DA',
           ]
        for field in keyed_averaging_fields:
            if not hasattr(ops_stats, field):
                continue
            their_dict = getattr(ops_stats, field)

            if hasattr(self, field):
                my_dict = getattr(self, field)
            else:
                my_dict = {}
                setattr(self, field, my_dict)

            for k,val in their_dict.items():
                if k in my_dict:
                    old_sum = my_dict[k]*self.sced_count
                    my_dict[k] = (old_sum+val)/(self.sced_count+1)
                else:
                    my_dict[k] = val

        # Flag which generators were used for quickstart at least once in the hour
        for g,used in ops_stats.used_as_quickstart.items():
            if g in self.used_as_quickstart:
                self.used_as_quickstart[g] = used or self.used_as_quickstart[g]
            else:
                self.used_as_quickstart[g] = used

        self.operations_stats.append(ops_stats)
