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
    from datetime import datetime
    from prescient.engine.data_extractors import ScedDataExtractor
    from prescient.engine.abstract_types import OperationsModel, G, L, B, S

from dataclasses import dataclass, field

@dataclass(init=False)
class OperationsStats:
    """Statistics for one SCED simulation"""

    timestamp: datetime

    # This is constant throughout the simulation, but we calculate and store it every hour.
    # We can/should move it somewhere else eventually
    thermal_fleet_capacity: float = 0.0

    sced_runtime: float = 0.0

    sced_duration_minutes: int = 0

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
    quick_start_capable: Dict[G, bool]
    used_as_quickstart: Dict[G, int]

    event_annotations: Sequence[str]

    generator_fuels: Dict[G, str]

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
    storage_types: Dict[S, Sequence[float]]

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

    def __init__(self, options, timestamp: datetime):
        self._options = options
        self.timestamp = timestamp
        self.event_annotations = []
        self.extensions = {}

    def populate_from_sced(self, 
                           sced: OperationsModel, 
                           runtime: float, 
                           lmp_sced: OperationsModel, 
                           pre_quickstart_cache: PreQuickstartCache,
                           extractor: ScedDataExtractor):
        self.sced_runtime = runtime
        self.sced_duration_minutes = extractor.get_sced_duration_minutes(sced)

        # This is a constant, doesn't need to be recalculated over and over, keep it here until
        # we decide on a better place to keep it.
        self.thermal_fleet_capacity = extractor.get_fleet_thermal_capacity(sced)

        self.total_demand = extractor.get_total_demand(sced)
        self.fixed_costs = extractor.get_fixed_costs(sced)
        self.variable_costs = extractor.get_variable_costs(sced)

        self.power_generated = extractor.get_total_power_generated(sced)

        self.load_shedding, self.over_generation = extractor.get_load_mismatches(sced)
        if self.load_shedding > 0.0:
            self.event_annotations.append('Load Shedding')
        if self.over_generation > 0.0:
            self.event_annotations.append('Over Generation')

        self.reserve_shortfall = extractor.get_reserve_shortfall(sced)
        if self.reserve_shortfall > 0.0:
            self.event_annotations.append('Reserve Shortfall')

        self.available_reserve = extractor.get_available_reserve(sced)
        self.available_quickstart = extractor.get_available_quick_start(sced)

        self.renewables_available = extractor.get_renewables_available(sced)
        self.renewables_used = extractor.get_renewables_used(sced)
        self.renewables_curtailment = extractor.get_total_renewables_curtailment(sced)

        self.on_offs, self.sum_on_off_ramps, self.sum_nominal_ramps = extractor.get_on_off_and_ramps(sced)

        self.price = extractor.get_price(self.total_demand, self.fixed_costs, self.variable_costs)

        self.quick_start_additional_costs = extractor.get_additional_quickstart_costs(pre_quickstart_cache, sced)
        self.quick_start_additional_power_generated = extractor.get_additional_quickstart_power_generated(pre_quickstart_cache, sced)
        self.quick_start_capable = extractor.get_all_thermal_quickstart_capable_flag(sced)
        self.used_as_quickstart = extractor.get_generator_quickstart_usage(pre_quickstart_cache, sced)

        self.generator_fuels = extractor.get_all_generator_fuels(sced)

        self.observed_thermal_dispatch_levels = extractor.get_all_thermal_dispatch_levels(sced)
        self.observed_thermal_headroom_levels = extractor.get_all_thermal_headroom_levels(sced)
        self.observed_thermal_states = extractor.get_all_thermal_states(sced)
        self.observed_costs = extractor.get_cost_per_generator(sced)
        self.observed_renewables_levels = extractor.get_all_nondispatchable_power_used(sced)
        self.observed_renewables_curtailment = extractor.get_all_renewables_curtailment(sced)

        self.observed_flow_levels = extractor.get_all_flow_levels(sced)

        self.bus_demands = extractor.get_all_bus_demands(sced)        

        self.observed_bus_mismatches = extractor.get_all_bus_mismatches(sced)
        self.observed_bus_LMPs = extractor.get_all_bus_LMPs(lmp_sced)

        self.storage_input_dispatch_levels = extractor.get_all_storage_input_dispatch_levels(sced)
        self.storage_output_dispatch_levels = extractor.get_all_storage_output_dispatch_levels(sced)
        self.storage_soc_dispatch_levels = extractor.get_all_storage_soc_dispatch_levels(sced)
        self.storage_types = extractor.get_all_storage_types(sced)

        self.reserve_requirement = extractor.get_reserve_requirement(sced)

        self.reserve_RT_price = extractor.get_reserve_RT_price(lmp_sced)

    def populate_market_settlement(self,
                                   sced: OperationsModel,
                                   extractor: ScedDataExtractor,
                                   ruc_market: RucMarket,
                                   time_index: int):

        self.planning_reserve_price = ruc_market.day_ahead_reserve_prices[time_index]
        self.planning_energy_prices = { b : ruc_market.day_ahead_prices[b,time_index] \
                                        for b in extractor.get_buses(sced) }

        self.thermal_gen_cleared_DA = { g : ruc_market.thermal_gen_cleared_DA[g,time_index] \
                                        for g in extractor.get_thermal_generators(sced) }

        self.renewable_gen_cleared_DA = { g : ruc_market.renewable_gen_cleared_DA[g,time_index] \
                                          for g in extractor.get_nondispatchable_generators(sced) }

        self.thermal_reserve_cleared_DA = { g : ruc_market.thermal_reserve_cleared_DA[g,time_index] \
                                            for g in extractor.get_thermal_generators(sced) }

        self.thermal_gen_revenue = dict()
        self.renewable_gen_revenue = dict()
        for b in extractor.get_buses(sced):
            price_DA = self.planning_energy_prices[b]
            price_RT = self.observed_bus_LMPs[b]

            for g in extractor.get_thermal_generators_at_bus(sced, b):
                self.thermal_gen_revenue[g] = \
                    (self.thermal_gen_cleared_DA[g]*price_DA + \
                     (self.observed_thermal_dispatch_levels[g] - self.thermal_gen_cleared_DA[g])*price_RT
                    ) * self.sced_duration_minutes / 60

            for g in extractor.get_nondispatchable_generators_at_bus(sced, b):
                self.renewable_gen_revenue[g] = \
                    (self.renewable_gen_cleared_DA[g]*price_DA + \
                     (self.observed_renewables_levels[g] - self.renewable_gen_cleared_DA[g])*price_RT
                    ) * self.sced_duration_minutes / 60


        r_price_DA = self.planning_reserve_price
        r_price_RT = self.reserve_RT_price
        self.thermal_reserve_revenue = { g : (self.thermal_reserve_cleared_DA[g]*r_price_DA + \
                                                ( self.observed_thermal_headroom_levels[g] - \
                                                   self.thermal_reserve_cleared_DA[g] )*r_price_RT
                                             ) * self.sced_duration_minutes / 60
                                         for g in extractor.get_thermal_generators(sced) }

        ## TODO: calculate uplift for the day
        self.thermal_uplift = { g : 0. for g in extractor.get_thermal_generators(sced) }

        ## TODO: calculate uplift for the day
        self.renewable_uplift = { g : 0. for g in extractor.get_nondispatchable_generators(sced) }

