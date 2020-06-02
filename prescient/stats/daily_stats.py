#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from dataclasses import dataclass, field
from typing import TypeVar, Dict, Sequence, Tuple
from datetime import date

from prescient.stats.hourly_stats import HourlyStats

G = TypeVar('G')
L = TypeVar('L')
B = TypeVar('B')
S = TypeVar('S')

@dataclass(init=False)
class DailyStats:
    """Statistics for a full day of simulation"""

    date: date

    # Stats for hours within this day (starts out empty, grows as more hours are completed in simulation)
    hourly_stats: Sequence[HourlyStats]

    thermal_fleet_capacity: float = 0.0
    max_hourly_demand: float = 0.0

    ############## These are now included in the hourly stats##########################
    ## These are arrays of hourly values, with values appended (arrays start out empty)
    #sced_runtimes: Sequence[float] = field(default_factory=list)
    #event_annotations: Sequence[float] = field(default_factory=list)
    #curtailments_by_hour: Sequence[float] = field(default_factory=list)
    #load_shedding_by_hour: Sequence[float] = field(default_factory=list)
    #over_generation_by_hour: Sequence[float] = field(default_factory=list)
    #reserve_requirements_by_hour: Sequence[float] = field(default_factory=list)
    #reserve_RT_price_by_hour: Sequence[float] = field(default_factory=list)
    #reserve_shortfalls_by_hour: Sequence[float] = field(default_factory=list)
    #available_reserves_by_hour: Sequence[float] = field(default_factory=list)
    #available_quickstart_by_hour: Sequence[float] = field(default_factory=list)
    #fixed_quick_start_generators_committed: Sequence[float] = field(default_factory=list)
    #unfixed_quick_start_generators_committed: Sequence[float] = field(default_factory=list)
    #quick_start_additional_costs_by_hour: Sequence[float] = field(default_factory=list)
    #quick_start_additional_power_generated_by_hour: Sequence[float] = field(default_factory=list)

    # These are cumulative scalars
    this_date_demand: float = 0.0
    this_date_fixed_costs: float = 0.0
    this_date_variable_costs: float = 0.0
    this_date_total_costs: float # implemented as read-only property
    this_date_over_generation: float = 0.0
    this_date_load_shedding: float = 0.0
    this_date_reserve_shortfall: float = 0.0
    this_date_renewables_available: float = 0.0
    this_date_renewables_used: float = 0.0
    this_date_renewables_penetration_rate: float # implemented as a read-only property
    this_date_renewables_curtailment: float = 0.0
    this_date_on_offs: int = 0
    this_date_sum_on_off_ramps: float = 0.0
    this_date_sum_nominal_ramps: float = 0.0
    this_date_quick_start_additional_costs: float = 0.0
    this_date_quick_start_additional_power_generated: float = 0.0
    this_date_average_price: float #implemented as read-only property

    ############### These are included in the hourly stats #####################
    ## These are indexed by a model entity, mapping to an array of 24 hourly values
    #observed_thermal_dispatch_levels: Dict[G, Sequence[float]]
    #observed_thermal_headroom_levels: Dict[G, Sequence[float]]
    #observed_renewables_levels: Dict[G, Sequence[float]]
    #observed_renewables_curtailment: Dict[G, Sequence[float]]
    #observed_thermal_states: Dict[G, Sequence[int]]
    #observed_costs: Dict[G, Sequence[float]]
    #observed_flow_levels: Dict[L, Sequence[float]]
    #observed_bus_mismatches: Dict[B, Sequence[float]]
    #observed_bus_LMPs: Dict[B, Sequence[float]]
    #storage_input_dispatchlevelsdict: Dict[S, Sequence[float]]
    #storage_output_dispatchlevelsdict: Dict[S, Sequence[float]]
    #storage_soc_dispatchlevelsdict: Dict[S, Sequence[float]]

    # This variable only has data if options.enable_quick_start_generator_commitment is True.
    # It is indexed by generator, mapping to an array of hourly values (which are 0 or 1).
    # The dictionary starts out with an entry per quick start generator, but the arrays
    # start out empty and are appended to each hour.
    ########### Commented out because this data is found in the hourly stats ##############
    #used_as_quick_start: Dict[G, Sequence[float]] = field(default_factory=dict)

    # These variables are only populated if options.compute_market_settlements is True
    # They are indexed by a (model entity, hour) tuple, and start out empty.
    ######### They are commented out for now #######################
    #this_date_planning_energy_prices: Dict[Tuple[B, int], float] = field(default_factory=dict)
    #this_date_planning_reserve_prices: Dict[int, float] = field(default_factory=dict)
    #this_date_planning_thermal_generation_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)
    #this_date_planning_thermal_reserve_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)
    #this_date_planning_renewable_generation_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)

    @property
    def this_date_total_costs(self):
        return self.this_date_fixed_costs + self.this_date_variable_costs

    @property
    def this_date_average_price(self):
        return 0.0 if self.this_date_demand == 0.0 else self.this_date_total_costs / self.this_date_demand

    @property 
    def this_date_renewables_penetration_rate(self):
        return (self.this_date_renewables_used / self.this_date_demand) * 100.0

    #def __init__(self, options, ruc):
    def __init__(self, options, day: date):
        self.date = day
        self.hourly_stats = []

        #self.observed_thermal_dispatch_levels = {g: np.repeat(0.0, 24)
        #                                         for g in rc.ThermalGenerators}

        #self.observed_thermal_headroom_levels = {g: np.repeat(0.0, 24)
        #                                         for g in rc.ThermalGenerators}

        #observed_renewables_levels = {g: np.repeat(0.0, 24)
        #                              for g in rc.AllNondispatchableGenerators}

        #observed_renewables_curtailment = {g: np.repeat(0.0, 24)
        #                                   for g in rc.AllNondispatchableGenerators}

        #observed_thermal_states = {g: np.repeat(-1, 24)
        #                           for g in rc.ThermalGenerators}

        #observed_costs = {g: np.repeat(0.0, 24)
        #                  for g in rc.ThermalGenerators}

        #observed_flow_levels = {l: np.repeat(0.0, 24) 
        #                        for l in ruc.TransmissionLines}

        #observed_bus_mismatches = {b: np.repeat(0.0, 24) 
        #                           for b in ruc.Buses}

        #observed_bus_LMPs = {b: np.repeat(0.0, 24) 
        #                     for b in ruc.Buses}

        #storage_input_dispatchlevelsdict = {s: np.repeat(0.0, 24) 
        #                                    for s in ruc.Storage}

        #storage_output_dispatchlevelsdict = {s: np.repeat(0.0, 24) 
        #                                     for s in ruc.Storage}

        #storage_soc_dispatchlevelsdict = {s: np.repeat(0.0, 24) 
        #                                  for s in ruc.Storage}

        #if options.enable_quick_start_generator_commitment:
        #    for g in ruc.QuickStartGenerators:
        #        self.used_as_quick_start[g]=[]

    def incorporate_hour_stats(self, hourly_stats: HourlyStats):
        self.thermal_fleet_capacity = hourly_stats.thermal_fleet_capacity
        self.hourly_stats.append(hourly_stats)
        self.max_hourly_demand = max(self.max_hourly_demand, hourly_stats.total_demand)

        self.this_date_demand += hourly_stats.total_demand
        self.this_date_fixed_costs += hourly_stats.fixed_costs
        self.this_date_variable_costs += hourly_stats.variable_costs
        self.this_date_over_generation += hourly_stats.over_generation
        self.this_date_load_shedding += hourly_stats.load_shedding
        self.this_date_reserve_shortfall += hourly_stats.reserve_shortfall
        self.this_date_renewables_available += hourly_stats.renewables_available
        self.this_date_renewables_used += hourly_stats.renewables_used
        self.this_date_renewables_curtailment += hourly_stats.renewables_curtailment
        self.this_date_on_offs += hourly_stats.on_offs
        self.this_date_sum_on_off_ramps += hourly_stats.sum_on_off_ramps
        self.this_date_sum_nominal_ramps += hourly_stats.sum_nominal_ramps
        self.this_date_quick_start_additional_costs += hourly_stats.quick_start_additional_costs
        self.this_date_quick_start_additional_power_generated += hourly_stats.quick_start_additional_power_generated
