#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from dataclasses import dataclass, field
from typing import TypeVar, Dict, Sequence, Tuple, Any
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

    # These variables are only populated if options.compute_market_settlements is True
    # They are indexed by a (model entity, hour) tuple, and start out empty.
    ######### They are commented out for now #######################
    #this_date_planning_energy_prices: Dict[Tuple[B, int], float] = field(default_factory=dict)
    #this_date_planning_reserve_prices: Dict[int, float] = field(default_factory=dict)
    #this_date_planning_thermal_generation_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)
    #this_date_planning_thermal_reserve_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)
    #this_date_planning_renewable_generation_cleared: Dict[Tuple[G, int], float] = field(default_factory=dict)

    extensions: Dict[Any, Any]

    @property
    def this_date_total_costs(self):
        return self.this_date_fixed_costs + self.this_date_variable_costs

    @property
    def this_date_average_price(self):
        return 0.0 if self.this_date_demand == 0.0 else self.this_date_total_costs / self.this_date_demand

    @property 
    def this_date_renewables_penetration_rate(self):
        return (self.this_date_renewables_used / self.this_date_demand) * 100.0

    def __init__(self, options, day: date):
        self.date = day
        self.hourly_stats = []
        self.extensions = {}

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
