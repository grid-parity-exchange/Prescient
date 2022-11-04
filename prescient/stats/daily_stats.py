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

    # These are cumulative scalars
    demand: float = 0.0
    power_generated: float = 0.0
    fixed_costs: float = 0.0
    variable_costs: float = 0.0
    total_costs: float # implemented as read-only property
    over_generation: float = 0.0
    load_shedding: float = 0.0
    reserve_shortfall: float = 0.0
    renewables_available: float = 0.0
    renewables_used: float = 0.0
    renewables_penetration_rate: float # implemented as a read-only property
    renewables_curtailment: float = 0.0
    on_offs: int = 0
    sum_on_off_ramps: float = 0.0
    sum_nominal_ramps: float = 0.0
    quick_start_additional_costs: float = 0.0
    quick_start_additional_power_generated: float = 0.0
    average_price: float #implemented as read-only property


    # These variables are only populated if options.compute_market_settlements is True
    thermal_energy_payments: float = 0.0
    renewable_energy_payments: float = 0.0
    virtual_energy_payments: float = 0.0

    energy_payments: float #implemented as read-only property

    reserve_payments: float = 0.0

    thermal_uplift: float = 0.0
    renewable_uplift: float = 0.0
    virtual_uplift: float = 0.0

    uplift_payments: float #implemented as read-only property

    total_payments: float #implemented as read-only property

    extensions: Dict[Any, Any]

    @property
    def total_costs(self):
        return self.fixed_costs + self.variable_costs

    @property
    def average_price(self):
        return 0.0 if self.demand == 0.0 else self.total_costs / self.demand

    @property 
    def renewables_penetration_rate(self):
        return 0.0 if self.power_generated == 0.0 else \
                (self.renewables_used / self.power_generated) * 100.0

    @property
    def energy_payments(self):
        return self.thermal_energy_payments + \
                self.renewable_energy_payments + \
                self.virtual_energy_payments

    @property
    def uplift_payments(self):
        return self.thermal_uplift + self.renewable_uplift + self.virtual_uplift

    @property
    def total_payments(self):
        return self.energy_payments + self.uplift_payments + self.reserve_payments

    @property
    def average_payments(self):
        return 0.0 if self.demand == 0.0 else self.total_payments / self.demand

    def operations_stats(self):
        ''' Return any operations_stats in this day.

            Note that operations_stats are discarded just after the day's stats are
            published, so this property only returns data for the current day.
        '''
        yield from (opstat for hrstat in self.hourly_stats for opstat in hrstat.operations_stats)

    def __init__(self, options, day: date):
        self.date = day
        self.hourly_stats = []
        self.extensions = {}
        self._options = options

    def incorporate_hour_stats(self, hourly_stats: HourlyStats):
        self.thermal_fleet_capacity = hourly_stats.thermal_fleet_capacity
        self.hourly_stats.append(hourly_stats)
        self.max_hourly_demand = max(self.max_hourly_demand, hourly_stats.total_demand)

        self.demand += hourly_stats.total_demand
        self.power_generated += hourly_stats.power_generated
        self.fixed_costs += hourly_stats.fixed_costs
        self.variable_costs += hourly_stats.variable_costs
        self.over_generation += hourly_stats.over_generation
        self.load_shedding += hourly_stats.load_shedding
        self.reserve_shortfall += hourly_stats.total_reserve_shortfall
        self.renewables_available += hourly_stats.renewables_available
        self.renewables_used += hourly_stats.renewables_used
        self.renewables_curtailment += hourly_stats.renewables_curtailment
        self.on_offs += hourly_stats.on_offs
        self.sum_on_off_ramps += hourly_stats.sum_on_off_ramps
        self.sum_nominal_ramps += hourly_stats.sum_nominal_ramps
        self.quick_start_additional_costs += hourly_stats.quick_start_additional_costs
        self.quick_start_additional_power_generated += hourly_stats.quick_start_additional_power_generated

        if self._options.compute_market_settlements:
            self.thermal_energy_payments += hourly_stats.thermal_energy_payments
            self.renewable_energy_payments += hourly_stats.renewable_energy_payments
            self.virtual_energy_payments += hourly_stats.virtual_energy_payments

            self.reserve_payments += hourly_stats.reserve_payments

            if hourly_stats.hour == 23:
                final_ops = hourly_stats.operations_stats[-1]
                for g in hourly_stats.thermal_gen_revenue.keys():
                    revenue = sum(stat.thermal_gen_revenue[g] + stat.get_per_gen_reserve_revenue(g)
                                  for stat in self.hourly_stats)
                    costs = sum(stat.observed_costs[g] for stat in self.hourly_stats)
                    final_ops.thermal_uplift[g] = max(costs - revenue, 0.0)

            self.thermal_uplift += hourly_stats.thermal_uplift_payments
            self.renewable_uplift += hourly_stats.renewable_uplift_payments
            self.virtual_uplift += hourly_stats.virtual_uplift_payments

    def finalize_day(self):
        ''' Indicate that the day is done.

        Once the day is finalized, its hourly and operations data is no longer available.
        '''
        self.hourly_stats = ()
