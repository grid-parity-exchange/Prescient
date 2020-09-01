#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from typing import Sequence, Dict, Any
from dataclasses import dataclass, field

from prescient.stats.daily_stats import DailyStats

@dataclass(init=False)
class OverallStats(object):
    """Statistics for the optimization as a whole"""

    # Stats for days within the simulation (starts out empty, grows as more days are simulated)
    daily_stats: Sequence[DailyStats]

    cumulative_demand: float = 0.0
    total_overall_fixed_costs: float = 0.0
    total_overall_generation_costs: float = 0.0
    total_overall_costs: float #implemented as read-only property
    total_overall_over_generation: float = 0.0
    total_overall_load_shedding: float = 0.0
    cumulative_renewables_used: float = 0.0
    total_overall_reserve_shortfall: float = 0.0
    total_overall_renewables_curtailment: float = 0.0
    total_on_offs: int = 0
    total_sum_on_off_ramps: float = 0.0
    total_sum_nominal_ramps: float = 0.0
    total_quick_start_additional_costs: float = 0.0
    total_quick_start_additional_power_generated: float = 0.0

    overall_renewables_penetration_rate:float #implemented as read-only property
    cumulative_average_price: float #implemented as read-only property

    max_hourly_demand: float = 0.0

    total_thermal_energy_payments: float = 0.0
    total_renewable_energy_payments: float = 0.0
    total_energy_payments: float # implemented as read-only property

    total_reserve_payments: float = 0.0

    total_thermal_uplift_payments: float = 0.0
    total_renewable_uplift_payments: float = 0.0
    total_uplift_payments: float # implemented as read-only property

    total_payments: float # implemented as read-only property
    cumulative_average_payments: float # implemented as read-only property

    extensions: Dict[Any, Any]

    @property
    def total_overall_costs(self):
        return self.total_overall_fixed_costs + self.total_overall_generation_costs

    @property
    def cumulative_average_price(self):
        if self.cumulative_demand == 0:
            return 0.0
        return self.total_overall_costs / self.cumulative_demand

    @property
    def overall_renewables_penetration_rate(self):
        if self.cumulative_demand == 0:
            return 0.0
        return (self.cumulative_renewables_used / self.cumulative_demand) * 100.0

    @property
    def cumulative_average_payments(self):
        return self.total_payments / self.cumulative_demand if self.cumulative_demand > 0.0 else 0.0

    @property
    def total_energy_payments(self):
        return self.total_thermal_energy_payments + self.total_renewable_energy_payments

    @property
    def total_uplift_payments(self):
        return self.total_thermal_uplift_payments + self.total_renewable_uplift_payments

    @property
    def total_payments(self):
        return self.total_energy_payments + self.total_uplift_payments

    def __init__(self, options):
        self.daily_stats = []
        self.extensions = {}
        self._options = options

    def incorporate_day_stats(self, day_stats: DailyStats):
        self.daily_stats.append(day_stats)

        self.cumulative_demand += day_stats.this_date_demand
        self.total_overall_fixed_costs += day_stats.this_date_fixed_costs
        self.total_overall_generation_costs += day_stats.this_date_variable_costs
        self.total_overall_load_shedding += day_stats.this_date_load_shedding
        self.total_overall_over_generation += day_stats.this_date_over_generation
        self.cumulative_renewables_used += day_stats.this_date_renewables_used
        self.total_overall_reserve_shortfall += day_stats.this_date_reserve_shortfall
        self.total_overall_renewables_curtailment += day_stats.this_date_renewables_curtailment
        self.total_on_offs += day_stats.this_date_on_offs
        self.total_sum_on_off_ramps += day_stats.this_date_sum_on_off_ramps
        self.total_sum_nominal_ramps += day_stats.this_date_sum_nominal_ramps
        self.total_quick_start_additional_costs += day_stats.this_date_quick_start_additional_costs
        self.total_quick_start_additional_power_generated += day_stats.this_date_quick_start_additional_power_generated
        self.max_hourly_demand = max(self.max_hourly_demand, day_stats.max_hourly_demand)

        self.total_thermal_energy_payments += day_stats.this_date_thermal_energy_payments
        self.total_renewable_energy_payments += day_stats.this_date_renewable_energy_payments
        self.total_reserve_payments += day_stats.this_date_reserve_payments
        self.total_thermal_uplift_payments += day_stats.this_date_thermal_uplift
        self.total_renewable_uplift_payments += day_stats.this_date_renewable_uplift
