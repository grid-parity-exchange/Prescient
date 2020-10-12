#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
from .manager import _Manager
import os.path

from typing import NamedTuple

class RucPlan(NamedTuple):
   simulation_actuals: RucModel
   deterministic_ruc_instance: RucModel
   ruc_market: [RucMarket, None]

class RucMarket(NamedTuple):
    day_ahead_prices: Dict
    day_ahead_reserve_prices: Dict
    thermal_gen_cleared_DA: Dict
    thermal_reserve_cleared_DA: Dict
    renewable_gen_cleared_DA: Dict

class DataManager(_Manager):
    def initialize(self, engine, options):
        self._ruc_stats_extractor = engine.ruc_data_extractor
        self.prior_sced_instance = None
        self.active_simulation_actuals = None
        self.pending_simulation_actuals = None
        self.active_ruc = None
        self.pending_ruc = None
        self.ruc_market_active = None
        self.ruc_market_pending = None
        self._extensions = {}

    def update_time(self, time):
        '''This takes a Time object and makes the appropiate updates to the data'''
        self._current_time = time

    def set_pending_ruc_plan(self, current_ruc_plan: RucPlan):
        self.pending_simulation_actuals = current_ruc_plan.simulation_actuals
        self.pending_ruc =current_ruc_plan.deterministic_ruc_instance
        self.ruc_market_pending = current_ruc_plan.ruc_market

    def activate_pending_ruc(self, options: Options):
        self.active_simulation_actuals = self.pending_simulation_actuals
        self.active_ruc = self.pending_ruc
        self.ruc_market_active = self.ruc_market_pending

        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next RUC is computed.
        self.set_actuals_for_new_ruc_instance()
        self.set_forecast_errors_for_new_ruc_instance(options)

        self.pending_simulation_actuals = None
        self.pending_ruc = None
        self.ruc_market_pending = None


    def set_forecast_errors_for_new_ruc_instance(self, options) -> None:
        ''' Generate new forecast errors from current ruc instances '''
        forecast_ruc = self.active_ruc
        actuals_ruc = self.active_simulation_actuals
        extractor = self._ruc_stats_extractor

        # print("NOTE: Positive forecast errors indicate projected values higher than actuals")
        self._demand_forecast_error = {}
        for b in extractor.get_buses(forecast_ruc):
            for t in range(1, extractor.get_num_time_periods(forecast_ruc) + 1):
                self._demand_forecast_error[b, t] = \
                    extractor.get_bus_demand(forecast_ruc, b, t) - \
                    extractor.get_bus_demand(actuals_ruc, b, t)

        self._renewables_forecast_error = {}
        for g in extractor.get_nondispatchable_generators(forecast_ruc):
            for t in range(1, extractor.get_num_time_periods(forecast_ruc) + 1):
                self._renewables_forecast_error[g, t] = \
                    extractor.get_max_nondispatchable_power(forecast_ruc, g, t) - \
                    extractor.get_max_nondispatchable_power(actuals_ruc, g, t)


    def update_forecast_errors_for_delayed_ruc(self, options):
        ''' update the demand and renewables forecast error dictionaries, using recently released forecasts '''
        actuals_ruc = self.pending_simulation_actuals
        forecast_ruc = self.pending_ruc
        extractor = self._ruc_stats_extractor

        for b in extractor.get_buses(forecast_ruc):
            for t in range(1, options.ruc_every_hours + 1):
                self._demand_forecast_error[b, t + options.ruc_every_hours] = \
                    extractor.get_bus_demand(forecast_ruc, b, t) - \
                    extractor.get_bus_demand(actuals_ruc, b, t)

        for g in extractor.get_nondispatchable_generators(forecast_ruc):
            for t in range(1, options.ruc_every_hours + 1):
                self._renewables_forecast_error[g, t + options.ruc_every_hours] = \
                    extractor.get_max_nondispatchable_power(forecast_ruc, g, t) - \
                    extractor.get_max_nondispatchable_power(actuals_ruc, g, t)


    def set_actuals_for_new_ruc_instance(self) -> None:
        # initialize the actual demand and renewables vectors - these will be incrementally
        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
        ruc = self.active_simulation_actuals
        extractor = self._ruc_stats_extractor
        times = range(1, extractor.get_num_time_periods(ruc) + 1)

        self._actual_demand = {(b, t): extractor.get_bus_demand(ruc, b, t)
                               for b in extractor.get_buses(ruc)
                               for t in times}

        self._actual_min_renewables = {}
        self._actual_max_renewables = {}
        for g in extractor.get_nondispatchable_generators(ruc):
            for t in times:
                self._actual_min_renewables[g, t] = extractor.get_min_nondispatchable_power(ruc, g, t)
                self._actual_max_renewables[g, t] = extractor.get_max_nondispatchable_power(ruc, g, t)

    def update_actuals_for_delayed_ruc(self, options):
        # update the second 'ruc_every_hours' hours of the current actual demand/renewables vectors
        extractor = self._ruc_stats_extractor
        actuals_ruc = self.pending_simulation_actuals

        for t in range(1, options.ruc_every_hours+1):
            for b in extractor.get_buses(actuals_ruc):
                self._actual_demand[b, t+options.ruc_every_hours] = extractor.get_bus_demand(actuals_ruc, b, t)
            for g in extractor.get_nondispatchable_generators(actuals_ruc):
                self._actual_min_renewables[g, t+options.ruc_every_hours] = extractor.get_min_nondispatchable_power(actuals_ruc, g, t)
                self._actual_max_renewables[g, t+options.ruc_every_hours] = extractor.get_max_nondispatchable_power(actuals_ruc, g, t)


    ##########
    # Properties
    ##########
    @property
    def current_time(self):
        return self._current_time

    @property
    def actual_demand(self):
        return self._actual_demand

    @property
    def actual_min_renewables(self):
        return self._actual_min_renewables

    @property
    def actual_max_renewables(self):
        return self._actual_max_renewables

    @property
    def demand_forecast_error(self):
        return self._demand_forecast_error

    @property
    def renewables_forecast_error(self):
        return self._renewables_forecast_error

    @property
    def extensions(self):
        return self._extensions
