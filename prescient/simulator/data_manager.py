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
    from typing import Dict, Optional, Sequence

import os.path

from typing import NamedTuple
from prescient.data.simulation_state import MutableSimulationState

from .manager import _Manager

class RucMarket(NamedTuple):
    day_ahead_prices: Dict
    DA_reserve_prices: Dict[ReserveProductID, Sequence[float]]
    DA_reserve_requirements: Dict[ReserveProductID, Sequence[float]]
    DA_reserve_shortfalls: Dict[ReserveProductID, Sequence[float]]
    thermal_gen_cleared_DA: Dict
    thermal_reserve_cleared_DA: Dict
    renewable_gen_cleared_DA: Dict
    virtual_gen_cleared_DA: Dict

class RucPlan(NamedTuple):
    deterministic_ruc_instance: RucModel
    ruc_market: Optional[RucMarket]

class DataManager(_Manager):
    def initialize(self, engine, options):
        self.ruc_market_active = None
        self.ruc_market_pending = None
        self._state = MutableSimulationState()
        self._extensions = {}
        self.prior_sced_instance = None

    @property
    def current_state(self):
        return self._state

    def update_time(self, time):
        self._current_time = time

    def apply_sced(self, options, sced):
        self._state.apply_sced(options, sced)
        self.prior_sced_instance = sced

    def apply_forecasts(self, options:Options, model_with_forecasts: EgretModel):
        self._state.apply_forecasts(options, model_with_forecasts)

    def apply_actuals(self, options:Options, model_with_actuals: EgretModel):
        self._state.apply_actuals(options, model_with_actuals)

    def apply_ruc(self, options:Options, ruc_instance:EgretModel):
        self._state.apply_ruc(options, ruc_instance)

    def set_pending_ruc(self, options:Options, pending_ruc_plan:RucPlan):
        self.ruc_market_pending = pending_ruc_plan.ruc_market

    def activate_pending_ruc(self, options: Options):
        self.ruc_market_active = self.ruc_market_pending
        self.ruc_market_pending = None

    ##########
    # Properties
    ##########
    @property
    def current_time(self):
        return self._current_time

    @property
    def extensions(self):
        return self._extensions
