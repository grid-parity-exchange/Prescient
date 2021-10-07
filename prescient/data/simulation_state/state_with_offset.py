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
    from prescient.engine.abstract_types import G, S, EgretModel
    from prescient.data.simulation_state.simulation_state import SimulationState

import itertools
import math

from .simulation_state import SimulationState
from . import _helper

class StateWithOffset(SimulationState):
    ''' Get the expected state some number of time periods from the current state.

    The offset state is identical to the state being offset, except that time periods
    before the offset time are skipped, and the initial state of generators and storage
    is provided by a sced instance.  The sced instance is also offset, so that the 
    initial state comes from the Nth time period of the sced.
    '''

    def __init__(self, parent_state:SimulationState, sced:EgretModel, offset:int):
        ''' Constructor.

        Arguments
        ---------
        parent_state:
            The state to project into the future.
        sced:
            A sced instance whose state after the offset is used as the initial state.
        offset:
            The number of time periods into the future this state should reflect.
        '''
        self._parent = parent_state
        self._offset = offset
        self._set_initial_state_from_sced(sced, offset)

    def _set_initial_state_from_sced(self, sced:EgretModel, sced_hour:int):
        self._init_gen_state = {}
        self._init_power_gen = {}
        self._init_soc = {}
        
        sced_index = sced_hour-1

        for gen_state in _helper.get_generator_states_at_sced_offset(self._parent,
                                                                     sced, sced_index):
            g = gen_state.generator
            self._init_gen_state[g] = gen_state.status
            self._init_power_gen[g] = gen_state.power_generated

        for s,soc in _helper.get_storage_socs_at_sced_offset(sced, sced_index):
            self._init_soc[s] = soc


    @property
    def timestep_count(self) -> int:
        ''' The number of timesteps we have data for '''
        return self._parent.timestep_count - self._offset

    def get_generator_commitment(self, g:G, time_index:int) -> int:
        ''' Get whether the generator is committed to be on (1) or off (0) 
        '''
        return self._parent.get_generator_commitment(g, time_index+self._offset)

    def get_initial_generator_state(self, g:G):
        ''' Get the generator's state in the previous time period '''
        return self._init_gen_state[g]

    def get_initial_power_generated(self, g:G):
        ''' Get how much power was generated in the previous time period '''
        return self._init_power_gen[g]

    def get_initial_state_of_charge(self, s:S):
        ''' Get state of charge in the previous time period '''
        return self._init_soc[s]

    def get_current_actuals(self, forecastable:str) -> float:
        ''' Get the current actual value for a forecastable data item

        Arguments
        ---------
        forecastable:str
            The unique identifier for the forecastable data item of interest,
            as returned by forecast_helper.get_forecastables()

        Returns
        -------
        Returns the actual value for the current time period (time index 0).
        '''
        return self._parent.get_future_actuals(forecastable)[self._offset]

    def get_forecasts(self, forecastable:str) -> Sequence[float]:
        ''' Get the forecast values for a forecastable

        Arguments
        ---------
        forecastable:str
            The unique identifier for the forecastable data item of interest,
            as returned by forecast_helper.get_forecastables()

        Returns
        -------
        Returns an array of forecast values, starting with the forecast
        for the current time at index 0.
        '''
        # Copy the relevent portion to a new array
        return list(itertools.islice(self._parent.get_forecasts(forecastable), self._offset, None))

    def get_future_actuals(self, forecastable:str) -> Sequence[float]:
        ''' Warning: Returns actual values of a forecastable for the current time AND FUTURE TIMES.

        Arguments
        ---------
        forecastable:str
            The unique identifier for the forecastable data item of interest,
            as returned by forecast_helper.get_forecastables()

        Returns
        -------
        Returns an array of actual values, starting with the actual value
        for the current time at index 0. All values beyond index 0 are actual
        values for future time periods, which cannot be known at the current time.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options.
        '''
        # Copy the relevent portion to a new array
        return list(itertools.islice(self._parent.get_future_actuals(forecastable), self._offset, None))
