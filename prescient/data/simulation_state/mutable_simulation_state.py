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
    from prescient.engine.abstract_types import G, S

from collections import deque
import math

from prescient.engine.forecast_helper import get_forecastables
from . import SimulationState
from . import _helper
from .state_with_offset import StateWithOffset


class MutableSimulationState(SimulationState):
    ''' A simulation state that can be updated with data pulled from RUCs and sceds.
    '''

    def __init__(self):
        self._forecasts = []
        self._actuals = []
        self._commits = {}

        self._init_gen_state = {}
        self._init_power_gen = {}
        self._init_soc = {}

    @property
    def timestep_count(self) -> int:
        ''' The number of timesteps we have data for '''
        return len(self._forecasts[0]) if len(self._forecasts) > 0 else 0

    def get_generator_commitment(self, g:G, time_index:int) -> Sequence[int]:
        ''' Get whether the generator is committed to be on (1) or off (0) for each time period
        '''
        return self._commits[g][time_index]

    def get_initial_generator_state(self, g:G) -> float:
        ''' Get the generator's state in the previous time period '''
        return self._init_gen_state[g]

    def get_initial_power_generated(self, g:G) -> float:
        ''' Get how much power was generated in the previous time period '''
        return self._init_power_gen[g]

    def get_initial_state_of_charge(self, s:S) -> float:
        ''' Get state of charge in the previous time period '''
        return self._init_soc[s]

    def get_current_actuals(self) -> Iterable[float]:
        ''' Get the current actual value for each forecastable.

        This is the actual value for the current time period (time index 0).
        Values are returned in the same order as forecast_helper.get_forecastables,
        but instead of returning arrays it returns a single value.
        '''
        for forecastable in self._actuals:
            yield forecastable[0]

    def get_forecasts(self) -> Iterable[Sequence[float]]:
        ''' Get the forecast values for each forecastable 

        This is very similar to forecast_helper.get_forecastables(); the 
        function yields an array per forecastable, in the same order as
        get_forecastables().

        Note that the value at index 0 is the forecast for the current time,
        not the actual value for the current time.
        '''
        for forecastable in self._forecasts:
            yield forecastable

    def get_future_actuals(self) -> Iterable[Sequence[float]]:
        ''' Warning: Returns actual values for the current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options, such as 
        '''
        for forecastable in self._actuals:
            yield forecastable

    def apply_ruc(self, options, ruc:RucModel) -> None:
        ''' Incorporate a RUC instance into the current state.

        This will save the ruc's forecasts, and for the very first ruc
        this will also save initial state info.

        If there is a ruc delay, as indicated by options.ruc_execution_hour and
        options.ruc_every_hours, then the RUC is applied to future time periods,
        offset by the ruc delay.  This does not apply to the very first RUC, which
        is used to set up the initial simulation state with no offset.
        '''
 
        ruc_delay = -(options.ruc_execution_hour % (-options.ruc_every_hours))

        # If we've never stored forecasts before...
        first_ruc = (len(self._forecasts) == 0)

        if first_ruc:
            # The is the first RUC, save initial state
            for g, g_dict in ruc.elements('generator', generator_type='thermal'):
                self._init_gen_state[g] = g_dict['initial_status']
                self._init_power_gen[g] = g_dict['initial_p_output']
                # Create a queue where we can store generator commitments
                # Fixed length so we can have old values fall of the list
                max_ruc_length = ruc_delay + options.ruc_horizon
                self._commits[g] = deque(maxlen=max_ruc_length)
            for s,s_dict in ruc.elements('storage'):
                self._init_state_of_charge[s] = s_dict['initial_state_of_charge']

        # Now save all generator commitments
        # Keep the first "ruc_delay" commitments from the prior ruc
        for g, g_dict in ruc.elements('generator', generator_type='thermal'):
            commits = self._commits[g]
            # This puts the first "ruc_delay" items at the end of the list.
            # As we add our new items, all other old items will roll off the end of the list.
            commits.rotate(-ruc_delay)

            # Put the new values into the new value queue
            commits.extend(int(round(g_dict['commitment']['values'][t]))
                           for t in range(0,options.ruc_horizon)) 

        # And finally, save forecastables
        _save_forecastables(options, ruc, self._forecasts)


    def apply_actuals(self, options, actuals) -> None:
        ''' Incorporate actuals into the current state.

        This will save the actuals RUC's forecastables. If there is a ruc delay, 
        as indicated by options.ruc_execution_hour and options.ruc_every_hours, 
        then the actuals are applied to future time periods, offset by the ruc delay.
        This does not apply to the very first actuals RUC, which is used to set up the 
        initial simulation state with no offset.
        '''
        _save_forecastables(options, actuals, self._actuals)

    def apply_sced(self, options, sced) -> None:
        ''' Incorporate a sced's results into the current state, and move to the next time period.

        This saves the sced's first time period of data as initial state information,
        and advances the current time forward by one time period.
        '''
        for gen_state in _helper.get_generator_states_at_sced_offset(self, sced, 0):
            g = gen_state.generator
            self._init_gen_state[g] = gen_state.status
            self._init_power_gen[g] = gen_state.power_generated

        for s,soc in _helper.get_storage_socs_at_sced_offset(sced, 0):
            self._init_soc[s] = soc

        # Advance one time period, dropping one period worth of data
        for value_deque in self._forecasts:
            value_deque.popleft()
        for value_deque in self._actuals:
            value_deque.popleft()
        for value_deque in self._commits.values():
            value_deque.popleft()

    def get_projected_state(self, projected_sced:OperationsModel, time_offset:int) -> SimulationState:
        ''' Get the state as it is projected to be some time in the future.

        Arguments
        ---------
        projected_sced:
            A sced with expected generator and storage activity from the current time 
            until the offset future time.
        time_offset:
            The number of time periods into the future that state is desired.

        The projected_sced is considered as starting at the same time as the current state.
        The first "time_offset" time periods of the projected sced are used to find the
        initial state of the future state.

        The returned future state references the state object it was derived from; changes
        in the parent state will cause changes in the future state.
        ''' 
        return StateWithOffset(self, projected_sced, time_offset)




def _save_forecastables(options, ruc, where_to_store):
    first_ruc = (len(where_to_store) == 0)
    ruc_delay = -(options.ruc_execution_hour % (-options.ruc_every_hours))
    max_length = ruc_delay + options.ruc_horizon

    # Save all forecastables, in forecastable order
    for idx, (new_ruc_vals,) in enumerate(get_forecastables(ruc)):
        if first_ruc:
            # append to storage array
            forecast = deque(maxlen=max_length)
            where_to_store.append(forecast)
        else:
            forecast = where_to_store[idx]

            # This puts the first "ruc_delay" items at the end of the list.
            # As we add our new items, all other old items will roll off the end of the list.
            forecast.rotate(-ruc_delay)

        # Put the new values into the new value queue
        forecast.extend(new_ruc_vals) 
