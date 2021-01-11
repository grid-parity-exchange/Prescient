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

import math
from typing import NamedTuple

from prescient.util.math_utils import interpolate_between
from prescient.util.sequence_interpolation import InterpolatingSequence, get_interpolated_index_position, get_interpolatable_index_count
from . import SimulationState

class TimeInterpolatedState(SimulationState):
    ''' Presents a SimulationState with a different time step interval than an underlying state,
        possibly starting after the underlying state's first time step.

    The underlying state is called the inner state. The inner state has two sets of
    time series data, namely forecasts and actuals.  The frequency at which the inner state
    has data for these two series may be different, such as forecasts every hour and actuals
    every 15 minutes.

    The state as it appears through a TimeInterpolatedState instance is called the outer state. 
    The outer state has a single data frequency, call the outer step length.  The TimeInterpolatedState
    presents the inner state's time series data as if it had the outer step length.  For example, if
    the outer step length is 5 minutes, then values for both forecasts and actuals will be provided at
    5 minute intervals even if the inner data is stored with other step lengths.  Inner values are
    interpolated as needed to provide data at the desired frequency.

    The outer state may start later than the inner state.  This is because a single inner time step may
    include multiple outer time steps.  As the simulation progresses, the current simulation time may not
    align with inner step boundaries.  Also note that, the number of minutes past the first inner 
    *forecast* may be different than the number of minutes past the first inner *actuals* data.  For 
    example, consider hourly forecasts and 30 minute actuals.  At 45 minutes into the simulation, the
    inner state may have discarded the actuals for time 0, so the number of minutes past the first
    remaining point in the actuals data series is 15, but the number of minutes since the first forecast
    is 45 minutes.  The TimeInterpolatedState class accommodates this by accepting an offset for each 
    type of data.
    '''

    def __init__(self, inner_state:SimulationState, 
                 minutes_per_inner_forecast:int,
                 minutes_past_first_forecast:int,
                 minutes_per_inner_actuals:int,
                 minutes_past_first_actuals:int,
                 minutes_per_outer_step:int):
        ''' Constructor

        Arguments
        ---------
        inner_state:SimulationState
            The state whose data will be interpolated
        minutes_per_inner_forecast:int
            The number of minutes in each forecast timestep in the inner state
        minutes_past_first_forecast:int
            The number of minutes past the start of the first inner forecast step that the first outer timestep starts 
        minutes_per_inner_actuals:int
            The number of minutes in each actuals timestep in the inner state
        minutes_past_first_actuals:int
            The number of minutes past the start of the first inner actuals step that the first outer timestep starts 
        minutes_per_outer_step:int
            The number of minutes per time step the state appears to have when
            viewed through this TimeInterpolatedState

        The underlying state (called the inner_state) may have two different time scales, one for
        forecasts and one for actuals.  The timestep length we want the state to appear to have (called 
        the outer step length) might match one, both, or neither of the inner step lengths.  If the outer 
        step length doesn't match an inner step length, then the first outer step might start between the
        first two inner steps.  The minutes_past_first_* variables indicate how long after the first inner 
        step the first outer step starts.

        For any outer time step that falls between two inner time steps, the outer step's values are
        calculated by interpolating between the two surrounding inner time step values.

        The initial values of the inner state are simply passed through.  It is only timestep values that
        are interpolated as necessary.
        '''

        self._inner_state = inner_state

        self._minutes_per_inner_forecast = minutes_per_inner_forecast
        self._minutes_past_first_forecast = minutes_past_first_forecast
        self._minutes_per_inner_actuals = minutes_per_inner_actuals
        self._minutes_past_first_actuals = minutes_past_first_actuals        
        self._minutes_per_outer_step = minutes_per_outer_step

    @property
    def timestep_count(self) -> int:
        ''' The number of timesteps for which we have forecast data 

        We only count timesteps that start no later than the inner state's last timestep
        '''
        return get_interpolatable_index_count(
             self._inner_state.timestep_count,
             self._minutes_past_first_forecast,
             self._minutes_per_inner_forecast,
             self._minutes_per_outer_step
            )

    @property
    def minutes_per_step(self) -> int:
        return self._minutes_per_outer_step

    def get_generator_commitment(self, g:G, time_index:int) -> int:
        ''' Get whether the generator is committed to be on (1) or off (0) 
        '''
        # Commitments aren't interpolated, they are repeated throughout the inner time step.
        fractional_index = get_interpolated_index_position(time_index, 
                                                           self._minutes_past_first_forecast,
                                                           self._minutes_per_inner_forecast, 
                                                           self._minutes_per_outer_step)
        return self._inner_state.get_generator_commitment(g, fractional_index.index_before)

    def get_initial_generator_state(self, g:G):
        ''' Get the generator's state in the previous time period '''
        return self._inner_state.get_initial_generator_state(g)

    def get_initial_power_generated(self, g:G):
        ''' Get how much power was generated in the previous time period '''
        return self._inner_state.get_initial_power_generated(g)

    def get_initial_state_of_charge(self, s:S):
        ''' Get state of charge in the previous time period '''
        return self._inner_state.get_initial_state_of_charge(s)

    def get_current_actuals(self) -> Iterable[float]:
        ''' Get the current actual value for each forecastable.

        This is the actual value for the current time period (outer step index 0).
        Values are returned in the same order as forecast_helper.get_forecastables,
        but instead of returning arrays it returns a single value.
        '''
        if self._minutes_past_first_actuals == 0:
            yield from self._inner_state.get_current_actuals()
        else:
            fractional_index = get_interpolated_index_position(0, 
                                                               self._minutes_past_first_actuals,
                                                               self._minutes_per_inner_actuals, 
                                                               self._minutes_per_outer_step)
            for forecastable in self._inner_state.get_future_actuals():
                yield interpolate_between(forecastable[fractional_index.index_before], 
                                          forecastable[fractional_index.index_after],
                                          fractional_index.fraction_between)

    def get_forecasts(self) -> Iterable[Sequence[float]]:
        ''' Get the forecast values for each forecastable 

        This is very similar to forecast_helper.get_forecastables(); the 
        function yields an array per forecastable, in the same order as
        get_forecastables().

        Note that the value at index 0 is the forecast for the current time,
        not the actual value for the current time.
        '''
        return self._get_forecastables(self._inner_state.get_forecasts(),
                                       self._minutes_per_inner_forecast,
                                       self._minutes_past_first_forecast)

    def get_future_actuals(self) -> Iterable[Sequence[float]]:
        ''' Warning: Returns actual values of forecastables for the current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options.
        '''
        return self._get_forecastables(self._inner_state.get_future_actuals(),
                                       self._minutes_per_inner_actuals,
                                       self._minutes_past_first_actuals)

    def _get_forecastables(self, forecastables:Iterable[Sequence[float]],
                           minutes_per_inner_step:int,
                           minutes_past_first:int
                          ) -> Iterable[Sequence[float]]:
        # if we don't have to interpolate...
        if minutes_per_inner_step == self._minutes_per_outer_step and minutes_past_first == 0:
            yield from forecastables

        # Build an InterpolatingSequence for each forecastable
        for f in forecastables:
            yield InterpolatingSequence(f, minutes_per_inner_step, self._minutes_per_outer_step, minutes_past_first)
