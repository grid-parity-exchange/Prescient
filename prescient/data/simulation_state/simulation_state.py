#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.engine.abstract_types import S,G
    from typing import Iterable, Sequence

class SimulationState(ABC):
    ''' Provides information known at the current simulation time '''
    
    @property
    @abstractmethod
    def timestep_count(self) -> int:
        ''' The number of timesteps we have data for '''
        pass

    @abstractmethod
    def get_generator_commitment(self, g:G, time_index:int) -> int:
        ''' Get whether the generator is committed to be on (1) or off (0) 
        '''
        pass

    @abstractmethod
    def get_initial_generator_state(self, g:G):
        ''' Get the generator's state in the previous time period '''
        pass

    @abstractmethod
    def get_initial_power_generated(self, g:G):
        ''' Get how much power was generated in the previous time period '''
        pass

    @abstractmethod
    def get_initial_state_of_charge(self, s:S):
        ''' Get state of charge in the previous time period '''
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_future_actuals(self, forecastable:str) -> Sequence[float]:
        ''' Warning: Returns actual values of forecastable for the current time AND FUTURE TIMES.

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
        pass
