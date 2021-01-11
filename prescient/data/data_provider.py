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
    from ..engine.abstract_types import EgretModel
    import datetime

from abc import ABC, abstractmethod


class DataProvider(ABC):
    '''
    Provides access to data needed by the simulation
    '''

    @abstractmethod
    def initialize(self, options: Options) -> None:
        ''' Do one-time initial setup
        '''
        pass

    @abstractmethod
    def negotiate_data_frequency(self, desired_frequency_minutes:int):
        ''' Get the number of minutes between each timestep of actuals data this provider will supply,
            given the requested frequency.

            Arguments
            ---------
            desired_frequency_minutes:int
                The number of minutes between actual values that the application would like to get
                from the data provider.

            Returns
            -------
            Returns the number of minutes between each timestep of data.

            The data provider may be able to provide data at different frequencies.  This method allows the 
            data provider to select an appropriate frequency of data samples, given a requested data frequency.
            
            Note that the frequency indicated by this method only applies to actuals data; estimates are always
            hourly.
        '''
        pass

    @abstractmethod
    def get_initial_model(self, options:Options, num_time_steps:int) -> EgretModel:
        ''' Get a model ready to be populated with data

        Returns
        -------
        A model object populated with static system information, such as
        buses and generators, and with time series arrays that are large
        enough to hold num_time_steps entries.

        Initial values in time time series do not have meaning.
        '''
        pass

    @abstractmethod
    def populate_initial_state_data(self, options:Options,
                                    day:date,
                                    model: EgretModel) -> None:
        ''' Populate an existing model with initial state data for the requested day

        Sets T0 information from actuals:
            * initial_state_of_charge for each storage element
            * initial_status for each generator
            * initial_p_output for each generator

        Arguments
        ---------
        options:
            Option values
        day:date
            The day whose initial state will be saved in the model
        model: EgretModel
            The model whose initial state data will be modifed
        '''
        pass

    @abstractmethod
    def populate_with_forecast_data(self, options:Options,
                                    start_time:datetime,
                                    num_time_periods: int,
                                    time_period_length_minutes: int,
                                    model: EgretModel
                                   ) -> None:
        ''' Populate an existing model with forecast data.

        Populates the following values for each requested time period:
            * demand for each bus
            * min and max non-dispatchable power for each non-dispatchable generator
            * reserve requirement
            
        Arguments
        ---------
        options:
            Option values
        start_time: datetime
            The time (day, hour, and minute) of the first time step for
            which forecast data will be provided
        num_time_periods: int
            The number of time steps for which forecast data will be provided.
        time_period_length_minutes: int
            The number of minutes between each time step
        model: EgretModel
            The model where forecast data will be stored

        Notes
        -----
        This will store forecast data in the model's existing data arrays, starting
        at index 0.  If the model's arrays are not big enough to hold all the
        requested time steps, only those steps for which there is sufficient storage
        will be saved.  If arrays are larger than the number of requested time 
        steps, the remaining array elements will be left unchanged.

        Note that this method has the same signature as populate_with_actuals.
        '''
        pass

    @abstractmethod
    def populate_with_actuals(self, options:Options,
                              start_time:datetime,
                              num_time_periods: int,
                              time_period_length_minutes: int,
                              model: EgretModel
                             ) -> None:
        ''' Populate an existing model with actual values.

        Populates the following values for each requested time period:
            * demand for each bus
            * min and max non-dispatchable power for each non-dispatchable generator
            * reserve requirement
            
        Arguments
        ---------
        options:
            Option values
        start_time: datetime
            The time (day, hour, and minute) of the first time step for
            which actual data will be provided
        num_time_periods: int
            The number of time steps for which actual data will be provided.
        time_period_length_minutes: int
            The number of minutes between each time step
        model: EgretModel
            The model where actuals data will be stored

        Notes
        -----
        This will store actuals data in the model's existing data arrays, starting
        at index 0.  If the model's arrays are not big enough to hold all the
        requested time steps, only those steps for which there is sufficient storage
        will be saved.  If arrays are larger than the number of requested time 
        steps, the remaining array elements will be left unchanged.

        Actuals data is always taken from the file matching the date of the time step.
        In other words, only the first 24 hours of each actuals file will ever be
        used.  If this isn't what you want, you'll need to handle that yourself.

        Note that this method has the same signature as populate_with_actuals.
        '''
        pass
