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
    from prescient.simulator.options import Options
    from typing import Dict, Any

import os.path
from datetime import datetime, timedelta
import copy

from egret.parsers import rts_gmlc_parser as parser
from egret.data.model_data import ModelData as EgretModel

from ..data_provider import DataProvider
from prescient.engine import forecast_helper

class GmlcDataProvider(DataProvider):
    ''' Provides data for RTS-GMLC like files
    '''

    def __init__(self, options:Options):
        # See how much extra time to parse for lookahead horizon
        sced_extra_minutes = options.sced_horizon * options.sced_frequency_minutes
        ruc_extra_minutes = (options.ruc_horizon - options.ruc_every_hours)*60
        extra_minutes = max(sced_extra_minutes, ruc_extra_minutes)

        # Move minutes to days as needed
        days=options.num_days
        while extra_minutes >= 24*60:
            days += 1
            extra_minutes -= 24*60

        # Start at midnight of start date
        self._start_time = datetime.combine(options.start_date, datetime.min.time())
        self._end_time = self._start_time + timedelta(days=days, minutes=extra_minutes)
        self._cache = parser.parse_to_cache(options.data_path, self._start_time, self._end_time, honor_lookahead=False)

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
        # This provider can only return data at multiples of the underlying data's frequency.
        native_frequency = self._cache.minutes_per_period['REAL_TIME']
        if desired_frequency_minutes % native_frequency == 0:
            return desired_frequency_minutes
        else:
            return native_frequency

    def get_initial_actuals_model(self, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        return self._get_initial_model("REAL_TIME", options, num_time_steps, minutes_per_timestep)

    def get_initial_forecast_model(self, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        return self._get_initial_model("DAY_AHEAD", options, num_time_steps, minutes_per_timestep)

    def _get_initial_model(self, sim_type:str, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        ''' Get a model ready to be populated with data

        Returns
        -------
        A model object populated with static system information, such as
        buses and generators, and with time series arrays that are large
        enough to hold num_time_steps entries.

        Initial values in time time series do not have meaning.
        '''
        data = self._cache.get_new_skeleton()
        data['system']['time_period_length_minutes'] = minutes_per_timestep
        data['system']['time_keys'] = [str(i) for i in range(1,num_time_steps+1)]
        md = EgretModel(data)
        self._ensure_forecastable_storage(sim_type, num_time_steps, md)
        return md

    def populate_initial_state_data(self, options:Options,
                                    model: EgretModel) -> None:
        ''' Populate an existing model with initial state data for the first day

        Sets T0 information from actuals:
          * initial_state_of_charge for each storage element
          * initial_status for each generator
          * initial_p_output for each generator

        Arguments
        ---------
        options:
            Option values
        model: EgretModel
            The model whose values will be modifed
        '''
        # RTS-GMLC models come with initial state data in the skeleton,
        # so initial state is already there, nothing to do.
        pass

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
            The duration of each time step
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
        self._populate_with_forecastable_data('DAY_AHEAD', start_time, num_time_periods, time_period_length_minutes, model)

    def populate_with_actuals(self, options:Options,
                              start_time:datetime,
                              num_time_periods: int,
                              time_period_length_minutes: int,
                              model: EgretModel
                             ) -> None:
        ''' Populate an existing model with actuals data.

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
            which data will be provided
        num_time_periods: int
            The number of time steps for which actuals data will be provided.
        time_period_length_minutes: int
            The duration of each time step
        model: EgretModel
            The model where actuals data will be stored

        Notes
        -----
        This will store actuals data in the model's existing data arrays, starting
        at index 0.  If the model's arrays are not big enough to hold all the
        requested time steps, only those steps for which there is sufficient storage
        will be saved.  If arrays are larger than the number of requested time 
        steps, the remaining array elements will be left unchanged.

        Note that this method has the same signature as populate_with_forecast_data.
        '''
        self._populate_with_forecastable_data('REAL_TIME', start_time, num_time_periods, time_period_length_minutes, model)

    def _populate_with_forecastable_data(self,
                                         sim_type:str,
                                         start_time:datetime,
                                         num_time_periods: int,
                                         time_period_length_minutes: int,
                                         model: EgretModel
                                        ) -> None:
        end_time = start_time + timedelta(minutes=num_time_periods*time_period_length_minutes)
        native_frequency = self._cache.minutes_per_period[sim_type]
        step_ratio = time_period_length_minutes // native_frequency
        
        if step_ratio == 1 and len(model.data['system']['time_keys']) == num_time_periods:
            self._cache.populate_skeleton_with_data(model.data, sim_type, start_time, end_time)
        else:
            copy_from = self._cache.generate_model(sim_type, start_time, end_time)
            _recurse_copy_at_ratio(copy_from.data, model.data, step_ratio)

        # Fill in the times
        time_labels = model.data['system']['time_keys']
        delta = timedelta(minutes=time_period_length_minutes)
        for i in range(len(time_labels)):
            dt = start_time + i*delta
            time_labels[i] = dt.strftime('%Y-%m-%d %H:%M')

    def _get_forecastable_locations(self, simulation_type:str, md:EgretModel) -> Iterable[Tuple[dict, str]]:
        ''' Get all recognized forecastable locations with a defined time series
        
        Each location is returned as a dict and the name of a key within the dict
        '''
        return self._cache.get_timeseries_locations(simulation_type, md)

    def _ensure_forecastable_storage(self, sim_type:str, num_entries:int, model:EgretModel) -> None:
        """ Ensure that the model has an array allocated for every type of forecastable data
        """
        for data, key in self._get_forecastable_locations(sim_type, model):
            if (key not in data or \
                type(data[key]) is not dict or \
                data[key]['data_type'] != 'time_series' or \
                len(data[key]['values'] != num_entries)
               ):
                data[key] = { 'data_type': 'time_series',
                              'values': [None]*num_entries}

def _recurse_copy_at_ratio(src:dict[str, Any], target:dict[str, Any], ratio:int) -> None:
    ''' Copy every Nth value from a src dict's time_series values into corresponding arrays in a target dict.
    '''
    for key, att in src.items():
        if isinstance(att, dict):
            if 'data_type' in att and att['data_type'] == 'time_series':
                src_vals = att['values']
                if type(target[key]) is dict and target[key]['data_type'] == 'time_series':
                    # If there is already a value array at the target, fill it in (to preserve array size)
                    target_vals = target[key]['values']
                    for s,t in zip(range(0, len(src_vals), ratio), range(len(target_vals))):
                        target_vals[t] = src_vals[s]
                else:
                    # Otherwise create a new timeseries array
                    target[key] = { 'data_type': 'time_series',
                                    'values' : [src_vals[i] for i in range(0, len(src_vals), ratio)] }
            else:
                _recurse_copy_at_ratio(att, target[key], ratio)
