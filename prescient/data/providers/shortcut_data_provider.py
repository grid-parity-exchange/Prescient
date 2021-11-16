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
from collections import defaultdict
import copy
import csv
import pandas as pd

from egret.parsers import rts_gmlc_parser as parser
from egret.data.model_data import ModelData as EgretModel

from ..data_provider import DataProvider
from .gmlc_data_provider import _recurse_copy_at_ratio
from prescient.engine import forecast_helper

class ShortcutDataProvider(DataProvider):
    ''' Provides data for shortcut simulator
    '''

    def __init__(self, options:Options):
                                            # midnight start
        self._start_time = datetime.combine(options.start_date, datetime.min.time())
        self._end_time = self._start_time + timedelta(days=options.num_days)

        # TODO: option-drive
        self._virtual_bus_capacity = 1e6

        self._generator_characteristics = _load_generator_characteristics(options.data_path)
        self._historical_prices, self._frequency_minutes = \
                _load_historical_prices(options.data_path, self._start_time, self._end_time)

        self._initial_model = { 'elements' : { 'bus' : {'virtual_bus':{}},
                                               'generator' : self._generator_characteristics,
                                             },
                                             'system' : {'baseMVA':100.,
                                                         'load_mismatch_cost':10000.,
                                                         'reserve_shortfall_cost':5000.,},
                              }
        self._initial_model['elements']['generator']['system'] = { 'p_min' : -self._virtual_bus_capacity,
                                                                   'p_max' :  self._virtual_bus_capacity,
                                                                   'p_cost' : { 'data_type' : 'time_series',
                                                                                'values' : None },
                                                                   'generator_type' : 'virtual',
                                                                   'bus' : 'virtual_bus' }

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
        native_frequency = self._frequency_minutes
        if desired_frequency_minutes % native_frequency == 0:
            return desired_frequency_minutes
        else:
            return native_frequency

    def get_initial_forecast_model(self, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        return self._get_initial_model(options, num_time_steps, minutes_per_timestep)

    def get_initial_actuals_model(self, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        return self._get_initial_model(options, num_time_steps, minutes_per_timestep)

    def _get_initial_model(self, options:Options, num_time_steps:int, minutes_per_timestep:int) -> EgretModel:
        ''' Get a model ready to be populated with data
        Returns
        -------
        A model object populated with static system information, such as
        buses and generators, and with time series arrays that are large
        enough to hold num_time_steps entries.

        Initial values in time time series do not have meaning.
        '''
        data = copy.deepcopy(self._initial_model)
        data['system']['time_period_length_minutes'] = minutes_per_timestep
        data['system']['time_keys'] = [str(i) for i in range(1,num_time_steps+1)]

        def _ensure_timeseries_allocated(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    if 'data_type' in v and v['data_type'] == 'time_series':
                        v['values'] = [None]*num_time_steps
                    else:
                        _ensure_timeseries_allocated(v)
        _ensure_timeseries_allocated(data)

        return EgretModel(data)

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
        self._populate_with_forecastable_data('day_ahead', start_time, num_time_periods, time_period_length_minutes, model)

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
        self._populate_with_forecastable_data('real_time', start_time, num_time_periods, time_period_length_minutes, model)

    def _populate_with_forecastable_data(self,
                                         sim_type:str,
                                         start_time:datetime,
                                         num_time_periods: int,
                                         time_period_length_minutes: int,
                                         model: EgretModel
                                        ) -> None:
                                                                                     # pandas is inclusive on ranges
        end_time = start_time + timedelta(minutes=num_time_periods*time_period_length_minutes) - timedelta(seconds=1)

        price_data = self._historical_prices[sim_type][start_time:end_time].to_list()

        if sim_type == 'day_ahead':
            step_ratio = 1
        else:
            step_ratio = time_period_length_minutes // self._frequency_minutes

        if step_ratio == 1 and len(model.data['system']['time_keys']) == num_time_periods:
            p_cost = model.data['elements']['generator']['system']['p_cost']['values'] 
            for i, val in enumerate(price_data):
                p_cost[i] = val
        else:
            copy_from = model.clone()
            copy_from.data['elements']['generator']['system']['p_cost']['values'] = price_data
            # if the data is at a higher frequency, it is more sensible
            # to average over the time periods for prices
            _recurse_average_at_ratio(copy_from.data, model.data, step_ratio)

        # Fill in times:
        time_labels = model.data['system']['time_keys']
        delta = timedelta(minutes=time_period_length_minutes)
        for i in range(len(time_labels)):
            dt = start_time + i*delta
            time_labels[i] = dt.strftime('%Y-%m-%d %H:%M')

def _load_generator_characteristics(data_directory):
    
    # hack the _read_generators function in the RTS-GMLC parser
    elements = {'generator': {},
            'bus' : {'virtual_bus':{'area':'virtual_area', 'zone':'virtual_zone'}},
            'area': {'virtual_area':{}},
            'zone': {'virtual_zone':{}},
            }
    bus_id_to_names = defaultdict(lambda: 'virtual_bus')
    parser._read_generators(data_directory, elements, bus_id_to_names)

    md = {'elements':elements}
    # looks for initial_status.csv itself
    parser.set_t0_data(md, data_directory)

    if os.path.exists(os.path.join(data_directory, 'shortcut_gens.csv')):
        msg_path = "ONLY"

        gens = []
        with open(os.path.join(data_directory, 'shortcut_gens.csv'), newline='') as csvfile:
            genreader = csv.reader(csvfile)
            for r in genreader:
                gens.extend(r)

        gen_dict = {}
        for g in gens:
            gen_dict[g] = elements['generator'][g]

    else:
        msg_path = "ALL"
        gen_dict = elements['generator']

    print("Shortcut Simulator loading "+msg_path+" generator(s) " + \
            ",".join(gen_dict.keys()) + f" from {os.path.join(data_directory, 'gen.csv')}")

    return gen_dict

def _load_historical_prices(data_directory, start_time, end_time):
    def parse_prices( fn ):
        fn = os.path.join(data_directory, fn)
        df = pd.read_csv(fn, parse_dates=True, index_col=0)
        return df[df.columns[0]]

    day_ahead = parse_prices(os.path.join(data_directory, 'day_ahead_prices.csv'))[start_time:end_time]
    real_time = parse_prices(os.path.join(data_directory, 'real_time_prices.csv'))[start_time:end_time]

    # for the day-ahead, only store hourly prices
    day_ahead = day_ahead.asfreq('H').copy()

    # get the real-time minutes as the space between first two data points
    real_time_minutes = (real_time.index[1] - real_time.index[0]).seconds//60

    # check against last periods
    assert real_time_minutes == ((real_time.index[-1] - real_time.index[-2]).seconds//60)

    return {'day_ahead':day_ahead, 'real_time':real_time}, real_time_minutes

def _recurse_average_at_ratio(src:dict[str, Any], target:dict[str, Any], ratio:int) -> None:
    ''' Average every N-1th to Nth value from a src dict's time_series values into corresponding arrays in a target dict.
    '''
    for key, att in src.items():
        if isinstance(att, dict):
            if 'data_type' in att and att['data_type'] == 'time_series':
                src_vals = att['values']
                if type(target[key]) is dict and target[key]['data_type'] == 'time_series':
                    # If there is already a value array at the target, fill it in (to preserve array size)
                    target_vals = target[key]['values']
                    for s,t in zip(range(0, len(src_vals), ratio), range(len(target_vals))):
                        target_vals[t] = sum(src_vals[s:s+ratio])/ratio
                else:
                    # Otherwise create a new timeseries array
                    target[key] = { 'data_type': 'time_series',
                                    'values' : [ sum(src_vals[s:s+ratio])/ratio for i \
                                                    in range(0, len(src_vals), ratio)] }
            else:
                _recurse_average_at_ratio(att, target[key], ratio)
