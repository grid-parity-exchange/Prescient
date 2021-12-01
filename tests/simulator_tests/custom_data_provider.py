from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.data.data_provider import DataProvider
    from prescient.simulator.options import Options

from datetime import datetime, timedelta
from dateutil import parser as date_parser
import pandas as pd

from prescient.data.providers.gmlc_data_provider import GmlcDataProvider
from egret.parsers.rts_gmlc.parsed_cache import ParsedCache, ScalarReserveData

def get_data_provider(options:Options) -> DataProvider:
    ''' Get a CustomDataProvider instance '''
    return CustomDataProvider(options)

class CustomDataProvider(GmlcDataProvider):
    def __init__(self, options:Options):
        self._start_time = datetime.combine(options.start_date, datetime.min.time())
        self._end_time = self._start_time + timedelta(days=options.num_days)
        self._cache = self.read_data_to_cache(options.data_path)

    def read_data_to_cache(self, filepath):
        import json
        from dateutil import parser

        # Read the JSON file
        with open(filepath) as f:
            data = json.load(f)

        # Convert Series arrays into pd.Series objects with time indices
        series_indices = {key:  [date_parser.parse(val) for val in data['timeseries_indices'][key]]
                          for key in data['timeseries_indices']}
        series_data = data['timeseries_data']['Series']
        sim_types = data['timeseries_data']['Simulation']
        for i in range(len(series_data)):
            index = series_indices[sim_types[i]]
            series_data[i] = pd.Series(series_data[i], index=index)

        # Convert the whole set of timeseries data to a DataFrame
        ts_df = pd.DataFrame(data['timeseries_data'])

        return ParsedCache(data['skeleton'], 
                           parser.parse(data['begin_time']),
                           parser.parse(data['end_time']),
                           data['minutes_per_day_ahead_period'], 
                           data['minutes_per_real_time_period'],
                           ts_df,
                           data['load_participation_factors'],
                           ScalarReserveData(data['scalar_reserve_data']['da_scalars'],
                                             data['scalar_reserve_data']['rt_scalars'])
                           )



def parsed_cache_to_json(f, cache):
    import json
    import numpy as np

    ts_df = cache.timeseries_df

    cur_sim = ts_df['Simulation'].iat[0]
    first_indices = {cur_sim:0}
    for i in range(1, len(ts_df)):
        if ts_df['Simulation'].iat[i] != cur_sim:
            cur_sim = ts_df['Simulation'].iat[i]
            first_indices[cur_sim] = i

    s = {'skeleton':cache.skeleton,
         'begin_time':str(cache.begin_time),
         'end_time':str(cache.end_time),
         'minutes_per_day_ahead_period':cache.minutes_per_period['DAY_AHEAD'],
         'minutes_per_real_time_period':cache.minutes_per_period['REAL_TIME'],
         'timeseries_data':cache.timeseries_df.to_dict('list'),
         'timeseries_indices':{ key: np.datetime_as_string(
                                        ts_df['Series'].iat[first_indices[key]].index.values, 'm'
                                     ).tolist()
                                for key in first_indices},
         'load_participation_factors':cache.load_participation_factors,
         'scalar_reserve_data':{'da_scalars':cache.scalar_reserve_data.da_scalars,
                                'rt_scalars':cache.scalar_reserve_data.rt_scalars}
    }
    s['timeseries_data']['Series'] = [series.values.tolist() 
                                      for series in s['timeseries_data']['Series']]
    json.dump(s, f)
