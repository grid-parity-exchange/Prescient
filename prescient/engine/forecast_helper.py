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
    from .abstract_types import EgretModel
    from typing import Iterable, Tuple, MutableSequence, Any

from enum import Enum, auto
from typing import NamedTuple

class InferralType(Enum):
    ''' The method used to infer forecast values past the first 24 hours
    '''
    COPY_FIRST_DAY=auto()
    REPEAT_LAST=auto()

class InferrableForecastable(NamedTuple):
    '''A single forecastable value array, with the method to infer values beyond the first 24 hours
    '''
    inferral_type: InferralType
    forecastable: MutableSequence[float]

def _recurse_into_time_series_values(name:str, data_dict: dict) -> Iterable[MutableSequence[float]]:
    for att_name, att in data_dict.items():
        if isinstance(att, dict):
            if 'data_type' in att and att['data_type'] == 'time_series':
                yield name+'__'+att_name, att['values']
            else:
                _recurse_into_time_series_values(name+'__'+att_name, att)

_egret_element_types = [
        'generator',
        'load',
        'branch',
        'dc_branch',
        'bus',
        'shunt',
        'storage',
        'area',
        'zone',
        'interface',
        'fuel_supply',
        'interchange',
        ]

def get_forecastables(model: EgretModel) -> Iterable[ Tuple[str, MutableSequence[float]] ]:
    ''' Get all data that are predicted by forecasting, for any number of models.

    The iterable returned by this function yields tuples containing one list from each model 
    passed to the function.  Each tuple of lists corresponds to one type of data that is included
    in forecast predictions, such as loads on a particular bus, or limits on a renewable generator.
    The lengths of the lists matches the number of time steps present in the underlying models.
    Modifying list values modifies the underlying model.
    '''
    for element_type in _egret_element_types:
        for name, data in model.elements(element_type):
            yield from _recurse_into_time_series_values(element_type+'__'+name, data)
    yield from _recurse_into_time_series_values('system', model.data['system'])

def get_forecastables_with_inferral_method(model:EgretModel) -> Iterable[InferrableForecastable]:
    """ Get all data predicted by forecasting in a model, with the method used to infer values after the first day
    """
    # Generator is the first element type
    for _, gdata in model.elements('generator'):
        if ('fuel' in gdata and gdata['fuel'].lower() in ('w', 'wind')):
            for _,vals in _recurse_into_time_series_values('',gdata):
                yield InferralType.REPEAT_LAST, vals
        else:
            for _,vals in _recurse_into_time_series_values('',gdata):
                yield InferralType.COPY_FIRST_DAY, vals

    for element_type in _egret_element_types[1:]:
        for _, data in model.elements(element_type):
            for _,vals in _recurse_into_time_series_values('',data):
                yield InferralType.COPY_FIRST_DAY, vals

    for _,vals in _recurse_into_time_series_values('',model.data['system']):
        yield InferralType.COPY_FIRST_DAY, vals
