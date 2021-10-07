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

from egret.model_library.transmission.tx_utils import element_types as _egret_element_types

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

def get_forecastables(model: EgretModel) -> Iterable[ Tuple[str, MutableSequence[float]] ]:
    ''' Get a value array for each property that is a time series in the supplied model.

    The iterable returned by this function yields tuples whose first element is a string uniquely 
    identifying a forecastable data element, and whose second element is the list of time series
    values for that data element. Each list corresponds to a type of data that is included
    in forecast predictions, such as loads on a particular bus, or limits on a renewable generator.
    The length of each list matches the number of time steps present in the underlying model.
    Values in lists can be modified; doing so modifies the underlying model.
    The identifying string should be treated as an opaque value with no meaning 
    '''
    for element_type in _egret_element_types():
        for name, data in model.elements(element_type):
            yield from _recurse_into_time_series_values(element_type+'__'+name, data)
    yield from _recurse_into_time_series_values('system', model.data['system'])

def get_forecastables_with_inferral_method(model:EgretModel) -> Iterable[InferrableForecastable]:
    """ Get each data series predicted by forecasting, with the method used to infer values after the first day
    """
    for element_type in _egret_element_types():
        for _, data in model.elements(element_type):
            # Decide which kind of inferral to do
            inferral = InferralType.COPY_FIRST_DAY
            if (element_type == 'generator' and 
                    'fuel' in data and 
                    data['fuel'].lower() in ('w', 'wind')):
                inferral = InferralType.REPEAT_LAST

            for _,vals in _recurse_into_time_series_values('',data):
                yield inferral, vals

    for _,vals in _recurse_into_time_series_values('',model.data['system']):
        yield InferralType.COPY_FIRST_DAY, vals
