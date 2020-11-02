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
    from typing import Iterable, Tuple, MutableSequence

def get_forecastables(*models: EgretModel) -> Iterable[ Tuple[MutableSequence[float]] ]:
    ''' Get all data that are predicted by forecasting, for any number of models.

    The iterable returned by this function yields tuples containing one list from each model 
    passed to the function.  Each tuple of lists corresponds to one type of data that is included
    in forecast predictions, such as loads on a particular bus, or limits on a renewable generator.
    The lengths of the lists matches the number of time steps present in the underlying models.
    Modifying list values modifies the underlying model.
    '''
    # Renewables limits
    model1 = models[0]
    for gen, gdata1 in model1.elements('generator', generator_type='renewable'):
        yield tuple(m.data['elements']['generator'][gen]['p_min']['values'] for m in models)
        yield tuple(m.data['elements']['generator'][gen]['p_max']['values'] for m in models)

    # Loads
    for bus, bdata1 in model1.elements('load'):
        yield tuple(m.data['elements']['load'][bus]['p_load']['values'] for m in models)

    # Reserve requirement
    yield tuple(m.data['system']['reserve_requirement']['values'] for m in models)

    return
