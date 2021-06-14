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
    from .data_provider import DataProvider

valid_input_formats = ('rts-gmlc',
                       'dat')

def get_data_provider(options:Options) -> DataProvider:
    if options.input_format == 'rts-gmlc':
        from .providers.gmlc_data_provider import GmlcDataProvider
        return GmlcDataProvider(options)

    elif options.input_format == 'dat':
        from .providers.dat_data_provider import DatDataProvider
        return DatDataProvider(options)

    else:
        raise ValueError(f"Invalid input format: {options.input_format}")
        return None
