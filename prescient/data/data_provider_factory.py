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

from enum import Enum, auto

class InputFormats(Enum):
    RTS_GMLC = auto()
    DAT = auto()
    SHORTCUT = auto()

def get_data_provider(options:Options) -> DataProvider:
    if options.input_format == InputFormats.RTS_GMLC:
        from .providers.gmlc_data_provider import GmlcDataProvider
        return GmlcDataProvider(options)

    elif options.input_format == InputFormats.DAT:
        from .providers.dat_data_provider import DatDataProvider
        return DatDataProvider(options)

    elif options.input_format == InputFormats.SHORTCUT:
        from .providers.shortcut_data_provider import ShortcutDataProvider
        return ShortcutDataProvider(options)

    else:
        raise ValueError(f"Invalid input format: {options.input_format}")
        return None
