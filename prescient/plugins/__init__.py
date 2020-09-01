from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .internal import PluginCallbackManager

def get_active_plugin_manager() -> PluginCallbackManager:
    '''
    Get the active plugin manager, creating one if necessary
    '''
    from . import internal
    if internal.active_plugin_manager is None:
        internal.active_plugin_manager = internal.PluginCallbackManager()
    return internal.active_plugin_manager

from .plugin_registration import *

