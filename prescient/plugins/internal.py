# Code that is not intended to be called by plugins
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, List
    from prescient.stats import HourlyStats, DailyStats, OverallStats

from typing import NamedTuple

class _StatisticsSubscribers(NamedTuple):
    hourly: List[Callable[[HourlyStats], None]] = list()
    daily: List[Callable[[DailyStats], None]] = list()
    overall: List[Callable[[OverallStats], None]] = list()

callbacks = ['options_preview',
             'update_operations_stats',
             'after_ruc_generation',
             'after_ruc_activation',
             'before_operations_solve',
             'before_ruc_solve',
             'after_operations']

class PluginCallbackManager():
    '''
    Keeps track of what callback methods have been registered, and 
    provides methods to invoke callbacks at appropriate times.
    '''
    def __init__(self):
        for cb in callbacks:
            self._setup_callback(cb)

        self._initialization_callbacks = []

        # stats callbacks registered by plugins but 
        # not yet added as subscribers
        self._pending_stats_subscribers = _StatisticsSubscribers()

    def _setup_callback(self, cb):
        list_name = f'_{cb}_callbacks'
        setattr(self, list_name, list())
        def register_func(callback):
            getattr(self, list_name).append(callback)
        setattr(self, f'register_{cb}_callback', register_func)
        def invoke_this(*args, **kargs):
            for cb in getattr(self, list_name):
                cb(*args, **kargs)
        setattr(self, f'invoke_{cb}_callbacks', invoke_this)

    def clear(self):
        self._pending_stats_subscribers = _StatisticsSubscribers()
        for cb in callbacks:
            list_name = f'_{cb}_callbacks'
            getattr(self, list_name).clear()


    ### Registration methods ###
    def register_initialization_callback(self, callback):
        self._initialization_callbacks.append(callback)

    def register_hourly_stats_callback(self, callback):
        self._pending_stats_subscribers.hourly.append(callback)

    def register_daily_stats_callback(self, callback):
        self._pending_stats_subscribers.daily.append(callback)

    def register_overall_stats_callback(self, callback):
        self._pending_stats_subscribers.overall.append(callback)


    ### Callback invocation methods ###

    def invoke_initialization_callbacks(self, options, simulator):
        for cb in self._initialization_callbacks:
            cb(options, simulator)

        # register stats callbacks as subscribers
        for s in self._pending_stats_subscribers.hourly:
            simulator.stats_manager.register_for_hourly_stats(s)
        self._pending_stats_subscribers.hourly.clear()
        for s in self._pending_stats_subscribers.daily:
            simulator.stats_manager.register_for_daily_stats(s)
        self._pending_stats_subscribers.daily.clear()
        for s in self._pending_stats_subscribers.overall:
            simulator.stats_manager.register_for_overall_stats(s)
        self._pending_stats_subscribers.overall.clear()
