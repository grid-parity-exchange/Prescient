from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optparse import Option
    from typing import Callable
    from prescient.simulator.options import Options
    from prescient.stats import DailyStats, HourlyStats, OverallStats, OperationsStats
    from prescient.simulator.simulator import Simulator
    from prescient.engine.abstract_types import OperationsModel, RucModel
from . import get_active_plugin_manager

def add_custom_commandline_option(option: Option) -> None:
    '''
    To add custom command-line options to Prescient, create a file that
    calls this function and include that file as a plugin on the command
    line (--plugin=my_file).
    '''
    from .internal import active_parser
    active_parser.add_option(option)

   
def register_for_hourly_stats(callback: Callable[[HourlyStats], None]) -> None:
    '''
    Called during plugin registration to request a subscription to hourly stats updates.

    To subscribe to statistics after plugin registration, call the stats_manager directly.
    '''
    from .internal import pending_hourly_subscribers
    pending_hourly_subscribers.append(callback)

def register_for_daily_stats(callback: Callable[[DailyStats], None]) -> None:
    '''
    Called during plugin registration to request a subscription to hourly stats updates

    To subscribe to statistics after plugin registration, call the stats_manager directly.
    '''
    from .internal import pending_daily_subscribers
    pending_daily_subscribers.append(callback)

def register_for_overall_stats(callback: Callable[[OverallStats], None]) -> None:
    '''
    Called during plugin registration to request a subscription to the final overall stats update

    To subscribe to statistics after plugin registration, call the stats_manager directly.
    '''
    from .internal import pending_overall_subscribers
    pending_overall_subscribers.append(callback)

def register_options_preview_callback(callback: Callable[[Options], None]) -> None:
    ''' Request a method be called after options have been parsed, but before they
        have been used to initialize simulation objects.
    '''
    get_active_plugin_manager().register_options_preview_callback(callback)

def register_initialization_callback(callback: Callable[[Options, Simulator], None]) -> None:
    ''' Request a method be called after core prescient objects have been initialized, but
        before the simulation has started.
    '''
    get_active_plugin_manager().register_initialization_callback(callback)

def register_before_ruc_solve_callback(callback: Callable[[Options, Simulator, RucModel, str, int], None]) -> None:
    ''' Register a callback to be called before a ruc model is solved.

        Called after a ruc model is created, just before it is solved.
    '''
    get_active_plugin_manager().register_before_ruc_solve_callback(callback)

def register_after_ruc_generation_callback(callback: Callable[[Options, Simulator, RucPlan, str, int], None]) -> None:
    ''' Register a callback to be called after each new RUC pair is generated.

        The callback is called after both the forecast and actuals RUCs have been 
        generated, just before they are stored in the DataManager.
    '''
    get_active_plugin_manager().register_after_ruc_generation_callback(callback)
    
def register_after_ruc_activation_callback(callback: Callable[[Options, Simulator], None]):
    ''' Register a callback to be called after activating a RUC.

        The callback is called just after the most recently generated RUC
        pair becomes the active RUC pair.  The new RUC instances are
        accessible through the data manager 
        (ex: simulator.data_manager.deterministic_ruc_instance_for_this_period).
    '''
    get_active_plugin_manager().register_after_ruc_activation_callback(callback)

def register_before_operations_solve_callback(callback: Callable[[Options, Simulator, OperationsModel], None]) -> None:
    ''' Register a callback to be called before an operations model is solved.

        Called after an operations model is created, just before it is solved.
    '''
    get_active_plugin_manager().register_before_operations_solve_callback(callback)

def register_after_operations_callback(callback: Callable[[Options, Simulator, OperationsModel], None]) -> None:
    ''' Register a callback to be called after the operations model has been created and
        solved, but before the LMP has been solved, and before statistics have been
        collected.
    '''
    get_active_plugin_manager().register_after_operations_callback(callback)

def register_update_operations_stats_callback(callback: Callable[[Options, Simulator, OperationsStats], None]) -> None:
    ''' Register a callback to be called after intial statistics have been gathered for an 
        solved operations model, but before the statistics have been published.
    '''
    get_active_plugin_manager().register_update_operations_stats_callback(callback)



### Placeholders for future callbacks

def register_before_ruc_generation_callback(callback):
    ''' Register a callback to be called just before each new RUC pair is generated.

        The callback will be called once for each forecast/actuals RUC pair.  It is
        called before the RUC generation process has started.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_before_new_forecast_ruc_callback(callback):
    ''' Register a callback to be called just before a forecast RUC is generated.

        The callback will be called once for each forecast RUC.  It is called after
        initial conditions have been identified (i.e., the projected_sced is available),
        but before the forecast RUC is created and solved.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_before_new_actuals_ruc_callback(callback):
    ''' Register a callback to be called just before an actuals RUC is generated.

        The callback will be called once for each actuals RUC.  It is called after
        the corresponding forecast RUC has been created and solved, but before the 
        actuals RUC is created and solved.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_before_ruc_activation_callback(callback):
    ''' Register a callback to be called before activating a RUC.

        The callback is called just before the most recently generated RUC
        pair becomes the active RUC pair.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_before_operations_callback(callback):
    ''' Register a callback to be called before an operations model is created and solved.

        Called just before an operations model is created and solved.  The active RUC is
        already in place.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_dispatch_level_provider_callback(callback):
    ''' Register a method that provides requested dispatch levels.

        This one has a different flavor than the other callbacks.  Prescient "pulls" the
        requested values from the callback, rather than the callback "pushing" data into
        the model when given the opportunity.  It's here as a discussion point, as some
        have expressed a preference for pull-style code over push-style code.
    '''
    raise RuntimeError("This callback is not yet implemented")

def register_after_lmp_callback(callback):
    ''' Register a callback to be called after the LMP has been calculated.

        Called after the LMP model has been solved, but before operations and LMP
        statistics have been collected.
    '''
    raise RuntimeError("This callback is not yet implemented")
