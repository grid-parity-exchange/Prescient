from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .internal import PluginCallbackManager
    from typing import Callable
    from prescient.simulator.options import Options
    from prescient.stats import DailyStats, HourlyStats, OverallStats, OperationsStats
    from prescient.simulator.simulator import Simulator
    from prescient.engine.abstract_types import OperationsModel, RucModel
    from prescient.simulator.config import PrescientConfig
from pyomo.common.fileutils import import_file

class PluginRegistrationContext:
    ''' Object that plugins and other code use to register their callbacks.

    This class provides a registration-centric view of a PluginCallbackManager.
    Despite its name, it can be used by more than just plugins. Plugins modules
    must supply a function called register_plugins; its implementation should
    register appropriate callbacks on the PluginRegistrationContext provided as
    an argument to that function. Non-plugin code that wishes to receive callbacks
    should call the appropriate registration functions as part of simulation setup.
    '''

    def __init__(self):
        from .internal import PluginCallbackManager
        self._callback_manager = PluginCallbackManager()

    def register_plugin(
        self, 
        plugin_module:'module', 
        config: PrescientConfig,
        plugin_config: ConfigDict) -> None:
        ''' Allow a plugin module to register for callbacks

        This method calls the module's register_plugins method, passing
        this registration context as an argument. The register_plugins
        method should use the context to register whatever callbacks are
        appropriate for the plugin. The method is also provided with the
        PrescientConfig to be used in the simulation, as well as the 
        plugin-specific portion of the configuration.
        '''
        register_func = getattr(plugin_module, "register_plugins", None)
        if register_func is None:
            raise RuntimeError(f"plugin module={plugin_module} does not have a required method, register_plugins")
        register_func(self, config, plugin_config)

    def register_for_operations_stats(
        self, 
        callback: Callable[[OperationsStats], None]
        ) -> None:
        '''
        Called during plugin registration to request a subscription to operations (sced) stats updates.

        To subscribe to statistics after plugin registration, call the stats_manager directly.
        '''
        self.callback_manager.register_operations_stats_callback(callback)

    def register_for_hourly_stats(
        self, 
        callback: Callable[[HourlyStats], None]
        ) -> None:
        '''
        Called during plugin registration to request a subscription to hourly stats updates.

        To subscribe to statistics after plugin registration, call the stats_manager directly.
        '''
        self.callback_manager.register_hourly_stats_callback(callback)

    def register_for_daily_stats(
        self, 
        callback: Callable[[DailyStats], None]
        ) -> None:
        '''
        Called during plugin registration to request a subscription to hourly stats updates

        To subscribe to statistics after plugin registration, call the stats_manager directly.
        '''
        self.callback_manager.register_daily_stats_callback(callback)

    def register_for_overall_stats(
        self, 
        callback: Callable[[OverallStats], None]
        ) -> None:
        '''
        Called during plugin registration to request a subscription to the final overall stats update

        To subscribe to statistics after plugin registration, call the stats_manager directly.
        '''
        self.callback_manager.register_overall_stats_callback(callback)

    def register_options_preview_callback(
        self, 
        callback: Callable[[Options], None]
        ) -> None:
        ''' Request a method be called after options have been parsed, but before they
            have been used to initialize simulation objects.
        '''
        self.callback_manager.register_options_preview_callback(callback)

    def register_initialization_callback(
        self, 
        callback: Callable[[Options, Simulator], None]
        ) -> None:
        ''' Request a method be called after core prescient objects have been initialized, but
            before the simulation has started.
        '''
        self.callback_manager.register_initialization_callback(callback)

    def register_after_get_initial_actuals_model_for_sced_callback(
        self,
        callback: Callable[[Options, OperationsModel], None]
        ) -> None:
        ''' Request a method to be called immediately after an actuals model for the sced has
            been generated, but before any data is loaded into it.
        '''
        self.callback_manager.register_after_get_initial_actuals_model_for_sced_callback(callback)

    def register_after_get_initial_forecast_model_for_ruc_callback(
        self,
        callback: Callable[[Options, RucModel], None]
        ) -> None:
        ''' Request a method to be called immediately after an forecast model for the ruc has
            been generated, but before any data is loaded into it.
        '''
        self.callback_manager.register_after_get_initial_forecast_model_for_ruc_callback(callback)

    def register_after_get_initial_actuals_model_for_simulation_actuals_callback(
        self,
        callback: Callable[[Options, RucModel], None]
        ) -> None:
        ''' Request a method to be called immediately after an actuals model for the simulation_actuals has
            been generated, but before any data is loaded into it.
        '''
        self.callback_manager.register_after_get_initial_actuals_model_for_simulation_actuals_callback(callback)

    def register_finalization_callback(
        self,
        callback: Callable[[Options, Simulator], None]
        ) -> None:
        ''' Request a method be called after prescient simulation is over.
        '''
        self.callback_manager.register_finalization_callback(callback)

    def register_before_ruc_solve_callback(
        self, 
        callback: Callable[[Options, Simulator, RucModel, str, int], None]
        ) -> None:
        ''' Register a callback to be called before a ruc model is solved.

            Called after a ruc model is created, just before it is solved.
        '''
        self.callback_manager.register_before_ruc_solve_callback(callback)

    def register_after_ruc_generation_callback(
        self, 
        callback: Callable[[Options, Simulator, RucPlan, str, int], None]
        ) -> None:
        ''' Register a callback to be called after each new RUC plan is generated.

            Called after each RUC has been solved, just before its results
            are stored in the DataManager.
        '''
        self.callback_manager.register_after_ruc_generation_callback(callback)
    
    def register_after_ruc_activation_callback(
        self, 
        callback: Callable[[Options, Simulator], None]
        ):
        ''' Register a callback to be called after activating a RUC.

            The callback is called just after the most recently generated RUC
            becomes active. The RUC becomes active when the simulation reaches
            the time period in which the RUC's commitment decisions are used.
        '''
        self.callback_manager.register_after_ruc_activation_callback(callback)

    def register_before_operations_solve_callback(
        self, 
        callback: Callable[[Options, Simulator, OperationsModel], None]
        ) -> None:
        ''' Register a callback to be called before an operations model is solved.

            Called after an operations model is created, just before it is solved.
        '''
        self.callback_manager.register_before_operations_solve_callback(callback)

    def register_after_operations_callback(
        self, 
        callback: Callable[[Options, Simulator, OperationsModel, OperationsModel], None]
        ) -> None:
        ''' Register a callback to be called after the operations model has been created and
            solved, before statistics have been collected.

            The callback will be called with the following arguments:
                options:Options
                    The options used for this simulation run
                simulator:Simulator
                    The simulator managing the simulation and its data
                sced:OperationsModel
                    The operations model that was just solved
                lmp_sced:OperationsModel
                    The solved LMP model for the operations model that was just solved
        '''
        self.callback_manager.register_after_operations_callback(callback)

    def register_update_operations_stats_callback(
        self,
        callback: Callable[[Options, Simulator, OperationsStats], None]
        ) -> None:
        ''' Register a callback to be called after intial statistics have been gathered for an 
            solved operations model, but before the statistics have been published.

            The operations stats object may be modified by the registered callback.
        '''
        self.callback_manager.register_update_operations_stats_callback(callback)



    ### Placeholders for future callbacks

    def register_before_ruc_generation_callback(self, callback):
        ''' Register a callback to be called just before each new RUC pair is generated.

            The callback will be called once for each forecast/actuals RUC pair.  It is
            called before the RUC generation process has started.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_before_new_forecast_ruc_callback(self, callback):
        ''' Register a callback to be called just before a forecast RUC is generated.

            The callback will be called once for each forecast RUC.  It is called after
            initial conditions have been identified (i.e., the projected_sced is available),
            but before the forecast RUC is created and solved.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_before_new_actuals_ruc_callback(self, callback):
        ''' Register a callback to be called just before an actuals RUC is generated.

            The callback will be called once for each actuals RUC.  It is called after
            the corresponding forecast RUC has been created and solved, but before the 
            actuals RUC is created and solved.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_before_ruc_activation_callback(self, callback):
        ''' Register a callback to be called before activating a RUC.

            The callback is called just before the most recently generated RUC
            pair becomes the active RUC pair.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_before_operations_callback(self, callback):
        ''' Register a callback to be called before an operations model is created and solved.

            Called just before an operations model is created and solved.  The active RUC is
            already in place.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_dispatch_level_provider_callback(self, callback):
        ''' Register a method that provides requested dispatch levels.

            This one has a different flavor than the other callbacks.  Prescient "pulls" the
            requested values from the callback, rather than the callback "pushing" data into
            the model when given the opportunity.  It's here as a discussion point, as some
            have expressed a preference for pull-style code over push-style code.
        '''
        raise RuntimeError("This callback is not yet implemented")

    def register_after_lmp_callback(self, callback):
        ''' Register a callback to be called after the LMP has been calculated.

            Called after the LMP model has been solved, but before operations and LMP
            statistics have been collected.
        '''
        raise RuntimeError("This callback is not yet implemented")

    @property
    def callback_manager(self) -> PluginCallbackManager:
        ''' Get the callback manager that will handle the callbacks registered with this context
        '''
        return self._callback_manager
