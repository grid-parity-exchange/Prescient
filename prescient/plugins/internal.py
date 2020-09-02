# Code that is not intended to be called by plugins

# The parser instance affect by plugins modifications
active_parser = None

# The instance of the PluginCallbackManager that handles
# calls to plugin_registration methods
active_plugin_manager = None

# stats callbacks registered by plugins but not yet
# added as subscribers
pending_hourly_subscribers = []
pending_daily_subscribers = []
pending_overall_subscribers = []

class PluginCallbackManager():
    '''
    Keeps track of what callback methods have been registered, and 
    provides methods to invoke callbacks at appropriate times.
    '''
    def __init__(self):
        callbacks = ['options_preview',
                     'update_operations_stats',
                     'after_ruc_generation',
                     'after_ruc_activation',
                     'before_operations_solve',
                     'before_ruc_solve',
                     'after_operations']
        for cb in callbacks:
            self._setup_callback(cb)

        self._initialization_callbacks = []

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


    ### Registration methods ###
    def register_initialization_callback(self, callback):
        self._initialization_callbacks.append(callback)


    ### Callback invocation methods ###

    def invoke_initialization_callbacks(self, options, simulator):
        for cb in self._initialization_callbacks:
            cb(options, simulator)
        # As part of initialization, we also register stats
        # callbacks as subscribers
        for s in pending_hourly_subscribers:
            simulator.stats_manager.register_for_hourly_stats(s)
        pending_hourly_subscribers.clear()
        for s in pending_daily_subscribers:
            simulator.stats_manager.register_for_daily_stats(s)
        pending_daily_subscribers.clear()
        for s in pending_overall_subscribers:
            simulator.stats_manager.register_for_overall_stats(s)
        pending_overall_subscribers.clear()
