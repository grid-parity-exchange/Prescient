Plugins
=======

Plugins provide opportunities for custom code to observe or modify simulation
data at specific points in the simulation lifecycle. Plugins are python
modules that include a specific set of functions that enable Prescient to
interact with the plugin module. Plugins are specified on the command line, or
in the options passed to the Prescient `simulate()` method if running Prescient
in code.

Plugin modules must include a registration function, through which the plugin
requests that custom code be called at specific points in the simulation process.
Each point at which custom code may be called is known as a *plugin point*. The
function that is called at a plugin point is known as a *callback*.

Plugin points come in two flavors: statistics plugin points and simulation plugin
points. Statistics points allow plugins to view statistics at various stages of
the simulation. Simulation plugin points provide a more detailed view of specific
steps within the simulation; some provide opportunities to customize simulation
behavior.

.. _Identifying Plugins:

Identifying Plugins
-------------------

Any plugins that will be included in a Prescient simulation must be specified
with the :ref:`--plugin<config_plugin>` simulation option. The syntax for this
option is a little different than other options, and is best explained by example.

Every plugin in a particular Prescient run is given an alias. This is the name by
which the plugin will be identified in the run. It determines where the plugin's
configuration options will be stored in the configuration object, and may be used
to give the plugin's custom options a unique name on the command line.

Plugins are identified by path to the python module (.py file), or by python
module name as it would appear in a python `imports` statement. If the module is
specified by module name, it must be able to be found by python's module import
system, such as being located in the `PYTHONPATH`.

The command line syntax to include a plugin is `--plugin \<alias\>:\<path or module
name\>`. For example, the following partial command line will include two plugin
modules, one specified by relative path and one specified by module name::

	python -m prescient.simulator --plugin plug1:custom/plugin1.py --plugin plug2:custom.plugin2 <etc...>

If a plugin defines new configuration options, values can be provided for the new options
anywhere after the plugin has been specified::

    python -m --plugin myplug:custom_plugin.py --custom-opt 100 <etc...>

When configuring Prescient from code, assign a nested dictionary to the `plugin`
element of the configuration options object. The following code example is
equivalent to the previous command line example:

.. code-block:: python

    from prescient.simulator import Prescient

    p = Prescient()
    config = p.config
    config.plugin = {
        'myplug':{
            'module':'custom_plugin.py', 
            'custom_opt':100
        }
    }
    # ...additional configuration ommitted
    p.simulate()

The outer dictionary assigned to the plugin option holds one entry per plugin.
Each entry's key is the plugin's alias, while the entry's value is a dictionary
holding the plugin's data. At a minimum, a plugin's data dictionary must include
a `'module'` element identifying the python module. If the plugin module defines
custom options, values for those options may be supplied as additional dictionary
entries. Values can also be set on separate lines of code:

.. code-block:: python

    config.plugin = {
        'myplug':{
            'module':'custom_plugin.py', 
        }
    }
    config.plugin.myplug.custom_opt = 100

.. _Plugin Module Initialization:

Plugin Module Initialization
----------------------------

A plugin module must have two functions with specific names and signatures.
Prescient initializes each plugin module by calling these two required functions
before the simulation starts, in the order listed below.

get_configuration()
...................

The `get_configuration()` function allows plugins to add custom options to the
Prescient configuration. Once a plugin has defined custom options, those options
can be set on the command line or in code just like standard configuration options.

The `get_configuration()` function must have the following signature:

.. code:: python

   def get_configuration(key: str): -> Optional[pyomo.common.config.ConfigDict]

The `key` is the plugin's alias specified in the configuration. The key may be
incorporated into the text of custom options.

The function should return a pyomo ConfigDict containing any custom options, or
None if the plugin has no custom options.

register_plugins()
..................

The `register_plugins()` function allows a plugin to indicate what callback
functions should be called, and at what plugin points.

The `register_plugins()` function must have the following signature:

.. code:: python

   def register_plugins(context: pplugins.PluginRegistrationContext,
                        options: PrescientConfig,
                        plugin_config: ConfigDict) -> None:

The `context` is an object used to register callbacks. The `context` object
has a registration function for each plugin point. Each registration function
takes a function (or other Callable) as an argument. A plugin's implementation
of `register_plugins()` should call the `context` object's registration
method for each plugin point of interest, passing in the callback function
to be called at the corresponding plugin point.

For example, the code below requests that a function named `my_stats_callback`
be called every time daily statistics are published:

.. code:: python

   context.register_for_daily_stats(my_stats_callback)

Registration function names follow a pattern that embeds the name of the plugin
point. The pattern used to name plugin registration functions differs for statistics
callbacks and simulation callbacks. For statistics callbacks, the pattern is
`register_for_<which>_stats()`, where *which* is the desired time frame. For
simulation callbacks, the pattern is `register_<which>_callback()`, where
*which* is the name of the plugin point.

The `options` argument is the full set of configuration options for the simulation.

The `plugin_config` is the plugin's custom options as defined by what was returned
from the plugin's `get_configuration()` method, with their values as set from
the command line or in code. It is the same as what is found at `options.plugins.<alias>`,
where `<alias>` is the plugin's alias passed to the `get_configuration()`
function earlier.

Statistics Plugin Points
------------------------

Statistics plugin points allow plugins to see statistics
about the simulation. Statistics are published at various time scales.

.. _plugin-operations_stats:

Operations Statistics
.....................

Operations statistics are published at the end of every timestep. They report the
results of a single SCED.

.. _plugin-hourly_stats:

Hourly Statistics
.................

Hourly statistics are published after the final timestep of every hour They report the
aggregate results of all SCEDs within the hour.

.. _plugin-daily_stats:

Daily Statistics
................

Daily statistics are published after the final timestep of every day (the last
timestep before midnight). They report the aggregate results of all SCEDs within
the day.

.. _plugin-overall_stats:

Overall Statistics
..................

Overal statistics are published after the final timestep of the simulation. They report
aggregate results for the full simulation.

Simulation Plugin Points
------------------------

Each plugin point occurs at a different place in the simulation process, and
serves a different purpose.

.. _plugin-options_preview:

The options_preview Callback
............................

This callback is called after command line options have been parsed, but before
they have been used to initialize simulation objects. The callback may modify
option values.

.. _plugin-initialization:

The initialization Callback
...........................

This callback is called after Prescient simulation objects have been created and
initialized. The callback may choose to initialize its own data structures at this
time.

.. _plugin-after_get_initial_model_for_simulation_actuals:

The after_get_initial_model_for_simulation_actuals Callback
...........................................................

Prescient manages actual values by periodically loading from the input data source
into an Egret model. This callback is called after an Egret model has been prepared
to hold actual values, but before the values have been loaded. The structure of the
model will be in place - network elements like generators and branches will be
present - but values will not have been loaded yet. The callback can insert any
non-standard elements and actual values it may use. The callback should not
populate values normally provided by Prescient, as those values will be overwritten
after this callback returns.

.. _plugin-after_get_initial_model_for_ruc:

The after_get_initial_model_for_ruc Callback
............................................

Prescient manages forecasts by periodically loading them into an Egret model from
the input data source. This callback is called after an Egret model has been
prepared to hold forecast values but before the values have been loaded. The
structure of the model will be in place - network elements like generators and
branches will be present - but values will not have been loaded yet. The callback
can insert any non-standard elements and forecast values it may use. The callback
should not insert forecast values normally provided by Prescient, as those values
will be overwritten after this callback returns.

.. _plugin-before_ruc_solve:

The before_ruc_solve Callback
.............................

This callback is called after an Egret model has been fully prepared for a RUC
and is about to be solved. The callback may modify the Egret model.

.. _plugin-after_ruc_generation:

The after_ruc_generation Callback
.................................

This callback is called after a RUC model has been solved. The callback is able
to see (and potentially modify) the resulting RUC plan.

.. _plugin-after_ruc_activation:

The after_ruc_activation Callback
.................................

This callback is called at the beginning of the effective period of a new RUC.
Unit commitment decisions made by the newly activated RUC will be honored until
the next time the *after_ruc_activation* callback is called.

.. _plugin-after_get_initial_model_for_sced:

The after_get_initial_model_for_sced Callback
.....................................................

This callback is called as a SCED model is being prepared. When this callback is
called, the structure of the model will be in place - network elements like
generators and branches will be present - but values will not have been loaded yet.
The callback can insert any non-standard elements and values it may use. The
callback should not insert values normally provided by Prescient, as those values
will be overwritten after this callback returns.

.. _plugin-before_operations_solve:

The before_operations_solve Callback
...........................................

This callback is called after a fully populated SCED Egret model has been generated,
before the model has been solved. The callback may modify the Egret model.

.. _plugin-after_operations:

The after_operations Callback
....................................

This callback is called after an Egret SCED model has been solved. The callback can
examine (and potentially modify) the results.

.. _plugin-update_operations_stats:

The update_operations_stats Callback
....................................

This callback is called after an Egret SCED model has been solved and examined by
any :ref:`plugin-after_operations` callbacks, just before the results are
incorporated into statistics.

.. _plugin-finalization:

The finalization Callback
.........................

This callback is called after the simulation is complete. It gives plugins a chance
to cleanly shut down.
