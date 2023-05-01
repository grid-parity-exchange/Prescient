Configuration Options
=====================

.. contents::
   :local:

Overview
--------

Prescient configuration options are used to indicate how 
the Prescient simulation should be run. Configuration
options can be specified on the command line, in a text
configuration file, or in code, depending on how Prescient
is launched (see :doc:`run`).

Each configuration option has a name, a data type, and a default value.
The name used on the command line and the name used in code vary slightly.
For example, the number of days to simulate is specified as `\-\-num-days`
on the command line, and `num_days` in code. 

Option Data Types
-----------------

Most options use self-explanatory data types like `String`,
`Integer`, and `Float`, but some data types require more
explanation and may be specified in code in ways that are unavailable 
on the command line:

.. list-table:: Configuration Data Types
   :header-rows: 1
   :class: table-top-align-cells table-wrap-headers

   * - Data type
     - Command-line/config file usage
     - In-code usage
   * - Path
     - A text string that refers to a file or folder. Can be
       relative or absolute, and may include special characters
       such as `~`.
     - Same as command-line
   * - Date
     - A string that can be converted to a date, such as *1776-07-04*.
     - Either a string or a datetime object.
   * - Flag
     - Simply include the option to set it to true. For example, the command
       below sets `simulate_out_of_sample` to true::

         runner.py --simulate-out-of-sample

     - Set the option by assigning True or False::

         config.simulate_out_of_sample = True
   * - Module
     - Refer to a python module in one of the following ways:

       * The name of a python module (such as `prescient.simulator.prescient`)
       * The path to a python file (such as `prescient/simulator/prescient.py`)
     - In addition to the two string options available to the command-line, 
       code may also use a python module object. For example::
         
         import my_custom_data_provider
         config.data_provider = my_custom_data_provider

List of Configuration Options
-----------------------------

The table below describes all available configuration options.

.. list-table:: Configuration Options
   :header-rows: 1
   :widths: 20 20 25 35
   :class: table-top-align-cells table-wrap-headers

   * - Command-line Option
     - In-Code Configuration Property
     - Argument
     - Description
   * - \-\-config-file
     - config_file
     - Path. Default=None.
     - Path to a file holding configuration options. Can be absolute or
       relative. Cannot be set in code directly on a configuration object, but
       can be passed to a configuration object's `parse_args()` function:

       .. code-block:: python

          p = Prescient()
          p.config.parse_args(["--config-file", "my-config.txt"])

       See :ref:`launch-with-runner-py` for a description of
       configuration file syntax.

   * - **General Options**
     -
     -
     -
   * - \-\-start-date
     - start_date
     - Date. Default=2020-01-01.
     - The start date for the simulation.
   * - \-\-num-days
     - num_days
     - Integer. Default=7
     - The number of days to simulate.
   * - **Data Options**
     -
     -
     -
   * - \-\-data-path 
        
       or 
       
       \-\-data-directory
     - data_path
     - Path. Default=input_data.
     - Path to a file or folder where input data is located. Whether it 
       should be a file or a folder depends on the input format. See :doc:`input`.
   * - \-\-input-format
     - input_format
     - String. Default=rts_gmlc.
     - The format of the input data. Valid values are *dat* and *rts_gmlc*.
       Ignored when using a custom data provider. See :doc:`input`.
   * - \-\-data-provider
     - data_provider
     - Module. Default=No custom data provider.
     - A python module with a custom data provider that will supply
       data to Prescient during the simulation. Don't specify this option
       unless you are using a custom data provider; use data_path and 
       input_format instead.
       See :ref:`custom-data-providers`.
   * - \-\-output-directory
     - output_directory
     - Path. Default=outdir.
     - The path to the root directory to which all generated simulation 
       output files and associated data are written.
   * - **RUC Options**
     -
     -
     -
   * - \-\-ruc_every-hours
     - ruc_every_hours
     - Integer. Default=24
     - How often a RUC is executed, in hours. Default is 24.
       Must be a divisor of 24.
   * - \-\-ruc-execution-hour
     - ruc_execution_hour
     - Integer. Default=16
     - Specifies an hour of the day the RUC process is executed.
       If multiple RUCs are executed each day (because `ruc_every_hours` 
       is less than 24), any of the execution times may be specified.
       Negative values indicate hours before midnight, positive after.
   * - .. _config_ruc-horizon:
   
       \-\-ruc-horizon
     - ruc_horizon
     - Integer. Default=48
     - The number of hours to include in each RUC.
       Must be >= `ruc_every_hours` and <= 48.
   * - .. _config_ruc-prescience-hour:
   
       \-\-ruc-prescience-hour
     - ruc_prescience_hour
     - Integer. Default=0.
     - The number of initial hours of each RUC in which linear blending of 
       forecasts and actual values is done, making some near-term
       forecasts more accurate.
   * - \-\-run-ruc-with-next-day-data
     - run_ruc_with_next_day_data
     - Flag. Default=false.
     - If false (the default), never use more than 24 hours of
       forecast data even if the RUC horizon is longer than 24 
       hours. Instead, infer values beyond 24 hours.
       
       If true, use forecast data for the full RUC horizon.
   * - \-\-simulate-out-of-sample
     - simulate_out_of_sample
     - Flag. Default=false.
     - If false, use forecast input data as both forecasts and actual
       values; the actual value input data is ignored. 
       
       If true, values for the current simulation time are taken from
       the actual value input, and actual values are used to blend 
       near-term values if `ruc_prescience_hour` is non-zero.
   * - \-\-ruc-network-type
     - ruc_network_type
     - String. Default=ptdf.
     - Specifies how the network is represented in RUC models. Choices are:
       * ptdf   -- power transfer distribution factor representation
       * btheta -- b-theta representation
   * - \-\-ruc-slack-type
     - ruc_slack_type
     - String. Default=every-bus.
     - Specifies the type of slack variables to use in the RUC model formulation.
       Choices are:
       * every-bus            -- slack variables at every system bus
       * ref-bus-and-branches -- slack variables at only reference bus and each system branch
   * - \-\-deterministic-ruc-solver
     - deterministic_ruc_solver
     - String. Default=cbc.
     - The name of the solver to use for RUCs.
   * - \-\-deterministic-ruc-solver-options
     - deterministic_ruc_solver_options
     - String. Default=None.
     - Solver options applied to all RUC solves.
   * - \-\-ruc-mipgap
     - ruc_mipgap
     - Float. Default=0.01.
     - The mipgap for all deterministic RUC solves.
   * - \-\-output-ruc-initial-conditions
     - output_ruc_initial_conditions
     - Flag. Default=false.
     - Print initial conditions to stdout prior to each RUC solve.
   * - \-\-output-ruc-solutions
     - output_ruc_solutions
     - Flag. Default=false.
     - Print RUC solution to stdout after each RUC solve.
   * - \-\-write-deterministic-ruc-instances
     - write_deterministic_ruc_instances
     - Flag. Default=false.
     - Save each individual RUC model to a file. The date and
       time the RUC was executed is indicated in the file name.
   * - \-\-deterministic-ruc-solver-plugin
     - deterministic_ruc_solver_plugin
     - Module. Default=None.
     - If the user has an alternative method to solve 
       RUCs, it should be specified here, e.g.,
       my_special_plugin.py.

       .. note::
          This option is  ignored if `\-\-simulator-plugin` is used.
   * - **SCED Options**
     -
     -
     -
   * - .. _config_sced-frequency-minutes:

       \-\-sced-frequency-minutes
     - sced_frequency_minutes
     - Integer. Default=60.
     - How often a SCED will be run, in minutes.
       Must divide evenly into 60, or be a multiple of 60.
   * - \-\-sced-horizon
     - sced_horizon
     - Integer. Default=1
     - The number of time periods to include in each SCED. 
       Must be at least 1.
   * - .. _config_run-sced-with-persistent-forecast-errors:
   
       \-\-run-sced-with-persistent-forecast-errors
     - run_sced_with_persistent_forecast_errors
     - Flag. Default=false.
     - If true, then values in SCEDs use persistent forecast errors.
       If false, all values in SCEDs use actual values for all time 
       periods, including future time periods. 
       See :ref:`forecast_smoothing`.
   * - \-\-enforce-sced-shutdown-ramprate
     - enforce_sced_shutdown_ramprate
     - Flag. Default=false.
     - Enforces shutdown ramp-rate constraints in the SCED.
       Enabling this option requires a long SCED look-ahead 
       (at least an hour) to ensure the shutdown ramp-rate
       constraints can be statisfied.
   * - \-\-sced-network-type
     - sced_network_type
     - String. Default=ptdf.
     - Specifies how the network is represented in SCED models. Choices are:
       * ptdf   -- power transfer distribution factor representation
       * btheta -- b-theta representation
   * - \-\-sced-slack-type
     - sced_slack_type
     - String. Default=every-bus.
     - Specifies the type of slack variables to use in SCED models. Choices are:
       * every-bus            -- slack variables at every system bus
       * ref-bus-and-branches -- slack variables at only reference bus and each system branch
   * - \-\-sced-solver
     - sced_solver
     - String. Default=cbc.
     - The name of the solver to use for SCEDs.
   * - \-\-sced-solver-options
     - sced_solver_options
     - String. Default=None.
     - Solver options applied to all SCED solves.
   * - \-\-print-sced
     - print_sced
     - Flag. Default=false.
     - Print results from SCED solves to stdout.
   * - \-\-output-sced-initial-conditions
     - output_sced_initial_conditions
     - Flag. Default=false.
     - Print SCED initial conditions to stdout prior to each solve.
   * - \-\-output-sced-loads
     - output_sced_loads
     - Flag. Default=false.
     - Print SCED loads to stdout prior to each solve.
   * - \-\-write-sced-instances
     - write_sced_instances
     - Flag. Default=false.
     - Save each individual SCED model to a file. The date and
       time the SCED was executed is indicated in the file name.
   * - **Output Options**
     -
     -
     -
   * - \-\-disable-stackgraphs
     - disable_stackgraphs
     - Flag. Default=false.
     - Disable stackgraph generation.
   * - \-\-output-max-decimal-places
     - output_max_decimal_places
     - Integer. Default=6.
     - The number of decimal places to output to summary files.
       Output is rounded to the specified accuracy.
   * - \-\-output-solver-logs
     - output_solver_logs
     - Flag. Default=false.
     - Whether to print solver logs to stdout during execution.
   * - **Miscellaneous Options**
     -
     -
     -
   * - \-\-reserve-factor
     - reserve_factor
     - Float. Default=0.0.
     - The reserve factor, expressed as a constant fraction of demand, for
       spinning reserves at each time period of the simulation. Applies to
       both RUC and SCED models.
   * - \-\-no-startup-shutdown-curves
     - no_startup_shutdown_curves
     - Flag. Default=False.
     - If true, then do not infer startup/shutdown ramping curves when starting-up
       and shutting-down thermal generators.
   * - \-\-symbolic-solver-labels
     - symbolic_solver_labels
     - Flag. Default=False.
     - Whether to use symbol names derived from the model when interfacing with 
       the solver.
   * - \-\-enable-quick-start-generator-commitment
     - enable_quick_start_generator_commitment
     - Flag. Default=False.
     - Whether to allow quick start generators to be committed if load shedding
       would otherwise occur.
   * - **Market and Pricing Options**
     -
     -
     -
   * - .. _config_compute-market-settlements:

       \-\-compute-market-settlements
     - compute_market_settlements
     - Flag. Default=False.
     - Whether to solve a day-ahead market as well as real-time market and
       report the daily profit for each generator based on the computed prices.
   * - \-\-day-ahead-pricing
     - day_ahead_pricing
     - String. Default=aCHP.
     - The pricing mechanism to use for the day-ahead market. Choices are:
       * LMP -- locational marginal price
       * ELMP -- enhanced locational marginal price
       * aCHP -- approximated convex hull price.
   * - \-\-price-threshold
     - price_threshold
     - Float. Default=10000.0.
     - Maximum possible value the price can take. If the price exceeds this 
       value due to Load Mismatch, then it is set to this value.
   * - \-\-reserve-price-threshold
     - reserve_price_threshold
     - Float. Default=10000.0.
     - Maximum possible value the reserve price can take. If the reserve price
       exceeds this value, then it is set to this value.
   * - **Plugin Options**
     -
     -
     -
   * - \-\-plugin
     - plugin
     - Module. Default=None.
     - Python plugins are analyst-provided code that Prescient calls at
       various points in the simulation process. See :doc:`plugins` for
       details.

       After Prescient has been initialized, the configuration object's
       `plugin` property holds plugin-specific setting values.
   * - \-\-simulator-plugin
     - simulator_plugin
     - Module. Default=None.
     - A module that implements the engine interface. Use this option
       to replace methods that setup and solve RUC and SCED models with
       custom implementations.
