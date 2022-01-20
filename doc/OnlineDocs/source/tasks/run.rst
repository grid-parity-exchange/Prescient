.. highlight:: none

Running Prescient
=================

There are three ways to launch and run Prescient:

* With a configuration file, :ref:`using runner.py<launch-with-runner-py>`
* With command line options, :ref:`using the prescient.simulator module<launch-with-prescient-module>`
* From python code, :ref:`using in-code configuration<launch-with-code>`

In all three cases, the analyst supplies configuration values that
identify input data and dictate which options to use during
the Prescient simulation. Configuration options can be specified in
a configuration file, on the command line, in-code, or a combination
of these methods, depending on how Prescient is launched.

To see what configuration options are available, see :doc:`./configure`.

.. _launch-with-runner-py:

Launch with runner.py
---------------------

Prescient can be run using `runner.py`, a utility which is installed
along with Prescient (see :ref:`install-prescient-package`).
Before executing `runner.py`, you must create a configuration file 
indicating how Prescient should be run. Here is an example of a configuration
file that can be used with `runner.py`::

   command/exec simulator.py
   --data-directory=example_scenario_input
   --output-directory=example_scenario_output
   --input-format=rts-gmlc
   --run-sced-with-persistent-forecast-errors 
   --start-date=07-11-2024
   --num-days=7
   --sced-horizon=1
   --sced-frequency-minutes=10
   --ruc-horizon=36

Because runner.py can potentially be used for more than launching 
Prescient, the first line of the configuration file must match the line
shown in the example above. Otherwise runner.py won't know that you
intend to run Prescient.

All subsequent lines set the value of a configuration option. Configuration 
options are described in :doc:`configure`.

Once you have the configuration file prepared, you can launch Prescient
using the following command::

   runner.py config.txt

where `config.txt` should be replaced with the name of your configuration file.

.. _launch-with-prescient-module:

Launch with the `prescient.simulator` module
--------------------------------------------

Another way to run Prescient is to execute the `prescient.simulator` module::

	python -m prescient.simulator <options>

where `options` specifies the configuration options for the run. An example might 
be something like this::

	python -m prescient.simulator --data-directory=example_scenario_input --output-directory=example_scenario_output --input-format=rts-gmlc --run-sced-with-persistent-forecast-errors --start-date=07-11-2024 --num-days=7 --sced-horizon=1 --sced-frequency-minutes=10 --ruc-horizon=36

Configuration options can also be specified in a configuration file::

	python -m prescient.simulator --config-file=config.txt

You can combine the `--config-file` option with other command line options. The contents of the configuration
file are effectively inserted into the command line at the location of the `--config-file` option. You can 
override values in a configuration file by repeating the option at some point after the `--config-file` option.

Running the `prescient.simulator` module allows you to run Prescient without explicitly installing it, as long 
as Prescient is found in the python module search path.

.. _launch-with-code:

Running Prescient from python code
----------------------------------

Prescient can be configured and launched from python code:

.. code-block:: python

    from prescient.simulator import Prescient

    Prescient().simulate(
            data_path='deterministic_scenarios',
            simulate_out_of_sample=True,
            run_sced_with_persistent_forecast_errors=True,
            output_directory='deterministic_simulation_output',
            start_date='07-10-2020',
            num_days=7,
            sced_horizon=4,
            reserve_factor=0.0,
            deterministic_ruc_solver='cbc',
            sced_solver='cbc',
            sced_frequency_minutes=60,
            ruc_horizon=36,
            enforce_sced_shutdown_ramprate=True,
            no_startup_shutdown_curves=True)

The code example above creates an instance of the Prescient class and passes 
configuration options to its `simulate()` method. An alternative is to set 
values on a configuration object, and then run the simulation after configuration
is done:

.. code-block:: python

    from prescient.simulator import Prescient

    p = Prescient()

    config = p.config
    config.data_path='deterministic_scenarios'
    config.simulate_out_of_sample=True
    config.run_sced_with_persistent_forecast_errors=True
    config.output_directory='deterministic_simulation_output'
    config.start_date='07-10-2020'
    config.num_days=7
    config.sced_horizon=4
    config.reserve_factor=0.0
    config.deterministic_ruc_solver='cbc'
    config.sced_solver='cbc'
    config.sced_frequency_minutes=60
    config.ruc_horizon=36
    config.enforce_sced_shutdown_ramprate=True
    config.no_startup_shutdown_curves=True

    p.simulate()

A third option is to store configuration values in a `dict`, which can potentially
be shared among multiple runs:

.. code-block:: python

    from prescient.simulator import Prescient

    options = {
        'data_path':'deterministic_scenarios',
        'simulate_out_of_sample':True,
        'run_sced_with_persistent_forecast_errors':True,
        'output_directory':'deterministic_simulation_output'
    }

    Prescient().simulate(**options)

These three methods can be used together quite flexibly. The example below 
demonstrates a combination of approaches to configuring a prescient run:

.. code-block:: python

    from prescient.simulator import Prescient

    simulator = Prescient()

    # Set some configuration options using the simulator's config object
    config = simulator.config
    config.data_path='deterministic_scenarios'
    config.simulate_out_of_sample=True
    config.run_sced_with_persistent_forecast_errors=True
    config.output_directory='deterministic_simulation_output'

    # Others will be stored in a dictionary that can 
    # potentially be shared among multiple prescient runs
    options = {
        'start_date':'07-10-2020',
        'sced_horizon':4,
        'reserve_factor':0.0,
        'deterministic_ruc_solver':'cbc',
        'sced_solver':'cbc',
        'sced_frequency_minutes':60,
        'ruc_horizon':36,
        'enforce_sced_shutdown_ramprate':True,
        'no_startup_shutdown_curves':True,
    }
    
    # And finally, pass the dictionary to the simulate() method, 
    # along with an additional function argument.
    simulator.simulate(**options, num_days=7)
