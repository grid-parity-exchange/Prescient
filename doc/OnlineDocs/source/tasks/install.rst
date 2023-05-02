Installation
============

The Prescient python package can be installed using pip, or it can be installed
from source. Python and a MILP solver are prerequisites for either installation
method.

To install Prescient, follow these steps:

.. contents::
   :local:


Install python
~~~~~~~~~~~~~~
Prescient requires python 3.7 or later. We recommend installing `Anaconda <https://www.anaconda.com>`_
to manage python and other dependencies.


Install a MILP solver
~~~~~~~~~~~~~~~~~~~~~~~
Prescient requires a mixed-integer linear programming (MILP) solver that is compatible with
`Pyomo <https://pyomo.readthedocs.io>`_. Options include open source solvers such as CBC or GLPK,
and commercial solvers such as CPLEX, Gurobi, or Xpress.

The specific mechanics of installing a solver is specific to the solver and/or the platform. An easy way to
install an open source solver on Linux and Mac is to install the CBC Anaconda package into the 
current conda environment::

	conda install -c conda-forge coincbc

.. tip::
   Be sure to activate the correct python environment before running the command above.

CBC binaries for Windows and other platforms may be available from https://github.com/coin-or/Cbc/releases.

Note that the CBC solver is used in most Prescient tests, so you may want to install it even if
you intend to use another solver in your own runs.

Instructions to install other solvers can be found :doc:`here <install_solvers>`.

.. toctree::
   :hidden:
   :maxdepth: 0

   install_solvers

Install Prescient Using Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prescient is available as a python package that can be installed using pip. To install the latest 
release of Prescient use the following command:

    pip install gridx-prescient

Be sure the intended python environment is active before issuing the command above.

Install Prescient From Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to install from source if you want to use the latest pre-release 
version of the code, or if you want to modify/contribute to the code yourself.
The steps required to install Prescient from source are described below:

Get Prescient source code
-------------------------
The latest version of Prescient can be acquired as source from
`the Prescient github project <https://github.com/grid-parity-exchange/Prescient>`_,
either by downloading a zip file of the source code or by cloning the `main` branch of
the github repository.

Install Python Dependencies
---------------------------
The python environment where you run Prescient must include a number of prerequisites.
You may want to create a python environment specifically for Prescient. To create a new
Anaconda environment and install Prescient's prerequisites into the new environment, issue
the following command from the root folder of the Prescient source code::

	conda env create -f environment.yml

The command above will create an environment named `prescient`. To use a different name for the
environment, add the `-n` option to the command above::

	conda env create -n nameOfYourChoice -f environment.yml

Once you have create the new environment, make it the active environment::

    conda activate prescient

If you are using something other than Anaconda to manage your python environment, use the 
information in `environment.yml` to identify which packages to install. 


.. _install-prescient-package:

Install Egret
-------------
When installing Prescient from the latest version of the source code, Egret may need 
to be installed manually because pre-release versions of Prescient sometimes depend 
on pre-release versions of EGRET. Install EGRET from source according to the instructions 
`here <https://github.com/grid-parity-exchange/Egret/blob/main/README.md>`.

Install the Prescient python package
------------------------------------
The steps above configure a python environment with Prescient's prerequisites. Now we must 
install Prescient itself. From the prescient python environment, issue the following command::

	pip install -e .

This will update the active python environment to include Prescient's source code. Any changes to Prescient 
source code will take affect each time Prescient is run.

This command will also install a few utilities that Prescient users may find useful, 
including `runner.py` (see :doc:`run`).

Verify your installation
------------------------
Prescient is packaged with tests to verify it has been set up correctly. To execute the tests, issue the following command::

	pytest -v prescient/simulator/tests/test_simulator.py

This command runs the tests using the CBC solver and will fail if you haven't installed CBC. The tests can take
as long as 30 minutes to run, depending on your machine. If Prescient was installed correctly then all tests should pass.

