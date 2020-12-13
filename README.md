![GitHub CI](https://github.com/grid-parity-exchange/Prescient/workflows/GitHub%20CI/badge.svg?event=schedule)

# Prescient
Code base for Prescient production cost model / scenario generation / prediction interval capabilities.

## Getting started

### Requirements
* Python 3.7 or later
* Pyomo 5.7.1 or later
* EGRET
* A mixed-integer linear programming (MILP) solver
  * Open source: CBC, GLPK, SCIP, ...
  * Commercial: CPLEX, Gurobi, Xpress, ...

### Installation
1. Install EGRET according to the instructions [here](https://github.com/grid-parity-exchange/Egret/blob/master/README.md).
2. Clone or download/extract the Prescient repository.
3. The root of the repository will contain a `setup.py` file. In the command prompt or terminal, change your working directory to the root of the repository. Execute the following command:
```
python setup.py develop
```
This will install Prescient as a Python module while obtaining necessary dependencies. It will also define some new commands in your operating system shell. After installation, your shell should recognize the following command:

```
D:\workspace\prescient>runner.py
You must list the file with the program configurations
after the program name
Usage: runner.py config_file
```

The installation of Prescient has added the `runner.py` command which is used to execute Prescient programs from the command line.

### Solvers
If not using one of the commerical solvers describe below, we recommend users install the open source CBC MILP solver. The specific mechanics of installing CBC are platform-specific. When using Anaconda on Linux and Mac platforms, this can be accomplished simply by:

```
conda install -c conda-forge coincbc
```

The COIN-OR organization - who develops CBC - also provides pre-built binaries for a full range of platforms (including Windows) on https://bintray.com/coin-or/download.


#### Using a commercial solver
If you have a license for a commericial MILP solver (CPLEX, Gurobi, or Xpress), it is recommended over CBC. Further, it is best to have the Python bindings installed for said solver. 

##### CPLEX
Instructions for installing Python bindings for CPLEX can be found [here](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).


##### Gurobi
Python bindings for Gurobi can be installed via conda:
```
conda install -c gurobi gurobi
```
Depending on your license, you may not be able to use the latest version of the solver.
A specific release can be installed as follows:
```
conda install -c gurobi gurobi=8
```

##### Xpress
Xpress Python bindings are available through PyPI, e.g.:
```
pip install xpress
```
Depending on your license, you may need to install a specific version of the solver, e.g.,
```
pip install xpress==8.8.6
```
Note for Xpress users: using the Python bindings for Xpress requires at least Pyomo 5.7.1.


### Testing Prescient
Prescient is packaged with some utility to test and demonstrate its functionality. Here we will walk through the simulation of an [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) case.

#### Additional requirements
* Git

From the command line, navigate to the following directory (relative to the repository root):

```
prescient/downloaders/
```

In that directory, there is a file named `rts_gmlc.py`. Run it:

```
python rts_gmlc.py
```

This script clones the repository of the [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) dataset and formats it for Prescient input. Once the process is complete, the location of the downloaded and processed files will be printed to the command prompt. By default, they will be in:

```
downloads/rts_gmlc/
```

Navigate to this directory. There will be a number of text files in there. These text files contain invocations and options for executing Prescient programs. For example, the `populate_with_network_deterministic.txt` file is used to run the Prescient populator for generating scenarios for one week of data using the RTS-GMLC data. These text files are provided to the `runner.py` command. Start by running the populator for one week of data.

```
runner.py populate_with_network_deterministic.txt
```

You can follow its progress through console output. Once complete, a new folder in your working directory `deterministic_with_network_scenarios` will appear. The folders within contain the scenarios generated. Now that you have scenarios, you can run the simulator for them. The instructions for running the simulator are in the `simulate_with_network_deterministic.txt` and you will once again provide them to the `runner.py` command:

```
runner.py simulate_with_network_deterministic.txt
```

Note: This is assuming that you have installed the solver CBC as recommended. If you are using another solver, you will need to edit the followiong options in the text file to specify those solvers:

```
--deterministic-ruc-solver=cbc
--sced-solver=cbc
```

For example, if you were using Xpress, these lines would be changed to:
```
--deterministic-ruc-solver=xpress
--sced-solver=xpress
```
and similarily for Gurobi (`gurobi`) and CPLEX (`cplex`).

You can watch the progress of the simulator as it is printed to the console. After a period of time, the simulation will be complete. The results will be found a in a new folder `deterministic_with_network_simulation_output`. In that directory, there will be a number of .csv files containing simulation results. An additional folder `plots` contains stack graphs for each day in the simulation, summarizing generation and demand at each time step.

### (For developers) Regression tests
A unit test for testing a simulator run on the based on a single-zone of the RTS-GMLC data is located [here](https://github.com/grid-parity-exchange/prescient/blob/master/tests/simulator_tests/test_sim_rts_mod.py):

```
tests/simulator_tests/test_sim_rts_mod.py
```
It runs two one-week simulations, and compares simulation output to static results (saved in the repo) and uses a diff check with tolerances to evaluate the test. It’s designed to be ran in a fresh instance via GitHub Actions so you need not provide any data or parameters.

The definitions of numeric fields are in numeric_fields.json; these define the fields in each output csv file with which to do results comparisons. The difference tolerances of those fields are similarly defined in tolerances.json but have default values if not explicitly defined.

You can run the script directly:

```
python test_sim_rts_mod.py
```

or use the Python unit test interface:

```
python -m unittest test_sim_rts_mod.py
```

### Developers

By contributing to this software project, you are agreeing to the following terms and conditions for your contributions:

You agree your contributions are submitted under the BSD license.
You represent you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.
