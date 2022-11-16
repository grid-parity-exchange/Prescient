[![GitHub CI](https://github.com/grid-parity-exchange/Prescient/workflows/GitHub%20CI/badge.svg?event=schedule)](https://github.com/grid-parity-exchange/Prescient/actions/workflows/master_tests.yml)

# Prescient

Prescient is a python library that provides production cost modeling capabilities
for power generation and distribution networks.

## Documentation

Documentation is available at https://prescient.readthedocs.io/.

## Getting started

### Requirements

The following must be installed before using Prescient:

* Python 3.7 or later
* A mixed-integer linear programming (MILP) solver
  * Open source: CBC, GLPK, SCIP, ...
  * Commercial: CPLEX, Gurobi, Xpress, ...

### Installation

#### Installing via pip

Prescient is available as a python package that can be installed using pip. To
install the latest release of Prescient use the following command:

```
pip install gridx-prescient
```

#### Installing from source

You may want to install from source if you want to use the latest pre-release
version of the code, or if you want to modify/contribute to the code yourself.
Install from source by following these steps:

1. Install EGRET from source according to the instructions [here](https://github.com/grid-parity-exchange/Egret/blob/main/README.md).
   This is necessary because pre-release versions of Prescient sometimes depend 
   on pre-release versions of EGRET.
2. Clone or download/extract the Prescient repository.
3. Install Prescient into the environment as an editable package by issuing the 
   following command from the root of the Prescient repository:
   ```
   pip install -e .
   ```
   This will install Prescient as a Python module while obtaining necessary dependencies. 

### Solver Installation

Prescient requires a pyomo-compatible MILP solver. A commercial solver (CPLEX, 
Gurobi, or Xpress) is recommended over open source solvers. If a commercial 
solver is not available we recommend installing the open source CBC MILP solver. 

#### CBC

On Linux and Mac platforms, CBC can be installed using Anaconda:

```
conda install -c conda-forge coincbc
```

Binaries for additional platforms, including Windows, may be available from https://github.com/coin-or/Cbc/releases.

#### CPLEX
Instructions for installing Python bindings for CPLEX can be found [here](https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).


#### Gurobi
Python bindings for Gurobi can be installed via conda:
```
conda install -c gurobi gurobi
```
Depending on your license, you may not be able to use the latest version of the solver.
A specific release can be installed as follows:
```
conda install -c gurobi gurobi=8
```

#### Xpress
Xpress Python bindings are available through PyPI, e.g.:
```
pip install xpress
```
Depending on your license, you may need to install a specific version of the solver, e.g.,
```
pip install xpress==8.8.6
```

### RTS-GMLC Example Model
Prescient is packaged with a utility to test and demonstrate its functionality. Here we
will walk through the simulation of an [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) case.

From the command line, navigate to the following directory (relative to the repository root):

```
cd prescient/downloaders/
```

In that directory, there is a file named `rts_gmlc.py`. Run it:

```
python rts_gmlc.py
```

This script clones the repository of the [RTS-GMLC](https://github.com/GridMod/RTS-GMLC)
dataset and formats it for Prescient input. Once the process is complete, the location
of the downloaded and processed files will be printed to the command prompt. Navigate to
that directory:

```
cd ../..
cd downloads/rts_gmlc/
```

There will be a number of text files in this directory. 
These text files contain invocations and options for executing Prescient 
programs. For example, the `populate_with_network_deterministic.txt` file is 
used to run the Prescient populator for generating scenarios for one week of 
data using the RTS-GMLC data. These text files are provided to the `runner.py` 
command (a utility installed with Prescient). Start by running the populator
for one week of data.

```
runner.py populate_with_network_deterministic.txt
```

You can follow its progress through console output. Once complete, a new folder 
in your working directory `deterministic_with_network_scenarios` will appear. The 
folders within contain the generated scenario. 

The configuration that will be used to run the simulator is in 
`simulate_with_network_deterministic.txt`. The configuration assumes you have
installed the solver CBC. If you are using another solver you will need to 
edit the following options in the configuration file to specify the appropriate
solver:

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

You can now run the simulator. Once again you will use the `runner.py` command:

```
runner.py simulate_with_network_deterministic.txt
```

You can watch the progress of the simulator as it is printed to the console. After
a period of time, the simulation will be complete. The results will be found a in
a new folder `deterministic_with_network_simulation_output`. In that directory,
there will be a number of .csv files containing simulation results. An additional
folder `plots` contains stack graphs for each day in the simulation, summarizing
generation and demand at each time step.

## Contributing to Prescient

### Contributor Agreement

By contributing to this software project, you are agreeing to the following terms and
conditions for your contributions:

* You agree your contributions are submitted under the BSD license.
* You represent you are authorized to make the contributions and grant the license. If
  your employer has rights to intellectual property that includes your contributions,
  you represent that you have received permission to make contributions and grant the
  required license on behalf of that employer.

### Developer Regression Tests

Regression tests are run automatically with each pull request. You can also run
regression tests on your local machine using the following command from the 
repository root:

```
pytest -v prescient/simulator/tests/test_simulator.py
```
