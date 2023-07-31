.. title:: Installing Specific MILP Solvers

.. raw:: html

	<h1>Installing Specific MILP Solvers</h1>

Prescient uses a mixed-integer linear programming (MILP) solver to make
dispatch and unit commitment decisions. Before running Prescient, a solver
must be installed and available to `Pyomo <https://pyomo.readthedocs.io>`_.
Installation guidance for a number of common solvers is found below.

.. contents:: Solvers
   :local:

The choice of solver often comes down to cost. CBC is free, but is typically
slower than commercial alternatives. Commercial solvers are faster, but
require a license and may require additional configuration. Some solvers
offer free licenses for academic or research use. Check with the appropriate
vendor for details.

CBC
---
CBC is a free, open-source MILP solver. On Linux and Mac platforms, CBC
can be installed using Anaconda::

	conda install -c conda-forge coincbc

Binaries for additional platforms, including Windows, may be available from
https://github.com/coin-or/Cbc/releases. When installing CBC manually, you
may need to modify your path to ensure the cbc executable is available on
the command line.

CBC is the solver Prescient uses for automated tests on github.

Gurobi
------

Gurobi is a highly regarded commercial solver. Gurobi requires a valid
license to be used by Prescient. See the documentation on the
`Gurobi <https://gurobi.com>`_ website for instructions on how to acquire
and install a license.

Python bindings for Gurobi can be installed via conda::

	conda install -c gurobi gurobi

or via pip::

	pip install gurobi

Depending on your license, you may not be able to use the latest version
of the solver. A specific release can be installed as follows::

	conda install -c gurobi gurobi=8

CPLEX
-----

CPLEX is a high performance solver with both free and paid versions
available. The free version, called the Community Edition, can be
installed using pip::

	pip install cplex

The free CPLEX Community Edition has limits on model size and may not be
sufficient for your models. To use the commercial edition, you must
acquire a CPLEX license and install the CPLEX software suite. After installing
CPLEX, you must then install Python bindings for CPLEX into your python
environment following the instructions found
`here <https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-setting-up-python-api>`_.

Xpress
------

Xpress Python bindings are available through conda::

	conda install -c fico-xpress xpress

or from PyPI using pip::

	pip install xpress

Depending on your license, you may need to install a specific version of the
solver, e.g., ::

	pip install xpress==8.8.6
