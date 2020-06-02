#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os
import string

from six import itervalues, iteritems

import pyomo.util.plugin

from pyomo.pysp import phextension

from pyomo.pysp.phsolverserverutils import transmit_external_function_invocation

import pyomo.solvers.plugins.smanager.phpyro

# core function for setting initial conditions is split out, to allow for non-PH plugin execution.
def load_initial_conditions(ph_base, scenario_tree, scenario, filename="ic.txt"):

    scenario_instance = ph_base._instances[scenario._name]

    ic_file = open(filename)

    for line in ic_file.readlines():
        pieces = line.strip().split()
        if len(pieces) > 1:
            generator = pieces[0]
            t0_state = int(pieces[1])
            t0_power = float(pieces[2])
            scenario_instance.UnitOnT0State[generator] = t0_state
            if t0_state > 0:
                scenario_instance.UnitOnT0[generator] = 1
            else:
                scenario_instance.UnitOnT0[generator] = 0
            scenario_instance.PowerGeneratedT0[generator] = t0_power
        else:
            reserve_factor = float(pieces[0])
            scenario_instance.ReserveFactor = reserve_factor

    reference_model_module=ph_base._scenario_instance_factory._model_module
    reference_model_module.reconstruct_instance_for_t0_changes(scenario_instance)

    scenario_instance.preprocess()

    ic_file.close()

# this is a PH extension to set the initial conditions for all scenarios to a common
# set of values, obtained from a well-known input file.

class InitialStateSetter(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements (phextension.IPHExtension)
    
    def pre_ph_initialization(self,ph):
        """Called before PH initialization."""
        pass

    def post_instance_creation(self, ph):
        """Called after PH initialization has created the scenario instances, but before any PH-related weights/variables/parameters/etc are defined!"""

        ic_filename = "ic.txt"

        print("Reading initial conditions from input file="+ic_filename)

        if not os.path.exists(ic_filename):
            raise RuntimeError("The initial conditions file="+ic_filename+" either does not exist or cannot be read")

        # if running in distributed mode, send a request to tell each PH solver server to
        # load initial conditions from ic.txt. otherwise, do the iteration over the local
        # PH instances explicitly.

        if isinstance(ph._solver_manager, pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
            transmit_external_function_invocation(ph, "initialstatesetter.py", "load_initial_conditions")

        # independent of serial or parallel mode, we need to modify the local instances.
        # TBD - unless, that is, we're culling constraints and allowing PH to run to convergence.

        # slightly wasteful, in the sense that the ic filename will be loaded
        # per instance. current code is structured to allow for scenario-at-a-time
        # invocation of the function, from ph solver servers. if this becomes a
        # performance issue (and it won't), then we could revisit.
        for scenario_name, scenario_instance in iteritems(ph._instances):            
            load_initial_conditions(ph, ph._scenario_tree, ph._scenario_tree.get_scenario(scenario_name), filename=ic_filename)

    def post_ph_initialization(self, ph):
        """Called after PH initialization!"""
        pass

    def post_iteration_0_solves(self, ph):
        """Called after the iteration 0 solves!"""
        pass

    def post_iteration_0(self, ph):
        """Called after the iteration 0 solves, averages computation, and weight computation"""
        pass

    def pre_iteration_k_solves(self, ph):
        """Called immediately before the iteration k solves!"""
        pass

    def post_iteration_k_solves(self, ph):
        """Called after the iteration k solves!"""
        pass

    def post_iteration_k(self, ph):
        """Called after the iteration k is finished, after weights have been updated!"""
        pass

    def post_ph_execution(self, ph):
        """Called after PH has terminated!"""
        pass
