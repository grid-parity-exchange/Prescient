#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
pyspgen.py

This file will contain all the functions and classes necessary for generating
the pysp files for simulation.
"""

import sys
import os
import importlib
from copy import deepcopy
from collections import OrderedDict, namedtuple

import prescient.gosm.basicclasses as basicclasses
import egret.parsers.prescient_dat_parser as pdp


class CommentedRawNodeData:
    """
    A scenario node which has comments which will be pasted in the
    individual scenario files constructed from the RawNode.

    This will be different in that instead of a dictionary being passed
    in, it will pass in a function which will return the dictionary of
    data specifying the scenario configurations. We do this in order to save
    space and only construct the data dictionary when needed.
    """
    def __init__(self, funcin, name, parentname, prob, comments=''):

        self.data_func = funcin
        self.name = name
        self.parent_name = parentname
        self.prob = prob
        self.comments = comments

    @property
    def valdict(self):
        valdict = OrderedDict()
        data_dict = self.data_func()
        for pname in data_dict:
            if isinstance(data_dict[pname], dict):
                if pname not in valdict:
                    valdict[pname] = OrderedDict()
                for pindex in data_dict[pname]:
                    valdict[pname][pindex] = data_dict[pname][pindex]
            else:
                valdict[pname] =  data_dict[pname][pindex]
        return valdict

class MyPySPTree(basicclasses.PySP_Tree):
    """
    This class is a modification of the PySP_Tree class in basicclasses.py.
    It is to enable a lowercase root.
    """

    def __init__(self, template, scenarios):
        """
        Args:
            template: a PySP template class object
            scenarios: a list of PySP scenario class objects
        """

        epsilon = 1e-6  # for testing probs
        self.template = template
        self.scenarios = scenarios
        # check scenario names for uniqueness and check total prob
        totalprob = 0.0
        scenarionames = []
        for scen in scenarios:
            sname = scen.name
            if scen in scenarionames:
                raise RuntimeError('Internal: Duplicate Scenario Name=' + sname)
            totalprob += scen.prob
        if totalprob > 1.0 + epsilon:
            raise RuntimeError('Internal: > 1.0, totalprob=' + str(totalprob))

        """
        In this "default" version, we are going to extract the tree information
        from the scenarios. One can envision deriving classes where that
        information is passed in by whatever created the scenarios in the first
        place. For now, keep it simple.
        """
        """
        We need to guard against (or allow) "conditional"
        raw node names, so we
        will form the node names by concatentation, which could get ugly.
        They are the dict indexes, by the way.
        Dec 2016: Let's assume that if there was a ROOT node given (uncommon)
        for any scenario, it was given for all.... (do I use this assumption?)
        """
        self.NodeNames = ['root']
        self.NodeStage = {'root': 0}  # zero based
        self.NodeProb = {'root': 1.0}
        self.NodeKids = {'root': []}
        self.LeafNode = {}  # indexed by scenario name
        self.ScenForLeaf = {}  # index by leaf node name, give scenario name

        rootoffset = 0
        for scen in scenarios:
            nodename = "root"  # *not* raw; PySP name
            for i in range(len(scen.raw.nodelist)):
                nd = scen.raw.nodelist[i]
                if nd.name == 'root':  # assumed uncommon, but allowed
                    rootoffset = 1
                    continue
                pname = nodename  # parent name
                nodename += "_" + nd.name
                if nodename not in self.NodeNames:
                    self.NodeNames.append(nodename)
                    self.NodeStage[nodename] = i + 1 - rootoffset
                    self.NodeProb[nodename] = nd.prob  # conditional
                    self.NodeKids[nodename] = []
                if nodename in self.NodeKids[pname]:
                    raise RuntimeError('Duplicate created node name=' + nodename)
                self.NodeKids[pname].append(nodename)
                self.LeafNode[scen.name] = nodename  # last assignment sticks..

        for scen in scenarios:
            self.ScenForLeaf[self.LeafNode[scen.name]] = scen.name  # invert


class TwoStageScenarioData:
    """
    This is a scenario object which encodes a two stage scenario. It is
    based on Raw_Scenario_Data; however, since there is only one stage, it is
    only necessary to pass a single node.
    """

    def __init__(self, node):
        """
        Args:
            node (CommentedRawNodeData): The node encoding the data for the
                scenario
        """
        self.name = "Scenario_{}".format(node.name)
        self.prob = node.prob
        self.node = node
        self.nodelist = [node]

        if not (0 <= node.prob <= 1):
            raise ValueError("Node probability is not in [0,1]")

    @property
    def valdict(self):
        return self.node.valdict

def import_model_file(model_file):
    """
    This function will import a module by the filepath. It will return the
    module for the reference model.

    This is a recipe taken from the docs for importlib.

    Args:
        model_file (str): The path to the reference model python file
    Returns:
        A module object representing the python file
    """
    # The name used here is a dummy, it is unused later
    spec = importlib.util.spec_from_file_location('ReferenceModel', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def check_scen_template_for_sources(template_file, source_names):
    """
    This function will check the template file has all of the source names
    set in the 'NondispatchableGeneratorsAtBus' variable. Specifically, we must
    have that each source name appears in at least one bus.

    This should have a line of the following sort in the scenario template file

    set NondispatchableGeneratorsAtBus[Bus1] := Wind;

    If any source is not found, then this will raise an error.

    Args:
        template_file (str): The path to the scenario skeleton file
        model_file (str): The path to the Reference Model used for pyomo. This
            must have a model attribute
        source_names (list[str]): A list of the source names to be checked
    """
    try:
        model = pdp.get_uc_model()
        instance = model.create_instance(template_file)
    except Exception as e:
        s = ""
        s += "Error reading scenario data into Egret"
        s += "Check file {}".format(template_file)
        s += "Error reported: {}".format(','.join(map(str, e.args)))
        raise RuntimeError(s)

    for source in source_names:
        # If the source is in at least one bus, we are good
        for bus, sources in instance.NondispatchableGeneratorsAtBus.items():
            if source in sources:
                break
        else:
            # If we get through all the buses and do not find the source.
            raise RuntimeError("Source {} not found in template file {}"
                .format(source, template_file))


def adapt_scen_template_to_sources(open_file, source_names):
    """
    Adapts the scenario template to the current sources. In particular, the set
    'NondispatchableGeneratorsAtBus' is filled with the sources' names.

    Args:
        open_file (file): the open template file
        source_names (list[str]): a list of all source names

    Returns:
        str: a string with the adapted template
    """

    output = ""
    for line in open_file:
        if line.startswith('set NondispatchableGeneratorsAtBus'):
            split_line = line.split(':=')
            adapted_line = split_line[0] + ':= ' + " ".join(source_names) + " ;\n"
            output += adapted_line
        else:
            output += line

    return output


def do_2stage_ampl_dir(dir_name, raw_nodes, scentemp, treetemp):
    """
    This function will take raw nodes, a tree template file, scenario template
    file and a directory and writes scenario files and a scenario structure
    file.

    Args:
        dir_name (str): The name of the directory to store the files
        raw_nodes (List[Raw_Node_Data]): A list of raw nodes representing
            each possible scenario.
        scentemp (PySP_Scenario_Template): The scenario template object
        treetemp (PySP_Tree_Template): The tree template object
    """
    # each node is a scenario.
    pysp_scen_list = []
    for r in raw_nodes:
        rs = TwoStageScenarioData(r)
        scentempcopy = deepcopy(scentemp)
        comments = r.comments.split('\n')
        comments = ['# ' + comment + '\n' for comment in comments]
        scentempcopy.templatedata = comments + scentempcopy.templatedata
        pysp_scen = basicclasses.PySP_Scenario(raw=rs, template=scentempcopy)
        pysp_scen_list.append(pysp_scen)
        fname = os.path.join(dir_name, pysp_scen.name + ".dat")
        pysp_scen.to_file(fname)

    # Note that there was no need to keep the raw scenario for
    # our purposes, but we need the raw nodes along with the PySP scenarios.

    tree = MyPySPTree(treetemp, pysp_scen_list)
    tree.Write_ScenarioStructure_dat_file(os.path.join(dir_name, 'ScenarioStructure.dat'))


def write_actuals_and_expected(scenario_set, write_directory, scentemp):
    """
    Lifted code from daps to properly integrate construction of expected and actuals
    scenarios. This will create PySP_Scenarios from the Actual and Expected
    scenarios contained in scenario_set.

    Args:
        scenario_set: a ScenarioSet object
        write_directory: directory containing json files
        scentemp (PySP_Scenario_Template): The scenario template object
    """

    actual_node, forecast_node = scenario_set.actual_and_expected_node()
    actual_scen_data = TwoStageScenarioData(actual_node)
    forecast_scen_data = TwoStageScenarioData(forecast_node)

    actual_scenario = basicclasses.PySP_Scenario(actual_scen_data, scentemp)
    forecast_scenario = basicclasses.PySP_Scenario(forecast_scen_data, scentemp)

    fname = os.path.join(write_directory, actual_scenario.name + ".dat")
    actual_scenario.to_file(fname)

    fname = os.path.join(write_directory, forecast_scenario.name + ".dat")
    forecast_scenario.to_file(fname)
