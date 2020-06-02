#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# geo_decomp code


import sys
import os
import shutil
import random
import traceback
import csv
import time
import multiprocessing
import pickle
import subprocess
import datetime
import math
from optparse import OptionParser, OptionGroup

try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

try:
    import cProfile as profile
except ImportError:
    import profile

import matplotlib
# the following forces matplotlib to not use any Xwindows backend.
# taken from stackoverflow, of course.
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from pandas import DataFrame

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

try:
    import dateutil
except:
    print("***Failed to import python dateutil module - try: easy_install python-dateutil")
    sys.exit(1)

from six import iterkeys, itervalues, iteritems
from pyomo.core import *
from pyomo.opt import *
from pyomo.pysp.ef import create_ef_instance
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory, ScenarioTree
from pyomo.pysp.phutils import find_active_objective, cull_constraints_from_instance
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
import pyutilib

import prescient.sim.MasterOptions as MasterOptions

# import plotting capabilities
import prescient.sim.graphutils as graphutils
import prescient.sim.storagegraphutils as storagegraphutils

from prescient.scripts.simulator import call_solver



def fix_generation(region_gen_list,model_LP):
    region=region_gen_list
    model_region=model_LP.clone()
    
    # make focus region a binary again
    
    for g in model_region.ThermalGenerators:
        if str(g) in region:
            # indicators for g
            for t,t_prime in model_region.ShutdownHotStartupPairs[g]:
                
                model_region.StartupIndicator[g,t,t_prime].domain=Binary
                
            # other binary variables for g    
            for t in model_region.TimePeriods:
                
                model_region.UnitOn[g,t].domain=Binary
                model_region.UnitStart[g,t].domain=Binary
                model_region.UnitStop[g,t].domain=Binary




        # fix generation and commitment values outside of the focus region
        
        else: 

            for t in model_region.TimePeriods:

                model_region.PowerGeneratedAboveMinimum[g,t].fix(model_LP.PowerGeneratedAboveMinimum[g,t].value)
                model_region.UnitOn[g,t].fix(model_LP.UnitOn[g,t].value)
                
                #print(model_region.PowerGeneratedAboveMinimum[g,t].value==model_LP.PowerGeneratedAboveMinimum[g,t].value)


    return model_region
    
############################################################################################################
def save_commmitments(model,region,UnitOn_Final):
    
    for g in model.ThermalGenerators:
        if g in region:
            for t in model.TimePeriods:
                UnitOn_Final[(str(g),str(t))]=model.UnitOn[g,t].value
    return UnitOn_Final



############################################################################################################
directory_name = os.path.dirname(os.path.realpath(__file__))
geo_file = os.path.abspath(os.path.join(directory_name, 'focus_regions.csv'))
    
def solve_deterministic_ruc(new_deterministic_ruc, solver, options, solve_options):
    #results = solver.solve(new_deterministic_ruc, tee=options.output_solver_logs)

    # clone to create and LP model
    model_LP=new_deterministic_ruc.clone()


    # Create LP
    model_LP.UnitOn.domain=UnitInterval
    #model_LP.UnitOn.bounds=(0,1)
    model_LP.UnitStart.domain=UnitInterval
    #model_LP.UnitStart.bounds=(0,1)
    model_LP.UnitStop.domain=UnitInterval
    #model_LP.UnitStop.bounds=(0,1)
    model_LP.StartupIndicator.domain=UnitInterval
    #model_LP.StartupIndicator.bounds=(0,1)

    ##################################################################################Solve LP

    #solver=SolverFactory('glpk')
    #solver.solve(model_LP, tee=True,keepfiles=True, symbolic_solver_labels=True)


    results = call_solver(solver,
                      model_LP, 
                      tee=options.output_solver_logs,
                      **solve_options[solver])

    if results.solution.status.key != "optimal":

        print("Failed to solve LP deterministic RUC instance - no feasible solution exists!") 
        
        output_filename = "bad_ruc_LP.lp"
        print("Writing failed LP RUC model to file=" + output_filename)
        lp_writer = ProblemWriter_cpxlp()            
        lp_writer(model_LP, output_filename, 
                  lambda x: True, {"symbolic_solver_labels" : True})

        if options.error_if_infeasible:
            raise RuntimeError("Halting due to infeasibility")

    model_LP.solutions.load_from(results)
    ##################################################################################Solve LP


    # Create Dictionaries and Save LP Results

    LP_PowerGeneratedAboveMinimum={}
    LP_UnitOn={}

    for g in model_LP.ThermalGenerators:
        for t in model_LP.TimePeriods:
            LP_PowerGeneratedAboveMinimum[(str(g),str(t))]=model_LP.PowerGeneratedAboveMinimum[g,t].value
            LP_UnitOn[(str(g),str(t))]=model_LP.UnitOn[g,t].value


    # loop over regions to obtain commitments

    UnitOn_Final={}

    df_regions=pd.read_csv(geo_file,index_col=0)
    region_dictionary={}
    for key in df_regions.region.unique():
        region_dictionary[key]=list(df_regions[df_regions['region']==str(key)]['generator'])


    for region in region_dictionary.values():

        model_region=fix_generation(region,model_LP) 

        ################################################################################## Solve Regional commitment

        results = call_solver(solver,
                          model_region, 
                          tee=options.output_solver_logs,
                          **solve_options[solver])

        if results.solution.status.key != "optimal":

            print("Failed to solve region "+ str(key)+ "deterministic RUC instance - no feasible solution exists!") 
            # ASK JP
            output_filename = "region "+ str(key) + "bad_ruc.lp"
            print("Writing failed region "+ str(key)+ "RUC model to file=" + output_filename)
            lp_writer = ProblemWriter_cpxlp()            
            lp_writer(model_region, output_filename, 
                      lambda x: True, {"symbolic_solver_labels" : True})

            if options.error_if_infeasible:
                raise RuntimeError("Halting due to infeasibility")

        model_region.solutions.load_from(results)
        ################################################################################## Solve Regional commitment



        UnitOn_Final=save_commmitments(model_region,region,UnitOn_Final)


    # Fix commitment decsions into the original model

    for g in new_deterministic_ruc.ThermalGenerators:
        for t in new_deterministic_ruc.TimePeriods:
            new_deterministic_ruc.UnitOn[g,t].fix(UnitOn_Final[(str(g),str(t))])

    ################################################################################## Solve for Feasibility


    results = call_solver(solver,
                      new_deterministic_ruc, 
                      tee=options.output_solver_logs,
                      **solve_options[solver])

    if results.solution.status.key != "optimal":

        print("Failed to solve Geo_RUC deterministic RUC instance - no feasible solution exists!") 
        # ASK JP
        output_filename = "bad_GEO_ruc.lp"
        print("Writing failed GEO_RUC model to file=" + output_filename)
        lp_writer = ProblemWriter_cpxlp()            
        lp_writer(new_deterministic_ruc, output_filename, 
                  lambda x: True, {"symbolic_solver_labels" : True})

        if options.error_if_infeasible:
            raise RuntimeError("Halting due to infeasibility")

    new_deterministic_ruc.solutions.load_from(results)
    ################################################################################## Solve for Feasibility

    # Unfix loaded commitment decsions into the original model

    for g in new_deterministic_ruc.ThermalGenerators:
        for t in new_deterministic_ruc.TimePeriods:
            new_deterministic_ruc.UnitOn[g,t].unfix()

    print("Successfully solved deterministic RUC with geographic decomposition")
