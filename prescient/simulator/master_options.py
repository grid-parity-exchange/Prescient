#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# SZ 26. Oct. 2016 : PINT2 uses the globals.py instead, but Prescient still uses this

####################################################
#                 MasterOptions                    #
####################################################
# Populate the MasterOptions, composition of ChainOptions, ForErrorOptions, and PRESCIENT options.
from __future__ import annotations
import sys
import os
from typing import Tuple, Dict

from optparse import OptionParser, OptionGroup
import prescient.plugins

def construct_options_parser() -> Tuple[OptionParser, Dict[str, Dict[str, bool]]]:
    '''
    Make a new parser that can parse standard and custom command line options.

    Custom options are provided by plugin modules. Plugins are specified
    as command-line arguments "--plugin=<module name>", where <module name>
    refers to a python module. The plugin may provide new command line
    options by calling "prescient.plugins.add_custom_commandline_option"
    when loaded.
    '''

    # To support the ability to add new command line options to the line that
    # is currently being parsed, we convert a standard OptionParser into a
    # two-pass parser by replacing the parser's parse_args method with a
    # modified version.  In the modified parse_args, a first pass through the
    # command line finds any --plugin arguments and allows the plugin to 
    # add new options to the parser.  The second pass does a full parse of
    # the command line using the parser's original parse_args method.
    parser, overrides = _construct_inner_options_parser()
    parser._inner_parse = parser.parse_args

    def outer_parse(args=None, values=None):
        if args is None:
            args = sys.argv[1:]

        from prescient.plugins.internal import active_parser
        prescient.plugins.internal.active_parser = parser

        # Manually check each argument against --plugin=<module>,
        # give plugins a chance to install their options.
        stand_alone_opt = '--plugin'
        prefix = "--plugin="
        next_arg_is_module = False
        for arg in args:
            if arg.startswith(prefix):
                module_name = arg[len(prefix):]
                _initialize_plugin(module_name)
            elif arg == stand_alone_opt:
                next_arg_is_module = True
            elif next_arg_is_module:
                module_name = arg
                _initialize_plugin(module_name)
                next_arg_is_module = False

        # Now parse for real, with any new options in place.
        return parser._inner_parse(args, values)

    parser.parse_args = outer_parse
    return parser, overrides

def _construct_inner_options_parser():

    parser = OptionParser()
    guiOverride = {}  # a dictionary of dictionaries to facilitate gui creation
    # guiOverride can also indicate that an option is not intended for user input
    
    directory_options = OptionGroup(parser, "Directory Options")
    structure_options = OptionGroup(parser, "Structure Options")
    solver_options = OptionGroup(parser, "Solver Options")
    populator_options = OptionGroup(parser, "Populator Options")
    MSSimulator_options = OptionGroup(parser, "MMSimulator Options")
    input_simulation_options = OptionGroup(parser, "InputSimulation Options")
    solver_simulation_options = OptionGroup(parser, "SolverSimulation Options")
    output_simulation_options = OptionGroup(parser, "OutputSimulation Options")
    extension_options = OptionGroup(parser, "Extension Options")
    other_simulation_options = OptionGroup(parser, "OtherSimulation Options")

    parser.add_option_group(directory_options)
    parser.add_option_group(structure_options)
    parser.add_option_group(solver_options)
    parser.add_option_group(populator_options)
    parser.add_option_group(MSSimulator_options)
    parser.add_option_group(input_simulation_options)
    parser.add_option_group(solver_simulation_options)
    parser.add_option_group(output_simulation_options)
    parser.add_option_group(extension_options)
    parser.add_option_group(other_simulation_options)


##########################
#   CHAIN ONLY OPTIONS   #
##########################
# # # # # # Added by Paulin to use multi stage Simulator

    MSSimulator_options.add_option('--MSSim-start-date',
                                   help='Specifies the day on which to begin the simulation (default 0).',
                                   action='store',
                                   dest='MSSim_start_date',
                                   type='string',
                                   default=None)
    guiOverride['--MSSim-start-date'] = {}
    guiOverride['--MSSim-start-date']['bpa'] = False

    MSSimulator_options.add_option('--MSSim-num-days',
                                   help='Specifies the number of simulated days to execute.',
                                   action='store',
                                   dest='MSSim_num_days',
                                   type='int',
                                   default=1)
    guiOverride['--MSSim-num-days'] = {}
    guiOverride['--MSSim-num-days']['bpa'] = False
    
    MSSimulator_options.add_option('--MSSim-Ruc-solver-tee',
                                   help='Specifies if the Ruc solver should print its solving steps and '
                                        'results in SCED and RUC problems.',
                                   action='store_true',
                                   dest='MSSim_Ruc_solver_tee',
                                   default=False)
    guiOverride['--MSSim-Ruc-solver-tee'] = {}
    guiOverride['--MSSim-Ruc-solver-tee']['bpa'] = False
    guiOverride["--MSSim-Ruc-solver-tee"]['advanced'] = True
                                 
    MSSimulator_options.add_option('--MSSim-Sced-solver-tee',
                                   help='Specifies if the Sced solver should print its solving steps and '
                                        'results in SCED and RUC problems.',
                                   action='store_true',
                                   dest='MSSim_Sced_solver_tee',
                                   default=False)
    guiOverride['--MSSim-Sced-solver-tee'] = {}
    guiOverride['--MSSim-Sced-solver-tee']['bpa'] = False
    guiOverride["--MSSim-Sced-solver-tee"]['advanced'] = True
                             
    MSSimulator_options.add_option("--simulate-actuals",
                                   help="Boolean which says if we run a simulation over the actuals",
                                   action="store_true",
                                   dest="simulate_actuals",
                                   default=False)
    guiOverride["--simulate-actuals"] = {}
    guiOverride["--simulate-actuals"]['bpa'] = False
                        
    MSSimulator_options.add_option("--multiStageSim-output-directory",
                                   help="path of the directory in which the simulations are saved",
                                   action="store",
                                   dest="multiStageSim_output_directory",
                                   default="")
    guiOverride["--multiStageSim-output-directory"] = {}
    guiOverride["--multiStageSim-output-directory"]['bpa'] = False
    guiOverride["--multiStageSim-output-directory"]['advanced'] = True
                                 
    MSSimulator_options.add_option("--commitment-cost-parameters-filename",
                                   help="File where the commitment and decommitment costs penalties factors "
                                        "are specified. There should be line for CommitmentFactor=.., "
                                        "DeommitmentFactor=.. and a factor for each stage Stage_N=..",
                                   action="store",
                                   dest="commitment_cost_parameters_filename",
                                   default=None)
    guiOverride["--commitment-cost-parameters-filename"] = {}
    guiOverride["--commitment-cost-parameters-filename"]['bpa'] = False
                                                                  
    MSSimulator_options.add_option("--simulate-forecast-updates",
                                   help="Boolean which says if we run a simulation to incorporate forecast updates",
                                   action="store_true",
                                   dest="simulate_forecast_updates",
                                   default=False)
    guiOverride["--simulate-forecast-updates"] = {}
    guiOverride["--simulate-forecast-updates"]['bpa'] = False
                                 
    MSSimulator_options.add_option("--MSSimulator-mipgap",
                                   help="Mipgap used across all the simulations",
                                   action="store",
                                   dest="MSSimulator_mipgap",
                                   type="float",
                                   default=0.1)
    guiOverride["--MSSimulator-mipgap"] = {}
    guiOverride["--MSSimulator-mipgap"]['bpa'] = False
    guiOverride["--MSSimulator-mipgap"]['advanced'] = True
                                 
    MSSimulator_options.add_option("--MSSim-run-ID-deterministic-for-everymodel",       
                                   help="if specified, the deterministic model with ACTUAL DATA will be run for "
                                        "every intraday stages (i.e. >2) for the three horses : det, MS and TS. "
                                        "It's assuming we have perfect updates on stage 2.",
                                   action="store_true",
                                   dest="MSSim_run_ID_deterministic_for_everymodel",
                                   default=False) 
    guiOverride["--MSSim-run-ID-deterministic-for-everymodel"] = {}
    guiOverride["--MSSim-run-ID-deterministic-for-everymodel"]['bpa'] = False
    
    MSSimulator_options.add_option("--write-MScosts-in-this-file",
                                   help="if specified, write the costs in the file instead of outputing them",
                                   action="store",
                                   dest="write_MScosts_in_this_file",
                                   type="string",
                                   default='')
    guiOverride["--write-MScosts-in-this-file"] = {}
    guiOverride["--write-MScosts-in-this-file"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-plotCharts",
                                   help="plot histogram of the generation dispatch for deterministic, "
                                        "MS and TS models.",
                                   action="store_true",
                                   dest="MSSim_plotCharts",
                                   default=False)
    guiOverride["--MSSim-plotCharts"] = {}
    guiOverride["--MSSim-plotCharts"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-plot-CommitmentCharts",
                                   help="plot chart of the commitments in different stages for deterministic, "
                                        "MS and TS models.",
                                   action="store_true",
                                   dest="MSSim_plot_CommitmentCharts",
                                   default=False)
    guiOverride["--MSSim-plot-CommitmentCharts"] = {}
    guiOverride["--MSSim-plot-CommitmentCharts"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-plot-adjusted-scenarios",
                                   help="save a plot of the adjusted scenarios that are computed in the "
                                        "intra day stages",
                                   action="store_true",
                                   dest="MSSim_plot_adjusted_scenarios",
                                   default=False)
    guiOverride["--MSSim-plot-adjusted-scenarios"] = {}
    guiOverride["--MSSim-plot-adjusted-scenarios"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-adjust-load",
                                   help="make an adjustment of the load scenarios according to the actual values "
                                        "at the stage time, for the intra day stages.",
                                   action="store_true",
                                   dest="MSSim_adjust_load",
                                   default=False)                               
    guiOverride["--MSSim-adjust-load"] = {}
    guiOverride["--MSSim-adjust-load"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-adjust-renewables",
                                   help="make an adjustment of the wind and solar scenarios according "
                                        "to the actual values at the stage time, for the intra day stages.",
                                   action="store_true",
                                   dest="MSSim_adjust_renewables",
                                   default=False)                                
    guiOverride["--MSSim-adjust-renewables"] = {}
    guiOverride["--MSSim-adjust-renewables"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-write-RUC-lp-files",
                                   help="write a lp file of the master instance for stochastic RUC in each stage, "
                                        "in MSSim results directory",
                                   action="store_true",
                                   dest="MSSim_write_RUC_lp_files",
                                   default=False) 
    guiOverride["--MSSim-write-RUC-lp-files"] = {}
    guiOverride["--MSSim-write-RUC-lp-files"]['bpa'] = False
    guiOverride["--MSSim-write-RUC-lp-files"]['advanced'] = True
                              
    MSSimulator_options.add_option("--MSSim-pprint-RUC",
                                   help="makes a pprint of the master instance for stochastic RUC in each stage, "
                                        "in MSSim results directory",
                                   action="store_true",
                                   dest="MSSim_pprint_RUC",
                                   default=False) 
    guiOverride["--MSSim-pprint-RUC"] = {}
    guiOverride["--MSSim-pprint-RUC"]['bpa'] = False
    guiOverride["--MSSim-pprint-RUC"]['advanced'] = True
                                  
    MSSimulator_options.add_option("--MSSim-perfect-updates-scenarios",
                                   help="enable to compute a multistage scenario tree that integrates "
                                        "perfect updates on stage 2",
                                   action="store_true",
                                   dest="MSSim_perfect_updates_scenarios",
                                   default=False) 
    guiOverride["--MSSim-perfect-updates-scenarios"] = {}
    guiOverride["--MSSim-perfect-updates-scenarios"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-initial-conditions-from-file",
                                   help="look for the file 'final_conditions.dat' in the results directory "
                                        "where the final conditions from a day should have been written.",
                                   action="store",
                                   dest="MSSim_initial_conditions_from_file",
                                   type="string",
                                   default=None) 
    guiOverride["--MSSim-initial-conditions-from-file"] = {}
    guiOverride["--MSSim-initial-conditions-from-file"]['bpa'] = False
                                   
    MSSimulator_options.add_option("--MSSim-use-adjusted-scenarios",
                                   help="enable to take the scenarios from the folder adjusted_scenarios in stage 2",
                                   action="store_true",
                                   dest="MSSim_use_adjusted_scenarios",
                                   default=False) 
    guiOverride["--MSSim-use-adjusted-scenarios"] = {}
    guiOverride["--MSSim-use-adjusted-scenarios"]['bpa'] = False
    guiOverride["--MSSim-use-adjusted-scenarios"]['advanced'] = True
                                   
    MSSimulator_options.add_option("--MSSim-use-same-model-for-stage2",
                                   help="enable to use the same model 'useLastCommitments' for multistage as well as "
                                        "TS and det., during the second RUC in stage 2.",
                                   action="store_true",
                                   dest="MSSim_use_same_model_for_stage2",
                                   default=False)
    guiOverride["--MSSim-use-same-model-for-stage2"] = {}
    guiOverride["--MSSim-use-same-model-for-stage2"]['bpa'] = False
    guiOverride["--MSSim-use-same-model-for-stage2"]['advanced'] = True
                                   
    MSSimulator_options.add_option("--MSSimulator-RUC-TimeLimit",
                                   help="If specified, will set a time limit for the resolution of stochastic RUC. "
                                        "Time given in seconds.",
                                   action="store",
                                   dest="MSSimulator_RUC_TimeLimit",
                                   type=int,
                                   default=None)
    guiOverride["--MSSimulator-RUC-TimeLimit"] = {}
    guiOverride["--MSSimulator-RUC-TimeLimit"]['bpa'] = False
    guiOverride["--MSSimulator-RUC-TimeLimit"]['advanced'] = True

    MSSimulator_options.add_option("--L1Linf-solver",
                                   help="Solver name for L1 and L-infinity norms (e.g., cplex",
                                   action="store",
                                   dest="L1Linf_solver",
                                   type="string",
                                   default="cbc")

    populator_options.add_option("--start-date",
                                 help="The start date for the simulation - specified in MM-DD-YYYY format. "
                                      "Defaults to 01-01-2011.",
                                 action="store",
                                 dest="start_date",
                                 type="string",
                                 default="01-01-2011")
    guiOverride["--start-date"] = {}
    guiOverride["--start-date"]['bpa'] = False

    populator_options.add_option("--num-days",
                                 help="The number of days for which to create scenearios",
                                 action="store",
                                 dest="num_days",
                                 type="int",
                                 default=7)
    guiOverride["--num-days"] = {}
    guiOverride["--num-days"]['bpa'] = False

    populator_options.add_option("--disable-fit-db-generation",
                                 help="Disable generation of fit database. "
                                      "Assumed to be in the output directory "
                                      "(specified wth the --output_directory option), "
                                      "in a sub-directory named \"fitdata\".",
                                 action="store_true",
                                 dest="disable_fit_db_generation",
                                 default=False)
    guiOverride["--disable-fit-db-generation"] = {}
    guiOverride["--disable-fit-db-generation"]['bpa'] = False
    guiOverride['--disable-fit-db-generation']['advanced'] = True

    populator_options.add_option("--output-directory",
                                 help="The root directory to which all of the generated simulation files and "
                                      "associated data are written.",
                                 action="store",
                                 dest="output_directory",
                                 type="string",
                                 default="outdir")
    guiOverride["--output-directory"] = {}
    guiOverride["--output-directory"]['bpa'] = False

#############################
#  PRESCIENT ONLY OPTIONS   #
#############################

# # PRESCIENT_INPUT_OPTIONS

    input_simulation_options.add_option('--data-directory',
                                        help='Specifies the directory to pull data from for confcomp.',
                                        action='store',
                                        dest='data_directory',
                                        type='string',
                                        default="confcomp_data")
    guiOverride['--data-directory'] = {}
    guiOverride['--data-directory']['bpa'] = False

    input_simulation_options.add_option("--plot-individual-generators",
                                        help="Display stack graph plots showing individual generators, "
                                             "as opposed to by-type generator behavior. Defaults to False.",
                                        action="store_true",
                                        dest="plot_individual_generators",
                                        default=False)
    guiOverride["--plot-individual-generators"] = {}
    guiOverride["--plot-individual-generators"]["--plot-individual-generators"] = False

    input_simulation_options.add_option("--plot-peak-demand",
                                        help="Generate stack graphs using the specified peak demand, "
                                             "instead of the thermal fleet capacity. "
                                             "Defaults to 0.0, indicating disabled.",
                                        action="store",
                                        dest="plot_peak_demand",
                                        type="float",
                                        default=0.0)
    guiOverride["--plot-peak-demand"] = {}
    guiOverride["--plot-peak-demand"]["--plot-peak-demand"] = False

    input_simulation_options.add_option("--disable-plot-legend",
                                        help="Disable generation of plot legends on individual stack graph plots. "
                                             "Default to False.",
                                        action="store_true",
                                        dest="disable_plot_legend",
                                        default=False)
    guiOverride["--disable-plot-legend"] = {}
    guiOverride["--disable-plot-legend"]["--disable-plot-legend"] = False

    input_simulation_options.add_option('--disable-ruc',
                                        help='Disables reliability unit commitment during simulation.',
                                        action='store_true',
                                        dest='disable_ruc',
                                        default=False)
    guiOverride['--disable-ruc'] = {}
    guiOverride['--disable-ruc']['bpa'] = False

    input_simulation_options.add_option('--run-deterministic-ruc',
                                        help='Invokes deterministic instead of stochastic reliability unit commitment '
                                             'during simulation.',
                                        action='store_true',
                                        dest='run_deterministic_ruc',
                                        default=False)
    guiOverride['--run-deterministic-ruc'] = {}
    guiOverride['--run-deterministic-ruc']['bpa'] = False

    extension_options.add_option('--plugin',
                                 help='The name of a python module that extends prescient behavior',
                                 type="string",
                                 action='append',
                                 dest='plugin_modules',
                                 default=[])

    input_simulation_options.add_option('--simulator-plugin',
                                        help='If the user has an alternative methods for the various simulator functions,'
                                             ' they should be specified here, e.g., prescient.plugin.my_special_plugin.',
                                        action='store',
                                        dest='simulator_plugin',
                                        default=None)
    guiOverride['--simulator-plugin'] = {}
    guiOverride['--simulator-plugin']['bpa'] = False

    input_simulation_options.add_option('--deterministic-ruc-solver-plugin',
                                        help='If the user has an alternative method to solve the deterministic RUCs,'
                                             ' it should be specified here, e.g., prescient.plugin.my_special_plugin.'
                                             'NOTE: This option is ignored if --simulator-plugin is used.',
                                        action='store',
                                        dest='deterministic_ruc_solver_plugin',
                                        default=None)
    guiOverride['--deterministic-ruc-solver-plugin'] = {}
    guiOverride['--deterministic-ruc-solver-plugin']['bpa'] = False

    input_simulation_options.add_option('--run-ruc-with-next-day-data',
                                        help='When running the RUC, use the data for the next day '
                                             'for tailing hours.',
                                        action='store_true',
                                        dest='run_ruc_with_next_day_data',
                                        default=False)
    guiOverride['--run-ruc-with-next-day-data'] = {}
    guiOverride['--run-ruc-with-next-day-data']['bpa'] = False

    input_simulation_options.add_option('--run-sced-with-persistent-forecast-errors',
                                        help='Create all SCED instances assuming persistent forecast error, '
                                             'instead of the default prescience. '
                                             'Only applicable when running deterministic RUC.',
                                        action='store_true',
                                        dest='run_sced_with_persistent_forecast_errors',
                                        default=False)
    guiOverride['--run-sced-with-persistent-forecast-errors'] = {}
    guiOverride['--run-sced-with-persistent-forecast-errors']['bpa'] = False

    input_simulation_options.add_option('--ruc-prescience-hour',
                                        help='Hour before which linear blending of forecast and actuals '
                                             'takes place when running deterministic ruc. A value of '
                                             '0 indicates we always take the forecast. Default is 0.',
                                        action='store',
                                        dest='ruc_prescience_hour',
                                        type=int,
                                        default=0)
    guiOverride['--ruc-prescience-hour'] = {}
    guiOverride['--ruc-prescience-hour']['bpa'] = False

    input_simulation_options.add_option('--disable-sced',
                                        help='Disables hourly or sub-hourly (depending on configuration) '
                                             'SCED during simulation.',
                                        action='store_true',
                                        dest='disable_sced',
                                        default=False)
    guiOverride['--disable-sced'] = {}
    guiOverride['--disable-sced']['bpa'] = False

    input_simulation_options.add_option("-m", '--model-directory',
                                        help="DEPRECATED; no need to set.",
                                        action="callback",
                                        type="string",
                                        callback=lambda a,b,c,d: None)
    guiOverride["-m"] = {}
    guiOverride["-m"]["-m"] = False

    input_simulation_options.add_option('--ruc-execution-hour',
                                        help="Specifies when the the RUC process is executed. "
                                            "Negative values indicate time before horizon, positive after.",
                                        action='store',
                                        dest='ruc_execution_hour',
                                        type='int',
                                        default=16)
    guiOverride['--ruc-execution-hour'] = {}
    guiOverride['--ruc-execution-hour']['bpa'] = False

    input_simulation_options.add_option('--ruc-every-hours',
                                        help="Specifies at which hourly interval the RUC process is executed. "
                                             "Default is 24. Should be a divisor of 24.",
                                        action='store',
                                        dest='ruc_every_hours',
                                        type='int',
                                        default=24)
    guiOverride['--ruc-every-hours'] = {}
    guiOverride['--ruc-every-hours']['bpa'] = False

    populator_options.add_option("--ruc-horizon",
                                 help="The number of hours for which the reliability unit commitment is executed. "
                                      "Must be <= 48 hours and >= --ruc-every-hours. "
                                      "Default is 48.",
                                 action="store",
                                 dest="ruc_horizon",
                                 type="int",
                                 default=48)
    guiOverride["--ruc-horizon"] = {}
    guiOverride["--ruc-horizon"]['bpa'] = False

    input_simulation_options.add_option('--sced-horizon',
                                        help="Specifies the number of hours in the look-ahead horizon "
                                             "when each SCED process is executed.",
                                        action='store',
                                        dest='sced_horizon',
                                        type='int',
                                        default=24)
    guiOverride['--sced-horizon'] = {}
    guiOverride['--sced-horizon']['bpa'] = False

    input_simulation_options.add_option('--sced-frequency-minutes',
                                        help="Specifies how often a SCED will be run, in minutes. "
                                             "Must divide evenly into 60, or be a multiple of 60.",
                                        action='store',
                                        dest='sced_frequency_minutes',
                                        type='int',
                                        default=60)

    input_simulation_options.add_option("--random-seed",
                                        help="Seed the random number generator used in the simulation. "
                                             "Defaults to 0, "
                                             "indicating the seed will be initialized using the current time.",
                                        action="store",
                                        dest="random_seed",
                                        type="int",
                                        default=0)
    guiOverride["--random-seed"] = {}
    guiOverride["--random-seed"]["--random-seed"] = False

    input_simulation_options.add_option("--simulate-out-of-sample",
                                        help="Execute the simulation using an out-of-sample scenario, "
                                             "specified in Scenario_actuals.dat files in the daily input directories. "
                                             "Defaults to False, "
                                             "indicating that either the expected-value scenario will be used "
                                             "(for deterministic RUC) or a random scenario sample will be used "
                                             "(for stochastic RUC).",
                                        action="store_true",
                                        dest="simulate_out_of_sample",
                                        default=False)
    guiOverride["--simulate-out-of-sample"] = {}
    guiOverride["--simulate-out-of-sample"]["--simulate-out-of-sample"] = False

    input_simulation_options.add_option("--reserve-factor",
                                        help="The reserve factor, expressed as a constant fraction of demand, "
                                             "for spinning reserves at each time period of the simulation. "
                                             "Applies to both stochastic RUC and deterministic SCED models.",
                                        action="store",
                                        dest="reserve_factor",
                                        type="float",
                                        default=0.0)
    guiOverride["--reserve-factor"] = {}
    guiOverride["--reserve-factor"]["--reserve-factor"] = False

    input_simulation_options.add_option('--compute-market-settlements',
                                        help="Solves a day-ahead as well as real-time market and reports "
                                             "the daily profit for each generator based on the computed prices. "
                                             "Only compatible with deterministic RUC (for now). "
                                             "Defaults to False.",
                                        action='store_true',
                                        dest='compute_market_settlements',
                                        default=False)
    guiOverride['--compute-market-settlements'] = {}
    guiOverride['--compute-market-settlements']['bpa'] = False

    input_simulation_options.add_option('--price-threshold',
                                        help="Maximum possible value the price can take "
                                             "If the price exceeds this value due to Load Mismatch, then "
                                             "it is set to this value.",
                                        action='store',
                                        type='float',
                                        dest='price_threshold',
                                        default=10000.)
    guiOverride['--price-threshold'] = {}
    guiOverride['--price-threshold']['bpa'] = False

    input_simulation_options.add_option('--reserve-price-threshold',
                                        help="Maximum possible value the reserve price can take "
                                             "If the reserve price exceeds this value, then "
                                             "it is set to this value.",
                                        action='store',
                                        type='float',
                                        dest='reserve_price_threshold',
                                        default=1000.)
    guiOverride['--reserve-price-threshold'] = {}
    guiOverride['--reserve-price-threshold']['bpa'] = False
                             
# # PRESCIENT_SOLVER_OPTIONS

    solver_simulation_options.add_option('--solve-with-ph',
                                         help="Indicate that stochastic optimization models should be solved with PH, "
                                              "as opposed to directly via the extensive form.",
                                         action='store_true',
                                         dest='solve_with_ph',
                                         default=False)
    guiOverride['--solve-with-ph'] = {}
    guiOverride['--solve-with-ph']['bpa'] = False

    solver_simulation_options.add_option('--ph-mode',
                                         help="Specify the mode in which PH should be executed. "
                                              "Currently accepted values are: 'serial' and 'localmpi'.",
                                         action='store',
                                         dest='ph_mode',
                                         type='string',
                                         default="serial")
    guiOverride['--ph-mode'] = {}
    guiOverride['--ph-mode']['bpa'] = False

    solver_simulation_options.add_option('--ph-max-iterations',
                                         help="Specify the maximum number of iterations for PH. Defaults to 20.",
                                         action='store',
                                         dest='ph_max_iterations',
                                         type='int',
                                         default=20)
    guiOverride['--ph-max-iterations'] = {}
    guiOverride['--ph-max-iterations']['bpa'] = False

    solver_simulation_options.add_option('--ph-options',
                                         help="The command-line options for PH. Defaults to the empty string.",
                                         action='store',
                                         dest='ph_options',
                                         type='string',
                                         default="")
    guiOverride['--ph-options'] = {}
    guiOverride['--ph-options']['bpa'] = False

    solver_simulation_options.add_option('--sced-solver',
                                         help="Choose the desired solver type for SCEDs",
                                         action='store',
                                         dest='sced_solver_type',
                                         type='string',
                                         default='cbc')
    guiOverride['--sced-solver'] = {}
    guiOverride['--sced-solver']['bpa'] = False

    solver_simulation_options.add_option('--deterministic-ruc-solver',
                                         help="Choose the desired solver type for deterministic RUCs",
                                         action='store',
                                         dest='deterministic_ruc_solver_type',
                                         type='string',
                                         default='cbc')
    guiOverride['--deterministic-ruc-solver'] = {}
    guiOverride['--deterministic-ruc-solver']['bpa'] = False

    solver_simulation_options.add_option('--ef-ruc-solver',
                                         help="Choose the desired solver type for stochastic extensive form RUCs",
                                         action='store',
                                         dest='ef_ruc_solver_type',
                                         type='string',
                                         default='cbc')
    guiOverride['--ef-ruc-solver'] = {}
    guiOverride['--ef-ruc-solver']['bpa'] = False

    solver_simulation_options.add_option('--ph-ruc-subproblem-solver',
                                         help="Choose the desired sub-problem solver type for progressive heding"
                                         " when solving stochastic RUCs",
                                         action='store',
                                         dest='ph_ruc_subproblem_solver_type',
                                         type='string',
                                         default='cbc')
    guiOverride['--ph-ruc-subproblem-solver'] = {}
    guiOverride['--ph-ruc-subproblem-solver']['bpa'] = False

    solver_simulation_options.add_option("--sced-solver-options",
                                         help="Solver options applied to all SCED solves",
                                         action="append",
                                         dest="sced_solver_options",
                                         type="string",
                                         default=[])
    guiOverride["--sced-solver-options"] = {}
    guiOverride["--sced-solver-options"]["--sced-solver-options"] = False

    solver_simulation_options.add_option("--deterministic-ruc-solver-options",
                                         help="Solver options applied to all deterministic RUC solves",
                                         action="append",
                                         dest="deterministic_ruc_solver_options",
                                         type="string",
                                         default=[])
    guiOverride["--deterministic-ruc-solver-options"] = {}
    guiOverride["--deterministic-ruc-solver-options"]["--deterministic-ruc-solver-options"] = False

    solver_simulation_options.add_option("--stochastic-ruc-ef-solver-options",
                                         help="Solver options applied to all stochastic RUC EF solves",
                                         action="append",
                                         dest="stochastic_ruc_ef_solver_options",
                                         type="string",
                                         default=[])
    guiOverride["--stochastic-ruc-ef-solver-options"] = {}
    guiOverride["--stochastic-ruc-ef-solver-options"]["--stochastic-ruc-ef-solver-options"] = False

    solver_simulation_options.add_option('--python-io',
                                         help='Use python bindings rather than file io.',
                                         action='store_true',
                                         dest='python_io',
                                         default=False)
    guiOverride['--python-io'] = {}
    guiOverride['--python-io']['bpa'] = False
    guiOverride['--python-io']['advanced'] = True

    solver_simulation_options.add_option('--write-deterministic-ruc-instances',
                                         help='Write all individual SCED instances.',
                                         action='store_true',
                                         dest='write_deterministic_ruc_instances',
                                         default=False)
    guiOverride['--write-sced-instances'] = {}
    guiOverride['--write-sced-instances']['bpa'] = False

    solver_simulation_options.add_option('--write-sced-instances',
                                         help='Write all individual SCED instances.',
                                         action='store_true',
                                         dest='write_sced_instances',
                                         default=False)
    guiOverride['--write-sced-instances'] = {}
    guiOverride['--write-sced-instances']['bpa'] = False

    solver_simulation_options.add_option('--relax-ramping-if-infeasible',
                                         help='Relax the generator ramp rates and re-solve '
                                              'if an infeasibility is found in SCED.',
                                         action='store_true',
                                         dest='relax_ramping_if_infeasible',
                                         default=False)
    guiOverride['--relax-ramping-if-infeasible'] = {}
    guiOverride['--relax-ramping-if-infeasible']['bpa'] = False

    solver_simulation_options.add_option('--relax-ramping-factor',
                                         help='Factor by which nominal ramping limits are scaled each time '
                                              'they are relaxed. Growth is geometric. Factor should be small, '
                                              'e.g., 0.01.',
                                         action='store',
                                         type='float',
                                         dest='relax_ramping_factor',
                                         default=0.01)
    guiOverride['--relax-ramping-factor'] = {}
    guiOverride['--relax-ramping-factor']['bpa'] = False
    guiOverride['--relax-ramping-factor']['advanced'] = True

    solver_simulation_options.add_option('--error-if-infeasible',
                                         help='Exit out if an infeasibility is found in SCED / RUC.',
                                         action='store_true',
                                         dest='error_if_infeasible',
                                         default=False)
    guiOverride['--error-if-infeasible'] = {}
    guiOverride['--error-if-infeasible']['bpa'] = False

    solver_simulation_options.add_option('--pause-if-infeasible',
                                         help='Pause after outputting information '
                                              'if an infeasibility is found in SCED / RUC.',
                                         action='store_true',
                                         dest='pause_if_infeasible',
                                         default=False)
    guiOverride['--pause-if-infeasible'] = {}
    guiOverride['--pause-if-infeasible']['bpa'] = False

    solver_simulation_options.add_option('--print-sced',
                                         help='Print results from SCED solves.',
                                         action='store_true',
                                         dest='print_sced',
                                         default=False)
    guiOverride['--print-sced'] = {}
    guiOverride['--print-sced']['bpa'] = False

    solver_simulation_options.add_option('--ef-mipgap',
                                         help='Specifies the mipgap for all stochsatic RUC extensive form solves '
                                              '(including those invoked by PH, if any).',
                                         action='store',
                                         dest='ef_mipgap',
                                         type='float',
                                         default=0.01)
    guiOverride['--ef-mipgap'] = {}
    guiOverride['--ef-mipgap']['bpa'] = False

    solver_simulation_options.add_option('--ruc-mipgap',
                                         help="Specifies the mipgap for all deterministic RUC solves.",
                                         action='store',
                                         dest='ruc_mipgap',
                                         type='float',
                                         default=0.01)
    guiOverride['--ruc-mipgap'] = {}
    guiOverride['--ruc-mipgap']['bpa'] = False

    solver_simulation_options.add_option('--sced-mipgap',
                                         help="Specifies the mipgap for all deterministic SCED solves. "
                                              "Necessary due to possibility of non-convex piecewise cost curves and "
                                              "the induced SOS variables.",
                                         action='store',
                                         dest='sced_mipgap',
                                         type='float',
                                         default=0.0)
    guiOverride['--sced-mipgap'] = {}
    guiOverride['--sced-mipgap']['bpa'] = False

    solver_simulation_options.add_option('--failure-rate',
                                         help='Specifies the hourly failure rate of elements in the simulation.',
                                         action='store',
                                         dest='hourly_failure_rate',
                                         type='float',
                                         default=0.0)
    guiOverride['--failure-rate'] = {}
    guiOverride['--failure-rate']['bpa'] = False

    solver_simulation_options.add_option('--symbolic-solver-labels',
                                         help="When interfacing with the solver, "
                                              "use symbol names derived from the model.",
                                         action='store_true',
                                         dest='symbolic_solver_labels',
                                         default=False)
    guiOverride['--symbolic-solver-labels'] = {}
    guiOverride['--symbolic-solver-labels']['bpa'] = False
    guiOverride['--symbolic-solver-labels']['advanced'] = True

    solver_simulation_options.add_option('--warmstart-ruc',
                                         help="Partially warmstart each RUC with previous solution, default is False.",
                                         action='store_true',
                                         dest='warmstart_ruc',
                                         default=False)
    guiOverride['--warmstart-ruc'] = {}
    guiOverride['--warmstart-ruc']['bpa'] = False

    solver_options.add_option('-k', '--keep-solver-files',
                              help="Retain temporary input and output files for scenario sub-problem solves",
                              action="store_true",
                              dest="keep_solver_files",
                              default=False)
    guiOverride['--keep-solver-files'] = {}
    guiOverride['--keep-solver-files']['bpa'] = False
    guiOverride['--keep-solver-files']['advanced'] = True
    
    solver_options.add_option('--enable-quick-start-generator-commitment',
                              help="Allows quick start generators to be committed if load shedding occurs",
                              action="store_true",
                              dest="enable_quick_start_generator_commitment",
                              default=False)

    solver_simulation_options.add_option('--day-ahead-pricing',
                                         help="Choose the pricing mechanism for the day-ahead market. Choices are "
                                              "LMP -- locational marginal price, "
                                              "ELMP -- enhanced locational marginal price, and "
                                              "aCHP -- approximated convex hull price. "
                                              "Default is aCHP.",
                                         action='store',
                                         dest='day_ahead_pricing',
                                         type='string',
                                         default='aCHP')
    guiOverride['--day-ahead-pricing'] = {}
    guiOverride['--day-ahead-pricing']['bpa'] = False

    solver_simulation_options.add_option('--relax-t0-ramping-initial-day',
                                         help="Relax t0 ramping for the initial RUC. Ignored for PH.",
                                         action='store_true',
                                         dest='relax_t0_ramping_initial_day',
                                         default=False)
    guiOverride['--relax-t0-ramping-initial-day'] = {}
    guiOverride['--relax-t0-ramping-initial-day']['bpa'] = False

                              
# # PRESCIENT_OUTPUT_OPTIONS

    output_simulation_options.add_option('--verbose',
                                         help='Generate verbose output, beyond the usual status output. '
                                              'Default is False.',
                                         action='store_true',
                                         dest='verbose',
                                         default=False)
    guiOverride['--verbose'] = {}
    guiOverride['--verbose']['bpa'] = False

    output_simulation_options.add_option('--output-ruc-initial-conditions',
                                         help="Output ruc (deterministic or stochastic) initial conditions prior "
                                              "to each solve. Default is False.",
                                         action='store_true',
                                         dest='output_ruc_initial_conditions',
                                         default=False)
    guiOverride['--output-ruc-initial-conditions'] = {}
    guiOverride['--output-ruc-initial-conditions']['bpa'] = False

    output_simulation_options.add_option('--output-ruc-solutions',
                                         help="Output ruc (deterministic or stochastic) solutions following each solve."
                                              " Default is False.",
                                         action='store_true',
                                         dest='output_ruc_solutions',
                                         default=False)
    guiOverride['--output-ruc-solutions'] = {}
    guiOverride['--output-ruc-solutions']['bpa'] = False

    output_simulation_options.add_option('--output-ruc-dispatches',
                                         help="Output ruc (deterministic or stochastic) scenario dispatches following "
                                              "each solve. Default is False. "
                                              "Only effective if --output-ruc-solutions is enabled.",
                                         action='store_true',
                                         dest='output_ruc_dispatches',
                                         default=False)
    guiOverride['--output-ruc-dispatches'] = {}
    guiOverride['--output-ruc-dispatches']['bpa'] = False

    output_simulation_options.add_option('--output-sced-initial-conditions',
                                         help='Output sced initial conditions prior to each solve. Default is False.',
                                         action='store_true',
                                         dest='output_sced_initial_conditions',
                                         default=False)
    guiOverride['--output-sced-initial-conditions'] = {}
    guiOverride['--output-sced-initial-conditions']['bpa'] = False

    output_simulation_options.add_option('--output-sced-demands',
                                         help='Output sced demands prior to each solve. Default is False.',
                                         action='store_true',
                                         dest='output_sced_demands',
                                         default=False)
    guiOverride['--output-sced-demands'] = {}
    guiOverride['--output-sced-demands']['bpa'] = False

    output_simulation_options.add_option('--output-sced-solutions',
                                         help='Output sced solutions following each solve. Default is False.',
                                         action='store_true',
                                         dest='output_sced_solutions',
                                         default=False)
    guiOverride['--output-sced-solutions'] = {}
    guiOverride['--output-sced-solutions']['bpa'] = False

    output_simulation_options.add_option('--output-solver-logs',
                                         help="Output solver logs during execution.",
                                         action='store_true',
                                         dest='output_solver_logs',
                                         default=False)
    guiOverride['--output-solver-logs'] = {}
    guiOverride['--output-solver-logs']['bpa'] = False

    output_simulation_options.add_option('--output-max-decimal-place',
                                         help="When writing summary files, this rounds the output to the "
                                              "specified accuracy. Default is 6.",
                                         action='store',
                                         type='int',
                                         dest='output_max_decimal_place',
                                         default=6)
    guiOverride['--output-max-decimal-place'] = {}
    guiOverride['--output-max-decimal-place']['bpa'] = False

    output_simulation_options.add_option('--disable-stackgraphs',
                                         help="Disable stackgraph generation",
                                         action='store_true',
                                         dest='disable_stackgraphs',
                                         default=False)


                              
# # PRESCIENT_OTHER_OPTIONS

    other_simulation_options.add_option('--profile',
                                        help='Enable profiling of Python code. '
                                             'The value of this option is the number of functions that are summarized.',
                                        action='store',
                                        type='int',
                                        dest='profile',
                                        default=0)
    guiOverride['--profile'] = {}
    guiOverride['--profile']['bpa'] = False

    other_simulation_options.add_option('--traceback',
                                     help='When an exception is thrown, show the entire call stack.'
                                          'Ignored if profiling is enabled. Default is False.',
                                     action='store_true',
                                     dest='traceback',
                                     default=False)
    guiOverride['--traceback'] = {}
    guiOverride['--traceback']['bpa'] = False
                             
    # options for testing purpose. Jan 2015

    structure_options.add_option('--recompute-probabilities-using-copula',
                                 help='RESEARCH! Take correlation between hours into account and '
                                      'recompute probabilities for scenarios. Set to copula, gauss. '
                                      'copula will use a linear combination of independent and '
                                      'anti or comonotonic copulas to recompute the probabilitis (fast). '
                                      'gauss will estimate a gaussian copula for non-trivial day part separators and '
                                      'use it to estimate the probabilities (slow).',
                                 action='store',
                                 type='string',
                                 dest='recompute_probabilities_using_copula',
                                 default=None)
    guiOverride['--recompute-probabilities-using-copula'] = {}
    guiOverride['--recompute-probabilities-using-copula']['bpa'] = False
    guiOverride['--recompute-probabilities-using-copula']['advanced'] = True
    directory_options.add_option('--copula-folder',
                                 help='RESEARCH! '
                                      'If the gaussian copula (pdf) of each day should be written into a file, '
                                      'set this option to where those copulas should be stored. '
                                      'default is None to disable writing copulas.',
                                 action='store',
                                 type='string',
                                 dest='copula_folder',
                                 default=None)
    guiOverride['--copula-folder'] = {}
    guiOverride['--copula-folder']['bpa'] = False
    guiOverride['--copula-folder']['advanced'] = True
    populator_options.add_option('--scatter-plot-copula-data',
                                 help='RESEARCH! '
                                      'Set, if scatter plots of the data used to derive the copula should be created. '
                                      'They will be put in a day specific subfolder of the copula_folder.',
                                 action='store_true',
                                 dest='scatter_plot_copula_data')
    guiOverride['--scatter-plot-copula-data'] = {}
    guiOverride['--scatter-plot-copula-data']['bpa'] = True

    return parser, guiOverride
    # return parser

from pyutilib.misc import import_file
def _initialize_plugin(module_name):
    '''
    Loads a plugin, allowing it to register its plugin behaviors
    '''
    custom_module = import_file(module_name)

if __name__ == '__main__':
    print("master_options.py cannot run from the command line.")
    sys.exit(1)
