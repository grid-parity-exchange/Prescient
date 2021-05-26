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
    parser = _construct_inner_options_parser()
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
    return parser

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

    populator_options.add_option("--start-date",
                                 help="The start date for the simulation - specified in MM-DD-YYYY format. "
                                      "Defaults to 01-01-2011.",
                                 action="store",
                                 dest="start_date",
                                 type="string",
                                 default="01-01-2011")

    populator_options.add_option("--num-days",
                                 help="The number of days for which to create scenearios",
                                 action="store",
                                 dest="num_days",
                                 type="int",
                                 default=7)

    populator_options.add_option("--output-directory",
                                 help="The root directory to which all of the generated simulation files and "
                                      "associated data are written.",
                                 action="store",
                                 dest="output_directory",
                                 type="string",
                                 default="outdir")

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

    input_simulation_options.add_option('--run-deterministic-ruc',
                                        help='DEPRECATED, must be used. Invokes deterministic instead '
                                             'of stochastic reliability unit commitment during simulation.',
                                        action='store_true',
                                        dest='run_deterministic_ruc',
                                        default=True)

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

    input_simulation_options.add_option('--deterministic-ruc-solver-plugin',
                                        help='If the user has an alternative method to solve the deterministic RUCs,'
                                             ' it should be specified here, e.g., prescient.plugin.my_special_plugin.'
                                             'NOTE: This option is ignored if --simulator-plugin is used.',
                                        action='store',
                                        dest='deterministic_ruc_solver_plugin',
                                        default=None)

    input_simulation_options.add_option('--run-ruc-with-next-day-data',
                                        help='When running the RUC, use the data for the next day '
                                             'for tailing hours.',
                                        action='store_true',
                                        dest='run_ruc_with_next_day_data',
                                        default=False)

    input_simulation_options.add_option('--run-sced-with-persistent-forecast-errors',
                                        help='Create all SCED instances assuming persistent forecast error, '
                                             'instead of the default prescience. '
                                             'Only applicable when running deterministic RUC.',
                                        action='store_true',
                                        dest='run_sced_with_persistent_forecast_errors',
                                        default=False)

    input_simulation_options.add_option('--ruc-prescience-hour',
                                        help='Hour before which linear blending of forecast and actuals '
                                             'takes place when running deterministic ruc. A value of '
                                             '0 indicates we always take the forecast. Default is 0.',
                                        action='store',
                                        dest='ruc_prescience_hour',
                                        type=int,
                                        default=0)

    input_simulation_options.add_option('--ruc-execution-hour',
                                        help="Specifies when the the RUC process is executed. "
                                            "Negative values indicate time before horizon, positive after.",
                                        action='store',
                                        dest='ruc_execution_hour',
                                        type='int',
                                        default=16)

    input_simulation_options.add_option('--ruc-every-hours',
                                        help="Specifies at which hourly interval the RUC process is executed. "
                                             "Default is 24. Should be a divisor of 24.",
                                        action='store',
                                        dest='ruc_every_hours',
                                        type='int',
                                        default=24)

    populator_options.add_option("--ruc-horizon",
                                 help="The number of hours for which the reliability unit commitment is executed. "
                                      "Must be <= 48 hours and >= --ruc-every-hours. "
                                      "Default is 48.",
                                 action="store",
                                 dest="ruc_horizon",
                                 type="int",
                                 default=48)

    input_simulation_options.add_option('--sced-horizon',
                                        help="Specifies the number of time periods "
                                             "in the look-ahead horizon for each SCED. "
                                             "Must be at least 1.",
                                        action='store',
                                        dest='sced_horizon',
                                        type='int',
                                        default=1)

    input_simulation_options.add_option('--sced-frequency-minutes',
                                        help="Specifies how often a SCED will be run, in minutes. "
                                             "Must divide evenly into 60, or be a multiple of 60.",
                                        action='store',
                                        dest='sced_frequency_minutes',
                                        type='int',
                                        default=60)

    input_simulation_options.add_option("--enforce-sced-shutdown-ramprate",
                                        help="Enforces shutdown ramp-rate constraints in the SCED. "
                                             "Enabling this options requires a long SCED look-ahead "
                                             "(at least an hour) to ensure the shutdown ramp-rate "
                                             "constraints can be statisfied.",
                                        action="store_true",
                                        dest="enforce_sced_shutdown_ramprate",
                                        default=False)

    input_simulation_options.add_option("--no-startup-shutdown-curves",
                                        help="For thermal generators, do not infer startup/shutdown "
                                             "ramping curves when starting-up and shutting-down.",
                                        action="store_true",
                                        dest="no_startup_shutdown_curves",
                                        default=False)

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

    input_simulation_options.add_option("--reserve-factor",
                                        help="The reserve factor, expressed as a constant fraction of demand, "
                                             "for spinning reserves at each time period of the simulation. "
                                             "Applies to both stochastic RUC and deterministic SCED models.",
                                        action="store",
                                        dest="reserve_factor",
                                        type="float",
                                        default=0.0)

    input_simulation_options.add_option('--compute-market-settlements',
                                        help="Solves a day-ahead as well as real-time market and reports "
                                             "the daily profit for each generator based on the computed prices. "
                                             "Only compatible with deterministic RUC (for now). "
                                             "Defaults to False.",
                                        action='store_true',
                                        dest='compute_market_settlements',
                                        default=False)

    input_simulation_options.add_option('--price-threshold',
                                        help="Maximum possible value the price can take "
                                             "If the price exceeds this value due to Load Mismatch, then "
                                             "it is set to this value.",
                                        action='store',
                                        type='float',
                                        dest='price_threshold',
                                        default=10000.)

    input_simulation_options.add_option('--reserve-price-threshold',
                                        help="Maximum possible value the reserve price can take "
                                             "If the reserve price exceeds this value, then "
                                             "it is set to this value.",
                                        action='store',
                                        type='float',
                                        dest='reserve_price_threshold',
                                        default=1000.)

# # PRESCIENT_SOLVER_OPTIONS

    solver_simulation_options.add_option('--sced-solver',
                                         help="Choose the desired solver type for SCEDs",
                                         action='store',
                                         dest='sced_solver_type',
                                         type='string',
                                         default='cbc')

    solver_simulation_options.add_option('--deterministic-ruc-solver',
                                         help="Choose the desired solver type for deterministic RUCs",
                                         action='store',
                                         dest='deterministic_ruc_solver_type',
                                         type='string',
                                         default='cbc')

    solver_simulation_options.add_option("--sced-solver-options",
                                         help="Solver options applied to all SCED solves",
                                         action="append",
                                         dest="sced_solver_options",
                                         type="string",
                                         default=[])

    solver_simulation_options.add_option("--deterministic-ruc-solver-options",
                                         help="Solver options applied to all deterministic RUC solves",
                                         action="append",
                                         dest="deterministic_ruc_solver_options",
                                         type="string",
                                         default=[])

    solver_simulation_options.add_option("--stochastic-ruc-ef-solver-options",
                                         help="Solver options applied to all stochastic RUC EF solves",
                                         action="append",
                                         dest="stochastic_ruc_ef_solver_options",
                                         type="string",
                                         default=[])

    solver_simulation_options.add_option('--write-deterministic-ruc-instances',
                                         help='Write all individual SCED instances.',
                                         action='store_true',
                                         dest='write_deterministic_ruc_instances',
                                         default=False)

    solver_simulation_options.add_option('--write-sced-instances',
                                         help='Write all individual SCED instances.',
                                         action='store_true',
                                         dest='write_sced_instances',
                                         default=False)

    solver_simulation_options.add_option('--print-sced',
                                         help='Print results from SCED solves.',
                                         action='store_true',
                                         dest='print_sced',
                                         default=False)

    solver_simulation_options.add_option('--ruc-mipgap',
                                         help="Specifies the mipgap for all deterministic RUC solves.",
                                         action='store',
                                         dest='ruc_mipgap',
                                         type='float',
                                         default=0.01)

    solver_simulation_options.add_option('--symbolic-solver-labels',
                                         help="When interfacing with the solver, "
                                              "use symbol names derived from the model.",
                                         action='store_true',
                                         dest='symbolic_solver_labels',
                                         default=False)

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

# # PRESCIENT_OUTPUT_OPTIONS

    output_simulation_options.add_option('--output-ruc-initial-conditions',
                                         help="Output ruc (deterministic or stochastic) initial conditions prior "
                                              "to each solve. Default is False.",
                                         action='store_true',
                                         dest='output_ruc_initial_conditions',
                                         default=False)

    output_simulation_options.add_option('--output-ruc-solutions',
                                         help="Output ruc (deterministic or stochastic) solutions following each solve."
                                              " Default is False.",
                                         action='store_true',
                                         dest='output_ruc_solutions',
                                         default=False)

    output_simulation_options.add_option('--output-sced-initial-conditions',
                                         help='Output sced initial conditions prior to each solve. Default is False.',
                                         action='store_true',
                                         dest='output_sced_initial_conditions',
                                         default=False)

    output_simulation_options.add_option('--output-sced-demands',
                                         help='Output sced demands prior to each solve. Default is False.',
                                         action='store_true',
                                         dest='output_sced_demands',
                                         default=False)

    output_simulation_options.add_option('--output-solver-logs',
                                         help="Output solver logs during execution.",
                                         action='store_true',
                                         dest='output_solver_logs',
                                         default=False)

    output_simulation_options.add_option('--output-max-decimal-places',
                                         help="When writing summary files, this rounds the output to the "
                                              "specified accuracy. Default is 6.",
                                         action='store',
                                         type='int',
                                         dest='output_max_decimal_places',
                                         default=6)

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

    other_simulation_options.add_option('--traceback',
                                     help='When an exception is thrown, show the entire call stack.'
                                          'Ignored if profiling is enabled. Default is False.',
                                     action='store_true',
                                     dest='traceback',
                                     default=False)

    return parser

from pyutilib.misc import import_file
def _initialize_plugin(module_name):
    '''
    Loads a plugin, allowing it to register its plugin behaviors
    '''
    custom_module = import_file(module_name)

if __name__ == '__main__':
    print("master_options.py cannot run from the command line.")
    sys.exit(1)
