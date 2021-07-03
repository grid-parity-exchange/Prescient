#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

####################################################
#                 Config                           #
####################################################

from __future__ import annotations
import sys
from typing import List

from argparse import ArgumentParser
from pyutilib.misc import import_file
from pyomo.common.config import (ConfigDict,
                                 ConfigValue,
                                 ConfigList,
                                 In,
                                 PositiveInt,
                                 NonNegativeInt,
                                 PositiveFloat,
                                 NonNegativeFloat,
                                 Path,
                                )

from prescient.plugins import PluginRegistrationContext

prescient_persistent_solvers = ("cplex", "gurobi", "xpress")
prescient_solvers = [ s+sa for sa in ["", "_direct", "_persistent"] for s in prescient_persistent_solvers ]
prescient_solvers += ["cbc", "glpk"]

class PrescientConfig(ConfigDict):

    def __init__(self):
        ##########################
        #   CHAIN ONLY OPTIONS   #
        ##########################
        super().__init__()

        # I want to do this:
        # self.plugin_context = PluginRegistrationContext()
        # but since I can't figure out how...
        self.declare("plugin_context", ConfigValue(
            domain=None,
            default=PluginRegistrationContext(),
            description="The object that will handle plugin registration",
            visibility=100
        ))

        # We put this first so that plugins will be registered before any other
        # options are applied, which lets them add custom command line options 
        # before they are potentially used.
        self.declare("plugin", ConfigList(
            domain=_PluginPath(self),
            default=[],
            description="The path of a python module that extends prescient behavior",
        )).declare_as_argument()

        self.declare("start_date", ConfigValue(
            domain=str,
            default="01-01-2020",
            description="The start date for the simulation - specified in MM-DD-YYYY format. "
                        "Defaults to 01-01-2020.",
        )).declare_as_argument()

        self.declare("num_days", ConfigValue(
            domain=PositiveInt,
            default=7,
            description="The number of days to simulate",
        )).declare_as_argument()

        self.declare("output_directory", ConfigValue(
            domain=str,
            default="outdir",
            description="The root directory to which all of the generated simulation files and "
                        "associated data are written.",
        )).declare_as_argument()

        #############################
        #  PRESCIENT ONLY OPTIONS   #
        #############################

        # # PRESCIENT_INPUT_OPTIONS

        self.declare("data_directory", ConfigValue(
            domain=str,
            default="input_data",
            description="Specifies the directory to pull data from",
        )).declare_as_argument()

        self.declare("input_format", ConfigValue(
            domain=In(["dat", "rts-gmlc"]),
            default="dat",
            description="Indicate the format input data is in",
        )).declare_as_argument()

        self.declare("simulator_plugin", ConfigValue(
            domain=str,
            default=None,
            description="If the user has an alternative methods for the various simulator functions,"
                        " they should be specified here, e.g., prescient.plugin.my_special_plugin.",
        )).declare_as_argument()

        self.declare("deterministic_ruc_solver_plugin", ConfigValue(
            domain=str,
            default=None,
            description="If the user has an alternative method to solve the deterministic RUCs,"
                        " it should be specified here, e.g., prescient.plugin.my_special_plugin."
                        " NOTE: This option is ignored if --simulator-plugin is used."
        )).declare_as_argument()

        self.declare("run_ruc_with_next_day_data", ConfigValue(
            domain=bool,
            default=False,
            description="When running the RUC, use the data for the next day "
                        "for tailing hours.",
        )).declare_as_argument()

        self.declare("run_sced_with_persistent_forecast_errors", ConfigValue(
            domain=bool,
            default=False,
            description="Create all SCED instances assuming persistent forecast error, "
                        "instead of the default prescience.",
        )).declare_as_argument()

        self.declare("ruc_prescience_hour", ConfigValue(
            domain=NonNegativeInt,
            default=0,
            description="Hour before which linear blending of forecast and actuals "
                        "takes place when running deterministic ruc. A value of "
                        "0 indicates we always take the forecast. Default is 0.",
        )).declare_as_argument()

        self.declare("ruc_execution_hour", ConfigValue(
            domain=int,
            default=16,
            description="Specifies when the the RUC process is executed. "
                        "Negative values indicate time before horizon, positive after.",
        )).declare_as_argument()

        self.declare("ruc_every_hours", ConfigValue(
            domain=PositiveInt,
            default=24,
            description="Specifies at which hourly interval the RUC process is executed. "
                        "Default is 24. Should be a divisor of 24.",
        )).declare_as_argument()

        self.declare("ruc_horizon", ConfigValue(
            domain=PositiveInt,
            default=48,
            description="The number of hours for which the reliability unit commitment is executed. "
                        "Must be <= 48 hours and >= --ruc-every-hours. "
                        "Default is 48.",
        )).declare_as_argument()

        self.declare("sced_horizon", ConfigValue(
            domain=PositiveInt,
            default=1,
            description="Specifies the number of time periods "
                        "in the look-ahead horizon for each SCED. "
                        "Must be at least 1.",
        )).declare_as_argument()

        self.declare("sced_frequency_minutes", ConfigValue(
            domain=PositiveInt,
            default=60,
            description="Specifies how often a SCED will be run, in minutes. "
                        "Must divide evenly into 60, or be a multiple of 60.",
        )).declare_as_argument()

        self.declare("enforce_sced_shutdown_ramprate", ConfigValue(
            domain=bool,
            default=False,
            description="Enforces shutdown ramp-rate constraints in the SCED. "
                        "Enabling this options requires a long SCED look-ahead "
                        "(at least an hour) to ensure the shutdown ramp-rate "
                        "constraints can be statisfied.",
        )).declare_as_argument()

        self.declare("no_startup_shutdown_curves", ConfigValue(
            domain=bool,
            default=False,
            description="For thermal generators, do not infer startup/shutdown "
                        "ramping curves when starting-up and shutting-down.",
        )).declare_as_argument()

        self.declare("simulate_out_of_sample", ConfigValue(
            domain=bool,
            default=False,
            description="Execute the simulation using an out-of-sample scenario, "
                        "specified in Scenario_actuals.dat files in the daily input directories. "
                        "Defaults to False, "
                        "indicating that either the expected-value scenario will be used "
                        "(for deterministic RUC) or a random scenario sample will be used "
                        "(for stochastic RUC).",
        )).declare_as_argument()

        self.declare("reserve_factor", ConfigValue(
            domain=NonNegativeFloat,
            default=0.0,
            description="The reserve factor, expressed as a constant fraction of demand, "
                        "for spinning reserves at each time period of the simulation. "
                        "Applies to both stochastic RUC and deterministic SCED models.",
        )).declare_as_argument()

        self.declare("compute_market_settlements", ConfigValue(
            domain=bool,
            default=False,
            description="Solves a day-ahead as well as real-time market and reports "
                        "the daily profit for each generator based on the computed prices.",
        )).declare_as_argument()

        self.declare("price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=10000.,
            description="Maximum possible value the price can take "
                        "If the price exceeds this value due to Load Mismatch, then "
                        "it is set to this value.",
        )).declare_as_argument()

        self.declare("reserve_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=1000.,
            description="Maximum possible value the reserve price can take "
                        "If the reserve price exceeds this value, then "
                        "it is set to this value.",
        )).declare_as_argument()

        # # PRESCIENT_SOLVER_OPTIONS

        self.declare("sced_solver", ConfigValue(
            domain=In(prescient_solvers),
            default="cbc",
            description="The name of the Pyomo solver for SCEDs",
        )).declare_as_argument()

        self.declare("deterministic_ruc_solver", ConfigValue(
            domain=In(prescient_solvers),
            default="cbc",
            description="The name of the Pyomo solver for RUCs",
        )).declare_as_argument()

        self.declare("sced_solver_options", ConfigList(
            domain=str,
            default=[],
            description="Solver options applied to all SCED solves",
        )).declare_as_argument()

        self.declare("deterministic_ruc_solver_options", ConfigList(
            domain=str,
            default=[],
            description="Solver options applied to all deterministic RUC solves",
        )).declare_as_argument()

        self.declare("write_deterministic_ruc_instances", ConfigValue(
            domain=bool,
            default=False,
            description="Write all individual SCED instances.",
        )).declare_as_argument()

        self.declare("write_sced_instances", ConfigValue(
            domain=bool,
            default=False,
            description="Write all individual SCED instances.",
        )).declare_as_argument()

        self.declare("print_sced", ConfigValue(
            domain=bool,
            default=False,
            description="Print results from SCED solves.",
        )).declare_as_argument()

        self.declare("ruc_mipgap",  ConfigValue(
            domain=NonNegativeFloat,
            default=0.01,
            description="Specifies the mipgap for all deterministic RUC solves.",
        )).declare_as_argument()

        self.declare("symbolic_solver_labels", ConfigValue(
            domain=bool,
            default=False,
            description="When interfacing with the solver, "
                        "use symbol names derived from the model.",
        )).declare_as_argument()

        self.declare("enable_quick_start_generator_commitment", ConfigValue(
            domain=bool,
            default=False,
            description="Allows quick start generators to be committed if load shedding occurs",
        )).declare_as_argument()

        self.declare("day_ahead_pricing", ConfigValue(
            domain=In(["LMP", "EMLP", "aCHP"]),
            default="aCHP",
            description="Choose the pricing mechanism for the day-ahead market. Choices are "
                        "LMP -- locational marginal price, "
                        "ELMP -- enhanced locational marginal price, and "
                        "aCHP -- approximated convex hull price. "
                        "Default is aCHP.",
        )).declare_as_argument()

        # # PRESCIENT_OUTPUT_OPTIONS

        self.declare("output_ruc_initial_conditions", ConfigValue(
            domain=bool,
            default=False,
            description="Output ruc (deterministic or stochastic) initial conditions prior "
                        "to each solve. Default is False.",
        )).declare_as_argument()

        self.declare("output_ruc_solutions", ConfigValue(
            domain=bool,
            default=False,
            description="Output ruc solutions following each solve."
                        " Default is False.",
        )).declare_as_argument()

        self.declare("output_sced_initial_conditions", ConfigValue(
            domain=bool,
            default=False,
            description="Output sced initial conditions prior to each solve. Default is False.",
        )).declare_as_argument()

        self.declare("output_sced_demands", ConfigValue(
            domain=bool,
            default=False,
            description="Output sced demands prior to each solve. Default is False.",
        )).declare_as_argument()

        self.declare("output_solver_logs", ConfigValue(
            domain=bool,
            default=False,
            description="Output solver logs during execution.",
        )).declare_as_argument()

        self.declare("output_max_decimal_places", ConfigValue(
            domain=PositiveInt,
            default=6,
            description="When writing summary files, this rounds the output to the "
                        "specified accuracy. Default is 6.",
        )).declare_as_argument()

        self.declare("disable_stackgraphs", ConfigValue(
            domain=bool,
            default=False,
            description="Disable stackgraph generation",
        )).declare_as_argument()

    def parse_args(self, args: List[str]) -> ConfigDict:
        parser = _construct_options_parser(self)
        args = parser.parse_args(args=args)
        self.import_argparse(args)
        return self

    def set_value(self, value, skip_implicit=False):
        if value is None:
            return self
        if (type(value) is not dict) and \
           (not isinstance(value, ConfigDict)):
            raise ValueError("Expected dict value for %s.set_value, found %s" %
                             (self.name(True), type(value).__name__))

        if 'plugin' in value:
            self.plugin.append(value['plugin'])
            value = {**value}
            del(value['plugin'])
        super().set_value(value)



def _construct_options_parser(config: PrescientConfig) -> ArgumentParser:
    '''
    Make a new parser that can parse standard and custom command line options.

    Custom options are provided by plugin modules. Plugins are specified
    as command-line arguments "--plugin=<module name>", where <module name>
    refers to a python module. The plugin may provide new command line
    options by calling "prescient.plugins.add_custom_commandline_option"
    when loaded.
    '''

    # To support the ability to add new command line options to the line that
    # is currently being parsed, we convert a standard ArgumentParser into a
    # two-pass parser by replacing the parser's parse_args method with a
    # modified version.  In the modified parse_args, a first pass through the
    # command line finds any --plugin arguments and allows the plugin to 
    # add new options to the parser.  The second pass does a full parse of
    # the command line using the parser's original parse_args method.
    parser = ArgumentParser()
    parser._inner_parse = parser.parse_args

    def outer_parse(args=None, values=None):
        if args is None:
            args = sys.argv[1:]

        # Manually check each argument against --plugin=<module>,
        # give plugins a chance to install their options.
        stand_alone_opt = '--plugin'
        prefix = "--plugin="
        next_arg_is_module = False
        # When a plugin is imported, its
        # plugin behaviors are registered.
        for arg in args:
            if arg.startswith(prefix):
                module_name = arg[len(prefix):]
                config.plugin_context.register_plugin(module_name, config)
            elif arg == stand_alone_opt:
                next_arg_is_module = True
            elif next_arg_is_module:
                module_name = arg
                config.plugin_context.register_plugin(module_name, config)
                next_arg_is_module = False
 
        # load the arguments into the ArgumentParser
        config.initialize_argparse(parser)

        # Remove plugins from args so they don't get re-handled
        i = 0
        while (i < len(args)):
            if args[i].startswith(prefix):
                del(args[i])
            elif args[i] == stand_alone_opt:
                del(args[i])
                del(args[i])
            else:
                i += 1

        # Now parse for real, with any new options in place.
        return parser._inner_parse(args, values)

    parser.parse_args = outer_parse
    return parser

class _PluginPath(Path):
    ''' A Path that registers a plugin when its path is set
    '''
    def __init__(self, parent_config: PrescientConfig):
        self.config = parent_config
        super().__init__()

    def __call__(self, data):
        path = super().__call__(data)
        self.config.plugin_context.register_plugin(path, self.config)
        return path


if __name__ == '__main__':
    print("master_options.py cannot run from the command line.")
    sys.exit(1)
