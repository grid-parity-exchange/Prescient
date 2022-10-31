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
import dateutil.parser

from typing import List
import argparse
from argparse import ArgumentParser
from datetime import date, datetime
from shlex import shlex

from pyomo.common.fileutils import import_file
from pyomo.common.config import (ConfigDict,
                                 ConfigValue,
                                 ConfigList,
                                 DynamicImplicitDomain,
                                 Module,
                                 In,
                                 InEnum,
                                 PositiveInt,
                                 NonNegativeInt,
                                 PositiveFloat,
                                 NonNegativeFloat,
                                 Path,
                                 MarkImmutable
                                )

from prescient.plugins import PluginRegistrationContext
from prescient.data.data_provider_factory import InputFormats
from prescient.engine.modeling_engine import PricingType, SlackType, NetworkType
from prescient.data import data_provider_factory

prescient_persistent_solvers = ("cplex", "gurobi", "xpress")
prescient_solvers = [ s+sa for sa in ["", "_direct", "_persistent"] for s in prescient_persistent_solvers ]
prescient_solvers += ["cbc", "glpk"]

class PrescientConfig(ConfigDict):

    __slots__ = ("plugin_context",)

    def __init__(self):
        ##########################
        #   CHAIN ONLY OPTIONS   #
        ##########################
        super().__init__()

        self.plugin_context = PluginRegistrationContext()

        def register_plugin(key, value):
            ''' Handle intial plugin setup
            Arguments
            ---------
            key - str
                The alias for this plugin in the configuration
            value - str, module, or dict
                If a string, the name of the python module or the python file for this plugin.
                If a module, the plugin's python module.
                If a dict, the initial values for any properties listed in the dict. One of the
                dict's keys MUST be 'module', and must be either a module, a string identifying
                the module, or a string identifying the module's *.py file.
            '''
            # Defaults, if value is not a dict
            mod_spec=value
            init_values = {}

            # Override defaults if value is a dict
            if isinstance(value, dict):
                if 'module' not in value:
                    raise RuntimeError(f"Attempt to register '{key}' plugin without a module attribute")
                mod_spec=value['module']
                init_values = value.copy()
                del(init_values['module'])

            domain = Module()
            module = domain(mod_spec)

            c = module.get_configuration(key)
            if c is None:
                c = ConfigDict()
            c.declare('module', ConfigValue(module, domain=domain))
            MarkImmutable(c.get('module'))
            c.set_value(init_values)
            return c

        class _PluginParserAction(argparse.Action):
            def __call__(self, parser, namespace, value, opt_str=None):
                # Split out the alias from the module name
                split_value = value.split(':', 1)
                if len(split_value) == 1:
                    raise ValueError("No alias found in plugin specification.  Correct format: <alias>:<module_path_or_name>")
                alias, mod_name = split_value

                # Load the module
                domain = Module()
                plugin_module = domain(mod_name)

                # Save the plugin name and module to the namespace
                if not hasattr(namespace, 'CONFIGBLOCK.plugin'):
                    namespace.__dict__['CONFIGBLOCK.plugin'] = {}
                namespace.__dict__['CONFIGBLOCK.plugin'][alias] = plugin_module

                # Get the module's config, which may identify new config options
                plugin_config = plugin_module.get_configuration(alias)

                # If it has any new options, create a sub-parser and store it on the main parser
                # (adding new options to the main parser mid-parse doesn't work)
                if plugin_config:
                    # Give the plugin config a parent so names work out correctly
                    if not hasattr(parser, '_plugin_configs'):
                        parser._plugin_configs = ConfigDict()
                        parser._plugin_configs.declare('plugin', ConfigDict(implicit=True))
                    parser._plugin_configs.plugin.declare(alias, plugin_config)

                    # Add a new plugin parser to the main parser
                    plugin_parser = argparse.ArgumentParser()
                    plugin_config.initialize_argparse(plugin_parser)
                    parser._plugin_parsers.append(plugin_parser)

        self.declare("start_date", ConfigValue(
            domain=_StartDate,
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
            domain=Path(),
            default="outdir",
            description="The root directory to which all of the generated simulation files and "
                        "associated data are written.",
        )).declare_as_argument()

        self.declare("data_provider", ConfigValue(
            domain=Module(),
            default=data_provider_factory,
            description="Python module that supplies a data provider implementation"
        )).declare_as_argument()

        #############################
        #  PRESCIENT ONLY OPTIONS   #
        #############################

        # # PRESCIENT_INPUT_OPTIONS

        self.declare("data_path", ConfigValue(
            domain=Path(),
            default="input_data",
            description="Specifies the file or directory to pull data from",
        )).declare_as_argument('--data-path', '--data-directory')

        self.declare("input_format", ConfigValue(
            domain=_InEnumStr(InputFormats),
            default="dat",
            description="Indicate the format input data is in",
        )).declare_as_argument()

        self.declare("simulator_plugin", ConfigValue(
            domain=Path(),
            default=None,
            description="If the user has an alternative methods for the various simulator functions,"
                        " they should be specified here, e.g., my_special_plugin.py.",
        )).declare_as_argument()

        self.declare("deterministic_ruc_solver_plugin", ConfigValue(
            domain=Path(),
            default=None,
            description="If the user has an alternative method to solve the deterministic RUCs,"
                        " it should be specified here, e.g., my_special_plugin.py."
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

        self.declare("ruc_network_type", ConfigValue(
            domain=_InEnumStr(NetworkType),
            default="ptdf",
            description="Specifies the type of network representation to use in RUC processes. Choices are "
                        "ptdf   -- power transfer distribution factor representation."
                        "btheta -- b-theta representation."
                        "Default is ptdf.",
        )).declare_as_argument()

        self.declare("ruc_slack_type", ConfigValue(
            domain=_InEnumStr(SlackType),
            default="every-bus",
            description="Specifies the type of slack variables to use in RUC processes. Choices are "
                        "every-bus            -- slack variables at every system bus."
                        "ref-bus-and-branches -- slack variables at only reference bus and each system branch."
                        "Default is every-bus.",
        )).declare_as_argument()

        self.declare("ruc_horizon", ConfigValue(
            domain=PositiveInt,
            default=48,
            description="The number of hours for which the reliability unit commitment is executed. "
                        "Must be <= 48 hours and >= ruc_every_hours + sced_horizon - 1. "
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

        self.declare("sced_network_type", ConfigValue(
            domain=_InEnumStr(NetworkType),
            default="ptdf",
            description="Specifies the type of network representation to use in SCED processes. Choices are "
                        "ptdf   -- power transfer distribution factor representation."
                        "btheta -- b-theta representation."
                        "Default is ptdf.",
        )).declare_as_argument()

        self.declare("sced_slack_type", ConfigValue(
            domain=_InEnumStr(SlackType),
            default="every-bus",
            description="Specifies the type of slack variables to use in SCED processes. Choices are "
                        "every-bus            -- slack variables at every system bus."
                        "ref-bus-and-branches -- slack variables at only reference bus and each system branch."
                        "Default is every-bus.",
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
            description="Maximum possible value the price can take due to load mismatch"
                        "If the price exceeds this value due to Load Mismatch, then "
                        "it is set to this value.",
        )).declare_as_argument()

        self.declare("transmission_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the price can take due to transmission limits."
                        "If the price exceeds this value due to transmission violations, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("contingency_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the price can take to contingency limits."
                        "If the price exceeds this value due to transmission contingency violations, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("interface_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the price can take for interface violations."
                        "If the price exceeds this value due to interface flow violations, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("reserve_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to reserve violations."
                        "If the reserve price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("regulation_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to regulation reserve violations."
                        "If the regulation price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("spinning_reserve_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to spinning reserve violations."
                        "If the spinning reserve price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("non_spinning_reserve_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to non-spinning reserve violations."
                        "If the non-spinning price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("supplemental_reserve_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to supplemental reserve violations."
                        "If the supplemental price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("flex_ramp_price_threshold", ConfigValue(
            domain=PositiveFloat,
            default=None,
            description="Maximum possible value the reserve price can take due to flexible ramping reserve violations."
                        "If the flexi-ramp price exceeds this value, then "
                        "it is set to this value. The default is based on the value of price_threshold.",
        )).declare_as_argument()

        self.declare("monitor_all_contingencies", ConfigValue(
            domain=bool,
            default=False,
            description="If enabled, will monitor and enforce non-disconnecting "
                        "transmission contingencies. Can be computationally expensive "
                        "for even medium-sized networks.",
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

        self.declare("sced_solver_options", ConfigValue(
            domain=_SolverOptions,
            default=None,
            description="Solver options applied to all SCED solves",
        )).declare_as_argument()

        self.declare("deterministic_ruc_solver_options", ConfigValue(
            domain=_SolverOptions,
            default=None,
            description="Solver options applied to all deterministic RUC solves",
        )).declare_as_argument()

        self.declare("write_deterministic_ruc_instances", ConfigValue(
            domain=bool,
            default=False,
            description="Write all individual RUC instances.",
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
            domain=_InEnumStr(PricingType),
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

        self.declare("output_sced_loads", ConfigValue(
            domain=bool,
            default=False,
            description="Output sced loads prior to each solve. Default is False.",
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

        self.declare("plugin", ConfigDict(
            implicit=True,
            implicit_domain=DynamicImplicitDomain(register_plugin),
            description="Settings for python modules that extend prescient behavior",
        )).declare_as_argument(metavar="<alias>:<plugin_module>",
                               help="A python module that extends prescient behavior,"
                                    " with an alias that the plugin will be known by"
                                    " in this Prescient session. Any settings stored by"
                                    " this plugin will be available in config.plugin.<alias>.",
                               action=_PluginParserAction)



    def __setattr__(self, name, value):
        if name in PrescientConfig.__slots__:
            super(ConfigDict, self).__setattr__(name, value)
        else:
            ConfigDict.__setattr__(self, name, value)

    def initialize_argparse(self, parser):
        super().initialize_argparse(parser)

        # Add --config-file support
        from prescient.scripts import runner
        class _ConfigFileParserAction(argparse.Action):
            def __call__(self, parser, namespace, value, opt_str=None):
                # Get the path from the value
                domain=Path()
                path = domain(value)
                # Get config file contents
                script, config_options = runner.parse_commands(path)
                if script and (script != 'simulator.py'):
                    raise RuntimeError(f"--config-file must be a simulator configuration text file, got {script}")
                # parse the options we just read
                _, rest = parser.parse_known_args(config_options, namespace)
                if rest:
                    raise RuntimeError(f"Unrecognized argument: {rest[0]}")

        parser.add_argument("--config-file", metavar="<filename>",
                            help="A file holding configuration options.",
                            action=_ConfigFileParserAction,
                            default=argparse.SUPPRESS)

        # Add support for plugin parsers
        parser._plugin_parsers = []

        def new_parse_known_args(args=None, namespace=None):
            ns, rest = parser._inner_parse_known_args(args, namespace)
            for p in parser._plugin_parsers:
                ns, rest = p.parse_known_args(rest, ns)
            return ns, rest

        parser._inner_parse_known_args = parser.parse_known_args
        parser.parse_known_args = new_parse_known_args


    def parse_args(self, args: List[str]) -> ConfigDict:
        parser = ArgumentParser()
        self.initialize_argparse(parser)
        args = parser.parse_args(args=args)
        self.import_argparse(args)
        return self



class _InEnumStr(InEnum):
    ''' A bit more forgiving string to enum parser
    '''
    def __call__(self, value):
        if isinstance(value, str):
            value = value.replace('-', '_').upper()
        return super().__call__(value)

def _StartDate(data):
    ''' A basic start date validator/converter
    '''
    if isinstance(data, date):
        return data
    if isinstance(data, str):
        try:
            data = dateutil.parser.parse(data)
        except ValueError:
            print(f"***ERROR: Illegally formatted start date={data} supplied!")
            raise
    if not isinstance(data, datetime):
        raise ValueError("start_date must be a string, datetime.date, or datetime.datetime")
    if data.hour != 0 or data.minute != 0:
        print(f"WARNING: Prescient simulations always begin a midnight; ignoring time {data.time()}")
    return data.date()

def _try_float(v):
    try:
        return float(v)
    except:
        return v

def _SolverOptions(data):
    ''' A basic solver options validator.
        Converts string options into a dictionary;
        otherwise requires a dictionary.
    '''
    if (data is None) or isinstance(data, dict):
        return data

    if isinstance(data, str):
        # idea borrowed from stack overflow:
        # https://stackoverflow.com/questions/38737250/extracting-key-value-pairs-from-string-with-quotes
        s = shlex(data, posix=True)

        # add ',' as whitespace for separation
        # was not supported before, but is easy and useful
        # other whitespace is ' ', '\t', '\r', '\n'
        # Spaces in options need to be escaped
        s.whitespace += ','

        # keep key=value pairs together
        s.wordchars += '='

        # maxsplit keeps = in value together
        # (definitely an edge case and is probably nonsensical)
        data_iter = (w.split('=', maxsplit=1) for w in s)
        return { k : _try_float(v) for k,v in data_iter }

    raise ValueError("Solver options must be a string or dictionary")

if __name__ == '__main__':
    print("config.py cannot run from the command line.")
    sys.exit(1)
