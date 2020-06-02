#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This class stores the options for GOSM.
The module name should not be changed, since it is used in other files to access the options.
"""

import sys
from argparse import ArgumentParser
from copy import deepcopy
import datetime


def set_globals(options_file=None):
    """
    Parses the options in the file specified in the command-line if no options file is passed.

    Args:
        options_file (str): an optional file name of an options file
    """

    def valid_date(s):
        """
        Method to check if the string passed as a day is an actual date.
        
        Args:
            s (str): the string to check 
        """
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            print("Error: Not a valid date: '{0}'.".format(s))

        return

    parser = ArgumentParser()

    # ----------------------------------------------------------
    # Options regarding file in- and output
    # ----------------------------------------------------------

    parser.add_argument('--sources-file',
                        help='The file containing the filenames for each of the data sources',
                        action='store',
                        type=str,
                        dest='sources_file')

    parser.add_argument('--output-directory',
                        help='The directory which will contain the scenario output files',
                        action='store',
                        type=str,
                        dest='output_directory',
                        default='scenario_output')

    parser.add_argument('--scenario-template-file',
                        help='The file which dictates data which should remain the same'
                             'for each of the different scenarios',
                        action='store',
                        type=str,
                        dest='scenario_template_file',
                        default='ScenTemplate.dat')

    parser.add_argument('--tree-template-file',
                        help='The file which dictates necessary information about the Scenario Tree'
                             'structure including stage variables and stage names. This data'
                             'will be used to construct the ScenarioStructure.dat file',
                        action='store',
                        type=str,
                        dest='tree_template_file',
                        default='TreeTemplate.dat')

    parser.add_argument('--hyperrectangles-file',
                        help='The file containing the hyperrectangle patterns.',
                        action='store',
                        type=str,
                        dest='hyperrectangles_file')

    parser.add_argument('--dps-file',
                        help='The file containing the day part separators and the skeleton point paths '
                             'for all sources.',
                        action='store',
                        type=str,
                        dest='dps_file')

    parser.add_argument('--daps-location',
                        help='The directory of daps (has to contain the file basicclasses.py).',
                        action='store',
                        type=str,
                        dest='daps_location')

    # ----------------------------------------------------------
    # General Options
    # ----------------------------------------------------------

    parser.add_argument('--load-scaling-factor',
                        help='Amount load is scaled by in the scenarios',
                        action='store',
                        type=float,
                        dest='load_scaling_factor',
                        default=0.045)

    parser.add_argument('--scenario-day',
                        help='The day for which the scenarios are supposed to be created',
                        action='store',
                        type=valid_date,
                        dest='scenario_day')

    parser.add_argument('--historic-data-start',
                        help='The first day from which the historic data is to be considered',
                        action='store',
                        type=valid_date,
                        dest='historic_data_start',
                        default=None)

    parser.add_argument('--historic-data-end',
                        help='The last day up to which the historic data is to be considered',
                        action='store',
                        type=valid_date,
                        dest='historic_data_end',
                        default=None)

    parser.add_argument('--copulas-across-dps',
                        help='Specifies whether to use copulas to compute the probabilities across day part '
                             'separators.',
                        action='store',
                        type=int,
                        dest='copulas_across_dps',
                        default=1)

    # ----------------------------------------------------------
    # Options regarding solar sources
    # ----------------------------------------------------------

    parser.add_argument('--solar-frac-nondispatch',
                        help='The fraction of solar generators which are nondispatchable',
                        action='store',
                        type=float,
                        dest='solar_frac_nondispatch',
                        default=0.5)

    parser.add_argument('--solar-scaling-factor',
                        help='Amount to scale solar by so the problem makes sense',
                        action='store',
                        type=float,
                        dest='solar_scaling_factor',
                        default=0.25)

    parser.add_argument('--power-level-sunrise-sunset',
                        help='The power level (after scaling) that has to be exceeded/deceeded in order '
                             'to indicate sunrise/sunset. It should be greater than 0.',
                        action='store',
                        type=float,
                        dest='power_level_sunrise_sunset',
                        default=1)

    parser.add_argument('--dps-sunrise',
                        help='The hour of sunrise. A day part separator ist created automatically at this hour. '
                             'You can only provide one sunrise hour for multiple sources.',
                        action='store',
                        type=int,
                        dest='dps_sunrise',
                        default=None)

    parser.add_argument('--dps-sunset',
                        help='The hour of sunset. A day part separator ist created automatically at this hour. '
                             'You can only provide one sunset hour for multiple sources.',
                        action='store',
                        type=int,
                        dest='dps_sunset',
                        default=None)

    # ----------------------------------------------------------
    # Options regarding wind sources
    # ----------------------------------------------------------

    parser.add_argument('--wind-frac-nondispatch',
                        help='The fraction of wind generators which are nondispatchable',
                        action='store',
                        type=float,
                        dest='wind_frac_nondispatch',
                        default=1)

    parser.add_argument('--wind-scaling-factor',
                        help='Amount to scale wind by so the problem makes sense',
                        action='store',
                        type=float,
                        dest='wind_scaling_factor',
                        default=1)

    # ----------------------------------------------------------
    # Options regarding multiple sources
    # ----------------------------------------------------------

    parser.add_argument('--cross-scenarios',
                        help='If set to 1, the scenarios are created for each source seperately and crossed '
                             'afterwards. If set to 0, scenarios are created from multivariate distributions.',
                        action='store',
                        type=int,
                        dest='cross_scenarios',
                        default=1)

    parser.add_argument('--copula-across-sources',
                        help='The name of the copula to be used to compute the skeleton point values in the case '
                             'of multiple sources that are not to be crossed.',
                        type=str,
                        dest='copula_across_sources')

    # ----------------------------------------------------------
    # Options regarding the preprocessor
    # ----------------------------------------------------------

    parser.add_argument('--solar-power-pos-threshold',
                        help='The solar power data points with a value greater than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='solar_power_pos_threshold',
                        default=None)

    parser.add_argument('--solar-power-neg-threshold',
                        help='The solar power data points with a value less than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='solar_power_neg_threshold',
                        default=None)

    parser.add_argument('--wind-power-pos-threshold',
                        help='The wind power data points with a value greater than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='wind_power_pos_threshold',
                        default=None)

    parser.add_argument('--wind-power-neg-threshold',
                        help='The wind power data points with a value less than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='wind_power_neg_threshold',
                        default=None)

    parser.add_argument('--load-pos-threshold',
                        help='The load data points with a value greater than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='load_pos_threshold',
                        default=None)

    parser.add_argument('--load-neg-threshold',
                        help='The load data points with a value less than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='load_neg_threshold',
                        default=None)

    # ----------------------------------------------------------
    # Options regarding the univariate epi-spline distribution
    # ----------------------------------------------------------

    parser.add_argument('--seg-N',
                        help='The parameter N of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=int,
                        dest='seg_N')

    parser.add_argument('--seg-kappa',
                        help='The parameter kappa of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=float,
                        dest='seg_kappa')

    parser.add_argument('--probability-constraint-distributions',
                        help='A parameter of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=int,
                        dest='probability_constraint_distributions',
                        default=1)

    parser.add_argument('--non-negativity-constraint-distributions',
                        help='A parameter of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=int,
                        dest='non_negativity_constraint_distributions',
                        default=0)

    parser.add_argument('--nonlinear-solver',
                        help='The nonlinear solver of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=str,
                        dest='nonlinear_solver',
                        default='ipopt')

    # ----------------------------------------------------------
    # Options regarding all distributions
    # ----------------------------------------------------------

    parser.add_argument('--copula-across-dps',
                        help='The name of the copula to be used to compute the scenario probabilities'
                             'across day part separators.',
                        type=str,
                        dest='copula_across_dps')

    parser.add_argument('--copula-prob-sum-tol',
                        help='If the sum of probabilities over all scenarios differs from 1 by more than '
                             'copula_prob_sum_tol, an error is thrown. Otherwise, the probabilities are rescaled.',
                        type=float,
                        dest='copula_prob_sum_tol',
                        default='1e-2')

    parser.add_argument('--plot-variable-gap',
                        help='The gap between two points at which the functions to be plotted are evaluated.',
                        action='store',
                        type=float,
                        dest='plot_variable_gap',
                        default='10')

    parser.add_argument('--plot-pdf',
                        help='If set to 1, all probability distribution functions will be plotted.',
                        action='store',
                        type=int,
                        dest='plot_pdf',
                        default='0')

    parser.add_argument('--plot-cdf',
                        help='If set to 1, all cumulative distribution functions will be plotted.',
                        action='store',
                        type=int,
                        dest='plot_cdf',
                        default='0')

    parser.add_argument('--cdf-inverse-max-refinements',
                        help='The maximum number of refinements (halve the stepsize) for computing the inverse cdf.',
                        action='store',
                        type=int,
                        dest='cdf_inverse_max_refinements',
                        default='10')

    parser.add_argument('--cdf-inverse-tolerance',
                        help='The tolerance for computing the inverse cdf.',
                        action='store',
                        type=float,
                        dest='cdf_inverse_tolerance',
                        default='1.0e-4')

    # If no options file is provided, parse the command-line arguments.
    if options_file is None:
        args = parser.parse_args()
    else:
        options = ConfigurationParser(options_file).parse_options()
        args = parser.parse_args(options)

    # Add the configuration options to the module as attributes.
    for arg in args.__dict__:
        setattr(sys.modules[__name__], arg, getattr(args, arg))

    # Copy options to distinguish between user options (which are not changed throughout the program)
    # and configuration options (which may be changed during execution).
    user_options = deepcopy(args)

    # Create a new attribute to store the user options.
    setattr(sys.modules[__name__], 'user_options', user_options)

    # Check if the last day up to which the historic data is to be considered lies behind the day (or is the exact
    # day) for which the scenarios are to be created.
    if getattr(sys.modules[__name__], 'historic_data_end') is not None:
        if getattr(sys.modules[__name__], 'historic_data_end') >= getattr(sys.modules[__name__], 'scenario_day'):
            raise RuntimeError('The specified last day of historic data lies behind the day (or is the exact day)'
                               'for which the scenarios are to be created.')


def copy_options(options):
    for option in options.__dict__:
        setattr(sys.modules[__name__], option, getattr(options, option))


class ConfigurationParser:
    """
    Copied from horse_racer.py and modified on 4/6/2017.
    """

    current_line = None

    def __init__(self, filename):
        self.file = open(filename).readlines()
        self.current_index = 0

    def parse_options(self):
        self._advance_line()
        options = []
        while self.current_line.startswith('--'):
            options.extend(self.current_line.split())
            self._advance_line()
        return options

    def _advance_line(self):
        """
        This should move the file pointer to the next line and clean
        it of all comments and extraneous whitespace.
        """
        self.current_index += 1
        if self.current_index >= len(self.file):
            self.current_line = 'EOF'
            return
        self.current_line = self.file[self.current_index].strip()
        while self.current_line.startswith('#') or self.current_line == '':
            self.current_index += 1
            if self.current_index >= len(self.file):
                self.current_line = None
                break
            self.current_line = self.file[self.current_index].strip()
        self._gobble_comments()

    def _gobble_comments(self):
        comment_start = self.current_line.find('#')
        if comment_start != -1:
            self.current_line = self.current_line[:comment_start].strip()
