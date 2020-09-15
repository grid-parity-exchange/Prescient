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
The module name should not be changed, since it is used in other
files to access the options.
"""

import sys
from argparse import ArgumentParser
from copy import deepcopy
import datetime


def set_globals(options_file=None, args=None):
    """
    Parses the options in the file specified in the command-line
    if no options file is passed.

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
                        help='The file containing the filenames for each of '
                             'the data sources',
                        action='store',
                        type=str,
                        dest='sources_file')

    parser.add_argument('--output-directory',
                        help='The directory which will contain the scenario'
                             ' output files',
                        action='store',
                        type=str,
                        dest='output_directory',
                        default='scenario_output')

    parser.add_argument('--scenario-template-file',
                        help='The file which dictates data which should remain the same'
                             'for each of the different scenarios',
                        action='store',
                        type=str,
                        dest='scenario_template_file')

    parser.add_argument('--tree-template-file',
                        help='The file which dictates necessary information '
                             'about the Scenario Tree structure including '
                             'stage variables and stage names. This data will '
                             'be used to construct the ScenarioStructure.dat file',
                        action='store',
                        type=str,
                        dest='tree_template_file')

    parser.add_argument('--hyperrectangles-file',
                        help='The file containing the hyperrectangle patterns.',
                        action='store',
                        type=str,
                        dest='hyperrectangles_file')

    parser.add_argument('--dps-file',
                        help='The file containing the day part separators and '
                             'the skeleton point paths for all sources.',
                        action='store',
                        type=str,
                        dest='dps_file')

    parser.add_argument('--reference-model-file',
                        help='DEPRECATED, this option is ignored',
                        action='store',
                        type=str,
                        default='',
                        dest='reference_model_file')

    # ----------------------------------------------------------
    # General Options
    # ----------------------------------------------------------

    parser.add_argument('--load-scaling-factor',
                        help='Amount load is scaled by in the scenarios',
                        action='store',
                        type=float,
                        dest='load_scaling_factor',
                        default=1)

    parser.add_argument('--scenario-day',
                        help='The day for which the scenarios are supposed to '
                             'be created',
                        action='store',
                        type=valid_date,
                        dest='scenario_day')

    parser.add_argument('--historic-data-start',
                        help='The first day from which the historic data is '
                             'to be considered',
                        action='store',
                        type=valid_date,
                        dest='historic_data_start',
                        default=None)

    parser.add_argument('--historic-data-end',
                        help='The last day up to which the historic data is '
                             'to be considered',
                        action='store',
                        type=valid_date,
                        dest='historic_data_end',
                        default=None)

    parser.add_argument('--use-temporal-copula', '--use-copulas-across-dps',
                        help='Specifies whether to use copulas to compute the '
                             'probabilities across day part separators.',
                        action='store_true',
                        dest='use_temporal_copula')

    parser.add_argument('--sample-skeleton-points',
                        help='Specifies whether the skeleton point values are '
                             'supposed to be randomly sampled or not.',
                        action='store_true',
                        dest='sample_skeleton_points')

    parser.add_argument('--number-scenarios',
                        help='Specifies the number of scenarios to create if '
                             'the sampling method is used.',
                        action='store',
                        type=int,
                        dest='number_scenarios',
                        default=1)

    parser.add_argument('--separate-dps-paths-files',
                        help='If this option is set, then each source must '
                             'specify its own dps-paths file in the sources '
                             'file. Note this only works with the new sources '
                             'file format.',
                        action='store_true',
                        dest='separate_dps_paths_files')

    # ----------------------------------------------------------
    # Options regarding solar sources
    # ----------------------------------------------------------

    parser.add_argument('--solar-frac-nondispatch',
                        help='The fraction of solar generators which are '
                             'nondispatchable. This option only works with '
                             'the old sources file format.',
                        action='store',
                        type=float,
                        dest='solar_frac_nondispatch',
                        default=1)

    parser.add_argument('--power-level-sunrise-sunset',
                        help='The power level (after scaling) that has to be '
                             'exceeded/deceeded in order to indicate '
                             'sunrise/sunset. It should be greater than 0.',
                        action='store',
                        type=float,
                        dest='power_level_sunrise_sunset',
                        default=1)

    parser.add_argument('--dps-sunrise',
                        help='The hour of sunrise. A day part separator is '
                             'created automatically at this hour. '
                             'You can only provide one sunrise hour for '
                             'multiple sources.',
                        action='store',
                        type=int,
                        dest='dps_sunrise',
                        default=None)

    parser.add_argument('--dps-sunset',
                        help='The hour of sunset. A day part separator is '
                             'created automatically at this hour. '
                             'You can only provide one sunset hour for '
                             'multiple sources.',
                        action='store',
                        type=int,
                        dest='dps_sunset',
                        default=None)

    # ----------------------------------------------------------
    # Options regarding wind sources
    # ----------------------------------------------------------

    parser.add_argument('--wind-frac-nondispatch',
                        help='The fraction of wind generators which are '
                             'nondispatchable. This option does nothing if '
                             'the new sources file format is used.',
                        action='store',
                        type=float,
                        dest='wind_frac_nondispatch',
                        default=1)

    # ----------------------------------------------------------
    # Options regarding multiple sources
    # ----------------------------------------------------------

    # These two arguments are designated as store_false so as to maintain
    # the original interface which designated disabling of the crossing
    # scenarios option. It may be better in the future to simply have this
    # be changed to a store_true in order to be more intuitive.
    parser.add_argument('--do-not-cross-scenarios',
                        help='If this option is set, the scenarios are created'
                             'using multivariate distributions and a '
                             'the hyperrectangles file must have '
                             'multidimensional rectangles. If not set '
                             'each source\'s scenarios are created '
                             'independently, and then the cross product of '
                             'each source\'s scenario set is taken as the '
                             'collection of scenarios.',
                        action='store_false',
                        dest='cross_scenarios')

    parser.add_argument('--use-multidim-rectangles',
                        help='If this option is set, then the scenario set '
                             'produced will be done using multidimensional '
                             'rectangles where each dimension of the '
                             'rectangles in the paths file will correspond '
                             'to a certain source.',
                        action='store_false',
                        dest='cross_scenarios')

    parser.add_argument('--use-separate-paths',
                        help='If this option is set, then each source will '
                             'have its own set of paths and day part '
                             'separators. Otherwise, if there are multiple '
                             'sources, then the paths will be drawn from '
                             'the path set with the keyword \'multiple\'.',
                        action='store_true',
                        dest='use_separate_paths')

    parser.add_argument('--use-same-paths-across-correlated-sources',
                        help='If this option is set, then when considering '
                             'scenarios generated from multiple sources, when '
                             'merging the scenarios, we will not consider the '
                             'cartesian product of each source\'s scenarios '
                             'rather we will only consider scenarios where '
                             'each source has the same path.',
                        action='store_true',
                        dest='use_same_paths_across_correlated_sources')

    parser.add_argument('--use-spatial-copula',
                        help='Set this option to be true if you wish to use '
                             'a copula to compute probabilities to account '
                             'for spatial dependencies between sources.',
                        action='store_true',
                        dest='use_spatial_copula')

    parser.add_argument('--spatial-copula', '--copula-across-sources',
                        help='The name of the copula to be used to compute the'
                             ' skeleton point values in the case '
                             'of multiple sources that are not to be crossed.',
                        type=str,
                        dest='spatial_copula')

    parser.add_argument('--partition-file',
                        help='This is the name of the file which contains the '
                             'partition of the sources for grouping dependent '
                             'sources together. This option must be specified '
                             'if using spatial copulas.',
                        type=str,
                        dest='partition_file')

    # ----------------------------------------------------------
    # Options regarding the preprocessor
    # ----------------------------------------------------------

    parser.add_argument('--preprocessor-list',
                        help='The file containing the list of files to feed'
                             ' into the preprocessor',
                        type=str,
                        dest='preprocessor_list')


    parser.add_argument('--solar-power-pos-threshold',
                        help='The solar power data points with a value greater'
                             ' than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='solar_power_pos_threshold',
                        default=None)

    parser.add_argument('--solar-power-neg-threshold',
                        help='The solar power data points with a value less '
                             'than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='solar_power_neg_threshold',
                        default=None)

    parser.add_argument('--wind-power-pos-threshold',
                        help='The wind power data points with a value greater '
                             'than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='wind_power_pos_threshold',
                        default=None)

    parser.add_argument('--wind-power-neg-threshold',
                        help='The wind power data points with a value less '
                             'than this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='wind_power_neg_threshold',
                        default=None)

    parser.add_argument('--load-pos-threshold',
                        help='The load data points with a value greater than '
                             'this threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='load_pos_threshold',
                        default=None)

    parser.add_argument('--load-neg-threshold',
                        help='The load data points with a value less than this'
                             ' threshold will be rectified.',
                        action='store',
                        type=float,
                        dest='load_neg_threshold',
                        default=None)

    # ----------------------------------------------------------
    # Options regarding the univariate epi-spline distribution
    # ----------------------------------------------------------

    parser.add_argument('--seg-N',
                        help='The parameter N of the model for fitting the '
                             'epi-spline distribution.',
                        action='store',
                        type=int,
                        dest='seg_N',
                        default=20)

    parser.add_argument("--epifit-error-norm",
                        help="The error norm used in the segmenter / epifit process.",
                        action="store",
                        dest="epifit_error_norm",
                        type=str,
                        default="L2")

    parser.add_argument('--seg-kappa',
                        help='The parameter kappa of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=float,
                        dest='seg_kappa',
                        default=100)

    parser.add_argument('--probability-constraint-of-distributions',
                        help='A parameter of the model for fitting the epi-spline distribution.',
                        action='store',
                        type=int,
                        dest='probability_constraint_of_distributions',
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

    parser.add_argument("--L1Linf-solver",
                        help="Solver name for L1 and L-infinity norms (e.g., cplex",
                        action="store",
                        dest="L1Linf_solver",
                        type=str,
                        default="gurobi")

    parser.add_argument("--L2Norm-solver",
                        help="Solver name for Norm minimization problems with L2 (e.g. ipopt)",
                        action="store",
                        dest="L2Norm_solver",
                        type=str,
                        default="gurobi")

    parser.add_argument("--error-distribution-domain",
                        help="Set to 'min', if the lowest value should be used as lower limit "
                             "for the computation of error distribution. "
                             "Set to 'max' if the highest value should be the upper limit. "
                             "Set to 'pos', if only positive values occur and the lower limit will be 0. "
                             "Set to 'neg', if only negative values occur and the upper limit will be 0. "
                             "Set to an int or float to specify the multiplier "
                             "for the variation to set the limits. The default is 4. "
                             "Set to 'None', if nothing of the above applies. The default is 'min'. "
                             "If multiple options should be used separate them using ',' "
                             "where later attributes overwrite the former limits. Example: '3,max'",
                        dest="error_distribution_domain",
                        type=str,
                        default='4,min,max')

    # ----------------------------------------------------------
    # Options regarding all distributions
    # ----------------------------------------------------------

    parser.add_argument('--disable-plots',
                        help='If this option is set, no plots will be made.',
                        action='store_true',
                        dest='disable_plots')

    parser.add_argument('--copula-across-dps', '--temporal-copula',
                        help='The name of the copula to be used to compute the scenario probabilities'
                             'across day part separators.',
                        type=str,
                        dest='temporal_copula',
                        default='gaussian-copula')

    parser.add_argument('--copula-prob-sum-tol',
                        help='If the sum of probabilities over all scenarios differs from 1 by more than '
                             'copula_prob_sum_tol, an error is thrown. Otherwise, the probabilities are rescaled.',
                        type=float,
                        dest='copula_prob_sum_tol',
                        default=1e-2)

    parser.add_argument('--plot-variable-gap',
                        help='The gap between two points at which the functions to be plotted are evaluated.',
                        action='store',
                        type=float,
                        dest='plot_variable_gap',
                        default=10)

    parser.add_argument('--plot-pdf',
                        help='If set to 1, all probability distribution functions will be plotted.',
                        action='store',
                        type=int,
                        dest='plot_pdf',
                        default=1)

    parser.add_argument('--plot-cdf',
                        help='If set to 1, all cumulative distribution functions will be plotted.',
                        action='store',
                        type=int,
                        dest='plot_cdf',
                        default=0)

    parser.add_argument('--cdf-inverse-max-refinements',
                        help='The maximum number of refinements (halve the stepsize) for computing the inverse cdf.',
                        action='store',
                        type=int,
                        dest='cdf_inverse_max_refinements',
                        default=10)

    parser.add_argument('--cdf-tolerance',
                        help='The tolerance for computing the inverse cdf.',
                        action='store',
                        type=float,
                        dest='cdf_tolerance',
                        default=1.0e-4)

    parser.add_argument('--cdf-inverse-tolerance',
                        help='The tolerance for computing the inverse cdf.',
                        action='store',
                        type=float,
                        dest='cdf_inverse_tolerance',
                        default=1.0e-4)

    parser.add_argument("--derivative-bounds",
                        help="These two values describe the fraction or derivatives "
                             "to be considered low or high respectively. Default is '0.3, 0.7' "
                             "meaning that 30 percent of derivatives will be -1 in terms of shape, "
                             "40 percent 0 and 30 percent 1.",
                        action="store",
                        dest="derivative_bounds",
                        type=str,
                        default="0.3,0.7")

    parser.add_argument("--monte-carlo-integration",
                        help="Set this option if you wish to use monte carlo "
                             "integration in the computation of the copula "
                             "probability. If this option is set, then the "
                             "option --number-of-samples may be set as well.",
                        action='store_true',
                        dest='monte_carlo_integration')

    parser.add_argument("--number-of-samples",
                        help="If --monte-carlo-integration is set, this option"
                             " determines how many samples will be used for "
                             "monte carlo integration.",
                        type=int,
                        dest='number_of_samples',
                        default=1000000)

    parser.add_argument("--lower-medium-bound-pattern",
                        help="The lower bound for values of forecasts to be counted as "
                             "~medium and considered for creating patterns",
                        action="store",
                        dest="lower_medium_bound_pattern",
                        type=int,
                        default=-1e9)

    parser.add_argument("--upper-medium-bound-pattern",
                        help="The upper bound for values of forecasts to be counted as "
                             "~medium and considered for creating patterns",
                        action="store",
                        dest="upper_medium_bound_pattern",
                        type=int,
                        default=1e9)

    parser.add_argument("--granularity",
                        help="Granularity for MCL algorithm",
                        action="store",
                        dest="granularity",
                        type=float,
                        default=5.0)


    # If no options file is provided, parse the command-line arguments.
    if options_file is None:
        if args is None:
            # This drops the program name from the arguments
            args = sys.argv[1:]
        args = parser.parse_args(args)
    else:
        options = ConfigurationParser(options_file).parse_options()
        args = parser.parse_args(options)

    # Add the configuration options to the module as attributes.
    for arg in vars(args):
        setattr(sys.modules[__name__], arg, getattr(args, arg))

    set_spline_options(args)
    set_distr_options(args)

    # Copy options to distinguish between user options
    # (which are not changed throughout the program)
    # and configuration options (which may be changed during execution).
    user_options = deepcopy(args)

    # Create a new attribute to store the user options.
    setattr(sys.modules[__name__], 'user_options', user_options)

    # Check if the last day up to which the historic data is to be considered
    # lies behind the day (or is the exact
    # day) for which the scenarios are to be created.
    if getattr(sys.modules[__name__], 'historic_data_end') is not None:
        if (getattr(sys.modules[__name__], 'historic_data_end')
            >= getattr(sys.modules[__name__], 'scenario_day')):
            raise RuntimeError('The specified last day of historic data lies '
                               'behind the day (or is the exact day)'
                               'for which the scenarios are to be created.')


def set_spline_options(args):
    """
    This function will store any options related to the spline in a
    module-level dictionary. This will make passing around the options for
    the spline much easier using the **kwargs syntax.

    The dictionary will be accessed with gosm_options.spline_options

    Args:
        args (Namespace): The parsed user arguments
    """
    global spline_options
    spline_options = {}

    spline_options['epifit_error_norm'] = epifit_error_norm
    spline_options['seg_N'] = seg_N
    spline_options['seg_kappa'] = seg_kappa
    spline_options['L1Linf_solver'] = L1Linf_solver
    spline_options['L2Norm_solver'] = L2Norm_solver


def set_distr_options(args):
    """
    This function will store any options related to the epispline-distr in a
    module-level dictionary. This will make passing around the options for
    the spline much easier using the **kwargs syntax.

    The dictionary will be accessed with gosm_options.distr_options

    Args:
        args (Namespace): The parsed user arguments
    """
    global distr_options
    distr_options = {}

    distr_options['error_distribution_domain'] = error_distribution_domain
    distr_options['seg_N'] = seg_N
    distr_options['seg_kappa'] = seg_kappa
    distr_options['nonlinear_solver'] = nonlinear_solver
    distr_options['non_negativity_constraint_distributions'] = \
        non_negativity_constraint_distributions
    distr_options['probability_constraint_of_distributions'] = \
        probability_constraint_of_distributions


def copy_options(options):
    """
    Copies the options from an other module to the current module.

    Args:
        options (Any): the module from where to copy the options

    Notes:
        The module "options" has to have an attribute called "options_list"
        which is a list containing all the names of the options to be copied.
        This is necessary because if we just copied all attributes, even those
        like "__name__" would be copied, which is of course not intended.
    """

    try:
        options_list = getattr(options, 'options_list')
    except AttributeError:
        raise AttributeError("The module '{}' must contain an attribute called"
                             " 'options_list'.".format(options.__name__))
    for option in options_list:
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
        while True:
            if self.current_line.startswith('--'):
                options.extend(self.current_line.split())
            self._advance_line()
            if self.current_line == 'EOF':
                break
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
                self.current_line = 'EOF'
                return
            self.current_line = self.file[self.current_index].strip()
        self._gobble_comments()

    def _gobble_comments(self):
        comment_start = self.current_line.find('#')
        if comment_start != -1:
            self.current_line = self.current_line[:comment_start].strip()
