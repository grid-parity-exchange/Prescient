#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
globals.py
This file sets all the options for the computation of derivatives and derivative patterns
"""

from argparse import ArgumentParser

GC = None


def assign_globals_from_parsed_args(args):
    """
    This function sets the global values, such as column headers.
    """
    global GC
    GC = GlobalConfig(args)


class GlobalConfig:
    def __init__(self, args):
        for option in args.__dict__:
            setattr(self, option, getattr(args, option))


def construct_argument_parser():
    """
    This function constructs all of the different arguments, also known as options
    that can be used by derivative_patterns.py.
    """
    parser = ArgumentParser()

    input_options = parser.add_argument_group("Input Options")

    input_options.add_argument("--input-directory",
                               help="The directory in which the input file resides, defaults to current directory",
                               action="store",
                               dest="input_directory",
                               type=str,
                               default=".")

    input_options.add_argument("--input-filename",
                               help="The file for which you wish to compute the derivatives and patterns from"
                                    "This will also be where additional columns will be written to containing"
                                    "the derivatives, first differences, and patterns",
                               action="store",
                               dest="input_filename",
                               type=str)

    input_options.add_argument("--column-name",
                               help="The column of data for which you wish to compute the derivatives and patterns",
                               action="store",
                               dest="column_name",
                               type=str,
                               default="forecasts")

    categorize_options = parser.add_argument_group("Categorize Options")

    categorize_options.add_argument("--derivative-bounds",
                                    help="These two values describe the fraction or derivatives "
                                         "to be considered low or high respectively. Default is '0.3, 0.7' "
                                         "meaning that 30 percent of derivatives will be -1 in terms of shape, "
                                         "40 percent 0 and 30 percent 1.",
                                    action="store",
                                    dest="derivative_bounds",
                                    type=str,
                                    default="0.3,0.7")

    categorize_options.add_argument("--epifit-error-norm",
                                    help="The error norm used in the segmenter / epifit process.",
                                    action="store",
                                    dest="epifit_error_norm",
                                    type=str,
                                    default="L2")

    categorize_options.add_argument("--lower-medium-bound-pattern",
                                    help="The lower bound for values of forecasts to be counted as "
                                         "~medium and considered for creating patterns",
                                    action="store",
                                    dest="lower_medium_bound_pattern",
                                    type=int,
                                    default=0)

    categorize_options.add_argument("--upper-medium-bound-pattern",
                                    help="The upper bound for values of forecasts to be counted as "
                                         "~medium and considered for creating patterns",
                                    action="store",
                                    dest="upper_medium_bound_pattern",
                                    type=int,
                                    default=0)

    categorize_options.add_argument("--seg-N",
                                    help="epi-fit N for segmentation use ",
                                    action="store",
                                    dest="seg_N",
                                    type=int,
                                    default=20)
    categorize_options.add_argument("--seg-s",
                                    help="epi-fit s for segmentation use ",
                                    action="store",
                                    dest="seg_s",
                                    type=float,
                                    default=0.01)
    categorize_options.add_argument("--seg-kappa",
                                    help="epi-fit kappa for segmentation use ",
                                    action="store",
                                    dest="seg_kappa",
                                    type=float,
                                    default=100.0)

    categorize_options.add_argument("--L1Linf-solver",
                                    help="Solver name for L1 and L-infinity norms (e.g., cplex",
                                    action="store",
                                    dest="L1Linf_solver",
                                    type=str,
                                    default="gurobi")

    categorize_options.add_argument("--L2Norm-solver",
                                    help="Solver name for Norm minimization problems with L2 (e.g. ipopt)",
                                    action="store",
                                    dest="L2Norm_solver",
                                    type=str,
                                    default="gurobi")

    categorize_options.add_argument("--nonlinear-solver",
                                    help="Solver name for nonlinear minimization problems (e.g. ipopt)",
                                    action="store",
                                    dest="nonlinear_solver",
                                    type=str,
                                    default="ipopt")

    categorize_options.add_argument("--non-negativity-constraint-distributions",
                                    help="Binary variable, {0,1}, which is 1 when we constrain the value of "
                                         "the constants s0, v0 to be non negative",
                                    action="store",
                                    dest="non_negativity_constraint_distributions",
                                    type=int,
                                    default=0)

    categorize_options.add_argument("--probability-constraint-of-distributions",
                                    help="Binary variable, {0,1}, which is 1 if we constrain the probability sum to be "
                                         "1 when we obtain the error distribution by an exponential epi spline",
                                    action="store",
                                    dest="probability_constraint_of_distributions",
                                    type=int,
                                    default=1)

    categorize_options.add_argument("--granularity",
                                    help="Granularity for MCL algorithm",
                                    action="store",
                                    dest="granularity",
                                    type=float,
                                    default=5.0)

    return parser
