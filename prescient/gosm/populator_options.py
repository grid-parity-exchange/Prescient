#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
populator_options.py

This module will house any and all functions related to setting options for the
populator.
"""

import sys
from argparse import ArgumentParser
import multiprocessing


def set_globals(args=None):
    """
    Parses the options in the file specified in the command-line.

    Args:
        args(List[str]): The command line arguments as a list of strings. By
            default these will just be pulled from sys.argv which will be the
            command line arguments. However, you can pass in your own list of
            strings to change the behavior.
    """
    if args is None:
        # This is to drop the program name from the arguments
        args = sys.argv[1:]

    def set_max_number_subprocesses(i):
        """
        Method to determine the maximum number of processes.

        Args:
            i (str): a string that should contain the maximum number of processes

        Returns:
            int: the maximum number of processes
        """

        if i is not None:
            try:
                return int(i)
            except ValueError:
                print('Error: The specified maximum number of subprocesses must be an integer value.')
                print('It has been set to {}.'.format(multiprocessing.cpu_count()))
                return multiprocessing.cpu_count()
        else:
            return multiprocessing.cpu_count()

    parser = ArgumentParser()

    parser.add_argument('--scenario-creator-options-file',
                        help="The file where the options for the scenario creator are stored",
                        action='store',
                        type=str,
                        dest='scenario_creator_options_file')

    parser.add_argument('--start-date',
                        help="The date to start the construction of scenarios at"
                             "Should be written in YYYY-MM-DD",
                        action='store',
                        dest='start_date')

    parser.add_argument('--end-date',
                        help="The date to end the construction of scenarios at"
                             "Should be written in YYYY-MM-DD",
                        action='store',
                        dest='end_date')

    parser.add_argument('--sources-file',
                        help='The file containing the filenames for each of the data sources',
                        action='store',
                        type=str,
                        dest='sources_file')

    parser.add_argument('--solar-scaling-factor',
                        help='Amount to scale solar by so the problem makes sense',
                        action='store',
                        type=float,
                        dest='solar_scaling_factor',
                        default=1)

    parser.add_argument('--wind-scaling-factor',
                        help='Amount to scale wind by so the problem makes sense',
                        action='store',
                        type=float,
                        dest='wind_scaling_factor',
                        default=1)

    parser.add_argument('--load-scaling-factor',
                        help='Amount load is scaled by in the scenarios',
                        action='store',
                        type=float,
                        dest='load_scaling_factor',
                        default=0.045)

    parser.add_argument('--output-directory',
                        help='The directory which will contain the scenario output files',
                        action='store',
                        type=str,
                        dest='output_directory',
                        default='scenario_output')

    parser.add_argument('--allow-multiprocessing',
                        help='Specifies whether multiprocessing is allowed or not.',
                        action='store',
                        type=int,
                        dest='allow_multiprocessing',
                        default=0)

    parser.add_argument('--max-number-subprocesses',
                        help='Specifies the maximum number of created '
                             'subprocesses. If the option is not set,'
                             'the number of available cores will be used '
                             '(if multiprocessing is enabled at all).',
                        action='store',
                        type=set_max_number_subprocesses,
                        dest='max_number_subprocesses',
                        default=None)

    parser.add_argument('--traceback',
                        help='Set this option if you wish to print the '
                             'traceback in the event of an error',
                        action='store_true',
                        dest='traceback')

    parser.add_argument('--diurnal-pattern-file',
                        help='This file will contain the daypattern for the'
                             ' purposes of determining sunrise and sunset.'
                             ' This option should only be set in the case of'
                             ' solar data.',
                        dest='diurnal_pattern_file')

    parser.add_argument('--number-dps',
                        help='This option dictates the number of day part '
                             'separators to use for solar sources. This '
                             'will dynamically modify the paths file '
                             'specified in dps_paths_file to have equally '
                             'spaced day part separators.',
                        dest='number_dps',
                        type=int)

    parser.add_argument('--dps-paths-file',
                        help='If --number-dps is specified, then this will be '
                             'the file to read in and change the day part '
                             'separators for. The program will write to a '
                             'temporary file which will be deleted on '
                             'completion of the program.',
                        dest='dps_paths_file')

    parser.add_argument('--interpolate-data',
                        help='If set, this will interpolate the data if any'
                             ' hours are missing.',
                        action='store_true',
                        dest='interpolate_data')

    parser.add_argument('--average-sunrise-sunset',
                        help='If set, this will use the average hours of '
                             'sunrise and sunset to estimate the sunrise and '
                             'sunset for the next day of scenario generation.',
                        action='store_true',
                        dest='average_sunrise_sunset')

    args = parser.parse_args(args)
    for arg in vars(args):
        setattr(sys.modules[__name__], arg, getattr(args, arg))

    # We add a list of the options' names in order to be able to copy only exactly these options
    # to an other module (look at gosm_options.copy_options for example).
    setattr(sys.modules[__name__], 'options_list', list(args.__dict__.keys()))
