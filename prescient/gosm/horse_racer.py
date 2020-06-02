#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
horse_racer.py
For right now, this script should accept a data source and
output the results of running the Prescient simulator on the scenarios
created based on this data.

Essentially the flow of this program should be
horse_racer
-> populator (creates scenarios via ScenMaker)
-> simulator (tests scenarios)
-> output
"""

# As of right now the options for horse_races seem to be the same
# as the options for the populator

import datetime
import subprocess
import multiprocessing
import sys
import os
from copy import deepcopy

import pandas as pd

from prescient.gosm.utilities import set_arguments

def error_callback(exception):
    print("Process died with exception '{}'".format(exception), file=sys.stderr)

class ConfigurationParser:
    current_line = None
    horses = []

    def __init__(self, filename):
        self.file = open(filename).readlines()
        self.current_index = 0
        self._parse_config_file()
        print("Done parsing configuration file.")

    def get_configurations(self):
        return self.horses

    def _parse_config_file(self):
        """
        This function should read a configuration specification file
        and return a list of Configuration Objects for each of the
        configurations listed in the file
        This will always create a deterministic simulator configuration
        regardless of whether or not one is specified
        """
        print("Parsing configuration file.")
        self._advance_line()
        while self.current_line.startswith('Horse:'):
            self.horses.append(self._parse_horse())

    def _parse_horse(self):
        """
        This should read in from a file pointer a configuration object
        """
        horse_name = self.current_line.split(': ')[1]
        self._advance_line()
        pop_options = self._parse_populator_options()
        sim_options = self._parse_simulator_options()
        return Configuration(horse_name, pop_options, sim_options)

    def _parse_populator_options(self):
        """
        Parses out the populator options that the file currently points at
        """
        if self.current_line != "Populator Options:":
            raise RuntimeError("The configuration for one of the experiments"
                               "includes no populator options"
                               "specifications.")
        self._advance_line()
        options = []
        while self.current_line.startswith('--'):
            options.extend(self.current_line.split())
            self._advance_line()
        return options

    def _parse_simulator_options(self):
        """
        Parses out the simulator options that the file currently points at
        """
        if self.current_line != "Simulator Options:":
            raise RuntimeError("The configuration for one of the experiments"
                               "includes no simulator options"
                               "specifications.")
        self._advance_line()
        options = []
        while self.current_line.startswith('--'):
            options.extend(self.current_line.split())
            self._advance_line()
        return options

    def _advance_line(self):
        """
        This should move the file pointer to the next line and clean
        it of all comments and extraneous whitespace
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


class Configuration:
    """
    This class defines a specific experiment configuration for scenario generation
    and simulation.
    It will be constructed from a file which specifies options for the populator
    and the simulator.
    """
    def __init__(self, name, populator_options, simulator_options):
        self.name = name
        self.populator_options = populator_options
        self.simulator_options = simulator_options
        self._copy_directory()
        self._copy_start_date()
        self._compute_number_of_days()
        for i, option in enumerate(simulator_options):
            if option.startswith('--output-directory'):
                break
        else:
            raise RuntimeError("No output directory option passed in, cannot compute configuration.")
        self.output_directory = simulator_options[i+1]

    def _copy_directory(self):
        """
        Copies the output directory from the populator options to pass it
        to the simulator as the data directory.
        """

        # Try to find the index of the simulator option 'data-directory'.
        try:
            index = self.simulator_options.index('--data-directory')
        except ValueError:
            index = None

        # Get the output directory from the populator options.
        data_dir = self.populator_options[self.populator_options.index('--output-directory') + 1]

        # Add the option to the list of options (or overwrite it).
        if index is not None:
            # If the item following the option's name starts with '--', no data directory was set.
            if self.simulator_options[index + 1].startswith('--'):
                self.simulator_options = self.simulator_options[:index + 1] + [data_dir] \
                                         + self.simulator_options[index + 1:]
            else:
                if self.simulator_options[index + 1] != data_dir:
                    print('Warning: The data directory in the simulator options has been overwritten by '
                          'the output directory in the populator options.')
                self.simulator_options[index + 1] = data_dir
        else:
            self.simulator_options.extend(['--data-directory', data_dir])

    def _compute_number_of_days(self):
        """
        Computes the number of days from the populator options to pass it to the simulator.
        """

        # Try to find the index of the option 'num-days'.
        try:
            index = self.simulator_options.index('--num-days')
        except ValueError:
            index = None

        # Translate the dates into datetime objects.
        try:
            start_date = datetime.datetime.strptime(self.populator_options[self.populator_options.index('--start-date')
                                                                           + 1], "%Y-%m-%d")
            end_date = datetime.datetime.strptime(self.populator_options[self.populator_options.index('--end-date')
                                                                         + 1], "%Y-%m-%d")
        except ValueError:
            raise RuntimeError('Start or end date not valid.')

        # Compute the difference.
        num_days = str((end_date - start_date).days + 1)

        # Add the option to the list of options (or overwrite it).
        if index is not None:
            # If the item following the option's name starts with '--', no number of days was set.
            if self.simulator_options[index + 1].startswith('--'):
                self.simulator_options = self.simulator_options[:index + 1] + [num_days] \
                                         + self.simulator_options[index + 1:]
            else:
                if self.simulator_options[index + 1] != num_days:
                    print('Warning: The number of days in the simulator options has been overwritten by '
                          'the number of days computed from start and end date in the populator options.')
                self.simulator_options[index + 1] = num_days
        else:
            self.simulator_options.extend(['--num-days', num_days])

        return

    def _copy_start_date(self):
        """
        Copies the start date from the populator to pass it to the simulator.
        """

        # Try to find the index of the simulator option 'start-date'.
        try:
            index = self.simulator_options.index('--start-date')
        except ValueError:
            index = None

        # Get the start date from the populator options.
        start_date = self.populator_options[self.populator_options.index('--start-date') + 1]

        # Add the option to the list of options (or overwrite it).
        if index is not None:
            # If the item following the option's name starts with '--', no start date was set.
            if self.simulator_options[index + 1].startswith('--'):
                self.simulator_options = self.simulator_options[:index + 1] + [start_date] \
                                         + self.simulator_options[index + 1:]
            else:
                if self.simulator_options[index + 1] != start_date:
                    print('Warning: The start date in the simulator options has been overwritten by '
                          'the start date in the populator options.')
                self.simulator_options[index + 1] = start_date
        else:
            self.simulator_options.extend(['--start-date', start_date])

        return

    def populate(self):
        if '--skip' not in self.populator_options:
            print(self.populator_options)
            # First argument is always the script name
            subprocess.call(['populator.py'] + self.populator_options)
        else:
            print("Skipping populate step.")

    def simulate(self):
        if '--skip' not in self.simulator_options:
            # First argument is always the script name
            subprocess.call(['simulator.py'] + self.simulator_options)
        else:
            print("Skipping simulate step.")

    def execute(self):
        self.populate()
        self.simulate()

    def __str__(self):
        return "\n".join([self.name] + self.populator_options + self.simulator_options)

    def __repr__(self):
        return "<Configuration Object {}>".format(self.name)


def create_deterministic_configuration(config):
    """
    This function will create a deterministic configuration from a given configuration
    Args:
        config: Configuration Object
    """
    config_copy = deepcopy(config)

    for i, option in enumerate(config.simulator_options):
        if option.startswith('--output-directory'):
            break
    else:
        raise RuntimeError("No output directory option passed in, cannot compute configuration.")
    option = config.simulator_options[i+1]
    option += '_deterministic'

    config_copy.simulator_options[i+1] = option
    config_copy.output_directory += '_deterministic'
    config_copy.simulator_options.append('--run-deterministic-ruc')
    config_copy.name = 'deterministic'
    return config_copy


def write_output(configurations, output_filename='results.txt'):
    frames = []
    for config in configurations:
        frames.append(pd.read_csv(config.output_directory + os.sep + 'daily_summary.csv',
                                  index_col=0, parse_dates=True, sep=' , ', engine='python'))

    output_fields = ['TotalCosts', 'LoadShedding', 'OverGeneration']
    with open(output_filename, 'w') as f:
        f.write("Horse, Total Costs, Load Shedding, Over Generation\n")
        for (config, frame) in zip(configurations, frames):
            results = frame.sum()[output_fields]
            f.write('{},{},{},{}\n'.format(config.name, *results))


def main():
    config_file = sys.argv[1]
    configs = ConfigurationParser(config_file).get_configurations()
    pool = multiprocessing.Pool()
    for config in configs:
        pool.apply_async(config.execute, error_callback=error_callback)
    pool.close()
    pool.join()

    # If the name of the output file is not provided, use "results.txt".
    try:
        result_file = sys.argv[2]
    except IndexError:
        result_file = 'output_horse_racer' + os.sep + 'results.csv'
    write_output(configs, result_file)


if __name__ == '__main__':
    main()
