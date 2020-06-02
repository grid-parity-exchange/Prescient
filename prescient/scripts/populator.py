#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
populator.py

This script acts as a driver for the scenario_creator script. For our purposes,
this is, at its most basic level, simply a looper over the date range passed
in via the options and calls scenario_creator.py with each of the dates.

It is also in this script that we perform any actions related to data which
should be independent of the any individual day. For instance, we compute
the derivatives and patterns for all data beforehand as it would be inefficient
to do this on a daily basis.
"""

import datetime
import multiprocessing
import os
import sys
import shutil
import traceback

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import prescient.scripts.scenario_creator as scenario_creator
import prescient.gosm.populator_options as populator_options
import prescient.gosm.gosm_options as gosm_options
import prescient.gosm.sources as sources


def error_callback(exception):
    print("Process died with exception '{}'".format(exception),
          file=sys.stderr)


def compute_errors(data_sources):
    """
    This will add an errors column to every data source. To do this, it
    computes actuals-forecasts for each source and stores it in the errors
    column.

    Args:
        data_sources (List[Source]): A list of Sources each with an actuals
            and forecasts column
    """
    for source in data_sources:
        source.add_column('errors', source.get_column('actuals')
                          - source.get_column('forecasts'))


def compute_derivatives(data_sources):
    """
    This will compute the derivatives for every source for every column
    which we'll be segmenting by shape for. This will add a derivatives
    column to every Source with a shape criterion.

    Args:
        List[ExtendedSource]: The sources read from the sources file
    """
    for source in data_sources:
        for criterion in source.criteria:
            if criterion.method == 'shape':
                print("Computing Derivatives")
                source.compute_derivatives(
                    criterion.column_name, end_date=populator_options.end_date,
                    **gosm_options.spline_options)


def populate():
    """
    Iterates over all dates and creates scenarios for each of these days.
    """

    # Create output directory (if it already exists, delete it first).
    if os.path.isdir(populator_options.output_directory):
        shutil.rmtree(populator_options.output_directory)
    os.makedirs(populator_options.output_directory)

    print("Reading in data")
    if populator_options.sources_file.endswith('.csv'):
        print("Warning: the csv-file format for sources is deprecated.")
        data_sources = sources.sources_from_sources_file(
            populator_options.sources_file)
        # We need to set frac_nondispatch and scaling factor manually
        for source in data_sources:
            if source.source_type == 'solar':
                source.scaling_factor = gosm_options.solar_scaling_factor
                source.frac_nondispatch = gosm_options.solar_frac_nondispatch
            elif source.source_type == 'wind':
                source.scaling_factor = gosm_options.wind_scaling_factor
                source.frac_nondispatch = gosm_options.wind_frac_nondispatch
            elif source.source_type == 'load':
                source.scaling_factor = gosm_options.load_scaling_factor
    elif populator_options.sources_file.endswith('.txt'):
        data_sources = sources.sources_from_new_sources_file(
            populator_options.sources_file)
    else:
        raise RuntimeError("Source file format unrecognized")

    scale_windows(data_sources)
    # We compute the errors here as they are needed for clustering.
    compute_errors(data_sources)
    compute_derivatives(data_sources)

    date_range = pd.date_range(populator_options.start_date,
                               populator_options.end_date, freq='D')

    # Start the scenario creation in parallel or sequentially.
    if populator_options.allow_multiprocessing:
        pool = multiprocessing.Pool(populator_options.max_number_subprocesses)
        for date in date_range:
            pool.apply_async(create_scenarios_mp,
                             args=(date, data_sources, sys.argv[1:]),
                             error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        for date in date_range:
            try:
                create_scenarios(date, data_sources)
            except Exception as e:
                print("Failed to create scenarios for day {}".format(date))
                print("Error: {}".format(','.join(map(str, e.args))))
                if populator_options.traceback:
                    exc_type, exc, tb = sys.exc_info()
                    print("{}: {}".format(exc_type, exc))
                    traceback.print_tb(tb)


def average_sunrise_sunset(date, window):
    """
    Given a solar source, this will compute the average sunrise and sunset
    hours over the past two weeks.

    Args:
        date (datetime-like): The date we wish to compute sunrise and sunset
        window (RollingWindow): A solar source
    Returns:
        (float, float): The hours of sunrise and sunset
    """
    historic_data = window.data
    end_date = date - datetime.timedelta(hours=1)
    start_date = date - datetime.timedelta(days=14)

    relevant_data = historic_data[start_date:end_date]
    daylight_data = relevant_data[relevant_data['actuals'] > 0]
    daylight_data['hours'] = daylight_data.index.hour

    # These are the first and last hours with postive actuals
    sunrises = daylight_data.groupby(daylight_data.index.date).min()['hours']
    sunsets = daylight_data.groupby(daylight_data.index.date).max()['hours']

    # These will be last and first hours for which there is a zero actual
    average_sunrise = int(max(round(sunrises.mean()) - 1, 0))
    average_sunset = int(min(round(sunsets.mean()) + 1, 23))

    return average_sunrise, average_sunset


def estimate_sunrise_sunset(date, diurnal_pattern):
    """
    This function when given a date and a diurnal_pattern dataframe will
    estimate the sunrise and sunset. It does this by finding the first hour
    which has a positive diurnal pattern and then finds the first hour after
    daylight which has 0 diurnal pattern.

    Args:
        date (datetime-like): The datetime at hour 0
        diurnal_pattern (pd.Series): A datetime-indexed series of diurnal
            patterns
    Returns:
        (float, float): The hours of sunrise and sunset
    """

    daily_pattern = diurnal_pattern[date:date+datetime.timedelta(hours=23)]

    sunrise, sunset = None, None

    # This will walk through finding first sunlight hour and first night hour
    for hour, pattern in enumerate(daily_pattern.values):
        if sunrise is None and pattern > 0:
            sunrise = hour

        # If sun has risen, and we have not found night and we reach a 0
        if sunrise is not None and sunset is None and pattern == 0:
            sunset = hour

    return sunrise, sunset


def check_if_empty(data_sources, day):
    """
    This will check if any day-of data is missing in the frame. We need all
    24 hours of a day's data to exist to generate scenarios.
    It raises an error if any hour of the day is missing in any frame

    Args:
        data_sources (List[Source]): The list of data sources to check
    """
    day = pd.Timestamp(day)
    for source in data_sources:
        day_of_data = source.data[day:day+datetime.timedelta(hours=23)]
        if len(day_of_data.index) < 24:
            raise RuntimeError("There is not enough data for {} to "
                               "generate scenarios with in source {}"
                               .format(day.date(), source.name))


def interpolate_windows(data_sources):
    """
    Function which interpolates the forecasts and actuals for each window
    in the list of windows.

    Args:
        data_sources (List[Source]): The list of data sources to interpolate
    """
    for source in data_sources:
        source.interpolate('forecasts')
        source.interpolate('actuals')


def scale_windows(data_sources):
    """
    This function will scale the source by the appropriate factor
    depending on the type. This modifies all the windows in-place. This will
    scale the forecasts and actuals column.

    Args:
        data_sources (List[Source]): The sources to scale
    """
    for data_source in data_sources:
        factor = data_source.scaling_factor

        data_source.scale('forecasts', factor)
        data_source.scale('actuals', factor)


def prepare_solar_dps(dt, source):
    """
    This function will prepare the day part separators for solar sources.
    To handle this, it will either estimate it from a diurnal pattern if
    provided or it will use the average hour for which there is first power
    generated over the past two weeks and similarly for the last hour.

    Args:
        dt (datetime-like): The date to prepare the day part separators
            for
        source (Source): The source of data
    """

    if populator_options.average_sunrise_sunset:
        sunrise, sunset = average_sunrise_sunset(dt, source)
    else:
        diurnal_pattern = pd.read_csv(populator_options.diurnal_pattern_file,
                                      index_col=0, parse_dates=True)
        sunrise, sunset = estimate_sunrise_sunset(
            dt, diurnal_pattern['diurnal pattern'])
    gosm_options.dps_sunrise = sunrise
    gosm_options.dps_sunset = sunset

    if populator_options.number_dps:
        temp_file = "{}{}temp_paths_{}.dat".format(
            populator_options.output_directory, os.sep,
            gosm_options.scenario_day.date())

        # If this option is specified, this will dynamically generate the
        # specified amount of day part separators from sunrise to sunset
        dps = list(map(round, np.linspace(sunrise, sunset,
                                          populator_options.number_dps)))

        dps = list(map(int, dps))

        dps[0] = 'sunrise'
        dps[-1] = 'sunset'

        paths_file = populator_options.dps_paths_file
        # The newly generated file will be the same as the specified paths file
        # except with the newly generated day part separators
        with open(paths_file) as reader, open(temp_file, 'w') as writer:
            for line in reader:
                if line.startswith('dps'):
                    writer.write('dps ' + ' '.join(map(str, dps)) + '\n')
                else:
                    writer.write(line)

        gosm_options.dps_file = temp_file


def cluster_sources(data_sources):
    """
    This will call the cluster method for every source which has a shape
    criterion. In order to cluster, the source must have a
    and a <field>_derivatives column. This will add a <field>_derivatives_
    patterns column with the patterns and a
    <field>_derivatives_patterns_clusters column composed of the clusters.

    Args:
        data_sources (List[Source]): The list of sources which may or may
            not be clustered
    """


    # This will cluster for every source which has a shape criterion
    for source in data_sources:

        # This will be the first time we use errors, so we need to compute
        # them now.
        errors = source.get_column('actuals') - source.get_column('forecasts')
        # We subtract one hour to not include midnight of the scenario day
        end_date = gosm_options.scenario_day-datetime.timedelta(hours=1)
        relevant_errors = errors[:end_date]
        source.data['errors'] = relevant_errors

        for criterion in source.criteria:
            if criterion.method == 'shape':
                # bound_string is of the form '<float>,<float>'.
                bound_string = gosm_options.derivative_bounds
                bounds = list(map(float, bound_string.split(',')))

                # Now we want to compute patterns and clusters for all hours
                # including the scenario day
                end_date = gosm_options.scenario_day \
                           + datetime.timedelta(hours=23)

                source.compute_patterns(criterion.column_name + '_derivatives',
                                        bounds, end_date=end_date)

                scenario_creator.print_progress("Clustering Patterns")

                # The name of the column will have _derivatives_patterns
                # appended to it as that is how the Source methods work
                source.cluster(criterion.column_name + '_derivatives_patterns',
                               gosm_options.granularity,
                               **gosm_options.distr_options, end_date=end_date)


def create_scenarios_mp(date, data_sources, pop_options):
    """
    This is a wrapper function to enable multiprocessing. It will
    set the populator optionas and scenario creator options for each
    subprocess (since this is not done automatically on windows).

    Args:
        date (datetime.datetime): the date
        data_sources (List[ExtendedSource]): The list of sources (including
            a load source and at least one power source) with segmentation
            criteria (and possibly upper bounds)
        pop_options: User specified populator options passed in through the
            arguments
    """
    populator_options.set_globals(pop_options)
    gosm_options.set_globals(populator_options.scenario_creator_options_file)
    create_scenarios(date, data_sources)


def create_scenarios(date, data_sources):
    """
    Reads the data and calls the scenario creator for a specific date.

    Args:
        date (datetime.datetime): the date
        data_sources (List[ExtendedSource]): The list of sources (including
            a load source and at least one power source) with segmentation
            criteria (and possibly upper bounds)
    """
    # Modify GOSM options for the current day.
    dt = gosm_options.scenario_day = pd.Timestamp(date)
    gosm_options.historic_data_end = date - datetime.timedelta(hours=1)

    if not os.path.isdir(populator_options.output_directory):
        os.mkdir(populator_options.output_directory)

    # this puts it in pyspdir_twostage because the simulator expects
    # a directory with that name
    gosm_options.output_directory = populator_options.output_directory + os.sep + \
        'pyspdir_twostage' + os.sep + str(gosm_options.scenario_day.date())

    # TODO: Restore ability to specify range of historic data
   # if gosm_options.historic_data_start is not None:
   #     historic_start = pd.Timestamp(gosm_options.historic_data_start)
   # else:
   #     historic_start = None

   # if gosm_options.historic_data_end is not None:
   #     historic_end = pd.Timestamp(gosm_options.historic_data_end)
   # else:
   #     historic_end = None

    check_if_empty(data_sources, date)
    if populator_options.interpolate_data:
        interpolate_windows(data_sources)

    # This function only does something if any source has a shape criterion
    cluster_sources(data_sources)

    scenario_creator.create_scenarios(data_sources, date)


def main(populator_args=None, scenario_creator_args=None):
    """
    This is the main execution of the populator script. By default it will pull
    options from the command line; however you can pass in your own options
    as a list of arguments.

    Args:
        populator_args (List[str]): A list of arguments as are passed from the
            command line
        scenario_creator_args (List[str]): A list of arguments that would
            be passed to the scenario creator
    """

    populator_options.set_globals(populator_args)

    # Set the options for the scenario creator.
    if scenario_creator_args is None:
        gosm_options.set_globals(
            options_file=populator_options.scenario_creator_options_file)
    else:
        gosm_options.set_globals(args=scenario_creator_args)

    # Copy the populator options to the module gosm_options
    # (overwrite options that already exist).
    gosm_options.copy_options(populator_options)

    populate()

if __name__ == '__main__':
    main()
