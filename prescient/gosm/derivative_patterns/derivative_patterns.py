#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
derivative_patterns.py
  This module will contain code which will be used to segment historic data based on derivative patterns.

  Usage: python runner.py derivative_patterns_exec.txt
"""

import os
import sys
import math
import datetime
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyomo.environ import *

from . import globals
from . import epimodel
from .graph_utilities import WeightedGraph
from prescient.gosm.splines import Spline
from prescient.util.distributions.distributions import UnivariateEpiSplineDistribution

def create_pattern(dt, historic_df, column_name, bounds):
    """
    When given a datetime dt, a dataframe historic_df, a column name,
    and a tuple of numbers, This should establish what pattern corresponds to
    the value in the dataframe at location dt, column_name.

    It returns an ordered triple corresponding to the what the values are at
    times dt-1hour, dt, and dt+1hour. If the value is less than the lower
    bound, the corresponding index is -1, if between bounds, the pattern is 0
    at that index, if it is larger than the upper bound, it is 1.

    If the previous time or next time are not in the dataframe,
    then it just uses the value at the current datetime
    """
    lower_time = dt - datetime.timedelta(hours=1)
    if (lower_time not in historic_df.index
        or np.isnan(historic_df[column_name][lower_time])):

        lower_time = dt

    upper_time = dt + datetime.timedelta(hours=1)
    if (upper_time not in historic_df.index
        or np.isnan(historic_df[column_name][upper_time])):

        upper_time = dt

    pattern = []

    for time in (lower_time, dt, upper_time):
        if historic_df[column_name][time] < bounds[0]:
            pattern.append(-1)
        elif historic_df[column_name][time] > bounds[1]:
            pattern.append(1)
        else:
            pattern.append(0)

    return tuple(pattern)


def wetsdistance_patterns(distr_1, distr_2):
    """
    Compute the Wets Distance between two distributions encoded in the
    OrderedDictionaries error_forecast_1 and error_forecast_2 mapping
    {x -> w(x)} x is a time, w(x) the forecast at that time.

    Returns a number representing the Wets Distance
    """

    xs = np.arange(0, 1 + 1/distr_1.seg_N, 1/distr_1.seg_N)

    # Compute the Wets distance
    wetdist = 0
    for x1 in xs:
        smallest_point_distance = 100
        if distr_1._normalized_cdf(x1) > distr_2._normalized_cdf(x1):
            for x2 in xs:
                local_point_distance = ((x1 - x2) ** 2
                    + (distr_1._normalized_cdf(x1)
                       - distr_2._normalized_cdf(x2)) ** 2) ** 0.5
                if local_point_distance < smallest_point_distance:
                    smallest_point_distance = local_point_distance
        else:
            for x2 in xs:
                local_point_distance = ((x1 - x2) ** 2
                    + (distr_2._normalized_cdf(x1)
                       - distr_1._normalized_cdf(x2)) ** 2) ** 0.5
                if local_point_distance < smallest_point_distance:
                    smallest_point_distance = local_point_distance
        if smallest_point_distance > wetdist:
            wetdist = smallest_point_distance
    return wetdist


def get_distance_dict(pattern_dict, time_series, **distr_options):
    """
    This function will compute the Wets distance between any two patterns
    listed in the pattern dict. It will return a dictionary mapping
    pairs of patterns to the distance between them
    """
    patterns_enough_data = [pattern for pattern in pattern_dict
                            if len(pattern_dict[pattern]) > 20]

    if not patterns_enough_data:
        print("Not enough datetimes were provided to compute patterns.")
        print("To compute patterns more datetimes must be provided.")
        sys.exit(1)

    # We first compute an epispline distribution for every pattern
    distributions = {}
    for pattern in patterns_enough_data:
        pattern_dates = pattern_dict[pattern]
        pattern_dates = [date for date in pattern_dates if date in time_series.index]
        pattern_values = time_series.loc[pattern_dates].dropna()
        distributions[pattern] = UnivariateEpiSplineDistribution(
            pattern_values.values, **distr_options)

    distance_pattern = {}
    for pattern_1 in patterns_enough_data:
        for pattern_2 in patterns_enough_data:
            if (pattern_1, pattern_2) in distance_pattern:
                continue
            if pattern_1 == pattern_2:
                # The distance from any point to itself is 0
                distance_pattern[(pattern_1, pattern_2)] = 0
            else:
                distance_pattern[(pattern_1, pattern_2)] = wetsdistance_patterns(
                    distributions[pattern_1], distributions[pattern_2])
                # Since metrics are symmetric
                distance_pattern[(pattern_2, pattern_1)] = distance_pattern[(pattern_1, pattern_2)]

    return distance_pattern


def get_clusters(pattern_dict, time_series, granularity=5, **distr_options):
    """
    Uses Markov Clustering Algorithm to cluster patterns together.
    Accepts a dictionary of patterns mapped to datetimes and a Pandas time series mapping
    the datetimes to values. Returns a partition of the patterns which is stored
    as a list of lists of patterns.
    Args:
        pattern_dict: A dictionary mapping patterns to lists of datetimes
        time_series: A Pandas series mapping datetimes to values
        error_distribution_domain: String specifying domain (described in ErrorParametersDict)
        seg_N: The number of segments desired when fitting the spline
        seg_kappa: Bound on the curvature of the spline
        nonlinear_solver: solver for optimization problem in fitting spline
        non_negativity_constraint_distributions: 0 or 1 specifying if spline is to be nonnegative
        probability_constraint_of_distributions: 0 or 1 specifying if spline should integrate to 1
        granularity: granularity for MCL algorithm
    Returns:
        List of lists of patterns partitioning the set of patterns
    """

    distance_pattern = get_distance_dict(pattern_dict, time_series,
                                         **distr_options)

    patterns_not_enough_data = [pattern for pattern in pattern_dict
                                if len(pattern_dict[pattern]) <= 20]

    pattern_graph = WeightedGraph(distance_pattern)

    # Since the patterns are originally in a dictionary and thus in
    # 'random order,' we sort the clusters to make the heuristic deterministic
    clusters = pattern_graph.get_clusters(granularity)
    clusters = sorted(sorted(cluster) for cluster in clusters)

    for pattern in patterns_not_enough_data:
        cluster = get_nearest_cluster_heuristic(pattern, clusters)
        cluster.append(pattern)

    return tuple(map(tuple, clusters))


def get_cluster_from_pattern(pattern, clusters):
    """
    Helper function to return the cluster a pattern is in
    Args:
        pattern: A tuple of three values
        clusters: A list of lists of patterns
    Returns:
        The list containing the pattern
    """
    for cluster in clusters:
        if pattern in cluster:
            return cluster
    else:
        raise RuntimeError("Pattern not found in cluster")


def get_nearest_cluster_heuristic(pattern, clusters):
    """
    Uses a heuristic to find the cluster of patterns which is closest to the passed in pattern.

    Args:
        pattern: A tuple which is the pattern
        clusters: A list of lists of already clustered patterns
    Returns:
        The list of the closest cluster
    """
    # Heuristic 2014 GCS. Find a pattern that has 2 values in common and the third entry doesn"t change by 2
    for cluster in clusters:
        for near_pattern in cluster:
            if near_pattern[0] == pattern[0] and near_pattern[1] == pattern[1]:
                if abs(near_pattern[2] - pattern[2]) != 2:
                    return cluster
            if near_pattern[0] == pattern[0] and near_pattern[2] == pattern[2]:
                if abs(near_pattern[1] - pattern[1]) != 2:
                    return cluster
            if near_pattern[2] == pattern[2] and near_pattern[1] == pattern[1]:
                if abs(near_pattern[0] - pattern[0]) != 2:
                    return cluster
    # Find a pattern that has 2 values in common.
    for cluster in clusters:
        for near_pattern in cluster:
            if near_pattern[0] == pattern[0] and near_pattern[1] == pattern[1]:
                return cluster
            if near_pattern[0] == pattern[0] and near_pattern[2] == pattern[2]:
                return cluster
            if near_pattern[2] == pattern[2] and near_pattern[1] == pattern[1]:
                return cluster
    # find pattern that has 1 value in common.
    for cluster in clusters:
        for near_pattern in cluster:
            if near_pattern[0] == pattern[0]:
                return cluster
            if near_pattern[1] == pattern[1]:
                return cluster
            if near_pattern[2] == pattern[2]:
                return cluster
    print('the pattern ', pattern, ' does not belong to any cluster of the List')
    return 'no cluster found'


def verify_derivative_bounds(bounds):
    """
    Checks if the bounds are valid, raises error if they are not

    bounds should be a list of two numbers between 0 and 1
    """
    if len(bounds) != 2:
        raise RuntimeError('***ERROR: The derivative bounds must have exactly 2 entries.')
    bounds = (float(bounds[0]), float(bounds[1]))
    if bounds[0] > bounds[1]:
        raise RuntimeError('***ERROR: The lower derivative bound must be lower than the higher derivative bound.')
    if bounds[0] <= 0.0 or bounds[1] >= 1.0:
        raise RuntimeError('***ERROR The derivative bounds must be inside the interval (0.0, 1.0).')


def get_derivative_bounds(derivative_list, bounds):
    """
    Accepts a list of derivative values and returns the
    corresponding quantiles of the list that were specified
    by the user in the options.
    Args:
        derivative_list: A list of derivative values
        bounds: A list of 2 numbers between 0 and 1, or a string
                of the form '<lower_number>,<upper_number>' where values
                in <> are replaced by corresponding numbers
    Returns:
        (low_derivative_bound, high_derivative_bound): A pair of values
    """

    if isinstance(bounds, str):
        bounds = bounds.split(',')
        bounds = [float(bound) for bound in bounds]
    verify_derivative_bounds(bounds)
    lower_bound, upper_bound = bounds
    derivatives = pd.Series(derivative_list)
    low_derivative_bound = derivatives.quantile(lower_bound)
    high_derivative_bound = derivatives.quantile(upper_bound)
    return low_derivative_bound, high_derivative_bound


def evaluate_derivatives(series, **spline_options):
    """
    Approximates the derivative of the time series contained in the
    OrderedDictionary series and returns an OrderedDictionary containing
    {time -> derivative}. It approximates it by fitting an epispline to the
    series and then computes the derivative of the epispline.

    You must pass in a time series, but the rest of the arguments are
    optional and are related to the fitting of the epispline.
    """
    if isinstance(series, OrderedDict):
        series = pd.Series(series)

    smallest_date = min(series.index.values)
    largest_date = max(series.index.values)

    derivatives_of_day = OrderedDict()

    # we compute the spline for each day and get the derivatives
    for day in pd.date_range(smallest_date, largest_date):

        current_day_data = series[series.index.date == day.date()]
        if len(current_day_data) <= 23: # hardcoded
            continue
        x = current_day_data.index.hour.tolist()
        y = current_day_data.values.tolist()
        spline = Spline(x, y, **spline_options)
        derivatives_of_day[day] = OrderedDict()
        for hour in x:
            derivatives_of_day[day][hour] = spline.derivative(hour)

    # we create a dictionary with all derivatives
    all_derivatives = OrderedDict()
    for day in derivatives_of_day:
        for hour in range(24):
            dt = day + datetime.timedelta(hours=hour)
            try:
                all_derivatives[dt] = derivatives_of_day[day][hour]
            except KeyError:
                # We may not have derivatives for all hours of the day
                continue

    return all_derivatives

def main():
    argument_parser = globals.construct_argument_parser()
    args = argument_parser.parse_args()
    globals.assign_globals_from_parsed_args(args)

    column_name = globals.GC.column_name
    data_filename = globals.GC.input_filename
    try:
        dataframe = pd.read_csv(globals.GC.input_directory + os.sep + data_filename, index_col=0, parse_dates=True)
    except (FileNotFoundError, OSError):
        print("{} was not found."
              " Please check that it actually resides in the directory specified.".format(data_filename))
        sys.exit(1)

    # Compute Derivatives
    if column_name not in dataframe.columns:
        print("The column {} was not found in {}."
              " Please check that the data has the column specified.".format(column_name, data_filename))
        sys.exit(1)
    column_data = dataframe[column_name]
    if column_data.empty:
        print("The column specified does not contain any values and derivatives cannot be computed.")
        sys.exit(1)
    derivatives = evaluate_derivatives(column_data, globals.GC.epifit_error_norm,
                                       globals.GC.seg_N, globals.GC.seg_kappa,
                                       globals.GC.L1Linf_solver,
                                       globals.GC.L2Norm_solver)
    bounds = get_derivative_bounds(list(derivatives.values()), globals.GC.derivative_bounds)
    dataframe[column_name + '_derivatives'] = pd.Series(derivatives)

    # Compute Patterns
    pattern_dictionary = {}  # maps patterns to lists of dates that have those patterns
    date_pattern_dict = {}  # maps datetimes to patterns
    for dt in dataframe.index:
        pattern = create_pattern(dt, dataframe, column_name + '_derivatives', bounds)
        if pattern in pattern_dictionary:
            pattern_dictionary[pattern].append(dt)
        else:
            pattern_dictionary[pattern] = [dt]
        date_pattern_dict[dt] = pattern

    clusters = get_clusters(pattern_dictionary, dataframe[column_name], seg_N=globals.GC.seg_N,
                 seg_kappa=globals.GC.seg_kappa, nonlinear_solver=globals.GC.nonlinear_solver,
                 non_negativity_constraint_distributions=globals.GC.non_negativity_constraint_distributions,
                 probability_constraint_of_distributions=globals.GC.probability_constraint_of_distributions,
                 granularity=globals.GC.granularity)


    for current_dt in date_pattern_dict:
        pattern = create_pattern(current_dt,
                                 dataframe,
                                 column_name + '_derivatives',
                                 bounds)
        closest_cluster = get_cluster_from_pattern(pattern, clusters)
        representative_pattern = closest_cluster[0]
        date_pattern_dict[current_dt] = representative_pattern

    dataframe[column_name + '_derivative_patterns'] = pd.Series(date_pattern_dict)

    dataframe.to_csv(globals.GC.input_directory + os.sep + data_filename)

if __name__ == '__main__':
    main()
