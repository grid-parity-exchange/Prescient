#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains one class to manage a single skeleton point path as well
as one class to manage a set of skeleton point paths. A skeleton point is
basically a hyperrectangle object with a float value assigned to it.
"""

import datetime
import os
import random
from itertools import product

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

from prescient.util.distributions.distribution_factory import distribution_factory
from prescient.gosm.structures.hyperrectangles import Hyperrectangle, Interval


def parse_dps_path_file_all_sources(filename):
    """
    This function will parse out all paths and day part separators from a
    dps-paths-file. It does this by first scanning the file for the source
    names and then calling parse_dps_path_file with each source name.

    Args:
        filename (str): The name of the dps-paths file
    Returns:
        dict: A dictionary of the form
            {source_name -> {'paths': list[paths], 'dps': list[int]}}
    """
    dps_paths = {}
    source_names = []
    with open(filename) as f:
        for line in f:
            if line.startswith('Source:'):
                _, source_name = line.rstrip().split()
                source_names.append(source_name)

    for source_name in source_names:
        dps, paths, _ = parse_dps_path_file(source_name, filename)
        dps_paths[source_name] = {'dps': dps, 'paths': paths}

    return dps_paths


def parse_dps_path_file(source_name, filename):
    """
    This function should parse a dps and path file and
    return a list of possible paths and dps.
    """
    with open(filename) as f:

        # Extract the lines from the file that belong to the specified source.
        lines = f.read().splitlines()
        first_line_index = None
        last_line_index = None
        for i, line in enumerate(lines):
            if line.startswith('Source'):
                if source_name == line.split()[1]:
                    first_line_index = i
                elif first_line_index is not None:
                    last_line_index = i - 1
                    break
        if first_line_index is None:
            raise RuntimeError("The source '{}' is not contained in the "
                               "dps-file '{}'.".format(source_name, filename))
        elif last_line_index is None:
            last_line_index = len(lines) - 1
        lines = lines[first_line_index:last_line_index + 1]

    # parse out day part separators.
    for line in lines:
        if line.startswith('dps'):
            dps = line.split()[1:]

            for i, hour in enumerate(dps):
                # Since a dps can be 'sunrise' or 'sunset', we may get an error
                # in casting the string to an int
                try:
                    dps[i] = int(hour)
                except ValueError:
                    continue
            break
    else:
        raise RuntimeError("No day part separators "
                           "found in file {}".format(filename))

    # A list of paths which use all the day part separators
    paths = []

    # A list of all remaining paths which are shorter
    remaining_paths = []

    # Iterate over all lines and read all lines
    # following the one starting with 'Paths'.
    read_line = 0
    nonempty_lines = [line for line in lines if line != '']
    for line in nonempty_lines:
        if read_line == 1:

            # The first word of input_data is the name of the pattern.
            pattern, *path = line.split()

            # If the length of the path equals the number of day part
            # separators, add the path to the list.
            if len(path) == len(dps) - 1:
                paths.append(Path(path, pattern))

            # If the length of the path is less than the number of day part
            # separators, save it temporarily to
            # check the actual paths for consistency.
            elif len(path) < len(dps) - 1:
                remaining_paths.append(Path(path, pattern))
            else:
                raise RuntimeError('There is a path consisting of more points '
                                   'than there are day part separators. '
                                   "Check '{}'.".format(filename))

        if line.startswith('Paths'):
            read_line = 1

    if read_line == 0:
        raise RuntimeError("Couldn't find a line starting with 'Paths' for "
                           "source {} in '{}'.".format(source_name, filename))

    if len(paths) == 0:
        raise RuntimeError("No paths of the proper length could be "
                           "constructed. Check the paths file to see if the "
                           "number of dps is appropriate.")

    return dps, paths, remaining_paths


def fix_solar_dps(hours, sunrise, sunset):
    """
    This will replace the day part separators in the list dps that are
    indicated by the words 'sunrise' and 'sunset' with the corresponding values
    passed in sunrise and sunset. It is assumed that sunrise is the first
    element of the list and sunset is the last element of the list. This
    requires the second dps occur after sunrise and the second-to-last dps
    occur before sunset.

    Args:
        hours: The list of day part separators with sunrise as the first element
            and sunset as the last element
        sunrise (int): The hour (zero-indexed) of sunrise
        sunset (int): The hour (zero-indexed) of sunset

    """
    dps = hours.copy()

    if dps[0] != 'sunrise' or dps[-1] != 'sunset':
        raise RuntimeError("The first and last dps for solar dps must be "
                           "sunrise and sunset respectively.")

    dps[0] = sunrise
    dps[-1] = sunset
    if dps[1] < sunrise:
        raise RuntimeError("The second dps occurs prior to sunrise.")
    elif dps[1] == sunrise:
        del dps[1]

    if dps[-2] > sunset:
        raise RuntimeError("The second-to-last dps occurs after sunset.")
    elif dps[-2] == sunset:
        del dps[-2]

    return dps


class Path:
    """
    This class will just store information about a single possible path.
    This means it will contain a trail of hyperrectangles or cutpoint sets
    used up to this point as well as a pattern for the possible next paths.

    Attributes:
        trail (List[str]): A list of names of hyperrectangles up to this point
        next_pattern (str): The name of the next pattern of hyperrectangles
    """
    def __init__(self, trail, next_pattern):
        self.trail = tuple(trail)
        self.next_pattern = next_pattern

    def to_skeleton_paths(self, dps, hyperrectangle_set, source_names=None):
        """
        This will construct a SkeletonPointPath object by replacing all the
        hyperrectangle names with corresponding Hyperrectangle objects as well
        as replacing the pattern with all possible Hyperrectangles in the
        pattern. It will also fill all the spaces in between with pseudo
        hyperrectangles.

        Args:
            dps (List[int]): A list of day part separators
            hyperrectangle_set (HyperrectangleSet): The object containing all
                the information about Hyperrectangles and corresponding
                patterns
            source_names (list[str]): Optionally, the source names that the
                dimensions of the hyperrectangle correspond to
        Returns:
            List[SkeletonPointPath]: A list of all possible SkeletonPointPaths
                which can be constructed from this Path object
        """
        path = [hyperrectangle_set.get_hyperrectangle(name)
                for name in self.trail]
        skeleton_paths = []
        pattern = hyperrectangle_set.get_pattern(self.next_pattern)
        next_hyperrectangles = pattern.hyperrectangles

        for last_rectangle in next_hyperrectangles:
            path_dict = {}
            for hour, rect in zip(dps, path + [last_rectangle]):
                path_dict[hour] = rect

            skeleton_paths.append(SkeletonPointPath(
                dps, path_dict, source_names=source_names))

        return skeleton_paths

    def to_one_dim_paths(self, interval_set):
        """
        This will construct a OneDimPath object from the Path and the
        Intervals contained in the interval_set object.

        Args:
            interval_set (HyperrectanglePatternSet): The set of Intervals
                the paths will be constructed from
        Returns:
            list[OneDimPath]: The list of OneDimPath objects that can be
                created from this Path
        """
        path = [interval_set.get_hyperrectangle(name)
                for name in self.trail]
        skeleton_paths = []
        pattern = interval_set.get_pattern(self.next_pattern)
        next_intervals = pattern.hyperrectangles

        for last_interval in next_intervals:
            skeleton_paths.append(OneDimPath(path + [last_interval]))

        return skeleton_paths

    def __str__(self):
        return '_'.join(self.trail + (self.next_pattern,))

    __repr__ = __str__

    def __len__(self):
        return len(self.trail) + 1

    def __lt__(self, other):
        """
        This should be just a lexical ordering based on lists of strings.
        """
        if len(self) < len(other):
            return True
        elif len(self) > len(other):
            return False

        if self.trail == other.trail:
            return self.next_pattern < other.next_pattern
        else:
            return self.trail < other.trail

    def __eq__(self, other):
        return (self.trail == other.trail and
                self.next_pattern == other.next_pattern)


def generate_paths_from_single_set(n, pattern_name, patterns):
    """
    This function will generate all the possible paths of length n
    which can be generated from simply repeating a pattern set n times.
    There will be |pattern_set|^n paths of length n and many more paths
    of shorter length.

    Args:
        n (int): The number of times to repeat the set, usually in alignment
            with say the number of day part separators
        pattern_name (str): The name of the pattern set
        patterns (List[str]): The names of the pattern in the pattern set
    Returns:
        List[Path]: The list of all generated paths
    """
    paths = []
    for i in range(n):
        for path in product(patterns, repeat=i):
            paths.append(Path(path, pattern_name))
    return paths


def write_paths_file(dps, paths, filename, source_name=None):
    """
    This will construct a paths file from a list of day part separators and
    a list of paths.

    Args:
        dps (List[int]): The list of hours that will be day part separators
        paths (List[Path]): All the path objects that will be contained in
            the file
        filename (str): The name of the file to write to
        source_name (str): An optional name of sources that this paths file
            will be associated with.
    """
    with open(filename, 'w') as f:
        if source_name is not None:
            f.write("Source: {}\n".format(source_name))
        f.write("dps {}\n".format(" ".join(map(str, dps))))
        f.write("Paths(decision, path)\n")
        for path in sorted(paths):
            f.write(path.next_pattern + ' ' + " ".join(path.trail) + '\n')


def sample_error_vector(dps, distributions, source_name, sunrise_value=None,
                        sunset_value=None):
    """
    This samples the distributions at the day part separators and then
    interpolates the values in between.

    It is assumed that we are sampling for a single source for now.

    It will be the case for solar data that first and last day part
    separators do not occur at 0 and 23, but somewhere in between. This
    will still return a 24 vector but the data before the first and last
    dps will not be interpolated, rather it will just equal the value at
    the first and last day part separator.

    sunrise_value and sunset_value can be passed in to stop computation
    at sunrise and sunset dps instead using the passed in values for those
    hours.

    Args:
        dps (list[int]): The day part separators to sample at
        hourly_distributions (dict[int,BaseDistribution]): A mapping from
            day part separators to distributions. The dimensionality of
            the distribution should match that of the hyperrectangles in
            the path
        source_name (list[str]): The name of the source
        sunrise_value (float): This is the value at sunrise to force
            the value to equal this instead of computing from a
            distribution
        sunset_value (float): This is the value at sunset to force
            the value to equal this instead of computing from a
            distribution
    Returns:
        dict[str,list[int]]: A mapping from source names to 24-error
            vectors
    """
    values = []
    if sunrise_value is not None and sunset_value is not None:
        for hour in dps:
            distr = hourly_distributions[hour]
            values.append(distr.sample_one())
    else:
        # For solar data, the first and last dps are sunrise and sunset
        # and are predetermined.
        values.append(sunrise_value)
        for hour in dps[1:-1]:
            distr = hourly_distributions[hour]
            values.append(distr.sample_one())
        values.append(sunset_value)
    return {source_name: list(np.interp(range(24), dps, values))}


class OneDimPath:
    """
    This is a one-dimensional analog of SkeletonPointPath. It represents a
    sequence of intervals. Internally, it stores Interval objects.
    This class exists for convenience.
    """
    def __init__(self, intervals, name=None):
        """
        Args:
            intervals (list[Interval]): The sequence of interval objects which
                correspond to the path in question.
            name (str): Optionally, a name of the interval
        """
        self.intervals = intervals
        if name is None:
            self.name = '_'.join(interval.name for interval in intervals)
        else:
            self.name = name

    def slice(self, start, end):
        """
        This will return the slice of the interval starting at start and going
        up to one before end. This will return a OneDimPath object with this
        slice of the interval

        Args:
            start (int): The index of the beginning of the slice of the path
            end (int): One past the index of the last element of the slice
        Returns:
            OneDimPath: The path constructed from intervals in the slice
        """
        return OneDimPath(self.intervals[start:end])

    def rectangle_from_path(self, dimkeys=None):
        """
        This should return a hyperrectangle constructed from the intervals at
        the day part separators. This method is only implemented for paths
        which have one-dimensional data at each hour.

        The names of the sources in the rectangle will be the hours.

        Args:
            dimkeys (list): The names of the dimensions, if not specified,
                defaults to integers 0,1,2,...
        Returns:
            Hyperrectangle: The constructed hyperrectangle
        """
        bounds = [(interval.a, interval.b) for interval in self.intervals]
        if dimkeys is None:
            dimkeys = list(range(len(bounds)))
        return Hyperrectangle(bounds=bounds, dimkeys=dimkeys)

    def compute_conditional_expectation_on(self, distributions,
            cdf_inverse_tolerance=1e-4, cdf_inverse_max_refinements=10, 
            cdf_tolerance=1e-4):
        """
        This function will compute the conditional expection of each of the
        distributions on the corresponding intervals. There must be the same
        amount of distributions as there are intervals in the object.
        The order of the distributions determines on which interval the
        conditional expectation is computed for each distribution

        Args:
            distributions (list[Distribution]): One distribution for each
                interval
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (float): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        Returns:
            list[float]: The conditional expectation of each distribution
                on its corresponding interval
        """
        def expectation(distr, interval):
            """
            Temporary function which passes the tolerance levels to the
            conditional expectation function.

            Args:
                distr (BaseDistribution): A given distribution to compute
                    the conditional expectation on
                interval (Interval): The interval conditioned on
            Returns:
                float: The conditional expectation
            """
            return distr.conditional_expectation(
                interval, cdf_inverse_tolerance, cdf_inverse_max_refinements,
                cdf_tolerance)

        return [expectation(distr, interval) for distr, interval
                in zip(distributions, self.intervals)]

    def __repr__(self):
        return "OneDimPath({})".format(self.name)

    def __str__(self):
        string = "OneDimPath({})\n".format(self.name)
        for interval in self.intervals:
            string += str(interval.interval) + '\n'
        return string


def independent_probability(path):
    """
    This will compute the probability of a given path assuming temporal
    independence. This is then just the product of the volumes of the
    rectangles included in the path.

    Args:
        path (OneDimPath): The path taken for a given scenario
    Returns:
        float: The probability of the path
    """
    product = 1
    for interval in path.intervals:
        product *= interval.volume
    return product


def temporally_dependent_probability(path, copula, dps, monte_carlo=False,
                                     sample_size=100000):
    """
    This will compute the probability of the path without the assumption of
    independence over the day part separators. This is done by passing in a
    copula fitted to the data at the day part separators and then computing
    the probability of being in each of the intervals from the copula.

    Args:
        path (OneDimPath): The path taken for a given scenario
        copula (MultivariateDistribution): The distribution fitted to the
            data at the day part separators
        dps (list[int]): The day part separators used for constructing the
            path and the copula
        montecarlo (bool): A boolean to be set to True if monte carlo
            integration is to be used
        sample_size (int): The number of samples to use in monte carlo
            integration if montecarlo is set to True
    Returns:
        float: The probability of the path
    """
    # We get the rectangle constructed from intervals in the path
    rect = path.rectangle_from_path(dps)

    if sorted(copula.dimkeys) != sorted(rect.dimkeys):
        raise ValueError("The dimension names of the copula must be the "
                         "integers corresponding to the hours of the dps.")
    if monte_carlo:
        return copula.mc_probability_on_rectangle(rect, n)
    else:
        return copula.probability_on_rectangle(rect)


def spatially_dependent_probability(dps, path_dict, copulas, singletons,
                                    monte_carlo=False, n=100000):
    """
    This computes the probability of the scenario assuming independence
    across time but not across space. 

    Each copula should have its dimkeys attribute initialized with the
    names of the sources used to fit the copulas.

    Args:
        dps (list[int]): The list of day part separators used for generating
            scenarios
        path_dict (dict): A mapping from source names to the OneDimPath used
            for that source.
        copulas (dict): This is a dictionary mapping a specific cluster
            of source names and a day part separator to a copula fitted to
            the data of those sources at the hour specified. This should
            be of the form
                {(source_names, dps) -> copula}
            where source_names is a tuple of names and dps an integer.
        singletons (list[str]): A list of the source names that are
            singletons. For these sources, we will just compute the
            probability independently.
        monte_carlo (bool): Set this option to True if you wish to use
            monte carlo integration to compute the probability
        n (int): The number of samples to use for monte carlo integration
    Returns:
        float: The probability of the scenario
    """
    probability = 1

    # For each cluster and copula fit  to that cluster
    for ((source_names, hour), copula) in copulas.items():
        interval_index = dps.index(hour)

        # Pull out the bounds used for that day part separator
        bounds = [path_dict[source].intervals[interval_index]
                  for source in source_names]

        # Construct hyperrectangle with the bounds
        rectangle = Hyperrectangle(bounds, dimkeys=source_names)
        if monte_carlo:
            probability *= copula.mc_probability_on_rectangle(rectangle, n)
        else:
            probability *= copula.probability_on_rectangle(rectangle)

    for source_name in singletons:
        path = path_dict[source_name]
        probability *= independent_probability(path)

    return probability


class SkeletonPointPath:
    """
    This class manages one single skeleton point path.
    """

    def __init__(self, dps, path_dict, name=None, source_names=None):
        """
        Initializes an instance of the SkeletonPointPath class.

        Args:
            dps (list[int]): the list of day part separators
            path_dict (dict): a dictionary like
                {dps -> Hyperrectangle}  describing the path
            name (str): the name of the path
            source_names (list[str]): Optionally, the names of sources this
                path is relevant to
        """
        self.dps = dps
        self.path_dict = path_dict
        if name is None:
            self.name = self._set_path_name()
        else:
            self.name = name

        # if no source_names, the names are just the dimension indices
        if source_names is None:
            # Pick an arbitrary rectangle to find its dimension
            rect = next(iter(path_dict.values()))
            self.source_names = list(range(rect.ndim))
        else:
            self.source_names = source_names

    def to_solar_path(self, sunrise_values=None, sunset_values=None):
        """
        This converts a skeleton path into a solar path where it is assumed
        that the first and last day part separators are sunrise and sunset.

        The sunrise and sunset values can be passed in to construct the solar
        path with these values.

        Args:
            sunrise_values (dict[str,float]): This maps source names to the
                error value at sunrise to force that specific source to have
                that error value at sunrise instead of computing from a
                distribution
            sunset_values (dict[str,float]): This maps source names to the
                error value at sunset to force that specific source to have
                that error value at sunset instead of computing from a
                distribution
        Returns:
            SolarPath: The equivalent SolarPath
        """
        return SolarPath(self.dps, self.path_dict, self.name,
                         self.source_names, sunrise_values, sunset_values)

    def compute_error_vector(self, hourly_distributions):
        """
        This computes the error vectors using the distributions to compute
        conditional expectations over the hyperrectangles. It interpolates
        to find the values in between.

        It will be the case for solar data that first and last day part
        separators do not occur at 0 and 23, but somewhere in between. This
        will still return a 24 vector but the data before the first and last
        dps will not be interpolated, rather it will just equal the value at
        the first and last day part separator.

        sunrise_value and sunset_value can be passed in to stop computation
        at sunrise and sunset dps instead using the passed in values for those
        hours.

        Args:
            hourly_distributions (dict[int,BaseDistribution]): A mapping from
                day part separators to distributions. The dimensionality of
                the distribution should match that of the hyperrectangles in
                the path
        Returns:
            dict[str,list[float]]: A mapping from source names to 24-error
                vectors
        """
        if len(self.source_names) == 1:
            # For a single source, conditional_expectation returns just a value
            # so we need to treat it differently.
            source_name = self.source_names[0]
            values = []
            for hour in self.dps:
                distr = hourly_distributions[hour]
                rect = self.path_dict[hour]
                values.append(distr.conditional_expectation(rect))
            print(values)

            errors = np.interp(range(24), self.dps, values)
            return {source_name: list(errors)}
        else:
            # For multiple sources, conditional_expectation returns a
            # dictionary which needs to be handled differently.
            error_vectors = {name:[] for name in self.source_names}
            for hour in self.dps:
                distr = hourly_distributions[hour]
                rect = self.path_dict[hour]
                mean_vector = distr.conditional_expectation(rect, as_dict=True)
                for name in self.source_names:
                    error_vectors[name].append(mean_vector[name])
            for name in self.source_names:
                values = error_vectors[name]
                error_vectors[name] = list(np.interp(
                    range(24), self.dps, values))
            return error_vectors

    def compute_mc_probability(self, copula, n):
        """
        The copula must have its dimkeys attribute set to the hours of the
        day part separators.

        It will compute the probability
        by computing the integral over the rectangle formed from the intervals.
        The difference between this and compute_probability is that this uses
        Monte Carlo integration to compute the integral involved in
        the probability computation.

        Args:
            dps (List[int]): The points that form the day part separators
            copula (CopulaBase): A fitted copula
        Returns:
            float: The probability of the path
        """
        rect = self.rectangle_at_dps(self.dps)
        return copula.mc_probability_on_rectangle(rect, n)

    def compute_probability(self, copula=None):
        """
        This computes the probability of the path. If copula is not passed in,
        it does this by computing the volume of the hyperrectangles at the
        day part separators and multiplying them together.

        The copula must have its dimkeys attribute set to the hours of the
        day part separators.

        If a copula is passed in, then it will instead compute the probability
        by computing the integral over the rectangle formed from the intervals.

        Args:
            dps (List[int]): The points that form the day part separators
            copula (CopulaBase): A fitted copula
        Returns:
            float: The probability of the path
        """
        if copula is None:
            probability = 1
            for h in self.dps:
                rect = self.path_dict[h]
                probability *= rect.compute_volume()
            return probability
        else:
            rect = self.rectangle_at_dps()
            return copula.probability_on_rectangle(rect)

    def rectangle_at_dps(self):
        """
        This should return a hyperrectangle constructed from the intervals at
        the day part separators. This method is only implemented for paths
        which have one-dimensional data at each hour.

        The names of the sources in the rectangle will be the hours.

        Returns:
            Hyperrectangle: The constructed hyperrectangle
        """
        bounds = {}
        for hour in self.dps:
            rect = self.path_dict[hour]
            # The dictionary should only have one element
            (interval,) = rect.bounds.values()
            bounds[hour] = interval
        return Hyperrectangle(input_data=bounds, sources=self.dps)

    def _set_path_name(self):
        """
        Returns the name of a path like "quicklow-quickhigh-quicklow".

        Returns:
            str: name of the path
        """

        names = []
        # Make sure, the day part separators are in the right order.
        for hour in range(24):
            if hour in self.dps:
                names.append(self.path_dict[hour].name)
        name = '_'.join(names)

        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        string = self.name + '\n'
        for hour in self.dps:
            string += str(self.path_dict[hour]) + '\n'
        return string


def rectangle_from_paths_at_hour(paths, hour):
    """
    This function will construct a hyperrectangle from the intervals of the
    SkeletonPointPath objects at the hour specified. It is assumed that each
    of the paths are composed of 1-dimensional intervals and that this is
    possible.

    Args:
        paths (List[SkeletonPointPath]): The paths to construct the rectangle
            from
        hour (int): The hour of interest
    Returns:
        Hyperrectangle: The rectangle composed of the intervals on the paths
            at the hour specified with dimension names being the sources
    """
    bounds = {}
    names = []
    for path in paths:
        rect = path.path_dict[hour]
        # The dictionary should only have one element as it is 1-dimensional
        (interval,) = rect.bounds.values()
        (name,) = path.source_names
        bounds[name] = interval
        names.append(name)
    return Hyperrectangle(bounds=bounds, dimkeys=names)


class SolarPath(SkeletonPointPath):
    """
    This class is a SkeletonPointPath, but it has knowledge of when sunrise
    and sunset is. This is important for solar data as we don't compute the
    probability at these hours (we just assert that the value should be zero).
    """
    def __init__(self, dps, path_dict, name=None, source_names=None,
                 sunrise_values=None, sunset_values=None):
        """
        To construct a SolarPath, one must pass a list of day part separators
        and a dictionary mapping day part separators to associated
        Hyperrectangles.

        It is assumed that the first and last day part separator are sunrise
        and sunset.

        Optionally, one can pass the pass a dictionary mapping source names
        to the value an error scenario should be at sunrise and sunset. This
        will force the computation of the error vector at those hours to
        be those values at sunrise and sunset. Generally, this is made
        to be the negative of the forecast to force scenarios to be zero
        at those hours.

        Args:
            dps (list[int]): the list of day part separators
            path_dict (dict): a dictionary like
                {dps -> Hyperrectangle}  describing the path
            name (str): Optionally, one can give the path a name
            source_names (list[str]): Optionally, the names of sources this
                path is relevant to
            sunrise_values (dict[str,float]): This maps source names to the
                error value at sunrise to force that specific source to have
                that error value at sunrise instead of computing from a
                distribution
            sunset_values (dict[str,float]): This maps source names to the
                error value at sunset to force that specific source to have
                that error value at sunset instead of computing from a
                distribution
        """
        SkeletonPointPath.__init__(self, dps[1:-1], path_dict, name,
                                   source_names)
        self.all_dps = dps
        self.sunrise = dps[0]
        self.sunset = dps[-1]
        self.sunrise_values = sunrise_values
        self.sunset_values = sunset_values

    def compute_error_vector(self, hourly_distributions):
        """
        This computes the error vectors using the distributions to compute
        conditional expectations over the hyperrectangles. It interpolates
        to find the values in between.

        It will be the case for solar data that first and last day part
        separators do not occur at 0 and 23, but somewhere in between. This
        will still return a 24 vector but the data before the first and last
        dps will not be interpolated, rather it will just equal the value at
        the first and last day part separator.

        Args:
            hourly_distributions (dict[int,BaseDistribution]): A mapping from
                day part separators to distributions. The dimensionality of
                the distribution should match that of the hyperrectangles in
                the path
        Returns:
            dict[str,list[float]]: A mapping from source names to 24-error
                vectors
        """
        error_vectors = {}
        for source_name in self.source_names:
            values = []
            values.append(self.sunrise_values[source_name])
            for hour in self.dps:
                distr = hourly_distributions[hour]
                rect = self.path_dict[hour]
                values.append(distr.conditional_expectation(rect))
            values.append(self.sunset_values[source_name])

            errors = np.interp(range(24), self.all_dps, values)
            error_vectors[source_name] = errors.tolist()

        return error_vectors
