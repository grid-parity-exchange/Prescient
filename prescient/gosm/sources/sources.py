#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
sources.py

This file will contain more general versions of data containers than
the sources defined in uncertainty_sources.py. The main class that this module
exports is the Source class which is intended to store data of any sort in
a dataframe. This class should not be modified (unless there is a bug) to be
made more specific; it should be subclassed. In addition, unless the
method will obviously change the state of the object, all methods should
produce new objects instead of modifying objects.
"""

import sys
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import prescient.util.distributions.copula as copula
from prescient.util.distributions.distributions import UnivariateEmpiricalDistribution
from prescient.util.distributions.distributions import UnivariateEpiSplineDistribution
import prescient.gosm.derivative_patterns.derivative_patterns as dp
from prescient.gosm.markov_chains.states import State
from prescient.gosm.sources.segmenter import Criterion

power_sources = ['solar', 'wind', 'hydro']
recognized_sources = ['solar', 'wind', 'load', 'hydro']

# Default parameters for the non-required parameters for sources
defaults = {
    'is_deterministic': False,
    'frac_nondispatch': 1,
    'scaling_factor': 1,
    'forecasts_as_actuals': False,
    'aggregate': False,
}

class Source:
    """
    This class should act as a container for all the data related to a single
    source. The data is stored internally in a Pandas Dataframe.
    This class should have methods for segmentation (more generally pick
    all datetimes that satisfy a certain a criterion) and selection.

    Attributes:
        name (str): The name of the source
        data (pd.DataFrame): The internal dataframe storing all the data
        source_type (str): The type of the source (wind, solar, load, etc.)

    Args:
        name (str): the name of the source
        dataframe (pd.DataFrame): The frame containing all the data
        source_type (str): the type of the source (e.g. 'solar')
    """

    def __init__(self, name, dataframe, source_type):
        self.name = name
        self.data = dataframe

        # A little validation is done here
        # We check for duplicates
        if dataframe.index.duplicated().any():
            duplicates = dataframe.index[dataframe.index.duplicated()]
            raise ValueError("Error: Source {} has duplicate datetimes at {}"
                .format(name, ", ".join(map(str, duplicates))))

        self.source_type = source_type.lower()
        if source_type.lower() not in recognized_sources:
            raise ValueError("The source type '{}' is unrecognized, the only "
                             "recognized sources are {}"
                             .format(source_type,
                                     ", ".join(recognized_sources)))

    def check_for_column(self, column_name):
        """
        This method will check if the source has a column with the name
        specified. If it doesn't it will raise a RuntimeError.

        Args:
            column_name (str): The name of the column to check
        """
        if column_name not in self.data.columns:
            raise RuntimeError("Source {} has no '{}' column".format(
                self.name, column_name))

    def window(self, column_name, lower_bound=-np.inf, upper_bound=np.inf):
        """
        Finds the window of data such that the column value is between
        the two bounds specified. Returns a Source object with data
        contained in the window. The bounds are inclusive.

        Args:
            column_name (str): The name of the column
            lower_bound (float): The lower bound, if None, no lower bound
            upper_bound (float): The upper bound, if None, no upper bound
        Returns:
            Source: The window of data
        """
        self.check_for_column(column_name)

        new_frame = self.data[(self.data[column_name] >= lower_bound) &
                              (self.data[column_name] <= upper_bound)]
        return Source(self.name, new_frame, self.source_type)

    def enumerate(self, column_name, value):
        """
        Finds the window of data such that the column field is equal to the
        value. Returns a Source object with the data contained in the window.

        Args:
            column_name (str): The name of the column
            value: The value you want all datetimes to have in the new window
        Returns:
            Source: The data will have all rows which match value
        """
        self.check_for_column(column_name)

        new_frame = self.data[self.data[column_name] == value]
        return Source(self.name, new_frame, self.source_type)

    def rolling_window(self, day, historic_data_start=None,
                       historic_data_end=None):
        """
        Creates a Rolling Window of data which contains a historic dataframe
        and a dayahead dataframe. The historic data is all data up to the day
        and the dayahead data is the data for that day.

        Using non-datetime objects (pd.TimeStamp, strings, np.datetime64)
        probably works but not guaranteed. This is contingent on pandas
        datetime indexing.

        Args:
            day (datetime.datetime): The datetime referring to hour zero of the
                desired day to create a window up to that day
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
        Returns:
            RollingWindow: The rolling window of data
        """

        # If start not specified, we take the first date in dataframe
        if historic_data_start is None:
            historic_data_start = min(self.data.index)
        # If end not specified, we take last date before the passed in day
        if historic_data_end is None:
            historic_data_end = day - datetime.timedelta(hours=1)
        historic_frame = self.data[historic_data_start:historic_data_end]
        dayahead_frame = self.data[day:day+datetime.timedelta(hours=23)]

        # This suppresses warnings, This should return a copy anyways, so don't
        # need a warning.
        historic_frame.is_copy = False
        dayahead_frame.is_copy = False

        return RollingWindow(self.name, historic_frame,
                    self.source_type, dayahead_frame)

    def solar_window(self, day, historic_data_start=None,
                     historic_data_end=None):
        """
        Creates a SolarWindow of data which contains a historic dataframe
        and a dayahead dataframe. The historic data is all data up to the day
        and the dayahead data is the data for that day.

        Using non-datetime objects (pd.TimeStamp, strings, np.datetime64)
        probably works but not guaranteed. This is contingent on pandas
        datetime indexing.

        Args:
            day (datetime.datetime): The datetime referring to hour zero of the
                desired day to create a window up to that day
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
        Returns:
            SolarWindow: The rolling window of data
        """
        window = self.rolling_window(day, historic_data_start,
                                     historic_data_end)
        return window.solar_window()

    def add_column(self, column_name, series):
        """
        Adds a column of data to the dataframe. This data should be indexed
        by datetime.

        Args:
            column_name (str): The name of the column to add
            series (pd.Series or dict[datetime.datetime,value]): The data
                indexed by datetime to add to the dataset
        """
        self.data[column_name] = pd.Series(series)

    def get_day_of_data(self, column_name, day):
        """
        This function returns a pandas Series of all the data in the column
        with the specific day as an index.

        Args:
            column_name (str): The desired column
            day (datetime-like): the day, which can be coerced into
                a pd.Timestamp
        Returns:
            pd.Series: A series of relevant data
        """
        self.check_for_column(column_name)
        dt = pd.Timestamp(day)
        column = self.data[column_name]
        return column[column.index.date == dt.date()]

    def get(self, column_name, row_index):
        """
        Get the value stored in column specified by column_name and the row
        specified by the row_index

        Args:
            column_name (str): The name of the column
            row_index (datetime.datetime): The datetime for which you want data
        """
        self.check_for_column(column_name)
        return self.data[column_name][row_index]

    def get_column(self, column_name):
        """
        Returns the column of data with that column name. This will also return
        a column without any nan values.

        Args:
            column_name (str): The name of the column
        Returns:
            pd.Series: The requested column
        """
        self.check_for_column(column_name)
        return self.data[column_name].dropna()

    def get_state_walk(self, state_description, class_=State):
        """
        This method should walk through the datetimes and construct a sequence
        of the different states of the historic data. The specification for
        what constitutes a state is passed in the state_description argument.

        Args:
            state_description (StateDescription): Specification for determining
                what the state for each datetime is
            class_ (Class): The type of state you wish to instantiate
        Returns:
            A dictionary of mapping datetimes to states constituting the walk
        """
        states = {}
        names = state_description.keys()
        for dt in self.data.index:
            name_value_mapping = {name: self.get(name, dt) for name in names}
            states[dt] = state_description.to_state(class_,
                                                    **name_value_mapping)

        return states

    def get_state(self, state_description, dt, class_=State):
        """
        This method should create the state for a specific datetime.
        The specification for what constitutes a state is passed in
        the state_description argument.

        Args:
            state_description (StateDescription): Specification for determining
                what the state for each datetime is
            dt (datetime.datetime): The relevant datetime
            class_ (Class): The type of state you wish to instantiate
        Returns:
            State: The state of the datetime
        """
        dt = pd.Timestamp(dt)
        names = state_description.keys()
        name_value_mapping = {name: self.get(name, dt) for name in names}
        return state_description.to_state(class_, **name_value_mapping)

    def get_quantiles(self, column_name, quantiles):
        """
        This method returns the quantiles of the column specified

        Args:
            column_name (str): The desired column to compute quantiles
            quantiles (List[float]): A list of floating points in [0,1]
        Returns:
            List[float]: The corresponding quantiles
        """
        self.check_for_column(column_name)
        return list(self.data[column_name].quantile(quantiles))

    def sample(self, column_name, lower_bound=-np.inf, upper_bound=np.inf):
        """
        This draws a sample from the data in the column that is between
        lower bound and upper bound. If no lower or upper bound is specified,
        then there is no bound on the data sampled.

        Args:
            column_name (str): The name of the column
            lower_bound (float): The lower bound, if not specified,
                no lower bound
            upper_bound (float): The upper bound, if not specified,
                no upper bound
        Returns:
            float: A single sampled value
        """
        self.check_for_column(column_name)
        window = self.window(column_name, lower_bound, upper_bound)
        column = window.get_column(column_name)
        return float(column.sample())

    def apply_bounds(self, column_name, lower_bound=-np.inf,
                     upper_bound=np.inf):
        """
        This function should take the column with the name specified and
        fix any value in the column below the corresponding value in
        lower_bound to the lower_bound and likewise for upper_bound.

        lower_bound and upper_bound may be Pandas Series or they may
        be a single value acting as a bound. If no lower bound is passed,
        the lower bound is minus infinity, similarly for upper bound, if none
        is passed, the upper bound is infinity.

        This function changes the state of the source.

        Args:
            column_name (str): Name of the column
            lower_bound (float or pd.Series): A lower bound for data
            upper_bound (float or pd.Series): An upper bound for data
        """
        self.check_for_column(column_name)

        if lower_bound is None:
            lower_bound = -np.inf
        if upper_bound is None:
            upper_bound = np.inf
        column = self.data[column_name]
        self.data[column_name] = column.clip(lower_bound, upper_bound)

    def interpolate(self, column_name):
        """
        This function will interpolate the column specified so that every
        hour between the start of the data and the end of the data has a value.

        This function changes the state of the source.

        Args:
            column_name (str): name of the column to interpolate
        """
        self.check_for_column(column_name)

        start_date = min(self.data.index)
        end_date = max(self.data.index)
        date_range = pd.date_range(start_date, end_date, freq='H')
        self.data = self.data.reindex(date_range)
        column = self.data[column_name]
        column = column.interpolate()
        self.data[column_name] = column

    def scale(self, column_name, factor):
        """
        This function will scale the requested column by the factor
        passed in.

        This function changes the state of the source.

        Args:
            column_name (str): name of the column to scale
            factor (float): The factor to scale the column by
        """
        self.check_for_column(column_name)
        self.data[column_name] *= factor

    def to_csv(self, filename):
        """
        This writes self.data to the filename specified.

        Args:
            filename (str): The path to the file to write.
        """
        self.data.to_csv(filename)

    def histogram(self, filename, column_name):
        """
        Plots the histogram for the specified column to the specified file.

        Args:
            filename (str): The path to the file to write
            column_name (str): The name of the column
        """
        plt.figure()
        self.data[column_name].hist()
        plt.savefig(filename)

    def get_column_at_hours(self, column_name, hours):
        """
        This will take the dataframe pull out the column and return
        a dataframe with columns composed of the data at the specific hours.
        The column names will be the same as the hours specified.

        Args:
            column_name (str): The name of the column
            hours (List[int]): The hours desired
        Returns:
            pd.DataFrame: The specified dataframe
        """
        df = self.data
        df_list = []

        for hour in hours:
            # Get (and segment) the respective data, rename the
            # column containing the errors according to the
            # dps and only keep the days of the datetimes.
            # For now we just select the historic data at the respective hour.
            frame = df.loc[df.index.hour == hour, column_name].rename(hour)
            frame.index = frame.index.date
            df_list.append(frame)

        # Concatenate the data frames such that the resulting
        # dataframe only consists of dates that are
        # contained in all data frames. Then write the data to a
        # dictionary with the dps as keys.
        result_frame = pd.concat(df_list, axis=1, join='inner')
        return result_frame

    def split_source_at_hours(self, hours):
        """
        This function will create a Source object for each hour in hours
        containing solely the date occurring at that hour of the day.
        """
        # This dictionary will be indexed by hours and refer to each source
        hourly_sources = {}
        for hour in hours:
            hourly_df = self.data[self.data.index.hour == hour]
            hourly_sources[hour] = Source(self.name, hourly_df,
                                          self.source_type)
        return hourly_sources


    def compute_derivatives(self, column_name, start_date=None, end_date=None,
                            **spline_options):
        """
        This method will compute the derivatives of the the column specified
        by fitting a spline to each day of data and computing the derivative
        of it.

        This will store the results in a column in the data object with the
        name '<column_name>_derivatives'.

        You can specify for which days you want to compute derivatives for
        with the start date and end date options.

        Args:
            column_name (str): The name of the column to compute derivatives
            start_date (datetime-like): The start date to compute derivatives
            end_date (datetime-like): The last date to compute derivatives
        """
        self.check_for_column(column_name)

        if start_date is None:
            start_date = min(self.data.index)
        else:
            start_date = pd.Timestamp(start_date)

        if end_date is None:
            end_date = max(self.data.index)
        else:
            end_date = pd.Timestamp(end_date) + datetime.timedelta(hours=23)

        field = column_name
        derivative_column = field + '_derivatives'
        if derivative_column not in self.data.columns:
            derivatives = dp.evaluate_derivatives(
                self.data[field][start_date:end_date], **spline_options)
            self.data[derivative_column] = pd.Series(derivatives)

    def fit_copula_at_hours(self, column, hours, copula_class,
                            **distr_options):
        """

        """
        # If we want to compute the probability using a copula,
        # we must fit a copula to data at the hours of the dps
        # and integrate over the product of intervals
        data_at_hours = self.get_column_at_hours(column, hours)
        data_dict = data_at_hours.to_dict(orient='list')

        us = {}
        # We transform the data to [0,1]^n using marginals
        for hour in hours:
            marginal = UnivariateEpiSplineDistribution.fit(
                data_dict[hour], **distr_options)
            us[hour] = [marginal.cdf(x) for x in data_dict[hour]]
        copula = copula_class.fit(us, dimkeys=hours)
        return copula

    def fit_distribution_to_column(self, column_name, distr_class,
                                   **distr_options):
        """
        This method will fit the specified distribution to the specified
        column. Arguments specific to the distribution can be passed through
        keyword arguments.

        Args:
            column_name (str): The name of the column
            distr_class: The class to fit to the column
        Returns:
            BaseDistribution: The fitted distribution
        """
        self.check_for_column(column_name)
        values = self.get_column(column_name).tolist()
        return distr_class.fit(values, **distr_options)

    def fit_multidistribution_to_hours(self, column_name, day_ahead, hours,
                                       marginal_class, copula_class,
                                       criterion=None, marginal_options=None,
                                       copula_options=None):
        """
        This function will fit a multivariate distribution in the form of a
        copula with marginals to the data at the specified hours. It will
        select the data in the given hour, perform any desired segmentation
        on the hourly data, then fit a marginal distribution to that data.

        Then it will take the data at the given hours and transform it to
        [0,1]^n before fitting a copula to the data. Then it will construct
        a CopulaWithMarginals object which will be the appropriate
        distribution.

        Args:
            column_name (str): The name of the column to fit the distribution
                to
            day_ahead (datetime-like): The datetime on which we will be
                segmenting on
            hours (list[int]): A list of integers from 0 to 23 specifying the
                hours corresponding to the dimensions of the distribution
            marginal_class: The specific univariate distribution to be fit
                to the marginal data
            copula_class: The copula class to fit to the multivariate data
            criterion (Criterion): The segmentation criterion which segments
                the univariate data
            marginal_options (dict): A dictionary of keyword-value pairs
                specifying parameters for the marginal distribution
            copula_options (dict): A dictionary of keyword-value pairs
                specifying parameters for the copula
        Returns:
            CopulaWithMarginals: A multivariate distribution
        """
        self.check_for_column(column_name)
        if marginal_options is None:
            marginal_options = {}
        if copula_options is None:
            copula_options = {}

        hourly_sources = source.split_source_at_hours(hours_in_range)
        hourly_windows = {hour: source.rolling_window(day_ahead)
                          for hour, source in hourly_sources.items()}

        #This segments the data and fits a univariate distribution to the
        #segmented data.
        segmented_windows = {}
        marginals = {}
        forecasts_hour = {}
        for hour, window in hourly_windows.items():
            curr_dt = day_ahead + datetime.timedelta(hours=hour)

            # If criterion is not passed in, we do no segmentation
            if criterion is not None:
                segmented_windows[hour] = window.segment(curr_dt, criterion)
            else:
                segmented_windows[hour] = window
            series = window.get_column(column_name).tolist()
            distr = marginal_class.fit(series, **marginal_options)
            marginals[hour] = distr

        #To fit a copula to the data we need all data, not only the seperated one.
        #We have to transform it to [0,1]^n for the purposes of fitting a copula.
        hourly_df = source.get_column_at_hours(column_name, hours_in_range)
        transformed_series = {}
        for hour in hours_in_range:
            hourly_df[hour] = hourly_df[hour] + forecasts_hour[hour]
            transformed = [marginals[hour].cdf(x) for x in hourly_df[hour]]
            transformed_series[hour] = transformed

        #First fitting a copula to the transformed data and then computing a
        #multivariate distribution using the copula and the marginals.
        fitted_copula = copula_class.fit(transformed_series, hours,
                                         **copula_options)
        f = copula.CopulaWithMarginals(fitted_copula, marginals, hours)

    def compute_patterns(self, column_name, bounds, start_date=None,
                         end_date=None):
        """
        This function will compute for every datetime a pattern which is a
        3-tuple in the set {-1, 0, 1}^3.

        This will store the results in the column '<column_name>_patterns'

        This algorithm works by first fitting a pattern to each time based
        on the values at time-1hour, time, and time+1hour. The
        pattern is a 3-tuple in {-1,0,1}^3 where the value is -1 if the
        derivative is very negative, 0 if derivative is close to 0, and
        1 if the derivative is very positive. What 'very negative', 'close to
        0', and 'very positive' mean exactly is specified by the bounds
        argument. For example, if bounds is (0.3, 0.7), then anything below
        0.3 quantile is -1, between quantiles is 0, and above is 1.

        Args:
            column_name (str): The name of the column in question
            bounds (tuple): A 2-tuple representing the quantiles of the column
                which will serve as the boundaries between low, middle, and
                high values.
            start_date (datetime-like): The start datetime for the window of
                data for which you wish to include the patterns, if None
                the start datetime will be the first datetime in data
            end_date (datetime-like): The end datetime for the window of data
                for which you wish to include patterns, inclusive, if None
                will be the last datetime in data
        """
        self.check_for_column(column_name)
        # If start_date or end_date is None, we use the smallest and largest
        # datetime in the data respectively
        if start_date is None:
            start_date = self.data.index.min()

        if end_date is None:
            end_date = self.data.index.max()

        all_data = self.data[start_date:end_date]

        # These constitute the boundaries between low, medium and high
        # derivatives
        lower, upper = dp.get_derivative_bounds(all_data[column_name], bounds)

        # compute patterns
        pattern_dictionary = {}  # maps patterns to lists of dates with pattern
        date_pattern_dict = {}  # maps datetimes to patterns
        for dt in all_data.index:
            # We skip any missing data, leaving it NaN in the final dataframe
            if np.isnan(all_data[column_name][dt]):
                continue

            pattern = dp.create_pattern(dt, all_data,
                                        column_name, (lower, upper))
            date_pattern_dict[dt] = pattern

        date_patterns = pd.Series(date_pattern_dict)
        self.data[column_name + '_patterns'] = date_patterns[self.data.index]

    def cluster(self, column_name, granularity=5, start_date=None,
                end_date=None, **distr_options):
        """
        This function will compute the cluster for each datetime in the
        dataframe and will store the results in the column
            <column_name>_clusters
        The result stored will be a the associated cluster.

        For every pattern, an exp-epispline distribution is fitted to the
        corresponding errors for all datetimes with that pattern. Then using
        the Wets-distance as a metric, this uses the Markov Clustering
        Algorithm to cluster the patterns.

        Args:
            column_name (str): The name of the column with the patterns
            granularity (float): A parameter for the Markov Clustering
                Algorithm
            start_date (datetime-like): The start datetime for the window of
                data for which you wish to include the patterns, if None
                the start datetime will be the first datetime in data
            end_date (datetime-like): The end datetime for the window of data
                for which you wish to include patterns, inclusive, if None
                will be the last datetime in data
            distr_options (through kwargs): Any specification for the
                epispline, see UnivariateEpiSplineDistribution for the
                full specification
        """
        self.check_for_column(column_name)
        # If start_date or end_date is None, we use the smallest and largest
        # datetime in the data respectively
        if start_date is None:
            start_date = self.data.index.min()

        if end_date is None:
            end_date = self.date.index.max()

        all_data = self.data[start_date:end_date]

        # pattern_dictionary will map patterns to lists of datetimes of that
        # pattern
        pattern_dictionary = {}
        for dt in all_data.index:
            # We need to call any, since the data is a tuple
            if np.isnan(all_data[column_name][dt]).any():
                continue

            pattern = all_data[column_name][dt]
            if pattern in pattern_dictionary:
                pattern_dictionary[pattern].append(dt)
            else:
                pattern_dictionary[pattern] = [dt]

        clusters = dp.get_clusters(pattern_dictionary, all_data['errors'],
                                   granularity=granularity, **distr_options)

        # Maps datetimes to associated cluster
        cluster_map = {}

        for dt in all_data.index:
            # This needs to be any since the stored data is a 3-tuple, not
            # a singular value.
            if np.isnan(all_data[column_name][dt]).any():
                continue

            pattern = all_data[column_name][dt]
            closest_cluster = dp.get_cluster_from_pattern(pattern, clusters)
            cluster_map[dt] = closest_cluster

        cluster_series = pd.Series(cluster_map)
        self.data[column_name + '_clusters'] = cluster_series[self.data.index]

    def estimate_sunrise_sunset(self, date):
        """
        This will estimate the hours of sunrise and sunset on the given date.
        It does this by taken the averages of the hours of first observed
        solar power generation and hours of last observed power generation
        for the two weeks before the given date. It then rounds the values.

        If this is not a solar source, this will raise a ValueError.

        Args:
            date (datetime-like): The date to estimate the sunrise and
                sunset, if a datetime object, should be at hour 0
        Returns:
            tuple[int, int]: The first is the hour of sunrise, and the second
                is the hour of sunset
        """

        if self.source_type != 'solar':
            raise ValueError("You can only estimate sunrise and sunset for "
                             "solar sources.")

        date = pd.Timestamp(date)
        historic_data = self.data
        # The range is 14 days ago to the end of yesterday
        start_date = date - datetime.timedelta(days=14)
        end_date = date - datetime.timedelta(hours=1)

        # We grab all hours where actual power is greater than 0
        relevant_data = historic_data[start_date:end_date]
        daylight_data = relevant_data[relevant_data['actuals'] > 0]

        # We do this to stop a warning from appearing, we know it's a copy
        daylight_data.is_copy = False
        daylight_data['hours'] = daylight_data.index.hour

        # Find the min and max hour for each day where we have positive
        # observed power generation.
        sunrises = daylight_data.groupby(daylight_data.index.date).min()['hours']
        sunsets = daylight_data.groupby(daylight_data.index.date).max()['hours']

        # We round in order to have an integer value for sunrise and sunset.
        average_sunrise = int(max(round(sunrises.mean()) - 1, 0))
        average_sunset = int(min(round(sunsets.mean()) + 1, 23))

        return average_sunrise, average_sunset

    def __repr__(self):
        return "Source({},{})".format(self.name, self.source_type)

    def __str__(self):
        string = "Source({},{}):\n".format(self.name, self.source_type)
        string += str(self.data.head()) + '\n'
        return string


class ExtendedSource(Source):
    """
    This is just a Source with information about segmentation criteria and
    a potential capacity.

    This overloads the __getattr__ method to look in the source_params
    dictionary for attributes if not explicitly defined.

    Attributes:
        criteria (List[Criterion]): The list of segmentation criteria
        bounds (dict[(datetime, datetime),float]): A dictionary mapping
            date ranges to an upper bound for that date range. None if
            there are to be no upper bounds
        diurnal_pattern (pd.Series): A timeseries specifying the diurnal
            pattern for solar sources. This should be specified only if the
            source type is 'solar'
        source_params (dict): Passed through keyword argument, these
            specify additional data relevant to a source.
    """
    def __init__(self, source, criteria=None, bounds=None,
                 diurnal_pattern=None, source_params=None):
        """
        Args:
            source (Source): A source object
            criteria (List[Criterion]): The list of segmentation criteria
            bounds (dict[(datetime, datetime),float]): A dictionary mapping
                date ranges to an upper bound for that date range. None if
                there are to be no upper bounds
            diurnal_pattern (pd.Series): A timeseries specifying the diurnal
                pattern for solar sources. This should be specified only if the
                source type is 'solar'
            source_params (dict): Passed through keyword argument, these
                specify additional data relevant to a source.
        """
        Source.__init__(self, source.name, source.data, source.source_type)

        if criteria is None:
            self.criteria = []
        else:
            self.criteria = criteria
        self.bounds = bounds

        if source.source_type != 'solar' and diurnal_pattern is not None:
            raise ValueError("The source type is not solar and the diurnal "
                             "pattern is specified.")
        self.diurnal_pattern = diurnal_pattern

        if source_params is None:
            self.source_params = {}
        else:
            self.source_params = source_params
        self._initialize_defaults()

        # We preserve these attributes to support code which uses them
        self.is_deterministic = self.source_params['is_deterministic']
        self.scaling_factor = self.source_params['scaling_factor']
        self.frac_nondispatch = self.source_params['frac_nondispatch']

    def _initialize_defaults(self):
        """
        This function will set all of the parameters in source_params to the
        default values if they are not set.
        """
        for key, value in defaults.items():
            if key not in self.source_params:
                self.source_params[key] = value


    def capacity(self, day):
        """
        This will return the capacity for a given day.

        Args:
            day (datetime-like): The day to get the capacity for
        Returns:
            float: The capacity, None if there is no capacity
        """
        # We calculate what the daily capacity is
        if self.bounds is not None:
            for (start_date, end_date), capacity in self.bounds.items():
                if start_date <= day <= end_date:
                    cap = capacity
                    break
            else:
                cap = None
        else:
            cap = None

        return cap

    def rolling_window(self, day, historic_data_start=None,
                       historic_data_end=None):
        """
        This will create an extended window from the extended source.
        The extended window will contain information about segmentation
        criteria and also an upper bound.

        Args:
            day (datetime-like): The date to make a rolling window from
            historic_data_start (datetime-like): The date to start considering
                historic data, None if to consider from beginning of data
            historic_data_end (datetime-like): The date for the end of the
                historic data, None if to consider to the end of the data
        Returns:
            ExtendedWindow: The rolling window of data with a capacity and
                a list of segmentation criteria
        """
        window = Source.rolling_window(self, day, historic_data_start,
                                       historic_data_end)

        cap = self.capacity(day)

        return ExtendedWindow(window, self.criteria, cap,
            self.diurnal_pattern, self.source_params)

    def estimate_sunrise_sunset(self, date, verbose=True):
        """
        This will estimate the hours of sunrise and sunset on the given date.
        If this source has a diurnal pattern specified, then it will take the
        hour of sunrise and sunset to be the first and last hour of the given
        day where there is positive diurnal pattern.

        Otherwise, it does this by taken the averages of the hours of first
        solar power generation and hours of last observed power generation
        for the two weeks before the given date. It then rounds the values.

        If this is not a solar source, this will raise a ValueError.

        Args:
            date (datetime-like): The date to estimate the sunrise and
                sunset, if a datetime object, should be at hour 0
            verbose (str): If set to False, will squelch warning that occurs
                when diurnal pattern is not found
        Returns:
            tuple[int, int]: The first is the hour of sunrise, and the second
                is the hour of sunset
        """
        if self.source_type != 'solar':
            raise ValueError("You can only estimate sunrise and sunset for "
                             "solar sources.")

        date = pd.Timestamp(date)

        if self.diurnal_pattern is None:
            if verbose:
                print("Warning: Source {} has no diurnal pattern, estimating "
                      "sunrise and sunset using average of past data."
                      .format(self.name), file=sys.stderr)
            return Source.estimate_sunrise_sunset(self, date)

        if verbose:
            print("{} {}: Using Diurnal Pattern to estimate sunrise and sunset"
                  .format(self.name, date.date()))

        diurnal_pattern = self.diurnal_pattern
        daily_pattern = diurnal_pattern[date:date+datetime.timedelta(hours=23)]

        sunrise, sunset = None, None

        # This will walk through finding first sun hour and first night hour
        for hour, pattern in enumerate(daily_pattern.values):
            if sunrise is None and pattern > 0:
                sunrise = hour

            # If sun has risen, and we have not found night and we reach a 0
            if sunrise is not None and sunset is None and pattern == 0:
                sunset = hour

        if sunrise is None and sunset is None:
            raise ValueError("No solar power was generated on {}".format(date))

        return sunrise, sunset

    def __repr__(self):
        return "ExtendedSource({},{})".format(self.name, self.source_type)

    def __str__(self):
        string = "ExtendedSource({},{}):\n".format(self.name, self.source_type)
        string += str(self.data.head()) + '\n'
        string += 'Criteria({})\n'.format(self.criteria)
        string += 'Bounds({})\n'.format(self.bounds)
        return string


class RollingWindow(Source):
    """
    This descendant of Source is a container for a historic dataframe
    and a dayahead dataframe. This sets the data attribute to the historic
    dataframe to enable use of parent methods to segment the historic dataframe

    Any of the parent class's methods which act on the self.data attribute
    will act on the self.historic_data attribute.

    Attributes:
        historic_data (pd.DataFrame): The historic data
        dayahead_data (pd.DataFrame): The dayahead data
        scenario_day (pd.Timestamp): The date of the dayahead data

    Args:
        name (str): the name of the source
        historic_dataframe (pd.DataFrame): The frame containing historic data
        source_type (str): the type of the source (e.g. 'solar')
        dayahead_dataframe (pd.DataFrame): The frame containing dayahead data
    """
    def __init__(self, name, historic_dataframe, source_type,
                 dayahead_dataframe):
        """
        Args:
            name (str): the name of the source
            historic_dataframe (pd.DataFrame): The frame containing historic data
            source_type (str): the type of the source (e.g. 'solar')
            dayahead_dataframe (pd.DataFrame): The frame containing dayahead data
        """
        Source.__init__(self, name, historic_dataframe, source_type)
        self.historic_data = self.data
        self.dayahead_data = dayahead_dataframe
        try:
            self.scenario_day = min(dayahead_dataframe.index)
        except:
            self.scenario_day = None

        self.number_of_hours = len(dayahead_dataframe.index)

    @property
    def all_data(self):
        """
        This will return the historic data and the dayahead data in a single
        frame. Note that any changes to this concatenated frame will not
        change either the historic or the dayahead frame.
        """
        return pd.concat([self.historic_data, self.dayahead_data])

    def segment(self, dt, criterion):
        """
        This method should segment the historic data and return a RollingWindow
        object which contains the data in the segment. Segmentation occurs
        according to predefined criteria in segmenter.py

        This is lifted to construct elements of the subclass should a subclass
        call this method. This will only work however, if the subclass has the
        same constructor interface.

        Args:
            dt (datetime.datetime): The datetime which we segment for
            criterion (Criterion): A criterion for segmentation defined in
                segmenter.py
        Returns:
            RollingWindow: The segment of data satisfying the criterion
        """

        if criterion.method == 'window':
            return self.segment_by_window(dt, criterion)
        elif criterion.method == 'enumerate':
            return self.segment_by_enumerate(dt, criterion)
        elif criterion.method == 'shape':
            return self.segment_by_shape(dt, criterion)
        else:
            raise ValueError("Unrecognized Criterion")

    def segment_by_window(self, dt, criterion):
        """
        This method finds the segment of data which is in a certain quantile
        window specified in criterion.window_size. If window_size is 0.4
        and the value is at the 0.3 quantile, this returns the window of data
        in the [0.1, 0.5] quantile range. If the quantile range is outside of
        [0,1] the window slides to get the amount of data requested.

        Args:
            dt (datetime.datetime): The datetime which we segment for
            criterion (Criterion): A criterion for segmentation defined in
                segmenter.py
        Returns:
            RollingWindow: The segment of data satisfying the criterion
        """
        data_to_segment_by = list(self.get_column(criterion.column_name))
        fitted_distr = UnivariateEmpiricalDistribution(data_to_segment_by)

        # day_ahead_value is the number which we wish to segment by
        day_ahead_value = self.get_dayahead_value(criterion.column_name, dt)

        window_size = criterion.window_size

        segmenter_data_cdf_val = fitted_distr.cdf(day_ahead_value)  # on 0,1
        if segmenter_data_cdf_val < window_size / 2:
            # Slide up window
            lower_cdf, upper_cdf = (0, window_size)
        elif segmenter_data_cdf_val > 1 - window_size / 2:
            # Slide down window
            lower_cdf, upper_cdf = (1 - window_size, 1)
        else:
            # Window fits in data
            lower_cdf, upper_cdf = (segmenter_data_cdf_val - window_size / 2,
                                    segmenter_data_cdf_val + window_size / 2)

        lower_bound, upper_bound = (fitted_distr.cdf_inverse(lower_cdf),
                                    fitted_distr.cdf_inverse(upper_cdf))
        segment = self.window(criterion.column_name, lower_bound, upper_bound)

        return RollingWindow(self.name, segment.data, self.source_type,
                             self.dayahead_data)

    def segment_by_enumerate(self, dt, criterion):
        """
        This method returns the segment of data whose column matches exactly
        the column at the specified dt.

        Args:
            dt (datetime.datetime): The datetime which we segment for
            criterion (Criterion): A criterion for segmentation defined in
                segmenter.py
        Returns:
            RollingWindow: The segment of data satisfying the criterion
        """
        column = criterion.column_name
        dayahead_value = self.get_dayahead_value(column, dt)
        segment = self.enumerate(column, dayahead_value)
        return RollingWindow(self.name, segment.data, self.source_type,
                             self.dayahead_data)

    def segment_by_shape(self, dt, criterion):
        """
        This will return the segment of data which have the exact same pattern
        as that specified in the column <column_name>_patterns at the dt.

        Args:
            dt (datetime.datetime): The datetime which we segment for
            criterion (Criterion): A criterion for segmentation defined in
                segmenter.py
        Returns:
            RollingWindow: The segment of data satisfying the criterion
        """
        enumerate_crit = Criterion(
            'patterns', criterion.column_name+'_derivatives_patterns_clusters',
            'enumerate')
        return self.segment_by_enumerate(dt, enumerate_crit)

    def segment_by_season(self, dt, winter = None, summer = None):
        """
        This function creates a RollingWindow containing only data from the
        same season like the segmentation dates season. For example, if you
        want to segment the data by season for a summer day, the function
        returns a RollingWindow containing only historic summer days.

        What exactly summer and winter means can be chosen by the user. The
        default values are:
        Winter: October to March
        Summer: April to September

        Args:
            dt (datetime.datetime): The datetime which we segment for.
            winter (list[int]): The months belonging to the winter season. If
                                nothing is provided, the default winter season
                                is used.
            summer (list[int]): The months belonging to the summer season. If
                                nothing is provided, the default summer season
                                is used.
        Returns:
            RollingWindow: Containing historic data only from the same season
                           like dt.
        """
        if winter == None:
            winter = [10, 11, 12, 1, 2, 3]
        if summer == None:
            summer = [4, 5, 6, 7, 8, 9]

        if dt.month in winter:
            ind = []
            for date in self.historic_data.index:
                if date.month in winter:
                    ind.append(date)
            segmented_data = self.historic_data.reindex(ind)
        else:
            ind = []
            for date in self.historic_data.index:
                if date.month in summer:
                    ind.append(date)
            segmented_data = self.historic_data.reindex(ind)

        return RollingWindow(self.name, segmented_data, self.source_type,
                             self.dayahead_data)

    def get_historic_value(self, column_name, row_index):
        """
        Get the historic value at column and row specified
        Args:
            column_name (str): The name of the column
            row_index (datetime.datetime): The datetime for which you want data
        """
        return self.get(column_name, row_index)

    def get_dayahead_value(self, column_name, row_index):
        """
        Get the dayahead value at the column and row specified
        Args:
            column_name (str): The name of the column
            row_index (datetime.datetime): The datetime for which you want data
        """
        return self.dayahead_data[column_name][row_index]

    def interpolate(self, column_name):
        """
        This function will interpolate the column specified so that every
        hour between the start of the data and the end of the data has a value.

        This will interpolate both historic data and dayahead data
        This function changes the state of the source.

        Args:
            column_name (str): name of the column to interpolate
        """
        sources.Source.interpolate(self, column_name)

        start_date = self.scenario_day
        end_date = self.scenario_day + datetime.timedelta(hours=23)

        date_range = pd.date_range(start_date, end_date, freq='H')
        self.dayahead_data = self.dayahead_data.reindex(date_range)
        column = self.dayahead_data[column_name]
        column = column.interpolate(limit_direction='both')
        self.dayahead_data[column_name] = column

    def solar_window(self):
        """
        This is a convenience function which produces a solar window from
        the existing rolling window. In essence, this purges all hours of data
        which occur at night.

        Returns:
            SolarWindow: A window without any night hours
        """
        return SolarWindow(self.name, self.historic_data, self.source_type,
                           self.dayahead_data)

    def scale(self, column_name, factor):
        """
        This function will scale the historic data and the dayahead data in
        column specified by column_name by the factor specified. This modifies
        the column in place.

        Args:
            column_name (str): The name of the column
            factor (float): The factor by which to scale the column
        """
        self.check_for_column(column_name)
        self.historic_data[column_name] *= factor
        self.dayahead_data[column_name] *= factor

    def __repr__(self):
        return "RollingWindow({},{},{})".format(self.name, self.source_type,
                                                self.scenario_day)

    def __str__(self):
        string = "RollingWindow({},{},{}):\n".format(
            self.name, self.source_type, self.scenario_day)
        string += 'Historic Data (First 5 rows):\n'
        string += str(self.data.head()) + '\n'
        string += 'Dayahead Data:\n'
        string += str(self.dayahead_data) + '\n'
        return string


class ExtendedWindow(RollingWindow):
    """
    This class exists for convenience and has more information regarding how
    to segment a particular source and what an upper bound on the installed
    capacity might be.

    It will contain a RollingWindow object as well as a list of Criteria
    for segmentation and a value for the upper bound

    Attributes:
        window (RollingWindow): The window object
        criteria (List[Criterion]): The list of segmentation criteria
        capacity (float): An upper bound on power production
        diurnal_pattern (pd.Series): A timeseries specifying the diurnal
            pattern for solar sources. This should be specified only if the
            source type is 'solar'
        source_params (dict): Passed through keyword argument, these
            specify additional data relevant to a source.
    """
    def __init__(self, window, criteria, capacity, diurnal_pattern=None,
                 source_params=None):
        RollingWindow.__init__(self, window.name, window.historic_data,
                               window.source_type, window.dayahead_data)
        self.criteria = criteria
        self.capacity = capacity
        self.diurnal_pattern = diurnal_pattern

        if source_params is None:
            self.source_params = {}
        else:
            self.source_params = source_params
        self._initialize_defaults()

        # We preserve these attributes to support code which uses them
        self.is_deterministic = self.source_params['is_deterministic']
        self.scaling_factor = self.source_params['scaling_factor']
        self.frac_nondispatch = self.source_params['frac_nondispatch']

    def _initialize_defaults(self):
        """
        This function will set all of the parameters in source_params to the
        default values if they are not set.
        """
        for key, value in defaults.items():
            if key not in self.source_params:
                self.source_params[key] = value

    def segment(self, dt):
        """
        This function uses the internally stored criteria to segment the
        window.

        Args:
            dt (datetime-like): The datetime for which you wish to segment by
        Returns:
            RollingWindow: The window of values which satisfy the criterion
        """
        window = self
        for criterion in self.criteria:
            window = RollingWindow.segment(window, dt, criterion)
        return ExtendedWindow(window, self.criteria, self.capacity,
                self.diurnal_pattern, self.source_params)

    def __repr__(self):
        return "ExtendedWindow({},{},{})".format(self.name, self.source_type,
                                                self.scenario_day)

    def __str__(self):
        string = "ExtendedWindow({},{},{}):\n".format(
            self.name, self.source_type, self.scenario_day)
        string += 'Historic Data (First 5 rows):\n'
        string += str(self.data.head()) + '\n'
        string += 'Dayahead Data:\n'
        string += str(self.dayahead_data) + '\n'
        string += 'Criteria({})\n'.format(self.criteria)
        string += 'Capacity({})\n'.format(self.capacity)
        return string


class WindowSet:
    """
    This class will provide convenience functions for accessing values from
    a collection of ExtendedWindows. It will expose a couple of necessary
    attributes the same across all such as the scenario day.

    It is assumed that the ExtendedWindows used to construct the WindowSet will
    all a rolling window for the same day.

    Attributes:
        scenario_day (pd.Timestamp): The day for which the rolling windows
            are set

    Args:
        windows (List[ExtendedWindow]): A list of ExtendedWindow objects
    """
    def __init__(self, windows):
        self.windows = windows

    def get_window_by_name(self, name):
        """
        This finds an ExtendedWindow which has the same name as the one passed
        in.

        Args:
            name (str): The name of the source
        Returns:
            (ExtendedWindow): The corresponding ExtendedWindow object
        """
        for window in self.windows:
            if window.name == name:
                return window
        else:
            raise ValueError("No source with that name.")

    def get_windows_by_type(self, source_type):
        """
        This returns all the windows with the same type as the one specified

        Args:
            source_type (str): The type of the source
        Returns:
            WindowSet: The list (possibly empty) of corresponding
                ExtendedWindows
        """
        return WindowSet([window for window in self.windows
                          if window.source_type == source_type])

    def get_power_windows(self):
        """
        This returns all windows which are of the source type 'wind' or 'solar'

        Returns:
            WindowSet: The list (possibly empty) of corresponding
                ExtendedWindows
        """
        return WindowSet([window for window in self.windows
                          if window.source_type in {'wind', 'solar'} and
                          not window.is_deterministic])

    def get_column_from_windows(self, column_name):
        """
        This constructs a dataframe composed of the data in each of the
        windows at the column specified. The column names will be the
        source names.

        Args:
            column_name (str): The name of the column
        Returns:
            pd.DataFrame: The frame with the column desired from each window
        """

        df_list = []

        for window in self.windows:
            df = window.historic_data

            # Get the respective data, rename the
            # column containing the errors.
            frame = df.loc[:, column_name].rename(window.name)
            df_list.append(frame)

        # Concatenate the data frames such that the resulting
        # dataframe only consists of dates that are
        # contained in all data frames.
        result_frame = pd.concat(df_list, axis=1, join='inner')
        return result_frame

    def __getitem__(self, index):
        return self.windows[index]

    def __iter__(self):
        return iter(self.windows)

    def __repr__(self):
        string = "WindowSet({})".format(self.windows)
        return string

    def __str__(self):
        string = "WindowSet:\n"
        for window in self.windows:
            string += '\t' + str(window)
        return string

    def __len__(self):
        return len(self.windows)
