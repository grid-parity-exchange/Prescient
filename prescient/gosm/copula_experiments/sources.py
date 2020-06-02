#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module initializes the sources of Uncertainty.
It was copied from PINT on 03/22/2017.
"""

import gosm_options
import math
import segmenter
import datetime as dt
import pandas as pd


class Source:
    """
    This class is a container for all the data realted to one single source of uncertainty.

    Attributes:
        name (str): the name of the source
        actuals_filename (str): the name of the file containing the actual data
        forecasts_filename (str): the name of the file containing the forecast data
        history_dataframe (pd.DataFrame): the historic data (forecasts and actuals)
        day_ahead_dataframe (pd.DataFrame): the data of the scenario day (forecasts and actuals, if available)
        source_type (str): the type of the source (e.g. 'solar')
        segment_filename (str): the name of the file containing the segmenting rules
        preprocessor (bool): True, if the source was created for the preprocessor, False, otherwise
        average_sunrise_per_month (dict[(int, int): int]): a dictionary like {(year, month) -> sunrise} to save
                                                           the average hour of sunrise for every month
        average_sunset_per_month (dict[(int, int): int]): a dictionary like {(year, month) -> sunset} to save
                                                          the average hour of sunset for every month
    """

    def __init__(self, name, actuals_filename, forecasts_filename, actuals_dataframe, forecasts_dataframe,
                 source_type, segment_filename, preprocessor):
        self.name = name
        self.actuals_filename = actuals_filename
        self.forecasts_filename = forecasts_filename
        self.source_type = source_type
        self.segment_filename = segment_filename

        # Create the history and day ahead dataframes.
        self.history_dataframe = self._create_history_dataframe(actuals_dataframe, forecasts_dataframe)
        self.day_ahead_dataframe = self._create_day_ahead_dataframe(actuals_dataframe, forecasts_dataframe)

        # If the source is of type 'solar', only use the data between sunrise and sunset.
        # If we are using the preprocessor, we do not want to discard data outside the sunshine hours.
        if source_type == 'solar' and not preprocessor:
            # Estimate sunrise and sunset for every day of the data.
            from skeleton_scenario import print_progress
            print_progress('Estimating sunrise and sunset.', self.name)
            self._estimate_sunrise_sunset()
            self._discard_non_sunshine_data()
            print_progress('The data before sunrise and after sunset has been discarded.', self.name)

        return

    @staticmethod
    def _create_history_dataframe(actuals_df, forecasts_df):
        """
        Creates one dataframe that contains the historic data of both forecasts and actuals.

        Args:
            actuals_df (pd.DataFrame): the actual data 
            forecasts_df (pd.DataFrame): the forecast data

        Returns:
            pd.DataFrame: the joined dataframe
        """

        history_dataframe = pd.concat([actuals_df, forecasts_df], axis=1, join='inner')

        # Select the dates specified by the user.
        if gosm_options.historic_data_start is not None:
            first_day = gosm_options.historic_data_start + dt.timedelta(hours=0)
        else:
            first_day = history_dataframe.index[0]
        if gosm_options.historic_data_end is not None:
            last_day = gosm_options.historic_data_end + dt.timedelta(hours=23)
        else:
            # last_day = gosm_options.scenario_day - dt.timedelta(days=1) + dt.timedelta(hours=23)
            last_day = gosm_options.scenario_day - dt.timedelta(hours=1)
        history_dataframe = history_dataframe.loc[first_day: last_day]

        # Compute the errors.
        history_dataframe = history_dataframe.assign(errors=history_dataframe['forecasts']
                                                            - history_dataframe['actuals'])

        return history_dataframe

    @staticmethod
    def _create_day_ahead_dataframe(actuals_df, forecasts_df):
        """
        Creates a data frame for the forecast of the current day.

        Args:
            actuals_df (pd.DataFrame): the actual data 
            forecasts_df (pd.DataFrame): the forecast data

        Returns:
            pd.DataFrame: the resulting dataframe
        """

        day_ahead_df = pd.concat([actuals_df, forecasts_df], axis=1, join='outer')

        return day_ahead_df[gosm_options.scenario_day: gosm_options.scenario_day + dt.timedelta(hours=23)]

    def _estimate_sunrise_sunset(self):
        """
        Estimates the hours of sunrise and sunset as a monthly average for all days of the actuals data frame. 
        """

        df = pd.concat([self.history_dataframe, self.day_ahead_dataframe])
        dates = pd.date_range(df.index.date[0], gosm_options.scenario_day, freq='D')

        # Dictionaries to save the average sunrise/sunset per month.
        self.average_sunrise_per_month = {(date.year, date.month): 0 for date in dates}
        self.average_sunset_per_month = {(date.year, date.month): 0 for date in dates}

        # Estimate sunrise for every month.
        for (year, month) in self.average_sunrise_per_month.keys():
            relevant_dates = [date for date in dates if date.year == year and date.month == month]
            len_relevant_dates = len(relevant_dates)
            for date in relevant_dates:
                for hour in range(0, 24):
                    try:
                        average_power = (df.loc[date + dt.timedelta(hours=hour)]['actuals']
                                         + df.loc[date + dt.timedelta(hours=hour)]['forecasts']) / 2
                        if average_power > gosm_options.power_level_sunrise_sunset:
                            self.average_sunrise_per_month[(year, month)] += max(hour - 1, 0) / len_relevant_dates
                            break
                    except KeyError:
                        continue
            # Round down to the next smaller integer.
            self.average_sunrise_per_month[(year, month)] = math.floor(self.average_sunrise_per_month[(year, month)])

        # Estimate sunset for every month.
        for (year, month) in self.average_sunset_per_month.keys():
            relevant_dates = [date for date in dates if date.year == year and date.month == month]
            len_relevant_dates = len(relevant_dates)
            for date in relevant_dates:
                for i in range(0, 24):
                    hour = 23 - i
                    try:
                        average_power = (df.loc[date + dt.timedelta(hours=hour)]['actuals']
                                         + df.loc[date + dt.timedelta(hours=hour)]['forecasts']) / 2
                        if average_power > gosm_options.power_level_sunrise_sunset:
                            self.average_sunset_per_month[(year, month)] += min(hour + 1, 23) / len_relevant_dates
                            break
                    except KeyError:
                        continue
            # Round up to the next bigger integer.
            self.average_sunset_per_month[(year, month)] = math.ceil(self.average_sunset_per_month[(year, month)])

        return

    def _discard_non_sunshine_data(self):
        """
        Discards all the data that doesn't lie between the estimated hours of sunrise and sunset.
        """

        df = self.history_dataframe
        sunshine_frames = []
        average_power = 0
        for (year, month) in self.average_sunrise_per_month.keys():
            sunrise = self.average_sunrise_per_month[(year, month)]
            sunset = self.average_sunset_per_month[(year, month)]

            df_sunshine = df.loc[(df.index.year == year) & (df.index.month == month) &
                                 (df.index.hour >= sunrise) & (df.index.hour <= sunset)]
            df_no_sunshine = df.loc[(df.index.year == year) & (df.index.month == month) &
                                    ((df.index.hour < sunrise) | (df.index.hour > sunset))]

            # Add the average power outside the sunshine hours.
            power_sum = df_no_sunshine.sum()
            average_power += power_sum / df_no_sunshine.shape[0]

            sunshine_frames.append(df_sunshine)

        average_power /= len(self.average_sunrise_per_month.keys())
        from skeleton_scenario import print_progress
        print_progress('Warning: The average power level of the actual data outside the sunshine hours '
                       'is {0:.4} per day.'.format(average_power['actuals']), self.name)
        print_progress('Warning: The average power level of the forecast data outside the sunshine hours '
                       'is {0:.4} per day.'.format(average_power['forecasts']), self.name)

        self.history_dataframe = pd.concat(sunshine_frames).sort_index()

        return

    def get_sunrise_sunset(self, date):
        """
        Returns the hours of sunrise and sunset for the specified date.

        Args:
            date (datetime): the date 

        Returns:
            tuple[int, int]: sunrise and sunset
        """

        if self.source_type == 'solar':
            return (self.average_sunrise_per_month[(date.year, date.month)],
                    self.average_sunset_per_month[(date.year, date.month)])

        return None


class DataSources:
    """
    This class is a container for all the data that is passed in from the user.
    We would like this to be as general as possible.
    It should support indexing by the date and return all pertinent
    information regarding that date.

    Required Attributes:
        sources_dataframe # load, wind, or solar
        source_names # name of all the sources, should be unique
        ndim # number of sources

    Required Methods:
        segment(segment_filename) # should return a frame segmented by the criteria in filename
        get_date(datetime) # return a dataframe

    """

    def __init__(self, preprocessor=False):
        """
        Initialies an instance of the DataSources class.

        Args:
            preprocessor (bool): Must be True, if the data sources are used for the preprocessor. In this case,
                                 we do not want to discard data outside the sunshine hours.
        """

        print('Reading data.')

        self.preprocessor = preprocessor
        self.sources = []
        self.source_names = []
        self._read_source_file()
        history_dfs = [source.history_dataframe for source in self.sources]
        day_ahead_dfs = [source.day_ahead_dataframe for source in self.sources]
        self.ndim = len(self.sources)

        #  We merge all the dataframe if there are multiple ones
        if len(self.sources) > 1:
            self.actuals_dataframe = self._merge_data_frames(history_dfs, self.source_names)
            self.forecasts_dataframe = self._merge_data_frames(day_ahead_dfs, self.source_names)
        else:
            self.actuals_dataframe = history_dfs[0]
            self.forecasts_dataframe = day_ahead_dfs[0]

        if self.actuals_dataframe.empty:
            print("WARNING: The actuals dataframe is empty.")
            print("Please make sure that your data files have actually data contained in them.")
            print("This may be due to there being malformed data or no overlap in the data files datetimes.")

        if self.forecasts_dataframe.empty:
            print("WARNING: The day ahead dataframe is empty.")
            print("Please make sure that your data files have actually data contained in them.")
            print("This may be due to there being malformed data or no overlap in the data files datetimes.")

        return

    def _read_source_file(self):
        """
        Parses the associated sources file and updates the relevant attributes
        of the instance. In particular, it sets the load frame if there is one
        for actuals and predicted as well as the dataframes for the energy sources.
        """

        with open(gosm_options.sources_file) as f:
            for line in f:
                comment_start = line.find('#')  # gobble up any comments
                if comment_start != -1:
                    line = line[:comment_start]
                if line.strip():  # Line is not empty

                    line_strip = [string.strip() for string in line.split(',')]
                    source_name, actuals_name, forecasts_name, source_type, segment_criteria = line_strip[:5]

                    # Read the data.
                    actuals_df = self._read_data_file(actuals_name, source_type, 'actuals')
                    forecasts_df = self._read_data_file(forecasts_name, source_type, 'forecasts')

                    if source_type in {'load', 'demand'}:
                        self.load_source = Source(source_name, actuals_name, forecasts_name, actuals_df, forecasts_df,
                                                  source_type, segment_criteria, self.preprocessor)
                        self.load_name = source_name
                    else:
                        self.sources.append(Source(source_name, actuals_name, forecasts_name, actuals_df, forecasts_df,
                                                   source_type, segment_criteria, self.preprocessor))
                        self.source_names.append(source_name)
        return

    @staticmethod
    def _read_data_file(filename, source_type, data_type):
        """
        Reads a data file, scales the values according to the type of the source and calculates the errors.
        The result is returned as a pd.DataFrame. In the case of a solar source, only the data between sunrise
        and sunset is kept.

        Args:
            filename (str): the name of the file 
            source_type (str): the type of the source 
            data_type (str): the type of the data (i.e. 'forecasts' or 'actuals')

        Returns:
            (pd.DataFrame): the resulting data
        """

        assert filename.endswith('.csv')
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        if df.index.name != 'datetimes':  # This means we are passed in data w/o headers
            df.loc[df.index.name] = df.columns  # Must restore first row
            df.index.name = 'datetimes'
            if data_type == 'forecasts':
                df.columns = ['forecasts']  # Assume that the first column contains forecasts.
            elif data_type == 'actuals':
                df.columns = ['actuals']  # Assume that the first column contains actuals.
            df = df.apply(pd.to_numeric)
        df = df[[data_type]]

        # Get the scaling factor.
        scaling_factor = 1
        if source_type == 'solar':
            scaling_factor = gosm_options.solar_scaling_factor
        elif source_type == 'wind':
            scaling_factor = gosm_options.wind_scaling_factor
        elif source_type in {'load', 'demand'}:
            scaling_factor = gosm_options.load_scaling_factor
        else:
            print('Warning: No scaling factor was provided for sources of type \'{}\'.'.format(source_type))

        # Scale the values.
        df = df.multiply(scaling_factor)

        # Discard rows with at least one missing value.
        df = df.dropna()

        return df

    @staticmethod
    def _merge_data_frames(frames, source_names):
        """
        This should merge the frames so that the resulting frame only has dates
        which are contained in all the frames.
        Args:
            frames: A list of dataframes
            source_names: A list of strings referring to names of sources
        Returns:
            The merged dataframe
        """

        return pd.concat(frames, join='inner', axis=1,
                         keys=source_names, names=['sources', 'datetimes'])

    def segment(self, datetime, source_name=None):
        """
        Segments each of the individual data sources (or only one) by the specified datetime and
        then merges each of the data sources into a single frame that is returned.

        Args:
            datetime (pd.datetime): the specific datetime
            source_name (str): the name of the source (if only one is to be segmented)

        Returns:
            pd.DataFrame: the resulting data frame
        """

        segmented_dfs = []
        for source in self.sources:
            if source_name is not None and source.name == source_name:
                segmented_frame = segmenter.OuterSegmenter(source.history_dataframe, source.day_ahead_dataframe,
                                                           source.segment_filename, datetime).retval_dataframe()
                return segmented_frame
            elif source_name is None:
                segmented_frame = segmenter.OuterSegmenter(source.history_dataframe, source.day_ahead_dataframe,
                                                           source.segment_filename, datetime).retval_dataframe()
                segmented_dfs.append(segmented_frame)

        if len(self.sources) == 1:
            return segmented_dfs[0]
        else:
            return self._merge_data_frames(segmented_dfs, self.source_names)

    def segment_load(self, datetime):
        """
        Segments data of the load source by the specified datetime.

        Args:
            datetime (pd.datetime): the specific datetime

        Returns:
            pd.DataFrame: the resulting data frame
        """

        load_source = self.load_source
        segmented_frame = segmenter.OuterSegmenter(load_source.history_dataframe, load_source.day_ahead_dataframe,
                                                   load_source.segment_filename, datetime).retval_dataframe()
        return segmented_frame

    def as_dict(self):
        """
        Converts the merged dataframe into an equivalent dictionary of dictionaries
        The dictionary should be of the form
        {(source, field) -> {datetime -> value}}

        If there is only one source is will just be {field -> {datetime -> value}}
        This will return two dictionaries, a actuals and a dayahead
        """

        return self.actuals_dataframe.to_dict(), self.forecasts_dataframe.to_dict()

    def get_source(self, source_name):
        """
        Returns the Source object with the given name.

        Args:
            source_name (str): the source's name 

        Returns:
            Source: the desired source
        """

        for source in self.sources:
            if source.name == source_name:
                return source
        raise RuntimeError('No source called \'{}\' could be found.'.format(source_name))

    def get_sunrise_sunset(self, date):
        """
        Returns the estimated hour of sunrise and sunset.

        Args:
            date (dt.datetime): the date of which to get sunrise and sunset

        Returns:
            tuple[int, int]: the hours of sunrise and sunset
        """

        solar_sources = [source for source in self.sources if source.source_type == 'solar']
        if not solar_sources:
            return (None, None)

        # If there are multiple solar sources, use the earliest sunrise and the latest sunset.
        sunrise = 23
        sunset = 0
        for source in solar_sources:
            if (date.year, date.month) in source.average_sunrise_per_month:
                sunrise = min(sunrise, source.average_sunrise_per_month[(date.year, date.month)])
            if (date.year, date.month) in source.average_sunset_per_month:
                sunset = max(sunset, source.average_sunset_per_month[(date.year, date.month)])

        # Check if sunrise and sunset were found at all.
        if sunrise == 23:
            sunrise = None
        if sunset == 0:
            sunset = None

        from skeleton_scenario import print_progress
        print_progress('The hour of sunrise was estimated to be {}.'.format(sunrise))
        print_progress('The hour of sunset was estimated to be {}.'.format(sunset))

        return (sunrise, sunset)
