#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
weather_state.py

This module exports the WeatherState class which is to be used
with Markov Chains to produce random walks for generating scenarios.
The WeatherState class itself does not have much beyond the baseclass
State, but it has attributes associated to power production as well 
as some methods for instantiating the state into its evaluated state.
"""
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from .states import State

class WeatherState(State):
    """
    This will be the class used for Markov Chains used in conjunction with
    creating power generation scenarios. 

    To instantiate a WeatherState, pass in an appropriate value
    as keyword arguments for error, forecast.
    derivative_pattern can be optionally passed as a 3-tuple in {-1,0,1}^3.
    The user may also pass in additional information as keyword arguments
    if those should be considered in the state information

    Args:
        errors: A tuple representing the interval the error is in
        forecasts: A tuple representing the interval the forecast is in
        derivative_pattern (tuple): A tuple representing the derivate pattern
    """
    def __init__(self, errors=None, forecasts=None,
                 derivative_pattern=None, **kwargs):
        self.errors = errors
        self.forecasts = forecasts
        self.derivative_pattern = derivative_pattern
        if derivative_pattern is None:
            State.__init__(self, errors=errors, forecast=forecasts, **kwargs)
        else:
            State.__init__(self, errors=errors, forecast=forecasts,
                           derivative_pattern=derivative_pattern, **kwargs)

    def uniform_sample(self):
        """
        This samples the values for the errors and forecasts assuming a
        uniform distribution on their corresponding intervals.
        
        Returns:
            dict: Dictionary representing a configuration
        """

        def sample(interval):
            return np.random.uniform(*interval)

        func_dict = {'errors': sample, 'forecasts': sample}

        if derivative_pattern is not None:
            func_dict['derivative_pattern'] =  lambda x: x
        return self.apply_function(func_dict)

    def sample_from_empirical(self, source):
        """
        This samples empirically from a data source between the quantiles
        specified in the error attribute as well as between the quantiles
        in the forecast attribute.

        Args:
            source (Source): The source to be pulled from.
        """
        def error_sample(bounds):
            lower, upper = source.get_quantiles('errors', bounds)
            return source.sample('errors', lower, upper)

        def forecast_sample(bounds):
            lower, upper = source.get_quantiles('forecasts', bounds)
            return source.sample('forecasts', lower, upper)

        func_dict = {'errors': error_sample, 'forecasts': forecast_sample}
        return self.apply_function(func_dict)
        

def plot_list_of_weather_states(states):
    """
    This is a helper function so that states can be just a list of states
    without needing any datetime information. It just pretends that they
    are in consecutive order an hour apart.

    Internally this constructs a dictionary of datetimes to states and
    calls plot_sequence_of_weather_states

    Args:   
        List[State]: A list of (presumably) consecutive states
    """
    
    # Arbitrary start_date
    start_date = pd.Timestamp('2000-01-01')
    date_dictionary = {}
    for i, state in enumerate(states):
        date_dictionary[start_date+datetime.timedelta(hours=i)] = state
    plot_sequence_of_weather_states(date_dictionary)


def plot_sequence_of_weather_states(states):
    """
    This function should plot the sequence of weather states by highlighting
    the region of the interval the state is in for both the forecast and
    the error. 

    Args:
        dict[datetime,WeatherState]: A dictionary mapping times to states
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    smallest_datetime = min(states.keys())

    timespan = (max(states.keys()) - smallest_datetime).components
    timespan = timespan.days * 24 + timespan.hours

    for dt, state in states.items():
        time_from_start = (dt - smallest_datetime).components
        time_from_start = time_from_start.days*24 + time_from_start.hours
        forecast_y = state.forecasts[0]
        forecast_height = state.forecasts[1] - state.forecasts[0]
        forecast_rectangle = patches.Rectangle((time_from_start, forecast_y), 1,
                                               forecast_height, color='blue',
                                               alpha=0.5)
        ax1.add_patch(forecast_rectangle)
        ax1.set_title('Forecasts')
        ax1.set_ylim(0, 5000)

        if hasattr(state, 'errors'):
            error_y  = state.errors[0]
            error_height = state.errors[1] - state.errors[0]

            error_rectangle = patches.Rectangle((time_from_start, error_y), 1,
                                                error_height, color='red',
                                                alpha=0.5)

            ax2.add_patch(error_rectangle)
            ax2.set_ylim(-1000, 1000)

        if hasattr(state, 'forecasts_derivatives'):
            lower, upper = state.forecasts_derivatives
            deriv_y = lower
            deriv_height = upper - lower

            deriv_rectangle = patches.Rectangle((time_from_start, deriv_y), 1,
                                                deriv_height, color='yellow',
                                                alpha=0.5)

            ax3.add_patch(deriv_rectangle)
            ax3.set_ylim(-1000, 1000)

    ax1.set_xlim(0, timespan)


def create_sample_scenario(state_walk, source):
    """
    This instantiates every state in the walk with values by sampling
    empirically from the historic data based on the bounds specified
    in each state.

    Args:
        state_walk (List[WeatherState]): A list of possible states
    Returns:
        List[{dict}]: A list of the dictionaries of each evaluated state
    """
    return [state.sample_from_empirical(source) for state in state_walk]


def plot_scenario(scenario_walk):
    """
    This plots the instantiated scenarios created in (for example)
    create_sample_scenario. This expects a list of dictionaries which have
    an 'errors' entry and a 'forecasts' entry.

    Args:
        scenario_walk (List[dict]): The instantiated scenario represented
            as a list of dictionaries for each hour of the day
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    forecasts = np.array(hour['forecasts'] for hour in scenario_walk)
    errors = np.array(hour['errors'] for hour in scenario_walk)

    ax1.plot(forecasts, color='blue')
    ax2.plot(errors, color='red')


def plot_corrected_scenario(scenario_walk):
    """
    This plots forecast+error for each day to correct the scenarios
    by the error.

    Args:
        scenario_walk (List[dict]): The instantiated scenario represented
            as a list of dictionaries for each hour of the day
    """
    fig, ax = plt.subplots()
    forecasts = np.array(hour['forecasts'] for hour in scenario_walk)
    errors = np.array(hour['errors'] for hour in scenario_walk)

    ax.plot(forecasts+errors, color='green')
