#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
descriptions.py

This module will export a bunch of commonly used state descriptions
for your convenience. This will be passed to get_state_walk of a Source object
to produce a sequence of states.
"""

from prescient.util.distributions.distributions import UnivariateEmpiricalDistribution
from gosm.markov_chains import states
from gosm.markov_chains.values import IntervalSampledValue
from gosm.markov_chains.values import QuantileSampledValue

def bin_description(bin_width, name):
    """
    This function will create a state description where the value
    given by name is described by an interval of width bin_width.

    The limits of the interval will always be multiples of the bin_width.

    If bin_width is 10, and the value under name is 12,
    then its state would be (10, 20).

    Args:
        bin_width (float): The desired width of the bin
        name (str): The name of the relevant value
    Returns:
        StateDescription: The bin description
    """

    def get_bin(value):
        floor_value = value - (value % bin_width)
        return floor_value, floor_value + bin_width

    return states.StateDescription(**{name:get_bin})


def sample_bin_description(bin_width, name, distribution):
    """
    This function will create a state description where the value
    given by name is described by an interval of width bin_width.

    The limits of the interval will always be multiples of the bin_width.

    If bin_width is 10, and the value under name is 12,
    then its state would be sampled from (10, 20).

    Evaluating this state will involve sampling from a
    distribution that is passed in.

    Args:
        bin_width (float): The desired width of the bin
        name (str): The name of the relevant value
        distribution (BaseDistribution): A distribution from which we sample
            conditioned on being in the bin of the state.
    Returns:
        StateDescription: The bin description
    """
    def get_bin_value(value):
        floor_value = value - (value % bin_width)
        return IntervalSampledValue(distribution, floor_value,
                                    floor_value + bin_width)
    
    return states.StateDescription(**{name:get_bin_value})


def quantile_description(quantiles, name, source):
    """
    This should determine which cutpoint range each point
    belongs to and designate that as its state.

    Example:
        If in our source, we have under error a value of 230 and
        we determine that this is between 0.1 and 0.2 quantile, then 
        the state will be State(error=(0.1, 0.2)).

    Args:
        quantiles (List[float]): This is the list of desired quantiles or
            cutpoints, e.g., [0.0, 0.1, 0.9, 1.0]
        name (str): The name of the relevant value in the Source
        source (Source): The Source of data, this is needed to compute the
            quantiles
    Returns:
        StateDescription: The quantile description
    """

    quants = source.get_quantiles(name, quantiles)

    def get_quantile(value):
        for i, (x1, x2) in enumerate(zip(quants, quants[1:])):
            if x1 <= value <= x2:
                return (quants[i], quants[i+1])

    return states.StateDescription(**{name:get_quantile})


def sample_quantile_description(quantiles, name, source, distribution):
    """
    This should determine which cutpoint range each point
    belongs to and designate this as the state. This will have the 
    QuantileSampledValue associated with it from which evaluating it will
    sample from the distribution passed in . The sample will have its
    quantile in the state's interval.

    Example:
        If in our source, we have under error a value of 230 and
        we determine that this is between 0.1 and 0.2 quantile, then 
        the state will be State(error=sample from (0.1, 0.2)).

    Args:
        quantiles (List[float]): This is the list of desired quantiles or
            cutpoints, e.g., [0.0, 0.1, 0.9, 1.0]
        name (str): The name of the relevant value in the Source
        source (Source): The Source of data, this is needed to compute the
            quantiles
        distribution (BaseDistribution): A distribution fitted to the values
            in the column of source
    Returns:
        StateDescription: The quantile description
    """
    quants = source.get_quantiles(name, quantiles)

    def get_quantile_value(value):
        for i, (x1, x2) in enumerate(zip(quants, quants[1:])):
            if x1 <= value <= x2:
                lower, upper = (quantiles[i], quantiles[i+1])
        return QuantileSampledValue(distribution, lower, upper)

    return states.StateDescription(**{name:get_quantile_value})


def identity_description(name):
    """
    For a certain value, the state of that value will be exactly that
    value.
    
    Example:
        If in our source, we have under derivative_pattern, we have (0,0,0),
        then the state will be State(derivative_pattern=(0,0,0))

    Args:
        name (str): The relevant value
    Returns:
        StateDescription: The identity description
    """
    def identity(x):
        return x

    return states.StateDescription(**{name:identity})
