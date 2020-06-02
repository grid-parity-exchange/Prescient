#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
values.py

This module exports a Value baseclass as well as a collection 
of classes which inherit from Value. This class is meant to be
an abstraction of a computation in order to delay evaluation for
use in States. We do this to have a method of turning more general States
into instantiated and evaluated events.

For example, a State can represent any event where the forecast lands
between the 20th percentile and the 80th percentile of a set of historic
data. We represent this in the following way:
    
    middle_forecast = State(forecast=SampledValue(distribution))

In this example, we represent the state as any event in the 20th and 80th
percentile and the method to generate a specific state as sampling from
a distribution. We could use this state to generate many different events
involving sampling from the distribution.

To convert a state into its corresponding values, we call the evaluate
method of the state. This returns a dictionary with the configuration
of the event.

A possible event might be as follows:
    middle_forecast.evaluate()
    -> {'forecast': 0.45232}
"""

import abc

class Value(metaclass=abc.ABCMeta):
    """
    This class will be at the heart of delayed evaluation needed
    for converting State objects into its equivalent evaluated state.

    To this end, every descendent of Value must have a corresponding
    evaluate method.
    """

    @abc.abstractmethod
    def evaluate():
        pass


class SampledValue(Value):
    """
    This value represents a sampled value from a distribution.
    To use this, a user should pass in a distribution with a sample
    method.

    Args:
        distribution: An instantiated Distribution object
    """
    def __init__(self, distribution):
        self.distribution = distribution

    def evaluate(self):
        return self.distribution.sample()

    def __str__(self):
        return "SampledValue({})".format(self.distribution)


class IntervalSampledValue(Value):
    """
    This value represents a value to be sampled from the distribution 
    conditioned on being in the interval. This relies on the sample_on_interval
    method of a distribution.
    """
    def __init__(self, distribution, a, b):
        """
        To instantiate this Value, one must pass in an instantiated
        distribution object. From this distribution, it will sample a value
        on the interval between a and b.

        Args:
            distribution: An instantiated UnivariateDistribution
            a (float): The lower quantile
            b (float): The upper quantile
        """
        self.distribution = distribution
        self.a = a
        self.b = b

    def evaluate(self):
        return self.distribution.sample_on_interval(self.a, self.b)

    def __hash__(self):
        return hash((self.distribution, self.a, self.b))

    def __lt__(self, other):
        return self.a < other.a

    def __eq__(self, other):
        return (self.distribution == other.distribution and 
                self.a == other.a and self.b == other.b)

    def __str__(self):
        return "IntervalSampledValue({},{},{})".format(self.distribution,
                                                       self.a, self.b)

class QuantileSampledValue(Value):
    """
    This value represents a value to be sampled between quantiles of the
    distribution. This relies on the sample_between_quantiles method of a
    distribution.
    """
    def __init__(self, distribution, a, b):
        """
        To instantiate this Value, one must pass in an instantiated
        distribution object. From this distribution, it will sample a value
        with quantile between a and b.

        Args:
            distribution: An instantiated UnivariateDistribution
            a (float): The lower quantile
            b (float): The upper quantile
        """
        self.distribution = distribution
        self.a = a
        self.b = b

    def evaluate(self):
        return self.distribution.sample_between_quantiles(self.a, self.b)

    def __hash__(self):
        return hash((self.distribution, self.a, self.b))

    def __lt__(self, other):
        return self.a < other.a

    def __eq__(self, other):
        return (self.distribution == other.distribution and
                self.a == other.a and self.b == other.b)

    def __str__(self):
        return "QuantileSampledValue({},{},{})".format(self.distribution,
                                                       self.a, self.b)


class ConditionalExpectedValue(Value):
    """
    This value represents taking the expected value of a distribution
    conditioned on being in a passed in region

    For now the region will be a interval contained in [0,1],
    Then we will apply the inverse cdf to the bounds and compute the
    conditional expectation on [F^-1(a), F^-1(b)]
    Args:
        distribution: An instantiated Distribution object
        region: A 2-tuple like (a,b) for now (maybe we can handle higher
                dimensions some day)
    """
    def __init__(self, distribution, region):
        self.distribution = distribution
        self.region = region

    def evaluate(self):
        a, b = self.region
        c = self.distribution.cdf_inverse(a)
        d = self.distribution.cdf_inverse(b)
        return self.distribution.region_expectation((c, d)) / (b - a)

    def __str__(self):
        return "ConditionalExpectedValue({},{})".format(self.distribution,
                                                        self.region)


class FunctionValue(Value):
    """
    This value represents the result of a function applied to arguments.

    Args:
        func: A function
        pargs: List of positional arguments passed to function
        kwargs: dictionary of keyword arguments
    """
    def __init__(self, func, pargs=None, kwargs=None):
        self.func = func
        if pargs is None:
            self.pargs = []
        else:
            self.pargs = pargs

        if kwargs is None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs

    def evaluate(self):
        return self.func(*self.pargs, **self.kwargs)
