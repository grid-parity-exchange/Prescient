#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
intervals.py

This module will export a class PredictionInterval which will encapsulate
some basic methods for handling intervals on power generation.
"""

from collections import Counter

class PredictionInterval:
    def __init__(self, center, lower, upper, alpha=None):
        """
        Args:
            center (float): The center of the interval (not necessarily
                halfway in between the lower and upper bounds)
            lower (float): The lower bound of the interval
            upper (float): The upper bound of the interval
            alpha (float): A value between 0 and 1 which should represent
                the proportion of data which is outside the interval
        """
        self.center = center
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.width = upper - lower

    def in_interval(self, x):
        """
        This method checks if a point is in an interval.

        Args:
            x (float): The point you are checking
        Returns:
            bool: True if x is in the interval
        """
        return lower <= x <= upper

    def location(self, x):
        """
        This method will return the location of the point x in the interval.
        If x is below the lower bound, this will be 'OUT_LEFT'
        If x is between the lower bound and the center, this will be 'IN_LEFT'
        If x is between the center and the upper bound, this will be 'IN_RIGHT'
        If x is above the upper bound, this will be 'OUT_RIGHT'

        Points right on the center will be put in the 'IN_RIGHT' category

        Args:
            x (float): the point in question
        Returns:
            str: A string specifying the location (see above)
        """
        if x < self.lower:
            return 'OUT_LEFT'
        elif x < self.center:
            return 'IN_LEFT'
        elif x <= self.upper:
            return 'IN_RIGHT'
        else:
            return 'OUT_UPPER'

    def evaluate(self, xs):
        """
        This function will return a summary of the performance of the interval
        when tested against the set of data points xs.

        It will return four values, the proportion out on the left, the
        proportion of the values between lower bound and center, the
        proportion of values between center and the upper bound, and
        the proportion above the upper bound.

        It will return these four values in order.

        Args:
            xs (list[float]): The collection of points to evaluate the
                prediction interval with
        Returns:
            list[float]: The proportions that are in each section as described
                above
        """
        locations = [self.location(x) for x in xs]
        counter = Counter(locations)
        return (counter['OUT_LEFT'], counter['IN_LEFT'], counter['IN_RIGHT'],
                counter['OUT_RIGHT'])

    def __repr__(self):
        return "PredictionInterval({},{},{})".format(self.lower, self.center,
                                                     self.upper)

    __str__ = __repr__
