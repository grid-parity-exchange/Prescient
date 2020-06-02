#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This abstract base class is the parent class of all distribution classes.
"""
from abc import ABCMeta, abstractmethod
from functools import wraps
import collections
import random
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from .utilities import memoize_method

class BaseDistribution(object):
    __metaclass__ = ABCMeta

    # --------------------------------------------------------------------
    # Abstract methods (have to be implemented within the subclass)
    # --------------------------------------------------------------------

    @abstractmethod
    def __init__(self, dimension=0):
        """
        Initializes the distribution.

        Args:
            dimension (int): the dimension of the distribution
            input_data: dict, OrderedDict or list of input data
        """
        self.name = self.__class__.__name__
        self.dimension = dimension

        # Lower and upper bound of the considered data
        self.alpha = None
        self.beta = None

    @abstractmethod
    def pdf(self, x):
        """
        Evaluates the probability density function at a given point x.

        Args:
            x (float): the point at which the pdf is to be evaluated

        Returns:
            float: the value of the pdf
        """
        pass

    @classmethod
    @abstractmethod
    def fit(cls, data):
        """
        This function will fit the distribution of this class to the passed
        in data. This will return an instance of the class.

        Args:
            data (List[float]): The data the distribution is to be fit to
        Returns:
            BaseDistribution: The fitted distribution
        """
        pass

    @staticmethod
    def seed_reset(seed=None):
        """
        Resets the random seed for sampling.
        If no argument is passed, the current time is used.

        Args:
            seed: the random seed
        """
        random.seed(seed)


class UnivariateDistribution(BaseDistribution):
    __metaclass__ = ABCMeta

    def __init__(self):
        BaseDistribution.__init__(self, dimension=1)

    def plot(self, plot_pdf=True, plot_cdf=True, output_file=None, title=None,
             xlabel=None, ylabel=None, output_directory='.'):
        """
        Plots the pdf/cdf within the interval [alpha, beta].
        If required, the input data is added as a scatter plot, where the size
        of the data points is correlated to their respective number of
        occurrences. If no output file is specified, the plots are shown at
        runtime.

        Args:
            plot_pdf (bool): True if the plot should include the pdf
            plot_cdf (bool): True if the plot should include the cdf
            output_file (str): name of an output file to save the plot
            title (str): the title of the plot
            xlabel (str): the name of the x-axis
            ylabel (str): the name of the y-axis
            output_directory (str): The name of the directory to save the
                files, defaults to the current working directory
        """
        if plot_pdf == 0 and plot_cdf == 0:
            print('Error: The print method was called, but no functions '
                  'were supposed to be plotted.')
            return

        directory = output_directory
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

        x_range = np.linspace(self.alpha, self.beta, 100)
        fig = plt.figure()

        # Plot the pdf if required.
        if plot_pdf:
            y_range = []
            for x in x_range:
                y_range.append(self.pdf(x))
            plt.plot(x_range, y_range, label='PDF', color='blue')

        # Plot the cdf if required.
        if plot_cdf:
            y_range = []
            for x in x_range:
                y_range.append(self.cdf(x))
            plt.plot(x_range, y_range, label='CDF', color='red')

        # Display a legend.
        lgd = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                         ncol=3, shadow=True)

        # Display a grid and the axes.
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')

        # Name the axes.
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.title(title, y=1.08)

        if output_file is None:
            # Display the plot.
            plt.show()
        else:
            # Save the plot.
            plt.savefig(directory + os.sep + output_file,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.close(fig)

    @memoize_method
    def cdf(self, x, epsabs=1e-4):
        """
        Evaluates the cumulative distribution function at a given point x.

        Args:
            x (float): the point at which the cdf is to be evaluated
            epsabs (float): The accuracy to which the cdf is to be calculated

        Returns:
            float: the value of the cdf
        """
        if x <= self.alpha:
            return 0
        elif x >= self.beta:
            return 1
        else:
            return integrate.quad(self.pdf, self.alpha, x, epsabs=epsabs)[0]

    @memoize_method
    def cdf_inverse(self, x, cdf_inverse_tolerance=1e-4,
                    cdf_inverse_max_refinements=10,
                    cdf_tolerance=1e-4):
        """
        Evaluates the inverse cumulative distribution function at a given
        point x.

        TODO: Explain better how this is calculated

        Args:
            x (float): the point at which the inverse cdf is to be evaluated
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        Returns:
            float: the value of the inverse cdf
        """

        # For ease in calculating the cdf, we define this temp function.
        cdf = lambda x: self.cdf(x, epsabs=cdf_tolerance)

        # This method calculates the cdf of start and then increases
        # (if the cdf value is less than or equal x) or decreases
        # (if the cdf value is greater than x) start iteratively by one
        # stepsize until x is passed. It returns the increased (or decreased)
        # start value and its cdf value.
        def approximate_inverse_value(start):
            cdf_val = cdf(start)
            if x >= cdf_val:
                while x >= cdf_val:
                    start += stepsize
                    cdf_val = cdf(start)
            else:
                while x <= cdf_val:
                    start -= stepsize
                    cdf_val = cdf(start)
            return cdf_val, start

        # Handle some special cases.
        if x < 0 or x > 1:
            return None
        elif abs(x) <= cdf_inverse_tolerance:
            return self.alpha
        elif abs(x-1) <= cdf_inverse_tolerance:
            return self.beta
        else:

            # Initialize variables.
            approx_x = 0
            result = None
            number_of_refinement = 0

            # The starting stepsize was chosen arbitrarily.
            stepsize = (self.beta - self.alpha)/10

            while abs(approx_x - x) > cdf_inverse_tolerance \
                    and number_of_refinement <= cdf_inverse_max_refinements:

                # If this is the first iteration, start at one of the bounds
                # of the domain.
                if number_of_refinement == 0:

                    # If x is greater than or equal 0.5, start the
                    # approximation at the upper bound of the domain.
                    if x >= 0.5:
                        approx_x, result = approximate_inverse_value(self.beta)

                    # If x is less than 0.5, start the approximation at
                    # the lower bound of the domain.
                    else:
                        approx_x, result = approximate_inverse_value(
                            self.alpha)
                else:

                    # If this is not the first iteration, halve the stepsize
                    # and call the approximation method.
                    stepsize /= 2
                    approx_x, result = approximate_inverse_value(result)

                number_of_refinement += 1

            return result

    def mean(self):
        """
        Computes the mean value (expectation) of the distribution.

        Returns:
            float: the mean value
        """

        # Use region_expectation to compute the mean value.
        return self.region_expectation((self.alpha, self.beta))

    @memoize_method
    def region_expectation(self, region):
        """
        Computes the mean value (expectation) of a specified region.

        Args:
            region: the region (tuple of dimension 2) of which the expectation
                is to be computed

        Returns:
            float: the expectation
        """

        # Check whether region is a tuple of dimension 2.
        if isinstance(region, tuple) and len(region) == 2:
            a, b = region
            if a > b:
                raise ValueError("Error: The upper bound of 'region' can't be "
                                 "less than the lower bound.")
        else:
            raise TypeError("Error: Parameter 'region' must be a tuple of "
                            "dimension 2.")

        integral, _ = integrate.quad(lambda x: x * self.pdf(x), a, b)

        return integral

    @memoize_method
    def region_probability(self, region):
        """
        Computes the probability of a specified region.

        Args:
            region: the region of which the probability is to be computed

        Returns:
            float: the probability
        """

        # Compute the region's probability by integration,

        # Check whether region is a tuple of dimension 2.
        if isinstance(region, tuple) and len(region) == 2:
            a, b = region
            integral, _ = integrate.quad(self.pdf, a, b)
        else:
            raise ValueError("Error: Parameter 'region' must be a tuple of"
                             " dimension 2.")

        return integral

    def conditional_expectation(self, interval, cdf_inverse_tolerance=1e-4,
                    cdf_inverse_max_refinements=10,
                    cdf_tolerance=1e-4):
        """
        This computes the conditional expectation of the distribution
        conditioned on being in the hyperrectangle passed in.
        The hyperrectangle will actually for this be just an interval contained
        in [0, 1] potentially with some cutouts. This will work for
        1-dimensional hyperrectangles, the multivariate distribution subclass
        should implement a different version of this.

        If the region is (a, b), this will compute the expectation on
        [cdf^-1(a), cdf^-1(b)] and divide it by (b-a).

        Args:
            Interval (Interval): An interval on which
                the conditional expectation is to be computed on
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        """
        a, b = interval.a, interval.b
        cdf_inverse = lambda x: self.cdf_inverse(x, cdf_inverse_tolerance,
                                                 cdf_inverse_max_refinements,
                                                 cdf_tolerance)

        lower, upper = cdf_inverse(a), cdf_inverse(b)
        expectation = self.region_expectation((lower, upper))
        probability = b-a

        # A hyperrectangle may subtract some intervals from the larger interval
        if hasattr(interval, 'cutouts'):
            for cutout in interval.cutouts:
                a, b = cutout.a, cutout.b
                lower, upper = cdf_inverse(a), cdf_inverse(b)
                expectation -= self.region_expectation((lower, upper))
                probability -= b-a
        return expectation / probability

    def sample_one(self):
        """
        Returns a single sample of the distribution

        Returns:
            float: the sample
        """

        return self.cdf_inverse(np.random.uniform())

    def sample_on_interval(self, a, b):
        """
        This samples from the distribution conditioned on X being in [a, b].
        This does this by sampling uniformly on [F(a), F(b)] and then applying
        the inverse transform to the result.

        Args:
            a (float): The lower limit of the interval
            b (float): The upper limit of the interval
        Returns:
            float: The sampled value in the interval
        """
        return self.sample_between_quantiles(self.cdf(a), self.cdf(b))

    def sample_between_quantiles(self, a, b):
        """
        This samples from the distribution conditioned on the quantile of the
        point being between a and b, i.e., it generates X given that
        a < F(X) < b. It does this by sampling from a uniform distribution on
        (a,b) and then applying the inverse transform to the point.

        Args:
            a (float): The lower quantile, must be between 0 and 1.
            b (float): The upper quantile, must be between 0 and 1.
        Returns:
            float: The sampled value
        """
        y = np.random.uniform(a, b)
        return self.cdf_inverse(y)

    def log_likelihood(self, data):
        """
        This method will return the log likelihood of the observed data
        given the fitted model.

        Args:
            data (list[float]): A list of observed values
        Returns:
            float: The computed log-likelihood
        """
        return sum(np.log(self.pdf(x)) for x in data)


class MultivariateDistribution(BaseDistribution):
    """
    This class will be the basis of copulas building.
    TODO: This docstring shuold be improved greatly!!
    """
    __metaclass__ = ABCMeta

    def __init__(self, dimension, dimkeys=None):
        """
        Args:
            dimension (int): The dimension of the distribution
            dimkeys (List): A list of the names of the dimensions, by default,
                these will just be the indices. If passed in, this will enable
                you to refer to values by the dimension name in certain
                functions
        """
        BaseDistribution.__init__(self, dimension)
        self.ndim = dimension
        if dimkeys is None:
            # We default to using the integers if no dimkeys are passed in.
            self.dimkeys = list(range(self.ndim))
        else:
            self.dimkeys = dimkeys

    def pdf(self, *xs):
        raise NotImplementedError

    def log_likelihood(self, data):
        """
        This method will return the log likelihood of the observed data
        given the fitted model.

        This method just naively computes the pdf and applies the logarithm.
        It would be more efficient in subclasses to find an expression for
        the log-likelihood.

        The argument data can either be a list of vectors for each dimension
        of the data or it can be a dictionary mapping dimension names to the
        corresponding vector of data.

        Args:
            data (list[list[float]] | dict[list[float]]): The observed values
        Returns:
            float: The computed log-likelihood
        """
        if isinstance(data, dict):
            vects = [data[dimkey] for dimkey in dimkeys]
        else:
            vects = data

        return sum(self.pdf(*xs) for xs in zip(*vects))

    @memoize_method
    def rect_prob(self, lowerdict, upperdict):
        tempdict = dict.fromkeys(self.dimkeys)
        def f(n):

            # recursive function that will calculate the cdf
            # It has a structure of binary tree
            if n == 0:
                return self.cdf(tempdict)
            else:
                tempdict[self.dimkeys[n - 1]] = upperdict[self.dimkeys[n - 1]]
                leftresult = f(n - 1)
                tempdict[self.dimkeys[n - 1]] = lowerdict[self.dimkeys[n - 1]]
                rightresult = f(n - 1)
                return leftresult - rightresult

        return f(self.dimension)


def fit_wrapper(method):
    """
    This is a function decorator which will wrap the fit method for
    multivariate distributions. It will allow for data to be passed using
    a dictionary mapping names to lists of data.

    Internally this transforms the data into a lists of lists and then fits
    the distribution to that data. Then it assigns to the dimkeys attribute
    the list of names.

    Args:
        method: The class method fit of a multivariate distribution
    Returns:
        method: The modified method to handle dictionaries of input data
    """
    @wraps(method)
    def fit(cls, data, dimkeys=None, **kwargs):
        """
        This function converts the dictionary into a list, passes it to the
        fit method and then assigns to the distribution the dimkeys attribute.
        """
        vectors = []
        if isinstance(data, dict):
            for key in dimkeys:
                vectors.append(data[key])
        else:
            vectors = data

        distribution = method(cls, vectors, dimkeys, **kwargs)
        return distribution

    return fit


def accepts_dict(method):
    """
    This function decorator will allow any of the methods which accept separate
    values for each dimension to also accept a dictionary which has keys
    mapping to each dimension.

    For example, the pdf for any distribution generally has the prototype
        def pdf(self, *x):
    This decorator will unpack the dictionary into its respective dimensions
    and pass it to the function.

    The function that this decorator is applied to must have a prototype like
        def f(self, *x)

    This will enable you to call a function in the following three ways.

    Suppose distr is a Distribution with distr.dimkeys = ['foo', 'bar', 'baz']
    If pdf is decorated with accepts_dict, we can call it like so
        1) distr.pdf(1, 2, 3)
        2) distr.pdf(foo=1, bar=2, baz=3)
        3) distr.pdf({'foo': 1, 'bar': 2, 'baz': 3})

    Args:
        method: The method accepting the different values for each dimensions
    Returns:
        method: The modified method to handle dictionaries of input data
    """
    @wraps(method)
    def f(self, *xs, **kwargs):
        if xs:
            # If xs is passed in, we check if the user passed it as each
            # dimension separately or as a dictionary
            if isinstance(xs[0], dict):
                # If the first element of xs, is a dict, assume only element.
                value_dict = xs[0]
                values = [value_dict[key] for key in self.dimkeys]
            else:
                # Otherwise, it is a list of the values at each dimension
                values = xs
        else:
            # Otherwise, we expect the values to be passed as keyword args.
            values = [kwargs[key] for key in self.dimkeys]
        return method(self, *values)

    return f


def returns_dict(method):
    """
    This function decorator will allow descendants of MultivariateDistribution
    which have methods which return values for each dimension to instead
    return a dictionary of values mapping dimension name to value.

    This adds an as_dict argument which if set to True, will pack the return
    value into a dictionary assuming the order is in that of the dimkeys
    attribute of the distribution.

    The as_dict argument must be passed by keyword.

    Args:
        method: The method which returns a list of values for each dimension
    Returns:
        method: The modified method to return a dictionary if specified to
    """

    @wraps(method)
    def f(self, *pargs, as_dict=False, **kwargs):
        values = method(self, *pargs, **kwargs)
        if as_dict:
            output = {key: value for key, value in zip(self.dimkeys, values)}
            return output
        else:
            return values

    return f
