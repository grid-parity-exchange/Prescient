#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module will house a ScenarioGenerator class which will act similar to the
way machine learning models in the packages sklearn and keras work. In this
sense, this class will contain a host of configurations which will modify how
scenario generation is performed.

This will provide an API which will supplement the command line
interface.
"""

import datetime
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from prescient.gosm.sources import Criterion, WindowSet
from prescient.util.distributions import UnivariateEpiSplineDistribution
from prescient.util.distributions.copula import GaussianCopula
from prescient.util.distributions.distribution_factory import distribution_factory
from prescient.gosm.structures import Hyperrectangle
import prescient.gosm.structures.skeleton_scenario as scenario
import prescient.gosm.structures.skeleton_point_paths as skelpaths
import prescient.gosm.sources as sources


class Method:
    """
    This will serve as an abstract base class for what a method for generating
    scenarios will be represented as in the code. To this end, we have a
    distinction between method-specific data and source-specific data.
    Method-specific data refers to the specific parameterization of the tools
    used for a specific scenario generation method. These parameters are not
    specific to any source. Source-specific parameters are the forecasts,
    actuals, segmentation information, and other information specific to a
    given source.

    Any information specific to a Method should be initialized in a subclass
    of Method or set with methods of the Method. Any information related
    to the sources themselves should be initialized in an associated
    FittedMethod to set any source-specific parameters.

    The bare minimum that each
    class should provide is an initialization method which sets the
    method-specific configurations.

    This class should not be instantiated, only subclassed.
    """
    def fit(self, sources, day):
        """
        This method will fit any source-specific data to the method. It will
        determine any source-specific parameters. This will fit for a
        specific day.

        Args:
            sources (list[Source]): The list of sources to be used in the
                method, a single Source object if the method expects a single
                source
            day (datetime-like): The date to fit the data
        Returns:
            FittedMethod: The method fitted to the data and date
        """
        pass


class FittedMethod:
    """
    This class will represent a Method object which has been fit to specific
    sources on a specific day. This will internally store a Method object
    as well as information regarding the source and day of scenario generation.
    It will also store method-specific fitting information in the subclasses.

    If the number of sources is equal to 1, there are a collection of
    attributes which are provided to make accessing the single source
    easier.

    The following attributes can only be used if the number of sources
    is equal to one.

    Attributes:
        source (Source): The single source if there is only one source
    """
    def __init__(self, method, sources, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        """
        Args:
            method (Method): The method of scenario generation
            sources (list[Source]): A list of source objects for which the
                method is fit
            day (datetime-like): The day for which scenarios are to be
                generated
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
        """
        self.method = deepcopy(method)
        self.sources = sources
        self.day = pd.Timestamp(day)
        self.historic_data_start = historic_data_start
        self.historic_data_end = historic_data_end
        self.verbose = verbose

    @property
    def source(self):
        """
        This method will return the source associated with the method

        Returns:
            Source: The associated source object
        """
        if len(self.sources) != 1:
            raise ValueError("This method has multiple sources, so this "
                             "attribute cannot be used.")
        return self.sources[0]

    def print_progress(self, message):
        """
        This message will print out the message of the progress of the certain
        method with information about the sources and the day.
        """
        source_names = [source.name for source in self.sources]
        print(", ".join(source_names) + ' ' + str(self.day.date())
              + ': ' + message)

    def generate(self):
        pass


class SampleMethod(Method):
    """
    This class will encapsulate information required for using samples from
    epispline distributions for generating scenarios. This will simply
    sample for every hour from a epispline distribution fit to all the errors
    for a given source. Then it will add the error to the forecast for the
    hour to get a 24 vector that will constitute the scenario.

    It will do this for as many samples as specified and assign each scenario
    a probability with 1 over the number of samples.
    """
    def __init__(self, number_of_samples=100,
                 distr_class=UnivariateEpiSplineDistribution):
        """
        Args:
            number_of_samples (int): The number of scenarios to generate
            distr_class: The class of the distribution to use, if not
                using the UnvariateEpiSplineDistribution
        """
        self.number_of_samples = number_of_samples
        self.distr_class = distr_class

    def fit(self, source, day, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This fits the sample method to a specific source and day.

        Args:
            source (Source): The source to fit the method to
            day (datetime-like): The day to generate scenarios for
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedSampleMethod: The method fitted to the data
        """
        return FittedSampleMethod(self, source, day, historic_data_start,
                                  historic_data_end, verbose)


class FittedSampleMethod(FittedMethod):
    """
    This method is the fitted version of SampleMethod.
    It fits a distribution to all the errors and simply samples a point
    for each hour of the day. It does this for the number of scenarios
    specified.

    Generally, this should not be instantiated directly, rather it should
    be constructed from the fit method.

    Args:
        method (SampleMethod): The SampleMethod object containing information
            about method specific parameters
        source (Source): The uncertainty source to fit the data to
        day (datetime-like): The day of scenario generation
        verbose (bool): If set to True, will print information about
           the state of the method
    """
    def __init__(self, method, source, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        """
        Args:
            method (SampleMethod): The SampleMethod object containing
                information about method specific parameters
            source (Source): The uncertainty source to fit the data to
            day (datetime-like): The day of scenario generation
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
               the state of the method
        """
        FittedMethod.__init__(self, method, [source], day, historic_data_start,
                              historic_data_end, verbose)
        # We add an errors column if not there originally
        errors = source.get_column('actuals') - source.get_column('forecasts')
        source.add_column('errors', errors)

        # We construct a rolling window for the day in particular
        day = pd.Timestamp(day)
        self.window = source.rolling_window(day, historic_data_start,
                                            historic_data_end)

        # This is the errors up to the day of scenario generation
        errors = self.window.get_column('errors').values

        if self.verbose:
            self.print_progress('Fitting Distributions')
        self.distribution = method.distr_class.fit(errors)

    def generate(self):
        """
        This will generate scenarios using the sample method discussed in
        the docstring of this class.

        Returns:
            list[PowerScenario]: A list of scenarios generated via sampling
        """
        self.scenarios = []
        forecasts = self.window.dayahead_data['forecasts'].values

        scenarios = []

        if self.verbose:
            self.print_progress('Generating Scenarios')

        # For each scenario
        for i in range(self.method.number_of_samples):
            # Construct scenario by sampling
            errors = [self.distribution.sample_one() for _ in range(24)]
            skeleton = (forecasts + errors).tolist()
            power_dict = {self.data.name: skeleton}

            # We assume the probability of each scenario is the same
            probability = 1/self.method.number_of_samples
            scenario_name = "Scenario_{}".format(i+1)
            comment = "Source {}: Sampled".format(self.window.name)
            scenarios.append(scenario.PowerScenario(
                scenario_name, power_dict, probability))

        return scenarios


class EpiSplineMethod(Method):
    """
    This class will encapsulate information required for using the epispline
    method for generating scenarios. This will be the basic epispline method
    which only considers a single source and requires day part separators and
    a set of paths (cutpoint sets).
    """
    def __init__(self, dps, paths, interval_set, copula_class=None,
                 monte_carlo=False, n=100000):
        """
        Args:
            dps (list[int]): The list of day part separators
            paths (list[Path]): The list of paths naming the intervals to
                take at each step
            interval_set (HyperrectanglePatternSet): The mapping from names
                to intervals
            copula_class (str): Optionally, the name of the copula to use
                if considering temporal dependence
            monte_carlo (bool): set to True if Monte Carlo
                integration is to be used
            n (int): The number of samples to use if Monte Carlo integration
                is used
        """
        self.dps = dps
        self.spline_options = {}
        self.tolerances = {}
        if copula_class is not None:
            self.copula_class = distribution_factory(copula_class)
        else:
            self.copula_class = None
        self.skeleton_paths = []
        for path in paths:
            self.skeleton_paths.extend(
                path.to_one_dim_paths(interval_set))
        self.monte_carlo = monte_carlo
        self.n = 100000

    def set_spline_options(self, error_distribution_domain='4,min,max',
                           specific_prob_constraint=None, seg_N=20,
                           seg_kappa=100,
                           non_negativity_constraint_distributions=0,
                           probability_constraint_of_distributions=1,
                           nonlinear_solver='ipopt'):
        """
        This method will set the specific options used for fitting the
        epispline distribution to the error data. This method will not fit
        the distribution to data and will only affect the fitted distributions
        if called before calling fit.

        Args:
            error_distribution_domain: A number (int or float) specifying
                how many standard deviations we want to consider as a domain of
                the distribution or a string that defines the sign of the
                domain (pos for positive and neg for negative).
            specific_prob_constraint: either a tuple or a list of length 2
                with values for alpha and beta
            seg_N (int): An integer specifying the number of knots
            seg_kappa (float): A bound on the curvature of the spline
            non_negativity_constraint_distributions: Set to 1 if u and w should
                be nonnegative
            probability_constraint_of_distributions: Set to 1 if integral of
                the distribution should sum to 1
            nonlinear_solver (str): String specifying which solver to use
        """
        self.spline_options = {
            'error_distribution_domain': error_distribution_domain,
            'specific_prob_constraint': specific_prob_constraint,
            'seg_N': seg_N,
            'seg_kappa': seg_kappa,
            'non_negativity_constraint_distributions':
                non_negativity_constraint_distributions,
            'probability_constraint_of_distributions':
                probability_constraint_of_distributions,
            'nonlinear_solver': nonlinear_solver
        }

    def set_monte_carlo(self, monte_carlo, n=100000):
        """
        This method will set this method to use monte carlo integration
        or not with the specified number of samples.

        Args:
            monte_carlo (bool): set to True if Monte Carlo
                integration is to be used
            n (int): The number of samples to use if Monte Carlo integration
                is used
        """
        self.monte_carlo = monte_carlo
        self.n = 100000

    def set_tolerance_levels(self, cdf_inverse_tolerance=1e-4,
                             cdf_inverse_max_refinements=10,
                             cdf_tolerance=1e-4):
        """
        This sets parameters which specify the accuracy to which the cdf and
        the cdf inverse are to be calculated which are used in the computation
        of the conditional expectation.

        This will update the self.tolerances dictionary.

        Args:
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        """
        self.tolerances = {
            'cdf_inverse_tolerance': cdf_inverse_tolerance,
            'cdf_inverse_max_refinements': cdf_inverse_max_refinements,
            'cdf_tolerance': cdf_tolerance
        }

    def fit(self, source, day, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This fits the epispline method to a specific source and day.

        Args:
            source (Source): The source to fit the method to
            day (datetime-like): The day to generate scenarios for
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedEpiSplineMethod: The method fitted to the data
        """
        return FittedEpiSplineMethod(self, source, day, historic_data_start,
                                     historic_data_end, verbose)


class FittedEpiSplineMethod(FittedMethod):
    """
    The fitted version of EpiSplineMethod.

    Attributes:
        method (EpiSplineMethod): The method-specific parameters for
            scenario creation
        distributions (dict): A dictionary mapping day part separators
            to distributions fitted to data from the source at that hour
    """
    def __init__(self, method, source, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        """
        This will fit distributions to errors for each of the day
        part separators. It will also construct a rolling window for the day
        in question.

        Args:
            method (EpiSplineMethod): The specific EpiSplineMethod with
                all the specific configurations
            source (Source): The source to fit the method to
            day (datetime-like): The day to generate scenarios for, must be
                coercible into a pandas Timestamp
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        """
        FittedMethod.__init__(self, method, [source], day, historic_data_start,
                              historic_data_end, verbose)
        # We add an error column to the source of data
        source.add_column('errors', source.get_column('actuals')
                                    - source.get_column('forecasts'))

        day = pd.Timestamp(day)
        self.window = source.rolling_window(day, historic_data_start,
                                            historic_data_end)
        if self.verbose:
            self.print_progress('Fitting Distributions')

        self.distributions = {}
        self.compute_distributions_at_dps()

        if self.method.copula_class is not None:
            if self.verbose:
                self.print_progress('Fitting Copula')
            self.copula = self.window.fit_copula_at_hours(
                'errors', method.dps, method.copula_class)
        else:
            self.copula = None

    def compute_distributions_at_dps(self):
        """
        This function will fit an epispline distribution to the errors at
        each of the hours specified in the dps. This will take the source
        associated with this method get its segment and fit the epispline
        distribution to the errors at the day part separator.

        Args:
            dps (list[int]): The hours to compute distributions at
        """
        # For every hour
        for hour in self.method.dps:
            dt = self.day + datetime.timedelta(hours=hour)
            # Segment the data and extract the errors
            segment = self.window.segment(dt)
            errors = segment.get_column('errors').values

            # Fit the epispline distribution to the hour's errors
            self.distributions[hour] = UnivariateEpiSplineDistribution(
                errors, **self.method.spline_options)

    def generate(self, with_paths=False):
        """
        This will generate a collection of PowerScenarios using the
        epispline method. This is done by computing the conditional expectation
        on each of the paths to determine the errors at the day part
        separators. Then in between these day part separators, the values
        are interpolated. This is then added to the forecast to create a
        scenario for each path.

        Args:
            with_paths (bool): Optionally, set to True, to instead return
                a dictionary mapping OneDimPath objects to the corresponding
                scenarios
        Returns:
            list[PowerScenario]: The list of generated scenarios, if
                with_paths is set to True, this will return a dictionary
                mapping OneDimPaths to corresponding scenarios
        """
        forecasts = self.window.dayahead_data['forecasts'].values

        scenarios = []

        if self.verbose:
            self.print_progress('Computing Conditional Expectations')

        for path in self.method.skeleton_paths:
            # At the day part separators, we take the error value to be
            # the conditional expectation on the defining interval for a given
            # path.
            distrs = [self.distributions[hour] for hour in self.method.dps]
            errors = path.compute_conditional_expectation_on(
                    distrs, **self.method.tolerances)
            interpolated_errors = np.interp(list(range(24)),
                                            self.method.dps, errors)
            skeleton = forecasts + interpolated_errors
            clipped = np.clip(skeleton, 0, self.window.capacity)
            power_dict = {self.source.name: clipped.tolist()}
            # If we are not to use a copula, self.copula should be None.
            if self.copula is None:
                prob = skelpaths.independent_probability(path)
            else:
                prob = skelpaths.temporally_dependent_probability(
                    path, self.copula, self.method.dps,
                    self.method.monte_carlo, self.method.n)

            # We store information about sources and paths in comments
            # on the PowerScenario object
            comment = "{}: EpiSpline {}".format(self.source.name, path.name)
            scenarios.append(scenario.PowerScenario(
                path.name, power_dict, prob, comment))

        if with_paths:
            # If with_paths is set, we return a list of ScenarioWithPaths
            # objects. This is needed for using spatial copulas.
            return [scenario.ScenarioWithPaths(scen, {self.source.name: path})
                    for path, scen
                    in zip(self.method.skeleton_paths, scenarios)]
        else:
            return scenarios


    def plot(self, output_directory, plot_pdf=True, plot_cdf=True):
        """
        This will plot the cdf and pdf of each constructed epispline to
        the specified output directory. This will create files of the form
        <source>_pdf_at_hour_<hour>.png and <source>_cdf_at_hour_<hour>.png
        for every hour.

        Args:
            output_directory (str): A string specifying where to plot the
                distributions
            plot_pdf (bool): True if the pdf is to be plotted
            plot_cdf (bool): True if the cdf is to be plotted
        """
        for hour, distr in self.distributions.items():
            pdf_file = "{}_pdf_at_hour_{}.png".format(self.source.name, hour)
            pdf_title = "{} Epispline PDF hour {}".format(
                self.source.name, hour)

            if plot_pdf:
                distr.plot(True, False, pdf_file, pdf_title,
                           "Error [Mw]", "Pdf", output_directory)

            cdf_file = "{}_cdf_at_hour_{}.png".format(self.source.name, hour)
            cdf_title = "{} Epispline CDF hour {}".format(
                self.source.name, hour)

            if plot_cdf:
                distr.plot(False, True, cdf_file, cdf_title,
                           "Error [Mw]", "Cdf", output_directory)


class SolarEpiSplineMethod(EpiSplineMethod):
    """
    This method works exactly the same as the EpiSplineMethod, but the first
    and last day part separators are taken to be the hours of sunrise and
    sunset which are in turn forced to have 0 power generation in the
    constructed scenario.

    Attributes:
        dps (list[int]): In this case, this will refer to purely the dps
            for which distributions are fit, i.e., all the ones but the first
            and last in the dps used to construct the distribution
    """
    def __init__(self, dps, paths, interval_set, copula_class=None,
                 monte_carlo=False, n=100000):
        """
        Args:
            dps (list[int]): The list of day part separators
            paths (list[Path]): The list of paths naming the intervals to
                take at each step
            interval_set (HyperrectanglePatternSet): The mapping from names
                to intervals
            copula_class (str): Optionally, the name of the copula to use
                if considering temporal dependence
            monte_carlo (bool): set to True if Monte Carlo
                integration is to be used
            n (int): The number of samples to use if Monte Carlo integration
                is used
        """
        self.all_dps = dps
        EpiSplineMethod.__init__(self, dps[1:-1], paths, interval_set,
                                 copula_class, monte_carlo, n)

    def fit(self, source, day, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This fits the epispline method to a specific source and day.

        Args:
            source (Source): The source to fit the method to
            day (datetime-like): The day to generate scenarios for
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedSolarEpiSplineMethod: The method fitted to the data
        """
        return FittedSolarEpiSplineMethod(self, source, day,
                                          historic_data_start,
                                          historic_data_end, verbose)


class FittedSolarEpiSplineMethod(FittedEpiSplineMethod):
    """
    The fitted version of SolarEpiSplineMethod
    """
    def __init__(self, method, source, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        """
        This will fit distributions to errors for each of the day
        part separators. It will also construct a rolling window for the day
        in question. This differs from the EpiSplineMethod slightly in that
        the first and last day part separators of the method are taken to be
        the hours of sunrise and sunset, meaning we assume 0 energy production
        at these hours and other daylight errors. No distributions are fit to
        these day part separators.

        Args:
            method (SolarEpiSplineMethod): The specific SolarEpiSplineMethod
                with all the specific configurations
            source (Source): The source of solar power data to fit the
                method to
            day (datetime-like): The day to generate scenarios for, must be
                coercible into a pandas Timestamp
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        """
        # The parent class does all the work needed
        # It will only fit distributions to the dps that are at daytime
        FittedEpiSplineMethod.__init__(self, method, source, day,
                                       historic_data_start, historic_data_end,
                                       verbose)

    def generate(self, with_paths=False):
        """
        This will generate scenarios for a solar source using the epispline
        method. This is done as in the normal epispline method except at the
        hours of sunrise and sunset, the values are forced to be 0. This is
        done by taking the error to be the negative of the forecast at these
        hours. Then for hours outside sunrise and sunset, the errors are
        forced to be 0.

        Args:
            with_paths (bool): Optionally, set to True, to instead return
                a list of ScenarioWithPath objects which have a "scenario"
                attribute and a "path" attribute
        Returns:
            list[PowerScenario]: The list of generated scenarios, if
                with_paths is set to True, this will return a list of
                ScenarioWithPath objects
        """
        forecasts = self.window.dayahead_data['forecasts'].values
        capacity = self.window.capacity

        scenarios = []

        sunrise_dps = self.method.all_dps[0]
        sunset_dps = self.method.all_dps[-1]
        # We just take the error at the sunrise and sunset to force the
        # scenario value to be 0.
        sunrise_error = -forecasts[sunrise_dps]
        sunset_error = -forecasts[sunset_dps]

        if self.verbose:
            self.print_progress('Computing Conditional Expectations')

        for path in self.method.skeleton_paths:
            distrs = [self.distributions[hour] for hour in self.method.dps]
            # We drop the first and last element from the path, since
            # we do not have distributions for those hours.
            error_at_dps = [distr.conditional_expectation(
                                interval, **self.method.tolerances)
                for distr, interval in zip(distrs, path.slice(1,-1).intervals)]
            errors = [sunrise_error] + error_at_dps + [sunset_error]
            interpolated_errors = np.interp(list(range(24)),
                                            self.method.all_dps, errors)
            skeleton = forecasts + interpolated_errors

            # Truncate the scenarios at 0 and at the capacity
            clipped = np.clip(skeleton, 0, capacity)

            # We zero out all hours that are out of daylight hours.
            for hour in range(24):
                if not (sunrise_dps < hour < sunset_dps):
                    clipped[hour] = 0

            power_dict = {self.source.name: clipped.tolist()}

            # If we are not to use a copula, self.copula should be None.
            if self.copula is None:
                prob = skelpaths.independent_probability(path)
            else:
                prob = skelpaths.temporally_dependent_probability(
                    path, self.copula, self.method.dps,
                    self.method.monte_carlo, self.method.n)

            # We store information about sources and paths in comments
            # on the PowerScenario object
            comment = "{}: EpiSpline {}".format(self.source.name, path.name)
            scenario.PowerScenario(path.name, power_dict, prob)
            scenarios.append(scenario.PowerScenario(
                path.name, power_dict, prob))

        if with_paths:
            # If with_paths is set, we return a list of ScenarioWithPaths
            # objects
            return [scenario.ScenarioWithPaths(scen, {self.source.name: path})
                    for path, scen
                    in zip(self.method.skeleton_paths, scenarios)]
        else:
            return scenarios


class ProductMethod(Method):
    """
    This method for scenario generation will use a
    single Method for generating scenarios independently for each of the
    sources. Alternatively, it will take a list of methods and use one of
    of these methods for each source.
    Then it will, assuming the sources are independent, take the
    cartesian product of all the sets of scenarios generated constructing
    new scenarios with probability equal to the product of each constituent
    scenario

    Example:
        >>> sample_method = SampleMethod(100)
        >>> source1 = gosm.sources.source_from_csv('wind.csv', 'wind', 'wind')
        >>> source2 = gosm.sources.source_from_csv('sol.csv', 'solar', 'solar')
        >>> product_method = ProductMethod(sample_method)
        >>> product_method.fit([source1, source2], '2013-06-01')
        >>> scenarios = product_method.generate()
    """
    def __init__(self, methods):
        """
        Args:
            method (Method): A list of different Methods to generate scenarios
                one for each possible source. Alternatively, a single Method
                which will be used for all sources passed in.
        """

        # We keep track of whether the user passed in a single method or not
        # If it is a single method, we will use this method for all sources.
        if isinstance(methods, Method):
            self.same_for_all = True
            self.method = methods
        elif isinstance(methods, list):
            self.same_for_all = False
        else:
            raise TypeError("methods must either be a subclass of Method or "
                            " a list of subclasses of Methhod")
        self.methods = methods

    def fit(self, sources, day, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This method will fit any source-specific data to the method. It will
        determine any source-specific parameters. This will fit for a
        specific day.

        Args:
            sources (list[Source]): The list of sources to be used in the
                method, a single Source object if the method expects a single
                source
            day (datetime-like): The date to fit the data
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedSpatialCopulaMethod: The method fitted to the data and date
        """
        return FittedProductMethod(self, sources, day, historic_data_start,
                                   historic_data_end, verbose)


class FittedProductMethod(FittedMethod):
    """
    The fitted version of ProductMethod

    Attributes:
        method (DeterministicMethod): The method-specific information
        sources (list[Source]): The associated sources of data
        day (pd.Timestamp): The date that scenarios are generated for
    """
    def __init__(self, method, sources, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        """
        Args:
            sources (list[Source]): The list of sources to be used in the
                method, a single Source object if the method expects a single
                source
            day (datetime-like): The date to fit the data
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method

        """
        FittedMethod.__init__(self, method, sources, day, verbose)

        # If have multiple methods and cannot map the methods
        # to sources.
        if not method.same_for_all and len(method.methods) != len(sources):
            raise ValueError("This method expects {} sources, only received "
                             "{} sources".format(len(method.methods),
                                                 len(sources)))

        # Internally store fitted methods for each source
        if method.same_for_all:
            self.fit_methods = [method.method.fit(source, day,
                                                  historic_data_start,
                                                  historic_data_end, verbose)
                                for source in sources]
        else:
            self.fit_methods = [method.fit(source, day, historic_data_start,
                                           historic_data_end, verbose)
                                for method, source in
                                zip(method.methods, sources)]

    def generate(self):
        """
        This will generate scenarios for each of the sources using their
        respective method. Then it will merge them assuming independence.

        Returns:
            list[PowerScenario]: The list of generated scenarios
        """
        scenario_sets = []
        for method in self.fit_methods:
            scenario_sets.append(method.generate())

        power_scenarios = []
        for scenarios in product(*scenario_sets):
            power_scenarios.append(
                scenario.merge_independent_scenarios(scenarios))

        return power_scenarios

    def plot(self, output_directory, **kwargs):
        """
        This function will call each submethods plot method with the specified
        keyword arguments.

        Args:
            output_directory (str): The name of the directory to be saving
                the plots in.
        """
        for method in self.fit_methods:
            method.plot(output_directory, **kwargs)


class DeterministicMethod(Method):
    """
    This is a method which will create one scenario from a single source
    which will be just the forecast for a given day. The scenario will have
    probability 1. To be instantiated with no arguments.
    """
    def fit(self, source, day, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This fits the deterministic method to a specific source and day.

        Args:
            source (Source): The source to fit the method to
            day (datetime-like): The day to generate scenarios for
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedDeterministiceMethod: The method fitted to the data
        """
        return FittedDeterministicMethod(self, source, day,
                                         historic_data_start,
                                         historic_data_end, verbose)


class FittedDeterministicMethod(FittedMethod):
    """
    The fitted version of DeterministicMethod

    Attributes:
        method (DeterministicMethod): The method-specific information
        source (Source): The associated source of data
        day (pd.Timestamp): The date that scenarios are generated for
    """
    def __init__(self, method, source, day, historic_data_start=None,
                 historic_data_end=None, verbose=False):
        FittedMethod.__init__(self, method, [source], day, verbose)
        self.window = source.rolling_window(day, historic_data_start,
                                            historic_data_end)

    def generate(self):
        """
        This function will generate a list of a single scenario which has
        probability 1 and is simply the forecast for the given source

        Returns:
            list[PowerScenario]: The list of a single deterministic scenario
        """
        if self.verbose:
            self.print_progress('Generating Deterministic Scenario')
        forecasts = self.window.dayahead_data['forecasts'].values.tolist()
        name = self.source.name + '_deterministic'
        power_dict = {self.source.name: forecasts}
        comment = "{}: Deterministic".format(self.source.name)
        return [scenario.PowerScenario(name, power_dict, 1, comment)]


class SpatialCopulaMethod(Method):
    """
    This method will apply the epispline method to each source individually.
    Then using a spatial copula, this will compute the probabilities of the
    combined scenarios. Put more concretely, for each day part separator, to
    compute the probability, we will compute the probability over the
    hyperrectangle constructed from the cartesian product of each interval.

    Then since we have independence over the day part separators, we get a
    final probability by multiplying the probabilities together.
    """
    def __init__(self, dps, paths, interval_set,
                 spatial_copula='gaussian-copula', same_path=True,
                 monte_carlo=False, n=100000):
        """
        Args:
            dps (list[int]): The list of day part separators
            paths (list[Path]): The list of paths to be taken in each scenario
            interval_set (HyperrectanglePatternSet): The collection of
                patterns, these must be of dimension 1
            spatial_copula (str): The name of the copula to be used
            same_path (bool): Optionally, True if sources in the same cluster
                are to use the exact same path in each scenario, False if can
                use different paths (very computationally intensive)
            monte_carlo (bool): Optionally, set to True if Monte Carlo
                integration is to be used
            n (int): The number of samples to use if Monte Carlo integration
                is used
        """
        self.dps = dps
        self.paths = paths
        self.interval_set = interval_set
        self.spline_options = {}
        self.tolerances = {}
        self.skeleton_paths = []
        for path in paths:
            self.skeleton_paths.extend(
                path.to_one_dim_paths(interval_set))
        self.spatial_copula = distribution_factory(spatial_copula)
        self.same_path = same_path
        self.monte_carlo = monte_carlo
        self.n = n

    def set_spline_options(self, error_distribution_domain='4,min,max',
                           specific_prob_constraint=None, seg_N=20,
                           seg_kappa=100,
                           non_negativity_constraint_distributions=0,
                           probability_constraint_of_distributions=1,
                           nonlinear_solver='ipopt'):
        """
        This method will set the specific options used for fitting the
        epispline distribution to the error data. This method will not itself
        fit the distribution to data and will only affect the fitted
        distributions if called before calling fit.

        Args:
            error_distribution_domain: A number (int or float) specifying
                how many standard deviations we want to consider as a domain of
                the distribution or a string that defines the sign of the
                domain (pos for positive and neg for negative).
            specific_prob_constraint: either a tuple or a list of length 2
                with values for alpha and beta
            seg_N (int): An integer specifying the number of knots
            seg_kappa (float): A bound on the curvature of the spline
            non_negativity_constraint_distributions: Set to 1 if u and w should
                be nonnegative
            probability_constraint_of_distributions: Set to 1 if integral of
                the distribution should sum to 1
            nonlinear_solver (str): String specifying which solver to use
        """
        self.spline_options = {
            'error_distribution_domain': error_distribution_domain,
            'specific_prob_constraint': specific_prob_constraint,
            'seg_N': seg_N,
            'seg_kappa': seg_kappa,
            'non_negativity_constraint_distributions':
                non_negativity_constraint_distributions,
            'probability_constraint_of_distributions':
                probability_constraint_of_distributions,
            'nonlinear_solver': nonlinear_solver
        }

    def set_monte_carlo(self, monte_carlo, n=100000):
        """
        This method will set this method to use monte carlo integration
        or not with the specified number of samples.

        Args:
            monte_carlo (bool): set to True if Monte Carlo
                integration is to be used
            n (int): The number of samples to use if Monte Carlo integration
                is used
        """
        self.monte_carlo = monte_carlo
        self.n = 100000

    def set_tolerance_levels(self, cdf_inverse_tolerance=1e-4,
                             cdf_inverse_max_refinements=10,
                             cdf_tolerance=1e-4):
        """
        This sets parameters which specify the accuracy to which the cdf and
        the cdf inverse are to be calculated which are used in the computation
        of the conditional expectation.

        This will update the self.tolerances dictionary.

        Args:
            cdf_inverse_tolerance (float): The accuracy which the inverse
                cdf is to be calculated to
            cdf_inverse_max_refinements (int): The number of times the
                the partition on the x-domain will be made finer
            cdf_tolerance (float): The accuracy to which the cdf is calculated
                to
        """
        self.tolerances = {
            'cdf_inverse_tolerance': cdf_inverse_tolerance,
            'cdf_inverse_max_refinements': cdf_inverse_max_refinements,
            'cdf_tolerance': cdf_tolerance
        }

    def fit(self, sources, day, partition, historic_data_start=None,
            historic_data_end=None, verbose=False):
        """
        This method will fit any source-specific data to the method. It will
        determine any source-specific parameters. This will fit for a
        specific day.

        Args:
            sources (list[Source]): The list of sources to be used in the
                method, a single Source object if the method expects a single
                source
            day (datetime-like): The date to fit the data
            partition (Partition): A partition of the names of the sources
                determining which sources are dependent
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        Returns:
            FittedSpatialCopulaMethod: The method fitted to the data and date
        """
        return FittedSpatialCopulaMethod(self, sources, day, partition,
                                         historic_data_start,
                                         historic_data_end, verbose)


class FittedSpatialCopulaMethod(FittedMethod):
    """
    The fitted version of SpatialCopulaMethod
    Attributes:
        sources (list[Source]): The list of sources to be fit
        method (SpatialCopulaMethod): The method-specific parameters, stores
            information about spline parameters, paths, partition, dps
        day (pd.Timestamp): The datetime of the day for which scenarios
            are generated
        partition (Partition): The partition of the source names
    """
    def __init__(self, method, sources, day, partition,
                 historic_data_start=None, historic_data_end=None,
                 verbose=False):
        """
        Args:
            method (SpatialCopulaMethod): All the method-specific parameters
            sources (list[Source]): The list of sources to be used in the
                method, a single Source object if the method expects a single
                source
            day (datetime-like): The date to fit the data
            partition (Partition): A partition of the names of the sources
                determining which sources are dependent
            historic_data_start (datetime.datetime): The datetime of the start
                of the historic data, if None just use start of data
            historic_data_end (datetime.datetime): The datetime of the end of
                the historic data, if None draws up to the day passed
            verbose (bool): If set to True, will print information about
                the state of the method
        """
        FittedMethod.__init__(self, method, sources, day, historic_data_start,
                              historic_data_end, verbose)
        if all([source.source_type == 'solar' for source in sources]):
            self.dps = self.method.dps[1:-1]
            self.source_type = 'solar'
        elif all([source.source_type == 'wind' for source in sources]):
            self.dps = self.method.dps
            self.source_type = 'wind'
        else:
            raise ValueError("All sources must be the same type to use a "
                             "spatial copula method")

        self.partition = partition

        # We use Epispline methods to produce the base scenarios
        self.fit_submethods = []

        # For solar sources, we use a slightly modified method
        if self.source_type == 'solar':
            submethod = SolarEpiSplineMethod(
                method.dps, method.paths, method.interval_set)
        else:
            submethod = EpiSplineMethod(
                method.dps, method.paths, method.interval_set)

        # We need to propagate spline and tolerance parameters to the
        # submethods.
        submethod.set_spline_options(**method.spline_options)
        submethod.set_tolerance_levels(**method.tolerances)

        self.fit_submethods = [submethod.fit(source, day, historic_data_start,
                                             historic_data_end, verbose)
                               for source in sources]
        self.windows = [meth.window for meth in self.fit_submethods]
        self.window_set = WindowSet(self.windows)

        # This will store copulas fit to each cluster of sources
        self.copulas = {}
        self.fit_copulas()

    def fit_copulas(self):
        """
        This will fit copulas to related sources as determined by the
        partition of the sources. This will then store the result in a
        dictionary in the attribute self.copulas.

        This dictionary will map tuples of the form (tuple[str], int) to
        distributions.

        Returns:
            dict: A dictionary mapping a pair consisting of a tuple of
                the cluster of source names and the day part separator to
                the copula fitted to the data associated with the cluster
                of sources at the specified hour
        """
        error_frame = self.window_set.get_column_from_windows('errors').dropna()

        # We first cosntruct a dictionary mapping day part separators
        # to dictionaries mapping source names to lists of errors
        # {hour -> {source_name -> errors}}
        error_dictionary = {}

        for hour in self.dps:
            errors_at_hour = error_frame[error_frame.index.hour == hour]
            errors = errors_at_hour.to_dict(orient='list')
            error_dictionary[hour] = errors

        if self.verbose:
            self.print_progress("Fitting Spatial Copulas")
        copulas = {}
        for cluster in self.partition:
            # We don't need to fit copulas to sources that are singletons.
            if len(cluster) == 1:
                continue

            source_names = tuple(sorted(cluster))
            for hour in self.dps:
                errors = error_dictionary[hour]

                source_errors = {}

                # We pull out the relevant source data
                # And we transform it to [0,1]^n
                for name in source_names:
                    marginal = UnivariateEpiSplineDistribution.fit(
                        errors[name], **self.method.spline_options)

                    source_errors[name] = [marginal.cdf(x)
                                           for x in errors[name]]

                copula = self.method.spatial_copula.fit(
                            source_errors, dimkeys=source_names)
                copulas[source_names, hour] = copula
        return copulas

    def compute_power_vectors(self):
        """
        This function will construct PowerScenario objects.
        """
        source_scenarios = {source.name: meth.generate(with_paths=True)
                            for source, meth
                            in zip(self.sources, self.fit_submethods)}

        merged_scenarios = []

        if self.method.same_path:
            first_merge = []
            # First we merge correlated scenarios by zipping
            for cluster in self.partition:
                scenario_sets = [source_scenarios[name] for name in cluster]
                cluster_scenarios = []
                for tup in zip(*scenario_sets):
                    cluster_scenarios.append(
                        scenario.merge_scenarios_with_paths(tup))
                first_merge.append(cluster_scenarios)

            # Then we merge uncorrelated scenarios by product
            for tup in product(*first_merge):
                merged_scenarios.append(
                    scenario.merge_scenarios_with_paths(tup))
        else:
            for tup in product(*source_scenarios.values()):
                merged_scenarios.append(
                    scenario.merge_scenarios_with_paths(tup))

        return merged_scenarios

    def spatial_probability(self, scen):
        """
        This will compute the probability of a given scenario based on
        the paths used for each source.

        For each day part separator and cluster of sources, we take
        the intervals for that hour for each source, then compute the integral
        over the product of the intervals on the copula fit to the cluster.

        We loop over each cluster and day part separator and multiply these
        all together.

        Args:
            scen (ScenarioWithPaths): The scenario to compute probability
                for with a path dictionary
        """
        probability = 1
        path_dict = scen.paths

        # For each cluster and copula fit to that cluster
        for ((source_names, hour), copula) in self.copulas.items():
            interval_index = self.method.dps.index(hour)

            # Pull out the bounds used for that day part separator
            bounds = [path_dict[source].intervals[interval_index]
                      for source in source_names]

            # Construct hyperrectangle with the bounds
            rectangle = Hyperrectangle(bounds, dimkeys=source_names)
            if self.method.monte_carlo:
                probability *= copula.mc_probability_on_rectangle(
                    rectangle, self.method.n)
            else:
                probability *= copula.probability_on_rectangle(rectangle)

        for source_name in self.partition.singletons():
            path = path_dict[source_name]
            probability *= skelpaths.independent_probability(path)

        return probability

    def generate(self):
        """
        Returns:
            list[PowerScenario]: The list of scenarios generated using a
                spatial copula
        """
        # We first construct a 24-vector for each possible scenario
        scenarios = self.compute_power_vectors()
        power_scenarios = []

        # Then we compute the probability for each using a spatial copula
        for scen in scenarios:
            power_scenario = scen.scenario
            probability = self.spatial_probability(scen)
            power_scenario.probability = probability
            
            # We add in a comment to specify that it was generated using
            # a spatial copula
            power_scenario.comments += '\nSpatial Copula'
            power_scenarios.append(power_scenario)
        return power_scenarios

    def plot(self, output_directory, plot_pdf=True, plot_cdf=True):
        """
        This will plot each of the submethod's distributions to the specified
        directory.

        Args:
            output_directory (str): A string specifying where to plot the
                distributions
            plot_pdf (bool): True if the pdf is to be plotted
            plot_cdf (bool): True if the cdf is to be plotted
        """
        for method in self.fit_submethods:
            method.plot(output_directory, plot_pdf, plot_cdf)
