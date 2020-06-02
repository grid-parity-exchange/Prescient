#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import sys
import math
from abc import ABCMeta

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.special as sps
import scipy.stats as spst
from scipy.stats import t, mvn
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.integrate import quad
from scipy.optimize import fmin

from .base_distribution import fit_wrapper
from .base_distribution import accepts_dict, returns_dict
from .base_distribution import MultivariateDistribution
from .distributions import MultiStudentDistribution
from .distributions import UnivariateEmpiricalDistribution
from .distribution_factory import distribution_factory
from .distribution_factory import register_distribution
from .utilities import memoize_method


class CopulaBase(MultivariateDistribution):
    """
    This class will be the basis of copulas building.
    """

    __metaclass__ = ABCMeta

    def __init__(self, dimension, dimkeys=None):
        """
        Args:
            dimension (int): The dimension of the distribution
        """
        MultivariateDistribution.__init__(self, dimension, dimkeys)

    @memoize_method
    def probability_on_rectangle(self, hyperrectangle, epsabs=1e-2):
        """
        This function should compute the probability of the random variable
        of this distribution being inside this rectangle.

        The hyperrectangle should have all the same dimension names as were
        used to construct this distribution.

        Args:
            hyperrectangle (Hyperrectangle): A hyperrectangle potentially with
                cutouts
        Returns: 
            float: A probability of being in the region
        """

        bounds = [hyperrectangle.bounds[key] for key in self.dimkeys]
        probability = spi.nquad(self.c, bounds, opts={'epsabs': epsabs})[0]

        if hasattr(hyperrectangle, 'cutouts'):
            for cutout in hyperrectangle.cutouts:
                bounds = [cutout.bounds[key] for key in self.dimkeys]
                probability -= spi.nquad(self.c, bounds,
                                         opts={'epsabs': epsabs})[0]

        return probability

    def mc_probability_on_rectangle(self, hyperrectangle, n):
        """
        Uses monte carlo integration to estimate the probability of the
        copula over the hyperrectangle passed in.

        Args:
            hyperrectangle (Hyperrectangle): The hyperrectangle to integrate
                on
            n (int): The number of samples to take
        Returns:
            float: The estimated probability on the rectangle
        """
        try:
            from gosm.mc_int import mc_integrate_on_rectangle
        except ImportError:
            raise ImportError("gosm is not installed, cannot use this method")
        bounds = [hyperrectangle.bounds[key] for key in self.dimkeys]
        volume = hyperrectangle.volume
        return mc_integrate_on_rectangle(self.c, bounds, n)[0]

    def conditional_expectation(self, hyperrectangle):
        """
        This should compute the expectation of the distribution on the
        hyperrectangle passed in. The result of this should be a mean vector.

        Right now, the way this works is to compute the expectation
        on each marginal and to compute the probability over the
        hyperrectangle. Then we divide the expectation by the probability.

        Args:
            hyperrectangle (Hyperrectangle): A hyperrectangle potentially with
                cutouts
        Returns:
            dict[name, value]: A dictionary mapping the name of each dimension
                to the corresponding expectation
        """
        probability = self.probability_on_rectangle(hyperrectangle)

        expectations = {}

        for name in self.dimkeys:
            marginal = self.marginals[name]

            lower, upper = hyperrectangle.bounds[name]
            inv_lower = marginal.cdf_inverse(lower)
            inv_upper = marginal.cdf_inverse(upper)
            expectation = marginal.region_expectation((inv_lower, inv_upper))

            if hasattr(hyperrectangle, 'cutouts'):
                for cutout in hyperrectangle.cutouts:
                    lower, upper = hyperrectangle.bounds[name]
                    inv_lower = marginal.cdf_inverse(lower)
                    inv_upper = marginal.cdf_inverse(upper)
                    expectation -= marginal.region_expectation((inv_lower,
                                                                inv_upper))
            expectations[name] = expectation / probability
        return expectations

    def C_from_sample(self, valuedict, n=10000):
        """
        This function is only used for a unittest to check if the values from
        the sample give the same C function.
        Args:
            valuedict(dict): the values where you want to compute C
            N(int): the size of the sampling
        Returns:
            the Copula function evaluated in valuedict
        """
        sample = generates_U(n)

        def condition(k): #Check if U[k] is inferior at valuedict
            for i in range(self.dimension):
                if U[k][i] > valuedict[self.dimkeys[i]]:
                    return False
            else:
                return True

        #Count the number of U inferior at value.
        count = sum(1 for k in range(n) if condition(k))
        return (float(count)/ n)

    def generates_U(self, n=1):
        pass

    @accepts_dict
    def pdf(self, *xs):
        return self.c(*xs)

    @accepts_dict
    def cdf(self, *xs):        return self.C(*xs)

    def kendall_function(self, w, n=1000):
        """
        This function will compute the empirical kendall function at w
        Args:
            w: the value where you want to compute
            n: the minimum size of your sample
        """
        sample = self.generates_U(n)

        total = 0
        for x in sample:
            if self.C(*x) <= w:
                total += self.c(*x)

        return total / n


class CopulaWithMarginals(MultivariateDistribution):
    """
    This class will define a distribution which uses a copula to encode the
    correlation structure between dimensions and marginal distributions to
    encode the structure of each individual dimension.
    """
    def __init__(self, copula, marginals, dimkeys=None):
        if isinstance(marginals, dict):
            self.marginals = [marginals[key] for key in dimkeys]
        else:
            self.marginals = marginals
        self.copula = copula
        MultivariateDistribution.__init__(self, copula.dimension, dimkeys)

    @accepts_dict
    def pdf(self, *xs):
        """
        This will compute the pdf of the distribution at the point specified
        in real space.

        Examples:
            This method may be called with each argument passed individually
            >>> distr.pdf(0, 0, 0)

            In this case, it is assumed that the order of the arguments passed
            in matches the order of the dimensions of the distribution.

            Alternatively, this method may be called using a dictionary thanks
            to the accepts_dict decorator. If the distribution is constructed
            with explicitly specified dimension names, the method may be called
            like so.
            >>> cop = CopulaWithMarginals(copula, margs, dimkeys=['A', 'B'])
            >>> cop.pdf({'A': 1, 'B': 0})
            >>> cop.pdf(B=1, A=0) # Alternative using keyword arguments

        Args:
            xs: The points to evaluate the pdf at, passed as individual
                arguments. Alternatively, this can be a dictionary which
                maps dimension names to the corresponding value
        Returns:
            float: The value of the pdf at the specified point
        """
        res=1
        cdfs = []
        for x, marginal in zip(xs, self.marginals):
            cdfs.append(marginal.cdf(x))
            res = res * marginal.pdf(x)
        res= res*self.copula.pdf(*cdfs)

        return res

    @accepts_dict
    def cdf(self, *xs):
        """
        Args:
            xs: The points to evaluate the cdf at, passed as individual
                arguments. Alternatively, this can be a dictionary which
                maps dimension names to the corresponding value
        Returns:
            float: The value of the pdf at the specified point
        """
        cdfs = []
        for x, marginal in zip(xs, self.marginals):
            cdfs.append(marginal.cdf(x))
        return self.copula.cdf(*cdfs)

    @classmethod
    def fit(cls, input_data, copula_class, marginal_classes, dimkeys=None):
        """
        This function will fit a CopulaWithMarginals to the data
        passed in. The data passed in may be either a list of lists or it may
        be a dictionary mapping dimension names to lists of data. In the
        second case, a list of dimension names must be passed as well in order
        to determine the ordering of the dimensions.

        This will first fit marginals to each of the dimensions and then
        apply each marginal cdf to each dimension, then it will fit the copula
        to the transformed data.

        Args:
            input_data (List[List[float]] | dict[name,List[float]]): The data to fit
                the distribution to
            copula_class (CopulaBase): The copula to instantiate
            marginal_classes (list[UnivariateDistribution] |
                dict[name,UnivariateDistribution]): The marginals to
                    instantiate. This must be a dictionary if dimkeys be passed
                    in
            dimkeys (List[name]): The names of the dimensions should data be
                a dictionary
        Returns:
            CopulaWithMarginals: The fitted distribution
        """
        if dimkeys is not None:
            input_data = [input_data[key] for key in dimkeys]
            marginal_classes = [marginal_classes[key] for key in dimkeys]

        marginals = []

        # This will contain vectors of the cdf applied to data along each
        # dimension.
        uss = []

        for vector, marginal_class in zip(input_data, marginal_classes):
            marginal = marginal_class.fit(vector)
            marginals.append(marginal)
            uss.append([marginal.cdf(x) for x in vector])

        #print('uss:',uss)

        copula = copula_class.fit(uss)

        return CopulaWithMarginals(copula, marginals, dimkeys)

    def _invert_bounds(self, hyperrectangle):
        """
        This is a convenience function to invert the bounds of the
        hyperrectangle from the space [0, 1]^n to R^n.

        Args:
            hyperrectangle (Hyperrectangle): A hyperrectangle with bounds in
                [0, 1]^n which have the same dimension names as used to
                construct the distribution
        Returns: Two dictionaries, lowers ({name -> lower_bound}) and
            uppers ({name -> upper_bounds})
        """
        lowers = {}
        uppers = {}
        for name in self.dimkeys:
            lower, upper = hyperrectangle.bounds[name]
            marginal = self.marginals[name]
            lowers[name] = marginal.cdf_inverse(lower)
            uppers[name] = marginal.cdf_inverse(upper)
        return lowers, uppers

    def c_log_likelihood(self, data):
        res = 0
        n = len(data[0])
        for i in range(n):

            temp = np.zeros(self.dimension)
            for j in range(self.dimension):
                temp[j] = self.marginals[self.dimkeys[j]].cdf(data[j, i])

            res = res + np.log(self.copula.c(*temp))

        return res/n


@register_distribution(name='gaussian-copula')
class GaussianCopula(CopulaBase):
    """
    This class will create a Gaussian Copula.
    """

    def __init__(self, R, dimkeys=None):
        """
        Args:
            R (np.array): A nxn correlation matrix
        """
        ndim, _ = R.shape
        CopulaBase.__init__(self, ndim, dimkeys)
        self.cor_matrix = R
        self.cor_matrix_inv = np.linalg.inv(self.cor_matrix)
        self.det_cor = np.linalg.det(self.cor_matrix)
        self.I = np.identity(self.dimension)

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        """
        This function will fit a Gaussian Copula to the data
        passed in. The data passed in may be either a list of lists or it may
        be a dictionary mapping dimension names to lists of data. In the
        second case, a list of dimension names must be passed as well in order
        to determine the ordering of the dimensions.

        Args:
            data (List[List[float]] | dict[name,List[float]]): The data to fit
                the distribution to
            dimkeys (List[name]): The names of the dimensions should data be
                a dictionary
        Returns:
            GaussianCopula: The fitted distribution
        """
        cor_matrix = np.corrcoef(data)
        
        return GaussianCopula(cor_matrix, dimkeys)

    @accepts_dict
    def C_partial_derivative(self, u, v):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order

        Args:
            valuedict (dict[key, float]): the value where you want to compute
        Returns:
            (float): dC(u, v)/dv in fact this should be equal to F(u|v) if the
            marginals are uniform variables in [0, 1]
        """
        if self.dimension != 2:
            raise ("Partial derivative only for 2 dimension copulas")

        rho = self.cor_matrix[0][1]
        return sps.ndtr((sps.ndtri(u)-rho*sps.ndtri(v))/self.det_cor**0.5)

    @accepts_dict
    def inverse_C_partial_derivative(self, u, v):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        rho = self.cor_matrix[0][1]
        return sps.ndtr(self.det_cor**0.5*sps.ndtri(u)+rho*sps.ndtri(v))

    @accepts_dict
    def c(self, *x):
        """
        This function computes the pdf of the copula. It accepts a variable
        amount of arguments to support any dimension of copula.

        Args:
            x (List[float]) the points where you want to compute
        Returns:
            float: The density of the copula
        """
        vect = sps.ndtri(x)
        intermediate = np.dot(np.dot(vect.T,(self.cor_matrix_inv - self.I)),
                              vect)
        res = np.exp(-intermediate / 2) / np.sqrt(self.det_cor)
        if np.isnan(res):
            res = 0
        return res

    @accepts_dict
    def C(self, *xs):
        """
        Args:
            value: the points where you want to calculate the gaussian copula
        Returns:
            the copula in tab, with a precision of quadstep.
        """
        bounds = []
        for x in xs:
            # specific case where 0 is an argument, so the result is 0.
            if x <= 0:
                return 0
            bounds.append([0, x])
        res, _ = spi.nquad(self.c, bounds, opts={'epsabs': 1e-3})
        return res

    def rect_prob_2(self, low, up):
        """
        Seems to work now even thaugh it used to give an uncorrect answer equal
        to 0.16.. every 3 or 4 times

        Seems much longer than classical rect_prob: rect_prob2 is useless ?
        """

        if isinstance(low, dict):
            low = [low[i] for i in self.dimkeys]
        if isinstance(up, dict):
            up = [up[i] for i in self.dimkeys]
        bounds = []
        for k in range(self.dimension):
            bounds.append([low[k], up[k]])

        res = spi.nquad(self._rect_integrand, bounds,
                        opts={'epsabs': self.quadstep})
        return res[0]

    def generates_U(self, n=1):
        X = np.random.multivariate_normal(np.zeros(self.dimension),
                                          self.cor_matrix, n)
        res = sps.ndtr(X)
        return res


@register_distribution(name='student-copula')
class StudentCopula(CopulaBase):
    """
    This class will create a Student Copula.
    The Student Copula is both lower and upper tail dependent.
    """

    def __init__(self, cov, df=5, dimkeys=None):
        """
        Args:
            dimkeys (list): keys for each dimension in dictin
            df (int): degrees of freedom of the the student copula
        """

        cov = np.array(cov)

        ndim, _ = cov.shape

        CopulaBase.__init__(self, ndim, dimkeys)

        self.cor_matrix = cov

        self.df = df

        self.cor_matrix_inv = np.linalg.inv(self.cor_matrix)
        self.det_cor_matrix = np.linalg.det(self.cor_matrix)

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        R = np.corrcoef(data)
        return StudentCopula(R, dimkeys=dimkeys)


    def _t(self, x):
        return t.cdf(x, df=self.df)

    def _inverse_t(self, x):
        return t.ppf(x, df=self.df)

    @accepts_dict
    def _integrand_for_C(self, *xs):
        """
        Args:
            *tab(list of size n): arguments of the density function
            u and v are for the bivariate case that we call with vine copulas.
        Returns:
            the integrand which is useful for calculating C
        """
        d = self.dimension
        nu = self.df
        n = len(self.dimkeys)

        R = self.cor_matrix
        R_inv = self.cor_matrix_inv

        factor_1 = sps.gamma(self.df / 2)
        factor_2 = np.sqrt((np.pi*self.df)**n * np.linalg.det(self.cor_matrix))
        denom = factor_1 * factor_2
        numer = sps.gamma((self.df + n) / 2)
        factor = numer / denom
        base = 1 + (np.dot(xs, np.dot(R_inv, np.transpose(xs)))) / self.df
        exponent = -(self.df + n) / 2

        res = factor * base ** exponent

        return res

    @accepts_dict
    def C_partial_derivative(self, u, v):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order
        Args:
            valuedict: the value where you want to compute
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v) if the
                marginals are uniform variables in [0, 1]
        """
        if self.dimension != 2:
            raise ("Partial derivative only for 2 dimension copulas")

        rho = self.cor_matrix[0][1]
        temp =((self.df+self._inverse_t(v)**2)*(1-rho**2)/(self.df+1))**0.5

        return self._t((self._inverse_t(u)-rho*self._inverse_t(v))/temp)

    @accepts_dict
    def inverse_C_partial_derivative(self, u, v):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        if self.dimension != 2:
            raise ("Partial derivative only for 2 dimension copulas")

        rho = self.cor_matrix[0][1]
        temp = ((self.df + self._inverse_t(v) ** 2) * (1 - rho ** 2) / (self.df + 1)) ** 0.5
        return self._t(temp*self._inverse_t(u)+rho*self._inverse_t(v))

    @accepts_dict
    def C(self, *xs):
        """
        Args:
            valuedict: the points where you want to calculate the gaussian copula
        Returns:
            the copula in tab, with a precision of 0.1%
        """

        bounds = []
        for x in xs:
            if x <= 0:
                return 0
            if x > 1:
                valuedict[k] = 1
            bounds.append([-np.inf, t.ppf(x, self.df)])
        res = spi.nquad(self._integrand_for_C, bounds, opts={'epsabs': 0.001})

        return res[0]

    @accepts_dict
    def c(self, *xs):
        """
        Args:
            *tab(list of size n): arguments of the density function
            u and v are for the bivariate case that we call with vine copulas.
        Returns:
            the density of the gaussian copula
        """
        d = self.dimension
        nu = self.df
        factor = sps.gamma((nu + d) / 2) * sps.gamma(nu / 2) ** (d - 1) / (
            sps.gamma((nu + 1) / 2) ** d * np.sqrt(self.det_cor_matrix))
        temp = []
        for x in xs:
            temp.append(self._inverse_t(x))

        for j in range(d):
            factor = factor / (1 + temp[j] ** 2 / nu) ** (-(nu + 1) / 2)

        res = factor * (1 + np.dot(np.transpose(temp), np.dot(self.cor_matrix_inv, temp))/nu) ** (-(nu + d) / 2)
        return res


    def generates_U(self, n=1):
        distr = MultiStudentDistribution(cov=self.cor_matrix,
                                     mean=[0]*self.dimension, df=self.df)
        X = distr.generates_X(n)
        U = np.zeros((n, self.dimension))
        # It is better to use numpy arrays
        # Because we can call the columns U[:,j].
        for i in range(n):
            temp = np.zeros(self.dimension)
            for j in range(self.dimension):
                temp[j] = self._t(X[i][j])
            U[i]= temp
        return U


@register_distribution(name='frank-copula', ndim=2)
class FrankCopula(CopulaBase):
    """
    This class will create the C function for Frank Copula.
    """
    def __init__(self, theta, dimkeys=None):
        """
        Args:
            dimkeys (list): keys for each dimension in dictin (e.g. a list of ints)
            theta (float): parameter of the copula, optional
        """

        self.theta = theta

        if self.theta==0:
            raise ValueError("Invalid parameter for Frank Copula: theta=0")

        CopulaBase.__init__(self, 2, dimkeys)

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        """
        This function will fit a Frank copula to the data
        passed in. The data passed in may be either a list of lists or it may
        be a dictionary mapping dimension names to lists of data. In the
        second case, a list of dimension names must be passed as well in order
        to determine the ordering of the dimensions.

        Args:
            data (List[List[float]] | dict[name,List[float]]): The data to fit
                the distribution to
            dimkeys (List[name]): The names of the dimensions should data be
                a dictionary
        Returns:
            FrankCopula: The fitted distribution
        """
        X = np.array(data[0])
        Y = np.array(data[1])

        if not ((X.ndim == 1) and (Y.ndim == 1)):
            raise ValueError('The dimension of array should be one.')

        # input arrays should have same size
        if X.size != Y.size:
            raise ValueError('The size of both array should be same.')


        # estimate Kendall'rank correlation
        tau, _ = kendalltau(X, Y)

        # estimate the parameter of copula
        theta = -fmin(cls._frank_fun, -5, args=(tau,), disp=False)[0]

        return FrankCopula(theta, dimkeys)

    @classmethod
    def _frank_fun(cls, alpha, tau):
        """
        Optimization of this function will give the parameter for the frank
        copula
        """
        diff = (1 - tau) / 4.0 - (cls._debye(-alpha) - 1) / alpha
        return diff ** 2

    @classmethod
    def _integrand_debye(cls, t):
        """
        Integrand for the first order debye function
        """
        return t / (np.exp(t) - 1)

    @classmethod
    def _debye(cls, alpha):
        """
        First order Debye function
        """
        value, _ = quad(cls._integrand_debye, sys.float_info.epsilon, alpha)
        return  value / alpha

    def generates_U(self, n=10000):
        """
        Generate random variables (u, v)
        Args:
            n: number of random copula to be generated
        Returns:
            U and V: generated copula
        """

        U = np.random.uniform(size=n)
        W = np.random.uniform(size=n)

        if abs(self.theta) > np.log(sys.float_info.max):
            V = (U < 0) + np.sign(self.theta) * U
        elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
            V = -np.log((np.exp(-self.theta * U) * (1 - W) / W + np.exp(-self.theta)
                         ) / (1 + np.exp(-self.theta * U) * (1 - W) / W)) / self.theta
        else:
            V = W

        res = np.transpose([U, V])
        return res

    @accepts_dict
    def C_partial_derivative(self, u, v):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order
        Args:
            valuedict: the value where you want to compute
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v)
                if the marginals are uniform variables in [0, 1]
        """

        if not(valuedict is None):
           u = valuedict.get(self.dimkeys[0])
           v = valuedict.get(self.dimkeys[1])

        numer = np.exp(self.theta * u) - 1

        denom = (np.exp(self.theta * u)
                 + np.exp(self.theta * v)
                 - np.exp(self.theta * (u+v-1)) - 1)

        return numer / denom

    @accepts_dict
    def inverse_C_partial_derivative(self, u, v):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        a = 1
        b = -1
        c = 1-np.exp(self.theta*(v-1))
        d = np.exp(self.theta*v)-1
        return np.log((d*u-b)/(a-c*u))/self.theta

    @accepts_dict
    def C(self, u, v):
        """
        Args:
            valuedict(dict): the values where you want to compute C
        Returns:
            float: the function C thanks to the C_from_sample function
        """


        C_from_calcul = -1/self.theta*np.log(1+((np.exp(-self.theta*u)-1)*(np.exp(-self.theta*v)-1)/(np.exp(-self.theta)-1)))
        return C_from_calcul

    @accepts_dict
    def c(self, u, v):
        """
        Args:
            valuedict: The value where you want to compute
                you can also directly use u and v
        Returns:
            float: The density of the copula: ie c(u, v) = d²C/dudv (u, v)
        """
        temp = np.exp(self.theta * u) + np.exp(self.theta * v) - np.exp(self.theta * (u + v - 1)) - 1
        return self.theta*np.exp(self.theta*(u+v))*(1-np.exp(-self.theta))/temp**2


@register_distribution(name='clayton-copula', ndim=2)
class ClaytonCopula(CopulaBase):
    """
    This class will create the C function for Clayton Copula.
    The Clayton Copula is lower tail dependant but not upper.
    Be careful to test it with no independent datas,
    because independent datas can cause troubles when estimating the
    theta coeff.
    """

    def __init__(self, theta, dimkeys):
        """
        Args:
            theta (float): parameter of the copula
            dimkeys (list): keys for each dimension in dictin
        """

        CopulaBase.__init__(self, 2, dimkeys)

        self.theta = theta

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        """
        This function will fit a Clayton Copula to the data
        passed in. The data passed in may be either a list of lists or it may
        be a dictionary mapping dimension names to lists of data. In the
        second case, a list of dimension names must be passed as well in order
        to determine the ordering of the dimensions.

        This will calibrate the parameter theta using the Kendall's Tau of the
        passed in data.

        Args:
            data (List[List[float]] | dict[name,List[float]]): The data to fit
                the distribution to
            dimkeys (List[name]): The names of the dimensions should data be
                a dictionary
        Returns:
            ClaytonCopula: The fitted distribution
        """
        if len(data) != 2:
            raise ValueError("ClaytonCopula only implemented for 2 dimensions")
        X = np.array(data[0])
        Y = np.array(data[1])

        # input array should have the same size
        if X.size != Y.size:
            raise ValueError('The size of both array should be same.')

        # estimate Kendall'rank correlation
        tau, _ = kendalltau(X, Y)

        # estimate the parameter of copula
        theta = 2 * tau / (1 - tau)
        if theta == 0:
            raise ValueError("Invalid parameter for Clayton Copula: theta=0")
        if theta < -1:
            raise ValueError("Invalid parameter for Clayton Copula: theta<-1")

        return ClaytonCopula(theta, dimkeys)

    def generates_U(self, n=1000):
        """
        Generate random variables (u, v)
        Args:
            n: number of random copula to be generated
        Returns:
            U and V:  generated copula
        """

        U = np.random.uniform(size=n)
        W = np.random.uniform(size=n)

        if self.theta < sys.float_info.epsilon:
            V = W
        else:
            V = U * (W ** (-self.theta / (1 + self.theta)) - 1 + U ** self.theta) ** (-1 / self.theta)

        res = np.transpose([U, V])

        return res

    @accepts_dict
    def C_partial_derivative(self, u, v):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order
        Args:
            valuedict: the value where you want to compute. You can also call
                this function with u and v directly
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v)
                if the marginals are uniform variables in [0, 1]
        """
        temp = u**(-self.theta)+v**(-self.theta)-1
        if temp>=0:
            return v**(-self.theta-1)*temp**(-(1/self.theta)-1)
        else:
            return 0

    @accepts_dict
    def inverse_C_partial_derivative(self, u, v):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        theta = self.theta

        base = u**(-theta/(theta+1)) * v ** -theta - v**(-theta) + 1

        return base**(-1/theta)

    @accepts_dict
    def C(self, u, v):
        """
        Args:
            valuedict(dict): the values where you want to compute C
        Returns:
            float: the function C thanks to the C_from_sample function
        """
        first = (u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1/self.theta)
        C_from_calcul = max(first, 0)
        return C_from_calcul

    @accepts_dict
    def c(self, u, v):
        """
        Args:
            valuedict: The value where you want to cumpute
                you can also directly use u and v
        Returns:
            The density of the copula: ie c(u, v) = d²C/dudv (u, v)
        """
        temp = u**(-self.theta)+v**(-self.theta)-1
        return (1+self.theta)*(u*v)**(-1-self.theta)*temp**(-1/self.theta-2)


@register_distribution(name='gumbel-copula', ndim=2)
class GumbelCopula(CopulaBase):
    """
    This class will create the C function for Gumbel Copula.
    The Gumbel Copula is upper tail dependent but not lower.
    Be careful to test it with no independent datas,
    because independent datas can cause troubles when estimating the
    theta coeff.
    """
    def __init__(self, theta, dimkeys=None):
        """
        Args:
            theta (float): parameter of the copula
            dimkeys (list): keys for each dimension in dictin
        """

        CopulaBase.__init__(self, 2, dimkeys)

        self.theta = theta

        if self.theta < 1:
            raise ValueError("Invalid parameter for Gumbel Copula: theta<1")

    @classmethod
    @fit_wrapper
    def fit(self, data, dimkeys=None):
        """
        This function will fit a Gumbel Copula to the data
        passed in. The data passed in may be either a list of lists or it may
        be a dictionary mapping dimension names to lists of data. In the
        second case, a list of dimension names must be passed as well in order
        to determine the ordering of the dimensions.

        Args:
            data (List[List[float]] | dict[name,List[float]]): The data to fit
                the distribution to
            dimkeys (List[name]): The names of the dimensions should data be
                a dictionary
        Returns:
            GumbelCopula: The fitted distribution
        """
        X = np.array(data[0])
        Y = np.array(data[1])

        if X.size != Y.size:
            raise ValueError('The size of both arrays should be same.')

        tau, _ = kendalltau(X, Y)

        # estimate the parameter of copula
        theta = max(1 / (1 - tau), 1)

        return GumbelCopula(theta, dimkeys)

    def generates_U(self, n=1000):
        """
        Generate random variables (u, v)
        Args:
            n: number of random copula to be generated
        Returns:
            U and V:  generated copula
        """

        if self.theta < 1 + sys.float_info.epsilon:
            U = np.random.uniform(size=n)
            V = np.random.uniform(size=n)
        else:
            u = np.random.uniform(size=n)
            w = np.random.uniform(size=n)
            w1 = np.random.uniform(size=n)
            w2 = np.random.uniform(size=n)

            u = (u - 0.5) * np.pi
            u2 = u + np.pi / 2;
            e = -np.log(w)
            t = np.cos(u - u2 / self.theta) / e
            gamma = (np.sin(u2 / self.theta) / t) ** (1 / self.theta) * t / np.cos(u)
            s1 = (-np.log(w1)) ** (1 / self.theta) / gamma
            s2 = (-np.log(w2)) ** (1 / self.theta) / gamma
            U = np.array(np.exp(-s1))
            V = np.array(np.exp(-s2))

        res = np.transpose([U, V])
        return res

    @accepts_dict
    def C_partial_derivative(self, u, v):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order
        Args:
            valuedict: the value where you want to compute
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v) if the
                marginals are uniform variables in [0, 1]
        """
        temp = (-np.log(u))**(self.theta)+(-np.log(v))**(self.theta)
        return (-np.log(v))**(self.theta-1)*temp**(1/self.theta-1)*np.exp(-temp**(1/self.theta))/v

    @accepts_dict
    def inverse_C_partial_derivative(self, u, v):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        W_arg = (v * u) ** (-1/(self.theta-1))*(-np.log(v)) / (self.theta-1)
        temp = np.real((self.theta-1)*sps.lambertw(W_arg))
        result = np.exp(-(temp**self.theta-(-np.log(v))**self.theta)**(1/self.theta))
        #The lambert W function is the inverse of x*exp(x).
        return result

    @accepts_dict
    def C(self, u, v):
        """
        Args:
            valuedict(dict): the values where you want to compute C
        Returns:
            float: the function C thanks to the C_from_sample function
        """
        theta = self.theta
        arg = -((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)

        C_from_calcul = np.exp(arg)
        return C_from_calcul

    @accepts_dict
    def c(self, u, v):
        """
        Args:
            valuedict: The value where you want to compute you can also
                directly use u and v
        Returns:
            float: The density of the copula: ie c(u, v) = d²C/dudv (u, v)
        """
        temp = (-np.log(u)) ** (self.theta) + (-np.log(v)) ** (self.theta)
        C_u_v = np.exp(-temp ** (1 / self.theta))
        return C_u_v/(u*v)*temp**(-2+2/self.theta)*(np.log(u)*np.log(v))**(self.theta-1)*(1+(self.theta-1)*temp**(-1/self.theta))


@register_distribution(name='weighted-combined-copula')
class WeightedCombinedCopula(CopulaBase):
    """
    This class will compute a weighted linear combination of copulas.
    """
    def __init__(self, input_data, copulas, weights, dimkeys=None):
        """
        Args:
            input_data (any): the raw data; given as lists for each dimension
        """
        self.length = len(copulas)
        self.copulas = []
        if self.length != len(weights):
            raise ValueError("The number of copulas and the number of "
                             "weights should be the same")

        # We first generates copulas thanks to the list of strings.
        # Each copula will be fitted with input_data
        for i in range(self.length):
            distr_class = distribution_factory(copulas[i])
            self.copulas.append(distr_class.fit(input_data))

        # We verify if the weights are positive and if their sum is equal to 1.
        if weights is not None:
            for i in range(self.length):
                if weights[i]<0:
                    raise ValueError("The weights should be positive")
        else:
            raise ValueError("Inference of weights not implemented yet"
                             " you should input your own weights")


        self.weights = np.asarray(weights)

        if sum(weights) != 1:
            print("Warning: the sum of the weight is different from one")
            print("So the weights will be normalized")
            self.weights = self.weights / sum

        CopulaBase.__init__(self, self.length, dimkeys)

    @classmethod
    @fit_wrapper
    def fit(cls, data, copulas, weights, dimkeys=None):
        return WeightedCombinationCopula(data, copulas, weights, dimkeys)

    def generates_U(self, n=1000):
        """
        Generate random variables (u, v)
        Args:
            n: number of random samples to be generated
        Returns:
            U and V:  generated copula
        """
        res = np.zeros(n)
        for weight, copula in zip(self.weights, self.copulas):
            res += weight * copula.generates_U(n)

        return res

    @accepts_dict
    def C(self, *xs):
        """
        Args:
            valuedict(dict): the values where you want to compute C
        Returns:
            the function C
        """
        res = 0
        for weight, copula in zip(self.weights, self.copulas):
            res += weight * copula.C(*xs)
        return res

    @accepts_dict
    def c(self, *xs):
        """
        Args:
            valuedict: The value where you want to cumpute
                you can also directly use u and v
        Returns:
            float: The density of the copula: ie c(u, v) = d²C/dudv (u, v)
        """
        res = 0
        for weight, copula in zip(self.weights, self.copulas):
            res += weight * copula.c(*xs)
        return res

    @accepts_dict
    def C_partial_derivative(self, *xs):
        """
        Args:
            valuedict: the value where you want to compute
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v) if the
                marginals are uniform variables in [0, 1]
        """
        res = 0
        for weight, copula in zip(self.weights, self.copulas):
            res += weight * copula.C_partial_derivative(*xs)
        return res

    @accepts_dict
    def C_inverse_partial_derivative(self, *xs):
        raise NotImplementedError


@register_distribution(name='independence-copula')
class IndependenceCopula(CopulaBase):
    """
    This class will compute a simple independence copula with a uniform density.
    """
    def __init__(self, dimension, dimkeys=None):
        CopulaBase.__init__(self, dimension, dimkeys)

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        return IndependenceCopula(len(data), dimkeys)

    @accepts_dict
    def c(self, *xs):
        return 1

    @accepts_dict
    def C(self, *xs):
        res = 1
        for x in xs:
            res *= x
        return res

    @accepts_dict
    def C_partial_derivative(self, *xs):
        """
        Since the roles of variable are symetric we can use this for both
        variables in any ways, if we are careful about the order
        Args:
            valuedict: the value where you want to compute
        Returns:
            float: dC(u, v)/dv in fact this should be equal to F(u|v)
                if the marginals are uniform variables in [0, 1]
        """
        return xs[0]

    @accepts_dict
    def inverse_C_partial_derivative(self, *xs):
        """
        This function is the "inverse" of C_partial derivate
        If we call this function G, G(.|v) is the inverse of F(.|v):
        It is only the inverse considering u.
        One can also call this function with u and v or with valuedict.
        """
        return xs[0]

    def generates_U(self, n=1):
        res = np.random.rand(n, self.dimension)

        return res


@register_distribution(name='empirical-copula')
class EmpiricalCopula(CopulaBase):
    def __init__(self, input_data, dimkeys=None, marginals=None):
        """
        Args:
            dimkeys (list): keys for each dimension in dictin
            input_data (any): the raw data; given as lists for each dimension
            marginals (dict): dictionary of marginals
        """
        ndim = len(input_data)
        if marginals is None:
            marginals = [UnivariateEmpiricalDistribution(input_data[i]) for i in range(ndim)]

        # n is the number of data points used to fit the distribution
        self.n = len(input_data[0])

        self.V = input_data.copy()

        for i, j in zip(range(ndim), range(self.n)):
            self.V[i][j] = marginals[i].cdf(input_data[i][j])

        CopulaBase.__init__(self, ndim, dimkeys)

    @classmethod
    @fit_wrapper
    def fit(cls, data, marginals=None, dimkeys=None):
        return EmpiricalCopula(data, marginals, dimkeys)

    @accepts_dict
    def C(self, *xs):
        count = 0
        for j in range(self.n):
            if all(self.V[i][j] <= xs[i] for i in range(self.dimension)):
                count += 1
        return count / self.n

    def generates_U(self, n=1):
        l = len(self.V[self.dimkeys[0]])
        res = np.zeros((n, self.dimension))

        for i in range(n):
            # We choose randomly one of the elements of V.
            k = np.random.random_integers(l-1)
            for j in range(self.dimension):
                #Then res[i] = this chosen value of V.
                res[i][j] = self.V[self.dimkeys[j]][k]

        return res
