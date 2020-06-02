#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from timer import Timer,tic,toc
import unittest
from copula import GaussianCopula,FrankCopula,GumbelCopula,ClaytonCopula,StudentCopula, WeightedCombinedCopula
import numpy as np
import scipy
import scipy.integrate as spi
import scipy.special as sps
import scipy.stats as spst
from base_distribution import BaseDistribution,MultiDistr
from distributions import UnivariateEmpiricalDistribution, UnivariateEpiSplineDistribution
from distributions import UnivariateNormalDistribution,MultiNormalDistribution,UnivariateStudentDistribution, MultiStudentDistribution
from vine import CVineCopula,DVineCopula
import matplotlib.pyplot as plt
import copula_experiments
from copula_experiments.copula_diagonal import diag
from copula_experiments.copula_evaluate import RankHistogram,emd_sort,emd_pyomo
from distribution_factory import distribution_factory

class EmpiricalDistributionTester(unittest.TestCase):
    def setUp(self):
        points = [1, 1, 2, 2, 3, 5, 6, 8, 9]
        self.distribution = UnivariateEmpiricalDistribution(points)

    def test_at_point(self):
        self.assertAlmostEqual(self.distribution.cdf(1), 2 / 10)
        self.assertAlmostEqual(self.distribution.cdf_inverse(2 / 10), 1)

    def test_before_first(self):
        self.assertAlmostEqual(self.distribution.cdf(0.5), 1 / 10)
        self.assertAlmostEqual(self.distribution.cdf_inverse(1 / 10), 0.5)

    def test_far_before_first(self):
        self.assertEqual(self.distribution.cdf(-4), 0)

    def test_between_points(self):
        self.assertAlmostEqual(self.distribution.cdf(4), 11 / 20)
        self.assertAlmostEqual(self.distribution.cdf_inverse(11 / 20), 4)

    def test_after_end(self):
        self.assertAlmostEqual(self.distribution.cdf(9.5), 19 / 20)
        self.assertAlmostEqual(self.distribution.cdf_inverse(19 / 20), 9.5)

    def test_far_after_end(self):
        self.assertAlmostEqual(self.distribution.cdf(20), 1)


class EpisplineTester(unittest.TestCase):
    def setUp(self):
        input_data = np.random.randn(1000)
        self.distribution = UnivariateEpiSplineDistribution(input_data)

    def test_cdf_values(self):
        self.assertAlmostEqual(self.distribution.cdf(self.distribution.alpha), 0)
        self.assertAlmostEqual(self.distribution.cdf(self.distribution.alpha - 100), 0)
        self.assertAlmostEqual(self.distribution.cdf(self.distribution.beta), 1)
        self.assertAlmostEqual(self.distribution.cdf(self.distribution.beta + 100), 1)

    def test_region_probability(self):
        # Tests the region probability by asserting the disjoint union of all regions must add up to 1
        midpoint = (self.distribution.alpha + self.distribution.beta) / 2

        integral_value = (self.distribution.region_probability((self.distribution.alpha, midpoint))
                          + self.distribution.region_probability((midpoint, self.distribution.beta)))
        self.assertAlmostEqual(integral_value, 1)

        one_third_way = (2*self.distribution.alpha + self.distribution.beta) / 3
        two_thirds_way = (self.distribution.alpha + 2*self.distribution.beta) / 3
        integral_value = (self.distribution.region_probability((self.distribution.alpha, one_third_way))
                          + self.distribution.region_probability((one_third_way, two_thirds_way))
                          + self.distribution.region_probability((two_thirds_way, self.distribution.beta)))

        self.assertAlmostEqual(integral_value, 1)

    def test_quick(self):
        print('Warning : this code must be called with runner.py')
        # Copy this code at the beginning of copula_test to see if it works
        # And enter python3 runner.py copula_experiments/run_test.txt
        gosm_options.set_globals()

        # Create output directory.
        if not (os.path.isdir(gosm_options.output_directory)):
            os.mkdir(gosm_options.output_directory)

        X = np.arange(300)
        tic()
        mydistr = UnivariateEpiSplineDistribution(X)
        for i in range(10):
            print(mydistr.cdf(i))
        toc()


class UnivariateNormalDistributionTester(unittest.TestCase):
    def test_quick(self):
        data = np.random.randn(1000)
        dist = UnivariateNormalDistribution(input_data=data)
        self.assertAlmostEqual(dist.rect_prob(-1.96,1.96),0.95,1)

    def test_pdf_cdf(self):
        x = -2 + 2 * np.random.randn(2000)
        mydistr = UnivariateNormalDistribution(input_data=x)
        res, i = spi.quad(mydistr.pdf, -1, 3)
        self.assertAlmostEqual(res,mydistr.rect_prob(-1, 3),5)

    def test_with_mean_var(self):
        sigma = 2
        mean = 3
        data = sigma*np.random.randn(10000)+mean
        dist = UnivariateNormalDistribution(input_data=data)
        self.assertAlmostEqual(dist.cdf(4),0.6915,1)
        dist = UnivariateNormalDistribution(mean = mean,var=sigma**2)
        self.assertAlmostEqual(dist.cdf(4),0.6915,3)

class MultiNormalDistributionTester(unittest.TestCase):
    def test_two_dimensions(self):
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)
        ourmean = [-4, 3]
        ourcov = [[2, 0], [0, 2]]
        lowerdict = {"solar": -1, "wind": 0}
        upperdict = {"solar": 3, "wind": 4}
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)
        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]
        dist = MultiNormalDistribution(dimkeys,input_data=data_dict)
        dist2 = MultiNormalDistribution(dimkeys,mean=ourmean,cov=ourcov)
        self.assertAlmostEqual(dist.rect_prob(lowerdict,upperdict),dist2.rect_prob(lowerdict,upperdict),2)
        self.assertAlmostEqual(np.mean(dist.generates_X(n=1000)[:,1]),ourmean[1],1)
        self.assertAlmostEqual(np.mean(dist.generates_X(n=1000)[:, 0]), ourmean[0], 1)

    def test_with_gaussian_copula_1_dim(self):
        mymean = 0
        myvar = 2
        dimkeys1 = ["solar"]
        lowerdict = {"solar": -2}
        upperdict = {"solar": 1}
        data_array1 = np.random.multivariate_normal([mymean], [[myvar]], 10000)
        data_dict1 = {"solar": data_array1[:, 0]}
        marginals1 = {"solar": UnivariateNormalDistribution(input_data=data_array1[:, 0])}
        unigaussian1 = GaussianCopula(input_data=data_dict1, dimkeys=dimkeys1, marginals=marginals1)
        unigaussian2 = MultiNormalDistribution(dimkeys1, input_data=data_dict1)
        self.assertAlmostEqual(unigaussian1.rect_prob(lowerdict, upperdict),unigaussian2.rect_prob(lowerdict, upperdict),3)

    def test_with_gaussian_copula_2_dim(self):
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)
        ourmean = [3, 4]
        ourmeandict = {"solar": 0, "wind": 0}
        ourcov = [[1, 0.5], [0.5, 1]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}
        valuedict = {"solar": 0, "wind": 0}
        lowerdict = {"solar": 2, "wind": 3}
        upperdict = {"solar": 4, "wind": 5}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 100000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        multigaussian1 = GaussianCopula(input_data=data_dict, dimkeys=dimkeys, marginals=marginals, quadstep=0.001)
        multigaussian2 = MultiNormalDistribution(dimkeys, input_data=data_dict)
        valuedict = {"solar": 0.45, "wind": 0.89}
        self.assertAlmostEqual(multigaussian1.rect_prob(lowerdict, upperdict),
                               multigaussian2.rect_prob(lowerdict, upperdict), 3)

    def test_with_gaussian_copula_3_dim(self):
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)
        # dictin = {"solar": np.random.randn(200), "wind": np.random.randn(200)}

        ourmean = [0, 0, 0]
        ourcov = [[1, 0.1, 0.3], [0.1, 2, 0], [0.3, 0, 3]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}
        valuedict = {"solar": 0, "wind": 0, "tide": 0}
        lowerdict = {"solar": -1, "wind": -1, "tide": -1}
        upperdict = {"solar": 1, "wind": 1, "tide": 1}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 1000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            GaussianCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
            data_dict[dimkeys[i]] = data_array[:, i]

        multigaussian1 = GaussianCopula(input_data=data_dict, dimkeys=dimkeys, marginals=marginals, quadstep=0.1)
        multigaussian2 = MultiNormalDistribution(dimkeys, input_data=data_dict)
        self.assertAlmostEqual(multigaussian1.rect_prob(lowerdict, upperdict),
                               multigaussian2.rect_prob(lowerdict, upperdict), 2)

        self.assertAlmostEqual(multigaussian1.rect_prob(lowerdict, upperdict),multigaussian2.rect_prob(lowerdict, upperdict), 1)

class UnivariateStudentDistributionTester(unittest.TestCase):
    def test_pdf_cdf(self):
        x = -2 + 2 * np.random.randn(2000)
        student = UnivariateStudentDistribution(input_data=x)
        res, i = spi.quad(student.pdf, -1, 3)
        self.assertAlmostEqual(res,student.rect_prob(-1, 3),5)

    def test_in_student_copula_cdf(self):
        dimkeys = ["solar", "wind"]
        x = np.random.randn(2000)
        dictin = {"solar": x, "wind": x + np.random.randn(2000)}
        student = StudentCopula(dimkeys, dictin)
        self.assertAlmostEqual(student._t(student._inverse_t(0.1)),0.1,7)
        self.assertAlmostEqual(student._inverse_t(student._t(-6)),-6,7)

class MultiStudentDistributionTester(unittest.TestCase):
    def test_generates_X(self):
        x = np.random.randn(200)
        dictin = {"solar": x, "wind": x + 0.5 * np.random.randn(200)}

        dimkeys = ["solar", "wind"]
        mydistr = MultiStudentDistribution(dictin)
        print(mydistr.generates_X(10))


def initialize(dim=2,precision = None,copula_string='independence-copula'):
    if dim==1:
        mymean = 0
        myvar = 2
        dimkeys = ["solar"]
        data_array = np.random.multivariate_normal([mymean], [[myvar]], 1000)
        dictin = {"solar": data_array[:, 0]}
        distr_class = distribution_factory(copula_string)
        mydistr = distr_class(dimkeys, dictin)

        return mydistr

    if dim==2:
        # For some tests, gaussian and student are less precised so we change so precision asked :

        dimkeys = ["solar", "wind"]
        ourmean = [3, 4]
        rho=0.5
        ourcov = [[1, rho], [rho, 1]]
        data_array = np.random.multivariate_normal(ourmean, ourcov, 1000)
        dictin = dict.fromkeys(dimkeys)

        for i in range(dim):
            dictin[dimkeys[i]] = data_array[:, i]

        valuedict = {"solar": 0.14, "wind": 0.49}
        distr_class = distribution_factory(copula_string)
        mydistr = distr_class(dimkeys, dictin)

        return mydistr

    if dim==3:
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)
        # dictin = {"solar": np.random.randn(200), "wind": np.random.randn(200)}

        ourmean = [0, 0, 0]
        rho01 = 0.1
        rho02 = 0.3
        rho12 = 0
        ourcov = [[1, rho01, rho02], [rho01, 2, rho12], [rho02, rho12, 3]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 1000)
        dictin = dict.fromkeys(dimkeys)

        for i in range(dimension):
            dictin[dimkeys[i]] = data_array[:, i]

        distr_class = distribution_factory(copula_string)
        mydistr = distr_class(dimkeys, dictin)

        return mydistr


class CopulaTester(unittest.TestCase):

    def test_quick(self,copula_string='independence-copula'):
        mydistr = initialize(copula_string=copula_string)

        valuedict = {"solar": 0.05, "wind": 0.12}
        valuedict = {"solar": 1, "wind": 0.34}
        self.assertAlmostEqual(mydistr.C(valuedict),0.34,3)
        valuedict = {"solar": 0.47, "wind": 1}
        self.assertAlmostEqual(mydistr.C(valuedict), 0.47,3)

    def test_C_with_sample(self,copula_string='independence-copula',dim=2):
        if dim==2:
            mydistr = initialize(copula_string=copula_string, dim=2)
            valuedict = {"solar": 0.05, "wind": 0.12}
            self.assertAlmostEqual(mydistr.C(valuedict),mydistr.C_from_sample(valuedict),2)
        if dim==3:
            if copula_string=='frank-copula'or copula_string=='clayton-copula' or copula_string=='gumbel-copula':
                print('3d not implemented for archimedian copulas')
            else:
                mydistr = initialize(copula_string=copula_string, dim=3)
                valuedict = {"solar": 0.12, "wind": 0.23, "tide": 0.31}
                self.assertAlmostEqual(mydistr.C_from_sample(valuedict, 1000), mydistr.C(valuedict), 1)

    def test_partial_derivative_C(self,copula_string='independence-copula'):
        """
                In this test, we check if the partial derivative is correct by integrating it
                and comparing the integral with the initial function.
        """
        valuedict = {"solar": 0.67, "wind": 0.82}
        mydistr= initialize(copula_string=copula_string)

        if copula_string=='student-copula':
            precision = 2
        elif copula_string=='gaussian-copula':
            precision = 4
        else:
            precision = 7

        def g(x):
            return mydistr.C_partial_derivative(u=valuedict.get("solar"),v=x)

        res,i= spi.quad(g,0,valuedict.get("wind"))
        self.assertAlmostEqual(mydistr.C(valuedict), res, precision)
        valuedict = {"solar": 0.14, "wind": 0.42}
        res, i = spi.quad(g, 0, valuedict.get("wind"))
        self.assertAlmostEqual(mydistr.C(valuedict), res, precision)

    def test_inverse_partial_C(self,copula_string='independence-copula'):
        """
            In this test, we check if the partial derivative f inverse is correct by computing
            f(inverse_f(x)) and inverse_f(f(x)) and checking if they are both equal to x.
        """
        valuedict = {"solar": 0.84, "wind": 0.17}
        mydistr = initialize(copula_string=copula_string)
        u = valuedict.get("solar")
        v = valuedict.get("wind")

        direct = mydistr.C_partial_derivative(valuedict=valuedict)
        inverse = mydistr.inverse_C_partial_derivative(valuedict=valuedict)
        self.assertAlmostEqual(u,mydistr.C_partial_derivative(u=inverse,v=v),8)
        self.assertAlmostEqual(u,mydistr.inverse_C_partial_derivative(u=direct,v=v),8)

    def test_c_with_C_2_dim(self,copula_string='independence-copula'):
        """
                In this test, we check if the partial derivative is correct by integrating it
                and comparing the integral with the initial function.
        """
        valuedict = {"solar": 0.34, "wind": 0.73}
        mydistr = initialize(copula_string=copula_string)

        def g(x,y):
            return mydistr.c(u=x,v=y)
        def low_bound(x):
            return 0

        def up_bound(x):
            return valuedict.get("wind")

        res,i= spi.dblquad(g,0,valuedict.get("solar"),low_bound,up_bound)
        self.assertAlmostEqual(mydistr.C(valuedict),res,4)
        valuedict = {"solar": 0.12, "wind": 0.21}
        res, i = spi.dblquad(g,0,valuedict.get("solar"),low_bound,up_bound)
        self.assertAlmostEqual(mydistr.C(valuedict), res,4)

    def test_c_with_partial_C_2_dim(self,copula_string='independence-copula'):
        """
        In this test, we check if the partial derivative is correct by integrating it
        and comparing the integral with the initial function.
        """

        mydistr = initialize(copula_string=copula_string)

        valuedict = {"solar": 0.14, "wind": 0.49}


        def g(x):
            return mydistr.c(u=x,v=valuedict.get("wind"))

        if copula_string=='student-copula':
            precision = 2
        else:
            precision = 6


        res,i= spi.quad(g,0,valuedict.get("solar"))
        self.assertAlmostEqual(mydistr.C_partial_derivative(valuedict),res,precision)
        valuedict = {"solar": 0.56, "wind": 0.37}
        res, i = spi.quad(g, 0, valuedict.get("solar"))
        self.assertAlmostEqual(mydistr.C_partial_derivative(valuedict), res,precision)

    def test_plot(self,copula_string='independence-copula',dim=2):
        if dim==2:

            mydistr = initialize(copula_string=copula_string,dim=dim)

            n = 30 #number of points you want to display
            U = mydistr.generates_U(n=n)
            diag2 = diag(2)
            for k in range(2):  # index of the diagonal where you want to project we do both
                plt.plot(U[:, 0], U[:, 1], 'go')
                plt.plot([diag2.list_of_diag[k][0][1], diag2.list_of_diag[k][1][1]], 'b')

                P = diag2.proj_vector(U,k)
                plt.plot(P[:, 0], P[:, 1], 'ro')
                plt.plot([U[:, 0], P[:, 0]], [U[:, 1], P[:, 1]], c='k')
                plt.show()
        if dim==3:
            if copula_string=='frank-copula'or copula_string=='clayton-copula' or copula_string=='gumbel-copula':
                print('Plot 3d not implemented for archimedian copulas')
            else:
                mydistr = initialize(dim=3,copula_string=copula_string)
                n = 20  # number of points to display
                U = mydistr.generates_U(n=n)
                d = 3
                diago = diag(d)
                P = []
                fig = plt.figure()
                center = 0.5 * np.ones(d)
                k = 2  # index of the diagonal where you want to project
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(U[:, 0], U[:, 1], U[:, 2], c='g', marker='o')
                for i in range(n):
                    P = diago.proj_vector(U[i], k)
                    ax.scatter(P[0, 0], P[0, 1], P[0, 2], c='r', marker='o')
                    ax.plot([U[i, 0], P[0, 0]], [U[i, 1], P[0, 1]], [U[i, 2], P[0, 2]], c='k')
                diagonal = diago.list_of_diag[k]
                ax.plot([diagonal[0][0], diagonal[1][0]], [diagonal[0][1], diagonal[1][1]],
                        [diagonal[0][2], diagonal[1][2]],
                        c='b')

                ax.set_xlabel(mydistr.dimkeys[0])
                ax.set_ylabel(mydistr.dimkeys[1])
                ax.set_zlabel(mydistr.dimkeys[2])

                plt.show()


class LogLikelihoodTester(unittest.TestCase):
    def test_gaussian_copula2d(self):
        n = 10000
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)
        ourmean = [2, 3]
        ourmeandict = {"solar": 0, "wind": 0}
        rho = 0.5
        rho2 = 0.7
        ourcov = [[1, rho], [rho, 1]]
        ourcov2 = [[1, rho2], [rho2, 1]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 100000)
        data_array2 = np.random.multivariate_normal(ourmean, ourcov2, 100000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        data_dict2 = dict.fromkeys(dimkeys)
        for i in range(dimension):
            data_dict2[dimkeys[i]] = data_array2[:, i]

        gumbel = GumbelCopula(dimkeys, data_dict, marginals)
        frank = FrankCopula(dimkeys, data_dict, marginals)
        clayton = ClaytonCopula(dimkeys, data_dict, marginals)
        student = StudentCopula(dimkeys, data_dict, marginals)

        multigaussian1 = GaussianCopula(dimkeys=dimkeys, input_data=data_dict, marginals=marginals, quadstep=0.001)
        multigaussian2 = GaussianCopula(dimkeys=dimkeys, input_data=data_dict, marginals=marginals, quadstep=0.001,
                                        cov=ourcov2)
        multigaussian3 = GaussianCopula(dimkeys=dimkeys, input_data=data_dict2, marginals=marginals, quadstep=0.001,
                                        cov=ourcov2)
        multigaussian4 = GaussianCopula(dimkeys=dimkeys, input_data=data_dict2, marginals=marginals, quadstep=0.001,
                                        cov=ourcov)


        l1=multigaussian1.c_log_likelihood()
        self.assertGreater(l1,multigaussian2.c_log_likelihood())
        self.assertGreater(multigaussian3.c_log_likelihood(),multigaussian4.c_log_likelihood())
        self.assertGreater(l1,gumbel.c_log_likelihood())
        self.assertGreater(l1, clayton.c_log_likelihood())
        self.assertGreater(l1, frank.c_log_likelihood())
        self.assertGreater(l1, student.c_log_likelihood())

    def test_weighted_combined_copula3d(self):
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)

        ourmean = [0, 0, 0]
        ourcov = [[1, 0.1, 0.3], [0.1, 2, 0], [0.3, 0, 3]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        copulas= ['student-copula', 'gaussian-copula']
        list_of_gaussian = ['gaussian-copula','gaussian-copula']
        list_of_student = ['student-copula','student-copula']
        weights =[0.12,0.88]
        mydistr = WeightedCombinedCopula(dimkeys,data_dict,marginals,copulas,weights)
        gaussian = GaussianCopula(dimkeys,data_dict,marginals)
        weightedgaussian = WeightedCombinedCopula(dimkeys,data_dict,marginals,list_of_gaussian,weights)
        weightedstudent = WeightedCombinedCopula(dimkeys, data_dict, marginals, list_of_student, weights)
        student = StudentCopula(dimkeys,data_dict,marginals)
        g = gaussian.c_log_likelihood()
        s = student.c_log_likelihood()
        m = mydistr.c_log_likelihood()
        self.assertAlmostEqual(weightedgaussian.c_log_likelihood(),g,7)
        self.assertAlmostEqual(weightedstudent.c_log_likelihood(),s,7)
        self.assertGreater(g,m)
        self.assertGreater(m,s)


class VineCopulaTester(unittest.TestCase):
    def test_quick_dim_2(self):
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)

        ourmean = [1, 0.5]
        ourcov = [[1, 0.3], [0.3, 2]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        pair_copulae_strings = [[None, 'student-copula'],
                                [None, None]]

        valuedict = {"solar": 0.96, "wind": 0.87}
        CVine = CVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        DVine = DVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        gaussiancopula = GaussianCopula(dimkeys,data_dict,marginals)
        gaussiancopula.c(valuedict)
        self.assertAlmostEqual(CVine.C(valuedict),DVine.C(valuedict),1)
        self.assertAlmostEqual(gaussiancopula.C(valuedict), DVine.C(valuedict), 1)
        self.assertAlmostEqual(CVine.C(valuedict), gaussiancopula.C(valuedict), 1)

    def test_quick_dim_3(self):
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)

        ourmean = [0, 0, 0]
        ourcov = [[1, 0.1, 0.3], [0.1, 2, 0], [0.3, 0, 3]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        pair_copulae_strings = [[None, 'student-copula', 'frank-copula'],
                                [None, None, 'clayton-copula'],
                                [None, None, None]]

        valuedict = {"solar": 0.43, "wind": 0.92, "tide": 0.27}

        print('CVine')
        CVine = CVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        print(CVine.C(valuedict=valuedict))
        print(CVine.c(valuedict))
        print('DVine')
        DVine = DVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        print(DVine.C(valuedict=valuedict))
        print(DVine.c(valuedict))

    def test_with_multinormal_3_dim(self):
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)
        ourmean = [0, 0, 0]
        ourcov = [[1, 0.1, 0.3], [0.1, 2, 0], [0.3, 0, 3]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}
        valuedict = {"solar": 0, "wind": 0, "tide": 0}
        lowerdict = {"solar": -3, "wind": -2.3, "tide": -0.9}
        upperdict = {"solar": 1, "wind": 1.4, "tide": 2.7}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        pair_copulae_strings = [[None, 'gaussian-copula', 'gaussian-copula'],
                               [None, None, 'gaussian-copula'],
                               [None, None, None]]

        with Timer('MultiNormal'):
            multigaussian = MultiNormalDistribution(dimkeys, input_data=data_dict)
            print(multigaussian.rect_prob(lowerdict, upperdict))
        cvine = CVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        with Timer('CVine rect_prob calculus'):
            print(cvine.rect_prob(lowerdict, upperdict))
        dvine = DVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        with Timer('DVine rect_prob calculus'):
            print(dvine.rect_prob(lowerdict, upperdict))

    def test_with_multinormal_4_dim(self):
        dimkeys = ["solar", "wind", "tide","geo"]
        dimension = len(dimkeys)
        ourmean = [0, 0, 0, 0]
        ourcov = [[1, 0.1, 0.3,0.4], [0.1, 2, 0,0], [0.3, 0, 3,0],[0.4,0,0,4]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2]),
                     "geo":UnivariateNormalDistribution(var=ourcov[3][3], mean=ourmean[3])}
        valuedict = {"solar": 0, "wind": 0, "tide": 0,"geo":0}
        lowerdict = {"solar": -1, "wind": -1, "tide": -1,"geo":-2}
        upperdict = {"solar": 1, "wind": 1, "tide": 1,"geo":2}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        pair_copulae_strings = [[None, 'gaussian-copula', 'gaussian-copula','gaussian-copula'],
                               [None, None, 'gaussian-copula','gaussian-copula'],
                               [None, None, None,'gaussian-copula'],
                               [None,None,None,None]]

        with Timer('MultiNormal'):
            multigaussian = MultiNormalDistribution(dimkeys, input_data=data_dict)
            print(multigaussian.rect_prob(lowerdict, upperdict))
        cvine = CVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        with Timer('CVine rect_prob calculus'):
            print(cvine.rect_prob(lowerdict, upperdict))
        dvine = DVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        with Timer('DVine rect_prob calculus'):
            print(dvine.rect_prob(lowerdict, upperdict))

    def test_plot(self):
        dimkeys = ["solar", "wind", "tide"]
        dimension = len(dimkeys)

        ourmean = [0, 0, 0]
        ourcov = [[1, 1.3, 1.2], [1.3, 2, 0], [1.2, 0, 1.5]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1]),
                     "tide": UnivariateNormalDistribution(var=ourcov[2][2], mean=ourmean[2])}
        data_array = np.random.multivariate_normal(ourmean, ourcov, 10000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        pair_copulae_strings = [[None, 'gaussian-copula', 'frank-copula'],
                                [None, None, 'gaussian-copula'],
                                [None, None, None]]

        valuedict = {"solar": 1, "wind": 1, "tide": 0.73}
        lowerdict = {"solar": -3, "wind": -2, "tide": 0}
        upperdict = {"solar": 0.5, "wind": 1, "tide": 1}

        mydistr = DVineCopula(dimkeys, data_dict, marginals, pair_copulae_strings)
        n = 20      #number of points to display
        U = mydistr.generates_U(n=n)
        d = 3
        diago = diag(d)
        P =[]
        fig = plt.figure()
        center = 0.5*np.ones(d)
        k = 2 #index of the diagonal where you want to project
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(U[:, 0], U[:, 1], U[:, 2], c='g', marker='o')
        for i in range(n):
            P = diago.proj(U[i],k)
            ax.scatter(P[0,0],P[0,1],P[0,2], c='r', marker='o')
            ax.plot([U[i,0], P[0,0]],[U[i,1], P[0,1]],[U[i,2], P[0,2]], c='k')
        diagonal = diago.list_of_diag[k]
        ax.plot([diagonal[0][0],diagonal[1][0]], [diagonal[0][1],diagonal[1][1]],[diagonal[0][2],diagonal[1][2]], c='b')

        ax.set_xlabel(dimkeys[0])
        ax.set_ylabel(dimkeys[1])
        ax.set_zlabel(dimkeys[2])

        plt.show()

class RankHistogramTester(unittest.TestCase):
    def test_normal_distribution(self):
        mu = 0
        sigma = 1
        m = 10000
        mydistr = UnivariateNormalDistribution(0, 1)
        rank_data = mu + sigma * np.random.randn(10000)
        rank = RankHistogram(mydistr, rank_data, 25)
        rank.plot()

    def test_gaussian_copula(self):
        n = 10000
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)
        ourmean = [2, 3]
        ourmeandict = {"solar": 0, "wind": 0}
        rho =0.5
        rho2 = 0.5
        ourcov = [[1, rho], [rho, 1]]
        ourcov2 = [[1, rho2], [rho2, 1]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 100000)
        data_array2 = np.random.multivariate_normal(ourmean, ourcov2, 100000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        data_dict2 = dict.fromkeys(dimkeys)
        for i in range(dimension):
            data_dict2[dimkeys[i]] = data_array2[:, i]

        multigaussian1 = GaussianCopula(input_data=data_dict, dimkeys=dimkeys, marginals=marginals, quadstep=0.001)
        multigaussian2 = GaussianCopula(input_data=data_dict2, dimkeys=dimkeys, marginals=marginals, quadstep=0.001)

        rank_data = multigaussian2.generates_U(10000)

        diag(2).rank_histogram(rank_data, 20, multigaussian1)

class EMDTester(unittest.TestCase):
    def test_different_comparison(self):
        """
        This test compare the different comparison we can imagine between a empirical distribution and the uniform distribution
        The EMD to the uniform distribution is difficult to compute so we represent the uniform distribution by a vector :
        Either by generating a random sample on [0,1] : Y
        Or with regular interval of length 1/n on [0,1] : Z
        Or with regular smaller regular intervals of length 1/m in [0,1] ; A
        :return:  print the histograms of the emd found for each vector when we compute 1000 of this 3 EMD
        """
        n = 10000
        m = 100
        H = np.zeros((1000, 3))
        Z = np.asarray(range(n)) / n
        A = np.zeros(n)
        for i in range(m):
            for j in range(int(n / m)):
                A[i * n / m + j] = i / m

        for k in range(1000):
            X = np.random.rand(n)
            Y = np.random.rand(n)

            H[k][0] = emd_sort(U=X, V=Y)
            H[k][1] = emd_sort(U=X, V=Z)
            H[k][2]= emd_sort(U=X, V=A)
            print(k)

        count, bins, ignored = plt.hist(H, normed='True', label='Y', color='brk')
        # EMD between X and Y will be in blue
        # EMD between X and Z will be in red
        # EMD between X and A will be in black
        plt.legend(loc='upper right')
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
        plt.show()

    def test_pyomo_with_sort(self):
        n = 100
        p=1
        normal1 = np.random.randn(n)
        normal2 = np.random.randn(n)
        uniform1 = np.random.rand(n)
        uniform2 = np.random.rand(n)
        linearprog = np.asarray(range(n)) / n
        U = linearprog
        V = normal1

        iter = []
        for i in range(n):
            for j in range(n):
                iter.append((i, j))
        print('Unsorted')
        print('EMD sort')
        tic()
        print(emd_sort(U, V,p))
        toc()
        print('EMD pyomo')
        tic()
        print(emd_pyomo(U, V,p)[0])
        toc()
        print(' ')
        print('EMD sort')
        tic()
        print(emd_sort(np.sort(U), np.sort(V),p))
        toc()
        print("sorted")
        print('EMD pyomo')
        tic()
        print(emd_pyomo(np.sort(U),np.sort(V),p)[0])
        toc()

    def test_gaussian_copula(self):
        #not finished yet
        print("Warning test not finished yet")
        n = 10000
        dimkeys = ["solar", "wind"]
        dimension = len(dimkeys)
        ourmean = [2, 3]
        ourmeandict = {"solar": 0, "wind": 0}
        rho =0.1
        rho2 = 0.9
        ourcov = [[1, rho], [rho, 1]]
        ourcov2 = [[1, rho2], [rho2, 1]]
        marginals = {"solar": UnivariateNormalDistribution(var=ourcov[0][0], mean=ourmean[0]),
                     "wind": UnivariateNormalDistribution(var=ourcov[1][1], mean=ourmean[1])}

        data_array = np.random.multivariate_normal(ourmean, ourcov, 100000)
        data_array2 = np.random.multivariate_normal(ourmean, ourcov2, 100000)
        data_dict = dict.fromkeys(dimkeys)

        for i in range(dimension):
            data_dict[dimkeys[i]] = data_array[:, i]

        data_dict2 = dict.fromkeys(dimkeys)
        for i in range(dimension):
            data_dict2[dimkeys[i]] = data_array2[:, i]

        multigaussian1 = GaussianCopula(input_data=data_dict, dimkeys=dimkeys, marginals=marginals, quadstep=0.001)
        multigaussian2 = GaussianCopula(input_data=data_dict2, dimkeys=dimkeys, marginals=marginals, quadstep=0.001)

        print(emd_sort(data_array,data_array))
        print(emd_sort(data_array2, data_array))
        print(emd_sort(data_array2, data_array2))
        #self.assertGreater(g, m)
        #self.assertGreater(m, s)



if __name__ == '__main__':
    i=0
    for distr in ['empirical-copula']:
        CopulaTester().test_plot(distr)
        i=+1
        print(i)
