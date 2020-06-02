#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
benchmark.py
"""

import time
import timeit

import numpy as np
import pandas as pd

import prescient.gosm.sources as sources
from prescient.gosm.markov_chains.descriptions import bin_description
import prescient.util.distributions.distributions as distributions
import prescient.util.distributions.distributions.copula as copula


copulas = [
    copula.GaussianCopula,
    copula.StudentCopula,
    copula.ClaytonCopula,
    copula.FrankCopula,
    copula.GumbelCopula
]

class Timer:
    @classmethod
    def time_one_function(cls, func, repeat=1, number=None, setup=None):
        if setup is not None:
            setup()

        if number is None:
            start_time = time.time()
            func()
            time_one_run = time.time() - start_time

            # We estimate the number to run by finding how many can run in 10s
            number = 10 / time_one_run
            i = 0
            while True:
                num = 10 ** i
                if num > number:
                    break
                i += 1
            number = num // 10

            # We want to run at least three times however
            number = max(3, number)

        for i in range(repeat):
            print("Timing {}: Run #{}".format(func.__name__, i+1))
            times = []
            for _ in range(number):
                start_time = time.time()
                func()
                net_time = time.time() - start_time
                times.append(net_time)
            print("After running {} times:".format(number))
            format_string = "{:<20} {:>.5e} seconds"
            print(format_string.format("Best time", min(times)))
            print(format_string.format("Average time", np.mean(times)))
            print(format_string.format("Worst time", max(times)))
            print(format_string.format("Total time", sum(times)))

    def time_all_methods(self, repeat=1):
        for field in dir(self):
            if field.startswith('time_'):
                func = getattr(self, field)
                if not callable(func):
                    continue

                # We skip the special methods defined in this class
                if field in ['time_one_function', 'time_all_methods']:
                    continue

                if 'setup' in dir(self):
                    setup = getattr(self, 'setup')
                else:
                    setup = None

                self.time_one_function(func, repeat=repeat, setup=setup)


class SourceTimer(Timer):
    def setup(self):
        df = pd.DataFrame()
        df['a'] = np.random.randn(1000)
        df['b'] = np.random.randn(1000)
        df['c'] = np.random.randn(1000)
        self.source = gosm.sources.Source('wind', df, 'wind')

    def time_window(self):
        self.source.window('a', -1, 1)

    def time_enumerate(self):
        self.source.enumerate('a', 0)

    def time_state_walk(self):
        state_description = bin_description(1, 'a')
        self.source.get_state_walk(state_description)

    def time_quantiles(self):
        self.source.get_quantiles('a', [0, 0.25, 0.5, 0.75, 1])

    def time_apply_bounds(self):
        self.source.apply_bounds('a', -1, 1)

    def time_scale(self):
        self.source.scale('a', 1.1)


class DistributionTimer(Timer):
    def __init__(self, distr_class):
        print("Timing Distribution {}".format(distr_class.__name__))
        self.distr_class = distr_class

    def setup(self):
        self.data = np.random.randn(1000)
        self.distribution = self.distr_class.fit(self.data)

    def time_fit(self):
        self.distr_class.fit(self.data)

    def time_pdf(self):
        x = np.random.randn()
        self.distribution.pdf(x)

    def time_cdf(self):
        x = np.random.randn()
        self.distribution.cdf(x)

    def time_cdf_inverse(self):
        y = np.random.rand()
        self.distribution.cdf_inverse(y)


class CopulaTimer(Timer):
    def __init__(self, distr_class):
        print("Timing Copula {}".format(distr_class.__name__))
        self.distr_class = distr_class
        if distr_class.registered_ndim is None:
            self.dimensions = [2, 3]
        else:
            self.dimensions = [2]

        for dimension in self.dimensions:
            fit_name = 'time_fit_{}'.format(dimension)
            fit_func = lambda: self.fit(dimension)
            fit_func.__name__ = fit_name
            setattr(self, fit_name, fit_func)
            pdf_name = 'time_pdf_{}'.format(dimension)
            pdf_func = lambda: self.pdf(dimension)
            pdf_func.__name__ = pdf_name
            setattr(self, pdf_name, pdf_func)
            cdf_name = 'time_cdf_{}'.format(dimension)
            cdf_func = lambda: self.cdf(dimension)
            cdf_func.__name__ = cdf_name
            setattr(self, cdf_name, cdf_func)

    def setup(self):
        self.distributions = {}
        self.data = {}
        for dim in self.dimensions:
            self.data[dim] = np.random.rand(dim, 1000)
            self.distributions[dim] = self.distr_class.fit(self.data[dim])

    def fit(self, n):
        self.distr_class.fit(self.data[n])

    def pdf(self, n):
        x = np.random.rand(n)
        self.distributions[n].pdf(*x)

    def cdf(self, n):
        x = np.random.rand(n)
        self.distributions[n].cdf(*x)


if __name__ == '__main__':
    """
    SourceTimer().time_all_methods()
    for field in dir(distributions):
        value = getattr(distributions, field)
        if hasattr(value, 'is_registered_distribution'):
            if value.registered_ndim == 1:
                DistributionTimer(value).time_all_methods()
    """

    for copula in copulas:
        CopulaTimer(copula).time_all_methods()
