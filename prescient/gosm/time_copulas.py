#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import signal
from contextlib import contextmanager
import time

import numpy as np

import prescient.gosm.hyperrectangles as hyperrectangles
import prescient.util.distributions.copula as copula

copulas = [
    copula.GaussianCopula,
    copula.StudentCopula,
    copula.ClaytonCopula,
    copula.FrankCopula,
    copula.GumbelCopula
]

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def random_rectangle(n):
    bounds = {i: sorted(np.random.rand(2)) for i in range(n)}
    return hyperrectangles.Hyperrectangle('lol', bounds, [*range(n)])


def time_func(func, args=(), repeat=1, number=100, limit=300):
    for i in range(repeat):
        print("Timing {}: Run #{}".format(func.__name__, i+1))
        times = []
        for _ in range(number):
            try:
                with time_limit(limit):
                    start_time = time.time()
                    func(*args)
                    net_time = time.time() - start_time
                    times.append(net_time)
            except TimeoutException:
                print("This run took longer than {} seconds".format(limit))
                print("Exiting timing function")
                return
        print("After running {} times:".format(number))
        format_string = "{:<20} {:>.5e} seconds"
        print(format_string.format("Best time", min(times)))
        print(format_string.format("Average time", np.mean(times)))
        print(format_string.format("Worst time", max(times)))
        print(format_string.format("Total time", sum(times)))

ns = [1000, 10000, 100000, 1000000]
epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

def time_copula(distribution, repeat=1, number=100):

    print("Timing Copula {}, {} dimensions".format(distribution.name, distribution.dimension))

    def mc_int_on_rand(n):
        rectangle = random_rectangle(distribution.dimension)
        distribution.mc_probability_on_rectangle(rectangle, n)

    def quad_int_on_rand(epsabs):
        rectangle = random_rectangle(distribution.dimension)
        distribution.probability_on_rectangle(rectangle, epsabs)

    for n in ns:
        print("Timing Monte Carlo integration with n={}".format(n))
        time_func(mc_int_on_rand, (n,))

    for epsilon in epsilons:
        print("Timing Quad with epsabs={}".format(epsilon))
        time_func(quad_int_on_rand, (epsilon,))

if __name__ == '__main__':
    for copula in copulas:
        if copula.registered_ndim == 2:
            dimensions = [2]
        else:
            dimensions = [2, 3, 4, 5, 10]

        for dimension in dimensions:
            xs = np.random.rand(dimension, 1000)
            distribution = copula.fit(xs)
            time_copula(distribution)
