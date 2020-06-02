#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import sys
import traceback

import numpy as np

import prescient.util.distributions.copula as copula
import prescient.util.distributions.distributions as distributions

def smoke_test_distribution(class_):
    """
    This function will instantiate a class in a variety of ways and test
    to see if the distribution will run without crashing. Obviously, this will
    not test for functionality; hence, it is a smoke test.

    Args:
        class_: The name of the class to test
    """

    data = np.random.rand(2, 1000)

    keys = ['foo', 'bar']
    print(class_)
    distr_1 = class_.fit(data)
    distr_2 = class_.fit(data, keys)
    distr_3 = class_.fit({key: vector for key, vector in zip(keys, data)}, keys)

    distr_1.pdf(0.5, 0.5)
    distr_1.pdf({0: 0.5, 1: 0.5})
    distr_1.cdf(0.5, 0.5)
    distr_1.cdf({0: 0.5, 1: 0.5})

    distr_2.pdf(0.5, 0.5)
    distr_2.pdf({'foo': 0.5, 'bar': 0.5})
    distr_2.pdf(foo=0.5, bar=0.5)
    distr_2.cdf(0.5, 0.5)
    distr_2.cdf({'foo': 0.5, 'bar': 0.5})
    distr_2.cdf(foo=0.5, bar=0.5)

    distr_3.pdf(0.5, 0.5)
    distr_3.pdf({'foo': 0.5, 'bar': 0.5})
    distr_3.pdf(foo=0.5, bar=0.5)
    distr_3.cdf(0.5, 0.5)
    distr_3.cdf({'foo': 0.5, 'bar': 0.5})
    distr_3.cdf(foo=0.5, bar=0.5)


if __name__ == '__main__':
    distrs = [distributions.MultiNormalDistribution]
    namespace = vars(copula)
    for name, value in namespace.items():
        if hasattr(value, 'is_registered_distribution'):
            distrs.append(value)

    for class_ in distrs:
        try:
            smoke_test_distribution(class_)
        except:
            exc_type, exc, tb = sys.exc_info()
            print("{}: {}".format(exc_type, exc))
            traceback.print_tb(tb)
