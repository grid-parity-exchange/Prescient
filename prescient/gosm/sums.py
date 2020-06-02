#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

""""
sums.py: sums of random variables using the gosm distributions
Initially by David L. Woodruff; Aug,Sept 2017
"""
from __future__ import division

import sys
import os
import datetime as dt

import pandas as pd
import numpy as np
import math
import random

from prescient.util.distributions.distribution_factory import distribution_factory

def UonSimplex(maxsum, dimkeys, retdict):
    """
    Populate retdict with values uniformly distributed on a regular simplex
    defined by maxsum.
    Based on Horst Kraemer’s algorithm as described in
    "Sampling Uniformly from the Unit Simplex"
    Noah A. Smith and Roy W. Tromble
    Johns Hopkins University
    August 2004
    NOTE: (but with one more random() so it doesn't always add to maxsum)
    inputs:
        maxsum: the maximum sum (e.g., one for the unit simplex)
        dimkeys: the keys in redict (in order)
        retdict: A dictionary in which to store the values
    returns:
        retdict populated with the values
    """
    p = len(dimkeys)
    u = [0]
    for i in range(p):
        u.append(random.random())
    # not p-1 then u.append(1); we want sum to leq 1 not 1
    
    # There are now have p+1 values in u.
    u.sort()
    for i in range(p):
        retdict[dimkeys[i]] = (u[i+1] - u[i]) * maxsum
    return retdict
        
def CDF_pos_of_sum(posdistr, dimkeys, cdfpt, N):
    """
    Compute the probability that the sum over the dimkeys is between 0 and cdfpt.
    In general, it could require integrating over a large space, since -1e10 + 1e10 = 0.
    So we REQUIRE distrs than do not have any negative support. 
    input:
        distr: a multivariate distribution object
        dimkeys: the keys in distr over which the sum is considered
        cdfpt: the value for which the cdf is to be calculated
    returns:
        probability
        std dev of the probability estimate

    method:
        As of Aug/Sep 2017 do the simplest, stupidest thing, which
        is straight monte carlo even though in the application of interest
        we know that the RVs are strongly correlated.
    ASSUME that all dimensions have a domain that extends as far as it needs to, but 
           HAS NO NEGATIVE PART. 
        (Oct 2017 TDB: add upper and lower bounds to each dimension. non-trivial)
    Note: the one pass variance estimate might have numerical issues.
    """
    assert(cdfpt >= 0)
    p = len(dimkeys) # the dimension
    x = dict.fromkeys(dimkeys)   # storage for the point
    sumsofar = 0
    sumsqsofar = 0
    for s in range(N):
        x = UonSimplex(cdfpt, dimkeys, x)
        pdfval = posdistr.pdf(x)
        sumsofar += pdfval
        sumsqsofar += pdfval * pdfval

    vol = pow(cdfpt, p) *  1 / math.factorial(p)
    mean = sumsofar * vol / N
    N1 = N-1
    var = (vol * vol * (sumsqsofar/N - (sumsofar*sumsofar) / (N1*N1))) / N

    return mean, math.sqrt(var)


if __name__ == '__main__':

    dim = 3
    mu = 4
    N = 10000
    dimkeys = []
    X = {}
    for i in range(dim):
        dimkeys.append("dim"+str(i+1))
        X[dimkeys[i]] = np.random.randn(N) + mu

    distr_class = distribution_factory('multivariate-normal')
    mydistr = distr_class(dimkeys, X)

    foo, bar = CDF_pos_of_sum(mydistr, dimkeys, dim*mu, N)
    print (str(foo), str(bar))


