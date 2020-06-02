#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import numpy as np


def sample_on_rectangle(bounds):
    """
    Draws a uniform sample from the rectangle whose bounds are given.

    Args:
        bounds (List[List[float]]): A list of the intervals which composed
            the rectangle
    Returns:
        List[float]: The desired sample
    """
    return [np.random.uniform(a, b) for a, b in bounds]


def n_samples_on_rectangle(bounds, n):
    """
    Draws n uniform sample from the rectangle whose bounds are given.

    Args:
        bounds (List[List[float]]): A list of the intervals which composed
            the rectangle
    Returns:
        List[List[float]]: The desired sample
    """
    dimensions = []
    for a, b in bounds:
        dimension_sample = np.random.uniform(a, b, n)
        dimensions.append(dimension_sample)
    return np.transpose(dimensions)


def volume_of_rectangle(bounds):
    """
    Computes the volume of the rectangle whose bounds are given

    Args:
        bounds (List[List[float]]): A list of the intervals which composed
            the rectangle
    """
    product = 1
    for a, b in bounds:
        product *= b - a
    return product


def mc_integrate(f, xs, volume):
    """
    This function will estimate the integral of f over a region R via Monte
    Carlo Integration. In essence, this is done by sampling from the region
    and computing the average value of our function f at those points and then
    multiplying it by the volume of the region.

    We arrive at this algorithm by noting that should f and R be well behaved,
    then there exists c in R such that
        f(c) = \int_R f(x) dV / Vol(R) = Average(f)
    We get
        \int_R f(x) dV = Average(f) * Vol(R)
    We then approximate the actual average of f by the average at our sample
    xs
        Average(f) \\approx (1/N) (f(x_1) + ... + f(x_N))

    This function returns an approximate error bound for the computed integral.
    This is not an exact error bound necessarily as the algorithm is
    probabilistic.

    Args:
        f (function): A function from R^n to R, this should be a function
            of n variables.
        xs (List[List[float]]): A list of the points which were sampled
            beforehand from the region integrated over
        volume (float): The volume of the region integrated over

    Return:
        (float, float): The first return value is an estimate of the integral
            and the second is an approximate error bound
    """
    n = len(xs)
    ys = [f(*x) for x in xs]
    sample_mean = np.mean(ys)
    sample_var = np.var(ys)
    integral_est = volume * sample_mean
    error_est = volume * np.sqrt(sample_var / n)
    return integral_est, error_est


def mc_integrate_on_rectangle(f, bounds, n, sampler=None):
    if sampler is None:
        samples = n_samples_on_rectangle(bounds, n)
    else:
        samples = [sampler() for _ in range(n)]
    volume = volume_of_rectangle(bounds)
    return mc_integrate(f, samples, volume)
