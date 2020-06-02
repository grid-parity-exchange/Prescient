#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
epimodel.py
"""

import math
from collections import OrderedDict

import numpy as np
from pyomo.environ import *
from pyomo.opt import *

def FitEpispline(x, y, positiveness_constraint=False, increasingness_constraint=False,
                 error_norm='L2', seg_N=20, seg_kappa=100, L1Linf_solver='gurobi',
                 L2Norm_solver='gurobi'):
    # Receives as input a set of points x and a function evaluated at those points y(x),
    # and fit an epi-spline s that approximates that function.
    if len(x) != len(y):
        raise RuntimeError('***ERROR: x and y must have the same length.')

    # We first create a new model
    model = ConcreteModel()

    # Sets
    model.I = Set(initialize=list(range(len(x))))
    model.intervals = RangeSet(int(seg_N))

    # Parameters
    model.N = Param(initialize=int(seg_N))
    model.kappa = Param(initialize=float(seg_kappa))

    model.alpha = Param(initialize=min(x))
    model.beta = Param(initialize=max(x))

    def x_init(m, i):
        return x[i]

    model.x = Param(model.I, initialize=x_init)

    def fx_init(m, i):
        return y[i]

    model.fx = Param(model.I, initialize=fx_init)

    model.delta = Param(initialize=float(model.beta - model.alpha) / model.N)

    def k_init(m, i):
        aux = int(math.ceil(float(m.x[i] - m.alpha) / m.delta))
        if aux == 0:
            aux = 1
        return aux

    model.k = Param(model.I, initialize=k_init)

    # Variables
    model.e = Var(model.I, within=Reals)
    model.s = Var(model.I, within=Reals)

    model.s0 = Var(within=Reals, initialize=0.0)
    model.v0 = Var(within=Reals, initialize=0.0)
    model.a = Var(model.intervals, bounds=(-model.kappa, model.kappa), initialize=0.0)

    # Constraints
    def compute_spline(m, i):
        return m.s[i] == m.s0 + m.v0 * m.x[i] + m.delta * sum(
            (m.x[i] - j * m.delta + 0.5 * m.delta) * m.a[j] for j in range(1, m.k[i])) \
                         + 0.5 * m.a[m.k[i]] * (m.x[i] - (m.k[i] - 1) * m.delta) ** 2

    model.ComputeSpline = Constraint(model.I, rule=compute_spline)

    # Positiveness
    if positiveness_constraint is True:
        eps = 0.01
        w = [float(i) for i in np.arange(min(x), max(x), eps)]
        model.J = Set(initialize=list(range(len(w))))

        def positive_spline(m, i):
            l = int(math.ceil(float(w[i] - m.alpha) / m.delta))
            if l == 0:
                l = 1
            return m.s0 + m.v0 * w[i] + m.delta * sum(
                (w[i] - j * m.delta + 0.5 * m.delta) * m.a[j] for j in
                range(1, l)) + 0.5 * m.a[l] * (w[i] - (l - 1) * m.delta) ** 2 >= 0

        model.PositiveSpline = Constraint(model.J, rule=positive_spline)

        # Increasingness
    if increasingness_constraint is True:

        # First derivative
        def increasing_spline(m, i):

            l = int(math.ceil(float(w[i] - m.alpha) / m.delta))
            if l == 0:
                l = 1
            return m.v0 + m.delta * sum(m.a[j] for j in
                                        range(1, l)) + m.a[l] * (w[i] - 2 * (l - 1) * m.delta) >= 0

        model.IncreasingSpline = Constraint(model.J, rule=increasing_spline)

    if error_norm == "L1":
        def ePositiveSide_rule(m, i):
            return m.e[i] >= m.fx[i] - m.s[i]

        model.eDefPos = Constraint(model.I, rule=ePositiveSide_rule)

        def eNegativeSide_rule(m, i):
            return m.e[i] >= - m.fx[i] + m.s[i]

        model.eDefNeg = Constraint(model.I, rule=eNegativeSide_rule)
    elif error_norm == "L2":
        def compute_error_rule(m, i):
            return m.e[i] == m.fx[i] - m.s[i]

        model.ComputeError = Constraint(model.I, rule=compute_error_rule)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    # Objective function
    if error_norm == 'L1':
        def Obj_rule(m):
            return summation(m.e)

        model.Obj = Objective(rule=Obj_rule)
    elif error_norm == 'L2':
        def Obj_rule(m):
            return sum(m.e[i] ** 2 for i in m.I)

        model.Obj = Objective(rule=Obj_rule, sense=minimize)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    # Instance creation and optimization
    model.preprocess()
    if error_norm == "L1":
        opt = SolverFactory(L1Linf_solver)
        opt.options.mip_tolerances_absmipgap = 0
        opt.options.mip_tolerances_mipgap = 0
        opt.options.mip_tolerances_integrality = 1e-9
    elif error_norm == 'L2':
        opt = SolverFactory(L2Norm_solver)
    else:
        raise RuntimeError("***ERROR: Unknown error norm=" + error_norm + " selected")

    opt.solve(model, tee=False)
    return model

# Optimization model to obtain the exp - epi spline that models the distribution of a vector or a dictionary of data e
def FitDistribution(time_series, dom=None, specific_prob_constraint=None,
                    seg_N=20, seg_kappa=100, nonlinear_solver='ipopt',
                    non_negativity_constraint_distributions=0,
                    probability_constraint_of_distributions=1):
    """The additional parameter dom defines special characteristics of the support of the distribution.
    It can be pos (positive domain), neg (negative domain)
    Or it can be also a float that defines how many standard deviations from the mean define the support.
    Or max or min to define the limits as the outmost data points."""
    N = int(seg_N)
    kappa = float(seg_kappa)

    # -------------------------------------------------------
    # Model construction
    # -------------------------------------------------------
    model = AbstractModel()
    # -------------------------------------------------------
    # Parameters
    # -------------------------------------------------------
    if isinstance(time_series, dict):
        days = list(time_series.keys())
    elif isinstance(time_series, list):
        days = list(range(len(time_series)))
    else:
        raise RuntimeError('***ERROR: Unknown type of input data.')

    intervals = list(range(1, N + 1))
    delta = float(1) / float(N)

    model.N = Param(within=PositiveReals, initialize=N)
    model.delta = Param(within=PositiveReals, initialize=delta)

    if specific_prob_constraint is None:
        alpha, beta = error_domain(time_series, dom)
    else:
        if isinstance(specific_prob_constraint, tuple):
            alpha, beta = specific_prob_constraint
            alpha = float(alpha)
            beta = float(beta)  # avoid numpy64
        elif isinstance(specific_prob_constraint, list):
            if len(specific_prob_constraint) == 2:
                alpha = specific_prob_constraint[0]
                beta = specific_prob_constraint[1]
            else:
                raise RuntimeError('***ERROR: The list specific_prob_constraint has to have a length of 2.')

        elif isinstance(specific_prob_constraint, str):
            alpha, beta = error_domain(time_series, dom)
        else:
            raise RuntimeError('***ERROR: specific_prob_constraint has either to be a tuple or a list of length 2.')

    if alpha == beta:  # this means there is only a CONSTANT bias
        return model, alpha, beta

    # Here we normalize the data. Then, m.et is in [0,1]
    def et_init(modelo, j, k=None):
        if k is not None:
            val = float(time_series[j, k] - alpha) / (beta - alpha)
            if val < 0.0:
                return 0.0
            if val > 1.0:
                return 1.0
            return val
        else:
            val = float(time_series[j] - alpha) / (beta - alpha)
            if val < 0.0:
                return 0.0
            if val > 1.0:
                return 1.0
            return val

    model.et = Param(days, initialize=et_init)

    def tau_init(modelo, i):
        return i * delta

    model.tau = Param(intervals, initialize=tau_init)
    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------
    if non_negativity_constraint_distributions == 1:
        model.w0 = Var(within=NonNegativeReals)
        model.u0 = Var(within=NonNegativeReals)
    else:
        model.w0 = Var()
        model.u0 = Var()
    model.a = Var(intervals, bounds=(0, kappa))
    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    if probability_constraint_of_distributions == 1:
        def prob_rule(modelo):  # The sum of the probabilities over all the domain must be 1.
            if specific_prob_constraint is not None:
                s = 0.01
                samp = np.arange(0.0, 1.0 + s, s)
                aux = 0
                for x in samp:
                    x = float(x)
                    k = int(math.ceil(N * x))
                    if k > 1:
                        tauk = modelo.tau[k - 1]
                    else:
                        tauk = 0
                        k = 1  # avoids errors when i = 0
                    aux += s * exp(
                        -(modelo.w0 + modelo.u0 * x + delta * sum(
                            (x - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in
                            range(1, k)) + 0.5 * modelo.a[k] * (x - tauk) ** 2))
            else:
                aux = delta * exp(-modelo.w0)
                for i in intervals:
                    aux += delta * exp(
                        -(modelo.w0 + modelo.u0 * modelo.tau[i] + delta * sum(
                            (modelo.tau[i] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in
                            range(1, i)) + 0.5 * modelo.a[i] * delta ** 2))
            return aux == 1

        model.prob = Constraint(rule=prob_rule)

    # -------------------------------------------------------
    # Objective function
    # -------------------------------------------------------
    def fobj_rule(modelo):  # appending _rule we don't need to define rule=rulename
        aux = 0
        for d in days:
            k = int(math.ceil(N * modelo.et[d]))
            if k > 1:
                tauk = modelo.tau[k - 1]
            else:
                tauk = 0
                k = 1  # avoids erros when i = 0
            aux += modelo.w0 + modelo.u0 * modelo.et[d] + delta * sum(
                (modelo.et[d] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in
                range(1, k)) + 0.5 * modelo.a[k] * (modelo.et[d] - tauk) ** 2
        aux /= len(days)

        # We add the integral
        if probability_constraint_of_distributions != 1:
            aux += delta * exp(-modelo.w0)
            for i in intervals:
                aux += delta * exp(
                    -(modelo.w0 + modelo.u0 * modelo.tau[i] + delta * sum(
                        (modelo.tau[i] - modelo.tau[j] + 0.5 * delta) * modelo.a[j] for j in
                        range(1, i)) + 0.5 * modelo.a[i] * delta ** 2))
        return aux

    model.fobj = Objective(rule=fobj_rule, sense=minimize)
    # -------------------------------------------------------
    # Instance creation and optimization
    # -------------------------------------------------------
    instance = model.create_instance()
    opt = SolverFactory(nonlinear_solver)
    results = opt.solve(instance, tee=False)
    # instance.load(results)
    return instance, alpha, beta


def error_domain(errors, dom=None):
    """
    This computes the parameters alpha and beta
    from a list or dictionary, errors, and a
    string dom which specifies how to compute alpha and beta
    
    alpha and beta act as a kind of bound on the data.
    If no domain is specified then these will be the minimum and maximum
    of the data

    dom should be a string of the following form "<field>,<field>,...,<field>"
    where <field> is replaced by one of the following:
        1. A number specifying how many standard deviations away from mean
           you want alpha and beta to be set to
        2. pos which fixes alpha to 0 if alpha was prior set to a negative value
        3. neg which sets beta to 0 if beta was prior set to a positive value
        4. min which sets alpha to min
        5. max which sets beta to max
    These fields are processed in order and set alpha and beta to subsequent
    values.

    Returns (alpha, beta)
    """

    if isinstance(errors, dict):
        data = list(errors.values())
    elif isinstance(errors, list):
        data = errors
    else:
        raise RuntimeError("errors should be a list or dictionary")

    mu = np.mean(data)
    sigma = np.std(data)
    alpha = _min = min(data)
    beta = _max = max(data)
    
    if dom is None:
        return alpha, beta
    
    pieces = dom.split(',')

    for piece in pieces:
        # We test if the piece is a number
        try: 
            num = float(piece)
        # Otherwise the string is one of the keywords
        except ValueError:
            num = None

        if num is not None:
            a = mu-num*sigma
            if a < alpha:
                alpha = a
            a = mu+num*sigma
            if a > beta:
                beta = a
        else:
            if piece == 'pos' and _min < 0:
                raise RuntimeError('***Error: You set the domain to be positive and there are some data with negative values')
            elif piece == 'neg' and _max > 0:
                raise RuntimeError('***Error: You set the domain to be negative and there are some data with positive values')
            elif piece == 'pos' and alpha < 0:
                alpha = 0
            elif piece == 'neg' and beta > 0:
                beta = 0
            elif piece == 'min' and alpha > _min:
                alpha = _min
            elif piece == 'max' and beta < _max:
                beta = _max

    return alpha, beta
