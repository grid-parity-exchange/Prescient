#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import math



def emd_sort(U=None, V=None,low_bound=0,up_bound=1,p=1):
    """
    This function calculates an earth mover distance between 2 empirical distribution in dimension 1
    It calculates it in a very simple way.
    I still have to verify if this distance is the same as the emd1.
    There is no reason why but I did not demonstrate it.
=
    :param U,V: The two vectors between which you wan to compute the EMD
    :param p: Which EMD you want to compute :
                the distance between to point U[i] and V[j] will be |U[i]-V[j]|^p
    :param low_bound: fraction between 0 and 1 where we start to take the indexes used to compute the emd
    :param up_bound: fraction between 0 and 1 where we start to take the indexes used to compute the emd
    :return: the p-th EMD between U and V (or between the empirical distribution defined with U and V)

    """
    n = len(U)
    if len(V)!=n:
        raise('The vectors should have the same size')
    res=0
    U_sort = np.copy(U)
    U_sort.sort()
    V_sort = np.copy(V)
    V_sort.sort()
    l=0
    for i in range(math.floor(n*low_bound),math.ceil(n*up_bound)):
        res += np.abs(U_sort[i]-V_sort[i])**p
        l += 1

    return res/l


def emd_pyomo(U=None,V=None,p=1):
    """
    This function will calculate the real EMD resolving the linear optimisation problem
    with pyomo and gurobi.
    :param U,V: The two vectors between which you wan to compute the EMD
    :param p: Which EMD you want to compute :
                the distance between to point U[i] and V[j] will be |U[i]-V[j]|^p
    :return: the p-th EMD between U and V (or between the empirical distribution defined with U and V)
    """
    n = len(U)
    if len(V) != n:
        raise ('The vectors should have the same size')

    # We create an iterable object (here a list) that contains all the couples i,j for 0<=i,j<n
    iter=[]
    for i in range(n):
        for j in range(n):
            iter.append((i,j))

    #We create a pyomo concrete model that will permit to solve a linear program that will give the result of the EMD
    flow = ConcreteModel()
    flow.f = Var(iter,within=PositiveReals)

    def obj_rule(flow):
        return sum(abs(U[i]-V[j])**p*flow.f[i,j] for i,j in iter)
    flow.obj = Objective(rule = obj_rule)

    def con_rule_equal(flow):
        # We define the constraint on our flows.
        return sum(flow.f[i,j] for i,j in iter)==n

    def con_rule_i(flow,i):
        return sum(flow.f[i,j] for j in range(n)) <= 1

    def con_rule_j(flow,j):
        return sum(flow.f[i, j] for i in range(n)) <= 1


    flow.coni = Constraint(range(n),rule=con_rule_i)
    flow.conj = Constraint(range(n), rule=con_rule_j)
    flow.con0 = Constraint(rule=con_rule_equal)


    opt = SolverFactory('gurobi')
    results = opt.solve(flow) #solves and updates flow

    matrix = np.identity(n)

    for i in range(n):
        for j in range(n):
            matrix[i][j] =flow.f[i,j].value

    return value(flow.obj)/n,matrix



class RankHistogram(object):
    def __init__(self,distr=None,rank_data=None,rank=10):
        self.distr =distr
        self.rank_data=np.copy(rank_data)
        self.r = rank   #number of rank we want



        # Rank is a vector : R[i] defines the i-th r-quantile
        self.Rank = np.zeros(self.r)

        if rank_data is None:
            for i in range(self.r):
                self.Rank[i] = (i+1)/self.r
        else:
            m = len(self.rank_data)
            # We define the i-th quantile or rank
            # We do that experimentally thanks to data rank_data of length m
            self.rank_data.sort()
            for i in range(self.r):
                self.Rank[i] = self.rank_data[int((i + 1) * m / self.r - 1)]
            # We divide here the input_array in r parts and each rank will be the value at the beginning of each part


    def plot(self, n=10000, sample = None, output_file =  None):
        """
        This function will plot a Rank Histogram
        :param n: Size of the sample that will fill the rank
        :param r: Number of rank we want = number of quantile we use = number of column of the histogram
        :return: a plot of the rank histogramsk
        """
        if sample is None:
            X= self.distr.generates_X(n)
        else:
            X = sample
        A = np.zeros(n)

        for i in range(n):
            #for all data we have we look for its rank.
            #i indicates the float in input_data whose rank we evaluate
            j=-1 # j indicates the rank we are visiting
            bool = True # bool tells us if our data is inferior to the data of rank j

            while bool and j<(self.r-1):
                j += 1
                bool = X[i]> self.Rank[j]
            A[i] = (j+0.5)/self.r #A[i] will increase the height of the column j by one.
        B = np.asarray(range(self.r+1))/self.r
        C = np.asarray(range(self.r))


        # bins are the indexes of the histograms
        # each column i of the histograms has its value between bins[i] and bins[i+1]
        count, bins, ignored = plt.hist(A, B,normed='True')
        # count is a list with the value of each column
        # count[i] will be the height of the column i
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='b')
        if output_file == None:
            plt.show()
        else:
            plt.savefig(output_file)