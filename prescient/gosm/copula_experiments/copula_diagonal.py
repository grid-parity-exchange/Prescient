#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import prescient.gosm.copula_experiments
from prescient.gosm.copula_experiments.copula_evaluate import RankHistogram, emd_sort, emd_pyomo


class diag(object):
    def __init__(self, d=1):
        """
        Temporary class to write the functions of projection
        :param d: dimension of the space
        self.list_diags gives a list of the pair f corners that represent the diagonals
        self.directions gives a list of vectors representing the  directions of the diagonals
        self.projections gives the matrix of the projections on the vectorial space defined by the direction
                Warning : the vectorial 1 dimensional space defined by the vector is different from the affine space define by the diagonal
                        This is why self.proj != self.projection
        self.proj is a function that project its arguement x on the diagonal i 
        self.n_diag is the number of diagonal : 2^(d-1) where d is the dimension of the space
        """
        self.d = d
        self.n_diag = 2**(d-1)
        temp0 = []
        temp1 = []
        for i in range(self.d):
            temp0.append(0)
            temp1.append(1)
        self.list_of_diag = []
        self.directions = []
        self.projections = []
        self.center = 0.5 * np.ones(self.d)

        def f(n):
            # recursive function that will give the list of the corners
            # It has a structure of binary tree
            if n == 0:
                corner0 = np.array(temp0[:])
                corner1 = np.array(temp1[:])
                self.list_of_diag.append([temp0[:], temp1[:]])
                vector = corner1 - corner0
                self.directions.append(vector)
                self.projections.append(1 / d * np.dot(np.transpose(np.matrix(vector)), np.matrix(vector)))
            else:
                f(n - 1)
                temp0[n] = (1 + temp0[n]) % 2  # % means modulo 2, so if this value was 1 it will be 0 and vice versa
                temp1[n] = (1 + temp1[n]) % 2
                f(n - 1)

        f(d - 1)  #We create the lists of corners projections and directions

    def proj_vector(self,x,i=0,kind='diagonal'):
        """
        This function will project a point on the diagonal or on the edge
        :param i: choice of which diagonal or which edge
        :param x: the point you want to project on the diagonal
        :param type: can either be 'diagonal' or 'edge
        :return: the point projected on the diagonal
        """
        if kind=='diagonal':
            if type(x) is np.ndarray:
                if type(x[0]) is np.ndarray:
                    res = np.ones((len(x),self.d))
                    for j in range(len(x)):
                        res[j]= np.dot(self.projections[i], x[j] - self.center) + self.center
                    return res
                else:
                    return np.dot(self.projections[i], x - self.center) + self.center
            else:
                return np.dot(self.projections[i], x - self.center) + self.center
        elif kind=='marginal':
            if type(x) is np.ndarray:
                if type(x[0]) is np.ndarray:
                    res = np.zeros((len(x),self.d))
                    for j in range(len(x)):
                        res[j,i]= x[j,i]
                    return res
                else:
                    res = np.zeros(self.d)
                    res[i] = x[i]
                    return res
            else:
                res = np.zeros(self.d)
                res[i] = x[i]
                return res
        else:
            raise('The only types available are diagonal and marginal')

    def proj_scalar(self,x,i=0,kind='diagonal',distr=None):
        """
            This function will return the first coordinate of the projected point on the diagonal
            or the only non null coordinate of the projected point on the edge.
            If we know on which diagonal we are, we can find out the other coordinates.
            This function is meant to be faster than proj_vector.
            This is why I use directly the sums and not the np.dot
            :param i: choice of which diagonal
            :param x: the point you want to project on the diagonal
            :return: the first coordinate of the projected point on the diagonal
                     or the only non null coordinate of the projected point on the edge
        """
        if kind=='diagonal':
            if type(x) is np.ndarray:
                if type(x[0]) is np.ndarray:
                    res = np.ones(len(x))
                    for j in range(len(x)):
                        res[j] = sum(self.projections[i][0,k]* (x[j,k] - 0.5) for k in range(self.d))+0.5
                    return res
                else:
                    return sum(self.projections[i][0,k]* (x[k] - 0.5) for k in range(self.d))+0.5
            else:
                return sum(self.projections[i][0,k]* (x[k] - 0.5) for k in range(self.d))+0.5
        elif kind=='marginal':
            if type(x) is np.ndarray:
                if type(x[0]) is np.ndarray:
                    res = np.ones(len(x))
                    for j in range(len(x)):
                        res[j] = x[j,i]
                    return res
                else:
                    return x[i]
            else:
                return x[i]
        elif kind=='kendall':
            if type(x) is np.ndarray:
                if type(x[0]) is np.ndarray:
                    res = np.ones(len(x))
                    for j in range(len(x)):
                        res[j] = distr.kendall_function(distr.cdf(x[j]))
                    return res
                else:
                    return distr.kendall_function(distr.cdf(x))
            else:
                return distr.kendall_function(distr.cdf(x))

        else:
            raise ('The only types available are diagonal and marginal')

    def rank_histogram(self,rank_data=None,rank=10,copula=None,index_diag=0,n=10000):
        m = len(rank_data)


        # P_ank is a rectangular matrix that takes an index of a diagonal in the first argument
        # and the rank/ quanditle in the second argument

        P_rank = self.proj_vector(rank_data, index_diag)
        # P_rank is a list of projected points on the diagonal.
        # So its coordonates should be x1,x2=x1 or 1-x1,...,xd=x1 or 1-xd
        # We are only interested in one dimension
        # It is the same to take the length of the partial diagonal or the fisrt coordonate
        # See paper for more information in section Diagonal and projection subsection distribution on diagonal
        # This the reason why we input P_rank[:,0]
        rank_hist = RankHistogram(copula, P_rank[:,0], rank)


        U = copula.generates_U(n)
        P = self.proj_vector(U, index_diag)
        rank_hist.plot(sample = P[:,0])

    def emd_sort_random(self,copula=None,index_diag=0,n=10000):
        U = copula.generates_U(n)
        P = self.proj(U, index_diag)
        V = np.random.rand(n)
        EMD.emd_sort(P[:,0],V)

    def emd_sort_regular(self,copula=None,index_diag=0,n=10000):
        U = copula.generates_U(n)
        P = self.proj(U,index_diag)
        V= np.asarray(range(n))/n
        EMD.emd_sort(P[:,0],V)
