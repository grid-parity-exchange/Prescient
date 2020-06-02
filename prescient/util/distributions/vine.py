#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import numpy as np

from .copula import CopulaBase
from .base_distribution import fit_wrapper, accepts_dict
from .distribution_factory import register_distribution
from .distribution_factory import distribution_factory


class Node:
    def __init__(self, value, label=None):
        self.value = value
        if label is None:
            self.label = str(value)
        else:
            self.label = label

    def __str__(self):
        return label


class Edge:
    def __init__(self, node1, node2, label=None):
        self.nodes = frozenset((node1, node2))
        if label is None:
            self.label = node1.label + ',' + node2.label
        else:
            self.label = label

    def __str__(self):
        return label


class Graph:
    """
    This class will be a representation of a graph using an adjacency list.
    This will be an undirected graph.
    This is not intended to be used for large graphs and is instead intended
    to be used with graphs with <100 nodes for the purposes of representing
    the conditional relationships between dimensions of a multivariate
    distribution.

    This will be instantiated with an adjacency list with the keys being the
    nodes and the values being the list of nodes that each node is adjacent
    to. Since the graph is undirected, it will on instantiation add the reverse
    edge for every edge if it is not already contained in the adjacency list.

    Attributes:
        nodes (list): A list of the nodes of the graph
        edges (list): A list of the edges present in the graph represented as
            tuples of nodes. This will have both the forward and reverse
            edge contained in the list
        adjacency_list (dict): A dictionary mapping nodes to the list of edges
            that the node is adjacent to
    """
    def __init__(self, adjacency_list):
        self.nodes = list(adjacency_list.keys())
        for node in self.nodes:
            if any(other not in self.nodes for other in adjacency_list[node]):
                raise ValueError("One of the node's list of adjacenct nodes "
                                 "contains an element which is not a node")

        self.adjacency_list = {node: [] for node in self.nodes}
        self.edges = []
        for node in adjacency_list:
            for other in adjacency_list[node]:
                # Here we add reverse edges if they are not contained in the
                # passed in adjacency list
                if other not in self.adjacency_list[node]:
                    self.adjacency_list[node].append(other)
                if node not in self.adjacency_list[other]:
                    self.adjacency_list[other].append(node)
        
        self.edges.append(frozenset((node, other)))

    def is_connected(self):
        """
        Returns True if the graph is connected. It checks this by performing
        a breadth first search and seeing if all nodes are visited at the end.

        Returns:
            bool: A boolean specifying if the graph is connected
        """
        seen = []
        nodes_to_visit = [self.nodes[0]]

        last_node = None
        while nodes_to_visit:
            curr_node = nodes_to_vist.pop(0)
            if curr_node in seen:
                continue
            seen.append(curr_node)

            # Pick all the next nodes except for the one just visited
            next_nodes = [node for node in self.edges[curr_node]
                          if node != last_node]
            nodes_to_visit.extend(next_nodes)
            last_node = curr_node

        # If we have seen as many nodes as there are in the graph, the graph
        # is connected
        return len(seen) == len(self.nodes)

    def is_tree(self):
        """
        Returns True if the graph is a tree. It checks this by performing a
        depth first search and returning False if in the process of the depth
        first search, it finds a node it has already visited.

        Returns:
            bool: A boolean specifying if the graph is a tree
        """

        if not(self.is_connected()):
            return False

        seen = []
        nodes_to_visit = [self.nodes[0]]

        # We keep track of what the last node we have seen is to not backtrack
        last_node = None
        while nodes_to_visit:
            # If we have traversed all the nodes without seeing a duplicate
            # we have a tree
            curr_node = nodes_to_vist.pop()
            if curr_node in seen:
                return False
            seen.append(curr_node)

            # Pick all the next nodes except for the one just visited
            next_nodes = [node for node in self.adjacency_list[curr_node]
                          if node != last_node]
            nodes_to_visit.extend(next_nodes)
            last_node = curr_node
        return True

    def add_node(self, node):
        """
        This function adds a node to the graph. This entails creating a new
        entry in the adjacency list

        Args:
            node: A hashable object that will represent a node
        """
        if node in self.nodes:
            raise ValueError("Node is already contained in the graph")

        self.nodes.append(node)
        self.adjacency_list[node] = []

    def add_edge(self, node1, node2):
        """
        This function adds an edge between the two nodes listed.
        This creates an entry for an edge in the adjacency list for node1 and
        node2.

        Args:
            node1: A hashable object representing a node
            node2: A hashable object representing a node
        """
        if (node1 in self.adjacency_list[node2] or
            node2 in self.adjacency_list[node1]):

            raise ValueError("There already is an edge between the two nodes.")

        self.adjacency_list[node1].append(node2)
        self.adjacency_list[node2].append(node1)
        # We add the forward and reverse edge
        self.edges.append(frozenset((node1, node2)))

    def networkx_graph(self):
        """
        This function will convert a Graph object into a networkx style graph.

        Returns:
            networkx.graph: A networkx style graph
        """
        try:
            import networkx as nx
        except ImportError:
            raise RuntimeError("networkx must be installed to use this "
                               "function")
        return nx.from_dict_of_lists(self.adjacency_list)

    def plot_graph(self, ax):
        """
        This function will use networkx to plot the graph.

        Args:
            ax (matplotlib.axis): An axis to plot the graph on
        """
        try:
            import networkx as nx
        except ImportError:
            raise RuntimeError("networkx must be installed to use this "
                               "function")
        nx_graph = self.networkx_graph()
        pos = nx.spring_layout(nx_graph)
        axes = nx.draw(nx_graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(nx_graph, pos=pos, ax=ax)


class VineEdgeLabel:
    """
    This class will represent an edge label in a vine. It will contain
    information about the constraint set (All nodes reachable by the vine)
    """
    def __init__(self, conditioning_set, conditioned_set):
        """
        Args:
            conditioning_set (set): The set of nodes indicating which variables
                we will be predicting.
            conditioned_set (set): The set of nodes indicating which variables
                we are conditioning on
        """
        self.conditioning_set = conditioning_set
        self.conditioned_set = conditioned_set
        self.constraint_set = conditioned_set | conditioning_set

    def __str__(self):
        conditioning_vars = sorted(self.conditioning_set)
        string = ','.join(map(str, conditioning_vars))
        conditioned_vars = sorted(self.conditioned_set)
        string += '|' + ','.join(map(str, conditioned_vars))
        return string

    __repr__ = __str__


def merge_vine_edges(edge1, edge2):
    """
    Args:
        edge1 (VineEdgeLabel): The first edge
        edge2 (VineEdgeLabel): The second edge
    """
    conditioning_set = edge1.constraint_set ^ edge2.constraint_set
    conditioned_set = edge1.constraint_set & edge2.constraint_set
    return VineEdgeLabel(conditioning_set, conditioned_set)


def construct_d_vine(nodes):
    """
    This function will construct a D vine from the list of nodes passed in.
    A D vine is a regular vine with the property that each node in each tree
    in the vine has at most degree 2.

    Note that the ordering of the nodes will determine the pairwise
    decomposition into each subsequent tree.

    A simple representation with 4 nodes A,B,C,D
    Tree 1:
        A -------- B --------- C --------- D
             AB           BC        CD
    Tree 2:
             AB --------- BC --------- CD
                  AC|B         BD|C
    Tree 3:        
                  AC|B ------- BD|C
                        AD|BC

    We construct the subsequent trees by first looking at the consecutive pairs
    of nodes for the current tree. For each edge, we will keep track of the
    constraint sets. This will be the set of nodes in the first tree reachable
    by the set membership relation. For the first tree the constraint sets will
    be the set of the two nodes that the edge is incident to, but
    for each consecutive tree, the constraint sets will be composed of the
    union of the constraint sets of its components.
    
    Args:
        nodes (list): A list of objects that will represent nodes of the vine
    """



class Vine:
    """
    This will be a representation of a graphical Vine. A vine is a collection
    of trees [T1, T2, ..., Tn] where the edges of T_i form the nodes of T_i+1.

    This will be a regular vine so the trees must satisfy the proximal
    condition. This means that for i = 2,...,n, if we have that two nodes in
    T_i are adjacent. Note that the nodes will be edges {a1, a2}, {b1, b2} of
    T_i-1. The proximal condition enforces that the intersection of the above
    sets will have exactly one element.
    """


@register_distribution(name='cvine-copula')
class CVineCopula(CopulaBase):

    def __init__(self, dimkeys=None, input_data=None, marginals=None,
                       pair_copulae_strings=None):
        """
        Args :
            dimkeys (list): keys for each dimension in dictin (e.g. a list of ints)
            input_data (any): the raw data; given as lists for each dimension
            marginals (dict): dictionary of marginals
            pair_copulae_strings : a matrix representing the types of pairs of
                copula we use to build our canonical vine copula :
            self.pair_copulae[i][j] is a BaseCopula representing the 2 dimension copula Ci,i+j|1,...,i-1
             0<=i,j<=dimension-1
             for i >=1 this should b a function (float)^i -> CopulaBase
             But for the moment we will consider this function constant.
             In this aproximation, pair copulae will be a matrix of CopulaBase
             if (i,j) is such that i+j>=dimension,  self.paircopulae[i][j]=None
        """
        ndim = len(input_data)
        CopulaBase.__init__(self, ndim, dimkeys)

        if not (pair_copulae_strings is None):
            # Here the user fixes the pair copulae types

            self.pair_copulae = [[None for i in range(self.dimension)] for j in range(self.dimension)]

            for i in range(self.dimension):
                for j in range(i+1,self.dimension):
                    distr_class = distribution_factory(pair_copulae_strings[i][j])

                    key_i =self.dimkeys[i]
                    key_j = self.dimkeys[j]

                    data_pair = {key_i : input_data[key_i],key_j : input_data[key_j]}
                    self.pair_copulae[i][j] = distr_class.fit(dimkeys=[key_i,key_j], data=data_pair)
        else:
            self.get_parameters_from_data() #But most of the time the parameters would be cleverly chosen with input_data.

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        raise NotImplementedError

    def generates_U(self, n=1):
        res= np.zeros((n,self.dimension))
        for l in range(n):
            w = np.random.rand(self.dimension)
            x = np.zeros(self.dimension)
            v = np.zeros((self.dimension, self.dimension))
            x[0] = w[0]
            v[0][0] = w[0]
            for i in range(1, self.dimension):
                v[i][0] = w[i]
                k = i - 1
                while k >= 0:
                    v[i][0] = self.pair_copulae[k][i].inverse_C_partial_derivative(u=v[i][0], v=v[k][k])
                    k = k - 1
                x[i] = v[i][0]
                if i==(self.dimension - 1):
                    res[l]=x
                else:
                    for j in range(i):
                        v[i][j + 1] = self.pair_copulae[j][i].C_partial_derivative(u=v[i][j], v=v[j][j])
        if n==1:
            return res[0]
        else:
            return res

    def cdf(self, valuedict):
        """
        The global C function of the copula
        Args:
            valuedict: arguments of C

        Returns:
            C(valuedict)
        """
        return self.C_from_sample(valuedict)

    @accepts_dict
    def pdf(self, *xs):
        """
        This is an approximation with conditional dependance c(i,j)=1 each time j!=1 
        """

        res = 1

        for i in range(self.dimension - 1):
            res = res * self.pair_copulae[0][i + 1].pdf(xs[1], xs[i + 1])

        return res

@register_distribution(name='dvine-copula')
class DVineCopula(CopulaBase):

    def __init__(self, dimkeys=None, input_data=None, marginals=None, pair_copulae_strings=None):
        """
        Args :
            dimkeys (list): keys for each dimension in dictin (e.g. a list of ints)
            input_data (any): the raw data; given as lists for each dimension
            marginals (dict): dictionary of marginals
            pair_copulae_strings : a matrix representing the types of pairs of copula we use to build our canonical vine copula :
            self.pair_copulae[i][j] is a BaseCopula representing the 2 dimension copula Ci,i+j|1,...,i-1
             0<=i,j<=dimension-1
             for i >=1 this should b a function (float)^i -> CopulaBase
             But for the moment we will consider this function constant.
             In this aproximation, pair copulae will be a matrix of CopulaBase
             if (i,j) is such that i+j>=dimension,  self.paircopulae[i][j]=None
        """
        dimension = len(input_data)
        CopulaBase.__init__(self, dimension, dimkeys)

        if not (pair_copulae_strings is None):
            # Here the user fixes the pair copulae types

            self.pair_copulae = [[None for i in range(self.dimension)] for j in range(self.dimension)]

            for i in range(self.dimension):
                for j in range(i+1,self.dimension):
                    distr_class = distribution_factory(pair_copulae_strings[i][j])

                    key_i =self.dimkeys[i]
                    key_j = self.dimkeys[j]

                    data_pair = {key_i : input_data[key_i],key_j : input_data[key_j]}
                    marginals_pair = {key_i : marginals[key_i], key_j : marginals[key_j]}
                    self.pair_copulae[i][j] = distr_class(dimkeys=[key_i,key_j], input_data=data_pair, marginals=marginals_pair)
        else:
            self.get_parameters_from_data() #But most of the time the parameters would be cleverly chosen with input_data.

    @classmethod
    @fit_wrapper
    def fit(cls, data, dimkeys=None):
        raise NotImplementedError

    def generates_U(self, n=1):
        res= np.zeros((n,self.dimension))
        for l in range(n):
            w = np.random.rand(self.dimension)
            x = np.zeros(self.dimension)
            v = np.zeros((self.dimension, self.dimension))
            x[0] = w[0]
            v[0][0] = w[0]
            v[1][0]= self.pair_copulae[0][1].inverse_C_partial_derivative(u=w[1],v=v[0][0])
            x[1] = v[1][0]
            if self.dimension<=2:
                res[l]=x
            else:
                v[1][1] = self.pair_copulae[0][1].C_partial_derivative(u=v[0][0],v=v[1][0])

            for i in range(3,self.dimension+1):
                v[i-1][0]=w[i-1]
                k = i-1
                while k>=2:
                    v[i - 1][0] = self.pair_copulae[i-k-1][i-1].inverse_C_partial_derivative(u=v[i-1][0],v=v[i-2][2*k-3])
                    k = k-1
                v[i-1][0] = self.pair_copulae[i-2][i-1].inverse_C_partial_derivative(u=v[i-1][0],v=v[i-2][0])
                x[i-1] = v[i-1][0]
                if i==self.dimension:
                    res[l]=x
                else:
                    v[i-1][1] = self.pair_copulae[i-2][i-1].C_partial_derivative(u=v[i-2][0],v=v[i-1][0])
                    v[i-1][2] = self.pair_copulae[i-2][i-1].C_partial_derivative(u=v[i-1][0],v=v[i-2][0])
                    if i>2:
                        for j in range(2,i-1):
                            v[i-1][2*j-1]= self.pair_copulae[i-j-1][i-1].C_partial_derivative(u=v[i-2][2*j-1],v=[i-1][2*j-2])
                            v[i-1][2*j] = self.pair_copulae[i-j-1][i-1].C_partial_derivative(u=v[i-1][2*j-2],v=[i-2][2*j-1])
                    v[i-1][2*i-3]=self.pair_copulae[0][i-1].C_partial_derivative(u=v[i-2][2*i-5],v=v[i-1][2*i-4])

        if n==1:
            return res[0]
        else:
            return res


    def cdf(self, valuedict):
        """
        The global C function of the copula
        Args:
            valuedict: arguments of C

        Returns:
            C(valuedict)
        """
        return self.C_from_sample(valuedict)

    @accepts_dict
    def pdf(self, *value):
        """
        This function computes the pdf of the copula. It accepts a variable
        amount of arguments to support any dimension of copula.

        Args:
            x (List[float]) the points where you want to compute
        Returns:
            float: The density of the copula
        """

        res=1
        for i in range(self.dimension-1):
            res = res *self.pair_copulae[i][i+1].pdf(value[i],value[i+1])

        return res
