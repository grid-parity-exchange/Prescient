#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
graph_utilities.py
This module will export a WeightedGraph class implemented for the sole
purpose of using the Markov Clustering Algorithm.
More functionality may be added later on.
"""

import numpy as np


class WeightedGraph:
    """
    This class represents a weighted graph for the purposes
    of determining clusters via the Markov Clustering Algorithm.
    To initialize an object of this class, pass in a dictionary
    which maps pairs (tuples) of vertices to the corresponding weight.

    Stores internally both an adjacency list and an adjacency matrix
    This is fine as the number of expected vertices is small.

    """
    def __init__(self, pair_weights):
        self.adjacency_list = self._construct_adjacency_list(pair_weights)
        self.vertices = list(self.adjacency_list.keys())
        self.num_vertices = len(self.vertices)
        self.adjacency_matrix = self._construct_adjacency_matrix()

    def get_clusters(self, granularity):
        """
        This method uses the Markov Clustering Algorithm
        to cluster vertices together.
        Args:
            granularity: The granularity with which to inflate columns
        Return:
            A dictionary which maps a vertex to the set of vertices it is in a cluster with
        """
        # Hardcoded in the expansion parameter, this reflects original implementation
        # May wish to change this to have some option
        e = 2

        matrix = transform_matrix(self.adjacency_matrix)
        matrix = normalize_columns(matrix)
        error_convergence = np.linalg.norm(matrix)
        while error_convergence > 10E-6:
            # Store previous matrix
            previous_matrix = matrix
            matrix = np.linalg.matrix_power(matrix, e)
            matrix = inflate_columns(matrix, granularity)
            error_convergence = np.linalg.norm(matrix - previous_matrix)

        return self._get_clusters(matrix)

    def _get_clusters(self, matrix):
        """
        Helper function to retrieve the list of clusters from the matrix
        """
        # clusters is a set to have only unique sets in the partition of the vertices
        clusters = set()
        for i, v1 in enumerate(self.vertices):
            # Already assigned a cluster
            if np.sum(matrix[i, :]) < 10E-6:  # If sum of row is essentially zero
                continue
            else:
                cluster = []
                for j, v2 in enumerate(self.vertices):
                    if matrix[i, j] > 10E-6:
                        cluster.append(v2)
                clusters.add(frozenset(cluster))

        clusters = [list(cluster) for cluster in clusters]
        return clusters

    def _construct_adjacency_list(self, pair_weights):
        """
        Constructs an adjacency list representation of the graph as
        a dictionary which maps vertices to a list of tuples (v, w) where
        v is the adjacent vertex and w is the weight of the edge.

        Args:
            pair_weights: A dictionary mapping pairs of vertices to weights
        Returns:
             An adjacency list
        """
        adjacency_list = {}
        for v1, v2 in pair_weights:
            weight = pair_weights[(v1, v2)]
            if v1 in adjacency_list:
                adjacency_list[v1].append((v2, weight))
            else:
                adjacency_list[v1] = [(v2, weight)]

            if v2 in adjacency_list:
                adjacency_list[v2].append((v1, weight))
            else:
                adjacency_list[v2] = [(v1, weight)]

        return adjacency_list

    def _construct_adjacency_matrix(self):
        """
        Constructs an adjacency matrix from the internally stored adjacency list
        Assigns M_ij to be the weight from vertex i to vertex j.

        Returns:
             The numpy matrix storing the weights
        """
        adjacency_matrix = np.identity(self.num_vertices)
        for i, v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices):
                v1_v2_weight = 0
                for vertex, weight in self.adjacency_list[v1]:
                    if v2 == vertex:
                        v1_v2_weight = weight
                        break

                adjacency_matrix[i][j] = v1_v2_weight
        return adjacency_matrix


def transform_matrix(matrix):
    """
    This function accepts a weighted graph stored as a matrix
    and updates the weights according to a certain transform.
    If the weight is larger than a certain threshold, in this case
    the mean of the weights, then it is set to 0.
    Otherwise, it is set to (-1 / MAX) * w_ij + 1
    where MAX is the maximum weight
    It returns a transformed matrix
    Args:
        matrix: A numpy matrix
    Returns:
        A transformed numpy matrix
    """
    num_vertices = matrix.shape[0]
    transformed_matrix = np.identity(num_vertices)
    max_weight = np.max(matrix)
    mean_weight = np.mean(matrix)
    for i in range(num_vertices):
        for j in range(num_vertices):
            weight = matrix[i][j]
            if weight > mean_weight:
                transformed_matrix[i][j] = 0
            else:
                transformed_matrix[i][j] = (-weight / max_weight) + 1
    return transformed_matrix


def normalize_columns(matrix):
    """
    Accepts a matrix and normalizes each column of the matrix by the sum of the column
    if the sum is not 0. Returns a new matrix
    Args:
        a numpy matrix
    Return:
        a normalized numpy matrix
    """
    num_cols = matrix.shape[1]
    normalized_matrix = np.identity(num_cols)
    for j in range(num_cols):
        if np.sum(matrix[:, j]) != 0:
            normalized_matrix[:, j] = matrix[:, j] / np.sum(matrix[:, j])
    return normalized_matrix


def inflate_columns(matrix, r):
    """
    This routine raises each individual element of the matrix to a power r
    and then renormalizes each of the columns.
    Args:
        matrix: A square numpy matrix
        r: The inflation parameter
    Return:
        An inflated matrix
    """
    num_cols = matrix.shape[1]
    inflated_matrix = np.identity(num_cols)
    for j in range(num_cols):
        inflated_matrix[:, j] = np.power(matrix[:, j], r)
        inflated_matrix = normalize_columns(inflated_matrix)
    return inflated_matrix
