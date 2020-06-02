#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This class will house any functions and classes related to creating a mesh
over a space for the purposes of numerical integration.
"""

import itertools
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.interpolate import interp1d

class Cell:
    """
    A cell is n-dimensional cube which is specified by its lower left corner
    and the width of the cube.

    Attributes:
        width (float): The width of the cube
        lower_left (np.array): A vector pointing to the lower left of the cube
        midpoint (np.array): The midpoint of the cube
        upper_right (np.array): The upper right corner of the cube
        dimension (int): The dimension of the cube
        volume (float): The volume of the cube
    """
    def __init__(self, lower_left, width):
        """
        Args:
            lower_left (list[float]): A vector denoting the lower left corner
            width (float): The width of the cube
        """
        self.width = width
        self.lower_left = np.array(lower_left)
        self.midpoint = self.lower_left + self.width/2
        self.upper_right = self.lower_left + self.width
        self.dimension = len(lower_left)
        self.volume = self.width ** self.dimension

    @property
    def bounds(self):
        """
        Returns a list of ordered pairs denoting the intervals on which the
        cube is defined.

        Returns:
            list[list[float]]: The bounds of each dimension of the cube
        """
        return [[x, x + self.width] for x in self.lower_left]

    def feasible(self, bound, capacity):
        """
        This determines if this cell is "feasible" for the purposes of
        integration. This has to do with how if a dimension of the cell lies
        beyond the capacity, then we should consider along that dimension
        the value lies at the capacity. Similarly, if the value is less than
        0, it should be truncated to zero.

        Then if the sum of the new values is less than the bound, then it is
        considered feasible.

        Args:
            bound (float): The bound that we are integrating toward, we want
                X1 + X2 + ... + XN <= bound
            capacity (float): The capacity of each individual dimension
        Returns:
            bool: True if the cell is feasible
        """
        lower_left = self.lower_left
        transformed = np.clip(lower_left, 0, capacity)
        return sum(transformed) <= bound

    def __eq__(self, other):
        return (np.allclose(self.lower_left, other.lower_left)
                and np.isclose(self.width, other.width))

    def __str__(self):
        return "Cell({},{})".format(self.lower_left, self.width)

    __repr__ = __str__


class Mesh:
    """
    This will be a container of cells, and have functions related to
    integrating on the collection of cells.
    """
    def __init__(self, cells):
        """
        Args:
            cells (list[Cell]): A nonempty collection of Cell objects. It is
                assumed that the cells are disjoint
        """
        self.cells = cells
        self.num_cells = len(cells)
        if len(cells) == 0:
            raise ValueError("The Mesh must have at least one Cell")
        self.dimension = cells[0].dimension

    def integrate_with(self, f):
        """
        This will approximate the integral of the function of f. On each Cell
        in the mesh, it will approximate the integral as
            Vol(Cell) * f(midpoint(Cell))

        f is expected to be a function of n variables where n is the dimension
        of the mesh.

        Args:
            f (function): A function to integrate with
        Returns:
            float: An approximation of the integral
        """
        return sum(cell.volume * f(*cell.midpoint) for cell in self.cells)

    def integrate_with_parallel(self, f, num_processes=4):
        """
        Parallel implementation of the integrate_with method.
        Args:
            f (function):
        Returns:
            float:
        """
        submeshes = []
        for i in range(num_processes):
            lower = i * self.num_cells // num_processes
            upper = (i + 1) * self.num_cells // num_processes
            submeshes.append(self.cells[lower:upper])

        def integrate_on(mesh):
            return sum(cell.volume * f(*cell.midpoint) for cell in mesh)

        with multiprocessing.Pool(num_processes) as pool:
            submesh_sums = [pool.apply_async(integrate_on, (submesh,))
                            for submesh in submeshes]
            results = [result.get() for result in submesh_sums]
        return sum(results)

    def feasible_mesh(self, bound, capacity):
        """
        This function returns a Mesh consisting of all the feasible cells
        contained in the mesh respective to the bound and capacity.

        Args:
            bound (float): The bound on the sum of each dimension
            capacity (float): The capacity on any individual dimension
        Returns:
            Mesh: The mesh of feasible cells
        """
        return Mesh([cell for cell in self.cells
                     if cell.feasible(bound, capacity)])

    def __str__(self):
        return "Mesh:\n" + "\n".join("{}".format(cell) for cell in self.cells)

    def __repr__(self):
        return "Mesh({} cells)".format(self.num_cells)

    def plot(self, axis):
        """
        This function will plot the mesh to an axis. It will plot each
        individual cell contained in the mesh for the purposes of
        visualization.

        This method will only work for 2-dimensional meshes.

        Args:
            axis (matplotlib.pyplot.axis): A matplotlib axis to plot to
        """
        if self.dimension != 2:
            raise ValueError("Plotting only works for meshes of dimension 2")

        lower_left = np.min([cell.lower_left for cell in self.cells], axis=0)
        upper_right = np.max([cell.upper_right for cell in self.cells], axis=0)

        axis.set_xlim(lower_left[0], upper_right[0])
        axis.set_ylim(lower_left[1], upper_right[1])

        for cell in self.cells:
            rect = mpatches.Rectangle(cell.lower_left, cell.width,
                                      cell.width, edgecolor='black')

            axis.add_patch(rect)

        return axis


class CubeMesh(Mesh):
    """
    This class will represent a subdivision of a cube into cells. It will
    have the same amount of subdivisions along each dimension.
    The cube is assumed to have its lower left corner at the origin.
    """
    def __init__(self, lower_lefts, length, n):
        """
        Args:
            lower_lefts (list[float]): A vector indicating the lower left
                corner
            length (float): The length of any side of the cube
            n (int): The number of subdivisions on a single dimension, there
                will be n^d cells total where d is the dimension of the cube
        """
        self.delta = delta = length/n
        # n is the number of subdivisions
        self.n = n
        self.length = length

        self.lower_left = np.array(lower_lefts)

        subdivisions = np.arange(0, length, self.delta)
        side_points = [(x + subdivisions).tolist() for x in self.lower_left]

        # This will be the lower left of each cell
        lowers = list(itertools.product(*side_points))
        cells = [Cell(lower, self.delta) for lower in lowers]
        Mesh.__init__(self, cells)

    def outer_shell(self):
        """
        This function will return the mesh of cubes that compose the shell
        that directly surround this cube mesh. Each cell in the shell has the
        same volume as the cells contained in the cube mesh.

        Returns:
            Mesh: The shell of cubes outside the mesh
        """
        new_lower_left = self.lower_left - self.delta
        return ShellMesh(new_lower_left, self.length + 2*self.delta, self.n+2)

    def add_shell(self):
        """
        This function will create a mesh as the union of this cube mesh and
        its outer shell.

        Returns:
            CubeMesh: A Mesh with the shell of cells surrounding the current
                cube mesh
        """
        new_lower_left = self.lower_left - self.delta
        return CubeMesh(new_lower_left, self.length + 2*self.delta, self.n+2)


class ShellMesh(Mesh):
    """
    Mesh representing a shell surrounding a smaller cube mesh.
    It is constructed by building an outer and inner cube mesh and removing
    any cells in the inner cube from the outer cube.
    """
    def __init__(self, lower_lefts, length, n):
        self.delta = length/n
        self.lower_left = np.array(lower_lefts)
        subdivisions = np.arange(0, length, self.delta)
        side_points = [x + subdivisions for x in self.lower_left]
        dim = len(self.lower_left)

        cells = []

        for num in range(1, dim+1):
            for groups in itertools.combinations(enumerate(side_points), num):
                group_indices = [i for i, _ in groups]
                rest = [(i, group) for i, group in enumerate(side_points)
                        if i not in group_indices]
                ends = [(i, [group[0], group[-1]]) for i, group in groups]
                middles = [(i, group[1:-1]) for i, group in rest]

                corners = sorted(ends + middles)
                corners = [group for _, group in corners]
                new_cells = [Cell(corner, self.delta)
                             for corner in itertools.product(*corners)]
                cells.extend(new_cells)
        Mesh.__init__(self, cells)


def generate_mesh(length, f, ndim, n=100, epsilon=1e-3):
    """
    This function attempts to find the domain of the function f for which
    the value is nonzero (or at least not close to 0). To this end, it
    generates a CubeMesh object and slowly expands using an approximation of
    the integral to estimate the support on the mesh.
    Once the approximated integral on the outer shell of the mesh is less than
    the specified epsilon, then the current mesh will be returned.

    f is assumed to be a function which approaches zero in any direction away
    from the origin.

    Args:
        length (float): The length of any side of the cube.
        f (function): The function which is used to compute the integral over
                the added shell.
        ndim (int): The number of dimensions the function expects
        n (int): The number of subdivisions on a single dimension, there
                will be n^d cells total where d is the dimension of the cube
        epsilon (float): The number for the breaking criterion.
    Returns:
        CubeMesh: A mesh which covers the support of the function
    """
    curr_mesh = CubeMesh([0]*ndim, length, n)
    while True:
        outer_shell = curr_mesh.outer_shell()
        outer_sum = outer_shell.integrate_with(f)
        if outer_sum < epsilon:
            break
        curr_mesh = curr_mesh.add_shell()
    return curr_mesh


def multi_cdf(m, bound, capacity, f):
    """
    This function computes the value of the cdf of the multivariate random
    variable (X1+X2+...Xn). For a given distribution it gives back the
    probability, if the sum of the random variables is lower or equal to a
    specific bound. Therefore it uses the feasible_mesh, so it includes
    the capacity for each dimension for computation.

    Args:
        m (mesh): The initial mesh.
        bound (float): The bound for the sum of the random variables.
        capacity (float): The capacity of the power generation for each hour
            of interest.
        f (pdf): The distribution of the random variable (X1+...+Xn), which
           has to be a pdf.
    Returns:
        integral (float): The probability, if the sum of the random variable
            (X1+...+Xn) is lower or equal than the bound.
    """
    integral=0
    fsbmesh = m.feasible_mesh(bound, capacity)
    integral += fsbmesh.integrate_with(f)
    return integral


def cdf_inverse(m, alpha, capacity, f, subint):
    """
    This function computes the inverse value of a specific probability for
    a given distribution.

    Args:
        m (mesh): The initial mesh.
        alpha (float): The probability for which the inverse value is computed.
        capacity (float): The capacity of the power generation for each hour
            of interest.
        f (pdf): The distribution of the random variable (X1+...+Xn), which has
            to be a pdf.
            subint (int): The number of subintervalls, which are used to
                interpolate the cdf.
    Returns:
        inverse_bound (float): The computed inverse value of alpha.
    """
    x = np.linspace(0, capacity, subint)
    y = []
    for i in x:
        yi = multi_cdf(m, i, capacity, f)
        j = int(np.argwhere(x==i))
        y.append(yi)
        if (j == 0) and (yi > alpha):
            inverse_alpha = 0
            break
        elif (j != 0):
            if y[j-1] <= alpha <= y[j]:
                lin = interp1d([y[j-1], y[j]], [x[j-1], x[j]])
                inverse_alpha = lin(alpha)
                break
    else:
        inverse_alpha = capacity

    return inverse_alpha
