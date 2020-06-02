#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
multidim_utils.py

This module will house functions and classes for the purposes of making
handling multidimensional objects easier.
"""

class MultiDimMixIn:
    """
    This class is to be inherited from and will provide utilities for
    accessing the dimensions of the multidimensional object. Most notably
    it will store a dimkeys attribute which will be a list of the names
    of the dimension (by default this will be just the integers [0,1,..,n-1]).

    In addition there will be methods for changing the names of the dimensions
    as well as checking if the dimensions of two objects match.
    """
    def __init__(self, ndim, dimkeys=None):
        """
        Args:
            ndim (int): The number of dimensions the object has
            dimkeys (list): A list of names of the dimensions
        """
        self.ndim = ndim
        if dimkeys is None:
            # We default to using the integers if no dimkeys are passed in.
            self.dimkeys = list(range(ndim))
        else:
            self.dimkeys = dimkeys

    def set_dimkeys(self, dimkeys):
        """
        This will set the names of the dimensions to the ones passed in.

        Args:
            dimkeys (list): A list of names of the dimensions
        """
        if len(dimkeys) != self.ndim:
            raise ValueError("The number of dimension names does not match "
                             "the number of dimensions for this object.")
        self.dimkeys = dimkeys

    def match(self, other):
        """
        This method returns True if the names of the dimensions for this
        object matches that of the other object.

        Args:
            other (MultiDimMixIn): An object which has a dimkeys attribute
        """
        return (self.ndim == other.ndim 
                and sorted(self.dimkeys) == sorted(other.dimkeys))
