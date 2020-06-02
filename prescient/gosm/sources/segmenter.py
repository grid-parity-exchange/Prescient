#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This file should generalize segmentation behavior for any and all data.
This module exports a Segmentation Object which is to be created from a file
which details the criteria by which the data should be segmented.

It was copied from PINT on 03/22/2017.
"""

import sys

import numpy as np
import pandas as pd


def parse_criterion(string):
    """
    This function parses out a string into a Criterion object
    for use in segmentation.

    Args:
        string: The string that can be split into 4 pieces. It should be of the form:
                  'name,column_name,method,window_size
    """
    try:
        pieces = string.rstrip().split(",")
        name = pieces[0]
        column_name = pieces[1]
        method = pieces[2]
        # If using window, need a 4th argument
        if len(pieces) == 4:
            window_size = float(pieces[3])
        else:
            window_size = None

        return Criterion(name, column_name, method, window_size)
    except ValueError:
        print("Segmentation Criteria Filename is not structured correctly")
        print("The correct format for the file is as follows:")
        print("name,col_name,window,window_size")
        print("You supplied: " + string)
        raise RuntimeError("Segmentation File formatted incorrectly")


def parse_segmentation_file(filename):
    """
    This function reads in a file and parses out the segmentation criteria
    returning them in a list of criteria.

    Args:
        filename (str): The name of the file which contains the segmentation criteria
    Returns:
        (List[Criterion]): A list of Criterion objects
    """
    criteria = []
    with open(filename, "r") as f:
        for line in f:
            # Gobble comments and empty lines
            line, *_ = line.split('#')
            if line == '':
                continue
            criteria.append(parse_criterion(line))
    return criteria


class Criterion:
    """
    This class just holds all the relevant fields for the Segmenter object to work with
    The constructor accepts a name, field name (what column it is in the spreadsheets) and
    cutpoint width.
    """
    def __init__(self, name, column_name, method, window_size=None):
        """
        This constructor splits a string provided by a file into pieces that can be further used by the program.

        Args:
        """
        self.name = name
        self.column_name = column_name
        self.method = method
        self.window_size = window_size

    def __repr__(self):
        return "Criterion({},{},{})".format(self.name, self.column_name,
                                            self.method)

    def __str__(self):
        string = "Criterion:\n"
        string += '\tName: {}\n'.format(self.name)
        string += '\tColumn: {}\n'.format(self.column_name)
        string += '\tMethod: {}\n'.format(self.method)
        if self.window_size is not None:
            string += '\tWindow Size: {}\n'.format(self.window_size)
        return string
