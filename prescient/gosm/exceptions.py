#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
exceptions.py

This module should export any and all exceptions related to the process
of creating scenarios. The purpose of these exceptions is to make it clearer
to the user what problem is should for any single day, scenarios fail to be
created.

It is hopeful that these will be clearer than simply raising RuntimeError
at any problem.
"""


class InputError(Exception):
    def __init__(self, filename, data_type, message):
        self.filename = filename
        self.data_type = data_type
        self.message = message


class DataError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Data', message)


class HyperrectangleError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Hyperrectangles', message)


class PathsError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Paths', message)


class OptionsError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Options', message)


class SegmentError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Segmentation', message)


class SourcesError(InputError):
    def __init__(self, filename, message):
        InputError.__init__(self, filename, 'Sources', message)


def print_input_error(error):
    print("INPUT ERROR:", file=sys.stderr)
    print("{} was raised.".format(error.__name__), file=sys.stderr)
    print("There is a problem with the {} file: {}".format(error.data_type,
                                                           error.filename),
          file=sys.stderr)
    print("'{}'".format(error.message), file=sys.stderr)

