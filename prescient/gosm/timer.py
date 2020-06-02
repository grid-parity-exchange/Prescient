#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import time

"""
Here are two ways to calculate the time of a function 
"""

def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        t = time.time() - startTime_for_tictoc
        print("Elapsed time is " + str(t) + " seconds.")
        return t
    else:
        print("Toc: start time not set")


class Timer(object):
    """
    It can be used as a contextmanager, use the following code
    with Timer('foo_stuff'):
        # do some foo
        # do some stuff
    """
    def __init__(self, name=None, output_file=None):
        self.name = name
        self.output_file = output_file

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            output = 'For Timer [{}] elapsed time is {} seconds.'.format(self.name, time.time() - self.tstart)
            if self.output_file is None:
                print(output)
            else:
                with open(self.output_file, 'a') as file:
                    file.write(output + '\n')
