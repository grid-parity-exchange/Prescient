#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import math

def round_small_values(x, p=1e-6):
    # Rounds values that are within (-1e-6, 1e-6) to 0.
    try:
        if math.fabs(x) < p:
            return 0.0
        return x
    except:
        raise RuntimeError("Utility function round_small_values failed on input=%s, p=%f" % (str(x), p))

def within_tolerance(v1, v2, tol=1e-5):
    return math.fabs(v1 - v2) <= tol
