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

def interpolate_between(val_before, val_after, fraction_between:float):
    ''' Interpolate between two values.

    Arguments
    ---------
    val_before: number
        The first value to interpolate between
    val_after: number
        The second value to interpolate between
    fraction_between: float
        The fractional position between the two values that you want to interpolate to.

    Returns
    -------
    * If fraction_between is 0.0, then val_before is returned directly, maintaining its original type.
    * If fraction_between is 1.0, then val_after is returned directly, maintaining its original type.
    * Any other value returns a float that interpolates (or potentially extrapolates) between the
      two supplied values.
    '''
    if fraction_between == 0.0:
        return val_before
    if fraction_between == 1.0:
        return val_after
    else:
        return (1.0 - fraction_between)*val_before + fraction_between*val_after
