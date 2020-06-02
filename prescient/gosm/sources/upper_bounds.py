#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This file contains utilities related to upper bounds
"""

import datetime

import pandas as pd

def parse_upper_bounds_file(filename):
    """
    This file should parse the upper bounds file and return a value which 
    denotes the upper bound for the day.

    This file is the same format as Prescient 1.0:
        first_date last_date value
    This is repeated for as many different thresholds there are.
    
    Dates should be in mm/dd/yy format.

    Lines beginning with # are ignored.

    Example:
        first_date last_date value
        07/01/12 03/31/13 4711
        04/01/13 04/01/13 4615

    Args:
        filename: The file which contains the upperbounds
        scenario_day: A datetime-like that denotes the day you wish to find
            the upper bound for
    """
    bounds = {}

    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not(line.strip()):
                continue

            if line.startswith('first_date'):
                continue

            start_date, last_date, capacity = line.split()
            start_date = datetime.datetime.strptime(start_date, '%m/%d/%y')
            last_date = datetime.datetime.strptime(last_date, '%m/%d/%y')

            bounds[(start_date, last_date)] = float(capacity)

    return bounds

def parse_diurnal_pattern_file(filename):
    """
    This file will read a diurnal pattern file and pull out a pandas Series
    from the column titled 'diurnal pattern'. The first column should be
    a column of datetimes.
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    if 'diurnal pattern' not in df.columns:
        raise ValueError("{} has no column with header 'diurnal pattern'"
                         .format(filename))
    
    # We check for duplicate datetimes
    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated()]
        raise ValueError("Diurnal pattern file {} has duplicate dates at {}"
            .format(filename, ", ".join(map(str, duplicates))))

    return df['diurnal pattern']
