#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#!/usr/bin/python3

import sys
import os

import numpy as np
import pandas as pd

wind_command = 'runner.py gosm_test/BPA_populator.txt'
solar_command = 'runner.py solar_test/run_populator_solar_caiso.txt' 

def check_for_equality(dir1, dir2):
    """

    """
    print("--------------------------------------------------------------")
    print("Checking for differences, to be listed below")
    print("--------------------------------------------------------------")
    dir1 = dir1 + os.sep + 'pyspdir_twostage'
    dir2 = dir2 + os.sep + 'pyspdir_twostage'

    dir1_dates = os.listdir(dir1)
    dir2_dates = os.listdir(dir2)

    dates_to_check = []

    for date in dir1_dates:
        if date not in dir2_dates:
            print("Error: {} in {} but not in {}".format(date, dir1, dir2))
        else:
            dates_to_check.append(date)

    for date in dir2_dates:
        if date not in dir1_dates:
            print("Error: {} in {} but not in {}".format(date, dir2, dir1))

    for date in dates_to_check:
        date_dir1 = os.path.join(dir1, date)
        date_dir2 = os.path.join(dir2, date)

        scens_1 = pd.read_csv(date_dir1 + os.sep + 'scenarios.csv')
        scens_2 = pd.read_csv(date_dir2 + os.sep + 'scenarios.csv')

        if list(scens_1.columns) != list(scens_2.columns):
            print("Error: on {}, the column names do not match".format(date))

        difference = (scens_1 - scens_2).values

        if not np.allclose(difference, 0, atol=1e-4):
            print("Error: on {}, the values differ by more than .0001".format(date))


def test_wind():
    os.system(wind_command)
    check_for_equality('basic_wind_scenarios', 'new_wind_scenarios')


def test_solar():
    os.system(solar_command)
    check_for_equality('basic_solar_scenarios', 'new_solar_scenarios')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_prescient.py type")
        print("type must be one of 'solar' or 'wind'")
        sys.exit(1)
    command = sys.argv[1]

    if command == 'solar':
        test_solar()
    elif command == 'wind':
        test_wind()
    else:
        print("Usage: python test_prescient.py type")
        print("type must be one of 'solar' or 'wind'")
        sys.exit(1) 
