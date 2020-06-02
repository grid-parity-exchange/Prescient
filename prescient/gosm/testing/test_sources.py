#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
test_sources.py

This file will contain methods for testing functionality
of the Source and RollingWindow classes in sources.py.
It will use the Python unittest module for this purpose.
"""

import unittest
import datetime

import numpy as np
import pandas as pd

import gosm.sources as sources

class TestSingleColumn(unittest.TestCase):
    def setUp(self):
        xs = list(range(1, 1001))
        df = pd.DataFrame(xs, index=pd.date_range('2000-01-01', periods=1000,
                                                  freq='H'))
        df.columns = ['forecasts']

        self.source = sources.Source('test', df, 'solar')

    def test_inside_window(self):
        window = self.source.window('forecasts', 200, 500)
        column = window.get_column('forecasts')
        self.assertEqual(len(column), 301)
        for num in range(200, 501):
            self.assertIn(num, column.values)

        # Test that all the values in the column are between 200 and 500
        for num in column.values:
            self.assertTrue(200 <= num <= 500)
        # Test that object is constructed correctly
        self.assertEqual(window.name, 'test')
        self.assertEqual(window.source_type, 'solar')

    def test_lower_window(self):
        window = self.source.window('forecasts', upper_bound=500)
        column = window.get_column('forecasts')
        self.assertEqual(len(column), 500)

        for num in range(1, 501):
            self.assertIn(num, column.values)

        for num in column.values:
            self.assertTrue(num <= 500)

    def test_upper_window(self):
        window = self.source.window('forecasts', lower_bound=501)
        column = window.get_column('forecasts')
        self.assertEqual(len(column), 500)

        for num in range(501, 1001):
            self.assertIn(num, column.values)

        for num in column.values:
            self.assertTrue(num > 500)

    def test_enumerate(self):
        window = self.source.enumerate('forecasts', 200)
        column = window.get_column('forecasts')
        self.assertEqual(len(column), 1)

        self.assertIn(200, column.values)

    def test_rolling_window(self):
        scenario_day = pd.Timestamp('2000-01-05 00:00:00')
        window = self.source.rolling_window(scenario_day)
        self.assertEqual(len(window.historic_data.index), 4*24)
        for dt in window.historic_data.index:
            self.assertLess(dt, pd.Timestamp(scenario_day))

        for hour in range(24):
            self.assertIn(scenario_day + datetime.timedelta(hours=hour),
                          window.dayahead_data.index)


if __name__ == '__main__':
    unittest.main()
