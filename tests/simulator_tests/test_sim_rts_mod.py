#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os
import subprocess
import sys
import unittest
import pandas as pd
import numpy as np

from prescient.downloaders import rts_gmlc
from prescient.scripts import runner
from tests.simulator_tests import simulator_diff

this_file_path = os.path.dirname(os.path.realpath(__file__))

class _SimulatorModRTSGMLC:
    """Test class for running the simulator."""
    # arbitrary comparison threshold
    COMPARISON_THRESHOLD = .01

    def setUp(self):
        self.this_file_path = this_file_path
        self.test_cases_path = os.path.join(self.this_file_path, 'test_cases')

        self._set_names()
        self._run_simulator()

        test_results_dir = os.path.join(self.test_cases_path, self.results_dir_name) 
        control_results_dir = os.path.join(self.test_cases_path, self.baseline_dir_name)

        self.bus_detail_a = pd.read_csv(test_results_dir + "/bus_detail.csv")
        self.Daily_summary_a = pd.read_csv(test_results_dir + "/Daily_summary.csv")
        self.Hourly_gen_summary_a = pd.read_csv(test_results_dir + "/Hourly_gen_summary.csv")
        self.hourly_summary_a = pd.read_csv(test_results_dir + "/hourly_summary.csv")
        self.line_detail_a = pd.read_csv(test_results_dir + "/line_detail.csv")
        self.Overall_simulation_output_a = pd.read_csv(test_results_dir + "/Overall_simulation_output.csv")
        self.renewables_detail_a = pd.read_csv(test_results_dir + "/renewables_detail.csv")
        self.runtimes_a = pd.read_csv(test_results_dir + "/runtimes.csv")
        self.thermal_detail_a = pd.read_csv(test_results_dir + "/thermal_detail.csv")

        self.bus_detail_b = pd.read_csv(control_results_dir + "/bus_detail.csv")
        self.Daily_summary_b = pd.read_csv(control_results_dir + "/Daily_summary.csv")
        self.Hourly_gen_summary_b = pd.read_csv(control_results_dir + "/Hourly_gen_summary.csv")
        self.hourly_summary_b = pd.read_csv(control_results_dir + "/hourly_summary.csv")
        self.line_detail_b = pd.read_csv(control_results_dir + "/line_detail.csv")
        self.Overall_simulation_output_b = pd.read_csv(control_results_dir + "/Overall_simulation_output.csv")
        self.renewables_detail_b = pd.read_csv(control_results_dir + "/renewables_detail.csv")
        self.runtimes_b = pd.read_csv(control_results_dir + "/runtimes.csv")
        self.thermal_detail_b = pd.read_csv(control_results_dir + "/thermal_detail.csv")

    def _run_simulator(self):
        """Runs the simulator for the test data set."""
        os.chdir(self.test_cases_path)

        simulator_config_filename = self.simulator_config_filename
        script, options = runner.parse_commands(simulator_config_filename)

        if sys.platform.startswith('win'):
            subprocess.call([script] + options, shell=True)
        else:
            subprocess.call([script] + options)

        os.chdir(self.this_file_path)
    
    def test_simulator(self):
        #test overall output
        self._assert_file_equality(self.Overall_simulation_output_b, self.Overall_simulation_output_a)

        #test thermal detail
        self._assert_column_equality(self.thermal_detail_a, self.thermal_detail_b, "Hour")
        self._assert_column_totals(self.thermal_detail_a, self.thermal_detail_b, "Dispatch")
        self._assert_column_totals(self.thermal_detail_a, self.thermal_detail_b, "Headroom")
        self._assert_column_totals(self.thermal_detail_a, self.thermal_detail_b, "Unit Cost")

        # test renewables detail
        self._assert_column_equality(self.renewables_detail_a, self.renewables_detail_b, "Hour")
        self._assert_column_totals(self.renewables_detail_b, self.renewables_detail_a, "Output")
        self._assert_column_totals(self.renewables_detail_b, self.renewables_detail_a, "Curtailment")

        # test hourly summary
        self._assert_file_equality(self.hourly_summary_a, self.hourly_summary_b)

        #test hourly gen summary
        self._assert_column_equality(self.Hourly_gen_summary_a, self.Hourly_gen_summary_b, "Available reserves")
        self._assert_column_equality(self.Hourly_gen_summary_a, self.Hourly_gen_summary_b, "Load shedding")
        self._assert_column_equality(self.Hourly_gen_summary_a, self.Hourly_gen_summary_b, "Reserve shortfall")
        self._assert_column_equality(self.Hourly_gen_summary_a, self.Hourly_gen_summary_b, "Over generation")


        #test line detail
        self._assert_file_equality(self.line_detail_a, self.line_detail_b)

        #assert that the busses are the same
        self._assert_column_equality(self.bus_detail_a, self.bus_detail_b, "Bus")

        #assert that the shortfall is the same
        self._assert_column_totals(self.bus_detail_a, self.bus_detail_b, "Shortfall")

        #assert that the LMP is the same
        self._assert_column_totals(self.bus_detail_a, self.bus_detail_b, "LMP")

        #assert that the Overgeneration is the same
        self._assert_column_totals(self.bus_detail_a, self.bus_detail_b, "Overgeneration")

    def _assert_file_equality(self, df_a, df_b):
        columns = list(df_a)
        for i in columns:
            self._assert_column_equality(df_a, df_b, i)

    def _assert_column_totals(self, df_a, df_b, column_name):
        diff = abs(df_a[column_name].sum() - df_b[column_name].sum())
        assert diff < self.COMPARISON_THRESHOLD, "Column: " + column_name + " diverges."

    def _assert_column_equality(self, df_a, df_b, column_name):
        dtype = df_a.dtypes[column_name]
        if dtype == 'float' or dtype == 'int':
            diff = np.allclose(df_a[column_name].to_numpy(dtype=dtype), df_b[column_name].to_numpy(dtype=dtype), atol=self.COMPARISON_THRESHOLD)
            assert diff, "Column: " + column_name + " diverges."
        elif column_name != 'Date' and column_name != 'Hour':
            diff = df_a[column_name].equals(df_b[column_name])
            assert diff, "Column: " + column_name + " diverges."


class TestSimulatorModRTSGMLCNetwork(_SimulatorModRTSGMLC, unittest.TestCase):
    def _set_names(self):
        self.simulator_config_filename = 'simulate_with_network_deterministic.txt'
        self.results_dir_name = 'deterministic_with_network_simulation_output'
        self.baseline_dir_name = 'deterministic_with_network_simulation_output_baseline'

class TestSimulatorModRTSGMLCCopperSheet(_SimulatorModRTSGMLC, unittest.TestCase):
    def _set_names(self):
        self.simulator_config_filename = 'simulate_deterministic.txt'
        self.results_dir_name = 'deterministic_simulation_output'
        self.baseline_dir_name = 'deterministic_simulation_output_baseline'

if __name__ == '__main__':
    unittest.main()
