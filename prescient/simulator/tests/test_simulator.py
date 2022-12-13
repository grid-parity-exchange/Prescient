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
import pytest
import pandas as pd
import numpy as np

from prescient.scripts import runner
from prescient.simulator import Prescient

this_file_path = os.path.dirname(os.path.realpath(__file__))

class SimulatorRegressionBase:
    """Test class for running the simulator."""
    # arbitrary comparison threshold
    COMPARISON_THRESHOLD = .1

    def setUp(self):
        self.this_file_path = this_file_path

        self._set_names()
        self._run_simulator()

        test_results_dir = os.path.join(self.test_case_path, self.results_dir_name) 
        control_results_dir = os.path.join(self.test_case_path, self.baseline_dir_name)

        output_files = ["bus_detail",
                        "daily_summary",
                        "hourly_gen_summary",
                        "hourly_summary",
                        "line_detail",
                        "overall_simulation_output",
                        "renewables_detail",
                        "runtimes",
                        "thermal_detail"
                       ]
        self.test_results = {}
        self.baseline_results = {}
        for f in output_files:
            self.test_results[f] = pd.read_csv(f"{test_results_dir}/{f}.csv")
            self.baseline_results[f] = pd.read_csv(f"{control_results_dir}/{f}.csv")

    def _run_simulator(self):
        """Runs the simulator for the test data set."""
        old_cwd = os.getcwd()
        os.chdir(self.test_case_path)

        simulator_config_filename = self.simulator_config_filename
        Prescient().simulate(config_file=simulator_config_filename)

        os.chdir(old_cwd)
    
    def test_simulator(self):
        #test overall output
        self._assert_file_equality("overall_simulation_output")

        #test thermal detail
        self._assert_column_equality("thermal_detail", "Hour")
        self._assert_column_equality("thermal_detail", "Dispatch")
        self._assert_column_equality("thermal_detail", "Headroom")
        self._assert_column_equality("thermal_detail", "Unit Cost")

        # test renewables detail
        self._assert_column_equality("renewables_detail", "Hour")
        self._assert_column_equality("renewables_detail", "Output")
        self._assert_column_equality("renewables_detail", "Curtailment")

        # test hourly summary
        self._assert_file_equality("hourly_summary")

        #test hourly gen summary
        self._assert_column_equality("hourly_gen_summary", "Available headroom")
        self._assert_column_equality("hourly_gen_summary", "Load shedding")
        self._assert_column_equality("hourly_gen_summary", "Reserve shortfall")
        self._assert_column_equality("hourly_gen_summary", "Over generation")

        #test line detail
        self._assert_file_equality("line_detail")

        #assert that the busses are the same
        self._assert_column_equality("bus_detail", "Bus")

        #assert that the shortfall is the same
        self._assert_column_totals("bus_detail", "Shortfall")

        #assert that the LMP is the same
        self._assert_column_totals("bus_detail", "LMP")

        #assert that the Overgeneration is the same
        self._assert_column_totals("bus_detail", "Overgeneration")

    def _assert_file_equality(self, filename):
        columns = list(self.test_results[filename])
        for col_name in columns:
            self._assert_column_equality(filename, col_name)

    def _assert_column_totals(self, filename, column_name):
        diff = abs(self.test_results[filename][column_name].sum() - self.baseline_results[filename][column_name].sum())
        assert diff < self.COMPARISON_THRESHOLD, f"Column: '{column_name}' of file: '{filename}.csv' diverges."

    def _assert_column_equality(self, filename, column_name):
        df_a = self.test_results[filename]
        df_b = self.baseline_results[filename]
        dtype = df_a.dtypes[column_name]
        if dtype.kind in "iuf":
            if not np.allclose(df_a[column_name].to_numpy(dtype=dtype),
                               df_b[column_name].to_numpy(dtype=dtype),
                               atol=self.COMPARISON_THRESHOLD):
                first_diff_idx = df_a[np.logical_not(np.isclose(df_a[column_name].to_numpy(dtype=dtype),
                                                                df_b[column_name].to_numpy(dtype=dtype),
                                                                atol=self.COMPARISON_THRESHOLD))].iloc[0].name
                diff_df = pd.DataFrame([df_a.loc[first_diff_idx], df_b.loc[first_diff_idx]])
                diff_df.index = ['result', 'baseline']
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)
                assert False, f"Column: '{column_name}' of File: '{filename}.csv' diverges at row {first_diff_idx}.\n{diff_df}"
        elif column_name != 'Date' and column_name != 'Hour':
            diff = df_a[column_name].equals(df_b[column_name])
            assert diff, f"Column: '{column_name}' of File: '{filename}.csv' diverges."

# test runner.py with plugin
class TestSimulatorModRtsGmlcCopperSheet(SimulatorRegressionBase, unittest.TestCase):
    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.simulator_config_filename = 'simulate_deterministic.txt'
        self.results_dir_name = 'deterministic_simulation_output'
        self.baseline_dir_name = 'deterministic_simulation_output_baseline'

# Python API tests
base_options = {'simulate_out_of_sample':True,
                'run_sced_with_persistent_forecast_errors':True,
                'start_date':'07-10-2020',
                'num_days': 7,
                'sced_horizon':4,
                'ruc_mipgap':0.0,
                'reserve_factor':0.0,
                'deterministic_ruc_solver':'cbc',
                'deterministic_ruc_solver_options':{"feas":"off", "DivingF":"on", "DivingP":"on", "DivingG":"on", "DivingS":"on", "DivingL":"on", "DivingV":"on"},
                'sced_solver':'cbc',
                'sced_solver_options':{"printingOptions":"normal"},
                'output_solver_logs':True,
                'sced_frequency_minutes':60,
                'ruc_horizon':36,
                'enforce_sced_shutdown_ramprate':True,
                'no_startup_shutdown_curves':True,
               }

# test csv / text file configuration
class TestSimulatorModRtsGmlcCopperSheet_csv_python_config_file(SimulatorRegressionBase, unittest.TestCase):
    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.simulator_config_filename = 'simulate_deterministic_csv.txt'
        self.results_dir_name = 'deterministic_simulation_csv_output'
        self.baseline_dir_name = 'deterministic_simulation_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_case_path)
        options = {'config_file' : self.simulator_config_filename}
        Prescient().simulate(**options)

# test plugin with Python and *.dat files
class TestSimulatorModRtsGmlcNetwork_python(SimulatorRegressionBase, unittest.TestCase):

    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.results_dir_name = 'deterministic_with_network_simulation_output_python'
        self.baseline_dir_name = 'deterministic_with_network_simulation_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_case_path)
        options = {**base_options}
        options['data_path'] = 'deterministic_with_network_scenarios'
        options['output_directory'] = 'deterministic_with_network_simulation_output_python'
        options['plugin'] = {'test':{'module':'test_plugin.py', 
                                     'print_callback_message':True}}
        Prescient().simulate(**options)

# test options are correctly re-freshed, Python, and network
@pytest.mark.xfail(sys.platform in ("darwin", "win32"), reason="unknown -- only seems to fail on GHA")
class TestSimulatorModRtsGmlcNetwork_python_csv(SimulatorRegressionBase, unittest.TestCase):

    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.results_dir_name = 'deterministic_with_network_simulation_output_python_csv'
        self.baseline_dir_name = 'deterministic_with_network_simulation_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_case_path)
        options = {**base_options}
        options['data_path'] = 'deterministic_with_network_scenarios_csv'
        options['output_directory'] = 'deterministic_with_network_simulation_output_python_csv'
        options['input_format'] = 'rts-gmlc'
        Prescient().simulate(**options)

# test shortcut / text file configuration
class TestShortcutSimulator_python_config_file(SimulatorRegressionBase, unittest.TestCase):
    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.simulator_config_filename = 'simulate_shortcut.txt'
        self.results_dir_name = 'deterministic_shortcut_output'
        self.baseline_dir_name = 'deterministic_shortcut_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_case_path)
        options = {'config_file' : self.simulator_config_filename}
        Prescient().simulate(**options)

class TestCustomDataSource(SimulatorRegressionBase, unittest.TestCase):
    def _set_names(self):
        self.test_case_path = os.path.join(self.this_file_path, 'regression_tests_data')
        # in self.test_case_path
        self.results_dir_name = 'custom_data_provider_output'
        self.baseline_dir_name = 'deterministic_simulation_output_baseline'

    def _run_simulator(self):
        options = {**base_options}
        options['output_directory'] = 'custom_data_provider_output'
        options['data_provider'] = os.path.join(self.this_file_path, 'custom_data_provider.py')
        options['data_path'] = 'custom_data.json'

        os.chdir(self.test_case_path)
        Prescient().simulate(**options)


if __name__ == '__main__':
    unittest.main()
