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

from prescient.simulator.prescient import Prescient

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
        self._assert_column_equality("hourly_gen_summary", "Available reserves")
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
        if dtype == 'float' or dtype == 'int':
            diff = np.allclose(df_a[column_name].to_numpy(dtype=dtype), df_b[column_name].to_numpy(dtype=dtype), atol=self.COMPARISON_THRESHOLD)
            assert diff, f"Column: '{column_name}' of File: '{filename}.csv' diverges."
        elif column_name != 'Date' and column_name != 'Hour':
            diff = df_a[column_name].equals(df_b[column_name])
            assert diff, f"Column: '{column_name}' of File: '{filename}.csv' diverges."


class TestSimulatorModRtsGmlcCopperSheet_csv(_SimulatorModRTSGMLC, unittest.TestCase):
    def _set_names(self):
        self.simulator_config_filename = 'simulate_deterministic_csv.txt'
        self.results_dir_name = 'deterministic_simulation_csv_output'
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
                'deterministic_ruc_solver_options':["feas=off", "DivingF=on", "DivingP=on", "DivingG=on", "DivingS=on", "DivingL=on", "DivingV=on"],
                'sced_solver':'cbc',
                'sced_solver_options':["printingOptions=normal"],
                'output_solver_logs':True,
                'sced_frequency_minutes':60,
                'ruc_horizon':36,
                'enforce_sced_shutdown_ramprate':True,
                'no_startup_shutdown_curves':True,
               }

class TestSimulatorModRtsGmlcCopperSheet_python(_SimulatorModRTSGMLC, unittest.TestCase):

    def _set_names(self):
        self.results_dir_name = 'deterministic_simulation_output_python'
        self.baseline_dir_name = 'deterministic_simulation_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_cases_path)
        options = {**base_options}
        options['data_directory'] = 'deterministic_scenarios'
        options['output_directory'] = 'deterministic_simulation_output_python'
        options['plugin'] = 'test_plugin.py'
        options['print_callback_message'] = True
        Prescient().simulate(**options)

class TestSimulatorModRtsGmlcNetwork_python(_SimulatorModRTSGMLC, unittest.TestCase):

    def _set_names(self):
        self.results_dir_name = 'deterministic_with_network_simulation_output_python'
        self.baseline_dir_name = 'deterministic_with_network_simulation_output_baseline'

    def _run_simulator(self):
        os.chdir(self.test_cases_path)
        options = {**base_options}
        options['data_directory'] = 'deterministic_with_network_scenarios'
        options['output_directory'] = 'deterministic_with_network_simulation_output_python'
        Prescient().simulate(**options)

if __name__ == '__main__':
    unittest.main()
