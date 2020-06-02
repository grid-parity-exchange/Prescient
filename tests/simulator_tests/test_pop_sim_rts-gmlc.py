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


def write_diffcheck_json():
    """Writes the JSON template files for numeric field and tolerance definitions for the simulator diffcheck function."""
    import json

    numeric_fields = {}

    numeric_fields['bus_detail'] = ['LMP', 'Overgeneration', 'Shortfall']
    numeric_fields['Daily_summary'] = ['Demand', 'Renewables available', 'Renewables used', 
                                        'Renewables penetration rate', 'Average price', 'Fixed costs',
                                        'Generation costs', 'Load shedding', 'Over generation',
                                        'Reserve shortfall', 'Renewables curtailment', 'On/off',
                                        'Sum on/off ramps', 'Sum nominal ramps', 'Number on/offs',
                                        ]
    numeric_fields['Hourly_gen_summary'] = ['Hour', 'Load shedding', 'Over generation', 'Reserve shortfall']
    numeric_fields['hourly_summary'] = [' TotalCosts ', ' FixedCosts ', ' VariableCosts ', 
                                        ' LoadShedding ', ' OverGeneration ', ' ReserveShortfall ',
                                        ' RenewablesUsed ', ' RenewablesCurtailed ', ' Demand ', 
                                        ' Price',
                                        ]
    numeric_fields['line_detail'] = ['Flow']
    numeric_fields['Overall_simulation_output'] = ['Total demand', 'Total fixed costs', 'Total generation costs',
                                                    'Total costs', 'Total load shedding', 'Total over generation',
                                                    'Total reserve shortfall', 'Total renewables curtialment', 'Total on/offs',
                                                    'Total sum on/off ramps', 'Total sum nominal ramps', 'Maximum observed demand',
                                                    'Overall renewables penetration rate', 'Cumulative average price',
                                                    ]
    numeric_fields['Quickstart_summary'] = ['Dispatch level of quick start generator', ]
    numeric_fields['renewables_detail'] = ['Output', 'Curtailment']
    numeric_fields['runtimes'] = ['Solve Time',]
    numeric_fields['thermal_detail'] = ['Dispatch', 'Headroom', 'Unit State']

    with open('numeric_fields.json', 'w') as f:
        json.dump(numeric_fields, f, indent=2)

    tolerances = {}

    tolerances['bus_detail'] = {'LMP': 1e-2,
                    'Overgeneration': 1e-2,
                    'Shortfall': 1e-2}
    tolerances['runtimes'] = {'Solve Time': 1e1}

    with open('tolerances.json', 'w') as f:
        json.dump(tolerances, f, indent=2)


class TestSimulatorWithRtsGmlc(unittest.TestCase):
    """Test class for downloading RTS-GMLC test case, running the populator, and running the simulator."""
    # arbitrary comparison threshold
    COMPARISON_THRESHOLD = .01

    def setUp(self):
        # Download RTS GMLC test data and process.
        self.this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.rts_download_path = os.path.realpath(os.path.join(self.this_file_path, os.path.normcase('../../downloads/rts_gmlc')))

        rts_gmlc.download()
        rts_gmlc.populate_input_data()
        print('Set up RTS-GMLC data in {0}'.format(self.rts_download_path))

        self._run_populator()
        self._run_simulator()

        test_results_dir = os.path.join(self.rts_download_path, 'deterministic_with_network_simulation_output')
        control_results_dir = os.path.join(self.this_file_path, '..', 'simulator_tests',
                                           'deterministic_with_network_simulation_output')

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

    def _run_populator(self):
        """Runs the populator for the test data set."""
        os.chdir(self.rts_download_path)

        populator_config_filename = 'populate_with_network_deterministic.txt'
        script, options = runner.parse_commands(populator_config_filename)

        if sys.platform.startswith('win'):
            subprocess.call([script] + options, shell=True)
        else:
            subprocess.call([script] + options)
    
    def _run_simulator(self):
        """Runs the simulator for the test data set."""
        os.chdir(self.rts_download_path)

        simulator_config_filename = 'simulate_with_network_deterministic.txt'
        script, options = runner.parse_commands(simulator_config_filename)

        if sys.platform.startswith('win'):
            subprocess.call([script] + options, shell=True)
        else:
            subprocess.call([script] + options)
    
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


if __name__ == '__main__':
    unittest.main()
