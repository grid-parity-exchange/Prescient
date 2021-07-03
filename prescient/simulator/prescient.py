#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

####################################################
#                   prescient                      #
####################################################

import sys

import prescient.plugins.internal
import prescient.simulator.config

from .config import PrescientConfig
from .options import Options
from .simulator import Simulator
from .data_manager import DataManager
from .time_manager import TimeManager
from .oracle_manager import OracleManager
from .stats_manager import StatsManager
from .reporting_manager import ReportingManager
from prescient.scripts import runner
from prescient.engine.egret import EgretEngine as Engine

from pyutilib.misc import import_file

class Prescient(Simulator):

    def __init__(self):

        self.config = PrescientConfig()

        engine = Engine()
        time_manager = TimeManager()
        data_manager = DataManager()
        oracle_manager = OracleManager()
        stats_manager = StatsManager()
        reporting_manager = ReportingManager()

        self.simulate_called = False

        super().__init__(engine, time_manager, data_manager, oracle_manager, 
                         stats_manager, reporting_manager, 
                         self.config.plugin_context.callback_manager)

    def simulate(self, **options):
        if 'config_file' in options:
            config_file = options.pop('config_file')
            if options:
                raise RuntimeError(f"If using a config_file, all options must be specified in the configuration file")
            script, config_options = runner.parse_commands(config_file)
            if script != 'simulator.py':
                raise RuntimeError(f"config_file must be a simulator configuration text file, got {script}")
            self.config.parse_args(args=config_options)

        else:
            self.config.set_value(options)

        return self._simulate(self.config)

    def _simulate(self, options: PrescientConfig):
        if self.simulate_called:
            raise RuntimeError(f"Each instance of Prescient should only be used once. "
                                "If you wish to simulate again create a new Prescient object.")
        self.simulate_called = True
        return super().simulate(options)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    try:
        p = Prescient()
        p.config.parse_args(args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    return p.simulate()

# MAIN ROUTINE STARTS NOW #
if __name__ == '__main__':
    result = main(sys.argv)
