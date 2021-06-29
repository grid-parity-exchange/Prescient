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

from .config import parse_args, PrescientConfig
from .options import Options
from .simulator import Simulator
from .data_manager import DataManager
from .time_manager import TimeManager
from .oracle_manager import OracleManager
from .stats_manager import StatsManager
from .reporting_manager import ReportingManager
from prescient.scripts import runner
from prescient.stats.overall_stats import OverallStats
from prescient.engine.egret import EgretEngine as Engine

from pyutilib.misc import import_file

class Prescient(Simulator):

    CONFIG = PrescientConfig

    def __init__(self):

        engine = Engine()
        time_manager = TimeManager()
        data_manager = DataManager()
        oracle_manager = OracleManager()
        stats_manager = StatsManager()
        reporting_manager = ReportingManager()

        self.simulate_called = False

        super().__init__(engine, time_manager, data_manager, oracle_manager, stats_manager, reporting_manager)

    def simulate(self, **options):
        # coming from Python
        # For safety, in case we re-run Prescient
        # again in the same Python script
        # (For the same reason, we don't accept options
        #  in __init__ above.)
        prescient.plugins.internal.clear_plugins()
        prescient.simulator.config.clear_prescient_config()

        if 'config_file' in options:
            config_file = options.pop('config_file')
            if options:
                raise RuntimeError(f"If using a config_file, all options must be specified in the configuration file")
            script, config_options = runner.parse_commands(config_file)
            if script != 'simulator.py':
                raise RuntimeError(f"config_file must be a simulator configuration text file, got {script}")
            options = parse_args(args=config_options)

        elif 'plugin' in options:
            # parse using the Config
            plugin_options = self.CONFIG({ 'plugin':options['plugin'] })
            for plugin in plugin_options.plugin:
                # importing the plugin will update
                # both the global config and the
                # global plugin registration
                import_file(plugin)
            # to reload after its changed by a plugin
            from .config import PrescientConfig
            options = PrescientConfig(options)
        else:
            options = self.CONFIG(options)

        return self._simulate(options)

    def _simulate(self, options):
        if self.simulate_called:
            raise RuntimeError(f"Each instance of Prescient should only be used once. "
                                "If you wish to simulate again create a new Prescient object.")
        self.simulate_called = True
        return super().simulate(options)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # For safety, in case we re-run Prescient
    # again in the same Python script
    prescient.plugins.internal.clear_plugins()
    prescient.simulator.config.clear_prescient_config()

    #
    # Parse command-line options.
    #
    try:
        options = parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    return Prescient()._simulate(options)

# MAIN ROUTINE STARTS NOW #
if __name__ == '__main__':
    result = main(sys.argv)
