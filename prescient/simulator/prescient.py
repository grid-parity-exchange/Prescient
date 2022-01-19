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
                         stats_manager, reporting_manager)

    def simulate(self, **options):
        ''' Execute a Prescient simulation.

        Arguments
        ---------
        config_file: str
            Path to a text file holding configuration options. If specified,
            this must be the only argument passed to the function.

        All other arguments:
            Any configuration properties accepted by PrescientConfig can be
            specified.

        Any options passed as arguments are applied to this instance's
        configuration, and then a simulation is run.
        
        Arguments can either be passed individually, or in a configuration
        file. You can't mix and match how configuration options are passed;
        either use a configuration file (config_file='path_to_config.txt')
        or pass configuration options as function arguments.
        '''
        if 'config_file' in options:
            config_file = options.pop('config_file')
            if options:
                raise RuntimeError(f"If using a config_file, all options must be specified in the configuration file")
            self.config.parse_args(args=['--config-file', config_file])

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
    result = main(sys.argv[1:])
