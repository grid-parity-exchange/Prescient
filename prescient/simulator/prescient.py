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

import traceback
import sys
import os
import pyutilib
import profile
from . import master_options as MasterOptions
from .options import Options
from .simulator import Simulator
from .data_manager import DataManager
from .time_manager import TimeManager
from .oracle_manager import OracleManager
from .stats_manager import StatsManager
from .reporting_manager import ReportingManager
from prescient.stats.overall_stats import OverallStats
import prescient.plugins 

from prescient.engine.egret import EgretEngine as Engine

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

def create_prescient(options: Options):
    engine = Engine()
    time_manager = TimeManager()
    data_manager = DataManager()
    oracle_manager = OracleManager()
    stats_manager = StatsManager()
    reporting_manager = ReportingManager()
    prescient = Simulator(engine, time_manager, data_manager, oracle_manager, stats_manager, reporting_manager)
    return prescient


def main_prescient(options: Options):
    ans = None

    if pstats_available and options.profile > 0:
        #
        # Call the main ef writer with profiling.
        #
        tfile = pyutilib.services.TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('simulate(options)', globals(), locals(), tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('cumulative', 'calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        pyutilib.services.TempfileManager.clear_tempfiles()
        ans = [tmp, None]

    else:
        if options.traceback is True:
            simulator = create_prescient(options)
            ans = simulator.simulate(options)
        else:
            errmsg = None
            try:
                simulator = create_prescient(options)
                ans = simulator.simulate(options)
            except ValueError:
                err = sys.exc_info()[1]
                errmsg = 'VALUE ERROR: %s' % err
            except KeyError:
                err = sys.exc_info()[1]
                errmsg = 'KEY ERROR: %s' % err
            except TypeError:
                err = sys.exc_info()[1]
                errmsg = 'TYPE ERROR: %s' % err
            except NameError:
                err = sys.exc_info()[1]
                errmsg = 'NAME ERROR: %s' % err
            except IOError:
                err = sys.exc_info()[1]
                errmsg = 'I/O ERROR: %s' % err
            #Can't find ConverterError? Commenting out for now.
            #except ConverterError:
                #err = sys.exc_info()[1]
                #errmsg = 'CONVERSION ERROR: %s' % err
            except RuntimeError:
                err = sys.exc_info()[1]
                errmsg = 'RUN-TIME ERROR: %s' % err
            #pyutilib no attribute named common? Commenting out for now.
            #except pyutilib.common.ApplicationError:
                #err = sys.exc_info()[1]
                #errmsg = 'APPLICATION ERROR: %s' % err
            except Exception:
                err = sys.exc_info()[1]
                errmsg = 'UNKNOWN ERROR: %s' % err
                traceback.print_exc()

            if errmsg is not None:
                sys.stderr.write(errmsg + '\n')

    return ans

def main(args=None):
    if args is None:
        args = sys.argv

    #
    # Parse command-line options.
    #
    try:
        options_parser, guiOverride = MasterOptions.construct_options_parser()
        (options, args) = options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    main_prescient(options)

# MAIN ROUTINE STARTS NOW #
if __name__ == '__main__':
    result = main(sys.argv)
