#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import datetime
import dateutil.parser
from typing import Callable

from .options import Options
from .manager import _Manager
from .time_manager import PrescientTime
from prescient.stats import OperationsStats, HourlyStats, DailyStats, OverallStats
from prescient.util.publish_subscribe import Dispatcher


class StatsManager(_Manager):
    '''Manages statistics gathering and notification.

       Statistics are gathered at each time step.  This is done by calling
       this class's methods in a specific order:
       * initialize() - Called once, before the simulation begins
           * begin_timestep() - Called at the beginning of each time step, before the
                                simulation carries out any of the time step's activites
           * collect_operations() - Called once per timestep after the operations sced has been run
           * end_timestep() = Called once per timestep, after the simulation has carried out
                              all of the time step's activities
       * end_simulation() - Called once at the end of the simulation

       Statistics are aggregated by hour, by day, and overall.  Objects can register to be notified
       when a new set of statistics at any of those three groupings becomes available.  Notification
       is done at the end of the timestep that completes each hour, day, or the overall simulation.
    '''

    # This class makes the following assumptions:
    #  * all simulations start at hour 0 of the first day
    #  * the time between time steps is constant
    #  * time steps are never more than 1 hour

    def __init__(self):
        self._current_sced_stats = None
        self._current_hour_stats = None
        self._current_day_stats = None
        self._overall_stats = None

        self._sced_publisher = Dispatcher()
        self._hourly_publisher = Dispatcher()
        self._daily_publisher = Dispatcher()
        self._overall_publisher = Dispatcher()
        self._subscribers = []

    def initialize(self, options: Options) -> None:
        '''Called before a simulation begins'''
        self._options = options

        self._step_interval = datetime.timedelta(minutes=options.sced_frequency_minutes)

        # Clear out any previous stats, if present
        self._overall_stats = None
        self._current_day_stats = None
        self._current_hour_stats = None
        self._current_sced_stats = None

    def begin_timestep(self, time_step: PrescientTime) -> None:
        '''Called just before the simulation advances to the indicated time'''

        ##### Overview of statistics management: #####
        #
        # begin_timestep() is responsible for ensuring there is a stats object for the
        # current hour, day, and overall simulation.
        #
        # collect_operations() is responsible for incorporating the most recent sced results
        # into the current hour's stats.
        #
        # end_timestep() is responsible for detecting when an hour or day is done, rolling up
        # stats accordingly, and notifying stats observers that the time period has been completed.
        #
        # end_simulation() is responsible for notifying stats observers that the simulation is done.
        #
        ###### Details ###########
        #
        # _current_hour_stats, _current_day_stats, and _overall_stats are set to None until begin_timestep()
        # is called for the first time within a given hour/day/simulation.
        # When begin_timestep() marks the beginning of a new hour/day/simulation, a fresh set of
        # empty stats is created for the relevant time period.  
        # 
        # Hourly stats are populated when collect_operations() is called within a timestep.
        # 
        # When end_timestep() is called for the final time period within an hour, the hourly
        # stats are reported to the current day's daily stats and _current_hour_stats is set
        # to None.  If the completed time step also marks the end of a day, the current day's
        # stats are reported to the overall stats, and _current_day_stats is set to None.
        #
        # Each level is added to its "parent" as it is completed.  For example, 
        # hourly stats are added to the daily stats when the final time step within
        # the hour is finished.  After adding stats to the parent, it is then set
        # to None to indicate we will be starting a fresh set of stats for the 
        # relevant time interval.e

        # If we are just starting a simulation, create overall stats
        if self._overall_stats is None:
            self._overall_stats = OverallStats(self._options)

        date = time_step.date

        # If we are just starting a new day, create new daily stats 
        if self._current_day_stats is None:
            self._current_day_stats = DailyStats(self._options, date)

        # If we are just starting a new hour, create new hourly stats
        if self._current_hour_stats is None:
            self._current_hour_stats = HourlyStats(self._options, date, time_step.hour)

        # if we are just starting a new sced (which we always are), create new sced stats
        if self._current_sced_stats is None:
            self._current_sced_stats = OperationsStats(self._options, time_step.datetime)


    def collect_operations(self, sced, runtime, lmp_sced, pre_quickstart_cache, extractor):
        '''Called when a new operations sced has been run
        
           Must be called within a timestep, i.e., after begin_timestep()
           and before the corresponding end_timestep().
        '''
        self._current_sced_stats.populate_from_sced(sced, runtime, lmp_sced, pre_quickstart_cache, extractor)
        return self._current_sced_stats

    def collect_market_settlement(self, sced, extractor, ruc_market, time_index):
        ''' Called after new operations and LMP sced has run and
            after collect_operations
        '''
        self._current_sced_stats.populate_market_settlement(sced, extractor, ruc_market, time_index)

    def collect_quickstart_data(self, pre_quickstart_cache, sced):
        self._current_hour_stats.update_with_quickstart_data(quickstart_cache, sced)
        return self._current_hour_stats

    def end_timestep(self, time_step: PrescientTime):
        '''Called after the simulation has completed the indicated time'''

        self._finish_timestep()

        this_time = time_step.datetime
        next_time = this_time + self._step_interval
        
        is_hour_end = next_time.hour != this_time.hour
        is_day_end = next_time.day != this_time.day

        if is_day_end:
            # this will finish the hour and then the day, ensuring everything is 
            # rolled up before stats observers are notified
            self._finish_day()
        elif is_hour_end:
            self._finish_hour()

    def end_simulation(self):
        self._finish_simulation()

    def register_for_sced_stats(self, callback: Callable[[OperationsStats], None], keep_alive=True):
        self._sced_publisher.subscribe(callback)
        if keep_alive:
            self._subscribers.append(callback)

    def register_for_hourly_stats(self, callback: Callable[[HourlyStats], None], keep_alive=True):
        self._hourly_publisher.subscribe(callback)
        if keep_alive:
            self._subscribers.append(callback)

    def register_for_daily_stats(self, callback: Callable[[DailyStats], None], keep_alive=True):
        self._daily_publisher.subscribe(callback)
        if keep_alive:
            self._subscribers.append(callback)

    def register_for_overall_stats(self, callback: Callable[[OverallStats], None], keep_alive=True):
        self._overall_publisher.subscribe(callback)
        if keep_alive:
            self._subscribers.append(callback)


    def _finish_timestep(self):
        if self._current_sced_stats is not None:
            self._current_hour_stats.incorporate_operations_stats(self._current_sced_stats)
            self._sced_publisher.publish(self._current_sced_stats)
            self._current_sced_stats = None

    def _finish_hour(self):
        ''' Close the latest batch of hourly statistics, and roll them into the current day's stats'''
        if self._current_hour_stats is not None:
            self._current_day_stats.incorporate_hour_stats(self._current_hour_stats)
            self._hourly_publisher.publish(self._current_hour_stats)
            self._current_hour_stats = None

    def _finish_day(self):
        ''' Close the latest batch of daily statistics, and roll them into overall stats.

            The current hour is also closed, if not already done
        '''
        if self._current_day_stats is not None:
            self._finish_hour()
            self._overall_stats.incorporate_day_stats(self._current_day_stats)
            self._daily_publisher.publish(self._current_day_stats)
            self._current_day_stats = None

    def _finish_simulation(self):
        '''Close overall statistics.

           The current day and hour are also closed, if not already done
        '''
        if self._overall_stats is not None:
            self._finish_day()
            self._overall_publisher.publish(self._overall_stats)
