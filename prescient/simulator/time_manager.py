#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from .manager import _Manager
from __future__ import annotations
import dateutil
import sys
import datetime
from datetime import timedelta
import time
from typing import Optional, Iterable

#from .manager import _Manager
#from .options import Options
from . import manager
from . import options


class PrescientTime:
    '''
    A point in time during a simulation, with one hour granularity
    '''
    def __init__(self, date:str, next_date:str, next_next_date:str, hour:int, is_planning_time:bool, is_ruc_start_hour:bool):
        self._date = date
        self._next_date = next_date
        self._next_next_date = next_next_date
        self._hour = hour
        self._is_planning_time = is_planning_time
        self._is_ruc_start_hour = is_ruc_start_hour


    @property
    def date(self) -> str:
        return self._date

    @property
    def next_date(self) -> Optional[str]:
        return self._next_date

    @property
    def next_next_date(self) -> Optional[str]:
        return self._next_next_date

    @property
    def hour(self) -> int:
        return self._hour

    @property
    def is_planning_time(self) -> bool:
        return self._is_planning_time

    @property
    def is_ruc_start_hour(self) -> bool:
        return self._is_ruc_start_hour

class TimeManager(manager._Manager):
    '''
    Provides information about the times included in a simulation, including all times of interest during a simulation
    '''
    def initialize(self, options: Options) -> None:
        # validate the start date
        try:
            self._start_date = dateutil.parser.parse(options.start_date).date()
        except ValueError:
            print("***ERROR: Illegally formatted start date=" + options.start_date + " supplied!")
            sys.exit(1)

        dates_to_simulate = [str(self._start_date + timedelta(n)) for n in range(0,options.num_days)]

        self._dates_to_simulate = dates_to_simulate
        self._end_date = dates_to_simulate[-1]

        if 24 % options.ruc_every_hours != 0:
            raise RuntimeError("--ruc-every-hours must be a divisor of 24! %d supplied!" % options.ruc_every_hours)
        self._ruc_start_hours = list(range(0, 24, options.ruc_every_hours))
        print("ruc_start_hours:", self._ruc_start_hours)

        list_hours = dict((self._ruc_start_hours[i], list(range(self._ruc_start_hours[i],
                                                                self._ruc_start_hours[i + 1])))
                          for i in range(0, len(self._ruc_start_hours) - 1))
        list_hours[self._ruc_start_hours[-1]] = list(range(self._ruc_start_hours[-1], 24))

        self._list_hours = list_hours

        if not options.ruc_every_hours <= options.ruc_horizon <= 48:
            raise RuntimeError(
                "--ruc-horizon must be greater than or equal --ruc-every-hours and less than or equal to 48! %d supplied!" % options.ruc_horizon)

        print("Dates to simulate:", dates_to_simulate)

        self._run_ruc = options.disable_ruc == False

        if self._run_ruc:
            last_ruc_date = dates_to_simulate[-1]
            print("")
            print("Last RUC date:", last_ruc_date)

        # ruc_execution_hour = one of the times of day that the ruc will run each day
        # if negative, the number of hours before midnight
        self._ruc_execution_hour = options.ruc_execution_hour

        # ruc_every_hours = number of hours between each ruc planning run
        self._ruc_every_hours = options.ruc_every_hours



    def time_steps(self) -> Iterable[PrescientTime]:
        '''a generator which yields a PrescientTime instance for each time of interest in the simulation'''
        next_dates = self._dates_to_simulate[1:] + [None]
        next_next_dates = next_dates[1:] + [None]
        for date, next_date, next_next_date in zip(self._dates_to_simulate, next_dates, next_next_dates):
            for ruc_hour in self._ruc_start_hours:
                for h in self._list_hours[ruc_hour]:
                    planning_time = False
                    if self._ruc_execution_hour % self._ruc_every_hours > 0:
                        uc_hour = (h - self._ruc_execution_hour % (- self._ruc_every_hours)) % 24
                    else:
                        uc_hour = h

                    if date != self._end_date:
                        end_of_ruc = False
                    elif uc_hour > 0:
                        end_of_ruc = False
                    elif self._ruc_execution_hour == 0:
                        end_of_ruc = False
                    else:
                        end_of_ruc = True

                    is_ruc_start_hour = (h in self._ruc_start_hours)

                    if (h % self._ruc_every_hours == 0) and (self._ruc_execution_hour == 0):
                        ## in this case, we need a SCED first because we ran a RUC before
                        if (self._start_date == date) and (h == 0):
                            is_ruc_hour = False
                        else:
                            is_ruc_hour = True
                    elif (h % self._ruc_every_hours == self._ruc_execution_hour % self._ruc_every_hours):
                        is_ruc_hour = True
                    else:
                        is_ruc_hour = False

                    # run RUC at D-X (except on the last day of the simulation), where X is the
                    # user-specified hour at which RUC is to be executed each day.
                    if self._run_ruc \
                            and (not end_of_ruc) \
                            and is_ruc_hour:

                        # print("DEBUG: Running RUC")
                        planning_time = True

                    time = PrescientTime(date, next_date, next_next_date, h, planning_time, is_ruc_start_hour)
                    yield time

    def get_first_time_step(self) -> PrescientTime:
        first_date = self._get_first_date()
        second_date = self._get_second_date()
        third_date = self._get_third_date()
        first_hour = 0
        planning_time = True
        is_ruc_start_hour = False
        return PrescientTime(first_date, second_date, third_date, first_hour, planning_time, is_ruc_start_hour)

    def is_first_time_step(self, time: PrescientTime) -> bool:
        return time.hour == 0 and time.date == self._get_first_date()

    def _get_first_date(self) -> str:
        return self._dates_to_simulate[0]

    def _get_second_date(self) -> Optional[str]:
        return self._dates_to_simulate[1] if len(self._dates_to_simulate) > 1 else None

    def _get_third_date(self) -> Optional[str]:
        return self._dates_to_simulate[2] if len(self._dates_to_simulate) > 2 else None


