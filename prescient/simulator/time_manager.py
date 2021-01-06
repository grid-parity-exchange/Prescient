#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from datetime import date

import dateutil
import sys
from datetime import timedelta, datetime, time
from typing import Optional, Iterable

from .manager import _Manager
from . import manager
from . import options


class PrescientTime:
    '''
    A point in time during a simulation
    '''
    def __init__(self, when:datetime, is_planning_time:bool, is_ruc_activation_time:bool):
        self._when = when
        self._is_planning_time = is_planning_time
        self._is_ruc_activation_time = is_ruc_activation_time

    @property
    def date(self) -> date:
        return self._when.date()

    @property
    def time(self) -> datetime.time:
        return self._when.time()

    @property 
    def datetime(self) -> datetime.datetime:
        return self._when

    @property
    def hour(self) -> int:
        return self._when.hour

    @property
    def is_planning_time(self) -> bool:
        return self._is_planning_time

    @property
    def is_ruc_activation_time(self) -> bool:
        return self._is_ruc_activation_time

    def __str__(self):
        return self._when.isoformat(sep=' ', timespec='minutes')

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

        self._stop_date = self._start_date + timedelta(days=options.num_days)
        print(f"Dates to simulate: {str(self._start_date)} to {str(self._stop_date - timedelta(days=1))}")

        # Validate SCED frequency
        if 60 % options.sced_frequency_minutes != 0:
            raise RuntimeError(
                f"--sced-frequency-minutes must divide evenly into 60! {options.sced_frequency_minutes} supplied!" )
        self._sced_delta = timedelta(minutes=options.sced_frequency_minutes)

        # Validate RUC frequency
        if 24 % options.ruc_every_hours != 0:
            raise RuntimeError(
                f"--ruc-every-hours must be a divisor of 24! {options.ruc_every_hours} supplied!" )
        self._ruc_every_hours = options.ruc_every_hours
        print("RUC activation hours:", ", ".join(str(hr) for hr in range(0, 24, options.ruc_every_hours)))

        # validate the RUC horizon
        if not options.ruc_every_hours <= options.ruc_horizon <= 48:
            raise RuntimeError(
                "--ruc-horizon must be greater than or equal --ruc-every-hours and"
                " less than or equal to 48! {options.ruc_horizon} supplied!")

        # Compute RUC delay
        self._ruc_delay = -(options.ruc_execution_hour%-self._ruc_every_hours)

        self._run_ruc = options.disable_ruc == False

        if self._run_ruc:
            print("Final RUC date:", str(self._stop_date - timedelta(days=1)))

        self._current_time = None


    def time_steps(self) -> Iterable[PrescientTime]:
        '''a generator which yields a PrescientTime instance for each time of interest in the simulation'''

        # We start at time 0 of first day, stop before time 0 of stop day
        current_time = datetime.combine(self._start_date, time(0))
        stop_time = datetime.combine(self._stop_date, time(0))

        # minutes between each time step
        step_delta = self._sced_delta

        # Set up the first planning and activation times.
        # The first time step is not considered a planning time or activation time,
        # even if it aligns with a normal planning or activation cycle.  That's 
        # because up-front initialization includes first-time planning and activation.
        # The first activation time is ruc_every_hours after t0, and the first 
        # planning time is ruc_delay hours before that.
        ruc_delta = timedelta(hours=self._ruc_every_hours)
        next_activation_time = current_time + ruc_delta
        next_planning_time = next_activation_time - timedelta(hours=self._ruc_delay)

        while current_time < stop_time:
            is_planning_time = current_time == next_planning_time
            if is_planning_time:
                next_planning_time += ruc_delta
                # If the next plan won't be activated until after the simulation has finished,
                # don't bother generating the plan. Push the next planning time out past the end.
                if next_planning_time + timedelta(hours=self._ruc_delay) >= stop_time:
                    next_planning_time = stop_time

            is_activation_time = current_time == next_activation_time
            if is_activation_time:
                next_activation_time += ruc_delta

            t = PrescientTime(current_time, is_planning_time, is_activation_time)
            self._current_time = t
            yield t

            current_time += step_delta


    @property
    def current_time(self):
        return self._current_time

    def get_first_time_step(self) -> PrescientTime:
        t0 = datetime.combine(self._start_date, time(0))
        return PrescientTime(t0, False, False)


