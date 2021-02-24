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
    from typing import Iterable
    from . import SimulationState
    from prescient.engine.abstract_types import EgretModel

from typing import NamedTuple
import math

from prescient.engine.abstract_types import G, S

GeneratorState = NamedTuple('GeneratorState', [('generator',G), 
                                               ('status',int), 
                                               ('power_generated',float)])
StorageSoc = NamedTuple('StorageSoc', [('storage',S),
                                       ('state_of_charge', float)])

def get_generator_states_at_sced_offset(original_state:SimulationState,
                                        sced:EgretModel, sced_index:int) -> Iterable[GeneratorState]:
    # We'll be converting between time periods and hours.
    # Make the data type of hours_per_period an int if it's an integer number of hours, float if fractional
    minutes_per_period = sced.data['system']['time_period_length_minutes']
    hours_per_period = minutes_per_period // 60 if minutes_per_period % 60 == 0 \
                       else minutes_per_period / 60

    for g, g_dict in sced.elements('generator', generator_type='thermal'):
        ### Get generator state (whether on or off, and for how long) ###
        init_state = original_state.get_initial_generator_state(g)
        # state is in hours, convert to integer number of time periods
        init_state = round(init_state / hours_per_period)
        state_duration = abs(init_state)
        unit_on = init_state > 0
        g_commit = g_dict['commitment']['values']

        # March forward, counting how long the state is on or off
        for t in range(0, sced_index+1):
            new_on = (int(round(g_commit[t])) > 0)
            if new_on == unit_on:
                state_duration += 1
            else:
                state_duration = 1
                unit_on = new_on

        if not unit_on:
            state_duration = -state_duration

        # Convert duration back into hours
        state_duration *= hours_per_period

        ### Get how much power was generated, within bounds ###
        power_generated = g_dict['pg']['values'][sced_index]

        # the validators are rather picky, in that tolerances are not acceptable.
        # given that the average power generated comes from an optimization 
        # problem solve, the average power generated can wind up being less
        # than or greater than the bounds by a small epsilon. touch-up in this
        # case.
        if isinstance(g_dict['p_min'], dict):
            min_power_output = g_dict['p_min']['values'][sced_index]
        else:
            min_power_output = g_dict['p_min']
        if isinstance(g_dict['p_max'], dict):
            max_power_output = g_dict['p_max']['values'][sced_index]
        else:
            max_power_output = g_dict['p_max']
                
        # TBD: Eventually make the 1e-5 an user-settable option.
        if unit_on == 0:
            # if the unit is off, then the power generated at t0 must be equal to 0
            power_generated = 0.0
        elif math.isclose(min_power_output, power_generated, rel_tol=0, abs_tol=1e-5):
            power_generated = min_power_output
        elif math.isclose(max_power_output, power_generated, rel_tol=0, abs_tol=1e-5):
            power_generated = max_power_output

        ### Yield the results for this generator ###
        yield GeneratorState(g,state_duration, power_generated)

def get_storage_socs_at_sced_offset(sced:EgretModel, sced_index:int) -> Iterable[StorageSoc]:
    for s,s_dict in sced.elements('storage'):
        yield StorageSoc(s, s_dict['state_of_charge']['values'][sced_index])
