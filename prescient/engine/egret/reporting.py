#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import math
from prescient.util.math_utils import round_small_values

def report_initial_conditions_for_deterministic_ruc(deterministic_instance):
    tgens = dict(deterministic_instance.elements('generator', generator_type='thermal'))
    if not tgens:
        max_thermal_generator_label_length = None
    else:
        max_thermal_generator_label_length = max((len(g) for g in tgens))

    print("")
    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated):")
    for g,gdict in tgens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s %5d %7d %12.2f" ) % 
              (g, 
               int(gdict['initial_status']>0),
               gdict['initial_status'],
               gdict['initial_p_output'],
               ))

    # it is generally useful to know something about the bounds on output capacity
    # of the thermal fleet from the initial condition to the first time period. 
    # one obvious use of this information is to aid analysis of infeasibile
    # initial conditions, which occur rather frequently when hand-constructing
    # instances.

    # output the total amount of power generated at T0
    total_t0_power_output = 0.0
    for g,gdict in tgens.items():
        total_t0_power_output += gdict['initial_p_output']
    print("")
    print("Power generated at T0=%8.2f" % total_t0_power_output)
    
    # compute the amount of new generation that can be brought on-line the first period.
    total_new_online_capacity = 0.0
    for g,gdict in tgens.items():
        t0_state = gdict['initial_status']
        if t0_state < 0: # the unit has been off
            if int(math.fabs(t0_state)) >= gdict['min_down_time']:
                if isinstance(gdict['p_max'], dict):
                    p_max = gdict['p_max']['values'][0]
                else:
                    p_max = gdict['p_max']
                total_new_online_capacity += min(gdict['startup_capacity'], p_max)
    print("")
    print("Total capacity at T=1 available to add from newly started units=%8.2f" % total_new_online_capacity)

    # compute the amount of generation that can be brough off-line in the first period
    # (to a shut-down state)
    total_new_offline_capacity = 0.0
    for g,gdict in tgens.items():
        t0_state = gdict['initial_status']
        if t0_state > 0: # the unit has been on
            if t0_state >= gdict['min_up_time']:
                if gdict['initial_p_output'] <= gdict['shutdown_capacity']:
                    total_new_offline_capacity += gdict['initial_p_output']
    print("")
    print("Total capacity at T=1 available to drop from newly shut-down units=%8.2f" % total_new_offline_capacity)

def report_demand_for_deterministic_ruc(ruc_instance, ruc_every_hours):
    load = ruc_instance.data['elements']['load']
    times = ruc_instance.data['system']['time_keys']
    max_bus_label_length = max(len(b) for b in load)
    print("")
    print("Projected Demand:")
    for b, ldict in load.items():
        print(("%-"+str(max_bus_label_length)+"s: ") % b, end=' ')
        for i in range(min(len(times), 36)):
            print("%8.2f"% ldict['p_load']['values'][i], end=' ')
            if i+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")

def report_fixed_costs_for_deterministic_ruc(ruc):
    costs = sum( sum(g_dict['commitment_cost']['values']) 
                 for _,g_dict in ruc.elements('generator', generator_type='thermal'))
    print("Fixed costs:    %12.2f" % costs)

def report_generation_costs_for_deterministic_ruc(ruc):
    costs = sum( sum(g_dict['production_cost']['values']) 
                 for _,g_dict in ruc.elements('generator', generator_type='thermal'))
    print("Variable costs: %12.2f" % costs)

def report_load_generation_mismatch_for_deterministic_ruc(ruc_instance):
    time_periods = ruc_instance.data['system']['time_keys']
    buses = ruc_instance.data['elements']['bus']

    for i,t in enumerate(time_periods):
        mismatch_reported = False
        sum_mismatch = round_small_values(sum(bdict['p_balance_violation']['values'][i]
                                              for bdict in buses.values()))
        if sum_mismatch != 0.0:
            posLoadGenerateMismatch = round_small_values(sum(max(bdict['p_balance_violation']['values'][i],0.)
                                                            for bdict in buses.values()))
            negLoadGenerateMismatch = round_small_values(sum(min(bdict['p_balance_violation']['values'][i],0.)
                                                            for bdict in buses.values()))
            if negLoadGenerateMismatch != 0.0:
                print("Projected over-generation reported at t=%s -   total=%12.2f" % (t, negLoadGenerateMismatch))
            if posLoadGenerateMismatch != 0.0:
                print("Projected load shedding reported at t=%s -     total=%12.2f" % (t, posLoadGenerateMismatch))

        if 'reserve_shortfall' in ruc_instance.data['system']:
            reserve_shortfall_value = round_small_values(ruc_instance.data['system']['reserve_shortfall']['values'][i])
            if reserve_shortfall_value != 0.0:
                print("Projected reserve shortfall reported at t=%s - total=%12.2f" % (t, reserve_shortfall_value))

def report_curtailment_for_deterministic_ruc(ruc):
    rn_gens = dict(ruc.elements('generator', generator_type='renewable'))
    time_periods = ruc.data['system']['time_keys']

    curtailment_in_some_period = False
    for i,t in enumerate(time_periods):
        quantity_curtailed_this_period = sum(gdict['p_max']['values'][i] - gdict['pg']['values'][i] \
                                            for gdict in rn_gens.values())
        if quantity_curtailed_this_period > 0.0:
            if curtailment_in_some_period == False:
                print("Renewables curtailment summary (time-period, aggregate_quantity):")
                curtailment_in_some_period = True
            print("%s %12.2f" % (t, quantity_curtailed_this_period))

def output_solution_for_deterministic_ruc(ruc_instance, 
                                          ruc_every_hours):
    thermal_gens = dict(ruc_instance.elements('generator', generator_type='thermal'))
    if len(thermal_gens) == 0:
        max_thermal_generator_label_length = None
    else:
        max_thermal_generator_label_length = max((len(this_generator) 
                                                  for this_generator in thermal_gens))

    last_time_period = min(len(ruc_instance.data['system']['time_keys']), 36)
    last_time_period_storage = min(len(ruc_instance.data['system']['time_keys']), 26)

    print("Generator Commitments:")
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            print("%2d"% int(round(gdict['commitment']['values'][t])), end=' ')
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Dispatch Levels:")
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            print("%7.2f"% gdict['pg']['values'][t], end=' ')
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")

    print("")
    print("Generator Reserve Headroom:")
    total_headroom = [0.0 for i in range(0, last_time_period)]  # add the 0 in for simplicity of indexing
    for g,gdict in thermal_gens.items():
        print(("%-"+str(max_thermal_generator_label_length)+"s: ") % g, end=' ')
        for t in range(last_time_period):
            headroom = gdict['headroom']['values'][t]
            print("%7.2f" % headroom, end=' ')
            total_headroom[t] += headroom
            if t+1 == ruc_every_hours: 
                print(" |", end=' ')
        print("")
    print(("%-"+str(max_thermal_generator_label_length)+"s: ") % "Total", end=' ')
    for t in range(last_time_period):
        print("%7.2f" % total_headroom[t], end=' ')
        if t+1 == ruc_every_hours: 
            print(" |", end=' ')
    print("")

    storage = dict(ruc_instance.elements(element_type='storage'))
    if len(storage) > 0:
        
        print("Storage Input levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['p_charge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")

        print("Storage Output levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['p_discharge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")

        print("Storage SOC levels")
        for s,sdict in storage.items():
            print("%30s: " % s, end=' ')
            for t in range(last_time_period_storage):
                print("%7.2f"% sdict['state_of_charge']['values'][t], end=' ')
                if t+1 == ruc_every_hours: 
                    print(" |", end=' ')
            print("")
