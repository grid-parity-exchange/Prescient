####################################################
#                   prescient                      #
####################################################
# # Aug 2015: HERE LIES THE COMBINATION OF CONFCOMP.PY, POPULATE_INPUT_DIRECTORIES.PY AND PRESCIENT_SIM.PY.
# # IT INCLUDES ALL FUNCTIONALITY. ALL OPTIONS
# # AND ALL BASH FILES USED FOR THE PREVIOUS ARE USABLE ON THIS WITH SMALL MODIFICATIONS.
# # 1) CHANGE THE PY FILE TO PRESCIENT.PY
# # 2) ADD THE OPTION --run-{x} where x = {populator,simulator,scenarios}
#
# # Note that bash files are outdated as of Sept 2016 and instead txt files paired with the runner.py.
# import random
import sys
import os
import shutil
import random
import traceback
import csv
import time
import datetime
import math
from optparse import OptionParser, OptionGroup

try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

try:
    import cProfile as profile
except ImportError:
    import profile

import matplotlib
# the following forces matplotlib to not use any Xwindows backend.
# taken from stackoverflow, of course.
matplotlib.use('Agg')

import numpy as np
import pandas as pd

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

try:
    import dateutil
except:
    print("***Failed to import python dateutil module - try: easy_install python-dateutil")
    sys.exit(1)

from six import iterkeys, itervalues, iteritems
from pyomo.core import *
from pyomo.opt import *
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
import pyutilib

import prescient.sim.MasterOptions as MasterOptions

# import plotting capabilities
import prescient.sim.graphutils as graphutils
import prescient.sim.storagegraphutils as storagegraphutils


# hacking in callbacks
from prescient.plugins.callbacks import RT_sced_callback, DA_ruc_callback, DA_market_callback

# random-but-useful global constants.
DEFAULT_MAX_LABEL_LENGTH = 15

#############################################################################################
###############################       START PRESCIENT        ################################
#############################################################################################

###############################
# Helper function definitions #
###############################

# the simulator relies on the presence of various methods in the reference
# model module - verify that these exist.

def validate_reference_model(module):
    required_methods = ["fix_binary_variables", "free_binary_variables", "status_var_generator", "define_suffixes", "load_model_parameters"]
    for method in required_methods:
        if not hasattr(module, method):
            raise RuntimeError("Reference model module does not have required method=%s" % method)

def call_solver(solver,instance,**kwargs):
    # # Needed to allow for persistent solver options July2015
    return solver.solve(instance, load_solutions=False, **kwargs)

def round_small_values(x, p=1e-6):
    # Rounds values that are within (-1e-6, 1e-6) to 0.
    try:
        if math.fabs(x) < p:
            return 0
        return x
    except:
        raise RuntimeError("Utility function round_small_values failed on input=%s, p=%f" % (str(x), p))

def daterange(start_day_date, end_day_date):
    for n in range(int ((end_day_date - start_day_date).days)+1):
        yield start_day_date + datetime.timedelta(n)


# NOTE: Unused
def take_generator_offline(model, instance, generator):
    if generator not in offline_generators:
        offline_generators.append(generator)
        for t in instance.TimePeriods:
            instance.UnitOn[generator,t] = 0
            instance.UnitOn[generator,t].fixed = True
            instance.PowerGenerated[generator,t] = 0
            instance.PowerGenerated[generator,t].fixed = True
            instance.UnitOnT0[generator] = 0
            instance.PowerGeneratedT0[generator] = 0
            if model == "multiSUSD":
                instance.UnitOff[generator, t] = 1
                instance.UnitOff[generator, t].fixed = True
                instance.UnitOffT0[generator] = 1
            else:
                instance.MaximumPowerAvailable[generator,t] = 0
                instance.MaximumPowerAvailable[generator,t].fixed = True


def reset_offline_elements(model, instance):
    for g in offline_generators:
        for t in instance.TimePeriods:
            #print "Resetting UnitOn and PowerGenerated for",g,t
            instance.UnitOn[g,t].fixed = False
            instance.PowerGenerated[g,t].fixed = False
            if model == "multiSUSD":
                instance.UnitOff[g, t].fixed = False
            else:
                instance.MaximumPowerAvailable[g,t].fixed = False

########################################################
# utility functions for reporting various aspects of a #
# multi-period SCED solution.                          #
########################################################

def report_costs_for_deterministic_sced(instance):

    # only worry about two-stage models for now..
    fixed_cost = value(sum(instance.StartupCost[g,1] + instance.ShutdownCost[g,1] for g in instance.ThermalGenerators) + \
                       sum(instance.UnitOn[g,1] * instance.MinimumProductionCost[g,1] * instance.TimePeriodLength for g in instance.ThermalGenerators))
    variable_cost = value(instance.TotalProductionCost[1])
    print("Fixed costs:    %12.2f" % fixed_cost)
    print("Variable costs: %12.2f" % variable_cost)
    return fixed_cost, variable_cost

def report_mismatches_for_deterministic_sced(instance):

    issues_found = False

    load_generation_mismatch_value = round_small_values(sum(value(instance.LoadGenerateMismatch[b, 1])
                                                            for b in instance.Buses))
    if load_generation_mismatch_value != 0.0:
        issues_found = True
        load_shedding_value = round_small_values(sum(value(instance.posLoadGenerateMismatch[b, 1])
                                                     for b in instance.Buses))
        over_generation_value = round_small_values(sum(value(instance.negLoadGenerateMismatch[b, 1])
                                                       for b in instance.Buses))
        if load_shedding_value != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f" % (1, load_shedding_value))
        if over_generation_value != 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f" % (1, over_generation_value))
    else:
        load_shedding_value = 0.0
        over_generation_value = 0.0

    available_quick_start = available_quick_start_for_deterministic_sced(instance)

    reserve_shortfall_value = round_small_values(value(instance.ReserveShortfall[1]))
    if reserve_shortfall_value != 0.0:
        issues_found = True
        print("Reserve shortfall reported at t=%2d: %12.2f" % (1, reserve_shortfall_value))
        # report if quick start generation is available during reserve shortfalls in SCED
        print("Quick start generation capacity available at t=%2d: %12.2f" % (1, available_quick_start))
        print("")

    available_reserve = sum(value(instance.MaximumPowerAvailable[g, 1]) - value(instance.PowerGenerated[g, 1])
                            for g in instance.ThermalGenerators)

    if issues_found:
        pass
#        print "***ISSUES FOUND***"
#        instance.ReserveRequirement.pprint()
#        instance.LoadGenerateMismatch.pprint()
#        instance.ReserveMismatch.pprint()
#        instance.MaximumPowerAvailable.pprint()
#        instance.PowerGenerated.pprint()

    # order of return values is: load-shedding, over-generation, reserve-shortfall, available_reserve
    return load_shedding_value, over_generation_value, reserve_shortfall_value, available_reserve, available_quick_start

def report_prices_for_deterministic_ruc(day_ahead_prices, day_ahead_reserve_prices, instance, options,
                                         max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    pricing_type = options.day_ahead_pricing
    print("")
    print(("%-"+str(max_bus_label_length)+"s %5s %14s") % ("Bus","Time", "Computed "+pricing_type))
    for t in range(0,options.ruc_every_hours):
        for bus in instance.Buses:
            print(("%-"+str(max_bus_label_length)+"s %5d %14.6f") % (bus, t, day_ahead_prices[bus,t]))

    print("")
    print(("Reserves %5s %14s") % ("Time", "Computed "+pricing_type+" reserve price"))
    for t in range(0,options.ruc_every_hours):
        print(("         %5d %14.6f") % (t, day_ahead_reserve_prices[t]))


def report_lmps_for_deterministic_sced(instance,
                                       max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print(("%-"+str(max_bus_label_length)+"s %14s") % ("Bus", "Computed LMP"))
    for bus in instance.Buses:
        for t in range(1,value(instance.NumTimePeriods)+1):
            print(("%-"+str(max_bus_label_length)+"s %14.6f") % (bus, instance.dual[instance.PowerBalance[bus,t]]))

def get_lmps_for_deterministic_sced(instance,
                                    max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")
    print(("%-"+str(max_bus_label_length)+"s %14s") % ("Bus", "Computed LMP"))
    for bus in instance.Buses:
        for h in range(1,value(instance.NumTimePeriods)+1):
            print('Hour = {}'.format(h))
            print(("%-"+str(max_bus_label_length)+"s %14.6f") % (bus, instance.dual[instance.PowerBalance[bus,h]]))

def report_at_limit_lines_for_deterministic_sced(instance,
                                                 max_line_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("")

    lines_at_limit = []

    for l in sorted(instance.TransmissionLines):
        if value(instance.LinePower[l,1]) < 0.0:
            if instance.ThermalLimit[l] - -1.0 * value(instance.LinePower[l,1]) <= 1e-5:
                lines_at_limit.append((l, value(instance.LinePower[l,1])))
        else:
            if instance.ThermalLimit[l] - value(instance.LinePower[l,1]) <= 1e-5:
                lines_at_limit.append((l, value(instance.LinePower[l,1])))

    if len(lines_at_limit) == 0:
        print("No lines were at thermal limits")
    else:
        print(("%-"+str(max_line_label_length)+"s %14s") % ("Line at thermal limit","Flow"))
        for line, flow in lines_at_limit:
            print(("%-"+str(max_line_label_length)+"s %14.6f") % (line, flow))

    interfaces_at_to_limit = []
    interfaces_at_from_limit = []
    for i in sorted(instance.Interfaces):
        interface_flow = value(sum(instance.LinePower[l,1] for l in instance.InterfaceLines[i]) )
        if interface_flow <= 0.0:
            if instance.InterfaceToLimit[i] - -1.0*interface_flow <= 1e-5:
                interfaces_at_to_limit.append((i, -1.0*interface_flow))
        if interface_flow >= 0.0:
            if instance.InterfaceFromLimit[i] - interface_flow <= 1e-5:
                interfaces_at_from_limit.append((i, interface_flow))

    if (len(interfaces_at_to_limit) + len(interfaces_at_from_limit) == 0) and \
            (len(instance.Interfaces) > 0):
        print("No interfaces were at limits")

    if len(interfaces_at_to_limit) > 0:
        print(("%-"+str(max_line_label_length)+"s %14s") % ("Interface at to limit","Flow to"))
        for i, flow in interfaces_at_to_limit:
            print(("%-"+str(max_line_label_length)+"s %14.6f") % (i, flow))

    if len(interfaces_at_from_limit) > 0:
        print(("%-"+str(max_line_label_length)+"s %14s") % ("Interface at from limit","Flow from"))
        for i, flow in interfaces_at_from_limit:
            print(("%-"+str(max_line_label_length)+"s %14.6f") % (i, flow))


def available_quick_start_for_deterministic_sced(instance):
    """Given a SCED instance with commitments from the RUC,
    determine how much quick start capacity is available
    """
    available_quick_start_capacity = 0.0
    for g in instance.QuickStartGenerators:
        available = True  # until proven otherwise
        if int(round(value(instance.UnitOn[g, 1]))) == 1:
            available = False  # unit was already committed in the RUC
        elif instance.MinimumDownTime[g] > 1:
            # minimum downtime should be 1 or less, by definition of a quick start
            available = False
        elif (value(instance.UnitOnT0[g]) - int(round(value(instance.UnitOn[g, 1])))) == 1:
            # there cannot have been a a shutdown in the previous hour
            available = False

        if available:  # add the amount of power that can be accessed in the first hour
            # use the min() because scaled startup ramps can be larger than the generator limit
            available_quick_start_capacity += min(value(instance.ScaledStartupRampLimit[g]), value(instance.MaximumPowerOutput[g]))

    return available_quick_start_capacity


def report_renewables_curtailment_for_deterministic_sced(instance):

    # we only extract curtailment statistics for time period 1

    total_curtailment = round_small_values(sum((value(instance.MaxNondispatchablePower[g, 1]) -
                                                value(instance.NondispatchablePowerUsed[g, 1]))
                                               for g in instance.AllNondispatchableGenerators))

    if total_curtailment > 0:
        print("")
        print("Renewables curtailment reported at t=%d - total=%12.2f" % (1, total_curtailment))

    return total_curtailment



def report_on_off_and_ramps_for_deterministic_sced(instance):

    num_on_offs = 0
    sum_on_off_ramps = 0.0
    sum_nominal_ramps = 0.0  # this is the total ramp change for units not switching on/off

    for g in instance.ThermalGenerators:
        unit_on = int(round(value(instance.UnitOn[g, 1])))
        power_generated = value(instance.PowerGenerated[g, 1])
        if value(instance.UnitOnT0State[g]) > 0:
            # unit was on in previous time period
            if unit_on:
                # no change in state
                sum_nominal_ramps += math.fabs(power_generated - value(instance.PowerGeneratedT0[g]))
            else:
                num_on_offs += 1
                sum_on_off_ramps += power_generated
        else: # value(instance.UnitOnT0State[g]) < 0
            # unit was off in previous time period
            if not unit_on:
                # no change in state
                sum_nominal_ramps += math.fabs(power_generated - value(instance.PowerGeneratedT0[g]))
            else:
                num_on_offs += 1
                sum_on_off_ramps += power_generated

    print("")
    print("Number on/offs:       %12d" % num_on_offs)
    print("Sum on/off ramps:     %12.2f" % sum_on_off_ramps)
    print("Sum nominal ramps:    %12.2f" % sum_nominal_ramps)

    return num_on_offs, sum_on_off_ramps, sum_nominal_ramps





#####################################################################
# utility functions for pretty-printing solutions, for SCED and RUC #
#####################################################################

def output_sced_initial_condition(sced_instance, hour=1, max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Initial condition detail (gen-name t0-unit-on t0-unit-on-state t0-power-generated t1-unit-on must-run):")
    for g in sorted(sced_instance.ThermalGenerators):
        if hour == 1:
            print(("%-"+str(max_thermal_generator_label_length)+"s %5d %12.2f %5d %6d") %
                  (g,
                   value(sced_instance.UnitOnT0[g]),
                   value(sced_instance.PowerGeneratedT0[g]),
                   value(sced_instance.UnitOn[g,hour]),
                   value(sced_instance.MustRun[g])))
        else:
            print(("%-"+str(max_thermal_generator_label_length)+"s %5d %12.2f %5d %6d") %
                  (g,
                   value(sced_instance.UnitOn[g,hour-1]),
                   value(sced_instance.PowerGenerated[g,hour-1]),
                   value(sced_instance.UnitOn[g,hour]),
                   value(sced_instance.MustRun[g])))

def output_sced_demand(sced_instance, max_bus_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Demand detail:")
    for b in sorted(sced_instance.Buses):
        print(("%-"+str(max_bus_label_length)+"s %12.2f") %
              (b,
               value(sced_instance.Demand[b, 1])))

    print("")
    print(("%-"+str(max_bus_label_length)+"s %12.2f") %
          ("Reserve requirement:",
           value(sced_instance.ReserveRequirement[1])))

    print("")
    print("Maximum non-dispatachable power available:")
    for b in sorted(sced_instance.Buses):
        total_max_nondispatchable_power = sum(value(sced_instance.MaxNondispatchablePower[g, 1])
                                              for g in sced_instance.NondispatchableGeneratorsAtBus[b])
        print("%-30s %12.2f" % (b, total_max_nondispatchable_power))

    print("")
    print("Minimum non-dispatachable power available:")
    for b in sorted(sced_instance.Buses):
        total_min_nondispatchable_power = sum(value(sced_instance.MinNondispatchablePower[g, 1])
                                              for g in sced_instance.NondispatchableGeneratorsAtBus[b])
        print("%-30s %12.2f" % (b, total_min_nondispatchable_power))

def output_sced_solution(sced_instance, max_thermal_generator_label_length=DEFAULT_MAX_LABEL_LENGTH):

    print("Solution detail:")
    print("")
    print("Dispatch Levels (unit-on, power-generated, reserve-headroom)")
    for g in sorted(sced_instance.ThermalGenerators):
        unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
        print(("%-"+str(max_thermal_generator_label_length)+"s %2d %12.2f %12.2f") %
              (g,
               unit_on,
               value(sced_instance.PowerGenerated[g, 1]),
               math.fabs(value(sced_instance.MaximumPowerAvailable[g,1]) -
                         value(sced_instance.PowerGenerated[g,1]))),
              end=' ')
        if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MaximumPowerOutput[g])) <= 1e-5):
            print(" << At max output", end=' ')
        elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MinimumPowerOutput[g])) <= 1e-5):
            print(" << At min output", end=' ')
        if value(sced_instance.MustRun[g]):
            print(" ***", end=' ')
        print("")

    print("")
    print("Total power dispatched      = %12.2f"
          % sum(value(sced_instance.PowerGenerated[g,1]) for g in sced_instance.ThermalGenerators))
    print("Total reserve available     = %12.2f"
          % sum(value(sced_instance.MaximumPowerAvailable[g,1]) - value(sced_instance.PowerGenerated[g,1])
                for g in sced_instance.ThermalGenerators))
    print("Total quick start available = %12.2f"
          % available_quick_start_for_deterministic_sced(sced_instance))
    print("")

    print("Cost Summary (unit-on production-cost no-load-cost startup-cost)")
    total_startup_costs = 0.0
    for g in sorted(sced_instance.ThermalGenerators):
        unit_on = int(round(value(sced_instance.UnitOn[g, 1])))
        unit_on_t0 = int(round(value(sced_instance.UnitOnT0[g])))
        startup_cost = 0.0
        if unit_on_t0 == 0 and unit_on == 1:
            startup_cost = value(sced_instance.StartupCost[g,1])
        total_startup_costs += startup_cost
        print(("%-"+str(max_thermal_generator_label_length)+"s %2d %12.2f %12.2f %12.2f") %
              (g,
               unit_on,
               value(sced_instance.ProductionCost[g, 1]),
               unit_on * value(sced_instance.MinimumProductionCost[g,1]),
               startup_cost),
              end=' ')
        if (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                         value(sced_instance.MaximumPowerOutput[g])) <= 1e-5):
            print(" << At max output", end=' ')
        elif (unit_on == 1) and (math.fabs(value(sced_instance.PowerGenerated[g, 1]) -
                                 value(sced_instance.MinimumPowerOutput[g])) <= 1e-5):  # TBD - still need a tolerance parameter
            print(" << At min output", end=' ')
        print("")

    print("")
    print("Total cost = %12.2f" % (value(sced_instance.TotalNoLoadCost[1]) + value(sced_instance.TotalProductionCost[1]) +
                                   total_startup_costs))

# useful in cases where ramp rate constraints have been violated.
# ramp rates are taken from the original sced instance. unit on
# values can be taken from either instance, as they should be the
# same. power generated values must be taken from the relaxed instance.

def output_sced_ramp_violations(original_sced_instance, relaxed_sced_instance):

    # we are assuming that there are only a handful of violations - if that
    # is not the case, we should shift to some kind of table output format.
    for g in original_sced_instance.ThermalGenerators:
        for t in original_sced_instance.TimePeriods:

            current_unit_on = int(round(value(original_sced_instance.UnitOn[g, t])))

            if t == 1:
                previous_unit_on = int(round(value(original_sced_instance.UnitOnT0[g])))
            else:
                previous_unit_on = int(round(value(original_sced_instance.UnitOn[g, t-1])))

            if current_unit_on == 0:
                if previous_unit_on == 0:
                    # nothing is on, nothing to worry about!
                    pass
                else:
                    # the unit is switching off.
                    # TBD - eventually deal with shutdown ramp limits
                    pass

            else:
                if previous_unit_on == 0:
                    # the unit is switching on.
                    # TBD - eventually deal with startup ramp limits
                    pass
                else:
                    # the unit is remaining on.
                    if t == 1:
                        delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(relaxed_sced_instance.PowerGeneratedT0[g])
                    else:
                        delta_power = value(relaxed_sced_instance.PowerGenerated[g, t]) - value(relaxed_sced_instance.PowerGenerated[g, t-1])
                    if delta_power > 0.0:
                        # the unit is ramping up
                        if delta_power > value(original_sced_instance.NominalRampUpLimit[g]):
                            print("Thermal unit=%s violated nominal ramp up limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                  % (g, t-1, t, delta_power, value(original_sced_instance.NominalRampUpLimit[g])))
                    else:
                        # the unit is ramping down
                        if math.fabs(delta_power) > value(original_sced_instance.NominalRampDownLimit[g]):
                            print("Thermal unit=%s violated nominal ramp down limits from time=%d to time=%d - observed delta=%f, nominal limit=%f"
                                  % (g, t-1, t, math.fabs(delta_power), value(original_sced_instance.NominalRampDownLimit[g])))



######################################
# main simulation routine starts now #
######################################

def simulate(options):

    if options.simulator_plugin != None:
        try:
            simulator_plugin_module = pyutilib.misc.import_file(options.simulator_plugin)
        except:
            raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

        import prescient.plugins.default_plugin as simulator_plugin_default

        create_sced_instance = getattr(simulator_plugin_module, "create_sced_instance", None)
        if create_sced_instance is None:
            print("***WARNING***: Could not find function 'create_sced_instance' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            create_sced_instance = simulator_plugin_default.create_sced_instance
        else:
            print("Loaded simulator plugin function 'create_sced_instance' form simulator plugin module=%s."% options.simulator_plugin)

        relax_sced_ramp_rates = getattr(simulator_plugin_module, "relax_sced_ramp_rates", None)
        if relax_sced_ramp_rates is None:
            print("***WARNING***: Could not find function 'relax_sced_ramp_rates' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            relax_sced_ramp_rates = simulator_plugin_default.relax_sced_ramp_rates
        else:
            print("Loaded simulator plugin function 'relax_sced_ramp_rates' form simulator plugin module=%s."% options.simulator_plugin)

        create_and_solve_deterministic_ruc = getattr(simulator_plugin_module, "create_and_solve_deterministic_ruc", None)
        if create_and_solve_deterministic_ruc is None:
            print("***WARNING***: Could not find function 'create_and_solve_deterministic_ruc' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            create_and_solve_deterministic_ruc = simulator_plugin_default.create_and_solve_deterministic_ruc
        else:
            print("Loaded simulator plugin function 'create_and_solve_deterministic_ruc' form simulator plugin module=%s."% options.simulator_plugin)

        create_and_solve_stochastic_ruc_via_ef = getattr(simulator_plugin_module, "create_and_solve_stochastic_ruc_via_ef", None)
        if create_and_solve_stochastic_ruc_via_ef is None:
            print("***WARNING***: Could not find function 'create_and_solve_stochastic_ruc_via_ef' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            create_and_solve_stochastic_ruc_via_ef = simulator_plugin_default.create_and_solve_stochastic_ruc_via_ef
        else:
            print("Loaded simulator plugin function 'solve_deterministic_ruc' form simulator plugin module=%s."% options.simulator_plugin)

        create_and_solve_stochastic_ruc_via_ph = getattr(simulator_plugin_module, "create_and_solve_stochastic_ruc_via_ph", None)
        if create_and_solve_stochastic_ruc_via_ph is None:
            print("***WARNING***: Could not find function 'create_and_solve_stochastic_ruc_via_ph' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            create_and_solve_stochastic_ruc_via_ph = simulator_plugin_default.create_and_solve_stochastic_ruc_via_ph
        else:
            print("Loaded simulator plugin function 'create_and_solve_stochastic_ruc_via_ph' form simulator plugin module=%s."% options.simulator_plugin)

        compute_simulation_filename_for_date = getattr(simulator_plugin_module, "compute_simulation_filename_for_date", None)
        if compute_simulation_filename_for_date is None:
            print("***WARNING***: Could not find function 'compute_simulation_filename_for_date' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            compute_simulation_filename_for_date = simulator_plugin_default.compute_simulation_filename_for_date
        else:
            print("Loaded simulator plugin function 'compute_simulation_filename_for_date' form simulator plugin module=%s."% options.simulator_plugin)

        solve_deterministic_day_ahead_pricing_problem = getattr(simulator_plugin_module, "solve_deterministic_day_ahead_pricing_problem", None)
        if solve_deterministic_day_ahead_pricing_problem is None:
            print("***WARNING***: Could not find function 'solve_deterministic_day_ahead_pricing_problem' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            solve_deterministic_day_ahead_pricing_problem = simulator_plugin_default.solve_deterministic_day_ahead_pricing_problem
        else:
            print("Loaded simulator plugin function 'solve_deterministic_day_ahead_pricing_problem' form simulator plugin module=%s."% options.simulator_plugin)

        compute_market_settlements = getattr(simulator_plugin_module, "compute_simulation_filename_for_date", None)
        if compute_market_settlements is None:
            print("***WARNING***: Could not find function 'compute_market_settlements' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            compute_market_settlements = simulator_plugin_default.compute_market_settlements
        else:
            print("Loaded simulator plugin function 'compute_market_settlements' form simulator plugin module=%s."% options.simulator_plugin)

        create_ruc_instance_to_simulate_next_period = getattr(simulator_plugin_module, "compute_simulation_filename_for_date", None)
        if create_ruc_instance_to_simulate_next_period is None:
            print("***WARNING***: Could not find function 'create_ruc_instance_to_simulate_next_period' in simulator plugin module=%s, using default!" % options.simulator_plugin)
            create_ruc_instance_to_simulate_next_period = simulator_plugin_default.create_ruc_instance_to_simulate_next_period
        else:
            print("Loaded simulator plugin function 'create_ruc_instance_to_simulate_next_period' form simulator plugin module=%s."% options.simulator_plugin)

    #NOTE: deterministic_ruc_solver_plugin is ignored if a simulator_plugin is passed
    elif options.deterministic_ruc_solver_plugin != None:
        try:
            solver_plugin_module = pyutilib.misc.import_file(options.deterministic_ruc_solver_plugin)
        except:
            raise RuntimeError("Could not locate simulator plugin module=%s" % options.simulator_plugin)

        solve_function = getattr(solver_plugin_module, "solve_deterministic_ruc", None)
        if solve_function is None:
            raise RuntimeError("Could not find function 'solve_deterministic_ruc' in simulator plugin module=%s, using default!" % options.deterministic_ruc_solver_plugin)
        else:
            print("Loaded deterministic ruc solver plugin function 'solve_deterministic_ruc' form simulator plugin module=%s."% options.deterministic_ruc_solver_plugin)
            from prescient.plugins.default_plugin import (create_sced_instance,
                                                          relax_sced_ramp_rates,
                                                          create_create_and_solve_deterministic_ruc,
                                                          create_and_solve_stochastic_ruc_via_ef,
                                                          create_and_solve_stochastic_ruc_via_ph,
                                                          compute_simulation_filename_for_date,
                                                          solve_deterministic_day_ahead_pricing_problem,
                                                          compute_market_settlements,
                                                          create_ruc_instance_to_simulate_next_period,
                                                          )
            create_and_solve_deterministic_ruc = create_create_and_solve_deterministic_ruc(solve_function)


    else:
        from prescient.plugins.default_plugin import (create_sced_instance,
                                                      relax_sced_ramp_rates,
                                                      create_and_solve_deterministic_ruc,
                                                      create_and_solve_stochastic_ruc_via_ef,
                                                      create_and_solve_stochastic_ruc_via_ph,
                                                      compute_simulation_filename_for_date,
                                                      solve_deterministic_day_ahead_pricing_problem,
                                                      compute_market_settlements,
                                                      create_ruc_instance_to_simulate_next_period,
                                                      get_dispatch_levels
                                                      )
        # Xian: import strategic bidding class
        from prescient.plugins.strategic_bidding import DAM_thermal_bidding

    # echo the key outputs for the simulation - for replicability.
    print("Initiating simulation...")
    print("")
    print("Model directory:", options.model_directory)
    print("Data directory:", options.data_directory)
    print("Output directory:", options.output_directory)
    print("Random seed:", options.random_seed)
    print("Reserve factor:", options.reserve_factor)
    print("")

    # do some simple high-level checking on the existence
    # of the model and input data directories.
    if not os.path.exists(options.model_directory):
        raise RuntimeError("Model directory=%s does not exist or cannot be read" % options.model_directory)

    if not os.path.exists(options.data_directory):
        raise RuntimeError("Data directory=%s does not exist or cannot be read" % options.data_directory)

    # echo the solver configuration
    if not options.run_deterministic_ruc:
        if not options.solve_with_ph:
            print("Solving stochastic RUC instances using the extensive form")
        else:
            print("Solving stochastic RUC instances using progressive hedging - mode=%s" % options.ph_mode)
        print("")

    # do a check, PH is really only compatable with running prescient in "normal" mode
    if options.solve_with_ph and (options.ruc_every_hours != 24):
        raise RuntimeError("Prescient currently only supports --solve-with-ph with "
                           "--ruc-every-hours=24")

    # not importing the time module here leads to a weird error message -
    # "local variable 'time' referenced before assignment"
    import time
    simulation_start_time = time.time()

    # seed the generator if a user-supplied seed is provided. otherwise,
    # python will seed from the current system time.
    if options.random_seed > 0:
        random.seed(options.random_seed)

    # validate the PH execution mode options.
    if options.ph_mode != "serial" and options.ph_mode != "localmpi":
        raise RuntimeError("Unknown PH execution mode=" + str(options.ph_mode) +
                           " specified - legal values are: 'serial' and 'localmpi'")

    # Model choice control
    run_ruc = (options.disable_ruc == False)
    run_sced = (options.disable_sced == False)

    model_filename = os.path.join(options.model_directory, "ReferenceModel.py")
    if not os.path.exists(model_filename):
        raise RuntimeError("The model %s either does not exist or cannot be read" % model_filename)

    ## check the pricing options
    pricing_list = ["LMP", "ELMP", "aCHP"]
    if options.day_ahead_pricing not in pricing_list:
        raise RuntimeError("Unknown pricing mechanism=" + options.day_ahead_pricing +
                           " specified - legal values are: "+", ".join(pricing_list) )

    if not options.run_deterministic_ruc and options.compute_market_settlements:
        raise RuntimeError("Prescient currently does not compute market settlements in stochastic RUC mode. "+
                           "Use option --run-deterministic-ruc to solve deterministic RUCs.")

    from pyutilib.misc import import_file
    reference_model_module = import_file(model_filename)

    # make sure all utility methods required by the simulator are defined in the reference model module.
    validate_reference_model(reference_model_module)

    ruc_model = reference_model_module.load_model_parameters()
    sced_model = reference_model_module.model

    # state variables
    offline_generators = []
    offline_lines_B = {}

    # NOTE: We eventually want specific solver options for the various types below.
    if options.python_io:
        deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type, solver_io="python")
        ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type, solver_io="python")
        sced_solver = SolverFactory(options.sced_solver_type, solver_io="python")
    else:
        deterministic_ruc_solver = SolverFactory(options.deterministic_ruc_solver_type)
        ef_ruc_solver = SolverFactory(options.ef_ruc_solver_type)
        sced_solver = SolverFactory(options.sced_solver_type)

    solve_options = {}
    solve_options[sced_solver] = {}
    solve_options[deterministic_ruc_solver] = {}
    solve_options[ef_ruc_solver] = {}

    for s in solve_options:
        solve_options[s]['symbolic_solver_labels'] = options.symbolic_solver_labels

    ##TODO: work out why we get an error from ProblemWriter_cpxlp when this is False
    if options.warmstart_ruc:
        solve_options[deterministic_ruc_solver]['warmstart'] = options.warmstart_ruc
        solve_options[ef_ruc_solver]['warmstart'] = options.warmstart_ruc

    if options.deterministic_ruc_solver_type == "cplex" or options.deterministic_ruc_solver_type == "cplex_persistent":
        deterministic_ruc_solver.options.mip_tolerances_mipgap = options.ruc_mipgap
    elif options.deterministic_ruc_solver_type == "gurobi" or options.deterministic_ruc_solver_type == "gurobi_persistent":
        deterministic_ruc_solver.options.MIPGap = options.ruc_mipgap
    elif options.deterministic_ruc_solver_type == "cbc":
        deterministic_ruc_solver.options.ratioGap = options.ruc_mipgap
    elif options.deterministic_ruc_solver_type == "glpk":
        deterministic_ruc_solver.options.mipgap = options.ruc_mipgap
    else:
        raise RuntimeError("Unknown solver type=%s specified" % options.deterministic_ruc_solver_type)

    if options.ef_ruc_solver_type == "cplex" or options.ef_ruc_solver_type == "cplex_persistent":
        ef_ruc_solver.options.mip_tolerances_mipgap = options.ef_mipgap
    elif options.ef_ruc_solver_type == "gurobi" or options.ef_ruc_solver_type == "gurobi_persistent":
        ef_ruc_solver.options.MIPGap = options.ef_mipgap
    elif options.ef_ruc_solver == "cbc":
        ef_ruc_solver.options.ratioGap = options.ruc_mipgap
    elif options.ef_ruc_solver == "glpk":
        ef_ruc_solver.options.mipgap = options.ruc_mipgap
    else:
        raise RuntimeError("Unknown solver type=%s specified" % options.ef_ruc_solver_type)

    ef_ruc_solver.set_options("".join(options.stochastic_ruc_ef_solver_options))
    deterministic_ruc_solver.set_options("".join(options.deterministic_ruc_solver_options))
    sced_solver.set_options("".join(options.sced_solver_options))

    # validate the start date
    try:
        start_date = dateutil.parser.parse(options.start_date)
    except ValueError:
        print("***ERROR: Illegally formatted start date=" + options.start_date + " supplied!")
        sys.exit(1)

    # Initialize nested dictionaries for holding output
    dates_to_simulate = [str(a_date)
                         for a_date in daterange(start_date.date(),
                                                 start_date.date() +
                                                 datetime.timedelta(options.num_days-1))]
    end_date = dates_to_simulate[-1]

    if 24%options.ruc_every_hours != 0:
        raise RuntimeError("--ruc-every-hours must be a divisor of 24! %d supplied!"%options.ruc_every_hours)
    ruc_start_hours = list(range(0, 24, options.ruc_every_hours))
    print("ruc_start_hours:", ruc_start_hours)

    list_hours = dict( (ruc_start_hours[i], list(range(ruc_start_hours[i], ruc_start_hours[i+1])))
                        for i in range(0, len(ruc_start_hours)-1))
    list_hours[ruc_start_hours[-1]] = list(range(ruc_start_hours[-1],24))

    # TBD: eventually option-drive the interval (minutely) within an hour that
    #      economic dispatch is executed. for now, because we don't have better
    #      data, drive it once per hour.
    list_minutes = list(range(0, 60, 60))

    ruc_horizon = options.ruc_horizon
    if not options.ruc_every_hours <= ruc_horizon <= 48:
        raise RuntimeError("--ruc-horizon must be greater than or equal --ruc-every-hours and less than or equal to 48! %d supplied!"%options.ruc_horizon)

    print("Dates to simulate:",dates_to_simulate)

    # we often want to write SCED instances, for diagnostics.
    lp_writer = ProblemWriter_cpxlp()

    # before the simulation starts, delete the existing contents of the output directory.
    if os.path.exists(options.output_directory):
        shutil.rmtree(options.output_directory)
    os.mkdir(options.output_directory)

    # create an output directory for plot data.
    # for now, it's only a single directory - not per-day.
    os.mkdir(os.path.join(options.output_directory,"plots"))

    if run_ruc:
        last_ruc_date = dates_to_simulate[-1]
        print("")
        print("Last RUC date:", last_ruc_date)

    scenario_instances_for_next_period = None
    scenario_tree_for_next_period = None

    deterministic_ruc_instance_for_next_period = None

    # build all of the per-date output directories up-front, for simplicity.
    for this_date in dates_to_simulate:
        simulation_directory_for_this_date = options.output_directory + os.sep + str(this_date)
        os.mkdir(options.output_directory + os.sep + str(this_date))

    ########################################################################################
    # we need to create the "yesterday" deterministic or stochastic ruc instance, to kick  #
    # off the simulation process. for now, simply solve RUC for the first day, as          #
    # specified in the instance files. in practice, this may not be the best idea but it   #
    # will at least get us a solution to start from.                                       #
    ########################################################################################

    print("")
    first_date = dates_to_simulate[0]
    first_hour = ruc_start_hours[0]
    if first_date == dates_to_simulate[-1]:
        second_date = None
    else:
        second_date = dates_to_simulate[1]

    # Xian: add 2 np arrays to store RUC and SCED schedules for the interested generator
    ruc_schedule_arr = np.zeros((24,options.num_days))
    sced_schedule_arr = np.zeros((24,options.num_days))

    # Xian: add 2 np arrays to store
    # 1. the actual total power output from the hybrid system
    # 2. power output from the thermal generator in the hybrid system
    total_power_delivered_arr = np.zeros((24,options.num_days)) # P_R
    thermal_power_delivered_arr = np.zeros((24,options.num_days)) # P_G
    thermal_power_generated_arr = np.zeros((24,options.num_days)) # P_T

    # initialize the model class
    thermalBid = DAM_thermal_bidding(n_scenario=10)

    # build bidding model
    if options.bidding:

        m_bid = thermalBid.create_bidding_model(generator = options.bidding_generator)
        price_forecast_dir = '../../prescient/plugins/price_forecasts/date={}_lmp_forecasts.csv'.format(first_date)
        cost_curve_store_dir = '../../prescient/plugins/cost_curves/'

        # solve the bidding model for the first simulation day
        thermalBid.stochastic_bidding(m_bid,price_forecast_dir,cost_curve_store_dir,first_date)

    # build tracking model
    if options.track_ruc_signal:
        print('Building a track model for RUC signals.')
        m_track_ruc = thermalBid.build_tracking_model(options.ruc_horizon,\
        generator = options.bidding_generator,track_type = 'RUC',hybrid = options.hybrid_tracking)

    elif options.track_sced_signal:
        print('Building a track model for SCED signals.')
        m_track_sced = thermalBid.build_tracking_model(options.sced_horizon,\
        generator = options.bidding_generator,track_type = 'SCED',hybrid = options.hybrid_tracking)

        # initialize a list/array to record the power output of sced tracking
        # model in real-time, so it can update the initial condition properly
        # every hour
        sced_tracker_power_record = {options.bidding_generator: \
        np.repeat(value(m_track_sced.pre_pow[options.bidding_generator]),\
        value(m_track_sced.pre_up_hour[options.bidding_generator]))}

    # not importing the time module here leads to a weird error message -
    # "local variable 'time' referenced before assignment"
    import time
    start_time = time.time()

    # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
    # prescient processing is identical in the case of deterministic and stochastic RUC.
    # persistent processing differs, as there is no point forecast for stochastic RUC.
    use_prescient_forecast_error_in_sced = True # always the default
    use_persistent_forecast_error_in_sced = False
    if options.run_sced_with_persistent_forecast_errors:
        print("Using persistent forecast error model when projecting demand and renewables in SCED")
        use_persistent_forecast_error_in_sced = True
        use_prescient_forecast_error_in_sced = False
    else:
        print("Using prescient forecast error model when projecting demand and renewables in SCED")
    print("")

    if options.run_ruc_with_next_day_data:
        print("Using next day data for 48 hour RUC solves")
        use_next_day_in_ruc = True
    else:
        print("Using only the next 24 hours of data for RUC solves")
        use_next_day_in_ruc = False

    if options.run_deterministic_ruc:
        deterministic_ruc_instance_for_next_period, \
        scenario_tree_for_next_period = create_and_solve_deterministic_ruc(deterministic_ruc_solver,
                                                                        solve_options,
                                                                        options,
                                                                        first_date,
                                                                        first_hour,
                                                                        second_date,
                                                                        None,
                                                                        # no yesterday deterministic ruc
                                                                        None,
                                                                        # no yesterday scenario tree
                                                                        options.output_ruc_initial_conditions,
                                                                        None, # no projected SCED instance
                                                                        None, # no SCED schedule hour
                                                                        ruc_horizon,
                                                                        use_next_day_in_ruc,
                                                                        )

    else:
        if options.solve_with_ph == False:
            scenario_instances_for_next_period, \
            scenario_tree_for_next_period = create_and_solve_stochastic_ruc_via_ef(ef_ruc_solver,
                                                                                solve_options,
                                                                                options,
                                                                                first_date,
                                                                                first_hour,
                                                                                second_date,
                                                                                None,
                                                                                # no yesterday stochastic ruc scenarios
                                                                                None,
                                                                                # no yesterday scenario tree
                                                                                options.output_ruc_initial_conditions,
                                                                                None, # no projected SCED instance
                                                                                None, # no SCED schedule hour
                                                                                ruc_horizon,
                                                                                use_next_day_in_ruc,
                                                                                )
        else:
            scenario_instances_for_next_period, \
            scenario_tree_for_next_period = create_and_solve_stochastic_ruc_via_ph(solver,
                                                                                None,
                                                                                options,
                                                                                first_date,
                                                                                first_hour,
                                                                                second_date,
                                                                                None,
                                                                                # no yesterday stochastic ruc scenarios
                                                                                None,
                                                                                # no yesterday scenario tree
                                                                                options.output_ruc_initial_conditions,
                                                                                None, # no projected SCED instance
                                                                                None, # no SCED schedule Hour
                                                                                ruc_horizon,
                                                                                use_next_day_in_ruc,
                                                                                )


    print("")
    print("Construction and solve time=%.2f seconds" % (time.time() - start_time))

    # Xian: printing out the dipatch level of a certain unit
    print("")
    print("Getting dispatch levels...\n")
    if options.run_deterministic_ruc:
        ruc_dispatch_level_for_next_period = get_dispatch_levels(deterministic_ruc_instance_for_next_period,\
        options.bidding_generator,verbose = True)

        # record the ruc signal from the first day
        ruc_schedule_arr[:,0] = np.array(ruc_dispatch_level_for_next_period[options.bidding_generator]).flatten()[:24]

    # Xian: pass this schedule to track
    if options.track_ruc_signal:
        thermalBid.pass_schedule_to_track_and_solve(m_track_ruc,ruc_dispatch_level_for_next_period,\
        RT_price=None, deviation_weight = options.deviation_weight, \
        ramping_weight = options.ramping_weight,\
        cost_weight = options.cost_weight)

        # record the track power output profile
        if options.hybrid_tracking == False:
            track_gen_pow_ruc = thermalBid.extract_pow_s_s(m_track_ruc, horizon =\
                                24, verbose = False)
            thermal_track_gen_pow_ruc = track_gen_pow_ruc
            thermal_generated_ruc = track_gen_pow_ruc
        else:
            track_gen_pow_ruc, thermal_track_gen_pow_ruc,thermal_generated_ruc =\
                                thermalBid.extract_pow_s_s(m_track_ruc,horizon =\
                                24, hybrid = True,verbose = False)

        # record the total power delivered
        # and thermal power delivered
        total_power_delivered_arr[:,0] = track_gen_pow_ruc[options.bidding_generator]
        thermal_power_delivered_arr[:,0] = thermal_track_gen_pow_ruc[options.bidding_generator]
        thermal_power_generated_arr[:,0] = thermal_generated_ruc[options.bidding_generator]

        # update the track model
        thermalBid.update_model_params(m_track_ruc,thermal_generated_ruc,hybrid = options.hybrid_tracking)
        thermalBid.reset_constraints(m_track_ruc,options.ruc_horizon)

    DA_ruc_callback(deterministic_ruc_instance_for_next_period, first_date)

    if options.compute_market_settlements:
        day_ahead_prices_for_next_period, \
        day_ahead_reserve_prices_for_next_period, \
        day_ahead_thermal_cleared_for_next_period, \
        day_ahead_reserve_cleared_for_next_period, \
        day_ahead_renewable_cleared_for_next_period = solve_deterministic_day_ahead_pricing_problem(deterministic_ruc_solver,
                                                                                              solve_options,
                                                                                              deterministic_ruc_instance_for_next_period,
                                                                                              options,
                                                                                              reference_model_module)

        report_prices_for_deterministic_ruc(day_ahead_prices_for_next_period, day_ahead_reserve_prices_for_next_period, deterministic_ruc_instance_for_next_period, options)
        DA_market_callback(first_date, day_ahead_prices_for_next_period, day_ahead_reserve_prices_for_next_period, day_ahead_thermal_cleared_for_next_period, day_ahead_reserve_prices_for_next_period, day_ahead_renewable_cleared_for_next_period)

    ##################################################################################
    # variables to track the aggregate simulation costs and load shedding quantities #
    ##################################################################################

    total_overall_fixed_costs = 0.0
    total_overall_generation_costs = 0.0
    total_overall_load_shedding = 0.0
    total_overall_over_generation = 0.0
    total_overall_reserve_shortfall = 0.0
    total_overall_renewables_curtailment = 0.0
    total_on_offs = 0
    total_sum_on_off_ramps = 0.0
    total_sum_nominal_ramps = 0.0
    total_quick_start_additional_costs = 0.0
    total_quick_start_additional_power_generated = 0.0

    if options.compute_market_settlements:

        total_thermal_energy_payments = 0.0
        total_renewable_energy_payments = 0.0
        total_energy_payments = 0.0

        total_reserve_payments = 0.0

        total_thermal_uplift_payments = 0.0
        total_renewable_uplift_payments = 0.0
        total_uplift_payments = 0.0

        total_payments = 0.0

    ###############################################################################
    # variables to track daily statistics, for plotting or summarization purposes #
    # IMPT: these are the quantities that are also written to a CSV file upon     #
    #       completion of the simulation.                                         #
    ###############################################################################

    daily_total_costs = []
    daily_fixed_costs = []
    daily_generation_costs = []
    daily_load_shedding = []
    daily_over_generation = []
    daily_reserve_shortfall = []
    daily_renewables_available = []
    daily_renewables_used = []
    daily_renewables_curtailment = []
    daily_demand = []
    daily_average_price = []
    daily_on_offs = []
    daily_sum_on_off_ramps = []
    daily_sum_nominal_ramps = []
    daily_quick_start_additional_costs = []
    daily_quick_start_additional_power_generated = []

    if options.compute_market_settlements:
        daily_thermal_energy_payments = []
        daily_renewable_energy_payments = []
        daily_energy_payments = []

        daily_reserve_payments = []

        daily_thermal_uplift_payments = []
        daily_renewable_uplift_payments = []
        daily_uplift_payments = []

        daily_total_payments = []


    # statistics accumulated over the entire simulation, to compute effective
    # renewables penetration rate across the time horizon of the simulation.
    cumulative_demand = 0.0
    cumulative_renewables_used = 0.0

    ############################
    # create output dataframes #
    ############################
    daily_summary_cols = ['Date','Demand','Renewables available', 'Renewables used', 'Renewables penetration rate','Average price','Fixed costs','Generation costs','Load shedding','Over generation','Reserve shortfall','Renewables curtailment','On/off','Sum on/off ramps','Sum nominal ramps']

    if options.compute_market_settlements:
        daily_summary_cols += ['Renewables energy payments', 'Renewables uplift payments', 'Thermal energy payments','Thermal uplift payments', 'Total energy payments','Total reserve payments', 'Total uplift payments','Total payments','Average payments' ]

    daily_summary_df = pd.DataFrame(columns=daily_summary_cols)

    options_df = pd.DataFrame(columns=['Date', 'Model directory', 'Data directory', 'Output directory', 'Random seed', 'Reserve factor', 'Ruc mipgap', 'Solver type','Num days', 'Sart Date', 'Run deterministic ruc', 'Run sced with persistend forecast errors', 'Output ruc solutions', 'Options ruc dispatches', 'Output solver log', 'Relax ramping if infeasible', 'Output sced solutions', 'Plot individual generators', 'Output sced initial conditions', 'Output sced demands', 'Simulate out of sample', 'Output ruc initial conditions', 'Sced horizon', 'Traceback', 'Run simulator'])

    thermal_generator_dispatch_df = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Dispatch', 'Headroom', 'Unit State'])

    renewables_production_df  = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Output', 'Curtailment'])

    line_df = pd.DataFrame(columns=['Date', 'Hour', 'Line', 'Flow'])

    if options.compute_market_settlements:
        bus_df = pd.DataFrame(columns=['Date', 'Hour', 'Bus', 'Shortfall', 'Overgeneration', 'LMP', 'LMP DA'])
    else:
        bus_df = pd.DataFrame(columns=['Date', 'Hour', 'Bus', 'Shortfall', 'Overgeneration', 'LMP'])


    overall_output_cols = ['Total demand','Total fixed costs','Total generation costs','Total costs','Total load shedding','Total over generation','Total reserve shortfall','Total renewables curtialment','Total on/offs','Total sum on/off ramps','Total sum nominal ramps','Maximum observed demand','Overall renewables penetration rate','Cumulative average price']
    if options.compute_market_settlements:
        overall_output_cols += ['Total energy payments','Total reserve payments', 'Total uplift payments','Total payments','Cumalative average payments']

    overall_simulation_output_df = pd.DataFrame(columns=overall_output_cols)

    quickstart_summary_df = pd.DataFrame(columns=['Date', 'Hour', 'Generator', 'Used as quickstart', 'Dispatch level of quick start generator'])

    hourly_gen_summary_df = pd.DataFrame(columns=['Date', 'Hour', 'Load shedding', 'Reserve shortfall', 'Available reserves', 'Over generation', 'Reserve Price DA', 'Reserve Price RT'])

    runtime_df = pd.DataFrame(columns=['Date','Hour','Type', 'Solve Time'])

    ###################################
    # prep the hourly output CSV file #
    ###################################

    csv_hourly_output_filename = os.path.join(options.output_directory,"hourly_summary.csv")
    csv_hourly_output_file = open(csv_hourly_output_filename,"w")
    sim_dates = [this_date for this_date in dates_to_simulate]
    print("Date", ",", "Hour", ",", "TotalCosts", ",", "FixedCosts", ",", "VariableCosts", ",", "LoadShedding", \
          ",", "OverGeneration", ",", "ReserveShortfall", ",", "RenewablesUsed", ",", "RenewablesCurtailed", ",",
          "Demand", ",", "Price", file=csv_hourly_output_file)

    #################################################################
    # construct the simiulation data associated with the first date #
    #################################################################

    # identify the .dat file from which (simulated) actual load data will be drawn.
    simulated_dat_filename = compute_simulation_filename_for_date(dates_to_simulate[0], options)

    print("")
    print("Actual simulation data drawn from file=" + simulated_dat_filename)

    if not os.path.exists(simulated_dat_filename):
        raise RuntimeError("The file " + simulated_dat_filename + " does not exist or cannot be read.")

    # the RUC instance to simulate only exists to store the actual demand and renewables outputs
    # to be realized during the course of a day. it also serves to provide a concrete instance,
    # from which static data and topological features of the system can be extracted.
    # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
    #            economic dispatch instances, as that would enable those instances to be
    #            prescient.
    print("")
    print("Creating RUC instance to simulate")
    ruc_instance_to_simulate_next_period = ruc_model.create_instance(simulated_dat_filename)

    #####################################################################################################
    # the main simulation engine starts here - loop through each day, performing ISO-related activities #
    #####################################################################################################

    if options.simulate_out_of_sample != False:
        print("")
        print("Executing simulation using out-of-sample scenarios")

    first_pass = True
    for this_date in dates_to_simulate:

        # preliminaries
        print("")
        print(">>>>Simulating date: "+this_date)
        if this_date == dates_to_simulate[-1]:
            next_date = None
        else:
            next_date = dates_to_simulate[dates_to_simulate.index(this_date)+1]
        ## for commitment
        if next_date is None or next_date == dates_to_simulate[-1]:
            next_next_date = None
        else:
            next_next_date = dates_to_simulate[dates_to_simulate.index(next_date)+1]
        print("")

        # Xian: make RUC schedule for the current day
        ruc_dispatch_level_current = ruc_dispatch_level_for_next_period

        # the thermal fleet capacity is necessarily a static quantity, so it
        # technically doesn't have to be computed each iteration. however, we
        # don't have an instance outside of the date loop, and it doesn't cost
        # anything to compute. this quantity is primarily used as a normalization
        # term for stackgraph generation.
        thermal_fleet_capacity = sum(value(ruc_instance_to_simulate_next_period.MaximumPowerOutput[g])
                                     for g in ruc_instance_to_simulate_next_period.ThermalGenerators)

        # track the peak demand in any given hour of the simulation, to report
        # at the end. the primary use of this data is to appropriate scale
        # the stack graph plots in subsequent runs.
        max_hourly_demand = 0.0

        # create a dictionary of observed dispatch values during the course of the
        # day - for purposes of generating stack plots and related outputs.
        observed_thermal_dispatch_levels = {}
        for g in ruc_instance_to_simulate_next_period.ThermalGenerators:
            observed_thermal_dispatch_levels[g] = np.array([0.0 for r in range(24)])

        # useful to know the headroom for thermal generators as well.
        # headroom - max-available-power minus actual dispatch.
        observed_thermal_headroom_levels = {}
        for g in ruc_instance_to_simulate_next_period.ThermalGenerators:
            observed_thermal_headroom_levels[g] = np.array([0.0 for r in range(24)])

        # do the above, but for renewables power used (*not* available)
        observed_renewables_levels = {}
        for g in ruc_instance_to_simulate_next_period.AllNondispatchableGenerators:
            observed_renewables_levels[g] = np.array([0.0 for r in range(24)])

        # and curtailment...
        observed_renewables_curtailment = {}
        for g in ruc_instance_to_simulate_next_period.AllNondispatchableGenerators:
            observed_renewables_curtailment[g] = np.array([0.0 for r in range(24)])

        # dictionary for on/off states of thermal generators in real-time dispatch.
        observed_thermal_states = {}
        for g in ruc_instance_to_simulate_next_period.ThermalGenerators:
            observed_thermal_states[g] = np.array([-1 for r in range(24)])

        # dictionary for obversed costs of thermal generators in real-time dispatch.
        observed_costs = {}
        for g in ruc_instance_to_simulate_next_period.ThermalGenerators:
            observed_costs[g] = np.array([0.0 for r in range(24)])

        # dictionary for line flows.
        observed_flow_levels = {}
        for l in ruc_instance_to_simulate_next_period.TransmissionLines:
            observed_flow_levels[l] = np.array([0.0 for r in range(24)])

        # dictionary for bus load-generate mismatch.
        observed_bus_mismatches = {}
        for b in ruc_instance_to_simulate_next_period.Buses:
            observed_bus_mismatches[b] = np.array([0.0 for r in range(24)])

        # dictionary for bus LMPs.
        observed_bus_LMPs = {}
        for b in ruc_instance_to_simulate_next_period.Buses:
            observed_bus_LMPs[b] = np.array([0.0 for r in range(24)])

        # dictionary of input and output levels for storage units
        storage_input_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_next_period.Storage:
            storage_input_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        storage_output_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_next_period.Storage:
            storage_output_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        storage_soc_dispatchlevelsdict = {}
        for s in ruc_instance_to_simulate_next_period.Storage:
            storage_soc_dispatchlevelsdict[s] = np.array([0.0 for r in range(24)])

        # SCED run-times, in seconds.
        sced_runtimes = []

        # keep track of any events that are worth annotating on daily generation plots.
        # the entries in this list should be (x,y) pairs, where x is the event hour and
        # and y is the associated text label.
        event_annotations = []

        # track the total curtailment across the day, for output / plot generation purposes.
        curtailments_by_hour = []

        # track the total load shedding across the day, for output / plot generation purposes.
        load_shedding_by_hour = []

        # track the total over generation across the day, for output / plot generation purposes.
        over_generation_by_hour = []

        # the reserve requirements as induced by the SCED.
        reserve_requirements_by_hour = []

        # the reserve price in the SCED
        reserve_RT_price_by_hour = []

        # shortfalls in reserve requirements as induced by the SCED.
        reserve_shortfalls_by_hour = []

        # total available reserve as computed by the SCED.
        available_reserves_by_hour = []

        # available quickstart by hour as computed by the SCED.
        available_quickstart_by_hour = []

        # quick start generators committed before unfixing the SCED
        fixed_quick_start_generators_committed = []

        # quick start generators committed after unfixing the SCED
        unfixed_quick_start_generators_committed = []

        # quick start additional costs by hour as computed by the SCED
        quick_start_additional_costs_by_hour = []

        # quick start additional power generated by hour as computed by the SCED
        quick_start_additional_power_generated_by_hour = []

        # generators used as quick start by hour as computed by the SCED
        used_as_quick_start = {}

        if options.enable_quick_start_generator_commitment:
            for g in ruc_instance_to_simulate_next_period.QuickStartGenerators:
                used_as_quick_start[g]=[]


        # NOTE: Not sure if the demand should be demand desired - or satisfied? Right now, it's the former.
        this_date_demand = 0.0
        this_date_fixed_costs = 0.0
        this_date_variable_costs = 0.0
        this_date_over_generation = 0.0
        this_date_load_shedding = 0.0
        this_date_reserve_shortfall = 0.0
        this_date_renewables_available = 0.0
        this_date_renewables_used = 0.0
        this_date_renewables_curtailment = 0.0
        this_date_on_offs = 0
        this_date_sum_on_off_ramps = 0.0
        this_date_sum_nominal_ramps = 0.0
        this_date_quick_start_additional_costs = 0.0
        this_date_quick_start_additional_power_generated = 0.0

        if options.compute_market_settlements:
            this_date_planning_energy_prices = {}
            this_date_planning_reserve_prices = {}
            this_date_planning_thermal_generation_cleared = {}
            this_date_planning_thermal_reserve_cleared = {}
            this_date_planning_renewable_generation_cleared = {}

        ## this is the beginning of the day, grab the data from the last market settlement
        if (options.ruc_execution_hour != 0) or first_pass:
            if options.compute_market_settlements:
                for t in range(0,options.ruc_every_hours):
                    for b in ruc_instance_to_simulate_next_period.Buses:
                        this_date_planning_energy_prices[b,t] = day_ahead_prices_for_next_period[b,t]
                    this_date_planning_reserve_prices[t] = day_ahead_reserve_prices_for_next_period[t]
                    for g in ruc_instance_to_simulate_next_period.ThermalGenerators:
                        this_date_planning_thermal_generation_cleared[g,t] = day_ahead_thermal_cleared_for_next_period[g,t]
                        this_date_planning_thermal_reserve_cleared[g,t] = day_ahead_reserve_cleared_for_next_period[g,t]
                    for g in ruc_instance_to_simulate_next_period.AllNondispatchableGenerators:
                        this_date_planning_renewable_generation_cleared[g,t] = day_ahead_renewable_cleared_for_next_period[g,t]

                print("\nUpdated prices")
                print("Current planning energy prices:")
                print(("%-"+str(DEFAULT_MAX_LABEL_LENGTH)+"s %5s %14s") % ("Bus","Time", "Computed "+options.day_ahead_pricing))
                for t in range(0,options.ruc_every_hours):
                    for bus in ruc_instance_to_simulate_next_period.Buses:
                        print(("%-"+str(DEFAULT_MAX_LABEL_LENGTH)+"s %5d %14.6f") % (bus, t, this_date_planning_energy_prices[bus,t]))

        for ruc_hour in ruc_start_hours:

            #print("DEBUG: Start of for ruc_hour in ruc_start_hours, ruc_hour =", ruc_hour)
            # compute a demand and renewables forecast error for all time periods being simulated
            # today. this is used when creating SCED instances, where it is useful / necessary
            # (which depends on the simulation option) in order to adjust the projected quantities
            # that was originally used in solving the RUC.

            # transfer over the ruc instance to simulate
            if (options.ruc_execution_hour != 0) or first_pass:
                ruc_instance_to_simulate_this_period = ruc_instance_to_simulate_next_period

                # initialize the actual demand and renewables vectors - these will be incrementally
                # updated when new forecasts are released, e.g., when the next-day RUC is computed.
                actual_demand = dict(((b, t), value(ruc_instance_to_simulate_this_period.Demand[b, t]))
                                     for b in ruc_instance_to_simulate_this_period.Buses
                                     for t in ruc_instance_to_simulate_this_period.TimePeriods)
                actual_min_renewables = dict(((g, t), value(ruc_instance_to_simulate_this_period.MinNondispatchablePower[g, t]))
                                             for g in ruc_instance_to_simulate_this_period.AllNondispatchableGenerators
                                             for t in ruc_instance_to_simulate_this_period.TimePeriods)
                actual_max_renewables = dict(((g, t), value(ruc_instance_to_simulate_this_period.MaxNondispatchablePower[g, t]))
                                             for g in ruc_instance_to_simulate_this_period.AllNondispatchableGenerators
                                             for t in ruc_instance_to_simulate_this_period.TimePeriods)

                # establish the stochastic ruc instance for this_period - we use this instance to track,
                # for better or worse, the projected and actual UnitOn states through the day.
                if options.run_deterministic_ruc:
                    deterministic_ruc_instance_for_this_period = deterministic_ruc_instance_for_next_period
                    scenario_tree_for_this_period = scenario_tree_for_next_period
                else:
                    scenario_instances_for_this_period = scenario_instances_for_next_period
                    scenario_tree_for_this_period = scenario_tree_for_next_period

                # now that we've established the stochastic ruc for this_period, null out the one for next_period.
                scenario_instances_for_next_period = None
                scenario_tree_for_next_period = None

            first_pass = False


            #print("DEBUG: After handoff. scenario_tree_for_this_period =", scenario_tree_for_this_period)
            #print("DEBUG: After handoff. scenario_tree_for_next_period =", scenario_tree_for_next_period)
            if options.run_deterministic_ruc:
                print("")
                #print("NOTE: Positive forecast errors indicate projected values higher than actuals")
                demand_forecast_error = {}  # maps (bus,time-period) pairs to an error, defined as forecast minus actual
                for b in deterministic_ruc_instance_for_this_period.Buses:
                    for t in range(1, value(deterministic_ruc_instance_for_this_period.NumTimePeriods)+1):  # TBD - not sure how to option-drive the upper bound on the time period value.
                        demand_forecast_error[b, t] = value(deterministic_ruc_instance_for_this_period.Demand[b, t]) - \
                                                      value(ruc_instance_to_simulate_this_period.Demand[b,t])
                        #print("Demand forecast error for bus=%s at t=%2d: %12.2f" % (b, t, demand_forecast_error[b,t]))

                print("")
                renewables_forecast_error = {}
                # maps (generator,time-period) pairs to an error, defined as forecast minus actual
                for g in deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators:
                    for t in range(1, value(deterministic_ruc_instance_for_this_period.NumTimePeriods)+1):
                        renewables_forecast_error[g,t] = value(deterministic_ruc_instance_for_this_period.MaxNondispatchablePower[g, t]) - \
                                                         value(ruc_instance_to_simulate_this_period.MaxNondispatchablePower[g, t])
                        #print("Renewables forecast error for generator=%s at t=%2d: %12.2f" % (g, t, renewables_forecast_error[g, t]))


            for h in list_hours[ruc_hour]:
                #print("DEBUG: Start of for h in list_hours[ruc_hour]. h =", h)

                print("")
                print(">>>>Simulating hour: " + str(h+1) + " (date: " + str(this_date) + ")")

                if run_sced:

                    # establish the previous sced instance, which is used as the basis for
                    # the initial conditions (T0 state, power generated) for the next
                    # sced instance.

                    if (h == 0) and (this_date == first_date):
                        # there is to prior sced instance - we'll have to take the initial
                        # conditions from the stochastic RUC initial conditions.
                        prior_sced_instance = None
                    else:
                        # take the sced instance from the prior iteration of the by-hour simulation loop.
                        prior_sced_instance = current_sced_instance

                if options.ruc_execution_hour%options.ruc_every_hours > 0:
                    uc_hour = (h-options.ruc_execution_hour%(-options.ruc_every_hours))%24
                else:
                    uc_hour = h

                if this_date != end_date:
                    end_of_ruc = False
                elif uc_hour > 0:
                    end_of_ruc = False
                elif options.ruc_execution_hour == 0:
                    end_of_ruc = False
                else:
                    end_of_ruc = True

                if (h%options.ruc_every_hours == 0) and (options.ruc_execution_hour == 0):
                    ## in this case, we need a SCED first because we ran a RUC before
                    if (first_date == this_date) and (h == 0):
                        is_ruc_hour = False
                    else:
                        is_ruc_hour = True
                elif (h%options.ruc_every_hours == options.ruc_execution_hour%options.ruc_every_hours):
                    is_ruc_hour = True
                else:
                    is_ruc_hour = False

                # run RUC at D-X (except on the last day of the simulation), where X is the
                # user-specified hour at which RUC is to be executed each day.
                if run_ruc \
                    and (not end_of_ruc) \
                    and is_ruc_hour:

                    #print("DEBUG: Running RUC")

                    if (uc_hour > 0 or options.ruc_execution_hour == 0):
                        uc_date = this_date
                        next_uc_date = next_date
                    else:
                        uc_date = next_date
                        next_uc_date = next_next_date

                    start_time = time.time()

                    # to create the RUC for tomorrow, we need to estimate initial conditions
                    # at midnight. we can do this most accurately by solving a sced, taking into account
                    # our current forecast error model. note that there is a bit of a cart-before-the-horse
                    # problem here, as we must estimate the midnight conditions prior to knowing the
                    # unit commitments for the next day (which are necessarily taken as a repeat of the
                    # current day commitments, lacking an alternative). we could in principle compute
                    # updated forecast errors for the next day, but that would seem to be of minimal use
                    # in that the binaries would not be correct - and it is a mouthful to explain. and
                    # probably not realistic.
                    if options.run_deterministic_ruc:
                        scenario_instances_for_this_period = None
                    else:
                        deterministic_ruc_instance_for_this_period = None
                        demand_forecast_error = None
                        renewables_forecast_error = None

                    if options.ruc_execution_hour%options.ruc_every_hours > 0:
                        print("")
                        print("Creating and solving SCED to determine UC initial conditions for date:", uc_date, "hour:", uc_hour)

                        # NOTE: the projected sced probably doesn't have to be run for a full 24 hours - just enough
                        #       to get you to midnight and a few hours beyond (to avoid end-of-horizon effects).
                        projected_sced_instance = create_sced_instance(sced_model, reference_model_module,
                                                                       deterministic_ruc_instance_for_this_period,
                                                                       scenario_instances_for_this_period,
                                                                       scenario_tree_for_this_period,
                                                                       None, None, None,
                                                                       # we're setting up for next_period - it's not here yet!
                                                                       ruc_instance_to_simulate_this_period, prior_sced_instance,
                                                                       actual_demand,
                                                                       demand_forecast_error,
                                                                       actual_min_renewables,
                                                                       actual_max_renewables,
                                                                       renewables_forecast_error,
                                                                       this_date,
                                                                       h%options.ruc_every_hours,
                                                                       options.reserve_factor,
                                                                       options,
                                                                       ## BK -- I'm not sure this was what it was. Don't we just want to
                                                                       ##       consider the next 24 hours or whatever?
                                                                       ## BK -- in the case of shorter ruc_horizons we may not have solutions
                                                                       ##       a full 24 hours out!
                                                                       hours_in_objective=min(24,ruc_horizon-h%options.ruc_every_hours),
                                                                       sced_horizon=min(24,ruc_horizon-h%options.ruc_every_hours),
                                                                       ruc_every_hours=options.ruc_every_hours,
                                                                       initialize_from_ruc=initialize_from_ruc,
                                                                       use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                                       use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
                                                                       )

                        sced_results = call_solver(sced_solver,projected_sced_instance,
                                                   tee=options.output_solver_logs,
                                                   keepfiles=options.keep_solver_files,
                                                   **solve_options[sced_solver])

                        sced_schedule_hour = (23-h+1)%options.ruc_every_hours
                        if sced_results.solution.status.key != "optimal":
                            print("Failed to solve initial condition SCED, writing to file.")
                            # for diagnostic purposes, save the failed SCED instance.
                            infeasible_sced_filename = options.output_directory + os.sep + str(next_date) + os.sep + \
                                                       "failed_initial_condition_sced.lp"
                            lp_writer(projected_sced_instance, infeasible_sced_filename, lambda x: True,
                                      {"symbolic_solver_labels" : True})
                            print("Infeasible SCED instance written to file=" + infeasible_sced_filename)

                            raise RuntimeError("Failed to solve initial condition SCED")
                        projected_sced_instance.solutions.load_from(sced_results)

                    else: ## if this isn't the case, we do the commitment immediate after the hourly SCED
                        print("")
                        print("Drawing UC initial conditions for date:", uc_date, "hour:", uc_hour, "from prior SCED instance.")
                        projected_sced_instance = current_sced_instance
                        sced_schedule_hour = 1

                    if options.bidding:
                        # Xian: solve bidding problem here
                        thermalBid.update_model_params(m_bid,ruc_dispatch_level_current)
                        thermalBid.reset_constraints(m_bid,options.ruc_horizon)

                        price_forecast_dir = '../../prescient/plugins/price_forecasts/date={}_lmp_forecasts.csv'.format(uc_date)
                        cost_curve_store_dir = '../../prescient/plugins/cost_curves/'

                        # solve the bidding model for the first simulation day
                        thermalBid.stochastic_bidding(m_bid,price_forecast_dir,cost_curve_store_dir,uc_date)

                    if options.run_deterministic_ruc:

                        deterministic_ruc_instance_for_next_period, \
                        scenario_tree_for_next_period = create_and_solve_deterministic_ruc(deterministic_ruc_solver,
                                                                                        solve_options,
                                                                                        options,
                                                                                        uc_date,
                                                                                        uc_hour,
                                                                                        next_uc_date,
                                                                                        deterministic_ruc_instance_for_this_period,
                                                                                        scenario_tree_for_this_period,
                                                                                        options.output_ruc_initial_conditions,
                                                                                        projected_sced_instance,
                                                                                        sced_schedule_hour,
                                                                                        ruc_horizon,
                                                                                        use_next_day_in_ruc,
                                                                                        )

                    else:
                        ##SOLVER CALL##
                        if options.solve_with_ph == False:
                            scenario_instances_for_next_period, \
                            scenario_tree_for_next_period = \
                                create_and_solve_stochastic_ruc_via_ef(ef_ruc_solver,
                                                                       solve_options,
                                                                       options,
                                                                       uc_date,
                                                                       uc_hour,
                                                                       next_uc_date,
                                                                       scenario_instances_for_this_period,
                                                                       scenario_tree_for_this_period,
                                                                       options.output_ruc_initial_conditions,
                                                                       projected_sced_instance,
                                                                       sced_schedule_hour,
                                                                       ruc_horizon,
                                                                       use_next_day_in_ruc,
                                                                       )
                        else:
                            scenario_instances_for_next_period, \
                            scenario_tree_for_next_period = \
                                create_and_solve_stochastic_ruc_via_ph(solver,
                                                                       None,
                                                                       options,
                                                                       uc_date,
                                                                       uc_hour,
                                                                       next_uc_date,
                                                                       scenario_instances_for_this_period,
                                                                       scenario_tree_for_this_period,
                                                                       options.output_ruc_initial_conditions,
                                                                       projected_sced_instance,
                                                                       sced_schedule_hour,
                                                                       ruc_horizon,
                                                                       use_next_day_in_ruc,
                                                                       )

                    print("")
                    print("Construction and solve time=%.2f seconds" % (time.time() - start_time))

                    # Xian: printing out the dipatch level of a certain unit
                    print("")
                    print("Getting dispatch levels...\n")

                    if options.run_deterministic_ruc:
                        ruc_dispatch_level_for_next_period = get_dispatch_levels(deterministic_ruc_instance_for_next_period,\
                        options.bidding_generator,verbose = True)

                        # record the ruc signal from this day
                        ruc_schedule_arr[:,dates_to_simulate.index(this_date)+1] = np.array(ruc_dispatch_level_for_next_period[options.bidding_generator]).flatten()[:24]

                    # Xian: pass this schedule to track
                    if options.track_ruc_signal:
                        thermalBid.pass_schedule_to_track_and_solve(m_track_ruc,\
                        ruc_dispatch_level_for_next_period,RT_price=None,\
                        deviation_weight = options.deviation_weight, \
                        ramping_weight = options.ramping_weight,\
                        cost_weight = options.cost_weight)

                        # record the track power output profile
                        # record the track power output profile
                        if options.hybrid_tracking == False:
                            track_gen_pow_ruc = thermalBid.extract_pow_s_s(m_track_ruc, horizon =\
                                                24, verbose = False)
                            thermal_track_gen_pow_ruc = track_gen_pow_ruc
                            thermal_generated_ruc = track_gen_pow_ruc
                        else:
                            track_gen_pow_ruc, thermal_track_gen_pow_ruc,thermal_generated_ruc =\
                                                thermalBid.extract_pow_s_s(m_track_ruc,horizon =\
                                                24, hybrid = True,verbose = False)

                        # record the total power delivered
                        # and thermal power delivered
                        total_power_delivered_arr[:,dates_to_simulate.index(this_date)+1] \
                        = track_gen_pow_ruc[options.bidding_generator]

                        thermal_power_delivered_arr[:,dates_to_simulate.index(this_date)+1] \
                        = thermal_track_gen_pow_ruc[options.bidding_generator]

                        thermal_power_generated_arr[:,dates_to_simulate.index(this_date)+1] \
                        = thermal_generated_ruc[options.bidding_generator]

                        # update the track model
                        thermalBid.update_model_params(m_track_ruc,thermal_generated_ruc,hybrid = options.hybrid_tracking)
                        thermalBid.reset_constraints(m_track_ruc,options.ruc_horizon)

                    DA_ruc_callback(deterministic_ruc_instance_for_next_period, uc_date)

                    if options.compute_market_settlements:
                        day_ahead_prices_for_next_period, \
                        day_ahead_reserve_prices_for_next_period, \
                        day_ahead_thermal_cleared_for_next_period, \
                        day_ahead_reserve_cleared_for_next_period, \
                        day_ahead_renewable_cleared_for_next_period \
                                                              = solve_deterministic_day_ahead_pricing_problem(deterministic_ruc_solver,
                                                                                                              solve_options,
                                                                                                              deterministic_ruc_instance_for_next_period,
                                                                                                              options,
                                                                                                              reference_model_module)
                        report_prices_for_deterministic_ruc(day_ahead_prices_for_next_period, day_ahead_reserve_prices_for_next_period, deterministic_ruc_instance_for_next_period, options)
                        DA_market_callback(uc_date, day_ahead_prices_for_next_period, day_ahead_reserve_prices_for_next_period, day_ahead_thermal_cleared_for_next_period, day_ahead_reserve_prices_for_next_period, day_ahead_renewable_cleared_for_next_period)

                        start_of_planning_period = h-(options.ruc_execution_hour%(-options.ruc_every_hours))
                        ## if this is the case, this plan is still for today.
                        ## if not, we'll put this in the new planning prices dictionary for the next day
                        ## NOTE: this code rests on the assumption that ruc-every-hours > abs(ruc-execution-hour),
                        ##       which is necessary for the current code structure anyways
                        if start_of_planning_period < 24:
                            for t in range(0,options.ruc_every_hours):
                                for b in ruc_instance_to_simulate_this_period.Buses:
                                    this_date_planning_energy_prices[b,t+start_of_planning_period] = \
                                                                            day_ahead_prices_for_next_period[b,t]
                                this_date_planning_reserve_prices[t+start_of_planning_period] = day_ahead_reserve_prices_for_next_period[t]
                                for g in ruc_instance_to_simulate_this_period.ThermalGenerators:
                                    this_date_planning_thermal_generation_cleared[g,t+start_of_planning_period] = \
                                                                                day_ahead_thermal_cleared_for_next_period[g,t]
                                    this_date_planning_thermal_reserve_cleared[g,t+start_of_planning_period] = \
                                                                                day_ahead_reserve_cleared_for_next_period[g,t]
                                for g in ruc_instance_to_simulate_this_period.AllNondispatchableGenerators:
                                    this_date_planning_renewable_generation_cleared[g,t+start_of_planning_period] = \
                                                                                day_ahead_renewable_cleared_for_next_period[g,t]
                            print("\nUpdated prices")
                            print("Current planning energy price:")
                            print(("%-"+str(DEFAULT_MAX_LABEL_LENGTH)+"s %5s %14s") % ("Bus","Time", "Computed "+options.day_ahead_pricing))
                            for t in range(0,options.ruc_every_hours+start_of_planning_period):
                                for bus in ruc_instance_to_simulate_this_period.Buses:
                                    print(("%-"+str(DEFAULT_MAX_LABEL_LENGTH)+"s %5d %14.6f") % (bus, t, this_date_planning_energy_prices[bus,t]))

                    ## in this case, we need to handoff here because "next_period" becomes "this_period" if the execution hour
                    ## in the same loop
                    if options.ruc_execution_hour == 0:
                        if options.run_deterministic_ruc:
                            deterministic_ruc_instance_for_this_period = deterministic_ruc_instance_for_next_period
                            scenario_tree_for_this_period = scenario_tree_for_next_period
                        else:
                            scenario_instances_for_this_period = scenario_instances_for_next_period
                            scenario_tree_for_this_period = scenario_tree_for_next_period

                        scenario_instances_for_next_period = None
                        scenario_tree_for_next_period = None

                    # we assume that the RUC solution time coincides with the availability of any new forecasted quantities
                    # for the next day, i.e., they are "released". these can and should be leveraged in all subsequent
                    # SCED solves, for both prescient and persistent modes. in principle, we could move release of the
                    # forecast into a separate point of the code, e.g., its own hour.

                    ruc_instance_to_simulate_next_period = create_ruc_instance_to_simulate_next_period(ruc_model,
                                                                                                       options,
                                                                                                       uc_date,
                                                                                                       uc_hour,
                                                                                                       next_uc_date,
                                                                                                       )

                    # demand and renewables forecast errors only make sense in the context of deterministic RUC.
                    if options.run_deterministic_ruc:

                        # update the demand and renewables forecast error dictionaries, using the recently released forecasts.
                        print("")
                        print("Updating forecast errors")
                        print("")
                        for b in sorted(deterministic_ruc_instance_for_this_period.Buses):
                            for t in range(1,1+options.ruc_every_hours):
                                demand_forecast_error[b, t+options.ruc_every_hours] = \
                                    value(deterministic_ruc_instance_for_next_period.Demand[b, t]) - \
                                    value(ruc_instance_to_simulate_next_period.Demand[b, t])
                                #print("Demand forecast error for bus=%s at t=%2d: %12.2f"
                                #      % (b, t, demand_forecast_error[b, t+options.ruc_every_hours]))

                        print("")

                        for g in sorted(deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators):
                            for t in range(1, 1+options.ruc_every_hours):
                                renewables_forecast_error[g, t+options.ruc_every_hours] = \
                                    value(deterministic_ruc_instance_for_next_period.MaxNondispatchablePower[g, t]) -\
                                    value(ruc_instance_to_simulate_next_period.MaxNondispatchablePower[g, t])
                                #print("Renewables forecast error for generator=%s at t=%2d: %12.2f"
                                #      % (g, t, renewables_forecast_error[g, t+options.ruc_every_hours]))


                    ## again, in this case, this beginning of the loop needs to be moved before the SCED
                    if options.ruc_execution_hour == 0:
                        ruc_instance_to_simulate_this_period = ruc_instance_to_simulate_next_period
                        # initialize the actual demand and renewables vectors - these will be incrementally
                        # updated when new forecasts are released, e.g., when the next-day RUC is computed.
                        actual_demand = dict(((b, t), value(ruc_instance_to_simulate_this_period.Demand[b, t]))
                                             for b in sorted(ruc_instance_to_simulate_this_period.Buses)
                                             for t in ruc_instance_to_simulate_this_period.TimePeriods)
                        actual_min_renewables = dict(((g, t), value(ruc_instance_to_simulate_this_period.MinNondispatchablePower[g, t]))
                                                     for g in sorted(ruc_instance_to_simulate_this_period.AllNondispatchableGenerators)
                                                     for t in ruc_instance_to_simulate_this_period.TimePeriods)
                        actual_max_renewables = dict(((g, t), value(ruc_instance_to_simulate_this_period.MaxNondispatchablePower[g, t]))
                                                     for g in sorted(ruc_instance_to_simulate_this_period.AllNondispatchableGenerators)
                                                     for t in ruc_instance_to_simulate_this_period.TimePeriods)
                    else:
                        # update the second 24 hours of the current actual demand/renewables vectors
                        for t in range(1,1+options.ruc_every_hours):
                            for b in sorted(ruc_instance_to_simulate_next_period.Buses):
                                actual_demand[b, t+options.ruc_every_hours] = value(ruc_instance_to_simulate_next_period.Demand[b,t])
                            for g in sorted(ruc_instance_to_simulate_next_period.AllNondispatchableGenerators):
                                actual_min_renewables[g, t+options.ruc_every_hours] = \
                                    value(ruc_instance_to_simulate_next_period.MinNondispatchablePower[g, t])
                                actual_max_renewables[g, t+options.ruc_every_hours] = \
                                    value(ruc_instance_to_simulate_next_period.MaxNondispatchablePower[g, t])


                if run_sced:
                    #print("DEBUG: Running SCED")

                    for m in list_minutes:
                        #print("DEBUG: start of for m in list_minutes, m =", m)

                        start_time = time.time()

                        print("")
                        print("Creating SCED optimization instance for day:", str(this_date), "hour:", str(h),
                              "minute:", str(m))

                        # if this is the first hour of the day, we might (often) want to establish initial conditions from
                        # something other than the prior sced instance. these reasons are not for purposes of realism, but
                        # rather pragmatism. for example, there may be discontinuities in the "projected" initial condition
                        # at this time (based on the stochastic RUC schedule being executed during the day) and that of the
                        # actual sced instance.
                        if (h == 0) and (m == list_minutes[0]):
                            if this_date == first_date:
                                # if this is the first date to simulate, we don't have anything
                                # else to go off of when it comes to initial conditions - use those
                                # found in the RUC.
                                initialize_from_ruc = True
                            else:
                                # if we're not in the first day, we should always simulate from the
                                # state of the prior SCED.
                                initialize_from_ruc = False

                        else:
                            initialize_from_ruc = False


                        if options.run_deterministic_ruc:
                            # sced start running

                            current_sced_instance = \
                                create_sced_instance(sced_model, reference_model_module,
                                                     deterministic_ruc_instance_for_this_period, None, scenario_tree_for_this_period,
                                                     deterministic_ruc_instance_for_next_period, None, scenario_tree_for_next_period,
                                                     ruc_instance_to_simulate_this_period, prior_sced_instance,
                                                     actual_demand,
                                                     demand_forecast_error,
                                                     actual_min_renewables,
                                                     actual_max_renewables,
                                                     renewables_forecast_error,
                                                     this_date,
                                                     h%options.ruc_every_hours,
                                                     options.reserve_factor,
                                                     options,
                                                     sced_horizon=options.sced_horizon,
                                                     ruc_every_hours=options.ruc_every_hours,
                                                     initialize_from_ruc=initialize_from_ruc,
                                                     use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                     use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
                                                     )

                        else:
                            current_sced_instance = \
                                create_sced_instance(sced_model, reference_model_module,
                                                     None, scenario_instances_for_this_period, scenario_tree_for_this_period,
                                                     None, scenario_instances_for_next_period, scenario_tree_for_next_period,
                                                     ruc_instance_to_simulate_this_period, prior_sced_instance,
                                                     actual_demand,
                                                     None,
                                                     actual_min_renewables,
                                                     actual_max_renewables,
                                                     None,
                                                     this_date,
                                                     h%options.ruc_every_hours,
                                                     options.reserve_factor,
                                                     options,
                                                     sced_horizon=options.sced_horizon,
                                                     ruc_every_hours=options.ruc_every_hours,
                                                     initialize_from_ruc=initialize_from_ruc,
                                                     use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                     use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
                                                     )

                        # for pretty-printing purposes, compute the maximum bus and generator label lengths.
                        max_bus_label_length = max((len(this_bus) for this_bus in current_sced_instance.Buses))

                        if len(current_sced_instance.TransmissionLines) == 0:
                            max_line_label_length = None
                        else:
                            max_line_label_length = max((len(this_line) for this_line in current_sced_instance.TransmissionLines))

                        if len(current_sced_instance.ThermalGenerators) == 0:
                            max_thermal_generator_label_length = None
                        else:
                            max_thermal_generator_label_length = max((len(this_generator) for this_generator in current_sced_instance.ThermalGenerators))

                        if len(current_sced_instance.AllNondispatchableGenerators) == 0:
                            max_nondispatchable_generator_label_length = None
                        else:
                            max_nondispatchable_generator_label_length = max((len(this_generator) for this_generator in current_sced_instance.AllNondispatchableGenerators))

                        if options.write_sced_instances:
                            current_sced_filename = options.output_directory + os.sep + str(this_date) + \
                                                    os.sep + "sced_hour_" + str(h) + ".lp"
                            lp_writer(current_sced_instance, current_sced_filename, lambda x: True,
                                      {"symbolic_solver_labels" : True})
                            print("SCED instance written to file=" + current_sced_filename)

                        if options.output_sced_initial_conditions:
                            print("")
                            output_sced_initial_condition(current_sced_instance,
                                                          max_thermal_generator_label_length=max_thermal_generator_label_length)

                        if options.output_sced_demands:
                            print("")
                            output_sced_demand(current_sced_instance,
                                               max_bus_label_length=max_bus_label_length)

                        print("")
                        print("Solving SCED instance")
                        infeasibilities_detected_and_corrected = False

                        if options.output_solver_logs:
                            print("")
                            print("------------------------------------------------------------------------------")

                        sced_results = call_solver(sced_solver,
                                                   current_sced_instance,
                                                   tee=options.output_solver_logs,
                                                   keepfiles=options.keep_solver_files,
                                                   **solve_options[sced_solver])

                        sced_runtimes.append(sced_results.solver.time)

                        if options.output_solver_logs:
                            print("")
                            print("------------------------------------------------------------------------------")
                            print("")

                        if sced_results.solution.status.key != "optimal":
                            print("SCED RESULTS STATUS=",sced_results.solution.status.key)
                            print("")
                            print("Failed to solve SCED optimization instance - no feasible solution exists!")
                            print("SCED RESULTS:", sced_results)


                            # for diagnostic purposes, save the failed SCED instance.
                            infeasible_sced_filename = options.output_directory + os.sep + str(this_date) + os.sep + \
                                                       "failed_sced_hour_" + str(h) + ".lp"
                            lp_writer(current_sced_instance, infeasible_sced_filename, lambda x: True,
                                      {"symbolic_solver_labels" : True})
                            print("Infeasible SCED instance written to file=" + infeasible_sced_filename)

                            # create a relaxed SCED instance, for manipulation purposes.
                            if options.run_deterministic_ruc:
                                relaxed_sced_instance = \
                                    create_sced_instance(sced_model, reference_model_module,
                                                         deterministic_ruc_instance_for_this_period, None,
                                                         scenario_tree_for_this_period,
                                                         deterministic_ruc_instance_for_next_period, None,
                                                         scenario_tree_for_next_period,
                                                         ruc_instance_to_simulate_this_period, prior_sced_instance,
                                                         actual_demand,
                                                         demand_forecast_error,
                                                         actual_min_renewables,
                                                         actual_max_renewables,
                                                         renewables_forecast_error,
                                                         this_date,
                                                         h%options.ruc_every_hours,
                                                         options.reserve_factor,
                                                         options,
                                                         sced_horizon=options.sced_horizon,
                                                         ruc_every_hours=options.ruc_every_hours,
                                                         initialize_from_ruc=initialize_from_ruc,
                                                         use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                         use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
                                                         )
                            else:  # changed Mar 2015, GCS: Some arguments were missing.
                                relaxed_sced_instance = \
                                    create_sced_instance(sced_model, reference_model_module,
                                                         None, scenario_instances_for_this_period, scenario_tree_for_this_period,
                                                         None, scenario_instances_for_next_period, scenario_tree_for_next_period,
                                                         ruc_instance_to_simulate_this_period, prior_sced_instance,
                                                         actual_demand,
                                                         None,
                                                         actual_min_renewables,
                                                         actual_max_renewables,
                                                         None,
                                                         this_date,
                                                         h%options.ruc_every_hours,
                                                         options.reserve_factor,
                                                         options,
                                                         sced_horizon=options.sced_horizon,
                                                         ruc_every_hours=options.ruc_every_hours,
                                                         initialize_from_ruc=initialize_from_ruc,
                                                         use_prescient_forecast_error=use_prescient_forecast_error_in_sced,
                                                         use_persistent_forecast_error=use_persistent_forecast_error_in_sced,
                                                         )

                            if options.relax_ramping_if_infeasible:
                                print("")
                                print("Trying to solve SCED optimization instance with relaxed ramping constraints")

                                if True:  # ramp_rate_violations:

                                    relax_iteration = 0
                                    current_inflation_factor = 1.0

                                    while True:

                                        relax_iteration += 1
                                        current_inflation_factor = current_inflation_factor * \
                                                                   (1.0 + options.relax_ramping_factor)

                                        print("Relaxing nominal ramp rates - applying scale factor=" +
                                              str(options.relax_ramping_factor) + "; iteration=" + str(relax_iteration))
                                        relax_sced_ramp_rates(relaxed_sced_instance, options.relax_ramping_factor)
                                        print("Cumulative inflation factor=" + str(current_inflation_factor))

                                        # for diagnostic purposes, save the failed SCED instance.
                                        ramp_relaxed_sced_filename = options.output_directory + os.sep + str(this_date) + \
                                                                     os.sep + "ramp_relaxed_sced_hour_" + str(h) + \
                                                                     "_iter_" + str(relax_iteration) + ".lp"
                                        lp_writer(relaxed_sced_instance, ramp_relaxed_sced_filename, lambda x: True,
                                                  {"symbolic_solver_labels" : True})
                                        print("Ramp-relaxed SCED instance written to file=" + ramp_relaxed_sced_filename)

                                        sced_results = call_solver(sced_solver,
                                                                   relaxed_sced_instance,
                                                                   tee=options.output_solver_logs,
                                                                   keepfiles=options.keep_solver_files,
                                                                   **solve_options[sced_solver])

                                        relaxed_sced_instance.load_from(sced_results)  # load so that we can report later...




                                        if sced_results.solution.status.key != "optimal":
                                            print("Failed to solve ramp-rate-relaxed SCED optimization instance - "
                                                  "still no feasible solution exists!")
                                        else:
                                            break

                                        # the "20" is arbitrary, but the point is that there are situations - if the T0 power is hopelessly messed
                                        # up, where ramp rate relaxations may do nothing...
                                        if relax_iteration >= 20:
                                            raise RuntimeError("Halting - failed to solve ramp-rate-relaxed SCED optimization instance after 20 iterations")

                                    print("Successfully solved ramp-rate-relaxed SCED - proceeding with simulation")
                                    infeasibilities_detected_and_corrected = True

                                else:

                                    raise RuntimeError("Halting - unknown root cause of SCED infeasibility, so no correction can be initiated.")

                            elif options.error_if_infeasible:
                                # TBD - modify this to generate the LP file automatically, for debug purposes.
                                # relaxed_sced_instance.pprint()
                                raise RuntimeError("Halting - option --halt-on-failed-solve enabled")
                            else:
                                # TBD - need to do something with respect to establishing the prior_sced_instance
                                #     - probably need to set the current instance to the previous instance,
                                #       to ensure the prior instance is always feasible, or at least has a solution.
                                print("WARNING: Continuing simulation despite infeasible SCED solution - watch out!")


                        # IMPORTANT: The sced results may yield infeasibilities in the current sced, due to relaxing of
                        #            ramping constraints. It depends on which logic branch above was taken.
                        current_sced_instance.solutions.load_from(sced_results)

                        if options.enable_quick_start_generator_commitment:

                            for g in current_sced_instance.QuickStartGenerators:
                                if current_sced_instance.UnitOn[g, 1]==1:
                                    fixed_quick_start_generators_committed.append(g)

                            # check for load shedding
                            if round_small_values(sum(value(current_sced_instance.posLoadGenerateMismatch[b, 1]) for b in current_sced_instance.Buses))>0:

                                # report solution/load shedding before unfixing Quick Start Generators
                                print("")
                                print("SCED Solution before unfixing Quick Start Generators")
                                print("")

                                if infeasibilities_detected_and_corrected:
                                    output_sced_ramp_violations(current_sced_instance, relaxed_sced_instance)

                                this_sced_without_quick_start_fixed_costs, \
                                this_sced_without_quick_start_variable_costs = report_costs_for_deterministic_sced(current_sced_instance)
                                print("")

                                report_mismatches_for_deterministic_sced(current_sced_instance)
                                print("")

                                report_renewables_curtailment_for_deterministic_sced(current_sced_instance)

                                report_on_off_and_ramps_for_deterministic_sced(current_sced_instance)
                                print("")

                                this_sced_without_quick_start_power_generated=sum(value(current_sced_instance.PowerGenerated[g,1]) for g in current_sced_instance.ThermalGenerators)

                                output_sced_solution(current_sced_instance, max_thermal_generator_label_length=max_thermal_generator_label_length)
                                print("")

                                # re-solve the sced after unfixing Quick Start Generators that have not already been committed
                                print("")
                                print("Re-solving SCED after unfixing Quick Start Generators")
                                for t in sorted(current_sced_instance.TimePeriods):
                                    for g in current_sced_instance.QuickStartGenerators:
                                        if current_sced_instance.UnitOn[g, t]==0:
                                            current_sced_instance.UnitOn[g,t].unfix()
                                load_shed_sced_results = call_solver(sced_solver,
                                                             current_sced_instance,
                                                             tee=options.output_solver_logs,
                                                             keepfiles=options.keep_solver_files,
                                                             **solve_options[sced_solver])
                                current_sced_instance.solutions.load_from(load_shed_sced_results)

                            for g in current_sced_instance.QuickStartGenerators:
                                if current_sced_instance.UnitOn[g, 1]==1:
                                    unfixed_quick_start_generators_committed.append(g)

                        print("Fixing binaries and solving for LMPs")
                        lmp_sced_instance = current_sced_instance.clone()

                        # In case of demand shortfall, the price skyrockets, so we threshold the value.
                        if value(lmp_sced_instance.LoadMismatchPenalty) > options.price_threshold:
                            lmp_sced_instance.LoadMismatchPenalty = options.price_threshold

                        # In case of reserve shortfall, the price skyrockets, so we threshold the value.
                        if value(lmp_sced_instance.ReserveShortfallPenalty) > options.reserve_price_threshold:
                            lmp_sced_instance.ReserveShortfallPenalty = options.reserve_price_threshold

                        reference_model_module.fix_binary_variables(lmp_sced_instance)

                        reference_model_module.define_suffixes(lmp_sced_instance)

                        lmp_sced_results = call_solver(sced_solver,lmp_sced_instance,
                                                       tee=options.output_solver_logs,
                                                       keepfiles=options.keep_solver_files,
                                                       **solve_options[sced_solver])

                        if lmp_sced_results.solution.status.key != "optimal":
                            raise RuntimeError("Failed to solve LMP SCED")

                        lmp_sced_instance.solutions.load_from(lmp_sced_results)

                        print("")
                        print("Results:")
                        print("")


                        RT_sced_callback(current_sced_instance, lmp_sced_instance, this_date, h-1)

                        # Xian: pass this schedule to track
                        if options.track_sced_signal:

                            print("")
                            print("Getting dispatch levels...\n")

                            sced_dispatch_level = get_dispatch_levels(current_sced_instance,\
                            options.bidding_generator,verbose = True)

                            # record the ruc signal from this day
                            sced_schedule_arr[h,dates_to_simulate.index(this_date)] = sced_dispatch_level[options.bidding_generator][0]

                            ## TODO: pass the real-time price into the function here
                            # get lmps in the current planning horizon
                            #get_lmps_for_deterministic_sced(lmp_sced_instance, max_bus_label_length=max_bus_label_length)

                            # slice the ruc dispatch for function calling below
                            ruc_dispatch_level_for_current_sced_track = {options.bidding_generator:\
                            ruc_dispatch_level_current[options.bidding_generator][h:h+options.sced_horizon]}

                            thermalBid.pass_schedule_to_track_and_solve(m_track_sced,\
                            ruc_dispatch_level_for_current_sced_track,\
                            SCED_dispatch = sced_dispatch_level,\
                            deviation_weight = options.deviation_weight, \
                            ramping_weight = options.ramping_weight,\
                            cost_weight = options.cost_weight)

                            # record the track power output profile
                            if options.hybrid_tracking == False:
                                track_gen_pow_sced = thermalBid.extract_pow_s_s(m_track_sced,\
                                horizon = options.sced_horizon, verbose = False)
                                thermal_track_gen_pow_sced = track_gen_pow_sced
                                thermal_generated_sced = track_gen_pow_sced
                            else:
                                # need to extract P_R and P_T
                                # for control power recording and updating the model
                                track_gen_pow_sced, thermal_track_gen_pow_sced, thermal_generated_sced =\
                                thermalBid.extract_pow_s_s(m_track_sced,horizon =\
                                options.sced_horizon, hybrid = True,verbose = False)

                            # record the total power delivered
                            # and thermal power delivered
                            total_power_delivered_arr[h,dates_to_simulate.index(this_date)] \
                            = track_gen_pow_sced[options.bidding_generator][0]

                            thermal_power_delivered_arr[h,dates_to_simulate.index(this_date)] \
                            = thermal_track_gen_pow_sced[options.bidding_generator][0]

                            thermal_power_generated_arr[h,dates_to_simulate.index(this_date)] \
                            = thermal_generated_sced[options.bidding_generator][0]

                            # use the schedule for this step to update the recorder
                            sced_tracker_power_record[options.bidding_generator][:-1] = sced_tracker_power_record[options.bidding_generator][1:]
                            sced_tracker_power_record[options.bidding_generator][-1] = thermal_generated_sced[options.bidding_generator][0]

                            # update the track model
                            thermalBid.update_model_params(m_track_sced,sced_tracker_power_record, hybrid = options.hybrid_tracking)
                            thermalBid.reset_constraints(m_track_sced,options.sced_horizon)

                        # if we had to correct infeasibilities by relaxing constraints, output diagnostics to
                        # help identify where the violations occurred.
                        # NOTE: we are not yet dealing with the scaled ramp limits - just the nominal ones.
                        if infeasibilities_detected_and_corrected:
                            output_sced_ramp_violations(current_sced_instance, relaxed_sced_instance)

                        this_sced_demand = value(current_sced_instance.TotalDemand[1])

                        max_hourly_demand = max(max_hourly_demand, this_sced_demand)

                        this_sced_fixed_costs, this_sced_variable_costs = report_costs_for_deterministic_sced(current_sced_instance)
                        print("")

                        this_sced_power_generated=sum(value(current_sced_instance.PowerGenerated[g,1]) for g in current_sced_instance.ThermalGenerators)

                        this_sced_load_shedding, \
                        this_sced_over_generation, \
                        this_sced_reserve_shortfall, \
                        this_sced_available_reserve, \
                        this_sced_available_quickstart = report_mismatches_for_deterministic_sced(current_sced_instance)

                        if len(current_sced_instance.TransmissionLines) > 0:
                            report_at_limit_lines_for_deterministic_sced(current_sced_instance,
                                                                         max_line_label_length=max_line_label_length)

                        report_lmps_for_deterministic_sced(lmp_sced_instance, max_bus_label_length=max_bus_label_length)

                        this_sced_renewables_available = sum(value(current_sced_instance.MaxNondispatchablePower[g, 1])
                                                             for g in current_sced_instance.AllNondispatchableGenerators)

                        this_sced_renewables_used = sum(value(current_sced_instance.NondispatchablePowerUsed[g, 1])
                                                        for g in current_sced_instance.AllNondispatchableGenerators)
                        this_sced_renewables_curtailment = \
                            report_renewables_curtailment_for_deterministic_sced(current_sced_instance)
                        this_sced_on_offs, this_sced_sum_on_off_ramps, this_sced_sum_nominal_ramps = \
                            report_on_off_and_ramps_for_deterministic_sced(current_sced_instance)

                        curtailments_by_hour.append(this_sced_renewables_curtailment)

                        reserve_requirements_by_hour.append(value(current_sced_instance.ReserveRequirement[1]))

                        reserve_RT_price_by_hour.append(value(lmp_sced_instance.dual[lmp_sced_instance.EnforceReserveRequirements[1]]))

                        load_shedding_by_hour.append(this_sced_load_shedding)
                        over_generation_by_hour.append(this_sced_over_generation)
                        reserve_shortfalls_by_hour.append(this_sced_reserve_shortfall)
                        available_reserves_by_hour.append(this_sced_available_reserve)

                        available_quickstart_by_hour.append(this_sced_available_quickstart)

                        if this_sced_over_generation > 0.0:
                            event_annotations.append((h, 'Over Generation'))
                        if this_sced_load_shedding > 0.0:
                            event_annotations.append((h, 'Load Shedding'))
                        if this_sced_reserve_shortfall > 0.0:
                            event_annotations.append((h, 'Reserve Shortfall'))

                        # 0 demand can happen, in some odd circumstances (not at the ISO level!).
                        if this_sced_demand != 0.0:
                            this_sced_price = (this_sced_fixed_costs + this_sced_variable_costs) / this_sced_demand
                        else:
                            this_sced_price = 0.0

                        # Track difference in costs/power generated with and without quick start generators
                        if options.enable_quick_start_generator_commitment:
                            if round_small_values(sum(value(current_sced_instance.posLoadGenerateMismatch[b, 1]) for b in current_sced_instance.Buses))>0:
                                this_sced_quick_start_additional_costs=round_small_values(this_sced_fixed_costs+this_sced_variable_costs-this_sced_without_quick_start_fixed_costs-this_sced_without_quick_start_variable_costs)
                                this_sced_quick_start_additional_power_generated=this_sced_power_generated-this_sced_without_quick_start_power_generated
                            else:
                                this_sced_quick_start_additional_costs=0.0
                                this_sced_quick_start_additional_power_generated=0.0
                        else:
                            this_sced_quick_start_additional_costs=0.0
                            this_sced_quick_start_additional_power_generated=0.0

                        quick_start_additional_costs_by_hour.append(this_sced_quick_start_additional_costs)
                        quick_start_additional_power_generated_by_hour.append(this_sced_quick_start_additional_power_generated)

                        # Track quick start generators used, if any
                        if options.enable_quick_start_generator_commitment:
                            for g in current_sced_instance.QuickStartGenerators:
                                if g in unfixed_quick_start_generators_committed and g not in fixed_quick_start_generators_committed:
                                    used_as_quick_start[g].append(1)
                                else:
                                    used_as_quick_start[g].append(0)
                            fixed_quick_start_generators_committed=[]
                            unfixed_quick_start_generators_committed=[]


                        ## make this reporting pretty
                        date_time_str_list = [str(this_date), str(h+1)]
                        report_value_list =[ this_sced_fixed_costs + this_sced_variable_costs,
                                             this_sced_fixed_costs,
                                             this_sced_variable_costs,
                                             this_sced_load_shedding,
                                             this_sced_over_generation,
                                             this_sced_reserve_shortfall,
                                             this_sced_renewables_used,
                                             this_sced_renewables_curtailment,
                                             this_sced_demand,
                                             this_sced_price,
                                           ]
                        report_str_list = [ str(round(val,options.output_max_decimal_place)) for val in report_value_list ]

                        # write summary statistics to the hourly CSV file.
                        print(",".join(date_time_str_list+report_str_list), file=csv_hourly_output_file)

                        # # Feb 2015, GCS # #
                        # to plot the demand
                        try:
                            if len(demand_list) == 25:
                                foo
                            demand_list.append(this_sced_demand)
                        except:
                            demand_list = [this_sced_demand]
                        #####################

                        this_date_demand += this_sced_demand
                        this_date_fixed_costs += this_sced_fixed_costs
                        this_date_variable_costs += this_sced_variable_costs
                        this_date_over_generation += this_sced_over_generation
                        this_date_load_shedding += this_sced_load_shedding
                        this_date_reserve_shortfall += this_sced_reserve_shortfall
                        this_date_renewables_available += this_sced_renewables_available
                        this_date_renewables_used += this_sced_renewables_used
                        this_date_renewables_curtailment += this_sced_renewables_curtailment
                        this_date_on_offs += this_sced_on_offs
                        this_date_sum_on_off_ramps += this_sced_sum_on_off_ramps
                        this_date_sum_nominal_ramps += this_sced_sum_nominal_ramps
                        this_date_quick_start_additional_costs += this_sced_quick_start_additional_costs
                        this_date_quick_start_additional_power_generated += this_sced_quick_start_additional_power_generated

                        if options.output_sced_solutions:
                            print("")
                            output_sced_solution(current_sced_instance, max_thermal_generator_label_length=max_thermal_generator_label_length)

                        # Create new row for hourly dataframe, all generators

                        for g in current_sced_instance.ThermalGenerators:
                            observed_thermal_dispatch_levels[g][h] = value(current_sced_instance.PowerGenerated[g, 1])

                        # Xian: add logic to change this dict for tracking power output
                        if options.track_ruc_signal:
                            print('Making changes in observed power output using tracking RUC model.')
                            g = options.bidding_generator
                            observed_thermal_dispatch_levels[g][h] = track_gen_pow_ruc[g][h]

                        elif options.track_sced_signal:
                            print('Making changes in observed power output using tracking SCED model.')
                            g = options.bidding_generator
                            observed_thermal_dispatch_levels[g][h] = track_gen_pow_sced[g][0]

                        for g in current_sced_instance.ThermalGenerators:
                            observed_thermal_headroom_levels[g][h] = max(value(current_sced_instance.MaximumPowerAvailable[g,1]) - observed_thermal_dispatch_levels[g][h],0.0)

                        for g in current_sced_instance.ThermalGenerators:
                            observed_thermal_states[g][h] = value(current_sced_instance.UnitOn[g, 1])

                        for g in current_sced_instance.ThermalGenerators:
                            observed_costs[g][h] = value(current_sced_instance.StartupCost[g,1]
                                                       + current_sced_instance.ShutdownCost[g,1]
                                                       + current_sced_instance.UnitOn[g,1] * current_sced_instance.MinimumProductionCost[g,1] * current_sced_instance.TimePeriodLength
                                                       + current_sced_instance.ProductionCost[g,1]
                                                        )

                        for g in current_sced_instance.AllNondispatchableGenerators:
                            observed_renewables_levels[g][h] = value(current_sced_instance.NondispatchablePowerUsed[g, 1])

                        for g in current_sced_instance.AllNondispatchableGenerators:
                            observed_renewables_curtailment[g][h] = value(current_sced_instance.MaxNondispatchablePower[g, 1]) - value(current_sced_instance.NondispatchablePowerUsed[g, 1])

                        for l in current_sced_instance.TransmissionLines:
                            observed_flow_levels[l][h] = value(current_sced_instance.LinePower[l, 1])

                        for b in current_sced_instance.Buses:
                            if value(current_sced_instance.LoadGenerateMismatch[b, 1]) >= 0.0:
                                observed_bus_mismatches[b][h] = value(current_sced_instance.posLoadGenerateMismatch[b, 1])
                            else:
                                observed_bus_mismatches[b][h] = -1.0 * value(current_sced_instance.negLoadGenerateMismatch[b, 1])

                        for b in current_sced_instance.Buses:
                            observed_bus_LMPs[b][h] = value(lmp_sced_instance.dual[lmp_sced_instance.PowerBalance[b, 1]])

                        for s in current_sced_instance.Storage:
                            storage_input_dispatchlevelsdict[s][h] = \
                                np.array([value(current_sced_instance.PowerInputStorage[s, 1])])

                        for s in current_sced_instance.Storage:
                            storage_output_dispatchlevelsdict[s][h] = \
                                np.array([value(current_sced_instance.PowerOutputStorage[s, 1])])

                        for s in current_sced_instance.Storage:
                            storage_soc_dispatchlevelsdict[s][h] = \
                                np.array([value(current_sced_instance.SocStorage[s, 1])])
                            print("")
                            print("Current State-Of-Charge Value: ", value(current_sced_instance.SocStorage[s,1]))
                            print("SOC Value at T0:", value(current_sced_instance.StorageSocOnT0[s]))
                        print("")
                        print(("Construction and solve time=%.2f seconds" % (time.time() - start_time)))

        this_date_renewables_penetration_rate = (float(this_date_renewables_used) / float(this_date_demand) * 100.0)

        this_date_average_price = (this_date_fixed_costs + this_date_variable_costs) / this_date_demand

        cumulative_demand += this_date_demand
        cumulative_renewables_used += this_date_renewables_used

        overall_renewables_penetration_rate = (float(cumulative_renewables_used) / float(cumulative_demand) * 100.0)

        cumulative_average_price = (total_overall_fixed_costs + total_overall_generation_costs) / cumulative_demand

        ## TODO: the below logic should be fixed for intra-day RUC
        ##       if there is no intra-day RUC, then this_period == today
        ## NOTE: we need to start pulling out some of this data after its computed,
        ##       like any data that compute_market_settlements uses, like reserve_provided,
        ##       thermal_gen_cleared, thermal_reserve_cleared, price_DA (or price_RUC),
        ##       and r_price_DA. While doing this we should also add hooks into the new
        ##       ancillary service model, i.e., to get prices for these ancillary services
        if options.compute_market_settlements:

            thermal_gen_cleared_DA, \
            thermal_gen_revenue, \
            thermal_reserve_cleared_DA, \
            thermal_reserve_revenue, \
            renewable_gen_cleared_DA, \
            renewable_gen_revenue = compute_market_settlements(deterministic_ruc_instance_for_this_period,
                                                               this_date_planning_thermal_generation_cleared,
                                                               this_date_planning_thermal_reserve_cleared,
                                                               this_date_planning_renewable_generation_cleared,
                                                               this_date_planning_energy_prices,
                                                               this_date_planning_reserve_prices,
                                                               observed_thermal_dispatch_levels,
                                                               observed_thermal_headroom_levels,
                                                               observed_renewables_levels,
                                                               observed_bus_LMPs,
                                                               reserve_RT_price_by_hour,
                                                               )

            this_date_thermal_energy_payments = sum(thermal_gen_revenue[g,t] for g in deterministic_ruc_instance_for_this_period.ThermalGenerators for t in range(0,24))
            this_date_renewable_energy_payments = sum(renewable_gen_revenue[g,t] for g in deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators for t in range(0,24))

            this_date_energy_payments =  this_date_thermal_energy_payments+this_date_renewable_energy_payments

            this_date_reserve_payments = sum(thermal_reserve_revenue[g,t] for g in deterministic_ruc_instance_for_this_period.ThermalGenerators for t in range(0,24))

            ## uplift payments ensure
            ## a generator is covered for
            ## its expenses each day
            thermal_uplift = {}
            renewable_uplift = {}

            ## NOTE: we "pay" the uplift in period 24
            uplift_hr = 23
            for g in deterministic_ruc_instance_for_this_period.ThermalGenerators:
                profit = 0.
                for t in range(0,24):
                    profit += (thermal_gen_revenue[g,t] + thermal_reserve_revenue[g,t] - observed_costs[g][t])
                    thermal_uplift[g,t] = 0.
                if profit < 0.:
                    thermal_uplift[g,uplift_hr] = -profit

            ## TODO: this needs to be re-thought. We shouldn't pay for generators that
            ##       dispatched themselves away from the DA solution.
            for g in deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators:
                profit = 0.
                for t in range(0,24):
                    #profit += (renewable_gen_revenue[g,t])
                    renewable_uplift[g,t] = 0.
                #if profit < 0.:
                #    renewable_uplift[g,uplift_hr] = -profit

            this_date_thermal_uplift = sum(thermal_uplift[g,uplift_hr] for g in deterministic_ruc_instance_for_this_period.ThermalGenerators)
            this_date_renewable_uplift = sum(renewable_uplift[g,uplift_hr] for g in deterministic_ruc_instance_for_this_period.AllNondispatchableGenerators)

            this_date_uplift_payments = this_date_thermal_uplift + this_date_renewable_uplift

            this_date_total_payout = this_date_energy_payments + this_date_reserve_payments + this_date_uplift_payments

        new_quickstart_df_entries = []
        for h in range (0,24):
            for g in sorted(current_sced_instance.ThermalGenerators):
                if g in used_as_quick_start:
                    this_gen_used_as_quick_start=used_as_quick_start[g][h]
                    this_gen_quick_start_dispatch=observed_thermal_dispatch_levels[g][h]
                    new_quickstart_df_entries.append({'Date':this_date,
                                                      'Hour':h+1,
                                                      'Generator':g,
                                                      'Used as quickstart':this_gen_used_as_quick_start,
                                                      'Dispatch level of quick start generator':this_gen_quick_start_dispatch})
        quickstart_summary_df = pd.concat([quickstart_summary_df, pd.DataFrame.from_records(new_quickstart_df_entries)], sort=True)

        new_thermal_generator_dispatch_entries = []
        for h in range(0, 24):
            for g in sorted(current_sced_instance.ThermalGenerators):
                new_thermal_generator_dispatch_entries.append({'Date':this_date,
                                                               'Hour':h+1,
                                                               'Generator':g,
                                                               'Dispatch':observed_thermal_dispatch_levels[g][h],
                                                               'Dispatch DA': thermal_gen_cleared_DA[g,h] if options.compute_market_settlements else np.nan,
                                                               'Headroom':observed_thermal_headroom_levels[g][h],
                                                               'Unit State': observed_thermal_states[g][h],
                                                               'Unit Cost': observed_costs[g][h],
                                                               'Unit Market Revenue': thermal_gen_revenue[g,h]+thermal_reserve_revenue[g,h] if options.compute_market_settlements else np.nan,
                                                               'Unit Uplift Payment': thermal_uplift[g,h] if options.compute_market_settlements else np.nan,
                                                               })
        thermal_generator_dispatch_df = pd.concat([thermal_generator_dispatch_df, pd.DataFrame.from_records(new_thermal_generator_dispatch_entries)], sort=True)

        new_renewables_production_entries = []
        for h in range(0, 24):
            for g in sorted(current_sced_instance.AllNondispatchableGenerators):
                new_renewables_production_entries.append({'Date':this_date,
                                                          'Hour':h+1,
                                                          'Generator':g,
                                                          'Output':observed_renewables_levels[g][h],
                                                          'Output DA':renewable_gen_cleared_DA[g,h] if options.compute_market_settlements else np.nan,
                                                          'Curtailment':observed_renewables_curtailment[g][h],
                                                          'Unit Market Revenue': renewable_gen_revenue[g,h] if options.compute_market_settlements else np.nan,
                                                          'Unit Uplift Payment': renewable_uplift[g,h] if options.compute_market_settlements else np.nan,
                                                          })
        renewables_production_df = pd.concat([renewables_production_df, pd.DataFrame.from_records(new_renewables_production_entries)], sort=True)

        new_line_entries = []
        for h in range(0, 24):
            for l in sorted(current_sced_instance.TransmissionLines):
                new_line_entries.append({'Date':this_date,
                                         'Hour':h+1,
                                         'Line':l,
                                         'Flow':observed_flow_levels[l][h]})
        line_df = pd.concat([line_df, pd.DataFrame.from_records(new_line_entries)], sort=True)

        new_bus_entries = []
        for h in range(0, 24):
            for b in sorted(current_sced_instance.Buses):
                this_mismatch = observed_bus_mismatches[b][h]
                if this_mismatch >= 0.0:
                    shortfall = this_mismatch
                    overgeneration = 0.0
                else:
                    shortfall = 0.0
                    overgeneration = -1.0 * this_mismatch
                new_bus_entries.append({'Date':this_date,
                                        'Hour':h+1,
                                        'Bus':b,
                                        'Shortfall':shortfall,
                                        'Overgeneration':overgeneration,
                                        'LMP':observed_bus_LMPs[b][h],
                                        'LMP DA':this_date_planning_energy_prices[b,h] if options.compute_market_settlements else np.nan})
        bus_df = pd.concat([bus_df, pd.DataFrame.from_records(new_bus_entries)], sort=True)

        new_hourly_gen_summary_entries = []
        for h in range(0, 24):
            new_hourly_gen_summary_entries.append({'Date':this_date,
                                                   'Hour':h+1,
                                                   'Load shedding':load_shedding_by_hour[h],
                                                   'Reserve shortfall':reserve_shortfalls_by_hour[h],
                                                   'Available reserves':available_reserves_by_hour[h],
                                                   'Over generation':over_generation_by_hour[h],
                                                   'Reserve Price DA':this_date_planning_reserve_prices[h] if options.compute_market_settlements else np.nan,
                                                   'Reserve Price RT':reserve_RT_price_by_hour[h]})
        hourly_gen_summary_df = pd.concat([hourly_gen_summary_df, pd.DataFrame.from_records(new_hourly_gen_summary_entries)], sort=True)

        runtime_df = runtime_df.append({'Date':this_date,
                                        'Hour':-1,
                                        'Type':'RUC',
                                        'Solve Time':0.0},
                                       ignore_index = True)

        for h in range(0, 24):
            runtime_df = runtime_df.append({'Date':this_date,
                                            'Hour':h+1,
                                            'Type':'SCED',
                                            'Solve Time':sced_runtimes[h]},
                                           ignore_index = True)

        # summarize daily costs / statistics
        print("")
        print("Date %s total demand:                                 %12.2f" % (str(this_date), this_date_demand))
        print("Date %s total renewables available:                   %12.2f" % (str(this_date), this_date_renewables_available))
        print("Date %s total renewables used:                        %12.2f" % (str(this_date), this_date_renewables_used))
        print("Date %s renewables penetration rate:                  %12.2f" % (str(this_date), this_date_renewables_penetration_rate))
        print("Date %s average price:                                %12.6f" % (str(this_date), this_date_average_price))

        print("")

        print("Date %s total fixed costs:                            %12.2f" % (str(this_date), this_date_fixed_costs))
        print("Date %s total generation costs:                       %12.2f" % (str(this_date), this_date_variable_costs))
        print("Date %s total load shedding:                          %12.2f" % (str(this_date), this_date_load_shedding))
        print("Date %s total over generation:                        %12.2f" % (str(this_date), this_date_over_generation))
        print("Date %s total reserve shortfall                       %12.2f" % (str(this_date), this_date_reserve_shortfall))
        print("Date %s total renewables curtailment:                 %12.2f" % (str(this_date), this_date_renewables_curtailment))
        print("Date %s total on/offs:                                %12d"   % (str(this_date), this_date_on_offs))
        print("Date %s total sum on/off ramps:                       %12.2f" % (str(this_date), this_date_sum_on_off_ramps))
        print("Date %s total sum nominal ramps:                      %12.2f" % (str(this_date), this_date_sum_nominal_ramps))
        print("Date %s total quick start additional costs:           %12.2f" % (str(this_date), this_date_quick_start_additional_costs))
        print("Date %s total quick start additional power generated: %12.2f" % (str(this_date), this_date_quick_start_additional_power_generated))

        if options.compute_market_settlements:
            print("")

            print("Date %s total thermal generator energy payments       %12.2f" % (str(this_date), this_date_thermal_energy_payments))
            print("Date %s total renewable generator energy payments     %12.2f" % (str(this_date), this_date_renewable_energy_payments))
            print("Date %s total energy payments                         %12.2f" % (str(this_date), this_date_energy_payments))

            print("Date %s total thermal generator reserve payments      %12.2f" % (str(this_date), this_date_reserve_payments))

            print("Date %s total thermal generator uplift payments       %12.2f" % (str(this_date), this_date_thermal_uplift))
            print("Date %s total renewable generator uplift payments     %12.2f" % (str(this_date), this_date_renewable_uplift))
            print("Date %s total uplift payments                         %12.2f" % (str(this_date), this_date_uplift_payments))

            print("Date %s total payments                                %12.2f" % (str(this_date), this_date_total_payout))
            print("Date %s average payments                              %12.2f" % (str(this_date), this_date_total_payout/this_date_demand))

        # update overall simulation costs / statistics
        total_overall_fixed_costs += this_date_fixed_costs
        total_overall_generation_costs += this_date_variable_costs
        total_overall_load_shedding += this_date_load_shedding
        total_overall_over_generation += this_date_over_generation
        total_overall_reserve_shortfall += this_date_reserve_shortfall

        total_overall_renewables_curtailment += this_date_renewables_curtailment
        total_on_offs += this_date_on_offs
        total_sum_on_off_ramps += this_date_sum_on_off_ramps
        total_sum_nominal_ramps += this_date_sum_nominal_ramps
        total_quick_start_additional_costs += this_date_quick_start_additional_costs
        total_quick_start_additional_power_generated += this_date_quick_start_additional_power_generated

        if options.compute_market_settlements:

            total_thermal_energy_payments += this_date_thermal_energy_payments
            total_renewable_energy_payments += this_date_renewable_energy_payments
            total_energy_payments += this_date_energy_payments

            total_reserve_payments += this_date_reserve_payments

            total_thermal_uplift_payments += this_date_thermal_uplift
            total_renewable_uplift_payments += this_date_renewable_uplift
            total_uplift_payments += this_date_uplift_payments

            total_payments += this_date_total_payout

            daily_thermal_energy_payments.append(this_date_thermal_energy_payments)
            daily_renewable_energy_payments.append(this_date_renewable_energy_payments)
            daily_energy_payments.append(this_date_energy_payments)

            daily_reserve_payments.append(this_date_reserve_payments)

            daily_thermal_uplift_payments.append(this_date_thermal_uplift)
            daily_renewable_uplift_payments.append(this_date_renewable_uplift)
            daily_uplift_payments.append(this_date_uplift_payments)

            daily_total_payments.append(this_date_total_payout)

        daily_total_costs.append(this_date_fixed_costs+this_date_variable_costs)
        daily_fixed_costs.append(this_date_fixed_costs)
        daily_generation_costs.append(this_date_variable_costs)
        daily_load_shedding.append(this_date_load_shedding)
        daily_over_generation.append(this_date_over_generation)
        daily_reserve_shortfall.append(this_date_reserve_shortfall)

        daily_renewables_available.append(this_date_renewables_available)
        daily_renewables_used.append(this_date_renewables_used)
        daily_renewables_curtailment.append(this_date_renewables_curtailment)
        daily_on_offs.append(this_date_on_offs)
        daily_sum_on_off_ramps.append(this_date_sum_on_off_ramps)
        daily_sum_nominal_ramps.append(this_date_sum_nominal_ramps)
        daily_quick_start_additional_costs.append(this_date_quick_start_additional_costs)
        daily_quick_start_additional_power_generated.append(this_date_quick_start_additional_power_generated)

        daily_average_price.append(this_date_average_price)

        daily_demand.append(this_date_demand)

        #print out the dataframe/generator detail

        if len(offline_generators) > 0:
            print("Generators knocked offline today:", offline_generators)
            reset_offline_elements(sced_inst)
            for g in offline_generators:
                sced_inst.UnitOnT0[g] = int(round(ruc_inst.UnitOn[g, 1]))
                sced_inst.PowerGeneratedT0[g] = ruc_inst.PowerGenerated[g, 1]
            offline_generators = []


        # we always generate plots - but we optionally display them in the simulation loop.
        generator_dispatch_levels = {}

        # we need a map between generator name and type, where the latter is a single
        # character, like 'C' or 'N'.
        generator_types = {}

        if not hasattr(current_sced_instance, "ThermalGeneratorType"):
            print("***SCED instance does not have \"ThermalGeneratorType\" attribute - "
                  "required for stack graph plot generation")
            sys.exit(1)

        if not hasattr(current_sced_instance, "NondispatchableGeneratorType"):
            print("***SCED instance does not have \"NondispatchableGeneratorType\" attribute - "
                  "required for stack graph plot generation")
            sys.exit(1)

        for g in current_sced_instance.ThermalGenerators:
            generator_dispatch_levels[g] = observed_thermal_dispatch_levels[g]
            generator_types[g] = current_sced_instance.ThermalGeneratorType[g]

        for g in current_sced_instance.AllNondispatchableGenerators:
            generator_dispatch_levels[g] = observed_renewables_levels[g]
            generator_types[g] = current_sced_instance.NondispatchableGeneratorType[g]

        plot_peak_demand = thermal_fleet_capacity
        if options.plot_peak_demand > 0.0:
            plot_peak_demand = options.plot_peak_demand

        summary_dict = {'Date':this_date,
                        'Demand':this_date_demand,
                        'Renewables available':this_date_renewables_available,
                        'Renewables used':this_date_renewables_used,
                        'Renewables penetration rate':this_date_renewables_penetration_rate,
                        'Average price':this_date_average_price,
                        'Fixed costs':this_date_fixed_costs,
                        'Generation costs':this_date_variable_costs,
                        'Load shedding':this_date_load_shedding,
                        'Over generation':this_date_over_generation,
                        'Reserve shortfall':this_date_reserve_shortfall,
                        'Renewables curtailment':this_date_renewables_curtailment,
                        'Number on/offs':this_date_on_offs,
                        'Sum on/off ramps':this_date_sum_on_off_ramps,
                        'Sum nominal ramps':this_date_sum_nominal_ramps}

        if options.compute_market_settlements:
            summary_dict.update( {'Renewables energy payments':this_date_renewable_energy_payments,
                                  'Renewables uplift payments':this_date_renewable_uplift,
                                  'Thermal energy payments':this_date_thermal_energy_payments,
                                  'Thermal uplift payments':this_date_thermal_uplift,
                                  'Total energy payments':this_date_energy_payments,
                                  'Total uplift payments':this_date_uplift_payments,
                                  'Total reserve payments':this_date_reserve_payments,
                                  'Total payments':this_date_total_payout,
                                  'Average payments':this_date_total_payout/this_date_demand,
                                  } )


        daily_summary_df = daily_summary_df.append(summary_dict, ignore_index=True)



        graphutils.generate_stack_graph(plot_peak_demand,  # for scale of the plot
                                        generator_types,
                                        generator_dispatch_levels,
                                        reserve_requirements_by_hour,
                                        this_date,
                                        curtailments_by_hour,  # is not used, Feb 2015
                                        load_shedding_by_hour,
                                        reserve_shortfalls_by_hour,
                                        available_reserves_by_hour,
                                        available_quickstart_by_hour,
                                        over_generation_by_hour,
                                        max_hourly_demand,
                                        quick_start_additional_power_generated_by_hour,
                                        annotations=event_annotations,
                                        display_plot=options.display_plots,
                                        show_plot_legend=(not options.disable_plot_legend),
                                        savetofile=True,
                                        output_directory=os.path.join(options.output_directory, "plots"),
                                        plot_individual_generators=options.plot_individual_generators,
                                        renewables_penetration_rate=this_date_renewables_penetration_rate,
                                        fixed_costs=this_date_fixed_costs,
                                        variable_costs=this_date_variable_costs,
                                        demand=demand_list)  # Feb 2015, GCS: To plot the demand

        if len(storage_soc_dispatchlevelsdict) > 0:
            storagegraphutils.generate_storage_graph(storage_input_dispatchlevelsdict,
                                                     storage_output_dispatchlevelsdict,
                                                     storage_soc_dispatchlevelsdict,
                                                     this_date,
                                                     save_to_file=True,
                                                     display_plot=options.display_plots,
                                                     plot_individual_generators=False,
                                                     output_directory=os.path.join(options.output_directory, "plots"))



    print("")
    print("Simulation complete!")




    print("")
    print("Total demand:                        %12.2f" % (cumulative_demand))
    print("")
    print("Total fixed costs:                   %12.2f" % (total_overall_fixed_costs))
    print("Total generation costs:              %12.2f" % (total_overall_generation_costs))
    print("Total costs:                         %12.2f" % (total_overall_fixed_costs + total_overall_generation_costs))
    print("")
    if options.compute_market_settlements:
        print("Total energy payments:               %12.2f" % (total_energy_payments))
        print("Total reserve payments:              %12.2f" % (total_reserve_payments))
        print("Total uplift payments:               %12.2f" % (total_uplift_payments))
        print("Total payments:                      %12.2f" % (total_payments))
        print("")
        print("Average payments:                    %12.2f" % (total_payments/cumulative_demand))
        print("")
    print("Total load shedding:                 %12.2f" % (total_overall_load_shedding))
    print("Total over generation:               %12.2f" % (total_overall_over_generation))
    print("Total reserve shortfall:             %12.2f" % (total_overall_reserve_shortfall))
    print("")
    print("Total renewables curtailment:        %12.2f" % (total_overall_renewables_curtailment))
    print("")
    print("Total on/offs:                       %12d"   % (total_on_offs))
    print("Total sum on/off ramps:              %12.2f" % (total_sum_on_off_ramps))
    print("Total sum nominal ramps:             %12.2f" % (total_sum_nominal_ramps))
    print("")
    print("Total quick start additional costs   %12.2f" % (total_quick_start_additional_costs))
    print("Total quick start additional power   %12.2f" % (total_quick_start_additional_power_generated))
    print("")
    print("Maximum observed demand:             %12.2f" % (max_hourly_demand))
    print("")
    print("Overall renewables penetration rate: %12.2f" % (overall_renewables_penetration_rate))
    print("")
    print("Cumulative average price:            %12.6f" % (cumulative_average_price))

    overall_simulation_dict = {'Total demand':cumulative_demand,
                               'Total fixed costs':total_overall_fixed_costs,
                               'Total generation costs':total_overall_generation_costs,
                               'Total costs':total_overall_fixed_costs + total_overall_generation_costs,
                               'Total load shedding':total_overall_load_shedding,
                               'Total over generation':total_overall_over_generation,
                               'Total reserve shortfall':total_overall_reserve_shortfall,
                               'Total renewables curtialment':total_overall_renewables_curtailment,
                               'Total on/offs':total_on_offs,
                               'Total sum on/off ramps':total_sum_on_off_ramps,
                               'Total sum nominal ramps':total_sum_nominal_ramps,
                               'Maximum observed demand':max_hourly_demand,
                               'Overall renewables penetration rate':overall_renewables_penetration_rate,
                               'Cumulative average price':cumulative_average_price}

    if options.compute_market_settlements:
        overall_simulation_dict.update( {'Total energy payments': total_energy_payments,
                                         'Total reserve payments': total_reserve_payments,
                                         'Total uplift payments': total_uplift_payments,
                                         'Total payments': total_payments,
                                         'Cumalative average payments': total_payments/cumulative_demand,
                                         } )

    overall_simulation_output_df = overall_simulation_output_df.append(overall_simulation_dict,ignore_index=True)

    #Create csv files for dataframes

    daily_summary_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'Daily_summary.csv'), index = False)

    options_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'Options.csv'), index = False)

    if options.track_sced_signal:
        thermal_generator_dispatch_df.round(options.output_max_decimal_place).to_csv(
                                                os.path.join(options.output_directory, 'bidding={}_track_sced_thermal_detail.csv'.format(options.bidding)), index = False)
    elif options.track_ruc_signal:
        thermal_generator_dispatch_df.round(options.output_max_decimal_place).to_csv(
                                                os.path.join(options.output_directory, 'bidding={}_track_ruc_thermal_detail.csv'.format(options.bidding)), index = False)
    else:
        thermal_generator_dispatch_df.round(options.output_max_decimal_place).to_csv(
                                                os.path.join(options.output_directory, 'bidding={}_thermal_detail.csv'.format(options.bidding)), index = False)

    renewables_production_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'renewables_detail.csv'), index = False)

    line_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'line_detail.csv'), index = False)

    bus_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'bidding={}_bus_detail.csv'.format(options.bidding)), index = False)

    overall_simulation_output_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'Overall_simulation_output.csv'), index = False)

    quickstart_summary_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'Quickstart_summary.csv'), index = False)

    hourly_gen_summary_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'Hourly_gen_summary.csv'), index = False)

    runtime_df.round(options.output_max_decimal_place).to_csv(
                                            os.path.join(options.output_directory, 'runtimes.csv'), index = False)

    graphutils.generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs,
                                           daily_load_shedding, daily_over_generation,
                                           daily_reserve_shortfall,
                                           daily_renewables_curtailment,
                                           display_plot=options.display_plots,
                                           save_to_file=True,
                                           output_directory=os.path.join(options.output_directory, "plots"))

    # Xian: record the RUC schedule and SCED schedule
    np.savetxt(os.path.join(options.output_directory, 'bidding={}_{}_RUC_schedule.csv'.format(options.bidding,options.bidding_generator)),\
    ruc_schedule_arr,delimiter = ',',fmt = "% .2f")
    np.savetxt(os.path.join(options.output_directory, 'bidding={}_{}_SCED_schedule.csv'.format(options.bidding,options.bidding_generator)),\
    sced_schedule_arr,delimiter = ',',fmt = "% .2f")

    np.savetxt(os.path.join(options.output_directory, 'bidding={}_{}_total_power_output.csv'.format(options.bidding,options.bidding_generator)),\
    total_power_delivered_arr,delimiter = ',',fmt = "% .2f")
    np.savetxt(os.path.join(options.output_directory, 'bidding={}_{}_thermal_power_output.csv'.format(options.bidding,options.bidding_generator)),\
    thermal_power_delivered_arr,delimiter = ',',fmt = "% .2f")
    np.savetxt(os.path.join(options.output_directory, 'bidding={}_{}_thermal_power_generated.csv'.format(options.bidding,options.bidding_generator)),\
    thermal_power_generated_arr,delimiter = ',',fmt = "% .2f")

    simulation_end_time = time.time()
    print("")
    print(("Total simulation run time=%.2f seconds" % (simulation_end_time - simulation_start_time)))

    # create a movie from the set of individual daily png files
    pyutilib.services.register_executable("ffmpeg")
    ffmpeg_executable = pyutilib.services.registered_executable("ffmpeg")
    if ffmpeg_executable == None:
        print("The executable ffmpeg is not installed - could not create movie of stack graphs")
    else:
        movie_filename = os.path.join(options.output_directory, "plots", "stackgraph_movie.mp4")
        if os.path.exists(movie_filename):
            os.remove(movie_filename)
        if os.path.exists("out.mp4"):
            os.remove("out.mp4")

        execution_string = "%s -r 1/2 -pattern_type glob -i " % str(ffmpeg_executable.get_path())
        execution_string += "'" + str(os.path.join(options.output_directory, "plots")) + os.sep + "stackgraph*.png" + "'"
        execution_string += " out.mp4"
        os.system(execution_string)
        shutil.move("out.mp4",movie_filename)

        print("Stackgraph movie written to file=" + movie_filename)

    ###################################################################
    # # output the daily summary statistics to a CSV file           # #
    ###################################################################

    ## BK -- Isn't this handled above? On OSX this will overwrite the file print above
    '''
    csv_daily_output_filename = os.path.join(options.output_directory, "daily_summary.csv")
    csv_daily_output_file = open(csv_daily_output_filename, "w")
    sim_dates = [this_date for this_date in dates_to_simulate]
    print("Date", ",", "TotalCosts", ",", "FixedCosts", ",", "VariableCosts", ",", "LoadShedding", ",",
          "OverGeneration", ",", "ReserveShortfall", ",", "RenewablesAvailable", ",", "RenewablesUsed", ",", "RenewablesCurtailed", ",",
          "Demand", ",", "AveragePrice", ",", "OnOffs", ",", "SumOnOffRamps", ",", "SumNominalRamps",
          file=csv_daily_output_file)
    for i in range(0,len(sim_dates)):
        this_date = sim_dates[i]
        print(this_date, ",", daily_total_costs[i], ",", daily_fixed_costs[i], ",", daily_generation_costs[i], ",",
              daily_load_shedding[i], ",", daily_over_generation[i], ",", daily_reserve_shortfall[i], ",",
              daily_renewables_available[i], ",", daily_renewables_used[i], ",", daily_renewables_curtailment[i], ",", daily_demand[i], ",",
              daily_average_price[i], ",", daily_on_offs[i], ",", daily_sum_on_off_ramps[i], ",",
              daily_sum_nominal_ramps[i], file=csv_daily_output_file)
    csv_daily_output_file.close()
    print("")
    print("CSV daily summary written to file=" + csv_daily_output_filename)
    '''

    #########################################
    # # close the hourly summary CSV file # #
    #########################################

    csv_hourly_output_file.close()
    print("")
    print("CSV hourly summary written to file=" + csv_hourly_output_filename)


###############################################################################
#########################        END PRESCIENT      ###########################
###############################################################################


###############################################################################
#########################     PRESCINT MAIN    ###############################
###############################################################################

def main_prescient(options):

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
            ans = simulate(options)
        else:
            errmsg = None
            try:
                ans = simulate(options)
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
            except ConverterError:
                err = sys.exc_info()[1]
                errmsg = 'CONVERSION ERROR: %s' % err
            except RuntimeError:
                err = sys.exc_info()[1]
                errmsg = 'RUN-TIME ERROR: %s' % err
            except pyutilib.common.ApplicationError:
                err = sys.exc_info()[1]
                errmsg = 'APPLICATION ERROR: %s' % err
            except Exception:
                err = sys.exc_info()[1]
                errmsg = 'UNKNOWN ERROR: %s' % err
                traceback.print_exc()

            if errmsg is not None:
                sys.stderr.write(errmsg+'\n')

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
