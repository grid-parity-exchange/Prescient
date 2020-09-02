from pyomo.environ import value

def _get_ruc_data(instance):
    num_time_periods = 24

    thermal_gen_cleared_RUC = {}
    thermal_reserve_cleared_RUC = {}
    renewable_gen_cleared_RUC = {}

    for t in range(0,num_time_periods):
        for g in instance.ThermalGenerators:
            thermal_gen_cleared_RUC[g,t] = value(instance.PowerGenerated[g,t+1])
            thermal_reserve_cleared_RUC[g,t] = value(instance.ReserveProvided[g,t+1])
        for g in instance.AllNondispatchableGenerators:
            renewable_gen_cleared_RUC[g,t] = value(instance.NondispatchablePowerUsed[g,t+1])

    return thermal_gen_cleared_RUC, thermal_reserve_cleared_RUC, renewable_gen_cleared_RUC

def _get_sced_data(instance, lmp_instance):
    observed_thermal_dispatch_levels = {}
    observed_thermal_states = {}
    observed_costs = {}
    observed_renewables_levels = {}
    observed_bus_mismatches = {}
    observed_bus_LMPs = {}

    for g in instance.ThermalGenerators:
        observed_thermal_dispatch_levels[g] = value(instance.PowerGenerated[g, 1])

    for g in instance.ThermalGenerators:
        observed_thermal_states[g] = value(instance.UnitOn[g, 1])

    for g in instance.ThermalGenerators:
        observed_costs[g] = value(instance.StartupCost[g,1]
                                   + instance.ShutdownCost[g,1]
                                   + instance.UnitOn[g,1] * instance.MinimumProductionCost[g] * instance.TimePeriodLength
                                   + instance.ProductionCost[g,1]
                                    )

    for g in instance.AllNondispatchableGenerators:
        observed_renewables_levels[g] = value(instance.NondispatchablePowerUsed[g, 1])

    for b in instance.Buses:
        if value(instance.LoadGenerateMismatch[b, 1]) >= 0.0:
            observed_bus_mismatches[b] = value(instance.posLoadGenerateMismatch[b, 1])
        else:
            observed_bus_mismatches[b] = -1.0 * value(instance.negLoadGenerateMismatch[b, 1])

    for b in instance.Buses:
        observed_bus_LMPs[b] = value(lmp_instance.dual[lmp_instance.PowerBalance[b, 1]])

    return observed_thermal_dispatch_levels, observed_thermal_states, observed_costs, observed_renewables_levels, observed_bus_mismatches, observed_bus_LMPs


def RT_sced_callback(instance, lmp_instance, this_date, hour):
    print("In sced_callback, instance name = {}".format(instance.name))
    print("this_date={0}, hour={1}".format(this_date, hour))
    observed_thermal_dispatch_levels, observed_thermal_states, observed_costs, observed_renewables_levels, observed_bus_mismatches, observed_bus_LMPs = _get_sced_data(instance, lmp_instance)

def DA_ruc_callback(instance, ruc_date):
    print("In ruc_callback, instance name = {}".format(instance.name))
    print("ruc_date={}".format(ruc_date))

    ## the returned data dictionaries are 0-indexed, and have the generatoration levels
    ## for each generator for every time period (0-indexed)
    thermal_gen_cleared_RUC, thermal_reserve_cleared_RUC, renewable_gen_cleared_RUC = _get_ruc_data(instance)

def DA_market_callback(ruc_date, day_ahead_prices, day_ahead_reserve_prices,\
                        thermal_gen_cleared_DA, thermal_reserve_cleared_DA,
                        renewable_gen_cleared_DA):
    '''
    ruc_date : string of the date

    all other are dictionaries, with keys: (generator_name, time_period); time_period is 0-indexed
    NOTE: These dictionaries should be treated as read-only!

    '''
    print("In DA_market_callback, ruc_date={}".format(ruc_date))
