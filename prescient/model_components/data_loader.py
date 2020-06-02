#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## loads and validates input unit commitment data
from __future__ import division
from pyomo.environ import *
import math
import six 

    
from prescient.model_components.decorators import add_model_attr 
component_name = 'data_loader'

def _verify_must_run_t0_state_consistency(model):
    # ensure that the must-run flag and the t0 state are consistent. in partcular, make
    # sure that the unit has satisifed its minimum down time condition if UnitOnT0 is negative.
    
    def verify_must_run_t0_state_consistency_rule(m, g):
        if value(m.MustRun[g]):
            t0_state = value(m.UnitOnT0State[g])
            if t0_state < 0:
                min_down_time = value(m.MinimumDownTime[g])
                if abs(t0_state) < min_down_time:
                    print("DATA ERROR: The generator %s has been flagged as must-run, but its T0 state=%d is inconsistent with its minimum down time=%d" % (g, t0_state, min_down_time))
                    return False
        return True
    
    model.VerifyMustRunT0StateConsistency = BuildAction(model.ThermalGenerators, rule=verify_must_run_t0_state_consistency_rule)
    

def _add_initial_time_periods_on_off_line(model):
    #######################################################################################
    # the number of time periods that a generator must initally on-line (off-line) due to #
    # its minimum up time (down time) constraint.                                         #
    #######################################################################################
    
    def initial_time_periods_online_rule(m, g):
       if not value(m.UnitOnT0[g]):
          return 0
       else:
          return int(min(value(m.NumTimePeriods),
                 round(max(0, value(m.MinimumUpTime[g]) - value(m.UnitOnT0State[g])) / value(m.TimePeriodLengthHours))))
    
    model.InitialTimePeriodsOnLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_online_rule, mutable=True)
    
    def initial_time_periods_offline_rule(m, g):
       if value(m.UnitOnT0[g]):
          return 0
       else:
          return int(min(value(m.NumTimePeriods),
                 round(max(0, value(m.MinimumDownTime[g]) + value(m.UnitOnT0State[g])) / value(m.TimePeriodLengthHours)))) # m.UnitOnT0State is negative if unit is off
    
    model.InitialTimePeriodsOffLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_offline_rule, mutable=True)

def _populate_reserve_requirements(model):
    def populate_reserve_requirements_rule(m):
       reserve_factor = value(m.ReserveFactor)
       if reserve_factor > 0.0:
          for t in m.TimePeriods:
             demand = sum(value(m.Demand[b,t]) for b in sorted(m.Buses))
             m.ReserveRequirement[t] = (reserve_factor * demand)
    
    model.PopulateReserveRequirements = BuildAction(rule=populate_reserve_requirements_rule)

@add_model_attr(component_name)
def load_basic_data(model):
    
    '''
    This loads the model from a dat file
    '''
    warn_neg_load = False
    #
    # Parameters
    #
    
    ##############################################
    # string indentifiers for the set of busses. #
    ##############################################
    
    model.Buses = Set()
    
    ###################
    #   Load Zones    #
    ###################
    #Aggregated loads are distributed in the system based on load coefficient values
    
    model.Zones = Set(initialize=['SingleZone'])
    
    def buildBusZone(m):
        an_element = six.next(m.Zones.__iter__())
        if len(m.Zones) == 1 and an_element == 'SingleZone':
            for b in m.Buses:
                m.BusZone[b] = an_element
        else:
            print("Multiple buses is not supported by buildBusZone in ReferenceModel.py -- someone should fix that!")
            exit(1)
    
    model.BusZone = Param(model.Buses, mutable=True)
    model.BuildBusZone = BuildAction(rule=buildBusZone)
    
    model.LoadCoefficient = Param(model.Buses, default=0.0)
    
    def total_load_coefficient_per_zone(m, z):
        return sum(m.LoadCoefficient[b] for b in m.Buses if str(value(m.BusZone[b]))==str(z))
    model.TotalLoadCoefficientPerZone = Param(model.Zones, initialize=total_load_coefficient_per_zone)
    
    def load_factors_per_bus(m,b):
        if (m.TotalLoadCoefficientPerZone[value(m.BusZone[b])] != 0.0):
            return m.LoadCoefficient[b]/m.TotalLoadCoefficientPerZone[value(m.BusZone[b])]
        else:
            return 0.0
    model.LoadFactor = Param(model.Buses, initialize=load_factors_per_bus, within=NonNegativeReals)
    
    ################################
    
    model.StageSet = Set(ordered=True) 
    
    # IMPORTANT: The stage set must be non-empty - otherwise, zero costs result.
    def check_stage_set(m):
       return (len(m.StageSet) != 0)
    model.CheckStageSet = BuildCheck(rule=check_stage_set)

    ## for backwards capatability (for now)
    model.TimePeriodLength = Param(default=1, within=PositiveReals)
    def time_period_length_validator(m):
        assert(m.TimePeriodLength == 1)
    model.TimePeriodLengthIsOne = BuildAction(rule=time_period_length_validator)
    
    ## IN HOURS, assert athat this must be a positive number
    model.TimePeriodLengthHours = Param(default=1.0, within=PositiveReals)

    ## in minutes, assert that this must be a positive integer
    model.TimePeriodLengthMinutes = Param(default=60, within=PositiveIntegers)

    ## sync the two time period lengths depending on the user's specification
    def harmonize_times(m):
        ## the user can only specify a non-default for 
        ## one of the time period lengths
        assert( (value(m.TimePeriodLengthHours) == 1.0) or (value(m.TimePeriodLengthMinutes) == 60) )
        if value(m.TimePeriodLengthHours) != 1.0:
            m.TimePeriodLengthMinutes = int(round(value(m.TimePeriodLengthHours)*60))
        if value(m.TimePeriodLengthMinutes) != 60:
            m.TimePeriodLengthHours = value(m.TimePeriodLengthMinutes)/60.
    
    model.HarmonizeTimes = BuildAction(rule=harmonize_times)

    model.NumTimePeriods = Param(within=PositiveIntegers, mutable=True)
    
    model.InitialTime = Param(within=PositiveIntegers, default=1)
    model.TimePeriods = RangeSet(model.InitialTime, model.NumTimePeriods)
    
    # the following sets must must come from the data files or from an initialization function that uses 
    # a parameter that tells you when the stages end (and that thing needs to come from the data files)
    model.CommitmentTimeInStage = Set(model.StageSet, within=model.TimePeriods) 
    model.GenerationTimeInStage = Set(model.StageSet, within=model.TimePeriods)
    
    ##############################################
    # Network definition (S)
    ##############################################

    ## for older .dat files
    model.NumTransmissionLines = Param(default=0)
    def num_transimission_lines_validator(m):
        assert(m.NumTransmissionLines == 0)
    model.NumTransmissionLinesIsZero = BuildAction(rule=num_transimission_lines_validator)
    
    model.TransmissionLines = Set()
    
    model.BusFrom = Param(model.TransmissionLines)
    model.BusTo   = Param(model.TransmissionLines)

    model.LinesTo = Set(model.Buses)
    model.LinesFrom = Set(model.Buses)

    def populate_lines_to_from_build_action_rule(m):
        for l in m.TransmissionLines:
            to_bus = value(m.BusTo[l])
            from_bus = value(m.BusFrom[l])
            m.LinesTo[to_bus].add(l)
            m.LinesFrom[from_bus].add(l)
    model.PopulateLinesToFrom = BuildAction(rule=populate_lines_to_from_build_action_rule)

    model.Impedence = Param(model.TransmissionLines, within=NonNegativeReals)

    model.ThermalLimit = Param(model.TransmissionLines) # max flow across the line

    ## Interfaces
    ## NOTE: Lines in iterfaces should be all go "from" the
    ##       other network "to" the modeled network
    model.Interfaces = Set()

    model.InterfaceLines = Set(model.Interfaces, within=model.TransmissionLines)
    model.InterfaceFromLimit = Param(model.Interfaces, within=NonNegativeReals)
    model.InterfaceToLimit = Param(model.Interfaces, within=NonNegativeReals)
    
    ##########################################################
    # string indentifiers for the set of thermal generators. #
    # and their locations. (S)                               #
    ##########################################################
    
    model.ThermalGenerators = Set()
    model.ThermalGeneratorsAtBus = Set(model.Buses)
    
    # thermal generator types must be specified as 'N', 'C', 'G', and 'H',
    # with the obvious interpretation.
    # TBD - eventually add a validator.
    
    model.ThermalGeneratorType = Param(model.ThermalGenerators, within=Any, default='C')
    
    def verify_thermal_generator_buses_rule(m, g):
       for b in m.Buses:
          if g in m.ThermalGeneratorsAtBus[b]:
             return 
       print("DATA ERROR: No bus assigned for thermal generator=%s" % g)
       assert(False)
    
    model.VerifyThermalGeneratorBuses = BuildAction(model.ThermalGenerators, rule=verify_thermal_generator_buses_rule)
    
    model.QuickStart = Param(model.ThermalGenerators, within=Boolean, default=False)
    
    def init_quick_start_generators(m):
        return [g for g in m.ThermalGenerators if value(m.QuickStart[g]) == 1]
    
    model.QuickStartGenerators = Set(within=model.ThermalGenerators, initialize=init_quick_start_generators)
    
    # optionally force a unit to be on.
    model.MustRun = Param(model.ThermalGenerators, within=Boolean, default=False)
    
    def init_must_run_generators(m):
        return [g for g in m.ThermalGenerators if value(m.MustRun[g]) == 1]
    
    model.MustRunGenerators = Set(within=model.ThermalGenerators, initialize=init_must_run_generators)
    
    def nd_gen_init(m,b):
        return []
    model.NondispatchableGeneratorsAtBus = Set(model.Buses, initialize=nd_gen_init)
    
    def NonNoBus_init(m):
        retval = set()
        for b in m.Buses:
            retval = retval.union([gen for gen in m.NondispatchableGeneratorsAtBus[b]])
        return retval
    
    model.AllNondispatchableGenerators = Set(initialize=NonNoBus_init)

    model.NondispatchableGeneratorType = Param(model.AllNondispatchableGenerators, within=Any, default='W')
    
    ######################
    #   Reserve Zones    #
    ######################
    
    # Generators are grouped in zones to provide zonal reserve requirements. #
    # All generators can contribute to global reserve requirements           #
    
    model.ReserveZones = Set()
    model.ZonalReserveRequirement = Param(model.ReserveZones, model.TimePeriods, default=0.0, mutable=True, within=NonNegativeReals)
    model.ReserveZoneLocation = Param(model.ThermalGenerators)
    
    def form_thermal_generator_reserve_zones(m,rz):
        return (g for g in m.ThermalGenerators if m.ReserveZoneLocation[g]==rz)
    model.ThermalGeneratorsInReserveZone = Set(model.ReserveZones, initialize=form_thermal_generator_reserve_zones)
    
    #################################################################
    # the global system demand, for each time period. units are MW. #
    # demand as at busses (S) so total demand is derived            #
    #################################################################
    
    # at the moment, we allow for negative demand. this is probably
    # not a good idea, as "stuff" representing negative demand - including
    # renewables, interchange schedules, etc. - should probably be modeled
    # explicitly.
    
    # Demand can also be given by Zones
    
    model.DemandPerZone = Param(model.Zones, model.TimePeriods, default=0.0, mutable=True)
    
    # Convert demand by zone to demand by bus
    def demand_per_bus_from_demand_per_zone(m,b,t):
        return m.DemandPerZone[value(m.BusZone[b]), t] * m.LoadFactor[b]
    model.Demand = Param(model.Buses, model.TimePeriods, initialize=demand_per_bus_from_demand_per_zone, mutable=True)
    
    def calculate_total_demand(m, t):
        return sum(value(m.Demand[b,t]) for b in sorted(m.Buses))
    model.TotalDemand = Param(model.TimePeriods, initialize=calculate_total_demand)
    
    # at this point, a user probably wants to see if they have negative demand.
    def warn_about_negative_demand_rule(m, b, t):
       this_demand = value(m.Demand[b,t])
       if this_demand < 0.0:
          print("***WARNING: The demand at bus="+str(b)+" for time period="+str(t)+" is negative - value="+str(this_demand)+"; model="+str(m.name)+".")
    
    if warn_neg_load:
        model.WarnAboutNegativeDemand = BuildAction(model.Buses, model.TimePeriods, rule=warn_about_negative_demand_rule)
    
    ##################################################################
    # the global system reserve, for each time period. units are MW. #
    # NOTE: We don't have per-bus / zonal reserve requirements. they #
    #       would be easy to add. (dlw oct 2013: this comment is incorrect, I think)                                   #
    ##################################################################
    
    # we provide two mechanisms to specify reserve requirements. the
    # first is a scaling factor relative to demand, on a per time 
    # period basis. the second is an explicit parameter that specifies
    # the reserver requirement on a per-time-period basis. if the 
    # reserve requirement factor is > 0, then it is used to populate
    # the reserve requirements. otherwise, the user-supplied reserve
    # requirements are used.
    
    model.ReserveFactor = Param(within=Reals, default=-1.0, mutable=True)
    
    model.ReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True)
    
    _populate_reserve_requirements(model)
    
    ##############################################################
    # failure probability for each generator, in any given hour. #
    # not used within the model itself at present, but rather    #
    # used by scripts that read / manipulate the model.          #
    ##############################################################
    
    def probability_failure_validator(m, v, g):
       return v >= 0.0 and v <= 1.0
    
    model.FailureProbability = Param(model.ThermalGenerators, validate=probability_failure_validator, default=0.0)
    
    #####################################################################################
    # a binary indicator as to whether or not each generator is on-line during a given  #
    # time period. intended to represent a sampled realization of the generator failure #
    # probability distributions. strictly speaking, we interpret this parameter value   #
    # as indicating whether or not the generator is contributing (injecting) power to   #
    # the PowerBalance constraint. this parameter is not intended to be used in the     #
    # context of ramping or time up/down constraints.                                   # 
    #####################################################################################
    
    model.GeneratorForcedOutage = Param(model.ThermalGenerators * model.TimePeriods, within=Binary, default=False)

    ## a UnitInterval indicator for de-rating a non-dispatchable generator
    model.NondispatchableGeneratorForcedOutage = Param(model.AllNondispatchableGenerators,
                                                      model.TimePeriods,
                                                      within=UnitInterval,
                                                      default=0)

    model.LineForcedOutage = Param(model.TransmissionLines, model.TimePeriods, within=Binary, default=False)
    
    ####################################################################################
    # minimum and maximum generation levels, for each thermal generator. units are MW. #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################
    
    # you can enter generator limits either once for the generator or for each period (or just take 0)
    
    model.MinimumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    
    def maximum_power_output_validator(m, v, g):
       return v >= value(m.MinimumPowerOutput[g])
    
    model.MaximumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, validate=maximum_power_output_validator, default=0.0)
    
    # wind is similar, but max and min will be equal for non-dispatchable wind
    
    model.MinNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True)
    
    def maximum_nd_output_validator(m, v, g, t):
       return v >= value(m.MinNondispatchablePower[g,t])
    
    model.MaxNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True, validate=maximum_nd_output_validator)
    
    #################################################
    # generator ramp up/down rates. units are MW/h. #
    # IMPORTANT: Generator ramp limits can exceed   #
    # the maximum power output, because it is the   #
    # ramp limit over an hour. If the unit can      #
    # fully ramp in less than an hour, then this    #
    # will occur.                                   #
    #################################################
    
    # limits for normal time periods
    model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=True)
    model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=True)
    
    # limits for time periods in which generators are brought on or off-line.
    # must be no less than the generator minimum output.
    def ramp_limit_validator(m, v, g):
       return v >= m.MinimumPowerOutput[g]

    ## These defaults follow what is in most market manuals
    ## We scale this for the time period below
    def startup_ramp_default(m, g):
        return m.MinimumPowerOutput[g]+m.NominalRampUpLimit[g]/2.
    def shutdown_ramp_default(m, g):
        return m.MinimumPowerOutput[g]+m.NominalRampDownLimit[g]/2.

    model.StartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, default=startup_ramp_default, validate=ramp_limit_validator, mutable=True)
    model.ShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, default=shutdown_ramp_default,  validate=ramp_limit_validator, mutable=True)
    
    ## These get used in the basic UC constraints, which implicity assume RU, RD <= Pmax
    def scale_ramp_up(m, g):
        temp = m.NominalRampUpLimit[g] * m.TimePeriodLengthHours
        if value(temp) > value(m.MaximumPowerOutput[g]):
            return m.MaximumPowerOutput[g]
        else:
            return temp
    model.ScaledNominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=scale_ramp_up, mutable=True)
    
    def scale_ramp_down(m, g):
        temp = m.NominalRampDownLimit[g] * m.TimePeriodLengthHours
        if value(temp) > value(m.MaximumPowerOutput[g]):
            return m.MaximumPowerOutput[g]
        else:
            return temp
    model.ScaledNominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=scale_ramp_down, mutable=True)
    
    def scale_startup_limit(m, g):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length, 
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = (m.StartupRampLimit[g] - m.MinimumPowerOutput[g])*m.TimePeriodLengthHours
        if value(temp) > value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
            return m.MaximumPowerOutput[g]
        else:
            return temp + m.MinimumPowerOutput[g]
    model.ScaledStartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, initialize=scale_startup_limit, mutable=True)
    
    def scale_shutdown_limit(m, g):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = (m.ShutdownRampLimit[g] - m.MinimumPowerOutput[g])*m.TimePeriodLengthHours
        if value(temp) > value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
            return m.MaximumPowerOutput[g]
        else:
            return temp + m.MinimumPowerOutput[g]
    model.ScaledShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, initialize=scale_shutdown_limit, mutable=True)
    
    
    ##########################################################################################################
    # the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
    ##########################################################################################################
    
    model.MinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)
    model.MinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)
    
    ## Assert that MUT and MDT are at least 1 in the time units of the model.
    ## Otherwise, turn on/offs may not be enforced correctly.
    def scale_min_uptime(m, g):
        scaled_up_time = int(round(m.MinimumUpTime[g] / m.TimePeriodLengthHours))
        return min(max(value(scaled_up_time),1), value(m.NumTimePeriods))
    model.ScaledMinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=scale_min_uptime)
    
    def scale_min_downtime(m, g):
        scaled_down_time = int(round(m.MinimumDownTime[g] / m.TimePeriodLengthHours))
        return min(max(value(scaled_down_time),1), value(m.NumTimePeriods))
    model.ScaledMinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=scale_min_downtime)
    
    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################
    
    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.
    
    def t0_state_nonzero_validator(m, v, g):
        return v != 0
    
    model.UnitOnT0State = Param(model.ThermalGenerators, within=Reals, validate=t0_state_nonzero_validator, mutable=True)
    
    def t0_unit_on_rule(m, g):
        return int(value(m.UnitOnT0State[g]) >= 1)
    
    model.UnitOnT0 = Param(model.ThermalGenerators, within=Binary, initialize=t0_unit_on_rule, mutable=True)
    
    _verify_must_run_t0_state_consistency(model)

    _add_initial_time_periods_on_off_line(model)
    
    ####################################################################
    # generator power output at t=0 (initial condition). units are MW. #
    ####################################################################
    
    def between_limits_validator(m, v, g):
       status = (v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g])))
       if status == False:
          print("Failed to validate PowerGeneratedT0 value for g="+g+"; new value="+str(v)+", UnitOnT0="+str(value(m.UnitOnT0[g])))
       return v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g]))
    model.PowerGeneratedT0 = Param(model.ThermalGenerators, within=NonNegativeReals, validate=between_limits_validator, mutable=True)
    
    
    ###############################################
    # startup cost parameters for each generator. #
    ###############################################
    
    # startup costs are conceptually expressed as pairs (x, y), where x represents the number of hours that a unit has been off and y represents
    # the cost associated with starting up the unit after being off for x hours. these are broken into two distinct ordered sets, as follows.
    
    def startup_lags_init_rule(m, g):
       return [value(m.MinimumDownTime[g])] 
    model.StartupLags = Set(model.ThermalGenerators, within=NonNegativeIntegers, ordered=True, initialize=startup_lags_init_rule) # units are hours / time periods.
    
    def startup_costs_init_rule(m, g):
       return [0.0] 
    
    model.StartupCosts = Set(model.ThermalGenerators, within=NonNegativeReals, ordered=True, initialize=startup_costs_init_rule) # units are $.
    
    # startup lags must be monotonically increasing...
    def validate_startup_lags_rule(m, g):
       startup_lags = list(m.StartupLags[g])
    
       if len(startup_lags) == 0:
          print("DATA ERROR: The number of startup lags for thermal generator="+str(g)+" must be >= 1.")
          assert(False)
    
       if startup_lags[0] != value(m.MinimumDownTime[g]):
          print("DATA ERROR: The first startup lag for thermal generator="+str(g)+" must be equal the minimum down time="+str(value(m.MinimumDownTime[g]))+".")
          assert(False)      
    
       for i in range(0, len(startup_lags)-1):
          if startup_lags[i] >= startup_lags[i+1]:
             print("DATA ERROR: Startup lags for thermal generator="+str(g)+" must be monotonically increasing.")
             assert(False)
    
    model.ValidateStartupLags = BuildAction(model.ThermalGenerators, rule=validate_startup_lags_rule)
    
    # while startup costs must be monotonically non-decreasing!
    def validate_startup_costs_rule(m, g):
       startup_costs = list(m.StartupCosts[g])
       for i in range(0, len(startup_costs)-2):
          if startup_costs[i] > startup_costs[i+1]:
             print("DATA ERROR: Startup costs for thermal generator="+str(g)+" must be monotonically non-decreasing.")
             assert(False)
    
    model.ValidateStartupCosts = BuildAction(model.ThermalGenerators, rule=validate_startup_costs_rule)
    
    def validate_startup_lag_cost_cardinalities(m, g):
       if len(m.StartupLags[g]) != len(m.StartupCosts[g]):
          print("DATA ERROR: The number of startup lag entries ("+str(len(m.StartupLags[g]))+") for thermal generator="+str(g)+" must equal the number of startup cost entries ("+str(len(m.StartupCosts[g]))+")")
          assert(False)
    
    model.ValidateStartupLagCostCardinalities = BuildAction(model.ThermalGenerators, rule=validate_startup_lag_cost_cardinalities)
    
    # for purposes of defining constraints, it is useful to have a set to index the various startup costs parameters.
    # entries are 1-based indices, because they are used as indicies into Pyomo sets - which use 1-based indexing.
    
    def startup_cost_indices_init_rule(m, g):
       return range(1, len(m.StartupLags[g])+1)
    
    model.StartupCostIndices = Set(model.ThermalGenerators, within=NonNegativeIntegers, initialize=startup_cost_indices_init_rule)
    
    ## scale the startup lags
    ## Again, assert that this must be at least one in the time units of the model
    def scaled_startup_lags_rule(m, g):
        return [ max(int(round(this_lag / m.TimePeriodLengthHours)),1) for this_lag in m.StartupLags[g] ]
    model.ScaledStartupLags = Set(model.ThermalGenerators, within=NonNegativeIntegers, ordered=True, initialize=scaled_startup_lags_rule)

    ##################################################################################
    # shutdown cost for each generator. in the literature, these are often set to 0. #
    ##################################################################################
    
    model.ShutdownFixedCost = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0) # units are $.
    
    ## BEGIN PRODUCTION COST
    ## NOTE: For better or worse, we handle scaling this to the time period length in the objective function.
    ##       In particular, this is done in objective.py.

    ##################################################################################################################
    # production cost coefficients (for the quadratic) a0=constant, a1=linear coefficient, a2=quadratic coefficient. #
    ##################################################################################################################
    
    model.ProductionCostA0 = Param(model.ThermalGenerators, default=0.0) # units are $/hr (or whatever the time unit is).
    model.ProductionCostA1 = Param(model.ThermalGenerators, default=0.0) # units are $/MWhr.
    model.ProductionCostA2 = Param(model.ThermalGenerators, default=0.0) # units are $/(MWhr^2).
    
    # the parameters below are populated if cost curves are specified as linearized heat rate increment segments.
    #
    # CostPiecewisePoints represents the power output levels defining the segment boundaries.
    # these *must* include the minimum and maximum power output points - a validation check
    # if performed below.
    # 
    # CostPiecewiseValues are the absolute heat rates / costs associated with the corresponding 
    # power output levels. the precise interpretation of whether a value is a heat rate or a cost
    # depends on the value of the FuelCost parameter, specified below.
    
    # there are many ways to interpret the cost piecewise point/value data, when translating into
    # an actual piecewise construct for the model. this interpretation is controlled by the following
    # string parameter, whose legal values are: NoPiecewise (no data provided), Absolute, Incremental, 
    # and SubSegmentation. NoPiecewise means that we're using quadraic cost curves, and will 
    # construct the piecewise data ourselves directly from that cost curve. 
    
    def piecewise_type_validator(m, v):
       return (v == "NoPiecewise") or (v == "Absolute") or (v == "Incremental") or (v != "SubSegementation")
    
    def piecewise_type_init(m):
        boo = False
        for g in m.ThermalGenerators:
            if not (m.ProductionCostA0[g] == 0.0 and m.ProductionCostA1[g] == 0.0 and m.ProductionCostA2[g] == 0.0):
                boo = True
                break
        if boo:
            return "NoPiecewise"
        else:
            return "Absolute"
    
    model.PiecewiseType = Param(validate=piecewise_type_validator,initialize=piecewise_type_init, mutable=True)  #irios: default="Absolute" initialize=piecewise_type_init
    
    def piecewise_init(m, g):
        return []
    
    model.CostPiecewisePoints = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)
    model.CostPiecewiseValues = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)
    
    # a check to ensure that the cost piecewise point parameter was correctly populated.
    # these are global checks, which cannot be performed as a set validation (which 
    # operates on a single element at a time).
    
    # irios: When the check fails, I add the missing PiecewisePoints and Values in order to make it work.
    # I did this in an arbitrary way, but you can change it. In particular, I erased those values which are not 
    # between the minimum power output and the maximum power output. Also, I added those values if they are not in
    # the input data. Finally, I added values (0 and this_generator_piecewise_values[-1] + 1) to end with the same 
    # number of points and values.
    
    def validate_cost_piecewise_points_and_values_rule(m, g):
    
        # IGNACIO: LOOK HERE - CAN YOU MERGE IN THE FOLLOWING INTO THIS RULE?
    #    if m.CostPiecewisePoints[g][1] > m.CorrectCostPiecewisePoints[g][1]:
    #        this_generator_piecewise_values.insert(0,0)
    #    if m.CostPiecewisePoints[g][len(m.CostPiecewisePoints[g])] < m.CorrectCostPiecewisePoints[g][len(m.CorrectCostPiecewisePoints[g])]:
    #        this_generator_piecewise_values.append(this_generator_piecewise_values[-1] + 1)
        
        if value(m.PiecewiseType) == "NoPiecewise":
            # if there isn't any piecewise data specified, we shouldn't find any.
            if len(m.CostPiecewisePoints[g]) > 0:
                print("DATA ERROR: The PiecewiseType parameter was set to NoPiecewise, but piecewise point data was specified!")
                return False
            # if there isn't anything to validate and we didn't expect piecewise 
            # points, we can safely skip the remaining validation steps.
            return True
        else:
            # if the user said there was going to be piecewise data and none was 
            # supplied, they should be notified as to such.
            if len(m.CostPiecewisePoints[g]) == 0:
                print("DATA ERROR: The PiecewiseType parameter was set to something other than NoPiecewise, but no piecewise point data was specified!")
                return False
    
       # per the requirement below, there have to be at least two piecewise points if there are any.
    
        min_output = value(m.MinimumPowerOutput[g])
        max_output = value(m.MaximumPowerOutput[g])   
    
        new_points = sorted(list(m.CostPiecewisePoints[g]))
        new_values = sorted(list(m.CostPiecewiseValues[g]))
    
        if min_output not in new_points:
            print("DATA WARNING: Cost piecewise points for generator g="+str(g)+" must contain the minimum output level="+str(min_output)+" - so we added it.")
            new_points.insert(0, min_output)
    
        if max_output not in new_points:
            print("DATA WARNING: Cost piecewise points for generator g="+str(g)+" must contain the maximum output level="+str(max_output)+" - so we added it.")
            new_points.append(max_output)
    
        # We delete those values which are not in the interval [min_output, max_output]
        new_points = [new_points[i] for i in range(len(new_points)) if (min_output <= new_points[i] and new_points[i] <= max_output)]
    
        # We have to make sure that we have the same number of Points and Values
        if len(new_points) < len(new_values): # if the number of points is less than the number of values, we take the first len(new_points) elements of new_values
            new_values = [new_values[i] for i in range(len(new_points))]
        if len(new_points) > len(new_values): # if the number of values is lower, then we add values at the end of new_values increasing by 1 each time.
            i = 1
            while len(new_points) != len(new_values):
                new_values.append(new_values[-1] + i)
                i += 1 
    
        if list(m.CostPiecewisePoints[g]) != new_points:
            m.CostPiecewisePoints[g].clear()
            #m.CostPiecewisePoints[g].add(*new_points) # dlw and Julia July 2014 - changed to below.
            for pcwpoint in new_points:
                m.CostPiecewisePoints[g].add(pcwpoint) 
    
        if list(m.CostPiecewiseValues[g]) != new_values:
            m.CostPiecewiseValues[g].clear()
            # m.CostPiecewiseValues[g].add(*new_values) # dlw and Julia July 2014 - changed to below.
            for pcwvalue in new_values:
                m.CostPiecewiseValues[g].add(pcwvalue)
    
        return True
    
    model.ValidateCostPiecewisePointsAndValues = BuildCheck(model.ThermalGenerators, rule=validate_cost_piecewise_points_and_values_rule)
    
    # Sets the cost of fuel to the generator.  Defaults to 1 so that we could just input cost as heat rates.
    model.FuelCost = Param(model.ThermalGenerators, default=1.0) 
    
    # Minimum production cost (needed because Piecewise constraint on ProductionCost 
    # has to have lower bound of 0, so the unit can cost 0 when off -- this is added
    # back in to the objective if a unit is on
    def minimum_production_cost(m, g):
        if len(m.CostPiecewisePoints[g]) > 1:
            return m.CostPiecewiseValues[g].first() * m.FuelCost[g]
        else:
            return  m.FuelCost[g] * \
                   (m.ProductionCostA0[g] + \
                    m.ProductionCostA1[g] * m.MinimumPowerOutput[g] + \
                    m.ProductionCostA2[g] * (m.MinimumPowerOutput[g]**2))
    model.MinimumProductionCost = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=minimum_production_cost, mutable=True)
    
    ##############################################################################################
    # number of pieces in the linearization of each generator's quadratic cost production curve. #
    ##############################################################################################
    
    model.NumGeneratorCostCurvePieces = Param(within=PositiveIntegers, default=2, mutable=True)


    #######################################################################
    # points for piecewise linearization of power generation cost curves. #
    #######################################################################
    
    # BK -- changed to reflect that the generator's power output variable is always above minimum in the ME model
    #       this simplifies things quite a bit..
    
    # maps a (generator, time-index) pair to a list of points defining the piecewise cost linearization breakpoints.
    # the time index is redundant, but required - in the current implementation of the Piecewise construct, the 
    # breakpoints must be indexed the same as the Piecewise construct itself.
    
    # the points are expected to be on the interval [0, maxpower-minpower], and must contain both endpoints. 
    # power generated can always be 0, and piecewise expects the entire variable domain to be represented.
    model.PowerGenerationPiecewisePoints = {}
    
    # NOTE: the values are relative to the minimum production cost, i.e., the values represent
    # incremental costs relative to the minimum production cost.
    
    # IMPORTANT: These values are *not* scaled by the FuelCost of the generator. This scaling is
    #            performed subsequently, in the "_production_cost_function" code, which is used
    #            by Piecewise to compute the production cost of the generator. all values must
    #            be non-negative.
    model.PowerGenerationPiecewiseValues = {}
    
    def power_generation_piecewise_points_rule(m, g, t):
    
        # factor out the fuel cost here, as the piecewise approximation is scaled by fuel cost
        # elsewhere in the model (i.e., in the Piecewise construct below).
        minimum_production_cost = value(m.MinimumProductionCost[g]) / value(m.FuelCost[g])
    
        # minimum output
        minimum_power_output = value(m.MinimumPowerOutput[g])
        
        piecewise_type = value(m.PiecewiseType)
    
        if piecewise_type == "Absolute":
    
           piecewise_values = list(m.CostPiecewiseValues[g])
           piecewise_points = list(m.CostPiecewisePoints[g])
           m.PowerGenerationPiecewiseValues[g,t] = {}
           m.PowerGenerationPiecewisePoints[g,t] = [] 
           for i in range(len(piecewise_points)):
              this_point = piecewise_points[i] - minimum_power_output
              m.PowerGenerationPiecewisePoints[g,t].append(this_point)
              m.PowerGenerationPiecewiseValues[g,t][this_point] = piecewise_values[i] - minimum_production_cost
    
    
        elif piecewise_type == "Incremental":
           # NOTE: THIS DOESN'T WORK!!!
           if len(m.CostPiecewisePoints[g]) > 0:
              PowerBreakpoints = list(m.CostPiecewisePoints[g])
              ## adjust these down to min power
              for i in range(len(PowerBreakpoints)):
                  PowerBreakpoints[i] = PowerBreakpoints[i] - minimum_power_output 
              assert(PowerBreakPoint[0] == 0.0) 
              IncrementalCost = list(m.CostPiecewiseValues[g])
              CostBreakpoints = {}
              CostBreakpoints[0] = 0
              for i in range(1, len(IncrementalCost) + 1):
                 CostBreakpoints[PowerBreakpoints[i]] = CostBreakpoints[PowerBreakpoints[i-1]] + \
                     (PowerBreakpoints[i] - PowerBreakpoints[i-1])* IncrementalCost[i-1]
              m.PowerGenerationPiecewisePoints[g,t] = list(PowerBreakpoints)
              m.PowerGenerationPiecewiseValues[g,t] = dict(CostBreakpoints)
              #        print g, t, m.PowerGenerationPiecewisePoints[g, t]
              #        print g, t, m.PowerGenerationPiecewiseValues[g, t]
    
           else:
              print("***BADGenerators must have at least 1 point in their incremental cost curve")
              assert(False)
    
        elif piecewise_type == "SubSegmentation":
    
           print("***BAD - we don't have logic for generating piecewise points of type SubSegmentation!!!")
           assert(False)
    
        else: # piecewise_type == "NoPiecewise"
    
           if value(m.ProductionCostA2[g]) == 0:
              # If cost is linear, we only need two points -- (0,0) and (MaxOutput-MinOutput, MaxCost-MinCost))
              min_power = value(m.MinimumPowerOutput[g])
              max_power = value(m.MaximumPowerOutput[g])
              if min_power == max_power:
                 m.PowerGenerationPiecewisePoints[g, t] = [0.0]
              else:
                 m.PowerGenerationPiecewisePoints[g, t] = [0.0, max_power-min_power]
    
              m.PowerGenerationPiecewiseValues[g,t] = {}
    
              m.PowerGenerationPiecewiseValues[g,t][0.0] = 0.0
    
              if min_power != max_power:
                 m.PowerGenerationPiecewiseValues[g,t][max_power-min_power] = \
                     value(m.ProductionCostA0[g]) + \
                     value(m.ProductionCostA1[g]) * max_power \
                     - minimum_production_cost
    
           else:
               min_power = value(m.MinimumPowerOutput[g])
               max_power = value(m.MaximumPowerOutput[g])
               n = value(m.NumGeneratorCostCurvePieces)
               width = (max_power - min_power) / float(n)
               if width == 0:
                   m.PowerGenerationPiecewisePoints[g, t] = [0]
               else:
                   m.PowerGenerationPiecewisePoints[g, t] = []
                   m.PowerGenerationPiecewisePoints[g, t].extend([0 + i*width for i in range(0,n+1)])
                   # NOTE: due to numerical precision limitations, the last point in the x-domain
                   #       of the generation piecewise cost curve may not be precisely equal to the 
                   #       maximum power output level of the generator. this can cause Piecewise to
                   #       sqawk, as it would like the upper bound of the variable to be represented
                   #       in the domain. so, we will make it so.
                   m.PowerGenerationPiecewisePoints[g, t][-1] = max_power - min_power
               m.PowerGenerationPiecewiseValues[g,t] = {}
               for i in range(len(m.PowerGenerationPiecewisePoints[g, t])):
                   m.PowerGenerationPiecewiseValues[g,t][m.PowerGenerationPiecewisePoints[g,t][i]] = \
                              value(m.ProductionCostA0[g]) + \
                              value(m.ProductionCostA1[g]) * (m.PowerGenerationPiecewisePoints[g, t][i] + min_power) + \
                              value(m.ProductionCostA2[g]) * (m.PowerGenerationPiecewisePoints[g, t][i] + min_power)**2 \
                              - minimum_production_cost
               assert(m.PowerGenerationPiecewisePoints[g, t][0] == 0)
        
        # validate the computed points, independent of the method used to generate them.
        # nothing should be negative, and the costs should be monotonically non-decreasing.
        for i in range(0, len(m.PowerGenerationPiecewisePoints[g, t])):
           this_level = m.PowerGenerationPiecewisePoints[g, t][i]
           assert this_level >= 0.0

        ## finally, make these tuples because they shouldn't be modified after this
        #m.PowerGenerationPiecewisePoints[g,t] = tuple(m.PowerGenerationPiecewisePoints[g,t])
        #m.PowerGenerationPiecewiseValues[g,t] = tuple(m.PowerGenerationPiecewiseValues[g,t])
    
    model.CreatePowerGenerationPiecewisePoints = BuildAction(model.ThermalGenerators * model.TimePeriods, rule=power_generation_piecewise_points_rule)

    ModeratelyBigPenalty = 1e3
    
    model.ReserveShortfallPenalty = Param(within=NonNegativeReals, default=ModeratelyBigPenalty, mutable=True)

    #########################################
    # penalty costs for constraint violation #
    #########################################
    
    BigPenalty = 1e4
    
    model.LoadMismatchPenalty = Param(within=NonNegativeReals, default=BigPenalty, mutable=True)

    ## END PRODUCTION COST CALCULATIONS

    #
    # STORAGE parameters
    #
    
    
    model.Storage = Set()
    model.StorageAtBus = Set(model.Buses, initialize=Set())
    
    def verify_storage_buses_rule(m, s):
        for b in m.Buses:
            if s in m.StorageAtBus[b]:
                return
        print("DATA ERROR: No bus assigned for storage element=%s" % s)
        assert(False)
    
    model.VerifyStorageBuses = BuildAction(model.Storage, rule=verify_storage_buses_rule)
    
    ####################################################################################
    # minimum and maximum power ratings, for each storage unit. units are MW.          #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################
    
    # Storage power output >0 when discharging
    
    model.MinimumPowerOutputStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)
    
    def maximum_power_output_validator_storage(m, v, s):
        return v >= value(m.MinimumPowerOutputStorage[s])
    
    model.MaximumPowerOutputStorage = Param(model.Storage, within=NonNegativeReals, validate=maximum_power_output_validator_storage, default=0.0)
    
    #Storage power input >0 when charging
    
    model.MinimumPowerInputStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)
    
    def maximum_power_input_validator_storage(m, v, s):
        return v >= value(m.MinimumPowerInputStorage[s])
    
    model.MaximumPowerInputStorage = Param(model.Storage, within=NonNegativeReals, validate=maximum_power_input_validator_storage, default=0.0)
    
    ###############################################
    # storage ramp up/down rates. units are MW/h. #
    ###############################################
    
    # ramp rate limits when discharging
    model.NominalRampUpLimitStorageOutput    = Param(model.Storage, within=NonNegativeReals)
    model.NominalRampDownLimitStorageOutput  = Param(model.Storage, within=NonNegativeReals)
    
    # ramp rate limits when charging
    model.NominalRampUpLimitStorageInput     = Param(model.Storage, within=NonNegativeReals)
    model.NominalRampDownLimitStorageInput   = Param(model.Storage, within=NonNegativeReals)
    
    def scale_storage_ramp_up_out(m, s):
        return m.NominalRampUpLimitStorageOutput[s] * m.TimePeriodLengthHours
    model.ScaledNominalRampUpLimitStorageOutput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_up_out)
    
    def scale_storage_ramp_down_out(m, s):
        return m.NominalRampDownLimitStorageOutput[s] * m.TimePeriodLengthHours
    model.ScaledNominalRampDownLimitStorageOutput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_down_out)
    
    def scale_storage_ramp_up_in(m, s):
        return m.NominalRampUpLimitStorageInput[s] * m.TimePeriodLengthHours
    model.ScaledNominalRampUpLimitStorageInput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_up_in)
    
    def scale_storage_ramp_down_in(m, s):
        return m.NominalRampDownLimitStorageInput[s] * m.TimePeriodLengthHours
    model.ScaledNominalRampDownLimitStorageInput = Param(model.Storage, within=NonNegativeReals, initialize=scale_storage_ramp_down_in)
    
    ####################################################################################
    # minimum state of charge (SOC) and maximum energy ratings, for each storage unit. #
    # units are MWh for energy rating and p.u. (i.e. [0,1]) for SOC     #
    ####################################################################################
    
    # you enter storage energy ratings once for each storage unit
    
    model.MaximumEnergyStorage = Param(model.Storage, within=NonNegativeReals, default=0.0)
    model.MinimumSocStorage = Param(model.Storage, within=PercentFraction, default=0.0)
    
    ################################################################################
    # round trip efficiency for each storage unit given as a fraction (i.e. [0,1]) #
    ################################################################################
    
    model.InputEfficiencyEnergy  = Param(model.Storage, within=PercentFraction, default=1.0)
    model.OutputEfficiencyEnergy = Param(model.Storage, within=PercentFraction, default=1.0)
    model.RetentionRate          = Param(model.Storage, within=PercentFraction, default=1.0) ## assumed to be %/hr

    ## this will be multiplied by itself 1/m.TimePeriodLengthHours times, so this is the scaling to
    ## get us back to %/hr
    def scaled_retention_rate(m,s):
        return value(m.RetentionRate[s])**value(m.TimePeriodLengthHours)
    model.ScaledRetentionRate = Param(model.Storage, within=PercentFraction, initialize=scaled_retention_rate)
    
    ########################################################################
    # end-point SOC for each storage unit. units are in p.u. (i.e. [0,1])  #
    ########################################################################
    
    # end-point values are the SOC targets at the final time period. With no end-point constraints
    # storage units will always be empty at the final time period.
    
    model.EndPointSocStorage = Param(model.Storage, within=PercentFraction, default=0.5)
    
    ############################################################
    # storage initial conditions: SOC, power output and input  #
    ############################################################
    
    def t0_storage_power_input_validator(m, v, s):
        return (v >= value(m.MinimumPowerInputStorage[s])) and (v <= value(m.MaximumPowerInputStorage[s]))
    
    def t0_storage_power_output_validator(m, v, s):
        return (v >= value(m.MinimumPowerInputStorage[s])) and (v <= value(m.MaximumPowerInputStorage[s]))
    
    model.StoragePowerOutputOnT0 = Param(model.Storage, within=NonNegativeReals, validate=t0_storage_power_output_validator, default=0.0)
    model.StoragePowerInputOnT0  = Param(model.Storage, within=NonNegativeReals, validate=t0_storage_power_input_validator, default=0.0)
    model.StorageSocOnT0         = Param(model.Storage, within=PercentFraction, default=0.5)

    return model
