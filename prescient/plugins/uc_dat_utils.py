#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

""" This module provides functions for interacting with the hopefully soon-to-be legacy unit commitment *.dat files

It uses pyomos dat file engine to parse the data and load it in to a GridNetwork object.

Large parts of this code are borrowed from prescient/prescient/models/knueven/ReferenceModel.py

ToDo
----
* Everything
"""
from pyomo.environ import *

import os.path
import egret.grid_elements as ge

import six

def create_dat_file_gridnetwork(dat_file):

    data = _definitions().create_instance(dat_file)

    case_name = os.path.basename(dat_file)

    gn = ge.GridNetwork(case_name, times=list(data.TimePeriods))

    ## global stuff
    # we have no baseMVA in the dat files, so we'll just make it 100
    gn.baseMVA = 100.
    
    # load the time periods
    gn.time_period_length = data.TimePeriodLength

    for t in data.TimePeriods:
        gn.time = t
        gn.global_reserve_requirement = data.ReserveRequirement[t]/gn.baseMVA

    gn.time = None

    # add the buses
    first_bus = True
    for b in data.Buses:

        BUS_I = str(b)
        ZONE = str(data.BusZone[b])

        bus = ge.Bus(BUS_I)

        bus.zone = ZONE
        bus.ID = BUS_I

        gn.add_bus(bus)

        for t in data.TimePeriods:
            gn.time = t
            bus.set_load(data.Demand[b,t]/gn.baseMVA, 0.)

        if first_bus:
            gn.set_reference_bus(BUS_I, 0.)
            first_bus = False

    gn.time = None

    # add the branches
    for l in data.TransmissionLines:
        name = str(l)
        F_BUS = data.BusFrom[l]
        T_BUS = data.BusTo[l]

        BR_R = None
        ## impedence is not in per unit
        BR_X = data.Impedence[l]*gn.baseMVA

        BR_B = None

        RATE_A = data.ThermalLimit[l]/gn.baseMVA
        RATE_B = RATE_A
        RATE_C = RATE_A

        TAP = 1.
        SHIFT = 0.0  # these hard-coded values are the defaults
        BR_STATUS = 1 #from the RTS-GMLC MATPOWER writer
        ANGMIN = -90.
        ANGMAX = 90.

        PF = None     # these values are not given
        QF = None
        PT = None
        QT = None

        branch = ge.Branch(name, F_BUS, T_BUS)
        branch.resistance = BR_R
        branch.reactance = BR_X
        branch.charging_susceptance = BR_B
        branch.transformer_tap_ratio = 1.0 if TAP==0.0 else TAP
        branch.transformer_phase_shift = SHIFT
        branch.rating_long_term = RATE_A
        branch.rating_short_term = RATE_B
        branch.rating_emergency = RATE_C
        branch.angle_min = ANGMIN
        branch.angle_max = ANGMAX
        assert(BR_STATUS == 0 or BR_STATUS == 1)
        if BR_STATUS == 1:
            branch.in_service = True
        else:
            branch.in_service = False
        branch.pf = PF
        branch.qf = QF
        branch.pt = PT
        branch.qt = QT

        gn.add_branch(branch)

    ## add the generators
    for b in data.Buses:
        for g in data.ThermalGeneratorsAtBus[b]:

            name = str(g)
            PMAX = data.MaximumPowerOutput[g]/gn.baseMVA
            PMIN = data.MinimumPowerOutput[g]/gn.baseMVA
            UNIT_TYPE = data.ThermalGeneratorType[g]
            FUEL = UNIT_TYPE
            MBASE = 100.
            GEN_STATUS = 1.
            QMIN = None
            QMAX = None
            RAMP_Q = None
            VG = None
            GEN_BUS = b

            gen = ge.ThermalGenerator(name)
            gen.attached_bus_name = GEN_BUS
            gen.Vg = VG
            gen.mBASE = MBASE
            gen.pg = None
            gen.qg = None
            gen.P_min = PMIN
            gen.P_max = PMAX
            gen.Q_min = QMIN
            gen.Q_max = QMAX
            gen.ramp_q = RAMP_Q
            gen.fuel = FUEL
            gen.unit_type = UNIT_TYPE

            gen.ramp_up_60 = data.NominalRampUpLimit[g]/gn.baseMVA
            gen.ramp_down_60  = data.NominalRampDownLimit[g]/gn.baseMVA

            gen.Pstartup_capacity = data.StartupRampLimit[g]/gn.baseMVA
            gen.Pshutdown_capacity = data.ShutdownRampLimit[g]/gn.baseMVA

            gen.fast_start = data.QuickStart[g]
            gen.must_run = data.MustRun[g]

            gen.min_up_time = data.MinimumUpTime[g]
            gen.min_down_time = data.MinimumDownTime[g]

            gen.startup_costs = list(zip(list(data.StartupLags[g]), list(data.StartupCosts[g])))
            gen.shutdown_cost = data.ShutdownFixedCost[g]

            gen.initial_P = data.PowerGeneratedT0[g]/gn.baseMVA
            gen.initial_status = data.UnitOnT0State[g]

            if len(list(data.CostPiecewisePoints[g])) == 0:
                gen.P_cost_coeffs = [ data.FuelCost[g]*data.ProductionCostA0[g], data.FuelCost[g]*data.ProductionCostA1[g]*gn.baseMVA, data.FuelCost[g]*data.ProductionCostA2[g]*(gn.baseMVA*gn.baseMVA) ]
                gen.cost_type = ge.CostType.quadratic

            else:
                gen.P_cost_coeffs = [ (data.CostPiecewisePoints[g][i]/gn.baseMVA, data.CostPiecewiseValues[g][i]) for i in range(1,len(data.CostPiecewisePoints[g])+1)]
                gen.cost_type = ge.CostType.piecewise

            gn.add_generator(gen)

            for t in data.TimePeriods:
                gn.time = t
                gen.in_service = not(data.GeneratorForcedOutage[g,t])
            gn.time = None

        for g in data.NondispatchableGeneratorsAtBus[b]:

            name = str(g)
            PMAX = None
            PMIN = None
            UNIT_TYPE = data.NondispatchableGeneratorType[g]
            FUEL = UNIT_TYPE
            MBASE = 100.
            GEN_STATUS = 1.
            QMIN = None
            QMAX = None
            RAMP_Q = None
            VG = None
            GEN_BUS = b

            gen = ge.RenewableGenerator(name)
            gen.attached_bus_name = GEN_BUS
            gen.Vg = VG
            gen.mBASE = MBASE
            gen.in_service = True
            gen.pg = None
            gen.qg = None
            gen.P_min = PMIN
            gen.P_max = PMAX
            gen.Q_min = QMIN
            gen.Q_max = QMAX
            gen.ramp_q = RAMP_Q
            gen.fuel = FUEL
            gen.unit_type = UNIT_TYPE

            gn.add_generator(gen)

            for t in data.TimePeriods:
                gn.time = t
                gen.P_min = data.MinNondispatchablePower[g,t]/gn.baseMVA
                gen.P_max = data.MaxNondispatchablePower[g,t]/gn.baseMVA

        for s in data.StorageAtBus[b]:

            name = str(s)
            st = ge.Storage(name)
            st.attached_bus_name = b
            st.capacity = data.MaximumEnergyStorage[s]/gn.baseMVA
            st.initial_SOC = None
            st.minimum_SOC = data.MinimumSocStorage[s]
            st.charge_efficiency = data.InputEfficiencyEnergy[s]
            st.discharge_efficiency = data.OutputEfficiencyEnergy[s]
            st.max_discharge_rate = data.MaximumPowerOutputStorage[s]/gn.baseMVA
            st.max_charge_rate = data.MaximumPowerInputStorage[s]/gn.baseMVA
            st.min_discharge_rate = data.MinimumPowerOutputStorage[s]/gn.baseMVA
            st.min_charge_rate = data.MinimumPowerInputStorage[s]/gn.baseMVA
            st.retention_rate = data.RetentionRate[s]
            st.ramp_up_input_60 = data.NominalRampUpLimitStorageInput[s]/gn.baseMVA
            st.ramp_down_input_60 = data.NominalRampDownLimitStorageInput[s]/gn.baseMVA
            st.ramp_up_output_60 = data.NominalRampUpLimitStorageOutput[s]/gn.baseMVA
            st.ramp_down_output_60 = data.NominalRampDownLimitStorageOutput[s]/gn.baseMVA
            st.final_SOC = data.EndPointSocStorage[s]

            gn.add_storage(st)

    return gn
                

def _definitions():
    need_mutable_parameters = False
    
    regulation_services = True
    
    reserve_services = True

    storage_services = True
    
    model = AbstractModel()
    
    model.name = "DataLoader"
    
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
    
    model.TimePeriodLength = Param(default=1.0)
    model.NumTimePeriods = Param(within=PositiveIntegers, mutable=need_mutable_parameters)
    
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
    
    # derived from TransmissionLines
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
    
    ##########################################################
    # string indentifiers for the set of thermal generators. #
    # and their locations. (S)                               #
    ##########################################################
    
    model.ThermalGenerators = Set()
    model.ThermalGeneratorsAtBus = Set(model.Buses)
    
    # thermal generator types must be specified as 'N', 'C', 'G', and 'H',
    # with the obvious interpretation. or maybe not so obvious interpration.
    # available options are:
    # N - nuclear
    # E - geothermal (i.e., earth!)
    # B - biomass
    # C - coal
    # G - gas
    # H - hydro
    # W - wind
    # S - solar
    # O - oil
    
    model.ThermalGeneratorType = Param(model.ThermalGenerators, within=Any, default='C')
    
    def verify_thermal_generator_buses_rule(m, g):
       for b in m.Buses:
          if g in m.ThermalGeneratorsAtBus[b]:
             return 
       print("DATA ERROR: No bus assigned for thermal generator=%s" % g)
       assert(False)
    
    model.VerifyThermalGeneratorBuses = BuildAction(model.ThermalGenerators, rule=verify_thermal_generator_buses_rule)
    
    model.QuickStart = Param(model.ThermalGenerators, within=Boolean, default=False)
    
    
    # optionally force a unit to be on.
    model.MustRun = Param(model.ThermalGenerators, within=Boolean, default=False)
    
    
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
    model.ZonalReserveRequirement = Param(model.ReserveZones, model.TimePeriods, default=0.0, mutable=need_mutable_parameters, within=NonNegativeReals)
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
    
    model.DemandPerZone = Param(model.Zones, model.TimePeriods, default=0.0, mutable=need_mutable_parameters)
    
    # Convert demand by zone to demand by bus
    def demand_per_bus_from_demand_per_zone(m,b,t):
        return m.DemandPerZone[value(m.BusZone[b]), t] * m.LoadFactor[b]
    model.Demand = Param(model.Buses, model.TimePeriods, initialize=demand_per_bus_from_demand_per_zone, mutable=need_mutable_parameters)
    
    def calculate_total_demand(m, t):
        return sum(value(m.Demand[b,t]) for b in m.Buses)
    model.TotalDemand = Param(model.TimePeriods, within=NonNegativeReals, initialize=calculate_total_demand)
    
    # at this point, a user probably wants to see if they have negative demand.
    def warn_about_negative_demand_rule(m, b, t):
       this_demand = value(m.Demand[b,t])
       if this_demand < 0.0:
          print("***WARNING: The demand at bus="+str(b)+" for time period="+str(t)+" is negative - value="+str(this_demand)+"; model="+str(m.name)+".")
    
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
    
    model.ReserveFactor = Param(within=Reals, default=-1.0, mutable=need_mutable_parameters)
    
    model.ReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=True)
    
    def populate_reserve_requirements_rule(m):
       reserve_factor = value(m.ReserveFactor)
       if reserve_factor > 0.0:
          for t in m.TimePeriods:
             demand = sum(value(m.Demand[b,t]) for b in m.Buses)
             m.ReserveRequirement[t] = reserve_factor * demand
    
    model.PopulateReserveRequirements = BuildAction(rule=populate_reserve_requirements_rule)
    
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
    
    ####################################################################################
    # minimum and maximum generation levels, for each thermal generator. units are MW. #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################
    
    # you can enter generator limits either once for the generator or for each period (or just take 0)
    
    model.MinimumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    
    def maximum_power_output_validator(m, v, g):
       return v >= value(m.MinimumPowerOutput[g])
    
    model.MaximumPowerOutput = Param(model.ThermalGenerators, within=NonNegativeReals, validate=maximum_power_output_validator, default=0.0)
    
    model.MinNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=need_mutable_parameters)
    
    def maximum_nd_output_validator(m, v, g, t):
       return v >= value(m.MinNondispatchablePower[g,t])
    
    model.MaxNondispatchablePower = Param(model.AllNondispatchableGenerators, model.TimePeriods, within=NonNegativeReals, default=0.0, mutable=need_mutable_parameters, validate=maximum_nd_output_validator)
    
    #################################################
    # generator ramp up/down rates. units are MW/h. #
    # IMPORTANT: Generator ramp limits can exceed   #
    # the maximum power output, because it is the   #
    # ramp limit over an hour. If the unit can      #
    # fully ramp in less than an hour, then this    #
    # will occur.                                   #
    #################################################
    
    # limits for normal time periods
    model.NominalRampUpLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=need_mutable_parameters)
    model.NominalRampDownLimit = Param(model.ThermalGenerators, within=NonNegativeReals, mutable=need_mutable_parameters)
    
    # limits for time periods in which generators are brought on or off-line.
    # must be no less than the generator minimum output.
    # We're ignoring this validator for right now and enforcing meaning when scaling
    def ramp_limit_validator(m, v, g):
       return True
       #return v >= m.MinimumPowerOutput[g] and v <= m.MaximumPowerOutput[g]
    
    model.StartupRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, mutable=need_mutable_parameters)
    model.ShutdownRampLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=ramp_limit_validator, mutable=need_mutable_parameters)
    
    
    ##########################################################################################################
    # the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
    ##########################################################################################################
    
    model.MinimumUpTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)
    model.MinimumDownTime = Param(model.ThermalGenerators, within=NonNegativeIntegers, default=0)
    
    
    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################
    
    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.
    
    def t0_state_nonzero_validator(m, v, g):
        return v != 0
    
    model.UnitOnT0State = Param(model.ThermalGenerators, within=Reals, validate=t0_state_nonzero_validator, mutable=need_mutable_parameters)
    
    def t0_unit_on_rule(m, g):
        return int(value(m.UnitOnT0State[g]) >= 1)
    
    model.UnitOnT0 = Param(model.ThermalGenerators, within=Binary, initialize=t0_unit_on_rule, mutable=need_mutable_parameters)
    
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
    
    #######################################################################################
    # the number of time periods that a generator must initally on-line (off-line) due to #
    # its minimum up time (down time) constraint.                                         #
    #######################################################################################
    
    def initial_time_periods_online_rule(m, g):
       if not value(m.UnitOnT0[g]):
          return 0
       else:
          return int(min(value(m.NumTimePeriods),
                 round(max(0, value(m.MinimumUpTime[g]) - value(m.UnitOnT0State[g])) / value(m.TimePeriodLength))))
    
    model.InitialTimePeriodsOnLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_online_rule, mutable=need_mutable_parameters)
    
    def initial_time_periods_offline_rule(m, g):
       if value(m.UnitOnT0[g]):
          return 0
       else:
          return int(min(value(m.NumTimePeriods),
                 round(max(0, value(m.MinimumDownTime[g]) + value(m.UnitOnT0State[g])) / value(m.TimePeriodLength)))) # m.UnitOnT0State is negative if unit is off
    
    model.InitialTimePeriodsOffLine = Param(model.ThermalGenerators, within=NonNegativeIntegers, initialize=initial_time_periods_offline_rule, mutable=need_mutable_parameters)
    
    ####################################################################
    # generator power output at t=0 (initial condition). units are MW. #
    ####################################################################
    
    def between_limits_validator(m, v, g):
       status = (v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g])))
       if status == False:
          print("Failed to validate PowerGeneratedT0 value for g="+g+"; new value="+str(v)+", UnitOnT0="+str(value(m.UnitOnT0[g])))
       return v <= (value(m.MaximumPowerOutput[g]) * value(m.UnitOnT0[g]))  and v >= (value(m.MinimumPowerOutput[g]) * value(m.UnitOnT0[g]))
    model.PowerGeneratedT0 = Param(model.ThermalGenerators, within=NonNegativeReals, validate=between_limits_validator, mutable=need_mutable_parameters)
    
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
    # and SubSegmentation. NoPiecewise means that we're using quadratic cost curves, and will 
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
    
    model.PiecewiseType = Param(validate=piecewise_type_validator,initialize=piecewise_type_init, mutable=need_mutable_parameters)  #irios: default="Absolute" initialize=piecewise_type_init
    
    def piecewise_init(m, g):
        return []
    
    model.CostPiecewisePoints = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)
    model.CostPiecewiseValues = Set(model.ThermalGenerators, initialize=piecewise_init, ordered=True, within=NonNegativeReals)
    
    
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
    model.MinimumProductionCost = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=minimum_production_cost, mutable=need_mutable_parameters)
    
    
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
    
    ##################################################################################
    # shutdown cost for each generator. in the literature, these are often set to 0. #
    ##################################################################################
    
    model.ShutdownFixedCost = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0) # units are $.
    
    #################################
    # Regulation ancillary services #
    #################################
    
    if regulation_services:
    
        # Regulation
        model.RegulationProvider = Param(model.ThermalGenerators, within=Binary, default=0) #indicates if a unit is offering regulation
    
        # When units are selected for regulation, their limits are bounded by the RegulationHighLimit and RegulationLowLimit
        # I'll refer to it as the "regulation band"
        def regulation_high_limit_validator(m, v, g):
            return v <= value(m.MaximumPowerOutput[g])
        def regulation_high_limit_init(m, g):
            return value(m.MaximumPowerOutput[g])   
        model.RegulationHighLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=regulation_high_limit_validator, initialize=regulation_high_limit_init)
    
        def calculate_max_power_minus_reg_high_limit_rule(m, g):
            return m.MaximumPowerOutput[g] - m.RegulationHighLimit[g]
        model.MaxPowerOutputMinusRegHighLimit = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=calculate_max_power_minus_reg_high_limit_rule)
    
        def regulation_low_limit_validator(m, v, g):
            return (v <= value(m.RegulationHighLimit[g]) and v >= value(m.MinimumPowerOutput[g]))
        def regulation_low_limit_init(m, g):
            return value(m.MinimumPowerOutput[g])
        model.RegulationLowLimit = Param(model.ThermalGenerators, within=NonNegativeReals, validate=regulation_low_limit_validator, initialize=regulation_low_limit_init)
    
        # Regulation capacity is calculated as the max of "regulation band" and 5*AutomaticResponseRate
        model.AutomaticResponseRate = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    
        def calculate_regulation_capability_rule(m, g):
            temp1 = 5 * m.AutomaticResponseRate[g]
            temp2 = (m.RegulationHighLimit[g] - m.RegulationLowLimit[g])/2
            if temp1 > temp2:
                return temp2
            else:
                return temp1
    
        model.RegulationCapability = Param(model.ThermalGenerators, within=NonNegativeReals, initialize=calculate_regulation_capability_rule, default=0.0)
        model.ZonalRegulationRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
        model.GlobalRegulationRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)
    
        # regulation cost
        model.RegulationOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    
    if reserve_services:
        # At ISO-NE the ancillary services are not "co-optimize" in the day-ahead.
        # However, this formulation handles ancillary service offers which are common in other markets.
        #
        # spinning reserve
        model.SpinningReserveTime = Param(within=NonNegativeReals, default=0.16666667) # in hours, varies among ISOs
        model.ZonalSpinningReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
        model.SystemSpinningReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)
        model.SpinningReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
    
        # non-spinning reserve
        model.NonSpinningReserveAvailable = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)  #ISO-NE's Claim10 parameter
        model.NonSpinningReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
        model.ZonalTenMinuteReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
        model.SystemTenMinuteReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)
    
        # Thirty-minute operating reserve
        model.OperatingReserveAvailable = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)  #ISO-NE's Claim30 parameter
        model.OperatingReserveOffer = Param(model.ThermalGenerators, within=NonNegativeReals, default=0.0)
        model.ZonalOperatingReserveRequirement = Param(model.ReserveZones, model.TimePeriods, within=NonNegativeReals, default=0.0)
        model.SystemOperatingReserveRequirement = Param(model.TimePeriods, within=NonNegativeReals, default=0.0)
    
    #
    # STORAGE parameters
    #
    
    if storage_services:
    
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
        model.RetentionRate          = Param(model.Storage, within=PercentFraction, default=1.0)
    
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

    ## NOTE: these don't get used, just for dat files with these
    ModeratelyBigPenalty = 1e4
    
    model.ReserveShortfallPenalty = Param(within=NonNegativeReals, default=ModeratelyBigPenalty)

    #########################################
    # penalty costs for constraint violation #
    #########################################
    
    BigPenalty = 2e4
    
    model.LoadMismatchPenalty = Param(within=NonNegativeReals, default=BigPenalty)


    return model
