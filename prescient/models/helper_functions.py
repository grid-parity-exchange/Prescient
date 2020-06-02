#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


#
# utilities to free/fix all binary variables in the model - present values are retained / not manipulated.
#

def status_var_generator(instance):
    ## NOTE: BK sorted here so we can step through two instances at the same time using zip()
    for t in sorted(instance.TimePeriods):
        for g in sorted(instance.ThermalGenerators):
            if instance.status_vars in ['CA_1bin_vars', 'garver_3bin_vars', 'garver_2bin_vars', 'garver_3bin_relaxed_stop_vars']:
                yield instance.UnitOn[g,t]
            if instance.status_vars in ['ALS_state_transition_vars']:
                yield instance.UnitStayOn[g,t]

def _binary_var_generator(instance):

    for t in instance.TimePeriods:
        for g in instance.ThermalGenerators:
            if instance.status_vars in ['CA_1bin_vars', 'garver_3bin_vars', 'garver_2bin_vars', 'garver_3bin_relaxed_stop_vars']:
                yield instance.UnitOn[g,t]
            if instance.status_vars in ['ALS_state_transition_vars']:
                yield instance.UnitStayOn[g,t]
            if instance.status_vars in ['garver_3bin_vars', 'garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars']:
                yield instance.UnitStart[g,t]
            if instance.status_vars in ['garver_3bin_vars', 'ALS_state_transition_vars']:
                yield instance.UnitStop[g,t]
            
            if instance.ancillary_services:
                yield instance.RegulationOn[g,t]
                
        if instance.storage_services:
            for s in instance.Storage:
                yield instance.OutputStorage[s,t]

    if instance.startup_costs in ['KOW_startup_costs']:
        for g,t_prime,t in instance.StartupIndicator_domain:
            yield instance.StartupIndicator[g,t_prime,t]
    elif instance.startup_costs in ['MLR_startup_costs', 'MLR_startup_costs2',]:
        for g,s,t in instance.StartupCostsIndexSet:
            yield instance.delta[g,s,t]

def fix_binary_variables(instance):
    for var in _binary_var_generator(instance):
        var.fix()


def free_binary_variables(instance):
    for var in _binary_var_generator(instance):
        var.unfix()

def define_suffixes(instance):
    from pyomo.environ import Suffix

    instance.dual = Suffix(direction=Suffix.IMPORT)


def _del_pyomo_components(instance, pyomo_components):
    for component_name in pyomo_components:
        component = getattr(instance, component_name) 
        instance.del_component(component)
        if hasattr(instance, component_name+'_index'):
            component_index = getattr(instance, component_name+'_index')
            instance.del_component(component_index)


def reconstruct_instance_for_pricing(instance, preprocess=True):
    if instance.reserve_requirement in ['MLR_reserve_constraints']:
        return
    ## else we'll rebuild it
    from prescient.model_components.reserve_requirement import _MLR_reserve_constraint 

    reserve_component = ['EnforceReserveRequirements']

    _del_pyomo_components(instance, reserve_component)

    _MLR_reserve_constraint(instance)

    if preprocess:
        instance.preprocess()

def load_model_parameters():

    import pyomo.environ as pe
    from prescient.model_components import data_loader

    model = pe.AbstractModel()
    
    model.name = "PrescientUCModelParameters"
    
    ## enforece time 1 ramp rates
    model.enforce_t1_ramp_rates = True

    ## prescient assumes we have storage services
    model.storage_services = True
    model.regulation_services = False
    model.reserve_services = False

    data_loader.load_basic_data(model)

    return model

# 
# a utility to reconstruct an instance if T0 and related dynamic parameters are changed.
#
# NOTE: isn't used by the current version of the simulator
def reconstruct_instance_for_t0_changes(instance):
    from prescient.model_components.data_loader import _add_initial_time_periods_on_off_line, _populate_reserve_requirements, _verify_must_run_t0_state_consistency
    import prescient.model_components.ramping_limits as ramping_limits_formulations
    import prescient.model_components.uptime_downtime as uptime_downtime_formulations
    import prescient.model_components.startup_costs as startup_costs_formulations 
    from prescient.model_components.objective import _add_shutdown_costs

    # NOTE: the following is really ugly, but the reconstruct() method in the current version of Pyomo doesn't work.

    ## intial online/offline
    pyomo_components = [ 'InitialTimePeriodsOnLine', 'InitialTimePeriodsOffLine', 'EnforceUpTimeConstraintsInitial', 'EnforceDownTimeConstraintsInitial', \
                                 'Logical', 'UpTime', 'DownTime', 'EnforceMustRun']

    pyomo_components += ['PopulateReserveRequirements', 'VerifyMustRunT0StateConsistency']

    ## ramping limits
    pyomo_components += [ 'EnforceMaxAvailableRampUpRates', 'EnforceScaledNominalRampDownLimits', ]
    if instance.ramping_limits in ['MLR_ramping']:
        pyomo_components.append('power_limit_t0_stop')
    elif instance.ramping_limits in ['OAV_ramping_enhanced_2period']:
        pyomo_components += ['OAVTwoPeriodRampUp', 'OAVTwoPeriodRampDown']
    elif instance.ramping_limits in ['damcikurt_ramping_2period']:
        pyomo_components += ['EnforceTwoPeriodRampUpRule', 'EnforceTwoPeriodRampDownRule']

    ## startup costs
    if instance.startup_costs in ['KOW_startup_costs']:
        pyomo_components += ['ValidShutdownTimePeriods', 'ShutdownHotStartupPairs', 'StartupIndicator_domain', 'StartupIndicator',\
                                    'StartupMatch', 'GeneratorShutdownPeriods', 'ShutdownMatch' ]
    elif instance.startup_costs in ['MLR_startup_costs', 'MLR_startup_costs2']:
        pyomo_components += ['StartupCostsIndexSet', 'delta', 'delta_eq', 'delta_ineq',]
    elif instance.startup_costs in ['KOW_3bin_startup_costs', 'CA_SHB_startup_costs','CA_startup_costs', ]:
        pyomo_components += ['StartupCostsIndexSet', ]
    elif instance.startup_costs in ['ALS_startup_costs', 'YZJMXD_startup_costs', 'KOW_3bin_startup_costs2', ]:
        pyomo_components += ['StartupCostsIndexSet', 'StartupCostOverHot', 'ComputeStartupCostsOverHot']
    pyomo_components += ['ComputeStartupCosts']

    ## let's try rebuilding the objective too...
    pyomo_components += ['ComputeShutdownCosts']

    _del_pyomo_components(instance, pyomo_components)        

    # IMPT: Due to the comment above, the definitions below should be identical to those in the main model definition.
    _verify_must_run_t0_state_consistency(instance)
    _populate_reserve_requirements(instance)
    _add_initial_time_periods_on_off_line(instance)

    # just rebuild the uptime/downtime limits
    uptime_downtime_formulation = instance.uptime_downtime
    del instance.uptime_downtime
    getattr(uptime_downtime_formulations, uptime_downtime_formulation)(instance)

    # just rebuild the ramping limits
    ramping_limits_formulation = instance.ramping_limits
    del instance.ramping_limits
    getattr(ramping_limits_formulations, ramping_limits_formulation)(instance)

    # rebuild the startup cost
    startup_costs_formulation = instance.startup_costs
    del instance.startup_costs
    getattr(startup_costs_formulations, startup_costs_formulation)(instance, add_startup_cost_var=False)

    # rebuild shutdown cost
    _add_shutdown_costs(instance, add_shutdown_cost_var=False)

    instance.preprocess()
