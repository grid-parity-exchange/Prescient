#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## tools for generating models
import pyomo.environ as pe
from prescient.model_components import *
from collections import namedtuple

UCFormulation = namedtuple('UCFormulation',
                            ['status_vars',
                             'power_vars',
                             'reserve_vars',
                             'generation_limits',
                             'ramping_limits',
                             'production_costs',
                             'uptime_downtime',
                             'startup_costs',
                             ]
                            )
                               
                             

def generate_model( _status_vars,
                    _power_vars,
                    _reserve_vars,
                    _non_dispatchable_vars,
                    _generation_limits,
                    _ramping_limits,
                    _production_costs,
                    _uptime_downtime,
                    _startup_costs,
                    _power_balance, 
                    _reserve_requirement, 
                    _objective, 
                    _relax_binaries = False,
                    ):

    
    model = pe.AbstractModel()
    
    model.name = "PrescientUCModel"
    
    ## enforece time 1 ramp rates
    model.enforce_t1_ramp_rates = True

    ## prescient assumes we have storage services
    model.storage_services = True

    ## new flag for ancillary services
    model.ancillary_services = False
    
    ## to relax binaries
    model.relax_binaries = _relax_binaries

    data_loader.load_basic_data(model)
    getattr(status_vars, _status_vars)(model)
    getattr(power_vars, _power_vars)(model)
    getattr(reserve_vars, _reserve_vars)(model)
    getattr(non_dispatchable_vars, _non_dispatchable_vars)(model)
    getattr(generation_limits, _generation_limits)(model)
    getattr(ramping_limits, _ramping_limits)(model)
    getattr(production_costs, _production_costs)(model)
    getattr(uptime_downtime, _uptime_downtime)(model)
    getattr(startup_costs, _startup_costs)(model)
    if model.storage_services:
        services.storage_services(model)
    if model.ancillary_services:
        services.ancillary_services(model)
    getattr(power_balance, _power_balance)(model)
    getattr(reserve_requirement, _reserve_requirement)(model)
    getattr(objective, _objective)(model)

    return model

def get_formulation_from_UCFormulation( uc_formulation ):
    return [  uc_formulation.status_vars,
              uc_formulation.power_vars,
              uc_formulation.reserve_vars,
              'file_non_dispatchable_vars',
              uc_formulation.generation_limits,
              uc_formulation.ramping_limits,
              uc_formulation.production_costs,
              uc_formulation.uptime_downtime,
              uc_formulation.startup_costs,
              'power_balance_constraints',
              'MLR_reserve_constraints' if uc_formulation.reserve_vars in ['MLR_reserve_vars'] else 'CA_reserve_constraints',
              'basic_objective',
            ]


def get_model( uc_formulation, relax_binaries = False ):
    '''
    returns a UC uc_formulation as an abstract model with the 
    components specified in a UCFormulation, with the option
    to relax the binary variables
    '''

    return generate_model( *get_formulation_from_UCFormulation( uc_formulation ), relax_binaries )

