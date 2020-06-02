#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


from prescient.models.model_generator import UCFormulation, get_model

## get the compact formulation
formulation = UCFormulation(
                            status_vars = 'garver_3bin_vars',
                            power_vars = 'garver_power_vars',
                            reserve_vars = 'garver_power_avail_vars',
                            generation_limits = 'MLR_generation_limits',
                            ramping_limits = 'CA_ramping_limits',
                            production_costs = 'basic_production_costs_envelope',
                            uptime_downtime = 'rajan_takriti_UT_DT',
                            startup_costs = 'MLR_startup_costs',
                            )

model = get_model(formulation)

## attach the helper functions simulator wants
from prescient.models.helper_functions import *
