#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


from prescient.models.model_generator import UCFormulation, get_model

# get the fast? formulation
formulation = UCFormulation(
                            status_vars = 'garver_3bin_vars',
                            power_vars = 'garver_power_vars',
                            reserve_vars = 'garver_power_avail_vars',
                            generation_limits = 'gentile_generation_limits',
                            ramping_limits = 'MLR_ramping',
                            production_costs = 'CA_production_costs',
                            uptime_downtime = 'rajan_takriti_UT_DT_2bin',
                            startup_costs = 'KOW_startup_costs',
                            )

model = get_model(formulation)

## attach the helper functions simulator wants
from prescient.models.helper_functions import *
