#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# For now, OperationsStats and HourlyStats are the same class.  In the future, 
# when operations may be done more frequently than once per hour, OperationsStats
# will hold the results of a single sced optimization.
from prescient.stats.hourly_stats import HourlyStats
OperationsStats = HourlyStats

