#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# This file just identifies the data type used to pass options within prescient.
# If we ever change options parsers, we can change the options type once here instead of
# changing it everywhere.

import optparse 

Options = optparse.Values

