#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import prescient.simulator.prescient

def main(args=None):
    prescient.simulator.prescient.main(args)

if __name__ == '__main__':
    import sys
    main(sys.argv)
