#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from typing import TypeVar

G = TypeVar('G') # Generator
B = TypeVar('B') # Bus
L = TypeVar('L') # Line
S = TypeVar('S') # Storage
OperationsModel = TypeVar('OperationsModel')
RucModel = TypeVar('RucModel')
ScenarioTree = TypeVar('ScenarioTree')
EgretModel = TypeVar('EgretModel')
