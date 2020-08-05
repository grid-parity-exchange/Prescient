#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#!/bin/usr/env python
import glob
import sys
import os

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(include=['prescient.*'])

setup(name='prescient',
      version='2.0',
      description='Power Generation Scenario creation and simulation utilities',
      url='https://github.com/jwatsonnm/prescient',
      author='Jean-Paul Watson, David Woodruff, Andrea Staid, Dominic Yang',
      author_email='jwatson@sandia.gov',
      packages=packages,
      entry_points={
            'console_scripts': [
                'runner.py = prescient.scripts.runner:main',
                'populator.py = prescient.scripts.populator:main',
                'scenario_creator.py = prescient.scripts.scenario_creator:main',
                'simulator.py = prescient.scripts.simulator:main',
                'prescient.py = prescient.simulator.prescient:main'
            ]
        },
      package_data={'prescient.downloaders.rts_gmlc_prescient':['runners/*.txt','runners/templates/*']},
      install_requires=['numpy','matplotlib','pandas','scipy','pyomo','six',
                        'pyutilib', 'python-dateutil', 'networkx']
     )
