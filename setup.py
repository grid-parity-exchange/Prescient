#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#!/bin/usr/env python
import sys

# We raise an error if trying to install with less than python 3.7
if sys.version_info < (3,7):
    sys.exit("This package requires Python 3.7 or greater")
if sys.version_info >= (4,0):
    sys.exit("Support for Python 4 is undetermined")

from setuptools import setup, find_namespace_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

packages = find_namespace_packages(include=['prescient.*'])

setup(name='gridx-prescient',
      version='2.2.2',
      description='Power Generation Scenario creation and simulation utilities',
      url='https://github.com/grid-parity-exchange/Prescient',
      author='Jean-Paul Watson, David Woodruff, Andrea Staid, Dominic Yang',
      author_email='jwatson@sandia.gov',
      python_requires='>=3.7, <4',
      packages=packages,
      long_description=long_description,
      long_description_content_type='text/markdown',
      entry_points={
            'console_scripts': [
                'runner.py = prescient.scripts.runner:main',
                'populator.py = prescient.scripts.populator:main',
                'scenario_creator.py = prescient.scripts.scenario_creator:main',
                'simulator.py = prescient.scripts.simulator:main',
            ]
        },
      package_data={'prescient.downloaders.rts_gmlc_prescient':['runners/*.txt','runners/templates/*'],
                    'prescient.simulator.tests':['regression_tests_data/**/*'],
                   },
      install_requires=['numpy','matplotlib','pandas','scipy','pyomo>=6.1.2',
                        'python-dateutil','networkx','jupyter', 'gridx-egret==0.5.5',
                       ],
     )
