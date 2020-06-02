#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os

s1 ="""command/exec scenario_creator.py

# Options regarding file in- and output:
--sources-file {sources_file}
--output-directory {output_directory}
--hyperrectangles-file {hyperrectangles_file}
--dps-file {dps_file}
--daps-location {daps_location}
--scenario-template-file {scenario_template_file}
--tree-template-file {tree_template_file}

# Scaling options
--wind-frac-nondispatch={wind_frac_nondispatch2}

# Options regarding the univariate epi-spline distribution:
--seg-N {seg_N}
--seg-kappa {seg_kappa}
--probability-constraint-of-distributions {probability_constraint_of_distributions}
--non-negativity-constraint-distributions {non_negativity_constraint_distributions}
--nonlinear-solver {nonlinear_solver}
--error-distribution-domain {error_distribution_domain}

# Options regarding all distributions:
--plot-variable-gap {plot_variable_gap}
--plot-pdf {plot_pdf}
--plot-cdf {plot_cdf}
--cdf-inverse-tolerance {cdf_inverse_tolerance}
"""

s2 ="""
Horse: {horse_name}
    
Populator Options:
--start-date {start_date}
--end-date {end_date}
--sources-file gosm_test/bpa_sourcelist.csv
--wind-scaling-factor={wind_factor}
--load-scaling-factor={load_scaling_factor}
--output-directory {populator_output}
--scenario-creator-options-file {scenario_options_file}
--allow-multiprocessing 1
--traceback

Simulator Options:
--data-directory {populator_output}
--simulate-out-of-sample
--model-directory .
--solver gurobi
--plot-individual-generators
--traceback
--output-sced-initial-conditions
--output-sced-demands
--output-sced-solutions
--output-ruc-initial-conditions
--output-ruc-solutions
--output-ruc-dispatches
--output-directory {simulator_output}
"""

#Wind Scaling Factors
wsArray = ('0.1')

#Cutpoint options
cutpointArray = ('../cutpoints/SC1_cutpoints.dat', '../cutpoints/BigExtreme.dat')
cutpointArraynames = ('SC1_cutpoints.dat', 'BigExtreme.dat')
cutpointArraynames2 = ('SC1_cutpoints', 'BigExtreme')

#General populator options
name = 'different wind scaling and cutpoints'
start_date_opt = '2013-01-01'
end_date_opt = '2013-01-30'
wind_frac_nondispatch_opt = 1.0
load_scaling_factor_opt = 0.045

#General scenario creator options
sources_file_opt = 'gosm_test/bpa_sourcelist.csv'
output_directory_opt = 'gosm_test/output_scenario_creator'
hyperrectangles_file_opt = '../cutpoints/hyperrectangle_names_1source.dat'
daps_location_opt = '../daps'
scenario_template_file_opt = 'gosm_test/simple_nostorage_skeleton.dat'
tree_template_file_opt = 'gosm_test/TreeTemplate.dat'

# Scaling options
load_scaling_factor_opt2 = 0.045
wind_frac_nondispatch_opt2=0.50

# Options regarding the univariate epi-spline distribution:
seg_N_opt = 20
seg_kappa_opt = 100
probability_constraint_of_distributions_opt = 1
non_negativity_constraint_distributions_opt = 0
nonlinear_solver_opt = 'ipopt'
error_distribution_domain_opt = 4

# Options regarding all distributions:
plot_variable_gap_opt = 10
plot_pdf_opt = 1
plot_cdf_opt = 0
cdf_inverse_tolerance_opt = 1.0e-3



horse_options =""
if not os.path.exists("horse_populator_output"):
    os.mkdir("horse_populator_output")
if not os.path.exists("horse_simulator_output"):
    os.mkdir("horse_simulator_output")
if not os.path.exists("cutpointsets"):
	os.mkdir("cutpointsets")


for i in range(7):
	for j in range(2):
		#write specified options to different "option" files
		scenario_options = s1.format(dps_file=cutpointArray[j],
                                     sources_file=sources_file_opt,
                                     output_directory=output_directory_opt,
                                     hyperrectangles_file=hyperrectangles_file_opt,
                                     daps_location=daps_location_opt,
                                     scenario_template_file=scenario_template_file_opt,
                                     tree_template_file=tree_template_file_opt,
                                     load_scaling_factor2=load_scaling_factor_opt2,
                                     wind_frac_nondispatch2=wind_frac_nondispatch_opt2,
                                     seg_N=seg_N_opt,
                                     seg_kappa=seg_kappa_opt,
                                     probability_constraint_of_distributions=probability_constraint_of_distributions_opt,
                                     non_negativity_constraint_distributions=non_negativity_constraint_distributions_opt,
                                     nonlinear_solver=nonlinear_solver_opt,
                                     error_distribution_domain=error_distribution_domain_opt,
                                     plot_variable_gap=plot_variable_gap_opt,
                                     plot_pdf=plot_pdf_opt,
                                     plot_cdf=plot_cdf_opt,
                                     cdf_inverse_tolerance=cdf_inverse_tolerance_opt)
                                     
		script_options_file = 'cutpointsets/options_file{}'.format(wsArray+cutpointArraynames[j])
		f = open(script_options_file, 'w')
		f.write(scenario_options)
		
		#specify output directories/file names
		horse_pop = 'horse_populator_output/{}'.format(wsArray+cutpointArraynames2[j])
		horse_sim = 'horse_simulator_output/{}'.format(wsArray+cutpointArraynames2[j])
		name = 'different wind scaling and cutpoints {}'.format(wsArray+cutpointArraynames[j])
		
		# creating the general file
		horse_options += str(s2.format(scenario_options_file=script_options_file,
										   horse_name=name,
										   populator_output=horse_pop,
										   simulator_output=horse_sim,
										   start_date=start_date_opt,
										   end_date=end_date_opt,
										   wind_factor=wsArray,
										   wind_frac_nondispatch=wind_frac_nondispatch_opt,
										   load_scaling_factor=load_scaling_factor_opt))
		horse_output = 'horse_racer_configuration.txt'
		f = open(horse_output, 'w')

f.write(horse_options)




