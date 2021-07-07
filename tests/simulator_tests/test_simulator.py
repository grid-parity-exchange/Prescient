#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

### basic prescient simulator test

## "short" does a week of RTS-GMLC for each of the four seasons
## "long" runs the entire year of RTS-GMLC
import sys
import os
import subprocess
from prescient.scripts.runner import parse_line
from filecmp import cmpfiles

base_options = [ 
                '--data-directory=rts_gmlc_deterministic_with_network_scenarios_year',
                '--simulate-out-of-sample',
                '--run-sced-with-persistent-forecast-errors',
                '--model-directory=..|..|prescient|models|tight',
                '--run-deterministic-ruc',
                '--sced-horizon=4',
                '--random-seed=10',
                '--output-max-decimal-places=4',
                '--output-sced-initial-conditions',
                '--output-sced-demands',
                '--output-sced-solutions',
                '--output-ruc-initial-conditions',
                '--output-ruc-solutions',
                '--output-ruc-dispatches',
                '--output-solver-logs',
                '--ruc-mipgap=0.005',
                '--symbolic-solver-labels',
                '--reserve-factor=0.03',
                '--deterministic-ruc-solver=cbc',
                '--deterministic-ruc-solver-options="method=1 cuts=2 symmetry=2 presolve=2 varbranch=1"',
                '--sced-solver=cbc',
                '--sced-solver-options="threads=1"',
                '--compute-market-settlements',
                '--price-threshold=1000',
                '--reserve-price-threshold=250',
                ]

output_files = [
                 'Daily_summary.csv',
                 'Hourly_gen_summary.csv',
                 'Options.csv',
                 'Overall_simulation_output.csv',
                 'Quickstart_summary.csv',
                 'bus_detail.csv',
                 'line_detail.csv',
                 'renewables_detail.csv',
                 'thermal_detail.csv',
               ]

date_season_tuples = [ ('01-08-2020','winter'),
                       ('04-08-2020','spring'),
                       ('07-08-2020','summer'),
                       ('10-08-2020','fall')  ]

short_dir = 'rts_gmlc_deterministic_with_network_simulation_output_week_'
long_dir = 'rts_gmlc_deterministic_with_network_simulation_output_year'

short_dir_baseline = 'rts_gmlc_deterministic_with_network_simulation_output_week_baseline_'
long_dir_baseline = 'rts_gmlc_deterministic_with_network_simulation_output_year_baseline'

def outputs_identical(dir_base, dir_results):
    match, mismatch, errors = cmpfiles(dir_base, dir_results, output_files, shallow=False)

    for file_name in mismatch:
        print("Error: There is a difference in summary file {}".format(file_name))

    for file_name in errors:
        print("Error: The file {} was not found in the results directory {}".format(file_name, dir_results))

    if len(mismatch) + len(errors) == 0:
        print("All files the same")
        return True
    return False

def get_long_options_strings(dir_base):
    return base_options + \
           [ '--output-directory='+dir_base, 
             '--start-date=01-02-2020',
             '--num-days=364',
           ] 

def get_short_options_strings(dir_base,date,season):
    return base_options + \
           [ '--output-directory='+dir_base+season, 
             '--start-date='+date,
             '--num-days=7',
           ] 

def run_populator():
    run_list = ['runner.py', 'populate_with_network_deterministic_year.txt']
    if sys.platform.startswith('win'):
        ret = subprocess.call(run_list, shell=True)
    else:
        ret = subprocess.call(run_list)
    if ret:
        raise Exception('Issue running populator')

def populate_input_data():
    from process_RTS_GMLC_data import create_timeseries
    from rtsgmlc_to_dat import write_template

    print("\nGathering time series data...")
    create_timeseries()

    print("\nWriting dat template file...")
    write_template(rts_gmlc_dir='../../downloads/rts_gmlc/RTS-GMLC', 
                   file_name= './templates/rts_with_network_template_hotstart.dat' )

    print("\nRunning the populator...")
    run_populator()

def populate_if_necessary():
    if os.path.exists('./rts_gmlc_deterministic_with_network_scenarios_year/'):
        return
    populate_input_data()
     
def run_long_test():
    populate_if_necessary()
    output_file = open(long_dir+"_tee.out", "w")
    print('Running simulator for year')
    sim_ret = run_simulator(get_long_options_strings(long_dir), output_file)
    if sim_ret:
        raise Exception('Issue running simulator for year')
    if outputs_identical(long_dir_baseline, long_dir):
        print("No issue found in year-long run")
    else:
        print("Issues found in year-long run. See above") 

def populate_long_baseline():
    populate_if_necessary()
    sim_ret = run_simulator(get_long_options_strings(long_dir_baseline))
    if sim_ret:
        raise Exception('Issue running simulator for year')

def run_short_test():
    populate_if_necessary()
    for date,season in date_season_tuples:
        output_file = open(short_dir+season+"_tee.out", "w")
        print('Running simulator for week in season {}'.format(season))
        sim_ret = run_simulator(get_short_options_strings(short_dir,date,season), output_file)
        if sim_ret:
            raise Exception('Issue running simulator for week in season {}'.format(season))
        dir_base = short_dir_baseline + season
        dir_results = short_dir + season
        if outputs_identical(dir_base, dir_results):
            print("No issues found for week in "+season)
        else:
            print("Issues found for week in "+season+". See above.")
	
def populate_short_baseline():
    populate_if_necessary()
    for date,season in date_season_tuples:
        sim_ret = run_simulator(get_short_options_strings(short_dir_baseline,date,season))
        if sim_ret:
            raise Exception('Issue running simulator for week in season {}'.format(season))

def get_subprocess_options(options_strings):
    options = []
    for opt_str in options_strings:
        options.extend(parse_line(opt_str))
    return options

def run_simulator(options_strings, output_file=None):
    options = get_subprocess_options(options_strings)
    if sys.platform.startswith('win'):
        ret = subprocess.call(['simulator.py']+options, shell=True, stdout=output_file, stderr=output_file)
    else:
        ret = subprocess.call(['simulator.py']+options, stdout=output_file, stderr=output_file)
    return ret

def display_usage_and_exit():
    print("You must specify either 'short' or 'long' after the program name")
    print("Usage: test_simulator.py short")
    print("or")
    print("Usage: test_simulator.py long")
    print("The 'short' option runs the simulator for four different weeks against")
    print("a baseline, and the 'long' option runs the simulator for an entire year")
    print("against a baseline.")
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        display_usage_and_exit()

    command = sys.argv[1].lower()

    if command == 'short':
        run_short_test()
    elif command == 'long':
        run_long_test()
    else:
        display_usage_and_exit()

if __name__ == '__main__':
    main()
