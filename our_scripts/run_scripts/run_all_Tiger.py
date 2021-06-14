# run_all_tiger.py: simple way to run batch operations on tiger
# authors: Ethan Reese
# email: ereese@princeton.edu
# created: June 14, 2021

import os
import run_helpers as rh

path_template = "./scenario_"
solar_path = "./solar_quotients.csv"
no_solar_path = "./no_solar_quotients.csv"
runs = 1000


def run(i):
        rh.copy_directory(i, path_template)
        os.chdir(path_template+str(i))
        rh.perturb_data(rh.file_paths_combined, solar_path, no_solar_path)
        rh.run_prescient(i, True)
        os.chdir("..")


# program body
os.chdir("..")
os.chdir("..")
os.chdir("./downloads")

for j in range(runs):
        run(j)