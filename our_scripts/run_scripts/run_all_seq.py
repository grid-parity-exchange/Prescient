# run_all_seq.py: testing version of script to run Prescient on local machine with fixed changes
# authors: Ethan Reese, Arvind Shrivats
# email: ereese@princeton.edu, shrivats@princeton.edu
# created: June 8, 2021

# first, we'll use the built-in function to download the RTS-GMLC system to Prescicent/downloads/rts_gmlc
import prescient.downloaders.rts_gmlc as rts_downloader
import prescient.scripts.runner as runner
import os
import pandas as pd
import shutil

# the download function has the path Prescient/downloads/rts_gmlc hard-coded.
# All it does is a 'git clone' of the RTS-GMLC repo
# rts_downloader.download()
# did_download = rts_downloader.download()
# if did_download:
#     rts_downloader.copy_templates()
# rts_downloader.populate_input_data()
os.chdir("..")
os.chdir("..")
# variables to adjust:
runs = 1
directory_out = "--output-directory=output"
dir_path = "./rts_gmlc"
new_path = "./working"


# helper functions

def run_prescient(index, populate='populate_with_network_deterministic.txt',
                  simulate='simulate_with_network_deterministic.txt'):
    with open(simulate, "r") as file:
        lines = file.readlines()
    with open(simulate, "w") as file:
        for line in lines:
            if (line.startswith("--output-directory=")):
                file.write(directory_out + "\n")
            elif (line.startswith("--num-days")):
                file.write("--num-days=1 \n")
            elif (line.startswith("--random-seed") or line.startswith("--output-sced-solutions") or line.startswith(
                    "--output-ruc-dispatches")):
                continue
            else:
                file.write(line)
    runner.run(populate)
    runner.run(simulate)


def modify_file(path):
    data = pd.read_csv(path)
    # placeholder modification -> could easily be replaced
    data['actuals'] = data['forecasts']
    data.to_csv(path, index=False)


def copy_directory(index):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        shutil.copytree(dir_path, new_path)
    else:
        shutil.copytree(dir_path, new_path)


def run(i):
    copy_directory(i)
    os.chdir("./working")
    modify_file("./timeseries_data_files/101_PV_" + str(i + 1) + "_forecasts_actuals.csv")
    run_prescient(i)
    os.chdir("..")
    src = './working/output'
    dst = './scenario_' + str(i + 1)
    if os.path.exists(dst):
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copytree('./working/output', './scenario_' + str(i + 1))
    shutil.rmtree('./working')


os.chdir("downloads")
for i in range(runs):
    run(i)

# Iterate over each source file and tweak
# try by making one solar asset zero

# run prescient for each using two lines above

# copy all the csv outputs to conserve the data -> label with input
