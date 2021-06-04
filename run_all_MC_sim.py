# first, we'll use the built-in function to download the RTS-GMLC system to Prescicent/downloads/rts_gmlc
import prescient.downloaders.rts_gmlc as rts_downloader
import prescient.scripts.runner as runner
import os
import pandas as pd
import shutil

# the download function has the path Prescient/downloads/rts_gmlc hard-coded.
# We don't need the code below as long as we've already downloaded the RTS data into the repo (or run rts_gmlc.py)
# All it does is a 'git clone' of the RTS-GMLC repo
# rts_downloader.download()
# did_download = rts_downloader.download()
# if did_download:
#     rts_downloader.copy_templates()
# rts_downloader.populate_input_data()

# variables to adjust:
runs = 1
directory_out = "--output-directory=output"
dir_path = "./rts_gmlc"
new_path = "./working"
os.chdir("downloads/rts_gmlc")

# all zone 1 file paths
file_paths_zone1 = ['./timeseries_data_files/101_PV_1_forecasts_actuals.csv','./timeseries_data_files/101_PV_2_forecasts_actuals.csv',
              './timeseries_data_files/101_PV_3_forecasts_actuals.csv','./timeseries_data_files/101_PV_4_forecasts_actuals.csv',
              './timeseries_data_files/102_PV_1_forecasts_actuals.csv','./timeseries_data_files/102_PV_2_forecasts_actuals.csv',
              './timeseries_data_files/103_PV_1_forecasts_actuals.csv','./timeseries_data_files/104_PV_1_forecasts_actuals.csv',
              './timeseries_data_files/113_PV_1_forecasts_actuals.csv','./timeseries_data_files/118_RTPV_1_forecasts_actuals.csv',
              './timeseries_data_files/118_RTPV_2_forecasts_actuals.csv','./timeseries_data_files/118_RTPV_3_forecasts_actuals.csv',
              './timeseries_data_files/118_RTPV_4_forecasts_actuals.csv','./timeseries_data_files/118_RTPV_5_forecasts_actuals.csv',
              './timeseries_data_files/118_RTPV_6_forecasts_actuals.csv','./timeseries_data_files/118_RTPV_7_forecasts_actuals.csv',
              './timeseries_data_files/118_RTPV_8_forecasts_actuals.csv','./timeseries_data_files/118_RTPV_9_forecasts_actuals.csv',
              './timeseries_data_files/118_RTPV_10_forecasts_actuals.csv','./timeseries_data_files/119_PV_1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_101_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_102_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_103_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_104_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_105_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_106_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_107_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_108_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_109_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_110_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_111_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_112_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_113_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_114_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_115_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_116_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_117_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_118_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_119_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_120_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_121_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_122_Load_zone1_forecasts_actuals.csv',
              './timeseries_data_files/Bus_123_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_124_Load_zone1_forecasts_actuals.csv' ]

# smaller set for testing
file_paths_test = ['./timeseries_data_files/101_PV_1_forecasts_actuals.csv','./timeseries_data_files/101_PV_2_forecasts_actuals.csv']


def read_files(file_paths):
    # file_paths: list of strings indicating file paths that are to be read in
    # output: data_lst - list of data frames containing all the information in each file
    # Note: we add  to a list and then concatenate as this is faster and takes less memory than growing the dataframe
    # each time

    data_lst = []
    i = 0
    # iterate across file paths
    for path in file_paths:
        data = pd.read_csv(path) # read in the file

        # rename the columns to be useful
        # the numbers below are hard coded for this particular case - they will have to change if the file structure
        # changes too
        data.columns = ['Time', path[24:-22]+'_forecasts', path[24:-22]+'_actuals']

        # if this is our first one, append all columns (including date/time), otherwise, just append forecasts/actuals
        # note: this assumes that all files have the exact same dates and times, which is supported in this case, but
        # may not be true generally
        if i == 0:
            data_lst.append(data)
        else:
            data_lst.append(data[[path[24:-22]+'_forecasts', path[24:-22]+'_actuals']])
        i += 1
    return data_lst


all_data = pd.concat(read_files(file_paths_zone1), axis=1)  # read in the data into a the data frame
all_data.to_csv('test.csv')  # print out results as a test


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
    data['actuals'].values[:] = 0
    data.to_csv(path, index=False)


def copy_directory(index):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        shutil.copytree(dir_path, new_path)
    else:
        shutil.copytree(dir_path, new_path)
