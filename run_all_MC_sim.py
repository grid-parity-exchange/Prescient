# first, we'll use the built-in function to download the RTS-GMLC system to Prescicent/downloads/rts_gmlc
import prescient.downloaders.rts_gmlc as rts_downloader
import prescient.scripts.runner as runner
import os
import pandas as pd
import shutil
import numpy as np

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

bus_names = [] # global variable that will contain a list of bus names. this will be populated in read_files and used elsewhere

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
        bus_names.append(path[24:-22]) # gives us a list of bus_names which we can use later on
        # if this is our first one, append all columns (including date/time), otherwise, just append forecasts/actuals
        # note: this assumes that all files have the exact same dates and times, which is supported in this case, but
        # may not be true generally
        if i == 0:
            data_lst.append(data)
        else:
            data_lst.append(data[[path[24:-22]+'_forecasts', path[24:-22]+'_actuals']])
        i += 1
    return data_lst


def filter_no_solar(combined_data):
    # combined_data: data frame of all forecasts and actuals for a list of buses
    # output: two data frames called s_data and ns_data.
    # This function filters all data into two parts - one where solars are active and one where solars are inactive
    # we will do this in a pretty naive way, simply based on one of the solar plants, which we are going to hard code
    # this is not ideal, but it should do for now

    determining_solar_plant = '101_PV_1'  # this is the solar plant that will determine the 'no solar' case

    ns_data = combined_data[combined_data[determining_solar_plant + '_forecasts'] == 0]
    #ns_data.to_csv('zz_no_solar_data.csv')  # print out results as a test

    s_data = combined_data[combined_data[determining_solar_plant + '_forecasts'] != 0]
    #s_data.to_csv("zz_solar_data.csv")

    return ns_data, s_data


def compute_actual_forecast_quotient(data):
    # data: data frame of forecasts and actuals, in the pattern of: forecast, actual
    # output: modified version of data containing additional columns with the quotient of actual / forecasts

    # iterate across bus names and take the relevant quotients
    for name in bus_names:
        temp_nm = name + '_quotient'
        data = data.assign(temp_nm=data[name+'_actuals'] / data[name+'_forecasts'])
        data.rename(columns={'temp_nm':temp_nm}, inplace=True)

    # get rid of NaNs and Infs
    # NaNs arise when we have 0/0, Infs arrive when we have x / 0, where x > 0
    data.fillna(0, inplace=True)
    data.replace(np.inf, 0, inplace=True)
    return data


def sample_quotients(pre_sunrise_hrs, post_sunset_hrs, s_data, ns_data):
    # pre_sunrise_hrs: number of hours before sunrise for the day we want to sample
    # post_sunset_hrs: number of hours after sunset for the day we want to sample
    # s_data: data frame of the active solar hours
    # ns_data: data frame of the inactive solar hours
    ns_quotients = ns_data.filter(regex='quotient$', axis=1)
    s_quotients = s_data.filter(regex='quotient$', axis=1)
    pre_sunrise_sample = ns_quotients.sample(pre_sunrise_hrs, replace=True)  # samples quotients for pre sunrise hours
    post_sunset_sample = ns_quotients.sample(post_sunset_hrs, replace=True)  # samples quotients for post sunset hours
    # samples quotients for daylight hours
    daylight_sample = s_quotients.sample(24 - pre_sunrise_hrs - post_sunset_hrs, replace=True)
    frames = [pre_sunrise_sample, daylight_sample, post_sunset_sample]
    day_sample = pd.concat(frames)
    return day_sample



all_data = pd.concat(read_files(file_paths_zone1), axis=1)  # read in the data into a the data frame
#all_data.to_csv('zz_all_data.csv')  # print out results as a test
no_solar_data, solar_data = filter_no_solar(all_data)
solar_data = compute_actual_forecast_quotient(solar_data)
no_solar_data = compute_actual_forecast_quotient(no_solar_data)

quotients_0710 = sample_quotients(6, 5, solar_data, no_solar_data)  # sampling the day in question
quotients_0709 = sample_quotients(6, 5, solar_data, no_solar_data)  # sampling the day before
quotients_0711 = sample_quotients(6, 5, solar_data, no_solar_data)  # sampling the day after

# need to apply the quotients to the proper forecasts and write to file in the format that is readable to prescient
# only need to write 1 day on either end of July 10 for now.


# the functions below are currently not used in the script above, but may be useful when we run prescient with the
# modified files
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
