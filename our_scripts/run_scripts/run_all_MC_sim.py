# run_all_MC_sim.py: testing version of script to run Prescient on local machine with sampling
# authors: Ethan Reese, Arvind Shrivats
# email: ereese@princeton.edu, shrivats@princeton.edu
# created: June 8, 2021

# first, we'll use the built-in function to download the RTS-GMLC system to Prescicent/downloads/rts_gmlc
import prescient.downloaders.rts_gmlc as rts_downloader
import prescient.scripts.runner as runner
import os
import pandas as pd
import shutil
import numpy as np
import time


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
path_template = "./scenario_"

# all zone 1 file paths
file_paths_combined = ['./timeseries_data_files/101_PV_1_forecasts_actuals.csv','./timeseries_data_files/101_PV_2_forecasts_actuals.csv',
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
              './timeseries_data_files/Bus_123_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_124_Load_zone1_forecasts_actuals.csv','./timeseries_data_files/Bus_214_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_223_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/215_PV_1_forecasts_actuals.csv', './timeseries_data_files/Bus_210_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/213_RTPV_1_forecasts_actuals.csv', './timeseries_data_files/Bus_218_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/222_HYDRO_2_forecasts_actuals.csv', './timeseries_data_files/Bus_207_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/201_HYDRO_4_forecasts_actuals.csv', './timeseries_data_files/Bus_203_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_204_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/RTPV_zone2_forecasts_actuals.csv',
             './timeseries_data_files/215_HYDRO_3_forecasts_actuals.csv', './timeseries_data_files/Hydro_zone2_forecasts_actuals.csv',
             './timeseries_data_files/222_HYDRO_4_forecasts_actuals.csv', './timeseries_data_files/215_HYDRO_1_forecasts_actuals.csv',
             './timeseries_data_files/Bus_217_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_220_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_208_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/222_HYDRO_6_forecasts_actuals.csv',
             './timeseries_data_files/Bus_213_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_224_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_202_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_219_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_206_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/222_HYDRO_1_forecasts_actuals.csv',
             './timeseries_data_files/Bus_211_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/222_HYDRO_3_forecasts_actuals.csv',
             './timeseries_data_files/Bus_222_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_215_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/222_HYDRO_5_forecasts_actuals.csv', './timeseries_data_files/Bus_212_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_221_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_216_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/PV_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_209_Load_zone2_forecasts_actuals.csv',
             './timeseries_data_files/215_HYDRO_2_forecasts_actuals.csv', './timeseries_data_files/Load_zone2_forecasts_actuals.csv',
            './timeseries_data_files/Bus_201_Load_zone2_forecasts_actuals.csv', './timeseries_data_files/Bus_205_Load_zone2_forecasts_actuals.csv',
            './timeseries_data_files/Bus_309_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/320_RTPV_2_forecasts_actuals.csv',
             './timeseries_data_files/Bus_316_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_321_Load_zone3_forecasts_actuals.csv',
             './timeseries_data_files/313_PV_2_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_7_forecasts_actuals.csv',
             './timeseries_data_files/313_RTPV_10_forecasts_actuals.csv', './timeseries_data_files/310_PV_1_forecasts_actuals.csv',
             './timeseries_data_files/Bus_312_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_325_Load_zone3_forecasts_actuals.csv',
             './timeseries_data_files/Bus_305_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/309_WIND_1_forecasts_actuals.csv',
             './timeseries_data_files/313_RTPV_5_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_12_forecasts_actuals.csv',
             './timeseries_data_files/314_PV_2_forecasts_actuals.csv', './timeseries_data_files/Bus_301_Load_zone3_forecasts_actuals.csv',
             './timeseries_data_files/314_PV_4_forecasts_actuals.csv', './timeseries_data_files/PV_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_306_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_319_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/322_HYDRO_1_forecasts_actuals.csv',
            './timeseries_data_files/320_RTPV_6_forecasts_actuals.csv', './timeseries_data_files/324_PV_3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_302_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_315_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_322_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_1_forecasts_actuals.csv',
            './timeseries_data_files/308_RTPV_1_forecasts_actuals.csv', './timeseries_data_files/322_HYDRO_3_forecasts_actuals.csv',
            './timeseries_data_files/324_PV_1_forecasts_actuals.csv', './timeseries_data_files/317_WIND_1_forecasts_actuals.csv',
            './timeseries_data_files/313_RTPV_9_forecasts_actuals.csv', './timeseries_data_files/Bus_311_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/320_RTPV_4_forecasts_actuals.csv', './timeseries_data_files/Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/322_HYDRO_4_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_6_forecasts_actuals.csv',
            './timeseries_data_files/314_PV_1_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_11_forecasts_actuals.csv',
            './timeseries_data_files/303_WIND_1_forecasts_actuals.csv', './timeseries_data_files/320_RTPV_3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_304_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_324_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/WIND_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_313_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/310_PV_2_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_4_forecasts_actuals.csv',
            './timeseries_data_files/313_RTPV_13_forecasts_actuals.csv', './timeseries_data_files/314_PV_3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_308_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_320_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_317_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/320_RTPV_1_forecasts_actuals.csv',
            './timeseries_data_files/313_PV_1_forecasts_actuals.csv', './timeseries_data_files/324_PV_2_forecasts_actuals.csv',
            './timeseries_data_files/Hydro_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_310_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_323_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/Bus_314_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/313_RTPV_2_forecasts_actuals.csv', './timeseries_data_files/RTPV_zone3_forecasts_actuals.csv',
            './timeseries_data_files/312_PV_1_forecasts_actuals.csv', './timeseries_data_files/319_PV_1_forecasts_actuals.csv',
            './timeseries_data_files/320_PV_1_forecasts_actuals.csv', './timeseries_data_files/313_RTPV_8_forecasts_actuals.csv',
            './timeseries_data_files/320_RTPV_5_forecasts_actuals.csv', './timeseries_data_files/Bus_303_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_307_Load_zone3_forecasts_actuals.csv',
            './timeseries_data_files/Bus_318_Load_zone3_forecasts_actuals.csv', './timeseries_data_files/322_HYDRO_2_forecasts_actuals.csv']

# smaller set for testing
file_paths_test = ['./timeseries_data_files/101_PV_1_forecasts_actuals.csv','./timeseries_data_files/101_PV_2_forecasts_actuals.csv']



def read_files(file_paths):
    # file_paths: list of strings indicating file paths that are to be read in
    # output: data_lst - list of data frames containing all the information in each file
    # Note: we add  to a list and then concatenate as this is faster and takes less memory than growing the dataframe
    # each time
    data_lst = []
    i = 0
    bus_names = []
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
    return data_lst, bus_names


def filter_no_solar(combined_data, determining_solar_plant):
    # combined_data: data frame of all forecasts and actuals for a list of buses
    # output: two data frames called s_data and ns_data.
    # This function filters all data into two parts - one where solars are active and one where solars are inactive
    # we will do this in a pretty naive way, simply based on one of the solar plants, which we are going to hard code
    # this is not ideal, but it should do for now


    ns_data = combined_data[combined_data[determining_solar_plant + '_forecasts'] == 0]
    #ns_data.to_csv('zz_no_solar_data.csv')  # print out results as a test

    s_data = combined_data[combined_data[determining_solar_plant + '_forecasts'] != 0]
    #s_data.to_csv("zz_solar_data.csv")

    return ns_data, s_data


def compute_actual_forecast_quotient(data, bus_names):
    # data: data frame of forecasts and actuals, in the pattern of: forecast, actual
    # output: modified version of data containing additional columns with the quotient of actual / forecasts

    # iterate across bus names and take the relevant quotients
    for name in bus_names:
        temp_nm = name + '_quotient'
        data = data.assign(temp_nm=np.minimum(data[name+'_actuals'] / data[name+'_forecasts'], 1.5))
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


def apply_day_quotients(quotients, day, file_paths):
    # quotients: dataframe with all the quotients to apply
    # day: string version of what day to modify with the quotients in form YYYY-MM-DD
    # output: directly modify the time series files to apply the quotients

    # if (day == "2020-07-09"):
    #     beg = 4561
    #     end = 4585
    # elif (day == "2020-07-10"):
    #     beg = 4585
    #     end = 4609
    # elif (day == "2020-07-11"):
    #     beg = 4609
    #     end = 4633
    for path in file_paths:
        file_data = pd.read_csv(path)
        count = 0 
        file_data = file_data.set_index('datetime')
        dts = pd.Series(pd.date_range(day, periods=24, freq='H'))
        t = dts.dt.strftime('%Y-%m-%d %H:%M:%S')
        file_data.loc[t, 'actuals'] = file_data.loc[t, 'forecasts'] * quotients[path[24:-22] + "_quotient"].tolist()
        file_data = file_data.truncate(before = '2020-07-09', after = '2020-07-12')
        # for index, row in file_data.iterrows():
        #     if(row['datetime'].startswith(day)):
        #         row['actuals'] = row['forecasts'] * quotients.iloc[count, : ].loc[path[24:-22] + "_quotient"]
        #         count += 1
        #         file_data.iloc[index,:] = row
        # for index in range(beg, end):
        #     file_data["actuals"].iat[index] = file_data['forecasts'].iat[index] * quotients.iloc[count, : ].loc[path[24:-22] + "_quotient"]
        #     count += 1
        # file_data.to_csv(path, index=False)

        file_data.to_csv(path, index=True)


# run all the data perturbation functions as a function call -> should be in working directory when called and will remain.
def perturb_data(file_paths, solar_path, no_solar_path):
    path = os.getcwd()
    os.chdir("..")
    solar_data_1 = pd.read_csv(solar_path)
    no_solar_data_1 = pd.read_csv(no_solar_path)
    os.chdir(path)
    quotients_0710_1 = sample_quotients(6, 5, solar_data_1, no_solar_data_1)  # sampling the day in question
    quotients_0709_1 = sample_quotients(6, 5, solar_data_1, no_solar_data_1)  # sampling the day before
    quotients_0711_1 = sample_quotients(6, 5, solar_data_1, no_solar_data_1)  # sampling the day after


    # need to apply the quotients to the proper forecasts and write to file in the format that is readable to prescient
    # only need to write 1 day on either end of July 10 for now.
    apply_day_quotients(quotients_0709_1, "2020-07-09", file_paths)
    apply_day_quotients(quotients_0710_1, "2020-07-10", file_paths)
    apply_day_quotients(quotients_0711_1, "2020-07-11", file_paths)

# should be in directory "/downloads" when called and will stay at that directory
def save_quotients(file_paths):
    os.chdir("./rts_gmlc")
    temp, bus_names_1 = read_files(file_paths)
    all_data_1 = pd.concat(temp, axis=1)  # read in the data into a the data frame
    #all_data.to_csv('zz_all_data.csv')  # print out results as a test
    no_solar_data_1, solar_data_1 = filter_no_solar(all_data_1, "101_PV_1")
    solar_data_1 = compute_actual_forecast_quotient(solar_data_1, bus_names_1)
    no_solar_data_1 = compute_actual_forecast_quotient(no_solar_data_1, bus_names_1)
    os.chdir("..")
    solar_data_1.to_csv("./solar_quotients.csv", index=False)
    no_solar_data_1.to_csv("./no_solar_quotients.csv", index=False)

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
            #elif (line.startswith("--deterministic-ruc-solver=cbc")):
                #file.write("--deterministic-ruc-solver=gurobi \n")
            #elif (line.startswith("--sced-solver=cbc")):
                #file.write("--sced-solver=gurobi \n") 
            else:
                file.write(line)
    runner.run(populate)
    runner.run(simulate)
    shutil.rmtree("./RTS-GMLC")


def modify_file(path):
    data = pd.read_csv(path)
    # placeholder modification -> could easily be replaced
    data['actuals'].values[:] = 0
    data.to_csv(path, index=False)


def copy_directory(index):
    new_path = path_template + str(index)
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        shutil.copytree(dir_path, new_path)
    else:
        shutil.copytree(dir_path, new_path)


def run(i):
    copy_directory(i)
    os.chdir(path_template+str(i))
    perturb_data(file_paths_combined, "./solar_quotients.csv", "./no_solar_quotients.csv")
    run_prescient(i)
    os.chdir("..")

os.chdir("downloads")

# check for the quotients data and if not then recalculate it
if (not os.path.exists("./solar_quotients.csv") or not os.path.exists("./no_solar_quotients.csv")):
    save_quotients(file_paths_combined)

for i in range(runs):
    run(i)
