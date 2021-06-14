# baseline.py: runs prescient with all actuals=forecasts to establish a baseline
# authors: Ethan Reese, Arvind Shrivats
# email: ereese@princeton.edu, shrivats@princeton.edu
# created: June 12, 2021

os.chdir("..")
os.chdir("..")

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
print(os.getcwd())

# all file paths
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

# variables to adjust:
runs = 2
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
            #elif (line.startswith("--deterministic-ruc-solver=cbc")):
                #file.write("--deterministic-ruc-solver=gurobi \n")
            #elif (line.startswith("--sced-solver=cbc")):
                #file.write("--sced-solver=gurobi \n") 
            else:
                file.write(line)
    runner.run(populate)
    runner.run(simulate)

def modify_files(paths):
	for path in paths:
		file_data = pd.read_csv(path)
		#file_data = file_data.truncate(before = '2020-07-09', after = '2020-07-12')
		file_data["actuals"] = file_data["forecasts"]

		file_data.to_csv(path, index=False)



def copy_directory(index):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        shutil.copytree(dir_path, new_path)
    else:
        shutil.copytree(dir_path, new_path)


def run(i):
    copy_directory(i)
    os.chdir("./working")
    modify_files(file_paths_combined)
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
