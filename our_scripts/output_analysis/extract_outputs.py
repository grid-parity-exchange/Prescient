# extract_outputs.py: collates the outputs from prescient into a single csv file
# author: Ethan Reese
# email: ereese@princeton.edu
# created: June 2021
import pandas as pd
import os


os.chdir("..")
os.chdir("..")
os.chdir("./downloads")
scenarios = os.listdir()

cols_daily = ['Renewables available', 'Renewables used']

frame_list = []


for dir in scenarios:
    if (dir.startswith("scen")):
        total_output = pd.read_csv(dir+"/output/overall_simulation_output.csv")
        daily_output = pd.read_csv(dir+"/output/daily_summary.csv")
        total_output = total_output.join(daily_output[cols_daily])
        frame_list.append(total_output)

output = pd.concat(frame_list, ignore_index=True)
os.chdir("..")

path = "./collated_output_"
ind = 1
while(os.path.exists(path+str(ind)+".csv")):
    ind += 1

output.to_csv(path+str(ind)+".csv")