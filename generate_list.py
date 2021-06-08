import os

os.chdir("./downloads/rts_gmlc/timeseries_data_files")
list = os.listdir()
list_1 = []
list_2 = []
list_3 = []
for item in list:
    if ("zone1" in item or item.startswith("1")):
        list_1.append(str("./timeseries_data_files/" + item))
    elif ("zone2" in item or item.startswith("2")):
        list_2.append(str("./timeseries_data_files/" + item))
    elif ("zone3" in item or item.startswith("3")):
        list_3.append(str("./timeseries_data_files/" + item))

print("One: ", list_1)
print("Two: ", list_2)
print("Three: ", list_3)