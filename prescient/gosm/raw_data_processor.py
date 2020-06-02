#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module extracts specific forecast data from the csv files published through the BPA website.
"""

import os

import pandas as pd
from pytz import timezone

def main():

    # The folder where all csv files must be stored in.
    folder = 'BPA Raw Data'
    # The specific hours (as a list of strings) the user wants to extract.
    hours = ['Hr01', 'Hr02']
    # The name of the target timezone (as used in the "pytz" module).
    target_timezone = 'US/Pacific'
    # The name of the output file.
    output_file = 'result_pacific.csv'

    # Read all csv-files in the specified folder.
    frames = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(folder + os.sep + file, index_col=0, parse_dates=True, comment='#')
            frames.append(df[hours])

    # Concatenate all data frames, drop duplicate rows and sort the rows.
    frame = pd.concat(frames)
    frame = frame.drop_duplicates(keep='first')
    frame = frame.sort_index()
    frame.index.name = 'datetime'

    # Convert the UTC time into Pacific Standard Time (PST) and Pacific Daylight Time (PDT), respectively.
    index_list = frame.index.tolist()
    new_index_list = []
    for i, dt in enumerate(index_list):
        try:
            datetime_utc = dt.replace(tzinfo=timezone('UTC'))
            datetime_pacific = datetime_utc.astimezone(timezone(target_timezone)).replace(tzinfo=None)
            new_index_list.append(datetime_pacific)
        except ValueError:
            frame = frame.drop(frame.index[i])
    frame.index = new_index_list
    frame.index.name = 'datetime'

    # Delete all rows which do not contain any data.
    frame = frame.apply(pd.to_numeric, errors='coerce')
    frame = frame.dropna()

    # Save the resulting data frame.
    frame.to_csv(folder + os.sep + 'Processed Data' + os.sep + output_file)

    return

if __name__ == '__main__':
    main()
