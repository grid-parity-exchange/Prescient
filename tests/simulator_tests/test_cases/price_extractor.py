import os
import pandas as pd
from dateutil.parser import parse as dateparser

def write_shortcut_prices( bus_name, source_dir='./', destination_dir='./' ):
    bus_detail = os.path.join(source_dir, 'bus_detail.csv')

    bus_detail = pd.read_csv( bus_detail, parse_dates=[[0,1,2]], date_parser=lambda d,h,m : dateparser(f"{d} {int(h):02}:{int(m):02}"))

    bus_detail.columns = ['DateTime', *bus_detail.columns[1:]]

    bus_detail = bus_detail[bus_detail['Bus'] == bus_name]

    bus_detail[['DateTime','LMP']].to_csv(os.path.join(destination_dir, 'real_time_prices.csv'), index=False)
    bus_detail[['DateTime','LMP DA']].to_csv(os.path.join(destination_dir, 'day_ahead_prices.csv'), index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python price_extractor.py bus_name prescient_output_dir price_output_dir")
    write_shortcut_prices( *sys.argv[1:] )
