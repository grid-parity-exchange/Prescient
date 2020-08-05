#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os

import pandas as pd
import numpy as np
import glob


def gmlc_to_prescient(source, aggregate=False, forecast_error=True):
    """
    This takes the RTS-GMLC time series data and 
    puts it into the format required for prescient
    :param source: options are WIND, PV, RTPV, or Hydro
    :param aggregate: Aggregate all sites within the file or not?
    :return: writes csv files of forecast/actual in prescient format
    """
    # Find the files you want, forecast and actual:
    data_folder = os.path.join(path,source)
    print (data_folder)
    forecast_file = glob.glob(os.path.join(data_folder,'DAY_AHEAD*'))[0]
    actual_file = glob.glob(os.path.join(data_folder,'REAL_TIME*'))[0]
    # Read in forecast data, identify site names, collect datetime
    forecast = pd.read_csv(forecast_file)
    site_list = forecast.columns.values.tolist()[4:forecast.shape[1]]
    dt = pd.to_datetime({'year':forecast.Year, 'month':forecast.Month, 'day':forecast.Day, 'hour':forecast.Period})
    # Read in actual data, create 5-min datetime, re-sample to hourly
    if forecast_error == True:
        actual_raw = pd.read_csv(actual_file)
        dt_5min = pd.date_range(dt[0], periods=actual_raw.shape[0], freq='5Min')
        actual = actual_raw[site_list]
        actual = actual.assign(time5min = dt_5min.values)
        actual = actual.set_index('time5min')
        actual = actual.resample('H').mean()
        actual = actual.reset_index()
    else:
        actual = forecast.copy()        

    # If you want to combine all sites (or regions, whatever), write one file for all data:
    if aggregate == True:
        agg_forecast = forecast[site_list].sum(axis=1)
        agg_actual = actual[site_list].sum(axis=1)
        prescient_format = pd.DataFrame({'datetime': dt, 'forecasts': agg_forecast, 'actuals': agg_actual})
        prescient_format = prescient_format[['datetime', 'forecasts', 'actuals']]
        prescient_format.to_csv(os.path.join(write_path,source + '_forecasts_actuals' + '.csv'), index=False)
    # If not, write separate files for each site
    elif aggregate == False:
        for site in site_list:
            prescient_format = pd.DataFrame({'datetime':dt, 'forecasts':forecast[site], 'actuals':actual[site]})
            prescient_format = prescient_format[['datetime', 'forecasts', 'actuals']]
            prescient_format.to_csv(os.path.join(write_path, site + '_forecasts_actuals' + '.csv'), index=False)


def gmlc_to_prescient_by_zone(source, forecast_error=True):
    """
    This takes the RTS-GMLC time series data and 
    puts it into the format required for prescient, 
    aggregated by zone (1, 2, or 3 for RTS-GMLC)
    :param source: options are WIND, PV, RTPV, or Hydro
    :param aggregate: Aggregate all sites within the file or not?
    :return: writes csv files of forecast/actual in prescient format
    """
    # Find the files you want, forecast and actual:
    data_folder = os.path.join(path,source)
    print(data_folder)
    forecast_file = glob.glob(os.path.join(data_folder,'DAY_AHEAD*'))[0]
    actual_file = glob.glob(os.path.join(data_folder,'REAL_TIME*'))[0]
    # Read in forecast data, identify site names, collect datetime
    forecast = pd.read_csv(forecast_file)
    site_list = forecast.columns.values.tolist()[4:forecast.shape[1]]
    dt = pd.to_datetime({'year':forecast.Year, 'month':forecast.Month, 'day':forecast.Day, 'hour':forecast.Period})
    # Read in actual data, create 5-min datetime, re-sample to hourly
    if forecast_error==True:
        actual_raw = pd.read_csv(actual_file)
        dt_5min = pd.date_range(dt[0], periods=actual_raw.shape[0], freq='5Min')
        actual = actual_raw[site_list]
        actual = actual.assign(time5min = dt_5min.values)
        actual = actual.set_index('time5min')
        actual = actual.resample('H').mean()
        actual = actual.reset_index()
    else:
        actual = forecast.copy()

    zone_numbers = [1, 2, 3]
    for zone in zone_numbers:
        zone_acts = pd.DataFrame()
        zone_fore = pd.DataFrame()
        for site in site_list:
            site_zone = int(site[0])
            if site_zone == zone:
                zone_acts[site] = actual[site]
                zone_fore[site] = forecast[site]
        if zone_acts.shape[1] >= 1:
            agg_acts = zone_acts.sum(axis=1)
            agg_fore = zone_fore.sum(axis=1)
            prescient_format = pd.DataFrame({'datetime': dt, 'forecasts': agg_fore, 'actuals': agg_acts})
            prescient_format = prescient_format[['datetime', 'forecasts', 'actuals']]
            prescient_format.to_csv(os.path.join(write_path,source + '_zone' + str(zone) + '_forecasts_actuals' + '.csv'), index=False)


def load_by_bus(source, forecast_error=True):
    """
    This gets bus-level time series of load
    data based on the load participation factors
    calculated from the ratio of bus loads in the 
    bus.csv file. 
    This takes the RTS-GMLC time series data and 
    puts it into the format required for prescient, 
    aggregated by zone (1, 2, or 3 for RTS-GMLC)
    :param source: options are WIND, PV, RTPV, or Hydro
    :param aggregate: Aggregate all sites within the file or not?
    :return: writes csv files of forecast/actual in prescient format
    """
    # Find the files you want, forecast and actual:
    data_folder = os.path.join(path,source)
    forecast_file = glob.glob(os.path.join(data_folder,'DAY_AHEAD*'))[0]
    actual_file = glob.glob(os.path.join(data_folder,'REAL_TIME*'))[0]
    bus_data_file = os.path.join(source_path,'bus.csv')
    bus_data = pd.read_csv(bus_data_file)
    # Read in forecast data, identify site names, collect datetime
    forecast = pd.read_csv(forecast_file)
    site_list = forecast.columns.values.tolist()[4:forecast.shape[1]]
    dt = pd.to_datetime({'year':forecast.Year, 'month':forecast.Month, 'day':forecast.Day, 'hour':forecast.Period})
    # Read in actual data, create 5-min datetime, re-sample to hourly
    if forecast_error==True:
        actual_raw = pd.read_csv(actual_file)
        dt_5min = pd.date_range(dt[0], periods=actual_raw.shape[0], freq='5Min')
        actual = actual_raw[site_list]
        actual = actual.assign(time5min = dt_5min.values)
        actual = actual.set_index('time5min')
        actual = actual.resample('H').mean()
        actual = actual.reset_index()
    else:
        actual = forecast.copy()

    zone_numbers = [1, 2, 3]
    for zone in zone_numbers:
        # Get list of bus ID's in each zone
        idx_area = np.where(bus_data['Area'] == zone)[0]
        area_total_load = sum(bus_data['MW Load'].loc[idx_area])
        bus_ids = bus_data['Bus ID'].loc[idx_area]
        for bus in bus_ids:
            load_factor = bus_data.loc[np.where(bus_data['Bus ID'] == bus)]['MW Load'] / area_total_load
            bus_acts = actual[str(zone)].apply(lambda x: x * load_factor)
            bus_fore = forecast[str(zone)].apply(lambda x: x * load_factor)
            prescient_format = pd.concat((dt, bus_fore, bus_acts), axis=1)
            prescient_format.columns = ['datetime', 'forecasts', 'actuals']
            prescient_format.to_csv(os.path.join(write_path,'Bus_' + str(bus) + '_' + source + '_zone' + str(zone) + '_forecasts_actuals' + '.csv'), index=False)


def source_contribution_factors(source, by_zone=True):
    """
    This calculates the average contribution factor of a 
    renewable source to the overall generation, based on
    actual values, and calculated either at the system 
    level or separately for each zone. 
    :param source: options are WIND, PV, RTPV, or Hydro 
    :param by_zone: Overall system or by zone?
    :return: file with source name and the contribution
             factor associated with that source
    """
    data_folder = os.path.join(path,source)
    actual_file = glob.glob(os.path.join(data_folder,'REAL_TIME*'))[0]
    actual_raw = pd.read_csv(actual_file)
    site_list = actual_raw.columns.values.tolist()[4:actual_raw.shape[1]]

    if by_zone == True:
        zone_numbers = [1, 2, 3]
        for zone in zone_numbers:
            zone_data = pd.DataFrame()
            for site in site_list:
                site_zone = int(site[0])
                if site_zone == zone:
                    zone_data[site] = actual_raw[site]
            if zone_data.shape[1] >= 1:
                zone_sum = np.sum(zone_data, axis=1)
                zone_sites = zone_data.columns.values.tolist()
                mean_contrib = pd.DataFrame()
                for i in zone_sites:
                    mean_contrib.loc[i, 1] = round(np.mean(zone_data[i] / zone_sum), 3)
                mean_contrib = mean_contrib.reset_index()
                mean_contrib.columns = ['source', 'proportion']
                if np.sum(mean_contrib['proportion']) != 1:
                    mean_contrib.iloc[-1, -1] = round(1 - np.sum(mean_contrib.iloc[:-1, -1]), 3)
                mean_contrib.to_csv(os.path.join(write_path,'disaggregate_' + source + '_zone_' + str(zone) + '.txt'), sep=',', index=False)
    else:
        source_sum = np.sum(actual_raw.iloc[:, 4:], axis=1)
        mean_contrib = pd.DataFrame()
        for i in site_list:
            mean_contrib.loc[i, 1] = round(np.mean(actual_raw[i] / source_sum), 3)
        mean_contrib = mean_contrib.reset_index()
        mean_contrib.columns = ['source', 'proportion']
        if np.sum(mean_contrib['proportion']) != 1:
            mean_contrib.iloc[-1, -1] = round(1 - np.sum(mean_contrib.iloc[:-1, -1]), 3)
        mean_contrib.to_csv(os.path.join(write_path,'disaggregate_' + source + '_allzone.txt'), sep=',', index=False)

def create_timeseries(base_dir=None):
    
    import prescient.downloaders.rts_gmlc as rts_gmlc
    rts_gmlc.download()

    if base_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))

    global path
    global source_path
    global write_path

    path = os.path.normpath(os.path.join(base_dir,"RTS-GMLC/RTS_Data/timeseries_data_files/"))
    source_path = os.path.normpath(os.path.join(base_dir, "RTS-GMLC/RTS_Data/SourceData/"))
    write_path = os.path.normpath(os.path.join(base_dir,"timeseries_data_files/"))
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    forecast_error = True

    # To run:
    gmlc_to_prescient('WIND', forecast_error=forecast_error)
    gmlc_to_prescient('PV', forecast_error=forecast_error)
    gmlc_to_prescient('RTPV', forecast_error=forecast_error)
    gmlc_to_prescient('Hydro', forecast_error=forecast_error)
    gmlc_to_prescient('WIND', aggregate=True, forecast_error=forecast_error)
    gmlc_to_prescient('PV', aggregate=True, forecast_error=forecast_error)
    gmlc_to_prescient('RTPV', aggregate=True, forecast_error=forecast_error)
    gmlc_to_prescient('Hydro', aggregate=True, forecast_error=forecast_error)
    gmlc_to_prescient('Load', aggregate=True, forecast_error=forecast_error)


    gmlc_to_prescient_by_zone('WIND', forecast_error=forecast_error)
    gmlc_to_prescient_by_zone('PV', forecast_error=forecast_error)
    gmlc_to_prescient_by_zone('Load', forecast_error=forecast_error)
    gmlc_to_prescient_by_zone('Hydro', forecast_error=forecast_error)
    gmlc_to_prescient_by_zone('RTPV', forecast_error=forecast_error)

    load_by_bus('Load', forecast_error=forecast_error)

    source_contribution_factors('WIND', by_zone=True)
    source_contribution_factors('WIND', by_zone=False)
    source_contribution_factors('PV', by_zone=True)
    source_contribution_factors('PV', by_zone=False)
    source_contribution_factors('RTPV', by_zone=True)
    source_contribution_factors('RTPV', by_zone=False)
    source_contribution_factors('Hydro', by_zone=True)
    source_contribution_factors('Hydro', by_zone=False)

if __name__ == '__main__':
    create_timeseries()
