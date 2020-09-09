#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""

This module has a method to create scenarios and several other methods that are
used to write the scenarios to files that can be read by the simulator.
"""

import os
import shutil
import datetime
import itertools

import matplotlib
matplotlib.use('Agg')
import pandas as pd

import prescient.gosm.basicclasses as basicclasses
import prescient.gosm.gosm_options as gosm_options
import prescient.gosm.sources as sources
import prescient.gosm.structures.skeleton_scenario as scenario
import prescient.gosm.structures.skeleton_point_paths as skelpaths
import prescient.gosm.scenario_generator as scengen
from prescient.gosm.pyspgen import do_2stage_ampl_dir, write_actuals_and_expected
from prescient.gosm.pyspgen import check_scen_template_for_sources
from prescient.gosm.structures import SkeletonScenarioSet
from prescient.gosm.structures import parse_dps_path_file_all_sources
from prescient.gosm.structures import one_dimensional_pattern_set_from_file
from prescient.gosm.structures import multi_dimensional_pattern_set_from_file
from prescient.gosm.structures import parse_partition_file
from prescient.util.distributions.distributions import UnivariateEpiSplineDistribution
from prescient.util.distributions.distribution_factory import distribution_factory

def print_progress(progress, source_name=None):
    """
    Prints the progress and adds the current day. This is especially useful
    to create an easily readible output when using the populator.

    Args:
        progress (str): a string describing the progress
        source_name (str): the name of the source (if required for output)
    """

    if source_name is not None and source_name != 'multiple':
        source = ': ' + source_name
    else:
        source = ''
    try:
        day = gosm_options.scenario_day.strftime('%Y-%m-%d')
    except AttributeError:
        day = ''
    if source == '' and day == '':
        colon = ''
    else:
        colon = ': '
    print(day + source + colon + progress)


def compute_historic_dates():
    """
    This function will return proper Timestamp objects for the user specified
    start date and end date for historic data. If either is unspecified,
    the corresponding value will be None.

    Returns:
        tuple: The first component is the date of the start of historic data,
            the second is the date corresponding to the end of the historic
            data.
    """
    if gosm_options.historic_data_start is not None:
        historic_start = pd.Timestamp(gosm_options.historic_data_start)
    else:
        historic_start = None

    if gosm_options.historic_data_end is not None:
        historic_end = pd.Timestamp(gosm_options.historic_data_end)
    else:
        historic_end = None

    return historic_start, historic_end


def construct_actual_expected_scenarios(data_sources):
    """
    This will construct PowerScenario objects composed from actuals and
    forecasts data for each source of power. For these scenarios, the values
    are simply drawn directly from the data and have probability 1 (this
    number is meaningless in this context however)

    Args:
        data_sources (list[Source]): All of the power sources
    Returns:
        (PowerScenario, PowerScenario): The actual power scenario and then
            the forecast power scenario
    """
    date = pd.Timestamp(gosm_options.scenario_day)
    actual_dict = {}
    expected_dict = {}
    end_of_day = date+datetime.timedelta(hours=23)

    for source in data_sources:
        name = source.name
        actual_column = source.data['actuals'][date:end_of_day]
        actuals = list(actual_column.values)
        expected_column = source.data['forecasts'][date:end_of_day]
        expecteds = list(expected_column.values)

        actual_dict[name] = actuals
        expected_dict[name] = expecteds

    return (scenario.PowerScenario('actuals', actual_dict, 1),
            scenario.PowerScenario('forecasts', expected_dict, 1))


def sample_scenarios_multiple_sources(data_sources, date):
    """
    This is a function which is a wrapper of the SampleMethod
    for sampling the error distributions to produce scenarios.

    This is here to handle to the fact that there might be multiple sources
    and samples each source individually to produce scenarios for each source
    and then takes the cartesian product of the sources.

    Args:
        data_sources (list[ExtendedSource]): A list of the sources of data
            for each power generation source
        date (datetime-like): The date to produce scenarios for
    Returns:
        list[PowerScenario]: A list of the PowerScenario objects created
    """
    sample_method = scengen.SampleMethod(gosm_options.number_scenarios)
    # This will compute the cartesian product of the scenario sets
    # produced using the sample method for each source
    prod_method = scengen.ProductMethod(sample_method)
    start_date, end_date = compute_historic_dates()
    prod_method.fit(data_sources, date, start_date, end_date, verbose=True)
    return prod_method.generate()


def epispline_scenarios(data_sources, date):
    """
    This function runs the epispline method for each source

    Args:
        data_sources (list[ExtendedSource]): A list of the sources of data
            for each power generation source
        date (datetime-like): The date to produce scenarios for
    Returns:
        list[PowerScenario]: A list of the PowerScenario objects created
    """
    methods = []
    try:
        if gosm_options.hyperrectangles_file is None:
            raise RuntimeError("Hyperrectangle file must be specified if using"
                               " nondeterministic sources with the "
                               "--hyperrectangles-file option")
        interval_set = one_dimensional_pattern_set_from_file(
            gosm_options.hyperrectangles_file)
    except FileNotFoundError:
        raise RuntimeError("Hyperrectangle File {} not found".format(
            gosm_options.hyperrectangles_file))

    for source in data_sources:
        try:
            if gosm_options.dps_file is None:
                raise RuntimeError("A DPS Paths file must be specified if "
                                   "using nondeterministic sources with the "
                                   "--dps-file option")
            dps, paths, _ = skelpaths.parse_dps_path_file(
                source.name, gosm_options.dps_file)
        except FileNotFoundError:
            raise RuntimeError("DPS Paths File {} not found".format(
                gosm_options.dps_file))
        if source.source_type == 'solar':
            # For solar sources, we need to estimate the sunrise and sunset.
            date = pd.Timestamp(date)

            # We either explicitly specify sunrise and sunset through the
            # options or we estimate them from historic data.
            if gosm_options.dps_sunrise and gosm_options.dps_sunset:
                print_progress("Taking hours of sunrise and sunset from "
                               "options --dps-sunrise and --dps-sunset.")
                sunrise = gosm_options.dps_sunrise
                sunset = gosm_options.dps_sunset
            else:
                print_progress("Since --dps-sunrise and --dps-sunset "
                               "not specified, will estimate hours of sunrise "
                               "and sunset.")
                sunrise, sunset = source.estimate_sunrise_sunset(date)
            # The first and last day part separators are set to sunrise
            # and sunset.
            dps[0], dps[-1] = sunrise, sunset
            if gosm_options.use_temporal_copula:
                method = scengen.SolarEpiSplineMethod(
                    dps, paths, interval_set, gosm_options.temporal_copula)
            else:
                method = scengen.SolarEpiSplineMethod(
                    dps, paths, interval_set)
        else:
            if gosm_options.use_temporal_copula:
                method = scengen.EpiSplineMethod(dps, paths, interval_set,
                        gosm_options.temporal_copula)
            else:
                method = scengen.EpiSplineMethod(dps, paths, interval_set)

        # Setting other user-specified options for scenario creation
        method.set_spline_options(**gosm_options.distr_options)
        method.set_tolerance_levels(gosm_options.cdf_inverse_tolerance,
                                    gosm_options.cdf_inverse_max_refinements,
                                    gosm_options.cdf_tolerance)
        methods.append(method)

    prod_method = scengen.ProductMethod(methods)


    start_date, end_date = compute_historic_dates()
    fit_method = prod_method.fit(data_sources, date, start_date, end_date,
                                 verbose=True)

    # We plot the distributions here where they are still in scope.
    if not gosm_options.disable_plots:
        plot_dir = gosm_options.output_directory + os.sep +'plots'
        fit_method.plot(plot_dir, plot_pdf=gosm_options.plot_pdf,
                        plot_cdf=gosm_options.plot_cdf)

    return fit_method.generate()


def spatial_copula_scenarios(data_sources, date):
    """
    This function executes the spatial copula routine for the sources of power
    specified.

    Args:
        data_sources (list[ExtendedSource]): A list of the sources of data
            for each power generation source
        date (datetime-like): The date to produce scenarios for
    Returns:
        list[PowerScenario]: A list of the PowerScenario objects created
    """
    dps_paths = parse_dps_path_file_all_sources(gosm_options.dps_file)

    # In order to use spatial copulas, there must be a path set with the
    # name 'multiple'
    if 'multiple' not in dps_paths:
        raise RuntimeError("{} does not have a dps-paths set with name "
                           "'multiple' which is required to use spatial "
                           "copulas.".format(gosm_options.dps_file))

    # We read in the day part separators, paths, and pattern set
    dps, paths = dps_paths['multiple']['dps'], dps_paths['multiple']['paths']
    interval_set = one_dimensional_pattern_set_from_file(
        gosm_options.hyperrectangles_file)

    # We also need a partition file to read in.
    if gosm_options.partition_file is None:
        raise RuntimeError('To use spatial copulas the --partition-file '
                           'option must be set')
    partition = parse_partition_file(gosm_options.partition_file)

    # We encapsulate all the scenario creation parameters in the
    # spatial copula methods and then fit the method to the sources.
    # Then we generate the scenarios.
    method = scengen.SpatialCopulaMethod(dps, paths, interval_set)
    method.set_spline_options(**gosm_options.distr_options)
    method.set_tolerance_levels(gosm_options.cdf_inverse_tolerance,
                                gosm_options.cdf_inverse_max_refinements,
                                gosm_options.cdf_tolerance)
    start_date, end_date = compute_historic_dates()

    fit_method = method.fit(data_sources, date, partition, start_date,
                            end_date, verbose=True)

    if not gosm_options.disable_plots:
        plot_dir = gosm_options.output_directory + os.sep +'plots'
        fit_method.plot(plot_dir, plot_pdf=gosm_options.plot_pdf,
                        plot_cdf=gosm_options.plot_cdf)

    return fit_method.generate()


def run_scenario_creators(data_sources, date):
    """
    This function will execute one of the separate scenario creation methods
    depending on the user options.

    Args:
        data_sources (list[ExtendedSource]): A list of the sources of data
            for each power generation source
        date (datetime-like): The date to produce scenarios for
    Returns:
        list[PowerScenario]: A list of the PowerScenario objects created
    """
    if gosm_options.sample_skeleton_points:
        return sample_scenarios_multiple_sources(data_sources, date)
    elif gosm_options.cross_scenarios and not gosm_options.use_spatial_copula:
        # Cross scenarios means compute scenarios for each source separately
        # and then combine them. Here we have different paths for each source.
        return epispline_scenarios(data_sources, date)
    elif gosm_options.cross_scenarios and gosm_options.use_spatial_copula:
        # We have a separate method to facilitate the use of spatial copulas
        return spatial_copula_scenarios(data_sources, date)
    else:
        # If we do not want to cross scenarios, we will fit multidimensional
        # distributions to the sources.
        rectangle_set = multi_dimensional_pattern_set_from_file(
            gosm_options.hyperrectangles_file)
        power_scenarios = construct_scenarios_multiple_sources(
            power_windows, rectangle_set)
        return power_scenarios


def create_deterministic_scenario(data_sources, date):
    """
    This function will construct a single scenario which is pulled directly
    from the forecasts for a given source.
    Args:
        data_sources (list[ExtendedSource]): A list of the sources of data
            for each power generation source
        date (datetime-like): The date to produce scenarios for
    Returns:
        PowerScenario: The deterministic scenario
    """
    start_date, end_date = compute_historic_dates()

    det_method = scengen.DeterministicMethod()
    # This runs deterministic method on each source and takes the cartesian
    # product of the scenarios produced. This will return a list of 1 scenario
    prod_method = scengen.ProductMethod(det_method)
    fit_method = prod_method.fit(data_sources, date, start_date, end_date,
                                verbose=True)
    return fit_method.generate()[0]


def construct_scenario_set(data_sources, date):
    """
    This function will prepare the data first by computing errors as
    actuals - forecasts and then extracts the relevant data from the windows.

    It then constructs individual PowerScenario objects depending from the
    windows of data crossing scenarios if specified by the options.

    It then uses the constructed scenarios in addition to actual and expected
    scenarios to construct a ScenarioSet object.

    Args:
        data_sources (List[ExtendedSource]): The list of sources of data
    Returns:
        ScenarioSet: The collection of scenarios
    """

    # This extracts the load data from the sources
    load_sources = [source for source in data_sources
                    if source.source_type == 'load']

    forecast_load_data, actual_load_data = {}, {}

    for source in load_sources:
        load_window = source.rolling_window(date)

        load_forecast = load_window.dayahead_data['forecasts']
        forecast_load_data[source.name] = list(load_forecast[
            date:(date+datetime.timedelta(hours=23))].values)

        load_actual = load_window.dayahead_data['actuals']
        actual_load_data[source.name] = list(load_actual[
            date:(date+datetime.timedelta(hours=23))].values)

    # For deterministic sources, we simply set the forecast as the sole
    # scenario.
    det_sources = [source for source in data_sources
                   if source.is_deterministic]

    # For nondeterministic sources, we actually run the scenario creation
    # procedures.
    nondet_sources = [source for source in data_sources if source.source_type in
                      sources.power_sources and not source.is_deterministic]

    power_scenarios = []
    nondet_scenarios = []
    det_scen = None

    # If we have only deterministic sources, we don't need to run stochastic
    # scenario creation
    if nondet_sources:
        nondet_scenarios = run_scenario_creators(nondet_sources, date)

    # We then need to merge the nondeterministic and deterministic scenarios
    # if we have both types of scenarios
    if det_sources:
        det_scen = create_deterministic_scenario(det_sources, date)

    if nondet_scenarios and det_scen:
        # Both deterministic and nondeterministic
        power_scenarios = [scenario.merge_independent_scenarios([scen, det_scen])
                           for scen in nondet_scenarios]
    elif nondet_scenarios:
        # Only nondeterministic scenarios
        power_scenarios = nondet_scenarios
    elif det_scen:
        # Only deterministic scenarios
        power_scenarios = [det_scen]
    else:
        raise RuntimeError("No power sources specified, this is a problem")

    power_sources = det_sources + nondet_sources

    # Here we convert our power generation scenarios into scenarios with
    # load data.
    scenarios = [scen.add_load_data(forecast_load_data, power_sources)
                 for scen in power_scenarios]

    # We then construct the scenarios composed of the actuals and forecasts
    # These are drawn directly from the data and have probability 1.
    actual, expected = construct_actual_expected_scenarios(power_sources)
    actual_scen, expect_scen = (
        actual.add_load_data(actual_load_data, power_sources),
        expected.add_load_data(forecast_load_data, power_sources))

    # We disaggregate the power values for meta sources.
    for source in data_sources:
        if source.source_params['aggregate']:
            disaggregation = source.source_params['disaggregation']
            for scen in scenarios + [actual_scen, expect_scen]:
                scen.disaggregate_source(source.name, disaggregation,
                                         source.source_type == 'load')

    scenario_set = scenario.SkeletonScenarioSet(scenarios,
        actual_scen, expect_scen)

    return scenario_set


def create_scenarios(data_sources, date):
    """
    This function will create scenarios using the data sources passed in.
    It will construct the scenarios and write the output files.

    Args:
        data_sources (list[ExtendedSource]): The collection of data sources
            which have historic information for forecasts and actuals and
            dayahead information for forecasts. One must be a load source and
            at least one must be a power source.
        date (datetime-like): The date for which scenarios are to be
            created
    """
    scenario_day = pd.Timestamp(date)

    # Create the directory to write the output in.
    if os.path.isdir(gosm_options.output_directory):
        shutil.rmtree(gosm_options.output_directory)
    os.makedirs(gosm_options.output_directory)

    # We write the raw data out to a file for ease in analysis
    raw_data_folder = gosm_options.output_directory
    if not os.path.isdir(raw_data_folder):
        os.mkdir(raw_data_folder)

    scenario_set = construct_scenario_set(data_sources, date)
    # Sometimes, due to floating point errors, it is not exactly 1, so we
    # rescale it
    scenario_set.normalize_probabilities()
    scenario_set.normalize_names()

    print_progress("Writing Output")
    scenario_set.write_raw_scenarios(raw_data_folder, scenario_day)

    if not gosm_options.disable_plots:
        print_progress("Plotting Scenarios")
        plot_directory = gosm_options.output_directory + os.sep +'plots'
        title = gosm_options.scenario_day.strftime('%Y-%m-%d ')
        scenario_set.plot_scenarios(plot_directory, title)


    if gosm_options.tree_template_file and gosm_options.scenario_template_file:
        write_pysp_files(data_sources, scenario_set)
    else:
        print("Since --tree-template-file or --scenario-template-file")
        print("were not specified, PySP files will not be produced")

    print_progress('Done creating scenarios.')


def write_pysp_files(data_sources, scenario_set):
    """
    This will construct the pysp files in the manner specified by the options
    and using the sources and the scenario tree constructed.

    Args:
        data_sources (List[Source]): A list of source objects
            corresponding to the power sources
        scenario_set (ScenarioSet): The constructed scenarios
    """
    raw_nodes = scenario_set.create_raw_nodes()

    # We construct PySP_TreeTemplate object
    treetemp = basicclasses.PySP_Tree_Template()
    try:
        treetemp.read_AMPL_template(gosm_options.tree_template_file)
    except FileNotFoundError:
        raise RuntimeError("Tree Template File {} not found".format(
            gosm_options.tree_template_file))

    source_names = scenario_set.source_names

    check_scen_template_for_sources(gosm_options.scenario_template_file,
                                    source_names)

    scentemp = basicclasses.PySP_Scenario_Template()
    try:
        scentemp.read_AMPL_template_with_tokens(
            gosm_options.scenario_template_file)
    except FileNotFoundError:
        raise RuntimeError("Scenario Template File {} not found".format(
            gosm_options.scenario_template_file))

    do_2stage_ampl_dir(
        gosm_options.output_directory, raw_nodes, scentemp, treetemp)

    # The actual and expected scenarios require a different method as they
    # are not part of the scenario tree
    write_actuals_and_expected(scenario_set, gosm_options.output_directory,
                               scentemp)


def read_sources():
    """
    This function will parse out the sources from the sources file.

    Returns:
        list[ExtendedSource]: The sources of data
    """
    if gosm_options.sources_file is None:
        raise RuntimeError("The Sources file must be specified with the "
                           "--sources-file option")

    if gosm_options.sources_file.endswith('.csv'):
        # If using the old sources file format
        print("Warning: the csv-file format for sources is deprecated.")
        data_sources = sources.sources_from_sources_file(
            gosm_options.sources_file)
        # We need to set frac_nondispatch manually
        for source in data_sources:
            if source.source_type == 'solar':
                source.frac_nondispatch = gosm_options.solar_frac_nondispatch
            elif source.source_type == 'wind':
                source.frac_nondispatch = gosm_options.wind_frac_nondispatch
    elif gosm_options.sources_file.endswith('.txt'):
        # If using the new sources file format

        # This means that the option was set, but it will be ignored
        if not(gosm_options.solar_frac_nondispatch == 1
               and gosm_options.wind_frac_nondispatch == 1):
           print("Warning one of --solar-frac-nondispatch and "
                 "--wind-frac-nondispatch but will be ignored. "
                 "To set these, add frac_nondispatch as a keyword "
                 "for the source in the sources file.")

        data_sources = sources.sources_from_new_sources_file(
            gosm_options.sources_file)

    return data_sources

def main():
    """
    Sets the option, reads in the sources of uncertainty and
    starts the scenario creation.
    """

    # Set the options.
    gosm_options.set_globals()

    try:
        data_sources = read_sources()
    except FileNotFoundError:
        raise RuntimeError("Cannot find sources file {}".format(
            gosm_options.sources_file))

    # Create the scenarios.
    create_scenarios(data_sources, gosm_options.scenario_day)

if __name__ == '__main__':
    main()
