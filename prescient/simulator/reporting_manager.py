#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os
import shutil
import time
import itertools

import numpy as np

from .manager import _Manager
from .stats_manager import StatsManager
from prescient.stats.overall_stats import OverallStats
from prescient.stats.daily_stats import DailyStats
from prescient.reporting.csv import CsvReporter, CsvMultiRowReporter
from prescient.util import graphutils


class ReportingManager(_Manager):

    def initialize(self, options, stats_manager: StatsManager):
        self._setup_output_folders(options)
        self.setup_default_reporting(options, stats_manager)

    def _setup_output_folders(self, options):
        # before the simulation starts, delete the existing contents of the output directory.
        if os.path.exists(options.output_directory):
            shutil.rmtree(options.output_directory)
            # delay because rmtree is asynchronous and mkdir can
            # fail if you don't wait for the file system to catch up
            time.sleep(0.1)
        os.mkdir(options.output_directory)
    
        # create an output directory for plot data.
        # for now, it's only a single directory - not per-day.
        os.mkdir(os.path.join(options.output_directory,"plots"))

    
    def setup_default_reporting(self, options, stats_manager: StatsManager):
        self.setup_runtimes(options, stats_manager)
        self.setup_thermal_detail(options, stats_manager)
        self.setup_renewables_detail(options, stats_manager)
        self.setup_bus_detail(options, stats_manager)
        self.setup_line_detail(options, stats_manager)
        self.setup_hourly_gen_summary(options, stats_manager)
        self.setup_hourly_summary(options, stats_manager)
        self.setup_daily_summary(options, stats_manager)
        self.setup_overall_simulation_output(options, stats_manager)
        self.setup_daily_stack_graph(options, stats_manager)
        self.setup_cost_summary_graph(options, stats_manager)

    def setup_runtimes(self, options, stats_manager: StatsManager):
        runtime_path = os.path.join(options.output_directory, 'runtimes.csv')
        runtime_file = open(runtime_path, 'w', newline='')
        runtime_columns = {"Date":       lambda hourly: str(hourly.date),
                           "Hour":       lambda hourly: hourly.hour + 1,
                           "Type":       lambda hourly: "SCED",
                           "Solve Time": lambda hourly: hourly.sced_runtime}
        runtime_writer = CsvReporter.from_dict(runtime_file, runtime_columns)
        stats_manager.register_for_hourly_stats(runtime_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: runtime_file.close())

    def setup_thermal_detail(self, options, stats_manager: StatsManager):
        thermal_details_path = os.path.join(options.output_directory, 'thermal_detail.csv')
        thermal_details_file = open(thermal_details_path, 'w', newline='')
        thermal_details_entries_per_hour = lambda hourly: hourly.observed_thermal_dispatch_levels.keys()
        thermal_details_columns = {"Date":       lambda hourly,g: str(hourly.date),
                                   "Hour":       lambda hourly,g: hourly.hour + 1,
                                   'Generator':  lambda hourly,g: g,
                                   'Dispatch':   lambda hourly,g: hourly.observed_thermal_dispatch_levels[g],
                                   'Dispatch DA': lambda hourly,g: hourly.thermal_gen_cleared_DA[g] if options.compute_market_settlements else None,
                                   'Headroom':   lambda hourly,g: hourly.observed_thermal_headroom_levels[g],
                                   'Unit State': lambda hourly,g: hourly.observed_thermal_states[g],
                                   'Unit Cost':  lambda hourly,g: hourly.observed_costs[g],
                                   'Unit Market Revenue': lambda hourly,g: hourly.thermal_gen_revenue[g] + hourly.thermal_reserve_revenue[g] \
                                                                 if options.compute_market_settlements else None,
                                   'Unit Uplift Payment': lambda hourly,g: hourly.thermal_uplift[g] if options.compute_market_settlements else None,
                                  }
        thermal_details_writer = CsvMultiRowReporter.from_dict(thermal_details_file, thermal_details_entries_per_hour, thermal_details_columns)
        stats_manager.register_for_hourly_stats(thermal_details_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: thermal_details_file.close())

    def setup_renewables_detail(self, options, stats_manager: StatsManager):
        renewables_production_path = os.path.join(options.output_directory, 'renewables_detail.csv')
        renewables_production_file = open(renewables_production_path, 'w', newline='')
        renewables_production_entries_per_hour = lambda hourly: hourly.observed_renewables_levels.keys()
        renewables_production_columns = {'Date':       lambda hourly,g: str(hourly.date),
                                         'Hour':       lambda hourly,g: hourly.hour + 1,
                                         'Generator':  lambda hourly,g: g,
                                         'Output':     lambda hourly,g: hourly.observed_renewables_levels[g],
                                         'Output DA':  lambda hourly,g: hourly.renewable_gen_cleared_DA[g] if options.compute_market_settlements else None,
                                         'Curtailment':   lambda hourly,g: hourly.observed_renewables_curtailment[g],
                                         'Unit Market Revenue': lambda hourly,g: hourly.renewable_gen_revenue[g] if options.compute_market_settlements else None,
                                         'Unit Uplift Payment': lambda hourly,g: hourly.renewable_uplift[g] if options.compute_market_settlements else None,
                                        }
        renewables_production_writer = CsvMultiRowReporter.from_dict(renewables_production_file, renewables_production_entries_per_hour, renewables_production_columns)
        stats_manager.register_for_hourly_stats(renewables_production_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: renewables_production_file.close())

    def setup_bus_detail(self, options, stats_manager: StatsManager):
        bus_path = os.path.join(options.output_directory, 'bus_detail.csv')
        bus_file = open(bus_path, 'w', newline='')
        bus_entries_per_hour = lambda hourly: hourly.observed_bus_mismatches.keys()
        bus_columns = {'Date':       lambda hourly,b: str(hourly.date),
                       'Hour':       lambda hourly,b: hourly.hour + 1,
                       'Bus':        lambda hourly,b: b,
                       'Shortfall':     lambda hourly,b: hourly.observed_bus_mismatches[b] if hourly.observed_bus_mismatches[b] > 0.0 else 0.0,
                       'Overgeneration':  lambda hourly,b: -hourly.observed_bus_mismatches[b] if hourly.observed_bus_mismatches[b] < 0.0 else 0.0,
                       'LMP':   lambda hourly,b: hourly.observed_bus_LMPs[b],
                       'LMP DA': lambda hourly,b: hourly.planning_energy_prices[b] if options.compute_market_settlements else None,
                      }
        bus_writer = CsvMultiRowReporter.from_dict(bus_file, bus_entries_per_hour, bus_columns)
        stats_manager.register_for_hourly_stats(bus_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: bus_file.close())

    def setup_line_detail(self, options, stats_manager: StatsManager):
        line_path = os.path.join(options.output_directory, 'line_detail.csv')
        line_file = open(line_path, 'w', newline='')
        line_entries_per_hour = lambda hourly: hourly.observed_flow_levels.keys()
        line_columns = {'Date': lambda hourly,l: str(hourly.date),
                        'Hour': lambda hourly,l: hourly.hour + 1,
                        'Line': lambda hourly,l: l,
                        'Flow': lambda hourly,l: hourly.observed_flow_levels[l]
                   }
        line_writer = CsvMultiRowReporter.from_dict(line_file, line_entries_per_hour, line_columns)
        stats_manager.register_for_hourly_stats(line_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: line_file.close())

    def setup_hourly_gen_summary(self, options, stats_manager: StatsManager):
        hourly_gen_path = os.path.join(options.output_directory, 'Hourly_gen_summary.csv')
        hourly_gen_file = open(hourly_gen_path, 'w', newline='')
        hourly_gen_columns = {'Date': lambda hourly: str(hourly.date),
                              'Hour': lambda hourly: hourly.hour + 1,
                              'Load shedding': lambda hourly: hourly.load_shedding,
                              'Reserve shortfall': lambda hourly: hourly.reserve_shortfall,
                              'Available reserves': lambda hourly: hourly.available_reserve,
                              'Over generation': lambda hourly: hourly.over_generation,
                              'Reserve Price DA': lambda hourly: hourly.planning_reserve_price if options.compute_market_settlements else None,
                              'Reserve Price RT': lambda hourly: hourly.reserve_RT_price
                             }
        hourly_gen_writer = CsvReporter.from_dict(hourly_gen_file, hourly_gen_columns)
        stats_manager.register_for_hourly_stats(hourly_gen_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: hourly_gen_file.close())

    def setup_hourly_summary(self, options, stats_manager: StatsManager):
        hourly_path = os.path.join(options.output_directory, 'hourly_summary.csv')
        hourly_file = open(hourly_path, 'w', newline='')
        hourly_columns = {'Date': lambda hourly: str(hourly.date),
                          'Hour': lambda hourly: hourly.hour + 1,
                          'TotalCosts': lambda hourly: hourly.total_costs,
                          'FixedCosts': lambda hourly: hourly.fixed_costs,
                          'VariableCosts': lambda hourly: hourly.variable_costs,
                          'LoadShedding': lambda hourly: hourly.load_shedding,
                          'OverGeneration': lambda hourly: hourly.over_generation,
                          'ReserveShortfall': lambda hourly: hourly.reserve_shortfall,
                          'RenewablesUsed': lambda hourly: hourly.renewables_used,
                          'RenewablesCurtailment': lambda hourly: hourly.renewables_curtailment,
                          'Demand': lambda hourly: hourly.total_demand,
                          'Price': lambda hourly: hourly.price
                         }
        hourly_writer = CsvReporter.from_dict(hourly_file, hourly_columns)
        stats_manager.register_for_hourly_stats(hourly_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: hourly_file.close())


    def setup_daily_summary(self, options, stats_manager: StatsManager):
        daily_path = os.path.join(options.output_directory, 'Daily_summary.csv')
        daily_file = open(daily_path, 'w', newline='')
        daily_columns = {'Date': lambda daily: str(daily.date),
                         'Demand': lambda daily: daily.this_date_demand,
                         'Renewables available': lambda daily: daily.this_date_renewables_available,
                         'Renewables used': lambda daily: daily.this_date_renewables_used,
                         'Renewables penetration rate': lambda daily: daily.this_date_renewables_penetration_rate, ############  TODO: Implement ##############
                         'Average price': lambda daily: daily.this_date_average_price,
                         'Fixed costs': lambda daily: daily.this_date_fixed_costs,
                         'Generation costs': lambda daily: daily.this_date_variable_costs,
                         'Load shedding': lambda daily: daily.this_date_load_shedding,
                         'Over generation': lambda daily: daily.this_date_over_generation,
                         'Reserve shortfall': lambda daily: daily.this_date_reserve_shortfall,
                         'Renewables curtailment': lambda daily: daily.this_date_renewables_curtailment,
                         'Number on/offs': lambda daily: daily.this_date_on_offs,
                         'Sum on/off ramps': lambda daily: daily.this_date_sum_on_off_ramps,
                         'Sum nominal ramps': lambda daily: daily.this_date_sum_nominal_ramps}
        if options.compute_market_settlements:
            daily_columns.update( {'Renewables energy payments': lambda daily: daily.this_date_renewable_energy_payments,
                                  'Renewables uplift payments': lambda daily: daily.this_date_renewable_uplift,
                                  'Thermal energy payments': lambda daily: daily.this_date_thermal_energy_payments,
                                  'Thermal uplift payments': lambda daily: daily.this_date_thermal_uplift,
                                  'Total energy payments': lambda daily: daily.this_date_energy_payments,
                                  'Total uplift payments': lambda daily: daily.this_date_uplift_payments,
                                  'Total reserve payments': lambda daily: daily.this_date_reserve_payments,
                                  'Total payments': lambda daily: daily.this_date_total_payments,
                                  'Average payments': lambda daily: daily.this_date_average_payments,
                                } )
        daily_writer = CsvReporter.from_dict(daily_file, daily_columns)
        stats_manager.register_for_daily_stats(daily_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: daily_file.close())

    def setup_overall_simulation_output(self, options, stats_manager: StatsManager):
        # We implement this as an inner function so that we can open and close the file
        # at the end of the simulation instead of keeping it open the whole session
        def write_overall_stats(overall: OverallStats):
            overall_path = os.path.join(options.output_directory, 'Overall_simulation_output.csv')
            overall_file = open(overall_path, 'w', newline='')
            overall_cols = {'Total demand':            lambda overall: overall.cumulative_demand,
                            'Total fixed costs':       lambda overall: overall.total_overall_fixed_costs,
                            'Total generation costs':  lambda overall: overall.total_overall_generation_costs,
                            'Total costs':             lambda overall: overall.total_overall_costs,
                            'Total load shedding':     lambda overall: overall.total_overall_load_shedding,
                            'Total over generation':   lambda overall: overall.total_overall_over_generation,
                            'Total reserve shortfall': lambda overall: overall.total_overall_reserve_shortfall,
                            'Total renewables curtailment': lambda overall: overall.total_overall_renewables_curtailment,
                            'Total on/offs':           lambda overall: overall.total_on_offs,
                            'Total sum on/off ramps':  lambda overall: overall.total_sum_on_off_ramps,
                            'Total sum nominal ramps': lambda overall: overall.total_sum_nominal_ramps,
                            'Maximum observed demand': lambda overall: overall.max_hourly_demand,
                            'Overall renewables penetration rate': lambda overall: overall.overall_renewables_penetration_rate,
                            'Cumulative average price':lambda overall: overall.cumulative_average_price}
            if options.compute_market_settlements:
                overall_cols.update({'Total energy payments': lambda overall: overall.total_thermal_energy_payments,
                                     'Total reserve payments':lambda overall: overall.total_reserve_payments,
                                     'Total uplift payments': lambda overall: overall.total_uplift_payments,
                                     'Total payments':        lambda overall: overall.total_payments,
                                     'Cumulative average payments': lambda overall: overall.cumulative_average_payments})
            overall_reporter = CsvReporter.from_dict(overall_file, overall_cols)
            overall_reporter.write_record(overall)
            overall_file.close()
        stats_manager.register_for_overall_stats(write_overall_stats)


    def setup_daily_stack_graph(self, options, stats_manager: StatsManager):
        stats_manager.register_for_daily_stats(
            lambda daily_stats: ReportingManager.generate_stack_graph(
                options, daily_stats, self.simulator.data_manager.prior_sced_instance))

    @staticmethod
    def generate_stack_graph(options, daily_stats: DailyStats, sced):
        plot_peak_demand = options.plot_peak_demand if options.plot_peak_demand > 0.0 else daily_stats.thermal_fleet_capacity

        daily_hours = range(0, 24)
        generator_types = {**{g: sced.ThermalGeneratorType[g] for g in sced.ThermalGenerators},
                           **{g: sced.NondispatchableGeneratorType[g] for g in sced.AllNondispatchableGenerators}}
        generator_dispatch_levels = {**{g: np.fromiter((daily_stats.hourly_stats[h].observed_thermal_dispatch_levels[g] for h in daily_hours), float)
                                        for g in sced.ThermalGenerators},
                                     **{g: np.fromiter((daily_stats.hourly_stats[h].observed_renewables_levels[g] for h in daily_hours), float)
                                        for g in sced.AllNondispatchableGenerators}}
        reserve_requirements_by_hour = [daily_stats.hourly_stats[h].reserve_requirement for h in daily_hours]
        curtailments_by_hour = [sum(daily_stats.hourly_stats[h].observed_renewables_curtailment.values()) for h in daily_hours]
        load_shedding_by_hour = [daily_stats.hourly_stats[h].load_shedding for h in daily_hours]
        reserve_shortfalls_by_hour = [daily_stats.hourly_stats[h].reserve_shortfall for h in daily_hours]
        available_reserves_by_hour = [daily_stats.hourly_stats[h].available_reserve for h in daily_hours]
        available_quickstart_by_hour = [daily_stats.hourly_stats[h].available_quickstart for h in daily_hours]
        over_generation_by_hour = [daily_stats.hourly_stats[h].over_generation for h in daily_hours]
        quick_start_additional_power_generated_by_hour = [daily_stats.hourly_stats[h].quick_start_additional_power_generated for h in daily_hours]
        event_annotations = itertools.chain(*(daily_stats.hourly_stats[h].event_annotations for h in daily_hours))
        demand_list = [daily_stats.hourly_stats[h].total_demand for h in daily_hours]

        graphutils.generate_stack_graph(plot_peak_demand,  # for scale of the plot
                                        generator_types,
                                        generator_dispatch_levels,
                                        reserve_requirements_by_hour,
                                        str(daily_stats.date),
                                        curtailments_by_hour,
                                        load_shedding_by_hour,
                                        reserve_shortfalls_by_hour,
                                        available_reserves_by_hour,
                                        available_quickstart_by_hour,
                                        over_generation_by_hour,
                                        daily_stats.max_hourly_demand,
                                        quick_start_additional_power_generated_by_hour,
                                        annotations=event_annotations, 
                                        display_plot=options.display_plots, 
                                        show_plot_legend=(not options.disable_plot_legend),
                                        savetofile=True, 
                                        output_directory=os.path.join(options.output_directory, "plots"),
                                        plot_individual_generators=options.plot_individual_generators,
                                        renewables_penetration_rate=daily_stats.this_date_renewables_penetration_rate,
                                        fixed_costs=daily_stats.this_date_fixed_costs,
                                        variable_costs=daily_stats.this_date_variable_costs,
                                        demand=demand_list)

    def setup_cost_summary_graph(self, options, stats_manager: StatsManager):
        stats_manager.register_for_overall_stats(
            lambda overall_stats: ReportingManager.generate_cost_summary_graph(options, overall_stats))

    @staticmethod
    def generate_cost_summary_graph(options, overall_stats: OverallStats):
        daily_fixed_costs = [daily_stats.this_date_fixed_costs for daily_stats in overall_stats.daily_stats]
        daily_generation_costs = [daily_stats.this_date_variable_costs for daily_stats in overall_stats.daily_stats]
        daily_load_shedding = [daily_stats.this_date_load_shedding for daily_stats in overall_stats.daily_stats]
        daily_over_generation = [daily_stats.this_date_over_generation for daily_stats in overall_stats.daily_stats]
        daily_reserve_shortfall = [daily_stats.this_date_reserve_shortfall for daily_stats in overall_stats.daily_stats]
        daily_renewables_curtailment = [daily_stats.this_date_renewables_curtailment for daily_stats in overall_stats.daily_stats]

        graphutils.generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs,
                                               daily_load_shedding, daily_over_generation,
                                               daily_reserve_shortfall,
                                               daily_renewables_curtailment,
                                               display_plot=options.display_plots,
                                               save_to_file=True,
                                               output_directory=os.path.join(options.output_directory, "plots"))

