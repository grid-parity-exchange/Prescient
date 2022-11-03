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

from egret.data.model_data import ModelData
from egret.models.unit_commitment import _time_series_dict

from .manager import _Manager
from .stats_manager import StatsManager
from prescient.stats.overall_stats import OverallStats
from prescient.stats.daily_stats import DailyStats
from prescient.reporting.csv import CsvReporter, CsvMultiRowReporter

# If appropriate back-ends for Matplotlib are not installed
# (e.g, gtk), then graphing will not be available.
try:
    from prescient.util import graphutils
    graphutils_functional = True
except ValueError:
    print("***Unable to load Gtk back-end for matplotlib - graphics generation is disabled")
    graphutils_functional = False

class ReportingManager(_Manager):

    def initialize(self, options, stats_manager: StatsManager):
        self._setup_output_folders(options)
        self._round = lambda val : round(val, options.output_max_decimal_places)
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
        self.setup_virtual_detail(options, stats_manager)
        self.setup_bus_detail(options, stats_manager)
        self.setup_line_detail(options, stats_manager)
        self.setup_contingency_detail(options, stats_manager)
        self.setup_reserves_detail(options, stats_manager)
        self.setup_hourly_gen_summary(options, stats_manager)
        self.setup_hourly_summary(options, stats_manager)
        self.setup_daily_summary(options, stats_manager)
        self.setup_overall_simulation_output(options, stats_manager)

        if graphutils_functional and not options.disable_stackgraphs:
            self.setup_daily_stack_graph(options, stats_manager)
            self.setup_cost_summary_graph(options, stats_manager)

    def setup_runtimes(self, options, stats_manager: StatsManager):
        _round = self._round
        runtime_path = os.path.join(options.output_directory, 'runtimes.csv')
        runtime_file = open(runtime_path, 'w', newline='')

        ops_runtime_columns = {"Date":       lambda ops: str(ops.timestamp.date()),
                               "Hour":       lambda ops: ops.timestamp.hour,
                               "Minute":     lambda ops: ops.timestamp.minute,
                               "Type":       lambda ops: "SCED",
                               "Solve Time": lambda ops: _round(ops.sced_runtime)}
        ops_runtime_writer = CsvReporter.from_dict(runtime_file, ops_runtime_columns)
        stats_manager.register_for_sced_stats(ops_runtime_writer.write_record)

        if options.sced_frequency_minutes != 60:
            hr_runtime_columns = {"Date":       lambda hourly: str(hourly.date),
                                  "Hour":       lambda hourly: hourly.hour,
                                  "Minute":     lambda hourly: 0,
                                  "Type":       lambda hourly: "Hourly Average",
                                  "Solve Time": lambda hourly: _round(hourly.average_sced_runtime)}
            hr_runtime_writer = CsvReporter.from_dict(runtime_file, hr_runtime_columns, write_headers=False)
            stats_manager.register_for_hourly_stats(hr_runtime_writer.write_record)

        stats_manager.register_for_overall_stats(lambda overall: runtime_file.close())

    def setup_thermal_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        thermal_details_path = os.path.join(options.output_directory, 'thermal_detail.csv')
        thermal_details_file = open(thermal_details_path, 'w', newline='')
        thermal_details_columns = {
            'Date':       lambda ops,g: str(ops.timestamp.date()),
            'Hour':       lambda ops,g: ops.timestamp.hour,
            'Minute':     lambda ops,g: ops.timestamp.minute,
            'Generator':  lambda ops,g: g,
            'Dispatch':   lambda ops,g: _round(ops.observed_thermal_dispatch_levels[g]),
            'Dispatch DA':lambda ops,g: _round(ops.thermal_gen_cleared_DA[g]) \
                    if options.compute_market_settlements else None,
            'Headroom':   lambda ops,g: _round(ops.observed_thermal_headroom_levels[g]),
            'Unit State': lambda ops,g: ops.observed_thermal_states[g],
            'Unit Cost':  lambda ops,g: _round(ops.observed_costs[g]),
            'Unit Market Revenue': lambda ops,g: _round(
                ops.thermal_gen_revenue[g] + ops.thermal_total_reserve_revenue[g]) \
                if options.compute_market_settlements else None,
            'Unit Uplift Payment': lambda ops,g: _round(ops.thermal_uplift[g])\
                    if options.compute_market_settlements else None,
           }
        thermal_details_entries_per_report = lambda stats: stats.observed_thermal_dispatch_levels.keys()
        thermal_details_writer = CsvMultiRowReporter.from_dict(thermal_details_file, thermal_details_entries_per_report, thermal_details_columns)
        stats_manager.register_for_sced_stats(thermal_details_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: thermal_details_file.close())

    def setup_renewables_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        renewables_production_path = os.path.join(options.output_directory, 'renewables_detail.csv')
        renewables_production_file = open(renewables_production_path, 'w', newline='')
        renewables_production_entries_per_hour = lambda ops: ops.observed_renewables_levels.keys()
        renewables_production_columns = {
            'Date':       lambda ops,g: str(ops.timestamp.date()),
            'Hour':       lambda ops,g: ops.timestamp.hour,
            'Minute':     lambda ops,g: ops.timestamp.minute,
            'Generator':  lambda ops,g: g,
            'Output':     lambda ops,g: _round(ops.observed_renewables_levels[g]),
            'Output DA':  lambda ops,g: _round(ops.renewable_gen_cleared_DA[g]) \
                    if options.compute_market_settlements else None,
            'Curtailment':   lambda ops,g: _round(ops.observed_renewables_curtailment[g]),
            'Unit Market Revenue': lambda ops,g: _round(ops.renewable_gen_revenue[g]) \
                    if options.compute_market_settlements else None,
            'Unit Uplift Payment': lambda ops,g: _round(ops.renewable_uplift[g]) \
                    if options.compute_market_settlements else None,
           }
        renewables_production_writer = CsvMultiRowReporter.from_dict(renewables_production_file, renewables_production_entries_per_hour, renewables_production_columns)
        stats_manager.register_for_sced_stats(renewables_production_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: renewables_production_file.close())

    def setup_virtual_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        virtual_production_path = os.path.join(options.output_directory, 'virtual_detail.csv')
        virtual_production_file = open(virtual_production_path, 'w', newline='')
        virtual_production_entries_per_hour = lambda ops: ops.observed_virtual_dispatch_levels.keys()
        virtual_production_columns = {
            'Date':       lambda ops,g: str(ops.timestamp.date()),
            'Hour':       lambda ops,g: ops.timestamp.hour,
            'Minute':     lambda ops,g: ops.timestamp.minute,
            'Generator':  lambda ops,g: g,
            'Output':     lambda ops,g: _round(ops.observed_virtual_dispatch_levels[g]),
            'Output DA':  lambda ops,g: _round(ops.virtual_gen_cleared_DA[g]) \
                    if options.compute_market_settlements else None,
            'Unit Market Revenue': lambda ops,g: _round(ops.virtual_gen_revenue[g]) \
                    if options.compute_market_settlements else None,
            'Unit Uplift Payment': lambda ops,g: _round(ops.virtual_uplift[g]) \
                    if options.compute_market_settlements else None,
           }
        virtual_production_writer = CsvMultiRowReporter.from_dict(virtual_production_file, virtual_production_entries_per_hour, virtual_production_columns)
        stats_manager.register_for_sced_stats(virtual_production_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: virtual_production_file.close())

    def setup_bus_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        bus_path = os.path.join(options.output_directory, 'bus_detail.csv')
        bus_file = open(bus_path, 'w', newline='')
        bus_entries_per_hour = lambda ops: ops.observed_bus_mismatches.keys()
        bus_columns = {
            'Date':       lambda ops,b: str(ops.timestamp.date()),
            'Hour':       lambda ops,b: ops.timestamp.hour,
            'Minute':     lambda ops,b: ops.timestamp.minute,
            'Bus':        lambda ops,b: b,
            'Demand':     lambda ops,b: _round(ops.bus_demands[b]),
            'Shortfall':  lambda ops,b: _round(ops.observed_bus_mismatches[b]) \
                    if ops.observed_bus_mismatches[b] > 0.0 else 0.0,
            'Overgeneration':  lambda ops,b: _round(-ops.observed_bus_mismatches[b]) \
                    if ops.observed_bus_mismatches[b] < 0.0 else 0.0,
            'LMP':        lambda ops,b: _round(ops.observed_bus_LMPs[b]),
            'LMP DA':     lambda ops,b: _round(ops.planning_energy_prices[b]) \
                    if options.compute_market_settlements else None,
           }
        bus_writer = CsvMultiRowReporter.from_dict(bus_file, bus_entries_per_hour, bus_columns)
        stats_manager.register_for_sced_stats(bus_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: bus_file.close())

    def setup_line_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        line_path = os.path.join(options.output_directory, 'line_detail.csv')
        line_file = open(line_path, 'w', newline='')
        line_entries_per_hour = lambda ops: ops.observed_flow_levels.keys()
        line_columns = {'Date': lambda ops,l: str(ops.timestamp.date()),
                        'Hour': lambda ops,l: ops.timestamp.hour,
                        'Minute': lambda ops,l: ops.timestamp.minute,
                        'Line': lambda ops,l: l,
                        'Flow': lambda ops,l: _round(ops.observed_flow_levels[l]),
                        'Violation': lambda ops,l: _round(abs(ops.observed_flow_violation_levels[l])),
                   }
        line_writer = CsvMultiRowReporter.from_dict(line_file, line_entries_per_hour, line_columns)
        stats_manager.register_for_sced_stats(line_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: line_file.close())

    def setup_contingency_detail(self, options, stats_manager: StatsManager):
        _round = self._round
        line_path = os.path.join(options.output_directory, 'contingency_detail.csv')
        line_file = open(line_path, 'w', newline='')
        line_entries_per_hour = lambda ops: ops.observed_contingency_flow_levels.keys()
        line_columns = {'Date': lambda ops,c_l: str(ops.timestamp.date()),
                        'Hour': lambda ops,c_l: ops.timestamp.hour,
                        'Minute': lambda ops,c_l: ops.timestamp.minute,
                        'Contingency': lambda ops,c_l: c_l[0],
                        'Line': lambda ops,c_l: c_l[1],
                        'Flow': lambda ops,c_l: _round(ops.observed_contingency_flow_levels[c_l]),
                        'Violation': lambda ops,c_l: _round(abs(ops.observed_contingency_flow_violation_levels[c_l])),
                   }
        line_writer = CsvMultiRowReporter.from_dict(line_file, line_entries_per_hour, line_columns)
        stats_manager.register_for_sced_stats(line_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: line_file.close())

    def setup_reserves_detail(self, options, stats_manager:StatsManager):
        _round = self._round
        path = os.path.join(options.output_directory, 'reserves_detail.csv')
        file = open(path, 'w', newline='')
        columns = {
               'Date': lambda ops,r: str(ops.timestamp.date()),
               'Hour': lambda ops,r: ops.timestamp.hour,
             'Minute': lambda ops,r: ops.timestamp.minute,
            'Reserve': lambda ops,r: r.reserve_name,
              'Scope': lambda ops,r: r.scope,
          'Magnitude': lambda ops,r: _round(ops.reserve_requirements[r]) \
                                     if r in ops.rt_reserve_products \
                                     else None,
          'Shortfall': lambda ops,r: _round(ops.reserve_shortfalls[r]) \
                                     if r in ops.rt_reserve_products \
                                     else None,
              'Price': lambda ops,r: _round(ops.reserve_RT_prices[r]) \
                                     if r in ops.rt_reserve_products \
                                     else None,
        }
        if options.compute_market_settlements:
            columns.update({'DA Magnitude': lambda ops,r: _round(ops.DA_reserve_requirements[r]) \
                                                          if r in ops.da_reserve_products \
                                                          else None,
                            'DA Shortfall': lambda ops,r: _round(ops.DA_reserve_shortfalls[r]) \
                                                          if r in ops.da_reserve_products \
                                                          else None,
                                'DA Price': lambda ops,r: _round(ops.DA_reserve_prices[r]) \
                                                          if r in ops.da_reserve_products \
                                                          else None,
                           })
        rows_per_ops = lambda ops: ops.all_reserve_products
        writer = CsvMultiRowReporter.from_dict(file, rows_per_ops, columns)
        stats_manager.register_for_sced_stats(writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: file.close())

    def setup_hourly_gen_summary(self, options, stats_manager: StatsManager):
        _round = self._round
        hourly_gen_path = os.path.join(options.output_directory, 'hourly_gen_summary.csv')
        hourly_gen_file = open(hourly_gen_path, 'w', newline='')
        hourly_gen_columns = {'Date': lambda hourly: str(hourly.date),
                              'Hour': lambda hourly: hourly.hour,
                              'Load shedding': lambda hourly: _round(hourly.load_shedding),
                              'Reserve shortfall': lambda hourly: _round(hourly.total_reserve_shortfall),
                              'Available headroom': lambda hourly: _round(hourly.total_thermal_headroom),
                              'Over generation': lambda hourly: _round(hourly.over_generation)
                             }
        hourly_gen_writer = CsvReporter.from_dict(hourly_gen_file, hourly_gen_columns)
        stats_manager.register_for_hourly_stats(hourly_gen_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: hourly_gen_file.close())

    def setup_hourly_summary(self, options, stats_manager: StatsManager):
        _round = self._round
        hourly_path = os.path.join(options.output_directory, 'hourly_summary.csv')
        hourly_file = open(hourly_path, 'w', newline='')
        hourly_columns = {'Date': lambda hourly: str(hourly.date),
                          'Hour': lambda hourly: hourly.hour,
                          'TotalCosts': lambda hourly: _round(hourly.total_costs),
                          'FixedCosts': lambda hourly: _round(hourly.fixed_costs),
                          'VariableCosts': lambda hourly: _round(hourly.variable_costs),
                          'LoadShedding': lambda hourly: _round(hourly.load_shedding),
                          'OverGeneration': lambda hourly: _round(hourly.over_generation),
                          'ReserveShortfall': lambda hourly: _round(hourly.total_reserve_shortfall),
                          'RenewablesUsed': lambda hourly: _round(hourly.renewables_used),
                          'RenewablesCurtailment': lambda hourly: _round(hourly.renewables_curtailment),
                          'Demand': lambda hourly: _round(hourly.total_demand),
                          'Price': lambda hourly: _round(hourly.price)
                         }
        hourly_writer = CsvReporter.from_dict(hourly_file, hourly_columns)
        stats_manager.register_for_hourly_stats(hourly_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: hourly_file.close())

    def setup_daily_summary(self, options, stats_manager: StatsManager):
        _round = self._round
        daily_path = os.path.join(options.output_directory, 'daily_summary.csv')
        daily_file = open(daily_path, 'w', newline='')
        daily_columns = {'Date': lambda daily: str(daily.date),
                         'Demand': lambda daily: _round(daily.demand),
                         'Renewables available': lambda daily: _round(
                             daily.renewables_available),
                         'Renewables used': lambda daily: _round(daily.renewables_used),
                         'Renewables penetration rate': lambda daily: _round(
                             daily.renewables_penetration_rate),
                         'Average price': lambda daily: _round(daily.average_price),
                         'Fixed costs': lambda daily: _round(daily.fixed_costs),
                         'Generation costs': lambda daily: _round(daily.variable_costs),
                         'Load shedding': lambda daily: _round(daily.load_shedding),
                         'Over generation': lambda daily: _round(daily.over_generation),
                         'Reserve shortfall': lambda daily: _round(daily.reserve_shortfall),
                         'Renewables curtailment': lambda daily: _round(daily.renewables_curtailment),
                         'Number on/offs': lambda daily: daily.on_offs,
                         'Sum on/off ramps': lambda daily: _round(daily.sum_on_off_ramps),
                         'Sum nominal ramps': lambda daily: _round(daily.sum_nominal_ramps)}
        if options.compute_market_settlements:
            daily_columns.update({'Renewables energy payments':
                                      lambda daily: _round(daily.renewable_energy_payments),
                                  'Renewables uplift payments':
                                      lambda daily: _round(daily.renewable_uplift),
                                  'Thermal energy payments':
                                      lambda daily: _round(daily.thermal_energy_payments),
                                  'Thermal uplift payments':
                                      lambda daily: _round(daily.thermal_uplift),
                                  'Total energy payments':
                                      lambda daily: _round(daily.energy_payments),
                                  'Total uplift payments':
                                      lambda daily: _round(daily.uplift_payments),
                                  'Total reserve payments':
                                      lambda daily: _round(daily.reserve_payments),
                                  'Total payments':
                                      lambda daily: _round(daily.total_payments),
                                  'Average payments':
                                      lambda daily: _round(daily.average_payments),
                                 })
        daily_writer = CsvReporter.from_dict(daily_file, daily_columns)
        stats_manager.register_for_daily_stats(daily_writer.write_record)
        stats_manager.register_for_overall_stats(lambda overall: daily_file.close())

    def setup_overall_simulation_output(self, options, stats_manager: StatsManager):
        _round = self._round
        # We implement this as an inner function so that we can open and close the file
        # at the end of the simulation instead of keeping it open the whole session
        def write_overall_stats(overall: OverallStats):
            overall_path = os.path.join(options.output_directory, 'overall_simulation_output.csv')
            overall_file = open(overall_path, 'w', newline='')
            overall_cols = {'Total demand':            lambda overall: _round(overall.cumulative_demand),
                            'Total fixed costs':       lambda overall: _round(overall.total_overall_fixed_costs),
                            'Total generation costs':  lambda overall: _round(overall.total_overall_generation_costs),
                            'Total costs':             lambda overall: _round(overall.total_overall_costs),
                            'Total load shedding':     lambda overall: _round(overall.total_overall_load_shedding),
                            'Total over generation':   lambda overall: _round(overall.total_overall_over_generation),
                            'Total reserve shortfall': lambda overall: _round(overall.total_overall_reserve_shortfall),
                            'Total renewables curtailment': lambda overall: _round(overall.total_overall_renewables_curtailment),
                            'Total on/offs':           lambda overall: overall.total_on_offs,
                            'Total sum on/off ramps':  lambda overall: _round(overall.total_sum_on_off_ramps),
                            'Total sum nominal ramps': lambda overall: _round(overall.total_sum_nominal_ramps),
                            'Maximum observed demand': lambda overall: _round(overall.max_hourly_demand),
                            'Overall renewables penetration rate': lambda overall: _round(overall.overall_renewables_penetration_rate),
                            'Cumulative average price':lambda overall: _round(overall.cumulative_average_price)}
            if options.compute_market_settlements:
                overall_cols.update({'Total energy payments': lambda overall: _round(overall.total_energy_payments),
                                     'Total reserve payments':lambda overall: _round(overall.total_reserve_payments),
                                     'Total uplift payments': lambda overall: _round(overall.total_uplift_payments),
                                     'Total payments':        lambda overall: _round(overall.total_payments),
                                     'Cumulative average payments': lambda overall: _round(overall.cumulative_average_payments)})
            overall_reporter = CsvReporter.from_dict(overall_file, overall_cols)
            overall_reporter.write_record(overall)
            overall_file.close()
        stats_manager.register_for_overall_stats(write_overall_stats)


    def setup_daily_stack_graph(self, options, stats_manager: StatsManager):
        stats_manager.register_for_daily_stats(
            lambda daily_stats: ReportingManager.generate_stack_graph(
                options, daily_stats) )

    @staticmethod
    def generate_stack_graph(options, daily_stats: DailyStats):

        md_dict = ModelData.empty_model_data_dict()

        system = md_dict['system']

        # put just the HH:MM in the graph
        system['time_keys'] = [ str(opstats.timestamp.time())[0:5] for opstats in daily_stats.operations_stats() ]

        system['reserve_requirement'] = _time_series_dict(
                                            [sum(opstats.reserve_requirements.values())
                                             for opstats in daily_stats.operations_stats()
                                            ])
        system['reserve_shortfall'] = _time_series_dict(
                                            [sum(opstats.reserve_shortfalls.values())
                                             for opstats in daily_stats.operations_stats() 
                                            ])

        elements = md_dict['elements']

        elements['load'] = { 'system_load' :
                                { 'p_load' : _time_series_dict(
                                    [ opstats.total_demand for opstats in daily_stats.operations_stats() ]
                                    )
                                }
                            }
        elements['bus'] = { 'system_load_shed' :
                                { 'p_balance_violation' : _time_series_dict(
                                    [ opstats.load_shedding for opstats in daily_stats.operations_stats() ]
                                    )
                                },
                            'system_over_generation' :
                                { 'p_balance_violation' : _time_series_dict(
                                    [ -opstats.over_generation for opstats in daily_stats.operations_stats() ]
                                    )
                                }
                          }

        ## load in generation, storage
        generator_fuels = {}
        thermal_quickstart = {}
        thermal_dispatch = {}
        thermal_headroom = {}
        thermal_states = {}
        renewables_dispatch = {}
        renewables_curtailment = {}
        virtual_dispatch = {}
        storage_input_dispatch = {}
        storage_output_dispatch = {}
        storage_types = {}
        for opstats in daily_stats.operations_stats():
            _collect_time_assert_equal(opstats.generator_fuels, generator_fuels)
            _collect_time_assert_equal(opstats.quick_start_capable, thermal_quickstart)
            _collect_time_assert_equal(opstats.storage_types, storage_types)

            _collect_time(opstats.observed_thermal_dispatch_levels, thermal_dispatch)
            _collect_time(opstats.observed_thermal_headroom_levels, thermal_headroom)
            _collect_time(opstats.observed_thermal_states, thermal_states)

            _collect_time(opstats.observed_renewables_levels, renewables_dispatch)
            _collect_time(opstats.observed_renewables_curtailment, renewables_curtailment)

            _collect_time(opstats.observed_virtual_dispatch_levels, virtual_dispatch)

            _collect_time(opstats.storage_input_dispatch_levels, storage_input_dispatch)
            _collect_time(opstats.storage_output_dispatch_levels, storage_output_dispatch)

        # load generation
        gen_dict = {}
        for g, fuel in generator_fuels.items():
            gen_dict[g] = { 'fuel' : fuel , 'generator_type' : 'renewable' } # will get re-set below for thermal units
        for g, quickstart in thermal_quickstart.items():
            gen_dict[g]['fast_start'] = quickstart
            gen_dict[g]['generator_type'] = 'thermal'
        for g in virtual_dispatch:
            gen_dict[g]['generator_type'] = 'virtual'

        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_dispatch, 'pg')
        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_headroom, 'headroom')
        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_states, 'commitment')

        _add_timeseries_attribute_to_egret_dict(gen_dict, renewables_dispatch, 'pg')
        _add_timeseries_attribute_to_egret_dict(gen_dict, renewables_curtailment, 'curtailment')

        _add_timeseries_attribute_to_egret_dict(gen_dict, virtual_dispatch, 'pg')

        for g_dict in gen_dict.values():
            if g_dict['generator_type'] == 'renewable':
                pg = g_dict['pg']['values']
                curtailment = g_dict['curtailment']['values']
                g_dict['p_max'] = _time_series_dict([pg_val+c_val for pg_val, c_val in zip(pg, curtailment)])

        elements['generator'] = gen_dict

        # load storage
        storage_dict = {}
        for s, stype in storage_types.items():
            storage_dict[s] = { 'fuel' : stype }
        _add_timeseries_attribute_to_egret_dict(storage_dict, storage_input_dispatch, 'p_charge')
        _add_timeseries_attribute_to_egret_dict(storage_dict, storage_output_dispatch, 'p_discharge')

        elements['storage'] = storage_dict

        figure_path = os.path.join(options.output_directory, "plots","stackgraph_"+str(daily_stats.date)+".png")

        graphutils.generate_stack_graph(ModelData(md_dict),
                                        bar_width=1,
                                        x_tick_frequency=4*(60//options.sced_frequency_minutes),
                                        title=str(daily_stats.date),
                                        save_fig=figure_path)

    def setup_cost_summary_graph(self, options, stats_manager: StatsManager):
        stats_manager.register_for_overall_stats(
            lambda overall_stats: ReportingManager.generate_cost_summary_graph(options, overall_stats))

    @staticmethod
    def generate_cost_summary_graph(options, overall_stats: OverallStats):
        daily_fixed_costs = [daily_stats.fixed_costs for daily_stats in overall_stats.daily_stats]
        daily_generation_costs = [daily_stats.variable_costs for daily_stats in overall_stats.daily_stats]
        daily_load_shedding = [daily_stats.load_shedding for daily_stats in overall_stats.daily_stats]
        daily_over_generation = [daily_stats.over_generation for daily_stats in overall_stats.daily_stats]
        daily_reserve_shortfall = [daily_stats.reserve_shortfall for daily_stats in overall_stats.daily_stats]
        daily_renewables_curtailment = [daily_stats.renewables_curtailment for daily_stats in overall_stats.daily_stats]

        graphutils.generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs,
                                               daily_load_shedding, daily_over_generation,
                                               daily_reserve_shortfall,
                                               daily_renewables_curtailment,
                                               output_directory=os.path.join(options.output_directory, "plots"))

def _collect_time_assert_equal(input_dict, output_dict):
    for k,v in input_dict.items():
        if k not in output_dict:
            output_dict[k] = v
        assert output_dict[k] == v

def _collect_time(input_dict, output_dict):
    for k,v in input_dict.items():
        if k not in output_dict:
            output_dict[k] = []
        output_dict[k].append(v)

def _add_timeseries_attribute_to_egret_dict(egret_dict, attribute_dict, egret_attribute_name):
    for g, vals in attribute_dict.items():
        egret_dict[g][egret_attribute_name] = _time_series_dict(vals)
