# example test plugin
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prescient.simulator.config import PrescientConfig
    import prescient.plugins as pplugins

from pyomo.common.config import ConfigValue

def msg(callback_name, options=None):
    if options is None or options.print_callback_message:
        print(f"Called plugin function {callback_name}")


# This is a required function, must have this name and signature
def register_plugins(context: pplugins.PluginRegistrationContext, 
                     config: PrescientConfig) -> None:

    config.declare('print_callback_message', 
                   ConfigValue(domain=bool,
                               description='Print a message when callback is called',
                               default=False)).declare_as_argument()

    def hourly_stats_callback(hourly_stats):
        msg('hourly_stats_callback')
    context.register_for_hourly_stats(hourly_stats_callback)

    def daily_stats_callback(daily_stats):
        msg('daily_stats_callback')
    context.register_for_daily_stats(daily_stats_callback)

    def overall_stats_callback(overall_stats):
        msg('overall_stats_callback')
    context.register_for_overall_stats(overall_stats_callback)

    def options_preview_callback(options):
        msg('options_preview_callback', options)
    context.register_options_preview_callback(options_preview_callback)

    def initialization_callback(options, simulator):
        msg('initialization_callback', options)
    context.register_initialization_callback(initialization_callback)

    def before_ruc_solve_callback(options, simulator, ruc_model, uc_date, uc_hour):
        msg('before_ruc_solve_callback', options)
    context.register_before_ruc_solve_callback(before_ruc_solve_callback)

    def after_ruc_generation_callback(options, simulator, ruc_plan, uc_date, uc_hour):
        msg('after_ruc_generation_callback', options)
    context.register_after_ruc_generation_callback(after_ruc_generation_callback)

    def after_ruc_activation_callback(options, simulator):
        msg('after_ruc_activation_callback', options)
    context.register_after_ruc_activation_callback(after_ruc_activation_callback)

    def before_operations_solve_callback(options, simulator, sced_model):
        msg('before_operations_solve_callback', options)
    context.register_before_operations_solve_callback(before_operations_solve_callback)

    def after_operations_callback(options, simulator, sced_model):
        msg('after_operations_callback', options)
    context.register_after_operations_callback(after_operations_callback)

    def update_operations_stats_callback(options, simulator, operations_stats):
        msg('update_operations_stats_callback', options)
    context.register_update_operations_stats_callback(update_operations_stats_callback)
