# example test plugin

import prescient.plugins as pplugins

def msg(callback_name, options=None):
    if options is None or options.print_callback_message:
        print(f"Called plugin function {callback_name}")

pplugins.add_custom_commandline_argument('--print-callback-message',
                                         help='Print a message when callback is called',
                                         action='store_true',
                                         dest='print_callback_message',
                                         default=False)

def hourly_stats_callback(hourly_stats):
    msg('hourly_stats_callback')
pplugins.register_for_hourly_stats(hourly_stats_callback)

def daily_stats_callback(daily_stats):
    msg('daily_stats_callback')
pplugins.register_for_daily_stats(daily_stats_callback)

def overall_stats_callback(overall_stats):
    msg('overall_stats_callback')
pplugins.register_for_overall_stats(overall_stats_callback)

def options_preview_callback(options):
    msg('options_preview_callback', options)
pplugins.register_options_preview_callback(options_preview_callback)

def initialization_callback(options, simulator):
    msg('initialization_callback', options)
pplugins.register_initialization_callback(initialization_callback)

def before_ruc_solve_callback(options, simulator, ruc_model, uc_date, uc_hour):
    msg('before_ruc_solve_callback', options)
pplugins.register_before_ruc_solve_callback(before_ruc_solve_callback)

def after_ruc_generation_callback(options, simulator, ruc_plan, uc_date, uc_hour):
    msg('after_ruc_generation_callback', options)
pplugins.register_after_ruc_generation_callback(after_ruc_generation_callback)

def after_ruc_activation_callback(options, simulator):
    msg('after_ruc_activation_callback', options)
pplugins.register_after_ruc_activation_callback(after_ruc_activation_callback)

def before_operations_solve_callback(options, simulator, sced_model):
    msg('before_operations_solve_callback', options)
pplugins.register_before_operations_solve_callback(before_operations_solve_callback)

def after_operations_callback(options, simulator, sced_model):
    msg('after_operations_callback', options)
pplugins.register_after_operations_callback(after_operations_callback)

def update_operations_stats_callback(options, simulator, operations_stats):
    msg('update_operations_stats_callback', options)
pplugins.register_update_operations_stats_callback(update_operations_stats_callback)
