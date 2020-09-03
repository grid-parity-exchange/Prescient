from optparse import Option
import prescient.plugins

print("Hello from Xian's plugin")

# Add command line options
opt = Option('--track-ruc-signal',
             help='When tracking the market signal, RUC signals are used instead of the SCED signal.',
             action='store_true',
             dest='track_ruc_signal',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--track-sced-signal',
             help='When tracking the market signal, SCED signals are used instead of the RUC signal.',
             action='store_true',
             dest='track_sced_signal',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--hybrid-tracking',
             help='When tracking the market signal, hybrid model is used.',
             action='store_true',
             dest='hybrid_tracking',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--track-horizon',
             help="Specifies the number of hours in the look-ahead horizon "
                  "when each tracking process is executed.",
             action='store',
             dest='track_horizon',
             type='int',
             default=48)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--bidding-generator',
             help="Specifies the generator we derive bidding strategis for.",
             action='store',
             dest='bidding_generator',
             type='string',
             default='102_STEAM_3')
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--bidding',
             help="Invoke generator strategic bidding when simulate.",
             action='store_true',
             dest='bidding',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--deviation-weight',
             help="Set the weight for deviation term when tracking",
             action='store',
             dest='deviation_weight',
             type='float',
             default=30)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--ramping-weight',
             help="Set the weight for ramping term when tracking",
             action='store',
             dest='ramping_weight',
             type='float',
             default=20)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--cost-weight',
             help="Set the weight for cost term when tracking",
             action='store',
             dest='cost_weight',
             type='float',
             default=1)
prescient.plugins.add_custom_commandline_option(opt)

### End add new command line options ###

from strategic_bidding import DAM_thermal_bidding
import dateutil.parser
import numpy as np

def initialize_plugin(options, simulator):
    # Xian: add 2 np arrays to store RUC and SCED schedules for the interested generator
    simulator.data_manager.extensions['ruc_schedule_arr'] = np.zeros((24,options.num_days))
    simulator.data_manager.extensions['sced_schedule_arr'] = np.zeros((24,options.num_days))

    # Xian: add 2 np arrays to store
    # 1. the actual total power output from the hybrid system
    # 2. power output from the thermal generator in the hybrid system
    simulator.data_manager.extensions['total_power_delivered_arr'] = np.zeros((24,options.num_days)) # P_R
    simulator.data_manager.extensions['thermal_power_delivered_arr'] = np.zeros((24,options.num_days)) # P_G
    simulator.data_manager.extensions['thermal_power_generated_arr'] = np.zeros((24,options.num_days)) # P_T

    # initialize the model class
    thermal_bid = DAM_thermal_bidding(n_scenario=10)
    simulator.data_manager.extensions['thermal_bid'] = thermal_bid

    first_date = str(dateutil.parser.parse(options.start_date).date())

    # build bidding model
    if options.bidding:
        m_bid = thermal_bid.create_bidding_model(generator = options.bidding_generator)
        price_forecast_dir = '../../prescient/plugins/price_forecasts/date={}_lmp_forecasts.csv'.format(first_date)
        cost_curve_store_dir = '../../prescient/plugins/cost_curves/'

        # solve the bidding model for the first simulation day
        #
        # Consider moving out of initialization, do it later in the process
        thermal_bid.stochastic_bidding(m_bid,price_forecast_dir,cost_curve_store_dir,first_date)

        simulator.data_manager.extensions['cost_curve_store_dir'] = cost_curve_store_dir
        simulator.data_manager.extensions['price_forecast_dir'] = price_forecast_dir
        simulator.data_manager.extensions['m_bid'] = m_bid

    # build tracking model
    if options.track_ruc_signal:
        print('Building a track model for RUC signals.')
        simulator.data_manager.extensions['m_track_ruc'] = \
            thermal_bid.build_tracking_model(options.ruc_horizon, generator = options.bidding_generator,
                                            track_type = 'RUC', hybrid = options.hybrid_tracking)

    elif options.track_sced_signal:
        print('Building a track model for SCED signals.')
        simulator.data_manager.extensions['m_track_sced'] = \
            thermalBid.build_tracking_model(options.sced_horizon, generator = options.bidding_generator,
                                            track_type = 'SCED',hybrid = options.hybrid_tracking)

        # initialize a list/array to record the power output of sced tracking
        # model in real-time, so it can update the initial condition properly
        # every hour
        simulator.data_manager.extensions['sced_tracker_power_record'] = \
            {options.bidding_generator: 
             np.repeat(value(m_track_sced.pre_pow[options.bidding_generator]),\
                       value(m_track_sced.pre_up_hour[options.bidding_generator]))
            }

prescient.plugins.register_initialization_callback(initialize_plugin)


def tweak_sced_before_solve(options, simulator, sced_instance):
    current_time = simulator.time_manager.current_time
    hour = current_time.hour
    date_as_string = current_time.date

    gen_name = options.bidding_generator
    gpionts, gvalues = get_gpoints_gvalues(simulator.data_manager.extensions['cost_curve_store_dir'],
                                            date=date_as_string, gen_name=gen_name)
    gen_dict = sced_instance.data['elements']['generator'][gen_name]

    p_cost = [(gpnt, gval) in zip(gpoints[hour], gvalues[hour])]

    gen_dict['p_cost'] = {'data_type' : 'cost_curve', 
                          'cost_curve_type':'piecewise', 
                          'values':p_cost 
                         }

prescient.plugins.register_before_operations_solve_callback(tweak_sced_before_solve)

def after_sced(options, simulator, sced_instance):
    current_time = simulator.time_manager.current_time
    h = current_time.hour
    date_as_string = current_time.date

    date_idx = simulator.time_manager.dates_to_simulate.index(date_as_string)

    g_dict = sced_instance.data['elements']['generator'][gen_name]
    sced_dispatch_level = {g: g_dict['pg']['values']}

    sced_schedule_arr = simulator.data_manager.extensions['sced_schedule_arr']
    sced_schedule_arr[h,date_idx] = sced_dispatch_level[options.bidding_generator][0]

    ## TODO: pass the real-time price into the function here
    # get lmps in the current planning horizon
    #get_lmps_for_deterministic_sced(lmp_sced_instance, max_bus_label_length=max_bus_label_length)

    ruc_dispatch_level_current = simulator.data_manager.extensions['ruc_dispatch_level_current']
    # slice the ruc dispatch for function calling below
    ruc_dispatch_level_for_current_sced_track = {options.bidding_generator:\
                    ruc_dispatch_level_current[options.bidding_generator][h:h+options.sced_horizon]}

    thermalBid = simulator.data_manager.extensions['thermal_bid']
    m_track_sced = simulator.data_manager.extensions['m_track_sced']
    thermalBid.pass_schedule_to_track_and_solve(m_track_sced,\
                                                ruc_dispatch_level_for_current_sced_track,\
                                                SCED_dispatch = sced_dispatch_level,\
                                                deviation_weight = options.deviation_weight, \
                                                ramping_weight = options.ramping_weight,\
                                                cost_weight = options.cost_weight)

    # record the track power output profile
    if options.hybrid_tracking == False:
        track_gen_pow_sced = thermalBid.extract_pow_s_s(m_track_sced,\
                                                    horizon = options.sced_horizon, verbose = False)
        thermal_track_gen_pow_sced = track_gen_pow_sced
        thermal_generated_sced = track_gen_pow_sced
    else:
        # need to extract P_R and P_T
        # for control power recording and updating the model
        track_gen_pow_sced, thermal_track_gen_pow_sced, thermal_generated_sced =\
                                        thermalBid.extract_pow_s_s(m_track_sced,horizon =\
                                        options.sced_horizon, hybrid = True,verbose = False)

    # record the total power delivered
    # and thermal power delivered
    total_power_delivered_arr[h,date_idx] = track_gen_pow_sced[options.bidding_generator][0]

    thermal_power_delivered_arr[h,date_idx] = thermal_track_gen_pow_sced[options.bidding_generator][0]

    thermal_power_generated_arr[h,date_idx] = thermal_generated_sced[options.bidding_generator][0]

    # use the schedule for this step to update the recorder
    sced_tracker_power_record = simulator.data_manager.extensions['sced_tracker_power_record']
    sced_tracker_power_record[options.bidding_generator][:-1] = sced_tracker_power_record[options.bidding_generator][1:]
    sced_tracker_power_record[options.bidding_generator][-1] = thermal_generated_sced[options.bidding_generator][0]

    # update the track model
    thermalBid.update_model_params(m_track_sced,sced_tracker_power_record, hybrid = options.hybrid_tracking)
    thermalBid.reset_constraints(m_track_sced,options.sced_horizon)

prescient.plugins.register_after_operations_generation_callback(after_sced)

def update_observed_thermal_dispatch(options, simulator, ops_stats):

    ## TODO: read current observed_thermal_dispatch and adjust observed_thermal_head_room
    ##       base on the difference with actuals
    ## TODO: get track_gen_pow_ruc and track_gen_pow_sced from simulator.data_manager.extensions 
    h = simulator.time_manager.hour
    if options.track_ruc_signal:
        print('Making changes in observed power output using tracking RUC model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = track_gen_pow_ruc[g][h]

    elif options.track_sced_signal:
        print('Making changes in observed power output using tracking SCED model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = track_gen_pow_sced[g][0]

prescient.plugins.register_update_operations_stats_callback(update_observed_thermal_dispatch)

def tweak_ruc_before_solve(options, simulator, ruc_instance, ruc_date, ruc_hour):
    if not options.bidding:
        return
    print("Getting cost cuves for UC.\n")
    current_time = simulator.time_manager.current_time
    if current_time is not None:
        date_as_string = ruc_date

        thermalBid = simulator.data_manager.extensions['thermal_bid']
        m_bid = simulator.data_manager.extensions['m_bid']

        # Xian: solve bidding problem here
        thermalBid.update_model_params(m_bid,ruc_dispatch_level_current)
        thermalBid.reset_constraints(m_bid,options.ruc_horizon)

        price_forecast_dir = simulation.data_manager.extensions['price_forecast_dir']
        cost_curve_store_dir = simulation.data_manager.extensions['cost_curve_store_dir']

        # solve the bidding model for the first simulation day
        thermalBid.stochastic_bidding(m_bid,price_forecast_dir,cost_curve_store_dir,date_as_string)

    else: # first RUC solve
        date_as_string = ruc_date

    gen_name = options.bidding_generator
    gpionts, gvalues = get_gpoints_gvalues(simulator.data_manager.extensions['cost_curve_store_dir'],
                                            date=date_as_string, gen_name=gen_name)
    gen_dict = ruc_instance.data['elements']['generator'][gen_name]

    p_cost = [[(gpnt, gval) in zip(gpoints[t], gvalues[t])] for t in range(24)]

    gen_dict['p_cost'] = {'data_type': 'time_series', 
                          'values': [{'data_type' : 'cost_curve', 
                                     'cost_curve_type':'piecewise', 
                                     'values':p_cost[t]} for t in range(24)]
                         }

prescient.plugins.register_before_ruc_solve_callback(tweak_ruc_before_solve)

def after_ruc(options, simulator, ruc_plan, ruc_date, ruc_hour):
    if not options.track_ruc_signal:
        return
    ruc_instance = ruc_plan.deterministic_ruc_instance
    
    date_idx = simulator.time_manager.dates_to_simulate.index(ruc_date)

    gen_name = options.bidding_generator
    
    g_dict = ruc_instance.data['elements']['generator'][gen_name]
    ruc_dispatch_level_for_next_period = {g: g_dict['pg']['values']}

    simulator.data_manager.extensions['ruc_dispatch_level_for_next_period'] = \
            ruc_dispatch_level_for_next_period 

    ruc_schedule_arr = simulator.data_manager.extensions['ruc_schedule_arr']

    # record the ruc signal from the first day
    ruc_schedule_arr[:,date_idx] = np.array(ruc_dispatch_level_for_next_period[options.bidding_generator]).flatten()[:options.ruc_horizon]

    m_track_ruc = simulator.data_manager.extensions['m_track_ruc']
    thermalBid = simulator.data_manager.extensions['thermal_bid']

    thermalBid.pass_schedule_to_track_and_solve(m_track_ruc,ruc_dispatch_level_for_next_period,\
                    RT_price=None, deviation_weight = options.deviation_weight, \
                    ramping_weight = options.ramping_weight,\
                    cost_weight = options.cost_weight)


    # record the track power output profile
    if options.hybrid_tracking == False:
        track_gen_pow_ruc = thermalBid.extract_pow_s_s(m_track_ruc, horizon =\
                            options.ruc_horizon, verbose = False)
        thermal_track_gen_pow_ruc = track_gen_pow_ruc
        thermal_generated_ruc = track_gen_pow_ruc
    else:
        track_gen_pow_ruc, thermal_track_gen_pow_ruc,thermal_generated_ruc =\
                            thermalBid.extract_pow_s_s(m_track_ruc,horizon =\
                            options.ruc_horizon, hybrid = True,verbose = False)

    # record the total power delivered
    # and thermal power delivered
    total_power_delivered_arr = simulator.data_manager.extensions['total_power_delivered_arr']
    thermal_power_delivered_arr = simulator.data_manager.extensions['thermal_power_delivered_arr']
    thermal_power_generated_arr = simulator.data_manager.extensions['thermal_power_generated_arr']

    total_power_delivered_arr[:,date_idx] = track_gen_pow_ruc[options.bidding_generator]
    thermal_power_delivered_arr[:,date_idx] = thermal_track_gen_pow_ruc[options.bidding_generator]
    thermal_power_generated_arr[:,date_idx] = thermal_generated_ruc[options.bidding_generator]

    # update the track model
    thermalBid.update_model_params(m_track_ruc,thermal_generated_ruc,hybrid = options.hybrid_tracking)
    thermalBid.reset_constraints(m_track_ruc,options.ruc_horizon)

prescient.plugins.register_after_ruc_generation_callback(after_ruc)

def after_ruc_activation(options, simulator):

    simulator.data_manager.extensions['ruc_dispatch_level_current'] = \
            simulator.data_manager.extensions['ruc_dispatch_level_for_next_period']
