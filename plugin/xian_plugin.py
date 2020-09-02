from optparse import Option
import prescient.plugins

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
    #
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
