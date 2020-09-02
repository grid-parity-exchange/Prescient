# Xian Gao
# Dowling Lab
# xgao1@nd.edu

import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy

class DAM_thermal_bidding:

    ## define a few class properties

    gen_name = ['101_CT_1','102_STEAM_3']

    n_units = len(gen_name)

    # fuel cost $/MMBTU
    fuel_cost = np.array([10.3494,2.11399])

    # start up heat MMBTU
    start_up_heat = np.array([5,5284.8])

    # start up cost
    start_cost = dict(zip(gen_name, fuel_cost * start_up_heat))

    capacity = {'101_CT_1':20,'102_STEAM_3':76}
    min_pow = {'101_CT_1':8,'102_STEAM_3':30}
    ramp_up = {'101_CT_1':180,'102_STEAM_3':120}
    ramp_dw = {'101_CT_1':180,'102_STEAM_3':120}
    ramp_start_up = {'101_CT_1':8,'102_STEAM_3':76}
    ramp_shut_dw = {'101_CT_1':8,'102_STEAM_3':76}
    min_dw_time = {'101_CT_1':1,'102_STEAM_3':4}
    min_up_time = {'101_CT_1':1,'102_STEAM_3':8}

    output_pts = {'101_CT_1': np.array([0,0.4,0.6,0.8,1]),\
    '102_STEAM_3':np.array([0,0.394736842,0.596491228,0.798245614,1])}
    output = {'101_CT_1':np.around(output_pts['101_CT_1']*capacity['101_CT_1'],1),\
    '102_STEAM_3':np.around(output_pts['102_STEAM_3']*capacity['102_STEAM_3'],1)} #MW
    power_increae = {'101_CT_1':np.diff(output['101_CT_1']),\
    '102_STEAM_3':np.diff(output['102_STEAM_3'])}
    increased_hr = {'101_CT_1': np.array([13114,9456,9476,10352])/1000,\
    '102_STEAM_3': np.array([11591,8734,9861,10651])/1000}
    prod_cost = {'101_CT_1': (increased_hr['101_CT_1']*fuel_cost[0]),\
    '102_STEAM_3': (increased_hr['102_STEAM_3']*fuel_cost[1])}
    tot_prod_cost = {'101_CT_1':np.cumsum(prod_cost['101_CT_1']*power_increae['101_CT_1']),\
    '102_STEAM_3':np.cumsum(prod_cost['102_STEAM_3']*power_increae['102_STEAM_3'])}

    diff_penalty = 30

    n_seg = 4

    def __init__(self,n_scenario):
        self.n_scenario = n_scenario

    # define the function
    def DAM_thermal(self, plan_horizon=24,generator = None,n_scenario = None):

        '''
        Input:
        1. plan_horizon: length of the planning horizon
        2. n_scenario: number of scenarios in DAM
        3. n_seg: number of segments needed to approx the cost function

        '''

        if n_scenario == None:
            n_scenario = self.n_scenario

        ## build the model
        m = ConcreteModel()

        ## define the sets
        m.HOUR = Set(initialize = range(plan_horizon))

        m.SCENARIOS = Set(initialize = range(n_scenario))
        m.SEGMENTS = Set(initialize = range(len(self.power_increae[self.gen_name[0]])))
        #m.SEGMENTS_bid = Set(initialize = range(self.n_seg))

        # the set for the genration units
        if generator == None:
            m.UNITS = Set(initialize = self.gen_name,ordered = True)
        else:
            m.UNITS = Set(initialize = [generator],ordered = True)

        ## define the parameters

        # DAM price forecasts
        m.DAM_price = Param(m.HOUR,m.SCENARIOS,initialize = 20, mutable = True)

        start_cost_dict = {i:self.start_cost[i] for i in m.UNITS}
        m.start_up_cost = Param(m.UNITS,initialize = start_cost_dict,mutable = False)

        #fix_cost_dollar_dict = {i: self.fix_cost_dollar[i] for i in m.UNITS}
        #m.fix_cost = Param(m.UNITS,initialize = 0,mutable = False)

        # production cost
        prod_cost_dict = {(i,l):self.prod_cost[i][l] for i in m.UNITS for l in m.SEGMENTS}
        m.prod_cost = Param(m.UNITS,m.SEGMENTS,initialize = prod_cost_dict,mutable = False)

        # capacity of generators: upper bound (MW)
        capacity_dict = {i: self.capacity[i] for i in m.UNITS}
        m.gen_cap = Param(m.UNITS,initialize = capacity_dict, mutable = True)

        # minimum power of generators: lower bound (MW)
        min_pow_dict = {i: self.min_pow[i] for i in m.UNITS}
        m.gen_min_pow = Param(m.UNITS,initialize = min_pow_dict, mutable = True)

        def seg_len_rule(m,j,l):
            return self.power_increae[j][l]
        #m.seg_len = Param(m.UNITS,m.SEGMENTS,initialize = seg_len_rule, mutable = False)

        # Ramp up limits (MW/h)
        ramp_up_dict = {i:self.ramp_up[i] for i in m.UNITS}
        m.ramp_up = Param(m.UNITS,initialize = ramp_up_dict,mutable = True)

        # Ramp down limits (MW/h)
        ramp_dw_dict = {i:self.ramp_dw[i] for i in m.UNITS}
        m.ramp_dw = Param(m.UNITS,initialize = ramp_dw_dict,mutable = True)

        # start up ramp limit
        ramp_start_up_dict = {i:self.ramp_start_up[i] for i in m.UNITS}
        m.ramp_start_up = Param(m.UNITS,initialize = ramp_start_up_dict)

        # shut down ramp limit
        ramp_shut_dw_dict = {i:self.ramp_shut_dw[i] for i in m.UNITS}
        m.ramp_shut_dw = Param(m.UNITS,initialize = ramp_shut_dw_dict)

        # minimum down time [hr]
        min_dw_time_dict = {i:self.min_dw_time[i] for i in m.UNITS}
        m.min_dw_time = Param(m.UNITS,initialize = min_dw_time_dict)

        # minimum up time [hr]
        min_up_time_dict = {i:self.min_up_time[i] for i in m.UNITS}
        m.min_up_time = Param(m.UNITS,initialize = min_up_time_dict)

        # power from the previous day (MW)
        # need to assume the power output is at least always at the minimum pow output

        # on/off status from previous day
        m.pre_on_off = Param(m.UNITS,within = Binary,default= 1,mutable = True)

        # number of hours the plant has been on/off in the previous day
        m.pre_up_hour = Param(m.UNITS,within = Integers,initialize = 24,mutable = True)
        m.pre_dw_hour = Param(m.UNITS,within = Integers,initialize = 0, mutable = True)

        # define a function to initialize the previous power params
        def init_pre_pow_fun(m,j):
            return m.pre_on_off[j]*m.gen_min_pow[j]
        m.pre_pow = Param(m.UNITS,initialize = init_pre_pow_fun, mutable = True)
        #m.pre_pow = Expression(m.UNITS,rule = init_pre_pow_fun)

        ## define the variables

        # generator power (MW)
        m.gen_pow = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = NonNegativeReals)

        # generator energy (MWh)
        m.gen_eng = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = NonNegativeReals)

        # binary variables indicating on/off
        m.on_off = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = True,within = Binary)

        # binary variables indicating  start_up
        m.start_up = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False,within = Binary)

        # binary variables indicating shut down
        m.shut_dw = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False, within = Binary)

        # power produced in each segment
        m.power_segment = Var(m.UNITS,m.HOUR,m.SCENARIOS,m.SEGMENTS,within = NonNegativeReals)

        ## Constraints

        # energy-power relation
        def eng_pow_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h == 0:
                return m.gen_eng[j,h,k] == (m.pre_pow[j] + m.gen_pow[j,h,k])/2
            else:
                return m.gen_eng[j,h,k] == (m.gen_pow[j,h-1,k] + m.gen_pow[j,h,k])/2
        def eng_pow_fun2(m,j,h,k):
            return m.gen_eng[j,h,k] == m.gen_pow[j,h,k]
        m.eng_pow_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = eng_pow_fun2)

        # linearized power
        def linear_power_fun(m,j,h,k):
            return m.gen_pow[j,h,k] == \
            sum(m.power_segment[j,h,k,l] for l in m.SEGMENTS)
        m.linear_power = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = linear_power_fun)

        # bounds on segment power
        def seg_pow_bnd_fun(m):
            for j in m.UNITS:
                for h in m.HOUR:
                    for k in m.SCENARIOS:
                        for l in m.SEGMENTS:
                            yield m.power_segment[j,h,k,l]<= self.power_increae[j][l]
        m.seg_pow_bnd = ConstraintList(rule = seg_pow_bnd_fun)

        # start up and shut down logic (Arroyo and Conejo 2000)
        def start_up_shut_dw_fun(m,j,h,k):
            if h == 0:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] -\
                m.pre_on_off[j]
            else:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] -\
                m.on_off[j,h-1,k]
        m.start_up_shut_dw = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = \
        start_up_shut_dw_fun)

        # either start up or shut down
        def start_up_or_shut_dw_fun(m,j,h,k):
            return m.start_up[j,h,k] + m.shut_dw[j,h,k] <= 1
        m.start_up_or_shut_dw = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = start_up_or_shut_dw_fun)

        # bounds on gen_pow
        def lhs_bnd_gen_pow_fun(m,j,h,k):
            return m.on_off[j,h,k] * m.gen_min_pow[j] <= m.gen_pow[j,h,k]
        m.lhs_bnd_gen_pow = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = lhs_bnd_gen_pow_fun)

        def rhs_bnd_gen_pow_fun(m,j,h,k):
            return m.gen_pow[j,h,k] <= m.on_off[j,h,k] * m.gen_cap[j]
        m.rhs_bnd_gen_pow = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = rhs_bnd_gen_pow_fun)

        # ramp up limits
        def ramp_up_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.gen_pow[j,h,k] <= m.pre_pow[j] \
                + m.ramp_up[j]*m.pre_on_off[j]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]\
                + m.gen_cap[j]*(1-m.on_off[j,h,k])
            else:
                return m.gen_pow[j,h,k] <= m.gen_pow[j,h-1,k] \
                + m.ramp_up[j]*m.on_off[j,h-1,k]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]\
                + m.gen_cap[j]*(1-m.on_off[j,h,k])
        m.ramp_up_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_up_fun)

        # ramp shut down limits
        def ramp_shut_dw_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.pre_pow[j] <= m.gen_cap[j]*m.on_off[j,h,k] \
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
            else:
                return m.gen_pow[j,h-1,k] <= m.gen_cap[j]*m.on_off[j,h,k] \
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
        m.ramp_shut_dw_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_shut_dw_fun)

        # ramp down limits
        def ramp_dw_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h == 0:
                return m.pre_pow[j] - m.gen_pow[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]\
                + m.gen_cap[j] * (1 - m.pre_on_off[j])
            else:
                return m.gen_pow[j,h-1,k] - m.gen_pow[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]\
                + m.gen_cap[j] * (1 - m.on_off[j,h-1,k])
        m.ramp_dw_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_dw_fun)

        # expressions for min up/down time
        def G_expr(m,j):
            # temp = max(0,value(m.min_up_time[j]-m.pre_up_hour[j]))
            # return min(plan_horizon,temp*value(m.pre_on_off[j]))
            return (m.min_up_time[j]-m.pre_up_hour[j])*m.pre_on_off[j]
        m.G = Expression(m.UNITS,rule = G_expr)

        def L_expr(m,j):
            # temp = max(0,value(m.min_dw_time[j]-m.pre_dw_hour[j]))
            # return min(plan_horizon,temp*(1-value(m.pre_on_off[j])))
            return (m.min_dw_time[j]-m.pre_dw_hour[j])*(1-m.pre_on_off[j])
        m.L = Expression(m.UNITS,rule = L_expr)

        ## min up time constraints
        def minimum_on_fun(m,j,k):
            if value(m.G[j]) <= 0:
                return Constraint.Skip
            elif 0<value(m.G[j])<= plan_horizon:
                return sum(1-m.on_off[j,h,k] for h in range(int(value(m.G[j])))) == 0
            else:
                return sum(1-m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        m.minimum_on_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_on_fun)

        # minimum up time in the subsequent hours
        def sub_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.G[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.G[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_up_time[j]+1):
                        yield sum(m.on_off[j,h,k] for h in range(h0,h0 + m.min_up_time[j]))\
                        >= m.min_up_time[j]*m.start_up[j,h0,k]
        m.sub_minimum_on_con = ConstraintList(rule = sub_minimum_on_fun)

        # minimum up time in the final hours
        def fin_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_up_time[j]):
                        # if goes into here, sub_minimum_on_con must have been
                        # skipped.
                        if value(m.G[j]) <= 0:
                            # if goes into here, minimum_on_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                        else:
                            for h0 in range(int(value(m.G[j])),plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_up_time[j]+1,plan_horizon):
                            yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                            for h in range(h0,plan_horizon)) >=0
        m.fin_minimum_on_con = ConstraintList(rule = fin_minimum_on_fun)

        ## min down time constraints
        def minimum_off_fun(m,j,k):
            if value(m.L[j])<= 0:
                return Constraint.Skip
            elif 0<value(m.L[j])<= plan_horizon:
                return sum(m.on_off[j,h,k] for h in range(int(value(m.L[j])))) == 0
            else:
                return sum(m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        m.minimum_off_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_off_fun)

        # minimum down time in the subsequent hours
        def sub_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.L[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.L[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_dw_time[j]+1):
                        yield sum(1-m.on_off[j,h,k] for h in range(h0,h0 + m.min_dw_time[j]))\
                        >= m.min_dw_time[j]* m.shut_dw[j,h0,k]
        m.sub_minimum_dw_con = ConstraintList(rule = sub_minimum_dw_fun)

        # minimum down time in the final hours
        def fin_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_dw_time[j]):
                        # if goes into here, sub_minimum_dw_con must have been
                        # skipped.
                        if value(m.L[j]) <= 0:
                            # if goes into here, minimum_off_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0
                        else:
                            for h0 in range(int(value(m.L[j])),plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_dw_time[j]+1,plan_horizon):
                            yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                            for h in range(h0,plan_horizon)) >=0
        m.fin_minimum_dw_con = ConstraintList(rule = fin_minimum_dw_fun)

        ## Expression
        # energy generated by all the units
        def total_eng_fun(m,h,k):
            return sum(m.gen_eng[j,h,k] for j in m.UNITS)
        m.tot_eng = Expression(m.HOUR,m.SCENARIOS,rule = total_eng_fun)

        def prod_cost_approx_fun(m,j,h,k):
            return m.prod_cost[j,0]*m.gen_min_pow[j]*m.on_off[j,h,k]\
            +sum(m.prod_cost[j,l]*m.power_segment[j,h,k,l] for l in range(1,len(m.SEGMENTS)))
        m.prod_cost_approx = Expression(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = prod_cost_approx_fun)

        # total cost
        def tot_cost_fun(m,h,k):
            return sum(m.prod_cost_approx[j,h,k]\
            + m.start_up_cost[j]*m.start_up[j,h,k] for j in m.UNITS)
        m.tot_cost = Expression(m.HOUR,m.SCENARIOS,rule = tot_cost_fun)

        ## Objective
        def exp_revenue_fun(m):
            return sum(m.tot_eng[h,k]*m.DAM_price[h,k]- m.tot_cost[h,k]\
            for h in m.HOUR for k in m.SCENARIOS)/n_scenario
        m.exp_revenue = Objective(rule = exp_revenue_fun,sense = maximize)

        return m

    # define the function
    def DAM_hybrid(self,plan_horizon=24,generator = None,n_scenario = None, storage_size = 4, storage_eff = 0.88):

        '''
        Input:
        1. n_scenario: number of scenarios in DAM
        '''

        if n_scenario == None:
            n_scenario = self.n_scenario

        ## build the model
        m = ConcreteModel()

        ## define the sets
        m.HOUR = Set(initialize = range(plan_horizon))

        m.SCENARIOS = Set(initialize = range(n_scenario))
        m.SEGMENTS = Set(initialize = range(len(self.power_increae[self.gen_name[0]])))
        #m.SEGMENTS_bid = Set(initialize = range(self.n_seg))

        # the set for the genration units
        if generator == None:
            m.UNITS = Set(initialize = self.gen_name,ordered = True)
        else:
            m.UNITS = Set(initialize = [generator],ordered = True)

        ## define the parameters

        # DAM price forecasts
        m.DAM_price = Param(m.HOUR,m.SCENARIOS,initialize = 20, mutable = True)

        start_cost_dict = {i:self.start_cost[i] for i in m.UNITS}
        m.start_up_cost = Param(m.UNITS,initialize = start_cost_dict,mutable = False)

        #fix_cost_dollar_dict = {i: self.fix_cost_dollar[i] for i in m.UNITS}
        #m.fix_cost = Param(m.UNITS,initialize = 0,mutable = False)

        # production cost
        prod_cost_dict = {(i,l):self.prod_cost[i][l] for i in m.UNITS for l in m.SEGMENTS}
        m.prod_cost = Param(m.UNITS,m.SEGMENTS,initialize = prod_cost_dict,mutable = False)

        # capacity of generators: upper bound (MW)
        capacity_dict = {i: self.capacity[i] for i in m.UNITS}
        m.gen_cap = Param(m.UNITS,initialize = capacity_dict, mutable = True)

        # minimum power of generators: lower bound (MW)
        min_pow_dict = {i: self.min_pow[i] for i in m.UNITS}
        m.gen_min_pow = Param(m.UNITS,initialize = min_pow_dict, mutable = True)

        def seg_len_rule(m,j,l):
            return self.power_increae[j][l]
        #m.seg_len = Param(m.UNITS,m.SEGMENTS,initialize = seg_len_rule, mutable = False)

        # Ramp up limits (MW/h)
        ramp_up_dict = {i:self.ramp_up[i] for i in m.UNITS}
        m.ramp_up = Param(m.UNITS,initialize = ramp_up_dict,mutable = True)

        # Ramp down limits (MW/h)
        ramp_dw_dict = {i:self.ramp_dw[i] for i in m.UNITS}
        m.ramp_dw = Param(m.UNITS,initialize = ramp_dw_dict,mutable = True)

        # start up ramp limit
        ramp_start_up_dict = {i:self.ramp_start_up[i] for i in m.UNITS}
        m.ramp_start_up = Param(m.UNITS,initialize = ramp_start_up_dict)

        # shut down ramp limit
        ramp_shut_dw_dict = {i:self.ramp_shut_dw[i] for i in m.UNITS}
        m.ramp_shut_dw = Param(m.UNITS,initialize = ramp_shut_dw_dict)

        # minimum down time [hr]
        min_dw_time_dict = {i:self.min_dw_time[i] for i in m.UNITS}
        m.min_dw_time = Param(m.UNITS,initialize = min_dw_time_dict)

        # minimum up time [hr]
        min_up_time_dict = {i:self.min_up_time[i] for i in m.UNITS}
        m.min_up_time = Param(m.UNITS,initialize = min_up_time_dict)

        # power from the previous day (MW)
        # need to assume the power output is at least always at the minimum pow output

        # on/off status from previous day
        m.pre_on_off = Param(m.UNITS,within = Binary,default= 1,mutable = True)

        # number of hours the plant has been on/off in the previous day
        m.pre_up_hour = Param(m.UNITS,within = Integers,initialize = 24,mutable = True)
        m.pre_dw_hour = Param(m.UNITS,within = Integers,initialize = 0, mutable = True)

        # define a function to initialize the previous power params
        def init_pre_pow_fun(m,j):
            return m.pre_on_off[j]*m.gen_min_pow[j]
        m.pre_pow = Param(m.UNITS,initialize = init_pre_pow_fun, mutable = True)

        # define a function to initialize the previous power params
        def init_pre_pow_fun(m,j):
            return m.pre_on_off[j]*m.gen_min_pow[j]
        # m.pre_P_T = Param(m.UNITS,initialize = init_pre_pow_fun, mutable = True)

        # storage size
        storage_capacity_dict = {i: self.capacity[i] * 0.5 for i in m.UNITS}
        m.storage_capacity = Param(m.UNITS,initialize=storage_capacity_dict,mutable = False)

        # initial soc of the storage
        def pre_soc_init(m,j):
            return m.storage_capacity[j]/2
        m.pre_SOC = Param(m.UNITS,initialize = pre_soc_init, mutable = True)

        # pmax of storage
        def storage_pmax_init(m,j):
            return m.storage_capacity[j]/storage_size
        m.pmax_storage = Param(m.UNITS,initialize = storage_pmax_init, mutable = False)

        # storage efficiency
        m.sqrteta = Param(initialize = sqrt(storage_eff),mutable = False)

        ## define the variables

        # define a rule for storage size
        def storage_power_bnd_rule(m,j,h,k):
            return (0,m.pmax_storage[j])

        def storage_capacity_bnd_rule(m,j,h,k):
            return (0,m.storage_capacity[j])

        # generator power (MW)
        m.P_T = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = NonNegativeReals)
        m.P_E = Var(m.UNITS,m.HOUR,m.SCENARIOS,bounds = \
        storage_power_bnd_rule)
        m.P_G = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = NonNegativeReals)

        # storage power (MW) to merge with thermal power
        m.P_S = Var(m.UNITS,m.HOUR,m.SCENARIOS,bounds = storage_power_bnd_rule) #discharge

        # storage power (MW) interacting with the market
        m.P_D = Var(m.UNITS,m.HOUR,m.SCENARIOS,bounds = storage_power_bnd_rule) #discharge
        m.P_C = Var(m.UNITS,m.HOUR,m.SCENARIOS,bounds = storage_power_bnd_rule) #charge

        # storage SOC
        m.S_SOC = Var(m.UNITS,m.HOUR,m.SCENARIOS,bounds = storage_capacity_bnd_rule)

        # total power to the grid()
        m.P_R = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = NonNegativeReals)

        # charge or discharge binary var
        m.y_S = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = Binary)
        m.y_E = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = Binary)
        m.y_D = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = Binary)
        m.y_C = Var(m.UNITS,m.HOUR,m.SCENARIOS,within = Binary)

        # binary variables indicating on/off
        m.on_off = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = True,\
        within = Binary)

        # binary variables indicating  start_up
        m.start_up = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False,\
        within = Binary)

        # binary variables indicating shut down
        m.shut_dw = Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False, \
        within = Binary)

        # power produced in each segment
        m.power_segment = Var(m.UNITS,m.HOUR,m.SCENARIOS,m.SEGMENTS,\
        within = NonNegativeReals)

        ## Constraints

        ## Big M constraints on storage power
        def big_m_on_storage_pow_con_rules(m):
            for j in m.UNITS:
                for h in m.HOUR:
                    for k in m.SCENARIOS:
                        yield m.P_S[j,h,k]<= m.pmax_storage[j]* m.y_S[j,h,k]
                        yield m.P_E[j,h,k]<= m.pmax_storage[j]* m.y_E[j,h,k]
                        yield m.P_D[j,h,k]<= m.pmax_storage[j]* m.y_D[j,h,k]
                        yield m.P_C[j,h,k]<= m.pmax_storage[j]* m.y_C[j,h,k]
        m.big_m_on_storage_pow_con = ConstraintList(rule = big_m_on_storage_pow_con_rules)

        def charge_or_discharge_fun3(m,j,h,k):
            return m.y_E[j,h,k] + m.y_S[j,h,k]<=1
        m.discharge_con3 = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = charge_or_discharge_fun3)

        def charge_or_discharge_fun4(m,j,h,k):
            return m.y_D[j,h,k] + m.y_C[j,h,k]<=1
        m.discharge_con4 = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = charge_or_discharge_fun4)

        def discharge_only_when_unit_on_fun(m,j,h,k):
            return 1-m.y_S[j,h,k] + m.on_off[j,h,k] >= 1
        m.discharge_only_when_unit_on = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = discharge_only_when_unit_on_fun)

        def charge_rate_exp(m,j,h,k):
            return m.P_E[j,h,k] + m.P_C[j,h,k]
        m.charge_rate = Expression(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =charge_rate_exp)

        def discharge_rate_exp(m,j,h,k):
            return m.P_S[j,h,k] + m.P_D[j,h,k]
        m.discharge_rate = Expression(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =discharge_rate_exp)

        # charging rate Constraints
        def charge_rate_fun(m,j,h,k):
            return m.charge_rate[j,h,k] <= m.pmax_storage[j]
        m.charge_rate_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =charge_rate_fun)

        def discharge_rate_fun(m,j,h,k):
            return m.discharge_rate[j,h,k] <= m.pmax_storage[j]
        m.discharge_rate_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =discharge_rate_fun)

        # bounds on gen_pow
        def lhs_bnd_gen_pow_fun(m,j,h,k):
            return m.on_off[j,h,k] * m.gen_min_pow[j] <= m.P_T[j,h,k]
        m.lhs_bnd_gen_pow = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = lhs_bnd_gen_pow_fun)

        def rhs_bnd_gen_pow_fun(m,j,h,k):
            return m.P_T[j,h,k] <= m.on_off[j,h,k] * m.gen_cap[j]
        m.rhs_bnd_gen_pow = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = rhs_bnd_gen_pow_fun)

        # thermal generator power balance
        def therm_pow_balance(m,j,h,k):
            return m.P_T[j,h,k] == m.P_E[j,h,k] + m.P_G[j,h,k]
        m.thermal_pow_balance_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =therm_pow_balance)

        # total power to the grid
        def total_gen_pow_G_fun(m,j,h,k):
            return m.P_R[j,h,k] == m.P_G[j,h,k] + m.P_S[j,h,k]
        m.total_pow_G_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule =total_gen_pow_G_fun)

        # storage energy balance
        def EnergyBalance(m,j,h,k):
            if h == 0 :
                return m.S_SOC[j,h,k] == m.pre_SOC[j] + m.charge_rate[j,h,k]\
                *m.sqrteta-m.discharge_rate[j,h,k]/m.sqrteta
            else :
                return m.S_SOC[j,h,k] == m.S_SOC[j,h-1,k] + m.charge_rate[j,h,k]\
                *m.sqrteta-m.discharge_rate[j,h,k]/m.sqrteta
        m.EnergyBalance_Con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS, \
        rule = EnergyBalance)

        # linearized power
        def linear_power_fun(m,j,h,k):
            return m.P_T[j,h,k] == \
            sum(m.power_segment[j,h,k,l] for l in m.SEGMENTS)
        m.linear_power = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = linear_power_fun)

        # bounds on segment power
        def seg_pow_bnd_fun(m):
            for j in m.UNITS:
                for h in m.HOUR:
                    for k in m.SCENARIOS:
                        for l in m.SEGMENTS:
                            yield m.power_segment[j,h,k,l]<= self.power_increae[j][l]
        m.seg_pow_bnd = ConstraintList(rule = seg_pow_bnd_fun)

        # start up and shut down logic (Arroyo and Conejo 2000)
        def start_up_shut_dw_fun(m,j,h,k):
            if h == 0:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] -\
                m.pre_on_off[j]
            else:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] -\
                m.on_off[j,h-1,k]
        m.start_up_shut_dw = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = \
        start_up_shut_dw_fun)

        # either start up or shut down
        def start_up_or_shut_dw_fun(m,j,h,k):
            return m.start_up[j,h,k] + m.shut_dw[j,h,k] <= 1
        m.start_up_or_shut_dw = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = start_up_or_shut_dw_fun)

        # ramp up limits
        def ramp_up_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.P_T[j,h,k] <= m.pre_pow[j] \
                + m.ramp_up[j]*m.pre_on_off[j]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]\
                + m.gen_cap[j]*(1-m.on_off[j,h,k])
            else:
                return m.P_T[j,h,k] <= m.P_T[j,h-1,k] \
                + m.ramp_up[j]*m.on_off[j,h-1,k]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]\
                + m.gen_cap[j]*(1-m.on_off[j,h,k])
        m.ramp_up_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_up_fun)

        # ramp shut down limits
        def ramp_shut_dw_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.pre_pow[j] <= m.gen_cap[j]*m.on_off[j,h,k] \
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
            else:
                return m.P_T[j,h-1,k] <= m.gen_cap[j]*m.on_off[j,h,k] \
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
        m.ramp_shut_dw_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_shut_dw_fun)

        # ramp down limits
        def ramp_dw_fun(m,j,h,k):
            '''
            j,h,k stand for unit, hour,scenario respectively.
            '''
            if h == 0:
                return m.pre_pow[j] - m.P_T[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]\
                + m.gen_cap[j] * (1 - m.pre_on_off[j])
            else:
                return m.P_T[j,h-1,k] - m.P_T[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]\
                + m.gen_cap[j] * (1 - m.on_off[j,h-1,k])
        m.ramp_dw_con = Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_dw_fun)

        # expressions for min up/down time
        def G_expr(m,j):
            # temp = max(0,value(m.min_up_time[j]-m.pre_up_hour[j]))
            # return min(plan_horizon,temp*value(m.pre_on_off[j]))
            return (m.min_up_time[j]-m.pre_up_hour[j])*m.pre_on_off[j]
        m.G = Expression(m.UNITS,rule = G_expr)

        def L_expr(m,j):
            # temp = max(0,value(m.min_dw_time[j]-m.pre_dw_hour[j]))
            # return min(plan_horizon,temp*(1-value(m.pre_on_off[j])))
            return (m.min_dw_time[j]-m.pre_dw_hour[j])*(1-m.pre_on_off[j])
        m.L = Expression(m.UNITS,rule = L_expr)

        ## min up time constraints
        def minimum_on_fun(m,j,k):
            if value(m.G[j]) <= 0:
                return Constraint.Skip
            elif 0<value(m.G[j])<= plan_horizon:
                return sum(1-m.on_off[j,h,k] for h in range(int(value(m.G[j])))) == 0
            else:
                return sum(1-m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        m.minimum_on_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_on_fun)

        # minimum up time in the subsequent hours
        def sub_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.G[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.G[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_up_time[j]+1):
                        yield sum(m.on_off[j,h,k] for h in range(h0,h0 + m.min_up_time[j]))\
                        >= m.min_up_time[j]*m.start_up[j,h0,k]
        m.sub_minimum_on_con = ConstraintList(rule = sub_minimum_on_fun)

        # minimum up time in the final hours
        def fin_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_up_time[j]):
                        # if goes into here, sub_minimum_on_con must have been
                        # skipped.
                        if value(m.G[j]) <= 0:
                            # if goes into here, minimum_on_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                        else:
                            for h0 in range(int(value(m.G[j])),plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_up_time[j]+1,plan_horizon):
                            yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                            for h in range(h0,plan_horizon)) >=0
        m.fin_minimum_on_con = ConstraintList(rule = fin_minimum_on_fun)

        ## min down time constraints
        def minimum_off_fun(m,j,k):
            if value(m.L[j])<= 0:
                return Constraint.Skip
            elif 0<value(m.L[j])<= plan_horizon:
                return sum(m.on_off[j,h,k] for h in range(int(value(m.L[j])))) == 0
            else:
                return sum(m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        m.minimum_off_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_off_fun)

        # minimum down time in the subsequent hours
        def sub_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.L[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.L[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_dw_time[j]+1):
                        yield sum(1-m.on_off[j,h,k] for h in range(h0,h0 + m.min_dw_time[j]))\
                        >= m.min_dw_time[j]* m.shut_dw[j,h0,k]
        m.sub_minimum_dw_con = ConstraintList(rule = sub_minimum_dw_fun)

        # minimum down time in the final hours
        def fin_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_dw_time[j]):
                        # if goes into here, sub_minimum_dw_con must have been
                        # skipped.
                        if value(m.L[j]) <= 0:
                            # if goes into here, minimum_off_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0
                        else:
                            for h0 in range(int(value(m.L[j])),plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_dw_time[j]+1,plan_horizon):
                            yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                            for h in range(h0,plan_horizon)) >=0
        m.fin_minimum_dw_con = ConstraintList(rule = fin_minimum_dw_fun)

        ## Expression
        # energy generated by all the units
        def total_eng_fun(m,h,k):
            return sum(m.P_R[j,h,k] + m.P_D[j,h,k] - m. P_C[j,h,k] for j in m.UNITS)
        m.tot_eng = Expression(m.HOUR,m.SCENARIOS,rule = total_eng_fun)

        def prod_cost_approx_fun(m,j,h,k):
            return m.prod_cost[j,0]*m.gen_min_pow[j]*m.on_off[j,h,k]\
            + sum(m.prod_cost[j,l]*m.power_segment[j,h,k,l] for l in range(1,len(m.SEGMENTS)))
        m.prod_cost_approx = Expression(m.UNITS,m.HOUR,m.SCENARIOS,\
        rule = prod_cost_approx_fun)

        # total cost
        def tot_cost_fun(m,h,k):
            return sum(m.prod_cost_approx[j,h,k]
            + m.start_up_cost[j]*m.start_up[j,h,k]\
            for j in m.UNITS)
        m.tot_cost = Expression(m.HOUR,m.SCENARIOS,rule = tot_cost_fun)

        ## Objective
        def exp_revenue_fun(m):
            return sum(m.tot_eng[h,k]*m.DAM_price[h,k]- m.tot_cost[h,k]\
            for h in m.HOUR for k in m.SCENARIOS)/n_scenario
        m.exp_revenue = Objective(rule = exp_revenue_fun,sense = maximize)

        return m

    @staticmethod
    def switch_model_mode(m,mode = '3b'):

        # tag the model by the mode
        m.mode = mode

        # first unfix all the vars we care, in case the model is switched before
        m.P_D.unfix()
        m.P_C.unfix()
        m.y_D.unfix()
        m.y_C.unfix()

        m.P_E.unfix()
        m.P_S.unfix()
        m.y_E.unfix()
        m.y_S.unfix()

        # hide the storage: cannot charge/discharge from/to market
        if mode == '1a' or mode == 'static':
            m.P_D.fix(0)
            m.P_C.fix(0)
            m.y_D.fix(0)
            m.y_C.fix(0)

        # hide the storage: it can only charge from the market
        elif mode == '1b':
            m.P_D.fix(0)
            m.y_D.fix(0)

        # storage and generators are independent
        elif mode == '2':
            m.P_E.fix(0)
            m.P_S.fix(0)
            m.y_E.fix(0)
            m.y_S.fix(0)

        # storage cannot charge from the market
        elif mode == '3a':
            m.P_C.fix(0)
            m.y_C.fix(0)

        return m

    # define a function to add non_anticipativity constraints to models
    @staticmethod
    def add_non_anticipativity(m,NA_con_range = 2):

        '''
        This function takes the thermal model and add non non_anticipativity cosntraints
        to it in order to do self-scheduling in DAM.

        '''
        from itertools import combinations

        # generate scenarios combinations
        scenario_comb = combinations(m.SCENARIOS,2)

        def NAC_list_rules(m):
            for k in scenario_comb:
                for j in m.UNITS:
                    for h in range(NA_con_range):
                        yield m.gen_pow[j,h,k[0]] == m.gen_pow[j,h,k[1]]
        m.NAC_SS_cons = ConstraintList(rule = NAC_list_rules)

        return m

    # define a function to add bidding constraints to models
    @staticmethod
    def add_bidding_constraints(m,NA_con_range = 2):

        '''
        This function takes the thermal model and add bidding cosntraints
        to it in order to bid into DAM.

        Note: it is a little different in the paper from what I have done with the
        energy storage system model. This constraint is enforced based on the energy
        output instead of power output.
        '''

        from itertools import combinations

        # generate scenarios combinations
        scenario_comb = combinations(m.SCENARIOS,2)

        # constraints for thermal generators
        def bidding_con_fun1(m):
            for k in scenario_comb:
                for j in m.UNITS:
                    for h in range(NA_con_range):
                        yield (m.gen_pow[j,h,k[0]] - m.gen_pow[j,h,k[1]])\
                        *(m.DAM_price[h,k[0]] - m.DAM_price[h,k[1]]) >= 0
        m.bidding_con1 = ConstraintList(rule = bidding_con_fun1)

        return m

    # define a function to extract power ouput for a hour
    @staticmethod
    def extract_pow_s_s(m, horizon = 24, hybrid = False, segment_power = False, verbose=False):

        if hybrid == False:
            gen_pow = {unit: [value(m.gen_pow[unit,hour,0]) for hour in \
            range(horizon)] for unit in m.UNITS}
        else:
            tot_pow_delivered =  {unit: [value(m.P_R[unit,hour,0]) for hour in \
            range(horizon)] for unit in m.UNITS}
            thermal_power_delivered = {unit: [value(m.P_G[unit,hour,0]) for hour in \
            range(horizon)] for unit in m.UNITS}
            thermal_power_generated = {unit: [value(m.P_T[unit,hour,0]) for hour in \
            range(horizon)] for unit in m.UNITS}

            return tot_pow_delivered, thermal_power_delivered,thermal_power_generated

        if verbose:
            for unit in m.UNITS:
                print('Unit {} tracking power output is {}'.format(unit,\
                gen_pow[unit]))

        if segment_power:

            gen_pow_seg = {(unit,hour,l): value(m.power_segment[unit,hour,0,l]) \
            for hour in range(horizon) for unit in m.UNITS for l in m.SEGMENTS}

            return gen_pow,gen_pow_seg

        return gen_pow

    # define a function to extract energy ouput for a hour
    @staticmethod
    def extract_eng_s_s(m):
        '''
        This function record the energy output of a self-schedule model at
        the first time step.
        Input:
        1. gen_eng: previously recorded generation energy. Type: dict. The
        keys are the unit and the values are lists of energy generation.
        '''
        gen_eng = {unit: [value(m.gen_eng[unit,hour,0]) for hour in \
        range(24)] for unit in m.UNITS}

        return gen_eng

    @staticmethod
    def extract_on_off_s_s(m,total_num = False):
        on_off = {unit: [value(m.on_off[unit,hour,0]) for hour in \
        range(24)] for unit in m.UNITS}

        if total_num:
            total = [on_off[unit].count(1) for unit in m.UNITS]
            return on_off,np.array(total)

        return on_off

    @staticmethod
    def extract_start_up_s_s(m,total_num = False):
        start_up = {unit: [value(m.start_up[unit,hour,0]) for hour in \
        range(24)] for unit in m.UNITS}

        if total_num:
            total = [start_up[unit].count(1) for unit in m.UNITS]
            return start_up,np.array(total)

        return start_up

    @staticmethod
    def extract_shut_dw_s_s(m,total_num = False):
        shut_dw = {unit: [value(m.shut_dw[unit,hour,0]) for hour in \
        range(24)] for unit in m.UNITS}

        if total_num:
            total = [shut_dw[unit].count(1) for unit in m.UNITS]
            return shut_dw,np.array(total)

        return shut_dw

    # define a function to calculate the revenue
    def calc_revenue_s_s(self,gen_eng,gen_pow_seg,on_off,start_up,real_price):
        '''
        Calculate daily (24 hours) DAM revenue.
        '''

        rev = []
        # loop thru hours
        for h in range(24):

            # calculate the income
            income_ = sum(gen_eng[j][h]*real_price[h] for j in range(self.n_units))

            # production cost
            prod_cost = sum(gen_pow_seg[(j,h,l)] * self.prod_cost[j][l]\
            + self.start_cost[j]*start_up[j][h] for h in range(24)\
            for l in range(len(self.power_increae[0])) for j in range(self.n_units))

            rev.append(income_ - prod_cost)

        return rev,sum(rev)

    def calc_revenue_by_unit(self,gen_eng,gen_pow_seg,on_off,eng_schedule,start_up,real_price):
       '''
       Calculate daily (24 hours) DAM revenue.
       '''

       rev_unit = []
       for j in range(self.n_units):

           # calculate the income
           income_ = sum(gen_eng[j][h]*real_price[h] for h in range(24))

           # production cost
           prod_cost = sum(gen_pow_seg[(j,h,l)] * self.prod_cost[j][l]\
           + self.start_cost[j]*start_up[j][h] for h in range(24)\
           for l in range(len(self.power_increae[0])))

           rev_unit.append(income_ - prod_cost)

       return np.array(rev_unit)

    # need a function to extract all the information
    def record_settlement(self,m,real_price,record_stats = False):

        if record_stats:

            gen_eng = self.extract_eng_s_s(m)
            gen_pow,gen_pow_seg = self.extract_pow_s_s(m,segment_power=True)
            on_off,tot_on_hours = self.extract_on_off_s_s(m,total_num = True)
            start_up,tot_start_times = self.extract_start_up_s_s(m,total_num = True)

            rev,tot_rev = self.calc_revenue_s_s(gen_eng,gen_pow_seg,on_off,start_up,real_price)

            return gen_eng,gen_pow,on_off,start_up,rev,tot_rev,tot_on_hours,\
            tot_start_times

        gen_eng = self.extract_eng_s_s(m)
        gen_pow,gen_pow_seg = self.extract_pow_s_s(m,segment_power=True)
        on_off = self.extract_on_off_s_s(m)
        start_up = self.extract_start_up_s_s(m)

        rev,tot_rev = self.calc_revenue_s_s(gen_eng,gen_pow_seg,on_off,start_up,real_price)

        return gen_eng,gen_pow,on_off,start_up,rev,tot_rev

    @staticmethod
    def extract_power(m):

        '''
        This function takes the solved thermal model to extract power output (MW) in
        each hour.

        Input:
        1. a pyomo model

        Output:
        1.  a dictionary whose keys are units and the values are numpy
        arrays (n_scenario*horizon) which store the energy output at each time in each scenario.
        '''

        power_output = {}

        for j in m.UNITS:
            unit_output = np.zeros((len(m.SCENARIOS),len(m.HOUR)))
            for k in m.SCENARIOS:
                for h in m.HOUR:
                    unit_output[k,h]=round(value(m.gen_pow[j,h,k]),1)

            power_output[j] = unit_output

        return power_output

    # define a fucntion to get the bidding power and corresponding prices
    def get_bid_power_price(self,m_bid,price_forecast,plan_horizon = 48,verbose = False):

        '''
        Get the bidding curves of the generator from the result of SP.
        '''

        assert plan_horizon > 24

        power_output = self.extract_power(m_bid)

        # sort the price forecast along scenario axis
        price_forecast_sorted = np.sort(price_forecast,axis = 0)

        power_output_dict = {}
        marginal_cost_dict = {}
        cost_dict = {}

        for unit in m_bid.UNITS:

            # sort the energy output in this unit, along scenario axis
            power_unit_sorted = np.sort(power_output[unit],axis = 0)

            # initialize lists to store the result
            power_unit_list = []
            price_unit_list = []
            cost_unit_list = []

            for h in range(plan_horizon):

                if h < 24:

                    # find power less than pmin and delete
                    del_inx = np.where(power_unit_sorted[:,h]<self.min_pow[unit])[0]
                    power_unit = np.delete(power_unit_sorted[:,h],del_inx)
                    price_forecast_unit = np.delete(price_forecast_sorted[:,h],del_inx)

                    # if the unit bids to be turned off, we bid into the true costs
                    if len(power_unit) == 0:

                        power_unit = self.output[unit][1:]
                        price_forecast_unit = self.prod_cost[unit]

                    # make sure original output points in the list
                    for p_idx, p in enumerate(self.output[unit]):

                        if p_idx == 0:
                            continue

                        if p in power_unit:
                            idx = np.where(power_unit==p)[0][0]
                            price_forecast_unit[idx] = max(price_forecast_unit[idx],self.prod_cost[unit][p_idx-1])
                        else:
                            # add p to corresponding position
                            insert_idx = np.searchsorted(power_unit,p)
                            power_unit = np.insert(power_unit,insert_idx,p)
                            price_forecast_unit = np.insert(price_forecast_unit,insert_idx,\
                            self.prod_cost[unit][p_idx-1])

                    # make sure pmin and pmax are in the curve
                    # if float(self.min_pow[unit]) in power_unit:
                    #     pass
                    # else:
                    #     # add pmin to the power output array
                    #     power_unit = np.insert(power_unit,0,self.min_pow[unit])
                    #     price_forecast_unit = np.insert(price_forecast_unit,0,\
                    #     self.prod_cost[unit][0])
                    #
                    # if float(self.capacity[unit]) in power_unit:
                    #     pass
                    # else:
                    #     # add pmax to power output array
                    #     power_unit = np.append(power_unit,self.capacity[unit])
                    #     price_forecast_unit = np.append(price_forecast_unit,\
                    #     self.prod_cost[unit][-1])

                    # calculate the unique elements in the power output and prices
                    power_unit,unique_power_idx = np.unique(power_unit,\
                    return_index=True)

                    price_forecast_unit = price_forecast_unit[unique_power_idx]
                    cost_unit = np.cumsum(np.diff(np.insert(power_unit,0,0))\
                    * price_forecast_unit)

                    # record the offering power and prices
                    power_unit_list.append(power_unit)
                    price_unit_list.append(price_forecast_unit)
                    cost_unit_list.append(cost_unit)

                    if verbose:
                        print("Unit {} Power {}\nMarginal Costs {}".format(unit,\
                        power_unit,price_forecast_unit))

                else:

                    power_unit = self.output[unit][1:]
                    price_forecast_unit = self.prod_cost[unit]
                    cost_unit = np.cumsum(np.diff(np.insert(power_unit,0,0))\
                    * price_forecast_unit)

                    # record the offering power and prices
                    power_unit_list.append(power_unit)
                    price_unit_list.append(price_forecast_unit)
                    cost_unit_list.append(cost_unit)

            power_output_dict[unit] = power_unit_list
            marginal_cost_dict[unit] = price_unit_list
            cost_dict[unit] = cost_unit_list

        return power_output_dict, marginal_cost_dict, cost_dict

    def create_bidding_model(self,n_scenario=10,generator=None,plan_horizon=48,NA_horizon=24):

        print("")
        print("In create_bidding_model\n")

        # build bidding model
        m = self.DAM_thermal(plan_horizon = plan_horizon,generator= generator)
        m = self.add_non_anticipativity(m,NA_con_range=NA_horizon)

        print("")
        print("Bidding model is built.")

        return m

    def stochastic_bidding(self,m,price_forecast_dir,cost_curve_store_dir,date):

        print("")
        print("In stochastic_bidding\n")

        # read forecasts
        forecasts = pd.read_csv(price_forecast_dir,header = None).values

        # pass the forecast into pyomo model
        for i in m.SCENARIOS:
            for t in m.HOUR:
                m.DAM_price[t,i] = forecasts[i,t]

        # solve the model
        solver = SolverFactory('gurobi')
        result = solver.solve(m,tee=True)

        # get the bidding curves
        power_output_dict, marginal_cost_dict, cost_dict = \
        self.get_bid_power_price(m,forecasts,verbose = True)

        # write the price curves to a csv file
        for j in m.UNITS:
            for h in range(24):
                curve = np.concatenate((power_output_dict[j][h].reshape(-1,1),\
                cost_dict[j][h].reshape(-1,1)),axis = 1)
                np.savetxt(cost_curve_store_dir+j+\
                '_date={}_hour={}_cost_curve.csv'.format(date,h),curve,delimiter=',')

        return

    # define a function to track the commitment to the market
    def build_tracking_model(self,track_horizon,generator,track_type = 'RUC',hybrid = False):

        print("")
        print("In build_tracking_model\n")

        if hybrid == False:
            # build another pyomo model with 1 scenario
            m_control = self.DAM_thermal(plan_horizon = track_horizon,generator=generator)

        else:
            # build a hybrid model
            m_control = self.DAM_hybrid(plan_horizon = track_horizon,generator=generator)

            # switch the model to 1a: because we want to hide the storage now
            m_control = self.switch_model_mode(m_control,'1a')

        # remove the current objective function
        m_control.exp_revenue.deactivate()

        # add power schedule as a Param
        m_control.power_commit = Param(m_control.UNITS,m_control.HOUR, \
        initialize = 0, mutable = True)

        # add params for objective weights
        m_control.deviation_weight = Param(initialize = 30, mutable = True)
        m_control.cost_weight = Param(initialize = 1, mutable = True)
        m_control.ramping_weight = Param(initialize = 20, mutable = True)

        ## USE L1 norm

        # add a slack variable
        m_control.slack_var_T = Var(m_control.UNITS,m_control.HOUR,\
        m_control.SCENARIOS,within = NonNegativeReals)

        # add a slack variable for ramping
        m_control.slack_var_Ramp = Var(m_control.UNITS,m_control.HOUR,\
        m_control.SCENARIOS,within = NonNegativeReals)

        if hybrid == False:

            # add constraints
            def abs_con_fun1(m,j,h,k):
                return -m.slack_var_T[j,h,k]<=m.gen_pow[j,h,0] - m.power_commit[j,h]
            m_control.abs_con1 = Constraint(m_control.UNITS,m_control.HOUR,\
            m_control.SCENARIOS,rule = abs_con_fun1)

            def abs_con_fun2(m,j,h,k):
                return m.slack_var_T[j,h,k]>=m.gen_pow[j,h,0] - m.power_commit[j,h]
            m_control.abs_con2 = Constraint(m_control.UNITS,\
            m_control.HOUR,m_control.SCENARIOS,rule = abs_con_fun2)

            # add constraints for ramping
            def abs_ramp_con_fun1(m,j,h,k):
                if h == 0:
                    return -m.slack_var_Ramp[j,h,k]<=m.gen_pow[j,h,0] - m.pre_pow[j]
                else:
                    return -m.slack_var_Ramp[j,h,k]<=m.gen_pow[j,h,0] - m.gen_pow[j,h-1,0]
            m_control.abs_ramp_con1 = Constraint(m_control.UNITS,m_control.HOUR,\
            m_control.SCENARIOS,rule = abs_ramp_con_fun1)

            def abs_ramp_con_fun2(m,j,h,k):
                if h == 0:
                    return m.slack_var_Ramp[j,h,k]>=m.gen_pow[j,h,0] - m.pre_pow[j]
                else:
                    return m.slack_var_Ramp[j,h,k]>=m.gen_pow[j,h,0] - m.gen_pow[j,h-1,0]
            m_control.abs_ramp_con2 = Constraint(m_control.UNITS,\
            m_control.HOUR,m_control.SCENARIOS,rule = abs_ramp_con_fun2)

            # burn in penalty type
            m_control.track_type =  track_type

            if track_type == 'RUC':

                def power_abs_least(m):
                    return sum(m.slack_var_T[j,h,k] for j in m.UNITS \
                    for h in m.HOUR for k in m.SCENARIOS)
                m_control.abs_dev = Objective(rule = power_abs_least,sense = minimize)

            elif track_type == 'SCED':

                # m_control.RT_price = Param(m_control.HOUR,initialize = 20, mutable = True)
                # m_control.small_penalty = Param(m_control.HOUR,initialize = 10, mutable = True)

                def rt_penalty_fun(m):
                    return sum(m.small_penalty[h]*m.slack_var_T[j,h,0]\
                    for j in m.UNITS for h in range(1,len(m.HOUR)))\
                    - (m.gen_pow[j,0,0] - m.power_commit[j,0])* m.RT_price[0]
                # m_control.rt_penalty = Objective(rule = rt_penalty_fun,sense = minimize)

                def power_abs_least(m):
                    return m.deviation_weight*sum(m.slack_var_T[j,h,k] \
                    for h in m.HOUR for k in m.SCENARIOS for j in m.UNITS) \
                    + m.ramping_weight * sum(m.slack_var_Ramp[j,h,k] \
                    for h in m.HOUR for k in m.SCENARIOS for j in m.UNITS)
                    + m.cost_weight * sum(m.tot_cost[h,k] for h in m.HOUR for k in m.SCENARIOS)
                m_control.abs_dev = Objective(rule = power_abs_least,sense = minimize)

        # hybrid case
        else:

            # add constraints
            def abs_con_fun1(m,j,h,k):
                return -m.slack_var_T[j,h,k]<=m.P_R[j,h,0] - m.power_commit[j,h]
            m_control.abs_con1 = Constraint(m_control.UNITS,m_control.HOUR,\
            m_control.SCENARIOS,rule = abs_con_fun1)

            def abs_con_fun2(m,j,h,k):
                return m.slack_var_T[j,h,k]>=m.P_R[j,h,0] - m.power_commit[j,h]
            m_control.abs_con2 = Constraint(m_control.UNITS,\
            m_control.HOUR,m_control.SCENARIOS,rule = abs_con_fun2)

            # add constraints for ramping
            def abs_ramp_con_fun1(m,j,h,k):
                if h == 0:
                    return -m.slack_var_Ramp[j,h,k]<=m.P_T[j,h,0] - m.pre_pow[j]
                else:
                    return -m.slack_var_Ramp[j,h,k]<=m.P_T[j,h,0] - m.P_T[j,h-1,0]
            m_control.abs_ramp_con1 = Constraint(m_control.UNITS,m_control.HOUR,\
            m_control.SCENARIOS,rule = abs_ramp_con_fun1)

            def abs_ramp_con_fun2(m,j,h,k):
                if h == 0:
                    return m.slack_var_Ramp[j,h,k]>=m.P_T[j,h,0] - m.pre_pow[j]
                else:
                    return m.slack_var_Ramp[j,h,k]>=m.P_T[j,h,0] - m.P_T[j,h-1,0]
            m_control.abs_ramp_con2 = Constraint(m_control.UNITS,\
            m_control.HOUR,m_control.SCENARIOS,rule = abs_ramp_con_fun2)

            # burn in penalty type
            m_control.track_type =  track_type

            if track_type == 'RUC':

                def power_abs_least(m):
                    return m.deviation_weight*sum(m.slack_var_T[j,h,k] \
                    for h in m.HOUR for k in m.SCENARIOS for j in m.UNITS) \
                    + sum(m.tot_cost[h,k] for h in m.HOUR for k in m.SCENARIOS)
                m_control.abs_dev = Objective(rule = power_abs_least,sense = minimize)

            elif track_type == 'SCED':

                # m_control.RT_price = Param(m_control.HOUR,initialize = 20, mutable = True)
                # m_control.small_penalty = Param(m_control.HOUR,initialize = 10, mutable = True)

                def rt_penalty_fun(m):
                    return sum(m.small_penalty[h]*m.slack_var_T[j,h,0]\
                    for j in m.UNITS for h in range(1,len(m.HOUR)))\
                    - (m.gen_pow[j,0,0] - m.power_commit[j,0])* m.RT_price[0]
                # m_control.rt_penalty = Objective(rule = rt_penalty_fun,sense = minimize)

                def power_abs_least(m):
                    return m.deviation_weight*sum(m.slack_var_T[j,h,k] \
                    for h in m.HOUR for k in m.SCENARIOS for j in m.UNITS) \
                    + m.ramping_weight * sum(m.slack_var_Ramp[j,h,k] \
                    for h in m.HOUR for k in m.SCENARIOS for j in m.UNITS)
                    + m.cost_weight * sum(m.tot_cost[h,k] for h in m.HOUR for k in m.SCENARIOS)
                m_control.abs_dev = Objective(rule = power_abs_least,sense = minimize)

        print("")
        print("Tracking model is built.")

        return m_control

    def pass_schedule_to_track_and_solve(self,m_control,RUC_dispatch,deviation_weight=30, \
    ramping_weight=20,cost_weight = 1,SCED_dispatch=None,RT_price=None):

        # pass the RUC dispatch level
        for j in m_control.UNITS:
            for h in m_control.HOUR:
                m_control.power_commit[j,h] = RUC_dispatch[j][h]

        # pass sced data
        if m_control.track_type == 'SCED':

            # real-time price
       # for h in m_control.HOUR:
            #     m_control.RT_price[h] = RT_price[h]

            # dispatch
            for j in m_control.UNITS:
                m_control.power_commit[j,0] = SCED_dispatch[j][0]

        m_control.deviation_weight = deviation_weight
        m_control.cost_weight = cost_weight
        m_control.ramping_weight = ramping_weight

        # solve the model
        solver = SolverFactory('gurobi')
        result = solver.solve(m_control,tee=True)

        return

    @staticmethod
    def reset_constraints(m,plan_horizon):

        '''
        Input:
        1. m: The model we want to reset constraints for.
        2. plan_horizon: unfortunately we have reiterate the planning horizon length for now
        '''

        ## min up time constraints
        def minimum_on_fun(m,j,k):
            if value(m.G[j]) <= 0:
                return Constraint.Skip
            elif 0<value(m.G[j])<= plan_horizon:
                return sum(1-m.on_off[j,h,k] for h in range(int(value(m.G[j])))) == 0
            else:
                return sum(1-m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        #m.minimum_on_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_on_fun)

        # minimum up time in the subsequent hours
        def sub_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.G[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.G[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_up_time[j]+1):
                        yield sum(m.on_off[j,h,k] for h in range(h0,h0 + m.min_up_time[j]))\
                        >= m.min_up_time[j]*m.start_up[j,h0,k]
        #m.sub_minimum_on_con = ConstraintList(rule = sub_minimum_on_fun)

        # minimum up time in the final hours
        def fin_minimum_on_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_up_time[j]):
                        # if goes into here, sub_minimum_on_con must have been
                        # skipped.
                        if value(m.G[j]) <= 0:
                            # if goes into here, minimum_on_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                        else:
                            for h0 in range(int(value(m.G[j])),plan_horizon):
                                yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_up_time[j]+1,plan_horizon):
                            yield sum(m.on_off[j,h,k] - m.start_up[j,h0,k] \
                            for h in range(h0,plan_horizon)) >=0
        #m.fin_minimum_on_con = ConstraintList(rule = fin_minimum_on_fun)

        ## min down time constraints
        def minimum_off_fun(m,j,k):
            if value(m.L[j])<= 0:
                return Constraint.Skip
            elif 0<value(m.L[j])<= plan_horizon:
                return sum(m.on_off[j,h,k] for h in range(int(value(m.L[j])))) == 0
            else:
                return sum(m.on_off[j,h,k] for h in range(plan_horizon)) == 0
        #m.minimum_off_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_off_fun)

        # minimum down time in the subsequent hours
        def sub_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:
                    if value(m.L[j])<=0:
                        low_bnd = 0
                    else:
                        low_bnd = int(value(m.L[j]))
                    for h0 in range(low_bnd,plan_horizon-m.min_dw_time[j]+1):
                        yield sum(1-m.on_off[j,h,k] for h in range(h0,h0 + m.min_dw_time[j]))\
                        >= m.min_dw_time[j]* m.shut_dw[j,h0,k]
        #m.sub_minimum_dw_con = ConstraintList(rule = sub_minimum_dw_fun)

        # minimum down time in the final hours
        def fin_minimum_dw_fun(m):
            for j in m.UNITS:
                for k in m.SCENARIOS:

                    if plan_horizon < value(m.min_dw_time[j]):
                        # if goes into here, sub_minimum_dw_con must have been
                        # skipped.
                        if value(m.L[j]) <= 0:
                            # if goes into here, minimum_off_con must have been
                            # skipped
                            for h0 in range(0,plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0
                        else:
                            for h0 in range(int(value(m.L[j])),plan_horizon):
                                yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                                for h in range(h0,plan_horizon)) >=0

                    else:
                        for h0 in range(plan_horizon-m.min_dw_time[j]+1,plan_horizon):
                            yield sum(1-m.on_off[j,h,k] - m.shut_dw[j,h0,k]\
                            for h in range(h0,plan_horizon)) >=0
        #m.fin_minimum_dw_con = ConstraintList(rule = fin_minimum_dw_fun)

        try:
            m.del_component(m.minimum_on_con)
            m.del_component(m.sub_minimum_on_con)
            m.del_component(m.fin_minimum_on_con)
            m.del_component(m.minimum_off_con)
            m.del_component(m.sub_minimum_dw_con)
            m.del_component(m.fin_minimum_dw_con)
            m.del_component(m.minimum_on_con_index)
            m.del_component(m.sub_minimum_on_con_index)
            m.del_component(m.fin_minimum_on_con_index)
            m.del_component(m.minimum_off_con_index)
            m.del_component(m.sub_minimum_dw_con_index)
            m.del_component(m.fin_minimum_dw_con_index)
            print("")
            print('Old constraints have been removed.')
            #return m
        except:
            print('The constraints may not have been constructed. Now I am constructing them.')
            m.minimum_on_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_on_fun)
            m.sub_minimum_on_con = ConstraintList(rule = sub_minimum_on_fun)
            m.fin_minimum_on_con = ConstraintList(rule = fin_minimum_on_fun)
            m.minimum_off_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_off_fun)
            m.sub_minimum_dw_con = ConstraintList(rule = sub_minimum_dw_fun)
            m.fin_minimum_dw_con = ConstraintList(rule = fin_minimum_dw_fun)
        else:
            m.minimum_on_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_on_fun)
            m.sub_minimum_on_con = ConstraintList(rule = sub_minimum_on_fun)
            m.fin_minimum_on_con = ConstraintList(rule = fin_minimum_on_fun)
            m.minimum_off_con = Constraint(m.UNITS,m.SCENARIOS,rule = minimum_off_fun)
            m.sub_minimum_dw_con = ConstraintList(rule = sub_minimum_dw_fun)
            m.fin_minimum_dw_con = ConstraintList(rule = fin_minimum_dw_fun)
            print("")
            print('Constraints have been reset successfully!')

        return

    @staticmethod
    def update_model_params(m,control_pow,hybrid = False):

        '''
        This method takes a thermal generator model and update its initial power
        profile according to the control strategies taken in the previous day.

        Input:
        1. m: the thermal generator model
        2. control_pow: the control strategies taken in the previous day. It is a
        dictionary which stores power output for all the units

        Output:
        1. m: the updated model
        '''

        def group_consecutive(a):
            return np.split(a, np.where(np.diff(a) != 1)[0] + 1)

        for unit in m.UNITS:

            ## update the model
            # if it was on in the previous day's last hour
            if control_pow[unit][-1] > 0:
                # find the power where they are 0
                pow_idx = np.where(np.array(control_pow[unit]) > 0)[0]

                # find the last chunk of indices where consecutively
                consecutive_idx = group_consecutive(pow_idx)[-1]

                m.pre_on_off[unit] = 1
                m.pre_up_hour[unit] = len(consecutive_idx)
                m.pre_dw_hour[unit] = 0
                m.pre_pow[unit] = control_pow[unit][-1]

            else:
                # find the power where they are 0
                pow_idx = np.where(np.array(control_pow[unit]) == 0)[0]

                # find the last chunk of indices where consecutively
                consecutive_idx = group_consecutive(pow_idx)[-1]

                m.pre_on_off[unit] = 0
                m.pre_up_hour[unit] = 0
                m.pre_dw_hour[unit] = len(consecutive_idx)
                m.pre_pow[unit] = 0

            if hybrid:
                # this only works for tracking sced signal
                m.pre_SOC[unit] = value(m.S_SOC[unit,0,0])

        return
