#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os.path
import numpy as np

import matplotlib as mpl
# no support (nor present need for) interactive graphics - we just write PNG files.
mpl.use('AGG') 

import matplotlib.pyplot as plt

import csv
import scipy.stats

from six import iterkeys

hours_in_day = np.arange(24) + 1

# default line width
width = 1

def generic_generator_function(dispatch_levels_by_gentype, gentype, base_dispatch_level, ax):

        solidcolorsdict = {}
        solidcolorsdict['N'] = '#A4302A'
        solidcolorsdict['E'] = '#CDE7B0'
        solidcolorsdict['B'] = '#A3BFA8'
        solidcolorsdict['C'] = '#333333'
        solidcolorsdict['H'] = '#B5D7E4'
        solidcolorsdict['G'] = '#748947'
        solidcolorsdict['W'] = '#5F93C8'
        solidcolorsdict['S'] = '#F6C24C'
        solidcolorsdict['O'] = '#F5D547'
        
        dispatch_level = dispatch_levels_by_gentype[gentype]
        if gentype == 'N':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Nuclear')
        elif gentype == 'C':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Coal')
        elif gentype == 'H':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Hydro')
        elif gentype == 'E':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Geothermal')
        elif gentype == 'B':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Biomass')
        elif gentype == 'G':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom= base_dispatch_level,label='Gas')
        elif gentype == 'O':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Oil')
        elif gentype == 'W':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label='Wind')
        elif gentype == 'S':
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25, 
                    bottom=base_dispatch_level, label='Solar')
        else:
            ax.bar(hours_in_day, dispatch_level, width,
                    color=solidcolorsdict[gentype], linewidth=0.25,
                    bottom=base_dispatch_level, label="ain't nobody got a label fo dat")

        base_dispatch_level += dispatch_level

        return base_dispatch_level


# NOTE: annotations are assumed to be a list of tuples (x,y), where x is the hour of the event, and y is the text label.
def generate_stack_graph(thermal_fleet_capacity,
                         gennamedict, 
                         dispatchlevelsdict, 
                         reserve_requirements, 
                         the_date, 
                         renewables_curtailments, 
                         load_shedding,
                         reserve_shortfalls, 
                         available_reserves,
                         available_quickstart,
                         over_generation_by_hour,
                         max_hourly_demand,
                         quick_start_additional_power_generated_by_hour,
                         y_axis_factor=1.4,
                         annotations=[], 
                         plot_individual_generators=False, 
                         show_plot_legend=True, 
                         output_directory=".", 
                         renewables_penetration_rate=None, 
                         fixed_costs=None, 
                         variable_costs=None, 
                         demand=None):
    

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig = plt.gcf()

    # maps (gen-type,gen-id) keys to the dispatch levels for the
    # corresponding generator. gen-id runs from 0 through N-1, 
    # where N is the total number of generators of a given type.
    dispatch_levels_by_gentype = {}

    # maps generator types to the number of generators of that type.
    gentype_counts = {}

    gentype_counts['N'] = 0
    gentype_counts['C'] = 0
    gentype_counts['G'] = 0
    gentype_counts['H'] = 0
    gentype_counts['W'] = 0
    gentype_counts['S'] = 0
    gentype_counts['B'] = 0
    gentype_counts['O'] = 0
    gentype_counts['E'] = 0 # Geothermal

    ## not going to worry about the other case
    assert plot_individual_generators is False

    # populate the dispatch_levels_by_gentype dictionary.
    for genname in dispatchlevelsdict.keys():
        gentype = gennamedict[genname]
        if gentype in dispatch_levels_by_gentype:
            dispatch_levels_by_gentype[gentype] += dispatchlevelsdict[genname]
        else:
            dispatch_levels_by_gentype[gentype] = dispatchlevelsdict[genname]

    base_dispatch_level = np.array([0.0 for r in range(24)])

    for gentype in ['N', 'E', 'B', 'C', 'H', 'G', 'O', 'W', 'S']:
        if gentype in dispatch_levels_by_gentype:
            base_dispatch_level = generic_generator_function(dispatch_levels_by_gentype, gentype,  base_dispatch_level, ax1)

    if sum(quick_start_additional_power_generated_by_hour) > 0.0:
        ax1.bar(hours_in_day,
                quick_start_additional_power_generated_by_hour,
                width,
                color="#0061ff",
                hatch='xxx',
                bottom=base_dispatch_level,
                label="Quick-Start Generator Output")
        mpl.rcParams['hatch.color'] = 'red'
        mpl.rcParams['hatch.linewidth'] = 1.0
        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += quick_start_additional_power_generated_by_hour[i]

    # plot any load shedding above the dispatch level.
    if sum(load_shedding) > 0.0:
        ax1.bar(hours_in_day, load_shedding, width,
                color="#FFFF00",
                bottom=base_dispatch_level,
                label="Load Shedding")

        for i in range(0, len(load_shedding)):
            base_dispatch_level[i] += load_shedding[i]

    # compute the maximum dispatch level for plotting purposes - we want labels above this.
    # PRESENTLY NOT USED.
    max_dispatch_level = max(base_dispatch_level[i] for i in range(0,24))

    # we want to highlight the load - with a bright and obvious color. 
    # TBD - this actually isn't the load yet, but it's a start...
    # TBD - we need a dynamic line width, based on the maximum dispatch level
    if demand == None:
        dispatch_list = base_dispatch_level.tolist()
        dispatch_list.insert(0,dispatch_list[0])
        ax1.step(list(range(0,25)), dispatch_list, linewidth=3, color='#000000', where='mid')
    else:
        demand.insert(0,demand[0])
        ax1.step(list(range(0, 25)), demand, linewidth=3, color='#000000', where='mid')

    # plot wind curtailments above the base dispatch level, if they exist.
    if sum(renewables_curtailments) > 0.0:
        ax1.bar(hours_in_day,
                renewables_curtailments,
                width,
                color="red",
                bottom=base_dispatch_level,
                label="Renewables Curtailed")
        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += renewables_curtailments[i]

    # plot any reseve shortfall immediately above the demand level.
    if sum(reserve_shortfalls) > 0.0:
        ax1.bar(hours_in_day, reserve_shortfalls, width,
                color="#ff00ff",
                bottom=base_dispatch_level,
                label="Reserve Shortfall")

        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += reserve_shortfalls[i]

    # add in the reserve requirements.
    if sum(reserve_requirements) > 0.0:
        reserve_margin_minus_shortfall = [reserve_requirements[i] - reserve_shortfalls[i] for i in range(0,len(reserve_requirements))]
        ax1.bar(hours_in_day,
                reserve_margin_minus_shortfall, 
                width, 
                color="#00c2ff",
                bottom=base_dispatch_level,
                label="Required Reserve")

        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += reserve_margin_minus_shortfall[i]

    # add in the implicit reserves.
    if sum(available_reserves) > 0.0:
        implicit_reserves = [max(0.0, available_reserves[i] - reserve_requirements[i]) for i in range(0,len(available_reserves))] 
        ax1.bar(hours_in_day,
                implicit_reserves, 
                width, 
                color="#00ffc7",
                bottom=base_dispatch_level,
                label="Implicit Reserve")

        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += implicit_reserves[i]

    # and finally, add in the available quick-start capacity.
    if sum(available_quickstart) > 0.0:
        ax1.bar(hours_in_day,
                available_quickstart, 
                width, 
                color="#494949",
                bottom=base_dispatch_level,
                label="Available Quick Start")
        
        for i in range(0, len(base_dispatch_level)):
            base_dispatch_level[i] += available_quickstart[i]

    ax1.set_title("Power Generation for "+str(the_date), fontsize=25)
    ax1.axis([1,24,0,thermal_fleet_capacity*1.2]) # the 1.2 factor is to allow for headroom, for annotations.
    ax1.set_ylabel("Power (MW)", fontsize=20)
    ax1.set_xlabel("Hour (h)", fontsize=20)
    ax1.set_xticks(hours_in_day + width/2.,("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"))
    
    ax1.set_ylim([0, max_hourly_demand*y_axis_factor])
    
    if show_plot_legend:
        ax1.legend(ncol = 3, fontsize=13, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=1.7, prop={'size':16})
        ax2.legend(loc = 1, ncol = 4, fontsize=13)
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.tick_params(axis='both', which='minor', labelsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=17)
    ax2.tick_params(axis='both', which='minor', labelsize=17)
    ax2.set_title("Excess Generation for "+str(the_date), fontsize=25)
    ax2.set_ylabel("Power (MW)", fontsize=20)
    ax2.set_xlabel("Hour (h)", fontsize=20)
    ax2.set_ylim([0, max_hourly_demand*y_axis_factor])


    # add the renewables penetration rate, fixed cost, and variable cost labels, if specified.
    if renewables_penetration_rate != None:
        penetration_rate_as_string = ("Renewables penetration rate: %5.2f" % renewables_penetration_rate)
        penetration_rate_as_string += "%"
        ax1.annotate(penetration_rate_as_string,
                     xy=(10,10), 
                     xycoords="figure pixels",
                     horizontalalignment="left",
                     verticalalignment='bottom',
                     fontsize=16)

    if fixed_costs != None:
        fixed_costs_as_string = ("Fixed costs:    %10.2f" % fixed_costs)
        ax1.annotate(fixed_costs_as_string,
                     xy=(10,40), 
                     xycoords="figure pixels",
                     horizontalalignment="left",
                     verticalalignment='bottom',
                     fontsize=16)

    if variable_costs != None:
        variable_costs_as_string = ("Variable costs: %10.2f" % variable_costs)
        ax1.annotate(variable_costs_as_string,
                     xy=(10,70), 
                     xycoords="figure pixels",
                     horizontalalignment="left",
                     verticalalignment='bottom',
                     fontsize=16)



    if sum(over_generation_by_hour) > 0.0:
        excess_generation_by_hour = [renewables_curtailments[i] + over_generation_by_hour[i] for i in range (0,24)]
        ax2.bar(hours_in_day,
                excess_generation_by_hour,
                width=1,
                color="0.5",
                label="Excess Generation")
            
    # NOTE: we should eventually option-drive the figure size

    fig.set_size_inches(11.0,14.0)

    plt.savefig(os.path.join(output_directory, "stackgraph_"+str(the_date)+".png"))

    plt.close()

#Function for plotting a comparison graph
def generate_comparison_dict(dispatchlevelsdict, dispatchlevelsdict_day2):
    
    if dispatchlevelsdict.values == dispatchlevelsdict_day2.values:
        pass
    else:
        comp_dict = {}
        for genname in list(dispatchlevelsdict.keys()) and list(dispatchlevelsdict_day2.keys()):
            comp_dict[genname] = np.subtract(dispatchlevelsdict[genname], dispatchlevelsdict_day2[genname])
        print(comp_dict)
    
    return comp_dict

#Generate cost summary graph
def generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs, 
                                daily_load_shedding, daily_over_generation,
                                daily_reserve_shortfall, 
                                daily_curtailments,
                                output_directory="."):

    max_cost = max(daily_fixed_costs) + max(daily_generation_costs)

    plt.axis([1,len(daily_fixed_costs),0,max_cost*1.8])

    plt.bar(list(range(1,len(daily_fixed_costs)+1)), daily_fixed_costs, 1, color = 'k', label="Fixed Costs")
    plt.bar(list(range(1,len(daily_fixed_costs)+1)), daily_generation_costs, 1, bottom=daily_fixed_costs, color = 'r', label="Variable Costs")

    plt.title("Daily Production Costs")
    plt.ylabel("Cost")
    plt.xlabel("Day")
    plt.xticks([i+0.5 for i in range(1,len(daily_fixed_costs)+1)], [str(i) for i in range(1,len(daily_fixed_costs)+1)],rotation=90)

    plt.gca().tick_params(axis='x',pad=15)
    plt.gca().tick_params(axis='y',pad=15)
    plt.gca().tick_params(width=1)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, ncol=2, bbox_to_anchor=(1.14, 1.01))

    plt.legend(loc = 1, ncol = 1)

    # with the base costs in place, annotate the days with load shedding and over-generation.
    label_height_factor = 1.05
    for i in range(1,len(daily_fixed_costs)+1):
        if daily_load_shedding[i-1] > 0.0:
            event_day = i
            plt.annotate("Load Shedding", xy=(event_day+0.5, daily_fixed_costs[i-1] + daily_generation_costs[i-1]), arrowprops=dict(arrowstyle='->'), xytext=(event_day, max_cost * label_height_factor))
            label_height_factor += 0.05
        if daily_over_generation[i-1] > 0.0:
            event_day = i
            plt.annotate("Excess Generation", xy=(event_day+0.5, daily_fixed_costs[i-1] + daily_generation_costs[i-1]), arrowprops=dict(arrowstyle='->'), xytext=(event_day, max_cost * label_height_factor))
            label_height_factor += 0.05
        if daily_reserve_shortfall[i-1] > 0.0:
            event_day = i
            plt.annotate("Reserve Shortfall", xy=(event_day+0.5, daily_fixed_costs[i-1] + daily_generation_costs[i-1]), arrowprops=dict(arrowstyle='->'), xytext=(event_day, max_cost * label_height_factor))
            label_height_factor += 0.05
        if daily_curtailments[i-1] > 0.0:
            event_day = i
            plt.annotate("Renewables Curtailment", xy=(event_day+0.5, daily_fixed_costs[i-1] + daily_generation_costs[i-1]), arrowprops=dict(arrowstyle='->'), xytext=(event_day, max_cost * label_height_factor))
            label_height_factor += 0.05

    plt.savefig(os.path.join(output_directory, "daily_costs.png"))

    plt.close()
