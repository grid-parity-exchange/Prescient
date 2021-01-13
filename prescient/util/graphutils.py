#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os.path

import matplotlib as mpl
# no support (nor present need for) interactive graphics - we just write PNG files.
mpl.use('AGG') 

import matplotlib.pyplot as plt

# If the matplotlib fails above, it will fail when this is
# imported. Prescient's is slightly more picky, so we'll keep it
from egret.viz.generate_graphs import generate_stack_graph

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
