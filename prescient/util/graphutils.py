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

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(0,max_cost*1.8)

    ax.bar(list(range(len(daily_fixed_costs))), daily_fixed_costs, 1, color = 'k', label="Fixed Costs")
    ax.bar(list(range(len(daily_fixed_costs))), daily_generation_costs, 1, bottom=daily_fixed_costs, color = 'r', label="Variable Costs")

    ax.set_title("Daily Production Costs")
    ax.set_ylabel("Cost")
    ax.set_xlabel("Day")
    ax.set_xticks(list(range(len(daily_fixed_costs))))
    ax.set_xticklabels(list(range(1,len(daily_fixed_costs)+1))) # start days at 1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # with the base costs in place, annotate the days with load shedding and over-generation.
    label_height_factor = 1.05
    have_arrow = False
    for i in range(len(daily_fixed_costs)):
        x_pnt = i
        y_pnt = daily_fixed_costs[i] + daily_generation_costs[i]
        xy = (x_pnt, y_pnt)
        if have_arrow:
            label_height_factor += 0.05 # add space between days, if we added the last day
        if label_height_factor > 1.8: # reset the height so we don't go off the screen
            label_height_factor = 1.05
        have_arrow = False
        arrow = dict(arrowstyle='simple')
        if daily_load_shedding[i] > 0.0:
            ax.annotate("Load Shedding", xy=xy, arrowprops=(None if have_arrow else arrow),\
                        xytext=(x_pnt-0.5, max_cost * label_height_factor))
            have_arrow = True
            label_height_factor += 0.05
        if daily_over_generation[i] > 0.0:
            ax.annotate("Excess Generation", xy=xy, arrowprops=(None if have_arrow else arrow),\
                        xytext=(x_pnt-0.5, max_cost * label_height_factor))
            have_arrow = True
            label_height_factor += 0.05
        if daily_reserve_shortfall[i] > 0.0:
            ax.annotate("Reserve Shortfall", xy=xy, arrowprops=(None if have_arrow else arrow),\
                        xytext=(x_pnt-0.5, max_cost * label_height_factor))
            have_arrow = True
            label_height_factor += 0.05
        if daily_curtailments[i] > 0.0:
            ax.annotate("Renewables Curtailment", xy=xy, arrowprops=(None if have_arrow else arrow),\
                        xytext=(x_pnt-0.5, max_cost * label_height_factor))
            have_arrow = True
            label_height_factor += 0.05

    plt.savefig(os.path.join(output_directory, "daily_costs.png"))

    plt.close()
