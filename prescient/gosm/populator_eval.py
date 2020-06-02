#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from datetime import datetime
import datetime
from rpy2.robjects import r, pandas2ri
import rpy2.interactive.packages
from rpy2.robjects.packages import STAP
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from multiprocessing import Pool
import rpy2.robjects as robjects
from datetime import timedelta, date
import matplotlib.dates as mdates


 
STAP = SignatureTranslatedAnonymousPackage
pandas2ri.activate()


os.system("R CMD build pRescient")
r.library('pRescient')


if not os.path.exists("horse_populator_scores"):
	os.mkdir("horse_populator_scores")
#if not os.path.exists("Scenario_plot_set"):
#	os.mkdir("horse_populator_scores/Scenario_plot_set")

wsArray = ('0.1')
cutpointArraynames = ('SC1_cutpoints', 'BigExtreme')
cutpointArraynames2 = ('SC1_cutpoints')
cutpointArraynames3 = ('BigExtreme')

def date_range(start_date, end):
	r = (end+datetime.timedelta(days=1)-start_date).days
	return [start_date+datetime.timedelta(days=i) for i in range(r)]
 
start_date = datetime.date(2013,1,1)
end = datetime.date(2013,1,30)
day_count = 30
dateArray = date_range(start_date, end)
otherdate = datetime.date(2013,1,29)
 
s1 = "./horse_populator_output/{file}/pyspdir_twostage/{date}/scenarios.csv"

populator_scores_df = pd.DataFrame(columns=['File', 'Date', 'Variogram Score', 'Energy Score'])


for z in range(2):
	for d in range(30):
		date1 = '{}'.format(dateArray[d])
		file1 = '{}'.format(wsArray+cutpointArraynames2)
		file2 = '{}'.format(wsArray+cutpointArraynames3)
		file3 = '{}'.format(wsArray+cutpointArraynames[z])
		
		#Energy and Variogram dataframes
		pop_df = pd.read_csv(s1.format(file=file3, date=date1), index_col=0, parse_dates=True)
		
		var_score = r.getVariogramScore(type=1, dataFrame=pop_df, upperBound=4782, p=2)
		e_score = r.getEnergyScore(type=1, dataFrame=pop_df, upperBound=4782)

		populator_scores_df = populator_scores_df.append({'File':file3, 'Date':date1, 'Variogram Score': var_score,'Energy Score': e_score}, ignore_index=True)
		populator_scores_df['Date'] = pd.to_datetime(populator_scores_df.Date)
		populator_scores_df.to_csv(os.path.join('horse_populator_scores', 'Populator_Scores.csv'), index = False)

dflist = []
otherdflist = []

for d in range(30):
	#Scenario dataframes
	dflist.append(pd.read_csv(s1.format(file=file1, date=date1), index_col=0, parse_dates=True))
	


	dflist2 = pd.concat(dflist)
	df = pd.DataFrame(dflist2, index=range(2, 24,1))
	df = df[df.index % 24 != 0]
	df.to_csv(os.path.join('horse_populator_scores/Scenario_plot_set', 'df.csv'), index = False)

	otherdflist.append(pd.read_csv(s1.format(file=file2, date=dateArray[d]), index_col=0, parse_dates=True))
	otherdflist2 = pd.concat(otherdflist)
	#other_df = other_df.set_index(populator_scores_df['Date'])
	other_df = pd.DataFrame(otherdflist2)


#for single_date in (start_date + timedelta(n) for n in range(day_count)):
#Create dataframes to be plotted



forecast_plot = df[df.index % 24 != 0]
forecast_plot = forecast_plot['Wind: forecasts']
actual_plot = df[df.index % 24 != 0]
actual_plot = actual_plot['Wind: actuals']
fc_df = pd.DataFrame(forecast_plot)
ac_df = pd.DataFrame(actual_plot)
ax = df.drop(['Wind: forecasts', 'Wind: actuals'], axis='columns')
ax = df[df.index % 24 != 0]
ax1 = ax


forecast_plot2 = other_df[other_df.index % 24 != 0]
forecast_plot2 = forecast_plot2['Wind: forecasts']
actual_plot2 = other_df[other_df.index % 24 != 0]
actual_plot2 = actual_plot2['Wind: actuals']
fc_df2 = pd.DataFrame(forecast_plot2)
ac_df2 = pd.DataFrame(actual_plot2)
ax2 = other_df[other_df.index % 24 != 0]
ax2 = ax2.drop(['Wind: forecasts', 'Wind: actuals'], axis='columns')
ax3 = ax2


####
e_plot1 = populator_scores_df.loc[populator_scores_df['File']==file1, ['Date', 'Energy Score']]
e_plot2 = populator_scores_df.loc[populator_scores_df['File']==file2, ['Date', 'Energy Score']]
v_plot1 = populator_scores_df.loc[populator_scores_df['File']==file1, ['Date', 'Variogram Score']]
v_plot2 = populator_scores_df.loc[populator_scores_df['File']==file2, ['Date', 'Variogram Score']]

e_plot1['Energy Score'] = e_plot1['Energy Score'].str.get(0)
e_plot2['Energy Score'] = e_plot2['Energy Score'].str.get(0)
v_plot1['Variogram Score'] = v_plot1['Variogram Score'].str.get(0)
v_plot2['Variogram Score'] = v_plot2['Variogram Score'].str.get(0)

e_plot1 = e_plot1.set_index(['Date'])
e_plot2 = e_plot2.set_index(['Date'])
v_plot1 = v_plot1.set_index(['Date'])
v_plot2 = v_plot2.set_index(['Date'])


#Plot scenario dataframes
date2 = '{}'.format(end)
date3 = '{}'.format(otherdate)
s2 = 'Scenario Plot {end_date}'
s3 = 'horse_populator_scores/scenario_plot_set/scenario_plot_{date}.pdf'
s4 = 'horse_populator_scores/energy_variogram_plot.pdf'
s5 = 'Energy Plot {end_date}'
s6 = 'Variogram Plot {end_date}'
s7 = 'Date: {datee}'
s8 = 'Energy Score {cutpoint}'
s9 = 'Variogram Score {cutpoint}'

#startDate = .index[0] #seed the while loop, format Timestamp
#while (startDate >= df.index[0]) & (startDate < df.index[-1]):
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 15))

linestyle1 = '--'
linestyle2 = '-.'


ax1.plot(ax=ax, title=s2.format(end_date=date1), color='b', legend=None, linewidth=0.5)
fc_df.plot(ax=ax, linestyle=linestyle1, color='k', linewidth=3.5)
ac_df.plot(ax=ax, linestyle=linestyle2, color='r', linewidth=3.5)

ax3.plot(ax=ax2, title=s2.format(end_date=date1), color='b', legend=None, linewidth=0.5)
fc_df2.plot(ax=ax2, linestyle=linestyle1, color='k', linewidth=3.5)
ac_df2.plot(ax=ax2, linestyle=linestyle2, color='r', linewidth=3.5)

ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Wind Power (MW)', fontsize=12)
ax.set_ylim(0, 4782)
ax.locator_params(tight=True, nbins=4)

ax2.set_xlabel('Hour', fontsize=12)
ax2.set_ylim(0, 4782)
ax2.locator_params(tight=True, nbins=4)

fig.savefig(s3.format(date=dateArray))


#Plot energy and variogram scores
f, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 15))

e_plot1.plot(ax=ax3, use_index=True, subplots=True, color='b')
e_plot2.plot(ax=ax3, use_index=True, subplots=True, color='r')
v_plot1.plot(ax=ax4, use_index=True, subplots=True, color='b')
v_plot2.plot(ax=ax4, use_index=True, subplots=True, color='r')

ax3.set_xlabel('Day', fontsize=12)
ax3.set_ylabel('Energy Score', fontsize=12)
ax3.legend([s8.format(cutpoint=file1), s8.format(cutpoint=file2)])


ax4.set_xlabel('Day', fontsize=12)
ax4.set_ylabel('Variogram Score', fontsize=12)
ax4.legend([s9.format(cutpoint=file1), s9.format(cutpoint=file2)])


f.savefig(s4)

#find upper bound


