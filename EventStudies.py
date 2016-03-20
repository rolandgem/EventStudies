""" 
Created on November 24, 2015
@author: Roland Gemayel
Description: EventStudies.py shows how stock returns change before and after an event.

Inputs: symbols_list, dt_start, dt_end, L1, window 
Outputs: 
Libraries: Requires pandas_datareader to be installed. In terminal paste "pip install pandas-datareader"
"""

import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from pandas.tseries.offsets import *
from collections import defaultdict
from collections import OrderedDict
from math import sqrt


def EventStudies():
	
	# Define list of stocks to conduct event analysis on.
	symbols_list = ['AES', 'AET', 'AFL', 'AVP', 'CLX', 'GM', '^GSPC']
	
	# Start and End dates
	dt_start = dt.datetime(2012, 1,1)
	dt_end = dt.datetime(2015, 1,1)

	# Download historical Adjusted Closing prices using Pandas downloader for Yahoo
	data = DataReader(symbols_list, 'yahoo', dt_start, dt_end)['Adj Close']

	# Create dataframe data_ret which includes returns
	data_ret = data/data.shift(1) - 1

	# Define event threshold variable daily_diff
	daily_diff = 0.03

	# Positive event if daily stock return > market return by daily_diff
	# Negative event if daily stock return < market return by daily_diff
	# otherwise no event has occurred.
	

	# Create an events data frame data_events, where columns = names of all stocks, and rows = daily dates 
	events_col = symbols_list[:] # Use [:] to deep copy the list
	events_col.remove('^GSPC') # We dont't need to create events for the S&P500
	events_index = data_ret.index # Copy the date index from data_ret to the events data frame
	data_events = pd.DataFrame(index=events_index, columns=events_col)
	
	# Fill in data_events with 1 for positive events, -1 for negative events, and NA otherwise.
	for i in events_col:
		data_events[i] = np.where((data_ret[i] - data_ret['^GSPC']) > daily_diff, 1, np.where((data_ret[i] - data_ret['^GSPC']) < -daily_diff, -1, np.nan))
	
	# Calculate abnormal returns based on market model (R_it = a_i + B_i*R_mt + e_it)
	# Define estimation period L1: the greater, the more accurate the model
	L1 = 30

	# Define window for forward and backward looking period. Should be less than L1,
	window = 20
	
	# Create 2 dictionaries of dictionaries (for positive and negative events) to store the 
	# abnormal returns (AR) values of each window day, for each stock.
	pos_dict = defaultdict(dict)
	neg_dict = defaultdict(dict)

	# For each stock, locate each event and calculate abnormal return for previous window days and future window days
	for s in events_col:
		pos_event_dates = data_events[s][data_events[s] == 1].index.tolist()
		neg_event_dates = data_events[s][data_events[s] == -1].index.tolist()

		# Create dictionary for each stock to store the AR values of each window day for each event
		pos_dict_s = defaultdict(dict)
		neg_dict_s = defaultdict(dict)

		for pos_event in pos_event_dates:
			date_loc = data_ret.index.get_loc(pos_event) 
			# Go to beginning of backward window and calculate AR from backward till forward window.
			date_loc = date_loc - window
			
			if date_loc > L1 and date_loc <= len(data_ret) - (2*window+1):
				index_range = (2*window) + 1

				# Create dictionairy to store the AR values for each day of this event
				pos_dict_s_event = OrderedDict() 
				for d in range(index_range):
					date_loc2 = date_loc + d

					# Parameters to estimate market model
					u_i = data_ret[s][date_loc2-L1 : date_loc2-1].mean()
					u_m = data_ret['^GSPC'][date_loc2-L1 : date_loc2-1].mean()
					R_i = data_ret.ix[date_loc2, s]
					R_m = data_ret.ix[date_loc2,'^GSPC']
					beta_i = ((R_i-u_i)*(R_m - u_m))/(R_m - u_m)**2
					alpha_i = u_i - (beta_i*u_m)
					var_err = (1/(L1 -2))*(R_i - alpha_i - (beta_i*R_m))**2
					AR_i = R_i - alpha_i - (beta_i*R_m)

					pos_dict_s_event[date_loc2] = AR_i
					
				pos_dict_s[pos_event] = pos_dict_s_event

		pos_dict[s] = pos_dict_s


		for neg_event in neg_event_dates:
			date_loc = data_ret.index.get_loc(neg_event)
			# Go to beginning of backward window and calculate AR from backward till forward window.
			date_loc = date_loc - window

			if date_loc > L1 and date_loc <= len(data_ret) - (2*window+1):
				index_range = (2*window) + 1

				# Create dictionairy to store the AR values for each day of this event
				neg_dict_s_event = OrderedDict() 
				for d in range(index_range):
					date_loc2 = date_loc + d

					# Parameters to estimate market model
					u_i = data_ret[s][date_loc2-L1 : date_loc2-1].mean()
					u_m = data_ret['^GSPC'][date_loc2-L1 : date_loc2-1].mean()
					R_i = data_ret.ix[date_loc2, s]
					R_m = data_ret.ix[date_loc2, '^GSPC']
					beta_i = ((R_i-u_i)*(R_m - u_m))/(R_m - u_m)**2
					alpha_i = u_i - (beta_i*u_m)
					var_err = (1/(L1 -2))*(R_i - alpha_i - (beta_i*R_m))**2
					AR_i = R_i - alpha_i - (beta_i*R_m)

					neg_dict_s_event[date_loc2] = AR_i

				neg_dict_s[neg_event] = neg_dict_s_event

		neg_dict[s] = neg_dict_s
 

	# Create empty Abnormal Returns data frame
	abret_col = symbols_list[:] # Use [:] to deep copy the list
	abret_col.remove('^GSPC') # We dont't need to calculate abnormal returns for the S&P500
	abret_index = range(-window, window+1)
	pos_data_abret = pd.DataFrame(index=abret_index, columns=abret_col)
	neg_data_abret = pd.DataFrame(index=abret_index, columns=abret_col)
	
	for h in abret_col:
		if h in pos_dict.keys():
			for z in abret_index:
				pos_data_abret[h][z] = np.mean([x.values()[z+window] for x in pos_dict[h].values()])

	for f in abret_col:
		if f in neg_dict.keys():
			for v in abret_index:
				neg_data_abret[f][v] = np.mean([x.values()[v+window] for x in neg_dict[f].values()])


	# Create Cumulative Abnormal Return (CAR) Tables pos_CAR and neg_CAR
	pos_CAR = pos_data_abret.cumsum()
	neg_CAR = neg_data_abret.cumsum()


	# Plot pos_CAR and neg_CAR
	plt.clf()
	plt.plot(pos_CAR)
	plt.legend(pos_CAR)
	plt.ylabel('CAR')
	plt.xlabel('Window')
	matplotlib.rcParams.update({'font.size': 8})
	plt.savefig('PositiveCAR_All.png', format='png')

	plt.clf()
	plt.plot(neg_CAR)
	plt.legend(neg_CAR)
	plt.ylabel('CAR')
	plt.xlabel('Window')
	matplotlib.rcParams.update({'font.size': 8})
	plt.savefig('NegativeCAR_All.png', format='png')

	# Sum CAR for positive and negative events to plot only the aggregate CAR

	pos_CAR['SUM'] = pos_CAR.sum(axis=1)
	neg_CAR['SUM'] = neg_CAR.sum(axis=1)

	plt.clf()
	plt.plot(pos_CAR['SUM'])
	plt.legend(pos_CAR['SUM'])
	plt.ylabel('CAR')
	plt.xlabel('Window')
	matplotlib.rcParams.update({'font.size': 8})
	plt.savefig('PositiveCAR_SUM.png', format='png')

	plt.clf()
	plt.plot(neg_CAR['SUM'])
	plt.legend(neg_CAR['SUM'])
	plt.ylabel('CAR')
	plt.xlabel('Window')
	matplotlib.rcParams.update({'font.size': 8})
	plt.savefig('NegativeCAR_SUM.png', format='png')


EventStudies()


