# -*- coding: utf-8 -*-
#########
# This script is meant to backtest JD's MG Weekly Picks
##########

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Root folder that contains subfolders for each week
fol = r'C:/Users/ryant/Documents/Stock_Market/Python/universe_data/sp500/stock_data/yf/benchmark'

# Folder structure should be organized by the week. Peek into the folder to
# iterate through each week
tradedate = '2022-03-14'

# Inside the weekly folder, load each stock

# Choose the time that the stocks will be purchased
tradetime = '06:30:00'

# Choose the STOP and LIMIT order percentages
stop_pct = 0.95
limit_pct = 1.02

# Initialize the buytime and EOW as a datetime object
buytime = datetime.datetime.strptime(tradedate + ' ' + tradetime,'%Y-%m-%d %H:%M:%S')
eow = datetime.datetime.combine(buytime.date() + datetime.timedelta(days=4), datetime.time(12,59,00))

# Read the dataframe
d = pd.read_csv(fol + '/0_0_1_0_SPY_2022-03-13_2022-03-19.csv')
d['Date'] = pd.to_datetime(d['Date'])
d = d.set_index('Date')

# Extract the BUY price using the buytime
pbuy = d.loc[buytime, 'Close']

# Determine the time of the first stop trigger or limit trigger
# Note: They may not occur before EOW
if not d[d['Close'] <= pbuy*stop_pct].empty:
    first_stop = d[d['Close'] <= pbuy*stop_pct].index[0]
else:
    first_stop = eow
    
if not d[d['Close'] >= pbuy*limit_pct].empty:
    first_limit = d[d['Close'] >= pbuy*limit_pct].index[0]
else:
    first_limit = eow

# Compare the timestamp for the first stop, the first limit, and eow
# The earliest trigger defines the termination of the trade
if (first_stop < eow) & (first_stop < first_limit):
    selltime = first_stop
    trigger='STOPPED OUT'
elif (first_limit < eow) & (first_limit < first_stop):
    selltime = first_limit
    trigger='LIMIT ORDER'
else:
    selltime = eow
    trigger='EOW REACHED'

# Extract the sell price based on the termination of the trade. Calculate pnl
psell = d.loc[selltime,'Close']
pnl = (psell/pbuy-1)*100

# Plot the trade
d['Close'].plot(grid=True, style='-', color='k', label='SPY', legend=True)
plt.axvline(x = buytime, color = 'm', label = 'BUY')
plt.axvline(x = selltime, color = 'm', label = 'SELL')
plt.axhline(y = pbuy*stop_pct, color = 'r', label = 'Stop', linestyle = '--') 
plt.axhline(y = pbuy*limit_pct, color = 'lime', label = 'Limit', linestyle = '--')

# Print the trade
print('\n*************************')
print('** INITIATING BACKTEST **')
print('*************************\n')
print(buytime)
print('SPY: Bought @ {:.2f}, {} @ {:.2f} at {}, PNL: {:.2f}%'.format(pbuy,trigger,psell,selltime,pnl))