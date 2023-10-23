# -*- coding: utf-8 -*-
#########
# This script is meant to backtest JD's MG Weekly Picks
##########

import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# https://seekingalpha.com/mp/1201-value-momentum-breakouts/articles/5274410-v-m-weekly-breakout-stocks

# Root folder that contains subfolders for each week
rootdir = r'C:/Users/ryant/Documents/Stock_Market/Python/universe_data/sp500/stock_data/VM Weekly Breakout/data/'

# Choose the time that the stocks will be purchased
tradetime = '06:30:00'
print('\nPurchase Time {}'.format(tradetime))

# Choose the STOP and LIMIT order percentages
stop_pct = 0.95
limit_pct = 1.03

# Initialize metrics
pnl_weekly = []
trade_dates = []

# loop over the list of csv files
for fol in os.listdir(rootdir):
    print('\n************************************')
    print('** INITIATING BACKTEST {} **'.format(fol))
    print('************************************')
    csv_files = glob.glob(os.path.join(rootdir+fol, "*.csv"))

    # Initialize metrics
    pnl_tickers = []

    # loop over the list of csv files
    for fpath in csv_files:
        file = os.path.basename(fpath)
        #print('\nFile Loop {}'.format(file))

        ticker = file.split('_')[4]
        tradedate = file.split('_')[5]

        # Initialize the buytime and EOW as a datetime object
        buytime = datetime.datetime.strptime(tradedate + ' ' + tradetime,'%Y-%m-%d %H:%M:%S')
        eow = datetime.datetime.combine(buytime.date() + datetime.timedelta(days=7), datetime.time(12,59,00))

        # Read the dataframe
        d = pd.read_csv(fpath)
        d['Datetime'] = pd.to_datetime(d['Datetime'])
        d = d.set_index('Datetime')

        # Extract the BUY price using the buytime
        pbuy = d.loc[buytime, 'close']

        # Determine the time of the first stop trigger or limit trigger
        # Note: They may not occur before EOW
        if not d[d['close'] <= pbuy*stop_pct].empty:
            first_stop = d[d['close'] <= pbuy*stop_pct].index[0]
        else:
            first_stop = eow

        if not d[d['close'] >= pbuy*limit_pct].empty:
            first_limit = d[d['close'] >= pbuy*limit_pct].index[0]
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
        psell = d.loc[selltime,'close']
        pnl = (psell/pbuy-1)*100
        pnl_tickers.append(pnl)

        # Print the trade
        print('{}: Bought @ {:.2f}, {} @ {:.2f} at {}, PNL: {:.2f}%'.format(ticker,pbuy,trigger,psell,selltime,pnl))

    print('Total Weekly PNL {:.2f}%'.format(np.average(pnl_tickers)))
    pnl_weekly.append(np.average(pnl_tickers))
    trade_dates.append(tradedate)

plt.close("all")
# Plot the trade
d['close'].plot(grid=True, style='-', color='k', label=ticker, legend=True)
plt.axvline(x = buytime, color = 'm', label = 'BUY')
plt.axvline(x = selltime, color = 'm', label = 'SELL')
plt.axhline(y = pbuy*stop_pct, color = 'r', label = 'Stop', linestyle = '--')
plt.axhline(y = pbuy*limit_pct, color = 'lime', label = 'Limit', linestyle = '--')

results = pd.DataFrame(pnl_weekly,index=trade_dates,columns=['PNL'])
fix,ax = plt.subplots()
colormat=np.where(results['PNL']>0, 'g','r')
ax.bar(trade_dates, results['PNL'], width=0.75, color=colormat)
plt.axhline(y=0,color='black',linestyle='-')



