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

# Choose the time that the stocks will be purchased. Lock to nearest good timestamp (not all stocks have valid data)
buytime = '7:00:00'
print('\nIntended Purchase Time {}'.format(buytime))

# Choose the STOP and LIMIT order percentages
stop_pct = 0.97
limit_pct = 1.05

# Initialize metrics
pnl_weekly = []
sell_dates = []

bad_ticks=[]

# loop over the list of csv files
for fol in os.listdir(rootdir):
    print('\n************************************')
    print('** INITIATING BACKTEST {} **'.format(fol))
    print('************************************')
    csv_files = glob.glob(os.path.join(rootdir+fol, "*.csv"))

    # Initialize metrics
    pnl_tickers = []

    # loop over each week of data (each folder)
    for fpath in csv_files:
        file = os.path.basename(fpath)

        ticker = file.split('_')[4]
        buydate = file.split('_')[5]

        # Initialize the buy datetime and EOW as a datetime object
        buydatetime = datetime.datetime.strptime(buydate + ' ' + buytime,'%Y-%m-%d %H:%M:%S')

        # Read the dataframe
        d = pd.read_csv(fpath)
        d['Datetime'] = pd.to_datetime(d['Datetime'])
        d = d.set_index('Datetime')
        d=d[d.index >= buydatetime]
        eow = d.index[-1]

        # Extract the BUY price using the buy datetime
        # Account for Christmas Eve, Good Friday, 4th of July, November Half-Day
        # JUST PICK CLOSEST TIME TO THE DESIRED BUYTIME
        pbuy = d.iloc[0]['close']
        actual_buydatetime = d.index[0]

        # Determine the time of the first stop trigger or limit trigger
        # Note: They may never occur
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
        print('{}: Bought @ {:.2f} on {}, {} @ {:.2f} on {}, PNL: {:.2f}%'.format(ticker,pbuy,actual_buydatetime,trigger,psell,selltime,pnl))

    print('Total Weekly PNL {:.2f}%'.format(np.nanmean(pnl_tickers)))
    pnl_weekly.append(np.nanmean(pnl_tickers))
    sell_dates.append(eow.date().strftime('%Y-%m-%d'))

dates_list = [datetime.datetime.strptime(sell_date, '%Y-%m-%d').date() for sell_date in sell_dates]
num_yrs = ((max(dates_list)-min(dates_list)).days)/365

print('\n***** SUMMARY')
print('PNL: {:.0f}%'.format(np.nansum(pnl_weekly)))

ann = ((1 + np.nansum(pnl_weekly)/100)**(1/num_yrs)-1)*100
print('Annualized: {:.1f}%'.format(ann))

plt.close("all")
# Plot the trade
d['close'].plot(grid=True, style='-', color='k', label=ticker, legend=True)
plt.axvline(x = actual_buydatetime, color = 'm', label = 'BUY')
plt.axvline(x = selltime, color = 'm', label = 'SELL')
plt.axhline(y = pbuy*stop_pct, color = 'r', label = 'Stop', linestyle = '--')
plt.axhline(y = pbuy*limit_pct, color = 'lime', label = 'Limit', linestyle = '--')

results = pd.DataFrame(pnl_weekly,index=sell_dates,columns=['PNL'])
fix,ax = plt.subplots()
colormat=np.where(results['PNL']>0, 'g','r')
ax.bar(sell_dates, results['PNL'], width=0.75, color=colormat)
plt.axhline(y=0,color='black',linestyle='-')



