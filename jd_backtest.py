# -*- coding: utf-8 -*-
#########
# This script is meant to backtest JD's MG Weekly Picks
##########

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pandas_market_calendars as mcal
import mplfinance as mpf

# This is a vectorized backtest
# https://seekingalpha.com/mp/1201-value-momentum-breakouts/articles/5274410-v-m-weekly-breakout-stocks

# Root folder that contains subfolders for each week
rootdir = r'C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout/'

ticker_sectors = pd.read_csv(rootdir + 'ticker_sectors.csv',
                      names=['Ticker','Sector'],index_col='Ticker')
ticker_sectors = ticker_sectors[~ticker_sectors.index.duplicated(keep='first')]

mrkt_ga = pd.read_csv(rootdir + 'daily-market-momentum-ga.csv',
                      names=['DateTime','Pos','Neg','Index','Ann1','Ann2'],
                      header=0,index_col=0).drop(['Ann1','Ann2'],axis=1).dropna()
sp500_ga = pd.read_csv(rootdir + 'daily-sp-500-momentum-ga.csv',
                       names=['DateTime','Pos','Neg','Index','Ann1','Ann2'],
                       header=0,index_col=0).drop(['Ann1','Ann2'],axis=1).dropna()

# Choose the time that the stocks will be purchased. Lock to nearest good timestamp (not all stocks have valid data)
buystart_time = '6:33:00'
buystop_time = '6:36:00'

nyse = mcal.get_calendar('NYSE')
mkt_cal = nyse.schedule(start_date='2018-01-01', end_date='2025-01-01')

# Choose the STOP and LIMIT order percentages
stop_pct = 0.93
limit_pct = 1.10

for buy_bound in ['high','low']:

    # Initialize metrics
    pnl_weekly = []
    sell_dates = []

    bad_ticks=[]
    num_trade = 0
    num_skip = 0

    # loop over the list of csv files
    for fol in os.listdir(rootdir + 'data/'):
        #print('\n********** INITIATING BACKTEST {} **'.format(fol))
        csv_files = glob.glob(os.path.join(rootdir+'data/'+fol, "*.csv"))

        date1 = (datetime.datetime.strptime(fol, '%Y-%m-%d') -
                 datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        date2 = fol

        buydate = mkt_cal[mkt_cal.index >= date1].index[0].strftime('%Y-%m-%d')
        selldate = mkt_cal[mkt_cal.index <= date2].index[-1].strftime('%Y-%m-%d')


        # If Market Gauges are Negative, don't buy
        if (mrkt_ga.shift(1)[mrkt_ga.index >= buydate].iloc[0].Neg >
            mrkt_ga.shift(1)[mrkt_ga.index >= buydate].iloc[0].Pos):
            num_skip = num_skip + 4
            continue

        # If Market Gauge Negative Value > 40, don't buy
        if mrkt_ga.shift(1)[mrkt_ga.index >= buydate].iloc[0].Neg >= 40:
            num_skip = num_skip + 4
            continue

        # If SP500 Gauges are Negative, don't buy
        if (sp500_ga.shift(1)[sp500_ga.index >= buydate].iloc[0].Neg >
            sp500_ga.shift(1)[sp500_ga.index >= buydate].iloc[0].Pos):
            num_skip = num_skip + 4
            continue

        # If SP500 Gauge Negative Value > 40, don't buy
        if sp500_ga.shift(1)[sp500_ga.index >= buydate].iloc[0].Neg >= 40:
            num_skip = num_skip + 4
            continue


        # Initialize metrics
        pnl_tickers = []

        # loop over each week of data (each folder)
        for fpath in csv_files:
            file = os.path.basename(fpath)
            ticker = file.split('_')[4]
            sector = ticker_sectors.loc[ticker].Sector
            print(sector)

            # TODO: Load all sector gauge data into dataframe
            # LOOKUP SECTOR for stock, do not enter trade if Neg > 40
            # Exit trade if Neg > Pos or Neg > 40

            # Read the dataframe
            d = pd.read_csv(fpath)
            d['Datetime'] = pd.to_datetime(d['Datetime'])
            d = d.set_index('Datetime')
            d=d[d.index >= (buydate + ' ' + buystart_time)]

            if d.index[-1].strftime('%Y-%m-%d') == selldate:
                eow = d.index[-55]
            else:
                #print('No data on sell date')
                num_skip = num_skip + 1
                continue

            # Extract the BUY price using the buy datetime
            if d.index[0].strftime('%Y-%m-%d') == buydate:
                pbuy_list = d[(d.index >= (buydate + ' ' + buystart_time)) &
                          (d.index <= (buydate + ' ' + buystop_time))]
            else:
                #print('{} No data on buy date {}'.format(ticker,buydate))
                num_skip = num_skip + 1
                continue

            if pbuy_list.empty:
                #print('{} No data at the open {}'.format(ticker,buydate))
                num_skip = num_skip + 1
                continue

            # Combine the "high" and "low" fields, take average and stdev. This becomes
            # the bounding cases for the buy price.
            pbuy_mean = pbuy_list[['high','low']].stack().mean()
            pbuy_high = pbuy_list['high'].max()
            pbuy_low = pbuy_list['low'].min()
            pbuy_std = pbuy_list[['low','high']].std()

            if buy_bound == 'high':
                pbuy = pbuy_high
                actual_buydatetime = pbuy_list[pbuy_list['high']==pbuy_high].index[0]
            elif buy_bound == 'low':
                pbuy = pbuy_low
                actual_buydatetime = pbuy_list[pbuy_list['low']==pbuy_low].index[0]


            ################## LIMIT AND STOP LOSSES ########################
            # Determine the time of the first stop trigger or limit trigger
            # Note: They may never occur
            # If there's a stoploss trigger somewhere, assign it
            if not d[d['low'] <= pbuy*stop_pct].empty:
                first_stop = d[d['low'] <= pbuy*stop_pct].index[0]
            # If there's not a stoploss trigger, then set it to be EOW
            else:
                first_stop = eow

            # If there's a limit trigger somewhere, assign it
            if not d[d['high'] >= pbuy*limit_pct].empty:
                first_limit = d[d['high'] >= pbuy*limit_pct].index[0]
            # If there's not a limit trigger, then set it to be EOW
            else:
                first_limit = eow


            #################### MARKET GAUGE SIGNAL ########################
            # If there's a Market Gauge trigger, Neg > 40, assign the timestamp
            # Timestamp will occur at the open of the following day
            dt_filter = (mrkt_ga.index > buydate) & (mrkt_ga.index < selldate)
            if not mrkt_ga[dt_filter][mrkt_ga[dt_filter]['Neg'] >= 40].empty:
                first_mrkt_neg40 = d[ d.index > mrkt_ga[dt_filter]
                                  [mrkt_ga[dt_filter]['Neg'] >= 40]
                                  .index[0]].index[0]
            # If there's not a trigger, then set it to be EOW
            else:
                first_mrkt_neg40 = eow

            # If there's a Market Gauge trigger, Neg > Pos, assign the timestamp
            # Timestamp will occur at the open of the following day
            if not mrkt_ga[dt_filter][mrkt_ga[dt_filter]['Neg'] >=
                                      mrkt_ga[dt_filter]['Pos']].empty:
                first_mrkt_negpos = d[ d.index > mrkt_ga[dt_filter]
                                  [mrkt_ga[dt_filter]['Neg'] >=
                                   mrkt_ga[dt_filter]['Pos']]
                                  .index[0]].index[0]
            # If there's not a trigger, then set it to be EOW
            else:
                first_mrkt_negpos = eow


            #################### SP500 GAUGE SIGNAL ########################
            # If there's a SP500 Gauge trigger, Neg > 40, assign the timestamp
            # Timestamp will occur at the open of the following day
            dt_filter = (sp500_ga.index > buydate) & (sp500_ga.index < selldate)
            if not sp500_ga[dt_filter][sp500_ga[dt_filter]['Neg'] >= 40].empty:
                first_sp500_neg40 = d[ d.index > sp500_ga[dt_filter]
                                  [sp500_ga[dt_filter]['Neg'] >= 40]
                                  .index[0]].index[0]
            # If there's not a trigger, then set it to be EOW
            else:
                first_sp500_neg40 = eow

            # If there's a SP500 Gauge trigger, Neg > Pos, assign the timestamp
            # Timestamp will occur at the open of the following day
            if not sp500_ga[dt_filter][sp500_ga[dt_filter]['Neg'] >=
                                      sp500_ga[dt_filter]['Pos']].empty:
                first_sp500_negpos = d[ d.index > sp500_ga[dt_filter]
                                  [sp500_ga[dt_filter]['Neg'] >=
                                   sp500_ga[dt_filter]['Pos']]
                                  .index[0]].index[0]
            # If there's not a trigger, then set it to be EOW
            else:
                first_sp500_negpos = eow


            # Compare the timestamp for the first stop, the first limit,
            # the first market and sp500 gauge signals, and eow
            # The earliest trigger defines the termination of the trade
            exit_dict = {'SellTime':
                       [first_mrkt_neg40, first_mrkt_negpos,
                        first_sp500_neg40, first_sp500_negpos,
                        first_limit, first_stop],
                       'Trigger':['Mrkt Neg > 40', 'Mrkt Neg > Pos',
                               'SP500 Neg > 40','SP500 Neg > Pos',
                               'Limit', 'Stop']}
            exit_df = pd.DataFrame(data=exit_dict).sort_values(by=['SellTime'])
            selltime = exit_df.iloc[0].SellTime
            if selltime == eow:
                trigger = 'EOW'
            else:
                trigger = exit_df.iloc[0].Trigger


            # Extract the sell price based on the termination of the trade. Calculate pnl
            psell = d.loc[selltime,'open']
            pnl = (psell/pbuy-1)*100
            pnl_tickers.append(pnl)

            # Print the trade
            num_trade = num_trade + 1
            #print('{}: Bought @ {:.2f} on {}, {} @ {:.2f} on {}, PNL: {:.2f}%'.format(ticker,pbuy,actual_buydatetime,trigger,psell,selltime,pnl))

        #print('Total Weekly PNL {:.2f}%'.format(np.nanmean(pnl_tickers)))
        pnl_weekly.append(np.nanmean(pnl_tickers))
        sell_dates.append(eow.date().strftime('%Y-%m-%d'))

    dates_list = [datetime.datetime.strptime(sell_date, '%Y-%m-%d').date() for sell_date in sell_dates]
    num_yrs = ((max(dates_list)-min(dates_list)).days)/365

    print('\n***** SUMMARY FOR {}'.format(buy_bound))
    print('Traded {} ({:.1f}%) of the securities. Skipped {}.'.format(num_trade,num_trade/(num_trade+num_skip)*100,num_skip))
    print('Purchased between {} and {}'.format(buystart_time, buystop_time))
    print('Total PNL: {:.0f}%'.format(np.nansum(pnl_weekly)))

    ann = np.nansum(pnl_weekly)/num_yrs
    print('Annualized: {:.1f}% (${:.2f} profit/loss per year, if trading $1,000 each week)\n'.format(ann,ann/100*1000))


plt.close("all")
# Plot the trade
wtmd = dict(warn_too_much_data=10000000000000)
mpf.plot(d, title=ticker, type='hollow_and_filled', style='yahoo',
        hlines=dict(hlines=[pbuy*stop_pct,pbuy*limit_pct],colors=['r','g'],linestyle='-'),
        vlines=dict(vlines=[actual_buydatetime,selltime],colors=['g','r'],linestyle='-',alpha=0.3,linewidths=2),
        **wtmd)

# Plot the weekly returns
results = pd.DataFrame(pnl_weekly,index=sell_dates,columns=['PNL'])
fig, ax = plt.subplots()
colormat=np.where(results['PNL']>0, 'g','r')
ax.bar(sell_dates, results['PNL'], width=0.75, color=colormat)
plt.axhline(y=0,color='black',linestyle='-')

# TODO Plot the annual returns

