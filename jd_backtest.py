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

ticker_sectors = pd.read_csv(rootdir + 'ticker_sector.csv',
                      names=['Ticker','Sector'],index_col='Ticker')
ticker_sectors = ticker_sectors[~ticker_sectors.index.duplicated(keep='first')]

# Load the daily market and sp500 gauges
mrkt_ga = pd.read_csv(rootdir + 'daily-market-momentum-ga.csv',
                      names=['DateTime','Pos','Neg','Index','Ann1','Ann2'],
                      header=0,index_col=0).drop(['Ann1','Ann2'],axis=1).dropna()
sp500_ga = pd.read_csv(rootdir + 'daily-sp-500-momentum-ga.csv',
                       names=['DateTime','Pos','Neg','Index','Ann1','Ann2'],
                       header=0,index_col=0).drop(['Ann1','Ann2'],axis=1).dropna()


# Load the sector daily gauges into a single dataframe
sindex = 0
csv_files = glob.glob(os.path.join(rootdir+'Daily Sector Gauges/', "*.csv"))
for fpath in csv_files:
    file = os.path.basename(fpath)
    sname = file[:-4]
    if sindex == 0:
        s = pd.read_csv(fpath,usecols=[0,1,2],
                    names=['Date',sname + ' Pos',sname + ' Neg'],
                    header=0).dropna()
        s['Date'] = pd.to_datetime(s['Date'])
        s['Date'] = pd.to_datetime(s['Date'].dt.date)
        s = s.set_index('Date')
    else:
        s1 = pd.read_csv(fpath, usecols=[0,1,2],
                    names=['Date',sname + ' Pos',sname + ' Neg'],
                    header=0).dropna()
        s1['Date'] = pd.to_datetime(s1['Date'])
        s1['Date'] = pd.to_datetime(s1['Date'].dt.date)
        s1 = s1.set_index('Date')
        s[sname + ' Pos'] = s1[sname + ' Pos']
        s[sname + ' Neg'] = s1[sname + ' Neg']
    sindex = sindex + 1

# Add time to sector index, to reflect updates after market close
s.index = s.index + datetime.timedelta(hours=20)

# Choose the time that the stocks will be purchased. Lock to nearest good timestamp (not all stocks have valid data)
buystart_time = '6:32:00'
buystop_time = '6:36:00'

nyse = mcal.get_calendar('NYSE')
mkt_cal = nyse.schedule(start_date='2021-01-01', end_date=datetime.datetime.now().date().strftime('%Y-%m-%d'))

# Choose the STOP and LIMIT order percentages
stop_pct = 0.90
limit_pct = 1.10

# Initialize metrics
dict_pnl = {'Best':{}, 'Worst': {}, 'Expected': {}}
pnl_weekly = []
sell_dates = []

bad_ticks=[]
num_trade = 0
num_skip = 0

# loop over the list of csv files
for fol in os.listdir(rootdir + 'data/'):
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


    # print(' ')
    #print('\n********** INITIATING BACKTEST {} **'.format(fol))

    # loop over each week of data (each folder)
    for fpath in csv_files:
        file = os.path.basename(fpath)
        ticker = file.split('_')[4]
        sector = ticker_sectors.loc[ticker].Sector

        # Do not enter trade if Sector Neg > Pos
        # if (s.shift(1)[s.index >= buydate].iloc[0][sector + ' Neg'] >=
        #     s.shift(1)[s.index >= buydate].iloc[0][sector + ' Pos']):
        #     num_skip = num_skip + 1
        #     # print('**** Avoided trade due to Sector Gauge Neg > Pos')
        #     continue

        # # Do not enter trade if Sector Neg > 40
        # if s.shift(1)[s.index >= buydate].iloc[0][sector + ' Neg'] >= 40:
        #     num_skip = num_skip + 1
        #     print('Avoided trade due to Sector Gauge Negative > 40')
        #     continue

        # Read the dataframe
        d = pd.read_csv(fpath)
        d['Datetime'] = pd.to_datetime(d['Datetime'])
        d = d.set_index('Datetime')
        d=d[d.index >= (buydate + ' ' + buystart_time)]

        if d.index[-1].strftime('%Y-%m-%d') == selldate:
            eow = d.index[-55]
        else:
            # print('**** No data on sell date')
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

        # Determine the buy price. Can choose:
        # "best" (buying at the low)
        # "worst" (buying at the high)
        # "expected" (buying at a weighted average of highs and lows)
        pmax = np.ceil(pbuy_list['high'].max()*100)/100
        pmin = np.floor(pbuy_list['low'].min()*100)/100
        prange = np.linspace(pmin,pmax,num=int((pmax-pmin)/0.01+1))

        dtest=pd.DataFrame(index=prange)
        for x in range(len(pbuy_list.index)):
            dtest[pbuy_list.index[x]] = 0
            dtest[pbuy_list.index[x]][(pbuy_list.iloc[x].low <= dtest.index) & (pbuy_list.iloc[x].high >= dtest.index-0.01)]=1
        dtest['sum'] = dtest.sum(axis=1)
        dtest['p(x)'] = dtest['sum']/dtest['sum'].sum()

        dp_scenarios = pd.DataFrame(data=[pbuy_list['low'].min(),
                                     pbuy_list['high'].max(),
                                     (dtest['p(x)']*dtest.index).sum()],
                                    index=['Best','Worst','Expected'],
                                    columns=['Price'])

        d[['Best Stop','Best Limit','Worst Stop','Worst Limit',
           'Expected Stop','Expected Limit']] = [
            dp_scenarios.loc['Best','Price']*stop_pct,
            dp_scenarios.loc['Best','Price']*limit_pct,
            dp_scenarios.loc['Worst','Price']*stop_pct,
            dp_scenarios.loc['Worst','Price']*limit_pct,
            dp_scenarios.loc['Expected','Price']*stop_pct,
            dp_scenarios.loc['Expected','Price']*limit_pct]


        d[['Expected Limit','Worst Limit','Best Limit']] = d[[
            'Expected Limit','Worst Limit','Best Limit']].gt(d['high'],axis='rows')
        d[['Expected Stop','Worst Stop','Best Stop']] = d[[
            'Expected Stop','Worst Stop','Best Stop']].lt(d['low'],axis='rows')

        actual_buydatetime = pbuy_list.index[-1]


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


        #################### SECTOR GAUGE SIGNAL ########################
        dt_filter = (s.index >= buydate) & (s.index < selldate)

        if not s[dt_filter][s[dt_filter][sector + ' Neg'] >= 40].empty:
            first_s_neg40 = d[ d.index >= s[dt_filter]
                              [s[dt_filter][sector + ' Neg'] >= 40]
                              .index[0]].index[0]
        # If there's not a trigger, then set it to be EOW
        else:
            first_s_neg40 = eow

        # If there's a SECTOR Gauge trigger, Neg > Pos, assign the timestamp
        # Timestamp will occur at the open of the following day
        if not s[dt_filter][s[dt_filter][sector + ' Neg'] >=
                                  s[dt_filter][sector + ' Pos']].empty:
            first_s_negpos = d[ d.index > s[dt_filter]
                              [s[dt_filter][sector + ' Neg'] >=
                               s[dt_filter][sector + ' Pos']]
                              .index[0]].index[0]
        # If there's not a trigger, then set it to be EOW
        else:
            first_s_negpos = eow


        ################## LIMIT AND STOP LOSSES ########################
        # Determine the time of the first stop trigger or limit trigger
        # Note: They may never occur

        for x in ['Best','Worst','Expected']:
            if not d[~d[x + ' Stop']].empty:
                dp_scenarios.loc[x,'Stop'] = d[~d[x + ' Stop']].index[0]
            else:
                dp_scenarios.loc[x,'Stop'] = eow

            if not d[~d[x + ' Limit']].empty:
                dp_scenarios.loc[x,'Limit'] = d[~d[x + ' Limit']].index[0]
            else:
                dp_scenarios.loc[x,'Limit'] = eow


        # Compare the timestamp for the first stop, the first limit,
        # the first market and sp500 gauge signals, and eow
        # The earliest trigger defines the termination of the trade
        num_trade = num_trade + 1
        for x in dp_scenarios.index:
            exit_dict = {'SellTime':
                       [first_mrkt_neg40, first_mrkt_negpos,
                        first_sp500_neg40, first_sp500_negpos,
                        # first_s_neg40,
                        # first_s_negpos,
                        dp_scenarios.loc[x,'Stop'], dp_scenarios.loc[x,'Limit']],
                       'Trigger':['Mrkt Neg > 40', 'Mrkt Neg > Pos',
                               'SP500 Neg > 40','SP500 Neg > Pos',
                                # sector + ' Neg > 40',
                                # sector + ' Neg > Pos',
                               'Stop','Limit']}
            exit_df = pd.DataFrame(data=exit_dict).sort_values(by=['SellTime'])
            selltime = exit_df.iloc[0].SellTime
            if selltime == eow:
                trigger = 'EOW'
            else:
                trigger = exit_df.iloc[0].Trigger


            # Extract the sell price based on the termination of the trade. Calculate pnl
            psell = d.loc[selltime,'open']
            dict_pnl[x][buydate,ticker] = (psell/dp_scenarios.loc[x,'Price']-1)*100

            # Print the trade
            # print('{} {}: Bought @ {:.2f} on {}, {} @ {:.2f} on {}, PNL: {:.2f}%'.format(x,
            #     ticker, dp_scenarios.loc[x,'Price'],
            #     actual_buydatetime, trigger, psell, selltime, dict_pnl[x][buydate,ticker]))
        # print('')

df_pnl = pd.DataFrame(dict_pnl)
df_pnl.index.names=['Date','Ticker']
df_pnl.index = df_pnl.index.set_levels([pd.to_datetime(df_pnl.index.levels[0]), df_pnl.index.levels[1]])

dates_list = df_pnl.index.get_level_values(0).unique()
num_yrs = ((dates_list.max()-dates_list.min()).days)/365

pnl_weekly = df_pnl.groupby('Date').mean()

pnl_yearly = df_pnl.groupby('Date').mean()
pnl_yearly.index = pnl_yearly.index.year
pnl_yearly = pnl_yearly.groupby('Date').sum()

print('\n***** SUMMARY')
print('Traded {} ({:.1f}%) of the securities. Skipped {}.'.format(num_trade,num_trade/(num_trade+num_skip)*100,num_skip))
print('Purchased between {} and {}'.format(buystart_time, buystop_time))
print('-- TOTAL PNL')
print('Best PNL: {:.0f}% ({:.0f}% Ann.)'.format(pnl_weekly.sum().Best, pnl_weekly.sum().Best/num_yrs))
print('Expected PNL: {:.0f}% ({:.0f}% Ann.)'.format(pnl_weekly.sum().Expected, pnl_weekly.sum().Expected/num_yrs))
print('Worst PNL: {:.0f}% ({:.0f}% Ann.)'.format(pnl_weekly.sum().Worst, pnl_weekly.sum().Worst/num_yrs))

plt.close("all")
# Plot the trade
wtmd = dict(warn_too_much_data=10000000000000)
mpf.plot(d, title=ticker, type='hollow_and_filled', style='yahoo',
        hlines=dict(hlines=[dp_scenarios.loc[x,'Price'],dp_scenarios.loc[x,'Price']*stop_pct,
                            dp_scenarios.loc[x,'Price']*limit_pct],
                            colors=['k','r','g'],linestyle='-'),
        vlines=dict(vlines=[actual_buydatetime,selltime],colors=['g','r'],linestyle='-',alpha=0.3,linewidths=2),
        **wtmd)

# # Plot the weekly returns
pnl_weekly[['Best','Expected','Worst']].plot.bar(color=['g','k','r'])
pnl_yearly[['Best','Expected','Worst']].plot.bar(color=['g','k','r'])

