# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:17:04 2022

@author: rtm
"""

######
# STEP 2: USE ALL US-STOCK DATA TO PRODUCE SIMULATED WATCHLISTS
#         THESE SIMULATE WHAT MY WATCHLISTS "WOULD HAVE BEEN" AT THAT TIME
######

import glob
import pandas as pd
import os
import datetime
from time import perf_counter
import json

# Folder containing all NYSE symbols with minute-data going back ~30 days
fol = r'C:/Users/rtm/Desktop/stock_data/'
csv_files = glob.glob(os.path.join(fol + 'US_data', "*.csv"))

# Create date ranges and market/premarket times
date_start = '2021-12-27'
market_mg = pd.read_csv(fol + 'daily-market-momentum-ga_2022-04-09.csv',
                        skiprows=1,
                        names=['Date','Positive','Negative','Russell2000']
                        ).dropna(axis=0,how='any')
market_mg['Date'] = pd.to_datetime(market_mg['Date']).dt.normalize()
market_mg = (market_mg.dropna(axis=0, how='any')).set_index('Date')
market_mg = market_mg.shift(1).fillna(False)
market_mg['Compare'] = ((market_mg['Positive'] > market_mg['Negative']) &
                        (market_mg['Negative'] < 40) )
dates = market_mg.loc[(market_mg.index > date_start) & market_mg['Compare']].index.date

# dates= [datetime.datetime.strptime('2022-04-01', '%Y-%m-%d').date()]
market_start = datetime.datetime.strptime('12:59', '%H:%M').time()
market_stop = datetime.datetime.strptime('06:36', '%H:%M').time()

pm_start = (datetime.datetime.strptime('00:00', '%H:%M')).time()
pm_stop = datetime.datetime.strptime('06:26', '%H:%M').time()


# General Filter Criteria
close_min = 0.75
close_max = 17
pm_bars = 8*5 # multiply by 5 for 1min bars

# Market filter criteria
m_pct_chg = 0.05
m_cumvol_min = 150000
m_vol_max = 100000/5 # divide by 5 for 1min bars

# Premarket filter criteria
pm_pct_chg = 0.05
pm_vol_max = 50000/5 # divide by 5 for 1min bars

market_watchlist = {}
premarket_watchlist = {}

sym_cnt_mod = round(len(csv_files)/10)
dt_cnt_mod = round(len(dates)/len(dates))
dt_cnt = 0
tic_total = time.perf_counter()
for idate in dates:
    tic_dt = time.perf_counter()

    if dt_cnt%dt_cnt_mod == 0:
        print('\nDATES {:.0f}% COMPLETE'.format(dt_cnt/len(dates)*100))
    dt_cnt = dt_cnt+1

    m_1 = (datetime.datetime.combine(idate,market_start) -
          datetime.timedelta(days=1))
    m_2 = datetime.datetime.combine(idate, market_stop)

    pm_1 = datetime.datetime.combine(idate, pm_start)
    pm_2 = datetime.datetime.combine(idate, pm_stop)

    # loop over the list of csv files
    sym_cnt=0
    syms_market = []
    syms_premarket = []
    for f in csv_files:

        if sym_cnt%sym_cnt_mod == 1:
            print('{} SYMBOLS {:.0f}% COMPLETE'.format(idate,sym_cnt/len(csv_files)*100))
        sym_cnt = sym_cnt+1

        sym = f.split('\\')[-1].split('_')[4]
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'])

        df = df.loc[
            (df['Date'] >= m_1) & (df['Date'] <= m_2) ].reset_index().drop(
            columns='index')

        if (
            df.loc[(df['Date'] >= pm_1) & (df['Date'] <= pm_2)].empty or
            df.loc[df['Date'] >= m_2].empty
            ):
            continue

        if (
            (df['Close'][0]>close_min) & (df['Close'][0]<close_max) and
            (df['Close'].iloc[-1]/df['Close'].iloc[0]-1) > m_pct_chg and
            df.loc[df['Date'].dt.date == m_2.date()]['Volume'].sum() > m_cumvol_min and
            df.loc[df['Date'] >= m_2-datetime.timedelta(hours=1)]['Volume'].max() > m_vol_max and
            len(df.loc[(df['Date'] >= pm_1) & (df['Date'] <= pm_2)]) > pm_bars
            ):
            syms_market.append(sym)

        if (
            (df.loc[df['Date']<=pm_2]['Close'].iloc[-1]>close_min) & (df.loc[df['Date']<=pm_2]['Close'].iloc[-1]<close_max) and
            (df.loc[df['Date']<=pm_2]['Close'].iloc[-1]/df.loc[df['Date']>=pm_1]['Close'].iloc[0]) > pm_pct_chg and
            df.loc[(df['Date'] >= pm_2-datetime.timedelta(hours=1)) & (df['Date'] <= pm_2)]['Volume'].max() > pm_vol_max and
            len(df.loc[(df['Date'] >= pm_1) & (df['Date'] <= pm_2)]) > pm_bars
            ):
            syms_premarket.append(sym)

    market_watchlist[(m_2.date()).strftime('%Y-%m-%d')] = syms_market
    premarket_watchlist[(m_2.date()).strftime('%Y-%m-%d')] = syms_premarket
    print('Date loop took {:.1f} minutes'.format( (time.perf_counter()-tic_dt)/60))
    print('{} Symbols in Market Watchlist'.format(len(syms_market)))
    print('{} Symbols in Premarket Watchlist'.format(len(syms_premarket)))

with open('{}market_watchlists_{}_{}.json'.format(fol,
                    dates[0].strftime('%Y-%m-%d'),
                    dates[-1].strftime('%Y-%m-%d')), 'w') as convert_file:
    convert_file.write(json.dumps(market_watchlist))

with open('{}premarket_watchlists_{}_{}.json'.format(fol,
                    dates[0].strftime('%Y-%m-%d'),
                    dates[-1].strftime('%Y-%m-%d')),'w') as convert_file:
    convert_file.write(json.dumps(premarket_watchlist))

print('Total run time: {:.1f} hours'.format( (time.perf_counter()-tic_total)/60/60))