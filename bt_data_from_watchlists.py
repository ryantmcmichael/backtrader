# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:06:02 2022

@author: rtm
"""

#####
# STEP 3: USE THE SIMULATED WATCHLISTS TO CREATE CURATED FOLDERS OF DATA
#         FOR BACKTRADER SIMULATIONS.
#
#         These are basically filtered versions of the US-Stock data. Each
#         trading day gets a folder, and in that folder are stocks from the
#         corresponding watchlist, filtered for the date of interest.
#####

import json
import pandas as pd
import os

stock_fol = r'C:/Users/rtm/Desktop/stock_data/'
sav_fol = r'C:/Users/rtm/Desktop/stock_data/Simulated_Watchlists/'

f = open(stock_fol + 'market_watchlists_2022-01-04_2022-04-05.json')
m_data = json.load(f)

f = open(stock_fol + 'premarket_watchlists_2022-01-04_2022-04-05.json')
pm_data = json.load(f)

# Create date ranges and market/premarket times
date_start = '2021-12-28'
market_mg = pd.read_csv(stock_fol + 'daily-market-momentum-ga_2022-04-09.csv',
                        skiprows=1,
                        names=['Date','Positive','Negative','Russell2000']
                        ).dropna(axis=0,how='any')
market_mg['Date'] = pd.to_datetime(market_mg['Date']).dt.normalize()
market_mg = (market_mg.dropna(axis=0, how='any')).set_index('Date')
market_mg = market_mg.shift(1).fillna(False)
market_mg['Compare'] = ((market_mg['Positive'] > market_mg['Negative']) &
                        (market_mg['Negative'] < 40) )
dates = market_mg.loc[(market_mg.index > date_start) & market_mg['Compare']].index.date

for idate in dates[0::1]:
    print('\nReading Watchlist for {}'.format(idate))
    m_syms = m_data[idate.strftime('%Y-%m-%d')]
    pm_syms = pm_data[idate.strftime('%Y-%m-%d')]

    isExist = os.path.exists(sav_fol + 'Market/' + idate.strftime('%Y-%m-%d'))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(sav_fol + 'Market/' + idate.strftime('%Y-%m-%d'))
        print("Created new directory for Market Data")

    isExist = os.path.exists(sav_fol + 'Premarket/' + idate.strftime('%Y-%m-%d'))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(sav_fol + 'Premarket/' + idate.strftime('%Y-%m-%d'))
        print("Created new directory for Premarket Data")

    if len(m_data)==0 or len(pm_data)==0:
        continue

    for isym in m_syms:
        data = pd.read_csv(stock_fol + 'US_data/0_0_1_0_{}_2021-12-25_2022-04-04.csv'.format(isym))
        data['Date'] = pd.to_datetime(data['Date'])

        index1 = data.index[ data['Date'].dt.date == idate ][0]-500
        if index1 < 0:
            index1 = 0
        index2 = data.index[ data['Date'].dt.date == idate ][-1]

        data = data.iloc[index1:index2]

        data.to_csv(sav_fol + 'Market/' + idate.strftime('%Y-%m-%d') +
                    '/0_0_1_0_{}_{}.csv'.format(isym,idate), index=False,
                    header=True)

    print('Completed Market Data')

    for isym in pm_syms:
        data = pd.read_csv(stock_fol + 'US_data/0_0_1_0_{}_2021-12-25_2022-04-04.csv'.format(isym))
        data['Date'] = pd.to_datetime(data['Date'])

        index1 = data.index[ data['Date'].dt.date == idate ][0]-500
        if index1 < 0:
            index1 = 0
        index2 = data.index[ data['Date'].dt.date == idate ][-1]

        data = data.iloc[index1:index2]

        data.to_csv(sav_fol + 'Premarket/' + idate.strftime('%Y-%m-%d') +
                    '/0_0_1_0_{}_{}.csv'.format(isym,idate), index=False,
                    header=True)

    print('Completed Premarket Data')