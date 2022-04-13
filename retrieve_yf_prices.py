# -*- coding: utf-8 -*-

import os
import glob
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

#financialmodelingprep

fol = r'C:/Users/rtm/Desktop/TOS_Watchlists/Market/'

# use glob to get all the csv files 
# in the folder
csv_files = glob.glob(os.path.join(fol, "*.csv"))

# loop over the list of csv files
for f in csv_files:
    tradedate = '-'.join(f.split("\\")[-1].split('-')[0:3])
    sdate = (datetime.datetime.strptime(tradedate,'%Y-%m-%d') -
             datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    edate = (datetime.datetime.strptime(tradedate,'%Y-%m-%d') +
             datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    # read the csv file
    ticks = pd.read_csv(f,skiprows=3)['Symbol'].tolist()

    data = yf.download(ticks, start=sdate,
                       end=edate, interval='1m', group_by='ticker')

    isExist = os.path.exists(fol + tradedate)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(fol + tradedate)
        print("Created new directory")

    for j in ticks:
        ind = ( data[j][['High','Low','Open','Close','Volume']]
           .reset_index().rename(columns={'Datetime':'Date'}) )
    
        ind['Date'] = (ind['Date'].dt.tz_convert(
                        'America/Los_Angeles').dt.tz_localize(None) )
    
        ind = ind.fillna(method='bfill')
        ind = ind.fillna(method='ffill')

        # print(ind.iloc[-1]['Close'])

        # ind = ind.append(pd.DataFrame(
        #     [[datetime.strptime('2022-04-02 06:30','%Y-%m-%d %H:%M'),0, 0, 0, 0,0]],
        #     columns=['Date','High','Low','Open','Close','Volume']),
        #     ignore_index=True)

        ind['High'] = ind['High'].replace(to_replace=0,method='bfill')
        ind['High'] = ind['High'].replace(to_replace=0,method='ffill')

        ind['Low'] = ind['Low'].replace(to_replace=0,method='bfill')
        ind['Low'] = ind['Low'].replace(to_replace=0,method='ffill')

        ind['Open'] = ind['Open'].replace(to_replace=0,method='bfill')
        ind['Open'] = ind['Open'].replace(to_replace=0,method='ffill')

        ind['Close'] = ind['Close'].replace(to_replace=0,method='bfill')
        ind['Close'] = ind['Close'].replace(to_replace=0,method='ffill')

        ind['Volume'] = ind['Volume'].replace(to_replace=0,method='bfill')
        ind['Volume'] = ind['Volume'].replace(to_replace=0,method='ffill')

        if (ind==0).sum().sum() > 0:
            print(j)
            print((ind==0).sum().sum())

        ind.to_csv(fol + tradedate + '/0_0_1_0_' + j + '_' + sdate + '_' + tradedate + '.csv',
                   header=True, index=False)