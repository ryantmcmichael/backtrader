# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:09:04 2022

@author: rtm
"""

import os
import glob
from eod import EodHistoricalData
from random import randint
import pandas as pd
import datetime

# load the key from the enviroment variables
# api_key = os.environ['API_EOD']
api_key = '5e98828c864a88.41999009'

# Create the instance 
client = EodHistoricalData(api_key)

fol = r'C:/Users/rtm/Desktop/TOS_Watchlists/Market/'

# use glob to get all the csv files 
# in the folder
csv_files = glob.glob(os.path.join(fol, "*.csv"))

# loop over the list of csv files
for f in csv_files:

    tradedate = f.split("\\")[-1].split('-')
    tradedate_file = '-'.join(tradedate[0:3])
    sdatetime = (datetime.datetime(int(tradedate[0]),
                                   int(tradedate[1]),
                                   int(tradedate[2]),0,0) -
            datetime.timedelta(days=1))
    edatetime = (datetime.datetime(int(tradedate[0]),
                                   int(tradedate[1]),
                                   int(tradedate[2]),0,0) +
            datetime.timedelta(days=1))

    # read the csv file
    ticks = pd.read_csv(f,skiprows=3)['Symbol'].tolist()

    isExist = os.path.exists(fol + tradedate_file)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(fol + tradedate_file)
        print("Created new directory")

    for j in ticks:
        print('Trying {}...'.format(j))
        data = pd.DataFrame.from_dict(
        client.get_prices_intraday(j, interval='1m', order='a',
            from_=datetime.datetime.timestamp(sdatetime),
            to=datetime.datetime.timestamp(edatetime)))

        if data.empty:
            print('--------- No data available for {}'.format(j))
            continue

        ind = ( data[['datetime','high','low','open','close','volume']]
           .reset_index().rename(columns={'datetime':'Date',
                                          'high':'High',
                                          'low':'Low',
                                          'open':'Open',
                                          'close':'Close',
                                          'volume':'Volume'}) )
    
        ind['Date'] = pd.to_datetime(ind['Date'],utc=True)
        ind['Date'] = (ind['Date'].dt.tz_convert(
                        'America/Los_Angeles').dt.tz_localize(None) )

        ind = ind.drop(columns='index')

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

        ind.to_csv(fol + tradedate_file + '/0_0_1_0_' + j + '_' +
                   sdatetime.date().strftime('%Y-%m-%d') + '_' +
                   tradedate_file + '.csv', header=True, index=False)