# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:09:04 2022

@author: rtm
"""

######
# STEP 1: DOWNLOAD ALL MINUTE-DATA FOR ALL US STOCK TICKERS
######

from eod import EodHistoricalData
from random import randint
import pandas as pd
import datetime

fol = r'C:/Users/rtm/Desktop/stock_data/US_data'

# load the key from the enviroment variables
# api_key = os.environ['API_EOD']
api_key = '5e98828c864a88.41999009'

# Create the instance 
client = EodHistoricalData(api_key)

'''
# US Exchanges
 'BATS',
 'NASDAQ',
 'NMFQS',
 'NYSE',
 'NYSE ARCA',
 'NYSE MKT',
 'OTC',
 'OTCBB',
 'OTCCE',
 'OTCGREY',
 'OTCMKTS',
 'OTCQB',
 'OTCQX',
 'PINK'
'''
exchanges = ['BATS','NASDAQ','NYSE','NYSE MKT','NYSE ARCA','OTC']
us_ticks = client.get_exchange_symbols(exchange='US')
us_ticks = pd.DataFrame.from_dict(us_ticks)
ticks = us_ticks.loc[us_ticks['Exchange'].isin(exchanges)]['Code'].tolist()

# Download last 100 days of ticker data
edatetime = datetime.datetime.now()
sdatetime = edatetime - datetime.timedelta(days=100)

skipped = {'Skipped':[]}
counter=0
for j in ticks:
    counter = counter + 1
    print('Trying {}...{:.1f}% complete'.format(j,counter/len(ticks)*100))

    data = pd.DataFrame.from_dict(
    client.get_prices_intraday(j, interval='1m', order='a',
        from_=datetime.datetime.timestamp(sdatetime),
        to=datetime.datetime.timestamp(edatetime)))

    if data.empty:
        skipped['Skipped'].append(j)
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

    ind.to_csv(fol + '/0_0_1_0_' + j + '_' +
                sdatetime.date().strftime('%Y-%m-%d') + '_' +
                edatetime.date().strftime('%Y-%m-%d') +
                '.csv', header=True, index=False)

skipped = pd.DataFrame.from_dict(skipped)
skipped.to_csv(fol + '/skipped/skipped_tickers.csv',header=True,index=False)
print('\nCOMPLETED...Skipped {} tickers'.format(len(skipped)))