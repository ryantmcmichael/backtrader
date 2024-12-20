# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:09:04 2022

@author: rtm
"""

######
# STEP 1: DOWNLOAD ALL MINUTE-DATA FOR ALL US STOCK TICKERS
######

from eodhd import APIClient
import os
import pandas as pd
import datetime
import time
import yfinance as yf

# Generate the weekly picks by downloading the .html version of the webpage
# and copy-pasting into an excel file.
fol = r'C:/Users/ryant/Documents/Stock_Market/Python/universe_data/VM Weekly Breakout'
vm_list = pd.read_csv(fol + '/JD_RAW_MDA_Weekly_Picks.csv')

# load the key from the enviroment variables
# api_key = os.environ['API_EOD']
API_KEY = '5e98828c864a88.41999009'
api = APIClient(API_KEY)

fail_dates=[]
sector_list = []
fail_sector = []

for dt in vm_list['Date'].unique():
    print(dt)

    d1 = datetime.datetime.strptime(dt + ' 06:00', '%m/%d/%Y %H:%M')
    d1 = d1 - datetime.timedelta(days=7)

    unixtime1 = time.mktime(d1.timetuple())
    d2 = d1 + datetime.timedelta(hours=180)
    unixtime2 = time.mktime(d2.timetuple())

    svpath = fol + '/data/' + d2.date().strftime('%Y-%m-%d')
    isExist = os.path.exists(svpath)
    if not isExist:
        os.makedirs(svpath)

    valid_days = []
    for ticker in vm_list.loc[vm_list['Date']==dt,'Ticker']:

        time.sleep(0.5)
        resp = api.get_intraday_historical_data(
            symbol = ticker,
            from_unix_time = unixtime1, to_unix_time = unixtime2,
            interval='1m')

        if not resp:
            print('.... {} FAILED'.format(ticker))
            continue

        df = pd.DataFrame(resp)
        df['Datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        df = df[['Datetime','open','high','low','close','volume']]

        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        df = (df.set_index('Datetime')
          .between_time('06:30', '13:00')
          .reset_index())

        valid_days.append(len(df['Datetime'].dt.date.unique()))

        df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df.to_csv(svpath + '/0_0_1_0_' + ticker + '_' +
                    d1.date().strftime('%Y-%m-%d') + '_' +
                    d2.date().strftime('%Y-%m-%d') +
                    '.csv', header=True, index=False)

        # Fetch the sector, append list
        # Need ERROR CATCHING on the sector
        tickerdata = yf.Ticker(ticker)
        if 'sector' in tickerdata.info:
            sector_list.append([ticker,tickerdata.info['sector']])
        else:
            print('**** {} Sector not found'.format(ticker))
            fail_sector.append([ticker,'No sector data in Yahoo'])

        print('.... {}'.format(ticker))

    if valid_days[0]<6 and len(set(valid_days))==1:
        fail_dates.append([dt,'Holiday?'])
        print('May have contained a HOLIDAY')
    elif len(set(valid_days))>1:
        fail_dates.append([dt,'Bad Ticker(s)?'])
        print('May have contained bad ticker(s)!')
    elif len(valid_days)<4:
        fail_dates.append([dt,'Bad Ticker(s)?'])
        print('May have contained bad ticker(s)!')

df_fail = pd.DataFrame(fail_dates,columns=['Date','Reason'])
df_fail.to_csv(fol + '/date_failures.csv',header=True,index=False)

for x in range(len(sector_list)):
    if sector_list[x][1] == 'Financial Services':
        sector_list[x][1] = 'Financial'
df_sector = pd.DataFrame(sector_list, columns=['Ticker','Sector'])
df_sector.to_csv(fol + '/ticker_sector_yahoo.csv',header=True,index=False)

df_fail_sector = pd.DataFrame(fail_sector,columns=['Ticker','Sector'])
df_fail_sector.to_csv(fol + '/sector_failures_yahoo.csv',header=True,index=False)