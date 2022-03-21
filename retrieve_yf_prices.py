# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd

# Enter the tickers for the past week (to retrieve from yfinance)
a= {'Date':[
        ['2022-03-14','2022-03-15'],
        ['2022-03-15','2022-03-16'],
        ['2022-03-16','2022-03-17'],
        ['2022-03-18','2022-03-19']
        ],
    'Tick':[
        ['SQQQ','MRNA'],
        ['AAL','KAVL','MULN','SQQQ'],
        ['IMPP','PLTR','SOFI'],
        ['PIK','NIO','MULN','TQQQ','GROM']
        ]
    }

fol = (r'C:/Users/rtm/Documents/Personal/Stock_Market/' +
           'Python/universe_data/sp500/stock_data/yf/')

for i in range(len(a['Date'])):

    sdate = a['Date'][i][0]
    edate = a['Date'][i][1]
    ticks = a['Tick'][i]

    data = yf.download(ticks, start=sdate,
                       end=edate, interval='1m', group_by='ticker')

    for j in ticks:
        ind = ( data[j][['High','Low','Open','Close','Volume']]
           .reset_index().rename(columns={'Datetime':'Date'}) )
    
        ind['Date'] = (ind['Date'].dt.tz_convert(
                        'America/Los_Angeles').dt.tz_localize(None) )
    
        ind = ind.fillna(method='bfill')
    
        ind.to_csv(fol + '0_0_1_0_' + j + '_' + sdate + '_' + edate + '.csv',
                   header=True, index=False)