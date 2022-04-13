# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 08:35:21 2022

@author: rtm
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv('agg_result.csv')
d['Date'] = pd.to_datetime(d['Date'])
d = d.set_index('Date')
d['outcome'] = (d['Daily PnL']>=0)*1
win=d['Daily PnL']>=0
lose=d['Daily PnL']<0

fol = r'C:/Users/rtm/Desktop/stock_data/'
market_mg = pd.read_csv(fol + 'daily-market-momentum-ga_2022-04-09.csv',
                        skiprows=1,
                        names=['Date','Positive','Negative','Russell2000']
                        ).dropna(axis=0,how='any')
market_mg['Date'] = pd.to_datetime(market_mg['Date']).dt.normalize()
market_mg['Russell2000 Chg'] = market_mg['Russell2000'].pct_change()
market_mg = (market_mg.loc[market_mg['Date'].isin(d.index)]
             .dropna(axis=0, how='any')).set_index('Date')

# Correlations
market_mg = market_mg.shift(1).fillna(False)
market_mg['Compare'] = ((market_mg['Positive'] > market_mg['Negative']) &
                        (market_mg['Negative'] < 40) )
days_market = len(d[market_mg['Compare']]['Daily PnL'])
pnl_market = d[market_mg['Compare']]['Daily PnL'].sum()
max_gain_market = d[market_mg['Compare']]['Daily PnL'].max()
min_gain_market = d[market_mg['Compare']]['Daily PnL'].min()

d[market_mg['Compare']]['Daily PnL'].cumsum().plot(grid=True, style='.-',label='Market',legend=True)

# sp500_mg = pd.read_csv(fol + 'daily-sp-500-momentum-ga_2022-04-09.csv',
#                         skiprows=1,
#                         names=['Date','Positive','Negative','SP500']
#                         ).dropna(axis=0,how='any')
# sp500_mg['Date'] = pd.to_datetime(sp500_mg['Date']).dt.normalize()
# sp500_mg['SP500 Chg'] = sp500_mg['SP500'].pct_change()
# sp500_mg = (sp500_mg.loc[sp500_mg['Date'].isin(d.index) ]
#               .dropna(axis=0, how='any')).set_index('Date')

# sp500_mg = sp500_mg.shift(1).fillna(False)
# sp500_mg['Compare'] = ((sp500_mg['Positive'] > sp500_mg['Negative']) &
#                         (sp500_mg['Negative'] < 40) )
# days_sp500 = len(d[sp500_mg['Compare']]['Daily PnL'])
# pnl_sp500 = d[sp500_mg['Compare']]['Daily PnL'].sum()

# d[sp500_mg['Compare']]['Daily PnL'].cumsum().plot(grid=True, style='.-',label='SP500',legend=True)

'''
fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

ax1.set_xlabel('Date', color='black')
ax1.set_ylabel('MG', color='black')
# ax2.set_ylabel('Daily PnL', color='C0')
ax1.tick_params(axis='x', labelcolor='black', labelrotation=60)
ax1.tick_params(axis='y', labelcolor='black')
# ax2.tick_params(axis='y', labelcolor='C0')

w = 0.3
xdt = date2num(market_mg.index)
bar_market_spread = ax1.bar(xdt+w/2, market_mg['Compare'],
                       label='Market Spread', color='black', alpha=0.5, width=w)
# bar_sp500_spread = ax1.bar(xdt+3*w/2, sp500_mg['Compare'],
#                        label='SP500 Spread',color='blue', alpha=0.5, width=w)
ax1.xaxis_date()

scatter_neg_pnl = ax1.scatter(d.index[lose], d[lose]['Daily PnL'], label='Daily PnL',
                      color='red')
scatter_pos_pnl = ax1.scatter(d.index[win], d[win]['Daily PnL'], label='Daily PnL',
                      color='green')

plt.grid('on', linestyle='--')

plt.show()
plt.tight_layout()
'''