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
limit_pct = 1.25

for buy_bound in ['expected']:

    # Initialize metrics
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


        print(' ')
        #print('\n********** INITIATING BACKTEST {} **'.format(fol))

        # Initialize metrics
        pnl_tickers = []

        # loop over each week of data (each folder)
        for fpath in csv_files:
            file = os.path.basename(fpath)
            ticker = file.split('_')[4]
            sector = ticker_sectors.loc[ticker].Sector

            # Do not enter trade if Sector Neg > Pos
            if (s.shift(1)[s.index >= buydate].iloc[0][sector + ' Neg'] >=
                s.shift(1)[s.index >= buydate].iloc[0][sector + ' Pos']):
                num_skip = num_skip + 1
                print('**** Avoided trade due to Sector Gauge Neg > Pos')
                continue

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
                print('**** No data on sell date')
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

            d[['Expected Stop','Expected Limit','Worst Stop','Worst Limit',
               'Best Stop','Best Limit']] = [
                (dtest['p(x)']*dtest.index).sum()*stop_pct,
                (dtest['p(x)']*dtest.index).sum()*limit_pct,
                pbuy_list['high'].max()*stop_pct,
                pbuy_list['high'].max()*limit_pct,
                pbuy_list['low'].min()*stop_pct,
                pbuy_list['low'].min()*limit_pct]


            d[['Expected Limit','Worst Limit','Best Limit']] = d[[
                'Expected Limit','Worst Limit','Best Limit']].gt(d['high'],axis='rows')
            d[['Expected Stop','Worst Stop','Best Stop']] = d[[
                'Expected Stop','Worst Stop','Best Stop']].lt(d['low'],axis='rows')

            actual_buydatetime = pbuy_list.index[-1]

            ################## LIMIT AND STOP LOSSES ########################
            # Determine the time of the first stop trigger or limit trigger
            # Note: They may never occur

            # WORST Case
            if not d[~(d['Worst Stop'] & d['Worst Limit'])].empty:
                worst_stop_limit = d[~(d['Worst Stop'] & d['Worst Limit'])].index[0]
            # If there's not a stop or limit trigger, then set it to be EOW
            else:
                worst_stop_limit = eow

            # BEST Case
            if not d[~(d['Best Stop'] & d['Best Limit'])].empty:
                best_stop_limit = d[~(d['Best Stop'] & d['Best Limit'])].index[0]
            # If there's not a stop or limit trigger, then set it to be EOW
            else:
                best_stop_limit = eow

            # EXPECTED Case
            if not d[~(d['Expected Stop'] & d['Expected Limit'])].empty:
                expected_stop_limit = d[~(d['Expected Stop'] & d['Expected Limit'])].index[0]
            # If there's not a stop or limit trigger, then set it to be EOW
            else:
                expected_stop_limit = eow


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


            # TODO: HERE'S WHERE I LEFT OFF...NEED TO AGGREGATE THE SELL TIMES
            # FOR EACH CASE EXPECTED, BEST, WORST

            # Compare the timestamp for the first stop, the first limit,
            # the first market and sp500 gauge signals, and eow
            # The earliest trigger defines the termination of the trade
            exit_dict = {'SellTime':
                       [first_mrkt_neg40, first_mrkt_negpos,
                        first_sp500_neg40, first_sp500_negpos,
                        first_s_neg40, first_s_negpos,
                        first_limit, first_stop],
                       'Trigger':['Mrkt Neg > 40', 'Mrkt Neg > Pos',
                               'SP500 Neg > 40','SP500 Neg > Pos',
                               sector + ' Neg > 40', sector + ' Neg > Pos',
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
            print('{}: Bought @ {:.2f} on {}, {} @ {:.2f} on {}, PNL: {:.2f}%'.format(ticker,pbuy,actual_buydatetime,trigger,psell,selltime,pnl))

        if len(pnl_tickers) == 0:
            continue
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

