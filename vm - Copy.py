# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 21:25:06 2022

@author: rtm
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import datetime
import os.path
import sys
import glob
import merit_vm_indicator
import time
import pandas as pd
import seaborn as sns
import numpy as np

class FixedValueSizer(bt.Sizer):
    params = (('value',1000),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        size = int(self.p.value/data.close[0])
        return size

class MinuteData(bt.feeds.GenericCSVData):
    params = (
        ('nullvalue', float('NaN')),
        ('headers',True),
        ('datetime', 0),
        ('time',-1),
        ('high', 1),
        ('low', 2),
        ('open', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
        ('dtformat','%Y-%m-%d %H:%M:%S'),
    )

class TestStrategy(bt.Strategy):

    params = dict(
        maxstop = 0.04,
        smashort = 4,
        smalong = 20,
        smaexit = 9,
        rsithresh = 50,
        mfithresh = 50,
        uothresh = 0.5,
        atrfactor = 1,
        tradestart = datetime.time(6,30),
        tradestop = datetime.time(7,30),
        printlog = False,
        )

    def log(self, txt, dt=None, doprint=False):
        if self.p.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime()
            print('{}, {}'.format(dt, txt))




    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - do nothing
            return
    
        dt, dn = self.datetime.datetime(), order.data._name.split('_')[0]
        self.log('{} {} Order {} Status {}'.format(
            dt, dn, order.ref, order.getstatusname()) )
    
        # Check if order has been completed
        if order.status in [order.Completed]:

            if order.isbuy():
                self.ntrades = self.ntrades+1
                self.log('{} BUY EXECUTED, Price: {:.2f}, PnL: {:.2f}'.format(
                    dn, order.executed.price, order.executed.pnl),
                    doprint=False)
                # self.buyprice[d] = order.executed.price

            elif order.issell():
                self.ntrades = self.ntrades-1
                self.log('{} SELL EXECUTED, Price: {:.2f}, PnL: {:.2f}'.format(
                    dn, order.executed.price, order.executed.pnl),
                    doprint=False)
    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('----- ORDER CANCELED/MARGIN/REJECTED')

        whichord = ['MARKET ORDER', 'STOP ORDER', 'MERIT EXIT', '???']
        if not order.alive(): # not alive - nullify
            dorders = self.o[order.data]
            idx = dorders.index(order)
            dorders[idx] = None
            # if idx>2:
            #     self.log('{}, {}'.format(idx,dn),doprint=True)
            self.log('-- No longer alive {}'.format(whichord[idx]),
                     doprint=False)

        if not any(dorders):
            dorders[:] = [] # empty list - New orders allowed




    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('Trade PnL, {:.2f}, ACCOUNT, {:.2f}'.format(
            trade.pnl, self.broker.getvalue()), doprint=False)




    def __init__(self):
        self.ntrades = 0
        self.o = dict()
        self.holding = dict()

        # Create a dictionary to hold all indicators
        self.inds = dict()
        # Loop over each datafeed to make an indicator for each feed

        for i, d in enumerate(self.datas):
            # Create dict to hold all datafeeds
            self.inds[d] = dict()

            # Create SMA Short indicator on all datafeeds
            self.inds[d]['smashort'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.p.smashort)

            # Create SMA Long indicator on all datafeeds
            self.inds[d]['smalong'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.p.smalong)

            # Create SMA Exit indicator on all datafeeds
            self.inds[d]['smaexit'] = bt.indicators.SimpleMovingAverage(
                d.close, period=self.p.smaexit)

            self.inds[d]['atr'] = bt.indicators.ATR(d, plot=False)

            # Create RSI for Plotting
            # self.inds[d]['rsi'] = bt.indicators.RSI(d.close, safediv=True,
            #                                         safehigh=100, safelow=0)

            # Create DMI for Plotting
            # self.inds[d]['dmi'] = bt.indicators.DMI(d)

            # Create MFI for Plotting
            # self.inds[d]['mfi'] = bt.indicators.MoneyFlowIndicator(d)

            # Create Ultimate Oscillator for Plotting
            # self.inds[d]['uo'] = bt.indicators.UltimateOscillator(d)

            # The Merit Function
            self.inds[d]['merit'] = merit_vm_indicator.meritsimple(d,
                        smashort=self.p.smashort, smalong=self.p.smalong,
                        rsithresh=self.p.rsithresh, mfithresh=self.p.mfithresh,
                        uothresh=self.p.uothresh, atrfactor=self.p.atrfactor)


    def next(self):
        for i, d in enumerate(self.datas):
            dt, dn, dtrade = (self.datetime.datetime(), d._name.split('_')[0],
                datetime.datetime.strptime(d._name.split('_')[1],'%Y-%m-%d').date())
            self.log('BROKER: {}'.format(self.broker.getvalue()),doprint=False)
            pos = self.getposition(d).size

            # Buy Logic
            # no position / no orders & correct date
            if (not pos and not self.o.get(d,None) and
                self.ntrades < 4 and
                dt.date() == dtrade and
                dt.time() >= self.p.tradestart and
                dt.time() < self.p.tradestop):

                if (self.inds[d]['merit'].l.m_up[-1] < 5 and
                    self.inds[d]['merit'].l.m_up[0] == 5):

                    pstp = max(d.close[0]-self.inds[d]['atr']*self.p.atrfactor,
                               self.p.maxstop)
                    o1 = self.buy(data=d, exectype=bt.Order.Market,
                                  transmit=False)
                    o2 = self.sell(data=d, exectype=bt.Order.Stop,
                                   price=pstp, size=o1.size,
                                   transmit=True, parent=o1)
                    self.o[d] = [o1, o2]
                    self.holding[d] = 0

                    self.log('BUY, {}, {:.2f}, {:.2f}'.format(dn, d.close[0], pstp),
                                     doprint=False)

            # Sell Logic
            elif pos:
                self.holding[d] = self.holding[d]+1

                if (d.close[0] < self.inds[d]['smaexit'] or
                    self.inds[d]['merit'].l.m_down[0] < 3):

                    # Cancel all open orders (e.g. Stop-Order)
                    for x in self.o[d]:
                        self.cancel(x)

                    # Create new close order
                    o = self.close(data=d)
                    self.o[d].append(o) # manual order to list of orders
                    self.log('MERIT SELL & CANCEL STOP, {}, {}'.format(dn, self.o[d]),
                             doprint=False)

    def stop(self):
        print('\nShort SMA: {}, '.format(self.p.smashort) +
                 'Long SMA: {}, '.format(self.p.smalong) +
                 'Exit SMA: {}, '.format(self.p.smaexit) +
                 'ATR Factor: {}, '.format(self.p.atrfactor) +
                 'Ending Value: ${:,.2f}'.format(self.broker.getvalue()))


if __name__ == '__main__':
    optim = False
    plotting = False
    plotreturn = False
    startcash = 10000
    size = 1000

    # Create date ranges and market/premarket times
    date_start = '2022-01-01'
    date_stop = '2022-04-04'

    dates = pd.date_range(
    start=datetime.datetime.strptime(date_start, '%Y-%m-%d'),
    end=datetime.datetime.strptime(date_stop, '%Y-%m-%d'),
    freq='B').date

    folder = r'C:/Users/rtm/Desktop/stock_data/'
    # Create date ranges and market/premarket times
    date_start = '2021-12-27'
    market_mg = pd.read_csv(folder + 'daily-market-momentum-ga_2022-04-09.csv',
                            skiprows=1,
                            names=['Date','Positive','Negative','Russell2000']
                            ).dropna(axis=0,how='any')
    market_mg['Date'] = pd.to_datetime(market_mg['Date']).dt.normalize()
    market_mg = (market_mg.dropna(axis=0, how='any')).set_index('Date')
    market_mg = market_mg.shift(1).fillna(False)
    market_mg['Compare'] = ((market_mg['Positive'] > market_mg['Negative']) &
                            (market_mg['Negative'] < 40) )
    dates = market_mg.loc[(market_mg.index > date_start) & market_mg['Compare']].index.date

    final_val = {}
    final_val[
        (datetime.datetime.strptime(date_start,'%Y-%m-%d') -
        datetime.timedelta(days=1)).date()] = startcash

    for idate in dates[0::1]:
        print('\n=========== BEGINNING SIMULATION FOR {}'.format(idate))
        t0 = time.perf_counter()

        # Determine whether it's a day to trade
        path = folder + 'Simulated_Watchlists/{}/*.csv'.format(idate.strftime('%Y-%m-%d'))
        if len(glob.glob(path)) == 0:
            continue

        # Instantiate Cerebro
        cerebro = bt.Cerebro(maxcpus=1, runonce=False, stdstats=True)

        # Add Data to Cerebro
        for datapath in glob.glob(path):
            nm = '_'.join(list(np.array(
                datapath.split('\\')[-1].split('_'))[[4, 5]])).replace('.csv','')
            #print('LOADING {}'.format(nm))
            data = MinuteData(dataname=datapath,
                              timeframe=bt.TimeFrame.Minutes, plot=plotting)
            # cerebro.adddata(data, name=nm)
            cerebro.resampledata(data, name=nm,
                                 timeframe=bt.TimeFrame.Minutes, compression=5)
    
        # Add Strategy
        if optim == True:
            strats = cerebro.optstrategy(
                TestStrategy,
                smashort=[4],
                smalong=[20],
                smaexit=[4],
                atrfactor=[0.4,0.5,0.6],
                maxstop=[0.04])
        else:
            cerebro.addstrategy(TestStrategy, smashort=4, smalong=20, smaexit=4,
                                atrfactor=0.5, maxstop=0.04)
            # Add Analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
    
        # Set Cash, Sizer, Commission
        cerebro.broker.setcash(startcash)
        cerebro.addsizer(FixedValueSizer, value=size) # Add sizer
        cerebro.broker.setcommission(commission=0) #divide by 100, in decimal

        result = cerebro.run()
    
        if optim == False:
            #print('');print('Final Portfolio Value: ${:,.2f}'.format(cerebro.broker.getvalue()))
            final_val[idate] = cerebro.broker.getvalue()
            result = result[0]
            trans_dict = result.analyzers.transactions.get_analysis();tradesym=[]
            draw_dict = result.analyzers.drawdown.get_analysis()
            trade_dict = result.analyzers.tradeanalyzer.get_analysis()
        
            if len(trade_dict) > 1:
                for x in trans_dict.keys():
                    for j in range(len(trans_dict[x])):
                        tradesym.append(trans_dict[x][j][3].split('_')[0])
                print('');print('Traded Symbols: {}'.format(list(set(tradesym))))
                print('');print('Max Money Drawdown: {:,.2f}'.format(draw_dict.moneydown))
                print('');print('Win Percentage: {:.1f}%'.format(
                    trade_dict.won.total/(trade_dict.won.total+trade_dict.lost.total)*100) )
                print('Total Trades: {}'.format(
                    trade_dict.total.total) )
            
                print('');print(
                    pd.concat([pd.DataFrame.from_dict(
                            trade_dict.won.pnl,orient='index',
                                columns=['Win PnL']).applymap("{0:,.2f}".format),
                        pd.DataFrame.from_dict(
                            trade_dict.lost.pnl,orient='index',
                                columns=['Lost PnL']).applymap("{0:,.2f}".format)],
                        axis=1))
                print(
                    pd.concat([pd.DataFrame.from_dict(
                            trade_dict.len.won,orient='index',
                                columns=['Win Length']).applymap("{:.1f}".format),
                        pd.DataFrame.from_dict(
                            trade_dict.len.lost,orient='index',
                                columns=['Lost Length']).applymap("{:.1f}".format)],
                        axis=1))

            if plotreturn == True:
                fignum=1
                fig=cerebro.plot(userfig=fignum,iplot=False,
                      fmt_x_ticks = '%Y-%m-%d %H:%M',style='candle',trend=True,
                      barup='green',bardown='red',
                      rowsmajor=2,rowsminor=1,
                      voloverlay=True,volup='gray',voldown='gray',voltrans=0.2,
                      subtxtsize=6,legendind=True,
                      hlinescolor=[0,0,1],hlinesstyle='-',hlineswidth=0.75,
                      tickrotation=30)
                fig[0][0].subplots_adjust(left=0.01,right=0.93,
                                      top=0.99,bottom=0.15)

        t1 = time.perf_counter()
        print('\nBacktest Completed in {:.2f} seconds'.format(t1-t0))

    agg_result = pd.DataFrame.from_dict(final_val,orient='index',
                           columns=['Backtest Val'])

    agg_result['Daily PnL'] = agg_result['Backtest Val']-startcash
    agg_result['Daily % PnL'] = (agg_result['Backtest Val']/startcash-1)*100
    agg_result['Cumulative PnL'] = agg_result['Daily PnL'].cumsum()
    agg_result['Portfolio Value'] = agg_result['Daily PnL'].cumsum()+startcash
    agg_result.to_csv('agg_result.csv', header=True, index_label='Date')

    d = pd.read_csv('agg_result.csv')
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.set_index('Date')

    win=d['Daily PnL']>=0
    lose=d['Daily PnL']<0

    bench_sym = 'TQQQ'
    bench = pd.read_csv(folder +
            'US_data/0_0_1_0_{}_2021-12-25_2022-04-04.csv'.format(bench_sym))
    bench['Date'] = pd.to_datetime(bench['Date'])
    bench = bench.loc[ (bench['Date'].dt.time <= datetime.time(13,0)) &
              (bench['Date'].dt.time >= datetime.time(6,30))]
    bench = bench.set_index('Date')
    bench_rs = bench.resample('1D').agg({
        'High':'max',
        'Low':'min',
        'Open':'first',
        'Close':'last'}).dropna(how='any')
    bench_rs = bench_rs.loc[bench_rs.index.isin(d.index)]
    bench_rs['PnL'] = (bench_rs['Close']/bench_rs['Open']-1)*startcash
    bench_rs['CumSum'] = bench_rs['PnL'].cumsum()+startcash

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Date', color='black')
    ax1.set_ylabel('Port Value', color='black')
    ax1.tick_params(axis='x', labelcolor='black', labelrotation=60)
    ax1.tick_params(axis='y', labelcolor='black')

    line_portval = ax1.plot(d.index, d['Portfolio Value'],
                              color='black')
    scatter_win = ax1.scatter(d.index[win], d[win]['Portfolio Value'],
                              color='green')
    scatter_lose = ax1.scatter(d.index[lose], d[lose]['Portfolio Value'],
                          color='red')
    line_benchval = ax1.plot(bench_rs.index, bench_rs['CumSum'],
                              'b-o')
    plt.grid('on', linestyle='--')
    plt.show()
    plt.tight_layout()