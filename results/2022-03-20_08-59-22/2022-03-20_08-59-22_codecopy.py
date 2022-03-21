import backtrader as bt
import datetime
import sys
import os
import shutil
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as plt_lines
import pickle
import merit

'''
MAJOR UPDATES IN THIS RELEASE:
 * Need to clean up the merit function
 * Need to figure out why the benchmark isn't working
'''

ERASE_LINE = '\x1b[2K' # erase line command

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
except:
    pass

class FixedPctSizer(bt.Sizer):
    params = (('pct',0.25),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        size = int((self.p.pct*cash)/data.close[0])
        return size

class FixedValueSizer(bt.Sizer):
    params = (('value',1000),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        size = int(self.p.value/data.close[0])
        return size

class IQFeedMinuteData(bt.feeds.GenericCSVData):
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
    
class IQFeedDailyData(bt.feeds.GenericCSVData):
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
        ('dtformat','%Y-%m-%d')
    )

class SMAshort(bt.indicators.SimpleMovingAverage):
    plotlines = dict(
        sma=dict(color='black',ls='-',lw=1.0)
        )

class SMAlong(bt.indicators.SimpleMovingAverage):
    plotlines = dict(
        sma=dict(color='blue')
        )

# Create a Strategy
class Strat1(bt.Strategy):
    # Add a parameter for the strategy
    params = (
    ('smashort',9),
    ('smalong', 20),
    ('printLog',True),
    ('logfile','')
    )

    def __init__(self):
        self.holding = dict()  # holding periods per data
        self.mb = dict()
        self.dataclose = dict() # self.datas[i].close[0]
        self.set_tradehistory(True) # Turn on trade history
        self.cnt = 0
        self.ntrades = 0
        self.o = dict()
        self.filter_buy = pd.DataFrame(columns=['Data','Merit'])
        
        self.inds = dict()
        for i, d in enumerate(self.datas):
            if(d._name.split('_')[0]=='SPY'):
                continue

            self.inds[d] = dict()


            # Be sure to include "d.close" at the beginning of an indicator
            # Add Merit Function BUY
            self.inds[d]['merit'] = merit.meritsimple(d,
                                        smashort = self.p.smashort,
                                        smalong = self.p.smalong)
            
            # Add Short SMA
            self.inds[d]['smashort'] = SMAshort(
                d.close, period=self.params.smashort)
            
            # Add Long SMA
            self.inds[d]['smalong'] = SMAlong(
                d.close, period=self.params.smalong)
            
            # Add Relative Strength Index
            self.inds[d]['rsi'] = bt.indicators.RSI(d.close)
            self.inds[d]['rsi'].plotinfo.plothlines=[30,50,70]
            
            self.o[d] = dict() # orders per data 
                               # (main,stop,limit,manual-close)

    
    def prenext(self):
        self.next()
        
    # "Next" defines what we do on each subsequent tick
    def next(self):
        dt = self.datetime.date()

        dict_list = []
        for i, d in enumerate(self.datas):
            if len(self.datas[i])>2: # Check that there is data loaded
                if(d._name.split('_')[0]=='SPY'):
                    continue

                pos = self.getposition(d).size
                # if IN a position, do the following
                if (pos and not self.o.get(d, None)):
                    # Close out if ticker is about to end (introspect bars)
                    if len(self.datas[i])+2 >= self.datas[i].buflen():
                        self.o[d] = [self.close(data=d,valid=None)]
                        # self.holding[d] = 0

                    # Close out if indicator signals
                    # Note: Could force to hold for at least 1 day
                    elif (self.inds[d]['smashort'] < self.inds[d]['smalong']):
                        self.o[d] = [self.close(data=d,valid=None)]
                        # self.holding[d] = 0
                        
                # if NOT IN a position and no pending orders
                elif (not pos and not self.o.get(d, None) 
                        and len(self.datas[i])+3 < self.datas[i].buflen()
                        and self.ntrades < 10
                        and self.inds[d]['smashort'] > self.inds[d]['smalong']):
                    # Create dictionary of stocks I might buy
                    dict_list.append({
                        'Data':d,
                        'Merit':self.inds[d]['merit'].l.mb[0] })
                        
        if len(dict_list)>0:
            self.log('Current Trades: {}'
                             .format(self.ntrades),dt,printout=False)
            self.filter_buy = (pd.DataFrame(dict_list)
                            .nlargest(10-self.ntrades,['Merit']))
            self.log('Added Trades: {}'
                            .format(len(self.filter_buy)),dt,printout=False)
            for d in self.filter_buy['Data']:
                self.o[d] = [self.buy(data=d,info={'datetime':dt})]
                # self.holding[d] = 0
        
    # Gather results of the orders
    def notify_order(self, order):
        dt, dn = self.datetime.datetime(), order.data._name 

        if order.status in [order.Submitted, order.Accepted]:
            return
          
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.ntrades = self.ntrades+1
                self.log(
                    'Notify Order {} BGHT, '
                    'Sz: {:.3f}, Pr: {:.2f}, '
                    'Csh: {:,.2f}'.format(
                      dn.split('_')[0], order.executed.size,
                      order.executed.price,self.broker.getcash()), dt)
            elif order.issell(): # Sell
                self.ntrades = self.ntrades-1
                self.log(
                    'Notify Order {} SLD, '
                    'Shr: {:.3f}, ShPrc: {:.2f}, '
                    'Csh: {:,.2f}'.format(
                      dn.split('_')[0], order.executed.size,
                      order.executed.price,self.broker.getcash()), dt)
            
        # Attention: Broker could reject order if not enough cash            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order.issell():
                self.log('**** Notify: SELL Order for {} was '
                       'Rejected, {}\n\n'.format(dn,order.status), dt)
            elif order.isbuy():
                self.log('**** Notify: BUY Order for {} was '
                        'Rejected, {}\n\n'.format(dn,order.status),
                        dt,printout=False)
        
        # Write down: no pending order (reset method)
        if not order.alive():  # not alive - nullify
            dorders = self.o[order.data]
            idx = dorders.index(order)
            dorders[idx] = None

            if all(x is None for x in dorders):
                dorders[:] = []  # empty list - New orders allowed  
    
    # Gather results of the trade
    def notify_trade(self, trade):
        # print(trade)
        # self.dateopen[trade.data] = bt.num2date(trade.dtopen).date()
        
        dt = self.datetime.date()
        if not trade.isclosed:
            return
        pnl_diff = (trade.history[1].event.order.executed.price-
                          trade.price)
        self.log(
            'Trade Pnl: {:,.2f} ({:.2f}%), Account FPnL: {:,.2f}'
            .format(trade.pnl, pnl_diff, self.broker.getvalue()),dt)
        
    def stop(self):
        #self.log('(MA Period %2d) Ending Value %.2f' %
        #         (self.params.maperiod, 
        #         self.broker.getvalue()), doprint=True)
        self.value = self.broker.getvalue()

    def log(self, txt, dt, doprint=False, printout=True):
        if self.params.printLog or doprint:
            if printout:
                print('%s, %s' % (dt.isoformat(), txt))
            if self.p.logfile:
                original = sys.stdout
                sys.stdout = open(self.p.logfile, 'a+')
                print('%s, %s' % (dt.isoformat(), txt))
                sys.stdout = original


class DataBroker(bt.observer.Observer):
    # I can use this observer to also track the trade duration
    _stclock=True
    
    lines = ('value_plus','value_minus','value')
    plotinfo = dict(plotlinelabels=True)
    
    plotlines = dict(
    value=dict(color='black'),
    value_plus=dict(color='green'),
    value_minus=dict(color='red')
    )
    
    def __init__(self):
        self.valplot = 0
        self.intrade = False
        self.startpos = 0
        self.endpos = 0
        self.init=0

    def next(self):        
        if self.data._name != 'SPY':
            val = (self._owner.broker.
                    getvalue(datas=[self.data]))
            
            a=self._owner._trades[self.data][0]
            if len(a)>0 and len(a[-1].history)>0:
                self.startpos = (a[-1].history[0].event.order.executed.price* 
                      a[-1].history[0].event.order.executed.size)
                if len(a[-1].history)>1:
                    self.endpos = (-a[-1].history[1]
                                       .event.order.executed.price* 
                                   a[-1].history[1]
                                       .event.order.executed.size)
    
            if not self.intrade and val==0:
                self.lines.value[0] = self.init
                if self.init > 0:
                    self.lines.value_plus[0] = self.init
                elif self.init < 0:
                    self.lines.value_minus[0] = self.init
                self.intrade=False
    
            elif self.intrade and val!=0:
                
                self.valplot = val-self.startpos+self.init
                
                self.lines.value[0] = self.valplot
                if self.valplot > 0:
                    self.lines.value_plus[0] = self.valplot
                elif self.valplot < 0:
                    self.lines.value_minus[0] = self.valplot
                self.intrade = True
    
            elif self.intrade and val==0:
                # Exiting the trade
                self.valplot = self.endpos-self.startpos+self.init
                
                self.lines.value[0] = self.valplot
                if self.valplot > 0:
                    self.lines.value_plus[0] = self.valplot
                elif self.valplot < 0:
                    self.lines.value_minus[0] = self.valplot
                self.intrade = False
                
                # Re-initialize "zero" reference
                self.init=self.valplot
    
            elif not self.intrade and val!=0:
                self.valplot = (val-self.startpos)+self.init
                
                self.lines.value[0] = self.valplot
                if self.valplot > 0:
                    self.lines.value_plus[0] = self.valplot
                elif self.valplot < 0:
                    self.lines.value_minus[0] = self.valplot
                self.intrade = True

def logger(path_to_file,text,printconsole=True):
    original = sys.stdout
    sys.stdout = open(path_to_file, 'a+')
    print(text)
    sys.stdout = original
    if printconsole:
        print(text)


if __name__ == '__main__':
    
    # Choose whether to save pickle files for each symbol and a summary note
    plot_option = True
    save_pickles = False
    save_note = False
    
    ds = datetime.datetime.now().date()
    ts_H = str(datetime.datetime.now().time()).split(':')[0]
    ts_M = str(datetime.datetime.now().time()).split(':')[1]
    ts_S = str(round(float(str(datetime.datetime.now().time())
                                              .split(':')[2])))
    dtstring = '{}_{}-{}-{}'.format(ds,ts_H,ts_M,ts_S)
    
    save_path = os.getcwd() + '/results/' + dtstring + '/'
    
    # Create the new folder
    try: 
        os.makedirs(save_path, exist_ok = True) 
        # print("Directory '%s' created successfully" %save_path)
    except OSError as error: 
        print("Directory cannot be created") 
    
    logfile = save_path + '{}_log.txt'.format(dtstring)
    
    if save_note:
        notestring = input('Summary of Backtest: ')    
        logger(logfile,'Summary of Backtest: {}'.format(notestring),False)
    else:
        logger(logfile,'Summary of Backtest: No note saved',False)
      
    shutil.copy(__file__,save_path) #copy the file to destination dir
    dst_file = os.path.join(save_path,os.path.basename(__file__))
    new_dst_file_name = os.path.join(save_path, 
                                     '{}_codecopy.py'.format(dtstring))
    os.rename(dst_file, new_dst_file_name) #rename
    
    # Create a cerebro object
    cerebro = bt.Cerebro(stdstats=False,
                         optreturn=False,
                         runonce=True)

    # Add a strategy
    cerebro.addstrategy(Strat1,logfile=logfile)

    ############
    # NEED TO EDIT
    ys=2022;ms=3;ds=13
    ye=2022;me=3;de=19
    mnt_folder = ('C:/Users/rtm/Documents/Personal/Stock_Market/Python/' +
                    'universe_data/sp500/stock_data/yf/')
    ############

    sdate=datetime.datetime(ys,ms,ds)
    edate=datetime.datetime(ye,me,de)
    logger(logfile,
           '\nStart Date of Backtest: {}\nEnd Date of Backtest: {}'
           .format(sdate,edate))

    logger(logfile,'\n*****************',False)
    logger(logfile,'** ADDING DATA **',False)
    logger(logfile,'*****************',False)

    # Add data feeds
    for filename in os.listdir(mnt_folder):
        if filename.endswith(".csv"):
            f = mnt_folder + filename
            substring = (filename.split('_')[4] + '_' + 
                         filename.split('_')[6]).replace('.csv','')
            agg_day = filename.split('_')[0]
            agg_hr = filename.split('_')[1]
            agg_min = filename.split('_')[2]
            agg_sec = filename.split('_')[3]
            y1 = int(filename.split('_')[5].split('-')[0])
            m1 = int(filename.split('_')[5].split('-')[1])
            d1 = int(filename.split('_')[5].split('-')[2])
            y2 = int(filename.split('_')[6].split('-')[0])
            m2 = int(filename.split('_')[6].split('-')[1])
            d2 = int(filename.split('_')[6].split('-')[2].replace('.csv',''))
            date1 = datetime.datetime(y1,m1,d1)
            date2 = datetime.datetime(y2,m2,d2)
            
            if ((agg_day != '0') and ((date1<=sdate and date1<=edate and
                 date2<=sdate and date2<=edate) or
                (date1>=sdate and date1>=edate and 
                 date2>=sdate and date2>=edate) or
                (date2-sdate).days < Strat1.params.smalong*2 or
                (edate-date1).days < Strat1.params.smalong*2) ):
                continue
            else:
                if agg_day != '0':
                    data = IQFeedDailyData(dataname=f,
                        fromdate=sdate,
                        todate=edate,
                        timeframe=bt.TimeFrame.Days,
                        plot=False,
                        compression=1)
                elif (agg_hr != '0') or (agg_min != '0'):
                    data = IQFeedMinuteData(dataname=f,
                              fromdate=sdate,
                              todate=edate,
                              timeframe=bt.TimeFrame.Minutes,
                              plot=False)

            cerebro.adddata(data, name=substring)
            logger(logfile,'Added Data {}'.format(f),False)
    
    # Set starting cash
    startcash = 10000
    cerebro.broker.setcash(startcash)
    cerebro.broker.set_shortcash(False)
    
    # Add a sizer
    cerebro.addsizer(FixedValueSizer, value=1000)
    
    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Add Observers
    cerebro.addobserver(bt.observers.TimeReturn,
                        timeframe=bt.TimeFrame.NoTimeFrame,plot=False)
    cerebro.addobservermulti(bt.observers.BuySell,barplot=True)
    cerebro.addobservermulti(DataBroker,plot=False,
                             subplot=False,plotabove=False)
    cerebro.addobserver(bt.observers.Broker,plot=False)
    
    logger(logfile,'\n*****************************************************')
    logger(logfile,'** LAUNCHING CEREBRO AT {} **'.format(
                                                datetime.datetime.now()))
    logger(logfile,'*****************************************************')
    # Run Cerebro without optimization
    tic2 = time.perf_counter()
    result = cerebro.run()
    toc2 = time.perf_counter()
    
    logger(logfile,'\nCerebro Execution Time: {:.2f} min'.format(
                                                (toc2-tic2)/60))
    
    #Get final portfolio Value
    portvalue = cerebro.broker.getvalue()
    endcash = cerebro.broker.getcash()
    pnl = portvalue - startcash
    
    #Print out the final result
    logger(logfile,'\n***********************')        
    logger(logfile,'** PORTFOLIO SUMMARY **')
    logger(logfile,'***********************')
    logger(logfile,'Starting Portfolio Value: {:,.2f}'.format(startcash))
    logger(logfile,'Final Portfolio Value: {:,.2f}'.format(portvalue))
    logger(logfile,'P/L: {:,.2f}'.format(pnl))
    logger(logfile,'Final Cash In Hand: {:,.2f}'.format(endcash))

    datex = []
    for i in range(-len(result[0])+1,1,1):
        datex.append(result[0].datetime.datetime(i))
    
    # Plot Results
    data_to_plot = []
    filter_ret = pd.DataFrame(columns=['Data','Name','Max','Min','FinalVal'])
    for x in range(len(result[0].datas)):
        # First check that there were non-zero number of buysell operations
        if np.sum(~np.isnan(
                    result[0].observers.buysell[x].lines[0].array))!=0:
            data_to_plot.append(x)
            # Collect data used for filters
            filter_ret = filter_ret.append({'Data': int(x), 
                        'Name': result[0].datas[x]._name.split('_')[0],
                        'Max': np.nanmax(result[0].observers.databroker[x]
                                        .lines[2].get(size=len(datex))), 
                        'Min': np.nanmin(result[0].observers.databroker[x]
                                        .lines[2].get(size=len(datex))), 
                        'FinalVal': result[0].observers.databroker[x]
                                        .lines[2].get(size=len(datex))[-1]}, 
                                       ignore_index=True)
    
    filter_ret = filter_ret.set_index(['Data'])    
    if len(data_to_plot)==0:
        logger(logfile,'\n*************************************')        
        logger(logfile,'WARNING: ZERO STOCKS MET CRITERIA')
        logger(logfile,'*************************************')
        
    # Percentile code for reference
    # max_winners = filter_ret['Max'].quantile(0.9)
    # min_losers = filter_ret['Min'].quantile(0.1)
    # final_winners = filter_ret['FinalVal'].quantile(0.9)
    # final_losers = filter_ret['FinalVal'].quantile(0.1)
    
    # Define filter criteria        
    max_winners = (filter_ret.sort_values('Max',ascending=False)[0:2]['Name']
                   .tolist())
    min_losers = (filter_ret.sort_values('Min',ascending=True)[0:2]['Name']
                  .tolist())
    final_winners = (filter_ret
                     .sort_values('FinalVal',ascending=False)[0:3]['Name']
                     .tolist())
    final_losers = (filter_ret
                    .sort_values('FinalVal',ascending=True)[0:3]['Name']
                    .tolist())

    logger(logfile,'\n********************',False)    
    logger(logfile,'** SAVING FIGURES **',False)
    logger(logfile,'********************',False)
    result[0].observers.timereturn.plotinfo.plot=False
    fignum=0
    for x in data_to_plot:
        if (plot_option and
                (filter_ret['Name'][x] in max_winners or
                filter_ret['Name'][x] in min_losers or
                filter_ret['Name'][x] in final_winners or
                filter_ret['Name'][x] in final_losers)):
            
            result[0].datas[x].plotinfo.plot=True
            result[0].observers.databroker[x].plotinfo.plot=True
            result[0].observers.databroker[x].plotinfo.subplot=True
            result[0].observers.databroker[x].plotinfo.plotabove=True
            
            fignum=fignum+1
            fig=cerebro.plot(userfig=fignum,iplot=False,
                 fmt_x_ticks = '%Y-%m-%d %H:%M',style='candle',trend=False,
                 barup='green',bardown='red',
                 rowsmajor=2,rowsminor=1,
                 voloverlay=False,volup='gray',voldown='gray',voltrans=0.5,
                 subtxtsize=6,legendind=True,
                 hlinescolor=[0,0,1],hlinesstyle='-',hlineswidth=0.75,
                 tickrotation=30)
            fig[0][0].subplots_adjust(left=0.01,right=0.93,
                                      top=0.99,bottom=0.15)
            
            result[0].datas[x].plotinfo.plot=False
            result[0].observers.databroker[x].plotinfo.plot=False
            result[0].observers.databroker[x].plotinfo.subplot=False
            result[0].observers.databroker[x].plotinfo.plotabove=False
            
            dataax=fig[0][0].axes[1].lines[0].get_xdata()
            
            buys = fig[0][0].axes[1].lines[0].get_ydata()
            sells = fig[0][0].axes[1].lines[1].get_ydata()
            
            for indx in range(len(dataax)):
                if np.isnan(buys[indx]) == False:
                    xpt = [dataax[indx],dataax[indx]]
                    for ax in fig[0][0].axes:
                        ybnd = ax.get_ybound()
                        l = plt_lines.Line2D(xpt, ybnd, c='lime',lw=0.75)
                        ax.add_line(l)
                elif np.isnan(sells[indx]) == False:
                    xpt = [dataax[indx],dataax[indx]]
                    for ax in fig[0][0].axes:
                        ybnd = ax.get_ybound()
                        l = plt_lines.Line2D(xpt, ybnd, c='red',lw=0.75)
                        ax.add_line(l)
            
            if save_pickles:
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()
                plt.show()
                plt.savefig('{}/{}.png'.format(
                    save_path,result[0].datas[x]._name.split('_')[0]), 
                    dpi=200,format='png',bbox_inches='tight')
                
                figout = open('{}/{}.pickle'.format(
                    save_path,result[0].datas[x]._name.split('_')[0]),'wb')
                pickle.dump(fig[0][0], figout)
                figout.close()
                logger(logfile,
                    'Pickled Figure for {}'.format(
                    result[0].datas[x]._name.split('_')[0]),False)
                mng.full_screen_toggle()
            
                
    # Print out
    logger(logfile,'\n*****************************')
    logger(logfile,'** WINNERS AND LOSERS ($$) **')
    logger(logfile,'*****************************')
    logger(logfile,'Biggest Winners')
    df_max = (filter_ret.sort_values('Max',ascending=False)[0:2]
          .reset_index().loc[:,['Name','Max']])
    logger(logfile,df_max)
    
    logger(logfile,'\nBiggest Losers')
    df_min = (filter_ret.sort_values('Min',ascending=True)[0:2]
          .reset_index().loc[:,['Name','Min']])
    logger(logfile,df_min)
    
    logger(logfile,'\nFINAL VALUE Winners')
    df_final_max = (filter_ret.sort_values('FinalVal',ascending=False)[0:3]
          .reset_index().loc[:,['Name','FinalVal']])
    logger(logfile,df_final_max)
    
    logger(logfile,'\nFINAL VALUE Losers')
    df_final_min = (filter_ret.sort_values('FinalVal',ascending=True)[0:3]
          .reset_index().loc[:,['Name','FinalVal']])
    logger(logfile,df_final_min)
    
    df_max.to_csv(save_path + 'top_max_results.csv',header=True,index=True)
    df_min.to_csv(save_path + 'top_min_results.csv',header=True,index=True)
    df_final_max.to_csv(save_path + 'top_final_max_results.csv',
                                                header=True,index=True)
    df_final_min.to_csv(save_path + 'top_final_min_results.csv',
                                                header=True,index=True)

    result[0].observers.broker.plotinfo.plot=True
    result[0].observers.timereturn.plotinfo.plot=True
    
    fig=cerebro.plot(userfig=fignum+1,iplot=False,fmt_x_ticks = '%Y-%m-%d',
              style='line',tickrotation=30) #bar or candle
    fig[0][0].subplots_adjust(left=0.01,right=0.92,top=0.99,bottom=0.15)
    
    f = ('C:/Users/rtm/Documents/Personal/Stock_Market/Python/' +
                    'universe_data/sp500/stock_data/yf/benchmark/' +
                    '0_0_1_0_SPY_2022-03-13_2022-03-19.csv')
    bench = pd.read_csv(f,index_col='Date',parse_dates=True)
    print(bench)
    bench.index = bench.index.date
    bench['Ret'] = bench['Close'].loc[
        (bench.index >= sdate.date()
              + datetime.timedelta(days=Strat1.params.smalong*1.5)) &
        (bench.index <= edate.date())]
    bench['Ret'] = (bench['Ret'].pct_change().add(1+0.002/252)
            .cumprod().subtract(1).fillna(0))
    
    xpt = []
    ypt = []
    for x in fig[0][0].axes[1].lines[0].get_xdata():
        xpt.append(x)
        dt = (datetime.datetime.strptime(
                fig[0][0].axes[1].format_xdata(x),'%Y-%m-%d %H:%M').date())
        ypt.append(bench['Ret'][bench.index==dt].values[0])
    
    # Create a 'line' that transfers the corresponding return for the date
    ax = fig[0][0].axes[0]
    l = plt_lines.Line2D(xpt, ypt, c='black', lw=1.5)
    ax.add_line(l)
    ax.set_ylim((min(min(ypt),min(result[0].observers.timereturn.lines[0]
            .get(size=len(datex))))*1.25),
                (max(max(ypt),max(result[0].observers.timereturn.lines[0]
            .get(size=len(datex)))))*1.25)
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    plt.savefig('{}.png'.format(save_path + 'Benchmark_Plot'), 
        dpi=200,bbox_inches='tight',format='png')
    
    figout = open('{}.pickle'.format(save_path + 'Benchmark_Plot'),'wb')
    pickle.dump(fig[0][0], figout)
    figout.close()
    mng.full_screen_toggle()
    
    # Save the returns as a CSV for future use
    ret=pd.DataFrame(result[0].observers.timereturn.lines[0]
                      .get(size=len(datex)),columns=['Strategy'])
    ret['Reference']=ypt
    ret['Date']=datex
    ret = ret.set_index('Date')
    ret.to_csv(save_path + 'benchmark_results.csv',header=True,index=True)