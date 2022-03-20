import backtrader as bt
import datetime

class meritsimple(bt.Indicator):
    lines = ('mb','exit','mb_smashort','mb_smalong','mb_gapper','mb_volume','mb_bullbar',
             'mb_bulltrend','mb_rsimin','mb_voltrend')
    
    params = (
            ('smashort',0),
            ('smalong',0),
            ('absvol',50000),
            ('volpct_tar',1.02),
            ('rsiperiod',12),
            ('rsimin',50),
            ('rsimax',70),
            ('tradestart',datetime.time(9,0,0)),
            ('tradestop',datetime.time(16,30,0)),
            ('Wsmashort',5),
            ('Wvoltrend',1),
            ('Wbullbar',1),
            ('Wbulltrend',1),
            ('Wrsimin',0),
            ('Wsmashortexit',8),
            ('Wbullbarexit',1),
            ('Wbulltrendexit',1),
            )

    # Plot options
    plotinfo = dict(
        # Add extra margins above and below the 1s and -1s
        plotymargin=0.15,

        # Plot a reference horizontal line at 1.0 and -1.0
        plothlines=[0, 25, 50, 75, 100],

        # Simplify the y scale to 1.0 and -1.0
        plotyticks=[0, 25, 50, 75, 100]
        )

    # Lines styles for plotting
    plotlines = dict(
        mb_smashort=dict(ls='',marker='^',fillstyle='none',color='black',markersize=8.0),
        mb_smalong=dict(ls='',marker='o',fillstyle='none',color='blue',markersize=9.0),
        mb_volume=dict(ls='',marker='*',fillstyle='none',color='red',markersize=8.5),
        mb_bullbar=dict(ls='',marker='.',fillstyle='full',color='black',markersize=8.0),
        mb_bulltrend=dict(ls='',marker='.',fillstyle='full',color='black',markersize=8.0),
        mb_voltrend=dict(ls='-',lw=1.0,color='blue'),
        mb_rsimin=dict(ls='-',lw=1.0,color='red'),
        mb=dict(ls='-',lw=2.0,color='black')
        )

    def __init__(self):
        self.addminperiod(self.p.smalong)
        self.rsi = bt.indicators.RSI(self.data.close,period=self.p.rsiperiod,
                                     safediv=True,safehigh=100,safelow=0)
        self.smashort = bt.indicators.SMA(self.data.close,period=self.p.smashort)
        self.smalong = bt.indicators.SMA(self.data.close,period=self.p.smalong)
        self.smagap = self.smashort-self.smalong
    
    def next(self):
        #### LAGRANGE MULTIPLIERS
        # SMA Long Constraint
        self.l.mb_smalong[0] = (self.data.close[0] > self.smalong[0] and
                            self.data.close[-1] > self.smalong[-1] and
                            (self.smalong[0]-self.smalong[-3])/3>0.01)
        
        # Volume > threshold
        self.l.mb_volume[0] = self.data.volume[0] > self.p.absvol
        
        # Gapper filter
        self.l.mb_gapper[0] = (
            max(self.data.close[0],self.data.open[0]) < self.data.close[-1]*1.05 and
            self.data.close[0] < self.data.open[0]*1.05)

        #### WEIGHTED CONSTRAINTS
        # SMA short Constraint
        self.l.mb_smashort[0] = ((self.data.close[0] > self.smashort[0]*1.01) and
            (self.data.close[-1] < self.smashort[-1] or
             self.data.close[-2] < self.smashort[-2] or
             self.data.close[-3] < self.smashort[-3]))

        # Volume should be increasing to support a trade
        self.l.mb_voltrend[0] = (self.data.volume[0] >= 
            self.data.volume[-1]*self.p.volpct_tar)
        
        # Requires bullish bar on current and previous bar
        self.l.mb_bullbar[0] = self.data.close[0]>self.data.open[0]
        
        # Requires a trend of 2 increasing bars
        self.l.mb_bulltrend[0] = self.data.close[0]>self.data.close[-1]
        
        # RSI must be below the min threshold
        self.l.mb_rsimin[0] = self.rsi[0] <= self.p.rsimin
        
        # COMPUTE LAGRANGE MULTIPLIER
        lagrangeb = self.l.mb_smalong[0]*self.l.mb_volume[0]*self.l.mb_gapper[0]
        
        # COMPUTE WEIGHTED SUM
        Wsum= (self.p.Wsmashort + self.p.Wvoltrend + self.p.Wbullbar + 
                       self.p.Wbulltrend + self.p.Wrsimin)
        weightedb = ((self.l.mb_smashort[0] * self.p.Wsmashort + 
        self.l.mb_voltrend[0] * self.p.Wvoltrend + 
        self.l.mb_bullbar[0] * self.p.Wbullbar + 
        self.l.mb_bulltrend[0] * self.p.Wbulltrend +
        self.l.mb_rsimin[0] * self.p.Wrsimin)
        / Wsum * 100);
        
        # COMPUTE TOTAL MERIT FUNCTION
        self.l.mb[0] = lagrangeb*weightedb
        
        # COMPUTE EXIT FUNCTION
        Wsumexit = (self.p.Wsmashortexit + self.p.Wbulltrendexit + 
                    self.p.Wbullbarexit)
        if ( (min(self.data.close[0],self.data.open[0]) < self.smashort[0]*0.95) and
                    (self.data.close[0]<self.smashort[0] or 
                     self.data.close[-1]<self.smashort[-1])):
            self.l.exit[0]=100
        elif (min(self.data.close[0],self.data.open[0]) < self.smashort[0]*0.99):
            self.l.exit[0]=((self.p.Wsmashortexit + 
                    (self.data.close[0] < self.data.close[-1])*self.p.Wbulltrendexit +
                    (self.data.close[0]<self.data.open[0])*self.p.Wbullbarexit)
                    / Wsumexit * 100)
        else:
            self.l.exit[0]=0
                
class meritweighted(bt.Indicator):
    lines = ('mb','mb_smashort','mb_smalong','mb_volume','mb_shpr','mb_bullbar',
             'mb_bulltrend','mb_rsi','mb_voltrend') #,'mb_tradehrs')
    
    params = (
            ('smashort',8),
            ('smalong',80),
            ('absvol',50000),
            ('abspr',1000),
            ('volpct_tar',1.01),
            ('rsiperiod',12),
            ('rsimin',30),
            ('rsimax',60),
            ('tradestart',datetime.time(9,0,0)),
            ('tradestop',datetime.time(16,30,0))
            )

    # Plot options
    plotinfo = dict(
        # Add extra margins above and below the 1s and -1s
        plotymargin=0.15,

        # Plot a reference horizontal line at 1.0 and -1.0
        plothlines=[1.0, 0.5, 0.0],

        # Simplify the y scale to 1.0 and -1.0
        plotyticks=[1.0, 0.0]
        )

    # Lines styles for plotting
    plotlines = dict(
        mb_smashort=dict(ls='',marker='^',fillstyle='none',color='black',markersize=8.0),
        mb_smalong=dict(ls='',marker='o',fillstyle='none',color='blue',markersize=9.0),
        mb_volume=dict(ls='',marker='*',fillstyle='none',color='red',markersize=8.5),
        mb_shpr=dict(ls='',marker='s',fillstyle='none',color='green',markersize=10.0),
        mb_bullbar=dict(ls='',marker='.',fillstyle='full',color='black',markersize=8.0),
        mb_bulltrend=dict(ls='',marker='.',fillstyle='full',color='black',markersize=8.0),
        mb_voltrend=dict(ls='-',lw=1.0,color='blue'),
        mb_rsi=dict(ls='-',lw=1.0,color='red'),
        mb=dict(ls='-',lw=2.0,color='black')
        )

    def __init__(self):
        self.addminperiod(self.p.smalong)
        self.rsi = bt.indicators.RSI(self.data.close,period=self.p.rsiperiod,
                                     safediv=True,safehigh=100,safelow=0)
        self.smashort = bt.indicators.SMA(self.data.close,period=self.p.smashort)
        self.smalong = bt.indicators.SMA(self.data.close,period=self.p.smalong)
        self.smagap = self.smashort-self.smalong
    
    def next(self):
        # Check the time
        # self.l.mb_tradehrs[0] = (self.data.datetime.time(0) >= self.p.tradestart and 
        #                     self.data.datetime.time(0) < self.p.tradestop)

        # Find the last 2 bars within trading hours
        dtind = -1
        # while (self.data.datetime.time(dtind) < self.p.tradestart or 
        #        self.data.datetime.time(dtind) >= self.p.tradestop):
        #     dtind=dtind-1
        # dtind2 = -1
        # while (self.data.datetime.time(dtind+dtind2) < self.p.tradestart or 
        #        self.data.datetime.time(dtind+dtind2) >= self.p.tradestop):
        #     dtind2=dtind2-1
        dtind2=-2

        # SMA short Constraint
        self.l.mb_smashort[0] = self.data.close[0] > self.smashort[0]
        # SMA Long Constraint
        self.l.mb_smalong[0] = ( (self.data.close[0] > self.smalong[0])
                # | ((self.data.close[0] <= self.smalong[0]) & (self.smalong[0]-self.smalong[-1] >= 0))
                # | ((self.data.close[0] <= self.smalong[0]) & (self.smagap[0]-self.smagap[-1] >= 0))
                )
        # Volume > threshold
        self.l.mb_volume[0] = self.data.volume[0] > self.p.absvol
        # Share price < threshold
        self.l.mb_shpr[0] = self.data.close[0] < self.p.abspr
        
        self.l.mb_bullbar[0] = int(
            self.data.close[0]>self.data.open[0] and
            self.data.close[-1]>self.data.open[-1]
            )
        
        self.l.mb_bulltrend[0] = int(
            self.data.close[0]>self.data.close[dtind] and
            self.data.close[dtind]>self.data.close[dtind2])
        
        #########################################################        
        # Compute weighted constraints for BUYING (varies 0 to 1)
        #########################################################
        
        # Volume should be increasing to support a trade
        self.l.mb_voltrend[0] = 0
        if self.data.volume[dtind]:
            volpct = (self.data.volume[0]-self.data.volume[dtind])/self.data.volume[dtind]+1
            if volpct>=1 and volpct<=self.p.volpct_tar:
                # self.l.mb_voltrend[0] = (volpct/self.p.volpct_tar)**2
                self.l.mb_voltrend[0] = 1
            elif volpct>self.p.volpct_tar:
                self.l.mb_voltrend[0] = 1
        
        # RSI constraint
        self.l.mb_rsi[0] = 0
        if self.rsi[0] <= self.p.rsimin:
            self.l.mb_rsi[0] = 1
        elif (self.rsi[0] < self.p.rsimax and
                  self.rsi[0] > self.p.rsimin):  
            # self.l.mb_rsi[0] = math.e**(-(3/(self.p.rsimax-self.p.rsimin))
            #              *(self.rsi[0]-self.p.rsimin))
            self.l.mb_rsi[0] = 1
            
        lagrangeb = (self.l.mb_smashort[0]*self.l.mb_smalong[0]*self.l.mb_rsi[0]*
             self.l.mb_volume[0]*self.l.mb_shpr[0]*
             self.l.mb_bullbar[0])
        
        self.l.mb[0] = lagrangeb

        # Increasing price action
        # mb_prtrend = 0
        # if self.inds[d]['pctchangelong'][0]:
        #     if (self.inds[d]['pctchangeshort'][0]>self.inds[d]['pctchangeshort'][-1] and 
        #             self.inds[d]['pctchangeshort'][-1]>0 and
        #             self.inds[d]['pctchangeshort'][0]>0 and
        #             (self.inds[d]['pctchangeshort'][0]+1)<
        #                     ((self.inds[d]['pctchangelong'][-1]+1)*self.params.pctchange)):
        #         mb_prtrend = 1
                # mb_prtrend = (1-abs(self.inds[d]['pctchangeshort'][0]-
                #     abs(self.inds[d]['pctchangelong'][0]*self.params.pctchange))/
                #     abs(self.inds[d]['pctchangelong'][0]*(self.params.pctchange-1)))
                # print('{}, {}, {}'.format(self.inds[d]['pctchangeshort'][0],
                #                           self.inds[d]['pctchangelong'][0],
                #                           mb_prtrend))