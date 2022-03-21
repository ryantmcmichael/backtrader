import backtrader as bt
import datetime

class meritsimple(bt.Indicator):
    lines = ('m','m_rsi','m_dmi','m_mfi','m_uo','m_greensignal')

    params = (
            ('smaperiod',10),
            ('rsiperiod',12),
            ('rsithresh',50),
            ('dmiperiod',14),
            ('mfiperiod',14),
            ('mfithresh',50),
            ('uop1',7),
            ('uop2',14),
            ('uop3',28),
            ('uoub',70),
            ('uolb',30),
            ('uothresh',0.5),
            ('tradestart',datetime.time(9,0,0)),
            ('tradestop',datetime.time(16,30,0)),
            )

    # Plot options
    plotinfo = dict(
        # Add extra margins above and below the 1s and -1s
        plotymargin=0.15,

        # Plot a reference horizontal line at 1.0 and -1.0
        plothlines=[0, 1, 2, 3, 4, 5],

        # Simplify the y scale to 1.0 and -1.0
        plotyticks=[0, 1, 2, 3, 4, 5]
        )

    # Lines styles for plotting
    plotlines = dict(
        m_rsi=dict(ls='',marker='^',fillstyle='none',color='black',markersize=8.0),
        m_dmi=dict(ls='',marker='o',fillstyle='none',color='blue',markersize=9.0),
        m_mfi=dict(ls='',marker='*',fillstyle='none',color='red',markersize=8.5),
        m_uo=dict(ls='',marker='.',fillstyle='full',color='black',markersize=8.0),
        m_greensignal=dict(ls='',marker='s',fillstyle='none',color='blue',markersize=8.0),
        m=dict(ls='-',lw=2.0,color='black')
        )

    # RSI, DMI, MFI, UO, ATR, GREENsignal (SMA), SMA

    def __init__(self):
        # Load all indicators
        self.addminperiod(self.p.smaperiod)

        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiperiod,
                                     safediv=True, safehigh=100, safelow=0)

        self.dmi = bt.indicators.DMI(self.data)

        self.mfi = bt.indicators.MoneyFlowIndicator(self.data, period=self.p.mfiperiod)

        self.uo = bt.indicators.UltimateOscillator(self.data,
                                                   p1 = self.p.uop1,
                                                   p2 = self.p.uop2,
                                                   p3 = self.p.uop3,
                                                   upperband = self.p.uoub,
                                                   lowerband = self.p.uolb)

        self.sma = bt.indicators.SMA(self.data.close, period=self.p.smaperiod)

    # RSI, DMI, MFI, UO, ATR, GREENsignal (SMA), SMA

    def next(self):
        #### LAGRANGE MULTIPLIERS
        # GREEN Signal is meant to be a crossover
        self.l.m_greensignal[0] = ( 1 if ((self.data.close[0] > self.sma[0])
                                    and (self.data.close[-1] > self.sma[-1])
                                    and (self.data.close[-2] > self.sma[-2])
                                    and (self.data.close[-3] > self.sma[-3])
                                    and (self.data.close[-4] > self.sma[-4])
                                    and (self.data.close[-5] > self.sma[-5])
                                    and (self.data.close[0] > self.data.open[-4])
                                    else 0 )

        # RSI must be above the threshold
        self.l.m_rsi[0] = 1 if self.rsi[0] >= self.p.rsithresh else 0

        # DMI +DI must be above -DI
        self.l.m_dmi[0] = 1 if self.dmi.plusDI[0] > self.dmi.minusDI[0] else 0

        # MFI must be above threshold
        self.l.m_mfi[0] = 1 if self.mfi[0] > self.p.mfithresh else 0

        # UO must be above threshold
        self.l.m_uo[0] = 1 if self.uo[0] > self.p.uothresh else 0

        # COMPUTE LAGRANGE MULTIPLIER (lookback xyz bars)
        lagrange = (
                    self.l.m_greensignal[0] +
                    self.l.m_rsi[0] +
                    self.l.m_dmi[0] +
                    self.l.m_mfi[0] +
                    self.l.m_uo[0]
                    )

        # COMPUTE TOTAL MERIT FUNCTION
        self.l.m[0] = lagrange


'''
input positionSize = 1000;
input percentRisk = -1;
input percentWin = 5;

def tradeSize = Floor(positionSize / open[-1]);
def buypercentChange = 100 * (close - EntryPrice()) / EntryPrice();

def agg = GetAggregationPeriod();
def marketopenFlag = (GetTime() - RegularTradingStart(GetYYYYMMDD())) > -agg and (GetTime() - RegularTradingEnd(GetYYYYMMDD())) < 0;
########################

input length = 14;
input rsi_thresh_up = 50;
input rsi_thresh_down = 50;
input rsiAverageType = AverageType.WILDERS;

# RSI
def rsi = reference RSI(price = close, length = length, averageType = rsiAverageType);
def RSIval = if rsi > rsi_thresh_up then 1 else if rsi < rsi_thresh_down then -1 else Double.NaN;
def RSIsignal = if !IsNaN(RSIval[1]) then RSIval[1] else if !IsNaN(RSIval[2]) then RSIval[2] else if !IsNaN(RSIval[3]) then RSIval[3] else RSIval[4];

#DMI
def dip = reference DIPlus();
def din = reference DIMinus();
def adx = reference ADX();
def adx_thresh = 25;
def DMIval = if ( (dip > din) ) then 1 else if ( (din < dip) ) then -1 else Double.NaN;
def DMIsignal = if !IsNaN(DMIval[1]) then DMIval[1] else if !IsNaN(DMIval[2]) then DMIval[2] else if !IsNaN(DMIval[3]) then DMIval[3] else DMIval[4];

# MFI
def mfi = reference MoneyFlowIndex();
def mfi_thresh = 50;
def MFIval = if mfi > mfi_thresh then 1 else if mfi < mfi_thresh then -1 else Double.Nan;
def MFIsignal = if !IsNaN(MFIval[1]) then MFIval[1] else if !IsNaN(MFIval[2]) then MFIval[2] else if !IsNaN(MFIval[3]) then MFIval[3] else MFIval[4];


# Ultimate Oscillator
def uo = reference UltimateOscillator();
def uo_thresh = 0.5;
def UOval = if uo > uo_thresh then 1 else if uo < uo_thresh then -1 else Double.Nan;
def UOsignal = if !IsNaN(UOval[1]) then UOval[1] else if !IsNaN(UOval[2]) then UOval[2] else if !IsNaN(UOval[3]) then UOval[3] else UOval[4];

# ATR
input MaxMinLength = 75;
def atr = reference ATR();
def ATRsignal = if atr[1] < Lowest(atr[1],MaxMinLength)*1.1 then 1 else if atr[1] > Highest(atr[1],MaxMinLength)*0.9 then -1 else Double.Nan;

def GREENsignal = if ( (close > Average(close,10)) and (close[1] > Average(close[1],10)) and (close[2] > Average(close[2],10)) and (close[3] > Average(close[3],10)) and (close[4] > Average(close[4],10)) and (close[5] > Average(close[5],10)) and (close>open[4]) ) then 1 else -1;

def Signal = RSIsignal + DMIsignal + MFIsignal + UOsignal + GREENsignal;

def upval = low*0.998;
def downval = high*1.002;

plot UpSignal = if Crosses(Signal,4.5,CrossingDirection.Above) then upval else Double.NaN;

plot DownSignal = if (Average(close, 3) crosses below Average(close, 20) or close<open[2]*0.97) then downval else if Signal < -2 then downval else Double.NaN;

UpSignal.SetDefaultColor(Color.MAGENTA);
UpSignal.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
UpSignal.SetLineWeight(3);
DownSignal.SetDefaultColor(Color.DARK_ORANGE);
DownSignal.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
DownSignal.SetLineWeight(3);


#________________ ORDERS ___________________
# ADD SMA for good exit as well
def BuyexitGood = (!IsNaN(DownSignal) or buypercentChange > percentWin) and marketopenFlag == 1;
def BuyexitBad = buypercentChange < percentRisk and marketopenFlag == 1;

AddOrder(OrderType.BUY_TO_OPEN, !IsNaN(UpSignal), open[-1], tradeSize, Color.CYAN, Color.CYAN,name="BUY");

AddOrder(OrderType.SELL_TO_CLOSE, BuyexitGood, open[-1], tradeSize, Color.GREEN, Color.GREEN,name=AsText(buypercentChange));
AddOrder(OrderType.SELL_TO_CLOSE, BuyexitBad, open[-1], tradeSize, Color.RED, Color.RED,name=AsText(buypercentChange));
'''