import backtrader as bt
import datetime

class meritsimple(bt.Indicator):
    lines = ('m_up',
             'm_down',
             'm_rsi',
             'm_dmi',
             'm_mfi',
             'm_uo',
             'm_sma')

    params = (
            ('smashort',4),
            ('smalong',20),
            ('rsithresh',50),
            ('mfithresh',50),
            ('uothresh',50),
            ('atrfactor',1),
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
        m_sma=dict(ls='',marker='s',fillstyle='none',color='blue',markersize=8.0),
        m_up=dict(ls='-',lw=2.0,color='green'),
        m_down=dict(ls='-',lw=2.0,color='red')
        )

    def __init__(self):
        # Load all indicators
        self.addminperiod(self.p.smalong)

        self.rsi = bt.indicators.RSI(self.data.close,
                                     safediv=True, safehigh=100, safelow=0)

        self.dmi = bt.indicators.DMI(self.data)

        self.mfi = bt.indicators.MoneyFlowIndicator(self.data)

        self.uo = bt.indicators.UltimateOscillator(self.data)

        self.smashort = bt.indicators.SMA(self.data.close, period=self.p.smashort)
        self.smalong = bt.indicators.SMA(self.data.close, period=self.p.smalong)

        self.atr = bt.indicators.ATR(self.data)


    def next(self):
        #### LAGRANGE MULTIPLIERS
        # RSI must be above the threshold
        self.l.m_rsi[0] = 1 if self.rsi[0] >= self.p.rsithresh else 0

        # DMI +DI must be above -DI
        self.l.m_dmi[0] = 1 if self.dmi.plusDI[0] > self.dmi.minusDI[0] else 0

        # MFI must be above threshold
        self.l.m_mfi[0] = 1 if self.mfi[0] > self.p.mfithresh else 0

        # UO must be above threshold
        self.l.m_uo[0] = 1 if self.uo[0] > self.p.uothresh else 0

        # SMA Signal
        self.l.m_sma[0] = (1 if self.smashort[0] - self.smalong[0]
                            > self.p.atrfactor*self.atr[0] else 0 )

        # COMPUTE TOTAL MERIT FUNCTION
        self.l.m_up[0] = (
                    self.l.m_sma[0] +
                    self.l.m_rsi[0] +
                    self.l.m_dmi[0] +
                    self.l.m_mfi[0] +
                    self.l.m_uo[0]
                    )

        self.l.m_down[0] = (
            self.l.m_rsi[0] +
            self.l.m_dmi[0] +
            self.l.m_mfi[0] +
            self.l.m_uo[0]
            )