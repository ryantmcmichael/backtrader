# Money Flow Index from Backtrader
# https://www.backtrader.com/recipes/indicators/mfi/mfi/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from . import Indicator, SumN
from . import DivByZero

class MFI(Indicator):
    lines = ('mfi',)
    params = dict(period=14)

    alias = ('MoneyFlowIndicator',)

    def __init__(self):
        tprice = (self.data.close + self.data.low + self.data.high) / 3.0
        mfraw = tprice * self.data.volume

        flowpos = SumN(mfraw * (tprice > tprice(-1)), period=self.p.period)
        flowneg = SumN(mfraw * (tprice < tprice(-1)), period=self.p.period)

        mfiratio = DivByZero(flowpos, flowneg, zero=100.0)
        self.l.mfi = 100.0 - 100.0 / (1.0 + mfiratio)