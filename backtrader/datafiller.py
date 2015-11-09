#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import datetime

from backtrader import AbstractDataBase, TimeFrame, date2num, num2date
from backtrader.utils.py3 import with_metaclass
from . import metabase

MAXDATE = datetime.datetime.max


class DataFiller(AbstractDataBase):
    '''This class will fill gaps in the source data using the following
    information bits from the underlying data source

      - timeframe and compression to dimension the output bars

      - sessionstart and sessionend

    If a data feed has missing bars in between 10:31 and 10:34 and the
    timeframe is minutes, the output will be filled with bars for minutes
    10:32 and 10:33 using the closing price of the last bar (10:31)

    Bars can be missinga amongst other things because

    Params:
      - ``fill_price`` (def: None): if None (or evaluates to False),the
        closing price will be used, else the passed value (which can be
        for example 'NaN' to have a missing bar in terms of evaluation but
        present in terms of time

      - ``fill_vol`` (def: NaN): used to fill the volume of missing bars

      - ``fill_oi`` (def: NaN): used to fill the openinterest of missing bars
    '''

    params = (
        ('fill_price', None),
        ('fill_vol', float('NaN')),
        ('fill_oi', float('NaN')),
        )

    def start(self):
        super(DataFiller, self).start()
        self._fillbars = collections.deque()
        self._dbar = False

    def preload(self):
        if len(self.p.dataname) == self.p.dataname.buflen():
            # if data is not preloaded .... do it
            self.p.dataname.start()
            self.p.dataname.preload()
            self.p.dataname.home()

        # Copy timeframe from data after start (some sources do autodetection)
        self.p.timeframe = self._timeframe = self.p.dataname._timeframe
        self.p.compression = self._compression = self.p.dataname._compression

        super(DataFiller, self).preload()

    def _copyfromdata(self):
        # Data is allowed - Copy size which is "number of lines"
        for i in range(self.p.dataname.size()):
            self.lines[i][0] = self.p.dataname.lines[i][0]

        self._dbar = False  # invalidate flag for read bar

        return True

    def _frombars(self):
        dtime, price = self._fillbars.popleft()

        price = self.p.fill_price or price

        self.lines.datetime[0] = date2num(dtime)
        self.lines.open[0] = price
        self.lines.high[0] = price
        self.lines.low[0] = price
        self.lines.close[0] = price
        self.lines.volume[0] = self.p.fill_vol
        self.lines.openinterest[0] = self.p.fill_oi

        return True

    # Minimum delta unit in between bars
    _tdeltas = {
        TimeFrame.Minutes: datetime.timedelta(seconds=60),
        TimeFrame.Seconds: datetime.timedelta(seconds=1),
        TimeFrame.MicroSeconds: datetime.timedelta(microseconds=1),
    }

    def _load(self):
        if not len(self.p.dataname):
            self.p.dataname.start()  # start data if not done somewhere else

            # Copy from underlying data
            self._timeframe = self.p.dataname._timeframe
            self._compression = self.p.dataname._compression

            self.p.timeframe = self._timeframe
            self.p.compression = self._compression

            # Calculate and save timedelta for timeframe
            self._tdunit = self._tdeltas[self._timeframe]
            self._tdunit *= self._compression

        if self._fillbars:
            return self._frombars()

        # use existing bar or fetch a bar
        self._dbar = self._dbar or self.p.dataname.next()
        if not self._dbar:
            return False  # no more data

        if len(self) == 1:
            # Cannot yet look backwards - deliver data as is
            return self._copyfromdata()

        # previous (delivered) close
        pclose = self.lines.close[-1]
        # Get time of previous (already delivered) bar
        dtime_prev = self.lines.datetime.datetime(-1)
        # Get time of current (from data source) bar
        dtime_cur = self.p.dataname.datetime.datetime(0)

        # Calculate session end for previous bar
        send = self.lines.datetime.tm2datetime(self.sessionend, ago=-1)

        if dtime_cur > send:  # if jumped boundary
            # 1. check for missing bars until boundary (end)
            dtime_prev += self._tdunit
            while dtime_prev < send:
                self._fillbars.append((dtime_prev, pclose))
                dtime_prev += self._tdunit

            # Calculate session start for new bar
            sstart = self.p.dataname.datetime.tm2datetime(self.sessionstart)

            # 2. check for missing bars from new boundary (start)
            # check gap from new sessionstart
            while sstart < dtime_cur:
                self._fillbars.append((sstart, pclose))
                sstart += self._tdunit
        else:
            # no boundary jumped - check gap until current time
            dtime_prev += self._tdunit
            while dtime_prev < dtime_cur:
                self._fillbars.append((dtime_prev, pclose))
                dtime_prev += self._tdunit

        if self._fillbars:
            self._dbar = True  # flag a pending data bar is available

            # return an accumulated bar in current cycle
            return self._frombars()

        return self._copyfromdata()


class SessionFiller(with_metaclass(metabase.MetaParams, object)):
    '''
    Bar Filler for a Data Source inside the declared session start/end times.

    The fill bars are constructed using the declared Data Source ``timeframe``
    and ``compression`` (used to calculate the intervening missing times)

    Params:

      - fill_price (def: None):

        If None is passed, the closing price of the previous bar will be
        used. To end up with a bar which for example takes time but it is not
        displayed in a plot ... use float('Nan')

      - fill_vol (def: float('NaN')):

        Value to use to fill the missing volume

      - fill_oi (def: float('NaN')):

        Value to use to fill the missing Open Interest

      - skip_first_fill (def: True):

        Upon seeing the 1st valid bar do not fill from the sessionstart up to
        that bar
    '''
    params = (('fill_price', None),
              ('fill_vol', float('NaN')),
              ('fill_oi', float('NaN')),
              ('skip_first_fill', True))

    # Minimum delta unit in between bars
    _tdeltas = {
        TimeFrame.Minutes: datetime.timedelta(seconds=60),
        TimeFrame.Seconds: datetime.timedelta(seconds=1),
        TimeFrame.MicroSeconds: datetime.timedelta(microseconds=1),
    }

    def __init__(self, data):
        # Calculate and save timedelta for timeframe
        self._tdunit = self._tdeltas[data._timeframe] * data._compression

        self.seenbar = False  # control if at least one bar has been seen
        self.sessend = MAXDATE  # maxdate is the control for bar in session

    def __call__(self, data):
        '''
        Params:
          - data: the data source to filter/process

        Returns:
          - False (always) because this filter does not remove bars from the
        stream

        The logic (starting with a session end control flag of MAXDATE)

          - If new bar is over session end (never true for 1st bar)

            Fill up to session end. Reset sessionend to MAXDATE & fall through

          - If session end is flagged as MAXDATE

            Recalculate session limits and check whether the bar is within them

            if so, fill up and record the last seen tim

          - Else ... the incoming bar is in the session, fill up to it
        '''
        # Get time of current (from data source) bar
        dtime_cur = data.datetime.datetime()

        if dtime_cur > self.sessend:
            # bar over session end - fill up and invalidate
            self._fillbars(data, self.dtime_prev, self.sessend + self._tdunit)
            self.sessend = MAXDATE

        # Fall through from previous check ... the bar which is over the
        # session could already be in a new session and within the limits
        if self.sessend == MAXDATE:
            # No bar seen yet or one went over previous session limit
            sessstart = data.datetime.tm2datetime(data.sessionstart)
            self.sessend = sessend = data.datetime.tm2datetime(data.sessionend)

            if sessstart <= dtime_cur <= sessend:
                # 1st bar from session in the session - fill from session start
                if self.seenbar or not self.p.skip_first_fill:
                    self._fillbars(data, sessstart - self._tdunit, dtime_cur)

            self.seenbar = True
            self.dtime_prev = dtime_cur

        else:
            # Seen a previous bar and this is in the session - fill up to it
            self._fillbars(data, self.dtime_prev, dtime_cur)
            self.dtime_prev = dtime_cur

        return False

    def _fillbars(self, data, time_start, time_end, forcedirty=False):
        '''
        Fills one by one bars as needed from time_start to time_end

        Invalidates the control dtime_prev if requested
        '''
        # Control flag - bars added to the stack
        dirty = False

        time_start += self._tdunit
        while time_start < time_end:
            dirty = self._fillbar(data, time_start)
            time_start += self._tdunit

        if dirty or forcedirty:
            data._save2stack(erase=True)

    def _fillbar(self, data, dtime):
        # Prepare an array of the needed size
        bar = [float('Nan')] * data.size()

        # Fill datetime
        bar[data.DateTime] = date2num(dtime)

        # Fill the prices
        price = self.p.fill_price or data.close[-1]
        for pricetype in [data.Open, data.High, data.Low, data.Close]:
            bar[pricetype] = price

        # Fill volume and open interest
        bar[data.Volume] = self.p.fill_vol
        bar[data.OpenInterest] = self.p.fill_oi

        # Fill extra lines the data feed may have defined beyond DateTime
        for i in range(data.DateTime + 1, data.size()):
            bar[i] = data.lines[i][0]

        # Add tot he stack of bars to save
        data._add2stack(bar)

        return True