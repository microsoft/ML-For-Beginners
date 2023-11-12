# -*- coding: utf-8 -*-
"""Descriptive Statistics for Time Series

Created on Sat Oct 30 14:24:08 2010

Author: josef-pktd
License: BSD(3clause)
"""

import numpy as np
from . import stattools as stt


#todo: check subclassing for descriptive stats classes
class TsaDescriptive:
    '''collection of descriptive statistical methods for time series

    '''

    def __init__(self, data, label=None, name=''):
        self.data = data
        self.label = label
        self.name = name

    def filter(self, num, den):
        from scipy.signal import lfilter
        xfiltered = lfilter(num, den, self.data)
        return self.__class__(xfiltered, self.label, self.name + '_filtered')

    def detrend(self, order=1):
        from . import tsatools
        xdetrended = tsatools.detrend(self.data, order=order)
        return self.__class__(xdetrended, self.label, self.name + '_detrended')

    def fit(self, order=(1,0,1), **kwds):
        from .arima_model import ARMA
        self.mod = ARMA(self.data)
        self.res = self.mod.fit(order=order, **kwds)
        #self.estimated_process =
        return self.res

    def acf(self, nlags=40):
        return stt.acf(self.data, nlags=nlags)

    def pacf(self, nlags=40):
        return stt.pacf(self.data, nlags=nlags)

    def periodogram(self):
        #does not return frequesncies
        return stt.periodogram(self.data)

    # copied from fftarma.py
    def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
        data = self.data
        acf = self.acf(nacf)
        pacf = self.pacf(nacf)
        w = np.linspace(0, np.pi, nfreq, endpoint=False)
        spdr = self.periodogram()[:nfreq] #(w)

        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        namestr = ' for %s' % self.name if self.name else ''
        ax.plot(data)
        ax.set_title('Time series' + namestr)

        ax = fig.add_subplot(2,2,2)
        ax.plot(acf)
        ax.set_title('Autocorrelation' + namestr)

        ax = fig.add_subplot(2,2,3)
        ax.plot(spdr) # (wr, spdr)
        ax.set_title('Power Spectrum' + namestr)

        ax = fig.add_subplot(2,2,4)
        ax.plot(pacf)
        ax.set_title('Partial Autocorrelation' + namestr)

        return fig
