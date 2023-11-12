'''functions and classes time series analysis


Status
------
work in progress

arima.py
^^^^^^^^

ARIMA : initial class, uses conditional least squares, needs merging with new class
arma2ar
arma2ma
arma_acf
arma_acovf
arma_generate_sample
arma_impulse_response
deconvolve
index2lpol
lpol2index
mcarma22

movstat.py
^^^^^^^^^^

I had tested the next group against matlab, but where are the tests ?
acf
acovf
ccf
ccovf
pacf_ols
pacf_yw

These hat incorrect array size, were my first implementation, slow compared
to cumsum version in la and cython version in pandas.
These need checking, and merging/comparing with new class MovStats
check_movorder
expandarr
movmean :
movmoment : corrected cutoff
movorder
movvar




'''


#from arima import *
from .movstat import movorder, movmean, movvar, movmoment  # noqa:F401
#from stattools import *
