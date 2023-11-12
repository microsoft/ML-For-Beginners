'''Collection of alternative implementations for time series analysis


>>> signal.fftconvolve(x,x[::-1])[len(x)-1:len(x)+10]/x.shape[0]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])
>>> sm.tsa.stattools.acovf(X, fft=True)[:order+1]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])

>>> import nitime.utils as ut
>>> ut.autocov(s)[:order+1]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])
'''
import numpy as np


def acovf_fft(x, demean=True):
    '''autocovariance function with call to fftconvolve, biased

    Parameters
    ----------
    x : array_like
        timeseries, signal
    demean : bool
        If true, then demean time series

    Returns
    -------
    acovf : ndarray
        autocovariance for data, same length as x

    might work for nd in parallel with time along axis 0

    '''
    from scipy import signal
    x = np.asarray(x)

    if demean:
        x = x - x.mean()

    signal.fftconvolve(x,x[::-1])[len(x)-1:len(x)+10]/x.shape[0]
