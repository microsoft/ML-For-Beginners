# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:53:25 2009

Author: josef-pktd

generate arma sample using fft with all the lfilter it looks slow
to get the ma representation first

apply arma filter (in ar representation) to time series to get white noise
but seems slow to be useful for fast estimation for nobs=10000

change/check: instead of using marep, use fft-transform of ar and ma
    separately, use ratio check theory is correct and example works
    DONE : feels much faster than lfilter
    -> use for estimation of ARMA
    -> use pade (scipy.interpolate) approximation to get starting polynomial
       from autocorrelation (is autocorrelation of AR(p) related to marep?)
       check if pade is fast, not for larger arrays ?
       maybe pade does not do the right thing for this, not tried yet
       scipy.pade([ 1.    ,  0.6,  0.25, 0.125, 0.0625, 0.1],2)
       raises LinAlgError: singular matrix
       also does not have roots inside unit circle ??
    -> even without initialization, it might be fast for estimation
    -> how do I enforce stationarity and invertibility,
       need helper function

get function drop imag if close to zero from numpy/scipy source, where?

"""

import numpy as np
import numpy.fft as fft
#import scipy.fftpack as fft
from scipy import signal
#from try_var_convolve import maxabs
from statsmodels.tsa.arima_process import ArmaProcess


#trying to convert old experiments to a class


class ArmaFft(ArmaProcess):
    '''fft tools for arma processes

    This class contains several methods that are providing the same or similar
    returns to try out and test different implementations.

    Notes
    -----
    TODO:
    check whether we do not want to fix maxlags, and create new instance if
    maxlag changes. usage for different lengths of timeseries ?
    or fix frequency and length for fft

    check default frequencies w, terminology norw  n_or_w

    some ffts are currently done without padding with zeros

    returns for spectral density methods needs checking, is it always the power
    spectrum hw*hw.conj()

    normalization of the power spectrum, spectral density: not checked yet, for
    example no variance of underlying process is used

    '''

    def __init__(self, ar, ma, n):
        #duplicates now that are subclassing ArmaProcess
        super(ArmaFft, self).__init__(ar, ma)

        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        self.nobs = n
        #could make the polynomials into cached attributes
        self.arpoly = np.polynomial.Polynomial(ar)
        self.mapoly = np.polynomial.Polynomial(ma)
        self.nar = len(ar)  #1d only currently
        self.nma = len(ma)

    def padarr(self, arr, maxlag, atend=True):
        '''pad 1d array with zeros at end to have length maxlag
        function that is a method, no self used

        Parameters
        ----------
        arr : array_like, 1d
            array that will be padded with zeros
        maxlag : int
            length of array after padding
        atend : bool
            If True (default), then the zeros are added to the end, otherwise
            to the front of the array

        Returns
        -------
        arrp : ndarray
            zero-padded array

        Notes
        -----
        This is mainly written to extend coefficient arrays for the lag-polynomials.
        It returns a copy.

        '''
        if atend:
            return np.r_[arr, np.zeros(maxlag-len(arr))]
        else:
            return np.r_[np.zeros(maxlag-len(arr)), arr]


    def pad(self, maxlag):
        '''construct AR and MA polynomials that are zero-padded to a common length

        Parameters
        ----------
        maxlag : int
            new length of lag-polynomials

        Returns
        -------
        ar : ndarray
            extended AR polynomial coefficients
        ma : ndarray
            extended AR polynomial coefficients

        '''
        arpad = np.r_[self.ar, np.zeros(maxlag-self.nar)]
        mapad = np.r_[self.ma, np.zeros(maxlag-self.nma)]
        return arpad, mapad

    def fftar(self, n=None):
        '''Fourier transform of AR polynomial, zero-padded at end to n

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftar : ndarray
            fft of zero-padded ar polynomial
        '''
        if n is None:
            n = len(self.ar)
        return fft.fft(self.padarr(self.ar, n))

    def fftma(self, n):
        '''Fourier transform of MA polynomial, zero-padded at end to n

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftar : ndarray
            fft of zero-padded ar polynomial
        '''
        if n is None:
            n = len(self.ar)
        return fft.fft(self.padarr(self.ma, n))

    def fftarma(self, n=None):
        '''Fourier transform of ARMA polynomial, zero-padded at end to n

        The Fourier transform of the ARMA process is calculated as the ratio
        of the fft of the MA polynomial divided by the fft of the AR polynomial.

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftarma : ndarray
            fft of zero-padded arma polynomial
        '''
        if n is None:
            n = self.nobs
        return (self.fftma(n) / self.fftar(n))

    def spd(self, npos):
        '''raw spectral density, returns Fourier transform

        n is number of points in positive spectrum, the actual number of points
        is twice as large. different from other spd methods with fft
        '''
        n = npos
        w = fft.fftfreq(2*n) * 2 * np.pi
        hw = self.fftarma(2*n)  #not sure, need to check normalization
        #return (hw*hw.conj()).real[n//2-1:]  * 0.5 / np.pi #does not show in plot
        return (hw*hw.conj()).real * 0.5 / np.pi, w

    def spdshift(self, n):
        '''power spectral density using fftshift

        currently returns two-sided according to fft frequencies, use first half
        '''
        #size = s1+s2-1
        mapadded = self.padarr(self.ma, n)
        arpadded = self.padarr(self.ar, n)
        hw = fft.fft(fft.fftshift(mapadded)) / fft.fft(fft.fftshift(arpadded))
        #return np.abs(spd)[n//2-1:]
        w = fft.fftfreq(n) * 2 * np.pi
        wslice = slice(n//2-1, None, None)
        #return (hw*hw.conj()).real[wslice], w[wslice]
        return (hw*hw.conj()).real, w

    def spddirect(self, n):
        '''power spectral density using padding to length n done by fft

        currently returns two-sided according to fft frequencies, use first half
        '''
        #size = s1+s2-1
        #abs looks wrong
        hw = fft.fft(self.ma, n) / fft.fft(self.ar, n)
        w = fft.fftfreq(n) * 2 * np.pi
        wslice = slice(None, n//2, None)
        #return (np.abs(hw)**2)[wslice], w[wslice]
        return (np.abs(hw)**2) * 0.5/np.pi, w

    def _spddirect2(self, n):
        '''this looks bad, maybe with an fftshift
        '''
        #size = s1+s2-1
        hw = (fft.fft(np.r_[self.ma[::-1],self.ma], n)
                / fft.fft(np.r_[self.ar[::-1],self.ar], n))
        return (hw*hw.conj()) #.real[n//2-1:]

    def spdroots(self, w):
        '''spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)
        '''
        return self._spdroots(self.arroots, self.maroots, w)

    def _spdroots(self, arroots, maroots, w):
        '''spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)

        Parameters
        ----------
        arroots : ndarray
            roots of ar (denominator) lag-polynomial
        maroots : ndarray
            roots of ma (numerator) lag-polynomial
        w : array_like
            frequencies for which spd is calculated

        Notes
        -----
        this should go into a function
        '''
        w = np.atleast_2d(w).T
        cosw = np.cos(w)
        #Greene 5th edt. p626, section 20.2.7.a.
        maroots = 1./maroots
        arroots = 1./arroots
        num = 1 + maroots**2 - 2* maroots * cosw
        den = 1 + arroots**2 - 2* arroots * cosw
        #print 'num.shape, den.shape', num.shape, den.shape
        hw = 0.5 / np.pi * num.prod(-1) / den.prod(-1) #or use expsumlog
        return np.squeeze(hw), w.squeeze()

    def spdpoly(self, w, nma=50):
        '''spectral density from MA polynomial representation for ARMA process

        References
        ----------
        Cochrane, section 8.3.3
        '''
        mpoly = np.polynomial.Polynomial(self.arma2ma(nma))
        hw = mpoly(np.exp(1j * w))
        spd = np.real_if_close(hw * hw.conj() * 0.5/np.pi)
        return spd, w

    def filter(self, x):
        '''
        filter a timeseries with the ARMA filter

        padding with zero is missing, in example I needed the padding to get
        initial conditions identical to direct filter

        Initial filtered observations differ from filter2 and signal.lfilter, but
        at end they are the same.

        See Also
        --------
        tsa.filters.fftconvolve

        '''
        n = x.shape[0]
        if n == self.fftarma:
            fftarma = self.fftarma
        else:
            fftarma = self.fftma(n) / self.fftar(n)
        tmpfft = fftarma * fft.fft(x)
        return fft.ifft(tmpfft)

    def filter2(self, x, pad=0):
        '''filter a time series using fftconvolve3 with ARMA filter

        padding of x currently works only if x is 1d
        in example it produces same observations at beginning as lfilter even
        without padding.

        TODO: this returns 1 additional observation at the end
        '''
        from statsmodels.tsa.filters import fftconvolve3
        if not pad:
            pass
        elif pad == 'auto':
            #just guessing how much padding
            x = self.padarr(x, x.shape[0] + 2*(self.nma+self.nar), atend=False)
        else:
            x = self.padarr(x, x.shape[0] + int(pad), atend=False)

        return fftconvolve3(x, self.ma, self.ar)


    def acf2spdfreq(self, acovf, nfreq=100, w=None):
        '''
        not really a method
        just for comparison, not efficient for large n or long acf

        this is also similarly use in tsa.stattools.periodogram with window
        '''
        if w is None:
            w = np.linspace(0, np.pi, nfreq)[:, None]
        nac = len(acovf)
        hw = 0.5 / np.pi * (acovf[0] +
                            2 * (acovf[1:] * np.cos(w*np.arange(1,nac))).sum(1))
        return hw

    def invpowerspd(self, n):
        '''autocovariance from spectral density

        scaling is correct, but n needs to be large for numerical accuracy
        maybe padding with zero in fft would be faster
        without slicing it returns 2-sided autocovariance with fftshift

        >>> ArmaFft([1, -0.5], [1., 0.4], 40).invpowerspd(2**8)[:10]
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        >>> ArmaFft([1, -0.5], [1., 0.4], 40).acovf(10)
        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,
                0.045   ,  0.0225  ,  0.01125 ,  0.005625])
        '''
        hw = self.fftarma(n)
        return np.real_if_close(fft.ifft(hw*hw.conj()), tol=200)[:n]

    def spdmapoly(self, w, twosided=False):
        '''ma only, need division for ar, use LagPolynomial
        '''
        if w is None:
            w = np.linspace(0, np.pi, nfreq)
        return 0.5 / np.pi * self.mapoly(np.exp(w*1j))


    def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
        """Plot results"""
        rvs = self.generate_sample(nsample=100, burnin=500)
        acf = self.acf(nacf)[:nacf]  #TODO: check return length
        pacf = self.pacf(nacf)
        w = np.linspace(0, np.pi, nfreq)
        spdr, wr = self.spdroots(w)

        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        ax.plot(rvs)
        ax.set_title('Random Sample \nar=%s, ma=%s' % (self.ar, self.ma))

        ax = fig.add_subplot(2,2,2)
        ax.plot(acf)
        ax.set_title('Autocorrelation \nar=%s, ma=%rs' % (self.ar, self.ma))

        ax = fig.add_subplot(2,2,3)
        ax.plot(wr, spdr)
        ax.set_title('Power Spectrum \nar=%s, ma=%s' % (self.ar, self.ma))

        ax = fig.add_subplot(2,2,4)
        ax.plot(pacf)
        ax.set_title('Partial Autocorrelation \nar=%s, ma=%s' % (self.ar, self.ma))

        return fig







def spdar1(ar, w):
    if np.ndim(ar) == 0:
        rho = ar
    else:
        rho = -ar[1]
    return 0.5 / np.pi /(1 + rho*rho - 2 * rho * np.cos(w))

if __name__ == '__main__':
    def maxabs(x,y):
        return np.max(np.abs(x-y))
    nobs = 200  #10000
    ar = [1, 0.0]
    ma = [1, 0.0]
    ar2 = np.zeros(nobs)
    ar2[:2] = [1, -0.9]



    uni = np.zeros(nobs)
    uni[0]=1.
    #arrep = signal.lfilter(ma, ar, ar2)
    #marep = signal.lfilter([1],arrep, uni)
    # same faster:
    arcomb = np.convolve(ar, ar2, mode='same')
    marep = signal.lfilter(ma,arcomb, uni) #[len(ma):]
    print(marep[:10])
    mafr = fft.fft(marep)

    rvs = np.random.normal(size=nobs)
    datafr = fft.fft(rvs)
    y = fft.ifft(mafr*datafr)
    print(np.corrcoef(np.c_[y[2:], y[1:-1], y[:-2]],rowvar=0))

    arrep = signal.lfilter([1],marep, uni)
    print(arrep[:20])  # roundtrip to ar
    arfr = fft.fft(arrep)
    yfr = fft.fft(y)
    x = fft.ifft(arfr*yfr).real  #imag part is e-15
    # the next two are equal, roundtrip works
    print(x[:5])
    print(rvs[:5])
    print(np.corrcoef(np.c_[x[2:], x[1:-1], x[:-2]],rowvar=0))


    # ARMA filter using fft with ratio of fft of ma/ar lag polynomial
    # seems much faster than using lfilter

    #padding, note arcomb is already full length
    arcombp = np.zeros(nobs)
    arcombp[:len(arcomb)] = arcomb
    map_ = np.zeros(nobs)    #rename: map was shadowing builtin
    map_[:len(ma)] = ma
    ar0fr = fft.fft(arcombp)
    ma0fr = fft.fft(map_)
    y2 = fft.ifft(ma0fr/ar0fr*datafr)
    #the next two are (almost) equal in real part, almost zero but different in imag
    print(y2[:10])
    print(y[:10])
    print(maxabs(y, y2))  # from chfdiscrete
    #1.1282071239631782e-014

    ar = [1, -0.4]
    ma = [1, 0.2]

    arma1 = ArmaFft([1, -0.5,0,0,0,00, -0.7, 0.3], [1, 0.8], nobs)

    nfreq = nobs
    w = np.linspace(0, np.pi, nfreq)
    w2 = np.linspace(0, 2*np.pi, nfreq)

    import matplotlib.pyplot as plt
    plt.close('all')

    plt.figure()
    spd1, w1 = arma1.spd(2**10)
    print(spd1.shape)
    _ = plt.plot(spd1)
    plt.title('spd fft complex')

    plt.figure()
    spd2, w2 = arma1.spdshift(2**10)
    print(spd2.shape)
    _ = plt.plot(w2, spd2)
    plt.title('spd fft shift')

    plt.figure()
    spd3, w3 = arma1.spddirect(2**10)
    print(spd3.shape)
    _ = plt.plot(w3, spd3)
    plt.title('spd fft direct')

    plt.figure()
    spd3b = arma1._spddirect2(2**10)
    print(spd3b.shape)
    _ = plt.plot(spd3b)
    plt.title('spd fft direct mirrored')

    plt.figure()
    spdr, wr = arma1.spdroots(w)
    print(spdr.shape)
    plt.plot(w, spdr)
    plt.title('spd from roots')

    plt.figure()
    spdar1_ = spdar1(arma1.ar, w)
    print(spdar1_.shape)
    _ = plt.plot(w, spdar1_)
    plt.title('spd ar1')


    plt.figure()
    wper, spdper = arma1.periodogram(nfreq)
    print(spdper.shape)
    _ = plt.plot(w, spdper)
    plt.title('periodogram')

    startup = 1000
    rvs = arma1.generate_sample(startup+10000)[startup:]
    import matplotlib.mlab as mlb
    plt.figure()
    sdm, wm = mlb.psd(x)
    print('sdm.shape', sdm.shape)
    sdm = sdm.ravel()
    plt.plot(wm, sdm)
    plt.title('matplotlib')

    from nitime.algorithms import LD_AR_est
    #yule_AR_est(s, order, Nfreqs)
    wnt, spdnt = LD_AR_est(rvs, 10, 512)
    plt.figure()
    print('spdnt.shape', spdnt.shape)
    _ = plt.plot(spdnt.ravel())
    print(spdnt[:10])
    plt.title('nitime')

    fig = plt.figure()
    arma1.plot4(fig)

    #plt.show()
