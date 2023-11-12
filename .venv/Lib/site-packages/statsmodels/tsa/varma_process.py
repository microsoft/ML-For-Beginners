# -*- coding: utf-8 -*-
""" Helper and filter functions for VAR and VARMA, and basic VAR class

Created on Mon Jan 11 11:04:23 2010
Author: josef-pktd
License: BSD

This is a new version, I did not look at the old version again, but similar
ideas.

not copied/cleaned yet:
 * fftn based filtering, creating samples with fft
 * Tests: I ran examples but did not convert them to tests
   examples look good for parameter estimate and forecast, and filter functions

main TODOs:
* result statistics
* see whether Bayesian dummy observation can be included without changing
  the single call to linalg.lstsq
* impulse response function does not treat correlation, see Hamilton and jplv

Extensions
* constraints, Bayesian priors/penalization
* Error Correction Form and Cointegration
* Factor Models Stock-Watson,  ???


see also VAR section in Notes.txt

"""
import numpy as np
from scipy import signal

from statsmodels.tsa.tsatools import lagmat


def varfilter(x, a):
    '''apply an autoregressive filter to a series x

    Warning: I just found out that convolve does not work as I
       thought, this likely does not work correctly for
       nvars>3


    x can be 2d, a can be 1d, 2d, or 3d

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    a : array_like
        autoregressive filter coefficients, ar lag polynomial
        see Notes

    Returns
    -------
    y : ndarray, 2d
        filtered array, number of columns determined by x and a

    Notes
    -----

    In general form this uses the linear filter ::

        y = a(L)x

    where
    x : nobs, nvars
    a : nlags, nvars, npoly

    Depending on the shape and dimension of a this uses different
    Lag polynomial arrays

    case 1 : a is 1d or (nlags,1)
        one lag polynomial is applied to all variables (columns of x)
    case 2 : a is 2d, (nlags, nvars)
        each series is independently filtered with its own
        lag polynomial, uses loop over nvar
    case 3 : a is 3d, (nlags, nvars, npoly)
        the ith column of the output array is given by the linear filter
        defined by the 2d array a[:,:,i], i.e. ::

            y[:,i] = a(.,.,i)(L) * x
            y[t,i] = sum_p sum_j a(p,j,i)*x(t-p,j)
                     for p = 0,...nlags-1, j = 0,...nvars-1,
                     for all t >= nlags


    Note: maybe convert to axis=1, Not

    TODO: initial conditions

    '''
    x = np.asarray(x)
    a = np.asarray(a)
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')
    nvar = x.shape[1]
    nlags = a.shape[0]
    ntrim = nlags//2
    # for x is 2d with ncols >1

    if a.ndim == 1:
        # case: identical ar filter (lag polynomial)
        return signal.convolve(x, a[:,None], mode='valid')
        # alternative:
        #return signal.lfilter(a,[1],x.astype(float),axis=0)
    elif a.ndim == 2:
        if min(a.shape) == 1:
            # case: identical ar filter (lag polynomial)
            return signal.convolve(x, a, mode='valid')

        # case: independent ar
        #(a bit like recserar in gauss, but no x yet)
        #(no, reserar is inverse filter)
        result = np.zeros((x.shape[0]-nlags+1, nvar))
        for i in range(nvar):
            # could also use np.convolve, but easier for swiching to fft
            result[:,i] = signal.convolve(x[:,i], a[:,i], mode='valid')
        return result

    elif a.ndim == 3:
        # case: vector autoregressive with lag matrices
        # Note: we must have shape[1] == shape[2] == nvar
        yf = signal.convolve(x[:,:,None], a)
        yvalid = yf[ntrim:-ntrim, yf.shape[1]//2,:]
        return yvalid


def varinversefilter(ar, nobs, version=1):
    '''creates inverse ar filter (MA representation) recursively

    The VAR lag polynomial is defined by ::

        ar(L) y_t = u_t  or
        y_t = -ar_{-1}(L) y_{t-1} + u_t

    the returned lagpolynomial is arinv(L)=ar^{-1}(L) in ::

        y_t = arinv(L) u_t



    Parameters
    ----------
    ar : ndarray, (nlags,nvars,nvars)
        matrix lagpolynomial, currently no exog
        first row should be identity

    Returns
    -------
    arinv : ndarray, (nobs,nvars,nvars)


    Notes
    -----

    '''
    nlags, nvars, nvarsex = ar.shape
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    arinv = np.zeros((nobs+1, nvarsex, nvars))
    arinv[0,:,:] = ar[0]
    arinv[1:nlags,:,:] = -ar[1:]
    if version == 1:
        for i in range(2,nobs+1):
            tmp = np.zeros((nvars,nvars))
            for p in range(1,nlags):
                tmp += np.dot(-ar[p],arinv[i-p,:,:])
            arinv[i,:,:] = tmp
    if version == 0:
        for i in range(nlags+1,nobs+1):
            print(ar[1:].shape, arinv[i-1:i-nlags:-1,:,:].shape)
            #arinv[i,:,:] = np.dot(-ar[1:],arinv[i-1:i-nlags:-1,:,:])
            #print(np.tensordot(-ar[1:],arinv[i-1:i-nlags:-1,:,:],axes=([2],[1])).shape
            #arinv[i,:,:] = np.tensordot(-ar[1:],arinv[i-1:i-nlags:-1,:,:],axes=([2],[1]))
            raise NotImplementedError('waiting for generalized ufuncs or something')

    return arinv


def vargenerate(ar, u, initvalues=None):
    '''generate an VAR process with errors u

    similar to gauss
    uses loop

    Parameters
    ----------
    ar : array (nlags,nvars,nvars)
        matrix lagpolynomial
    u : array (nobs,nvars)
        exogenous variable, error term for VAR

    Returns
    -------
    sar : array (1+nobs,nvars)
        sample of var process, inverse filtered u
        does not trim initial condition y_0 = 0

    Examples
    --------
    # generate random sample of VAR
    nobs, nvars = 10, 2
    u = numpy.random.randn(nobs,nvars)
    a21 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0. ],
                     [ 0.,  -0.6]]])
    vargenerate(a21,u)

    # Impulse Response to an initial shock to the first variable
    imp = np.zeros((nobs, nvars))
    imp[0,0] = 1
    vargenerate(a21,imp)

    '''
    nlags, nvars, nvarsex = ar.shape
    nlagsm1 = nlags - 1
    nobs = u.shape[0]
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    if u.shape[1] != nvars:
        raise ValueError('u needs to have nvars columns')
    if initvalues is None:
        sar = np.zeros((nobs+nlagsm1, nvars))
        start = nlagsm1
    else:
        start = max(nlagsm1, initvalues.shape[0])
        sar = np.zeros((nobs+start, nvars))
        sar[start-initvalues.shape[0]:start] = initvalues
    #sar[nlagsm1:] = u
    sar[start:] = u
    #if version == 1:
    for i in range(start,start+nobs):
        for p in range(1,nlags):
            sar[i] += np.dot(sar[i-p,:],-ar[p])

    return sar


def padone(x, front=0, back=0, axis=0, fillvalue=0):
    '''pad with zeros along one axis, currently only axis=0


    can be used sequentially to pad several axis

    Examples
    --------
    >>> padone(np.ones((2,3)),1,3,axis=1)
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])

    >>> padone(np.ones((2,3)),1,1, fillvalue=np.nan)
    array([[ NaN,  NaN,  NaN],
           [  1.,   1.,   1.],
           [  1.,   1.,   1.],
           [ NaN,  NaN,  NaN]])
    '''
    #primitive version
    shape = np.array(x.shape)
    shape[axis] += (front + back)
    shapearr = np.array(x.shape)
    out = np.empty(shape)
    out.fill(fillvalue)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shapearr
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    #print(myslice
    #print(out.shape
    #print(out[tuple(myslice)].shape
    out[tuple(myslice)] = x
    return out


def trimone(x, front=0, back=0, axis=0):
    '''trim number of array elements along one axis


    Examples
    --------
    >>> xp = padone(np.ones((2,3)),1,3,axis=1)
    >>> xp
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])
    >>> trimone(xp,1,3,1)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    '''
    shape = np.array(x.shape)
    shape[axis] -= (front + back)
    #print(shape, front, back
    shapearr = np.array(x.shape)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    #print(myslice
    #print(shape, endind
    #print(x[tuple(myslice)].shape
    return x[tuple(myslice)]


def ar2full(ar):
    '''make reduced lagpolynomial into a right side lagpoly array
    '''
    nlags, nvar,nvarex = ar.shape
    return np.r_[np.eye(nvar,nvarex)[None,:,:],-ar]


def ar2lhs(ar):
    '''convert full (rhs) lagpolynomial into a reduced, left side lagpoly array

    this is mainly a reminder about the definition
    '''
    return -ar[1:]


class _Var:
    '''obsolete VAR class, use tsa.VAR instead, for internal use only


    Examples
    --------

    >>> v = Var(ar2s)
    >>> v.fit(1)
    >>> v.arhat
    array([[[ 1.        ,  0.        ],
            [ 0.        ,  1.        ]],

           [[-0.77784898,  0.01726193],
            [ 0.10733009, -0.78665335]]])

    '''

    def __init__(self, y):
        self.y = y
        self.nobs, self.nvars = y.shape

    def fit(self, nlags):
        '''estimate parameters using ols

        Parameters
        ----------
        nlags : int
            number of lags to include in regression, same for all variables

        Returns
        -------
        None, but attaches

        arhat : array (nlags, nvar, nvar)
            full lag polynomial array
        arlhs : array (nlags-1, nvar, nvar)
            reduced lag polynomial for left hand side
        other statistics as returned by linalg.lstsq : need to be completed



        This currently assumes all parameters are estimated without restrictions.
        In this case SUR is identical to OLS

        estimation results are attached to the class instance


        '''
        self.nlags = nlags # without current period
        nvars = self.nvars
        #TODO: ar2s looks like a module variable, bug?
        #lmat = lagmat(ar2s, nlags, trim='both', original='in')
        lmat = lagmat(self.y, nlags, trim='both', original='in')
        self.yred = lmat[:,:nvars]
        self.xred = lmat[:,nvars:]
        res = np.linalg.lstsq(self.xred, self.yred, rcond=-1)
        self.estresults = res
        self.arlhs = res[0].reshape(nlags, nvars, nvars)
        self.arhat = ar2full(self.arlhs)
        self.rss = res[1]
        self.xredrank = res[2]

    def predict(self):
        '''calculate estimated timeseries (yhat) for sample

        '''

        if not hasattr(self, 'yhat'):
            self.yhat = varfilter(self.y, self.arhat)
        return self.yhat

    def covmat(self):
        ''' covariance matrix of estimate
        # not sure it's correct, need to check orientation everywhere
        # looks ok, display needs getting used to
        >>> v.rss[None,None,:]*np.linalg.inv(np.dot(v.xred.T,v.xred))[:,:,None]
        array([[[ 0.37247445,  0.32210609],
                [ 0.1002642 ,  0.08670584]],

               [[ 0.1002642 ,  0.08670584],
                [ 0.45903637,  0.39696255]]])
        >>>
        >>> v.rss[0]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.37247445,  0.1002642 ],
               [ 0.1002642 ,  0.45903637]])
        >>> v.rss[1]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.32210609,  0.08670584],
               [ 0.08670584,  0.39696255]])
       '''

        #check if orientation is same as self.arhat
        self.paramcov = (self.rss[None,None,:] *
            np.linalg.inv(np.dot(self.xred.T, self.xred))[:,:,None])

    def forecast(self, horiz=1, u=None):
        '''calculates forcast for horiz number of periods at end of sample

        Parameters
        ----------
        horiz : int (optional, default=1)
            forecast horizon
        u : array (horiz, nvars)
            error term for forecast periods. If None, then u is zero.

        Returns
        -------
        yforecast : array (nobs+horiz, nvars)
            this includes the sample and the forecasts
        '''
        if u is None:
            u = np.zeros((horiz, self.nvars))
        return vargenerate(self.arhat, u, initvalues=self.y)


class VarmaPoly:
    '''class to keep track of Varma polynomial format


    Examples
    --------

    ar23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    ma22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[ 0.4,  0. ],
                     [ 0.2, 0.3]]])


    '''
    def __init__(self, ar, ma=None):
        self.ar = ar
        self.ma = ma
        nlags, nvarall, nvars = ar.shape
        self.nlags, self.nvarall, self.nvars = nlags, nvarall, nvars
        self.isstructured = not (ar[0,:nvars] == np.eye(nvars)).all()
        if self.ma is None:
            self.ma = np.eye(nvars)[None,...]
            self.isindependent = True
        else:
            self.isindependent = not (ma[0] == np.eye(nvars)).all()
        self.malags = ar.shape[0]
        self.hasexog = nvarall > nvars
        self.arm1 = -ar[1:]

    #@property
    def vstack(self, a=None, name='ar'):
        '''stack lagpolynomial vertically in 2d array

        '''
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.reshape(-1, self.nvarall)

    #@property
    def hstack(self, a=None, name='ar'):
        '''stack lagpolynomial horizontally in 2d array

        '''
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.swapaxes(1,2).reshape(-1, self.nvarall).T

    #@property
    def stacksquare(self, a=None, name='ar', orientation='vertical'):
        '''stack lagpolynomial vertically in 2d square array with eye

        '''
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        astacked = a.reshape(-1, self.nvarall)
        lenpk, nvars = astacked.shape #[0]
        amat = np.eye(lenpk, k=nvars)
        amat[:,:nvars] = astacked
        return amat

    #@property
    def vstackarma_minus1(self):
        '''stack ar and lagpolynomial vertically in 2d array

        '''
        a = np.concatenate((self.ar[1:], self.ma[1:]),0)
        return a.reshape(-1, self.nvarall)

    #@property
    def hstackarma_minus1(self):
        '''stack ar and lagpolynomial vertically in 2d array

        this is the Kalman Filter representation, I think
        '''
        a = np.concatenate((self.ar[1:], self.ma[1:]),0)
        return a.swapaxes(1,2).reshape(-1, self.nvarall)

    def getisstationary(self, a=None):
        '''check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isstationary : bool

        *attaches*

        areigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        '''
        if a is not None:
            a = a
        else:
            if self.isstructured:
                a = -self.reduceform(self.ar)[1:]
            else:
                a = -self.ar[1:]
        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.areigenvalues = ev
        return (np.abs(ev) < 1).all()

    def getisinvertible(self, a=None):
        '''check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isinvertible : bool

        *attaches*

        maeigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        '''
        if a is not None:
            a = a
        else:
            if self.isindependent:
                a = self.reduceform(self.ma)[1:]
            else:
                a = self.ma[1:]
        if a.shape[0] == 0:
            # no ma lags
            self.maeigenvalues = np.array([], np.complex)
            return True

        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.maeigenvalues = ev
        return (np.abs(ev) < 1).all()

    def reduceform(self, apoly):
        '''

        this assumes no exog, todo

        '''
        if apoly.ndim != 3:
            raise ValueError('apoly needs to be 3d')
        nlags, nvarsex, nvars = apoly.shape

        a = np.empty_like(apoly)
        try:
            a0inv = np.linalg.inv(a[0,:nvars, :])
        except np.linalg.LinAlgError:
            raise ValueError('matrix not invertible',
                             'ask for implementation of pinv')

        for lag in range(nlags):
            a[lag] = np.dot(a0inv, apoly[lag])

        return a


if __name__ == "__main__":
    # some example lag polynomials
    a21 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0. ],
                     [ 0.,  -0.6]]])

    a22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0. ],
                     [ 0.1, -0.8]]])

    a23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0.2],
                     [ 0.1, -0.6]]])

    a24 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    a31 = np.r_[np.eye(3)[None,:,:], 0.8*np.eye(3)[None,:,:]]
    a32 = np.array([[[ 1. ,  0. ,  0. ],
                     [ 0. ,  1. ,  0. ],
                     [ 0. ,  0. ,  1. ]],

                    [[ 0.8,  0. ,  0. ],
                     [ 0.1,  0.6,  0. ],
                     [ 0. ,  0. ,  0.9]]])

    ########
    ut = np.random.randn(1000,2)
    ar2s = vargenerate(a22,ut)
    #res = np.linalg.lstsq(lagmat(ar2s,1)[:,1:], ar2s)
    res = np.linalg.lstsq(lagmat(ar2s,1), ar2s, rcond=-1)
    bhat = res[0].reshape(1,2,2)
    arhat = ar2full(bhat)
    #print(maxabs(arhat - a22)

    v = _Var(ar2s)
    v.fit(1)
    v.forecast()
    v.forecast(25)[-30:]

    ar23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    ma22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[ 0.4,  0. ],
                     [ 0.2, 0.3]]])

    ar23ns = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-1.9,  0. ],
                     [ 0.4, -0.6]],

                    [[ 0.3,  0. ],
                     [ 0.1, -0.1]]])

    vp = VarmaPoly(ar23, ma22)
    print(vars(vp))
    print(vp.vstack())
    print(vp.vstack(a24))
    print(vp.hstackarma_minus1())
    print(vp.getisstationary())
    print(vp.getisinvertible())

    vp2 = VarmaPoly(ar23ns)
    print(vp2.getisstationary())
    print(vp2.getisinvertible()) # no ma lags
