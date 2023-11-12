"""ARMA process and estimation with scipy.signal.lfilter

Notes
-----
* written without textbook, works but not sure about everything
  briefly checked and it looks to be standard least squares, see below

* theoretical autocorrelation function of general ARMA
  Done, relatively easy to guess solution, time consuming to get
  theoretical test cases, example file contains explicit formulas for
  acovf of MA(1), MA(2) and ARMA(1,1)

Properties:
Judge, ... (1985): The Theory and Practise of Econometrics

Author: josefpktd
License: BSD
"""
import warnings

from statsmodels.compat.pandas import Appender

import numpy as np
from scipy import linalg, optimize, signal

from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like

__all__ = [
    "arma_acf",
    "arma_acovf",
    "arma_generate_sample",
    "arma_impulse_response",
    "arma2ar",
    "arma2ma",
    "deconvolve",
    "lpol2index",
    "index2lpol",
]


NONSTATIONARY_ERROR = """\
The model's autoregressive parameters (ar) indicate that the process
 is non-stationary. arma_acovf can only be used with stationary processes.
"""


def arma_generate_sample(
    ar, ma, nsample, scale=1, distrvs=None, axis=0, burnin=0
):
    """
    Simulate data from an ARMA.

    Parameters
    ----------
    ar : array_like
        The coefficient for autoregressive lag polynomial, including zero lag.
    ma : array_like
        The coefficient for moving-average lag polynomial, including zero lag.
    nsample : int or tuple of ints
        If nsample is an integer, then this creates a 1d timeseries of
        length size. If nsample is a tuple, creates a len(nsample)
        dimensional time series where time is indexed along the input
        variable ``axis``. All series are unless ``distrvs`` generates
        dependent data.
    scale : float
        The standard deviation of noise.
    distrvs : function, random number generator
        A function that generates the random numbers, and takes ``size``
        as argument. The default is np.random.standard_normal.
    axis : int
        See nsample for details.
    burnin : int
        Number of observation at the beginning of the sample to drop.
        Used to reduce dependence on initial values.

    Returns
    -------
    ndarray
        Random sample(s) from an ARMA process.

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR parameters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARIMA(y, (2, 0, 2), trend='n').fit(disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    distrvs = np.random.standard_normal if distrvs is None else distrvs
    if np.ndim(nsample) == 0:
        nsample = [nsample]
    if burnin:
        # handle burin time for nd arrays
        # maybe there is a better trick in scipy.fft code
        newsize = list(nsample)
        newsize[axis] += burnin
        newsize = tuple(newsize)
        fslice = [slice(None)] * len(newsize)
        fslice[axis] = slice(burnin, None, None)
        fslice = tuple(fslice)
    else:
        newsize = tuple(nsample)
        fslice = tuple([slice(None)] * np.ndim(newsize))
    eta = scale * distrvs(size=newsize)
    return signal.lfilter(ma, ar, eta, axis=axis)[fslice]


def arma_acovf(ar, ma, nobs=10, sigma2=1, dtype=None):
    """
    Theoretical autocovariances of stationary ARMA processes

    Parameters
    ----------
    ar : array_like, 1d
        The coefficients for autoregressive lag polynomial, including zero lag.
    ma : array_like, 1d
        The coefficients for moving-average lag polynomial, including zero lag.
    nobs : int
        The number of terms (lags plus zero lag) to include in returned acovf.
    sigma2 : float
        Variance of the innovation term.

    Returns
    -------
    ndarray
        The autocovariance of ARMA process given by ar, ma.

    See Also
    --------
    arma_acf : Autocorrelation function for ARMA processes.
    acovf : Sample autocovariance estimation.

    References
    ----------
    .. [*] Brockwell, Peter J., and Richard A. Davis. 2009. Time Series:
        Theory and Methods. 2nd ed. 1991. New York, NY: Springer.
    """
    if dtype is None:
        dtype = np.common_type(np.array(ar), np.array(ma), np.array(sigma2))

    p = len(ar) - 1
    q = len(ma) - 1
    m = max(p, q) + 1

    if sigma2.real < 0:
        raise ValueError("Must have positive innovation variance.")

    # Short-circuit for trivial corner-case
    if p == q == 0:
        out = np.zeros(nobs, dtype=dtype)
        out[0] = sigma2
        return out
    elif p > 0 and np.max(np.abs(np.roots(ar))) >= 1:
        raise ValueError(NONSTATIONARY_ERROR)

    # Get the moving average representation coefficients that we need
    ma_coeffs = arma2ma(ar, ma, lags=m)

    # Solve for the first m autocovariances via the linear system
    # described by (BD, eq. 3.3.8)
    A = np.zeros((m, m), dtype=dtype)
    b = np.zeros((m, 1), dtype=dtype)
    # We need a zero-right-padded version of ar params
    tmp_ar = np.zeros(m, dtype=dtype)
    tmp_ar[: p + 1] = ar
    for k in range(m):
        A[k, : (k + 1)] = tmp_ar[: (k + 1)][::-1]
        A[k, 1 : m - k] += tmp_ar[(k + 1) : m]
        b[k] = sigma2 * np.dot(ma[k : q + 1], ma_coeffs[: max((q + 1 - k), 0)])
    acovf = np.zeros(max(nobs, m), dtype=dtype)
    try:
        acovf[:m] = np.linalg.solve(A, b)[:, 0]
    except np.linalg.LinAlgError:
        raise ValueError(NONSTATIONARY_ERROR)

    # Iteratively apply (BD, eq. 3.3.9) to solve for remaining autocovariances
    if nobs > m:
        zi = signal.lfiltic([1], ar, acovf[:m:][::-1])
        acovf[m:] = signal.lfilter(
            [1], ar, np.zeros(nobs - m, dtype=dtype), zi=zi
        )[0]

    return acovf[:nobs]


def arma_acf(ar, ma, lags=10):
    """
    Theoretical autocorrelation function of an ARMA process.

    Parameters
    ----------
    ar : array_like
        Coefficients for autoregressive lag polynomial, including zero lag.
    ma : array_like
        Coefficients for moving-average lag polynomial, including zero lag.
    lags : int
        The number of terms (lags plus zero lag) to include in returned acf.

    Returns
    -------
    ndarray
        The autocorrelations of ARMA process given by ar and ma.

    See Also
    --------
    arma_acovf : Autocovariances from ARMA processes.
    acf : Sample autocorrelation function estimation.
    acovf : Sample autocovariance function estimation.
    """
    acovf = arma_acovf(ar, ma, lags)
    return acovf / acovf[0]


def arma_pacf(ar, ma, lags=10):
    """
    Theoretical partial autocorrelation function of an ARMA process.

    Parameters
    ----------
    ar : array_like, 1d
        The coefficients for autoregressive lag polynomial, including zero lag.
    ma : array_like, 1d
        The coefficients for moving-average lag polynomial, including zero lag.
    lags : int
        The number of terms (lags plus zero lag) to include in returned pacf.

    Returns
    -------
    ndarrray
        The partial autocorrelation of ARMA process given by ar and ma.

    Notes
    -----
    Solves yule-walker equation for each lag order up to nobs lags.

    not tested/checked yet
    """
    # TODO: Should use rank 1 inverse update
    apacf = np.zeros(lags)
    acov = arma_acf(ar, ma, lags=lags + 1)

    apacf[0] = 1.0
    for k in range(2, lags + 1):
        r = acov[:k]
        apacf[k - 1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf


def arma_periodogram(ar, ma, worN=None, whole=0):
    """
    Periodogram for ARMA process given by lag-polynomials ar and ma.

    Parameters
    ----------
    ar : array_like
        The autoregressive lag-polynomial with leading 1 and lhs sign.
    ma : array_like
        The moving average lag-polynomial with leading 1.
    worN : {None, int}, optional
        An option for scipy.signal.freqz (read "w or N").
        If None, then compute at 512 frequencies around the unit circle.
        If a single integer, the compute at that many frequencies.
        Otherwise, compute the response at frequencies given in worN.
    whole : {0,1}, optional
        An options for scipy.signal.freqz/
        Normally, frequencies are computed from 0 to pi (upper-half of
        unit-circle.  If whole is non-zero compute frequencies from 0 to 2*pi.

    Returns
    -------
    w : ndarray
        The frequencies.
    sd : ndarray
        The periodogram, also known as the spectral density.

    Notes
    -----
    Normalization ?

    This uses signal.freqz, which does not use fft. There is a fft version
    somewhere.
    """
    w, h = signal.freqz(ma, ar, worN=worN, whole=whole)
    sd = np.abs(h) ** 2 / np.sqrt(2 * np.pi)
    if np.any(np.isnan(h)):
        # this happens with unit root or seasonal unit root'
        import warnings

        warnings.warn(
            "Warning: nan in frequency response h, maybe a unit " "root",
            RuntimeWarning,
            stacklevel=2,
        )
    return w, sd


def arma_impulse_response(ar, ma, leads=100):
    """
    Compute the impulse response function (MA representation) for ARMA process.

    Parameters
    ----------
    ar : array_like, 1d
        The auto regressive lag polynomial.
    ma : array_like, 1d
        The moving average lag polynomial.
    leads : int
        The number of observations to calculate.

    Returns
    -------
    ndarray
        The impulse response function with nobs elements.

    Notes
    -----
    This is the same as finding the MA representation of an ARMA(p,q).
    By reversing the role of ar and ma in the function arguments, the
    returned result is the AR representation of an ARMA(p,q), i.e

    ma_representation = arma_impulse_response(ar, ma, leads=100)
    ar_representation = arma_impulse_response(ma, ar, leads=100)

    Fully tested against matlab

    Examples
    --------
    AR(1)

    >>> arma_impulse_response([1.0, -0.8], [1.], leads=10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    this is the same as

    >>> 0.8**np.arange(10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    MA(2)

    >>> arma_impulse_response([1.0], [1., 0.5, 0.2], leads=10)
    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

    ARMA(1,2)

    >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], leads=10)
    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])
    """
    impulse = np.zeros(leads)
    impulse[0] = 1.0
    return signal.lfilter(ma, ar, impulse)


def arma2ma(ar, ma, lags=100):
    """
    A finite-lag approximate MA representation of an ARMA process.

    Parameters
    ----------
    ar : ndarray
        The auto regressive lag polynomial.
    ma : ndarray
        The moving average lag polynomial.
    lags : int
        The number of coefficients to calculate.

    Returns
    -------
    ndarray
        The coefficients of AR lag polynomial with nobs elements.

    Notes
    -----
    Equivalent to ``arma_impulse_response(ma, ar, leads=100)``
    """
    return arma_impulse_response(ar, ma, leads=lags)


def arma2ar(ar, ma, lags=100):
    """
    A finite-lag AR approximation of an ARMA process.

    Parameters
    ----------
    ar : array_like
        The auto regressive lag polynomial.
    ma : array_like
        The moving average lag polynomial.
    lags : int
        The number of coefficients to calculate.

    Returns
    -------
    ndarray
        The coefficients of AR lag polynomial with nobs elements.

    Notes
    -----
    Equivalent to ``arma_impulse_response(ma, ar, leads=100)``
    """
    return arma_impulse_response(ma, ar, leads=lags)


# moved from sandbox.tsa.try_fi
def ar2arma(ar_des, p, q, n=20, mse="ar", start=None):
    """
    Find arma approximation to ar process.

    This finds the ARMA(p,q) coefficients that minimize the integrated
    squared difference between the impulse_response functions (MA
    representation) of the AR and the ARMA process. This does not  check
    whether the MA lag polynomial of the ARMA process is invertible, neither
    does it check the roots of the AR lag polynomial.

    Parameters
    ----------
    ar_des : array_like
        The coefficients of original AR lag polynomial, including lag zero.
    p : int
        The length of desired AR lag polynomials.
    q : int
        The length of desired MA lag polynomials.
    n : int
        The number of terms of the impulse_response function to include in the
        objective function for the approximation.
    mse : str, 'ar'
        Not used.
    start : ndarray
        Initial values to use when finding the approximation.

    Returns
    -------
    ar_app : ndarray
        The coefficients of the AR lag polynomials of the approximation.
    ma_app : ndarray
        The coefficients of the MA lag polynomials of the approximation.
    res : tuple
        The result of optimize.leastsq.

    Notes
    -----
    Extension is possible if we want to match autocovariance instead
    of impulse response function.
    """

    # TODO: convert MA lag polynomial, ma_app, to be invertible, by mirroring
    # TODO: roots outside the unit interval to ones that are inside. How to do
    # TODO: this?

    # p,q = pq
    def msear_err(arma, ar_des):
        ar, ma = np.r_[1, arma[: p - 1]], np.r_[1, arma[p - 1 :]]
        ar_approx = arma_impulse_response(ma, ar, n)
        return ar_des - ar_approx  # ((ar - ar_approx)**2).sum()

    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p - 1), np.zeros(q - 1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)
    arma_app = np.atleast_1d(res[0])
    ar_app = (np.r_[1, arma_app[: p - 1]],)
    ma_app = np.r_[1, arma_app[p - 1 :]]
    return ar_app, ma_app, res


_arma_docs = {"ar": arma2ar.__doc__, "ma": arma2ma.__doc__}


def lpol2index(ar):
    """
    Remove zeros from lag polynomial

    Parameters
    ----------
    ar : array_like
        coefficients of lag polynomial

    Returns
    -------
    coeffs : ndarray
        non-zero coefficients of lag polynomial
    index : ndarray
        index (lags) of lag polynomial with non-zero elements
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.ComplexWarning)
        ar = array_like(ar, "ar")
    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return coeffs, index


def index2lpol(coeffs, index):
    """
    Expand coefficients to lag poly

    Parameters
    ----------
    coeffs : ndarray
        non-zero coefficients of lag polynomial
    index : ndarray
        index (lags) of lag polynomial with non-zero elements

    Returns
    -------
    ar : array_like
        coefficients of lag polynomial
    """
    n = max(index)
    ar = np.zeros(n + 1)
    ar[index] = coeffs
    return ar


def lpol_fima(d, n=20):
    """MA representation of fractional integration

    .. math:: (1-L)^{-d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ma : ndarray
        coefficients of lag polynomial
    """
    # hide import inside function until we use this heavily
    from scipy.special import gammaln

    j = np.arange(n)
    return np.exp(gammaln(d + j) - gammaln(j + 1) - gammaln(d))


# moved from sandbox.tsa.try_fi
def lpol_fiar(d, n=20):
    """AR representation of fractional integration

    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ar : ndarray
        coefficients of lag polynomial

    Notes:
    first coefficient is 1, negative signs except for first term,
    ar(L)*x_t
    """
    # hide import inside function until we use this heavily
    from scipy.special import gammaln

    j = np.arange(n)
    ar = -np.exp(gammaln(-d + j) - gammaln(j + 1) - gammaln(-d))
    ar[0] = 1
    return ar


# moved from sandbox.tsa.try_fi
def lpol_sdiff(s):
    """return coefficients for seasonal difference (1-L^s)

    just a trivial convenience function

    Parameters
    ----------
    s : int
        number of periods in season

    Returns
    -------
    sdiff : list, length s+1
    """
    return [1] + [0] * (s - 1) + [-1]


def deconvolve(num, den, n=None):
    """Deconvolves divisor out of signal, division of polynomials for n terms

    calculates den^{-1} * num

    Parameters
    ----------
    num : array_like
        signal or lag polynomial
    denom : array_like
        coefficients of lag polynomial (linear filter)
    n : None or int
        number of terms of quotient

    Returns
    -------
    quot : ndarray
        quotient or filtered series
    rem : ndarray
        remainder

    Notes
    -----
    If num is a time series, then this applies the linear filter den^{-1}.
    If both num and den are both lag polynomials, then this calculates the
    quotient polynomial for n terms and also returns the remainder.

    This is copied from scipy.signal.signaltools and added n as optional
    parameter.
    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    N = len(num)
    D = len(den)
    if D > N and n is None:
        quot = []
        rem = num
    else:
        if n is None:
            n = N - D + 1
        input = np.zeros(n, float)
        input[0] = 1
        quot = signal.lfilter(num, den, input)
        num_approx = signal.convolve(den, quot, mode="full")
        if len(num) < len(num_approx):  # 1d only ?
            num = np.concatenate((num, np.zeros(len(num_approx) - len(num))))
        rem = num - num_approx
    return quot, rem


_generate_sample_doc = Docstring(arma_generate_sample.__doc__)
_generate_sample_doc.remove_parameters(["ar", "ma"])
_generate_sample_doc.replace_block("Notes", [])
_generate_sample_doc.replace_block("Examples", [])


class ArmaProcess:
    r"""
    Theoretical properties of an ARMA process for specified lag-polynomials.

    Parameters
    ----------
    ar : array_like
        Coefficient for autoregressive lag polynomial, including zero lag.
        Must be entered using the signs from the lag polynomial representation.
        See the notes for more information about the sign.
    ma : array_like
        Coefficient for moving-average lag polynomial, including zero lag.
    nobs : int, optional
        Length of simulated time series. Used, for example, if a sample is
        generated. See example.

    Notes
    -----
    Both the AR and MA components must include the coefficient on the
    zero-lag. In almost all cases these values should be 1. Further, due to
    using the lag-polynomial representation, the AR parameters should
    have the opposite sign of what one would write in the ARMA representation.
    See the examples below.

    The ARMA(p,q) process is described by

    .. math::

        y_{t}=\phi_{1}y_{t-1}+\ldots+\phi_{p}y_{t-p}+\theta_{1}\epsilon_{t-1}
               +\ldots+\theta_{q}\epsilon_{t-q}+\epsilon_{t}

    and the parameterization used in this function uses the lag-polynomial
    representation,

    .. math::

        \left(1-\phi_{1}L-\ldots-\phi_{p}L^{p}\right)y_{t} =
            \left(1+\theta_{1}L+\ldots+\theta_{q}L^{q}\right)\epsilon_{t}

    Examples
    --------
    ARMA(2,2) with AR coefficients 0.75 and -0.25, and MA coefficients 0.65 and 0.35

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARIMA(y, (2, 0, 2), trend='n').fit(disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])

    The same ARMA(2,2) Using the from_coeffs class method

    >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    """

    # TODO: Check unit root behavior
    def __init__(self, ar=None, ma=None, nobs=100):
        if ar is None:
            ar = np.array([1.0])
        if ma is None:
            ma = np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            self.ar = array_like(ar, "ar")
            self.ma = array_like(ma, "ma")
        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]
        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)
        self.nobs = nobs

    @classmethod
    def from_roots(cls, maroots=None, arroots=None, nobs=100):
        """
        Create ArmaProcess from AR and MA polynomial roots.

        Parameters
        ----------
        maroots : array_like, optional
            Roots for the MA polynomial
            1 + theta_1*z + theta_2*z^2 + ..... + theta_n*z^n
        arroots : array_like, optional
            Roots for the AR polynomial
            1 - phi_1*z - phi_2*z^2 - ..... - phi_n*z^n
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arroots = [.75, -.25]
        >>> maroots = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_roots(arroots, maroots)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        if arroots is not None and len(arroots):
            arpoly = np.polynomial.polynomial.Polynomial.fromroots(arroots)
            arcoefs = arpoly.coef[1:] / arpoly.coef[0]
        else:
            arcoefs = []

        if maroots is not None and len(maroots):
            mapoly = np.polynomial.polynomial.Polynomial.fromroots(maroots)
            macoefs = mapoly.coef[1:] / mapoly.coef[0]
        else:
            macoefs = []

        # As from_coeffs will create a polynomial with constant 1/-1,(MA/AR)
        # we need to scale the polynomial coefficients accordingly
        return cls(np.r_[1, arcoefs], np.r_[1, macoefs], nobs=nobs)

    @classmethod
    def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
        """
        Create ArmaProcess from an ARMA representation.

        Parameters
        ----------
        arcoefs : array_like
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information.
        macoefs : array_like
            Coefficient for moving-average lag polynomial, excluding zero lag.
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arparams = [.75, -.25]
        >>> maparams = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(ar, ma)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        arcoefs = [] if arcoefs is None else arcoefs
        macoefs = [] if macoefs is None else macoefs
        return cls(
            np.r_[1, -np.asarray(arcoefs)],
            np.r_[1, np.asarray(macoefs)],
            nobs=nobs,
        )

    @classmethod
    def from_estimation(cls, model_results, nobs=None):
        """
        Create an ArmaProcess from the results of an ARIMA estimation.

        Parameters
        ----------
        model_results : ARIMAResults instance
            A fitted model.
        nobs : int, optional
            If None, nobs is taken from the results.

        Returns
        -------
        ArmaProcess
            Class instance initialized from model_results.

        See Also
        --------
        statsmodels.tsa.arima.model.ARIMA
            The models class used to create the ArmaProcess
        """
        nobs = nobs or model_results.nobs
        return cls(
            model_results.polynomial_reduced_ar,
            model_results.polynomial_reduced_ma,
            nobs=nobs,
        )

    def __mul__(self, oth):
        if isinstance(oth, self.__class__):
            ar = (self.arpoly * oth.arpoly).coef
            ma = (self.mapoly * oth.mapoly).coef
        else:
            try:
                aroth, maoth = oth
                arpolyoth = np.polynomial.Polynomial(aroth)
                mapolyoth = np.polynomial.Polynomial(maoth)
                ar = (self.arpoly * arpolyoth).coef
                ma = (self.mapoly * mapolyoth).coef
            except:
                raise TypeError("Other type is not a valid type")
        return self.__class__(ar, ma, nobs=self.nobs)

    def __repr__(self):
        msg = "ArmaProcess({0}, {1}, nobs={2}) at {3}"
        return msg.format(
            self.ar.tolist(), self.ma.tolist(), self.nobs, hex(id(self))
        )

    def __str__(self):
        return "ArmaProcess\nAR: {0}\nMA: {1}".format(
            self.ar.tolist(), self.ma.tolist()
        )

    @Appender(remove_parameters(arma_acovf.__doc__, ["ar", "ma", "sigma2"]))
    def acovf(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    @Appender(remove_parameters(arma_acf.__doc__, ["ar", "ma"]))
    def acf(self, lags=None):
        lags = lags or self.nobs
        return arma_acf(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma_pacf.__doc__, ["ar", "ma"]))
    def pacf(self, lags=None):
        lags = lags or self.nobs
        return arma_pacf(self.ar, self.ma, lags=lags)

    @Appender(
        remove_parameters(
            arma_periodogram.__doc__, ["ar", "ma", "worN", "whole"]
        )
    )
    def periodogram(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_periodogram(self.ar, self.ma, worN=nobs)

    @Appender(remove_parameters(arma_impulse_response.__doc__, ["ar", "ma"]))
    def impulse_response(self, leads=None):
        leads = leads or self.nobs
        return arma_impulse_response(self.ar, self.ma, leads=leads)

    @Appender(remove_parameters(arma2ma.__doc__, ["ar", "ma"]))
    def arma2ma(self, lags=None):
        lags = lags or self.lags
        return arma2ma(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma2ar.__doc__, ["ar", "ma"]))
    def arma2ar(self, lags=None):
        lags = lags or self.lags
        return arma2ar(self.ar, self.ma, lags=lags)

    @property
    def arroots(self):
        """Roots of autoregressive lag-polynomial"""
        return self.arpoly.roots()

    @property
    def maroots(self):
        """Roots of moving average lag-polynomial"""
        return self.mapoly.roots()

    @property
    def isstationary(self):
        """
        Arma process is stationary if AR roots are outside unit circle.

        Returns
        -------
        bool
             True if autoregressive roots are outside unit circle.
        """
        if np.all(np.abs(self.arroots) > 1.0):
            return True
        else:
            return False

    @property
    def isinvertible(self):
        """
        Arma process is invertible if MA roots are outside unit circle.

        Returns
        -------
        bool
             True if moving average roots are outside unit circle.
        """
        if np.all(np.abs(self.maroots) > 1):
            return True
        else:
            return False

    def invertroots(self, retnew=False):
        """
        Make MA polynomial invertible by inverting roots inside unit circle.

        Parameters
        ----------
        retnew : bool
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial.

        Returns
        -------
        manew : ndarray
           A new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : bool
           True if the MA lag-polynomial was already invertible, returned if
           retnew is false.
        armaprocess : new instance of class
           If retnew is true, then return a new instance with invertible
           MA-polynomial.
        """
        # TODO: variable returns like this?
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1.0 / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]

        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
        else:
            return mainv, invertible

    @Appender(str(_generate_sample_doc))
    def generate_sample(
        self, nsample=100, scale=1.0, distrvs=None, axis=0, burnin=0
    ):
        return arma_generate_sample(
            self.ar, self.ma, nsample, scale, distrvs, axis=axis, burnin=burnin
        )
