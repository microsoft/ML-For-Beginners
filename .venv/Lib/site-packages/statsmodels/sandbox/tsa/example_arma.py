'''trying to verify theoretical acf of arma

explicit functions for autocovariance functions of ARIMA(1,1), MA(1), MA(2)
plus 3 functions from nitime.utils

'''
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf

ar = [1., -0.6]
#ar = [1., 0.]
ma = [1., 0.4]
#ma = [1., 0.4, 0.6]
#ma = [1., 0.]
mod = ''#'ma2'
x = arma_generate_sample(ar, ma, 5000)
x_acf = acf(x)[:10]
x_ir = arma_impulse_response(ar, ma)

#print x_acf[:10]
#print x_ir[:10]
#irc2 = np.correlate(x_ir,x_ir,'full')[len(x_ir)-1:]
#print irc2[:10]
#print irc2[:10]/irc2[0]
#print irc2[:10-1] / irc2[1:10]
#print x_acf[:10-1] / x_acf[1:10]

# detrend helper from matplotlib.mlab
def detrend(x, key=None):
    if key is None or key=='constant':
        return detrend_mean(x)
    elif key=='linear':
        return detrend_linear(x)

def demean(x, axis=0):
    "Return x minus its mean along the specified axis"
    x = np.asarray(x)
    if axis:
        ind = [slice(None)] * axis
        ind.append(np.newaxis)
        return x - x.mean(axis)[ind]
    return x - x.mean(axis)

def detrend_mean(x):
    "Return x minus the mean(x)"
    return x - x.mean()

def detrend_none(x):
    "Return x: no detrending"
    return x

def detrend_linear(y):
    "Return y minus best fit line; 'linear' detrending "
    # This is faster than an algorithm based on linalg.lstsq.
    x = np.arange(len(y), dtype=np.float_)
    C = np.cov(x, y, bias=1)
    b = C[0,1]/C[0,0]
    a = y.mean() - b*x.mean()
    return y - (b*x + a)

def acovf_explicit(ar, ma, nobs):
    '''add correlation of MA representation explicitely

    '''
    ir = arma_impulse_response(ar, ma)
    acovfexpl = [np.dot(ir[:nobs-t], ir[t:nobs]) for t in range(10)]
    return acovfexpl

def acovf_arma11(ar, ma):
    # ARMA(1,1)
    # Florens et al page 278
    # wrong result ?
    # new calculation bigJudge p 311, now the same
    a = -ar[1]
    b = ma[1]
    #rho = [1.]
    #rho.append((1-a*b)*(a-b)/(1.+a**2-2*a*b))
    rho = [(1.+b**2+2*a*b)/(1.-a**2)]
    rho.append((1+a*b)*(a+b)/(1.-a**2))
    for _ in range(8):
        last = rho[-1]
        rho.append(a*last)
    return np.array(rho)

#    print acf11[:10]
#    print acf11[:10] /acf11[0]

def acovf_ma2(ma):
    # MA(2)
    # from Greene p616 (with typo), Florens p280
    b1 = -ma[1]
    b2 = -ma[2]
    rho = np.zeros(10)
    rho[0] = (1 + b1**2 + b2**2)
    rho[1] = (-b1 + b1*b2)
    rho[2] = -b2
    return rho

#    rho2 = rho/rho[0]
#    print rho2
#    print irc2[:10]/irc2[0]

def acovf_ma1(ma):
    # MA(1)
    # from Greene p616 (with typo), Florens p280
    b = -ma[1]
    rho = np.zeros(10)
    rho[0] = (1 + b**2)
    rho[1] = -b
    return rho

#    rho2 = rho/rho[0]
#    print rho2
#    print irc2[:10]/irc2[0]


ar1 = [1., -0.8]
ar0 = [1., 0.]
ma1 = [1., 0.4]
ma2 = [1., 0.4, 0.6]
ma0 = [1., 0.]

comparefn = dict(
        [('ma1', acovf_ma1),
        ('ma2', acovf_ma2),
        ('arma11', acovf_arma11),
        ('ar1', acovf_arma11)])

cases = [('ma1', (ar0, ma1)),
        ('ma2', (ar0, ma2)),
        ('arma11', (ar1, ma1)),
        ('ar1', (ar1, ma0))]

for c, args in cases:

    ar, ma = args
    print('')
    print(c, ar, ma)
    myacovf = arma_acovf(ar, ma, nobs=10)
    myacf = arma_acf(ar, ma, lags=10)
    if c[:2]=='ma':
        othacovf = comparefn[c](ma)
    else:
        othacovf = comparefn[c](ar, ma)
    print(myacovf[:5])
    print(othacovf[:5])
    #something broke again,
    #for high persistence case eg ar=0.99, nobs of IR has to be large
    #made changes to arma_acovf
    assert_array_almost_equal(myacovf, othacovf,10)
    assert_array_almost_equal(myacf, othacovf/othacovf[0],10)


#from nitime.utils
def ar_generator(N=512, sigma=1.):
    # this generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    # where v(n) is a stationary stochastic process with zero mean
    # and variance = sigma
    # this sequence is shown to be estimated well by an order 8 AR system
    taps = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    v = np.random.normal(size=N, scale=sigma**0.5)
    u = np.zeros(N)
    P = len(taps)
    for l in range(P):
        u[l] = v[l] + np.dot(u[:l][::-1], taps[:l])
    for l in range(P,N):
        u[l] = v[l] + np.dot(u[l-P:l][::-1], taps)
    return u, v, taps

#JP: small differences to using np.correlate, because assumes mean(s)=0
#    denominator is N, not N-k, biased estimator
#    misnomer: (biased) autocovariance not autocorrelation
#from nitime.utils
def autocorr(s, axis=-1):
    """Returns the autocorrelation of signal s at all lags. Adheres to the
definition r(k) = E{s(n)s*(n-k)} where E{} is the expectation operator.
"""
    N = s.shape[axis]
    S = np.fft.fft(s, n=2*N-1, axis=axis)
    sxx = np.fft.ifft(S*S.conjugate(), axis=axis).real[:N]
    return sxx/N

#JP: with valid this returns a single value, if x and y have same length
#   e.g. norm_corr(x, x)
#   using std subtracts mean, but correlate does not, requires means are exactly 0
#   biased, no n-k correction for laglength
#from nitime.utils
def norm_corr(x,y,mode = 'valid'):
    """Returns the correlation between two ndarrays, by calling np.correlate in
'same' mode and normalizing the result by the std of the arrays and by
their lengths. This results in a correlation = 1 for an auto-correlation"""

    return ( np.correlate(x,y,mode) /
             (np.std(x)*np.std(y)*(x.shape[-1])) )



# from matplotlib axes.py
# note: self is axis
def pltacorr(self, x, **kwargs):
    r"""
    call signature::

        acorr(x, normed=True, detrend=detrend_none, usevlines=True,
              maxlags=10, **kwargs)

    Plot the autocorrelation of *x*.  If *normed* = *True*,
    normalize the data by the autocorrelation at 0-th lag.  *x* is
    detrended by the *detrend* callable (default no normalization).

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length 2*maxlags+1 lag vector

      - *c* is the 2*maxlags+1 auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
        returned by :meth:`plot`

    The default *linestyle* is None and the default *marker* is
    ``'o'``, though these can be overridden with keyword args.
    The cross correlation is performed with
    :func:`numpy.correlate` with *mode* = 2.

    If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`
    rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw
    vertical lines from the origin to the acorr.  Otherwise, the
    plot style is determined by the kwargs, which are
    :class:`~matplotlib.lines.Line2D` properties.

    *maxlags* is a positive integer detailing the number of lags
    to show.  The default value of *None* will return all
    :math:`2 \mathrm{len}(x) - 1` lags.

    The return value is a tuple (*lags*, *c*, *linecol*, *b*)
    where

    - *linecol* is the
      :class:`~matplotlib.collections.LineCollection`

    - *b* is the *x*-axis.

    .. seealso::

        :meth:`~matplotlib.axes.Axes.plot` or
        :meth:`~matplotlib.axes.Axes.vlines`
           For documentation on valid kwargs.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """
    return self.xcorr(x, x, **kwargs)

def pltxcorr(self, x, y, normed=True, detrend=detrend_none,
          usevlines=True, maxlags=10, **kwargs):
    """
    call signature::

        def xcorr(self, x, y, normed=True, detrend=detrend_none,
          usevlines=True, maxlags=10, **kwargs):

    Plot the cross correlation between *x* and *y*.  If *normed* =
    *True*, normalize the data by the cross correlation at 0-th
    lag.  *x* and y are detrended by the *detrend* callable
    (default no normalization).  *x* and *y* must be equal length.

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length ``2*maxlags+1`` lag vector

      - *c* is the ``2*maxlags+1`` auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
         returned by :func:`~matplotlib.pyplot.plot`.

    The default *linestyle* is *None* and the default *marker* is
    'o', though these can be overridden with keyword args.  The
    cross correlation is performed with :func:`numpy.correlate`
    with *mode* = 2.

    If *usevlines* is *True*:

       :func:`~matplotlib.pyplot.vlines`
       rather than :func:`~matplotlib.pyplot.plot` is used to draw
       vertical lines from the origin to the xcorr.  Otherwise the
       plotstyle is determined by the kwargs, which are
       :class:`~matplotlib.lines.Line2D` properties.

       The return value is a tuple (*lags*, *c*, *linecol*, *b*)
       where *linecol* is the
       :class:`matplotlib.collections.LineCollection` instance and
       *b* is the *x*-axis.

    *maxlags* is a positive integer detailing the number of lags to show.
    The default value of *None* will return all ``(2*len(x)-1)`` lags.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """


    Nx = len(x)
    if Nx!=len(y):
        raise ValueError('x and y must be equal length')

    x = detrend(np.asarray(x))
    y = detrend(np.asarray(y))

    c = np.correlate(x, y, mode=2)

    if normed:
        c /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags,maxlags+1)
    c = c[Nx-1-maxlags:Nx+maxlags]


    if usevlines:
        a = self.vlines(lags, [0], c, **kwargs)
        b = self.axhline(**kwargs)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        d = self.plot(lags, c, **kwargs)
    else:

        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        a, = self.plot(lags, c, **kwargs)
        b = None
    return lags, c, a, b






arrvs = ar_generator()
##arma = ARIMA()
##res = arma.fit(arrvs[0], 4, 0)
arma = ARIMA(arrvs[0])
res = arma.fit((4,0, 0))

print(res[0])

acf1 = acf(arrvs[0])
acovf1b = acovf(arrvs[0], unbiased=False)
acf2 = autocorr(arrvs[0])
acf2m = autocorr(arrvs[0]-arrvs[0].mean())
print(acf1[:10])
print(acovf1b[:10])
print(acf2[:10])
print(acf2m[:10])


x = arma_generate_sample([1.0, -0.8], [1.0], 500)
print(acf(x)[:20])
print(regression.yule_walker(x, 10))

#ax = plt.axes()
plt.plot(x)
#plt.show()

plt.figure()
pltxcorr(plt,x,x)
plt.figure()
pltxcorr(plt,x,x, usevlines=False)
plt.figure()
#FIXME: plotacf was moved to graphics/tsaplots.py, and interface changed
plot_acf(plt, acf1[:20], np.arange(len(acf1[:20])), usevlines=True)
plt.figure()
ax = plt.subplot(211)
plot_acf(ax, acf1[:20], usevlines=True)
ax = plt.subplot(212)
plot_acf(ax, acf1[:20], np.arange(len(acf1[:20])), usevlines=False)

#plt.show()
