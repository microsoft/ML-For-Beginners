
import numpy as np
from scipy.signal import fftconvolve

from statsmodels.tools.validation import array_like, PandasWrapper


def bkfilter(x, low=6, high=32, K=12):
    """
    Filter a time series using the Baxter-King bandpass filter.

    Parameters
    ----------
    x : array_like
        A 1 or 2d ndarray. If 2d, variables are assumed to be in columns.
    low : float
        Minimum period for oscillations, ie., Baxter and King suggest that
        the Burns-Mitchell U.S. business cycle has 6 for quarterly data and
        1.5 for annual data.
    high : float
        Maximum period for oscillations BK suggest that the U.S.
        business cycle has 32 for quarterly data and 8 for annual data.
    K : int
        Lead-lag length of the filter. Baxter and King propose a truncation
        length of 12 for quarterly data and 3 for annual data.

    Returns
    -------
    ndarray
        The cyclical component of x.

    See Also
    --------
    statsmodels.tsa.filters.cf_filter.cffilter
        The Christiano Fitzgerald asymmetric, random walk filter.
    statsmodels.tsa.filters.bk_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.seasonal.seasonal_decompose
        Decompose a time series using moving averages.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    Returns a centered weighted moving average of the original series. Where
    the weights a[j] are computed ::

      a[j] = b[j] + theta, for j = 0, +/-1, +/-2, ... +/- K
      b[0] = (omega_2 - omega_1)/pi
      b[j] = 1/(pi*j)(sin(omega_2*j)-sin(omega_1*j), for j = +/-1, +/-2,...

    and theta is a normalizing constant ::

      theta = -sum(b)/(2K+1)

    See the notebook `Time Series Filters
    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.

    References
    ----------
    Baxter, M. and R. G. King. "Measuring Business Cycles: Approximate
        Band-Pass Filters for Economic Time Series." *Review of Economics and
        Statistics*, 1999, 81(4), 575-593.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    >>> dta.set_index(index, inplace=True)

    >>> cycles = sm.tsa.filters.bkfilter(dta[['realinv']], 6, 24, 12)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cycles.plot(ax=ax, style=['r--', 'b-'])
    >>> plt.show()

    .. plot:: plots/bkf_plot.py
    """
    # TODO: change the docstring to ..math::?
    # TODO: allow windowing functions to correct for Gibb's Phenomenon?
    # adjust bweights (symmetrically) by below before demeaning
    # Lancosz Sigma Factors np.sinc(2*j/(2.*K+1))
    pw = PandasWrapper(x)
    x = array_like(x, 'x', maxdim=2)
    omega_1 = 2. * np.pi / high  # convert from freq. to periodicity
    omega_2 = 2. * np.pi / low
    bweights = np.zeros(2 * K + 1)
    bweights[K] = (omega_2 - omega_1) / np.pi  # weight at zero freq.
    j = np.arange(1, int(K) + 1)
    weights = 1 / (np.pi * j) * (np.sin(omega_2 * j) - np.sin(omega_1 * j))
    bweights[K + j] = weights  # j is an idx
    bweights[:K] = weights[::-1]  # make symmetric weights
    bweights -= bweights.mean()  # make sure weights sum to zero
    if x.ndim == 2:
        bweights = bweights[:, None]
    x = fftconvolve(x, bweights, mode='valid')
    # get a centered moving avg/convolution

    return pw.wrap(x, append='cycle', trim_start=K, trim_end=K)
