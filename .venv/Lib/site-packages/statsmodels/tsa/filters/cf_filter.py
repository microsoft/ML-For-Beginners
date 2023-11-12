import numpy as np

from statsmodels.tools.validation import PandasWrapper, array_like

# the data is sampled quarterly, so cut-off frequency of 18

# Wn is normalized cut-off freq
#Cutoff frequency is that frequency where the magnitude response of the filter
# is sqrt(1/2.). For butter, the normalized cutoff frequency Wn must be a
# number between  0 and 1, where 1 corresponds to the Nyquist frequency, p
# radians per sample.


# NOTE: uses a loop, could probably be sped-up for very large datasets
def cffilter(x, low=6, high=32, drift=True):
    """
    Christiano Fitzgerald asymmetric, random walk filter.

    Parameters
    ----------
    x : array_like
        The 1 or 2d array to filter. If 2d, variables are assumed to be in
        columns.
    low : float
        Minimum period of oscillations. Features below low periodicity are
        filtered out. Default is 6 for quarterly data, giving a 1.5 year
        periodicity.
    high : float
        Maximum period of oscillations. Features above high periodicity are
        filtered out. Default is 32 for quarterly data, giving an 8 year
        periodicity.
    drift : bool
        Whether or not to remove a trend from the data. The trend is estimated
        as np.arange(nobs)*(x[-1] - x[0])/(len(x)-1).

    Returns
    -------
    cycle : array_like
        The features of x between the periodicities low and high.
    trend : array_like
        The trend in the data with the cycles removed.

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King filter.
    statsmodels.tsa.filters.bk_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.seasonal.seasonal_decompose
        Decompose a time series using moving averages.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    See the notebook `Time Series Filters
    <../examples/notebooks/generated/tsa_filters.html>`__ for an overview.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    >>> dta.set_index(index, inplace=True)

    >>> cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cf_cycles.plot(ax=ax, style=['r--', 'b-'])
    >>> plt.show()

    .. plot:: plots/cff_plot.py
    """
    #TODO: cythonize/vectorize loop?, add ability for symmetric filter,
    #      and estimates of theta other than random walk.
    if low < 2:
        raise ValueError("low must be >= 2")
    pw = PandasWrapper(x)
    x = array_like(x, 'x', ndim=2)
    nobs, nseries = x.shape
    a = 2*np.pi/high
    b = 2*np.pi/low

    if drift:  # get drift adjusted series
        x = x - np.arange(nobs)[:, None] * (x[-1] - x[0]) / (nobs - 1)

    J = np.arange(1, nobs + 1)
    Bj = (np.sin(b * J) - np.sin(a * J)) / (np.pi * J)
    B0 = (b - a) / np.pi
    Bj = np.r_[B0, Bj][:, None]
    y = np.zeros((nobs, nseries))

    for i in range(nobs):
        B = -.5 * Bj[0] - np.sum(Bj[1:-i - 2])
        A = -Bj[0] - np.sum(Bj[1:-i - 2]) - np.sum(Bj[1:i]) - B
        y[i] = (Bj[0] * x[i] + np.dot(Bj[1:-i - 2].T, x[i + 1:-1]) +
                B * x[-1] + np.dot(Bj[1:i].T, x[1:i][::-1]) + A * x[0])
    y = y.squeeze()

    cycle, trend = y.squeeze(), x.squeeze() - y

    return pw.wrap(cycle, append='cycle'), pw.wrap(trend, append='trend')


if __name__ == "__main__":
    import statsmodels as sm
    dta = sm.datasets.macrodata.load().data[['infl','tbilrate']].view((float,2))[1:]
    cycle, trend = cffilter(dta, 6, 32, drift=True)
    dta = sm.datasets.macrodata.load().data['tbilrate'][1:]
    cycle2, trend2 = cffilter(dta, 6, 32, drift=True)
