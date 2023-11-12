"""Correlation plot functions."""
from statsmodels.compat.pandas import deprecate_kwarg

import calendar

import numpy as np
import pandas as pd

from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf


def _prepare_data_corr_plot(x, lags, zero):
    zero = bool(zero)
    irregular = False if zero else True
    if lags is None:
        # GH 4663 - use a sensible default value
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs - 1)
        lags = np.arange(not zero, lim + 1)
    elif np.isscalar(lags):
        lags = np.arange(not zero, int(lags) + 1)  # +1 for zero lag
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(int)
    nlags = lags.max(0)

    return lags, nlags, irregular


def _plot_corr(
    ax,
    title,
    acf_x,
    confint,
    lags,
    irregular,
    use_vlines,
    vlines_kwargs,
    auto_ylims=False,
    **kwargs,
):
    if irregular:
        acf_x = acf_x[lags]
        if confint is not None:
            confint = confint[lags]

    if use_vlines:
        ax.vlines(lags, [0], acf_x, **vlines_kwargs)
        ax.axhline(**kwargs)

    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 5)
    if "ls" not in kwargs:
        # gh-2369
        kwargs.setdefault("linestyle", "None")
    ax.margins(0.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.set_title(title)

    ax.set_ylim(-1, 1)
    if auto_ylims:
        ax.set_ylim(
            1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)),
            1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)),
        )

    if confint is not None:
        if lags[0] == 0:
            lags = lags[1:]
            confint = confint[1:]
            acf_x = acf_x[1:]
        lags = lags.astype(float)
        lags[0] -= 0.5
        lags[-1] += 0.5
        ax.fill_between(
            lags, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25
        )


@deprecate_kwarg("unbiased", "adjusted")
def plot_acf(
    x,
    ax=None,
    lags=None,
    *,
    alpha=0.05,
    use_vlines=True,
    adjusted=False,
    fft=False,
    missing="none",
    title="Autocorrelation",
    zero=True,
    auto_ylims=False,
    bartlett_confint=True,
    vlines_kwargs=None,
    **kwargs,
):
    """
    Plot the autocorrelation function

    Plots lags on the horizontal and the correlations on vertical axis.

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula. If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    adjusted : bool
        If True, then denominators for autocovariance are n-k, otherwise n
    fft : bool, optional
        If True, computes the ACF via FFT.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how
        the NaNs are to be treated.
    title : str, optional
        Title to place on plot.  Default is 'Autocorrelation'
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.
    auto_ylims : bool, optional
        If True, adjusts automatically the y-axis limits to ACF values.
    bartlett_confint : bool, default True
        Confidence intervals for ACF values are generally placed at 2
        standard errors around r_k. The formula used for standard error
        depends upon the situation. If the autocorrelations are being used
        to test for randomness of residuals as part of the ARIMA routine,
        the standard errors are determined assuming the residuals are white
        noise. The approximate formula for any lag is that standard error
        of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on
        the 1/sqrt(N) result. For more elementary discussion, see section
        5.3.2 in [2].
        For the ACF of raw data, the standard error at a lag k is
        found as if the right model was an MA(k-1). This allows the
        possible interpretation that if all autocorrelations past a
        certain lag are within the limits, the model might be an MA of
        order defined by the last significant autocorrelation. In this
        case, a moving average model is assumed for the data and the
        standard errors for the confidence intervals should be
        generated using Bartlett's formula. For more details on
        Bartlett formula result, see section 7.2 in [1].
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    matplotlib.pyplot.xcorr
    matplotlib.pyplot.acorr

    Notes
    -----
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    kwargs is used to pass matplotlib optional arguments to both the line
    tracing the autocorrelations and for the horizontal line at 0. These
    options must be valid for a Line2D object.

    vlines_kwargs is used to pass additional optional arguments to the
    vertical lines connecting each autocorrelation to the axis.  These options
    must be valid for a LineCollection object.

    References
    ----------
    [1] Brockwell and Davis, 1987. Time Series Theory and Methods
    [2] Brockwell and Davis, 2010. Introduction to Time Series and
    Forecasting, 2nd edition.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.sunspots.load_pandas().data
    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    >>> del dta["YEAR"]
    >>> sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
    >>> plt.show()

    .. plot:: plots/graphics_tsa_plot_acf.py
    """
    fig, ax = utils.create_mpl_ax(ax)

    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs

    confint = None
    # acf has different return type based on alpha
    acf_x = acf(
        x,
        nlags=nlags,
        alpha=alpha,
        fft=fft,
        bartlett_confint=bartlett_confint,
        adjusted=adjusted,
        missing=missing,
    )
    if alpha is not None:
        acf_x, confint = acf_x[:2]

    _plot_corr(
        ax,
        title,
        acf_x,
        confint,
        lags,
        irregular,
        use_vlines,
        vlines_kwargs,
        auto_ylims=auto_ylims,
        **kwargs,
    )

    return fig


def plot_pacf(
    x,
    ax=None,
    lags=None,
    alpha=0.05,
    method="ywm",
    use_vlines=True,
    title="Partial Autocorrelation",
    zero=True,
    vlines_kwargs=None,
    **kwargs,
):
    """
    Plot the partial autocorrelation function

    Parameters
    ----------
    x : array_like
        Array of time-series values
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))
    method : str
        Specifies which method for the calculations to use:

        - "ywm" or "ywmle" : Yule-Walker without adjustment. Default.
        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
          denominator for acovf. Default.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
          correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
          correction.

    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    title : str, optional
        Title to place on plot.  Default is 'Partial Autocorrelation'
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    matplotlib.pyplot.xcorr
    matplotlib.pyplot.acorr

    Notes
    -----
    Plots lags on the horizontal and the correlations on vertical axis.
    Adapted from matplotlib's `xcorr`.

    Data are plotted as ``plot(lags, corr, **kwargs)``

    kwargs is used to pass matplotlib optional arguments to both the line
    tracing the autocorrelations and for the horizontal line at 0. These
    options must be valid for a Line2D object.

    vlines_kwargs is used to pass additional optional arguments to the
    vertical lines connecting each autocorrelation to the axis.  These options
    must be valid for a LineCollection object.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.sunspots.load_pandas().data
    >>> dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    >>> del dta["YEAR"]
    >>> sm.graphics.tsa.plot_pacf(dta.values.squeeze(), lags=40, method="ywm")
    >>> plt.show()

    .. plot:: plots/graphics_tsa_plot_pacf.py
    """
    fig, ax = utils.create_mpl_ax(ax)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, zero)

    confint = None
    if alpha is None:
        acf_x = pacf(x, nlags=nlags, alpha=alpha, method=method)
    else:
        acf_x, confint = pacf(x, nlags=nlags, alpha=alpha, method=method)

    _plot_corr(
        ax,
        title,
        acf_x,
        confint,
        lags,
        irregular,
        use_vlines,
        vlines_kwargs,
        **kwargs,
    )

    return fig


def seasonal_plot(grouped_x, xticklabels, ylabel=None, ax=None):
    """
    Consider using one of month_plot or quarter_plot unless you need
    irregular plotting.

    Parameters
    ----------
    grouped_x : iterable of DataFrames
        Should be a GroupBy object (or similar pair of group_names and groups
        as DataFrames) with a DatetimeIndex or PeriodIndex
    xticklabels : list of str
        List of season labels, one for each group.
    ylabel : str
        Lable for y axis
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    """
    fig, ax = utils.create_mpl_ax(ax)
    start = 0
    ticks = []
    for season, df in grouped_x:
        df = df.copy()  # or sort balks for series. may be better way
        df.sort_index()
        nobs = len(df)
        x_plot = np.arange(start, start + nobs)
        ticks.append(x_plot.mean())
        ax.plot(x_plot, df.values, "k")
        ax.hlines(
            df.values.mean(), x_plot[0], x_plot[-1], colors="r", linewidth=3
        )
        start += nobs

    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.margins(0.1, 0.05)
    return fig


def month_plot(x, dates=None, ylabel=None, ax=None):
    """
    Seasonal plot of monthly data.

    Parameters
    ----------
    x : array_like
        Seasonal data to plot. If dates is None, x must be a pandas object
        with a PeriodIndex or DatetimeIndex with a monthly frequency.
    dates : array_like, optional
        If `x` is not a pandas object, then dates must be supplied.
    ylabel : str, optional
        The label for the y-axis. Will attempt to use the `name` attribute
        of the Series.
    ax : Axes, optional
        Existing axes instance.

    Returns
    -------
    Figure
       If `ax` is provided, the Figure instance attached to `ax`. Otherwise
       a new Figure instance.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd

    >>> dta = sm.datasets.elnino.load_pandas().data
    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    >>> dta = dta.set_index('YEAR').T.unstack()
    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
    ...                                 dta.index.values)))
    >>> dta.index = pd.DatetimeIndex(dates, freq='MS')
    >>> fig = sm.graphics.tsa.month_plot(dta)

    .. plot:: plots/graphics_tsa_month_plot.py
    """

    if dates is None:
        from statsmodels.tools.data import _check_period_index

        _check_period_index(x, freq="M")
    else:
        x = pd.Series(x, index=pd.PeriodIndex(dates, freq="M"))

    # there's no zero month
    xticklabels = list(calendar.month_abbr)[1:]
    return seasonal_plot(
        x.groupby(lambda y: y.month), xticklabels, ylabel=ylabel, ax=ax
    )


def quarter_plot(x, dates=None, ylabel=None, ax=None):
    """
    Seasonal plot of quarterly data

    Parameters
    ----------
    x : array_like
        Seasonal data to plot. If dates is None, x must be a pandas object
        with a PeriodIndex or DatetimeIndex with a monthly frequency.
    dates : array_like, optional
        If `x` is not a pandas object, then dates must be supplied.
    ylabel : str, optional
        The label for the y-axis. Will attempt to use the `name` attribute
        of the Series.
    ax : matplotlib.axes, optional
        Existing axes instance.

    Returns
    -------
    Figure
       If `ax` is provided, the Figure instance attached to `ax`. Otherwise
       a new Figure instance.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd

    >>> dta = sm.datasets.elnino.load_pandas().data
    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    >>> dta = dta.set_index('YEAR').T.unstack()
    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
    ...                                 dta.index.values)))
    >>> dta.index = dates.to_period('Q')
    >>> fig = sm.graphics.tsa.quarter_plot(dta)

    .. plot:: plots/graphics_tsa_quarter_plot.py
    """

    if dates is None:
        from statsmodels.tools.data import _check_period_index

        _check_period_index(x, freq="Q")
    else:
        x = pd.Series(x, index=pd.PeriodIndex(dates, freq="Q"))

    xticklabels = ["q1", "q2", "q3", "q4"]
    return seasonal_plot(
        x.groupby(lambda y: y.quarter), xticklabels, ylabel=ylabel, ax=ax
    )


def plot_predict(
    result,
    start=None,
    end=None,
    dynamic=False,
    alpha=0.05,
    ax=None,
    **predict_kwargs,
):
    """

    Parameters
    ----------
    result : Result
        Any model result supporting ``get_prediction``.
    start : int, str, or datetime, optional
        Zero-indexed observation number at which to start forecasting,
        i.e., the first forecast is start. Can also be a date string to
        parse or a datetime type. Default is the the zeroth observation.
    end : int, str, or datetime, optional
        Zero-indexed observation number at which to end forecasting, i.e.,
        the last forecast is end. Can also be a date string to
        parse or a datetime type. However, if the dates index does not
        have a fixed frequency, end must be an integer index if you
        want out of sample prediction. Default is the last observation in
        the sample.
    dynamic : bool, int, str, or datetime, optional
        Integer offset relative to `start` at which to begin dynamic
        prediction. Can also be an absolute date string to parse or a
        datetime type (these are not interpreted as offsets).
        Prior to this observation, true endogenous values will be used for
        prediction; starting with this observation and continuing through
        the end of prediction, forecasted endogenous values will be used
        instead.
    alpha : {float, None}
        The tail probability not covered by the confidence interval. Must
        be in (0, 1). Confidence interval is constructed assuming normally
        distributed shocks. If None, figure will not show the confidence
        interval.
    ax : AxesSubplot
        matplotlib Axes instance to use
    **predict_kwargs
        Any additional keyword arguments to pass to ``result.get_prediction``.

    Returns
    -------
    Figure
        matplotlib Figure containing the prediction plot
    """
    from statsmodels.graphics.utils import _import_mpl, create_mpl_ax

    _ = _import_mpl()
    fig, ax = create_mpl_ax(ax)
    from statsmodels.tsa.base.prediction import PredictionResults

    # use predict so you set dates
    pred: PredictionResults = result.get_prediction(
        start=start, end=end, dynamic=dynamic, **predict_kwargs
    )
    mean = pred.predicted_mean
    if isinstance(mean, (pd.Series, pd.DataFrame)):
        x = mean.index
        mean.plot(ax=ax, label="forecast")
    else:
        x = np.arange(mean.shape[0])
        ax.plot(x, mean)

    if alpha is not None:
        label = f"{1-alpha:.0%} confidence interval"
        ci = pred.conf_int(alpha)
        conf_int = np.asarray(ci)

        ax.fill_between(
            x,
            conf_int[:, 0],
            conf_int[:, 1],
            color="gray",
            alpha=0.5,
            label=label,
        )

    ax.legend(loc="best")

    return fig
