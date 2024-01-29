from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.python import lmap

import calendar
from io import BytesIO
import locale

import numpy as np
from numpy.testing import assert_, assert_equal
import pandas as pd
import pytest

from statsmodels.datasets import elnino, macrodata
from statsmodels.graphics.tsaplots import (
    month_plot,
    plot_accf_grid,
    plot_acf,
    plot_ccf,
    plot_pacf,
    plot_predict,
    quarter_plot,
    seasonal_plot,
)
from statsmodels.tsa import arima_process as tsp
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


@pytest.mark.matplotlib
def test_plot_acf(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_acf(acf, ax=ax, lags=10)
    plot_acf(acf, ax=ax)
    plot_acf(acf, ax=ax, alpha=None)


@pytest.mark.matplotlib
def test_plot_acf_irregular(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_acf(acf, ax=ax, lags=np.arange(1, 11))
    plot_acf(acf, ax=ax, lags=10, zero=False)
    plot_acf(acf, ax=ax, alpha=None, zero=False)


@pytest.mark.matplotlib
def test_plot_pacf(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_pacf(pacf, ax=ax)
    plot_pacf(pacf, ax=ax, alpha=None)


@pytest.mark.matplotlib
def test_plot_pacf_kwargs(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)

    buff = BytesIO()
    plot_pacf(pacf, ax=ax)
    fig.savefig(buff, format="rgba")

    buff_linestyle = BytesIO()
    fig_linestyle = plt.figure()
    ax = fig_linestyle.add_subplot(111)
    plot_pacf(pacf, ax=ax, ls="-")
    fig_linestyle.savefig(buff_linestyle, format="rgba")

    buff_with_vlines = BytesIO()
    fig_with_vlines = plt.figure()
    ax = fig_with_vlines.add_subplot(111)
    vlines_kwargs = {"linestyles": "dashdot"}
    plot_pacf(pacf, ax=ax, vlines_kwargs=vlines_kwargs)
    fig_with_vlines.savefig(buff_with_vlines, format="rgba")

    buff.seek(0)
    buff_linestyle.seek(0)
    buff_with_vlines.seek(0)
    plain = buff.read()
    linestyle = buff_linestyle.read()
    with_vlines = buff_with_vlines.read()

    assert_(plain != linestyle)
    assert_(with_vlines != plain)
    assert_(linestyle != with_vlines)


@pytest.mark.matplotlib
def test_plot_acf_kwargs(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)

    buff = BytesIO()
    plot_acf(acf, ax=ax)
    fig.savefig(buff, format="rgba")

    buff_with_vlines = BytesIO()
    fig_with_vlines = plt.figure()
    ax = fig_with_vlines.add_subplot(111)
    vlines_kwargs = {"linestyles": "dashdot"}
    plot_acf(acf, ax=ax, vlines_kwargs=vlines_kwargs)
    fig_with_vlines.savefig(buff_with_vlines, format="rgba")

    buff.seek(0)
    buff_with_vlines.seek(0)
    plain = buff.read()
    with_vlines = buff_with_vlines.read()

    assert_(with_vlines != plain)


@pytest.mark.matplotlib
def test_plot_acf_missing(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    acf[::13] = np.nan

    buff = BytesIO()
    plot_acf(acf, ax=ax, missing="drop")
    fig.savefig(buff, format="rgba")
    buff.seek(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    buff_conservative = BytesIO()
    plot_acf(acf, ax=ax, missing="conservative")
    fig.savefig(buff_conservative, format="rgba")
    buff_conservative.seek(0)
    assert_(buff.read() != buff_conservative.read())


@pytest.mark.matplotlib
def test_plot_pacf_irregular(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_pacf(pacf, ax=ax, lags=np.arange(1, 11))
    plot_pacf(pacf, ax=ax, lags=10, zero=False)
    plot_pacf(pacf, ax=ax, alpha=None, zero=False)


@pytest.mark.matplotlib
def test_plot_ccf(close_figures):
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    x1 = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    x2 = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_ccf(x1, x2)
    plot_ccf(x1, x2, ax=ax, lags=10)
    plot_ccf(x1, x2, ax=ax)
    plot_ccf(x1, x2, ax=ax, alpha=None)
    plot_ccf(x1, x2, ax=ax, negative_lags=True)
    plot_ccf(x1, x2, ax=ax, adjusted=True)
    plot_ccf(x1, x2, ax=ax, fft=True)
    plot_ccf(x1, x2, ax=ax, title='CCF')
    plot_ccf(x1, x2, ax=ax, auto_ylims=True)
    plot_ccf(x1, x2, ax=ax, use_vlines=False)


@pytest.mark.matplotlib
def test_plot_accf_grid(close_figures):
    # Just test that it runs.
    fig = plt.figure()

    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    x = np.vstack([
        armaprocess.generate_sample(100, distrvs=rs.standard_normal),
        armaprocess.generate_sample(100, distrvs=rs.standard_normal),
    ]).T
    plot_accf_grid(x)
    plot_accf_grid(pd.DataFrame({'x': x[:, 0], 'y': x[:, 1]}))
    plot_accf_grid(x, fig=fig, lags=10)
    plot_accf_grid(x, fig=fig)
    plot_accf_grid(x, fig=fig, negative_lags=False)
    plot_accf_grid(x, fig=fig, alpha=None)
    plot_accf_grid(x, fig=fig, adjusted=True)
    plot_accf_grid(x, fig=fig, fft=True)
    plot_accf_grid(x, fig=fig, auto_ylims=True)
    plot_accf_grid(x, fig=fig, use_vlines=False)


@pytest.mark.matplotlib
def test_plot_month(close_figures):
    dta = elnino.load_pandas().data
    dta["YEAR"] = dta.YEAR.astype(int).apply(str)
    dta = dta.set_index("YEAR").T.unstack()
    dates = pd.to_datetime(["-".join([x[1], x[0]]) for x in dta.index.values])

    # test dates argument
    fig = month_plot(dta.values, dates=dates, ylabel="el nino")

    # test with a TimeSeries DatetimeIndex with no freq
    dta.index = pd.DatetimeIndex(dates)
    fig = month_plot(dta)

    # w freq
    dta.index = pd.DatetimeIndex(dates, freq="MS")
    fig = month_plot(dta)

    # test with a TimeSeries PeriodIndex
    dta.index = pd.PeriodIndex(dates, freq="M")
    fig = month_plot(dta)

    # test localized xlabels
    try:
        with calendar.different_locale("DE_de"):
            fig = month_plot(dta)
            labels = [_.get_text() for _ in fig.axes[0].get_xticklabels()]
            expected = [
                "Jan",
                "Feb",
                ("Mär", "Mrz"),
                "Apr",
                "Mai",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Okt",
                "Nov",
                "Dez",
            ]
            for lbl, exp in zip(labels, expected):
                if isinstance(exp, tuple):
                    assert lbl in exp
                else:
                    assert lbl == exp
    except locale.Error:
        pytest.xfail(reason="Failure due to unsupported locale")


@pytest.mark.matplotlib
def test_plot_quarter(close_figures):
    dta = macrodata.load_pandas().data
    dates = lmap(
        "-Q".join,
        zip(
            dta.year.astype(int).apply(str), dta.quarter.astype(int).apply(str)
        ),
    )
    # test dates argument
    quarter_plot(dta.unemp.values, dates)

    # test with a DatetimeIndex with no freq
    dta.set_index(pd.DatetimeIndex(dates, freq="QS-Oct"), inplace=True)
    quarter_plot(dta.unemp)

    # w freq
    # see pandas #6631
    dta.index = pd.DatetimeIndex(dates, freq="QS-Oct")
    quarter_plot(dta.unemp)

    # w PeriodIndex
    dta.index = pd.PeriodIndex(dates, freq="Q")
    quarter_plot(dta.unemp)


@pytest.mark.matplotlib
def test_seasonal_plot(close_figures):
    rs = np.random.RandomState(1234)
    data = rs.randn(20, 12)
    data += 6 * np.sin(np.arange(12.0) / 11 * np.pi)[None, :]
    data = data.ravel()
    months = np.tile(np.arange(1, 13), (20, 1))
    months = months.ravel()
    df = pd.DataFrame([data, months], index=["data", "months"]).T
    grouped = df.groupby("months")["data"]
    labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig = seasonal_plot(grouped, labels)
    ax = fig.get_axes()[0]
    output = [tl.get_text() for tl in ax.get_xticklabels()]
    assert_equal(labels, output)


@pytest.mark.matplotlib
@pytest.mark.parametrize(
    "model_and_args",
    [(AutoReg, dict(lags=2, old_names=False)), (ARIMA, dict(order=(2, 0, 0)))],
)
@pytest.mark.parametrize("use_pandas", [True, False])
@pytest.mark.parametrize("alpha", [None, 0.10])
def test_predict_plot(use_pandas, model_and_args, alpha):
    model, kwargs = model_and_args
    rs = np.random.RandomState(0)
    y = rs.standard_normal(1000)
    for i in range(2, 1000):
        y[i] += 1.8 * y[i - 1] - 0.9 * y[i - 2]
    y = y[100:]
    if use_pandas:
        index = pd.date_range(
            "1960-1-1", freq=MONTH_END, periods=y.shape[0] + 24
        )
        start = index[index.shape[0] // 2]
        end = index[-1]
        y = pd.Series(y, index=index[:-24])
    else:
        start = y.shape[0] // 2
        end = y.shape[0] + 24
    res = model(y, **kwargs).fit()
    fig = plot_predict(res, start, end, alpha=alpha)
    assert isinstance(fig, plt.Figure)


@pytest.mark.matplotlib
def test_plot_pacf_small_sample():
    idx = [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(10)]
    df = pd.DataFrame(
        index=idx,
        columns=["a"],
        data=list(range(10))
    )
    plot_pacf(df)
