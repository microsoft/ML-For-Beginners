from statsmodels.compat.pandas import MONTH_END

import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, "results", "stl_test_results.csv")
results = pd.read_csv(file_path)
results.columns = [c.strip() for c in results.columns]
results.scenario = results.scenario.apply(str.strip)
results = results.set_index(["scenario", "idx"])


@pytest.fixture(scope="module", params=[True, False])
def robust(request):
    return request.param


def default_kwargs_base():
    file_path = os.path.join(cur_dir, "results", "stl_co2.csv")
    co2 = np.asarray(pd.read_csv(file_path, header=None).iloc[:, 0])
    y = co2
    nobs = y.shape[0]
    nperiod = 12
    work = np.zeros((nobs + 2 * nperiod, 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    return dict(
        y=y,
        n=y.shape[0],
        np=nperiod,
        ns=35,
        nt=19,
        nl=13,
        no=2,
        ni=1,
        nsjump=4,
        ntjump=2,
        nljump=2,
        isdeg=1,
        itdeg=1,
        ildeg=1,
        rw=rw,
        trend=trend,
        season=season,
        work=work,
    )


@pytest.fixture(scope="function")
def default_kwargs():
    return default_kwargs_base()


@pytest.fixture(scope="function")
def default_kwargs_short():
    kwargs = default_kwargs_base()
    y = kwargs["y"][:-1]
    nobs = y.shape[0]
    work = np.zeros((nobs + 2 * kwargs["np"], 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    kwargs.update(
        dict(y=y, n=nobs, rw=rw, trend=trend, season=season, work=work)
    )
    return kwargs


def _to_class_kwargs(kwargs, robust=False):
    endog = kwargs["y"]
    np = kwargs["np"]
    ns = kwargs["ns"]
    nt = kwargs["nt"]
    nl = kwargs["nl"]
    isdeg = kwargs["isdeg"]
    itdeg = kwargs["itdeg"]
    ildeg = kwargs["ildeg"]
    nsjump = kwargs["nsjump"]
    ntjump = kwargs["ntjump"]
    nljump = kwargs["nljump"]
    outer_iter = kwargs["no"]
    inner_iter = kwargs["ni"]
    class_kwargs = dict(
        endog=endog,
        period=np,
        seasonal=ns,
        trend=nt,
        low_pass=nl,
        seasonal_deg=isdeg,
        trend_deg=itdeg,
        low_pass_deg=ildeg,
        robust=robust,
        seasonal_jump=nsjump,
        trend_jump=ntjump,
        low_pass_jump=nljump,
    )
    return class_kwargs, outer_iter, inner_iter


def test_baseline_class(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)
    resid = class_kwargs["endog"] - expected.trend - expected.season
    assert_allclose(res.resid, resid)


def test_short_class(default_kwargs_short):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs_short)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["short"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_ntjump_1_class(default_kwargs):
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_ntjump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1-ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_parameter_checks_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    endog2 = np.hstack((endog[:, None], endog[:, None]))
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="endog is required to have ndim 1"):
        STL(endog=endog2, period=period)
    match = "period must be a positive integer >= 2"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=1)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=-12)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=4.0)


def test_parameter_checks_seasonal(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "seasonal must be an odd positive integer >= 3"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=2)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=-7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=13.0)


def test_parameter_checks_trend(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "trend must be an odd positive integer >= 3 where trend > period"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=11)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=19.0)


def test_parameter_checks_low_pass(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]

    match = (
        "low_pass must be an odd positive integer >= 3 where"
        " low_pass > period"
    )
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=19.0)


def test_jump_errors(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=0)
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=1.0)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=0)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=1.0)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=0)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=1.0)


def test_defaults_smoke(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    mod.fit()


def test_pandas(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = pd.Series(class_kwargs["endog"], name="y")
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_period_detection(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()

    del class_kwargs["period"]
    endog = class_kwargs["endog"]
    index = pd.date_range("1-1-1959", periods=348, freq=MONTH_END)
    class_kwargs["endog"] = pd.Series(endog, index=index)
    mod = STL(**class_kwargs)

    res_implicit_period = mod.fit()
    assert_allclose(res.seasonal, res_implicit_period.seasonal)


def test_no_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    del class_kwargs["period"]
    class_kwargs["endog"] = pd.Series(class_kwargs["endog"])
    with pytest.raises(ValueError, match="Unable to determine period from"):
        STL(**class_kwargs)


@pytest.mark.matplotlib
def test_plot(default_kwargs, close_figures):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs).fit(outer_iter=outer, inner_iter=inner)
    res.plot()

    class_kwargs["endog"] = pd.Series(class_kwargs["endog"], name="CO2")
    res = STL(**class_kwargs).fit()
    res.plot()


def test_default_trend(default_kwargs):
    # GH 6686
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    class_kwargs["seasonal"] = 17
    class_kwargs["trend"] = None
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected

    class_kwargs["seasonal"] = 7
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected


def test_pickle(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()
    pkl = pickle.dumps(mod)
    reloaded = pickle.loads(pkl)
    res2 = reloaded.fit()
    assert_allclose(res.trend, res2.trend)
    assert_allclose(res.seasonal, res2.seasonal)
    assert mod.config == reloaded.config


def test_squezable_to_1d():
    data = co2.load().data
    data = data.resample(MONTH_END).mean().ffill()
    res = STL(data).fit()
    assert isinstance(res, DecomposeResult)
