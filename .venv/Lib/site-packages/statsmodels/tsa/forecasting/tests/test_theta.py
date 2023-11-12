from itertools import product

import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.forecasting.theta import ThetaModel

SMOKE_PARAMS = list(product(["array", "dataframe", "series"], [True, False]))
SMOKE_IDS = [f"type: {typ}, exponential: {exp}" for typ, exp in SMOKE_PARAMS]


@pytest.fixture(params=SMOKE_PARAMS, ids=SMOKE_IDS)
def data(request):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    scale = 0.01 if request.param[1] else 1
    y = np.cumsum(scale + scale * rs.standard_normal((300)))
    if request.param[1]:
        y = np.exp(y)
    index = pd.date_range("2000-01-01", periods=300)
    if request.param[0] == "array":
        return y
    elif request.param[0] == "dataframe":
        return pd.DataFrame(y[:, None], columns=["y"], index=index)
    else:
        return pd.Series(y, name="y", index=index)


@pytest.fixture(params=["datetime", "period", "range", "nofreq"])
def indexed_data(request):
    rs = np.random.RandomState([3290328901, 323293105, 121029109])
    scale = 0.01
    y = np.cumsum(scale + scale * rs.standard_normal((300)))
    y = np.exp(y)
    if request.param == "datetime":
        index = pd.date_range("2000-1-1", periods=300)
    elif request.param == "period":
        index = pd.period_range("2000-1-1", periods=300, freq="M")
    elif request.param == "range":
        index = pd.RangeIndex(100, 100 + 2 * 300, 2)
    else:  # request.param == "nofreq"
        index = pd.date_range("2000-1-1", periods=1000)
        locs = np.unique(rs.randint(0, 1000, size=500))
        index = index[locs[:300]]
    return pd.Series(y, index=index, name=f"y_{request.param}")


@pytest.mark.smoke
@pytest.mark.parametrize("period", [None, 4, 12])
@pytest.mark.parametrize("use_mle", [True, False])
@pytest.mark.parametrize("deseasonalize", [True, False])
@pytest.mark.parametrize("use_test", [True, False])
@pytest.mark.parametrize("diff", [True, False])
@pytest.mark.parametrize("model", ["auto", "additive", "multiplicative"])
def test_smoke(data, period, use_mle, deseasonalize, use_test, diff, model):
    if period is None and isinstance(data, np.ndarray):
        return
    res = ThetaModel(
        data,
        period=period,
        deseasonalize=deseasonalize,
        use_test=use_test,
        difference=diff,
        method=model,
    ).fit(use_mle=use_mle)
    assert "b0" in str(res.summary())
    res.forecast(36)
    res.forecast_components(47)
    assert res.model.use_test is (use_test and res.model.deseasonalize)
    assert res.model.difference is diff


@pytest.mark.smoke
def test_alt_index(indexed_data):
    idx = indexed_data.index
    date_like = not hasattr(idx, "freq") or getattr(idx, "freq", None) is None
    period = 12 if date_like else None
    res = ThetaModel(indexed_data, period=period).fit()
    if hasattr(idx, "freq") and idx.freq is None:
        with pytest.warns(UserWarning):
            res.forecast_components(37)
        with pytest.warns(UserWarning):
            res.forecast(23)
    else:
        res.forecast_components(37)
        res.forecast(23)


def test_no_freq():
    idx = pd.date_range("2000-1-1", periods=300)
    locs = []
    for i in range(100):
        locs.append(2 * i + int((i % 2) == 1))
    y = pd.Series(np.random.standard_normal(100), index=idx[locs])
    with pytest.raises(ValueError, match="You must specify a period or"):
        ThetaModel(y)


def test_forecast_errors(data):
    res = ThetaModel(data, period=12).fit()
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        res.forecast(-1)
    with pytest.raises(ValueError, match="theta must be a float"):
        res.forecast(7, theta=0.99)
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        res.forecast_components(0)


def test_pi_width():
    # GH 7075
    rs = np.random.RandomState(1233091)
    y = np.arange(100) + rs.standard_normal(100)

    th = ThetaModel(y, period=12, deseasonalize=False)
    res = th.fit()
    pi = res.prediction_intervals(24)
    d = np.squeeze(np.diff(np.asarray(pi), axis=1))
    assert np.all(np.diff(d) > 0)


# GH7544
@pytest.mark.parametrize("period", [4, 12])
def test_forecast_seasonal_alignment(data, period):
    res = ThetaModel(
        data,
        period=period,
        deseasonalize=True,
        use_test=False,
        difference=False,
    ).fit(use_mle=False)
    seasonal = res._seasonal
    comp = res.forecast_components(32)
    index = np.arange(data.shape[0], data.shape[0] + comp.shape[0])
    expected = seasonal[index % period]
    np.testing.assert_allclose(comp.seasonal, expected)


def test_auto(reset_randomstate):
    m = 250
    e = np.random.standard_normal(m)
    s = 10 * np.sin(np.linspace(0, np.pi, 12))
    s = np.tile(s, (m // 12 + 1))[:m]
    idx = pd.period_range("2000-01-01", freq="M", periods=m)
    x = e + s
    y = pd.DataFrame(10 + x - x.min(), index=idx)

    tm = ThetaModel(y, method="auto")
    assert tm.method == "mul"
    res = tm.fit()

    tm = ThetaModel(y, method="mul")
    assert tm.method == "mul"
    res2 = tm.fit()

    np.testing.assert_allclose(res.params, res2.params)

    tm = ThetaModel(y - y.mean(), method="auto")
    assert tm.method == "add"
    res3 = tm.fit()

    assert not np.allclose(res.params, res3.params)
