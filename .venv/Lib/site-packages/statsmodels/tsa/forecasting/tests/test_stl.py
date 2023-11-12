import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.datasets
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing,
)


@pytest.fixture(scope="module")
def data(request):
    rs = np.random.RandomState(987654321)
    err = rs.standard_normal(500)
    index = pd.date_range("1980-1-1", freq="M", periods=500)
    fourier = Fourier(12, 1)
    terms = fourier.in_sample(index)
    det = np.squeeze(np.asarray(terms @ np.array([[2], [1]])))
    for i in range(1, 500):
        err[i] += 0.9 * err[i - 1] + det[i]
    return pd.Series(err, index=index)


def test_smoke(data):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    res.forecast(37)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res.stl, STL)
    assert isinstance(res.result, DecomposeResult)
    assert isinstance(res.model, ARIMA)
    assert hasattr(res.model_result, "forecast")


@pytest.mark.matplotlib
def test_sharex(data):
    stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    plt = res.result.plot()
    grouper_view = plt.axes[0].get_shared_x_axes()
    sibs = grouper_view.get_siblings(plt.axes[1])
    assert len(sibs) == 4


MODELS = [
    (ARIMA, {"order": (2, 0, 0), "trend": "c"}),
    (ExponentialSmoothing, {"trend": True}),
    (AutoReg, {"lags": 2, "old_names": False}),
    (ETSModel, {}),
]
MODELS = MODELS[-1:]
IDS = [str(c[0]).split(".")[-1][:-2] for c in MODELS]


@pytest.mark.parametrize("config", MODELS, ids=IDS)
@pytest.mark.parametrize("horizon", [1, 7, 23])
def test_equivalence_forecast(data, config, horizon):
    model, kwargs = config

    stl = STL(data)
    stl_fit = stl.fit()
    resids = data - stl_fit.seasonal
    mod = model(resids, **kwargs)
    fit_kwarg = {}
    if model is ETSModel:
        fit_kwarg["disp"] = False
    res = mod.fit(**fit_kwarg)
    stlf = STLForecast(data, model, model_kwargs=kwargs).fit(
        fit_kwargs=fit_kwarg
    )

    seasonal = np.asarray(stl_fit.seasonal)[-12:]
    seasonal = np.tile(seasonal, 1 + horizon // 12)
    fcast = res.forecast(horizon) + seasonal[:horizon]
    actual = stlf.forecast(horizon)
    assert_allclose(actual, fcast, rtol=1e-4)
    if not hasattr(res, "get_prediction"):
        return
    pred = stlf.get_prediction(data.shape[0], data.shape[0] + horizon - 1)
    assert isinstance(pred, PredictionResults)
    assert_allclose(pred.predicted_mean, fcast, rtol=1e-4)

    half = data.shape[0] // 2
    stlf.get_prediction(half, data.shape[0] + horizon - 1)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=True)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=half // 2)
    if hasattr(data, "index"):
        loc = data.index[half + half // 2]
        a = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.strftime("%Y-%m-%d")
        )
        b = stlf.get_prediction(
            half, data.shape[0] + horizon - 1, dynamic=loc.to_pydatetime()
        )
        c = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc)
        assert_allclose(a.predicted_mean, b.predicted_mean, rtol=1e-4)
        assert_allclose(a.predicted_mean, c.predicted_mean, rtol=1e-4)


def test_exceptions(data):
    class BadModel:
        def __init__(self, *args, **kwargs):
            pass

    with pytest.raises(AttributeError, match="model must expose"):
        STLForecast(data, BadModel)

    class NoForecast(BadModel):
        def fit(self, *args, **kwargs):
            return BadModel()

    with pytest.raises(AttributeError, match="The model's result"):
        STLForecast(data, NoForecast).fit()

    class BadResult:
        def forecast(self, *args, **kwargs):
            pass

    class FakeModel(BadModel):
        def fit(self, *args, **kwargs):
            return BadResult()

    with pytest.raises(AttributeError, match="The model result does not"):
        STLForecast(data, FakeModel).fit().summary()

    class BadResultSummary(BadResult):
        def summary(self, *args, **kwargs):
            return object()

    class FakeModelSummary(BadModel):
        def fit(self, *args, **kwargs):
            return BadResultSummary()

    with pytest.raises(TypeError, match="The model result's summary"):
        STLForecast(data, FakeModelSummary).fit().summary()


@pytest.fixture(scope="function")
def sunspots():
    df = statsmodels.datasets.sunspots.load_pandas().data
    df.index = np.arange(df.shape[0])
    return df.iloc[:, 0]


def test_get_prediction(sunspots):
    # GH7309
    stlf_model = STLForecast(
        sunspots, model=ARIMA, model_kwargs={"order": (2, 2, 0)}, period=11
    )
    stlf_res = stlf_model.fit()
    pred = stlf_res.get_prediction()
    assert pred.predicted_mean.shape == (309,)
    assert pred.var_pred_mean.shape == (309,)


@pytest.mark.parametrize("not_implemented", [True, False])
def test_no_var_pred(sunspots, not_implemented):
    class DummyPred:
        def __init__(self, predicted_mean, row_labels):
            self.predicted_mean = predicted_mean
            self.row_labels = row_labels

            def f():
                raise NotImplementedError

            if not_implemented:
                self.forecast = property(f)

    class DummyRes:
        def __init__(self, res):
            self._res = res

        def forecast(self, *args, **kwargs):
            return self._res.forecast(*args, **kwargs)

        def get_prediction(self, *args, **kwargs):
            pred = self._res.get_prediction(*args, **kwargs)

            return DummyPred(pred.predicted_mean, pred.row_labels)

    class DummyMod:
        def __init__(self, y):
            self._mod = ARIMA(y)

        def fit(self, *args, **kwargs):
            res = self._mod.fit(*args, **kwargs)
            return DummyRes(res)

    stl_mod = STLForecast(sunspots, model=DummyMod, period=11)
    stl_res = stl_mod.fit()
    with pytest.warns(UserWarning, match="The variance of"):
        pred = stl_res.get_prediction()
    assert np.all(np.isnan(pred.var_pred_mean))
