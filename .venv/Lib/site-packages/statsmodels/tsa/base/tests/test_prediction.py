from statsmodels.compat.pandas import MONTH_END

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tsa.base.prediction import PredictionResults


@pytest.fixture(params=[True, False])
def data(request):
    mean = np.arange(10.0)
    variance = np.arange(1, 11.0)
    if not request.param:
        return mean, variance
    idx = pd.date_range("2000-1-1", periods=10, freq=MONTH_END)
    return pd.Series(mean, index=idx), pd.Series(variance, index=idx)


def test_basic(data):
    is_pandas = isinstance(data[0], pd.Series)
    pred = PredictionResults(data[0], data[1])
    np.testing.assert_allclose(data[0], pred.predicted_mean)
    np.testing.assert_allclose(data[1], pred.var_pred_mean)
    if is_pandas:
        assert isinstance(pred.predicted_mean, pd.Series)
        assert isinstance(pred.var_pred_mean, pd.Series)
        assert isinstance(pred.se_mean, pd.Series)
    frame = pred.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(
        frame.columns == ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]
    )


@pytest.mark.parametrize("dist", [None, "norm", "t", stats.norm()])
def test_dist(data, dist):
    df = 10 if dist == "t" else None
    pred = PredictionResults(data[0], data[1], dist=dist, df=df)
    basic = PredictionResults(data[0], data[1])
    ci = pred.conf_int()
    basic_ci = basic.conf_int()
    if dist == "t":
        assert np.all(np.asarray(ci != basic_ci))
    else:
        assert np.all(np.asarray(ci == basic_ci))
