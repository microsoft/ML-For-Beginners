from typing import NamedTuple

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest

from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
    ARDL,
    UECM,
    ARDLResults,
    ardl_select_order,
)
from statsmodels.tsa.deterministic import DeterministicProcess

dane_data = danish_data.load_pandas().data


class Dataset(NamedTuple):
    y: pd.Series
    x: pd.DataFrame


@pytest.fixture(scope="module", params=[None, 0, 3, [1, 2, 4]])
def lags(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        2,
        [1, 3, 5],
        {"lry": 3, "ibo": 2, "ide": 1},
        {"lry": 3, "ibo": [2], "ide": [1, 3]},
    ],
)
def order(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        2,
        {"lry": 3, "ibo": 2, "ide": 1},
    ],
)
def uecm_order(request):
    return request.param


@pytest.fixture(scope="module", params=[None, 3])
def uecm_lags(request):
    return request.param


@pytest.fixture(scope="module", params=["n", "c", "ct"])
def trend(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def causal(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 3])
def maxlag(request):
    return request.param


@pytest.fixture(scope="module", params=[2, {"lry": 3, "ibo": 2, "ide": 1}])
def maxorder(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def seasonal(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def use_numpy(request):
    return request.param


@pytest.fixture(scope="module", params=[None, 10])
def hold_back(request):
    return request.param


@pytest.fixture(scope="module")
def data():
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    return Dataset(y, x)


@pytest.fixture(scope="module", params=[None, 2])
def fixed(request):
    if request.param is None:
        return None
    index = dane_data.lrm.index
    gen = np.random.default_rng(0)
    return pd.DataFrame(
        gen.standard_t(10, (dane_data.lrm.shape[0], 2)),
        index=index,
        columns=["z0", "z1"],
    )


def check_results(res: ARDLResults):
    model: ARDL = res.model
    n, k = model._x.shape
    assert model.score(res.params).shape == (k,)
    assert model.hessian(res.params).shape == (k, k)
    assert isinstance(model.loglike(res.params), float)
    assert model.information(res.params).shape == (k, k)
    assert isinstance(model.exog_names, list)
    assert model.ar_lags is None or isinstance(model.ar_lags, list)
    assert isinstance(model.seasonal, bool)
    assert isinstance(model.hold_back, int)
    assert isinstance(model.ardl_order, tuple)
    assert isinstance(model.df_model, int)
    assert isinstance(model.nobs, int)
    assert isinstance(model.endog_names, str)
    assert isinstance(model.k_constant, int)
    res.summary()
    res.test_heteroskedasticity()
    res.diagnostic_summary()
    assert res.ar_lags is None or isinstance(res.ar_lags, list)
    assert res.df_resid == (res.nobs - res.df_model)
    assert res.scale == res.sigma2
    assert isinstance(res.fpe, float)


def _convert_to_numpy(data, fixed, order, seasonal, use_numpy):
    y = data.y
    x = data.x
    z = fixed
    period = None
    if use_numpy:
        y = np.asarray(y)
        x = np.asarray(x)
        if isinstance(order, dict):
            order = {i: v for i, v in enumerate(order.values())}
        if fixed is not None:
            z = np.asarray(fixed)
        period = 4 if seasonal else None
    return y, x, z, order, period


def test_model_init(
    data: Dataset, lags, order, trend, causal, fixed, use_numpy, seasonal
):
    y, x, z, order, period = _convert_to_numpy(
        data, fixed, order, seasonal, use_numpy
    )

    mod = ARDL(
        y,
        lags,
        x,
        order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    res = mod.fit()
    check_results(res)
    res.predict()
    fixed_oos = None
    if z is not None:
        fixed_oos = np.array(z)[:12]
        if isinstance(z, pd.DataFrame):
            fixed_oos = pd.DataFrame(fixed_oos, columns=z.columns)
    exog_oos = None
    if x is not None:
        exog_oos = np.array(x)[:12]
        if isinstance(x, pd.DataFrame):
            exog_oos = pd.DataFrame(exog_oos, columns=x.columns)

    res.forecast(12, exog=exog_oos, fixed=fixed_oos)


def test_ardl_order_exceptions(data):
    with pytest.raises(ValueError, match="lags must be a non-negative"):
        ARDL(data.y, -1)
    with pytest.raises(
        ValueError, match="All values in lags must be positive"
    ):
        ARDL(data.y, [-1, 0, 2])
    with pytest.raises(ValueError, match="integer orders must be at least"):
        ARDL(data.y, 2, data.x, order=0, causal=True)
    with pytest.raises(ValueError, match="integer orders must be at least"):
        ARDL(data.y, 2, data.x, -1, causal=False)
    with pytest.raises(
        ValueError, match="sequence orders must be strictly positive"
    ):
        ARDL(
            data.y, 2, data.x, {"lry": [0, 1], "ibo": 3, "ide": 0}, causal=True
        )
    with pytest.raises(
        TypeError, match="sequence orders must contain non-negative"
    ):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, "apple"], "ibo": 3, "ide": 1},
            causal=True,
        )
    with pytest.raises(
        ValueError, match="sequence orders must contain distinct"
    ):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, 1, 2, 3], "ibo": 3, "ide": [1, 1, 1]},
            causal=True,
        )
    with pytest.raises(
        ValueError, match="sequence orders must be strictly positive"
    ):
        ARDL(data.y, 2, data.x, [0, 1, 2], causal=True)


def test_ardl_order_keys_exceptions(data):
    with pytest.raises(
        ValueError, match="order dictionary contains keys for exogenous"
    ):
        ARDL(
            data.y,
            2,
            data.x,
            {"lry": [1, 2], "ibo": 3, "other": 4},
            causal=False,
        )
    with pytest.warns(
        SpecificationWarning, match="exog contains variables that"
    ):
        ARDL(data.y, 2, data.x, {"lry": [1, 2]}, causal=False)


def test_ardl_deterministic_exceptions(data):
    with pytest.raises(TypeError):
        ARDL(data.y, 2, data.x, 2, deterministic="seasonal")
    with pytest.warns(
        SpecificationWarning, match="When using deterministic, trend"
    ):
        deterministic = DeterministicProcess(
            data.y.index, constant=True, order=1
        )
        ARDL(data.y, 2, data.x, 2, deterministic=deterministic, trend="ct")


def test_ardl_holdback_exceptions(data):
    with pytest.raises(ValueError, match="hold_back must be >="):
        ARDL(data.y, 2, data.x, 2, hold_back=1)


def test_ardl_fixed_exceptions(data):
    fixed = np.random.standard_normal((2, 200))
    with pytest.raises(ValueError, match="fixed must be an"):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)
    fixed = np.random.standard_normal((dane_data.lrm.shape[0], 2))
    fixed[20, 0] = -np.inf
    with pytest.raises(ValueError, match="fixed must be an"):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)


def test_ardl_select_order(
    data: Dataset,
    maxlag,
    maxorder,
    trend,
    causal,
    fixed,
    use_numpy,
    seasonal,
    hold_back,
):
    y, x, z, maxorder, period = _convert_to_numpy(
        data, fixed, maxorder, seasonal, use_numpy
    )
    res = ardl_select_order(
        y,
        maxlag,
        x,
        maxorder,
        trend,
        fixed=fixed,
        causal=causal,
        hold_back=hold_back,
        period=period,
        seasonal=seasonal,
        glob=seasonal,
    )
    assert isinstance(res.model, ARDL)
    assert isinstance(res.aic, pd.Series)
    assert isinstance(res.bic, pd.Series)
    assert isinstance(res.hqic, pd.Series)
    assert res.period == period
    assert res.trend == trend
    assert res.seasonal == seasonal
    assert isinstance(res.dl_lags, dict)
    assert res.ar_lags is None or isinstance(res.ar_lags, list)


def test_ardl_no_regressors(data):
    res = ARDL(
        data.y,
        None,
        data.x,
        {"lry": None, "ibo": None, "ide": None},
        trend="n",
    ).fit()
    assert res.params.shape[0] == 0
    check_results(res)


def test_ardl_only_y_lag(data):
    res = ARDL(data.y, 3, data.x, None, trend="n").fit()
    assert res.params.shape[0] == 3
    check_results(res)


def test_ardl_only_x(data):
    res = ARDL(
        data.y, None, data.x, {"lry": 1, "ibo": 2, "ide": 3}, trend="n"
    ).fit()
    assert res.params.shape[0] == 9
    res = ARDL(
        data.y,
        None,
        data.x,
        {"lry": 1, "ibo": 2, "ide": 3},
        trend="n",
        causal=True,
    ).fit()
    assert res.params.shape[0] == 6
    check_results(res)


def test_ardl_only_trend(data):
    res = ARDL(data.y, None, data.x, None, trend="c").fit()
    assert res.params.shape[0] == 1
    check_results(res)


def test_ardl_only_seasonal(data):
    res = ARDL(data.y, None, data.x, None, trend="n", seasonal=True).fit()
    assert res.params.shape[0] == 4
    check_results(res)


def test_ardl_only_deterministic(data):
    deterministic = DeterministicProcess(data.y.index, constant=True, order=3)
    res = ARDL(
        data.y, None, data.x, None, trend="n", deterministic=deterministic
    ).fit()
    assert res.params.shape[0] == 4
    check_results(res)


def test_ardl_no_endog_exog(data):
    res = ARDL(data.y, None, data.x, None, trend="ct", seasonal=True).fit()
    assert res.params.shape[0] == 5
    check_results(res)


def test_ardl_no_exog(data):
    res = ARDL(data.y, [1, 4], data.x, None, trend="ct", seasonal=True).fit()
    assert res.params.shape[0] == 7
    check_results(res)


def test_ardl_parameter_names(data):
    mod = ARDL(data.y, 2, data.x, 2, causal=True, trend="c")
    expected = [
        "const",
        "lrm.L1",
        "lrm.L2",
        "lry.L1",
        "lry.L2",
        "ibo.L1",
        "ibo.L2",
        "ide.L1",
        "ide.L2",
    ]
    assert mod.exog_names == expected
    mod = ARDL(
        np.asarray(data.y), 2, np.asarray(data.x), 2, causal=False, trend="ct"
    )
    expected = [
        "const",
        "trend",
        "y.L1",
        "y.L2",
        "x0.L0",
        "x0.L1",
        "x0.L2",
        "x1.L0",
        "x1.L1",
        "x1.L2",
        "x2.L0",
        "x2.L1",
        "x2.L2",
    ]
    assert mod.exog_names == expected
    mod = ARDL(
        np.asarray(data.y),
        [2],
        np.asarray(data.x),
        None,
        causal=False,
        trend="n",
        seasonal=True,
        period=4,
    )
    expected = ["s(1,4)", "s(2,4)", "s(3,4)", "s(4,4)", "y.L2"]
    assert mod.exog_names == expected


@pytest.mark.matplotlib
def test_diagnostics_plot(data, close_figures):
    import matplotlib.figure

    res = ARDL(
        data.y,
        2,
        data.x,
        {"lry": 3, "ibo": 2, "ide": [1, 3]},
        trend="ct",
        seasonal=True,
    ).fit()

    fig = res.plot_diagnostics()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_against_autoreg(data, trend, seasonal):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()
    assert_allclose(ar_res.params, ardl_res.params)
    assert ar_res.ar_lags == ardl_res.ar_lags
    assert ar.trend == ardl.trend
    assert ar.seasonal == ardl.seasonal

    ar_fcast = ar_res.forecast(12)
    ardl_fcast = ardl_res.forecast(12)
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)

    ar_fcast = ar_res.predict()
    ardl_fcast = ardl_res.predict()
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)


@pytest.mark.parametrize("start", [None, 0, 2, 4])
@pytest.mark.parametrize("end", [None, 20])
@pytest.mark.parametrize("dynamic", [20, True])
def test_against_autoreg_predict_start_end(
    data, trend, seasonal, start, end, dynamic
):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()

    ar_fcast = ar_res.predict(start=start, end=end, dynamic=dynamic)
    ardl_fcast = ardl_res.predict(start=start, end=end, dynamic=dynamic)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)
    assert_allclose(ar_fcast, ardl_fcast)


def test_invalid_init(data):
    with pytest.raises(ValueError, match="lags must be a non-negative"):
        ARDL(data.y, -1)
    with pytest.raises(
        ValueError, match="All values in lags must be positive"
    ):
        ARDL(data.y, [-1, 1, 2])
    with pytest.raises(
        ValueError, match="All values in lags must be positive"
    ):
        ARDL(data.y, [1, 2, 2, 3])
    with pytest.raises(ValueError, match="hold_back must be "):
        ARDL(data.y, 3, data.x, 4, hold_back=3)


def test_prediction_oos_no_new_data(data):
    res = ARDL(data.y, 2, data.x, 3, causal=True).fit()
    val = res.forecast(1)
    assert val.shape[0] == 1
    res = ARDL(data.y, [3], data.x, [3]).fit()
    val = res.forecast(3)
    assert val.shape[0] == 3


def test_prediction_exceptions(data, fixed, use_numpy):
    y, x, z, order, _ = _convert_to_numpy(data, None, 3, False, use_numpy)
    res = ARDL(y, 2, x, 3, causal=False).fit()
    with pytest.raises(ValueError, match="exog_oos must be"):
        res.forecast(1)
    if isinstance(x, pd.DataFrame):
        exog_oos = np.asarray(data.x)[:12]
        with pytest.raises(
            TypeError, match="exog_oos must be a DataFrame when"
        ):
            res.forecast(12, exog=exog_oos)
        with pytest.raises(ValueError, match="must have the same columns"):
            res.forecast(12, exog=data.x.iloc[:12, :1])


def test_prediction_replacements(data, fixed):
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    direct = res.predict()
    alt = res.predict(exog=data.x)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)

    res = ARDL(data.y, 4, data.x, [1, 3], fixed=fixed).fit()
    direct = res.predict()
    alt = res.predict(fixed=fixed)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)


def test_prediction_wrong_shape(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, x, [1, 3]).fit()
    with pytest.raises(ValueError, match="exog must have the same number"):
        res.predict(exog=np.asarray(data.x)[:, :1])
    with pytest.raises(
        ValueError, match="exog must have the same number of rows"
    ):
        res.predict(exog=np.asarray(data.x)[:-2])
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    with pytest.raises(ValueError, match="exog must have the same columns"):
        res.predict(exog=data.x.iloc[:, :1])
    with pytest.raises(
        ValueError, match="exog must have the same number of rows"
    ):
        res.predict(exog=data.x.iloc[:-2])


def test_prediction_wrong_shape_fixed(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, fixed=x).fit()
    with pytest.raises(ValueError, match="fixed must have the same number"):
        res.predict(fixed=np.asarray(data.x)[:, :1])
    with pytest.raises(
        ValueError, match="fixed must have the same number of rows"
    ):
        res.predict(fixed=np.asarray(data.x)[:-2])
    res = ARDL(data.y, 4, fixed=data.x).fit()
    with pytest.raises(ValueError, match="fixed must have the same number"):
        res.predict(fixed=data.x.iloc[:, :1])
    with pytest.raises(
        ValueError, match="fixed must have the same number of rows"
    ):
        res.predict(fixed=data.x.iloc[:-2])


def test_insuficient_oos(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, fixed=x).fit()
    with pytest.raises(ValueError, match="fixed_oos must be provided"):
        res.forecast(12)
    with pytest.raises(ValueError, match="fixed_oos must have at least"):
        res.forecast(12, fixed=x[:11])
    res = ARDL(data.y, 4, data.x, 3, causal=True).fit()
    with pytest.raises(ValueError, match="exog_oos must be provided"):
        res.forecast(12)
    with pytest.raises(ValueError, match="exog_oos must have at least"):
        res.forecast(12, exog=data.x.iloc[-10:])


def test_insuficient_data(data):
    with pytest.raises(ValueError, match=r"The number of regressors \(36\)"):
        ARDL(data.y, 20, data.x, 4)


def test_forecast_date(data):
    res = ARDL(data.y, 3).fit()
    numeric = res.forecast(12)
    date = res.forecast("1990-07-01")
    assert_allclose(numeric, date)
    assert_index_equal(numeric.index, date.index)


def test_get_prediction(data):
    res = ARDL(data.y, 3).fit()
    ar_res = AutoReg(data.y, 3).fit()
    pred = res.get_prediction(end="2020-01-01")
    ar_pred = ar_res.get_prediction(end="2020-01-01")
    assert_allclose(pred.predicted_mean, ar_pred.predicted_mean)
    assert_allclose(pred.var_pred_mean, ar_pred.var_pred_mean)


@pytest.mark.matplotlib
@pytest.mark.smoke
@pytest.mark.parametrize("trend", ["n", "c", "ct"])
@pytest.mark.parametrize("seasonal", [True, False])
def test_ardl_smoke_plots(data, seasonal, trend, close_figures):
    from matplotlib.figure import Figure

    mod = ARDL(
        data.y,
        3,
        trend=trend,
        seasonal=seasonal,
    )
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=100)
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=75, alpha=None, in_sample=False)
    assert isinstance(fig, Figure)
    assert isinstance(res.summary(), Summary)


def test_uecm_model_init(
    data: Dataset,
    uecm_lags,
    uecm_order,
    trend,
    causal,
    fixed,
    use_numpy,
    seasonal,
):
    y, x, z, uecm_order, period = _convert_to_numpy(
        data, fixed, uecm_order, seasonal, use_numpy
    )

    mod = UECM(
        y,
        uecm_lags,
        x,
        uecm_order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    res = mod.fit()
    check_results(res)
    res.predict()

    ardl = ARDL(
        y,
        uecm_lags,
        x,
        uecm_order,
        trend,
        causal=causal,
        fixed=z,
        seasonal=seasonal,
        period=period,
    )
    uecm = UECM.from_ardl(ardl)
    uecm_res = uecm.fit()
    check_results(uecm_res)
    uecm_res.predict()


def test_from_ardl_none(data):
    with pytest.warns(SpecificationWarning):
        mod = UECM.from_ardl(
            ARDL(data.y, 2, data.x, {"lry": 2, "ide": 2, "ibo": None})
        )
    assert mod.ardl_order == (2, 2, 2)


def test_uecm_model_formula(
    data: Dataset,
    uecm_lags,
    uecm_order,
    trend,
    causal,
    fixed,
    seasonal,
):
    fmla = "lrm ~ lry + ibo + ide"
    df = pd.concat([data.y, data.x], axis=1)
    if fixed is not None:
        fmla += " | " + " + ".join(fixed.columns)
        df = pd.concat([df, fixed], axis=1)
    mod = UECM.from_formula(
        fmla,
        df,
        uecm_lags,
        uecm_order,
        trend,
        causal=causal,
        seasonal=seasonal,
    )
    res = mod.fit()
    check_results(res)
    res.predict()


def test_uecm_errors(data):
    with pytest.raises(TypeError, match="order must be None"):
        UECM(data.y, 2, data.x, [0, 1, 2])
    with pytest.raises(TypeError, match="lags must be an"):
        UECM(data.y, [1, 2], data.x, 2)
    with pytest.raises(TypeError, match="order values must be positive"):
        UECM(data.y, 2, data.x, {"ibo": [1, 2]})
    with pytest.raises(ValueError, match="Model must contain"):
        UECM(data.y, 2, data.x, None)
    with pytest.raises(ValueError, match="All included exog"):
        UECM(data.y, 2, data.x, {"lry": 2, "ide": 2, "ibo": 0})
    with pytest.raises(ValueError, match="hold_back must be"):
        UECM(data.y, 3, data.x, 5, hold_back=4)
    with pytest.raises(ValueError, match="The number of"):
        UECM(data.y, 20, data.x, 4)
    ardl = ARDL(data.y, 2, data.x, {"lry": [1, 2], "ide": 2, "ibo": 2})
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, 2, data.x, {"lry": [0, 2], "ide": 2, "ibo": 2})
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, [1, 3], data.x, 2)
    with pytest.raises(ValueError, match="UECM can only be created from"):
        UECM.from_ardl(ardl)
    res = UECM(data.y, 2, data.x, 2).fit()
    with pytest.raises(NotImplementedError):
        res.predict(end=100)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=True)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=25)


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("use_t", [True, False])
def test_uecm_ci_repr(use_numpy, use_t):
    y = dane_data.lrm
    x = dane_data[["lry", "ibo", "ide"]]
    if use_numpy:
        y = np.asarray(y)
        x = np.asarray(x)
    mod = UECM(y, 3, x, 3)
    res = mod.fit(use_t=use_t)
    if use_numpy:
        ci_params = res.params[:5].copy()
        ci_params /= ci_params[1]
    else:
        ci_params = res.params.iloc[:5].copy()
        ci_params /= ci_params["lrm.L1"]
    assert_allclose(res.ci_params, ci_params)
    assert res.ci_bse.shape == (5,)
    assert res.ci_tvalues.shape == (5,)
    assert res.ci_pvalues.shape == (5,)
    assert "Cointegrating Vector" in str(res.ci_summary())
    assert res.ci_conf_int().shape == (5, 2)
    assert res.ci_cov_params().shape == (5, 5)
    assert res.ci_resids.shape == dane_data.lrm.shape


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_bounds_test(case):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    expected = {
        1: 0.7109023,
        2: 5.116768,
        3: 6.205875,
        4: 5.430622,
        5: 6.785325,
    }
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.stat, expected[case])
    assert "BoundsTestResult" in str(bounds_result)


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_bounds_test_simulation(case):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(
        case=case, asymptotic=False, seed=[1, 2, 3, 4], nsim=10_000
    )
    assert (bounds_result.p_values >= 0.0).all()
    assert (bounds_result.p_values <= 1.0).all()
    assert (bounds_result.crit_vals > 0.0).all().all()


@pytest.mark.parametrize(
    "seed",
    [None, np.random.RandomState(0), 0, [1, 2], np.random.default_rng([1, 2])],
)
def test_bounds_test_seed(seed):
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(
        case=3, asymptotic=False, seed=seed, nsim=10_000
    )
    assert (bounds_result.p_values >= 0.0).all()
    assert (bounds_result.p_values <= 1.0).all()
    assert (bounds_result.crit_vals > 0.0).all().all()


def test_bounds_test_simulate_order():
    mod = UECM(
        dane_data.lrm,
        3,
        dane_data[["lry", "ibo", "ide"]],
        {"lry": 1, "ibo": 3, "ide": 2},
    )
    res = mod.fit()
    bounds_result = res.bounds_test(3)
    assert "BoundsTestResult" in str(bounds_result)
    bounds_result_sim = res.bounds_test(
        3, asymptotic=False, nsim=10_000, seed=[1, 2, 3]
    )
    assert_allclose(bounds_result.stat, bounds_result_sim.stat)
    assert (bounds_result_sim.p_values > bounds_result.p_values).all()
