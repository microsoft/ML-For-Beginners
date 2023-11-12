from io import BytesIO
from itertools import product
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS


def gen_data(nobs, nvar, const, pandas=False, missing=0.0, weights=False):
    rs = np.random.RandomState(987499302)
    x = rs.standard_normal((nobs, nvar))
    cols = ["x{0}".format(i) for i in range(nvar)]
    if const:
        x = tools.add_constant(x)
        cols = ["const"] + cols
    if missing > 0.0:
        mask = rs.random_sample(x.shape) < missing
        x[mask] = np.nan
    if x.shape[1] > 1:
        y = x[:, :-1].sum(1) + rs.standard_normal(nobs)
    else:
        y = x.sum(1) + rs.standard_normal(nobs)
    w = rs.chisquare(5, y.shape[0]) / 5
    if pandas:
        idx = pd.date_range("12-31-1999", periods=nobs)
        x = pd.DataFrame(x, index=idx, columns=cols)
        y = pd.Series(y, index=idx, name="y")
        w = pd.Series(w, index=idx, name="weights")
    if not weights:
        w = None

    return y, x, w


nobs = (250,)
nvar = (3, 0)
tf = (True, False)
missing = (0, 0.1)
params = list(product(nobs, nvar, tf, tf, missing))
params = [param for param in params if param[1] + param[2] > 0]
ids = ["-".join(map(str, param)) for param in params]

basic_params = [param for param in params if params[2] and params[4]]
weighted_params = [param + (tf,) for param in params for tf in (True, False)]
weighted_ids = ["-".join(map(str, param)) for param in weighted_params]


@pytest.fixture(scope="module", params=params, ids=ids)
def data(request):
    return gen_data(*request.param)


@pytest.fixture(scope="module", params=basic_params, ids=ids)
def basic_data(request):
    return gen_data(*request.param)


@pytest.fixture(scope="module", params=weighted_params, ids=weighted_ids)
def weighted_data(request):
    return gen_data(*request.param)


def get_single(x, idx):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[idx]
    return x[idx]


def get_sub(x, idx, window):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.iloc[idx - window : idx]
        return np.asarray(out)
    return x[idx - window : idx]


def test_has_nan(data):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    has_nan = np.zeros(y.shape[0], dtype=bool)
    for i in range(100, y.shape[0] + 1):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        has_nan[i - 1] = np.squeeze(
            (np.any(np.isnan(_y)) or np.any(np.isnan(_x)))
        )
    assert_array_equal(mod._has_nan, has_nan)


def test_weighted_against_wls(weighted_data):
    y, x, w = weighted_data
    mod = RollingWLS(y, x, weights=w, window=100)
    res = mod.fit(use_t=True)
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        if w is not None:
            _w = get_sub(w, i, 100)
        else:
            _w = np.ones_like(_y)
        wls = WLS(_y, _x, weights=_w, missing="drop").fit()
        rolling_params = get_single(res.params, i - 1)
        rolling_nobs = get_single(res.nobs, i - 1)
        assert_allclose(rolling_params, wls.params)
        assert_allclose(rolling_nobs, wls.nobs)
        assert_allclose(get_single(res.ssr, i - 1), wls.ssr)
        assert_allclose(get_single(res.llf, i - 1), wls.llf)
        assert_allclose(get_single(res.aic, i - 1), wls.aic)
        assert_allclose(get_single(res.bic, i - 1), wls.bic)
        assert_allclose(get_single(res.centered_tss, i - 1), wls.centered_tss)
        assert_allclose(res.df_model, wls.df_model)
        assert_allclose(get_single(res.df_resid, i - 1), wls.df_resid)
        assert_allclose(get_single(res.ess, i - 1), wls.ess, atol=1e-8)
        assert_allclose(res.k_constant, wls.k_constant)
        assert_allclose(get_single(res.mse_model, i - 1), wls.mse_model)
        assert_allclose(get_single(res.mse_resid, i - 1), wls.mse_resid)
        assert_allclose(get_single(res.mse_total, i - 1), wls.mse_total)
        assert_allclose(
            get_single(res.rsquared, i - 1), wls.rsquared, atol=1e-8
        )
        assert_allclose(
            get_single(res.rsquared_adj, i - 1), wls.rsquared_adj, atol=1e-8
        )
        assert_allclose(
            get_single(res.uncentered_tss, i - 1), wls.uncentered_tss
        )


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
@pytest.mark.parametrize("use_t", [None, True, False])
def test_against_wls_inference(data, use_t, cov_type):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    res = mod.fit(use_t=use_t, cov_type=cov_type)
    ci = res.conf_int()

    # This is a smoke test of cov_params to make sure it works
    res.cov_params()

    # Skip to improve performance
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        wls = WLS(_y, _x, missing="drop").fit(use_t=use_t, cov_type=cov_type)
        assert_allclose(get_single(res.tvalues, i - 1), wls.tvalues)
        assert_allclose(get_single(res.bse, i - 1), wls.bse)
        assert_allclose(get_single(res.pvalues, i - 1), wls.pvalues, atol=1e-8)
        assert_allclose(get_single(res.fvalue, i - 1), wls.fvalue)
        with np.errstate(invalid="ignore"):
            assert_allclose(
                get_single(res.f_pvalue, i - 1), wls.f_pvalue, atol=1e-8
            )
        assert res.cov_type == wls.cov_type
        assert res.use_t == wls.use_t
        wls_ci = wls.conf_int()
        if isinstance(ci, pd.DataFrame):
            ci_val = ci.iloc[i - 1]
            ci_val = np.asarray(ci_val).reshape((-1, 2))
        else:
            ci_val = ci[i - 1].T
        assert_allclose(ci_val, wls_ci)


def test_raise(data):
    y, x, w = data

    mod = RollingWLS(y, x, window=100, missing="drop", weights=w)
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.all(np.isfinite(params[99:]))

    if not np.any(np.isnan(y)):
        return
    mod = RollingWLS(y, x, window=100, missing="skip")
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.any(np.isnan(params[100:]))


def test_error():
    y, x, _ = gen_data(250, 2, True)
    with pytest.raises(ValueError, match="reset must be a positive integer"):
        RollingWLS(y, x,).fit(reset=-1)
    with pytest.raises(ValueError):
        RollingWLS(y, x).fit(method="unknown")
    with pytest.raises(ValueError, match="min_nobs must be larger"):
        RollingWLS(y, x, min_nobs=1)
    with pytest.raises(ValueError, match="min_nobs must be larger"):
        RollingWLS(y, x, window=60, min_nobs=100)


def test_save_load(data):
    y, x, w = data
    res = RollingOLS(y, x, window=60).fit()
    fh = BytesIO()
    # test wrapped results load save pickle
    res.save(fh)
    fh.seek(0, 0)
    res_unpickled = res.__class__.load(fh)
    assert type(res_unpickled) is type(res)  # noqa: E721

    fh = BytesIO()
    # test wrapped results load save pickle
    res.save(fh, remove_data=True)
    fh.seek(0, 0)
    res_unpickled = res.__class__.load(fh)
    assert type(res_unpickled) is type(res)  # noqa: E721


def test_formula():
    y, x, w = gen_data(250, 3, True, pandas=True)
    fmla = "y ~ 1 + x0 + x1 + x2"
    data = pd.concat([y, x], axis=1)
    mod = RollingWLS.from_formula(fmla, window=100, data=data, weights=w)
    res = mod.fit()
    alt = RollingWLS(y, x, window=100)
    alt_res = alt.fit()
    assert_allclose(res.params, alt_res.params)
    ols_mod = RollingOLS.from_formula(fmla, window=100, data=data)
    ols_mod.fit()


@pytest.mark.matplotlib
def test_plot():
    import matplotlib.pyplot as plt

    y, x, w = gen_data(250, 3, True, pandas=True)
    fmla = "y ~ 1 + x0 + x1 + x2"
    data = pd.concat([y, x], axis=1)
    mod = RollingWLS.from_formula(fmla, window=100, data=data, weights=w)
    res = mod.fit()
    fig = res.plot_recursive_coefficient()
    assert isinstance(fig, plt.Figure)
    res.plot_recursive_coefficient(variables=2, alpha=None, figsize=(30, 7))
    res.plot_recursive_coefficient(variables="x0", alpha=None, figsize=(30, 7))
    res.plot_recursive_coefficient(
        variables=[0, 2], alpha=None, figsize=(30, 7)
    )
    res.plot_recursive_coefficient(
        variables=["x0"], alpha=None, figsize=(30, 7)
    )
    res.plot_recursive_coefficient(
        variables=["x0", "x1", "x2"], alpha=None, figsize=(30, 7)
    )
    with pytest.raises(ValueError, match="variable x4 is not an integer"):
        res.plot_recursive_coefficient(variables="x4")

    fig = plt.Figure()
    # Just silence the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = res.plot_recursive_coefficient(fig=fig)
    assert out is fig
    res.plot_recursive_coefficient(alpha=None, figsize=(30, 7))


@pytest.mark.parametrize("params_only", [True, False])
def test_methods(basic_data, params_only):
    y, x, _ = basic_data
    mod = RollingOLS(y, x, 150)
    res_inv = mod.fit(method="inv", params_only=params_only)
    res_lstsq = mod.fit(method="lstsq", params_only=params_only)
    res_pinv = mod.fit(method="pinv", params_only=params_only)
    assert_allclose(res_inv.params, res_lstsq.params)
    assert_allclose(res_inv.params, res_pinv.params)


@pytest.mark.parametrize("method", ["inv", "lstsq", "pinv"])
def test_params_only(basic_data, method):
    y, x, _ = basic_data
    mod = RollingOLS(y, x, 150)
    res = mod.fit(method=method, params_only=False)
    res_params_only = mod.fit(method=method, params_only=True)
    # use assert_allclose to incorporate for numerical errors on x86 platforms
    assert_allclose(res_params_only.params, res.params)


def test_min_nobs(basic_data):
    y, x, w = basic_data
    if not np.any(np.isnan(np.asarray(x))):
        return
    mod = RollingOLS(y, x, 150)
    res = mod.fit()
    # Ensures that the constraint binds
    min_nobs = res.nobs[res.nobs != 0].min() + 1
    mod = RollingOLS(y, x, 150, min_nobs=min_nobs)
    res = mod.fit()
    assert np.all(res.nobs[res.nobs != 0] >= min_nobs)


def test_expanding(basic_data):
    y, x, w = basic_data
    xa = np.asarray(x)
    mod = RollingOLS(y, x, 150, min_nobs=50, expanding=True)
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.all(np.isnan(params[:49]))
    first = np.where(np.cumsum(np.all(np.isfinite(xa), axis=1)) >= 50)[0][0]
    assert np.all(np.isfinite(params[first:]))
