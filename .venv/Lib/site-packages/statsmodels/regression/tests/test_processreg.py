# -*- coding: utf-8 -*-
from statsmodels.compat.platform import PLATFORM_OSX

from statsmodels.regression.process_regression import (
       ProcessMLE, GaussianCovariance)
import numpy as np
import pandas as pd
import pytest

import collections
import statsmodels.tools.numdiff as nd
from numpy.testing import assert_allclose, assert_equal


# Parameters for a test model, with or without additive
# noise.
def model1(noise):

    mn_par = np.r_[1, 0, -1, 0]
    sc_par = np.r_[1, 1]
    sm_par = np.r_[0.5, 0.1]

    if noise:
        no_par = np.r_[0.25, 0.25]
    else:
        no_par = np.array([])

    return mn_par, sc_par, sm_par, no_par


def setup1(n, get_model, noise):

    mn_par, sc_par, sm_par, no_par = get_model(noise)

    groups = np.kron(np.arange(n // 5), np.ones(5))
    time = np.kron(np.ones(n // 5), np.arange(5))
    time_z = (time - time.mean()) / time.std()

    x_mean = np.random.normal(size=(n, len(mn_par)))
    x_sc = np.random.normal(size=(n, len(sc_par)))
    x_sc[:, 0] = 1
    x_sc[:, 1] = time_z
    x_sm = np.random.normal(size=(n, len(sm_par)))
    x_sm[:, 0] = 1
    x_sm[:, 1] = time_z

    mn = np.dot(x_mean, mn_par)
    sc = np.exp(np.dot(x_sc, sc_par))
    sm = np.exp(np.dot(x_sm, sm_par))

    if noise:
        x_no = np.random.normal(size=(n, len(no_par)))
        x_no[:, 0] = 1
        x_no[:, 1] = time_z
        no = np.exp(np.dot(x_no, no_par))
    else:
        x_no = None

    y = mn.copy()

    gc = GaussianCovariance()

    ix = collections.defaultdict(lambda: [])
    for i, g in enumerate(groups):
        ix[g].append(i)

    for g, ii in ix.items():
        c = gc.get_cov(time[ii], sc[ii], sm[ii])
        r = np.linalg.cholesky(c)
        y[ii] += np.dot(r, np.random.normal(size=len(ii)))

    # Additive white noise
    if noise:
        y += no * np.random.normal(size=y.shape)

    return y, x_mean, x_sc, x_sm, x_no, time, groups


def run_arrays(n, get_model, noise):

    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise)

    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    return preg.fit()


@pytest.mark.slow
@pytest.mark.parametrize("noise", [False, True])
def test_arrays(noise):

    np.random.seed(8234)

    f = run_arrays(1000, model1, noise)
    mod = f.model

    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.concatenate(model1(noise))
    assert_allclose(f.params, epar, atol=0.3, rtol=0.3)

    # Test the fitted covariance matrix
    cv = f.covariance(mod.time[0:5], mod.exog_scale[0:5, :],
                      mod.exog_smooth[0:5, :])
    assert_allclose(cv, cv.T)  # Check symmetry
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)  # Check PSD

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, mod.endog)[0, 1] > 0.2, True)
    yhatm = f.predict(exog=mod.exog)
    assert_equal(yhat, yhatm)
    yhat0 = mod.predict(params=f.params, exog=mod.exog)
    assert_equal(yhat, yhat0)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))


def run_formula(n, get_model, noise):

    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise)

    df = pd.DataFrame({
        "y": y,
        "x1": x_mean[:, 0],
        "x2": x_mean[:, 1],
        "x3": x_mean[:, 2],
        "x4": x_mean[:, 3],
        "xsc1": x_sc[:, 0],
        "xsc2": x_sc[:, 1],
        "xsm1": x_sm[:, 0],
        "xsm2": x_sm[:, 1],
        "time": time,
        "groups": groups
    })

    if noise:
        df["xno1"] = x_no[:, 0]
        df["xno2"] = x_no[:, 1]

    mean_formula = "y ~ 0 + x1 + x2 + x3 + x4"
    scale_formula = "0 + xsc1 + xsc2"
    smooth_formula = "0 + xsm1 + xsm2"

    if noise:
        noise_formula = "0 + xno1 + xno2"
    else:
        noise_formula = None

    preg = ProcessMLE.from_formula(
        mean_formula,
        data=df,
        scale_formula=scale_formula,
        smooth_formula=smooth_formula,
        noise_formula=noise_formula,
        time="time",
        groups="groups")
    f = preg.fit()

    return f, df


@pytest.mark.slow
@pytest.mark.parametrize("noise", [False, True])
def test_formulas(noise):

    np.random.seed(8789)

    f, df = run_formula(1000, model1, noise)
    mod = f.model

    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.concatenate(model1(noise))
    assert_allclose(f.params, epar, atol=0.1, rtol=1)

    # Test the fitted covariance matrix
    exog_scale = pd.DataFrame(mod.exog_scale[0:5, :],
                              columns=["xsc1", "xsc2"])
    exog_smooth = pd.DataFrame(mod.exog_smooth[0:5, :],
                               columns=["xsm1", "xsm2"])
    cv = f.covariance(mod.time[0:5], exog_scale, exog_smooth)
    assert_allclose(cv, cv.T)
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, mod.endog)[0, 1] > 0.2, True)
    yhatm = f.predict(exog=df)
    assert_equal(yhat, yhatm)
    yhat0 = mod.predict(params=f.params, exog=df)
    assert_equal(yhat, yhat0)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))


# Test the score functions using numerical derivatives.
@pytest.mark.parametrize("noise", [False, True])
def test_score_numdiff(noise):

    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(1000, model1, noise)

    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    def loglike(x):
        return preg.loglike(x)

    q = x_mean.shape[1] + x_sc.shape[1] + x_sm.shape[1]
    if noise:
        q += x_no.shape[1]

    np.random.seed(342)

    atol = 2e-3 if PLATFORM_OSX else 1e-2
    for _ in range(5):
        par0 = preg._get_start()
        par = par0 + 0.1 * np.random.normal(size=q)
        score = preg.score(par)
        score_nd = nd.approx_fprime(par, loglike, epsilon=1e-7)
        assert_allclose(score, score_nd, atol=atol, rtol=1e-4)
