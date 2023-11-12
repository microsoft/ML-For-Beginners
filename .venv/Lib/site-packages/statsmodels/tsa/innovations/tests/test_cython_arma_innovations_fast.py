"""
Tests for fast version of ARMA innovations algorithm
"""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose

from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX


def test_brockwell_davis_ex533():
    # See Brockwell and Davis (2009) - Time Series Theory and Methods
    # Example 5.3.3: ARMA(1, 1) process, p.g. 177
    nobs = 10

    ar_params = np.array([0.2])
    ma_params = np.array([0.4])
    sigma2 = 8.92
    p = len(ar_params)
    q = len(ma_params)
    m = max(p, q)

    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # First, get the autocovariance of the process
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    unconditional_variance = (
        sigma2 * (1 + 2 * ar_params[0] * ma_params[0] + ma_params[0]**2) /
        (1 - ar_params[0]**2))
    assert_allclose(arma_process_acovf[0], unconditional_variance)

    # Next, get the autocovariance of the transformed process
    # Note: as required by {{prefix}}arma_transformed_acovf, we first divide
    # through by sigma^2
    arma_process_acovf /= sigma2
    unconditional_variance /= sigma2
    transformed_acovf = _arma_innovations.darma_transformed_acovf_fast(
        ar, ma, arma_process_acovf)
    acovf, acovf2 = (np.array(arr) for arr in transformed_acovf)

    # `acovf` is an m^2 x m^2 matrix, where m = max(p, q)
    # but it is only valid for the autocovariances of the first m observations
    # (this means in particular that the block `acovf[m:, m:]` should *not* be
    # used)
    # `acovf2` then contains the (time invariant) autocovariance terms for
    # the observations m + 1, ..., nobs - since the autocovariance is the same
    # for these terms, to save space we do not construct the autocovariance
    # matrix as we did for the first m terms. Thus `acovf2[0]` is the variance,
    # `acovf2[1]` is the first autocovariance, etc.

    # Test the autocovariance function for observations m + 1, ..., nobs
    # (it is time invariant here)
    assert_equal(acovf2.shape, (nobs - m,))
    assert_allclose(acovf2[0], 1 + ma_params[0]**2)
    assert_allclose(acovf2[1], ma_params[0])
    assert_allclose(acovf2[2:], 0)

    # Test the autocovariance function for observations 1, ..., m
    # (it is time varying here)
    assert_equal(acovf.shape, (m * 2, m * 2))

    # (we need to check `acovf[:m * 2, :m]`, i.e. `acovf[:2, :1])`
    ix = np.diag_indices_from(acovf)
    ix_lower = (ix[0][:-1] + 1, ix[1][:-1])

    # acovf[ix] is the diagonal, and we want to check the first m
    # elements of the diagonal
    assert_allclose(acovf[ix][:m], unconditional_variance)

    # acovf[ix_lower] is the first lower off-diagonal
    assert_allclose(acovf[ix_lower][:m], ma_params[0])

    # Now, check that we compute the moving average coefficients and the
    # associated variances correctly
    out = _arma_innovations.darma_innovations_algo_fast(
        nobs, ar_params, ma_params, acovf, acovf2)
    theta = np.array(out[0])
    v = np.array(out[1])

    # Test v (see eq. 5.3.13)
    desired_v = np.zeros(nobs)
    desired_v[0] = unconditional_variance
    for i in range(1, nobs):
        desired_v[i] = 1 + (1 - 1 / desired_v[i - 1]) * ma_params[0]**2
    assert_allclose(v, desired_v)

    # Test theta (see eq. 5.3.13)
    # Note that they will have shape (nobs, m + 1) here, not (nobs, nobs - 1)
    # as in the original (non-fast) version
    assert_equal(theta.shape, (nobs, m + 1))
    desired_theta = np.zeros(nobs)
    for i in range(1, nobs):
        desired_theta[i] = ma_params[0] / desired_v[i - 1]
    assert_allclose(theta[:, 0], desired_theta)
    assert_allclose(theta[:, 1:], 0)

    # Test against Table 5.3.1
    endog = np.array([
        -1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228,
        -1.093])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    # Note: Table 5.3.1 has \hat X_n+1 = -0.5340 for n = 1, but this seems to
    # be a typo, since equation 5.3.12 gives the form of the prediction
    # equation as \hat X_n+1 = \phi X_n + \theta_n1 (X_n - \hat X_n)
    # Then for n = 1 we have:
    # \hat X_n+1 = 0.2 (-1.1) + (0.2909) (-1.1 - 0) = -0.5399
    # And for n = 2 if we use what we have computed, then we get:
    # \hat X_n+1 = 0.2 (0.514) + (0.3833) (0.514 - (-0.54)) = 0.5068
    # as desired, whereas if we used the book's number for n=1 we would get:
    # \hat X_n+1 = 0.2 (0.514) + (0.3833) (0.514 - (-0.534)) = 0.5045
    # which is not what Table 5.3.1 shows.
    desired_hat = np.array([
        0, -0.540, 0.5068, -0.1321, -0.4539, 0.7046, -0.5620, -0.3614,
        -0.8748, -0.3869])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=1e-4)


def test_brockwell_davis_ex534():
    # See Brockwell and Davis (2009) - Time Series Theory and Methods
    # Example 5.3.4: ARMA(1, 1) process, p.g. 178
    nobs = 10

    ar_params = np.array([1, -0.24])
    ma_params = np.array([0.4, 0.2, 0.1])
    sigma2 = 1
    p = len(ar_params)
    q = len(ma_params)
    m = max(p, q)

    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # First, get the autocovariance of the process
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    assert_allclose(arma_process_acovf[:3],
                    [7.17133, 6.44139, 5.06027], atol=1e-5)

    # Next, get the autocovariance of the transformed process
    transformed_acovf = _arma_innovations.darma_transformed_acovf_fast(
        ar, ma, arma_process_acovf)
    acovf, acovf2 = (np.array(arr) for arr in transformed_acovf)
    # See test_brockwell_davis_ex533 for details on acovf vs acovf2

    # Test acovf
    assert_equal(acovf.shape, (m * 2, m * 2))

    ix = np.diag_indices_from(acovf)
    ix_lower1 = (ix[0][:-1] + 1, ix[1][:-1])
    ix_lower2 = (ix[0][:-2] + 2, ix[1][:-2])
    ix_lower3 = (ix[0][:-3] + 3, ix[1][:-3])
    ix_lower4 = (ix[0][:-4] + 4, ix[1][:-4])

    assert_allclose(acovf[ix][:m], 7.17133, atol=1e-5)
    desired = [6.44139, 6.44139, 0.816]
    assert_allclose(acovf[ix_lower1][:m], desired, atol=1e-5)
    assert_allclose(acovf[ix_lower2][0], 5.06027, atol=1e-5)
    assert_allclose(acovf[ix_lower2][1:m], 0.34, atol=1e-5)
    assert_allclose(acovf[ix_lower3][:m], 0.1, atol=1e-5)
    assert_allclose(acovf[ix_lower4][:m], 0, atol=1e-5)

    # Test acovf2
    assert_equal(acovf2.shape, (nobs - m,))
    assert_allclose(acovf2[:4], [1.21, 0.5, 0.24, 0.1])
    assert_allclose(acovf2[4:], 0)

    # Test innovations algorithm output
    out = _arma_innovations.darma_innovations_algo_fast(
        nobs, ar_params, ma_params, acovf, acovf2)
    theta = np.array(out[0])
    v = np.array(out[1])

    # Test v (see Table 5.3.2)
    desired_v = [7.1713, 1.3856, 1.0057, 1.0019, 1.0016, 1.0005, 1.0000,
                 1.0000, 1.0000, 1.0000]
    assert_allclose(v, desired_v, atol=1e-4)

    # Test theta (see Table 5.3.2)
    assert_equal(theta.shape, (nobs, m + 1))
    desired_theta = np.array([
        [0, 0.8982, 1.3685, 0.4008, 0.3998, 0.3992, 0.4000, 0.4000, 0.4000,
         0.4000],
        [0, 0, 0.7056, 0.1806, 0.2020, 0.1995, 0.1997, 0.2000, 0.2000, 0.2000],
        [0, 0, 0, 0.0139, 0.0722, 0.0994, 0.0998, 0.0998, 0.0999, 0.1]]).T
    assert_allclose(theta[:, :m], desired_theta, atol=1e-4)
    assert_allclose(theta[:, m:], 0)

    # Test innovations filter output
    endog = np.array([1.704, 0.527, 1.041, 0.942, 0.555, -1.002, -0.585, 0.010,
                      -0.638, 0.525])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    desired_hat = np.array([
        0, 1.5305, -0.1710, 1.2428, 0.7443, 0.3138, -1.7293, -0.1688,
        0.3193, -0.8731])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=1e-4)


@pytest.mark.parametrize("ar_params,ma_params,sigma2", [
    (np.array([]), np.array([]), 1),
    (np.array([0.]), np.array([0.]), 1),
    (np.array([0.9]), np.array([]), 1),
    (np.array([]), np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_innovations_algo_filter_kalman_filter(ar_params, ma_params, sigma2):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process

    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    endog = np.random.normal(size=10)
    nobs = len(endog)

    # Innovations algorithm approach
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    transformed_acov = _arma_innovations.darma_transformed_acovf_fast(
        ar, ma, arma_process_acovf / sigma2)
    acovf, acovf2 = (np.array(mv) for mv in transformed_acov)
    theta, r = _arma_innovations.darma_innovations_algo_fast(
        nobs, ar_params, ma_params, acovf, acovf2)
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params,
                                                   theta)

    v = np.array(r) * sigma2
    u = np.array(u)

    llf_obs = -0.5 * u**2 / v - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    assert_allclose(u, res.forecasts_error[0])
    # assert_allclose(theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1])
    assert_allclose(llf_obs, res.llf_obs)

    # Get llf_obs directly
    llf_obs2 = _arma_innovations.darma_loglikeobs_fast(
        endog, ar_params, ma_params, sigma2)

    assert_allclose(llf_obs2, res.llf_obs)


@pytest.mark.parametrize("ar_params,ma_params,sigma2", [
    (np.array([]), np.array([]), 1),
    (np.array([0.]), np.array([0.]), 1),
    (np.array([0.9]), np.array([]), 1),
    (np.array([]), np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_innovations_algo_direct_filter_kalman_filter(ar_params, ma_params,
                                                      sigma2):
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process, using the direct
    # function.

    endog = np.random.normal(size=10)

    # Innovations algorithm approach
    u, r = arma_innovations.arma_innovations(endog, ar_params, ma_params,
                                             sigma2)

    v = np.array(r) * sigma2
    u = np.array(u)

    llf_obs = -0.5 * u**2 / v - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    assert_allclose(u, res.forecasts_error[0])
    # assert_allclose(theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1])
    assert_allclose(llf_obs, res.llf_obs)

    # Get llf_obs directly
    llf_obs2 = _arma_innovations.darma_loglikeobs_fast(
        endog, ar_params, ma_params, sigma2)

    assert_allclose(llf_obs2, res.llf_obs)


@pytest.mark.parametrize("ar_params,diff,ma_params,sigma2", [
    (np.array([]), 1, np.array([]), 1),
    (np.array([0.]), 1, np.array([0.]), 1),
    (np.array([0.9]), 1, np.array([]), 1),
    (np.array([]), 1, np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), 1, np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), 1, np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
    (np.array([0.5, 0.1]), 2, np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_integrated_process(ar_params, diff, ma_params, sigma2):
    # Test loglikelihood computation when model has integration

    nobs = 100

    endog = np.cumsum(np.random.normal(size=nobs))

    # Innovations algorithm approach
    llf_obs = arma_innovations.arma_loglikeobs(
        np.diff(endog, diff), ar_params, ma_params, sigma2)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), diff, len(ma_params)),
                  simple_differencing=True)
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    assert_allclose(llf_obs, res.llf_obs)


@pytest.mark.parametrize("ar_params,ma_params,sigma2", [
    (np.array([]), np.array([]), 1),
    (np.array([0.]), np.array([0.]), 1),
    (np.array([0.9]), np.array([]), 1),
    (np.array([]), np.array([0.9]), 1),
    (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123),
    (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123),
])
def test_regression_with_arma_errors(ar_params, ma_params, sigma2):
    # Test loglikelihood computation when model has regressors
    nobs = 100

    eps = np.random.normal(nobs)
    exog = np.c_[np.ones(nobs), np.random.uniform(size=nobs)]
    beta = [5, -0.2]
    endog = np.dot(exog, beta) + eps

    # Innovations algorithm approach
    beta_hat = np.squeeze(np.linalg.pinv(exog).dot(endog))
    demeaned = endog - np.dot(exog, beta_hat)
    llf_obs = arma_innovations.arma_loglikeobs(
        demeaned, ar_params, ma_params, sigma2)

    # Kalman filter approach
    # (this works since we impose here that the regression coefficients are
    # beta_hat - in practice, the MLE estimates will not necessarily match
    # the OLS estimates beta_hat)
    mod = SARIMAX(endog, exog=exog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[beta_hat, ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    assert_allclose(llf_obs, res.llf_obs)
