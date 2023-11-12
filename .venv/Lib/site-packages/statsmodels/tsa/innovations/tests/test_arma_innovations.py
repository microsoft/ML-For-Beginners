"""
Tests for ARMA innovations algorithm wrapper
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    np.random.seed(42)
    endog = np.random.normal(size=100)

    # Innovations algorithm approach
    llf = arma_innovations.arma_loglike(endog, ar_params, ma_params, sigma2)
    llf_obs = arma_innovations.arma_loglikeobs(endog, ar_params, ma_params,
                                               sigma2)
    score = arma_innovations.arma_score(endog, ar_params, ma_params, sigma2)
    score_obs = arma_innovations.arma_scoreobs(endog, ar_params, ma_params,
                                               sigma2)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    params = np.r_[ar_params, ma_params, sigma2]

    # Test that the two approaches are the same
    assert_allclose(llf, mod.loglike(params))
    assert_allclose(llf_obs, mod.loglikeobs(params))
    # Note: the tolerance on the two gets worse as more nobs are added
    assert_allclose(score, mod.score(params), atol=1e-5)
    assert_allclose(score_obs, mod.score_obs(params), atol=1e-5)


@pytest.mark.parametrize("ar_params", ([1.9, -0.8], [1.0], [2.0, -1.0]))
def test_innovations_nonstationary(ar_params):
    np.random.seed(42)
    endog = np.random.normal(size=100)
    with pytest.raises(ValueError, match="The model's autoregressive"):
        arma_innovations.arma_innovations(endog, ar_params=ar_params)
