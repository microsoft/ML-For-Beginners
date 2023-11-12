import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.burg import burg


@pytest.mark.low_precision('Test against Example 5.1.3 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_513():
    # Test against Example 5.1.3 in Brockwell and Davis (2016)
    # (low-precision test, since we are testing against values printed in the
    # textbook)

    # Difference and demean the series
    endog = dowj.diff().iloc[1:]

    # Burg
    res, _ = burg(endog, ar_order=1, demean=True)
    assert_allclose(res.ar_params, [0.4371], atol=1e-4)
    assert_allclose(res.sigma2, 0.1423, atol=1e-4)


@pytest.mark.low_precision('Test against Example 5.1.4 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_514():
    # Test against Example 5.1.4 in Brockwell and Davis (2016)
    # (low-precision test, since we are testing against values printed in the
    # textbook)

    # Get the lake data
    endog = lake.copy()

    # Should have 98 observations
    assert_equal(len(endog), 98)
    desired = 9.0041
    assert_allclose(endog.mean(), desired, atol=1e-4)

    # Burg
    res, _ = burg(endog, ar_order=2, demean=True)
    assert_allclose(res.ar_params, [1.0449, -0.2456], atol=1e-4)
    assert_allclose(res.sigma2, 0.4706, atol=1e-4)


def check_itsmr(lake):
    # Test against R itsmr::burg; see results/results_burg.R
    res, _ = burg(lake, 10, demean=True)
    desired_ar_params = [
        1.05853631096, -0.32639150878, 0.04784765122, 0.02620476111,
        0.04444511374, -0.04134010262, 0.02251178970, -0.01427524694,
        0.22223486915, -0.20935524387]
    assert_allclose(res.ar_params, desired_ar_params)

    # itsmr always returns the innovations algorithm estimate of sigma2,
    # whereas we return Burg's estimate
    u, v = arma_innovations(np.array(lake) - np.mean(lake),
                            ar_params=res.ar_params, sigma2=1)
    desired_sigma2 = 0.4458956354
    assert_allclose(np.sum(u**2 / v) / len(u), desired_sigma2)


def test_itsmr():
    # Note: apparently itsmr automatically demeans (there is no option to
    # control this)
    endog = lake.copy()

    check_itsmr(endog)           # Pandas series
    check_itsmr(endog.values)    # Numpy array
    check_itsmr(endog.tolist())  # Python list


def test_nonstationary_series():
    # Test against R stats::ar.burg; see results/results_burg.R
    endog = np.arange(1, 12) * 1.0
    res, _ = burg(endog, 2, demean=False)

    desired_ar_params = [1.9669331547, -0.9892846679]
    assert_allclose(res.ar_params, desired_ar_params)
    desired_sigma2 = 0.02143066427
    assert_allclose(res.sigma2, desired_sigma2)

    # With var.method = 1, stats::ar.burg also returns something equivalent to
    # the innovations algorithm estimate of sigma2
    u, v = arma_innovations(endog, ar_params=res.ar_params, sigma2=1)
    desired_sigma2 = 0.02191056906
    assert_allclose(np.sum(u**2 / v) / len(u), desired_sigma2)


def test_invalid():
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, burg, endog, ar_order=2)
    assert_raises(ValueError, burg, endog, ar_order=-1)
    assert_raises(ValueError, burg, endog, ar_order=1.5)

    endog = np.arange(10) * 1.0
    assert_raises(ValueError, burg, endog, ar_order=[1, 3])


def test_misc():
    # Test defaults (order = 0, demean=True)
    endog = lake.copy()
    res, _ = burg(endog)
    assert_allclose(res.params, np.var(endog))

    # Test that integer input gives the same result as float-coerced input.
    endog = np.array([1, 2, 5, 3, -2, 1, -3, 5, 2, 3, -1], dtype=int)
    res_int, _ = burg(endog, 2)
    res_float, _ = burg(endog * 1.0, 2)
    assert_allclose(res_int.params, res_float.params)
