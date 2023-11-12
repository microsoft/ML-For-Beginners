import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker


@pytest.mark.low_precision('Test against Example 5.1.1 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_511():
    # Make the series stationary
    endog = dowj.diff().iloc[1:]

    # Should have 77 observations
    assert_equal(len(endog), 77)

    # Autocovariances
    desired = [0.17992, 0.07590, 0.04885]
    assert_allclose(acovf(endog, fft=True, nlag=2), desired, atol=1e-5)

    # Yule-Walker
    yw, _ = yule_walker(endog, ar_order=1, demean=True)
    assert_allclose(yw.ar_params, [0.4219], atol=1e-4)
    assert_allclose(yw.sigma2, 0.1479, atol=1e-4)


@pytest.mark.low_precision('Test against Example 5.1.4 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_514():
    # Note: this example is primarily tested in
    # test_burg::test_brockwell_davis_example_514.

    # Get the lake data, demean
    endog = lake.copy()

    # Yule-Walker
    res, _ = yule_walker(endog, ar_order=2, demean=True)
    assert_allclose(res.ar_params, [1.0538, -0.2668], atol=1e-4)
    assert_allclose(res.sigma2, 0.4920, atol=1e-4)


def check_itsmr(lake):
    # Test against R itsmr::yw; see results/results_yw_dl.R
    yw, _ = yule_walker(lake, 5)

    desired = [1.08213598501, -0.39658257147, 0.11793957728, -0.03326633983,
               0.06209208707]
    assert_allclose(yw.ar_params, desired)

    # stats::ar.yw return the innovations algorithm estimate of the variance
    u, v = arma_innovations(np.array(lake) - np.mean(lake),
                            ar_params=yw.ar_params, sigma2=1)
    desired_sigma2 = 0.4716322564
    assert_allclose(np.sum(u**2 / v) / len(u), desired_sigma2)


def test_itsmr():
    # Note: apparently itsmr automatically demeans (there is no option to
    # control this)
    endog = lake.copy()

    check_itsmr(endog)           # Pandas series
    check_itsmr(endog.values)    # Numpy array
    check_itsmr(endog.tolist())  # Python list


def test_invalid():
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=-1)
    assert_raises(ValueError, yule_walker, endog, ar_order=1.5)

    endog = np.arange(10) * 1.0
    assert_raises(ValueError, yule_walker, endog, ar_order=[1, 3])


@pytest.mark.xfail(reason='TODO: this does not raise an error due to the way'
                          ' linear_model.yule_walker works.')
def test_invalid_xfail():
    endog = np.arange(2) * 1.0

    # TODO: this does not raise an error due to the way Statsmodels'
    # yule_walker function works
    assert_raises(ValueError, yule_walker, endog, ar_order=2)
