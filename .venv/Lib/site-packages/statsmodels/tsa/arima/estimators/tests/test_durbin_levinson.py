import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_raises

from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson


@pytest.mark.low_precision('Test against Example 5.1.1 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_511():
    # Note: this example is primarily tested in
    # test_yule_walker::test_brockwell_davis_example_511.

    # Difference the series
    endog = dowj.diff().iloc[1:]

    # Durbin-Levinson
    dl, _ = durbin_levinson(endog, ar_order=2, demean=True)

    assert_allclose(dl[0].params, np.var(endog))
    assert_allclose(dl[1].params, [0.4219, 0.1479], atol=1e-4)
    assert_allclose(dl[2].params, [0.3739, 0.1138, 0.1460], atol=1e-4)


def check_itsmr(lake):
    # Test against R itsmr::yw; see results/results_yw_dl.R
    dl, _ = durbin_levinson(lake, 5)

    assert_allclose(dl[0].params, np.var(lake))
    assert_allclose(dl[1].ar_params, [0.8319112104])
    assert_allclose(dl[2].ar_params, [1.0538248798, -0.2667516276])
    desired = [1.0887037577, -0.4045435867, 0.1307541335]
    assert_allclose(dl[3].ar_params, desired)
    desired = [1.08425065810, -0.39076602696, 0.09367609911, 0.03405704644]
    assert_allclose(dl[4].ar_params, desired)
    desired = [1.08213598501, -0.39658257147, 0.11793957728, -0.03326633983,
               0.06209208707]
    assert_allclose(dl[5].ar_params, desired)

    # itsmr::yw returns the innovations algorithm estimate of the variance
    # we'll just check for p=5
    u, v = arma_innovations(np.array(lake) - np.mean(lake),
                            ar_params=dl[5].ar_params, sigma2=1)
    desired_sigma2 = 0.4716322564
    assert_allclose(np.sum(u**2 / v) / len(u), desired_sigma2)


def test_itsmr():
    # Note: apparently itsmr automatically demeans (there is no option to
    # control this)
    endog = lake.copy()

    check_itsmr(endog)           # Pandas series
    check_itsmr(endog.values)    # Numpy array
    check_itsmr(endog.tolist())  # Python list


def test_nonstationary_series():
    # Test against R stats::ar.yw; see results/results_yw_dl.R
    endog = np.arange(1, 12) * 1.0
    res, _ = durbin_levinson(endog, 2, demean=False)

    desired_ar_params = [0.92318534179, -0.06166314306]
    assert_allclose(res[2].ar_params, desired_ar_params)


@pytest.mark.xfail(reason='Different computation of variances')
def test_nonstationary_series_variance():
    # See `test_nonstationary_series`. This part of the test has been broken
    # out as an xfail because we compute a different estimate of the variance
    # from stats::ar.yw, but keeping the test in case we want to also implement
    # that variance estimate in the future.
    endog = np.arange(1, 12) * 1.0
    res, _ = durbin_levinson(endog, 2, demean=False)

    desired_sigma2 = 15.36526603
    assert_allclose(res[2].sigma2, desired_sigma2)


def test_invalid():
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, durbin_levinson, endog, ar_order=2)
    assert_raises(ValueError, durbin_levinson, endog, ar_order=-1)
    assert_raises(ValueError, durbin_levinson, endog, ar_order=1.5)

    endog = np.arange(10) * 1.0
    assert_raises(ValueError, durbin_levinson, endog, ar_order=[1, 3])


def test_misc():
    # Test defaults (order = 0, demean=True)
    endog = lake.copy()
    res, _ = durbin_levinson(endog)
    assert_allclose(res[0].params, np.var(endog))

    # Test that integer input gives the same result as float-coerced input.
    endog = np.array([1, 2, 5, 3, -2, 1, -3, 5, 2, 3, -1], dtype=int)
    res_int, _ = durbin_levinson(endog, 2, demean=False)
    res_float, _ = durbin_levinson(endog * 1.0, 2, demean=False)
    assert_allclose(res_int[0].params, res_float[0].params)
    assert_allclose(res_int[1].params, res_float[1].params)
    assert_allclose(res_int[2].params, res_float[2].params)
