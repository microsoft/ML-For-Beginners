import numpy as np

import pytest
from numpy.testing import (
    assert_, assert_allclose, assert_warns, assert_raises)

from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import (
    dowj, lake, oshorts)
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations, innovations_mle)


@pytest.mark.low_precision('Test against Example 5.1.5 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_515():
    # Difference and demean the series
    endog = dowj.diff().iloc[1:]

    # Innvations algorithm (MA)
    p, _ = innovations(endog, ma_order=17, demean=True)

    # First BD show the MA(2) coefficients resulting from the m=17 computations
    assert_allclose(p[17].ma_params[:2], [.4269, .2704], atol=1e-4)
    assert_allclose(p[17].sigma2, 0.1122, atol=1e-4)

    # Then they separately show the full MA(17) coefficients
    desired = [.4269, .2704, .1183, .1589, .1355, .1568, .1284, -.0060, .0148,
               -.0017, .1974, -.0463, .2023, .1285, -.0213, -.2575, .0760]
    assert_allclose(p[17].ma_params, desired, atol=1e-4)


def check_innovations_ma_itsmr(lake):
    # Test against R itsmr::ia; see results/results_innovations.R
    ia, _ = innovations(lake, 10, demean=True)

    desired = [
        1.0816255264, 0.7781248438, 0.5367164430, 0.3291559246, 0.3160039850,
        0.2513754550, 0.2051536531, 0.1441070313, 0.3431868340, 0.1827400798]
    assert_allclose(ia[10].ma_params, desired)

    # itsmr::ia returns the innovations algorithm estimate of the variance
    u, v = arma_innovations(np.array(lake) - np.mean(lake),
                            ma_params=ia[10].ma_params, sigma2=1)
    desired_sigma2 = 0.4523684344
    assert_allclose(np.sum(u**2 / v) / len(u), desired_sigma2)


def test_innovations_ma_itsmr():
    # Note: apparently itsmr automatically demeans (there is no option to
    # control this)
    endog = lake.copy()

    check_innovations_ma_itsmr(endog)           # Pandas series
    check_innovations_ma_itsmr(endog.values)    # Numpy array
    check_innovations_ma_itsmr(endog.tolist())  # Python list


def test_innovations_ma_invalid():
    endog = np.arange(2)
    assert_raises(ValueError, innovations, endog, ma_order=2)
    assert_raises(ValueError, innovations, endog, ma_order=-1)
    assert_raises(ValueError, innovations, endog, ma_order=1.5)

    endog = np.arange(10)
    assert_raises(ValueError, innovations, endog, ma_order=[1, 3])


@pytest.mark.low_precision('Test against Example 5.2.4 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_524():
    # Difference and demean the series
    endog = dowj.diff().iloc[1:]

    # Use Burg method to get initial coefficients for MLE
    initial, _ = burg(endog, ar_order=1, demean=True)

    # Fit MLE via innovations algorithm
    p, _ = innovations_mle(endog, order=(1, 0, 0), demean=True,
                           start_params=initial.params)

    assert_allclose(p.ar_params, 0.4471, atol=1e-4)


@pytest.mark.low_precision('Test against Example 5.2.4 in Brockwell and Davis'
                           ' (2016)')
@pytest.mark.xfail(reason='Suspicious result reported in Brockwell and Davis'
                          ' (2016).')
def test_brockwell_davis_example_524_variance():
    # See `test_brockwell_davis_example_524` for the main test
    # TODO: the test for sigma2 fails, but the value reported by BD (0.02117)
    # is suspicious. For example, the Burg results have an AR coefficient of
    # 0.4371 and sigma2 = 0.1423. It seems unlikely that the small difference
    # in AR coefficient would result in an order of magniture reduction in
    # sigma2 (see test_burg::test_brockwell_davis_example_513). Should run
    # this in the ITSM program to check its output.
    endog = dowj.diff().iloc[1:]

    # Use Burg method to get initial coefficients for MLE
    initial, _ = burg(endog, ar_order=1, demean=True)

    # Fit MLE via innovations algorithm
    p, _ = innovations_mle(endog, order=(1, 0, 0), demean=True,
                           start_params=initial.params)

    assert_allclose(p.sigma2, 0.02117, atol=1e-4)


@pytest.mark.low_precision('Test against Example 5.2.5 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_525():
    # Difference and demean the series
    endog = lake.copy()

    # Use HR method to get initial coefficients for MLE
    initial, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True)

    # Fit MLE via innovations algorithm
    p, _ = innovations_mle(endog, order=(1, 0, 1), demean=True,
                           start_params=initial.params)

    assert_allclose(p.params, [0.7446, 0.3213, 0.4750], atol=1e-4)

    # Fit MLE via innovations algorithm, with default starting parameters
    p, _ = innovations_mle(endog, order=(1, 0, 1), demean=True)

    assert_allclose(p.params, [0.7446, 0.3213, 0.4750], atol=1e-4)


@pytest.mark.low_precision('Test against Example 5.4.1 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_541():
    # Difference and demean the series
    endog = oshorts.copy()

    # Use innovations MA method to get initial coefficients for MLE
    initial, _ = innovations(endog, ma_order=1, demean=True)

    # Fit MLE via innovations algorithm
    p, _ = innovations_mle(endog, order=(0, 0, 1), demean=True,
                           start_params=initial[1].params)

    assert_allclose(p.ma_params, -0.818, atol=1e-3)

    # TODO: the test for sigma2 fails; we get 2040.85 whereas BD reports
    # 2040.75. Unclear if this is optimizers finding different maxima, or a
    # reporting error by BD (i.e. typo where the 8 got reported as a 7). Should
    # check this out with ITSM program. NB: state space also finds 2040.85 as
    # the MLE value.
    # assert_allclose(p.sigma2, 2040.75, atol=1e-2)


def test_innovations_mle_statespace():
    # Test innovations output against state-space output.
    endog = lake.copy()
    endog = endog - endog.mean()

    start_params = [0, 0, np.var(endog)]
    p, mleres = innovations_mle(endog, order=(1, 0, 1), demean=False,
                                start_params=start_params)

    mod = sarimax.SARIMAX(endog, order=(1, 0, 1))

    # Test that the maximized log-likelihood found via applications of the
    # innovations algorithm matches the log-likelihood found by the Kalman
    # filter at the same parameters
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)

    # Test MLE fitting
    # To avoid small numerical differences with MLE fitting, start at the
    # parameters found from innovations_mle
    res2 = mod.fit(start_params=p.params, disp=0)

    # Test that the state space approach confirms the MLE values found by
    # innovations_mle
    assert_allclose(p.params, res2.params)

    # Test that starting parameter estimation succeeds and isn't terrible
    # (i.e. leads to the same MLE)
    p2, _ = innovations_mle(endog, order=(1, 0, 1), demean=False)
    # (does not need to be high-precision test since it's okay if different
    # starting parameters give slightly different MLE)
    assert_allclose(p.params, p2.params, atol=1e-5)


def test_innovations_mle_statespace_seasonal():
    # Test innovations output against state-space output.
    endog = lake.copy()
    endog = endog - endog.mean()

    start_params = [0, np.var(endog)]
    p, mleres = innovations_mle(endog, seasonal_order=(1, 0, 0, 4),
                                demean=False, start_params=start_params)

    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4))

    # Test that the maximized log-likelihood found via applications of the
    # innovations algorithm matches the log-likelihood found by the Kalman
    # filter at the same parameters
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)

    # Test MLE fitting
    # To avoid small numerical differences with MLE fitting, start at the
    # parameters found from innovations_mle
    res2 = mod.fit(start_params=p.params, disp=0)

    # Test that the state space approach confirms the MLE values found by
    # innovations_mle
    assert_allclose(p.params, res2.params)

    # Test that starting parameter estimation succeeds and isn't terrible
    # (i.e. leads to the same MLE)
    p2, _ = innovations_mle(endog, seasonal_order=(1, 0, 0, 4), demean=False)
    # (does not need to be high-precision test since it's okay if different
    # starting parameters give slightly different MLE)
    assert_allclose(p.params, p2.params, atol=1e-5)


def test_innovations_mle_statespace_nonconsecutive():
    # Test innovations output against state-space output.
    endog = lake.copy()
    endog = endog - endog.mean()

    start_params = [0, 0, np.var(endog)]
    p, mleres = innovations_mle(endog, order=([0, 1], 0, [0, 1]),
                                demean=False, start_params=start_params)

    mod = sarimax.SARIMAX(endog, order=([0, 1], 0, [0, 1]))

    # Test that the maximized log-likelihood found via applications of the
    # innovations algorithm matches the log-likelihood found by the Kalman
    # filter at the same parameters
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)

    # Test MLE fitting
    # To avoid small numerical differences with MLE fitting, start at the
    # parameters found from innovations_mle
    res2 = mod.fit(start_params=p.params, disp=0)

    # Test that the state space approach confirms the MLE values found by
    # innovations_mle
    assert_allclose(p.params, res2.params)

    # Test that starting parameter estimation succeeds and isn't terrible
    # (i.e. leads to the same MLE)
    p2, _ = innovations_mle(endog, order=([0, 1], 0, [0, 1]), demean=False)
    # (does not need to be high-precision test since it's okay if different
    # starting parameters give slightly different MLE)
    assert_allclose(p.params, p2.params, atol=1e-5)


def test_innovations_mle_integrated():
    endog = np.r_[0, np.cumsum(lake.copy())]

    start_params = [0, np.var(lake.copy())]
    with assert_warns(UserWarning):
        p, mleres = innovations_mle(endog, order=(1, 1, 0),
                                    demean=False, start_params=start_params)

    mod = sarimax.SARIMAX(endog, order=(1, 1, 0),
                          simple_differencing=True)

    # Test that the maximized log-likelihood found via applications of the
    # innovations algorithm matches the log-likelihood found by the Kalman
    # filter at the same parameters
    res = mod.filter(p.params)
    assert_allclose(-mleres.minimize_results.fun, res.llf)

    # Test MLE fitting
    # To avoid small numerical differences with MLE fitting, start at the
    # parameters found from innovations_mle
    res2 = mod.fit(start_params=p.params, disp=0)

    # Test that the state space approach confirms the MLE values found by
    # innovations_mle
    # Note: atol is required only due to precision issues on Windows
    assert_allclose(p.params, res2.params, atol=1e-6)

    # Test that the result is equivalent to order=(1, 0, 0) on the differenced
    # data
    p2, _ = innovations_mle(lake.copy(), order=(1, 0, 0), demean=False,
                            start_params=start_params)
    # (does not need to be high-precision test since it's okay if different
    # starting parameters give slightly different MLE)
    assert_allclose(p.params, p2.params, atol=1e-5)


def test_innovations_mle_misc():
    endog = np.arange(20)**2 * 1.0

    # Check that when Hannan-Rissanen estimates non-stationary starting
    # parameters, innovations_mle sets it to zero
    hr, _ = hannan_rissanen(endog, ar_order=1, demean=False)
    assert_(hr.ar_params[0] > 1)
    _, res = innovations_mle(endog, order=(1, 0, 0))
    assert_allclose(res.start_params[0], 0)

    # Check that when Hannan-Rissanen estimates non-invertible starting
    # parameters, innovations_mle sets it to zero
    hr, _ = hannan_rissanen(endog, ma_order=1, demean=False)
    assert_(hr.ma_params[0] > 1)
    _, res = innovations_mle(endog, order=(0, 0, 1))
    assert_allclose(res.start_params[0], 0)


def test_innovations_mle_invalid():
    endog = np.arange(2) * 1.0
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 2))
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, -1))
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 1.5))

    endog = lake.copy()
    assert_raises(ValueError, innovations_mle, endog, order=(1, 0, 0),
                  start_params=[1., 1.])
    assert_raises(ValueError, innovations_mle, endog, order=(0, 0, 1),
                  start_params=[1., 1.])
