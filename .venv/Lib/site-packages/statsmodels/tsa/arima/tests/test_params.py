import numpy as np
import pandas as pd

from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises

from statsmodels.tsa.arima import specification, params


def test_init():
    # Test initialization of the params

    # Basic test, with 1 of each parameter
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test things copied over from spec
    assert_equal(p.spec, spec)
    assert_equal(p.exog_names, ['a'])
    assert_equal(p.ar_names, ['ar.L1'])
    assert_equal(p.ma_names, ['ma.L1'])
    assert_equal(p.seasonal_ar_names, ['ar.S.L4'])
    assert_equal(p.seasonal_ma_names, ['ma.S.L4'])
    assert_equal(p.param_names, ['a', 'ar.L1', 'ma.L1', 'ar.S.L4', 'ma.S.L4',
                                 'sigma2'])

    assert_equal(p.k_exog_params, 1)
    assert_equal(p.k_ar_params, 1)
    assert_equal(p.k_ma_params, 1)
    assert_equal(p.k_seasonal_ar_params, 1)
    assert_equal(p.k_seasonal_ma_params, 1)
    assert_equal(p.k_params, 6)

    # Initial parameters should all be NaN
    assert_equal(p.params, np.nan)
    assert_equal(p.ar_params, [np.nan])
    assert_equal(p.ma_params, [np.nan])
    assert_equal(p.seasonal_ar_params, [np.nan])
    assert_equal(p.seasonal_ma_params, [np.nan])
    assert_equal(p.sigma2, np.nan)
    assert_equal(p.ar_poly.coef, np.r_[1, np.nan])
    assert_equal(p.ma_poly.coef, np.r_[1, np.nan])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, np.nan])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, np.nan])
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, [np.nan] * 5])
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, [np.nan] * 5])

    # Test other properties, methods
    assert_(not p.is_complete)
    assert_(not p.is_valid)
    assert_raises(ValueError, p.__getattribute__, 'is_stationary')
    assert_raises(ValueError, p.__getattribute__, 'is_invertible')
    desired = {
        'exog_params': [np.nan],
        'ar_params': [np.nan],
        'ma_params': [np.nan],
        'seasonal_ar_params': [np.nan],
        'seasonal_ma_params': [np.nan],
        'sigma2': np.nan}
    assert_equal(p.to_dict(), desired)
    desired = pd.Series([np.nan] * spec.k_params, index=spec.param_names)
    assert_allclose(p.to_pandas(), desired)

    # Test with different numbers of parameters for each
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(3, 1, 2), seasonal_order=(5, 1, 6, 4))
    p = params.SARIMAXParams(spec=spec)
    # No real need to test names here, since they are already tested above for
    # the 1-param case, and tested more extensively in test for
    # SARIMAXSpecification
    assert_equal(p.k_exog_params, 2)
    assert_equal(p.k_ar_params, 3)
    assert_equal(p.k_ma_params, 2)
    assert_equal(p.k_seasonal_ar_params, 5)
    assert_equal(p.k_seasonal_ma_params, 6)
    assert_equal(p.k_params, 2 + 3 + 2 + 5 + 6 + 1)


def test_set_params_single():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has only a single parameters
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    def check(is_stationary='raise', is_invertible='raise'):
        assert_(not p.is_complete)
        assert_(not p.is_valid)
        if is_stationary == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_stationary')
        else:
            assert_equal(p.is_stationary, is_stationary)
        if is_invertible == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_invertible')
        else:
            assert_equal(p.is_invertible, is_invertible)

    # Set params one at a time, as scalars
    p.exog_params = -6.
    check()
    p.ar_params = -5.
    check()
    p.ma_params = -4.
    check()
    p.seasonal_ar_params = -3.
    check(is_stationary=False)
    p.seasonal_ma_params = -2.
    check(is_stationary=False, is_invertible=False)
    p.sigma2 = -1.
    # Finally, we have a complete set.
    assert_(p.is_complete)
    # But still not valid
    assert_(not p.is_valid)

    assert_equal(p.params, [-6, -5, -4, -3, -2, -1])
    assert_equal(p.exog_params, [-6])
    assert_equal(p.ar_params, [-5])
    assert_equal(p.ma_params, [-4])
    assert_equal(p.seasonal_ar_params, [-3])
    assert_equal(p.seasonal_ma_params, [-2])
    assert_equal(p.sigma2, -1.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 5])
    assert_equal(p.ma_poly.coef, np.r_[1, -4])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 3])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -2])
    # (1 - a L) (1 - b L^4) = (1 - a L - b L^4 + a b L^5)
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, 5, 0, 0, 3, 15])
    # (1 + a L) (1 + b L^4) = (1 + a L + b L^4 + a b L^5)
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, -4, 0, 0, -2, 8])

    # Override again, one at a time, now using lists
    p.exog_params = [1.]
    p.ar_params = [2.]
    p.ma_params = [3.]
    p.seasonal_ar_params = [4.]
    p.seasonal_ma_params = [5.]
    p.sigma2 = [6.]

    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Override again, one at a time, now using arrays
    p.exog_params = np.array(6.)
    p.ar_params = np.array(5.)
    p.ma_params = np.array(4.)
    p.seasonal_ar_params = np.array(3.)
    p.seasonal_ma_params = np.array(2.)
    p.sigma2 = np.array(1.)

    assert_equal(p.params, [6, 5, 4, 3, 2, 1])
    assert_equal(p.exog_params, [6])
    assert_equal(p.ar_params, [5])
    assert_equal(p.ma_params, [4])
    assert_equal(p.seasonal_ar_params, [3])
    assert_equal(p.seasonal_ma_params, [2])
    assert_equal(p.sigma2, 1.)

    # Override again, now setting params all at once
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, -2])
    assert_equal(p.ma_poly.coef, np.r_[1, 3])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, -4])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, 5])
    # (1 - a L) (1 - b L^4) = (1 - a L - b L^4 + a b L^5)
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, -2, 0, 0, -4, 8])
    # (1 + a L) (1 + b L^4) = (1 + a L + b L^4 + a b L^5)
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, 3, 0, 0, 5, 15])


def test_set_params_single_nonconsecutive():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has only a single parameters but has non-consecutive
    # lag orders
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=([0, 1], 1, [0, 1]),
        seasonal_order=([0, 1], 1, [0, 1], 4))
    p = params.SARIMAXParams(spec=spec)

    def check(is_stationary='raise', is_invertible='raise'):
        assert_(not p.is_complete)
        assert_(not p.is_valid)
        if is_stationary == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_stationary')
        else:
            assert_equal(p.is_stationary, is_stationary)
        if is_invertible == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_invertible')
        else:
            assert_equal(p.is_invertible, is_invertible)

    # Set params one at a time, as scalars
    p.exog_params = -6.
    check()
    p.ar_params = -5.
    check()
    p.ma_params = -4.
    check()
    p.seasonal_ar_params = -3.
    check(is_stationary=False)
    p.seasonal_ma_params = -2.
    check(is_stationary=False, is_invertible=False)
    p.sigma2 = -1.
    # Finally, we have a complete set.
    assert_(p.is_complete)
    # But still not valid
    assert_(not p.is_valid)

    assert_equal(p.params, [-6, -5, -4, -3, -2, -1])
    assert_equal(p.exog_params, [-6])
    assert_equal(p.ar_params, [-5])
    assert_equal(p.ma_params, [-4])
    assert_equal(p.seasonal_ar_params, [-3])
    assert_equal(p.seasonal_ma_params, [-2])
    assert_equal(p.sigma2, -1.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, [1, 0, 5])
    assert_equal(p.ma_poly.coef, [1, 0, -4])
    assert_equal(p.seasonal_ar_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, 3])
    assert_equal(p.seasonal_ma_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, -2])
    # (1 - a L^2) (1 - b L^8) = (1 - a L^2 - b L^8 + a b L^10)
    assert_equal(p.reduced_ar_poly.coef, [1, 0, 5, 0, 0, 0, 0, 0, 3, 0, 15])
    # (1 + a L^2) (1 + b L^4) = (1 + a L^2 + b L^8 + a b L^10)
    assert_equal(p.reduced_ma_poly.coef, [1, 0, -4, 0, 0, 0, 0, 0, -2, 0, 8])

    # Override again, now setting params all at once
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 0, -2])
    assert_equal(p.ma_poly.coef, np.r_[1, 0, 3])
    assert_equal(p.seasonal_ar_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, -4])
    assert_equal(p.seasonal_ma_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, 5])
    # (1 - a L^2) (1 - b L^8) = (1 - a L^2 - b L^8 + a b L^10)
    assert_equal(p.reduced_ar_poly.coef, [1, 0, -2, 0, 0, 0, 0, 0, -4, 0, 8])
    # (1 + a L^2) (1 + b L^4) = (1 + a L^2 + b L^8 + a b L^10)
    assert_equal(p.reduced_ma_poly.coef, [1, 0, 3, 0, 0, 0, 0, 0, 5, 0, 15])


def test_set_params_multiple():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has multiple a single parameters
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)

    p.params = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11]
    assert_equal(p.params,
                 [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 3, -4])
    assert_equal(p.ma_poly.coef, np.r_[1, -5, 6])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10])
    # (1 - a_1 L - a_2 L^2) (1 - b_1 L^4 - b_2 L^8) =
    #     (1 - b_1 L^4 - b_2 L^8) +
    #     (-a_1 L + a_1 b_1 L^5 + a_1 b_2 L^9) +
    #     (-a_2 L^2 + a_2 b_1 L^6 + a_2 b_2 L^10) =
    #     1 - a_1 L - a_2 L^2 - b_1 L^4 + a_1 b_1 L^5 +
    #     a_2 b_1 L^6 - b_2 L^8 + a_1 b_2 L^9 + a_2 b_2 L^10
    assert_equal(p.reduced_ar_poly.coef,
                 [1, 3, -4, 0, 7, (-3 * -7), (4 * -7), 0, -8, (-3 * 8), 4 * 8])
    # (1 + a_1 L + a_2 L^2) (1 + b_1 L^4 + b_2 L^8) =
    #     (1 + b_1 L^4 + b_2 L^8) +
    #     (a_1 L + a_1 b_1 L^5 + a_1 b_2 L^9) +
    #     (a_2 L^2 + a_2 b_1 L^6 + a_2 b_2 L^10) =
    #     1 + a_1 L + a_2 L^2 + b_1 L^4 + a_1 b_1 L^5 +
    #     a_2 b_1 L^6 + b_2 L^8 + a_1 b_2 L^9 + a_2 b_2 L^10
    assert_equal(p.reduced_ma_poly.coef,
                 [1, -5, 6, 0, -9, (-5 * -9), (6 * -9),
                  0, 10, (-5 * 10), (6 * 10)])


def test_set_poly_short_lags():
    # Basic example (short lag orders)
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Valid polynomials
    p.ar_poly = [1, -0.5]
    assert_equal(p.ar_params, [0.5])
    p.ar_poly = np.polynomial.Polynomial([1, -0.55])
    assert_equal(p.ar_params, [0.55])
    p.ma_poly = [1, 0.3]
    assert_equal(p.ma_params, [0.3])
    p.ma_poly = np.polynomial.Polynomial([1, 0.35])
    assert_equal(p.ma_params, [0.35])

    p.seasonal_ar_poly = [1, 0, 0, 0, -0.2]
    assert_equal(p.seasonal_ar_params, [0.2])
    p.seasonal_ar_poly = np.polynomial.Polynomial([1, 0, 0, 0, -0.25])
    assert_equal(p.seasonal_ar_params, [0.25])
    p.seasonal_ma_poly = [1, 0, 0, 0, 0.1]
    assert_equal(p.seasonal_ma_params, [0.1])
    p.seasonal_ma_poly = np.polynomial.Polynomial([1, 0, 0, 0, 0.15])
    assert_equal(p.seasonal_ma_params, [0.15])

    # Invalid polynomials
    # Must have 1 in the initial position
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [2, -0.5])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [2, 0.3])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [2, 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [2, 0, 0, 0, 0.1])
    # Too short
    assert_raises(ValueError, p.__setattr__, 'ar_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', [1, 0, 0, 0])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', [1, 0, 0, 0])
    # Too long
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1, -0.5, 0.2])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1, 0.3, 0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 0, 0, 0, 0.1, 0])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 0, 0, 0, 0.1, 0])
    # Number in invalid location (only for seasonal polynomials)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 1, 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 1, 0, 0, 0, 0.1])


def test_set_poly_short_lags_nonconsecutive():
    # Short but non-consecutive lag orders
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=([0, 1], 1, [0, 1]),
        seasonal_order=([0, 1], 1, [0, 1], 4))
    p = params.SARIMAXParams(spec=spec)

    # Valid polynomials
    p.ar_poly = [1, 0, -0.5]
    assert_equal(p.ar_params, [0.5])
    p.ar_poly = np.polynomial.Polynomial([1, 0, -0.55])
    assert_equal(p.ar_params, [0.55])
    p.ma_poly = [1, 0, 0.3]
    assert_equal(p.ma_params, [0.3])
    p.ma_poly = np.polynomial.Polynomial([1, 0, 0.35])
    assert_equal(p.ma_params, [0.35])

    p.seasonal_ar_poly = [1, 0, 0, 0, 0, 0, 0, 0, -0.2]
    assert_equal(p.seasonal_ar_params, [0.2])
    p.seasonal_ar_poly = (
        np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, -0.25]))
    assert_equal(p.seasonal_ar_params, [0.25])
    p.seasonal_ma_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0.1]
    assert_equal(p.seasonal_ma_params, [0.1])
    p.seasonal_ma_poly = (
        np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, 0.15]))
    assert_equal(p.seasonal_ma_params, [0.15])

    # Invalid polynomials
    # Number in invalid (i.e. an excluded lag) location
    # (now also for non-seasonal polynomials)
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1, 1, -0.5])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1, 1, 0.3])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 0, 0, 0, 1., 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 0, 0, 0, 1., 0, 0, 0, 0.1])


def test_set_poly_longer_lags():
    # Test with higher order polynomials
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)

    # Setup the non-AR/MA values
    p.exog_params = [-1, 2]
    p.sigma2 = -11

    # Lag polynomials
    p.ar_poly = np.r_[1, 3, -4]
    p.ma_poly = np.r_[1, -5, 6]
    p.seasonal_ar_poly = np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8]
    p.seasonal_ma_poly = np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10]

    # Test that parameters were set correctly
    assert_equal(p.params,
                 [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)


def test_is_stationary():
    # Tests for the `is_stationary` property
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test stationarity
    assert_raises(ValueError, p.__getattribute__, 'is_stationary')
    p.ar_params = [0.5]
    p.seasonal_ar_params = [0]
    assert_(p.is_stationary)
    p.ar_params = [1.0]
    assert_(not p.is_stationary)

    p.ar_params = [0]
    p.seasonal_ar_params = [0.5]
    assert_(p.is_stationary)
    p.seasonal_ar_params = [1.0]
    assert_(not p.is_stationary)

    p.ar_params = [0.2]
    p.seasonal_ar_params = [0.2]
    assert_(p.is_stationary)
    p.ar_params = [0.99]
    p.seasonal_ar_params = [0.99]
    assert_(p.is_stationary)
    p.ar_params = [1.]
    p.seasonal_ar_params = [1.]
    assert_(not p.is_stationary)


def test_is_invertible():
    # Tests for the `is_invertible` property
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test invertibility
    assert_raises(ValueError, p.__getattribute__, 'is_invertible')
    p.ma_params = [0.5]
    p.seasonal_ma_params = [0]
    assert_(p.is_invertible)
    p.ma_params = [1.0]
    assert_(not p.is_invertible)

    p.ma_params = [0]
    p.seasonal_ma_params = [0.5]
    assert_(p.is_invertible)
    p.seasonal_ma_params = [1.0]
    assert_(not p.is_invertible)

    p.ma_params = [0.2]
    p.seasonal_ma_params = [0.2]
    assert_(p.is_invertible)
    p.ma_params = [0.99]
    p.seasonal_ma_params = [0.99]
    assert_(p.is_invertible)
    p.ma_params = [1.]
    p.seasonal_ma_params = [1.]
    assert_(not p.is_invertible)


def test_is_valid():
    # Additional tests for the `is_valid` property (tests for NaN checks were
    # already done in `test_set_params_single`).
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
        enforce_stationarity=True, enforce_invertibility=True)
    p = params.SARIMAXParams(spec=spec)

    # Doesn't start out as valid
    assert_(not p.is_valid)
    # Given stationary / invertible values, it is valid
    p.params = [0.5, 0.5, 0.5, 0.5, 1.]
    assert_(p.is_valid)
    # With either non-stationary or non-invertible values, not valid
    p.params = [1., 0.5, 0.5, 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 1., 0.5, 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 0.5, 1., 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 0.5, 0.5, 1., 1.]
    assert_(not p.is_valid)


def test_repr_str():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Check when we haven't given any parameters
    assert_equal(repr(p), 'SARIMAXParams(exog=[nan nan], ar=[nan], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[nan nan nan nan nan nan nan]')

    p.exog_params = [1, 2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[nan], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[ 1.  2. nan nan nan nan nan]')

    p.ar_params = [0.5]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[1.  2.  0.5 nan nan nan nan]')

    p.ma_params = [0.2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[1.  2.  0.5 0.2 nan nan nan]')

    p.seasonal_ar_params = [0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[nan],'
                          ' sigma2=nan)')
    # assert_equal(str(p),
    #              '[1.e+00 2.e+00 5.e-01 2.e-01 1.e-03    nan    nan]')

    p.seasonal_ma_params = [-0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[-0.001],'
                          ' sigma2=nan)')
    # assert_equal(str(p), '[ 1.e+00  2.e+00  5.e-01  2.e-01  1.e-03'
    #                      ' -1.e-03     nan]')

    p.sigma2 = 10.123
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[-0.001],'
                          ' sigma2=10.123)')
    # assert_equal(str(p), '[ 1.0000e+00  2.0000e+00  5.0000e-01  2.0000e-01'
    #                      '  1.0000e-03 -1.0000e-03\n  1.0123e+01]')
