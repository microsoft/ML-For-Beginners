import numpy as np

from numpy.testing import assert_equal, assert_raises

from statsmodels.tsa.arima.tools import (
    standardize_lag_order, validate_basic)


def test_standardize_lag_order_int():
    # Integer input
    assert_equal(standardize_lag_order(0, title='test'), 0)
    assert_equal(standardize_lag_order(3), 3)


def test_standardize_lag_order_list_int():
    # List input, lags
    assert_equal(standardize_lag_order([]), 0)
    assert_equal(standardize_lag_order([1, 2]), 2)
    assert_equal(standardize_lag_order([1, 3]), [1, 3])


def test_standardize_lag_order_tuple_int():
    # Non-list iterable input, lags
    assert_equal(standardize_lag_order((1, 2)), 2)
    assert_equal(standardize_lag_order((1, 3)), [1, 3])


def test_standardize_lag_order_ndarray_int():
    assert_equal(standardize_lag_order(np.array([1, 2])), 2)
    assert_equal(standardize_lag_order(np.array([1, 3])), [1, 3])


def test_standardize_lag_order_list_bool():
    # List input, booleans
    assert_equal(standardize_lag_order([0]), 0)
    assert_equal(standardize_lag_order([1]), 1)
    assert_equal(standardize_lag_order([0, 1]), [2])
    assert_equal(standardize_lag_order([0, 1, 0, 1]), [2, 4])


def test_standardize_lag_order_tuple_bool():
    # Non-list iterable input, lags
    assert_equal(standardize_lag_order((0)), 0)
    assert_equal(standardize_lag_order((1)), 1)
    assert_equal(standardize_lag_order((0, 1)), [2])
    assert_equal(standardize_lag_order((0, 1, 0, 1)), [2, 4])


def test_standardize_lag_order_ndarray_bool():
    assert_equal(standardize_lag_order(np.array([0])), 0)
    assert_equal(standardize_lag_order(np.array([1])), 1)
    assert_equal(standardize_lag_order(np.array([0, 1])), [2])
    assert_equal(standardize_lag_order(np.array([0, 1, 0, 1])), [2, 4])


def test_standardize_lag_order_misc():
    # Misc.
    assert_equal(standardize_lag_order(np.array([[1], [3]])), [1, 3])


def test_standardize_lag_order_invalid():
    # Invalid input
    assert_raises(TypeError, standardize_lag_order, None)
    assert_raises(ValueError, standardize_lag_order, 1.2)
    assert_raises(ValueError, standardize_lag_order, -1)
    assert_raises(ValueError, standardize_lag_order,
                  np.arange(4).reshape(2, 2))
    # Boolean list can't have 2, lag order list can't have 0
    assert_raises(ValueError, standardize_lag_order, [0, 2])
    # Can't have duplicates
    assert_raises(ValueError, standardize_lag_order, [1, 1, 2])


def test_validate_basic():
    # Valid parameters
    assert_equal(validate_basic([], 0, title='test'), [])
    assert_equal(validate_basic(0, 1), [0])
    assert_equal(validate_basic([0], 1), [0])
    assert_equal(validate_basic(np.array([1.2, 0.5 + 1j]), 2),
                 np.array([1.2, 0.5 + 1j]))
    assert_equal(
        validate_basic([np.nan, -np.inf, np.inf], 3, allow_infnan=True),
        [np.nan, -np.inf, np.inf])

    # Invalid parameters
    assert_raises(ValueError, validate_basic, [], 1, title='test')
    assert_raises(ValueError, validate_basic, 0, 0)
    assert_raises(ValueError, validate_basic, 'a', 1)
    assert_raises(ValueError, validate_basic, None, 1)
    assert_raises(ValueError, validate_basic, np.nan, 1)
    assert_raises(ValueError, validate_basic, np.inf, 1)
    assert_raises(ValueError, validate_basic, -np.inf, 1)
    assert_raises(ValueError, validate_basic, [1, 2], 1)
