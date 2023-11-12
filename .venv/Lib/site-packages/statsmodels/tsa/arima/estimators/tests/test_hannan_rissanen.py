import numpy as np

import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
    hannan_rissanen, _validate_fixed_params,
    _package_fixed_and_free_params_info,
    _stitch_fixed_and_free_params
)
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch


@pytest.mark.low_precision('Test against Example 5.1.7 in Brockwell and Davis'
                           ' (2016)')
def test_brockwell_davis_example_517():
    # Get the lake data
    endog = lake.copy()

    # BD do not implement the "bias correction" third step that they describe,
    # so we can't use their results to test that. Thus here `unbiased=False`.
    # Note: it's not clear why BD use initial_order=22 (and they don't mention
    # that they do this), but it is the value that allows the test to pass.
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False)
    assert_allclose(hr.ar_params, [0.6961], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3788], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=1e-4)


def test_itsmr():
    # This is essentially a high precision version of
    # test_brockwell_davis_example_517, where the desired values were computed
    # from R itsmr::hannan; see results/results_hr.R
    endog = lake.copy()
    hr, _ = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True,
                            initial_ar_order=22, unbiased=False)

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_initial_order():
    endog = np.arange(20) * 1.0
    # TODO: shouldn't allow initial_ar_order <= ar_order
    hannan_rissanen(endog, ar_order=2, ma_order=0, initial_ar_order=1)
    # TODO: shouldn't allow initial_ar_order <= ma_order
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=1)
    # TODO: shouldn't allow initial_ar_order >= dataset
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=20)


@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_invalid_orders():
    endog = np.arange(2) * 1.0
    # TODO: shouldn't allow ar_order >= dataset
    hannan_rissanen(endog, ar_order=2)
    # TODO: shouldn't allow ma_order >= dataset
    hannan_rissanen(endog, ma_order=2)


@pytest.mark.todo('Improve checks on valid order parameters.')
@pytest.mark.smoke
def test_nonconsecutive_lags():
    endog = np.arange(20) * 1.0
    hannan_rissanen(endog, ar_order=[1, 4])
    hannan_rissanen(endog, ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[1, 4], ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[0, 0, 1])
    hannan_rissanen(endog, ma_order=[0, 0, 1])
    hannan_rissanen(endog, ar_order=[0, 0, 1], ma_order=[0, 0, 1])

    hannan_rissanen(endog, ar_order=0, ma_order=0)


def test_unbiased_error():
    # Test that we get the appropriate error when we specify unbiased=True
    # but the second-stage yields non-stationary parameters.
    endog = (np.arange(1000) * 1.0)
    with pytest.raises(ValueError, match='Cannot perform third step'):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)


def test_set_default_unbiased():
    # setting unbiased=None with stationary and invertible parameters should
    # yield the exact same results as setting unbiased=True
    endog = lake.copy()
    p_1, other_results_2 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=None
    )

    # unbiased=True
    p_2, other_results_1 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=True
    )

    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)

    # unbiased=False
    p_3, _ = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=False
    )
    assert not np.array_equal(p_1.ar_params, p_3.ar_params)


@pytest.mark.parametrize(
    "ar_order, ma_order, fixed_params, invalid_fixed_params",
    [
        # no fixed param
        (2, [1, 0, 1], None, None),
        ([0, 1], 0, {}, None),
        # invalid fixed params
        (1, 3, {"ar.L2": 1, "ma.L2": 0}, ["ar.L2"]),
        ([0, 1], [0, 0, 1], {"ma.L1": 0, "sigma2": 1}, ["ma.L2", "sigma2"]),
        (0, 0, {"ma.L1": 0, "ar.L1": 0}, ["ar.L1", "ma.L1"]),
        (5, [1, 0], {"random_param": 0, "ar.L1": 0}, ["random_param"]),
        # valid fixed params
        (0, 2, {"ma.L1": -1, "ma.L2": 1}, None),
        (1, 0, {"ar.L1": 0}, None),
        ([1, 0, 1], 3, {"ma.L2": 1, "ar.L3": -1}, None),
        # all fixed
        (2, 2, {"ma.L1": 1, "ma.L2": 1, "ar.L1": 1, "ar.L2": 1}, None)
    ]
)
def test_validate_fixed_params(ar_order, ma_order, fixed_params,
                               invalid_fixed_params):
    # test validation with both _validate_fixed_params and directly with
    # hannan_rissanen

    endog = np.random.normal(size=100)
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)

    if invalid_fixed_params is None:
        _validate_fixed_params(fixed_params, spec.param_names)
        hannan_rissanen(
            endog, ar_order=ar_order, ma_order=ma_order,
            fixed_params=fixed_params, unbiased=False
        )
    else:
        valid_params = sorted(list(set(spec.param_names) - {'sigma2'}))
        msg = (
            f"Invalid fixed parameter(s): {invalid_fixed_params}. "
            f"Please select among {valid_params}."
        )
        # using direct `assert` to test error message instead of `match` since
        # the error message contains regex characters
        with pytest.raises(ValueError) as e:
            _validate_fixed_params(fixed_params, spec.param_names)
            assert e.msg == msg
        with pytest.raises(ValueError) as e:
            hannan_rissanen(
                endog, ar_order=ar_order, ma_order=ma_order,
                fixed_params=fixed_params, unbiased=False
            )
            assert e.msg == msg


@pytest.mark.parametrize(
    "fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch",
    [
        ({}, [1], [], Bunch(
            # lags
            fixed_ar_lags=[], fixed_ma_lags=[],
            free_ar_lags=[1], free_ma_lags=[],
            # ixs
            fixed_ar_ix=np.array([], dtype=int),
            fixed_ma_ix=np.array([], dtype=int),
            free_ar_ix=np.array([0], dtype=int),
            free_ma_ix=np.array([], dtype=int),
            # fixed params
            fixed_ar_params=np.array([]), fixed_ma_params=np.array([]),
        )),
        ({"ar.L2": 0.1, "ma.L1": 0.2}, [2], [1, 3], Bunch(
            # lags
            fixed_ar_lags=[2], fixed_ma_lags=[1],
            free_ar_lags=[], free_ma_lags=[3],
            # ixs
            fixed_ar_ix=np.array([1], dtype=int),
            fixed_ma_ix=np.array([0], dtype=int),
            free_ar_ix=np.array([], dtype=int),
            free_ma_ix=np.array([2], dtype=int),
            # fixed params
            fixed_ar_params=np.array([0.1]), fixed_ma_params=np.array([0.2]),
        )),
        ({"ma.L5": 0.1, "ma.L10": 0.2}, [], [5, 10], Bunch(
            # lags
            fixed_ar_lags=[], fixed_ma_lags=[5, 10],
            free_ar_lags=[], free_ma_lags=[],
            # ixs
            fixed_ar_ix=np.array([], dtype=int),
            fixed_ma_ix=np.array([4, 9], dtype=int),
            free_ar_ix=np.array([], dtype=int),
            free_ma_ix=np.array([], dtype=int),
            # fixed params
            fixed_ar_params=np.array([]), fixed_ma_params=np.array([0.1, 0.2]),
        )),
    ]
)
def test_package_fixed_and_free_params_info(fixed_params, spec_ar_lags,
                                            spec_ma_lags, expected_bunch):
    actual_bunch = _package_fixed_and_free_params_info(
        fixed_params, spec_ar_lags, spec_ma_lags
    )
    assert isinstance(actual_bunch, Bunch)
    assert len(actual_bunch) == len(expected_bunch)
    assert actual_bunch.keys() == expected_bunch.keys()

    # check lags
    lags = ['fixed_ar_lags', 'fixed_ma_lags', 'free_ar_lags', 'free_ma_lags']
    for k in lags:
        assert isinstance(actual_bunch[k], list)
        assert actual_bunch[k] == expected_bunch[k]

    # check lags
    ixs = ['fixed_ar_ix', 'fixed_ma_ix', 'free_ar_ix', 'free_ma_ix']
    for k in ixs:
        assert isinstance(actual_bunch[k], np.ndarray)
        assert actual_bunch[k].dtype in [np.int64, np.int32]
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])

    params = ['fixed_ar_params', 'fixed_ma_params']
    for k in params:
        assert isinstance(actual_bunch[k], np.ndarray)
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])


@pytest.mark.parametrize(
    "fixed_lags, free_lags, fixed_params, free_params, "
    "spec_lags, expected_all_params",
    [
        ([], [], [], [], [], []),
        ([2], [], [0.2], [], [2], [0.2]),
        ([], [1], [], [0.2], [1], [0.2]),
        ([1], [3], [0.2], [-0.2], [1, 3],  [0.2, -0.2]),
        ([3], [1, 2], [0.2], [0.3, -0.2], [1, 2, 3], [0.3, -0.2, 0.2]),
        ([3, 1], [2, 4], [0.3, 0.1], [0.5, 0.],
         [1, 2, 3, 4], [0.1, 0.5, 0.3, 0.]),
        ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2],
         [1, 2, 3, 10], [0.3, -0.2, 0.2, 0.5]),
        # edge case where 'spec_lags' is somehow not sorted
        ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2],
         [3, 1, 10, 2], [0.2, 0.3, 0.5, -0.2]),
    ]
)
def test_stitch_fixed_and_free_params(fixed_lags, free_lags, fixed_params,
                                      free_params, spec_lags,
                                      expected_all_params):
    actual_all_params = _stitch_fixed_and_free_params(
        fixed_lags, fixed_params, free_lags, free_params, spec_lags
    )
    assert actual_all_params == expected_all_params


@pytest.mark.parametrize(
    "fixed_params",
    [
        {"ar.L1": 0.69607715},  # fix ar
        {"ma.L1": 0.37879692},  # fix ma
        {"ar.L1": 0.69607715, "ma.L1": 0.37879692},  # no free params
    ]
)
def test_itsmr_with_fixed_params(fixed_params):
    # This test is a variation of test_itsmr where we fix 1 or more parameters
    # for Example 5.1.7 in Brockwell and Davis (2016) and check that free
    # parameters are still correct'.

    endog = lake.copy()
    hr, _ = hannan_rissanen(
        endog, ar_order=1, ma_order=1, demean=True,
        initial_ar_order=22, unbiased=False,
        fixed_params=fixed_params
    )

    assert_allclose(hr.ar_params, [0.69607715], atol=1e-4)
    assert_allclose(hr.ma_params, [0.3787969217], atol=1e-4)

    # Because our fast implementation of the innovations algorithm does not
    # allow for non-stationary processes, the estimate of the variance returned
    # by `hannan_rissanen` is based on the residuals from the least-squares
    # regression, rather than (as reported by BD) based on the innovations
    # algorithm output. Since the estimates here do correspond to a stationary
    # series, we can compute the innovations variance manually to check
    # against BD.
    u, v = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params,
                            sigma2=1)
    tmp = u / v**0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=1e-4)


def test_unbiased_error_with_fixed_params():
    # unbiased=True with fixed params should throw NotImplementedError for now
    endog = np.random.normal(size=1000)
    msg = (
        "Third step of Hannan-Rissanen estimation to remove parameter bias"
        " is not yet implemented for the case with fixed parameters."
    )
    with pytest.raises(NotImplementedError, match=msg):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True,
                        fixed_params={"ar.L1": 0})


def test_set_default_unbiased_with_fixed_params():
    # setting unbiased=None with fixed params should yield the exact same
    # results as setting unbiased=False
    endog = np.random.normal(size=1000)
    # unbiased=None
    p_1, other_results_2 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=None,
        fixed_params={"ar.L1": 0.69607715}
    )
    # unbiased=False
    p_2, other_results_1 = hannan_rissanen(
        endog, ar_order=1, ma_order=1, unbiased=False,
        fixed_params={"ar.L1": 0.69607715}
    )

    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)
