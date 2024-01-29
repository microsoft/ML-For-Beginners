"""
Tests for tools

Author: Chad Fulton
License: Simplified-BSD
"""

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
                           assert_array_equal, assert_almost_equal)
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf


class TestCompanionMatrix:

    cases = [
        (2, np.array([[0, 1], [0, 0]])),
        ([1, -1, -2], np.array([[1, 1],
                                [2, 0]])),
        ([1, -1, -2, -3], np.array([[1, 1, 0],
                                    [2, 0, 1],
                                    [3, 0, 0]])),
        ([1, -np.array([[1, 2], [3, 4]]), -np.array([[5, 6], [7, 8]])],
         np.array([[1, 2, 5, 6],
                   [3, 4, 7, 8],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0]]).T),
        # GH 5570
        (np.int64(2), np.array([[0, 1], [0, 0]]))
    ]

    def test_cases(self):
        for polynomial, result in self.cases:
            assert_equal(tools.companion_matrix(polynomial), result)


class TestDiff:

    x = np.arange(10)
    cases = [
        # diff = 1
        ([1, 2, 3], 1, None, 1, [1, 1]),
        # diff = 2
        (x, 2, None, 1, [0]*8),
        # diff = 1, seasonal_diff=1, seasonal_periods=4
        (x, 1, 1, 4, [0]*5),
        (x**2, 1, 1, 4, [8]*5),
        (x**3, 1, 1, 4, [60, 84, 108, 132, 156]),
        # diff = 1, seasonal_diff=2, seasonal_periods=2
        (x, 1, 2, 2, [0]*5),
        (x**2, 1, 2, 2, [0]*5),
        (x**3, 1, 2, 2, [24]*5),
        (x**4, 1, 2, 2, [240, 336, 432, 528, 624]),
    ]

    # TODO: use pytest.mark.parametrize?
    def test_cases(self):
        # Basic cases
        for series, diff, seas_diff, seasonal_periods, result in self.cases:
            seasonal_diff = seas_diff

            # Test numpy array
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)

            # Test as Pandas Series
            series = pd.Series(series)

            # Rewrite to test as n-dimensional array
            series = np.c_[series, series]
            result = np.c_[result, result]

            # Test Numpy array
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)

            # Test as Pandas DataFrame
            series = pd.DataFrame(series)
            x = tools.diff(series, diff, seasonal_diff, seasonal_periods)
            assert_almost_equal(x, result)


class TestSolveDiscreteLyapunov:

    def solve_dicrete_lyapunov_direct(self, a, q, complex_step=False):
        # This is the discrete Lyapunov solver as "real function of real
        # variables":  the difference between this and the usual, complex,
        # version is that in the Kronecker product the second argument is
        # *not* conjugated here.
        if not complex_step:
            lhs = np.kron(a, a.conj())
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())
        else:
            lhs = np.kron(a, a)
            lhs = np.eye(lhs.shape[0]) - lhs
            x = np.linalg.solve(lhs, q.flatten())

        return np.reshape(x, q.shape)

    def test_univariate(self):
        # Real case
        a = np.array([[0.5]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = np.array([[0.5+1j]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = np.array([[0.5+1j]])
        q = np.array([[10.]])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)

    def test_multivariate(self):
        # Real case
        a = tools.companion_matrix([1, -0.4, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q)
        desired = solve_discrete_lyapunov(a, q)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a complex
        # function)
        a = tools.companion_matrix([1, -0.4+0.1j, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=False)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=False)
        assert_allclose(actual, desired)

        # Complex case (where the Lyapunov equation is taken as a real
        # function)
        a = tools.companion_matrix([1, -0.4+0.1j, 0.5])
        q = np.diag([10., 5.])
        actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
        desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
        assert_allclose(actual, desired)


class TestConcat:

    x = np.arange(10)

    valid = [
        (((1, 2, 3), (4,)), (1, 2, 3, 4)),
        (((1, 2, 3), [4]), (1, 2, 3, 4)),
        (([1, 2, 3], np.r_[4]), (1, 2, 3, 4)),
        ((np.r_[1, 2, 3], pd.Series([4])), 0, True, (1, 2, 3, 4)),
        ((pd.Series([1, 2, 3]), pd.Series([4])), 0, True, (1, 2, 3, 4)),
        ((np.c_[x[:2], x[:2]], np.c_[x[2:3], x[2:3]]), np.c_[x[:3], x[:3]]),
        ((np.c_[x[:2], x[:2]].T, np.c_[x[2:3], x[2:3]].T),
         1, np.c_[x[:3], x[:3]].T),
        ((pd.DataFrame(np.c_[x[:2], x[:2]]), np.c_[x[2:3], x[2:3]]),
         0, True, np.c_[x[:3], x[:3]]),
    ]

    invalid = [
        (((1, 2, 3), pd.Series([4])), ValueError),
        (((1, 2, 3), np.array([[1, 2]])), ValueError)
    ]

    def test_valid(self):
        for args in self.valid:
            assert_array_equal(tools.concat(*args[:-1]), args[-1])

    def test_invalid(self):
        for args in self.invalid:
            with pytest.raises(args[-1]):
                tools.concat(*args[:-1])


class TestIsInvertible:

    cases = [
        ([1, -0.5], True),
        ([1, 1-1e-9], True),
        ([1, 1], False),
        ([1, 0.9, 0.1], True),
        (np.array([1, 0.9, 0.1]), True),
        (pd.Series([1, 0.9, 0.1]), True)
    ]

    def test_cases(self):
        for polynomial, invertible in self.cases:
            assert_equal(tools.is_invertible(polynomial), invertible)


class TestConstrainStationaryUnivariate:

    cases = [
        (np.array([2.]), -2./((1+2.**2)**0.5))
    ]

    def test_cases(self):
        for unconstrained, constrained in self.cases:
            result = tools.constrain_stationary_univariate(unconstrained)
            assert_equal(result, constrained)


class TestUnconstrainStationaryUnivariate:

    cases = [
        (np.array([-2./((1+2.**2)**0.5)]), np.array([2.]))
    ]

    def test_cases(self):
        for constrained, unconstrained in self.cases:
            result = tools.unconstrain_stationary_univariate(constrained)
            assert_allclose(result, unconstrained)


class TestStationaryUnivariate:
    # Test that the constraint and unconstrained functions are inverses

    constrained_cases = [
        np.array([0]), np.array([0.1]), np.array([-0.5]), np.array([0.999])]
    unconstrained_cases = [
        np.array([10.]), np.array([-40.42]), np.array([0.123])]

    def test_cases(self):
        for constrained in self.constrained_cases:
            unconstrained = tools.unconstrain_stationary_univariate(constrained)  # noqa:E501
            reconstrained = tools.constrain_stationary_univariate(unconstrained)  # noqa:E501
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            constrained = tools.constrain_stationary_univariate(unconstrained)
            reunconstrained = tools.unconstrain_stationary_univariate(constrained)  # noqa:E501
            assert_allclose(reunconstrained, unconstrained)


class TestValidateMatrixShape:
    # name, shape, nrows, ncols, nobs
    valid = [
        ('TEST', (5, 2), 5, 2, None),
        ('TEST', (5, 2), 5, 2, 10),
        ('TEST', (5, 2, 10), 5, 2, 10),
    ]
    invalid = [
        ('TEST', (5,), 5, None, None),
        ('TEST', (5, 1, 1, 1), 5, 1, None),
        ('TEST', (5, 2), 10, 2, None),
        ('TEST', (5, 2), 5, 1, None),
        ('TEST', (5, 2, 10), 5, 2, None),
        ('TEST', (5, 2, 10), 5, 2, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_matrix_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            with pytest.raises(ValueError):
                tools.validate_matrix_shape(*args)


class TestValidateVectorShape:
    # name, shape, nrows, ncols, nobs
    valid = [
        ('TEST', (5,), 5, None),
        ('TEST', (5,), 5, 10),
        ('TEST', (5, 10), 5, 10),
    ]
    invalid = [
        ('TEST', (5, 2, 10), 5, 10),
        ('TEST', (5,), 10, None),
        ('TEST', (5, 10), 5, None),
        ('TEST', (5, 10), 5, 5),
    ]

    def test_valid_cases(self):
        for args in self.valid:
            # Just testing that no exception is raised
            tools.validate_vector_shape(*args)

    def test_invalid_cases(self):
        for args in self.invalid:
            with pytest.raises(ValueError):
                tools.validate_vector_shape(*args)


def test_multivariate_acovf():
    _acovf = tools._compute_multivariate_acovf_from_coefficients

    # Test for a VAR(1) process. From Lutkepohl (2007), pages 27-28.
    # See (2.1.14) for Phi_1, (2.1.33) for Sigma_u, and (2.1.34) for Gamma_0
    Sigma_u = np.array([[2.25, 0,   0],
                        [0,    1.0, 0.5],
                        [0,    0.5, 0.74]])
    Phi_1 = np.array([[0.5, 0,   0],
                      [0.1, 0.1, 0.3],
                      [0,   0.2, 0.3]])
    Gamma_0 = np.array([[3.0,   0.161, 0.019],
                        [0.161, 1.172, 0.674],
                        [0.019, 0.674, 0.954]])
    assert_allclose(_acovf([Phi_1], Sigma_u)[0], Gamma_0, atol=1e-3)

    # Test for a VAR(2) process. From Lutkepohl (2007), pages 28-29
    # See (2.1.40) for Phi_1, Phi_2, (2.1.14) for Sigma_u, and (2.1.42) for
    # Gamma_0, Gamma_1
    Sigma_u = np.diag([0.09, 0.04])
    Phi_1 = np.array([[0.5, 0.1],
                      [0.4, 0.5]])
    Phi_2 = np.array([[0,    0],
                      [0.25, 0]])
    Gamma_0 = np.array([[0.131, 0.066],
                        [0.066, 0.181]])
    Gamma_1 = np.array([[0.072, 0.051],
                        [0.104, 0.143]])
    Gamma_2 = np.array([[0.046, 0.040],
                        [0.113, 0.108]])
    Gamma_3 = np.array([[0.035, 0.031],
                        [0.093, 0.083]])

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=0),
        [Gamma_0], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=1),
        [Gamma_0, Gamma_1], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u),
        [Gamma_0, Gamma_1], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=2),
        [Gamma_0, Gamma_1, Gamma_2], atol=1e-3)

    assert_allclose(
        _acovf([Phi_1, Phi_2], Sigma_u, maxlag=3),
        [Gamma_0, Gamma_1, Gamma_2, Gamma_3], atol=1e-3)

    # Test sample acovf in the univariate case against sm.tsa.acovf
    x = np.arange(20)*1.0
    assert_allclose(
        np.squeeze(tools._compute_multivariate_sample_acovf(x, maxlag=4)),
        acovf(x, fft=False)[:5])


def test_multivariate_pacf():
    # Test sample acovf in the univariate case against sm.tsa.acovf
    np.random.seed(1234)
    x = np.arange(10000)
    y = np.random.normal(size=10000)
    # Note: could make this test more precise with higher nobs, but no need to
    assert_allclose(
        tools._compute_multivariate_sample_pacf(np.c_[x, y], maxlag=1)[0],
        np.diag([1, 0]), atol=1e-2)


class TestConstrainStationaryMultivariate:

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2.]]), np.eye(1), np.array([[2./((1+2.**2)**0.5)]])),
        # Same as above, but now a list input / output
        ([np.array([[2.]])], np.eye(1), [np.array([[2./((1+2.**2)**0.5)]])])
    ]

    eigval_cases = [
        [np.array([[0]])],
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, .3], [.5, -30]])],
    ]

    def test_cases(self):
        # Test against known results
        for unconstrained, error_variance, constrained in self.cases:
            result = tools.constrain_stationary_multivariate(
                unconstrained, error_variance)
            assert_allclose(result[0], constrained)

        # Test that the constrained results correspond to companion matrices
        # with eigenvalues less than 1 in modulus
        for unconstrained in self.eigval_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)  # noqa:E501
            companion = tools.companion_matrix(
                [1] + [-np.squeeze(constrained[i])
                       for i in range(len(constrained))]
            ).T
            assert_array_less(np.abs(np.linalg.eigvals(companion)), 1)


class TestUnconstrainStationaryMultivariate:

    cases = [
        # This is the same test as the univariate case above, except notice
        # the sign difference; this is an array input / output
        (np.array([[2./((1+2.**2)**0.5)]]), np.eye(1), np.array([[2.]])),
        # Same as above, but now a list input / output
        ([np.array([[2./((1+2.**2)**0.5)]])], np.eye(1), [np.array([[2.]])])
    ]

    def test_cases(self):
        for constrained, error_variance, unconstrained in self.cases:
            result = tools.unconstrain_stationary_multivariate(
                constrained, error_variance)
            assert_allclose(result[0], unconstrained)


class TestStationaryMultivariate:
    # Test that the constraint and unconstrained functions are inverses

    constrained_cases = [
        np.array([[0]]), np.array([[0.1]]),
        np.array([[-0.5]]), np.array([[0.999]]),
        [np.array([[0]])],
        np.array([[0.8, -0.2]]),
        [np.array([[0.8]]), np.array([[-0.2]])],
        [np.array([[0.3, 0.01], [-0.23, 0.15]]),
         np.array([[0.1, 0.03], [0.05, -0.3]])],
        np.array([[0.3, 0.01, 0.1, 0.03], [-0.23, 0.15, 0.05, -0.3]])
    ]
    unconstrained_cases = [
        np.array([[0]]), np.array([[-40.42]]), np.array([[0.123]]),
        [np.array([[0]])],
        np.array([[100, 50]]),
        [np.array([[100]]), np.array([[50]])],
        [np.array([[30, 1], [-23, 15]]), np.array([[10, .3], [.5, -30]])],
        np.array([[30, 1, 10, .3], [-23, 15, .5, -30]])
    ]

    def test_cases(self):
        for constrained in self.constrained_cases:
            if type(constrained) is list:
                cov = np.eye(constrained[0].shape[0])
            else:
                cov = np.eye(constrained.shape[0])
            unconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)  # noqa:E501
            reconstrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)  # noqa:E501
            assert_allclose(reconstrained, constrained)

        for unconstrained in self.unconstrained_cases:
            if type(unconstrained) is list:
                cov = np.eye(unconstrained[0].shape[0])
            else:
                cov = np.eye(unconstrained.shape[0])
            constrained, _ = tools.constrain_stationary_multivariate(unconstrained, cov)  # noqa:E501
            reunconstrained, _ = tools.unconstrain_stationary_multivariate(constrained, cov)  # noqa:E501
            # Note: low tolerance comes from last example in
            # unconstrained_cases, but is not a real problem
            assert_allclose(reunconstrained, unconstrained, atol=1e-4)


def test_reorder_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 3

    missing = np.zeros((k_endog, nobs))
    given = np.zeros((k_endog, k_states, nobs))
    given[:, :, :] = np.array([[11, 12, 13],
                               [21, 22, 23],
                               [31, 32, 33]])[:, :, np.newaxis]
    desired = given.copy()

    missing[0, 0] = 1
    given[:, :, 0] = np.array([[21, 22, 23],
                               [31, 32, 33],
                               [0,  0,  0]])
    desired[0, :, 0] = 0

    missing[:2, 1] = 1
    given[:, :, 1] = np.array([[31, 32, 33],
                               [0,  0,  0],
                               [0,  0,  0]])
    desired[:2, :, 1] = 0

    missing[0, 2] = 1
    missing[2, 2] = 1
    given[:, :, 2] = np.array([[21, 22, 23],
                               [0,  0,  0],
                               [0,  0,  0]])
    desired[0, :, 2] = 0
    desired[2, :, 2] = 0

    missing[1, 3] = 1
    given[:, :, 3] = np.array([[11, 12, 13],
                               [31, 32, 33],
                               [0,  0,  0]])
    desired[1, :, 3] = 0

    missing[2, 4] = 1
    given[:, :, 4] = np.array([[11, 12, 13],
                               [21, 22, 23],
                               [0,  0,  0]])
    desired[2, :, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing,
                                 True, False, False, inplace=True)

    assert_equal(actual, desired)


def test_reorder_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 3

    missing = np.zeros((k_endog, nobs))
    given = np.zeros((k_endog, k_states, nobs))
    given[:, :, :] = np.array([[11, 12, 13],
                               [21, 22, 23],
                               [31, 32, 33]])[:, :, np.newaxis]
    desired = given.copy()

    missing[0, 0] = 1
    given[:, :, :] = np.array([[12, 13, 0],
                               [22, 23, 0],
                               [32, 33, 0]])[:, :, np.newaxis]
    desired[:, 0, 0] = 0

    missing[:2, 1] = 1
    given[:, :, 1] = np.array([[13, 0, 0],
                               [23, 0, 0],
                               [33, 0, 0]])
    desired[:, :2, 1] = 0

    missing[0, 2] = 1
    missing[2, 2] = 1
    given[:, :, 2] = np.array([[12, 0, 0],
                               [22, 0, 0],
                               [32, 0, 0]])
    desired[:, 0, 2] = 0
    desired[:, 2, 2] = 0

    missing[1, 3] = 1
    given[:, :, 3] = np.array([[11, 13, 0],
                               [21, 23, 0],
                               [31, 33, 0]])
    desired[:, 1, 3] = 0

    missing[2, 4] = 1
    given[:, :, 4] = np.array([[11, 12, 0],
                               [21, 22, 0],
                               [31, 32, 0]])
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing,
                                 False, True, False, inplace=True)

    assert_equal(actual[:, :, 4], desired[:, :, 4])


def test_reorder_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, k_endog, nobs))
    given[:, :, :] = np.array([[11, 12, 13],
                               [21, 22, 23],
                               [31, 32, 33]])[:, :, np.newaxis]
    desired = given.copy()

    given[:, :, 0] = np.array([[22, 23, 0],
                               [32, 33, 0],
                               [0,  0,  0]])
    desired[0, :, 0] = 0
    desired[:, 0, 0] = 0

    given[:, :, 1] = np.array([[33, 0, 0],
                               [0,  0, 0],
                               [0,  0,  0]])
    desired[:2, :, 1] = 0
    desired[:, :2, 1] = 0

    given[:, :, 2] = np.array([[22, 0, 0],
                               [0,  0, 0],
                               [0,  0, 0]])
    desired[0, :, 2] = 0
    desired[:, 0, 2] = 0
    desired[2, :, 2] = 0
    desired[:, 2, 2] = 0

    given[:, :, 3] = np.array([[11, 13, 0],
                               [31, 33, 0],
                               [0,  0,  0]])
    desired[1, :, 3] = 0
    desired[:, 1, 3] = 0

    given[:, :, 4] = np.array([[11, 12, 0],
                               [21, 22, 0],
                               [0,  0,  0]])
    desired[2, :, 4] = 0
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing,
                                 True, True, False, inplace=True)

    assert_equal(actual, desired)


def test_reorder_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, k_endog, nobs))
    given[:, :, :] = np.array([[11, 00, 00],
                               [00, 22, 00],
                               [00, 00, 33]])[:, :, np.newaxis]
    desired = given.copy()

    given[:, :, 0] = np.array([[22, 00, 0],
                               [00, 33, 0],
                               [0,  0,  0]])
    desired[0, :, 0] = 0
    desired[:, 0, 0] = 0

    given[:, :, 1] = np.array([[33, 0, 0],
                               [0,  0, 0],
                               [0,  0,  0]])
    desired[:2, :, 1] = 0
    desired[:, :2, 1] = 0

    given[:, :, 2] = np.array([[22, 0, 0],
                               [0,  0, 0],
                               [0,  0, 0]])
    desired[0, :, 2] = 0
    desired[:, 0, 2] = 0
    desired[2, :, 2] = 0
    desired[:, 2, 2] = 0

    given[:, :, 3] = np.array([[11, 00, 0],
                               [00, 33, 0],
                               [0,  0,  0]])
    desired[1, :, 3] = 0
    desired[:, 1, 3] = 0

    given[:, :, 4] = np.array([[11, 00, 0],
                               [00, 22, 0],
                               [0,  0,  0]])
    desired[2, :, 4] = 0
    desired[:, 2, 4] = 0

    actual = np.asfortranarray(given.copy())
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing,
                                 True, True, False, inplace=True)
    assert_equal(actual, desired)

    actual = np.asfortranarray(given.copy())
    tools.reorder_missing_matrix(actual, missing,
                                 True, True, True, inplace=True)
    assert_equal(actual, desired)


def test_reorder_vector():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    given = np.zeros((k_endog, nobs))
    given[:, :] = np.array([1, 2, 3])[:, np.newaxis]
    desired = given.copy()

    given[:, 0] = [2, 3, 0]
    desired[:, 0] = [0, 2, 3]
    given[:, 1] = [3, 0, 0]
    desired[:, 1] = [0, 0, 3]
    given[:, 2] = [2, 0, 0]
    desired[:, 2] = [0, 2, 0]
    given[:, 3] = [1, 3, 0]
    desired[:, 3] = [1, 0, 3]
    given[:, 4] = [1, 2, 0]
    desired[:, 4] = [1, 2, 0]

    actual = np.asfortranarray(given.copy())
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_vector(actual, missing, inplace=True)
    assert_equal(actual, desired)


def test_copy_missing_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 2

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_states, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :, t] = 1.
    B = np.zeros((k_endog, k_states, nobs), order='F')

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, False, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 2

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_states, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:, :n, t] = 1.
    B = np.zeros((k_states, k_endog, nobs), order='F')

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, False, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :n, t] = 1.
    B = np.zeros((k_endog, k_endog, nobs), order='F')

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_missing_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, :n, t] = np.eye(n)
    B = np.zeros((k_endog, k_endog, nobs), order='F')

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_matrix(A, B, missing, True, True, False, inplace=True)
    assert_equal(B, A)

    B = np.zeros((k_endog, k_endog, nobs), order='F')
    tools.copy_missing_matrix(A, B, missing, True, True, True, inplace=True)
    assert_equal(B, A)


def test_copy_missing_vector():
    nobs = 5
    k_endog = 3

    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1

    A = np.zeros((k_endog, nobs))
    for t in range(nobs):
        n = int(k_endog - np.sum(missing[:, t]))
        A[:n, t] = 1.
    B = np.zeros((k_endog, nobs), order='F')

    missing = np.asfortranarray(missing.astype(np.int32))
    tools.copy_missing_vector(A, B, missing, inplace=True)
    assert_equal(B, A)


def test_copy_index_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 2

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_states, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, :, t] = 1.
    B = np.zeros((k_endog, k_states, nobs), order='F')

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, False, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_matrix_cols():
    nobs = 5
    k_endog = 3
    k_states = 2

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_states, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[:, i, t] = 1.
    B = np.zeros((k_states, k_endog, nobs), order='F')

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, False, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_submatrix():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, :, t] = 1.
                A[:, i, t] = 1.
    B = np.zeros((k_endog, k_endog, nobs), order='F')

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, True, False, inplace=True)
    assert_equal(B, A)


def test_copy_index_diagonal_submatrix():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, i, t] = 1.
    B = np.zeros((k_endog, k_endog, nobs), order='F')

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_matrix(A, B, index, True, True, False, inplace=True)
    assert_equal(B, A)

    B = np.zeros((k_endog, k_endog, nobs), order='F')
    tools.copy_index_matrix(A, B, index, True, True, True, inplace=True)
    assert_equal(B, A)


def test_copy_index_vector():
    nobs = 5
    k_endog = 3

    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1

    A = np.zeros((k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, t] = 1.
    B = np.zeros((k_endog, nobs), order='F')

    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_vector(A, B, index, inplace=True)
    assert_equal(B, A)
