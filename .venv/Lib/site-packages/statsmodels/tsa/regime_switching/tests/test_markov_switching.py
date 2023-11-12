"""
General tests for Markov switching models

Author: Chad Fulton
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd

from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching


def test_params():
    def check_transtion_2(params):
        assert_equal(params['regime_transition'], np.s_[0:2])
        assert_equal(params[0, 'regime_transition'], [0])
        assert_equal(params[1, 'regime_transition'], [1])
        assert_equal(params['regime_transition', 0], [0])
        assert_equal(params['regime_transition', 1], [1])

    def check_transition_3(params):
        assert_equal(params['regime_transition'], np.s_[0:6])
        assert_equal(params[0, 'regime_transition'], [0, 3])
        assert_equal(params[1, 'regime_transition'], [1, 4])
        assert_equal(params[2, 'regime_transition'], [2, 5])
        assert_equal(params['regime_transition', 0], [0, 3])
        assert_equal(params['regime_transition', 1], [1, 4])
        assert_equal(params['regime_transition', 2], [2, 5])

    params = markov_switching.MarkovSwitchingParams(k_regimes=2)
    params['regime_transition'] = [1]
    assert_equal(params.k_params, 1 * 2)
    assert_equal(params[0], [0])
    assert_equal(params[1], [1])
    check_transtion_2(params)

    params['exog'] = [0, 1]
    assert_equal(params.k_params, 1 * 2 + 1 + 1 * 2)
    assert_equal(params[0], [0, 2, 3])
    assert_equal(params[1], [1, 2, 4])
    check_transtion_2(params)
    assert_equal(params['exog'], np.s_[2:5])
    assert_equal(params[0, 'exog'], [2, 3])
    assert_equal(params[1, 'exog'], [2, 4])
    assert_equal(params['exog', 0], [2, 3])
    assert_equal(params['exog', 1], [2, 4])

    params = markov_switching.MarkovSwitchingParams(k_regimes=3)
    params['regime_transition'] = [1, 1]
    assert_equal(params.k_params, 2 * 3)
    assert_equal(params[0], [0, 3])
    assert_equal(params[1], [1, 4])
    assert_equal(params[2], [2, 5])
    check_transition_3(params)

    # Test for invalid parameter setting
    assert_raises(IndexError, params.__setitem__, None, [1, 1])

    # Test for invalid parameter selection
    assert_raises(IndexError, params.__getitem__, None)
    assert_raises(IndexError, params.__getitem__, (0, 0))
    assert_raises(IndexError, params.__getitem__, ('exog', 'exog'))
    assert_raises(IndexError, params.__getitem__, ('exog', 0, 1))


def test_init_endog():
    index = pd.date_range(start='1950-01-01', periods=10, freq='D')
    endog = [
        np.ones(10), pd.Series(np.ones(10), index=index), np.ones((10, 1)),
        pd.DataFrame(np.ones((10, 1)), index=index)
    ]
    for _endog in endog:
        mod = markov_switching.MarkovSwitching(_endog, k_regimes=2)
        assert_equal(mod.nobs, 10)
        assert_equal(mod.endog, _endog.squeeze())
        assert_equal(mod.k_regimes, 2)
        assert_equal(mod.tvtp, False)
        assert_equal(mod.k_tvtp, 0)
        assert_equal(mod.k_params, 2)

    # Invalid: k_regimes < 2
    endog = np.ones(10)
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog,
                  k_regimes=1)

    # Invalid: multiple endog columns
    endog = np.ones((10, 2))
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog,
                  k_regimes=2)


def test_init_exog_tvtp():
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2,
                                           exog_tvtp=exog_tvtp)
    assert_equal(mod.tvtp, True)
    assert_equal(mod.k_tvtp, 2)

    # Invalid exog_tvtp (too many obs)
    exog_tvtp = np.c_[np.ones((11, 1)), (np.arange(11) + 1)[:, np.newaxis]]
    assert_raises(ValueError, markov_switching.MarkovSwitching, endog,
                  k_regimes=2, exog_tvtp=exog_tvtp)


def test_transition_matrix():
    # k_regimes = 2
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0., 0., 1.]
    transition_matrix = np.zeros((2, 2, 1))
    transition_matrix[1, :] = 1.
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)

    # k_regimes = 3
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=3)
    params = np.r_[[0]*3, [0.2]*3, 1.]
    transition_matrix = np.zeros((3, 3, 1))
    transition_matrix[1, :, 0] = 0.2
    transition_matrix[2, :, 0] = 0.8
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix)

    # k_regimes = 2, tvtp
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2,
                                           exog_tvtp=exog_tvtp)

    # If all TVTP regression coefficients are zero, then the logit transform
    # results in exp(0) / (1 + exp(0)) = 0.5 for all parameters; since it's
    # k_regimes=2 the remainder calculation is also 0.5.
    params = np.r_[0, 0, 0, 0]
    assert_allclose(mod.regime_transition_matrix(params), 0.5)

    # Manually compute the TVTP coefficients
    params = np.r_[1, 2, 1, 2]
    transition_matrix = np.zeros((2, 2, 10))

    coeffs0 = np.sum(exog_tvtp, axis=1)
    p11 = np.exp(coeffs0) / (1 + np.exp(coeffs0))
    transition_matrix[0, 0, :] = p11
    transition_matrix[1, 0, :] = 1 - p11

    coeffs1 = np.sum(2 * exog_tvtp, axis=1)
    p21 = np.exp(coeffs1) / (1 + np.exp(coeffs1))
    transition_matrix[0, 1, :] = p21
    transition_matrix[1, 1, :] = 1 - p21
    assert_allclose(mod.regime_transition_matrix(params), transition_matrix,
                    atol=1e-10)

    # k_regimes = 3, tvtp
    endog = np.ones(10)
    exog_tvtp = np.c_[np.ones((10, 1)), (np.arange(10) + 1)[:, np.newaxis]]
    mod = markov_switching.MarkovSwitching(
        endog, k_regimes=3, exog_tvtp=exog_tvtp)

    # If all TVTP regression coefficients are zero, then the logit transform
    # results in exp(0) / (1 + exp(0) + exp(0)) = 1/3 for all parameters;
    # since it's k_regimes=3 the remainder calculation is also 1/3.
    params = np.r_[[0]*12]
    assert_allclose(mod.regime_transition_matrix(params), 1 / 3)

    # Manually compute the TVTP coefficients for the first column
    params = np.r_[[0]*6, [2]*6]
    transition_matrix = np.zeros((3, 3, 10))

    p11 = np.zeros(10)
    p12 = 2 * np.sum(exog_tvtp, axis=1)
    tmp = np.exp(np.c_[p11, p12]).T
    transition_matrix[:2, 0, :] = tmp / (1 + np.sum(tmp, axis=0))
    transition_matrix[2, 0, :] = (
        1 - np.sum(transition_matrix[:2, 0, :], axis=0))
    assert_allclose(mod.regime_transition_matrix(params)[:, 0, :],
                    transition_matrix[:, 0, :], atol=1e-10)


def test_initial_probabilities():
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2)
    params = np.r_[0.5, 0.5, 1.]

    # Valid known initial probabilities
    mod.initialize_known([0.2, 0.8])
    assert_allclose(mod.initial_probabilities(params), [0.2, 0.8])

    # Invalid known initial probabilities (too many elements)
    assert_raises(ValueError, mod.initialize_known, [0.2, 0.2, 0.6])

    # Invalid known initial probabilities (does not sum to 1)
    assert_raises(ValueError, mod.initialize_known, [0.2, 0.2])

    # Valid steady-state probabilities
    mod.initialize_steady_state()
    assert_allclose(mod.initial_probabilities(params), [0.5, 0.5])

    # Invalid steady-state probabilities (when mod has tvtp)
    endog = np.ones(10)
    mod = markov_switching.MarkovSwitching(endog, k_regimes=2, exog_tvtp=endog)
    assert_raises(ValueError, mod.initialize_steady_state)


def test_logistic():
    logistic = markov_switching._logistic

    # For a number, logistic(x) = np.exp(x) / (1 + np.exp(x))
    cases = [0, 10., -4]
    for x in cases:
        # Have to use allclose b/c logistic() actually uses logsumexp, so
        # they're not equal
        assert_allclose(logistic(x), np.exp(x) / (1 + np.exp(x)))

    # For a vector, logistic(x) returns
    # np.exp(x[i]) / (1 + np.sum(np.exp(x[:]))) for each i
    # but squeezed
    cases = [[1.], [0, 1.], [-2, 3., 1.2, -30.]]
    for x in cases:
        actual = logistic(x)
        desired = [np.exp(i) / (1 + np.sum(np.exp(x))) for i in x]
        assert_allclose(actual, desired)

    # For a 2-dim, logistic(x) returns
    # np.exp(x[i, t]) / (1 + np.sum(np.exp(x[:, t]))) for each i, each t
    # but squeezed
    case = [[1.]]
    actual = logistic(case)
    assert_equal(actual.shape, (1, 1))
    assert_allclose(actual, np.exp(1) / (1 + np.exp(1)))

    # Here, np.array(case) is 2x1, so it is interpreted as i=0, 1 and t=0
    case = [[0], [1.]]
    actual = logistic(case)
    desired = [np.exp(i) / (1 + np.sum(np.exp(case))) for i in case]
    assert_allclose(actual, desired)

    # Here, np.array(case) is 1x2, so it is interpreted as i=0 and t=0, 1
    case = [[0, 1.]]
    actual = logistic(case)
    desired = np.exp(case) / (1 + np.exp(case))
    assert_allclose(actual, desired)

    # For a 3-dim, logistic(x) returns
    # np.exp(x[i, j, t]) / (1 + np.sum(np.exp(x[:, j, t])))
    # for each i, each j, each t
    case = np.arange(2*3*4).reshape(2, 3, 4)
    actual = logistic(case)
    for j in range(3):
        assert_allclose(actual[:, j, :], logistic(case[:, j, :]))


def test_partials_logistic():
    # Here we compare to analytic derivatives and to finite-difference
    # approximations
    logistic = markov_switching._logistic
    partials_logistic = markov_switching._partials_logistic

    # For a number, logistic(x) = np.exp(x) / (1 + np.exp(x))
    # Then d/dx = logistix(x) - logistic(x)**2
    cases = [0, 10., -4]
    for x in cases:
        assert_allclose(partials_logistic(x), logistic(x) - logistic(x)**2)
        assert_allclose(partials_logistic(x), approx_fprime_cs([x], logistic))

    # For a vector, logistic(x) returns
    # np.exp(x[i]) / (1 + np.sum(np.exp(x[:]))) for each i
    # Then d logistic(x[i]) / dx[i] = (logistix(x) - logistic(x)**2)[i]
    # And d logistic(x[i]) / dx[j] = -(logistic(x[i]) * logistic[x[j]])
    cases = [[1.], [0, 1.], [-2, 3., 1.2, -30.]]
    for x in cases:
        evaluated = np.atleast_1d(logistic(x))
        partials = np.diag(evaluated - evaluated**2)
        for i in range(len(x)):
            for j in range(i):
                partials[i, j] = partials[j, i] = -evaluated[i] * evaluated[j]
        assert_allclose(partials_logistic(x), partials)
        assert_allclose(partials_logistic(x), approx_fprime_cs(x, logistic))

    # For a 2-dim, logistic(x) returns
    # np.exp(x[i, t]) / (1 + np.sum(np.exp(x[:, t]))) for each i, each t
    # but squeezed
    case = [[1.]]
    evaluated = logistic(case)
    partial = [evaluated - evaluated**2]
    assert_allclose(partials_logistic(case), partial)
    assert_allclose(partials_logistic(case), approx_fprime_cs(case, logistic))

    # # Here, np.array(case) is 2x1, so it is interpreted as i=0, 1 and t=0
    case = [[0], [1.]]
    evaluated = logistic(case)[:, 0]
    partials = np.diag(evaluated - evaluated**2)
    partials[0, 1] = partials[1, 0] = -np.multiply(*evaluated)
    assert_allclose(partials_logistic(case)[:, :, 0], partials)
    assert_allclose(partials_logistic(case),
                    approx_fprime_cs(np.squeeze(case), logistic)[..., None])

    # Here, np.array(case) is 1x2, so it is interpreted as i=0 and t=0, 1
    case = [[0, 1.]]
    evaluated = logistic(case)
    partials = (evaluated - evaluated**2)[None, ...]
    assert_allclose(partials_logistic(case), partials)
    assert_allclose(partials_logistic(case),
                    approx_fprime_cs(case, logistic).T)

    # For a 3-dim, logistic(x) returns
    # np.exp(x[i, j, t]) / (1 + np.sum(np.exp(x[:, j, t])))
    # for each i, each j, each t
    case = np.arange(2*3*4).reshape(2, 3, 4)
    evaluated = logistic(case)
    partials = partials_logistic(case)
    for t in range(4):
        for j in range(3):
            desired = np.diag(evaluated[:, j, t] - evaluated[:, j, t]**2)
            desired[0, 1] = desired[1, 0] = -np.multiply(*evaluated[:, j, t])
            assert_allclose(partials[..., j, t], desired)
