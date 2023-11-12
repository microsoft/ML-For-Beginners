"""
Tests for initialization

Author: Chad Fulton
License: Simplified-BSD
"""


import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises


def check_initialization(mod, init, a_true, Pinf_true, Pstar_true):
    # Check the Python version
    a, Pinf, Pstar = init(model=mod)
    assert_allclose(a, a_true)
    assert_allclose(Pinf, Pinf_true)
    assert_allclose(Pstar, Pstar_true)

    # Check the Cython version
    mod.ssm._initialize_representation()
    init._initialize_initialization(prefix=mod.ssm.prefix)
    _statespace = mod.ssm._statespace
    _statespace.initialize(init)
    assert_allclose(np.array(_statespace.initial_state), a_true)
    assert_allclose(np.array(_statespace.initial_diffuse_state_cov), Pinf_true)
    assert_allclose(np.array(_statespace.initial_state_cov), Pstar_true)


def test_global_known():
    # Test for global known initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, 'known', constant=[1.5])
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([0]))

    # Known, covariance
    init = Initialization(mod.k_states, 'known', stationary_cov=np.diag([1]))
    check_initialization(mod, init, [0], np.diag([0]), np.diag([1]))

    # Known, both
    init = Initialization(mod.k_states, 'known', constant=[1.5],
                          stationary_cov=np.diag([1]))
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([1]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    # Known, mean
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2])
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]),
                         np.diag([0, 0]))

    # Known, covariance
    init = Initialization(mod.k_states, 'known',
                          stationary_cov=np.diag([1, 4.2]))
    check_initialization(mod, init, [0, 0], np.diag([0, 0]),
                         np.diag([1, 4.2]))

    # Known, both
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2],
                          stationary_cov=np.diag([1, 4.2]))
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]),
                         np.diag([1, 4.2]))


def test_global_diffuse():
    # Test for global diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, 'diffuse')
    check_initialization(mod, init, [0], np.eye(1), np.diag([0]))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, 'diffuse')
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))


def test_global_approximate_diffuse():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))

    init = Initialization(mod.k_states, 'approximate_diffuse')
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse', constant=[1.2])
    check_initialization(mod, init, [1.2], np.diag([0]), np.eye(1) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          approximate_diffuse_variance=1e10)
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 1e10)

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))

    init = Initialization(mod.k_states, 'approximate_diffuse')
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          constant=[1.2, -0.2])
    check_initialization(mod, init, [1.2, -0.2], np.diag([0, 0]),
                         np.eye(2) * 1e6)

    init = Initialization(mod.k_states, 'approximate_diffuse',
                          approximate_diffuse_variance=1e10)
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 1e10)


def test_global_stationary():
    # Test for global approximate diffuse initialization

    # - 1-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend='c')

    # no intercept
    intercept = 0
    phi = 0.5
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    check_initialization(mod, init, [0], np.diag([0]),
                         np.eye(1) * sigma2 / (1 - phi**2))

    # intercept
    intercept = 1.2
    phi = 0.5
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    check_initialization(mod, init, [intercept / (1 - phi)], np.diag([0]),
                         np.eye(1) * sigma2 / (1 - phi**2))

    # - n-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0), trend='c')

    # no intercept
    intercept = 0
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    T = np.array([[0.5, 1],
                  [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), desired_cov)

    # intercept
    intercept = 1.2
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    desired_intercept = np.linalg.inv(np.eye(2) - T).dot([intercept, 0])
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]),
                         desired_cov)


def test_mixed_basic():
    # Performs a number of tests for setting different initialization for
    # different blocks

    # - 2-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[phi, sigma2])

    # known has constant
    init = Initialization(mod.k_states)
    init.set(0, 'known', constant=[1.2])

    # > known has constant
    init.set(1, 'known', constant=[-0.2])
    check_initialization(mod, init, [1.2, -0.2], np.diag([0, 0]),
                         np.diag([0, 0]))

    # > diffuse
    init.unset(1)
    init.set(1, 'diffuse')
    check_initialization(mod, init, [1.2, 0], np.diag([0, 1]), np.diag([0, 0]))

    # > approximate diffuse
    init.unset(1)
    init.set(1, 'approximate_diffuse')
    check_initialization(mod, init, [1.2, 0], np.diag([0, 0]),
                         np.diag([0, 1e6]))

    # > stationary
    init.unset(1)
    init.set(1, 'stationary')
    check_initialization(mod, init, [1.2, 0], np.diag([0, 0]), np.diag([0, 0]))

    # known has cov
    init = Initialization(mod.k_states)
    init.set(0, 'known', stationary_cov=np.diag([1]))
    init.set(1, 'diffuse')
    check_initialization(mod, init, [0, 0], np.diag([0, 1]), np.diag([1, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set(0, 'known', constant=[1.2], stationary_cov=np.diag([1]))
    init.set(1, 'diffuse')
    check_initialization(mod, init, [1.2, 0], np.diag([0, 1]), np.diag([1, 0]))

    # - 3-dimensional -
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(3, 0, 0))

    # known has constant
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', constant=[1.2, -0.2])
    init.set(2, 'diffuse')
    check_initialization(mod, init, [1.2, -0.2, 0], np.diag([0, 0, 1]),
                         np.diag([0, 0, 0]))

    # known has cov
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', stationary_cov=np.diag([1, 4.2]))
    init.set(2, 'diffuse')
    check_initialization(mod, init, [0, 0, 0], np.diag([0, 0, 1]),
                         np.diag([1, 4.2, 0]))

    # known has both
    init = Initialization(mod.k_states)
    init.set((0, 2), 'known', constant=[1.2, -0.2],
             stationary_cov=np.diag([1, 4.2]))
    init.set(2, 'diffuse')
    check_initialization(mod, init, [1.2, -0.2, 0], np.diag([0, 0, 1]),
                         np.diag([1, 4.2, 0]))


def test_mixed_stationary():
    # More specific tests when one or more blocks are initialized as stationary
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 1, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.
    mod.update(np.r_[phi, sigma2])

    init = Initialization(mod.k_states)
    init.set(0, 'diffuse')
    init.set((1, 3), 'stationary')
    desired_cov = np.zeros((3, 3))
    T = np.array([[0.5, 1],
                  [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov[1:, 1:] = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'approximate_diffuse')
    T = np.array([[0.5]])
    Q = np.diag([sigma2])
    desired_cov = np.diag([0, np.squeeze(solve_discrete_lyapunov(T, Q)), 1e6])
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'stationary')
    desired_cov[2, 2] = 0
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)

    # Test with a VAR model
    endog = np.zeros((10, 2))
    mod = varmax.VARMAX(endog, order=(1, 0), )
    intercept = [1.5, -0.1]
    transition = np.array([[0.5, -0.2],
                           [0.1, 0.8]])
    cov = np.array([[1.2, -0.4],
                    [-0.4, 0.4]])
    tril = np.tril_indices(2)
    params = np.r_[intercept, transition.ravel(),
                   np.linalg.cholesky(cov)[tril]]
    mod.update(params)

    # > stationary, global
    init = Initialization(mod.k_states, 'stationary')
    desired_intercept = np.linalg.solve(np.eye(2) - transition, intercept)
    desired_cov = solve_discrete_lyapunov(transition, cov)
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]),
                         desired_cov)

    # > diffuse, global
    init.set(None, 'diffuse')
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))

    # > stationary, individually
    init.unset(None)
    init.set(0, 'stationary')
    init.set(1, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    desired_intercept = [intercept[0] / (1 - transition[0, 0]),
                         intercept[1] / (1 - transition[1, 1])]
    desired_cov = np.diag([cov[0, 0] / (1 - transition[0, 0]**2),
                           cov[1, 1] / (1 - transition[1, 1]**2)])
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]),
                         desired_cov)


def test_nested():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(6, 0, 0))
    phi = [0.5, -0.2, 0.1, 0.0, 0.1, 0.0]
    sigma2 = 2.
    mod.update(np.r_[phi, sigma2])

    # Create the initialization object as a series of nested objects
    init1_1 = Initialization(3)
    init1_1_1 = Initialization(2, 'stationary')
    init1_1_2 = Initialization(1, 'approximate_diffuse',
                               approximate_diffuse_variance=1e9)
    init1_1.set((0, 2), init1_1_1)
    init1_1.set(2, init1_1_2)

    init1_2 = Initialization(3)
    init1_2_1 = Initialization(1, 'known', constant=[1], stationary_cov=[[2.]])
    init1_2.set(0, init1_2_1)
    init1_2_2 = Initialization(1, 'diffuse')
    init1_2.set(1, init1_2_2)
    init1_2_3 = Initialization(1, 'approximate_diffuse')
    init1_2.set(2, init1_2_3)

    init = Initialization(6)
    init.set((0, 3), init1_1)
    init.set((3, 6), init1_2)

    # Check the output
    desired_cov = np.zeros((6, 6))
    T = np.array([[0.5, 1],
                  [-0.2, 0]])
    Q = np.array([[sigma2, 0],
                  [0, 0]])
    desired_cov[:2, :2] = solve_discrete_lyapunov(T, Q)
    desired_cov[2, 2] = 1e9
    desired_cov[3, 3] = 2.
    desired_cov[5, 5] = 1e6
    check_initialization(mod, init, [0, 0, 0, 1, 0, 0],
                         np.diag([0, 0, 0, 0, 1, 0]),
                         desired_cov)


def test_invalid():
    # Invalid initializations (also tests for some invalid calls to set)
    assert_raises(ValueError, Initialization, 5, '')
    assert_raises(ValueError, Initialization, 5, 'stationary', constant=[1, 2])
    assert_raises(ValueError, Initialization, 5, 'stationary',
                  stationary_cov=[1, 2])
    assert_raises(ValueError, Initialization, 5, 'known')
    assert_raises(ValueError, Initialization, 5, 'known', constant=[1])
    assert_raises(ValueError, Initialization, 5, 'known', stationary_cov=[0])

    # Invalid set() / unset() calls
    init = Initialization(5)
    assert_raises(ValueError, init.set, -1, 'diffuse')
    assert_raises(ValueError, init.unset, -1)
    assert_raises(ValueError, init.set, 5, 'diffuse')
    assert_raises(ValueError, init.set, 'x', 'diffuse')
    assert_raises(ValueError, init.unset, 'x')
    assert_raises(ValueError, init.set, (1, 2, 3), 'diffuse')
    assert_raises(ValueError, init.unset, (1, 2, 3))
    init.set(None, 'diffuse')
    assert_raises(ValueError, init.set, 1, 'diffuse')
    init.clear()
    init.set(1, 'diffuse')
    assert_raises(ValueError, init.set, None, 'stationary')

    init.clear()
    assert_raises(ValueError, init.unset, 1)

    # Invalid __call__
    init = Initialization(2)
    assert_raises(ValueError, init)
    init = Initialization(2, 'stationary')
    assert_raises(ValueError, init)
