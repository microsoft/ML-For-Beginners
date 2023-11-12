"""
Basic tests for the BM model

These test standard that the state space model is set up as desired, that
expected exceptions are raised, etc.
"""
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
    dynamic_factor,
    dynamic_factor_mq,
    sarimax,
)
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo


SKIP_MONTE_CARLO_TESTS = True


def test_default():
    # Includes 2 monthly and 2 quarterly series
    # Default is factors=1, factor_orders=1, idiosyncratic_ar1=True

    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=1, factor_orders=1,
        idiosyncratic_ar1=True)

    # Test dimensions
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 5 + 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 1 + 2 + 2)

    # Test names
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = (['0', 'L1.0', 'L2.0', 'L3.0', 'L4.0'] +
               ['eps_M.M0', 'eps_M.M1', 'eps_Q.Q0', 'eps_Q.Q1'] +
               ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] +
               ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] +
               ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] +
               ['L4.eps_Q.Q0', 'L4.eps_Q.Q1'])
    assert_equal(mod.state_names, desired)
    desired = [
        'loading.0->M0', 'loading.0->M1', 'loading.0->Q0',
        'loading.0->Q1',
        'L1.0->0', 'fb(0).cov.chol[1,1]',
        'L1.eps_M.M0', 'L1.eps_M.M1',
        'L1.eps_Q.Q0', 'L1.eps_Q.Q1',
        'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)

    # Test fixed elements of state space representation
    assert_allclose(mod['obs_intercept'], 0)

    assert_allclose(mod['design', :2, 5:7], np.eye(2))
    assert_allclose(mod['design', 2:, 7:9], np.eye(2))
    assert_allclose(mod['design', 2:, 9:11], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 11:13], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 13:15], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 15:17], np.eye(2))
    assert_allclose(np.sum(mod['design']), 20)

    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)

    assert_allclose(mod['transition', 1:5, :4], np.eye(4))
    assert_allclose(mod['transition', 9:17, 7:15], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 12)

    assert_allclose(mod['selection', 0, 0], np.eye(1))
    assert_allclose(mod['selection', 5:7, 1:3], np.eye(2))
    assert_allclose(mod['selection', 7:9, 3:5], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 5)

    assert_allclose(mod['state_cov'], 0)

    # Test parameter entry
    mod.update(np.arange(mod.k_params) + 2)

    # -> obs_intercept
    assert_allclose(mod['obs_intercept'], 0)

    # -> design
    desired = np.array([
        [2., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [3., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [4., 8., 12, 8., 4., 0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1., 0.],
        [5., 10, 15, 10, 5., 0., 0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1.]])
    assert_allclose(mod['design'], desired)

    # -> obs_cov
    assert_allclose(mod['obs_cov'], 0)

    # -> state_intercept
    assert_allclose(mod['state_intercept'], 0)

    # -> transition
    desired = np.array([
        [6., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 8., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 9., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 11., 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1., 0, 0]])
    assert_allclose(mod['transition'], desired)

    # -> selection
    assert_allclose(np.sum(mod['selection']), 5)

    # -> state_cov
    desired = np.array([[49., 0., 0., 0., 0.],
                        [0., 12., 0., 0., 0.],
                        [0., 0., 13., 0., 0.],
                        [0., 0., 0., 14., 0.],
                        [0., 0., 0., 0., 15.]])
    assert_allclose(mod['state_cov'], desired)


def test_k_factors_gt1():
    # Includes 2 monthly and 2 quarterly series
    # This case: k_factors=2, factor_orders=1, idiosyncratic_ar1=True

    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=2, factor_orders={('0', '1'): 1},
        idiosyncratic_ar1=True)

    # Test dimensions
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 5 * 2 + 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 2 + 2 + 2)

    # Test names
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = (['0', '1', 'L1.0', 'L1.1', 'L2.0', 'L2.1',
                'L3.0', 'L3.1', 'L4.0', 'L4.1'] +
               ['eps_M.M0', 'eps_M.M1', 'eps_Q.Q0', 'eps_Q.Q1'] +
               ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] +
               ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] +
               ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] +
               ['L4.eps_Q.Q0', 'L4.eps_Q.Q1'])
    assert_equal(mod.state_names, desired)
    desired = [
        'loading.0->M0', 'loading.1->M0', 'loading.0->M1',
        'loading.1->M1',
        'loading.0->Q0', 'loading.1->Q0', 'loading.0->Q1',
        'loading.1->Q1',
        'L1.0->0', 'L1.1->0', 'L1.0->1', 'L1.1->1',
        'fb(0).cov.chol[1,1]', 'fb(0).cov.chol[2,1]', 'fb(0).cov.chol[2,2]',
        'L1.eps_M.M0', 'L1.eps_M.M1',
        'L1.eps_Q.Q0', 'L1.eps_Q.Q1',
        'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)

    # Test fixed elements of state space representation
    assert_allclose(mod['obs_intercept'], 0)

    assert_allclose(mod['design', :2, 10:12], np.eye(2))
    assert_allclose(mod['design', 2:, 12:14], np.eye(2))
    assert_allclose(mod['design', 2:, 14:16], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 16:18], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 18:20], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 20:22], np.eye(2))
    assert_allclose(np.sum(mod['design']), 20)

    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)

    assert_allclose(mod['transition', 2:10, :8], np.eye(8))
    assert_allclose(mod['transition', 14:22, 12:20], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 16)

    assert_allclose(mod['selection', :2, :2], np.eye(2))
    assert_allclose(mod['selection', 10:12, 2:4], np.eye(2))
    assert_allclose(mod['selection', 12:14, 4:6], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 6)

    assert_allclose(mod['state_cov'], 0)

    # Test parameter entry
    mod.update(np.arange(mod.k_params) + 2)

    # -> obs_intercept
    assert_allclose(mod['obs_intercept'], 0)

    # -> design
    desired = np.array([
        [2., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
        [4., 5., 0., 0., 0., 0., 0., 0., 0., 0.],
        [6., 7., 12, 14, 18, 21, 12, 14, 6., 7.],
        [8., 9., 16, 18, 24, 27, 16, 18, 8., 9.]])
    assert_allclose(mod['design', :, :10], desired)
    desired = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1., 0.],
        [0., 0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1.]])
    assert_allclose(mod['design', :, 10:], desired)

    # -> obs_cov
    assert_allclose(mod['obs_cov'], 0)

    # -> state_intercept
    assert_allclose(mod['state_intercept'], 0)

    # -> transition
    desired = np.array([
        [10, 11, 0., 0., 0., 0., 0., 0., 0., 0.],
        [12, 13, 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', :10, :10], desired)
    assert_allclose(mod['transition', :10, 10:], 0)
    assert_allclose(mod['transition', 10:, :10], 0)
    desired = np.array([
        [17, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 18, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 19, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 20, 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', 10:, 10:], desired)

    # -> selection
    assert_allclose(np.sum(mod['selection']), 6)

    # -> state_cov
    L = np.array([[14., 0],
                  [15., 16.]])
    desired = np.array([[0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 21, 0., 0., 0.],
                        [0., 0., 0., 22, 0., 0.],
                        [0., 0., 0., 0., 23, 0.],
                        [0., 0., 0., 0., 0., 24]])
    desired[:2, :2] = np.dot(L, L.T)
    assert_allclose(mod['state_cov'], desired)


def test_factor_order_gt1():
    # Includes 2 monthly and 2 quarterly series
    # This case: factors=1, factor_orders=6, idiosyncratic_ar1=True

    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=1, factor_orders=6,
        idiosyncratic_ar1=True)

    # Test dimensions
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 6 + 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 1 + 2 + 2)

    # Test names
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = (['0', 'L1.0', 'L2.0', 'L3.0', 'L4.0',
                'L5.0'] +
               ['eps_M.M0', 'eps_M.M1', 'eps_Q.Q0', 'eps_Q.Q1'] +
               ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] +
               ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] +
               ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] +
               ['L4.eps_Q.Q0', 'L4.eps_Q.Q1'])
    assert_equal(mod.state_names, desired)
    desired = [
        'loading.0->M0', 'loading.0->M1', 'loading.0->Q0',
        'loading.0->Q1',
        'L1.0->0', 'L2.0->0', 'L3.0->0', 'L4.0->0',
        'L5.0->0', 'L6.0->0',
        'fb(0).cov.chol[1,1]',
        'L1.eps_M.M0', 'L1.eps_M.M1',
        'L1.eps_Q.Q0', 'L1.eps_Q.Q1',
        'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)

    # Test fixed elements of state space representation
    assert_allclose(mod['obs_intercept'], 0)

    assert_allclose(mod['design', :2, 6:8], np.eye(2))
    assert_allclose(mod['design', 2:, 8:10], np.eye(2))
    assert_allclose(mod['design', 2:, 10:12], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 12:14], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 14:16], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 16:18], np.eye(2))
    assert_allclose(np.sum(mod['design']), 20)

    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)

    assert_allclose(mod['transition', 1:6, :5], np.eye(5))
    assert_allclose(mod['transition', 10:18, 8:16], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 13)

    assert_allclose(mod['selection', 0, 0], np.eye(1))
    assert_allclose(mod['selection', 6:8, 1:3], np.eye(2))
    assert_allclose(mod['selection', 8:10, 3:5], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 5)

    assert_allclose(mod['state_cov'], 0)

    # Test parameter entry
    mod.update(np.arange(mod.k_params) + 2)

    # -> obs_intercept
    assert_allclose(mod['obs_intercept'], 0)

    # -> design
    desired = np.array([
        [2., 0., 0., 0., 0., 0.],
        [3., 0., 0., 0., 0., 0.],
        [4., 8., 12, 8., 4., 0.],
        [5., 10, 15, 10, 5., 0.]])
    assert_allclose(mod['design', :, :6], desired)
    desired = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1., 0.],
        [0., 0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1.]])
    assert_allclose(mod['design', :, 6:], desired)

    # -> obs_cov
    assert_allclose(mod['obs_cov'], 0)

    # -> state_intercept
    assert_allclose(mod['state_intercept'], 0)

    # -> transition
    desired = np.array([
        [6., 7., 8., 9., 10, 11, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 13, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 14, 0., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 15, 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 16, 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0, 0],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0, 0]
    ])
    assert_allclose(mod['transition'], desired)

    # -> selection
    assert_allclose(np.sum(mod['selection']), 5)

    # -> state_cov
    desired = np.array([[144, 0., 0., 0., 0.],
                        [0., 17., 0., 0., 0.],
                        [0., 0., 18., 0., 0.],
                        [0., 0., 0., 19., 0.],
                        [0., 0., 0., 0., 20.]])
    assert_allclose(mod['state_cov'], desired)


def test_k_factors_gt1_factor_order_gt1():
    # Includes 2 monthly and 2 quarterly series
    # This case: kactors=2, factor_orders=6, idiosyncratic_ar1=True

    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=2, factor_orders={('0', '1'): 6},
        idiosyncratic_ar1=True)

    # Test dimensions
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 6 * 2 + 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 2 + 2 + 2)

    # Test names
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = (['0', '1', 'L1.0', 'L1.1', 'L2.0', 'L2.1',
                'L3.0', 'L3.1', 'L4.0', 'L4.1', 'L5.0',
                'L5.1'] +
               ['eps_M.M0', 'eps_M.M1', 'eps_Q.Q0', 'eps_Q.Q1'] +
               ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] +
               ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] +
               ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] +
               ['L4.eps_Q.Q0', 'L4.eps_Q.Q1'])
    assert_equal(mod.state_names, desired)
    desired = [
        'loading.0->M0', 'loading.1->M0', 'loading.0->M1',
        'loading.1->M1',
        'loading.0->Q0', 'loading.1->Q0', 'loading.0->Q1',
        'loading.1->Q1',
        'L1.0->0', 'L1.1->0', 'L2.0->0', 'L2.1->0',
        'L3.0->0', 'L3.1->0', 'L4.0->0', 'L4.1->0',
        'L5.0->0', 'L5.1->0', 'L6.0->0', 'L6.1->0',
        'L1.0->1', 'L1.1->1', 'L2.0->1', 'L2.1->1',
        'L3.0->1', 'L3.1->1', 'L4.0->1', 'L4.1->1',
        'L5.0->1', 'L5.1->1', 'L6.0->1', 'L6.1->1',
        'fb(0).cov.chol[1,1]', 'fb(0).cov.chol[2,1]', 'fb(0).cov.chol[2,2]',
        'L1.eps_M.M0', 'L1.eps_M.M1',
        'L1.eps_Q.Q0', 'L1.eps_Q.Q1',
        'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)

    # Test fixed elements of state space representation
    assert_allclose(mod['obs_intercept'], 0)

    assert_allclose(mod['design', :2, 12:14], np.eye(2))
    assert_allclose(mod['design', 2:, 14:16], np.eye(2))
    assert_allclose(mod['design', 2:, 16:18], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 18:20], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 20:22], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 22:24], np.eye(2))
    assert_allclose(np.sum(mod['design']), 20)

    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)

    assert_allclose(mod['transition', 2:12, :10], np.eye(10))
    assert_allclose(mod['transition', 16:24, 14:22], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 18)

    assert_allclose(mod['selection', :2, :2], np.eye(2))
    assert_allclose(mod['selection', 12:14, 2:4], np.eye(2))
    assert_allclose(mod['selection', 14:16, 4:6], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 6)

    assert_allclose(mod['state_cov'], 0)

    # Test parameter entry
    mod.update(np.arange(mod.k_params) + 2)

    # -> obs_intercept
    assert_allclose(mod['obs_intercept'], 0)

    # -> design
    desired = np.array([
        [2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [4., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [6., 7., 12, 14, 18, 21, 12, 14, 6., 7., 0., 0.],
        [8., 9., 16, 18, 24, 27, 16, 18, 8., 9., 0., 0.]])
    assert_allclose(mod['design', :, :12], desired)
    desired = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1., 0.],
        [0., 0., 0., 1., 0., 2., 0., 3., 0., 2., 0., 1.]])
    assert_allclose(mod['design', :, 12:], desired)

    # -> obs_cov
    assert_allclose(mod['obs_cov'], 0)

    # -> state_intercept
    assert_allclose(mod['state_intercept'], 0)

    # -> transition
    desired = np.array([
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', :12, :12], desired)
    assert_allclose(mod['transition', :12, 12:], 0)
    assert_allclose(mod['transition', 12:, :12], 0)
    desired = np.array([
        [37, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 38, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 39, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 40, 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', 12:, 12:], desired)

    # -> selection
    assert_allclose(np.sum(mod['selection']), 6)

    # -> state_cov
    L = np.array([[34., 0],
                  [35., 36.]])
    desired = np.array([[0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 41, 0., 0., 0.],
                        [0., 0., 0., 42, 0., 0.],
                        [0., 0., 0., 0., 43, 0.],
                        [0., 0., 0., 0., 0., 44]])
    desired[:2, :2] = np.dot(L, L.T)
    assert_allclose(mod['state_cov'], desired)


def test_k_factors_gt1_factor_order_gt1_no_idiosyncratic_ar1():
    # Includes 2 monthly and 2 quarterly series
    # This case: factors=2, factor_orders=6, idiosyncratic_ar1=False

    # Create the datasets
    index_M = pd.period_range(start='2000', periods=12, freq='M')
    index_Q = pd.period_range(start='2000', periods=4, freq='Q')

    dta_M = pd.DataFrame(np.zeros((12, 2)), index=index_M,
                         columns=['M0', 'M1'])
    dta_Q = pd.DataFrame(np.zeros((4, 2)), index=index_Q, columns=['Q0', 'Q1'])
    # Add some noise so the variables aren't constants
    dta_M.iloc[0] = 1.
    dta_Q.iloc[1] = 1.

    # Create the model instance
    mod = dynamic_factor_mq.DynamicFactorMQ(
        dta_M, endog_quarterly=dta_Q, factors=2, factor_orders={('0', '1'): 6},
        idiosyncratic_ar1=False)

    # Test dimensions
    assert_equal(mod.k_endog, 2 + 2)
    assert_equal(mod.k_states, 6 * 2 + 2 * 5)
    assert_equal(mod.ssm.k_posdef, 2 + 2)

    # Test names
    assert_equal(mod.endog_names, ['M0', 'M1', 'Q0', 'Q1'])
    desired = (['0', '1', 'L1.0', 'L1.1', 'L2.0', 'L2.1',
                'L3.0', 'L3.1', 'L4.0', 'L4.1', 'L5.0',
                'L5.1'] +
               ['eps_Q.Q0', 'eps_Q.Q1'] +
               ['L1.eps_Q.Q0', 'L1.eps_Q.Q1'] +
               ['L2.eps_Q.Q0', 'L2.eps_Q.Q1'] +
               ['L3.eps_Q.Q0', 'L3.eps_Q.Q1'] +
               ['L4.eps_Q.Q0', 'L4.eps_Q.Q1'])
    assert_equal(mod.state_names, desired)
    desired = [
        'loading.0->M0', 'loading.1->M0', 'loading.0->M1',
        'loading.1->M1',
        'loading.0->Q0', 'loading.1->Q0', 'loading.0->Q1',
        'loading.1->Q1',
        'L1.0->0', 'L1.1->0', 'L2.0->0', 'L2.1->0',
        'L3.0->0', 'L3.1->0', 'L4.0->0', 'L4.1->0',
        'L5.0->0', 'L5.1->0', 'L6.0->0', 'L6.1->0',
        'L1.0->1', 'L1.1->1', 'L2.0->1', 'L2.1->1',
        'L3.0->1', 'L3.1->1', 'L4.0->1', 'L4.1->1',
        'L5.0->1', 'L5.1->1', 'L6.0->1', 'L6.1->1',
        'fb(0).cov.chol[1,1]', 'fb(0).cov.chol[2,1]', 'fb(0).cov.chol[2,2]',
        'sigma2.M0', 'sigma2.M1', 'sigma2.Q0', 'sigma2.Q1']
    assert_equal(mod.param_names, desired)

    # Test fixed elements of state space representation
    assert_allclose(mod['obs_intercept'], 0)

    assert_allclose(mod['design', 2:, 12:14], np.eye(2))
    assert_allclose(mod['design', 2:, 14:16], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 16:18], 3 * np.eye(2))
    assert_allclose(mod['design', 2:, 18:20], 2 * np.eye(2))
    assert_allclose(mod['design', 2:, 20:22], np.eye(2))
    assert_allclose(np.sum(mod['design']), 18)

    assert_allclose(mod['obs_cov'], 0)
    assert_allclose(mod['state_intercept'], 0)

    assert_allclose(mod['transition', 2:12, :10], np.eye(10))
    assert_allclose(mod['transition', 14:22, 12:20], np.eye(2 * 4))
    assert_allclose(np.sum(mod['transition']), 18)

    assert_allclose(mod['selection', :2, :2], np.eye(2))
    assert_allclose(mod['selection', 12:14, 2:4], np.eye(2))
    assert_allclose(np.sum(mod['selection']), 4)

    assert_allclose(mod['state_cov'], 0)

    # Test parameter entry
    mod.update(np.arange(mod.k_params) + 2)

    # -> obs_intercept
    assert_allclose(mod['obs_intercept'], 0)

    # -> design
    desired = np.array([
        [2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [4., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [6., 7., 12, 14, 18, 21, 12, 14, 6., 7., 0., 0.],
        [8., 9., 16, 18, 24, 27, 16, 18, 8., 9., 0., 0.]])
    assert_allclose(mod['design', :, :12], desired)
    desired = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 2., 0., 3., 0., 2., 0., 1., 0.],
        [0., 1., 0., 2., 0., 3., 0., 2., 0., 1.]])
    assert_allclose(mod['design', :, 12:], desired)

    # -> obs_cov
    assert_allclose(mod['obs_cov'], np.diag([37, 38, 0, 0]))

    # -> state_intercept
    assert_allclose(mod['state_intercept'], 0)

    # -> transition
    desired = np.array([
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', :12, :12], desired)
    assert_allclose(mod['transition', :12, 12:], 0)
    assert_allclose(mod['transition', 12:, :12], 0)
    desired = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    assert_allclose(mod['transition', 12:, 12:], desired)

    # -> selection
    assert_allclose(np.sum(mod['selection']), 4)

    # -> state_cov
    L = np.array([[34., 0],
                  [35., 36.]])
    desired = np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 39, 0.],
                        [0., 0., 0., 40]])
    desired[:2, :2] = np.dot(L, L.T)
    assert_allclose(mod['state_cov'], desired)


def test_invalid_model_specification():
    # Test for errors that can be raised for invalid model specifications
    # during __init__
    dta = np.zeros((10, 2))
    dta[0] = 1.
    dta_pd = pd.DataFrame(dta)
    dta_period_W = pd.DataFrame(
        dta, index=pd.period_range(start='2000', periods=10, freq='W'))
    dta_date_W = pd.DataFrame(
        dta, index=pd.date_range(start='2000', periods=10, freq='W'))
    dta_period_M = pd.DataFrame(
        dta, index=pd.period_range(start='2000', periods=10, freq='M'))
    dta_date_M = pd.DataFrame(
        dta, index=pd.date_range(start='2000', periods=10, freq='M'))
    dta_period_Q = pd.DataFrame(
        dta, index=pd.period_range(start='2000', periods=10, freq='Q'))

    # Error if k_factors == 0
    msg = 'The model must contain at least one factor.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=0)

    # Error if k_factors, factor_multiplicities, or factor_orders is something
    # besides int or dict
    msg = ('`factors` argument must an integer number of factors, a list of'
           ' global factor names, or a dictionary, mapping observed variables'
           ' to factors.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=True)
    msg = '`factor_orders` argument must either be an integer or a dictionary.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factor_orders=True)
    msg = ('`factor_multiplicities` argument must either be an integer or a'
           ' dictionary.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factor_multiplicities=True)

    # Error if k_factors > k_endog_M
    msg = fr'Number of factors \({dta.shape[1] + 1}\) cannot be greater than'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=dta.shape[1] + 1)

    # Error if factor assigned to more than one block
    factor_orders = {('a', 'b'): 1, 'b': 2}
    msg = ('Each factor can be assigned to at most one block of factors in'
           ' `factor_orders`.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors=['a', 'b'],
                                          factor_orders=factor_orders)

    # Error if k_endog_monthly and endog_quarterly both specified
    msg = ('If `endog_quarterly` is specified, then `endog` must contain only'
           ' monthly variables, and so `k_endog_monthly` cannot be specified'
           ' since it will be inferred from the shape of `endog`.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, k_endog_monthly=2,
                                          endog_quarterly=dta)

    # Error if invalid standardize
    msg = 'Invalid value passed for `standardize`.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, standardize='a')

    # No factors for one endog
    msg = ('If a `factors` dictionary is provided, then it must include'
           ' entries for each observed variable.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, factors={'y1': ['a']})
    msg = ('Each observed variable must be mapped to at'
           ' least one factor in the `factors` dictionary.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(
            dta, factors={'y1': ['a'], 'y2': []})

    # Singular column of data + standardization
    msg = (r'Constant variable\(s\) found in observed variables, but constants'
           ' cannot be included in this model.')
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta * 0)

    # Non-pandas data when given both monthly and quarterly
    msg = 'Given monthly dataset is not a Pandas object.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta, endog_quarterly=dta)
    msg = 'Given quarterly dataset is not a Pandas object.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, endog_quarterly=dta)

    # Pandas data without date index when given both monthly and quarterly
    msg = 'Given monthly dataset has an index with non-date values.'
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_pd, endog_quarterly=dta_period_Q)
    msg = 'Given quarterly dataset has an index with non-date values.'
    # (test once with period index for monthly...)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_period_M, endog_quarterly=dta_pd)
    # (...and once with date index for monthly)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(dta_date_M, endog_quarterly=dta_pd)

    # Pandas data with date index of wrong freq
    msg = 'Index of given monthly dataset has a non-monthly frequency'
    # (test once with period index for monthly...)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(
            dta_period_W, endog_quarterly=dta_period_Q)
    # (...and once with date index for monthly)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(
            dta_date_W, endog_quarterly=dta_period_Q)
    msg = 'Index of given quarterly dataset has a non-quarterly frequency'
    # (test once with period index for quarterly...)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(
            dta_period_M, endog_quarterly=dta_period_W)
    # (and once with date index for quarterly...)
    with pytest.raises(ValueError, match=msg):
        dynamic_factor_mq.DynamicFactorMQ(
            dta_date_M, endog_quarterly=dta_date_W)


@pytest.mark.parametrize('freq_Q', ['Q', 'Q-DEC', 'Q-JAN', 'QS',
                                    'QS-DEC', 'QS-APR'])
@pytest.mark.parametrize('freq_M', ['M', 'MS'])
def test_date_indexes(reset_randomstate, freq_M, freq_Q):
    # Test that using either PeriodIndex or DatetimeIndex for monthly or
    # quarterly data, with a variety of DatetimeIndex frequencies, works

    # Monthly datasets
    nobs_M = 10
    dates_M = pd.date_range(start='2000', periods=nobs_M, freq=freq_M)
    periods_M = pd.period_range(start='2000', periods=nobs_M, freq='M')
    dta_M = np.random.normal(size=(nobs_M, 2))
    endog_period_M = pd.DataFrame(dta_M.copy(), index=periods_M)
    endog_date_M = pd.DataFrame(dta_M.copy(), index=dates_M)

    # Quarterly datasets
    nobs_Q = 3
    dates_Q = pd.date_range(start='2000', periods=nobs_Q, freq=freq_Q)
    periods_Q = pd.period_range(start='2000', periods=nobs_Q, freq='Q')
    dta_Q = np.random.normal(size=(nobs_Q, 2))
    endog_period_Q = pd.DataFrame(dta_Q.copy(), index=periods_Q)
    endog_date_Q = pd.DataFrame(dta_Q.copy(), index=dates_Q)

    # Baseline is monthly periods and quarterly periods
    mod_base = dynamic_factor_mq.DynamicFactorMQ(
        endog_period_M, endog_quarterly=endog_period_Q)

    # Test against monthly dates and quarterly dates
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog_date_M, endog_quarterly=endog_date_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)

    # Test against monthly dates and quarterly periods
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog_date_M, endog_quarterly=endog_period_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)

    # Test against monthly periods and quarterly dates
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog_period_M, endog_quarterly=endog_date_Q)
    assert_(mod._index.equals(mod_base._index))
    assert_allclose(mod.endog, mod_base.endog)


def gen_dfm_data(k_endog=2, nobs=1000):
    if k_endog > 10:
        raise ValueError('Only allows for k_endog <= 10')
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.DataFrame([[0] * k_endog], index=ix)
    mod = dynamic_factor.DynamicFactor(faux, k_factors=1, factor_order=1)
    loadings = [0.5, -0.9, 0.2, 0.7, -0.1, -0.1, 0.4, 0.4, 0.8, 0.8][:k_endog]
    phi = 0.5
    sigma2 = 1.0
    idio_ar1 = [0] * k_endog
    idio_var = [1.0, 0.2, 1.5, 0.8, 0.8, 1.4, 0.1, 0.2, 0.4, 0.5][:k_endog]
    params = np.r_[loadings, idio_var, phi]
    endog = mod.simulate(params, nobs)
    return endog, loadings, phi, sigma2, idio_ar1, idio_var


def test_results_factors(reset_randomstate):
    # Tests for the `factors` attribute in the results object
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=2, nobs=1000)

    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=['global'], factor_multiplicities=2,
        standardize=False, idiosyncratic_ar1=False)
    res_dfm = mod_dfm.smooth(mod_dfm.start_params)

    assert_allclose(res_dfm.factors.smoothed,
                    res_dfm.states.smoothed[['global.1', 'global.2']])
    assert_allclose(res_dfm.factors.smoothed_cov.values,
                    res_dfm.states.smoothed_cov.values, atol=1e-12)


def test_coefficient_of_determination(
        reset_randomstate, close_figures):
    # Get simulated data, and add in some missing entries
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=3, nobs=1000)
    endog.iloc[0, 10:20] = np.nan
    endog.iloc[2, 15:25] = np.nan

    # Setup the model and get the results
    factors = {
        0: ['global', 'block'],
        1: ['global', 'block'],
        2: ['global']
    }
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=factors, standardize=False, idiosyncratic_ar1=False)
    res = mod.smooth(mod.start_params)

    # For most of the tests, we'll use smoothed estimates, which are the
    # default
    factors = res.factors.smoothed

    # Test for method='individual'
    actual = res.get_coefficients_of_determination(method='individual')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2],
                           columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, j])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))
    assert_allclose(actual, desired)

    # Test for method='joint'
    actual = res.get_coefficients_of_determination(method='joint')
    desired = pd.Series(np.zeros(3), index=[0, 1, 2])
    for i in range(3):
        y = endog.iloc[:, i]
        if i == 2:
            X = add_constant(factors.iloc[:, 0])
        else:
            X = add_constant(factors)
        mod_ols = OLS(y, X, missing='drop')
        res_ols = mod_ols.fit()
        desired.iloc[i] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_allclose(actual, desired)

    # Test for method='cumulative'
    actual = res.get_coefficients_of_determination(method='cumulative')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2],
                           columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, :j + 1])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))
    assert_allclose(actual, desired)

    # Test for method='individual', which='filtered'
    factors = res.factors.filtered
    actual = res.get_coefficients_of_determination(
        method='individual', which='filtered')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2],
                           columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, j])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_allclose(actual, desired)

    # Optional smoke test for plot_coefficient_of_determination
    try:
        import matplotlib.pyplot as plt
        try:
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
        except ImportError:
            pass
        fig1 = plt.figure()
        res.plot_coefficients_of_determination(method='individual', fig=fig1)
        fig2 = plt.figure()
        res.plot_coefficients_of_determination(method='joint', fig=fig2)
        fig3 = plt.figure()
        res.plot_coefficients_of_determination(method='cumulative', fig=fig3)
        fig4 = plt.figure()
        res.plot_coefficients_of_determination(which='filtered', fig=fig4)
    except ImportError:
        pass


@pytest.mark.filterwarnings("ignore:Log-likelihood decreased")
def test_quasi_newton_fitting(reset_randomstate):
    # Test that the typical state space quasi-Newton fitting mechanisms work
    # here too, even if they aren't used much
    # Note: to match the quasi-Newton results, which use the stationary
    # initialization, we need to set em_initialization=False
    # Note: one thing that this test reveals is that there are some numerical
    # issues with the EM algorithm when em_initialization=False around the
    # true optimum, where apparently numerical issues case the EM algorithm to
    # decrease the log-likelihood. In particular, testing updating each block
    # of parameters shows that it is the updating of the factor autoregressive
    # coefficients and (it appears) the factor error variance term that are
    # to blame here.
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=2, nobs=1000)

    # Create the test Dynamic Factor MQ models
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                standardize=False,
                                                idiosyncratic_ar1=False)
    mod_dfm_ar1 = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                    standardize=False,
                                                    idiosyncratic_ar1=True)

    # Check that transform and untransform works
    x = mod_dfm_ar1.start_params
    y = mod_dfm_ar1.untransform_params(x)
    z = mod_dfm_ar1.transform_params(y)
    assert_allclose(x, z)

    # Check lbfgs and em converge to the same thing: idiosyncratic_ar1=False
    res_lbfgs = mod_dfm.fit(method='lbfgs')
    params_lbfgs = res_lbfgs.params.copy()

    start_params = params_lbfgs.copy()
    start_params['L1.0->0'] += 1e-2
    start_params['fb(0).cov.chol[1,1]'] += 1e-2
    res_em = mod_dfm.fit(start_params, em_initialization=False)
    params_em = res_em.params.copy()

    assert_allclose(res_lbfgs.llf, res_em.llf, atol=5e-2, rtol=1e-5)
    assert_allclose(params_lbfgs, params_em, atol=5e-2, rtol=1e-5)

    # Check lbfgs and em converge to the same thing: idiosyncratic_ar1=True
    res_lbfgs = mod_dfm_ar1.fit(method='lbfgs')
    params_lbfgs = res_lbfgs.params.copy()

    start_params = params_lbfgs.copy()
    start_params['L1.0->0'] += 1e-2
    start_params['fb(0).cov.chol[1,1]'] += 1e-2
    res_em = mod_dfm_ar1.fit(params_lbfgs, em_initialization=False)
    params_em = res_em.params.copy()

    assert_allclose(res_lbfgs.llf, res_em.llf, atol=5e-2, rtol=1e-5)
    assert_allclose(params_lbfgs, params_em, atol=5e-2, rtol=1e-5)


def test_summary(reset_randomstate):
    # Smoke tests for summaries
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=10, nobs=100)

    # Create the test Dynamic Factor MQ models
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                standardize=False,
                                                idiosyncratic_ar1=False)
    res_dfm = mod_dfm.smooth(mod_dfm.start_params)
    mod_dfm_ar1 = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                    standardize=False,
                                                    idiosyncratic_ar1=True)
    res_dfm_ar1 = mod_dfm_ar1.smooth(mod_dfm_ar1.start_params)

    mod_dfm.summary()
    assert_equal(str(mod_dfm), str(mod_dfm.summary()))
    res_dfm.summary()
    mod_dfm_ar1.summary()
    res_dfm_ar1.summary()


def test_append_extend_apply(reset_randomstate):
    # Most of `append`, `extend`, and `apply` are tested in
    # `test_standardized_{monthly,MQ}`, but here we test a couple of minor
    # things
    endog, loadings, phi, sigma2, _, idio_var = (
        gen_dfm_data(k_endog=10, nobs=100))
    endog1 = endog.iloc[:-10]
    endog2 = endog.iloc[-10:]
    mod = dynamic_factor_mq.DynamicFactorMQ(endog1, factor_orders=1,
                                            standardize=False,
                                            idiosyncratic_ar1=False)
    params = np.r_[loadings, phi, sigma2, idio_var]
    res = mod.smooth(params)

    # Test that error is raised if we try to extend the sample with a dataset
    # of a different dimension
    msg = 'Cannot append data of a different dimension to a model.'
    with pytest.raises(ValueError, match=msg):
        res.append(endog2.iloc[:, :3])
    with pytest.raises(ValueError, match=msg):
        res.extend(endog2.iloc[:, :3])

    # Test that copy_initialization works
    mod.initialize_known([0.1], [[1.0]])
    res2 = mod.smooth(params)
    # These asserts just show that the initialization really is different
    assert_allclose(res.filter_results.initial_state, 0)
    assert_allclose(res.filter_results.initial_state_cov, 4 / 3.)
    assert_allclose(res2.filter_results.initial_state, 0.1)
    assert_allclose(res2.filter_results.initial_state_cov, 1.0)

    # Check append
    res3 = res2.append(endog2, copy_initialization=False)
    assert_allclose(res3.filter_results.initial_state, 0)
    assert_allclose(res3.filter_results.initial_state_cov, 4 / 3.)

    res4 = res2.append(endog2, copy_initialization=True)
    assert_allclose(res4.filter_results.initial_state, 0.1)
    assert_allclose(res4.filter_results.initial_state_cov, 1.0)

    # Check apply
    res5 = res2.apply(endog, copy_initialization=False)
    assert_allclose(res5.filter_results.initial_state, 0)
    assert_allclose(res5.filter_results.initial_state_cov, 4 / 3.)

    res6 = res2.apply(endog, copy_initialization=True)
    assert_allclose(res6.filter_results.initial_state, 0.1)
    assert_allclose(res6.filter_results.initial_state_cov, 1.0)


def test_news_monthly(reset_randomstate):
    # Most of `news` is tested in `test_standardized_{monthly,MQ}`, but here
    # we test a couple of minor things
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=10, nobs=100)
    endog_pre = endog.iloc[:-1].copy()
    endog_pre.iloc[-1, 0] *= 1.2
    endog_pre.iloc[-1, 1] = np.nan

    # Base model
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_pre, factor_orders=1,
                                            standardize=False,
                                            idiosyncratic_ar1=False)
    params = mod.start_params
    res = mod.smooth(params)

    # Updated results and desired news output (created using a results object)
    mod2 = mod.clone(endog)
    res2 = mod2.smooth(params)
    desired = res2.news(res, start=endog.index[-1], periods=1,
                        comparison_type='previous')

    # Actual news output, created using the updated dataset
    actual = res.news(endog, start=endog.index[-1], periods=1,
                      comparison_type='updated')

    attributes = [
        'total_impacts', 'update_impacts', 'revision_impacts', 'news',
        'weights', 'update_forecasts', 'update_realized',
        'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc',
        'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attributes:
        w = getattr(actual, attr)
        x = getattr(desired, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


def test_news_MQ(reset_randomstate):
    # Most of `news` is tested in `test_standardized_{monthly,MQ}`, but here
    # we test a couple of minor things
    endog_M, endog_Q, f1 = test_dynamic_factor_mq_monte_carlo.gen_k_factor1(
        100, k=2, idiosyncratic_ar1=False)
    endog_M_pre = endog_M.iloc[:-1].copy()
    endog_M_pre.iloc[-1, 0] *= 1.2
    endog_M_pre.iloc[-1, 1] = np.nan

    endog_Q_pre = endog_Q.iloc[:-1].copy()
    endog_Q_pre.iloc[-1, 0] *= 1.2
    endog_Q_pre.iloc[-1, 1] = np.nan

    # Base model
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog_M_pre, endog_quarterly=endog_Q_pre, factor_orders=1,
        standardize=False, idiosyncratic_ar1=False)
    params = mod.start_params
    res = mod.smooth(params)

    # Updated results and desired news output (created using a results object)
    mod2 = mod.clone(endog_M, endog_quarterly=endog_Q)
    res2 = mod2.smooth(params)
    desired = res2.news(res, start=endog_M.index[-1], periods=1,
                        comparison_type='previous')

    # Actual news output, created using the updated dataset
    actual = res.news(endog_M, endog_quarterly=endog_Q,
                      start=endog_M.index[-1], periods=1,
                      comparison_type='updated')

    attributes = [
        'total_impacts', 'update_impacts', 'revision_impacts', 'news',
        'weights', 'update_forecasts', 'update_realized',
        'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc',
        'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attributes:
        w = getattr(actual, attr)
        x = getattr(desired, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


def test_ar6_no_quarterly(reset_randomstate):
    # Test to make sure that an example with > 5 lags (which are the number of
    # lags that are always included in models with quarterly data) works,
    # for the case without quarterly data

    # Generate test data
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.Series([0], index=ix)
    mod = sarimax.SARIMAX(faux, order=(6, 0, 0))
    params = np.r_[0., 0., 0., 0., 0., 0.5, 1.0]
    endog = mod.simulate(params, 100)

    # Create the baseline SARIMAX and the test Dynamic Factor MQ model
    mod_ar = sarimax.SARIMAX(endog, order=(6, 0, 0))
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=6,
                                                standardize=False,
                                                idiosyncratic_ar1=False)

    # Test that SARIMAX and DFM MQ produce the same loglike
    llf_ar = mod_ar.loglike(params)
    llf_dfm = mod_dfm.loglike(np.r_[1, params, 0.])
    assert_allclose(llf_dfm, llf_ar)

    # Monte Carlo-type test, skipped by default
    if not SKIP_MONTE_CARLO_TESTS:
        # Test for MLE fitting: this is a Monte Carlo test, which requires a
        # large sample size; e.g., use nobs=10000 for atol=1e-2
        res_dfm = mod_dfm.fit()
        actual = res_dfm.params
        # normalize loading = 1, which requires us to multiply std. dev. of
        # factor (which is `actual[-2]`) by the estimated loading (`actual[0]`)
        actual[-2] *= actual[0]
        actual[0] = 1
        assert_allclose(res_dfm.params[1:-1], params, atol=1e-2)


def test_idiosyncratic_ar1_False(reset_randomstate):
    # Test the case with idiosyncratic_ar1=False (which is not tested in
    # the test_dynamic_factor_mq_frbny_nowcast) by comparison to a model with
    # idiosyncratic_ar1=True but no actual serial correlation in the
    # idiosyncratic component
    endog, loadings, phi, sigma2, idio_ar1, idio_var = gen_dfm_data(
        k_endog=10, nobs=1000)

    # Create the baseline SARIMAX and the test Dynamic Factor MQ model
    mod_base = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    mod_dfm = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                standardize=False,
                                                idiosyncratic_ar1=False)
    mod_dfm_ar1 = dynamic_factor_mq.DynamicFactorMQ(endog, factor_orders=1,
                                                    standardize=False,
                                                    idiosyncratic_ar1=True)

    params = np.r_[loadings, idio_var, phi]
    params_dfm = np.r_[loadings, phi, sigma2, idio_var]
    params_dfm_ar1 = np.r_[loadings, phi, sigma2, idio_ar1, idio_var]

    # Test that these models all produce the same loglikelihood
    llf_base = mod_base.loglike(params)
    llf_dfm = mod_dfm.loglike(params_dfm)
    llf_dfm_ar1 = mod_dfm_ar1.loglike(params_dfm_ar1)

    assert_allclose(llf_dfm_ar1, llf_dfm)
    assert_allclose(llf_dfm, llf_base)
    assert_allclose(llf_dfm_ar1, llf_base)

    # Test that these methods produce the same smoothed estimates of the
    # idiosyncratic disturbance
    res0_dfm = mod_dfm.smooth(params_dfm)
    res0_dfm_ar1 = mod_dfm_ar1.smooth(params_dfm_ar1)
    assert_allclose(res0_dfm.smoothed_measurement_disturbance,
                    res0_dfm_ar1.smoothed_state[1:])
    assert_allclose(res0_dfm.smoothed_measurement_disturbance_cov,
                    res0_dfm_ar1.smoothed_state_cov[1:, 1:, :])

    # Note: it is difficult to test the EM algorithm for the idiosyncratic
    # variance terms, because actually the EM algorithm is slightly different
    # depending on whether idiosyncratic_ar1 is True or False.
    # - If idiosyncratic_ar1=False, then the variance term is computed
    #   conditional on the *updated estimate for the design matrix*, but the
    #   smoothed moments that were generated conditional on the *previous*
    #   estimate for the design matrix.
    # - If idiosyntratic_ar1=True, then the variance term is computed
    #   conditional on the *previous estimate for the design matrix* (this is
    #   because the estimated disturbance is part of the state vector, the
    #   estimates of which are not modified to account for the updated design
    #   matrix within this iteration)
    #
    # So instead, we'll just use a Monte Carlo-type test

    # Monte Carlo-type test, skipped by default
    if not SKIP_MONTE_CARLO_TESTS:
        # Test for MLE fitting: this is a Monte Carlo test, which requires a
        # quite large sample size; e.g., use nobs=50000 for atol=1e-1
        res_dfm = mod_dfm.fit()
        actual_dfm = res_dfm.params.copy()
        # normalize loadings and factor std dev.
        scalar = actual_dfm[0] / params_dfm[0]
        actual_dfm[11] *= scalar
        actual_dfm[:10] /= scalar
        assert_allclose(actual_dfm, params_dfm, atol=1e-1)

        res_dfm_ar1 = mod_dfm_ar1.fit()
        actual_dfm_ar1 = res_dfm_ar1.params.copy()
        # normalize loadings and factor std dev.
        scalar = actual_dfm_ar1[0] / params_dfm[0]
        actual_dfm_ar1[11] *= scalar
        actual_dfm_ar1[:10] /= scalar
        assert_allclose(actual_dfm_ar1, params_dfm_ar1, atol=1e-1)

        # Check the methods against each other
        desired = np.r_[actual_dfm_ar1[:12], actual_dfm_ar1[-10:]]
        assert_allclose(actual_dfm, desired, atol=1e-1)


def test_invalid_standardize_1d():
    endog = np.zeros(100) + 10
    endog_pd = pd.Series(endog, name='y1')

    # Wrong shape
    options = [([], 10), (10, []), ([], []), ([1, 2], [1.]), ([1], [1, 2.])]
    msg = 'Invalid value passed for `standardize`: each element must be shaped'
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(
                endog, factors=1, factor_orders=1, idiosyncratic_ar1=False,
                standardize=standardize)

    # Wrong index: ndarray
    options = [
        (pd.Series(10), pd.Series(10)),
        (pd.Series(10, index=['y']), pd.Series(10, index=['y1'])),
        (pd.Series(10, index=['y1']), pd.Series(10, index=['y1'])),
        (pd.Series([10], index=['y']), pd.Series([10, 1], index=['y1', 'y2']))]
    msg = ('Invalid value passed for `standardize`: if a Pandas Series, must'
           ' have index')
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(
                endog, factors=1, factor_orders=1, idiosyncratic_ar1=False,
                standardize=standardize)

    # Wrong index: pd.Series
    options = [
        (pd.Series(10), pd.Series(10)),
        (pd.Series(10, index=['y']), pd.Series(10, index=['y1'])),
        (pd.Series(10, index=['y']), pd.Series(10, index=['y'])),
        (pd.Series([10], index=['y']), pd.Series([10, 1], index=['y1', 'y2']))]
    msg = ('Invalid value passed for `standardize`: if a Pandas Series, must'
           ' have index')
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(
                endog_pd, factors=1, factor_orders=1, idiosyncratic_ar1=False,
                standardize=standardize)


@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('standardize', [
    (10, 10), ([10], [10]), (np.array(10), np.array(10)),
    (pd.Series([10], index=['y']), pd.Series([10], index=['y']))])
def test_simulate_standardized_1d(standardize, use_pandas):
    endog = np.zeros(100) + 10
    if use_pandas:
        endog = pd.Series(endog, name='y')

    # Create the model, get the results
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=1, factor_orders=1, idiosyncratic_ar1=False,
        standardize=standardize)
    lambda1 = 2.0
    phi = 0.5
    params = [lambda1, phi, 0.0, 0]
    res = mod.smooth(params)

    # Desired value from the simulation
    mean = np.atleast_1d(standardize[0])[0]
    std = np.atleast_1d(standardize[1])[0]
    desired = phi**np.arange(10) * lambda1 * std + mean
    desired_nd = desired[:, None] if use_pandas else desired[:, None, None]

    # Test without repetitition
    actual = res.simulate(10, initial_state=[1.])
    assert_equal(actual.shape, (10,))
    assert_allclose(actual, desired)

    # Test with 1 repetitition
    actual = res.simulate(10, initial_state=[1.], repetitions=1)
    desired_shape = (10, 1) if use_pandas else (10, 1, 1)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, desired_nd)

    # Test with 2 repetititions
    actual = res.simulate(10, initial_state=[1.], repetitions=2)
    desired_shape = (10, 2) if use_pandas else (10, 1, 2)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, np.repeat(desired_nd, 2, axis=-1))


@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('standardize', [
    ([10, -4], [10., 10.]), (np.array([10, -4]), np.array([10, 10])),
    (pd.Series([10, -4], index=['y1', 'y2']),
     pd.Series([10, 10], index=['y1', 'y2']))])
def test_simulate_standardized_2d(standardize, use_pandas):
    endog = np.zeros((100, 2)) + [10, -4]
    if use_pandas:
        endog = pd.DataFrame(endog, columns=['y1', 'y2'])

    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=1, factor_orders=1, idiosyncratic_ar1=False,
        standardize=standardize)
    lambda1 = 2.0
    lambda2 = 0.5
    phi = 0.5
    params = [lambda1, lambda2, phi, 0.0, 0, 0.]
    res = mod.smooth(params)

    # Desired value from the simulation
    means = np.atleast_1d(standardize[0])
    stds = np.atleast_1d(standardize[1])
    desired = np.c_[phi**np.arange(10) * lambda1 * stds[0] + means[0],
                    phi**np.arange(10) * lambda2 * stds[1] + means[1]]
    desired_nd = desired if use_pandas else desired[..., None]

    # Test without repetitition
    actual = res.simulate(10, initial_state=[1.])
    assert_equal(actual.shape, (10, 2))
    assert_allclose(actual, desired)

    # Test with 1 repetitition
    actual = res.simulate(10, initial_state=[1.], repetitions=1)
    desired_shape = (10, 2) if use_pandas else (10, 2, 1)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, desired_nd)

    # Test with 2 repetititions
    actual = res.simulate(10, initial_state=[1.], repetitions=2)
    desired_shape = (10, 4) if use_pandas else (10, 2, 2)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, np.repeat(desired_nd, 2, axis=-1))


def check_standardized_results(res1, res2, check_diagnostics=True):
    # res1 must be the standardized results, while res2 must be the
    # the "manual standardization" results.
    mod1 = res1.model
    mod2 = res2.model

    # - Test attributes ------------------------------------------------------
    # The difference in the standard deviation (from standardization) results
    # in a difference in loglikelihood computation through the determinant
    # term. If we add that term in to the model that has been standardized, we
    # get back the loglikelihood from the model with "manual" standardization
    tmp = ((1 - res1.filter_results.missing.T) *
           np.array(mod1._endog_std)[None, :]**2)
    mask = res1.filter_results.missing.T.astype(bool)
    tmp[mask] = 1.0
    llf_obs_diff = -0.5 * np.log(tmp).sum(axis=1)
    assert_allclose(res1.llf_obs + llf_obs_diff, res2.llf_obs)

    assert_allclose(res1.mae, res2.mae)
    assert_allclose(res1.mse, res2.mse)
    assert_allclose(res1.sse, res2.sse)

    # The reverse transformation has not been applied to the fittedvalues and
    # resid attributes
    std = np.array(mod1._endog_std)
    mean = np.array(mod1._endog_mean)
    if mod1.k_endog > 1:
        std = std[None, :]
        mean = mean[None, :]

    if mod1.k_endog == 1:
        assert_allclose(res1.fittedvalues.shape, (mod1.nobs,))
    else:
        assert_allclose(res1.fittedvalues.shape, (mod1.nobs, mod1.k_endog))

    actual = np.array(res1.fittedvalues) * std + mean
    assert_allclose(actual, res2.fittedvalues)
    actual = np.array(res1.resid) * std
    assert_allclose(actual, res2.resid)

    # - Test diagnostics -----------------------------------------------------
    if check_diagnostics:
        actual = res1.test_normality(method='jarquebera')
        desired = res2.test_normality(method='jarquebera')
        assert_allclose(actual, desired)

        actual = res1.test_heteroskedasticity(method='breakvar')
        desired = res2.test_heteroskedasticity(method='breakvar')
        assert_allclose(actual, desired)

        lags = min(10, res1.nobs_effective // 5)
        actual = res1.test_serial_correlation(method='ljungbox', lags=lags)
        desired = res2.test_serial_correlation(method='ljungbox', lags=lags)
        assert_allclose(actual, desired)

    # - Test prediction/forecasting ------------------------------------------

    # Baseline model
    start = res1.nobs // 10
    dynamic = res1.nobs // 10
    end = res1.nobs + 10
    predict_actual = res1.predict()
    forecast_actual = res1.forecast(10)
    predict_dynamic_forecast_actual = res1.predict(
        start=start, end=end, dynamic=dynamic)

    get_predict_actual = res1.get_prediction()
    get_forecast_actual = res1.get_forecast(10)
    get_predict_dynamic_forecast_actual = res1.get_prediction(
        start=start, end=end, dynamic=dynamic)

    # "Manual" standardization
    predict_desired = res2.predict()
    forecast_desired = res2.forecast(10)
    predict_dynamic_forecast_desired = res2.predict(
        start=start, end=end, dynamic=dynamic)

    get_predict_desired = res2.get_prediction()
    get_forecast_desired = res2.get_forecast(10)
    get_predict_dynamic_forecast_desired = res2.get_prediction(
        start=start, end=end, dynamic=dynamic)

    assert_allclose(predict_actual, predict_desired)
    assert_allclose(forecast_actual, forecast_desired)
    assert_allclose(predict_dynamic_forecast_actual,
                    predict_dynamic_forecast_desired)

    for i in range(mod1.k_endog):
        assert_allclose(get_predict_actual.summary_frame(endog=i),
                        get_predict_desired.summary_frame(endog=i))
        assert_allclose(get_forecast_actual.summary_frame(endog=i),
                        get_forecast_desired.summary_frame(endog=i))
        assert_allclose(
            get_predict_dynamic_forecast_actual.summary_frame(endog=i),
            get_predict_dynamic_forecast_desired.summary_frame(endog=i))

    # - Test simulation ------------------------------------------------------

    # Generate shocks
    np.random.seed(1234)
    nsimulations = 100
    initial_state = np.random.multivariate_normal(
        res1.filter_results.initial_state,
        res1.filter_results.initial_state_cov)
    raw_measurement_shocks = np.random.multivariate_normal(
        np.zeros(mod1.k_endog), np.eye(mod1.k_endog), size=nsimulations)
    state_shocks = np.random.multivariate_normal(
        np.zeros(mod1.ssm.k_posdef), mod1['state_cov'], size=nsimulations)

    L1 = np.diag(mod1['obs_cov'].diagonal()**0.5)
    measurement_shocks1 = (L1 @ raw_measurement_shocks.T).T

    L2 = np.diag(mod2['obs_cov'].diagonal()**0.5)
    measurement_shocks2 = (L2 @ raw_measurement_shocks.T).T

    # Default simulation
    sim_actual = res1.simulate(
        nsimulations=nsimulations, initial_state=initial_state,
        measurement_shocks=measurement_shocks1, state_shocks=state_shocks)
    sim_desired = res2.simulate(
        nsimulations=nsimulations, initial_state=initial_state,
        measurement_shocks=measurement_shocks2, state_shocks=state_shocks)

    assert_allclose(sim_actual, sim_desired)

    # Anchored simulation
    sim_actual = res1.simulate(
        nsimulations=nsimulations, initial_state=initial_state,
        measurement_shocks=measurement_shocks1, state_shocks=state_shocks,
        anchor='end')
    sim_desired = res2.simulate(
        nsimulations=nsimulations, initial_state=initial_state,
        measurement_shocks=measurement_shocks2, state_shocks=state_shocks,
        anchor='end')

    assert_allclose(sim_actual, sim_desired)

    # - Test impulse responses -----------------------------------------------

    irfs_actual = res1.impulse_responses(10)
    irfs_desired = res2.impulse_responses(10)
    assert_allclose(irfs_actual, irfs_desired)

    irfs_actual = res1.impulse_responses(10, orthogonalized=True)
    irfs_desired = res2.impulse_responses(10, orthogonalized=True)
    assert_allclose(irfs_actual, irfs_desired)

    irfs_actual = res1.impulse_responses(10, cumulative=True)
    irfs_desired = res2.impulse_responses(10, cumulative=True)
    assert_allclose(irfs_actual, irfs_desired)

    irfs_actual = res1.impulse_responses(
        10, orthogonalized=True, cumulative=True)
    irfs_desired = res2.impulse_responses(
        10, orthogonalized=True, cumulative=True)
    assert_allclose(irfs_actual, irfs_desired)

    irfs_actual = res1.impulse_responses(
        10, orthogonalized=True, cumulative=True, anchor='end')
    irfs_desired = res2.impulse_responses(
        10, orthogonalized=True, cumulative=True, anchor='end')
    assert_allclose(irfs_actual, irfs_desired)


def check_identical_models(mod1, mod2, check_nobs=True):
    # Dimensions
    if check_nobs:
        assert_equal(mod2.nobs, mod1.nobs)
    assert_equal(mod2.k_endog, mod1.k_endog)
    assert_equal(mod2.k_endog_M, mod1.k_endog_M)
    assert_equal(mod2.k_endog_Q, mod1.k_endog_Q)
    assert_equal(mod2.k_states, mod1.k_states)
    assert_equal(mod2.ssm.k_posdef, mod1.ssm.k_posdef)

    # Standardization
    assert_allclose(mod2._endog_mean, mod1._endog_mean)
    assert_allclose(mod2._endog_std, mod1._endog_std)
    assert_allclose(mod2.standardize, mod1.standardize)

    # Parameterization
    assert_equal(mod2.factors, mod1.factors)
    assert_equal(mod2.factor_orders, mod1.factor_orders)
    assert_equal(mod2.factor_multiplicities, mod1.factor_multiplicities)
    assert_equal(mod2.idiosyncratic_ar1, mod1.idiosyncratic_ar1)
    assert_equal(mod2.init_t0, mod1.init_t0)
    assert_equal(mod2.obs_cov_diag, mod1.obs_cov_diag)

    # Other attributes
    assert_allclose(mod2.endog_factor_map, mod1.endog_factor_map)
    assert_allclose(mod2.factor_block_orders, mod1.factor_block_orders)
    assert_equal(mod2.endog_names, mod1.endog_names)
    assert_equal(mod2.factor_names, mod1.factor_names)
    assert_equal(mod2.k_factors, mod1.k_factors)
    assert_equal(mod2.k_factor_blocks, mod1.k_factor_blocks)
    assert_equal(mod2.max_factor_order, mod1.max_factor_order)


def check_append(res1, res2, endog_M2, endog_Q2):
    # res1 is the usual results object
    # res2 is an "manually standardized" results object
    # endog_M2 is the monthly data to append
    # endog_Q2 is the quarterly data to append

    # Test for appending to the usual results object
    res1_append = res1.append(endog_M2, endog_quarterly=endog_Q2)
    mod1_append = res1_append.model
    mod1 = res1.model

    # Check that the models are the same
    check_identical_models(mod1, mod1_append, check_nobs=False)
    assert_equal(mod1_append.nobs, mod1.nobs + len(endog_M2))
    assert_allclose(mod1_append.endog[:mod1.nobs], mod1.endog)

    # Check that the results are the same prior to the appended data
    assert_allclose(res1_append.filter_results.initial_state_cov,
                    res1.filter_results.initial_state_cov)

    assert_allclose(res1_append.llf_obs[:mod1.nobs], res1.llf_obs)
    assert_allclose(res1_append.filter_results.forecasts[:, :mod1.nobs],
                    res1.filter_results.forecasts)
    assert_allclose(res1_append.filter_results.forecasts_error[:, :mod1.nobs],
                    res1.filter_results.forecasts_error)

    assert_allclose(res1_append.filter_results.initial_state,
                    res1.filter_results.initial_state)
    assert_allclose(res1_append.filter_results.initial_state_cov,
                    res1.filter_results.initial_state_cov)

    assert_allclose(res1_append.filter_results.filtered_state[:, :mod1.nobs],
                    res1.filter_results.filtered_state)
    assert_allclose(
        res1_append.filter_results.filtered_state_cov[..., :mod1.nobs],
        res1.filter_results.filtered_state_cov)

    # Test that "manual standardization" continues to work with append
    res2_append = res2.append(endog_M2, endog_quarterly=endog_Q2)
    mod2_append = res2_append.model
    mod2 = res2.model

    # Because our mod2 has manual changes, we need to copy those over and
    # re-create the appended results object
    mod2_append.update(res2_append.params)
    mod2_append['obs_intercept'] = mod2['obs_intercept']
    mod2_append['design'] = mod2['design']
    mod2_append['obs_cov'] = mod2['obs_cov']
    mod2_append.update = lambda params, **kwargs: params
    res2_append = mod2_append.smooth(res2_append.params)

    # Check output
    check_identical_models(mod2, mod2_append, check_nobs=False)

    # Check that the results are the same prior to the appended data
    assert_allclose(res2_append.filter_results.initial_state_cov,
                    res2.filter_results.initial_state_cov)

    assert_allclose(res2_append.llf_obs[:mod2.nobs], res2.llf_obs)
    assert_allclose(res2_append.filter_results.forecasts[:, :mod2.nobs],
                    res2.filter_results.forecasts)
    assert_allclose(res2_append.filter_results.forecasts_error[:, :mod2.nobs],
                    res2.filter_results.forecasts_error)

    assert_allclose(res2_append.filter_results.initial_state,
                    res2.filter_results.initial_state)
    assert_allclose(res2_append.filter_results.initial_state_cov,
                    res2.filter_results.initial_state_cov)

    assert_allclose(res2_append.filter_results.filtered_state[:, :mod2.nobs],
                    res2.filter_results.filtered_state)
    assert_allclose(
        res2_append.filter_results.filtered_state_cov[..., :mod2.nobs],
        res2.filter_results.filtered_state_cov)

    # Check res1_append vs res2_append
    check_standardized_results(res1_append, res2_append)


def check_extend(res1, res2, endog_M2, endog_Q2):
    # res1 is the usual results object
    # endog_M2 is the monthly data to append
    # endog_Q2 is the, optional, quarterly data to append
    # res2 is an optional "manually standardized" results object

    # Test for extending to the usual results object
    res1_extend = res1.extend(endog_M2, endog_quarterly=endog_Q2)
    mod1_extend = res1_extend.model
    mod1 = res1.model

    # Check that the models are the same
    check_identical_models(mod1, mod1_extend, check_nobs=False)
    assert_equal(mod1_extend.nobs, len(endog_M2))

    # Test that "manual standardization" continues to work with extend
    res2_extend = res2.extend(endog_M2, endog_quarterly=endog_Q2)
    mod2_extend = res2_extend.model
    mod2 = res2.model

    # Because our mod2 has manual changes, we need to copy those over and
    # re-create the extended results object
    mod2_extend.update(res2_extend.params)
    mod2_extend['obs_intercept'] = mod2['obs_intercept']
    mod2_extend['design'] = mod2['design']
    mod2_extend['obs_cov'] = mod2['obs_cov']
    mod2_extend.update = lambda params, **kwargs: params
    res2_extend = mod2_extend.smooth(res2_extend.params)

    # Check that the models are the same
    check_identical_models(mod2, mod2_extend, check_nobs=False)

    # Check res1_extend vs res2_extend
    check_standardized_results(res1_extend, res2_extend,
                               check_diagnostics=False)


def check_apply(res1, res2, endog_M, endog_Q):
    # res1 is the usual results object
    # endog_M2 is the monthly data to append
    # endog_Q2 is the, optional, quarterly data to append
    # res2 is an optional "manually standardized" results object

    # Test for applying to the usual results object
    res1_apply = res1.apply(endog_M, endog_quarterly=endog_Q)
    mod1_apply = res1_apply.model
    mod1 = res1.model

    # Check that the models are the same
    check_identical_models(mod1, mod1_apply, check_nobs=False)
    assert_equal(mod1_apply.nobs, len(endog_M))

    # Test that "manual standardization" continues to work with apply
    res2_apply = res2.apply(endog_M, endog_quarterly=endog_Q)
    mod2_apply = res2_apply.model
    mod2 = res2.model

    # Because our mod2 has manual changes, we need to copy those over and
    # re-create the applyed results object
    mod2_apply.update(res2_apply.params)
    mod2_apply['obs_intercept'] = mod2['obs_intercept']
    mod2_apply['design'] = mod2['design']
    mod2_apply['obs_cov'] = mod2['obs_cov']
    mod2_apply.update = lambda params, **kwargs: params
    res2_apply = mod2_apply.smooth(res2_apply.params)

    # Check that the models are the same
    check_identical_models(mod2, mod2_apply, check_nobs=False)

    # Check res1_apply vs res2_apply
    check_standardized_results(res1_apply, res2_apply,
                               check_diagnostics=False)


@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('k_endog', [1, 2])
@pytest.mark.parametrize('idiosyncratic_ar1', [True, False])
def test_standardized_monthly(reset_randomstate, idiosyncratic_ar1, k_endog,
                              use_pandas):
    nobs = 100
    k2 = 2
    _, _, f2 = test_dynamic_factor_mq_monte_carlo.gen_k_factor2(
        nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)

    if k_endog == 1:
        endog = f2.iloc[:, 0]
        endog_mean = pd.Series([10], index=['f1'])
        endog_std = pd.Series([1], index=['f1'])
    else:
        endog = f2
        endog_mean = pd.Series([10, -4], index=['f1', 'f2'])
        endog_std = pd.Series([1, 1], index=['f1', 'f2'])

    if not use_pandas:
        endog = endog.values
        endog_mean = endog_mean.values
        endog_std = endog_std.values

    # - Actual ---------------------------------------------------------------
    # Baseline model
    mod1 = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=1, factor_multiplicities=1, factor_orders=1,
        idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=(endog_mean, endog_std))

    params = pd.Series(mod1.start_params, index=mod1.param_names)
    res1 = mod1.smooth(params)

    # - Desired --------------------------------------------------------------
    # Create an identical model, but without standardization
    mod2 = dynamic_factor_mq.DynamicFactorMQ(
        endog, factors=1, factor_multiplicities=1, factor_orders=1,
        idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=False)
    # Start with the parameters from the standardized model
    mod2.update(params)

    # Update the observation equation to manually implement the standardization
    mod2['obs_intercept'] = np.array(endog_mean)
    mod2['design'] *= np.array(endog_std)[:, None]
    mod2['obs_cov'] *= np.array(endog_std)[:, None]**2

    # Prevent the model from overwriting our changes
    mod2.update = lambda params, **kwargs: params

    # Create the results object based on the model with manual standardization
    res2 = mod2.smooth(params)

    # - Test results ---------------------------------------------------------
    check_standardized_results(res1, res2)


@pytest.mark.parametrize('idiosyncratic_ar1', [True, False])
def test_standardized_MQ(reset_randomstate, idiosyncratic_ar1):
    nobs = 100
    idiosyncratic_ar1 = False
    k1 = 2
    k2 = 2
    endog1_M, endog1_Q, f1 = test_dynamic_factor_mq_monte_carlo.gen_k_factor1(
        nobs, k=k1, idiosyncratic_ar1=idiosyncratic_ar1)
    endog2_M, endog2_Q, f2 = test_dynamic_factor_mq_monte_carlo.gen_k_factor2(
        nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)

    endog_M = pd.concat([endog1_M, f2, endog2_M], axis=1, sort=True)
    endog_Q = pd.concat([endog1_Q, endog2_Q], axis=1, sort=True)

    endog_M1 = endog_M.loc[:'1957-12']
    endog_Q1 = endog_Q.loc[:'1957Q4']
    endog_M2 = endog_M.loc['1958-01':]
    endog_Q2 = endog_Q.loc['1958Q1':]

    factors = {f'yM{i + 1}_f1': ['a'] for i in range(k1)}
    factors.update({f'f{i + 1}': ['b'] for i in range(2)})
    factors.update({f'yM{i + 1}_f2': ['b'] for i in range(k2)})
    factors.update({f'yQ{i + 1}_f1': ['a'] for i in range(k1)})
    factors.update({f'yQ{i + 1}_f2': ['b'] for i in range(k2)})
    factor_multiplicities = {'b': 2}

    # - Actual ---------------------------------------------------------------
    # Baseline model
    endog_mean = pd.Series(
        np.random.normal(size=len(factors)), index=factors.keys())
    endog_std = pd.Series(
        np.abs(np.random.normal(size=len(factors))), index=factors.keys())

    mod1 = dynamic_factor_mq.DynamicFactorMQ(
        endog_M1, endog_quarterly=endog_Q1,
        factors=factors, factor_multiplicities=factor_multiplicities,
        factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=(endog_mean, endog_std))

    params = pd.Series(mod1.start_params, index=mod1.param_names)
    res1 = mod1.smooth(params)

    # - Desired --------------------------------------------------------------
    # Create an identical model, but without standardization
    mod2 = dynamic_factor_mq.DynamicFactorMQ(
        endog_M1, endog_quarterly=endog_Q1,
        factors=factors, factor_multiplicities=factor_multiplicities,
        factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=False)
    # Start with the parameters from the standardized model
    mod2.update(params)

    # Update the observation equation to manually implement the standardization
    mod2['obs_intercept'] = endog_mean
    mod2['design'] *= np.array(endog_std)[:, None]
    mod2['obs_cov'] *= np.array(endog_std)[:, None]**2

    # Prevent the model from overwriting our changes
    mod2.update = lambda params, **kwargs: params

    # Create the results object based on the model with manual standardization
    res2 = mod2.smooth(params)

    # - Test results ---------------------------------------------------------
    check_standardized_results(res1, res2)

    # - Test append, extend, apply -------------------------------------------
    check_append(res1, res2, endog_M2, endog_Q2)
    check_extend(res1, res2, endog_M2, endog_Q2)
    check_apply(res1, res2, endog_M, endog_Q)

    # - Test news ------------------------------------------------------------
    res1_apply = res1.apply(endog_M, endog_quarterly=endog_Q)

    res2_apply = res2.apply(endog_M, endog_quarterly=endog_Q)
    mod2_apply = res2_apply.model
    # Because our mod2 has manual changes, we need to copy those over and
    # re-create the applyed results object
    mod2_apply.update(res2_apply.params)
    mod2_apply['obs_intercept'] = mod2['obs_intercept']
    mod2_apply['design'] = mod2['design']
    mod2_apply['obs_cov'] = mod2['obs_cov']
    mod2_apply.update = lambda params, **kwargs: params
    res2_apply = mod2_apply.smooth(res2_apply.params)

    news1 = res1_apply.news(res1, start='1958-01', end='1958-03',
                            comparison_type='previous')
    news2 = res2_apply.news(res2, start='1958-01', end='1958-03',
                            comparison_type='previous')
    attributes = [
        'total_impacts', 'update_impacts', 'revision_impacts', 'news',
        'weights', 'update_forecasts', 'update_realized',
        'prev_impacted_forecasts', 'post_impacted_forecasts', 'revisions_iloc',
        'revisions_ix', 'updates_iloc', 'updates_ix']
    for attr in attributes:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)
