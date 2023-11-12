# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:42:00 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.distributions.tools as dt


def test_grid():
    # test bivariate independent beta
    k1, k2 = 3, 5
    xg1 = np.arange(k1) / (k1 - 1)
    xg2 = np.arange(k2) / (k2 - 1)

    # histogram values for distribution
    distr1 = stats.beta(2, 5)
    distr2 = stats.beta(4, 3)
    cdf1 = distr1.cdf(xg1)
    cdf2 = distr2.cdf(xg2)
    prob1 = np.diff(cdf1, prepend=0)
    prob2 = np.diff(cdf2, prepend=0)
    cd2d = cdf1[:, None] * cdf2
    pd2d = prob1[:, None] * prob2

    probs = dt.cdf2prob_grid(cd2d)
    cdfs = dt.prob2cdf_grid(pd2d)

    assert_allclose(cdfs, cd2d, atol=1e-12)
    assert_allclose(probs, pd2d, atol=1e-12)

    # check random sample
    nobs = 1000
    np.random.seed(789123)
    rvs = np.column_stack([distr1.rvs(size=nobs), distr2.rvs(size=nobs)])
    hist = np.histogramdd(rvs, [xg1, xg2])
    assert_allclose(probs[1:, 1:], hist[0] / len(rvs), atol=0.02)


def test_average_grid():
    x1 = np.arange(1, 4)
    x2 = np.arange(4)
    y = x1[:, None] * x2

    res1 = np.array([[0.75, 2.25, 3.75],
                     [1.25, 3.75, 6.25]])

    res0 = dt.average_grid(y, coords=[x1, x2])
    assert_allclose(res0, res1, rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1, x2], _method="slicing")
    assert_allclose(res0, res1, rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1, x2], _method="convolve")
    assert_allclose(res0, res1, rtol=1e-13)

    res0 = dt.average_grid(y, coords=[x1 / x1.max(), x2 / x2.max()])
    assert_allclose(res0, res1 / x1.max() / x2.max(), rtol=1e-13)
    res0 = dt.average_grid(y, coords=[x1 / x1.max(), x2 / x2.max()],
                           _method="convolve")
    assert_allclose(res0, res1 / x1.max() / x2.max(), rtol=1e-13)


def test_grid_class():

    res = {'k_grid': [3, 5],
           'x_marginal': [np.array([0., 0.5, 1.]),
                          np.array([0., 0.25, 0.5, 0.75, 1.])],
           'idx_flat.T': np.array([
               [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
               [0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4.]])
           }
    gg = dt._Grid([3, 5])
    assert_equal(gg.k_grid, res["k_grid"])
    assert gg.x_marginal, res["x_marginal"]
    assert_allclose(gg.idx_flat, res["idx_flat.T"].T, atol=1e-12)
    assert_allclose(gg.x_flat, res["idx_flat.T"].T / [2, 4], atol=1e-12)

    gg = dt._Grid([3, 5], eps=0.001)
    assert_allclose(gg.x_flat.min(), 0.001, atol=1e-12)
    assert_allclose(gg.x_flat.max(), 0.999, atol=1e-12)
    xmf = np.concatenate(gg.x_marginal)
    assert_allclose(xmf.min(), 0.001, atol=1e-12)
    assert_allclose(xmf.max(), 0.999, atol=1e-12)

    # 1-dim
    gg = dt._Grid([5], eps=0.001)
    res = {'k_grid': [5],
           'x_marginal': [np.array([0.001, 0.25, 0.5, 0.75, 0.999])],
           'idx_flat.T': np.array([[0., 1., 2., 3., 4.]])
           }
    assert_equal(gg.k_grid, res["k_grid"])
    assert gg.x_marginal, res["x_marginal"]
    assert_allclose(gg.idx_flat, res["idx_flat.T"].T, atol=1e-12)
    # x_flat is 2-dim even if grid is 1-dim, TODO: maybe change
    assert_allclose(gg.x_flat, res["x_marginal"][0][:, None], atol=1e-12)

    # 3-dim
    gg = dt._Grid([3, 3, 2], eps=0.)
    res = {'k_grid': [3, 3, 2],
           'x_marginal': [np.array([0., 0.5, 1.]),
                          np.array([0., 0.5, 1.]),
                          np.array([0., 1.])],
           'idx_flat.T': np.array([
               [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2.,
                2., 2.],
               [0., 0., 1., 1., 2., 2., 0., 0., 1., 1., 2., 2., 0., 0., 1., 1.,
                2., 2.],
               [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
                0., 1.]])
           }
    assert_equal(gg.k_grid, res["k_grid"])
    assert gg.x_marginal, res["x_marginal"]
    assert_allclose(gg.idx_flat, res["idx_flat.T"].T, atol=1e-12)
    assert_allclose(gg.x_flat, res["idx_flat.T"].T / [2, 2, 1], atol=1e-12)


def test_bernstein_1d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    xg2 = np.arange(2 * k) / (2 * k - 1)
    # verify linear coefficients are mapped to themselves
    res_bp = dt._eval_bernstein_1d(xg2, xg1)
    assert_allclose(res_bp, xg2, atol=1e-12)

    res_bp = dt._eval_bernstein_1d(xg2, xg1, method="beta")
    assert_allclose(res_bp, xg2, atol=1e-12)

    res_bp = dt._eval_bernstein_1d(xg2, xg1, method="bpoly")
    assert_allclose(res_bp, xg2, atol=1e-12)


def test_bernstein_2d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    cd2d = xg1[:, None] * xg1
    # verify linear coefficients are mapped to themselves
    for evalbp in (dt._eval_bernstein_2d, dt._eval_bernstein_dd):
        k_x = 2 * k
        # create flattened grid of bivariate values
        x2d = np.column_stack(
                np.unravel_index(np.arange(k_x * k_x), (k_x, k_x))
                ).astype(float)
        x2d /= x2d.max(0)

        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, np.prod(x2d, axis=1), atol=1e-12)

        # check univariate margins
        x2d = np.column_stack((np.arange(k_x) / (k_x - 1), np.ones(k_x)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 0], atol=1e-12)

        # check univariate margins
        x2d = np.column_stack((np.ones(k_x), np.arange(k_x) / (k_x - 1)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 1], atol=1e-12)
