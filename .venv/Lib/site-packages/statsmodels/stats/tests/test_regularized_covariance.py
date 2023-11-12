import numpy as np
from numpy.testing import assert_equal, assert_
from statsmodels.stats.regularized_covariance import (
    _calc_nodewise_row, _calc_nodewise_weight,
    _calc_approx_inv_cov, RegularizedInvCovariance)


def test_calc_nodewise_row():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat = _calc_nodewise_row(X, 0, 0.01)
    assert_equal(ghat.shape, (2,))


def test_calc_nodewise_weight():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat = np.random.normal(size=2)
    that = _calc_nodewise_weight(X, ghat, 0, 0.01)
    assert_(isinstance(that, float))


def test_calc_approx_inv_cov():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    ghat_l = []
    that_l = []
    for i in range(3):
        ghat = _calc_nodewise_row(X, i, 0.01)
        that = _calc_nodewise_weight(X, ghat, i, 0.01)
        ghat_l.append(ghat)
        that_l.append(that)
    theta_hat = _calc_approx_inv_cov(np.array(ghat_l), np.array(that_l))
    assert_equal(theta_hat.shape, (3, 3))


def test_fit():

    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    inv = np.linalg.inv(np.cov(X.T))
    regcov = RegularizedInvCovariance(exog=X)
    regcov.fit()
    # check that unregularized is what we expect
    diff = np.linalg.norm(regcov.approx_inv_cov() - inv)
    assert_(diff < 0.1)

    # check that regularizing actually does something
    regcov.fit(alpha=0.5)
    assert_(np.sum(regcov.approx_inv_cov() == 0) > np.sum(inv == 0))
