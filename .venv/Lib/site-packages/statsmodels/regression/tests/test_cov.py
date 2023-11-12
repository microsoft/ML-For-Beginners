"""Example: minimal OLS

"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import statsmodels.api as sm


def test_HC_use():
    np.random.seed(0)
    nsample = 100
    x = np.linspace(0,10, 100)
    X = sm.add_constant(np.column_stack((x, x**2)), prepend=False)
    beta = np.array([1, 0.1, 10])
    y = np.dot(X, beta) + np.random.normal(size=nsample)

    results = sm.OLS(y, X).fit()

    # test cov_params
    idx = np.array([1, 2])
    cov12 = results.cov_params(column=[1, 2], cov_p=results.cov_HC0)
    assert_almost_equal(cov12, results.cov_HC0[idx[:, None], idx], decimal=15)

    #test t_test
    tvals = results.params/results.HC0_se
    ttest = results.t_test(np.eye(3), cov_p=results.cov_HC0)
    assert_almost_equal(ttest.tvalue, tvals, decimal=14)
    assert_almost_equal(ttest.sd, results.HC0_se, decimal=14)

    #test f_test
    ftest = results.f_test(np.eye(3)[:-1], cov_p=results.cov_HC0)
    slopes = results.params[:-1]
    idx = np.array([0,1])
    cov_slopes = results.cov_HC0[idx[:,None], idx]
    fval = np.dot(slopes, np.dot(np.linalg.inv(cov_slopes), slopes))/len(idx)
    assert_allclose(ftest.fvalue, fval, rtol=12)
