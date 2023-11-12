# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:26:01 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.transform_model import StandardizeTransform


def test_standardize1():

    np.random.seed(123)
    x = 1 + np.random.randn(5, 4)

    transf = StandardizeTransform(x)
    xs1 = transf(x)

    assert_allclose(transf.mean, x.mean(0), rtol=1e-13)
    assert_allclose(transf.scale, x.std(0, ddof=1), rtol=1e-13)

    xs2 = stats.zscore(x, ddof=1)
    assert_allclose(xs1, xs2, rtol=1e-13, atol=1e-20)

    # check we use stored transformation
    xs4 = transf(2 * x)
    assert_allclose(xs4, (2*x - transf.mean) / transf.scale,
                    rtol=1e-13, atol=1e-20)

    # affine transform does not change standardized
    x2 = 2 * x + np.random.randn(4)
    transf2 = StandardizeTransform(x2)
    xs3 = transf2(x2)
    assert_allclose(xs3, xs1, rtol=1e-13, atol=1e-20)

    # check constant
    x5 = np.column_stack((np.ones(x.shape[0]), x))
    transf5 = StandardizeTransform(x5)
    xs5 = transf5(x5)

    assert_equal(transf5.const_idx, 0)
    assert_equal(xs5[:, 0], np.ones(x.shape[0]))
    assert_allclose(xs5[:, 1:], xs1, rtol=1e-13, atol=1e-20)


def test_standardize_ols():

    np.random.seed(123)
    nobs = 20
    x = 1 + np.random.randn(nobs, 4)
    exog = np.column_stack((np.ones(nobs), x))
    endog = exog.sum(1) + np.random.randn(nobs)

    res2 = OLS(endog, exog).fit()
    transf = StandardizeTransform(exog)
    exog_st = transf(exog)
    res1 = OLS(endog, exog_st).fit()
    params = transf.transform_params(res1.params)
    assert_allclose(params, res2.params, rtol=1e-13)
