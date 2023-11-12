# -*- coding: utf-8 -*-
"""
unit test for spline and other smoother classes

Author: Luca Puggini

"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.gam.smooth_basis import (UnivariatePolynomialSmoother,
                                          PolynomialSmoother,
                                          BSplines)


def test_univariate_polynomial_smoother():
    x = np.linspace(0, 1, 5)
    pol = UnivariatePolynomialSmoother(x, degree=3)
    assert_equal(pol.basis.shape, (5, 3))
    assert_allclose(pol.basis[:, 2], x.ravel() ** 3)


def test_multivariate_polynomial_basis():
    np.random.seed(1)
    x = np.random.normal(0, 1, (10, 2))
    degrees = [3, 4]
    mps = PolynomialSmoother(x, degrees)
    for i, deg in enumerate(degrees):
        uv_basis = UnivariatePolynomialSmoother(x[:, i], degree=deg).basis
        assert_allclose(mps.smoothers[i].basis, uv_basis)


@pytest.mark.parametrize(
    "x, df, degree",
    [
        (
            np.c_[np.linspace(0, 1, 100), np.linspace(0, 10, 100)],
            [5, 6],
            [3, 5]
        ),
        (np.linspace(0, 1, 100), 6, 3),
    ]
)
def test_bsplines(x, df, degree):
    bspline = BSplines(x, df, degree)
    bspline.transform(x)
