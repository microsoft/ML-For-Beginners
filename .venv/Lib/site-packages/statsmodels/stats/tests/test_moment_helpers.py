# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:33:56 2011

Author: Josef Perktold
"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_, assert_equal

from statsmodels.stats import moment_helpers
from statsmodels.stats.moment_helpers import (cov2corr, mvsk2mc, mc2mvsk,
                                              mnc2mc, mc2mnc, cum2mc, mc2cum,
                                              mnc2cum)


def test_cov2corr():
    cov_a = np.ones((3, 3)) + np.diag(np.arange(1, 4) ** 2 - 1)
    corr_a = np.array([[1, 1 / 2., 1 / 3.],
                       [1 / 2., 1, 1 / 2. / 3.],
                       [1 / 3., 1 / 2. / 3., 1]])

    corr = cov2corr(cov_a)
    assert_almost_equal(corr, corr_a, decimal=15)

    cov_mat = cov_a
    corr_mat = cov2corr(cov_mat)
    assert_(isinstance(corr_mat, np.ndarray))
    assert_equal(corr_mat, corr)

    cov_ma = np.ma.array(cov_a)
    corr_ma = cov2corr(cov_ma)
    assert_equal(corr_mat, corr)

    assert_(isinstance(corr_ma, np.ma.core.MaskedArray))

    cov_ma2 = np.ma.array(cov_a, mask=[[False, True, False],
                                       [True, False, False],
                                       [False, False, False]])

    corr_ma2 = cov2corr(cov_ma2)
    assert_(np.ma.allclose(corr_ma, corr, atol=1e-15))
    assert_equal(corr_ma2.mask, cov_ma2.mask)


ms = [([0.0, 1, 0, 3], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
      ([1.0, 1, 0, 3], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 6.0]),
      ([0.0, 1, 1, 3], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]),
      ([1.0, 1, 1, 3], [1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 2.0]),
      ([1.0, 1, 1, 4], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 3.0]),
      ([1.0, 2, 0, 3], [1.0, 2.0, 0.0, -9.0], [1.0, 1.0, -4.0, 9.0]),
      ([0.0, 2, 1, 3], [0.0, 2.0, 1.0, -9.0], [0.0, 2.0, 1.0, -9.0]),
      # neg.variance if mnc2<mnc1
      ([1.0, 0.5, 0, 3], [1.0, 0.5, 0.0, 2.25], [1.0, -0.5, 0.5, 2.25]),
      ([0.0, 0.5, 1, 3], [0.0, 0.5, 1.0, 2.25], [0.0, 0.5, 1.0, 2.25]),
      ([0.0, 1, 0, 3, 0], [0.0, 1.0, 0.0, 0.0, 0.0], [0., 1., 0., 0., 0.]),
      ([1.0, 1, 0, 3, 1], [1.0, 1.0, 0.0, 0.0, 1.0], [1., 0., -1., 6., -20.])]


@pytest.mark.parametrize('mom', ms)
def test_moment_conversion(mom):
    # this was initially written for an old version of moment_helpers
    # I'm not sure whether there are not redundant cases after moving functions

    # test moment -> cumulant
    assert_equal(mnc2cum(mc2mnc(mom[0])), mom[1])
    assert_equal(mnc2cum(mom[0]), mom[2])
    if len(mom) <= 4:
        assert_equal(mc2cum(mom[0]), mom[1])

    # test   cumulant -> moment
    assert_equal(cum2mc(mom[1]), mom[0])
    assert_equal(mc2mnc(cum2mc(mom[2])), mom[0])
    if len(mom) <= 4:
        assert_equal(cum2mc(mom[1]), mom[0])

    # round trip: mnc -> cum -> mc == mnc -> mc,
    assert_equal(cum2mc(mnc2cum(mom[0])), mnc2mc(mom[0]))

    # round trip: mc -> mnc -> mc ==  mc,
    assert_equal(mc2mnc(mnc2mc(mom[0])), mom[0])

    if len(mom[0]) == 4:
        # round trip: mc -> mvsk -> mc ==  mc
        assert_equal(mvsk2mc(mc2mvsk(mom[0])), mom[0])

        # round trip: mc -> mvsk -> mnc ==  mc -> mnc
        # TODO: mvsk2mnc not defined
        # assert_equal(mvsk2mnc(mc2mvsk(mom[0])), mc2mnc(mom[0]))


rs = np.random.RandomState(12345)
random_vals = rs.randint(0, 100, 12).reshape(4, 3)
multidimension_test_vals = [np.array([[5., 10., 1.],
                                      [5., 10., 1.],
                                      [5., 10., 1.],
                                      [80., 310., 4.]]),
                            random_vals]


@pytest.mark.parametrize('test_vals', multidimension_test_vals)
def test_multidimensional(test_vals):
    assert_almost_equal(cum2mc(mnc2cum(mc2mnc(test_vals).T).T).T, test_vals)
    assert_almost_equal(cum2mc(mc2cum(test_vals).T).T, test_vals)
    assert_almost_equal(mvsk2mc(mc2mvsk(test_vals).T).T, test_vals)


@pytest.mark.parametrize('func_name', ['cum2mc', 'cum2mc', 'mc2cum', 'mc2mnc',
                                       'mc2mvsk', 'mnc2cum', 'mnc2mc',
                                       'mnc2mc', 'mvsk2mc', 'mvsk2mnc'])
def test_moment_conversion_types(func_name):
    # written in 2009
    # TODO: why did I use list as return type?
    func = getattr(moment_helpers, func_name)

    assert (isinstance(func([1.0, 1, 0, 3]), list) or
            isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray)))

    assert (isinstance(func(np.array([1.0, 1, 0, 3])), list) or
            isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray)))

    assert (isinstance(func(tuple([1.0, 1, 0, 3])), list) or
            isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray)))
