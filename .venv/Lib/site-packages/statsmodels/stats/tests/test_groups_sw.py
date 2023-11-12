# -*- coding: utf-8 -*-
"""Test for a helper function for PanelHAC robust covariance

the functions should be rewritten to make it more efficient

Created on Thu May 17 21:09:41 2012

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import GroupSorted


class CheckPanelLagMixin:

    @classmethod
    def calculate(cls):
        cls.g = g = GroupSorted(cls.gind)  # pylint: disable-msg=W0201
        cls.alla = [(lag, sw.lagged_groups(cls.x, lag, g.groupidx))  # pylint: disable-msg=W0201
                         for lag in range(5)]

    def test_values(self):
        for lag, (y0, ylag) in self.alla:
            assert_equal(y0, self.alle[lag].T)
            assert_equal(y0, ylag + lag)

    def test_raises(self):
        mlag = self.mlag
        assert_raises(ValueError, sw.lagged_groups, self.x, mlag,
                      self.g.groupidx)


class TestBalanced(CheckPanelLagMixin):

    @classmethod
    def setup_class(cls):
        cls.gind = np.repeat([0,1,2], 5)
        cls.mlag = 5
        x = np.arange(15)
        x += 10**cls.gind
        cls.x = x[:,None]
        #expected result
        cls.alle = {
            0 : np.array([[  1,   2,   3,   4,   5,  15,  16,  17,  18,  19,
                           110, 111, 112, 113, 114]]),
            1 : np.array([[  2,   3,   4,   5,  16,  17,  18,  19, 111, 112,
                           113, 114]]),
            2 : np.array([[  3,   4,   5,  17,  18,  19, 112, 113, 114]]),
            3 : np.array([[  4,   5,  18,  19, 113, 114]]),
            4 : np.array([[  5,  19, 114]])
            }
        cls.calculate()

class TestUnBalanced(CheckPanelLagMixin):

    @classmethod
    def setup_class(cls):
        cls.gind = gind = np.repeat([0,1,2], [3, 5, 10])
        cls.mlag = 10  #maxlag
        x = np.arange(18)
        x += 10**gind
        cls.x = x[:,None]

        #expected result
        cls.alle = {
            0 : np.array([[  1,   2,   3,  13,  14,  15,  16,  17, 108, 109,
                           110, 111, 112, 113, 114, 115, 116, 117]]),
            1 : np.array([[  2,   3,  14,  15,  16,  17, 109, 110, 111, 112,
                           113, 114, 115, 116, 117]]),
            2 : np.array([[  3,  15,  16,  17, 110, 111, 112, 113, 114, 115,
                           116, 117]]),
            3 : np.array([[ 16,  17, 111, 112, 113, 114, 115, 116, 117]]),
            4 : np.array([[ 17, 112, 113, 114, 115, 116, 117]]),
            5 : np.array([[113, 114, 115, 116, 117]]),
            }
        cls.calculate()
