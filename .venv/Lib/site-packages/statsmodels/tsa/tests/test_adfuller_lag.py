# -*- coding: utf-8 -*-
"""Test for autolag of adfuller

Created on Wed May 30 21:39:46 2012
Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from statsmodels.datasets import macrodata
import statsmodels.tsa.stattools as tsast


def test_adf_autolag():
    #see issue #246
    #this is mostly a unit test
    d2 = macrodata.load_pandas().data
    x = np.log(d2['realgdp'].values)
    for k_trend, tr in enumerate(['n', 'c', 'ct', 'ctt']):
        x = np.log(d2['realgdp'].values)
        xd = np.diff(x)

        #check exog
        adf3 = tsast.adfuller(x, maxlag=None, autolag='aic',
                              regression=tr, store=True, regresults=True)
        st2 = adf3[-1]

        assert_equal(len(st2.autolag_results), 15 + 1)  #+1 for lagged level
        for i, res in sorted(st2.autolag_results.items())[:5]:
            lag = i - k_trend
            #assert correct design matrices in _autolag
            assert_equal(res.model.exog[-10:,k_trend], x[-11:-1])
            assert_equal(res.model.exog[-1,k_trend+1:], xd[-lag:-1][::-1])
            #min-ic lag of dfgls in Stata is also 2, or 9 for maic with notrend
            assert_equal(st2.usedlag, 2)

        #same result with lag fixed at usedlag of autolag
        adf2 = tsast.adfuller(x, maxlag=2, autolag=None, regression=tr)
        assert_almost_equal(adf3[:2], adf2[:2], decimal=12)

    tr = 'c'
    #check maxlag with autolag
    adf3 = tsast.adfuller(x, maxlag=5, autolag='aic',
                          regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 5 + 1)
    adf3 = tsast.adfuller(x, maxlag=0, autolag='aic',
                          regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 0 + 1)
