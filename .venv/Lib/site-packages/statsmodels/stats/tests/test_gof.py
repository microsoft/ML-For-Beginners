# -*- coding: utf-8 -*-
"""

Created on Thu Feb 28 13:24:59 2013

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from statsmodels.stats.gof import (chisquare, chisquare_power,
                                   chisquare_effectsize)
from statsmodels.tools.testing import Holder


def test_chisquare_power():
    from .results.results_power import pwr_chisquare
    for case in pwr_chisquare.values():
        power = chisquare_power(case.w, case.N, case.df + 1,
                                alpha=case.sig_level)
        assert_almost_equal(power, case.power, decimal=6,
                            err_msg=repr(vars(case)))

def test_chisquare():
    # TODO: no tests for ``value`` yet
    res1 = Holder()
    res2 = Holder()
    #> freq = c(1048,  660,  510,  420,  362)
    #> pr1 = c(1020,  690,  510,  420,  360)
    #> pr2 = c(1050,  660,  510,  420,  360)
    #> c = chisq.test(freq, p=pr1, rescale.p = TRUE)
    #> cat_items(c, "res1.")
    res1.statistic = 2.084086388178453
    res1.parameter = 4
    res1.p_value = 0.72029651761105
    res1.method = 'Chi-squared test for given probabilities'
    res1.data_name = 'freq'
    res1.observed = np.array([
         1048, 660, 510, 420, 362
        ])
    res1.expected = np.array([
         1020, 690, 510, 420, 360
        ])
    res1.residuals = np.array([
         0.876714007519206, -1.142080481440321, -2.517068894406109e-15,
         -2.773674830645328e-15, 0.105409255338946
        ])


    #> c = chisq.test(freq, p=pr2, rescale.p = TRUE)
    #> cat_items(c, "res2.")
    res2.statistic = 0.01492063492063492
    res2.parameter = 4
    res2.p_value = 0.999972309849908
    res2.method = 'Chi-squared test for given probabilities'
    res2.data_name = 'freq'
    res2.observed = np.array([
         1048, 660, 510, 420, 362
        ])
    res2.expected = np.array([
         1050, 660, 510, 420, 360
        ])
    res2.residuals = np.array([
         -0.06172133998483677, 0, -2.517068894406109e-15,
         -2.773674830645328e-15, 0.105409255338946
        ])

    freq = np.array([1048,  660,  510,  420,  362])
    pr1 = np.array([1020,  690,  510,  420,  360])
    pr2 = np.array([1050,  660,  510,  420,  360])

    for pr, res in zip([pr1, pr2], [res1, res2]):
        stat, pval = chisquare(freq, pr)
        assert_almost_equal(stat, res.statistic, decimal=12)
        assert_almost_equal(pval, res.p_value, decimal=13)


def test_chisquare_effectsize():

    pr1 = np.array([1020,  690,  510,  420,  360])
    pr2 = np.array([1050,  660,  510,  420,  360])
    #> library(pwr)
    #> ES.w1(pr1/3000, pr2/3000)
    es_r = 0.02699815282115563
    es1 = chisquare_effectsize(pr1, pr2)
    es2 = chisquare_effectsize(pr1, pr2, cohen=False)
    assert_almost_equal(es1, es_r, decimal=14)
    assert_almost_equal(es2, es_r**2, decimal=14)

    # regression tests for correction
    res1 = chisquare_effectsize(pr1, pr2, cohen=False,
                                correction=(3000, len(pr1)-1))
    res0 = 0 #-0.00059994422693327625
    assert_equal(res1, res0)
    pr3 = pr2 + [0,0,0,50,50]
    res1 = chisquare_effectsize(pr1, pr3, cohen=False,
                                correction=(3000, len(pr1)-1))
    res0 = 0.0023106468846296755
    assert_almost_equal(res1, res0, decimal=14)
    # compare
    # res_nc = chisquare_effectsize(pr1, pr3, cohen=False)
    # 0.0036681143072077533
