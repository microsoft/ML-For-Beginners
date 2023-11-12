# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:13:59 2020

Author: Josef Perktold
License: BSD-3

"""

from scipy import stats
from numpy.testing import assert_allclose

from statsmodels.stats.effect_size import (
        _noncentrality_chisquare, _noncentrality_f, _noncentrality_t)


def test_noncent_chi2():
    # > lochi(7.5,2,.95)
    # [1] 0.03349255 0.97499458
    # > hichi(7.5,2,.95)
    # [1] 20.76049805 0.02500663

    chi2_stat, df = 7.5, 2
    ci_nc = [0.03349255, 20.76049805]
    res = _noncentrality_chisquare(chi2_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    # verify umvue unbiased
    mean = stats.ncx2.mean(df, res.nc)
    assert_allclose(chi2_stat, mean, rtol=1e-8)

    assert_allclose(stats.ncx2.cdf(chi2_stat, df, res.confint), [0.975, 0.025],
                    rtol=1e-8)


def test_noncent_f():
    # F(4, 75) = 3.5, confidence level = .95, two-sided CI:
    # > lof(3.5,4,75,.95)
    # [1] 0.7781436 0.9750039
    # > hif(3.5,4,75,.95)
    # [1] 29.72949219 0.02499965
    f_stat, df1, df2 = 3.5, 4, 75

    ci_nc = [0.7781436, 29.72949219]
    res = _noncentrality_f(f_stat, df1, df2, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    # verify umvue unbiased
    mean = stats.ncf.mean(df1, df2, res.nc)
    assert_allclose(f_stat, mean, rtol=1e-8)

    # Relax tolerance due to changes in SciPy and Boost
    assert_allclose(stats.ncf.cdf(f_stat, df1, df2, res.confint),
                    [0.975, 0.025], rtol=5e-5)


def test_noncent_t():
    # t(98) = 1.5, confidence level = .95, two-sided CI:
    # > lot(1.5,98,.95)
    # [1] -0.4749756 0.9750024
    # > hit(1.5,98,.95)
    # [1] 3.467285 0.025005

    # > conf.limits.nct(1.5,98,.95)
    #  Lower.Limit Prob.Low.Limit Upper.Limit Prob.Up.Limit
    # Values -0.474934 0.975 3.467371 0.02499999

    t_stat, df = 1.5, 98

    ci_nc = [-0.474934, 3.467371]
    res = _noncentrality_t(t_stat, df, alpha=0.05)
    assert_allclose(res.confint, ci_nc, rtol=0.005)
    # verify umvue unbiased
    mean = stats.nct.mean(df, res.nc)
    assert_allclose(t_stat, mean, rtol=1e-8)

    # Tolerancee relaxed due to Boost integration in SciPy
    assert_allclose(stats.nct.cdf(t_stat, df, res.confint), [0.975, 0.025],
                    rtol=1e-6)
