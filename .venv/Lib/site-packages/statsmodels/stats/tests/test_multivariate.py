# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:48:19 2017

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa

from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder


def test_mv_mean():
    # names = ['id', 'mpg1', 'mpg2', 'add']
    x = np.asarray([[1.0, 24.0, 23.5, 1.0],
                    [2.0, 25.0, 24.5, 1.0],
                    [3.0, 21.0, 20.5, 1.0],
                    [4.0, 22.0, 20.5, 1.0],
                    [5.0, 23.0, 22.5, 1.0],
                    [6.0, 18.0, 16.5, 1.0],
                    [7.0, 17.0, 16.5, 1.0],
                    [8.0, 28.0, 27.5, 1.0],
                    [9.0, 24.0, 23.5, 1.0],
                    [10.0, 27.0, 25.5, 1.0],
                    [11.0, 21.0, 20.5, 1.0],
                    [12.0, 23.0, 22.5, 1.0],
                    [1.0, 20.0, 19.0, 0.0],
                    [2.0, 23.0, 22.0, 0.0],
                    [3.0, 21.0, 20.0, 0.0],
                    [4.0, 25.0, 24.0, 0.0],
                    [5.0, 18.0, 17.0, 0.0],
                    [6.0, 17.0, 16.0, 0.0],
                    [7.0, 18.0, 17.0, 0.0],
                    [8.0, 24.0, 23.0, 0.0],
                    [9.0, 20.0, 19.0, 0.0],
                    [10.0, 24.0, 22.0, 0.0],
                    [11.0, 23.0, 22.0, 0.0],
                    [12.0, 19.0, 18.0, 0.0]])

    res = smmv.test_mvmean(x[:, 1:3], [21, 21])

    res_stata = Holder(p_F=1.25062334808e-09,
                       df_r=22,
                       df_m=2,
                       F=59.91609589041116,
                       T2=125.2791095890415)

    assert_allclose(res.statistic, res_stata.F, rtol=1e-10)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-10)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])

    # diff of paired sample
    mask = x[:, -1] == 1
    x1 = x[mask, 1:3]
    x0 = x[~mask, 1:3]
    res_p = smmv.test_mvmean(x1 - x0, [0, 0])

    # result Stata hotelling
    res_stata = Holder(T2=9.698067632850247,
                       df=10,
                       k=2,
                       N=12,
                       F=4.4082126,  # not in return List
                       p_F=0.0424)  # not in return List

    res = res_p
    assert_allclose(res.statistic, res_stata.F, atol=5e-7)
    assert_allclose(res.pvalue, res_stata.p_F, atol=5e-4)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.k, res_stata.df])

    # mvtest means diff1 diff2, zero
    res_stata = Holder(p_F=.0423949782937231,
                       df_r=10,
                       df_m=2,
                       F=4.408212560386478,
                       T2=9.69806763285025)

    assert_allclose(res.statistic, res_stata.F, rtol=1e-12)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-12)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-12)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])

    dw = weightstats.DescrStatsW(x)
    ci0 = dw.tconfint_mean(alpha=0.05)

    nobs = len(x[:, 1:])
    ci1 = confint_mvmean_fromstats(dw.mean, np.diag(dw.var), nobs,
                                   lin_transf=np.eye(4), alpha=0.05)
    ci2 = confint_mvmean_fromstats(dw.mean, dw.cov, nobs,
                                   lin_transf=np.eye(4), alpha=0.05)

    assert_allclose(ci1[:2], ci0, rtol=1e-13)
    assert_allclose(ci2[:2], ci0, rtol=1e-13)

    # test from data
    res = smmv.confint_mvmean(x, lin_transf=np.eye(4), alpha=0.05)
    assert_allclose(res, ci2, rtol=1e-13)


def test_mvmean_2indep():
    x = np.asarray([[1.0, 24.0, 23.5, 1.0],
                    [2.0, 25.0, 24.5, 1.0],
                    [3.0, 21.0, 20.5, 1.0],
                    [4.0, 22.0, 20.5, 1.0],
                    [5.0, 23.0, 22.5, 1.0],
                    [6.0, 18.0, 16.5, 1.0],
                    [7.0, 17.0, 16.5, 1.0],
                    [8.0, 28.0, 27.5, 1.0],
                    [9.0, 24.0, 23.5, 1.0],
                    [10.0, 27.0, 25.5, 1.0],
                    [11.0, 21.0, 20.5, 1.0],
                    [12.0, 23.0, 22.5, 1.0],
                    [1.0, 20.0, 19.0, 0.0],
                    [2.0, 23.0, 22.0, 0.0],
                    [3.0, 21.0, 20.0, 0.0],
                    [4.0, 25.0, 24.0, 0.0],
                    [5.0, 18.0, 17.0, 0.0],
                    [6.0, 17.0, 16.0, 0.0],
                    [7.0, 18.0, 17.0, 0.0],
                    [8.0, 24.0, 23.0, 0.0],
                    [9.0, 20.0, 19.0, 0.0],
                    [10.0, 24.0, 22.0, 0.0],
                    [11.0, 23.0, 22.0, 0.0],
                    [12.0, 19.0, 18.0, 0.0]])

    y = np.asarray([[1.1, 24.1, 23.4, 1.1],
                    [1.9, 25.2, 24.3, 1.2],
                    [3.2, 20.9, 20.2, 1.3],
                    [4.1, 21.8, 20.6, 0.9],
                    [5.2, 23.0, 22.7, 0.8],
                    [6.3, 18.1, 16.8, 0.7],
                    [7.1, 17.2, 16.5, 1.0],
                    [7.8, 28.3, 27.4, 1.1],
                    [9.5, 23.9, 23.3, 1.2],
                    [10.1, 26.8, 25.2, 1.3],
                    [10.5, 26.7, 20.6, 0.9],
                    [12.1, 23.0, 22.7, 0.8],
                    [1.1, 20.1, 19.0, 0.7],
                    [1.8, 23.2, 22.0, 0.1],
                    [3.2, 21.3, 20.3, 0.2],
                    [4.3, 24.9, 24.2, 0.3],
                    [5.5, 17.9, 17.1, 0.0],
                    [5.5, 17.8, 16.0, 0.6],
                    [7.1, 17.7, 16.7, 0.0],
                    [7.7, 24.0, 22.8, 0.5],
                    [9.1, 20.1, 18.9, 0.0],
                    [10.2, 24.2, 22.3, 0.3],
                    [11.3, 23.3, 22.2, 0.0],
                    [11.7, 18.8, 18.1, 0.1]])

    res = smmv.test_mvmean_2indep(x, y)

    res_stata = Holder(p_F=0.6686659171701677,
                       df_r=43,
                       df_m=4,
                       F=0.594263378678938,
                       T2=2.5428944576028973)

    assert_allclose(res.statistic, res_stata.F, rtol=1e-10)
    assert_allclose(res.pvalue, res_stata.p_F, rtol=1e-10)
    assert_allclose(res.t2, res_stata.T2, rtol=1e-10)
    assert_equal(res.df, [res_stata.df_m, res_stata.df_r])


def test_confint_simult():
    # example from book for simultaneous confint

    m = [526.29, 54.69, 25.13]
    cov = [[5808.06, 597.84, 222.03],
           [597.84, 126.05, 23.39],
           [222.03, 23.39, 23.11]]
    nobs = 87
    res_ci = confint_mvmean_fromstats(m, cov, nobs, lin_transf=np.eye(3),
                                      simult=True)

    cii = [confint_mvmean_fromstats(
                m, cov, nobs, lin_transf=np.eye(3)[i], simult=True)[:2]
           for i in range(3)]
    cii = np.array(cii).squeeze()
    # these might use rounded numbers in intermediate computation
    res_ci_book = np.array([[503.06, 550.12], [51.22, 58.16], [23.65, 26.61]])

    assert_allclose(res_ci[0], res_ci_book[:, 0], rtol=1e-3)  # low
    assert_allclose(res_ci[0], res_ci_book[:, 0], rtol=1e-3)  # upp

    assert_allclose(res_ci[0], cii[:, 0], rtol=1e-13)
    assert_allclose(res_ci[1], cii[:, 1], rtol=1e-13)

    res_constr = confint_mvmean_fromstats(m, cov, nobs, lin_transf=[0, 1, -1],
                                          simult=True)

    assert_allclose(res_constr[0], 29.56 - 3.12, rtol=1e-3)
    assert_allclose(res_constr[1], 29.56 + 3.12, rtol=1e-3)

    # TODO: this assumes separate constraints,
    #       but we want multiplicity correction
    # test if several constraints or transformations work
    # original, flipping sign, multiply by 2
    lt = [[0, 1, -1], [0, -1, 1], [0, 2, -2]]
    res_constr2 = confint_mvmean_fromstats(m, cov, nobs, lin_transf=lt,
                                           simult=True)

    lows = res_constr[0], - res_constr[1], 2 * res_constr[0]
    upps = res_constr[1], - res_constr[0], 2 * res_constr[1]
    # TODO: check return dimensions
    lows = np.asarray(lows).squeeze()
    upps = np.asarray(upps).squeeze()
    assert_allclose(res_constr2[0], lows, rtol=1e-13)
    assert_allclose(res_constr2[1], upps, rtol=1e-13)


class TestCovStructure:

    @classmethod
    def setup_class(cls):
        # computed from data with ``cov = np.cov(dta1, rowvar=0, ddof=1)``
        cls.cov = np.array(
            [[28.965925000000002, 17.215358333333327, 2.6945666666666654],
             [17.215358333333327, 21.452852666666672, 6.044527833333332],
             [2.6945666666666654, 6.044527833333332, 13.599042333333331]])
        cls.nobs = 25

    def test_spherical(self):
        cov, nobs = self.cov, self.nobs
        # from Stata 14
        p_chi2 = 0.0006422366870356
        # df = 5
        chi2 = 21.53275509455011

        stat, pv = smmv.test_cov_spherical(cov, nobs)
        assert_allclose(stat, chi2, rtol=1e-7)
        assert_allclose(pv, p_chi2, rtol=1e-6)

    def test_diagonal(self):
        cov, nobs = self.cov, self.nobs
        # from Stata 14
        p_chi2 = 0.0004589987613319
        # df = 3
        chi2 = 17.91025335733012

        stat, pv = smmv.test_cov_diagonal(cov, nobs)
        assert_allclose(stat, chi2, rtol=1e-8)
        assert_allclose(pv, p_chi2, rtol=1e-7)

    def test_blockdiagonal(self):
        cov, nobs = self.cov, self.nobs
        # from Stata 14
        p_chi2 = 0.1721758850671037
        # df = 2
        chi2 = 3.518477474111563

        # cov_blocks = cov[:2, :2], cov[-1:, -1:]
        # stat, pv = smmv.test_cov_blockdiagonal(cov, nobs, cov_blocks)
        block_len = [2, 1]
        stat, pv = smmv.test_cov_blockdiagonal(cov, nobs, block_len)
        assert_allclose(stat, chi2, rtol=1e-7)
        assert_allclose(pv, p_chi2, rtol=1e-6)

    def test_covmat(self):
        cov, nobs = self.cov, self.nobs
        # from Stata 14
        p_chi2 = 0.4837049015162541
        # df = 6
        chi2 = 5.481422374989864

        cov_null = np.array([[30, 15, 0], [15, 20, 0], [0, 0, 10]])
        stat, pv = smmv.test_cov(cov, nobs, cov_null)
        assert_allclose(stat, chi2, rtol=1e-7)
        assert_allclose(pv, p_chi2, rtol=1e-6)


def test_cov_oneway():
    # from Stata 14
    p_chi2 = .1944866419800838
    chi2 = 13.55075120374669
    df = 10
    p_F_Box = .1949865290585139
    df_r_Box = 18377.68924302788
    df_m_Box = 10
    F_Box = 1.354282822767436

    nobs = [32, 32]
    cov_m = np.array(
        [[5.192540322580645, 4.545362903225806, 6.522177419354839, 5.25],
         [4.545362903225806, 13.184475806451612, 6.76008064516129,
          6.266129032258064],
         [6.522177419354839, 6.76008064516129, 28.673387096774192,
          14.46774193548387],
         [5.25, 6.266129032258064, 14.46774193548387, 16.64516129032258]])
    cov_f = np.array(
        [[9.13608870967742, 7.549395161290322, 4.86391129032258,
          4.151209677419355],
         [7.549395161290322, 18.60383064516129, 10.224798387096774,
          5.445564516129032],
         [4.86391129032258, 10.224798387096774, 30.039314516129032,
          13.493951612903226],
         [4.151209677419355, 5.445564516129032, 13.493951612903226,
          27.995967741935484]])

    res = smmv.test_cov_oneway([cov_m, cov_f], nobs)
    stat, pv = res
    assert_allclose(stat, F_Box, rtol=1e-10)
    assert_allclose(pv, p_F_Box, rtol=1e-6)
    assert_allclose(res.statistic_f, F_Box, rtol=1e-10)
    assert_allclose(res.pvalue_f, p_F_Box, rtol=1e-6)
    assert_allclose(res.df_f, (df_m_Box, df_r_Box), rtol=1e-13)

    assert_allclose(res.statistic_chi2, chi2, rtol=1e-10)
    assert_allclose(res.pvalue_chi2, p_chi2, rtol=1e-6)
    assert_equal(res.df_chi2, df)
