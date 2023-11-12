'''Tests for multipletests and fdr pvalue corrections

Author : Josef Perktold


['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n', 'fdr_tsbh']
are tested against R:multtest

'hommel' is tested against R stats p_adjust (not available in multtest

'fdr_gbs', 'fdr_2sbky' I did not find them in R, currently tested for
    consistency only

'''
import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose)

from statsmodels.stats.multitest import (multipletests, fdrcorrection,
                                         fdrcorrection_twostage,
                                         NullDistribution,
                                         local_fdr, multitest_methods_names)
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version

pval0 = np.array([
    0.838541367553,  0.642193923795,  0.680845947633,
    0.967833824309,  0.71626938238,  0.177096952723,  5.23656777208e-005,
    0.0202732688798,  0.00028140506198,  0.0149877310796])

res_multtest1 = np.array([
    [5.2365677720800003e-05,   5.2365677720800005e-04,
     5.2365677720800005e-04,   5.2365677720800005e-04,
     5.2353339704891422e-04,   5.2353339704891422e-04,
     5.2365677720800005e-04,   1.5337740764175588e-03],
    [2.8140506198000000e-04,   2.8140506197999998e-03,
     2.5326455578199999e-03,   2.5326455578199999e-03,
     2.8104897961789277e-03,   2.5297966317768816e-03,
     1.4070253098999999e-03,   4.1211324652269442e-03],
    [1.4987731079600001e-02,   1.4987731079600000e-01,
     1.1990184863680001e-01,   1.1990184863680001e-01,
     1.4016246580579017e-01,   1.1379719679449507e-01,
     4.9959103598666670e-02,   1.4632862843720582e-01],
    [2.0273268879800001e-02,   2.0273268879799999e-01,
     1.4191288215860001e-01,   1.4191288215860001e-01,
     1.8520270949069695e-01,   1.3356756197485375e-01,
     5.0683172199499998e-02,   1.4844940238274187e-01],
    [1.7709695272300000e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     8.5760763426056130e-01,   6.8947825122356643e-01,
     3.5419390544599999e-01,   1.0000000000000000e+00],
    [6.4219392379499995e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     9.9996560644133570e-01,   9.9413539782557070e-01,
     8.9533672797500008e-01,   1.0000000000000000e+00],
    [6.8084594763299999e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     9.9998903512635740e-01,   9.9413539782557070e-01,
     8.9533672797500008e-01,   1.0000000000000000e+00],
    [7.1626938238000004e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     9.9999661886871472e-01,   9.9413539782557070e-01,
     8.9533672797500008e-01,   1.0000000000000000e+00],
    [8.3854136755300002e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     9.9999998796038225e-01,   9.9413539782557070e-01,
     9.3171263061444454e-01,   1.0000000000000000e+00],
    [9.6783382430900000e-01,   1.0000000000000000e+00,
     1.0000000000000000e+00,   9.6783382430900000e-01,
     9.9999999999999878e-01,   9.9413539782557070e-01,
     9.6783382430900000e-01,   1.0000000000000000e+00]])


res_multtest2_columns = [
    'rawp', 'Bonferroni', 'Holm', 'Hochberg', 'SidakSS', 'SidakSD',
    'BH', 'BY', 'ABH', 'TSBH_0.05']

rmethods = {
    'rawp': (0, 'pval'),
    'Bonferroni': (1, 'b'),
    'Holm': (2, 'h'),
    'Hochberg': (3, 'sh'),
    'SidakSS': (4, 's'),
    'SidakSD': (5, 'hs'),
    'BH': (6, 'fdr_i'),
    'BY': (7, 'fdr_n'),
    'TSBH_0.05': (9, 'fdr_tsbh')
}

NA = np.nan
# all rejections, except for Bonferroni and Sidak
res_multtest2 = np.array([
     0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.012, 0.024, 0.036, 0.048,
     0.06, 0.072, 0.012, 0.02, 0.024, 0.024, 0.024, 0.024, 0.012, 0.012,
     0.012, 0.012, 0.012, 0.012, 0.01194015976019192, 0.02376127616613988,
     0.03546430060660932, 0.04705017875634587, 0.058519850599,
     0.06987425045000606, 0.01194015976019192, 0.01984063872102404,
     0.02378486270400004, 0.023808512, 0.023808512, 0.023808512, 0.012,
     0.012, 0.012, 0.012, 0.012, 0.012, 0.0294, 0.0294, 0.0294, 0.0294,
     0.0294, 0.0294, NA, NA, NA, NA, NA, NA, 0, 0, 0, 0, 0, 0
    ]).reshape(6, 10, order='F')

res_multtest3 = np.array([
     0.001, 0.002, 0.003, 0.004, 0.005, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01,
     0.02, 0.03, 0.04, 0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.018, 0.024,
     0.028, 0.03, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.018, 0.024, 0.028,
     0.03, 0.09, 0.09, 0.09, 0.09, 0.09, 0.00995511979025177,
     0.01982095664805061, 0.02959822305108317, 0.03928762649718986,
     0.04888986953422814, 0.4012630607616213, 0.4613848859051006,
     0.5160176928207072, 0.5656115457763677, 0.6105838818818925,
     0.00995511979025177, 0.0178566699880266, 0.02374950634358763,
     0.02766623106147537, 0.02962749064373438, 0.2262190625000001,
     0.2262190625000001, 0.2262190625000001, 0.2262190625000001,
     0.2262190625000001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.08333333333333334,
     0.0857142857142857, 0.0875, 0.0888888888888889, 0.09,
     0.02928968253968254, 0.02928968253968254, 0.02928968253968254,
     0.02928968253968254, 0.02928968253968254, 0.2440806878306878,
     0.2510544217687075, 0.2562847222222222, 0.2603527336860670,
     0.2636071428571428, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0.005,
     0.005, 0.005, 0.005, 0.005, 0.04166666666666667, 0.04285714285714286,
     0.04375, 0.04444444444444445, 0.045
    ]).reshape(10, 10, order='F')

res0_large = np.array([
     0.00031612, 0.0003965, 0.00048442, 0.00051932, 0.00101436, 0.00121506,
     0.0014516, 0.00265684, 0.00430043, 0.01743686, 0.02080285, 0.02785414,
     0.0327198, 0.03494679, 0.04206808, 0.08067095, 0.23882767, 0.28352304,
     0.36140401, 0.43565145, 0.44866768, 0.45368782, 0.48282088,
     0.49223781, 0.55451638, 0.6207473, 0.71847853, 0.72424145, 0.85950263,
     0.89032747, 0.0094836, 0.011895, 0.0145326, 0.0155796, 0.0304308,
     0.0364518, 0.043548, 0.0797052, 0.1290129, 0.5231058, 0.6240855,
     0.8356242, 0.981594, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 0.0094836, 0.0114985, 0.01356376, 0.01402164, 0.02637336,
     0.0303765, 0.0348384, 0.06110732, 0.09460946, 0.36617406, 0.416057,
     0.52922866, 0.5889564, 0.59409543, 0.67308928, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 0.0094836, 0.0114985, 0.01356376, 0.01402164,
     0.02637336, 0.0303765, 0.0348384, 0.06110732, 0.09460946, 0.36617406,
     0.416057, 0.52922866, 0.5889564, 0.59409543, 0.67308928, 0.89032747,
     0.89032747, 0.89032747, 0.89032747, 0.89032747, 0.89032747,
     0.89032747, 0.89032747, 0.89032747, 0.89032747, 0.89032747,
     0.89032747, 0.89032747, 0.89032747, 0.89032747, 0.009440257627368331,
     0.01182686507401931, 0.01443098172617119, 0.01546285007478554,
     0.02998742566629453, 0.03581680249125385, 0.04264369065603335,
     0.0767094173291795, 0.1212818694859857, 0.410051586220387,
     0.4677640287633493, 0.5715077903157826, 0.631388450393325,
     0.656016359012282, 0.724552174001554, 0.919808283456286,
     0.999721715014484, 0.9999547032674126, 0.9999985652190126,
     0.999999964809746, 0.999999982525548, 0.999999986719131,
     0.999999997434160, 0.999999998521536, 0.999999999970829,
     0.999999999999767, 1, 1, 1, 1, 0.009440257627368331,
     0.01143489901147732, 0.0134754287611275, 0.01392738605848343,
     0.0260416568490015, 0.02993768724817902, 0.0342629726119179,
     0.0593542206208364, 0.09045742964699988, 0.308853956167216,
     0.343245865702423, 0.4153483370083637, 0.4505333180190900,
     0.453775200643535, 0.497247406680671, 0.71681858015803,
     0.978083969553718, 0.986889206426321, 0.995400461639735,
     0.9981506396214986, 0.9981506396214986, 0.9981506396214986,
     0.9981506396214986, 0.9981506396214986, 0.9981506396214986,
     0.9981506396214986, 0.9981506396214986, 0.9981506396214986,
     0.9981506396214986, 0.9981506396214986, 0.0038949, 0.0038949,
     0.0038949, 0.0038949, 0.0060753, 0.0060753, 0.006221142857142857,
     0.00996315, 0.01433476666666667, 0.05231058, 0.05673504545454545,
     0.06963535, 0.07488597857142856, 0.07488597857142856, 0.08413616,
     0.15125803125, 0.421460594117647, 0.4725384, 0.570637910526316,
     0.6152972625, 0.6152972625, 0.6152972625, 0.6152972625, 0.6152972625,
     0.665419656, 0.7162468846153845, 0.775972982142857, 0.775972982142857,
     0.889140651724138, 0.89032747, 0.01556007537622183,
     0.01556007537622183, 0.01556007537622183, 0.01556007537622183,
     0.02427074531648065, 0.02427074531648065, 0.02485338565390302,
     0.0398026560334295, 0.0572672083580799, 0.2089800939109816,
     0.2266557764630925, 0.2781923271071372, 0.2991685206792373,
     0.2991685206792373, 0.336122876445059, 0.6042738882921044, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.00220711, 0.00220711, 0.00220711,
     0.00220711, 0.00344267, 0.00344267, 0.003525314285714285, 0.005645785,
     0.00812303444444444, 0.029642662, 0.0321498590909091,
     0.03946003166666667, 0.04243538785714285, 0.04243538785714285,
     0.0476771573333333, 0.085712884375, 0.23882767, 0.26777176,
     0.323361482631579, 0.34866844875, 0.34866844875, 0.34866844875,
     0.34866844875, 0.34866844875, 0.3770711384, 0.4058732346153846,
     0.4397180232142857, 0.4397180232142857, 0.503846369310345,
     0.504518899666667, 0.00272643, 0.00272643, 0.00272643, 0.00272643,
     0.00425271, 0.00425271, 0.0043548, 0.006974205, 0.01003433666666667,
     0.036617406, 0.03971453181818182, 0.048744745, 0.052420185,
     0.052420185, 0.058895312, 0.105880621875, 0.295022415882353,
     0.33077688, 0.399446537368421, 0.43070808375, 0.43070808375,
     0.43070808375, 0.43070808375, 0.43070808375, 0.4657937592,
     0.5013728192307692, 0.5431810875, 0.5431810875, 0.622398456206897,
     0.623229229
    ]).reshape(30, 10, order='F')


class CheckMultiTestsMixin:

    @pytest.mark.parametrize('key,val', sorted(rmethods.items()))
    def test_multi_pvalcorrection_rmethods(self, key, val):
        # test against R package multtest mt.rawp2adjp

        res_multtest = self.res2
        pval0 = res_multtest[:, 0]

        if val[1] in self.methods:
            reject, pvalscorr = multipletests(pval0,
                                              alpha=self.alpha,
                                              method=val[1])[:2]
            assert_almost_equal(pvalscorr, res_multtest[:, val[0]], 15)
            assert_equal(reject, pvalscorr <= self.alpha)

    def test_multi_pvalcorrection(self):
        # test against R package multtest mt.rawp2adjp

        res_multtest = self.res2
        pval0 = res_multtest[:, 0]

        pvalscorr = np.sort(fdrcorrection(pval0, method='n')[1])
        assert_almost_equal(pvalscorr, res_multtest[:, 7], 15)
        pvalscorr = np.sort(fdrcorrection(pval0, method='i')[1])
        assert_almost_equal(pvalscorr, res_multtest[:, 6], 15)


class TestMultiTests1(CheckMultiTestsMixin):
    @classmethod
    def setup_class(cls):
        cls.methods = ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n']
        cls.alpha = 0.1
        cls.res2 = res_multtest1


class TestMultiTests2(CheckMultiTestsMixin):
    # case: all hypothesis rejected (except 'b' and 's'
    @classmethod
    def setup_class(cls):
        cls.methods = ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n']
        cls.alpha = 0.05
        cls.res2 = res_multtest2


class TestMultiTests3(CheckMultiTestsMixin):
    @classmethod
    def setup_class(cls):
        cls.methods = ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n',
                       'fdr_tsbh']
        cls.alpha = 0.05
        cls.res2 = res0_large


class TestMultiTests4(CheckMultiTestsMixin):
    # in simulations, all two stage fdr, fdr_tsbky, fdr_tsbh, fdr_gbs, have in
    # some cases (cases with large Alternative) an FDR that looks too large
    # this is the first case #rejected = 12, DGP : has 10 false
    @classmethod
    def setup_class(cls):
        cls.methods = ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n',
                       'fdr_tsbh']
        cls.alpha = 0.05
        cls.res2 = res_multtest3


@pytest.mark.parametrize('alpha', [0.01, 0.05, 0.1])
@pytest.mark.parametrize('method', ['b', 's', 'sh', 'hs', 'h', 'hommel',
                                    'fdr_i', 'fdr_n', 'fdr_tsbky',
                                    'fdr_tsbh', 'fdr_gbs'])
@pytest.mark.parametrize('ii', list(range(11)))
def test_pvalcorrection_reject(alpha, method, ii):
    # consistency test for reject boolean and pvalscorr

    pval1 = np.hstack((np.linspace(0.0001, 0.0100, ii),
                       np.linspace(0.05001, 0.11, 10 - ii)))
    # using .05001 instead of 0.05 to avoid edge case issue #768
    reject, pvalscorr = multipletests(pval1, alpha=alpha,
                                      method=method)[:2]

    msg = 'case %s %3.2f rejected:%d\npval_raw=%r\npvalscorr=%r' % (
                     method, alpha, reject.sum(), pval1, pvalscorr)
    assert_equal(reject, pvalscorr <= alpha, err_msg=msg)


def test_hommel():
    # tested against R stats p_adjust(pval0, method='hommel')
    pval0 = np.array([
        0.00116,  0.00924,  0.01075,  0.01437,  0.01784,  0.01918,
        0.02751,  0.02871,  0.03054,  0.03246,  0.04259,  0.06879,
        0.0691,   0.08081,  0.08593,  0.08993,  0.09386,  0.09412,
        0.09718,  0.09758,  0.09781,  0.09788,  0.13282,  0.20191,
        0.21757,  0.24031,  0.26061,  0.26762,  0.29474,  0.32901,
        0.41386,  0.51479,  0.52461,  0.53389,  0.56276,  0.62967,
        0.72178,  0.73403,  0.87182,  0.95384])

    result_ho = np.array([
        0.0464,              0.25872,             0.29025,
        0.3495714285714286,  0.41032,             0.44114,
        0.57771,             0.60291,             0.618954,
        0.6492,              0.7402725000000001,  0.86749,
        0.86749,             0.8889100000000001,  0.8971477777777778,
        0.8993,              0.9175374999999999,  0.9175374999999999,
        0.9175374999999999,  0.9175374999999999,  0.9175374999999999,
        0.9175374999999999,  0.95384,             0.9538400000000001,
        0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
        0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
        0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
        0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
        0.9538400000000001,  0.9538400000000001,  0.9538400000000001,
        0.9538400000000001])

    rej, pvalscorr, _, _ = multipletests(pval0, alpha=0.1, method='ho')
    assert_almost_equal(pvalscorr, result_ho, 15)
    assert_equal(rej, result_ho < 0.1)


def test_fdr_bky():
    # test for fdrcorrection_twostage
    # example from BKY
    pvals = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
        0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    # no test for corrected p-values, but they are inherited
    # same number of rejection as in BKY paper:
    # single step-up:4, two-stage:8, iterated two-step:9
    # also alpha_star is the same as theirs for TST

    # alpha_star for stage 2
    with pytest.warns(FutureWarning, match="iter keyword"):
        res_tst = fdrcorrection_twostage(pvals, alpha=0.05, iter=False)
    assert_almost_equal([0.047619, 0.0649], res_tst[-1][:2], 3)
    assert_equal(8, res_tst[0].sum())

    # reference number from Prism, see #8619
    res2 = np.array([
        0.0012, 0.0023, 0.0073, 0.0274, 0.0464, 0.0492, 0.0492, 0.0497,
        0.0589, 0.3742, 0.4475, 0.5505, 0.5800, 0.6262, 0.77
        ])
    assert_allclose(res_tst[1], res2, atol=6e-5)

    # issue #8619, problems if no or all rejected, ordering
    pvals = np.array([0.2, 0.8, 0.3, 0.5, 1])
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky')
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky')
    assert_equal(res1[0], res2[0])
    assert_allclose(res1[1], res2[1], atol=6e-5)
    # confirmed with Prism
    res_pv = np.array([0.7875, 1., 0.7875, 0.875 , 1.])
    assert_allclose(res1[1], res_pv, atol=6e-5)


def test_fdr_twostage():
    # test for iteration in fdrcorrection_twostage, new maxiter
    # example from BKY
    pvals = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
        0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    n = len(pvals)

    # bh twostage fdr
    k = 0
    # same pvalues as one-stage fdr
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bh', maxiter=k,
                                  iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbh', maxiter=k)
    assert_allclose(res1[1], res0[1])
    assert_allclose(res2[1], res1[1])

    k = 1
    # pvalues corrected by first stage number of rejections
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bh', maxiter=k,
                                  iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbh', maxiter=k)
    res3 = multipletests(pvals, alpha=0.05, method='fdr_tsbh')
    assert_allclose(res1[1], res0[1] * (1 - res0[0].sum() / n))
    assert_allclose(res2[1], res1[1])
    assert_allclose(res3[1], res1[1])  # check default maxiter

    # bky has an extra factor 1+alpha in fdr twostage independent of iter
    fact = 1 + 0.05
    k = 0
    # same pvalues as one-stage fdr
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky', maxiter=k,
                                  iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky', maxiter=k)
    assert_allclose(res1[1], np.clip(res0[1] * fact, 0, 1))
    assert_allclose(res2[1], res1[1])

    k = 1
    # pvalues corrected by first stage number of rejections
    res0 = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res1 = fdrcorrection_twostage(pvals, alpha=0.05, method='bky', maxiter=k,
                                  iter=None)
    res2 = multipletests(pvals, alpha=0.05, method='fdr_tsbky', maxiter=k)
    res3 = multipletests(pvals, alpha=0.05, method='fdr_tsbky')
    assert_allclose(res1[1], res0[1] * (1 - res0[0].sum() / n) * fact)
    assert_allclose(res2[1], res1[1])
    assert_allclose(res3[1], res1[1])  # check default maxiter


@pytest.mark.parametrize('method', sorted(multitest_methods_names))
def test_issorted(method):
    # test that is_sorted keyword works correctly
    # the fdrcorrection functions are tested indirectly

    # data generated as random numbers np.random.beta(0.2, 0.5, size=10)
    pvals = np.array([31, 9958111, 7430818, 8653643, 9892855, 876, 2651691,
                      145836, 9931, 6174747]) * 1e-7
    sortind = np.argsort(pvals)
    sortrevind = sortind.argsort()
    pvals_sorted = pvals[sortind]

    res1 = multipletests(pvals, method=method, is_sorted=False)
    res2 = multipletests(pvals_sorted, method=method, is_sorted=True)
    assert_equal(res2[0][sortrevind], res1[0])
    assert_allclose(res2[0][sortrevind], res1[0], rtol=1e-10)


@pytest.mark.parametrize('method', sorted(multitest_methods_names))
def test_floating_precision(method):
    # issue #7465
    pvals = np.full(6000, 0.99)
    pvals[0] = 1.138569e-56
    assert multipletests(pvals, method=method)[1][0] > 1e-60


def test_tukeyhsd():
    # example multicomp in R p 83

    res = '''\
    pair      diff        lwr        upr       p adj
    P-M   8.150000 -10.037586 26.3375861 0.670063958
    S-M  -3.258333 -21.445919 14.9292527 0.982419709
    T-M  23.808333   5.620747 41.9959194 0.006783701
    V-M   4.791667 -13.395919 22.9792527 0.931020848
    S-P -11.408333 -29.595919  6.7792527 0.360680099
    T-P  15.658333  -2.529253 33.8459194 0.113221634
    V-P  -3.358333 -21.545919 14.8292527 0.980350080
    T-S  27.066667   8.879081 45.2542527 0.002027122
    V-S   8.050000 -10.137586 26.2375861 0.679824487
    V-T -19.016667 -37.204253 -0.8290806 0.037710044
    '''

    res = np.array([
        [8.150000,  -10.037586, 26.3375861, 0.670063958],
        [-3.258333,  -21.445919, 14.9292527, 0.982419709],
        [23.808333,    5.620747, 41.9959194, 0.006783701],
        [4.791667,  -13.395919, 22.9792527, 0.931020848],
        [-11.408333, -29.595919,  6.7792527, 0.360680099],
        [15.658333,  -2.529253,  33.8459194, 0.113221634],
        [-3.358333, -21.545919,  14.8292527, 0.980350080],
        [27.066667,   8.879081,  45.2542527, 0.002027122],
        [8.050000, -10.137586,  26.2375861, 0.679824487],
        [-19.016667, -37.204253, -0.8290806, 0.037710044]])

    m_r = [94.39167, 102.54167,  91.13333, 118.20000,  99.18333]
    myres = tukeyhsd(m_r, 6, 110.8254416667, alpha=0.05, df=4)
    pairs, reject, meandiffs, std_pairs, confint, q_crit = myres[:6]
    assert_almost_equal(meandiffs, res[:, 0], decimal=5)
    assert_almost_equal(confint, res[:, 1:3], decimal=2)
    assert_equal(reject, res[:, 3] < 0.05)

    # check p-values (divergence of high values is expected)
    small_pvals_idx = [2, 5, 7, 9]

    # Remove this check when minimum SciPy version is 1.7+ (gh-8035)
    scipy_version = (version.parse(scipy.version.version) >=
                     version.parse('1.7.0'))
    rtol = 1e-5 if scipy_version else 1e-2
    assert_allclose(myres[8][small_pvals_idx], res[small_pvals_idx, 3],
                    rtol=rtol)


def test_local_fdr():

    # Create a mixed population of Z-scores: 1000 standard normal and
    # 20 uniformly distributed between 3 and 4.
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))

    # Exact local FDR for U(3, 4) component.
    f1 = np.exp(-z1**2 / 2) / np.sqrt(2*np.pi)
    r = len(z1) / float(len(z0) + len(z1))
    f1 /= (1 - r) * f1 + r

    for alpha in None, 0, 1e-8:
        if alpha is None:
            fdr = local_fdr(zs)
        else:
            fdr = local_fdr(zs, alpha=alpha)
        fdr1 = fdr[len(z0):]
        assert_allclose(f1, fdr1, rtol=0.05, atol=0.1)


def test_null_distribution():

    # Create a mixed population of Z-scores: 1000 standard normal and
    # 20 uniformly distributed between 3 and 4.
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))
    emp_null = NullDistribution(zs, estimate_null_proportion=True)

    assert_allclose(emp_null.mean, 0, atol=1e-5, rtol=1e-5)
    assert_allclose(emp_null.sd, 1, atol=1e-5, rtol=1e-2)
    assert_allclose(emp_null.null_proportion, 0.98, atol=1e-5, rtol=1e-2)

    # consistency check
    assert_allclose(emp_null.pdf(np.r_[-1, 0, 1]),
                    norm.pdf(np.r_[-1, 0, 1],
                             loc=emp_null.mean, scale=emp_null.sd),
                    rtol=1e-13)


@pytest.mark.parametrize('estimate_prob', [True, False])
@pytest.mark.parametrize('estimate_scale', [True, False])
@pytest.mark.parametrize('estimate_mean', [True, False])
def test_null_constrained(estimate_mean, estimate_scale, estimate_prob):

    # Create a mixed population of Z-scores: 1000 standard normal and
    # 20 uniformly distributed between 3 and 4.
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))

    emp_null = NullDistribution(zs, estimate_mean=estimate_mean,
                                estimate_scale=estimate_scale,
                                estimate_null_proportion=estimate_prob)

    if not estimate_mean:
        assert_allclose(emp_null.mean, 0, atol=1e-5, rtol=1e-5)
    if not estimate_scale:
        assert_allclose(emp_null.sd, 1, atol=1e-5, rtol=1e-2)
    if not estimate_prob:
        assert_allclose(emp_null.null_proportion, 1, atol=1e-5, rtol=1e-2)

    # consistency check
    assert_allclose(emp_null.pdf(np.r_[-1, 0, 1]),
                    norm.pdf(np.r_[-1, 0, 1], loc=emp_null.mean,
                             scale=emp_null.sd),
                    rtol=1e-13)
