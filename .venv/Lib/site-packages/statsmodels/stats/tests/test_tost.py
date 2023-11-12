# -*- coding: utf-8 -*-
"""

Created on Wed Oct 17 09:48:34 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest

import statsmodels.stats.weightstats as smws


from statsmodels.tools.testing import Holder


def assert_almost_equal_inf(x, y, decimal=6, msg=None):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    assert_equal(np.isposinf(x), np.isposinf(y))
    assert_equal(np.isneginf(x), np.isneginf(y))
    assert_equal(np.isnan(x), np.isnan(y))
    assert_almost_equal(x[np.isfinite(x)], y[np.isfinite(y)])


raw_clinic = '''\
1     1 2.84 4.00 3.45 2.55 2.46
2     1 2.51 3.26 3.10 2.82 2.48
3     1 2.41 4.14 3.37 2.99 3.04
4     1 2.95 3.42 2.82 3.37 3.35
5     1 3.14 3.25 3.31 2.87 3.41
6     1 3.79 4.34 3.88 3.40 3.16
7     1 4.14 4.97 4.25 3.43 3.06
8     1 3.85 4.31 3.92 3.58 3.91
9     1 3.02 3.11 2.20 2.24 2.28
10    1 3.45 3.41 3.80 3.86 3.91
11    1 5.37 5.02 4.59 3.99 4.27
12    1 3.81 4.21 4.08 3.18 1.86
13    1 4.19 4.59 4.79 4.17 2.60
14    1 3.16 5.30 4.69 4.83 4.51
15    1 3.84 4.32 4.25 3.87 2.93
16    2 2.60 3.76 2.86 2.41 2.71
17    2 2.82 3.66 3.20 2.49 2.49
18    2 2.18 3.65 3.87 3.00 2.65
19    2 3.46 3.60 2.97 1.80 1.74
20    2 4.01 3.48 4.42 3.06 2.76
21    2 3.04 2.87 2.87 2.71 2.87
22    2 3.47 3.24 3.47 3.26 3.14
23    2 4.06 3.92 3.18 3.06 1.74
24    2 2.91 3.99 3.06 2.02 3.18
25    2 3.59 4.21 4.02 3.26 2.85
26    2 4.51 4.21 3.78 2.63 1.92
27    2 3.16 3.31 3.28 3.25 3.52
28    2 3.86 3.61 3.28 3.19 3.09
29    2 3.31 2.97 3.76 3.18 2.60
30    2 3.02 2.73 3.87 3.50 2.93'''.split()
clinic = np.array(raw_clinic, float).reshape(-1,7)


#t = tost(-clinic$var2[16:30] + clinic$var2[1:15], eps=0.6)
tost_clinic_paired = Holder()
tost_clinic_paired.sample = 'paired'
tost_clinic_paired.mean_diff = 0.5626666666666665
tost_clinic_paired.se_diff = 0.2478276410785118
tost_clinic_paired.alpha = 0.05
tost_clinic_paired.ci_diff = (0.1261653305099018, 0.999168002823431)
tost_clinic_paired.df = 14
tost_clinic_paired.epsilon = 0.6
tost_clinic_paired.result = 'not rejected'
tost_clinic_paired.p_value = 0.4412034046017588
tost_clinic_paired.check_me = (0.525333333333333, 0.6)

#> t = tost(-clinic$var1[16:30] + clinic$var1[1:15], eps=0.6)
#> cat_items(t, prefix="tost_clinic_paired_1.")
tost_clinic_paired_1 = Holder()
tost_clinic_paired_1.mean_diff = 0.1646666666666667
tost_clinic_paired_1.se_diff = 0.1357514067862445
tost_clinic_paired_1.alpha = 0.05
tost_clinic_paired_1.ci_diff = (-0.0744336620516462, 0.4037669953849797)
tost_clinic_paired_1.df = 14
tost_clinic_paired_1.epsilon = 0.6
tost_clinic_paired_1.result = 'rejected'
tost_clinic_paired_1.p_value = 0.003166881489265175
tost_clinic_paired_1.check_me = (-0.2706666666666674, 0.600000000000001)


#> t = tost(clinic$var2[1:15], clinic$var2[16:30], eps=0.6)
#> cat_items(t, prefix="tost_clinic_indep.")
tost_clinic_indep = Holder()
tost_clinic_indep.sample = 'independent'
tost_clinic_indep.mean_diff = 0.562666666666666
tost_clinic_indep.se_diff = 0.2149871904637392
tost_clinic_indep.alpha = 0.05
tost_clinic_indep.ci_diff = (0.194916250699966, 0.930417082633366)
tost_clinic_indep.df = 24.11000151062728
tost_clinic_indep.epsilon = 0.6
tost_clinic_indep.result = 'not rejected'
tost_clinic_indep.p_value = 0.4317936812594803
tost_clinic_indep.check_me = (0.525333333333332, 0.6)

#> t = tost(clinic$var1[1:15], clinic$var1[16:30], eps=0.6)
#> cat_items(t, prefix="tost_clinic_indep_1.")
tost_clinic_indep_1 = Holder()
tost_clinic_indep_1.sample = 'independent'
tost_clinic_indep_1.mean_diff = 0.1646666666666667
tost_clinic_indep_1.se_diff = 0.2531625991083627
tost_clinic_indep_1.alpha = 0.05
tost_clinic_indep_1.ci_diff = (-0.2666862980722534, 0.596019631405587)
tost_clinic_indep_1.df = 26.7484787582315
tost_clinic_indep_1.epsilon = 0.6
tost_clinic_indep_1.result = 'rejected'
tost_clinic_indep_1.p_value = 0.04853083976236974
tost_clinic_indep_1.check_me = (-0.2706666666666666, 0.6)

#pooled variance
#> t = tost(clinic$var1[1:15], clinic$var1[16:30], eps=0.6, var.equal = TRUE)
#> cat_items(t, prefix="tost_clinic_indep_1_pooled.")
tost_clinic_indep_1_pooled = Holder()
tost_clinic_indep_1_pooled.mean_diff = 0.1646666666666667
tost_clinic_indep_1_pooled.se_diff = 0.2531625991083628
tost_clinic_indep_1_pooled.alpha = 0.05
tost_clinic_indep_1_pooled.ci_diff = (-0.2659960620757337, 0.595329395409067)
tost_clinic_indep_1_pooled.df = 28
tost_clinic_indep_1_pooled.epsilon = 0.6
tost_clinic_indep_1_pooled.result = 'rejected'
tost_clinic_indep_1_pooled.p_value = 0.04827315100761467
tost_clinic_indep_1_pooled.check_me = (-0.2706666666666666, 0.6)

#> t = tost(clinic$var2[1:15], clinic$var2[16:30], eps=0.6, var.equal = TRUE)
#> cat_items(t, prefix="tost_clinic_indep_2_pooled.")
tost_clinic_indep_2_pooled = Holder()
tost_clinic_indep_2_pooled.mean_diff = 0.562666666666666
tost_clinic_indep_2_pooled.se_diff = 0.2149871904637392
tost_clinic_indep_2_pooled.alpha = 0.05
tost_clinic_indep_2_pooled.ci_diff = (0.1969453064978777, 0.928388026835454)
tost_clinic_indep_2_pooled.df = 28
tost_clinic_indep_2_pooled.epsilon = 0.6
tost_clinic_indep_2_pooled.result = 'not rejected'
tost_clinic_indep_2_pooled.p_value = 0.43169347692374
tost_clinic_indep_2_pooled.check_me = (0.525333333333332, 0.6)


#tost ratio, log transformed
#> t = tost(log(clinic$var1[1:15]), log(clinic$var1[16:30]), eps=log(1.25), paired=TRUE)
#> cat_items(t, prefix="tost_clinic_1_paired.")
tost_clinic_1_paired = Holder()
tost_clinic_1_paired.mean_diff = 0.0431223318225235
tost_clinic_1_paired.se_diff = 0.03819576328421437
tost_clinic_1_paired.alpha = 0.05
tost_clinic_1_paired.ci_diff = (-0.02415225319362176, 0.1103969168386687)
tost_clinic_1_paired.df = 14
tost_clinic_1_paired.epsilon = 0.2231435513142098
tost_clinic_1_paired.result = 'rejected'
tost_clinic_1_paired.p_value = 0.0001664157928976468
tost_clinic_1_paired.check_me = (-0.1368988876691603, 0.2231435513142073)

#> t = tost(log(clinic$var1[1:15]), log(clinic$var1[16:30]), eps=log(1.25), paired=FALSE)
#> cat_items(t, prefix="tost_clinic_1_indep.")
tost_clinic_1_indep = Holder()
tost_clinic_1_indep.mean_diff = 0.04312233182252334
tost_clinic_1_indep.se_diff = 0.073508371131806
tost_clinic_1_indep.alpha = 0.05
tost_clinic_1_indep.ci_diff = (-0.0819851930203655, 0.1682298566654122)
tost_clinic_1_indep.df = 27.61177037646526
tost_clinic_1_indep.epsilon = 0.2231435513142098
tost_clinic_1_indep.result = 'rejected'
tost_clinic_1_indep.p_value = 0.01047085593138891
tost_clinic_1_indep.check_me = (-0.1368988876691633, 0.22314355131421)

#> t = tost(log(y), log(x), eps=log(1.25), paired=TRUE)
#> cat_items(t, prefix="tost_s_paired.")
tost_s_paired = Holder()
tost_s_paired.mean_diff = 0.06060076667771316
tost_s_paired.se_diff = 0.04805826005366752
tost_s_paired.alpha = 0.05
tost_s_paired.ci_diff = (-0.0257063329659993, 0.1469078663214256)
tost_s_paired.df = 11
tost_s_paired.epsilon = 0.2231435513142098
tost_s_paired.result = 'rejected'
tost_s_paired.p_value = 0.003059338540563293
tost_s_paired.check_me = (-0.1019420179587835, 0.2231435513142098)

#multiple endpoints
#> compvall <- multeq.diff(data=clinic,grp="fact",method="step.up",margin.up=rep(0.6,5), margin.lo=c(-1.0, -1.0, -1.5, -1.5, -1.5))
#> cat_items(compvall, prefix="tost_clinic_all_no_multi.")
tost_clinic_all_no_multi = Holder()
tost_clinic_all_no_multi.comp_name = '2-1'
tost_clinic_all_no_multi.estimate = np.array([
     -0.1646666666666667, -0.562666666666666, -0.3073333333333332,
     -0.5553333333333335, -0.469333333333333])
tost_clinic_all_no_multi.degr_fr = np.array([
     26.74847875823152, 24.1100015106273, 23.90046331918926,
     25.71678948210178, 24.88436709341423])
tost_clinic_all_no_multi.test_stat = np.array([
     3.020456692101513, 2.034229724989578, 4.052967897750272,
     4.37537447933403, 4.321997343344])
tost_clinic_all_no_multi.p_value = np.array([
     0.00274867705173331, 0.02653543052872217, 0.0002319468040526358,
     8.916466517494902e-05, 0.00010890038649094043])
tost_clinic_all_no_multi.lower = np.array([
     -0.596019631405587, -0.930417082633366, -0.690410573009442,
     -0.92373513818557, -0.876746448909633])
tost_clinic_all_no_multi.upper = np.array([
     0.2666862980722534, -0.194916250699966, 0.07574390634277595,
     -0.186931528481097, -0.06192021775703377])
tost_clinic_all_no_multi.margin_lo = np.array([
     -1, -1, -1.5, -1.5, -1.5])
tost_clinic_all_no_multi.margin_up = np.array([
     0.6, 0.6, 0.6, 0.6, 0.6])
tost_clinic_all_no_multi.base = 1
tost_clinic_all_no_multi.method = 'step.up'
tost_clinic_all_no_multi.var_equal = '''FALSE'''
tost_clinic_all_no_multi.FWER = 0.05



#> comp <- multeq.diff(data=clinic,grp="fact", resp=c("var1"),method="step.up",margin.up=rep(0.6), margin.lo=rep(-1.5))
#> cat_items(comp, prefix="tost_clinic_1_asym.")
tost_clinic_1_asym = Holder
tost_clinic_1_asym.comp_name = '2-1'
tost_clinic_1_asym.estimate = -0.1646666666666667
tost_clinic_1_asym.degr_fr = 26.74847875823152
tost_clinic_1_asym.test_stat = 3.020456692101513
tost_clinic_1_asym.p_value = 0.00274867705173331
tost_clinic_1_asym.lower = -0.596019631405587
tost_clinic_1_asym.upper = 0.2666862980722534
tost_clinic_1_asym.margin_lo = -1.5
tost_clinic_1_asym.margin_up = 0.6
tost_clinic_1_asym.base = 1
tost_clinic_1_asym.method = 'step.up'
tost_clinic_1_asym.var_equal = '''FALSE'''
tost_clinic_1_asym.FWER = 0.05

#TODO: not used yet, some p-values are multi-testing adjusted
#      not implemented
#> compvall <- multeq.diff(data=clinic,grp="fact",method="step.up",margin.up=rep(0.6,5), margin.lo=c(-0.5, -0.5, -1.5, -1.5, -1.5))
#> cat_items(compvall, prefix="tost_clinic_all_multi.")
tost_clinic_all_multi = Holder()
tost_clinic_all_multi.comp_name = '2-1'
tost_clinic_all_multi.estimate = np.array([
     -0.1646666666666667, -0.562666666666666, -0.3073333333333332,
     -0.5553333333333335, -0.469333333333333])
tost_clinic_all_multi.degr_fr = np.array([
     26.74847875823152, 24.1100015106273, 23.90046331918926,
     25.71678948210178, 24.88436709341423])
tost_clinic_all_multi.test_stat = np.array([
     1.324576910311299, -0.2914902349832590, 4.052967897750272,
     4.37537447933403, 4.321997343344])
tost_clinic_all_multi.p_value = np.array([
     0.0982588867413542, 0.6134151998456164, 0.0006958404121579073,
     0.0002674939955248471, 0.0003267011594728213])
tost_clinic_all_multi.lower = np.array([
     -0.596019631405587, -0.930417082633366, -0.812901144055456,
     -1.040823983574101, -1.006578759345919])
tost_clinic_all_multi.upper = np.array([
     0.2666862980722534, -0.194916250699966, 0.1982344773887895,
     -0.0698426830925655, 0.0679120926792529])
tost_clinic_all_multi.margin_lo = np.array([
     -0.5, -0.5, -1.5, -1.5, -1.5])
tost_clinic_all_multi.margin_up = np.array([
     0.6, 0.6, 0.6, 0.6, 0.6])
tost_clinic_all_multi.base = 1
tost_clinic_all_multi.method = 'step.up'
tost_clinic_all_multi.var_equal = '''FALSE'''
tost_clinic_all_multi.FWER = 0.05


#t-tests

#> tt = t.test(clinic$var1[16:30], clinic$var1[1:15], data=clinic, mu=-0., alternative="two.sided", paired=TRUE)
#> cat_items(tt, prefix="ttest_clinic_paired_1.")
ttest_clinic_paired_1 = Holder()
ttest_clinic_paired_1.statistic = 1.213001548676048
ttest_clinic_paired_1.parameter = 14
ttest_clinic_paired_1.p_value = 0.245199929713149
ttest_clinic_paired_1.conf_int = (-0.1264911434745851, 0.4558244768079186)
ttest_clinic_paired_1.estimate = 0.1646666666666667
ttest_clinic_paired_1.null_value = 0
ttest_clinic_paired_1.alternative = 'two.sided'
ttest_clinic_paired_1.method = 'Paired t-test'
ttest_clinic_paired_1.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'



#> ttless = t.test(clinic$var1[1:15], clinic$var1[16:30],, data=clinic, mu=-0., alternative="less", paired=FALSE)
#> cat_items(ttless, prefix="ttest_clinic_paired_1_l.")
ttest_clinic_paired_1_l = Holder()
ttest_clinic_paired_1_l.statistic = 0.650438363512706
ttest_clinic_paired_1_l.parameter = 26.7484787582315
ttest_clinic_paired_1_l.p_value = 0.739521349864458
ttest_clinic_paired_1_l.conf_int = (-np.inf, 0.596019631405587)
ttest_clinic_paired_1_l.estimate = (3.498, 3.333333333333333)
ttest_clinic_paired_1_l.null_value = 0
ttest_clinic_paired_1_l.alternative = 'less'
ttest_clinic_paired_1_l.method = 'Welch Two Sample t-test'
ttest_clinic_paired_1_l.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'

#> cat_items(tt, prefix="ttest_clinic_indep_1_g.")
ttest_clinic_indep_1_g = Holder()
ttest_clinic_indep_1_g.statistic = 0.650438363512706
ttest_clinic_indep_1_g.parameter = 26.7484787582315
ttest_clinic_indep_1_g.p_value = 0.2604786501355416
ttest_clinic_indep_1_g.conf_int = (-0.2666862980722534, np.inf)
ttest_clinic_indep_1_g.estimate = (3.498, 3.333333333333333)
ttest_clinic_indep_1_g.null_value = 0
ttest_clinic_indep_1_g.alternative = 'greater'
ttest_clinic_indep_1_g.method = 'Welch Two Sample t-test'
ttest_clinic_indep_1_g.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'

#> cat_items(ttless, prefix="ttest_clinic_indep_1_l.")
ttest_clinic_indep_1_l = Holder()
ttest_clinic_indep_1_l.statistic = 0.650438363512706
ttest_clinic_indep_1_l.parameter = 26.7484787582315
ttest_clinic_indep_1_l.p_value = 0.739521349864458
ttest_clinic_indep_1_l.conf_int = (-np.inf, 0.596019631405587)
ttest_clinic_indep_1_l.estimate = (3.498, 3.333333333333333)
ttest_clinic_indep_1_l.null_value = 0
ttest_clinic_indep_1_l.alternative = 'less'
ttest_clinic_indep_1_l.method = 'Welch Two Sample t-test'
ttest_clinic_indep_1_l.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'

#> ttless = t.test(clinic$var1[1:15], clinic$var1[16:30],, data=clinic, mu=1., alternative="less", paired=FALSE)
#> cat_items(ttless, prefix="ttest_clinic_indep_1_l_mu.")
ttest_clinic_indep_1_l_mu = Holder()
ttest_clinic_indep_1_l_mu.statistic = -3.299592184135306
ttest_clinic_indep_1_l_mu.parameter = 26.7484787582315
ttest_clinic_indep_1_l_mu.p_value = 0.001372434925571605
ttest_clinic_indep_1_l_mu.conf_int = (-np.inf, 0.596019631405587)
ttest_clinic_indep_1_l_mu.estimate = (3.498, 3.333333333333333)
ttest_clinic_indep_1_l_mu.null_value = 1
ttest_clinic_indep_1_l_mu.alternative = 'less'
ttest_clinic_indep_1_l_mu.method = 'Welch Two Sample t-test'
ttest_clinic_indep_1_l_mu.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'


#> tt2 = t.test(clinic$var1[1:15], clinic$var1[16:30],, data=clinic, mu=1, alternative="two.sided", paired=FALSE)
#> cat_items(tt2, prefix="ttest_clinic_indep_1_two_mu.")
ttest_clinic_indep_1_two_mu = Holder()
ttest_clinic_indep_1_two_mu.statistic = -3.299592184135306
ttest_clinic_indep_1_two_mu.parameter = 26.7484787582315
ttest_clinic_indep_1_two_mu.p_value = 0.00274486985114321
ttest_clinic_indep_1_two_mu.conf_int = (-0.3550087243406, 0.6843420576739336)
ttest_clinic_indep_1_two_mu.estimate = (3.498, 3.333333333333333)
ttest_clinic_indep_1_two_mu.null_value = 1
ttest_clinic_indep_1_two_mu.alternative = 'two.sided'
ttest_clinic_indep_1_two_mu.method = 'Welch Two Sample t-test'
ttest_clinic_indep_1_two_mu.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'

#> tt2 = t.test(clinic$var1[1:15], clinic$var1[16:30],, data=clinic, mu=1, alternative="two.sided", paired=FALSE, var.equal=TRUE)
#> cat_items(tt2, prefix="ttest_clinic_indep_1_two_mu_pooled.")
ttest_clinic_indep_1_two_mu_pooled = Holder()
ttest_clinic_indep_1_two_mu_pooled.statistic = -3.299592184135305
ttest_clinic_indep_1_two_mu_pooled.parameter = 28
ttest_clinic_indep_1_two_mu_pooled.p_value = 0.002643203760742494
ttest_clinic_indep_1_two_mu_pooled.conf_int = (-0.35391340938235, 0.6832467427156834)
ttest_clinic_indep_1_two_mu_pooled.estimate = (3.498, 3.333333333333333)
ttest_clinic_indep_1_two_mu_pooled.null_value = 1
ttest_clinic_indep_1_two_mu_pooled.alternative = 'two.sided'
ttest_clinic_indep_1_two_mu_pooled.method = ' Two Sample t-test'
ttest_clinic_indep_1_two_mu_pooled.data_name = 'clinic$var1[1:15] and clinic$var1[16:30]'




res1 = smws.ttost_paired(clinic[:15, 2], clinic[15:, 2], -0.6, 0.6, transform=None)
res2 = smws.ttost_paired(clinic[:15, 3], clinic[15:, 3], -0.6, 0.6, transform=None)
res = smws.ttost_ind(clinic[:15, 3], clinic[15:, 3], -0.6, 0.6, usevar='unequal')


class CheckTostMixin:

    def test_pval(self):
        assert_almost_equal(self.res1.pvalue, self.res2.p_value, decimal=13)
        #assert_almost_equal(self.res1.df, self.res2.df, decimal=13)

class TestTostp1(CheckTostMixin):
    #paired var1
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_paired_1
        x1, x2 = clinic[:15, 2], clinic[15:, 2]
        cls.res1 = Holder()
        res = smws.ttost_paired(x1, x2, -0.6, 0.6, transform=None)
        cls.res1.pvalue = res[0]
        #cls.res1.df = res[1][-1] not yet
        res_ds = smws.DescrStatsW(x1 - x2, weights=None, ddof=0)
        #tost confint 2*alpha TODO: check again
        cls.res1.tconfint_diff = res_ds.tconfint_mean(0.1)
        cls.res1.confint_05 = res_ds.tconfint_mean(0.05)
        cls.res1.mean_diff = res_ds.mean
        cls.res1.std_mean_diff = res_ds.std_mean

        cls.res2b = ttest_clinic_paired_1

    def test_special(self):
        #TODO: add attributes to other cases and move to superclass
        assert_almost_equal(self.res1.tconfint_diff, self.res2.ci_diff,
                            decimal=13)
        assert_almost_equal(self.res1.mean_diff, self.res2.mean_diff,
                            decimal=13)
        assert_almost_equal(self.res1.std_mean_diff, self.res2.se_diff,
                            decimal=13)
        #compare with ttest
        assert_almost_equal(self.res1.confint_05, self.res2b.conf_int,
                            decimal=13)


class TestTostp2(CheckTostMixin):
    #paired var2
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_paired
        x, y = clinic[:15, 3], clinic[15:, 3]
        cls.res1 = Holder()
        res = smws.ttost_paired(x, y, -0.6, 0.6, transform=None)
        cls.res1.pvalue = res[0]

class TestTosti1(CheckTostMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep_1
        x, y = clinic[:15, 2], clinic[15:, 2]
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='unequal')
        cls.res1.pvalue = res[0]

class TestTosti2(CheckTostMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep
        x, y = clinic[:15, 3], clinic[15:, 3]
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='unequal')
        cls.res1.pvalue = res[0]

class TestTostip1(CheckTostMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep_1_pooled
        x, y = clinic[:15, 2], clinic[15:, 2]
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='pooled')
        cls.res1.pvalue = res[0]

class TestTostip2(CheckTostMixin):
    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep_2_pooled
        x, y = clinic[:15, 3], clinic[15:, 3]
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='pooled')
        cls.res1.pvalue = res[0]

#transform=np.log
#class TestTostp1_log(CheckTost):
def test_tost_log():
    x1, x2 = clinic[:15, 2], clinic[15:, 2]

    resp = smws.ttost_paired(x1, x2, 0.8, 1.25, transform=np.log)
    assert_almost_equal(resp[0], tost_clinic_1_paired.p_value, 13)

    resi = smws.ttost_ind(x1, x2, 0.8, 1.25, transform=np.log, usevar='unequal')
    assert_almost_equal(resi[0], tost_clinic_1_indep.p_value, 13)

def test_tost_asym():
    x1, x2 = clinic[:15, 2], clinic[15:, 2]
    #Note: x1, x2 reversed by definition in multeq.dif
    assert_almost_equal(x2.mean() - x1.mean(), tost_clinic_1_asym.estimate, 13)
    resa = smws.ttost_ind(x2, x1, -1.5, 0.6, usevar='unequal')
    assert_almost_equal(resa[0], tost_clinic_1_asym.p_value, 13)

    #multi-endpoints, asymmetric bounds, vectorized
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7],
                           [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6,
                           usevar='unequal')
    assert_almost_equal(resall[0], tost_clinic_all_no_multi.p_value, 13)

    #SMOKE tests: foe multi-endpoint vectorized, k on k
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7],
                           np.exp([-1.0, -1.0, -1.5, -1.5, -1.5]), 0.6,
                           usevar='unequal', transform=np.log)
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7],
                           [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6,
                           usevar='unequal', transform=np.exp)

    resall = smws.ttost_paired(clinic[15:, 2:7], clinic[:15, 2:7],
                              [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6,
                              transform=np.log)
    resall = smws.ttost_paired(clinic[15:, 2:7], clinic[:15, 2:7],
                              [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6,
                              transform=np.exp)

    resall = smws.ttest_ind(clinic[15:, 2:7], clinic[:15, 2:7],
                              value=[-1.0, -1.0, -1.5, -1.5, -1.5])

    #k on 1: compare all with reference
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:3],
                           [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal')
    resa3_2 = smws.ttost_ind(clinic[15:, 3:4], clinic[:15, 2:3],
                           [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal')
    assert_almost_equal(resall[0][1], resa3_2[0][1], decimal=13)
    resall = smws.ttost_ind(clinic[15:, 2], clinic[:15, 2],
                           [-1.0, -0.5, -0.7, -1.5, -1.5], 0.6, usevar='unequal')
    resall = smws.ttost_ind(clinic[15:, 2], clinic[:15, 2],
                           [-1.0, -0.5, -0.7, -1.5, -1.5],
                           np.repeat(0.6,5), usevar='unequal')

def test_ttest():
    x1, x2 = clinic[:15, 2], clinic[15:, 2]
    all_tests = []
    t1 = smws.ttest_ind(x1, x2, alternative='larger', usevar='unequal')
    all_tests.append((t1, ttest_clinic_indep_1_g))
    t2 = smws.ttest_ind(x1, x2, alternative='smaller', usevar='unequal')
    all_tests.append((t2, ttest_clinic_indep_1_l))
    t3 = smws.ttest_ind(x1, x2, alternative='smaller', usevar='unequal',
                        value=1)
    all_tests.append((t3, ttest_clinic_indep_1_l_mu))

    for res1, res2 in all_tests:
        assert_almost_equal(res1[0], res2.statistic, decimal=13)
        assert_almost_equal(res1[1], res2.p_value, decimal=13)
        #assert_almost_equal(res1[2], res2.df, decimal=13)

    cm = smws.CompareMeans(smws.DescrStatsW(x1), smws.DescrStatsW(x2))
    ci = cm.tconfint_diff(alternative='two-sided', usevar='unequal')
    assert_almost_equal(ci, ttest_clinic_indep_1_two_mu.conf_int, decimal=13)
    ci = cm.tconfint_diff(alternative='two-sided', usevar='pooled')
    assert_almost_equal(ci, ttest_clinic_indep_1_two_mu_pooled.conf_int, decimal=13)
    ci = cm.tconfint_diff(alternative='smaller', usevar='unequal')
    assert_almost_equal_inf(ci, ttest_clinic_indep_1_l.conf_int, decimal=13)
    ci = cm.tconfint_diff(alternative='larger', usevar='unequal')
    assert_almost_equal_inf(ci, ttest_clinic_indep_1_g.conf_int, decimal=13)


    #test get_compare
    cm = smws.CompareMeans(smws.DescrStatsW(x1), smws.DescrStatsW(x2))
    cm1 = cm.d1.get_compare(cm.d2)
    cm2 = cm.d1.get_compare(x2)
    cm3 = cm.d1.get_compare(np.hstack((x2,x2)))
    #all use the same d1, no copying
    assert_(cm.d1 is cm1.d1)
    assert_(cm.d1 is cm2.d1)
    assert_(cm.d1 is cm3.d1)


@pytest.mark.xfail(reason="shape mismatch between res1[1:] and res_sas[1:]",
                   raises=AssertionError, strict=True)
def test_tost_transform_paired():
    raw = np.array('''\
       103.4 90.11  59.92 77.71  68.17 77.71  94.54 97.51
       69.48 58.21  72.17 101.3  74.37 79.84  84.44 96.06
       96.74 89.30  94.26 97.22  48.52 61.62  95.68 85.80'''.split(), float)

    x, y = raw.reshape(-1,2).T

    res1 = smws.ttost_paired(x, y, 0.8, 1.25, transform=np.log)
    res_sas = (0.0031, (3.38, 0.0031), (-5.90, 0.00005))
    assert_almost_equal(res1[0], res_sas[0], 3)
    assert_almost_equal(res1[1:], res_sas[1:], 2)
    #result R tost
    assert_almost_equal(res1[0], tost_s_paired.p_value, 13)
