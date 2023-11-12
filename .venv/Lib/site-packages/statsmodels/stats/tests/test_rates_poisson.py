

import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats

# we cannot import test_poisson_2indep directly, pytest treats that as test
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
    # test_poisson, # cannot import functions that start with test
    confint_poisson,
    tolerance_int_poisson,
    confint_quantile_poisson,
    etest_poisson_2indep,
    confint_poisson_2indep,
    nonequivalence_poisson_2indep,
    power_poisson_ratio_2indep,
    power_equivalence_poisson_2indep,
    power_poisson_diff_2indep,
    power_equivalence_neginb_2indep,
    power_negbin_ratio_2indep,
    method_names_poisson_1samp,
    method_names_poisson_2indep,
    )

methods = ["wald", "score", "exact-c", "waldccv", "sqrt-a", "sqrt-v", "midp-c",
           "sqrt",
           ]


@pytest.mark.parametrize('method', methods)
def test_rate_poisson_consistency(method):
    # check consistency between test and confint for one poisson rate
    count, nobs = 15, 400
    ci = confint_poisson(count, nobs, method=method)
    pv1 = smr.test_poisson(count, nobs, value=ci[0], method=method).pvalue
    pv2 = smr.test_poisson(count, nobs, value=ci[1], method=method).pvalue

    rtol = 1e-10
    if method in ["midp-c"]:
        # numerical root finding, lower precision
        rtol = 1e-6
    assert_allclose(pv1, 0.05, rtol=rtol)
    assert_allclose(pv2, 0.05, rtol=rtol)

    # check one-sided, note all confint are central
    pv1 = smr.test_poisson(count, nobs, value=ci[0], method=method,
                           alternative="larger").pvalue
    pv2 = smr.test_poisson(count, nobs, value=ci[1], method=method,
                           alternative="smaller").pvalue

    assert_allclose(pv1, 0.025, rtol=rtol)
    assert_allclose(pv2, 0.025, rtol=rtol)


def test_rate_poisson_r():
    count, nobs = 15, 400

    # DescTools
    # PoissonCI(x=15, n=400, method = c("exact","score", "wald", "byar"))
    # exact 0.0375 0.0209884653319583 0.0618505471787146
    # score 0.0375 0.0227264749053794 0.0618771721463559
    # wald  0.0375 0.0185227303217751 0.0564772696782249
    # byar  0.0375 0.0219084369245707 0.0602933510943048

    # > rr = poisson.exact(15, 400, r=0.05, tsmethod="central")
    # > rr$p.value
    # > rr$conf.int
    pv2 = 0.313026269279486
    ci2 = (0.0209884653319583, 0.0618505471787146)
    rt = smr.test_poisson(count, nobs, value=0.05, method="exact-c")
    ci = confint_poisson(count, nobs, method="exact-c")
    assert_allclose(rt.pvalue, pv2, rtol=1e-12)
    assert_allclose(ci, ci2, rtol=1e-12)

    # R package ratesci
    # sr = scoreci(15, 400, contrast="p", distrib = "poi", skew=FALSE,
    #      theta0=0.05)
    # pval computed from 2 * min(pval_left, pval_right)
    pv2 = 0.263552477282973
    # ci2 = (0.022726, 0.061877)  # no additional digits available
    ci2 = (0.0227264749053794, 0.0618771721463559)  # from DescTools
    rt = smr.test_poisson(count, nobs, value=0.05, method="score")
    ci = confint_poisson(count, nobs, method="score")
    assert_allclose(rt.pvalue, pv2, rtol=1e-12)
    assert_allclose(ci, ci2, rtol=1e-12)

    # > jeffreysci(15, 400, distrib = "poi")
    ci2 = (0.0219234232268444, 0.0602898619930649)
    ci = confint_poisson(count, nobs, method="jeff")
    assert_allclose(ci, ci2, rtol=1e-12)

    # from DescTools
    ci2 = (0.0185227303217751, 0.0564772696782249)
    ci = confint_poisson(count, nobs, method="wald")
    assert_allclose(ci, ci2, rtol=1e-12)

    # from ratesci
    # rateci(15, 400, distrib = "poi")
    # I think their lower ci is wrong, it also doesn't match for exact_c
    ci2 = (0.0243357599260795, 0.0604627555786095)
    ci = confint_poisson(count, nobs, method="midp-c")
    assert_allclose(ci[1], ci2[1], rtol=1e-5)


cases_tolint = [
    ("wald", 15, 1, 1, (3, 32), (3, np.inf), (0, 31)),
    ("score", 15, 1, 1, (4, 35), (4, np.inf), (0, 33)),
    ("wald", 150, 100, 100, (104, 200), (108, np.inf), (0, 196)),
    ("score", 150, 100, 100, (106, 202), (109, np.inf), (0, 198)),
    ("exact-c", 150, 100, 100, (105, 202), (109, np.inf), (0, 198)),
    ]


@pytest.mark.parametrize('case', cases_tolint)
def test_tol_int(case):
    # results values are from R package `tolerance, e.g.
    # poistol.int(15, 1, m = 1, alpha = 0.05, P = 0.975, side = 1, method="LS")
    # poistol.int(150, 100, m = 100, alpha = 0.05, P = 0.95, side = 2,
    #     method="SC")
    # exact-c is method="TAB"

    prob = 0.95
    prob_one = 0.975
    meth, count, exposure, exposure_new, r2, rs, rl = case
    ti = tolerance_int_poisson(
        count, exposure, prob, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="two-sided")
    assert_equal(ti, r2)
    ti = tolerance_int_poisson(
        count, exposure, prob_one, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="larger")
    assert_equal(ti, rl)
    ti = tolerance_int_poisson(
        count, exposure, prob_one, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="smaller")
    assert_equal(ti, rs)

    # check without parameter uncertainty, confint at alpha=1
    if meth not in ["exact-c"]:
        # confint for "exact-c" does not collapse to point if alpha=1, check
        ti = tolerance_int_poisson(
            count, exposure, prob, exposure_new=exposure_new,
            method=meth, alpha=0.99999, alternative="two-sided")
        ci = stats.poisson.interval(prob, count / exposure * exposure_new)
        assert_equal(ti, ci)

    # check confint_quantile_poisson,
    # should same as upper tol_int
    ciq = confint_quantile_poisson(
        count, exposure, prob_one, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="two-sided")

    assert_equal(ciq[1], r2[1])

    ciq = confint_quantile_poisson(
        count, exposure, prob_one, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="larger")

    assert_equal(ciq[1], rl[1])

    # should same as lower tol_int
    prob_low = 0.025
    ciq = confint_quantile_poisson(
        count, exposure, prob_low, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="two-sided")

    assert_equal(ciq[0], r2[0])

    ciq = confint_quantile_poisson(
        count, exposure, prob_low, exposure_new=exposure_new,
        method=meth, alpha=0.05, alternative="smaller")

    assert_equal(ciq[0], rs[0])


class TestMethodsCompar1samp():
    # check that all methods for test and confint produce similar results

    @pytest.mark.parametrize('meth', method_names_poisson_1samp["test"])
    def test_test(self, meth):
        count1, n1 = 60, 514.775
        tst = smr.test_poisson(count1, n1, method=meth, value=0.1,
                               alternative='two-sided')

        assert_allclose(tst.pvalue, 0.25, rtol=0.1)

    @pytest.mark.parametrize('meth', method_names_poisson_1samp["confint"])
    def test_confint(self, meth):
        count1, n1 = 60, 514.775
        ci = confint_poisson(count1, n1, method=meth, alpha=0.05)
        assert_allclose(ci, [0.089, 0.158], rtol=0.1)


# ######## below are 2indep tests


methods_diff = ["wald", "score", "waldccv",
                ]


@pytest.mark.parametrize('method', methods_diff)
def test_rate_poisson_diff_consistency(method):
    # check consistency between test and confint for diff of poisson rates
    count1, n1, count2, n2 = 30, 400 / 10, 7, 300 / 10   # avoid low=0
    ci = confint_poisson_2indep(count1, n1, count2, n2, method=method,
                                compare="diff")
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0],
                                  method=method, compare="diff").pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1],
                                  method=method, compare="diff").pvalue

    rtol = 1e-10
    if method in ["score"]:
        # numerical root finding, lower precision
        rtol = 1e-6
    assert_allclose(pv1, 0.05, rtol=rtol)
    assert_allclose(pv2, 0.05, rtol=rtol)

    # check one-sided, note all confint are central
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0],
                                  method=method,
                                  compare="diff",
                                  alternative="larger").pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1],
                                  method=method,
                                  compare="diff",
                                  alternative="smaller").pvalue

    assert_allclose(pv1, 0.025, rtol=rtol)
    assert_allclose(pv2, 0.025, rtol=rtol)


methods_ratio = ["wald-log", "score-log",  # "waldccv",
                 ]


@pytest.mark.parametrize('method', methods_ratio)
def test_rate_poisson_ratio_consistency(method):
    compare = "ratio"
    # check consistency between test and confint for ratio of poisson rates
    count1, n1, count2, n2 = 30, 400 / 10, 7, 300 / 10   # avoid low=0
    ci = confint_poisson_2indep(count1, n1, count2, n2, method=method,
                                compare=compare)
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0],
                                  method=method, compare=compare).pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1],
                                  method=method, compare=compare).pvalue

    rtol = 1e-10
    if method in ["score", "score-log"]:
        # numerical root finding, lower precision
        rtol = 1e-6
    assert_allclose(pv1, 0.05, rtol=rtol)
    assert_allclose(pv2, 0.05, rtol=rtol)

    # check one-sided, note all confint are central
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0],
                                  method=method,
                                  compare=compare,
                                  alternative="larger").pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1],
                                  method=method,
                                  compare=compare,
                                  alternative="smaller").pvalue

    assert_allclose(pv1, 0.025, rtol=rtol)
    assert_allclose(pv2, 0.025, rtol=rtol)


methods_diff_ratio = ["wald", "score", "etest", "etest-wald",
                      ]


@pytest.mark.parametrize('method', methods_diff_ratio)
def test_rate_poisson_diff_ratio_consistency(method):
    # check consistency between test for rate and diff for equality null
    count1, n1, count2, n2 = 30, 400 / 10, 7, 300 / 10   # avoid low=0
    t1 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="ratio")
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="diff")

    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)

    t1 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="ratio",
                                 alternative="larger")
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="diff",
                                 alternative="larger")

    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)

    t1 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="ratio",
                                 alternative="smaller")
    t2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                 method=method, compare="diff",
                                 alternative="smaller")

    assert_allclose(t1.tuple, t2.tuple, rtol=1e-13)


def test_twosample_poisson():
    # testing against two examples in Gu et al

    # example 1
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-6)
    assert_allclose(s1, 3.384913, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.417402, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='wald-log')
    pv2r = 0.000420
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.3393, atol=0, rtol=5e-6)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='score-log')
    pv2r = 0.000200
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.5406, atol=0, rtol=5e-5)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-6)
    assert_allclose(s2, 3.445485, atol=0, rtol=5e-6)  # regression test

    # two-sided
    # example2
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      value=1.5)
    pv1r = 0.2309
    assert_allclose(pv1, pv1r*2, rtol=0, atol=5e-4)
    assert_allclose(s1, 0.735447, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      value=1.5)
    pv2r = 0.2398
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-4)
    assert_allclose(s2, 0.706631, atol=0, rtol=5e-6)  # regression test

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='wald-log', value=1.5)
    pv2r = 0.2402
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-4)
    assert_allclose(s2, 0.7056, atol=0, rtol=5e-4)

    with pytest.warns(FutureWarning):
        s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                          method='score-log', ratio_null=1.5)
    pv2r = 0.2303
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-4)
    assert_allclose(s2, 0.7380, atol=0, rtol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      value=1.5)
    pv2r = 0.2499
    assert_allclose(pv2, pv2r*2, rtol=0, atol=5e-3)
    assert_allclose(s2, 0.674401, atol=0, rtol=5e-6)  # regression test

    # one-sided
    # example 1 onesided
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      alternative='larger')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-6)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      alternative='larger')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      alternative='larger')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-6)

    # 'exact-cond', 'cond-midp'

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='exact-cond',
                                      value=1, alternative='larger')
    pv2r = 0.000428  # typo in Gu et al, switched pvalues between C and M
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='cond-midp',
                                      value=1, alternative='larger')
    pv2r = 0.000310
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='score',
                                   alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)

    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='wald',
                                   alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=5e-4)

    # example2 onesided
    # I don't know why it's only 2.5 decimal agreement, rounding?
    count1, n1, count2, n2 = 41, 28010, 15, 19017
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald',
                                      value=1.5, alternative='larger')
    pv1r = 0.2309
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score',
                                      value=1.5, alternative='larger')
    pv2r = 0.2398
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt',
                                      value=1.5, alternative='larger')
    pv2r = 0.2499
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    # 'exact-cond', 'cond-midp'

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='exact-cond',
                                      value=1.5, alternative='larger')
    pv2r = 0.2913
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2,
                                      method='cond-midp',
                                      value=1.5, alternative='larger')
    pv2r = 0.2450
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-4)

    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='score',
                                   value=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)

    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2,
                                   method='wald',
                                   value=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=5e-4)


cases_diff_ng = [
    ("wald",  (2.2047, 0.0137), (1.5514, 0.06040)),
    ("score", (2.0818, 0.0187), (1.5023, 0.06651)),
    ("etest-wald",  (2.2047, 0.0184), (1.5514, 0.06626)),
    # Ng reports 0.07088 at value=0.0002, looks strange
    ("etest-score", (2.0818, 0.0179), (1.5023, 0.06626)),  # 0.07088)),
    ]


@pytest.mark.parametrize('case', cases_diff_ng)
def test_twosample_poisson_diff(case):
    # testing against two examples Ng et al 2007
    # methods_diff = ["wald", "score", "etest-wald", "etest"]

    meth, res1, res2 = case
    count1, exposure1, count2, exposure2 = 41, 28010, 15, 19017

    value = 0
    t = smr.test_poisson_2indep(count1, exposure1, count2, exposure2,
                                value=value,
                                method=meth, compare='diff',
                                alternative='larger', etest_kwds=None)
    assert_allclose((t.statistic, t.pvalue), res1, atol=0.0006)

    value = 0.0002
    t = smr.test_poisson_2indep(count1, exposure1, count2, exposure2,
                                value=value,
                                method=meth, compare='diff',
                                alternative='larger', etest_kwds=None)
    assert_allclose((t.statistic, t.pvalue), res2, atol=0.0007)


def test_twosample_poisson_r():
    # testing against R package `exactci
    from .results.results_rates import res_pexact_cond, res_pexact_cond_midp

    # example 1 from Gu
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7

    res2 = res_pexact_cond
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond')
    assert_allclose(res1.pvalue, res2.p_value, rtol=1e-13)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)

    res2 = res_pexact_cond_midp
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp')
    assert_allclose(res1.pvalue, res2.p_value, rtol=0, atol=5e-6)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)

    # one-sided
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #                      alternative="less", tsmethod="minlike", midp=TRUE)
    # > pe$p.value
    pv2 = 0.9949053964701466
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp',
                                   value=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #           alternative="greater", tsmethod="minlike", midp=TRUE)
    # > pe$p.value
    pv2 = 0.005094603529853279
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp',
                                   value=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #           alternative="greater", tsmethod="minlike", midp=FALSE)
    # > pe$p.value
    pv2 = 0.006651774552714537
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond',
                                   value=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    # > pe = poisson.exact(c(60, 30), c(51477.5, 54308.7), r=1.2,
    #                      alternative="less", tsmethod="minlike", midp=FALSE)
    # > pe$p.value
    pv2 = 0.9964625674930079
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond',
                                   value=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)


def test_confint_poisson_2indep():
    # test other confint methods

    count1, exposure1, count2, exposure2 = 60, 51477.5, 30, 54308.7

    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='mover', compare='ratio', alpha=0.1,
                                method_mover="jeff",
                                )
    ci1 = (1.4667, 3.0608)  # LTW 2014
    assert_allclose(ci, ci1, atol=0.05)
    # moverci(60, 51477.5, 30, 54308.7, type = "jeff", distrib="poi",
    # contrast="RR", cc=0, level=0.9)
    #   Lower Estimate    Upper
    # est 1.466768 2.104135 3.058634
    # ratesci uses different center for mover-jeffreys
    ci1 = (1.466768, 3.058634)
    assert_allclose(ci, ci1, rtol=0.001)

    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='mover', compare='ratio', alpha=0.1,
                                method_mover="score",
                                )
    ci1 = (1.4611, 3.0424)  # LTW 2014
    assert_allclose(ci, ci1, atol=0.05)

    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='waldcc', compare='ratio', alpha=0.1,
                                )
    ci1 = (1.4523, 3.0154)  # LTW 2014
    assert_allclose(ci, ci1, atol=0.0005)

    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='score', compare='ratio', alpha=0.05,
                                )
    ci1 = (1.365962, 3.259306)
    assert_allclose(ci, ci1, atol=5e-6)

    # with diff
    # use larger rates for diff
    exposure1 /= 1000
    exposure2 /= 1000
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='mover', compare='diff', alpha=0.05,
                                method_mover="jeff",
                                )
    # moverci(60, 51477.5/1000, 30, 54308.7/1000, type = "jeff", distrib="poi",
    # contrast="RD", cc=0, level=0.95)
    ci1 = (0.2629322, 0.9786493)
    assert_allclose(ci, ci1, atol=0.005)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2,
                                method='score', compare='diff', alpha=0.05,
                                )
    # scoreci(60, 51477.5/1000, 30, 54308.7/1000, distrib="poi", contrast="RD",
    # skew=FALSE, cc=0, level=0.95)
    ci1 = (0.265796, 0.989192)
    assert_allclose(ci, ci1, atol=5e-6)

    # example in Li et al 2011, scaled
    ci = confint_poisson_2indep(count2, exposure2, count1, exposure1,
                                method='mover', compare='diff', alpha=0.1,
                                method_mover="jeff",
                                )
    # moverci(30, 54308.7/1000, 60, 51477.5/1000, type = "jeff", distrib="poi",
    # contrast="RD", cc=0, level=0.90)
    ci1 = (-0.9183272231752, -0.3188611692202)
    assert_allclose(ci, ci1, atol=0.005)
    ci1 = (-0.9195, -0.3193)  # from Li et al 2011
    assert_allclose(ci, ci1, atol=0.005)

    ci = confint_poisson_2indep(count2, exposure2, count1, exposure1,
                                method='mover', compare='diff', alpha=0.1,
                                method_mover="jeff",
                                )
    # scoreci(30, 54308.7/1000, 60, 51477.5/1000, distrib="poi", contrast="RD",
    # skew=FALSE, cc=0, level=0.90)
    ci1 = (-0.9232, -0.3188)  # from Li et al 2011
    assert_allclose(ci, ci1, atol=0.006)


def test_tost_poisson():

    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7
    # # central conf_int from R exactci
    low, upp = 1.339735721772650, 3.388365573616252

    res = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp,
                                  method="exact-cond")

    assert_allclose(res.pvalue, 0.025, rtol=1e-12)
    methods = ['wald', 'score', 'sqrt', 'exact-cond', 'cond-midp']

    # test that we are in the correct range for other methods
    for meth in methods:
        res = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp,
                                      method=meth)
        assert_allclose(res.pvalue, 0.025, atol=0.01)


cases_alt = {
    ("two-sided", "wald"): 0.07136366497984171,
    ("two-sided", "score"): 0.0840167525117227,
    ("two-sided", "sqrt"): 0.0804675114297235,
    ("two-sided", "exact-cond"): 0.1301269270479679,
    ("two-sided", "cond-midp"): 0.09324590196774807,
    ("two-sided", "etest"): 0.09054824785458056,
    ("two-sided", "etest-wald"): 0.06895289560607239,

    ("larger", "wald"): 0.03568183248992086,
    ("larger", "score"): 0.04200837625586135,
    ("larger", "sqrt"): 0.04023375571486175,
    ("larger", "exact-cond"): 0.08570447732927276,
    ("larger", "cond-midp"): 0.04882345224905293,
    ("larger", "etest"): 0.043751060642682936,
    ("larger", "etest-wald"): 0.043751050280207024,

    ("smaller", "wald"): 0.9643181675100791,
    ("smaller", "score"): 0.9579916237441386,
    ("smaller", "sqrt"): 0.9597662442851382,
    ("smaller", "exact-cond"): 0.9880575728311669,
    ("smaller", "cond-midp"): 0.9511765477509471,
    ("smaller", "etest"): 0.9672396898656999,
    ("smaller", "etest-wald"): 0.9672397002281757
    }


@pytest.mark.parametrize('case', list(cases_alt.keys()))
def test_alternative(case):
    # regression test numbers, but those are close to each other
    alt, meth = case
    count1, n1, count2, n2 = 6, 51., 1, 54.
    _, pv = smr.test_poisson_2indep(count1, n1, count2, n2, method=meth,
                                    value=1.2, alternative=alt)
    assert_allclose(pv, cases_alt[case], rtol=1e-13)


class TestMethodsCompare2indep():
    # check that all methods for test and confint produce similar results

    @pytest.mark.parametrize(
        "compare, meth",
        [("ratio", meth) for meth
         in method_names_poisson_2indep["test"]["ratio"]] +
        [("diff", meth) for meth
         in method_names_poisson_2indep["test"]["diff"]]
        )
    def test_test(self, meth, compare):
        count1, n1, count2, n2 = 60, 514.775, 40, 543.0870000
        tst = smr.test_poisson_2indep(count1, n1, count2, n2, method=meth,
                                      compare=compare,
                                      value=None, alternative='two-sided')

        assert_allclose(tst.pvalue, 0.0245, rtol=0.2)

        # compare with degenerate interval for nonequivalence
        if compare == "ratio":
            f = 1.00
            low, upp = 1 / f, f
        else:
            v = 0.00
            low, upp = -v, v

        tst2 = nonequivalence_poisson_2indep(count1, n1, count2, n2, low, upp,
                                             method=meth,
                                             compare=compare)
        if "cond" in meth or "etest" in meth:
            # comparison is not exact for noncentral methods
            rtol = 0.1
        else:
            rtol = 1e-12
        assert_allclose(tst2.pvalue, tst.pvalue, rtol=rtol)

        # check corner case count2 = 0, see issue #8313
        with pytest.warns(RuntimeWarning):
            tst = smr.test_poisson_2indep(count1, n1, 0, n2, method=meth,
                                          compare=compare,
                                          value=None, alternative='two-sided')

    @pytest.mark.parametrize(
        "compare, meth",
        [("ratio", meth) for meth
         in method_names_poisson_2indep["confint"]["ratio"]] +
        [("diff", meth) for meth
         in method_names_poisson_2indep["confint"]["diff"]]
        )
    def test_confint(self, meth, compare):
        count1, n1, count2, n2 = 60, 514.775, 40, 543.0870000
        if compare == "ratio":
            ci_val = [1.04, 2.34]
        else:
            ci_val = [0.0057, 0.081]

        ci = confint_poisson_2indep(count1, n1, count2, n2, method=meth,
                                    compare=compare, alpha=0.05)
        assert_allclose(ci, ci_val, rtol=0.1)

    @pytest.mark.parametrize(
        "compare, meth",
        [("ratio", meth) for meth
         in method_names_poisson_2indep["test"]["ratio"]] +
        [("diff", meth) for meth
         in method_names_poisson_2indep["test"]["diff"]]
        )
    def test_test_vectorized(self, meth, compare):
        if "etest" in meth:
            pytest.skip("nonequivalence etest not vectorized")

        count1, n1, count2, n2 = 60, 514.775, 40, 543.0870000
        count1v = np.array([count1, count2])
        n1v = np.array([n1, n2])
        nfact = 1.
        # scipy binom test requires int dtype
        count2v = np.array([count2, count1 * nfact], dtype=int)
        n2v = np.array([n2, n1 * nfact])
        count1, n1, count2, n2 = count1v, n1v, count2v, n2v

        # compare with degenerate interval for nonequivalence
        if compare == "ratio":
            f = 1.00
            low, upp = 1 / f, f
        else:
            v = 0.00
            low, upp = -v, v

        tst2 = nonequivalence_poisson_2indep(count1, n1, count2, n2, low, upp,
                                             method=meth,
                                             compare=compare)
        assert tst2.statistic.shape == (2,)
        assert tst2.pvalue.shape == (2,)

        if not ("cond" in meth or "etest" in meth):
            # not all methods are vectorized for smr.test_poisson_2indep
            tst = smr.test_poisson_2indep(count1, n1, count2, n2, method=meth,
                                          compare=compare,
                                          value=None, alternative='two-sided')

            assert_allclose(tst2.pvalue, tst.pvalue, rtol=1e-12)

        # check vectorized equivalence test
        if compare == "ratio":
            f = 1.5
            low, upp = 1 / f, f
        else:
            v = 0.5
            low, upp = -v, v

        tst0 = smr.tost_poisson_2indep(count1[0], n1[0], count2[0], n2[0],
                                       low, upp,
                                       method=meth,
                                       compare=compare)
        tst1 = smr.tost_poisson_2indep(count1[1], n1[1], count2[1], n2[1],
                                       low, upp,
                                       method=meth,
                                       compare=compare)

        tst2 = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp,
                                       method=meth,
                                       compare=compare)
        assert tst2.statistic.shape == (2,)
        assert tst2.pvalue.shape == (2,)

        assert_allclose(tst2.statistic[0], tst0.statistic, rtol=1e-12)
        assert_allclose(tst2.pvalue[0], tst0.pvalue, rtol=1e-12)
        assert_allclose(tst2.statistic[1], tst1.statistic, rtol=1e-12)
        assert_allclose(tst2.pvalue[1], tst1.pvalue, rtol=1e-12)


def test_y_grid_regression():
    y_grid = arange(1000)

    _, pv = etest_poisson_2indep(60, 51477.5, 30, 54308.7, y_grid=y_grid)
    assert_allclose(pv, 0.000567261758250953, atol=1e-15)

    _, pv = etest_poisson_2indep(41, 28010, 15, 19017, y_grid=y_grid)
    assert_allclose(pv, 0.03782053187021494, atol=1e-15)

    _, pv = etest_poisson_2indep(1, 1, 1, 1, y_grid=[1])
    assert_allclose(pv, 0.1353352832366127, atol=1e-15)


def test_invalid_y_grid():
    # check ygrid deprecation
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        etest_poisson_2indep(1, 1, 1, 1, ygrid=[1])
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "ygrid" in str(w[0].message)

    # check y_grid validity
    with pytest.raises(ValueError) as e:
        etest_poisson_2indep(1, 1, 1, 1, y_grid=1)
    assert "y_grid" in str(e.value)


def test_poisson_power_2ratio():
    # power compared to PASS documentation

    rate1, rate2 = 2.2, 2.2
    nobs1, nobs2 = 95, 95
    # nobs_ratio = 1
    alpha = 0.025
    exposure = 2.5
    low, upp = 0.8, 1.25
    dispersion = 1

    # check power of equivalence test
    cases = [
        (1.9, 704, 704, 0.90012),
        (2.0, 246, 246, 0.90057),
        (2.2, 95, 95, 0.90039),
        (2.5, 396, 396, 0.90045),
    ]

    for case in cases:
        rate1, nobs1, nobs2, p = case
        pow_ = power_equivalence_poisson_2indep(
            rate1, rate2, nobs1, low, upp,
            nobs_ratio=nobs2 / nobs1,
            exposure=exposure, alpha=alpha,
            dispersion=dispersion)
        assert_allclose(pow_, p, atol=5e-5)

        # compare other method_var for similar results
        pow_2 = power_equivalence_poisson_2indep(
            rate1, rate2, nobs1, low, upp,
            nobs_ratio=nobs2 / nobs1,
            exposure=exposure,
            alpha=alpha,
            method_var="score",
            dispersion=dispersion)

        assert_allclose(pow_2, p, rtol=5e-3)

    # check power of onesided test, smaller with a margin
    # non-inferiority
    # alternative smaller H1: rate1 / rate2 < R
    cases = [
        (1.8, 29, 29, 0.90056),
        (1.9, 39, 39, 0.90649),
        (2.2, 115, 115, 0.90014),
        (2.4, 404, 404, 0.90064),
    ]

    low = 1.2
    for case in cases:
        rate1, nobs1, nobs2, p = case
        pow_ = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.025, dispersion=1,
            alternative="smaller")

        assert_allclose(pow_, p, atol=5e-5)

        pow_ = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.05,
            dispersion=1, alternative="two-sided")
        assert_allclose(pow_, p, atol=5e-5)

    # check size, power at null
    pow_ = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=rate1 / rate2,
            alpha=0.05, dispersion=1, alternative="two-sided")
    assert_allclose(pow_, 0.05, atol=5e-5)

    # check power of onesided test, larger with a margin (superiority)
    # alternative larger H1: rate1 / rate2 > R
    # here I just reverse the case of smaller alternative

    cases = [
        (1.8, 29, 29, 0.90056),
        (1.9, 39, 39, 0.90649),
        (2.2, 115, 115, 0.90014),
        (2.4, 404, 404, 0.90064),
    ]

    rate1 = 2.2
    low = 1 / 1.2
    for case in cases:
        rate2, nobs1, nobs2, p = case
        pow_ = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.025,
            dispersion=1, alternative="larger")
        assert_allclose(pow_, p, atol=5e-5)

        # compare other method_var for similar results
        pow_2 = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.025,
            method_var="score",
            dispersion=1, alternative="larger")
        assert_allclose(pow_2, p, rtol=5e-3)

        pow_ = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.05,
            dispersion=1, alternative="two-sided")
        assert_allclose(pow_, p, atol=5e-5)

        # compare other method_var for similar results
        pow_2 = power_poisson_ratio_2indep(
            rate1, rate2, nobs1, nobs_ratio=nobs2 / nobs1,
            exposure=exposure, value=low, alpha=0.05,
            method_var="score",
            dispersion=1, alternative="two-sided")
        assert_allclose(pow_2, p, rtol=5e-3)


def test_power_poisson_equal():

    # Example from Chapter 436: Tests for the Difference Between Two Poisson
    # Rates
    # equality null H0: rate1 = rate2
    #
    # Power N1 N2 N rate1 rate2 diff(rate2-rate1) ratio(rate2/rate1) Alpha
    # 0.82566 8 6 14 10.00 15.00 5.00 1.5000 0.050
    #
    # "1" is reference group
    # we use group 2 reference

    nobs1, nobs2 = 6, 8
    nobs_ratio = nobs2 / nobs1
    rate1, rate2 = 15, 10

    pow_ = power_poisson_diff_2indep(
        rate1, rate2, nobs1, nobs_ratio=nobs_ratio, alpha=0.05, value=0,
        method_var="alt", alternative='larger', return_results=True)
    assert_allclose(pow_.power, 0.82566, atol=5e-5)

    # This supports now nonzero value
    # example in table 1 of Stucke, Kieser (2013)
    # regression numbers but close to exact power in table

    pow_ = power_poisson_diff_2indep(
        0.6, 0.6, 97, 3 / 2,
        value=0.3,
        alpha=0.025,
        alternative="smaller",
        method_var="score",
        return_results=True)

    assert_allclose(pow_.power, 0.802596, atol=5e-5)

    pow_ = power_poisson_diff_2indep(
        0.6, 0.6, 128, 2 / 3,
        value=0.3,
        alpha=0.025,
        alternative="smaller",
        method_var="score",
        return_results=True)

    assert_allclose(pow_.power, 0.80194, atol=5e-5)


def test_power_negbin():

    # example from PASS ch. 468, last example Zhu 2016
    # Note, the table has wrong RU, does not correspond to text
    rate1, rate2 = 2.5, 2.5
    nobs1, nobs2 = 965, 965
    alpha = 0.05
    exposure = 0.9
    low, upp = 0.875, 1 / 0.875
    dispersion = 0.35

    pow1 = 0.90022
    pow_ = power_equivalence_neginb_2indep(
        rate1, rate2, nobs1, low, upp,
        nobs_ratio=nobs2 / nobs1,
        exposure=exposure,
        alpha=alpha,
        dispersion=dispersion, method_var="alt")
    assert_allclose(pow_, pow1, atol=5e-5)

    nobs1, nobs2 = 966, 966

    pow1 = 0.90015
    pow_ = power_equivalence_neginb_2indep(
        rate1, rate2, nobs1, low, upp,
        nobs_ratio=nobs2 / nobs1,
        exposure=exposure,
        alpha=alpha,
        dispersion=dispersion, method_var="ftotal")
    assert_allclose(pow_, pow1, atol=5e-5)

    pow1 = 0.90034
    pow_ = power_equivalence_neginb_2indep(
        rate1, rate2, nobs1, low, upp,
        nobs_ratio=nobs2 / nobs1,
        exposure=exposure,
        alpha=alpha,
        dispersion=dispersion, method_var="score")
    assert_allclose(pow_, pow1, atol=5e-5)

    # test with unequal sample size agains R
    # R packages MKmisc and PASSED have only equality test, ratio=1, for negbin
    # package PASSED has same results as MKmisc but different signature
    # in R: treatment and control are reversed, theta = 1 / dispersion
    # > power_NegativeBinomial(n1 = 100, n2=50, mu1 = 0.5, mu2 = 0.3, theta=2,
    #   duration = 2, approach = 3))

    rate2, nobs2, rate1, nobs1, exposure = 0.3, 50, 0.5, 100, 2
    pow1 = 0.6207448
    pow_ = power_negbin_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=0.5,
        alternative="two-sided",
        method_var="score",
        return_results=False,
        )
    assert_allclose(pow_, pow1, atol=5e-2)

    # flip nobs ratio
    pow1 = 0.5825763
    nobs1, nobs2 = nobs2, nobs1
    pow_ = power_negbin_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=0.5,
        alternative="two-sided",
        method_var="score",
        return_results=False)
    assert_allclose(pow_, pow1, atol=5e-2)

    # corner case poisson
    # > power_NegativeBinomial(n1 = 50, n2=100, mu1 = 0.5, mu2 = 0.3,
    #   theta=200000, duration = 2, approach = 3)
    pow1 = 0.7248956
    pow_ = power_negbin_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=0, alternative="two-sided",
        method_var="score",
        return_results=False)

    assert_allclose(pow_, pow1, atol=5e-2)

    pow_p = power_poisson_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=1, alternative="two-sided",
        method_var="score",
        return_results=True)

    assert_allclose(pow_p, pow1, atol=5e-2)
    assert_allclose(pow_p, pow_, rtol=1e-13)

    # test one-sided
    pow1 = 0.823889
    pow_ = power_negbin_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=0, alternative="smaller",
        method_var="score",
        return_results=False)

    pow_p = power_poisson_ratio_2indep(
        rate2, rate1, nobs2,
        nobs_ratio=nobs1 / nobs2,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=1, alternative="smaller",
        method_var="score",
        return_results=True)

    assert_allclose(pow_p, pow1, atol=5e-2)
    assert_allclose(pow_p, pow_, rtol=1e-13)

    # reverse 2 samples
    pow_ = power_negbin_ratio_2indep(
        rate1, rate2, nobs1,
        nobs_ratio=nobs2 / nobs1,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=0, alternative="larger",
        method_var="score",
        return_results=False)

    pow_p = power_poisson_ratio_2indep(
        rate1, rate2, nobs1,
        nobs_ratio=nobs2 / nobs1,
        exposure=exposure,
        value=1,
        alpha=alpha, dispersion=1, alternative="larger",
        method_var="score",
        return_results=True)

    assert_allclose(pow_p, pow1, atol=5e-2)
    assert_allclose(pow_p, pow_, rtol=1e-13)
