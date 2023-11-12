# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:45:51 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import pytest

from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
    confint_effectsize_oneway, confint_noncentrality, effectsize_oneway,
    anova_oneway,
    anova_generic, equivalence_oneway, equivalence_oneway_generic,
    power_equivalence_oneway, _power_equivalence_oneway_emp,
    f2_to_wellek, fstat_to_wellek, wellek_to_f2)
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
    wald_test_noncent_generic, wald_test_noncent, _offset_constraint)


def test_oneway_effectsize():
    # examole 3 in Steiger 2004 Beyond the F-test, p. 169
    F = 5
    df1 = 3
    df2 = 76
    nobs = 80

    ci = confint_noncentrality(F, (df1, df2), alpha=0.05,
                               alternative="two-sided")

    ci_es = confint_effectsize_oneway(F, (df1, df2), alpha=0.05)
    ci_steiger = ci_es.ci_f * np.sqrt(4 / 3)
    res_ci_steiger = [0.1764, 0.7367]
    res_ci_nc = np.asarray([1.8666, 32.563])

    assert_allclose(ci, res_ci_nc, atol=0.0001)
    assert_allclose(ci_es.ci_f_corrected, res_ci_steiger, atol=0.00006)
    assert_allclose(ci_steiger, res_ci_steiger, atol=0.00006)
    assert_allclose(ci_es.ci_f**2, res_ci_nc / nobs, atol=0.00006)
    assert_allclose(ci_es.ci_nc, res_ci_nc, atol=0.0001)


def test_effectsize_power():
    # example and results from PASS documentation
    n_groups = 3
    means = [527.86, 660.43, 649.14]
    vars_ = 107.4304**2
    nobs = 12
    es = effectsize_oneway(means, vars_, nobs, use_var="equal", ddof_between=0)
    es = np.sqrt(es)

    alpha = 0.05
    power = 0.8
    nobs_t = nobs * n_groups
    kwds = {'effect_size': es, 'nobs': nobs_t, 'alpha': alpha, 'power': power,
            'k_groups': n_groups}

    from statsmodels.stats.power import FTestAnovaPower

    res_pow = 0.8251
    res_es = 0.559
    kwds_ = kwds.copy()
    del kwds_['power']
    p = FTestAnovaPower().power(**kwds_)
    assert_allclose(p, res_pow, atol=0.0001)
    assert_allclose(es, res_es, atol=0.0006)

    # example unequal sample sizes
    nobs = np.array([15, 9, 9])
    kwds['nobs'] = nobs
    es = effectsize_oneway(means, vars_, nobs, use_var="equal", ddof_between=0)
    es = np.sqrt(es)
    kwds['effect_size'] = es
    p = FTestAnovaPower().power(**kwds_)

    res_pow = 0.8297
    res_es = 0.590
    assert_allclose(p, res_pow, atol=0.005)  # lower than print precision
    assert_allclose(es, res_es, atol=0.0006)


def test_effectsize_fstat():
    # results from R package `effectsize`, confint is 0.9 confidence
    # > es = F_to_eta2(45.8, 3, 35)
    Eta_Sq_partial = 0.796983758700696
    CI_eta2 = 0.685670133284926, 0.855981325777856  # reformated from output
    # > es = F_to_epsilon2(45.8, 3, 35)
    Epsilon_Sq_partial = 0.779582366589327
    CI_eps2 = 0.658727573280777, 0.843636867987386
    # > es = F_to_omega2(45.8, 3, 35)
    Omega_Sq_partial = 0.775086505190311
    CI_omega2 = 0.65286429480169, 0.840179680453464
    # > es = F_to_f(45.8, 3, 35)
    Cohens_f_partial = 1.98134153686695
    CI_f = 1.47694659580859, 2.43793847155554

    f_stat, df1, df2 = 45.8, 3, 35
    # nobs = df1 + df2 + 1  # not directly used in the following, only df
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(np.sqrt(fes.f2), Cohens_f_partial, rtol=1e-13)
    assert_allclose(fes.eta2, Eta_Sq_partial, rtol=1e-13)
    assert_allclose(fes.eps2, Epsilon_Sq_partial, rtol=1e-13)
    assert_allclose(fes.omega2, Omega_Sq_partial, rtol=1e-13)

    ci_nc = confint_noncentrality(f_stat, (df1, df2), alpha=0.1)
    # the following replicates R package effectsize
    ci_es = smo._fstat2effectsize(ci_nc / df1, (df1, df2))
    assert_allclose(ci_es.eta2, CI_eta2, rtol=2e-4)
    assert_allclose(ci_es.eps2, CI_eps2, rtol=2e-4)
    assert_allclose(ci_es.omega2, CI_omega2, rtol=2e-4)
    assert_allclose(np.sqrt(ci_es.f2), CI_f, rtol=2e-4)


def test_effectsize_fstat_stata():
    # reference numbers computed with Stata 14
    # Stata 16 does not seem to have confint for omega2

    # esizei 2 40 7.47403193349075, level(90)
    eta2 = 0.2720398648288652
    lb_eta2 = 0.0742092468714613
    ub_eta2 = 0.4156116886974804
    omega2 = 0.2356418580703085
    lb_omega2 = 0.0279197092150344
    ub_omega2 = 0.3863922731323545
    # level = 90

    f_stat, df1, df2 = 7.47403193349075, 2, 40
    fes = smo._fstat2effectsize(f_stat, (df1, df2))
    assert_allclose(fes.eta2, eta2, rtol=1e-13)
    assert_allclose(fes.omega2, omega2, rtol=0.02)  # low agreement
    ci_es = smo.confint_effectsize_oneway(f_stat, (df1, df2), alpha=0.1)
    assert_allclose(ci_es.eta2, (lb_eta2, ub_eta2), rtol=1e-4)
    assert_allclose(ci_es.ci_omega2, (lb_omega2, ub_omega2), rtol=0.025)


@pytest.mark.parametrize("center", ['median', 'mean', 'trimmed'])
def test_scale_transform(center):
    x = np.random.randn(5, 3)
    xt = scale_transform(x, center=center, transform='abs', trim_frac=0.2,
                         axis=0)
    xtt = scale_transform(x.T, center=center, transform='abs', trim_frac=0.2,
                          axis=1)
    assert_allclose(xt.T, xtt, rtol=1e-13)
    xt0 = scale_transform(x[:, 0], center=center, transform='abs',
                          trim_frac=0.2)
    assert_allclose(xt0, xt[:, 0], rtol=1e-13)
    assert_allclose(xt0, xtt[0, :], rtol=1e-13)


class TestOnewayEquivalenc:

    @classmethod
    def setup_class(cls):
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931,
              95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618,
              105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254,
              99.510, 89.005, 108.200, 82.209, 100.104, 103.706,
              107.067]
        y3 = [100.825, 100.255, 103.363, 93.230, 95.325, 100.288,
              94.750, 107.129, 98.246, 96.365, 99.740, 106.049,
              92.691, 93.111, 98.243]

        n_groups = 4
        arrs_w = [np.asarray(yi) for yi in [y0, y1, y2, y3]]
        nobs = np.asarray([len(yi) for yi in arrs_w])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in arrs_w])
        stds = np.asarray([yi.std(ddof=1) for yi in arrs_w])
        cls.data = arrs_w  # TODO use `data`
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_equivalence_equal(self):
        # reference numbers from Jan and Shieh 2019, p. 5
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups

        eps = 0.5
        res0 = anova_generic(means, stds**2, nobs, use_var="equal")
        f = res0.statistic
        res = equivalence_oneway_generic(f, n_groups, nobs.sum(), eps,
                                         res0.df, alpha=0.05,
                                         margin_type="wellek")
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

        # the agreement for f-stat looks too low
        assert_allclose(f, 0.0926, atol=0.0006)

        res = equivalence_oneway(self.data, eps, use_var="equal",
                                 margin_type="wellek")
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

    def test_equivalence_welch(self):
        # reference numbers from Jan and Shieh 2019, p. 6
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        vars_ = stds**2

        eps = 0.5
        res0 = anova_generic(means, vars_, nobs, use_var="unequal",
                             welch_correction=False)
        f_stat = res0.statistic
        res = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), eps,
                                         res0.df, alpha=0.05,
                                         margin_type="wellek")
        assert_allclose(res.pvalue, 0.0110, atol=0.001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)

        # agreement for Welch f-stat looks too low b/c welch_correction=False
        assert_allclose(f_stat, 0.1102, atol=0.007)

        res = equivalence_oneway(self.data, eps, use_var="unequal",
                                 margin_type="wellek")
        assert_allclose(res.pvalue, 0.0110, atol=1e-4)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(res.f_stat, 0.1102, atol=1e-4)  # 0.007)

        # check post-hoc power, JS p. 6
        pow_ = _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps,
                                             res0.df)
        assert_allclose(pow_, 0.1552, atol=0.007)

        pow_ = power_equivalence_oneway(eps, eps, nobs.sum(),
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="wellek")
        assert_allclose(pow_, 0.05, atol=1e-13)

        nobs_t = nobs.sum()
        es = effectsize_oneway(means, vars_, nobs, use_var="unequal")
        es = np.sqrt(es)
        es_w0 = f2_to_wellek(es**2, n_groups)
        es_w = np.sqrt(fstat_to_wellek(f_stat, n_groups, nobs_t / n_groups))

        pow_ = power_equivalence_oneway(es_w, eps, nobs_t,
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="wellek")
        assert_allclose(pow_, 0.1552, atol=0.007)
        assert_allclose(es_w0, es_w, atol=0.007)

        margin = wellek_to_f2(eps, n_groups)
        pow_ = power_equivalence_oneway(es**2, margin, nobs_t,
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="f2")
        assert_allclose(pow_, 0.1552, atol=0.007)


class TestOnewayScale:

    @classmethod
    def setup_class(cls):
        yt0 = np.array([102., 320., 0., 107., 198., 200., 4., 20., 110., 128.,
                       7., 119., 309.])

        yt1 = np.array([0., 1., 228., 81., 87., 119., 79., 181., 43., 12., 90.,
                       105., 108., 119., 0., 9.])
        yt2 = np.array([33., 294., 134., 216., 83., 105., 69., 20., 20., 63.,
                       98., 155., 78., 75.])

        y0 = np.array([452., 874., 554., 447., 356., 754., 558., 574., 664.,
                       682., 547., 435., 245.])
        y1 = np.array([546., 547., 774., 465., 459., 665., 467., 365., 589.,
                       534., 456., 651., 654., 665., 546., 537.])
        y2 = np.array([785., 458., 886., 536., 669., 857., 821., 772., 732.,
                       689., 654., 597., 830., 827.])

        n_groups = 3
        data = [y0, y1, y2]
        nobs = np.asarray([len(yi) for yi in data])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in data])
        stds = np.asarray([yi.std(ddof=1) for yi in data])
        cls.data = data
        cls.data_transformed = [yt0, yt1, yt2]
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_means(self):

        # library onewaystats, BF test for equality of means
        # st = bf.test(y ~ g, df3)
        statistic = 7.10900606421182
        parameter = [2, 31.4207256105052]
        p_value = 0.00283841965791224
        # method = 'Brown-Forsythe Test'
        res = anova_oneway(self.data, use_var="bf")

        # R bf.test uses original BF df_num
        assert_allclose(res.pvalue2, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose([res.df_num2, res.df_denom], parameter)

    def test_levene(self):
        data = self.data

        # lawstat: Test Statistic = 1.0866123063642, p-value = 0.3471072204516
        statistic = 1.0866123063642
        p_value = 0.3471072204516
        res0 = smo.test_scale_oneway(data, method='equal', center='median',
                                     transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)

        # library car
        # > lt = leveneTest(y ~ g, df3, center=mean, trim=0.2)
        statistic = 1.10732113109744
        p_value = 0.340359251994645
        df = [2, 40]
        res0 = smo.test_scale_oneway(data, method='equal', center='trimmed',
                                     transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, df)

        # library(onewaytests)
        # test uses mean as center
        # > st = homog.test(y ~ g, df3)
        statistic = 1.07894485177512
        parameter = [2, 40]  # df
        p_value = 0.349641166869223
        # method = "Levene's Homogeneity Test"
        res0 = smo.test_scale_oneway(data, method='equal', center='mean',
                                     transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, parameter)

        # > st = homog.test(y ~ g, df3, method = "Bartlett")
        statistic = 3.01982414477323
        # parameter = 2  # scipy bartlett does not return df
        p_value = 0.220929402900495
        # method = "Bartlett's Homogeneity Test"
        # Bartlett is in scipy.stats
        from scipy import stats
        stat, pv = stats.bartlett(*data)
        assert_allclose(pv, p_value, rtol=1e-13)
        assert_allclose(stat, statistic, rtol=1e-13)

    def test_options(self):
        # regression tests for options,
        # many might not be implemented in other packages
        data = self.data

        # regression numbers from initial run
        statistic, p_value = 1.0173464626246675, 0.3763806150460239
        df = (2.0, 24.40374758005409)
        res = smo.test_scale_oneway(data, method='unequal', center='median',
                                    transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)

        statistic, p_value = 1.0329722145270606, 0.3622778213868562
        df = (1.83153791573948, 30.6733640949525)
        p_value2 = 0.3679999679787619
        df2 = (2, 30.6733640949525)
        res = smo.test_scale_oneway(data, method='bf', center='median',
                                    transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        assert_allclose(res.pvalue2, p_value2, rtol=1e-13)
        assert_allclose(res.df2, df2)

        statistic, p_value = 1.7252431333701745, 0.19112038168209514
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean',
                                    transform='square', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_equal(res.df, df)

        statistic, p_value = 0.4129696057329463, 0.6644711582864451
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean',
                                    transform=lambda x: np.log(x * x),  # noqa
                                    trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)

        # compare no transform with standard anova
        res = smo.test_scale_oneway(data, method='unequal', center=0,
                                    transform='identity', trim_frac_mean=0.2)
        res2 = anova_oneway(self.data, use_var="unequal")

        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)

    def test_equivalence(self):
        data = self.data

        # compare no transform with standard anova
        res = smo.equivalence_scale_oneway(data, 0.5, method='unequal',
                                           center=0,
                                           transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var="unequal")

        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)

        res = smo.equivalence_scale_oneway(data, 0.5, method='bf',
                                           center=0,
                                           transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var="bf")

        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)


class TestOnewayOLS:

    @classmethod
    def setup_class(cls):
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931,
              95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618,
              105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254,
              99.510, 89.005, 108.200, 82.209, 100.104, 103.706,
              107.067]
        y3 = [100.825, 100.255, 103.363, 93.230, 95.325, 100.288,
              94.750, 107.129, 98.246, 96.365, 99.740, 106.049,
              92.691, 93.111, 98.243]

        cls.k_groups = k = 4
        cls.data = data = [y0, y1, y2, y3]
        cls.nobs = nobs = np.asarray([len(yi) for yi in data])
        groups = np.repeat(np.arange(k), nobs)
        cls.ex = (groups[:, None] == np.arange(k)).astype(np.int64)
        cls.y = np.concatenate(data)

    def test_ols_noncentrality(self):
        k = self.k_groups

        res_ols = OLS(self.y, self.ex).fit()
        nobs_t = res_ols.model.nobs

        # constraint
        c_equal = -np.eye(k)[1:]
        c_equal[:, 0] = 1
        v = np.zeros(c_equal.shape[0])

        # noncentrality at estimated parameters
        wt = res_ols.wald_test(c_equal, scalar=True)
        df_num, df_denom = wt.df_num, wt.df_denom

        cov_p = res_ols.cov_params()

        nc_wt = wald_test_noncent_generic(res_ols.params, c_equal, v, cov_p,
                                          diff=None, joint=True)
        assert_allclose(nc_wt, wt.statistic * wt.df_num, rtol=1e-13)

        nc_wt2 = wald_test_noncent(res_ols.params, c_equal, v, res_ols,
                                   diff=None, joint=True)
        assert_allclose(nc_wt2, nc_wt, rtol=1e-13)

        es_ols = nc_wt / nobs_t
        es_oneway = smo.effectsize_oneway(res_ols.params, res_ols.scale,
                                          self.nobs, use_var="equal")
        assert_allclose(es_ols, es_oneway, rtol=1e-13)

        alpha = 0.05
        pow_ols = smpwr.ftest_power(np.sqrt(es_ols), df_denom, df_num, alpha,
                                    ncc=1)
        pow_oneway = smpwr.ftest_anova_power(np.sqrt(es_oneway), nobs_t, alpha,
                                             k_groups=k, df=None)
        assert_allclose(pow_ols, pow_oneway, rtol=1e-13)

        # noncentrality at other params
        params_alt = res_ols.params * 0.75
        # compute constraint value so we can get noncentrality from wald_test
        v_off = _offset_constraint(c_equal, res_ols.params, params_alt)
        wt_off = res_ols.wald_test((c_equal, v + v_off), scalar=True)
        nc_wt_off = wald_test_noncent_generic(params_alt, c_equal, v,
                                              cov_p, diff=None, joint=True)
        assert_allclose(nc_wt_off, wt_off.statistic * wt_off.df_num,
                        rtol=1e-13)

        # check vectorized version, joint=False
        nc_wt_vec = wald_test_noncent_generic(params_alt, c_equal, v,
                                              cov_p, diff=None, joint=False)
        for i in range(c_equal.shape[0]):
            nc_wt_i = wald_test_noncent_generic(params_alt, c_equal[i : i + 1],  # noqa
                                                v[i : i + 1], cov_p, diff=None,  # noqa
                                                joint=False)
            assert_allclose(nc_wt_vec[i], nc_wt_i, rtol=1e-13)


def test_simulate_equivalence():
    # regression test, needs large k_mc to be reliable

    k_groups = 4
    k_repl = 10
    nobs = np.array([10, 12, 13, 15]) * k_repl
    means = np.array([-1, 0, 0, 1]) * 0.12
    vars_ = np.array([1, 2, 3, 4])
    nobs_t = nobs.sum()

    eps = 0.0191 * 10
    opt_var = ["unequal", "equal", "bf"]
    k_mc = 100
    np.random.seed(987126)
    res_mc = smo.simulate_power_equivalence_oneway(
        means, nobs, eps, vars_=vars_, k_mc=k_mc, trim_frac=0.1,
        options_var=opt_var, margin_type="wellek")

    frac_reject = (res_mc.pvalue <= 0.05).sum(0) / k_mc
    assert_allclose(frac_reject, [0.17, 0.18, 0.14], atol=0.001)
    # result with k_mc = 10000 is [0.1466, 0.1871, 0.1606]
    # similar to asy below, but not very close for all

    es_alt_li = []
    for uv in opt_var:
        es = effectsize_oneway(means, vars_, nobs, use_var=uv)
        es_alt_li.append(es)

    # compute asy power as comparison
    margin = wellek_to_f2(eps, k_groups)
    pow_ = [power_equivalence_oneway(
        es_, margin, nobs_t, n_groups=k_groups, df=None, alpha=0.05,
        margin_type="f2") for es_ in es_alt_li]
    # regression test numbers
    assert_allclose(pow_, [0.147749, 0.173358, 0.177412], atol=0.007)
