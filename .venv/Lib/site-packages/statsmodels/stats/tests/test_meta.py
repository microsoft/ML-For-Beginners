# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:08:37 2020

Author: Josef Perktold
License: BSD-3

"""

import io

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_equal, assert_allclose

from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM

from statsmodels.stats.meta_analysis import (
    effectsize_smd, effectsize_2proportions, combine_effects,
    _fit_tau_iterative, _fit_tau_mm, _fit_tau_iter_mm)

from .results import results_meta


class TestEffectsizeBinom:

    @classmethod
    def setup_class(cls):
        cls.results = results_meta.eff_prop1
        ss = """\
            study,nei,nci,e1i,c1i,e2i,c2i,e3i,c3i,e4i,c4i
            1,19,22,16.0,20.0,11,12,4.0,8.0,4,3
            2,34,35,22.0,22.0,18,12,15.0,8.0,15,6
            3,72,68,44.0,40.0,21,15,10.0,3.0,3,0
            4,22,20,19.0,12.0,14,5,5.0,4.0,2,3
            5,70,32,62.0,27.0,42,13,26.0,6.0,15,5
            6,183,94,130.0,65.0,80,33,47.0,14.0,30,11
            7,26,50,24.0,30.0,13,18,5.0,10.0,3,9
            8,61,55,51.0,44.0,37,30,19.0,19.0,11,15
            9,36,25,30.0,17.0,23,12,13.0,4.0,10,4
            10,45,35,43.0,35.0,19,14,8.0,4.0,6,0
            11,246,208,169.0,139.0,106,76,67.0,42.0,51,35
            12,386,141,279.0,97.0,170,46,97.0,21.0,73,8
            13,59,32,56.0,30.0,34,17,21.0,9.0,20,7
            14,45,15,42.0,10.0,18,3,9.0,1.0,9,1
            15,14,18,14.0,18.0,13,14,12.0,13.0,9,12
            16,26,19,21.0,15.0,12,10,6.0,4.0,5,1
            17,74,75,,,42,40,,,23,30"""
        df3 = pd.read_csv(io.StringIO(ss))
        df_12y = df3[["e2i", "nei", "c2i", "nci"]]
        # TODO: currently 1 is reference, switch labels
        # cls.count2, cls.nobs2, cls.count1, cls.nobs1 = df_12y.values.T
        cls.count1, cls.nobs1, cls.count2, cls.nobs2 = df_12y.values.T

    def test_effectsize(self):
        res2 = self.results
        dta = (self.count1, self.nobs1, self.count2, self.nobs2)
        # count1, nobs1, count2, nobs2 = dta

        eff, var_eff = effectsize_2proportions(*dta)
        assert_allclose(eff, res2.y_rd, rtol=1e-13)
        assert_allclose(var_eff, res2.v_rd, rtol=1e-13)

        eff, var_eff = effectsize_2proportions(*dta, statistic="rr")
        assert_allclose(eff, res2.y_rr, rtol=1e-13)
        assert_allclose(var_eff, res2.v_rr, rtol=1e-13)

        eff, var_eff = effectsize_2proportions(*dta, statistic="or")
        assert_allclose(eff, res2.y_or, rtol=1e-13)
        assert_allclose(var_eff, res2.v_or, rtol=1e-13)

        eff, var_eff = effectsize_2proportions(*dta, statistic="as")
        assert_allclose(eff, res2.y_as, rtol=1e-13)
        assert_allclose(var_eff, res2.v_as, rtol=1e-13)


class TestEffSmdMeta:

    @classmethod
    def setup_class(cls):
        # example from book Applied Meta-Analysis with R
        data = [
            ["Carroll", 94, 22, 60, 92, 20, 60],
            ["Grant", 98, 21, 65, 92, 22, 65],
            ["Peck", 98, 28, 40, 88, 26, 40],
            ["Donat", 94, 19, 200, 82, 17, 200],
            ["Stewart", 98, 21, 50, 88, 22, 45],
            ["Young", 96, 21, 85, 92, 22, 85]]
        colnames = ["study", "mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"]
        dframe = pd.DataFrame(data, columns=colnames)
        cls.dta = np.asarray(dframe[["mean_t", "sd_t", "n_t",
                                     "mean_c", "sd_c", "n_c"]]).T
        cls.row_names = dframe["study"]

    def test_smd(self):
        # compare with metafor
        yi = np.array([
            0.09452415852032972, 0.27735586626551018, 0.36654442951591998,
            0.66438496832691396, 0.46180628128769841, 0.18516443739910043])

        vi_asy = np.array([
            0.03337056173559990, 0.03106510106366112, 0.05083971761755720,
            0.01055175923267344, 0.04334466980873156, 0.02363025255552155])
        vi_ub = np.array([
            0.03337176211751222, 0.03107388569950075, 0.05088098670518214,
            0.01055698026322296, 0.04339077140867459, 0.02363252645927709])

        eff, var_eff = effectsize_smd(*self.dta)
        # agreement with metafor is lower, atol for var 2.5e-06
        # It's likely a small difference in bias correction
        assert_allclose(eff, yi, rtol=1e-5)
        assert_allclose(var_eff, vi_ub, rtol=1e-4)
        assert_allclose(var_eff, vi_asy, rtol=2e-3)  # not the same definition

        # with unequal variance, not available yet
        # > r = escalc(measure="SMDH", m1i=m.t, sd1i=sd.t, n1i=n.t, m2i=m.c,
        #             sd2i=sd.c, n2i=n.c, data=dat, vtype="UB")
        yi = np.array([
            0.09452415852032972, 0.27735586626551023, 0.36654442951591998,
            0.66438496832691396, 0.46122883016705268, 0.18516443739910043])
        vi_ub = np.array([
            0.03350541862210323, 0.03118164624093491, 0.05114625874744853,
            0.01057160214284120, 0.04368303906568672, 0.02369839436451885])

        # compare with package `meta`
        # high agreement, using smd function was written based on meta example

        # > rm = metacont(n.t,m.t,sd.t,n.c,m.c,sd.c,
        # +               data=dat,studlab=rownames(dat),sm="SMD")

        # > rm$TE
        yi_m = np.array([
            0.09452437336063831, 0.27735640148036095, 0.36654634845797818,
            0.66438509989113559, 0.46180797677414176, 0.18516464424648887])
        # > rm$seTE**2
        vi_m = np.array([
            0.03337182573880991, 0.03107434965484927, 0.05088322525353587,
            0.01055724834741877, 0.04339324466573324, 0.02363264537147130])
        assert_allclose(eff, yi_m, rtol=1e-13)
        assert_allclose(var_eff, vi_m, rtol=1e-13)


class TestMetaK1:

    @classmethod
    def setup_class(cls):

        cls.eff = np.array([61.00, 61.40, 62.21, 62.30, 62.34, 62.60, 62.70,
                            62.84, 65.90])
        cls.var_eff = np.array([0.2025, 1.2100, 0.0900, 0.2025, 0.3844, 0.5625,
                                0.0676, 0.0225, 1.8225])

    def test_tau_kacker(self):
        # test iterative and two-step methods, Kacker 2004
        # PM CA DL C2 from table 1 first row p. 135
        # test for PM and DL are also against R metafor in other tests
        eff, var_eff = self.eff, self.var_eff
        t_PM, t_CA, t_DL, t_C2 = [0.8399, 1.1837, 0.5359, 0.9352]

        tau2, converged = _fit_tau_iterative(eff, var_eff,
                                             tau2_start=0.1, atol=1e-8)
        assert_equal(converged, True)
        assert_allclose(np.sqrt(tau2), t_PM, atol=6e-5)

        k = len(eff)
        # cochrane uniform weights
        tau2_ca = _fit_tau_mm(eff, var_eff, np.ones(k) / k)
        assert_allclose(np.sqrt(tau2_ca), t_CA, atol=6e-5)

        # DL one step, and 1 iteration, reduced agreement with Kacker
        tau2_dl = _fit_tau_mm(eff, var_eff, 1 / var_eff)
        assert_allclose(np.sqrt(tau2_dl), t_DL, atol=1e-3)

        tau2_dl_, converged = _fit_tau_iter_mm(eff, var_eff, tau2_start=0,
                                               maxiter=1)
        assert_equal(converged, False)
        assert_allclose(tau2_dl_, tau2_dl, atol=1e-10)

        # C2 two step, start with CA
        tau2_c2, converged = _fit_tau_iter_mm(eff, var_eff,
                                              tau2_start=tau2_ca,
                                              maxiter=1)
        assert_equal(converged, False)
        assert_allclose(np.sqrt(tau2_c2), t_C2, atol=6e-5)

    def test_pm(self):
        res = results_meta.exk1_metafor
        eff, var_eff = self.eff, self.var_eff

        tau2, converged = _fit_tau_iterative(eff, var_eff,
                                             tau2_start=0.1, atol=1e-8)
        assert_equal(converged, True)
        assert_allclose(tau2, res.tau2, atol=1e-10)

        # compare with WLS, PM weights
        mod_wls = WLS(eff, np.ones(len(eff)), weights=1 / (var_eff + tau2))
        res_wls = mod_wls.fit(cov_type="fixed_scale")

        assert_allclose(res_wls.params, res.b, atol=1e-13)
        assert_allclose(res_wls.bse, res.se, atol=1e-10)
        ci_low, ci_upp = res_wls.conf_int()[0]
        assert_allclose(ci_low, res.ci_lb, atol=1e-10)
        assert_allclose(ci_upp, res.ci_ub, atol=1e-10)

        # need stricter atol to match metafor,
        # I also used higher precision in metafor
        res3 = combine_effects(eff, var_eff, method_re="pm", atol=1e-7)
        # TODO: asserts below are copy paste, DRY?
        assert_allclose(res3.tau2, res.tau2, atol=1e-10)
        assert_allclose(res3.mean_effect_re, res.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_re, res.se, atol=1e-10)

        ci = res3.conf_int(use_t=False)[1]
        assert_allclose(ci[0], res.ci_lb, atol=1e-10)
        assert_allclose(ci[1], res.ci_ub, atol=1e-10)

        assert_allclose(res3.q, res.QE, atol=1e-10)
        # the following does not pass yet
        # assert_allclose(res3.i2, res.I2 / 100, atol=1e-10)  # percent in R
        # assert_allclose(res3.h2, res.H2, atol=1e-10)
        th = res3.test_homogeneity()
        q, pv = th
        df = th.df
        assert_allclose(pv, res.QEp, atol=1e-10)
        assert_allclose(q, res.QE, atol=1e-10)
        assert_allclose(df, 9 - 1, atol=1e-10)

    def test_dl(self):
        res = results_meta.exk1_dl
        eff, var_eff = self.eff, self.var_eff

        tau2 = _fit_tau_mm(eff, var_eff, 1 / var_eff)
        assert_allclose(tau2, res.tau2, atol=1e-10)

        res3 = combine_effects(eff, var_eff, method_re="dl")
        assert_allclose(res3.tau2, res.tau2, atol=1e-10)
        assert_allclose(res3.mean_effect_re, res.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_re, res.se, atol=1e-10)
        ci = res3.conf_int(use_t=False)  # fe, re, fe_wls, re_wls
        assert_allclose(ci[1][0], res.ci_lb, atol=1e-10)
        assert_allclose(ci[1][1], res.ci_ub, atol=1e-10)

        assert_allclose(res3.q, res.QE, atol=1e-10)
        # I2 is in percent in metafor
        assert_allclose(res3.i2, res.I2 / 100, atol=1e-10)
        assert_allclose(res3.h2, res.H2, atol=1e-10)
        th = res3.test_homogeneity()
        q, pv = th
        df = th.df
        assert_allclose(pv, res.QEp, atol=1e-10)
        assert_allclose(q, res.QE, atol=1e-10)
        assert_allclose(df, 9 - 1, atol=1e-10)

        # compare FE estimate
        res_fe = results_meta.exk1_fe
        assert_allclose(res3.mean_effect_fe, res_fe.b, atol=1e-13)
        assert_allclose(res3.sd_eff_w_fe, res_fe.se, atol=1e-10)

        assert_allclose(ci[0][0], res_fe.ci_lb, atol=1e-10)
        assert_allclose(ci[0][1], res_fe.ci_ub, atol=1e-10)

        # compare FE, RE with HKSJ adjustment
        res_dls = results_meta.exk1_dl_hksj
        res_fes = results_meta.exk1_fe_hksj

        assert_allclose(res3.mean_effect_re, res_dls.b, atol=1e-13)
        assert_allclose(res3.mean_effect_fe, res_fes.b, atol=1e-13)

        assert_allclose(res3.sd_eff_w_fe * np.sqrt(res3.scale_hksj_fe),
                        res_fes.se, atol=1e-10)
        assert_allclose(res3.sd_eff_w_re * np.sqrt(res3.scale_hksj_re),
                        res_dls.se, atol=1e-10)
        assert_allclose(np.sqrt(res3.var_hksj_fe), res_fes.se, atol=1e-10)
        assert_allclose(np.sqrt(res3.var_hksj_re), res_dls.se, atol=1e-10)

        # metafor uses t distribution for hksj
        ci = res3.conf_int(use_t=True)  # fe, re, fe_wls, re_wls
        assert_allclose(ci[3][0], res_dls.ci_lb, atol=1e-10)
        assert_allclose(ci[3][1], res_dls.ci_ub, atol=1e-10)
        assert_allclose(ci[2][0], res_fes.ci_lb, atol=1e-10)
        assert_allclose(ci[2][1], res_fes.ci_ub, atol=1e-10)

        th = res3.test_homogeneity()
        q, pv = th
        df = th.df
        assert_allclose(pv, res_dls.QEp, atol=1e-10)
        assert_allclose(q, res_dls.QE, atol=1e-10)
        assert_allclose(df, 9 - 1, atol=1e-10)


class TestMetaBinOR:
    # testing against results from R package `meta`

    @classmethod
    def setup_class(cls):
        cls.res2 = res2 = results_meta.results_or_dl_hk
        cls.dta = (res2.event_e, res2.n_e, res2.event_c, res2.n_c)

        eff, var_eff = effectsize_2proportions(*cls.dta, statistic="or")
        res1 = combine_effects(eff, var_eff, method_re="chi2", use_t=True)
        cls.eff = eff
        cls.var_eff = var_eff
        cls.res1 = res1

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        assert_allclose(self.eff, res2.TE, rtol=1e-13)
        assert_allclose(self.var_eff, res2.seTE**2, rtol=1e-13)

        assert_allclose(res1.mean_effect_fe, res2.TE_fixed, rtol=1e-13)
        # R meta does not adjust sd FE for HKSJ
        assert_allclose(res1.sd_eff_w_fe, res2.seTE_fixed, rtol=1e-13)

        assert_allclose(res1.q, res2.Q, rtol=1e-13)
        assert_allclose(res1.tau2, res2.tau2, rtol=1e-10)

        assert_allclose(res1.mean_effect_re, res2.TE_random, rtol=1e-13)
        assert_allclose(res1.sd_eff_w_re_hksj, res2.seTE_random, rtol=1e-13)

        th = res1.test_homogeneity()
        q, pv = th
        df = th.df
        assert_allclose(q, res2.Q, rtol=1e-13)
        assert_allclose(pv, res2.pval_Q, rtol=1e-13)
        assert_allclose(df, res2.df_Q, rtol=1e-13)

        assert_allclose(res1.i2, res2.I2, rtol=1e-13)
        assert_allclose(res1.h2, res2.H**2, rtol=1e-13)

        ci = res1.conf_int(use_t=True)  # fe, re, fe_wls, re_wls
        # R meta does not adjust FE for HKSJ, still uses normal dist
        # assert_allclose(ci[0][0], res2.lower_fixed, atol=1e-10)
        # assert_allclose(ci[0][1], res2.upper_fixed, atol=1e-10)
        assert_allclose(ci[3][0], res2.lower_random, rtol=1e-13)
        assert_allclose(ci[3][1], res2.upper_random, rtol=1e-10)

        ci = res1.conf_int(use_t=False)  # fe, re, fe_wls, re_wls
        assert_allclose(ci[0][0], res2.lower_fixed, rtol=1e-13)
        assert_allclose(ci[0][1], res2.upper_fixed, rtol=1e-13)

        weights = 1 / self.var_eff
        mod_glm = GLM(self.eff, np.ones(len(self.eff)),
                      var_weights=weights)
        res_glm = mod_glm.fit()
        assert_allclose(res_glm.params, res2.TE_fixed, rtol=1e-13)

        weights = 1 / (self.var_eff + res1.tau2)
        mod_glm = GLM(self.eff, np.ones(len(self.eff)),
                      var_weights=weights)
        res_glm = mod_glm.fit()
        assert_allclose(res_glm.params, res2.TE_random, rtol=1e-13)

    @pytest.mark.matplotlib
    def test_plot(self):
        # smoke tests
        res1 = self.res1
        # `use_t=False` avoids warning about missing nobs for use_t is true
        res1.plot_forest(use_t=False)
        res1.plot_forest(use_exp=True, use_t=False)
        res1.plot_forest(alpha=0.01, use_t=False)
        with pytest.raises(TypeError, match="unexpected keyword"):
            res1.plot_forest(junk=5, use_t=False)
