# -*- coding: utf-8 -*-
"""
unit test for GAM

Author: Josef Perktold

"""

import os

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd

import pytest

import patsy

from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen

from statsmodels.gam.smooth_basis import (BSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
    GLMGam, GLMGamResults, GLMGamResultsWrapper)

from statsmodels.tools.linalg import matrix_sqrt, transf_constraints

from .results import results_pls, results_mpg_bs, results_mpg_bs_poisson


class PoissonPenalized(PenalizedMixin, Poisson):
    pass


class LogitPenalized(PenalizedMixin, Logit):
    pass


class ProbitPenalized(PenalizedMixin, Probit):
    pass


class GLMPenalized(PenalizedMixin, GLM):
    pass


cur_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(cur_dir, "results", "motorcycle.csv")
data_mcycle = pd.read_csv(file_path)

file_path = os.path.join(cur_dir, "results", "autos.csv")
df_autos_ = pd.read_csv(file_path)
df_autos = df_autos_[['city_mpg', 'fuel', 'drive', 'weight', 'hp']].dropna()


class CheckGAMMixin:

    @classmethod
    def _init(cls):
        # TODO: CyclicCubicSplines raises when using pandas
        cc_h = CyclicCubicSplines(np.asarray(data_mcycle['times']), df=[6])

        constraints = np.atleast_2d(cc_h.basis.mean(0))
        transf = transf_constraints(constraints)

        exog = cc_h.basis.dot(transf)
        penalty_matrix = transf.T.dot(cc_h.penalty_matrices[0]).dot(transf)
        restriction = matrix_sqrt(penalty_matrix)
        return exog, penalty_matrix, restriction

    def test_params(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, rtol=1e-5)
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Vp * self.covp_corrfact, rtol=1e-6, atol=1e-9)

        assert_allclose(res1.scale, res2.scale * self.covp_corrfact,
                        rtol=1e-8)

        assert_allclose(np.asarray(res1.bse),
                        res2.se * np.sqrt(self.covp_corrfact),
                        rtol=1e-6, atol=1e-9)

    def test_fitted(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.fittedvalues, res2.fitted_values,
                        rtol=self.rtol_fitted)

    @pytest.mark.smoke
    def test_null_smoke(self):
        self.res1.llnull


class TestTheilPLS5(CheckGAMMixin):

    cov_type = 'data-prior'

    @classmethod
    def setup_class(cls):
        exog, penalty_matrix, restriction = cls._init()
        endog = data_mcycle['accel']
        modp = TheilGLS(endog, exog, r_matrix=restriction)
        # scaling of penweith in R mgcv
        s_scale_r = 0.02630734
        # Theil penweight uses preliminary sigma2_e to scale penweight
        sigma_e = 1405.7950179165323
        cls.pw = pw = 1 / sigma_e / s_scale_r
        cls.res1 = modp.fit(pen_weight=pw, cov_type=cls.cov_type)
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-7
        cls.covp_corrfact = 0.99786932844203202

    def test_cov_robust(self):
        res1 = self.res1
        res2 = self.res2
        pw = res1.penalization_factor
        res1 = res1.model.fit(pen_weight=pw, cov_type='sandwich')
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Ve * self.covp_corrfact, rtol=1e-4)

    def test_null_smoke(self):
        pytest.skip("llnull not available")


class TestGLMPenalizedPLS5(CheckGAMMixin):

    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        exog, penalty_matrix, restriction = cls._init()
        endog = data_mcycle['accel']
        pen = smpen.L2ConstraintsPenalty(restriction=restriction)
        mod = GLMPenalized(endog, exog, family=family.Gaussian(),
                           penal=pen)
        # scaling of penweight in R mgcv
        s_scale_r = 0.02630734
        # set pen_weight to correspond to R mgcv example
        cls.pw = mod.pen_weight = 1 / s_scale_r / 2
        cls.res1 = mod.fit(cov_type=cls.cov_type, method='bfgs', maxiter=100,
                           disp=0, trim=False, scale='x2')
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-5
        # edf is currently not available with PenalizedMixin
        # need correction for difference in scale denominator
        cls.covp_corrfact = 1.0025464444310588

    def _test_cov_robust(self):
        # TODO: HC0 differs from Theil sandwich, difference is large
        res1 = self.res1
        res2 = self.res2
        pw = res1.model.pen_weight
        res1 = res1.model.fit(pen_weight=pw, cov_type='HC0')
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Ve * self.covp_corrfact, rtol=1e-4)


class TestGAM5Pirls(CheckGAMMixin):

    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214

        x = data_mcycle['times'].values
        endog = data_mcycle['accel']
        cc = CyclicCubicSplines(x, df=[6], constraints='center')
        gam_cc = GLMGam(endog, smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit()
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-12
        # cls.covp_corrfact = 1.0025464444310588  # without edf
        # edf is implemented
        cls.covp_corrfact = 1


class TestGAM5Bfgs(CheckGAMMixin):

    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214

        x = data_mcycle['times'].values
        endog = data_mcycle['accel']
        cc = CyclicCubicSplines(x, df=[6], constraints='center')
        gam_cc = GLMGam(endog, smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-5
        # cls.covp_corrfact = 1.0025464444310588  # without edf
        # edf is implemented
        cls.covp_corrfact = 1

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        predicted = res1.predict(None, res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4],
                        rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4],
                        rtol=self.rtol_fitted)


class TestGAM6Pirls:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214

        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6])
        gam_cc = GLMGam(data_mcycle['accel'], smoother=cc,
                        alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit()

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-7
        pls6_fittedvalues = np.array([
            2.45008146537851, 3.14145063965465, 5.24130119353225,
            6.63476330674223, 7.99704341866374, 13.9351103077006,
            14.5508371638833, 14.785647621276, 15.1176070735895,
            14.8053514054347, 13.790412967255, 13.790412967255,
            11.2997845518655, 9.51681958051473, 8.4811626302547])
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)

        predicted = res1.predict(None, res1.model.smoother.x[2:4])
        assert_allclose(predicted, pls6_fittedvalues[2:4],
                        rtol=self.rtol_fitted)


class TestGAM6Bfgs:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214

        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6])
        gam_cc = GLMGam(data_mcycle['accel'], smoother=cc,
                        alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-5
        pls6_fittedvalues = np.array([
            2.45008146537851, 3.14145063965465, 5.24130119353225,
            6.63476330674223, 7.99704341866374, 13.9351103077006,
            14.5508371638833, 14.785647621276, 15.1176070735895,
            14.8053514054347, 13.790412967255, 13.790412967255,
            11.2997845518655, 9.51681958051473, 8.4811626302547])
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)


class TestGAM6Bfgs0:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214  # noqa: F841

        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6])
        gam_cc = GLMGam(data_mcycle['accel'], smoother=cc,
                        alpha=0)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-5
        pls6_fittedvalues = np.array([
            2.63203377595747, 3.41285892739456, 5.78168657308338,
            7.35344779586831, 8.89178704316853, 15.7035642157176,
            16.4510219628328, 16.7474993878412, 17.3397025587698,
            17.1062522298643, 16.1786066072489, 16.1786066072489,
            13.7402485937614, 11.9531909618517, 10.9073964111009])
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)


pls6_fittedvalues = np.array([
    2.45008146537851, 3.14145063965465, 5.24130119353225,
    6.63476330674223, 7.99704341866374, 13.9351103077006,
    14.5508371638833, 14.785647621276, 15.1176070735895,
    14.8053514054347, 13.790412967255, 13.790412967255,
    11.2997845518655, 9.51681958051473, 8.4811626302547])

pls6_exog = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -0.334312615555276, -0.302733562622373,
    -0.200049479196403, -0.12607681525989, -0.0487229716135211,
    0.397628373646056, 0.475396222437879, 0.51311526571058,
    0.685638355361239, 0.745083051531164, -0.633518318499726,
    -0.634362488928233, -0.635472088268483, -0.634802453890957,
    -0.632796625534419, -0.589886140629009, -0.574834708734556,
    -0.566315983948608, -0.51289784236512, -0.486061743835595,
    -0.353449234316442, -0.348107090921062, -0.328814083307981,
    -0.313617048982477, -0.296913301955505, -0.191949693921079,
    -0.173001127145111, -0.163813487426548, -0.12229019995063,
    -0.108463798212062, -0.33613551740577, -0.327911471033406,
    -0.303620832999443, -0.287786799373968, -0.272279566127816,
    -0.194325957984873, -0.18175817334823, -0.175688807660186,
    -0.147654475500976, -0.137597948224942, -0.406564043706154,
    -0.409594429953082, -0.412391645561287, -0.409453786864986,
    -0.403086590828732, -0.322579243114146, -0.302545882788086,
    -0.29221622484174, -0.239207291311699, -0.218194346676734
    ]).reshape(10, 6, order='F')


class TestGAM6ExogBfgs:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        nobs = data_mcycle['times'].shape[0]
        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6],
                                constraints='center')
        gam_cc = GLMGam(data_mcycle['accel'], np.ones(nobs),
                        smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-5

        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)

    def test_exog(self):
        exog = self.res1.model.exog
        assert_allclose(exog[:10], pls6_exog,
                        rtol=1e-13)


class TestGAM6ExogPirls:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        nobs = data_mcycle['times'].shape[0]
        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6],
                                constraints='center')
        gam_cc = GLMGam(data_mcycle['accel'], np.ones((nobs, 1)),
                        smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='pirls')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-5

        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues,
                        rtol=self.rtol_fitted)

    def test_exog(self):
        exog = self.res1.model.exog
        assert_allclose(exog[:10], pls6_exog,
                        rtol=1e-13)


class TestGAMMPG:

    @classmethod
    def setup_class(cls):

        sp = np.array([6.46225497484073, 0.81532465890585])
        s_scale = np.array([2.95973613706629e-07, 0.000126203730141359])

        x_spline = df_autos[['weight', 'hp']].values
        exog = patsy.dmatrix('fuel + drive', data=df_autos)
        cc = CyclicCubicSplines(x_spline, df=[6, 5], constraints='center')
        # TODO alpha needs to be list
        gam_cc = GLMGam(df_autos['city_mpg'], exog=exog, smoother=cc,
                        alpha=(1 / s_scale * sp / 2).tolist())
        cls.res1a = gam_cc.fit()
        gam_cc = GLMGam(df_autos['city_mpg'], exog=exog, smoother=cc,
                        alpha=(1 / s_scale * sp / 2).tolist())
        cls.res1b = gam_cc.fit(method='newton')

    def test_exog(self):
        file_path = os.path.join(cur_dir, "results", "autos_exog.csv")
        df_exog = pd.read_csv(file_path)
        res2_exog = df_exog.values
        for res1 in [self.res1a, self.res1b]:
            exog = res1.model.exog
            # exog contains zeros
            assert_allclose(exog, res2_exog, atol=1e-14)

    def test_fitted(self):
        file_path = os.path.join(cur_dir, "results", "autos_predict.csv")
        df_pred = pd.read_csv(file_path, index_col="Row.names")
        df_pred.index = df_pred.index - 1
        res2_fittedvalues = df_pred["fit"].values
        res2_se_mean = df_pred["se_fit"].values
        for res1 in [self.res1a, self.res1b]:
            pred = res1.get_prediction()
            self.rtol_fitted = 1e-5

            assert_allclose(res1.fittedvalues, res2_fittedvalues,
                            rtol=1e-10)
            assert_allclose(pred.predicted_mean, res2_fittedvalues,
                            rtol=1e-10)

            # TODO: no edf, edf corrected df_resid
            # scale estimate differs
            # corr_fact = np.sqrt(191.669417019567 / 190)  # without edf
            # edf is implemented
            corr_fact = 1
            assert_allclose(pred.se_mean, res2_se_mean * corr_fact, rtol=1e-10)


class TestGAMMPGBS(CheckGAMMixin):
    # This has matching results from mgcv

    @classmethod
    def setup_class(cls):

        sp = np.array([0.830689464223685, 425.361212061649])
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])

        x_spline = df_autos[['weight', 'hp']].values
        # We need asarray to remove the design_info
        # If design_info is attached,
        #     then exog_linear will also be transformed in predict.
        cls.exog = np.asarray(patsy.dmatrix('fuel + drive', data=df_autos))
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3],
                      variable_names=['weight', 'hp'],
                      constraints='center',
                      include_intercept=True)
        # TODO alpha needs to be list
        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam(df_autos['city_mpg'], exog=cls.exog, smoother=bs,
                        alpha=(alpha0).tolist())
        cls.res1a = gam_bs.fit(use_t=True)

        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs.mpg_bs

        cls.rtol_fitted = 1e-8
        cls.covp_corrfact = 1  # not needed

        # for checking that alpha model attribute is unchanged, same as alpha0
        cls.alpha = [169947.78222669504, 26767.58046340008]

    @classmethod
    def _init(cls):
        pass

    def test_edf(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.edf, res2.edf_all, rtol=1e-6)
        hat = res1.get_hat_matrix_diag()
        assert_allclose(hat, res2.hat, rtol=1e-6)

    def test_smooth(self):
        res1 = self.res1
        res2 = self.res2
        smoothers = res1.model.smoother.smoothers
        pen_matrix0 = smoothers[0].cov_der2
        assert_allclose(pen_matrix0, res2.smooth0.S * res2.smooth0.S_scale,
                        rtol=1e-6)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        predicted = res1.predict(self.exog[2:4], res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4],
                        rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4],
                        rtol=self.rtol_fitted)

    def test_crossval(self):
        # includes some checks that penalization in the model is unchanged
        mod = self.res1.model
        assert_equal(mod.alpha, self.alpha)  # assert unchanged
        assert_allclose(self.res1.scale, 4.7064821354391118, rtol=1e-13)

        alpha_aic = mod.select_penweight()[0]
        # regression number, but in the right ball park
        assert_allclose(alpha_aic, [112487.81362014, 129.89155677], rtol=1e-3)
        assert_equal(mod.alpha, self.alpha)  # assert unchanged
        assert_equal(mod.penal.start_idx, 4)
        pm = mod.penal.penalty_matrix()
        assert_equal(pm[:, :4], 0)
        assert_equal(pm[:4, :], 0)
        assert_allclose(self.res1.scale, 4.7064821354391118, rtol=1e-13)

        np.random.seed(987125)
        alpha_cv, _ = mod.select_penweight_kfold(k_folds=3, k_grid=6)
        # regression number, but in the right ball park
        assert_allclose(alpha_cv, [10000000.0, 630.957344480193], rtol=1e-5)
        assert_equal(mod.alpha, self.alpha)  # assert unchanged
        assert_equal(mod.penal.start_idx, 4)
        pm = mod.penal.penalty_matrix()
        assert_equal(pm[:, :4], 0)
        assert_equal(pm[:4, :], 0)
        assert_allclose(self.res1.scale, 4.7064821354391118, rtol=1e-13)


class TestGAMMPGBSPoisson(CheckGAMMixin):
    # This has matching results from mgcv

    @classmethod
    def setup_class(cls):

        sp = np.array([40491.3940640059, 232455.530262537])
        # s_scale is same as before
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])

        x_spline = df_autos[['weight', 'hp']].values
        cls.exog = patsy.dmatrix('fuel + drive', data=df_autos)
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3],
                      variable_names=['weight', 'hp'],
                      constraints='center',
                      include_intercept=True)
        # TODO alpha needs to be list
        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam(df_autos['city_mpg'], exog=cls.exog, smoother=bs,
                        family=family.Poisson(), alpha=alpha0)

        xnames = cls.exog.design_info.column_names + gam_bs.smoother.col_names
        gam_bs.exog_names[:] = xnames
        cls.res1a = gam_bs.fit(use_t=False)

        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs_poisson.mpg_bs_poisson

        cls.rtol_fitted = 1e-8
        cls.covp_corrfact = 1  # not needed

    @classmethod
    def _init(cls):
        pass

    def test_edf(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.edf, res2.edf_all, rtol=1e-6)
        hat = res1.get_hat_matrix_diag()
        assert_allclose(hat, res2.hat, rtol=1e-6)
        assert_allclose(res1.aic, res2.aic, rtol=1e-8)
        assert_allclose(res1.deviance, res2.deviance, rtol=1e-8)
        assert_allclose(res1.df_resid, res2.residual_df, rtol=1e-8)

    def test_smooth(self):
        res1 = self.res1
        res2 = self.res2

        smoothers = res1.model.smoother.smoothers
        pen_matrix0 = smoothers[0].cov_der2
        assert_allclose(pen_matrix0, res2.smooth0.S * res2.smooth0.S_scale,
                        rtol=1e-6)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        # this uses transform also for exog_linear
        # predicted = res1.predict(self.exog[2:4], res1.model.smoother.x[2:4])
        predicted = res1.predict(df_autos.iloc[2:4],
                                 res1.model.smoother.x[2:4])
        assert_allclose(predicted, res1.fittedvalues[2:4],
                        rtol=1e-13)
        assert_allclose(predicted, res2.fitted_values[2:4],
                        rtol=self.rtol_fitted)

        # linpred = res1.predict(self.exog[2:4], res1.model.smoother.x[2:4],
        #                        linear=True)
        xp = pd.DataFrame(res1.model.smoother.x[2:4])
        linpred = res1.predict(df_autos.iloc[2:4], xp,
                               which="linear")
        assert_allclose(linpred, res2.linear_predictors[2:4],
                        rtol=self.rtol_fitted)

        assert_equal(predicted.index.values, [2, 3])
        assert_equal(linpred.index.values, [2, 3])

    def test_wald(self):
        res1 = self.res1
        res2 = self.res2
        wtt = res1.wald_test_terms(skip_single=True,
                                   combine_terms=['fuel', 'drive',
                                                  'weight', 'hp'],
                                   scalar=True)
        # mgcv has term test for linear part
        assert_allclose(wtt.statistic[:2], res2.pTerms_chi_sq, rtol=1e-7)
        assert_allclose(wtt.pvalues[:2], res2.pTerms_pv, rtol=1e-6)
        assert_equal(wtt.df_constraints[:2], res2.pTerms_df)

    def test_select_alpha(self):
        res1 = self.res1
        alpha_mgcv = res1.model.alpha
        res_s = res1.model.select_penweight()
        assert_allclose(res_s[0], alpha_mgcv, rtol=5e-5)


class TestGAMMPGBSPoissonFormula(TestGAMMPGBSPoisson):
    # This is the same as the previous but with from_formula

    @classmethod
    def setup_class(cls):

        sp = np.array([40491.3940640059, 232455.530262537])
        # s_scale is same as before
        cls.s_scale = s_scale = np.array([2.443955e-06, 0.007945455])

        cls.exog = patsy.dmatrix('fuel + drive', data=df_autos)

        x_spline = df_autos[['weight', 'hp']].values
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3],
                      variable_names=['weight', 'hp'],
                      constraints='center',
                      include_intercept=True)

        alpha0 = 1 / s_scale * sp / 2
        gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', df_autos,
                                     smoother=bs, family=family.Poisson(),
                                     alpha=alpha0)

        cls.res1a = gam_bs.fit(use_t=False)

        cls.res1b = gam_bs.fit(method='newton', use_t=True)
        cls.res1 = cls.res1a._results
        cls.res2 = results_mpg_bs_poisson.mpg_bs_poisson

        cls.rtol_fitted = 1e-8
        cls.covp_corrfact = 1  # not needed

    def test_names_wrapper(self):
        res1a = self.res1a
        xnames = ['Intercept', 'fuel[T.gas]', 'drive[T.fwd]', 'drive[T.rwd]',
                  'weight_s0', 'weight_s1', 'weight_s2', 'weight_s3',
                  'weight_s4', 'weight_s5', 'weight_s6', 'weight_s7',
                  'weight_s8', 'weight_s9', 'weight_s10',
                  'hp_s0', 'hp_s1', 'hp_s2', 'hp_s3', 'hp_s4', 'hp_s5',
                  'hp_s6', 'hp_s7', 'hp_s8']

        assert_equal(res1a.model.exog_names, xnames)
        assert_equal(res1a.model.design_info_linear.column_names,
                     xnames[:4])

        assert_equal(res1a.fittedvalues.iloc[2:4].index.values, [2, 3])
        assert_equal(res1a.params.index.values, xnames)
        assert_(isinstance(res1a.params, pd.Series))

        assert_(isinstance(res1a, GLMGamResultsWrapper))
        assert_(isinstance(res1a._results, GLMGamResults))
