# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:18:12 2018

Author: Josef Perktold
"""
from statsmodels.compat.pandas import testing as pdt

import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

from statsmodels.stats.outliers_influence import MLEInfluence

cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'binary_constrict.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
data_bin = pd.read_csv(file_path, index_col=0)

file_name = 'results_influence_logit.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
results_sas_df = pd.read_csv(file_path, index_col=0)


def test_influence_glm_bernoulli():
    # example uses Finney's data and is used in Pregibon 1981

    df = data_bin
    results_sas = np.asarray(results_sas_df)

    res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

    infl = res.get_influence(observed=False)

    k_vars = 3
    assert_allclose(infl.dfbetas, results_sas[:, 5:8], atol=1e-4)
    assert_allclose(infl.d_params, results_sas[:, 5:8] * res.bse.values, atol=1e-4)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-5)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-5)

    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-5)


class InfluenceCompareExact:
    # Mixin to compare and test two Influence instances

    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)

        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12, atol=1e-7)

        cd_rtol = getattr(self, 'cd_rtol', 1e-7)
        assert_allclose(infl0.cooks_distance[0], infl1.cooks_distance[0],
                        rtol=cd_rtol, atol=1e-14)  # very small values possible
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=1e-9, atol=5e-9)
        assert_allclose(infl0.d_params, infl1.d_params, rtol=1e-9, atol=5e-9)
        assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues,
                        rtol=5e-9, atol=1e-14)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.d_fittedvalues_scaled,
                        rtol=5e-9, atol=1e-14)

    @pytest.mark.smoke
    @pytest.mark.matplotlib
    def test_plots(self, close_figures):
        infl1 = self.infl1
        infl0 = self.infl0

        fig = infl0.plot_influence(external=False)
        fig = infl1.plot_influence(external=False)

        fig = infl0.plot_index('resid', threshold=0.2, title='')
        fig = infl1.plot_index('resid', threshold=0.2, title='')

        fig = infl0.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('dfbeta', idx=1, threshold=0.2, title='')

        fig = infl0.plot_index('cook', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('cook', idx=1, threshold=0.2, title='')

        fig = infl0.plot_index('hat', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('hat', idx=1, threshold=0.2, title='')


    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        assert_allclose(df0.values, df1.values, rtol=5e-5, atol=1e-14)
        pdt.assert_index_equal(df0.index, df1.index)


def _check_looo(self):
    infl = self.infl1
    # unwrap if needed
    results = getattr(infl.results, '_results', infl.results)

    res_looo = infl._res_looo
    mask_infl = infl.cooks_distance[0] > 2 * infl.cooks_distance[0].std()
    mask_low = ~mask_infl
    diff_params = results.params - res_looo['params']
    assert_allclose(infl.d_params[mask_low], diff_params[mask_low], atol=0.05)
    assert_allclose(infl.params_one[mask_low], res_looo['params'][mask_low], rtol=0.01)


class TestInfluenceLogitGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceBinomialGLMMLE(InfluenceCompareExact):
    # example based on Williams and R docs

    @classmethod
    def setup_class(cls):
        yi = np.array([0, 2, 14, 19, 30])
        ni = 40 * np.ones(len(yi))
        xi = np.arange(1, len(yi) + 1)
        exog = np.column_stack((np.ones(len(yi)), xi))
        endog = np.column_stack((yi, ni - yi))

        res = GLM(endog, exog, family=families.Binomial()).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)
        cls.cd_rtol = 5e-5

    def test_looo(self):
        _check_looo(self)

    def test_r(self):
        # values from R,
        # > xi <- 1:5
        # > yi <- c(0,2,14,19,30)    # number of mice responding to dose xi
        # > mi <- rep(40, 5)         # number of mice exposed
        # > glmI <- glm(cbind(yi, mi -yi) ~ xi, family = binomial)
        # > imI <- influence.measures(glmI)
        # > t(imI$infmat)

        # dfbeta/dfbetas and dffits do not make sense to me and are furthe away from
        # looo than mine
        # resid seem to be resid_deviance based and not resid_pearson
        # I did not compare cov.r
        infl1 = self.infl1
        cooks_d = [0.25220202795934726, 0.26107981497746285, 1.28985614424132389,
                   0.08449722285516942, 0.36362110845918005]
        hat = [0.2594393406119333,  0.3696442663244837,  0.3535768402250521,
               0.389209198535791057,  0.6281303543027403]

        assert_allclose(infl1.hat_matrix_diag, hat, rtol=5e-6)
        assert_allclose(infl1.cooks_distance[0], cooks_d, rtol=1e-5)


class TestInfluenceGaussianGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        #res = GLM(endog, exog).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceGaussianGLMOLS(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res0 = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        res1 = OLS.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        cls.infl1 = res1.get_influence()
        cls.infl0 = res0.get_influence()

    def test_basics(self):
        # needs to override attributes that are not equivalent,
        # i.e. not available or different definition like external vs internal
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)
        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12, atol=1e-7)
        assert_allclose(infl0.cooks_distance, infl1.cooks_distance,
                        rtol=1e-7, atol=1e-14)  # very small values possible
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=0.1) # changed
        # OLSInfluence only has looo dfbeta/d_params
        assert_allclose(infl0.d_params, infl1.dfbeta, rtol=1e-9, atol=1e-14)
        # d_fittedvalues is not available in OLSInfluence, i.e. only scaled dffits
        # assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=1e-9)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.dffits_internal[0], rtol=1e-9)

        # specific to linear link
        assert_allclose(infl0.d_linpred,
                        infl0.d_fittedvalues, rtol=1e-12)
        assert_allclose(infl0.d_linpred_scaled,
                        infl0.d_fittedvalues_scaled, rtol=1e-12)

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        # just some basic check on overlap except for dfbetas
        cols = ['cooks_d', 'standard_resid', 'hat_diag', 'dffits_internal']
        assert_allclose(df0[cols].values, df1[cols].values, rtol=1e-5)
        pdt.assert_index_equal(df0.index, df1.index)


class TestInfluenceLogitCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
                  family=families.Binomial())
        res = mod.fit(method="newton", tol=1e-10)
        from statsmodels.discrete.discrete_model import Logit
        mod2 = Logit(df['constrict'], df[['const', 'log_rate', 'log_volumne']])
        res2 = mod2.fit(method="newton", tol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = res2.get_influence()


class TestInfluenceProbitCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
                  family=families.Binomial(link=families.links.Probit()))
        res = mod.fit(method="newton", tol=1e-10)
        from statsmodels.discrete.discrete_model import Probit
        mod2 = Probit(df['constrict'], df[['const', 'log_rate', 'log_volumne']])
        res2 = mod2.fit(method="newton", tol=1e-10)

        cls.infl1 = MLEInfluence(res)  # res.get_influence()
        cls.infl0 = res2.get_influence()

    def test_basics_specific(self):
        infl1 = self.infl1
        infl0 = self.infl0
        res1 = self.infl1.results
        res0 = self.infl0.results

        assert_allclose(res1.params, res1.params, rtol=1e-10)

        d1 = res1.model._deriv_mean_dparams(res1.params)
        d0 = res1.model._deriv_mean_dparams(res0.params)
        assert_allclose(d0, d1, rtol=1e-10)

        d1 = res1.model._deriv_score_obs_dendog(res1.params)
        d0 = res1.model._deriv_score_obs_dendog(res0.params)
        assert_allclose(d0, d1, rtol=1e-10)

        s1 = res1.model.score_obs(res1.params)
        s0 = res1.model.score_obs(res0.params)
        assert_allclose(s0, s1, rtol=1e-10)

        assert_allclose(infl0.hessian, infl1.hessian, rtol=1e-10)


class TestInfluencePoissonCompare(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        mod = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
                  family=families.Poisson())
        res = mod.fit(attach_wls=True, atol=1e-10)
        from statsmodels.discrete.discrete_model import Poisson
        mod2 = Poisson(df['constrict'],
                       df[['const', 'log_rate', 'log_volumne']])
        res2 = mod2.fit(tol=1e-10)

        cls.infl0 = res.get_influence()
        cls.infl1 = res2.get_influence()
