# -*- coding: utf-8 -*-
"""

Created on Mon Dec 09 21:29:20 2013

Author: Josef Perktold
"""

import os
import numpy as np
import pandas as pd
import pytest

import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant


from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt


# get data and results as module global for now, TODO: move to class
from .results import results_count_robust_cluster as results_st

cur_dir = os.path.dirname(os.path.abspath(__file__))

filepath = os.path.join(cur_dir, "results", "ships.csv")
data_raw = pd.read_csv(filepath, index_col=False)
data = data_raw.dropna()

#mod = smd.Poisson.from_formula('accident ~ yr_con + op_75_79', data=dat)
# Do not use formula for tests against Stata because intercept needs to be last
endog = data['accident']
exog_data = data['yr_con op_75_79'.split()]
exog = add_constant(exog_data, prepend=False)
group = np.asarray(data['ship'], int)
exposure = np.asarray(data['service'])


# TODO get the test methods from regression/tests
class CheckCountRobustMixin:


    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        if len(res1.params) == (len(res2.params) - 1):
            # Stata includes lnalpha in table for NegativeBinomial
            mask = np.ones(len(res2.params), np.bool_)
            mask[-2] = False
            res2_params = res2.params[mask]
            res2_bse = res2.bse[mask]
        else:
            res2_params = res2.params
            res2_bse = res2.bse

        assert_allclose(res1._results.params, res2_params, 1e-4)

        assert_allclose(self.bse_rob / self.corr_fact, res2_bse, 6e-5)

    @classmethod
    def get_robust_clu(cls):
        res1 = cls.res1
        cov_clu = sw.cov_cluster(res1, group)
        cls.bse_rob = sw.se_cov(cov_clu)

        cls.corr_fact = cls.get_correction_factor(res1)

    @classmethod
    def get_correction_factor(cls, results, sub_kparams=True):
        mod = results.model
        nobs, k_vars = mod.exog.shape

        if sub_kparams:
            # TODO: document why we adjust by k_params for some classes
            #   but not others.
            k_params = len(results.params)
        else:
            k_params = 0

        corr_fact = (nobs - 1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        return np.sqrt(corr_fact)


    def test_oth(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1._results.llf, res2.ll, 1e-4)
        assert_allclose(res1._results.llnull, res2.ll_0, 1e-4)


    def test_ttest(self):
        smt.check_ttest_tvalues(self.res1)


    def test_waldtest(self):
        smt.check_ftest_pvalues(self.res1)


class TestPoissonClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


class TestPoissonCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)
        cls.res1 = res1 = mod.fit(disp=False)

        debug = False
        if debug:
            # for debugging
            cls.bse_nonrobust = cls.res1.bse.copy()
            cls.res1 = res1 = mod.fit(disp=False)
            cls.get_robust_clu()
            cls.res3 = cls.res1
            cls.bse_rob3 = cls.bse_rob.copy()
            cls.res1 = res1 = mod.fit(disp=False)

        from statsmodels.base.covtype import get_robustcov_results

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'cluster',
                                                  groups=group,
                                                  use_correction=True,
                                                  df_correction=True,  #TODO has no effect
                                                  use_t=False, #True,
                                                  use_self=True)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestPoissonHC1Generic(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1
        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)

        from statsmodels.base.covtype import get_robustcov_results

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'HC1', use_self=True)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)


# TODO: refactor xxxFit to full testing results
class TestPoissonCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):


        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)

        # scaling of cov_params_default to match Stata
        # TODO should the default be changed?
        nobs, k_params = mod.exog.shape
        # TODO: this is similar but not identical to logic in
        #   get_correction_factor; can we de-duplicate?
        sc_fact = (nobs-1.) / float(nobs - k_params)

        cls.res1 = mod.fit(disp=False, cov_type='cluster',
                           cov_kwds=dict(groups=group,
                                         use_correction=True,
                                         scaling_factor=1. / sc_fact,
                                         df_correction=True),  #TODO has no effect
                           use_t=False, #True,
                           )

        # The model results, t_test, ... should also work without
        # normalized_cov_params, see #2209
        # Note: we cannot set on the wrapper res1, we need res1._results
        cls.res1._results.normalized_cov_params = None

        cls.bse_rob = cls.res1.bse

        # backwards compatibility with inherited test methods
        cls.corr_fact = 1


    def test_basic_inference(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-7
        assert_allclose(res1.params, res2.params, rtol=1e-8)
        assert_allclose(res1.bse, res2.bse, rtol=rtol)
        assert_allclose(res1.tvalues, res2.tvalues, rtol=rtol, atol=1e-8)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=rtol, atol=1e-20)
        ci = res2.params_table[:, 4:6]
        assert_allclose(res1.conf_int(), ci, rtol=5e-7, atol=1e-20)


class TestPoissonHC1Fit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1
        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False, cov_type='HC1')

        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)


class TestPoissonHC1FitExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_hc1
        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False, cov_type='HC1')

        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)


class TestPoissonCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_clu #nonrobust
        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


class TestPoissonCluExposureGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_clu #nonrobust
        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = res1 = mod.fit(disp=False)

        from statsmodels.base.covtype import get_robustcov_results

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'cluster',
                                                  groups=group,
                                                  use_correction=True,
                                                  df_correction=True,  #TODO has no effect
                                                  use_t=False, #True,
                                                  use_self=True)
        cls.bse_rob = cls.res1.bse #sw.se_cov(cov_clu)

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestGLMPoissonClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit()
        cls.get_robust_clu()


class TestGLMPoissonCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = res1 = mod.fit()

        get_robustcov_results(cls.res1._results, 'cluster',
                                                  groups=group,
                                                  use_correction=True,
                                                  df_correction=True,  #TODO has no effect
                                                  use_t=False, #True,
                                                  use_self=True)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestGLMPoissonHC1Generic(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit()

        #res_hc0_ = cls.res1.get_robustcov_results('HC1')
        get_robustcov_results(cls.res1._results, 'HC1', use_self=True)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)


# TODO: refactor xxxFit to full testing results
class TestGLMPoissonCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = res1 = mod.fit(cov_type='cluster',
                                  cov_kwds=dict(groups=group,
                                                use_correction=True,
                                                df_correction=True),  #TODO has no effect
                                  use_t=False, #True,
                                  )

        # The model results, t_test, ... should also work without
        # normalized_cov_params, see #2209
        # Note: we cannot set on the wrapper res1, we need res1._results
        cls.res1._results.normalized_cov_params = None

        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestGLMPoissonHC1Fit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_hc1
        mod = GLM(endog, exog, family=families.Poisson())
        cls.res1 = mod.fit(cov_type='HC1')

        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1, sub_kparams=False)


class TestNegbinClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = mod.fit(disp=False, gtol=1e-7)
        cls.get_robust_clu()


class TestNegbinCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_exposure_clu #nonrobust
        mod = smd.NegativeBinomial(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


#        mod_nbe = smd.NegativeBinomial(endog, exog, exposure=data['service'])
#        res_nbe = mod_nbe.fit()
#        mod_nb = smd.NegativeBinomial(endog, exog)
#        res_nb = mod_nb.fit()
#
#        cov_clu_nb = sw.cov_cluster(res_nb, group)
#        k_params = k_vars + 1
#        print sw.se_cov(cov_clu_nb / ((nobs-1.) / float(nobs - k_params)))
#
#        wt = res_nb.wald_test(np.eye(len(res_nb.params))[1:3], cov_p=cov_clu_nb/((nobs-1.) / float(nobs - k_params)))
#        print wt
#
#        print dir(results_st)

class TestNegbinCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = res1 = mod.fit(disp=False, gtol=1e-7)

        get_robustcov_results(cls.res1._results, 'cluster',
                                                  groups=group,
                                                  use_correction=True,
                                                  df_correction=True,  #TODO has no effect
                                                  use_t=False, #True,
                                                  use_self=True)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestNegbinCluFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = res1 = mod.fit(disp=False, cov_type='cluster',
                                  cov_kwds=dict(groups=group,
                                                use_correction=True,
                                                df_correction=True),  #TODO has no effect
                                  use_t=False, #True,
                                  gtol=1e-7)
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class TestNegbinCluExposureFit(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_exposure_clu #nonrobust
        mod = smd.NegativeBinomial(endog, exog, exposure=exposure)
        cls.res1 = res1 = mod.fit(disp=False, cov_type='cluster',
                                  cov_kwds=dict(groups=group,
                                                use_correction=True,
                                                df_correction=True),  #TODO has no effect
                                  use_t=False, #True,
                                  )
        cls.bse_rob = cls.res1.bse

        cls.corr_fact = cls.get_correction_factor(cls.res1)


class CheckDiscreteGLM:
    # compare GLM with other models, no verified reference results

    def test_basic(self):
        res1 = self.res1  # GLM model
        res2 = self.res2  # comparison model, discrete or OLS

        assert_equal(res1.cov_type, self.cov_type)
        assert_equal(res2.cov_type, self.cov_type)

        rtol = getattr(res1, 'rtol', 1e-13)
        assert_allclose(res1.params, res2.params, rtol=rtol)
        assert_allclose(res1.bse, res2.bse, rtol=1e-10)

    def test_score_hessian(self):
        res1 = self.res1
        res2 = self.res2

        # We need to fix scale in GLM and OLS,
        # discrete MLE have it always fixed
        if isinstance(res2.model, OLS):
            kwds = {'scale': res2.scale}
        else:
            kwds = {}
        if isinstance(res2.model, OLS):
            sgn = + 1
        else:
            sgn = -1  # see #4714

        score1 = res1.model.score(res1.params * 0.98, scale=res1.scale)
        score2 = res2.model.score(res1.params * 0.98, **kwds)
        assert_allclose(score1, score2, rtol=1e-13)

        hess1 = res1.model.hessian(res1.params, scale=res1.scale)
        hess2 = res2.model.hessian(res1.params, **kwds)
        assert_allclose(hess1, hess2, rtol=1e-10)

        if isinstance(res2.model, OLS):
            # skip the rest
            return
        scoref1 = res1.model.score_factor(res1.params, scale=res1.scale)
        scoref2 = res2.model.score_factor(res1.params, **kwds)
        assert_allclose(scoref1, scoref2, rtol=1e-10)

        hessf1 = res1.model.hessian_factor(res1.params, scale=res1.scale)
        hessf2 = res2.model.hessian_factor(res1.params, **kwds)
        assert_allclose(sgn * hessf1, hessf2, rtol=1e-10)

    def test_score_test(self):
        res1 = self.res1
        res2 = self.res2

        if isinstance(res2.model, OLS):
            # skip
            return

        fitted = self.res1.fittedvalues
        exog_extra = np.column_stack((fitted**2, fitted**3))
        res_lm1 = res1.score_test(exog_extra, cov_type='nonrobust')
        res_lm2 = res2.score_test(exog_extra, cov_type='nonrobust')
        assert_allclose(np.hstack(res_lm1), np.hstack(res_lm2), rtol=5e-7)

    def test_margeff(self):
        if (isinstance(self.res2.model, OLS) or
                hasattr(self.res1.model, "offset")):
            pytest.skip("not available yet")

        marg1 = self.res1.get_margeff()
        marg2 = self.res2.get_margeff()
        assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
        assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)

        marg1 = self.res1.get_margeff(count=True, dummy=True)
        marg2 = self.res2.get_margeff(count=True, dummy=True)
        assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
        assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)


class TestGLMPoisson(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        np.random.seed(987125643)  # not intentional seed
        endog_count = np.random.poisson(endog)
        cls.cov_type = 'HC0'

        mod1 = GLM(endog_count, exog, family=families.Poisson())
        cls.res1 = mod1.fit(cov_type='HC0')

        mod1 = smd.Poisson(endog_count, exog)
        cls.res2 = mod1.fit(cov_type='HC0')

        cls.res1.rtol = 1e-11


class TestGLMLogit(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'

        mod1 = GLM(endog_bin, exog, family=families.Binomial())
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Logit(endog_bin, exog)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))


class TestGLMLogitOffset(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'
        offset = np.ones(endog_bin.shape[0])

        mod1 = GLM(endog_bin, exog, family=families.Binomial(), offset=offset)
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Logit(endog_bin, exog, offset=offset)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

class TestGLMProbit(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'

        mod1 = GLM(endog_bin, exog, family=families.Binomial(link=links.Probit()))
        cls.res1 = mod1.fit(method='newton',
                            cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Probit(endog_bin, exog)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
        cls.rtol = 1e-6

    def test_score_hessian(self):
        res1 = self.res1
        res2 = self.res2
        # Note scale is fixed at 1, so we do not need to fix it explicitly
        score1 = res1.model.score(res1.params * 0.98)
        score2 = res2.model.score(res1.params * 0.98)
        assert_allclose(score1, score2, rtol=1e-13)

        hess1 = res1.model.hessian(res1.params)
        hess2 = res2.model.hessian(res1.params)
        assert_allclose(hess1, hess2, rtol=1e-13)


class TestGLMProbitOffset(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        endog_bin = (endog > endog.mean()).astype(int)
        cls.cov_type = 'cluster'
        offset = np.ones(endog_bin.shape[0])

        mod1 = GLM(endog_bin, exog,
                   family=families.Binomial(link=links.Probit()),
                   offset=offset)
        cls.res1 = mod1.fit(method='newton',
                            cov_type='cluster', cov_kwds=dict(groups=group))

        mod1 = smd.Probit(endog_bin, exog, offset=offset)
        cls.res2 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))
        cls.rtol = 1e-6


class TestGLMGaussNonRobust(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'nonrobust'

        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit()

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit()


class TestGLMGaussClu(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'cluster'

        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='cluster', cov_kwds=dict(groups=group))

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='cluster', cov_kwds=dict(groups=group))


class TestGLMGaussHC(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'HC0'

        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HC0')

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HC0')


class TestGLMGaussHAC(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):

        cls.cov_type = 'HAC'

        kwds={'maxlags':2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds)


class TestGLMGaussHAC2(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):

        cls.cov_type = 'HAC'

        # check kernel specified as string
        kwds = {'kernel': 'bartlett', 'maxlags': 2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        kwds2 = {'maxlags': 2}
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds2)


class TestGLMGaussHACUniform(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):

        cls.cov_type = 'HAC'

        kwds={'kernel':sw.weights_uniform, 'maxlags':2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds)

        #for debugging
        cls.res3 = mod2.fit(cov_type='HAC', cov_kwds={'maxlags':2})


    def test_cov_options(self):

        # check keyword `weights_func
        kwdsa = {'weights_func': sw.weights_uniform, 'maxlags': 2}
        res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsa)
        res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsa)
        assert_allclose(res1a.bse, self.res1.bse, rtol=1e-12)
        assert_allclose(res2a.bse, self.res2.bse, rtol=1e-12)

        # regression test for bse values
        bse = np.array([  2.82203924,   4.60199596,  11.01275064])
        assert_allclose(res1a.bse, bse, rtol=1e-6)

        assert_(res1a.cov_kwds['weights_func'] is sw.weights_uniform)

        kwdsb = {'kernel': sw.weights_bartlett, 'maxlags': 2}
        res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsb)
        res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsb)
        assert_allclose(res1a.bse, res2a.bse, rtol=1e-12)

        # regression test for bse values
        bse = np.array([  2.502264,  3.697807,  9.193303])
        assert_allclose(res1a.bse, bse, rtol=1e-6)



class TestGLMGaussHACUniform2(TestGLMGaussHACUniform):

    @classmethod
    def setup_class(cls):

        cls.cov_type = 'HAC'

        kwds={'kernel': sw.weights_uniform, 'maxlags': 2}
        mod1 = GLM(endog, exog, family=families.Gaussian())
        cls.res1 = mod1.fit(cov_type='HAC', cov_kwds=kwds)

        # check kernel as string
        mod2 = OLS(endog, exog)
        kwds2 = {'kernel': 'uniform', 'maxlags': 2}
        cls.res2 = mod2.fit(cov_type='HAC', cov_kwds=kwds)


class TestGLMGaussHACPanel(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'hac-panel'
        # time index is just made up to have a test case
        time = np.tile(np.arange(7), 5)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(time=time,
                    maxlags=2,
                    kernel=sw.weights_uniform,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)
        cls.res1b = mod1.fit(cov_type='nw-panel', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)

    def test_kwd(self):
        # test corrected keyword name
        assert_allclose(self.res1b.bse, self.res1.bse, rtol=1e-12)


class TestGLMGaussHACPanelGroups(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'hac-panel'
        # time index is just made up to have a test case
        groups = np.repeat(np.arange(5), 7)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(groups=pd.Series(groups),  # check for #3606
                    maxlags=2,
                    kernel=sw.weights_uniform,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)


class TestGLMGaussHACGroupsum(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'hac-groupsum'
        # time index is just made up to have a test case
        time = np.tile(np.arange(7), 5)[:-1]
        mod1 = GLM(endog, exog, family=families.Gaussian())
        kwds = dict(time=pd.Series(time),  # check for #3606
                    maxlags=2,
                    use_correction='hac',
                    df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-groupsum', cov_kwds=kwds)
        cls.res1b = mod1.fit(cov_type='nw-groupsum', cov_kwds=kwds)

        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-groupsum', cov_kwds=kwds)

    def test_kwd(self):
        # test corrected keyword name
        assert_allclose(self.res1b.bse, self.res1.bse, rtol=1e-12)
