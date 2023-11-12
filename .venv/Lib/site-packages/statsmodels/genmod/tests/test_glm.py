"""
Test functions for models.GLM
"""
import os
import warnings

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
)
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
    approx_fprime,
    approx_fprime_cs,
    approx_hess,
    approx_hess_cs,
)
from statsmodels.tools.sm_exceptions import (
    DomainWarning,
    PerfectSeparationWarning,
    ValueWarning,
)
from statsmodels.tools.tools import add_constant

# Test Precisions
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

pdf_output = False

if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_glm.pdf")
else:
    pdf = None


def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)


def teardown_module():
    if pdf_output:
        pdf.close()


@pytest.fixture(scope="module")
def iris():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return np.genfromtxt(os.path.join(cur_dir, 'results', 'iris.csv'),
                         delimiter=",", skip_header=1)


class CheckModelResultsMixin:
    '''
    res2 should be either the results from RModelWrap
    or the results as defined in model_results_data
    '''

    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params,
                self.decimal_params)

    decimal_bse = DECIMAL_4
    def test_standard_errors(self):
        assert_allclose(self.res1.bse, self.res2.bse,
                        atol=10**(-self.decimal_bse), rtol=1e-5)

    decimal_resids = DECIMAL_4
    def test_residuals(self):
        # fix incorrect numbers in resid_working results
        # residuals for Poisson are also tested in test_glm_weights.py
        import copy

        # new numpy would have copy method
        resid2 = copy.copy(self.res2.resids)
        resid2[:, 2] *= self.res1.family.link.deriv(self.res1.mu)**2

        atol = 10**(-self.decimal_resids)
        resid_a = self.res1.resid_anscombe_unscaled
        resids = np.column_stack((self.res1.resid_pearson,
                self.res1.resid_deviance, self.res1.resid_working,
                resid_a, self.res1.resid_response))
        assert_allclose(resids, resid2, rtol=1e-6, atol=atol)

    decimal_aic_R = DECIMAL_4

    def test_aic_R(self):
        # R includes the estimation of the scale as a lost dof
        # Does not with Gamma though
        if self.res1.scale != 1:
            dof = 2
        else:
            dof = 0
        if isinstance(self.res1.model.family, (sm.families.NegativeBinomial)):
            llf = self.res1.model.family.loglike(self.res1.model.endog,
                                                 self.res1.mu,
                                                 self.res1.model.var_weights,
                                                 self.res1.model.freq_weights,
                                                 scale=1)
            aic = (-2*llf+2*(self.res1.df_model+1))
        else:
            aic = self.res1.aic
        assert_almost_equal(aic+dof, self.res2.aic_R,
                self.decimal_aic_R)

    decimal_aic_Stata = DECIMAL_4
    def test_aic_Stata(self):
        # Stata uses the below llf for aic definition for these families
        if isinstance(self.res1.model.family, (sm.families.Gamma,
                                               sm.families.InverseGaussian,
                                               sm.families.NegativeBinomial)):
            llf = self.res1.model.family.loglike(self.res1.model.endog,
                                                 self.res1.mu,
                                                 self.res1.model.var_weights,
                                                 self.res1.model.freq_weights,
                                                 scale=1)
            aic = (-2*llf+2*(self.res1.df_model+1))/self.res1.nobs
        else:
            aic = self.res1.aic/self.res1.nobs
        assert_almost_equal(aic, self.res2.aic_Stata, self.decimal_aic_Stata)

    decimal_deviance = DECIMAL_4
    def test_deviance(self):
        assert_almost_equal(self.res1.deviance, self.res2.deviance,
                self.decimal_deviance)

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                self.decimal_scale)

    decimal_loglike = DECIMAL_4
    def test_loglike(self):
        # Stata uses the below llf for these families
        # We differ with R for them
        if isinstance(self.res1.model.family, (sm.families.Gamma,
                                               sm.families.InverseGaussian,
                                               sm.families.NegativeBinomial)):
            llf = self.res1.model.family.loglike(self.res1.model.endog,
                                                 self.res1.mu,
                                                 self.res1.model.var_weights,
                                                 self.res1.model.freq_weights,
                                                 scale=1)
        else:
            llf = self.res1.llf
        assert_almost_equal(llf, self.res2.llf, self.decimal_loglike)

    decimal_null_deviance = DECIMAL_4
    def test_null_deviance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DomainWarning)

            assert_almost_equal(self.res1.null_deviance,
                                self.res2.null_deviance,
                                self.decimal_null_deviance)

    decimal_bic = DECIMAL_4
    def test_bic(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_almost_equal(self.res1.bic,
                                self.res2.bic_Stata,
                                self.decimal_bic)

    def test_degrees(self):
        assert_equal(self.res1.model.df_resid,self.res2.df_resid)

    decimal_fittedvalues = DECIMAL_4
    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues,
                self.decimal_fittedvalues)

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))
        if isinstance(tvalues, pd.Series):
            assert_series_equal(self.res1.tvalues, tvalues)
        else:
            assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)

    def test_pearson_chi2(self):
        if hasattr(self.res2, 'pearson_chi2'):
            assert_allclose(self.res1.pearson_chi2, self.res2.pearson_chi2,
                            atol=1e-6, rtol=1e-6)

    def test_prsquared(self):
        if hasattr(self.res2, 'prsquared'):
            assert_allclose(self.res1.pseudo_rsquared(kind="mcf"),
                            self.res2.prsquared, rtol=0.05)

        if hasattr(self.res2, 'prsquared_cox_snell'):
            assert_allclose(float(self.res1.pseudo_rsquared(kind="cs")),
                            self.res2.prsquared_cox_snell, rtol=0.05)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DomainWarning)
            self.res1.summary2()

    def test_get_distribution(self):
        res1 = self.res1
        if not hasattr(res1.model.family, "get_distribution"):
            # only Tweedie has not get_distribution
            pytest.skip("get_distribution not available")

        if isinstance(res1.model.family, sm.families.NegativeBinomial):
            res_scale = 1  # QMLE scale can differ from 1
        else:
            res_scale = res1.scale

        distr = res1.model.family.get_distribution(res1.fittedvalues,
                                                   res_scale)
        var_endog = res1.model.family.variance(res1.fittedvalues) * res_scale
        m, v = distr.stats()
        assert_allclose(res1.fittedvalues, m, rtol=1e-13)
        assert_allclose(var_endog, v, rtol=1e-13)
        # check model method
        distr2 = res1.model.get_distribution(res1.params, res_scale)
        for k in distr2.kwds:
            assert_allclose(distr.kwds[k], distr2.kwds[k], rtol=1e-13)

        # compare var with predict
        var_ = res1.predict(which="var_unscaled")
        assert_allclose(var_ * res_scale, var_endog, rtol=1e-13)

        # check get_distribution of results instance
        if getattr(self, "has_edispersion", False):
            with pytest.warns(UserWarning, match="using scale=1"):
                distr3 = res1.get_distribution()
        else:
            distr3 = res1.get_distribution()
        for k in distr2.kwds:
            assert_allclose(distr3.kwds[k], distr2.kwds[k], rtol=1e-13)


class CheckComparisonMixin:

    def test_compare_discrete(self):
        res1 = self.res1
        resd = self.resd

        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params * 0.98)
        score_obsd = resd.model.score_obs(resd.params * 0.98)
        assert_allclose(score_obs1, score_obsd, rtol=1e-10)

        # score
        score1 = res1.model.score(res1.params * 0.98)
        assert_allclose(score1, score_obs1.sum(0), atol=1e-20)
        score0 = res1.model.score(res1.params)
        assert_allclose(score0, np.zeros(score_obs1.shape[1]), atol=5e-7)

        hessian1 = res1.model.hessian(res1.params * 0.98, observed=False)
        hessiand = resd.model.hessian(resd.params * 0.98)
        assert_allclose(hessian1, hessiand, rtol=1e-10)

        hessian1 = res1.model.hessian(res1.params * 0.98, observed=True)
        hessiand = resd.model.hessian(resd.params * 0.98)
        assert_allclose(hessian1, hessiand, rtol=1e-9)

    def test_score_test(self):
        res1 = self.res1
        # fake example, should be zero, k_constraint should be 0
        st, pv, df = res1.model.score_test(res1.params, k_constraints=1)
        assert_allclose(st, 0, atol=1e-20)
        assert_allclose(pv, 1, atol=1e-10)
        assert_equal(df, 1)

        st, pv, df = res1.model.score_test(res1.params, k_constraints=0)
        assert_allclose(st, 0, atol=1e-20)
        assert_(np.isnan(pv), msg=repr(pv))
        assert_equal(df, 0)

        # TODO: no verified numbers largely SMOKE test
        exog_extra = res1.model.exog[:,1]**2
        st, pv, df = res1.model.score_test(res1.params, exog_extra=exog_extra)
        assert_array_less(0.1, st)
        assert_array_less(0.1, pv)
        assert_equal(df, 1)

    def test_get_prediction(self):
        pred1 = self.res1.get_prediction()  # GLM
        predd = self.resd.get_prediction()  # discrete class
        assert_allclose(predd.predicted, pred1.predicted_mean, rtol=1e-11)
        assert_allclose(predd.se, pred1.se_mean, rtol=1e-6)
        assert_allclose(predd.summary_frame().values,
                        pred1.summary_frame().values, rtol=1e-6)

        pred1 = self.res1.get_prediction(which="mean")  # GLM
        predd = self.resd.get_prediction()  # discrete class
        assert_allclose(predd.predicted, pred1.predicted, rtol=1e-11)
        assert_allclose(predd.se, pred1.se, rtol=1e-6)
        assert_allclose(predd.summary_frame().values,
                        pred1.summary_frame().values, rtol=1e-6)


class TestGlmGaussian(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        cls.decimal_resids = DECIMAL_3
        cls.decimal_params = DECIMAL_2
        cls.decimal_bic = DECIMAL_0
        cls.decimal_bse = DECIMAL_3

        from statsmodels.datasets.longley import load
        cls.data = load()
        cls.data.endog = np.require(cls.data.endog, requirements="W")
        cls.data.exog = np.require(cls.data.exog, requirements="W")
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GLM(cls.data.endog, cls.data.exog,
                        family=sm.families.Gaussian()).fit()
        from .results.results_glm import Longley
        cls.res2 = Longley()


    def test_compare_OLS(self):
        res1 = self.res1
        # OLS does not define score_obs
        from statsmodels.regression.linear_model import OLS
        resd = OLS(self.data.endog, self.data.exog).fit(use_t=False)
        self.resd = resd  # attach to access from the outside

        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params, scale=None)
        score_obsd = resd.resid[:, None] / resd.scale * resd.model.exog
        # low precision because of badly scaled exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-8)

        score_obs1 = res1.model.score_obs(res1.params, scale=1)
        score_obsd = resd.resid[:, None] * resd.model.exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-8)

        hess_obs1 = res1.model.hessian(res1.params, scale=None)
        hess_obsd = -1. / resd.scale * resd.model.exog.T.dot(resd.model.exog)
        # low precision because of badly scaled exog
        assert_allclose(hess_obs1, hess_obsd, rtol=1e-8)

        pred1 = res1.get_prediction()  # GLM
        predd = resd.get_prediction()  # discrete class
        assert_allclose(predd.predicted, pred1.predicted_mean, rtol=1e-11)
        assert_allclose(predd.se, pred1.se_mean, rtol=1e-6)
        assert_allclose(predd.summary_frame().values[:, :4],
                        pred1.summary_frame().values, rtol=1e-6)

        pred1 = self.res1.get_prediction(which="mean")  # GLM
        predd = self.resd.get_prediction()  # discrete class
        assert_allclose(predd.predicted, pred1.predicted, rtol=1e-11)
        assert_allclose(predd.se, pred1.se, rtol=1e-6)
        assert_allclose(predd.summary_frame().values[:, :4],
                        pred1.summary_frame().values, rtol=1e-6)

# FIXME: enable or delete
#    def setup_method(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        Gauss = r.gaussian
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm, family=Gauss)
#        self.res2.resids = np.array(self.res2.resid)[:,None]*np.ones((1,5))
#        self.res2.null_deviance = 185008826 # taken from R. Rpy bug?


class TestGlmGaussianGradient(TestGlmGaussian):
    @classmethod
    def setup_class(cls):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        cls.decimal_resids = DECIMAL_3
        cls.decimal_params = DECIMAL_2
        cls.decimal_bic = DECIMAL_0
        cls.decimal_bse = DECIMAL_2

        from statsmodels.datasets.longley import load
        cls.data = load()
        cls.data.endog = np.require(cls.data.endog, requirements="W")
        cls.data.exog = np.require(cls.data.exog, requirements="W")
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GLM(cls.data.endog, cls.data.exog,
                       family=sm.families.Gaussian()).fit(method='newton')
        from .results.results_glm import Longley
        cls.res2 = Longley()


class TestGaussianLog(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precision
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_aic_Stata = DECIMAL_2
        cls.decimal_loglike = DECIMAL_0
        cls.decimal_null_deviance = DECIMAL_1

        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
#        y = 1.0 - .02*x - .001*x**2 + 0.001 * np.random.randn(nobs)
        cls.X = np.c_[np.ones((nobs,1)),x,x**2]
        cls.lny = np.exp(-(-1.0 + 0.02*x + 0.0001*x**2)) +\
                        0.001 * np.random.randn(nobs)

        GaussLog_Model = GLM(cls.lny, cls.X,
                             family=sm.families.Gaussian(sm.families.links.Log()))
        cls.res1 = GaussLog_Model.fit()
        from .results.results_glm import GaussianLog
        cls.res2 = GaussianLog()

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed"
#        GaussLogLink = r.gaussian(link = "log")
#        GaussLog_Res_R = RModel(cls.lny, cls.X, r.glm, family=GaussLogLink)
#        cls.res2 = GaussLog_Res_R

class TestGaussianInverse(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precisions
        cls.decimal_bic = DECIMAL_1
        cls.decimal_aic_R = DECIMAL_1
        cls.decimal_aic_Stata = DECIMAL_3
        cls.decimal_loglike = DECIMAL_1
        cls.decimal_resids = DECIMAL_3

        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
        y = 1.0 + 2.0 * x + x**2 + 0.1 * np.random.randn(nobs)
        cls.X = np.c_[np.ones((nobs,1)),x,x**2]
        cls.y_inv = (1. + .02*x + .001*x**2)**-1 + .001 * np.random.randn(nobs)
        InverseLink_Model = GLM(cls.y_inv, cls.X,
                family=sm.families.Gaussian(sm.families.links.InversePower()))
        InverseLink_Res = InverseLink_Model.fit()
        cls.res1 = InverseLink_Res
        from .results.results_glm import GaussianInverse
        cls.res2 = GaussianInverse()

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        InverseLink = r.gaussian(link = "inverse")
#        InverseLink_Res_R = RModel(cls.y_inv, cls.X, r.glm, family=InverseLink)
#        cls.res2 = InverseLink_Res_R

class TestGlmBinomial(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        '''
        Test Binomial family with canonical logit link using star98 dataset.
        '''
        cls.decimal_resids = DECIMAL_1
        cls.decimal_bic = DECIMAL_2

        from statsmodels.datasets.star98 import load

        from .results.results_glm import Star98
        data = load()
        data.endog = np.require(data.endog, requirements="W")
        data.exog = np.require(data.exog, requirements="W")
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GLM(data.endog, data.exog,
                       family=sm.families.Binomial()).fit()
        # NOTE: if you want to replicate with RModel
        # res2 = RModel(data.endog[:,0]/trials, data.exog, r.glm,
        #        family=r.binomial, weights=trials)

        cls.res2 = Star98()

    def test_endog_dtype(self):
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        endog = data.endog.astype(int)
        res2 = GLM(endog, data.exog, family=sm.families.Binomial()).fit()
        assert_allclose(res2.params, self.res1.params)
        endog = data.endog.astype(np.double)
        res3 = GLM(endog, data.exog, family=sm.families.Binomial()).fit()
        assert_allclose(res3.params, self.res1.params)

    def test_invalid_endog(self, reset_randomstate):
        # GH2733 inspired check
        endog = np.random.randint(0, 100, size=(1000, 3))
        exog = np.random.standard_normal((1000, 2))
        with pytest.raises(ValueError, match='endog has more than 2 columns'):
            GLM(endog, exog, family=sm.families.Binomial())

    def test_invalid_endog_formula(self, reset_randomstate):
        # GH2733
        n = 200
        exog = np.random.normal(size=(n, 2))
        endog = np.random.randint(0, 3, size=n).astype(str)
        # formula interface
        data = pd.DataFrame({"y": endog, "x1": exog[:, 0], "x2": exog[:, 1]})
        with pytest.raises(ValueError, match='array with multiple columns'):
            sm.GLM.from_formula("y ~ x1 + x2", data,
                                family=sm.families.Binomial())

    def test_get_distribution_binom_count(self):
        # test for binomial counts with n_trials > 1
        res1 = self.res1
        res_scale = 1  # QMLE scale can differ from 1

        mu_prob = res1.fittedvalues
        n = res1.model.n_trials
        distr = res1.model.family.get_distribution(mu_prob, res_scale,
                                                   n_trials=n)
        var_endog = res1.model.family.variance(mu_prob) * res_scale
        m, v = distr.stats()
        assert_allclose(mu_prob * n, m, rtol=1e-13)
        assert_allclose(var_endog * n, v, rtol=1e-13)

        # check model method
        distr2 = res1.model.get_distribution(res1.params, res_scale,
                                             n_trials=n)
        for k in distr2.kwds:
            assert_allclose(distr.kwds[k], distr2.kwds[k], rtol=1e-13)


# FIXME: enable/xfail/skip or delete
# TODO:
# Non-Canonical Links for the Binomial family require the algorithm to be
# slightly changed
# class TestGlmBinomialLog(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialLogit(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialProbit(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialCloglog(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialPower(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialLoglog(CheckModelResultsMixin):
#    pass

# class TestGlmBinomialLogc(CheckModelResultsMixin):
# TODO: need include logc link
#    pass


class TestGlmBernoulli(CheckModelResultsMixin, CheckComparisonMixin):
    @classmethod
    def setup_class(cls):
        from .results.results_glm import Lbw
        cls.res2 = Lbw()
        cls.res1 = GLM(cls.res2.endog, cls.res2.exog,
                       family=sm.families.Binomial()).fit()

        modd = discrete.Logit(cls.res2.endog, cls.res2.exog)
        cls.resd = modd.fit(start_params=cls.res1.params * 0.9, disp=False)

    def test_score_r(self):
        res1 = self.res1
        res2 = self.res2
        st, pv, df = res1.model.score_test(res1.params,
                                           exog_extra=res1.model.exog[:, 1]**2)
        st_res = 0.2837680293459376  # (-0.5326988167303712)**2
        assert_allclose(st, st_res, rtol=1e-4)

        st, pv, df = res1.model.score_test(res1.params,
                                          exog_extra=res1.model.exog[:, 0]**2)
        st_res = 0.6713492821514992  # (-0.8193590679009413)**2
        assert_allclose(st, st_res, rtol=1e-4)

        select = list(range(9))
        select.pop(7)

        res1b = GLM(res2.endog, res2.exog.iloc[:, select],
                    family=sm.families.Binomial()).fit()
        tres = res1b.model.score_test(res1b.params,
                                      exog_extra=res1.model.exog[:, -2])
        tres = np.asarray(tres[:2]).ravel()
        tres_r = (2.7864148487452, 0.0950667)
        assert_allclose(tres, tres_r, rtol=1e-4)

        cmd_r = """\
        data = read.csv("...statsmodels\\statsmodels\\genmod\\tests\\results\\stata_lbw_glm.csv")

        data["race_black"] = data["race"] == "black"
        data["race_other"] = data["race"] == "other"
        mod = glm(low ~ age + lwt + race_black + race_other + smoke + ptl + ht + ui, family=binomial, data=data)
        options(digits=16)
        anova(mod, test="Rao")

        library(statmod)
        s = glm.scoretest(mod, data["age"]**2)
        s**2
        s = glm.scoretest(mod, data["lwt"]**2)
        s**2
        """

# class TestGlmBernoulliIdentity(CheckModelResultsMixin):
#    pass

# class TestGlmBernoulliLog(CheckModelResultsMixin):
#    pass

# class TestGlmBernoulliProbit(CheckModelResultsMixin):
#    pass

# class TestGlmBernoulliCloglog(CheckModelResultsMixin):
#    pass

# class TestGlmBernoulliPower(CheckModelResultsMixin):
#    pass

# class TestGlmBernoulliLoglog(CheckModelResultsMixin):
#    pass

# class test_glm_bernoulli_logc(CheckModelResultsMixin):
#    pass


class TestGlmGamma(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        '''
        Tests Gamma family with canonical inverse link (power -1)
        '''
        # Test Precisions
        cls.decimal_aic_R = -1 #TODO: off by about 1, we are right with Stata
        cls.decimal_resids = DECIMAL_2

        from statsmodels.datasets.scotland import load

        from .results.results_glm import Scotvote
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res1 = GLM(data.endog, data.exog,
                       family=sm.families.Gamma()).fit()
        cls.res1 = res1
#        res2 = RModel(data.endog, data.exog, r.glm, family=r.Gamma)
        res2 = Scotvote()
        res2.aic_R += 2 # R does not count degree of freedom for scale with gamma
        cls.res2 = res2


class TestGlmGammaLog(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precisions
        cls.decimal_resids = DECIMAL_3
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_fittedvalues = DECIMAL_3

        from .results.results_glm import CancerLog
        res2 = CancerLog()
        cls.res1 = GLM(res2.endog, res2.exog,
            family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        cls.res2 = res2

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        cls.res2 = RModel(cls.data.endog, cls.data.exog, r.glm,
#            family=r.Gamma(link="log"))
#        cls.res2.null_deviance = 27.92207137420696 # From R (bug in rpy)
#        cls.res2.bic = -154.1582089453923 # from Stata


class TestGlmGammaIdentity(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precisions
        cls.decimal_resids = -100 #TODO Very off from Stata?
        cls.decimal_params = DECIMAL_2
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_loglike = DECIMAL_1

        from .results.results_glm import CancerIdentity
        res2 = CancerIdentity()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fam = sm.families.Gamma(link=sm.families.links.Identity())
            cls.res1 = GLM(res2.endog, res2.exog, family=fam).fit()
        cls.res2 = res2

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        cls.res2 = RModel(cls.data.endog, cls.data.exog, r.glm,
#            family=r.Gamma(link="identity"))
#        cls.res2.null_deviance = 27.92207137420696 # from R, Rpy bug

class TestGlmPoisson(CheckModelResultsMixin, CheckComparisonMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        '''
        from .results.results_glm import Cpunish
        cls.data = cpunish.load()
        cls.data.endog = np.require(cls.data.endog, requirements="W")
        cls.data.exog = np.require(cls.data.exog, requirements="W")
        cls.data.exog[:, 3] = np.log(cls.data.exog[:, 3])
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        cls.res1 = GLM(cls.data.endog, cls.data.exog,
                       family=sm.families.Poisson()).fit()
        cls.res2 = Cpunish()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(cls.data.endog, cls.data.exog)
        cls.resd = modd.fit(start_params=cls.res1.params * 0.9, disp=False)

#class TestGlmPoissonIdentity(CheckModelResultsMixin):
#    pass

#class TestGlmPoissonPower(CheckModelResultsMixin):
#    pass


class TestGlmInvgauss(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests the Inverse Gaussian family in GLM.

        Notes
        -----
        Used the rndivgx.ado file provided by Hardin and Hilbe to
        generate the data.  Results are read from model_results, which
        were obtained by running R_ig.s
        '''
        # Test Precisions
        cls.decimal_aic_R = DECIMAL_0
        cls.decimal_loglike = DECIMAL_0

        from .results.results_glm import InvGauss
        res2 = InvGauss()
        res1 = GLM(res2.endog, res2.exog,
                   family=sm.families.InverseGaussian()).fit()
        cls.res1 = res1
        cls.res2 = res2

    def test_get_distribution(self):
        res1 = self.res1
        distr = res1.model.family.get_distribution(res1.fittedvalues,
                                                   res1.scale)
        var_endog = res1.model.family.variance(res1.fittedvalues) * res1.scale
        m, v = distr.stats()
        assert_allclose(res1.fittedvalues, m, rtol=1e-13)
        assert_allclose(var_endog, v, rtol=1e-13)


class TestGlmInvgaussLog(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precisions
        cls.decimal_aic_R = -10 # Big difference vs R.
        cls.decimal_resids = DECIMAL_3

        from .results.results_glm import InvGaussLog
        res2 = InvGaussLog()
        cls.res1 = GLM(res2.endog, res2.exog,
            family=sm.families.InverseGaussian(
                link=sm.families.links.Log())).fit()
        cls.res2 = res2

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        cls.res2 = RModel(cls.data.endog, cls.data.exog, r.glm,
#            family=r.inverse_gaussian(link="log"))
#        cls.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
#        cls.res2.llf = -12162.72308 # from Stata, R's has big rounding diff


class TestGlmInvgaussIdentity(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        # Test Precisions
        cls.decimal_aic_R = -10 #TODO: Big difference vs R
        cls.decimal_fittedvalues = DECIMAL_3
        cls.decimal_params = DECIMAL_3

        from .results.results_glm import Medpar1
        data = Medpar1()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.res1 = GLM(data.endog, data.exog,
                            family=sm.families.InverseGaussian(
                                link=sm.families.links.Identity())).fit()
        from .results.results_glm import InvGaussIdentity
        cls.res2 = InvGaussIdentity()

# FIXME: enable or delete
#    def setup(cls):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        cls.res2 = RModel(cls.data.endog, cls.data.exog, r.glm,
#            family=r.inverse_gaussian(link="identity"))
#        cls.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
#        cls.res2.llf = -12163.25545    # from Stata, big diff with R


class TestGlmNegbinomial(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        '''
        Test Negative Binomial family with log link
        '''
        # Test Precision
        cls.decimal_resid = DECIMAL_1
        cls.decimal_params = DECIMAL_3
        cls.decimal_resids = -1 # 1 % mismatch at 0
        cls.decimal_fittedvalues = DECIMAL_1

        from statsmodels.datasets.committee import load
        cls.data = load()
        cls.data.endog = np.require(cls.data.endog, requirements="W")
        cls.data.exog = np.require(cls.data.exog, requirements="W")
        cls.data.exog[:,2] = np.log(cls.data.exog[:,2])
        interaction = cls.data.exog[:,2]*cls.data.exog[:,1]
        cls.data.exog = np.column_stack((cls.data.exog,interaction))
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DomainWarning)
            with pytest.warns(UserWarning):
                fam = sm.families.NegativeBinomial()

        cls.res1 = GLM(cls.data.endog, cls.data.exog,
                family=fam).fit(scale='x2')
        from .results.results_glm import Committee
        res2 = Committee()
        res2.aic_R += 2 # They do not count a degree of freedom for the scale
        cls.res2 = res2
        cls.has_edispersion = True

# FIXME: enable or delete
#    def setup_method(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed"
#        r.library('MASS')  # this does not work when done in rmodelwrap?
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#                family=r.negative_binomial(1))
#        self.res2.null_deviance = 27.8110469364343

# FIXME: enable/xfail/skip or delete
#class TestGlmNegbinomial_log(CheckModelResultsMixin):
#    pass

# FIXME: enable/xfail/skip or delete
#class TestGlmNegbinomial_power(CheckModelResultsMixin):
#    pass

# FIXME: enable/xfail/skip or delete
#class TestGlmNegbinomial_nbinom(CheckModelResultsMixin):
#    pass


class TestGlmPoissonOffset(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        from .results.results_glm import Cpunish_offset
        cls.decimal_params = DECIMAL_4
        cls.decimal_bse = DECIMAL_4
        cls.decimal_aic_R = 3
        data = cpunish.load()
        data.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        data.exog[:, 3] = np.log(data.exog[:, 3])
        data.exog = add_constant(data.exog, prepend=True)
        exposure = [100] * len(data.endog)
        cls.data = data
        cls.exposure = exposure
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Poisson(),
                       exposure=exposure).fit()
        cls.res2 = Cpunish_offset()

    def test_missing(self):
        # make sure offset is dropped correctly
        endog = self.data.endog.copy()
        endog[[2,4,6,8]] = np.nan
        mod = GLM(endog, self.data.exog, family=sm.families.Poisson(),
                    exposure=self.exposure, missing='drop')
        assert_equal(mod.exposure.shape[0], 13)

    def test_offset_exposure(self):
        # exposure=x and offset=log(x) should have the same effect
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100,3))
        exposure = np.random.uniform(1, 2, 100)
        offset = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(),
                   offset=offset, exposure=exposure).fit()
        offset2 = offset + np.log(exposure)
        mod2 = GLM(endog, exog, family=sm.families.Poisson(),
                   offset=offset2).fit()
        assert_almost_equal(mod1.params, mod2.params)
        assert_allclose(mod1.null, mod2.null, rtol=1e-10)

        # test recreating model
        mod1_ = mod1.model
        kwds = mod1_._get_init_kwds()
        assert_allclose(kwds['exposure'], exposure, rtol=1e-14)
        assert_allclose(kwds['offset'], mod1_.offset, rtol=1e-14)
        mod3 = mod1_.__class__(mod1_.endog, mod1_.exog, **kwds)
        assert_allclose(mod3.exposure, mod1_.exposure, rtol=1e-14)
        assert_allclose(mod3.offset, mod1_.offset, rtol=1e-14)

        # test fit_regularized exposure, see #4605
        resr1 = mod1.model.fit_regularized()
        resr2 = mod2.model.fit_regularized()
        assert_allclose(resr1.params, resr2.params, rtol=1e-10)


    def test_predict(self):
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100,3))
        exposure = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(),
                   exposure=exposure).fit()
        exog1 = np.random.normal(size=(10,3))
        exposure1 = np.random.uniform(1, 2, 10)

        # Doubling exposure time should double expected response
        pred1 = mod1.predict(exog=exog1, exposure=exposure1)
        pred2 = mod1.predict(exog=exog1, exposure=2*exposure1)
        assert_almost_equal(pred2, 2*pred1)

        # Check exposure defaults
        pred3 = mod1.predict()
        pred4 = mod1.predict(exposure=exposure)
        pred5 = mod1.predict(exog=exog, exposure=exposure)
        assert_almost_equal(pred3, pred4)
        assert_almost_equal(pred4, pred5)

        # Check offset defaults
        offset = np.random.uniform(1, 2, 100)
        mod2 = GLM(endog, exog, offset=offset,
                   family=sm.families.Poisson()).fit()
        pred1 = mod2.predict()
        pred2 = mod2.predict(which="mean", offset=offset)
        pred3 = mod2.predict(exog=exog, which="mean", offset=offset)
        assert_almost_equal(pred1, pred2)
        assert_almost_equal(pred2, pred3)

        # Check that offset shifts the linear predictor
        mod3 = GLM(endog, exog, family=sm.families.Poisson()).fit()
        offset = np.random.uniform(1, 2, 10)
        with pytest.warns(FutureWarning):
            # deprecation warning for linear keyword
            pred1 = mod3.predict(exog=exog1, offset=offset, linear=True)
        pred2 = mod3.predict(exog=exog1, offset=2*offset, which="linear")
        assert_almost_equal(pred2, pred1+offset)

        # Passing exposure as a pandas series should not effect output type
        assert isinstance(
            mod1.predict(exog=exog1, exposure=pd.Series(exposure1)),
            np.ndarray
        )


def test_perfect_pred(iris):
    y = iris[:, -1]
    X = iris[:, :-1]
    X = X[y != 2]
    y = y[y != 2]
    X = add_constant(X, prepend=True)
    glm = GLM(y, X, family=sm.families.Binomial())

    with pytest.warns(PerfectSeparationWarning):
        glm.fit()


def test_score_test_ols():
    # nicer example than Longley
    from statsmodels.regression.linear_model import OLS
    np.random.seed(5)
    nobs = 100
    sige = 0.5
    x = np.random.uniform(0, 1, size=(nobs, 5))
    x[:, 0] = 1
    beta = 1. / np.arange(1., x.shape[1] + 1)
    y = x.dot(beta) + sige * np.random.randn(nobs)

    res_ols = OLS(y, x).fit()
    res_olsc = OLS(y, x[:, :-2]).fit()
    co = res_ols.compare_lm_test(res_olsc, demean=False)

    res_glm = GLM(y, x[:, :-2], family=sm.families.Gaussian()).fit()
    co2 = res_glm.model.score_test(res_glm.params, exog_extra=x[:, -2:])
    # difference in df_resid versus nobs in scale see #1786
    assert_allclose(co[0] * 97 / 100., co2[0], rtol=1e-13)


def test_attribute_writable_resettable():
    # Regression test for mutables and class constructors.
    data = sm.datasets.longley.load()
    endog, exog = data.endog, data.exog
    glm_model = sm.GLM(endog, exog)
    assert_equal(glm_model.family.link.power, 1.0)
    glm_model.family.link.power = 2.
    assert_equal(glm_model.family.link.power, 2.0)
    glm_model2 = sm.GLM(endog, exog)
    assert_equal(glm_model2.family.link.power, 1.0)


class TestStartParams(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        cls.decimal_resids = DECIMAL_3
        cls.decimal_params = DECIMAL_2
        cls.decimal_bic = DECIMAL_0
        cls.decimal_bse = DECIMAL_3

        from statsmodels.datasets.longley import load
        cls.data = load()
        cls.data.exog = add_constant(cls.data.exog, prepend=False)
        params = sm.OLS(cls.data.endog, cls.data.exog).fit().params
        cls.res1 = GLM(cls.data.endog, cls.data.exog,
                        family=sm.families.Gaussian()).fit(start_params=params)
        from .results.results_glm import Longley
        cls.res2 = Longley()


def test_glm_start_params():
    # see 1604
    y2 = np.array('0 1 0 0 0 1'.split(), int)
    wt = np.array([50,1,50,1,5,10])
    y2 = np.repeat(y2, wt)
    x2 = np.repeat([0,0,0.001,100,-1,-1], wt)
    mod = sm.GLM(y2, sm.add_constant(x2), family=sm.families.Binomial())
    res = mod.fit(start_params=[-4, -5])
    np.testing.assert_almost_equal(res.params, [-4.60305022, -5.29634545], 6)


def test_loglike_no_opt():
    # see 1728

    y = np.asarray([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    x = np.arange(10, dtype=np.float64)

    def llf(params):
        lin_pred = params[0] + params[1]*x
        pr = 1 / (1 + np.exp(-lin_pred))
        return np.sum(y*np.log(pr) + (1-y)*np.log(1-pr))

    for params in [0,0], [0,1], [0.5,0.5]:
        mod = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
        res = mod.fit(start_params=params, maxiter=0)
        like = llf(params)
        assert_almost_equal(like, res.llf)


def test_formula_missing_exposure():
    # see 2083
    import statsmodels.formula.api as smf

    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan],
         'constant': [1] * 4, 'exposure': np.random.uniform(size=4),
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    family = sm.families.Gaussian(link=sm.families.links.Log())

    mod = smf.glm("Foo ~ Bar", data=df, exposure=df.exposure,
                  family=family)
    assert_(type(mod.exposure) is np.ndarray, msg='Exposure is not ndarray')

    exposure = pd.Series(np.random.uniform(size=5))
    df.loc[3, 'Bar'] = 4   # nan not relevant for Valueerror for shape mismatch
    assert_raises(ValueError, smf.glm, "Foo ~ Bar", data=df,
                  exposure=exposure, family=family)
    assert_raises(ValueError, GLM, df.Foo, df[['constant', 'Bar']],
                  exposure=exposure, family=family)


@pytest.mark.matplotlib
def test_plots(close_figures):

    np.random.seed(378)
    n = 200
    exog = np.random.normal(size=(n, 2))
    lin_pred = exog[:, 0] + exog[:, 1]**2
    prob = 1 / (1 + np.exp(-lin_pred))
    endog = 1 * (np.random.uniform(size=n) < prob)

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    import pandas as pd

    from statsmodels.graphics.regressionplots import add_lowess

    # array interface
    for j in 0,1:
        fig = result.plot_added_variable(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_partial_residuals(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_ceres_residuals(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)

    # formula interface
    data = pd.DataFrame({"y": endog, "x1": exog[:, 0], "x2": exog[:, 1]})
    model = sm.GLM.from_formula("y ~ x1 + x2", data, family=sm.families.Binomial())
    result = model.fit()
    for j in 0,1:
        xname = ["x1", "x2"][j]
        fig = result.plot_added_variable(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_partial_residuals(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_ceres_residuals(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)

def gen_endog(lin_pred, family_class, link, binom_version=0):

    np.random.seed(872)

    fam = sm.families

    mu = link().inverse(lin_pred)

    if family_class == fam.Binomial:
        if binom_version == 0:
            endog = 1*(np.random.uniform(size=len(lin_pred)) < mu)
        else:
            endog = np.empty((len(lin_pred), 2))
            n = 10
            endog[:, 0] = (np.random.uniform(size=(len(lin_pred), n)) < mu[:, None]).sum(1)
            endog[:, 1] = n - endog[:, 0]
    elif family_class == fam.Poisson:
        endog = np.random.poisson(mu)
    elif family_class == fam.Gamma:
        endog = np.random.gamma(2, mu)
    elif family_class == fam.Gaussian:
        endog = mu + 2 * np.random.normal(size=len(lin_pred))
    elif family_class == fam.NegativeBinomial:
        from scipy.stats.distributions import nbinom
        endog = nbinom.rvs(mu, 0.5)
    elif family_class == fam.InverseGaussian:
        from scipy.stats.distributions import invgauss
        endog = invgauss.rvs(mu, scale=20)
    else:
        raise ValueError

    return endog


@pytest.mark.smoke
def test_summary():
    np.random.seed(4323)

    n = 100
    exog = np.random.normal(size=(n, 2))
    exog[:, 0] = 1
    endog = np.random.normal(size=n)

    for method in ["irls", "cg"]:
        fa = sm.families.Gaussian()
        model = sm.GLM(endog, exog, family=fa)
        rslt = model.fit(method=method)
        s = rslt.summary()


def check_score_hessian(results):
    # compare models core and hessian with numerical derivatives

    params = results.params
    # avoid checking score at MLE, score close to zero
    sc = results.model.score(params * 0.98, scale=1)
    # cs currently (0.9) does not work for all families
    llfunc = lambda x: results.model.loglike(x, scale=1)  # noqa
    sc2 = approx_fprime(params * 0.98, llfunc)
    assert_allclose(sc, sc2, rtol=1e-4, atol=1e-4)

    hess = results.model.hessian(params, scale=1)
    hess2 = approx_hess(params, llfunc)
    assert_allclose(hess, hess2, rtol=1e-4)
    scfunc = lambda x: results.model.score(x, scale=1)  # noqa
    hess3 = approx_fprime(params, scfunc)
    assert_allclose(hess, hess3, rtol=1e-4)


def test_gradient_irls():
    # Compare the results when using gradient optimization and IRLS.

    # TODO: Find working examples for inverse_squared link

    np.random.seed(87342)

    fam = sm.families
    lnk = sm.families.links
    families = [(fam.Binomial, [lnk.Logit, lnk.Probit, lnk.CLogLog, lnk.Log, lnk.Cauchy]),
                (fam.Poisson, [lnk.Log, lnk.Identity, lnk.Sqrt]),
                (fam.Gamma, [lnk.Log, lnk.Identity, lnk.InversePower]),
                (fam.Gaussian, [lnk.Identity, lnk.Log, lnk.InversePower]),
                (fam.InverseGaussian, [lnk.Log, lnk.Identity, lnk.InversePower, lnk.InverseSquared]),
                (fam.NegativeBinomial, [lnk.Log, lnk.InversePower, lnk.InverseSquared, lnk.Identity])]

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1

    skip_one = False
    for family_class, family_links in families:
        for link in family_links:
            for binom_version in 0,1:

                if family_class != fam.Binomial and binom_version == 1:
                    continue

                if (family_class, link) == (fam.Poisson, lnk.Identity):
                    lin_pred = 20 + exog.sum(1)
                elif (family_class, link) == (fam.Binomial, lnk.Log):
                    lin_pred = -1 + exog.sum(1) / 8
                elif (family_class, link) == (fam.Poisson, lnk.Sqrt):
                    lin_pred = 2 + exog.sum(1)
                elif (family_class, link) == (fam.InverseGaussian, lnk.Log):
                    #skip_zero = True
                    lin_pred = -1 + exog.sum(1)
                elif (family_class, link) == (fam.InverseGaussian, lnk.Identity):
                    lin_pred = 20 + 5*exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (fam.InverseGaussian, lnk.InverseSquared):
                    lin_pred = 0.5 + exog.sum(1) / 5
                    continue # skip due to non-convergence
                elif (family_class, link) == (fam.InverseGaussian, lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5
                elif (family_class, link) == (fam.NegativeBinomial, lnk.Identity):
                    lin_pred = 20 + 5*exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (fam.NegativeBinomial, lnk.InverseSquared):
                    lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                    continue # skip due to non-convergence
                elif (family_class, link) == (fam.NegativeBinomial, lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5

                elif (family_class, link) == (fam.Gaussian, lnk.InversePower):
                    # adding skip because of convergence failure
                    skip_one = True
                # the following fails with Identity link, because endog < 0
                # elif family_class == fam.Gamma:
                #     lin_pred = 0.5 * exog.sum(1) + np.random.uniform(size=exog.shape[0])
                else:
                    lin_pred = np.random.uniform(size=exog.shape[0])

                endog = gen_endog(lin_pred, family_class, link, binom_version)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_irls = sm.GLM(endog, exog, family=family_class(link=link()))
                rslt_irls = mod_irls.fit(method="IRLS")

                if not (family_class, link) in [(fam.Poisson, lnk.Sqrt),
                                                (fam.Gamma, lnk.InversePower),
                                                (fam.InverseGaussian, lnk.Identity)
                                                ]:
                    check_score_hessian(rslt_irls)

                # Try with and without starting values.
                for max_start_irls, start_params in (0, rslt_irls.params), (3, None):
                    # TODO: skip convergence failures for now
                    if max_start_irls > 0 and skip_one:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod_gradient = sm.GLM(endog, exog, family=family_class(link=link()))
                    rslt_gradient = mod_gradient.fit(max_start_irls=max_start_irls,
                                                     start_params=start_params,
                                                     method="newton", maxiter=300)

                    assert_allclose(rslt_gradient.params,
                                    rslt_irls.params, rtol=1e-6, atol=5e-5)

                    assert_allclose(rslt_gradient.llf, rslt_irls.llf,
                                    rtol=1e-6, atol=1e-6)

                    assert_allclose(rslt_gradient.scale, rslt_irls.scale,
                                    rtol=1e-6, atol=1e-6)

                    # Get the standard errors using expected information.
                    gradient_bse = rslt_gradient.bse
                    ehess = mod_gradient.hessian(rslt_gradient.params, observed=False)
                    gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))
                    assert_allclose(gradient_bse, rslt_irls.bse, rtol=1e-6, atol=5e-5)
                    # rslt_irls.bse corresponds to observed=True
                    assert_allclose(rslt_gradient.bse, rslt_irls.bse, rtol=0.2, atol=5e-5)

                    rslt_gradient_eim = mod_gradient.fit(max_start_irls=0,
                                                         cov_type='eim',
                                                         start_params=rslt_gradient.params,
                                                         method="newton", maxiter=300)
                    assert_allclose(rslt_gradient_eim.bse, rslt_irls.bse, rtol=5e-5, atol=0)


def test_gradient_irls_eim():
    # Compare the results when using eime gradient optimization and IRLS.

    # TODO: Find working examples for inverse_squared link

    np.random.seed(87342)

    fam = sm.families
    lnk = sm.families.links
    families = [(fam.Binomial, [lnk.Logit, lnk.Probit, lnk.CLogLog, lnk.Log,
                                lnk.Cauchy]),
                (fam.Poisson, [lnk.Log, lnk.Identity, lnk.Sqrt]),
                (fam.Gamma, [lnk.Log, lnk.Identity, lnk.InversePower]),
                (fam.Gaussian, [lnk.Identity, lnk.Log, lnk.InversePower]),
                (fam.InverseGaussian, [lnk.Log, lnk.Identity,
                                       lnk.InversePower,
                                       lnk.InverseSquared]),
                (fam.NegativeBinomial, [lnk.Log, lnk.InversePower,
                                        lnk.InverseSquared, lnk.Identity])]

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1

    skip_one = False
    for family_class, family_links in families:
        for link in family_links:
            for binom_version in 0, 1:

                if family_class != fam.Binomial and binom_version == 1:
                    continue

                if (family_class, link) == (fam.Poisson, lnk.Identity):
                    lin_pred = 20 + exog.sum(1)
                elif (family_class, link) == (fam.Binomial, lnk.Log):
                    lin_pred = -1 + exog.sum(1) / 8
                elif (family_class, link) == (fam.Poisson, lnk.Sqrt):
                    lin_pred = 2 + exog.sum(1)
                elif (family_class, link) == (fam.InverseGaussian, lnk.Log):
                    # skip_zero = True
                    lin_pred = -1 + exog.sum(1)
                elif (family_class, link) == (fam.InverseGaussian,
                                              lnk.Identity):
                    lin_pred = 20 + 5*exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (fam.InverseGaussian,
                                              lnk.InverseSquared):
                    lin_pred = 0.5 + exog.sum(1) / 5
                    continue  # skip due to non-convergence
                elif (family_class, link) == (fam.InverseGaussian,
                                              lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5
                elif (family_class, link) == (fam.NegativeBinomial,
                                              lnk.Identity):
                    lin_pred = 20 + 5*exog.sum(1)
                    lin_pred = np.clip(lin_pred, 1e-4, np.inf)
                elif (family_class, link) == (fam.NegativeBinomial,
                                              lnk.InverseSquared):
                    lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                    continue  # skip due to non-convergence
                elif (family_class, link) == (fam.NegativeBinomial,
                                              lnk.InversePower):
                    lin_pred = 1 + exog.sum(1) / 5

                elif (family_class, link) == (fam.Gaussian, lnk.InversePower):
                    # adding skip because of convergence failure
                    skip_one = True
                else:
                    lin_pred = np.random.uniform(size=exog.shape[0])

                endog = gen_endog(lin_pred, family_class, link, binom_version)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_irls = sm.GLM(endog, exog,
                                      family=family_class(link=link()))
                rslt_irls = mod_irls.fit(method="IRLS")

                # Try with and without starting values.
                for max_start_irls, start_params in ((0, rslt_irls.params),
                                                     (3, None)):
                    # TODO: skip convergence failures for now
                    if max_start_irls > 0 and skip_one:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod_gradient = sm.GLM(endog, exog,
                                              family=family_class(link=link()))
                    rslt_gradient = mod_gradient.fit(
                            max_start_irls=max_start_irls,
                            start_params=start_params,
                            method="newton",
                            optim_hessian='eim'
                    )

                    assert_allclose(rslt_gradient.params, rslt_irls.params,
                                    rtol=1e-6, atol=5e-5)

                    assert_allclose(rslt_gradient.llf, rslt_irls.llf,
                                    rtol=1e-6, atol=1e-6)

                    assert_allclose(rslt_gradient.scale, rslt_irls.scale,
                                    rtol=1e-6, atol=1e-6)

                    # Get the standard errors using expected information.
                    ehess = mod_gradient.hessian(rslt_gradient.params,
                                                 observed=False)
                    gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))

                    assert_allclose(gradient_bse, rslt_irls.bse, rtol=1e-6,
                                    atol=5e-5)


def test_glm_irls_method():
    nobs, k_vars = 50, 4
    np.random.seed(987126)
    x = np.random.randn(nobs, k_vars - 1)
    exog = add_constant(x, has_constant='add')
    y = exog.sum(1) + np.random.randn(nobs)

    mod = GLM(y, exog)
    res1 = mod.fit()
    res2 = mod.fit(wls_method='pinv', attach_wls=True)
    res3 = mod.fit(wls_method='qr', attach_wls=True)
    # fit_gradient does not attach mle_settings
    res_g1 = mod.fit(start_params=res1.params, method='bfgs')

    for r in [res1, res2, res3]:
        assert_equal(r.mle_settings['optimizer'], 'IRLS')
        assert_equal(r.method, 'IRLS')

    assert_equal(res1.mle_settings['wls_method'], 'lstsq')
    assert_equal(res2.mle_settings['wls_method'], 'pinv')
    assert_equal(res3.mle_settings['wls_method'], 'qr')

    assert_(hasattr(res2.results_wls.model, 'pinv_wexog'))
    assert_(hasattr(res3.results_wls.model, 'exog_Q'))

    # fit_gradient currently does not attach mle_settings
    assert_equal(res_g1.method, 'bfgs')


class CheckWtdDuplicationMixin:
    decimal_params = DECIMAL_4

    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load()
        cls.data.endog = np.asarray(cls.data.endog)
        cls.data.exog = np.asarray(cls.data.exog)
        cls.endog = cls.data.endog
        cls.exog = cls.data.exog
        np.random.seed(1234)
        cls.weight = np.random.randint(5, 100, len(cls.endog))
        cls.endog_big = np.repeat(cls.endog, cls.weight)
        cls.exog_big = np.repeat(cls.exog, cls.weight, axis=0)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,  atol=1e-6,
                        rtol=1e-6)

    decimal_bse = DECIMAL_4

    def test_standard_errors(self):
        assert_allclose(self.res1.bse, self.res2.bse, rtol=1e-5, atol=1e-6)

    decimal_resids = DECIMAL_4

    # TODO: This does not work... Arrays are of different shape.
    # Perhaps we use self.res1.model.family.resid_XXX()?
    """
    def test_residuals(self):
        resids1 = np.column_stack((self.res1.resid_pearson,
                                   self.res1.resid_deviance,
                                   self.res1.resid_working,
                                   self.res1.resid_anscombe,
                                   self.res1.resid_response))
        resids2 = np.column_stack((self.res1.resid_pearson,
                                   self.res2.resid_deviance,
                                   self.res2.resid_working,
                                   self.res2.resid_anscombe,
                                   self.res2.resid_response))
        assert_allclose(resids1, resids2, self.decimal_resids)
    """

    def test_aic(self):
        # R includes the estimation of the scale as a lost dof
        # Does not with Gamma though
        assert_allclose(self.res1.aic, self.res2.aic,  atol=1e-6, rtol=1e-6)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance,  atol=1e-6,
                        rtol=1e-6)

    def test_scale(self):
        assert_allclose(self.res1.scale, self.res2.scale, atol=1e-6, rtol=1e-6)

    def test_loglike(self):
        # Stata uses the below llf for these families
        # We differ with R for them
        assert_allclose(self.res1.llf, self.res2.llf, 1e-6)

    decimal_null_deviance = DECIMAL_4

    def test_null_deviance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DomainWarning)

            assert_allclose(self.res1.null_deviance,
                            self.res2.null_deviance,
                            atol=1e-6,
                            rtol=1e-6)

    decimal_bic = DECIMAL_4

    def test_bic(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_allclose(self.res1.bic, self.res2.bic,  atol=1e-6, rtol=1e-6)

    decimal_fittedvalues = DECIMAL_4

    def test_fittedvalues(self):
        res2_fitted = self.res2.predict(self.res1.model.exog)
        assert_allclose(self.res1.fittedvalues, res2_fitted, atol=1e-5,
                        rtol=1e-5)

    decimal_tpvalues = DECIMAL_4

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        assert_allclose(self.res1.tvalues, self.res2.tvalues, atol=1e-6,
                        rtol=2e-4)
        assert_allclose(self.res1.pvalues, self.res2.pvalues, atol=1e-6,
                        rtol=1e-6)
        assert_allclose(self.res1.conf_int(), self.res2.conf_int(), atol=1e-6,
                        rtol=1e-6)


class TestWtdGlmPoisson(CheckWtdDuplicationMixin):

    @classmethod
    def setup_class(cls):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoisson, cls).setup_class()
        cls.endog = np.asarray(cls.endog)
        cls.exog = np.asarray(cls.exog)

        cls.res1 = GLM(cls.endog, cls.exog,
                        freq_weights=cls.weight,
                        family=sm.families.Poisson()).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                        family=sm.families.Poisson()).fit()


class TestWtdGlmPoissonNewton(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonNewton, cls).setup_class()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        fit_kwds = dict(method='newton')
        cls.res1 = GLM(cls.endog, cls.exog,
                        freq_weights=cls.weight,
                        family=sm.families.Poisson()).fit(**fit_kwds)
        fit_kwds = dict(method='newton', start_params=start_params)
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                        family=sm.families.Poisson()).fit(**fit_kwds)


class TestWtdGlmPoissonHC0(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):

        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonHC0, cls).setup_class()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        fit_kwds = dict(cov_type='HC0')
        cls.res1 = GLM(cls.endog, cls.exog,
                        freq_weights=cls.weight,
                        family=sm.families.Poisson()).fit(**fit_kwds)
        fit_kwds = dict(cov_type='HC0', start_params=start_params)
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                        family=sm.families.Poisson()).fit(**fit_kwds)


class TestWtdGlmPoissonClu(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):

        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonClu, cls).setup_class()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        gid = np.arange(1, len(cls.endog) + 1) // 2
        fit_kwds = dict(cov_type='cluster', cov_kwds={'groups': gid, 'use_correction':False})

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.res1 = GLM(cls.endog, cls.exog,
                            freq_weights=cls.weight,
                            family=sm.families.Poisson()).fit(**fit_kwds)
            gidr = np.repeat(gid, cls.weight)
            fit_kwds = dict(cov_type='cluster', cov_kwds={'groups': gidr, 'use_correction':False})
            cls.res2 = GLM(cls.endog_big, cls.exog_big,
                            family=sm.families.Poisson()).fit(start_params=start_params,
                                                              **fit_kwds)


class TestWtdGlmBinomial(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):

        '''
        Tests Binomial family with canonical logit link.
        '''
        super(TestWtdGlmBinomial, cls).setup_class()
        cls.endog = cls.endog / 100
        cls.endog_big = cls.endog_big / 100
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=sm.families.Binomial()).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=sm.families.Binomial()).fit()


class TestWtdGlmNegativeBinomial(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):

        '''
        Tests Negative Binomial family with canonical link
        g(p) = log(p/(p + 1/alpha))
        '''
        super(TestWtdGlmNegativeBinomial, cls).setup_class()
        alpha = 1.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DomainWarning)
            family_link = sm.families.NegativeBinomial(
                link=sm.families.links.NegativeBinomial(alpha=alpha),
                alpha=alpha)
            cls.res1 = GLM(cls.endog, cls.exog,
                           freq_weights=cls.weight,
                           family=family_link).fit()
            cls.res2 = GLM(cls.endog_big, cls.exog_big,
                           family=family_link).fit()


class TestWtdGlmGamma(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):

        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGamma, cls).setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link).fit()


class TestWtdGlmGaussian(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Gaussian family with log link.
        '''
        super(TestWtdGlmGaussian, cls).setup_class()
        family_link = sm.families.Gaussian(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link).fit()


class TestWtdGlmInverseGaussian(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests InverseGaussian family with log link.
        '''
        super(TestWtdGlmInverseGaussian, cls).setup_class()
        family_link = sm.families.InverseGaussian(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link).fit()


class TestWtdGlmGammaNewton(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaNewton, cls).setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link
                       ).fit(method='newton')
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link
                       ).fit(method='newton')

    def test_init_kwargs(self):
        family_link = sm.families.Gamma(sm.families.links.Log())

        with pytest.warns(ValueWarning, match="unknown kwargs"):
            GLM(self.endog, self.exog, family=family_link,
                weights=self.weight,  # incorrect keyword
                )


class TestWtdGlmGammaScale_X2(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaScale_X2, cls).setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link,
                       ).fit(scale='X2')
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link,
                       ).fit(scale='X2')


class TestWtdGlmGammaScale_dev(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaScale_dev, cls).setup_class()
        family_link = sm.families.Gamma(sm.families.links.Log())
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link,
                       ).fit(scale='dev')
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link,
                       ).fit(scale='dev')

    def test_missing(self):
        endog = self.data.endog.copy()
        exog = self.data.exog.copy()
        exog[0, 0] = np.nan
        endog[[2, 4, 6, 8]] = np.nan
        freq_weights = self.weight
        mod_misisng = GLM(endog, exog, family=self.res1.model.family,
                          freq_weights=freq_weights, missing='drop')
        assert_equal(mod_misisng.freq_weights.shape[0],
                     mod_misisng.endog.shape[0])
        assert_equal(mod_misisng.freq_weights.shape[0],
                     mod_misisng.exog.shape[0])
        keep_idx = np.array([1,  3,  5,  7,  9, 10, 11, 12, 13, 14, 15, 16])
        assert_equal(mod_misisng.freq_weights, self.weight[keep_idx])


class TestWtdTweedieLog(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Tweedie family with log link and var_power=1.
        '''
        super(TestWtdTweedieLog, cls).setup_class()
        family_link = sm.families.Tweedie(link=sm.families.links.Log(),
                                          var_power=1)
        cls.res1 = GLM(cls.endog, cls.exog,
                        freq_weights=cls.weight,
                        family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                        family=family_link).fit()


class TestWtdTweediePower2(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Tweedie family with Power(1) link and var_power=2.
        '''
        cls.data = cpunish.load_pandas()
        cls.endog = cls.data.endog
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        np.random.seed(1234)
        cls.weight = np.random.randint(5, 100, len(cls.endog))
        cls.endog_big = np.repeat(cls.endog.values, cls.weight)
        cls.exog_big = np.repeat(cls.exog.values, cls.weight, axis=0)
        link = sm.families.links.Power()
        family_link = sm.families.Tweedie(link=link, var_power=2)
        cls.res1 = GLM(cls.endog, cls.exog,
                       freq_weights=cls.weight,
                       family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                       family=family_link).fit()


class TestWtdTweediePower15(CheckWtdDuplicationMixin):
    @classmethod
    def setup_class(cls):
        '''
        Tests Tweedie family with Power(0.5) link and var_power=1.5.
        '''
        super(TestWtdTweediePower15, cls).setup_class()
        family_link = sm.families.Tweedie(link=sm.families.links.Power(0.5),
                                          var_power=1.5)
        cls.res1 = GLM(cls.endog, cls.exog,
                        freq_weights=cls.weight,
                        family=family_link).fit()
        cls.res2 = GLM(cls.endog_big, cls.exog_big,
                        family=family_link).fit()


def test_wtd_patsy_missing():
    import pandas as pd
    data = cpunish.load()
    data.endog = np.require(data.endog, requirements="W")
    data.exog = np.require(data.exog, requirements="W")
    data.exog[0, 0] = np.nan
    data.endog[[2, 4, 6, 8]] = np.nan
    data.pandas = pd.DataFrame(data.exog, columns=data.exog_name)
    data.pandas['EXECUTIONS'] = data.endog
    weights = np.arange(1, len(data.endog)+1)
    formula = """EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + VC100k96 +
                 SOUTH + DEGREE"""
    mod_misisng = GLM.from_formula(formula, data=data.pandas,
                                   freq_weights=weights)
    assert_equal(mod_misisng.freq_weights.shape[0],
                 mod_misisng.endog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0],
                 mod_misisng.exog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0], 12)
    keep_weights = np.array([2,  4,  6,  8, 10, 11, 12, 13, 14, 15, 16, 17])
    assert_equal(mod_misisng.freq_weights, keep_weights)


class CheckTweedie:
    def test_resid(self):
        idx1 = len(self.res1.resid_response) - 1
        idx2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.resid_response[:17],
                                        [self.res1.resid_response[idx1]])),
                        np.concatenate((self.res2.resid_response[:17],
                                        [self.res2.resid_response[idx2]])),
                        rtol=1e-5, atol=1e-5)
        assert_allclose(np.concatenate((self.res1.resid_pearson[:17],
                                        [self.res1.resid_pearson[idx1]])),
                        np.concatenate((self.res2.resid_pearson[:17],
                                        [self.res2.resid_pearson[idx2]])),
                        rtol=1e-5, atol=1e-5)
        assert_allclose(np.concatenate((self.res1.resid_deviance[:17],
                                        [self.res1.resid_deviance[idx1]])),
                        np.concatenate((self.res2.resid_deviance[:17],
                                        [self.res2.resid_deviance[idx2]])),
                        rtol=1e-5, atol=1e-5)

        assert_allclose(np.concatenate((self.res1.resid_working[:17],
                                        [self.res1.resid_working[idx1]])),
                        np.concatenate((self.res2.resid_working[:17],
                                        [self.res2.resid_working[idx2]])),
                        rtol=1e-5, atol=1e-5)


    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-6, rtol=1e6)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5,
                        rtol=1e-5)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance, atol=1e-6,
                        rtol=1e-6)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_fittedvalues(self):
        idx1 = len(self.res1.fittedvalues) - 1
        idx2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.fittedvalues[:17],
                                        [self.res1.fittedvalues[idx1]])),
                        np.concatenate((self.res2.fittedvalues[:17],
                                        [self.res2.fittedvalues[idx2]])),
                        atol=1e-4, rtol=1e-4)

    def test_summary(self):
        self.res1.summary()
        self.res1.summary2()


class TestTweediePower15(CheckTweedie):
    @classmethod
    def setup_class(cls):
        from .results.results_glm import CpunishTweediePower15
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Power(1),
                                          var_power=1.5)
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family_link).fit()
        cls.res2 = CpunishTweediePower15()


class TestTweediePower2(CheckTweedie):
    @classmethod
    def setup_class(cls):
        from .results.results_glm import CpunishTweediePower2
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Power(1),
                                          var_power=2.)
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family_link).fit()
        cls.res2 = CpunishTweediePower2()


class TestTweedieLog1(CheckTweedie):
    @classmethod
    def setup_class(cls):
        from .results.results_glm import CpunishTweedieLog1
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Log(),
                                          var_power=1.)
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family_link).fit()
        cls.res2 = CpunishTweedieLog1()


class TestTweedieLog15Fair(CheckTweedie):
    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.fair import load_pandas

        from .results.results_glm import FairTweedieLog15
        data = load_pandas()
        family_link = sm.families.Tweedie(link=sm.families.links.Log(),
                                          var_power=1.5)
        cls.res1 = sm.GLM(endog=data.endog,
                          exog=data.exog[['rate_marriage', 'age',
                                          'yrs_married']],
                          family=family_link).fit()
        cls.res2 = FairTweedieLog15()


class CheckTweedieSpecial:
    def test_mu(self):
        assert_allclose(self.res1.mu, self.res2.mu, rtol=1e-5, atol=1e-5)

    def test_resid(self):
        assert_allclose(self.res1.resid_response, self.res2.resid_response,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_pearson, self.res2.resid_pearson,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_deviance, self.res2.resid_deviance,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_working, self.res2.resid_working,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_anscombe_unscaled,
                        self.res2.resid_anscombe_unscaled,
                        rtol=1e-5, atol=1e-5)


class TestTweedieSpecialLog0(CheckTweedieSpecial):
    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family1 = sm.families.Gaussian(link=sm.families.links.Log())
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.Log(),
                                      var_power=0)
        cls.res2 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family2).fit()


class TestTweedieSpecialLog1(CheckTweedieSpecial):
    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family1 = sm.families.Poisson(link=sm.families.links.Log())
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.Log(),
                                      var_power=1)
        cls.res2 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family2).fit()


class TestTweedieSpecialLog2(CheckTweedieSpecial):
    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family1 = sm.families.Gamma(link=sm.families.links.Log())
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.Log(),
                                      var_power=2)
        cls.res2 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family2).fit()


class TestTweedieSpecialLog3(CheckTweedieSpecial):
    @classmethod
    def setup_class(cls):
        cls.data = cpunish.load_pandas()
        cls.exog = cls.data.exog[['INCOME', 'SOUTH']]
        cls.endog = cls.data.endog
        family1 = sm.families.InverseGaussian(link=sm.families.links.Log())
        cls.res1 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.Log(),
                                      var_power=3)
        cls.res2 = sm.GLM(endog=cls.data.endog,
                          exog=cls.data.exog[['INCOME', 'SOUTH']],
                          family=family2).fit()

def gen_tweedie(p):

    np.random.seed(3242)
    n = 500
    x = np.random.normal(size=(n, 4))
    lpr = np.dot(x, np.r_[1, -1, 0, 0.5])
    mu = np.exp(lpr)
    lam = 10 * mu**(2 - p) / (2 - p)
    alp = (2 - p) / (p - 1)
    bet = 10 * mu**(1 - p) / (p - 1)

    # Generate Tweedie values using commpound Poisson distribution
    y = np.empty(n)
    N = np.random.poisson(lam)
    for i in range(n):
        y[i] = np.random.gamma(alp, 1 / bet[i], N[i]).sum()

    return y, x

@pytest.mark.filterwarnings("ignore:GLM ridge optimization")
def test_tweedie_EQL():
    # All tests below are regression tests, but the results
    # are very close to the population values.

    p = 1.5
    y, x = gen_tweedie(p)

    # Un-regularized fit using gradients
    fam = sm.families.Tweedie(var_power=p, eql=True)
    model1 = sm.GLM(y, x, family=fam)
    result1 = model1.fit(method="newton")
    assert_allclose(result1.params,
       np.array([1.00350497, -0.99656954, 0.00802702, 0.50713209]),
       rtol=1e-5, atol=1e-5)

    # Un-regularized fit using IRLS
    model1x = sm.GLM(y, x, family=fam)
    result1x = model1x.fit(method="irls")
    assert_allclose(result1.params, result1x.params)
    assert_allclose(result1.bse, result1x.bse, rtol=1e-2)

    # Lasso fit using coordinate-wise descent
    # TODO: The search gets trapped in an infinite oscillation, so use
    # a slack convergence tolerance.
    model2 = sm.GLM(y, x, family=fam)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.07, maxiter=200,
                   cnvrg_tol=0.01)

    rtol, atol = 1e-2, 1e-4
    assert_allclose(result2.params,
        np.array([0.976831, -0.952854, 0., 0.470171]),
        rtol=rtol, atol=atol)

    # Series of ridge fits using gradients
    ev = (np.array([1.001778, -0.99388, 0.00797, 0.506183]),
          np.array([0.98586638, -0.96953481, 0.00749983, 0.4975267]),
          np.array([0.206429, -0.164547, 0.000235, 0.102489]))
    for j, alpha in enumerate([0.05, 0.5, 0.7]):
        model3 = sm.GLM(y, x, family=fam)
        result3 = model3.fit_regularized(L1_wt=0, alpha=alpha)
        assert_allclose(result3.params, ev[j], rtol=rtol, atol=atol)
        result4 = model3.fit_regularized(L1_wt=0, alpha=alpha * np.ones(x.shape[1]))
        assert_allclose(result4.params, result3.params, rtol=rtol, atol=atol)
        alpha = alpha * np.ones(x.shape[1])
        alpha[0] = 0
        result5 = model3.fit_regularized(L1_wt=0, alpha=alpha)
        assert not np.allclose(result5.params, result4.params)

def test_tweedie_elastic_net():
    # Check that the coefficients vanish one-by-one
    # when using the elastic net.

    p = 1.5 # Tweedie variance exponent
    y, x = gen_tweedie(p)

    # Un-regularized fit using gradients
    fam = sm.families.Tweedie(var_power=p, eql=True)
    model1 = sm.GLM(y, x, family=fam)

    nnz = []
    for alpha in np.linspace(0, 10, 20):
        result1 = model1.fit_regularized(L1_wt=0.5, alpha=alpha)
        nnz.append((np.abs(result1.params) > 0).sum())
    nnz = np.unique(nnz)
    assert len(nnz) == 5


def test_tweedie_EQL_poisson_limit():
    # Test the limiting Poisson case of the Nelder/Pregibon/Tweedie
    # EQL.

    np.random.seed(3242)
    n = 500

    x = np.random.normal(size=(n, 3))
    x[:, 0] = 1
    lpr = 4 + x[:, 1:].sum(1)
    mn = np.exp(lpr)
    y = np.random.poisson(mn)

    for scale in 1.0, 'x2', 'dev':

        # Un-regularized fit using gradients not IRLS
        fam = sm.families.Tweedie(var_power=1, eql=True)
        model1 = sm.GLM(y, x, family=fam)
        result1 = model1.fit(method="newton", scale=scale)

        # Poisson GLM
        model2 = sm.GLM(y, x, family=sm.families.Poisson())
        result2 = model2.fit(method="newton", scale=scale)

        assert_allclose(result1.params, result2.params, atol=1e-6, rtol=1e-6)
        assert_allclose(result1.bse, result2.bse, 1e-6, 1e-6)


def test_tweedie_EQL_upper_limit():
    # Test the limiting case of the Nelder/Pregibon/Tweedie
    # EQL with var = mean^2.  These are tests against population
    # values so accuracy is not high.

    np.random.seed(3242)
    n = 500

    x = np.random.normal(size=(n, 3))
    x[:, 0] = 1
    lpr = 4 + x[:, 1:].sum(1)
    mn = np.exp(lpr)
    y = np.random.poisson(mn)

    for scale in 'x2', 'dev', 1.0:

        # Un-regularized fit using gradients not IRLS
        fam = sm.families.Tweedie(var_power=2, eql=True)
        model1 = sm.GLM(y, x, family=fam)
        result1 = model1.fit(method="newton", scale=scale)
        assert_allclose(result1.params, np.r_[4, 1, 1], atol=1e-3, rtol=1e-1)


def testTweediePowerEstimate():
    # Test the Pearson estimate of the Tweedie variance and scale parameters.
    #
    # Ideally, this would match the following R code, but I cannot make it work...
    #
    # setwd('c:/workspace')
    # data <- read.csv('cpunish.csv', sep=",")
    #
    # library(tweedie)
    #
    # y <- c(1.00113835e+05,   6.89668315e+03,   6.15726842e+03,
    #        1.41718806e+03,   5.11776456e+02,   2.55369154e+02,
    #        1.07147443e+01,   3.56874698e+00,   4.06797842e-02,
    #        7.06996731e-05,   2.10165106e-07,   4.34276938e-08,
    #        1.56354040e-09,   0.00000000e+00,   0.00000000e+00,
    #        0.00000000e+00,   0.00000000e+00)
    #
    # data$NewY <- y
    #
    # out <- tweedie.profile( NewY ~ INCOME + SOUTH - 1,
    #                         p.vec=c(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    #                                 1.9), link.power=0,
    #                         data=data,do.plot = TRUE)
    data = cpunish.load_pandas()
    y = [1.00113835e+05,   6.89668315e+03,   6.15726842e+03,
         1.41718806e+03,   5.11776456e+02,   2.55369154e+02,
         1.07147443e+01,   3.56874698e+00,   4.06797842e-02,
         7.06996731e-05,   2.10165106e-07,   4.34276938e-08,
         1.56354040e-09,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00]
    model1 = sm.GLM(y, data.exog[['INCOME', 'SOUTH']],
                    family=sm.families.Tweedie(link=sm.families.links.Log(),
                                               var_power=1.5))
    res1 = model1.fit()
    model2 = sm.GLM((y - res1.mu) ** 2,
                    np.column_stack((np.ones(len(res1.mu)), np.log(res1.mu))),
                    family=sm.families.Gamma(sm.families.links.Log()))
    res2 = model2.fit()
    # Sample may be too small for this...
    # assert_allclose(res1.scale, np.exp(res2.params[0]), rtol=0.25)
    p = model1.estimate_tweedie_power(res1.mu)
    assert_allclose(p, res2.params[1], rtol=0.25)

def test_glm_lasso_6431():

    # Based on issue #6431
    # Fails with newton-cg as optimizer
    np.random.seed(123)

    from statsmodels.regression.linear_model import OLS

    n = 50
    x = np.ones((n, 2))
    x[:, 1] = np.arange(0, n)
    y = 1000 + x[:, 1] + np.random.normal(0, 1, n)

    params = np.r_[999.82244338, 1.0077889]

    for method in "bfgs", None:
        for fun in [OLS, GLM]:

            # Changing L1_wtValue from 0 to 1e-9 changes
            # the algorithm from scipy gradient optimization
            # to statsmodels coordinate descent
            for L1_wtValue in [0, 1e-9]:
                model = fun(y, x)
                if fun == OLS:
                    fit = model.fit_regularized(alpha=0, L1_wt=L1_wtValue)
                else:
                    fit = model._fit_ridge(alpha=0, start_params=None, method=method)
                assert_allclose(params, fit.params, atol=1e-6, rtol=1e-6)

class TestRegularized:

    def test_regularized(self):

        import os

        from .results import glmnet_r_results

        for dtype in "binomial", "poisson":

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            data = np.loadtxt(os.path.join(cur_dir, "results", "enet_%s.csv" % dtype),
                              delimiter=",")

            endog = data[:, 0]
            exog = data[:, 1:]

            fam = {"binomial" : sm.families.Binomial,
                   "poisson" : sm.families.Poisson}[dtype]

            for j in range(9):

                vn = "rslt_%s_%d" % (dtype, j)
                r_result = getattr(glmnet_r_results, vn)
                L1_wt = r_result[0]
                alpha = r_result[1]
                params = r_result[2:]

                model = GLM(endog, exog, family=fam())
                sm_result = model.fit_regularized(L1_wt=L1_wt, alpha=alpha)

                # Agreement is OK, see below for further check
                assert_allclose(params, sm_result.params, atol=1e-2, rtol=0.3)

                # The penalized log-likelihood that we are maximizing.
                def plf(params):
                    llf = model.loglike(params) / len(endog)
                    llf = llf - alpha * ((1 - L1_wt)*np.sum(params**2) / 2 + L1_wt*np.sum(np.abs(params)))
                    return llf

                # Confirm that we are doing better than glmnet.
                llf_r = plf(params)
                llf_sm = plf(sm_result.params)
                assert_equal(np.sign(llf_sm - llf_r), 1)


class TestConvergence:
    @classmethod
    def setup_class(cls):
        '''
        Test Binomial family with canonical logit link using star98 dataset.
        '''
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.model = GLM(data.endog, data.exog,
                         family=sm.families.Binomial())

    def _when_converged(self, atol=1e-8, rtol=0, tol_criterion='deviance'):
        for i, dev in enumerate(self.res.fit_history[tol_criterion]):
            orig = self.res.fit_history[tol_criterion][i]
            new = self.res.fit_history[tol_criterion][i + 1]
            if np.allclose(orig, new, atol=atol, rtol=rtol):
                return i
        raise ValueError('CONVERGENCE CHECK: It seems this doens\'t converge!')

    def test_convergence_atol_only(self):
        atol = 1e-8
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_rtol_only(self):
        atol = 0
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_rtol(self):
        atol = 1e-8
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_only_params(self):
        atol = 1e-8
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_rtol_only_params(self):
        atol = 0
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_rtol_params(self):
        atol = 1e-8
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)


def test_poisson_deviance():
    # see #3355 missing term in deviance if resid_response.sum() != 0
    np.random.seed(123987)
    nobs, k_vars = 50, 3-1
    x = sm.add_constant(np.random.randn(nobs, k_vars))

    mu_true = np.exp(x.sum(1))
    y = np.random.poisson(mu_true, size=nobs)

    mod = sm.GLM(y, x[:, :], family=sm.genmod.families.Poisson())
    res = mod.fit()

    d_i = res.resid_deviance
    d = res.deviance
    lr = (mod.family.loglike(y, y+1e-20) -
          mod.family.loglike(y, res.fittedvalues)) * 2

    assert_allclose(d, (d_i**2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)

    # case without constant, resid_response.sum() != 0
    mod_nc = sm.GLM(y, x[:, 1:], family=sm.genmod.families.Poisson())
    res_nc = mod_nc.fit()

    d_i = res_nc.resid_deviance
    d = res_nc.deviance
    lr = (mod.family.loglike(y, y+1e-20) -
          mod.family.loglike(y, res_nc.fittedvalues)) * 2

    assert_allclose(d, (d_i**2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)


def test_non_invertible_hessian_fails_summary():
    # Test when the hessian fails the summary is still available.
    data = cpunish.load_pandas()

    data.endog[:] = 1
    with warnings.catch_warnings():
        # we filter DomainWarning, the convergence problems
        # and warnings in summary
        warnings.simplefilter("ignore")
        mod = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
        res = mod.fit(maxiter=1, method='bfgs', max_start_irls=0)
        res.summary()


def test_int_scale():
    # GH-6627, make sure it works with int scale
    data = longley.load()
    mod = GLM(data.endog, data.exog, family=sm.families.Gaussian())
    res = mod.fit(scale=1)
    assert isinstance(res.params, pd.Series)
    assert res.scale.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_int_exog(dtype):
    # GH-6627, make use of floats internally
    count1, n1, count2, n2 = 60, 51477.5, 30, 54308.7
    y = [count1, count2]
    x = np.asarray([[1, 1], [1, 0]]).astype(dtype)
    exposure = np.asarray([n1, n2])
    mod = GLM(y, x, exposure=exposure, family=sm.families.Poisson())
    res = mod.fit(method='bfgs', max_start_irls=0)
    assert isinstance(res.params, np.ndarray)


def test_glm_bic(iris):
    X = np.c_[np.ones(100), iris[50:, :4]]
    y = np.array(iris)[50:, 4].astype(np.int32)
    y -= 1
    SET_USE_BIC_LLF(True)
    model = GLM(y, X, family=sm.families.Binomial()).fit()
    # 34.9244 is what glm() of R yields
    assert_almost_equal(model.bic, 34.9244, decimal=3)
    assert_almost_equal(model.bic_llf, 34.9244, decimal=3)
    SET_USE_BIC_LLF(False)
    assert_almost_equal(model.bic, model.bic_deviance, decimal=3)
    SET_USE_BIC_LLF(None)


def test_glm_bic_warning(iris):
    X = np.c_[np.ones(100), iris[50:, :4]]
    y = np.array(iris)[50:, 4].astype(np.int32)
    y -= 1
    model = GLM(y, X, family=sm.families.Binomial()).fit()
    with pytest.warns(FutureWarning, match="The bic"):
        assert isinstance(model.bic, float)


def test_output_exposure_null(reset_randomstate):
    # GH 6953

    x0 = [np.sin(i / 20) + 2 for i in range(1000)]
    rs = np.random.RandomState(0)
    # Variable exposures for each observation
    exposure = rs.randint(100, 200, size=1000)
    y = [np.sum(rs.poisson(x, size=e)) for x, e in zip(x0, exposure)]
    x = add_constant(x0)

    model = GLM(
        endog=y, exog=x, exposure=exposure, family=sm.families.Poisson()
    ).fit()
    null_model = GLM(
        endog=y, exog=x[:, 0], exposure=exposure, family=sm.families.Poisson()
    ).fit()
    null_model_without_exposure = GLM(
        endog=y, exog=x[:, 0], family=sm.families.Poisson()
    ).fit()
    assert_allclose(model.llnull, null_model.llf)
    # Check that they are different
    assert np.abs(null_model_without_exposure.llf - model.llnull) > 1


def test_qaic():

    # Example from documentation of R package MuMIn
    import patsy
    ldose = np.concatenate((np.arange(6), np.arange(6)))
    sex = ["M"]*6 + ["F"]*6
    numdead = [10, 4, 9, 12, 18, 20, 0, 2, 6, 10, 12, 16]
    df = pd.DataFrame({"ldose": ldose, "sex": sex, "numdead": numdead})
    df["numalive"] = 20 - df["numdead"]
    df["SF"] = df["numdead"]

    y = df[["numalive", "numdead"]].values
    x = patsy.dmatrix("sex*ldose", data=df, return_type='dataframe')
    m = GLM(y, x, family=sm.families.Binomial())
    r = m.fit()
    scale = 2.412699
    qaic = r.info_criteria(crit="qaic", scale=scale)

    # R gives 31.13266 because it uses a df that is 1 greater,
    # presumably because they count the scale parameter in df.
    # This won't matter when comparing models by differencing
    # QAICs.
    # Binomial doesn't have a scale parameter, so adding +1 is not correct.
    assert_allclose(qaic, 29.13266, rtol=1e-5, atol=1e-5)
    qaic1 = r.info_criteria(crit="qaic", scale=scale, dk_params=1)
    assert_allclose(qaic1, 31.13266, rtol=1e-5, atol=1e-5)


def test_tweedie_score():

    np.random.seed(3242)
    n = 500
    x = np.random.normal(size=(n, 4))
    lpr = np.dot(x, np.r_[1, -1, 0, 0.5])
    mu = np.exp(lpr)

    p0 = 1.5
    lam = 10 * mu**(2 - p0) / (2 - p0)
    alp = (2 - p0) / (p0 - 1)
    bet = 10 * mu**(1 - p0) / (p0 - 1)
    y = np.empty(n)
    N = np.random.poisson(lam)
    for i in range(n):
        y[i] = np.random.gamma(alp, 1 / bet[i], N[i]).sum()

    for eql in [True, False]:
        for p in [1, 1.5, 2]:
            if eql is False and SP_LT_17:
                pytest.skip('skip, scipy too old, no bessel_wright')

            fam = sm.families.Tweedie(var_power=p, eql=eql)
            model = GLM(y, x, family=fam)
            result = model.fit()

            pa = result.params + 0.2*np.random.normal(size=result.params.size)

            ngrad = approx_fprime_cs(pa, lambda x: model.loglike(x, scale=1))
            agrad = model.score(pa, scale=1)
            assert_allclose(ngrad, agrad, atol=1e-8, rtol=1e-8)

            nhess = approx_hess_cs(pa, lambda x: model.loglike(x, scale=1))
            ahess = model.hessian(pa, scale=1)
            assert_allclose(nhess, ahess, atol=5e-8, rtol=5e-8)
