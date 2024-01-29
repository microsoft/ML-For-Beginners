# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:48:37 2021

Author: Josef Perktod
License: BSD-3
"""

import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import pytest

from statsmodels.tools.tools import add_constant

from statsmodels.base._prediction_inference import PredictionResultsMonotonic

from statsmodels.discrete.discrete_model import (
    BinaryModel,
    Logit,
    Probit,
    Poisson,
    NegativeBinomial,
    NegativeBinomialP,
    GeneralizedPoisson,
    )
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedGeneralizedPoisson,
    )

from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp


# copied from `test_gmm_poisson.TestGMMAddOnestep`
XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()
endog_name = 'docvis'
exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
endog = DATA[endog_name]
exog = DATA[exog_names]


class CheckPredict():

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        # Note we have alpha, stata has lnalpha
        sl1 = slice(self.k_infl, -1, None)
        sl2 = slice(0, -(self.k_infl + 1), None)
        assert_allclose(res1.params[sl1], res2.params[sl2], rtol=self.rtol)
        assert_allclose(res1.bse[sl1], res2.bse[sl2], rtol=30 * self.rtol)
        params1 = np.asarray(res1.params)
        params2 = np.asarray(res2.params)
        assert_allclose(params1[-1], np.exp(params2[-1]), rtol=self.rtol)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        ex = np.asarray(exog).mean(0)

        # test for which="mean"
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"].iloc[0], rtol=1e-4)
        assert_allclose(pred.se, rdf["se"].iloc[0], rtol=1e-4,  atol=1e-4)
        if isinstance(pred, PredictionResultsMonotonic):
            # default method is endpoint transformation for non-ZI models
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=1e-3,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=1e-3,  atol=1e-4)

            ci = pred.conf_int(method="delta")[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=1e-4,  atol=1e-4)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=1e-4,  atol=1e-4)

        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=1e-4,  atol=1e-4)

        rdf = res2.results_margins_mean
        pred = res1.get_prediction(average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"].iloc[0], rtol=3e-4)  # self.rtol)
        assert_allclose(pred.se, rdf["se"].iloc[0], rtol=3e-3,  atol=1e-4)
        if isinstance(pred, PredictionResultsMonotonic):
            # default method is endpoint transformation for non-ZI models
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=1e-3,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=1e-3,  atol=1e-4)

            ci = pred.conf_int(method="delta")[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=1e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=1e-4,  atol=1e-4)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf["ll"].iloc[0], rtol=5e-4,  atol=1e-4)
            assert_allclose(ci[1], rdf["ul"].iloc[0], rtol=5e-4,  atol=1e-4)

        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=1e-4,  atol=1e-4)

        # test for which="prob"
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, which="prob", y_values=np.arange(2),
                                   **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"].iloc[1:3], rtol=3e-4)  # self.rtol)
        assert_allclose(pred.se, rdf["se"].iloc[1:3], rtol=3e-3,  atol=1e-4)

        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf["ll"].iloc[1:3], rtol=5e-4,  atol=1e-4)
        assert_allclose(ci[:, 1], rdf["ul"].iloc[1:3], rtol=5e-4,  atol=1e-4)

        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=1e-4,  atol=1e-4)

        rdf = res2.results_margins_mean
        pred = res1.get_prediction(which="prob", y_values=np.arange(2),
                                   average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf["b"].iloc[1:3], rtol=5e-3)  # self.rtol)
        assert_allclose(pred.se, rdf["se"].iloc[1:3], rtol=3e-3,  atol=5e-4)

        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf["ll"].iloc[1:3], rtol=5e-4,  atol=1e-3)
        assert_allclose(ci[:, 1], rdf["ul"].iloc[1:3], rtol=5e-4,  atol=5e-3)

        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=1e-4,  atol=1e-4)
        stat, _ = pred.t_test(value=pred.predicted)
        assert_equal(stat, 0)

        # test agg_weights
        df6 = exog[:6]
        aw = np.zeros(len(res1.model.endog))
        aw[:6] = 1
        aw /= aw.mean()
        pm6 = res1.get_prediction(exog=df6, which="mean", average=True,
                                  **self.pred_kwds_6)
        dfm6 = pm6.summary_frame()
        pmw = res1.get_prediction(which="mean", average=True, agg_weights=aw)
        dfmw = pmw.summary_frame()
        assert_allclose(pmw.predicted, pm6.predicted, rtol=1e-13)
        assert_allclose(dfmw, dfm6, rtol=1e-7)

    def test_diagnostic(self):
        # smoke test for now
        res1 = self.res1

        dia = res1.get_diagnostic(y_max=21)
        res_chi2 = dia.test_chisquare_prob(bin_edges=np.arange(4))
        assert_equal(res_chi2.diff1.shape[1], 3)
        assert_equal(dia.probs_predicted.shape[1], 22)

        try:
            dia.plot_probs(upp_xlim=20)
        except ImportError:
            pass


class CheckExtras():

    def test_predict_linear(self):
        res1 = self.res1
        ex = np.asarray(exog[:5])
        pred = res1.get_prediction(ex, which="linear", **self.pred_kwds_mean)
        k_extra = len(res1.params) - ex.shape[1]
        if k_extra > 0:
            # not zero-inflated models have params_infl first
            ex = np.column_stack((ex, np.zeros((ex.shape[0], k_extra))))
        tt = res1.t_test(ex)
        cip = pred.conf_int()  # assumes no offset
        cit = tt.conf_int()
        assert_allclose(cip, cit, rtol=1e-12)

    def test_score_test(self):
        res1 = self.res1
        modr = self.klass(endog, exog.values[:, :-1])
        resr = modr.fit(method="newton", maxiter=300)
        params_restr = np.concatenate([resr.params[:-1], [0],
                                       resr.params[-1:]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -2] = 1
        exog_extra = res1.model.exog[:, -1:]

        from statsmodels.base._parameter_inference import score_test
        sc1 = score_test(res1, params_constrained=params_restr,
                         k_constraints=1)
        sc2 = score_test(resr, exog_extra=(exog_extra, None))
        assert_allclose(sc2[:2], sc1[:2])

        sc1_hc = score_test(res1, params_constrained=params_restr,
                            k_constraints=1, r_matrix=r_matrix, cov_type="HC0")
        sc2_hc = score_test(resr, exog_extra=(exog_extra, None),
                            cov_type="HC0")
        assert_allclose(sc2_hc[:2], sc1_hc[:2])

    def test_score_test_alpha(self):
        # this is mostly sanity check
        # we need heteroscedastic model, i.e. exog_dispersion as comparison
        # res1 = self.res1
        modr = self.klass(endog, exog.values[:, :-1])
        resr = modr.fit(method="newton", maxiter=300)
        params_restr = np.concatenate([resr.params[:], [0]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -1] = 1
        # use random expg_extra, then we don't reject null
        # exog_extra = res1.model.exog[:, 1:2]
        np.random.seed(987125643)
        exog_extra = 0.01 * np.random.randn(endog.shape[0])

        from statsmodels.base._parameter_inference import (
            score_test, _scorehess_extra)

        # note: we need params for the restricted model here
        # if params is not given, then it will be taked from results instance
        sh = _scorehess_extra(resr, exog_extra=None,
                              exog2_extra=exog_extra, hess_kwds=None)
        assert not np.isnan(sh[0]).any()

        # sc1 = score_test(res1, params_constrained=params_restr,
        #                  k_constraints=1)
        sc2 = score_test(resr, exog_extra=(None, exog_extra))
        assert sc2[1] > 0.01

        # sc1_hc = score_test(res1, params_constrained=params_restr,
        #                   k_constraints=1, r_matrix=r_matrix, cov_type="HC0")
        sc2_hc = score_test(resr, exog_extra=(None, exog_extra),
                            cov_type="HC0")
        assert sc2_hc[1] > 0.01

    def test_influence(self):
        # currently only smoke test
        res1 = self.res1
        from statsmodels.stats.outliers_influence import MLEInfluence

        influ = MLEInfluence(res1)
        attrs = ['cooks_distance', 'd_fittedvalues', 'd_fittedvalues_scaled',
                 'd_params', 'dfbetas', 'hat_matrix_diag', 'resid_studentized'
                 ]
        for attr in attrs:
            getattr(influ, attr)

        influ.summary_frame()


class TestNegativeBinomialPPredict(CheckPredict, CheckExtras):

    @classmethod
    def setup_class(cls):
        cls.klass = NegativeBinomialP
        # using newton has results much closer to Stata than bfgs
        res1 = NegativeBinomialP(endog, exog).fit(method="newton", maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_nb_docvis
        cls.pred_kwds_mean = {}
        cls.pred_kwds_6 = {}
        cls.k_infl = 0
        cls.rtol = 1e-8


class TestZINegativeBinomialPPredict(CheckPredict):

    @classmethod
    def setup_class(cls):
        exog_infl = add_constant(DATA["aget"], prepend=False)
        mod_zinb = ZeroInflatedNegativeBinomialP(endog, exog,
                                                 exog_infl=exog_infl, p=2)

        sp = np.array([
            -6.58, -1.28, 0.19, 0.08, 0.22, -0.05, 0.03, 0.17, 0.27, 0.68,
            0.62])
        # using newton. bfgs has non-invertivle hessian at convergence
        # start_params not needed, but speed up
        res1 = mod_zinb.fit(start_params=sp, method="newton", maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_zinb_docvis
        cls.pred_kwds_mean = {"exog_infl": exog_infl.mean(0)}
        cls.pred_kwds_6 = {"exog_infl": exog_infl[:6]}
        cls.k_infl = 2
        cls.rtol = 1e-4


class TestGeneralizedPoissonPredict(CheckExtras):

    @classmethod
    def setup_class(cls):
        cls.klass = GeneralizedPoisson
        mod1 = GeneralizedPoisson(endog, exog)
        res1 = mod1.fit(method="newton")
        cls.res1 = res1
        cls.res2 = resp.results_nb_docvis
        cls.pred_kwds_mean = {}
        cls.pred_kwds_6 = {}
        cls.k_infl = 0
        cls.rtol = 1e-8


mu = np.log(1.5)
alpha = 1.5
w = -1.5
models = [
    (Logit, {}, np.array([0.1])),
    (Probit, {}, np.array([0.1])),
    (ZeroInflatedPoisson, {}, np.array([w, mu])),
    (ZeroInflatedGeneralizedPoisson, {}, np.array([w, mu, alpha])),
    (ZeroInflatedGeneralizedPoisson, {"p": 1}, np.array([w, mu, alpha])),
    (ZeroInflatedNegativeBinomialP, {}, np.array([w, mu, alpha])),
    (ZeroInflatedNegativeBinomialP, {"p": 1}, np.array([w, mu, alpha])),
    (Poisson, {}, np.array([mu])),
    (NegativeBinomialP, {}, np.array([mu, alpha])),
    (NegativeBinomialP, {"p": 1}, np.array([mu, alpha])),
    (GeneralizedPoisson, {}, np.array([mu, alpha])),
    (GeneralizedPoisson, {"p": 2}, np.array([mu, alpha])),
    (NegativeBinomial, {}, np.array([mu, alpha])),
    (NegativeBinomial, {"loglike_method": 'nb1'}, np.array([mu, alpha])),
    (NegativeBinomial, {"loglike_method": 'geometric'}, np.array([mu])),
    ]

models_influ = [
    Logit,
    Probit,
    Poisson,
    NegativeBinomialP,
    GeneralizedPoisson,
    ZeroInflatedPoisson,
    ZeroInflatedGeneralizedPoisson,
    ZeroInflatedNegativeBinomialP,
    ]


def get_data_simulated():
    np.random.seed(987456348)
    nobs = 500
    x = np.ones((nobs, 1))
    yn = np.random.randn(nobs)
    y = 1 * (1.5 + yn)**2
    y = np.trunc(y+0.5)
    return y, x


y_count, x_const = get_data_simulated()


@pytest.mark.parametrize("case", models)
def test_distr(case):
    y, x = y_count, x_const
    nobs = len(y)
    np.random.seed(987456348)

    cls_model, kwds, params = case
    if issubclass(cls_model, BinaryModel):
        y = (y > 0.5).astype(float)

    mod = cls_model(y, x, **kwds)
    # res = mod.fit()
    params_dgp = params
    distr = mod.get_distribution(params_dgp)
    assert distr.pmf(1).ndim == 1
    try:
        y2 = distr.rvs(size=(nobs, 1)).squeeze()
    except ValueError:
        y2 = distr.rvs(size=nobs).squeeze()

    mod = cls_model(y2, x, **kwds)
    res = mod.fit(start_params=params_dgp, method="bfgs", maxiter=500)
    # params are not close enough to dgp, zero-inflation estimate is noisy
    # assert_allclose(res.params, params_dgp, rtol=0.25)
    distr2 = mod.get_distribution(res.params)
    assert_allclose(distr2.mean().squeeze()[0], y2.mean(), rtol=0.2)
    assert_allclose(distr2.var().squeeze()[0], y2.var(), rtol=0.2)
    var_ = res.predict(which="var")
    assert_allclose(var_, distr2.var().squeeze(), rtol=1e-12)
    mean = res.predict()

    assert_allclose(res.resid_pearson, (y2 - mean) / np.sqrt(var_), rtol=1e-13)

    if not issubclass(cls_model, BinaryModel):
        # smoke, shape, consistency test
        probs = res.predict(which="prob", y_values=np.arange(5))
        assert probs.shape == (len(mod.endog), 5)
        probs2 = res.get_prediction(
            which="prob", y_values=np.arange(5), average=True)
        assert_allclose(probs2.predicted, probs.mean(0), rtol=1e-10)
        dia = res.get_diagnostic()
        dia.probs_predicted
        # fig = dia.plot_probs();
        # fig.suptitle(cls_model.__name__ + repr(kwds), fontsize=16)

    if cls_model in models_influ:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # ZI models warn about missing hat_matrix_diag
            influ = res.get_influence()
            influ.summary_frame()

        assert influ.resid.shape == (len(y2), )

        try:
            resid = influ.resid_score_factor()
            assert resid.shape == (len(y2), )
        except AttributeError:
            # no score_factor in ZI models
            pass
        resid = influ.resid_score()
        assert resid.shape == (len(y2), )

        f_sc = influ.d_fittedvalues_scaled  # requires se from get_prediction()
        assert f_sc.shape == (len(y2), )

        try:
            with warnings.catch_warnings():
                # ZI models warn about missing hat_matrix_diag
                warnings.simplefilter("ignore", category=UserWarning)
                influ.plot_influence()
        except ImportError:
            pass
