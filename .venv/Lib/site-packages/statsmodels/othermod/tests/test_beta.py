from __future__ import print_function
import io
import os

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
    ValueWarning,
    )
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb

links = families.links

cur_dir = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(cur_dir, "results")


# betareg(I(food/income) ~ income + persons, data = FoodExpenditure)
_income_estimates_mean = u"""\
varname        Estimate  StdError   zvalue     Pr(>|z|)
(Intercept) -0.62254806 0.223853539 -2.781051 5.418326e-03
income      -0.01229884 0.003035585 -4.051556 5.087819e-05
persons      0.11846210 0.035340667  3.352005 8.022853e-04"""

_income_estimates_precision = u"""\
varname  Estimate StdError  zvalue     Pr(>|z|)
(phi) 35.60975   8.079598 4.407366 1.046351e-05
"""

_methylation_estimates_mean = u"""\
varname      Estimate StdError zvalue Pr(>|z|)
(Intercept)  1.44224    0.03401  42.404   2e-16
genderM      0.06986    0.04359   1.603    0.109
CpGCpG_1     0.60735    0.04834  12.563   2e-16
CpGCpG_2     0.97355    0.05311  18.331   2e-16"""

_methylation_estimates_precision = u"""\
varname Estimate StdError zvalue Pr(>|z|)
(Intercept)  8.22829    1.79098   4.594 4.34e-06
age         -0.03471    0.03276  -1.059    0.289"""


expected_income_mean = pd.read_table(
    io.StringIO(_income_estimates_mean), sep=r"\s+")
expected_income_precision = pd.read_table(
    io.StringIO(_income_estimates_precision), sep=r"\s+")

expected_methylation_mean = pd.read_table(
    io.StringIO(_methylation_estimates_mean), sep=r"\s+")
expected_methylation_precision = pd.read_table(
    io.StringIO(_methylation_estimates_precision), sep=r"\s+")

income = pd.read_csv(os.path.join(res_dir, 'foodexpenditure.csv'))
methylation = pd.read_csv(os.path.join(res_dir, 'methylation-test.csv'))


def check_same(a, b, eps, name):
    assert np.allclose(a, b, rtol=0.01, atol=eps), \
            ("different from expected", name, list(a), list(b))


def assert_close(a, b, eps):
    assert np.allclose(a, b, rtol=0.01, atol=eps), (list(a), list(b))


class TestBetaModel:

    @classmethod
    def setup_class(cls):
        model = "I(food/income) ~ income + persons"
        cls.income_fit = BetaModel.from_formula(model, income).fit()

        model = cls.model = "methylation ~ gender + CpG"
        Z = cls.Z = patsy.dmatrix("~ age", methylation)
        mod = BetaModel.from_formula(model, methylation, exog_precision=Z,
                                     link_precision=links.Identity())
        cls.meth_fit = mod.fit()
        mod = BetaModel.from_formula(model, methylation, exog_precision=Z,
                                     link_precision=links.Log())
        cls.meth_log_fit = mod.fit()

    def test_income_coefficients(self):
        rslt = self.income_fit
        assert_close(rslt.params[:-1], expected_income_mean['Estimate'], 1e-3)
        assert_close(rslt.tvalues[:-1], expected_income_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-1], expected_income_mean['Pr(>|z|)'], 1e-3)

    def test_income_precision(self):

        rslt = self.income_fit
        # note that we have to exp the phi results for now.
        assert_close(np.exp(rslt.params[-1:]),
                     expected_income_precision['Estimate'], 1e-3)
        # yield check_same, rslt.tvalues[-1:],
        #                   expected_income_precision['zvalue'], 0.1, "z-score"
        assert_close(rslt.pvalues[-1:],
                     expected_income_precision['Pr(>|z|)'], 1e-3)

    def test_methylation_coefficients(self):
        rslt = self.meth_fit
        assert_close(rslt.params[:-2],
                     expected_methylation_mean['Estimate'], 1e-2)
        assert_close(rslt.tvalues[:-2],
                     expected_methylation_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-2],
                     expected_methylation_mean['Pr(>|z|)'], 1e-2)

    def test_methylation_precision(self):
        # R results are from log link_precision
        rslt = self.meth_log_fit
        assert_allclose(rslt.params[-2:],
                        expected_methylation_precision['Estimate'],
                        atol=1e-5, rtol=1e-10)
        #     expected_methylation_precision['Estimate']
        # yield check_same, links.logit()(rslt.params[-2:]),
        #     expected_methylation_precision['Estimate'], 1e-3, "estimate"
        # yield check_same, rslt.tvalues[-2:],
        #     expected_methylation_precision['zvalue'], 0.1, "z-score"

    def test_precision_formula(self):
        m = BetaModel.from_formula(self.model, methylation,
                                   exog_precision_formula='~ age',
                                   link_precision=links.Identity())
        rslt = m.fit()
        assert_close(rslt.params, self.meth_fit.params, 1e-10)
        assert isinstance(rslt.params, pd.Series)

        with pytest.warns(ValueWarning, match="unknown kwargs"):
            BetaModel.from_formula(self.model, methylation,
                                   exog_precision_formula='~ age',
                                   link_precision=links.Identity(),
                                   junk=False)

    def test_scores(self):
        model, Z = self.model, self.Z
        for link in (links.Identity(), links.Log()):
            mod2 = BetaModel.from_formula(model, methylation, exog_precision=Z,
                                          link_precision=link)
            rslt_m = mod2.fit()

            # evaluate away from optimum to get larger score
            analytical = rslt_m.model.score(rslt_m.params * 1.01)
            numerical = rslt_m.model._score_check(rslt_m.params * 1.01)
            assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-6)
            assert_allclose(link.inverse(analytical[3:]),
                            link.inverse(numerical[3:]), rtol=5e-7, atol=5e-6)

    def test_results_other(self):

        rslt = self.meth_fit
        distr = rslt.get_distribution()
        mean, var = distr.stats()
        assert_allclose(rslt.fittedvalues, mean, rtol=1e-13)
        assert_allclose(rslt.model._predict_var(rslt.params), var, rtol=1e-13)
        resid = rslt.model.endog - mean
        assert_allclose(rslt.resid, resid, rtol=1e-12)
        assert_allclose(rslt.resid_pearson, resid / np.sqrt(var), rtol=1e-12)


class TestBetaMeth():

    @classmethod
    def setup_class(cls):
        formula = "methylation ~ gender + CpG"
        mod = BetaModel.from_formula(formula, methylation,
                                     exog_precision_formula="~ age",
                                     link_precision=links.Log())
        cls.res1 = mod.fit(cov_type="eim")
        cls.res2 = resultsb.results_meth

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        k_mean = 4
        p, se, zv, pv = res2.table_mean.T
        assert_allclose(res1.params[:k_mean], p, rtol=1e-6)
        assert_allclose(res1.bse[:k_mean], se, rtol=1e-6)
        assert_allclose(res1.tvalues[:k_mean], zv, rtol=1e-6)
        assert_allclose(res1.pvalues[:k_mean], pv, rtol=1e-5)

        p, se, zv, pv = res2.table_precision.T
        assert_allclose(res1.params[k_mean:], p, rtol=1e-6)
        assert_allclose(res1.bse[k_mean:], se, rtol=1e-6)
        assert_allclose(res1.tvalues[k_mean:], zv, rtol=1e-6)
        assert_allclose(res1.pvalues[k_mean:], pv, rtol=1e-5)

        assert_allclose(res1.llf, res2.loglik, rtol=1e-10)
        assert_allclose(res1.aic, res2.aic, rtol=1e-10)
        assert_allclose(res1.bic, res2.bic, rtol=1e-10)
        # dofferent definitions for prsquared
        assert_allclose(res1.prsquared, res2.pseudo_r_squared, atol=0.01)
        assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.nobs, res2.nobs)

        # null model compared to R betareg and lmtest
        df_c = res1.df_resid_null - res1.df_resid
        assert_equal(res1.k_null, 2)

        # > lrt = lrtest(res_meth_null, res_meth)  # results from R
        pv = 7.21872953868659e-18
        lln = 60.88809589492269
        llf = 104.14802840534323
        chisq = 86.51986502084107
        dfc = 4
        # stats.chi2.sf(86.51986502093865, 4)
        assert_equal(df_c, dfc)
        assert_allclose(res1.llf, llf, rtol=1e-10)
        assert_allclose(res1.llnull, lln, rtol=1e-10)
        assert_allclose(res1.llr, chisq, rtol=1e-10)
        assert_allclose(res1.llr_pvalue, pv, rtol=1e-6)

    def test_resid(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.fittedvalues, res2.resid['fittedvalues'],
                        rtol=1e-8)
        assert_allclose(res1.resid, res2.resid['response'],
                        atol=1e-8, rtol=1e-8)

    def test_oim(self):
        # estimate with default oim, cov_type nonrobust
        res1 = self.res1.model.fit()
        res2 = self.res2

        k_mean = 4
        # R betareg uses numerical derivatives from bfgs, has lower precision
        p, se, zv, pv = res2.table_mean_oim.T
        assert_allclose(res1.params[:k_mean], p, rtol=1e-6)
        assert_allclose(res1.bse[:k_mean], se, rtol=1e-5)
        assert_allclose(res1.tvalues[:k_mean], zv, rtol=1e-5)
        assert_allclose(res1.pvalues[:k_mean], pv, atol=1e-6, rtol=1e-5)

        p, se, zv, pv = res2.table_precision_oim.T
        assert_allclose(res1.params[k_mean:], p, rtol=1e-6)
        assert_allclose(res1.bse[k_mean:], se, rtol=1e-3)
        assert_allclose(res1.tvalues[k_mean:], zv, rtol=1e-3)
        assert_allclose(res1.pvalues[k_mean:], pv, rtol=1e-2)

    def test_predict_distribution(self):
        res1 = self.res1
        mean = res1.predict()
        var_ = res1.model._predict_var(res1.params)
        distr = res1.get_distribution()
        m2, v2 = distr.stats()
        assert_allclose(mean, m2, rtol=1e-13)
        assert_allclose(var_, v2, rtol=1e-13)

        # from R: > predict(res_meth, type="variance")
        var_r6 = [
            3.1090848852102e-04, 2.4509604000073e-04, 3.7199753140565e-04,
            2.8088261358738e-04, 2.7561111800350e-04, 3.3929220526847e-04]
        n = 6
        assert_allclose(v2[:n], var_r6, rtol=1e-7)

        ex = res1.model.exog[:n]
        ex_prec = res1.model.exog_precision[:n]
        mean6 = res1.predict(ex, transform=False)
        prec = res1.predict(which="precision")
        # todo: prec6 wrong exog if not used as keyword, no exception raised
        prec6 = res1.predict(exog_precision=ex_prec, which="precision",
                             transform=False)
        var6 = res1.model._predict_var(res1.params, exog=ex,
                                       exog_precision=ex_prec)

        assert_allclose(mean6, mean[:n], rtol=1e-13)
        assert_allclose(prec6, prec[:n], rtol=1e-13)
        assert_allclose(var6, var_[:n], rtol=1e-13)
        assert_allclose(var6, var_r6, rtol=1e-7)

        distr6 = res1.model.get_distribution(res1.params,
                                             exog=ex, exog_precision=ex_prec)
        m26, v26 = distr6.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)

        distr6f = res1.get_distribution(exog=ex, exog_precision=ex_prec,
                                        transform=False)
        m26, v26 = distr6f.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)

        # check formula transform works for predict, currently mean only
        df6 = methylation.iloc[:6]
        mean6f = res1.predict(df6)
        # todo: prec6 wrong exog if not used as keyword, no exception raised
        #       formula not supported for exog_precision in predict
        # prec6f = res1.predict(exog_precision=ex_prec, which="precision")
        assert_allclose(mean6f, mean[:n], rtol=1e-13)
        # assert_allclose(prec6f, prec[:n], rtol=1e-13)

        distr6f = res1.get_distribution(exog=df6, exog_precision=ex_prec)
        m26, v26 = distr6f.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)
        # check that we don't have pandas in distr
        assert isinstance(distr6f.args[0], np.ndarray)

        # minimal checks for get_prediction
        pma = res1.get_prediction(which="mean", average=True)
        dfma = pma.summary_frame()
        assert_allclose(pma.predicted, mean.mean(), rtol=1e-13)
        assert_equal(dfma.shape, (1, 4))
        pm = res1.get_prediction(exog=df6, which="mean", average=False)
        dfm = pm.summary_frame()
        assert_allclose(pm.predicted, mean6, rtol=1e-13)
        assert_equal(dfm.shape, (6, 4))
        pv = res1.get_prediction(exog=df6, exog_precision=ex_prec,
                                 which="var", average=False)
        dfv = pv.summary_frame()
        assert_allclose(pv.predicted, var6, rtol=1e-13)
        assert_equal(dfv.shape, (6, 4))
        # smoke tests
        res1.get_prediction(which="linear", average=False)
        res1.get_prediction(which="precision", average=True)
        res1.get_prediction(exog_precision=ex_prec, which="precision",
                            average=False)
        res1.get_prediction(which="linear-precision", average=True)

        # test agg_weights
        pm = res1.get_prediction(exog=df6, which="mean", average=True)
        dfm = pm.summary_frame()
        aw = np.zeros(len(res1.model.endog))
        aw[:6] = 1
        aw /= aw.mean()
        pm6 = res1.get_prediction(exog=df6, which="mean", average=True)
        dfm6 = pm6.summary_frame()
        pmw = res1.get_prediction(which="mean", average=True, agg_weights=aw)
        dfmw = pmw.summary_frame()
        assert_allclose(pmw.predicted, pm6.predicted, rtol=1e-13)
        assert_allclose(dfmw, dfm6, rtol=1e-13)


class TestBetaIncome():

    @classmethod
    def setup_class(cls):

        formula = "I(food/income) ~ income + persons"
        exog_prec = patsy.dmatrix("~ persons", income)
        mod_income = BetaModel.from_formula(
            formula,
            income,
            exog_precision=exog_prec,
            link_precision=links.Log()
            )
        res_income = mod_income.fit(method="newton")

        mod_restricted = BetaModel.from_formula(
            formula,
            income,
            link_precision=links.Log()
            )
        res_restricted = mod_restricted.fit(method="newton")

        cls.res1 = res_income
        cls.resr = res_restricted

    def test_score_test(self):
        res1 = self.res1
        resr = self.resr
        params_restr = np.concatenate([resr.params, [0]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -1] = 1
        exog_prec_extra = res1.model.exog_precision[:, 1:]

        from statsmodels.base._parameter_inference import score_test
        sc1 = score_test(res1, params_constrained=params_restr,
                         k_constraints=1)
        sc2 = score_test(resr, exog_extra=(None, exog_prec_extra))
        assert_allclose(sc2[:2], sc1[:2])

        sc1_hc = score_test(res1, params_constrained=params_restr,
                            k_constraints=1, r_matrix=r_matrix, cov_type="HC0")
        sc2_hc = score_test(resr, exog_extra=(None, exog_prec_extra),
                            cov_type="HC0")
        assert_allclose(sc2_hc[:2], sc1_hc[:2])

    def test_influence(self):
        # currently only smoke test
        res1 = self.res1
        from statsmodels.stats.outliers_influence import MLEInfluence

        influ0 = MLEInfluence(res1)
        influ = res1.get_influence()
        attrs = ['cooks_distance', 'd_fittedvalues', 'd_fittedvalues_scaled',
                 'd_params', 'dfbetas', 'hat_matrix_diag', 'resid_studentized'
                 ]
        for attr in attrs:
            getattr(influ, attr)

        frame = influ.summary_frame()
        frame0 = influ0.summary_frame()
        assert_allclose(frame, frame0, rtol=1e-13, atol=1e-13)
