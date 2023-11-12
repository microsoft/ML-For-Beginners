# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:38:13 2017

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from statsmodels.discrete.discrete_model import Poisson
import statsmodels.discrete._diagnostics_count as dia
from statsmodels.discrete.diagnostic import PoissonDiagnostic


class TestCountDiagnostic:

    @classmethod
    def setup_class(cls):

        expected_params = [1, 1, 0.5]
        np.random.seed(987123)
        nobs = 500
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 0
        # offset is used to create misspecification of the model
        # for predicted probabilities conditional moment test
        # offset = 0.5 * np.random.randn(nobs)
        # range_mix = 0.5
        # offset = -range_mix / 2 + range_mix * np.random.rand(nobs)
        offset = 0
        mu_true = np.exp(exog.dot(expected_params[:-1]) + offset)

        endog_poi = np.random.poisson(mu_true / 5)
        # endog3 = distr.zigenpoisson.rvs(mu_true, 0,
        #                                2, 0.01, size=mu_true.shape)

        model_poi = Poisson(endog_poi, exog)
        res_poi = model_poi.fit(method='bfgs', maxiter=5000, disp=False)
        cls.exog = exog
        cls.endog = endog_poi
        cls.res = res_poi
        cls.nobs = nobs

    def test_count(self):
        # partially smoke
        tzi1 = dia.test_poisson_zeroinflation_jh(self.res)

        tzi2 = dia.test_poisson_zeroinflation_broek(self.res)
        # compare two implementation in special case
        assert_allclose(tzi1[:2], (tzi2[0]**2, tzi2[1]), rtol=1e-5)

        tzi3 = dia.test_poisson_zeroinflation_jh(self.res, self.exog)

        # regression test
        tzi3_1 = (0.79863597832443878, 0.67077736750318928)
        assert_allclose(tzi3, tzi3_1, rtol=5e-4)
        assert_equal(tzi3.df, 2)

    @pytest.mark.matplotlib
    def test_probs(self, close_figures):
        nobs = self.nobs
        probs = self.res.predict_prob()
        freq = np.bincount(self.endog) / nobs

        tzi = dia.test_chisquare_prob(self.res, probs[:, :2])
        # regression numbers
        tzi1 = (0.387770845, 0.5334734738)
        assert_allclose(tzi[:2], tzi1, rtol=5e-5)

        # smoke test for plot
        dia.plot_probs(freq, probs.mean(0))


class TestPoissonDiagnosticClass():

    @classmethod
    def setup_class(cls):
        np.random.seed(987125643)
        nr = 1
        n_groups = 2
        labels = np.arange(n_groups)
        x = np.repeat(labels, np.array([40, 60]) * nr)
        nobs = x.shape[0]
        exog = (x[:, None] == labels).astype(np.float64)
        # reparameterize to explicit constant
        # exog[:, 1] = 1
        beta = np.array([0.1, 0.3], np.float64)

        linpred = exog @ beta
        mean = np.exp(linpred)
        y = np.random.poisson(mean)

        cls.endog = y
        cls.exog = exog

    def test_spec_tests(self):
        # regression test, numbers similar to Monte Carlo simulation
        res_dispersion = np.array([
            [0.1396096387543, 0.8889684245877],
            [0.1396096387543, 0.8889684245877],
            [0.2977840351238, 0.7658680002106],
            [0.1307899995877, 0.8959414342111],
            [0.1307899995877, 0.8959414342111],
            [0.1357101381056, 0.8920504328246],
            [0.2776587511235, 0.7812743277372]
            ])

        res_zi = np.array([
            [00.1389582826821, 0.7093188241734],
            [-0.3727710861669, 0.7093188241734],
            [-0.2496729648642, 0.8028402670888],
            [00.0601651553909, 0.8062350958880],
            ])

        respoi = Poisson(self.endog, self.exog).fit(disp=0)
        dia = PoissonDiagnostic(respoi)
        t_disp = dia.test_dispersion()
        res_disp = np.column_stack(((t_disp.statistic, t_disp.pvalue)))
        assert_allclose(res_disp, res_dispersion, rtol=1e-8)

        nobs = self.endog.shape[0]
        t_zi_jh = dia.test_poisson_zeroinflation(method="broek",
                                                 exog_infl=np.ones(nobs))
        t_zib = dia.test_poisson_zeroinflation(method="broek")
        t_zim = dia.test_poisson_zeroinflation(method="prob")
        t_zichi2 = dia.test_chisquare_prob(bin_edges=np.arange(3))

        t_zi = np.vstack([t_zi_jh[:2], t_zib[:2], t_zim[:2], t_zichi2[:2]])
        assert_allclose(t_zi, res_zi, rtol=1e-8)

        # test jansakul and hinde with exog_infl
        t_zi_ex = dia.test_poisson_zeroinflation(method="broek",
                                                 exog_infl=self.exog)
        res_zi_ex = np.array([3.7813218150779, 0.1509719973257])
        assert_allclose(t_zi_ex[:2], res_zi_ex, rtol=1e-8)
