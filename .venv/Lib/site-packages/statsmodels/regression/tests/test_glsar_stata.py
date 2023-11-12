# -*- coding: utf-8 -*-
"""Testing GLSAR against STATA

Created on Wed May 30 09:25:24 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal

from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata


class CheckStataResultsMixin:

    def test_params_table(self):
        res, results = self.res, self.results
        assert_almost_equal(res.params, results.params, 3)
        assert_almost_equal(res.bse, results.bse, 3)
        # assert_almost_equal(res.tvalues, results.tvalues, 3) 0.0003
        assert_allclose(res.tvalues, results.tvalues, atol=0, rtol=0.004)
        assert_allclose(res.pvalues, results.pvalues, atol=1e-7, rtol=0.004)


class CheckStataResultsPMixin(CheckStataResultsMixin):

    def test_predicted(self):
        res, results = self.res, self.results
        assert_allclose(res.fittedvalues, results.fittedvalues, rtol=0.002)
        predicted = res.predict(res.model.exog) #should be equal
        assert_allclose(predicted, results.fittedvalues, rtol=0.0016)
        # not yet
        # assert_almost_equal(res.fittedvalues_se, results.fittedvalues_se, 4)


class TestGLSARCorc(CheckStataResultsPMixin):

    @classmethod
    def setup_class(cls):
        d2 = macrodata.load_pandas().data
        g_gdp = 400*np.diff(np.log(d2['realgdp'].values))
        g_inv = 400*np.diff(np.log(d2['realinv'].values))
        exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values], prepend=False)

        mod1 = GLSAR(g_inv, exogg, 1)
        cls.res = mod1.iterative_fit(5)

        from .results.macro_gr_corc_stata import results
        cls.results = results

    def test_rho(self):
        assert_almost_equal(self.res.model.rho, self.results.rho, 3)  # pylint: disable-msg=E1101

        assert_almost_equal(self.res.llf, self.results.ll, 4)

    def test_glsar_arima(self):
        from statsmodels.tsa.arima.model import ARIMA

        endog = self.res.model.endog
        exog = self.res.model.exog
        mod1 = GLSAR(endog, exog, 3)
        res = mod1.iterative_fit(10)
        mod_arma = ARIMA(endog, order=(3,0,0), exog=exog[:, :-1])
        res_arma = mod_arma.fit()
        assert_allclose(res.params, res_arma.params[[1,2,0]], atol=0.01, rtol=1e-2)
        assert_allclose(res.model.rho, res_arma.params[3:6], atol=0.05, rtol=1e-3)
        assert_allclose(res.bse, res_arma.bse[[1,2,0]], atol=0.1, rtol=1e-3)

        assert_equal(len(res.history['params']), 5)
        # this should be identical, history has last fit
        assert_equal(res.history['params'][-1], res.params)

        res2 = mod1.iterative_fit(4, rtol=0)
        assert_equal(len(res2.history['params']), 4)
        assert_equal(len(res2.history['rho']), 4)

    def test_glsar_iter0(self):
        endog = self.res.model.endog
        exog = self.res.model.exog

        rho = np.array([ 0.207,  0.275,  1.033])
        mod1 = GLSAR(endog, exog, rho)
        res1 = mod1.fit()
        res0 = mod1.iterative_fit(0)
        res0b = mod1.iterative_fit(1)
        # check iterative_fit(0) or iterative_fit(1) does not update rho
        assert_allclose(res0.params, res1.params, rtol=1e-11)
        assert_allclose(res0b.params, res1.params, rtol=1e-11)
        assert_allclose(res0.model.rho, rho, rtol=1e-11)
        assert_allclose(res0b.model.rho, rho, rtol=1e-11)
