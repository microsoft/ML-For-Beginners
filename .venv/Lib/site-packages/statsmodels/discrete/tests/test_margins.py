# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:08:49 2017

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose

# load data into module namespace
from statsmodels.datasets.cpunish import load
from statsmodels.discrete.discrete_model import (
    NegativeBinomial,
    NegativeBinomialP,
    Poisson,
)
import statsmodels.discrete.tests.results.results_count_margins as res_stata
from statsmodels.tools.tools import add_constant

cpunish_data = load()
cpunish_data.exog = np.asarray(cpunish_data.exog)
cpunish_data.endog = np.asarray(cpunish_data.endog)
cpunish_data.exog[:,3] = np.log(cpunish_data.exog[:,3])
exog = add_constant(cpunish_data.exog, prepend=False)
endog = cpunish_data.endog - 1 # avoid zero-truncation
exog /= np.round(exog.max(0), 3)

class CheckMarginMixin:
    rtol_fac = 1

    def test_margins_table(self):
        res1 = self.res1
        sl = self.res1_slice
        rf = self.rtol_fac
        assert_allclose(self.margeff.margeff, self.res1.params[sl], rtol=1e-5 * rf)
        assert_allclose(self.margeff.margeff_se, self.res1.bse[sl], rtol=1e-6 * rf)
        assert_allclose(self.margeff.pvalues, self.res1.pvalues[sl], rtol=5e-6 * rf)
        assert_allclose(self.margeff.conf_int(), res1.margins_table[sl, 4:6],
                        rtol=1e-6 * rf)


class TestPoissonMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [14.1709, 0.7085, -3.4548, -0.539, 3.2368,  -7.9299,
                        -5.0529]
        mod_poi = Poisson(endog, exog)
        res_poi = mod_poi.fit(start_params=start_params)
        #res_poi = mod_poi.fit(maxiter=100)
        marge_poi = res_poi.get_margeff()
        cls.res = res_poi
        cls.margeff = marge_poi

        cls.rtol_fac = 1
        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_poisson_margins_cont


class TestPoissonMarginDummy(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [14.1709, 0.7085, -3.4548, -0.539, 3.2368,  -7.9299,
                        -5.0529]
        mod_poi = Poisson(endog, exog)
        res_poi = mod_poi.fit(start_params=start_params)
        marge_poi = res_poi.get_margeff(dummy=True)
        cls.res = res_poi
        cls.margeff = marge_poi

        cls.res1_slice = [0, 1, 2, 3, 5, 6]
        cls.res1 = res_stata.results_poisson_margins_dummy


class TestNegBinMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomial(endog, exog)
        res = mod.fit(start_params=start_params, method='nm', maxiter=2000)
        marge = res.get_margeff()
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_negbin_margins_cont
        cls.rtol_fac = 5e1
        # negbin has lower agreement with Stata in this case


class TestNegBinMarginDummy(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomial(endog, exog)
        res = mod.fit(start_params=start_params, method='nm', maxiter=2000)
        marge = res.get_margeff(dummy=True)
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = cls.res1_slice = [0, 1, 2, 3, 5, 6]
        cls.res1 = res_stata.results_negbin_margins_dummy
        cls.rtol_fac = 5e1


class TestNegBinPMargin(CheckMarginMixin):
    # this is the same as the nb2 version above for NB-P, p=2

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomialP(endog, exog)   # checks also that default p=2
        res = mod.fit(start_params=start_params, method='nm', maxiter=2000)
        marge = res.get_margeff()
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_negbin_margins_cont
        cls.rtol_fac = 5e1
        # negbin has lower agreement with Stata in this case
