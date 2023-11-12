"""
Created on Feb 3, 2022 1:04:22 PM

Author: Josef Perktold
License: BSD-3
"""

import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Probit
from statsmodels.treatment.treatment_effects import (
    TreatmentEffect
    )

from .results import results_teffects as res_st


cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'cataneo2.csv'
file_path = os.path.join(cur_dir, 'results', file_name)

dta_cat = pd.read_csv(file_path)

formula = 'mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu'
res_probit = Probit.from_formula(formula, dta_cat).fit()

methods = [
    ("ra", res_st.results_ra),
    ("ipw", res_st.results_ipw),
    ("aipw", res_st.results_aipw),
    ("aipw_wls", res_st.results_aipw_wls),
    ("ipw_ra", res_st.results_ipwra),
    ]


class TestTEffects():

    @classmethod
    def setup_class(cls):
        formula_outcome = 'bweight ~ prenatal1_ + mmarried_ + mage + fbaby_'
        mod = OLS.from_formula(formula_outcome, dta_cat)
        tind = np.asarray(dta_cat['mbsmoke_'])
        cls.teff = TreatmentEffect(mod, tind, results_select=res_probit)

    def test_aux(self):
        prob = res_probit.predict()
        assert prob.shape == (4642,)

    @pytest.mark.parametrize('case', methods)
    def test_effects(self, case):
        meth, res2 = case
        teff = self.teff

        res1 = getattr(teff, meth)(return_results=False)
        assert_allclose(res1[:2], res2.table[:2, 0], rtol=1e-4)

        # if meth in ["ipw", "aipw", "aipw_wls", "ra", "ipw_ra"]:
        res0 = getattr(teff, meth)(return_results=True)
        assert_allclose(res1, res0.effect, rtol=1e-4)
        res1 = res0.results_gmm
        # TODO: check ra and ipw difference 5e-6, others pass at 1e-12
        assert_allclose(res0.start_params, res1.params, rtol=1e-5)
        assert_allclose(res1.params[:2], res2.table[:2, 0], rtol=1e-5)
        assert_allclose(res1.bse[:2], res2.table[:2, 1], rtol=1e-3)
        assert_allclose(res1.tvalues[:2], res2.table[:2, 2], rtol=1e-3)
        assert_allclose(res1.pvalues[:2], res2.table[:2, 3],
                        rtol=1e-4, atol=1e-15)
        ci = res1.conf_int()
        assert_allclose(ci[:2, 0], res2.table[:2, 4], rtol=5e-4)
        assert_allclose(ci[:2, 1], res2.table[:2, 5], rtol=5e-4)

        # test all GMM params
        # constant is in different position in Stata, `idx` rearanges
        k_p = len(res1.params)
        if k_p == 8:
            # IPW, no outcome regression
            idx = [0, 1, 7, 2, 3, 4, 5, 6]
        elif k_p == 18:
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10, 17, 12, 13, 14,
                   15, 16]
        elif k_p == 12:
            # RA, no selection regression
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10]
        else:
            idx = np.arange(k_p)

        # TODO: check if improved optimization brings values closer
        assert_allclose(res1.params, res2.table[idx, 0], rtol=1e-4)
        assert_allclose(res1.bse, res2.table[idx, 1], rtol=0.05)

        # test effects on the treated, not available for aipw
        if not meth.startswith("aipw"):
            table = res2.table_t

            res1 = getattr(teff, meth)(return_results=False, effect_group=1)
            assert_allclose(res1[:2], table[:2, 0], rtol=1e-4)

            res0 = getattr(teff, meth)(return_results=True, effect_group=1)
            # TODO: check ipw difference 1e-5, others pass at 1e-12
            assert_allclose(res1, res0.effect, rtol=2e-5)
            res1 = res0.results_gmm
            # TODO: check ra difference 4e-5, others pass at 1e-12
            assert_allclose(res0.start_params, res1.params, rtol=5e-5)
            assert_allclose(res1.params[:2], table[:2, 0], rtol=5e-5)
            assert_allclose(res1.bse[:2], table[:2, 1], rtol=1e-3)
            assert_allclose(res1.tvalues[:2], table[:2, 2], rtol=1e-3)
            assert_allclose(res1.pvalues[:2], table[:2, 3],
                            rtol=1e-4, atol=1e-15)
            ci = res1.conf_int()
            assert_allclose(ci[:2, 0], table[:2, 4], rtol=5e-4)
            assert_allclose(ci[:2, 1], table[:2, 5], rtol=5e-4)

            # consistency check, effect on untreated,  not in Stata
            res1 = getattr(teff, meth)(return_results=False, effect_group=0)
            res0 = getattr(teff, meth)(return_results=True, effect_group=0)
            assert_allclose(res1, res0.effect, rtol=1e-12)
            assert_allclose(res0.start_params, res0.results_gmm.params,
                            rtol=1e-12)
