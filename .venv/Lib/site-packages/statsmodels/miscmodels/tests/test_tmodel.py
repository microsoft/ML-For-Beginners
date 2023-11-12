# -*- coding: utf-8 -*-
"""

Created on Sun Jun 30 20:25:22 2013

Author: Josef Perktold
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.miscmodels.tmodel import TLinearModel


mm = Holder()
mm.date_label = ["Apr.1982",  "Apr.1983", "Apr.1984", "Apr.1985", "Apr.1986",
                 "Aug.1982", "Aug.1983",  "Aug.1984", "Aug.1985", "Aug.1986",
                 "Dec.1982", "Dec.1983", "Dec.1984",  "Dec.1985", "Dec.1986",
                 "Feb.1284", "Feb.1982", "Feb.1983", "Feb.1985",  "Feb.1986",
                 "Jan.1982", "Jan.1983", "Jan.1984", "Jan.1985", "Jan.1986",
                 "Jul.1982", "July1983", "July1984", "July1985", "July1986",
                 "June1982",  "June1983", "June1984", "June1985", "June1986",
                 "Mar.1982", "Mar.1983",  "Mar.1984", "Mar.1985", "Mar.1986",
                 "May1982", "May1983", "May1984",  "May1985", "May1986",
                 "Nov.1982", "Nov.1983", "Nov.1984", "Nov.1985",  "Nov.1986",
                 "Oct.1982", "Oct.1983", "Oct.1984", "Oct.1985", "Oct.1986",
                 "Sept.1982", "Sept.1983", "Sept.1984", "Sept.1985",
                 "Sept.1986"]

mm.m_marietta = np.array([
     -0.1365, -0.0769, -0.0575, 0.0526, -0.0449, -0.0859, -0.0742, 0.6879,
     -0.077, 0.085, 0.003, 0.0754, -0.0412, -0.089, 0.2319, 0.1087, 0.0375,
     0.0958, 0.0174, -0.0724, 0.075, -0.0588, -0.062, -0.0378, 0.0169,
     -0.0799, -0.0147, 0.0106, -0.0421, -0.0036, 0.0876, 0.1025, -0.0499,
     0.1953, -0.0714, 0.0469, 0.1311, 0.0461, -0.0328, -0.0096, 0.1272,
     -0.0077, 0.0165, -0.015, -0.1479, -0.0065, 0.039, 0.0223, -0.069,
     0.1338, 0.1458, 0.0063, 0.0692, -0.0239, -0.0568, 0.0814, -0.0889,
     -0.0887, 0.1037, -0.1163
    ])
mm.CRSP = np.array([
     -0.03, -0.0584, -0.0181, 0.0306, -0.0397, -0.0295, -0.0316, 0.1176,
     0.0075, 0.1098, 0.0408, 0.0095, 0.0301, 0.0221, 0.0269, 0.0655,
     -0.003, 0.0325, -0.0374, 0.0049, 0.0105, -0.0257, 0.0186, -0.0155,
     -0.0165, -0.044, 0.0094, -0.0028, -0.0591, 0.0158, -0.0238, 0.1031,
     -0.0065, -0.0067, -0.0167, 0.0188, 0.0733, 0.0105, -0.007, -0.0099,
     0.0521, 0.0117, -0.0099, -0.0102, -0.0428, 0.0376, 0.0628, 0.0391,
     2e-04, 0.0688, 0.0486, -0.0174, 0.046, 0.01, -0.0594, 0.068, -0.0839,
     0.0481, 0.0136, -0.0322
    ])
mm.am_can = np.array([
     -0.0596, -0.17, 0.0276, 0.0058, -0.0106, 0.045, -0.0243, 0.1135,
     -0.0331, 0.0468, -0.0223, -0.0026, 0.0166, 0.0343, 0.0443, 0.1477,
     0.1728, -0.0372, -0.0451, -0.0257, 0.0509, 0.0035, 0.1334, -0.0458,
     0.1199, -0.0766, -0.0511, -0.0194, -0.0687, 0.0928, -0.0704, 0.0905,
     0.0232, -0.0054, 0.0082, 0.0242, 0.0153, 0.0016, 0.028, 0.0088,
     0.0734, 0.0315, -0.0276, 0.0162, -0.0975, 0.0563, 0.1368, -0.069,
     0.1044, 0.1636, -0.019, -0.0746, 0.0433, 0.0306, 0.0636, 0.0917,
     -0.0796, 0.0778, -0.0353, -0.0137
    ])
mm.date = np.array([
     21, 17, 36, 1, 41, 31, 26, 6, 56, 51, 46, 11, 22, 18, 37, 2, 42, 32,
     27, 7, 57, 52, 47, 12, 23, 16, 38, 3, 43, 33, 28, 8, 58, 53, 48, 13,
     24, 19, 39, 4, 44, 34, 29, 9, 59, 54, 49, 14, 25, 20, 40, 5, 45, 35,
     30, 10, 60, 55, 50, 15
    ])


class CheckTLinearModelMixin:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        # location
        assert_allclose(res1.params[:-2], res2.loc_fit.coefficients, atol=3e-5)
        assert_allclose(res1.bse[:-2], res2.loc_fit.table[:, 1], rtol=0.003, atol=1e-5)
        assert_allclose(res1.tvalues[:-2], res2.loc_fit.table[:, 2], rtol=0.003, atol=1e-5)
        assert_allclose(res1.pvalues[:-2], res2.loc_fit.table[:, 3], rtol=0.009, atol=1e-5)

        # df
        assert_allclose(res1.params[-2], res2.dof, rtol=5e-5)
        assert_allclose(res1.bse[-2], res2.dofse, rtol=0.16, atol=1e-5)
        # scale
        scale_est = np.sqrt(res2.scale_fit.fitted_values.mean())
        assert_allclose(res1.params[-1], scale_est, atol=1e-5)

        assert_allclose(res1.llf, res2.logLik, atol=1e-5)

    def test_bse(self):
        # check that they are roughly the same
        res1 = self.res1
        assert_allclose(res1.bsejac, res1.bse, rtol=0.15, atol=0.002)
        assert_allclose(res1.bsejac, res1.bse, rtol=0.1, atol=0.004)

    def test_fitted(self):
        res1 = self.res1
        res2 = self.res2

        fittedvalues = res1.predict()
        resid = res1.model.endog - fittedvalues
        assert_allclose(fittedvalues, res2.loc_fit.fitted_values, rtol=0.00025)
        assert_allclose(resid, res2.loc_fit.residuals, atol=2e-6) #rtol=0.00036)
        #TODO: no resid available as attribute
        #assert_allclose(res1.resid, res2.loc_fit.residuals)
        #assert_allclose(res1.fittedvalues, res2.loc_fit.fitted_values)

    def test_formula(self):
        res1 = self.res1
        resf = self.resf
        # converges slightly differently why?
        assert_allclose(res1.params, resf.params,  atol=1e-4) #rtol=2e-5,
        assert_allclose(res1.bse, resf.bse, rtol=5e-5)

        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_df(self):
        res = self.res1
        k_extra = getattr(self, "k_extra", 0)
        nobs, k_vars = res.model.exog.shape
        assert res.df_resid == nobs - k_vars - k_extra
        assert res.df_model == k_vars - 1  # -1 for constant
        assert len(res.params) == k_vars + k_extra

    @pytest.mark.smoke
    def test_smoke(self):  # TODO: break into well-scoped tests
        res1 = self.res1
        resf = self.resf
        contr = np.eye(len(res1.params))

        # smoke test for summary and t_test, f_test
        res1.summary()
        res1.t_test(contr)
        res1.f_test(contr)

        resf.summary()
        resf.t_test(contr)
        resf.f_test(contr)


class TestTModel(CheckTLinearModelMixin):

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta
        exog = add_constant(mm.CRSP)
        mod = TLinearModel(endog, exog)
        res = mod.fit(method='bfgs', disp=False)
        modf = TLinearModel.from_formula("price ~ CRSP",
                                data={"price":mm.m_marietta, "CRSP":mm.CRSP})
        resf = modf.fit(method='bfgs', disp=False)
        from .results_tmodel import res_t_dfest as res2
        cls.res2 = res2
        cls.res1 = res  # take from module scope temporarily
        cls.resf = resf
        cls.k_extra = 2


class TestTModelFixed:

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta
        exog = add_constant(mm.CRSP)
        mod = TLinearModel(endog, exog, fix_df=3)
        res = mod.fit(method='bfgs', disp=False)
        modf = TLinearModel.from_formula("price ~ CRSP",
                                data={"price":mm.m_marietta, "CRSP":mm.CRSP},
                                fix_df=3)
        resf = modf.fit(method='bfgs', disp=False)
        #TODO: no reference results yet
        #from results_tmodel import res_t_dfest as res2
        #cls.res2 = res2
        cls.res1 = res  # take from module scope temporarily
        cls.resf = resf
        cls.k_extra = 1

    @pytest.mark.smoke
    def test_smoke(self):  # TODO: break into well-scoped tests
        res1 = self.res1
        resf = self.resf
        contr = np.eye(len(res1.params))

        # smoke test for summary and t_test, f_test
        res1.summary()
        res1.t_test(contr)
        res1.f_test(contr)

        resf.summary()
        resf.t_test(contr)
        resf.f_test(contr)
