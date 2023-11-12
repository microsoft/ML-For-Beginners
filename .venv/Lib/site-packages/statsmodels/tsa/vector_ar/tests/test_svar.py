"""
Test SVAR estimation
"""
from statsmodels.compat.platform import PLATFORM_WIN

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4


class TestSVAR:
    @classmethod
    def setup_class(cls):
        mdata = statsmodels.datasets.macrodata.load_pandas().data
        mdata = mdata[['realgdp', 'realcons', 'realinv']]
        data = mdata.values
        data = np.diff(np.log(data), axis=0)
        A = np.asarray([[1, 0, 0], ['E', 1, 0], ['E', 'E', 1]], dtype="U")
        B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']], dtype="U")
        results = SVAR(data, svar_type='AB', A=A, B=B).fit(maxlags=3)
        cls.res1 = results
        #cls.res2 = results_svar.SVARdataResults()
        from .results import results_svar_st
        cls.res2 = results_svar_st.results_svar1_small

    def _reformat(self, x):
        return x[[1, 4, 7, 2, 5, 8, 3, 6, 9, 0], :].ravel("F")

    def test_A(self):
        assert_almost_equal(self.res1.A, self.res2.A, DECIMAL_4)

    def test_B(self):
        # see issue #3148, adding np.abs to make solution positive
        # general case will need positive sqrt of covariance matrix
        assert_almost_equal(np.abs(self.res1.B), self.res2.B, DECIMAL_4)

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(self._reformat(res1.params), res2.b_var, atol=1e-12)
        bse_st = np.sqrt(np.diag(res2.V_var))
        assert_allclose(self._reformat(res1.bse), bse_st, atol=1e-12)

    def test_llf_ic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.ll_var, atol=1e-12)
        # different definition, missing constant term ?
        corr_const = -8.51363119922803
        assert_allclose(res1.fpe, res2.fpe_var, atol=1e-12)
        assert_allclose(res1.aic - corr_const, res2.aic_var, atol=1e-12)
        assert_allclose(res1.bic - corr_const, res2.sbic_var, atol=1e-12)
        assert_allclose(res1.hqic - corr_const, res2.hqic_var, atol=1e-12)

    @pytest.mark.smoke
    def test_irf(self):
        # mostly SMOKE, API test
        # this only checks that the methods work and produce the same result
        res1 = self.res1
        errband1 = res1.sirf_errband_mc(orth=False, repl=50, steps=10,
                                        signif=0.05, seed=987123, burn=100,
                                        cum=False)

        irf = res1.irf()
        errband2 = irf.errband_mc(orth=False, svar=True, repl=50,
                                  signif=0.05, seed=987123, burn=100)
        # Windows precision limits require non-zero atol
        atol = 1e-6 if PLATFORM_WIN else 1e-8
        assert_allclose(errband1, errband2, rtol=1e-8, atol=atol)
