# -*- coding: utf-8 -*-
"""

Created on Fri Oct 04 13:19:01 2013

Author: Josef Perktold
"""
from statsmodels.compat.python import lrange, lmap

import os
import copy

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd

from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm


def get_griliches76_data():
    curdir = os.path.split(__file__)[0]
    path = os.path.join(curdir, 'griliches76.dta')
    griliches76_data = pd.read_stata(path)

    # create year dummies
    years = griliches76_data['year'].unique()
    N = griliches76_data.shape[0]

    for yr in years:
        griliches76_data['D_%i' % yr] = np.zeros(N)
        for i in range(N):
            if griliches76_data.loc[griliches76_data.index[i], 'year'] == yr:
                griliches76_data.loc[griliches76_data.index[i], 'D_%i' % yr] = 1
            else:
                pass

    griliches76_data['const'] = 1

    X = add_constant(griliches76_data[['s', 'iq', 'expr', 'tenure', 'rns',
                                       'smsa', 'D_67', 'D_68', 'D_69', 'D_70',
                                       'D_71', 'D_73']],
                                       #prepend=False)  # for Stata comparison
                                       prepend=True)  # for R comparison

    Z = add_constant(griliches76_data[['expr', 'tenure', 'rns', 'smsa', \
                                       'D_67', 'D_68', 'D_69', 'D_70', 'D_71',
                                       'D_73', 'med', 'kww', 'age', 'mrt']])
    Y = griliches76_data['lw']

    return Y, X, Z

# use module global to load only once
yg_df, xg_df, zg_df = get_griliches76_data()

endog = np.asarray(yg_df, dtype=float)  # TODO: why is yg_df float32
exog, instrument = lmap(np.asarray, [xg_df, zg_df])

assert exog.dtype == np.float64
assert instrument.dtype == np.float64


# from R
#-----------------
varnames = np.array(["(Intercept)", "s", "iq", "expr", "tenure", "rns", "smsa", "D_67", "D_68", "D_69", "D_70",
       "D_71", "D_73"])
params = np.array([ 4.03350989,  0.17242531, -0.00909883,  0.04928949,  0.04221709,
       -0.10179345,  0.12611095, -0.05961711,  0.04867956,  0.15281763,
        0.17443605,  0.09166597,  0.09323977])
bse = np.array([ 0.31816162,  0.02091823,  0.00474527,  0.00822543,  0.00891969,
        0.03447337,  0.03119615,  0.05577582,  0.05246796,  0.05201092,
        0.06027671,  0.05461436,  0.05767865])
tvalues = np.array([ 12.6775501,   8.2428242,  -1.9174531,   5.9923305,   4.7330205,
        -2.9528144,   4.0425165,  -1.0688701,   0.9277959,   2.9381834,
         2.8939212,   1.6784225,   1.6165385])
pvalues = np.array([  1.72360000e-33,   7.57025400e-16,   5.55625000e-02,
         3.21996700e-09,   2.64739100e-06,   3.24794100e-03,
         5.83809900e-05,   2.85474400e-01,   3.53813900e-01,
         3.40336100e-03,   3.91575100e-03,   9.36840200e-02,
         1.06401300e-01])
    #-----------------

def test_iv2sls_r():

    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()

    # print(res.params)
    # print(res.params - params)

    n, k = exog.shape

    assert_allclose(res.params, params, rtol=1e-7, atol=1e-9)
    # TODO: check df correction
    #assert_allclose(res.bse * np.sqrt((n - k) / (n - k - 1.)), bse,
    assert_allclose(res.bse, bse, rtol=0, atol=3e-7)

    # GH 3849
    assert not hasattr(mod, '_results')



def test_ivgmm0_r():
    n, k = exog.shape
    nobs, k_instr = instrument.shape

    w0inv = np.dot(instrument.T, instrument) / nobs
    w0 = np.linalg.inv(w0inv)

    mod = gmm.IVGMM(endog, exog, instrument)
    res = mod.fit(np.ones(exog.shape[1], float), maxiter=0, inv_weights=w0inv,
                  optim_method='bfgs',
                  optim_args={'gtol':1e-8, 'disp': 0})


    assert_allclose(res.params, params, rtol=1e-4, atol=1e-4)
    # TODO : res.bse and bse are not the same, rtol=0.09 is large in this case
    #res.bse is still robust?, bse is not a sandwich ?
    assert_allclose(res.bse, bse, rtol=0.09, atol=0)

    score = res.model.score(res.params, w0)
    assert_allclose(score, np.zeros(score.shape), rtol=0, atol=5e-6) # atol=1e-8) ??


def test_ivgmm1_stata():

    # copied constant to the beginning
    params_stata = np.array(
          [ 4.0335099 ,  0.17242531, -0.00909883,  0.04928949,  0.04221709,
           -0.10179345,  0.12611095, -0.05961711,  0.04867956,  0.15281763,
            0.17443605,  0.09166597,  0.09323976])

    # robust bse with gmm onestep
    bse_stata = np.array(
          [ 0.33503289,  0.02073947,  0.00488624,  0.0080498 ,  0.00946363,
            0.03371053,  0.03081138,  0.05171372,  0.04981322,  0.0479285 ,
            0.06112515,  0.0554618 ,  0.06084901])

    n, k = exog.shape
    nobs, k_instr = instrument.shape

    w0inv = np.dot(instrument.T, instrument) / nobs
    w0 = np.linalg.inv(w0inv)
    start = OLS(endog, exog).fit().params

    mod = gmm.IVGMM(endog, exog, instrument)
    res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0})


# move constant to end for Stata
idx = lrange(len(params))
idx = idx[1:] + idx[:1]
exog_st = exog[:, idx]


class TestGMMOLS:

    @classmethod
    def setup_class(cls):
        exog = exog_st  # with const at end
        res_ols = OLS(endog, exog).fit()

        #  use exog as instrument
        nobs, k_instr = exog.shape
        w0inv = np.dot(exog.T, exog) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, exog)
        res = mod.fit(np.ones(exog.shape[1], float), maxiter=0, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0})

        cls.res1 = res
        cls.res2 = res_ols


    def test_basic(self):
        res1, res2 = self.res1, self.res2
        # test both absolute and relative difference
        assert_allclose(res1.params, res2.params, rtol=5e-4, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-5)

        n = res1.model.exog.shape[0]
        dffac = 1#np.sqrt((n - 1.) / n)   # currently different df in cov calculation
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=5e-6, atol=0)
        assert_allclose(res1.bse * dffac, res2.HC0_se, rtol=0, atol=1e-7)

    @pytest.mark.xfail(reason="Not asserting anything meaningful",
                       raises=NotImplementedError, strict=True)
    def test_other(self):
        res1, res2 = self.res1, self.res2
        raise NotImplementedError



class CheckGMM:

    params_tol = [5e-6, 5e-6]
    bse_tol = [5e-7, 5e-7]

    def test_basic(self):
        res1, res2 = self.res1, self.res2
        # test both absolute and relative difference
        rtol,  atol = self.params_tol
        assert_allclose(res1.params, res2.params, rtol=rtol, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=atol)

        n = res1.model.exog.shape[0]
        dffac = 1 #np.sqrt((n - 1.) / n)   # currently different df in cov calculation
        rtol,  atol = self.bse_tol
        assert_allclose(res1.bse * dffac, res2.bse, rtol=rtol, atol=0)
        assert_allclose(res1.bse * dffac, res2.bse, rtol=0, atol=atol)

    def test_other(self):
        # TODO: separate Q and J tests
        res1, res2 = self.res1, self.res2
        assert_allclose(res1.q, res2.Q, rtol=5e-6, atol=0)
        assert_allclose(res1.jval, res2.J, rtol=5e-5, atol=0)

    def test_hypothesis(self):
        res1, res2 = self.res1, self.res2
        restriction = np.eye(len(res1.params))
        res_t = res1.t_test(restriction)
        assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
        assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
        rtol,  atol = self.bse_tol
        assert_allclose(res_t.tvalue, res2.tvalues, rtol=rtol*10, atol=atol)
        assert_allclose(res_t.pvalue, res2.pvalues, rtol=rtol*10, atol=atol)

        res_f = res1.f_test(restriction[:-1]) # without constant
        # comparison with fvalue is not possible, those are not defined
        # assert_allclose(res_f.fvalue, res1.fvalue, rtol=1e-12, atol=0)
        # assert_allclose(res_f.pvalue, res1.f_pvalue, rtol=1e-12, atol=0)
        # assert_allclose(res_f.fvalue, res2.F, rtol=1e-10, atol=0)
        # assert_allclose(res_f.pvalue, res2.Fp, rtol=1e-08, atol=0)

        # Smoke test for Wald
        res_wald = res1.wald_test(restriction[:-1], scalar=True)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        # len + 1 is for header line
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)

    def test_use_t(self):
        # Copy to avoid cache
        res1 = copy.deepcopy(self.res1)
        res1.use_t = True
        summ = res1.summary()
        assert 'P>|t|' in str(summ)
        assert 'P>|z|' not in str(summ)


class TestGMMSt1(CheckGMM):

    @classmethod
    def setup_class(cls):
        #cls.bse_tol = [5e-7, 5e-7]
        # compare to Stata default options, iterative GMM
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(start, maxiter=10, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False})
        cls.res1 = res10

        from .results_gmm_griliches_iter import results
        cls.res2 = results

class TestGMMStTwostep(CheckGMM):
    #compares has_optimal_weights=True with Stata's has_optimal_weights=False

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        cls.params_tol = [5e-5, 5e-6]
        cls.bse_tol = [5e-6, 5e-7]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False})
        cls.res1 = res10

        from .results_gmm_griliches import results_twostep as results
        cls.res2 = results


class TestGMMStTwostepNO(CheckGMM):
    #with Stata default `has_optimal_weights=False`

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        cls.params_tol = [5e-5, 5e-6]
        cls.bse_tol = [1e-6, 5e-5]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res10 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res10

        from .results_gmm_griliches import results_twostep as results
        cls.res2 = results


class TestGMMStOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        cls.params_tol = [5e-4, 5e-5]
        cls.bse_tol = [7e-3, 5e-4]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=0, inv_weights=w0inv,
                        optim_method='bfgs',
                        optim_args={'gtol':1e-6, 'disp': 0})
        cls.res1 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    def test_bse_other(self):
        res1, res2 = self.res1, self.res2
        # try other versions for bse,
        # TODO: next two produce the same as before (looks like)
        bse = np.sqrt(np.diag((res1._cov_params(has_optimal_weights=False))))
                                            #weights=res1.weights))))
        # TODO: does not look different
        #assert_allclose(res1.bse, res2.bse, rtol=5e-06, atol=0)
        #nobs = instrument.shape[0]
        #w0inv = np.dot(instrument.T, instrument) / nobs
        q = self.res1.model.gmmobjective(self.res1.params, np.linalg.inv(self.res1.weights))
        #assert_allclose(q, res2.Q, rtol=5e-6, atol=0)

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOnestep, self).test_other()


class TestGMMStOnestepNO(CheckGMM):
    # matches Stats's defaults wargs={'centered':False}, has_optimal_weights=False

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        cls.params_tol = [1e-5, 1e-6]
        cls.bse_tol = [5e-6, 5e-7]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=0, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOnestepNO, self).test_other()


class TestGMMStOneiter(CheckGMM):

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        # this uses maxiter=1, one iteration in loop
        cls.params_tol = [5e-4, 5e-5]
        cls.bse_tol = [7e-3, 5e-4]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0})
        cls.res1 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOneiter, self).test_other()

    def test_bse_other(self):
        res1, res2 = self.res1, self.res2

        moms = res1.model.momcond(res1.params)
        w = res1.model.calc_weightmatrix(moms)
        # try other versions for bse,
        # TODO: next two produce the same as before (looks like)
        bse = np.sqrt(np.diag((res1._cov_params(has_optimal_weights=False,
                                            weights=res1.weights))))
        # TODO: does not look different
        #assert_allclose(res1.bse, res2.bse, rtol=5e-06, atol=0)
        bse = np.sqrt(np.diag((res1._cov_params(has_optimal_weights=False))))
                                                #use_weights=True #weights=w
        #assert_allclose(res1.bse, res2.bse, rtol=5e-06, atol=0)

        #This does not replicate Stata oneway either
        nobs = instrument.shape[0]
        w0inv = np.dot(instrument.T, instrument) / nobs
        q = self.res1.model.gmmobjective(self.res1.params, w)#self.res1.weights)
        #assert_allclose(q, res2.Q, rtol=5e-6, atol=0)


class TestGMMStOneiterNO(CheckGMM):

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        # this uses maxiter=1, one iteration in loop
        cls.params_tol = [1e-5, 1e-6]
        cls.bse_tol = [5e-6, 5e-7]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOneiterNO, self).test_other()


#------------ Crosscheck subclasses

class TestGMMStOneiterNO_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        # this uses maxiter=1, one iteration in loop
        cls.params_tol = [5e-9, 1e-9]
        cls.bse_tol = [5e-10, 1e-10]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.LinearIVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res3 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOneiterNO_Linear, self).test_other()


class TestGMMStOneiterNO_Nonlinear(CheckGMM):

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, onestep GMM
        # this uses maxiter=1, one iteration in loop
        cls.params_tol = [5e-5, 5e-6]
        cls.bse_tol = [5e-6, 1e-1]
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        def func(params, exog):
            return np.dot(exog, params)

        mod = gmm.NonlinearIVGMM(endog, exog, instrument, func)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res3 = res

        from .results_gmm_griliches import results_onestep as results
        cls.res2 = results

    @pytest.mark.xfail(reason="q vs Q comparison fails",
                       raises=AssertionError, strict=True)
    def test_other(self):
        super(TestGMMStOneiterNO_Nonlinear, self).test_other()

    def test_score(self):
        params = self.res1.params * 1.1
        weights = self.res1.weights
        sc1 = self.res1.model.score(params, weights)
        sc2 = super(self.res1.model.__class__, self.res1.model).score(params,
                                                                      weights)
        assert_allclose(sc1, sc2, rtol=1e-6, atol=0)
        assert_allclose(sc1, sc2, rtol=0, atol=1e-7)

        # score at optimum
        sc1 = self.res1.model.score(self.res1.params, weights)
        assert_allclose(sc1, np.zeros(len(params)), rtol=0, atol=1e-8)


class TestGMMStOneiterOLS_Linear(CheckGMM):

    @classmethod
    def setup_class(cls):
        # replicating OLS by GMM - high agreement
        cls.params_tol = [1e-11, 1e-12]
        cls.bse_tol = [1e-12, 1e-12]
        exog = exog_st  # with const at end
        res_ols = OLS(endog, exog).fit()
        #Note: start is irrelevant but required
        start = np.ones(len(res_ols.params))
        nobs, k_instr = instrument.shape
        w0inv = np.dot(exog.T, exog) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.LinearIVGMM(endog, exog, exog)
        res = mod.fit(start, maxiter=0, inv_weights=w0inv,
                        #optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0},
                        optim_args={'disp': 0},
                        weights_method='iid',
                        wargs={'centered':False, 'ddof':'k_params'},
                        has_optimal_weights=True)

        # fix use of t distribution see #2495 comment
        res.use_t = True
        res.df_resid = res.nobs - len(res.params)
        cls.res1 = res

        #from .results_gmm_griliches import results_onestep as results
        #cls.res2 = results
        cls.res2 = res_ols

    @pytest.mark.xfail(reason="RegressionResults has no `Q` attribute",
                       raises=AttributeError, strict=True)
    def test_other(self):
        super(TestGMMStOneiterOLS_Linear, self).test_other()


# ------------------

class TestGMMSt2:
    # this looks like an old version, trying out different comparisons
    # of options with Stats

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, iterative GMM
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        #w0 = np.linalg.inv(w0inv)

        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=2, inv_weights=w0inv,
                      wargs={'ddof':0, 'centered':False},
                      optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0})
        cls.res1 = res

        from .results_ivreg2_griliches import results_gmm2s_robust as results
        cls.res2 = results

        # TODO: remove after testing, compare bse from 1 iteration
        # see test_basic
        mod = gmm.IVGMM(endog, exog, instrument)
        res = mod.fit(start, maxiter=1, inv_weights=w0inv,
                      wargs={'ddof':0, 'centered':False},
                      optim_method='bfgs', optim_args={'gtol':1e-6, 'disp': 0})
        cls.res3 = res


    def test_basic(self):
        res1, res2 = self.res1, self.res2
        # test both absolute and relative difference
        assert_allclose(res1.params, res2.params, rtol=5e-05, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=5e-06)

        n = res1.model.exog.shape[0]
        # TODO: check df correction np.sqrt(745./758 )*res1.bse matches better
        dffact = np.sqrt(745. / 758 )
        assert_allclose(res1.bse * dffact, res2.bse, rtol=5e-03, atol=0)
        assert_allclose(res1.bse * dffact, res2.bse, rtol=0, atol=5e-03)

        # try other versions for bse,
        # TODO: next two produce the same as before (looks like)
        bse = np.sqrt(np.diag((res1._cov_params(has_optimal_weights=True,
                                            weights=res1.weights))))
        assert_allclose(res1.bse, res2.bse, rtol=5e-01, atol=0)

        bse = np.sqrt(np.diag((res1._cov_params(has_optimal_weights=True,
                                               weights=res1.weights,
                                               use_weights=True))))
        assert_allclose(res1.bse, res2.bse, rtol=5e-02, atol=0)

        # TODO: resolve this
        # try bse from previous step, is closer to Stata
        # guess: Stata ivreg2 does not calc for bse update after final iteration
        # need better test case, bse difference is close to numerical optimization precision
        assert_allclose(self.res3.bse, res2.bse, rtol=5e-05, atol=0)
        assert_allclose(self.res3.bse, res2.bse, rtol=0, atol=5e-06)



        # TODO; tvalues are not available yet, no inheritance
        #assert_allclose(res1.tvalues, res2.tvalues, rtol=5e-10, atol=0)




class CheckIV2SLS:

    def test_basic(self):
        res1, res2 = self.res1, self.res2
        # test both absolute and relative difference
        assert_allclose(res1.params, res2.params, rtol=1e-9, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=1e-10)

        n = res1.model.exog.shape[0]
        assert_allclose(res1.bse, res2.bse, rtol=1e-10, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=1e-11)

        assert_allclose(res1.tvalues, res2.tvalues, rtol=5e-10, atol=0)


    def test_other(self):
        res1, res2 = self.res1, self.res2
        assert_allclose(res1.rsquared, res2.r2, rtol=1e-7, atol=0)
        assert_allclose(res1.rsquared_adj, res2.r2_a, rtol=1e-7, atol=0)

        # TODO: why is fvalue different, IV2SLS uses inherited linear
        assert_allclose(res1.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res1.f_pvalue, res2.Fp, rtol=1e-8, atol=0)
        assert_allclose(np.sqrt(res1.mse_resid), res2.rmse, rtol=1e-10, atol=0)
        assert_allclose(res1.ssr, res2.rss, rtol=1e-10, atol=0)
        assert_allclose(res1.uncentered_tss, res2.yy, rtol=1e-10, atol=0)
        assert_allclose(res1.centered_tss, res2.yyc, rtol=1e-10, atol=0)
        assert_allclose(res1.ess, res2.mss, rtol=1e-9, atol=0)

        assert_equal(res1.df_model, res2.df_m)
        assert_equal(res1.df_resid, res2.df_r)

        # TODO: llf raise NotImplementedError
        #assert_allclose(res1.llf, res2.ll, rtol=1e-10, atol=0)


    def test_hypothesis(self):
        res1, res2 = self.res1, self.res2
        restriction = np.eye(len(res1.params))
        res_t = res1.t_test(restriction)
        assert_allclose(res_t.tvalue, res1.tvalues, rtol=1e-12, atol=0)
        assert_allclose(res_t.pvalue, res1.pvalues, rtol=1e-12, atol=0)
        res_f = res1.f_test(restriction[:-1]) # without constant
        # TODO res1.fvalue problem, see issue #1104
        assert_allclose(res_f.fvalue, res1.fvalue, rtol=1e-12, atol=0)
        assert_allclose(res_f.pvalue, res1.f_pvalue, rtol=1e-10, atol=0)
        assert_allclose(res_f.fvalue, res2.F, rtol=1e-10, atol=0)
        assert_allclose(res_f.pvalue, res2.Fp, rtol=1e-08, atol=0)

    def test_hausman(self):
        res1, res2 = self.res1, self.res2
        hausm = res1.spec_hausman()
        # hausman uses se2 = ssr / nobs, no df correction
        assert_allclose(hausm[0], res2.hausman['DWH'], rtol=1e-11, atol=0)
        assert_allclose(hausm[1], res2.hausman['DWHp'], rtol=1e-10, atol=1e-25)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)


class TestIV2SLSSt1(CheckIV2SLS):

    @classmethod
    def setup_class(cls):
        exog = exog_st  # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape

        mod = gmm.IV2SLS(endog, exog, instrument)
        res = mod.fit()
        cls.res1 = res

        from .results_ivreg2_griliches import results_small as results
        cls.res2 = results


    # See GH #2720
    def test_input_dimensions(self):
        rs = np.random.RandomState(1234)
        x = rs.randn(200, 2)
        z = rs.randn(200)
        x[:, 0] = np.sqrt(0.5) * x[:, 0] + np.sqrt(0.5) * z
        z = np.column_stack((x[:, [1]], z[:, None]))
        e = np.sqrt(0.5) * rs.randn(200) + np.sqrt(0.5) * x[:, 0]

        y_1d = y = x[:, 0] + x[:, 1] + e
        y_2d = y[:, None]
        y_series = pd.Series(y)
        y_df = pd.DataFrame(y_series)
        x_1d = x[:, 0]
        x_2d = x
        x_df = pd.DataFrame(x)
        x_df_single = x_df.iloc[:, [0]]
        x_series = x_df.iloc[:, 0]
        z_2d = z
        z_series = pd.Series(z[:, 1])
        z_1d = z_series.values
        z_df = pd.DataFrame(z)

        ys = (y_df, y_series, y_2d, y_1d)
        xs = (x_2d, x_1d, x_df_single, x_df, x_series)
        zs = (z_1d, z_2d, z_series, z_df)
        res2 = gmm.IV2SLS(y_1d, x_2d, z_2d).fit()
        res1 = gmm.IV2SLS(y_1d, x_1d, z_1d).fit()
        res1_2sintr = gmm.IV2SLS(y_1d, x_1d, z_2d).fit()


        for _y in ys:
            for _x in xs:
                for _z in zs:
                    x_1d = np.size(_x) == _x.shape[0]
                    z_1d = np.size(_z) == _z.shape[0]
                    if z_1d and not x_1d:
                        continue
                    res = gmm.IV2SLS(_y, _x, _z).fit()
                    if z_1d:
                        assert_allclose(res.params, res1.params)
                    elif x_1d and not z_1d:
                        assert_allclose(res.params, res1_2sintr.params)
                    else:
                        assert_allclose(res.params, res2.params)

def test_noconstant():
    exog = exog_st[:, :-1]  # with const removed at end

    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()

    assert_equal(res.fvalue, np.nan)
    # smoke test
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)


def test_gmm_basic():
    # this currently tests mainly the param names, exog_names
    # see #4340
    cd = np.array([1.5, 1.5, 1.7, 2.2, 2.0, 1.8, 1.8, 2.2, 1.9, 1.6, 1.8, 2.2,
                   2.0, 1.5, 1.1, 1.5, 1.4, 1.7, 1.42, 1.9])
    dcd = np.array([0, 0.2 ,0.5, -0.2, -0.2, 0, 0.4, -0.3, -0.3, 0.2, 0.4,
                    -0.2, -0.5, -0.4, 0.4, -0.1, 0.3, -0.28, 0.48, 0.2])
    inst = np.column_stack((np.ones(len(cd)), cd))

    class GMMbase(gmm.GMM):
        def momcond(self, params):
            p0, p1, p2, p3 = params
            endog = self.endog[:, None]
            exog = self.exog
            inst = self.instrument

            mom0 = (endog - p0 - p1 * exog) * inst
            mom1 = ((endog - p0 - p1 * exog)**2 -
                    p2 * (exog**(2 * p3)) / 12) * inst
            g = np.column_stack((mom0, mom1))
            return g

    beta0 = np.array([0.1, 0.1, 0.01, 1])
    res = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4,
                  k_params=4).fit(beta0, optim_args={'disp': 0})
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)
    pnames = ['p%2d' % i for i in range(len(res.params))]
    assert_equal(res.model.exog_names, pnames)

    # check set_param_names method
    mod = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4,
                  k_params=4)
    # use arbitrary names
    pnames = ['beta', 'gamma', 'psi', 'phi']
    mod.set_param_names(pnames)
    res1 = mod.fit(beta0, optim_args={'disp': 0})
    assert_equal(res1.model.exog_names, pnames)
