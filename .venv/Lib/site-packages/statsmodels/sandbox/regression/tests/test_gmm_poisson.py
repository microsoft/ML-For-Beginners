'''

TestGMMMultTwostepDefault() has lower precision

'''

from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm

from numpy.testing import assert_allclose, assert_equal


def get_data():
    import os
    curdir = os.path.split(__file__)[0]
    dt = pandas.read_csv(os.path.join(curdir, 'racd10data_with_transformed.csv'))

    # Transformations compared to original data
    ##dt3['income'] /= 10.
    ##dt3['aget'] = (dt3['age'] - dt3['age'].min()) / 5.
    ##dt3['aget2'] = dt3['aget']**2

    # How do we do this with pandas
    mask = ~((np.asarray(dt['private']) == 1) & (dt['medicaid'] == 1))
    mask = mask & (dt['docvis'] <= 70)
    dt3 = dt[mask]
    dt3['const'] = 1   # add constant
    return dt3

DATA = get_data()

#------------- moment conditions for example

def moment_exponential_add(params, exog, exp=True):

    if not np.isfinite(params).all():
        print("invalid params", params)

    # moment condition without instrument
    if exp:
        predicted = np.exp(np.dot(exog, params))
        #if not np.isfinite(predicted).all():
            #print "invalid predicted", predicted
            #raise RuntimeError('invalid predicted')
        predicted = np.clip(predicted, 0, 1e100)  # try to avoid inf
    else:
        predicted = np.dot(exog, params)

    return predicted


def moment_exponential_mult(params, data, exp=True):
    # multiplicative error model

    endog = data[:,0]
    exog = data[:,1:]

    if not np.isfinite(params).all():
        print("invalid params", params)

    # moment condition without instrument
    if exp:
        predicted = np.exp(np.dot(exog, params))
        predicted = np.clip(predicted, 0, 1e100)  # avoid inf
        resid = endog / predicted - 1
        if not np.isfinite(resid).all():
            print("invalid resid", resid)

    else:
        resid = endog - np.dot(exog, params)

    return resid

#------------------- test classes

# copied from test_gmm.py, with changes
class CheckGMM:

    # default tolerance, overwritten by subclasses
    params_tol = [5e-6, 5e-6]
    bse_tol = [5e-7, 5e-7]
    q_tol = [5e-6, 1e-9]
    j_tol = [5e-5, 1e-9]

    def test_basic(self):
        res1, res2 = self.res1, self.res2
        # test both absolute and relative difference
        rtol,  atol = self.params_tol
        assert_allclose(res1.params, res2.params, rtol=rtol, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=atol)

        rtol,  atol = self.bse_tol
        assert_allclose(res1.bse, res2.bse, rtol=rtol, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=atol)

    def test_other(self):
        res1, res2 = self.res1, self.res2
        rtol,  atol = self.q_tol
        assert_allclose(res1.q, res2.Q, rtol=atol, atol=rtol)
        rtol,  atol = self.j_tol
        assert_allclose(res1.jval, res2.J, rtol=atol, atol=rtol)

        j, jpval, jdf = res1.jtest()
        # j and jval should be the same
        assert_allclose(res1.jval, res2.J, rtol=13, atol=13)
        #pvalue is not saved in Stata results
        pval = stats.chi2.sf(res2.J, res2.J_df)
        #assert_allclose(jpval, pval, rtol=1e-4, atol=1e-6)
        assert_allclose(jpval, pval, rtol=rtol, atol=atol)
        assert_equal(jdf, res2.J_df)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)


class TestGMMAddOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])


        cls.bse_tol = [5e-6, 5e-7]
        q_tol = [0.04, 0]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(np.log(endog+1), exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog, exog, instrument, moment_exponential_add)
        res0 = mod.fit(start, maxiter=0, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False})
        cls.res1 = res0

        from .results_gmm_poisson import results_addonestep as results
        cls.res2 = results


class TestGMMAddTwostep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])


        cls.bse_tol = [5e-6, 5e-7]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(np.log(endog+1), exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog, exog, instrument, moment_exponential_add)
        res0 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res0

        from .results_gmm_poisson import results_addtwostep as results
        cls.res2 = results


class TestGMMMultOnestep(CheckGMM):
    #compares has_optimal_weights=True with Stata's has_optimal_weights=False

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income medicaid ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])

        # Need to add all data into exog
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))


        cls.bse_tol = [5e-6, 5e-7]
        cls.q_tol = [0.04, 0]
        cls.j_tol = [0.04, 0]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(start, maxiter=0, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res0

        from .results_gmm_poisson import results_multonestep as results
        cls.res2 = results

class TestGMMMultTwostep(CheckGMM):
    #compares has_optimal_weights=True with Stata's has_optimal_weights=False

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income medicaid ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])

        # Need to add all data into exog
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))


        cls.bse_tol = [5e-6, 5e-7]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':False}, has_optimal_weights=False)
        cls.res1 = res0

        from .results_gmm_poisson import results_multtwostep as results
        cls.res2 = results


class TestGMMMultTwostepDefault(CheckGMM):
    # compares my defaults with the same options in Stata
    # agreement is not very high, maybe vce(unadjusted) is different after all

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income medicaid ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])

        # Need to add all data into exog
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))


        cls.bse_tol = [0.004, 5e-4]
        cls.params_tol = [5e-5, 5e-5]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        #wargs={'centered':True}, has_optimal_weights=True
                       )
        cls.res1 = res0

        from .results_gmm_poisson import results_multtwostepdefault as results
        cls.res2 = results


class TestGMMMultTwostepCenter(CheckGMM):
    #compares my defaults with the same options in Stata

    @classmethod
    def setup_class(cls):
        # compare to Stata default options, twostep GMM
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()

        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income medicaid ssiratio'.split() + XLISTEXOG2 + ['const']

        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])

        # Need to add all data into exog
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))


        cls.bse_tol = [5e-4, 5e-5]
        cls.params_tol = [5e-5, 5e-5]
        q_tol = [5e-5, 1e-8]
        # compare to Stata default options, iterative GMM
        # with const at end
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs

        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(start, maxiter=2, inv_weights=w0inv,
                        optim_method='bfgs', optim_args={'gtol':1e-8, 'disp': 0},
                        wargs={'centered':True}, has_optimal_weights=False
                       )
        cls.res1 = res0

        from .results_gmm_poisson import results_multtwostepcenter as results
        cls.res2 = results

    def test_more(self):

        # from Stata `overid`
        J_df = 1
        J_p = 0.332254330027383
        J = 0.940091427212973

        j, jpval, jdf = self.res1.jtest()

        assert_allclose(jpval, J_p, rtol=5e-5, atol=0)



if __name__ == '__main__':
    tt = TestGMMAddOnestep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()

    tt = TestGMMAddTwostep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()

    tt = TestGMMMultOnestep()
    tt.setup_class()
    tt.test_basic()
    #tt.test_other()

    tt = TestGMMMultTwostep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()

    tt = TestGMMMultTwostepDefault()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()

    tt = TestGMMMultTwostepCenter()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
