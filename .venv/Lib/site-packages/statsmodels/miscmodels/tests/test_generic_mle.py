# -*- coding: utf-8 -*-
"""

Created on Fri Jun 28 14:19:26 2013

Author: Josef Perktold
"""


import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

from numpy.testing import (assert_array_less, assert_almost_equal,
                           assert_allclose)

class MyPareto(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation pareto distribution

    first version: iid case, with constant parameters
    '''

    def initialize(self):   #TODO needed or not
        super(MyPareto, self).initialize()
        extra_params_names = ['shape', 'loc', 'scale']
        self._set_extra_params_names(extra_params_names)

        #start_params needs to be attribute
        self.start_params = np.array([1.5, self.endog.min() - 1.5, 1.])


    #copied from stats.distribution
    def pdf(self, x, b):
        return b * x**(-b-1)

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    # TODO: design start_params needs to be an attribute,
    # so it can be overwritten
#    @property
#    def start_params(self):
#        return np.array([1.5, self.endog.min() - 1.5, 1.])

    def nloglikeobs(self, params):
        #print params.shape
        if self.fixed_params is not None:
            #print 'using fixed'
            params = self.expandparams(params)
        b = params[0]
        loc = params[1]
        scale = params[2]
        #loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        logpdf = np.log(b) - (b+1.)*np.log(x)  #use np_log(1 + x) for Pareto II
        logpdf -= np.log(scale)
        #lb = loc + scale
        #logpdf[endog<lb] = -inf
        #import pdb; pdb.set_trace()
        logpdf[x<1] = -10000 #-np.inf
        return -logpdf


class CheckGenericMixin:
    # mostly smoke tests for now

    def test_summary(self):
        summ = self.res1.summary()
        check_str = 'P>|t|' if self.res1.use_t else 'P>|z|'
        assert check_str in str(summ)

    def test_use_t_summary(self):
        orig_val = self.res1.use_t
        self.res1.use_t = True
        summ = self.res1.summary()
        assert 'P>|t|' in str(summ)
        self.res1.use_t = orig_val

    def test_ttest(self):
        self.res1.t_test(np.eye(len(self.res1.params)))

    def test_params(self):
        params = self.res1.params

        params_true = np.array([2,0,2])
        if self.res1.model.fixed_paramsmask is not None:
            params_true = params_true[self.res1.model.fixed_paramsmask]
        assert_allclose(params, params_true, atol=1.5)
        assert_allclose(params, np.zeros(len(params)), atol=4)

        assert_allclose(self.res1.bse, np.zeros(len(params)), atol=0.5)
        if not self.skip_bsejac:
            assert_allclose(self.res1.bse, self.res1.bsejac, rtol=0.05,
                            atol=0.15)
            # bsejhj is very different from the other two
            # use huge atol as sanity check for availability
            assert_allclose(self.res1.bsejhj, self.res1.bsejac,
                            rtol=0.05, atol=1.5)

    def test_df(self):
        res = self.res1
        k_extra = getattr(self, "k_extra", 0)
        if res.model.exog is not None:
            nobs, k_vars = res.model.exog.shape
            k_constant = 1
        else:
            nobs, k_vars = res.model.endog.shape[0], 0
            k_constant = 0
        assert res.df_resid == nobs - k_vars - k_extra
        assert res.df_model == k_vars - k_constant
        assert len(res.params) == k_vars + k_extra


class TestMyPareto1(CheckGenericMixin):

    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 100
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))

        mod_par = MyPareto(rvs)
        mod_par.fixed_params = None
        mod_par.fixed_paramsmask = None
        mod_par.df_model = 0
        mod_par.k_extra = k_extra = 3
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model -  k_extra
        mod_par.data.xnames = ['shape', 'loc', 'scale']

        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)
        cls.k_extra = k_extra

        # Note: possible problem with parameters close to min data boundary
        # see issue #968
        cls.skip_bsejac = True

    def test_minsupport(self):
        # rough sanity checks for convergence
        params = self.res1.params
        x_min = self.res1.endog.min()
        p_min = params[1] + params[2]
        assert_array_less(p_min, x_min)
        assert_almost_equal(p_min, x_min, decimal=2)

class TestMyParetoRestriction(CheckGenericMixin):


    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 50
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))

        mod_par = MyPareto(rvs)
        fixdf = np.nan * np.ones(3)
        fixdf[1] = -0.1
        mod_par.fixed_params = fixdf
        mod_par.fixed_paramsmask = np.isnan(fixdf)
        mod_par.start_params = mod_par.start_params[mod_par.fixed_paramsmask]
        mod_par.df_model = 0
        mod_par.k_extra = k_extra = 2
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model - k_extra
        mod_par.data.xnames = ['shape', 'scale']

        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)
        cls.k_extra = k_extra

        # Note: loc is fixed, no problems with parameters close to min data
        cls.skip_bsejac = False


class TwoPeakLLHNoExog(GenericLikelihoodModel):
    """Fit height of signal peak over background."""
    start_params = [10, 1000]
    cloneattr = ['start_params', 'signal', 'background']
    exog_names = ['n_signal', 'n_background']
    endog_names = ['alpha']

    def __init__(self, endog, exog=None, signal=None, background=None,
                 *args, **kwargs):
        # assume we know the shape + location of the two components,
        # so we re-use their PDFs here
        self.signal = signal
        self.background = background
        super(TwoPeakLLHNoExog, self).__init__(
            endog=endog,
            exog=exog,
            *args,
            extra_params_names=self.exog_names,
            **kwargs
            )

    def loglike(self, params):        # pylint: disable=E0202
        return -self.nloglike(params)

    def nloglike(self, params):
        endog = self.endog
        return self.nlnlike(params, endog)

    def nlnlike(self, params, endog):
        n_sig = params[0]
        n_bkg = params[1]
        if (n_sig < 0) or n_bkg < 0:
            return np.inf
        n_tot = n_bkg + n_sig
        alpha = endog
        sig = self.signal.pdf(alpha)
        bkg = self.background.pdf(alpha)
        sumlogl = np.sum(np.log((n_sig * sig) + (n_bkg * bkg)))
        sumlogl -= n_tot
        return -sumlogl


class TestTwoPeakLLHNoExog:

    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        pdf_a = stats.halfcauchy(loc=0, scale=1)
        pdf_b = stats.uniform(loc=0, scale=100)

        n_a = 50
        n_b = 200
        params = [n_a, n_b]

        X = np.concatenate([pdf_a.rvs(size=n_a),
                            pdf_b.rvs(size=n_b),
                            ])[:, np.newaxis]
        cls.X = X
        cls.params = params
        cls.pdf_a = pdf_a
        cls.pdf_b = pdf_b

    def test_fit(self):
        np.random.seed(42)
        llh_noexog = TwoPeakLLHNoExog(self.X,
                                      signal=self.pdf_a,
                                      background=self.pdf_b)

        res = llh_noexog.fit()
        assert_allclose(res.params, self.params, rtol=1e-1)
        # TODO: nan if exog is None,
        assert res.df_resid == 248
        assert res.df_model == 0
        res_bs = res.bootstrap(nrep=50)
        assert_allclose(res_bs[2].mean(0), self.params, rtol=1e-1)
        # SMOKE test,
        res.summary()
