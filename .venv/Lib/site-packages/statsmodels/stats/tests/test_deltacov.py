# -*- coding: utf-8 -*-
"""Unit tests for NonlinearDeltaCov and LikelihoodResults._get_wald_nonlinear
Created on Sun Mar 01 01:05:35 2015

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._delta_method import NonlinearDeltaCov


class TestDeltacovOLS:

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 100, 4
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        y = x[:, :-1].sum(1) + np.random.randn(nobs)
        cls.res = OLS(y, x).fit()

    def test_method(self):
        # test Results.method is same as calling function/class
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params)**2

        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        nlm = res._get_wald_nonlinear(fun)
        # margeff excludes constant, last parameter in this case
        assert_allclose(nlm.se_vectorized(), nlm.se_vectorized(), rtol=1e-12)
        assert_allclose(nlm.predicted(), nlm.predicted(), rtol=1e-12)
        df = res.df_resid
        t1 = nl.summary(use_t=True, df=df)
        t2 = nlm.summary(use_t=True, df=df)
        assert_equal(str(t1), str(t2))

    def test_ttest(self):
        # check with linear function against t_test
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params)

        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        predicted = nl.predicted()
        se = nl.se_vectorized()
        assert_allclose(predicted, fun(res.params), rtol=1e-12)
        assert_allclose(se, np.sqrt(np.diag(nl.cov())), rtol=1e-12)

        tt = res.t_test(x, use_t=False)
        assert_allclose(predicted, tt.effect, rtol=1e-12)
        assert_allclose(se, tt.sd, rtol=1e-12)
        assert_allclose(nl.conf_int(), tt.conf_int(), rtol=1e-12)
        t1 = nl.summary()
        t2 = tt.summary()
        # equal because nl.summary uses also ContrastResults
        assert_equal(str(t1), str(t2))

        # use_t = True
        predicted = nl.predicted()
        se = nl.se_vectorized()

        df = res.df_resid
        tt = res.t_test(x, use_t=True)
        assert_allclose(nl.conf_int(use_t=True, df=df), tt.conf_int(),
                        rtol=1e-12, atol=1e-10)
        t1 = nl.summary(use_t=True, df=df)
        t2 = tt.summary()
        # equal because nl.summary uses also ContrastResults
        assert_equal(str(t1), str(t2))

    def test_diff(self):
        res = self.res
        x = res.model.exog

        def fun(params):
            return np.dot(x, params) - np.dot(x[:, 1:], params[1:])

        nl = NonlinearDeltaCov(fun, res.params, res.cov_params())
        # the following two use broadcasting
        assert_allclose(nl.predicted(), res.params[0], rtol=1e-12)
        assert_allclose(nl.se_vectorized(), res.bse[0], rtol=1e-12)


def test_deltacov_margeff():
    # compare with discrete margins
    import statsmodels.discrete.tests.test_discrete as dt
    tc = dt.TestPoissonNewton()
    tc.setup_class()
    res_poi = tc.res1
    res_poi.model._derivative_exog

    # 2d f doesn't work correctly,
    # se_vectorized and predicted are 2d column vector

    def f(p):
        ex = res_poi.model.exog.mean(0)[None, :]
        fv = res_poi.model._derivative_exog(p, ex)
        return np.squeeze(fv)

    nlp = NonlinearDeltaCov(f, res_poi.params, res_poi.cov_params())

    marg = res_poi.get_margeff(at='mean')
    # margeff excludes constant, last parameter in this case
    assert_allclose(nlp.se_vectorized()[:-1], marg.margeff_se, rtol=1e-13)
    assert_allclose(nlp.predicted()[:-1], marg.margeff, rtol=1e-13)

    nlpm = res_poi._get_wald_nonlinear(f)
    # margeff excludes constant, last parameter in this case
    assert_allclose(nlpm.se_vectorized()[:-1], marg.margeff_se, rtol=1e-13)
    assert_allclose(nlpm.predicted()[:-1], marg.margeff, rtol=1e-13)
