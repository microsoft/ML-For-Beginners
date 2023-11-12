'''
Created on Nov. 29, 2022

Author: Josef Perktold
License: BSD-3
'''

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg

from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM

ols = OLS.from_formula
quantreg = QuantReg.from_formula


def mean_func(x):
    """mean function for example"""
    return x + 0.25 * x**2


def std_func(x):
    """standard deviation function for example"""
    return 0.1 * np.exp(2.5 + 0.75 * np.abs(x))


class TestMQuantiles():

    @classmethod
    def setup_class(cls):

        np.random.seed(654123)

        # generate an interesting dataset with heteroscedasticity
        nobs = 200
        x = np.random.uniform(-4, 4, nobs)
        y = mean_func(x) + std_func(x) * np.random.randn(nobs)

        cls.df = pd.DataFrame({'temp': x, 'dens': y})

    def test_ols(self):
        # test expectile at q=0.5 versus OLS

        res_ols = ols('dens ~ temp + I(temp ** 2.0)', self.df).fit(use_t=False)

        y = res_ols.model.endog
        xx = res_ols.model.exog

        mq_norm = norms.MQuantileNorm(0.5, norms.LeastSquares())
        mod_rlm = RLM(y, xx, M=mq_norm)
        res_rlm = mod_rlm.fit()

        assert_allclose(res_rlm.params, res_ols.params, rtol=1e-10)
        assert_allclose(res_rlm.bse, res_ols.bse, rtol=1e-10)
        assert_allclose(res_rlm.pvalues, res_ols.pvalues, rtol=1e-10)

    def test_quantreg(self):
        # test HuberT mquantile versus QuantReg
        # cov_params inference will not agree
        t_eps = 1e-6

        mod1 = quantreg('dens ~ temp + I(temp ** 2.0)', self.df)
        y = mod1.endog
        xx = mod1.exog

        for q in [0.25, 0.75]:
            # Note HuberT does not agree very closely with quantreg for g=0.5

            res1 = mod1.fit(q=q)

            mq_norm = norms.MQuantileNorm(q, norms.HuberT(t=t_eps))
            mod_rlm = RLM(y, xx, M=mq_norm)
            res_rlm = mod_rlm.fit()

            assert_allclose(res_rlm.params, res1.params, rtol=5e-4)
            assert_allclose(res_rlm.fittedvalues, res1.fittedvalues, rtol=1e-3)

        # for q=0.5, we compare with plain HuberT norm
        q = 0.5
        t_eps = 1e-2  # RuntimeWarning an bse are nan 0/0 if t_eps too small
        mod1 = RLM(y, xx, M=norms.HuberT(t=t_eps))
        res1 = mod1.fit()

        mq_norm = norms.MQuantileNorm(q, norms.HuberT(t=t_eps))
        mod_rlm = RLM(y, xx, M=mq_norm)
        res_rlm = mod_rlm.fit()

        assert_allclose(res_rlm.params, res1.params, rtol=1e-10)
        assert_allclose(res_rlm.fittedvalues, res1.fittedvalues, rtol=1e-10)
