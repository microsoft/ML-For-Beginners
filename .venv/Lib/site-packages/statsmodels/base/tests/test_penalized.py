# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:39:33 2015

Author: Josef Perktold
License: BSD-3
"""

import warnings

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen


class PoissonPenalized(PenalizedMixin, Poisson):
    pass


class LogitPenalized(PenalizedMixin, Logit):
    pass


class ProbitPenalized(PenalizedMixin, Probit):
    pass


class GLMPenalized(PenalizedMixin, GLM):
    pass


class CheckPenalizedPoisson:

    @classmethod
    def setup_class(cls):
        # simulate data
        np.random.seed(987865)

        nobs, k_vars = 500, 10
        k_nonzero = 4
        x = ((np.random.rand(nobs, k_vars) +
              0.5* (np.random.rand(nobs, 1) - 0.5)) * 2 - 1)
        x *= 1.2
        x[:, 0] = 1
        beta = np.zeros(k_vars)
        beta[:k_nonzero] = 1. / np.arange(1, k_nonzero + 1)
        linpred = x.dot(beta)
        y = cls._generate_endog(linpred)

        cls.k_nonzero = k_nonzero
        cls.x = x
        cls.y = y

        # defaults to be overwritten by subclasses
        cls.rtol = 1e-4
        cls.atol = 1e-6
        cls.exog_index = slice(None, None, None)
        cls.k_params = k_vars
        cls.skip_hessian = False  # can be overwritten in _initialize
        cls.penalty = smpen.SCADSmoothed(0.1, c0=0.0001)  # default for tests
        cls._initialize()

    @classmethod
    def _generate_endog(cls, linpred):
        mu = np.exp(linpred)
        np.random.seed(999)
        y = np.random.poisson(mu)
        return y

    def test_params_table(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params[self.exog_index], res2.params,
                        rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.bse[self.exog_index], res2.bse, rtol=self.rtol,
                        atol=self.atol)
        with warnings.catch_warnings():
            # silence scipy distribution warnigns becaus of zero bse
            warnings.simplefilter('ignore', RuntimeWarning)
            assert_allclose(res1.pvalues[self.exog_index], res2.pvalues,
                            rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.predict(), res2.predict(), rtol=0.05)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        summ = self.res1.summary2()
        assert isinstance(summ.as_latex(), str)
        assert isinstance(summ.as_html(), str)
        assert isinstance(summ.as_text(), str)

    def test_numdiff(self):
        res1 = self.res1

        # avoid checking score at MLE, score close to zero
        p = res1.params * 0.98
        # GLM concentrates out scale which affects derivatives, see #4616
        kwds = {'scale': 1} if isinstance(res1.model, GLM) else {}

        assert_allclose(res1.model.score(p, **kwds)[self.exog_index],
                        res1.model.score_numdiff(p, **kwds)[self.exog_index],
                        rtol=0.025)

        if not self.skip_hessian:
            if isinstance(self.exog_index, slice):
                idx1 = idx2 = self.exog_index
            else:
                idx1 = self.exog_index[:, None]
                idx2 = self.exog_index

            h1 = res1.model.hessian(res1.params, **kwds)[idx1, idx2]
            h2 = res1.model.hessian_numdiff(res1.params, **kwds)[idx1, idx2]
            assert_allclose(h1, h2, rtol=0.02)


class TestPenalizedPoissonNonePenal(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x

        modp = Poisson(y, x)
        cls.res2 = modp.fit(disp=0)

        mod = PoissonPenalized(y, x)  # default no penalty
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.atol = 5e-6


class TestPenalizedPoissonNoPenal(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x

        modp = Poisson(y, x)
        cls.res2 = modp.fit(disp=0)

        mod = PoissonPenalized(y, x)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.atol = 5e-6


class TestPenalizedGLMPoissonNoPenal(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x

        modp = GLM(y, x, family=family.Poisson())
        cls.res2 = modp.fit()

        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.atol = 5e-6


class TestPenalizedPoissonOracle(CheckPenalizedPoisson):
    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(disp=0)

        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedGLMPoissonOracle(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Poisson())
        cls.res2 = modp.fit()

        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5  # same as discrete Poisson
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedPoissonOracleHC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        cov_type = 'HC0'
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3

    def test_cov_type(self):
        res1 = self.res1
        res2 = self.res2

        assert_equal(self.res1.cov_type, 'HC0')
        cov_kwds = {'description': 'Standard Errors are heteroscedasticity '
                    'robust (HC0)',
                    'adjust_df': False, 'use_t': False, 'scaling_factor': None}
        assert_equal(self.res1.cov_kwds, cov_kwds)
        # numbers are regression test using bfgs
        params = np.array([0.96817787574701109, 0.43674374940137434,
                           0.33096260487556745, 0.27415680046693747])
        bse = np.array([0.028126650444581985, 0.033099984564283147,
                        0.033184585514904545, 0.034282504130503301])
        assert_allclose(res2.params[:self.k_nonzero], params, atol=1e-5)
        assert_allclose(res2.bse[:self.k_nonzero], bse, rtol=1e-6)
        assert_allclose(res1.params[:self.k_nonzero], params, atol=self.atol)
        assert_allclose(res1.bse[:self.k_nonzero], bse, rtol=0.02)


class TestPenalizedGLMPoissonOracleHC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Poisson())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5  # same as ddiscrete Poisson
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedPoissonGLMOracleHC(CheckPenalizedPoisson):
    # compare discrete Poisson and GLM-Poisson

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        cov_type = 'HC0'
        modp = PoissonPenalized(y, x, penal=cls.penalty)
        modp.pen_weight *= 1.5  # same as discrete Poisson 1.5
        modp.penal.tau = 0.05
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5  # same as discrete Poisson 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0)

        cls.exog_index = slice(None, None, None)

        cls.atol = 1e-4


class TestPenalizedPoissonOraclePenalized(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)

        mod = PoissonPenalized(y, x, penal=cls.penalty)
        # mod.pen_weight *= 1.5
        # mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-3


class TestPenalizedPoissonOraclePenalized2(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 10  # need to penalize more to get oracle selection
        modp.penal.tau = 0.05
        sp2 = np.array([0.96817921, 0.43673551, 0.33096011, 0.27416614])
        cls.res2 = modp.fit(start_params=sp2 * 0.5, method='bfgs',
                            maxiter=100, disp=0)

        params_notrim = np.array([
            9.68178874e-01, 4.36744981e-01, 3.30965041e-01, 2.74161883e-01,
            -2.58988461e-06, -1.24352640e-06, 4.48584458e-08, -2.46876149e-06,
            -1.02471074e-05, -4.39248098e-06])

        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 10  # need to penalize more to get oracle selection
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(start_params=params_notrim * 0.5,
                           method='bfgs', maxiter=100, trim=True, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-8
        cls.k_params = cls.k_nonzero

    def test_zeros(self):

        # first test for trimmed result
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        # we also set bse to zero, TODO: check fit_regularized
        assert_equal(self.res1.bse[self.k_nonzero:], 0)


class TestPenalizedPoissonOraclePenalized2HC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        cov_type = 'HC0'
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 10  # need to penalize more to get oracle selection
        modp.penal.tau = 0.05
        sp2 = np.array([0.96817921, 0.43673551, 0.33096011, 0.27416614])
        cls.res2 = modp.fit(start_params=sp2 * 0.5, cov_type=cov_type,
                            method='bfgs', maxiter=100, disp=0)

        params_notrim = np.array([
            9.68178874e-01, 4.36744981e-01, 3.30965041e-01, 2.74161883e-01,
            -2.58988461e-06, -1.24352640e-06, 4.48584458e-08, -2.46876149e-06,
            -1.02471074e-05, -4.39248098e-06])

        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 10  # need to penalize more to get oracle selection
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(start_params=params_notrim * 0.5,cov_type=cov_type,
                           method='bfgs', maxiter=100, trim=True, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 1e-12
        cls.k_params = cls.k_nonzero

    def test_cov_type(self):
        res1 = self.res1
        res2 = self.res2

        assert_equal(self.res1.cov_type, 'HC0')
        assert_equal(self.res1.results_constrained.cov_type, 'HC0')
        cov_kwds = {'description': 'Standard Errors are heteroscedasticity '
                                   'robust (HC0)',
                    'adjust_df': False, 'use_t': False, 'scaling_factor': None}
        assert_equal(self.res1.cov_kwds, cov_kwds)
        assert_equal(self.res1.cov_kwds, self.res1.results_constrained.cov_kwds)

        # numbers are regression test using bfgs
        params = np.array([0.96817787574701109, 0.43674374940137434,
                           0.33096260487556745, 0.27415680046693747])
        bse = np.array([0.028126650444581985, 0.033099984564283147,
                        0.033184585514904545, 0.034282504130503301])
        assert_allclose(res2.params[:self.k_nonzero], params, atol=1e-5)
        assert_allclose(res2.bse[:self.k_nonzero], bse, rtol=5e-6)
        assert_allclose(res1.params[:self.k_nonzero], params, atol=1e-5)
        assert_allclose(res1.bse[:self.k_nonzero], bse, rtol=5e-6)


# the following classes are copies of Poisson with model adjustments

class CheckPenalizedLogit(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        mu = 1 / (1 + np.exp(-linpred + linpred.mean() - 0.5))
        np.random.seed(999)
        y = np.random.rand(len(mu)) < mu
        return y


class TestPenalizedLogitNoPenal(CheckPenalizedLogit):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x

        modp = Logit(y, x)
        cls.res2 = modp.fit(disp=0)

        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(disp=0)

        cls.atol = 1e-4  # why not closer ?


class TestPenalizedLogitOracle(CheckPenalizedLogit):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = Logit(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(disp=0)

        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= .5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedGLMLogitOracle(CheckPenalizedLogit):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial())
        cls.res2 = modp.fit(disp=0)

        mod = GLMPenalized(y, x, family=family.Binomial(), penal=cls.penalty)
        mod.pen_weight *= .5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedLogitOraclePenalized(CheckPenalizedLogit):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)

        mod = LogitPenalized(y, x, penal=cls.penalty)
        # mod.pen_weight *= 1.5
        # mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-3


class TestPenalizedLogitOraclePenalized2(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 0.5
        modp.penal.tau = 0.05
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)

        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 0.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=True, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-8
        cls.k_params = cls.k_nonzero

    def test_zeros(self):

        # test for trimmed result
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        # we also set bse to zero
        assert_equal(self.res1.bse[self.k_nonzero:], 0)


# the following classes are copies of Poisson with model adjustments
class CheckPenalizedBinomCount(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        mu = 1 / (1 + np.exp(-linpred + linpred.mean() - 0.5))
        np.random.seed(999)
        n_trials = 5 * np.ones(len(mu), int)
        n_trials[:len(mu)//2] += 5
        y = np.random.binomial(n_trials, mu)
        return np.column_stack((y, n_trials - y))


class TestPenalizedGLMBinomCountNoPenal(CheckPenalizedBinomCount):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        x = x[:, :4]
        offset = -0.25 * np.ones(len(y))  # also check offset
        modp = GLM(y, x, family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(method='bfgs', max_start_irls=100)

        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset,
                           penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', max_start_irls=3, maxiter=100, disp=0,
                           start_params=cls.res2.params*0.9)

        cls.atol = 1e-10
        cls.k_params = 4

    def test_deriv(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.model.score(res2.params * 0.98),
                        res2.model.score(res2.params * 0.98), rtol=1e-10)
        assert_allclose(res1.model.score_obs(res2.params * 0.98),
                        res2.model.score_obs(res2.params * 0.98), rtol=1e-10)


class TestPenalizedGLMBinomCountOracleHC(CheckPenalizedBinomCount):
    # TODO: There are still problems with this case
    # using the standard optimization, I get convergence failures and
    # different estimates depending on details, e.g. small changes in pen_weight
    # most likely convexity fails with SCAD in this case

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        offset = -0.25 * np.ones(len(y))  # also check offset
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial(),
                   offset=offset)
        cls.res2 = modp.fit(cov_type=cov_type, method='newton', maxiter=1000,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset,
                           penal=cls.penalty)
        mod.pen_weight *= 1  # lower than in other cases
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', max_start_irls=0,
                           maxiter=100, disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-3


class TestPenalizedGLMBinomCountOracleHC2(CheckPenalizedBinomCount):
    # TODO: There are still problems with this case, see other class
    # with trimming of small parameters, needs larger trim threshold

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        offset = -0.25 * np.ones(len(y))  # also check offset
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial(),
                   offset=offset)
        cls.res2 = modp.fit(cov_type=cov_type, method='newton', maxiter=1000,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset,
                           penal=cls.penalty)
        mod.pen_weight *= 1  # lower than in other cases
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', max_start_irls=0,
                           maxiter=100, disp=0, trim=0.001)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-3
        cls.k_params = cls.k_nonzero


# the following classes are copies of Poisson with model adjustments
class CheckPenalizedGaussian(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        sig_e = np.sqrt(0.1)
        np.random.seed(999)
        y = linpred + sig_e * np.random.rand(len(linpred))
        return y


class TestPenalizedGLMGaussianOracleHC(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        # adding 10 to avoid strict rtol at predicted values close to zero
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=cls.penalty)
        mod.pen_weight *= 1.5  # same as discrete Poisson
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-6
        cls.rtol = 1e-6


class TestPenalizedGLMGaussianOracleHC2(CheckPenalizedGaussian):
    # with trimming

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        # adding 10 to avoid strict rtol at predicted values close to zero
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=cls.penalty)
        mod.pen_weight *= 1.5  # same as discrete Poisson
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0, trim=True)

        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = cls.k_nonzero
        cls.atol = 1e-5
        cls.rtol = 1e-5


class TestPenalizedGLMGaussianL2(CheckPenalizedGaussian):
    # L2 penalty on redundant exog

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        # adding 10 to avoid strict rtol at predicted values close to zero
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                            disp=0)

        weights = (np.arange(x.shape[1]) >= 4).astype(float)
        mod = GLMPenalized(y, x, family=family.Gaussian(),
                           penal=smpen.L2ConstraintsPenalty(weights=weights))
        # make pen_weight large to force redundant to close to zero
        mod.pen_weight *= 500
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0, trim=False)

        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = x.shape[1]
        cls.atol = 1e-5
        cls.rtol = 1e-5


class TestPenalizedGLMGaussianL2Theil(CheckPenalizedGaussian):
    # L2 penalty on redundant exog

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        # adding 10 to avoid strict rtol at predicted values close to zero
        y = y + 10
        k = x.shape[1]
        cov_type = 'HC0'
        restriction = np.eye(k)[2:]
        modp = TheilGLS(y, x, r_matrix=restriction)
        # the corresponding Theil penweight seems to be 2 * nobs / sigma2_e
        cls.res2 = modp.fit(pen_weight=120.74564413221599 * 1000, use_t=False)

        pen = smpen.L2ConstraintsPenalty(restriction=restriction)
        mod = GLMPenalized(y, x, family=family.Gaussian(),
                           penal=pen)
        # use default weight for GLMPenalized
        mod.pen_weight *= 1
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100,
                           disp=0, trim=False)

        cls.k_nonzero = k
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = x.shape[1]
        cls.atol = 1e-5
        cls.rtol = 1e-5

    def test_params_table(self):
        # override inherited because match is not good except for params and predict
        # The cov_type in GLMPenalized and in TheilGLS are not the same
        # both use sandwiches but TheilGLS sandwich is not HC
        # relative difference in bse up to 7%
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params, res2.params, rtol=self.rtol,
                        atol=self.atol)

        exog_index = slice(None, None, None)
        assert_allclose(res1.bse[exog_index], res2.bse[exog_index],
                        rtol=0.1, atol=self.atol)
        assert_allclose(res1.tvalues[exog_index], res2.tvalues[exog_index],
                        rtol=0.08, atol=5e-3)
        assert_allclose(res1.pvalues[exog_index], res2.pvalues[exog_index],
                        rtol=0.1, atol=5e-3)
        assert_allclose(res1.predict(), res2.predict(), rtol=1e-5)
