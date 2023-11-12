# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:53:27 2018

Author: Josef Perktold

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd

from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening


class PoissonPenalized(PenalizedMixin, Poisson):
    pass


class LogitPenalized(PenalizedMixin, Logit):
    pass


class GLMPenalized(PenalizedMixin, GLM):
    pass


def _get_poisson_data():
    np.random.seed(987865)

    nobs, k_vars = 100, 500
    k_nonzero = 5
    x = (np.random.rand(nobs, k_vars) +
         1. * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2

    x = (x - x.mean(0)) / x.std(0)
    x[:, 0] = 1
    beta = np.zeros(k_vars)
    idx_nonzero_true = [0, 100, 300, 400, 411]
    beta[idx_nonzero_true] = 1. / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)  # make small coefficients larger
    linpred = x.dot(beta)
    mu = np.exp(linpred)
    y = np.random.poisson(mu)
    return y, x, idx_nonzero_true, beta


def test_poisson_screening():

    np.random.seed(987865)

    y, x, idx_nonzero_true, beta = _get_poisson_data()
    nobs = len(y)

    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true,
                              columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = Poisson(y, xframe_true).fit()
    parameters['oracle'] = res_oracle.params

    mod_initial = PoissonPenalized(y, np.ones(nobs), pen_weight=nobs * 5)

    screener = VariableScreening(mod_initial)
    exog_candidates = x[:, 1:]
    res_screen = screener.screen_exog(exog_candidates, maxiter=10)

    assert_equal(np.sort(res_screen.idx_nonzero), idx_nonzero_true)

    xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
    xnames[0] = 'const'

    # smoke test
    res_screen.results_final.summary(xname=xnames)
    res_screen.results_pen.summary()
    assert_equal(res_screen.results_final.mle_retvals['converged'], True)

    ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
    parameters = parameters.join(ps, how='outer')

    assert_allclose(parameters['oracle'], parameters['final'], atol=5e-6)


def test_screen_iterated():
    np.random.seed(987865)

    nobs, k_nonzero = 100, 5

    x = (np.random.rand(nobs, k_nonzero - 1) +
         1.* (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2
    x = (x - x.mean(0)) / x.std(0)
    x = np.column_stack((np.ones(nobs), x))

    beta = 1. / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)  # make small coefficients larger
    linpred = x.dot(beta)
    mu = np.exp(linpred)
    y = np.random.poisson(mu)

    common = x[:, 1:].sum(1)[:, None]

    x_nonzero = x

    def exog_iterator():
        k_vars = 100

        n_batches = 6
        for i in range(n_batches):
            x = (0.05 * common + np.random.rand(nobs, k_vars) +
                 1.* (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
            x *= 1.2
            if i < k_nonzero - 1:
                # hide a nonezero
                x[:, 10] = x_nonzero[:, i + 1]
            x = (x - x.mean(0)) / x.std(0)
            yield x

    dummy = np.ones(nobs)
    dummy[:nobs // 2] = 0
    exog_keep = np.column_stack((np.ones(nobs), dummy))
    for k in [1, 2]:
        mod_initial = PoissonPenalized(y, exog_keep[:, :k], pen_weight=nobs * 500)
        screener = VariableScreening(mod_initial)
        screener.k_max_add = 30

        final = screener.screen_exog_iterator(exog_iterator())
        names = ['var0_10', 'var1_10', 'var2_10', 'var3_10']
        assert_equal(final.exog_final_names, names)
        idx_full = np.array([[ 0, 10],
                             [ 1, 10],
                             [ 2, 10],
                             [ 3, 10]], dtype=np.int64)
        assert_equal(final.idx_nonzero_batches, idx_full)


def test_glmpoisson_screening():

    y, x, idx_nonzero_true, beta = _get_poisson_data()
    nobs = len(y)

    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true, columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = GLMPenalized(y, xframe_true, family=family.Poisson()).fit()
    parameters['oracle'] = res_oracle.params

    mod_initial = GLMPenalized(y, np.ones(nobs), family=family.Poisson())

    screener = VariableScreening(mod_initial)
    exog_candidates = x[:, 1:]
    res_screen = screener.screen_exog(exog_candidates, maxiter=10)

    assert_equal(np.sort(res_screen.idx_nonzero), idx_nonzero_true)

    xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
    xnames[0] = 'const'

    # smoke test
    res_screen.results_final.summary(xname=xnames)
    res_screen.results_pen.summary()
    assert_equal(res_screen.results_final.mle_retvals['converged'], True)

    ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
    parameters = parameters.join(ps, how='outer')

    assert_allclose(parameters['oracle'], parameters['final'], atol=5e-6)


def _get_logit_data():
    np.random.seed(987865)

    nobs, k_vars = 300, 500
    k_nonzero = 5
    x = (np.random.rand(nobs, k_vars) +
         0.01 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2

    x = (x - x.mean(0)) / x.std(0)
    x[:, 0] = 1
    beta = np.zeros(k_vars)
    idx_nonzero_true = [0, 100, 300, 400, 411]
    beta[idx_nonzero_true] = 1. / np.arange(1, k_nonzero + 1)**0.5
    beta = np.sqrt(beta)  # make small coefficients larger
    linpred = x.dot(beta)
    mu = 1 / (1 + np.exp(-linpred))
    y = (np.random.rand(len(mu)) < mu).astype(int)
    return y, x, idx_nonzero_true, beta


def test_logit_screening():

    y, x, idx_nonzero_true, beta = _get_logit_data()
    nobs = len(y)
    # test uses
    screener_kwds = dict(pen_weight=nobs * 0.7, threshold_trim=1e-3)

    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true,
                              columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = Logit(y, xframe_true).fit()
    parameters['oracle'] = res_oracle.params

    mod_initial = LogitPenalized(y, np.ones(nobs), pen_weight=nobs * 0.5)
    screener = VariableScreening(mod_initial, **screener_kwds)
    screener.k_max_add = 30
    exog_candidates = x[:, 1:]
    res_screen = screener.screen_exog(exog_candidates, maxiter=30)

    # we have extra variables, check index for larger params
    mask = np.abs(res_screen.results_final.params) > 0.1
    assert_equal(np.sort(res_screen.idx_nonzero[mask]), idx_nonzero_true)
    # regression test
    idx_r = np.array([0, 74, 100, 163, 300, 400, 411])
    assert_equal(np.sort(res_screen.idx_nonzero), idx_r)

    xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
    xnames[0] = 'const'

    # smoke test
    res_screen.results_final.summary(xname=xnames)
    res_screen.results_pen.summary()
    assert_equal(res_screen.results_final.mle_retvals['converged'], True)

    ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
    # changed the following to allow for some extra params
    # parameters = parameters.join(ps, how='outer')
    parameters['final'] = ps

    assert_allclose(parameters['oracle'], parameters['final'], atol=0.005)


def test_glmlogit_screening():

    y, x, idx_nonzero_true, beta = _get_logit_data()
    nobs = len(y)

    # test uses
    screener_kwds = dict(pen_weight=nobs * 0.75, threshold_trim=1e-3,
                         ranking_attr='model.score_factor')

    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true,
                              columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = GLMPenalized(y, xframe_true, family=family.Binomial()).fit()
    parameters['oracle'] = res_oracle.params

    #mod_initial = LogitPenalized(y, np.ones(nobs), pen_weight=nobs * 0.5)
    mod_initial = GLMPenalized(y, np.ones(nobs), family=family.Binomial())

    screener = VariableScreening(mod_initial, **screener_kwds)
    screener.k_max_add = 10
    exog_candidates = x[:, 1:]
    res_screen = screener.screen_exog(exog_candidates, maxiter=30)

    res_screen.idx_nonzero

    res_screen.results_final

    xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
    xnames[0] = 'const'

    # smoke test
    res_screen.results_final.summary(xname=xnames)
    res_screen.results_pen.summary()
    assert_equal(res_screen.results_final.mle_retvals['converged'], True)

    ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
    # changed the following to allow for some extra params
    # parameters = parameters.join(ps, how='outer')
    parameters['final'] = ps

    assert_allclose(parameters['oracle'], parameters['final'], atol=0.005)


def _get_gaussian_data():
    np.random.seed(987865)

    nobs, k_vars = 100, 500
    k_nonzero = 5
    x = (np.random.rand(nobs, k_vars) +
         0.01 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2

    x = (x - x.mean(0)) / x.std(0)
    x[:, 0] = 1  # make first column into constant
    beta = np.zeros(k_vars)
    idx_nonzero_true = [0, 1, 300, 400, 411]
    beta[idx_nonzero_true] = 1. / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)  # make small coefficients larger
    linpred = x.dot(beta)
    y = linpred + 0.1 * np.random.randn(len(linpred))

    return y, x, idx_nonzero_true, beta


def test_glmgaussian_screening():

    y, x, idx_nonzero_true, beta = _get_gaussian_data()
    nobs = len(y)
    # demeaning makes constant zero, checks that exog_keep are not trimmed
    y = y - y.mean(0)

    # test uses
    screener_kwds = dict(pen_weight=nobs * 0.75, threshold_trim=1e-3,
                         ranking_attr='model.score_factor')

    xnames_true = ['var%4d' % ii for ii in idx_nonzero_true]
    xnames_true[0] = 'const'
    parameters = pd.DataFrame(beta[idx_nonzero_true], index=xnames_true,
                              columns=['true'])

    xframe_true = pd.DataFrame(x[:, idx_nonzero_true], columns=xnames_true)
    res_oracle = GLMPenalized(y, xframe_true, family=family.Gaussian()).fit()
    parameters['oracle'] = res_oracle.params

    for k_keep in [1, 2]:
        mod_initial = GLMPenalized(y, x[:, :k_keep], family=family.Gaussian())
        screener = VariableScreening(mod_initial, **screener_kwds)
        exog_candidates = x[:, k_keep:]
        res_screen = screener.screen_exog(exog_candidates, maxiter=30)

        assert_equal(np.sort(res_screen.idx_nonzero), idx_nonzero_true)

        xnames = ['var%4d' % ii for ii in res_screen.idx_nonzero]
        xnames[0] = 'const'

        # smoke test
        res_screen.results_final.summary(xname=xnames)
        res_screen.results_pen.summary()
        assert_equal(res_screen.results_final.mle_retvals['converged'], True)

        ps = pd.Series(res_screen.results_final.params, index=xnames, name='final')
        parameters = parameters.join(ps, how='outer')

        assert_allclose(parameters['oracle'], parameters['final'], atol=1e-5)
        # we need to remove 'final' again for next iteration
        del parameters['final']
