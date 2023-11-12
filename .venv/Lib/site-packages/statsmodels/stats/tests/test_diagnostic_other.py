# -*- coding: utf-8 -*-
"""Unit tests for generic score/LM tests and conditional moment tests

Created on Mon Nov 17 08:44:06 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao


class CheckCMT:

    def test_score(self):
        expected = self.results_score
        for msg, actual in self.res_score():
            # not all cases provide all 3 elements,
            # TODO: fix API, returns of functions
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13,
                            err_msg=msg)

    def test_scorehc0(self):
        expected = self.results_hc0
        for msg, actual in self.res_hc0():
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13,
                            err_msg=msg)

    def test_scoreopg(self):
        expected = self.results_opg
        for msg, actual in self.res_opg():
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13,
                            err_msg=msg)


class TestCMTOLS(CheckCMT):

    @classmethod
    def setup_class(cls):

        np.random.seed(864259)

        nobs, k_vars = 100, 4
        sig_e = 1
        x0 = np.random.randn(nobs, k_vars)
        x0[:,0] = 1
        y_true = x0.sum(1)
        y = y_true + sig_e * np.random.randn(nobs)

        x1 = np.random.randn(nobs, 2)
        x = np.column_stack((x0, x1))

        cls.exog_full = x
        cls.exog_add = x1
        cls.res_ols = OLS(y, x0).fit()

        cls.attach_moment_conditions()

        # results from initial run, not reference package (drooped 2 digits)
        cls.results_score = (1.6857659627548, 0.43046770240535, 2)
        cls.results_hc0 = (1.6385932313952, 0.4407415561953, 2)
        cls.results_opg = (1.72226002418488, 0.422684174119544, 2.0)

    @classmethod
    def attach_moment_conditions(cls):
        # TODO: a better structure ?
        res_ols = cls.res_ols
        # assumes x = column_stack([x0, x1])
        x = cls.exog_full
        #x0 = cls.res_ols.model.exog  # not used here
        x1 = cls.exog_add

        nobs, k_constraints = x1.shape

        # TODO: cleanup after initial copy past
        moms_obs = res_ols.resid[:, None] * x
        moms = moms_obs.sum(0)
        cov_moms =  res_ols.mse_resid * x.T.dot(x) #np.linalg.inv(x.T.dot(x))
        cov_moms *=  res_ols.df_resid / nobs

        # weights used for GMM to replicate OLS
        weights = np.linalg.inv(cov_moms)
        # we do not use last two variables
        weights[:, -k_constraints:] = 0
        weights[-k_constraints:, :] = 0

        k_moms = moms.shape[0]
        # TODO: Newey has different versions that all produce the same result
        #       in example
        L = np.eye(k_moms)[-k_constraints:] #.dot(np.linalg.inv(cov_moms))

        moms_deriv = cov_moms[:, :-k_constraints]

        covm = moms_obs.T.dot(moms_obs)

        #attach
        cls.nobs = nobs
        cls.moms = moms
        cls.moms_obs = moms_obs
        cls.cov_moms = cov_moms
        cls.covm = covm
        cls.moms_deriv = moms_deriv
        cls.weights = weights
        cls.L = L

    def res_score(self):
        res_ols = self.res_ols
        nobs = self.nobs
        moms = self.moms
        moms_obs = self.moms_obs
        cov_moms = self.cov_moms
        covm = self.covm
        moms_deriv = self.moms_deriv
        weights = self.weights
        L = self.L
        x = self.exog_full  # for auxiliary regression only

        res_all = []

        # auxiliary regression
        stat = nobs * OLS(res_ols.resid, x).fit().rsquared
        res_all.append(('ols R2', stat))

        stat = moms.dot(np.linalg.solve(cov_moms, moms))
        res_all.append(('score simple', stat))

        tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:],
                              np.linalg.inv(cov_moms), cov_moms)
        res_all.append(('score mle', tres))

        tres = CMTNewey(moms, cov_moms, cov_moms[:,:-2], weights, L).chisquare
        res_all.append(('Newey', tres))

        tres = CMTTauchen(moms[:-2], cov_moms[:-2, :-2], moms[-2:],
                          cov_moms[-2:, :-2], cov_moms).chisquare
        res_all.append(('Tauchen', tres))

        return res_all

    def res_opg(self):
        res_ols = self.res_ols
        nobs = self.nobs
        moms = self.moms
        moms_obs = self.moms_obs
        covm = self.covm
        moms_deriv = self.moms_deriv
        weights = self.weights
        L = self.L
        x = self.exog_full

        res_ols2_hc0 = OLS(res_ols.model.endog, x).fit(cov_type='HC0')

        res_all = []

        # auxiliary regression
        ones = np.ones(nobs)
        stat = nobs * OLS(ones, moms_obs).fit().rsquared
        res_all.append(('ols R2', stat))

        tres = res_ols2_hc0.compare_lm_test(res_ols, demean=False)
        res_all.append(('comp_lm uc', tres))

        tres = CMTNewey(moms, covm, covm[:,:-2], weights, L).chisquare
        res_all.append(('Newey', tres))

        tres = CMTTauchen(moms[:-2], covm[:-2, :-2], moms[-2:], covm[-2:, :-2],
                          covm).chisquare
        res_all.append(('Tauchen', tres))

        tres = diao.lm_robust_subset(moms[-2:], 2, covm, covm)
        res_all.append(('score subset QMLE', tres))

        tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:],
                              np.linalg.inv(covm), covm, cov_params=None)
        res_all.append(('scoreB QMLE', tres))

        tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:],
                              np.linalg.inv(covm), None,
                              cov_params=np.linalg.inv(covm))
        res_all.append(('scoreV QMLE', tres))

        return res_all

    def res_hc0(self):
        res_ols = self.res_ols
        nobs = self.nobs
        moms = self.moms
        moms_obs = self.moms_obs
        cov_moms = self.cov_moms   # Hessian with scale
        covm = self.covm
        moms_deriv = self.moms_deriv
        weights = self.weights
        L = self.L

        x0 = res_ols.model.exog
        x1 = self.exog_add

        res_all = []

        tres = diao.cm_test_robust(resid=res_ols.resid, resid_deriv=x0,
                                   instruments=x1, weights=1)
        # TODO: extra return and no df in cm_test_robust Wooldridge
        res_all.append(('Wooldridge', tres[:2]))

        tres = CMTNewey(moms, covm, moms_deriv, weights, L).chisquare
        res_all.append(('Newey', tres))

        tres = CMTTauchen(moms[:-2], cov_moms[:-2, :-2], moms[-2:],
                          cov_moms[-2:, :-2], covm).chisquare
        res_all.append(('Tauchen', tres))

        tres = diao.lm_robust_subset(moms[-2:], 2, cov_moms, covm)
        res_all.append(('score subset QMLE', tres))

        tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:],
                              np.linalg.inv(cov_moms), covm)
        res_all.append(('scoreB QMLE', tres))

        # need sandwich cov_params V
        Ainv = np.linalg.inv(cov_moms)
        vv = Ainv.dot(covm).dot(Ainv)
        tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:],
                              np.linalg.inv(cov_moms), None,
                              cov_params=vv)
        res_all.append(('scoreV QMLE', tres))

        tres = diao.conditional_moment_test_generic(moms_obs[:, -2:],
                                                    cov_moms[-2:, :-2],
                                                    moms_obs[:,:-2],
                                                    cov_moms[:-2, :-2])
        tres_ = (tres.stat_cmt, tres.pval_cmt)
        res_all.append(('cmt', tres_))

        # using unscaled hessian instead of scaled
        x = self.exog_full
        hess_unscaled = x.T.dot(x)
        tres = diao.conditional_moment_test_generic(moms_obs[:, -2:],
                    hess_unscaled[-2:, :-2], moms_obs[:,:-2],
                    hess_unscaled[:-2, :-2])#, covm)
        tres_ = (tres.stat_cmt, tres.pval_cmt)
        res_all.append(('cmt', tres_))

        score_deriv_uu = cov_moms[:-2, :-2]
        score_deriv_cu = cov_moms[-2:, :-2]
        cov_score_cc = covm[-2:, -2:]
        cov_score_cu = covm[-2:, :-2]
        cov_score_uu = covm[:-2, :-2]
        moms[-2:], 2, cov_moms, covm
        tres = diao.lm_robust_subset_parts(moms[-2:], 2, score_deriv_uu,
                                     score_deriv_cu, cov_score_cc,
                                     cov_score_cu, cov_score_uu)

        res_all.append(('score subset_parts QMLE', tres))

        params_deriv = np.eye(x.shape[1], x.shape[1] - 2)
        #params_deriv[[-2, -1], [-2, -1]] = 0
        score = moms
        score_deriv = cov_moms
        cov_score = covm

        tres = diao.lm_robust_reparameterized(score, params_deriv,
                           score_deriv, cov_score)

        res_all.append(('score reparam QMLE', tres))

        return res_all
