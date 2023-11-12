# -*- coding: utf-8 -*-
# STATA adds a constant no matter if you want to or not,
# so I cannot test for having no intercept. This also would make
# no sense for Oaxaca. All of these stata_results
# are from using the oaxaca command in STATA.
# Variance from STATA is bootstrapped. Sometimes STATA
# does not converge correctly, so mulitple iterations
# must be done.

import numpy as np

from statsmodels.datasets.ccard.data import load_pandas
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.tools.tools import add_constant

pandas_df = load_pandas()
endog = pandas_df.endog.values
exog = add_constant(pandas_df.exog.values, prepend=False)
pd_endog, pd_exog = pandas_df.endog, add_constant(
    pandas_df.exog, prepend=False
)


class TestOaxaca:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(endog, exog, 3)

    def test_results(self):
        np.random.seed(0)
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        stata_results_std = np.array([653.10389, 64.584796, 655.0323717])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        endow_var, coef_var, inter_var = self.model.three_fold(True).std
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)

        np.testing.assert_almost_equal(endow_var, stata_results_std[0], 3)
        np.testing.assert_almost_equal(coef_var, stata_results_std[1], 3)
        np.testing.assert_almost_equal(inter_var, stata_results_std[2], 3)


class TestOaxacaNoSwap:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(endog, exog, 3, swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandas:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(pd_endog, pd_exog, "OWNRENT")

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoSwap:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(pd_endog, pd_exog, "OWNRENT", swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaNoConstPassed:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(
            pandas_df.endog.values, pandas_df.exog.values, 3, hasconst=False
        )

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaNoSwapNoConstPassed:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(
            pandas_df.endog.values,
            pandas_df.exog.values,
            3,
            hasconst=False,
            swap=False,
        )

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoConstPassed:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(
            pandas_df.endog, pandas_df.exog, "OWNRENT", hasconst=False
        )

    def test_results(self):
        stata_results = np.array([158.7504, 321.7482, 75.45371, -238.4515])
        stata_results_pooled = np.array([158.7504, 130.8095, 27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOaxacaPandasNoSwapNoConstPassed:
    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(
            pandas_df.endog,
            pandas_df.exog,
            "OWNRENT",
            hasconst=False,
            swap=False,
        )

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)

        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)


class TestOneModel:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.one_model = OaxacaBlinder(
            pandas_df.endog, pandas_df.exog, "OWNRENT", hasconst=False
        ).two_fold(True, two_fold_type="self_submitted", submitted_weight=1)

    def test_results(self):
        unexp, exp, gap = self.one_model.params
        unexp_std, exp_std = self.one_model.std
        one_params_stata_results = np.array([75.45370, 83.29673, 158.75044])
        one_std_stata_results = np.array([64.58479, 71.05619])

        np.testing.assert_almost_equal(unexp, one_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, one_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, one_params_stata_results[2], 3)

        np.testing.assert_almost_equal(unexp_std, one_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, one_std_stata_results[1], 3)


class TestZeroModel:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.zero_model = OaxacaBlinder(
            pandas_df.endog, pandas_df.exog, "OWNRENT", hasconst=False
        ).two_fold(True, two_fold_type="self_submitted", submitted_weight=0)

    def test_results(self):
        unexp, exp, gap = self.zero_model.params
        unexp_std, exp_std = self.zero_model.std
        zero_params_stata_results = np.array([-162.9978, 321.7482, 158.75044])
        zero_std_stata_results = np.array([668.1512, 653.10389])

        np.testing.assert_almost_equal(unexp, zero_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, zero_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, zero_params_stata_results[2], 3)

        np.testing.assert_almost_equal(unexp_std, zero_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, zero_std_stata_results[1], 3)


class TestOmegaModel:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.omega_model = OaxacaBlinder(
            pandas_df.endog, pandas_df.exog, "OWNRENT", hasconst=False
        ).two_fold(True, two_fold_type="nuemark")

    def test_results(self):
        unexp, exp, gap = self.omega_model.params
        unexp_std, exp_std = self.omega_model.std
        nue_params_stata_results = np.array([19.52467, 139.22577, 158.75044])
        nue_std_stata_results = np.array([59.82744, 48.25425])

        np.testing.assert_almost_equal(unexp, nue_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, nue_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, nue_params_stata_results[2], 3)

        np.testing.assert_almost_equal(unexp_std, nue_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, nue_std_stata_results[1], 3)


class TestPooledModel:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.pooled_model = OaxacaBlinder(
            pandas_df.endog, pandas_df.exog, "OWNRENT", hasconst=False
        ).two_fold(True)

    def test_results(self):
        unexp, exp, gap = self.pooled_model.params
        unexp_std, exp_std = self.pooled_model.std
        pool_params_stata_results = np.array(
            [27.940908, 130.809536, 158.75044]
        )
        pool_std_stata_results = np.array([89.209487, 58.612367])

        np.testing.assert_almost_equal(unexp, pool_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, pool_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, pool_params_stata_results[2], 3)

        np.testing.assert_almost_equal(unexp_std, pool_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, pool_std_stata_results[1], 3)
