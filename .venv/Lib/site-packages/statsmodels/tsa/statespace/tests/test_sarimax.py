"""
Tests for SARIMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
import os
import warnings

from statsmodels.compat.platform import PLATFORM_WIN

import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
    assert_, assert_equal, assert_almost_equal, assert_raises, assert_allclose
)


current_path = os.path.dirname(os.path.abspath(__file__))

realgdp_path = os.path.join('results', 'results_realgdpar_stata.csv')
realgdp_results = pd.read_csv(current_path + os.sep + realgdp_path)

coverage_path = os.path.join('results', 'results_sarimax_coverage.csv')
coverage_results = pd.read_csv(os.path.join(current_path, coverage_path))


class TestSARIMAXStatsmodels:
    """
    Test ARIMA model using SARIMAX class against statsmodels ARIMA class

    Notes
    -----

    Standard errors are quite good for the OPG case.
    """
    @classmethod
    def setup_class(cls):
        cls.true = results_sarimax.wpi1_stationary
        endog = cls.true['data']
        # Old results from statsmodels.arima.ARIMA taken before it was removed
        # to let test continue to run. On old statsmodels, can run
        # result_a = arima.ARIMA(endog, order=(1, 1, 1)).fit(disp=-1)
        result_a = Bunch()
        result_a.llf = -135.3513139733829
        result_a.aic = 278.7026279467658
        result_a.bic = 289.9513653682555
        result_a.hqic = 283.27183681851653
        result_a.params = np.array([0.74982449, 0.87421135, -0.41202195])
        result_a.bse = np.array([0.29207409, 0.06377779, 0.12208469])
        cls.result_a = result_a
        cls.model_b = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='c',
                                      simple_differencing=True,
                                      hamilton_representation=True)
        cls.result_b = cls.model_b.fit(disp=-1)

    def test_loglike(self):
        assert_allclose(self.result_b.llf, self.result_a.llf)

    def test_aic(self):
        assert_allclose(self.result_b.aic, self.result_a.aic)

    def test_bic(self):
        assert_allclose(self.result_b.bic, self.result_a.bic)

    def test_hqic(self):
        assert_allclose(self.result_b.hqic, self.result_a.hqic)

    def test_mle(self):
        # ARIMA estimates the mean of the process, whereas SARIMAX estimates
        # the intercept. Convert the mean to intercept to compare
        params_a = self.result_a.params.copy()
        params_a[0] = (1 - params_a[1]) * params_a[0]
        assert_allclose(self.result_b.params[:-1], params_a, atol=5e-5)

    def test_bse(self):
        # Test the complex step approximated BSE values
        cpa = self.result_b._cov_params_approx(approx_complex_step=True)
        bse = cpa.diagonal()**0.5
        assert_allclose(bse[1:-1], self.result_a.bse[1:], atol=1e-5)

    def test_t_test(self):
        import statsmodels.tools._testing as smt
        # to trigger failure, un-comment the following:
        #  self.result_b._cache['pvalues'] += 1
        smt.check_ttest_tvalues(self.result_b)
        smt.check_ftest_pvalues(self.result_b)


class TestRealGDPARStata:
    """
    Includes tests of filtered states and standardized forecast errors.

    Notes
    -----
    Could also test the usual things like standard errors, etc. but those are
    well-tested elsewhere.
    """
    @classmethod
    def setup_class(cls):
        dlgdp = np.log(realgdp_results['value']).diff()[1:].values
        cls.model = sarimax.SARIMAX(dlgdp, order=(12, 0, 0), trend='n',
                                    hamilton_representation=True)
        # Estimated by Stata
        params = [
            .40725515, .18782621, -.01514009, -.01027267, -.03642297,
            .11576416, .02573029, -.00766572, .13506498, .08649569, .06942822,
            -.10685783, .00007999607
        ]
        cls.results = cls.model.filter(params)

    def test_filtered_state(self):
        for i in range(12):
            assert_allclose(
                realgdp_results.iloc[1:]['u%d' % (i+1)],
                self.results.filter_results.filtered_state[i],
                atol=1e-6
            )

    def test_standardized_forecasts_error(self):
        assert_allclose(
            realgdp_results.iloc[1:]['rstd'],
            self.results.filter_results.standardized_forecasts_error[0],
            atol=1e-3
        )


class SARIMAXStataTests:
    def test_loglike(self):
        assert_almost_equal(
            self.result.llf,
            self.true['loglike'], 4
        )

    def test_aic(self):
        assert_almost_equal(
            self.result.aic,
            self.true['aic'], 3
        )

    def test_bic(self):
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 3
        )

    def test_hqic(self):
        hqic = (
            -2*self.result.llf +
            2*np.log(np.log(self.result.nobs_effective)) *
            self.result.params.shape[0]
        )
        assert_almost_equal(
            self.result.hqic,
            hqic, 3
        )

    def test_standardized_forecasts_error(self):
        cython_sfe = self.result.standardized_forecasts_error
        self.result._standardized_forecasts_error = None
        python_sfe = self.result.standardized_forecasts_error
        assert_allclose(cython_sfe, python_sfe)


class ARIMA(SARIMAXStataTests):
    """
    ARIMA model

    Stata arima documentation, Example 1
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = true['data']

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='c',
                                    *args, **kwargs)

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3
        )


class TestARIMAStationary(ARIMA):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestARIMAStationary, cls).setup_class(
            results_sarimax.wpi1_stationary
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-7)
        assert_allclose(self.result.bse[2], self.true['se_ma_opg'], atol=1e-7)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-7)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-7)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # finite difference, non-centered
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-1)

        #     # finite difference, centered
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[1], self.true['se_ar_oim'], atol=1e-3)
        assert_allclose(oim_bse[2], self.true['se_ma_oim'], atol=1e-2)

    def test_bse_robust(self):
        robust_oim_bse = self.result.cov_params_robust_oim.diagonal()**0.5
        cpra = self.result.cov_params_robust_approx
        robust_approx_bse = cpra.diagonal()**0.5
        true_robust_bse = np.r_[
            self.true['se_ar_robust'], self.true['se_ma_robust']
        ]

        assert_allclose(robust_oim_bse[1:3], true_robust_bse, atol=1e-2)
        assert_allclose(robust_approx_bse[1:3], true_robust_bse, atol=1e-3)


class TestARIMADiffuse(ARIMA):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls, **kwargs):
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = (
            results_sarimax.wpi1_diffuse['initial_variance']
        )
        super(TestARIMADiffuse, cls).setup_class(results_sarimax.wpi1_diffuse,
                                                 **kwargs)

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-7)
        assert_allclose(self.result.bse[2], self.true['se_ma_opg'], atol=1e-7)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered : failure
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

        #     # finite difference, centered : failure
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-1)


class AdditiveSeasonal(SARIMAXStataTests):
    """
    ARIMA model with additive seasonal effects

    Stata arima documentation, Example 2
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, order=(1, 1, (1, 0, 0, 1)), trend='c', *args, **kwargs
        )

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3
        )


class TestAdditiveSeasonal(AdditiveSeasonal):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAdditiveSeasonal, cls).setup_class(
            results_sarimax.wpi1_seasonal
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-6)
        assert_allclose(self.result.bse[2:4], self.true['se_ma_opg'],
                        atol=1e-5)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-4)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-1)


class Airline(SARIMAXStataTests):
    """
    Multiplicative SARIMA model: "Airline" model

    Stata arima documentation, Example 3
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
            trend='n', *args, **kwargs
        )

        params = np.r_[true['params_ma'], true['params_seasonal_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = self.model.fit(disp=-1)
            assert_allclose(
                result.params, self.result.params,
                atol=1e-4
            )


class TestAirlineHamilton(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineHamilton, cls).setup_class(
            results_sarimax.air2_stationary
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'],
                        atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-6)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-2)

        #     # finite difference, centered
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class TestAirlineHarvey(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineHarvey, cls).setup_class(
            results_sarimax.air2_stationary, hamilton_representation=False
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'],
                        atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-6)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-2)

        #     # finite difference, centered
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class TestAirlineStateDifferencing(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineStateDifferencing, cls).setup_class(
            results_sarimax.air2_stationary, simple_differencing=False,
            hamilton_representation=False
        )

    def test_bic(self):
        # Due to diffuse component of the state (which technically changes the
        # BIC calculation - see Durbin and Koopman section 7.4), this is the
        # best we can do for BIC
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 0
        )

    def test_mle(self):
        result = self.model.fit(method='nm', maxiter=1000, disp=0)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3)

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'],
                        atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-4)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered : failure with NaNs
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-2)

        #     # finite difference, centered : failure with NaNs
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'],
        #                     atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class Friedman(SARIMAXStataTests):
    """
    ARMAX model: Friedman quantity theory of money

    Stata arima documentation, Example 4
    """
    @classmethod
    def setup_class(cls, true, exog=None, *args, **kwargs):
        cls.true = true
        endog = np.r_[true['data']['consump']]
        if exog is None:
            exog = add_constant(true['data']['m2'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, exog=exog, order=(1, 0, 1), *args, **kwargs
        )

        params = np.r_[true['params_exog'], true['params_ar'],
                       true['params_ma'], true['params_variance']]

        cls.result = cls.model.filter(params)


class TestFriedmanMLERegression(Friedman):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestFriedmanMLERegression, cls).setup_class(
            results_sarimax.friedman2_mle
        )

    def test_mle(self):
        result = self.model.fit(disp=-1)
        # Use ratio to make atol more meaningful parameter scale differs
        ratio = result.params / self.result.params
        assert_allclose(ratio, np.ones(5), atol=1e-2, rtol=1e-3)

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0:2], self.true['se_exog_opg'],
                        atol=1e-4)
        assert_allclose(self.result.bse[2], self.true['se_ar_opg'], atol=1e-6)
        assert_allclose(self.result.bse[3], self.true['se_ma_opg'], atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0:2], self.true['se_exog_oim'], atol=1e-4)
        assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-6)
        assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        #     assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        #     assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)


class TestFriedmanStateRegression(Friedman):
    """
    Notes
    -----

    MLE is not very close and standard errors are not very close for any set of
    parameters.

    This is likely because we're comparing against the model where the
    regression coefficients are also estimated by MLE. So this test should be
    considered just a very basic "sanity" test.
    """
    @classmethod
    def setup_class(cls):
        # Remove the regression coefficients from the parameters, since they
        # will be estimated as part of the state vector
        true = dict(results_sarimax.friedman2_mle)
        exog = add_constant(true['data']['m2']) / 10.

        true['mle_params_exog'] = true['params_exog'][:]
        true['mle_se_exog'] = true['se_exog_opg'][:]

        true['params_exog'] = []
        true['se_exog'] = []

        super(TestFriedmanStateRegression, cls).setup_class(
            true, exog=exog, mle_regression=False
        )

        cls.true_params = np.r_[true['params_exog'], true['params_ar'],
                                true['params_ma'], true['params_variance']]

        cls.result = cls.model.filter(cls.true_params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-1, rtol=2e-1
        )

    def test_regression_parameters(self):
        # The regression effects are integrated into the state vector as
        # the last two states (thus the index [-2:]). The filtered
        # estimates of the state vector produced by the Kalman filter and
        # stored in `filtered_state` for these state elements give the
        # recursive least squares estimates of the regression coefficients
        # at each time period. To get the estimates conditional on the
        # entire dataset, use the filtered states from the last time
        # period (thus the index [-1]).
        assert_almost_equal(
            self.result.filter_results.filtered_state[-2:, -1] / 10.,
            self.true['mle_params_exog'], 1
        )

    # Loglikelihood (and so aic, bic) is slightly different when states are
    # integrated into the state vector
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ar_opg'], atol=1e-2)
        assert_allclose(self.result.bse[1], self.true['se_ma_opg'], atol=1e-2)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-1)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered :
        #     #  failure (catastrophic cancellation)
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered : failure (nan)
        #     cpa = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True)
        #     bse = cpa.diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-1)


class TestFriedmanPredict(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, prediction

    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This follows the given Stata example, although it is not truly forecasting
    because it compares using the actual data (which is available in the
    example but just not used in the parameter MLE estimation) against dynamic
    prediction of that data. Here `test_predict` matches the first case, and
    `test_dynamic_predict` matches the second.
    """
    @classmethod
    def setup_class(cls):
        super(TestFriedmanPredict, cls).setup_class(
            results_sarimax.friedman2_predict
        )

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_predict(self):
        assert_almost_equal(
            self.result.predict(),
            self.true['predict'], 3
        )

    def test_dynamic_predict(self):
        dynamic = len(self.true['data']['consump'])-15-1
        assert_almost_equal(
            self.result.predict(dynamic=dynamic),
            self.true['dynamic_predict'], 3
        )


class TestFriedmanForecast(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, forecasts

    Variation on:
    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This is a variation of the Stata example, in which the endogenous data is
    actually made to be missing so that the predict command must forecast.

    As another unit test, we also compare against the case in State when
    predict is used against missing data (so forecasting) with the dynamic
    option also included. Note, however, that forecasting in State space models
    amounts to running the Kalman filter against missing datapoints, so it is
    not clear whether "dynamic" forecasting (where instead of missing
    datapoints for lags, we plug in previous forecasted endog values) is
    meaningful.
    """
    @classmethod
    def setup_class(cls):
        true = dict(results_sarimax.friedman2_predict)

        true['forecast_data'] = {
            'consump': true['data']['consump'][-15:],
            'm2': true['data']['m2'][-15:]
        }
        true['data'] = {
            'consump': true['data']['consump'][:-15],
            'm2': true['data']['m2'][:-15]
        }

        super(TestFriedmanForecast, cls).setup_class(true)

        cls.result = cls.model.filter(cls.result.params)

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, exog=exog),
            self.true['forecast'], 3
        )

    def test_dynamic_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        dynamic = len(self.true['data']['consump'])-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, dynamic=dynamic, exog=exog),
            self.true['dynamic_forecast'], 3
        )


class SARIMAXCoverageTest:
    @classmethod
    def setup_class(cls, i, decimal=4, endog=None, *args, **kwargs):
        # Dataset
        if endog is None:
            endog = results_sarimax.wpi1_data

        # Loglikelihood, parameters
        cls.true_loglike = coverage_results.loc[i]['llf']
        cls.true_params = np.array([
            float(x) for x in coverage_results.loc[i]['parameters'].split(',')]
        )
        # Stata reports the standard deviation; make it the variance
        cls.true_params[-1] = cls.true_params[-1]**2

        # Test parameters
        cls.decimal = decimal

        # Compare using the Hamilton representation and simple differencing
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(endog, *args, **kwargs)

    def test_loglike(self):
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.llf,
            self.true_loglike,
            atol=0.7 * 10**(-self.decimal)
        )

    def test_start_params(self):
        # just a quick test that start_params is not throwing an exception
        # (other than related to invertibility)
        stat = self.model.enforce_stationarity
        inv = self.model.enforce_invertibility
        self.model.enforce_stationarity = False
        self.model.enforce_invertibility = False
        self.model.start_params
        self.model.enforce_stationarity = stat
        self.model.enforce_invertibility = inv

    def test_transform_untransform(self):
        model = self.model
        stat, inv = model.enforce_stationarity, model.enforce_invertibility
        true_constrained = self.true_params

        # Sometimes the parameters given by Stata are not stationary and / or
        # invertible, so we need to skip those transformations for those
        # parameter sets
        model.update(self.true_params)

        par = model.polynomial_ar
        psar = model.polynomial_seasonal_ar
        contracted_psar = psar[psar.nonzero()]
        model.enforce_stationarity = (
            (model.k_ar == 0 or tools.is_invertible(np.r_[1, -par[1:]])) and
            (len(contracted_psar) <= 1 or
                tools.is_invertible(np.r_[1, -contracted_psar[1:]]))
        )

        pma = model.polynomial_ma
        psma = model.polynomial_seasonal_ma
        contracted_psma = psma[psma.nonzero()]
        model.enforce_invertibility = (
            (model.k_ma == 0 or tools.is_invertible(np.r_[1, pma[1:]])) and
            (len(contracted_psma) <= 1 or
                tools.is_invertible(np.r_[1, contracted_psma[1:]]))
        )

        unconstrained = model.untransform_params(true_constrained)
        constrained = model.transform_params(unconstrained)

        assert_almost_equal(constrained, true_constrained, 4)
        model.enforce_stationarity = stat
        model.enforce_invertibility = inv

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg
        self.result.cov_params_robust_oim
        self.result.cov_params_robust_approx

    @pytest.mark.matplotlib
    def test_plot_diagnostics(self, close_figures):
        # Make sure that no exceptions are thrown during plot_diagnostics
        self.result = self.model.filter(self.true_params)
        self.result.plot_diagnostics()

    def test_predict(self):
        result = self.model.filter(self.true_params)
        # Test predict does not throw exceptions, and produces the right shaped
        # output
        predict = result.predict()
        assert_equal(predict.shape, (self.model.nobs,))

        predict = result.predict(start=10, end=20)
        assert_equal(predict.shape, (11,))

        predict = result.predict(start=10, end=20, dynamic=10)
        assert_equal(predict.shape, (11,))

        # Test forecasts
        if self.model.k_exog == 0:
            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10)
            assert_equal(predict.shape, (11,))

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10)

            forecast = result.forecast()
            assert_equal(forecast.shape, (1,))

            forecast = result.forecast(10)
            assert_equal(forecast.shape, (10,))
        else:
            k_exog = self.model.k_exog
            exog = np.r_[[0]*k_exog*11].reshape(11, k_exog)

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10,
                                     exog=exog)
            assert_equal(predict.shape, (11,))

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10,
                                     exog=exog)

            exog = np.r_[[0]*k_exog].reshape(1, k_exog)
            forecast = result.forecast(exog=exog)
            assert_equal(forecast.shape, (1,))

    def test_init_keys_replicate(self):
        mod1 = self.model

        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog

        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-6 if PLATFORM_WIN else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)


class Test_ar(SARIMAXCoverageTest):
    # // AR: (p, 0, 0) x (0, 0, 0, 0)
    # arima wpi, arima(3, 0, 0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        super(Test_ar, cls).setup_class(0, *args, **kwargs)


class Test_ar_as_polynomial(SARIMAXCoverageTest):
    # // AR: (p, 0, 0) x (0, 0, 0, 0)
    # arima wpi, arima(3, 0, 0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = ([1, 1, 1], 0, 0)
        super(Test_ar_as_polynomial, cls).setup_class(0, *args, **kwargs)


class Test_ar_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(3, 0, 0) noconstant vce(oim)
    # save_results 2
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['trend'] = 'c'
        super(Test_ar_trend_c, cls).setup_class(1, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[0] = (1 - tps[1:4].sum()) * tps[0]


class Test_ar_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(3, 0, 0) noconstant vce(oim)
    # save_results 3
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['trend'] = 'ct'
        super(Test_ar_trend_ct, cls).setup_class(2, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_ar_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, arima(3, 0, 0) noconstant vce(oim)
    # save_results 4
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['trend'] = [1, 0, 0, 1]
        super(Test_ar_trend_polynomial, cls).setup_class(3, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_ar_diff(SARIMAXCoverageTest):
    # // AR and I(d): (p, d, 0) x (0, 0, 0, 0)
    # arima wpi, arima(3, 2, 0) noconstant vce(oim)
    # save_results 5
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 0)
        super(Test_ar_diff, cls).setup_class(4, *args, **kwargs)


class Test_ar_seasonal_diff(SARIMAXCoverageTest):
    # // AR and I(D): (p, 0, 0) x (0, D, 0, s)
    # arima wpi, arima(3, 0, 0) sarima(0, 2, 0, 4) noconstant vce(oim)
    # save_results 6
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['seasonal_order'] = (0, 2, 0, 4)
        super(Test_ar_seasonal_diff, cls).setup_class(5, *args, **kwargs)


class Test_ar_diffuse(SARIMAXCoverageTest):
    # // AR and diffuse initialization
    # arima wpi, arima(3, 0, 0) noconstant vce(oim) diffuse
    # save_results 7
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_ar_diffuse, cls).setup_class(6, *args, **kwargs)


class Test_ar_no_enforce(SARIMAXCoverageTest):
    # // AR: (p, 0, 0) x (0, 0, 0, 0)
    # arima wpi, arima(3, 0, 0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['enforce_stationarity'] = False
        kwargs['enforce_invertibility'] = False
        kwargs['initial_variance'] = 1e9
        kwargs['loglikelihood_burn'] = 0
        super(Test_ar_no_enforce, cls).setup_class(6, *args, **kwargs)
        # Reset loglikelihood burn, which gets automatically set to the number
        # of states if enforce_stationarity = False
        cls.model.ssm.loglikelihood_burn = 0

    def test_init_keys_replicate(self):
        mod1 = self.model

        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog

        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        # Fixes needed for edge case model
        model2.ssm.initialization = mod1.ssm.initialization

        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-6 if PLATFORM_WIN else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)


class Test_ar_exogenous(SARIMAXCoverageTest):
    # // ARX
    # arima wpi x, arima(3, 0, 0) noconstant vce(oim)
    # save_results 8
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_ar_exogenous, cls).setup_class(7, *args, **kwargs)


class Test_ar_exogenous_in_state(SARIMAXCoverageTest):
    # // ARX
    # arima wpi x, arima(3, 0, 0) noconstant vce(oim)
    # save_results 8
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['mle_regression'] = False
        super(Test_ar_exogenous_in_state, cls).setup_class(7, *args, **kwargs)
        cls.true_regression_coefficient = cls.true_params[0]
        cls.true_params = cls.true_params[1:]

    def test_loglike(self):
        # Regression in the state vector gives a different loglikelihood, so
        # just check that it's approximately the same
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.llf,
            self.true_loglike,
            atol=2
        )

    def test_regression_coefficient(self):
        # Test that the regression coefficient (estimated as the last filtered
        # state estimate for the regression state) is the same as the Stata
        # MLE state
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.filter_results.filtered_state[3][-1],
            self.true_regression_coefficient,
            self.decimal
        )


class Test_ma(SARIMAXCoverageTest):
    # // MA: (0, 0, q) x (0, 0, 0, 0)
    # arima wpi, arima(0, 0, 3) noconstant vce(oim)
    # save_results 9
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        super(Test_ma, cls).setup_class(8, *args, **kwargs)


class Test_ma_as_polynomial(SARIMAXCoverageTest):
    # // MA: (0, 0, q) x (0, 0, 0, 0)
    # arima wpi, arima(0, 0, 3) noconstant vce(oim)
    # save_results 9
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, [1, 1, 1])
        super(Test_ma_as_polynomial, cls).setup_class(8, *args, **kwargs)


class Test_ma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(0, 0, 3) noconstant vce(oim)
    # save_results 10
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        kwargs['trend'] = 'c'
        super(Test_ma_trend_c, cls).setup_class(9, *args, **kwargs)


class Test_ma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(0, 0, 3) noconstant vce(oim)
    # save_results 11
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        kwargs['trend'] = 'ct'
        super(Test_ma_trend_ct, cls).setup_class(10, *args, **kwargs)


class Test_ma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, arima(0, 0, 3) noconstant vce(oim)
    # save_results 12
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        kwargs['trend'] = [1, 0, 0, 1]
        super(Test_ma_trend_polynomial, cls).setup_class(11, *args, **kwargs)


class Test_ma_diff(SARIMAXCoverageTest):
    # // MA and I(d): (0, d, q) x (0, 0, 0, 0)
    # arima wpi, arima(0, 2, 3) noconstant vce(oim)
    # save_results 13
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 3)
        super(Test_ma_diff, cls).setup_class(12, *args, **kwargs)


class Test_ma_seasonal_diff(SARIMAXCoverageTest):
    # // MA and I(D): (p, 0, 0) x (0, D, 0, s)
    # arima wpi, arima(0, 0, 3) sarima(0, 2, 0, 4) noconstant vce(oim)
    # save_results 14
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        kwargs['seasonal_order'] = (0, 2, 0, 4)
        super(Test_ma_seasonal_diff, cls).setup_class(13, *args, **kwargs)


class Test_ma_diffuse(SARIMAXCoverageTest):
    # // MA and diffuse initialization
    # arima wpi, arima(0, 0, 3) noconstant vce(oim) diffuse
    # save_results 15
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_ma_diffuse, cls).setup_class(14, *args, **kwargs)


class Test_ma_exogenous(SARIMAXCoverageTest):
    # // MAX
    # arima wpi x, arima(0, 0, 3) noconstant vce(oim)
    # save_results 16
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 3)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_ma_exogenous, cls).setup_class(15, *args, **kwargs)


class Test_arma(SARIMAXCoverageTest):
    # // ARMA: (p, 0, q) x (0, 0, 0, 0)
    # arima wpi, arima(3, 0, 3) noconstant vce(oim)
    # save_results 17
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 3)
        super(Test_arma, cls).setup_class(16, *args, **kwargs)


class Test_arma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(3, 0, 2) noconstant vce(oim)
    # save_results 18
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        kwargs['trend'] = 'c'
        super(Test_arma_trend_c, cls).setup_class(17, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:1] = (1 - tps[1:4].sum()) * tps[:1]


class Test_arma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(3, 0, 2) noconstant vce(oim)
    # save_results 19
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        kwargs['trend'] = 'ct'
        super(Test_arma_trend_ct, cls).setup_class(18, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_arma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, arima(3, 0, 2) noconstant vce(oim)
    # save_results 20
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        kwargs['trend'] = [1, 0, 0, 1]
        super(Test_arma_trend_polynomial, cls).setup_class(19, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_arma_diff(SARIMAXCoverageTest):
    # // ARMA and I(d): (p, d, q) x (0, 0, 0, 0)
    # arima wpi, arima(3, 2, 2) noconstant vce(oim)
    # save_results 21
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        super(Test_arma_diff, cls).setup_class(20, *args, **kwargs)


class Test_arma_seasonal_diff(SARIMAXCoverageTest):
    # // ARMA and I(D): (p, 0, q) x (0, D, 0, s)
    # arima wpi, arima(3, 0, 2) sarima(0, 2, 0, 4) noconstant vce(oim)
    # save_results 22
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        kwargs['seasonal_order'] = (0, 2, 0, 4)
        super(Test_arma_seasonal_diff, cls).setup_class(21, *args, **kwargs)


class Test_arma_diff_seasonal_diff(SARIMAXCoverageTest):
    # // ARMA and I(d) and I(D): (p, d, q) x (0, D, 0, s)
    # arima wpi, arima(3, 2, 2) sarima(0, 2, 0, 4) noconstant vce(oim)
    # save_results 23
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (0, 2, 0, 4)
        super(Test_arma_diff_seasonal_diff, cls).setup_class(
            22, *args, **kwargs)


class Test_arma_diffuse(SARIMAXCoverageTest):
    # // ARMA and diffuse initialization
    # arima wpi, arima(3, 0, 2) noconstant vce(oim) diffuse
    # save_results 24
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_arma_diffuse, cls).setup_class(23, *args, **kwargs)


class Test_arma_exogenous(SARIMAXCoverageTest):
    # // ARMAX
    # arima wpi x, arima(3, 0, 2) noconstant vce(oim)
    # save_results 25
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 2)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_arma_exogenous, cls).setup_class(24, *args, **kwargs)


class Test_seasonal_ar(SARIMAXCoverageTest):
    # // SAR: (0, 0, 0) x (P, 0, 0, s)
    # arima wpi, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 26
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        super(Test_seasonal_ar, cls).setup_class(25, *args, **kwargs)


class Test_seasonal_ar_as_polynomial(SARIMAXCoverageTest):
    # // SAR: (0, 0, 0) x (P, 0, 0, s)
    # arima wpi, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 26
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = ([1, 1, 1], 0, 0, 4)
        super(Test_seasonal_ar_as_polynomial, cls).setup_class(
            25, *args, **kwargs)


class Test_seasonal_ar_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 27
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        kwargs['trend'] = 'c'
        super(Test_seasonal_ar_trend_c, cls).setup_class(26, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:1] = (1 - tps[1:4].sum()) * tps[:1]


class Test_seasonal_ar_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 28
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_ar_trend_ct, cls).setup_class(27, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_seasonal_ar_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 29
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        kwargs['trend'] = [1, 0, 0, 1]
        super(Test_seasonal_ar_trend_polynomial, cls).setup_class(
            28, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_seasonal_ar_diff(SARIMAXCoverageTest):
    # // SAR and I(d): (0, d, 0) x (P, 0, 0, s)
    # arima wpi, arima(0, 2, 0) sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 30
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        super(Test_seasonal_ar_diff, cls).setup_class(29, *args, **kwargs)


class Test_seasonal_ar_seasonal_diff(SARIMAXCoverageTest):
    # // SAR and I(D): (0, 0, 0) x (P, D, 0, s)
    # arima wpi, sarima(3, 2, 0, 4) noconstant vce(oim)
    # save_results 31
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 2, 0, 4)
        super(Test_seasonal_ar_seasonal_diff, cls).setup_class(
            30, *args, **kwargs)


class Test_seasonal_ar_diffuse(SARIMAXCoverageTest):
    # // SAR and diffuse initialization
    # arima wpi, sarima(3, 0, 0, 4) noconstant vce(oim) diffuse
    # save_results 32
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_ar_diffuse, cls).setup_class(31, *args, **kwargs)


class Test_seasonal_ar_exogenous(SARIMAXCoverageTest):
    # // SARX
    # arima wpi x, sarima(3, 0, 0, 4) noconstant vce(oim)
    # save_results 33
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 0, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_ar_exogenous, cls).setup_class(32, *args, **kwargs)


class Test_seasonal_ma(SARIMAXCoverageTest):
    # // SMA
    # arima wpi, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 34
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        super(Test_seasonal_ma, cls).setup_class(33, *args, **kwargs)


class Test_seasonal_ma_as_polynomial(SARIMAXCoverageTest):
    # // SMA
    # arima wpi, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 34
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, [1, 1, 1], 4)
        super(Test_seasonal_ma_as_polynomial, cls).setup_class(
            33, *args, **kwargs)


class Test_seasonal_ma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 35
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        kwargs['trend'] = 'c'
        kwargs['decimal'] = 3
        super(Test_seasonal_ma_trend_c, cls).setup_class(34, *args, **kwargs)


class Test_seasonal_ma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 36
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_ma_trend_ct, cls).setup_class(35, *args, **kwargs)


class Test_seasonal_ma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 37
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        kwargs['trend'] = [1, 0, 0, 1]
        kwargs['decimal'] = 3
        super(Test_seasonal_ma_trend_polynomial, cls).setup_class(
            36, *args, **kwargs)


class Test_seasonal_ma_diff(SARIMAXCoverageTest):
    # // SMA and I(d): (0, d, 0) x (0, 0, Q, s)
    # arima wpi, arima(0, 2, 0) sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 38
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        super(Test_seasonal_ma_diff, cls).setup_class(37, *args, **kwargs)


class Test_seasonal_ma_seasonal_diff(SARIMAXCoverageTest):
    # // SMA and I(D): (0, 0, 0) x (0, D, Q, s)
    # arima wpi, sarima(0, 2, 3, 4) noconstant vce(oim)
    # save_results 39
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 2, 3, 4)
        super(Test_seasonal_ma_seasonal_diff, cls).setup_class(
            38, *args, **kwargs)


class Test_seasonal_ma_diffuse(SARIMAXCoverageTest):
    # // SMA and diffuse initialization
    # arima wpi, sarima(0, 0, 3, 4) noconstant vce(oim) diffuse
    # save_results 40
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_ma_diffuse, cls).setup_class(39, *args, **kwargs)


class Test_seasonal_ma_exogenous(SARIMAXCoverageTest):
    # // SMAX
    # arima wpi x, sarima(0, 0, 3, 4) noconstant vce(oim)
    # save_results 41
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (0, 0, 3, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_ma_exogenous, cls).setup_class(40, *args, **kwargs)


class Test_seasonal_arma(SARIMAXCoverageTest):
    # // SARMA: (0, 0, 0) x (P, 0, Q, s)
    # arima wpi, sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 42
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        super(Test_seasonal_arma, cls).setup_class(41, *args, **kwargs)


class Test_seasonal_arma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 43
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        kwargs['trend'] = 'c'
        super(Test_seasonal_arma_trend_c, cls).setup_class(42, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:1] = (1 - tps[1:4].sum()) * tps[:1]


class Test_seasonal_arma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 44
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_arma_trend_ct, cls).setup_class(
            43, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]


class Test_seasonal_arma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1, 0, 0, 1]
    # arima wpi c t3, sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 45
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        kwargs['trend'] = [1, 0, 0, 1]
        kwargs['decimal'] = 3
        super(Test_seasonal_arma_trend_polynomial, cls).setup_class(
            44, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        # Known failure due to the complex step inducing non-stationary
        # parameters, causing a failure in the solve_discrete_lyapunov call
        # self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg


class Test_seasonal_arma_diff(SARIMAXCoverageTest):
    # // SARMA and I(d): (0, d, 0) x (P, 0, Q, s)
    # arima wpi, arima(0, 2, 0) sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 46
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        super(Test_seasonal_arma_diff, cls).setup_class(45, *args, **kwargs)


class Test_seasonal_arma_seasonal_diff(SARIMAXCoverageTest):
    # // SARMA and I(D): (0, 0, 0) x (P, D, Q, s)
    # arima wpi, sarima(3, 2, 2, 4) noconstant vce(oim)
    # save_results 47
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        super(Test_seasonal_arma_seasonal_diff, cls).setup_class(
            46, *args, **kwargs)


class Test_seasonal_arma_diff_seasonal_diff(SARIMAXCoverageTest):
    # // SARMA and I(d) and I(D): (0, d, 0) x (P, D, Q, s)
    # arima wpi, arima(0, 2, 0) sarima(3, 2, 2, 4) noconstant vce(oim)
    # save_results 48
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 0)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        super(Test_seasonal_arma_diff_seasonal_diff, cls).setup_class(
            47, *args, **kwargs)

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        # Known failure due to the complex step inducing non-stationary
        # parameters, causing a failure in the solve_discrete_lyapunov call
        # self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg


class Test_seasonal_arma_diffuse(SARIMAXCoverageTest):
    # // SARMA and diffuse initialization
    # arima wpi, sarima(3, 0, 2, 4) noconstant vce(oim) diffuse
    # save_results 49
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        kwargs['decimal'] = 3
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_arma_diffuse, cls).setup_class(48, *args, **kwargs)


class Test_seasonal_arma_exogenous(SARIMAXCoverageTest):
    # // SARMAX
    # arima wpi x, sarima(3, 0, 2, 4) noconstant vce(oim)
    # save_results 50
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 0, 0)
        kwargs['seasonal_order'] = (3, 0, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_arma_exogenous, cls).setup_class(
            49, *args, **kwargs)


class Test_sarimax_exogenous(SARIMAXCoverageTest):
    # // SARIMAX and exogenous
    # arima wpi x, arima(3, 2, 2) sarima(3, 2, 2, 4) noconstant vce(oim)
    # save_results 51
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_sarimax_exogenous, cls).setup_class(50, *args, **kwargs)

    def test_results_params(self):
        result = self.model.filter(self.true_params)
        assert_allclose(self.true_params[1:4], result.arparams)
        assert_allclose(self.true_params[4:6], result.maparams)
        assert_allclose(self.true_params[6:9], result.seasonalarparams)
        assert_allclose(self.true_params[9:11], result.seasonalmaparams)


class Test_sarimax_exogenous_not_hamilton(SARIMAXCoverageTest):
    # // SARIMAX and exogenous
    # arima wpi x, arima(3, 2, 2) sarima(3, 2, 2, 4) noconstant vce(oim)
    # save_results 51
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['hamilton_representation'] = False
        kwargs['simple_differencing'] = False
        super(Test_sarimax_exogenous_not_hamilton, cls).setup_class(
            50, *args, **kwargs)


class Test_sarimax_exogenous_diffuse(SARIMAXCoverageTest):
    # // SARIMAX and exogenous diffuse
    # arima wpi x, arima(3, 2, 2) sarima(3, 2, 2, 4) noconstant vce(oim)
    # diffuse
    # save_results 52
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['decimal'] = 2
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_sarimax_exogenous_diffuse, cls).setup_class(
            51, *args, **kwargs)


class Test_arma_exog_trend_polynomial_missing(SARIMAXCoverageTest):
    # // ARMA and exogenous and trend polynomial and missing
    # gen wpi2 = wpi
    # replace wpi2 = . in 10/19
    # arima wpi2 x c t3, arima(3, 0, 2) noconstant vce(oim)
    # save_results 53
    @classmethod
    def setup_class(cls, *args, **kwargs):
        endog = np.r_[results_sarimax.wpi1_data]
        # Note we're using the non-missing exog data
        kwargs['exog'] = ((endog - np.floor(endog))**2)[1:]
        endog[9:19] = np.nan
        endog = endog[1:] - endog[:-1]
        endog[9] = np.nan
        kwargs['order'] = (3, 0, 2)
        kwargs['trend'] = [0, 0, 0, 1]
        kwargs['decimal'] = 1
        super(Test_arma_exog_trend_polynomial_missing, cls).setup_class(
            52, endog=endog, *args, **kwargs)

        # Modify true params to convert from mean to intercept form
        tps = cls.true_params
        cls.true_params[0] = (1 - tps[2:5].sum()) * tps[0]


# Miscellaneous coverage tests
def test_simple_time_varying():
    # This tests time-varying parameters regression when in fact the parameters
    # are not time-varying, and in fact the regression fit is perfect
    endog = np.arange(100)*1.0
    exog = 2*endog
    mod = sarimax.SARIMAX(
        endog,
        exog=exog,
        order=(0, 0, 0),
        time_varying_regression=True,
        mle_regression=False)

    # Ignore the warning that MLE does not converge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=-1)

    # Test that the estimated variances of the errors are essentially zero
    # 5 digits necessary to accommodate 32-bit numpy/scipy with OpenBLAS 0.2.18
    assert_almost_equal(res.params, [0, 0], 5)

    # Test that the time-varying coefficients are all 0.5 (except the first
    # one)
    assert_almost_equal(res.filter_results.filtered_state[0][1:], [0.5]*99, 9)


def test_invalid_time_varying():
    assert_raises(
        ValueError,
        sarimax.SARIMAX,
        endog=[1, 2, 3],
        mle_regression=True,
        time_varying_regression=True)


def test_manual_stationary_initialization():
    endog = results_sarimax.wpi1_data

    # Create the first model to compare against
    mod1 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    res1 = mod1.filter([0.5, 0.2, 0.1, 1])

    # Create a second model with "known" initialization
    mod2 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    mod2.ssm.initialize_known(res1.filter_results.initial_state,
                              res1.filter_results.initial_state_cov)
    res2 = mod2.filter([0.5, 0.2, 0.1, 1])

    # Create a third model with "known" initialization, but specified in kwargs
    mod3 = sarimax.SARIMAX(
        endog, order=(3, 0, 0),
        initialization='known',
        initial_state=res1.filter_results.initial_state,
        initial_state_cov=res1.filter_results.initial_state_cov)
    res3 = mod3.filter([0.5, 0.2, 0.1, 1])

    # Create the forth model with stationary initialization specified in kwargs
    mod4 = sarimax.SARIMAX(endog, order=(3, 0, 0), initialization='stationary')
    res4 = mod4.filter([0.5, 0.2, 0.1, 1])

    # Just test a couple of things to make sure the results are the same
    assert_almost_equal(res1.llf, res2.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res2.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res3.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res3.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res4.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res4.filter_results.filtered_state)


def test_manual_approximate_diffuse_initialization():
    endog = results_sarimax.wpi1_data

    # Create the first model to compare against
    mod1 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    mod1.ssm.initialize_approximate_diffuse(1e9)
    res1 = mod1.filter([0.5, 0.2, 0.1, 1])

    # Create a second model with "known" initialization
    mod2 = sarimax.SARIMAX(endog, order=(3, 0, 0))
    mod2.ssm.initialize_known(res1.filter_results.initial_state,
                              res1.filter_results.initial_state_cov)
    res2 = mod2.filter([0.5, 0.2, 0.1, 1])

    # Create a third model with "known" initialization, but specified in kwargs
    mod3 = sarimax.SARIMAX(
        endog, order=(3, 0, 0),
        initialization='known',
        initial_state=res1.filter_results.initial_state,
        initial_state_cov=res1.filter_results.initial_state_cov)
    res3 = mod3.filter([0.5, 0.2, 0.1, 1])

    # Create the forth model with approximate diffuse initialization specified
    # in kwargs
    mod4 = sarimax.SARIMAX(endog, order=(3, 0, 0),
                           initialization='approximate_diffuse',
                           initial_variance=1e9)
    res4 = mod4.filter([0.5, 0.2, 0.1, 1])

    # Just test a couple of things to make sure the results are the same
    assert_almost_equal(res1.llf, res2.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res2.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res3.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res3.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res4.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res4.filter_results.filtered_state)


def test_results():
    endog = results_sarimax.wpi1_data

    mod = sarimax.SARIMAX(endog, order=(1, 0, 1))
    res = mod.filter([0.5, -0.5, 1], cov_type='oim')

    assert_almost_equal(res.arroots, 2.)
    assert_almost_equal(res.maroots, 2.)

    assert_almost_equal(res.arfreq, np.arctan2(0, 2) / (2*np.pi))
    assert_almost_equal(res.mafreq, np.arctan2(0, 2) / (2*np.pi))

    assert_almost_equal(res.arparams, [0.5])
    assert_almost_equal(res.maparams, [-0.5])


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        sarimax.SARIMAX(endog, exog=exog1, order=(1, 1, 0)),
        sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0)),
        sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0),
                        simple_differencing=False),
        sarimax.SARIMAX(endog_pd, exog=exog1_pd, order=(1, 1, 0)),
        sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0)),
        sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0),
                        simple_differencing=False),
    ]

    for mod in models:
        # Smoke tests
        mod.start_params
        res = mod.fit(disp=False)
        res.summary()
        res.predict()
        res.predict(dynamic=True)
        res.get_prediction()

        oos_exog = np.random.normal(size=(1, mod.k_exog))
        res.forecast(steps=1, exog=oos_exog)
        res.get_forecast(steps=1, exog=oos_exog)

        # Smoke tests for invalid exog
        oos_exog = np.random.normal(size=(2, mod.k_exog))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

    # Test invalid model specifications
    assert_raises(ValueError, sarimax.SARIMAX, endog, exog=np.zeros((10, 4)),
                  order=(1, 1, 0))


@pytest.mark.smoke
def test_datasets():
    # Test that some unusual types of datasets work

    np.random.seed(232849)
    endog = np.random.binomial(1, 0.5, size=100)
    exog = np.random.binomial(1, 0.5, size=100)
    mod = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0))
    mod.fit(disp=-1)


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=50))
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)


def test_arima000():
    # Test an ARIMA(0, 0, 0) with measurement error model (i.e. just estimating
    # a variance term)
    np.random.seed(328423)
    nobs = 50
    endog = pd.DataFrame(np.random.normal(size=nobs))
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state, endog.T)

    # ARIMA(0, 1, 0)
    mod = sarimax.SARIMAX(endog, order=(0, 1, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state[1:, 1:], endog.diff()[1:].T)

    # Exogenous variables
    error = np.random.normal(size=nobs)
    endog = np.ones(nobs) * 10 + error
    exog = np.ones(nobs)

    # OLS
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog)
    mod.ssm.filter_univariate = True
    res = mod.smooth([10., 1.])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)

    # RLS
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog,
                          mle_regression=False)
    mod.ssm.filter_univariate = True
    mod.initialize_known([0., 10.], np.diag([1., 0.]))
    res = mod.smooth([1.])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)
    assert_allclose(res.smoothed_state[1], 10, atol=1e-10)

    # RLS + TVP
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog,
                          mle_regression=False, time_varying_regression=True)
    mod.ssm.filter_univariate = True
    mod.initialize_known([10.], np.diag([0.]))
    res = mod.smooth([0., 1.])
    assert_allclose(res.smoothed_state[0], 10, atol=1e-10)


def check_concentrated_scale(filter_univariate=False):
    # Test that concentrating the scale out of the likelihood function works
    endog = np.diff(results_sarimax.wpi1_data)

    orders = [(1, 0, 0), (2, 2, 2)]
    seasonal_orders = [(0, 0, 0, 0), (1, 1, 1, 4)]

    simple_differencings = [True, False]
    exogs = [None, np.ones_like(endog)]
    trends = [None, 't']
    # Disabled, see discussion below in setting k_snr for details
    time_varying_regressions = [True, False]
    measurement_errors = [True, False]

    import itertools
    names = ['exog', 'order', 'seasonal_order', 'trend', 'measurement_error',
             'time_varying_regression', 'simple_differencing']
    for element in itertools.product(exogs, orders, seasonal_orders, trends,
                                     measurement_errors,
                                     time_varying_regressions,
                                     simple_differencings):
        kwargs = dict(zip(names, element))
        if kwargs.get('time_varying_regression', False):
            kwargs['mle_regression'] = False

        # Sometimes we can have slight differences if the Kalman filters
        # converge at different observations, so disable convergence.
        kwargs['tolerance'] = 0

        mod_orig = sarimax.SARIMAX(endog, **kwargs)
        mod_conc = sarimax.SARIMAX(endog, concentrate_scale=True, **kwargs)

        mod_orig.ssm.filter_univariate = filter_univariate
        mod_conc.ssm.filter_univariate = filter_univariate

        # The base parameters are the starting parameters from the concentrated
        # model
        conc_params = mod_conc.start_params
        res_conc = mod_conc.smooth(conc_params)

        # We need to map the concentrated parameters to the non-concentrated
        # model
        # The first thing is to add an additional parameter for the scale
        # (here set to 1 because we will multiply by res_conc.scale below, but
        # because the scale is factored out of the entire obs_cov and state_cov
        # matrices we may need to multiply more parameters by res_conc.scale
        # also)
        orig_params = np.r_[conc_params, 1]
        k_snr = 1

        # If we have time-varying regressions, then in the concentrated model
        # we actually are computing signal-to-noise ratios, and we
        # need to multiply it by the scale to get the variances
        # the non-concentrated model will expect as parameters
        if kwargs['time_varying_regression'] and kwargs['exog'] is not None:
            k_snr += 1
        # Note: the log-likelihood is not exactly the same between concentrated
        # and non-concentrated models with time-varying regression, so this
        # combinations raises NotImplementedError.

        # If we have measurement error, then in the concentrated model
        # we actually are computing the signal-to-noise ratio, and we
        # need to multiply it by the scale to get the measurement error
        # variance that the non-concentrated model will expect as a
        # parameter
        if kwargs['measurement_error']:
            k_snr += 1

        atol = 1e-5
        if kwargs['measurement_error'] or kwargs['time_varying_regression']:
            atol = 1e-3

        orig_params = np.r_[orig_params[:-k_snr],
                            res_conc.scale * orig_params[-k_snr:]]
        res_orig = mod_orig.smooth(orig_params)

        # Test loglike
        # Need to reduce the tolerance when we have measurement error.
        assert_allclose(res_conc.llf, res_orig.llf, atol=atol)

        # Test state space representation matrices
        for name in mod_conc.ssm.shapes:
            if name == 'obs':
                continue
            assert_allclose(getattr(res_conc.filter_results, name),
                            getattr(res_orig.filter_results, name))

        # Test filter / smoother output
        d = res_conc.loglikelihood_burn

        filter_attr = ['predicted_state', 'filtered_state', 'forecasts',
                       'forecasts_error', 'kalman_gain']

        for name in filter_attr:
            actual = getattr(res_conc.filter_results, name)
            desired = getattr(res_orig.filter_results, name)
            assert_allclose(actual, desired, atol=atol)

        # Note: do not want to compare the elements from any diffuse
        # initialization for things like covariances, so only compare for
        # periods past the loglikelihood_burn period
        filter_attr_burn = ['llf_obs', 'standardized_forecasts_error',
                            'predicted_state_cov', 'filtered_state_cov',
                            'tmp1', 'tmp2', 'tmp3', 'tmp4']
        # Also need to ignore covariances of states with diffuse initialization
        # when time_varying_regression is True
        diffuse_mask = (res_orig.filter_results.initial_state_cov.diagonal() ==
                        mod_orig.ssm.initial_variance)
        ix = np.s_[~diffuse_mask, ~diffuse_mask, :]

        for name in filter_attr_burn:
            actual = getattr(res_conc.filter_results, name)[..., d:]
            desired = getattr(res_orig.filter_results, name)[..., d:]
            # Note: Cannot compare predicted or filtered cov for the time
            # varying regression state due to effects of approximate diffuse
            # initialization
            if (name in ['predicted_state_cov', 'filtered_state_cov'] and
                    kwargs['time_varying_regression']):
                assert_allclose(actual[ix], desired[ix], atol=atol)
            else:
                assert_allclose(actual, desired, atol=atol)

        smoothed_attr = ['smoothed_state', 'smoothed_state_cov',
                         'smoothed_state_autocov',
                         'smoothed_state_disturbance',
                         'smoothed_state_disturbance_cov',
                         'smoothed_measurement_disturbance',
                         'smoothed_measurement_disturbance_cov',
                         'scaled_smoothed_estimator',
                         'scaled_smoothed_estimator_cov', 'smoothing_error',
                         'smoothed_forecasts', 'smoothed_forecasts_error',
                         'smoothed_forecasts_error_cov']

        for name in smoothed_attr:
            actual = getattr(res_conc.filter_results, name)[..., d:]
            desired = getattr(res_orig.filter_results, name)[..., d:]
            if (name in ['smoothed_state_cov', 'smoothed_state_autocov'] and
                    kwargs['time_varying_regression']):
                assert_allclose(actual[ix], desired[ix], atol=atol)
            else:
                assert_allclose(actual, desired, atol=atol)

        # Test non-covariance-matrix MLEResults output
        output = ['aic', 'bic', 'hqic', 'loglikelihood_burn']
        for name in output:
            actual = getattr(res_conc, name)
            desired = getattr(res_orig, name)
            assert_allclose(actual, desired, atol=atol)

        # Test diagnostic output
        actual = res_conc.test_normality(method='jarquebera')
        desired = res_orig.test_normality(method='jarquebera')
        assert_allclose(actual, desired, rtol=1e-5, atol=atol)

        actual = res_conc.test_heteroskedasticity(method='breakvar')
        desired = res_orig.test_heteroskedasticity(method='breakvar')
        assert_allclose(actual, desired, rtol=1e-5, atol=atol)

        actual = res_conc.test_serial_correlation(method='ljungbox')
        desired = res_orig.test_serial_correlation(method='ljungbox')
        assert_allclose(actual, desired, rtol=1e-5, atol=atol)

        # Test predict
        exog = None
        if kwargs['exog'] is not None:
            exog = np.ones((130 - mod_conc.nobs + 1, 1))
        actual = res_conc.get_prediction(start=100, end=130, dynamic=10,
                                         exog=exog)
        desired = res_orig.get_prediction(start=100, end=130, dynamic=10,
                                          exog=exog)
        assert_allclose(actual.predicted_mean, desired.predicted_mean,
                        atol=atol)
        assert_allclose(actual.se_mean, desired.se_mean, atol=atol)

        # Test simulate
        # Simulate is currently broken for time-varying models, so do not try
        # to test it here
        np.random.seed(13847)
        if mod_conc.ssm.time_invariant:
            measurement_shocks = np.random.normal(size=10)
            state_shocks = np.random.normal(size=10)
            initial_state = np.random.normal(size=(mod_conc.k_states, 1))
            actual = res_conc.simulate(10, measurement_shocks, state_shocks,
                                       initial_state)
            desired = res_orig.simulate(10, measurement_shocks, state_shocks,
                                        initial_state)
            assert_allclose(actual, desired, atol=atol)

        # Test impulse responses
        if mod_conc.ssm.time_invariant:
            actual = res_conc.impulse_responses(10)
            desired = res_orig.impulse_responses(10)
            assert_allclose(actual, desired, atol=atol)


@pytest.mark.slow
def test_concentrated_scale():
    check_concentrated_scale(filter_univariate=False)
    check_concentrated_scale(filter_univariate=True)


def test_forecast_exog():
    # Test forecasting with various shapes of `exog`
    nobs = 100
    endog = np.ones(nobs) * 2.0
    exog = np.ones(nobs)

    mod = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0))
    res = mod.smooth([2.0, 0.0, 1.0])

    # 1-step-ahead, valid
    exog_fcast_scalar = 1.
    exog_fcast_1dim = np.ones(1)
    exog_fcast_2dim = np.ones((1, 1))

    assert_allclose(res.forecast(1, exog=exog_fcast_scalar), 2.)
    assert_allclose(res.forecast(1, exog=exog_fcast_1dim), 2.)
    assert_allclose(res.forecast(1, exog=exog_fcast_2dim), 2.)

    # h-steps-ahead, valid
    h = 10
    exog_fcast_1dim = np.ones(h)
    exog_fcast_2dim = np.ones((h, 1))

    assert_allclose(res.forecast(h, exog=exog_fcast_1dim), 2.)
    assert_allclose(res.forecast(h, exog=exog_fcast_2dim), 2.)

    # h-steps-ahead, invalid
    assert_raises(ValueError, res.forecast, h, exog=1.)
    assert_raises(ValueError, res.forecast, h, exog=[1, 2])
    assert_raises(ValueError, res.forecast, h, exog=np.ones((h, 2)))


def check_equivalent_models(mod, mod2):
    attrs = [
        'measurement_error', 'state_error', 'mle_regression',
        'state_regression', 'time_varying_regression', 'simple_differencing',
        'enforce_stationarity', 'enforce_invertibility',
        'hamilton_representation', 'trend', 'polynomial_ar', 'polynomial_ma',
        'polynomial_seasonal_ar', 'polynomial_seasonal_ma', 'polynomial_trend',
        'k_ar', 'k_ar_params', 'k_diff', 'k_ma', 'k_ma_params',
        'seasonal_periods', 'k_seasonal_ar', 'k_seasonal_ar_params',
        'k_seasonal_diff', 'k_seasonal_ma', 'k_seasonal_ma_params',
        'k_trend', 'k_exog']

    ssm_attrs = [
        'nobs', 'k_endog', 'k_states', 'k_posdef', 'obs_intercept', 'design',
        'obs_cov', 'state_intercept', 'transition', 'selection', 'state_cov']

    for attr in attrs:
        print(attr)
        assert_equal(getattr(mod2, attr), getattr(mod, attr))

    for attr in ssm_attrs:
        assert_equal(getattr(mod2.ssm, attr), getattr(mod.ssm, attr))

    assert_equal(mod2._get_init_kwds(), mod._get_init_kwds())


def test_recreate_model():
    nobs = 100
    endog = np.ones(nobs) * 2.0
    exog = np.ones(nobs)

    orders = [(1, 0, 0), (2, 2, 2)]
    seasonal_orders = [(0, 0, 0, 0), (1, 1, 1, 4)]

    simple_differencings = [True, False]
    exogs = [None, np.ones_like(endog)]
    trends = [None, 't']
    # Disabled, see discussion below in setting k_snr for details
    time_varying_regressions = [True, False]
    measurement_errors = [True, False]

    import itertools
    names = ['exog', 'order', 'seasonal_order', 'trend', 'measurement_error',
             'time_varying_regression', 'simple_differencing']
    for element in itertools.product(exogs, orders, seasonal_orders, trends,
                                     measurement_errors,
                                     time_varying_regressions,
                                     simple_differencings):
        kwargs = dict(zip(names, element))
        if kwargs.get('time_varying_regression', False):
            kwargs['mle_regression'] = False
        exog = kwargs.pop('exog', None)

        mod = sarimax.SARIMAX(endog, exog=exog, **kwargs)
        mod2 = sarimax.SARIMAX(endog, exog=exog, **mod._get_init_kwds())

        check_equivalent_models(mod, mod2)


def test_append_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0), trend='t')
    res1 = mod1.smooth(params)

    mod2 = sarimax.SARIMAX(endog[:50], exog=exog[:50], order=(1, 0, 0),
                           trend='t')
    res2 = mod2.smooth(params)
    res3 = res2.append(endog[50:], exog=exog[50:])

    assert_equal(res1.specification, res3.specification)

    assert_allclose(res3.cov_params_default, res2.cov_params_default)
    for attr in ['nobs', 'llf', 'llf_obs', 'loglikelihood_burn']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))

    # Check that we get an error if we try to append without exog
    with pytest.raises(ValueError, match='Cloning a model with an exogenous'):
        res2.append(endog[50:])


def test_extend_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0), trend='t')
    res1 = mod1.smooth(params)

    mod2 = sarimax.SARIMAX(endog[:50], exog=exog[:50], order=(1, 0, 0),
                           trend='t')
    res2 = mod2.smooth(params)

    res3 = res2.extend(endog[50:], exog=exog[50:])

    assert_allclose(res3.llf_obs, res1.llf_obs[50:])

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        desired = getattr(res1, attr)
        if desired is not None:
            desired = desired[..., 50:]
        assert_equal(getattr(res3, attr), desired)

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))

    # Check that we get an error if we try to extend without exog
    with pytest.raises(ValueError, match='Cloning a model with an exogenous'):
        res2.extend(endog[50:])


def test_extend_by_one():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0), trend='t')
    res1 = mod1.smooth(params)

    mod2 = sarimax.SARIMAX(endog[:-1], exog=exog[:-1], order=(1, 0, 0),
                           trend='t')
    res2 = mod2.smooth(params)

    res3 = res2.extend(endog[-1:], exog=exog[-1:])

    assert_allclose(res3.llf_obs, res1.llf_obs[-1:])

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        desired = getattr(res1, attr)
        if desired is not None:
            desired = desired[..., 99:]
        assert_equal(getattr(res3, attr), desired)

    assert_allclose(res3.forecast(10, exog=np.ones(10) * 2),
                    res1.forecast(10, exog=np.ones(10) * 2))

    # Check that we get an error if we try to extend without exog
    with pytest.raises(ValueError, match='Cloning a model with an exogenous'):
        res2.extend(endog[-1:])


def test_apply_results():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1., 1., 0.1, 1.]

    mod1 = sarimax.SARIMAX(endog[:50], exog=exog[:50], order=(1, 0, 0),
                           trend='t')
    res1 = mod1.smooth(params)

    mod2 = sarimax.SARIMAX(endog[50:], exog=exog[50:], order=(1, 0, 0),
                           trend='t')
    res2 = mod2.smooth(params)

    res3 = res2.apply(endog[:50], exog=exog[:50])

    assert_equal(res1.specification, res3.specification)

    assert_allclose(res3.cov_params_default, res2.cov_params_default)
    for attr in ['nobs', 'llf', 'llf_obs', 'loglikelihood_burn']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
            'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov',
            'scaled_smoothed_estimator',
            'scaled_smoothed_estimator_cov', 'smoothing_error',
            'smoothed_state',
            'smoothed_state_cov', 'smoothed_state_autocov',
            'smoothed_measurement_disturbance',
            'smoothed_state_disturbance',
            'smoothed_measurement_disturbance_cov',
            'smoothed_state_disturbance_cov']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))

    assert_allclose(res3.forecast(10, exog=np.ones(10)),
                    res1.forecast(10, exog=np.ones(10)))

    # Check that we get an error if we try to apply without exog
    with pytest.raises(ValueError, match='Cloning a model with an exogenous'):
        res2.apply(endog[50:])


def test_start_params_small_nobs():
    # Test that starting parameters work even when nobs is very small, but
    # issues a warning.
    endog = np.log(realgdp_results['value']).diff()[1:].values

    # Regular ARMA
    mod = sarimax.SARIMAX(endog[:4], order=(4, 0, 0))
    match = ('Too few observations to estimate starting parameters for ARMA'
             ' and trend.')
    with pytest.warns(UserWarning, match=match):
        start_params = mod.start_params
        assert_allclose(start_params, [0, 0, 0, 0, np.var(endog[:4])])

    # Seasonal ARMA
    mod = sarimax.SARIMAX(endog[:4], order=(0, 0, 0),
                          seasonal_order=(1, 0, 0, 4))
    match = ('Too few observations to estimate starting parameters for'
             ' seasonal ARMA.')
    with pytest.warns(UserWarning, match=match):
        start_params = mod.start_params
        assert_allclose(start_params, [0, np.var(endog[:4])])


def test_simple_differencing_int64index():
    values = np.log(realgdp_results['value']).values
    endog = pd.Series(values, index=pd.Index(range(len(values))))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)

    assert_(mod._index.equals(endog.index[1:]))


def test_simple_differencing_rangeindex():
    values = np.log(realgdp_results['value']).values
    endog = pd.Series(values, index=pd.RangeIndex(start=0, stop=len(values)))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)

    assert_(mod._index.equals(endog.index[1:]))


def test_simple_differencing_dateindex():
    values = np.log(realgdp_results['value']).values
    endog = pd.Series(values, index=pd.period_range(
        start='2000', periods=len(values), freq='M'))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)

    assert_(mod._index.equals(endog.index[1:]))


def test_simple_differencing_strindex():
    values = np.log(realgdp_results['value']).values
    index = pd.Index(range(len(values))).map(str)
    endog = pd.Series(values, index=index)
    with pytest.warns(UserWarning):
        mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)

    assert_(mod._index.equals(pd.RangeIndex(start=0, stop=len(values) - 1)))
    assert_(mod.data.row_labels.equals(index[1:]))


def test_invalid_order():
    endog = np.zeros(10)
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, order=(1,))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, order=(1, 2, 3, 4))


def test_invalid_seasonal_order():
    endog = np.zeros(10)
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1,))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 2, 3, 4, 5))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 0, 0, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(0, 0, 1, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(1, 0, 1, 0))
    with pytest.raises(ValueError):
        sarimax.SARIMAX(endog, seasonal_order=(0, 0, 0, 1))


def test_dynamic_str():
    data = results_sarimax.wpi1_stationary["data"]
    index = pd.date_range("1980-1-1", freq="MS", periods=len(data))
    series = pd.Series(data, index=index)
    mod = sarimax.SARIMAX(series, order=(1, 1, 0), trend="c")
    res = mod.fit()
    dynamic = index[-12]
    desired = res.get_prediction(index[-24], dynamic=12)
    actual = res.get_prediction(index[-24], dynamic=dynamic)
    assert_allclose(actual.predicted_mean, desired.predicted_mean)
    actual = res.get_prediction(index[-24], dynamic=dynamic.to_pydatetime())
    assert_allclose(actual.predicted_mean, desired.predicted_mean)
    actual = res.get_prediction(index[-24],
                                dynamic=dynamic.strftime("%Y-%m-%d"))
    assert_allclose(actual.predicted_mean, desired.predicted_mean)


@pytest.mark.matplotlib
def test_plot_too_few_obs(reset_randomstate):
    # GH 6173
    # SO https://stackoverflow.com/questions/55930880/
    #    arima-models-plot-diagnostics-share-error/58051895#58051895
    mod = sarimax.SARIMAX(
        np.random.normal(size=10), order=(10, 0, 0), enforce_stationarity=False
    )
    with pytest.warns(UserWarning, match="Too few"):
        results = mod.fit()
    with pytest.raises(ValueError, match="Length of endogenous"):
        results.plot_diagnostics(figsize=(15, 5))
    y = np.random.standard_normal(9)
    mod = sarimax.SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with pytest.warns(UserWarning, match="Too few"):
        results = mod.fit()
    with pytest.raises(ValueError, match="Length of endogenous"):
        results.plot_diagnostics(figsize=(30, 15))


def test_sarimax_starting_values_few_obsevations(reset_randomstate):
    # GH 6396, 6801
    y = np.random.standard_normal(17)

    sarimax_model = sarimax.SARIMAX(
        endog=y, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12), trend="n"
    ).fit(disp=False)

    assert np.all(
        np.isfinite(sarimax_model.predict(start=len(y), end=len(y) + 11))
    )


def test_sarimax_starting_values_few_obsevations_long_ma(reset_randomstate):
    # GH 8232
    y = np.random.standard_normal(9)
    y = [
        3066.3, 3260.2, 3573.7, 3423.6, 3598.5, 3802.8, 3353.4, 4026.1,
        4684. , 4099.1, 3883.1, 3801.5, 3104. , 3574. , 3397.2, 3092.9,
        3083.8, 3106.7, 2939.6
    ]

    sarimax_model = sarimax.SARIMAX(
        endog=y, order=(0, 1, 5), trend="n"
    ).fit(disp=False)

    assert np.all(
        np.isfinite(sarimax_model.predict(start=len(y), end=len(y) + 11))
    )


def test_sarimax_forecast_exog_trend(reset_randomstate):
    # Test that an error is not raised that the given `exog` for the forecast
    # period is a constant when forecating with an intercept
    # GH 7019
    y = np.zeros(10)
    x = np.zeros(10)

    mod = sarimax.SARIMAX(endog=y, exog=x, order=(1, 0, 0), trend='c')
    res = mod.smooth([0.2, 0.4, 0.5, 1.0])

    # Test for h=1
    assert_allclose(res.forecast(1, exog=1), 0.2 + 0.4)

    # Test for h=2
    assert_allclose(res.forecast(2, exog=[1., 1.]), 0.2 + 0.4, 0.2 + 0.4 + 0.5)
