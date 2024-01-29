r"""
Tests for exponential smoothing models

Notes
-----

These tests are primarily against the `fpp` functions `ses`, `holt`, and `hw`
and against the `forecast` function `ets`. There are a couple of details about
how these packages work that are relevant for the tests:

Trend smoothing parameterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that `fpp` and `ets` use
different parameterizations for the trend smoothing parameter. Our
implementation in `statespace.exponential_smoothing` uses the same
parameterization as `ets`.

The `fpp` package follows Holt's recursive equations directly, in which the
trend updating is:

.. math::

    b_t = \beta^* (\ell_t - \ell_{t-1}) + (1 - \beta^*) b_{t-1}

In our implementation, state updating is done by the Kalman filter, in which
the trend updating equation is:

.. math::

    b_{t|t} = b_{t|t-1} + \beta (y_t - l_{t|t-1})

by rewriting the Kalman updating equation in the form of Holt's method, we
find that we must have :math:`\beta = \beta^* \alpha`. This is the same
parameterization used by `ets`, which does not use the Kalman fitler but
instead uses an innovations state space framework.

Loglikelihood
^^^^^^^^^^^^^

The `ets` package has a `loglik` output value, but it does not compute the
loglikelihood itself, but rather a version without the constant parameters. It
appears to compute:

.. math::

    -\frac{n}{2} \log \left (\sum_{t=1}^n \varepsilon_t^2 \right)

while the loglikelihood is:

.. math::

    -\frac{n}{2}
    \log \left (2 \pi e \frac{1}{n} \sum_{t=1}^n \varepsilon_t^2 \right)

See Hyndman et al. (2008), pages 68-69. In particular, the former equation -
which is the value returned by `ets` - is -0.5 times equation (5.3), since for
these models we have :math:`r(x_{t-1}) = 1`. The latter equation is the log
of the likelihood formula given at the top of page 69.

Confidence intervals
^^^^^^^^^^^^^^^^^^^^

The range of the confidence intervals depends on the estimated variance,
sigma^2. In our default, we concentrate this variance out of the loglikelihood
function, meaning that the default is to use the maximum likelihood estimate
for forecasting purposes. forecast::ets uses a degree-of-freedom-corrected
estimate of sigma^2, and so our default confidence bands will differ. To
correct for this in the tests, we set `concentrate_scale=False` and use the
estimated variance from forecast::ets.

TODO: may want to add a parameter allowing specification of the variance
      estimator.

Author: Chad Fulton
License: BSD-3
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import pytest
from numpy.testing import assert_, assert_equal, assert_allclose

from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing)

current_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_path, 'results')
params_path = os.path.join(results_path, 'exponential_smoothing_params.csv')
predict_path = os.path.join(results_path, 'exponential_smoothing_predict.csv')
states_path = os.path.join(results_path, 'exponential_smoothing_states.csv')
results_params = pd.read_csv(params_path, index_col=[0])
results_predict = pd.read_csv(predict_path, index_col=[0])
results_states = pd.read_csv(states_path, index_col=[0])

# R, fpp: oildata <- window(oil,start=1996,end=2007)
oildata = pd.Series([
    446.6565229, 454.4733065, 455.6629740, 423.6322388, 456.2713279,
    440.5880501, 425.3325201, 485.1494479, 506.0481621, 526.7919833,
    514.2688890, 494.2110193],
    index=pd.period_range(start='1996', end='2007', freq='Y'))

# R, fpp: air <- window(ausair,start=1990,end=2004)
air = pd.Series([
    17.553400, 21.860100, 23.886600, 26.929300, 26.888500,
    28.831400, 30.075100, 30.953500, 30.185700, 31.579700,
    32.577569, 33.477398, 39.021581, 41.386432, 41.596552],
    index=pd.period_range(start='1990', end='2004', freq='Y'))

# R, fpp: aust <- window(austourists,start=2005)
aust = pd.Series([
    41.727458, 24.041850, 32.328103, 37.328708, 46.213153,
    29.346326, 36.482910, 42.977719, 48.901525, 31.180221,
    37.717881, 40.420211, 51.206863, 31.887228, 40.978263,
    43.772491, 55.558567, 33.850915, 42.076383, 45.642292,
    59.766780, 35.191877, 44.319737, 47.913736],
    index=pd.period_range(start='2005Q1', end='2010Q4', freq='Q-OCT'))


class CheckExponentialSmoothing:
    @classmethod
    def setup_class(cls, name, res):
        cls.name = name
        cls.res = res
        cls.nobs = res.nobs
        cls.nforecast = len(results_predict['%s_mean' % cls.name]) - cls.nobs
        cls.forecast = res.get_forecast(cls.nforecast)

    def test_fitted(self):
        predicted = results_predict['%s_mean' % self.name]
        assert_allclose(self.res.fittedvalues, predicted.iloc[:self.nobs])

    def test_output(self):
        # There are two types of output, depending on some internal switch of
        # fpp::ses that appears to depend on if parameters are estimated. If
        # they are estimated, then llf and mse are available but sse is not.
        # Otherwise, sse is available and the other two aren't.
        has_llf = ~np.isnan(results_params[self.name]['llf'])
        if has_llf:
            assert_allclose(self.res.mse, results_params[self.name]['mse'])
            # As noted in the file docstring, `ets` does not return the actual
            # loglikelihood, but instead a transformation of it. Here we
            # compute that transformation based on our results, so as to
            # compare with the `ets` output.
            actual = -0.5 * self.nobs * np.log(np.sum(self.res.resid**2))
            assert_allclose(actual, results_params[self.name]['llf'])
        else:
            assert_allclose(self.res.sse, results_params[self.name]['sse'])

    def test_forecasts(self):
        # Forecast mean
        predicted = results_predict['%s_mean' % self.name]
        assert_allclose(
            self.forecast.predicted_mean,
            predicted.iloc[self.nobs:]
        )

    def test_conf_int(self):
        # Forecast confidence intervals
        ci_95 = self.forecast.conf_int(alpha=0.05)
        lower = results_predict['%s_lower' % self.name]
        upper = results_predict['%s_upper' % self.name]

        assert_allclose(ci_95['lower y'], lower.iloc[self.nobs:])
        assert_allclose(ci_95['upper y'], upper.iloc[self.nobs:])

    def test_initial_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[0]
        assert_allclose(self.res.initial_state.iloc[0], desired)

    def test_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[1:]
        assert_allclose(self.res.filtered_state[1:].T, desired)

    def test_misc(self):
        mod = self.res.model
        assert_equal(mod.k_params, len(mod.start_params))
        assert_equal(mod.k_params, len(mod.param_names))

        # Smoke test for summary creation
        self.res.summary()


class TestSESFPPFixed02(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test simple exponential smoothing (FPP: 7.1) against fpp::ses, with
        # a fixed coefficient 0.2 and simple initialization
        mod = ExponentialSmoothing(oildata, initialization_method='simple')
        res = mod.filter([results_params['oil_fpp1']['alpha']])

        super().setup_class('oil_fpp1', res)


class TestSESFPPFixed06(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test simple exponential smoothing (FPP: 7.1) against fpp::ses, with
        # a fixed coefficient 0.6 and simple initialization
        mod = ExponentialSmoothing(oildata, initialization_method='simple')
        res = mod.filter([results_params['oil_fpp2']['alpha']])

        super().setup_class('oil_fpp2', res)


class TestSESFPPEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test simple exponential smoothing (FPP: 7.1) against fpp::ses, with
        # estimated coefficients
        mod = ExponentialSmoothing(oildata, initialization_method='estimated',
                                   concentrate_scale=False)
        res = mod.filter([results_params['oil_fpp3']['alpha'],
                          results_params['oil_fpp3']['sigma2'],
                          results_params['oil_fpp3']['l0']])

        super().setup_class('oil_fpp3', res)


class TestSESETSEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test simple exponential smoothing (FPP: 7.1) against forecast::ets,
        # with estimated coefficients
        mod = ExponentialSmoothing(oildata, initialization_method='estimated',
                                   concentrate_scale=False)
        res = mod.filter([results_params['oil_ets']['alpha'],
                          results_params['oil_ets']['sigma2'],
                          results_params['oil_ets']['l0']])

        super().setup_class('oil_ets', res)

    def test_mle_estimates(self):
        # Test that our fitted coefficients are at least as good as those from
        # `ets`
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltFPPFixed(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt's linear trend method (FPP: 7.2) against fpp::holt,
        # with fixed coefficients and simple initialization

        mod = ExponentialSmoothing(air, trend=True, concentrate_scale=False,
                                   initialization_method='simple')
        # alpha, beta^*
        params = [results_params['air_fpp1']['alpha'],
                  results_params['air_fpp1']['beta_star'],
                  results_params['air_fpp1']['sigma2']]
        # beta = alpha * beta^*
        params[1] = params[0] * params[1]
        res = mod.filter(params)

        super().setup_class('air_fpp1', res)

    def test_conf_int(self):
        # Note: cannot test against the output of the `holt` command in this
        # case, as `holt` seems to have a bug: while it is parametrized in
        # terms of `beta_star`, its confidence intervals are computed as though
        # beta_star was actually beta = alpha * beta_star.
        # Instead, we'll compare against a direct computation as in
        # Hyndman et al. (2008) equation (6.1).
        j = np.arange(1, 14)
        alpha, beta, sigma2 = self.res.params
        c = np.r_[0, alpha + beta * j]
        se = (sigma2 * (1 + np.cumsum(c**2)))**0.5
        assert_allclose(self.forecast.se_mean, se)


class TestHoltDampedFPPEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt's linear trend method (FPP: 7.2) with a damped trend
        # against fpp::holt, with estimated coefficients

        mod = ExponentialSmoothing(air, trend=True, damped_trend=True,
                                   concentrate_scale=False)
        params = [results_params['air_fpp2']['alpha'],
                  results_params['air_fpp2']['beta'],
                  results_params['air_fpp2']['phi'],
                  results_params['air_fpp2']['sigma2'],
                  results_params['air_fpp2']['l0'],
                  results_params['air_fpp2']['b0']]
        res = mod.filter(params)

        super().setup_class('air_fpp2', res)


class TestHoltDampedETSEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt's linear trend method (FPP: 7.2) with a damped trend
        # against forecast::ets, with estimated coefficients

        mod = ExponentialSmoothing(air, trend=True, damped_trend=True,
                                   concentrate_scale=False)
        params = [results_params['air_ets']['alpha'],
                  results_params['air_ets']['beta'],
                  results_params['air_ets']['phi'],
                  results_params['air_ets']['sigma2'],
                  results_params['air_ets']['l0'],
                  results_params['air_ets']['b0']]
        res = mod.filter(params)

        super().setup_class('air_ets', res)

    def test_mle_estimates(self):
        # Test that our fitted coefficients are at least as good as those from
        # `ets`
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltWintersFPPEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt-Winters seasonal method (FPP: 7.5) against fpp::hw,
        # with estimated coefficients

        mod = ExponentialSmoothing(aust, trend=True, seasonal=4,
                                   concentrate_scale=False)
        params = np.r_[
            results_params['aust_fpp1']['alpha'],
            results_params['aust_fpp1']['beta'],
            results_params['aust_fpp1']['gamma'],
            results_params['aust_fpp1']['sigma2'],
            results_params['aust_fpp1']['l0'],
            results_params['aust_fpp1']['b0'],
            results_params['aust_fpp1']['s0_0'],
            results_params['aust_fpp1']['s0_1'],
            results_params['aust_fpp1']['s0_2']]
        res = mod.filter(params)

        super().setup_class('aust_fpp1', res)


class TestHoltWintersETSEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt-Winters seasonal method (FPP: 7.5) against forecast::ets,
        # with estimated coefficients

        mod = ExponentialSmoothing(aust, trend=True, seasonal=4,
                                   concentrate_scale=False)
        params = np.r_[
            results_params['aust_ets1']['alpha'],
            results_params['aust_ets1']['beta'],
            results_params['aust_ets1']['gamma'],
            results_params['aust_ets1']['sigma2'],
            results_params['aust_ets1']['l0'],
            results_params['aust_ets1']['b0'],
            results_params['aust_ets1']['s0_0'],
            results_params['aust_ets1']['s0_1'],
            results_params['aust_ets1']['s0_2']]
        res = mod.filter(params)

        super().setup_class('aust_ets1', res)


class TestHoltWintersDampedETSEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt-Winters seasonal method (FPP: 7.5) with a damped trend
        # against forecast::ets, with estimated coefficients

        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True,
                                   seasonal=4, concentrate_scale=False)
        params = np.r_[
            results_params['aust_ets2']['alpha'],
            results_params['aust_ets2']['beta'],
            results_params['aust_ets2']['gamma'],
            results_params['aust_ets2']['phi'],
            results_params['aust_ets2']['sigma2'],
            results_params['aust_ets2']['l0'],
            results_params['aust_ets2']['b0'],
            results_params['aust_ets2']['s0_0'],
            results_params['aust_ets2']['s0_1'],
            results_params['aust_ets2']['s0_2']]
        res = mod.filter(params)

        super().setup_class('aust_ets2', res)

    def test_mle_estimates(self):
        # Test that our fitted coefficients are at least as good as those from
        # `ets`
        mle_res = self.res.model.fit(disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltWintersNoTrendETSEstimated(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        # Test Holt-Winters seasonal method (FPP: 7.5) with no trend
        # against forecast::ets, with estimated coefficients

        mod = ExponentialSmoothing(aust, seasonal=4, concentrate_scale=False)
        params = np.r_[
            results_params['aust_ets3']['alpha'],
            results_params['aust_ets3']['gamma'],
            results_params['aust_ets3']['sigma2'],
            results_params['aust_ets3']['l0'],
            results_params['aust_ets3']['s0_0'],
            results_params['aust_ets3']['s0_1'],
            results_params['aust_ets3']['s0_2']]
        res = mod.filter(params)

        super().setup_class('aust_ets3', res)

    def test_conf_int(self):
        # `forecast::ets` seems to have a bug in this case related to the
        # seasonal component of the standard error computation. From
        # Hyndman et al. (2008) Table 6.2, "d_{j,m} = 1 if j = 0 (mod m) and
        # 0 otherwise". This implies that the seasonal effect on the standard
        # error computation should only start when the j = m (here m = 4) term
        # is included in the standard error computation, which happens at the
        # fifth forecast (h=5). However, `ets` is starting it at the fourth
        # forecast (h=4).
        # Instead, we'll compare against a direct computation as in
        # Hyndman et al. (2008) equation (6.1).
        j = np.arange(1, 5)
        alpha, gamma, sigma2 = self.res.params[:3]
        c = np.r_[0, alpha + gamma * ((j % 4) == 0).astype(int)]
        se = (sigma2 * (1 + np.cumsum(c**2)))**0.5
        assert_allclose(self.forecast.se_mean, se)

    def test_mle_estimates(self):
        # Test that our fitted coefficients are at least as good as those from
        # `ets`
        start_params = [0.5, 0.4, 4, 32, 2.3, -2, -9]
        mle_res = self.res.model.fit(start_params, disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)


class CheckKnownInitialization:
    @classmethod
    def setup_class(cls, mod, start_params):
        # Base model, with estimated initialization
        # Note: we use start_params here that are pretty close to MLE so that
        # tests run quicker.
        cls.mod = mod
        cls.start_params = start_params
        endog = mod.data.orig_endog
        cls.res = cls.mod.fit(start_params, disp=0, maxiter=100)

        # Get the estimated initial parameters
        cls.initial_level = cls.res.params.get('initial_level', None)
        cls.initial_trend = cls.res.params.get('initial_trend', None)
        cls.initial_seasonal = None
        if cls.mod.seasonal:
            cls.initial_seasonal = (
                [cls.res.params['initial_seasonal']]
                + [cls.res.params['initial_seasonal.L%d' % i]
                   for i in range(1, cls.mod.seasonal_periods - 1)])

        # Get the estimated parameters
        cls.params = cls.res.params[:'initial_level'].drop('initial_level')
        cls.init_params = cls.res.params['initial_level':]

        # Create a model with the given known initialization
        cls.known_mod = cls.mod.clone(endog, initialization_method='known',
                                      initial_level=cls.initial_level,
                                      initial_trend=cls.initial_trend,
                                      initial_seasonal=cls.initial_seasonal)

    def test_given_params(self):
        # Test fixed initialization with given parameters
        # And filter with the given other parameters
        known_res = self.known_mod.filter(self.params)

        assert_allclose(known_res.llf, self.res.llf)
        assert_allclose(known_res.predicted_state, self.res.predicted_state)
        assert_allclose(known_res.predicted_state_cov,
                        self.res.predicted_state_cov)
        assert_allclose(known_res.filtered_state, self.res.filtered_state)

    def test_estimated_params(self):
        # Now fit the original model with a fixed initial_level and make sure
        # that it gives the same result as the fitted second model
        fit_res1 = self.mod.fit_constrained(
            self.init_params.to_dict(), start_params=self.start_params,
            includes_fixed=True, disp=0)
        fit_res2 = self.known_mod.fit(
            self.start_params[:'initial_level'].drop('initial_level'), disp=0)

        assert_allclose(
            fit_res1.params[:'initial_level'].drop('initial_level'),
            fit_res2.params)
        assert_allclose(fit_res1.llf, fit_res2.llf)
        assert_allclose(fit_res1.scale, fit_res2.scale)
        assert_allclose(fit_res1.predicted_state, fit_res2.predicted_state)
        assert_allclose(fit_res1.predicted_state_cov,
                        fit_res2.predicted_state_cov)
        assert_allclose(fit_res1.filtered_state, fit_res2.filtered_state)


class TestSESKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata)
        start_params = pd.Series([0.8, 440.], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True)

        start_params = pd.Series(
            [0.95, 0.0005, 15., 1.5], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltDampedKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True)
        start_params = pd.Series(
            [0.9, 0.0005, 0.9, 14., 2.], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltWintersKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.5, 33., 0.4, 2.5, -2., -9.],
            index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltWintersDampedKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True,
                                   seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0005, 0.95, 17.0, 1.5, -0.2, 0.1, 0.4],
            index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltWintersNoTrendKnownInitialization(CheckKnownInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series(
            [0.5, 0.49, 30., 2., -2, -9], index=mod.param_names)
        super().setup_class(mod, start_params)


class CheckHeuristicInitialization:
    @classmethod
    def setup_class(cls, mod):
        cls.mod = mod
        cls.res = cls.mod.filter(cls.mod.start_params)

        # Save the heuristic values
        init_heuristic = np.r_[cls.mod._initial_level]
        if cls.mod.trend:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_trend]
        if cls.mod.seasonal:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_seasonal]
        cls.init_heuristic = init_heuristic

        # Create a model with the given known initialization
        endog = cls.mod.data.orig_endog
        initial_seasonal = cls.mod._initial_seasonal
        cls.known_mod = cls.mod.clone(endog, initialization_method='known',
                                      initial_level=cls.mod._initial_level,
                                      initial_trend=cls.mod._initial_trend,
                                      initial_seasonal=initial_seasonal)
        cls.known_res = cls.mod.filter(cls.mod.start_params)


class TestSESHeuristicInitialization(CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        # See Hyndman et al. (2008), section 2.6
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(oildata.values[:nobs])[0]
        assert_allclose(self.init_heuristic, desired)


class TestHoltHeuristicInitialization(CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True,
                                   initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        # See Hyndman et al. (2008), section 2.6
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(air.values[:nobs])
        assert_allclose(self.init_heuristic, desired)


class TestHoltDampedHeuristicInitialization(CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True,
                                   initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltHeuristicInitialization.test_heuristic(self)


class TestHoltWintersHeuristicInitialization(CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4,
                                   initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        # See Hyndman et al. (2008), section 2.6

        # Get trend from 2x4 MA filter
        trend = (aust[:20].rolling(4).mean()
                          .rolling(2).mean().shift(-2).dropna())
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(trend[:nobs])
        if not self.mod.trend:
            desired = desired[:1]

        # Get seasonal initial states
        detrended = aust - trend
        initial_seasonal = np.nanmean(detrended.values.reshape(6, 4), axis=0)
        # The above command gets seasonals for observations 1, 2, 3, 4.
        # Lagging these four periods gives us initial seasonals for lags
        # L3, L2, L1, L0, but the state vector is ordered L0, L1, L2, L3, so we
        # need to reverse the order of this vector.
        initial_seasonal = initial_seasonal[::-1]
        desired = np.r_[desired, initial_seasonal - np.mean(initial_seasonal)]

        assert_allclose(self.init_heuristic, desired)


class TestHoltWintersDampedHeuristicInitialization(
        CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True,
                                   seasonal=4,
                                   initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltWintersHeuristicInitialization.test_heuristic(self)


class TestHoltWintersNoTrendHeuristicInitialization(
        CheckHeuristicInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4,
                                   initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltWintersHeuristicInitialization.test_heuristic(self)


def test_concentrated_initialization():
    # Compare a model where initialization is concentrated out versus
    # numarical maximum likelihood estimation
    mod1 = ExponentialSmoothing(oildata, initialization_method='concentrated')
    mod2 = ExponentialSmoothing(oildata)

    # First, fix the other parameters at a particular value
    res1 = mod1.filter([0.1])
    res2 = mod2.fit_constrained({'smoothing_level': 0.1}, disp=0)

    # Alternatively, estimate the remaining parameters
    res1 = mod1.fit(disp=0)
    res2 = mod2.fit(disp=0)

    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.initial_state, res2.initial_state, rtol=1e-5)


class CheckConcentratedInitialization:
    @classmethod
    def setup_class(cls, mod, start_params=None, atol=0, rtol=1e-7):
        # Note: because of the different computations methods (linear
        # regression in the concentrated case versus numerical MLE in the base
        # case), we have relatively large tolerances for these tests,
        # particularly when the other parameters are estimated, and we specify
        # start parameters relatively close to the MLE to avoid problems with
        # the methods finding different local maxima.
        cls.start_params = start_params
        cls.atol = atol
        cls.rtol = rtol

        # Compare a model where initialization is concentrated out versus
        # numarical maximum likelihood estimation
        cls.mod = mod
        cls.conc_mod = mod.clone(mod.data.orig_endog,
                                 initialization_method='concentrated')

        # Generate some fixed parameters
        cls.params = pd.Series([0.5, 0.2, 0.2, 0.95], index=[
            'smoothing_level', 'smoothing_trend', 'smoothing_seasonal',
            'damping_trend'])

        drop = []
        if not cls.mod.trend:
            drop += ['smoothing_trend', 'damping_trend']
        elif not cls.mod.damped_trend:
            drop += ['damping_trend']
        if not cls.mod.seasonal:
            drop += ['smoothing_seasonal']
        cls.params.drop(drop, inplace=True)

    def test_given_params(self):
        # First, fix the other parameters at a particular value
        # (for the non-concentrated model, we need to fit the inital values
        # directly by MLE)
        res = self.mod.fit_constrained(self.params.to_dict(), disp=0)
        conc_res = self.conc_mod.filter(self.params.values)

        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(conc_res.initial_state, res.initial_state,
                        atol=self.atol, rtol=self.rtol)

    def test_estimated_params(self):
        # Alternatively, estimate the remaining parameters
        res = self.mod.fit(self.start_params, disp=0, maxiter=100)
        np.set_printoptions(suppress=True)
        conc_res = self.conc_mod.fit(self.start_params[:len(self.params)],
                                     disp=0)

        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(conc_res.initial_state, res.initial_state,
                        atol=self.atol, rtol=self.rtol)


class TestSESConcentratedInitialization(CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata)
        start_params = pd.Series([0.85, 447.], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-5)


class TestHoltConcentratedInitialization(CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True)
        start_params = pd.Series(
            [0.95, 0.0005, 15., 1.5], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-4)


class TestHoltDampedConcentratedInitialization(
        CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True)
        start_params = pd.Series(
            [0.95, 0.0005,  0.9, 15.,  2.5], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-1)


class TestHoltWintersConcentratedInitialization(
        CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0002, 33., 0.4, 2.2, -2., -9.3],
            index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-3)


class TestHoltWintersDampedConcentratedInitialization(
        CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True,
                                   seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0005, 0.95, 17.0, 1.5, -0.2, 0.1, 0.4],
            index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-1)


class TestHoltWintersNoTrendConcentratedInitialization(
        CheckConcentratedInitialization):
    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series(
            [0.5, 0.49, 32., 2.3, -2.1, -9.3], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-4)


class TestMultiIndex(CheckExponentialSmoothing):
    @classmethod
    def setup_class(cls):
        oildata_copy = oildata.copy()
        oildata_copy.name = ("oil", "data")
        mod = ExponentialSmoothing(oildata_copy,
                                   initialization_method='simple')
        res = mod.filter([results_params['oil_fpp2']['alpha']])

        super().setup_class('oil_fpp2', res)

    def test_conf_int(self):
        # Forecast confidence intervals
        ci_95 = self.forecast.conf_int(alpha=0.05)
        lower = results_predict['%s_lower' % self.name]
        upper = results_predict['%s_upper' % self.name]
        assert_allclose(ci_95["lower ('oil', 'data')"], lower.iloc[self.nobs:])
        assert_allclose(ci_95["upper ('oil', 'data')"], upper.iloc[self.nobs:])


def test_invalid():
    # Tests for invalid model specifications that raise ValueErrors
    with pytest.raises(
            ValueError, match='Cannot have a seasonal period of 1.'):
        mod = ExponentialSmoothing(aust, seasonal=1)

    with pytest.raises(TypeError, match=(
            'seasonal must be integer_like'
            r' \(int or np.integer, but not bool or timedelta64\) or None')):
        mod = ExponentialSmoothing(aust, seasonal=True)

    with pytest.raises(
            ValueError, match='Invalid initialization method "invalid".'):
        mod = ExponentialSmoothing(aust, initialization_method='invalid')

    with pytest.raises(ValueError, match=(
            '`initial_level` argument must be provided'
            ' when initialization method is set to'
            ' "known".')):
        mod = ExponentialSmoothing(aust, initialization_method='known')

    with pytest.raises(ValueError, match=(
            '`initial_trend` argument must be provided'
            ' for models with a trend component when'
            ' initialization method is set to "known".')):
        mod = ExponentialSmoothing(
            aust, trend=True, initialization_method='known', initial_level=0)

    with pytest.raises(ValueError, match=(
            '`initial_seasonal` argument must be provided'
            ' for models with a seasonal component when'
            ' initialization method is set to "known".')):
        mod = ExponentialSmoothing(
            aust, seasonal=4, initialization_method='known', initial_level=0)

    for arg in ['initial_level', 'initial_trend', 'initial_seasonal']:
        msg = ('Cannot give `%s` argument when initialization is "estimated"'
               % arg)
        with pytest.raises(ValueError, match=msg):
            mod = ExponentialSmoothing(aust, **{arg: 0})

    with pytest.raises(ValueError, match=(
            'Invalid length of initial seasonal values. Must be'
            ' one of s or s-1, where s is the number of seasonal'
            ' periods.')):
        mod = ExponentialSmoothing(
            aust, seasonal=4, initialization_method='known', initial_level=0,
            initial_seasonal=0)

    with pytest.raises(NotImplementedError,
                       match='ExponentialSmoothing does not support `exog`.'):
        mod = ExponentialSmoothing(aust)
        mod.clone(aust, exog=air)


def test_parameterless_model(reset_randomstate):
    # GH 6687
    x = np.cumsum(np.random.standard_normal(1000))
    ses = ExponentialSmoothing(x, initial_level=x[0],
                               initialization_method="known")
    with ses.fix_params({'smoothing_level': 0.5}):
        res = ses.fit()
    assert np.isnan(res.bse).all()
    assert res.fixed_params == ["smoothing_level"]
