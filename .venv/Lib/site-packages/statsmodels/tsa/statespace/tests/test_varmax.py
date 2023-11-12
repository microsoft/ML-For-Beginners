"""
Tests for VARMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
import os
import re
import warnings

import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest

from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg

from .results import results_varmax

current_path = os.path.dirname(os.path.abspath(__file__))

var_path = os.path.join('results', 'results_var_stata.csv')
var_results = pd.read_csv(os.path.join(current_path, var_path))

varmax_path = os.path.join('results', 'results_varmax_stata.csv')
varmax_results = pd.read_csv(os.path.join(current_path, varmax_path))


class CheckVARMAX:
    """
    Test Vector Autoregression against Stata's `dfactor` code (Stata's
    `var` function uses OLS and not state space / MLE, so we cannot get
    equivalent log-likelihoods)
    """

    def test_mle(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            # Fit with all transformations
            # results = self.model.fit(method='powell', disp=-1)
            results = self.model.fit(maxiter=100, disp=False)
            # Fit now without transformations
            self.model.enforce_stationarity = False
            self.model.enforce_invertibility = False
            results = self.model.fit(results.params, method='nm', maxiter=1000,
                                     disp=False)
            self.model.enforce_stationarity = True
            self.model.enforce_invertibility = True
            assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    @pytest.mark.smoke
    def test_params(self):
        # Smoke test to make sure the start_params are well-defined and
        # lead to a well-defined model
        model = self.model

        model.filter(model.start_params)
        # Similarly a smoke test for param_names
        assert len(model.start_params) == len(model.param_names)

        # Finally make sure the transform and untransform do their job
        actual = model.transform_params(
            model.untransform_params(model.start_params))
        assert_allclose(actual, model.start_params)

        # Also in the case of enforce invertibility and stationarity = False
        model.enforce_stationarity = False
        model.enforce_invertibility = False
        actual = model.transform_params(
            model.untransform_params(model.start_params))

        model.enforce_stationarity = True
        model.enforce_invertibility = True
        assert_allclose(actual, model.start_params)

    @pytest.mark.smoke
    def test_results(self):
        # Smoke test for creating the summary
        self.results.summary()

        model = self.model
        # Test cofficient matrix creation
        #   (via a different, more direct, method)
        if model.k_ar > 0:
            params_ar = np.array(self.results.params[model._params_ar])
            coefficients = params_ar.reshape(model.k_endog,
                                             model.k_endog * model.k_ar)
            coefficient_matrices = np.array([
                coefficients[:model.k_endog,
                             i*model.k_endog:(i+1)*model.k_endog]
                for i in range(model.k_ar)
            ])
            assert_equal(self.results.coefficient_matrices_var,
                         coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)
        if model.k_ma > 0:
            params_ma = np.array(self.results.params[model._params_ma])
            coefficients = params_ma.reshape(model.k_endog,
                                             model.k_endog * model.k_ma)

            coefficient_matrices = np.array([
                coefficients[:model.k_endog,
                             i*model.k_endog:(i+1)*model.k_endog]
                for i in range(model.k_ma)
            ])
            assert_equal(self.results.coefficient_matrices_vma,
                         coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_vma, None)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-6)

    def test_aic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, end, atol=1e-6, **kwargs):
        # Tests predict + forecast
        assert_allclose(
            self.results.predict(end=end, **kwargs),
            self.true['predict'],
            atol=atol)

    def test_dynamic_predict(self, end, dynamic, atol=1e-6, **kwargs):
        # Tests predict + dynamic predict + forecast
        assert_allclose(
            self.results.predict(end=end, dynamic=dynamic, **kwargs),
            self.true['dynamic_predict'],
            atol=atol)

    def test_standardized_forecasts_error(self):
        cython_sfe = self.results.standardized_forecasts_error
        self.results._standardized_forecasts_error = None
        python_sfe = self.results.standardized_forecasts_error
        assert_allclose(cython_sfe, python_sfe)


class CheckLutkepohl(CheckVARMAX):
    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx',
                    included_vars=['dln_inv', 'dln_inc', 'dln_consump'],
                    **kwargs):
        cls.true = true
        # 1960:Q1 - 1982:Q4
        dta = pd.DataFrame(
            results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
            index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))

        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()

        endog = dta.loc['1960-04-01':'1978-10-01', included_vars]

        cls.model = varmax.VARMAX(endog, order=order, trend=trend,
                                  error_cov_type=error_cov_type, **kwargs)

        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)

    def test_predict(self, **kwargs):
        super(CheckLutkepohl, self).test_predict(end='1982-10-01', **kwargs)

    def test_dynamic_predict(self, **kwargs):
        super(CheckLutkepohl, self).test_dynamic_predict(end='1982-10-01',
                                                         dynamic='1961-01-01',
                                                         **kwargs)


class TestVAR(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1.copy()
        true['predict'] = var_results.iloc[1:][['predict_1',
                                                'predict_2',
                                                'predict_3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_1',
                                                        'dyn_predict_2',
                                                        'dyn_predict_3']]
        super(TestVAR, cls).setup_class(
            true,  order=(1, 0), trend='n',
            error_cov_type="unstructured")

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-4)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VAR\(1\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 8

            # -> Check that we have the right coefficients
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2],
                             table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 11

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%.4f' % (names[i], params[i]), table)


class TestVAR_diagonal(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag.copy()
        true['predict'] = var_results.iloc[1:][['predict_diag1',
                                                'predict_diag2',
                                                'predict_diag3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_diag1',
                                                        'dyn_predict_diag2',
                                                        'dyn_predict_diag3']]
        super(TestVAR_diagonal, cls).setup_class(
            true,  order=(1, 0), trend='n',
            error_cov_type="diagonal")

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-5)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VAR\(1\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 8

            # -> Check that we have the right coefficients
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2],
                             table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 8

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%.4f' % (names[i], params[i]), table)


class TestVAR_measurement_error(CheckLutkepohl):
    """
    Notes
    -----
    There does not appear to be a way to get Stata to estimate a VAR with
    measurement errors. Thus this test is mostly a smoke test that measurement
    errors are setup correctly: it uses the same params from TestVAR_diagonal
    and sets the measurement errors variance params to zero to check that the
    loglike and predict are the same.

    It also checks that the state-space representation with positive
    measurement errors is correct.
    """
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag_meas.copy()
        true['predict'] = var_results.iloc[1:][['predict_diag1',
                                                'predict_diag2',
                                                'predict_diag3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_diag1',
                                                        'dyn_predict_diag2',
                                                        'dyn_predict_diag3']]
        super(TestVAR_measurement_error, cls).setup_class(
            true,  order=(1, 0), trend='n',
            error_cov_type="diagonal", measurement_error=True)

        # Create another filter results with positive measurement errors
        cls.true_measurement_error_variances = [1., 2., 3.]
        params = np.r_[true['params'][:-3],
                       cls.true_measurement_error_variances]
        cls.results2 = cls.model.smooth(params)

    def test_mle(self):
        # With the additional measurment error parameters, this would not be
        # a meaningful test
        pass

    def test_bse_approx(self):
        # This would just test the same thing
        # as TestVAR_diagonal.test_bse_approx
        pass

    def test_bse_oim(self):
        # This would just test the same thing as TestVAR_diagonal.test_bse_oim
        pass

    def test_aic(self):
        # Since the measurement error is added, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the measurement error is added, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_representation(self):
        # Test that the state space representation in the measurement error
        # case is correct
        for name in self.model.ssm.shapes.keys():
            if name == 'obs':
                pass
            elif name == 'obs_cov':
                actual = self.results2.filter_results.obs_cov
                desired = np.diag(
                    self.true_measurement_error_variances)[:, :, np.newaxis]
                assert_equal(actual, desired)
            else:
                assert_equal(getattr(self.results2.filter_results, name),
                             getattr(self.results.filter_results, name))

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VAR\(1\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 9

            # -> Check that we have the right coefficients
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2],
                             table)
            assert re.search('measurement_variance +%.4g' % params[-(i+1)],
                             table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 8

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%.4f' % (names[i], params[i]), table)


class TestVAR_obs_intercept(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_obs_intercept.copy()
        true['predict'] = var_results.iloc[1:][['predict_int1',
                                                'predict_int2',
                                                'predict_int3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_int1',
                                                        'dyn_predict_int2',
                                                        'dyn_predict_int3']]
        super(TestVAR_obs_intercept, cls).setup_class(
            true, order=(1, 0), trend='n',
            error_cov_type="diagonal", obs_intercept=true['obs_intercept'])

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-4)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_aic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass


class TestVAR_exog(CheckLutkepohl):
    # Note: unlike the other tests in this file, this is against the Stata
    # var function rather than the Stata dfactor function
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog.copy()
        true['predict'] = var_results.iloc[1:76][['predict_exog1_1',
                                                  'predict_exog1_2',
                                                  'predict_exog1_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.iloc[76:][['fcast_exog1_dln_inv',
                                               'fcast_exog1_dln_inc',
                                               'fcast_exog1_dln_consump']]
        exog = np.arange(75) + 2
        super(TestVAR_exog, cls).setup_class(
            true, order=(1, 0), trend='n', error_cov_type='unstructured',
            exog=exog, initialization='approximate_diffuse',
            loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        # Stata's var calculates AIC differently
        pass

    def test_bic(self):
        # Stata's var calculates BIC differently
        pass

    def test_bse_approx(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse[:-6]**2, self.true['var_oim'], atol=1e-5)

    def test_bse_oim(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[:-6]**2, self.true['var_oim'], atol=1e-5)

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=1e-3)

    def test_dynamic_predict(self):
        # Stata's var cannot subsequently use dynamic
        pass

    def test_forecast(self):
        # Tests forecast
        exog = (np.arange(75, 75+16) + 2)[:, np.newaxis]

        # Test it through the results class wrapper
        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-6)

        # Test it directly (i.e. without the wrapping done in
        # VARMAXResults.get_prediction which converts exog to state_intercept)
        # beta = self.results.params[-9:-6]
        # state_intercept = np.concatenate([
        #     exog*beta[0], exog*beta[1], exog*beta[2]], axis=1).T
        # desired = mlemodel.MLEResults.get_prediction(
        #     self.results._results, start=75, end=75+15,
        #     state_intercept=state_intercept).predicted_mean
        # assert_allclose(desired, self.true['fcast'], atol=1e-6)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VARX\(1\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 9

            # -> Check that we have the right coefficients
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search(
                'L1.dln_consump +%.4f' % params[offset + 2], table)
            assert re.search(
                'beta.x1 +' + forg(params[self.model._params_regression][i],
                                   prec=4),
                table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 11

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%.4f' % (names[i], params[i]), table)


class TestVAR_exog2(CheckLutkepohl):
    # This is a regression test, to make sure that the setup with multiple exog
    # works correctly. The params are from Stata, but the loglike is from
    # this model. Likely the small discrepancy (see the results file) is from
    # the approximate diffuse initialization.
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog2.copy()
        true['predict'] = var_results.iloc[1:76][['predict_exog2_1',
                                                  'predict_exog2_2',
                                                  'predict_exog2_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.iloc[76:][['fcast_exog2_dln_inv',
                                               'fcast_exog2_dln_inc',
                                               'fcast_exog2_dln_consump']]
        exog = np.c_[np.ones((75, 1)), (np.arange(75) + 2)[:, np.newaxis]]
        super(TestVAR_exog2, cls).setup_class(
            true, order=(1, 0), trend='n', error_cov_type='unstructured',
            exog=exog, initialization='approximate_diffuse',
            loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse_approx(self):
        pass

    def test_bse_oim(self):
        pass

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=1e-3)

    def test_dynamic_predict(self):
        # Stata's var cannot subsequently use dynamic
        pass

    def test_forecast(self):
        # Tests forecast
        exog = np.c_[np.ones((16, 1)),
                     (np.arange(75, 75+16) + 2)[:, np.newaxis]]

        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-6)


class TestVAR2(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var2.copy()
        true['predict'] = var_results.iloc[1:][['predict_var2_1',
                                                'predict_var2_2']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_var2_1',
                                                        'dyn_predict_var2_2']]
        super(TestVAR2, cls).setup_class(
            true, order=(2, 0), trend='n', error_cov_type='unstructured',
            included_vars=['dln_inv', 'dln_inc'])

    def test_bse_approx(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse[:-3]**2, self.true['var_oim'][:-3], atol=1e-5)

    def test_bse_oim(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[:-3]**2, self.true['var_oim'][:-3], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VAR\(2\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog * self.model.k_ar
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 9

            # -> Check that we have the right coefficients
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L2.dln_inv +%.4f' % params[offset + 2], table)
            assert re.search('L2.dln_inc +%.4f' % params[offset + 3], table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 8

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%.4f' % (names[i], params[i]), table)


class CheckFREDManufacturing(CheckVARMAX):
    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx',
                    **kwargs):
        cls.true = true
        # 1960:Q1 - 1982:Q4
        path = os.path.join(current_path, 'results', 'manufac.dta')
        with open(path, 'rb') as test_data:
            dta = pd.read_stata(test_data)
        dta.index = pd.DatetimeIndex(dta.month, freq='MS')
        dta['dlncaputil'] = dta['lncaputil'].diff()
        dta['dlnhours'] = dta['lnhours'].diff()

        endog = dta.loc['1972-02-01':, ['dlncaputil', 'dlnhours']]

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            cls.model = varmax.VARMAX(endog, order=order, trend=trend,
                                      error_cov_type=error_cov_type, **kwargs)

        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)


class TestVARMA(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_varma11.copy()
        true['predict'] = varmax_results.iloc[1:][['predict_varma11_1',
                                                   'predict_varma11_2']]
        true['dynamic_predict'] = varmax_results.iloc[1:][[
            'dyn_predict_varma11_1', 'dyn_predict_varma11_2']]

        super(TestVARMA, cls).setup_class(
              true, order=(1, 1), trend='n', error_cov_type='diagonal')

    def test_mle(self):
        # Since the VARMA model here is generic (we're just forcing zeros
        # in some params) whereas Stata's is restricted, the MLE test is not
        # meaninful
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_approx(self):
        # Standard errors do not match Stata's
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_oim(self):
        # Standard errors do not match Stata's
        pass

    def test_aic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_predict(self):
        super(TestVARMA, self).test_predict(end='2009-05-01', atol=1e-4)

    def test_dynamic_predict(self):
        super(TestVARMA, self).test_dynamic_predict(end='2009-05-01',
                                                    dynamic='2000-01-01')

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert re.search(r'Model:.*VARMA\(1,1\)', tables[0])

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_ar = i * self.model.k_endog
            offset_ma = (self.model.k_endog**2 * self.model.k_ar +
                         i * self.model.k_endog)
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)

            # -> Make sure it's the right size
            assert len(table.split('\n')) == 9

            # -> Check that we have the right coefficients
            assert re.search(
                'L1.dlncaputil +' + forg(params[offset_ar + 0], prec=4),
                table)
            assert re.search(
                'L1.dlnhours +' + forg(params[offset_ar + 1], prec=4),
                table)
            assert re.search(
                r'L1.e\(dlncaputil\) +' + forg(params[offset_ma + 0], prec=4),
                table)
            assert re.search(
                r'L1.e\(dlnhours\) +' + forg(params[offset_ma + 1], prec=4),
                table)

        # Test the error covariance matrix table
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 7

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('%s +%s' % (names[i], forg(params[i], prec=4)),
                             table)


class TestVMA1(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_vma1.copy()
        true['predict'] = varmax_results.iloc[1:][['predict_vma1_1',
                                                   'predict_vma1_2']]
        true['dynamic_predict'] = varmax_results.iloc[1:][[
            'dyn_predict_vma1_1', 'dyn_predict_vma1_2']]

        super(TestVMA1, cls).setup_class(
              true, order=(0, 1), trend='n', error_cov_type='diagonal')

    def test_mle(self):
        # Since the VARMA model here is generic (we're just forcing zeros
        # in some params) whereas Stata's is restricted, the MLE test is not
        # meaninful
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_approx(self):
        # Standard errors do not match Stata's
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_oim(self):
        # Standard errors do not match Stata's
        pass

    def test_aic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_predict(self):
        super(TestVMA1, self).test_predict(end='2009-05-01', atol=1e-4)

    def test_dynamic_predict(self):
        super(TestVMA1, self).test_dynamic_predict(end='2009-05-01',
                                                   dynamic='2000-01-01')


def test_specifications():
    # Tests for model specification and state space creation
    endog = np.arange(20).reshape(10, 2)
    exog = np.arange(10)
    exog2 = pd.Series(exog, index=pd.date_range('2000-01-01', '2009-01-01',
                                                freq='AS'))

    # Test successful model creation
    varmax.VARMAX(endog, exog=exog, order=(1, 0))

    # Test successful model creation with pandas exog
    varmax.VARMAX(endog, exog=exog2, order=(1, 0))


def test_misspecifications():
    varmax.__warningregistry__ = {}

    # Tests for model specification and misspecification exceptions
    endog = np.arange(20).reshape(10, 2)

    # Bad trend specification
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(1, 0), trend='')

    # Bad error_cov_type specification
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(1, 0), error_cov_type='')

    # Bad order specification
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(0, 0))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        varmax.VARMAX(endog, order=(1, 1))

    # Warning with VARMA specification
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        varmax.VARMAX(endog, order=(1, 1))

        message = ('Estimation of VARMA(p,q) models is not generically robust,'
                   ' due especially to identification issues.')
        assert str(w[0].message) == message

    warnings.resetwarnings()


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 2
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    endog[2:6, 1] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        varmax.VARMAX(endog, exog=exog1, order=(1, 0)),
        varmax.VARMAX(endog, exog=exog2, order=(1, 0)),
        varmax.VARMAX(endog_pd, exog=exog1_pd, order=(1, 0)),
        varmax.VARMAX(endog_pd, exog=exog2_pd, order=(1, 0)),
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
        with pytest.raises(ValueError):
            res.forecast(steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        with pytest.raises(ValueError):
            res.forecast(steps=1, exog=oos_exog)

    # Test invalid model specifications
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, exog=np.zeros((10, 4)), order=(1, 0))


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=(50, 2)))
    mod = varmax.VARMAX(endog, order=(1, 0))
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert out.index.equals(pd.Index(['a']))


def test_forecast_exog():
    # Test forecasting with various shapes of `exog`
    nobs = 100
    endog = np.ones((nobs, 2)) * 2.0
    exog = np.ones(nobs)

    mod = varmax.VARMAX(endog, order=(1, 0), exog=exog, trend='n')
    res = mod.smooth(np.r_[[0] * 4, 2.0, 2.0, 1, 0, 1])

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
        'order', 'trend', 'error_cov_type', 'measurement_error',
        'enforce_stationarity', 'enforce_invertibility', 'k_params']

    ssm_attrs = [
        'nobs', 'k_endog', 'k_states', 'k_posdef', 'obs_intercept', 'design',
        'obs_cov', 'state_intercept', 'transition', 'selection', 'state_cov']

    for attr in attrs:
        assert_equal(getattr(mod2, attr), getattr(mod, attr))

    for attr in ssm_attrs:
        assert_equal(getattr(mod2.ssm, attr), getattr(mod.ssm, attr))

    assert_equal(mod2._get_init_kwds(), mod._get_init_kwds())


def test_recreate_model():
    nobs = 100
    endog = np.ones((nobs, 3)) * 2.0
    exog = np.ones(nobs)

    orders = [(1, 0), (1, 1)]
    trends = ['t', 'n']
    error_cov_types = ['diagonal', 'unstructured']
    measurement_errors = [False, True]
    enforce_stationarities = [False, True]
    enforce_invertibilities = [False, True]

    import itertools
    names = ['order', 'trend', 'error_cov_type', 'measurement_error',
             'enforce_stationarity', 'enforce_invertibility']
    for element in itertools.product(orders, trends, error_cov_types,
                                     measurement_errors,
                                     enforce_stationarities,
                                     enforce_invertibilities):
        kwargs = dict(zip(names, element))

        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore')
            mod = varmax.VARMAX(endog, exog=exog, **kwargs)
            mod2 = varmax.VARMAX(endog, exog=exog, **mod._get_init_kwds())
        check_equivalent_models(mod, mod2)


def test_append_results():
    endog = np.arange(200).reshape(100, 2)
    exog = np.ones(100)
    params = [0.1, 0.2,
              0.5, -0.1, 0.0, 0.2,
              1., 2.,
              1., 0., 1.]

    mod1 = varmax.VARMAX(endog, order=(1, 0), trend='t', exog=exog)
    res1 = mod1.smooth(params)

    mod2 = varmax.VARMAX(endog[:50], order=(1, 0), trend='t', exog=exog[:50])
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


@pytest.mark.parametrize('trend', ['n', 'c', 'ct'])
@pytest.mark.parametrize('forecast', [True, False])
def test_extend_results(trend, forecast):
    endog = np.arange(200).reshape(100, 2)
    trend_params = []
    if trend == 'c':
        trend_params = [0.1, 0.2]
    if trend == 'ct':
        trend_params = [0.1, 0.2, 1., 2.]
    params = np.r_[trend_params,
                   0.5, -0.1, 0.0, 0.2,
                   1., 0., 1.]

    mod1 = varmax.VARMAX(endog, order=(1, 0), trend=trend)
    res1 = mod1.smooth(params)
    if forecast:
        # Call `forecast` to trigger the _set_final_exog and
        # _set_final_predicted_state context managers
        res1.forecast()

    mod2 = mod1.clone(endog[:50])
    res2 = mod2.smooth(params)
    if forecast:
        # Call `forecast` to trigger the _set_final_exog and
        # _set_final_predicted_state context managers
        res2.forecast()
    res3 = res2.extend(endog[50:])

    assert_allclose(res3.llf_obs, res1.llf_obs[50:])

    for attr in [
            'filtered_state', 'filtered_state_cov', 'predicted_state',
            'predicted_state_cov', 'forecasts', 'forecasts_error',
            'forecasts_error_cov', 'standardized_forecasts_error',
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
        assert_allclose(getattr(res3, attr), desired, atol=1e-12)

    assert_allclose(res3.forecast(10), res1.forecast(10))


def test_extend_results_exog():
    endog = np.arange(200).reshape(100, 2)
    exog = np.ones(100)
    params = [0.1, 0.2,
              0.5, -0.1, 0.0, 0.2,
              1., 2.,
              1., 0., 1.]

    mod1 = varmax.VARMAX(endog, order=(1, 0), trend='t', exog=exog)
    res1 = mod1.smooth(params)

    mod2 = varmax.VARMAX(endog[:50], order=(1, 0), trend='t', exog=exog[:50])
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


def test_apply_results():
    endog = np.arange(200).reshape(100, 2)
    exog = np.ones(100)
    params = [0.1, 0.2,
              0.5, -0.1, 0.0, 0.2,
              1., 2.,
              1., 0., 1.]

    mod1 = varmax.VARMAX(endog[:50], order=(1, 0), trend='t', exog=exog[:50])
    res1 = mod1.smooth(params)

    mod2 = varmax.VARMAX(endog[50:], order=(1, 0), trend='t', exog=exog[50:])
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


def test_vma1_exog():
    # Test the VMAX(1) case against univariate MAX(1) models
    dta = pd.DataFrame(
        results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
        index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
    dta = np.log(dta).diff().iloc[1:]

    endog = dta.iloc[:, :2]
    exog = dta.iloc[:, 2]

    ma_params1 = [-0.01, 1.4, -0.3, 0.002]
    ma_params2 = [0.004, 0.8, -0.5, 0.0001]

    vma_params = [ma_params1[0], ma_params2[0],
                  ma_params1[2], 0,
                  0, ma_params2[2],
                  ma_params1[1], ma_params2[1],
                  ma_params1[3], ma_params2[3]]

    # Joint VMA model
    mod_vma = varmax.VARMAX(endog, exog=exog, order=(0, 1),
                            error_cov_type='diagonal')
    mod_vma.ssm.initialize_diffuse()
    res_mva = mod_vma.smooth(vma_params)

    # Smoke test that start_params does not raise an error
    sp = mod_vma.start_params
    assert_equal(len(sp), len(mod_vma.param_names))

    # Univariate MA models
    mod_ma1 = sarimax.SARIMAX(endog.iloc[:, 0], exog=exog, order=(0, 0, 1),
                              trend='c')
    mod_ma1.ssm.initialize_diffuse()
    mod_ma2 = sarimax.SARIMAX(endog.iloc[:, 1], exog=exog, order=(0, 0, 1),
                              trend='c')
    mod_ma2.ssm.initialize_diffuse()
    res_ma1 = mod_ma1.smooth(ma_params1)
    res_ma2 = mod_ma2.smooth(ma_params2)

    # Have to ignore first 2 observations due to differences in initialization
    assert_allclose(res_mva.llf_obs[2:],
                    (res_ma1.llf_obs + res_ma2.llf_obs)[2:])


def test_param_names_trend():
    endog = np.zeros((3, 2))
    base_names = ['L1.y1.y1', 'L1.y2.y1', 'L1.y1.y2', 'L1.y2.y2',
                  'sqrt.var.y1', 'sqrt.cov.y1.y2', 'sqrt.var.y2']
    base_params = [0.5, 0, 0, 0.4, 1.0, 0.0, 1.0]

    # No trend
    mod = varmax.VARMAX(endog, order=(1, 0), trend='n')
    desired = base_names
    assert_equal(mod.param_names, desired)

    # Intercept
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1])
    desired = ['intercept.y1', 'intercept.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1.2, -0.5] + base_params)
    assert_allclose(mod['state_intercept'], [1.2, -0.5])

    # Intercept + drift
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1, 1])
    desired = (['intercept.y1', 'drift.y1',
                'intercept.y2', 'drift.y2'] + base_names)
    assert_equal(mod.param_names, desired)
    mod.update([1.2, 0, -0.5, 0] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2)
    assert_allclose(mod['state_intercept', 1], -0.5)
    mod.update([0, 1, 0, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5))
    mod.update([1.2, 1, -0.5, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2 + np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], -0.5 + 1.1 * np.arange(2, 5))

    # Drift only
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[0, 1])
    desired = ['drift.y1', 'drift.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5))

    # Intercept + third order
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1, 0, 1])
    desired = (['intercept.y1', 'trend.2.y1',
                'intercept.y2', 'trend.2.y2'] + base_names)
    assert_equal(mod.param_names, desired)
    mod.update([1.2, 0, -0.5, 0] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2)
    assert_allclose(mod['state_intercept', 1], -0.5)
    mod.update([0, 1, 0, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5)**2)
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5)**2)
    mod.update([1.2, 1, -0.5, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2 + np.arange(2, 5)**2)
    assert_allclose(mod['state_intercept', 1], -0.5 + 1.1 * np.arange(2, 5)**2)
