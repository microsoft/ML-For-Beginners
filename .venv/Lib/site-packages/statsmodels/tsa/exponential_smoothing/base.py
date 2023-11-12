from collections import OrderedDict
import contextlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from statsmodels.base.data import PandasData
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
from statsmodels.tools.sm_exceptions import PrecisionWarning
from statsmodels.tools.numdiff import (
    _get_epsilon,
    approx_fprime,
    approx_fprime_cs,
    approx_hess_cs,
)
from statsmodels.tools.tools import pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.statespace.tools import _safe_cond


class StateSpaceMLEModel(tsbase.TimeSeriesModel):
    """
    This is a temporary base model from ETS, here I just copy everything I need
    from statespace.mlemodel.MLEModel
    """

    def __init__(
        self, endog, exog=None, dates=None, freq=None, missing="none", **kwargs
    ):
        # TODO: this was changed from the original, requires some work when
        # using this as base class for state space and exponential smoothing
        super().__init__(
            endog=endog, exog=exog, dates=dates, freq=freq, missing=missing
        )

        # Store kwargs to recreate model
        self._init_kwargs = kwargs

        # Prepared the endog array: C-ordered, shape=(nobs x k_endog)
        self.endog, self.exog = self.prepare_data(self.data)
        self.use_pandas = isinstance(self.data, PandasData)

        # Dimensions
        self.nobs = self.endog.shape[0]

        # Setup holder for fixed parameters
        self._has_fixed_params = False
        self._fixed_params = None
        self._params_index = None
        self._fixed_params_index = None
        self._free_params_index = None

    @staticmethod
    def prepare_data(data):
        raise NotImplementedError

    def clone(self, endog, exog=None, **kwargs):
        raise NotImplementedError

    def _validate_can_fix_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError(
                    'Invalid parameter name passed: "%s".' % param_name
                )

    @property
    def k_params(self):
        return len(self.param_names)

    @contextlib.contextmanager
    def fix_params(self, params):
        """
        Fix parameters to specific values (context manager)

        Parameters
        ----------
        params : dict
            Dictionary describing the fixed parameter values, of the form
            `param_name: fixed_value`. See the `param_names` property for valid
            parameter names.

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> with mod.fix_params({'ar.L1': 0.5}):
                res = mod.fit()
        """
        # Initialization (this is done here rather than in the constructor
        # because param_names may not be available at that point)
        if self._fixed_params is None:
            self._fixed_params = {}
            self._params_index = OrderedDict(
                zip(self.param_names, np.arange(self.k_params))
            )

        # Cache the current fixed parameters
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index

        # Validate parameter names and values
        all_fixed_param_names = (
            set(params.keys()) | set(self._fixed_params.keys())
        )
        self._validate_can_fix_params(all_fixed_param_names)

        # Set the new fixed parameters, keeping the order as given by
        # param_names
        self._fixed_params.update(params)
        self._fixed_params = OrderedDict(
            [
                (name, self._fixed_params[name])
                for name in self.param_names
                if name in self._fixed_params
            ]
        )

        # Update associated values
        self._has_fixed_params = True
        self._fixed_params_index = [
            self._params_index[key] for key in self._fixed_params.keys()
        ]
        self._free_params_index = list(
            set(np.arange(self.k_params)).difference(self._fixed_params_index)
        )

        try:
            yield
        finally:
            # Reset the fixed parameters
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """
        Fit the model with some parameters subject to equality constraints.

        Parameters
        ----------
        constraints : dict
            Dictionary of constraints, of the form `param_name: fixed_value`.
            See the `param_names` property for valid parameter names.
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the remaining parameters.

        Returns
        -------
        results : Results instance

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> res = mod.fit_constrained({'ar.L1': 0.5})
        """
        with self.fix_params(constraints):
            res = self.fit(start_params, **fit_kwds)
        return res

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        if hasattr(self, "_start_params"):
            return self._start_params
        else:
            raise NotImplementedError

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        if hasattr(self, "_param_names"):
            return self._param_names
        else:
            try:
                names = ["param.%d" % i for i in range(len(self.start_params))]
            except NotImplementedError:
                names = []
            return names

    @classmethod
    def from_formula(
        cls, formula, data, subset=None, drop_cols=None, *args, **kwargs
    ):
        """
        Not implemented for state space models
        """
        raise NotImplementedError

    def _wrap_data(self, data, start_idx, end_idx, names=None):
        # TODO: check if this is reasonable for statespace
        # squeezing data: data may be:
        # - m x n: m dates, n simulations -> squeeze does nothing
        # - m x 1: m dates, 1 simulation -> squeeze removes last dimension
        # - 1 x n: don't squeeze, already fine
        # - 1 x 1: squeeze only second axis
        if data.ndim > 1 and data.shape[1] == 1:
            data = np.squeeze(data, axis=1)
        if self.use_pandas:
            if data.shape[0]:
                _, _, _, index = self._get_prediction_index(start_idx, end_idx)
            else:
                index = None
            if data.ndim < 2:
                data = pd.Series(data, index=index, name=names)
            else:
                data = pd.DataFrame(data, index=index, columns=names)
        return data

    def _wrap_results(
        self,
        params,
        result,
        return_raw,
        cov_type=None,
        cov_kwds=None,
        results_class=None,
        wrapper_class=None,
    ):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs["cov_type"] = cov_type
            if cov_kwds is not None:
                result_kwargs["cov_kwds"] = cov_kwds

            if results_class is None:
                results_class = self._res_classes["fit"][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes["fit"][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def _score_complex_step(self, params, **kwargs):
        # the default epsilon can be too small
        # inversion_method = INVERT_UNIVARIATE | SOLVE_LU
        epsilon = _get_epsilon(params, 2., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        return approx_fprime_cs(params, self.loglike, epsilon=epsilon,
                                kwargs=kwargs)

    def _score_finite_difference(self, params, approx_centered=False,
                                 **kwargs):
        kwargs['transformed'] = True
        return approx_fprime(params, self.loglike, kwargs=kwargs,
                             centered=approx_centered)

    def _hessian_finite_difference(self, params, approx_centered=False,
                                   **kwargs):
        params = np.array(params, ndmin=1)

        warnings.warn('Calculation of the Hessian using finite differences'
                      ' is usually subject to substantial approximation'
                      ' errors.',
                      PrecisionWarning,
                      stacklevel=3,
                      )

        if not approx_centered:
            epsilon = _get_epsilon(params, 3, None, len(params))
        else:
            epsilon = _get_epsilon(params, 4, None, len(params)) / 2
        hessian = approx_fprime(params, self._score_finite_difference,
                                epsilon=epsilon, kwargs=kwargs,
                                centered=approx_centered)

        # TODO: changed this to nobs_effective, has to be changed when merging
        # with statespace mlemodel
        return hessian / (self.nobs_effective)

    def _hessian_complex_step(self, params, **kwargs):
        """
        Hessian matrix computed by second-order complex-step differentiation
        on the `loglike` function.
        """
        # the default epsilon can be too small
        epsilon = _get_epsilon(params, 3., None, len(params))
        kwargs['transformed'] = True
        kwargs['complex_step'] = True
        hessian = approx_hess_cs(
            params, self.loglike, epsilon=epsilon, kwargs=kwargs)

        # TODO: changed this to nobs_effective, has to be changed when merging
        # with statespace mlemodel
        return hessian / (self.nobs_effective)


class StateSpaceMLEResults(tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : ndarray
        Fitted parameters

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    """

    def __init__(self, model, params, scale=1.0):
        self.data = model.data
        self.endog = model.data.orig_endog

        super().__init__(model, params, None, scale=scale)

        # Save the fixed parameters
        self._has_fixed_params = self.model._has_fixed_params
        self._fixed_params_index = self.model._fixed_params_index
        self._free_params_index = self.model._free_params_index
        # TODO: seems like maybe self.fixed_params should be the dictionary
        # itself, not just the keys?
        if self._has_fixed_params:
            self._fixed_params = self.model._fixed_params.copy()
            self.fixed_params = list(self._fixed_params.keys())
        else:
            self._fixed_params = None
            self.fixed_params = []
        self.param_names = [
            "%s (fixed)" % name if name in self.fixed_params else name
            for name in (self.data.param_names or [])
        ]

        # Dimensions
        self.nobs = self.model.nobs
        self.k_params = self.model.k_params

        self._rank = None

    @cache_readonly
    def nobs_effective(self):
        raise NotImplementedError

    @cache_readonly
    def df_resid(self):
        return self.nobs_effective - self.df_model

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        return aic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def aicc(self):
        """
        (float) Akaike Information Criterion with small sample correction
        """
        return aicc(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        return bic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def fittedvalues(self):
        # TODO
        raise NotImplementedError

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return (-2 * self.llf +
        #         2 * np.log(np.log(self.nobs_effective)) * self.df_model)
        return hqic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        raise NotImplementedError

    @cache_readonly
    def mae(self):
        """
        (float) Mean absolute error
        """
        return np.mean(np.abs(self.resid))

    @cache_readonly
    def mse(self):
        """
        (float) Mean squared error
        """
        return self.sse / self.nobs

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        pvalues = np.zeros_like(self.zvalues) * np.nan
        mask = np.ones_like(pvalues, dtype=bool)
        mask[self._free_params_index] = True
        mask &= ~np.isnan(self.zvalues)
        pvalues[mask] = norm.sf(np.abs(self.zvalues[mask])) * 2
        return pvalues

    @cache_readonly
    def resid(self):
        raise NotImplementedError

    @cache_readonly
    def sse(self):
        """
        (float) Sum of squared errors
        """
        return np.sum(self.resid ** 2)

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def _get_prediction_start_index(self, anchor):
        """Returns a valid numeric start index for predictions/simulations"""
        if anchor is None or anchor == "start":
            iloc = 0
        elif anchor == "end":
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start
            iloc += 1  # anchor is one before start of prediction/simulation

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError("Cannot anchor simulation outside of the sample.")
        return iloc

    def _cov_params_approx(
        self, approx_complex_step=True, approx_centered=False
    ):
        evaluated_hessian = self.nobs_effective * self.model.hessian(
            params=self.params,
            transformed=True,
            includes_fixed=True,
            method="approx",
            approx_complex_step=approx_complex_step,
            approx_centered=approx_centered,
        )
        # TODO: Case with "not approx_complex_step" is not hit in
        # tests as of 2017-05-19

        if len(self.fixed_params) > 0:
            mask = np.ix_(self._free_params_index, self._free_params_index)
            if len(self.fixed_params) < self.k_params:
                (tmp, singular_values) = pinv_extended(evaluated_hessian[mask])
            else:
                tmp, singular_values = np.nan, [np.nan]
            neg_cov = np.zeros_like(evaluated_hessian) * np.nan
            neg_cov[mask] = tmp
        else:
            (neg_cov, singular_values) = pinv_extended(evaluated_hessian)

        self.model.update(self.params, transformed=True, includes_fixed=True)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        return self._cov_params_approx(
            self._cov_approx_complex_step, self._cov_approx_centered
        )

    def test_serial_correlation(self, method, lags=None):
        """
        Ljung-Box test for no serial correlation of standardized residuals

        Null hypothesis is no serial correlation.

        Parameters
        ----------
        method : {'ljungbox','boxpierece', None}
            The statistical test for serial correlation. If None, an attempt is
            made to select an appropriate test.
        lags : None, int or array_like
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length.
            If lags is a list or array, then all lags are included up to the
            largest lag in the list, however only the tests for the lags in the
            list are reported.
            If lags is None, then the default maxlag is min(10, nobs//5) for
            non-seasonal time series and min (2*m, nobs//5) for seasonal time
            series.

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable and each lag. The array is then sized
            `(k_endog, 2, lags)`. If the method is called as
            `ljungbox = res.test_serial_correlation()`, then `ljungbox[i]`
            holds the results of the Ljung-Box test (as would be returned by
            `statsmodels.stats.diagnostic.acorr_ljungbox`) for the `i` th
            endogenous variable.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox
            Ljung-Box test for serial correlation.

        Notes
        -----
        For statespace models: let `d` = max(loglikelihood_burn, nobs_diffuse);
        this test is calculated ignoring the first `d` residuals.

        Output is nan for any endogenous variable which has missing values.
        """
        if method is None:
            method = 'ljungbox'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'ljungbox' or method == 'boxpierce':
            from statsmodels.stats.diagnostic import acorr_ljungbox
            if hasattr(self, "loglikelihood_burn"):
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
                # This differs from self.nobs_effective because here we want to
                # exclude exact diffuse periods, whereas self.nobs_effective
                # only excludes explicitly burned (usually approximate diffuse)
                # periods.
                nobs_effective = self.nobs - d
            else:
                nobs_effective = self.nobs_effective
            output = []

            # Default lags for acorr_ljungbox is 40, but may not always have
            # that many observations
            if lags is None:
                seasonal_periods = getattr(self.model, "seasonal_periods", 0)
                if seasonal_periods:
                    lags = min(2 * seasonal_periods, nobs_effective // 5)
                else:
                    lags = min(10, nobs_effective // 5)

            cols = [2, 3] if method == 'boxpierce' else [0, 1]
            for i in range(self.model.k_endog):
                if hasattr(self, "filter_results"):
                    x = self.filter_results.standardized_forecasts_error[i][d:]
                else:
                    x = self.standardized_forecasts_error
                results = acorr_ljungbox(
                    x, lags=lags, boxpierce=(method == 'boxpierce')
                )
                output.append(np.asarray(results)[:, cols].T)

            output = np.c_[output]
        else:
            raise NotImplementedError('Invalid serial correlation test'
                                      ' method.')
        return output

    def test_heteroskedasticity(self, method, alternative='two-sided',
                                use_f=True):
        r"""
        Test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
        is of no heteroskedasticity.

        Parameters
        ----------
        method : {'breakvar', None}
            The statistical test for heteroskedasticity. Must be 'breakvar'
            for test of a break in the variance. If None, an attempt is
            made to select an appropriate test.
        alternative : str, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        use_f : bool, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is True (i.e. default is to compare against an F
            distribution).

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        Notes
        -----
        The null hypothesis is of no heteroskedasticity. That means different
        things depending on which alternative is selected:

        - Increasing: Null hypothesis is that the variance is not increasing
          throughout the sample; that the sum-of-squares in the later
          subsample is *not* greater than the sum-of-squares in the earlier
          subsample.
        - Decreasing: Null hypothesis is that the variance is not decreasing
          throughout the sample; that the sum-of-squares in the earlier
          subsample is *not* greater than the sum-of-squares in the later
          subsample.
        - Two-sided: Null hypothesis is that the variance is not changing
          throughout the sample. Both that the sum-of-squares in the earlier
          subsample is not greater than the sum-of-squares in the later
          subsample *and* that the sum-of-squares in the later subsample is
          not greater than the sum-of-squares in the earlier subsample.

        For :math:`h = [T/3]`, the test statistic is:

        .. math::

            H(h) = \sum_{t=T-h+1}^T  \tilde v_t^2
            \Bigg / \sum_{t=d+1}^{d+1+h} \tilde v_t^2

        where :math:`d` = max(loglikelihood_burn, nobs_diffuse)` (usually
        corresponding to diffuse initialization under either the approximate
        or exact approach).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\chi_h^2`; this second test can be applied by passing
        `asymptotic=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        References
        ----------
        .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
               *Models and the Kalman Filter.* Cambridge University Press.
        """
        if method is None:
            method = 'breakvar'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'breakvar':
            # Store some values
            if hasattr(self, "filter_results"):
                squared_resid = (
                    self.filter_results.standardized_forecasts_error**2
                )
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
                # This differs from self.nobs_effective because here we want to
                # exclude exact diffuse periods, whereas self.nobs_effective
                # only excludes explicitly burned (usually approximate diffuse)
                # periods.
                nobs_effective = self.nobs - d
            else:
                squared_resid = self.standardized_forecasts_error**2
                if squared_resid.ndim == 1:
                    squared_resid = np.asarray(squared_resid)
                    squared_resid = squared_resid[np.newaxis, :]
                nobs_effective = self.nobs_effective
                d = 0
            squared_resid = np.asarray(squared_resid)

            test_statistics = []
            p_values = []
            for i in range(self.model.k_endog):
                h = int(np.round(nobs_effective / 3))
                numer_resid = squared_resid[i, -h:]
                numer_resid = numer_resid[~np.isnan(numer_resid)]
                numer_dof = len(numer_resid)

                denom_resid = squared_resid[i, d:d + h]
                denom_resid = denom_resid[~np.isnan(denom_resid)]
                denom_dof = len(denom_resid)

                if numer_dof < 2:
                    warnings.warn('Early subset of data for variable %d'
                                  '  has too few non-missing observations to'
                                  ' calculate test statistic.' % i,
                                  stacklevel=2,
                                  )
                    numer_resid = np.nan
                if denom_dof < 2:
                    warnings.warn('Later subset of data for variable %d'
                                  '  has too few non-missing observations to'
                                  ' calculate test statistic.' % i,
                                  stacklevel=2,
                                  )
                    denom_resid = np.nan

                test_statistic = np.sum(numer_resid) / np.sum(denom_resid)

                # Setup functions to calculate the p-values
                if use_f:
                    from scipy.stats import f
                    pval_lower = lambda test_statistics: f.cdf(  # noqa:E731
                        test_statistics, numer_dof, denom_dof)
                    pval_upper = lambda test_statistics: f.sf(  # noqa:E731
                        test_statistics, numer_dof, denom_dof)
                else:
                    from scipy.stats import chi2
                    pval_lower = lambda test_statistics: chi2.cdf(  # noqa:E731
                        numer_dof * test_statistics, denom_dof)
                    pval_upper = lambda test_statistics: chi2.sf(  # noqa:E731
                        numer_dof * test_statistics, denom_dof)

                # Calculate the one- or two-sided p-values
                alternative = alternative.lower()
                if alternative in ['i', 'inc', 'increasing']:
                    p_value = pval_upper(test_statistic)
                elif alternative in ['d', 'dec', 'decreasing']:
                    test_statistic = 1. / test_statistic
                    p_value = pval_upper(test_statistic)
                elif alternative in ['2', '2-sided', 'two-sided']:
                    p_value = 2 * np.minimum(
                        pval_lower(test_statistic),
                        pval_upper(test_statistic)
                    )
                else:
                    raise ValueError('Invalid alternative.')

                test_statistics.append(test_statistic)
                p_values.append(p_value)

            output = np.c_[test_statistics, p_values]
        else:
            raise NotImplementedError('Invalid heteroskedasticity test'
                                      ' method.')

        return output

    def test_normality(self, method):
        """
        Test for normality of standardized residuals.

        Null hypothesis is normality.

        Parameters
        ----------
        method : {'jarquebera', None}
            The statistical test for normality. Must be 'jarquebera' for
            Jarque-Bera normality test. If None, an attempt is made to select
            an appropriate test.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera
            The Jarque-Bera test of normality.

        Notes
        -----
        For statespace models: let `d` = max(loglikelihood_burn, nobs_diffuse);
        this test is calculated ignoring the first `d` residuals.

        In the case of missing data, the maintained hypothesis is that the
        data are missing completely at random. This test is then run on the
        standardized residuals excluding those corresponding to missing
        observations.
        """
        if method is None:
            method = 'jarquebera'

        if self.standardized_forecasts_error is None:
            raise ValueError('Cannot compute test statistic when standardized'
                             ' forecast errors have not been computed.')

        if method == 'jarquebera':
            from statsmodels.stats.stattools import jarque_bera
            if hasattr(self, "loglikelihood_burn"):
                d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
            else:
                d = 0
            output = []
            for i in range(self.model.k_endog):
                if hasattr(self, "fiter_results"):
                    resid = self.filter_results.standardized_forecasts_error[
                        i, d:
                    ]
                else:
                    resid = self.standardized_forecasts_error
                mask = ~np.isnan(resid)
                output.append(jarque_bera(resid[mask]))
        else:
            raise NotImplementedError('Invalid normality test method.')

        return np.array(output)

    def summary(
        self,
        alpha=0.05,
        start=None,
        title=None,
        model_name=None,
        display_params=True,
    ):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        model_name : str
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary

        # Model specification results
        model = self.model
        if title is None:
            title = "Statespace Model Results"

        if start is None:
            start = 0
        if self.model._index_dates:
            ix = self.model._index
            d = ix[start]
            sample = ["%02d-%02d-%02d" % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ["- " + "%02d-%02d-%02d" % (d.month, d.day, d.year)]
        else:
            sample = [str(start), " - " + str(self.nobs)]

        # Standardize the model name as a list of str
        if model_name is None:
            model_name = model.__class__.__name__

        # Diagnostic tests results
        try:
            het = self.test_heteroskedasticity(method="breakvar")
        except Exception:  # FIXME: catch something specific
            het = np.array([[np.nan] * 2])
        try:
            lb = self.test_serial_correlation(method="ljungbox")
        except Exception:  # FIXME: catch something specific
            lb = np.array([[np.nan] * 2]).reshape(1, 2, 1)
        try:
            jb = self.test_normality(method="jarquebera")
        except Exception:  # FIXME: catch something specific
            jb = np.array([[np.nan] * 4])

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [("Dep. Variable:", None)]
        top_left.append(("Model:", [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(("", ["+ " + model_name[i]]))
        top_left += [
            ("Date:", None),
            ("Time:", None),
            ("Sample:", [sample[0]]),
            ("", [sample[1]]),
        ]

        top_right = [
            ("No. Observations:", [self.nobs]),
            ("Log Likelihood", ["%#5.3f" % self.llf]),
        ]
        if hasattr(self, "rsquared"):
            top_right.append(("R-squared:", ["%#8.3f" % self.rsquared]))
        top_right += [
            ("AIC", ["%#5.3f" % self.aic]),
            ("BIC", ["%#5.3f" % self.bic]),
            ("HQIC", ["%#5.3f" % self.hqic]),
        ]

        if hasattr(self, "filter_results"):
            if (
                    self.filter_results is not None
                    and self.filter_results.filter_concentrated
            ):
                top_right.append(("Scale", ["%#5.3f" % self.scale]))
        else:
            top_right.append(("Scale", ["%#5.3f" % self.scale]))

        if hasattr(self, "cov_type"):
            top_left.append(("Covariance Type:", [self.cov_type]))

        format_str = lambda array: [  # noqa:E731
            ", ".join(["{0:.2f}".format(i) for i in array])
        ]
        diagn_left = [
            ("Ljung-Box (Q):", format_str(lb[:, 0, -1])),
            ("Prob(Q):", format_str(lb[:, 1, -1])),
            ("Heteroskedasticity (H):", format_str(het[:, 0])),
            ("Prob(H) (two-sided):", format_str(het[:, 1])),
        ]

        diagn_right = [
            ("Jarque-Bera (JB):", format_str(jb[:, 0])),
            ("Prob(JB):", format_str(jb[:, 1])),
            ("Skew:", format_str(jb[:, 2])),
            ("Kurtosis:", format_str(jb[:, 3])),
        ]

        summary = Summary()
        summary.add_table_2cols(
            self, gleft=top_left, gright=top_right, title=title
        )
        if len(self.params) > 0 and display_params:
            summary.add_table_params(
                self, alpha=alpha, xname=self.param_names, use_t=False
            )
        summary.add_table_2cols(
            self, gleft=diagn_left, gright=diagn_right, title=""
        )

        # Add warnings/notes, added to text format only
        etext = []
        if hasattr(self, "cov_type") and "description" in self.cov_kwds:
            etext.append(self.cov_kwds["description"])
        if self._rank < (len(self.params) - len(self.fixed_params)):
            cov_params = self.cov_params()
            if len(self.fixed_params) > 0:
                mask = np.ix_(self._free_params_index, self._free_params_index)
                cov_params = cov_params[mask]
            etext.append(
                "Covariance matrix is singular or near-singular,"
                " with condition number %6.3g. Standard errors may be"
                " unstable." % _safe_cond(cov_params)
            )

        if etext:
            etext = [
                "[{0}] {1}".format(i + 1, text) for i, text in enumerate(etext)
            ]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)

        return summary
