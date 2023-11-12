"""
SARIMAX parameters class.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic


class SARIMAXParams:
    """
    SARIMAX parameters.

    Parameters
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.

    Attributes
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.
    exog_names : list of str
        Names associated with exogenous parameters.
    ar_names : list of str
        Names associated with (non-seasonal) autoregressive parameters.
    ma_names : list of str
        Names associated with (non-seasonal) moving average parameters.
    seasonal_ar_names : list of str
        Names associated with seasonal autoregressive parameters.
    seasonal_ma_names : list of str
        Names associated with seasonal moving average parameters.
    param_names :list of str
        Names of all model parameters.
    k_exog_params : int
        Number of parameters associated with exogenous variables.
    k_ar_params : int
        Number of parameters associated with (non-seasonal) autoregressive
        lags.
    k_ma_params : int
        Number of parameters associated with (non-seasonal) moving average
        lags.
    k_seasonal_ar_params : int
        Number of parameters associated with seasonal autoregressive lags.
    k_seasonal_ma_params : int
        Number of parameters associated with seasonal moving average lags.
    k_params : int
        Total number of model parameters.
    """

    def __init__(self, spec):
        self.spec = spec

        # Local copies of relevant attributes
        self.exog_names = spec.exog_names
        self.ar_names = spec.ar_names
        self.ma_names = spec.ma_names
        self.seasonal_ar_names = spec.seasonal_ar_names
        self.seasonal_ma_names = spec.seasonal_ma_names
        self.param_names = spec.param_names

        self.k_exog_params = spec.k_exog_params
        self.k_ar_params = spec.k_ar_params
        self.k_ma_params = spec.k_ma_params
        self.k_seasonal_ar_params = spec.k_seasonal_ar_params
        self.k_seasonal_ma_params = spec.k_seasonal_ma_params
        self.k_params = spec.k_params

        # Cache for holding parameter values
        self._params_split = spec.split_params(
            np.zeros(self.k_params) * np.nan, allow_infnan=True)
        self._params = None

    @property
    def exog_params(self):
        """(array) Parameters associated with exogenous variables."""
        return self._params_split['exog_params']

    @exog_params.setter
    def exog_params(self, value):
        if np.isscalar(value):
            value = [value] * self.k_exog_params
        self._params_split['exog_params'] = validate_basic(
            value, self.k_exog_params, title='exogenous coefficients')
        self._params = None

    @property
    def ar_params(self):
        """(array) Autoregressive (non-seasonal) parameters."""
        return self._params_split['ar_params']

    @ar_params.setter
    def ar_params(self, value):
        if np.isscalar(value):
            value = [value] * self.k_ar_params
        self._params_split['ar_params'] = validate_basic(
            value, self.k_ar_params, title='AR coefficients')
        self._params = None

    @property
    def ar_poly(self):
        """(Polynomial) Autoregressive (non-seasonal) lag polynomial."""
        coef = np.zeros(self.spec.max_ar_order + 1)
        coef[0] = 1
        ix = self.spec.ar_lags
        coef[ix] = -self._params_split['ar_params']
        return Polynomial(coef)

    @ar_poly.setter
    def ar_poly(self, value):
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, self.spec.max_ar_order + 1,
                               title='AR polynomial')
        if value[0] != 1:
            raise ValueError('AR polynomial constant must be equal to 1.')
        ar_params = []
        for i in range(1, self.spec.max_ar_order + 1):
            if i in self.spec.ar_lags:
                ar_params.append(-value[i])
            elif value[i] != 0:
                raise ValueError('AR polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.ar_params = ar_params

    @property
    def ma_params(self):
        """(array) Moving average (non-seasonal) parameters."""
        return self._params_split['ma_params']

    @ma_params.setter
    def ma_params(self, value):
        if np.isscalar(value):
            value = [value] * self.k_ma_params
        self._params_split['ma_params'] = validate_basic(
            value, self.k_ma_params, title='MA coefficients')
        self._params = None

    @property
    def ma_poly(self):
        """(Polynomial) Moving average (non-seasonal) lag polynomial."""
        coef = np.zeros(self.spec.max_ma_order + 1)
        coef[0] = 1
        ix = self.spec.ma_lags
        coef[ix] = self._params_split['ma_params']
        return Polynomial(coef)

    @ma_poly.setter
    def ma_poly(self, value):
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, self.spec.max_ma_order + 1,
                               title='MA polynomial')
        if value[0] != 1:
            raise ValueError('MA polynomial constant must be equal to 1.')
        ma_params = []
        for i in range(1, self.spec.max_ma_order + 1):
            if i in self.spec.ma_lags:
                ma_params.append(value[i])
            elif value[i] != 0:
                raise ValueError('MA polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.ma_params = ma_params

    @property
    def seasonal_ar_params(self):
        """(array) Seasonal autoregressive parameters."""
        return self._params_split['seasonal_ar_params']

    @seasonal_ar_params.setter
    def seasonal_ar_params(self, value):
        if np.isscalar(value):
            value = [value] * self.k_seasonal_ar_params
        self._params_split['seasonal_ar_params'] = validate_basic(
            value, self.k_seasonal_ar_params, title='seasonal AR coefficients')
        self._params = None

    @property
    def seasonal_ar_poly(self):
        """(Polynomial) Seasonal autoregressive lag polynomial."""
        # Need to expand the polynomial according to the season
        s = self.spec.seasonal_periods
        coef = [1]
        if s > 0:
            expanded = np.zeros(self.spec.max_seasonal_ar_order)
            ix = np.array(self.spec.seasonal_ar_lags, dtype=int) - 1
            expanded[ix] = -self._params_split['seasonal_ar_params']
            coef = np.r_[1, np.pad(np.reshape(expanded, (-1, 1)),
                                   [(0, 0), (s - 1, 0)], 'constant').flatten()]
        return Polynomial(coef)

    @seasonal_ar_poly.setter
    def seasonal_ar_poly(self, value):
        s = self.spec.seasonal_periods
        # Note: assume that we are given coefficients from the full polynomial
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, 1 + s * self.spec.max_seasonal_ar_order,
                               title='seasonal AR polynomial')
        if value[0] != 1:
            raise ValueError('Polynomial constant must be equal to 1.')
        seasonal_ar_params = []
        for i in range(1, self.spec.max_seasonal_ar_order + 1):
            if i in self.spec.seasonal_ar_lags:
                seasonal_ar_params.append(-value[s * i])
            elif value[s * i] != 0:
                raise ValueError('AR polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.seasonal_ar_params = seasonal_ar_params

    @property
    def seasonal_ma_params(self):
        """(array) Seasonal moving average parameters."""
        return self._params_split['seasonal_ma_params']

    @seasonal_ma_params.setter
    def seasonal_ma_params(self, value):
        if np.isscalar(value):
            value = [value] * self.k_seasonal_ma_params
        self._params_split['seasonal_ma_params'] = validate_basic(
            value, self.k_seasonal_ma_params, title='seasonal MA coefficients')
        self._params = None

    @property
    def seasonal_ma_poly(self):
        """(Polynomial) Seasonal moving average lag polynomial."""
        # Need to expand the polynomial according to the season
        s = self.spec.seasonal_periods
        coef = np.array([1])
        if s > 0:
            expanded = np.zeros(self.spec.max_seasonal_ma_order)
            ix = np.array(self.spec.seasonal_ma_lags, dtype=int) - 1
            expanded[ix] = self._params_split['seasonal_ma_params']
            coef = np.r_[1, np.pad(np.reshape(expanded, (-1, 1)),
                                   [(0, 0), (s - 1, 0)], 'constant').flatten()]
        return Polynomial(coef)

    @seasonal_ma_poly.setter
    def seasonal_ma_poly(self, value):
        s = self.spec.seasonal_periods
        # Note: assume that we are given coefficients from the full polynomial
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, 1 + s * self.spec.max_seasonal_ma_order,
                               title='seasonal MA polynomial',)
        if value[0] != 1:
            raise ValueError('Polynomial constant must be equal to 1.')
        seasonal_ma_params = []
        for i in range(1, self.spec.max_seasonal_ma_order + 1):
            if i in self.spec.seasonal_ma_lags:
                seasonal_ma_params.append(value[s * i])
            elif value[s * i] != 0:
                raise ValueError('MA polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.seasonal_ma_params = seasonal_ma_params

    @property
    def sigma2(self):
        """(float) Innovation variance."""
        return self._params_split['sigma2']

    @sigma2.setter
    def sigma2(self, params):
        length = int(not self.spec.concentrate_scale)
        self._params_split['sigma2'] = validate_basic(
            params, length, title='sigma2').item()
        self._params = None

    @property
    def reduced_ar_poly(self):
        """(Polynomial) Reduced form autoregressive lag polynomial."""
        return self.ar_poly * self.seasonal_ar_poly

    @property
    def reduced_ma_poly(self):
        """(Polynomial) Reduced form moving average lag polynomial."""
        return self.ma_poly * self.seasonal_ma_poly

    @property
    def params(self):
        """(array) Complete parameter vector."""
        if self._params is None:
            self._params = self.spec.join_params(**self._params_split)
        return self._params.copy()

    @params.setter
    def params(self, value):
        self._params_split = self.spec.split_params(value)
        self._params = None

    @property
    def is_complete(self):
        """(bool) Are current parameter values all filled in (i.e. not NaN)."""
        return not np.any(np.isnan(self.params))

    @property
    def is_valid(self):
        """(bool) Are current parameter values valid (e.g. variance > 0)."""
        valid = True
        try:
            self.spec.validate_params(self.params)
        except ValueError:
            valid = False
        return valid

    @property
    def is_stationary(self):
        """(bool) Is the reduced autoregressive lag poylnomial stationary."""
        validate_basic(self.ar_params, self.k_ar_params,
                       title='AR coefficients')
        validate_basic(self.seasonal_ar_params, self.k_seasonal_ar_params,
                       title='seasonal AR coefficients')

        ar_stationary = True
        seasonal_ar_stationary = True
        if self.k_ar_params > 0:
            ar_stationary = is_invertible(self.ar_poly.coef)
        if self.k_seasonal_ar_params > 0:
            seasonal_ar_stationary = is_invertible(self.seasonal_ar_poly.coef)

        return ar_stationary and seasonal_ar_stationary

    @property
    def is_invertible(self):
        """(bool) Is the reduced moving average lag poylnomial invertible."""
        # Short-circuit if there is no MA component
        validate_basic(self.ma_params, self.k_ma_params,
                       title='MA coefficients')
        validate_basic(self.seasonal_ma_params, self.k_seasonal_ma_params,
                       title='seasonal MA coefficients')

        ma_stationary = True
        seasonal_ma_stationary = True
        if self.k_ma_params > 0:
            ma_stationary = is_invertible(self.ma_poly.coef)
        if self.k_seasonal_ma_params > 0:
            seasonal_ma_stationary = is_invertible(self.seasonal_ma_poly.coef)

        return ma_stationary and seasonal_ma_stationary

    def to_dict(self):
        """
        Return the parameters split by type into a dictionary.

        Returns
        -------
        split_params : dict
            Dictionary with keys 'exog_params', 'ar_params', 'ma_params',
            'seasonal_ar_params', 'seasonal_ma_params', and (unless
            `concentrate_scale=True`) 'sigma2'. Values are the parameters
            associated with the key, based on the `params` argument.
        """
        return self._params_split.copy()

    def to_pandas(self):
        """
        Return the parameters as a Pandas series.

        Returns
        -------
        series : pd.Series
            Pandas series with index set to the parameter names.
        """
        return pd.Series(self.params, index=self.param_names)

    def __repr__(self):
        """Represent SARIMAXParams object as a string."""
        components = []
        if self.k_exog_params:
            components.append('exog=%s' % str(self.exog_params))
        if self.k_ar_params:
            components.append('ar=%s' % str(self.ar_params))
        if self.k_ma_params:
            components.append('ma=%s' % str(self.ma_params))
        if self.k_seasonal_ar_params:
            components.append('seasonal_ar=%s' %
                              str(self.seasonal_ar_params))
        if self.k_seasonal_ma_params:
            components.append('seasonal_ma=%s' %
                              str(self.seasonal_ma_params))
        if not self.spec.concentrate_scale:
            components.append('sigma2=%s' % self.sigma2)
        return 'SARIMAXParams(%s)' % ', '.join(components)
