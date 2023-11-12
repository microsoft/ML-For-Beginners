"""
State space approach to estimating SARIMAX models.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np

from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams


def statespace(endog, exog=None, order=(0, 0, 0),
               seasonal_order=(0, 0, 0, 0), include_constant=True,
               enforce_stationarity=True, enforce_invertibility=True,
               concentrate_scale=False, start_params=None, fit_kwargs=None):
    """
    Estimate SARIMAX parameters using state space methods.

    Parameters
    ----------
    endog : array_like
        Input time series array.
    order : tuple, optional
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. Default is (0, 0, 0).
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0).
    include_constant : bool, optional
        Whether to add a constant term in `exog` if it's not already there.
        The estimate of the constant will then appear as one of the `exog`
        parameters. If `exog` is None, then the constant will represent the
        mean of the process.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization. The
        AR polynomial must be stationary. If `enforce_invertibility=True` the
        MA poylnomial must be invertible. If not provided, default starting
        parameters are computed using the Hannan-Rissanen method.
    fit_kwargs : dict, optional
        Arguments to pass to the state space model's `fit` method.

    Returns
    -------
    parameters : SARIMAXParams object
    other_results : Bunch
        Includes two components, `spec`, containing the `SARIMAXSpecification`
        instance corresponding to the input arguments; and
        `state_space_results`, corresponding to the results from the underlying
        state space model and Kalman filter / smoother.

    Notes
    -----
    The primary reference is [1]_.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """
    # Handle including the constant (need to do it now so that the constant
    # parameter can be included in the specification as part of `exog`.)
    if include_constant:
        exog = np.ones_like(endog) if exog is None else add_constant(exog)

    # Create the specification
    spec = SARIMAXSpecification(
        endog, exog=exog, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale)
    endog = spec.endog
    exog = spec.exog
    p = SARIMAXParams(spec=spec)

    # Check start parameters
    if start_params is not None:
        sp = SARIMAXParams(spec=spec)
        sp.params = start_params

        if spec.enforce_stationarity and not sp.is_stationary:
            raise ValueError('Given starting parameters imply a non-stationary'
                             ' AR process with `enforce_stationarity=True`.')

        if spec.enforce_invertibility and not sp.is_invertible:
            raise ValueError('Given starting parameters imply a non-invertible'
                             ' MA process with `enforce_invertibility=True`.')

    # Create and fit the state space model
    mod = SARIMAX(endog, exog=exog, order=spec.order,
                  seasonal_order=spec.seasonal_order,
                  enforce_stationarity=spec.enforce_stationarity,
                  enforce_invertibility=spec.enforce_invertibility,
                  concentrate_scale=spec.concentrate_scale)
    if fit_kwargs is None:
        fit_kwargs = {}
    fit_kwargs.setdefault('disp', 0)
    res_ss = mod.fit(start_params=start_params, **fit_kwargs)

    # Construct results
    p.params = res_ss.params
    res = Bunch({
        'spec': spec,
        'statespace_results': res_ss,
    })

    return p, res
