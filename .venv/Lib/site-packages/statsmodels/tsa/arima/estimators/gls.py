"""
Feasible generalized least squares for regression with SARIMA errors.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import warnings

from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.tools import diff

from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations, innovations_mle)
from statsmodels.tsa.arima.estimators.statespace import statespace

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams


def gls(endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0),
        include_constant=None, n_iter=None, max_iter=50, tolerance=1e-8,
        arma_estimator='innovations_mle', arma_estimator_kwargs=None):
    """
    Estimate ARMAX parameters by GLS.

    Parameters
    ----------
    endog : array_like
        Input time series array.
    exog : array_like, optional
        Array of exogenous regressors. If not included, then `include_constant`
        must be True, and then `exog` will only include the constant column.
    order : tuple, optional
        The (p,d,q) order of the ARIMA model. Default is (0, 0, 0).
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal ARIMA model.
        Default is (0, 0, 0, 0).
    include_constant : bool, optional
        Whether to add a constant term in `exog` if it's not already there.
        The estimate of the constant will then appear as one of the `exog`
        parameters. If `exog` is None, then the constant will represent the
        mean of the process. Default is True if the specified model does not
        include integration and False otherwise.
    n_iter : int, optional
        Optionally iterate feasible GSL a specific number of times. Default is
        to iterate to convergence. If set, this argument overrides the
        `max_iter` and `tolerance` arguments.
    max_iter : int, optional
        Maximum number of feasible GLS iterations. Default is 50. If `n_iter`
        is set, it overrides this argument.
    tolerance : float, optional
        Tolerance for determining convergence of feasible GSL iterations. If
        `iter` is set, this argument has no effect.
        Default is 1e-8.
    arma_estimator : str, optional
        The estimator used for estimating the ARMA model. This option should
        not generally be used, unless the default method is failing or is
        otherwise unsuitable. Not all values will be valid, depending on the
        specified model orders (`order` and `seasonal_order`). Possible values
        are:
        * 'innovations_mle' - can be used with any specification
        * 'statespace' - can be used with any specification
        * 'hannan_rissanen' - can be used with any ARMA non-seasonal model
        * 'yule_walker' - only non-seasonal consecutive
          autoregressive (AR) models
        * 'burg' - only non-seasonal, consecutive autoregressive (AR) models
        * 'innovations' - only non-seasonal, consecutive moving
          average (MA) models.
        The default is 'innovations_mle'.
    arma_estimator_kwargs : dict, optional
        Arguments to pass to the ARMA estimator.

    Returns
    -------
    parameters : SARIMAXParams object
        Contains the parameter estimates from the final iteration.
    other_results : Bunch
        Includes eight components: `spec`, `params`, `converged`,
        `differences`, `iterations`, `arma_estimator`, 'arma_estimator_kwargs',
        and `arma_results`.

    Notes
    -----
    The primary reference is [1]_, section 6.6. In particular, the
    implementation follows the iterative procedure described in section 6.6.2.
    Construction of the transformed variables used to compute the GLS estimator
    described in section 6.6.1 is done via an application of the innovations
    algorithm (rather than explicit construction of the transformation matrix).

    Note that if the specified model includes integration, both the `endog` and
    `exog` series will be differenced prior to estimation and a warning will
    be issued to alert the user.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    """
    # Handle n_iter
    if n_iter is not None:
        max_iter = n_iter
        tolerance = np.inf

    # Default for include_constant is True if there is no integration and
    # False otherwise
    integrated = order[1] > 0 or seasonal_order[1] > 0
    if include_constant is None:
        include_constant = not integrated
    elif include_constant and integrated:
        raise ValueError('Cannot include a constant in an integrated model.')

    # Handle including the constant (need to do it now so that the constant
    # parameter can be included in the specification as part of `exog`.)
    if include_constant:
        exog = np.ones_like(endog) if exog is None else add_constant(exog)

    # Create the SARIMAX specification
    spec = SARIMAXSpecification(endog, exog=exog, order=order,
                                seasonal_order=seasonal_order)
    endog = spec.endog
    exog = spec.exog

    # Handle integration
    if spec.is_integrated:
        # TODO: this is the approach suggested by BD (see Remark 1 in
        # section 6.6.2 and Example 6.6.3), but maybe there are some cases
        # where we don't want to force this behavior on the user?
        warnings.warn('Provided `endog` and `exog` series have been'
                      ' differenced to eliminate integration prior to GLS'
                      ' parameter estimation.')
        endog = diff(endog, k_diff=spec.diff,
                     k_seasonal_diff=spec.seasonal_diff,
                     seasonal_periods=spec.seasonal_periods)
        exog = diff(exog, k_diff=spec.diff,
                    k_seasonal_diff=spec.seasonal_diff,
                    seasonal_periods=spec.seasonal_periods)
    augmented = np.c_[endog, exog]

    # Validate arma_estimator
    spec.validate_estimator(arma_estimator)
    if arma_estimator_kwargs is None:
        arma_estimator_kwargs = {}

    # Step 1: OLS
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    exog_params = res_ols.params
    resid = res_ols.resid

    # 0th iteration parameters
    p = SARIMAXParams(spec=spec)
    p.exog_params = exog_params
    if spec.max_ar_order > 0:
        p.ar_params = np.zeros(spec.k_ar_params)
    if spec.max_seasonal_ar_order > 0:
        p.seasonal_ar_params = np.zeros(spec.k_seasonal_ar_params)
    if spec.max_ma_order > 0:
        p.ma_params = np.zeros(spec.k_ma_params)
    if spec.max_seasonal_ma_order > 0:
        p.seasonal_ma_params = np.zeros(spec.k_seasonal_ma_params)
    p.sigma2 = res_ols.scale

    ar_params = p.ar_params
    seasonal_ar_params = p.seasonal_ar_params
    ma_params = p.ma_params
    seasonal_ma_params = p.seasonal_ma_params
    sigma2 = p.sigma2

    # Step 2 - 4: iterate feasible GLS to convergence
    arma_results = [None]
    differences = [None]
    parameters = [p]
    converged = False if n_iter is None else None
    i = 0

    def _check_arma_estimator_kwargs(kwargs, method):
        if kwargs:
            raise ValueError(
                f"arma_estimator_kwargs not supported for method {method}"
            )

    for i in range(1, max_iter + 1):
        prev = exog_params

        # Step 2: ARMA
        # TODO: allow estimator-specific kwargs?
        if arma_estimator == 'yule_walker':
            p_arma, res_arma = yule_walker(
                resid, ar_order=spec.ar_order, demean=False,
                **arma_estimator_kwargs)
        elif arma_estimator == 'burg':
            _check_arma_estimator_kwargs(arma_estimator_kwargs, "burg")
            p_arma, res_arma = burg(resid, ar_order=spec.ar_order,
                                    demean=False)
        elif arma_estimator == 'innovations':
            _check_arma_estimator_kwargs(arma_estimator_kwargs, "innovations")
            out, res_arma = innovations(resid, ma_order=spec.ma_order,
                                        demean=False)
            p_arma = out[-1]
        elif arma_estimator == 'hannan_rissanen':
            p_arma, res_arma = hannan_rissanen(
                resid, ar_order=spec.ar_order, ma_order=spec.ma_order,
                demean=False, **arma_estimator_kwargs)
        else:
            # For later iterations, use a "warm start" for parameter estimates
            # (speeds up estimation and convergence)
            start_params = (
                None if i == 1 else np.r_[ar_params, ma_params,
                                          seasonal_ar_params,
                                          seasonal_ma_params, sigma2])
            # Note: in each case, we do not pass in the order of integration
            # since we have already differenced the series
            tmp_order = (spec.order[0], 0, spec.order[2])
            tmp_seasonal_order = (spec.seasonal_order[0], 0,
                                  spec.seasonal_order[2],
                                  spec.seasonal_order[3])
            if arma_estimator == 'innovations_mle':
                p_arma, res_arma = innovations_mle(
                    resid, order=tmp_order, seasonal_order=tmp_seasonal_order,
                    demean=False, start_params=start_params,
                    **arma_estimator_kwargs)
            else:
                p_arma, res_arma = statespace(
                    resid, order=tmp_order, seasonal_order=tmp_seasonal_order,
                    include_constant=False, start_params=start_params,
                    **arma_estimator_kwargs)

        ar_params = p_arma.ar_params
        seasonal_ar_params = p_arma.seasonal_ar_params
        ma_params = p_arma.ma_params
        seasonal_ma_params = p_arma.seasonal_ma_params
        sigma2 = p_arma.sigma2
        arma_results.append(res_arma)

        # Step 3: GLS
        # Compute transformed variables that satisfy OLS assumptions
        # Note: In section 6.1.1 of Brockwell and Davis (2016), these
        # transformations are developed as computed by left multiplcation
        # by a matrix T. However, explicitly constructing T and then
        # performing the left-multiplications does not scale well when nobs is
        # large. Instead, we can retrieve the transformed variables as the
        # residuals of the innovations algorithm (the `normalize=True`
        # argument applies a Prais-Winsten-type normalization to the first few
        # observations to ensure homoskedasticity). Brockwell and Davis
        # mention that they also take this approach in practice.

        # GH-6540: AR must be stationary

        if not p_arma.is_stationary:
            raise ValueError(
                "Roots of the autoregressive parameters indicate that data is"
                "non-stationary. GLS cannot be used with non-stationary "
                "parameters. You should consider differencing the model data"
                "or applying a nonlinear transformation (e.g., natural log)."
            )
        tmp, _ = arma_innovations.arma_innovations(
            augmented, ar_params=ar_params, ma_params=ma_params,
            normalize=True)
        u = tmp[:, 0]
        x = tmp[:, 1:]

        # OLS on transformed variables
        mod_gls = OLS(u, x)
        res_gls = mod_gls.fit()
        exog_params = res_gls.params
        resid = endog - np.dot(exog, exog_params)

        # Construct the parameter vector for the iteration
        p = SARIMAXParams(spec=spec)
        p.exog_params = exog_params
        if spec.max_ar_order > 0:
            p.ar_params = ar_params
        if spec.max_seasonal_ar_order > 0:
            p.seasonal_ar_params = seasonal_ar_params
        if spec.max_ma_order > 0:
            p.ma_params = ma_params
        if spec.max_seasonal_ma_order > 0:
            p.seasonal_ma_params = seasonal_ma_params
        p.sigma2 = sigma2
        parameters.append(p)

        # Check for convergence
        difference = np.abs(exog_params - prev)
        differences.append(difference)
        if n_iter is None and np.all(difference < tolerance):
            converged = True
            break
    else:
        if n_iter is None:
            warnings.warn('Feasible GLS failed to converge in %d iterations.'
                          ' Consider increasing the maximum number of'
                          ' iterations using the `max_iter` argument or'
                          ' reducing the required tolerance using the'
                          ' `tolerance` argument.' % max_iter)

    # Construct final results
    p = parameters[-1]
    other_results = Bunch({
        'spec': spec,
        'params': parameters,
        'converged': converged,
        'differences': differences,
        'iterations': i,
        'arma_estimator': arma_estimator,
        'arma_estimator_kwargs': arma_estimator_kwargs,
        'arma_results': arma_results,
    })

    return p, other_results
