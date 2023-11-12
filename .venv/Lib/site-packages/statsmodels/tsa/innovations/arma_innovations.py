import numpy as np

from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from . import _arma_innovations

NON_STATIONARY_ERROR = """\
The model's autoregressive parameters (ar_params) indicate that the process
 is non-stationary. The innovations algorithm cannot be used.
"""


def arma_innovations(endog, ar_params=None, ma_params=None, sigma2=1,
                     normalize=False, prefix=None):
    """
    Compute innovations using a given ARMA process.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process, may be univariate or multivariate.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    normalize : bool, optional
        Whether or not to normalize the returned innovations. Default is False.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    innovations : ndarray
        Innovations (one-step-ahead prediction errors) for the given `endog`
        series with predictions based on the given ARMA process. If
        `normalize=True`, then the returned innovations have been "whitened" by
        dividing through by the square root of the mean square error.
    innovations_mse : ndarray
        Mean square error for the innovations.
    """
    # Parameters
    endog = np.array(endog)
    squeezed = endog.ndim == 1
    if squeezed:
        endog = endog[:, None]

    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    nobs, k_endog = endog.shape
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]

    # Get BLAS prefix
    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]

    # Make arrays contiguous for BLAS calls
    endog = np.asfortranarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = dtype(sigma2).item()

    # Get the appropriate functions
    arma_transformed_acovf_fast = getattr(
        _arma_innovations, prefix + 'arma_transformed_acovf_fast')
    arma_innovations_algo_fast = getattr(
        _arma_innovations, prefix + 'arma_innovations_algo_fast')
    arma_innovations_filter = getattr(
        _arma_innovations, prefix + 'arma_innovations_filter')

    # Run the innovations algorithm for ARMA coefficients
    arma_acovf = arima_process.arma_acovf(ar, ma,
                                          sigma2=sigma2, nobs=nobs) / sigma2
    acovf, acovf2 = arma_transformed_acovf_fast(ar, ma, arma_acovf)
    theta, v = arma_innovations_algo_fast(nobs, ar_params, ma_params,
                                          acovf, acovf2)
    v = np.array(v)
    if (np.any(v < 0) or
            not np.isfinite(theta).all() or
            not np.isfinite(v).all()):
        # This is defensive code that is hard to hit
        raise ValueError(NON_STATIONARY_ERROR)

    # Run the innovations filter across each series
    u = []
    for i in range(k_endog):
        u_i = np.array(arma_innovations_filter(endog[:, i], ar_params,
                                               ma_params, theta))
        u.append(u_i)
    u = np.vstack(u).T
    if normalize:
        u /= v[:, None]**0.5

    # Post-processing
    if squeezed:
        u = u.squeeze()

    return u, v


def arma_loglike(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute the log-likelihood of the given data assuming an ARMA process.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    float
        The joint loglikelihood.
    """
    llf_obs = arma_loglikeobs(endog, ar_params=ar_params, ma_params=ma_params,
                              sigma2=sigma2, prefix=prefix)
    return np.sum(llf_obs)


def arma_loglikeobs(endog, ar_params=None, ma_params=None, sigma2=1,
                    prefix=None):
    """
    Compute the log-likelihood for each observation assuming an ARMA process.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    ndarray
        Array of loglikelihood values for each observation.
    """
    endog = np.array(endog)
    ar_params = np.atleast_1d([] if ar_params is None else ar_params)
    ma_params = np.atleast_1d([] if ma_params is None else ma_params)

    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [endog, ar_params, ma_params, np.array(sigma2)])
    dtype = prefix_dtype_map[prefix]

    endog = np.ascontiguousarray(endog, dtype=dtype)
    ar_params = np.asfortranarray(ar_params, dtype=dtype)
    ma_params = np.asfortranarray(ma_params, dtype=dtype)
    sigma2 = dtype(sigma2).item()

    func = getattr(_arma_innovations, prefix + 'arma_loglikeobs_fast')
    return func(endog, ar_params, ma_params, sigma2)


def arma_score(endog, ar_params=None, ma_params=None, sigma2=1,
               prefix=None):
    """
    Compute the score (gradient of the log-likelihood function).

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    ndarray
        Score, evaluated at the given parameters.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglike(endog, params[:p], params[p:p + q], params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)


def arma_scoreobs(endog, ar_params=None, ma_params=None, sigma2=1,
                  prefix=None):
    """
    Compute the score (gradient) per observation.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    ndarray
        Score per observation, evaluated at the given parameters.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    """
    ar_params = [] if ar_params is None else ar_params
    ma_params = [] if ma_params is None else ma_params

    p = len(ar_params)
    q = len(ma_params)

    def func(params):
        return arma_loglikeobs(endog, params[:p], params[p:p + q],
                               params[p + q:])

    params0 = np.r_[ar_params, ma_params, sigma2]
    epsilon = _get_epsilon(params0, 2., None, len(params0))
    return approx_fprime_cs(params0, func, epsilon)
