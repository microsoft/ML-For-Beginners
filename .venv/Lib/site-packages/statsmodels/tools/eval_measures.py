# -*- coding: utf-8 -*-
"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011
Updated on Wed Jun 03 10:42:20 2020

Authors: Josef Perktold & Peter Prescott
License: BSD-3

"""
import numpy as np

from statsmodels.tools.validation import array_like


def mse(x1, x2, axis=0):
    """mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)


def rmse(x1, x2, axis=0):
    """root mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))


def rmspe(y, y_hat, axis=0, zeros=np.nan):
    """
    Root Mean Squared Percentage Error

    Parameters
    ----------
    y : array_like
      The actual value.
    y_hat : array_like
       The predicted value.
    axis : int
       Axis along which the summary statistic is calculated
    zeros : float
       Value to assign to error where y is zero

    Returns
    -------
    rmspe : ndarray or float
       Root Mean Squared Percentage Error along given axis.
    """
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    error = y - y_hat
    loc = y != 0
    loc = loc.ravel()
    percentage_error = np.full_like(error, zeros)
    percentage_error.flat[loc] = error.flat[loc] / y.flat[loc]
    mspe = np.nanmean(percentage_error ** 2, axis=axis) * 100
    return np.sqrt(mspe)


def maxabs(x1, x2, axis=0):
    """maximum absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    maxabs : ndarray or float
       maximum absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.max(np.abs(x1 - x2), axis=axis)


def meanabs(x1, x2, axis=0):
    """mean absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    meanabs : ndarray or float
       mean absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(np.abs(x1 - x2), axis=axis)


def medianabs(x1, x2, axis=0):
    """median absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianabs : ndarray or float
       median absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1 - x2), axis=axis)


def bias(x1, x2, axis=0):
    """bias, mean error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    bias : ndarray or float
       bias, or mean difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(x1 - x2, axis=axis)


def medianbias(x1, x2, axis=0):
    """median bias, median error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianbias : ndarray or float
       median bias, or median difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(x1 - x2, axis=axis)


def vare(x1, x2, ddof=0, axis=0):
    """variance of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    vare : ndarray or float
       variance of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.var(x1 - x2, ddof=ddof, axis=axis)


def stde(x1, x2, ddof=0, axis=0):
    """standard deviation of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    stde : ndarray or float
       standard deviation of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1 - x2, ddof=ddof, axis=axis)


def iqr(x1, x2, axis=0):
    """
    Interquartile range of error

    Parameters
    ----------
    x1 : array_like
       One of the inputs into the IQR calculation.
    x2 : array_like
       The other input into the IQR calculation.
    axis : {None, int}
       axis along which the summary statistic is calculated

    Returns
    -------
    irq : {float, ndarray}
       Interquartile range along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they must broadcast.
    """
    x1 = array_like(x1, "x1", dtype=None, ndim=None)
    x2 = array_like(x2, "x1", dtype=None, ndim=None)
    if axis is None:
        x1 = x1.ravel()
        x2 = x2.ravel()
        axis = 0
    xdiff = np.sort(x1 - x2, axis=axis)
    nobs = x1.shape[axis]
    idx = np.round((nobs - 1) * np.array([0.25, 0.75])).astype(int)
    sl = [slice(None)] * xdiff.ndim
    sl[axis] = idx
    iqr = np.diff(xdiff[tuple(sl)], axis=axis)
    iqr = np.squeeze(iqr)  # drop reduced dimension
    return iqr


# Information Criteria
# ---------------------


def aic(llf, nobs, df_modelwc):
    """
    Akaike information criterion

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    return -2.0 * llf + 2.0 * df_modelwc


def aicc(llf, nobs, df_modelwc):
    """
    Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    Notes
    -----
    Returns +inf if the effective degrees of freedom, defined as
    ``nobs - df_modelwc - 1.0``, is <= 0.
    """
    dof_eff = nobs - df_modelwc - 1.0
    if dof_eff > 0:
        return -2.0 * llf + 2.0 * df_modelwc * nobs / dof_eff
    else:
        return np.inf


def bic(llf, nobs, df_modelwc):
    """
    Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    References
    ----------
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    return -2.0 * llf + np.log(nobs) * df_modelwc


def hqic(llf, nobs, df_modelwc):
    """
    Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    References
    ----------
    Wikipedia does not say much
    """
    return -2.0 * llf + 2 * np.log(np.log(nobs)) * df_modelwc


# IC based on residual sigma


def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    r"""
    Akaike information criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. The information criteria should be used to compare
    only comparable models.

    For example, AIC is defined in terms of the loglikelihood as

    :math:`-2 llf + 2 k`

    in terms of :math:`\hat{\sigma}^2`

    :math:`log(\hat{\sigma}^2) + 2 k / n`

    in terms of the determinant of :math:`\hat{\Sigma}`

    :math:`log(\|\hat{\Sigma}\|) + 2 k / n`

    Note: In our definition we do not divide by n in the log-likelihood
    version.

    TODO: Latex math

    reference for example lecture notes by Herman Bierens

    See Also
    --------

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs


def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    """
    Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs


def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs


def hqic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    xxx
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + hqic(0, nobs, df_modelwc) / nobs


# from var_model.py, VAR only? separates neqs and k_vars per equation
# def fpe_sigma():
#     ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)


__all__ = [
    maxabs,
    meanabs,
    medianabs,
    medianbias,
    mse,
    rmse,
    rmspe,
    stde,
    vare,
    aic,
    aic_sigma,
    aicc,
    aicc_sigma,
    bias,
    bic,
    bic_sigma,
    hqic,
    hqic_sigma,
    iqr,
]
