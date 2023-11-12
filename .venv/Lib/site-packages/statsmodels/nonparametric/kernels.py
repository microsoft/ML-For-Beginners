"""
Module of kernels that are able to handle continuous as well as categorical
variables (both ordered and unordered).

This is a slight deviation from the current approach in
statsmodels.nonparametric.kernels where each kernel is a class object.

Having kernel functions rather than classes makes extension to a multivariate
kernel density estimation much easier.

NOTE: As it is, this module does not interact with the existing API
"""

import numpy as np
from scipy.special import erf


#TODO:
# - make sure we only receive int input for wang-ryzin and aitchison-aitken
# - Check for the scalar Xi case everywhere


def aitchison_aitken(h, Xi, x, num_levels=None):
    r"""
    The Aitchison-Aitken kernel, used for unordered discrete random variables.

    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 2-D ndarray of ints, shape (nobs, K)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.
    num_levels : bool, optional
        Gives the user the option to specify the number of levels for the
        random variable.  If False, the number of levels is calculated from
        the data.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p.18 of [2]_ for details.  The value of the kernel L if :math:`X_{i}=x`
    is :math:`1-\lambda`, otherwise it is :math:`\frac{\lambda}{c-1}`.
    Here :math:`c` is the number of levels plus one of the RV.

    References
    ----------
    .. [*] J. Aitchison and C.G.G. Aitken, "Multivariate binary discrimination
           by the kernel method", Biometrika, vol. 63, pp. 413-420, 1976.
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    Xi = Xi.reshape(Xi.size)  # seems needed in case Xi is scalar
    if num_levels is None:
        num_levels = np.asarray(np.unique(Xi).size)

    kernel_value = np.ones(Xi.size) * h / (num_levels - 1)
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value


def wang_ryzin(h, Xi, x):
    r"""
    The Wang-Ryzin kernel, used for ordered discrete random variables.

    Parameters
    ----------
    h : scalar or 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : ndarray of ints, shape (nobs, K)
        The value of the training set.
    x : scalar or 1-D ndarray of shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p. 19 in [1]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\lambda`, otherwise it is
    :math:`\frac{1-\lambda}{2}\lambda^{|X_{i}-x|}`, where :math:`\lambda` is
    the bandwidth.

    References
    ----------
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
           http://dx.doi.org/10.1561/0800000009
    .. [*] M.-C. Wang and J. van Ryzin, "A class of smooth estimators for
           discrete distributions", Biometrika, vol. 68, pp. 301-309, 1981.
    """
    Xi = Xi.reshape(Xi.size)  # seems needed in case Xi is scalar
    kernel_value = 0.5 * (1 - h) * (h ** abs(Xi - x))
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value


def gaussian(h, Xi, x):
    """
    Gaussian Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.
    """
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))


def tricube(h, Xi, x):
    """
    Tricube Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    u[np.abs(u) > 1] = 0
    return (70. / 81) * (1 - np.abs(u)**3)**3


def gaussian_convolution(h, Xi, x):
    """ Calculates the Gaussian Convolution Kernel """
    return (1. / np.sqrt(4 * np.pi)) * np.exp(- (Xi - x)**2 / (h**2 * 4.))


def wang_ryzin_convolution(h, Xi, Xj):
    # This is the equivalent of the convolution case with the Gaussian Kernel
    # However it is not exactly convolution. Think of a better name
    # References
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        ordered += wang_ryzin(h, Xi, x) * wang_ryzin(h, Xj, x)

    return ordered


def aitchison_aitken_convolution(h, Xi, Xj):
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels) * \
                   aitchison_aitken(h, Xj, x, num_levels=num_levels)

    return ordered


def gaussian_cdf(h, Xi, x):
    return 0.5 * h * (1 + erf((x - Xi) / (h * np.sqrt(2))))


def aitchison_aitken_cdf(h, Xi, x_u):
    x_u = int(x_u)
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        if x <= x_u:  #FIXME: why a comparison for unordered variables?
            ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels)

    return ordered


def wang_ryzin_cdf(h, Xi, x_u):
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        if x <= x_u:
            ordered += wang_ryzin(h, Xi, x)

    return ordered


def d_gaussian(h, Xi, x):
    # The derivative of the Gaussian Kernel
    return 2 * (Xi - x) * gaussian(h, Xi, x) / h**2


def aitchison_aitken_reg(h, Xi, x):
    """
    A version for the Aitchison-Aitken kernel for nonparametric regression.

    Suggested by Li and Racine.
    """
    kernel_value = np.ones(Xi.size)
    ix = Xi != x
    inDom = ix * h
    kernel_value[ix] = inDom[ix]
    return kernel_value


def wang_ryzin_reg(h, Xi, x):
    """
    A version for the Wang-Ryzin kernel for nonparametric regression.

    Suggested by Li and Racine in [1] ch.4
    """
    return h ** abs(Xi - x)
