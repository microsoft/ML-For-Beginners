# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats

from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.distributions.copula.copulas import Copula


class IndependenceCopula(Copula):
    """Independence copula.

    Copula with independent random variables.

    .. math::

        C_\theta(u,v) = uv

    Parameters
    ----------
    k_dim : int
        Dimension, number of components in the multivariate random variable.

    Notes
    -----
    IndependenceCopula does not have copula parameters.
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    """
    def __init__(self, k_dim=2):
        super().__init__(k_dim=k_dim)

    def _handle_args(self, args):
        if args != () and args is not None:
            msg = ("Independence copula does not use copula parameters.")
            raise ValueError(msg)
        else:
            return args

    def rvs(self, nobs=1, args=(), random_state=None):
        self._handle_args(args)
        rng = check_random_state(random_state)
        x = rng.random((nobs, self.k_dim))
        return x

    def pdf(self, u, args=()):
        u = np.asarray(u)
        return np.ones(u.shape[:-1])

    def cdf(self, u, args=()):
        return np.prod(u, axis=-1)

    def tau(self):
        return 0

    def plot_pdf(self, *args):
        raise NotImplementedError("PDF is constant over the domain.")


def rvs_kernel(sample, size, bw=1, k_func=None, return_extras=False):
    """Random sampling from empirical copula using Beta distribution

    Parameters
    ----------
    sample : ndarray
        Sample of multivariate observations in (o, 1) interval.
    size : int
        Number of observations to simulate.
    bw : float
        Bandwidth for Beta sampling. The beta copula corresponds to a kernel
        estimate of the distribution. bw=1 corresponds to the empirical beta
        copula. A small bandwidth like bw=0.001 corresponds to small noise
        added to the empirical distribution. Larger bw, e.g. bw=10 corresponds
        to kernel estimate with more smoothing.
    k_func : None or callable
        The default kernel function is currently a beta function with 1 added
        to the first beta parameter.
    return_extras : bool
        If this is False, then only the random sample will be returned.
        If true, then extra information is returned that is mainly of interest
        for verification.

    Returns
    -------
    rvs : ndarray
        Multivariate sample with ``size`` observations drawn from the Beta
        Copula.

    Notes
    -----
    Status: experimental, API will change.
    """
    # vectorized for observations
    n = sample.shape[0]
    if k_func is None:
        kfunc = _kernel_rvs_beta1
    idx = np.random.randint(0, n, size=size)
    xi = sample[idx]
    krvs = np.column_stack([kfunc(xii, bw) for xii in xi.T])

    if return_extras:
        return krvs, idx, xi
    else:
        return krvs


def _kernel_rvs_beta(x, bw):
    # Beta kernel for density, pdf, estimation
    return stats.beta.rvs(x / bw + 1, (1 - x) / bw + 1, size=x.shape)


def _kernel_rvs_beta1(x, bw):
    # Beta kernel for density, pdf, estimation
    # Kiriliouk, Segers, Tsukuhara 2020 arxiv, using bandwith 1/nobs sample
    return stats.beta.rvs(x / bw, (1 - x) / bw + 1)
