# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
Author: Pamphile Roy
License: BSD-3

"""
import numpy as np
from scipy import stats
# scipy compat:
from statsmodels.compat.scipy import multivariate_t

from statsmodels.distributions.copula.copulas import Copula


class EllipticalCopula(Copula):
    """Base class for elliptical copula

    This class requires subclassing and currently does not have generic
    methods based on an elliptical generator.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    """
    def _handle_args(self, args):
        if args != () and args is not None:
            msg = ("Methods in elliptical copulas use copula parameters in"
                   " attributes. `arg` in the method is ignored")
            raise ValueError(msg)
        else:
            return args

    def rvs(self, nobs=1, args=(), random_state=None):
        self._handle_args(args)
        x = self.distr_mv.rvs(size=nobs, random_state=random_state)
        return self.distr_uv.cdf(x)

    def pdf(self, u, args=()):
        self._handle_args(args)
        ppf = self.distr_uv.ppf(u)
        mv_pdf_ppf = self.distr_mv.pdf(ppf)

        return mv_pdf_ppf / np.prod(self.distr_uv.pdf(ppf), axis=-1)

    def cdf(self, u, args=()):
        self._handle_args(args)
        ppf = self.distr_uv.ppf(u)
        return self.distr_mv.cdf(ppf)

    def tau(self, corr=None):
        """Bivariate kendall's tau based on correlation coefficient.

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Kendall's tau that corresponds to pearson correlation in the
        elliptical copula.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]
        rho = 2 * np.arcsin(corr) / np.pi
        return rho

    def corr_from_tau(self, tau):
        """Pearson correlation from kendall's tau.

        Parameters
        ----------
        tau : array_like
            Kendall's tau correlation coefficient.

        Returns
        -------
        Pearson correlation coefficient for given tau in elliptical
        copula. This can be used as parameter for an elliptical copula.
        """
        corr = np.sin(tau * np.pi / 2)
        return corr

    def fit_corr_param(self, data):
        """Copula correlation parameter using Kendall's tau of sample data.

        Parameters
        ----------
        data : array_like
            Sample data used to fit `theta` using Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.
            If k_dim > 2, then average tau is used.
        """
        x = np.asarray(data)

        if x.shape[1] == 2:
            tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
        else:
            k = self.k_dim
            tau = np.eye(k)
            for i in range(k):
                for j in range(i+1, k):
                    tau_ij = stats.kendalltau(x[..., i], x[..., j])[0]
                    tau[i, j] = tau[j, i] = tau_ij

        return self._arg_from_tau(tau)


class GaussianCopula(EllipticalCopula):
    r"""Gaussian copula.

    It is constructed from a multivariate normal distribution over
    :math:`\mathbb{R}^d` by using the probability integral transform.

    For a given correlation matrix :math:`R \in[-1, 1]^{d \times d}`,
    the Gaussian copula with parameter matrix :math:`R` can be written
    as:

    .. math::

        C_R^{\text{Gauss}}(u) = \Phi_R\left(\Phi^{-1}(u_1),\dots,
        \Phi^{-1}(u_d) \right),

    where :math:`\Phi^{-1}` is the inverse cumulative distribution function
    of a standard normal and :math:`\Phi_R` is the joint cumulative
    distribution function of a multivariate normal distribution with mean
    vector zero and covariance matrix equal to the correlation
    matrix :math:`R`.

    Parameters
    ----------
    corr : scalar or array_like
        Correlation or scatter matrix for the elliptical copula. In the
        bivariate case, ``corr` can be a scalar and is then considered as
        the correlation coefficient. If ``corr`` is None, then the scatter
        matrix is the identity matrix.
    k_dim : int
        Dimension, number of components in the multivariate random variable.
    allow_singular : bool
        Allow singular correlation matrix.
        The behavior when the correlation matrix is singular is determined by
        `scipy.stats.multivariate_normal`` and might not be appropriate for
        all copula or copula distribution metnods. Behavior might change in
        future versions.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    References
    ----------
    .. [1] Joe, Harry, 2014, Dependence modeling with copulas. CRC press.
        p. 163

    """

    def __init__(self, corr=None, k_dim=2, allow_singular=False):
        super().__init__(k_dim=k_dim)
        if corr is None:
            corr = np.eye(k_dim)
        elif k_dim == 2 and np.size(corr) == 1:
            corr = np.array([[1., corr], [corr, 1.]])

        self.corr = np.asarray(corr)
        self.args = (self.corr,)
        self.distr_uv = stats.norm
        self.distr_mv = stats.multivariate_normal(
            cov=corr, allow_singular=allow_singular)

    def dependence_tail(self, corr=None):
        """
        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : any
            Tail dependence for Gaussian copulas is always zero.
            Argument will be ignored

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        """

        return 0, 0

    def _arg_from_tau(self, tau):
        # for generic compat
        return self.corr_from_tau(tau)


class StudentTCopula(EllipticalCopula):
    """Student t copula.

    Parameters
    ----------
    corr : scalar or array_like
        Correlation or scatter matrix for the elliptical copula. In the
        bivariate case, ``corr` can be a scalar and is then considered as
        the correlation coefficient. If ``corr`` is None, then the scatter
        matrix is the identity matrix.
    df : float (optional)
        Degrees of freedom of the multivariate t distribution.
    k_dim : int
        Dimension, number of components in the multivariate random variable.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    References
    ----------
    .. [1] Joe, Harry, 2014, Dependence modeling with copulas. CRC press.
        p. 181
    """

    def __init__(self, corr=None, df=None, k_dim=2):
        super().__init__(k_dim=k_dim)
        if corr is None:
            corr = np.eye(k_dim)
        elif k_dim == 2 and np.size(corr) == 1:
            corr = np.array([[1., corr], [corr, 1.]])

        self.df = df
        self.corr = np.asarray(corr)
        self.args = (corr, df)
        # both uv and mv are frozen distributions
        self.distr_uv = stats.t(df=df)
        self.distr_mv = multivariate_t(shape=corr, df=df)

    def cdf(self, u, args=()):
        raise NotImplementedError("CDF not available in closed form.")
        # ppf = self.distr_uv.ppf(u)
        # mvt = MVT([0, 0], self.corr, self.df)
        # return mvt.cdf(ppf)

    def spearmans_rho(self, corr=None):
        """
        Bivariate Spearman's rho based on correlation coefficient.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Spearman's rho that corresponds to pearson correlation in the
        elliptical copula.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]

        tau = 6 * np.arcsin(corr / 2) / np.pi
        return tau

    def dependence_tail(self, corr=None):
        """
        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]

        df = self.df
        t = - np.sqrt((df + 1) * (1 - corr) / 1 + corr)
        # Note self.distr_uv is frozen, df cannot change, use stats.t instead
        lam = 2 * stats.t.cdf(t, df + 1)
        return lam, lam

    def _arg_from_tau(self, tau):
        # for generic compat
        # this does not provide an estimate of df
        return self.corr_from_tau(tau)
