# -*- coding: utf-8 -*-
"""
Penalty classes for Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

"""

import numpy as np
from scipy.linalg import block_diag
from statsmodels.base._penalties import Penalty


class UnivariateGamPenalty(Penalty):
    """
    Penalty for smooth term in Generalized Additive Models

    Parameters
    ----------
    univariate_smoother : instance
        instance of univariate smoother or spline class
    alpha : float
        default penalty weight, alpha can be provided to each method
    weights:
        TODO: not used and verified, might be removed

    Attributes
    ----------
    Parameters are stored, additionally
    nob s: The number of samples used during the estimation
    n_columns : number of columns in smoother basis
    """

    def __init__(self, univariate_smoother, alpha=1, weights=1):
        self.weights = weights
        self.alpha = alpha
        self.univariate_smoother = univariate_smoother
        self.nobs = self.univariate_smoother.nobs
        self.n_columns = self.univariate_smoother.dim_basis

    def func(self, params, alpha=None):
        """evaluate penalization at params

        Parameters
        ----------
        params : ndarray
            coefficients for the spline basis in the regression model
        alpha : float
            default penalty weight

        Returns
        -------
        func : float
            value of the penalty evaluated at params
        """
        if alpha is None:
            alpha = self.alpha

        f = params.dot(self.univariate_smoother.cov_der2.dot(params))
        return alpha * f / self.nobs

    def deriv(self, params, alpha=None):
        """evaluate derivative of penalty with respect to params

        Parameters
        ----------
        params : ndarray
            coefficients for the spline basis in the regression model
        alpha : float
            default penalty weight

        Returns
        -------
        deriv : ndarray
            derivative, gradient of the penalty with respect to params
        """
        if alpha is None:
            alpha = self.alpha

        d = 2 * alpha * np.dot(self.univariate_smoother.cov_der2, params)
        d /= self.nobs
        return d

    def deriv2(self, params, alpha=None):
        """evaluate second derivative of penalty with respect to params

        Parameters
        ----------
        params : ndarray
            coefficients for the spline basis in the regression model
        alpha : float
            default penalty weight

        Returns
        -------
        deriv2 : ndarray, 2-Dim
            second derivative, hessian of the penalty with respect to params
        """
        if alpha is None:
            alpha = self.alpha

        d2 = 2 * alpha * self.univariate_smoother.cov_der2
        d2 /= self.nobs
        return d2

    def penalty_matrix(self, alpha=None):
        """penalty matrix for the smooth term of a GAM

        Parameters
        ----------
        alpha : list of floats or None
            penalty weights

        Returns
        -------
        penalty matrix
            square penalty matrix for quadratic penalization. The number
            of rows and columns are equal to the number of columns in the
            smooth terms, i.e. the number of parameters for this smooth
            term in the regression model
        """
        if alpha is None:
            alpha = self.alpha

        return alpha * self.univariate_smoother.cov_der2


class MultivariateGamPenalty(Penalty):
    """
    Penalty for Generalized Additive Models

    Parameters
    ----------
    multivariate_smoother : instance
        instance of additive smoother or spline class
    alpha : list of float
        default penalty weight, list with length equal to the number of smooth
        terms. ``alpha`` can also be provided to each method.
    weights : array_like
        currently not used
        is a list of doubles of the same length as alpha or a list
        of ndarrays where each component has the length equal to the number
        of columns in that component
    start_idx : int
        number of parameters that come before the smooth terms. If the model
        has a linear component, then the parameters for the smooth components
        start at ``start_index``.

    Attributes
    ----------
    Parameters are stored, additionally
    nob s: The number of samples used during the estimation

    dim_basis : number of columns of additive smoother. Number of columns
        in all smoothers.
    k_variables : number of smooth terms
    k_params : total number of parameters in the regression model
    """

    def __init__(self, multivariate_smoother, alpha, weights=None,
                 start_idx=0):

        if len(multivariate_smoother.smoothers) != len(alpha):
            msg = ('all the input values should be of the same length.'
                   ' len(smoothers)=%d, len(alphas)=%d') % (
                   len(multivariate_smoother.smoothers), len(alpha))
            raise ValueError(msg)

        self.multivariate_smoother = multivariate_smoother
        self.dim_basis = self.multivariate_smoother.dim_basis
        self.k_variables = self.multivariate_smoother.k_variables
        self.nobs = self.multivariate_smoother.nobs
        self.alpha = alpha
        self.start_idx = start_idx
        self.k_params = start_idx + self.dim_basis

        # TODO: Review this,
        if weights is None:
            # weights should have total length as params
            # but it can also be scalar in individual component
            self.weights = [1. for _ in range(self.k_variables)]
        else:
            import warnings
            warnings.warn('weights is currently ignored')
            self.weights = weights

        self.mask = [np.zeros(self.k_params, dtype=bool)
                     for _ in range(self.k_variables)]
        param_count = start_idx
        for i, smoother in enumerate(self.multivariate_smoother.smoothers):
            # the mask[i] contains a vector of length k_columns. The index
            # corresponding to the i-th input variable are set to True.
            self.mask[i][param_count: param_count + smoother.dim_basis] = True
            param_count += smoother.dim_basis

        self.gp = []
        for i in range(self.k_variables):
            gp = UnivariateGamPenalty(self.multivariate_smoother.smoothers[i],
                                      weights=self.weights[i],
                                      alpha=self.alpha[i])
            self.gp.append(gp)

    def func(self, params, alpha=None):
        """evaluate penalization at params

        Parameters
        ----------
        params : ndarray
            coefficients in the regression model
        alpha : float or list of floats
            penalty weights

        Returns
        -------
        func : float
            value of the penalty evaluated at params
        """
        if alpha is None:
            alpha = [None] * self.k_variables

        cost = 0
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i, alpha=alpha[i])

        return cost

    def deriv(self, params, alpha=None):
        """evaluate derivative of penalty with respect to params

        Parameters
        ----------
        params : ndarray
            coefficients in the regression model
        alpha : list of floats or None
            penalty weights

        Returns
        -------
        deriv : ndarray
            derivative, gradient of the penalty with respect to params
        """
        if alpha is None:
            alpha = [None] * self.k_variables

        grad = [np.zeros(self.start_idx)]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].deriv(params_i, alpha=alpha[i]))

        return np.concatenate(grad)

    def deriv2(self, params, alpha=None):
        """evaluate second derivative of penalty with respect to params

        Parameters
        ----------
        params : ndarray
            coefficients in the regression model
        alpha : list of floats or None
            penalty weights

        Returns
        -------
        deriv2 : ndarray, 2-Dim
            second derivative, hessian of the penalty with respect to params
        """
        if alpha is None:
            alpha = [None] * self.k_variables

        deriv2 = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            deriv2.append(self.gp[i].deriv2(params_i, alpha=alpha[i]))

        return block_diag(*deriv2)

    def penalty_matrix(self, alpha=None):
        """penalty matrix for generalized additive model

        Parameters
        ----------
        alpha : list of floats or None
            penalty weights

        Returns
        -------
        penalty matrix
            block diagonal, square penalty matrix for quadratic penalization.
            The number of rows and columns are equal to the number of
            parameters in the regression model ``k_params``.

        Notes
        -----
        statsmodels does not support backwards compatibility when keywords are
        used as positional arguments. The order of keywords might change.
        We might need to add a ``params`` keyword if the need arises.
        """
        if alpha is None:
            alpha = self.alpha

        s_all = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            s_all.append(self.gp[i].penalty_matrix(alpha=alpha[i]))

        return block_diag(*s_all)
