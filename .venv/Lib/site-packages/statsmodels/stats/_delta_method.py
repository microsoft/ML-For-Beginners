# -*- coding: utf-8 -*-
"""
Author: Josef Perktold
License: BSD-3

"""

from __future__ import print_function
import numpy as np
from scipy import stats


class NonlinearDeltaCov:
    '''Asymptotic covariance by Deltamethod

    The function is designed for 2d array, with rows equal to
    the number of equations or constraints and columns equal to the number
    of parameters. 1d params work by chance ?

    fun: R^{m*k) -> R^{m}  where m is number of equations and k is
    the number of parameters.

    equations follow Greene

    This class does not use any caching. The intended usage is as a helper
    function. Extra methods have been added for convenience but might move
    to calling functions.

    The naming in this class uses params for the original random variable, and
    cov_params for it's covariance matrix. However, this class is independent
    of the use cases in support of the models.

    Parameters
    ----------
    func : callable, f(params)
        Nonlinear function of the estimation parameters. The return of
        the function can be vector valued, i.e. a 1-D array.
    params : ndarray
        Parameters at which function `func` is evaluated.
    cov_params : ndarray
        Covariance matrix of the parameters `params`.
    deriv : function or None
        First derivative or Jacobian of func. If deriv is None, then a
        numerical derivative will be used. If func returns a 1-D array,
        then the `deriv` should have rows corresponding to the elements
        of the return of func.
    func_args : None
        Not yet implemented.


    '''
    def __init__(self, func, params, cov_params, deriv=None, func_args=None):
        self.fun = func
        self.params = params
        self.cov_params = cov_params
        self._grad = deriv
        self.func_args = func_args if func_args is not None else ()
        if func_args is not None:
            raise NotImplementedError('func_args not yet implemented')

    def grad(self, params=None, **kwds):
        """First derivative, jacobian of func evaluated at params.

        Parameters
        ----------
        params : None or ndarray
            Values at which gradient is evaluated. If params is None, then
            the attached params are used.
            TODO: should we drop this
        kwds : keyword arguments
            This keyword arguments are used without changes in the calulation
            of numerical derivatives. These are only used if a `deriv` function
            was not provided.

        Returns
        -------
        grad : ndarray
            gradient or jacobian of the function
        """
        if params is None:
            params = self.params
        if self._grad is not None:
            return self._grad(params)
        else:
            # copied from discrete_margins
            try:
                from statsmodels.tools.numdiff import approx_fprime_cs
                jac = approx_fprime_cs(params, self.fun, **kwds)
            except TypeError:  # norm.cdf doesn't take complex values
                from statsmodels.tools.numdiff import approx_fprime
                jac = approx_fprime(params, self.fun, **kwds)

            return jac

    def cov(self):
        """Covariance matrix of the transformed random variable.
        """
        g = self.grad()
        covar = np.dot(np.dot(g, self.cov_params), g.T)
        return covar

    def predicted(self):
        """Value of the function evaluated at the attached params.

        Note: This is not equal to the expected value if the transformation is
        nonlinear. If params is the maximum likelihood estimate, then
        `predicted` is the maximum likelihood estimate of the value of the
        nonlinear function.
        """

        predicted = self.fun(self.params)

        # TODO: why do I need to squeeze in poisson example
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        return predicted

    def wald_test(self, value):
        """Joint hypothesis tests that H0: f(params) = value.

        The alternative hypothesis is two-sided H1: f(params) != value.

        Warning: this might be replaced with more general version that returns
        ContrastResults.
        currently uses chisquare distribution, use_f option not yet implemented

        Parameters
        ----------
        value : float or ndarray
            value of f(params) under the Null Hypothesis

        Returns
        -------
        statistic : float
            Value of the test statistic.
        pvalue : float
            The p-value for the hypothesis test, based and chisquare
            distribution and implies a two-sided hypothesis test
        """
        # TODO: add use_t option or not?
        m = self.predicted()
        v = self.cov()
        df_constraints = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return lmstat, stats.chi2.sf(lmstat, df_constraints)

    def var(self):
        """standard error for each equation (row) treated separately

        """
        g = self.grad()
        var = (np.dot(g, self.cov_params) * g).sum(-1)

        if var.ndim == 2:
            var = var.T
        return var

    def se_vectorized(self):
        """standard error for each equation (row) treated separately

        """
        var = self.var()
        return np.sqrt(var)

    def conf_int(self, alpha=0.05, use_t=False, df=None, var_extra=None,
                 predicted=None, se=None):
        """
        Confidence interval for predicted based on delta method.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        use_t : boolean
            If use_t is False (default), then the normal distribution is used
            for the confidence interval, otherwise the t distribution with
            `df` degrees of freedom is used.
        df : int or float
            degrees of freedom for t distribution. Only used and required if
            use_t is True.
        var_extra : None or array_like float
            Additional variance that is added to the variance based on the
            delta method. This can be used to obtain confidence intervalls for
            new observations (prediction interval).
        predicted : ndarray (float)
            Predicted value, can be used to avoid repeated calculations if it
            is already available.
        se : ndarray (float)
            Standard error, can be used to avoid repeated calculations if it
            is already available.

        Returns
        -------
        conf_int : array
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.
        """

        # TODO: predicted and se as arguments to avoid duplicate calculations
        # or leave unchanged?
        if not use_t:
            dist = stats.norm
            dist_args = ()
        else:
            if df is None:
                raise ValueError('t distribution requires df')
            dist = stats.t
            dist_args = (df,)

        if predicted is None:
            predicted = self.predicted()
        if se is None:
            se = self.se_vectorized()
        if var_extra is not None:
            se = np.sqrt(se**2 + var_extra)

        q = dist.ppf(1 - alpha / 2., *dist_args)
        lower = predicted - q * se
        upper = predicted + q * se
        ci = np.column_stack((lower, upper))
        if ci.shape[1] != 2:
            raise RuntimeError('something wrong: ci not 2 columns')
        return ci

    def summary(self, xname=None, alpha=0.05, title=None, use_t=False,
                df=None):
        """Summarize the Results of the nonlinear transformation.

        This provides a parameter table equivalent to `t_test` and reuses
        `ContrastResults`.

        Parameters
        -----------
        xname : list of strings, optional
            Default is `c_##` for ## in p the number of regressors
        alpha : float
            Significance level for the confidence intervals. Default is
            alpha = 0.05 which implies a confidence level of 95%.
        title : string, optional
            Title for the params table. If not None, then this replaces the
            default title
        use_t : boolean
            If use_t is False (default), then the normal distribution is used
            for the confidence interval, otherwise the t distribution with
            `df` degrees of freedom is used.
        df : int or float
            degrees of freedom for t distribution. Only used and required if
            use_t is True.

        Returns
        -------
        smry : string or Summary instance
            This contains a parameter results table in the case of t or z test
            in the same form as the parameter results table in the model
            results summary.
            For F or Wald test, the return is a string.
        """
        # this is an experimental reuse of ContrastResults
        from statsmodels.stats.contrast import ContrastResults
        predicted = self.predicted()
        se = self.se_vectorized()
        # TODO check shape for scalar case, ContrastResults requires iterable
        predicted = np.atleast_1d(predicted)
        if predicted.ndim > 1:
            predicted = predicted.squeeze()
        se = np.atleast_1d(se)

        statistic = predicted / se
        if use_t:
            df_resid = df
            cr = ContrastResults(effect=predicted, t=statistic, sd=se,
                                 df_denom=df_resid)
        else:
            cr = ContrastResults(effect=predicted, statistic=statistic, sd=se,
                                 df_denom=None, distribution='norm')

        return cr.summary(xname=xname, alpha=alpha, title=title)
