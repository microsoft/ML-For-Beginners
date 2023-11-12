# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 20:20:16 2011

Author: Josef Perktold
License: BSD-3

TODO:
check orientation, size and alpha should be increasing for interp1d,
but what is alpha? can be either sf or cdf probability
change it to use one consistent notation

check: instead of bound checking I could use the fill-value of the
interpolators
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf

from statsmodels.tools.decorators import cache_readonly


class TableDist:
    """
    Distribution, critical values and p-values from tables

    currently only 1 extra parameter, e.g. sample size

    Parameters
    ----------
    alpha : array_like, 1d
        probabiliy in the table, could be either sf (right tail) or cdf (left
        tail)
    size : array_like, 1d
        The sample sizes for the table
    crit_table : array_like, 2d
        The sample sizes in the table
        array with critical values for sample size in rows and probability in
        columns
    asymptotic : callable, optional
        Callable function with the form fn(nobs) that returns len(alpha)
        critical values where the critical value in position i corresponds to
        alpha[i]
    min_nobs : int, optional
        Minimum number of observations to use the asymptotic distribution. If
        not provided, uses max(size).
    max_nobs : int, optional
        Maximum number of observations to use the tabular distribution. If not
        provided, uses max(size)

    Notes
    -----
    size and alpha must be sorted and increasing.

    If both min_nobs and max_nobs are provided, then
    the critical values from the tabular distribution and the asymptotic
    distribution are linearly blended using the formula
    :math:`w cv_a + (1-w) cv_t` where the weight is
    :math:`w = (n - a_{min}) / (a_{max} - a_{min})`. This ensures the
    transition between the tabular and the asymptotic critical values is
    continuous. If these are not provided, then the asymptotic critical value
    is used for nobs > max(size).
    """

    def __init__(self, alpha, size, crit_table, asymptotic=None,
                 min_nobs=None, max_nobs=None):
        self.alpha = np.asarray(alpha)
        if self.alpha.ndim != 1:
            raise ValueError('alpha is not 1d')
        elif (np.diff(self.alpha) <= 0).any():
            raise ValueError('alpha is not sorted')
        self.size = np.asarray(size)
        if self.size.ndim != 1:
            raise ValueError('size is not 1d')
        elif (np.diff(self.size) <= 0).any():
            raise ValueError('size is not sorted')
        if self.size.ndim == 1:
            if (np.diff(alpha) <= 0).any():
                raise ValueError('alpha is not sorted')
        self.crit_table = np.asarray(crit_table)
        if self.crit_table.shape != (self.size.shape[0], self.alpha.shape[0]):
            raise ValueError('crit_table must have shape'
                             '(len(size), len(alpha))')

        self.n_alpha = len(alpha)
        self.signcrit = np.sign(np.diff(self.crit_table, 1).mean())
        if self.signcrit > 0:  # increasing
            self.critv_bounds = self.crit_table[:, [0, 1]]
        else:
            self.critv_bounds = self.crit_table[:, [1, 0]]
        self.asymptotic = None
        max_size = self.max_size = max(size)

        if asymptotic is not None:
            try:
                cv = asymptotic(self.max_size + 1)
            except Exception as exc:
                raise type(exc)('Calling asymptotic(self.size+1) failed. The '
                                'error message was:'
                                '\n\n{err_msg}'.format(err_msg=exc.args[0]))
            if len(cv) != len(alpha):
                raise ValueError('asymptotic does not return len(alpha) '
                                 'values')
            self.asymptotic = asymptotic

        self.min_nobs = max_size if min_nobs is None else min_nobs
        self.max_nobs = max_size if max_nobs is None else max_nobs
        if self.min_nobs > max_size:
            raise ValueError('min_nobs > max(size)')
        if self.max_nobs > max_size:
            raise ValueError('max_nobs > max(size)')

    @cache_readonly
    def polyn(self):
        polyn = [interp1d(self.size, self.crit_table[:, i])
                 for i in range(self.n_alpha)]
        return polyn

    @cache_readonly
    def poly2d(self):
        # check for monotonicity ?
        # fix this, interp needs increasing
        poly2d = interp2d(self.size, self.alpha, self.crit_table)
        return poly2d

    @cache_readonly
    def polyrbf(self):
        xs, xa = np.meshgrid(self.size.astype(float), self.alpha)
        polyrbf = Rbf(xs.ravel(), xa.ravel(), self.crit_table.T.ravel(),
                      function='linear')
        return polyrbf

    def _critvals(self, n):
        """
        Rows of the table, linearly interpolated for given sample size

        Parameters
        ----------
        n : float
            sample size, second parameter of the table

        Returns
        -------
        critv : ndarray, 1d
            critical values (ppf) corresponding to a row of the table

        Notes
        -----
        This is used in two step interpolation, or if we want to know the
        critical values for all alphas for any sample size that we can obtain
        through interpolation
        """
        if n > self.max_size:
            if self.asymptotic is not None:
                cv = self.asymptotic(n)
            else:
                raise ValueError('n is above max(size) and no asymptotic '
                                 'distribtuion is provided')
        else:
            cv = ([p(n) for p in self.polyn])
            if n > self.min_nobs:
                w = (n - self.min_nobs) / (self.max_nobs - self.min_nobs)
                w = min(1.0, w)
                a_cv = self.asymptotic(n)
                cv = w * a_cv + (1 - w) * cv

        return cv

    def prob(self, x, n):
        """
        Find pvalues by interpolation, either cdf(x)

        Returns extreme probabilities, 0.001 and 0.2, for out of range

        Parameters
        ----------
        x : array_like
            observed value, assumed to follow the distribution in the table
        n : float
            sample size, second parameter of the table

        Returns
        -------
        prob : array_like
            This is the probability for each value of x, the p-value in
            underlying distribution is for a statistical test.
        """
        critv = self._critvals(n)
        alpha = self.alpha

        if self.signcrit < 1:
            # reverse if critv is decreasing
            critv, alpha = critv[::-1], alpha[::-1]

        # now critv is increasing
        if np.size(x) == 1:
            if x < critv[0]:
                return alpha[0]
            elif x > critv[-1]:
                return alpha[-1]
            return interp1d(critv, alpha)(x)[()]
        else:
            # vectorized
            cond_low = (x < critv[0])
            cond_high = (x > critv[-1])
            cond_interior = ~np.logical_or(cond_low, cond_high)

            probs = np.nan * np.ones(x.shape)  # mistake if nan left
            probs[cond_low] = alpha[0]
            probs[cond_low] = alpha[-1]
            probs[cond_interior] = interp1d(critv, alpha)(x[cond_interior])

            return probs

    def crit(self, prob, n):
        """
        Returns interpolated quantiles, similar to ppf or isf

        use two sequential 1d interpolation, first by n then by prob

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob
        """
        prob = np.asarray(prob)
        alpha = self.alpha
        critv = self._critvals(n)

        # vectorized
        cond_ilow = (prob > alpha[0])
        cond_ihigh = (prob < alpha[-1])
        cond_interior = np.logical_or(cond_ilow, cond_ihigh)

        # scalar
        if prob.size == 1:
            if cond_interior:
                return interp1d(alpha, critv)(prob)
            else:
                return np.nan

        # vectorized
        quantile = np.nan * np.ones(prob.shape)  # nans for outside
        quantile[cond_interior] = interp1d(alpha, critv)(prob[cond_interior])
        return quantile

    def crit3(self, prob, n):
        """
        Returns interpolated quantiles, similar to ppf or isf

        uses Rbf to interpolate critical values as function of `prob` and `n`

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob, returns nan for arguments
            that are outside of the table bounds
        """
        prob = np.asarray(prob)
        alpha = self.alpha

        # vectorized
        cond_ilow = (prob > alpha[0])
        cond_ihigh = (prob < alpha[-1])
        cond_interior = np.logical_or(cond_ilow, cond_ihigh)

        # scalar
        if prob.size == 1:
            if cond_interior:
                return self.polyrbf(n, prob)
            else:
                return np.nan

        # vectorized
        quantile = np.nan * np.ones(prob.shape)  # nans for outside

        quantile[cond_interior] = self.polyrbf(n, prob[cond_interior])
        return quantile
