#!/usr/bin/env python

'''
Quantile regression model

Model parameters are estimated using iterated reweighted least squares. The
asymptotic covariance matrix estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
'''

import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
                                                 RegressionResults,
                                                 RegressionResultsWrapper)
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                             IterationLimitWarning)

class QuantReg(RegressionModel):
    '''Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.

    Parameters
    ----------
    endog : array or dataframe
        endogenous/response variable
    exog : array or dataframe
        exogenous/explanatory variable(s)

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P.(1999). Applied Econometrics Using MATLAB,

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    '''

    def __init__(self, endog, exog, **kwargs):
        self._check_kwargs(kwargs)
        super(QuantReg, self).__init__(endog, exog, **kwargs)

    def whiten(self, data):
        """
        QuantReg model whitener does nothing: returns data.
        """
        return data

    def fit(self, q=.5, vcov='robust', kernel='epa', bandwidth='hsheather',
            max_iter=1000, p_tol=1e-6, **kwargs):
        """
        Solve by Iterative Weighted Least Squares

        Parameters
        ----------
        q : float
            Quantile must be strictly between 0 and 1
        vcov : str, method used to calculate the variance-covariance matrix
            of the parameters. Default is ``robust``:

            - robust : heteroskedasticity robust standard errors (as suggested
              in Greene 6th edition)
            - iid : iid errors (as in Stata 12)

        kernel : str, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth : str, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        """

        if q <= 0 or q >= 1:
            raise Exception('q must be strictly between 0 and 1')

        kern_names = ['biw', 'cos', 'epa', 'gau', 'par']
        if kernel not in kern_names:
            raise Exception("kernel must be one of " + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]

        if bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        elif bandwidth == 'chamberlain':
            bandwidth = chamberlain
        else:
            raise Exception("bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'")

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        exog_rank = np.linalg.matrix_rank(self.exog)
        self.rank = exog_rank
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = self.nobs - self.rank
        n_iter = 0
        xstar = exog

        beta = np.ones(exog.shape[1])
        # TODO: better start, initial beta is used only for convergence check

        # Note the following does not work yet,
        # the iteration loop always starts with OLS as initial beta
        # if start_params is not None:
        #    if len(start_params) != rank:
        #       raise ValueError('start_params has wrong length')
        #       beta = start_params
        #    else:
        #       # start with OLS
        #       beta = np.dot(np.linalg.pinv(exog), endog)

        diff = 10
        cycle = False

        history = dict(params = [], mse=[])
        while n_iter < max_iter and diff > p_tol and not cycle:
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)

            mask = np.abs(resid) < .000001
            resid[mask] = ((resid[mask] >= 0) * 2 - 1) * .000001
            resid = np.where(resid < 0, q * resid, (1-q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))
            history['params'].append(beta)
            history['mse'].append(np.mean(resid*resid))

            if (n_iter >= 300) and (n_iter % 100 == 0):
                # check for convergence circle, should not happen
                for ii in range(2, 10):
                    if np.all(beta == history['params'][-ii]):
                        cycle = True
                        warnings.warn("Convergence cycle detected", ConvergenceWarning)
                        break

        if n_iter == max_iter:
            warnings.warn("Maximum number of iterations (" + str(max_iter) +
                          ") reached.", IterationLimitWarning)

        e = endog - np.dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        # h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog),
                iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1. / (nobs * h) * np.sum(kernel(e / h))

        if vcov == 'robust':
            d = np.where(e > 0, (q/fhat0)**2, ((1-q)/fhat0)**2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = xtxi @ xtdx @ xtxi
        elif vcov == 'iid':
            vcov = (1. / fhat0)**2 * q * (1 - q) * pinv(np.dot(exog.T, exog))
        else:
            raise Exception("vcov must be 'robust' or 'iid'")

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)

        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1. / fhat0
        lfit.bandwidth = h
        lfit.history = history

        return RegressionResultsWrapper(lfit)


def _parzen(u):
    z = np.where(np.abs(u) <= .5, 4./3 - 8. * u**2 + 8. * np.abs(u)**3,
                 8. * (1 - np.abs(u))**3 / 3.)
    z[np.abs(u) > 1] = 0
    return z


kernels = {}
kernels['biw'] = lambda u: 15. / 16 * (1 - u**2)**2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= .5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3. / 4 * (1-u**2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['gau'] = norm.pdf
kernels['par'] = _parzen
#kernels['bet'] = lambda u: np.where(np.abs(u) <= 1, .75 * (1 - u) * (1 + u), 0)
#kernels['log'] = lambda u: logistic.pdf(u) * (1 - logistic.pdf(u))
#kernels['tri'] = lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
#kernels['trw'] = lambda u: 35. / 32 * (1 - u**2)**3 * np.where(np.abs(u) <= 1, 1, 0)
#kernels['uni'] = lambda u: 1. / 2 * np.where(np.abs(u) <= 1, 1, 0)


def hall_sheather(n, q, alpha=.05):
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z)**2.
    den = 2. * z**2. + 1.
    h = n**(-1. / 3) * norm.ppf(1. - alpha / 2.)**(2./3) * (num / den)**(1./3)
    return h


def bofinger(n, q):
    num = 9. / 2 * norm.pdf(2 * norm.ppf(q))**4
    den = (2 * norm.ppf(q)**2 + 1)**2
    h = n**(-1. / 5) * (num / den)**(1. / 5)
    return h


def chamberlain(n, q, alpha=.05):
    return norm.ppf(1 - alpha / 2) * np.sqrt(q*(1 - q) / n)


class QuantRegResults(RegressionResults):
    '''Results instance for the QuantReg model'''

    @cache_readonly
    def prsquared(self):
        q = self.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        return 1 - np.sum(e) / np.sum(ered)

    #@cache_readonly
    def scale(self):
        return 1.

    @cache_readonly
    def bic(self):
        return np.nan

    @cache_readonly
    def aic(self):
        return np.nan

    @cache_readonly
    def llf(self):
        return np.nan

    @cache_readonly
    def rsquared(self):
        return np.nan

    @cache_readonly
    def rsquared_adj(self):
        return np.nan

    @cache_readonly
    def mse(self):
        return np.nan

    @cache_readonly
    def mse_model(self):
        return np.nan

    @cache_readonly
    def mse_total(self):
        return np.nan

    @cache_readonly
    def centered_tss(self):
        return np.nan

    @cache_readonly
    def uncentered_tss(self):
        return np.nan

    @cache_readonly
    def HC0_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC1_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC2_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC3_se(self):
        raise NotImplementedError

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        eigvals = self.eigenvals
        condno = self.condition_number

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None)
                    ]

        top_right = [('Pseudo R-squared:', ["%#8.4g" % self.prsquared]),
                     ('Bandwidth:', ["%#8.4g" % self.bandwidth]),
                     ('Sparsity:', ["%#8.4g" % self.sparsity]),
                     ('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None)
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        # add warnings/notes, added to text format only
        etext = []
        if eigvals[-1] < 1e-10:
            wstr = "The smallest eigenvalue is %6.3g. This might indicate "
            wstr += "that there are\n"
            wstr += "strong multicollinearity problems or that the design "
            wstr += "matrix is singular."
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended
            wstr = "The condition number is large, %6.3g. This might "
            wstr += "indicate that there are\n"
            wstr += "strong multicollinearity or other numerical "
            wstr += "problems."
            wstr = wstr % condno
            etext.append(wstr)

        if etext:
            smry.add_extra_txt(etext)

        return smry
