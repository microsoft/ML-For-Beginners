# TODO: Determine which tests are valid for GLSAR, and under what conditions
# TODO: Fix issue with constant and GLS
# TODO: GLS: add options Iterative GLS, for iterative fgls if sigma is None
# TODO: GLS: default if sigma is none should be two-step GLS
# TODO: Check nesting when performing model based tests, lr, wald, lm
"""
This module implements standard regression models:

Generalized Least Squares (GLS)
Ordinary Least Squares (OLS)
Weighted Least Squares (WLS)
Generalized Least Squares with autoregressive error terms GLSAR(p)

Models are specified with an endogenous response variable and an
exogenous design matrix and are fit using their `fit` method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

D. C. Montgomery and E.A. Peck. "Introduction to Linear Regression
    Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

R. Davidson and J.G. MacKinnon.  "Econometric Theory and Methods," Oxford,
    2004.

W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
"""
from __future__ import annotations

from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip

from typing import Literal, Sequence
import warnings

import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
# need import in module instead of lazily to copy `__doc__`
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
    InvalidTestWarning,
    ValueWarning,
    )
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like

from . import _prediction as pred

__docformat__ = 'restructuredtext en'

__all__ = ['GLS', 'WLS', 'OLS', 'GLSAR', 'PredictionResults',
           'RegressionResultsWrapper']


_fit_regularized_doc =\
        r"""
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : str
            Either 'elastic_net' or 'sqrt_lasso'.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        L1_wt : scalar
            The fraction of the penalty given to the L1 penalty term.
            Must be between 0 and 1 (inclusive).  If 0, the fit is a
            ridge fit, if 1 it is a lasso fit.
        start_params : array_like
            Starting values for ``params``.
        profile_scale : bool
            If True the penalized fit is computed using the profile
            (concentrated) log-likelihood for the Gaussian model.
            Otherwise the fit uses the residual sum of squares.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.

        Returns
        -------
        statsmodels.base.elastic_net.RegularizedResults
            The regularized results.

        Notes
        -----
        The elastic net uses a combination of L1 and L2 penalties.
        The implementation closely follows the glmnet package in R.

        The function that is minimized is:

        .. math::

            0.5*RSS/n + alpha*((1-L1\_wt)*|params|_2^2/2 + L1\_wt*|params|_1)

        where RSS is the usual regression sum of squares, n is the
        sample size, and :math:`|*|_1` and :math:`|*|_2` are the L1 and L2
        norms.

        For WLS and GLS, the RSS is calculated using the whitened endog and
        exog data.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.

        The square root lasso approach is a variation of the Lasso
        that is largely self-tuning (the optimal tuning parameter
        does not depend on the standard deviation of the regression
        errors).  If the errors are Gaussian, the tuning parameter
        can be taken to be

        alpha = 1.1 * np.sqrt(n) * norm.ppf(1 - 0.05 / (2 * p))

        where n is the sample size and p is the number of predictors.

        The square root lasso uses the following keyword arguments:

        zero_tol : float
            Coefficients below this threshold are treated as zero.

        The cvxopt module is required to estimate model using the square root
        lasso.

        References
        ----------
        .. [*] Friedman, Hastie, Tibshirani (2008).  Regularization paths for
           generalized linear models via coordinate descent.  Journal of
           Statistical Software 33(1), 1-22 Feb 2010.

        .. [*] A Belloni, V Chernozhukov, L Wang (2011).  Square-root Lasso:
           pivotal recovery of sparse signals via conic programming.
           Biometrika 98(4), 791-806. https://arxiv.org/pdf/1009.5689.pdf
        """


def _get_sigma(sigma, nobs):
    """
    Returns sigma (matrix, nobs by nobs) for GLS and the inverse of its
    Cholesky decomposition.  Handles dimensions and checks integrity.
    If sigma is None, returns None, None. Otherwise returns sigma,
    cholsigmainv.
    """
    if sigma is None:
        return None, None
    sigma = np.asarray(sigma).squeeze()
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, nobs)
    if sigma.ndim == 1:
        if sigma.shape != (nobs,):
            raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                             "array of shape %s x %s" % (nobs, nobs, nobs))
        cholsigmainv = 1/np.sqrt(sigma)
    else:
        if sigma.shape != (nobs, nobs):
            raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                             "array of shape %s x %s" % (nobs, nobs, nobs))
        cholsigmainv, info = dtrtri(cholesky(sigma, lower=True),
                                    lower=True, overwrite_c=True)
        if info > 0:
            raise np.linalg.LinAlgError('Cholesky decomposition of sigma '
                                        'yields a singular matrix')
        elif info < 0:
            raise ValueError('Invalid input to dtrtri (info = %d)' % info)
    return sigma, cholsigmainv


class RegressionModel(base.LikelihoodModel):
    """
    Base class for linear regression models. Should not be directly called.

    Intended for subclassing.
    """
    def __init__(self, endog, exog, **kwargs):
        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self.pinv_wexog: Float64Array | None = None
        self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])

    def initialize(self):
        """Initialize model components."""
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])

        self._df_model = None
        self._df_resid = None
        self.rank = None

    @property
    def df_model(self):
        """
        The model degree of freedom.

        The dof is defined as the rank of the regressor matrix minus 1 if a
        constant is included.
        """
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom.

        The dof is defined as the number of observations minus the rank of
        the regressor matrix.
        """

        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def whiten(self, x):
        """
        Whiten method that must be overwritten by individual models.

        Parameters
        ----------
        x : array_like
            Data to be whitened.
        """
        raise NotImplementedError("Subclasses must implement.")

    def fit(
            self,
            method: Literal["pinv", "qr"] = "pinv",
            cov_type: Literal[
                "nonrobust",
                "fixed scale",
                "HC0",
                "HC1",
                "HC2",
                "HC3",
                "HAC",
                "hac-panel",
                "hac-groupsum",
                "cluster",
            ] = "nonrobust",
            cov_kwds=None,
            use_t: bool | None = None,
            **kwargs
    ):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str, optional
            Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
            to solve the least squares problem. "qr" uses the QR
            factorization.
        cov_type : str, optional
            See `regression.linear_model.RegressionResults` for a description
            of the available covariance estimators.
        cov_kwds : list or None, optional
            See `linear_model.RegressionResults.get_robustcov_results` for a
            description required keywords for alternative covariance
            estimators.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.  Default behavior depends on cov_type. See
            `linear_model.RegressionResults.get_robustcov_results` for
            implementation details.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.

        Returns
        -------
        RegressionResults
            The model estimation results.

        See Also
        --------
        RegressionResults
            The results container.
        RegressionResults.get_robustcov_results
            A method to change the covariance estimator used when fitting the
            model.

        Notes
        -----
        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.
        """
        if method == "pinv":
            if not (hasattr(self, 'pinv_wexog') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):

                self.pinv_wexog, singular_values = pinv_extended(self.wexog)
                self.normalized_cov_params = np.dot(
                    self.pinv_wexog, np.transpose(self.pinv_wexog))

                # Cache these singular values for use later.
                self.wexog_singular_values = singular_values
                self.rank = np.linalg.matrix_rank(np.diag(singular_values))

            beta = np.dot(self.pinv_wexog, self.wendog)

        elif method == "qr":
            if not (hasattr(self, 'exog_Q') and
                    hasattr(self, 'exog_R') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):
                Q, R = np.linalg.qr(self.wexog)
                self.exog_Q, self.exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

                # Cache singular values from R.
                self.wexog_singular_values = np.linalg.svd(R, 0, 0)
                self.rank = np.linalg.matrix_rank(R)
            else:
                Q, R = self.exog_Q, self.exog_R
            # Needed for some covariance estimators, see GH #8157
            self.pinv_wexog = np.linalg.pinv(self.wexog)
            # used in ANOVA
            self.effects = effects = np.dot(Q.T, self.wendog)
            beta = np.linalg.solve(R, effects)
        else:
            raise ValueError('method has to be "pinv" or "qr"')

        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        if isinstance(self, OLS):
            lfit = OLSResults(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        else:
            lfit = RegressionResults(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                **kwargs)
        return RegressionResultsWrapper(lfit)

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model.
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        array_like
            An array of fitted values.

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        # JP: this does not look correct for GLMAR
        # SS: it needs its own predict method

        if exog is None:
            exog = self.exog

        return np.dot(exog, params)

    def get_distribution(self, params, scale, exog=None, dist_class=None):
        """
        Construct a random number generator for the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters (regression coefficients).
        scale : scalar
            The variance parameter.
        exog : array_like
            The predictor variable matrix.
        dist_class : class
            A random number generator class.  Must take 'loc' and 'scale'
            as arguments and return a random number generator implementing
            an ``rvs`` method for simulating random values. Defaults to normal.

        Returns
        -------
        gen
            Frozen random number generator object with mean and variance
            determined by the fitted linear model.  Use the ``rvs`` method
            to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``,
        the returned random number generator must be called with
        ``gen.rvs(n)`` where ``n`` is the number of observations in
        the data set used to fit the model.  If any other value is
        used for ``n``, misleading results will be produced.
        """
        fit = self.predict(params, exog)
        if dist_class is None:
            from scipy.stats.distributions import norm
            dist_class = norm
        gen = dist_class(loc=fit, scale=np.sqrt(scale))
        return gen


class GLS(RegressionModel):
    __doc__ = r"""
    Generalized Least Squares

    %(params)s
    sigma : scalar or array
        The array or scalar `sigma` is the weighting matrix of the covariance.
        The default is None for no scaling.  If `sigma` is a scalar, it is
        assumed that `sigma` is an n x n diagonal matrix with the given
        scalar, `sigma` as the value of each diagonal element.  If `sigma`
        is an n-length vector, then `sigma` is assumed to be a diagonal
        matrix with the given `sigma` on the diagonal.  This should be the
        same as WLS.
    %(extra_params)s

    Attributes
    ----------
    pinv_wexog : ndarray
        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse of `wexog`.
    cholsimgainv : ndarray
        The transpose of the Cholesky decomposition of the pseudoinverse.
    df_model : float
        p - 1, where p is the number of regressors including the intercept.
        of freedom.
    df_resid : float
        Number of observations n less the number of parameters p.
    llf : float
        The value of the likelihood function of the fitted model.
    nobs : float
        The number of observations n.
    normalized_cov_params : ndarray
        p x p array :math:`(X^{T}\Sigma^{-1}X)^{-1}`
    results : RegressionResults instance
        A property that returns the RegressionResults class if fit.
    sigma : ndarray
        `sigma` is the n x n covariance structure of the error terms.
    wexog : ndarray
        Design matrix whitened by `cholsigmainv`
    wendog : ndarray
        Response variable whitened by `cholsigmainv`

    See Also
    --------
    WLS : Fit a linear model using Weighted Least Squares.
    OLS : Fit a linear model using Ordinary Least Squares.

    Notes
    -----
    If sigma is a function of the data making one of the regressors
    a constant, then the current postestimation statistics will not be correct.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> ols_resid = sm.OLS(data.endog, data.exog).fit().resid
    >>> res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()
    >>> rho = res_fit.params

    `rho` is a consistent estimator of the correlation of the residuals from
    an OLS fit of the longley data.  It is assumed that this is the true rho
    of the AR process data.

    >>> from scipy.linalg import toeplitz
    >>> order = toeplitz(np.arange(16))
    >>> sigma = rho**order

    `sigma` is an n x n matrix of the autocorrelation structure of the
    data.

    >>> gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
    >>> gls_results = gls_model.fit()
    >>> print(gls_results.summary())
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, sigma=None, missing='none', hasconst=None,
                 **kwargs):
        if type(self) is GLS:
            self._check_kwargs(kwargs)
        # TODO: add options igls, for iterative fgls if sigma is None
        # TODO: default if sigma is none should be two-step GLS
        sigma, cholsigmainv = _get_sigma(sigma, len(endog))

        super(GLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, sigma=sigma,
                                  cholsigmainv=cholsigmainv, **kwargs)

        # store attribute names for data arrays
        self._data_attr.extend(['sigma', 'cholsigmainv'])

    def whiten(self, x):
        """
        GLS whiten method.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        ndarray
            The value np.dot(cholsigmainv,X).

        See Also
        --------
        GLS : Fit a linear model using Generalized Least Squares.
        """
        x = np.asarray(x)
        if self.sigma is None or self.sigma.shape == ():
            return x
        elif self.sigma.ndim == 1:
            if x.ndim == 1:
                return x * self.cholsigmainv
            else:
                return x * self.cholsigmainv[:, None]
        else:
            return np.dot(self.cholsigmainv, x)

    def loglike(self, params):
        r"""
        Compute the value of the Gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array_like
            The model parameters.

        Returns
        -------
        float
            The value of the log-likelihood function for a GLS Model.

        Notes
        -----
        The log-likelihood function for the normal distribution is

        .. math:: -\frac{n}{2}\log\left(\left(Y-\hat{Y}\right)^{\prime}
                   \left(Y-\hat{Y}\right)\right)
                  -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
                  -\frac{1}{2}\log\left(\left|\Sigma\right|\right)

        Y and Y-hat are whitened.
        """
        # TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params))**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with likelihood constant
        if np.any(self.sigma):
            # FIXME: robust-enough check? unneeded if _det_sigma gets defined
            if self.sigma.ndim == 2:
                det = np.linalg.slogdet(self.sigma)
                llf -= .5*det[1]
            else:
                llf -= 0.5*np.sum(np.log(self.sigma))
            # with error covariance matrix
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute weights for calculating Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        if self.sigma is None or self.sigma.shape == ():
            return np.ones(self.exog.shape[0])
        elif self.sigma.ndim == 1:
            return self.cholsigmainv
        else:
            return np.diag(self.cholsigmainv)

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        # Need to adjust since RSS/n term in elastic net uses nominal
        # n in denominator
        if self.sigma is not None:
            if self.sigma.ndim == 2:
                var_obs = np.diag(self.sigma)
            elif self.sigma.ndim == 1:
                var_obs = self.sigma
            else:
                raise ValueError("sigma should be 1-dim or 2-dim")

            alpha = alpha * np.sum(1 / var_obs) / len(self.endog)

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        from statsmodels.base.elastic_net import (
            RegularizedResults,
            RegularizedResultsWrapper,
        )
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)


class WLS(RegressionModel):
    __doc__ = """
    Weighted Least Squares

    The weights are presumed to be (proportional to) the inverse of
    the variance of the observations.  That is, if the variables are
    to be transformed by 1/sqrt(W) you must supply weights = 1/W.

    %(params)s
    weights : array_like, optional
        A 1d array of weights.  If you supply 1/W then the variables are
        pre- multiplied by 1/sqrt(W).  If no weights are supplied the
        default value is 1 and WLS results are the same as OLS.
    %(extra_params)s

    Attributes
    ----------
    weights : ndarray
        The stored weights supplied as an argument.

    See Also
    --------
    GLS : Fit a linear model using Generalized Least Squares.
    OLS : Fit a linear model using Ordinary Least Squares.

    Notes
    -----
    If the weights are a function of the data, then the post estimation
    statistics such as fvalue and mse_model might not be correct, as the
    package does not yet support no-constant regression.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> wls_model = sm.WLS(Y,X, weights=list(range(1,8)))
    >>> results = wls_model.fit()
    >>> results.params
    array([ 2.91666667,  0.0952381 ])
    >>> results.tvalues
    array([ 2.0652652 ,  0.35684428])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([ 2.91666667]), sd=array([[ 1.41224801]]),
     t=array([[ 2.0652652]]), p=array([[ 0.04690139]]), df_denom=5>
    >>> print(results.f_test([0, 1]))
    <F test: F=array([[ 0.12733784]]), p=[[ 0.73577409]], df_denom=5, df_num=1>
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, weights=1., missing='none', hasconst=None,
                 **kwargs):
        if type(self) is WLS:
            self._check_kwargs(kwargs)
        weights = np.array(weights)
        if weights.shape == ():
            if (missing == 'drop' and 'missing_idx' in kwargs and
                    kwargs['missing_idx'] is not None):
                # patsy may have truncated endog
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        # handle case that endog might be of len == 1
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super(WLS, self).__init__(endog, exog, missing=missing,
                                  weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, x):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights).

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The whitened values sqrt(weights)*X.
        """

        x = np.asarray(x)
        if x.ndim == 1:
            return x * np.sqrt(self.weights)
        elif x.ndim == 2:
            return np.sqrt(self.weights)[:, None] * x

    def loglike(self, params):
        r"""
        Compute the value of the gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array_like
            The parameter estimates.

        Returns
        -------
        float
            The value of the log-likelihood function for a WLS Model.

        Notes
        -----
        .. math:: -\frac{n}{2}\log SSR
                  -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
                  -\frac{1}{2}\log\left(\left|W\right|\right)

        where :math:`W` is a diagonal weight matrix matrix and
        :math:`SSR=\left(Y-\hat{Y}\right)^\prime W \left(Y-\hat{Y}\right)` is
        the sum of the squared weighted residuals.
        """
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params))**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        llf += 0.5 * np.sum(np.log(self.weights))
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute the weights for calculating the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        return self.weights

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):
        # Docstring attached below
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        # Need to adjust since RSS/n in elastic net uses nominal n in
        # denominator
        alpha = alpha * np.sum(self.weights) / len(self.weights)

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        from statsmodels.base.elastic_net import (
            RegularizedResults,
            RegularizedResultsWrapper,
        )
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)


class OLS(WLS):
    __doc__ = """
    Ordinary Least Squares

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.

    See Also
    --------
    WLS : Fit a linear model using Weighted Least Squares.
    GLS : Fit a linear model using Generalized Least Squares.

    Notes
    -----
    No constant is added by the model unless you are using formulas.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    >>> Y = duncan_prestige.data['income']
    >>> X = duncan_prestige.data['education']
    >>> X = sm.add_constant(X)
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> results.params
    const        10.603498
    education     0.594859
    dtype: float64

    >>> results.tvalues
    const        2.039813
    education    6.892802
    dtype: float64

    >>> print(results.t_test([1, 0]))
                                 Test for Constraints
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    c0            10.6035      5.198      2.040      0.048       0.120      21.087
    ==============================================================================

    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[159.63031026]]), p=1.2607168903696672e-20,
     df_denom=43, df_num=2>
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 **kwargs):
        if "weights" in kwargs:
            msg = ("Weights are not supported in OLS and will be ignored"
                   "An exception will be raised in the next version.")
            warnings.warn(msg, ValueWarning)
        super(OLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, **kwargs)
        if "weights" in self._init_keys:
            self._init_keys.remove("weights")

        if type(self) is OLS:
            self._check_kwargs(kwargs, ["offset"])

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        float
            The likelihood function evaluated at params.
        """
        nobs2 = self.nobs / 2.0
        nobs = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        if hasattr(self, 'offset'):
            resid -= self.offset
        ssr = np.sum(resid**2)
        if scale is None:
            # profile log likelihood
            llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
        else:
            # log-likelihood
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2*scale)
        return llf

    def whiten(self, x):
        """
        OLS model whitener does nothing.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The input array unmodified.

        See Also
        --------
        OLS : Fit a linear model using Ordinary Least Squares.
        """
        return x

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The score vector.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            return -self.nobs * sdr / ssr
        else:
            return -sdr / scale

    def _setup_score_hess(self):
        y = self.wendog
        if hasattr(self, 'offset'):
            y = y - self.offset
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog = np.dot(self.wexog.T, y)

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The Hessian matrix.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            ssrp = -2*self._wexog_x_wendog + 2*xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr**2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Calculate the weights for the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        return np.ones(self.exog.shape[0])

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):

        # In the future we could add support for other penalties, e.g. SCAD.
        if method not in ("elastic_net", "sqrt_lasso"):
            msg = "Unknown method '%s' for fit_regularized" % method
            raise ValueError(msg)

        # Set default parameters.
        defaults = {"maxiter":  50, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-8}
        defaults.update(kwargs)

        if method == "sqrt_lasso":
            from statsmodels.base.elastic_net import (
                RegularizedResults,
                RegularizedResultsWrapper,
            )
            params = self._sqrt_lasso(alpha, refit, defaults["zero_tol"])
            results = RegularizedResults(self, params)
            return RegularizedResultsWrapper(results)

        from statsmodels.base.elastic_net import fit_elasticnet

        if L1_wt == 0:
            return self._fit_ridge(alpha)

        # If a scale parameter is passed in, the non-profile
        # likelihood (residual sum of squares divided by -2) is used,
        # otherwise the profile likelihood is used.
        if profile_scale:
            loglike_kwds = {}
            score_kwds = {}
            hess_kwds = {}
        else:
            loglike_kwds = {"scale": 1}
            score_kwds = {"scale": 1}
            hess_kwds = {"scale": 1}

        return fit_elasticnet(self, method=method,
                              alpha=alpha,
                              L1_wt=L1_wt,
                              start_params=start_params,
                              loglike_kwds=loglike_kwds,
                              score_kwds=score_kwds,
                              hess_kwds=hess_kwds,
                              refit=refit,
                              check_step=False,
                              **defaults)

    def _sqrt_lasso(self, alpha, refit, zero_tol):

        try:
            import cvxopt
        except ImportError:
            msg = 'sqrt_lasso fitting requires the cvxopt module'
            raise ValueError(msg)

        n = len(self.endog)
        p = self.exog.shape[1]

        h0 = cvxopt.matrix(0., (2*p+1, 1))
        h1 = cvxopt.matrix(0., (n+1, 1))
        h1[1:, 0] = cvxopt.matrix(self.endog, (n, 1))

        G0 = cvxopt.spmatrix([], [], [], (2*p+1, 2*p+1))
        for i in range(1, 2*p+1):
            G0[i, i] = -1
        G1 = cvxopt.matrix(0., (n+1, 2*p+1))
        G1[0, 0] = -1
        G1[1:, 1:p+1] = self.exog
        G1[1:, p+1:] = -self.exog

        c = cvxopt.matrix(alpha / n, (2*p + 1, 1))
        c[0] = 1 / np.sqrt(n)

        from cvxopt import solvers
        solvers.options["show_progress"] = False

        rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
        x = np.asarray(rslt['x']).flat
        bp = x[1:p+1]
        bn = x[p+1:]
        params = bp - bn

        if not refit:
            return params

        ii = np.flatnonzero(np.abs(params) > zero_tol)
        rfr = OLS(self.endog, self.exog[:, ii]).fit()
        params *= 0
        params[ii] = rfr.params

        return params

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """

        u, s, vt = np.linalg.svd(self.exog, 0)
        v = vt.T
        q = np.dot(u.T, self.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * self.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)

        from statsmodels.base.elastic_net import RegularizedResults
        return RegularizedResults(self, params)


class GLSAR(GLS):
    __doc__ = """
    Generalized Least Squares with AR covariance structure

    %(params)s
    rho : int
        The order of the autoregressive covariance.
    %(extra_params)s

    Notes
    -----
    GLSAR is considered to be experimental.
    The linear autoregressive process of order p--AR(p)--is defined as:
    TODO

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> Y = [1,3,4,5,8,10,9]
    >>> model = sm.GLSAR(Y, X, rho=2)
    >>> for i in range(6):
    ...     results = model.fit()
    ...     print("AR coefficients: {0}".format(model.rho))
    ...     rho, sigma = sm.regression.yule_walker(results.resid,
    ...                                            order=model.order)
    ...     model = sm.GLSAR(Y, X, rho)
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.6104153  -0.86656458]
    AR coefficients: [-0.60439494 -0.857867  ]
    AR coefficients: [-0.6048218  -0.85846157]
    AR coefficients: [-0.60479146 -0.85841922]
    >>> results.params
    array([-0.66661205,  1.60850853])
    >>> results.tvalues
    array([ -2.10304127,  21.8047269 ])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([-0.66661205]), sd=array([[ 0.31697526]]),
     t=array([[-2.10304127]]), p=array([[ 0.06309969]]), df_denom=3>
    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[ 1815.23061844]]), p=[[ 0.00002372]],
     df_denom=3, df_num=2>

    Or, equivalently

    >>> model2 = sm.GLSAR(Y, X, rho=2)
    >>> res = model2.iterative_fit(maxiter=6)
    >>> model2.rho
    array([-0.60479146, -0.85841922])
    """ % {'params': base._model_params_doc,
           'extra_params': base._missing_param_doc + base._extra_param_doc}
    # TODO: Complete docstring

    def __init__(self, endog, exog=None, rho=1, missing='none', hasconst=None,
                 **kwargs):
        # this looks strange, interpreting rho as order if it is int
        if isinstance(rho, (int, np.integer)):
            self.order = int(rho)
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0, 1]:
                raise ValueError("AR parameters must be a scalar or a vector")
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            # JP this looks wrong, should be a regression on constant
            # results for rho estimate now identical to yule-walker on y
            # super(AR, self).__init__(endog, add_constant(endog))
            super(GLSAR, self).__init__(endog, np.ones((endog.shape[0], 1)),
                                        missing=missing, hasconst=None,
                                        **kwargs)
        else:
            super(GLSAR, self).__init__(endog, exog, missing=missing,
                                        **kwargs)

    def iterative_fit(self, maxiter=3, rtol=1e-4, **kwargs):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated iteratively.

        Parameters
        ----------
        maxiter : int, optional
            The number of iterations.
        rtol : float, optional
            Relative tolerance between estimated coefficients to stop the
            estimation.  Stops if max(abs(last - current) / abs(last)) < rtol.
        **kwargs
            Additional keyword arguments passed to `fit`.

        Returns
        -------
        RegressionResults
            The results computed using an iterative fit.
        """
        # TODO: update this after going through example.
        converged = False
        i = -1  # need to initialize for maxiter < 1 (skip loop)
        history = {'params': [], 'rho': [self.rho]}
        for i in range(maxiter - 1):
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            self.initialize()
            results = self.fit()
            history['params'].append(results.params)
            if i == 0:
                last = results.params
            else:
                diff = np.max(np.abs(last - results.params) / np.abs(last))
                if diff < rtol:
                    converged = True
                    break
                last = results.params
            self.rho, _ = yule_walker(results.resid,
                                      order=self.order, df=None)
            history['rho'].append(self.rho)

        # why not another call to self.initialize
        # Use kwarg to insert history
        if not converged and maxiter > 0:
            # maxiter <= 0 just does OLS
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            self.initialize()

        # if converged then this is a duplicate fit, because we did not
        # update rho
        results = self.fit(history=history, **kwargs)
        results.iter = i + 1
        # add last fit to history, not if duplicate fit
        if not converged:
            results.history['params'].append(results.params)
            results.iter += 1

        results.converged = converged

        return results

    def whiten(self, x):
        """
        Whiten a series of columns according to an AR(p) covariance structure.

        Whitening using this method drops the initial p observations.

        Parameters
        ----------
        x : array_like
            The data to be whitened.

        Returns
        -------
        ndarray
            The whitened data.
        """
        # TODO: notation for AR process
        x = np.asarray(x, np.float64)
        _x = x.copy()

        # the following loops over the first axis,  works for 1d and nd
        for i in range(self.order):
            _x[(i + 1):] = _x[(i + 1):] - self.rho[i] * x[0:-(i + 1)]
        return _x[self.order:]


def yule_walker(x, order=1, method="adjusted", df=None, inv=False,
                demean=True):
    """
    Estimate AR(p) parameters from a sequence using the Yule-Walker equations.

    Adjusted or maximum-likelihood estimator (mle)

    Parameters
    ----------
    x : array_like
        A 1d array.
    order : int, optional
        The order of the autoregressive process.  Default is 1.
    method : str, optional
       Method can be 'adjusted' or 'mle' and this determines
       denominator in estimate of autocorrelation function (ACF) at
       lag k. If 'mle', the denominator is n=X.shape[0], if 'adjusted'
       the denominator is n-k.  The default is adjusted.
    df : int, optional
       Specifies the degrees of freedom. If `df` is supplied, then it
       is assumed the X has `df` degrees of freedom rather than `n`.
       Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is
        False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho : ndarray
        AR(p) coefficients computed using the Yule-Walker method.
    sigma : float
        The estimate of the residual standard deviation.

    See Also
    --------
    burg : Burg's AR estimator.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Autoregressive_moving_average_model for
    further details.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma = sm.regression.yule_walker(data.endog, order=4,
    ...                                        method="mle")

    >>> rho
    array([ 1.28310031, -0.45240924, -0.20770299,  0.04794365])
    >>> sigma
    16.808022730464351
    """
    # TODO: define R better, look back at notes and technical notes on YW.
    # First link here is useful
    # http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YuleWalkerAndMore.htm

    method = string_like(
        method, "method", options=("adjusted", "unbiased", "mle")
    )
    if method == "unbiased":
        warnings.warn(
            "unbiased is deprecated in factor of adjusted to reflect that the "
            "term is adjusting the sample size used in the autocovariance "
            "calculation rather than estimating an unbiased autocovariance. "
            "After release 0.13, using 'unbiased' will raise.",
            FutureWarning,
        )
        method = "adjusted"

    if method not in ("adjusted", "mle"):
        raise ValueError("ACF estimation method must be 'adjusted' or 'MLE'")
    x = np.array(x, dtype=np.float64)
    if demean:
        x -= x.mean()
    n = df or x.shape[0]

    # this handles df_resid ie., n - p
    adj_needed = method == "adjusted"

    if x.ndim > 1 and x.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")
    r = np.zeros(order+1, np.float64)
    r[0] = (x ** 2).sum() / n
    for k in range(1, order+1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * adj_needed)
    R = toeplitz(r[:-1])

    try:
        rho = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            warnings.warn("Matrix is singular. Using pinv.", ValueWarning)
            rho = np.linalg.pinv(R) @ r[1:]
        else:
            raise

    sigmasq = r[0] - (r[1:]*rho).sum()
    if not np.isnan(sigmasq) and sigmasq > 0:
        sigma = np.sqrt(sigmasq)
    else:
        sigma = np.nan
    if inv:
        return rho, sigma, np.linalg.inv(R)
    else:
        return rho, sigma


def burg(endog, order=1, demean=True):
    """
    Compute Burg's AP(p) parameter estimator.

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    order : int, optional
        Order of the AR.  Default is 1.
    demean : bool, optional
        Flag indicating to subtract the mean from endog before estimation.

    Returns
    -------
    rho : ndarray
        The AR(p) coefficients computed using Burg's algorithm.
    sigma2 : float
        The estimate of the residual variance.

    See Also
    --------
    yule_walker : Estimate AR parameters using the Yule-Walker method.

    Notes
    -----
    AR model estimated includes a constant that is estimated using the sample
    mean (see [1]_). This value is not reported.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma2 = sm.regression.linear_model.burg(data.endog, order=4)

    >>> rho
    array([ 1.30934186, -0.48086633, -0.20185982,  0.05501941])
    >>> sigma2
    271.2467306963966
    """
    # Avoid circular imports
    from statsmodels.tsa.stattools import levinson_durbin_pacf, pacf_burg

    endog = np.squeeze(np.asarray(endog))
    if endog.ndim != 1:
        raise ValueError('endog must be 1-d or squeezable to 1-d.')
    order = int(order)
    if order < 1:
        raise ValueError('order must be an integer larger than 1')
    if demean:
        endog = endog - endog.mean()
    pacf, sigma = pacf_burg(endog, order, demean=demean)
    ar, _ = levinson_durbin_pacf(pacf)
    return ar, sigma[-1]


class RegressionResults(base.LikelihoodModelResults):
    r"""
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.

    Attributes
    ----------
    pinv_wexog
        See model class docstring for implementation details.
    cov_type
        Parameter covariance estimator used for standard errors and t-stats.
    df_model
        Model degrees of freedom. The number of regressors `p`. Does not
        include the constant if one is present.
    df_resid
        Residual degrees of freedom. `n - p - 1`, if a constant is present.
        `n - p` if a constant is not included.
    het_scale
        adjusted squared residuals for heteroscedasticity robust standard
        errors. Is only available after `HC#_se` or `cov_HC#` is called.
        See HC#_se for more information.
    history
        Estimation history for iterative estimators.
    model
        A pointer to the model instance that called fit() or results.
    params
        The linear coefficients that minimize the least squares
        criterion.  This is usually called Beta for the classical
        linear model.
    """

    _cache = {}  # needs to be a class attribute for scale setter?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        super(RegressionResults, self).__init__(
            model, params, normalized_cov_params, scale)

        self._cache = {}
        if hasattr(model, 'wexog_singular_values'):
            self._wexog_singular_values = model.wexog_singular_values
        else:
            self._wexog_singular_values = None

        self.df_model = model.df_model
        self.df_resid = model.df_resid

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {
                'description': 'Standard Errors assume that the ' +
                'covariance matrix of the errors is correctly ' +
                'specified.'}
            if use_t is None:
                use_t = True  # TODO: class default
            self.use_t = use_t
        else:
            if cov_kwds is None:
                cov_kwds = {}
            if 'use_t' in cov_kwds:
                # TODO: we want to get rid of 'use_t' in cov_kwds
                use_t_2 = cov_kwds.pop('use_t')
                if use_t is None:
                    use_t = use_t_2
                # TODO: warn or not?
            self.get_robustcov_results(cov_type=cov_type, use_self=True,
                                       use_t=use_t, **cov_kwds)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def conf_int(self, alpha=.05, cols=None):
        """
        Compute the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Columns to include in returned confidence intervals.

        Returns
        -------
        array_like
            The confidence intervals.

        Notes
        -----
        The confidence interval is based on Student's t-distribution.
        """
        # keep method for docstring for now
        ci = super(RegressionResults, self).conf_int(alpha=alpha, cols=cols)
        return ci

    @cache_readonly
    def nobs(self):
        """Number of observations n."""
        return float(self.model.wexog.shape[0])

    @cache_readonly
    def fittedvalues(self):
        """The predicted values for the original (unwhitened) design."""
        return self.model.predict(self.params, self.model.exog)

    @cache_readonly
    def wresid(self):
        """
        The residuals of the transformed/whitened regressand and regressor(s).
        """
        return self.model.wendog - self.model.predict(
            self.params, self.model.wexog)

    @cache_readonly
    def resid(self):
        """The residuals of the model."""
        return self.model.endog - self.model.predict(
            self.params, self.model.exog)

    # TODO: fix writable example
    @cache_writable()
    def scale(self):
        """
        A scale factor for the covariance matrix.

        The Default value is ssr/(n-p).  Note that the square root of `scale`
        is often called the standard error of the regression.
        """
        wresid = self.wresid
        return np.dot(wresid, wresid) / self.df_resid

    @cache_readonly
    def ssr(self):
        """Sum of squared (whitened) residuals."""
        wresid = self.wresid
        return np.dot(wresid, wresid)

    @cache_readonly
    def centered_tss(self):
        """The total (weighted) sum of squares centered about the mean."""
        model = self.model
        weights = getattr(model, 'weights', None)
        sigma = getattr(model, 'sigma', None)
        if weights is not None:
            mean = np.average(model.endog, weights=weights)
            return np.sum(weights * (model.endog - mean)**2)
        elif sigma is not None:
            # Exactly matches WLS when sigma is diagonal
            iota = np.ones_like(model.endog)
            iota = model.whiten(iota)
            mean = model.wendog.dot(iota) / iota.dot(iota)
            err = model.endog - mean
            err = model.whiten(err)
            return np.sum(err**2)
        else:
            centered_endog = model.wendog - model.wendog.mean()
            return np.dot(centered_endog, centered_endog)

    @cache_readonly
    def uncentered_tss(self):
        """
        Uncentered sum of squares.

        The sum of the squared values of the (whitened) endogenous response
        variable.
        """
        wendog = self.model.wendog
        return np.dot(wendog, wendog)

    @cache_readonly
    def ess(self):
        """
        The explained sum of squares.

        If a constant is present, the centered total sum of squares minus the
        sum of squared residuals. If there is no constant, the uncentered total
        sum of squares is used.
        """

        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        """
        R-squared of the model.

        This is defined here as 1 - `ssr`/`centered_tss` if the constant is
        included in the model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
        """
        if self.k_constant:
            return 1 - self.ssr/self.centered_tss
        else:
            return 1 - self.ssr/self.uncentered_tss

    @cache_readonly
    def rsquared_adj(self):
        """
        Adjusted R-squared.

        This is defined here as 1 - (`nobs`-1)/`df_resid` * (1-`rsquared`)
        if a constant is included and 1 - `nobs`/`df_resid` * (1-`rsquared`) if
        no constant is included.
        """
        return 1 - (np.divide(self.nobs - self.k_constant, self.df_resid)
                    * (1 - self.rsquared))

    @cache_readonly
    def mse_model(self):
        """
        Mean squared error the model.

        The explained sum of squares divided by the model degrees of freedom.
        """
        if np.all(self.df_model == 0.0):
            return np.full_like(self.ess, np.nan)
        return self.ess/self.df_model

    @cache_readonly
    def mse_resid(self):
        """
        Mean squared error of the residuals.

        The sum of squared residuals divided by the residual degrees of
        freedom.
        """
        if np.all(self.df_resid == 0.0):
            return np.full_like(self.ssr, np.nan)
        return self.ssr/self.df_resid

    @cache_readonly
    def mse_total(self):
        """
        Total mean squared error.

        The uncentered total sum of squares divided by the number of
        observations.
        """
        if np.all(self.df_resid + self.df_model == 0.0):
            return np.full_like(self.centered_tss, np.nan)
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @cache_readonly
    def fvalue(self):
        """
        F-statistic of the fully specified model.

        Calculated as the mean squared error of the model divided by the mean
        squared error of the residuals if the nonrobust covariance is used.
        Otherwise computed using a Wald-like quadratic form that tests whether
        all coefficients (excluding the constant) are zero.
        """
        if hasattr(self, 'cov_type') and self.cov_type != 'nonrobust':
            # with heteroscedasticity or correlation robustness
            k_params = self.normalized_cov_params.shape[0]
            mat = np.eye(k_params)
            const_idx = self.model.data.const_idx
            # TODO: What if model includes implicit constant, e.g. all
            #       dummies but no constant regressor?
            # TODO: Restats as LM test by projecting orthogonalizing
            #       to constant?
            if self.model.data.k_constant == 1:
                # if constant is implicit, return nan see #2444
                if const_idx is None:
                    return np.nan

                idx = lrange(k_params)
                idx.pop(const_idx)
                mat = mat[idx]  # remove constant
                if mat.size == 0:  # see  #3642
                    return np.nan
            ft = self.f_test(mat)
            # using backdoor to set another attribute that we already have
            self._cache['f_pvalue'] = float(ft.pvalue)
            return float(ft.fvalue)
        else:
            # for standard homoscedastic case
            return self.mse_model/self.mse_resid

    @cache_readonly
    def f_pvalue(self):
        """The p-value of the F-statistic."""
        # Special case for df_model 0
        if self.df_model == 0:
            return np.full_like(self.fvalue, np.nan)
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)

    @cache_readonly
    def bse(self):
        """The standard errors of the parameter estimates."""
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def aic(self):
        r"""
        Akaike's information criteria.

        For a model with a constant :math:`-2llf + 2(df\_model + 1)`. For a
        model without a constant :math:`-2llf + 2(df\_model)`.
        """
        return self.info_criteria("aic")

    @cache_readonly
    def bic(self):
        r"""
        Bayes' information criteria.

        For a model with a constant :math:`-2llf + \log(n)(df\_model+1)`.
        For a model without a constant :math:`-2llf + \log(n)(df\_model)`.
        """
        return self.info_criteria("bic")

    def info_criteria(self, crit, dk_params=0):
        """Return an information criterion for the model.

        Parameters
        ----------
        crit : string
            One of 'aic', 'bic', 'aicc' or 'hqic'.
        dk_params : int or float
            Correction to the number of parameters used in the information
            criterion. By default, only mean parameters are included, the
            scale parameter is not included in the parameter count.
            Use ``dk_params=1`` to include scale in the parameter count.

        Returns
        -------
        Value of information criterion.

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        crit = crit.lower()
        k_params = self.df_model + self.k_constant + dk_params

        if crit == "aic":
            return -2 * self.llf + 2 * k_params
        elif crit == "bic":
            bic = -2*self.llf + np.log(self.nobs) * k_params
            return bic
        elif crit == "aicc":
            from statsmodels.tools.eval_measures import aicc
            return aicc(self.llf, self.nobs, k_params)
        elif crit == "hqic":
            from statsmodels.tools.eval_measures import hqic
            return hqic(self.llf, self.nobs, k_params)

    @cache_readonly
    def eigenvals(self):
        """
        Return eigenvalues sorted in decreasing order.
        """
        if self._wexog_singular_values is not None:
            eigvals = self._wexog_singular_values ** 2
        else:
            wx = self.model.wexog
            eigvals = np.linalg.linalg.eigvalsh(wx.T @ wx)
        return np.sort(eigvals)[::-1]

    @cache_readonly
    def condition_number(self):
        """
        Return condition number of exogenous matrix.

        Calculated as ratio of largest to smallest singular value of the
        exogenous variables. This value is the same as the square root of
        the ratio of the largest to smallest eigenvalue of the inner-product
        of the exogenous variables.
        """
        eigvals = self.eigenvals
        return np.sqrt(eigvals[0]/eigvals[-1])

    # TODO: make these properties reset bse
    def _HCCM(self, scale):
        H = np.dot(self.model.pinv_wexog,
                   scale[:, None] * self.model.pinv_wexog.T)
        return H

    def _abat_diagonal(self, a, b):
        # equivalent to np.diag(a @ b @ a.T)
        return np.einsum('ij,ik,kj->i', a, a, b)

    @cache_readonly
    def cov_HC0(self):
        """
        Heteroscedasticity robust covariance matrix. See HC0_se.
        """
        self.het_scale = self.wresid**2
        cov_HC0 = self._HCCM(self.het_scale)
        return cov_HC0

    @cache_readonly
    def cov_HC1(self):
        """
        Heteroscedasticity robust covariance matrix. See HC1_se.
        """
        self.het_scale = self.nobs/(self.df_resid)*(self.wresid**2)
        cov_HC1 = self._HCCM(self.het_scale)
        return cov_HC1

    @cache_readonly
    def cov_HC2(self):
        """
        Heteroscedasticity robust covariance matrix. See HC2_se.
        """
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = self.wresid**2/(1-h)
        cov_HC2 = self._HCCM(self.het_scale)
        return cov_HC2

    @cache_readonly
    def cov_HC3(self):
        """
        Heteroscedasticity robust covariance matrix. See HC3_se.
        """
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = (self.wresid / (1 - h))**2
        cov_HC3 = self._HCCM(self.het_scale)
        return cov_HC3

    @cache_readonly
    def HC0_se(self):
        """
        White's (1980) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i].

        When HC0_se or cov_HC0 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is just
        resid**2.
        """
        return np.sqrt(np.diag(self.cov_HC0))

    @cache_readonly
    def HC1_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as sqrt(diag(n/(n-p)*HC_0).

        When HC1_se or cov_HC1 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        n/(n-p)*resid**2.
        """
        return np.sqrt(np.diag(self.cov_HC1))

    @cache_readonly
    def HC2_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T

        When HC2_se or cov_HC2 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii).
        """
        return np.sqrt(np.diag(self.cov_HC2))

    @cache_readonly
    def HC3_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T.

        When HC3_se or cov_HC3 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii)^(2).
        """
        return np.sqrt(np.diag(self.cov_HC3))

    @cache_readonly
    def resid_pearson(self):
        """
        Residuals, normalized to have unit variance.

        Returns
        -------
        array_like
            The array `wresid` normalized by the sqrt of the scale to have
            unit variance.
        """

        if not hasattr(self, 'resid'):
            raise ValueError('Method requires residuals.')
        eps = np.finfo(self.wresid.dtype).eps
        if np.sqrt(self.scale) < 10 * eps * self.model.endog.mean():
            # do not divide if scale is zero close to numerical precision
            warnings.warn(
                "All residuals are 0, cannot compute normed residuals.",
                RuntimeWarning
            )
            return self.wresid
        else:
            return self.wresid / np.sqrt(self.scale)

    def _is_nested(self, restricted):
        """
        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current
            model. The result instance of the restricted model is required to
            have two attributes, residual sum of squares, `ssr`, residual
            degrees of freedom, `df_resid`.

        Returns
        -------
        nested : bool
            True if nested, otherwise false

        Notes
        -----
        A most nests another model if the regressors in the smaller
        model are spanned by the regressors in the larger model and
        the regressand is identical.
        """

        if self.model.nobs != restricted.model.nobs:
            return False

        full_rank = self.model.rank
        restricted_rank = restricted.model.rank
        if full_rank <= restricted_rank:
            return False

        restricted_exog = restricted.model.wexog
        full_wresid = self.wresid

        scores = restricted_exog * full_wresid[:, None]
        score_l2 = np.sqrt(np.mean(scores.mean(0) ** 2))
        # TODO: Could be improved, and may fail depending on scale of
        # regressors
        return np.allclose(score_l2, 0)

    def compare_lm_test(self, restricted, demean=True, use_lr=False):
        """
        Use Lagrange Multiplier test to test a set of linear restrictions.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.
        demean : bool
            Flag indicating whether the demean the scores based on the
            residuals from the restricted model.  If True, the covariance of
            the scores are used and the LM test is identical to the large
            sample version of the LR test.
        use_lr : bool
            A flag indicating whether to estimate the covariance of the model
            scores using the unrestricted model. Setting the to True improves
            the power of the test.

        Returns
        -------
        lm_value : float
            The test statistic which has a chi2 distributed.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The LM test examines whether the scores from the restricted model are
        0. If the null is true, and the restrictions are valid, then the
        parameters of the restricted model should be close to the minimum of
        the sum of squared errors, and so the scores should be close to zero,
        on average.
        """
        from numpy.linalg import inv

        import statsmodels.stats.sandwich_covariance as sw

        if not self._is_nested(restricted):
            raise ValueError("Restricted model is not nested by full model.")

        wresid = restricted.wresid
        wexog = self.model.wexog
        scores = wexog * wresid[:, None]

        n = self.nobs
        df_full = self.df_resid
        df_restr = restricted.df_resid
        df_diff = (df_restr - df_full)

        s = scores.mean(axis=0)
        if use_lr:
            scores = wexog * self.wresid[:, None]
            demean = False

        if demean:
            scores = scores - scores.mean(0)[None, :]
        # Form matters here.  If homoskedastics can be sigma^2 (X'X)^-1
        # If Heteroskedastic then the form below is fine
        # If HAC then need to use HAC
        # If Cluster, should use cluster

        cov_type = getattr(self, 'cov_type', 'nonrobust')
        if cov_type == 'nonrobust':
            sigma2 = np.mean(wresid**2)
            xpx = np.dot(wexog.T, wexog) / n
            s_inv = inv(sigma2 * xpx)
        elif cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
            s_inv = inv(np.dot(scores.T, scores) / n)
        elif cov_type == 'HAC':
            maxlags = self.cov_kwds['maxlags']
            s_inv = inv(sw.S_hac_simple(scores, maxlags) / n)
        elif cov_type == 'cluster':
            # cluster robust standard errors
            groups = self.cov_kwds['groups']
            # TODO: Might need demean option in S_crosssection by group?
            s_inv = inv(sw.S_crosssection(scores, groups))
        else:
            raise ValueError('Only nonrobust, HC, HAC and cluster are ' +
                             'currently connected')

        lm_value = n * (s @ s_inv @ s.T)
        p_value = stats.chi2.sf(lm_value, df_diff)
        return lm_value, p_value, df_diff

    def compare_f_test(self, restricted):
        """
        Use F test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.

        Returns
        -------
        f_value : float
            The test statistic which has an F distribution.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in
            df between models.

        Notes
        -----
        See mailing list discussion October 17,

        This test compares the residual sum of squares of the two
        models.  This is not a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results under
        the assumption of homoscedasticity and no autocorrelation
        (sphericity).
        """

        has_robust1 = getattr(self, 'cov_type', 'nonrobust') != 'nonrobust'
        has_robust2 = (getattr(restricted, 'cov_type', 'nonrobust') !=
                       'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('F test for comparison is likely invalid with ' +
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        ssr_full = self.ssr
        ssr_restr = restricted.ssr
        df_full = self.df_resid
        df_restr = restricted.df_resid

        df_diff = (df_restr - df_full)
        f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
        p_value = stats.f.sf(f_value, df_diff, df_full)
        return f_value, p_value, df_diff

    def compare_lr_test(self, restricted, large_sample=False):
        """
        Likelihood ratio test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current model.
            The result instance of the restricted model is required to have two
            attributes, residual sum of squares, `ssr`, residual degrees of
            freedom, `df_resid`.

        large_sample : bool
            Flag indicating whether to use a heteroskedasticity robust version
            of the LR test, which is a modified LM test.

        Returns
        -------
        lr_stat : float
            The likelihood ratio which is chisquare distributed with df_diff
            degrees of freedom.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The exact likelihood ratio is valid for homoskedastic data,
        and is defined as

        .. math:: D=-2\\log\\left(\\frac{\\mathcal{L}_{null}}
           {\\mathcal{L}_{alternative}}\\right)

        where :math:`\\mathcal{L}` is the likelihood of the
        model. With :math:`D` distributed as chisquare with df equal
        to difference in number of parameters or equivalently
        difference in residual degrees of freedom.

        The large sample version of the likelihood ratio is defined as

        .. math:: D=n s^{\\prime}S^{-1}s

        where :math:`s=n^{-1}\\sum_{i=1}^{n} s_{i}`

        .. math:: s_{i} = x_{i,alternative} \\epsilon_{i,null}

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.
        """
        # TODO: put into separate function, needs tests

        # See mailing list discussion October 17,

        if large_sample:
            return self.compare_lm_test(restricted, use_lr=True)

        has_robust1 = (getattr(self, 'cov_type', 'nonrobust') != 'nonrobust')
        has_robust2 = (
            getattr(restricted, 'cov_type', 'nonrobust') != 'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('Likelihood Ratio test is likely invalid with ' +
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        llf_full = self.llf
        llf_restr = restricted.llf
        df_full = self.df_resid
        df_restr = restricted.df_resid

        lrdf = (df_restr - df_full)
        lrstat = -2*(llf_restr - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lr_pvalue, lrdf

    def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwargs):
        """
        Create new results instance with robust covariance as default.

        Parameters
        ----------
        cov_type : str
            The type of robust sandwich estimator to use. See Notes below.
        use_t : bool
            If true, then the t distribution is used for inference.
            If false, then the normal distribution is used.
            If `use_t` is None, then an appropriate default is used, which is
            `True` if the cov_type is nonrobust, and `False` in all other
            cases.
        **kwargs
            Required or optional arguments for robust covariance calculation.
            See Notes below.

        Returns
        -------
        RegressionResults
            This method creates a new results instance with the
            requested robust covariance as the default covariance of
            the parameters.  Inferential statistics like p-values and
            hypothesis tests will be based on this covariance matrix.

        Notes
        -----
        The following covariance types and required or optional arguments are
        currently available:

        - 'fixed scale' uses a predefined scale

          ``scale``: float, optional
            Argument to set the scale. Default is 1.

        - 'HC0', 'HC1', 'HC2', 'HC3': heteroscedasticity robust covariance

          - no keyword arguments

        - 'HAC': heteroskedasticity-autocorrelation robust covariance

          ``maxlags`` :  integer, required
            number of lags to use

          ``kernel`` : {callable, str}, optional
            kernels currently available kernels are ['bartlett', 'uniform'],
            default is Bartlett

          ``use_correction``: bool, optional
            If true, use small sample correction

        - 'cluster': clustered covariance estimator

          ``groups`` : array_like[int], required :
            Integer-valued index of clusters or groups.

          ``use_correction``: bool, optional
            If True the sandwich covariance is calculated with a small
            sample correction.
            If False the sandwich covariance is calculated without
            small sample correction.

          ``df_correction``: bool, optional
            If True (default), then the degrees of freedom for the
            inferential statistics and hypothesis tests, such as
            pvalues, f_pvalue, conf_int, and t_test and f_test, are
            based on the number of groups minus one instead of the
            total number of observations minus the number of explanatory
            variables. `df_resid` of the results instance is also
            adjusted. When `use_t` is also True, then pvalues are
            computed using the Student's t distribution using the
            corrected values. These may differ substantially from
            p-values based on the normal is the number of groups is
            small.
            If False, then `df_resid` of the results instance is not
            adjusted.

        - 'hac-groupsum': Driscoll and Kraay, heteroscedasticity and
          autocorrelation robust covariance for panel data
          # TODO: more options needed here

          ``time`` : array_like, required
            index of time periods
          ``maxlags`` : integer, required
            number of lags to use
          ``kernel`` : {callable, str}, optional
            The available kernels are ['bartlett', 'uniform']. The default is
            Bartlett.
          ``use_correction`` : {False, 'hac', 'cluster'}, optional
            If False the the sandwich covariance is calculated without small
            sample correction. If `use_correction = 'cluster'` (default),
            then the same small sample correction as in the case of
            `covtype='cluster'` is used.
          ``df_correction`` : bool, optional
            The adjustment to df_resid, see cov_type 'cluster' above

        - 'hac-panel': heteroscedasticity and autocorrelation robust standard
          errors in panel data. The data needs to be sorted in this case, the
          time series for each panel unit or cluster need to be stacked. The
          membership to a time series of an individual or group can be either
          specified by group indicators or by increasing time periods. One of
          ``groups`` or ``time`` is required. # TODO: we need more options here

          ``groups`` : array_like[int]
            indicator for groups
          ``time`` : array_like[int]
            index of time periods
          ``maxlags`` : int, required
            number of lags to use
          ``kernel`` : {callable, str}, optional
            Available kernels are ['bartlett', 'uniform'], default
            is Bartlett
          ``use_correction`` : {False, 'hac', 'cluster'}, optional
            If False the sandwich covariance is calculated without
            small sample correction.
          ``df_correction`` : bool, optional
            Adjustment to df_resid, see cov_type 'cluster' above

        **Reminder**: ``use_correction`` in "hac-groupsum" and "hac-panel" is
        not bool, needs to be in {False, 'hac', 'cluster'}.

        .. todo:: Currently there is no check for extra or misspelled keywords,
             except in the case of cov_type `HCx`
        """
        from statsmodels.base.covtype import descriptions, normalize_cov_type
        import statsmodels.stats.sandwich_covariance as sw

        cov_type = normalize_cov_type(cov_type)

        if 'kernel' in kwargs:
            kwargs['weights_func'] = kwargs.pop('kernel')
        if 'weights_func' in kwargs and not callable(kwargs['weights_func']):
            kwargs['weights_func'] = sw.kernel_dict[kwargs['weights_func']]

        # TODO: make separate function that returns a robust cov plus info
        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        res.cov_type = cov_type
        # use_t might already be defined by the class, and already set
        if use_t is None:
            use_t = self.use_t
        res.cov_kwds = {'use_t': use_t}  # store for information
        res.use_t = use_t

        adjust_df = False
        if cov_type in ['cluster', 'hac-panel', 'hac-groupsum']:
            df_correction = kwargs.get('df_correction', None)
            # TODO: check also use_correction, do I need all combinations?
            if df_correction is not False:  # i.e. in [None, True]:
                # user did not explicitely set it to False
                adjust_df = True

        res.cov_kwds['adjust_df'] = adjust_df

        # verify and set kwargs, and calculate cov
        # TODO: this should be outsourced in a function so we can reuse it in
        #       other models
        # TODO: make it DRYer   repeated code for checking kwargs
        if cov_type in ['fixed scale', 'fixed_scale']:
            res.cov_kwds['description'] = descriptions['fixed_scale']

            res.cov_kwds['scale'] = scale = kwargs.get('scale', 1.)
            res.cov_params_default = scale * res.normalized_cov_params
        elif cov_type.upper() in ('HC0', 'HC1', 'HC2', 'HC3'):
            if kwargs:
                raise ValueError('heteroscedasticity robust covariance '
                                 'does not use keywords')
            res.cov_kwds['description'] = descriptions[cov_type.upper()]
            res.cov_params_default = getattr(self, 'cov_' + cov_type.upper())
        elif cov_type.lower() == 'hac':
            # TODO: check if required, default in cov_hac_simple
            maxlags = kwargs['maxlags']
            res.cov_kwds['maxlags'] = maxlags
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            use_correction = kwargs.get('use_correction', False)
            res.cov_kwds['use_correction'] = use_correction
            res.cov_kwds['description'] = descriptions['HAC'].format(
                maxlags=maxlags,
                correction=['without', 'with'][use_correction])

            res.cov_params_default = sw.cov_hac_simple(
                self, nlags=maxlags, weights_func=weights_func,
                use_correction=use_correction)
        elif cov_type.lower() == 'cluster':
            # cluster robust standard errors, one- or two-way
            groups = kwargs['groups']
            if not hasattr(groups, 'shape'):
                groups = np.asarray(groups).T

            if groups.ndim >= 2:
                groups = groups.squeeze()

            res.cov_kwds['groups'] = groups
            use_correction = kwargs.get('use_correction', True)
            res.cov_kwds['use_correction'] = use_correction
            if groups.ndim == 1:
                if adjust_df:
                    # need to find number of groups
                    # duplicate work
                    self.n_groups = n_groups = len(np.unique(groups))
                res.cov_params_default = sw.cov_cluster(
                    self, groups, use_correction=use_correction)

            elif groups.ndim == 2:
                if hasattr(groups, 'values'):
                    groups = groups.values

                if adjust_df:
                    # need to find number of groups
                    # duplicate work
                    n_groups0 = len(np.unique(groups[:, 0]))
                    n_groups1 = len(np.unique(groups[:, 1]))
                    self.n_groups = (n_groups0, n_groups1)
                    n_groups = min(n_groups0, n_groups1)  # use for adjust_df

                # Note: sw.cov_cluster_2groups has 3 returns
                res.cov_params_default = sw.cov_cluster_2groups(
                    self, groups, use_correction=use_correction)[0]
            else:
                raise ValueError('only two groups are supported')
            res.cov_kwds['description'] = descriptions['cluster']

        elif cov_type.lower() == 'hac-panel':
            # cluster robust standard errors
            res.cov_kwds['time'] = time = kwargs.get('time', None)
            res.cov_kwds['groups'] = groups = kwargs.get('groups', None)
            # TODO: nlags is currently required
            # nlags = kwargs.get('nlags', True)
            # res.cov_kwds['nlags'] = nlags
            # TODO: `nlags` or `maxlags`
            res.cov_kwds['maxlags'] = maxlags = kwargs['maxlags']
            use_correction = kwargs.get('use_correction', 'hac')
            res.cov_kwds['use_correction'] = use_correction
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            if groups is not None:
                groups = np.asarray(groups)
                tt = (np.nonzero(groups[:-1] != groups[1:])[0] + 1).tolist()
                nobs_ = len(groups)
            elif time is not None:
                time = np.asarray(time)
                # TODO: clumsy time index in cov_nw_panel
                tt = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
                nobs_ = len(time)
            else:
                raise ValueError('either time or groups needs to be given')
            groupidx = lzip([0] + tt, tt + [nobs_])
            self.n_groups = n_groups = len(groupidx)
            res.cov_params_default = sw.cov_nw_panel(
                self,
                maxlags,
                groupidx,
                weights_func=weights_func,
                use_correction=use_correction
            )
            res.cov_kwds['description'] = descriptions['HAC-Panel']

        elif cov_type.lower() == 'hac-groupsum':
            # Driscoll-Kraay standard errors
            res.cov_kwds['time'] = time = kwargs['time']
            # TODO: nlags is currently required
            # nlags = kwargs.get('nlags', True)
            # res.cov_kwds['nlags'] = nlags
            # TODO: `nlags` or `maxlags`
            res.cov_kwds['maxlags'] = maxlags = kwargs['maxlags']
            use_correction = kwargs.get('use_correction', 'cluster')
            res.cov_kwds['use_correction'] = use_correction
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            if adjust_df:
                # need to find number of groups
                tt = (np.nonzero(time[1:] < time[:-1])[0] + 1)
                self.n_groups = n_groups = len(tt) + 1
            res.cov_params_default = sw.cov_nw_groupsum(
                self, maxlags, time, weights_func=weights_func,
                use_correction=use_correction)
            res.cov_kwds['description'] = descriptions['HAC-Groupsum']
        else:
            raise ValueError('cov_type not recognized. See docstring for ' +
                             'available options and spelling')

        if adjust_df:
            # Note: df_resid is used for scale and others, add new attribute
            res.df_resid_inference = n_groups - 1

        return res

    @Appender(pred.get_prediction.__doc__)
    def get_prediction(self, exog=None, transform=True, weights=None,
                       row_labels=None, **kwargs):

        return pred.get_prediction(
            self, exog=exog, transform=transform, weights=weights,
            row_labels=row_labels, **kwargs)

    def summary(
            self,
            yname: str | None = None,
            xname: Sequence[str] | None = None,
            title: str | None = None,
            alpha: float = 0.05,
            slim: bool = False,
    ):
        """
        Summarize the Regression Results.

        Parameters
        ----------
        yname : str, optional
            Name of endogenous (response) variable. The Default is `y`.
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float, optional
            The significance level for the confidence intervals.
        slim : bool, optional
            Flag indicating to produce reduced set or diagnostic information.
            Default is False.

        Returns
        -------
        Summary
            Instance holding the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : A class that holds summary results.
        """
        from statsmodels.stats.stattools import (
            durbin_watson,
            jarque_bera,
            omni_normtest,
        )
        alpha = float_like(alpha, "alpha", optional=False)
        slim = bool_like(slim, "slim", optional=False, strict=True)

        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        eigvals = self.eigenvals
        condno = self.condition_number

        # TODO: Avoid adding attributes in non-__init__
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv, condno=condno,
                          mineigval=eigvals[-1])

        # TODO not used yet
        # diagn_left_header = ['Models stats']
        # diagn_right_header = ['Residual stats']

        # TODO: requiring list/iterable is a bit annoying
        #   need more control over formatting
        # TODO: default do not work if it's not identically spelled

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None),
                    ('Df Model:', None),
                    ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        rsquared_type = '' if self.k_constant else ' (uncentered)'
        top_right = [('R-squared' + rsquared_type + ':',
                      ["%#8.3f" % self.rsquared]),
                     ('Adj. R-squared' + rsquared_type + ':',
                      ["%#8.3f" % self.rsquared_adj]),
                     ('F-statistic:', ["%#8.4g" % self.fvalue]),
                     ('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     ('Log-Likelihood:', None),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        if slim:
            slimlist = ['Dep. Variable:', 'Model:', 'No. Observations:',
                        'Covariance Type:', 'R-squared:', 'Adj. R-squared:',
                        'F-statistic:', 'Prob (F-statistic):']
            diagn_left = diagn_right = []
            top_left = [elem for elem in top_left if elem[0] in slimlist]
            top_right = [elem for elem in top_right if elem[0] in slimlist]
            top_right = top_right + \
                [("", [])] * (len(top_left) - len(top_right))
        else:
            diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                          ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                          ('Skew:', ["%#6.3f" % skew]),
                          ('Kurtosis:', ["%#6.3f" % kurtosis])
                          ]

            diagn_right = [('Durbin-Watson:',
                            ["%#8.3f" % durbin_watson(self.wresid)]
                            ),
                           ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                           ('Prob(JB):', ["%#8.3g" % jbpv]),
                           ('Cond. No.', ["%#8.3g" % condno])
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
        if not slim:
            smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                                 yname=yname, xname=xname,
                                 title="")

        # add warnings/notes, added to text format only
        etext = []
        if not self.k_constant:
            etext.append(
                "R² is computed without centering (uncentered) since the "
                "model does not contain a constant."
            )
        if hasattr(self, 'cov_type'):
            etext.append(self.cov_kwds['description'])
        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            etext.append(wstr)
        if eigvals[-1] < 1e-10:
            wstr = "The smallest eigenvalue is %6.3g. This might indicate "
            wstr += "that there are\n"
            wstr += "strong multicollinearity problems or that the design "
            wstr += "matrix is singular."
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended?
            wstr = "The condition number is large, %6.3g. This might "
            wstr += "indicate that there are\n"
            wstr += "strong multicollinearity or other numerical "
            wstr += "problems."
            wstr = wstr % condno
            etext.append(wstr)

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Notes:")
            smry.add_extra_txt(etext)

        return smry

    def summary2(
            self,
            yname: str | None = None,
            xname: Sequence[str] | None = None,
            title: str | None = None,
            alpha: float = 0.05,
            float_format: str = "%.4f",
    ):
        """
        Experimental summary function to summarize the regression results.

        Parameters
        ----------
        yname : str
            The name of the dependent variable (optional).
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.
        float_format : str
            The format for floats in parameters summary.

        Returns
        -------
        Summary
            Instance holding the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary
            A class that holds summary results.
        """
        # Diagnostics
        from statsmodels.stats.stattools import (
            durbin_watson,
            jarque_bera,
            omni_normtest,
        )

        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)
        dw = durbin_watson(self.wresid)
        eigvals = self.eigenvals
        condno = self.condition_number
        diagnostic = dict([
            ('Omnibus:',  "%.3f" % omni),
            ('Prob(Omnibus):', "%.3f" % omnipv),
            ('Skew:', "%.3f" % skew),
            ('Kurtosis:', "%.3f" % kurtosis),
            ('Durbin-Watson:', "%.3f" % dw),
            ('Jarque-Bera (JB):', "%.3f" % jb),
            ('Prob(JB):', "%.3f" % jbpv),
            ('Condition No.:', "%.0f" % condno)
            ])

        # Summary
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format,
                      xname=xname, yname=yname, title=title)
        smry.add_dict(diagnostic)

        etext = []

        if not self.k_constant:
            etext.append(
                "R² is computed without centering (uncentered) since the \
                model does not contain a constant."
            )
        if hasattr(self, 'cov_type'):
            etext.append(self.cov_kwds['description'])
        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            etext.append(wstr)

        # Warnings
        if eigvals[-1] < 1e-10:
            warn = "The smallest eigenvalue is %6.3g. This might indicate that\
                there are strong multicollinearity problems or that the design\
                matrix is singular." % eigvals[-1]
            etext.append(warn)
        elif condno > 1000:
            warn = "The condition number is large, %6.3g. This might indicate\
                that there are strong multicollinearity or other numerical\
                problems." % condno
            etext.append(warn)

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Notes:")

        for line in etext:
            smry.add_text(line)

        return smry


class OLSResults(RegressionResults):
    """
    Results class for for an OLS model.

    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.

    See Also
    --------
    RegressionResults
        Results store for WLS and GLW models.

    Notes
    -----
    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:

    - get_influence
    - outlier_test
    - el_test
    - conf_int_el
    """

    def get_influence(self):
        """
        Calculate influence and outlier measures.

        Returns
        -------
        OLSInfluence
            The instance containing methods to calculate the main influence and
            outlier measures for the OLS regression.

        See Also
        --------
        statsmodels.stats.outliers_influence.OLSInfluence
            A class that exposes methods to examine observation influence.
        """
        from statsmodels.stats.outliers_influence import OLSInfluence
        return OLSInfluence(self)

    def outlier_test(self, method='bonf', alpha=.05, labels=None,
                     order=False, cutoff=None):
        """
        Test observations for outliers according to method.

        Parameters
        ----------
        method : str
            The method to use in the outlier test.  Must be one of:

            - `bonferroni` : one-step correction
            - `sidak` : one-step correction
            - `holm-sidak` :
            - `holm` :
            - `simes-hochberg` :
            - `hommel` :
            - `fdr_bh` : Benjamini/Hochberg
            - `fdr_by` : Benjamini/Yekutieli

            See `statsmodels.stats.multitest.multipletests` for details.
        alpha : float
            The familywise error rate (FWER).
        labels : None or array_like
            If `labels` is not None, then it will be used as index to the
            returned pandas DataFrame. See also Returns below.
        order : bool
            Whether or not to order the results by the absolute value of the
            studentized residuals. If labels are provided they will also be
            sorted.
        cutoff : None or float in [0, 1]
            If cutoff is not None, then the return only includes observations
            with multiple testing corrected p-values strictly below the cutoff.
            The returned array or dataframe can be empty if t.

        Returns
        -------
        array_like
            Returns either an ndarray or a DataFrame if labels is not None.
            Will attempt to get labels from model_results if available. The
            columns are the Studentized residuals, the unadjusted p-value,
            and the corrected p-value according to method.

        Notes
        -----
        The unadjusted p-value is stats.t.sf(abs(resid), df) where
        df = df_resid - 1.
        """
        from statsmodels.stats.outliers_influence import outlier_test
        return outlier_test(self, method, alpha, labels=labels,
                            order=order, cutoff=cutoff)

    def el_test(self, b0_vals, param_nums, return_weights=0, ret_params=0,
                method='nm', stochastic_exog=1):
        """
        Test single or joint hypotheses using Empirical Likelihood.

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value of the parameter to be tested.
        param_nums : 1darray
            The parameter number to be tested.
        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals. The default is False.
        ret_params : bool
            If true, returns the parameter vector that maximizes the likelihood
            ratio at b0_vals.  Also returns the weights.  The default is False.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors. The default is True.

        Returns
        -------
        tuple
            The p-value and -2 times the log-likelihood ratio for the
            hypothesized values.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.stackloss.load()
        >>> endog = data.endog
        >>> exog = sm.add_constant(data.exog)
        >>> model = sm.OLS(endog, exog)
        >>> fitted = model.fit()
        >>> fitted.params
        >>> array([-39.91967442,   0.7156402 ,   1.29528612,  -0.15212252])
        >>> fitted.rsquared
        >>> 0.91357690446068196
        >>> # Test that the slope on the first variable is 0
        >>> fitted.el_test([0], [1])
        >>> (27.248146353888796, 1.7894660442330235e-07)
        """
        params = np.copy(self.params)
        opt_fun_inst = _ELRegOpts()  # to store weights
        if len(param_nums) == len(params):
            llr = opt_fun_inst._opt_nuis_regress(
                [],
                param_nums=param_nums,
                endog=self.model.endog,
                exog=self.model.exog,
                nobs=self.model.nobs,
                nvar=self.model.exog.shape[1],
                params=params,
                b0_vals=b0_vals,
                stochastic_exog=stochastic_exog)
            pval = 1 - stats.chi2.cdf(llr, len(param_nums))
            if return_weights:
                return llr, pval, opt_fun_inst.new_weights
            else:
                return llr, pval
        x0 = np.delete(params, param_nums)
        args = (param_nums, self.model.endog, self.model.exog,
                self.model.nobs, self.model.exog.shape[1], params,
                b0_vals, stochastic_exog)
        if method == 'nm':
            llr = optimize.fmin(opt_fun_inst._opt_nuis_regress, x0,
                                maxfun=10000, maxiter=10000, full_output=1,
                                disp=0, args=args)[1]
        if method == 'powell':
            llr = optimize.fmin_powell(opt_fun_inst._opt_nuis_regress, x0,
                                       full_output=1, disp=0,
                                       args=args)[1]

        pval = 1 - stats.chi2.cdf(llr, len(param_nums))
        if ret_params:
            return llr, pval, opt_fun_inst.new_weights, opt_fun_inst.new_params
        elif return_weights:
            return llr, pval, opt_fun_inst.new_weights
        else:
            return llr, pval

    def conf_int_el(self, param_num, sig=.05, upper_bound=None,
                    lower_bound=None, method='nm', stochastic_exog=True):
        """
        Compute the confidence interval using Empirical Likelihood.

        Parameters
        ----------
        param_num : float
            The parameter for which the confidence interval is desired.
        sig : float
            The significance level.  Default is 0.05.
        upper_bound : float
            The maximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.
        lower_bound : float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  The default is True.

        Returns
        -------
        lowerl : float
            The lower bound of the confidence interval.
        upperl : float
            The upper bound of the confidence interval.

        See Also
        --------
        el_test : Test parameters using Empirical Likelihood.

        Notes
        -----
        This function uses brentq to find the value of beta where
        test_beta([beta], param_num)[1] is equal to the critical value.

        The function returns the results of each iteration of brentq at each
        value of beta.

        The current function value of the last printed optimization should be
        the critical value at the desired significance level. For alpha=.05,
        the value is 3.841459.

        To ensure optimization terminated successfully, it is suggested to do
        el_test([lower_limit], [param_num]).

        If the optimization does not terminate successfully, consider switching
        optimization algorithms.

        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number between 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.
        """
        r0 = stats.chi2.ppf(1 - sig, 1)
        if upper_bound is None:
            upper_bound = self.conf_int(.01)[param_num][1]
        if lower_bound is None:
            lower_bound = self.conf_int(.01)[param_num][0]

        def f(b0):
            return self.el_test(np.array([b0]), np.array([param_num]),
                                method=method,
                                stochastic_exog=stochastic_exog)[0] - r0

        lowerl = optimize.brenth(f, lower_bound,
                                 self.params[param_num])
        upperl = optimize.brenth(f, self.params[param_num],
                                 upper_bound)
        #  ^ Seems to be faster than brentq in most cases
        return (lowerl, upperl)


class RegressionResultsWrapper(wrap.ResultsWrapper):

    _attrs = {
        'chisq': 'columns',
        'sresid': 'rows',
        'weights': 'rows',
        'wresid': 'rows',
        'bcov_unscaled': 'cov',
        'bcov_scaled': 'cov',
        'HC0_se': 'columns',
        'HC1_se': 'columns',
        'HC2_se': 'columns',
        'HC3_se': 'columns',
        'norm_resid': 'rows',
    }

    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._attrs,
                                   _attrs)

    _methods = {}

    _wrap_methods = wrap.union_dicts(
                        base.LikelihoodResultsWrapper._wrap_methods,
                        _methods)


wrap.populate_wrapper(RegressionResultsWrapper,
                      RegressionResults)
