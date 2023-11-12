# -*- coding: utf-8 -*-

u"""
Beta regression for modeling rates and proportions.

References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.

Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""

import numpy as np
from scipy.special import gammaln as lgamma
import patsy

import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
    GenericLikelihoodModel, GenericLikelihoodModelResults, _LLRMixin)
from statsmodels.genmod import families


_init_example = """

    Beta regression with default of logit-link for exog and log-link
    for precision.

    >>> mod = BetaModel(endog, exog)
    >>> rslt = mod.fit()
    >>> print(rslt.summary())

    We can also specify a formula and a specific structure and use the
    identity-link for precision.

    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = BetaModel.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                              dat, exog_precision=Z,
    ...                              link_precision=identity())

    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:

    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> formula = 'methylation ~ disease + age + gender + coverage'
    >>> mod = BetaModel.from_formula(formula, df, Z)
    >>> rslt = mod.fit()

"""


class BetaModel(GenericLikelihoodModel):
    __doc__ = """Beta Regression.

    The Model is parameterized by mean and precision. Both can depend on
    explanatory variables through link functions.

    Parameters
    ----------
    endog : array_like
        1d array of endogenous response variable.
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user (models specified using a formula
        include an intercept by default). See `statsmodels.tools.add_constant`.
    exog_precision : array_like
        2d array of variables for the precision.
    link : link
        Any link in sm.families.links for mean, should have range in
        interval [0, 1]. Default is logit-link.
    link_precision : link
        Any link in sm.families.links for precision, should have
        range in positive line. Default is log-link.
    **kwds : extra keywords
        Keyword options that will be handled by super classes.
        Not all general keywords will be supported in this class.

    Notes
    -----
    Status: experimental, new in 0.13.
    Core results are verified, but api can change and some extra results
    specific to Beta regression are missing.

    Examples
    --------
    {example}

    See Also
    --------
    :ref:`links`

    """.format(example=_init_example)

    def __init__(self, endog, exog, exog_precision=None,
                 link=families.links.Logit(),
                 link_precision=families.links.Log(), **kwds):

        etmp = np.array(endog)
        assert np.all((0 < etmp) & (etmp < 1))
        if exog_precision is None:
            extra_names = ['precision']
            exog_precision = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in
                           (exog_precision.columns
                            if hasattr(exog_precision, 'columns')
                            else range(1, exog_precision.shape[1] + 1))]

        kwds['extra_params_names'] = extra_names

        super(BetaModel, self).__init__(endog, exog,
                                        exog_precision=exog_precision,
                                        **kwds)
        self.link = link
        self.link_precision = link_precision
        # not needed, handled by super:
        # self.exog_precision = exog_precision
        # inherited df do not account for precision params
        self.nobs = self.endog.shape[0]
        self.k_extra = 1
        self.df_model = self.nparams - 2
        self.df_resid = self.nobs - self.nparams
        assert len(self.exog_precision) == len(self.endog)
        self.hess_type = "oim"
        if 'exog_precision' not in self._init_keys:
            self._init_keys.extend(['exog_precision'])
        self._init_keys.extend(['link', 'link_precision'])
        self._null_drop_keys = ['exog_precision']
        del kwds['extra_params_names']
        self._check_kwargs(kwds)
        self.results_class = BetaResults
        self.results_class_wrapper = BetaResultsWrapper

    @classmethod
    def from_formula(cls, formula, data, exog_precision_formula=None,
                     *args, **kwargs):
        if exog_precision_formula is not None:
            if 'subset' in kwargs:
                d = data.ix[kwargs['subset']]
                Z = patsy.dmatrix(exog_precision_formula, d)
            else:
                Z = patsy.dmatrix(exog_precision_formula, data)
            kwargs['exog_precision'] = Z

        return super(BetaModel, cls).from_formula(formula, data, *args,
                                                  **kwargs)

    def _get_exogs(self):
        return (self.exog, self.exog_precision)

    def predict(self, params, exog=None, exog_precision=None, which="mean"):
        """Predict values for mean or precision

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for precision parameter.
        which : str

            - "mean" : mean, conditional expectation E(endog | exog)
            - "precision" : predicted precision
            - "linear" : linear predictor for the mean function
            - "linear-precision" : linear predictor for the precision parameter

        Returns
        -------
        ndarray, predicted values
        """
        # compatibility with old names and misspelling
        if which == "linpred":
            which = "linear"
        if which in ["linpred_precision", "linear_precision"]:
            which = "linear-precision"

        k_mean = self.exog.shape[1]
        if which in ["mean",  "linear"]:
            if exog is None:
                exog = self.exog
            params_mean = params[:k_mean]
            # Zparams = params[k_mean:]
            linpred = np.dot(exog, params_mean)
            if which == "mean":
                mu = self.link.inverse(linpred)
                res = mu
            else:
                res = linpred

        elif which in ["precision", "linear-precision"]:
            if exog_precision is None:
                exog_precision = self.exog_precision
            params_prec = params[k_mean:]
            linpred_prec = np.dot(exog_precision, params_prec)

            if which == "precision":
                phi = self.link_precision.inverse(linpred_prec)
                res = phi
            else:
                res = linpred_prec

        elif which == "var":
            res = self._predict_var(
                params,
                exog=exog,
                exog_precision=exog_precision
                )

        else:
            raise ValueError('which = %s is not available' % which)

        return res

    def _predict_precision(self, params, exog_precision=None):
        """Predict values for precision function for given exog_precision.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted precision.
        """
        if exog_precision is None:
            exog_precision = self.exog_precision

        k_mean = self.exog.shape[1]
        params_precision = params[k_mean:]
        linpred_prec = np.dot(exog_precision, params_precision)
        phi = self.link_precision.inverse(linpred_prec)

        return phi

    def _predict_var(self, params, exog=None, exog_precision=None):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted conditional variance.
        """
        mean = self.predict(params, exog=exog)
        precision = self._predict_precision(params,
                                            exog_precision=exog_precision)

        var_endog = mean * (1 - mean) / (1 + precision)
        return var_endog

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of the Beta regressionmodel.

        Parameters
        ----------
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        return self._llobs(self.endog, self.exog, self.exog_precision, params)

    def _llobs(self, endog, exog, exog_precision, params):
        """
        Loglikelihood for observations with data arguments.

        Parameters
        ----------
        endog : ndarray
            1d array of endogenous variable.
        exog : ndarray
            2d array of explanatory variables.
        exog_precision : ndarray
            2d array of explanatory variables for precision.
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        y, X, Z = endog, exog, exog_precision
        nz = Z.shape[1]

        params_mean = params[:-nz]
        params_prec = params[-nz:]
        linpred = np.dot(X, params_mean)
        linpred_prec = np.dot(Z, params_prec)

        mu = self.link.inverse(linpred)
        phi = self.link_precision.inverse(linpred_prec)

        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)

        ll = (lgamma(phi) - lgamma(alpha)
              - lgamma(beta)
              + (mu * phi - 1) * np.log(y)
              + (((1 - mu) * phi) - 1) * np.log(1 - y))

        return ll

    def score(self, params):
        """
        Returns the score vector of the log-likelihood.

        http://www.tandfonline.com/doi/pdf/10.1080/00949650903389993

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score : ndarray
            First derivative of loglikelihood function.
        """
        sf1, sf2 = self.score_factor(params)

        d1 = np.dot(sf1, self.exog)
        d2 = np.dot(sf2, self.exog_precision)
        return np.concatenate((d1, d2))

    def _score_check(self, params):
        """Inherited score with finite differences

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score based on numerical derivatives
        """
        return super(BetaModel, self).score(params)

    def score_factor(self, params, endog=None):
        """Derivative of loglikelihood function w.r.t. linear predictors.

        This needs to be multiplied with the exog to obtain the score_obs.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.

        Notes
        -----
        The score_obs can be obtained from score_factor ``sf`` using

            - d1 = sf[:, :1] * exog
            - d2 = sf[:, 1:2] * exog_precision

        """
        from scipy import special
        digamma = special.psi

        y = self.endog if endog is None else endog
        X, Z = self.exog, self.exog_precision
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]

        # NO LINKS
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))

        eps_lb = 1e-200  # lower bound for evaluating digamma, avoids -inf
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)

        ystar = np.log(y / (1. - y))
        dig_beta = digamma(beta)
        mustar = digamma(alpha) - dig_beta
        yt = np.log(1 - y)
        mut = dig_beta - digamma(phi)

        t = 1. / self.link.deriv(mu)
        h = 1. / self.link_precision.deriv(phi)
        #
        sf1 = phi * t * (ystar - mustar)
        sf2 = h * (mu * (ystar - mustar) + yt - mut)

        return (sf1, sf2)

    def score_hessian_factor(self, params, return_hessian=False,
                             observed=True):
        """Derivatives of loglikelihood function w.r.t. linear predictors.

        This calculates score and hessian factors at the same time, because
        there is a large overlap in calculations.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        return_hessian : bool
            If False, then only score_factors are returned
            If True, the both score and hessian factors are returned
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.
        (-jbb, -jbg, -jgg) : tuple
            A tuple with 3 hessian factors, corresponding to the upper
            triangle of the Hessian matrix.
            TODO: check why there are minus
        """
        from scipy import special
        digamma = special.psi

        y, X, Z = self.endog, self.exog, self.exog_precision
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]

        # NO LINKS
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))

        # We need to prevent mu = 0 and (1-mu) = 0 in digamma call
        eps_lb = 1e-200  # lower bound for evaluating digamma, avoids -inf
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)

        ystar = np.log(y / (1. - y))
        dig_beta = digamma(beta)
        mustar = digamma(alpha) - dig_beta
        yt = np.log(1 - y)
        mut = dig_beta - digamma(phi)

        t = 1. / self.link.deriv(mu)
        h = 1. / self.link_precision.deriv(phi)

        ymu_star = (ystar - mustar)
        sf1 = phi * t * ymu_star
        sf2 = h * (mu * ymu_star + yt - mut)

        if return_hessian:
            trigamma = lambda x: special.polygamma(1, x)  # noqa
            trig_beta = trigamma(beta)
            var_star = trigamma(alpha) + trig_beta
            var_t = trig_beta - trigamma(phi)

            c = - trig_beta
            s = self.link.deriv2(mu)
            q = self.link_precision.deriv2(phi)

            jbb = (phi * t) * var_star
            if observed:
                jbb += s * t**2 * ymu_star

            jbb *= t * phi

            jbg = phi * t * h * (mu * var_star + c)
            if observed:
                jbg -= ymu_star * t * h

            jgg = h**2 * (mu**2 * var_star + 2 * mu * c + var_t)
            if observed:
                jgg += (mu * ymu_star + yt - mut) * q * h**3    # **3 ?

            return (sf1, sf2), (-jbb, -jbg, -jgg)
        else:
            return (sf1, sf2)

    def score_obs(self, params):
        """
        Score, first derivative of the loglikelihood for each observation.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """
        sf1, sf2 = self.score_factor(params)

        # elementwise product for each row (observation)
        d1 = sf1[:, None] * self.exog
        d2 = sf2[:, None] * self.exog_precision
        return np.column_stack((d1, d2))

    def hessian(self, params, observed=None):
        """Hessian, second derivative of loglikelihood function

        Parameters
        ----------
        params : ndarray
            Parameter at which Hessian is evaluated.
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        hessian : ndarray
            Hessian, i.e. observed information, or expected information matrix.
        """
        if self.hess_type == "eim":
            observed = False
        else:
            observed = True
        _, hf = self.score_hessian_factor(params, return_hessian=True,
                                          observed=observed)

        hf11, hf12, hf22 = hf

        # elementwise product for each row (observation)
        d11 = (self.exog.T * hf11).dot(self.exog)
        d12 = (self.exog.T * hf12).dot(self.exog_precision)
        d22 = (self.exog_precision.T * hf22).dot(self.exog_precision)
        return np.block([[d11, d12], [d12.T, d22]])

    def hessian_factor(self, params, observed=True):
        """Derivatives of loglikelihood function w.r.t. linear predictors.
        """
        _, hf = self.score_hessian_factor(params, return_hessian=True,
                                          observed=observed)
        return hf

    def _start_params(self, niter=2, return_intermediate=False):
        """find starting values

        Parameters
        ----------
        niter : int
            Number of iterations of WLS approximation
        return_intermediate : bool
            If False (default), then only the preliminary parameter estimate
            will be returned.
            If True, then also the two results instances of the WLS estimate
            for mean parameters and for the precision parameters will be
            returned.

        Returns
        -------
        sp : ndarray
            start parameters for the optimization
        res_m2 : results instance (optional)
            Results instance for the WLS regression of the mean function.
        res_p2 : results instance (optional)
            Results instance for the WLS regression of the precision function.

        Notes
        -----
        This calculates a few iteration of weighted least squares. This is not
        a full scoring algorithm.
        """
        # WLS of the mean equation uses the implied weights (inverse variance),
        # WLS for the precision equations uses weights that only take
        # account of the link transformation of the precision endog.
        from statsmodels.regression.linear_model import OLS, WLS
        res_m = OLS(self.link(self.endog), self.exog).fit()
        fitted = self.link.inverse(res_m.fittedvalues)
        resid = self.endog - fitted

        prec_i = fitted * (1 - fitted) / np.maximum(np.abs(resid), 1e-2)**2 - 1
        res_p = OLS(self.link_precision(prec_i), self.exog_precision).fit()
        prec_fitted = self.link_precision.inverse(res_p.fittedvalues)
        # sp = np.concatenate((res_m.params, res_p.params))

        for _ in range(niter):
            y_var_inv = (1 + prec_fitted) / (fitted * (1 - fitted))
            # y_var = fitted * (1 - fitted) / (1 + prec_fitted)

            ylink_var_inv = y_var_inv / self.link.deriv(fitted)**2
            res_m2 = WLS(self.link(self.endog), self.exog,
                         weights=ylink_var_inv).fit()
            fitted = self.link.inverse(res_m2.fittedvalues)
            resid2 = self.endog - fitted

            prec_i2 = (fitted * (1 - fitted) /
                       np.maximum(np.abs(resid2), 1e-2)**2 - 1)
            w_p = 1. / self.link_precision.deriv(prec_fitted)**2
            res_p2 = WLS(self.link_precision(prec_i2), self.exog_precision,
                         weights=w_p).fit()
            prec_fitted = self.link_precision.inverse(res_p2.fittedvalues)
            sp2 = np.concatenate((res_m2.params, res_p2.params))

        if return_intermediate:
            return sp2, res_m2, res_p2

        return sp2

    def fit(self, start_params=None, maxiter=1000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model by maximum likelihood.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        kwds :
            Keyword arguments for the optimizer.

        Returns
        -------
        BetaResults instance.
        """

        if start_params is None:
            start_params = self._start_params()
#           # http://www.ime.usp.br/~sferrari/beta.pdf suggests starting phi
#           # on page 8

        if "cov_type" in kwds:
            # this is a workaround because we cannot tell super to use eim
            if kwds["cov_type"].lower() == "eim":
                self.hess_type = "eim"
                del kwds["cov_type"]
        else:
            self.hess_type = "oim"

        res = super(BetaModel, self).fit(start_params=start_params,
                                         maxiter=maxiter, method=method,
                                         disp=disp, **kwds)
        if not isinstance(res, BetaResultsWrapper):
            # currently GenericLikelihoodModel doe not add wrapper
            res = BetaResultsWrapper(res)
        return res

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        not verified yet

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        link = self.link
        lin_pred = self.predict(params, which="linear")
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return np.column_stack((dmat, np.zeros(self.exog_precision.shape)))

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)

        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        # deriv is 2d vector
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2] * self.exog_precision

        return np.column_stack((d1, d2))

    def get_distribution_params(self, params, exog=None, exog_precision=None):
        """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
        mean = self.predict(params, exog=exog)
        precision = self.predict(params, exog_precision=exog_precision,
                                 which="precision")
        return precision * mean, precision * (1 - mean)

    def get_distribution(self, params, exog=None, exog_precision=None):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.

        Returns
        -------
        Instance of a scipy frozen distribution based on estimated
        parameters.

        See Also
        --------
        predict

        Notes
        -----
        This function delegates to the predict method to handle exog and
        exog_precision, which in turn makes any required transformations.

        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        from scipy import stats
        args = self.get_distribution_params(params, exog=exog,
                                            exog_precision=exog_precision)
        distr = stats.beta(*args)
        return distr


class BetaResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Beta regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    # GenericLikeihoodmodel doesn't define fittedvalues, residuals and similar
    @cache_readonly
    def fittedvalues(self):
        """In-sample predicted mean, conditional expectation."""
        return self.model.predict(self.params)

    @cache_readonly
    def fitted_precision(self):
        """In-sample predicted precision"""
        return self.model.predict(self.params, which="precision")

    @cache_readonly
    def resid(self):
        """Response residual"""
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_pearson(self):
        """Pearson standardize residual"""
        std = np.sqrt(self.model.predict(self.params, which="var"))
        return self.resid / std

    @cache_readonly
    def prsquared(self):
        """Cox-Snell Likelihood-Ratio pseudo-R-squared.

        1 - exp((llnull - .llf) * (2 / nobs))
        """
        return self.pseudo_rsquared(kind="lr")

    def get_distribution_params(self, exog=None, exog_precision=None,
                                transform=True):
        """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
        mean = self.predict(exog=exog, transform=transform)
        precision = self.predict(exog_precision=exog_precision,
                                 which="precision", transform=transform)
        return precision * mean, precision * (1 - mean)

    def get_distribution(self, exog=None, exog_precision=None, transform=True):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        Instance of a scipy frozen distribution based on estimated
        parameters.

        See Also
        --------
        predict

        Notes
        -----
        This function delegates to the predict method to handle exog and
        exog_precision, which in turn makes any required transformations.

        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        from scipy import stats
        args = self.get_distribution_params(exog=exog,
                                            exog_precision=exog_precision,
                                            transform=transform)
        args = (np.asarray(arg) for arg in args)
        distr = stats.beta(*args)
        return distr

    def get_influence(self):
        """
        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        Support for mutli-link and multi-exog models is still experimental
        in MLEInfluence. Interface and some definitions might still change.

        Note: Difference to R betareg: Betareg has the same general leverage
        as this model. However, they use a linear approximation hat matrix
        to scale and studentize influence and residual statistics.
        MLEInfluence uses the generalized leverage as hat_matrix_diag.
        Additionally, MLEInfluence uses pearson residuals for residual
        analusis.

        References
        ----------
        todo

        """
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def bootstrap(self, *args, **kwargs):
        raise NotImplementedError


class BetaResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(BetaResultsWrapper,
                      BetaResults)
