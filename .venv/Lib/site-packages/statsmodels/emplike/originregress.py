"""
This module implements empirical likelihood regression that is forced through
the origin.

This is different than regression not forced through the origin because the
maximum empirical likelihood estimate is calculated with a vector of ones in
the exogenous matrix but restricts the intercept parameter to be 0.  This
results in significantly more narrow confidence intervals and different
parameter estimates.

For notes on regression not forced through the origin, see empirical likelihood
methods in the OLSResults class.

General References
------------------
Owen, A.B. (2001). Empirical Likelihood.  Chapman and Hall. p. 82.

"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2

from statsmodels.regression.linear_model import OLS, RegressionResults
# When descriptive merged, this will be changed
from statsmodels.tools.tools import add_constant


class ELOriginRegress:
    """
    Empirical Likelihood inference and estimation for linear regression
    through the origin.

    Parameters
    ----------
    endog: nx1 array
        Array of response variables.

    exog: nxk array
        Array of exogenous variables.  Assumes no array of ones

    Attributes
    ----------
    endog : nx1 array
        Array of response variables

    exog : nxk array
        Array of exogenous variables.  Assumes no array of ones.

    nobs : float
        Number of observations.

    nvar : float
        Number of exogenous regressors.
    """
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog
        self.nobs = self.exog.shape[0]
        try:
            self.nvar = float(exog.shape[1])
        except IndexError:
            self.nvar = 1.

    def fit(self):
        """
        Fits the model and provides regression results.

        Returns
        -------
        Results : class
            Empirical likelihood regression class.
        """
        exog_with = add_constant(self.exog, prepend=True)
        restricted_model = OLS(self.endog, exog_with)
        restricted_fit = restricted_model.fit()
        restricted_el = restricted_fit.el_test(
        np.array([0]), np.array([0]), ret_params=1)
        params = np.squeeze(restricted_el[3])
        beta_hat_llr = restricted_el[0]
        llf = np.sum(np.log(restricted_el[2]))
        return OriginResults(restricted_model, params, beta_hat_llr, llf)

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(add_constant(exog, prepend=True), params)


class OriginResults(RegressionResults):
    """
    A Results class for empirical likelihood regression through the origin.

    Parameters
    ----------
    model : class
        An OLS model with an intercept.

    params : 1darray
        Fitted parameters.

    est_llr : float
        The log likelihood ratio of the model with the intercept restricted to
        0 at the maximum likelihood estimates of the parameters.
        llr_restricted/llr_unrestricted

    llf_el : float
        The log likelihood of the fitted model with the intercept restricted to 0.

    Attributes
    ----------
    model : class
        An OLS model with an intercept.

    params : 1darray
        Fitted parameter.

    llr : float
        The log likelihood ratio of the maximum empirical likelihood estimate.

    llf_el : float
        The log likelihood of the fitted model with the intercept restricted to 0.

    Notes
    -----
    IMPORTANT.  Since EL estimation does not drop the intercept parameter but
    instead estimates the slope parameters conditional on the slope parameter
    being 0, the first element for params will be the intercept, which is
    restricted to 0.

    IMPORTANT.  This class inherits from RegressionResults but inference is
    conducted via empirical likelihood.  Therefore, any methods that
    require an estimate of the covariance matrix will not function.  Instead
    use el_test and conf_int_el to conduct inference.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.bc.load()
    >>> model = sm.emplike.ELOriginRegress(data.endog, data.exog)
    >>> fitted = model.fit()
    >>> fitted.params #  0 is the intercept term.
    array([ 0.        ,  0.00351813])

    >>> fitted.el_test(np.array([.0034]), np.array([1]))
    (3.6696503297979302, 0.055411808127497755)
    >>> fitted.conf_int_el(1)
    (0.0033971871114706867, 0.0036373150174892847)

    # No covariance matrix so normal inference is not valid
    >>> fitted.conf_int()
    TypeError: unsupported operand type(s) for *: 'instancemethod' and 'float'
    """
    def __init__(self, model, params, est_llr, llf_el):
        self.model = model
        self.params = np.squeeze(params)
        self.llr = est_llr
        self.llf_el = llf_el
    def el_test(self, b0_vals, param_nums, method='nm',
                            stochastic_exog=1, return_weights=0):
        """
        Returns the llr and p-value for a hypothesized parameter value
        for a regression that goes through the origin.

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value to be tested.

        param_num : 1darray
            Which parameters to test.  Note this uses python
            indexing but the '0' parameter refers to the intercept term,
            which is assumed 0.  Therefore, param_num should be > 0.

        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals.  Default is False.

        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            Default is 'nm'.

        stochastic_exog : bool
            When TRUE, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  Default is TRUE.

        Returns
        -------
        res : tuple
            pvalue and likelihood ratio.
        """
        b0_vals = np.hstack((0, b0_vals))
        param_nums = np.hstack((0, param_nums))
        test_res = self.model.fit().el_test(b0_vals, param_nums, method=method,
                                  stochastic_exog=stochastic_exog,
                                  return_weights=return_weights)
        llr_test = test_res[0]
        llr_res = llr_test - self.llr
        pval = chi2.sf(llr_res, self.model.exog.shape[1] - 1)
        if return_weights:
            return llr_res, pval, test_res[2]
        else:
            return llr_res, pval

    def conf_int_el(self, param_num, upper_bound=None,
                       lower_bound=None, sig=.05, method='nm',
                       stochastic_exog=1):
        """
        Returns the confidence interval for a regression parameter when the
        regression is forced through the origin.

        Parameters
        ----------
        param_num : int
            The parameter number to be tested.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.

        upper_bound : float
            The maximum value the upper confidence limit can be.  The
            closer this is to the confidence limit, the quicker the
            computation.  Default is .00001 confidence limit under normality.

        lower_bound : float
            The minimum value the lower confidence limit can be.
            Default is .00001 confidence limit under normality.

        sig : float, optional
            The significance level.  Default .05.

        method : str, optional
             Algorithm to optimize of nuisance params.  Can be 'nm' or
            'powell'.  Default is 'nm'.

        Returns
        -------
        ci: tuple
            The confidence interval for the parameter 'param_num'.
        """
        r0 = chi2.ppf(1 - sig, 1)
        param_num = np.array([param_num])
        if upper_bound is None:
            upper_bound = (np.squeeze(self.model.fit().
                                      conf_int(.0001)[param_num])[1])
        if lower_bound is None:
            lower_bound = (np.squeeze(self.model.fit().conf_int(.00001)
                                      [param_num])[0])
        f = lambda b0:  self.el_test(np.array([b0]), param_num,
                                     method=method,
                                 stochastic_exog=stochastic_exog)[0] - r0
        _param = np.squeeze(self.params[param_num])
        lowerl = optimize.brentq(f, np.squeeze(lower_bound), _param)
        upperl = optimize.brentq(f, _param, np.squeeze(upper_bound))
        return (lowerl, upperl)
