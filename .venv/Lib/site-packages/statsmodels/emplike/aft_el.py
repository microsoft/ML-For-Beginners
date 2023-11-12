"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Stute, W. (1993). "Consistent Estimation Under Random Censorship when
Covariables are Present." Journal of Multivariate Analysis.
Vol. 45. Iss. 1. 89-103

EL and AFT References
---------------------

Zhou, Kim And Bathke. "Empirical Likelihood Analysis for the Heteroskedastic
Accelerated Failure Time Model." Manuscript:
URL: www.ms.uky.edu/~mai/research/CasewiseEL20080724.pdf

Zhou, M. (2005). Empirical Likelihood Ratio with Arbitrarily Censored/
Truncated Data by EM Algorithm.  Journal of Computational and Graphical
Statistics. 14:3, 643-656.


"""
import warnings

import numpy as np
#from elregress import ElReg
from scipy import optimize
from scipy.stats import chi2

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from .descriptive import _OptFuncts


class OptAFT(_OptFuncts):
    """
    Provides optimization functions used in estimating and conducting
    inference in an AFT model.

    Methods
    ------

    _opt_wtd_nuis_regress:
        Function optimized over nuisance parameters to compute
        the profile likelihood

    _EM_test:
        Uses the modified Em algorithm of Zhou 2005 to maximize the
        likelihood of a parameter vector.
    """
    def __init__(self):
        pass

    def _opt_wtd_nuis_regress(self, test_vals):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------

        params: 1d array
            The regression coefficients of the model.  This includes the
            nuisance and parameters of interests.

        Returns
        -------
        llr : float
            -2 times the log likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
        test_params = test_vals.reshape(self.model.nvar, 1)
        est_vect = self.model.uncens_exog * (self.model.uncens_endog -
                                            np.dot(self.model.uncens_exog,
                                                         test_params))
        eta_star = self._modif_newton(np.zeros(self.model.nvar), est_vect,
                                         self.model._fit_weights)
        denom = np.sum(self.model._fit_weights) + np.dot(eta_star, est_vect.T)
        self.new_weights = self.model._fit_weights / denom
        return -1 * np.sum(np.log(self.new_weights))

    def _EM_test(self, nuisance_params, params=None, param_nums=None,
                 b0_vals=None, F=None, survidx=None, uncens_nobs=None,
                numcensbelow=None, km=None, uncensored=None, censored=None,
                maxiter=None, ftol=None):
        """
        Uses EM algorithm to compute the maximum likelihood of a test

        Parameters
        ----------

        nuisance_params : ndarray
            Vector of values to be used as nuisance params.

        maxiter : int
            Number of iterations in the EM algorithm for a parameter vector

        Returns
        -------
        -2 ''*'' log likelihood ratio at hypothesized values and
        nuisance params

        Notes
        -----
        Optional parameters are provided by the test_beta function.
        """
        iters = 0
        params[param_nums] = b0_vals

        nuis_param_index = np.int_(np.delete(np.arange(self.model.nvar),
                                           param_nums))
        params[nuis_param_index] = nuisance_params
        to_test = params.reshape(self.model.nvar, 1)
        opt_res = np.inf
        diff = np.inf
        while iters < maxiter and diff > ftol:
            F = F.flatten()
            death = np.cumsum(F[::-1])
            survivalprob = death[::-1]
            surv_point_mat = np.dot(F.reshape(-1, 1),
                                1. / survivalprob[survidx].reshape(1, - 1))
            surv_point_mat = add_constant(surv_point_mat)
            summed_wts = np.cumsum(surv_point_mat, axis=1)
            wts = summed_wts[np.int_(np.arange(uncens_nobs)),
                             numcensbelow[uncensored]]
            # ^E step
            # See Zhou 2005, section 3.
            self.model._fit_weights = wts
            new_opt_res = self._opt_wtd_nuis_regress(to_test)
                # ^ Uncensored weights' contribution to likelihood value.
            F = self.new_weights
                # ^ M step
            diff = np.abs(new_opt_res - opt_res)
            opt_res = new_opt_res
            iters = iters + 1
        death = np.cumsum(F.flatten()[::-1])
        survivalprob = death[::-1]
        llike = -opt_res + np.sum(np.log(survivalprob[survidx]))
        wtd_km = km.flatten() / np.sum(km)
        survivalmax = np.cumsum(wtd_km[::-1])[::-1]
        llikemax = np.sum(np.log(wtd_km[uncensored])) + \
          np.sum(np.log(survivalmax[censored]))
        if iters == maxiter:
            warnings.warn('The EM reached the maximum number of iterations',
                          IterationLimitWarning)
        return -2 * (llike - llikemax)

    def _ci_limits_beta(self, b0, param_num=None):
        """
        Returns the difference between the log likelihood for a
        parameter and some critical value.

        Parameters
        ----------
        b0: float
            Value of a regression parameter
        param_num : int
            Parameter index of b0
        """
        return self.test_beta([b0], [param_num])[0] - self.r0


class emplikeAFT:
    """

    Class for estimating and conducting inference in an AFT model.

    Parameters
    ----------

    endog: nx1 array
        Response variables that are subject to random censoring

    exog: nxk array
        Matrix of covariates

    censors: nx1 array
        array with entries 0 or 1.  0 indicates a response was
        censored.

    Attributes
    ----------
    nobs : float
        Number of observations
    endog : ndarray
        Endog attay
    exog : ndarray
        Exogenous variable matrix
    censors
        Censors array but sets the max(endog) to uncensored
    nvar : float
        Number of exogenous variables
    uncens_nobs : float
        Number of uncensored observations
    uncens_endog : ndarray
        Uncensored response variables
    uncens_exog : ndarray
        Exogenous variables of the uncensored observations

    Methods
    -------

    params:
        Fits model parameters

    test_beta:
        Tests if beta = b0 for any vector b0.

    Notes
    -----

    The data is immediately sorted in order of increasing endogenous
    variables

    The last observation is assumed to be uncensored which makes
    estimation and inference possible.
    """
    def __init__(self, endog, exog, censors):
        self.nobs = np.shape(exog)[0]
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = np.asarray(censors).reshape(self.nobs, 1)
        self.nvar = self.exog.shape[1]
        idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[idx]
        self.exog = self.exog[idx]
        self.censors = self.censors[idx]
        self.censors[-1] = 1  # Sort in init, not in function
        self.uncens_nobs = int(np.sum(self.censors))
        mask = self.censors.ravel().astype(bool)
        self.uncens_endog = self.endog[mask, :].reshape(-1, 1)
        self.uncens_exog = self.exog[mask, :]


    def _is_tied(self, endog, censors):
        """
        Indicated if an observation takes the same value as the next
        ordered observation.

        Parameters
        ----------
        endog : ndarray
            Models endogenous variable
        censors : ndarray
            arrat indicating a censored array

        Returns
        -------
        indic_ties : ndarray
            ties[i]=1 if endog[i]==endog[i+1] and
            censors[i]=censors[i+1]
        """
        nobs = int(self.nobs)
        endog_idx = endog[np.arange(nobs - 1)] == (
            endog[np.arange(nobs - 1) + 1])
        censors_idx = censors[np.arange(nobs - 1)] == (
            censors[np.arange(nobs - 1) + 1])
        indic_ties = endog_idx * censors_idx  # Both true
        return np.int_(indic_ties)

    def _km_w_ties(self, tie_indic, untied_km):
        """
        Computes KM estimator value at each observation, taking into acocunt
        ties in the data.

        Parameters
        ----------
        tie_indic: 1d array
            Indicates if the i'th observation is the same as the ith +1
        untied_km: 1d array
            Km estimates at each observation assuming no ties.
        """
        # TODO: Vectorize, even though it is only 1 pass through for any
        # function call
        num_same = 1
        idx_nums = []
        for obs_num in np.arange(int(self.nobs - 1))[::-1]:
            if tie_indic[obs_num] == 1:
                idx_nums.append(obs_num)
                num_same = num_same + 1
                untied_km[obs_num] = untied_km[obs_num + 1]
            elif tie_indic[obs_num] == 0 and num_same > 1:
                idx_nums.append(max(idx_nums) + 1)
                idx_nums = np.asarray(idx_nums)
                untied_km[idx_nums] = untied_km[idx_nums]
                num_same = 1
                idx_nums = []
        return untied_km.reshape(self.nobs, 1)

    def _make_km(self, endog, censors):
        """

        Computes the Kaplan-Meier estimate for the weights in the AFT model

        Parameters
        ----------
        endog: nx1 array
            Array of response variables
        censors: nx1 array
            Censor-indicating variable

        Returns
        -------
        Kaplan Meier estimate for each observation

        Notes
        -----

        This function makes calls to _is_tied and km_w_ties to handle ties in
        the data.If a censored observation and an uncensored observation has
        the same value, it is assumed that the uncensored happened first.
        """
        nobs = self.nobs
        num = (nobs - (np.arange(nobs) + 1.))
        denom = ((nobs - (np.arange(nobs) + 1.) + 1.))
        km = (num / denom).reshape(nobs, 1)
        km = km ** np.abs(censors - 1.)
        km = np.cumprod(km)  # If no ties, this is kaplan-meier
        tied = self._is_tied(endog, censors)
        wtd_km = self._km_w_ties(tied, km)
        return (censors / wtd_km).reshape(nobs, 1)

    def fit(self):
        """

        Fits an AFT model and returns results instance

        Parameters
        ----------
        None


        Returns
        -------
        Results instance.

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        return AFTResults(self)

    def predict(self, params, endog=None):
        if endog is None:
            endog = self.endog
        return np.dot(endog, params)


class AFTResults(OptAFT):
    def __init__(self, model):
        self.model = model

    def params(self):
        """

        Fits an AFT model and returns parameters.

        Parameters
        ----------
        None


        Returns
        -------
        Fitted params

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        self.model.modif_censors = np.copy(self.model.censors)
        self.model.modif_censors[-1] = 1
        wts = self.model._make_km(self.model.endog, self.model.modif_censors)
        res = WLS(self.model.endog, self.model.exog, wts).fit()
        params = res.params
        return params

    def test_beta(self, b0_vals, param_nums, ftol=10 ** - 5, maxiter=30,
                  print_weights=1):
        """
        Returns the profile log likelihood for regression parameters
        'param_num' at 'b0_vals.'

        Parameters
        ----------
        b0_vals : list
            The value of parameters to be tested
        param_num : list
            Which parameters to be tested
        maxiter : int, optional
            How many iterations to use in the EM algorithm.  Default is 30
        ftol : float, optional
            The function tolerance for the EM optimization.
            Default is 10''**''-5
        print_weights : bool
            If true, returns the weights tate maximize the profile
            log likelihood. Default is False

        Returns
        -------

        test_results : tuple
            The log-likelihood and p-pvalue of the test.

        Notes
        -----

        The function will warn if the EM reaches the maxiter.  However, when
        optimizing over nuisance parameters, it is possible to reach a
        maximum number of inner iterations for a specific value for the
        nuisance parameters while the resultsof the function are still valid.
        This usually occurs when the optimization over the nuisance parameters
        selects parameter values that yield a log-likihood ratio close to
        infinity.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import numpy as np

        # Test parameter is .05 in one regressor no intercept model
        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, x, cens)
        >>> res=model.test_beta([0], [0])
        >>> res
        (1.4657739632606308, 0.22601365256959183)

        #Test slope is 0 in  model with intercept

        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, sm.add_constant(x), cens)
        >>> res = model.test_beta([0], [1])
        >>> res
        (4.623487775078047, 0.031537049752572731)
        """
        censors = self.model.censors
        endog = self.model.endog
        exog = self.model.exog
        uncensored = (censors == 1).flatten()
        censored = (censors == 0).flatten()
        uncens_endog = endog[uncensored]
        uncens_exog = exog[uncensored, :]
        reg_model = OLS(uncens_endog, uncens_exog).fit()
        llr, pval, new_weights = reg_model.el_test(b0_vals, param_nums,
                                      return_weights=True)  # Needs to be changed
        km = self.model._make_km(endog, censors).flatten()  # when merged
        uncens_nobs = self.model.uncens_nobs
        F = np.asarray(new_weights).reshape(uncens_nobs)
        # Step 0 ^
        params = self.params()
        survidx = np.where(censors == 0)
        survidx = survidx[0] - np.arange(len(survidx[0]))
        numcensbelow = np.int_(np.cumsum(1 - censors))
        if len(param_nums) == len(params):
            llr = self._EM_test([], F=F, params=params,
                                      param_nums=param_nums,
                                b0_vals=b0_vals, survidx=survidx,
                             uncens_nobs=uncens_nobs,
                             numcensbelow=numcensbelow, km=km,
                             uncensored=uncensored, censored=censored,
                             ftol=ftol, maxiter=25)
            return llr, chi2.sf(llr, self.model.nvar)
        else:
            x0 = np.delete(params, param_nums)
            try:
                res = optimize.fmin(self._EM_test, x0,
                                   (params, param_nums, b0_vals, F, survidx,
                                    uncens_nobs, numcensbelow, km, uncensored,
                                    censored, maxiter, ftol), full_output=1,
                                    disp=0)

                llr = res[1]
                return llr, chi2.sf(llr, len(param_nums))
            except np.linalg.linalg.LinAlgError:
                return np.inf, 0

    def ci_beta(self, param_num, beta_high, beta_low, sig=.05):
        """
        Returns the confidence interval for a regression
        parameter in the AFT model.

        Parameters
        ----------
        param_num : int
            Parameter number of interest
        beta_high : float
            Upper bound for the confidence interval
        beta_low : float
            Lower bound for the confidence interval
        sig : float, optional
            Significance level.  Default is .05

        Notes
        -----
        If the function returns f(a) and f(b) must have different signs,
        consider widening the search area by adjusting beta_low and
        beta_high.

        Also note that this process is computational intensive.  There
        are 4 levels of optimization/solving.  From outer to inner:

        1) Solving so that llr-critical value = 0
        2) maximizing over nuisance parameters
        3) Using  EM at each value of nuisamce parameters
        4) Using the _modified_Newton optimizer at each iteration
           of the EM algorithm.

        Also, for very unlikely nuisance parameters, it is possible for
        the EM algorithm to not converge.  This is not an indicator
        that the solver did not find the correct solution.  It just means
        for a specific iteration of the nuisance parameters, the optimizer
        was unable to converge.

        If the user desires to verify the success of the optimization,
        it is recommended to test the limits using test_beta.
        """
        params = self.params()
        self.r0 = chi2.ppf(1 - sig, 1)
        ll = optimize.brentq(self._ci_limits_beta, beta_low,
                             params[param_num], (param_num))
        ul = optimize.brentq(self._ci_limits_beta,
                             params[param_num], beta_high, (param_num))
        return ll, ul
