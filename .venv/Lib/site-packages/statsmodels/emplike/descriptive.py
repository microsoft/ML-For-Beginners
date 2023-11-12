"""
Empirical likelihood inference on descriptive statistics

This module conducts hypothesis tests and constructs confidence
intervals for the mean, variance, skewness, kurtosis and correlation.

If matplotlib is installed, this module can also generate multivariate
confidence region plots as well as mean-variance contour plots.

See _OptFuncts docstring for technical details and optimization variable
definitions.

General References:
------------------
Owen, A. (2001). "Empirical Likelihood." Chapman and Hall

"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils


def DescStat(endog):
    """
    Returns an instance to conduct inference on descriptive statistics
    via empirical likelihood.  See DescStatUV and DescStatMV for more
    information.

    Parameters
    ----------
    endog : ndarray
         Array of data

    Returns : DescStat instance
        If k=1, the function returns a univariate instance, DescStatUV.
        If k>1, the function returns a multivariate instance, DescStatMV.
    """
    if endog.ndim == 1:
        endog = endog.reshape(len(endog), 1)
    if endog.shape[1] == 1:
        return DescStatUV(endog)
    if endog.shape[1] > 1:
        return DescStatMV(endog)


class _OptFuncts:
    """
    A class that holds functions that are optimized/solved.

    The general setup of the class is simple.  Any method that starts with
    _opt_ creates a vector of estimating equations named est_vect such that
    np.dot(p, (est_vect))=0 where p is the weight on each
    observation as a 1 x n array and est_vect is n x k.  Then _modif_Newton is
    called to determine the optimal p by solving for the Lagrange multiplier
    (eta) in the profile likelihood maximization problem.  In the presence
    of nuisance parameters, _opt_ functions are  optimized over to profile
    out the nuisance parameters.

    Any method starting with _ci_limits calculates the log likelihood
    ratio for a specific value of a parameter and then subtracts a
    pre-specified critical value.  This is solved so that llr - crit = 0.
    """

    def __init__(self, endog):
        pass

    def _log_star(self, eta, est_vect, weights, nobs):
        """
        Transforms the log of observation probabilities in terms of the
        Lagrange multiplier to the log 'star' of the probabilities.

        Parameters
        ----------
        eta : float
            Lagrange multiplier

        est_vect : ndarray (n,k)
            Estimating equations vector

        wts : nx1 array
            Observation weights

        Returns
        ------
        data_star : ndarray
            The weighted logstar of the estimting equations

        Notes
        -----
        This function is only a placeholder for the _fit_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computing the log likelihood ratio.
        """
        data_star = np.log(weights) + (np.sum(weights) +\
                                       np.dot(est_vect, eta))
        idx = data_star < 1. / nobs
        not_idx = ~idx
        nx = nobs * data_star[idx]
        data_star[idx] = np.log(1. / nobs) - 1.5 + nx * (2. - nx / 2)
        data_star[not_idx] = np.log(data_star[not_idx])
        return data_star

    def _hess(self, eta, est_vect, weights, nobs):
        """
        Calculates the hessian of a weighted empirical likelihood
        problem.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        hess : m x m array
            Weighted hessian used in _wtd_modif_newton
        """
        #eta = np.squeeze(eta)
        data_star_doub_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_doub_prime < 1. / nobs
        not_idx = ~idx
        data_star_doub_prime[idx] = - nobs ** 2
        data_star_doub_prime[not_idx] = - (data_star_doub_prime[not_idx]) ** -2
        wtd_dsdp = weights * data_star_doub_prime
        return np.dot(est_vect.T, wtd_dsdp[:, None] * est_vect)

    def _grad(self, eta, est_vect, weights, nobs):
        """
        Calculates the gradient of a weighted empirical likelihood
        problem

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        gradient : ndarray (m,1)
            The gradient used in _wtd_modif_newton
        """
        #eta = np.squeeze(eta)
        data_star_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_prime < 1. / nobs
        not_idx = ~idx
        data_star_prime[idx] = nobs * (2 - nobs * data_star_prime[idx])
        data_star_prime[not_idx] = 1. / data_star_prime[not_idx]
        return np.dot(weights * data_star_prime, est_vect)

    def _modif_newton(self,  eta, est_vect, weights):
        """
        Modified Newton's method for maximizing the log 'star' equation.  This
        function calls _fit_newton to find the optimal values of eta.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        params : 1xm array
            Lagrange multiplier that maximizes the log-likelihood
        """
        nobs = len(est_vect)
        f = lambda x0: - np.sum(self._log_star(x0, est_vect, weights, nobs))
        grad = lambda x0: - self._grad(x0, est_vect, weights, nobs)
        hess = lambda x0: - self._hess(x0, est_vect, weights, nobs)
        kwds = {'tol': 1e-8}
        eta = eta.squeeze()
        res = _fit_newton(f, grad, eta, (), kwds, hess=hess, maxiter=50, \
                              disp=0)
        return res[0]

    def _find_eta(self, eta):
        """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
        eta when computing ELR for univariate mean.

        Parameters
        ----------
        eta : float
            Lagrange multiplier in the empirical likelihood maximization

        Returns
        -------
        llr : float
            n times the log likelihood value for a given value of eta
        """
        return np.sum((self.endog - self.mu0) / \
              (1. + eta * (self.endog - self.mu0)))

    def _ci_limits_mu(self, mu):
        """
        Calculates the difference between the log likelihood of mu_test and a
        specified critical value.

        Parameters
        ----------
        mu : float
           Hypothesized value of the mean.

        Returns
        -------
        diff : float
            The difference between the log likelihood value of mu0 and
            a specified value.
        """
        return self.test_mean(mu)[0] - self.r0

    def _find_gamma(self, gamma):
        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean

        Parameters
        ----------
        gamma : float
            Lagrange multiplier when computing confidence interval

        Returns
        -------
        diff : float
            The difference between the log-liklihood when the Lagrange
            multiplier is gamma and a pre-specified value
        """
        denom = np.sum((self.endog - gamma) ** -1)
        new_weights = (self.endog - gamma) ** -1 / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - \
            self.r0

    def _opt_var(self, nuisance_mu, pval=False):
        """
        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance

        Parameters
        ----------
        nuisance_mu : float
            Value of a nuisance mean parameter

        Returns
        -------
        llr : float
            Log likelihood of a pre-specified variance holding the nuisance
            parameter constant
        """
        endog = self.endog
        nobs = self.nobs
        sig_data = ((endog - nuisance_mu) ** 2 \
                    - self.sig2_0)
        mu_data = (endog - nuisance_mu)
        est_vect = np.column_stack((mu_data, sig_data))
        eta_star = self._modif_newton(np.array([1. / nobs,
                                               1. / nobs]), est_vect,
                                                np.ones(nobs) * (1. / nobs))

        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1. / nobs * 1. / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        if pval:  # Used for contour plotting
            return chi2.sf(-2 * llr, 1)
        return -2 * llr

    def _ci_limits_var(self, var):
        """
        Used to determine the confidence intervals for the variance.
        It calls test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        Parameters
        ----------
        var_test : float
            Hypothesized value of the variance

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at var_test and a
            pre-specified value.
        """
        return self.test_var(var)[0] - self.r0

    def _opt_skew(self, nuis_params):
        """
        Called by test_skew.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a  nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-specified skewness holding
            the nuisance parameters constant.
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = ((endog - nuis_params[0]) ** 2) - nuis_params[1]
        skew_data = ((((endog - nuis_params[0]) ** 3) /
                    (nuis_params[1] ** 1.5))) - self.skew0
        est_vect = np.column_stack((mu_data, sig_data, skew_data))
        eta_star = self._modif_newton(np.array([1. / nobs,
                                               1. / nobs,
                                               1. / nobs]), est_vect,
                                               np.ones(nobs) * (1. / nobs))
        denom = 1. + np.dot(eta_star, est_vect.T)
        self.new_weights = 1. / nobs * 1. / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_kurt(self, nuis_params):
        """
        Called by test_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-speified kurtosis holding the
            nuisance parameters constant
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = ((endog - nuis_params[0]) ** 2) - nuis_params[1]
        kurt_data = (((((endog - nuis_params[0]) ** 4) / \
                    (nuis_params[1] ** 2))) - 3) - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, kurt_data))
        eta_star = self._modif_newton(np.array([1. / nobs,
                                               1. / nobs,
                                               1. / nobs]), est_vect,
                                               np.ones(nobs) * (1. / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1. / nobs * 1. / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_skew_kurt(self, nuis_params):
        """
        Called by test_joint_skew_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        ------
        llr : float
            The log likelihood ratio of a pre-speified skewness and
            kurtosis holding the nuisance parameters constant.
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = ((endog - nuis_params[0]) ** 2) - nuis_params[1]
        skew_data = ((((endog - nuis_params[0]) ** 3) / \
                    (nuis_params[1] ** 1.5))) - self.skew0
        kurt_data = (((((endog - nuis_params[0]) ** 4) / \
                    (nuis_params[1] ** 2))) - 3) - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, skew_data, kurt_data))
        eta_star = self._modif_newton(np.array([1. / nobs,
                                               1. / nobs,
                                               1. / nobs,
                                               1. / nobs]), est_vect,
                                               np.ones(nobs) * (1. / nobs))
        denom = 1. + np.dot(eta_star, est_vect.T)
        self.new_weights = 1. / nobs * 1. / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_skew(self, skew):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of skewness

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at skew and a
            pre-specified value.
        """
        return self.test_skew(skew)[0] - self.r0

    def _ci_limits_kurt(self, kurt):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of kurtosis

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at kurt and a
            pre-specified value.
        """
        return self.test_kurt(kurt)[0] - self.r0

    def _opt_correl(self, nuis_params, corr0, endog, nobs, x0, weights0):
        """
        Parameters
        ----------
        nuis_params : 1darray
            Array containing two nuisance means and two nuisance variances

        Returns
        -------
        llr : float
            The log-likelihood of the correlation coefficient holding nuisance
            parameters constant
        """
        mu1_data, mu2_data = (endog - nuis_params[::2]).T
        sig1_data = mu1_data ** 2 - nuis_params[1]
        sig2_data = mu2_data ** 2 - nuis_params[3]
        correl_data = ((mu1_data * mu2_data) - corr0 *
                    (nuis_params[1] * nuis_params[3]) ** .5)
        est_vect = np.column_stack((mu1_data, sig1_data,
                                    mu2_data, sig2_data, correl_data))
        eta_star = self._modif_newton(x0, est_vect, weights0)
        denom = 1. + np.dot(est_vect, eta_star)
        self.new_weights = 1. / nobs * 1. / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_corr(self, corr):
        return self.test_corr(corr)[0] - self.r0


class DescStatUV(_OptFuncts):
    """
    A class to compute confidence intervals and hypothesis tests involving
    mean, variance, kurtosis and skewness of a univariate random variable.

    Parameters
    ----------
    endog : 1darray
        Data to be analyzed

    Attributes
    ----------
    endog : 1darray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = np.squeeze(endog)
        self.nobs = endog.shape[0]

    def test_mean(self, mu0, return_weights=False):
        """
        Returns - 2 x log-likelihood ratio, p-value and weights
        for a hypothesis test of the mean.

        Parameters
        ----------
        mu0 : float
            Mean value to be tested

        return_weights : bool
            If return_weights is True the function returns
            the weights of the observations under the null hypothesis.
            Default is False

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of mu0
        """
        self.mu0 = mu0
        endog = self.endog
        nobs = self.nobs
        eta_min = (1. - (1. / nobs)) / (self.mu0 - max(endog))
        eta_max = (1. - (1. / nobs)) / (self.mu0 - min(endog))
        eta_star = optimize.brentq(self._find_eta, eta_min, eta_max)
        new_weights = (1. / nobs) * 1. / (1. + eta_star * (endog - self.mu0))
        llr = -2 * np.sum(np.log(nobs * new_weights))
        if return_weights:
            return llr, chi2.sf(llr, 1), new_weights
        else:
            return llr, chi2.sf(llr, 1)

    def ci_mean(self, sig=.05, method='gamma', epsilon=10 ** -8,
                 gamma_low=-10 ** 10, gamma_high=10 ** 10):
        """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig : float
            significance level. Default is .05

        method : str
            Root finding method,  Can be 'nested-brent' or
            'gamma'.  Default is 'gamma'

            'gamma' Tries to solve for the gamma parameter in the
            Lagrange (see Owen pg 22) and then determine the weights.

            'nested brent' uses brents method to find the confidence
            intervals but must maximize the likelihood ratio on every
            iteration.

            gamma is generally much faster.  If the optimizations does not
            converge, try expanding the gamma_high and gamma_low
            variable.

        gamma_low : float
            Lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low.

        gamma_high : float
            Upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high.

        epsilon : float
            When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ``**`` -6) consider shrinking epsilon

            When using 'gamma', amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If function returns f(a) and f(b) must have different signs,
            consider lowering epsilon.

        Returns
        -------
        Interval : tuple
            Confidence interval for the mean
        """
        endog = self.endog
        sig = 1 - sig
        if method == 'nested-brent':
            self.r0 = chi2.ppf(sig, 1)
            middle = np.mean(endog)
            epsilon_u = (max(endog) - np.mean(endog)) * epsilon
            epsilon_l = (np.mean(endog) - min(endog)) * epsilon
            ulim = optimize.brentq(self._ci_limits_mu, middle,
                max(endog) - epsilon_u)
            llim = optimize.brentq(self._ci_limits_mu, middle,
                min(endog) + epsilon_l)
            return llim, ulim

        if method == 'gamma':
            self.r0 = chi2.ppf(sig, 1)
            gamma_star_l = optimize.brentq(self._find_gamma, gamma_low,
                min(endog) - epsilon)
            gamma_star_u = optimize.brentq(self._find_gamma, \
                         max(endog) + epsilon, gamma_high)
            weights_low = ((endog - gamma_star_l) ** -1) / \
                np.sum((endog - gamma_star_l) ** -1)
            weights_high = ((endog - gamma_star_u) ** -1) / \
                np.sum((endog - gamma_star_u) ** -1)
            mu_low = np.sum(weights_low * endog)
            mu_high = np.sum(weights_high * endog)
            return mu_low,  mu_high

    def test_var(self, sig2_0, return_weights=False):
        """
        Returns  -2 x log-likelihood ratio and the p-value for the
        hypothesized variance

        Parameters
        ----------
        sig2_0 : float
            Hypothesized variance to be tested

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of observing sig2_0. Default is False

        Returns
        -------
        test_results : tuple
            The  log-likelihood ratio and the p_value  of sig2_0

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(1000)*100
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> hyp_test = el_analysis.test_var(9500)
        """
        self.sig2_0 = sig2_0
        mu_max = max(self.endog)
        mu_min = min(self.endog)
        llr = optimize.fminbound(self._opt_var, mu_min, mu_max, \
                                 full_output=1)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return llr, p_val, self.new_weights.T
        else:
            return llr, p_val

    def ci_var(self, lower_bound=None, upper_bound=None, sig=.05):
        """
        Returns the confidence interval for the variance.

        Parameters
        ----------
        lower_bound : float
            The minimum value the lower confidence interval can
            take. The p-value from test_var(lower_bound) must be lower
            than 1 - significance level. Default is .99 confidence
            limit assuming normality

        upper_bound : float
            The maximum value the upper confidence interval
            can take. The p-value from test_var(upper_bound) must be lower
            than 1 - significance level.  Default is .99 confidence
            limit assuming normality

        sig : float
            The significance level. Default is .05

        Returns
        -------
        Interval : tuple
            Confidence interval for the variance

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(100)
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> el_analysis.ci_var()
        (0.7539322567470305, 1.229998852496268)
        >>> el_analysis.ci_var(.5, 2)
        (0.7539322567469926, 1.2299988524962664)

        Notes
        -----
        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering lower_bound and raising
        upper_bound.
        """
        endog = self.endog
        if upper_bound is None:
            upper_bound = ((self.nobs - 1) * endog.var()) / \
              (chi2.ppf(.0001, self.nobs - 1))
        if lower_bound is None:
            lower_bound = ((self.nobs - 1) * endog.var()) / \
              (chi2.ppf(.9999, self.nobs - 1))
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_var, lower_bound, endog.var())
        ulim = optimize.brentq(self._ci_limits_var, endog.var(), upper_bound)
        return llim, ulim

    def plot_contour(self, mu_low, mu_high, var_low, var_high, mu_step,
                        var_step,
                        levs=[.2, .1, .05, .01, .001]):
        """
        Returns a plot of the confidence region for a univariate
        mean and variance.

        Parameters
        ----------
        mu_low : float
            Lowest value of the mean to plot

        mu_high : float
            Highest value of the mean to plot

        var_low : float
            Lowest value of the variance to plot

        var_high : float
            Highest value of the variance to plot

        mu_step : float
            Increments to evaluate the mean

        var_step : float
            Increments to evaluate the mean

        levs : list
            Which values of significance the contour lines will be drawn.
            Default is [.2, .1, .05, .01, .001]

        Returns
        -------
        Figure
            The contour plot
        """
        fig, ax = utils.create_mpl_ax()
        ax.set_ylabel('Variance')
        ax.set_xlabel('Mean')
        mu_vect = list(np.arange(mu_low, mu_high, mu_step))
        var_vect = list(np.arange(var_low, var_high, var_step))
        z = []
        for sig0 in var_vect:
            self.sig2_0 = sig0
            for mu0 in mu_vect:
                z.append(self._opt_var(mu0, pval=True))
        z = np.asarray(z).reshape(len(var_vect), len(mu_vect))
        ax.contour(mu_vect, var_vect, z, levels=levs)
        return fig

    def test_skew(self, skew0, return_weights=False):
        """
        Returns  -2 x log-likelihood and p-value for the hypothesized
        skewness.

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p_value of skew0
        """
        self.skew0 = skew0
        start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])

        llr = optimize.fmin_powell(self._opt_skew, start_nuisance,
                                     full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return llr, p_val,  self.new_weights.T
        return llr, p_val

    def test_kurt(self, kurt0, return_weights=False):
        """
        Returns -2 x log-likelihood and the p-value for the hypothesized
        kurtosis.

        Parameters
        ----------
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of kurt0
        """
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])

        llr = optimize.fmin_powell(self._opt_kurt, start_nuisance,
                                     full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return llr, p_val, self.new_weights.T
        return llr, p_val

    def test_joint_skew_kurt(self, skew0, kurt0, return_weights=False):
        """
        Returns - 2 x log-likelihood and the p-value for the joint
        hypothesis test for skewness and kurtosis

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value  of the joint hypothesis test.
        """
        self.skew0 = skew0
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])

        llr = optimize.fmin_powell(self._opt_skew_kurt, start_nuisance,
                                     full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 2)
        if return_weights:
            return llr, p_val, self.new_weights.T
        return llr, p_val

    def ci_skew(self, sig=.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for skewness.

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of skewness the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of skewness the lower limit can be.
            Default is .99 confidence level assuming normality.

        Returns
        -------
        Interval : tuple
            Confidence interval for the skewness

        Notes
        -----
        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bounds
        """
        nobs = self.nobs
        endog = self.endog
        if upper_bound is None:
            upper_bound = skew(endog) + \
            2.5 * ((6. * nobs * (nobs - 1.)) / \
              ((nobs - 2.) * (nobs + 1.) * \
               (nobs + 3.))) ** .5
        if lower_bound is None:
            lower_bound = skew(endog) - \
            2.5 * ((6. * nobs * (nobs - 1.)) / \
              ((nobs - 2.) * (nobs + 1.) * \
               (nobs + 3.))) ** .5
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_skew, lower_bound, skew(endog))
        ulim = optimize.brentq(self._ci_limits_skew, skew(endog), upper_bound)
        return llim, ulim

    def ci_kurt(self, sig=.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for kurtosis.

        Parameters
        ----------

        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of kurtosis the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of kurtosis the lower limit can be.
            Default is .99 confidence limit assuming normality.

        Returns
        -------
        Interval : tuple
            Lower and upper confidence limit

        Notes
        -----
        For small n, upper_bound and lower_bound may have to be
        provided by the user.  Consider using test_kurt to find
        values close to the desired significance level.

        If function returns f(a) and f(b) must have different signs, consider
        expanding the bounds.
        """
        endog = self.endog
        nobs = self.nobs
        if upper_bound is None:
            upper_bound = kurtosis(endog) + \
            (2.5 * (2. * ((6. * nobs * (nobs - 1.)) / \
              ((nobs - 2.) * (nobs + 1.) * \
               (nobs + 3.))) ** .5) * \
               (((nobs ** 2.) - 1.) / ((nobs - 3.) *\
                 (nobs + 5.))) ** .5)
        if lower_bound is None:
            lower_bound = kurtosis(endog) - \
            (2.5 * (2. * ((6. * nobs * (nobs - 1.)) / \
              ((nobs - 2.) * (nobs + 1.) * \
               (nobs + 3.))) ** .5) * \
               (((nobs ** 2.) - 1.) / ((nobs - 3.) *\
                 (nobs + 5.))) ** .5)
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_kurt, lower_bound, \
                             kurtosis(endog))
        ulim = optimize.brentq(self._ci_limits_kurt, kurtosis(endog), \
                             upper_bound)
        return llim, ulim


class DescStatMV(_OptFuncts):
    """
    A class for conducting inference on multivariate means and correlation.

    Parameters
    ----------
    endog : ndarray
        Data to be analyzed

    Attributes
    ----------
    endog : ndarray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = endog
        self.nobs = endog.shape[0]

    def mv_test_mean(self, mu_array, return_weights=False):
        """
        Returns -2 x log likelihood and the p-value
        for a multivariate hypothesis test of the mean

        Parameters
        ----------
        mu_array  : 1d array
            Hypothesized values for the mean.  Must have same number of
            elements as columns in endog

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of mu_array. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value for mu_array
        """
        endog = self.endog
        nobs = self.nobs
        if len(mu_array) != endog.shape[1]:
            raise ValueError('mu_array must have the same number of '
                             'elements as the columns of the data.')
        mu_array = mu_array.reshape(1, endog.shape[1])
        means = np.ones((endog.shape[0], endog.shape[1]))
        means = mu_array * means
        est_vect = endog - means
        start_vals = 1. / nobs * np.ones(endog.shape[1])
        eta_star = self._modif_newton(start_vals, est_vect,
                                      np.ones(nobs) * (1. / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1 / nobs * 1 / denom
        llr = -2 * np.sum(np.log(nobs * self.new_weights))
        p_val = chi2.sf(llr, mu_array.shape[1])
        if return_weights:
            return llr, p_val,  self.new_weights.T
        else:
            return llr, p_val

    def mv_mean_contour(self, mu1_low, mu1_upp, mu2_low, mu2_upp, step1, step2,
                        levs=(.001, .01, .05, .1, .2), var1_name=None,
                        var2_name=None, plot_dta=False):
        """
        Creates a confidence region plot for the mean of bivariate data

        Parameters
        ----------
        m1_low : float
            Minimum value of the mean for variable 1
        m1_upp : float
            Maximum value of the mean for variable 1
        mu2_low : float
            Minimum value of the mean for variable 2
        mu2_upp : float
            Maximum value of the mean for variable 2
        step1 : float
            Increment of evaluations for variable 1
        step2 : float
            Increment of evaluations for variable 2
        levs : list
            Levels to be drawn on the contour plot.
            Default =  (.001, .01, .05, .1, .2)
        plot_dta : bool
            If True, makes a scatter plot of the data on
            top of the contour plot. Defaultis False.
        var1_name : str
            Name of variable 1 to be plotted on the x-axis
        var2_name : str
            Name of variable 2 to be plotted on the y-axis

        Notes
        -----
        The smaller the step size, the more accurate the intervals
        will be

        If the function returns optimization failed, consider narrowing
        the boundaries of the plot

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> two_rvs = np.random.standard_normal((20,2))
        >>> el_analysis = sm.emplike.DescStat(two_rvs)
        >>> contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)
        >>> contourp.show()
        """
        if self.endog.shape[1] != 2:
            raise ValueError('Data must contain exactly two variables')
        fig, ax = utils.create_mpl_ax()
        if var2_name is None:
            ax.set_ylabel('Variable 2')
        else:
            ax.set_ylabel(var2_name)
        if var1_name is None:
            ax.set_xlabel('Variable 1')
        else:
            ax.set_xlabel(var1_name)
        x = np.arange(mu1_low, mu1_upp, step1)
        y = np.arange(mu2_low, mu2_upp, step2)
        pairs = itertools.product(x, y)
        z = []
        for i in pairs:
            z.append(self.mv_test_mean(np.asarray(i))[0])
        X, Y = np.meshgrid(x, y)
        z = np.asarray(z)
        z = z.reshape(X.shape[1], Y.shape[0])
        ax.contour(x, y, z.T, levels=levs)
        if plot_dta:
            ax.plot(self.endog[:, 0], self.endog[:, 1], 'bo')
        return fig

    def test_corr(self, corr0, return_weights=0):
        """
        Returns -2 x log-likelihood ratio and  p-value for the
        correlation coefficient between 2 variables

        Parameters
        ----------
        corr0 : float
            Hypothesized value to be tested

        return_weights : bool
            If true, returns the weights that maximize
            the log-likelihood at the hypothesized value
        """
        nobs = self.nobs
        endog = self.endog
        if endog.shape[1] != 2:
            raise NotImplementedError('Correlation matrix not yet implemented')
        nuis0 = np.array([endog[:, 0].mean(),
                              endog[:, 0].var(),
                              endog[:, 1].mean(),
                              endog[:, 1].var()])

        x0 = np.zeros(5)
        weights0 = np.array([1. / nobs] * int(nobs))
        args = (corr0, endog, nobs, x0, weights0)
        llr = optimize.fmin(self._opt_correl, nuis0, args=args,
                                     full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return llr, p_val, self.new_weights.T
        return llr, p_val

    def ci_corr(self, sig=.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence intervals for the correlation coefficient

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value the upper confidence limit can be.
            Default is  99% confidence limit assuming normality.

        lower_bound : float
            Minimum value the lower confidence limit can be.
            Default is 99% confidence limit assuming normality.

        Returns
        -------
        interval : tuple
            Confidence interval for the correlation
        """
        endog = self.endog
        nobs = self.nobs
        self.r0 = chi2.ppf(1 - sig, 1)
        point_est = np.corrcoef(endog[:, 0], endog[:, 1])[0, 1]
        if upper_bound is None:
            upper_bound = min(.999, point_est + \
                          2.5 * ((1. - point_est ** 2.) / \
                          (nobs - 2.)) ** .5)

        if lower_bound is None:
            lower_bound = max(- .999, point_est - \
                          2.5 * (np.sqrt((1. - point_est ** 2.) / \
                          (nobs - 2.))))

        llim = optimize.brenth(self._ci_limits_corr, lower_bound, point_est)
        ulim = optimize.brenth(self._ci_limits_corr, point_est, upper_bound)
        return llim, ulim
