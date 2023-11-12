'''
The one parameter exponential family distributions used by GLM.
'''
# TODO: quasi, quasibinomial, quasipoisson
# see
# http://www.biostat.jhsph.edu/~qli/biostatistics_r_doc/library/stats/html/family.html
# for comparison to R, and McCullagh and Nelder


import inspect
import warnings

import numpy as np
from scipy import special, stats

from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
    ValueWarning,
    )
from . import links as L, varfuncs as V

FLOAT_EPS = np.finfo(float).eps


class Family:
    """
    The parent class for one-parameter exponential families.

    Parameters
    ----------
    link : a link function instance
        Link is the linear transformation function.
        See the individual families for available links.
    variance : a variance function
        Measures the variance as a function of the mean probabilities.
        See the individual families for the default variance function.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    See Also
    --------
    :ref:`links` : Further details on links.
    """
    # TODO: change these class attributes, use valid somewhere...
    valid = [-np.inf, np.inf]
    links = []

    def _setlink(self, link):
        """
        Helper method to set the link for a family.

        Raises a ``ValueError`` exception if the link is not available. Note
        that  the error message might not be that informative because it tells
        you that the link should be in the base class for the link function.

        See statsmodels.genmod.generalized_linear_model.GLM for a list of
        appropriate links for each family but note that not all of these are
        currently available.
        """
        # TODO: change the links class attribute in the families to hold
        # meaningful information instead of a list of links instances such as
        # [<statsmodels.family.links.Log object at 0x9a4240c>,
        #  <statsmodels.family.links.Power object at 0x9a423ec>,
        #  <statsmodels.family.links.Power object at 0x9a4236c>]
        # for Poisson...
        self._link = link
        if self._check_link:
            if not isinstance(link, L.Link):
                raise TypeError("The input should be a valid Link object.")
            if hasattr(self, "links"):
                validlink = max([isinstance(link, _) for _ in self.links])
                if not validlink:
                    msg = "Invalid link for family, should be in %s. (got %s)"
                    raise ValueError(msg % (repr(self.links), link))

    def _getlink(self):
        """
        Helper method to get the link for a family.
        """
        return self._link

    # link property for each family is a pointer to link instance
    link = property(_getlink, _setlink, doc="Link function for family")

    def __init__(self, link, variance, check_link=True):
        self._check_link = check_link
        if inspect.isclass(link):
            warnmssg = (
                "Calling Family(..) with a link class is not allowed. Use an "
                "instance of a link class instead."
            )
            raise TypeError(warnmssg)

        self.link = link
        self.variance = variance

    def starting_mu(self, y):
        r"""
        Starting value for mu in the IRLS algorithm.

        Parameters
        ----------
        y : ndarray
            The untransformed response variable.

        Returns
        -------
        mu_0 : ndarray
            The first guess on the transformed response variable.

        Notes
        -----
        .. math::

           \mu_0 = (Y + \overline{Y})/2

        Only the Binomial family takes a different initial value.
        """
        return (y + y.mean())/2.

    def weights(self, mu):
        r"""
        Weights for IRLS steps

        Parameters
        ----------
        mu : array_like
            The transformed mean response variable in the exponential family

        Returns
        -------
        w : ndarray
            The weights for the IRLS steps

        Notes
        -----
        .. math::

           w = 1 / (g'(\mu)^2  * Var(\mu))
        """
        return 1. / (self.link.deriv(mu)**2 * self.variance(mu))

    def deviance(self, endog, mu, var_weights=1., freq_weights=1., scale=1.):
        r"""
        The deviance function evaluated at (endog, mu, var_weights,
        freq_weights, scale) for the distribution.

        Deviance is usually defined as twice the loglikelihood ratio.

        Parameters
        ----------
        endog : array_like
            The endogenous response variable
        mu : array_like
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        freq_weights : array_like
            1d array of frequency weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        Deviance : ndarray
            The value of deviance function defined below.

        Notes
        -----
        Deviance is defined

        .. math::

           D = 2\sum_i (freq\_weights_i * var\_weights *
           (llf(endog_i, endog_i) - llf(endog_i, \mu_i)))

        where y is the endogenous variable. The deviance functions are
        analytically defined for each family.

        Internally, we calculate deviance as:

        .. math::
           D = \sum_i freq\_weights_i * var\_weights * resid\_dev_i  / scale
        """
        resid_dev = self._resid_dev(endog, mu)
        return np.sum(resid_dev * freq_weights * var_weights / scale)

    def resid_dev(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The deviance residuals

        Parameters
        ----------
        endog : array_like
            The endogenous response variable
        mu : array_like
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        The deviance residuals are defined by the contribution D_i of
        observation i to the deviance as

        .. math::
           resid\_dev_i = sign(y_i-\mu_i) \sqrt{D_i}

        D_i is calculated from the _resid_dev method in each family.
        Distribution-specific documentation of the calculation is available
        there.
        """
        resid_dev = self._resid_dev(endog, mu)
        resid_dev *= var_weights / scale
        return np.sign(endog - mu) * np.sqrt(np.clip(resid_dev, 0., np.inf))

    def fitted(self, lin_pred):
        r"""
        Fitted values based on linear predictors lin_pred.

        Parameters
        ----------
        lin_pred : ndarray
            Values of the linear predictor of the model.
            :math:`X \cdot \beta` in a classical linear model.

        Returns
        -------
        mu : ndarray
            The mean response variables given by the inverse of the link
            function.
        """
        fits = self.link.inverse(lin_pred)
        return fits

    def predict(self, mu):
        """
        Linear predictors based on given mu values.

        Parameters
        ----------
        mu : ndarray
            The mean response variables

        Returns
        -------
        lin_pred : ndarray
            Linear predictors based on the mean response variables.  The value
            of the link function at the given mu.
        """
        return self.link(mu)

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        This is defined for each family. endog and mu are not restricted to
        ``endog`` and ``mu`` respectively.  For instance, you could call
        both ``loglike(endog, endog)`` and ``loglike(endog, mu)`` to get the
        log-likelihood ratio.
        """
        raise NotImplementedError

    def loglike(self, endog, mu, var_weights=1., freq_weights=1., scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        freq_weights : array_like
            1d array of frequency weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, freq_weights, scale) as defined below.

        Notes
        -----
        Where :math:`ll_i` is the by-observation log-likelihood:

        .. math::
           ll = \sum(ll_i * freq\_weights_i)

        ``ll_i`` is defined for each family. endog and mu are not restricted
        to ``endog`` and ``mu`` respectively.  For instance, you could call
        both ``loglike(endog, endog)`` and ``loglike(endog, mu)`` to get the
        log-likelihood ratio.
        """
        ll_obs = self.loglike_obs(endog, mu, var_weights, scale)
        return np.sum(ll_obs * freq_weights)

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        See Also
        --------
        statsmodels.genmod.families.family.Family : `resid_anscombe` for the
          individual families for more information

        Notes
        -----
        Anscombe residuals are defined by

        .. math::
           resid\_anscombe_i = \frac{A(y)-A(\mu)}{A'(\mu)\sqrt{Var[\mu]}} *
           \sqrt(var\_weights)

        where :math:`A'(y)=v(y)^{-\frac{1}{3}}` and :math:`v(\mu)` is the
        variance function :math:`Var[y]=\frac{\phi}{w}v(mu)`.
        The transformation :math:`A(y)` makes the residuals more normal
        distributed.
        """
        raise NotImplementedError

    def _clean(self, x):
        """
        Helper function to trim the data so that it is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)


class Poisson(Family):
    """
    Poisson exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Poisson family is the log link. Available
        links are log, identity, and sqrt. See statsmodels.families.links for
        more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Poisson.link : a link instance
        The link function of the Poisson instance.
    Poisson.variance : varfuncs instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.mu

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """
    links = [L.Log, L.Identity, L.Sqrt]
    variance = V.mu
    valid = [0, np.inf]
    safe_links = [L.Log, ]

    def __init__(self, link=None, check_link=True):
        if link is None:
            link = L.Log()
        super(Poisson, self).__init__(
            link=link,
            variance=Poisson.variance,
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Poisson deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\_dev_i = 2 * (endog_i * \ln(endog_i / \mu_i) -
           (endog_i - \mu_i))
        """
        endog_mu = self._clean(endog / mu)
        resid_dev = endog * np.log(endog_mu) - (endog - mu)
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Poisson distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        .. math::
            ll_i = var\_weights_i / scale * (endog_i * \ln(\mu_i) - \mu_i -
            \ln \Gamma(endog_i + 1))
        """
        return var_weights / scale * (endog * np.log(mu) - mu -
                                      special.gammaln(endog + 1))

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals for the Poisson family defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = (3/2) * (endog_i^{2/3} - \mu_i^{2/3}) /
           \mu_i^{1/6} * \sqrt(var\_weights)
        """
        resid = ((3 / 2.) * (endog**(2 / 3.) - mu**(2 / 3.)) /
                 (mu ** (1 / 6.) * scale ** 0.5))
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1., var_weights=1.):
        r"""
        Frozen Poisson distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is ignored.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
            var_weights are ignored for Poisson.

        Returns
        -------
        distribution instance

        """

        return stats.poisson(mu)


class Gaussian(Family):
    """
    Gaussian exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Gaussian.link : a link instance
        The link function of the Gaussian instance
    Gaussian.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.constant

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """

    links = [L.Log, L.Identity, L.InversePower]
    variance = V.constant
    safe_links = links

    def __init__(self, link=None, check_link=True):
        if link is None:
            link = L.Identity()
        super(Gaussian, self).__init__(
            link=link,
            variance=Gaussian.variance,
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Gaussian deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\_dev_i = (endog_i - \mu_i) ** 2
        """
        return (endog - mu) ** 2

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Gaussian distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        If the link is the identity link function then the
        loglikelihood function is the same as the classical OLS model.

        .. math::

           llf = -nobs / 2 * (\log(SSR) + (1 + \log(2 \pi / nobs)))

        where

        .. math::

           SSR = \sum_i (Y_i - g^{-1}(\mu_i))^2

        If the links is not the identity link then the loglikelihood
        function is defined as

        .. math::

           ll_i = -1 / 2 \sum_i  * var\_weights * ((Y_i - mu_i)^2 / scale +
                                                \log(2 * \pi * scale))
        """
        ll_obs = -var_weights * (endog - mu) ** 2 / scale
        ll_obs += -np.log(scale / var_weights) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals for the Gaussian family defined below

        Notes
        -----
        For the Gaussian distribution, Anscombe residuals are the same as
        deviance residuals.

        .. math::

           resid\_anscombe_i = (Y_i - \mu_i) / \sqrt{scale} *
           \sqrt(var\_weights)
        """
        resid = (endog - mu) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.):
        r"""
        Frozen Gaussian distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is required argument for get_distribution.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.

        Returns
        -------
        distribution instance

        """

        scale_n = scale / var_weights
        return stats.norm(loc=mu, scale=np.sqrt(scale_n))


class Gamma(Family):
    """
    Gamma exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gamma family is the inverse link.
        Available links are log, identity, and inverse.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Gamma.link : a link instance
        The link function of the Gamma instance
    Gamma.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.family.varfuncs.mu_squared

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.
    """
    links = [L.Log, L.Identity, L.InversePower]
    variance = V.mu_squared
    safe_links = [L.Log, ]

    def __init__(self, link=None, check_link=True):
        if link is None:
            link = L.InversePower()
        super(Gamma, self).__init__(
            link=link,
            variance=Gamma.variance,
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Gamma deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\_dev_i = 2 * ((endog_i - \mu_i) / \mu_i -
           \log(endog_i / \mu_i))
        """
        endog_mu = self._clean(endog / mu)
        resid_dev = -np.log(endog_mu) + (endog - mu) / mu
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Gamma distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        .. math::

           ll_i = var\_weights_i / scale * (\ln(var\_weights_i * endog_i /
           (scale * \mu_i)) - (var\_weights_i * endog_i) /
           (scale * \mu_i)) - \ln \Gamma(var\_weights_i / scale) - \ln(\mu_i)
        """
        endog_mu = self._clean(endog / mu)
        weight_scale = var_weights / scale
        ll_obs = weight_scale * np.log(weight_scale * endog_mu)
        ll_obs -= weight_scale * endog_mu
        ll_obs -= special.gammaln(weight_scale) + np.log(endog)
        return ll_obs

        # in Stata scale is set to equal 1 for reporting llf
        # in R it's the dispersion, though there is a loss of precision vs.
        # our results due to an assumed difference in implementation

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals for the Gamma family defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = 3 * (endog_i^{1/3} - \mu_i^{1/3}) / \mu_i^{1/3}
           / \sqrt{scale} * \sqrt(var\_weights)
        """
        resid = 3 * (endog**(1/3.) - mu**(1/3.)) / mu**(1/3.) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.):
        r"""
        Frozen Gamma distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is required argument for get_distribution.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.

        Returns
        -------
        distribution instance

        """
        # combine var_weights with scale
        scale_ = scale / var_weights
        shape = 1 / scale_
        scale_g = mu * scale_
        return stats.gamma(shape, scale=scale_g)


class Binomial(Family):
    """
    Binomial exponential family distribution.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Binomial family is the logit link.
        Available links are logit, probit, cauchy, log, loglog, and cloglog.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Binomial.link : a link instance
        The link function of the Binomial instance
    Binomial.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.binary

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    endog for Binomial can be specified in one of three ways:
    A 1d array of 0 or 1 values, indicating failure or success
    respectively.
    A 2d array, with two columns. The first column represents the
    success count and the second column represents the failure
    count.
    A 1d array of proportions, indicating the proportion of
    successes, with parameter `var_weights` containing the
    number of trials for each row.
    """

    links = [L.Logit, L.Probit, L.Cauchy, L.Log, L.LogC, L.CLogLog, L.LogLog,
             L.Identity]
    variance = V.binary  # this is not used below in an effort to include n

    # Other safe links, e.g. cloglog and probit are subclasses
    safe_links = [L.Logit, L.CDFLink]

    def __init__(self, link=None, check_link=True):  # , n=1.):
        if link is None:
            link = L.Logit()
        # TODO: it *should* work for a constant n>1 actually, if freq_weights
        # is equal to n
        self.n = 1
        # overwritten by initialize if needed but always used to initialize
        # variance since endog is assumed/forced to be (0,1)
        super(Binomial, self).__init__(
            link=link,
            variance=V.Binomial(n=self.n),
            check_link=check_link
            )

    def starting_mu(self, y):
        r"""
        The starting values for the IRLS algorithm for the Binomial family.
        A good choice for the binomial family is :math:`\mu_0 = (Y_i + 0.5)/2`
        """
        return (y + .5)/2

    def initialize(self, endog, freq_weights):
        '''
        Initialize the response variable.

        Parameters
        ----------
        endog : ndarray
            Endogenous response variable
        freq_weights : ndarray
            1d array of frequency weights

        Returns
        -------
        If `endog` is binary, returns `endog`

        If `endog` is a 2d array, then the input is assumed to be in the format
        (successes, failures) and
        successes/(success + failures) is returned.  And n is set to
        successes + failures.
        '''
        # if not np.all(np.asarray(freq_weights) == 1):
        #     self.variance = V.Binomial(n=freq_weights)
        if endog.ndim > 1 and endog.shape[1] > 2:
            raise ValueError('endog has more than 2 columns. The Binomial '
                             'link supports either a single response variable '
                             'or a paired response variable.')
        elif endog.ndim > 1 and endog.shape[1] > 1:
            y = endog[:, 0]
            # overwrite self.freq_weights for deviance below
            self.n = endog.sum(1)
            return y*1./self.n, self.n
        else:
            return endog, np.ones(endog.shape[0])

    def _resid_dev(self, endog, mu):
        r"""
        Binomial deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\_dev_i = 2 * n * (endog_i * \ln(endog_i /\mu_i) +
           (1 - endog_i) * \ln((1 - endog_i) / (1 - \mu_i)))
        """
        endog_mu = self._clean(endog / (mu + 1e-20))
        n_endog_mu = self._clean((1. - endog) / (1. - mu + 1e-20))
        resid_dev = endog * np.log(endog_mu) + (1 - endog) * np.log(n_endog_mu)
        return 2 * self.n * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Binomial distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        If the endogenous variable is binary:

        .. math::

         ll_i = \sum_i (y_i * \log(\mu_i/(1-\mu_i)) + \log(1-\mu_i)) *
               var\_weights_i

        If the endogenous variable is binomial:

        .. math::

           ll_i = \sum_i var\_weights_i * (\ln \Gamma(n+1) -
                  \ln \Gamma(y_i + 1) - \ln \Gamma(n_i - y_i +1) + y_i *
                  \log(\mu_i / (n_i - \mu_i)) + n * \log(1 - \mu_i/n_i))

        where :math:`y_i = Y_i * n_i` with :math:`Y_i` and :math:`n_i` as
        defined in Binomial initialize.  This simply makes :math:`y_i` the
        original number of successes.
        """
        n = self.n     # Number of trials
        y = endog * n  # Number of successes

        # note that mu is still in (0,1), i.e. not converted back
        return (
            special.gammaln(n + 1) - special.gammaln(y + 1) -
            special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu + 1e-20)) +
            n * np.log(1 - mu + 1e-20)) * var_weights

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r'''
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals as defined below.

        Notes
        -----
        .. math::

            n^{2/3}*(cox\_snell(endog)-cox\_snell(mu)) /
            (mu*(1-mu/n)*scale^3)^{1/6} * \sqrt(var\_weights)

        where cox_snell is defined as
        cox_snell(x) = betainc(2/3., 2/3., x)*betainc(2/3.,2/3.)
        where betainc is the incomplete beta function as defined in scipy,
        which uses a regularized version (with the unregularized version, one
        would just have :math:`cox_snell(x) = Betainc(2/3., 2/3., x)`).

        The name 'cox_snell' is idiosyncratic and is simply used for
        convenience following the approach suggested in Cox and Snell (1968).
        Further note that
        :math:`cox\_snell(x) = \frac{3}{2}*x^{2/3} *
        hyp2f1(2/3.,1/3.,5/3.,x)`
        where hyp2f1 is the hypergeometric 2f1 function.  The Anscombe
        residuals are sometimes defined in the literature using the
        hyp2f1 formulation.  Both betainc and hyp2f1 can be found in scipy.

        References
        ----------
        Anscombe, FJ. (1953) "Contribution to the discussion of H. Hotelling's
            paper." Journal of the Royal Statistical Society B. 15, 229-30.

        Cox, DR and Snell, EJ. (1968) "A General Definition of Residuals."
            Journal of the Royal Statistical Society B. 30, 248-75.
        '''
        endog = endog * self.n  # convert back to successes
        mu = mu * self.n  # convert back to successes

        def cox_snell(x):
            return special.betainc(2/3., 2/3., x) * special.beta(2/3., 2/3.)

        resid = (self.n ** (2/3.) * (cox_snell(endog * 1. / self.n) -
                                     cox_snell(mu * 1. / self.n)) /
                 (mu * (1 - mu * 1. / self.n) * scale ** 3) ** (1 / 6.))
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1., var_weights=1., n_trials=1):
        r"""
        Frozen Binomial distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is ignored.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
            var_weights are ignored for Poisson.
        n_trials : int
            Number of trials for the binomial distribution. The default is 1
            which corresponds to a Bernoulli random variable.

        Returns
        -------
        distribution instance

        """

        return stats.binom(n=n_trials, p=mu)


class InverseGaussian(Family):
    """
    InverseGaussian exponential family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the inverse Gaussian family is the
        inverse squared link.
        Available links are InverseSquared, Inverse, Log, and Identity.
        See statsmodels.genmod.families.links for more information.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    InverseGaussian.link : a link instance
        The link function of the inverse Gaussian instance
    InverseGaussian.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.mu_cubed

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    The inverse Gaussian distribution is sometimes referred to in the
    literature as the Wald distribution.
    """

    links = [L.InverseSquared, L.InversePower, L.Identity, L.Log]
    variance = V.mu_cubed
    safe_links = [L.InverseSquared, L.Log, ]

    def __init__(self, link=None, check_link=True):
        if link is None:
            link = L.InverseSquared()
        super(InverseGaussian, self).__init__(
            link=link,
            variance=InverseGaussian.variance,
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Inverse Gaussian deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

           resid\_dev_i = 1 / (endog_i * \mu_i^2) * (endog_i - \mu_i)^2
        """
        return 1. / (endog * mu ** 2) * (endog - mu) ** 2

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Inverse Gaussian distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        .. math::

           ll_i = -1/2 * (var\_weights_i * (endog_i - \mu_i)^2 /
           (scale * endog_i * \mu_i^2) + \ln(scale * \endog_i^3 /
           var\_weights_i) - \ln(2 * \pi))
        """
        ll_obs = -var_weights * (endog - mu) ** 2 / (scale * endog * mu ** 2)
        ll_obs += -np.log(scale * endog ** 3 / var_weights) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals for the inverse Gaussian distribution  as
            defined below

        Notes
        -----
        .. math::

           resid\_anscombe_i = \log(Y_i / \mu_i) / \sqrt{\mu_i * scale} *
           \sqrt(var\_weights)
        """
        resid = np.log(endog / mu) / np.sqrt(mu * scale)
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale, var_weights=1.):
        r"""
        Frozen Inverse Gaussian distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is required argument for get_distribution.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.

        Returns
        -------
        distribution instance

        """
        # combine var_weights with scale
        scale_ = scale / var_weights
        mu_ig = mu * scale_
        return stats.invgauss(mu_ig, scale=1 / scale_)


class NegativeBinomial(Family):
    r"""
    Negative Binomial exponential family (corresponds to NB2).

    Parameters
    ----------
    link : a link instance, optional
        The default link for the negative binomial family is the log link.
        Available links are log, cloglog, identity, nbinom and power.
        See statsmodels.genmod.families.links for more information.
    alpha : float, optional
        The ancillary parameter for the negative binomial distribution.
        For now ``alpha`` is assumed to be nonstochastic.  The default value
        is 1.  Permissible values are usually assumed to be between .01 and 2.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    NegativeBinomial.link : a link instance
        The link function of the negative binomial instance
    NegativeBinomial.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.nbinom

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    Power link functions are not yet supported.

    Parameterization for :math:`y=0, 1, 2, \ldots` is

    .. math::

       f(y) = \frac{\Gamma(y+\frac{1}{\alpha})}{y!\Gamma(\frac{1}{\alpha})}
              \left(\frac{1}{1+\alpha\mu}\right)^{\frac{1}{\alpha}}
              \left(\frac{\alpha\mu}{1+\alpha\mu}\right)^y

    with :math:`E[Y]=\mu\,` and :math:`Var[Y]=\mu+\alpha\mu^2`.
    """
    links = [L.Log, L.CLogLog, L.Identity, L.NegativeBinomial, L.Power]
    # TODO: add the ability to use the power links with an if test
    # similar to below
    variance = V.nbinom
    safe_links = [L.Log, ]

    def __init__(self, link=None, alpha=1., check_link=True):
        self.alpha = 1. * alpha  # make it at least float
        if alpha is self.__init__.__defaults__[1]:  # `is` is intentional
            warnings.warn("Negative binomial dispersion parameter alpha not "
                          f"set. Using default value alpha={alpha}.",
                          ValueWarning)
        if link is None:
            link = L.Log()
        super(NegativeBinomial, self).__init__(
            link=link,
            variance=V.NegativeBinomial(alpha=self.alpha),
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Negative Binomial deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        .. math::

            resid_dev_i = 2 * (endog_i * \ln(endog_i /
            \mu_i) - (endog_i + 1 / \alpha) * \ln((endog_i + 1 / \alpha) /
            (\mu_i + 1 / \alpha)))
        """
        endog_mu = self._clean(endog / mu)
        endog_alpha = endog + 1 / self.alpha
        mu_alpha = mu + 1 / self.alpha
        resid_dev = endog * np.log(endog_mu)
        resid_dev -= endog_alpha * np.log(endog_alpha / mu_alpha)
        return 2 * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Negative Binomial distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        Defined as:

        .. math::

           llf = \sum_i var\_weights_i / scale * (Y_i * \log{(\alpha * \mu_i /
                 (1 + \alpha * \mu_i))} - \log{(1 + \alpha * \mu_i)}/
                 \alpha + Constant)

        where :math:`Constant` is defined as:

        .. math::

           Constant = \ln \Gamma{(Y_i + 1/ \alpha )} - \ln \Gamma(Y_i + 1) -
                      \ln \Gamma{(1/ \alpha )}

        constant = (special.gammaln(endog + 1 / self.alpha) -
                    special.gammaln(endog+1)-special.gammaln(1/self.alpha))
        return (endog * np.log(self.alpha * mu / (1 + self.alpha * mu)) -
                np.log(1 + self.alpha * mu) / self.alpha +
                constant) * var_weights / scale
        """
        ll_obs = endog * np.log(self.alpha * mu)
        ll_obs -= (endog + 1 / self.alpha) * np.log(1 + self.alpha * mu)
        ll_obs += special.gammaln(endog + 1 / self.alpha)
        ll_obs -= special.gammaln(1 / self.alpha)
        ll_obs -= special.gammaln(endog + 1)
        return var_weights / scale * ll_obs

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals as defined below.

        Notes
        -----
        Anscombe residuals for Negative Binomial are the same as for Binomial
        upon setting :math:`n=-\frac{1}{\alpha}`. Due to the negative value of
        :math:`-\alpha*Y` the representation with the hypergeometric function
        :math:`H2F1(x) =  hyp2f1(2/3.,1/3.,5/3.,x)` is advantageous

        .. math::

            resid\_anscombe_i = \frac{3}{2} *
            (Y_i^(2/3)*H2F1(-\alpha*Y_i) - \mu_i^(2/3)*H2F1(-\alpha*\mu_i))
            / (\mu_i * (1+\alpha*\mu_i) * scale^3)^(1/6) * \sqrt(var\_weights)

        Note that for the (unregularized) Beta function, one has
        :math:`Beta(z,a,b) = z^a/a * H2F1(a,1-b,a+1,z)`
        """
        def hyp2f1(x):
            return special.hyp2f1(2 / 3., 1 / 3., 5 / 3., x)

        resid = (3 / 2. * (endog ** (2 / 3.) * hyp2f1(-self.alpha * endog) -
                           mu ** (2 / 3.) * hyp2f1(-self.alpha * mu)) /
                 (mu * (1 + self.alpha * mu) *
                 scale ** 3) ** (1 / 6.))
        resid *= np.sqrt(var_weights)
        return resid

    def get_distribution(self, mu, scale=1., var_weights=1.):
        r"""
        Frozen NegativeBinomial distribution instance for given parameters

        Parameters
        ----------
        mu : ndarray
            Usually but not always the fitted mean response variable.
        scale : float
            The scale parameter is ignored.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
            var_weights are ignored for NegativeBinomial.

        Returns
        -------
        distribution instance

        """
        size = 1. / self.alpha
        prob = size / (size + mu)
        return stats.nbinom(size, prob)


class Tweedie(Family):
    """
    Tweedie family.

    Parameters
    ----------
    link : a link instance, optional
        The default link for the Tweedie family is the log link.
        Available links are log, Power and any aliases of power.
        See statsmodels.genmod.families.links for more information.
    var_power : float, optional
        The variance power. The default is 1.
    eql : bool
        If True, the Extended Quasi-Likelihood is used, else the
        likelihood is used.
        In both cases, for likelihood computations the var_power
        must be between 1 and 2.
    check_link : bool
        If True (default), then and exception is raised if the link is invalid
        for the family.
        If False, then the link is not checked.

    Attributes
    ----------
    Tweedie.link : a link instance
        The link function of the Tweedie instance
    Tweedie.variance : varfunc instance
        ``variance`` is an instance of
        statsmodels.genmod.families.varfuncs.Power
    Tweedie.var_power : float
        The power parameter of the variance function.

    See Also
    --------
    statsmodels.genmod.families.family.Family : Parent class for all links.
    :ref:`links` : Further details on links.

    Notes
    -----
    Loglikelihood function not implemented because of the complexity of
    calculating an infinite series of summations. The variance power can be
    estimated using the ``estimate_tweedie_power`` function that is part of the
    statsmodels.genmod.generalized_linear_model.GLM class.
    """
    links = [L.Log, L.Power]
    variance = V.Power(power=1.5)
    safe_links = [L.Log, L.Power]

    def __init__(self, link=None, var_power=1., eql=False, check_link=True):
        self.var_power = var_power
        self.eql = eql
        if eql and (var_power < 1 or var_power > 2):
            raise ValueError("Tweedie: if EQL=True then var_power must fall "
                             "between 1 and 2")
        if link is None:
            link = L.Log()
        super(Tweedie, self).__init__(
            link=link,
            variance=V.Power(power=var_power * 1.),
            check_link=check_link
            )

    def _resid_dev(self, endog, mu):
        r"""
        Tweedie deviance residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable.
        mu : ndarray
            The inverse of the link function at the linear predicted values.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        When :math:`p = 1`,

        .. math::

            dev_i = \mu_i

        when :math:`endog_i = 0` and

        .. math::

            dev_i = endog_i * \log(endog_i / \mu_i) + (\mu_i - endog_i)

        otherwise.

        When :math:`p = 2`,

        .. math::

            dev_i =  (endog_i - \mu_i) / \mu_i - \log(endog_i / \mu_i)

        For all other p,

        .. math::

            dev_i = endog_i^{2 - p} / ((1 - p) * (2 - p)) -
                    endog_i * \mu_i^{1 - p} / (1 - p) + \mu_i^{2 - p} /
                    (2 - p)

        The deviance residual is then

        .. math::

            resid\_dev_i = 2 * dev_i
        """
        p = self.var_power
        if p == 1:
            dev = np.where(endog == 0,
                           mu,
                           endog * np.log(endog / mu) + (mu - endog))
        elif p == 2:
            endog1 = self._clean(endog)
            dev = ((endog - mu) / mu) - np.log(endog1 / mu)
        else:
            dev = (endog ** (2 - p) / ((1 - p) * (2 - p)) -
                   endog * mu ** (1-p) / (1 - p) + mu ** (2 - p) / (2 - p))
        return 2 * dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Tweedie distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        If eql is True, the Extended Quasi-Likelihood is used.  At present,
        this method returns NaN if eql is False.  When the actual likelihood
        is implemented, it will be accessible by setting eql to False.

        References
        ----------
        R Kaas (2005).  Compound Poisson Distributions and GLM's -- Tweedie's
        Distribution.
        https://core.ac.uk/download/pdf/6347266.pdf#page=11

        JA Nelder, D Pregibon (1987).  An extended quasi-likelihood function.
        Biometrika 74:2, pp 221-232.  https://www.jstor.org/stable/2336136
        """
        p = self.var_power
        endog = np.atleast_1d(endog)
        if p == 1:
            return Poisson().loglike_obs(
                endog=endog,
                mu=mu,
                var_weights=var_weights,
                scale=scale
            )
        elif p == 2:
            return Gamma().loglike_obs(
                endog=endog,
                mu=mu,
                var_weights=var_weights,
                scale=scale
            )

        if not self.eql:
            if p < 1 or p > 2:
                # We have not yet implemented the actual likelihood
                return np.nan

            # scipy compat bessel_wright added in 1.7
            if SP_LT_17:
                # old return was nan
                return np.nan

            # See: Dunn, Smyth (2004) "Series evaluation of Tweedie
            # exponential dispersion model densities"
            # pdf(y, mu, p, phi) = f(y, theta, phi)
            # = c(y, phi) * exp(1/phi (y theta - kappa(theta)))
            # kappa = cumulant function
            # theta = function of expectation mu and power p
            # alpha = (2-p)/(1-p)
            # phi = scale
            # for 1<p<2:
            # c(y, phi) = 1/y * wright_bessel(a, b, x)
            # a = -alpha
            # b = 0
            # x = (p-1)**alpha/(2-p) / y**alpha / phi**(1-alpha)
            scale = scale / var_weights
            theta = mu ** (1 - p) / (1 - p)
            kappa = mu ** (2 - p) / (2 - p)
            alpha = (2 - p) / (1 - p)

            ll_obs = (endog * theta - kappa) / scale
            idx = endog > 0
            if np.any(idx):
                if not np.isscalar(endog):
                    endog = endog[idx]
                if not np.isscalar(scale):
                    scale = scale[idx]
                x = ((p - 1) * scale / endog) ** alpha
                x /= (2 - p) * scale
                wb = special.wright_bessel(-alpha, 0, x)
                ll_obs[idx] += np.log(1/endog * wb)
            return ll_obs
        else:
            # Equations 4 of Kaas
            llf = np.log(2 * np.pi * scale) + p * np.log(endog)
            llf -= np.log(var_weights)
            llf /= -2
            u = (endog ** (2 - p)
                 - (2 - p) * endog * mu ** (1 - p)
                 + (1 - p) * mu ** (2 - p))
            u *= var_weights / (scale * (1 - p) * (2 - p))

        return llf - u

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        r"""
        The Anscombe residuals

        Parameters
        ----------
        endog : ndarray
            The endogenous response variable
        mu : ndarray
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional argument to divide the residuals by sqrt(scale).
            The default is 1.

        Returns
        -------
        resid_anscombe : ndarray
            The Anscombe residuals as defined below.

        Notes
        -----
        When :math:`p = 3`, then

        .. math::

            resid\_anscombe_i = \log(endog_i / \mu_i) / \sqrt{\mu_i * scale} *
            \sqrt(var\_weights)

        Otherwise,

        .. math::

            c = (3 - p) / 3

        .. math::

            resid\_anscombe_i = (1 / c) * (endog_i^c - \mu_i^c) / \mu_i^{p / 6}
            / \sqrt{scale} * \sqrt(var\_weights)
        """
        if self.var_power == 3:
            resid = np.log(endog / mu) / np.sqrt(mu * scale)
        else:
            c = (3. - self.var_power) / 3.
            resid = ((1. / c) * (endog ** c - mu ** c) /
                     mu ** (self.var_power / 6.)) / scale ** 0.5
        resid *= np.sqrt(var_weights)
        return resid
