import numpy as np

from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere

from statsmodels.base.model import GenericLikelihoodModel


class genpoisson_p_gen(rv_discrete):
    '''Generalized Poisson distribution
    '''
    def _argcheck(self, mu, alpha, p):
        return (mu >= 0) & (alpha==alpha) & (p > 0)

    def _logpmf(self, x, mu, alpha, p):
        mu_p = mu ** (p - 1.)
        a1 = np.maximum(np.nextafter(0, 1), 1 + alpha * mu_p)
        a2 = np.maximum(np.nextafter(0, 1), mu + (a1 - 1.) * x)
        logpmf_ = np.log(mu) + (x - 1.) * np.log(a2)
        logpmf_ -=  x * np.log(a1) + gammaln(x + 1.) + a2 / a1
        return logpmf_

    def _pmf(self, x, mu, alpha, p):
        return np.exp(self._logpmf(x, mu, alpha, p))

    def mean(self, mu, alpha, p):
        return mu

    def var(self, mu, alpha, p):
        dispersion_factor = (1 + alpha * mu**(p - 1))**2
        var = dispersion_factor * mu
        return var


genpoisson_p = genpoisson_p_gen(name='genpoisson_p',
                                longname='Generalized Poisson')


class zipoisson_gen(rv_discrete):
    '''Zero Inflated Poisson distribution
    '''
    def _argcheck(self, mu, w):
        return (mu > 0) & (w >= 0) & (w<=1)

    def _logpmf(self, x, mu, w):
        return _lazywhere(x != 0, (x, mu, w),
                          (lambda x, mu, w: np.log(1. - w) + x * np.log(mu) -
                          gammaln(x + 1.) - mu),
                          np.log(w + (1. - w) * np.exp(-mu)))

    def _pmf(self, x, mu, w):
        return np.exp(self._logpmf(x, mu, w))

    def _cdf(self, x, mu, w):
        # construct cdf from standard poisson's cdf and the w inflation of zero
        return w + poisson(mu=mu).cdf(x) * (1 - w)

    def _ppf(self, q, mu, w):
        # we just translated and stretched q to remove zi
        q_mod = (q - w) / (1 - w)
        x = poisson(mu=mu).ppf(q_mod)
        # set to zero if in the zi range
        x[q < w] = 0
        return x

    def mean(self, mu, w):
        return (1 - w) * mu

    def var(self, mu, w):
        dispersion_factor = 1 + w * mu
        var = (dispersion_factor * self.mean(mu, w))
        return var

    def _moment(self, n, mu, w):
        return (1 - w) * poisson.moment(n, mu)


zipoisson = zipoisson_gen(name='zipoisson',
                          longname='Zero Inflated Poisson')

class zigeneralizedpoisson_gen(rv_discrete):
    '''Zero Inflated Generalized Poisson distribution
    '''
    def _argcheck(self, mu, alpha, p, w):
        return (mu > 0) & (w >= 0) & (w<=1)

    def _logpmf(self, x, mu, alpha, p, w):
        return _lazywhere(x != 0, (x, mu, alpha, p, w),
                          (lambda x, mu, alpha, p, w: np.log(1. - w) +
                          genpoisson_p.logpmf(x, mu, alpha, p)),
                          np.log(w + (1. - w) *
                          genpoisson_p.pmf(x, mu, alpha, p)))

    def _pmf(self, x, mu, alpha, p, w):
        return np.exp(self._logpmf(x, mu, alpha, p, w))

    def mean(self, mu, alpha, p, w):
        return (1 - w) * mu

    def var(self, mu, alpha, p, w):
        p = p - 1
        dispersion_factor = (1 + alpha * mu ** p) ** 2 + w * mu
        var = (dispersion_factor * self.mean(mu, alpha, p, w))
        return var


zigenpoisson = zigeneralizedpoisson_gen(
    name='zigenpoisson',
    longname='Zero Inflated Generalized Poisson')


class zinegativebinomial_gen(rv_discrete):
    '''Zero Inflated Generalized Negative Binomial distribution
    '''
    def _argcheck(self, mu, alpha, p, w):
        return (mu > 0) & (w >= 0) & (w<=1)

    def _logpmf(self, x, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        return _lazywhere(x != 0, (x, s, p, w),
                          (lambda x, s, p, w: np.log(1. - w) +
                          nbinom.logpmf(x, s, p)),
                          np.log(w + (1. - w) *
                          nbinom.pmf(x, s, p)))

    def _pmf(self, x, mu, alpha, p, w):
        return np.exp(self._logpmf(x, mu, alpha, p, w))

    def _cdf(self, x, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        # construct cdf from standard negative binomial cdf
        # and the w inflation of zero
        return w + nbinom.cdf(x, s, p) * (1 - w)

    def _ppf(self, q, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        # we just translated and stretched q to remove zi
        q_mod = (q - w) / (1 - w)
        x = nbinom.ppf(q_mod, s, p)
        # set to zero if in the zi range
        x[q < w] = 0
        return x

    def mean(self, mu, alpha, p, w):
        return (1 - w) * mu

    def var(self, mu, alpha, p, w):
        dispersion_factor = 1 + alpha * mu ** (p - 1) + w * mu
        var = (dispersion_factor * self.mean(mu, alpha, p, w))
        return var

    def _moment(self, n, mu, alpha, p, w):
        s, p = self.convert_params(mu, alpha, p)
        return (1 - w) * nbinom.moment(n, s, p)

    def convert_params(self, mu, alpha, p):
        size = 1. / alpha * mu**(2-p)
        prob = size / (size + mu)
        return (size, prob)

zinegbin = zinegativebinomial_gen(name='zinegbin',
    longname='Zero Inflated Generalized Negative Binomial')


class truncatedpoisson_gen(rv_discrete):
    '''Truncated Poisson discrete random variable
    '''
    # TODO: need cdf, and rvs

    def _argcheck(self, mu, truncation):
        # this does not work
        # vector bound breaks some generic methods
        # self.a = truncation + 1 # max(truncation + 1, 0)
        return (mu >= 0) & (truncation >= -1)

    def _get_support(self, mu, truncation):
        return truncation + 1, self.b

    def _logpmf(self, x, mu, truncation):
        pmf = 0
        for i in range(int(np.max(truncation)) + 1):
            pmf += poisson.pmf(i, mu)

        logpmf_ = poisson.logpmf(x, mu) - np.log(1 - pmf)
        #logpmf_[x < truncation + 1] = - np.inf
        return logpmf_

    def _pmf(self, x, mu, truncation):
        return np.exp(self._logpmf(x, mu, truncation))

truncatedpoisson = truncatedpoisson_gen(name='truncatedpoisson',
                                        longname='Truncated Poisson')

class truncatednegbin_gen(rv_discrete):
    '''Truncated Generalized Negative Binomial (NB-P) discrete random variable
    '''
    def _argcheck(self, mu, alpha, p, truncation):
        return (mu >= 0) & (truncation >= -1)

    def _get_support(self, mu, alpha, p, truncation):
        return truncation + 1, self.b

    def _logpmf(self, x, mu, alpha, p, truncation):
        size, prob = self.convert_params(mu, alpha, p)
        pmf = 0
        for i in range(int(np.max(truncation)) + 1):
            pmf += nbinom.pmf(i, size, prob)

        logpmf_ = nbinom.logpmf(x, size, prob) - np.log(1 - pmf)
        # logpmf_[x < truncation + 1] = - np.inf
        return logpmf_

    def _pmf(self, x, mu, alpha, p, truncation):
        return np.exp(self._logpmf(x, mu, alpha, p, truncation))

    def convert_params(self, mu, alpha, p):
        size = 1. / alpha * mu**(2-p)
        prob = size / (size + mu)
        return (size, prob)

truncatednegbin = truncatednegbin_gen(name='truncatednegbin',
    longname='Truncated Generalized Negative Binomial')

class DiscretizedCount(rv_discrete):
    """Count distribution based on discretized distribution

    Parameters
    ----------
    distr : distribution instance
    d_offset : float
        Offset for integer interval, default is zero.
        The discrete random variable is ``y = floor(x + offset)`` where x is
        the continuous random variable.
        Warning: not verified for all methods.
    add_scale : bool
        If True (default), then the scale of the base distribution is added
        as parameter for the discrete distribution. The scale parameter is in
        the last position.
    kwds : keyword arguments
        The extra keyword arguments are used delegated to the ``__init__`` of
        the super class.
        Their usage has not been checked, e.g. currently the support of the
        distribution is assumed to be all non-negative integers.

    Notes
    -----
    `loc` argument is currently not supported, scale is not available for
    discrete distributions in scipy. The scale parameter of the underlying
    continuous distribution is the last shape parameter in this
    DiscretizedCount distribution if ``add_scale`` is True.

    The implementation was based mainly on [1]_ and [2]_. However, many new
    discrete distributions have been developed based on the approach that we
    use here. Note, that in many cases authors reparameterize the distribution,
    while this class inherits the parameterization from the underlying
    continuous distribution.

    References
    ----------
    .. [1] Chakraborty, Subrata, and Dhrubajyoti Chakravarty. "Discrete gamma
       distributions: Properties and parameter estimations." Communications in
       Statistics-Theory and Methods 41, no. 18 (2012): 3301-3324.

    .. [2] Alzaatreh, Ayman, Carl Lee, and Felix Famoye. 2012. “On the Discrete
       Analogues of Continuous Distributions.” Statistical Methodology 9 (6):
       589–603.


    """

    def __new__(cls, *args, **kwds):
        # rv_discrete.__new__ does not allow `kwds`, skip it
        # only does dispatch to multinomial
        return super(rv_discrete, cls).__new__(cls)

    def __init__(self, distr, d_offset=0, add_scale=True, **kwds):
        # kwds are extras in rv_discrete
        self.distr = distr
        self.d_offset = d_offset
        self._ctor_param = distr._ctor_param
        self.add_scale = add_scale
        if distr.shapes is not None:
            self.k_shapes = len(distr.shapes.split(","))
            if add_scale:
                kwds.update({"shapes": distr.shapes + ", s"})
                self.k_shapes += 1
        else:
            # no shape parameters in underlying distribution
            if add_scale:
                kwds.update({"shapes": "s"})
                self.k_shapes = 1
            else:
                self.k_shapes = 0

        super().__init__(**kwds)

    def _updated_ctor_param(self):
        dic = super()._updated_ctor_param()
        dic["distr"] = self.distr
        return dic

    def _unpack_args(self, args):
        if self.add_scale:
            scale = args[-1]
            args = args[:-1]
        else:
            scale = 1
        return args, scale

    def _rvs(self, *args, size=None, random_state=None):
        args, scale = self._unpack_args(args)
        if size is None:
            size = getattr(self, "_size", 1)
        rv = np.trunc(self.distr.rvs(*args, scale=scale, size=size,
                                     random_state=random_state) +
                      self.d_offset)
        return rv

    def _pmf(self, x, *args):
        distr = self.distr
        if self.d_offset != 0:
            x = x + self.d_offset

        args, scale = self._unpack_args(args)

        p = (distr.sf(x, *args, scale=scale) -
             distr.sf(x + 1, *args, scale=scale))
        return p

    def _cdf(self, x, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        if self.d_offset != 0:
            x = x + self.d_offset
        p = distr.cdf(x + 1, *args, scale=scale)
        return p

    def _sf(self, x, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        if self.d_offset != 0:
            x = x + self.d_offset
        p = distr.sf(x + 1, *args, scale=scale)
        return p

    def _ppf(self, p, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)

        qc = distr.ppf(p, *args, scale=scale)
        if self.d_offset != 0:
            qc = qc + self.d_offset
        q = np.floor(qc * (1 - 1e-15))
        return q

    def _isf(self, p, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)

        qc = distr.isf(p, *args, scale=scale)
        if self.d_offset != 0:
            qc = qc + self.d_offset
        q = np.floor(qc * (1 - 1e-15))
        return q


class DiscretizedModel(GenericLikelihoodModel):
    """experimental model to fit discretized distribution

    Count models based on discretized distributions can be used to model
    data that is under- or over-dispersed relative to Poisson or that has
    heavier tails.

    Parameters
    ----------
    endog : array_like, 1-D
        Univariate data for fitting the distribution.
    exog : None
        Explanatory variables are not supported. The ``exog`` argument is
        only included for consistency in the signature across models.
    distr : DiscretizedCount instance
        (required) Instance of a DiscretizedCount distribution.

    See Also
    --------
    DiscretizedCount

    Examples
    --------
    >>> from scipy import stats
    >>> from statsmodels.distributions.discrete import (
            DiscretizedCount, DiscretizedModel)

    >>> dd = DiscretizedCount(stats.gamma)
    >>> mod = DiscretizedModel(y, distr=dd)
    >>> res = mod.fit()
    >>> probs = res.predict(which="probs", k_max=5)

    """
    def __init__(self, endog, exog=None, distr=None):
        if exog is not None:
            raise ValueError("exog is not supported")

        super().__init__(endog, exog, distr=distr)
        self._init_keys.append('distr')
        self.df_resid = len(endog) - distr.k_shapes
        self.df_model = 0
        self.k_extra = distr.k_shapes  # no constant subtracted
        self.k_constant = 0
        self.nparams = distr.k_shapes  # needed for start_params
        self.start_params = 0.5 * np.ones(self.nparams)

    def loglike(self, params):

        # this does not allow exog yet,
        # model `params` are also distribution `args`
        # For regression model this needs to be replaced by a conversion method
        args = params
        ll = np.log(self.distr._pmf(self.endog, *args))
        return ll.sum()

    def predict(self, params, exog=None, which=None, k_max=20):

        if exog is not None:
            raise ValueError("exog is not supported")

        args = params
        if which == "probs":
            pr = self.distr.pmf(np.arange(k_max), *args)
            return pr
        else:
            raise ValueError('only which="probs" is currently implemented')

    def get_distr(self, params):
        """frozen distribution instance of the discrete distribution.
        """
        args = params
        distr = self.distr(*args)
        return distr
