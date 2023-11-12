# -*- coding: utf-8 -*-
"""Multivariate Normal and t distributions



Created on Sat May 28 15:38:23 2011

@author: Josef Perktold

TODO:
* renaming,
    - after adding t distribution, cov does not make sense for Sigma    DONE
    - should mean also be renamed to mu, if there will be distributions
      with mean != mu
* not sure about corner cases
    - behavior with (almost) singular sigma or transforms
    - df <= 2, is everything correct if variance is not finite or defined ?
* check to return possibly univariate distribution for marginals or conditional
    distributions, does univariate special case work? seems ok for conditional
* are all the extra transformation methods useful outside of testing ?
  - looks like I have some mixup in definitions of standardize, normalize
* new methods marginal, conditional, ... just added, typos ?
  - largely tested for MVNormal, not yet for MVT   DONE
* conditional: reusing, vectorizing, should we reuse a projection matrix or
  allow for a vectorized, conditional_mean similar to OLS.predict
* add additional things similar to LikelihoodModelResults? quadratic forms,
  F distribution, and others ???
* add Delta method for nonlinear functions here, current function is hidden
  somewhere in miscmodels
* raise ValueErrors for wrong input shapes, currently only partially checked

* quantile method (ppf for equal bounds for multiple testing) is missing
  http://svitsrv25.epfl.ch/R-doc/library/mvtnorm/html/qmvt.html seems to use
  just a root finder for inversion of cdf

* normalize has ambiguous definition, and mixing it up in different versions
  std from sigma or std from cov ?
  I would like to get what I need for mvt-cdf, or not
  univariate standard t distribution has scale=1 but std>1
  FIXED: add std_sigma, and normalize uses std_sigma

* more work: bivariate distributions,
  inherit from multivariate but overwrite some methods for better efficiency,
  e.g. cdf and expect

I kept the original MVNormal0 class as reference, can be deleted


See Also
--------
sandbox/examples/ex_mvelliptical.py

Examples
--------

Note, several parts of these examples are random and the numbers will not be
(exactly) the same.

>>> import numpy as np
>>> import statsmodels.sandbox.distributions.mv_normal as mvd
>>>
>>> from numpy.testing import assert_array_almost_equal
>>>
>>> cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
...                    [ 0.5 ,  1.5 ,  0.6 ],
...                    [ 0.75,  0.6 ,  2.  ]])

>>> mu = np.array([-1, 0.0, 2.0])

multivariate normal distribution
--------------------------------

>>> mvn3 = mvd.MVNormal(mu, cov3)
>>> mvn3.rvs(size=3)
array([[-0.08559948, -1.0319881 ,  1.76073533],
       [ 0.30079522,  0.55859618,  4.16538667],
       [-1.36540091, -1.50152847,  3.87571161]])

>>> mvn3.std
array([ 1.        ,  1.22474487,  1.41421356])
>>> a = [0.0, 1.0, 1.5]
>>> mvn3.pdf(a)
0.013867410439318712
>>> mvn3.cdf(a)
0.31163181123730122

Monte Carlo integration

>>> mvn3.expect_mc(lambda x: (x<a).all(-1), size=100000)
0.30958999999999998
>>> mvn3.expect_mc(lambda x: (x<a).all(-1), size=1000000)
0.31197399999999997

multivariate t distribution
---------------------------

>>> mvt3 = mvd.MVT(mu, cov3, 4)
>>> mvt3.rvs(size=4)
array([[-0.94185437,  0.3933273 ,  2.40005487],
       [ 0.07563648,  0.06655433,  7.90752238],
       [ 1.06596474,  0.32701158,  2.03482886],
       [ 3.80529746,  7.0192967 ,  8.41899229]])

>>> mvt3.pdf(a)
0.010402959362646937
>>> mvt3.cdf(a)
0.30269483623249821
>>> mvt3.expect_mc(lambda x: (x<a).all(-1), size=1000000)
0.30271199999999998

>>> mvt3.cov
array([[ 2. ,  1. ,  1.5],
       [ 1. ,  3. ,  1.2],
       [ 1.5,  1.2,  4. ]])
>>> mvt3.corr
array([[ 1.        ,  0.40824829,  0.53033009],
       [ 0.40824829,  1.        ,  0.34641016],
       [ 0.53033009,  0.34641016,  1.        ]])

get normalized distribution

>>> mvt3n = mvt3.normalized()
>>> mvt3n.sigma
array([[ 1.        ,  0.40824829,  0.53033009],
       [ 0.40824829,  1.        ,  0.34641016],
       [ 0.53033009,  0.34641016,  1.        ]])
>>> mvt3n.cov
array([[ 2.        ,  0.81649658,  1.06066017],
       [ 0.81649658,  2.        ,  0.69282032],
       [ 1.06066017,  0.69282032,  2.        ]])

What's currently there?

>>> [i for i in dir(mvn3) if not i[0]=='_']
['affine_transformed', 'cdf', 'cholsigmainv', 'conditional', 'corr', 'cov',
'expect_mc', 'extra_args', 'logdetsigma', 'logpdf', 'marginal', 'mean',
'normalize', 'normalized', 'normalized2', 'nvars', 'pdf', 'rvs', 'sigma',
'sigmainv', 'standardize', 'standardized', 'std', 'std_sigma', 'whiten']

>>> [i for i in dir(mvt3) if not i[0]=='_']
['affine_transformed', 'cdf', 'cholsigmainv', 'corr', 'cov', 'df', 'expect_mc',
'extra_args', 'logdetsigma', 'logpdf', 'marginal', 'mean', 'normalize',
'normalized', 'normalized2', 'nvars', 'pdf', 'rvs', 'sigma', 'sigmainv',
'standardize', 'standardized', 'std', 'std_sigma', 'whiten']

"""
import numpy as np
from scipy import special

from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf


def expect_mc(dist, func=lambda x: 1, size=50000):
    '''calculate expected value of function by Monte Carlo integration

    Parameters
    ----------
    dist : distribution instance
        needs to have rvs defined as a method for drawing random numbers
    func : callable
        function for which expectation is calculated, this function needs to
        be vectorized, integration is over axis=0
    size : int
        number of random samples to use in the Monte Carlo integration,


    Notes
    -----
    this does not batch

    Returns
    -------
    expected value : ndarray
        return of function func integrated over axis=0 by MonteCarlo, this will
        have the same shape as the return of func without axis=0

    Examples
    --------

    integrate probability that both observations are negative

    >>> mvn = mve.MVNormal([0,0],2.)
    >>> mve.expect_mc(mvn, lambda x: (x<np.array([0,0])).all(-1), size=100000)
    0.25306000000000001

    get tail probabilities of marginal distribution (should be 0.1)

    >>> c = stats.norm.isf(0.05, scale=np.sqrt(2.))
    >>> expect_mc(mvn, lambda x: (np.abs(x)>np.array([c, c])), size=100000)
    array([ 0.09969,  0.0986 ])

    or calling the method

    >>> mvn.expect_mc(lambda x: (np.abs(x)>np.array([c, c])), size=100000)
    array([ 0.09937,  0.10075])


    '''
    def fun(x):
        return func(x) # * dist.pdf(x)
    rvs = dist.rvs(size=size)
    return fun(rvs).mean(0)

def expect_mc_bounds(dist, func=lambda x: 1, size=50000, lower=None, upper=None,
                     conditional=False, overfact=1.2):
    '''calculate expected value of function by Monte Carlo integration

    Parameters
    ----------
    dist : distribution instance
        needs to have rvs defined as a method for drawing random numbers
    func : callable
        function for which expectation is calculated, this function needs to
        be vectorized, integration is over axis=0
    size : int
        minimum number of random samples to use in the Monte Carlo integration,
        the actual number used can be larger because of oversampling.
    lower : None or array_like
        lower integration bounds, if None, then it is set to -inf
    upper : None or array_like
        upper integration bounds, if None, then it is set to +inf
    conditional : bool
        If true, then the expectation is conditional on being in within
        [lower, upper] bounds, otherwise it is unconditional
    overfact : float
        oversampling factor, the actual number of random variables drawn in
        each attempt are overfact * remaining draws. Extra draws are also
        used in the integration.


    Notes
    -----
    this does not batch

    Returns
    -------
    expected value : ndarray
        return of function func integrated over axis=0 by MonteCarlo, this will
        have the same shape as the return of func without axis=0

    Examples
    --------
    >>> mvn = mve.MVNormal([0,0],2.)
    >>> mve.expect_mc_bounds(mvn, lambda x: np.ones(x.shape[0]),
                                lower=[-10,-10],upper=[0,0])
    0.24990416666666668

    get 3 marginal moments with one integration

    >>> mvn = mve.MVNormal([0,0],1.)
    >>> mve.expect_mc_bounds(mvn, lambda x: np.dstack([x, x**2, x**3, x**4]),
        lower=[-np.inf,-np.inf], upper=[np.inf,np.inf])
    array([[  2.88629497e-03,   9.96706297e-01,  -2.51005344e-03,
              2.95240921e+00],
           [ -5.48020088e-03,   9.96004409e-01,  -2.23803072e-02,
              2.96289203e+00]])
    >>> from scipy import stats
    >>> [stats.norm.moment(i) for i in [1,2,3,4]]
    [0.0, 1.0, 0.0, 3.0]


    '''
    #call rvs once to find length of random vector
    rvsdim = dist.rvs(size=1).shape[-1]
    if lower is None:
        lower = -np.inf * np.ones(rvsdim)
    else:
        lower = np.asarray(lower)
    if upper is None:
        upper = np.inf * np.ones(rvsdim)
    else:
        upper = np.asarray(upper)

    def fun(x):
        return func(x) # * dist.pdf(x)

    rvsli = []
    used = 0 #remain = size  #inplace changes size
    total = 0
    while True:
        remain = size - used  #just a temp variable
        rvs = dist.rvs(size=int(remain * overfact))
        total += int(size * overfact)

        rvsok = rvs[((rvs >= lower) & (rvs <= upper)).all(-1)]
        #if rvsok.ndim == 1: #possible shape problems if only 1 random vector
        rvsok = np.atleast_2d(rvsok)
        used += rvsok.shape[0]

        rvsli.append(rvsok)   #[:remain]) use extras instead
        print(used)
        if used >= size:
            break
    rvs = np.vstack(rvsli)
    print(rvs.shape)
    assert used == rvs.shape[0] #saftey check
    mean_conditional = fun(rvs).mean(0)
    if conditional:
        return mean_conditional
    else:
        return mean_conditional * (used * 1. / total)


def bivariate_normal(x, mu, cov):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    X, Y = np.transpose(x)
    mux, muy = mu
    sigmax, sigmaxy, tmp, sigmay = np.ravel(cov)
    sigmax, sigmay = np.sqrt(sigmax), np.sqrt(sigmay)
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp( -z/(2*(1-rho**2))) / denom



class BivariateNormal:


    #TODO: make integration limits more flexible
    #      or normalize before integration

    def __init__(self, mean, cov):
        self.mean = mu
        self.cov = cov
        self.sigmax, self.sigmaxy, tmp, self.sigmay = np.ravel(cov)
        self.nvars = 2

    def rvs(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        return bivariate_normal(x, self.mean, self.cov)

    def logpdf(self, x):
        #TODO: replace this
        return np.log(self.pdf(x))

    def cdf(self, x):
        return self.expect(upper=x)

    def expect(self, func=lambda x: 1, lower=(-10,-10), upper=(10,10)):
        def fun(x, y):
            x = np.column_stack((x,y))
            return func(x) * self.pdf(x)
        from scipy.integrate import dblquad
        return dblquad(fun, lower[0], upper[0], lambda y: lower[1],
                       lambda y: upper[1])

    def kl(self, other):
        '''Kullback-Leibler divergence between this and another distribution

        int f(x) (log f(x) - log g(x)) dx

        where f is the pdf of self, and g is the pdf of other

        uses double integration with scipy.integrate.dblquad

        limits currently hardcoded

        '''
        fun = lambda x : self.logpdf(x) - other.logpdf(x)
        return self.expect(fun)

    def kl_mc(self, other, size=500000):
        fun = lambda x : self.logpdf(x) - other.logpdf(x)
        rvs = self.rvs(size=size)
        return fun(rvs).mean()

class MVElliptical:
    '''Base Class for multivariate elliptical distributions, normal and t

    contains common initialization, and some common methods
    subclass needs to implement at least rvs and logpdf methods

    '''
    #getting common things between normal and t distribution


    def __init__(self, mean, sigma, *args, **kwds):
        '''initialize instance

        Parameters
        ----------
        mean : array_like
            parameter mu (might be renamed), for symmetric distributions this
            is the mean
        sigma : array_like, 2d
            dispersion matrix, covariance matrix in normal distribution, but
            only proportional to covariance matrix in t distribution
        args : list
            distribution specific arguments, e.g. df for t distribution
        kwds : dict
            currently not used

        '''

        self.extra_args = []
        self.mean = np.asarray(mean)
        self.sigma = sigma = np.asarray(sigma)
        sigma = np.squeeze(sigma)
        self.nvars = nvars = len(mean)
        #self.covchol = np.linalg.cholesky(sigma)


        #in the following sigma is original, self.sigma is full matrix
        if sigma.shape == ():
            #iid
            self.sigma = np.eye(nvars) * sigma
            self.sigmainv = np.eye(nvars) / sigma
            self.cholsigmainv = np.eye(nvars) / np.sqrt(sigma)
        elif (sigma.ndim == 1) and (len(sigma) == nvars):
            #independent heteroskedastic
            self.sigma = np.diag(sigma)
            self.sigmainv = np.diag(1. / sigma)
            self.cholsigmainv = np.diag( 1. / np.sqrt(sigma))
        elif sigma.shape == (nvars, nvars): #python tuple comparison
            #general
            self.sigmainv = np.linalg.pinv(sigma)
            self.cholsigmainv = np.linalg.cholesky(self.sigmainv).T
        else:
            raise ValueError('sigma has invalid shape')

        #store logdetsigma for logpdf
        self.logdetsigma = np.log(np.linalg.det(self.sigma))

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension


        '''
        raise NotImplementedError

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        does not work now because of dot in whiten

        '''


        raise NotImplementedError

    def cdf(self, x, **kwds):
        '''cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        '''
        raise NotImplementedError


    def affine_transformed(self, shift, scale_matrix):
        '''affine transformation define in subclass because of distribution
        specific restrictions'''
        #implemented in subclass at least for now
        raise NotImplementedError

    def whiten(self, x):
        """
        whiten the data by linear transformation

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholsigmainv.T)

        Notes
        -----
        This only does rescaling, it does not subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.
        """
        x = np.asarray(x)
        return np.dot(x, self.cholsigmainv.T)

    def pdf(self, x):
        '''probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        '''
        return np.exp(self.logpdf(x))

    def standardize(self, x):
        '''standardize the random variable, i.e. subtract mean and whiten

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x - self.mean, self.cholsigmainv.T)

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        '''
        return self.whiten(x - self.mean)

    def standardized(self):
        '''return new standardized MVNormal instance
        '''
        return self.affine_transformed(-self.mean, self.cholsigmainv)


    def normalize(self, x):
        '''normalize the random variable, i.e. subtract mean and rescale

        The distribution will have zero mean and sigma equal to correlation

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        (x - self.mean)/std_sigma

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        '''
        std_ = np.atleast_2d(self.std_sigma)
        return (x - self.mean)/std_ #/std_.T

    def normalized(self, demeaned=True):
        '''return a normalized distribution where sigma=corr

        if demeaned is True, then mean will be set to zero

        '''
        if demeaned:
            mean_new = np.zeros_like(self.mean)
        else:
            mean_new = self.mean / self.std_sigma
        sigma_new = self.corr
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

    def normalized2(self, demeaned=True):
        '''return a normalized distribution where sigma=corr



        second implementation for testing affine transformation
        '''
        if demeaned:
            shift = -self.mean
        else:
            shift = self.mean * (1. / self.std_sigma - 1.)
        return self.affine_transformed(shift, np.diag(1. / self.std_sigma))
        #the following "standardizes" cov instead
        #return self.affine_transformed(shift, self.cholsigmainv)



    @property
    def std(self):
        '''standard deviation, square root of diagonal elements of cov
        '''
        return np.sqrt(np.diag(self.cov))

    @property
    def std_sigma(self):
        '''standard deviation, square root of diagonal elements of sigma
        '''
        return np.sqrt(np.diag(self.sigma))


    @property
    def corr(self):
        '''correlation matrix'''
        return self.cov / np.outer(self.std, self.std)

    expect_mc = expect_mc

    def marginal(self, indices):
        '''return marginal distribution for variables given by indices

        this should be correct for normal and t distribution

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution

        Returns
        -------
        mvdist : instance
            new instance of the same multivariate distribution class that
            contains the marginal distribution of the variables given in
            indices

        '''
        indices = np.asarray(indices)
        mean_new = self.mean[indices]
        sigma_new = self.sigma[indices[:,None], indices]
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)


#parts taken from linear_model, but heavy adjustments
class MVNormal0:
    '''Class for Multivariate Normal Distribution

    original full version, kept for testing, new version inherits from
    MVElliptical

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    '''


    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov = np.asarray(cov)
        cov = np.squeeze(cov)
        self.nvars = nvars = len(mean)


        #in the following cov is original, self.cov is full matrix
        if cov.shape == ():
            #iid
            self.cov = np.eye(nvars) * cov
            self.covinv = np.eye(nvars) / cov
            self.cholcovinv = np.eye(nvars) / np.sqrt(cov)
        elif (cov.ndim == 1) and (len(cov) == nvars):
            #independent heteroskedastic
            self.cov = np.diag(cov)
            self.covinv = np.diag(1. / cov)
            self.cholcovinv = np.diag( 1. / np.sqrt(cov))
        elif cov.shape == (nvars, nvars): #python tuple comparison
            #general
            self.covinv = np.linalg.pinv(cov)
            self.cholcovinv = np.linalg.cholesky(self.covinv).T
        else:
            raise ValueError('cov has invalid shape')

        #store logdetcov for logpdf
        self.logdetcov = np.log(np.linalg.det(self.cov))

    def whiten(self, x):
        """
        whiten the data by linear transformation

        Parameters
        ----------
        X : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholcovinv.T)

        Notes
        -----
        This only does rescaling, it does not subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.
        """
        x = np.asarray(x)
        if np.any(self.cov):
            #return np.dot(self.cholcovinv, x)
            return np.dot(x, self.cholcovinv.T)
        else:
            return x

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        '''probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        '''

        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        does not work now because of dot in whiten

        '''
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened**2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2. * np.pi)
        llf -= self.logdetcov
        llf *= 0.5
        return llf

    expect_mc = expect_mc


class MVNormal(MVElliptical):
    '''Class for Multivariate Normal Distribution

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    '''
    __name__ == 'Multivariate Normal Distribution'


    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        return np.random.multivariate_normal(self.mean, self.sigma, size=size)

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        does not work now because of dot in whiten

        '''
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened**2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2. * np.pi)
        llf -= self.logdetsigma
        llf *= 0.5
        return llf

    def cdf(self, x, **kwds):
        '''cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        '''
        #lower = -np.inf * np.ones_like(x)
        #return mvstdnormcdf(lower, self.standardize(x), self.corr, **kwds)
        return mvnormcdf(x, self.mean, self.cov, **kwds)

    @property
    def cov(self):
        '''covariance matrix'''
        return self.sigma

    def affine_transformed(self, shift, scale_matrix):
        '''return distribution of an affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVNormal
            instance of multivariate normal distribution given by affine
            transformation

        Notes
        -----
        the affine transformation is defined by
        y = a + B x

        where a is shift,
        B is a scale matrix for the linear transformation

        Notes
        -----
        This should also work to select marginal distributions, but not
        tested for this case yet.

        currently only tested because it's called by standardized

        '''
        B = scale_matrix  #tmp variable
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVNormal(mean_new, sigma_new)

    def conditional(self, indices, values):
        r'''return conditional distribution

        indices are the variables to keep, the complement is the conditioning
        set
        values are the values of the conditioning variables

        \bar{\mu} = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} \left( a - \mu_2 \right)

        and covariance matrix

        \overline{\Sigma} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}.T

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution
        given : array_like
            values of the conditioning variables

        Returns
        -------
        mvn : instance of MVNormal
            new instance of the MVNormal class that contains the conditional
            distribution of the variables given in indices for given
             values of the excluded variables.


        '''
        #indices need to be nd arrays for broadcasting
        keep = np.asarray(indices)
        given = np.asarray([i for i in range(self.nvars) if i not in keep])
        sigmakk = self.sigma[keep[:, None], keep]
        sigmagg = self.sigma[given[:, None], given]
        sigmakg = self.sigma[keep[:, None], given]
        sigmagk = self.sigma[given[:, None], keep]


        sigma_new = sigmakk - np.dot(sigmakg, np.linalg.solve(sigmagg, sigmagk))
        mean_new = self.mean[keep] +  \
            np.dot(sigmakg, np.linalg.solve(sigmagg, values-self.mean[given]))

#        #or
#        sig = np.linalg.solve(sigmagg, sigmagk).T
#        mean_new = self.mean[keep] + np.dot(sigmakg, values-self.mean[given])
#        sigma_new = sigmakk - np.dot(sigmakg, sig)
        return MVNormal(mean_new, sigma_new)


#redefine some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

class MVT(MVElliptical):

    __name__ == 'Multivariate Student T Distribution'

    def __init__(self, mean, sigma, df):
        '''initialize instance

        Parameters
        ----------
        mean : array_like
            parameter mu (might be renamed), for symmetric distributions this
            is the mean
        sigma : array_like, 2d
            dispersion matrix, covariance matrix in normal distribution, but
            only proportional to covariance matrix in t distribution
        args : list
            distribution specific arguments, e.g. df for t distribution
        kwds : dict
            currently not used

        '''
        super(MVT, self).__init__(mean, sigma)
        self.extra_args = ['df']  #overwrites extra_args of super
        self.df = df

    def rvs(self, size=1):
        '''random variables with Student T distribution

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension
            - TODO: Not sure if this works for size tuples with len>1.

        Notes
        -----
        generated as a chi-square mixture of multivariate normal random
        variables.
        does this require df>2 ?


        '''
        from .multivariate import multivariate_t_rvs
        return multivariate_t_rvs(self.mean, self.sigma, df=self.df, n=size)


    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector

        '''

        x = np.asarray(x)

        df = self.df
        nvars = self.nvars

        x_whitened = self.whiten(x - self.mean) #should be float

        llf = - nvars * np_log(df * np_pi)
        llf -= self.logdetsigma
        llf -= (df + nvars) * np_log(1 + np.sum(x_whitened**2,-1) / df)
        llf *= 0.5
        llf += sps_gamln((df + nvars) / 2.) - sps_gamln(df / 2.)

        return llf

    def cdf(self, x, **kwds):
        '''cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        '''
        lower = -np.inf * np.ones_like(x)
        #std_sigma = np.sqrt(np.diag(self.sigma))
        upper = (x - self.mean)/self.std_sigma
        return mvstdtprob(lower, upper, self.corr, self.df, **kwds)
        #mvstdtcdf does not exist yet
        #return mvstdtcdf(lower, x, self.corr, df, **kwds)

    @property
    def cov(self):
        '''covariance matrix

        The covariance matrix for the t distribution does not exist for df<=2,
        and is equal to sigma * df/(df-2) for df>2

        '''
        if self.df <= 2:
            return np.nan * np.ones_like(self.sigma)
        else:
            return self.df / (self.df - 2.) * self.sigma

    def affine_transformed(self, shift, scale_matrix):
        '''return distribution of a full rank affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVT
            instance of multivariate t distribution given by affine
            transformation


        Notes
        -----

        This checks for eigvals<=0, so there are possible problems for cases
        with positive eigenvalues close to zero.

        see: http://www.statlect.com/mcdstu1.htm

        I'm not sure about general case, non-full rank transformation are not
        multivariate t distributed.

        y = a + B x

        where a is shift,
        B is full rank scale matrix with same dimension as sigma

        '''
        #full rank method could also be in elliptical and called with super
        #after the rank check
        B = scale_matrix  #tmp variable as shorthand
        if not B.shape == (self.nvars, self.nvars):
            if (np.linalg.eigvals(B) <= 0).any():
                raise ValueError('affine transform has to be full rank')

        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVT(mean_new, sigma_new, self.df)


def quad2d(func=lambda x: 1, lower=(-10,-10), upper=(10,10)):
    def fun(x, y):
        x = np.column_stack((x,y))
        return func(x)
    from scipy.integrate import dblquad
    return dblquad(fun, lower[0], upper[0], lambda y: lower[1],
                   lambda y: upper[1])

if __name__ == '__main__':

    from numpy.testing import assert_almost_equal, assert_array_almost_equal

    examples = ['mvn']

    mu = (0,0)
    covx = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu3 = [-1, 0., 2.]
    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                     [ 0.5 ,  1.5 ,  0.6 ],
                     [ 0.75,  0.6 ,  2.  ]])


    if 'mvn' in examples:
        bvn = BivariateNormal(mu, covx)
        rvs = bvn.rvs(size=1000)
        print(rvs.mean(0))
        print(np.cov(rvs, rowvar=0))
        print(bvn.expect())
        print(bvn.cdf([0,0]))
        bvn1 = BivariateNormal(mu, np.eye(2))
        bvn2 = BivariateNormal(mu, 4*np.eye(2))
        fun = lambda x : np.log(bvn1.pdf(x)) - np.log(bvn.pdf(x))
        print(bvn1.expect(fun))
        print(bvn1.kl(bvn2), bvn1.kl_mc(bvn2))
        print(bvn2.kl(bvn1), bvn2.kl_mc(bvn1))
        print(bvn1.kl(bvn), bvn1.kl_mc(bvn))
        mvn = MVNormal(mu, covx)
        mvn.pdf([0,0])
        mvn.pdf(np.zeros((2,2)))
        #np.dot(mvn.cholcovinv.T, mvn.cholcovinv) - mvn.covinv

        cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                         [ 0.5 ,  1.5 ,  0.6 ],
                         [ 0.75,  0.6 ,  2.  ]])
        mu3 = [-1, 0., 2.]
        mvn3 = MVNormal(mu3, cov3)
        mvn3.pdf((0., 2., 3.))
        mvn3.logpdf((0., 2., 3.))
        #comparisons with R mvtnorm::dmvnorm
        #decimal=14
#        mvn3.logpdf(cov3) - [-7.667977543898155, -6.917977543898155, -5.167977543898155]
#        #decimal 18
#        mvn3.pdf(cov3) - [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
#        #cheating new mean, same cov
#        mvn3.mean = np.array([0,0,0])
#        #decimal= 16
#        mvn3.pdf(cov3) - [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]

        #as asserts
        r_val = [-7.667977543898155, -6.917977543898155, -5.167977543898155]
        assert_array_almost_equal( mvn3.logpdf(cov3), r_val, decimal = 14)
        #decimal 18
        r_val = [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
        assert_array_almost_equal( mvn3.pdf(cov3), r_val, decimal = 17)
        #cheating new mean, same cov, too dangerous, got wrong instance in tests
        #mvn3.mean = np.array([0,0,0])
        mvn3c = MVNormal(np.array([0,0,0]), cov3)
        r_val = [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]
        assert_array_almost_equal( mvn3c.pdf(cov3), r_val, decimal = 16)

        mvn3b = MVNormal((0,0,0), 1)
        fun = lambda x : np.log(mvn3.pdf(x)) - np.log(mvn3b.pdf(x))
        print(mvn3.expect_mc(fun))
        print(mvn3.expect_mc(fun, size=200000))


    mvt = MVT((0,0), 1, 5)
    assert_almost_equal(mvt.logpdf(np.array([0.,0.])), -1.837877066409345,
                        decimal=15)
    assert_almost_equal(mvt.pdf(np.array([0.,0.])), 0.1591549430918953,
                        decimal=15)

    mvt.logpdf(np.array([1.,1.]))-(-3.01552989458359)

    mvt1 = MVT((0,0), 1, 1)
    mvt1.logpdf(np.array([1.,1.]))-(-3.48579549941151) #decimal=16

    rvs = mvt.rvs(100000)
    assert_almost_equal(np.cov(rvs, rowvar=0), mvt.cov, decimal=1)

    mvt31 = MVT(mu3, cov3, 1)
    assert_almost_equal(mvt31.pdf(cov3),
        [0.0007276818698165781, 0.0009980625182293658, 0.0027661422056214652],
        decimal=18)

    mvt = MVT(mu3, cov3, 3)
    assert_almost_equal(mvt.pdf(cov3),
        [0.000863777424247410, 0.001277510788307594, 0.004156314279452241],
        decimal=17)
