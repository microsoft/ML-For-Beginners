'''Multivariate Distribution

Probability of a multivariate t distribution

Now also mvstnormcdf has tests against R mvtnorm

Still need non-central t, extra options, and convenience function for
location, scale version.

Author: Josef Perktold
License: BSD (3-clause)

Reference:
Genz and Bretz for formula

'''
import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi

from .extras import mvstdnormcdf

from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln

def chi2_pdf(self, x, df):
    '''pdf of chi-square distribution'''
    #from scipy.stats.distributions
    Px = x**(df/2.0-1)*np.exp(-x/2.0)
    Px /= special.gamma(df/2.0)* 2**(df/2.0)
    return Px

def chi_pdf(x, df):
    tmp = (df-1.)*np_log(x) + (-x*x*0.5) - (df*0.5-1)*np_log(2.0) \
          - sps_gammaln(df*0.5)
    return np_exp(tmp)
    #return x**(df-1.)*np_exp(-x*x*0.5)/(2.0)**(df*0.5-1)/sps_gamma(df*0.5)

def chi_logpdf(x, df):
    tmp = (df-1.)*np_log(x) + (-x*x*0.5) - (df*0.5-1)*np_log(2.0) \
          - sps_gammaln(df*0.5)
    return tmp

def funbgh(s, a, b, R, df):
    sqrt_df = np.sqrt(df+0.5)
    ret = chi_logpdf(s,df)
    ret += np_log(mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R,
                                         maxpts=1000000, abseps=1e-6))
    ret = np_exp(ret)
    return ret

def funbgh2(s, a, b, R, df):
    n = len(a)
    sqrt_df = np.sqrt(df)
    #np.power(s, df-1) * np_exp(-s*s*0.5)
    return np_exp((df-1)*np_log(s)-s*s*0.5) \
           * mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R[np.tril_indices(n, -1)],
                          maxpts=1000000, abseps=1e-4)

def bghfactor(df):
    return np.power(2.0, 1-df*0.5) / sps_gamma(df*0.5)


def mvstdtprob(a, b, R, df, ieps=1e-5, quadkwds=None, mvstkwds=None):
    """
    Probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.
    """
    kwds = dict(args=(a, b, R, df), epsabs=1e-4, epsrel=1e-2, limit=150)
    if quadkwds is not None:
        kwds.update(quadkwds)
    lower, upper = chi.ppf([ieps, 1 - ieps], df)
    res, err = integrate.quad(funbgh2, lower, upper, **kwds)
    prob = res * bghfactor(df)
    return prob

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal




if __name__ == '__main__':
    corr = np.asarray([[1.0, 0, 0.5],[0,1,0],[0.5,0,1]])
    corr_indep = np.asarray([[1.0, 0, 0],[0,1,0],[0,0,1]])
    corr_equal = np.asarray([[1.0, 0.5, 0.5],[0.5,1,0.5],[0.5,0.5,1]])
    R = corr_equal
    a = np.array([-np.inf,-np.inf,-100.0])
    a = np.array([-0.96,-0.96,-0.96])
    b = np.array([0.0,0.0,0.0])
    b = np.array([0.96,0.96, 0.96])
    a[:] = -1
    b[:] = 3
    df = 10.
    sqrt_df = np.sqrt(df)
    print(mvstdnormcdf(a, b, corr, abseps=1e-6))

    #print integrate.quad(funbgh, 0, np.inf, args=(a,b,R,df))
    print((stats.t.cdf(b[0], df) - stats.t.cdf(a[0], df))**3)

    s = 1
    print(mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R))


    df=4
    print(mvstdtprob(a, b, R, df))

    S = np.array([[1.,.5],[.5,1.]])
    print(multivariate_t_rvs([10.,20.], S, 2, 5))

    nobs = 10000
    rvst = multivariate_t_rvs([10.,20.], S, 2, nobs)
    print(np.sum((rvst<[10.,20.]).all(1),0) * 1. / nobs)
    print(mvstdtprob(-np.inf*np.ones(2), np.zeros(2), R[:2,:2], 2))


    '''
        > lower <- -1
        > upper <- 3
        > df <- 4
        > corr <- diag(3)
        > delta <- rep(0, 3)
        > pmvt(lower=lower, upper=upper, delta=delta, df=df, corr=corr)
        [1] 0.5300413
        attr(,"error")
        [1] 4.321136e-05
        attr(,"msg")
        [1] "Normal Completion"
        > (pt(upper, df) - pt(lower, df))**3
        [1] 0.4988254

    '''
