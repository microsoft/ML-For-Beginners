import math
import numpy as np
from scipy import linalg, stats, special

from .linalg_decomp_1 import SvdArray


#univariate standard normal distribution
#following from scipy.stats.distributions with adjustments
sqrt2pi = math.sqrt(2 * np.pi)
logsqrt2pi = math.log(sqrt2pi)

class StandardNormal:
    '''Distribution of vector x, with independent distribution N(0,1)

    this is the same as univariate normal for pdf and logpdf

    other methods not checked/adjusted yet

    '''
    def rvs(self, size):
        return np.random.standard_normal(size)
    def pdf(self, x):
        return np.exp(-x**2 * 0.5) / sqrt2pi
    def logpdf(self, x):
        return -x**2 * 0.5 - logsqrt2pi
    def _cdf(self, x):
        return special.ndtr(x)
    def _logcdf(self, x):
        return np.log(special.ndtr(x))
    def _ppf(self, q):
        return special.ndtri(q)


class AffineTransform:
    '''affine full rank transformation of a multivariate distribution

    no dimension checking, assumes everything broadcasts correctly
    first version without bound support

    provides distribution of y given distribution of x
    y = const + tmat * x

    '''
    def __init__(self, const, tmat, dist):
        self.const = const
        self.tmat = tmat
        self.dist = dist
        self.nrv = len(const)
        if not np.equal(self.nrv, tmat.shape).all():
            raise ValueError('dimension of const and tmat do not agree')

        #replace the following with a linalgarray class
        self.tmatinv = linalg.inv(tmat)
        self.absdet = np.abs(np.linalg.det(self.tmat))
        self.logabsdet = np.log(np.abs(np.linalg.det(self.tmat)))
        self.dist

    def rvs(self, size):
        #size can only be integer not yet tuple
        print((size,)+(self.nrv,))
        return self.transform(self.dist.rvs(size=(size,)+(self.nrv,)))

    def transform(self, x):
        #return np.dot(self.tmat, x) + self.const
        return np.dot(x, self.tmat) + self.const

    def invtransform(self, y):
        return np.dot(self.tmatinv, y - self.const)

    def pdf(self, x):
        return 1. / self.absdet * self.dist.pdf(self.invtransform(x))

    def logpdf(self, x):
        return - self.logabsdet + self.dist.logpdf(self.invtransform(x))




class MultivariateNormalChol:
    '''multivariate normal distribution with cholesky decomposition of sigma

    ignoring mean at the beginning, maybe

    needs testing for broadcasting to contemporaneously but not intertemporaly
    correlated random variable, which axis?,
    maybe swapaxis or rollaxis if x.ndim != mean.ndim == (sigma.ndim - 1)

    initially 1d is ok, 2d should work with iid in axis 0 and mvn in axis 1

    '''

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.sigmainv = sigmainv
        self.cholsigma = linalg.cholesky(sigma)
        #the following makes it lower triangular with increasing time
        self.cholsigmainv = linalg.cholesky(sigmainv)[::-1,::-1]
        #todo: this might be a trick todo backward instead of forward filtering

    def whiten(self, x):
        return np.dot(cholsigmainv, x)

    def logpdf_obs(self, x):
        x = x - self.mean
        x_whitened = self.whiten(x)

        #sigmainv = linalg.cholesky(sigma)
        logdetsigma = np.log(np.linalg.det(sigma))

        sigma2 = 1. # error variance is included in sigma

        llike  =  0.5 * (np.log(sigma2)
                         - 2.* np.log(np.diagonal(self.cholsigmainv))
                         + (x_whitened**2)/sigma2
                         +  np.log(2*np.pi))

        return llike

    def logpdf(self, x):
        return self.logpdf_obs(x).sum(-1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))



class MultivariateNormal:

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = SvdArray(sigma)




def loglike_ar1(x, rho):
    '''loglikelihood of AR(1) process, as a test case

    sigma_u partially hard coded

    Greene chapter 12 eq. (12-31)
    '''
    x = np.asarray(x)
    u = np.r_[x[0], x[1:] - rho * x[:-1]]
    sigma_u2 = 2*(1-rho**2)
    loglik = 0.5*(-(u**2).sum(0) / sigma_u2 + np.log(1-rho**2)
                  - x.shape[0] * (np.log(2*np.pi) + np.log(sigma_u2)))
    return loglik


def ar2transform(x, arcoefs):
    '''

    (Greene eq 12-30)
    '''
    a1, a2 = arcoefs
    y = np.zeros_like(x)
    y[0] = np.sqrt((1+a2) * ((1-a2)**2 - a1**2) / (1-a2)) * x[0]
    y[1] = np.sqrt(1-a2**2) * x[2] - a1 * np.sqrt(1-a1**2)/(1-a2) * x[1] #TODO:wrong index in x
    y[2:] = x[2:] - a1 * x[1:-1] - a2 * x[:-2]
    return y


def mvn_loglike(x, sigma):
    '''loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    '''
    #see numpy thread
    #Sturla: sqmahal = (cx*cho_solve(cho_factor(S),cx.T).T).sum(axis=1)
    sigmainv = linalg.inv(sigma)
    logdetsigma = np.log(np.linalg.det(sigma))
    nobs = len(x)

    llf = - np.dot(x, np.dot(sigmainv, x))
    llf -= nobs * np.log(2 * np.pi)
    llf -= logdetsigma
    llf *= 0.5
    return llf

def mvn_nloglike_obs(x, sigma):
    '''loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    '''
    #see numpy thread
    #Sturla: sqmahal = (cx*cho_solve(cho_factor(S),cx.T).T).sum(axis=1)

    #Still wasteful to calculate pinv first
    sigmainv = linalg.inv(sigma)
    cholsigmainv = linalg.cholesky(sigmainv)
    #2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(A)))) #Dag mailinglist
    # logdet not needed ???
    #logdetsigma = 2 * np.sum(np.log(np.diagonal(cholsigmainv)))
    x_whitened = np.dot(cholsigmainv, x)

    #sigmainv = linalg.cholesky(sigma)
    logdetsigma = np.log(np.linalg.det(sigma))

    sigma2 = 1. # error variance is included in sigma

    llike  =  0.5 * (np.log(sigma2) - 2.* np.log(np.diagonal(cholsigmainv))
                          + (x_whitened**2)/sigma2
                          +  np.log(2*np.pi))

    return llike, (x_whitened**2)

nobs = 10
x = np.arange(nobs)
autocov = 2*0.8**np.arange(nobs)# +0.01 * np.random.randn(nobs)
sigma = linalg.toeplitz(autocov)
#sigma = np.diag(1+np.random.randn(10)**2)

cholsigma = linalg.cholesky(sigma).T#, lower=True)

sigmainv = linalg.inv(sigma)
cholsigmainv = linalg.cholesky(sigmainv)
#2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(A)))) #Dag mailinglist
# logdet not needed ???
#logdetsigma = 2 * np.sum(np.log(np.diagonal(cholsigmainv)))
x_whitened = np.dot(cholsigmainv, x)

#sigmainv = linalg.cholesky(sigma)
logdetsigma = np.log(np.linalg.det(sigma))

sigma2 = 1. # error variance is included in sigma

llike  =  0.5 * (np.log(sigma2) - 2.* np.log(np.diagonal(cholsigmainv))
                      + (x_whitened**2)/sigma2
                      +  np.log(2*np.pi))

ll, ls = mvn_nloglike_obs(x, sigma)
#the following are all the same for diagonal sigma
print(ll.sum(), 'll.sum()')
print(llike.sum(), 'llike.sum()')
print(np.log(stats.norm._pdf(x_whitened)).sum() - 0.5 * logdetsigma,)
print('stats whitened')
print(np.log(stats.norm.pdf(x,scale=np.sqrt(np.diag(sigma)))).sum(),)
print('stats scaled')
print(0.5*(np.dot(linalg.cho_solve((linalg.cho_factor(sigma, lower=False)[0].T,
                                    False),x.T), x)
           + nobs*np.log(2*np.pi)
           - 2.* np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5*(np.dot(linalg.cho_solve((linalg.cho_factor(sigma)[0].T, False),x.T), x) + nobs*np.log(2*np.pi)- 2.* np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5*(np.dot(linalg.cho_solve(linalg.cho_factor(sigma),x.T), x) + nobs*np.log(2*np.pi)- 2.* np.log(np.diagonal(cholsigmainv)).sum()))
print(mvn_loglike(x, sigma))


normtransf = AffineTransform(np.zeros(nobs), cholsigma, StandardNormal())
print(normtransf.logpdf(x_whitened).sum())
#print(normtransf.rvs(5)
print(loglike_ar1(x, 0.8))

mch = MultivariateNormalChol(np.zeros(nobs), sigma)
print(mch.logpdf(x))

#from .linalg_decomp_1 import tiny2zero
#print(tiny2zero(mch.cholsigmainv / mch.cholsigmainv[-1,-1])

xw = mch.whiten(x)
print('xSigmax', np.dot(xw,xw))
print('xSigmax', np.dot(x,linalg.cho_solve(linalg.cho_factor(mch.sigma),x)))
print('xSigmax', np.dot(x,linalg.cho_solve((mch.cholsigma, False),x)))
