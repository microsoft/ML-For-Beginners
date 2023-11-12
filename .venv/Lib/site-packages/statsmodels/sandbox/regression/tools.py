'''gradient/Jacobian of normal and t loglikelihood

use chain rule

normal derivative wrt mu, sigma and beta

new version: loc-scale distributions, derivative wrt loc, scale

also includes "standardized" t distribution (for use in GARCH)

TODO:
* use sympy for derivative of loglike wrt shape parameters
  it works for df of t distribution dlog(gamma(a))da = polygamma(0,a) check
  polygamma is available in scipy.special
* get loc-scale example to work with mean = X*b
* write some full unit test examples

A: josef-pktd

'''

import numpy as np
from scipy import special
from scipy.special import gammaln


def norm_lls(y, params):
    '''normal loglikelihood given observations and mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows

    Returns
    -------
    lls : ndarray
        contribution to loglikelihood for each observation
    '''

    mu, sigma2 = params.T
    lls = -0.5*(np.log(2*np.pi) + np.log(sigma2) + (y-mu)**2/sigma2)
    return lls

def norm_lls_grad(y, params):
    '''Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt variance in second column

    Notes
    -----
    this is actually the derivative wrt sigma not sigma**2, but evaluated
    with parameter sigma2 = sigma**2

    '''
    mu, sigma2 = params.T
    dllsdmu = (y-mu)/sigma2
    dllsdsigma2 = ((y-mu)**2/sigma2 - 1)/np.sqrt(sigma2)
    return np.column_stack((dllsdmu, dllsdsigma2))


def mean_grad(x, beta):
    '''gradient/Jacobian for d (x*beta)/ d beta
    '''
    return x

def normgrad(y, x, params):
    '''Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable with mean x*beta, and variance sigma2
    x : ndarray, 2d
        explanatory variables, observation in rows, variables in columns
    params : array_like, (nvars + 1)
        array of coefficients and variance (beta, sigma2)

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt scale (sigma) in second column
    assume params = (beta, sigma2)

    Notes
    -----
    TODO: for heteroscedasticity need sigma to be a 1d array

    '''
    beta = params[:-1]
    sigma2 = params[-1]*np.ones((len(y),1))
    dmudbeta = mean_grad(x, beta)
    mu = np.dot(x, beta)
    #print(beta, sigma2)
    params2 = np.column_stack((mu,sigma2))
    dllsdms = norm_lls_grad(y,params2)
    grad = np.column_stack((dllsdms[:,:1]*dmudbeta, dllsdms[:,:1]))
    return grad



def tstd_lls(y, params, df):
    '''t loglikelihood given observations and mean mu and variance sigma2 = 1

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows
    df : int
        degrees of freedom of the t distribution

    Returns
    -------
    lls : ndarray
        contribution to loglikelihood for each observation

    Notes
    -----
    parametrized for garch
    '''

    mu, sigma2 = params.T
    df = df*1.0
    #lls = gammaln((df+1)/2.) - gammaln(df/2.) - 0.5*np.log((df-2)*np.pi)
    #lls -= (df+1)/2. * np.log(1. + (y-mu)**2/(df-2.)/sigma2) + 0.5 * np.log(sigma2)
    lls = gammaln((df+1)/2.) - gammaln(df/2.) - 0.5*np.log((df-2)*np.pi)
    lls -= (df+1)/2. * np.log(1. + (y-mu)**2/(df-2)/sigma2) + 0.5 * np.log(sigma2)

    return lls

def norm_dlldy(y):
    '''derivative of log pdf of standard normal with respect to y
    '''
    return -y


def tstd_pdf(x, df):
    '''pdf for standardized (not standard) t distribution, variance is one

    '''

    r = np.array(df*1.0)
    Px = np.exp(special.gammaln((r+1)/2.)-special.gammaln(r/2.))/np.sqrt((r-2)*np.pi)
    Px /= (1+(x**2)/(r-2))**((r+1)/2.)
    return Px

def ts_lls(y, params, df):
    '''t loglikelihood given observations and mean mu and variance sigma2 = 1

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows
    df : int
        degrees of freedom of the t distribution

    Returns
    -------
    lls : ndarray
        contribution to loglikelihood for each observation

    Notes
    -----
    parametrized for garch
    normalized/rescaled so that sigma2 is the variance

    >>> df = 10; sigma = 1.
    >>> stats.t.stats(df, loc=0., scale=sigma.*np.sqrt((df-2.)/df))
    (array(0.0), array(1.0))
    >>> sigma = np.sqrt(2.)
    >>> stats.t.stats(df, loc=0., scale=sigma*np.sqrt((df-2.)/df))
    (array(0.0), array(2.0))
    '''
    print(y, params, df)
    mu, sigma2 = params.T
    df = df*1.0
    #lls = gammaln((df+1)/2.) - gammaln(df/2.) - 0.5*np.log((df-2)*np.pi)
    #lls -= (df+1)/2. * np.log(1. + (y-mu)**2/(df-2.)/sigma2) + 0.5 * np.log(sigma2)
    lls = gammaln((df+1)/2.) - gammaln(df/2.) - 0.5*np.log((df)*np.pi)
    lls -= (df+1.)/2. * np.log(1. + (y-mu)**2/(df)/sigma2) + 0.5 * np.log(sigma2)
    return lls


def ts_dlldy(y, df):
    '''derivative of log pdf of standard t with respect to y

    Parameters
    ----------
    y : array_like
        data points of random variable at which loglike is evaluated
    df : array_like
        degrees of freedom,shape parameters of log-likelihood function
        of t distribution

    Returns
    -------
    dlldy : ndarray
        derivative of loglikelihood wrt random variable y evaluated at the
        points given in y

    Notes
    -----
    with mean 0 and scale 1, but variance is df/(df-2)

    '''
    df = df*1.
    #(df+1)/2. / (1 + y**2/(df-2.)) * 2.*y/(df-2.)
    #return -(df+1)/(df-2.) / (1 + y**2/(df-2.)) * y
    return -(df+1)/(df) / (1 + y**2/(df)) * y

def tstd_dlldy(y, df):
    '''derivative of log pdf of standardized t with respect to y

        Parameters
        ----------
    y : array_like
        data points of random variable at which loglike is evaluated
    df : array_like
        degrees of freedom,shape parameters of log-likelihood function
        of t distribution

    Returns
    -------
    dlldy : ndarray
        derivative of loglikelihood wrt random variable y evaluated at the
        points given in y


    Notes
    -----
    parametrized for garch, standardized to variance=1
    '''
    #(df+1)/2. / (1 + y**2/(df-2.)) * 2.*y/(df-2.)
    return -(df+1)/(df-2.) / (1 + y**2/(df-2.)) * y
    #return (df+1)/(df) / (1 + y**2/(df)) * y

def locscale_grad(y, loc, scale, dlldy, *args):
    '''derivative of log-likelihood with respect to location and scale

    Parameters
    ----------
    y : array_like
        data points of random variable at which loglike is evaluated
    loc : float
        location parameter of distribution
    scale : float
        scale parameter of distribution
    dlldy : function
        derivative of loglikelihood fuction wrt. random variable x
    args : array_like
        shape parameters of log-likelihood function

    Returns
    -------
    dlldloc : ndarray
        derivative of loglikelihood wrt location evaluated at the
        points given in y
    dlldscale : ndarray
        derivative of loglikelihood wrt scale evaluated at the
        points given in y

    '''
    yst = (y-loc)/scale    #ystandardized
    dlldloc = -dlldy(yst, *args) / scale
    dlldscale = -1./scale - dlldy(yst, *args) * (y-loc)/scale**2
    return dlldloc, dlldscale

if __name__ == '__main__':
    verbose = 0
    if verbose:
        sig = 0.1
        beta = np.ones(2)
        rvs = np.random.randn(10,3)
        x = rvs[:,1:]
        y = np.dot(x,beta) + sig*rvs[:,0]

        params = [1,1,1]
        print(normgrad(y, x, params))

        dllfdbeta = (y-np.dot(x, beta))[:,None]*x   #for sigma = 1
        print(dllfdbeta)

        print(locscale_grad(y, np.dot(x, beta), 1, norm_dlldy))
        print(y-np.dot(x, beta))

    from scipy import stats, misc

    def llt(y,loc,scale,df):
        return np.log(stats.t.pdf(y, df, loc=loc, scale=scale))
    def lltloc(loc,y,scale,df):
        return np.log(stats.t.pdf(y, df, loc=loc, scale=scale))
    def lltscale(scale,y,loc,df):
        return np.log(stats.t.pdf(y, df, loc=loc, scale=scale))

    def llnorm(y,loc,scale):
        return np.log(stats.norm.pdf(y, loc=loc, scale=scale))
    def llnormloc(loc,y,scale):
        return np.log(stats.norm.pdf(y, loc=loc, scale=scale))
    def llnormscale(scale,y,loc):
        return np.log(stats.norm.pdf(y, loc=loc, scale=scale))

    if verbose:
        print('\ngradient of t')
        print(misc.derivative(llt, 1, dx=1e-6, n=1, args=(0,1,10), order=3))
        print('t ', locscale_grad(1, 0, 1, tstd_dlldy, 10))
        print('ts', locscale_grad(1, 0, 1, ts_dlldy, 10))
        print(misc.derivative(llt, 1.5, dx=1e-10, n=1, args=(0,1,20), order=3),)
        print('ts', locscale_grad(1.5, 0, 1, ts_dlldy, 20))
        print(misc.derivative(llt, 1.5, dx=1e-10, n=1, args=(0,2,20), order=3),)
        print('ts', locscale_grad(1.5, 0, 2, ts_dlldy, 20))
        print(misc.derivative(llt, 1.5, dx=1e-10, n=1, args=(1,2,20), order=3),)
        print('ts', locscale_grad(1.5, 1, 2, ts_dlldy, 20))
        print(misc.derivative(lltloc, 1, dx=1e-10, n=1, args=(1.5,2,20), order=3),)
        print(misc.derivative(lltscale, 2, dx=1e-10, n=1, args=(1.5,1,20), order=3))
        y,loc,scale,df = 1.5, 1, 2, 20
        print('ts', locscale_grad(y,loc,scale, ts_dlldy, 20))
        print(misc.derivative(lltloc, loc, dx=1e-10, n=1, args=(y,scale,df), order=3),)
        print(misc.derivative(lltscale, scale, dx=1e-10, n=1, args=(y,loc,df), order=3))

        print('\ngradient of norm')
        print(misc.derivative(llnorm, 1, dx=1e-6, n=1, args=(0,1), order=3))
        print(locscale_grad(1, 0, 1, norm_dlldy))
        y,loc,scale = 1.5, 1, 2
        print('ts', locscale_grad(y,loc,scale, norm_dlldy))
        print(misc.derivative(llnormloc, loc, dx=1e-10, n=1, args=(y,scale), order=3),)
        print(misc.derivative(llnormscale, scale, dx=1e-10, n=1, args=(y,loc), order=3))
        y,loc,scale = 1.5, 0, 1
        print('ts', locscale_grad(y,loc,scale, norm_dlldy))
        print(misc.derivative(llnormloc, loc, dx=1e-10, n=1, args=(y,scale), order=3),)
        print(misc.derivative(llnormscale, scale, dx=1e-10, n=1, args=(y,loc), order=3))
        #print('still something wrong with handling of scale and variance'
        #looks ok now
        print('\nloglike of t')
        print(tstd_lls(1, np.array([0,1]), 100), llt(1,0,1,100), 'differently standardized')
        print(tstd_lls(1, np.array([0,1]), 10), llt(1,0,1,10), 'differently standardized')
        print(ts_lls(1, np.array([0,1]), 10), llt(1,0,1,10))
        print(tstd_lls(1, np.array([0,1.*10./8.]), 10), llt(1.,0,1.,10))
        print(ts_lls(1, np.array([0,1]), 100), llt(1,0,1,100))

        print(tstd_lls(1, np.array([0,1]), 10), llt(1,0,1.*np.sqrt(8/10.),10))


    from numpy.testing import assert_almost_equal
    params =[(0, 1), (1.,1.), (0.,2.), ( 1., 2.)]
    yt = np.linspace(-2.,2.,11)
    for loc,scale in params:
        dlldlo = misc.derivative(llnormloc, loc, dx=1e-10, n=1, args=(yt,scale), order=3)
        dlldsc = misc.derivative(llnormscale, scale, dx=1e-10, n=1, args=(yt,loc), order=3)
        gr = locscale_grad(yt, loc, scale, norm_dlldy)
        assert_almost_equal(dlldlo, gr[0], 5, err_msg='deriv loc')
        assert_almost_equal(dlldsc, gr[1], 5, err_msg='deriv scale')
    for df in [3, 10, 100]:
        for loc,scale in params:
            dlldlo = misc.derivative(lltloc, loc, dx=1e-10, n=1, args=(yt,scale,df), order=3)
            dlldsc = misc.derivative(lltscale, scale, dx=1e-10, n=1, args=(yt,loc,df), order=3)
            gr = locscale_grad(yt, loc, scale, ts_dlldy, df)
            assert_almost_equal(dlldlo, gr[0], 4, err_msg='deriv loc')
            assert_almost_equal(dlldsc, gr[1], 4, err_msg='deriv scale')
            assert_almost_equal(ts_lls(yt, np.array([loc, scale**2]), df),
                                llt(yt,loc,scale,df), 5,
                                err_msg='loglike')
            assert_almost_equal(tstd_lls(yt, np.array([loc, scale**2]), df),
                                llt(yt,loc,scale*np.sqrt((df-2.)/df),df), 5,
                                err_msg='loglike')
