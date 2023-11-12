'''patching scipy to fit distributions and expect method

This adds new methods to estimate continuous distribution parameters with some
fixed/frozen parameters. It also contains functions that calculate the expected
value of a function for any continuous or discrete distribution

It temporarily also contains Bootstrap and Monte Carlo function for testing the
distribution fit, but these are neither general nor verified.

Author: josef-pktd
License: Simplified BSD
'''
from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate


########## patching scipy

#vonmises does not define finite bounds, because it is intended for circular
#support which does not define a proper pdf on the real line

stats.distributions.vonmises.a = -np.pi
stats.distributions.vonmises.b = np.pi

#the next 3 functions are for fit with some fixed parameters
#As they are written, they do not work as functions, only as methods

def _fitstart(self, x):
    '''example method, method of moment estimator as starting values

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    This example was written for the gamma distribution, but not verified
    with literature

    '''
    loc = np.min([x.min(),0])
    a = 4/stats.skew(x)**2
    scale = np.std(x) / np.sqrt(a)
    return (a, loc, scale)

def _fitstart_beta(self, x, fixed=None):
    '''method of moment estimator as starting values for beta distribution

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated
    fixed : None or array_like
        sequence of numbers and np.nan to indicate fixed parameters and parameters
        to estimate

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    References
    ----------
    for method of moment estimator for known loc and scale
    https://en.wikipedia.org/wiki/Beta_distribution#Parameter_estimation
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    NIST reference also includes reference to MLE in
    Johnson, Kotz, and Balakrishan, Volume II, pages 221-235

    '''
    #todo: separate out this part to be used for other compact support distributions
    #      e.g. rdist, vonmises, and truncnorm
    #      but this might not work because it might still be distribution specific
    a, b = x.min(), x.max()
    eps = (a-b)*0.01
    if fixed is None:
        #this part not checked with books
        loc = a - eps
        scale = (a - b) * (1 + 2*eps)
    else:
        if np.isnan(fixed[-2]):
            #estimate loc
            loc = a - eps
        else:
            loc = fixed[-2]
        if np.isnan(fixed[-1]):
            #estimate scale
            scale = (b + eps) - loc
        else:
            scale = fixed[-1]

    #method of moment for known loc scale:
    scale = float(scale)
    xtrans = (x - loc)/scale
    xm = xtrans.mean()
    xv = xtrans.var()
    tmp = (xm*(1-xm)/xv - 1)
    p = xm * tmp
    q = (1 - xm) * tmp

    return (p, q, loc, scale)  #check return type and should fixed be returned ?

def _fitstart_poisson(self, x, fixed=None):
    '''maximum likelihood estimator as starting values for Poisson distribution

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated
    fixed : None or array_like
        sequence of numbers and np.nan to indicate fixed parameters and parameters
        to estimate

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    References
    ----------
    MLE :
    https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood

    '''
    #todo: separate out this part to be used for other compact support distributions
    #      e.g. rdist, vonmises, and truncnorm
    #      but this might not work because it might still be distribution specific
    a = x.min()
    eps = 0 # is this robust ?
    if fixed is None:
        #this part not checked with books
        loc = a - eps
    else:
        if np.isnan(fixed[-1]):
            #estimate loc
            loc = a - eps
        else:
            loc = fixed[-1]

    #MLE for standard (unshifted, if loc=0) Poisson distribution

    xtrans = (x - loc)
    lambd = xtrans.mean()
    #second derivative d loglike/ dlambd Not used
    #dlldlambd = 1/lambd # check

    return (lambd, loc)  #check return type and should fixed be returned ?


def nnlf_fr(self, thetash, x, frmask):
    # new frozen version
    # - sum (log pdf(x, theta),axis=0)
    #   where theta are the parameters (including loc and scale)
    #
    try:
        if frmask is not None:
            theta = frmask.copy()
            theta[np.isnan(frmask)] = thetash
        else:
            theta = thetash
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError("Not enough input arguments.")
    if not self._argcheck(*args) or scale <= 0:
        return np.inf
    x = np.array((x-loc) / scale)
    cond0 = (x <= self.a) | (x >= self.b)
    if (np.any(cond0)):
        return np.inf
    else:
        N = len(x)
        #raise ValueError
        return self._nnlf(x, *args) + N*np.log(scale)

def fit_fr(self, data, *args, **kwds):
    '''estimate distribution parameters by MLE taking some parameters as fixed

    Parameters
    ----------
    data : ndarray, 1d
        data for which the distribution parameters are estimated,
    args : list ? check
        starting values for optimization
    kwds :

      - 'frozen' : array_like
           values for frozen distribution parameters and, for elements with
           np.nan, the corresponding parameter will be estimated

    Returns
    -------
    argest : ndarray
        estimated parameters


    Examples
    --------
    generate random sample
    >>> np.random.seed(12345)
    >>> x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)

    estimate all parameters
    >>> stats.gamma.fit(x)
    array([ 2.0243194 ,  0.20395655,  1.44411371])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
    array([ 2.0243194 ,  0.20395655,  1.44411371])

    keep loc fixed, estimate shape and scale parameters
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, np.nan])
    array([ 2.45603985,  1.27333105])

    keep loc and scale fixed, estimate shape parameter
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    array([ 3.00048828])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2])
    array([ 2.57792969])

    estimate only scale parameter for fixed shape and loc
    >>> stats.gamma.fit_fr(x, frozen=[2.5, 0.0, np.nan])
    array([ 1.25087891])

    Notes
    -----
    self is an instance of a distribution class. This can be attached to
    scipy.stats.distributions.rv_continuous

    *Todo*

    * check if docstring is correct
    * more input checking, args is list ? might also apply to current fit method

    '''
    loc0, scale0 = lmap(kwds.get, ['loc', 'scale'],[0.0, 1.0])
    Narg = len(args)

    if Narg == 0 and hasattr(self, '_fitstart'):
        x0 = self._fitstart(data)
    elif Narg > self.numargs:
        raise ValueError("Too many input arguments.")
    else:
        args += (1.0,)*(self.numargs-Narg)
        # location and scale are at the end
        x0 = args + (loc0, scale0)

    if 'frozen' in kwds:
        frmask = np.array(kwds['frozen'])
        if len(frmask) != self.numargs+2:
            raise ValueError("Incorrect number of frozen arguments.")
        else:
            # keep starting values for not frozen parameters
            for n in range(len(frmask)):
                # Troubleshooting ex_generic_mle_tdist
                if isinstance(frmask[n], np.ndarray) and frmask[n].size == 1:
                    frmask[n] = frmask[n].item()

            # If there were array elements, then frmask will be object-dtype,
            #  in which case np.isnan will raise TypeError
            frmask = frmask.astype(np.float64)
            x0  = np.array(x0)[np.isnan(frmask)]
    else:
        frmask = None

    #print(x0
    #print(frmask
    return optimize.fmin(self.nnlf_fr, x0,
                args=(np.ravel(data), frmask), disp=0)


#The next two functions/methods calculate expected value of an arbitrary
#function, however for the continuous functions intquad is use, which might
#require continuouity or smoothness in the function.


#TODO: add option for Monte Carlo integration

def expect(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    '''calculate expected value of a function with respect to the distribution

    location and scale only tested on a few examples

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution
        conditional : bool (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float

    Notes
    -----
    This function has not been checked for it's behavior when the integral is
    not finite. The integration behavior is inherited from scipy.integrate.quad.

    '''
    if fn is None:
        def fun(x, *args):
            return x*self.pdf(x, loc=loc, scale=scale, *args)
    else:
        def fun(x, *args):
            return fn(x)*self.pdf(x, loc=loc, scale=scale, *args)
    if lb is None:
        lb = loc + self.a * scale #(self.a - loc)/(1.0*scale)
    if ub is None:
        ub = loc + self.b * scale #(self.b - loc)/(1.0*scale)
    if conditional:
        invfac = (self.sf(lb, loc=loc, scale=scale, *args)
                  - self.sf(ub, loc=loc, scale=scale, *args))
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub,
                                args=args)[0]/invfac


def expect_v2(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    '''calculate expected value of a function with respect to the distribution

    location and scale only tested on a few examples

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set using
           quantiles of the distribution, see Notes
        conditional : bool (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float

    Notes
    -----
    This function has not been checked for it's behavior when the integral is
    not finite. The integration behavior is inherited from scipy.integrate.quad.

    The default limits are lb = self.ppf(1e-9, *args), ub = self.ppf(1-1e-9, *args)

    For some heavy tailed distributions, 'alpha', 'cauchy', 'halfcauchy',
    'levy', 'levy_l', and for 'ncf', the default limits are not set correctly
    even  when the expectation of the function is finite. In this case, the
    integration limits, lb and ub, should be chosen by the user. For example,
    for the ncf distribution, ub=1000 works in the examples.

    There are also problems with numerical integration in some other cases,
    for example if the distribution is very concentrated and the default limits
    are too large.

    '''
    #changes: 20100809
    #correction and refactoring how loc and scale are handled
    #uses now _pdf
    #needs more testing for distribution with bound support, e.g. genpareto

    if fn is None:
        def fun(x, *args):
            return (loc + x*scale)*self._pdf(x, *args)
    else:
        def fun(x, *args):
            return fn(loc + x*scale)*self._pdf(x, *args)
    if lb is None:
        #lb = self.a
        try:
            lb = self.ppf(1e-9, *args)  #1e-14 quad fails for pareto
        except ValueError:
            lb = self.a
    else:
        lb = max(self.a, (lb - loc)/(1.0*scale)) #transform to standardized
    if ub is None:
        #ub = self.b
        try:
            ub = self.ppf(1-1e-9, *args)
        except ValueError:
            ub = self.b
    else:
        ub = min(self.b, (ub - loc)/(1.0*scale))
    if conditional:
        invfac = self._sf(lb,*args) - self._sf(ub,*args)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub,
                                args=args, limit=500)[0]/invfac

### for discrete distributions

#TODO: check that for a distribution with finite support the calculations are
#      done with one array summation (np.dot)

#based on _drv2_moment(self, n, *args), but streamlined
def expect_discrete(self, fn=None, args=(), loc=0, lb=None, ub=None,
                    conditional=False):
    '''calculate expected value of a function with respect to the distribution
    for discrete distribution

    Parameters
    ----------
        (self : distribution instance as defined in scipy stats)
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        optional keyword parameters
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution, lb and ub are inclusive (ul<=k<=ub)
        conditional : bool (False)
           If true then the expectation is corrected by the conditional
           probability of the integration interval. The return value is the
           expectation of the function, conditional on being in the given
           interval (k such that ul<=k<=ub).

    Returns
    -------
        expected value : float

    Notes
    -----
    * function is not vectorized
    * accuracy: uses self.moment_tol as stopping criterium
        for heavy tailed distribution e.g. zipf(4), accuracy for
        mean, variance in example is only 1e-5,
        increasing precision (moment_tol) makes zipf very slow
    * suppnmin=100 internal parameter for minimum number of points to evaluate
        could be added as keyword parameter, to evaluate functions with
        non-monotonic shapes, points include integers in (-suppnmin, suppnmin)
    * uses maxcount=1000 limits the number of points that are evaluated
        to break loop for infinite sums
        (a maximum of suppnmin+1000 positive plus suppnmin+1000 negative integers
        are evaluated)


    '''

    #moment_tol = 1e-12 # increase compared to self.moment_tol,
    # too slow for only small gain in precision for zipf

    #avoid endless loop with unbound integral, eg. var of zipf(2)
    maxcount = 1000
    suppnmin = 100  #minimum number of points to evaluate (+ and -)

    if fn is None:
        def fun(x):
            #loc and args from outer scope
            return (x+loc)*self._pmf(x, *args)
    else:
        def fun(x):
            #loc and args from outer scope
            return fn(x+loc)*self._pmf(x, *args)
    # used pmf because _pmf does not check support in randint
    # and there might be problems(?) with correct self.a, self.b at this stage
    # maybe not anymore, seems to work now with _pmf

    self._argcheck(*args) # (re)generate scalar self.a and self.b
    if lb is None:
        lb = (self.a)
    else:
        lb = lb - loc

    if ub is None:
        ub = (self.b)
    else:
        ub = ub - loc
    if conditional:
        invfac = self.sf(lb,*args) - self.sf(ub+1,*args)
    else:
        invfac = 1.0

    tot = 0.0
    low, upp = self._ppf(0.001, *args), self._ppf(0.999, *args)
    low = max(min(-suppnmin, low), lb)
    upp = min(max(suppnmin, upp), ub)
    supp = np.arange(low, upp+1, self.inc) #check limits
    #print('low, upp', low, upp
    tot = np.sum(fun(supp))
    diff = 1e100
    pos = upp + self.inc
    count = 0

    #handle cases with infinite support

    while (pos <= ub) and (diff > self.moment_tol) and count <= maxcount:
        diff = fun(pos)
        tot += diff
        pos += self.inc
        count += 1

    if self.a < 0: #handle case when self.a = -inf
        diff = 1e100
        pos = low - self.inc
        while (pos >= lb) and (diff > self.moment_tol) and count <= maxcount:
            diff = fun(pos)
            tot += diff
            pos -= self.inc
            count += 1
    if count > maxcount:
        # replace with proper warning
        print('sum did not converge')
    return tot/invfac

stats.distributions.rv_continuous.fit_fr = fit_fr
stats.distributions.rv_continuous.nnlf_fr = nnlf_fr
stats.distributions.rv_continuous.expect = expect
stats.distributions.rv_discrete.expect = expect_discrete
stats.distributions.beta_gen._fitstart = _fitstart_beta  #not tried out yet
stats.distributions.poisson_gen._fitstart = _fitstart_poisson  #not tried out yet

########## end patching scipy


def distfitbootstrap(sample, distr, nrepl=100):
    '''run bootstrap for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data for bootstrap
    distr : distribution instance with fit_fr method
    nrepl : int
        number of bootstrap replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all bootstrap replications

    '''
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        rvsind = np.random.randint(nobs, size=nobs)
        x = sample[rvsind]
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res

def distfitmc(sample, distr, nrepl=100, distkwds={}):
    '''run Monte Carlo for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data, in Monte Carlo only used to get nobs,
    distr : distribution instance with fit_fr method
    nrepl : int
        number of Monte Carlo replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all Monte Carlo replications

    '''
    arg = distkwds.pop('arg')
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        x = distr.rvs(arg, size=nobs, **distkwds)
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res


def printresults(sample, arg, bres, kind='bootstrap'):
    '''calculate and print(Bootstrap or Monte Carlo result

    Parameters
    ----------
    sample : ndarray
        original sample data
    arg : float   (for general case will be array)
    bres : ndarray
        parameter estimates from Bootstrap or Monte Carlo run
    kind : {'bootstrap', 'montecarlo'}
        output is printed for Mootstrap (default) or Monte Carlo

    Returns
    -------
    None, currently only printing

    Notes
    -----
    still a bit a mess because it is used for both Bootstrap and Monte Carlo

    made correction:
        reference point for bootstrap is estimated parameter

    not clear:
        I'm not doing any ddof adjustment in estimation of variance, do we
        need ddof>0 ?

    todo: return results and string instead of printing

    '''
    print('true parameter value')
    print(arg)
    print('MLE estimate of parameters using sample (nobs=%d)'% (nobs))
    argest = distr.fit_fr(sample, frozen=[np.nan, 0.0, 1.0])
    print(argest)
    if kind == 'bootstrap':
        #bootstrap compares to estimate from sample
        argorig = arg
        arg = argest

    print('%s distribution of parameter estimate (nrepl=%d)'% (kind, nrepl))
    print('mean = %f, bias=%f' % (bres.mean(0), bres.mean(0)-arg))
    print('median', np.median(bres, axis=0))
    print('var and std', bres.var(0), np.sqrt(bres.var(0)))
    bmse = ((bres - arg)**2).mean(0)
    print('mse, rmse', bmse, np.sqrt(bmse))
    bressorted = np.sort(bres)
    print('%s confidence interval (90%% coverage)' % kind)
    print(bressorted[np.floor(nrepl*0.05)], bressorted[np.floor(nrepl*0.95)])
    print('%s confidence interval (90%% coverage) normal approximation' % kind)
    print(stats.norm.ppf(0.05, loc=bres.mean(), scale=bres.std()),)
    print(stats.norm.isf(0.05, loc=bres.mean(), scale=bres.std()))
    print('Kolmogorov-Smirnov test for normality of %s distribution' % kind)
    print(' - estimated parameters, p-values not really correct')
    print(stats.kstest(bres, 'norm', (bres.mean(), bres.std())))


if __name__ == '__main__':

    examplecases = ['largenumber', 'bootstrap', 'montecarlo'][:]

    if 'largenumber' in examplecases:

        print('\nDistribution: vonmises')

        for nobs in [200]:#[20000, 1000, 100]:
            x = stats.vonmises.rvs(1.23, loc=0, scale=1, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('1.23, loc=0, scale=1')
            print('unconstrained')
            print(stats.vonmises.fit(x))
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))

        print('\nDistribution: gamma')
        distr = stats.gamma
        arg, loc, scale = 2.5, 0., 20.

        for nobs in [200]:#[20000, 1000, 100]:
            x = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('%f, loc=%f, scale=%f' % (arg, loc, scale))
            print('unconstrained')
            print(distr.fit(x))
            print(distr.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))
            print('with fixed loc')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, np.nan]))


    ex = ['gamma', 'vonmises'][0]

    if ex == 'gamma':
        distr = stats.gamma
        arg, loc, scale = 2.5, 0., 1
    elif ex == 'vonmises':
        distr = stats.vonmises
        arg, loc, scale = 1.5, 0., 1
    else:
        raise ValueError('wrong example')

    nobs = 100
    nrepl = 1000

    sample = distr.rvs(arg, loc=loc, scale=scale, size=nobs)

    print('\nDistribution:', distr)
    if 'bootstrap' in examplecases:
        print('\nBootstrap')
        bres = distfitbootstrap(sample, distr, nrepl=nrepl )
        printresults(sample, arg, bres)

    if 'montecarlo' in examplecases:
        print('\nMonteCarlo')
        mcres = distfitmc(sample, distr, nrepl=nrepl,
                          distkwds=dict(arg=arg, loc=loc, scale=scale))
        printresults(sample, arg, mcres, kind='montecarlo')
