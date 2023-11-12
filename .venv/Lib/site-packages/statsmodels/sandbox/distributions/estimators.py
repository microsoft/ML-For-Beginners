'''estimate distribution parameters by various methods
method of moments or matching quantiles, and Maximum Likelihood estimation
based on binned data and Maximum Product-of-Spacings

Warning: I'm still finding cut-and-paste and refactoring errors, e.g.
    hardcoded variables from outer scope in functions
    some results do not seem to make sense for Pareto case,
    looks better now after correcting some name errors

initially loosely based on a paper and blog for quantile matching
  by John D. Cook
  formula for gamma quantile (ppf) matching by him (from paper)
  http://www.codeproject.com/KB/recipes/ParameterPercentile.aspx
  http://www.johndcook.com/blog/2010/01/31/parameters-from-percentiles/
  this is what I actually used (in parts):
  http://www.bepress.com/mdandersonbiostat/paper55/

quantile based estimator
^^^^^^^^^^^^^^^^^^^^^^^^
only special cases for number or parameters so far
Is there a literature for GMM estimation of distribution parameters? check
    found one: Wu/Perloff 2007


binned estimator
^^^^^^^^^^^^^^^^
* I added this also
* use it for chisquare tests with estimation distribution parameters
* move this to distribution_extras (next to gof tests powerdiscrepancy and
  continuous) or add to distribution_patch


example: t-distribution
* works with quantiles if they contain tail quantiles
* results with momentcondquant do not look as good as mle estimate

TODOs
* rearange and make sure I do not use module globals (as I did initially) DONE
  make two version exactly identified method of moments with fsolve
  and GMM (?) version with fmin
  and maybe the special cases of JD Cook
  update: maybe exact (MM) version is not so interesting compared to GMM
* add semifrozen version of moment and quantile based estimators,
  e.g. for beta (both loc and scale fixed), or gamma (loc fixed)
* add beta example to the semifrozen MLE, fitfr, code
  -> added method of moment estimator to _fitstart for beta
* start a list of how well different estimators, especially current mle work
  for the different distributions
* need general GMM code (with optimal weights ?), looks like a good example
  for it
* get example for binned data estimation, mailing list a while ago
* any idea when these are better than mle ?
* check language: I use quantile to mean the value of the random variable, not
  quantile between 0 and 1.
* for GMM: move moment conditions to separate function, so that they can be
  used for further analysis, e.g. covariance matrix of parameter estimates
* question: Are GMM properties different for matching quantiles with cdf or
  ppf? Estimate should be the same, but derivatives of moment conditions
  differ.
* add maximum spacings estimator, Wikipedia, Per Brodtkorb -> basic version Done
* add parameter estimation based on empirical characteristic function
  (Carrasco/Florens), especially for stable distribution
* provide a model class based on estimating all distributions, and collect
  all distribution specific information


References
----------

Ximing Wu, Jeffrey M. Perloff, GMM estimation of a maximum entropy
distribution with interval data, Journal of Econometrics, Volume 138,
Issue 2, 'Information and Entropy Econometrics' - A Volume in Honor of
Arnold Zellner, June 2007, Pages 532-546, ISSN 0304-4076,
DOI: 10.1016/j.jeconom.2006.05.008.
http://www.sciencedirect.com/science/article/B6VC0-4K606TK-4/2/78bc07c6245546374490f777a6bdbbcc
http://escholarship.org/uc/item/7jf5w1ht  (working paper)

Johnson, Kotz, Balakrishnan: Volume 2


Author : josef-pktd
License : BSD
created : 2010-04-20

changes:
added Maximum Product-of-Spacings 2010-05-12

'''

import numpy as np
from scipy import stats, optimize, special

cache = {}   #module global storage for temp results, not used


# the next two use distfn from module scope - not anymore
def gammamomentcond(distfn, params, mom2, quantile=None):
    '''estimate distribution parameters based method of moments (mean,
    variance) for distributions with 1 shape parameter and fixed loc=0.

    Returns
    -------
    cond : function

    Notes
    -----
    first test version, quantile argument not used

    '''
    def cond(params):
        alpha, scale = params
        mom2s = distfn.stats(alpha, 0.,scale)
        #quantil
        return np.array(mom2)-mom2s
    return cond

def gammamomentcond2(distfn, params, mom2, quantile=None):
    '''estimate distribution parameters based method of moments (mean,
    variance) for distributions with 1 shape parameter and fixed loc=0.

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical moments

    Notes
    -----
    first test version, quantile argument not used

    The only difference to previous function is return type.

    '''
    alpha, scale = params
    mom2s = distfn.stats(alpha, 0.,scale)
    return np.array(mom2)-mom2s



######### fsolve does not move in small samples, fmin not very accurate
def momentcondunbound(distfn, params, mom2, quantile=None):
    '''moment conditions for estimating distribution parameters using method
    of moments, uses mean, variance and one quantile for distributions
    with 1 shape parameter.

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical moments and quantiles

    '''
    shape, loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc,scale)) - mom2
    if quantile is not None:
        pq, xq = quantile
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return np.concatenate([mom2diff, cdfdiff[:1]])
    return mom2diff


###### loc scale only
def momentcondunboundls(distfn, params, mom2, quantile=None, shape=None):
    '''moment conditions for estimating loc and scale of a distribution
    with method of moments using either 2 quantiles or 2 moments (not both).

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical moments or quantiles

    '''
    loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if quantile is not None:
        pq, xq = quantile
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        #return np.concatenate([mom2diff, cdfdiff[:1]])
        return cdfdiff
    return mom2diff



######### try quantile GMM with identity weight matrix
#(just a guess that's what it is

def momentcondquant(distfn, params, mom2, quantile=None, shape=None):
    '''moment conditions for estimating distribution parameters by matching
    quantiles, defines as many moment conditions as quantiles.

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical quantiles

    Notes
    -----
    This can be used for method of moments or for generalized method of
    moments.

    '''
    #this check looks redundant/unused know
    if len(params) == 2:
        loc, scale = params
    elif len(params) == 3:
        shape, loc, scale = params
    else:
        #raise NotImplementedError
        pass #see whether this might work, seems to work for beta with 2 shape args

    #mom2diff = np.array(distfn.stats(*params)) - mom2
    #if not quantile is None:
    pq, xq = quantile
    #ppfdiff = distfn.ppf(pq, alpha)
    cdfdiff = distfn.cdf(xq, *params) - pq
    #return np.concatenate([mom2diff, cdfdiff[:1]])
    return cdfdiff
    #return mom2diff

def fitquantilesgmm(distfn, x, start=None, pquant=None, frozen=None):
    if pquant is None:
        pquant = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
    if start is None:
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(x)
        else:
            start = [1]*distfn.numargs + [0.,1.]
    #TODO: vectorize this:
    xqs = [stats.scoreatpercentile(x, p) for p in pquant*100]
    mom2s = None
    parest = optimize.fmin(lambda params:np.sum(
        momentcondquant(distfn, params, mom2s,(pquant,xqs), shape=None)**2), start)
    return parest



def fitbinned(distfn, freq, binedges, start, fixed=None):
    '''estimate parameters of distribution function for binned data using MLE

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : ndarray, 1d
        frequency count, e.g. obtained by histogram
    binedges : ndarray, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length

    Returns
    -------
    paramest : ndarray
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    lnnobsfact = special.gammaln(nobs+1)

    def nloglike(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''
        prob = np.diff(distfn.cdf(binedges, *params))
        return -(lnnobsfact + np.sum(freq*np.log(prob)- special.gammaln(freq+1)))
    return optimize.fmin(nloglike, start)


def fitbinnedgmm(distfn, freq, binedges, start, fixed=None, weightsoptimal=True):
    '''estimate parameters of distribution function for binned data using GMM

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : ndarray, 1d
        frequency count, e.g. obtained by histogram
    binedges : ndarray, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length
    fixed : None
        not used yet
    weightsoptimal : bool
        If true, then the optimal weighting matrix for GMM is used. If false,
        then the identity matrix is used

    Returns
    -------
    paramest : ndarray
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    if weightsoptimal:
        weights = freq/float(nobs)
    else:
        weights = np.ones(len(freq))
    freqnormed = freq/float(nobs)
    # skip turning weights into matrix diag(freq/float(nobs))

    def gmmobjective(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''
        prob = np.diff(distfn.cdf(binedges, *params))
        momcond = freqnormed - prob
        return np.dot(momcond*weights, momcond)
    return optimize.fmin(gmmobjective, start)

#Addition from try_maxproductspacings:
"""Estimating Parameters of Log-Normal Distribution with Maximum
Likelihood and Maximum Product-of-Spacings

MPS definiton from JKB page 233

Created on Tue May 11 13:52:50 2010
Author: josef-pktd
License: BSD
"""

def hess_ndt(fun, pars, args, options):
    import numdifftools as ndt
    if not ('stepMax' in options or 'stepFix' in options):
        options['stepMax'] = 1e-5
    f = lambda params: fun(params, *args)
    h = ndt.Hessian(f, **options)
    return h(pars), h

def logmps(params, xsorted, dist):
    '''calculate negative log of Product-of-Spacings

    Parameters
    ----------
    params : array_like, tuple ?
        parameters of the distribution funciton
    xsorted : array_like
        data that is already sorted
    dist : instance of a distribution class
        only cdf method is used

    Returns
    -------
    mps : float
        negative log of Product-of-Spacings


    Notes
    -----
    MPS definiton from JKB page 233
    '''
    xcdf = np.r_[0., dist.cdf(xsorted, *params), 1.]
    D = np.diff(xcdf)
    return -np.log(D).mean()

def getstartparams(dist, data):
    '''get starting values for estimation of distribution parameters

    Parameters
    ----------
    dist : distribution instance
        the distribution instance needs to have either a method fitstart
        or an attribute numargs
    data : ndarray
        data for which preliminary estimator or starting value for
        parameter estimation is desired

    Returns
    -------
    x0 : ndarray
        preliminary estimate or starting value for the parameters of
        the distribution given the data, including loc and scale

    '''
    if hasattr(dist, 'fitstart'):
        #x0 = getattr(dist, 'fitstart')(data)
        x0 = dist.fitstart(data)
    else:
        if np.isfinite(dist.a):
            x0 = np.r_[[1.]*dist.numargs, (data.min()-1), 1.]
        else:
            x0 = np.r_[[1.]*dist.numargs, (data.mean()-1), 1.]
    return x0

def fit_mps(dist, data, x0=None):
    '''Estimate distribution parameters with Maximum Product-of-Spacings

    Parameters
    ----------
    params : array_like, tuple ?
        parameters of the distribution funciton
    xsorted : array_like
        data that is already sorted
    dist : instance of a distribution class
        only cdf method is used

    Returns
    -------
    x : ndarray
        estimates for the parameters of the distribution given the data,
        including loc and scale


    '''
    xsorted = np.sort(data)
    if x0 is None:
        x0 = getstartparams(dist, xsorted)
    args = (xsorted, dist)
    print(x0)
    #print(args)
    return optimize.fmin(logmps, x0, args=args)



if __name__ == '__main__':

    #Example: gamma - distribution
    #-----------------------------

    print('\n\nExample: gamma Distribution')
    print(    '---------------------------')

    alpha = 2
    xq = [0.5, 4]
    pq = [0.1, 0.9]
    print(stats.gamma.ppf(pq, alpha))
    xq = stats.gamma.ppf(pq, alpha)
    print(np.diff((stats.gamma.ppf(pq, np.linspace(0.01,4,10)[:,None])*xq[::-1])))
    #optimize.bisect(lambda alpha: np.diff((stats.gamma.ppf(pq, alpha)*xq[::-1])))
    print(optimize.fsolve(lambda alpha: np.diff((stats.gamma.ppf(pq, alpha)*xq[::-1])), 3.))

    distfn = stats.gamma
    mcond = gammamomentcond(distfn, [5.,10], mom2=stats.gamma.stats(alpha, 0.,1.), quantile=None)
    print(optimize.fsolve(mcond, [1.,2.]))
    mom2 = stats.gamma.stats(alpha, 0.,1.)
    print(optimize.fsolve(lambda params:gammamomentcond2(distfn, params, mom2), [1.,2.]))

    grvs = stats.gamma.rvs(alpha, 0.,2., size=1000)
    mom2 = np.array([grvs.mean(), grvs.var()])
    alphaestq = optimize.fsolve(lambda params:gammamomentcond2(distfn, params, mom2), [1.,3.])
    print(alphaestq)
    print('scale = ', xq/stats.gamma.ppf(pq, alphaestq))


    #Example beta - distribution
    #---------------------------

    #Warning: this example had cut-and-paste errors

    print('\n\nExample: beta Distribution')
    print(    '--------------------------')

    #monkey patching :
##    if hasattr(stats.beta, '_fitstart'):
##        del stats.beta._fitstart  #bug in _fitstart  #raises AttributeError: _fitstart
    #stats.distributions.beta_gen._fitstart = lambda self, data : np.array([1,1,0,1])
    #_fitstart seems to require a tuple
    stats.distributions.beta_gen._fitstart = lambda self, data : (5,5,0,1)

    pq = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
    #rvsb = stats.beta.rvs(0.5,0.15,size=200)
    rvsb = stats.beta.rvs(10,15,size=2000)
    print('true params', 10, 15, 0, 1)
    print(stats.beta.fit(rvsb))
    xqsb = [stats.scoreatpercentile(rvsb, p) for p in pq*100]
    mom2s = np.array([rvsb.mean(), rvsb.var()])
    betaparest_gmmquantile = optimize.fmin(lambda params:np.sum(momentcondquant(stats.beta, params, mom2s,(pq,xqsb), shape=None)**2),
                                           [10,10, 0., 1.], maxiter=2000)
    print('betaparest_gmmquantile',  betaparest_gmmquantile)
    #result sensitive to initial condition


    #Example t - distribution
    #------------------------

    print('\n\nExample: t Distribution')
    print(    '-----------------------')

    nobs = 1000
    distfn = stats.t
    pq = np.array([0.1,0.9])
    paramsdgp = (5, 0, 1)
    trvs = distfn.rvs(5, 0, 1, size=nobs)
    xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
    mom2th = distfn.stats(*paramsdgp)
    mom2s = np.array([trvs.mean(), trvs.var()])
    tparest_gmm3quantilefsolve = optimize.fsolve(lambda params:momentcondunbound(distfn,params, mom2s,(pq,xqs)), [10,1.,2.])
    print('tparest_gmm3quantilefsolve', tparest_gmm3quantilefsolve)
    tparest_gmm3quantile = optimize.fmin(lambda params:np.sum(momentcondunbound(distfn,params, mom2s,(pq,xqs))**2), [10,1.,2.])
    print('tparest_gmm3quantile', tparest_gmm3quantile)
    print(distfn.fit(trvs))

    ##

    ##distfn = stats.t
    ##pq = np.array([0.1,0.9])
    ##paramsdgp = (5, 0, 1)
    ##trvs = distfn.rvs(5, 0, 1, size=nobs)
    ##xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
    ##mom2th = distfn.stats(*paramsdgp)
    ##mom2s = np.array([trvs.mean(), trvs.var()])
    print(optimize.fsolve(lambda params:momentcondunboundls(distfn, params, mom2s,shape=5), [1.,2.]))
    print(optimize.fmin(lambda params:np.sum(momentcondunboundls(distfn, params, mom2s,shape=5)**2), [1.,2.]))
    print(distfn.fit(trvs))
    #loc, scale, based on quantiles
    print(optimize.fsolve(lambda params:momentcondunboundls(distfn, params, mom2s,(pq,xqs),shape=5), [1.,2.]))

    ##

    pq = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
    #paramsdgp = (5, 0, 1)
    xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
    tparest_gmmquantile = optimize.fmin(lambda params:np.sum(momentcondquant(distfn, params, mom2s,(pq,xqs), shape=None)**2), [10, 1.,2.])
    print('tparest_gmmquantile', tparest_gmmquantile)
    tparest_gmmquantile2 = fitquantilesgmm(distfn, trvs, start=[10, 1.,2.], pquant=None, frozen=None)
    print('tparest_gmmquantile2', tparest_gmmquantile2)


    ##


    #use trvs from before
    bt = stats.t.ppf(np.linspace(0,1,21),5)
    ft,bt = np.histogram(trvs,bins=bt)
    print('fitbinned t-distribution')
    tparest_mlebinew = fitbinned(stats.t, ft, bt, [10, 0, 1])
    tparest_gmmbinewidentity = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1])
    tparest_gmmbinewoptimal = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1], weightsoptimal=False)
    print(paramsdgp)

    #Note: this can be used for chisquare test and then has correct asymptotic
    #   distribution for a distribution with estimated parameters, find ref again
    #TODO combine into test with binning included, check rule for number of bins

    #bt2 = stats.t.ppf(np.linspace(trvs.,1,21),5)
    ft2,bt2 = np.histogram(trvs,bins=50)
    'fitbinned t-distribution'
    tparest_mlebinel = fitbinned(stats.t, ft2, bt2, [10, 0, 1])
    tparest_gmmbinelidentity = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1])
    tparest_gmmbineloptimal = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1], weightsoptimal=False)
    tparest_mle = stats.t.fit(trvs)

    np.set_printoptions(precision=6)
    print('sample size', nobs)
    print('true (df, loc, scale)      ', paramsdgp)
    print('parest_mle                 ', tparest_mle)
    print
    print('tparest_mlebinel           ', tparest_mlebinel)
    print('tparest_gmmbinelidentity   ', tparest_gmmbinelidentity)
    print('tparest_gmmbineloptimal    ', tparest_gmmbineloptimal)
    print
    print('tparest_mlebinew           ', tparest_mlebinew)
    print('tparest_gmmbinewidentity   ', tparest_gmmbinewidentity)
    print('tparest_gmmbinewoptimal    ', tparest_gmmbinewoptimal)
    print
    print('tparest_gmmquantileidentity', tparest_gmmquantile)
    print('tparest_gmm3quantilefsolve ', tparest_gmm3quantilefsolve)
    print('tparest_gmm3quantile       ', tparest_gmm3quantile)

    ''' example results:
    standard error for df estimate looks large
    note: iI do not impose that df is an integer, (b/c not necessary)
    need Monte Carlo to check variance of estimators


    sample size 1000
    true (df, loc, scale)       (5, 0, 1)
    parest_mle                  [ 4.571405 -0.021493  1.028584]

    tparest_mlebinel            [ 4.534069 -0.022605  1.02962 ]
    tparest_gmmbinelidentity    [ 2.653056  0.012807  0.896958]
    tparest_gmmbineloptimal     [ 2.437261 -0.020491  0.923308]

    tparest_mlebinew            [ 2.999124 -0.0199    0.948811]
    tparest_gmmbinewidentity    [ 2.900939 -0.020159  0.93481 ]
    tparest_gmmbinewoptimal     [ 2.977764 -0.024925  0.946487]

    tparest_gmmquantileidentity [ 3.940797 -0.046469  1.002001]
    tparest_gmm3quantilefsolve  [ 10.   1.   2.]
    tparest_gmm3quantile        [ 6.376101 -0.029322  1.112403]
    '''

    #Example with Maximum Product of Spacings Estimation
    #===================================================

    #Example: Lognormal Distribution
    #-------------------------------

    #tough problem for MLE according to JKB
    #but not sure for which parameters

    print('\n\nExample: Lognormal Distribution')
    print(    '-------------------------------')

    sh = np.exp(10)
    sh = 0.01
    print(sh)
    x = stats.lognorm.rvs(sh,loc=100, scale=10,size=200)

    print(x.min())
    print(stats.lognorm.fit(x,  1.,loc=x.min()-1,scale=1))

    xsorted = np.sort(x)

    x0 = [1., x.min()-1, 1]
    args = (xsorted, stats.lognorm)
    print(optimize.fmin(logmps,x0,args=args))


    #Example: Lomax, Pareto, Generalized Pareto Distributions
    #--------------------------------------------------------

    #partially a follow-up to the discussion about numpy.random.pareto
    #Reference: JKB
    #example Maximum Product of Spacings Estimation

    # current results:
    # does not look very good yet sensitivity to starting values
    # Pareto and Generalized Pareto look like a tough estimation problemprint('\n\nExample: Lognormal Distribution'

    print('\n\nExample: Lomax, Pareto, Generalized Pareto Distributions')
    print(    '--------------------------------------------------------')

    p2rvs = stats.genpareto.rvs(2, size=500)
    #Note: is Lomax without +1; and classical Pareto with +1
    p2rvssorted = np.sort(p2rvs)
    argsp = (p2rvssorted, stats.pareto)
    x0p = [1., p2rvs.min()-5, 1]
    print(optimize.fmin(logmps,x0p,args=argsp))
    print(stats.pareto.fit(p2rvs, 0.5, loc=-20, scale=0.5))
    print('gpdparest_ mle', stats.genpareto.fit(p2rvs))
    parsgpd = fit_mps(stats.genpareto, p2rvs)
    print('gpdparest_ mps', parsgpd)
    argsgpd = (p2rvssorted, stats.genpareto)
    options = dict(stepFix=1e-7)
    #hess_ndt(fun, pars, argsgdp, options)
    #the results for the following look strange, maybe refactoring error
    he, h = hess_ndt(logmps, parsgpd, argsgpd, options)
    print(np.linalg.eigh(he)[0])
    f = lambda params: logmps(params, *argsgpd)
    print(f(parsgpd))
    #add binned
    fp2, bp2 = np.histogram(p2rvs, bins=50)
    'fitbinned t-distribution'
    gpdparest_mlebinel = fitbinned(stats.genpareto, fp2, bp2, x0p)
    gpdparest_gmmbinelidentity = fitbinnedgmm(stats.genpareto, fp2, bp2, x0p)
    print('gpdparest_mlebinel', gpdparest_mlebinel)
    print('gpdparest_gmmbinelidentity', gpdparest_gmmbinelidentity)
    gpdparest_gmmquantile2 = fitquantilesgmm(
        stats.genpareto, p2rvs, start=x0p, pquant=None, frozen=None)
    print('gpdparest_gmmquantile2', gpdparest_gmmquantile2)

    print(fitquantilesgmm(stats.genpareto, p2rvs, start=x0p,
                          pquant=np.linspace(0.01,0.99,10), frozen=None))
    fp2, bp2 = np.histogram(
        p2rvs,
        bins=stats.genpareto(2).ppf(np.linspace(0,0.99,10)))
    print('fitbinnedgmm equal weight bins')
    print(fitbinnedgmm(stats.genpareto, fp2, bp2, x0p))
