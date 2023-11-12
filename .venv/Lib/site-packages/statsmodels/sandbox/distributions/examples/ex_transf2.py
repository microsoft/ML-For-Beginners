# -*- coding: utf-8 -*-
"""
Created on Sun May 09 22:23:22 2010
Author: josef-pktd
Licese: BSD
"""
import numpy as np

from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
    ExpTransf_gen, LogTransf_gen,
    squarenormalg, absnormalg, negsquarenormalg, squaretg)

#define these as module globals
l, s = 0.0, 1.0
ppfq = [0.1, 0.5, 0.9]
xx = [0.95, 1.0, 1.1]
nxx = [-0.95, -1.0, -1.1]


def test_loggamma():
    #'Results for expgamma'
    loggammaexpg = LogTransf_gen(stats.gamma)
    cdftr = loggammaexpg._cdf(1,10)
    cdfst = stats.loggamma.cdf(1,10)
    assert_almost_equal(cdfst, cdftr, 14)

    cdftr = loggammaexpg._cdf(2,15)
    cdfst = stats.loggamma.cdf(2,15)
    assert_almost_equal(cdfst, cdftr, 14)

def test_loglaplace():
    #if x is laplace then y = exp(x) is loglaplace
    #parameters are tricky
    #the stats.loglaplace parameter is the inverse scale of x
    loglaplaceexpg = ExpTransf_gen(stats.laplace)

    cdfst = stats.loglaplace.cdf(3,3)
    #0.98148148148148151
    #the parameters are shape, loc and scale of underlying laplace
    cdftr = loglaplaceexpg._cdf(3,0,1./3)
    assert_almost_equal(cdfst, cdftr, 14)

class CheckDistEquivalence:

    #no args, kwds yet

    def test_cdf(self):
        #'\nsquare of standard normal random variable is chisquare with dof=1 distributed'
        cdftr = self.dist.cdf(xx, *self.trargs, **self.trkwds)
        sfctr = 1-self.dist.sf(xx, *self.trargs, **self.trkwds) #sf complement
        cdfst = self.statsdist.cdf(xx, *self.stargs, **self.stkwds)
        assert_almost_equal(cdfst, cdftr, 14)
        assert_almost_equal(cdfst, sfctr, 14)

    def test_pdf(self):
        #'\nsquare of standard normal random variable is chisquare with dof=1 distributed'
        pdftr = self.dist.pdf(xx, *self.trargs, **self.trkwds)
        pdfst = self.statsdist.pdf(xx, *self.stargs, **self.stkwds)
        assert_almost_equal(pdfst, pdftr, 13)

    def test_ppf(self):
        #'\nsquare of standard normal random variable is chisquare with dof=1 distributed'
        ppftr = self.dist.ppf(ppfq, *self.trargs, **self.trkwds)
        ppfst = self.statsdist.ppf(ppfq, *self.stargs, **self.stkwds)
        assert_almost_equal(ppfst, ppftr, 13)

    def test_rvs(self):
        rvs = self.dist.rvs(*self.trargs, **{'size':100})
        mean_s = rvs.mean(0)
        mean_d, var_d = self.dist.stats(*self.trargs, **{'moments':'mv'})
        if np.any(np.abs(mean_d) < 1):
            assert_almost_equal(mean_d, mean_s, 1)
        else:
            assert_almost_equal(mean_s/mean_d, 1., 0) #tests 0.5<meanration<1.5

    def test_stats(self):
        trkwds = {'moments':'mvsk'}
        trkwds.update(self.stkwds)
        stkwds = {'moments':'mvsk'}
        stkwds.update(self.stkwds)
        mvsktr = np.array(self.dist.stats(*self.trargs, **trkwds))
        mvskst = np.array(self.statsdist.stats(*self.stargs, **stkwds))
        assert_almost_equal(mvskst[:2], mvsktr[:2], 8)
        if np.any(np.abs(mvskst[2:]) < 1):
            assert_almost_equal(mvskst[2:], mvsktr[2:], 1)
        else:
            assert_almost_equal(mvskst[2:]/mvsktr[2:], np.ones(2), 0)
            #tests 0.5<meanration<1.5



class TestLoggamma_1(CheckDistEquivalence):

    def __init__(self):
        self.dist = LogTransf_gen(stats.gamma)
        self.trargs = (10,)
        self.trkwds = {}
        self.statsdist = stats.loggamma
        self.stargs = (10,)
        self.stkwds = {}


class TestSquaredNormChi2_1(CheckDistEquivalence):

    def __init__(self):
        self.dist = squarenormalg
        self.trargs = ()
        self.trkwds = {}
        self.statsdist = stats.chi2
        self.stargs = (1,)
        self.stkwds = {}

class TestSquaredNormChi2_2(CheckDistEquivalence):

    def __init__(self):
        self.dist = squarenormalg
        self.trargs = ()
        self.trkwds = dict(loc=-10, scale=20)
        self.statsdist = stats.chi2
        self.stargs = (1,)
        self.stkwds = dict(loc=-10, scale=20)

class TestAbsNormHalfNorm(CheckDistEquivalence):

    def __init__(self):
        self.dist = absnormalg
        self.trargs = ()
        self.trkwds = {}
        self.statsdist = stats.halfnorm
        self.stargs = ()
        self.stkwds = {}

class TestSquaredTF(CheckDistEquivalence):

    def __init__(self):
        self.dist = squaretg
        self.trargs = (10,)
        self.trkwds = {}

        self.statsdist = stats.f
        self.stargs = (1,10)
        self.stkwds = {}

def test_squared_normal_chi2():
    #'\nsquare of standard normal random variable is chisquare with dof=1 distributed'
    cdftr = squarenormalg.cdf(xx,loc=l, scale=s)
    sfctr = 1-squarenormalg.sf(xx,loc=l, scale=s) #sf complement
    cdfst = stats.chi2.cdf(xx,1)
    assert_almost_equal(cdfst, cdftr, 14)
    assert_almost_equal(cdfst, sfctr, 14)

#    print('sqnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squarenormalg.pdf(xx,loc=l, scale=s)
#    print('chi2    pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.pdf(xx,1)
#    print('sqnorm  ppf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squarenormalg.ppf(ppfq,loc=l, scale=s)
#    print('chi2    ppf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.ppf(ppfq,1)
#    print('sqnorm  cdf with loc scale', squarenormalg.cdf(xx,loc=-10, scale=20)
#    print('chi2    cdf with loc scale', stats.chi2.cdf(xx,1,loc=-10, scale=20)



if __name__ == '__main__':

    #Examples for Transf2_gen, u- or hump shaped transformation
    #copied from transformtwo.py
    l,s = 0.0, 1.0
    ppfq = [0.1, 0.5, 0.9]
    xx = [0.95, 1.0, 1.1]
    nxx = [-0.95, -1.0, -1.1]
    print
    #print(invnormalg.__doc__
    print('\nsquare of standard normal random variable is chisquare with dof=1 distributed')
    print('sqnorm  cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squarenormalg.cdf(xx,loc=l, scale=s))
    print('sqnorm 1-sf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), 1-squarenormalg.sf(xx,loc=l, scale=s))
    print('chi2    cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.cdf(xx,1))
    print('sqnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squarenormalg.pdf(xx,loc=l, scale=s))
    print('chi2    pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.pdf(xx,1))
    print('sqnorm  ppf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squarenormalg.ppf(ppfq,loc=l, scale=s))
    print('chi2    ppf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.ppf(ppfq,1))
    print('sqnorm  cdf with loc scale', squarenormalg.cdf(xx,loc=-10, scale=20))
    print('chi2    cdf with loc scale', stats.chi2.cdf(xx,1,loc=-10, scale=20))
#    print('cdf for [0.5]:', squarenormalg.cdf(0.5,loc=l, scale=s))
#    print('chi square distribution')
#    print('chi2 pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.pdf(xx,1))
#    print('cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.cdf(xx,1))

    print('\nabsolute value of standard normal random variable is foldnorm(0) and ')
    print('halfnorm distributed:')
    print('absnorm  cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), absnormalg.cdf(xx,loc=l, scale=s))
    print('absnorm 1-sf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), 1-absnormalg.sf(xx,loc=l, scale=s))
    print('foldn    cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.foldnorm.cdf(xx,1e-5))
    print('halfn    cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.halfnorm.cdf(xx))
    print('absnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), absnormalg.pdf(xx,loc=l, scale=s))
    print('foldn    pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.foldnorm.pdf(xx,1e-5))
    print('halfn    pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.halfnorm.pdf(xx))
    print('absnorm  ppf for (%3.2f, %3.2f, %3.2f):' % tuple(ppfq), absnormalg.ppf(ppfq,loc=l, scale=s))
    print('foldn    ppf for (%3.2f, %3.2f, %3.2f):' % tuple(ppfq), stats.foldnorm.ppf(ppfq,1e-5))
    print('halfn    ppf for (%3.2f, %3.2f, %3.2f):' % tuple(ppfq), stats.halfnorm.ppf(ppfq))
#    print('cdf for [0.5]:', squarenormalg.cdf(0.5,loc=l, scale=s)
#    print('chi square distribution'
#    print('chi2 pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.pdf(xx,1)
#    print('cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.cdf(xx,1)

    print('\nnegative square of standard normal random variable is')
    print('1-chisquare with dof=1 distributed')
    print('this is mainly for testing')
    print('the following should be outside of the support - returns nan')
    print('nsqnorm  cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), negsquarenormalg.cdf(xx,loc=l, scale=s))
    print('nsqnorm 1-sf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), 1-negsquarenormalg.sf(xx,loc=l, scale=s))
    print('nsqnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), negsquarenormalg.pdf(xx,loc=l, scale=s))

    print('nsqnorm  cdf for (%3.2f, %3.2f, %3.2f):' % tuple(nxx), negsquarenormalg.cdf(nxx,loc=l, scale=s))
    print('nsqnorm 1-sf for (%3.2f, %3.2f, %3.2f):' % tuple(nxx), 1-negsquarenormalg.sf(nxx,loc=l, scale=s))
    print('chi2      sf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.sf(xx,1))
    print('nsqnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(nxx), negsquarenormalg.pdf(nxx,loc=l, scale=s))
    print('chi2     pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.chi2.pdf(xx,1))
    print('nsqnorm  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(nxx), negsquarenormalg.pdf(nxx,loc=l, scale=s))



    print('\nsquare of a t distributed random variable with dof=10 is')
    print('        F with dof=1,10 distributed')
    print('sqt  cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squaretg.cdf(xx,10))
    print('sqt 1-sf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), 1-squaretg.sf(xx,10))
    print('f    cdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.f.cdf(xx,1,10))
    print('sqt  pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), squaretg.pdf(xx,10))
    print('f    pdf for (%3.2f, %3.2f, %3.2f):' % tuple(xx), stats.f.pdf(xx,1,10))
    print('sqt  ppf for (%3.2f, %3.2f, %3.2f):' % tuple(ppfq), squaretg.ppf(ppfq,10))
    print('f    ppf for (%3.2f, %3.2f, %3.2f):' % tuple(ppfq), stats.f.ppf(ppfq,1,10))
    print('sqt  cdf for 100:', squaretg.cdf(100,10))
    print('f    cdf for 100:', stats.f.cdf(100,1,10))
    print('sqt  stats:', squaretg.stats(10, moments='mvsk'))
    print('f    stats:', stats.f.stats(1,10, moments='mvsk'))
    #Note the results differ for skew and kurtosis. I think the 3rd and 4th moment
    #    in the scipy.stats.f distribution is incorrect.
    # I corrected it now in stats.distributions.py in bzr branch
    v1=1
    v2=10
    g1 = 2*(v2+2*v1-2.)/(v2-6.)*np.sqrt(2*(v2-4.)/(v1*(v2+v1-2.)))
    g2 = 3/(2.*v2-16)*(8+g1*g1*(v2-6.))
    print('corrected skew, kurtosis of f(1,10) is', g1, g2)
    print(squarenormalg.rvs())
    print(squarenormalg.rvs(size=(2,4)))
    print('sqt random variables')
    print(stats.f.rvs(1,10,size=4))
    print(squaretg.rvs(10,size=4))

    #a large number check:
    np.random.seed(464239857)
    rvstsq = squaretg.rvs(10,size=100000)
    squaretg.moment(4,10)
    (rvstsq**4).mean()
    squaretg.moment(3,10)
    (rvstsq**3).mean()
    squaretg.stats(10, moments='mvsk')
    stats.describe(rvstsq)

    '''
    >>> np.random.seed(464239857)
    >>> rvstsq = squaretg.rvs(10,size=100000)
    >>> squaretg.moment(4,10)
    2734.3750000000009
    >>> (rvstsq**4).mean()
    2739.672765170933
    >>> squaretg.moment(3,10)
    78.124999999997044
    >>> (rvstsq**3).mean()
    84.13950048850549
    >>> squaretg.stats(10, moments='mvsk')
    (array(1.2500000000000022), array(4.6874999999630909), array(5.7735026919777912), array(106.00000000170148))
    >>> stats.describe(rvstsq)
    (100000, (3.2953470738423724e-009, 92.649615690914473), 1.2534924690963247, 4.7741427958594098, 6.1562177957041895, 100.99331166052181)
    '''
    # checking the distribution
    # fraction of observations in each decile
    dec = squaretg.ppf(np.linspace(0.,1,11),10)
    freq,edges = np.histogram(rvstsq, bins=dec)
    print(freq/float(len(rvstsq)))

    import matplotlib.pyplot as plt
    freq,edges,_ = plt.hist(rvstsq, bins=50, range=(0,4),normed=True)
    edges += (edges[1]-edges[0])/2.0
    plt.plot(edges[:-1], squaretg.pdf(edges[:-1], 10), 'r')
    #plt.show()
    #plt.close()

    '''
    >>> plt.plot(edges[:-1], squaretg.pdf(edges[:-1], 10), 'r')
    [<matplotlib.lines.Line2D object at 0x06EBFDB0>]
    >>> plt.fill(edges[4:8], squaretg.pdf(edges[4:8], 10), 'r')
    [<matplotlib.patches.Polygon object at 0x0725BA90>]
    >>> plt.show()
    >>> plt.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), y2=0, 'r')
    SyntaxError: non-keyword arg after keyword arg (<console>, line 1)
    >>> plt.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    AttributeError: 'module' object has no attribute 'fill_between'
    >>> fig = figure()
    Traceback (most recent call last):
    NameError: name 'figure' is not defined
    >>> ax1 = fig.add_subplot(311)
    Traceback (most recent call last):
    NameError: name 'fig' is not defined
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(111)
    >>> ax1.fill_between(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    AttributeError: 'AxesSubplot' object has no attribute 'fill_between'
    >>> ax1.fill(edges[4:8], squaretg.pdf(edges[4:8], 10), 0, 'r')
    Traceback (most recent call last):
    '''

    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
