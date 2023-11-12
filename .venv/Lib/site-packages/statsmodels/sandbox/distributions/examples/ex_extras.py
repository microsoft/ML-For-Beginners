# -*- coding: utf-8 -*-
"""

Created on Wed Feb 19 12:39:49 2014

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

from statsmodels.sandbox.distributions.extras import (SkewNorm_gen, skewnorm,
                                ACSkewT_gen,
                                NormExpan_gen, pdf_moments,
                                ExpTransf_gen, LogTransf_gen)
from statsmodels.stats.moment_helpers import mc2mvsk, mnc2mc, mvsk2mnc


def example_n():

    print(skewnorm.pdf(1,0), stats.norm.pdf(1), skewnorm.pdf(1,0) - stats.norm.pdf(1))
    print(skewnorm.pdf(1,1000), stats.chi.pdf(1,1), skewnorm.pdf(1,1000) - stats.chi.pdf(1,1))
    print(skewnorm.pdf(-1,-1000), stats.chi.pdf(1,1), skewnorm.pdf(-1,-1000) - stats.chi.pdf(1,1))
    rvs = skewnorm.rvs(0,size=500)
    print('sample mean var: ', rvs.mean(), rvs.var())
    print('theoretical mean var', skewnorm.stats(0))
    rvs = skewnorm.rvs(5,size=500)
    print('sample mean var: ', rvs.mean(), rvs.var())
    print('theoretical mean var', skewnorm.stats(5))
    print(skewnorm.cdf(1,0), stats.norm.cdf(1), skewnorm.cdf(1,0) - stats.norm.cdf(1))
    print(skewnorm.cdf(1,1000), stats.chi.cdf(1,1), skewnorm.cdf(1,1000) - stats.chi.cdf(1,1))
    print(skewnorm.sf(0.05,1000), stats.chi.sf(0.05,1), skewnorm.sf(0.05,1000) - stats.chi.sf(0.05,1))


def example_T():
    skewt = ACSkewT_gen()
    rvs = skewt.rvs(10,0,size=500)
    print('sample mean var: ', rvs.mean(), rvs.var())
    print('theoretical mean var', skewt.stats(10,0))
    print('t mean var', stats.t.stats(10))
    print(skewt.stats(10,1000)) # -> folded t distribution, as alpha -> inf
    rvs = np.abs(stats.t.rvs(10,size=1000))
    print(rvs.mean(), rvs.var())



def examples_normexpand():
    skewnorm = SkewNorm_gen()
    rvs = skewnorm.rvs(5,size=100)
    normexpan = NormExpan_gen(rvs, mode='sample')

    smvsk = stats.describe(rvs)[2:]
    print('sample: mu,sig,sk,kur')
    print(smvsk)

    dmvsk = normexpan.stats(moments='mvsk')
    print('normexpan: mu,sig,sk,kur')
    print(dmvsk)
    print('mvsk diff distribution - sample')
    print(np.array(dmvsk) - np.array(smvsk))
    print('normexpan attributes mvsk')
    print(mc2mvsk(normexpan.cnt))
    print(normexpan.mvsk)

    mnc = mvsk2mnc(dmvsk)
    mc = mnc2mc(mnc)
    print('central moments')
    print(mc)
    print('non-central moments')
    print(mnc)


    pdffn = pdf_moments(mc)
    print('\npdf approximation from moments')
    print('pdf at', mc[0]-1,mc[0]+1)
    print(pdffn([mc[0]-1,mc[0]+1]))
    print(normexpan.pdf([mc[0]-1,mc[0]+1]))


def examples_transf():
    ##lognormal = ExpTransf(a=0.0, xa=-10.0, name = 'Log transformed normal')
    ##print(lognormal.cdf(1))
    ##print(stats.lognorm.cdf(1,1))
    ##print(lognormal.stats())
    ##print(stats.lognorm.stats(1))
    ##print(lognormal.rvs(size=10))

    print('Results for lognormal')
    lognormalg = ExpTransf_gen(stats.norm, a=0, name = 'Log transformed normal general')
    print(lognormalg.cdf(1))
    print(stats.lognorm.cdf(1,1))
    print(lognormalg.stats())
    print(stats.lognorm.stats(1))
    print(lognormalg.rvs(size=5))

    ##print('Results for loggamma')
    ##loggammag = ExpTransf_gen(stats.gamma)
    ##print(loggammag._cdf(1,10))
    ##print(stats.loggamma.cdf(1,10))

    print('Results for expgamma')
    loggammaexpg = LogTransf_gen(stats.gamma)
    print(loggammaexpg._cdf(1,10))
    print(stats.loggamma.cdf(1,10))
    print(loggammaexpg._cdf(2,15))
    print(stats.loggamma.cdf(2,15))


    # this requires change in scipy.stats.distribution
    #print(loggammaexpg.cdf(1,10))

    print('Results for loglaplace')
    loglaplaceg = LogTransf_gen(stats.laplace)
    print(loglaplaceg._cdf(2))
    print(stats.loglaplace.cdf(2,1))
    loglaplaceexpg = ExpTransf_gen(stats.laplace)
    print(loglaplaceexpg._cdf(2))
    stats.loglaplace.cdf(3,3)
    #0.98148148148148151
    loglaplaceexpg._cdf(3,0,1./3)
    #0.98148148148148151

if __name__ == '__main__':
    example_n()
    example_T()
    examples_normexpand()
    examples_transf()
