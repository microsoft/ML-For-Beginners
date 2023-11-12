# -*- coding: utf-8 -*-
"""


Created on Sun May 09 22:35:21 2010
Author: josef-pktd
License: BSD

todo:
change moment calculation, (currently uses default _ppf method - I think)
# >>> lognormalg.moment(4)
Warning: The algorithm does not converge.  Roundoff error is detected
  in the extrapolation table.  It is assumed that the requested tolerance
  cannot be achieved, and that the returned result (if full_output = 1) is
  the best which can be obtained.
array(2981.0032380193438)
"""
import warnings # for silencing, see above...
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats, special
from statsmodels.sandbox.distributions.extras import (
    squarenormalg, absnormalg, negsquarenormalg, squaretg)


# some patches to scipy.stats.distributions so tests work and pass
# this should be necessary only for older scipy

#patch frozen distributions with a name
stats.distributions.rv_frozen.name = property(lambda self: self.dist.name)

#patch f distribution, correct skew and maybe kurtosis
def f_stats(self, dfn, dfd):
    arr, where, inf, sqrt, nan = np.array, np.where, np.inf, np.sqrt, np.nan
    v2 = arr(dfd*1.0)
    v1 = arr(dfn*1.0)
    mu = where(v2 > 2, v2 / arr(v2 - 2), inf)
    mu2 = 2*v2*v2*(v2+v1-2)/(v1*(v2-2)**2 * (v2-4))
    mu2 = where(v2 > 4, mu2, inf)
    #g1 = 2*(v2+2*v1-2)/(v2-6)*sqrt((2*v2-4)/(v1*(v2+v1-2)))
    g1 = 2*(v2+2*v1-2.)/(v2-6.)*np.sqrt(2*(v2-4.)/(v1*(v2+v1-2.)))
    g1 = where(v2 > 6, g1, nan)
    #g2 = 3/(2*v2-16)*(8+g1*g1*(v2-6))
    g2 = 3/(2.*v2-16)*(8+g1*g1*(v2-6.))
    g2 = where(v2 > 8, g2, nan)
    return mu, mu2, g1, g2

#stats.distributions.f_gen._stats = f_stats
stats.f.__class__._stats = f_stats

#correct kurtosis by subtracting 3 (Fisher)
#after this it matches halfnorm for arg close to zero
def foldnorm_stats(self, c):
    arr, where, inf, sqrt, nan = np.array, np.where, np.inf, np.sqrt, np.nan
    exp = np.exp
    pi = np.pi

    fac = special.erf(c/sqrt(2))
    mu = sqrt(2.0/pi)*exp(-0.5*c*c)+c*fac
    mu2 = c*c + 1 - mu*mu
    c2 = c*c
    g1 = sqrt(2/pi)*exp(-1.5*c2)*(4-pi*exp(c2)*(2*c2+1.0))
    g1 += 2*c*fac*(6*exp(-c2) + 3*sqrt(2*pi)*c*exp(-c2/2.0)*fac + \
                   pi*c*(fac*fac-1))
    g1 /= pi*mu2**1.5

    g2 = c2*c2+6*c2+3+6*(c2+1)*mu*mu - 3*mu**4
    g2 -= 4*exp(-c2/2.0)*mu*(sqrt(2.0/pi)*(c2+2)+c*(c2+3)*exp(c2/2.0)*fac)
    g2 /= mu2**2.0
    g2 -= 3.
    return mu, mu2, g1, g2

#stats.distributions.foldnorm_gen._stats = foldnorm_stats
stats.foldnorm.__class__._stats = foldnorm_stats


#-----------------------------

DECIMAL = 5

class Test_Transf2:

    @classmethod
    def setup_class(cls):
        cls.dist_equivalents = [
            #transf, stats.lognorm(1))
            #The below fails on the SPARC box with scipy 10.1
            #(lognormalg, stats.lognorm(1)),
            #transf2
            (squarenormalg, stats.chi2(1)),
            (absnormalg, stats.halfnorm),
            (absnormalg, stats.foldnorm(1e-5)),  #try frozen
            #(negsquarenormalg, 1-stats.chi2),  # will not work as distribution
            (squaretg(10), stats.f(1, 10))
        ]      #try both frozen

        l,s = 0.0, 1.0
        cls.ppfq = [0.1,0.5,0.9]
        cls.xx = [0.95,1.0,1.1]
        cls.nxx = [-0.95,-1.0,-1.1]

    def test_equivalent(self):
        xx, ppfq = self.xx, self.ppfq
        for j,(d1,d2) in enumerate(self.dist_equivalents):
##            print d1.name
            assert_almost_equal(d1.cdf(xx), d2.cdf(xx), err_msg='cdf'+d1.name)
            assert_almost_equal(d1.pdf(xx), d2.pdf(xx),
                                err_msg='pdf '+d1.name+d2.name)
            assert_almost_equal(d1.sf(xx), d2.sf(xx),
                                err_msg='sf '+d1.name+d2.name)
            assert_almost_equal(d1.ppf(ppfq), d2.ppf(ppfq),
                                err_msg='ppq '+d1.name+d2.name)
            assert_almost_equal(d1.isf(ppfq), d2.isf(ppfq),
                                err_msg='isf '+d1.name+d2.name)
            self.d1 = d1
            self.d2 = d2
##            print d1, d2
##            print d1.moment(3)
##            print d2.moment(3)
            #work around bug#1293
            if hasattr(d2, 'dist'):
                d2mom = d2.dist.moment(3, *d2.args)
            else:
                d2mom = d2.moment(3)
            if j==3:
                print("now")
            assert_almost_equal(d1.moment(3), d2mom,
                                DECIMAL,
                                err_msg='moment '+d1.name+d2.name)
            # silence warnings in scipy, works for versions
            # after print changed to warning in scipy
            orig_filter = warnings.filters[:]
            warnings.simplefilter('ignore')
            try:
                s1 = d1.stats(moments='mvsk')
                s2 = d2.stats(moments='mvsk')
            finally:
                warnings.filters = orig_filter
            #stats(moments='k') prints warning for lognormalg
            assert_almost_equal(s1[:2], s2[:2],
                                err_msg='stats '+d1.name+d2.name)
            assert_almost_equal(s1[2:], s2[2:],
                                decimal=2, #lognorm for kurtosis
                                err_msg='stats '+d1.name+d2.name)



    def test_equivalent_negsq(self):
        #special case negsquarenormalg
        #negsquarenormalg.cdf(x) == stats.chi2(1).cdf(-x), for x<=0

        xx, nxx, ppfq = self.xx, self.nxx, self.ppfq
        d1,d2 = (negsquarenormalg, stats.chi2(1))
        #print d1.name
        assert_almost_equal(d1.cdf(nxx), 1-d2.cdf(xx), err_msg='cdf'+d1.name)
        assert_almost_equal(d1.pdf(nxx), d2.pdf(xx))
        assert_almost_equal(d1.sf(nxx), 1-d2.sf(xx))
        assert_almost_equal(d1.ppf(ppfq), -d2.ppf(ppfq)[::-1])
        assert_almost_equal(d1.isf(ppfq), -d2.isf(ppfq)[::-1])
        assert_almost_equal(d1.moment(3), -d2.moment(3))
        ch2oddneg = [v*(-1)**(i+1) for i,v in
                     enumerate(d2.stats(moments='mvsk'))]
        assert_almost_equal(d1.stats(moments='mvsk'), ch2oddneg,
                            err_msg='stats '+d1.name+d2.name)


if __name__ == '__main__':
    tt = Test_Transf2()
    tt.test_equivalent()
    tt.test_equivalent_negsq()

    debug = 0
    if debug:
        print(negsquarenormalg.ppf([0.1,0.5,0.9]))
        print(stats.chi2.ppf([0.1,0.5,0.9],1))
        print(negsquarenormalg.a)
        print(negsquarenormalg.b)

        print(absnormalg.stats( moments='mvsk'))
        print(stats.foldnorm(1e-10).stats( moments='mvsk'))
        print(stats.halfnorm.stats( moments='mvsk'))
