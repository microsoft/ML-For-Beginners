# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:02:13 2011
@author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose

from statsmodels.sandbox.distributions.multivariate import (
    mvstdtprob, mvstdnormcdf)
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal


class Test_MVN_MVT_prob:
    #test for block integratal, cdf, of multivariate t and normal
    #comparison results from R

    @classmethod
    def setup_class(cls):
        cls.corr_equal = np.asarray([[1.0, 0.5, 0.5],[0.5,1,0.5],[0.5,0.5,1]])
        cls.a = -1 * np.ones(3)
        cls.b = 3 * np.ones(3)
        cls.df = 4

        corr2 = cls.corr_equal.copy()
        corr2[2,1] = -0.5
        cls.corr2 = corr2

    def test_mvn_mvt_1(self):
        a, b = self.a, self.b
        df = self.df
        corr_equal = self.corr_equal
        #result from R, mvtnorm with option
        #algorithm = GenzBretz(maxpts = 100000, abseps = 0.000001, releps = 0)
        #     or higher
        probmvt_R = 0.60414   #report, ed error approx. 7.5e-06
        probmvn_R = 0.673970  #reported error approx. 6.4e-07
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
        assert_almost_equal(probmvn_R,
                            mvstdnormcdf(a, b, corr_equal, abseps=1e-5), 4)

        mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-8, maxpts=10000000)
        assert_almost_equal(probmvn_R, mvn_high, 5)
        #this still barely fails sometimes at 6 why?? error is -7.2627419411830374e-007
        #>>> 0.67396999999999996 - 0.67397072627419408
        #-7.2627419411830374e-007
        #>>> assert_almost_equal(0.67396999999999996, 0.67397072627419408, 6)
        #Fail

    def test_mvn_mvt_2(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        probmvn_R = 0.6472497 #reported error approx. 7.7e-08
        probmvt_R = 0.5881863 #highest reported error up to approx. 1.99e-06
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr2, abseps=1e-5), 4)

    def test_mvn_mvt_3(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        a2 = a.copy()
        a2[:] = -np.inf
        # using higher precision in R, error approx. 6.866163e-07
        probmvn_R = 0.9961141
        # using higher precision in R, error approx. 1.6e-07
        probmvt_R = 0.9522146
        quadkwds = {'epsabs': 1e-08}
        probmvt = mvstdtprob(a2, b, corr2, df, quadkwds=quadkwds)
        assert_allclose(probmvt_R, probmvt, atol=5e-4)
        probmvn = mvstdnormcdf(a2, b, corr2, maxpts=100000, abseps=1e-5)
        assert_allclose(probmvn_R, probmvn, atol=1e-4)

    def test_mvn_mvt_4(self):
        a, bl = self.a, self.b
        df = self.df
        corr2 = self.corr2

        #from 0 to inf
        #print '0 inf'
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.1666667 #error approx. 6.1e-08
        probmvt_R = 0.1666667 #error approx. 8.2e-08
        assert_almost_equal(probmvt_R, mvstdtprob(np.zeros(3), -a2, corr2, df), 4)
        assert_almost_equal(probmvn_R,
                            mvstdnormcdf(np.zeros(3), -a2, corr2,
                                         maxpts=100000, abseps=1e-5), 4)

    def test_mvn_mvt_5(self):
        a, bl = self.a, self.b
        df = self.df
        corr2 = self.corr2

        #unequal integration bounds
        #print "ue"
        a3 = np.array([0.5, -0.5, 0.5])
        probmvn_R = 0.06910487 #using higher precision in R, error approx. 3.5e-08
        probmvt_R = 0.05797867 #using higher precision in R, error approx. 5.8e-08
        assert_almost_equal(mvstdtprob(a3, a3+1, corr2, df), probmvt_R, 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a3, a3+1, corr2,
                                                maxpts=100000, abseps=1e-5), 4)


class TestMVDistributions:
    #this is not well organized

    @classmethod
    def setup_class(cls):
        covx = np.array([[1.0, 0.5], [0.5, 1.0]])
        mu3 = [-1, 0., 2.]
        cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                         [ 0.5 ,  1.5 ,  0.6 ],
                         [ 0.75,  0.6 ,  2.  ]])
        cls.mu3 = mu3
        cls.cov3 = cov3

        mvn3 = MVNormal(mu3, cov3)
        mvn3c = MVNormal(np.array([0,0,0]), cov3)
        cls.mvn3 = mvn3
        cls.mvn3c = mvn3c

    def test_mvn_pdf(self):
        cov3 = self.cov3
        mvn3 = self.mvn3

        r_val = [
            -7.667977543898155, -6.917977543898155, -5.167977543898155
        ]
        assert_allclose(mvn3.logpdf(cov3), r_val, rtol=1e-13)

        r_val = [
            0.000467562492721686, 0.000989829804859273, 0.005696077243833402
        ]
        assert_allclose(mvn3.pdf(cov3), r_val, rtol=1e-13)

        mvn3b = MVNormal(np.array([0, 0, 0]), cov3)
        r_val = [
            0.02914269740502042, 0.02269635555984291, 0.01767593948287269
        ]
        assert_allclose(mvn3b.pdf(cov3), r_val, rtol=1e-13)

    def test_mvt_pdf(self, reset_randomstate):
        cov3 = self.cov3
        mu3 = self.mu3

        mvt = MVT((0, 0), 1, 5)
        assert_almost_equal(mvt.logpdf(np.array([0., 0.])), -1.837877066409345,
                            decimal=15)
        assert_almost_equal(mvt.pdf(np.array([0., 0.])), 0.1591549430918953,
                            decimal=15)

        mvt.logpdf(np.array([1., 1.])) - (-3.01552989458359)

        mvt1 = MVT((0, 0), 1, 1)
        mvt1.logpdf(np.array([1., 1.])) - (-3.48579549941151)  # decimal=16

        rvs = mvt.rvs(100000)
        assert_almost_equal(np.cov(rvs, rowvar=False), mvt.cov, decimal=1)

        mvt31 = MVT(mu3, cov3, 1)
        assert_almost_equal(mvt31.pdf(cov3),
                            [0.0007276818698165781, 0.0009980625182293658,
                             0.0027661422056214652],
                            decimal=17)

        mvt = MVT(mu3, cov3, 3)
        assert_almost_equal(mvt.pdf(cov3),
                            [0.000863777424247410, 0.001277510788307594,
                             0.004156314279452241],
                            decimal=17)


if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '-vvs', '-x', '--pdb'])
