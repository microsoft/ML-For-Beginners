# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:13:36 2011

@author: josef
"""

import numpy as np
from numpy.testing import assert_, assert_almost_equal

from statsmodels.sandbox.distributions.extras import (skewnorm,
                skewnorm2, ACSkewT_gen)


def test_skewnorm():
    #library("sn")
    #dsn(c(-2,-1,0,1,2), shape=10)
    #psn(c(-2,-1,0,1,2), shape=10)
    #noquote(sprintf("%.15e,", snp))
    pdf_r = np.array([2.973416551551523e-90, 3.687562713971017e-24,
                      3.989422804014327e-01, 4.839414490382867e-01,
                      1.079819330263761e-01])
    pdf_sn = skewnorm.pdf([-2,-1,0,1,2], 10)

    #res = (snp-snp_r)/snp
    assert_(np.allclose(pdf_sn, pdf_r,rtol=1e-13, atol=0))

    pdf_sn2 = skewnorm2.pdf([-2,-1,0,1,2], 10)
    assert_(np.allclose(pdf_sn2, pdf_r, rtol=1e-13, atol=0))


    cdf_r = np.array([0.000000000000000e+00, 0.000000000000000e+00,
                      3.172551743055357e-02, 6.826894921370859e-01,
                      9.544997361036416e-01])
    cdf_sn = skewnorm.cdf([-2,-1,0,1,2], 10)
    maxabs = np.max(np.abs(cdf_sn - cdf_r))
    maxrel = np.max(np.abs(cdf_sn - cdf_r)/(cdf_r+1e-50))
    msg = "maxabs=%15.13g, maxrel=%15.13g\n%r\n%r" % (maxabs, maxrel, cdf_sn,
                                                       cdf_r)
    #assert_(np.allclose(cdf_sn, cdf_r, rtol=1e-13, atol=1e-25), msg=msg)
    assert_almost_equal(cdf_sn, cdf_r, decimal=10)

    cdf_sn2 = skewnorm2.cdf([-2,-1,0,1,2], 10)
    maxabs = np.max(np.abs(cdf_sn2 - cdf_r))
    maxrel = np.max(np.abs(cdf_sn2 - cdf_r)/(cdf_r+1e-50))
    msg = "maxabs=%15.13g, maxrel=%15.13g" % (maxabs, maxrel)
    #assert_(np.allclose(cdf_sn2, cdf_r, rtol=1e-13, atol=1e-25), msg=msg)
    assert_almost_equal(cdf_sn2, cdf_r, decimal=10, err_msg=msg)


def test_skewt():
    skewt = ACSkewT_gen()
    x = [-2, -1, -0.5, 0, 1, 2]
    #noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10)))
    #default in R:sn is df=inf
    pdf_r = np.array([2.973416551551523e-90, 3.687562713971017e-24,
                      2.018401586422970e-07, 3.989422804014327e-01,
                      4.839414490382867e-01, 1.079819330263761e-01])
    pdf_st = skewt.pdf(x, 1000000, 10)
    pass
    np.allclose(pdf_st, pdf_r, rtol=0, atol=1e-6)
    np.allclose(pdf_st, pdf_r, rtol=1e-1, atol=0)


    #noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10)))
    cdf_r = np.array([0.000000000000000e+00, 0.000000000000000e+00,
                      3.729478836866917e-09, 3.172551743055357e-02,
                      6.826894921370859e-01, 9.544997361036416e-01])
    cdf_st = skewt.cdf(x, 1000000, 10)
    np.allclose(cdf_st, cdf_r, rtol=0, atol=1e-6)
    np.allclose(cdf_st, cdf_r, rtol=1e-1, atol=0)
    #assert_(np.allclose(cdf_st, cdf_r, rtol=1e-13, atol=1e-15))


    #noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10, df=5)))
    pdf_r = np.array([2.185448836190663e-07, 1.272381597868587e-05,
                      5.746937644959992e-04, 3.796066898224945e-01,
                      4.393468708859825e-01, 1.301804021075493e-01])
    pdf_st = skewt.pdf(x, 5, 10)  #args = (df, alpha)
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))

    #noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10, df=5)))
    cdf_r = np.array([8.822783669199699e-08, 2.638467463775795e-06,
                      6.573106017198583e-05, 3.172551743055352e-02,
                      6.367851708183412e-01, 8.980606093979784e-01])
    cdf_st = skewt.cdf(x, 5, 10)  #args = (df, alpha)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-10, atol=0))


    #noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10, df=1)))
    pdf_r = np.array([3.941955996757291e-04, 1.568067236862745e-03,
                      6.136996029432048e-03, 3.183098861837907e-01,
                      3.167418189469279e-01, 1.269297588738406e-01])
    pdf_st = skewt.pdf(x, 1, 10)  #args = (df, alpha) = (1, 10))
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))

    #noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10, df=1)))
    cdf_r = np.array([7.893671370544414e-04, 1.575817262600422e-03,
                      3.128720749105560e-03, 3.172551743055351e-02,
                      5.015758172626005e-01, 7.056221318361879e-01])
    cdf_st = skewt.cdf(x, 1, 10)  #args = (df, alpha) = (1, 10)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-13, atol=1e-25))



if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
    print('Done')


'''
>>> skewt.pdf([-2,-1,0,1,2], 10000000, 10)
array([  2.98557345e-90,   3.68850289e-24,   3.98942271e-01,
         4.83941426e-01,   1.07981952e-01])
>>> skewt.pdf([-2,-1,0,1,2], np.inf, 10)
array([ nan,  nan,  nan,  nan,  nan])
'''
