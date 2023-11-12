'''tests for pca and arma to ar and ma representation

compared with matlab princomp, and garchar, garchma

TODO:
* convert to generators with yield to have individual tests
* incomplete: test relationship of pca-evecs and pinv (adding constant)
'''

import numpy as np
from numpy.testing import assert_array_almost_equal
from statsmodels.sandbox.tools import pca, pcasvd

from statsmodels.multivariate.tests.results.datamlw import (
    princomp1, princomp2, princomp3, data)


def check_pca_princomp(pcares, princomp):
    factors, evals, evecs = pcares[1:]
    #res_princomp.coef, res_princomp.factors, res_princomp.values
    msign = (evecs/princomp.coef)[0]
    assert_array_almost_equal(msign*evecs, princomp.coef, 13)
    assert_array_almost_equal(msign*factors, princomp.factors, 13)
    assert_array_almost_equal(evals, princomp.values.ravel(), 13)

def check_pca_svd(pcares, pcasvdres):
    xreduced, factors, evals, evecs  = pcares
    xred_svd, factors_svd, evals_svd, evecs_svd = pcasvdres
    assert_array_almost_equal(evals_svd, evals, 14)
    msign = (evecs/evecs_svd)[0]
    assert_array_almost_equal(msign*evecs_svd, evecs, 13)
    assert_array_almost_equal(msign*factors_svd, factors, 13)
    assert_array_almost_equal(xred_svd, xreduced, 13)


xf = data.xo/1000.

def test_pca_princomp():
    pcares = pca(xf)
    check_pca_princomp(pcares, princomp1)
    pcares = pca(xf[:20,:])
    check_pca_princomp(pcares, princomp2)
    pcares = pca(xf[:20,:]-xf[:20,:].mean(0))
    check_pca_princomp(pcares, princomp3)
    pcares = pca(xf[:20,:]-xf[:20,:].mean(0), demean=0)
    check_pca_princomp(pcares, princomp3)


def test_pca_svd():
    xreduced, factors, evals, evecs  = pca(xf)
    factors_wconst = np.c_[factors, np.ones((factors.shape[0],1))]
    beta = np.dot(np.linalg.pinv(factors_wconst), xf)
    #np.dot(np.linalg.pinv(factors_wconst),x2/1000.).T[:,:4] - evecs
    assert_array_almost_equal(beta.T[:,:4], evecs, 14)

    xred_svd, factors_svd, evals_svd, evecs_svd = pcasvd(xf, keepdim=0)
    assert_array_almost_equal(evals_svd, evals, 14)
    msign = (evecs/evecs_svd)[0]
    assert_array_almost_equal(msign*evecs_svd, evecs, 13)
    assert_array_almost_equal(msign*factors_svd, factors, 12)
    assert_array_almost_equal(xred_svd, xreduced, 13)

    pcares = pca(xf, keepdim=2)
    pcasvdres = pcasvd(xf, keepdim=2)
    check_pca_svd(pcares, pcasvdres)

#print np.dot(factors[:,:3], evecs.T[:3,:])[:5]


if __name__ == '__main__':
    test_pca_svd()
