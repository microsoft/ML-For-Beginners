# -*- coding: utf-8 -*-
"""Tests for finding a positive semi-definite correlation or covariance matrix

Created on Mon May 27 12:07:02 2013

Author: Josef Perktold
"""
import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest

from statsmodels.stats.correlation_tools import (
    corr_nearest, corr_clipped, cov_nearest,
    _project_correlation_factors, corr_nearest_factor, _spg_optim,
    corr_thresholded, cov_nearest_factor_homog, FactoredPSDMatrix)
from statsmodels.tools.testing import Holder


def norm_f(x, y):
    '''Frobenious norm (squared sum) of difference between two arrays
    '''
    d = ((x - y)**2).sum()
    return np.sqrt(d)


# R library Matrix results
cov1_r = Holder()
#> nc  <- nearPD(pr, conv.tol = 1e-7, keepDiag = TRUE, doDykstra =FALSE, corr=TRUE)
#> cat_items(nc, prefix="cov1_r.")
cov1_r.mat = '''<S4 object of class structure("dpoMatrix", package = "Matrix")>'''
cov1_r.eigenvalues = np.array([
     4.197315628646795, 0.7540460243978023, 0.5077608149667492,
     0.3801267599652769, 0.1607508970775889, 4.197315628646795e-08
    ])
cov1_r.corr = '''TRUE'''
cov1_r.normF = 0.0743805226512533
cov1_r.iterations = 11
cov1_r.rel_tol = 8.288594638441735e-08
cov1_r.converged = '''TRUE'''
#> mkarray2(as.matrix(nc$mat), name="cov1_r.mat")
cov1_r.mat = np.array([
     1, 0.487968018215892, 0.642651880010906, 0.4906386709070835,
     0.6440990530811909, 0.8087111845493985, 0.487968018215892, 1,
     0.5141147294352735, 0.2506688108312097, 0.672351311297074,
     0.725832055882795, 0.642651880010906, 0.5141147294352735, 1,
     0.596827778712154, 0.5821917790519067, 0.7449631633814129,
     0.4906386709070835, 0.2506688108312097, 0.596827778712154, 1,
     0.729882058012399, 0.772150225146826, 0.6440990530811909,
     0.672351311297074, 0.5821917790519067, 0.729882058012399, 1,
     0.813191720191944, 0.8087111845493985, 0.725832055882795,
     0.7449631633814129, 0.772150225146826, 0.813191720191944, 1
    ]).reshape(6,6, order='F')


cov_r = Holder()
#nc  <- nearPD(pr+0.01*diag(6), conv.tol = 1e-7, keepDiag = TRUE, doDykstra =FALSE, corr=FALSE)
#> cat_items(nc, prefix="cov_r.")
#cov_r.mat = '''<S4 object of class structure("dpoMatrix", package = "Matrix")>'''
cov_r.eigenvalues = np.array([
     4.209897516692652, 0.7668341923072066, 0.518956980021938,
     0.390838551407132, 0.1734728460460068, 4.209897516692652e-08
    ])
cov_r.corr = '''FALSE'''
cov_r.normF = 0.0623948693159157
cov_r.iterations = 11
cov_r.rel_tol = 5.83987595937896e-08
cov_r.converged = '''TRUE'''

#> mkarray2(as.matrix(nc$mat), name="cov_r.mat")
cov_r.mat = np.array([
     1.01, 0.486207476951913, 0.6428524769306785, 0.4886092840296514,
     0.645175579158233, 0.811533860074678, 0.486207476951913, 1.01,
     0.514394615153752, 0.2478398278204047, 0.673852495852274,
     0.7297661648968664, 0.6428524769306785, 0.514394615153752, 1.01,
     0.5971503271420517, 0.582018469844712, 0.7445177382760834,
     0.4886092840296514, 0.2478398278204047, 0.5971503271420517, 1.01,
     0.73161232298669, 0.7766852947049376, 0.645175579158233,
     0.673852495852274, 0.582018469844712, 0.73161232298669, 1.01,
     0.8107916469252828, 0.811533860074678, 0.7297661648968664,
     0.7445177382760834, 0.7766852947049376, 0.8107916469252828, 1.01
    ]).reshape(6,6, order='F')


def test_corr_psd():
    # test positive definite matrix is unchanged
    x = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])

    y = corr_nearest(x, n_fact=100)
    #print np.max(np.abs(x - y))
    assert_almost_equal(x, y, decimal=14)

    y = corr_clipped(x)
    assert_almost_equal(x, y, decimal=14)

    y = cov_nearest(x, n_fact=100)
    assert_almost_equal(x, y, decimal=14)

    x2 = x + 0.001 * np.eye(3)
    y = cov_nearest(x2, n_fact=100)
    assert_almost_equal(x2, y, decimal=14)


class CheckCorrPSDMixin:

    def test_nearest(self):
        x = self.x
        res_r = self.res
        y = corr_nearest(x, threshold=1e-7, n_fact=100)
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=3)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.0015)
        evals = np.linalg.eigvalsh(y)
        #print 'evals', evals / res_r.eigenvalues[::-1] - 1
        assert_allclose(evals, res_r.eigenvalues[::-1], rtol=0.003, atol=1e-7)
        #print evals[0] / 1e-7 - 1
        assert_allclose(evals[0], 1e-7, rtol=1e-6)

    def test_clipped(self):
        x = self.x
        res_r = self.res
        y = corr_clipped(x, threshold=1e-7)
        #print np.max(np.abs(x - y)), np.max(np.abs((x - y) / y))
        assert_almost_equal(y, res_r.mat, decimal=1)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.15)

        evals = np.linalg.eigvalsh(y)
        assert_allclose(evals, res_r.eigenvalues[::-1], rtol=0.1, atol=1e-7)
        assert_allclose(evals[0], 1e-7, rtol=0.02)

    def test_cov_nearest(self):
        x = self.x
        res_r = self.res
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = cov_nearest(x, method='nearest', threshold=1e-7)
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=2)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.0015)


class TestCovPSD:

    @classmethod
    def setup_class(cls):
        x = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                       0.477, 1,     0.516, 0.233, 0.682, 0.75,
                       0.644, 0.516, 1,     0.599, 0.581, 0.742,
                       0.478, 0.233, 0.599, 1,     0.741, 0.8,
                       0.651, 0.682, 0.581, 0.741, 1,     0.798,
                       0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)
        cls.x = x + 0.01 * np.eye(6)
        cls.res = cov_r

    def test_cov_nearest(self):
        x = self.x
        res_r = self.res
        y = cov_nearest(x, method='nearest')
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=3)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.001)

        y = cov_nearest(x, method='clipped')
        #print np.max(np.abs(x - y))
        assert_almost_equal(y, res_r.mat, decimal=2)
        d = norm_f(x, y)
        assert_allclose(d, res_r.normF, rtol=0.15)


class TestCorrPSD1(CheckCorrPSDMixin):

    @classmethod
    def setup_class(cls):
        x = np.array([ 1,     0.477, 0.644, 0.478, 0.651, 0.826,
                       0.477, 1,     0.516, 0.233, 0.682, 0.75,
                       0.644, 0.516, 1,     0.599, 0.581, 0.742,
                       0.478, 0.233, 0.599, 1,     0.741, 0.8,
                       0.651, 0.682, 0.581, 0.741, 1,     0.798,
                       0.826, 0.75,  0.742, 0.8,   0.798, 1]).reshape(6,6)
        cls.x = x
        cls.res = cov1_r


@pytest.mark.parametrize('threshold', [0, 1e-15, 1e-10, 1e-6])
def test_corrpsd_threshold(threshold):
    x = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])

    y = corr_nearest(x, n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=1e-6, atol=1e-15)

    y = corr_clipped(x, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)

    y = cov_nearest(x, method='nearest', n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=1e-6, atol=1e-15)

    y = cov_nearest(x, n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)


class Test_Factor:

    def test_corr_nearest_factor_arrpack(self):

        # regression results for svds call
        u2 = np.array([[
         6.39407581e-19,   9.15225947e-03,   1.82631698e-02,
         2.72917181e-02,   3.61975557e-02,   4.49413101e-02,
         5.34848732e-02,   6.17916613e-02,   6.98268388e-02,
         7.75575058e-02,   8.49528448e-02,   9.19842264e-02,
         9.86252769e-02,   1.04851906e-01,   1.10642305e-01,
         1.15976906e-01,   1.20838331e-01,   1.25211306e-01,
         1.29082570e-01,   1.32440778e-01,   1.35276397e-01,
         1.37581605e-01,   1.39350201e-01,   1.40577526e-01,
         1.41260396e-01,   1.41397057e-01,   1.40987160e-01,
         1.40031756e-01,   1.38533306e-01,   1.36495727e-01,
         1.33924439e-01,   1.30826443e-01,   1.27210404e-01,
         1.23086750e-01,   1.18467769e-01,   1.13367717e-01,
         1.07802909e-01,   1.01791811e-01,   9.53551023e-02,
         8.85157320e-02,   8.12989329e-02,   7.37322125e-02,
         6.58453049e-02,   5.76700847e-02,   4.92404406e-02,
         4.05921079e-02,   3.17624629e-02,   2.27902803e-02,
         1.37154584e-02,   4.57871801e-03,  -4.57871801e-03,
        -1.37154584e-02,  -2.27902803e-02,  -3.17624629e-02,
        -4.05921079e-02,  -4.92404406e-02,  -5.76700847e-02,
        -6.58453049e-02,  -7.37322125e-02,  -8.12989329e-02,
        -8.85157320e-02,  -9.53551023e-02,  -1.01791811e-01,
        -1.07802909e-01,  -1.13367717e-01,  -1.18467769e-01,
        -1.23086750e-01,  -1.27210404e-01,  -1.30826443e-01,
        -1.33924439e-01,  -1.36495727e-01,  -1.38533306e-01,
        -1.40031756e-01,  -1.40987160e-01,  -1.41397057e-01,
        -1.41260396e-01,  -1.40577526e-01,  -1.39350201e-01,
        -1.37581605e-01,  -1.35276397e-01,  -1.32440778e-01,
        -1.29082570e-01,  -1.25211306e-01,  -1.20838331e-01,
        -1.15976906e-01,  -1.10642305e-01,  -1.04851906e-01,
        -9.86252769e-02,  -9.19842264e-02,  -8.49528448e-02,
        -7.75575058e-02,  -6.98268388e-02,  -6.17916613e-02,
        -5.34848732e-02,  -4.49413101e-02,  -3.61975557e-02,
        -2.72917181e-02,  -1.82631698e-02,  -9.15225947e-03,
        -3.51829569e-17]]).T
        s2 = np.array([ 24.88812183])

        d = 100
        dm = 1

        # Construct a test matrix with exact factor structure
        X = np.zeros((d,dm), dtype=np.float64)
        x = np.linspace(0, 2*np.pi, d)
        for j in range(dm):
            X[:,j] = np.sin(x*(j+1))
        _project_correlation_factors(X)
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1.)

        from scipy.sparse.linalg import svds
        u, s, vt = svds(mat, dm)

        #difference in sign
        dsign = np.sign(u[1]) * np.sign(u2[1])

        assert_allclose(u, dsign * u2, rtol=1e-6, atol=1e-14)
        assert_allclose(s, s2, rtol=1e-6)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_corr_nearest_factor(self, dm):

        objvals = [np.array([6241.8, 6241.8, 579.4, 264.6, 264.3]),
                   np.array([2104.9, 2104.9, 710.5, 266.3, 286.1])]

        d = 100

        # Construct a test matrix with exact factor structure
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        np.random.seed(10)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1)) + 1e-10 * np.random.randn(d)

        _project_correlation_factors(X)
        assert np.isfinite(X).all()
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1.)

        # Try to recover the structure
        rslt = corr_nearest_factor(mat, dm, maxiter=10000)
        err_msg = 'rank=%d, niter=%d' % (dm, len(rslt.objective_values))
        assert_allclose(rslt.objective_values[:5], objvals[dm - 1],
                        rtol=0.5, err_msg=err_msg)
        assert rslt.Converged

        mat1 = rslt.corr.to_matrix()
        assert_allclose(mat, mat1, rtol=0.25, atol=1e-3, err_msg=err_msg)

    @pytest.mark.slow
    @pytest.mark.parametrize('dm', [1, 2])
    def test_corr_nearest_factor_sparse(self, dm):
        # Test that result is the same if the input is dense or sparse
        d = 200

        # Generate a test matrix of factors
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2 * np.pi, d)
        rs = np.random.RandomState(10)
        for j in range(dm):
            X[:, j] = np.sin(x * (j + 1)) + rs.randn(d)

        # Get the correlation matrix
        _project_correlation_factors(X)
        X *= 0.7
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, 1)

        # Threshold it
        mat.flat[np.abs(mat.flat) < 0.35] = 0.0
        smat = sparse.csr_matrix(mat)

        dense_rslt = corr_nearest_factor(mat, dm, maxiter=10000)
        sparse_rslt = corr_nearest_factor(smat, dm, maxiter=10000)

        mat_dense = dense_rslt.corr.to_matrix()
        mat_sparse = sparse_rslt.corr.to_matrix()

        assert dense_rslt.Converged is sparse_rslt.Converged
        assert dense_rslt.Converged is True

        assert_allclose(mat_dense, mat_sparse, rtol=.25, atol=1e-3)

    # Test on a quadratic function.
    def test_spg_optim(self, reset_randomstate):

        dm = 100

        ind = np.arange(dm)
        indmat = np.abs(ind[:,None] - ind[None,:])
        M = 0.8**indmat

        def obj(x):
            return np.dot(x, np.dot(M, x))

        def grad(x):
            return 2*np.dot(M, x)

        def project(x):
            return x

        x = np.random.normal(size=dm)
        rslt = _spg_optim(obj, grad, x, project)
        xnew = rslt.params
        assert rslt.Converged is True
        assert_almost_equal(obj(xnew), 0, decimal=3)

    def test_decorrelate(self, reset_randomstate):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()
        rmat = np.linalg.cholesky(mat)
        dcr = fac.decorrelate(rmat)
        idm = np.dot(dcr, dcr.T)
        assert_almost_equal(idm, np.eye(d))

        rhs = np.random.normal(size=(d, 5))
        mat2 = np.dot(rhs.T, np.linalg.solve(mat, rhs))
        mat3 = fac.decorrelate(rhs)
        mat3 = np.dot(mat3.T, mat3)
        assert_almost_equal(mat2, mat3)

    def test_logdet(self, reset_randomstate):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 4))
        fac = FactoredPSDMatrix(dg, root)
        mat = fac.to_matrix()

        _, ld = np.linalg.slogdet(mat)
        ld2 = fac.logdet()

        assert_almost_equal(ld, ld2)

    def test_solve(self, reset_randomstate):

        d = 30
        dg = np.linspace(1, 2, d)
        root = np.random.normal(size=(d, 2))
        fac = FactoredPSDMatrix(dg, root)
        rhs = np.random.normal(size=(d, 5))
        sr1 = fac.solve(rhs)
        mat = fac.to_matrix()
        sr2 = np.linalg.solve(mat, rhs)
        assert_almost_equal(sr1, sr2)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_cov_nearest_factor_homog(self, dm):

        d = 100

        # Construct a test matrix with exact factor structure
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2*np.pi, d)
        for j in range(dm):
            X[:, j] = np.sin(x*(j+1))
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, np.diag(mat) + 3.1)

        # Try to recover the structure
        rslt = cov_nearest_factor_homog(mat, dm)
        mat1 = rslt.to_matrix()

        assert_allclose(mat, mat1, rtol=0.25, atol=1e-3)

    @pytest.mark.parametrize('dm', [1, 2])
    def test_cov_nearest_factor_homog_sparse(self, dm):
        # Check that dense and sparse inputs give the same result

        d = 100

        # Construct a test matrix with exact factor structure
        X = np.zeros((d, dm), dtype=np.float64)
        x = np.linspace(0, 2*np.pi, d)
        for j in range(dm):
            X[:, j] = np.sin(x*(j+1))
        mat = np.dot(X, X.T)
        np.fill_diagonal(mat, np.diag(mat) + 3.1)

        # Fit to dense
        rslt = cov_nearest_factor_homog(mat, dm)
        mat1 = rslt.to_matrix()

        # Fit to sparse
        smat = sparse.csr_matrix(mat)
        rslt = cov_nearest_factor_homog(smat, dm)
        mat2 = rslt.to_matrix()

        assert_allclose(mat1, mat2, rtol=0.25, atol=1e-3)

    def test_corr_thresholded(self, reset_randomstate):

        import datetime

        t1 = datetime.datetime.now()
        X = np.random.normal(size=(2000,10))
        tcor = corr_thresholded(X, 0.2, max_elt=4e6)
        t2 = datetime.datetime.now()
        ss = (t2-t1).seconds

        fcor = np.corrcoef(X)
        fcor *= (np.abs(fcor) >= 0.2)

        assert_allclose(tcor.todense(), fcor, rtol=0.25, atol=1e-3)
