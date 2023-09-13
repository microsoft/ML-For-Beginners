"""
Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as SciPy

Testing Code for LSMR.

03 Jun 2010: First version release with lsmr.py

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b


class TestLSMR:
    def setup_method(self):
        self.n = 10
        self.m = 10

    def assertCompatibleSystem(self, A, xtrue):
        Afun = aslinearoperator(A)
        b = Afun.matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)

    def testIdentityACase1(self):
        A = eye(self.n)
        xtrue = zeros((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase2(self):
        A = eye(self.n)
        xtrue = ones((self.n,1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase3(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n,0,-1))
        self.assertCompatibleSystem(A, xtrue)

    def testBidiagonalA(self):
        A = lowerBidiagonalMatrix(20,self.n)
        xtrue = transpose(arange(self.n,0,-1))
        self.assertCompatibleSystem(A,xtrue)

    def testScalarB(self):
        A = array([[1.0, 2.0]])
        b = 3.0
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b) == pytest.approx(0)

    def testComplexX(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexX0(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1))
        b = aslinearoperator(A).matvec(xtrue)
        x0 = zeros(self.n, dtype=complex)
        x = lsmr(A, b, x0=x0)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)

    def testComplexA(self):
        A = 4 * eye(self.n) + 1j * ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1).astype(complex))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexB(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        b = aslinearoperator(A).matvec(xtrue)
        x = lsmr(A, b)[0]
        assert norm(x - xtrue) == pytest.approx(0, abs=1e-5)

    def testColumnB(self):
        A = eye(self.n)
        b = ones((self.n, 1))
        x = lsmr(A, b)[0]
        assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)

    def testInitialization(self):
        # Test that the default setting is not modified
        x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
        assert_allclose(norm(b - G@x_ref), normr_ref, atol=1e-6)

        # Test passing zeros yields similiar result
        x0 = zeros(b.shape)
        x = lsmr(G, b, x0=x0)[0]
        assert_allclose(x, x_ref)

        # Test warm-start with single iteration
        x0 = lsmr(G, b, maxiter=1)[0]

        x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
        assert_allclose(norm(b - G@x), normr, atol=1e-6)

        # NOTE(gh-12139): This doesn't always converge to the same value as
        # ref because error estimates will be slightly different when calculated
        # from zeros vs x0 as a result only compare norm and itn (not x).

        # x generally converges 1 iteration faster because it started at x0.
        # itn == itn_ref means that lsmr(x0) took an extra iteration see above.
        # -1 is technically possible but is rare (1 in 100000) so it's more
        # likely to be an error elsewhere.
        assert itn - itn_ref in (0, 1)

        # If an extra iteration is performed normr may be 0, while normr_ref
        # may be much larger.
        assert normr < normr_ref * (1 + 1e-6)


class TestLSMRReturns:
    def setup_method(self):
        self.n = 10
        self.A = lowerBidiagonalMatrix(20, self.n)
        self.xtrue = transpose(arange(self.n, 0, -1))
        self.Afun = aslinearoperator(self.A)
        self.b = self.Afun.matvec(self.xtrue)
        self.x0 = ones(self.n)
        self.x00 = self.x0.copy()
        self.returnValues = lsmr(self.A, self.b)
        self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)

    def test_unchanged_x0(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0
        assert_allclose(self.x00, self.x0)

    def testNormr(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert norm(self.b - self.Afun.matvec(x)) == pytest.approx(normr)

    def testNormar(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert (norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x)))
                == pytest.approx(normar))

    def testNormx(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert norm(x) == pytest.approx(normx)


def lowerBidiagonalMatrix(m, n):
    # This is a simple example for testing LSMR.
    # It uses the leading m*n submatrix from
    # A = [ 1
    #       1 2
    #         2 3
    #           3 4
    #             ...
    #               n ]
    # suitably padded by zeros.
    #
    # 04 Jun 2010: First version for distribution with lsmr.py
    if m <= n:
        row = hstack((arange(m, dtype=int),
                      arange(1, m, dtype=int)))
        col = hstack((arange(m, dtype=int),
                      arange(m-1, dtype=int)))
        data = hstack((arange(1, m+1, dtype=float),
                       arange(1,m, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m,n))
    else:
        row = hstack((arange(n, dtype=int),
                      arange(1, n+1, dtype=int)))
        col = hstack((arange(n, dtype=int),
                      arange(n, dtype=int)))
        data = hstack((arange(1, n+1, dtype=float),
                       arange(1,n+1, dtype=float)))
        return coo_matrix((data,(row, col)), shape=(m,n))
