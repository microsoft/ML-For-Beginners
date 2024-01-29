import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr

# Set up a test problem
n = 35
G = np.eye(n)
normal = np.random.normal
norm = np.linalg.norm

for jj in range(5):
    gg = normal(size=n)
    hh = gg * gg.T
    G += (hh + hh.T) * 0.5
    G += normal(size=n) * normal(size=n)

b = normal(size=n)

# tolerance for atol/btol keywords of lsqr()
tol = 2e-10
# tolerances for testing the results of the lsqr() call with assert_allclose
# These tolerances are a bit fragile - see discussion in gh-15301.
atol_test = 4e-10
rtol_test = 2e-8
show = False
maxit = None


def test_lsqr_basic():
    b_copy = b.copy()
    xo, *_ = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    assert_array_equal(b_copy, b)

    svx = np.linalg.solve(G, b)
    assert_allclose(xo, svx, atol=atol_test, rtol=rtol_test)

    # Now the same but with damp > 0.
    # This is equivalent to solving the extended system:
    # ( G      ) @ x = ( b )
    # ( damp*I )       ( 0 )
    damp = 1.5
    xo, *_ = lsqr(
        G, b, damp=damp, show=show, atol=tol, btol=tol, iter_lim=maxit)

    Gext = np.r_[G, damp * np.eye(G.shape[1])]
    bext = np.r_[b, np.zeros(G.shape[1])]
    svx, *_ = np.linalg.lstsq(Gext, bext, rcond=None)
    assert_allclose(xo, svx, atol=atol_test, rtol=rtol_test)


def test_gh_2466():
    row = np.array([0, 0])
    col = np.array([0, 1])
    val = np.array([1, -1])
    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(1, 2))
    b = np.asarray([4])
    lsqr(A, b)


def test_well_conditioned_problems():
    # Test that sparse the lsqr solver returns the right solution
    # on various problems with different random seeds.
    # This is a non-regression test for a potential ZeroDivisionError
    # raised when computing the `test2` & `test3` convergence conditions.
    n = 10
    A_sparse = scipy.sparse.eye(n, n)
    A_dense = A_sparse.toarray()

    with np.errstate(invalid='raise'):
        for seed in range(30):
            rng = np.random.RandomState(seed + 10)
            beta = rng.rand(n)
            beta[beta == 0] = 0.00001  # ensure that all the betas are not null
            b = A_sparse @ beta[:, np.newaxis]
            output = lsqr(A_sparse, b, show=show)

            # Check that the termination condition corresponds to an approximate
            # solution to Ax = b
            assert_equal(output[1], 1)
            solution = output[0]

            # Check that we recover the ground truth solution
            assert_allclose(solution, beta)

            # Sanity check: compare to the dense array solver
            reference_solution = np.linalg.solve(A_dense, b).ravel()
            assert_allclose(solution, reference_solution)


def test_b_shapes():
    # Test b being a scalar.
    A = np.array([[1.0, 2.0]])
    b = 3.0
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b) == pytest.approx(0)

    # Test b being a column vector.
    A = np.eye(10)
    b = np.ones((10, 1))
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)


def test_initialization():
    # Test the default setting is the same as zeros
    b_copy = b.copy()
    x_ref = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    x0 = np.zeros(x_ref[0].shape)
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_array_equal(b_copy, b)
    assert_allclose(x_ref[0], x[0])

    # Test warm-start with single iteration
    x0 = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=1)[0]
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_allclose(x_ref[0], x[0])
    assert_array_equal(b_copy, b)
