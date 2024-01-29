"""Test functions for the sparse.linalg._expm_multiply module."""
from functools import partial
from itertools import product

import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
                           suppress_warnings)
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
        _onenormest_matrix_power, expm_multiply, _expm_multiply_simple,
        _expm_multiply_interval)
from scipy._lib._util import np_long


IMPRECISE = {np.single, np.csingle}
REAL_DTYPES = {np.intc, np_long, np.longlong,
               np.float32, np.float64, np.longdouble}
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}
# use sorted list to ensure fixed order of tests
DTYPES = sorted(REAL_DTYPES ^ COMPLEX_DTYPES, key=str)


def estimated(func):
    """If trace is estimated, it should warn.

    We warn that estimation of trace might impact performance.
    All result have to be correct nevertheless!

    """
    def wrapped(*args, **kwds):
        with pytest.warns(UserWarning,
                          match="Trace of LinearOperator not available"):
            return func(*args, **kwds)
    return wrapped


def less_than_or_close(a, b):
    return np.allclose(a, b) or (a < b)


class TestExpmActionSimple:
    """
    These tests do not consider the case of multiple time steps in one call.
    """

    def test_theta_monotonicity(self):
        pairs = sorted(_theta.items())
        for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
            assert_(theta_a < theta_b)

    def test_p_max_default(self):
        m_max = 55
        expected_p_max = 8
        observed_p_max = _compute_p_max(m_max)
        assert_equal(observed_p_max, expected_p_max)

    def test_p_max_range(self):
        for m_max in range(1, 55+1):
            p_max = _compute_p_max(m_max)
            assert_(p_max*(p_max - 1) <= m_max + 1)
            p_too_big = p_max + 1
            assert_(p_too_big*(p_too_big - 1) > m_max + 1)

    def test_onenormest_matrix_power(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            for p in range(4):
                if not p:
                    M = np.identity(n)
                else:
                    M = np.dot(M, A)
                estimated = _onenormest_matrix_power(A, p)
                exact = np.linalg.norm(M, 1)
                assert_(less_than_or_close(estimated, exact))
                assert_(less_than_or_close(exact, 3*estimated))

    def test_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            expected = np.dot(sp_expm(A), B)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            assert_allclose(observed, expected)
            traceA = np.trace(A)
            observed = expm_multiply(aslinearoperator(A), B, traceA=traceA)
            assert_allclose(observed, expected)

    def test_matrix_vector_multiply(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            v = np.random.randn(n)
            observed = expm_multiply(A, v)
            expected = np.dot(sp_expm(A), v)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), v)
            assert_allclose(observed, expected)

    def test_scaled_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i, t in product(range(nsamples), [0.2, 1.0, 1.5]):
            with np.errstate(invalid='ignore'):
                A = scipy.linalg.inv(np.random.randn(n, n))
                B = np.random.randn(n, k)
                observed = _expm_multiply_simple(A, B, t=t)
                expected = np.dot(sp_expm(t*A), B)
                assert_allclose(observed, expected)
                observed = estimated(_expm_multiply_simple)(
                    aslinearoperator(A), B, t=t
                )
                assert_allclose(observed, expected)

    def test_scaled_expm_multiply_single_timepoint(self):
        np.random.seed(1234)
        t = 0.1
        n = 5
        k = 2
        A = np.random.randn(n, n)
        B = np.random.randn(n, k)
        observed = _expm_multiply_simple(A, B, t=t)
        expected = sp_expm(t*A).dot(B)
        assert_allclose(observed, expected)
        observed = estimated(_expm_multiply_simple)(
            aslinearoperator(A), B, t=t
        )
        assert_allclose(observed, expected)

    def test_sparse_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning,
                           "splu converted its input to CSC format")
                sup.filter(SparseEfficiencyWarning,
                           "spsolve is more efficient when sparse b is in the"
                           " CSC matrix format")
                expected = sp_expm(A).dot(B)
            assert_allclose(observed, expected)
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            assert_allclose(observed, expected)

    def test_complex(self):
        A = np.array([
            [1j, 1j],
            [0, 1j]], dtype=complex)
        B = np.array([1j, 1j])
        observed = expm_multiply(A, B)
        expected = np.array([
            1j * np.exp(1j) + 1j * (1j*np.cos(1) - np.sin(1)),
            1j * np.exp(1j)], dtype=complex)
        assert_allclose(observed, expected)
        observed = estimated(expm_multiply)(aslinearoperator(A), B)
        assert_allclose(observed, expected)


class TestExpmActionInterval:

    def test_sparse_expm_multiply_interval(self):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        n = 40
        k = 3
        endpoint = True
        for num in (14, 13, 2):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            v = np.random.randn(n)
            for target in (B, v):
                X = expm_multiply(A, target, start=start, stop=stop,
                                  num=num, endpoint=endpoint)
                samples = np.linspace(start=start, stop=stop,
                                      num=num, endpoint=endpoint)
                with suppress_warnings() as sup:
                    sup.filter(SparseEfficiencyWarning,
                               "splu converted its input to CSC format")
                    sup.filter(SparseEfficiencyWarning,
                               "spsolve is more efficient when sparse b is in"
                               " the CSC matrix format")
                    for solution, t in zip(X, samples):
                        assert_allclose(solution, sp_expm(t*A).dot(target))

    def test_expm_multiply_interval_vector(self):
        np.random.seed(1234)
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        for num, n in product([14, 13, 2], [1, 2, 5, 20, 40]):
            A = scipy.linalg.inv(np.random.randn(n, n))
            v = np.random.randn(n)
            samples = np.linspace(num=num, **interval)
            X = expm_multiply(A, v, num=num, **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t*A).dot(v))
            # test for linear operator with unknown trace -> estimate trace
            Xguess = estimated(expm_multiply)(aslinearoperator(A), v,
                                              num=num, **interval)
            # test for linear operator with given trace
            Xgiven = expm_multiply(aslinearoperator(A), v, num=num, **interval,
                                   traceA=np.trace(A))
            # test robustness for linear operator with wrong trace
            Xwrong = expm_multiply(aslinearoperator(A), v, num=num, **interval,
                                   traceA=np.trace(A)*5)
            for sol_guess, sol_given, sol_wrong, t in zip(Xguess, Xgiven,
                                                          Xwrong, samples):
                correct = sp_expm(t*A).dot(v)
                assert_allclose(sol_guess, correct)
                assert_allclose(sol_given, correct)
                assert_allclose(sol_wrong, correct)

    def test_expm_multiply_interval_matrix(self):
        np.random.seed(1234)
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        for num, n, k in product([14, 13, 2], [1, 2, 5, 20, 40], [1, 2]):
            A = scipy.linalg.inv(np.random.randn(n, n))
            B = np.random.randn(n, k)
            samples = np.linspace(num=num, **interval)
            X = expm_multiply(A, B, num=num, **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t*A).dot(B))
            X = estimated(expm_multiply)(aslinearoperator(A), B, num=num,
                                         **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t*A).dot(B))

    def test_sparse_expm_multiply_interval_dtypes(self):
        # Test A & B int
        A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))

        # Test A complex, B int
        A = scipy.sparse.diags(-1j*np.arange(5),format='csr', dtype=complex)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(-1j*np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))

        # Test A int, B complex
        A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
        B = np.full(5, 1j, dtype=complex)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))

    def test_expm_multiply_interval_status_0(self):
        self._help_test_specific_expm_interval_status(0)

    def test_expm_multiply_interval_status_1(self):
        self._help_test_specific_expm_interval_status(1)

    def test_expm_multiply_interval_status_2(self):
        self._help_test_specific_expm_interval_status(2)

    def _help_test_specific_expm_interval_status(self, target_status):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        num = 13
        endpoint = True
        n = 5
        k = 2
        nrepeats = 10
        nsuccesses = 0
        for num in [14, 13, 2] * nrepeats:
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            status = _expm_multiply_interval(A, B,
                    start=start, stop=stop, num=num, endpoint=endpoint,
                    status_only=True)
            if status == target_status:
                X, status = _expm_multiply_interval(A, B,
                        start=start, stop=stop, num=num, endpoint=endpoint,
                        status_only=False)
                assert_equal(X.shape, (num, n, k))
                samples = np.linspace(start=start, stop=stop,
                        num=num, endpoint=endpoint)
                for solution, t in zip(X, samples):
                    assert_allclose(solution, sp_expm(t*A).dot(B))
                nsuccesses += 1
        if not nsuccesses:
            msg = 'failed to find a status-' + str(target_status) + ' interval'
            raise Exception(msg)


@pytest.mark.parametrize("dtype_a", DTYPES)
@pytest.mark.parametrize("dtype_b", DTYPES)
@pytest.mark.parametrize("b_is_matrix", [False, True])
def test_expm_multiply_dtype(dtype_a, dtype_b, b_is_matrix):
    """Make sure `expm_multiply` handles all numerical dtypes correctly."""
    assert_allclose_ = (partial(assert_allclose, rtol=1.2e-3, atol=1e-5)
                        if {dtype_a, dtype_b} & IMPRECISE else assert_allclose)
    rng = np.random.default_rng(1234)
    # test data
    n = 7
    b_shape = (n, 3) if b_is_matrix else (n, )
    if dtype_a in REAL_DTYPES:
        A = scipy.linalg.inv(rng.random([n, n])).astype(dtype_a)
    else:
        A = scipy.linalg.inv(
            rng.random([n, n]) + 1j*rng.random([n, n])
        ).astype(dtype_a)
    if dtype_b in REAL_DTYPES:
        B = (2*rng.random(b_shape)).astype(dtype_b)
    else:
        B = (rng.random(b_shape) + 1j*rng.random(b_shape)).astype(dtype_b)

    # single application
    sol_mat = expm_multiply(A, B)
    sol_op = estimated(expm_multiply)(aslinearoperator(A), B)
    direct_sol = np.dot(sp_expm(A), B)
    assert_allclose_(sol_mat, direct_sol)
    assert_allclose_(sol_op, direct_sol)
    sol_op = expm_multiply(aslinearoperator(A), B, traceA=np.trace(A))
    assert_allclose_(sol_op, direct_sol)

    # for time points
    interval = {'start': 0.1, 'stop': 3.2, 'num': 13, 'endpoint': True}
    samples = np.linspace(**interval)
    X_mat = expm_multiply(A, B, **interval)
    X_op = estimated(expm_multiply)(aslinearoperator(A), B, **interval)
    for sol_mat, sol_op, t in zip(X_mat, X_op, samples):
        direct_sol = sp_expm(t*A).dot(B)
        assert_allclose_(sol_mat, direct_sol)
        assert_allclose_(sol_op, direct_sol)
