import pytest

import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_

from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds


A = np.array([
    [0.171, -0.057],
    [-0.049, -0.248],
    [-0.166, 0.054],
])
b = np.array([0.074, 1.014, -0.383])


class BaseMixin:
    def setup_method(self):
        self.rnd = np.random.RandomState(0)

    def test_dense_no_bounds(self):
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, method=self.method, lsq_solver=lsq_solver)
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
            assert_allclose(res.x, res.unbounded_sol[0])

    def test_dense_bounds(self):
        # Solutions for comparison are taken from MATLAB.
        lb = np.array([-1, -10])
        ub = np.array([1, 0])
        unbounded_sol = lstsq(A, b, rcond=-1)[0]
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        lb = np.array([0.0, -np.inf])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.0, -4.084174437334673]),
                            atol=1e-6)
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        lb = np.array([-1, 0])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.448427311733504, 0]),
                            atol=1e-15)
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        ub = np.array([np.inf, -5])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([-0.105560998682388, -5]))
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        ub = np.array([-1, np.inf])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([-1, -4.181102129483254]))
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

        lb = np.array([0, -4])
        ub = np.array([1, 0])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.005236663400791, -4]))
            assert_allclose(res.unbounded_sol[0], unbounded_sol)

    def test_bounds_variants(self):
        x = np.array([1, 3])
        A = self.rnd.uniform(size=(2, 2))
        b = A@x
        lb = np.array([1, 1])
        ub = np.array([2, 2])
        bounds_old = (lb, ub)
        bounds_new = Bounds(lb, ub)
        res_old = lsq_linear(A, b, bounds_old)
        res_new = lsq_linear(A, b, bounds_new)
        assert not np.allclose(res_new.x, res_new.unbounded_sol[0])
        assert_allclose(res_old.x, res_new.x)

    def test_np_matrix(self):
        # gh-10711
        with np.testing.suppress_warnings() as sup:
            sup.filter(PendingDeprecationWarning)
            A = np.matrix([[20, -4, 0, 2, 3], [10, -2, 1, 0, -1]])
        k = np.array([20, 15])
        lsq_linear(A, k)

    def test_dense_rank_deficient(self):
        A = np.array([[-0.307, -0.184]])
        b = np.array([0.773])
        lb = [-0.1, -0.1]
        ub = [0.1, 0.1]
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, [-0.1, -0.1])
            assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

        A = np.array([
            [0.334, 0.668],
            [-0.516, -1.032],
            [0.192, 0.384],
        ])
        b = np.array([-1.436, 0.135, 0.909])
        lb = [0, -1]
        ub = [1, -0.5]
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.optimality, 0, atol=1e-11)
            assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

    def test_full_result(self):
        lb = np.array([0, -4])
        ub = np.array([1, 0])
        res = lsq_linear(A, b, (lb, ub), method=self.method)

        assert_allclose(res.x, [0.005236663400791, -4])
        assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])

        r = A.dot(res.x) - b
        assert_allclose(res.cost, 0.5 * np.dot(r, r))
        assert_allclose(res.fun, r)

        assert_allclose(res.optimality, 0.0, atol=1e-12)
        assert_equal(res.active_mask, [0, -1])
        assert_(res.nit < 15)
        assert_(res.status == 1 or res.status == 3)
        assert_(isinstance(res.message, str))
        assert_(res.success)

    # This is a test for issue #9982.
    def test_almost_singular(self):
        A = np.array(
            [[0.8854232310355122, 0.0365312146937765, 0.0365312146836789],
             [0.3742460132129041, 0.0130523214078376, 0.0130523214077873],
             [0.9680633871281361, 0.0319366128718639, 0.0319366128718388]])

        b = np.array(
            [0.0055029366538097, 0.0026677442422208, 0.0066612514782381])

        result = lsq_linear(A, b, method=self.method)
        assert_(result.cost < 1.1e-8)

    def test_large_rank_deficient(self):
        np.random.seed(0)
        n, m = np.sort(np.random.randint(2, 1000, size=2))
        m *= 2   # make m >> n
        A = 1.0 * np.random.randint(-99, 99, size=[m, n])
        b = 1.0 * np.random.randint(-99, 99, size=[m])
        bounds = 1.0 * np.sort(np.random.randint(-99, 99, size=(2, n)), axis=0)
        bounds[1, :] += 1.0  # ensure up > lb

        # Make the A matrix strongly rank deficient by replicating some columns
        w = np.random.choice(n, n)  # Select random columns with duplicates
        A = A[:, w]

        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)

    def test_convergence_small_matrix(self):
        A = np.array([[49.0, 41.0, -32.0],
                      [-19.0, -32.0, -8.0],
                      [-13.0, 10.0, 69.0]])
        b = np.array([-41.0, -90.0, 47.0])
        bounds = np.array([[31.0, -44.0, 26.0],
                           [54.0, -32.0, 28.0]])

        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)


class SparseMixin:
    def test_sparse_and_LinearOperator(self):
        m = 5000
        n = 1000
        A = rand(m, n, random_state=0)
        b = self.rnd.randn(m)
        res = lsq_linear(A, b)
        assert_allclose(res.optimality, 0, atol=1e-6)

        A = aslinearoperator(A)
        res = lsq_linear(A, b)
        assert_allclose(res.optimality, 0, atol=1e-6)

    def test_sparse_bounds(self):
        m = 5000
        n = 1000
        A = rand(m, n, random_state=0)
        b = self.rnd.randn(m)
        lb = self.rnd.randn(n)
        ub = lb + 1
        res = lsq_linear(A, b, (lb, ub))
        assert_allclose(res.optimality, 0.0, atol=1e-6)

        res = lsq_linear(A, b, (lb, ub), lsmr_tol=1e-13,
                         lsmr_maxiter=1500)
        assert_allclose(res.optimality, 0.0, atol=1e-6)

        res = lsq_linear(A, b, (lb, ub), lsmr_tol='auto')
        assert_allclose(res.optimality, 0.0, atol=1e-6)

    def test_sparse_ill_conditioned(self):
        # Sparse matrix with condition number of ~4 million
        data = np.array([1., 1., 1., 1. + 1e-6, 1.])
        row = np.array([0, 0, 1, 2, 2])
        col = np.array([0, 2, 1, 0, 2])
        A = coo_matrix((data, (row, col)), shape=(3, 3))

        # Get the exact solution
        exact_sol = lsq_linear(A.toarray(), b, lsq_solver='exact')

        # Default lsmr arguments should not fully converge the solution
        default_lsmr_sol = lsq_linear(A, b, lsq_solver='lsmr')
        with pytest.raises(AssertionError, match=""):
            assert_allclose(exact_sol.x, default_lsmr_sol.x)

        # By increasing the maximum lsmr iters, it will converge
        conv_lsmr = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=10)
        assert_allclose(exact_sol.x, conv_lsmr.x)


class TestTRF(BaseMixin, SparseMixin):
    method = 'trf'
    lsq_solvers = ['exact', 'lsmr']


class TestBVLS(BaseMixin):
    method = 'bvls'
    lsq_solvers = ['exact']


class TestErrorChecking:
    def test_option_lsmr_tol(self):
        # Should work with a positive float, string equal to 'auto', or None
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=1e-2)
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='auto')
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=None)

        # Should raise error with negative float, strings
        # other than 'auto', and integers
        err_message = "`lsmr_tol` must be None, 'auto', or positive float."
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=-0.1)
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='foo')
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=1)

    def test_option_lsmr_maxiter(self):
        # Should work with positive integers or None
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=1)
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=None)

        # Should raise error with 0 or negative max iter
        err_message = "`lsmr_maxiter` must be None or positive integer."
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=0)
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=-1)
