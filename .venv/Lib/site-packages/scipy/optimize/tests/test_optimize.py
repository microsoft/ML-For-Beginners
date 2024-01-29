"""
Unit tests for optimization routines from optimize.py

Authors:
   Ed Schofield, Nov 2005
   Andrew Straw, April 2008

To run it in its simplest form::
  nosetests test_optimize.py

"""
import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_almost_equal,
                           assert_no_warnings, assert_warns,
                           assert_array_less, suppress_warnings)
import pytest
from pytest import raises as assert_raises

from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
                                      MINIMIZE_METHODS_NEW_CB,
                                      MINIMIZE_SCALAR_METHODS)
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess

from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
                          csr_array, csc_array)

def test_check_grad():
    # Verify if check_grad is able to estimate the derivative of the
    # expit (logistic sigmoid) function.

    def expit(x):
        return 1 / (1 + np.exp(-x))

    def der_expit(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    x0 = np.array([1.5])

    r = optimize.check_grad(expit, der_expit, x0)
    assert_almost_equal(r, 0)
    r = optimize.check_grad(expit, der_expit, x0,
                            direction='random', seed=1234)
    assert_almost_equal(r, 0)

    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-6)
    assert_almost_equal(r, 0)
    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-6,
                            direction='random', seed=1234)
    assert_almost_equal(r, 0)

    # Check if the epsilon parameter is being considered.
    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=1e-1) - 0)
    assert r > 1e-7
    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=1e-1,
                                direction='random', seed=1234) - 0)
    assert r > 1e-7

    def x_sinx(x):
        return (x*np.sin(x)).sum()

    def der_x_sinx(x):
        return np.sin(x) + x*np.cos(x)

    x0 = np.arange(0, 2, 0.2)

    r = optimize.check_grad(x_sinx, der_x_sinx, x0,
                            direction='random', seed=1234)
    assert_almost_equal(r, 0)

    assert_raises(ValueError, optimize.check_grad,
                  x_sinx, der_x_sinx, x0,
                  direction='random_projection', seed=1234)

    # checking can be done for derivatives of vector valued functions
    r = optimize.check_grad(himmelblau_grad, himmelblau_hess, himmelblau_x0,
                            direction='all', seed=1234)
    assert r < 5e-7


class CheckOptimize:
    """ Base test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def setup_method(self):
        self.F = np.array([[1, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [1, 0, 0],
                           [1, 0, 0]])
        self.K = np.array([1., 0.3, 0.5])
        self.startparams = np.zeros(3, np.float64)
        self.solution = np.array([0., -0.524869316, 0.487525860])
        self.maxiter = 1000
        self.funccalls = 0
        self.gradcalls = 0
        self.trace = []

    def func(self, x):
        self.funccalls += 1
        if self.funccalls > 6000:
            raise RuntimeError("too many iterations in optimization routine")
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        f = logZ - np.dot(self.K, x)
        self.trace.append(np.copy(x))
        return f

    def grad(self, x):
        self.gradcalls += 1
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        p = np.exp(log_pdot - logZ)
        return np.dot(self.F.transpose(), p) - self.K

    def hess(self, x):
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        p = np.exp(log_pdot - logZ)
        return np.dot(self.F.T,
                      np.dot(np.diag(p), self.F - np.dot(self.F.T, p)))

    def hessp(self, x, p):
        return np.dot(self.hess(x), p)


class CheckOptimizeParameterized(CheckOptimize):

    def test_cg(self):
        # conjugate gradient optimization routine
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='CG', jac=self.grad,
                                    options=opts)
            params, fopt, func_calls, grad_calls, warnflag = \
                res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        else:
            retval = optimize.fmin_cg(self.func, self.startparams,
                                      self.grad, (), maxiter=self.maxiter,
                                      full_output=True, disp=self.disp,
                                      retall=False)
            (params, fopt, func_calls, grad_calls, warnflag) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls == 9, self.funccalls
        assert self.gradcalls == 7, self.gradcalls

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[2:4],
                        [[0, -0.5, 0.5],
                         [0, -5.05700028e-01, 4.95985862e-01]],
                        atol=1e-14, rtol=1e-7)

    def test_cg_cornercase(self):
        def f(r):
            return 2.5 * (1 - np.exp(-1.5*(r - 0.5)))**2

        # Check several initial guesses. (Too far away from the
        # minimum, the function ends up in the flat region of exp.)
        for x0 in np.linspace(-0.75, 3, 71):
            sol = optimize.minimize(f, [x0], method='CG')
            assert sol.success
            assert_allclose(sol.x, [0.5], rtol=1e-5)

    def test_bfgs(self):
        # Broyden-Fletcher-Goldfarb-Shanno optimization routine
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams,
                                    jac=self.grad, method='BFGS', args=(),
                                    options=opts)

            params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = (
                    res['x'], res['fun'], res['jac'], res['hess_inv'],
                    res['nfev'], res['njev'], res['status'])
        else:
            retval = optimize.fmin_bfgs(self.func, self.startparams, self.grad,
                                        args=(), maxiter=self.maxiter,
                                        full_output=True, disp=self.disp,
                                        retall=False)
            (params, fopt, gopt, Hopt,
             func_calls, grad_calls, warnflag) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls == 10, self.funccalls
        assert self.gradcalls == 8, self.gradcalls

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[6:8],
                        [[0, -5.25060743e-01, 4.87748473e-01],
                         [0, -5.24885582e-01, 4.87530347e-01]],
                        atol=1e-14, rtol=1e-7)
    
    def test_bfgs_hess_inv0_neg(self):
        # Ensure that BFGS does not accept neg. def. initial inverse 
        # Hessian estimate.
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't "
                           "positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            opts = {'disp': self.disp, 'hess_inv0': -np.eye(5)}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(),
                              options=opts)
    
    def test_bfgs_hess_inv0_semipos(self):
        # Ensure that BFGS does not accept semi pos. def. initial inverse 
        # Hessian estimate.
        with pytest.raises(ValueError, match="'hess_inv0' matrix isn't "
                           "positive definite."):
            x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
            hess_inv0 = np.eye(5)
            hess_inv0[0, 0] = 0
            opts = {'disp': self.disp, 'hess_inv0': hess_inv0}
            optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(),
                              options=opts)
    
    def test_bfgs_hess_inv0_sanity(self):
        # Ensure that BFGS handles `hess_inv0` parameter correctly.
        fun = optimize.rosen
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        opts = {'disp': self.disp, 'hess_inv0': 1e-2 * np.eye(5)}
        res = optimize.minimize(fun, x0=x0, method='BFGS', args=(), 
                                options=opts)
        res_true = optimize.minimize(fun, x0=x0, method='BFGS', args=(), 
                                     options={'disp': self.disp})
        assert_allclose(res.fun, res_true.fun, atol=1e-6)
            
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_bfgs_infinite(self):
        # Test corner case where -Inf is the minimum.  See gh-2019.
        def func(x):
            return -np.e ** (-x)
        def fprime(x):
            return -func(x)
        x0 = [0]
        with np.errstate(over='ignore'):
            if self.use_wrapper:
                opts = {'disp': self.disp}
                x = optimize.minimize(func, x0, jac=fprime, method='BFGS',
                                      args=(), options=opts)['x']
            else:
                x = optimize.fmin_bfgs(func, x0, fprime, disp=self.disp)
            assert not np.isfinite(func(x))

    def test_bfgs_xrtol(self):
        # test for #17345 to test xrtol parameter
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res = optimize.minimize(optimize.rosen,
                                x0, method='bfgs', options={'xrtol': 1e-3})
        ref = optimize.minimize(optimize.rosen,
                                x0, method='bfgs', options={'gtol': 1e-3})
        assert res.nit != ref.nit

    def test_bfgs_c1(self):
        # test for #18977 insufficiently low value of c1 leads to precision loss
        # for poor starting parameters
        x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
        res_c1_small = optimize.minimize(optimize.rosen,
                                         x0, method='bfgs', options={'c1': 1e-8})
        res_c1_big = optimize.minimize(optimize.rosen,
                                       x0, method='bfgs', options={'c1': 1e-1})

        assert res_c1_small.nfev > res_c1_big.nfev

    def test_bfgs_c2(self):
        # test that modification of c2 parameter
        # results in different number of iterations
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res_default = optimize.minimize(optimize.rosen,
                                        x0, method='bfgs', options={'c2': .9})
        res_mod = optimize.minimize(optimize.rosen,
                                    x0, method='bfgs', options={'c2': 1e-2})
        assert res_default.nit > res_mod.nit
    
    @pytest.mark.parametrize(["c1", "c2"], [[0.5, 2],
                                            [-0.1, 0.1],
                                            [0.2, 0.1]])
    def test_invalid_c1_c2(self, c1, c2):
        with pytest.raises(ValueError, match="'c1' and 'c2'"):
            x0 = [10.3, 20.7, 10.8, 1.9, -1.2]
            optimize.minimize(optimize.rosen, x0, method='cg',
                              options={'c1': c1, 'c2': c2})

    def test_powell(self):
        # Powell (direction set) optimization routine
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Powell', options=opts)
            params, fopt, direc, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['direc'], res['nit'],
                    res['nfev'], res['status'])
        else:
            retval = optimize.fmin_powell(self.func, self.startparams,
                                          args=(), maxiter=self.maxiter,
                                          full_output=True, disp=self.disp,
                                          retall=False)
            (params, fopt, direc, numiter, func_calls, warnflag) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)
        # params[0] does not affect the objective function
        assert_allclose(params[1:], self.solution[1:], atol=5e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        #
        # However, some leeway must be added: the exact evaluation
        # count is sensitive to numerical error, and floating-point
        # computations are not bit-for-bit reproducible across
        # machines, and when using e.g., MKL, data alignment
        # etc., affect the rounding error.
        #
        assert self.funccalls <= 116 + 20, self.funccalls
        assert self.gradcalls == 0, self.gradcalls

    @pytest.mark.xfail(reason="This part of test_powell fails on some "
                       "platforms, but the solution returned by powell is "
                       "still valid.")
    def test_powell_gh14014(self):
        # This part of test_powell started failing on some CI platforms;
        # see gh-14014. Since the solution is still correct and the comments
        # in test_powell suggest that small differences in the bits are known
        # to change the "trace" of the solution, seems safe to xfail to get CI
        # green now and investigate later.

        # Powell (direction set) optimization routine
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Powell', options=opts)
            params, fopt, direc, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['direc'], res['nit'],
                    res['nfev'], res['status'])
        else:
            retval = optimize.fmin_powell(self.func, self.startparams,
                                          args=(), maxiter=self.maxiter,
                                          full_output=True, disp=self.disp,
                                          retall=False)
            (params, fopt, direc, numiter, func_calls, warnflag) = retval

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[34:39],
                        [[0.72949016, -0.44156936, 0.47100962],
                         [0.72949016, -0.44156936, 0.48052496],
                         [1.45898031, -0.88313872, 0.95153458],
                         [0.72949016, -0.44156936, 0.47576729],
                         [1.72949016, -0.44156936, 0.47576729]],
                        atol=1e-14, rtol=1e-7)

    def test_powell_bounded(self):
        # Powell (direction set) optimization routine
        # same as test_powell above, but with bounds
        bounds = [(-np.pi, np.pi) for _ in self.startparams]
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    bounds=bounds,
                                    method='Powell', options=opts)
            params, func_calls = (res['x'], res['nfev'])

            assert func_calls == self.funccalls
            assert_allclose(self.func(params), self.func(self.solution),
                            atol=1e-6, rtol=1e-5)

            # The exact evaluation count is sensitive to numerical error, and
            # floating-point computations are not bit-for-bit reproducible
            # across machines, and when using e.g. MKL, data alignment etc.
            # affect the rounding error.
            # It takes 155 calls on my machine, but we can add the same +20
            # margin as is used in `test_powell`
            assert self.funccalls <= 155 + 20
            assert self.gradcalls == 0

    def test_neldermead(self):
        # Nelder-Mead simplex algorithm
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Nelder-mead', options=opts)
            params, fopt, numiter, func_calls, warnflag = (
                    res['x'], res['fun'], res['nit'], res['nfev'],
                    res['status'])
        else:
            retval = optimize.fmin(self.func, self.startparams,
                                   args=(), maxiter=self.maxiter,
                                   full_output=True, disp=self.disp,
                                   retall=False)
            (params, fopt, numiter, func_calls, warnflag) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls == 167, self.funccalls
        assert self.gradcalls == 0, self.gradcalls

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[76:78],
                        [[0.1928968, -0.62780447, 0.35166118],
                         [0.19572515, -0.63648426, 0.35838135]],
                        atol=1e-14, rtol=1e-7)

    def test_neldermead_initial_simplex(self):
        # Nelder-Mead simplex algorithm
        simplex = np.zeros((4, 3))
        simplex[...] = self.startparams
        for j in range(3):
            simplex[j+1, j] += 0.1

        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': False,
                    'return_all': True, 'initial_simplex': simplex}
            res = optimize.minimize(self.func, self.startparams, args=(),
                                    method='Nelder-mead', options=opts)
            params, fopt, numiter, func_calls, warnflag = (res['x'],
                                                           res['fun'],
                                                           res['nit'],
                                                           res['nfev'],
                                                           res['status'])
            assert_allclose(res['allvecs'][0], simplex[0])
        else:
            retval = optimize.fmin(self.func, self.startparams,
                                   args=(), maxiter=self.maxiter,
                                   full_output=True, disp=False, retall=False,
                                   initial_simplex=simplex)

            (params, fopt, numiter, func_calls, warnflag) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.17.0. Don't allow them to increase.
        assert self.funccalls == 100, self.funccalls
        assert self.gradcalls == 0, self.gradcalls

        # Ensure that the function behaves the same; this is from SciPy 0.15.0
        assert_allclose(self.trace[50:52],
                        [[0.14687474, -0.5103282, 0.48252111],
                         [0.14474003, -0.5282084, 0.48743951]],
                        atol=1e-14, rtol=1e-7)

    def test_neldermead_initial_simplex_bad(self):
        # Check it fails with a bad simplices
        bad_simplices = []

        simplex = np.zeros((3, 2))
        simplex[...] = self.startparams[:2]
        for j in range(2):
            simplex[j+1, j] += 0.1
        bad_simplices.append(simplex)

        simplex = np.zeros((3, 3))
        bad_simplices.append(simplex)

        for simplex in bad_simplices:
            if self.use_wrapper:
                opts = {'maxiter': self.maxiter, 'disp': False,
                        'return_all': False, 'initial_simplex': simplex}
                assert_raises(ValueError,
                              optimize.minimize,
                              self.func,
                              self.startparams,
                              args=(),
                              method='Nelder-mead',
                              options=opts)
            else:
                assert_raises(ValueError, optimize.fmin,
                              self.func, self.startparams,
                              args=(), maxiter=self.maxiter,
                              full_output=True, disp=False, retall=False,
                              initial_simplex=simplex)

    def test_ncg_negative_maxiter(self):
        # Regression test for gh-8241
        opts = {'maxiter': -1}
        result = optimize.minimize(self.func, self.startparams,
                                   method='Newton-CG', jac=self.grad,
                                   args=(), options=opts)
        assert result.status == 1

    def test_ncg(self):
        # line-search Newton conjugate gradient optimization routine
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        params = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls == 7, self.funccalls
        assert self.gradcalls <= 22, self.gradcalls  # 0.13.0
        # assert self.gradcalls <= 18, self.gradcalls  # 0.9.0
        # assert self.gradcalls == 18, self.gradcalls  # 0.8.0
        # assert self.gradcalls == 22, self.gradcalls  # 0.7.0

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)

    def test_ncg_hess(self):
        # Newton conjugate gradient with Hessian
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       hess=self.hess,
                                       args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       fhess=self.hess,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        params = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls <= 7, self.funccalls  # gh10673
        assert self.gradcalls <= 18, self.gradcalls  # 0.9.0
        # assert self.gradcalls == 18, self.gradcalls  # 0.8.0
        # assert self.gradcalls == 22, self.gradcalls  # 0.7.0

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)

    def test_ncg_hessp(self):
        # Newton conjugate gradient with Hessian times a vector p.
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': self.disp,
                    'return_all': False}
            retval = optimize.minimize(self.func, self.startparams,
                                       method='Newton-CG', jac=self.grad,
                                       hessp=self.hessp,
                                       args=(), options=opts)['x']
        else:
            retval = optimize.fmin_ncg(self.func, self.startparams, self.grad,
                                       fhess_p=self.hessp,
                                       args=(), maxiter=self.maxiter,
                                       full_output=False, disp=self.disp,
                                       retall=False)

        params = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls <= 7, self.funccalls  # gh10673
        assert self.gradcalls <= 18, self.gradcalls  # 0.9.0
        # assert self.gradcalls == 18, self.gradcalls  # 0.8.0
        # assert self.gradcalls == 22, self.gradcalls  # 0.7.0

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        assert_allclose(self.trace[3:5],
                        [[-4.35700753e-07, -5.24869435e-01, 4.87527480e-01],
                         [-4.35700753e-07, -5.24869401e-01, 4.87527774e-01]],
                        atol=1e-6, rtol=1e-7)


def test_maxfev_test():
    rng = np.random.default_rng(271707100830272976862395227613146332411)

    def cost(x):
        return rng.random(1) * 1000  # never converged problem

    for imaxfev in [1, 10, 50]:
        # "TNC" and "L-BFGS-B" also supports max function evaluation, but
        # these may violate the limit because of evaluating gradients
        # by numerical differentiation. See the discussion in PR #14805.
        for method in ['Powell', 'Nelder-Mead']:
            result = optimize.minimize(cost, rng.random(10),
                                       method=method,
                                       options={'maxfev': imaxfev})
            assert result["nfev"] == imaxfev


def test_wrap_scalar_function_with_validation():

    def func_(x):
        return x

    fcalls, func = optimize._optimize.\
        _wrap_scalar_function_maxfun_validation(func_, np.asarray(1), 5)

    for i in range(5):
        func(np.asarray(i))
        assert fcalls[0] == i+1

    msg = "Too many function calls"
    with assert_raises(optimize._optimize._MaxFuncCallError, match=msg):
        func(np.asarray(i))  # exceeded maximum function call

    fcalls, func = optimize._optimize.\
        _wrap_scalar_function_maxfun_validation(func_, np.asarray(1), 5)

    msg = "The user-provided objective function must return a scalar value."
    with assert_raises(ValueError, match=msg):
        func(np.array([1, 1]))


def test_obj_func_returns_scalar():
    match = ("The user-provided "
             "objective function must "
             "return a scalar value.")
    with assert_raises(ValueError, match=match):
        optimize.minimize(lambda x: x, np.array([1, 1]), method='BFGS')


def test_neldermead_iteration_num():
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = optimize._minimize._minimize_neldermead(optimize.rosen, x0,
                                                  xatol=1e-8)
    assert res.nit <= 339


def test_neldermead_respect_fp():
    # Nelder-Mead should respect the fp type of the input + function
    x0 = np.array([5.0, 4.0]).astype(np.float32)
    def rosen_(x):
        assert x.dtype == np.float32
        return optimize.rosen(x)

    optimize.minimize(rosen_, x0, method='Nelder-Mead')


def test_neldermead_xatol_fatol():
    # gh4484
    # test we can call with fatol, xatol specified
    def func(x):
        return x[0] ** 2 + x[1] ** 2

    optimize._minimize._minimize_neldermead(func, [1, 1], maxiter=2,
                                            xatol=1e-3, fatol=1e-3)


def test_neldermead_adaptive():
    def func(x):
        return np.sum(x ** 2)
    p0 = [0.15746215, 0.48087031, 0.44519198, 0.4223638, 0.61505159,
          0.32308456, 0.9692297, 0.4471682, 0.77411992, 0.80441652,
          0.35994957, 0.75487856, 0.99973421, 0.65063887, 0.09626474]

    res = optimize.minimize(func, p0, method='Nelder-Mead')
    assert_equal(res.success, False)

    res = optimize.minimize(func, p0, method='Nelder-Mead',
                            options={'adaptive': True})
    assert_equal(res.success, True)


def test_bounded_powell_outsidebounds():
    # With the bounded Powell method if you start outside the bounds the final
    # should still be within the bounds (provided that the user doesn't make a
    # bad choice for the `direc` argument).
    def func(x):
        return np.sum(x ** 2)
    bounds = (-1, 1), (-1, 1), (-1, 1)
    x0 = [-4, .5, -.8]

    # we're starting outside the bounds, so we should get a warning
    with assert_warns(optimize.OptimizeWarning):
        res = optimize.minimize(func, x0, bounds=bounds, method="Powell")
    assert_allclose(res.x, np.array([0.] * len(x0)), atol=1e-6)
    assert_equal(res.success, True)
    assert_equal(res.status, 0)

    # However, now if we change the `direc` argument such that the
    # set of vectors does not span the parameter space, then we may
    # not end up back within the bounds. Here we see that the first
    # parameter cannot be updated!
    direc = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    # we're starting outside the bounds, so we should get a warning
    with assert_warns(optimize.OptimizeWarning):
        res = optimize.minimize(func, x0,
                                bounds=bounds, method="Powell",
                                options={'direc': direc})
    assert_allclose(res.x, np.array([-4., 0, 0]), atol=1e-6)
    assert_equal(res.success, False)
    assert_equal(res.status, 4)


def test_bounded_powell_vs_powell():
    # here we test an example where the bounded Powell method
    # will return a different result than the standard Powell
    # method.

    # first we test a simple example where the minimum is at
    # the origin and the minimum that is within the bounds is
    # larger than the minimum at the origin.
    def func(x):
        return np.sum(x ** 2)
    bounds = (-5, -1), (-10, -0.1), (1, 9.2), (-4, 7.6), (-15.9, -2)
    x0 = [-2.1, -5.2, 1.9, 0, -2]

    options = {'ftol': 1e-10, 'xtol': 1e-10}

    res_powell = optimize.minimize(func, x0, method="Powell", options=options)
    assert_allclose(res_powell.x, 0., atol=1e-6)
    assert_allclose(res_powell.fun, 0., atol=1e-6)

    res_bounded_powell = optimize.minimize(func, x0, options=options,
                                           bounds=bounds,
                                           method="Powell")
    p = np.array([-1, -0.1, 1, 0, -2])
    assert_allclose(res_bounded_powell.x, p, atol=1e-6)
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-6)

    # now we test bounded Powell but with a mix of inf bounds.
    bounds = (None, -1), (-np.inf, -.1), (1, np.inf), (-4, None), (-15.9, -2)
    res_bounded_powell = optimize.minimize(func, x0, options=options,
                                           bounds=bounds,
                                           method="Powell")
    p = np.array([-1, -0.1, 1, 0, -2])
    assert_allclose(res_bounded_powell.x, p, atol=1e-6)
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-6)

    # next we test an example where the global minimum is within
    # the bounds, but the bounded Powell method performs better
    # than the standard Powell method.
    def func(x):
        t = np.sin(-x[0]) * np.cos(x[1]) * np.sin(-x[0] * x[1]) * np.cos(x[1])
        t -= np.cos(np.sin(x[1] * x[2]) * np.cos(x[2]))
        return t**2

    bounds = [(-2, 5)] * 3
    x0 = [-0.5, -0.5, -0.5]

    res_powell = optimize.minimize(func, x0, method="Powell")
    res_bounded_powell = optimize.minimize(func, x0,
                                           bounds=bounds,
                                           method="Powell")
    assert_allclose(res_powell.fun, 0.007136253919761627, atol=1e-6)
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-6)

    # next we test the previous example where the we provide Powell
    # with (-inf, inf) bounds, and compare it to providing Powell
    # with no bounds. They should end up the same.
    bounds = [(-np.inf, np.inf)] * 3

    res_bounded_powell = optimize.minimize(func, x0,
                                           bounds=bounds,
                                           method="Powell")
    assert_allclose(res_powell.fun, res_bounded_powell.fun, atol=1e-6)
    assert_allclose(res_powell.nfev, res_bounded_powell.nfev, atol=1e-6)
    assert_allclose(res_powell.x, res_bounded_powell.x, atol=1e-6)

    # now test when x0 starts outside of the bounds.
    x0 = [45.46254415, -26.52351498, 31.74830248]
    bounds = [(-2, 5)] * 3
    # we're starting outside the bounds, so we should get a warning
    with assert_warns(optimize.OptimizeWarning):
        res_bounded_powell = optimize.minimize(func, x0,
                                               bounds=bounds,
                                               method="Powell")
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-6)


def test_onesided_bounded_powell_stability():
    # When the Powell method is bounded on only one side, a
    # np.tan transform is done in order to convert it into a
    # completely bounded problem. Here we do some simple tests
    # of one-sided bounded Powell where the optimal solutions
    # are large to test the stability of the transformation.
    kwargs = {'method': 'Powell',
              'bounds': [(-np.inf, 1e6)] * 3,
              'options': {'ftol': 1e-8, 'xtol': 1e-8}}
    x0 = [1, 1, 1]

    # df/dx is constant.
    def f(x):
        return -np.sum(x)
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -3e6, atol=1e-4)

    # df/dx gets smaller and smaller.
    def f(x):
        return -np.abs(np.sum(x)) ** (0.1) * (1 if np.all(x > 0) else -1)

    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -(3e6) ** (0.1))

    # df/dx gets larger and larger.
    def f(x):
        return -np.abs(np.sum(x)) ** 10 * (1 if np.all(x > 0) else -1)

    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -(3e6) ** 10, rtol=1e-7)

    # df/dx gets larger for some of the variables and smaller for others.
    def f(x):
        t = -np.abs(np.sum(x[:2])) ** 5 - np.abs(np.sum(x[2:])) ** (0.1)
        t *= (1 if np.all(x > 0) else -1)
        return t

    kwargs['bounds'] = [(-np.inf, 1e3)] * 3
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -(2e3) ** 5 - (1e6) ** (0.1), rtol=1e-7)


class TestOptimizeWrapperDisp(CheckOptimizeParameterized):
    use_wrapper = True
    disp = True


class TestOptimizeWrapperNoDisp(CheckOptimizeParameterized):
    use_wrapper = True
    disp = False


class TestOptimizeNoWrapperDisp(CheckOptimizeParameterized):
    use_wrapper = False
    disp = True


class TestOptimizeNoWrapperNoDisp(CheckOptimizeParameterized):
    use_wrapper = False
    disp = False


class TestOptimizeSimple(CheckOptimize):

    def test_bfgs_nan(self):
        # Test corner case where nan is fed to optimizer.  See gh-2067.
        def func(x):
            return x
        def fprime(x):
            return np.ones_like(x)
        x0 = [np.nan]
        with np.errstate(over='ignore', invalid='ignore'):
            x = optimize.fmin_bfgs(func, x0, fprime, disp=False)
            assert np.isnan(func(x))

    def test_bfgs_nan_return(self):
        # Test corner cases where fun returns NaN. See gh-4793.

        # First case: NaN from first call.
        def func(x):
            return np.nan
        with np.errstate(invalid='ignore'):
            result = optimize.minimize(func, 0)

        assert np.isnan(result['fun'])
        assert result['success'] is False

        # Second case: NaN from second call.
        def func(x):
            return 0 if x == 0 else np.nan
        def fprime(x):
            return np.ones_like(x)  # Steer away from zero.
        with np.errstate(invalid='ignore'):
            result = optimize.minimize(func, 0, jac=fprime)

        assert np.isnan(result['fun'])
        assert result['success'] is False

    def test_bfgs_numerical_jacobian(self):
        # BFGS with numerical Jacobian and a vector epsilon parameter.
        # define the epsilon parameter using a random vector
        epsilon = np.sqrt(np.spacing(1.)) * np.random.rand(len(self.solution))

        params = optimize.fmin_bfgs(self.func, self.startparams,
                                    epsilon=epsilon, args=(),
                                    maxiter=self.maxiter, disp=False)

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_finite_differences_jac(self):
        methods = ['BFGS', 'CG', 'TNC']
        jacs = ['2-point', '3-point', None]
        for method, jac in itertools.product(methods, jacs):
            result = optimize.minimize(self.func, self.startparams,
                                       method=method, jac=jac)
            assert_allclose(self.func(result.x), self.func(self.solution),
                            atol=1e-6)

    def test_finite_differences_hess(self):
        # test that all the methods that require hess can use finite-difference
        # For Newton-CG, trust-ncg, trust-krylov the FD estimated hessian is
        # wrapped in a hessp function
        # dogleg, trust-exact actually require true hessians at the moment, so
        # they're excluded.
        methods = ['trust-constr', 'Newton-CG', 'trust-ncg', 'trust-krylov']
        hesses = FD_METHODS + (optimize.BFGS,)
        for method, hess in itertools.product(methods, hesses):
            if hess is optimize.BFGS:
                hess = hess()
            result = optimize.minimize(self.func, self.startparams,
                                       method=method, jac=self.grad,
                                       hess=hess)
            assert result.success

        # check that the methods demand some sort of Hessian specification
        # Newton-CG creates its own hessp, and trust-constr doesn't need a hess
        # specified either
        methods = ['trust-ncg', 'trust-krylov', 'dogleg', 'trust-exact']
        for method in methods:
            with pytest.raises(ValueError):
                optimize.minimize(self.func, self.startparams,
                                  method=method, jac=self.grad,
                                  hess=None)

    def test_bfgs_gh_2169(self):
        def f(x):
            if x < 0:
                return 1.79769313e+308
            else:
                return x + 1./x
        xs = optimize.fmin_bfgs(f, [10.], disp=False)
        assert_allclose(xs, 1.0, rtol=1e-4, atol=1e-4)

    def test_bfgs_double_evaluations(self):
        # check BFGS does not evaluate twice in a row at same point
        def f(x):
            xp = x[0]
            assert xp not in seen
            seen.add(xp)
            return 10*x**2, 20*x

        seen = set()
        optimize.minimize(f, -100, method='bfgs', jac=True, tol=1e-7)

    def test_l_bfgs_b(self):
        # limited-memory bound-constrained BFGS algorithm
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams,
                                        self.grad, args=(),
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

        # Ensure that function call counts are 'known good'; these are from
        # SciPy 0.7.0. Don't allow them to increase.
        assert self.funccalls == 7, self.funccalls
        assert self.gradcalls == 5, self.gradcalls

        # Ensure that the function behaves the same; this is from SciPy 0.7.0
        # test fixed in gh10673
        assert_allclose(self.trace[3:5],
                        [[8.117083e-16, -5.196198e-01, 4.897617e-01],
                         [0., -0.52489628, 0.48753042]],
                        atol=1e-14, rtol=1e-7)

    def test_l_bfgs_b_numjac(self):
        # L-BFGS-B with numerical Jacobian
        retval = optimize.fmin_l_bfgs_b(self.func, self.startparams,
                                        approx_grad=True,
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_l_bfgs_b_funjac(self):
        # L-BFGS-B with combined objective function and Jacobian
        def fun(x):
            return self.func(x), self.grad(x)

        retval = optimize.fmin_l_bfgs_b(fun, self.startparams,
                                        maxiter=self.maxiter)

        (params, fopt, d) = retval

        assert_allclose(self.func(params), self.func(self.solution),
                        atol=1e-6)

    def test_l_bfgs_b_maxiter(self):
        # gh7854
        # Ensure that not more than maxiters are ever run.
        class Callback:
            def __init__(self):
                self.nit = 0
                self.fun = None
                self.x = None

            def __call__(self, x):
                self.x = x
                self.fun = optimize.rosen(x)
                self.nit += 1

        c = Callback()
        res = optimize.minimize(optimize.rosen, [0., 0.], method='l-bfgs-b',
                                callback=c, options={'maxiter': 5})

        assert_equal(res.nit, 5)
        assert_almost_equal(res.x, c.x)
        assert_almost_equal(res.fun, c.fun)
        assert_equal(res.status, 1)
        assert res.success is False
        assert_equal(res.message,
                     'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT')

    def test_minimize_l_bfgs_b(self):
        # Minimize with L-BFGS-B method
        opts = {'disp': False, 'maxiter': self.maxiter}
        r = optimize.minimize(self.func, self.startparams,
                              method='L-BFGS-B', jac=self.grad,
                              options=opts)
        assert_allclose(self.func(r.x), self.func(self.solution),
                        atol=1e-6)
        assert self.gradcalls == r.njev

        self.funccalls = self.gradcalls = 0
        # approximate jacobian
        ra = optimize.minimize(self.func, self.startparams,
                               method='L-BFGS-B', options=opts)
        # check that function evaluations in approximate jacobian are counted
        # assert_(ra.nfev > r.nfev)
        assert self.funccalls == ra.nfev
        assert_allclose(self.func(ra.x), self.func(self.solution),
                        atol=1e-6)

        self.funccalls = self.gradcalls = 0
        # approximate jacobian
        ra = optimize.minimize(self.func, self.startparams, jac='3-point',
                               method='L-BFGS-B', options=opts)
        assert self.funccalls == ra.nfev
        assert_allclose(self.func(ra.x), self.func(self.solution),
                        atol=1e-6)

    def test_minimize_l_bfgs_b_ftol(self):
        # Check that the `ftol` parameter in l_bfgs_b works as expected
        v0 = None
        for tol in [1e-1, 1e-4, 1e-7, 1e-10]:
            opts = {'disp': False, 'maxiter': self.maxiter, 'ftol': tol}
            sol = optimize.minimize(self.func, self.startparams,
                                    method='L-BFGS-B', jac=self.grad,
                                    options=opts)
            v = self.func(sol.x)

            if v0 is None:
                v0 = v
            else:
                assert v < v0

            assert_allclose(v, self.func(self.solution), rtol=tol)

    def test_minimize_l_bfgs_maxls(self):
        # check that the maxls is passed down to the Fortran routine
        sol = optimize.minimize(optimize.rosen, np.array([-1.2, 1.0]),
                                method='L-BFGS-B', jac=optimize.rosen_der,
                                options={'disp': False, 'maxls': 1})
        assert not sol.success

    def test_minimize_l_bfgs_b_maxfun_interruption(self):
        # gh-6162
        f = optimize.rosen
        g = optimize.rosen_der
        values = []
        x0 = np.full(7, 1000)

        def objfun(x):
            value = f(x)
            values.append(value)
            return value

        # Look for an interesting test case.
        # Request a maxfun that stops at a particularly bad function
        # evaluation somewhere between 100 and 300 evaluations.
        low, medium, high = 30, 100, 300
        optimize.fmin_l_bfgs_b(objfun, x0, fprime=g, maxfun=high)
        v, k = max((y, i) for i, y in enumerate(values[medium:]))
        maxfun = medium + k
        # If the minimization strategy is reasonable,
        # the minimize() result should not be worse than the best
        # of the first 30 function evaluations.
        target = min(values[:low])
        xmin, fmin, d = optimize.fmin_l_bfgs_b(f, x0, fprime=g, maxfun=maxfun)
        assert_array_less(fmin, target)

    def test_custom(self):
        # This function comes from the documentation example.
        def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1,
                    maxiter=100, callback=None, **options):
            bestx = x0
            besty = fun(x0)
            funcalls = 1
            niter = 0
            improved = True
            stop = False

            while improved and not stop and niter < maxiter:
                improved = False
                niter += 1
                for dim in range(np.size(x0)):
                    for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                        testx = np.copy(bestx)
                        testx[dim] = s
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            besty = testy
                            bestx = testx
                            improved = True
                    if callback is not None:
                        callback(bestx)
                    if maxfev is not None and funcalls >= maxfev:
                        stop = True
                        break

            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter,
                                           nfev=funcalls, success=(niter > 1))

        x0 = [1.35, 0.9, 0.8, 1.1, 1.2]
        res = optimize.minimize(optimize.rosen, x0, method=custmin,
                                options=dict(stepsize=0.05))
        assert_allclose(res.x, 1.0, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="output not reliable on all platforms")
    def test_gh13321(self, capfd):
        # gh-13321 reported issues with console output in fmin_l_bfgs_b;
        # check that iprint=0 works.
        kwargs = {'func': optimize.rosen, 'x0': [4, 3],
                  'fprime': optimize.rosen_der, 'bounds': ((3, 5), (3, 5))}

        # "L-BFGS-B" is always in output; should show when iprint >= 0
        # "At iterate" is iterate info; should show when iprint >= 1

        optimize.fmin_l_bfgs_b(**kwargs, iprint=-1)
        out, _ = capfd.readouterr()
        assert "L-BFGS-B" not in out and "At iterate" not in out

        optimize.fmin_l_bfgs_b(**kwargs, iprint=0)
        out, _ = capfd.readouterr()
        assert "L-BFGS-B" in out and "At iterate" not in out

        optimize.fmin_l_bfgs_b(**kwargs, iprint=1)
        out, _ = capfd.readouterr()
        assert "L-BFGS-B" in out and "At iterate" in out

        # `disp is not None` overrides `iprint` behavior
        # `disp=0` should suppress all output
        # `disp=1` should be the same as `iprint = 1`

        optimize.fmin_l_bfgs_b(**kwargs, iprint=1, disp=False)
        out, _ = capfd.readouterr()
        assert "L-BFGS-B" not in out and "At iterate" not in out

        optimize.fmin_l_bfgs_b(**kwargs, iprint=-1, disp=True)
        out, _ = capfd.readouterr()
        assert "L-BFGS-B" in out and "At iterate" in out

    def test_gh10771(self):
        # check that minimize passes bounds and constraints to a custom
        # minimizer without altering them.
        bounds = [(-2, 2), (0, 3)]
        constraints = 'constraints'

        def custmin(fun, x0, **options):
            assert options['bounds'] is bounds
            assert options['constraints'] is constraints
            return optimize.OptimizeResult()

        x0 = [1, 1]
        optimize.minimize(optimize.rosen, x0, method=custmin,
                          bounds=bounds, constraints=constraints)

    def test_minimize_tol_parameter(self):
        # Check that the minimize() tol= argument does something
        def func(z):
            x, y = z
            return x**2*y**2 + x**4 + 1

        def dfunc(z):
            x, y = z
            return np.array([2*x*y**2 + 4*x**3, 2*x**2*y])

        for method in ['nelder-mead', 'powell', 'cg', 'bfgs',
                       'newton-cg', 'l-bfgs-b', 'tnc',
                       'cobyla', 'slsqp']:
            if method in ('nelder-mead', 'powell', 'cobyla'):
                jac = None
            else:
                jac = dfunc

            sol1 = optimize.minimize(func, [1, 1], jac=jac, tol=1e-10,
                                     method=method)
            sol2 = optimize.minimize(func, [1, 1], jac=jac, tol=1.0,
                                     method=method)
            assert func(sol1.x) < func(sol2.x), \
                   f"{method}: {func(sol1.x)} vs. {func(sol2.x)}"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # See gh-18547
    @pytest.mark.parametrize('method',
                             ['fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs',
                              'fmin_ncg', 'fmin_l_bfgs_b', 'fmin_tnc',
                              'fmin_slsqp'] + MINIMIZE_METHODS)
    def test_minimize_callback_copies_array(self, method):
        # Check that arrays passed to callbacks are not modified
        # inplace by the optimizer afterward

        if method in ('fmin_tnc', 'fmin_l_bfgs_b'):
            def func(x):
                return optimize.rosen(x), optimize.rosen_der(x)
        else:
            func = optimize.rosen
            jac = optimize.rosen_der
            hess = optimize.rosen_hess

        x0 = np.zeros(10)

        # Set options
        kwargs = {}
        if method.startswith('fmin'):
            routine = getattr(optimize, method)
            if method == 'fmin_slsqp':
                kwargs['iter'] = 5
            elif method == 'fmin_tnc':
                kwargs['maxfun'] = 100
            elif method in ('fmin', 'fmin_powell'):
                kwargs['maxiter'] = 3500
            else:
                kwargs['maxiter'] = 5
        else:
            def routine(*a, **kw):
                kw['method'] = method
                return optimize.minimize(*a, **kw)

            if method == 'tnc':
                kwargs['options'] = dict(maxfun=100)
            else:
                kwargs['options'] = dict(maxiter=5)

        if method in ('fmin_ncg',):
            kwargs['fprime'] = jac
        elif method in ('newton-cg',):
            kwargs['jac'] = jac
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = jac
            kwargs['hess'] = hess

        # Run with callback
        results = []

        def callback(x, *args, **kwargs):
            assert not isinstance(x, optimize.OptimizeResult)
            results.append((x, np.copy(x)))

        routine(func, x0, callback=callback, **kwargs)

        # Check returned arrays coincide with their copies
        # and have no memory overlap
        assert len(results) > 2
        assert all(np.all(x == y) for x, y in results)
        combinations = itertools.combinations(results, 2)
        assert not any(np.may_share_memory(x[0], y[0]) for x, y in combinations)

    @pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg',
                                        'bfgs', 'newton-cg', 'l-bfgs-b',
                                        'tnc', 'cobyla', 'slsqp'])
    def test_no_increase(self, method):
        # Check that the solver doesn't return a value worse than the
        # initial point.

        def func(x):
            return (x - 1)**2

        def bad_grad(x):
            # purposefully invalid gradient function, simulates a case
            # where line searches start failing
            return 2*(x - 1) * (-1) - 2

        x0 = np.array([2.0])
        f0 = func(x0)
        jac = bad_grad
        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)
        if method in ['nelder-mead', 'powell', 'cobyla']:
            jac = None
        sol = optimize.minimize(func, x0, jac=jac, method=method,
                                options=options)
        assert_equal(func(sol.x), sol.fun)

        if method == 'slsqp':
            pytest.xfail("SLSQP returns slightly worse")
        assert func(sol.x) <= f0

    def test_slsqp_respect_bounds(self):
        # Regression test for gh-3108
        def f(x):
            return sum((x - np.array([1., 2., 3., 4.]))**2)

        def cons(x):
            a = np.array([[-1, -1, -1, -1], [-3, -3, -2, -1]])
            return np.concatenate([np.dot(a, x) + np.array([5, 10]), x])

        x0 = np.array([0.5, 1., 1.5, 2.])
        res = optimize.minimize(f, x0, method='slsqp',
                                constraints={'type': 'ineq', 'fun': cons})
        assert_allclose(res.x, np.array([0., 2, 5, 8])/3, atol=1e-12)

    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                                        'Newton-CG', 'L-BFGS-B', 'SLSQP',
                                        'trust-constr', 'dogleg', 'trust-ncg',
                                        'trust-exact', 'trust-krylov'])
    def test_respect_maxiter(self, method):
        # Check that the number of iterations equals max_iter, assuming
        # convergence doesn't establish before
        MAXITER = 4

        x0 = np.zeros(10)

        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der,
                            optimize.rosen_hess, None, None)

        # Set options
        kwargs = {'method': method, 'options': dict(maxiter=MAXITER)}

        if method in ('Newton-CG',):
            kwargs['jac'] = sf.grad
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = sf.grad
            kwargs['hess'] = sf.hess

        sol = optimize.minimize(sf.fun, x0, **kwargs)
        assert sol.nit == MAXITER
        assert sol.nfev >= sf.nfev
        if hasattr(sol, 'njev'):
            assert sol.njev >= sf.ngev

        # method specific tests
        if method == 'SLSQP':
            assert sol.status == 9  # Iteration limit reached

    @pytest.mark.parametrize('method', ['Nelder-Mead', 'Powell',
                                        'fmin', 'fmin_powell'])
    def test_runtime_warning(self, method):
        x0 = np.zeros(10)
        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der,
                            optimize.rosen_hess, None, None)
        options = {"maxiter": 1, "disp": True}
        with pytest.warns(RuntimeWarning,
                          match=r'Maximum number of iterations'):
            if method.startswith('fmin'):
                routine = getattr(optimize, method)
                routine(sf.fun, x0, **options)
            else:
                optimize.minimize(sf.fun, x0, method=method, options=options)

    def test_respect_maxiter_trust_constr_ineq_constraints(self):
        # special case of minimization with trust-constr and inequality
        # constraints to check maxiter limit is obeyed when using internal
        # method 'tr_interior_point'
        MAXITER = 4
        f = optimize.rosen
        jac = optimize.rosen_der
        hess = optimize.rosen_hess

        def fun(x):
            return np.array([0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])
        cons = ({'type': 'ineq',
                 'fun': fun},)

        x0 = np.zeros(10)
        sol = optimize.minimize(f, x0, constraints=cons, jac=jac, hess=hess,
                                method='trust-constr',
                                options=dict(maxiter=MAXITER))
        assert sol.nit == MAXITER

    def test_minimize_automethod(self):
        def f(x):
            return x**2

        def cons(x):
            return x - 2

        x0 = np.array([10.])
        sol_0 = optimize.minimize(f, x0)
        sol_1 = optimize.minimize(f, x0, constraints=[{'type': 'ineq',
                                                       'fun': cons}])
        sol_2 = optimize.minimize(f, x0, bounds=[(5, 10)])
        sol_3 = optimize.minimize(f, x0,
                                  constraints=[{'type': 'ineq', 'fun': cons}],
                                  bounds=[(5, 10)])
        sol_4 = optimize.minimize(f, x0,
                                  constraints=[{'type': 'ineq', 'fun': cons}],
                                  bounds=[(1, 10)])
        for sol in [sol_0, sol_1, sol_2, sol_3, sol_4]:
            assert sol.success
        assert_allclose(sol_0.x, 0, atol=1e-7)
        assert_allclose(sol_1.x, 2, atol=1e-7)
        assert_allclose(sol_2.x, 5, atol=1e-7)
        assert_allclose(sol_3.x, 5, atol=1e-7)
        assert_allclose(sol_4.x, 2, atol=1e-7)

    def test_minimize_coerce_args_param(self):
        # Regression test for gh-3503
        def Y(x, c):
            return np.sum((x-c)**2)

        def dY_dx(x, c=None):
            return 2*(x-c)

        c = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        xinit = np.random.randn(len(c))
        optimize.minimize(Y, xinit, jac=dY_dx, args=(c), method="BFGS")

    def test_initial_step_scaling(self):
        # Check that optimizer initial step is not huge even if the
        # function and gradients are

        scales = [1e-50, 1, 1e50]
        methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']

        def f(x):
            if first_step_size[0] is None and x[0] != x0[0]:
                first_step_size[0] = abs(x[0] - x0[0])
            if abs(x).max() > 1e4:
                raise AssertionError("Optimization stepped far away!")
            return scale*(x[0] - 1)**2

        def g(x):
            return np.array([scale*(x[0] - 1)])

        for scale, method in itertools.product(scales, methods):
            if method in ('CG', 'BFGS'):
                options = dict(gtol=scale*1e-8)
            else:
                options = dict()

            if scale < 1e-10 and method in ('L-BFGS-B', 'Newton-CG'):
                # XXX: return initial point if they see small gradient
                continue

            x0 = [-1.0]
            first_step_size = [None]
            res = optimize.minimize(f, x0, jac=g, method=method,
                                    options=options)

            err_msg = f"{method} {scale}: {first_step_size}: {res}"

            assert res.success, err_msg
            assert_allclose(res.x, [1.0], err_msg=err_msg)
            assert res.nit <= 3, err_msg

            if scale > 1e-10:
                if method in ('CG', 'BFGS'):
                    assert_allclose(first_step_size[0], 1.01, err_msg=err_msg)
                else:
                    # Newton-CG and L-BFGS-B use different logic for the first
                    # step, but are both scaling invariant with step sizes ~ 1
                    assert first_step_size[0] > 0.5 and first_step_size[0] < 3, err_msg
            else:
                # step size has upper bound of ||grad||, so line
                # search makes many small steps
                pass

    @pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg', 'bfgs',
                                        'newton-cg', 'l-bfgs-b', 'tnc',
                                        'cobyla', 'slsqp', 'trust-constr',
                                        'dogleg', 'trust-ncg', 'trust-exact',
                                        'trust-krylov'])
    def test_nan_values(self, method):
        # Check nan values result to failed exit status
        np.random.seed(1234)

        count = [0]

        def func(x):
            return np.nan

        def func2(x):
            count[0] += 1
            if count[0] > 2:
                return np.nan
            else:
                return np.random.rand()

        def grad(x):
            return np.array([1.0])

        def hess(x):
            return np.array([[1.0]])

        x0 = np.array([1.0])

        needs_grad = method in ('newton-cg', 'trust-krylov', 'trust-exact',
                                'trust-ncg', 'dogleg')
        needs_hess = method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                                'dogleg')

        funcs = [func, func2]
        grads = [grad] if needs_grad else [grad, None]
        hesss = [hess] if needs_hess else [hess, None]
        options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.*")
            sup.filter(RuntimeWarning, ".*does not use Hessian.*")
            sup.filter(RuntimeWarning, ".*does not use gradient.*")

            for f, g, h in itertools.product(funcs, grads, hesss):
                count = [0]
                sol = optimize.minimize(f, x0, jac=g, hess=h, method=method,
                                        options=options)
                assert_equal(sol.success, False)

    @pytest.mark.parametrize('method', ['nelder-mead', 'cg', 'bfgs',
                                        'l-bfgs-b', 'tnc',
                                        'cobyla', 'slsqp', 'trust-constr',
                                        'dogleg', 'trust-ncg', 'trust-exact',
                                        'trust-krylov'])
    def test_duplicate_evaluations(self, method):
        # check that there are no duplicate evaluations for any methods
        jac = hess = None
        if method in ('newton-cg', 'trust-krylov', 'trust-exact',
                      'trust-ncg', 'dogleg'):
            jac = self.grad
        if method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                      'dogleg'):
            hess = self.hess

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            # for trust-constr
            sup.filter(UserWarning, "delta_grad == 0.*")
            optimize.minimize(self.func, self.startparams,
                              method=method, jac=jac, hess=hess)

        for i in range(1, len(self.trace)):
            if np.array_equal(self.trace[i - 1], self.trace[i]):
                raise RuntimeError(
                    f"Duplicate evaluations made by {method}")

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('method', MINIMIZE_METHODS_NEW_CB)
    @pytest.mark.parametrize('new_cb_interface', [0, 1, 2])
    def test_callback_stopiteration(self, method, new_cb_interface):
        # Check that if callback raises StopIteration, optimization
        # terminates with the same result as if iterations were limited

        def f(x):
            f.flag = False  # check that f isn't called after StopIteration
            return optimize.rosen(x)
        f.flag = False

        def g(x):
            f.flag = False
            return optimize.rosen_der(x)

        def h(x):
            f.flag = False
            return optimize.rosen_hess(x)

        maxiter = 5

        if new_cb_interface == 1:
            def callback_interface(*, intermediate_result):
                assert intermediate_result.fun == f(intermediate_result.x)
                callback()
        elif new_cb_interface == 2:
            class Callback:
                def __call__(self, intermediate_result: OptimizeResult):
                    assert intermediate_result.fun == f(intermediate_result.x)
                    callback()
            callback_interface = Callback()
        else:
            def callback_interface(xk, *args):  # type: ignore[misc]
                callback()

        def callback():
            callback.i += 1
            callback.flag = False
            if callback.i == maxiter:
                callback.flag = True
                raise StopIteration()
        callback.i = 0
        callback.flag = False

        kwargs = {'x0': [1.1]*5, 'method': method,
                  'fun': f, 'jac': g, 'hess': h}

        res = optimize.minimize(**kwargs, callback=callback_interface)
        if method == 'nelder-mead':
            maxiter = maxiter + 1  # nelder-mead counts differently
        ref = optimize.minimize(**kwargs, options={'maxiter': maxiter})
        assert res.fun == ref.fun
        assert_equal(res.x, ref.x)
        assert res.nit == ref.nit == maxiter
        assert res.status == (3 if method == 'trust-constr' else 99)

    def test_ndim_error(self):
        msg = "'x0' must only have one dimension."
        with assert_raises(ValueError, match=msg):
            optimize.minimize(lambda x: x, np.ones((2, 1)))

    @pytest.mark.parametrize('method', ('nelder-mead', 'l-bfgs-b', 'tnc',
                                        'powell', 'cobyla', 'trust-constr'))
    def test_minimize_invalid_bounds(self, method):
        def f(x):
            return np.sum(x**2)

        bounds = Bounds([1, 2], [3, 4])
        msg = 'The number of bounds is not compatible with the length of `x0`.'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)

        bounds = Bounds([1, 6, 1], [3, 4, 2])
        msg = 'An upper bound is less than the corresponding lower bound.'
        with pytest.raises(ValueError, match=msg):
            optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)

    @pytest.mark.parametrize('method', ['bfgs', 'cg', 'newton-cg', 'powell'])
    def test_minimize_warnings_gh1953(self, method):
        # test that minimize methods produce warnings rather than just using
        # `print`; see gh-1953.
        kwargs = {} if method=='powell' else {'jac': optimize.rosen_der}
        warning_type = (RuntimeWarning if method=='powell'
                        else optimize.OptimizeWarning)

        options = {'disp': True, 'maxiter': 10}
        with pytest.warns(warning_type, match='Maximum number'):
            optimize.minimize(lambda x: optimize.rosen(x), [0, 0],
                              method=method, options=options, **kwargs)

        options['disp'] = False
        optimize.minimize(lambda x: optimize.rosen(x), [0, 0],
                          method=method, options=options, **kwargs)


@pytest.mark.parametrize(
    'method',
    ['l-bfgs-b', 'tnc', 'Powell', 'Nelder-Mead']
)
def test_minimize_with_scalar(method):
    # checks that minimize works with a scalar being provided to it.
    def f(x):
        return np.sum(x ** 2)

    res = optimize.minimize(f, 17, bounds=[(-100, 100)], method=method)
    assert res.success
    assert_allclose(res.x, [0.0], atol=1e-5)


class TestLBFGSBBounds:
    def setup_method(self):
        self.bounds = ((1, None), (None, None))
        self.solution = (1, 0)

    def fun(self, x, p=2.0):
        return 1.0 / p * (x[0]**p + x[1]**p)

    def jac(self, x, p=2.0):
        return x**(p - 1)

    def fj(self, x, p=2.0):
        return self.fun(x, p), self.jac(x, p)

    def test_l_bfgs_b_bounds(self):
        x, f, d = optimize.fmin_l_bfgs_b(self.fun, [0, -1],
                                         fprime=self.jac,
                                         bounds=self.bounds)
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-6)

    def test_l_bfgs_b_funjac(self):
        # L-BFGS-B with fun and jac combined and extra arguments
        x, f, d = optimize.fmin_l_bfgs_b(self.fj, [0, -1], args=(2.0, ),
                                         bounds=self.bounds)
        assert d['warnflag'] == 0, d['task']
        assert_allclose(x, self.solution, atol=1e-6)

    def test_minimize_l_bfgs_b_bounds(self):
        # Minimize with method='L-BFGS-B' with bounds
        res = optimize.minimize(self.fun, [0, -1], method='L-BFGS-B',
                                jac=self.jac, bounds=self.bounds)
        assert res['success'], res['message']
        assert_allclose(res.x, self.solution, atol=1e-6)

    @pytest.mark.parametrize('bounds', [
        ([(10, 1), (1, 10)]),
        ([(1, 10), (10, 1)]),
        ([(10, 1), (10, 1)])
    ])
    def test_minimize_l_bfgs_b_incorrect_bounds(self, bounds):
        with pytest.raises(ValueError, match='.*bound.*'):
            optimize.minimize(self.fun, [0, -1], method='L-BFGS-B',
                              jac=self.jac, bounds=bounds)

    def test_minimize_l_bfgs_b_bounds_FD(self):
        # test that initial starting value outside bounds doesn't raise
        # an error (done with clipping).
        # test all different finite differences combos, with and without args

        jacs = ['2-point', '3-point', None]
        argss = [(2.,), ()]
        for jac, args in itertools.product(jacs, argss):
            res = optimize.minimize(self.fun, [0, -1], args=args,
                                    method='L-BFGS-B',
                                    jac=jac, bounds=self.bounds,
                                    options={'finite_diff_rel_step': None})
            assert res['success'], res['message']
            assert_allclose(res.x, self.solution, atol=1e-6)


class TestOptimizeScalar:
    def setup_method(self):
        self.solution = 1.5

    def fun(self, x, a=1.5):
        """Objective function"""
        return (x - a)**2 - 0.8

    def test_brent(self):
        x = optimize.brent(self.fun)
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.brent(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.brent(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-6)

        x = optimize.brent(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-6)

        message = r"\(f\(xb\) < f\(xa\)\) and \(f\(xb\) < f\(xc\)\)"
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(-1, 0, 1))

        message = r"\(xa < xb\) and \(xb < xc\)"
        with pytest.raises(ValueError, match=message):
            optimize.brent(self.fun, brack=(0, -1, 1))

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_golden(self):
        x = optimize.golden(self.fun)
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.golden(self.fun, brack=(-3, -2))
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.golden(self.fun, full_output=True)
        assert_allclose(x[0], self.solution, atol=1e-6)

        x = optimize.golden(self.fun, brack=(-15, -1, 15))
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.golden(self.fun, tol=0)
        assert_allclose(x, self.solution)

        maxiter_test_cases = [0, 1, 5]
        for maxiter in maxiter_test_cases:
            x0 = optimize.golden(self.fun, maxiter=0, full_output=True)
            x = optimize.golden(self.fun, maxiter=maxiter, full_output=True)
            nfev0, nfev = x0[2], x[2]
            assert_equal(nfev - nfev0, maxiter)

        message = r"\(f\(xb\) < f\(xa\)\) and \(f\(xb\) < f\(xc\)\)"
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(-1, 0, 1))

        message = r"\(xa < xb\) and \(xb < xc\)"
        with pytest.raises(ValueError, match=message):
            optimize.golden(self.fun, brack=(0, -1, 1))

    def test_fminbound(self):
        x = optimize.fminbound(self.fun, 0, 1)
        assert_allclose(x, 1, atol=1e-4)

        x = optimize.fminbound(self.fun, 1, 5)
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.fminbound(self.fun, np.array([1]), np.array([5]))
        assert_allclose(x, self.solution, atol=1e-6)
        assert_raises(ValueError, optimize.fminbound, self.fun, 5, 1)

    def test_fminbound_scalar(self):
        with pytest.raises(ValueError, match='.*must be finite scalars.*'):
            optimize.fminbound(self.fun, np.zeros((1, 2)), 1)

        x = optimize.fminbound(self.fun, 1, np.array(5))
        assert_allclose(x, self.solution, atol=1e-6)

    def test_gh11207(self):
        def fun(x):
            return x**2
        optimize.fminbound(fun, 0, 0)

    def test_minimize_scalar(self):
        # combine all tests above for the minimize_scalar wrapper
        x = optimize.minimize_scalar(self.fun).x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='Brent')
        assert x.success

        x = optimize.minimize_scalar(self.fun, method='Brent',
                                     options=dict(maxiter=3))
        assert not x.success

        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2),
                                     args=(1.5, ), method='Brent').x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='Brent',
                                     args=(1.5,)).x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15),
                                     args=(1.5, ), method='Brent').x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-3, -2),
                                     args=(1.5, ), method='golden').x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, method='golden',
                                     args=(1.5,)).x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15),
                                     args=(1.5, ), method='golden').x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bounds=(0, 1), args=(1.5,),
                                     method='Bounded').x
        assert_allclose(x, 1, atol=1e-4)

        x = optimize.minimize_scalar(self.fun, bounds=(1, 5), args=(1.5, ),
                                     method='bounded').x
        assert_allclose(x, self.solution, atol=1e-6)

        x = optimize.minimize_scalar(self.fun, bounds=(np.array([1]),
                                                       np.array([5])),
                                     args=(np.array([1.5]), ),
                                     method='bounded').x
        assert_allclose(x, self.solution, atol=1e-6)

        assert_raises(ValueError, optimize.minimize_scalar, self.fun,
                      bounds=(5, 1), method='bounded', args=(1.5, ))

        assert_raises(ValueError, optimize.minimize_scalar, self.fun,
                      bounds=(np.zeros(2), 1), method='bounded', args=(1.5, ))

        x = optimize.minimize_scalar(self.fun, bounds=(1, np.array(5)),
                                     method='bounded').x
        assert_allclose(x, self.solution, atol=1e-6)

    def test_minimize_scalar_custom(self):
        # This function comes from the documentation example.
        def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.1,
                    maxiter=100, callback=None, **options):
            bestx = (bracket[1] + bracket[0]) / 2.0
            besty = fun(bestx)
            funcalls = 1
            niter = 0
            improved = True
            stop = False

            while improved and not stop and niter < maxiter:
                improved = False
                niter += 1
                for testx in [bestx - stepsize, bestx + stepsize]:
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        besty = testy
                        bestx = testx
                        improved = True
                if callback is not None:
                    callback(bestx)
                if maxfev is not None and funcalls >= maxfev:
                    stop = True
                    break

            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter,
                                           nfev=funcalls, success=(niter > 1))

        res = optimize.minimize_scalar(self.fun, bracket=(0, 4),
                                       method=custmin,
                                       options=dict(stepsize=0.05))
        assert_allclose(res.x, self.solution, atol=1e-6)

    def test_minimize_scalar_coerce_args_param(self):
        # Regression test for gh-3503
        optimize.minimize_scalar(self.fun, args=1.5)

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_disp(self, method):
        # test that all minimize_scalar methods accept a disp option.
        for disp in [0, 1, 2, 3]:
            optimize.minimize_scalar(self.fun, options={"disp": disp})

    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_result_attributes(self, method):
        kwargs = {"bounds": [-10, 10]} if method == 'bounded' else {}
        result = optimize.minimize_scalar(self.fun, method=method, **kwargs)
        assert hasattr(result, "x")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "fun")
        assert hasattr(result, "nfev")
        assert hasattr(result, "nit")

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
    def test_nan_values(self, method):
        # Check nan values result to failed exit status
        np.random.seed(1234)

        count = [0]

        def func(x):
            count[0] += 1
            if count[0] > 4:
                return np.nan
            else:
                return x**2 + 0.1 * np.sin(x)

        bracket = (-1, 0, 1)
        bounds = (-1, 1)

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.*")
            sup.filter(RuntimeWarning, ".*does not use Hessian.*")
            sup.filter(RuntimeWarning, ".*does not use gradient.*")

            count = [0]

            kwargs = {"bounds": bounds} if method == 'bounded' else {}
            sol = optimize.minimize_scalar(func, bracket=bracket,
                                           **kwargs, method=method,
                                           options=dict(maxiter=20))
            assert_equal(sol.success, False)

    def test_minimize_scalar_defaults_gh10911(self):
        # Previously, bounds were silently ignored unless `method='bounds'`
        # was chosen. See gh-10911. Check that this is no longer the case.
        def f(x):
            return x**2

        res = optimize.minimize_scalar(f)
        assert_allclose(res.x, 0, atol=1e-8)

        res = optimize.minimize_scalar(f, bounds=(1, 100),
                                       options={'xatol': 1e-10})
        assert_allclose(res.x, 1)

    def test_minimize_non_finite_bounds_gh10911(self):
        # Previously, minimize_scalar misbehaved with infinite bounds.
        # See gh-10911. Check that it now raises an error, instead.
        msg = "Optimization bounds must be finite scalars."
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(1, np.inf))
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, bounds=(np.nan, 1))

    @pytest.mark.parametrize("method", ['brent', 'golden'])
    def test_minimize_unbounded_method_with_bounds_gh10911(self, method):
        # Previously, `bounds` were silently ignored when `method='brent'` or
        # `method='golden'`. See gh-10911. Check that error is now raised.
        msg = "Use of `bounds` is incompatible with..."
        with pytest.raises(ValueError, match=msg):
            optimize.minimize_scalar(np.sin, method=method, bounds=(1, 2))

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize("method", MINIMIZE_SCALAR_METHODS)
    @pytest.mark.parametrize("tol", [1, 1e-6])
    @pytest.mark.parametrize("fshape", [(), (1,), (1, 1)])
    def test_minimize_scalar_dimensionality_gh16196(self, method, tol, fshape):
        # gh-16196 reported that the output shape of `minimize_scalar` was not
        # consistent when an objective function returned an array. Check that
        # `res.fun` and `res.x` are now consistent.
        def f(x):
            return np.array(x**4).reshape(fshape)

        a, b = -0.1, 0.2
        kwargs = (dict(bracket=(a, b)) if method != "bounded"
                  else dict(bounds=(a, b)))
        kwargs.update(dict(method=method, tol=tol))

        res = optimize.minimize_scalar(f, **kwargs)
        assert res.x.shape == res.fun.shape == f(res.x).shape == fshape

    @pytest.mark.parametrize('method', ['bounded', 'brent', 'golden'])
    def test_minimize_scalar_warnings_gh1953(self, method):
        # test that minimize_scalar methods produce warnings rather than just
        # using `print`; see gh-1953.
        def f(x):
            return (x - 1)**2

        kwargs = {}
        kwd = 'bounds' if method == 'bounded' else 'bracket'
        kwargs[kwd] = [-2, 10]

        options = {'disp': True, 'maxiter': 3}
        with pytest.warns(optimize.OptimizeWarning, match='Maximum number'):
            optimize.minimize_scalar(f, method=method, options=options,
                                     **kwargs)

        options['disp'] = False
        optimize.minimize_scalar(f, method=method, options=options, **kwargs)


class TestBracket:

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_errors_and_status_false(self):
        # Check that `bracket` raises the errors it is supposed to
        def f(x):  # gh-14858
            return x**2 if ((-1 < x) & (x < 1)) else 100.0

        message = "The algorithm terminated without finding a valid bracket."
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -1, 1)
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -1, np.inf)
        with pytest.raises(RuntimeError, match=message):
            optimize.brent(f, brack=(-1, 1))
        with pytest.raises(RuntimeError, match=message):
            optimize.golden(f, brack=(-1, 1))

        def f(x):  # gh-5899
            return -5 * x**5 + 4 * x**4 - 12 * x**3 + 11 * x**2 - 2 * x + 1

        message = "No valid bracket was found before the iteration limit..."
        with pytest.raises(RuntimeError, match=message):
            optimize.bracket(f, -0.5, 0.5, maxiter=10)

    @pytest.mark.parametrize('method', ('brent', 'golden'))
    def test_minimize_scalar_success_false(self, method):
        # Check that status information from `bracket` gets to minimize_scalar
        def f(x):  # gh-14858
            return x**2 if ((-1 < x) & (x < 1)) else 100.0

        message = "The algorithm terminated without finding a valid bracket."

        res = optimize.minimize_scalar(f, bracket=(-1, 1), method=method)
        assert not res.success
        assert message in res.message
        assert res.nfev == 3
        assert res.nit == 0
        assert res.fun == 100


def test_brent_negative_tolerance():
    assert_raises(ValueError, optimize.brent, np.cos, tol=-.01)


class TestNewtonCg:
    def test_rosenbrock(self):
        x0 = np.array([-1.2, 1.0])
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess=optimize.rosen_hess,
                                tol=1e-5,
                                method='Newton-CG')
        assert sol.success, sol.message
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)

    def test_himmelblau(self):
        x0 = np.array(himmelblau_x0)
        sol = optimize.minimize(himmelblau,
                                x0,
                                jac=himmelblau_grad,
                                hess=himmelblau_hess,
                                method='Newton-CG',
                                tol=1e-6)
        assert sol.success, sol.message
        assert_allclose(sol.x, himmelblau_xopt, rtol=1e-4)
        assert_allclose(sol.fun, himmelblau_min, atol=1e-4)

    def test_finite_difference(self):
        x0 = np.array([-1.2, 1.0])
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess='2-point',
                                tol=1e-5,
                                method='Newton-CG')
        assert sol.success, sol.message
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)

    def test_hessian_update_strategy(self):
        x0 = np.array([-1.2, 1.0])
        sol = optimize.minimize(optimize.rosen, x0,
                                jac=optimize.rosen_der,
                                hess=optimize.BFGS(),
                                tol=1e-5,
                                method='Newton-CG')
        assert sol.success, sol.message
        assert_allclose(sol.x, np.array([1, 1]), rtol=1e-4)


def test_line_for_search():
    # _line_for_search is only used in _linesearch_powell, which is also
    # tested below. Thus there are more tests of _line_for_search in the
    # test_linesearch_powell_bounded function.

    line_for_search = optimize._optimize._line_for_search
    # args are x0, alpha, lower_bound, upper_bound
    # returns lmin, lmax

    lower_bound = np.array([-5.3, -1, -1.5, -3])
    upper_bound = np.array([1.9, 1, 2.8, 3])

    # test when starting in the bounds
    x0 = np.array([0., 0, 0, 0])
    # and when starting outside of the bounds
    x1 = np.array([0., 2, -3, 0])

    all_tests = (
        (x0, np.array([1., 0, 0, 0]), -5.3, 1.9),
        (x0, np.array([0., 1, 0, 0]), -1, 1),
        (x0, np.array([0., 0, 1, 0]), -1.5, 2.8),
        (x0, np.array([0., 0, 0, 1]), -3, 3),
        (x0, np.array([1., 1, 0, 0]), -1, 1),
        (x0, np.array([1., 0, -1, 2]), -1.5, 1.5),
        (x0, np.array([2., 0, -1, 2]), -1.5, 0.95),
        (x1, np.array([1., 0, 0, 0]), -5.3, 1.9),
        (x1, np.array([0., 1, 0, 0]), -3, -1),
        (x1, np.array([0., 0, 1, 0]), 1.5, 5.8),
        (x1, np.array([0., 0, 0, 1]), -3, 3),
        (x1, np.array([1., 1, 0, 0]), -3, -1),
        (x1, np.array([1., 0, -1, 0]), -5.3, -1.5),
    )

    for x, alpha, lmin, lmax in all_tests:
        mi, ma = line_for_search(x, alpha, lower_bound, upper_bound)
        assert_allclose(mi, lmin, atol=1e-6)
        assert_allclose(ma, lmax, atol=1e-6)

    # now with infinite bounds
    lower_bound = np.array([-np.inf, -1, -np.inf, -3])
    upper_bound = np.array([np.inf, 1, 2.8, np.inf])

    all_tests = (
        (x0, np.array([1., 0, 0, 0]), -np.inf, np.inf),
        (x0, np.array([0., 1, 0, 0]), -1, 1),
        (x0, np.array([0., 0, 1, 0]), -np.inf, 2.8),
        (x0, np.array([0., 0, 0, 1]), -3, np.inf),
        (x0, np.array([1., 1, 0, 0]), -1, 1),
        (x0, np.array([1., 0, -1, 2]), -1.5, np.inf),
        (x1, np.array([1., 0, 0, 0]), -np.inf, np.inf),
        (x1, np.array([0., 1, 0, 0]), -3, -1),
        (x1, np.array([0., 0, 1, 0]), -np.inf, 5.8),
        (x1, np.array([0., 0, 0, 1]), -3, np.inf),
        (x1, np.array([1., 1, 0, 0]), -3, -1),
        (x1, np.array([1., 0, -1, 0]), -5.8, np.inf),
    )

    for x, alpha, lmin, lmax in all_tests:
        mi, ma = line_for_search(x, alpha, lower_bound, upper_bound)
        assert_allclose(mi, lmin, atol=1e-6)
        assert_allclose(ma, lmax, atol=1e-6)


def test_linesearch_powell():
    # helper function in optimize.py, not a public function.
    linesearch_powell = optimize._optimize._linesearch_powell
    # args are func, p, xi, fval, lower_bound=None, upper_bound=None, tol=1e-3
    # returns new_fval, p + direction, direction
    def func(x):
        return np.sum((x - np.array([-1.0, 2.0, 1.5, -0.4])) ** 2)
    p0 = np.array([0., 0, 0, 0])
    fval = func(p0)
    lower_bound = np.array([-np.inf] * 4)
    upper_bound = np.array([np.inf] * 4)

    all_tests = (
        (np.array([1., 0, 0, 0]), -1),
        (np.array([0., 1, 0, 0]), 2),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), 1.25),
        (np.array([0., 0, 1, 1]), .55),
        (np.array([2., 0, -1, 1]), -.65),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi,
                                            fval=fval, tol=1e-5)
        assert_allclose(f, func(l * xi), atol=1e-6)
        assert_allclose(p, l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)

        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(l * xi), atol=1e-6)
        assert_allclose(p, l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)


def test_linesearch_powell_bounded():
    # helper function in optimize.py, not a public function.
    linesearch_powell = optimize._optimize._linesearch_powell
    # args are func, p, xi, fval, lower_bound=None, upper_bound=None, tol=1e-3
    # returns new_fval, p+direction, direction
    def func(x):
        return np.sum((x - np.array([-1.0, 2.0, 1.5, -0.4])) ** 2)
    p0 = np.array([0., 0, 0, 0])
    fval = func(p0)

    # first choose bounds such that the same tests from
    # test_linesearch_powell should pass.
    lower_bound = np.array([-2.]*4)
    upper_bound = np.array([2.]*4)

    all_tests = (
        (np.array([1., 0, 0, 0]), -1),
        (np.array([0., 1, 0, 0]), 2),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), 1.25),
        (np.array([0., 0, 1, 1]), .55),
        (np.array([2., 0, -1, 1]), -.65),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(l * xi), atol=1e-6)
        assert_allclose(p, l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)

    # now choose bounds such that unbounded vs bounded gives different results
    lower_bound = np.array([-.3]*3 + [-1])
    upper_bound = np.array([.45]*3 + [.9])

    all_tests = (
        (np.array([1., 0, 0, 0]), -.3),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), .45),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), .3),
        (np.array([0., 0, 1, 1]), .45),
        (np.array([2., 0, -1, 1]), -.15),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(l * xi), atol=1e-6)
        assert_allclose(p, l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)

    # now choose as above but start outside the bounds
    p0 = np.array([-1., 0, 0, 2])
    fval = func(p0)

    all_tests = (
        (np.array([1., 0, 0, 0]), .7),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), .45),
        (np.array([0., 0, 0, 1]), -2.4),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(p0 + l * xi), atol=1e-6)
        assert_allclose(p, p0 + l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)

    # now mix in inf
    p0 = np.array([0., 0, 0, 0])
    fval = func(p0)

    # now choose bounds that mix inf
    lower_bound = np.array([-.3, -np.inf, -np.inf, -1])
    upper_bound = np.array([np.inf, .45, np.inf, .9])

    all_tests = (
        (np.array([1., 0, 0, 0]), -.3),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -.4),
        (np.array([-1., 0, 1, 0]), .3),
        (np.array([0., 0, 1, 1]), .55),
        (np.array([2., 0, -1, 1]), -.15),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(l * xi), atol=1e-6)
        assert_allclose(p, l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)

    # now choose as above but start outside the bounds
    p0 = np.array([-1., 0, 0, 2])
    fval = func(p0)

    all_tests = (
        (np.array([1., 0, 0, 0]), .7),
        (np.array([0., 1, 0, 0]), .45),
        (np.array([0., 0, 1, 0]), 1.5),
        (np.array([0., 0, 0, 1]), -2.4),
    )

    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-5,
                                            lower_bound=lower_bound,
                                            upper_bound=upper_bound,
                                            fval=fval)
        assert_allclose(f, func(p0 + l * xi), atol=1e-6)
        assert_allclose(p, p0 + l * xi, atol=1e-6)
        assert_allclose(direction, l * xi, atol=1e-6)


def test_powell_limits():
    # gh15342 - powell was going outside bounds for some function evaluations.
    bounds = optimize.Bounds([0, 0], [0.6, 20])

    def fun(x):
        a, b = x
        assert (x >= bounds.lb).all() and (x <= bounds.ub).all()
        return a ** 2 + b ** 2

    optimize.minimize(fun, x0=[0.6, 20], method='Powell', bounds=bounds)

    # Another test from the original report - gh-13411
    bounds = optimize.Bounds(lb=[0,], ub=[1,], keep_feasible=[True,])

    def func(x):
        assert x >= 0 and x <= 1
        return np.exp(x)

    optimize.minimize(fun=func, x0=[0.5], method='powell', bounds=bounds)


class TestRosen:

    def test_hess(self):
        # Compare rosen_hess(x) times p with rosen_hess_prod(x,p). See gh-1775.
        x = np.array([3, 4, 5])
        p = np.array([2, 2, 2])
        hp = optimize.rosen_hess_prod(x, p)
        dothp = np.dot(optimize.rosen_hess(x), p)
        assert_equal(hp, dothp)


def himmelblau(p):
    """
    R^2 -> R^1 test function for optimization. The function has four local
    minima where himmelblau(xopt) == 0.
    """
    x, y = p
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b


def himmelblau_grad(p):
    x, y = p
    return np.array([4*x**3 + 4*x*y - 42*x + 2*y**2 - 14,
                     2*x**2 + 4*x*y + 4*y**3 - 26*y - 22])


def himmelblau_hess(p):
    x, y = p
    return np.array([[12*x**2 + 4*y - 42, 4*x + 4*y],
                     [4*x + 4*y, 4*x + 12*y**2 - 26]])


himmelblau_x0 = [-0.27, -0.9]
himmelblau_xopt = [3, 2]
himmelblau_min = 0.0


def test_minimize_multiple_constraints():
    # Regression test for gh-4240.
    def func(x):
        return np.array([25 - 0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])

    def func1(x):
        return np.array([x[1]])

    def func2(x):
        return np.array([x[2]])

    cons = ({'type': 'ineq', 'fun': func},
            {'type': 'ineq', 'fun': func1},
            {'type': 'ineq', 'fun': func2})

    def f(x):
        return -1 * (x[0] + x[1] + x[2])

    res = optimize.minimize(f, [0, 0, 0], method='SLSQP', constraints=cons)
    assert_allclose(res.x, [125, 0, 0], atol=1e-10)


class TestOptimizeResultAttributes:
    # Test that all minimizers return an OptimizeResult containing
    # all the OptimizeResult attributes
    def setup_method(self):
        self.x0 = [5, 5]
        self.func = optimize.rosen
        self.jac = optimize.rosen_der
        self.hess = optimize.rosen_hess
        self.hessp = optimize.rosen_hess_prod
        self.bounds = [(0., 10.), (0., 10.)]

    def test_attributes_present(self):
        attributes = ['nit', 'nfev', 'x', 'success', 'status', 'fun',
                      'message']
        skip = {'cobyla': ['nit']}
        for method in MINIMIZE_METHODS:
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning,
                           ("Method .+ does not use (gradient|Hessian.*)"
                            " information"))
                res = optimize.minimize(self.func, self.x0, method=method,
                                        jac=self.jac, hess=self.hess,
                                        hessp=self.hessp)
            for attribute in attributes:
                if method in skip and attribute in skip[method]:
                    continue

                assert hasattr(res, attribute)
                assert attribute in dir(res)

            # gh13001, OptimizeResult.message should be a str
            assert isinstance(res.message, str)


def f1(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)


def f2(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))


def f3(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))


def brute_func(z, *params):
    return f1(z, *params) + f2(z, *params) + f3(z, *params)


class TestBrute:
    # Test the "brute force" method
    def setup_method(self):
        self.params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
        self.rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
        self.solution = np.array([-1.05665192, 1.80834843])

    def brute_func(self, z, *params):
        # an instance method optimizing
        return brute_func(z, *params)

    def test_brute(self):
        # test fmin
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True, finish=optimize.fmin)
        assert_allclose(resbrute[0], self.solution, atol=1e-3)
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params),
                        atol=1e-3)

        # test minimize
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True,
                                  finish=optimize.minimize)
        assert_allclose(resbrute[0], self.solution, atol=1e-3)
        assert_allclose(resbrute[1], brute_func(self.solution, *self.params),
                        atol=1e-3)

        # test that brute can optimize an instance method (the other tests use
        # a non-class based function
        resbrute = optimize.brute(self.brute_func, self.rranges,
                                  args=self.params, full_output=True,
                                  finish=optimize.minimize)
        assert_allclose(resbrute[0], self.solution, atol=1e-3)

    def test_1D(self):
        # test that for a 1-D problem the test function is passed an array,
        # not a scalar.
        def f(x):
            assert len(x.shape) == 1
            assert x.shape[0] == 1
            return x ** 2

        optimize.brute(f, [(-1, 1)], Ns=3, finish=None)

    def test_workers(self):
        # check that parallel evaluation works
        resbrute = optimize.brute(brute_func, self.rranges, args=self.params,
                                  full_output=True, finish=None)

        resbrute1 = optimize.brute(brute_func, self.rranges, args=self.params,
                                   full_output=True, finish=None, workers=2)

        assert_allclose(resbrute1[-1], resbrute[-1])
        assert_allclose(resbrute1[0], resbrute[0])

    def test_runtime_warning(self):
        rng = np.random.default_rng(1234)

        def func(z, *params):
            return rng.random(1) * 1000  # never converged problem

        msg = "final optimization did not succeed.*|Maximum number of function eval.*"
        with pytest.warns(RuntimeWarning, match=msg):
            optimize.brute(func, self.rranges, args=self.params, disp=True)

    def test_coerce_args_param(self):
        # optimize.brute should coerce non-iterable args to a tuple.
        def f(x, *args):
            return x ** args[0]

        resbrute = optimize.brute(f, (slice(-4, 4, .25),), args=2)
        assert_allclose(resbrute, 0)


def test_cobyla_threadsafe():

    # Verify that cobyla is threadsafe. Will segfault if it is not.

    import concurrent.futures
    import time

    def objective1(x):
        time.sleep(0.1)
        return x[0]**2

    def objective2(x):
        time.sleep(0.1)
        return (x[0]-1)**2

    min_method = "COBYLA"

    def minimizer1():
        return optimize.minimize(objective1,
                                      [0.0],
                                      method=min_method)

    def minimizer2():
        return optimize.minimize(objective2,
                                      [0.0],
                                      method=min_method)

    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = []
        tasks.append(pool.submit(minimizer1))
        tasks.append(pool.submit(minimizer2))
        for t in tasks:
            t.result()


class TestIterationLimits:
    # Tests that optimisation does not give up before trying requested
    # number of iterations or evaluations. And that it does not succeed
    # by exceeding the limits.
    def setup_method(self):
        self.funcalls = 0

    def slow_func(self, v):
        self.funcalls += 1
        r, t = np.sqrt(v[0]**2+v[1]**2), np.arctan2(v[0], v[1])
        return np.sin(r*20 + t)+r*0.5

    def test_neldermead_limit(self):
        self.check_limits("Nelder-Mead", 200)

    def test_powell_limit(self):
        self.check_limits("powell", 1000)

    def check_limits(self, method, default_iters):
        for start_v in [[0.1, 0.1], [1, 1], [2, 2]]:
            for mfev in [50, 500, 5000]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxfev": mfev})
                assert self.funcalls == res["nfev"]
                if res["success"]:
                    assert res["nfev"] < mfev
                else:
                    assert res["nfev"] >= mfev
            for mit in [50, 500, 5000]:
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit})
                if res["success"]:
                    assert res["nit"] <= mit
                else:
                    assert res["nit"] >= mit
            for mfev, mit in [[50, 50], [5000, 5000], [5000, np.inf]]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                assert self.funcalls == res["nfev"]
                if res["success"]:
                    assert res["nfev"] < mfev and res["nit"] <= mit
                else:
                    assert res["nfev"] >= mfev or res["nit"] >= mit
            for mfev, mit in [[np.inf, None], [None, np.inf]]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                assert self.funcalls == res["nfev"]
                if res["success"]:
                    if mfev is None:
                        assert res["nfev"] < default_iters*2
                    else:
                        assert res["nit"] <= default_iters*2
                else:
                    assert (res["nfev"] >= default_iters*2
                            or res["nit"] >= default_iters*2)


def test_result_x_shape_when_len_x_is_one():
    def fun(x):
        return x * x

    def jac(x):
        return 2. * x

    def hess(x):
        return np.array([[2.]])

    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC',
               'COBYLA', 'SLSQP']
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method)
        assert res.x.shape == (1,)

    # use jac + hess
    methods = ['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
               'trust-krylov', 'Newton-CG']
    for method in methods:
        res = optimize.minimize(fun, np.array([0.1]), method=method, jac=jac,
                                hess=hess)
        assert res.x.shape == (1,)


class FunctionWithGradient:
    def __init__(self):
        self.number_of_calls = 0

    def __call__(self, x):
        self.number_of_calls += 1
        return np.sum(x**2), 2 * x


@pytest.fixture
def function_with_gradient():
    return FunctionWithGradient()


def test_memoize_jac_function_before_gradient(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function(x0), 5.0)
    assert function_with_gradient.number_of_calls == 1

    assert_allclose(memoized_function.derivative(x0), 2 * x0)
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if gradient is requested after function value"

    assert_allclose(
        memoized_function(2 * x0), 20.0,
        err_msg="different input triggers new computation")
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_memoize_jac_gradient_before_function(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function.derivative(x0), 2 * x0)
    assert function_with_gradient.number_of_calls == 1

    assert_allclose(memoized_function(x0), 5.0)
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if function value is requested after gradient"

    assert_allclose(
        memoized_function.derivative(2 * x0), 4 * x0,
        err_msg="different input triggers new computation")
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_memoize_jac_with_bfgs(function_with_gradient):
    """ Tests that using MemoizedJac in combination with ScalarFunction
        and BFGS does not lead to repeated function evaluations.
        Tests changes made in response to GH11868.
    """
    memoized_function = MemoizeJac(function_with_gradient)
    jac = memoized_function.derivative
    hess = optimize.BFGS()

    x0 = np.array([1.0, 0.5])
    scalar_function = ScalarFunction(
        memoized_function, x0, (), jac, hess, None, None)
    assert function_with_gradient.number_of_calls == 1

    scalar_function.fun(x0 + 0.1)
    assert function_with_gradient.number_of_calls == 2

    scalar_function.fun(x0 + 0.2)
    assert function_with_gradient.number_of_calls == 3


def test_gh12696():
    # Test that optimize doesn't throw warning gh-12696
    with assert_no_warnings():
        optimize.fminbound(
            lambda x: np.array([x**2]), -np.pi, np.pi, disp=False)


# --- Test minimize with equal upper and lower bounds --- #

def setup_test_equal_bounds():

    np.random.seed(0)
    x0 = np.random.rand(4)
    lb = np.array([0, 2, -1, -1.0])
    ub = np.array([3, 2, 2, -1.0])
    i_eb = (lb == ub)

    def check_x(x, check_size=True, check_values=True):
        if check_size:
            assert x.size == 4
        if check_values:
            assert_allclose(x[i_eb], lb[i_eb])

    def func(x):
        check_x(x)
        return optimize.rosen(x)

    def grad(x):
        check_x(x)
        return optimize.rosen_der(x)

    def callback(x, *args):
        check_x(x)

    def constraint1(x):
        check_x(x, check_values=False)
        return x[0:1] - 1

    def jacobian1(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[0] = 1
        return dc

    def constraint2(x):
        check_x(x, check_values=False)
        return x[2:3] - 0.5

    def jacobian2(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[2] = 1
        return dc

    c1a = NonlinearConstraint(constraint1, -np.inf, 0)
    c1b = NonlinearConstraint(constraint1, -np.inf, 0, jacobian1)
    c2a = NonlinearConstraint(constraint2, -np.inf, 0)
    c2b = NonlinearConstraint(constraint2, -np.inf, 0, jacobian2)

    # test using the three methods that accept bounds, use derivatives, and
    # have some trouble when bounds fix variables
    methods = ('L-BFGS-B', 'SLSQP', 'TNC')

    # test w/out gradient, w/ gradient, and w/ combined objective/gradient
    kwds = ({"fun": func, "jac": False},
            {"fun": func, "jac": grad},
            {"fun": (lambda x: (func(x), grad(x))),
             "jac": True})

    # test with both old- and new-style bounds
    bound_types = (lambda lb, ub: list(zip(lb, ub)),
                   Bounds)

    # Test for many combinations of constraints w/ and w/out jacobian
    # Pairs in format: (test constraints, reference constraints)
    # (always use analytical jacobian in reference)
    constraints = ((None, None), ([], []),
                   (c1a, c1b), (c2b, c2b),
                   ([c1b], [c1b]), ([c2a], [c2b]),
                   ([c1a, c2a], [c1b, c2b]),
                   ([c1a, c2b], [c1b, c2b]),
                   ([c1b, c2b], [c1b, c2b]))

    # test with and without callback function
    callbacks = (None, callback)

    data = {"methods": methods, "kwds": kwds, "bound_types": bound_types,
            "constraints": constraints, "callbacks": callbacks,
            "lb": lb, "ub": ub, "x0": x0, "i_eb": i_eb}

    return data


eb_data = setup_test_equal_bounds()


# This test is about handling fixed variables, not the accuracy of the solvers
@pytest.mark.xfail_on_32bit("Failures due to floating point issues, not logic")
@pytest.mark.parametrize('method', eb_data["methods"])
@pytest.mark.parametrize('kwds', eb_data["kwds"])
@pytest.mark.parametrize('bound_type', eb_data["bound_types"])
@pytest.mark.parametrize('constraints', eb_data["constraints"])
@pytest.mark.parametrize('callback', eb_data["callbacks"])
def test_equal_bounds(method, kwds, bound_type, constraints, callback):
    """
    Tests that minimizers still work if (bounds.lb == bounds.ub).any()
    gh12502 - Divide by zero in Jacobian numerical differentiation when
    equality bounds constraints are used
    """
    # GH-15051; slightly more skips than necessary; hopefully fixed by GH-14882
    if (platform.machine() == 'aarch64' and method == "TNC"
            and kwds["jac"] is False and callback is not None):
        pytest.skip('Tolerance violation on aarch')

    lb, ub = eb_data["lb"], eb_data["ub"]
    x0, i_eb = eb_data["x0"], eb_data["i_eb"]

    test_constraints, reference_constraints = constraints
    if test_constraints and not method == 'SLSQP':
        pytest.skip('Only SLSQP supports nonlinear constraints')
    # reference constraints always have analytical jacobian
    # if test constraints are not the same, we'll need finite differences
    fd_needed = (test_constraints != reference_constraints)

    bounds = bound_type(lb, ub)  # old- or new-style

    kwds.update({"x0": x0, "method": method, "bounds": bounds,
                 "constraints": test_constraints, "callback": callback})
    res = optimize.minimize(**kwds)

    expected = optimize.minimize(optimize.rosen, x0, method=method,
                                 jac=optimize.rosen_der, bounds=bounds,
                                 constraints=reference_constraints)

    # compare the output of a solution with FD vs that of an analytic grad
    assert res.success
    assert_allclose(res.fun, expected.fun, rtol=1.5e-6)
    assert_allclose(res.x, expected.x, rtol=5e-4)

    if fd_needed or kwds['jac'] is False:
        expected.jac[i_eb] = np.nan
    assert res.jac.shape[0] == 4
    assert_allclose(res.jac[i_eb], expected.jac[i_eb], rtol=1e-6)

    if not (kwds['jac'] or test_constraints or isinstance(bounds, Bounds)):
        # compare the output to an equivalent FD minimization that doesn't
        # need factorization
        def fun(x):
            new_x = np.array([np.nan, 2, np.nan, -1])
            new_x[[0, 2]] = x
            return optimize.rosen(new_x)

        fd_res = optimize.minimize(fun,
                                   x0[[0, 2]],
                                   method=method,
                                   bounds=bounds[::2])
        assert_allclose(res.fun, fd_res.fun)
        # TODO this test should really be equivalent to factorized version
        # above, down to res.nfev. However, testing found that when TNC is
        # called with or without a callback the output is different. The two
        # should be the same! This indicates that the TNC callback may be
        # mutating something when it shouldn't.
        assert_allclose(res.x[[0, 2]], fd_res.x, rtol=2e-6)


@pytest.mark.parametrize('method', eb_data["methods"])
def test_all_bounds_equal(method):
    # this only tests methods that have parameters factored out when lb==ub
    # it does not test other methods that work with bounds
    def f(x, p1=1):
        return np.linalg.norm(x) + p1

    bounds = [(1, 1), (2, 2)]
    x0 = (1.0, 3.0)
    res = optimize.minimize(f, x0, bounds=bounds, method=method)
    assert res.success
    assert_allclose(res.fun, f([1.0, 2.0]))
    assert res.nfev == 1
    assert res.message == 'All independent variables were fixed by bounds.'

    args = (2,)
    res = optimize.minimize(f, x0, bounds=bounds, method=method, args=args)
    assert res.success
    assert_allclose(res.fun, f([1.0, 2.0], 2))

    if method.upper() == 'SLSQP':
        def con(x):
            return np.sum(x)
        nlc = NonlinearConstraint(con, -np.inf, 0.0)
        res = optimize.minimize(
            f, x0, bounds=bounds, method=method, constraints=[nlc]
        )
        assert res.success is False
        assert_allclose(res.fun, f([1.0, 2.0]))
        assert res.nfev == 1
        message = "All independent variables were fixed by bounds, but"
        assert res.message.startswith(message)

        nlc = NonlinearConstraint(con, -np.inf, 4)
        res = optimize.minimize(
            f, x0, bounds=bounds, method=method, constraints=[nlc]
        )
        assert res.success is True
        assert_allclose(res.fun, f([1.0, 2.0]))
        assert res.nfev == 1
        message = "All independent variables were fixed by bounds at values"
        assert res.message.startswith(message)


def test_eb_constraints():
    # make sure constraint functions aren't overwritten when equal bounds
    # are employed, and a parameter is factored out. GH14859
    def f(x):
        return x[0]**3 + x[1]**2 + x[2]*x[3]

    def cfun(x):
        return x[0] + x[1] + x[2] + x[3] - 40

    constraints = [{'type': 'ineq', 'fun': cfun}]

    bounds = [(0, 20)] * 4
    bounds[1] = (5, 5)
    optimize.minimize(
        f,
        x0=[1, 2, 3, 4],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )
    assert constraints[0]['fun'] == cfun


def test_show_options():
    solver_methods = {
        'minimize': MINIMIZE_METHODS,
        'minimize_scalar': MINIMIZE_SCALAR_METHODS,
        'root': ROOT_METHODS,
        'root_scalar': ROOT_SCALAR_METHODS,
        'linprog': LINPROG_METHODS,
        'quadratic_assignment': QUADRATIC_ASSIGNMENT_METHODS,
    }
    for solver, methods in solver_methods.items():
        for method in methods:
            # testing that `show_options` works without error
            show_options(solver, method)

    unknown_solver_method = {
        'minimize': "ekki",  # unknown method
        'maximize': "cg",  # unknown solver
        'maximize_scalar': "ekki",  # unknown solver and method
    }
    for solver, method in unknown_solver_method.items():
        # testing that `show_options` raises ValueError
        assert_raises(ValueError, show_options, solver, method)


def test_bounds_with_list():
    # gh13501. Bounds created with lists weren't working for Powell.
    bounds = optimize.Bounds(lb=[5., 5.], ub=[10., 10.])
    optimize.minimize(
        optimize.rosen, x0=np.array([9, 9]), method='Powell', bounds=bounds
    )


def test_x_overwritten_user_function():
    # if the user overwrites the x-array in the user function it's likely
    # that the minimizer stops working properly.
    # gh13740
    def fquad(x):
        a = np.arange(np.size(x))
        x -= a
        x *= x
        return np.sum(x)

    def fquad_jac(x):
        a = np.arange(np.size(x))
        x *= 2
        x -= 2 * a
        return x

    def fquad_hess(x):
        return np.eye(np.size(x)) * 2.0

    meth_jac = [
        'newton-cg', 'dogleg', 'trust-ncg', 'trust-exact',
        'trust-krylov', 'trust-constr'
    ]
    meth_hess = [
        'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr'
    ]

    x0 = np.ones(5) * 1.5

    for meth in MINIMIZE_METHODS:
        jac = None
        hess = None
        if meth in meth_jac:
            jac = fquad_jac
        if meth in meth_hess:
            hess = fquad_hess
        res = optimize.minimize(fquad, x0, method=meth, jac=jac, hess=hess)
        assert_allclose(res.x, np.arange(np.size(x0)), atol=2e-4)


class TestGlobalOptimization:

    def test_optimize_result_attributes(self):
        def func(x):
            return x ** 2

        # Note that `brute` solver does not return `OptimizeResult`
        results = [optimize.basinhopping(func, x0=1),
                   optimize.differential_evolution(func, [(-4, 4)]),
                   optimize.shgo(func, [(-4, 4)]),
                   optimize.dual_annealing(func, [(-4, 4)]),
                   optimize.direct(func, [(-4, 4)]),
                   ]

        for result in results:
            assert isinstance(result, optimize.OptimizeResult)
            assert hasattr(result, "x")
            assert hasattr(result, "success")
            assert hasattr(result, "message")
            assert hasattr(result, "fun")
            assert hasattr(result, "nfev")
            assert hasattr(result, "nit")


def test_approx_fprime():
    # check that approx_fprime (serviced by approx_derivative) works for
    # jac and hess
    g = optimize.approx_fprime(himmelblau_x0, himmelblau)
    assert_allclose(g, himmelblau_grad(himmelblau_x0), rtol=5e-6)

    h = optimize.approx_fprime(himmelblau_x0, himmelblau_grad)
    assert_allclose(h, himmelblau_hess(himmelblau_x0), rtol=5e-6)


def test_gh12594():
    # gh-12594 reported an error in `_linesearch_powell` and
    # `_line_for_search` when `Bounds` was passed lists instead of arrays.
    # Check that results are the same whether the inputs are lists or arrays.

    def f(x):
        return x[0]**2 + (x[1] - 1)**2

    bounds = Bounds(lb=[-10, -10], ub=[10, 10])
    res = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)
    bounds = Bounds(lb=np.array([-10, -10]), ub=np.array([10, 10]))
    ref = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)

    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x)


@pytest.mark.parametrize('method', ['Newton-CG', 'trust-constr'])
@pytest.mark.parametrize('sparse_type', [coo_matrix, csc_matrix, csr_matrix,
                                         coo_array, csr_array, csc_array])
def test_sparse_hessian(method, sparse_type):
    # gh-8792 reported an error for minimization with `newton_cg` when `hess`
    # returns a sparse matrix. Check that results are the same whether `hess`
    # returns a dense or sparse matrix for optimization methods that accept
    # sparse Hessian matrices.

    def sparse_rosen_hess(x):
        return sparse_type(rosen_hess(x))

    x0 = [2., 2.]

    res_sparse = optimize.minimize(rosen, x0, method=method,
                                   jac=rosen_der, hess=sparse_rosen_hess)
    res_dense = optimize.minimize(rosen, x0, method=method,
                                  jac=rosen_der, hess=rosen_hess)

    assert_allclose(res_dense.fun, res_sparse.fun)
    assert_allclose(res_dense.x, res_sparse.x)
    assert res_dense.nfev == res_sparse.nfev
    assert res_dense.njev == res_sparse.njev
    assert res_dense.nhev == res_sparse.nhev
