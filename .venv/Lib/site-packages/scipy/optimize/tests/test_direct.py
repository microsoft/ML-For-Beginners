"""
Unit test for DIRECT optimization algorithm.
"""
from numpy.testing import (assert_allclose,
                           assert_array_less)
import pytest
import numpy as np
from scipy.optimize import direct, Bounds


class TestDIRECT:

    def setup_method(self):
        self.fun_calls = 0
        self.bounds_sphere = 4*[(-2, 3)]
        self.optimum_sphere_pos = np.zeros((4, ))
        self.optimum_sphere = 0.0
        self.bounds_stylinski_tang = Bounds([-4., -4.], [4., 4.])
        self.maxiter = 1000

    # test functions
    def sphere(self, x):
        self.fun_calls += 1
        return np.square(x).sum()

    def inv(self, x):
        if np.sum(x) == 0:
            raise ZeroDivisionError()
        return 1/np.sum(x)

    def nan_fun(self, x):
        return np.nan

    def inf_fun(self, x):
        return np.inf

    def styblinski_tang(self, pos):
        x, y = pos
        return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_direct(self, locally_biased):
        res = direct(self.sphere, self.bounds_sphere,
                     locally_biased=locally_biased)

        # test accuracy
        assert_allclose(res.x, self.optimum_sphere_pos,
                        rtol=1e-3, atol=1e-3)
        assert_allclose(res.fun, self.optimum_sphere, atol=1e-5, rtol=1e-5)

        # test that result lies within bounds
        _bounds = np.asarray(self.bounds_sphere)
        assert_array_less(_bounds[:, 0], res.x)
        assert_array_less(res.x, _bounds[:, 1])

        # test number of function evaluations. Original DIRECT overshoots by
        # up to 500 evaluations in last iteration
        assert res.nfev <= 1000 * (len(self.bounds_sphere) + 1)
        # test that number of function evaluations is correct
        assert res.nfev == self.fun_calls

        # test that number of iterations is below supplied maximum
        assert res.nit <= self.maxiter

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_direct_callback(self, locally_biased):
        # test that callback does not change the result
        res = direct(self.sphere, self.bounds_sphere,
                     locally_biased=locally_biased)

        def callback(x):
            x = 2*x
            dummy = np.square(x)
            print("DIRECT minimization algorithm callback test")
            return dummy

        res_callback = direct(self.sphere, self.bounds_sphere,
                              locally_biased=locally_biased,
                              callback=callback)

        assert_allclose(res.x, res_callback.x)

        assert res.nit == res_callback.nit
        assert res.nfev == res_callback.nfev
        assert res.status == res_callback.status
        assert res.success == res_callback.success
        assert res.fun == res_callback.fun
        assert_allclose(res.x, res_callback.x)
        assert res.message == res_callback.message

        # test accuracy
        assert_allclose(res_callback.x, self.optimum_sphere_pos,
                        rtol=1e-3, atol=1e-3)
        assert_allclose(res_callback.fun, self.optimum_sphere,
                        atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_exception(self, locally_biased):
        bounds = 4*[(-10, 10)]
        with pytest.raises(ZeroDivisionError):
            direct(self.inv, bounds=bounds,
                   locally_biased=locally_biased)

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_nan(self, locally_biased):
        bounds = 4*[(-10, 10)]
        direct(self.nan_fun, bounds=bounds,
               locally_biased=locally_biased)

    @pytest.mark.parametrize("len_tol", [1e-3, 1e-4])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_len_tol(self, len_tol, locally_biased):
        bounds = 4*[(-10., 10.)]
        res = direct(self.sphere, bounds=bounds, len_tol=len_tol,
                     vol_tol=1e-30, locally_biased=locally_biased)
        assert res.status == 5
        assert res.success
        assert_allclose(res.x, np.zeros((4, )))
        message = ("The side length measure of the hyperrectangle containing "
                   "the lowest function value found is below "
                   f"len_tol={len_tol}")
        assert res.message == message

    @pytest.mark.parametrize("vol_tol", [1e-6, 1e-8])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_vol_tol(self, vol_tol, locally_biased):
        bounds = 4*[(-10., 10.)]
        res = direct(self.sphere, bounds=bounds, vol_tol=vol_tol,
                     len_tol=0., locally_biased=locally_biased)
        assert res.status == 4
        assert res.success
        assert_allclose(res.x, np.zeros((4, )))
        message = ("The volume of the hyperrectangle containing the lowest "
                   f"function value found is below vol_tol={vol_tol}")
        assert res.message == message

    @pytest.mark.parametrize("f_min_rtol", [1e-3, 1e-5, 1e-7])
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_f_min(self, f_min_rtol, locally_biased):
        # test that desired function value is reached within
        # relative tolerance of f_min_rtol
        f_min = 1.
        bounds = 4*[(-2., 10.)]
        res = direct(self.sphere, bounds=bounds, f_min=f_min,
                     f_min_rtol=f_min_rtol,
                     locally_biased=locally_biased)
        assert res.status == 3
        assert res.success
        assert res.fun < f_min * (1. + f_min_rtol)
        message = ("The best function value found is within a relative "
                   f"error={f_min_rtol} of the (known) global optimum f_min")
        assert res.message == message

    def circle_with_args(self, x, a, b):
        return np.square(x[0] - a) + np.square(x[1] - b).sum()

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_f_circle_with_args(self, locally_biased):
        bounds = 2*[(-2.0, 2.0)]

        res = direct(self.circle_with_args, bounds, args=(1, 1), maxfun=1250,
                     locally_biased=locally_biased)
        assert_allclose(res.x, np.array([1., 1.]), rtol=1e-5)

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_failure_maxfun(self, locally_biased):
        # test that if optimization runs for the maximal number of
        # evaluations, success = False is returned

        maxfun = 100
        result = direct(self.styblinski_tang, self.bounds_stylinski_tang,
                        maxfun=maxfun, locally_biased=locally_biased)
        assert result.success is False
        assert result.status == 1
        assert result.nfev >= maxfun
        message = ("Number of function evaluations done is "
                   f"larger than maxfun={maxfun}")
        assert result.message == message

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_failure_maxiter(self, locally_biased):
        # test that if optimization runs for the maximal number of
        # iterations, success = False is returned

        maxiter = 10
        result = direct(self.styblinski_tang, self.bounds_stylinski_tang,
                        maxiter=maxiter, locally_biased=locally_biased)
        assert result.success is False
        assert result.status == 2
        assert result.nit >= maxiter
        message = f"Number of iterations is larger than maxiter={maxiter}"
        assert result.message == message

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_bounds_variants(self, locally_biased):
        # test that new and old bounds yield same result

        lb = [-6., 1., -5.]
        ub = [-1., 3., 5.]
        x_opt = np.array([-1., 1., 0.])
        bounds_old = list(zip(lb, ub))
        bounds_new = Bounds(lb, ub)

        res_old_bounds = direct(self.sphere, bounds_old,
                                locally_biased=locally_biased)
        res_new_bounds = direct(self.sphere, bounds_new,
                                locally_biased=locally_biased)

        assert res_new_bounds.nfev == res_old_bounds.nfev
        assert res_new_bounds.message == res_old_bounds.message
        assert res_new_bounds.success == res_old_bounds.success
        assert res_new_bounds.nit == res_old_bounds.nit
        assert_allclose(res_new_bounds.x, res_old_bounds.x)
        assert_allclose(res_new_bounds.x, x_opt, rtol=1e-2)

    @pytest.mark.parametrize("locally_biased", [True, False])
    @pytest.mark.parametrize("eps", [1e-5, 1e-4, 1e-3])
    def test_epsilon(self, eps, locally_biased):
        result = direct(self.styblinski_tang, self.bounds_stylinski_tang,
                        eps=eps, vol_tol=1e-6,
                        locally_biased=locally_biased)
        assert result.status == 4
        assert result.success

    @pytest.mark.xslow
    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_no_segmentation_fault(self, locally_biased):
        # test that an excessive number of function evaluations
        # does not result in segmentation fault
        bounds = [(-5., 20.)] * 100
        result = direct(self.sphere, bounds, maxfun=10000000,
                        maxiter=1000000, locally_biased=locally_biased)
        assert result is not None

    @pytest.mark.parametrize("locally_biased", [True, False])
    def test_inf_fun(self, locally_biased):
        # test that an objective value of infinity does not crash DIRECT
        bounds = [(-5., 5.)] * 2
        result = direct(self.inf_fun, bounds,
                        locally_biased=locally_biased)
        assert result is not None

    @pytest.mark.parametrize("len_tol", [-1, 2])
    def test_len_tol_validation(self, len_tol):
        error_msg = "len_tol must be between 0 and 1."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   len_tol=len_tol)

    @pytest.mark.parametrize("vol_tol", [-1, 2])
    def test_vol_tol_validation(self, vol_tol):
        error_msg = "vol_tol must be between 0 and 1."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   vol_tol=vol_tol)

    @pytest.mark.parametrize("f_min_rtol", [-1, 2])
    def test_fmin_rtol_validation(self, f_min_rtol):
        error_msg = "f_min_rtol must be between 0 and 1."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   f_min_rtol=f_min_rtol, f_min=0.)

    @pytest.mark.parametrize("maxfun", [1.5, "string", (1, 2)])
    def test_maxfun_wrong_type(self, maxfun):
        error_msg = "maxfun must be of type int."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxfun=maxfun)

    @pytest.mark.parametrize("maxiter", [1.5, "string", (1, 2)])
    def test_maxiter_wrong_type(self, maxiter):
        error_msg = "maxiter must be of type int."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxiter=maxiter)

    def test_negative_maxiter(self):
        error_msg = "maxiter must be > 0."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxiter=-1)

    def test_negative_maxfun(self):
        error_msg = "maxfun must be > 0."
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   maxfun=-1)

    @pytest.mark.parametrize("bounds", ["bounds", 2., 0])
    def test_invalid_bounds_type(self, bounds):
        error_msg = ("bounds must be a sequence or "
                     "instance of Bounds class")
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, bounds)

    @pytest.mark.parametrize("bounds",
                             [Bounds([-1., -1], [-2, 1]),
                              Bounds([-np.nan, -1], [-2, np.nan]),
                              ]
                             )
    def test_incorrect_bounds(self, bounds):
        error_msg = 'Bounds are not consistent min < max'
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, bounds)

    def test_inf_bounds(self):
        error_msg = 'Bounds must not be inf.'
        bounds = Bounds([-np.inf, -1], [-2, np.inf])
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, bounds)

    @pytest.mark.parametrize("locally_biased", ["bias", [0, 0], 2.])
    def test_locally_biased_validation(self, locally_biased):
        error_msg = 'locally_biased must be True or False.'
        with pytest.raises(ValueError, match=error_msg):
            direct(self.styblinski_tang, self.bounds_stylinski_tang,
                   locally_biased=locally_biased)
