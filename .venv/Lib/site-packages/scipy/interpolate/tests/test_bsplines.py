import os
import operator
import itertools

import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest

from scipy.interpolate import (
        BSpline, BPoly, PPoly, make_interp_spline, make_lsq_spline, _bspl,
        splev, splrep, splprep, splder, splantider, sproot, splint, insert,
        CubicSpline, NdBSpline, make_smoothing_spline, RegularGridInterpolator,
)
import scipy.linalg as sl

from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
                                        _woodbury_algorithm, _periodic_knots,
                                         _make_interp_per_full_matr)
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError


class TestBSpline:

    def test_ctor(self):
        # knots should be an ordered 1-D array of finite real numbers
        assert_raises((TypeError, ValueError), BSpline,
                **dict(t=[1, 1.j], c=[1.], k=0))
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, BSpline, **dict(t=[1, np.nan], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, np.inf], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[1, -1], c=[1.], k=0))
        assert_raises(ValueError, BSpline, **dict(t=[[1], [1]], c=[1.], k=0))

        # for n+k+1 knots and degree k need at least n coefficients
        assert_raises(ValueError, BSpline, **dict(t=[0, 1, 2], c=[1], k=0))
        assert_raises(ValueError, BSpline,
                **dict(t=[0, 1, 2, 3, 4], c=[1., 1.], k=2))

        # non-integer orders
        assert_raises(TypeError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k="cubic"))
        assert_raises(TypeError, BSpline,
                **dict(t=[0., 0., 1., 2., 3., 4.], c=[1., 1., 1.], k=2.5))

        # basic interval cannot have measure zero (here: [1..1])
        assert_raises(ValueError, BSpline,
                **dict(t=[0., 0, 1, 1, 2, 3], c=[1., 1, 1], k=2))

        # tck vs self.tck
        n, k = 11, 3
        t = np.arange(n+k+1)
        c = np.random.random(n)
        b = BSpline(t, c, k)

        assert_allclose(t, b.t)
        assert_allclose(c, b.c)
        assert_equal(k, b.k)

    def test_tck(self):
        b = _make_random_spline()
        tck = b.tck

        assert_allclose(b.t, tck[0], atol=1e-15, rtol=1e-15)
        assert_allclose(b.c, tck[1], atol=1e-15, rtol=1e-15)
        assert_equal(b.k, tck[2])

        # b.tck is read-only
        with pytest.raises(AttributeError):
            b.tck = 'foo'

    def test_degree_0(self):
        xx = np.linspace(0, 1, 10)

        b = BSpline(t=[0, 1], c=[3.], k=0)
        assert_allclose(b(xx), 3)

        b = BSpline(t=[0, 0.35, 1], c=[3, 4], k=0)
        assert_allclose(b(xx), np.where(xx < 0.35, 3, 4))

    def test_degree_1(self):
        t = [0, 1, 2, 3, 4]
        c = [1, 2, 3]
        k = 1
        b = BSpline(t, c, k)

        x = np.linspace(1, 3, 50)
        assert_allclose(c[0]*B_012(x) + c[1]*B_012(x-1) + c[2]*B_012(x-2),
                        b(x), atol=1e-14)
        assert_allclose(splev(x, (t, c, k)), b(x), atol=1e-14)

    def test_bernstein(self):
        # a special knot vector: Bernstein polynomials
        k = 3
        t = np.asarray([0]*(k+1) + [1]*(k+1))
        c = np.asarray([1., 2., 3., 4.])
        bp = BPoly(c.reshape(-1, 1), [0, 1])
        bspl = BSpline(t, c, k)

        xx = np.linspace(-1., 2., 10)
        assert_allclose(bp(xx, extrapolate=True),
                        bspl(xx, extrapolate=True), atol=1e-14)
        assert_allclose(splev(xx, (t, c, k)),
                        bspl(xx), atol=1e-14)

    def test_rndm_naive_eval(self):
        # test random coefficient spline *on the base interval*,
        # t[k] <= x < t[-k-1]
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k-1], 50)
        y_b = b(xx)

        y_n = [_naive_eval(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n, atol=1e-14)

        y_n2 = [_naive_eval_2(x, t, c, k) for x in xx]
        assert_allclose(y_b, y_n2, atol=1e-14)

    def test_rndm_splev(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k-1], 50)
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)

    def test_rndm_splrep(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)

        tck = splrep(x, y)
        b = BSpline(*tck)

        t, k = b.t, b.k
        xx = np.linspace(t[k], t[-k-1], 80)
        assert_allclose(b(xx), splev(xx, tck), atol=1e-14)

    def test_rndm_unity(self):
        b = _make_random_spline()
        b.c = np.ones_like(b.c)
        xx = np.linspace(b.t[b.k], b.t[-b.k-1], 100)
        assert_allclose(b(xx), 1.)

    def test_vectorization(self):
        n, k = 22, 3
        t = np.sort(np.random.random(n))
        c = np.random.random(size=(n, 6, 7))
        b = BSpline(t, c, k)
        tm, tp = t[k], t[-k-1]
        xx = tm + (tp - tm) * np.random.random((3, 4, 5))
        assert_equal(b(xx).shape, (3, 4, 5, 6, 7))

    def test_len_c(self):
        # for n+k+1 knots, only first n coefs are used.
        # and BTW this is consistent with FITPACK
        n, k = 33, 3
        t = np.sort(np.random.random(n+k+1))
        c = np.random.random(n)

        # pad coefficients with random garbage
        c_pad = np.r_[c, np.random.random(k+1)]

        b, b_pad = BSpline(t, c, k), BSpline(t, c_pad, k)

        dt = t[-1] - t[0]
        xx = np.linspace(t[0] - dt, t[-1] + dt, 50)
        assert_allclose(b(xx), b_pad(xx), atol=1e-14)
        assert_allclose(b(xx), splev(xx, (t, c, k)), atol=1e-14)
        assert_allclose(b(xx), splev(xx, (t, c_pad, k)), atol=1e-14)

    def test_endpoints(self):
        # base interval is closed
        b = _make_random_spline()
        t, _, k = b.tck
        tm, tp = t[k], t[-k-1]
        for extrap in (True, False):
            assert_allclose(b([tm, tp], extrap),
                            b([tm + 1e-10, tp - 1e-10], extrap), atol=1e-9)

    def test_continuity(self):
        # assert continuity at internal knots
        b = _make_random_spline()
        t, _, k = b.tck
        assert_allclose(b(t[k+1:-k-1] - 1e-10), b(t[k+1:-k-1] + 1e-10),
                atol=1e-9)

    def test_extrap(self):
        b = _make_random_spline()
        t, c, k = b.tck
        dt = t[-1] - t[0]
        xx = np.linspace(t[k] - dt, t[-k-1] + dt, 50)
        mask = (t[k] < xx) & (xx < t[-k-1])

        # extrap has no effect within the base interval
        assert_allclose(b(xx[mask], extrapolate=True),
                        b(xx[mask], extrapolate=False))

        # extrapolated values agree with FITPACK
        assert_allclose(b(xx, extrapolate=True),
                splev(xx, (t, c, k), ext=0))

    def test_default_extrap(self):
        # BSpline defaults to extrapolate=True
        b = _make_random_spline()
        t, _, k = b.tck
        xx = [t[0] - 1, t[-1] + 1]
        yy = b(xx)
        assert_(not np.all(np.isnan(yy)))

    def test_periodic_extrap(self):
        np.random.seed(1234)
        t = np.sort(np.random.random(8))
        c = np.random.random(4)
        k = 3
        b = BSpline(t, c, k, extrapolate='periodic')
        n = t.size - (k + 1)

        dt = t[-1] - t[0]
        xx = np.linspace(t[k] - dt, t[n] + dt, 50)
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        assert_allclose(b(xx), splev(xy, (t, c, k)))

        # Direct check
        xx = [-1, 0, 0.5, 1]
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        assert_equal(b(xx, extrapolate='periodic'), b(xy, extrapolate=True))

    def test_ppoly(self):
        b = _make_random_spline()
        t, c, k = b.tck
        pp = PPoly.from_spline((t, c, k))

        xx = np.linspace(t[k], t[-k], 100)
        assert_allclose(b(xx), pp(xx), atol=1e-14, rtol=1e-14)

    def test_derivative_rndm(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[0], t[-1], 50)
        xx = np.r_[xx, t]

        for der in range(1, k+1):
            yd = splev(xx, (t, c, k), der=der)
            assert_allclose(yd, b(xx, nu=der), atol=1e-14)

        # higher derivatives all vanish
        assert_allclose(b(xx, nu=k+1), 0, atol=1e-14)

    def test_derivative_jumps(self):
        # example from de Boor, Chap IX, example (24)
        # NB: knots augmented & corresp coefs are zeroed out
        # in agreement with the convention (29)
        k = 2
        t = [-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7]
        np.random.seed(1234)
        c = np.r_[0, 0, np.random.random(5), 0, 0]
        b = BSpline(t, c, k)

        # b is continuous at x != 6 (triple knot)
        x = np.asarray([1, 3, 4, 6])
        assert_allclose(b(x[x != 6] - 1e-10),
                        b(x[x != 6] + 1e-10))
        assert_(not np.allclose(b(6.-1e-10), b(6+1e-10)))

        # 1st derivative jumps at double knots, 1 & 6:
        x0 = np.asarray([3, 4])
        assert_allclose(b(x0 - 1e-10, nu=1),
                        b(x0 + 1e-10, nu=1))
        x1 = np.asarray([1, 6])
        assert_(not np.all(np.allclose(b(x1 - 1e-10, nu=1),
                                       b(x1 + 1e-10, nu=1))))

        # 2nd derivative is not guaranteed to be continuous either
        assert_(not np.all(np.allclose(b(x - 1e-10, nu=2),
                                       b(x + 1e-10, nu=2))))

    def test_basis_element_quadratic(self):
        xx = np.linspace(-1, 4, 20)
        b = BSpline.basis_element(t=[0, 1, 2, 3])
        assert_allclose(b(xx),
                        splev(xx, (b.t, b.c, b.k)), atol=1e-14)
        assert_allclose(b(xx),
                        B_0123(xx), atol=1e-14)

        b = BSpline.basis_element(t=[0, 1, 1, 2])
        xx = np.linspace(0, 2, 10)
        assert_allclose(b(xx),
                np.where(xx < 1, xx*xx, (2.-xx)**2), atol=1e-14)

    def test_basis_element_rndm(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b(xx), _sum_basis_elements(xx, t, c, k), atol=1e-14)

    def test_cmplx(self):
        b = _make_random_spline()
        t, c, k = b.tck
        cc = c * (1. + 3.j)

        b = BSpline(t, cc, k)
        b_re = BSpline(t, b.c.real, k)
        b_im = BSpline(t, b.c.imag, k)

        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b(xx).real, b_re(xx), atol=1e-14)
        assert_allclose(b(xx).imag, b_im(xx), atol=1e-14)

    def test_nan(self):
        # nan in, nan out.
        b = BSpline.basis_element([0, 1, 1, 2])
        assert_(np.isnan(b(np.nan)))

    def test_derivative_method(self):
        b = _make_random_spline(k=5)
        t, c, k = b.tck
        b0 = BSpline(t, c, k)
        xx = np.linspace(t[k], t[-k-1], 20)
        for j in range(1, k):
            b = b.derivative()
            assert_allclose(b0(xx, j), b(xx), atol=1e-12, rtol=1e-12)

    def test_antiderivative_method(self):
        b = _make_random_spline()
        t, c, k = b.tck
        xx = np.linspace(t[k], t[-k-1], 20)
        assert_allclose(b.antiderivative().derivative()(xx),
                        b(xx), atol=1e-14, rtol=1e-14)

        # repeat with N-D array for c
        c = np.c_[c, c, c]
        c = np.dstack((c, c))
        b = BSpline(t, c, k)
        assert_allclose(b.antiderivative().derivative()(xx),
                        b(xx), atol=1e-14, rtol=1e-14)

    def test_integral(self):
        b = BSpline.basis_element([0, 1, 2])  # x for x < 1 else 2 - x
        assert_allclose(b.integrate(0, 1), 0.5)
        assert_allclose(b.integrate(1, 0), -1 * 0.5)
        assert_allclose(b.integrate(1, 0), -0.5)

        # extrapolate or zeros outside of [0, 2]; default is yes
        assert_allclose(b.integrate(-1, 1), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=True), 0)
        assert_allclose(b.integrate(-1, 1, extrapolate=False), 0.5)
        assert_allclose(b.integrate(1, -1, extrapolate=False), -1 * 0.5)

        # Test ``_fitpack._splint()``
        assert_allclose(b.integrate(1, -1, extrapolate=False),
                        _impl.splint(1, -1, b.tck))

        # Test ``extrapolate='periodic'``.
        b.extrapolate = 'periodic'
        i = b.antiderivative()
        period_int = i(2) - i(0)

        assert_allclose(b.integrate(0, 2), period_int)
        assert_allclose(b.integrate(2, 0), -1 * period_int)
        assert_allclose(b.integrate(-9, -7), period_int)
        assert_allclose(b.integrate(-8, -4), 2 * period_int)

        assert_allclose(b.integrate(0.5, 1.5), i(1.5) - i(0.5))
        assert_allclose(b.integrate(1.5, 3), i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5 + 12, 3 + 12),
                        i(1) - i(0) + i(2) - i(1.5))
        assert_allclose(b.integrate(1.5, 3 + 12),
                        i(1) - i(0) + i(2) - i(1.5) + 6 * period_int)

        assert_allclose(b.integrate(0, -1), i(0) - i(1))
        assert_allclose(b.integrate(-9, -10), i(0) - i(1))
        assert_allclose(b.integrate(0, -9), i(1) - i(2) - 4 * period_int)

    def test_integrate_ppoly(self):
        # test .integrate method to be consistent with PPoly.integrate
        x = [0, 1, 2, 3, 4]
        b = make_interp_spline(x, x)
        b.extrapolate = 'periodic'
        p = PPoly.from_spline(b)

        for x0, x1 in [(-5, 0.5), (0.5, 5), (-4, 13)]:
            assert_allclose(b.integrate(x0, x1),
                            p.integrate(x0, x1))

    def test_subclassing(self):
        # classmethods should not decay to the base class
        class B(BSpline):
            pass

        b = B.basis_element([0, 1, 2, 2])
        assert_equal(b.__class__, B)
        assert_equal(b.derivative().__class__, B)
        assert_equal(b.antiderivative().__class__, B)

    @pytest.mark.parametrize('axis', range(-4, 4))
    def test_axis(self, axis):
        n, k = 22, 3
        t = np.linspace(0, 1, n + k + 1)
        sh = [6, 7, 8]
        # We need the positive axis for some of the indexing and slices used
        # in this test.
        pos_axis = axis % 4
        sh.insert(pos_axis, n)   # [22, 6, 7, 8] etc
        c = np.random.random(size=sh)
        b = BSpline(t, c, k, axis=axis)
        assert_equal(b.c.shape,
                     [sh[pos_axis],] + sh[:pos_axis] + sh[pos_axis+1:])

        xp = np.random.random((3, 4, 5))
        assert_equal(b(xp).shape,
                     sh[:pos_axis] + list(xp.shape) + sh[pos_axis+1:])

        # -c.ndim <= axis < c.ndim
        for ax in [-c.ndim - 1, c.ndim]:
            assert_raises(AxisError, BSpline,
                          **dict(t=t, c=c, k=k, axis=ax))

        # derivative, antiderivative keeps the axis
        for b1 in [BSpline(t, c, k, axis=axis).derivative(),
                   BSpline(t, c, k, axis=axis).derivative(2),
                   BSpline(t, c, k, axis=axis).antiderivative(),
                   BSpline(t, c, k, axis=axis).antiderivative(2)]:
            assert_equal(b1.axis, b.axis)

    def test_neg_axis(self):
        k = 2
        t = [0, 1, 2, 3, 4, 5, 6]
        c = np.array([[-1, 2, 0, -1], [2, 0, -3, 1]])

        spl = BSpline(t, c, k, axis=-1)
        spl0 = BSpline(t, c[0], k)
        spl1 = BSpline(t, c[1], k)
        assert_equal(spl(2.5), [spl0(2.5), spl1(2.5)])

    def test_design_matrix_bc_types(self):
        '''
        Splines with different boundary conditions are built on different
        types of vectors of knots. As far as design matrix depends only on
        vector of knots, `k` and `x` it is useful to make tests for different
        boundary conditions (and as following different vectors of knots).
        '''
        def run_design_matrix_tests(n, k, bc_type):
            '''
            To avoid repetition of code the following function is provided.
            '''
            np.random.seed(1234)
            x = np.sort(np.random.random_sample(n) * 40 - 20)
            y = np.random.random_sample(n) * 40 - 20
            if bc_type == "periodic":
                y[0] = y[-1]

            bspl = make_interp_spline(x, y, k=k, bc_type=bc_type)

            c = np.eye(len(bspl.t) - k - 1)
            des_matr_def = BSpline(bspl.t, c, k)(x)
            des_matr_csr = BSpline.design_matrix(x,
                                                 bspl.t,
                                                 k).toarray()
            assert_allclose(des_matr_csr @ bspl.c, y, atol=1e-14)
            assert_allclose(des_matr_def, des_matr_csr, atol=1e-14)

        # "clamped" and "natural" work only with `k = 3`
        n = 11
        k = 3
        for bc in ["clamped", "natural"]:
            run_design_matrix_tests(n, k, bc)

        # "not-a-knot" works with odd `k`
        for k in range(3, 8, 2):
            run_design_matrix_tests(n, k, "not-a-knot")

        # "periodic" works with any `k` (even more than `n`)
        n = 5  # smaller `n` to test `k > n` case
        for k in range(2, 7):
            run_design_matrix_tests(n, k, "periodic")

    @pytest.mark.parametrize('extrapolate', [False, True, 'periodic'])
    @pytest.mark.parametrize('degree', range(5))
    def test_design_matrix_same_as_BSpline_call(self, extrapolate, degree):
        """Test that design_matrix(x) is equivalent to BSpline(..)(x)."""
        np.random.seed(1234)
        x = np.random.random_sample(10 * (degree + 1))
        xmin, xmax = np.amin(x), np.amax(x)
        k = degree
        t = np.r_[np.linspace(xmin - 2, xmin - 1, degree),
                  np.linspace(xmin, xmax, 2 * (degree + 1)),
                  np.linspace(xmax + 1, xmax + 2, degree)]
        c = np.eye(len(t) - k - 1)
        bspline = BSpline(t, c, k, extrapolate)
        assert_allclose(
            bspline(x), BSpline.design_matrix(x, t, k, extrapolate).toarray()
        )

        # extrapolation regime
        x = np.array([xmin - 10, xmin - 1, xmax + 1.5, xmax + 10])
        if not extrapolate:
            with pytest.raises(ValueError):
                BSpline.design_matrix(x, t, k, extrapolate)
        else:
            assert_allclose(
                bspline(x),
                BSpline.design_matrix(x, t, k, extrapolate).toarray()
            )

    def test_design_matrix_x_shapes(self):
        # test for different `x` shapes
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)
        y = np.random.random_sample(n) * 40 - 20

        bspl = make_interp_spline(x, y, k=k)
        for i in range(1, 4):
            xc = x[:i]
            yc = y[:i]
            des_matr_csr = BSpline.design_matrix(xc,
                                                 bspl.t,
                                                 k).toarray()
            assert_allclose(des_matr_csr @ bspl.c, yc, atol=1e-14)

    def test_design_matrix_t_shapes(self):
        # test for minimal possible `t` shape
        t = [1., 1., 1., 2., 3., 4., 4., 4.]
        des_matr = BSpline.design_matrix(2., t, 3).toarray()
        assert_allclose(des_matr,
                        [[0.25, 0.58333333, 0.16666667, 0.]],
                        atol=1e-14)

    def test_design_matrix_asserts(self):
        np.random.seed(1234)
        n = 10
        k = 3
        x = np.sort(np.random.random_sample(n) * 40 - 20)
        y = np.random.random_sample(n) * 40 - 20
        bspl = make_interp_spline(x, y, k=k)
        # invalid vector of knots (should be a 1D non-descending array)
        # here the actual vector of knots is reversed, so it is invalid
        with assert_raises(ValueError):
            BSpline.design_matrix(x, bspl.t[::-1], k)
        k = 2
        t = [0., 1., 2., 3., 4., 5.]
        x = [1., 2., 3., 4.]
        # out of bounds
        with assert_raises(ValueError):
            BSpline.design_matrix(x, t, k)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped',
                                         'periodic', 'not-a-knot'])
    def test_from_power_basis(self, bc_type):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20)
        if bc_type == 'periodic':
            y[-1] = y[0]
        cb = CubicSpline(x, y, bc_type=bc_type)
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
        xx = np.linspace(0, 1, 20)
        assert_allclose(cb(xx), bspl(xx), atol=1e-15)
        bspl_new = make_interp_spline(x, y, bc_type=bc_type)
        assert_allclose(bspl.c, bspl_new.c, atol=1e-15)

    @pytest.mark.parametrize('bc_type', ['natural', 'clamped',
                                         'periodic', 'not-a-knot'])
    def test_from_power_basis_complex(self, bc_type):
        np.random.seed(1234)
        x = np.sort(np.random.random(20))
        y = np.random.random(20) + np.random.random(20) * 1j
        if bc_type == 'periodic':
            y[-1] = y[0]
        cb = CubicSpline(x, y, bc_type=bc_type)
        bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
        bspl_new_real = make_interp_spline(x, y.real, bc_type=bc_type)
        bspl_new_imag = make_interp_spline(x, y.imag, bc_type=bc_type)
        assert_equal(bspl.c.dtype, (bspl_new_real.c
                                    + 1j * bspl_new_imag.c).dtype)
        assert_allclose(bspl.c, bspl_new_real.c
                        + 1j * bspl_new_imag.c, atol=1e-15)

    def test_from_power_basis_exmp(self):
        '''
        For x = [0, 1, 2, 3, 4] and y = [1, 1, 1, 1, 1]
        the coefficients of Cubic Spline in the power basis:

        $[[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[1, 1, 1, 1, 1]]$

        It could be shown explicitly that coefficients of the interpolating
        function in B-spline basis are c = [1, 1, 1, 1, 1, 1, 1]
        '''
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 1, 1, 1, 1])
        bspl = BSpline.from_power_basis(CubicSpline(x, y, bc_type='natural'),
                                        bc_type='natural')
        assert_allclose(bspl.c, [1, 1, 1, 1, 1, 1, 1], atol=1e-15)

    def test_read_only(self):
        # BSpline must work on read-only knots and coefficients.
        t = np.array([0, 1])
        c = np.array([3.0])
        t.setflags(write=False)
        c.setflags(write=False)

        xx = np.linspace(0, 1, 10)
        xx.setflags(write=False)

        b = BSpline(t=t, c=c, k=0)
        assert_allclose(b(xx), 3)


def test_knots_multiplicity():
    # Take a spline w/ random coefficients, throw in knots of varying
    # multiplicity.

    def check_splev(b, j, der=0, atol=1e-14, rtol=1e-14):
        # check evaluations against FITPACK, incl extrapolations
        t, c, k = b.tck
        x = np.unique(t)
        x = np.r_[t[0]-0.1, 0.5*(x[1:] + x[:1]), t[-1]+0.1]
        assert_allclose(splev(x, (t, c, k), der), b(x, der),
                atol=atol, rtol=rtol, err_msg=f'der = {der}  k = {b.k}')

    # test loop itself
    # [the index `j` is for interpreting the traceback in case of a failure]
    for k in [1, 2, 3, 4, 5]:
        b = _make_random_spline(k=k)
        for j, b1 in enumerate(_make_multiples(b)):
            check_splev(b1, j)
            for der in range(1, k+1):
                check_splev(b1, j, der, 1e-12, 1e-12)


### stolen from @pv, verbatim
def _naive_B(x, k, i, t):
    """
    Naive way to compute B-spline basis functions. Useful only for testing!
    computes B(x; t[i],..., t[i+k+1])
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * _naive_B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _naive_B(x, k-1, i+1, t)
    return (c1 + c2)


### stolen from @pv, verbatim
def _naive_eval(x, t, c, k):
    """
    Naive B-spline evaluation. Useful only for testing!
    """
    if x == t[k]:
        i = k
    else:
        i = np.searchsorted(t, x) - 1
    assert t[i] <= x <= t[i+1]
    assert i >= k and i < len(t) - k
    return sum(c[i-j] * _naive_B(x, k, i-j, t) for j in range(0, k+1))


def _naive_eval_2(x, t, c, k):
    """Naive B-spline evaluation, another way."""
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    assert t[k] <= x <= t[n]
    return sum(c[i] * _naive_B(x, k, i, t) for i in range(n))


def _sum_basis_elements(x, t, c, k):
    n = len(t) - (k+1)
    assert n >= k+1
    assert len(c) >= n
    s = 0.
    for i in range(n):
        b = BSpline.basis_element(t[i:i+k+2], extrapolate=False)(x)
        s += c[i] * np.nan_to_num(b)   # zero out out-of-bounds elements
    return s


def B_012(x):
    """ A linear B-spline function B(x | 0, 1, 2)."""
    x = np.atleast_1d(x)
    return np.piecewise(x, [(x < 0) | (x > 2),
                            (x >= 0) & (x < 1),
                            (x >= 1) & (x <= 2)],
                           [lambda x: 0., lambda x: x, lambda x: 2.-x])


def B_0123(x, der=0):
    """A quadratic B-spline function B(x | 0, 1, 2, 3)."""
    x = np.atleast_1d(x)
    conds = [x < 1, (x > 1) & (x < 2), x > 2]
    if der == 0:
        funcs = [lambda x: x*x/2.,
                 lambda x: 3./4 - (x-3./2)**2,
                 lambda x: (3.-x)**2 / 2]
    elif der == 2:
        funcs = [lambda x: 1.,
                 lambda x: -2.,
                 lambda x: 1.]
    else:
        raise ValueError('never be here: der=%s' % der)
    pieces = np.piecewise(x, conds, funcs)
    return pieces


def _make_random_spline(n=35, k=3):
    np.random.seed(123)
    t = np.sort(np.random.random(n+k+1))
    c = np.random.random(n)
    return BSpline.construct_fast(t, c, k)


def _make_multiples(b):
    """Increase knot multiplicity."""
    c, k = b.c, b.k

    t1 = b.t.copy()
    t1[17:19] = t1[17]
    t1[22] = t1[21]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[:k+1] = t1[0]
    yield BSpline(t1, c, k)

    t1 = b.t.copy()
    t1[-k-1:] = t1[-1]
    yield BSpline(t1, c, k)


class TestInterop:
    #
    # Test that FITPACK-based spl* functions can deal with BSpline objects
    #
    def setup_method(self):
        xx = np.linspace(0, 4.*np.pi, 41)
        yy = np.cos(xx)
        b = make_interp_spline(xx, yy)
        self.tck = (b.t, b.c, b.k)
        self.xx, self.yy, self.b = xx, yy, b

        self.xnew = np.linspace(0, 4.*np.pi, 21)

        c2 = np.c_[b.c, b.c, b.c]
        self.c2 = np.dstack((c2, c2))
        self.b2 = BSpline(b.t, self.c2, b.k)

    def test_splev(self):
        xnew, b, b2 = self.xnew, self.b, self.b2

        # check that splev works with 1-D array of coefficients
        # for array and scalar `x`
        assert_allclose(splev(xnew, b),
                        b(xnew), atol=1e-15, rtol=1e-15)
        assert_allclose(splev(xnew, b.tck),
                        b(xnew), atol=1e-15, rtol=1e-15)
        assert_allclose([splev(x, b) for x in xnew],
                        b(xnew), atol=1e-15, rtol=1e-15)

        # With N-D coefficients, there's a quirck:
        # splev(x, BSpline) is equivalent to BSpline(x)
        with assert_raises(ValueError, match="Calling splev.. with BSpline"):
            splev(xnew, b2)

        # However, splev(x, BSpline.tck) needs some transposes. This is because
        # BSpline interpolates along the first axis, while the legacy FITPACK
        # wrapper does list(map(...)) which effectively interpolates along the
        # last axis. Like so:
        sh = tuple(range(1, b2.c.ndim)) + (0,)   # sh = (1, 2, 0)
        cc = b2.c.transpose(sh)
        tck = (b2.t, cc, b2.k)
        assert_allclose(splev(xnew, tck),
                        b2(xnew).transpose(sh), atol=1e-15, rtol=1e-15)

    def test_splrep(self):
        x, y = self.xx, self.yy
        # test that "new" splrep is equivalent to _impl.splrep
        tck = splrep(x, y)
        t, c, k = _impl.splrep(x, y)
        assert_allclose(tck[0], t, atol=1e-15)
        assert_allclose(tck[1], c, atol=1e-15)
        assert_equal(tck[2], k)

        # also cover the `full_output=True` branch
        tck_f, _, _, _ = splrep(x, y, full_output=True)
        assert_allclose(tck_f[0], t, atol=1e-15)
        assert_allclose(tck_f[1], c, atol=1e-15)
        assert_equal(tck_f[2], k)

        # test that the result of splrep roundtrips with splev:
        # evaluate the spline on the original `x` points
        yy = splev(x, tck)
        assert_allclose(y, yy, atol=1e-15)

        # ... and also it roundtrips if wrapped in a BSpline
        b = BSpline(*tck)
        assert_allclose(y, b(x), atol=1e-15)

    def test_splrep_errors(self):
        # test that both "old" and "new" splrep raise for an N-D ``y`` array
        # with n > 1
        x, y = self.xx, self.yy
        y2 = np.c_[y, y]
        with assert_raises(ValueError):
            splrep(x, y2)
        with assert_raises(ValueError):
            _impl.splrep(x, y2)

        # input below minimum size
        with assert_raises(TypeError, match="m > k must hold"):
            splrep(x[:3], y[:3])
        with assert_raises(TypeError, match="m > k must hold"):
            _impl.splrep(x[:3], y[:3])

    def test_splprep(self):
        x = np.arange(15).reshape((3, 5))
        b, u = splprep(x)
        tck, u1 = _impl.splprep(x)

        # test the roundtrip with splev for both "old" and "new" output
        assert_allclose(u, u1, atol=1e-15)
        assert_allclose(splev(u, b), x, atol=1e-15)
        assert_allclose(splev(u, tck), x, atol=1e-15)

        # cover the ``full_output=True`` branch
        (b_f, u_f), _, _, _ = splprep(x, s=0, full_output=True)
        assert_allclose(u, u_f, atol=1e-15)
        assert_allclose(splev(u_f, b_f), x, atol=1e-15)

    def test_splprep_errors(self):
        # test that both "old" and "new" code paths raise for x.ndim > 2
        x = np.arange(3*4*5).reshape((3, 4, 5))
        with assert_raises(ValueError, match="too many values to unpack"):
            splprep(x)
        with assert_raises(ValueError, match="too many values to unpack"):
            _impl.splprep(x)

        # input below minimum size
        x = np.linspace(0, 40, num=3)
        with assert_raises(TypeError, match="m > k must hold"):
            splprep([x])
        with assert_raises(TypeError, match="m > k must hold"):
            _impl.splprep([x])

        # automatically calculated parameters are non-increasing
        # see gh-7589
        x = [-50.49072266, -50.49072266, -54.49072266, -54.49072266]
        with assert_raises(ValueError, match="Invalid inputs"):
            splprep([x])
        with assert_raises(ValueError, match="Invalid inputs"):
            _impl.splprep([x])

        # given non-increasing parameter values u
        x = [1, 3, 2, 4]
        u = [0, 0.3, 0.2, 1]
        with assert_raises(ValueError, match="Invalid inputs"):
            splprep(*[[x], None, u])

    def test_sproot(self):
        b, b2 = self.b, self.b2
        roots = np.array([0.5, 1.5, 2.5, 3.5])*np.pi
        # sproot accepts a BSpline obj w/ 1-D coef array
        assert_allclose(sproot(b), roots, atol=1e-7, rtol=1e-7)
        assert_allclose(sproot((b.t, b.c, b.k)), roots, atol=1e-7, rtol=1e-7)

        # ... and deals with trailing dimensions if coef array is N-D
        with assert_raises(ValueError, match="Calling sproot.. with BSpline"):
            sproot(b2, mest=50)

        # and legacy behavior is preserved for a tck tuple w/ N-D coef
        c2r = b2.c.transpose(1, 2, 0)
        rr = np.asarray(sproot((b2.t, c2r, b2.k), mest=50))
        assert_equal(rr.shape, (3, 2, 4))
        assert_allclose(rr - roots, 0, atol=1e-12)

    def test_splint(self):
        # test that splint accepts BSpline objects
        b, b2 = self.b, self.b2
        assert_allclose(splint(0, 1, b),
                        splint(0, 1, b.tck), atol=1e-14)
        assert_allclose(splint(0, 1, b),
                        b.integrate(0, 1), atol=1e-14)

        # ... and deals with N-D arrays of coefficients
        with assert_raises(ValueError, match="Calling splint.. with BSpline"):
            splint(0, 1, b2)

        # and the legacy behavior is preserved for a tck tuple w/ N-D coef
        c2r = b2.c.transpose(1, 2, 0)
        integr = np.asarray(splint(0, 1, (b2.t, c2r, b2.k)))
        assert_equal(integr.shape, (3, 2))
        assert_allclose(integr,
                        splint(0, 1, b), atol=1e-14)

    def test_splder(self):
        for b in [self.b, self.b2]:
            # pad the c array (FITPACK convention)
            ct = len(b.t) - len(b.c)
            if ct > 0:
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]

            for n in [1, 2, 3]:
                bd = splder(b)
                tck_d = _impl.splder((b.t, b.c, b.k))
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                assert_equal(bd.k, tck_d[2])
                assert_(isinstance(bd, BSpline))
                assert_(isinstance(tck_d, tuple))  # back-compat: tck in and out

    def test_splantider(self):
        for b in [self.b, self.b2]:
            # pad the c array (FITPACK convention)
            ct = len(b.t) - len(b.c)
            if ct > 0:
                b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]

            for n in [1, 2, 3]:
                bd = splantider(b)
                tck_d = _impl.splantider((b.t, b.c, b.k))
                assert_allclose(bd.t, tck_d[0], atol=1e-15)
                assert_allclose(bd.c, tck_d[1], atol=1e-15)
                assert_equal(bd.k, tck_d[2])
                assert_(isinstance(bd, BSpline))
                assert_(isinstance(tck_d, tuple))  # back-compat: tck in and out

    def test_insert(self):
        b, b2, xx = self.b, self.b2, self.xx

        j = b.t.size // 2
        tn = 0.5*(b.t[j] + b.t[j+1])

        bn, tck_n = insert(tn, b), insert(tn, (b.t, b.c, b.k))
        assert_allclose(splev(xx, bn),
                        splev(xx, tck_n), atol=1e-15)
        assert_(isinstance(bn, BSpline))
        assert_(isinstance(tck_n, tuple))   # back-compat: tck in, tck out

        # for N-D array of coefficients, BSpline.c needs to be transposed
        # after that, the results are equivalent.
        sh = tuple(range(b2.c.ndim))
        c_ = b2.c.transpose(sh[1:] + (0,))
        tck_n2 = insert(tn, (b2.t, c_, b2.k))

        bn2 = insert(tn, b2)

        # need a transpose for comparing the results, cf test_splev
        assert_allclose(np.asarray(splev(xx, tck_n2)).transpose(2, 0, 1),
                        bn2(xx), atol=1e-15)
        assert_(isinstance(bn2, BSpline))
        assert_(isinstance(tck_n2, tuple))   # back-compat: tck in, tck out


class TestInterp:
    #
    # Test basic ways of constructing interpolating splines.
    #
    xx = np.linspace(0., 2.*np.pi)
    yy = np.sin(xx)

    def test_non_int_order(self):
        with assert_raises(TypeError):
            make_interp_spline(self.xx, self.yy, k=2.5)

    def test_order_0(self):
        b = make_interp_spline(self.xx, self.yy, k=0)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        b = make_interp_spline(self.xx, self.yy, k=0, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_linear(self):
        b = make_interp_spline(self.xx, self.yy, k=1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        b = make_interp_spline(self.xx, self.yy, k=1, axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, k):
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        with assert_raises(ValueError, match="Shapes of x"):
            make_interp_spline(x, y, k=k)

    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, k):
        x = [0, 1, 1, 2, 3, 4]      # duplicates
        y = [0, 1, 2, 3, 4, 5]
        with assert_raises(ValueError, match="x to not have duplicates"):
            make_interp_spline(x, y, k=k)

        x = [0, 2, 1, 3, 4, 5]      # unsorted
        with assert_raises(ValueError, match="Expect x to be a 1D strictly"):
            make_interp_spline(x, y, k=k)

        x = [0, 1, 2, 3, 4, 5]
        x = np.asarray(x).reshape((1, -1))     # 1D
        with assert_raises(ValueError, match="Expect x to be a 1D strictly"):
            make_interp_spline(x, y, k=k)

    def test_not_a_knot(self):
        for k in [3, 5]:
            b = make_interp_spline(self.xx, self.yy, k)
            assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_periodic(self):
        # k = 5 here for more derivatives
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic')
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        # in periodic case it is expected equality of k-1 first
        # derivatives at the boundaries
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)
        # tests for axis=-1
        b = make_interp_spline(self.xx, self.yy, k=5, bc_type='periodic', axis=-1)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        for i in range(1, 5):
            assert_allclose(b(self.xx[0], nu=i), b(self.xx[-1], nu=i), atol=1e-11)

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
    def test_periodic_random(self, k):
        # tests for both cases (k > n and k <= n)
        n = 5
        np.random.seed(1234)
        x = np.sort(np.random.random_sample(n) * 10)
        y = np.random.random_sample(n) * 100
        y[0] = y[-1]
        b = make_interp_spline(x, y, k=k, bc_type='periodic')
        assert_allclose(b(x), y, atol=1e-14)

    def test_periodic_axis(self):
        n = self.xx.shape[0]
        np.random.seed(1234)
        x = np.random.random_sample(n) * 2 * np.pi
        x = np.sort(x)
        x[0] = 0.
        x[-1] = 2 * np.pi
        y = np.zeros((2, n))
        y[0] = np.sin(x)
        y[1] = np.cos(x)
        b = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
        for i in range(n):
            assert_allclose(b(x[i]), y[:, i], atol=1e-14)
        assert_allclose(b(x[0]), b(x[-1]), atol=1e-14)

    def test_periodic_points_exception(self):
        # first and last points should match when periodic case expected
        np.random.seed(1234)
        k = 5
        n = 8
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        y[0] = y[-1] - 1  # to be sure that they are not equal
        with assert_raises(ValueError):
            make_interp_spline(x, y, k=k, bc_type='periodic')

    def test_periodic_knots_exception(self):
        # `periodic` case does not work with passed vector of knots
        np.random.seed(1234)
        k = 3
        n = 7
        x = np.sort(np.random.random_sample(n))
        y = np.random.random_sample(n)
        t = np.zeros(n + 2 * k)
        with assert_raises(ValueError):
            make_interp_spline(x, y, k, t, 'periodic')

    @pytest.mark.parametrize('k', [2, 3, 4, 5])
    def test_periodic_splev(self, k):
        # comparison values of periodic b-spline with splev
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        tck = splrep(self.xx, self.yy, per=True, k=k)
        spl = splev(self.xx, tck)
        assert_allclose(spl, b(self.xx), atol=1e-14)

        # comparison derivatives of periodic b-spline with splev
        for i in range(1, k):
            spl = splev(self.xx, tck, der=i)
            assert_allclose(spl, b(self.xx, nu=i), atol=1e-10)

    def test_periodic_cubic(self):
        # comparison values of cubic periodic b-spline with CubicSpline
        b = make_interp_spline(self.xx, self.yy, k=3, bc_type='periodic')
        cub = CubicSpline(self.xx, self.yy, bc_type='periodic')
        assert_allclose(b(self.xx), cub(self.xx), atol=1e-14)

        # edge case: Cubic interpolation on 3 points
        n = 3
        x = np.sort(np.random.random_sample(n) * 10)
        y = np.random.random_sample(n) * 100
        y[0] = y[-1]
        b = make_interp_spline(x, y, k=3, bc_type='periodic')
        cub = CubicSpline(x, y, bc_type='periodic')
        assert_allclose(b(x), cub(x), atol=1e-14)

    def test_periodic_full_matrix(self):
        # comparison values of cubic periodic b-spline with
        # solution of the system with full matrix
        k = 3
        b = make_interp_spline(self.xx, self.yy, k=k, bc_type='periodic')
        t = _periodic_knots(self.xx, k)
        c = _make_interp_per_full_matr(self.xx, self.yy, t, k)
        b1 = np.vectorize(lambda x: _naive_eval(x, t, c, k))
        assert_allclose(b(self.xx), b1(self.xx), atol=1e-14)

    def test_quadratic_deriv(self):
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # derivative at right-hand edge
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(None, der))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der[0][1], atol=1e-14, rtol=1e-14)

        # derivative at left-hand edge
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(der, None))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der[0][1], atol=1e-14, rtol=1e-14)

    def test_cubic_deriv(self):
        k = 3

        # first derivatives at left & right edges:
        der_l, der_r = [(1, 3.)], [(1, 4.)]
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 1), b(self.xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

        # 'natural' cubic spline, zero out 2nd derivatives at the boundaries
        der_l, der_r = [(2, 0)], [(2, 0)]
        b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_quintic_derivs(self):
        k, n = 5, 7
        x = np.arange(n).astype(np.float64)
        y = np.sin(x)
        der_l = [(1, -12.), (2, 1)]
        der_r = [(1, 8.), (2, 3.)]
        b = make_interp_spline(x, y, k=k, bc_type=(der_l, der_r))
        assert_allclose(b(x), y, atol=1e-14, rtol=1e-14)
        assert_allclose([b(x[0], 1), b(x[0], 2)],
                        [val for (nu, val) in der_l])
        assert_allclose([b(x[-1], 1), b(x[-1], 2)],
                        [val for (nu, val) in der_r])

    @pytest.mark.xfail(reason='unstable')
    def test_cubic_deriv_unstable(self):
        # 1st and 2nd derivative at x[0], no derivative information at x[-1]
        # The problem is not that it fails [who would use this anyway],
        # the problem is that it fails *silently*, and I've no idea
        # how to detect this sort of instability.
        # In this particular case: it's OK for len(t) < 20, goes haywire
        # at larger `len(t)`.
        k = 3
        t = _augknt(self.xx, k)

        der_l = [(1, 3.), (2, 4.)]
        b = make_interp_spline(self.xx, self.yy, k, t, bc_type=(der_l, None))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)

    def test_knots_not_data_sites(self):
        # Knots need not coincide with the data sites.
        # use a quadratic spline, knots are at data averages,
        # two additional constraints are zero 2nd derivatives at edges
        k = 2
        t = np.r_[(self.xx[0],)*(k+1),
                  (self.xx[1:] + self.xx[:-1]) / 2.,
                  (self.xx[-1],)*(k+1)]
        b = make_interp_spline(self.xx, self.yy, k, t,
                               bc_type=([(2, 0)], [(2, 0)]))

        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(self.xx[0], 2), b(self.xx[-1], 2)], [0., 0.],
                atol=1e-14)

    def test_minimum_points_and_deriv(self):
        # interpolation of f(x) = x**3 between 0 and 1. f'(x) = 3 * xx**2 and
        # f'(0) = 0, f'(1) = 3.
        k = 3
        x = [0., 1.]
        y = [0., 1.]
        b = make_interp_spline(x, y, k, bc_type=([(1, 0.)], [(1, 3.)]))

        xx = np.linspace(0., 1.)
        yy = xx**3
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_deriv_spec(self):
        # If one of the derivatives is omitted, the spline definition is
        # incomplete.
        x = y = [1.0, 2, 3, 4, 5, 6]

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=([(1, 0.)], None))

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(1, 0.))

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=[(1, 0.)])

        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=42)

        # CubicSpline expects`bc_type=(left_pair, right_pair)`, while
        # here we expect `bc_type=(iterable, iterable)`.
        l, r = (1, 0.0), (1, 0.0)
        with assert_raises(ValueError):
            make_interp_spline(x, y, bc_type=(l, r))

    def test_complex(self):
        k = 3
        xx = self.xx
        yy = self.yy + 1.j*self.yy

        # first derivatives at left & right edges:
        der_l, der_r = [(1, 3.j)], [(1, 4.+2.j)]
        b = make_interp_spline(xx, yy, k, bc_type=(der_l, der_r))
        assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose([b(xx[0], 1), b(xx[-1], 1)],
                        [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)

        # also test zero and first order
        for k in (0, 1):
            b = make_interp_spline(xx, yy, k=k)
            assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)

    def test_int_xy(self):
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)

        # Cython chokes on "buffer type mismatch" (construction) or
        # "no matching signature found" (evaluation)
        for k in (0, 1, 2, 3):
            b = make_interp_spline(x, y, k=k)
            b(x)

    def test_sliced_input(self):
        # Cython code chokes on non C contiguous arrays
        xx = np.linspace(-1, 1, 100)

        x = xx[::5]
        y = xx[::5]

        for k in (0, 1, 2, 3):
            make_interp_spline(x, y, k=k)

    def test_check_finite(self):
        # check_finite defaults to True; nans and such trigger a ValueError
        x = np.arange(10).astype(float)
        y = x**2

        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            assert_raises(ValueError, make_interp_spline, x, y)

    @pytest.mark.parametrize('k', [1, 2, 3, 5])
    def test_list_input(self, k):
        # regression test for gh-8714: TypeError for x, y being lists and k=2
        x = list(range(10))
        y = [a**2 for a in x]
        make_interp_spline(x, y, k=k)

    def test_multiple_rhs(self):
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [1., 2.])]
        der_r = [(1, [3., 4.])]

        b = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        assert_allclose(b(self.xx), yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der_l[0][1], atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der_r[0][1], atol=1e-14, rtol=1e-14)

    def test_shapes(self):
        np.random.seed(1234)
        k, n = 3, 22
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))

        b = make_interp_spline(x, y, k)
        assert_equal(b.c.shape, (n, 5, 6, 7))

        # now throw in some derivatives
        d_l = [(1, np.random.random((5, 6, 7)))]
        d_r = [(1, np.random.random((5, 6, 7)))]
        b = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        assert_equal(b.c.shape, (n + k - 1, 5, 6, 7))

    def test_string_aliases(self):
        yy = np.sin(self.xx)

        # a single string is duplicated
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='natural')
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(2, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)

        # two strings are handled
        b1 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=('natural', 'clamped'))
        b2 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=([(2, 0)], [(1, 0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)

        # one-sided BCs are OK
        b1 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, 'clamped'))
        b2 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, [(1, 0.0)]))
        assert_allclose(b1.c, b2.c, atol=1e-15)

        # 'not-a-knot' is equivalent to None
        b1 = make_interp_spline(self.xx, yy, k=3, bc_type='not-a-knot')
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=None)
        assert_allclose(b1.c, b2.c, atol=1e-15)

        # unknown strings do not pass
        with assert_raises(ValueError):
            make_interp_spline(self.xx, yy, k=3, bc_type='typo')

        # string aliases are handled for 2D values
        yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
        der_l = [(1, [0., 0.])]
        der_r = [(2, [0., 0.])]
        b2 = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
        b1 = make_interp_spline(self.xx, yy, k=3,
                                bc_type=('clamped', 'natural'))
        assert_allclose(b1.c, b2.c, atol=1e-15)

        # ... and for N-D values:
        np.random.seed(1234)
        k, n = 3, 22
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=(n, 5, 6, 7))

        # now throw in some derivatives
        d_l = [(1, np.zeros((5, 6, 7)))]
        d_r = [(1, np.zeros((5, 6, 7)))]
        b1 = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        b2 = make_interp_spline(x, y, k, bc_type='clamped')
        assert_allclose(b1.c, b2.c, atol=1e-15)

    def test_full_matrix(self):
        np.random.seed(1234)
        k, n = 3, 7
        x = np.sort(np.random.random(size=n))
        y = np.random.random(size=n)
        t = _not_a_knot(x, k)

        b = make_interp_spline(x, y, k, t)
        cf = make_interp_full_matr(x, y, t, k)
        assert_allclose(b.c, cf, atol=1e-14, rtol=1e-14)

    def test_woodbury(self):
        '''
        Random elements in diagonal matrix with blocks in the
        left lower and right upper corners checking the
        implementation of Woodbury algorithm.
        '''
        np.random.seed(1234)
        n = 201
        for k in range(3, 32, 2):
            offset = int((k - 1) / 2)
            a = np.diagflat(np.random.random((1, n)))
            for i in range(1, offset + 1):
                a[:-i, i:] += np.diagflat(np.random.random((1, n - i)))
                a[i:, :-i] += np.diagflat(np.random.random((1, n - i)))
            ur = np.random.random((offset, offset))
            a[:offset, -offset:] = ur
            ll = np.random.random((offset, offset))
            a[-offset:, :offset] = ll
            d = np.zeros((k, n))
            for i, j in enumerate(range(offset, -offset - 1, -1)):
                if j < 0:
                    d[i, :j] = np.diagonal(a, offset=j)
                else:
                    d[i, j:] = np.diagonal(a, offset=j)
            b = np.random.random(n)
            assert_allclose(_woodbury_algorithm(d, ur, ll, b, k),
                            np.linalg.solve(a, b), atol=1e-14)


def make_interp_full_matr(x, y, t, k):
    """Assemble an spline order k with knots t to interpolate
    y(x) using full matrices.
    Not-a-knot BC only.

    This routine is here for testing only (even though it's functional).
    """
    assert x.size == y.size
    assert t.size == x.size + k + 1
    n = x.size

    A = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        xval = x[j]
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j, left-k:left+1] = bb

    c = sl.solve(A, y)
    return c


def make_lsq_full_matrix(x, y, t, k=3):
    """Make the least-square spline, full matrices."""
    x, y, t = map(np.asarray, (x, y, t))
    m = x.size
    n = t.size - k - 1

    A = np.zeros((m, n), dtype=np.float64)

    for j in range(m):
        xval = x[j]
        # find interval
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j, left-k:left+1] = bb

    # have observation matrix, can solve the LSQ problem
    B = np.dot(A.T, A)
    Y = np.dot(A.T, y)
    c = sl.solve(B, Y)

    return c, (A, Y)


class TestLSQ:
    #
    # Test make_lsq_spline
    #
    np.random.seed(1234)
    n, k = 13, 3
    x = np.sort(np.random.random(n))
    y = np.random.random(n)
    t = _augknt(np.linspace(x[0], x[-1], 7), k)

    def test_lstsq(self):
        # check LSQ construction vs a full matrix version
        x, y, t, k = self.x, self.y, self.t, self.k

        c0, AY = make_lsq_full_matrix(x, y, t, k)
        b = make_lsq_spline(x, y, t, k)

        assert_allclose(b.c, c0)
        assert_equal(b.c.shape, (t.size - k - 1,))

        # also check against numpy.lstsq
        aa, yy = AY
        c1, _, _, _ = np.linalg.lstsq(aa, y, rcond=-1)
        assert_allclose(b.c, c1)

    def test_weights(self):
        # weights = 1 is same as None
        x, y, t, k = self.x, self.y, self.t, self.k
        w = np.ones_like(x)

        b = make_lsq_spline(x, y, t, k)
        b_w = make_lsq_spline(x, y, t, k, w=w)

        assert_allclose(b.t, b_w.t, atol=1e-14)
        assert_allclose(b.c, b_w.c, atol=1e-14)
        assert_equal(b.k, b_w.k)

    def test_multiple_rhs(self):
        x, t, k, n = self.x, self.t, self.k, self.n
        y = np.random.random(size=(n, 5, 6, 7))

        b = make_lsq_spline(x, y, t, k)
        assert_equal(b.c.shape, (t.size-k-1, 5, 6, 7))

    def test_complex(self):
        # cmplx-valued `y`
        x, t, k = self.x, self.t, self.k
        yc = self.y * (1. + 2.j)

        b = make_lsq_spline(x, yc, t, k)
        b_re = make_lsq_spline(x, yc.real, t, k)
        b_im = make_lsq_spline(x, yc.imag, t, k)

        assert_allclose(b(x), b_re(x) + 1.j*b_im(x), atol=1e-15, rtol=1e-15)

    def test_int_xy(self):
        x = np.arange(10).astype(int)
        y = np.arange(10).astype(int)
        t = _augknt(x, k=1)
        # Cython chokes on "buffer type mismatch"
        make_lsq_spline(x, y, t, k=1)

    def test_sliced_input(self):
        # Cython code chokes on non C contiguous arrays
        xx = np.linspace(-1, 1, 100)

        x = xx[::3]
        y = xx[::3]
        t = _augknt(x, 1)
        make_lsq_spline(x, y, t, k=1)

    def test_checkfinite(self):
        # check_finite defaults to True; nans and such trigger a ValueError
        x = np.arange(12).astype(float)
        y = x**2
        t = _augknt(x, 3)

        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            assert_raises(ValueError, make_lsq_spline, x, y, t)

    def test_read_only(self):
        # Check that make_lsq_spline works with read only arrays
        x, y, t = self.x, self.y, self.t
        x.setflags(write=False)
        y.setflags(write=False)
        t.setflags(write=False)
        make_lsq_spline(x=x, y=y, t=t)


def data_file(basename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)


class TestSmoothingSpline:
    #
    # test make_smoothing_spline
    #
    def test_invalid_input(self):
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        # ``x`` and ``y`` should have same shapes (1-D array)
        with assert_raises(ValueError):
            make_smoothing_spline(x, y[1:])
        with assert_raises(ValueError):
            make_smoothing_spline(x[1:], y)
        with assert_raises(ValueError):
            make_smoothing_spline(x.reshape(1, n), y)

        # ``x`` should be an ascending array
        with assert_raises(ValueError):
            make_smoothing_spline(x[::-1], y)

        x_dupl = np.copy(x)
        x_dupl[0] = x_dupl[1]

        with assert_raises(ValueError):
            make_smoothing_spline(x_dupl, y)

        # x and y length must be >= 5
        x = np.arange(4)
        y = np.ones(4)
        exception_message = "``x`` and ``y`` length must be at least 5"
        with pytest.raises(ValueError, match=exception_message):
            make_smoothing_spline(x, y)

    def test_compare_with_GCVSPL(self):
        """
        Data is generated in the following way:
        >>> np.random.seed(1234)
        >>> n = 100
        >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
        >>> y = np.sin(x) + np.random.normal(scale=.5, size=n)
        >>> np.savetxt('x.csv', x)
        >>> np.savetxt('y.csv', y)

        We obtain the result of performing the GCV smoothing splines
        package (by Woltring, gcvspl) on the sample data points
        using its version for Octave (https://github.com/srkuberski/gcvspl).
        In order to use this implementation, one should clone the repository
        and open the folder in Octave.
        In Octave, we load up ``x`` and ``y`` (generated from Python code
        above):

        >>> x = csvread('x.csv');
        >>> y = csvread('y.csv');

        Then, in order to access the implementation, we compile gcvspl files in
        Octave:

        >>> mex gcvsplmex.c gcvspl.c
        >>> mex spldermex.c gcvspl.c

        The first function computes the vector of unknowns from the dataset
        (x, y) while the second one evaluates the spline in certain points
        with known vector of coefficients.

        >>> c = gcvsplmex( x, y, 2 );
        >>> y0 = spldermex( x, c, 2, x, 0 );

        If we want to compare the results of the gcvspl code, we can save
        ``y0`` in csv file:

        >>> csvwrite('y0.csv', y0);

        """
        # load the data sample
        data = np.load(data_file('gcvspl.npz'))
        # data points
        x = data['x']
        y = data['y']

        y_GCVSPL = data['y_GCVSPL']
        y_compr = make_smoothing_spline(x, y)(x)

        # such tolerance is explained by the fact that the spline is built
        # using an iterative algorithm for minimizing the GCV criteria. These
        # algorithms may vary, so the tolerance should be rather low.
        assert_allclose(y_compr, y_GCVSPL, atol=1e-4, rtol=1e-4)

    def test_non_regularized_case(self):
        """
        In case the regularization parameter is 0, the resulting spline
        is an interpolation spline with natural boundary conditions.
        """
        # create data sample
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        spline_GCV = make_smoothing_spline(x, y, lam=0.)
        spline_interp = make_interp_spline(x, y, 3, bc_type='natural')

        grid = np.linspace(x[0], x[-1], 2 * n)
        assert_allclose(spline_GCV(grid),
                        spline_interp(grid),
                        atol=1e-15)

    def test_weighted_smoothing_spline(self):
        # create data sample
        np.random.seed(1234)
        n = 100
        x = np.sort(np.random.random_sample(n) * 4 - 2)
        y = x**2 * np.sin(4 * x) + x**3 + np.random.normal(0., 1.5, n)

        spl = make_smoothing_spline(x, y)

        # in order not to iterate over all of the indices, we select 10 of
        # them randomly
        for ind in np.random.choice(range(100), size=10):
            w = np.ones(n)
            w[ind] = 30.
            spl_w = make_smoothing_spline(x, y, w)
            # check that spline with weight in a certain point is closer to the
            # original point than the one without weights
            orig = abs(spl(x[ind]) - y[ind])
            weighted = abs(spl_w(x[ind]) - y[ind])

            if orig < weighted:
                raise ValueError(f'Spline with weights should be closer to the'
                                 f' points than the original one: {orig:.4} < '
                                 f'{weighted:.4}')


################################
# NdBSpline tests
def bspline2(xy, t, c, k):
    """A naive 2D tensort product spline evaluation."""
    x, y = xy
    tx, ty = t
    nx = len(tx) - k - 1
    assert (nx >= k+1)
    ny = len(ty) - k - 1
    assert (ny >= k+1)
    return sum(c[ix, iy] * B(x, k, ix, tx) * B(y, k, iy, ty)
               for ix in range(nx) for iy in range(ny))


def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2


def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))


class NdBSpline0:
    def __init__(self, t, c, k=3):
        """Tensor product spline object.

        c[i1, i2, ..., id] * B(x1, i1) * B(x2, i2) * ... * B(xd, id)

        Parameters
        ----------
        c : ndarray, shape (n1, n2, ..., nd, ...)
            b-spline coefficients
        t : tuple of 1D ndarrays
            knot vectors in directions 1, 2, ... d
            ``len(t[i]) == n[i] + k + 1``
        k : int or length-d tuple of integers
            spline degrees.
        """
        ndim = len(t)
        assert ndim <= len(c.shape)

        try:
            len(k)
        except TypeError:
            # make k a tuple
            k = (k,)*ndim

        self.k = tuple(operator.index(ki) for ki in k)
        self.t = tuple(np.asarray(ti, dtype=float) for ti in t)
        self.c = c

    def __call__(self, x):
        ndim = len(self.t)
        # a single evaluation point: `x` is a 1D array_like, shape (ndim,)
        assert len(x) == ndim

        # get the indices in an ndim-dimensional vector
        i = ['none', ]*ndim
        for d in range(ndim):
            td, xd = self.t[d], x[d]
            k = self.k[d]

            # find the index for x[d]
            if xd == td[k]:
                i[d] = k
            else:
                i[d] = np.searchsorted(td, xd) - 1
            assert td[i[d]] <= xd <= td[i[d]+1]
            assert i[d] >= k and i[d] < len(td) - k
        i = tuple(i)

        # iterate over the dimensions, form linear combinations of
        # products B(x_1) * B(x_2) * ... B(x_N) of (k+1)**N b-splines
        # which are non-zero at `i = (i_1, i_2, ..., i_N)`.
        result = 0
        iters = [range(i[d] - self.k[d], i[d] + 1) for d in range(ndim)]
        for idx in itertools.product(*iters):
            term = self.c[idx] * np.prod([B(x[d], self.k[d], idx[d], self.t[d])
                                          for d in range(ndim)])
            result += term
        return result


class TestNdBSpline:

    def test_1D(self):
        # test ndim=1 agrees with BSpline
        rng = np.random.default_rng(12345)
        n, k = 11, 3
        n_tr = 7
        t = np.sort(rng.uniform(size=n + k + 1))
        c = rng.uniform(size=(n, n_tr))

        b = BSpline(t, c, k)
        nb = NdBSpline((t,), c, k)

        xi = rng.uniform(size=21)
        # NdBSpline expects xi.shape=(npts, ndim)
        assert_allclose(nb(xi[:, None]),
                        b(xi), atol=1e-14)
        assert nb(xi[:, None]).shape == (xi.shape[0], c.shape[1])

    def make_2d_case(self):
        # make a 2D separable spline
        x = np.arange(6)
        y = x**3
        spl = make_interp_spline(x, y, k=3)

        y_1 = x**3 + 2*x
        spl_1 = make_interp_spline(x, y_1, k=3)

        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, 3

    def make_2d_mixed(self):
        # make a 2D separable spline w/ kx=3, ky=2
        x = np.arange(6)
        y = x**3
        spl = make_interp_spline(x, y, k=3)

        x = np.arange(5) + 1.5
        y_1 = x**2 + 2*x
        spl_1 = make_interp_spline(x, y_1, k=2)

        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, spl.k, spl_1.k

    def test_2D_separable(self):
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        t2, c2, k = self.make_2d_case()
        target = [x**3 * (y**3 + 2*y) for (x, y) in xi]

        # sanity check: bspline2 gives the product as constructed
        assert_allclose([bspline2(xy, t2, c2, k) for xy in xi],
                        target,
                        atol=1e-14)

        # check evaluation on a 2D array: the 1D array of 2D points
        bspl2 = NdBSpline(t2, c2, k=3)
        assert bspl2(xi).shape == (len(xi), )
        assert_allclose(bspl2(xi),
                        target, atol=1e-14)

        # now check on a multidim xi
        rng = np.random.default_rng(12345)
        xi = rng.uniform(size=(4, 3, 2)) * 5
        result = bspl2(xi)
        assert result.shape == (4, 3)

        # also check the values
        x, y = xi.reshape((-1, 2)).T
        assert_allclose(result.ravel(),
                        x**3 * (y**3 + 2*y), atol=1e-14)

    def test_2D_separable_2(self):
        # test `c` with trailing dimensions, i.e. c.ndim > ndim
        ndim = 2
        xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
        target = [x**3 * (y**3 + 2*y) for (x, y) in xi]

        t2, c2, k = self.make_2d_case()
        c2_4 = np.dstack((c2, c2, c2, c2))   # c22.shape = (6, 6, 4)

        xy = (1.5, 2.5)
        bspl2_4 = NdBSpline(t2, c2_4, k=3)
        result = bspl2_4(xy)
        val_single = NdBSpline(t2, c2, k)(xy)
        assert result.shape == (4,)
        assert_allclose(result,
                        [val_single, ]*4, atol=1e-14)

        # now try the array xi : the output.shape is (3, 4) where 3
        # is the number of points in xi and 4 is the trailing dimension of c
        assert bspl2_4(xi).shape == np.shape(xi)[:-1] + bspl2_4.c.shape[ndim:]
        assert_allclose(bspl2_4(xi) - np.asarray(target)[:, None],
                        0, atol=5e-14)

        # two trailing dimensions
        c2_22 = c2_4.reshape((6, 6, 2, 2))
        bspl2_22 = NdBSpline(t2, c2_22, k=3)

        result = bspl2_22(xy)
        assert result.shape == (2, 2)
        assert_allclose(result,
                        [[val_single, val_single],
                         [val_single, val_single]], atol=1e-14)

        # now try the array xi : the output shape is (3, 2, 2)
        # for 3 points in xi and c trailing dimensions being (2, 2)
        assert (bspl2_22(xi).shape ==
                np.shape(xi)[:-1] + bspl2_22.c.shape[ndim:])
        assert_allclose(bspl2_22(xi) - np.asarray(target)[:, None, None],
                        0, atol=5e-14)

    def test_2D_random(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size-k-1, ty.size-k-1))

        spl = NdBSpline((tx, ty), c, k=k)

        xi = (1., 1.)
        assert_allclose(spl(xi),
                        bspline2(xi, (tx, ty), c, k), atol=1e-14)

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]
        assert_allclose(spl(xi),
                        [bspline2(xy, (tx, ty), c, k) for xy in xi],
                        atol=1e-14)

    def test_2D_mixed(self):
        t2, c2, kx, ky = self.make_2d_mixed()
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        target = [x**3 * (y**2 + 2*y) for (x, y) in xi]

        bspl2 = NdBSpline(t2, c2, k=(kx, ky))
        assert bspl2(xi).shape == (len(xi), )
        assert_allclose(bspl2(xi),
                        target, atol=1e-14)

    def test_2D_derivative(self):
        t2, c2, kx, ky = self.make_2d_mixed()
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        bspl2 = NdBSpline(t2, c2, k=(kx, ky))

        der = bspl2(xi, nu=(1, 0))
        assert_allclose(der,
                        [3*x**2 * (y**2 + 2*y) for x, y in xi], atol=1e-14)

        der = bspl2(xi, nu=(1, 1))
        assert_allclose(der,
                        [3*x**2 * (2*y + 2) for x, y in xi], atol=1e-14)

        der = bspl2(xi, nu=(0, 0))
        assert_allclose(der,
                        [x**3 * (y**2 + 2*y) for x, y in xi], atol=1e-14)

    def test_2D_mixed_random(self):
        rng = np.random.default_rng(12345)
        kx, ky = 2, 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size - kx - 1, ty.size - ky - 1))

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        bspl2 = NdBSpline((tx, ty), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx, ty), c, k=(kx, ky))

        assert_allclose(bspl2(xi),
                        [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_tx_neq_ty(self):
        # 2D separable spline w/ len(tx) != len(ty)
        x = np.arange(6)
        y = np.arange(7) + 1.5

        spl_x = make_interp_spline(x, x**3, k=3)
        spl_y = make_interp_spline(y, y**2 + 2*y, k=3)
        cc = spl_x.c[:, None] * spl_y.c[None, :]
        bspl = NdBSpline((spl_x.t, spl_y.t), cc, (spl_x.k, spl_y.k))

        values = (x**3)[:, None] * (y**2 + 2*y)[None, :]
        rgi = RegularGridInterpolator((x, y), values)

        xi = [(a, b) for a, b in itertools.product(x, y)]
        bxi = bspl(xi)

        assert not np.isnan(bxi).any()
        assert_allclose(bxi, rgi(xi), atol=1e-14)
        assert_allclose(bxi.reshape(values.shape), values, atol=1e-14)

    def make_3d_case(self):
        # make a 3D separable spline
        x = np.arange(6)
        y = x**3
        spl = make_interp_spline(x, y, k=3)

        y_1 = x**3 + 2*x
        spl_1 = make_interp_spline(x, y_1, k=3)

        y_2 = x**3 + 3*x + 1
        spl_2 = make_interp_spline(x, y_2, k=3)

        t2 = (spl.t, spl_1.t, spl_2.t)
        c2 = (spl.c[:, None, None] *
              spl_1.c[None, :, None] *
              spl_2.c[None, None, :])

        return t2, c2, 3

    def test_3D_separable(self):
        rng = np.random.default_rng(12345)
        x, y, z = rng.uniform(size=(3, 11)) * 5
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        xi = [_ for _ in zip(x, y, z)]
        result = bspl3(xi)
        assert result.shape == (11,)
        assert_allclose(result, target, atol=1e-14)

    def test_3D_derivative(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)
        rng = np.random.default_rng(12345)
        x, y, z = rng.uniform(size=(3, 11)) * 5
        xi = [_ for _ in zip(x, y, z)]

        assert_allclose(bspl3(xi, nu=(1, 0, 0)),
                        3*x**2 * (y**3 + 2*y) * (z**3 + 3*z + 1), atol=1e-14)

        assert_allclose(bspl3(xi, nu=(2, 0, 0)),
                        6*x * (y**3 + 2*y) * (z**3 + 3*z + 1), atol=1e-14)

        assert_allclose(bspl3(xi, nu=(2, 1, 0)),
                        6*x * (3*y**2 + 2) * (z**3 + 3*z + 1), atol=1e-14)

        assert_allclose(bspl3(xi, nu=(2, 1, 3)),
                        6*x * (3*y**2 + 2) * (6), atol=1e-14)

        assert_allclose(bspl3(xi, nu=(2, 1, 4)),
                        np.zeros(len(xi)), atol=1e-14)

    def test_3D_random(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1))

        spl = NdBSpline((tx, ty, tz), c, k=k)
        spl_0 = NdBSpline0((tx, ty, tz), c, k=k)

        xi = (1., 1., 1)
        assert_allclose(spl(xi), spl_0(xi), atol=1e-14)

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        assert_allclose(spl(xi), [spl_0(xp) for xp in xi], atol=1e-14)

    def test_3D_random_complex(self):
        rng = np.random.default_rng(12345)
        k = 3
        tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
        ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
        c = (rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1)) +
             rng.uniform(size=(tx.size-k-1, ty.size-k-1, tz.size-k-1))*1j)

        spl = NdBSpline((tx, ty, tz), c, k=k)
        spl_re = NdBSpline((tx, ty, tz), c.real, k=k)
        spl_im = NdBSpline((tx, ty, tz), c.imag, k=k)

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        assert_allclose(spl(xi),
                        spl_re(xi) + 1j*spl_im(xi), atol=1e-14)

    @pytest.mark.parametrize('cls_extrap', [None, True])
    @pytest.mark.parametrize('call_extrap', [None, True])
    def test_extrapolate_3D_separable(self, cls_extrap, call_extrap):
        # test that extrapolate=True does extrapolate
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi, extrapolate=call_extrap)
        assert_allclose(result, target, atol=1e-14)

    @pytest.mark.parametrize('extrap', [(False, True), (True, None)])
    def test_extrapolate_3D_separable_2(self, extrap):
        # test that call(..., extrapolate=None) defers to self.extrapolate,
        # otherwise supersedes self.extrapolate
        t3, c3, k = self.make_3d_case()
        cls_extrap, call_extrap = extrap
        bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi, extrapolate=call_extrap)
        assert_allclose(result, target, atol=1e-14)

    def test_extrapolate_false_3D_separable(self):
        # test that extrapolate=False produces nans for out-of-bounds values
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        x, y, z = [-2, 1, 7], [-3, 0.5, 6.5], [-1, 1.5, 7.5]
        x, y, z = map(np.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi, extrapolate=False)
        assert np.isnan(result[0])
        assert np.isnan(result[-1])
        assert_allclose(result[1:-1], target[1:-1], atol=1e-14)

    def test_x_nan_3D(self):
        # test that spline(nan) is nan
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        x = np.asarray([-2, 3, np.nan, 1, 2, 7, np.nan])
        y = np.asarray([-3, 3.5, 1, np.nan, 3, 6.5, 6.5])
        z = np.asarray([-1, 3.5, 2, 3, np.nan, 7.5, 7.5])
        xi = [_ for _ in zip(x, y, z)]
        target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)
        mask = np.isnan(x) | np.isnan(y) | np.isnan(z)
        target[mask] = np.nan

        result = bspl3(xi)
        assert np.isnan(result[mask]).all()
        assert_allclose(result, target, atol=1e-14)

    def test_non_c_contiguous(self):
        # check that non C-contiguous inputs are OK
        rng = np.random.default_rng(12345)
        kx, ky = 3, 3
        tx = np.sort(rng.uniform(low=0, high=4, size=16))
        tx = np.r_[(tx[0],)*kx, tx, (tx[-1],)*kx]
        ty = np.sort(rng.uniform(low=0, high=4, size=16))
        ty = np.r_[(ty[0],)*ky, ty, (ty[-1],)*ky]

        assert not tx[::2].flags.c_contiguous
        assert not ty[::2].flags.c_contiguous

        c = rng.uniform(size=(tx.size//2 - kx - 1, ty.size//2 - ky - 1))
        c = c.T
        assert not c.flags.c_contiguous

        xi = np.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        bspl2 = NdBSpline((tx[::2], ty[::2]), c, k=(kx, ky))
        bspl2_0 = NdBSpline0((tx[::2], ty[::2]), c, k=(kx, ky))

        assert_allclose(bspl2(xi),
                        [bspl2_0(xp) for xp in xi], atol=1e-14)

    def test_readonly(self):
        t3, c3, k = self.make_3d_case()
        bspl3 = NdBSpline(t3, c3, k=3)

        for i in range(3):
            t3[i].flags.writeable = False
        c3.flags.writeable = False

        bspl3_ = NdBSpline(t3, c3, k=3)

        assert bspl3((1, 2, 3)) == bspl3_((1, 2, 3))
