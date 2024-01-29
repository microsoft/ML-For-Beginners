# mypy: disable-error-code="attr-defined"
import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
                           assert_, suppress_warnings)
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num

from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
                             cumulative_trapezoid, cumtrapz, trapz, trapezoid,
                             quad, simpson, simps, fixed_quad, AccuracyWarning,
                             qmc_quad, cumulative_simpson)
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
                                      _EVALUEERR, _ECALLBACK, _EINPROGRESS)

class TestFixedQuad:
    def test_scalar(self):
        n = 4
        expected = 1/(2*n)
        got, _ = fixed_quad(lambda x: x**(2*n - 1), 0, 1, n=n)
        # quadrature exact for this input
        assert_allclose(got, expected, rtol=1e-12)

    def test_vector(self):
        n = 4
        p = np.arange(1, 2*n)
        expected = 1/(p + 1)
        got, _ = fixed_quad(lambda x: x**p[:, None], 0, 1, n=n)
        assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestQuadrature:
    def quad(self, x, a, b, args):
        raise NotImplementedError

    def test_quadrature(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8))
        table_val = 0.30614353532540296487
        assert_almost_equal(val, table_val, decimal=7)

    def test_quadrature_rtol(self):
        def myfunc(x, n, z):       # Bessel function integrand
            return 1e90 * cos(n*x-z*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8), rtol=1e-10)
        table_val = 1e90 * 0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_quadrature_miniter(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        table_val = 0.30614353532540296487
        for miniter in [5, 52]:
            val, err = quadrature(myfunc, 0, pi, (2, 1.8), miniter=miniter)
            assert_almost_equal(val, table_val, decimal=7)
            assert_(err < 1.0)

    def test_quadrature_single_args(self):
        def myfunc(x, n):
            return 1e90 * cos(n*x-1.8*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, args=2, rtol=1e-10)
        table_val = 1e90 * 0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romberg(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8))
        table_val = 0.30614353532540296487
        assert_almost_equal(val, table_val, decimal=7)

    def test_romberg_rtol(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return 1e19*cos(n*x-z*sin(x))/pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8), rtol=1e-10)
        table_val = 1e19*0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romb(self):
        assert_equal(romb(np.arange(17)), 128)

    def test_romb_gh_3731(self):
        # Check that romb makes maximal use of data points
        x = np.arange(2**4+1)
        y = np.cos(0.2*x)
        val = romb(y)
        val2, err = quad(lambda x: np.cos(0.2*x), x.min(), x.max())
        assert_allclose(val, val2, rtol=1e-8, atol=0)

        # should be equal to romb with 2**k+1 samples
        with suppress_warnings() as sup:
            sup.filter(AccuracyWarning, "divmax .4. exceeded")
            val3 = romberg(lambda x: np.cos(0.2*x), x.min(), x.max(), divmax=4)
        assert_allclose(val, val3, rtol=1e-12, atol=0)

    def test_non_dtype(self):
        # Check that we work fine with functions returning float
        import math
        valmath = romberg(math.sin, 0, 1)
        expected_val = 0.45969769413185085
        assert_almost_equal(valmath, expected_val, decimal=7)

    def test_newton_cotes(self):
        """Test the first few degrees, for evenly spaced points."""
        n = 1
        wts, errcoff = newton_cotes(n, 1)
        assert_equal(wts, n*np.array([0.5, 0.5]))
        assert_almost_equal(errcoff, -n**3/12.0)

        n = 2
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 4.0, 1.0])/6.0)
        assert_almost_equal(errcoff, -n**5/2880.0)

        n = 3
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 3.0, 3.0, 1.0])/8.0)
        assert_almost_equal(errcoff, -n**5/6480.0)

        n = 4
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([7.0, 32.0, 12.0, 32.0, 7.0])/90.0)
        assert_almost_equal(errcoff, -n**7/1935360.0)

    def test_newton_cotes2(self):
        """Test newton_cotes with points that are not evenly spaced."""

        x = np.array([0.0, 1.5, 2.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 8.0/3
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

        x = np.array([0.0, 1.4, 2.1, 3.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 9.0
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

    # ignore the DeprecationWarning emitted by the even kwd
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_simpson(self):
        y = np.arange(17)
        assert_equal(simpson(y), 128)
        assert_equal(simpson(y, dx=0.5), 64)
        assert_equal(simpson(y, x=np.linspace(0, 4, 17)), 32)

        y = np.arange(4)
        x = 2**y
        assert_equal(simpson(y, x=x, even='avg'), 13.875)
        assert_equal(simpson(y, x=x, even='first'), 13.75)
        assert_equal(simpson(y, x=x, even='last'), 14)

        # `even='simpson'`
        # integral should be exactly 21
        x = np.linspace(1, 4, 4)
        def f(x):
            return x**2

        assert_allclose(simpson(f(x), x=x, even='simpson'), 21.0)
        assert_allclose(simpson(f(x), x=x, even='avg'), 21 + 1/6)

        # integral should be exactly 114
        x = np.linspace(1, 7, 4)
        assert_allclose(simpson(f(x), dx=2.0, even='simpson'), 114)
        assert_allclose(simpson(f(x), dx=2.0, even='avg'), 115 + 1/3)

        # `even='simpson'`, test multi-axis behaviour
        a = np.arange(16).reshape(4, 4)
        x = np.arange(64.).reshape(4, 4, 4)
        y = f(x)
        for i in range(3):
            r = simpson(y, x=x, even='simpson', axis=i)
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                integral = x[tuple(idx)][-1]**3 / 3 - x[tuple(idx)][0]**3 / 3
                assert_allclose(r[it.multi_index], integral)

        # test when integration axis only has two points
        x = np.arange(16).reshape(8, 2)
        y = f(x)
        for even in ['simpson', 'avg', 'first', 'last']:
            r = simpson(y, x=x, even=even, axis=-1)

            integral = 0.5 * (y[:, 1] + y[:, 0]) * (x[:, 1] - x[:, 0])
            assert_allclose(r, integral)

        # odd points, test multi-axis behaviour
        a = np.arange(25).reshape(5, 5)
        x = np.arange(125).reshape(5, 5, 5)
        y = f(x)
        for i in range(3):
            r = simpson(y, x=x, axis=i)
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                integral = x[tuple(idx)][-1]**3 / 3 - x[tuple(idx)][0]**3 / 3
                assert_allclose(r[it.multi_index], integral)

        # Tests for checking base case
        x = np.array([3])
        y = np.power(x, 2)
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)

        x = np.array([3, 3, 3, 3])
        y = np.power(x, 2)
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)

        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]])
        y = np.power(x, 2)
        zero_axis = [0.0, 0.0, 0.0, 0.0]
        default_axis = [170 + 1/3] * 3   # 8**3 / 3 - 1/3
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        # the following should be exact for even='simpson'
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)

        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 8, 16, 32]])
        y = np.power(x, 2)
        zero_axis = [0.0, 136.0, 1088.0, 8704.0]
        default_axis = [170 + 1/3, 170 + 1/3, 32**3 / 3 - 1/3]
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)

    def test_simpson_deprecations(self):
        x = np.linspace(0, 3, 4)
        y = x**2
        with pytest.deprecated_call(match="The 'even' keyword is deprecated"):
            simpson(y, x=x, even='first')
        with pytest.deprecated_call(match="use keyword arguments"):
            simpson(y, x)

    @pytest.mark.parametrize('droplast', [False, True])
    def test_simpson_2d_integer_no_x(self, droplast):
        # The inputs are 2d integer arrays.  The results should be
        # identical to the results when the inputs are floating point.
        y = np.array([[2, 2, 4, 4, 8, 8, -4, 5],
                      [4, 4, 2, -4, 10, 22, -2, 10]])
        if droplast:
            y = y[:, :-1]
        result = simpson(y, axis=-1)
        expected = simpson(np.array(y, dtype=np.float64), axis=-1)
        assert_equal(result, expected)

    def test_simps(self):
        # Basic coverage test for the alias
        y = np.arange(5)
        x = 2**y
        with pytest.deprecated_call(match="simpson"):
            assert_allclose(
                simpson(y, x=x, dx=0.5),
                simps(y, x=x, dx=0.5)
            )


@pytest.mark.parametrize('func', [romberg, quadrature])
def test_deprecate_integrator(func):
    message = f"`scipy.integrate.{func.__name__}` is deprecated..."
    with pytest.deprecated_call(match=message):
        func(np.exp, 0, 1)


class TestCumulative_trapezoid:
    def test_1d(self):
        x = np.linspace(-2, 2, num=5)
        y = x
        y_int = cumulative_trapezoid(y, x, initial=0)
        y_expected = [0., -1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, x, initial=None)
        assert_allclose(y_int, y_expected[1:])

    def test_y_nd_x_nd(self):
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        y_int = cumulative_trapezoid(y, x, initial=0)
        y_expected = np.array([[[0., 0.5, 2., 4.5],
                                [0., 4.5, 10., 16.5]],
                               [[0., 8.5, 18., 28.5],
                                [0., 12.5, 26., 40.5]],
                               [[0., 16.5, 34., 52.5],
                                [0., 20.5, 42., 64.5]]])

        assert_allclose(y_int, y_expected)

        # Try with all axes
        shapes = [(2, 2, 4), (3, 1, 4), (3, 2, 3)]
        for axis, shape in zip([0, 1, 2], shapes):
            y_int = cumulative_trapezoid(y, x, initial=0, axis=axis)
            assert_equal(y_int.shape, (3, 2, 4))
            y_int = cumulative_trapezoid(y, x, initial=None, axis=axis)
            assert_equal(y_int.shape, shape)

    def test_y_nd_x_1d(self):
        y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        x = np.arange(4)**2
        # Try with all axes
        ys_expected = (
            np.array([[[4., 5., 6., 7.],
                       [8., 9., 10., 11.]],
                      [[40., 44., 48., 52.],
                       [56., 60., 64., 68.]]]),
            np.array([[[2., 3., 4., 5.]],
                      [[10., 11., 12., 13.]],
                      [[18., 19., 20., 21.]]]),
            np.array([[[0.5, 5., 17.5],
                       [4.5, 21., 53.5]],
                      [[8.5, 37., 89.5],
                       [12.5, 53., 125.5]],
                      [[16.5, 69., 161.5],
                       [20.5, 85., 197.5]]]))

        for axis, y_expected in zip([0, 1, 2], ys_expected):
            y_int = cumulative_trapezoid(y, x=x[:y.shape[axis]], axis=axis,
                                         initial=None)
            assert_allclose(y_int, y_expected)

    def test_x_none(self):
        y = np.linspace(-2, 2, num=5)

        y_int = cumulative_trapezoid(y)
        y_expected = [-1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, initial=0)
        y_expected = [0, -1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, dx=3)
        y_expected = [-4.5, -6., -4.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, dx=3, initial=0)
        y_expected = [0, -4.5, -6., -4.5, 0.]
        assert_allclose(y_int, y_expected)

    @pytest.mark.parametrize(
        "initial", [1, 0.5]
    )
    def test_initial_warning(self, initial):
        """If initial is not None or 0, a ValueError is raised."""
        y = np.linspace(0, 10, num=10)
        with pytest.deprecated_call(match="`initial`"):
            res = cumulative_trapezoid(y, initial=initial)
        assert_allclose(res, [initial, *np.cumsum(y[1:] + y[:-1])/2]) 

    def test_zero_len_y(self):
        with pytest.raises(ValueError, match="At least one point is required"):
            cumulative_trapezoid(y=[])

    def test_cumtrapz(self):
        # Basic coverage test for the alias
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        with pytest.deprecated_call(match="cumulative_trapezoid"):
            assert_allclose(cumulative_trapezoid(y, x, dx=0.5, axis=0, initial=0),
                            cumtrapz(y, x, dx=0.5, axis=0, initial=0),
                            rtol=1e-14)


class TestTrapezoid:
    def test_simple(self):
        x = np.arange(-10, 10, .1)
        r = trapezoid(np.exp(-.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
        # check integral of normal equals 1
        assert_allclose(r, 1)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None,:, None] + z[None, None,:]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # n-d `x`
        r = trapezoid(q, x=x[:, None, None], axis=0)
        assert_allclose(r, qx)
        r = trapezoid(q, x=y[None,:, None], axis=1)
        assert_allclose(r, qy)
        r = trapezoid(q, x=z[None, None,:], axis=2)
        assert_allclose(r, qz)

        # 1-d `x`
        r = trapezoid(q, x=x, axis=0)
        assert_allclose(r, qx)
        r = trapezoid(q, x=y, axis=1)
        assert_allclose(r, qy)
        r = trapezoid(q, x=z, axis=2)
        assert_allclose(r, qz)

    def test_masked(self):
        # Testing that masked arrays behave as if the function is 0 where
        # masked
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0  # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        assert_allclose(trapezoid(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        assert_allclose(trapezoid(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        assert_allclose(trapezoid(y, xm), r)

    def test_trapz_alias(self):
        # Basic coverage test for the alias
        y = np.arange(4)
        x = 2**y
        with pytest.deprecated_call(match="trapezoid"):
            assert_equal(trapezoid(y, x=x, dx=0.5, axis=0),
                         trapz(y, x=x, dx=0.5, axis=0))


class TestQMCQuad:
    def test_input_validation(self):
        message = "`func` must be callable."
        with pytest.raises(TypeError, match=message):
            qmc_quad("a duck", [0, 0], [1, 1])

        message = "`func` must evaluate the integrand at points..."
        with pytest.raises(ValueError, match=message):
            qmc_quad(lambda: 1, [0, 0], [1, 1])

        def func(x):
            assert x.ndim == 1
            return np.sum(x)
        message = "Exception encountered when attempting vectorized call..."
        with pytest.warns(UserWarning, match=message):
            qmc_quad(func, [0, 0], [1, 1])

        message = "`n_points` must be an integer."
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], n_points=1024.5)

        message = "`n_estimates` must be an integer."
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], n_estimates=8.5)

        message = "`qrng` must be an instance of scipy.stats.qmc.QMCEngine."
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], qrng="a duck")

        message = "`qrng` must be initialized with dimensionality equal to "
        with pytest.raises(ValueError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], qrng=stats.qmc.Sobol(1))

        message = r"`log` must be boolean \(`True` or `False`\)."
        with pytest.raises(TypeError, match=message):
            qmc_quad(lambda x: 1, [0, 0], [1, 1], log=10)

    def basic_test(self, n_points=2**8, n_estimates=8, signs=np.ones(2)):

        ndim = 2
        mean = np.zeros(ndim)
        cov = np.eye(ndim)

        def func(x):
            return stats.multivariate_normal.pdf(x.T, mean, cov)

        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        a = np.zeros(ndim)
        b = np.ones(ndim) * signs
        res = qmc_quad(func, a, b, n_points=n_points,
                       n_estimates=n_estimates, qrng=qrng)
        ref = stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        atol = sc.stdtrit(n_estimates-1, 0.995) * res.standard_error  # 99% CI
        assert_allclose(res.integral, ref, atol=atol)
        assert np.prod(signs)*res.integral > 0

        rng = np.random.default_rng(2879434385674690281)
        qrng = stats.qmc.Sobol(ndim, seed=rng)
        logres = qmc_quad(lambda *args: np.log(func(*args)), a, b,
                          n_points=n_points, n_estimates=n_estimates,
                          log=True, qrng=qrng)
        assert_allclose(np.exp(logres.integral), res.integral, rtol=1e-14)
        assert np.imag(logres.integral) == (np.pi if np.prod(signs) < 0 else 0)
        assert_allclose(np.exp(logres.standard_error),
                        res.standard_error, rtol=1e-14, atol=1e-16)

    @pytest.mark.parametrize("n_points", [2**8, 2**12])
    @pytest.mark.parametrize("n_estimates", [8, 16])
    def test_basic(self, n_points, n_estimates):
        self.basic_test(n_points, n_estimates)

    @pytest.mark.parametrize("signs", [[1, 1], [-1, -1], [-1, 1], [1, -1]])
    def test_sign(self, signs):
        self.basic_test(signs=signs)

    @pytest.mark.parametrize("log", [False, True])
    def test_zero(self, log):
        message = "A lower limit was equal to an upper limit, so"
        with pytest.warns(UserWarning, match=message):
            res = qmc_quad(lambda x: 1, [0, 0], [0, 1], log=log)
        assert res.integral == (-np.inf if log else 0)
        assert res.standard_error == 0

    def test_flexible_input(self):
        # check that qrng is not required
        # also checks that for 1d problems, a and b can be scalars
        def func(x):
            return stats.norm.pdf(x, scale=2)

        res = qmc_quad(func, 0, 1)
        ref = stats.norm.cdf(1, scale=2) - stats.norm.cdf(0, scale=2)
        assert_allclose(res.integral, ref, 1e-2)


def cumulative_simpson_nd_reference(y, *, x=None, dx=None, initial=None, axis=-1):
    # Use cumulative_trapezoid if length of y < 3
    if y.shape[axis] < 3:
        if initial is None:
            return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=None)
        else:
            return initial + cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=0)

    # Ensure that working axis is last axis
    y = np.moveaxis(y, axis, -1)
    x = np.moveaxis(x, axis, -1) if np.ndim(x) > 1 else x
    dx = np.moveaxis(dx, axis, -1) if np.ndim(dx) > 1 else dx
    initial = np.moveaxis(initial, axis, -1) if np.ndim(initial) > 1 else initial

    # If `x` is not present, create it from `dx`
    n = y.shape[-1]
    x = dx * np.arange(n) if dx is not None else x
    # Similarly, if `initial` is not present, set it to 0
    initial_was_none = initial is None
    initial = 0 if initial_was_none else initial

    # `np.apply_along_axis` accepts only one array, so concatenate arguments
    x = np.broadcast_to(x, y.shape)
    initial = np.broadcast_to(initial, y.shape[:-1] + (1,))
    z = np.concatenate((y, x, initial), axis=-1)

    # Use `np.apply_along_axis` to compute result
    def f(z):
        return cumulative_simpson(z[:n], x=z[n:2*n], initial=z[2*n:])
    res = np.apply_along_axis(f, -1, z)

    # Remove `initial` and undo axis move as needed
    res = res[..., 1:] if initial_was_none else res
    res = np.moveaxis(res, -1, axis)
    return res


class TestCumulativeSimpson:
    x0 = np.arange(4)
    y0 = x0**2

    @pytest.mark.parametrize('use_dx', (False, True))
    @pytest.mark.parametrize('use_initial', (False, True))
    def test_1d(self, use_dx, use_initial):
        # Test for exact agreement with polynomial of highest
        # possible order (3 if `dx` is constant, 2 otherwise).
        rng = np.random.default_rng(82456839535679456794)
        n = 10

        # Generate random polynomials and ground truth
        # integral of appropriate order
        order = 3 if use_dx else 2
        dx = rng.random()
        x = (np.sort(rng.random(n)) if order == 2
             else np.arange(n)*dx + rng.random())
        i = np.arange(order + 1)[:, np.newaxis]
        c = rng.random(order + 1)[:, np.newaxis]
        y = np.sum(c*x**i, axis=0)
        Y = np.sum(c*x**(i + 1)/(i + 1), axis=0)
        ref = Y if use_initial else (Y-Y[0])[1:]

        # Integrate with `cumulative_simpson`
        initial = Y[0] if use_initial else None
        kwarg = {'dx': dx} if use_dx else {'x': x}
        res = cumulative_simpson(y, **kwarg, initial=initial)

        # Compare result against reference
        if not use_dx:
            assert_allclose(res, ref, rtol=2e-15)
        else:
            i0 = 0 if use_initial else 1
            # all terms are "close"
            assert_allclose(res, ref, rtol=0.0025)
            # only even-interval terms are "exact"
            assert_allclose(res[i0::2], ref[i0::2], rtol=2e-15)

    @pytest.mark.parametrize('axis', np.arange(-3, 3))
    @pytest.mark.parametrize('x_ndim', (1, 3))
    @pytest.mark.parametrize('x_len', (1, 2, 7))
    @pytest.mark.parametrize('i_ndim', (None, 0, 3,))
    @pytest.mark.parametrize('dx', (None, True))
    def test_nd(self, axis, x_ndim, x_len, i_ndim, dx):
        # Test behavior of `cumulative_simpson` with N-D `y`
        rng = np.random.default_rng(82456839535679456794)

        # determine shapes
        shape = [5, 6, x_len]
        shape[axis], shape[-1] = shape[-1], shape[axis]
        shape_len_1 = shape.copy()
        shape_len_1[axis] = 1
        i_shape = shape_len_1 if i_ndim == 3 else ()

        # initialize arguments
        y = rng.random(size=shape)
        x, dx = None, None
        if dx:
            dx = rng.random(size=shape_len_1) if x_ndim > 1 else rng.random()
        else:
            x = (np.sort(rng.random(size=shape), axis=axis) if x_ndim > 1
                 else np.sort(rng.random(size=shape[axis])))
        initial = None if i_ndim is None else rng.random(size=i_shape)

        # compare results
        res = cumulative_simpson(y, x=x, dx=dx, initial=initial, axis=axis)
        ref = cumulative_simpson_nd_reference(y, x=x, dx=dx, initial=initial, axis=axis)
        np.testing.assert_allclose(res, ref, rtol=1e-15)

    @pytest.mark.parametrize(('message', 'kwarg_update'), [
        ("x must be strictly increasing", dict(x=[2, 2, 3, 4])),
        ("x must be strictly increasing", dict(x=[x0, [2, 2, 4, 8]], y=[y0, y0])),
        ("x must be strictly increasing", dict(x=[x0, x0, x0], y=[y0, y0, y0], axis=0)),
        ("At least one point is required", dict(x=[], y=[])),
        ("`axis=4` is not valid for `y` with `y.ndim=1`", dict(axis=4)),
        ("shape of `x` must be the same as `y` or 1-D", dict(x=np.arange(5))),
        ("`initial` must either be a scalar or...", dict(initial=np.arange(5))),
        ("`dx` must either be a scalar or...", dict(x=None, dx=np.arange(5))),
    ])
    def test_simpson_exceptions(self, message, kwarg_update):
        kwargs0 = dict(y=self.y0, x=self.x0, dx=None, initial=None, axis=-1)
        with pytest.raises(ValueError, match=message):
            cumulative_simpson(**dict(kwargs0, **kwarg_update))

    def test_special_cases(self):
        # Test special cases not checked elsewhere
        rng = np.random.default_rng(82456839535679456794)
        y = rng.random(size=10)
        res = cumulative_simpson(y, dx=0)
        assert_equal(res, 0)

        # Should add tests of:
        # - all elements of `x` identical
        # These should work as they do for `simpson`

    def _get_theoretical_diff_between_simps_and_cum_simps(self, y, x):
        """`cumulative_simpson` and `simpson` can be tested against other to verify
        they give consistent results. `simpson` will iteratively be called with 
        successively higher upper limits of integration. This function calculates
        the theoretical correction required to `simpson` at even intervals to match
        with `cumulative_simpson`.
        """
        d = np.diff(x, axis=-1)
        sub_integrals_h1 = _cumulative_simpson_unequal_intervals(y, d)
        sub_integrals_h2 = _cumulative_simpson_unequal_intervals(
            y[..., ::-1], d[..., ::-1]
        )[..., ::-1]

        # Concatenate to build difference array
        zeros_shape = (*y.shape[:-1], 1)
        theoretical_difference = np.concatenate(
            [
                np.zeros(zeros_shape),
                (sub_integrals_h1[..., 1:] - sub_integrals_h2[..., :-1]),
                np.zeros(zeros_shape),
            ],
            axis=-1,
        )
        # Differences only expected at even intervals. Odd intervals will
        # match exactly so there is no correction
        theoretical_difference[..., 1::2] = 0.0
        # Note: the first interval will not match from this correction as 
        # `simpson` uses the trapezoidal rule
        return theoretical_difference

    @given(
        y=hyp_num.arrays(
            np.float64, 
            hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10),
            elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-7)
        )
    )
    def test_cumulative_simpson_against_simpson_with_default_dx(
        self, y
    ):
        """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""
        def simpson_reference(y):
            return np.stack(
                [simpson(y[..., :i], dx=1.0) for i in range(2, y.shape[-1]+1)], axis=-1,
            )

        res = cumulative_simpson(y, dx=1.0)
        ref = simpson_reference(y)
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(
            y, x=np.arange(y.shape[-1])
        )
        np.testing.assert_allclose(
            res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:]
        )

    
    @given(
        y=hyp_num.arrays(
            np.float64, 
            hyp_num.array_shapes(max_dims=4, min_side=3, max_side=10),
            elements=st.floats(-10, 10, allow_nan=False).filter(lambda x: abs(x) > 1e-7)
        )
    )
    def test_cumulative_simpson_against_simpson(
        self, y
    ):
        """Theoretically, the output of `cumulative_simpson` will be identical
        to `simpson` at all even indices and in the last index. The first index
        will not match as `simpson` uses the trapezoidal rule when there are only two
        data points. Odd indices after the first index are shown to match with
        a mathematically-derived correction."""
        interval = 10/(y.shape[-1] - 1)
        x = np.linspace(0, 10, num=y.shape[-1])
        x[1:] = x[1:] + 0.2*interval*np.random.uniform(-1, 1, len(x) - 1)
        
        def simpson_reference(y, x):
            return np.stack(
                [simpson(y[..., :i], x=x[..., :i]) for i in range(2, y.shape[-1]+1)],
                axis=-1,
            )

        res = cumulative_simpson(y, x=x)
        ref = simpson_reference(y, x)
        theoretical_difference = self._get_theoretical_diff_between_simps_and_cum_simps(
            y, x
        )
        np.testing.assert_allclose(
            res[..., 1:], ref[..., 1:] + theoretical_difference[..., 1:]
        )


class TestTanhSinh:

    # Test problems from [1] Section 6
    def f1(self, t):
        return t * np.log(1 + t)

    f1.ref = 0.25
    f1.b = 1

    def f2(self, t):
        return t ** 2 * np.arctan(t)

    f2.ref = (np.pi - 2 + 2 * np.log(2)) / 12
    f2.b = 1

    def f3(self, t):
        return np.exp(t) * np.cos(t)

    f3.ref = (np.exp(np.pi / 2) - 1) / 2
    f3.b = np.pi / 2

    def f4(self, t):
        a = np.sqrt(2 + t ** 2)
        return np.arctan(a) / ((1 + t ** 2) * a)

    f4.ref = 5 * np.pi ** 2 / 96
    f4.b = 1

    def f5(self, t):
        return np.sqrt(t) * np.log(t)

    f5.ref = -4 / 9
    f5.b = 1

    def f6(self, t):
        return np.sqrt(1 - t ** 2)

    f6.ref = np.pi / 4
    f6.b = 1

    def f7(self, t):
        return np.sqrt(t) / np.sqrt(1 - t ** 2)

    f7.ref = 2 * np.sqrt(np.pi) * sc.gamma(3 / 4) / sc.gamma(1 / 4)
    f7.b = 1

    def f8(self, t):
        return np.log(t) ** 2

    f8.ref = 2
    f8.b = 1

    def f9(self, t):
        return np.log(np.cos(t))

    f9.ref = -np.pi * np.log(2) / 2
    f9.b = np.pi / 2

    def f10(self, t):
        return np.sqrt(np.tan(t))

    f10.ref = np.pi * np.sqrt(2) / 2
    f10.b = np.pi / 2

    def f11(self, t):
        return 1 / (1 + t ** 2)

    f11.ref = np.pi / 2
    f11.b = np.inf

    def f12(self, t):
        return np.exp(-t) / np.sqrt(t)

    f12.ref = np.sqrt(np.pi)
    f12.b = np.inf

    def f13(self, t):
        return np.exp(-t ** 2 / 2)

    f13.ref = np.sqrt(np.pi / 2)
    f13.b = np.inf

    def f14(self, t):
        return np.exp(-t) * np.cos(t)

    f14.ref = 0.5
    f14.b = np.inf

    def f15(self, t):
        return np.sin(t) / t

    f15.ref = np.pi / 2
    f15.b = np.inf

    def error(self, res, ref, log=False):
        err = abs(res - ref)

        if not log:
            return err

        with np.errstate(divide='ignore'):
            return np.log10(err)

    def test_input_validation(self):
        f = self.f1

        message = '`f` must be callable.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(42, 0, f.b)

        message = '...must be True or False.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, log=2)

        message = '...must be real numbers.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 1+1j, f.b)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol='ekki')
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=pytest)

        message = '...must be non-negative and finite.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf)

        message = '...may not be positive infinity.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=np.inf, log=True)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf, log=True)

        message = '...must be integers.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=object())
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=1+1j)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel="migratory coconut")

        message = '...must be non-negative.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel=-1)

        message = '...must be callable.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, callback='elderberry')

    @pytest.mark.parametrize("limits, ref", [
        [(0, np.inf), 0.5],  # b infinite
        [(-np.inf, 0), 0.5],  # a infinite
        [(-np.inf, np.inf), 1],  # a and b infinite
        [(np.inf, -np.inf), -1],  # flipped limits
        [(1, -1), stats.norm.cdf(-1) -  stats.norm.cdf(1)],  # flipped limits
    ])
    def test_integral_transforms(self, limits, ref):
        # Check that the integral transforms are behaving for both normal and
        # log integration
        dist = stats.norm()

        res = _tanhsinh(dist.pdf, *limits)
        assert_allclose(res.integral, ref)

        logres = _tanhsinh(dist.logpdf, *limits, log=True)
        assert_allclose(np.exp(logres.integral), ref)
        # Transformation should not make the result complex unnecessarily
        assert (np.issubdtype(logres.integral.dtype, np.floating) if ref > 0
                else np.issubdtype(logres.integral.dtype, np.complexfloating))

        assert_allclose(np.exp(logres.error), res.error, atol=1e-16)

    # 15 skipped intentionally; it's very difficult numerically
    @pytest.mark.parametrize('f_number', range(1, 15))
    def test_basic(self, f_number):
        f = getattr(self, f"f{f_number}")
        rtol = 2e-8
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert_allclose(res.integral, f.ref, rtol=rtol)
        if f_number not in {14}:  # mildly underestimates error here
            true_error = abs(self.error(res.integral, f.ref)/res.integral)
            assert true_error < res.error

        if f_number in {7, 10, 12}:  # succeeds, but doesn't know it
            return

        assert res.success
        assert res.status == 0

    @pytest.mark.parametrize('ref', (0.5, [0.4, 0.6]))
    @pytest.mark.parametrize('case', stats._distr_params.distcont)
    def test_accuracy(self, ref, case):
        distname, params = case
        if distname in {'dgamma', 'dweibull', 'laplace', 'kstwo'}:
            # should split up interval at first-derivative discontinuity
            pytest.skip('tanh-sinh is not great for non-smooth integrands')
        dist = getattr(stats, distname)(*params)
        x = dist.interval(ref)
        res = _tanhsinh(dist.pdf, *x)
        assert_allclose(res.integral, ref)

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        # Test for correct functionality, output shapes, and dtypes for various
        # input shapes.
        rng = np.random.default_rng(82456839535679456794)
        a = rng.random(shape)
        b = rng.random(shape)
        p = rng.random(shape)
        n = np.prod(shape)

        def f(x, p):
            f.ncall += 1
            f.feval += 1 if (x.size == n or x.ndim <=1) else x.shape[-1]
            return x**p
        f.ncall = 0
        f.feval = 0

        @np.vectorize
        def _tanhsinh_single(a, b, p):
            return _tanhsinh(lambda x: x**p, a, b)

        res = _tanhsinh(f, a, b, args=(p,))
        refs = _tanhsinh_single(a, b, p).ravel()

        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            res_attr = getattr(res, attr)
            assert_allclose(res_attr.ravel(), ref_attr, rtol=1e-15)
            assert_equal(res_attr.shape, shape)

        assert np.issubdtype(res.success.dtype, np.bool_)
        assert np.issubdtype(res.status.dtype, np.integer)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        assert np.issubdtype(res.maxlevel.dtype, np.integer)
        assert_equal(np.max(res.nfev), f.feval)
        # maxlevel = 2 -> 3 function calls (2 initialization, 1 work)
        assert np.max(res.maxlevel) >= 2
        assert_equal(np.max(res.maxlevel), f.ncall)

    def test_flags(self):
        # Test cases that should produce different status flags; show that all
        # can be produced simultaneously.
        def f(xs, js):
            f.nit += 1
            funcs = [lambda x: np.exp(-x**2),  # converges
                     lambda x: np.exp(x),  # reaches maxiter due to order=2
                     lambda x: np.full_like(x, np.nan)[()]]  # stops due to NaN
            res = [funcs[j](x) for x, j in zip(xs, js.ravel())]
            return res
        f.nit = 0

        args = (np.arange(3, dtype=np.int64),)
        res = _tanhsinh(f, [np.inf]*3, [-np.inf]*3, maxlevel=5, args=args)
        ref_flags = np.array([0, -2, -3])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        # demonstrate that number of accurate digits doubles each iteration
        f = self.f1
        last_logerr = 0
        for i in range(4):
            res = _tanhsinh(f, 0, f.b, minlevel=0, maxlevel=i)
            logerr = self.error(res.integral, f.ref, log=True)
            assert (logerr < last_logerr * 2 or logerr < -15.5)
            last_logerr = logerr

    def test_options_and_result_attributes(self):
        # demonstrate that options are behaving as advertised and status
        # messages are as intended
        def f(x):
            f.calls += 1
            f.feval += np.size(x)
            return self.f2(x)
        f.ref = self.f2.ref
        f.b = self.f2.b
        default_rtol = 1e-12
        default_atol = f.ref * default_rtol  # effective default absolute tol

        # Test default options
        f.feval, f.calls = 0, 0
        ref = _tanhsinh(f, 0, f.b)
        assert self.error(ref.integral, f.ref) < ref.error < default_atol
        assert ref.nfev == f.feval
        ref.calls = f.calls  # reference number of function calls
        assert ref.success
        assert ref.status == 0

        # Test `maxlevel` equal to required max level
        # We should get all the same results
        f.feval, f.calls = 0, 0
        maxlevel = ref.maxlevel
        res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
        res.calls = f.calls
        assert res == ref

        # Now reduce the maximum level. We won't meet tolerances.
        f.feval, f.calls = 0, 0
        maxlevel -= 1
        assert maxlevel >= 2  # can't compare errors otherwise
        res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
        assert self.error(res.integral, f.ref) < res.error > default_atol
        assert res.nfev == f.feval < ref.nfev
        assert f.calls == ref.calls - 1
        assert not res.success
        assert res.status == _ECONVERR

        # `maxfun` is currently not enforced

        # # Test `maxfun` equal to required number of function evaluations
        # # We should get all the same results
        # f.feval, f.calls = 0, 0
        # maxfun = ref.nfev
        # res = _tanhsinh(f, 0, f.b, maxfun = maxfun)
        # assert res == ref
        #
        # # Now reduce `maxfun`. We won't meet tolerances.
        # f.feval, f.calls = 0, 0
        # maxfun -= 1
        # res = _tanhsinh(f, 0, f.b, maxfun=maxfun)
        # assert self.error(res.integral, f.ref) < res.error > default_atol
        # assert res.nfev == f.feval < ref.nfev
        # assert f.calls == ref.calls - 1
        # assert not res.success
        # assert res.status == 2

        # Take this result to be the new reference
        ref = res
        ref.calls = f.calls

        # Test `atol`
        f.feval, f.calls = 0, 0
        # With this tolerance, we should get the exact same result as ref
        atol = np.nextafter(ref.error, np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
        assert res.integral == ref.integral
        assert res.error == ref.error
        assert res.nfev == f.feval == ref.nfev
        assert f.calls == ref.calls
        # Except the result is considered to be successful
        assert res.success
        assert res.status == 0

        f.feval, f.calls = 0, 0
        # With a tighter tolerance, we should get a more accurate result
        atol = np.nextafter(ref.error, -np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
        assert self.error(res.integral, f.ref) < res.error < atol
        assert res.nfev == f.feval > ref.nfev
        assert f.calls > ref.calls
        assert res.success
        assert res.status == 0

        # Test `rtol`
        f.feval, f.calls = 0, 0
        # With this tolerance, we should get the exact same result as ref
        rtol = np.nextafter(ref.error/ref.integral, np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert res.integral == ref.integral
        assert res.error == ref.error
        assert res.nfev == f.feval == ref.nfev
        assert f.calls == ref.calls
        # Except the result is considered to be successful
        assert res.success
        assert res.status == 0

        f.feval, f.calls = 0, 0
        # With a tighter tolerance, we should get a more accurate result
        rtol = np.nextafter(ref.error/ref.integral, -np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert self.error(res.integral, f.ref)/f.ref < res.error/res.integral < rtol
        assert res.nfev == f.feval > ref.nfev
        assert f.calls > ref.calls
        assert res.success
        assert res.status == 0

    @pytest.mark.parametrize('rtol', [1e-4, 1e-14])
    def test_log(self, rtol):
        # Test equivalence of log-integration and regular integration
        dist = stats.norm()

        test_tols = dict(atol=1e-18, rtol=1e-15)

        # Positive integrand (real log-integrand)
        res = _tanhsinh(dist.logpdf, -1, 2, log=True, rtol=np.log(rtol))
        ref = _tanhsinh(dist.pdf, -1, 2, rtol=rtol)
        assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
        assert_allclose(np.exp(res.error), ref.error, **test_tols)
        assert res.nfev == ref.nfev

        # Real integrand (complex log-integrand)
        def f(x):
            return -dist.logpdf(x)*dist.pdf(x)

        def logf(x):
            return np.log(dist.logpdf(x) + 0j) + dist.logpdf(x) + np.pi * 1j

        res = _tanhsinh(logf, -np.inf, np.inf, log=True)
        ref = _tanhsinh(f, -np.inf, np.inf)
        # In gh-19173, we saw `invalid` warnings on one CI platform.
        # Silencing `all` because I can't reproduce locally and don't want
        # to risk the need to run CI again.
        with np.errstate(all='ignore'):
            assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
            assert_allclose(np.exp(res.error), ref.error, **test_tols)
        assert res.nfev == ref.nfev

    def test_complex(self):
        # Test integration of complex integrand
        # Finite limits
        def f(x):
            return np.exp(1j * x)

        res = _tanhsinh(f, 0, np.pi/4)
        ref = np.sqrt(2)/2 + (1-np.sqrt(2)/2)*1j
        assert_allclose(res.integral, ref)

        # Infinite limits
        dist1 = stats.norm(scale=1)
        dist2 = stats.norm(scale=2)
        def f(x):
            return dist1.pdf(x) + 1j*dist2.pdf(x)

        res = _tanhsinh(f, np.inf, -np.inf)
        assert_allclose(res.integral, -(1+1j))

    @pytest.mark.parametrize("maxlevel", range(4))
    def test_minlevel(self, maxlevel):
        # Verify that minlevel does not change the values at which the
        # integrand is evaluated or the integral/error estimates, only the
        # number of function calls
        def f(x):
            f.calls += 1
            f.feval += np.size(x)
            f.x = np.concatenate((f.x, x.ravel()))
            return self.f2(x)
        f.feval, f.calls, f.x = 0, 0, np.array([])

        ref = _tanhsinh(f, 0, self.f2.b, minlevel=0, maxlevel=maxlevel)
        ref_x = np.sort(f.x)

        for minlevel in range(0, maxlevel + 1):
            f.feval, f.calls, f.x = 0, 0, np.array([])
            options = dict(minlevel=minlevel, maxlevel=maxlevel)
            res = _tanhsinh(f, 0, self.f2.b, **options)
            # Should be very close; all that has changed is the order of values
            assert_allclose(res.integral, ref.integral, rtol=4e-16)
            # Difference in absolute errors << magnitude of integral
            assert_allclose(res.error, ref.error, atol=4e-16 * ref.integral)
            assert res.nfev == f.feval == len(f.x)
            assert f.calls == maxlevel - minlevel + 1 + 1  # 1 validation call
            assert res.status == ref.status
            assert_equal(ref_x, np.sort(f.x))

    def test_improper_integrals(self):
        # Test handling of infinite limits of integration (mixed with finite limits)
        def f(x):
            return np.exp(-x**2)
        a = [-np.inf, 0, -np.inf, np.inf, -20, -np.inf, -20]
        b = [np.inf, np.inf, 0, -np.inf, 20, 20, np.inf]
        ref = np.sqrt(np.pi)
        res = _tanhsinh(f, a, b)
        assert_allclose(res.integral, [ref, ref/2, ref/2, -ref, ref, ref, ref])

    @pytest.mark.parametrize("limits", ((0, 3), ([-np.inf, 0], [3, 3])))
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_dtype(self, limits, dtype):
        # Test that dtypes are preserved
        a, b = np.asarray(limits, dtype=dtype)[()]

        def f(x):
            assert x.dtype == dtype
            return np.exp(x)

        rtol = 1e-12 if dtype == np.float64 else 1e-5
        res = _tanhsinh(f, a, b, rtol=rtol)
        assert res.integral.dtype == dtype
        assert res.error.dtype == dtype
        assert np.all(res.success)
        assert_allclose(res.integral, np.exp(b)-np.exp(a), rtol=rtol)

    def test_maxiter_callback(self):
        # Test behavior of `maxiter` parameter and `callback` interface
        a, b = -np.inf, np.inf
        def f(x):
            return np.exp(-x*x)

        minlevel, maxlevel = 0, 2
        maxiter = maxlevel - minlevel + 1
        kwargs = dict(minlevel=minlevel, maxlevel=maxlevel, rtol=1e-15)
        res = _tanhsinh(f, a, b, **kwargs)
        assert not res.success
        assert res.maxlevel == maxlevel

        def callback(res):
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'integral')
            assert res.status == 1
            if callback.iter == maxiter:
                raise StopIteration
        callback.iter = -1  # callback called once before first iteration
        callback.res = None

        del kwargs['maxlevel']
        res2 = _tanhsinh(f, a, b, **kwargs, callback=callback)
        # terminating with callback is identical to terminating due to maxiter
        # (except for `status`)
        for key in res.keys():
            if key == 'status':
                assert callback.res[key] == 1
                assert res[key] == -2
                assert res2[key] == -4
            else:
                assert res2[key] == callback.res[key] == res[key]

    def test_jumpstart(self):
        # The intermediate results at each level i should be the same as the
        # final results when jumpstarting at level i; i.e. minlevel=maxlevel=i
        a, b = -np.inf, np.inf
        def f(x):
            return np.exp(-x*x)

        def callback(res):
            callback.integrals.append(res.integral)
            callback.errors.append(res.error)
        callback.integrals = []
        callback.errors = []

        maxlevel = 4
        _tanhsinh(f, a, b, minlevel=0, maxlevel=maxlevel, callback=callback)

        integrals = []
        errors = []
        for i in range(maxlevel + 1):
            res = _tanhsinh(f, a, b, minlevel=i, maxlevel=i)
            integrals.append(res.integral)
            errors.append(res.error)

        assert_allclose(callback.integrals[1:], integrals, rtol=1e-15)
        assert_allclose(callback.errors[1:], errors, rtol=1e-15, atol=1e-16)

    def test_special_cases(self):
        # Test edge cases and other special cases

        # Test that integers are not passed to `f`
        # (otherwise this would overflow)
        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99

        res = _tanhsinh(f, 0, 1)
        assert res.success
        assert_allclose(res.integral, 1/100)

        # Test levels 0 and 1; error is NaN
        res = _tanhsinh(f, 0, 1, maxlevel=0)
        assert res.integral > 0
        assert_equal(res.error, np.nan)
        res = _tanhsinh(f, 0, 1, maxlevel=1)
        assert res.integral > 0
        assert_equal(res.error, np.nan)

        # Tes equal left and right integration limits
        res = _tanhsinh(f, 1, 1)
        assert res.success
        assert res.maxlevel == -1
        assert_allclose(res.integral, 0)

        # Test scalar `args` (not in tuple)
        def f(x, c):
            return x**c

        res = _tanhsinh(f, 0, 1, args=99)
        assert_allclose(res.integral, 1/100)

        # Test NaNs
        a = [np.nan, 0, 0, 0]
        b = [1, np.nan, 1, 1]
        c = [1, 1, np.nan, 1]
        res = _tanhsinh(f, a, b, args=(c,))
        assert_allclose(res.integral, [np.nan, np.nan, np.nan, 0.5])
        assert_allclose(res.error[:3], np.nan)
        assert_equal(res.status, [-3, -3, -3, 0])
        assert_equal(res.success, [False, False, False, True])
        assert_equal(res.nfev[:3], 1)

        # Test complex integral followed by real integral
        # Previously, h0 was of the result dtype. If the `dtype` were complex,
        # this could lead to complex cached abscissae/weights. If these get
        # cast to real dtype for a subsequent real integral, we would get a
        # ComplexWarning. Check that this is avoided.
        _pair_cache.xjc = np.empty(0)
        _pair_cache.wj = np.empty(0)
        _pair_cache.indices = [0]
        _pair_cache.h0 = None
        res = _tanhsinh(lambda x: x*1j, 0, 1)
        assert_allclose(res.integral, 0.5*1j)
        res = _tanhsinh(lambda x: x, 0, 1)
        assert_allclose(res.integral, 0.5)

        # Test zero-size
        shape = (0, 3)
        res = _tanhsinh(lambda x: x, 0, np.zeros(shape))
        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            assert_equal(res[attr].shape, shape)
