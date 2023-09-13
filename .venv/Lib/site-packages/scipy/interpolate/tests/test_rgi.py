import itertools

import pytest
import numpy as np

from numpy.testing import (assert_allclose, assert_equal, assert_warns,
                           assert_array_almost_equal, assert_array_equal)
from pytest import raises as assert_raises

from scipy.interpolate import (RegularGridInterpolator, interpn,
                               RectBivariateSpline,
                               NearestNDInterpolator, LinearNDInterpolator)

from scipy.sparse._sputils import matrix

parametrize_rgi_interp_methods = pytest.mark.parametrize(
    "method", ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip']
)

class TestRegularGridInterpolator:
    def _get_sample_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 4
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_2(self):
        # create another 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_3(self):
        # create another 4-D grid of 7 points in each dimension
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)] * 4
        values = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_4(self):
        # create another 4-D grid of 2 points in each dimension
        points = [(0.0, 1.0)] * 4
        values = np.asarray([0.0, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    @parametrize_rgi_interp_methods
    def test_list_input(self, method):
        points, values = self._get_sample_4d_3()

        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        interp = RegularGridInterpolator(points,
                                         values.tolist(),
                                         method=method)
        v1 = interp(sample.tolist())
        interp = RegularGridInterpolator(points,
                                         values,
                                         method=method)
        v2 = interp(sample)
        assert_allclose(v1, v2)

    @pytest.mark.parametrize('method', ['cubic', 'quintic', 'pchip'])
    def test_spline_dim_error(self, method):
        points, values = self._get_sample_4d_4()
        match = "points in dimension"

        # Check error raise when creating interpolator
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values, method=method)

        # Check error raise when creating interpolator
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        with pytest.raises(ValueError, match=match):
            interp(sample, method=method)

    @pytest.mark.parametrize(
        "points_values, sample",
        [
            (
                _get_sample_4d,
                np.asarray(
                    [[0.1, 0.1, 1.0, 0.9],
                     [0.2, 0.1, 0.45, 0.8],
                     [0.5, 0.5, 0.5, 0.5]]
                ),
            ),
            (_get_sample_4d_2, np.asarray([0.1, 0.1, 10.0, 9.0])),
        ],
    )
    def test_linear_and_slinear_close(self, points_values, sample):
        points, values = points_values(self)
        interp = RegularGridInterpolator(points, values, method="linear")
        v1 = interp(sample)
        interp = RegularGridInterpolator(points, values, method="slinear")
        v2 = interp(sample)
        assert_allclose(v1, v2)

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        points, values = self._get_sample_4d_3()
        values = values - 2j*values
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        interp = RegularGridInterpolator(points, values, method=method)
        rinterp = RegularGridInterpolator(points, values.real, method=method)
        iinterp = RegularGridInterpolator(points, values.imag, method=method)

        v1 = interp(sample)
        v2 = rinterp(sample) + 1j*iinterp(sample)
        assert_allclose(v1, v2)

    def test_cubic_vs_pchip(self):
        x, y = [1, 2, 3, 4], [1, 2, 3, 4]
        xg, yg = np.meshgrid(x, y, indexing='ij')

        values = (lambda x, y: x**4 * y**4)(xg, yg)
        cubic = RegularGridInterpolator((x, y), values, method='cubic')
        pchip = RegularGridInterpolator((x, y), values, method='pchip')

        vals_cubic = cubic([1.5, 2])
        vals_pchip = pchip([1.5, 2])
        assert not np.allclose(vals_cubic, vals_pchip, atol=1e-14, rtol=0)

    def test_linear_xi1d(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([0.1, 0.1, 10., 9.])
        wanted = 1001.1
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_xi3d(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    @pytest.mark.parametrize(
        "sample, wanted",
        [
            (np.asarray([0.1, 0.1, 0.9, 0.9]), 1100.0),
            (np.asarray([0.1, 0.1, 0.1, 0.1]), 0.0),
            (np.asarray([0.0, 0.0, 0.0, 0.0]), 0.0),
            (np.asarray([1.0, 1.0, 1.0, 1.0]), 1111.0),
            (np.asarray([0.1, 0.4, 0.6, 0.9]), 1055.0),
        ],
    )
    def test_nearest(self, sample, wanted):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_edges(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        wanted = np.asarray([0., 1111.])
        assert_array_almost_equal(interp(sample), wanted)

    def test_valid_create(self):
        # create a 2-D grid of 3 points in each dimension
        points = [(0., .5, 1.), (0., 1., .5)]
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis]
        values1 = values[np.newaxis, :]
        values = (values0 + values1 * 10)
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [((0., .5, 1.), ), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, .75, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values,
                      method="undefmethod")

    def test_valid_call(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        assert_raises(ValueError, interp, sample, "undefmethod")
        sample = np.asarray([[0., 0., 0.], [1., 1., 1.]])
        assert_raises(ValueError, interp, sample)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.1]])
        assert_raises(ValueError, interp, sample)

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0., 1111., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = np.asarray([-111.1, 1222.1, -11068., -1186.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0., 11., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = np.asarray([-12.1, 133.1, -1069., -97.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_fill(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=np.nan)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([np.nan, np.nan, np.nan])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        assert_array_almost_equal(interp(sample, method="linear"), wanted)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = NearestNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    def test_linear_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = LinearNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_duck_typed_values(self, method):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        values = MyValue((5, 7))

        interp = RegularGridInterpolator((x, y), values, method=method)
        v1 = interp([0.4, 0.7])

        interp = RegularGridInterpolator((x, y), values._v, method=method)
        v2 = interp([0.4, 0.7])
        assert_allclose(v1, v2)

    def test_invalid_fill_value(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)

        # integers can be cast to floats
        RegularGridInterpolator((x, y), values, fill_value=1)

        # complex values cannot
        assert_raises(ValueError, RegularGridInterpolator,
                      (x, y), values, fill_value=1+2j)

    def test_fillvalue_type(self):
        # from #3703; test that interpolator object construction succeeds
        values = np.ones((10, 20, 30), dtype='>f4')
        points = [np.arange(n) for n in values.shape]
        # xi = [(1, 1, 1)]
        RegularGridInterpolator(points, values)
        RegularGridInterpolator(points, values, fill_value=0.)

    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.
        def f(x, y):
            return x + y
        x = np.linspace(1, 1, 1)
        y = np.linspace(1, 10, 10)
        data = f(*np.meshgrid(x, y, indexing="ij", sparse=True))

        interp = RegularGridInterpolator((x, y), data, method="linear",
                                         bounds_error=False, fill_value=101)

        # check values at the grid
        assert_allclose(interp(np.array([[1, 1], [1, 5], [1, 10]])),
                        [2, 6, 11],
                        atol=1e-14)

        # check off-grid interpolation is indeed linear
        assert_allclose(interp(np.array([[1, 1.4], [1, 5.3], [1, 10]])),
                        [2.4, 6.3, 11],
                        atol=1e-14)

        # check exrapolation w/ fill_value
        assert_allclose(interp(np.array([1.1, 2.4])),
                        interp.fill_value,
                        atol=1e-14)

        # check extrapolation: linear along the `y` axis, const along `x`
        interp.fill_value = None
        assert_allclose(interp([[1, 0.3], [1, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        assert_allclose(interp([[1.5, 0.3], [1.9, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        # extrapolation with method='nearest'
        interp = RegularGridInterpolator((x, y), data, method="nearest",
                                         bounds_error=False, fill_value=None)
        assert_allclose(interp([[1.5, 1.8], [-4, 5.1]]),
                        [3, 6],
                        atol=1e-15)

    @pytest.mark.parametrize("fill_value", [None, np.nan, np.pi])
    @pytest.mark.parametrize("method", ['linear', 'nearest'])
    def test_length_one_axis2(self, fill_value, method):
        options = {"fill_value": fill_value, "bounds_error": False,
                   "method": method}

        x = np.linspace(0, 2*np.pi, 20)
        z = np.sin(x)

        fa = RegularGridInterpolator((x,), z[:], **options)
        fb = RegularGridInterpolator((x, [0]), z[:, None], **options)

        x1a = np.linspace(-1, 2*np.pi+1, 100)
        za = fa(x1a)

        # evaluated at provided y-value, fb should behave exactly as fa
        y1b = np.zeros(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        assert_allclose(zb, za)

        # evaluated at a different y-value, fb should return fill value
        y1b = np.ones(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        if fill_value is None:
            assert_allclose(zb, za)
        else:
            assert_allclose(zb, fill_value)

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    def test_nan_x_1d(self, method):
        # gh-6624 : if x is nan, result should be nan
        f = RegularGridInterpolator(([1, 2, 3],), [10, 20, 30], fill_value=1,
                                    bounds_error=False, method=method)
        assert np.isnan(f([np.nan]))

        # test arbitrary nan pattern
        rng = np.random.default_rng(8143215468)
        x = rng.random(size=100)*4
        i = rng.random(size=100) > 0.5
        x[i] = np.nan
        with np.errstate(invalid='ignore'):
            # out-of-bounds comparisons, `out_of_bounds += x < grid[0]`,
            # generate numpy warnings if `x` contains nans.
            # These warnings should propagate to user (since `x` is user
            # input) and we simply filter them out.
            res = f(x)

        assert_equal(res[i], np.nan)
        assert_equal(res[~i], f(x[~i]))

        # also test the length-one axis f(nan)
        x = [1, 2, 3]
        y = [1, ]
        data = np.ones((3, 1))
        f = RegularGridInterpolator((x, y), data, fill_value=1,
                                    bounds_error=False, method=method)
        assert np.isnan(f([np.nan, 1]))
        assert np.isnan(f([1, np.nan]))

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    def test_nan_x_2d(self, method):
        x, y = np.array([0, 1, 2]), np.array([1, 3, 7])

        def f(x, y):
            return x**2 + y**2

        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)
        interp = RegularGridInterpolator((x, y), data,
                                         method=method, bounds_error=False)

        with np.errstate(invalid='ignore'):
            res = interp([[1.5, np.nan], [1, 1]])
        assert_allclose(res[1], 2, atol=1e-14)
        assert np.isnan(res[0])

        # test arbitrary nan pattern
        rng = np.random.default_rng(8143215468)
        x = rng.random(size=100)*4-1
        y = rng.random(size=100)*8
        i1 = rng.random(size=100) > 0.5
        i2 = rng.random(size=100) > 0.5
        i = i1 | i2
        x[i1] = np.nan
        y[i2] = np.nan
        z = np.array([x, y]).T
        with np.errstate(invalid='ignore'):
            # out-of-bounds comparisons, `out_of_bounds += x < grid[0]`,
            # generate numpy warnings if `x` contains nans.
            # These warnings should propagate to user (since `x` is user
            # input) and we simply filter them out.
            res = interp(z)

        assert_equal(res[i], np.nan)
        assert_equal(res[~i], interp(z[~i]))

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize(("ndims", "func"), [
        (2, lambda x, y: 2 * x ** 3 + 3 * y ** 2),
        (3, lambda x, y, z: 2 * x ** 3 + 3 * y ** 2 - z),
        (4, lambda x, y, z, a: 2 * x ** 3 + 3 * y ** 2 - z + a),
        (5, lambda x, y, z, a, b: 2 * x ** 3 + 3 * y ** 2 - z + a * b),
    ])
    def test_descending_points_nd(self, method, ndims, func):
        rng = np.random.default_rng(42)
        sample_low = 1
        sample_high = 5
        test_points = rng.uniform(sample_low, sample_high, size=(2, ndims))

        ascending_points = [np.linspace(sample_low, sample_high, 12)
                            for _ in range(ndims)]

        ascending_values = func(*np.meshgrid(*ascending_points,
                                             indexing="ij",
                                             sparse=True))

        ascending_interp = RegularGridInterpolator(ascending_points,
                                                   ascending_values,
                                                   method=method)
        ascending_result = ascending_interp(test_points)

        descending_points = [xi[::-1] for xi in ascending_points]
        descending_values = func(*np.meshgrid(*descending_points,
                                              indexing="ij",
                                              sparse=True))
        descending_interp = RegularGridInterpolator(descending_points,
                                                    descending_values,
                                                    method=method)
        descending_result = descending_interp(test_points)

        assert_array_equal(ascending_result, descending_result)

    def test_invalid_points_order(self):
        def val_func_2d(x, y):
            return 2 * x ** 3 + 3 * y ** 2

        x = np.array([.5, 2., 0., 4., 5.5])  # not ascending or descending
        y = np.array([.5, 2., 3., 4., 5.5])
        points = (x, y)
        values = val_func_2d(*np.meshgrid(*points, indexing='ij',
                                          sparse=True))
        match = "must be strictly ascending or descending"
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values)

    @parametrize_rgi_interp_methods
    def test_fill_value(self, method):
        interp = RegularGridInterpolator([np.arange(6)], np.ones(6),
                                         method=method, bounds_error=False)
        assert np.isnan(interp([10]))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        # Verify that non-scalar valued values also works
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        rng = np.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)
        assert_equal(v.shape, (7, 3, 8), err_msg=method)

        vs = []
        for j in range(8):
            interp = RegularGridInterpolator(points, values[..., j],
                                             method=method,
                                             bounds_error=False)
            vs.append(interp(sample))
        v2 = np.array(vs).transpose(1, 2, 0)

        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize("flip_points", [False, True])
    def test_nonscalar_values_2(self, method, flip_points):
        # Verify that non-scalar valued values also work : use different
        # lengths of axes to simplify tracing the internals
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        # verify, that strictly decreasing dimensions work
        if flip_points:
            points = [tuple(reversed(p)) for p in points]

        rng = np.random.default_rng(1234)

        trailing_points = (3, 2)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)   # a single sample point !

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                vs[i, j] = interp(sample).item()
        v2 = np.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    def test_nonscalar_values_linear_2D(self):
        # Verify that non-scalar values work in the 2D fast path
        method = 'linear'
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), ]

        rng = np.random.default_rng(1234)

        trailing_points = (3, 4)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, *trailing_points))
        sample = rng.random(2)   # a single sample point !

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                vs[i, j] = interp(sample).item()
        v2 = np.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("xi_dtype", [np.float32, np.float64])
    def test_float32_values(self, dtype, xi_dtype):
        # regression test for gh-17718: values.dtype=float32 fails
        def f(x, y):
            return 2 * x**3 + 3 * y**2

        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)

        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)

        data = data.astype(dtype)

        interp = RegularGridInterpolator((x, y), data)

        pts = np.array([[2.1, 6.2],
                        [3.3, 5.2]], dtype=xi_dtype)

        # the values here are just what the call returns; the test checks that
        # that the call succeeds at all, instead of failing with cython not
        # having a float32 kernel
        assert_allclose(interp(pts), [134.10469388, 153.40069388], atol=1e-7)


class MyValue:
    """
    Minimal indexable object
    """

    def __init__(self, shape):
        self.ndim = 2
        self.shape = shape
        self._v = np.arange(np.prod(shape)).reshape(shape)

    def __getitem__(self, idx):
        return self._v[idx]

    def __array_interface__(self):
        return None

    def __array__(self):
        raise RuntimeError("No array representation")


class TestInterpN:
    def _sample_2d_data(self):
        x = np.array([.5, 2., 3., 4., 5.5, 6.])
        y = np.array([.5, 2., 3., 4., 5.5, 6.])
        z = np.array(
            [
                [1, 2, 1, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 3, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
            ]
        )
        return x, y, z

    def test_spline_2d(self):
        x, y, z = self._sample_2d_data()
        lut = RectBivariateSpline(x, y, z)

        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        assert_array_almost_equal(interpn((x, y), z, xi, method="splinef2d"),
                                  lut.ev(xi[:, 0], xi[:, 1]))

    @parametrize_rgi_interp_methods
    def test_list_input(self, method):
        x, y, z = self._sample_2d_data()
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        v1 = interpn((x, y), z, xi, method=method)
        v2 = interpn(
            (x.tolist(), y.tolist()), z.tolist(), xi.tolist(), method=method
        )
        assert_allclose(v1, v2, err_msg=method)

    def test_spline_2d_outofbounds(self):
        x = np.array([.5, 2., 3., 4., 5.5])
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)

        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
        actual = interpn((x, y), z, xi, method="splinef2d",
                         bounds_error=False, fill_value=999.99)
        expected = lut.ev(xi[:, 0], xi[:, 1])
        expected[2:4] = 999.99
        assert_array_almost_equal(actual, expected)

        # no extrapolation for splinef2d
        assert_raises(ValueError, interpn, (x, y), z, xi, method="splinef2d",
                      bounds_error=False, fill_value=None)

    def _sample_4d_data(self):
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def test_linear_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="linear")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_linear_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="linear",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_nearest_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values, method="nearest")
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="nearest")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_nearest_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="nearest",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_xi_1d(self):
        # verify that 1-D xi works as expected
        points, values = self._sample_4d_data()
        sample = np.asarray([0.1, 0.1, 10., 9.])
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None,:], bounds_error=False)
        assert_allclose(v1, v2)

    def test_xi_nd(self):
        # verify that higher-d xi works as expected
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        sample = np.random.rand(2, 3, 4)

        v1 = interpn(points, values, sample, method='nearest',
                     bounds_error=False)
        assert_equal(v1.shape, (2, 3))

        v2 = interpn(points, values, sample.reshape(-1, 4),
                     method='nearest', bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_xi_broadcast(self, method):
        # verify that the interpolators broadcast xi
        x, y, values = self._sample_2d_data()
        points = (x, y)

        xi = np.linspace(0, 1, 2)
        yi = np.linspace(0, 3, 3)

        sample = (xi[:, None], yi[None, :])
        v1 = interpn(points, values, sample, method=method, bounds_error=False)
        assert_equal(v1.shape, (2, 3))

        xx, yy = np.meshgrid(xi, yi)
        sample = np.c_[xx.T.ravel(), yy.T.ravel()]

        v2 = interpn(points, values, sample,
                     method=method, bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        # Verify that non-scalar valued values also works
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        rng = np.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))

        v = interpn(points, values, sample, method=method,
                    bounds_error=False)
        assert_equal(v.shape, (7, 3, 8), err_msg=method)

        vs = [interpn(points, values[..., j], sample, method=method,
                      bounds_error=False) for j in range(8)]
        v2 = np.array(vs).transpose(1, 2, 0)

        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    def test_nonscalar_values_2(self, method):
        # Verify that non-scalar valued values also work : use different
        # lengths of axes to simplify tracing the internals
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        rng = np.random.default_rng(1234)

        trailing_points = (3, 2)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)   # a single sample point !

        v = interpn(points, values, sample, method=method, bounds_error=False)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = [[
                interpn(points, values[..., i, j], sample, method=method,
                        bounds_error=False) for i in range(values.shape[-2])
              ] for j in range(values.shape[-1])]

        assert_allclose(v, np.asarray(vs).T, atol=1e-14, err_msg=method)

    def test_non_scalar_values_splinef2d(self):
        # Vector-valued splines supported with fitpack
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        values = np.random.rand(3, 3, 3, 3, 6)
        sample = np.random.rand(7, 11, 4)
        assert_raises(ValueError, interpn, points, values, sample,
                      method='splinef2d')

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j*values

        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        v1 = interpn(points, values, sample, method=method)
        v2r = interpn(points, values.real, sample, method=method)
        v2i = interpn(points, values.imag, sample, method=method)
        v2 = v2r + 1j*v2i
        assert_allclose(v1, v2)

    def test_complex_spline2fd(self):
        # Complex-valued data not supported by spline2fd
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j*values

        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        with assert_warns(np.ComplexWarning):
            interpn(points, values, sample, method='splinef2d')

    @pytest.mark.parametrize(
        "method",
        ["linear", "nearest"]
    )
    def test_duck_typed_values(self, method):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        values = MyValue((5, 7))

        v1 = interpn((x, y), values, [0.4, 0.7], method=method)
        v2 = interpn((x, y), values._v, [0.4, 0.7], method=method)
        assert_allclose(v1, v2)

    @parametrize_rgi_interp_methods
    def test_matrix_input(self, method):
        x = np.linspace(0, 2, 6)
        y = np.linspace(0, 1, 7)

        values = matrix(np.random.rand(6, 7))

        sample = np.random.rand(3, 7, 2)

        v1 = interpn((x, y), values, sample, method=method)
        v2 = interpn((x, y), np.asarray(values), sample, method=method)
        assert_allclose(v1, v2)

    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.

        values = np.array([[0.1, 1, 10]])
        xi = np.array([[1, 2.2], [1, 3.2], [1, 3.8]])

        res = interpn(([1], [2, 3, 4]), values, xi)
        wanted = [0.9*0.2 + 0.1,   # on [2, 3) it's 0.9*(x-2) + 0.1
                  9*0.2 + 1,       # on [3, 4] it's 9*(x-3) + 1
                  9*0.8 + 1]

        assert_allclose(res, wanted, atol=1e-15)

        # check extrapolation
        xi = np.array([[1.1, 2.2], [1.5, 3.2], [-2.3, 3.8]])
        res = interpn(([1], [2, 3, 4]), values, xi,
                      bounds_error=False, fill_value=None)

        assert_allclose(res, wanted, atol=1e-15)

    def test_descending_points(self):
        def value_func_4d(x, y, z, a):
            return 2 * x ** 3 + 3 * y ** 2 - z - a

        x1 = np.array([0, 1, 2, 3])
        x2 = np.array([0, 10, 20, 30])
        x3 = np.array([0, 10, 20, 30])
        x4 = np.array([0, .1, .2, .30])
        points = (x1, x2, x3, x4)
        values = value_func_4d(
            *np.meshgrid(*points, indexing='ij', sparse=True))
        pts = (0.1, 0.3, np.transpose(np.linspace(0, 30, 4)),
               np.linspace(0, 0.3, 4))
        correct_result = interpn(points, values, pts)

        x1_descend = x1[::-1]
        x2_descend = x2[::-1]
        x3_descend = x3[::-1]
        x4_descend = x4[::-1]
        points_shuffled = (x1_descend, x2_descend, x3_descend, x4_descend)
        values_shuffled = value_func_4d(
            *np.meshgrid(*points_shuffled, indexing='ij', sparse=True))
        test_result = interpn(points_shuffled, values_shuffled, pts)

        assert_array_equal(correct_result, test_result)

    def test_invalid_points_order(self):
        x = np.array([.5, 2., 0., 4., 5.5])  # not ascending or descending
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T

        match = "must be strictly ascending or descending"
        with pytest.raises(ValueError, match=match):
            interpn((x, y), z, xi)

    def test_invalid_xi_dimensions(self):
        # https://github.com/scipy/scipy/issues/16519
        points = [(0, 1)]
        values = [0, 1]
        xi = np.ones((1, 1, 3))
        msg = ("The requested sample points xi have dimension 3, but this "
               "RegularGridInterpolator has dimension 1")
        with assert_raises(ValueError, match=msg):
            interpn(points, values, xi)

    def test_readonly_grid(self):
        # https://github.com/scipy/scipy/issues/17716
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)
        values = np.ones((5, 6, 7))
        point = np.array([2.21, 3.12, 1.15])
        for d in points:
            d.flags.writeable = False
        values.flags.writeable = False
        point.flags.writeable = False
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    def test_2d_readonly_grid(self):
        # https://github.com/scipy/scipy/issues/17716
        # test special 2d case
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        points = (x, y)
        values = np.ones((5, 6))
        point = np.array([2.21, 3.12])
        for d in points:
            d.flags.writeable = False
        values.flags.writeable = False
        point.flags.writeable = False
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    def test_non_c_contiguous_grid(self):
        # https://github.com/scipy/scipy/issues/17716
        x = np.linspace(0, 4, 5)
        x = np.vstack((x, np.empty_like(x))).T.copy()[:, 0]
        assert not x.flags.c_contiguous
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)
        values = np.ones((5, 6, 7))
        point = np.array([2.21, 3.12, 1.15])
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)

    @pytest.mark.parametrize("dtype", ['>f8', '<f8'])
    def test_endianness(self, dtype):
        # https://github.com/scipy/scipy/issues/17716
        # test special 2d case
        x = np.linspace(0, 4, 5, dtype=dtype)
        y = np.linspace(0, 5, 6, dtype=dtype)
        points = (x, y)
        values = np.ones((5, 6), dtype=dtype)
        point = np.array([2.21, 3.12], dtype=dtype)
        interpn(points, values, point)
        RegularGridInterpolator(points, values)(point)
