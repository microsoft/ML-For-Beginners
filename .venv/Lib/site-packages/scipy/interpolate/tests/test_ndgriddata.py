import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises

from scipy.interpolate import (griddata, NearestNDInterpolator,
                               LinearNDInterpolator,
                               CloughTocher2DInterpolator)


parametrize_interpolators = pytest.mark.parametrize(
    "interpolator", [NearestNDInterpolator, LinearNDInterpolator,
                     CloughTocher2DInterpolator]
)

class TestGriddata:
    def test_fill_value(self):
        x = [(0,0), (0,1), (1,0)]
        y = [1, 2, 3]

        yi = griddata(x, y, [(1,1), (1,2), (0,0)], fill_value=-1)
        assert_array_equal(yi, [-1., -1, 1])

        yi = griddata(x, y, [(1,1), (1,2), (0,0)])
        assert_array_equal(yi, [np.nan, np.nan, 1])

    def test_alternative_call(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = (np.arange(x.shape[0], dtype=np.float64)[:,None]
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata((x[:,0], x[:,1]), y, (x[:,0], x[:,1]), method=method,
                              rescale=rescale)
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)

    def test_multivalue_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = (np.arange(x.shape[0], dtype=np.float64)[:,None]
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, x, method=method, rescale=rescale)
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)

    def test_multipoint_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)

        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, xi, method=method, rescale=rescale)

                assert_equal(yi.shape, (5, 3), err_msg=msg)
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),
                                atol=1e-14, err_msg=msg)

    def test_complex_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 2j*y[::-1]

        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, xi, method=method, rescale=rescale)

                assert_equal(yi.shape, (5, 3), err_msg=msg)
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),
                                atol=1e-14, err_msg=msg)

    def test_1d(self):
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        for method in ('nearest', 'linear', 'cubic'):
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-14)

    def test_1d_borders(self):
        # Test for nearest neighbor case with xi outside
        # the range of the values.
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])
        xi = np.array([0.9, 6.5])
        yi_should = np.array([1.0, 1.0])

        method = 'nearest'
        assert_allclose(griddata(x, y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        assert_allclose(griddata(x.reshape(6, 1), y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        assert_allclose(griddata((x, ), y, (xi, ),
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)

    def test_1d_unsorted(self):
        x = np.array([2.5, 1, 4.5, 5, 6, 3])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        for method in ('nearest', 'linear', 'cubic'):
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-10)

    def test_square_rescale_manual(self):
        points = np.array([(0,0), (0,100), (10,100), (10,0), (1, 5)], dtype=np.float64)
        points_rescaled = np.array([(0,0), (0,1), (1,1), (1,0), (0.1, 0.05)],
                                   dtype=np.float64)
        values = np.array([1., 2., -3., 5., 9.], dtype=np.float64)

        xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
                                     np.linspace(0, 100, 14)[None,:])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()

        for method in ('nearest', 'linear', 'cubic'):
            msg = method
            zi = griddata(points_rescaled, values, xi/np.array([10, 100.]),
                          method=method)
            zi_rescaled = griddata(points, values, xi, method=method,
                                   rescale=True)
            assert_allclose(zi, zi_rescaled, err_msg=msg,
                            atol=1e-12)

    def test_xi_1d(self):
        # Check that 1-D xi is interpreted as a coordinate
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 2j*y[::-1]

        xi = np.array([0.5, 0.5])

        for method in ('nearest', 'linear', 'cubic'):
            p1 = griddata(x, y, xi, method=method)
            p2 = griddata(x, y, xi[None,:], method=method)
            assert_allclose(p1, p2, err_msg=method)

            xi1 = np.array([0.5])
            xi3 = np.array([0.5, 0.5, 0.5])
            assert_raises(ValueError, griddata, x, y, xi1,
                          method=method)
            assert_raises(ValueError, griddata, x, y, xi3,
                          method=method)


class TestNearestNDInterpolator:
    def test_nearest_options(self):
        # smoke test that NearestNDInterpolator accept cKDTree options
        npts, nd = 4, 3
        x = np.arange(npts*nd).reshape((npts, nd))
        y = np.arange(npts)
        nndi = NearestNDInterpolator(x, y)

        opts = {'balanced_tree': False, 'compact_nodes': False}
        nndi_o = NearestNDInterpolator(x, y, tree_options=opts)
        assert_allclose(nndi(x), nndi_o(x), atol=1e-14)

    def test_nearest_list_argument(self):
        nd = np.array([[0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1, 2]])
        d = nd[:, 3:]

        # z is np.array
        NI = NearestNDInterpolator((d[0], d[1]), d[2])
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])

        # z is list
        NI = NearestNDInterpolator((d[0], d[1]), list(d[2]))
        assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])

    def test_nearest_query_options(self):
        nd = np.array([[0, 0.5, 0, 1],
                       [0, 0, 0.5, 1],
                       [0, 1, 1, 2]])
        delta = 0.1
        query_points = [0 + delta, 1 + delta], [0 + delta, 1 + delta]

        # case 1 - query max_dist is smaller than
        # the query points' nearest distance to nd.
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-7
        assert_array_equal(NI(query_points, distance_upper_bound=distance_upper_bound),
                           [np.nan, np.nan])

        # case 2 - query p is inf, will return [0, 2]
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) - 1e-7
        p = np.inf
        assert_array_equal(
            NI(query_points, distance_upper_bound=distance_upper_bound, p=p),
            [0, 2]
        )

        # case 3 - query max_dist is larger, so should return non np.nan
        distance_upper_bound = np.sqrt(delta ** 2 + delta ** 2) + 1e-7
        assert_array_equal(
            NI(query_points, distance_upper_bound=distance_upper_bound),
            [0, 2]
        )

    def test_nearest_query_valid_inputs(self):
        nd = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1],
                       [0, 1, 1, 2]])
        NI = NearestNDInterpolator((nd[0], nd[1]), nd[2])
        with assert_raises(TypeError):
            NI([0.5, 0.5], query_options="not a dictionary")


class TestNDInterpolators:
    @parametrize_interpolators
    def test_broadcastable_input(self, interpolator):
        # input data
        np.random.seed(0)
        x = np.random.random(10)
        y = np.random.random(10)
        z = np.hypot(x, y)

        # x-y grid for interpolation
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(X, Y)
        XY = np.vstack((X.ravel(), Y.ravel())).T
        interp = interpolator(list(zip(x, y)), z)
        # single array input
        interp_points0 = interp(XY)
        # tuple input
        interp_points1 = interp((X, Y))
        interp_points2 = interp((X, 0.0))
        # broadcastable input
        interp_points3 = interp(X, Y)
        interp_points4 = interp(X, 0.0)

        assert_equal(interp_points0.size ==
                     interp_points1.size ==
                     interp_points2.size ==
                     interp_points3.size ==
                     interp_points4.size, True)

    @parametrize_interpolators
    def test_read_only(self, interpolator):
        # input data
        np.random.seed(0)
        xy = np.random.random((10, 2))
        x, y = xy[:, 0], xy[:, 1]
        z = np.hypot(x, y)

        # interpolation points
        XY = np.random.random((50, 2))

        xy.setflags(write=False)
        z.setflags(write=False)
        XY.setflags(write=False)

        interp = interpolator(xy, z)
        interp(XY)
