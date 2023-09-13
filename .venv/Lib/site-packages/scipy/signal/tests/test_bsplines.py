# pylint: disable=missing-docstring
import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_almost_equal, suppress_warnings)
import pytest
from pytest import raises

import scipy.signal._bsplines as bsp
from scipy import signal


class TestBSplines:
    """Test behaviors of B-splines. Some of the values tested against were
    returned as of SciPy 1.1.0 and are included for regression testing
    purposes. Others (at integer points) are compared to theoretical
    expressions (cf. Unser, Aldroubi, Eden, IEEE TSP 1993, Table 1)."""

    def test_spline_filter(self):
        np.random.seed(12457)
        # Test the type-error branch
        raises(TypeError, bsp.spline_filter, array([0]), 0)
        # Test the real branch
        np.random.seed(12457)
        data_array_real = np.random.rand(12, 12)
        # make the magnitude exceed 1, and make some negative
        data_array_real = 10*(1-2*data_array_real)
        result_array_real = array(
            [[-.463312621, 8.33391222, .697290949, 5.28390836,
              5.92066474, 6.59452137, 9.84406950, -8.78324188,
              7.20675750, -8.17222994, -4.38633345, 9.89917069],
             [2.67755154, 6.24192170, -3.15730578, 9.87658581,
              -9.96930425, 3.17194115, -4.50919947, 5.75423446,
              9.65979824, -8.29066885, .971416087, -2.38331897],
             [-7.08868346, 4.89887705, -1.37062289, 7.70705838,
              2.51526461, 3.65885497, 5.16786604, -8.77715342e-03,
              4.10533325, 9.04761993, -.577960351, 9.86382519],
             [-4.71444301, -1.68038985, 2.84695116, 1.14315938,
              -3.17127091, 1.91830461, 7.13779687, -5.35737482,
              -9.66586425, -9.87717456, 9.93160672, 4.71948144],
             [9.49551194, -1.92958436, 6.25427993, -9.05582911,
              3.97562282, 7.68232426, -1.04514824, -5.86021443,
              -8.43007451, 5.47528997, 2.06330736, -8.65968112],
             [-8.91720100, 8.87065356, 3.76879937, 2.56222894,
              -.828387146, 8.72288903, 6.42474741, -6.84576083,
              9.94724115, 6.90665380, -6.61084494, -9.44907391],
             [9.25196790, -.774032030, 7.05371046, -2.73505725,
              2.53953305, -1.82889155, 2.95454824, -1.66362046,
              5.72478916, -3.10287679, 1.54017123, -7.87759020],
             [-3.98464539, -2.44316992, -1.12708657, 1.01725672,
              -8.89294671, -5.42145629, -6.16370321, 2.91775492,
              9.64132208, .702499998, -2.02622392, 1.56308431],
             [-2.22050773, 7.89951554, 5.98970713, -7.35861835,
              5.45459283, -7.76427957, 3.67280490, -4.05521315,
              4.51967507, -3.22738749, -3.65080177, 3.05630155],
             [-6.21240584, -.296796126, -8.34800163, 9.21564563,
              -3.61958784, -4.77120006, -3.99454057, 1.05021988e-03,
              -6.95982829, 6.04380797, 8.43181250, -2.71653339],
             [1.19638037, 6.99718842e-02, 6.72020394, -2.13963198,
              3.75309875, -5.70076744, 5.92143551, -7.22150575,
              -3.77114594, -1.11903194, -5.39151466, 3.06620093],
             [9.86326886, 1.05134482, -7.75950607, -3.64429655,
              7.81848957, -9.02270373, 3.73399754, -4.71962549,
              -7.71144306, 3.78263161, 6.46034818, -4.43444731]])
        assert_allclose(bsp.spline_filter(data_array_real, 0),
                        result_array_real)

    def test_bspline(self):
        # Verify with theoretical results at integer points up to order 5
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            assert_allclose(bsp.bspline([-1, 0, 1], 0),
                            array([0, 1, 0]))
            assert_allclose(bsp.bspline([-1, 0, 1], 1),
                            array([0, 1, 0]))
            assert_allclose(bsp.bspline([-2, -1, 0, 1, 2], 2),
                            array([0, 1, 6, 1, 0])/8.)
            assert_allclose(bsp.bspline([-2, -1, 0, 1, 2], 3),
                            array([0, 1, 4, 1, 0])/6.)
            assert_allclose(bsp.bspline([-3, -2, -1, 0, 1, 2, 3], 4),
                            array([0, 1, 76, 230, 76, 1, 0])/384.)
            assert_allclose(bsp.bspline([-3, -2, -1, 0, 1, 2, 3], 5),
                            array([0, 1, 26, 66, 26, 1, 0]) / 120.)
            # Compare with SciPy 1.1.0
            np.random.seed(12458)
            assert_allclose(bsp.bspline(np.random.rand(1, 1), 2),
                            array([[0.73694695]]))

    def test_gauss_spline(self):
        np.random.seed(12459)
        assert_almost_equal(bsp.gauss_spline(0, 0), 1.381976597885342)
        assert_allclose(bsp.gauss_spline(array([1.]), 1), array([0.04865217]))

    def test_gauss_spline_list(self):
        # regression test for gh-12152 (accept array_like)
        knots = [-1.0, 0.0, -1.0]
        assert_almost_equal(bsp.gauss_spline(knots, 3),
                            array([0.15418033, 0.6909883, 0.15418033]))

    def test_cubic(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            np.random.seed(12460)
            # Verify with theoretical results at integer points (see docstring)
            assert_allclose(bsp.cubic([-2, -1, 0, 1, 2]),
                            array([0, 1, 4, 1, 0])/6.)

    def test_quadratic(self):
        np.random.seed(12461)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            # Verify correct results at integer points
            assert_allclose(bsp.quadratic([-2, -1, 0, 1, 2]),
                            array([0, 1, 6, 1, 0])/8.)

    def test_cspline1d(self):
        np.random.seed(12462)
        assert_array_equal(bsp.cspline1d(array([0])), [0.])
        c1d = array([1.21037185, 1.86293902, 2.98834059, 4.11660378,
                     4.78893826])
        # test lamda != 0
        assert_allclose(bsp.cspline1d(array([1., 2, 3, 4, 5]), 1), c1d)
        c1d0 = array([0.78683946, 2.05333735, 2.99981113, 3.94741812,
                      5.21051638])
        assert_allclose(bsp.cspline1d(array([1., 2, 3, 4, 5])), c1d0)

    def test_qspline1d(self):
        np.random.seed(12463)
        assert_array_equal(bsp.qspline1d(array([0])), [0.])
        # test lamda != 0
        raises(ValueError, bsp.qspline1d, array([1., 2, 3, 4, 5]), 1.)
        raises(ValueError, bsp.qspline1d, array([1., 2, 3, 4, 5]), -1.)
        q1d0 = array([0.85350007, 2.02441743, 2.99999534, 3.97561055,
                      5.14634135])
        assert_allclose(bsp.qspline1d(array([1., 2, 3, 4, 5])), q1d0)

    def test_cspline1d_eval(self):
        np.random.seed(12464)
        assert_allclose(bsp.cspline1d_eval(array([0., 0]), [0.]), array([0.]))
        assert_array_equal(bsp.cspline1d_eval(array([1., 0, 1]), []),
                           array([]))
        x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        dx = x[1]-x[0]
        newx = [-6., -5.5, -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1.,
                -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.,
                6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
                12.5]
        y = array([4.216, 6.864, 3.514, 6.203, 6.759, 7.433, 7.874, 5.879,
                   1.396, 4.094])
        cj = bsp.cspline1d(y)
        newy = array([6.203, 4.41570658, 3.514, 5.16924703, 6.864, 6.04643068,
                      4.21600281, 6.04643068, 6.864, 5.16924703, 3.514,
                      4.41570658, 6.203, 6.80717667, 6.759, 6.98971173, 7.433,
                      7.79560142, 7.874, 7.41525761, 5.879, 3.18686814, 1.396,
                      2.24889482, 4.094, 2.24889482, 1.396, 3.18686814, 5.879,
                      7.41525761, 7.874, 7.79560142, 7.433, 6.98971173, 6.759,
                      6.80717667, 6.203, 4.41570658])
        assert_allclose(bsp.cspline1d_eval(cj, newx, dx=dx, x0=x[0]), newy)

    def test_qspline1d_eval(self):
        np.random.seed(12465)
        assert_allclose(bsp.qspline1d_eval(array([0., 0]), [0.]), array([0.]))
        assert_array_equal(bsp.qspline1d_eval(array([1., 0, 1]), []),
                           array([]))
        x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        dx = x[1]-x[0]
        newx = [-6., -5.5, -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1.,
                -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.,
                6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
                12.5]
        y = array([4.216, 6.864, 3.514, 6.203, 6.759, 7.433, 7.874, 5.879,
                   1.396, 4.094])
        cj = bsp.qspline1d(y)
        newy = array([6.203, 4.49418159, 3.514, 5.18390821, 6.864, 5.91436915,
                      4.21600002, 5.91436915, 6.864, 5.18390821, 3.514,
                      4.49418159, 6.203, 6.71900226, 6.759, 7.03980488, 7.433,
                      7.81016848, 7.874, 7.32718426, 5.879, 3.23872593, 1.396,
                      2.34046013, 4.094, 2.34046013, 1.396, 3.23872593, 5.879,
                      7.32718426, 7.874, 7.81016848, 7.433, 7.03980488, 6.759,
                      6.71900226, 6.203, 4.49418159])
        assert_allclose(bsp.qspline1d_eval(cj, newx, dx=dx, x0=x[0]), newy)


def test_sepfir2d_invalid_filter():
    filt = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    image = np.random.rand(7, 9)
    # No error for odd lengths
    signal.sepfir2d(image, filt, filt[2:])

    # Row or column filter must be odd
    with pytest.raises(ValueError, match="odd length"):
        signal.sepfir2d(image, filt, filt[1:])
    with pytest.raises(ValueError, match="odd length"):
        signal.sepfir2d(image, filt[1:], filt)

    # Filters must be 1-dimensional
    with pytest.raises(ValueError, match="object too deep"):
        signal.sepfir2d(image, filt.reshape(1, -1), filt)
    with pytest.raises(ValueError, match="object too deep"):
        signal.sepfir2d(image, filt, filt.reshape(1, -1))

def test_sepfir2d_invalid_image():
    filt = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    image = np.random.rand(8, 8)

    # Image must be 2 dimensional
    with pytest.raises(ValueError, match="object too deep"):
        signal.sepfir2d(image.reshape(4, 4, 4), filt, filt)

    with pytest.raises(ValueError, match="object of too small depth"):
        signal.sepfir2d(image[0], filt, filt)


def test_cspline2d():
    np.random.seed(181819142)
    image = np.random.rand(71, 73)
    signal.cspline2d(image, 8.0)


def test_qspline2d():
    np.random.seed(181819143)
    image = np.random.rand(71, 73)
    signal.qspline2d(image)
