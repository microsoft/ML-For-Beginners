"""Includes test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005
Modified for Array API, 2023

"""
from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
    array_api_compatible,
    skip_if_array_api_gpu,
    skip_if_array_api_backend
)
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft

_5_smooth_numbers = [
    2, 3, 4, 5, 6, 8, 9, 10,
    2 * 3 * 5,
    2**3 * 3**5,
    2**3 * 3**3 * 5**2,
]

def test_next_fast_len():
    for n in _5_smooth_numbers:
        assert_equal(next_fast_len(n), n)


def _assert_n_smooth(x, n):
    x_orig = x
    if n < 2:
        assert False

    while True:
        q, r = divmod(x, 2)
        if r != 0:
            break
        x = q

    for d in range(3, n+1, 2):
        while True:
            q, r = divmod(x, d)
            if r != 0:
                break
            x = q

    assert x == 1, \
           f'x={x_orig} is not {n}-smooth, remainder={x}'


class TestNextFastLen:

    def test_next_fast_len(self):
        np.random.seed(1234)

        def nums():
            yield from range(1, 1000)
            yield 2**5 * 3**5 * 4**5 + 1

        for n in nums():
            m = next_fast_len(n)
            _assert_n_smooth(m, 11)
            assert m == next_fast_len(n, False)

            m = next_fast_len(n, True)
            _assert_n_smooth(m, 5)

    def test_np_integers(self):
        ITYPES = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
        for ityp in ITYPES:
            x = ityp(12345)
            testN = next_fast_len(x)
            assert_equal(testN, next_fast_len(int(x)))

    def testnext_fast_len_small(self):
        hams = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15,
            16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000
        }
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    @pytest.mark.xfail(sys.maxsize < 2**32,
                       reason="Hamming Numbers too large for 32-bit",
                       raises=ValueError, strict=True)
    def testnext_fast_len_big(self):
        hams = {
            510183360: 510183360, 510183360 + 1: 512000000,
            511000000: 512000000,
            854296875: 854296875, 854296875 + 1: 859963392,
            196608000000: 196608000000, 196608000000 + 1: 196830000000,
            8789062500000: 8789062500000, 8789062500000 + 1: 8796093022208,
            206391214080000: 206391214080000,
            206391214080000 + 1: 206624260800000,
            470184984576000: 470184984576000,
            470184984576000 + 1: 470715894135000,
            7222041363087360: 7222041363087360,
            7222041363087360 + 1: 7230196133913600,
            # power of 5    5**23
            11920928955078125: 11920928955078125,
            11920928955078125 - 1: 11920928955078125,
            # power of 3    3**34
            16677181699666569: 16677181699666569,
            16677181699666569 - 1: 16677181699666569,
            # power of 2   2**54
            18014398509481984: 18014398509481984,
            18014398509481984 - 1: 18014398509481984,
            # above this, int(ceil(n)) == int(ceil(n+1))
            19200000000000000: 19200000000000000,
            19200000000000000 + 1: 19221679687500000,
            288230376151711744: 288230376151711744,
            288230376151711744 + 1: 288325195312500000,
            288325195312500000 - 1: 288325195312500000,
            288325195312500000: 288325195312500000,
            288325195312500000 + 1: 288555831593533440,
        }
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    def test_keyword_args(self):
        assert next_fast_len(11, real=True) == 12
        assert next_fast_len(target=7, real=False) == 7


class Test_init_nd_shape_and_axes:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_0d_defaults(self, xp):
        x = xp.asarray(4)
        shape = None
        axes = None

        shape_expected = ()
        axes_expected = []

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_0d_defaults(self, xp):
        x = xp.asarray(7.)
        shape = None
        axes = None

        shape_expected = ()
        axes_expected = []

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_1d_defaults(self, xp):
        x = xp.asarray([1, 2, 3])
        shape = None
        axes = None

        shape_expected = (3,)
        axes_expected = [0]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_1d_defaults(self, xp):
        x = xp.arange(0, 1, .1)
        shape = None
        axes = None

        shape_expected = (10,)
        axes_expected = [0]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_py_2d_defaults(self, xp):
        x = xp.asarray([[1, 2, 3, 4],
                        [5, 6, 7, 8]])
        shape = None
        axes = None

        shape_expected = (2, 4)
        axes_expected = [0, 1]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_2d_defaults(self, xp):
        x = xp.arange(0, 1, .1)
        x = xp.reshape(x, (5, 2))
        shape = None
        axes = None

        shape_expected = (5, 2)
        axes_expected = [0, 1]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_defaults(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = None
        axes = None

        shape_expected = (6, 2, 5, 3, 4)
        axes_expected = [0, 1, 2, 3, 4]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_shape(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = [10, -1, -1, 1, 4]
        axes = None

        shape_expected = (10, 2, 5, 1, 4)
        axes_expected = [0, 1, 2, 3, 4]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_axes(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = None
        axes = [4, 1, 2]

        shape_expected = (4, 2, 5)
        axes_expected = [4, 1, 2]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_xp_5d_set_shape_axes(self, xp):
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = [10, -1, 2]
        axes = [1, 0, 3]

        shape_expected = (10, 6, 2)
        axes_expected = [1, 0, 3]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_shape_axes_subset(self, xp):
        x = xp.zeros((2, 3, 4, 5))
        shape, axes = _init_nd_shape_and_axes(x, shape=(5, 5, 5), axes=None)

        assert shape == (5, 5, 5)
        assert axes == [1, 2, 3]

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_errors(self, xp):
        x = xp.zeros(1)
        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=None, axes=[[1, 2], [3, 4]])

        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=None, axes=[1., 2., 3., 4.])

        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            _init_nd_shape_and_axes(x, shape=None, axes=[1])

        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            _init_nd_shape_and_axes(x, shape=None, axes=[-2])

        with assert_raises(ValueError,
                           match="all axes must be unique"):
            _init_nd_shape_and_axes(x, shape=None, axes=[0, 0])

        with assert_raises(ValueError, match="shape must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=[[1, 2], [3, 4]], axes=None)

        with assert_raises(ValueError, match="shape must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=[1., 2., 3., 4.], axes=None)

        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            _init_nd_shape_and_axes(xp.zeros([1, 1, 1, 1]),
                                    shape=[1, 2, 3], axes=[1])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[0\]\) specified"):
            _init_nd_shape_and_axes(x, shape=[0], axes=None)

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[-2\]\) specified"):
            _init_nd_shape_and_axes(x, shape=-2, axes=None)


class TestFFTShift:

    # torch.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_definition(self, xp):
        x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1])
        y = xp.asarray([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        xp_assert_close(fft.fftshift(x), y)
        xp_assert_close(fft.ifftshift(y), x)
        x = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
        y = xp.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
        xp_assert_close(fft.fftshift(x), y)
        xp_assert_close(fft.ifftshift(y), x)

    # torch.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_inverse(self, xp):
        for n in [1, 4, 9, 100, 211]:
            x = xp.asarray(np.random.random((n,)))
            xp_assert_close(fft.ifftshift(fft.fftshift(x)), x)

    # torch.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_axes_keyword(self, xp):
        freqs = xp.asarray([[0, 1, 2], [3, 4, -4], [-3, -2, -1]])
        shifted = xp.asarray([[-1, -3, -2], [2, 0, 1], [-4, 3, 4]])
        xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shifted)
        xp_assert_close(fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,)))
        xp_assert_close(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        xp_assert_close(fft.ifftshift(shifted, axes=0),
                        fft.ifftshift(shifted, axes=(0,)))
        xp_assert_close(fft.fftshift(freqs), shifted)
        xp_assert_close(fft.ifftshift(shifted), freqs)
    
    # torch.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('torch')
    @array_api_compatible
    def test_uneven_dims(self, xp):
        """ Test 2D input, which has uneven dimension sizes """
        freqs = xp.asarray([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        # shift in dimension 0
        shift_dim0 = xp.asarray([
            [4, 5],
            [0, 1],
            [2, 3]
        ])
        xp_assert_close(fft.fftshift(freqs, axes=0), shift_dim0)
        xp_assert_close(fft.ifftshift(shift_dim0, axes=0), freqs)
        xp_assert_close(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        xp_assert_close(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        shift_dim1 = xp.asarray([
            [1, 0],
            [3, 2],
            [5, 4]
        ])
        xp_assert_close(fft.fftshift(freqs, axes=1), shift_dim1)
        xp_assert_close(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = xp.asarray([
            [5, 4],
            [1, 0],
            [3, 2]
        ])
        xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        xp_assert_close(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        xp_assert_close(fft.fftshift(freqs, axes=None), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=None), freqs)
        xp_assert_close(fft.fftshift(freqs), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both), freqs)


class TestFFTFreq:

    # fft not yet implemented by numpy.array_api
    @skip_if_array_api_backend('numpy.array_api')
    # cupy.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('cupy')
    @array_api_compatible
    def test_definition(self, xp):
        device = SCIPY_DEVICE
        try:
            x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1],
                           dtype=xp.float64, device=device)
            x2 = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1],
                            dtype=xp.float64, device=device)
        except TypeError:
            x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=xp.float64)
            x2 = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1],
                            dtype=xp.float64)

        y = xp.asarray(9 * fft.fftfreq(9, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)
        y = xp.asarray(9 * xp.pi * fft.fftfreq(9, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)

        y = xp.asarray(10 * fft.fftfreq(10, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)
        y = xp.asarray(10 * xp.pi * fft.fftfreq(10, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)


class TestRFFTFreq:

    # fft not yet implemented by numpy.array_api
    @skip_if_array_api_backend('numpy.array_api')
    # cupy.fft not yet implemented by array-api-compat
    @skip_if_array_api_backend('cupy')
    @array_api_compatible
    def test_definition(self, xp):
        device = SCIPY_DEVICE
        try:
            x = xp.asarray([0, 1, 2, 3, 4], dtype=xp.float64, device=device)
            x2 = xp.asarray([0, 1, 2, 3, 4, 5], dtype=xp.float64, device=device)
        except TypeError:
            # work around the `device` keyword not being implemented in numpy yet
            x = xp.asarray([0, 1, 2, 3, 4], dtype=xp.float64)
            x2 = xp.asarray([0, 1, 2, 3, 4, 5], dtype=xp.float64)

        y = xp.asarray(9 * fft.rfftfreq(9, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)
        y = xp.asarray(9 * xp.pi * fft.rfftfreq(9, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x)

        y = xp.asarray(10 * fft.rfftfreq(10, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)
        y = xp.asarray(10 * xp.pi * fft.rfftfreq(10, xp.pi, xp=xp), dtype=xp.float64)
        xp_assert_close(y, x2)
