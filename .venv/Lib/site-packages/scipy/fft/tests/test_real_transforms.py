import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack

import math
SQRT_2 = math.sqrt(2)

# scipy.fft wraps the fftpack versions but with normalized inverse transforms.
# So, the forward transforms and definitions are already thoroughly tested in
# fftpack/test_real_transforms.py


@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 16])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("orthogonalize", [False, True])
def test_identity_1d(forward, backward, type, n, axis, norm, orthogonalize):
    # Test the identity f^-1(f(x)) == x
    x = np.random.rand(n, n)

    y = forward(x, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    z = backward(y, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    assert_allclose(z, x)

    pad = [(0, 0)] * 2
    pad[axis] = (0, 4)

    y2 = np.pad(y, pad, mode='edge')
    z2 = backward(y2, type, n, axis, norm, orthogonalize=orthogonalize)
    assert_allclose(z2, x)


@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_identity_1d_overwrite(forward, backward, type, dtype, axis, norm,
                               overwrite_x):
    # Test the identity f^-1(f(x)) == x
    x = np.random.rand(7, 8).astype(dtype)
    x_orig = x.copy()

    y = forward(x, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    y_orig = y.copy()
    z = backward(y, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    if not overwrite_x:
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)
    else:
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 4), 0),
                             ((4, 4), 1),
                             ((4, 4), None),
                             ((4, 4), (0, 1)),
                             ((10, 12), None),
                             ((10, 12), (0, 1)),
                             ((4, 5, 6), None),
                             ((4, 5, 6), 1),
                             ((4, 5, 6), (0, 2)),
                         ])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("orthogonalize", [False, True])
def test_identity_nd(forward, backward, type, shape, axes, norm,
                     orthogonalize):
    # Test the identity f^-1(f(x)) == x

    x = np.random.random(shape)

    if axes is not None:
        shape = np.take(shape, axes)

    y = forward(x, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    z = backward(y, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    assert_allclose(z, x)

    if axes is None:
        pad = [(0, 4)] * x.ndim
    elif isinstance(axes, int):
        pad = [(0, 0)] * x.ndim
        pad[axes] = (0, 4)
    else:
        pad = [(0, 0)] * x.ndim

        for a in axes:
            pad[a] = (0, 4)

    y2 = np.pad(y, pad, mode='edge')
    z2 = backward(y2, type, shape, axes, norm, orthogonalize=orthogonalize)
    assert_allclose(z2, x)


@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 5), 0),
                             ((4, 5), 1),
                             ((4, 5), None),
                         ])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("overwrite_x", [False, True])
def test_identity_nd_overwrite(forward, backward, type, shape, axes, dtype,
                               norm, overwrite_x):
    # Test the identity f^-1(f(x)) == x

    x = np.random.random(shape).astype(dtype)
    x_orig = x.copy()

    if axes is not None:
        shape = np.take(shape, axes)

    y = forward(x, type, axes=axes, norm=norm)
    y_orig = y.copy()
    z = backward(y, type, axes=axes, norm=norm)
    if overwrite_x:
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)
    else:
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)


@pytest.mark.parametrize("func", ['dct', 'dst', 'dctn', 'dstn'])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
def test_fftpack_equivalience(func, type, norm):
    x = np.random.rand(8, 16)
    fft_res = getattr(fft, func)(x, type, norm=norm)
    fftpack_res = getattr(fftpack, func)(x, type, norm=norm)

    assert_allclose(fft_res, fftpack_res)


@pytest.mark.parametrize("func", [dct, dst, dctn, dstn])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
def test_orthogonalize_default(func, type):
    # Test orthogonalize is the default when norm="ortho", but not otherwise
    x = np.random.rand(100)

    for norm, ortho in [
            ("forward", False),
            ("backward", False),
            ("ortho", True),
    ]:
        a = func(x, type=type, norm=norm, orthogonalize=ortho)
        b = func(x, type=type, norm=norm)
        assert_allclose(a, b)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func, type", [
    (dct, 4), (dst, 1), (dst, 4)])
def test_orthogonalize_noop(func, type, norm):
    # Transforms where orthogonalize is a no-op
    x = np.random.rand(100)
    y1 = func(x, type=type, norm=norm, orthogonalize=True)
    y2 = func(x, type=type, norm=norm, orthogonalize=False)
    assert_allclose(y1, y2)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_orthogonalize_dct1(norm):
    x = np.random.rand(100)

    x2 = x.copy()
    x2[0] *= SQRT_2
    x2[-1] *= SQRT_2

    y1 = dct(x, type=1, norm=norm, orthogonalize=True)
    y2 = dct(x2, type=1, norm=norm, orthogonalize=False)

    y2[0] /= SQRT_2
    y2[-1] /= SQRT_2
    assert_allclose(y1, y2)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func", [dct, dst])
def test_orthogonalize_dcst2(func, norm):
    x = np.random.rand(100)
    y1 = func(x, type=2, norm=norm, orthogonalize=True)
    y2 = func(x, type=2, norm=norm, orthogonalize=False)

    y2[0] /= SQRT_2
    assert_allclose(y1, y2)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("func", [dct, dst])
def test_orthogonalize_dcst3(func, norm):
    x = np.random.rand(100)
    x2 = x.copy()
    x2[0] *= SQRT_2

    y1 = func(x, type=3, norm=norm, orthogonalize=True)
    y2 = func(x2, type=3, norm=norm, orthogonalize=False)
    assert_allclose(y1, y2)
