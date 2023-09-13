# This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
'''
A unit test module for czt.py
'''
import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np


def check_czt(x):
    # Check that czt is the equivalent of normal fft
    y = fft(x)
    y1 = czt(x)
    assert_allclose(y1, y, rtol=1e-13)

    # Check that interpolated czt is the equivalent of normal fft
    y = fft(x, 100*len(x))
    y1 = czt(x, 100*len(x))
    assert_allclose(y1, y, rtol=1e-12)


def check_zoom_fft(x):
    # Check that zoom_fft is the equivalent of normal fft
    y = fft(x)
    y1 = zoom_fft(x, [0, 2-2./len(y)], endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, [0, 2])
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)

    # Test fn scalar
    y1 = zoom_fft(x, 2-2./len(y), endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, 2)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)

    # Check that zoom_fft with oversampling is equivalent to zero padding
    over = 10
    yover = fft(x, over*len(x))
    y2 = zoom_fft(x, [0, 2-2./len(yover)], m=len(yover), endpoint=True)
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)
    y2 = zoom_fft(x, [0, 2], m=len(yover))
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)

    # Check that zoom_fft works on a subrange
    w = np.linspace(0, 2-2./len(x), len(x))
    f1, f2 = w[3], w[6]
    y3 = zoom_fft(x, [f1, f2], m=3*over+1, endpoint=True)
    idx3 = slice(3*over, 6*over+1)
    assert_allclose(y3, yover[idx3], rtol=1e-13)


def test_1D():
    # Test of 1D version of the transforms

    np.random.seed(0)  # Deterministic randomness

    # Random signals
    lengths = np.random.randint(8, 200, 20)
    np.append(lengths, 1)
    for length in lengths:
        x = np.random.random(length)
        check_zoom_fft(x)
        check_czt(x)

    # Gauss
    t = np.linspace(-2, 2, 128)
    x = np.exp(-t**2/0.01)
    check_zoom_fft(x)

    # Linear
    x = [1, 2, 3, 4, 5, 6, 7]
    check_zoom_fft(x)

    # Check near powers of two
    check_zoom_fft(range(126-31))
    check_zoom_fft(range(127-31))
    check_zoom_fft(range(128-31))
    check_zoom_fft(range(129-31))
    check_zoom_fft(range(130-31))

    # Check transform on n-D array input
    x = np.reshape(np.arange(3*2*28), (3, 2, 28))
    y1 = zoom_fft(x, [0, 2-2./28])
    y2 = zoom_fft(x[2, 0, :], [0, 2-2./28])
    assert_allclose(y1[2, 0], y2, rtol=1e-13, atol=1e-12)

    y1 = zoom_fft(x, [0, 2], endpoint=False)
    y2 = zoom_fft(x[2, 0, :], [0, 2], endpoint=False)
    assert_allclose(y1[2, 0], y2, rtol=1e-13, atol=1e-12)

    # Random (not a test condition)
    x = np.random.rand(101)
    check_zoom_fft(x)

    # Spikes
    t = np.linspace(0, 1, 128)
    x = np.sin(2*np.pi*t*5)+np.sin(2*np.pi*t*13)
    check_zoom_fft(x)

    # Sines
    x = np.zeros(100, dtype=complex)
    x[[1, 5, 21]] = 1
    check_zoom_fft(x)

    # Sines plus complex component
    x += 1j*np.linspace(0, 0.5, x.shape[0])
    check_zoom_fft(x)


def test_large_prime_lengths():
    np.random.seed(0)  # Deterministic randomness
    for N in (101, 1009, 10007):
        x = np.random.rand(N)
        y = fft(x)
        y1 = czt(x)
        assert_allclose(y, y1, rtol=1e-12)


@pytest.mark.slow
def test_czt_vs_fft():
    np.random.seed(123)
    random_lengths = np.random.exponential(100000, size=10).astype('int')
    for n in random_lengths:
        a = np.random.randn(n)
        assert_allclose(czt(a), fft(a), rtol=1e-11)


def test_empty_input():
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt([])
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        zoom_fft([], 0.5)


def test_0_rank_input():
    with pytest.raises(IndexError, match='tuple index out of range'):
        czt(5)
    with pytest.raises(IndexError, match='tuple index out of range'):
        zoom_fft(5, 0.5)


@pytest.mark.parametrize('impulse', ([0, 0, 1], [0, 0, 1, 0, 0],
                                     np.concatenate((np.array([0, 0, 1]),
                                                     np.zeros(100)))))
@pytest.mark.parametrize('m', (1, 3, 5, 8, 101, 1021))
@pytest.mark.parametrize('a', (1, 2, 0.5, 1.1))
# Step that tests away from the unit circle, but not so far it explodes from
# numerical error
@pytest.mark.parametrize('w', (None, 0.98534 + 0.17055j))
def test_czt_math(impulse, m, w, a):
    # z-transform of an impulse is 1 everywhere
    assert_allclose(czt(impulse[2:], m=m, w=w, a=a),
                    np.ones(m), rtol=1e-10)

    # z-transform of a delayed impulse is z**-1
    assert_allclose(czt(impulse[1:], m=m, w=w, a=a),
                    czt_points(m=m, w=w, a=a)**-1, rtol=1e-10)

    # z-transform of a 2-delayed impulse is z**-2
    assert_allclose(czt(impulse, m=m, w=w, a=a),
                    czt_points(m=m, w=w, a=a)**-2, rtol=1e-10)


def test_int_args():
    # Integer argument `a` was producing all 0s
    assert_allclose(abs(czt([0, 1], m=10, a=2)), 0.5*np.ones(10), rtol=1e-15)
    assert_allclose(czt_points(11, w=2), 1/(2**np.arange(11)), rtol=1e-30)


def test_czt_points():
    for N in (1, 2, 3, 8, 11, 100, 101, 10007):
        assert_allclose(czt_points(N), np.exp(2j*np.pi*np.arange(N)/N),
                        rtol=1e-30)

    assert_allclose(czt_points(7, w=1), np.ones(7), rtol=1e-30)
    assert_allclose(czt_points(11, w=2.), 1/(2**np.arange(11)), rtol=1e-30)

    func = CZT(12, m=11, w=2., a=1)
    assert_allclose(func.points(), 1/(2**np.arange(11)), rtol=1e-30)


@pytest.mark.parametrize('cls, args', [(CZT, (100,)), (ZoomFFT, (100, 0.2))])
def test_CZT_size_mismatch(cls, args):
    # Data size doesn't match function's expected size
    myfunc = cls(*args)
    with pytest.raises(ValueError, match='CZT defined for'):
        myfunc(np.arange(5))


def test_invalid_range():
    with pytest.raises(ValueError, match='2-length sequence'):
        ZoomFFT(100, [1, 2, 3])


@pytest.mark.parametrize('m', [0, -11, 5.5, 4.0])
def test_czt_points_errors(m):
    # Invalid number of points
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt_points(m)


@pytest.mark.parametrize('size', [0, -5, 3.5, 4.0])
def test_nonsense_size(size):
    # Numpy and Scipy fft() give ValueError for 0 output size, so we do, too
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(size, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(size, 0.2, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(3, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(3, 0.2, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt([1, 2, 3], size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        zoom_fft([1, 2, 3], 0.2, size)
