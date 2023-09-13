# This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
"""
Chirp z-transform.

We provide two interfaces to the chirp z-transform: an object interface
which precalculates part of the transform and can be applied efficiently
to many different data sets, and a functional interface which is applied
only to the given data set.

Transforms
----------

CZT : callable (x, axis=-1) -> array
   Define a chirp z-transform that can be applied to different signals.
ZoomFFT : callable (x, axis=-1) -> array
   Define a Fourier transform on a range of frequencies.

Functions
---------

czt : array
   Compute the chirp z-transform for a signal.
zoom_fft : array
   Compute the Fourier transform on a range of frequencies.
"""

import cmath
import numbers
import numpy as np
from numpy import pi, arange
from scipy.fft import fft, ifft, next_fast_len

__all__ = ['czt', 'zoom_fft', 'CZT', 'ZoomFFT', 'czt_points']


def _validate_sizes(n, m):
    if n < 1 or not isinstance(n, numbers.Integral):
        raise ValueError('Invalid number of CZT data '
                         f'points ({n}) specified. '
                         'n must be positive and integer type.')

    if m is None:
        m = n
    elif m < 1 or not isinstance(m, numbers.Integral):
        raise ValueError('Invalid number of CZT output '
                         f'points ({m}) specified. '
                         'm must be positive and integer type.')

    return m


def czt_points(m, w=None, a=1+0j):
    """
    Return the points at which the chirp z-transform is computed.

    Parameters
    ----------
    m : int
        The number of points desired.
    w : complex, optional
        The ratio between points in each step.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    out : ndarray
        The points in the Z plane at which `CZT` samples the z-transform,
        when called with arguments `m`, `w`, and `a`, as complex numbers.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    czt : Convenience function for quickly calculating CZT.

    Examples
    --------
    Plot the points of a 16-point FFT:

    >>> import numpy as np
    >>> from scipy.signal import czt_points
    >>> points = czt_points(16)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()

    and a 91-point logarithmic spiral that crosses the unit circle:

    >>> m, w, a = 91, 0.995*np.exp(-1j*np.pi*.05), 0.8*np.exp(1j*np.pi/6)
    >>> points = czt_points(m, w, a)
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()
    """
    m = _validate_sizes(1, m)

    k = arange(m)

    a = 1.0 * a  # at least float

    if w is None:
        # Nothing specified, default to FFT
        return a * np.exp(2j * pi * k / m)
    else:
        # w specified
        w = 1.0 * w  # at least float
        return a * w**-k


class CZT:
    """
    Create a callable chirp z-transform function.

    Transform to compute the frequency response around a spiral.
    Objects of this class are callables which can compute the
    chirp z-transform on their inputs.  This object precalculates the constant
    chirps used in the given transform.

    Parameters
    ----------
    n : int
        The size of the signal.
    m : int, optional
        The number of output points desired.  Default is `n`.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    f : CZT
        Callable object ``f(x, axis=-1)`` for computing the chirp z-transform
        on `x`.

    See Also
    --------
    czt : Convenience function for quickly calculating CZT.
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``f(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, m)`` is equivalent to
    ``fft.fft(x, m)``.

    If `w` does not lie on the unit circle, then the transform will be
    around a spiral with exponentially-increasing radius.  Regardless,
    angle will increase linearly.

    For transforms that do lie on the unit circle, accuracy is better when
    using `ZoomFFT`, since any numerical error in `w` is
    accumulated for long data lengths, drifting away from the unit circle.

    The chirp z-transform can be faster than an equivalent FFT with
    zero padding.  Try it with your own array sizes to see.

    However, the chirp z-transform is considerably less precise than the
    equivalent zero-padded FFT.

    As this CZT is implemented using the Bluestein algorithm, it can compute
    large prime-length Fourier transforms in O(N log N) time, rather than the
    O(N**2) time required by the direct DFT calculation.  (`scipy.fft` also
    uses Bluestein's algorithm'.)

    (The name "chirp z-transform" comes from the use of a chirp in the
    Bluestein algorithm.  It does not decompose signals into chirps, like
    other transforms with "chirp" in the name.)

    References
    ----------
    .. [1] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).
    .. [2] Rabiner, Schafer, and Rader, "The chirp z-transform algorithm and
           its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    Examples
    --------
    Compute multiple prime-length FFTs:

    >>> from scipy.signal import CZT
    >>> import numpy as np
    >>> a = np.random.rand(7)
    >>> b = np.random.rand(7)
    >>> c = np.random.rand(7)
    >>> czt_7 = CZT(n=7)
    >>> A = czt_7(a)
    >>> B = czt_7(b)
    >>> C = czt_7(c)

    Display the points at which the FFT is calculated:

    >>> czt_7.points()
    array([ 1.00000000+0.j        ,  0.62348980+0.78183148j,
           -0.22252093+0.97492791j, -0.90096887+0.43388374j,
           -0.90096887-0.43388374j, -0.22252093-0.97492791j,
            0.62348980-0.78183148j])
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(czt_7.points().real, czt_7.points().imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()
    """

    def __init__(self, n, m=None, w=None, a=1+0j):
        m = _validate_sizes(n, m)

        k = arange(max(m, n), dtype=np.min_scalar_type(-max(m, n)**2))

        if w is None:
            # Nothing specified, default to FFT-like
            w = cmath.exp(-2j*pi/m)
            wk2 = np.exp(-(1j * pi * ((k**2) % (2*m))) / m)
        else:
            # w specified
            wk2 = w**(k**2/2.)

        a = 1.0 * a  # at least float

        self.w, self.a = w, a
        self.m, self.n = m, n

        nfft = next_fast_len(n + m - 1)
        self._Awk2 = a**-k[:n] * wk2[:n]
        self._nfft = nfft
        self._Fwk2 = fft(1/np.hstack((wk2[n-1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n-1, n+m-1)

    def __call__(self, x, *, axis=-1):
        """
        Calculate the chirp z-transform of a signal.

        Parameters
        ----------
        x : array
            The signal to transform.
        axis : int, optional
            Axis over which to compute the FFT. If not given, the last axis is
            used.

        Returns
        -------
        out : ndarray
            An array of the same dimensions as `x`, but with the length of the
            transformed axis set to `m`.
        """
        x = np.asarray(x)
        if x.shape[axis] != self.n:
            raise ValueError(f"CZT defined for length {self.n}, not "
                             f"{x.shape[axis]}")
        # Calculate transpose coordinates, to allow operation on any given axis
        trnsp = np.arange(x.ndim)
        trnsp[[axis, -1]] = [-1, axis]
        x = x.transpose(*trnsp)
        y = ifft(self._Fwk2 * fft(x*self._Awk2, self._nfft))
        y = y[..., self._yidx] * self._wk2
        return y.transpose(*trnsp)

    def points(self):
        """
        Return the points at which the chirp z-transform is computed.
        """
        return czt_points(self.m, self.w, self.a)


class ZoomFFT(CZT):
    """
    Create a callable zoom FFT transform function.

    This is a specialization of the chirp z-transform (`CZT`) for a set of
    equally-spaced frequencies around the unit circle, used to calculate a
    section of the FFT more efficiently than calculating the entire FFT and
    truncating.

    Parameters
    ----------
    n : int
        The size of the signal.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  Default is `n`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    f : ZoomFFT
        Callable object ``f(x, axis=-1)`` for computing the zoom FFT on `x`.

    See Also
    --------
    zoom_fft : Convenience function for calculating a zoom FFT.

    Notes
    -----
    The defaults are chosen such that ``f(x, 2)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, 2, m)`` is equivalent to
    ``fft.fft(x, m)``.

    Sampling frequency is 1/dt, the time step between samples in the
    signal `x`.  The unit circle corresponds to frequencies from 0 up
    to the sampling frequency.  The default sampling frequency of 2
    means that `f1`, `f2` values up to the Nyquist frequency are in the
    range [0, 1). For `f1`, `f2` values expressed in radians, a sampling
    frequency of 2*pi should be used.

    Remember that a zoom FFT can only interpolate the points of the existing
    FFT.  It cannot help to resolve two separate nearby frequencies.
    Frequency resolution can only be increased by increasing acquisition
    time.

    These functions are implemented using Bluestein's algorithm (as is
    `scipy.fft`). [2]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 29 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf
    .. [2] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).

    Examples
    --------
    To plot the transform results use something like the following:

    >>> import numpy as np
    >>> from scipy.signal import ZoomFFT
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> transform = ZoomFFT(len(x), [f1, f2], len(x), fs=1021)
    >>> X = transform(x)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """

    def __init__(self, n, fn, m=None, *, fs=2, endpoint=False):
        m = _validate_sizes(n, m)

        k = arange(max(m, n), dtype=np.min_scalar_type(-max(m, n)**2))

        if np.size(fn) == 2:
            f1, f2 = fn
        elif np.size(fn) == 1:
            f1, f2 = 0.0, fn
        else:
            raise ValueError('fn must be a scalar or 2-length sequence')

        self.f1, self.f2, self.fs = f1, f2, fs

        if endpoint:
            scale = ((f2 - f1) * m) / (fs * (m - 1))
        else:
            scale = (f2 - f1) / fs
        a = cmath.exp(2j * pi * f1/fs)
        wk2 = np.exp(-(1j * pi * scale * k**2) / m)

        self.w = cmath.exp(-2j*pi/m * scale)
        self.a = a
        self.m, self.n = m, n

        ak = np.exp(-2j * pi * f1/fs * k[:n])
        self._Awk2 = ak * wk2[:n]

        nfft = next_fast_len(n + m - 1)
        self._nfft = nfft
        self._Fwk2 = fft(1/np.hstack((wk2[n-1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n-1, n+m-1)


def czt(x, m=None, w=None, a=1+0j, *, axis=-1):
    """
    Compute the frequency response around a spiral in the Z plane.

    Parameters
    ----------
    x : array
        The signal to transform.
    m : int, optional
        The number of output points desired.  Default is the length of the
        input data.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        An array of the same dimensions as `x`, but with the length of the
        transformed axis set to `m`.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    zoom_fft : Convenience function for partial FFT calculations.

    Notes
    -----
    The defaults are chosen such that ``signal.czt(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.czt(x, m)`` is
    equivalent to ``fft.fft(x, m)``.

    If the transform needs to be repeated, use `CZT` to construct a
    specialized transform function which can be reused without
    recomputing constants.

    An example application is in system identification, repeatedly evaluating
    small slices of the z-transform of a system, around where a pole is
    expected to exist, to refine the estimate of the pole's true location. [1]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 20 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf

    Examples
    --------
    Generate a sinusoid:

    >>> import numpy as np
    >>> f1, f2, fs = 8, 10, 200  # Hz
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = np.sin(2*np.pi*t*f2)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    Its discrete Fourier transform has all of its energy in a single frequency
    bin:

    >>> from scipy.fft import rfft, rfftfreq
    >>> from scipy.signal import czt, czt_points
    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    However, if the sinusoid is logarithmically-decaying:

    >>> x = np.exp(-t*f1) * np.sin(2*np.pi*t*f2)
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    the DFT will have spectral leakage:

    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    While the DFT always samples the z-transform around the unit circle, the
    chirp z-transform allows us to sample the Z-transform along any
    logarithmic spiral, such as a circle with radius smaller than unity:

    >>> M = fs // 2  # Just positive frequencies, like rfft
    >>> a = np.exp(-f1/fs)  # Starting point of the circle, radius < 1
    >>> w = np.exp(-1j*np.pi/M)  # "Step size" of circle
    >>> points = czt_points(M + 1, w, a)  # M + 1 to include Nyquist
    >>> plt.plot(points.real, points.imag, '.')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal'); plt.axis([-1.05, 1.05, -0.05, 1.05])
    >>> plt.show()

    With the correct radius, this transforms the decaying sinusoid (and others
    with the same decay rate) without spectral leakage:

    >>> z_vals = czt(x, M + 1, w, a)  # Include Nyquist for comparison to rfft
    >>> freqs = np.angle(points)*fs/(2*np.pi)  # angle = omega, radius = sigma
    >>> plt.plot(freqs, abs(z_vals))
    >>> plt.margins(0, 0.1)
    >>> plt.show()
    """
    x = np.asarray(x)
    transform = CZT(x.shape[axis], m=m, w=w, a=a)
    return transform(x, axis=axis)


def zoom_fft(x, fn, m=None, *, fs=2, endpoint=False, axis=-1):
    """
    Compute the DFT of `x` only for frequencies in range `fn`.

    Parameters
    ----------
    x : array
        The signal to transform.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  The default is the length of `x`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        The transformed signal.  The Fourier transform will be calculated
        at the points f1, f1+df, f1+2df, ..., f2, where df=(f2-f1)/m.

    See Also
    --------
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``signal.zoom_fft(x, 2)`` is equivalent
    to ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.zoom_fft(x, 2, m)``
    is equivalent to ``fft.fft(x, m)``.

    To graph the magnitude of the resulting transform, use::

        plot(linspace(f1, f2, m, endpoint=False), abs(zoom_fft(x, [f1, f2], m)))

    If the transform needs to be repeated, use `ZoomFFT` to construct
    a specialized transform function which can be reused without
    recomputing constants.

    Examples
    --------
    To plot the transform results use something like the following:

    >>> import numpy as np
    >>> from scipy.signal import zoom_fft
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> X = zoom_fft(x, [f1, f2], len(x), fs=1021)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """
    x = np.asarray(x)
    transform = ZoomFFT(x.shape[axis], fn, m=m, fs=fs, endpoint=endpoint)
    return transform(x, axis=axis)
