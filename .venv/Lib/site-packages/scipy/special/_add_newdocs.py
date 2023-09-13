# Docstrings for generated ufuncs
#
# The syntax is designed to look like the function add_newdoc is being
# called from numpy.lib, but in this file add_newdoc puts the
# docstrings in a dictionary. This dictionary is used in
# _generate_pyx.py to generate the docstrings for the ufuncs in
# scipy.special at the C level when the ufuncs are created at compile
# time.

docdict: dict[str, str] = {}


def get(name):
    return docdict.get(name)


def add_newdoc(name, doc):
    docdict[name] = doc


add_newdoc("_sf_error_test_function",
    """
    Private function; do not use.
    """)


add_newdoc("_cosine_cdf",
    """
    _cosine_cdf(x)

    Cumulative distribution function (CDF) of the cosine distribution::

                 {             0,              x < -pi
        cdf(x) = { (pi + x + sin(x))/(2*pi),   -pi <= x <= pi
                 {             1,              x > pi

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    scalar or ndarray
        The cosine distribution CDF evaluated at `x`.

    """)

add_newdoc("_cosine_invcdf",
    """
    _cosine_invcdf(p)

    Inverse of the cumulative distribution function (CDF) of the cosine
    distribution.

    The CDF of the cosine distribution is::

        cdf(x) = (pi + x + sin(x))/(2*pi)

    This function computes the inverse of cdf(x).

    Parameters
    ----------
    p : array_like
        `p` must contain real numbers in the interval ``0 <= p <= 1``.
        `nan` is returned for values of `p` outside the interval [0, 1].

    Returns
    -------
    scalar or ndarray
        The inverse of the cosine distribution CDF evaluated at `p`.

    """)

add_newdoc("sph_harm",
    r"""
    sph_harm(m, n, theta, phi, out=None)

    Compute spherical harmonics.

    The spherical harmonics are defined as

    .. math::

        Y^m_n(\theta,\phi) = \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}}
          e^{i m \theta} P^m_n(\cos(\phi))

    where :math:`P_n^m` are the associated Legendre functions; see `lpmv`.

    Parameters
    ----------
    m : array_like
        Order of the harmonic (int); must have ``|m| <= n``.
    n : array_like
       Degree of the harmonic (int); must have ``n >= 0``. This is
       often denoted by ``l`` (lower case L) in descriptions of
       spherical harmonics.
    theta : array_like
       Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    phi : array_like
       Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y_mn : complex scalar or ndarray
       The harmonic :math:`Y^m_n` sampled at ``theta`` and ``phi``.

    Notes
    -----
    There are different conventions for the meanings of the input
    arguments ``theta`` and ``phi``. In SciPy ``theta`` is the
    azimuthal angle and ``phi`` is the polar angle. It is common to
    see the opposite convention, that is, ``theta`` as the polar angle
    and ``phi`` as the azimuthal angle.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `lpmv`.

    With SciPy's conventions, the first several spherical harmonics
    are

    .. math::

        Y_0^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{1}{\pi}} \\
        Y_1^{-1}(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                    e^{-i\theta} \sin(\phi) \\
        Y_1^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{\pi}}
                                 \cos(\phi) \\
        Y_1^1(\theta, \phi) &= -\frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                 e^{i\theta} \sin(\phi).

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 14.30.
           https://dlmf.nist.gov/14.30
    .. [2] https://en.wikipedia.org/wiki/Spherical_harmonics#Condon.E2.80.93Shortley_phase
    """)

add_newdoc("_ellip_harm",
    """
    Internal function, use `ellip_harm` instead.
    """)

add_newdoc("_ellip_norm",
    """
    Internal function, use `ellip_norm` instead.
    """)

add_newdoc("_lambertw",
    """
    Internal function, use `lambertw` instead.
    """)

add_newdoc("voigt_profile",
    r"""
    voigt_profile(x, sigma, gamma, out=None)

    Voigt profile.

    The Voigt profile is a convolution of a 1-D Normal distribution with
    standard deviation ``sigma`` and a 1-D Cauchy distribution with half-width at
    half-maximum ``gamma``.

    If ``sigma = 0``, PDF of Cauchy distribution is returned.
    Conversely, if ``gamma = 0``, PDF of Normal distribution is returned.
    If ``sigma = gamma = 0``, the return value is ``Inf`` for ``x = 0``, and ``0`` for all other ``x``.

    Parameters
    ----------
    x : array_like
        Real argument
    sigma : array_like
        The standard deviation of the Normal distribution part
    gamma : array_like
        The half-width at half-maximum of the Cauchy distribution part
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The Voigt profile at the given arguments

    Notes
    -----
    It can be expressed in terms of Faddeeva function

    .. math:: V(x; \sigma, \gamma) = \frac{Re[w(z)]}{\sigma\sqrt{2\pi}},
    .. math:: z = \frac{x + i\gamma}{\sqrt{2}\sigma}

    where :math:`w(z)` is the Faddeeva function.

    See Also
    --------
    wofz : Faddeeva function

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Voigt_profile

    Examples
    --------
    Calculate the function at point 2 for ``sigma=1`` and ``gamma=1``.

    >>> from scipy.special import voigt_profile
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> voigt_profile(2, 1., 1.)
    0.09071519942627544

    Calculate the function at several points by providing a NumPy array
    for `x`.

    >>> values = np.array([-2., 0., 5])
    >>> voigt_profile(values, 1., 1.)
    array([0.0907152 , 0.20870928, 0.01388492])

    Plot the function for different parameter sets.

    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> x = np.linspace(-10, 10, 500)
    >>> parameters_list = [(1.5, 0., "solid"), (1.3, 0.5, "dashed"),
    ...                    (0., 1.8, "dotted"), (1., 1., "dashdot")]
    >>> for params in parameters_list:
    ...     sigma, gamma, linestyle = params
    ...     voigt = voigt_profile(x, sigma, gamma)
    ...     ax.plot(x, voigt, label=rf"$\sigma={sigma},\, \gamma={gamma}$",
    ...             ls=linestyle)
    >>> ax.legend()
    >>> plt.show()

    Verify visually that the Voigt profile indeed arises as the convolution
    of a normal and a Cauchy distribution.

    >>> from scipy.signal import convolve
    >>> x, dx = np.linspace(-10, 10, 500, retstep=True)
    >>> def gaussian(x, sigma):
    ...     return np.exp(-0.5 * x**2/sigma**2)/(sigma * np.sqrt(2*np.pi))
    >>> def cauchy(x, gamma):
    ...     return gamma/(np.pi * (np.square(x)+gamma**2))
    >>> sigma = 2
    >>> gamma = 1
    >>> gauss_profile = gaussian(x, sigma)
    >>> cauchy_profile = cauchy(x, gamma)
    >>> convolved = dx * convolve(cauchy_profile, gauss_profile, mode="same")
    >>> voigt = voigt_profile(x, sigma, gamma)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> ax.plot(x, gauss_profile, label="Gauss: $G$", c='b')
    >>> ax.plot(x, cauchy_profile, label="Cauchy: $C$", c='y', ls="dashed")
    >>> xx = 0.5*(x[1:] + x[:-1])  # midpoints
    >>> ax.plot(xx, convolved[1:], label="Convolution: $G * C$", ls='dashdot',
    ...         c='k')
    >>> ax.plot(x, voigt, label="Voigt", ls='dotted', c='r')
    >>> ax.legend()
    >>> plt.show()
    """)

add_newdoc("wrightomega",
    r"""
    wrightomega(z, out=None)

    Wright Omega function.

    Defined as the solution to

    .. math::

        \omega + \log(\omega) = z

    where :math:`\log` is the principal branch of the complex logarithm.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate the Wright Omega function
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    omega : scalar or ndarray
        Values of the Wright Omega function

    Notes
    -----
    .. versionadded:: 0.19.0

    The function can also be defined as

    .. math::

        \omega(z) = W_{K(z)}(e^z)

    where :math:`K(z) = \lceil (\Im(z) - \pi)/(2\pi) \rceil` is the
    unwinding number and :math:`W` is the Lambert W function.

    The implementation here is taken from [1]_.

    See Also
    --------
    lambertw : The Lambert W function

    References
    ----------
    .. [1] Lawrence, Corless, and Jeffrey, "Algorithm 917: Complex
           Double-Precision Evaluation of the Wright :math:`\omega`
           Function." ACM Transactions on Mathematical Software,
           2012. :doi:`10.1145/2168773.2168779`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import wrightomega, lambertw

    >>> wrightomega([-2, -1, 0, 1, 2])
    array([0.12002824, 0.27846454, 0.56714329, 1.        , 1.5571456 ])

    Complex input:

    >>> wrightomega(3 + 5j)
    (1.5804428632097158+3.8213626783287937j)

    Verify that ``wrightomega(z)`` satisfies ``w + log(w) = z``:

    >>> w = -5 + 4j
    >>> wrightomega(w + np.log(w))
    (-5+4j)

    Verify the connection to ``lambertw``:

    >>> z = 0.5 + 3j
    >>> wrightomega(z)
    (0.0966015889280649+1.4937828458191993j)
    >>> lambertw(np.exp(z))
    (0.09660158892806493+1.4937828458191993j)

    >>> z = 0.5 + 4j
    >>> wrightomega(z)
    (-0.3362123489037213+2.282986001579032j)
    >>> lambertw(np.exp(z), k=1)
    (-0.33621234890372115+2.282986001579032j)
    """)


add_newdoc("agm",
    """
    agm(a, b, out=None)

    Compute the arithmetic-geometric mean of `a` and `b`.

    Start with a_0 = a and b_0 = b and iteratively compute::

        a_{n+1} = (a_n + b_n)/2
        b_{n+1} = sqrt(a_n*b_n)

    a_n and b_n converge to the same limit as n increases; their common
    limit is agm(a, b).

    Parameters
    ----------
    a, b : array_like
        Real values only. If the values are both negative, the result
        is negative. If one value is negative and the other is positive,
        `nan` is returned.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The arithmetic-geometric mean of `a` and `b`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import agm
    >>> a, b = 24.0, 6.0
    >>> agm(a, b)
    13.458171481725614

    Compare that result to the iteration:

    >>> while a != b:
    ...     a, b = (a + b)/2, np.sqrt(a*b)
    ...     print("a = %19.16f  b=%19.16f" % (a, b))
    ...
    a = 15.0000000000000000  b=12.0000000000000000
    a = 13.5000000000000000  b=13.4164078649987388
    a = 13.4582039324993694  b=13.4581390309909850
    a = 13.4581714817451772  b=13.4581714817060547
    a = 13.4581714817256159  b=13.4581714817256159

    When array-like arguments are given, broadcasting applies:

    >>> a = np.array([[1.5], [3], [6]])  # a has shape (3, 1).
    >>> b = np.array([6, 12, 24, 48])    # b has shape (4,).
    >>> agm(a, b)
    array([[  3.36454287,   5.42363427,   9.05798751,  15.53650756],
           [  4.37037309,   6.72908574,  10.84726853,  18.11597502],
           [  6.        ,   8.74074619,  13.45817148,  21.69453707]])
    """)

add_newdoc("airy",
    r"""
    airy(z, out=None)

    Airy functions and their derivatives.

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    Ai, Aip, Bi, Bip : 4-tuple of scalar or ndarray
        Airy functions Ai and Bi, and their derivatives Aip and Bip.

    Notes
    -----
    The Airy functions Ai and Bi are two independent solutions of

    .. math:: y''(x) = x y(x).

    For real `z` in [-10, 10], the computation is carried out by calling
    the Cephes [1]_ `airy` routine, which uses power series summation
    for small `z` and rational minimax approximations for large `z`.

    Outside this range, the AMOS [2]_ `zairy` and `zbiry` routines are
    employed.  They are computed using power series for :math:`|z| < 1` and
    the following relations to modified Bessel functions for larger `z`
    (where :math:`t \equiv 2 z^{3/2}/3`):

    .. math::

        Ai(z) = \frac{1}{\pi \sqrt{3}} K_{1/3}(t)

        Ai'(z) = -\frac{z}{\pi \sqrt{3}} K_{2/3}(t)

        Bi(z) = \sqrt{\frac{z}{3}} \left(I_{-1/3}(t) + I_{1/3}(t) \right)

        Bi'(z) = \frac{z}{\sqrt{3}} \left(I_{-2/3}(t) + I_{2/3}(t)\right)

    See also
    --------
    airye : exponentially scaled Airy functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compute the Airy functions on the interval [-15, 5].

    >>> import numpy as np
    >>> from scipy import special
    >>> x = np.linspace(-15, 5, 201)
    >>> ai, aip, bi, bip = special.airy(x)

    Plot Ai(x) and Bi(x).

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, ai, 'r', label='Ai(x)')
    >>> plt.plot(x, bi, 'b--', label='Bi(x)')
    >>> plt.ylim(-0.5, 1.0)
    >>> plt.grid()
    >>> plt.legend(loc='upper left')
    >>> plt.show()

    """)

add_newdoc("airye",
    """
    airye(z, out=None)

    Exponentially scaled Airy functions and their derivatives.

    Scaling::

        eAi  = Ai  * exp(2.0/3.0*z*sqrt(z))
        eAip = Aip * exp(2.0/3.0*z*sqrt(z))
        eBi  = Bi  * exp(-abs(2.0/3.0*(z*sqrt(z)).real))
        eBip = Bip * exp(-abs(2.0/3.0*(z*sqrt(z)).real))

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    eAi, eAip, eBi, eBip : 4-tuple of scalar or ndarray
        Exponentially scaled Airy functions eAi and eBi, and their derivatives
        eAip and eBip

    Notes
    -----
    Wrapper for the AMOS [1]_ routines `zairy` and `zbiry`.

    See also
    --------
    airy

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    We can compute exponentially scaled Airy functions and their derivatives:

    >>> import numpy as np
    >>> from scipy.special import airye
    >>> import matplotlib.pyplot as plt
    >>> z = np.linspace(0, 50, 500)
    >>> eAi, eAip, eBi, eBip = airye(z)
    >>> f, ax = plt.subplots(2, 1, sharex=True)
    >>> for ind, data in enumerate([[eAi, eAip, ["eAi", "eAip"]],
    ...                             [eBi, eBip, ["eBi", "eBip"]]]):
    ...     ax[ind].plot(z, data[0], "-r", z, data[1], "-b")
    ...     ax[ind].legend(data[2])
    ...     ax[ind].grid(True)
    >>> plt.show()

    We can compute these using usual non-scaled Airy functions by:

    >>> from scipy.special import airy
    >>> Ai, Aip, Bi, Bip = airy(z)
    >>> np.allclose(eAi, Ai * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eAip, Aip * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eBi, Bi * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True
    >>> np.allclose(eBip, Bip * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True

    Comparing non-scaled and exponentially scaled ones, the usual non-scaled
    function quickly underflows for large values, whereas the exponentially
    scaled function does not.

    >>> airy(200)
    (0.0, 0.0, nan, nan)
    >>> airye(200)
    (0.07501041684381093, -1.0609012305109042, 0.15003188417418148, 2.1215836725571093)

    """)

add_newdoc("bdtr",
    r"""
    bdtr(k, n, p, out=None)

    Binomial distribution cumulative distribution function.

    Sum of the terms 0 through `floor(k)` of the Binomial probability density.

    .. math::
        \mathrm{bdtr}(k, n, p) = \sum_{j=0}^{\lfloor k \rfloor} {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        Number of successes (double), rounded down to the nearest integer.
    n : array_like
        Number of events (int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Probability of `floor(k)` or fewer successes in `n` independent events with
        success probabilities of `p`.

    Notes
    -----
    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{bdtr}(k, n, p) = I_{1 - p}(n - \lfloor k \rfloor, \lfloor k \rfloor + 1).

    Wrapper for the Cephes [1]_ routine `bdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("bdtrc",
    r"""
    bdtrc(k, n, p, out=None)

    Binomial distribution survival function.

    Sum of the terms `floor(k) + 1` through `n` of the binomial probability
    density,

    .. math::
        \mathrm{bdtrc}(k, n, p) = \sum_{j=\lfloor k \rfloor +1}^n {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        Number of successes (double), rounded down to nearest integer.
    n : array_like
        Number of events (int)
    p : array_like
        Probability of success in a single event.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Probability of `floor(k) + 1` or more successes in `n` independent
        events with success probabilities of `p`.

    See also
    --------
    bdtr
    betainc

    Notes
    -----
    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{bdtrc}(k, n, p) = I_{p}(\lfloor k \rfloor + 1, n - \lfloor k \rfloor).

    Wrapper for the Cephes [1]_ routine `bdtrc`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("bdtri",
    r"""
    bdtri(k, n, y, out=None)

    Inverse function to `bdtr` with respect to `p`.

    Finds the event probability `p` such that the sum of the terms 0 through
    `k` of the binomial probability density is equal to the given cumulative
    probability `y`.

    Parameters
    ----------
    k : array_like
        Number of successes (float), rounded down to the nearest integer.
    n : array_like
        Number of events (float)
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    p : scalar or ndarray
        The event probability such that `bdtr(\lfloor k \rfloor, n, p) = y`.

    See also
    --------
    bdtr
    betaincinv

    Notes
    -----
    The computation is carried out using the inverse beta integral function
    and the relation,::

        1 - p = betaincinv(n - k, k + 1, y).

    Wrapper for the Cephes [1]_ routine `bdtri`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("bdtrik",
    """
    bdtrik(y, n, p, out=None)

    Inverse function to `bdtr` with respect to `k`.

    Finds the number of successes `k` such that the sum of the terms 0 through
    `k` of the Binomial probability density for `n` events with probability
    `p` is equal to the given cumulative probability `y`.

    Parameters
    ----------
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    n : array_like
        Number of events (float).
    p : array_like
        Success probability (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    k : scalar or ndarray
        The number of successes `k` such that `bdtr(k, n, p) = y`.

    See also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `k`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.

    """)

add_newdoc("bdtrin",
    """
    bdtrin(k, y, p, out=None)

    Inverse function to `bdtr` with respect to `n`.

    Finds the number of events `n` such that the sum of the terms 0 through
    `k` of the Binomial probability density for events with probability `p` is
    equal to the given cumulative probability `y`.

    Parameters
    ----------
    k : array_like
        Number of successes (float).
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    p : array_like
        Success probability (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    n : scalar or ndarray
        The number of events `n` such that `bdtr(k, n, p) = y`.

    See also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `n`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    """)

add_newdoc(
    "binom",
    r"""
    binom(x, y, out=None)

    Binomial coefficient considered as a function of two real variables.

    For real arguments, the binomial coefficient is defined as

    .. math::

        \binom{x}{y} = \frac{\Gamma(x + 1)}{\Gamma(y + 1)\Gamma(x - y + 1)} =
            \frac{1}{(x + 1)\mathrm{B}(x - y + 1, y + 1)}

    Where :math:`\Gamma` is the Gamma function (`gamma`) and :math:`\mathrm{B}`
    is the Beta function (`beta`) [1]_.

    Parameters
    ----------
    x, y: array_like
       Real arguments to :math:`\binom{x}{y}`.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Value of binomial coefficient.

    See Also
    --------
    comb : The number of combinations of N things taken k at a time.

    Notes
    -----
    The Gamma function has poles at non-positive integers and tends to either
    positive or negative infinity depending on the direction on the real line
    from which a pole is approached. When considered as a function of two real
    variables, :math:`\binom{x}{y}` is thus undefined when `x` is a negative
    integer.  `binom` returns ``nan`` when ``x`` is a negative integer. This
    is the case even when ``x`` is a negative integer and ``y`` an integer,
    contrary to the usual convention for defining :math:`\binom{n}{k}` when it
    is considered as a function of two integer variables.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_coefficient

    Examples
    --------
    The following examples illustrate the ways in which `binom` differs from
    the function `comb`.

    >>> from scipy.special import binom, comb

    When ``exact=False`` and ``x`` and ``y`` are both positive, `comb` calls
    `binom` internally.

    >>> x, y = 3, 2
    >>> (binom(x, y), comb(x, y), comb(x, y, exact=True))
    (3.0, 3.0, 3)

    For larger values, `comb` with ``exact=True`` no longer agrees
    with `binom`.

    >>> x, y = 43, 23
    >>> (binom(x, y), comb(x, y), comb(x, y, exact=True))
    (960566918219.9999, 960566918219.9999, 960566918220)

    `binom` returns ``nan`` when ``x`` is a negative integer, but is otherwise
    defined for negative arguments. `comb` returns 0 whenever one of ``x`` or
    ``y`` is negative or ``x`` is less than ``y``.

    >>> x, y = -3, 2
    >>> (binom(x, y), comb(x, y), comb(x, y, exact=True))
    (nan, 0.0, 0)

    >>> x, y = -3.1, 2.2
    >>> (binom(x, y), comb(x, y), comb(x, y, exact=True))
    (18.714147876804432, 0.0, 0)

    >>> x, y = 2.2, 3.1
    >>> (binom(x, y), comb(x, y), comb(x, y, exact=True))
    (0.037399983365134115, 0.0, 0)
    """
)

add_newdoc("btdtria",
    r"""
    btdtria(p, b, x, out=None)

    Inverse of `btdtr` with respect to `a`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `a`, returning the value of `a` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    p : array_like
        Cumulative probability, in [0, 1].
    b : array_like
        Shape parameter (`b` > 0).
    x : array_like
        The quantile, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    a : scalar or ndarray
        The value of the shape parameter `a` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtrib : Inverse with respect to `b`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `a` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `a`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.

    """)

add_newdoc("btdtrib",
    r"""
    btdtria(a, p, x, out=None)

    Inverse of `btdtr` with respect to `b`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `b`, returning the value of `b` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    a : array_like
        Shape parameter (`a` > 0).
    p : array_like
        Cumulative probability, in [0, 1].
    x : array_like
        The quantile, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    b : scalar or ndarray
        The value of the shape parameter `b` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtria : Inverse with respect to `a`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `b` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `b`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.


    """)

add_newdoc("bei",
    r"""
    bei(x, out=None)

    Kelvin function bei.

    Defined as

    .. math::

        \mathrm{bei}(x) = \Im[J_0(x e^{3 \pi i / 4})]

    where :math:`J_0` is the Bessel function of the first kind of
    order zero (see `jv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    ber : the corresponding real part
    beip : the derivative of bei
    jv : Bessel function of the first kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using Bessel functions.

    >>> import numpy as np
    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.jv(0, x * np.exp(3 * np.pi * 1j / 4)).imag
    array([0.24956604, 0.97229163, 1.93758679, 2.29269032])
    >>> sc.bei(x)
    array([0.24956604, 0.97229163, 1.93758679, 2.29269032])

    """)

add_newdoc("beip",
    r"""
    beip(x, out=None)

    Derivative of the Kelvin function bei.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of bei.

    See Also
    --------
    bei

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("ber",
    r"""
    ber(x, out=None)

    Kelvin function ber.

    Defined as

    .. math::

        \mathrm{ber}(x) = \Re[J_0(x e^{3 \pi i / 4})]

    where :math:`J_0` is the Bessel function of the first kind of
    order zero (see `jv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    bei : the corresponding real part
    berp : the derivative of bei
    jv : Bessel function of the first kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using Bessel functions.

    >>> import numpy as np
    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.jv(0, x * np.exp(3 * np.pi * 1j / 4)).real
    array([ 0.98438178,  0.75173418, -0.22138025, -2.56341656])
    >>> sc.ber(x)
    array([ 0.98438178,  0.75173418, -0.22138025, -2.56341656])

    """)

add_newdoc("berp",
    r"""
    berp(x, out=None)

    Derivative of the Kelvin function ber.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of ber.

    See Also
    --------
    ber

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("besselpoly",
    r"""
    besselpoly(a, lmb, nu, out=None)

    Weighted integral of the Bessel function of the first kind.

    Computes

    .. math::

       \int_0^1 x^\lambda J_\nu(2 a x) \, dx

    where :math:`J_\nu` is a Bessel function and :math:`\lambda=lmb`,
    :math:`\nu=nu`.

    Parameters
    ----------
    a : array_like
        Scale factor inside the Bessel function.
    lmb : array_like
        Power of `x`
    nu : array_like
        Order of the Bessel function.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Value of the integral.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Evaluate the function for one parameter set.

    >>> from scipy.special import besselpoly
    >>> besselpoly(1, 1, 1)
    0.24449718372863877

    Evaluate the function for different scale factors.

    >>> import numpy as np
    >>> factors = np.array([0., 3., 6.])
    >>> besselpoly(factors, 1, 1)
    array([ 0.        , -0.00549029,  0.00140174])

    Plot the function for varying powers, orders and scales.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> powers = np.linspace(0, 10, 100)
    >>> orders = [1, 2, 3]
    >>> scales = [1, 2]
    >>> all_combinations = [(order, scale) for order in orders
    ...                     for scale in scales]
    >>> for order, scale in all_combinations:
    ...     ax.plot(powers, besselpoly(scale, powers, order),
    ...             label=rf"$\nu={order}, a={scale}$")
    >>> ax.legend()
    >>> ax.set_xlabel(r"$\lambda$")
    >>> ax.set_ylabel(r"$\int_0^1 x^{\lambda} J_{\nu}(2ax)\,dx$")
    >>> plt.show()
    """)

add_newdoc("beta",
    r"""
    beta(a, b, out=None)

    Beta function.

    This function is defined in [1]_ as

    .. math::

        B(a, b) = \int_0^1 t^{a-1}(1-t)^{b-1}dt
                = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a, b : array_like
        Real-valued arguments
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the beta function

    See Also
    --------
    gamma : the gamma function
    betainc :  the regularized incomplete beta function
    betaln : the natural logarithm of the absolute
             value of the beta function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions,
           Eq. 5.12.1. https://dlmf.nist.gov/5.12

    Examples
    --------
    >>> import scipy.special as sc

    The beta function relates to the gamma function by the
    definition given above:

    >>> sc.beta(2, 3)
    0.08333333333333333
    >>> sc.gamma(2)*sc.gamma(3)/sc.gamma(2 + 3)
    0.08333333333333333

    As this relationship demonstrates, the beta function
    is symmetric:

    >>> sc.beta(1.7, 2.4)
    0.16567527689031739
    >>> sc.beta(2.4, 1.7)
    0.16567527689031739

    This function satisfies :math:`B(1, b) = 1/b`:

    >>> sc.beta(1, 4)
    0.25

    """)

add_newdoc("betainc",
    r"""
    betainc(a, b, x, out=None)

    Regularized incomplete beta function.

    Computes the regularized incomplete beta function, defined as [1]_:

    .. math::

        I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^x
        t^{a-1}(1-t)^{b-1}dt,

    for :math:`0 \leq x \leq 1`.

    Parameters
    ----------
    a, b : array_like
           Positive, real-valued parameters
    x : array_like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Value of the regularized incomplete beta function

    See Also
    --------
    beta : beta function
    betaincinv : inverse of the regularized incomplete beta function

    Notes
    -----
    The term *regularized* in the name of this function refers to the
    scaling of the function by the gamma function terms shown in the
    formula.  When not qualified as *regularized*, the name *incomplete
    beta function* often refers to just the integral expression,
    without the gamma terms.  One can use the function `beta` from
    `scipy.special` to get this "nonregularized" incomplete beta
    function by multiplying the result of ``betainc(a, b, x)`` by
    ``beta(a, b)``.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------

    Let :math:`B(a, b)` be the `beta` function.

    >>> import scipy.special as sc

    The coefficient in terms of `gamma` is equal to
    :math:`1/B(a, b)`. Also, when :math:`x=1`
    the integral is equal to :math:`B(a, b)`.
    Therefore, :math:`I_{x=1}(a, b) = 1` for any :math:`a, b`.

    >>> sc.betainc(0.2, 3.5, 1.0)
    1.0

    It satisfies
    :math:`I_x(a, b) = x^a F(a, 1-b, a+1, x)/ (aB(a, b))`,
    where :math:`F` is the hypergeometric function `hyp2f1`:

    >>> a, b, x = 1.4, 3.1, 0.5
    >>> x**a * sc.hyp2f1(a, 1 - b, a + 1, x)/(a * sc.beta(a, b))
    0.8148904036225295
    >>> sc.betainc(a, b, x)
    0.8148904036225296

    This functions satisfies the relationship
    :math:`I_x(a, b) = 1 - I_{1-x}(b, a)`:

    >>> sc.betainc(2.2, 3.1, 0.4)
    0.49339638807619446
    >>> 1 - sc.betainc(3.1, 2.2, 1 - 0.4)
    0.49339638807619446

    """)

add_newdoc("betaincinv",
    r"""
    betaincinv(a, b, y, out=None)

    Inverse of the regularized incomplete beta function.

    Computes :math:`x` such that:

    .. math::

        y = I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
        \int_0^x t^{a-1}(1-t)^{b-1}dt,

    where :math:`I_x` is the normalized incomplete beta
    function `betainc` and
    :math:`\Gamma` is the `gamma` function [1]_.

    Parameters
    ----------
    a, b : array_like
        Positive, real-valued parameters
    y : array_like
        Real-valued input
    out : ndarray, optional
        Optional output array for function values

    Returns
    -------
    scalar or ndarray
        Value of the inverse of the regularized incomplete beta function

    See Also
    --------
    betainc : regularized incomplete beta function
    gamma : gamma function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------
    >>> import scipy.special as sc

    This function is the inverse of `betainc` for fixed
    values of :math:`a` and :math:`b`.

    >>> a, b = 1.2, 3.1
    >>> y = sc.betainc(a, b, 0.2)
    >>> sc.betaincinv(a, b, y)
    0.2
    >>>
    >>> a, b = 7.5, 0.4
    >>> x = sc.betaincinv(a, b, 0.5)
    >>> sc.betainc(a, b, x)
    0.5

    """)

add_newdoc("betaln",
    """
    betaln(a, b, out=None)

    Natural logarithm of absolute value of beta function.

    Computes ``ln(abs(beta(a, b)))``.

    Parameters
    ----------
    a, b : array_like
        Positive, real-valued parameters
    out : ndarray, optional
        Optional output array for function values

    Returns
    -------
    scalar or ndarray
        Value of the betaln function

    See Also
    --------
    gamma : the gamma function
    betainc :  the regularized incomplete beta function
    beta : the beta function

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import betaln, beta

    Verify that, for moderate values of ``a`` and ``b``, ``betaln(a, b)``
    is the same as ``log(beta(a, b))``:

    >>> betaln(3, 4)
    -4.0943445622221

    >>> np.log(beta(3, 4))
    -4.0943445622221

    In the following ``beta(a, b)`` underflows to 0, so we can't compute
    the logarithm of the actual value.

    >>> a = 400
    >>> b = 900
    >>> beta(a, b)
    0.0

    We can compute the logarithm of ``beta(a, b)`` by using `betaln`:

    >>> betaln(a, b)
    -804.3069951764146

    """)

add_newdoc("boxcox",
    """
    boxcox(x, lmbda, out=None)

    Compute the Box-Cox transformation.

    The Box-Cox transformation is::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Returns `nan` if ``x < 0``.
    Returns `-inf` if ``x == 0`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox
    >>> boxcox([1, 4, 10], 2.5)
    array([   0.        ,   12.4       ,  126.09110641])
    >>> boxcox(2, [0, 1, 2])
    array([ 0.69314718,  1.        ,  1.5       ])
    """)

add_newdoc("boxcox1p",
    """
    boxcox1p(x, lmbda, out=None)

    Compute the Box-Cox transformation of 1 + `x`.

    The Box-Cox transformation computed by `boxcox1p` is::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0

    Returns `nan` if ``x < -1``.
    Returns `-inf` if ``x == -1`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox1p
    >>> boxcox1p(1e-4, [0, 0.5, 1])
    array([  9.99950003e-05,   9.99975001e-05,   1.00000000e-04])
    >>> boxcox1p([0.01, 0.1], 0.25)
    array([ 0.00996272,  0.09645476])
    """)

add_newdoc("inv_boxcox",
    """
    inv_boxcox(y, lmbda, out=None)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Parameters
    ----------
    y : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.special import boxcox, inv_boxcox
    >>> y = boxcox([1, 4, 10], 2.5)
    >>> inv_boxcox(y, 2.5)
    array([1., 4., 10.])
    """)

add_newdoc("inv_boxcox1p",
    """
    inv_boxcox1p(y, lmbda, out=None)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0

    Parameters
    ----------
    y : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.special import boxcox1p, inv_boxcox1p
    >>> y = boxcox1p([1, 4, 10], 2.5)
    >>> inv_boxcox1p(y, 2.5)
    array([1., 4., 10.])
    """)

add_newdoc("btdtr",
    r"""
    btdtr(a, b, x, out=None)

    Cumulative distribution function of the beta distribution.

    Returns the integral from zero to `x` of the beta probability density
    function,

    .. math::
        I = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        Shape parameter (a > 0).
    b : array_like
        Shape parameter (b > 0).
    x : array_like
        Upper limit of integration, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Cumulative distribution function of the beta distribution with
        parameters `a` and `b` at `x`.

    See Also
    --------
    betainc

    Notes
    -----
    This function is identical to the incomplete beta integral function
    `betainc`.

    Wrapper for the Cephes [1]_ routine `btdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("btdtri",
    r"""
    btdtri(a, b, p, out=None)

    The `p`-th quantile of the beta distribution.

    This function is the inverse of the beta cumulative distribution function,
    `btdtr`, returning the value of `x` for which `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    a : array_like
        Shape parameter (`a` > 0).
    b : array_like
        Shape parameter (`b` > 0).
    p : array_like
        Cumulative probability, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        The quantile corresponding to `p`.

    See Also
    --------
    betaincinv
    btdtr

    Notes
    -----
    The value of `x` is found by interval halving or Newton iterations.

    Wrapper for the Cephes [1]_ routine `incbi`, which solves the equivalent
    problem of finding the inverse of the incomplete beta integral.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("cbrt",
    """
    cbrt(x, out=None)

    Element-wise cube root of `x`.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The cube root of each value in `x`.

    Examples
    --------
    >>> from scipy.special import cbrt

    >>> cbrt(8)
    2.0
    >>> cbrt([-8, -3, 0.125, 1.331])
    array([-2.        , -1.44224957,  0.5       ,  1.1       ])

    """)

add_newdoc("chdtr",
    r"""
    chdtr(v, x, out=None)

    Chi square cumulative distribution function.

    Returns the area under the left tail (from 0 to `x`) of the Chi
    square probability density function with `v` degrees of freedom:

    .. math::

        \frac{1}{2^{v/2} \Gamma(v/2)} \int_0^x t^{v/2 - 1} e^{-t/2} dt

    Here :math:`\Gamma` is the Gamma function; see `gamma`. This
    integral can be expressed in terms of the regularized lower
    incomplete gamma function `gammainc` as
    ``gammainc(v / 2, x / 2)``. [1]_

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    x : array_like
        Upper bound of the integral.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the cumulative distribution function.

    See Also
    --------
    chdtrc, chdtri, chdtriv, gammainc

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It can be expressed in terms of the regularized lower incomplete
    gamma function.

    >>> v = 1
    >>> x = np.arange(4)
    >>> sc.chdtr(v, x)
    array([0.        , 0.68268949, 0.84270079, 0.91673548])
    >>> sc.gammainc(v / 2, x / 2)
    array([0.        , 0.68268949, 0.84270079, 0.91673548])

    """)

add_newdoc("chdtrc",
    r"""
    chdtrc(v, x, out=None)

    Chi square survival function.

    Returns the area under the right hand tail (from `x` to infinity)
    of the Chi square probability density function with `v` degrees of
    freedom:

    .. math::

        \frac{1}{2^{v/2} \Gamma(v/2)} \int_x^\infty t^{v/2 - 1} e^{-t/2} dt

    Here :math:`\Gamma` is the Gamma function; see `gamma`. This
    integral can be expressed in terms of the regularized upper
    incomplete gamma function `gammaincc` as
    ``gammaincc(v / 2, x / 2)``. [1]_

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    x : array_like
        Lower bound of the integral.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the survival function.

    See Also
    --------
    chdtr, chdtri, chdtriv, gammaincc

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It can be expressed in terms of the regularized upper incomplete
    gamma function.

    >>> v = 1
    >>> x = np.arange(4)
    >>> sc.chdtrc(v, x)
    array([1.        , 0.31731051, 0.15729921, 0.08326452])
    >>> sc.gammaincc(v / 2, x / 2)
    array([1.        , 0.31731051, 0.15729921, 0.08326452])

    """)

add_newdoc("chdtri",
    """
    chdtri(v, p, out=None)

    Inverse to `chdtrc` with respect to `x`.

    Returns `x` such that ``chdtrc(v, x) == p``.

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    p : array_like
        Probability.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    x : scalar or ndarray
        Value so that the probability a Chi square random variable
        with `v` degrees of freedom is greater than `x` equals `p`.

    See Also
    --------
    chdtrc, chdtr, chdtriv

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It inverts `chdtrc`.

    >>> v, p = 1, 0.3
    >>> sc.chdtrc(v, sc.chdtri(v, p))
    0.3
    >>> x = 1
    >>> sc.chdtri(v, sc.chdtrc(v, x))
    1.0

    """)

add_newdoc("chdtriv",
    """
    chdtriv(p, x, out=None)

    Inverse to `chdtr` with respect to `v`.

    Returns `v` such that ``chdtr(v, x) == p``.

    Parameters
    ----------
    p : array_like
        Probability that the Chi square random variable is less than
        or equal to `x`.
    x : array_like
        Nonnegative input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Degrees of freedom.

    See Also
    --------
    chdtr, chdtrc, chdtri

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It inverts `chdtr`.

    >>> p, x = 0.5, 1
    >>> sc.chdtr(sc.chdtriv(p, x), x)
    0.5000000000202172
    >>> v = 1
    >>> sc.chdtriv(sc.chdtr(v, x), v)
    1.0000000000000013

    """)

add_newdoc("chndtr",
    r"""
    chndtr(x, df, nc, out=None)

    Non-central chi square cumulative distribution function

    The cumulative distribution function is given by:

    .. math::

        P(\chi^{\prime 2} \vert \nu, \lambda) =\sum_{j=0}^{\infty}
        e^{-\lambda /2}
        \frac{(\lambda /2)^j}{j!} P(\chi^{\prime 2} \vert \nu + 2j),

    where :math:`\nu > 0` is the degrees of freedom (``df``) and
    :math:`\lambda \geq 0` is the non-centrality parameter (``nc``).

    Parameters
    ----------
    x : array_like
        Upper bound of the integral; must satisfy ``x >= 0``
    df : array_like
        Degrees of freedom; must satisfy ``df > 0``
    nc : array_like
        Non-centrality parameter; must satisfy ``nc >= 0``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    x : scalar or ndarray
        Value of the non-central chi square cumulative distribution function.

    See Also
    --------
    chndtrix, chndtridf, chndtrinc

    """)

add_newdoc("chndtrix",
    """
    chndtrix(p, df, nc, out=None)

    Inverse to `chndtr` vs `x`

    Calculated using a search to find a value for `x` that produces the
    desired value of `p`.

    Parameters
    ----------
    p : array_like
        Probability; must satisfy ``0 <= p < 1``
    df : array_like
        Degrees of freedom; must satisfy ``df > 0``
    nc : array_like
        Non-centrality parameter; must satisfy ``nc >= 0``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    x : scalar or ndarray
        Value so that the probability a non-central Chi square random variable
        with `df` degrees of freedom and non-centrality, `nc`, is greater than
        `x` equals `p`.

    See Also
    --------
    chndtr, chndtridf, chndtrinc

    """)

add_newdoc("chndtridf",
    """
    chndtridf(x, p, nc, out=None)

    Inverse to `chndtr` vs `df`

    Calculated using a search to find a value for `df` that produces the
    desired value of `p`.

    Parameters
    ----------
    x : array_like
        Upper bound of the integral; must satisfy ``x >= 0``
    p : array_like
        Probability; must satisfy ``0 <= p < 1``
    nc : array_like
        Non-centrality parameter; must satisfy ``nc >= 0``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    df : scalar or ndarray
        Degrees of freedom

    See Also
    --------
    chndtr, chndtrix, chndtrinc

    """)

add_newdoc("chndtrinc",
    """
    chndtrinc(x, df, p, out=None)

    Inverse to `chndtr` vs `nc`

    Calculated using a search to find a value for `df` that produces the
    desired value of `p`.

    Parameters
    ----------
    x : array_like
        Upper bound of the integral; must satisfy ``x >= 0``
    df : array_like
        Degrees of freedom; must satisfy ``df > 0``
    p : array_like
        Probability; must satisfy ``0 <= p < 1``
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    nc : scalar or ndarray
        Non-centrality

    See Also
    --------
    chndtr, chndtrix, chndtrinc

    """)

add_newdoc("cosdg",
    """
    cosdg(x, out=None)

    Cosine of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cosine of the input.

    See Also
    --------
    sindg, tandg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using cosine directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cosdg(x)
    array([-0.,  0., -0.])
    >>> np.cos(x * np.pi / 180)
    array([ 6.1232340e-17, -1.8369702e-16,  3.0616170e-16])

    """)

add_newdoc("cosm1",
    """
    cosm1(x, out=None)

    cos(x) - 1 for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real valued argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``cos(x) - 1``.

    See Also
    --------
    expm1, log1p

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than computing ``cos(x) - 1`` directly for
    ``x`` around 0.

    >>> x = 1e-30
    >>> np.cos(x) - 1
    0.0
    >>> sc.cosm1(x)
    -5.0000000000000005e-61

    """)

add_newdoc("cotdg",
    """
    cotdg(x, out=None)

    Cotangent of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cotangent at the input.

    See Also
    --------
    sindg, cosdg, tandg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using cotangent directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cotdg(x)
    array([0., 0., 0.])
    >>> 1 / np.tan(x * np.pi / 180)
    array([6.1232340e-17, 1.8369702e-16, 3.0616170e-16])

    """)

add_newdoc("dawsn",
    """
    dawsn(x, out=None)

    Dawson's integral.

    Computes::

        exp(-x**2) * integral(exp(t**2), t=0..x).

    Parameters
    ----------
    x : array_like
        Function parameter.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        Value of the integral.

    See Also
    --------
    wofz, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-15, 15, num=1000)
    >>> plt.plot(x, special.dawsn(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$dawsn(x)$')
    >>> plt.show()

    """)

add_newdoc("ellipe",
    r"""
    ellipe(m, out=None)

    Complete elliptic integral of the second kind

    This function is defined as

    .. math:: E(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{1/2} dt

    Parameters
    ----------
    m : array_like
        Defines the parameter of the elliptic integral.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    E : scalar or ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpe`.

    For `m > 0` the computation uses the approximation,

    .. math:: E(m) \approx P(1-m) - (1-m) \log(1-m) Q(1-m),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  For
    `m < 0`, the relation

    .. math:: E(m) = E(m/(m - 1)) \sqrt(1-m)

    is used.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre E integral is related to Carlson's symmetric R_D or R_G
    functions in multiple ways [3]_. For example,

    .. math:: E(m) = 2 R_G(0, 1-k^2, 1) .

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i

    Examples
    --------
    This function is used in finding the circumference of an
    ellipse with semi-major axis `a` and semi-minor axis `b`.

    >>> import numpy as np
    >>> from scipy import special

    >>> a = 3.5
    >>> b = 2.1
    >>> e_sq = 1.0 - b**2/a**2  # eccentricity squared

    Then the circumference is found using the following:

    >>> C = 4*a*special.ellipe(e_sq)  # circumference formula
    >>> C
    17.868899204378693

    When `a` and `b` are the same (meaning eccentricity is 0),
    this reduces to the circumference of a circle.

    >>> 4*a*special.ellipe(0.0)  # formula for ellipse with a = b
    21.991148575128552
    >>> 2*np.pi*a  # formula for circle of radius a
    21.991148575128552

    """)

add_newdoc("ellipeinc",
    r"""
    ellipeinc(phi, m, out=None)

    Incomplete elliptic integral of the second kind

    This function is defined as

    .. math:: E(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{1/2} dt

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral.
    m : array_like
        parameter of the elliptic integral.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    E : scalar or ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellie`.

    Computation uses arithmetic-geometric means algorithm.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre E incomplete integral can be related to combinations
    of Carlson's symmetric integrals R_D, R_F, and R_G in multiple
    ways [3]_. For example, with :math:`c = \csc^2\phi`,

    .. math::
      E(\phi, m) = R_F(c-1, c-k^2, c)
        - \frac{1}{3} k^2 R_D(c-1, c-k^2, c) .

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i
    """)

add_newdoc("ellipj",
    """
    ellipj(u, m, out=None)

    Jacobian elliptic functions

    Calculates the Jacobian elliptic functions of parameter `m` between
    0 and 1, and real argument `u`.

    Parameters
    ----------
    m : array_like
        Parameter.
    u : array_like
        Argument.
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    sn, cn, dn, ph : 4-tuple of scalar or ndarray
        The returned functions::

            sn(u|m), cn(u|m), dn(u|m)

        The value `ph` is such that if `u = ellipkinc(ph, m)`,
        then `sn(u|m) = sin(ph)` and `cn(u|m) = cos(ph)`.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpj`.

    These functions are periodic, with quarter-period on the real axis
    equal to the complete elliptic integral `ellipk(m)`.

    Relation to incomplete elliptic integral: If `u = ellipkinc(phi,m)`, then
    `sn(u|m) = sin(phi)`, and `cn(u|m) = cos(phi)`. The `phi` is called
    the amplitude of `u`.

    Computation is by means of the arithmetic-geometric mean algorithm,
    except when `m` is within 1e-9 of 0 or 1. In the latter case with `m`
    close to 1, the approximation applies only for `phi < pi/2`.

    See also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("ellipkm1",
    """
    ellipkm1(p, out=None)

    Complete elliptic integral of the first kind around `m` = 1

    This function is defined as

    .. math:: K(p) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt

    where `m = 1 - p`.

    Parameters
    ----------
    p : array_like
        Defines the parameter of the elliptic integral as `m = 1 - p`.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpk`.

    For `p <= 1`, computation uses the approximation,

    .. math:: K(p) \\approx P(p) - \\log(p) Q(p),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  The
    argument `p` is used internally rather than `m` so that the logarithmic
    singularity at `m = 1` will be shifted to the origin; this preserves
    maximum accuracy.  For `p > 1`, the identity

    .. math:: K(p) = K(1/p)/\\sqrt(p)

    is used.

    See Also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("ellipk",
    r"""
    ellipk(m, out=None)

    Complete elliptic integral of the first kind.

    This function is defined as

    .. math:: K(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{-1/2} dt

    Parameters
    ----------
    m : array_like
        The parameter of the elliptic integral.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral.

    Notes
    -----
    For more precision around point m = 1, use `ellipkm1`, which this
    function calls.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [1]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre K integral is related to Carlson's symmetric R_F
    function by [2]_:

    .. math:: K(m) = R_F(0, 1-k^2, 1) .

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind around m = 1
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i

    """)

add_newdoc("ellipkinc",
    r"""
    ellipkinc(phi, m, out=None)

    Incomplete elliptic integral of the first kind

    This function is defined as

    .. math:: K(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{-1/2} dt

    This function is also called :math:`F(\phi, m)`.

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral
    m : array_like
        parameter of the elliptic integral
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the elliptic integral

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellik`.  The computation is
    carried out using the arithmetic-geometric mean algorithm.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    The Legendre K incomplete integral (or F integral) is related to
    Carlson's symmetric R_F function [3]_.
    Setting :math:`c = \csc^2\phi`,

    .. math:: F(\phi, m) = R_F(c-1, c-k^2, c) .

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind
    elliprf : Completely-symmetric elliptic integral of the first kind.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [3] NIST Digital Library of Mathematical
           Functions. http://dlmf.nist.gov/, Release 1.0.28 of
           2020-09-15. See Sec. 19.25(i) https://dlmf.nist.gov/19.25#i
    """)

add_newdoc(
    "elliprc",
    r"""
    elliprc(x, y, out=None)

    Degenerate symmetric elliptic integral.

    The function RC is defined as [1]_

    .. math::

        R_{\mathrm{C}}(x, y) =
           \frac{1}{2} \int_0^{+\infty} (t + x)^{-1/2} (t + y)^{-1} dt
           = R_{\mathrm{F}}(x, y, y)

    Parameters
    ----------
    x, y : array_like
        Real or complex input parameters. `x` can be any number in the
        complex plane cut along the negative real axis. `y` must be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If `y` is real and negative, the Cauchy
        principal value is returned. If both of `x` and `y` are real, the
        return value is real. Otherwise, the return value is complex.

    Notes
    -----
    RC is a degenerate case of the symmetric integral RF: ``elliprc(x, y) ==
    elliprf(x, y, y)``. It is an elementary function rather than an elliptic
    integral.

    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order. [2]_

    .. versionadded:: 1.8.0

    See Also
    --------
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E6
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprc

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> scale = 0.3 + 0.4j
    >>> elliprc(scale*x, scale*y)
    (0.5484493976710874-0.4169557678995833j)

    >>> elliprc(x, y)/np.sqrt(scale)
    (0.5484493976710874-0.41695576789958333j)

    When the two arguments coincide, the integral is particularly
    simple:

    >>> x = 1.2 + 3.4j
    >>> elliprc(x, x)
    (0.4299173120614631-0.3041729818745595j)

    >>> 1/np.sqrt(x)
    (0.4299173120614631-0.30417298187455954j)

    Another simple case: the first argument vanishes:

    >>> y = 1.2 + 3.4j
    >>> elliprc(0, y)
    (0.6753125346116815-0.47779380263880866j)

    >>> np.pi/2/np.sqrt(y)
    (0.6753125346116815-0.4777938026388088j)

    When `x` and `y` are both positive, we can express
    :math:`R_C(x,y)` in terms of more elementary functions.  For the
    case :math:`0 \le x < y`,

    >>> x = 3.2
    >>> y = 6.
    >>> elliprc(x, y)
    0.44942991498453444

    >>> np.arctan(np.sqrt((y-x)/x))/np.sqrt(y-x)
    0.44942991498453433

    And for the case :math:`0 \le y < x`,

    >>> x = 6.
    >>> y = 3.2
    >>> elliprc(x,y)
    0.4989837501576147

    >>> np.log((np.sqrt(x)+np.sqrt(x-y))/np.sqrt(y))/np.sqrt(x-y)
    0.49898375015761476

    """)

add_newdoc(
    "elliprd",
    r"""
    elliprd(x, y, z, out=None)

    Symmetric elliptic integral of the second kind.

    The function RD is defined as [1]_

    .. math::

        R_{\mathrm{D}}(x, y, z) =
           \frac{3}{2} \int_0^{+\infty} [(t + x) (t + y)]^{-1/2} (t + z)^{-3/2}
           dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x` or `y` can be any number in the
        complex plane cut along the negative real axis, but at most one of them
        can be zero, while `z` must be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the
        return value is real. Otherwise, the return value is complex.

    Notes
    -----
    RD is a degenerate case of the elliptic integral RJ: ``elliprd(x, y, z) ==
    elliprj(x, y, z, z)``.

    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order. [2]_

    .. versionadded:: 1.8.0

    See Also
    --------
    elliprc : Degenerate symmetric elliptic integral.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E5
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprd

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprd(scale*x, scale*y, scale*z)
    (-0.03703043835680379-0.24500934665683802j)

    >>> elliprd(x, y, z)*np.power(scale, -1.5)
    (-0.0370304383568038-0.24500934665683805j)

    All three arguments coincide:

    >>> x = 1.2 + 3.4j
    >>> elliprd(x, x, x)
    (-0.03986825876151896-0.14051741840449586j)

    >>> np.power(x, -1.5)
    (-0.03986825876151894-0.14051741840449583j)

    The so-called "second lemniscate constant":

    >>> elliprd(0, 2, 1)/3
    0.5990701173677961

    >>> from scipy.special import gamma
    >>> gamma(0.75)**2/np.sqrt(2*np.pi)
    0.5990701173677959

    """)

add_newdoc(
    "elliprf",
    r"""
    elliprf(x, y, z, out=None)

    Completely-symmetric elliptic integral of the first kind.

    The function RF is defined as [1]_

    .. math::

        R_{\mathrm{F}}(x, y, z) =
           \frac{1}{2} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2} dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x`, `y`, or `z` can be any number in
        the complex plane cut along the negative real axis, but at most one of
        them can be zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the return
        value is real. Otherwise, the return value is complex.

    Notes
    -----
    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order (cf.:
    https://dlmf.nist.gov/19.36.i) and the AGM algorithm for the complete
    integral. [2]_

    .. versionadded:: 1.8.0

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.
    elliprj : Symmetric elliptic integral of the third kind.

    References
    ----------
    .. [1] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E1
    .. [2] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprf

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprf(scale*x, scale*y, scale*z)
    (0.5328051227278146-0.4008623567957094j)

    >>> elliprf(x, y, z)/np.sqrt(scale)
    (0.5328051227278147-0.4008623567957095j)

    All three arguments coincide:

    >>> x = 1.2 + 3.4j
    >>> elliprf(x, x, x)
    (0.42991731206146316-0.30417298187455954j)

    >>> 1/np.sqrt(x)
    (0.4299173120614631-0.30417298187455954j)

    The so-called "first lemniscate constant":

    >>> elliprf(0, 1, 2)
    1.3110287771460598

    >>> from scipy.special import gamma
    >>> gamma(0.25)**2/(4*np.sqrt(2*np.pi))
    1.3110287771460598

    """)

add_newdoc(
    "elliprg",
    r"""
    elliprg(x, y, z, out=None)

    Completely-symmetric elliptic integral of the second kind.

    The function RG is defined as [1]_

    .. math::

        R_{\mathrm{G}}(x, y, z) =
           \frac{1}{4} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2}
           \left(\frac{x}{t + x} + \frac{y}{t + y} + \frac{z}{t + z}\right) t
           dt

    Parameters
    ----------
    x, y, z : array_like
        Real or complex input parameters. `x`, `y`, or `z` can be any number in
        the complex plane cut along the negative real axis.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, and `z` are real, the return
        value is real. Otherwise, the return value is complex.

    Notes
    -----
    The implementation uses the relation [1]_

    .. math::

        2 R_{\mathrm{G}}(x, y, z) =
           z R_{\mathrm{F}}(x, y, z) -
           \frac{1}{3} (x - z) (y - z) R_{\mathrm{D}}(x, y, z) +
           \sqrt{\frac{x y}{z}}

    and the symmetry of `x`, `y`, `z` when at least one non-zero parameter can
    be chosen as the pivot. When one of the arguments is close to zero, the AGM
    method is applied instead. Other special cases are computed following Ref.
    [2]_

    .. versionadded:: 1.8.0

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprj : Symmetric elliptic integral of the third kind.

    References
    ----------
    .. [1] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293
    .. [2] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.16.E1
           https://dlmf.nist.gov/19.20.ii

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprg

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> scale = 0.3 + 0.4j
    >>> elliprg(scale*x, scale*y, scale*z)
    (1.195936862005246+0.8470988320464167j)

    >>> elliprg(x, y, z)*np.sqrt(scale)
    (1.195936862005246+0.8470988320464165j)

    Simplifications:

    >>> elliprg(0, y, y)
    1.756203682760182

    >>> 0.25*np.pi*np.sqrt(y)
    1.7562036827601817

    >>> elliprg(0, 0, z)
    1.224744871391589

    >>> 0.5*np.sqrt(z)
    1.224744871391589

    The surface area of a triaxial ellipsoid with semiaxes ``a``, ``b``, and
    ``c`` is given by

    .. math::

        S = 4 \pi a b c R_{\mathrm{G}}(1 / a^2, 1 / b^2, 1 / c^2).

    >>> def ellipsoid_area(a, b, c):
    ...     r = 4.0 * np.pi * a * b * c
    ...     return r * elliprg(1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c))
    >>> print(ellipsoid_area(1, 3, 5))
    108.62688289491807
    """)

add_newdoc(
    "elliprj",
    r"""
    elliprj(x, y, z, p, out=None)

    Symmetric elliptic integral of the third kind.

    The function RJ is defined as [1]_

    .. math::

        R_{\mathrm{J}}(x, y, z, p) =
           \frac{3}{2} \int_0^{+\infty} [(t + x) (t + y) (t + z)]^{-1/2}
           (t + p)^{-1} dt

    .. warning::
        This function should be considered experimental when the inputs are
        unbalanced.  Check correctness with another independent implementation.

    Parameters
    ----------
    x, y, z, p : array_like
        Real or complex input parameters. `x`, `y`, or `z` are numbers in
        the complex plane cut along the negative real axis (subject to further
        constraints, see Notes), and at most one of them can be zero. `p` must
        be non-zero.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    R : scalar or ndarray
        Value of the integral. If all of `x`, `y`, `z`, and `p` are real, the
        return value is real. Otherwise, the return value is complex.

        If `p` is real and negative, while `x`, `y`, and `z` are real,
        non-negative, and at most one of them is zero, the Cauchy principal
        value is returned. [1]_ [2]_

    Notes
    -----
    The code implements Carlson's algorithm based on the duplication theorems
    and series expansion up to the 7th order. [3]_ The algorithm is slightly
    different from its earlier incarnation as it appears in [1]_, in that the
    call to `elliprc` (or ``atan``/``atanh``, see [4]_) is no longer needed in
    the inner loop. Asymptotic approximations are used where arguments differ
    widely in the order of magnitude. [5]_

    The input values are subject to certain sufficient but not necessary
    constaints when input arguments are complex. Notably, ``x``, ``y``, and
    ``z`` must have non-negative real parts, unless two of them are
    non-negative and complex-conjugates to each other while the other is a real
    non-negative number. [1]_ If the inputs do not satisfy the sufficient
    condition described in Ref. [1]_ they are rejected outright with the output
    set to NaN.

    In the case where one of ``x``, ``y``, and ``z`` is equal to ``p``, the
    function ``elliprd`` should be preferred because of its less restrictive
    domain.

    .. versionadded:: 1.8.0

    See Also
    --------
    elliprc : Degenerate symmetric integral.
    elliprd : Symmetric elliptic integral of the second kind.
    elliprf : Completely-symmetric elliptic integral of the first kind.
    elliprg : Completely-symmetric elliptic integral of the second kind.

    References
    ----------
    .. [1] B. C. Carlson, "Numerical computation of real or complex elliptic
           integrals," Numer. Algorithm, vol. 10, no. 1, pp. 13-26, 1995.
           https://arxiv.org/abs/math/9409227
           https://doi.org/10.1007/BF02198293
    .. [2] B. C. Carlson, ed., Chapter 19 in "Digital Library of Mathematical
           Functions," NIST, US Dept. of Commerce.
           https://dlmf.nist.gov/19.20.iii
    .. [3] B. C. Carlson, J. FitzSimmons, "Reduction Theorems for Elliptic
           Integrands with the Square Root of Two Quadratic Factors," J.
           Comput. Appl. Math., vol. 118, nos. 1-2, pp. 71-85, 2000.
           https://doi.org/10.1016/S0377-0427(00)00282-X
    .. [4] F. Johansson, "Numerical Evaluation of Elliptic Functions, Elliptic
           Integrals and Modular Forms," in J. Blumlein, C. Schneider, P.
           Paule, eds., "Elliptic Integrals, Elliptic Functions and Modular
           Forms in Quantum Field Theory," pp. 269-293, 2019 (Cham,
           Switzerland: Springer Nature Switzerland)
           https://arxiv.org/abs/1806.06725
           https://doi.org/10.1007/978-3-030-04480-0
    .. [5] B. C. Carlson, J. L. Gustafson, "Asymptotic Approximations for
           Symmetric Elliptic Integrals," SIAM J. Math. Anls., vol. 25, no. 2,
           pp. 288-303, 1994.
           https://arxiv.org/abs/math/9310223
           https://doi.org/10.1137/S0036141092228477

    Examples
    --------
    Basic homogeneity property:

    >>> import numpy as np
    >>> from scipy.special import elliprj

    >>> x = 1.2 + 3.4j
    >>> y = 5.
    >>> z = 6.
    >>> p = 7.
    >>> scale = 0.3 - 0.4j
    >>> elliprj(scale*x, scale*y, scale*z, scale*p)
    (0.10834905565679157+0.19694950747103812j)

    >>> elliprj(x, y, z, p)*np.power(scale, -1.5)
    (0.10834905565679556+0.19694950747103854j)

    Reduction to simpler elliptic integral:

    >>> elliprj(x, y, z, z)
    (0.08288462362195129-0.028376809745123258j)

    >>> from scipy.special import elliprd
    >>> elliprd(x, y, z)
    (0.08288462362195136-0.028376809745123296j)

    All arguments coincide:

    >>> elliprj(x, x, x, x)
    (-0.03986825876151896-0.14051741840449586j)

    >>> np.power(x, -1.5)
    (-0.03986825876151894-0.14051741840449583j)

    """)

add_newdoc("entr",
    r"""
    entr(x, out=None)

    Elementwise function for computing entropy.

    .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0 \\ -\infty & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    res : scalar or ndarray
        The value of the elementwise entropy function at the given points `x`.

    See Also
    --------
    kl_div, rel_entr, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is concave.

    The origin of this function is in convex programming; see [1]_.
    Given a probability distribution :math:`p_1, \ldots, p_n`,
    the definition of entropy in the context of *information theory* is

    .. math::

        \sum_{i = 1}^n \mathrm{entr}(p_i).

    To compute the latter quantity, use `scipy.stats.entropy`.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`

    """)

add_newdoc("erf",
    """
    erf(z, out=None)

    Returns the error function of complex argument.

    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    res : scalar or ndarray
        The values of the error function at the given points `x`.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Notes
    -----
    The cumulative of the unit normal distribution is given by
    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover,
        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
    .. [3] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erf(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erf(x)$')
    >>> plt.show()

    """)

add_newdoc("erfc",
    """
    erfc(x, out=None)

    Complementary error function, ``1 - erf(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the complementary error function

    See Also
    --------
    erf, erfi, erfcx, dawsn, wofz

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfc(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfc(x)$')
    >>> plt.show()

    """)

add_newdoc("erfi",
    """
    erfi(z, out=None)

    Imaginary error function, ``-i erf(i z)``.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the imaginary error function

    See Also
    --------
    erf, erfc, erfcx, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfi(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfi(x)$')
    >>> plt.show()

    """)

add_newdoc("erfcx",
    """
    erfcx(x, out=None)

    Scaled complementary error function, ``exp(x**2) * erfc(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the scaled complementary error function


    See Also
    --------
    erf, erfc, erfi, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfcx(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfcx(x)$')
    >>> plt.show()

    """)

add_newdoc(
    "erfinv",
    """
    erfinv(y, out=None)

    Inverse of the error function.

    Computes the inverse of the error function.

    In the complex domain, there is no unique complex number w satisfying
    erf(w)=z. This indicates a true inverse function would be multivalued.
    When the domain restricts to the real, -1 < x < 1, there is a unique real
    number satisfying erf(erfinv(x)) = x.

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [-1, 1]
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    erfinv : scalar or ndarray
        The inverse of erf of y, element-wise

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfcinv : Inverse of the complementary error function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import erfinv, erf

    >>> erfinv(0.5)
    0.4769362762044699

    >>> y = np.linspace(-1.0, 1.0, num=9)
    >>> x = erfinv(y)
    >>> x
    array([       -inf, -0.81341985, -0.47693628, -0.22531206,  0.        ,
            0.22531206,  0.47693628,  0.81341985,         inf])

    Verify that ``erf(erfinv(y))`` is ``y``.

    >>> erf(x)
    array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    Plot the function:

    >>> y = np.linspace(-1, 1, 200)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y, erfinv(y))
    >>> ax.grid(True)
    >>> ax.set_xlabel('y')
    >>> ax.set_title('erfinv(y)')
    >>> plt.show()

    """)

add_newdoc(
    "erfcinv",
    """
    erfcinv(y, out=None)

    Inverse of the complementary error function.

    Computes the inverse of the complementary error function.

    In the complex domain, there is no unique complex number w satisfying
    erfc(w)=z. This indicates a true inverse function would be multivalued.
    When the domain restricts to the real, 0 < x < 2, there is a unique real
    number satisfying erfc(erfcinv(x)) = erfcinv(erfc(x)).

    It is related to inverse of the error function by erfcinv(1-x) = erfinv(x)

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [0, 2]
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    erfcinv : scalar or ndarray
        The inverse of erfc of y, element-wise

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfinv : Inverse of the error function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import erfcinv

    >>> erfcinv(0.5)
    0.4769362762044699

    >>> y = np.linspace(0.0, 2.0, num=11)
    >>> erfcinv(y)
    array([        inf,  0.9061938 ,  0.59511608,  0.37080716,  0.17914345,
           -0.        , -0.17914345, -0.37080716, -0.59511608, -0.9061938 ,
                  -inf])

    Plot the function:

    >>> y = np.linspace(0, 2, 200)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y, erfcinv(y))
    >>> ax.grid(True)
    >>> ax.set_xlabel('y')
    >>> ax.set_title('erfcinv(y)')
    >>> plt.show()

    """)

add_newdoc("eval_jacobi",
    r"""
    eval_jacobi(n, alpha, beta, x, out=None)

    Evaluate Jacobi polynomial at a point.

    The Jacobi polynomials can be defined via the Gauss hypergeometric
    function :math:`{}_2F_1` as

    .. math::

        P_n^{(\alpha, \beta)}(x) = \frac{(\alpha + 1)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 1 + \alpha + \beta + n; \alpha + 1; (1 - z)/2)

    where :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
    :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.42 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    beta : array_like
        Parameter
    x : array_like
        Points at which to evaluate the polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the Jacobi polynomial

    See Also
    --------
    roots_jacobi : roots and quadrature weights of Jacobi polynomials
    jacobi : Jacobi polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_sh_jacobi",
    r"""
    eval_sh_jacobi(n, p, q, x, out=None)

    Evaluate shifted Jacobi polynomial at a point.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1} P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the n-th Jacobi
    polynomial. See 22.5.2 in [AS]_ for details.

    Parameters
    ----------
    n : int
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `binom` and `eval_jacobi`.
    p : float
        Parameter
    q : float
        Parameter
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    G : scalar or ndarray
        Values of the shifted Jacobi polynomial.

    See Also
    --------
    roots_sh_jacobi : roots and quadrature weights of shifted Jacobi
                      polynomials
    sh_jacobi : shifted Jacobi polynomial object
    eval_jacobi : evaluate Jacobi polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_gegenbauer",
    r"""
    eval_gegenbauer(n, alpha, x, out=None)

    Evaluate Gegenbauer polynomial at a point.

    The Gegenbauer polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    x : array_like
        Points at which to evaluate the Gegenbauer polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    C : scalar or ndarray
        Values of the Gegenbauer polynomial

    See Also
    --------
    roots_gegenbauer : roots and quadrature weights of Gegenbauer
                       polynomials
    gegenbauer : Gegenbauer polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebyt",
    r"""
    eval_chebyt(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind at a point.

    The Chebyshev polynomials of the first kind can be defined via the
    Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        T_n(x) = {}_2F_1(n, -n; 1/2; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.47 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    T : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyt : roots and quadrature weights of Chebyshev
                   polynomials of the first kind
    chebyu : Chebychev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    Notes
    -----
    This routine is numerically stable for `x` in ``[-1, 1]`` at least
    up to order ``10000``.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebyu",
    r"""
    eval_chebyu(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind at a point.

    The Chebyshev polynomials of the second kind can be defined via
    the Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        U_n(x) = (n + 1) {}_2F_1(-n, n + 2; 3/2; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.48 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    U : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyu : roots and quadrature weights of Chebyshev
                   polynomials of the second kind
    chebyu : Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebys",
    r"""
    eval_chebys(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        S_n(x) = U_n(x/2)

    where :math:`U_n` is a Chebyshev polynomial of the second
    kind. See 22.5.13 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    S : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebys : roots and quadrature weights of Chebyshev
                   polynomials of the second kind on [-2, 2]
    chebys : Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    second kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebys(3, x)
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])
    >>> sc.eval_chebyu(3, x / 2)
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])

    """)

add_newdoc("eval_chebyc",
    r"""
    eval_chebyc(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        C_n(x) = 2 T_n(x/2)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.11 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    C : scalar or ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyc : roots and quadrature weights of Chebyshev
                   polynomials of the first kind on [-2, 2]
    chebyc : Chebyshev polynomial object
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
    eval_chebyt : evaluate Chebycshev polynomials of the first kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    first kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebyc(3, x)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])
    >>> 2 * sc.eval_chebyt(3, x / 2)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])

    """)

add_newdoc("eval_sh_chebyt",
    r"""
    eval_sh_chebyt(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the first kind at a
    point.

    These polynomials are defined as

    .. math::

        T_n^*(x) = T_n(2x - 1)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.14 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    T : scalar or ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyt : roots and quadrature weights of shifted
                      Chebyshev polynomials of the first kind
    sh_chebyt : shifted Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_sh_chebyu",
    r"""
    eval_sh_chebyu(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the second kind at a
    point.

    These polynomials are defined as

    .. math::

        U_n^*(x) = U_n(2x - 1)

    where :math:`U_n` is a Chebyshev polynomial of the first kind. See
    22.5.15 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    U : scalar or ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyu : roots and quadrature weights of shifted
                      Chebychev polynomials of the second kind
    sh_chebyu : shifted Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_legendre",
    r"""
    eval_legendre(n, x, out=None)

    Evaluate Legendre polynomial at a point.

    The Legendre polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        P_n(x) = {}_2F_1(-n, n + 1; 1; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.49 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Legendre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the Legendre polynomial

    See Also
    --------
    roots_legendre : roots and quadrature weights of Legendre
                     polynomials
    legendre : Legendre polynomial object
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import eval_legendre

    Evaluate the zero-order Legendre polynomial at x = 0

    >>> eval_legendre(0, 0)
    1.0

    Evaluate the first-order Legendre polynomial between -1 and 1

    >>> X = np.linspace(-1, 1, 5)  # Domain of Legendre polynomials
    >>> eval_legendre(1, X)
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    Evaluate Legendre polynomials of order 0 through 4 at x = 0

    >>> N = range(0, 5)
    >>> eval_legendre(N, 0)
    array([ 1.   ,  0.   , -0.5  ,  0.   ,  0.375])

    Plot Legendre polynomials of order 0 through 4

    >>> X = np.linspace(-1, 1)

    >>> import matplotlib.pyplot as plt
    >>> for n in range(0, 5):
    ...     y = eval_legendre(n, X)
    ...     plt.plot(X, y, label=r'$P_{}(x)$'.format(n))

    >>> plt.title("Legendre Polynomials")
    >>> plt.xlabel("x")
    >>> plt.ylabel(r'$P_n(x)$')
    >>> plt.legend(loc='lower right')
    >>> plt.show()

    """)

add_newdoc("eval_sh_legendre",
    r"""
    eval_sh_legendre(n, x, out=None)

    Evaluate shifted Legendre polynomial at a point.

    These polynomials are defined as

    .. math::

        P_n^*(x) = P_n(2x - 1)

    where :math:`P_n` is a Legendre polynomial. See 2.2.11 in [AS]_
    for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the value is
        determined via the relation to `eval_legendre`.
    x : array_like
        Points at which to evaluate the shifted Legendre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    P : scalar or ndarray
        Values of the shifted Legendre polynomial

    See Also
    --------
    roots_sh_legendre : roots and quadrature weights of shifted
                        Legendre polynomials
    sh_legendre : shifted Legendre polynomial object
    eval_legendre : evaluate Legendre polynomials
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_genlaguerre",
    r"""
    eval_genlaguerre(n, alpha, x, out=None)

    Evaluate generalized Laguerre polynomial at a point.

    The generalized Laguerre polynomials can be defined via the
    confluent hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n^{(\alpha)}(x) = \binom{n + \alpha}{n}
          {}_1F_1(-n, \alpha + 1, x).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.54 in [AS]_ for details. The Laguerre
    polynomials are the special case where :math:`\alpha = 0`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the confluent hypergeometric
        function.
    alpha : array_like
        Parameter; must have ``alpha > -1``
    x : array_like
        Points at which to evaluate the generalized Laguerre
        polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    L : scalar or ndarray
        Values of the generalized Laguerre polynomial

    See Also
    --------
    roots_genlaguerre : roots and quadrature weights of generalized
                        Laguerre polynomials
    genlaguerre : generalized Laguerre polynomial object
    hyp1f1 : confluent hypergeometric function
    eval_laguerre : evaluate Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_laguerre",
    r"""
    eval_laguerre(n, x, out=None)

    Evaluate Laguerre polynomial at a point.

    The Laguerre polynomials can be defined via the confluent
    hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n(x) = {}_1F_1(-n, 1, x).

    See 22.5.16 and 22.5.54 in [AS]_ for details. When :math:`n` is an
    integer the result is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the confluent hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Laguerre polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    L : scalar or ndarray
        Values of the Laguerre polynomial

    See Also
    --------
    roots_laguerre : roots and quadrature weights of Laguerre
                     polynomials
    laguerre : Laguerre polynomial object
    numpy.polynomial.laguerre.Laguerre : Laguerre series
    eval_genlaguerre : evaluate generalized Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

     """)

add_newdoc("eval_hermite",
    r"""
    eval_hermite(n, x, out=None)

    Evaluate physicist's Hermite polynomial at a point.

    Defined by

    .. math::

        H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`. See 22.11.7 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    H : scalar or ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermite : roots and quadrature weights of physicist's
                    Hermite polynomials
    hermite : physicist's Hermite polynomial object
    numpy.polynomial.hermite.Hermite : Physicist's Hermite series
    eval_hermitenorm : evaluate Probabilist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_hermitenorm",
    r"""
    eval_hermitenorm(n, x, out=None)

    Evaluate probabilist's (normalized) Hermite polynomial at a
    point.

    Defined by

    .. math::

        He_n(x) = (-1)^n e^{x^2/2} \frac{d^n}{dx^n} e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`. See 22.11.8 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    He : scalar or ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermitenorm : roots and quadrature weights of probabilist's
                        Hermite polynomials
    hermitenorm : probabilist's Hermite polynomial object
    numpy.polynomial.hermite_e.HermiteE : Probabilist's Hermite series
    eval_hermite : evaluate physicist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("exp1",
    r"""
    exp1(z, out=None)

    Exponential integral E1.

    For complex :math:`z \ne 0` the exponential integral can be defined as
    [1]_

    .. math::

       E_1(z) = \int_z^\infty \frac{e^{-t}}{t} dt,

    where the path of the integral does not cross the negative real
    axis or pass through the origin.

    Parameters
    ----------
    z: array_like
        Real or complex argument.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the exponential integral E1

    See Also
    --------
    expi : exponential integral :math:`Ei`
    expn : generalization of :math:`E_1`

    Notes
    -----
    For :math:`x > 0` it is related to the exponential integral
    :math:`Ei` (see `expi`) via the relation

    .. math::

       E_1(x) = -Ei(-x).

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 6.2.1
           https://dlmf.nist.gov/6.2#E1

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It has a pole at 0.

    >>> sc.exp1(0)
    inf

    It has a branch cut on the negative real axis.

    >>> sc.exp1(-1)
    nan
    >>> sc.exp1(complex(-1, 0))
    (-1.8951178163559368-3.141592653589793j)
    >>> sc.exp1(complex(-1, -0.0))
    (-1.8951178163559368+3.141592653589793j)

    It approaches 0 along the positive real axis.

    >>> sc.exp1([1, 10, 100, 1000])
    array([2.19383934e-01, 4.15696893e-06, 3.68359776e-46, 0.00000000e+00])

    It is related to `expi`.

    >>> x = np.array([1, 2, 3, 4])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> -sc.expi(-x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    """)


add_newdoc(
    "_scaled_exp1",
    """
    _scaled_exp1(x, out=None):

    Compute the scaled exponential integral.

    This is a private function, subject to change or removal with no
    deprecation.

    This function computes F(x), where F is the factor remaining in E_1(x)
    when exp(-x)/x is factored out.  That is,::

        E_1(x) = exp(-x)/x * F(x)

    or

        F(x) = x * exp(x) * E_1(x)

    The function is defined for real x >= 0.  For x < 0, nan is returned.

    F has the properties:

    * F(0) = 0
    * F(x) is increasing on [0, inf).
    * The limit as x goes to infinity of F(x) is 1.

    Parameters
    ----------
    x: array_like
        The input values. Must be real.  The implementation is limited to
        double precision floating point, so other types will be cast to
        to double precision.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the scaled exponential integral.

    See Also
    --------
    exp1 : exponential integral E_1

    Examples
    --------
    >>> from scipy.special import _scaled_exp1
    >>> _scaled_exp1([0, 0.1, 1, 10, 100])

    """
)


add_newdoc("exp10",
    """
    exp10(x, out=None)

    Compute ``10**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``10**x``, computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import exp10

    >>> exp10(3)
    1000.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp10(x)
    array([[  0.1       ,   0.31622777,   1.        ],
           [  3.16227766,  10.        ,  31.6227766 ]])

    """)

add_newdoc("exp2",
    """
    exp2(x, out=None)

    Compute ``2**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``2**x``, computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import exp2

    >>> exp2(3)
    8.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp2(x)
    array([[ 0.5       ,  0.70710678,  1.        ],
           [ 1.41421356,  2.        ,  2.82842712]])
    """)

add_newdoc("expi",
    r"""
    expi(x, out=None)

    Exponential integral Ei.

    For real :math:`x`, the exponential integral is defined as [1]_

    .. math::

        Ei(x) = \int_{-\infty}^x \frac{e^t}{t} dt.

    For :math:`x > 0` the integral is understood as a Cauchy principal
    value.

    It is extended to the complex plane by analytic continuation of
    the function on the interval :math:`(0, \infty)`. The complex
    variant has a branch cut on the negative real axis.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the exponential integral

    Notes
    -----
    The exponential integrals :math:`E_1` and :math:`Ei` satisfy the
    relation

    .. math::

        E_1(x) = -Ei(-x)

    for :math:`x > 0`.

    See Also
    --------
    exp1 : Exponential integral :math:`E_1`
    expn : Generalized exponential integral :math:`E_n`

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 6.2.5
           https://dlmf.nist.gov/6.2#E5

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is related to `exp1`.

    >>> x = np.array([1, 2, 3, 4])
    >>> -sc.expi(-x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    The complex variant has a branch cut on the negative real axis.

    >>> sc.expi(-1 + 1e-12j)
    (-0.21938393439552062+3.1415926535894254j)
    >>> sc.expi(-1 - 1e-12j)
    (-0.21938393439552062-3.1415926535894254j)

    As the complex variant approaches the branch cut, the real parts
    approach the value of the real variant.

    >>> sc.expi(-1)
    -0.21938393439552062

    The SciPy implementation returns the real variant for complex
    values on the branch cut.

    >>> sc.expi(complex(-1, 0.0))
    (-0.21938393439552062-0j)
    >>> sc.expi(complex(-1, -0.0))
    (-0.21938393439552062-0j)

    """)

add_newdoc('expit',
    """
    expit(x, out=None)

    Expit (a.k.a. logistic sigmoid) ufunc for ndarrays.

    The expit function, also known as the logistic sigmoid function, is
    defined as ``expit(x) = 1/(1+exp(-x))``.  It is the inverse of the
    logit function.

    Parameters
    ----------
    x : ndarray
        The ndarray to apply expit to element-wise.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        An ndarray of the same shape as x. Its entries
        are `expit` of the corresponding entry of x.

    See Also
    --------
    logit

    Notes
    -----
    As a ufunc expit takes a number of optional
    keyword arguments. For more information
    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_

    .. versionadded:: 0.10.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import expit, logit

    >>> expit([-np.inf, -1.5, 0, 1.5, np.inf])
    array([ 0.        ,  0.18242552,  0.5       ,  0.81757448,  1.        ])

    `logit` is the inverse of `expit`:

    >>> logit(expit([-2.5, 0, 3.1, 5.0]))
    array([-2.5,  0. ,  3.1,  5. ])

    Plot expit(x) for x in [-6, 6]:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-6, 6, 121)
    >>> y = expit(x)
    >>> plt.plot(x, y)
    >>> plt.grid()
    >>> plt.xlim(-6, 6)
    >>> plt.xlabel('x')
    >>> plt.title('expit(x)')
    >>> plt.show()

    """)

add_newdoc("expm1",
    """
    expm1(x, out=None)

    Compute ``exp(x) - 1``.

    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
    ``expm1(x)`` is implemented to avoid the loss of precision that occurs when
    `x` is near zero.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``exp(x) - 1`` computed element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import expm1

    >>> expm1(1.0)
    1.7182818284590451
    >>> expm1([-0.2, -0.1, 0, 0.1, 0.2])
    array([-0.18126925, -0.09516258,  0.        ,  0.10517092,  0.22140276])

    The exact value of ``exp(7.5e-13) - 1`` is::

        7.5000000000028125000000007031250000001318...*10**-13.

    Here is what ``expm1(7.5e-13)`` gives:

    >>> expm1(7.5e-13)
    7.5000000000028135e-13

    Compare that to ``exp(7.5e-13) - 1``, where the subtraction results in
    a "catastrophic" loss of precision:

    >>> np.exp(7.5e-13) - 1
    7.5006667543675576e-13

    """)

add_newdoc("expn",
    r"""
    expn(n, x, out=None)

    Generalized exponential integral En.

    For integer :math:`n \geq 0` and real :math:`x \geq 0` the
    generalized exponential integral is defined as [dlmf]_

    .. math::

        E_n(x) = x^{n - 1} \int_x^\infty \frac{e^{-t}}{t^n} dt.

    Parameters
    ----------
    n : array_like
        Non-negative integers
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the generalized exponential integral

    See Also
    --------
    exp1 : special case of :math:`E_n` for :math:`n = 1`
    expi : related to :math:`E_n` when :math:`n = 1`

    References
    ----------
    .. [dlmf] Digital Library of Mathematical Functions, 8.19.2
              https://dlmf.nist.gov/8.19#E2

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    Its domain is nonnegative n and x.

    >>> sc.expn(-1, 1.0), sc.expn(1, -1.0)
    (nan, nan)

    It has a pole at ``x = 0`` for ``n = 1, 2``; for larger ``n`` it
    is equal to ``1 / (n - 1)``.

    >>> sc.expn([0, 1, 2, 3, 4], 0)
    array([       inf,        inf, 1.        , 0.5       , 0.33333333])

    For n equal to 0 it reduces to ``exp(-x) / x``.

    >>> x = np.array([1, 2, 3, 4])
    >>> sc.expn(0, x)
    array([0.36787944, 0.06766764, 0.01659569, 0.00457891])
    >>> np.exp(-x) / x
    array([0.36787944, 0.06766764, 0.01659569, 0.00457891])

    For n equal to 1 it reduces to `exp1`.

    >>> sc.expn(1, x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    """)

add_newdoc("exprel",
    r"""
    exprel(x, out=None)

    Relative error exponential, ``(exp(x) - 1)/x``.

    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
    ``exprel(x)`` is implemented to avoid the loss of precision that occurs when
    `x` is near zero.

    Parameters
    ----------
    x : ndarray
        Input array.  `x` must contain real numbers.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        ``(exp(x) - 1)/x``, computed element-wise.

    See Also
    --------
    expm1

    Notes
    -----
    .. versionadded:: 0.17.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import exprel

    >>> exprel(0.01)
    1.0050167084168056
    >>> exprel([-0.25, -0.1, 0, 0.1, 0.25])
    array([ 0.88479687,  0.95162582,  1.        ,  1.05170918,  1.13610167])

    Compare ``exprel(5e-9)`` to the naive calculation.  The exact value
    is ``1.00000000250000000416...``.

    >>> exprel(5e-9)
    1.0000000025

    >>> (np.exp(5e-9) - 1)/5e-9
    0.99999999392252903
    """)

add_newdoc("fdtr",
    r"""
    fdtr(dfn, dfd, x, out=None)

    F cumulative distribution function.

    Returns the value of the cumulative distribution function of the
    F-distribution, also known as Snedecor's F-distribution or the
    Fisher-Snedecor distribution.

    The F-distribution with parameters :math:`d_n` and :math:`d_d` is the
    distribution of the random variable,

    .. math::
        X = \frac{U_n/d_n}{U_d/d_d},

    where :math:`U_n` and :math:`U_d` are random variables distributed
    :math:`\chi^2`, with :math:`d_n` and :math:`d_d` degrees of freedom,
    respectively.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        The CDF of the F-distribution with parameters `dfn` and `dfd` at `x`.

    See Also
    --------
    fdtrc : F distribution survival function
    fdtri : F distribution inverse cumulative distribution
    scipy.stats.f : F distribution

    Notes
    -----
    The regularized incomplete beta function is used, according to the
    formula,

    .. math::
        F(d_n, d_d; x) = I_{xd_n/(d_d + xd_n)}(d_n/2, d_d/2).

    Wrapper for the Cephes [1]_ routine `fdtr`. The F distribution is also
    available as `scipy.stats.f`. Calling `fdtr` directly can improve
    performance compared to the ``cdf`` method of `scipy.stats.f` (see last
    example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function for ``dfn=1`` and ``dfd=2`` at ``x=1``.

    >>> import numpy as np
    >>> from scipy.special import fdtr
    >>> fdtr(1, 2, 1)
    0.5773502691896258

    Calculate the function at several points by providing a NumPy array for
    `x`.

    >>> x = np.array([0.5, 2., 3.])
    >>> fdtr(1, 2, x)
    array([0.4472136 , 0.70710678, 0.77459667])

    Plot the function for several parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [1, 5, 10, 50]
    >>> dfd_parameters = [1, 1, 2, 3]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtr_vals = fdtr(dfn, dfd, x)
    ...     ax.plot(x, fdtr_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("F distribution cumulative distribution function")
    >>> plt.show()

    The F distribution is also available as `scipy.stats.f`. Using `fdtr`
    directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.f`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.f(dfn, dfd).cdf(x)=fdtr(dfn, dfd, x)``.

    >>> from scipy.stats import f
    >>> dfn, dfd = 1, 2
    >>> x = 1
    >>> fdtr_res = fdtr(dfn, dfd, x)  # this will often be faster than below
    >>> f_dist_res = f(dfn, dfd).cdf(x)
    >>> fdtr_res == f_dist_res  # test that results are equal
    True
    """)

add_newdoc("fdtrc",
    r"""
    fdtrc(dfn, dfd, x, out=None)

    F survival function.

    Returns the complemented F-distribution function (the integral of the
    density from `x` to infinity).

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    y : scalar or ndarray
        The complemented F-distribution function with parameters `dfn` and
        `dfd` at `x`.

    See Also
    --------
    fdtr : F distribution cumulative distribution function
    fdtri : F distribution inverse cumulative distribution function
    scipy.stats.f : F distribution

    Notes
    -----
    The regularized incomplete beta function is used, according to the
    formula,

    .. math::
        F(d_n, d_d; x) = I_{d_d/(d_d + xd_n)}(d_d/2, d_n/2).

    Wrapper for the Cephes [1]_ routine `fdtrc`. The F distribution is also
    available as `scipy.stats.f`. Calling `fdtrc` directly can improve
    performance compared to the ``sf`` method of `scipy.stats.f` (see last
    example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function for ``dfn=1`` and ``dfd=2`` at ``x=1``.

    >>> import numpy as np
    >>> from scipy.special import fdtrc
    >>> fdtrc(1, 2, 1)
    0.42264973081037427

    Calculate the function at several points by providing a NumPy array for
    `x`.

    >>> x = np.array([0.5, 2., 3.])
    >>> fdtrc(1, 2, x)
    array([0.5527864 , 0.29289322, 0.22540333])

    Plot the function for several parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [1, 5, 10, 50]
    >>> dfd_parameters = [1, 1, 2, 3]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtrc_vals = fdtrc(dfn, dfd, x)
    ...     ax.plot(x, fdtrc_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("F distribution survival function")
    >>> plt.show()

    The F distribution is also available as `scipy.stats.f`. Using `fdtrc`
    directly can be much faster than calling the ``sf`` method of
    `scipy.stats.f`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.f(dfn, dfd).sf(x)=fdtrc(dfn, dfd, x)``.

    >>> from scipy.stats import f
    >>> dfn, dfd = 1, 2
    >>> x = 1
    >>> fdtrc_res = fdtrc(dfn, dfd, x)  # this will often be faster than below
    >>> f_dist_res = f(dfn, dfd).sf(x)
    >>> f_dist_res == fdtrc_res  # test that results are equal
    True
    """)

add_newdoc("fdtri",
    r"""
    fdtri(dfn, dfd, p, out=None)

    The `p`-th quantile of the F-distribution.

    This function is the inverse of the F-distribution CDF, `fdtr`, returning
    the `x` such that `fdtr(dfn, dfd, x) = p`.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    p : array_like
        Cumulative probability, in [0, 1].
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    x : scalar or ndarray
        The quantile corresponding to `p`.

    See Also
    --------
    fdtr : F distribution cumulative distribution function
    fdtrc : F distribution survival function
    scipy.stats.f : F distribution

    Notes
    -----
    The computation is carried out using the relation to the inverse
    regularized beta function, :math:`I^{-1}_x(a, b)`.  Let
    :math:`z = I^{-1}_p(d_d/2, d_n/2).`  Then,

    .. math::
        x = \frac{d_d (1 - z)}{d_n z}.

    If `p` is such that :math:`x < 0.5`, the following relation is used
    instead for improved stability: let
    :math:`z' = I^{-1}_{1 - p}(d_n/2, d_d/2).` Then,

    .. math::
        x = \frac{d_d z'}{d_n (1 - z')}.

    Wrapper for the Cephes [1]_ routine `fdtri`.

    The F distribution is also available as `scipy.stats.f`. Calling
    `fdtri` directly can improve performance compared to the ``ppf``
    method of `scipy.stats.f` (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    `fdtri` represents the inverse of the F distribution CDF which is
    available as `fdtr`. Here, we calculate the CDF for ``df1=1``, ``df2=2``
    at ``x=3``. `fdtri` then returns ``3`` given the same values for `df1`,
    `df2` and the computed CDF value.

    >>> import numpy as np
    >>> from scipy.special import fdtri, fdtr
    >>> df1, df2 = 1, 2
    >>> x = 3
    >>> cdf_value =  fdtr(df1, df2, x)
    >>> fdtri(df1, df2, cdf_value)
    3.000000000000006

    Calculate the function at several points by providing a NumPy array for
    `x`.

    >>> x = np.array([0.1, 0.4, 0.7])
    >>> fdtri(1, 2, x)
    array([0.02020202, 0.38095238, 1.92156863])

    Plot the function for several parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> dfn_parameters = [50, 10, 1, 50]
    >>> dfd_parameters = [0.5, 1, 1, 5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(dfn_parameters, dfd_parameters,
    ...                            linestyles))
    >>> x = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     dfn, dfd, style = parameter_set
    ...     fdtri_vals = fdtri(dfn, dfd, x)
    ...     ax.plot(x, fdtri_vals, label=rf"$d_n={dfn},\, d_d={dfd}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> title = "F distribution inverse cumulative distribution function"
    >>> ax.set_title(title)
    >>> ax.set_ylim(0, 30)
    >>> plt.show()

    The F distribution is also available as `scipy.stats.f`. Using `fdtri`
    directly can be much faster than calling the ``ppf`` method of
    `scipy.stats.f`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.f(dfn, dfd).ppf(x)=fdtri(dfn, dfd, x)``.

    >>> from scipy.stats import f
    >>> dfn, dfd = 1, 2
    >>> x = 0.7
    >>> fdtri_res = fdtri(dfn, dfd, x)  # this will often be faster than below
    >>> f_dist_res = f(dfn, dfd).ppf(x)
    >>> f_dist_res == fdtri_res  # test that results are equal
    True
    """)

add_newdoc("fdtridfd",
    """
    fdtridfd(dfn, p, x, out=None)

    Inverse to `fdtr` vs dfd

    Finds the F density argument dfd such that ``fdtr(dfn, dfd, x) == p``.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    p : array_like
        Cumulative probability, in [0, 1].
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    dfd : scalar or ndarray
        `dfd` such that ``fdtr(dfn, dfd, x) == p``.

    See Also
    --------
    fdtr : F distribution cumulative distribution function
    fdtrc : F distribution survival function
    fdtri : F distribution quantile function
    scipy.stats.f : F distribution

    Examples
    --------
    Compute the F distribution cumulative distribution function for one
    parameter set.

    >>> from scipy.special import fdtridfd, fdtr
    >>> dfn, dfd, x = 10, 5, 2
    >>> cdf_value = fdtr(dfn, dfd, x)
    >>> cdf_value
    0.7700248806501017

    Verify that `fdtridfd` recovers the original value for `dfd`:

    >>> fdtridfd(dfn, cdf_value, x)
    5.0
    """)

'''
commented out as fdtridfn seems to have bugs and is not in functions.json
see: https://github.com/scipy/scipy/pull/15622#discussion_r811440983

add_newdoc(
    "fdtridfn",
    """
    fdtridfn(p, dfd, x, out=None)

    Inverse to `fdtr` vs dfn

    finds the F density argument dfn such that ``fdtr(dfn, dfd, x) == p``.


    Parameters
    ----------
    p : array_like
        Cumulative probability, in [0, 1].
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    dfn : scalar or ndarray
        `dfn` such that ``fdtr(dfn, dfd, x) == p``.

    See Also
    --------
    fdtr, fdtrc, fdtri, fdtridfd


    """)
'''

add_newdoc("fresnel",
    r"""
    fresnel(z, out=None)

    Fresnel integrals.

    The Fresnel integrals are defined as

    .. math::

       S(z) &= \int_0^z \sin(\pi t^2 /2) dt \\
       C(z) &= \int_0^z \cos(\pi t^2 /2) dt.

    See [dlmf]_ for details.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : 2-tuple of ndarrays, optional
        Optional output arrays for the function results

    Returns
    -------
    S, C : 2-tuple of scalar or ndarray
        Values of the Fresnel integrals

    See Also
    --------
    fresnel_zeros : zeros of the Fresnel integrals

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/7.2#iii

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    As z goes to infinity along the real axis, S and C converge to 0.5.

    >>> S, C = sc.fresnel([0.1, 1, 10, 100, np.inf])
    >>> S
    array([0.00052359, 0.43825915, 0.46816998, 0.4968169 , 0.5       ])
    >>> C
    array([0.09999753, 0.7798934 , 0.49989869, 0.4999999 , 0.5       ])

    They are related to the error function `erf`.

    >>> z = np.array([1, 2, 3, 4])
    >>> zeta = 0.5 * np.sqrt(np.pi) * (1 - 1j) * z
    >>> S, C = sc.fresnel(z)
    >>> C + 1j*S
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])
    >>> 0.5 * (1 + 1j) * sc.erf(zeta)
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])

    """)

add_newdoc("gamma",
    r"""
    gamma(z, out=None)

    gamma function.

    The gamma function is defined as

    .. math::

       \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt

    for :math:`\Re(z) > 0` and is extended to the rest of the complex
    plane by analytic continuation. See [dlmf]_ for more details.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the gamma function

    Notes
    -----
    The gamma function is often referred to as the generalized
    factorial since :math:`\Gamma(n + 1) = n!` for natural numbers
    :math:`n`. More generally it satisfies the recurrence relation
    :math:`\Gamma(z + 1) = z \cdot \Gamma(z)` for complex :math:`z`,
    which, combined with the fact that :math:`\Gamma(1) = 1`, implies
    the above identity for :math:`z = n`.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5.2#E1

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import gamma, factorial

    >>> gamma([0, 0.5, 1, 5])
    array([         inf,   1.77245385,   1.        ,  24.        ])

    >>> z = 2.5 + 1j
    >>> gamma(z)
    (0.77476210455108352+0.70763120437959293j)
    >>> gamma(z+1), z*gamma(z)  # Recurrence property
    ((1.2292740569981171+2.5438401155000685j),
     (1.2292740569981158+2.5438401155000658j))

    >>> gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
    3.1415926535897927

    Plot gamma(x) for real x

    >>> x = np.linspace(-3.5, 5.5, 2251)
    >>> y = gamma(x)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')
    >>> k = np.arange(1, 7)
    >>> plt.plot(k, factorial(k-1), 'k*', alpha=0.6,
    ...          label='(x-1)!, x = 1, 2, ...')
    >>> plt.xlim(-3.5, 5.5)
    >>> plt.ylim(-10, 25)
    >>> plt.grid()
    >>> plt.xlabel('x')
    >>> plt.legend(loc='lower right')
    >>> plt.show()

    """)

add_newdoc("gammainc",
    r"""
    gammainc(a, x, out=None)

    Regularized lower incomplete gamma function.

    It is defined as

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the lower incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammaincc` is the regularized upper
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    See also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the CDF of the gamma distribution, so it starts at 0 and
    monotonically increases to 1.

    >>> sc.gammainc(0.5, [0, 1, 10, 100])
    array([0.        , 0.84270079, 0.99999226, 1.        ])

    It is equal to one minus the upper incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammainc(a, x)
    0.6289066304773024
    >>> 1 - sc.gammaincc(a, x)
    0.6289066304773024

    """)

add_newdoc("gammaincc",
    r"""
    gammaincc(a, x, out=None)

    Regularized upper incomplete gamma function.

    It is defined as

    .. math::

        Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the upper incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammainc` is the regularized lower
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    See also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the survival function of the gamma distribution, so it
    starts at 1 and monotonically decreases to 0.

    >>> sc.gammaincc(0.5, [0, 1, 10, 100, 1000])
    array([1.00000000e+00, 1.57299207e-01, 7.74421643e-06, 2.08848758e-45,
           0.00000000e+00])

    It is equal to one minus the lower incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammaincc(a, x)
    0.37109336952269756
    >>> 1 - sc.gammainc(a, x)
    0.37109336952269756

    """)

add_newdoc("gammainccinv",
    """
    gammainccinv(a, y, out=None)

    Inverse of the regularized upper incomplete gamma function.

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = Q(a, x)`. Here :math:`Q` is the regularized upper
    incomplete gamma function; see `gammaincc`. This is well-defined
    because the upper incomplete gamma function is monotonic as can
    be seen from its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Argument between 0 and 1, inclusive
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the upper incomplete gamma function

    See Also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at infinity and monotonically decreases to 0.

    >>> sc.gammainccinv(0.5, [0, 0.1, 0.5, 1])
    array([       inf, 1.35277173, 0.22746821, 0.        ])

    It inverts the upper incomplete gamma function.

    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammaincc(a, sc.gammainccinv(a, x))
    array([0. , 0.1, 0.5, 1. ])

    >>> a, x = 0.5, [0, 10, 50]
    >>> sc.gammainccinv(a, sc.gammaincc(a, x))
    array([ 0., 10., 50.])

    """)

add_newdoc("gammaincinv",
    """
    gammaincinv(a, y, out=None)

    Inverse to the regularized lower incomplete gamma function.

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = P(a, x)`. Here :math:`P` is the regularized lower
    incomplete gamma function; see `gammainc`. This is well-defined
    because the lower incomplete gamma function is monotonic as can be
    seen from its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Parameter between 0 and 1, inclusive
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the lower incomplete gamma function

    See Also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincc : regularized upper incomplete gamma function
    gammainccinv : inverse of the regularized upper incomplete gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at 0 and monotonically increases to infinity.

    >>> sc.gammaincinv(0.5, [0, 0.1 ,0.5, 1])
    array([0.        , 0.00789539, 0.22746821,        inf])

    It inverts the lower incomplete gamma function.

    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammainc(a, sc.gammaincinv(a, x))
    array([0. , 0.1, 0.5, 1. ])

    >>> a, x = 0.5, [0, 10, 25]
    >>> sc.gammaincinv(a, sc.gammainc(a, x))
    array([ 0.        , 10.        , 25.00001465])

    """)

add_newdoc("gammaln",
    r"""
    gammaln(x, out=None)

    Logarithm of the absolute value of the gamma function.

    Defined as

    .. math::

       \ln(\lvert\Gamma(x)\rvert)

    where :math:`\Gamma` is the gamma function. For more details on
    the gamma function, see [dlmf]_.

    Parameters
    ----------
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the log of the absolute value of gamma

    See Also
    --------
    gammasgn : sign of the gamma function
    loggamma : principal branch of the logarithm of the gamma function

    Notes
    -----
    It is the same function as the Python standard library function
    :func:`math.lgamma`.

    When used in conjunction with `gammasgn`, this function is useful
    for working in logspace on the real axis without having to deal
    with complex numbers via the relation ``exp(gammaln(x)) =
    gammasgn(x) * gamma(x)``.

    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It has two positive zeros.

    >>> sc.gammaln([1, 2])
    array([0., 0.])

    It has poles at nonpositive integers.

    >>> sc.gammaln([0, -1, -2, -3, -4])
    array([inf, inf, inf, inf, inf])

    It asymptotically approaches ``x * log(x)`` (Stirling's formula).

    >>> x = np.array([1e10, 1e20, 1e40, 1e80])
    >>> sc.gammaln(x)
    array([2.20258509e+11, 4.50517019e+21, 9.11034037e+41, 1.83206807e+82])
    >>> x * np.log(x)
    array([2.30258509e+11, 4.60517019e+21, 9.21034037e+41, 1.84206807e+82])

    """)

add_newdoc("gammasgn",
    r"""
    gammasgn(x, out=None)

    Sign of the gamma function.

    It is defined as

    .. math::

       \text{gammasgn}(x) =
       \begin{cases}
         +1 & \Gamma(x) > 0 \\
         -1 & \Gamma(x) < 0
       \end{cases}

    where :math:`\Gamma` is the gamma function; see `gamma`. This
    definition is complete since the gamma function is never zero;
    see the discussion after [dlmf]_.

    Parameters
    ----------
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Sign of the gamma function

    Notes
    -----
    The gamma function can be computed as ``gammasgn(x) *
    np.exp(gammaln(x))``.

    See Also
    --------
    gamma : the gamma function
    gammaln : log of the absolute value of the gamma function
    loggamma : analytic continuation of the log of the gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5.2#E1

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is 1 for `x > 0`.

    >>> sc.gammasgn([1, 2, 3, 4])
    array([1., 1., 1., 1.])

    It alternates between -1 and 1 for negative integers.

    >>> sc.gammasgn([-0.5, -1.5, -2.5, -3.5])
    array([-1.,  1., -1.,  1.])

    It can be used to compute the gamma function.

    >>> x = [1.5, 0.5, -0.5, -1.5]
    >>> sc.gammasgn(x) * np.exp(sc.gammaln(x))
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])
    >>> sc.gamma(x)
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])

    """)

add_newdoc("gdtr",
    r"""
    gdtr(a, b, x, out=None)

    Gamma distribution cumulative distribution function.

    Returns the integral from zero to `x` of the gamma probability density
    function,

    .. math::

        F = \int_0^x \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        The rate parameter of the gamma distribution, sometimes denoted
        :math:`\beta` (float).  It is also the reciprocal of the scale
        parameter :math:`\theta`.
    b : array_like
        The shape parameter of the gamma distribution, sometimes denoted
        :math:`\alpha` (float).
    x : array_like
        The quantile (upper limit of integration; float).
    out : ndarray, optional
        Optional output array for the function values

    See also
    --------
    gdtrc : 1 - CDF of the gamma distribution.
    scipy.stats.gamma: Gamma distribution

    Returns
    -------
    F : scalar or ndarray
        The CDF of the gamma distribution with parameters `a` and `b`
        evaluated at `x`.

    Notes
    -----
    The evaluation is carried out using the relation to the incomplete gamma
    integral (regularized gamma function).

    Wrapper for the Cephes [1]_ routine `gdtr`. Calling `gdtr` directly can
    improve performance compared to the ``cdf`` method of `scipy.stats.gamma`
    (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``a=1``, ``b=2`` at ``x=5``.

    >>> import numpy as np
    >>> from scipy.special import gdtr
    >>> import matplotlib.pyplot as plt
    >>> gdtr(1., 2., 5.)
    0.9595723180054873

    Compute the function for ``a=1`` and ``b=2`` at several points by
    providing a NumPy array for `x`.

    >>> xvalues = np.array([1., 2., 3., 4])
    >>> gdtr(1., 1., xvalues)
    array([0.63212056, 0.86466472, 0.95021293, 0.98168436])

    `gdtr` can evaluate different parameter sets by providing arrays with
    broadcasting compatible shapes for `a`, `b` and `x`. Here we compute the
    function for three different `a` at four positions `x` and ``b=3``,
    resulting in a 3x4 array.

    >>> a = np.array([[0.5], [1.5], [2.5]])
    >>> x = np.array([1., 2., 3., 4])
    >>> a.shape, x.shape
    ((3, 1), (4,))

    >>> gdtr(a, 3., x)
    array([[0.01438768, 0.0803014 , 0.19115317, 0.32332358],
           [0.19115317, 0.57680992, 0.82642193, 0.9380312 ],
           [0.45618688, 0.87534798, 0.97974328, 0.9972306 ]])

    Plot the function for four different parameter sets.

    >>> a_parameters = [0.3, 1, 2, 6]
    >>> b_parameters = [2, 10, 15, 20]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(a_parameters, b_parameters, linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     a, b, style = parameter_set
    ...     gdtr_vals = gdtr(a, b, x)
    ...     ax.plot(x, gdtr_vals, label=f"$a= {a},\, b={b}$", ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("Gamma distribution cumulative distribution function")
    >>> plt.show()

    The gamma distribution is also available as `scipy.stats.gamma`. Using
    `gdtr` directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.gamma`, especially for small arrays or individual values.
    To get the same results one must use the following parametrization:
    ``stats.gamma(b, scale=1/a).cdf(x)=gdtr(a, b, x)``.

    >>> from scipy.stats import gamma
    >>> a = 2.
    >>> b = 3
    >>> x = 1.
    >>> gdtr_result = gdtr(a, b, x)  # this will often be faster than below
    >>> gamma_dist_result = gamma(b, scale=1/a).cdf(x)
    >>> gdtr_result == gamma_dist_result  # test that results are equal
    True
    """)

add_newdoc("gdtrc",
    r"""
    gdtrc(a, b, x, out=None)

    Gamma distribution survival function.

    Integral from `x` to infinity of the gamma probability density function,

    .. math::

        F = \int_x^\infty \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        The rate parameter of the gamma distribution, sometimes denoted
        :math:`\beta` (float). It is also the reciprocal of the scale
        parameter :math:`\theta`.
    b : array_like
        The shape parameter of the gamma distribution, sometimes denoted
        :math:`\alpha` (float).
    x : array_like
        The quantile (lower limit of integration; float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    F : scalar or ndarray
        The survival function of the gamma distribution with parameters `a`
        and `b` evaluated at `x`.

    See Also
    --------
    gdtr: Gamma distribution cumulative distribution function
    scipy.stats.gamma: Gamma distribution
    gdtrix

    Notes
    -----
    The evaluation is carried out using the relation to the incomplete gamma
    integral (regularized gamma function).

    Wrapper for the Cephes [1]_ routine `gdtrc`. Calling `gdtrc` directly can
    improve performance compared to the ``sf`` method of `scipy.stats.gamma`
    (see last example below).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``a=1`` and ``b=2`` at ``x=5``.

    >>> import numpy as np
    >>> from scipy.special import gdtrc
    >>> import matplotlib.pyplot as plt
    >>> gdtrc(1., 2., 5.)
    0.04042768199451279

    Compute the function for ``a=1``, ``b=2`` at several points by providing
    a NumPy array for `x`.

    >>> xvalues = np.array([1., 2., 3., 4])
    >>> gdtrc(1., 1., xvalues)
    array([0.36787944, 0.13533528, 0.04978707, 0.01831564])

    `gdtrc` can evaluate different parameter sets by providing arrays with
    broadcasting compatible shapes for `a`, `b` and `x`. Here we compute the
    function for three different `a` at four positions `x` and ``b=3``,
    resulting in a 3x4 array.

    >>> a = np.array([[0.5], [1.5], [2.5]])
    >>> x = np.array([1., 2., 3., 4])
    >>> a.shape, x.shape
    ((3, 1), (4,))

    >>> gdtrc(a, 3., x)
    array([[0.98561232, 0.9196986 , 0.80884683, 0.67667642],
           [0.80884683, 0.42319008, 0.17357807, 0.0619688 ],
           [0.54381312, 0.12465202, 0.02025672, 0.0027694 ]])

    Plot the function for four different parameter sets.

    >>> a_parameters = [0.3, 1, 2, 6]
    >>> b_parameters = [2, 10, 15, 20]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(a_parameters, b_parameters, linestyles))
    >>> x = np.linspace(0, 30, 1000)
    >>> fig, ax = plt.subplots()
    >>> for parameter_set in parameters_list:
    ...     a, b, style = parameter_set
    ...     gdtrc_vals = gdtrc(a, b, x)
    ...     ax.plot(x, gdtrc_vals, label=f"$a= {a},\, b={b}$", ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("Gamma distribution survival function")
    >>> plt.show()

    The gamma distribution is also available as `scipy.stats.gamma`.
    Using `gdtrc` directly can be much faster than calling the ``sf`` method
    of `scipy.stats.gamma`, especially for small arrays or individual
    values. To get the same results one must use the following parametrization:
    ``stats.gamma(b, scale=1/a).sf(x)=gdtrc(a, b, x)``.

    >>> from scipy.stats import gamma
    >>> a = 2
    >>> b = 3
    >>> x = 1.
    >>> gdtrc_result = gdtrc(a, b, x)  # this will often be faster than below
    >>> gamma_dist_result = gamma(b, scale=1/a).sf(x)
    >>> gdtrc_result == gamma_dist_result  # test that results are equal
    True
    """)

add_newdoc("gdtria",
    """
    gdtria(p, b, x, out=None)

    Inverse of `gdtr` vs a.

    Returns the inverse with respect to the parameter `a` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    p : array_like
        Probability values.
    b : array_like
        `b` parameter values of `gdtr(a, b, x)`. `b` is the "shape" parameter
        of the gamma distribution.
    x : array_like
        Nonnegative real values, from the domain of the gamma distribution.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`.  `out` is then the
        array returned by the function.

    Returns
    -------
    a : scalar or ndarray
        Values of the `a` parameter such that `p = gdtr(a, b, x)`.  `1/a`
        is the "scale" parameter of the gamma distribution.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.
    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `a` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `a`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtria
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtria(p, 3.4, 5.6)
    1.2
    """)

add_newdoc("gdtrib",
    """
    gdtrib(a, p, x, out=None)

    Inverse of `gdtr` vs b.

    Returns the inverse with respect to the parameter `b` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    a : array_like
        `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"
        parameter of the gamma distribution.
    p : array_like
        Probability values.
    x : array_like
        Nonnegative real values, from the domain of the gamma distribution.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`.  `out` is then the
        array returned by the function.

    Returns
    -------
    b : scalar or ndarray
        Values of the `b` parameter such that `p = gdtr(a, b, x)`.  `b` is
        the "shape" parameter of the gamma distribution.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `b` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `b`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtrib
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtrib(1.2, p, 5.6)
    3.3999999999723882
    """)

add_newdoc("gdtrix",
    """
    gdtrix(a, b, p, out=None)

    Inverse of `gdtr` vs x.

    Returns the inverse with respect to the parameter `x` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution. This is also known as the pth quantile of the
    distribution.

    Parameters
    ----------
    a : array_like
        `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"
        parameter of the gamma distribution.
    b : array_like
        `b` parameter values of `gdtr(a, b, x)`. `b` is the "shape" parameter
        of the gamma distribution.
    p : array_like
        Probability values.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`. `out` is then the
        array returned by the function.

    Returns
    -------
    x : scalar or ndarray
        Values of the `x` parameter such that `p = gdtr(a, b, x)`.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `x` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `x`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtrix
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtrix(1.2, 3.4, p)
    5.5999999999999996
    """)

add_newdoc("hankel1",
    r"""
    hankel1(v, z, out=None)

    Hankel function of the first kind

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the Hankel function of the first kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.

    See also
    --------
    hankel1e : ndarray
        This function with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel1e",
    r"""
    hankel1e(v, z, out=None)

    Exponentially scaled Hankel function of the first kind

    Defined as::

        hankel1e(v, z) = hankel1(v, z) * exp(-1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled Hankel function.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel2",
    r"""
    hankel2(v, z, out=None)

    Hankel function of the second kind

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the Hankel function of the second kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\imath \pi v/2) K_v(z \exp(\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    See also
    --------
    hankel2e : this function with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel2e",
    r"""
    hankel2e(v, z, out=None)

    Exponentially scaled Hankel function of the second kind

    Defined as::

        hankel2e(v, z) = hankel2(v, z) * exp(1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled Hankel function of the second kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\frac{\imath \pi v}{2}) K_v(z exp(\frac{\imath\pi}{2}))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    """)

add_newdoc("huber",
    r"""
    huber(delta, r, out=None)

    Huber loss function.

    .. math:: \text{huber}(\delta, r) = \begin{cases} \infty & \delta < 0  \\ \frac{1}{2}r^2 & 0 \le \delta, | r | \le \delta \\ \delta ( |r| - \frac{1}{2}\delta ) & \text{otherwise} \end{cases}

    Parameters
    ----------
    delta : ndarray
        Input array, indicating the quadratic vs. linear loss changepoint.
    r : ndarray
        Input array, possibly representing residuals.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The computed Huber loss function values.

    See also
    --------
    pseudo_huber : smooth approximation of this function

    Notes
    -----
    `huber` is useful as a loss function in robust statistics or machine
    learning to reduce the influence of outliers as compared to the common
    squared error loss, residuals with a magnitude higher than `delta` are
    not squared [1]_.

    Typically, `r` represents residuals, the difference
    between a model prediction and data. Then, for :math:`|r|\leq\delta`,
    `huber` resembles the squared error and for :math:`|r|>\delta` the
    absolute error. This way, the Huber loss often achieves
    a fast convergence in model fitting for small residuals like the squared
    error loss function and still reduces the influence of outliers
    (:math:`|r|>\delta`) like the absolute error loss. As :math:`\delta` is
    the cutoff between squared and absolute error regimes, it has
    to be tuned carefully for each problem. `huber` is also
    convex, making it suitable for gradient based optimization.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter Huber. "Robust Estimation of a Location Parameter",
           1964. Annals of Statistics. 53 (1): 73 - 101.

    Examples
    --------
    Import all necessary modules.

    >>> import numpy as np
    >>> from scipy.special import huber
    >>> import matplotlib.pyplot as plt

    Compute the function for ``delta=1`` at ``r=2``

    >>> huber(1., 2.)
    1.5

    Compute the function for different `delta` by providing a NumPy array or
    list for `delta`.

    >>> huber([1., 3., 5.], 4.)
    array([3.5, 7.5, 8. ])

    Compute the function at different points by providing a NumPy array or
    list for `r`.

    >>> huber(2., np.array([1., 1.5, 3.]))
    array([0.5  , 1.125, 4.   ])

    The function can be calculated for different `delta` and `r` by
    providing arrays for both with compatible shapes for broadcasting.

    >>> r = np.array([1., 2.5, 8., 10.])
    >>> deltas = np.array([[1.], [5.], [9.]])
    >>> print(r.shape, deltas.shape)
    (4,) (3, 1)

    >>> huber(deltas, r)
    array([[ 0.5  ,  2.   ,  7.5  ,  9.5  ],
           [ 0.5  ,  3.125, 27.5  , 37.5  ],
           [ 0.5  ,  3.125, 32.   , 49.5  ]])

    Plot the function for different `delta`.

    >>> x = np.linspace(-4, 4, 500)
    >>> deltas = [1, 2, 3]
    >>> linestyles = ["dashed", "dotted", "dashdot"]
    >>> fig, ax = plt.subplots()
    >>> combined_plot_parameters = list(zip(deltas, linestyles))
    >>> for delta, style in combined_plot_parameters:
    ...     ax.plot(x, huber(delta, x), label=f"$\delta={delta}$", ls=style)
    >>> ax.legend(loc="upper center")
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("Huber loss function $h_{\delta}(x)$")
    >>> ax.set_xlim(-4, 4)
    >>> ax.set_ylim(0, 8)
    >>> plt.show()
    """)

add_newdoc("hyp0f1",
    r"""
    hyp0f1(v, z, out=None)

    Confluent hypergeometric limit function 0F1.

    Parameters
    ----------
    v : array_like
        Real-valued parameter
    z : array_like
        Real- or complex-valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The confluent hypergeometric limit function

    Notes
    -----
    This function is defined as:

    .. math:: _0F_1(v, z) = \sum_{k=0}^{\infty}\frac{z^k}{(v)_k k!}.

    It's also the limit as :math:`q \to \infty` of :math:`_1F_1(q; v; z/q)`,
    and satisfies the differential equation :math:`f''(z) + vf'(z) =
    f(z)`. See [1]_ for more information.

    References
    ----------
    .. [1] Wolfram MathWorld, "Confluent Hypergeometric Limit Function",
           http://mathworld.wolfram.com/ConfluentHypergeometricLimitFunction.html

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is one when `z` is zero.

    >>> sc.hyp0f1(1, 0)
    1.0

    It is the limit of the confluent hypergeometric function as `q`
    goes to infinity.

    >>> q = np.array([1, 10, 100, 1000])
    >>> v = 1
    >>> z = 1
    >>> sc.hyp1f1(q, v, z / q)
    array([2.71828183, 2.31481985, 2.28303778, 2.27992985])
    >>> sc.hyp0f1(v, z)
    2.2795853023360673

    It is related to Bessel functions.

    >>> n = 1
    >>> x = np.linspace(0, 1, 5)
    >>> sc.jv(n, x)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])
    >>> (0.5 * x)**n / sc.factorial(n) * sc.hyp0f1(n + 1, -0.25 * x**2)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])

    """)

add_newdoc("hyp1f1",
    r"""
    hyp1f1(a, b, x, out=None)

    Confluent hypergeometric function 1F1.

    The confluent hypergeometric function is defined by the series

    .. math::

       {}_1F_1(a; b; x) = \sum_{k = 0}^\infty \frac{(a)_k}{(b)_k k!} x^k.

    See [dlmf]_ for more details. Here :math:`(\cdot)_k` is the
    Pochhammer symbol; see `poch`.

    Parameters
    ----------
    a, b : array_like
        Real parameters
    x : array_like
        Real or complex argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the confluent hypergeometric function

    See also
    --------
    hyperu : another confluent hypergeometric function
    hyp0f1 : confluent hypergeometric limit function
    hyp2f1 : Gaussian hypergeometric function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/13.2#E2

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is one when `x` is zero:

    >>> sc.hyp1f1(0.5, 0.5, 0)
    1.0

    It is singular when `b` is a nonpositive integer.

    >>> sc.hyp1f1(0.5, -1, 0)
    inf

    It is a polynomial when `a` is a nonpositive integer.

    >>> a, b, x = -1, 0.5, np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.hyp1f1(a, b, x)
    array([-1., -3., -5., -7.])
    >>> 1 + (a / b) * x
    array([-1., -3., -5., -7.])

    It reduces to the exponential function when `a = b`.

    >>> sc.hyp1f1(2, 2, [1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])
    >>> np.exp([1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])

    """)

add_newdoc("hyp2f1",
    r"""
    hyp2f1(a, b, c, z, out=None)

    Gauss hypergeometric function 2F1(a, b; c; z)

    Parameters
    ----------
    a, b, c : array_like
        Arguments, should be real-valued.
    z : array_like
        Argument, real or complex.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    hyp2f1 : scalar or ndarray
        The values of the gaussian hypergeometric function.

    See also
    --------
    hyp0f1 : confluent hypergeometric limit function.
    hyp1f1 : Kummer's (confluent hypergeometric) function.

    Notes
    -----
    This function is defined for :math:`|z| < 1` as

    .. math::

       \mathrm{hyp2f1}(a, b, c, z) = \sum_{n=0}^\infty
       \frac{(a)_n (b)_n}{(c)_n}\frac{z^n}{n!},

    and defined on the rest of the complex z-plane by analytic
    continuation [1]_.
    Here :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
    :math:`n` is an integer the result is a polynomial of degree :math:`n`.

    The implementation for complex values of ``z`` is described in [2]_,
    except for ``z`` in the region defined by

    .. math::

         0.9 <= \left|z\right| < 1.1,
         \left|1 - z\right| >= 0.9,
         \mathrm{real}(z) >= 0

    in which the implementation follows [4]_.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/15.2
    .. [2] S. Zhang and J.M. Jin, "Computation of Special Functions", Wiley 1996
    .. [3] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [4] J.L. Lopez and N.M. Temme, "New series expansions of the Gauss
           hypergeometric function", Adv Comput Math 39, 349-365 (2013).
           https://doi.org/10.1007/s10444-012-9283-y

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It has poles when `c` is a negative integer.

    >>> sc.hyp2f1(1, 1, -2, 1)
    inf

    It is a polynomial when `a` or `b` is a negative integer.

    >>> a, b, c = -1, 1, 1.5
    >>> z = np.linspace(0, 1, 5)
    >>> sc.hyp2f1(a, b, c, z)
    array([1.        , 0.83333333, 0.66666667, 0.5       , 0.33333333])
    >>> 1 + a * b * z / c
    array([1.        , 0.83333333, 0.66666667, 0.5       , 0.33333333])

    It is symmetric in `a` and `b`.

    >>> a = np.linspace(0, 1, 5)
    >>> b = np.linspace(0, 1, 5)
    >>> sc.hyp2f1(a, b, 1, 0.5)
    array([1.        , 1.03997334, 1.1803406 , 1.47074441, 2.        ])
    >>> sc.hyp2f1(b, a, 1, 0.5)
    array([1.        , 1.03997334, 1.1803406 , 1.47074441, 2.        ])

    It contains many other functions as special cases.

    >>> z = 0.5
    >>> sc.hyp2f1(1, 1, 2, z)
    1.3862943611198901
    >>> -np.log(1 - z) / z
    1.3862943611198906

    >>> sc.hyp2f1(0.5, 1, 1.5, z**2)
    1.098612288668109
    >>> np.log((1 + z) / (1 - z)) / (2 * z)
    1.0986122886681098

    >>> sc.hyp2f1(0.5, 1, 1.5, -z**2)
    0.9272952180016117
    >>> np.arctan(z) / z
    0.9272952180016122

    """)

add_newdoc("hyperu",
    r"""
    hyperu(a, b, x, out=None)

    Confluent hypergeometric function U

    It is defined as the solution to the equation

    .. math::

       x \frac{d^2w}{dx^2} + (b - x) \frac{dw}{dx} - aw = 0

    which satisfies the property

    .. math::

       U(a, b, x) \sim x^{-a}

    as :math:`x \to \infty`. See [dlmf]_ for more details.

    Parameters
    ----------
    a, b : array_like
        Real-valued parameters
    x : array_like
        Real-valued argument
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of `U`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematics Functions
              https://dlmf.nist.gov/13.2#E6

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It has a branch cut along the negative `x` axis.

    >>> x = np.linspace(-0.1, -10, 5)
    >>> sc.hyperu(1, 1, x)
    array([nan, nan, nan, nan, nan])

    It approaches zero as `x` goes to infinity.

    >>> x = np.array([1, 10, 100])
    >>> sc.hyperu(1, 1, x)
    array([0.59634736, 0.09156333, 0.00990194])

    It satisfies Kummer's transformation.

    >>> a, b, x = 2, 1, 1
    >>> sc.hyperu(a, b, x)
    0.1926947246463881
    >>> x**(1 - b) * sc.hyperu(a - b + 1, 2 - b, x)
    0.1926947246463881

    """)

add_newdoc("i0",
    r"""
    i0(x, out=None)

    Modified Bessel function of order 0.

    Defined as,

    .. math::
        I_0(x) = \sum_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2} = J_0(\imath x),

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the modified Bessel function of order 0 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i0`.

    See also
    --------
    iv: Modified Bessel function of any order
    i0e: Exponentially scaled modified Bessel function of order 0

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import i0
    >>> i0(1.)
    1.2660658777520082

    Calculate at several points:

    >>> import numpy as np
    >>> i0(np.array([-2., 0., 3.5]))
    array([2.2795853 , 1.        , 7.37820343])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("i0e",
    """
    i0e(x, out=None)

    Exponentially scaled modified Bessel function of order 0.

    Defined as::

        i0e(x) = exp(-abs(x)) * i0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the exponentially scaled modified Bessel function of order 0
        at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i0`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i0e`. `i0e`
    is useful for large arguments `x`: for these, `i0` quickly overflows.

    See also
    --------
    iv: Modified Bessel function of the first kind
    i0: Modified Bessel function of order 0

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `i0` returns infinity whereas `i0e` still returns
    a finite number.

    >>> from scipy.special import i0, i0e
    >>> i0(1000.), i0e(1000.)
    (inf, 0.012617240455891257)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> i0e(np.array([-2., 0., 3.]))
    array([0.30850832, 1.        , 0.24300035])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i0e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
    """)

add_newdoc("i1",
    r"""
    i1(x, out=None)

    Modified Bessel function of order 1.

    Defined as,

    .. math::
        I_1(x) = \frac{1}{2}x \sum_{k=0}^\infty \frac{(x^2/4)^k}{k! (k + 1)!}
               = -\imath J_1(\imath x),

    where :math:`J_1` is the Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the modified Bessel function of order 1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i1`.

    See also
    --------
    iv: Modified Bessel function of the first kind
    i1e: Exponentially scaled modified Bessel function of order 1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import i1
    >>> i1(1.)
    0.5651591039924851

    Calculate the function at several points:

    >>> import numpy as np
    >>> i1(np.array([-2., 0., 6.]))
    array([-1.59063685,  0.        , 61.34193678])

    Plot the function between -10 and 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("i1e",
    """
    i1e(x, out=None)

    Exponentially scaled modified Bessel function of order 1.

    Defined as::

        i1e(x) = exp(-abs(x)) * i1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        Value of the exponentially scaled modified Bessel function of order 1
        at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i1`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i1e`. `i1e`
    is useful for large arguments `x`: for these, `i1` quickly overflows.

    See also
    --------
    iv: Modified Bessel function of the first kind
    i1: Modified Bessel function of order 1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `i1` returns infinity whereas `i1e` still returns
    a finite number.

    >>> from scipy.special import i1, i1e
    >>> i1(1000.), i1e(1000.)
    (inf, 0.01261093025692863)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> i1e(np.array([-2., 0., 6.]))
    array([-0.21526929,  0.        ,  0.15205146])

    Plot the function between -10 and 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> y = i1e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
    """)

add_newdoc("_igam_fac",
    """
    Internal function, do not use.
    """)

add_newdoc("it2i0k0",
    r"""
    it2i0k0(x, out=None)

    Integrals related to modified Bessel functions of order 0.

    Computes the integrals

    .. math::

        \int_0^x \frac{I_0(t) - 1}{t} dt \\
        \int_x^\infty \frac{K_0(t)}{t} dt.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ii0 : scalar or ndarray
        The integral for `i0`
    ik0 : scalar or ndarray
        The integral for `k0`

    References
    ----------
    .. [1] S. Zhang and J.M. Jin, "Computation of Special Functions",
           Wiley 1996

    Examples
    --------
    Evaluate the functions at one point.

    >>> from scipy.special import it2i0k0
    >>> int_i, int_k = it2i0k0(1.)
    >>> int_i, int_k
    (0.12897944249456852, 0.2085182909001295)

    Evaluate the functions at several points.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> int_i, int_k = it2i0k0(points)
    >>> int_i, int_k
    (array([0.03149527, 0.30187149, 1.50012461]),
     array([0.66575102, 0.0823715 , 0.00823631]))

    Plot the functions from 0 to 5.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 5., 1000)
    >>> int_i, int_k = it2i0k0(x)
    >>> ax.plot(x, int_i, label=r"$\int_0^x \frac{I_0(t)-1}{t}\,dt$")
    >>> ax.plot(x, int_k, label=r"$\int_x^{\infty} \frac{K_0(t)}{t}\,dt$")
    >>> ax.legend()
    >>> ax.set_ylim(0, 10)
    >>> plt.show()
    """)

add_newdoc("it2j0y0",
    r"""
    it2j0y0(x, out=None)

    Integrals related to Bessel functions of the first kind of order 0.

    Computes the integrals

    .. math::

        \int_0^x \frac{1 - J_0(t)}{t} dt \\
        \int_x^\infty \frac{Y_0(t)}{t} dt.

    For more on :math:`J_0` and :math:`Y_0` see `j0` and `y0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ij0 : scalar or ndarray
        The integral for `j0`
    iy0 : scalar or ndarray
        The integral for `y0`

    References
    ----------
    .. [1] S. Zhang and J.M. Jin, "Computation of Special Functions",
           Wiley 1996

    Examples
    --------
    Evaluate the functions at one point.

    >>> from scipy.special import it2j0y0
    >>> int_j, int_y = it2j0y0(1.)
    >>> int_j, int_y
    (0.12116524699506871, 0.39527290169929336)

    Evaluate the functions at several points.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> int_j, int_y = it2j0y0(points)
    >>> int_j, int_y
    (array([0.03100699, 0.26227724, 0.85614669]),
     array([ 0.26968854,  0.29769696, -0.02987272]))

    Plot the functions from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> int_j, int_y = it2j0y0(x)
    >>> ax.plot(x, int_j, label=r"$\int_0^x \frac{1-J_0(t)}{t}\,dt$")
    >>> ax.plot(x, int_y, label=r"$\int_x^{\infty} \frac{Y_0(t)}{t}\,dt$")
    >>> ax.legend()
    >>> ax.set_ylim(-2.5, 2.5)
    >>> plt.show()
    """)

add_newdoc("it2struve0",
    r"""
    it2struve0(x, out=None)

    Integral related to the Struve function of order 0.

    Returns the integral,

    .. math::
        \int_x^\infty \frac{H_0(t)}{t}\,dt

    where :math:`H_0` is the Struve function of order 0.

    Parameters
    ----------
    x : array_like
        Lower limit of integration.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        The value of the integral.

    See also
    --------
    struve

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    Evaluate the function at one point.

    >>> import numpy as np
    >>> from scipy.special import it2struve0
    >>> it2struve0(1.)
    0.9571973506383524

    Evaluate the function at several points by supplying
    an array for `x`.

    >>> points = np.array([1., 2., 3.5])
    >>> it2struve0(points)
    array([0.95719735, 0.46909296, 0.10366042])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10., 10., 1000)
    >>> it2struve0_values = it2struve0(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, it2struve0_values)
    >>> ax.set_xlabel(r'$x$')
    >>> ax.set_ylabel(r'$\int_x^{\infty}\frac{H_0(t)}{t}\,dt$')
    >>> plt.show()
    """)

add_newdoc(
    "itairy",
    r"""
    itairy(x, out=None)

    Integrals of Airy functions

    Calculates the integrals of Airy functions from 0 to `x`.

    Parameters
    ----------

    x : array_like
        Upper limit of integration (float).
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    Apt : scalar or ndarray
        Integral of Ai(t) from 0 to x.
    Bpt : scalar or ndarray
        Integral of Bi(t) from 0 to x.
    Ant : scalar or ndarray
        Integral of Ai(-t) from 0 to x.
    Bnt : scalar or ndarray
        Integral of Bi(-t) from 0 to x.

    Notes
    -----

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------

    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    Compute the functions at ``x=1.``.

    >>> import numpy as np
    >>> from scipy.special import itairy
    >>> import matplotlib.pyplot as plt
    >>> apt, bpt, ant, bnt = itairy(1.)
    >>> apt, bpt, ant, bnt
    (0.23631734191710949,
     0.8727691167380077,
     0.46567398346706845,
     0.3730050096342943)

    Compute the functions at several points by providing a NumPy array for `x`.

    >>> x = np.array([1., 1.5, 2.5, 5])
    >>> apt, bpt, ant, bnt = itairy(x)
    >>> apt, bpt, ant, bnt
    (array([0.23631734, 0.28678675, 0.324638  , 0.33328759]),
     array([  0.87276912,   1.62470809,   5.20906691, 321.47831857]),
     array([0.46567398, 0.72232876, 0.93187776, 0.7178822 ]),
     array([ 0.37300501,  0.35038814, -0.02812939,  0.15873094]))

    Plot the functions from -10 to 10.

    >>> x = np.linspace(-10, 10, 500)
    >>> apt, bpt, ant, bnt = itairy(x)
    >>> fig, ax = plt.subplots(figsize=(6, 5))
    >>> ax.plot(x, apt, label="$\int_0^x\, Ai(t)\, dt$")
    >>> ax.plot(x, bpt, ls="dashed", label="$\int_0^x\, Bi(t)\, dt$")
    >>> ax.plot(x, ant, ls="dashdot", label="$\int_0^x\, Ai(-t)\, dt$")
    >>> ax.plot(x, bnt, ls="dotted", label="$\int_0^x\, Bi(-t)\, dt$")
    >>> ax.set_ylim(-2, 1.5)
    >>> ax.legend(loc="lower right")
    >>> plt.show()
    """)

add_newdoc("iti0k0",
    r"""
    iti0k0(x, out=None)

    Integrals of modified Bessel functions of order 0.

    Computes the integrals

    .. math::

        \int_0^x I_0(t) dt \\
        \int_0^x K_0(t) dt.

    For more on :math:`I_0` and :math:`K_0` see `i0` and `k0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ii0 : scalar or ndarray
        The integral for `i0`
    ik0 : scalar or ndarray
        The integral for `k0`

    References
    ----------
    .. [1] S. Zhang and J.M. Jin, "Computation of Special Functions",
           Wiley 1996

    Examples
    --------
    Evaluate the functions at one point.

    >>> from scipy.special import iti0k0
    >>> int_i, int_k = iti0k0(1.)
    >>> int_i, int_k
    (1.0865210970235892, 1.2425098486237771)

    Evaluate the functions at several points.

    >>> import numpy as np
    >>> points = np.array([0., 1.5, 3.])
    >>> int_i, int_k = iti0k0(points)
    >>> int_i, int_k
    (array([0.        , 1.80606937, 6.16096149]),
     array([0.        , 1.39458246, 1.53994809]))

    Plot the functions from 0 to 5.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 5., 1000)
    >>> int_i, int_k = iti0k0(x)
    >>> ax.plot(x, int_i, label="$\int_0^x I_0(t)\,dt$")
    >>> ax.plot(x, int_k, label="$\int_0^x K_0(t)\,dt$")
    >>> ax.legend()
    >>> plt.show()
    """)

add_newdoc("itj0y0",
    r"""
    itj0y0(x, out=None)

    Integrals of Bessel functions of the first kind of order 0.

    Computes the integrals

    .. math::

        \int_0^x J_0(t) dt \\
        \int_0^x Y_0(t) dt.

    For more on :math:`J_0` and :math:`Y_0` see `j0` and `y0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ij0 : scalar or ndarray
        The integral of `j0`
    iy0 : scalar or ndarray
        The integral of `y0`

    References
    ----------
    .. [1] S. Zhang and J.M. Jin, "Computation of Special Functions",
           Wiley 1996

    Examples
    --------
    Evaluate the functions at one point.

    >>> from scipy.special import itj0y0
    >>> int_j, int_y = itj0y0(1.)
    >>> int_j, int_y
    (0.9197304100897596, -0.637069376607422)

    Evaluate the functions at several points.

    >>> import numpy as np
    >>> points = np.array([0., 1.5, 3.])
    >>> int_j, int_y = itj0y0(points)
    >>> int_j, int_y
    (array([0.        , 1.24144951, 1.38756725]),
     array([ 0.        , -0.51175903,  0.19765826]))

    Plot the functions from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> int_j, int_y = itj0y0(x)
    >>> ax.plot(x, int_j, label="$\int_0^x J_0(t)\,dt$")
    >>> ax.plot(x, int_y, label="$\int_0^x Y_0(t)\,dt$")
    >>> ax.legend()
    >>> plt.show()

    """)

add_newdoc("itmodstruve0",
    r"""
    itmodstruve0(x, out=None)

    Integral of the modified Struve function of order 0.

    .. math::
        I = \int_0^x L_0(t)\,dt

    Parameters
    ----------
    x : array_like
        Upper limit of integration (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        The integral of :math:`L_0` from 0 to `x`.

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    See Also
    --------
    modstruve: Modified Struve function which is integrated by this function

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    Evaluate the function at one point.

    >>> import numpy as np
    >>> from scipy.special import itmodstruve0
    >>> itmodstruve0(1.)
    0.3364726286440384

    Evaluate the function at several points by supplying
    an array for `x`.

    >>> points = np.array([1., 2., 3.5])
    >>> itmodstruve0(points)
    array([0.33647263, 1.588285  , 7.60382578])

    Plot the function from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10., 10., 1000)
    >>> itmodstruve0_values = itmodstruve0(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, itmodstruve0_values)
    >>> ax.set_xlabel(r'$x$')
    >>> ax.set_ylabel(r'$\int_0^xL_0(t)\,dt$')
    >>> plt.show()
    """)

add_newdoc("itstruve0",
    r"""
    itstruve0(x, out=None)

    Integral of the Struve function of order 0.

    .. math::
        I = \int_0^x H_0(t)\,dt

    Parameters
    ----------
    x : array_like
        Upper limit of integration (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    I : scalar or ndarray
        The integral of :math:`H_0` from 0 to `x`.

    See also
    --------
    struve: Function which is integrated by this function

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    Evaluate the function at one point.

    >>> import numpy as np
    >>> from scipy.special import itstruve0
    >>> itstruve0(1.)
    0.30109042670805547

    Evaluate the function at several points by supplying
    an array for `x`.

    >>> points = np.array([1., 2., 3.5])
    >>> itstruve0(points)
    array([0.30109043, 1.01870116, 1.96804581])

    Plot the function from -20 to 20.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-20., 20., 1000)
    >>> istruve0_values = itstruve0(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, istruve0_values)
    >>> ax.set_xlabel(r'$x$')
    >>> ax.set_ylabel(r'$\int_0^{x}H_0(t)\,dt$')
    >>> plt.show()
    """)

add_newdoc("iv",
    r"""
    iv(v, z, out=None)

    Modified Bessel function of the first kind of real order.

    Parameters
    ----------
    v : array_like
        Order. If `z` is of real type and negative, `v` must be integer
        valued.
    z : array_like of float or complex
        Argument.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the modified Bessel function.

    Notes
    -----
    For real `z` and :math:`v \in [-50, 50]`, the evaluation is carried out
    using Temme's method [1]_.  For larger orders, uniform asymptotic
    expansions are applied.

    For complex `z` and positive `v`, the AMOS [2]_ `zbesi` routine is
    called. It uses a power series for small `z`, the asymptotic expansion
    for large `abs(z)`, the Miller algorithm normalized by the Wronskian
    and a Neumann series for intermediate magnitudes, and the uniform
    asymptotic expansions for :math:`I_v(z)` and :math:`J_v(z)` for large
    orders. Backward recurrence is used to generate sequences or reduce
    orders when necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    See also
    --------
    ive : This function with leading exponential behavior stripped off.
    i0 : Faster version of this function for order 0.
    i1 : Faster version of this function for order 1.

    References
    ----------
    .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import iv
    >>> iv(0, 1.)
    1.2660658777520084

    Evaluate the function at one point for different orders.

    >>> iv(0, 1.), iv(1, 1.), iv(1.5, 1.)
    (1.2660658777520084, 0.565159103992485, 0.2935253263474798)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> iv([0, 1, 1.5], 1.)
    array([1.26606588, 0.5651591 , 0.29352533])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> import numpy as np
    >>> points = np.array([-2., 0., 3.])
    >>> iv(0, points)
    array([2.2795853 , 1.        , 4.88079259])

    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    To calculate the orders 0 and 1 for an 1D array:

    >>> orders = np.array([[0], [1]])
    >>> orders.shape
    (2, 1)

    >>> iv(orders, points)
    array([[ 2.2795853 ,  1.        ,  4.88079259],
           [-1.59063685,  0.        ,  3.95337022]])

    Plot the functions of order 0 to 3 from -5 to 5.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-5., 5., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, iv(i, x), label=f'$I_{i!r}$')
    >>> ax.legend()
    >>> plt.show()

    """)

add_newdoc("ive",
    r"""
    ive(v, z, out=None)

    Exponentially scaled modified Bessel function of the first kind.

    Defined as::

        ive(v, z) = iv(v, z) * exp(-abs(z.real))

    For imaginary numbers without a real part, returns the unscaled
    Bessel function of the first kind `iv`.

    Parameters
    ----------
    v : array_like of float
        Order.
    z : array_like of float or complex
        Argument.
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of the exponentially scaled modified Bessel function.

    Notes
    -----
    For positive `v`, the AMOS [1]_ `zbesi` routine is called. It uses a
    power series for small `z`, the asymptotic expansion for large
    `abs(z)`, the Miller algorithm normalized by the Wronskian and a
    Neumann series for intermediate magnitudes, and the uniform asymptotic
    expansions for :math:`I_v(z)` and :math:`J_v(z)` for large orders.
    Backward recurrence is used to generate sequences or reduce orders when
    necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    `ive` is useful for large arguments `z`: for these, `iv` easily overflows,
    while `ive` does not due to the exponential scaling.

    See also
    --------
    iv: Modified Bessel function of the first kind
    i0e: Faster implementation of this function for order 0
    i1e: Faster implementation of this function for order 1

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    In the following example `iv` returns infinity whereas `ive` still returns
    a finite number.

    >>> from scipy.special import iv, ive
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> iv(3, 1000.), ive(3, 1000.)
    (inf, 0.01256056218254712)

    Evaluate the function at one point for different orders by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> ive([0, 1, 1.5], 1.)
    array([0.46575961, 0.20791042, 0.10798193])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> points = np.array([-2., 0., 3.])
    >>> ive(0, points)
    array([0.30850832, 1.        , 0.24300035])

    Evaluate the function at several points for different orders by
    providing arrays for both `v` for `z`. Both arrays have to be
    broadcastable to the correct shape. To calculate the orders 0, 1
    and 2 for a 1D array of points:

    >>> ive([[0], [1], [2]], points)
    array([[ 0.30850832,  1.        ,  0.24300035],
           [-0.21526929,  0.        ,  0.19682671],
           [ 0.09323903,  0.        ,  0.11178255]])

    Plot the functions of order 0 to 3 from -5 to 5.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-5., 5., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, ive(i, x), label=f'$I_{i!r}(z)\cdot e^{{-|z|}}$')
    >>> ax.legend()
    >>> ax.set_xlabel(r"$z$")
    >>> plt.show()
    """)

add_newdoc("j0",
    r"""
    j0(x, out=None)

    Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function of the first kind of order 0 at `x`.

    Notes
    -----
    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval the following rational approximation is used:

    .. math::

        J_0(x) \approx (w - r_1^2)(w - r_2^2) \frac{P_3(w)}{Q_8(w)},

    where :math:`w = x^2` and :math:`r_1`, :math:`r_2` are the zeros of
    :math:`J_0`, and :math:`P_3` and :math:`Q_8` are polynomials of degrees 3
    and 8, respectively.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `j0`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    See also
    --------
    jv : Bessel function of real order and complex argument.
    spherical_jn : spherical Bessel functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import j0
    >>> j0(1.)
    0.7651976865579665

    Calculate the function at several points:

    >>> import numpy as np
    >>> j0(np.array([-2., 0., 4.]))
    array([ 0.22389078,  1.        , -0.39714981])

    Plot the function from -20 to 20.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-20., 20., 1000)
    >>> y = j0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("j1",
    """
    j1(x, out=None)

    Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function of the first kind of order 1 at `x`.

    Notes
    -----
    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 24 term Chebyshev expansion is used. In the second, the
    asymptotic trigonometric representation is employed using two rational
    functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `j1`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    See also
    --------
    jv: Bessel function of the first kind
    spherical_jn: spherical Bessel functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import j1
    >>> j1(1.)
    0.44005058574493355

    Calculate the function at several points:

    >>> import numpy as np
    >>> j1(np.array([-2., 0., 4.]))
    array([-0.57672481,  0.        , -0.06604333])

    Plot the function from -20 to 20.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-20., 20., 1000)
    >>> y = j1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("jn",
    """
    jn(n, x, out=None)

    Bessel function of the first kind of integer order and real argument.

    Parameters
    ----------
    n : array_like
        order of the Bessel function
    x : array_like
        argument of the Bessel function
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The value of the bessel function

    See also
    --------
    jv
    spherical_jn : spherical Bessel functions.

    Notes
    -----
    `jn` is an alias of `jv`.
    Not to be confused with the spherical Bessel functions (see
    `spherical_jn`).

    """)

add_newdoc("jv",
    r"""
    jv(v, z, out=None)

    Bessel function of the first kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the Bessel function, :math:`J_v(z)`.

    See also
    --------
    jve : :math:`J_v` with leading exponential behavior stripped off.
    spherical_jn : spherical Bessel functions.
    j0 : faster version of this function for order 0.
    j1 : faster version of this function for order 1.

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    Not to be confused with the spherical Bessel functions (see `spherical_jn`).

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import jv
    >>> jv(0, 1.)
    0.7651976865579666

    Evaluate the function at one point for different orders.

    >>> jv(0, 1.), jv(1, 1.), jv(1.5, 1.)
    (0.7651976865579666, 0.44005058574493355, 0.24029783912342725)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> jv([0, 1, 1.5], 1.)
    array([0.76519769, 0.44005059, 0.24029784])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> import numpy as np
    >>> points = np.array([-2., 0., 3.])
    >>> jv(0, points)
    array([ 0.22389078,  1.        , -0.26005195])

    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    To calculate the orders 0 and 1 for an 1D array:

    >>> orders = np.array([[0], [1]])
    >>> orders.shape
    (2, 1)

    >>> jv(orders, points)
    array([[ 0.22389078,  1.        , -0.26005195],
           [-0.57672481,  0.        ,  0.33905896]])

    Plot the functions of order 0 to 3 from -10 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, jv(i, x), label=f'$J_{i!r}$')
    >>> ax.legend()
    >>> plt.show()

    """)

add_newdoc("jve",
    r"""
    jve(v, z, out=None)

    Exponentially scaled Bessel function of the first kind of order `v`.

    Defined as::

        jve(v, z) = jv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    J : scalar or ndarray
        Value of the exponentially scaled Bessel function.

    See also
    --------
    jv: Unscaled Bessel function of the first kind

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    Exponentially scaled Bessel functions are useful for large arguments `z`:
    for these, the unscaled Bessel functions can easily under-or overflow.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compare the output of `jv` and `jve` for large complex arguments for `z`
    by computing their values for order ``v=1`` at ``z=1000j``. We see that
    `jv` overflows but `jve` returns a finite number:

    >>> import numpy as np
    >>> from scipy.special import jv, jve
    >>> v = 1
    >>> z = 1000j
    >>> jv(v, z), jve(v, z)
    ((inf+infj), (7.721967686709077e-19+0.012610930256928629j))

    For real arguments for `z`, `jve` returns the same as `jv`.

    >>> v, z = 1, 1000
    >>> jv(v, z), jve(v, z)
    (0.004728311907089523, 0.004728311907089523)

    The function can be evaluated for several orders at the same time by
    providing a list or NumPy array for `v`:

    >>> jve([1, 3, 5], 1j)
    array([1.27304208e-17+2.07910415e-01j, -4.99352086e-19-8.15530777e-03j,
           6.11480940e-21+9.98657141e-05j])

    In the same way, the function can be evaluated at several points in one
    call by providing a list or NumPy array for `z`:

    >>> jve(1, np.array([1j, 2j, 3j]))
    array([1.27308412e-17+0.20791042j, 1.31814423e-17+0.21526929j,
           1.20521602e-17+0.19682671j])

    It is also possible to evaluate several orders at several points
    at the same time by providing arrays for `v` and `z` with
    compatible shapes for broadcasting. Compute `jve` for two different orders
    `v` and three points `z` resulting in a 2x3 array.

    >>> v = np.array([[1], [3]])
    >>> z = np.array([1j, 2j, 3j])
    >>> v.shape, z.shape
    ((2, 1), (3,))

    >>> jve(v, z)
    array([[1.27304208e-17+0.20791042j,  1.31810070e-17+0.21526929j,
            1.20517622e-17+0.19682671j],
           [-4.99352086e-19-0.00815531j, -1.76289571e-18-0.02879122j,
            -2.92578784e-18-0.04778332j]])
    """)

add_newdoc("k0",
    r"""
    k0(x, out=None)

    Modified Bessel function of the second kind of order 0, :math:`K_0`.

    This function is also sometimes referred to as the modified Bessel
    function of the third kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the modified Bessel function :math:`K_0` at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0`.

    See also
    --------
    kv: Modified Bessel function of the second kind of any order
    k0e: Exponentially scaled modified Bessel function of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import k0
    >>> k0(1.)
    0.42102443824070823

    Calculate the function at several points:

    >>> import numpy as np
    >>> k0(np.array([0.5, 2., 3.]))
    array([0.92441907, 0.11389387, 0.0347395 ])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("k0e",
    """
    k0e(x, out=None)

    Exponentially scaled modified Bessel function K of order 0

    Defined as::

        k0e(x) = exp(x) * k0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the exponentially scaled modified Bessel function K of order
        0 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0e`. `k0e` is
    useful for large arguments: for these, `k0` easily underflows.

    See also
    --------
    kv: Modified Bessel function of the second kind of any order
    k0: Modified Bessel function of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `k0` returns 0 whereas `k0e` still returns a
    useful finite number:

    >>> from scipy.special import k0, k0e
    >>> k0(1000.), k0e(1000)
    (0., 0.03962832160075422)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> k0e(np.array([0.5, 2., 3.]))
    array([1.52410939, 0.84156822, 0.6977616 ])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k0e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
    """)

add_newdoc("k1",
    """
    k1(x, out=None)

    Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the modified Bessel function K of order 1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1`.

    See also
    --------
    kv: Modified Bessel function of the second kind of any order
    k1e: Exponentially scaled modified Bessel function K of order 1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import k1
    >>> k1(1.)
    0.6019072301972346

    Calculate the function at several points:

    >>> import numpy as np
    >>> k1(np.array([0.5, 2., 3.]))
    array([1.65644112, 0.13986588, 0.04015643])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("k1e",
    """
    k1e(x, out=None)

    Exponentially scaled modified Bessel function K of order 1

    Defined as::

        k1e(x) = exp(x) * k1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    K : scalar or ndarray
        Value of the exponentially scaled modified Bessel function K of order
        1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1e`.

    See also
    --------
    kv: Modified Bessel function of the second kind of any order
    k1: Modified Bessel function of the second kind of order 1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    In the following example `k1` returns 0 whereas `k1e` still returns a
    useful floating point number.

    >>> from scipy.special import k1, k1e
    >>> k1(1000.), k1e(1000.)
    (0., 0.03964813081296021)

    Calculate the function at several points by providing a NumPy array or
    list for `x`:

    >>> import numpy as np
    >>> k1e(np.array([0.5, 2., 3.]))
    array([2.73100971, 1.03347685, 0.80656348])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = k1e(x)
    >>> ax.plot(x, y)
    >>> plt.show()
    """)

add_newdoc("kei",
    r"""
    kei(x, out=None)

    Kelvin function kei.

    Defined as

    .. math::

        \mathrm{kei}(x) = \Im[K_0(x e^{\pi i / 4})]

    where :math:`K_0` is the modified Bessel function of the second
    kind (see `kv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    ker : the corresponding real part
    keip : the derivative of kei
    kv : modified Bessel function of the second kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using the modified Bessel function of the
    second kind.

    >>> import numpy as np
    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.kv(0, x * np.exp(np.pi * 1j / 4)).imag
    array([-0.49499464, -0.20240007, -0.05112188,  0.0021984 ])
    >>> sc.kei(x)
    array([-0.49499464, -0.20240007, -0.05112188,  0.0021984 ])

    """)

add_newdoc("keip",
    r"""
    keip(x, out=None)

    Derivative of the Kelvin function kei.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of kei.

    See Also
    --------
    kei

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("kelvin",
    """
    kelvin(x, out=None)

    Kelvin functions as complex numbers

    Parameters
    ----------
    x : array_like
        Argument
    out : tuple of ndarray, optional
        Optional output arrays for the function values

    Returns
    -------
    Be, Ke, Bep, Kep : 4-tuple of scalar or ndarray
        The tuple (Be, Ke, Bep, Kep) contains complex numbers
        representing the real and imaginary Kelvin functions and their
        derivatives evaluated at `x`.  For example, kelvin(x)[0].real =
        ber x and kelvin(x)[0].imag = bei x with similar relationships
        for ker and kei.
    """)

add_newdoc("ker",
    r"""
    ker(x, out=None)

    Kelvin function ker.

    Defined as

    .. math::

        \mathrm{ker}(x) = \Re[K_0(x e^{\pi i / 4})]

    Where :math:`K_0` is the modified Bessel function of the second
    kind (see `kv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    kei : the corresponding imaginary part
    kerp : the derivative of ker
    kv : modified Bessel function of the second kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using the modified Bessel function of the
    second kind.

    >>> import numpy as np
    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.kv(0, x * np.exp(np.pi * 1j / 4)).real
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])
    >>> sc.ker(x)
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])

    """)

add_newdoc("kerp",
    r"""
    kerp(x, out=None)

    Derivative of the Kelvin function ker.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of ker.

    See Also
    --------
    ker

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("kl_div",
    r"""
    kl_div(x, y, out=None)

    Elementwise function for computing Kullback-Leibler divergence.

    .. math::

        \mathrm{kl\_div}(x, y) =
          \begin{cases}
            x \log(x / y) - x + y & x > 0, y > 0 \\
            y & x = 0, y \ge 0 \\
            \infty & \text{otherwise}
          \end{cases}

    Parameters
    ----------
    x, y : array_like
        Real arguments
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Kullback-Liebler divergence.

    See Also
    --------
    entr, rel_entr, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is non-negative and is jointly convex in `x` and `y`.

    The origin of this function is in convex programming; see [1]_ for
    details. This is why the function contains the extra :math:`-x
    + y` terms over what might be expected from the Kullback-Leibler
    divergence. For a version of the function without the extra terms,
    see `rel_entr`.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`

    """)

add_newdoc("kn",
    r"""
    kn(n, x, out=None)

    Modified Bessel function of the second kind of integer order `n`

    Returns the modified Bessel function of the second kind for integer order
    `n` at real `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.

    Parameters
    ----------
    n : array_like of int
        Order of Bessel functions (floats will truncate with a warning)
    x : array_like of float
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Value of the Modified Bessel function of the second kind,
        :math:`K_n(x)`.

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    See Also
    --------
    kv : Same function, but accepts real order and complex argument
    kvp : Derivative of this function

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    Examples
    --------
    Plot the function of several orders for real input:

    >>> import numpy as np
    >>> from scipy.special import kn
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in range(6):
    ...     plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kn([4, 5, 6], 1)
    array([   44.23241585,   360.9605896 ,  3653.83831186])
    """)

add_newdoc("kolmogi",
    """
    kolmogi(p, out=None)

    Inverse Survival Function of Kolmogorov distribution

    It is the inverse function to `kolmogorov`.
    Returns y such that ``kolmogorov(y) == p``.

    Parameters
    ----------
    p : float array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of kolmogi(p)

    Notes
    -----
    `kolmogorov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.kstwobign` distribution.

    See Also
    --------
    kolmogorov : The Survival Function for the distribution
    scipy.stats.kstwobign : Provides the functionality as a continuous distribution
    smirnov, smirnovi : Functions for the one-sided distribution

    Examples
    --------
    >>> from scipy.special import kolmogi
    >>> kolmogi([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    array([        inf,  1.22384787,  1.01918472,  0.82757356,  0.67644769,
            0.57117327,  0.        ])

    """)

add_newdoc("kolmogorov",
    r"""
    kolmogorov(y, out=None)

    Complementary cumulative distribution (Survival Function) function of
    Kolmogorov distribution.

    Returns the complementary cumulative distribution function of
    Kolmogorov's limiting distribution (``D_n*\sqrt(n)`` as n goes to infinity)
    of a two-sided test for equality between an empirical and a theoretical
    distribution. It is equal to the (limit as n->infinity of the)
    probability that ``sqrt(n) * max absolute deviation > y``.

    Parameters
    ----------
    y : float array_like
      Absolute deviation between the Empirical CDF (ECDF) and the target CDF,
      multiplied by sqrt(n).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of kolmogorov(y)

    Notes
    -----
    `kolmogorov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.kstwobign` distribution.

    See Also
    --------
    kolmogi : The Inverse Survival Function for the distribution
    scipy.stats.kstwobign : Provides the functionality as a continuous distribution
    smirnov, smirnovi : Functions for the one-sided distribution

    Examples
    --------
    Show the probability of a gap at least as big as 0, 0.5 and 1.0.

    >>> import numpy as np
    >>> from scipy.special import kolmogorov
    >>> from scipy.stats import kstwobign
    >>> kolmogorov([0, 0.5, 1.0])
    array([ 1.        ,  0.96394524,  0.26999967])

    Compare a sample of size 1000 drawn from a Laplace(0, 1) distribution against
    the target distribution, a Normal(0, 1) distribution.

    >>> from scipy.stats import norm, laplace
    >>> rng = np.random.default_rng()
    >>> n = 1000
    >>> lap01 = laplace(0, 1)
    >>> x = np.sort(lap01.rvs(n, random_state=rng))
    >>> np.mean(x), np.std(x)
    (-0.05841730131499543, 1.3968109101997568)

    Construct the Empirical CDF and the K-S statistic Dn.

    >>> target = norm(0,1)  # Normal mean 0, stddev 1
    >>> cdfs = target.cdf(x)
    >>> ecdfs = np.arange(n+1, dtype=float)/n
    >>> gaps = np.column_stack([cdfs - ecdfs[:n], ecdfs[1:] - cdfs])
    >>> Dn = np.max(gaps)
    >>> Kn = np.sqrt(n) * Dn
    >>> print('Dn=%f, sqrt(n)*Dn=%f' % (Dn, Kn))
    Dn=0.043363, sqrt(n)*Dn=1.371265
    >>> print(chr(10).join(['For a sample of size n drawn from a N(0, 1) distribution:',
    ...   ' the approximate Kolmogorov probability that sqrt(n)*Dn>=%f is %f' %  (Kn, kolmogorov(Kn)),
    ...   ' the approximate Kolmogorov probability that sqrt(n)*Dn<=%f is %f' %  (Kn, kstwobign.cdf(Kn))]))
    For a sample of size n drawn from a N(0, 1) distribution:
     the approximate Kolmogorov probability that sqrt(n)*Dn>=1.371265 is 0.046533
     the approximate Kolmogorov probability that sqrt(n)*Dn<=1.371265 is 0.953467

    Plot the Empirical CDF against the target N(0, 1) CDF.

    >>> import matplotlib.pyplot as plt
    >>> plt.step(np.concatenate([[-3], x]), ecdfs, where='post', label='Empirical CDF')
    >>> x3 = np.linspace(-3, 3, 100)
    >>> plt.plot(x3, target.cdf(x3), label='CDF for N(0, 1)')
    >>> plt.ylim([0, 1]); plt.grid(True); plt.legend();
    >>> # Add vertical lines marking Dn+ and Dn-
    >>> iminus, iplus = np.argmax(gaps, axis=0)
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color='r', linestyle='dashed', lw=4)
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1], color='r', linestyle='dashed', lw=4)
    >>> plt.show()
    """)

add_newdoc("_kolmogc",
    r"""
    Internal function, do not use.
    """)

add_newdoc("_kolmogci",
    r"""
    Internal function, do not use.
    """)

add_newdoc("_kolmogp",
    r"""
    Internal function, do not use.
    """)

add_newdoc("kv",
    r"""
    kv(v, z, out=None)

    Modified Bessel function of the second kind of real order `v`

    Returns the modified Bessel function of the second kind for real order
    `v` at complex `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.  They are defined as those solutions
    of the modified Bessel equation for which,

    .. math::
        K_v(x) \sim \sqrt{\pi/(2x)} \exp(-x)

    as :math:`x \to \infty` [3]_.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The results. Note that input must be of complex type to get complex
        output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    See Also
    --------
    kve : This function with leading exponential behavior stripped off.
    kvp : Derivative of this function

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
    .. [3] NIST Digital Library of Mathematical Functions,
           Eq. 10.25.E3. https://dlmf.nist.gov/10.25.E3

    Examples
    --------
    Plot the function of several orders for real input:

    >>> import numpy as np
    >>> from scipy.special import kv
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in np.linspace(0, 6, 5):
    ...     plt.plot(x, kv(N, x), label='$K_{{{}}}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_\nu(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kv([4, 4.5, 5], 1+2j)
    array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])

    """)

add_newdoc("kve",
    r"""
    kve(v, z, out=None)

    Exponentially scaled modified Bessel function of the second kind.

    Returns the exponentially scaled, modified Bessel function of the
    second kind (sometimes called the third kind) for real order `v` at
    complex `z`::

        kve(v, z) = kv(v, z) * exp(z)

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The exponentially scaled modified Bessel function of the second kind.

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    See Also
    --------
    kv : This function without exponential scaling.
    k0e : Faster version of this function for order 0.
    k1e : Faster version of this function for order 1.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    Examples
    --------
    In the following example `kv` returns 0 whereas `kve` still returns
    a useful finite number.

    >>> import numpy as np
    >>> from scipy.special import kv, kve
    >>> import matplotlib.pyplot as plt
    >>> kv(3, 1000.), kve(3, 1000.)
    (0.0, 0.03980696128440973)

    Evaluate the function at one point for different orders by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> kve([0, 1, 1.5], 1.)
    array([1.14446308, 1.63615349, 2.50662827])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> points = np.array([1., 3., 10.])
    >>> kve(0, points)
    array([1.14446308, 0.6977616 , 0.39163193])

    Evaluate the function at several points for different orders by
    providing arrays for both `v` for `z`. Both arrays have to be
    broadcastable to the correct shape. To calculate the orders 0, 1
    and 2 for a 1D array of points:

    >>> kve([[0], [1], [2]], points)
    array([[1.14446308, 0.6977616 , 0.39163193],
           [1.63615349, 0.80656348, 0.41076657],
           [4.41677005, 1.23547058, 0.47378525]])

    Plot the functions of order 0 to 3 from 0 to 5.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 5., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, kve(i, x), label=f'$K_{i!r}(z)\cdot e^z$')
    >>> ax.legend()
    >>> ax.set_xlabel(r"$z$")
    >>> ax.set_ylim(0, 4)
    >>> ax.set_xlim(0, 5)
    >>> plt.show()
    """)

add_newdoc("_lanczos_sum_expg_scaled",
    """
    Internal function, do not use.
    """)

add_newdoc("_lgam1p",
    """
    Internal function, do not use.
    """)

add_newdoc("log1p",
    """
    log1p(x, out=None)

    Calculates log(1 + x) for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real or complex valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``log(1 + x)``.

    See Also
    --------
    expm1, cosm1

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using ``log(1 + x)`` directly for ``x``
    near 0. Note that in the below example ``1 + 1e-17 == 1`` to
    double precision.

    >>> sc.log1p(1e-17)
    1e-17
    >>> np.log(1 + 1e-17)
    0.0

    """)

add_newdoc("_log1pmx",
    """
    Internal function, do not use.
    """)

add_newdoc('log_expit',
    """
    log_expit(x, out=None)

    Logarithm of the logistic sigmoid function.

    The SciPy implementation of the logistic sigmoid function is
    `scipy.special.expit`, so this function is called ``log_expit``.

    The function is mathematically equivalent to ``log(expit(x))``, but
    is formulated to avoid loss of precision for inputs with large
    (positive or negative) magnitude.

    Parameters
    ----------
    x : array_like
        The values to apply ``log_expit`` to element-wise.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    out : scalar or ndarray
        The computed values, an ndarray of the same shape as ``x``.

    See Also
    --------
    expit

    Notes
    -----
    As a ufunc, ``log_expit`` takes a number of optional keyword arguments.
    For more information see
    `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_

    .. versionadded:: 1.8.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import log_expit, expit

    >>> log_expit([-3.0, 0.25, 2.5, 5.0])
    array([-3.04858735, -0.57593942, -0.07888973, -0.00671535])

    Large negative values:

    >>> log_expit([-100, -500, -1000])
    array([ -100.,  -500., -1000.])

    Note that ``expit(-1000)`` returns 0, so the naive implementation
    ``log(expit(-1000))`` return ``-inf``.

    Large positive values:

    >>> log_expit([29, 120, 400])
    array([-2.54366565e-013, -7.66764807e-053, -1.91516960e-174])

    Compare that to the naive implementation:

    >>> np.log(expit([29, 120, 400]))
    array([-2.54463117e-13,  0.00000000e+00,  0.00000000e+00])

    The first value is accurate to only 3 digits, and the larger inputs
    lose all precision and return 0.
    """)

add_newdoc('logit',
    """
    logit(x, out=None)

    Logit ufunc for ndarrays.

    The logit function is defined as logit(p) = log(p/(1-p)).
    Note that logit(0) = -inf, logit(1) = inf, and logit(p)
    for p<0 or p>1 yields nan.

    Parameters
    ----------
    x : ndarray
        The ndarray to apply logit to element-wise.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        An ndarray of the same shape as x. Its entries
        are logit of the corresponding entry of x.

    See Also
    --------
    expit

    Notes
    -----
    As a ufunc logit takes a number of optional
    keyword arguments. For more information
    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_

    .. versionadded:: 0.10.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import logit, expit

    >>> logit([0, 0.25, 0.5, 0.75, 1])
    array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])

    `expit` is the inverse of `logit`:

    >>> expit(logit([0.1, 0.75, 0.999]))
    array([ 0.1  ,  0.75 ,  0.999])

    Plot logit(x) for x in [0, 1]:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1, 501)
    >>> y = logit(x)
    >>> plt.plot(x, y)
    >>> plt.grid()
    >>> plt.ylim(-6, 6)
    >>> plt.xlabel('x')
    >>> plt.title('logit(x)')
    >>> plt.show()

    """)

add_newdoc("lpmv",
    r"""
    lpmv(m, v, x, out=None)

    Associated Legendre function of integer order and real degree.

    Defined as

    .. math::

        P_v^m = (-1)^m (1 - x^2)^{m/2} \frac{d^m}{dx^m} P_v(x)

    where

    .. math::

        P_v = \sum_{k = 0}^\infty \frac{(-v)_k (v + 1)_k}{(k!)^2}
                \left(\frac{1 - x}{2}\right)^k

    is the Legendre function of the first kind. Here :math:`(\cdot)_k`
    is the Pochhammer symbol; see `poch`.

    Parameters
    ----------
    m : array_like
        Order (int or float). If passed a float not equal to an
        integer the function returns NaN.
    v : array_like
        Degree (float).
    x : array_like
        Argument (float). Must have ``|x| <= 1``.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    pmv : scalar or ndarray
        Value of the associated Legendre function.

    See Also
    --------
    lpmn : Compute the associated Legendre function for all orders
           ``0, ..., m`` and degrees ``0, ..., n``.
    clpmn : Compute the associated Legendre function at complex
            arguments.

    Notes
    -----
    Note that this implementation includes the Condon-Shortley phase.

    References
    ----------
    .. [1] Zhang, Jin, "Computation of Special Functions", John Wiley
           and Sons, Inc, 1996.

    """)

add_newdoc("mathieu_a",
    """
    mathieu_a(m, q, out=None)

    Characteristic value of even Mathieu functions

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Characteristic value for the even solution, ``ce_m(z, q)``, of
        Mathieu's equation.

    See Also
    --------
    mathieu_b, mathieu_cem, mathieu_sem

    """)

add_newdoc("mathieu_b",
    """
    mathieu_b(m, q, out=None)

    Characteristic value of odd Mathieu functions

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Characteristic value for the odd solution, ``se_m(z, q)``, of Mathieu's
        equation.

    See Also
    --------
    mathieu_a, mathieu_cem, mathieu_sem

    """)

add_newdoc("mathieu_cem",
    """
    mathieu_cem(m, q, x, out=None)

    Even Mathieu function and its derivative

    Returns the even Mathieu function, ``ce_m(x, q)``, of order `m` and
    parameter `q` evaluated at `x` (given in degrees).  Also returns the
    derivative with respect to `x` of ce_m(x, q)

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_a, mathieu_b, mathieu_sem

    """)

add_newdoc("mathieu_modcem1",
    """
    mathieu_modcem1(m, q, x, out=None)

    Even modified Mathieu function of the first kind and its derivative

    Evaluates the even modified Mathieu function of the first kind,
    ``Mc1m(x, q)``, and its derivative at `x` for order `m` and parameter
    `q`.

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_modsem1

    """)

add_newdoc("mathieu_modcem2",
    """
    mathieu_modcem2(m, q, x, out=None)

    Even modified Mathieu function of the second kind and its derivative

    Evaluates the even modified Mathieu function of the second kind,
    Mc2m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter `q`.

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_modsem2

    """)

add_newdoc("mathieu_modsem1",
    """
    mathieu_modsem1(m, q, x, out=None)

    Odd modified Mathieu function of the first kind and its derivative

    Evaluates the odd modified Mathieu function of the first kind,
    Ms1m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter `q`.

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_modcem1

    """)

add_newdoc("mathieu_modsem2",
    """
    mathieu_modsem2(m, q, x, out=None)

    Odd modified Mathieu function of the second kind and its derivative

    Evaluates the odd modified Mathieu function of the second kind,
    Ms2m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter q.

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_modcem2

    """)

add_newdoc(
    "mathieu_sem",
    """
    mathieu_sem(m, q, x, out=None)

    Odd Mathieu function and its derivative

    Returns the odd Mathieu function, se_m(x, q), of order `m` and
    parameter `q` evaluated at `x` (given in degrees).  Also returns the
    derivative with respect to `x` of se_m(x, q).

    Parameters
    ----------
    m : array_like
        Order of the function
    q : array_like
        Parameter of the function
    x : array_like
        Argument of the function, *given in degrees, not radians*.
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    y : scalar or ndarray
        Value of the function
    yp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    mathieu_a, mathieu_b, mathieu_cem

    """)

add_newdoc("modfresnelm",
    """
    modfresnelm(x, out=None)

    Modified Fresnel negative integrals

    Parameters
    ----------
    x : array_like
        Function argument
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    fm : scalar or ndarray
        Integral ``F_-(x)``: ``integral(exp(-1j*t*t), t=x..inf)``
    km : scalar or ndarray
        Integral ``K_-(x)``: ``1/sqrt(pi)*exp(1j*(x*x+pi/4))*fp``

    See Also
    --------
    modfresnelp

    """)

add_newdoc("modfresnelp",
    """
    modfresnelp(x, out=None)

    Modified Fresnel positive integrals

    Parameters
    ----------
    x : array_like
        Function argument
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    fp : scalar or ndarray
        Integral ``F_+(x)``: ``integral(exp(1j*t*t), t=x..inf)``
    kp : scalar or ndarray
        Integral ``K_+(x)``: ``1/sqrt(pi)*exp(-1j*(x*x+pi/4))*fp``

    See Also
    --------
    modfresnelm

    """)

add_newdoc("modstruve",
    r"""
    modstruve(v, x, out=None)

    Modified Struve function.

    Return the value of the modified Struve function of order `v` at `x`.  The
    modified Struve function is defined as,

    .. math::
        L_v(x) = -\imath \exp(-\pi\imath v/2) H_v(\imath x),

    where :math:`H_v` is the Struve function.

    Parameters
    ----------
    v : array_like
        Order of the modified Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    L : scalar or ndarray
        Value of the modified Struve function of order `v` at `x`.

    Notes
    -----
    Three methods discussed in [1]_ are used to evaluate the function:

    - power series
    - expansion in Bessel functions (if :math:`|x| < |v| + 20`)
    - asymptotic large-x expansion (if :math:`x \geq 0.7v + 12`)

    Rounding errors are estimated based on the largest terms in the sums, and
    the result associated with the smallest error is returned.

    See also
    --------
    struve

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/11

    Examples
    --------
    Calculate the modified Struve function of order 1 at 2.

    >>> import numpy as np
    >>> from scipy.special import modstruve
    >>> import matplotlib.pyplot as plt
    >>> modstruve(1, 2.)
    1.102759787367716

    Calculate the modified Struve function at 2 for orders 1, 2 and 3 by
    providing a list for the order parameter `v`.

    >>> modstruve([1, 2, 3], 2.)
    array([1.10275979, 0.41026079, 0.11247294])

    Calculate the modified Struve function of order 1 for several points
    by providing an array for `x`.

    >>> points = np.array([2., 5., 8.])
    >>> modstruve(1, points)
    array([  1.10275979,  23.72821578, 399.24709139])

    Compute the modified Struve function for several orders at several
    points by providing arrays for `v` and `z`. The arrays have to be
    broadcastable to the correct shapes.

    >>> orders = np.array([[1], [2], [3]])
    >>> points.shape, orders.shape
    ((3,), (3, 1))

    >>> modstruve(orders, points)
    array([[1.10275979e+00, 2.37282158e+01, 3.99247091e+02],
           [4.10260789e-01, 1.65535979e+01, 3.25973609e+02],
           [1.12472937e-01, 9.42430454e+00, 2.33544042e+02]])

    Plot the modified Struve functions of order 0 to 3 from -5 to 5.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-5., 5., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, modstruve(i, x), label=f'$L_{i!r}$')
    >>> ax.legend(ncol=2)
    >>> ax.set_xlim(-5, 5)
    >>> ax.set_title(r"Modified Struve functions $L_{\nu}$")
    >>> plt.show()
    """)

add_newdoc("nbdtr",
    r"""
    nbdtr(k, n, p, out=None)

    Negative binomial cumulative distribution function.

    Returns the sum of the terms 0 through `k` of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=0}^k {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that `k` or fewer failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    F : scalar or ndarray
        The probability of `k` or fewer failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See also
    --------
    nbdtrc : Negative binomial survival function
    nbdtrik : Negative binomial quantile function
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtr}(k, n, p) = I_{p}(n, k + 1).

    Wrapper for the Cephes [1]_ routine `nbdtr`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtr` directly can improve performance
    compared to the ``cdf`` method of `scipy.stats.nbinom` (see last example).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``k=10`` and ``n=5`` at ``p=0.5``.

    >>> import numpy as np
    >>> from scipy.special import nbdtr
    >>> nbdtr(10, 5, 0.5)
    0.940765380859375

    Compute the function for ``n=10`` and ``p=0.5`` at several points by
    providing a NumPy array or list for `k`.

    >>> nbdtr([5, 10, 15], 10, 0.5)
    array([0.15087891, 0.58809853, 0.88523853])

    Plot the function for four different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> k = np.arange(130)
    >>> n_parameters = [20, 20, 20, 80]
    >>> p_parameters = [0.2, 0.5, 0.8, 0.5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters,
    ...                            linestyles))
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtr_vals = nbdtr(k, n, p)
    ...     ax.plot(k, nbdtr_vals, label=rf"$n={n},\, p={p}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$k$")
    >>> ax.set_title("Negative binomial cumulative distribution function")
    >>> plt.show()

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtr` directly can be much faster than
    calling the ``cdf`` method of `scipy.stats.nbinom`, especially for small
    arrays or individual values. To get the same results one must use the
    following parametrization: ``nbinom(n, p).cdf(k)=nbdtr(k, n, p)``.

    >>> from scipy.stats import nbinom
    >>> k, n, p = 5, 3, 0.5
    >>> nbdtr_res = nbdtr(k, n, p)  # this will often be faster than below
    >>> stats_res = nbinom(n, p).cdf(k)
    >>> stats_res, nbdtr_res  # test that results are equal
    (0.85546875, 0.85546875)

    `nbdtr` can evaluate different parameter sets by providing arrays with
    shapes compatible for broadcasting for `k`, `n` and `p`. Here we compute
    the function for three different `k` at four locations `p`, resulting in
    a 3x4 array.

    >>> k = np.array([[5], [10], [15]])
    >>> p = np.array([0.3, 0.5, 0.7, 0.9])
    >>> k.shape, p.shape
    ((3, 1), (4,))

    >>> nbdtr(k, 5, p)
    array([[0.15026833, 0.62304687, 0.95265101, 0.9998531 ],
           [0.48450894, 0.94076538, 0.99932777, 0.99999999],
           [0.76249222, 0.99409103, 0.99999445, 1.        ]])
    """)

add_newdoc("nbdtrc",
    r"""
    nbdtrc(k, n, p, out=None)

    Negative binomial survival function.

    Returns the sum of the terms `k + 1` to infinity of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that more than `k` failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    F : scalar or ndarray
        The probability of `k + 1` or more failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See also
    --------
    nbdtr : Negative binomial cumulative distribution function
    nbdtrik : Negative binomial percentile function
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtrc}(k, n, p) = I_{1 - p}(k + 1, n).

    Wrapper for the Cephes [1]_ routine `nbdtrc`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtrc` directly can improve performance
    compared to the ``sf`` method of `scipy.stats.nbinom` (see last example).

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Compute the function for ``k=10`` and ``n=5`` at ``p=0.5``.

    >>> import numpy as np
    >>> from scipy.special import nbdtrc
    >>> nbdtrc(10, 5, 0.5)
    0.059234619140624986

    Compute the function for ``n=10`` and ``p=0.5`` at several points by
    providing a NumPy array or list for `k`.

    >>> nbdtrc([5, 10, 15], 10, 0.5)
    array([0.84912109, 0.41190147, 0.11476147])

    Plot the function for four different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> k = np.arange(130)
    >>> n_parameters = [20, 20, 20, 80]
    >>> p_parameters = [0.2, 0.5, 0.8, 0.5]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters,
    ...                            linestyles))
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtrc_vals = nbdtrc(k, n, p)
    ...     ax.plot(k, nbdtrc_vals, label=rf"$n={n},\, p={p}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_xlabel("$k$")
    >>> ax.set_title("Negative binomial distribution survival function")
    >>> plt.show()

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtrc` directly can be much faster than
    calling the ``sf`` method of `scipy.stats.nbinom`, especially for small
    arrays or individual values. To get the same results one must use the
    following parametrization: ``nbinom(n, p).sf(k)=nbdtrc(k, n, p)``.

    >>> from scipy.stats import nbinom
    >>> k, n, p = 3, 5, 0.5
    >>> nbdtr_res = nbdtrc(k, n, p)  # this will often be faster than below
    >>> stats_res = nbinom(n, p).sf(k)
    >>> stats_res, nbdtr_res  # test that results are equal
    (0.6367187499999999, 0.6367187499999999)

    `nbdtrc` can evaluate different parameter sets by providing arrays with
    shapes compatible for broadcasting for `k`, `n` and `p`. Here we compute
    the function for three different `k` at four locations `p`, resulting in
    a 3x4 array.

    >>> k = np.array([[5], [10], [15]])
    >>> p = np.array([0.3, 0.5, 0.7, 0.9])
    >>> k.shape, p.shape
    ((3, 1), (4,))

    >>> nbdtrc(k, 5, p)
    array([[8.49731667e-01, 3.76953125e-01, 4.73489874e-02, 1.46902600e-04],
           [5.15491059e-01, 5.92346191e-02, 6.72234070e-04, 9.29610100e-09],
           [2.37507779e-01, 5.90896606e-03, 5.55025308e-06, 3.26346760e-13]])
    """)

add_newdoc(
    "nbdtri",
    r"""
    nbdtri(k, n, y, out=None)

    Returns the inverse with respect to the parameter `p` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    p : scalar or ndarray
        Probability of success in a single event (float) such that
        `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtrc : Negative binomial survival function.
    scipy.stats.nbinom : negative binomial distribution.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `nbdtri`.

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. Using `nbdtri` directly can improve performance
    compared to the ``ppf`` method of `scipy.stats.nbinom`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    `nbdtri` is the inverse of `nbdtr` with respect to `p`.
    Up to floating point errors the following holds:
    ``nbdtri(k, n, nbdtr(k, n, p))=p``.

    >>> import numpy as np
    >>> from scipy.special import nbdtri, nbdtr
    >>> k, n, y = 5, 10, 0.2
    >>> cdf_val = nbdtr(k, n, y)
    >>> nbdtri(k, n, cdf_val)
    0.20000000000000004

    Compute the function for ``k=10`` and ``n=5`` at several points by
    providing a NumPy array or list for `y`.

    >>> y = np.array([0.1, 0.4, 0.8])
    >>> nbdtri(3, 5, y)
    array([0.34462319, 0.51653095, 0.69677416])

    Plot the function for three different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> n_parameters = [5, 20, 30, 30]
    >>> k_parameters = [20, 20, 60, 80]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(n_parameters, k_parameters, linestyles))
    >>> cdf_vals = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     n, k, style = parameter_set
    ...     nbdtri_vals = nbdtri(k, n, cdf_vals)
    ...     ax.plot(cdf_vals, nbdtri_vals, label=rf"$k={k},\ n={n}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_ylabel("$p$")
    >>> ax.set_xlabel("$CDF$")
    >>> title = "nbdtri: inverse of negative binomial CDF with respect to $p$"
    >>> ax.set_title(title)
    >>> plt.show()

    `nbdtri` can evaluate different parameter sets by providing arrays with
    shapes compatible for broadcasting for `k`, `n` and `p`. Here we compute
    the function for three different `k` at four locations `p`, resulting in
    a 3x4 array.

    >>> k = np.array([[5], [10], [15]])
    >>> y = np.array([0.3, 0.5, 0.7, 0.9])
    >>> k.shape, y.shape
    ((3, 1), (4,))

    >>> nbdtri(k, 5, y)
    array([[0.37258157, 0.45169416, 0.53249956, 0.64578407],
           [0.24588501, 0.30451981, 0.36778453, 0.46397088],
           [0.18362101, 0.22966758, 0.28054743, 0.36066188]])
    """)

add_newdoc("nbdtrik",
    r"""
    nbdtrik(y, n, p, out=None)

    Negative binomial percentile function.

    Returns the inverse with respect to the parameter `k` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    k : scalar or ndarray
        The maximum number of allowed failures such that `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtrc : Survival function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
    scipy.stats.nbinom : Negative binomial distribution

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `k`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Compute the negative binomial cumulative distribution function for an
    exemplary parameter set.

    >>> import numpy as np
    >>> from scipy.special import nbdtr, nbdtrik
    >>> k, n, p = 5, 2, 0.5
    >>> cdf_value = nbdtr(k, n, p)
    >>> cdf_value
    0.9375

    Verify that `nbdtrik` recovers the original value for `k`.

    >>> nbdtrik(cdf_value, n, p)
    5.0

    Plot the function for different parameter sets.

    >>> import matplotlib.pyplot as plt
    >>> p_parameters = [0.2, 0.5, 0.7, 0.5]
    >>> n_parameters = [30, 30, 30, 80]
    >>> linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    >>> parameters_list = list(zip(p_parameters, n_parameters, linestyles))
    >>> cdf_vals = np.linspace(0, 1, 1000)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for parameter_set in parameters_list:
    ...     p, n, style = parameter_set
    ...     nbdtrik_vals = nbdtrik(cdf_vals, n, p)
    ...     ax.plot(cdf_vals, nbdtrik_vals, label=rf"$n={n},\ p={p}$",
    ...             ls=style)
    >>> ax.legend()
    >>> ax.set_ylabel("$k$")
    >>> ax.set_xlabel("$CDF$")
    >>> ax.set_title("Negative binomial percentile function")
    >>> plt.show()

    The negative binomial distribution is also available as
    `scipy.stats.nbinom`. The percentile function  method ``ppf``
    returns the result of `nbdtrik` rounded up to integers:

    >>> from scipy.stats import nbinom
    >>> q, n, p = 0.6, 5, 0.5
    >>> nbinom.ppf(q, n, p), nbdtrik(q, n, p)
    (5.0, 4.800428460273882)

    """)

add_newdoc("nbdtrin",
    r"""
    nbdtrin(k, y, p, out=None)

    Inverse of `nbdtr` vs `n`.

    Returns the inverse with respect to the parameter `n` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    p : array_like
        Probability of success in a single event (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    n : scalar or ndarray
        The number of successes `n` such that `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `n`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Compute the negative binomial cumulative distribution function for an
    exemplary parameter set.

    >>> from scipy.special import nbdtr, nbdtrin
    >>> k, n, p = 5, 2, 0.5
    >>> cdf_value = nbdtr(k, n, p)
    >>> cdf_value
    0.9375

    Verify that `nbdtrin` recovers the original value for `n` up to floating
    point accuracy.

    >>> nbdtrin(k, cdf_value, p)
    1.999999999998137
    """)

add_newdoc("ncfdtr",
    r"""
    ncfdtr(dfn, dfd, nc, f, out=None)

    Cumulative distribution function of the non-central F distribution.

    The non-central F describes the distribution of,

    .. math::
        Z = \frac{X/d_n}{Y/d_d}

    where :math:`X` and :math:`Y` are independently distributed, with
    :math:`X` distributed non-central :math:`\chi^2` with noncentrality
    parameter `nc` and :math:`d_n` degrees of freedom, and :math:`Y`
    distributed :math:`\chi^2` with :math:`d_d` degrees of freedom.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e. the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        The calculated CDF.  If all inputs are scalar, the return will be a
        float.  Otherwise it will be an array.

    See Also
    --------
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdffnc`.

    The cumulative distribution function is computed using Formula 26.6.20 of
    [2]_:

    .. math::
        F(d_n, d_d, n_c, f) = \sum_{j=0}^\infty e^{-n_c/2} \frac{(n_c/2)^j}{j!} I_{x}(\frac{d_n}{2} + j, \frac{d_d}{2}),

    where :math:`I` is the regularized incomplete beta function, and
    :math:`x = f d_n/(f d_n + d_d)`.

    The computation time required for this routine is proportional to the
    noncentrality parameter `nc`.  Very large values of this parameter can
    consume immense computer resources.  This is why the search range is
    bounded by 10,000.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central F distribution, for nc=0.  Compare with the
    F-distribution from scipy.stats:

    >>> x = np.linspace(-1, 8, num=500)
    >>> dfn = 3
    >>> dfd = 2
    >>> ncf_stats = stats.f.cdf(x, dfn, dfd)
    >>> ncf_special = special.ncfdtr(dfn, dfd, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, ncf_stats, 'b-', lw=3)
    >>> ax.plot(x, ncf_special, 'r-')
    >>> plt.show()

    """)

add_newdoc("ncfdtri",
    """
    ncfdtri(dfn, dfd, nc, p, out=None)

    Inverse with respect to `f` of the CDF of the non-central F distribution.

    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    f : scalar or ndarray
        Quantiles, i.e., the upper limit of integration.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtri

    Compute the CDF for several values of `f`:

    >>> f = [0.5, 1, 1.5]
    >>> p = ncfdtr(2, 3, 1.5, f)
    >>> p
    array([ 0.20782291,  0.36107392,  0.47345752])

    Compute the inverse.  We recover the values of `f`, as expected:

    >>> ncfdtri(2, 3, 1.5, p)
    array([ 0.5,  1. ,  1.5])

    """)

add_newdoc("ncfdtridfd",
    """
    ncfdtridfd(dfn, p, nc, f, out=None)

    Calculate degrees of freedom (denominator) for the noncentral F-distribution.

    This is the inverse with respect to `dfd` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    dfd : scalar or ndarray
        Degrees of freedom of the denominator sum of squares.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    The value of the cumulative noncentral F distribution is not necessarily
    monotone in either degrees of freedom. There thus may be two values that
    provide a given CDF value. This routine assumes monotonicity and will
    find an arbitrary one of the two values.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfd

    Compute the CDF for several values of `dfd`:

    >>> dfd = [1, 2, 3]
    >>> p = ncfdtr(2, dfd, 0.25, 15)
    >>> p
    array([ 0.8097138 ,  0.93020416,  0.96787852])

    Compute the inverse.  We recover the values of `dfd`, as expected:

    >>> ncfdtridfd(2, p, 0.25, 15)
    array([ 1.,  2.,  3.])

    """)

add_newdoc("ncfdtridfn",
    """
    ncfdtridfn(p, dfd, nc, f, out=None)

    Calculate degrees of freedom (numerator) for the noncentral F-distribution.

    This is the inverse with respect to `dfn` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    p : array_like
        Value of the cumulative distribution function. Must be in the
        range [0, 1].
    dfd : array_like
        Degrees of freedom of the denominator sum of squares. Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : float
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    dfn : scalar or ndarray
        Degrees of freedom of the numerator sum of squares.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    The value of the cumulative noncentral F distribution is not necessarily
    monotone in either degrees of freedom. There thus may be two values that
    provide a given CDF value. This routine assumes monotonicity and will
    find an arbitrary one of the two values.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfn

    Compute the CDF for several values of `dfn`:

    >>> dfn = [1, 2, 3]
    >>> p = ncfdtr(dfn, 2, 0.25, 15)
    >>> p
    array([ 0.92562363,  0.93020416,  0.93188394])

    Compute the inverse. We recover the values of `dfn`, as expected:

    >>> ncfdtridfn(p, 2, 0.25, 15)
    array([ 1.,  2.,  3.])

    """)

add_newdoc("ncfdtrinc",
    """
    ncfdtrinc(dfn, dfd, p, f, out=None)

    Calculate non-centrality parameter for non-central F distribution.

    This is the inverse with respect to `nc` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares. Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares. Range (0, inf).
    p : array_like
        Value of the cumulative distribution function. Must be in the
        range [0, 1].
    f : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    nc : scalar or ndarray
        Noncentrality parameter.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtrinc

    Compute the CDF for several values of `nc`:

    >>> nc = [0.5, 1.5, 2.0]
    >>> p = ncfdtr(2, 3, nc, 15)
    >>> p
    array([ 0.96309246,  0.94327955,  0.93304098])

    Compute the inverse. We recover the values of `nc`, as expected:

    >>> ncfdtrinc(2, 3, p, 15)
    array([ 0.5,  1.5,  2. ])

    """)

add_newdoc("nctdtr",
    """
    nctdtr(df, nc, t, out=None)

    Cumulative distribution function of the non-central `t` distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        The calculated CDF. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central t distribution, for nc=0. Compare with the
    t-distribution from scipy.stats:

    >>> x = np.linspace(-5, 5, num=500)
    >>> df = 3
    >>> nct_stats = stats.t.cdf(x, df)
    >>> nct_special = special.nctdtr(df, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, nct_stats, 'b-', lw=3)
    >>> ax.plot(x, nct_special, 'r-')
    >>> plt.show()

    """)

add_newdoc("nctdtridf",
    """
    nctdtridf(p, nc, t, out=None)

    Calculate degrees of freedom for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        The calculated CDF. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    """)

add_newdoc("nctdtrinc",
    """
    nctdtrinc(df, p, t, out=None)

    Calculate non-centrality parameter for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    p : array_like
        CDF values, in range (0, 1].
    t : array_like
        Quantiles, i.e., the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    nc : scalar or ndarray
        Noncentrality parameter

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.

    """)

add_newdoc("nctdtrit",
    """
    nctdtrit(df, nc, p, out=None)

    Inverse cumulative distribution function of the non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    p : array_like
        CDF values, in range (0, 1].
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    t : scalar or ndarray
        Quantiles

    See Also
    --------
    nctdtr :  CDF of the non-central `t` distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    """)

add_newdoc("ndtr",
    r"""
    ndtr(x, out=None)

    Cumulative distribution of the standard normal distribution.

    Returns the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`

    .. math::

       \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x \exp(-t^2/2) dt

    Parameters
    ----------
    x : array_like, real or complex
        Argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value of the normal CDF evaluated at `x`

    See Also
    --------
    log_ndtr : Logarithm of ndtr
    ndtri : Inverse of ndtr, standard normal percentile function
    erf : Error function
    erfc : 1 - erf
    scipy.stats.norm : Normal distribution

    Examples
    --------
    Evaluate `ndtr` at one point.

    >>> import numpy as np
    >>> from scipy.special import ndtr
    >>> ndtr(0.5)
    0.6914624612740131

    Evaluate the function at several points by providing a NumPy array
    or list for `x`.

    >>> ndtr([0, 0.5, 2])
    array([0.5       , 0.69146246, 0.97724987])

    Plot the function.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5, 5, 100)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, ndtr(x))
    >>> ax.set_title("Standard normal cumulative distribution function $\Phi$")
    >>> plt.show()
    """)


add_newdoc("nrdtrimn",
    """
    nrdtrimn(p, x, std, out=None)

    Calculate mean of normal distribution given other params.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    x : array_like
        Quantiles, i.e. the upper limit of integration.
    std : array_like
        Standard deviation.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    mn : scalar or ndarray
        The mean of the normal distribution.

    See Also
    --------
    nrdtrimn, ndtr

    """)

add_newdoc("nrdtrisd",
    """
    nrdtrisd(p, x, mn, out=None)

    Calculate standard deviation of normal distribution given other params.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    x : array_like
        Quantiles, i.e. the upper limit of integration.
    mn : scalar or ndarray
        The mean of the normal distribution.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    std : scalar or ndarray
        Standard deviation.

    See Also
    --------
    ndtr

    """)

add_newdoc("log_ndtr",
    """
    log_ndtr(x, out=None)

    Logarithm of Gaussian cumulative distribution function.

    Returns the log of the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`::

        log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))

    Parameters
    ----------
    x : array_like, real or complex
        Argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value of the log of the normal CDF evaluated at `x`

    See Also
    --------
    erf
    erfc
    scipy.stats.norm
    ndtr

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import log_ndtr, ndtr

    The benefit of ``log_ndtr(x)`` over the naive implementation
    ``np.log(ndtr(x))`` is most evident with moderate to large positive
    values of ``x``:

    >>> x = np.array([6, 7, 9, 12, 15, 25])
    >>> log_ndtr(x)
    array([-9.86587646e-010, -1.27981254e-012, -1.12858841e-019,
           -1.77648211e-033, -3.67096620e-051, -3.05669671e-138])

    The results of the naive calculation for the moderate ``x`` values
    have only 5 or 6 correct significant digits. For values of ``x``
    greater than approximately 8.3, the naive expression returns 0:

    >>> np.log(ndtr(x))
    array([-9.86587701e-10, -1.27986510e-12,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    """)

add_newdoc("ndtri",
    """
    ndtri(y, out=None)

    Inverse of `ndtr` vs x

    Returns the argument x for which the area under the standard normal
    probability density function (integrated from minus infinity to `x`)
    is equal to y.

    Parameters
    ----------
    p : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    x : scalar or ndarray
        Value of x such that ``ndtr(x) == p``.

    See Also
    --------
    ndtr : Standard normal cumulative probability distribution
    ndtri_exp : Inverse of log_ndtr

    Examples
    --------
    `ndtri` is the percentile function of the standard normal distribution.
    This means it returns the inverse of the cumulative density `ndtr`. First,
    let us compute a cumulative density value.

    >>> import numpy as np
    >>> from scipy.special import ndtri, ndtr
    >>> cdf_val = ndtr(2)
    >>> cdf_val
    0.9772498680518208

    Verify that `ndtri` yields the original value for `x` up to floating point
    errors.

    >>> ndtri(cdf_val)
    2.0000000000000004

    Plot the function. For that purpose, we provide a NumPy array as argument.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.01, 1, 200)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, ndtri(x))
    >>> ax.set_title("Standard normal percentile function")
    >>> plt.show()
    """)

add_newdoc("obl_ang1",
    """
    obl_ang1(m, n, c, x, out=None)

    Oblate spheroidal angular function of the first kind and its derivative

    Computes the oblate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_ang1_cv

    """)

add_newdoc("obl_ang1_cv",
    """
    obl_ang1_cv(m, n, c, cv, x, out=None)

    Oblate spheroidal angular function obl_ang1 for precomputed characteristic value

    Computes the oblate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_ang1

    """)

add_newdoc("obl_cv",
    """
    obl_cv(m, n, c, out=None)

    Characteristic value of oblate spheroidal function

    Computes the characteristic value of oblate spheroidal wave
    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cv : scalar or ndarray
        Characteristic value

    """)

add_newdoc("obl_rad1",
    """
    obl_rad1(m, n, c, x, out=None)

    Oblate spheroidal radial function of the first kind and its derivative

    Computes the oblate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_rad1_cv

    """)

add_newdoc("obl_rad1_cv",
    """
    obl_rad1_cv(m, n, c, cv, x, out=None)

    Oblate spheroidal radial function obl_rad1 for precomputed characteristic value

    Computes the oblate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_rad1

    """)

add_newdoc("obl_rad2",
    """
    obl_rad2(m, n, c, x, out=None)

    Oblate spheroidal radial function of the second kind and its derivative.

    Computes the oblate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_rad2_cv

    """)

add_newdoc("obl_rad2_cv",
    """
    obl_rad2_cv(m, n, c, cv, x, out=None)

    Oblate spheroidal radial function obl_rad2 for precomputed characteristic value

    Computes the oblate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Mode parameter m (nonnegative)
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Parameter x (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x

    See Also
    --------
    obl_rad2
    """)

add_newdoc("pbdv",
    """
    pbdv(v, x, out=None)

    Parabolic cylinder function D

    Returns (d, dp) the parabolic cylinder function Dv(x) in d and the
    derivative, Dv'(x) in dp.

    Parameters
    ----------
    v : array_like
        Real parameter
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    d : scalar or ndarray
        Value of the function
    dp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pbvv",
    """
    pbvv(v, x, out=None)

    Parabolic cylinder function V

    Returns the parabolic cylinder function Vv(x) in v and the
    derivative, Vv'(x) in vp.

    Parameters
    ----------
    v : array_like
        Real parameter
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    v : scalar or ndarray
        Value of the function
    vp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pbwa",
    r"""
    pbwa(a, x, out=None)

    Parabolic cylinder function W.

    The function is a particular solution to the differential equation

    .. math::

        y'' + \left(\frac{1}{4}x^2 - a\right)y = 0,

    for a full definition see section 12.14 in [1]_.

    Parameters
    ----------
    a : array_like
        Real parameter
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    w : scalar or ndarray
        Value of the function
    wp : scalar or ndarray
        Value of the derivative in x

    Notes
    -----
    The function is a wrapper for a Fortran routine by Zhang and Jin
    [2]_. The implementation is accurate only for ``|a|, |x| < 5`` and
    returns NaN outside that range.

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 14.30.
           https://dlmf.nist.gov/14.30
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    """)

add_newdoc("pdtr",
    r"""
    pdtr(k, m, out=None)

    Poisson cumulative distribution function.

    Defined as the probability that a Poisson-distributed random
    variable with event rate :math:`m` is less than or equal to
    :math:`k`. More concretely, this works out to be [1]_

    .. math::

       \exp(-m) \sum_{j = 0}^{\lfloor{k}\rfloor} \frac{m^j}{j!}.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    m : array_like
        Shape parameter (nonnegative, real)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Poisson cumulative distribution function

    See Also
    --------
    pdtrc : Poisson survival function
    pdtrik : inverse of `pdtr` with respect to `k`
    pdtri : inverse of `pdtr` with respect to `m`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is a cumulative distribution function, so it converges to 1
    monotonically as `k` goes to infinity.

    >>> sc.pdtr([1, 10, 100, np.inf], 1)
    array([0.73575888, 0.99999999, 1.        , 1.        ])

    It is discontinuous at integers and constant between integers.

    >>> sc.pdtr([1, 1.5, 1.9, 2], 1)
    array([0.73575888, 0.73575888, 0.73575888, 0.9196986 ])

    """)

add_newdoc("pdtrc",
    """
    pdtrc(k, m, out=None)

    Poisson survival function

    Returns the sum of the terms from k+1 to infinity of the Poisson
    distribution: sum(exp(-m) * m**j / j!, j=k+1..inf) = gammainc(
    k+1, m). Arguments must both be non-negative doubles.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    m : array_like
        Shape parameter (nonnegative, real)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Poisson survival function

    See Also
    --------
    pdtr : Poisson cumulative distribution function
    pdtrik : inverse of `pdtr` with respect to `k`
    pdtri : inverse of `pdtr` with respect to `m`

    """)

add_newdoc("pdtri",
    """
    pdtri(k, y, out=None)

    Inverse to `pdtr` vs m

    Returns the Poisson variable `m` such that the sum from 0 to `k` of
    the Poisson density is equal to the given probability `y`:
    calculated by ``gammaincinv(k + 1, y)``. `k` must be a nonnegative
    integer and `y` between 0 and 1.

    Parameters
    ----------
    k : array_like
        Number of occurrences (nonnegative, real)
    y : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the shape paramter `m` such that ``pdtr(k, m) = p``

    See Also
    --------
    pdtr : Poisson cumulative distribution function
    pdtrc : Poisson survival function
    pdtrik : inverse of `pdtr` with respect to `k`

    """)

add_newdoc("pdtrik",
    """
    pdtrik(p, m, out=None)

    Inverse to `pdtr` vs `m`.

    Parameters
    ----------
    m : array_like
        Shape parameter (nonnegative, real)
    p : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The number of occurrences `k` such that ``pdtr(k, m) = p``

    See Also
    --------
    pdtr : Poisson cumulative distribution function
    pdtrc : Poisson survival function
    pdtri : inverse of `pdtr` with respect to `m`

    """)

add_newdoc("poch",
    r"""
    poch(z, m, out=None)

    Pochhammer symbol.

    The Pochhammer symbol (rising factorial) is defined as

    .. math::

        (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

    For positive integer `m` it reads

    .. math::

        (z)_m = z (z + 1) ... (z + m - 1)

    See [dlmf]_ for more details.

    Parameters
    ----------
    z, m : array_like
        Real-valued arguments.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value of the function.

    References
    ----------
    .. [dlmf] Nist, Digital Library of Mathematical Functions
        https://dlmf.nist.gov/5.2#iii

    Examples
    --------
    >>> import scipy.special as sc

    It is 1 when m is 0.

    >>> sc.poch([1, 2, 3, 4], 0)
    array([1., 1., 1., 1.])

    For z equal to 1 it reduces to the factorial function.

    >>> sc.poch(1, 5)
    120.0
    >>> 1 * 2 * 3 * 4 * 5
    120

    It can be expressed in terms of the gamma function.

    >>> z, m = 3.7, 2.1
    >>> sc.poch(z, m)
    20.529581933776953
    >>> sc.gamma(z + m) / sc.gamma(z)
    20.52958193377696

    """)

add_newdoc("powm1", """
    powm1(x, y, out=None)

    Computes ``x**y - 1``.

    This function is useful when `y` is near 0, or when `x` is near 1.

    The function is implemented for real types only (unlike ``numpy.power``,
    which accepts complex inputs).

    Parameters
    ----------
    x : array_like
        The base. Must be a real type (i.e. integer or float, not complex).
    y : array_like
        The exponent. Must be a real type (i.e. integer or float, not complex).

    Returns
    -------
    array_like
        Result of the calculation

    Notes
    -----
    .. versionadded:: 1.10.0

    The underlying code is implemented for single precision and double
    precision floats only.  Unlike `numpy.power`, integer inputs to
    `powm1` are converted to floating point, and complex inputs are
    not accepted.

    Note the following edge cases:

    * ``powm1(x, 0)`` returns 0 for any ``x``, including 0, ``inf``
      and ``nan``.
    * ``powm1(1, y)`` returns 0 for any ``y``, including ``nan``
      and ``inf``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import powm1

    >>> x = np.array([1.2, 10.0, 0.9999999975])
    >>> y = np.array([1e-9, 1e-11, 0.1875])
    >>> powm1(x, y)
    array([ 1.82321557e-10,  2.30258509e-11, -4.68749998e-10])

    It can be verified that the relative errors in those results
    are less than 2.5e-16.

    Compare that to the result of ``x**y - 1``, where the
    relative errors are all larger than 8e-8:

    >>> x**y - 1
    array([ 1.82321491e-10,  2.30258035e-11, -4.68750039e-10])

    """)


add_newdoc("pro_ang1",
    """
    pro_ang1(m, n, c, x, out=None)

    Prolate spheroidal angular function of the first kind and its derivative

    Computes the prolate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pro_ang1_cv",
    """
    pro_ang1_cv(m, n, c, cv, x, out=None)

    Prolate spheroidal angular function pro_ang1 for precomputed characteristic value

    Computes the prolate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pro_cv",
    """
    pro_cv(m, n, c, out=None)

    Characteristic value of prolate spheroidal function

    Computes the characteristic value of prolate spheroidal wave
    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cv : scalar or ndarray
        Characteristic value
    """)

add_newdoc("pro_rad1",
    """
    pro_rad1(m, n, c, x, out=None)

    Prolate spheroidal radial function of the first kind and its derivative

    Computes the prolate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pro_rad1_cv",
    """
    pro_rad1_cv(m, n, c, cv, x, out=None)

    Prolate spheroidal radial function pro_rad1 for precomputed characteristic value

    Computes the prolate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pro_rad2",
    """
    pro_rad2(m, n, c, x, out=None)

    Prolate spheroidal radial function of the second kind and its derivative

    Computes the prolate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pro_rad2_cv",
    """
    pro_rad2_cv(m, n, c, cv, x, out=None)

    Prolate spheroidal radial function pro_rad2 for precomputed characteristic value

    Computes the prolate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Parameters
    ----------
    m : array_like
        Nonnegative mode parameter m
    n : array_like
        Mode parameter n (>= m)
    c : array_like
        Spheroidal parameter
    cv : array_like
        Characteristic value
    x : array_like
        Real parameter (``|x| < 1.0``)
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Value of the function
    sp : scalar or ndarray
        Value of the derivative vs x
    """)

add_newdoc("pseudo_huber",
    r"""
    pseudo_huber(delta, r, out=None)

    Pseudo-Huber loss function.

    .. math:: \mathrm{pseudo\_huber}(\delta, r) = \delta^2 \left( \sqrt{ 1 + \left( \frac{r}{\delta} \right)^2 } - 1 \right)

    Parameters
    ----------
    delta : array_like
        Input array, indicating the soft quadratic vs. linear loss changepoint.
    r : array_like
        Input array, possibly representing residuals.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    res : scalar or ndarray
        The computed Pseudo-Huber loss function values.

    See also
    --------
    huber: Similar function which this function approximates

    Notes
    -----
    Like `huber`, `pseudo_huber` often serves as a robust loss function
    in statistics or machine learning to reduce the influence of outliers.
    Unlike `huber`, `pseudo_huber` is smooth.

    Typically, `r` represents residuals, the difference
    between a model prediction and data. Then, for :math:`|r|\leq\delta`,
    `pseudo_huber` resembles the squared error and for :math:`|r|>\delta` the
    absolute error. This way, the Pseudo-Huber loss often achieves
    a fast convergence in model fitting for small residuals like the squared
    error loss function and still reduces the influence of outliers
    (:math:`|r|>\delta`) like the absolute error loss. As :math:`\delta` is
    the cutoff between squared and absolute error regimes, it has
    to be tuned carefully for each problem. `pseudo_huber` is also
    convex, making it suitable for gradient based optimization. [1]_ [2]_

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Hartley, Zisserman, "Multiple View Geometry in Computer Vision".
           2003. Cambridge University Press. p. 619
    .. [2] Charbonnier et al. "Deterministic edge-preserving regularization
           in computed imaging". 1997. IEEE Trans. Image Processing.
           6 (2): 298 - 311.

    Examples
    --------
    Import all necessary modules.

    >>> import numpy as np
    >>> from scipy.special import pseudo_huber, huber
    >>> import matplotlib.pyplot as plt

    Calculate the function for ``delta=1`` at ``r=2``.

    >>> pseudo_huber(1., 2.)
    1.2360679774997898

    Calculate the function at ``r=2`` for different `delta` by providing
    a list or NumPy array for `delta`.

    >>> pseudo_huber([1., 2., 4.], 3.)
    array([2.16227766, 3.21110255, 4.        ])

    Calculate the function for ``delta=1`` at several points by providing
    a list or NumPy array for `r`.

    >>> pseudo_huber(2., np.array([1., 1.5, 3., 4.]))
    array([0.47213595, 1.        , 3.21110255, 4.94427191])

    The function can be calculated for different `delta` and `r` by
    providing arrays for both with compatible shapes for broadcasting.

    >>> r = np.array([1., 2.5, 8., 10.])
    >>> deltas = np.array([[1.], [5.], [9.]])
    >>> print(r.shape, deltas.shape)
    (4,) (3, 1)

    >>> pseudo_huber(deltas, r)
    array([[ 0.41421356,  1.6925824 ,  7.06225775,  9.04987562],
           [ 0.49509757,  2.95084972, 22.16990566, 30.90169944],
           [ 0.49846624,  3.06693762, 27.37435121, 40.08261642]])

    Plot the function for different `delta`.

    >>> x = np.linspace(-4, 4, 500)
    >>> deltas = [1, 2, 3]
    >>> linestyles = ["dashed", "dotted", "dashdot"]
    >>> fig, ax = plt.subplots()
    >>> combined_plot_parameters = list(zip(deltas, linestyles))
    >>> for delta, style in combined_plot_parameters:
    ...     ax.plot(x, pseudo_huber(delta, x), label=f"$\delta={delta}$",
    ...             ls=style)
    >>> ax.legend(loc="upper center")
    >>> ax.set_xlabel("$x$")
    >>> ax.set_title("Pseudo-Huber loss function $h_{\delta}(x)$")
    >>> ax.set_xlim(-4, 4)
    >>> ax.set_ylim(0, 8)
    >>> plt.show()

    Finally, illustrate the difference between `huber` and `pseudo_huber` by
    plotting them and their gradients with respect to `r`. The plot shows
    that `pseudo_huber` is continuously differentiable while `huber` is not
    at the points :math:`\pm\delta`.

    >>> def huber_grad(delta, x):
    ...     grad = np.copy(x)
    ...     linear_area = np.argwhere(np.abs(x) > delta)
    ...     grad[linear_area]=delta*np.sign(x[linear_area])
    ...     return grad
    >>> def pseudo_huber_grad(delta, x):
    ...     return x* (1+(x/delta)**2)**(-0.5)
    >>> x=np.linspace(-3, 3, 500)
    >>> delta = 1.
    >>> fig, ax = plt.subplots(figsize=(7, 7))
    >>> ax.plot(x, huber(delta, x), label="Huber", ls="dashed")
    >>> ax.plot(x, huber_grad(delta, x), label="Huber Gradient", ls="dashdot")
    >>> ax.plot(x, pseudo_huber(delta, x), label="Pseudo-Huber", ls="dotted")
    >>> ax.plot(x, pseudo_huber_grad(delta, x), label="Pseudo-Huber Gradient",
    ...         ls="solid")
    >>> ax.legend(loc="upper center")
    >>> plt.show()
    """)

add_newdoc("psi",
    """
    psi(z, out=None)

    The digamma function.

    The logarithmic derivative of the gamma function evaluated at ``z``.

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : ndarray, optional
        Array for the computed values of ``psi``.

    Returns
    -------
    digamma : scalar or ndarray
        Computed values of ``psi``.

    Notes
    -----
    For large values not close to the negative real axis, ``psi`` is
    computed using the asymptotic series (5.11.2) from [1]_. For small
    arguments not close to the negative real axis, the recurrence
    relation (5.5.2) from [1]_ is used until the argument is large
    enough to use the asymptotic series. For values close to the
    negative real axis, the reflection formula (5.5.4) from [1]_ is
    used first. Note that ``psi`` has a family of zeros on the
    negative real axis which occur between the poles at nonpositive
    integers. Around the zeros the reflection formula suffers from
    cancellation and the implementation loses precision. The sole
    positive zero and the first negative zero, however, are handled
    separately by precomputing series expansions using [2]_, so the
    function should maintain full accuracy around the origin.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/5
    .. [2] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
           (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> from scipy.special import psi
    >>> z = 3 + 4j
    >>> psi(z)
    (1.55035981733341+1.0105022091860445j)

    Verify psi(z) = psi(z + 1) - 1/z:

    >>> psi(z + 1) - 1/z
    (1.55035981733341+1.0105022091860445j)
    """)

add_newdoc("radian",
    """
    radian(d, m, s, out=None)

    Convert from degrees to radians.

    Returns the angle given in (d)egrees, (m)inutes, and (s)econds in
    radians.

    Parameters
    ----------
    d : array_like
        Degrees, can be real-valued.
    m : array_like
        Minutes, can be real-valued.
    s : array_like
        Seconds, can be real-valued.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the inputs in radians.

    Examples
    --------
    >>> import scipy.special as sc

    There are many ways to specify an angle.

    >>> sc.radian(90, 0, 0)
    1.5707963267948966
    >>> sc.radian(0, 60 * 90, 0)
    1.5707963267948966
    >>> sc.radian(0, 0, 60**2 * 90)
    1.5707963267948966

    The inputs can be real-valued.

    >>> sc.radian(1.5, 0, 0)
    0.02617993877991494
    >>> sc.radian(1, 30, 0)
    0.02617993877991494

    """)

add_newdoc("rel_entr",
    r"""
    rel_entr(x, y, out=None)

    Elementwise function for computing relative entropy.

    .. math::

        \mathrm{rel\_entr}(x, y) =
            \begin{cases}
                x \log(x / y) & x > 0, y > 0 \\
                0 & x = 0, y \ge 0 \\
                \infty & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    x, y : array_like
        Input arrays
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Relative entropy of the inputs

    See Also
    --------
    entr, kl_div, scipy.stats.entropy

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is jointly convex in x and y.

    The origin of this function is in convex programming; see
    [1]_. Given two discrete probability distributions :math:`p_1,
    \ldots, p_n` and :math:`q_1, \ldots, q_n`, the definition of relative
    entropy in the context of *information theory* is

    .. math::

        \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

    To compute the latter quantity, use `scipy.stats.entropy`.

    See [2]_ for details.

    References
    ----------
    .. [1] Boyd, Stephen and Lieven Vandenberghe. *Convex optimization*.
           Cambridge University Press, 2004.
           :doi:`https://doi.org/10.1017/CBO9780511804441`
    .. [2] Kullback-Leibler divergence,
           https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    """)

add_newdoc("rgamma",
    r"""
    rgamma(z, out=None)

    Reciprocal of the gamma function.

    Defined as :math:`1 / \Gamma(z)`, where :math:`\Gamma` is the
    gamma function. For more on the gamma function see `gamma`.

    Parameters
    ----------
    z : array_like
        Real or complex valued input
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Function results

    Notes
    -----
    The gamma function has no zeros and has simple poles at
    nonpositive integers, so `rgamma` is an entire function with zeros
    at the nonpositive integers. See the discussion in [dlmf]_ for
    more details.

    See Also
    --------
    gamma, gammaln, loggamma

    References
    ----------
    .. [dlmf] Nist, Digital Library of Mathematical functions,
        https://dlmf.nist.gov/5.2#i

    Examples
    --------
    >>> import scipy.special as sc

    It is the reciprocal of the gamma function.

    >>> sc.rgamma([1, 2, 3, 4])
    array([1.        , 1.        , 0.5       , 0.16666667])
    >>> 1 / sc.gamma([1, 2, 3, 4])
    array([1.        , 1.        , 0.5       , 0.16666667])

    It is zero at nonpositive integers.

    >>> sc.rgamma([0, -1, -2, -3])
    array([0., 0., 0., 0.])

    It rapidly underflows to zero along the positive real axis.

    >>> sc.rgamma([10, 100, 179])
    array([2.75573192e-006, 1.07151029e-156, 0.00000000e+000])

    """)

add_newdoc("round",
    """
    round(x, out=None)

    Round to the nearest integer.

    Returns the nearest integer to `x`.  If `x` ends in 0.5 exactly,
    the nearest even integer is chosen.

    Parameters
    ----------
    x : array_like
        Real valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The nearest integers to the elements of `x`. The result is of
        floating type, not integer type.

    Examples
    --------
    >>> import scipy.special as sc

    It rounds to even.

    >>> sc.round([0.5, 1.5])
    array([0., 2.])

    """)

add_newdoc("shichi",
    r"""
    shichi(x, out=None)

    Hyperbolic sine and cosine integrals.

    The hyperbolic sine integral is

    .. math::

      \int_0^x \frac{\sinh{t}}{t}dt

    and the hyperbolic cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cosh{t} - 1}{t} dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principal branch of the logarithm [1]_.

    Parameters
    ----------
    x : array_like
        Real or complex points at which to compute the hyperbolic sine
        and cosine integrals.
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    si : scalar or ndarray
        Hyperbolic sine integral at ``x``
    ci : scalar or ndarray
        Hyperbolic cosine integral at ``x``

    See Also
    --------
    sici : Sine and cosine integrals.
    exp1 : Exponential integral E1.
    expi : Exponential integral Ei.

    Notes
    -----
    For real arguments with ``x < 0``, ``chi`` is the real part of the
    hyperbolic cosine integral. For such points ``chi(x)`` and ``chi(x
    + 0j)`` differ by a factor of ``1j*pi``.

    For real arguments the function is computed by calling Cephes'
    [2]_ *shichi* routine. For complex arguments the algorithm is based
    on Mpmath's [3]_ *shi* and *chi* routines.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
           (See Section 5.2.)
    .. [2] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [3] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point
           arithmetic" (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import shichi, sici

    `shichi` accepts real or complex input:

    >>> shichi(0.5)
    (0.5069967498196671, -0.05277684495649357)
    >>> shichi(0.5 + 2.5j)
    ((0.11772029666668238+1.831091777729851j),
     (0.29912435887648825+1.7395351121166562j))

    The hyperbolic sine and cosine integrals Shi(z) and Chi(z) are
    related to the sine and cosine integrals Si(z) and Ci(z) by

    * Shi(z) = -i*Si(i*z)
    * Chi(z) = Ci(-i*z) + i*pi/2

    >>> z = 0.25 + 5j
    >>> shi, chi = shichi(z)
    >>> shi, -1j*sici(1j*z)[0]            # Should be the same.
    ((-0.04834719325101729+1.5469354086921228j),
     (-0.04834719325101729+1.5469354086921228j))
    >>> chi, sici(-1j*z)[1] + 1j*np.pi/2  # Should be the same.
    ((-0.19568708973868087+1.556276312103824j),
     (-0.19568708973868087+1.556276312103824j))

    Plot the functions evaluated on the real axis:

    >>> xp = np.geomspace(1e-8, 4.0, 250)
    >>> x = np.concatenate((-xp[::-1], xp))
    >>> shi, chi = shichi(x)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, shi, label='Shi(x)')
    >>> ax.plot(x, chi, '--', label='Chi(x)')
    >>> ax.set_xlabel('x')
    >>> ax.set_title('Hyperbolic Sine and Cosine Integrals')
    >>> ax.legend(shadow=True, framealpha=1, loc='lower right')
    >>> ax.grid(True)
    >>> plt.show()

    """)

add_newdoc("sici",
    r"""
    sici(x, out=None)

    Sine and cosine integrals.

    The sine integral is

    .. math::

      \int_0^x \frac{\sin{t}}{t}dt

    and the cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cos{t} - 1}{t}dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principal branch of the logarithm [1]_.

    Parameters
    ----------
    x : array_like
        Real or complex points at which to compute the sine and cosine
        integrals.
    out : tuple of ndarray, optional
        Optional output arrays for the function results

    Returns
    -------
    si : scalar or ndarray
        Sine integral at ``x``
    ci : scalar or ndarray
        Cosine integral at ``x``

    See Also
    --------
    shichi : Hyperbolic sine and cosine integrals.
    exp1 : Exponential integral E1.
    expi : Exponential integral Ei.

    Notes
    -----
    For real arguments with ``x < 0``, ``ci`` is the real part of the
    cosine integral. For such points ``ci(x)`` and ``ci(x + 0j)``
    differ by a factor of ``1j*pi``.

    For real arguments the function is computed by calling Cephes'
    [2]_ *sici* routine. For complex arguments the algorithm is based
    on Mpmath's [3]_ *si* and *ci* routines.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
           (See Section 5.2.)
    .. [2] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [3] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point
           arithmetic" (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import sici, exp1

    `sici` accepts real or complex input:

    >>> sici(2.5)
    (1.7785201734438267, 0.2858711963653835)
    >>> sici(2.5 + 3j)
    ((4.505735874563953+0.06863305018999577j),
    (0.0793644206906966-2.935510262937543j))

    For z in the right half plane, the sine and cosine integrals are
    related to the exponential integral E1 (implemented in SciPy as
    `scipy.special.exp1`) by

    * Si(z) = (E1(i*z) - E1(-i*z))/2i + pi/2
    * Ci(z) = -(E1(i*z) + E1(-i*z))/2

    See [1]_ (equations 5.2.21 and 5.2.23).

    We can verify these relations:

    >>> z = 2 - 3j
    >>> sici(z)
    ((4.54751388956229-1.3991965806460565j),
    (1.408292501520851+2.9836177420296055j))

    >>> (exp1(1j*z) - exp1(-1j*z))/2j + np.pi/2  # Same as sine integral
    (4.54751388956229-1.3991965806460565j)

    >>> -(exp1(1j*z) + exp1(-1j*z))/2            # Same as cosine integral
    (1.408292501520851+2.9836177420296055j)

    Plot the functions evaluated on the real axis; the dotted horizontal
    lines are at pi/2 and -pi/2:

    >>> x = np.linspace(-16, 16, 150)
    >>> si, ci = sici(x)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, si, label='Si(x)')
    >>> ax.plot(x, ci, '--', label='Ci(x)')
    >>> ax.legend(shadow=True, framealpha=1, loc='upper left')
    >>> ax.set_xlabel('x')
    >>> ax.set_title('Sine and Cosine Integrals')
    >>> ax.axhline(np.pi/2, linestyle=':', alpha=0.5, color='k')
    >>> ax.axhline(-np.pi/2, linestyle=':', alpha=0.5, color='k')
    >>> ax.grid(True)
    >>> plt.show()

    """)

add_newdoc("sindg",
    """
    sindg(x, out=None)

    Sine of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Sine at the input.

    See Also
    --------
    cosdg, tandg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using sine directly.

    >>> x = 180 * np.arange(3)
    >>> sc.sindg(x)
    array([ 0., -0.,  0.])
    >>> np.sin(x * np.pi / 180)
    array([ 0.0000000e+00,  1.2246468e-16, -2.4492936e-16])

    """)

add_newdoc("smirnov",
    r"""
    smirnov(n, d, out=None)

    Kolmogorov-Smirnov complementary cumulative distribution function

    Returns the exact Kolmogorov-Smirnov complementary cumulative
    distribution function,(aka the Survival Function) of Dn+ (or Dn-)
    for a one-sided test of equality between an empirical and a
    theoretical distribution. It is equal to the probability that the
    maximum difference between a theoretical distribution and an empirical
    one based on `n` samples is greater than d.

    Parameters
    ----------
    n : int
      Number of samples
    d : float array_like
      Deviation between the Empirical CDF (ECDF) and the target CDF.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of smirnov(n, d), Prob(Dn+ >= d) (Also Prob(Dn- >= d))

    See Also
    --------
    smirnovi : The Inverse Survival Function for the distribution
    scipy.stats.ksone : Provides the functionality as a continuous distribution
    kolmogorov, kolmogi : Functions for the two-sided distribution

    Notes
    -----
    `smirnov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.ksone` distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import smirnov
    >>> from scipy.stats import norm

    Show the probability of a gap at least as big as 0, 0.5 and 1.0 for a
    sample of size 5.

    >>> smirnov(5, [0, 0.5, 1.0])
    array([ 1.   ,  0.056,  0.   ])

    Compare a sample of size 5 against N(0, 1), the standard normal
    distribution with mean 0 and standard deviation 1.

    `x` is the sample.

    >>> x = np.array([-1.392, -0.135, 0.114, 0.190, 1.82])

    >>> target = norm(0, 1)
    >>> cdfs = target.cdf(x)
    >>> cdfs
    array([0.0819612 , 0.44630594, 0.5453811 , 0.57534543, 0.9656205 ])

    Construct the empirical CDF and the K-S statistics (Dn+, Dn-, Dn).

    >>> n = len(x)
    >>> ecdfs = np.arange(n+1, dtype=float)/n
    >>> cols = np.column_stack([x, ecdfs[1:], cdfs, cdfs - ecdfs[:n],
    ...                        ecdfs[1:] - cdfs])
    >>> with np.printoptions(precision=3):
    ...    print(cols)
    [[-1.392  0.2    0.082  0.082  0.118]
     [-0.135  0.4    0.446  0.246 -0.046]
     [ 0.114  0.6    0.545  0.145  0.055]
     [ 0.19   0.8    0.575 -0.025  0.225]
     [ 1.82   1.     0.966  0.166  0.034]]
    >>> gaps = cols[:, -2:]
    >>> Dnpm = np.max(gaps, axis=0)
    >>> print(f'Dn-={Dnpm[0]:f}, Dn+={Dnpm[1]:f}')
    Dn-=0.246306, Dn+=0.224655
    >>> probs = smirnov(n, Dnpm)
    >>> print(f'For a sample of size {n} drawn from N(0, 1):',
    ...       f' Smirnov n={n}: Prob(Dn- >= {Dnpm[0]:f}) = {probs[0]:.4f}',
    ...       f' Smirnov n={n}: Prob(Dn+ >= {Dnpm[1]:f}) = {probs[1]:.4f}',
    ...       sep='\n')
    For a sample of size 5 drawn from N(0, 1):
     Smirnov n=5: Prob(Dn- >= 0.246306) = 0.4711
     Smirnov n=5: Prob(Dn+ >= 0.224655) = 0.5245

    Plot the empirical CDF and the standard normal CDF.

    >>> import matplotlib.pyplot as plt
    >>> plt.step(np.concatenate(([-2.5], x, [2.5])),
    ...          np.concatenate((ecdfs, [1])),
    ...          where='post', label='Empirical CDF')
    >>> xx = np.linspace(-2.5, 2.5, 100)
    >>> plt.plot(xx, target.cdf(xx), '--', label='CDF for N(0, 1)')

    Add vertical lines marking Dn+ and Dn-.

    >>> iminus, iplus = np.argmax(gaps, axis=0)
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color='r',
    ...            alpha=0.5, lw=4)
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1], color='m',
    ...            alpha=0.5, lw=4)

    >>> plt.grid(True)
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.show()
    """)

add_newdoc("smirnovi",
    """
    smirnovi(n, p, out=None)

    Inverse to `smirnov`

    Returns `d` such that ``smirnov(n, d) == p``, the critical value
    corresponding to `p`.

    Parameters
    ----------
    n : int
      Number of samples
    p : float array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The value(s) of smirnovi(n, p), the critical values.

    See Also
    --------
    smirnov : The Survival Function (SF) for the distribution
    scipy.stats.ksone : Provides the functionality as a continuous distribution
    kolmogorov, kolmogi : Functions for the two-sided distribution
    scipy.stats.kstwobign : Two-sided Kolmogorov-Smirnov distribution, large n

    Notes
    -----
    `smirnov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.ksone` distribution.

    Examples
    --------
    >>> from scipy.special import smirnovi, smirnov

    >>> n = 24
    >>> deviations = [0.1, 0.2, 0.3]

    Use `smirnov` to compute the complementary CDF of the Smirnov
    distribution for the given number of samples and deviations.

    >>> p = smirnov(n, deviations)
    >>> p
    array([0.58105083, 0.12826832, 0.01032231])

    The inverse function ``smirnovi(n, p)`` returns ``deviations``.

    >>> smirnovi(n, p)
    array([0.1, 0.2, 0.3])

    """)

add_newdoc("_smirnovc",
    """
    _smirnovc(n, d)
     Internal function, do not use.
    """)

add_newdoc("_smirnovci",
    """
     Internal function, do not use.
    """)

add_newdoc("_smirnovp",
    """
    _smirnovp(n, p)
     Internal function, do not use.
    """)

add_newdoc("spence",
    r"""
    spence(z, out=None)

    Spence's function, also known as the dilogarithm.

    It is defined to be

    .. math::
      \int_1^z \frac{\log(t)}{1 - t}dt

    for complex :math:`z`, where the contour of integration is taken
    to avoid the branch cut of the logarithm. Spence's function is
    analytic everywhere except the negative real axis where it has a
    branch cut.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate Spence's function
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    s : scalar or ndarray
        Computed values of Spence's function

    Notes
    -----
    There is a different convention which defines Spence's function by
    the integral

    .. math::
      -\int_0^z \frac{\log(1 - t)}{t}dt;

    this is our ``spence(1 - z)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import spence
    >>> import matplotlib.pyplot as plt

    The function is defined for complex inputs:

    >>> spence([1-1j, 1.5+2j, 3j, -10-5j])
    array([-0.20561676+0.91596559j, -0.86766909-1.39560134j,
           -0.59422064-2.49129918j, -1.14044398+6.80075924j])

    For complex inputs on the branch cut, which is the negative real axis,
    the function returns the limit for ``z`` with positive imaginary part.
    For example, in the following, note the sign change of the imaginary
    part of the output for ``z = -2`` and ``z = -2 - 1e-8j``:

    >>> spence([-2 + 1e-8j, -2, -2 - 1e-8j])
    array([2.32018041-3.45139229j, 2.32018042-3.4513923j ,
           2.32018041+3.45139229j])

    The function returns ``nan`` for real inputs on the branch cut:

    >>> spence(-1.5)
    nan

    Verify some particular values: ``spence(0) = pi**2/6``,
    ``spence(1) = 0`` and ``spence(2) = -pi**2/12``.

    >>> spence([0, 1, 2])
    array([ 1.64493407,  0.        , -0.82246703])
    >>> np.pi**2/6, -np.pi**2/12
    (1.6449340668482264, -0.8224670334241132)

    Verify the identity::

        spence(z) + spence(1 - z) = pi**2/6 - log(z)*log(1 - z)

    >>> z = 3 + 4j
    >>> spence(z) + spence(1 - z)
    (-2.6523186143876067+1.8853470951513935j)
    >>> np.pi**2/6 - np.log(z)*np.log(1 - z)
    (-2.652318614387606+1.885347095151394j)

    Plot the function for positive real input.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 6, 400)
    >>> ax.plot(x, spence(x))
    >>> ax.grid()
    >>> ax.set_xlabel('x')
    >>> ax.set_title('spence(x)')
    >>> plt.show()
    """)

add_newdoc(
    "stdtr",
    r"""
    stdtr(df, t, out=None)

    Student t distribution cumulative distribution function

    Returns the integral:

    .. math::
        \frac{\Gamma((df+1)/2)}{\sqrt{\pi df} \Gamma(df/2)}
        \int_{-\infty}^t (1+x^2/df)^{-(df+1)/2}\, dx

    Parameters
    ----------
    df : array_like
        Degrees of freedom
    t : array_like
        Upper bound of the integral
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Value of the Student t CDF at t

    See Also
    --------
    stdtridf : inverse of stdtr with respect to `df`
    stdtrit : inverse of stdtr with respect to `t`
    scipy.stats.t : student t distribution

    Notes
    -----
    The student t distribution is also available as `scipy.stats.t`.
    Calling `stdtr` directly can improve performance compared to the
    ``cdf`` method of `scipy.stats.t` (see last example below).

    Examples
    --------
    Calculate the function for ``df=3`` at ``t=1``.

    >>> import numpy as np
    >>> from scipy.special import stdtr
    >>> import matplotlib.pyplot as plt
    >>> stdtr(3, 1)
    0.8044988905221148

    Plot the function for three different degrees of freedom.

    >>> x = np.linspace(-10, 10, 1000)
    >>> fig, ax = plt.subplots()
    >>> parameters = [(1, "solid"), (3, "dashed"), (10, "dotted")]
    >>> for (df, linestyle) in parameters:
    ...     ax.plot(x, stdtr(df, x), ls=linestyle, label=f"$df={df}$")
    >>> ax.legend()
    >>> ax.set_title("Student t distribution cumulative distribution function")
    >>> plt.show()

    The function can be computed for several degrees of freedom at the same
    time by providing a NumPy array or list for `df`:

    >>> stdtr([1, 2, 3], 1)
    array([0.75      , 0.78867513, 0.80449889])

    It is possible to calculate the function at several points for several
    different degrees of freedom simultaneously by providing arrays for `df`
    and `t` with shapes compatible for broadcasting. Compute `stdtr` at
    4 points for 3 degrees of freedom resulting in an array of shape 3x4.

    >>> dfs = np.array([[1], [2], [3]])
    >>> t = np.array([2, 4, 6, 8])
    >>> dfs.shape, t.shape
    ((3, 1), (4,))

    >>> stdtr(dfs, t)
    array([[0.85241638, 0.92202087, 0.94743154, 0.96041658],
           [0.90824829, 0.97140452, 0.98666426, 0.99236596],
           [0.93033702, 0.98599577, 0.99536364, 0.99796171]])

    The t distribution is also available as `scipy.stats.t`. Calling `stdtr`
    directly can be much faster than calling the ``cdf`` method of
    `scipy.stats.t`. To get the same results, one must use the following
    parametrization: ``scipy.stats.t(df).cdf(x) = stdtr(df, x)``.

    >>> from scipy.stats import t
    >>> df, x = 3, 1
    >>> stdtr_result = stdtr(df, x)  # this can be faster than below
    >>> stats_result = t(df).cdf(x)
    >>> stats_result == stdtr_result  # test that results are equal
    True
    """)

add_newdoc("stdtridf",
    """
    stdtridf(p, t, out=None)

    Inverse of `stdtr` vs df

    Returns the argument df such that stdtr(df, t) is equal to `p`.

    Parameters
    ----------
    p : array_like
        Probability
    t : array_like
        Upper bound of the integral
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    df : scalar or ndarray
        Value of `df` such that ``stdtr(df, t) == p``

    See Also
    --------
    stdtr : Student t CDF
    stdtrit : inverse of stdtr with respect to `t`
    scipy.stats.t : Student t distribution

    Examples
    --------
    Compute the student t cumulative distribution function for one
    parameter set.

    >>> from scipy.special import stdtr, stdtridf
    >>> df, x = 5, 2
    >>> cdf_value = stdtr(df, x)
    >>> cdf_value
    0.9490302605850709

    Verify that `stdtridf` recovers the original value for `df` given
    the CDF value and `x`.

    >>> stdtridf(cdf_value, x)
    5.0
    """)

add_newdoc("stdtrit",
    """
    stdtrit(df, p, out=None)

    The `p`-th quantile of the student t distribution.

    This function is the inverse of the student t distribution cumulative
    distribution function (CDF), returning `t` such that `stdtr(df, t) = p`.

    Returns the argument `t` such that stdtr(df, t) is equal to `p`.

    Parameters
    ----------
    df : array_like
        Degrees of freedom
    p : array_like
        Probability
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    t : scalar or ndarray
        Value of `t` such that ``stdtr(df, t) == p``

    See Also
    --------
    stdtr : Student t CDF
    stdtridf : inverse of stdtr with respect to `df`
    scipy.stats.t : Student t distribution

    Notes
    -----
    The student t distribution is also available as `scipy.stats.t`. Calling
    `stdtrit` directly can improve performance compared to the ``ppf``
    method of `scipy.stats.t` (see last example below).

    Examples
    --------
    `stdtrit` represents the inverse of the student t distribution CDF which
    is available as `stdtr`. Here, we calculate the CDF for ``df`` at
    ``x=1``. `stdtrit` then returns ``1`` up to floating point errors
    given the same value for `df` and the computed CDF value.

    >>> import numpy as np
    >>> from scipy.special import stdtr, stdtrit
    >>> import matplotlib.pyplot as plt
    >>> df = 3
    >>> x = 1
    >>> cdf_value = stdtr(df, x)
    >>> stdtrit(df, cdf_value)
    0.9999999994418539

    Plot the function for three different degrees of freedom.

    >>> x = np.linspace(0, 1, 1000)
    >>> parameters = [(1, "solid"), (2, "dashed"), (5, "dotted")]
    >>> fig, ax = plt.subplots()
    >>> for (df, linestyle) in parameters:
    ...     ax.plot(x, stdtrit(df, x), ls=linestyle, label=f"$df={df}$")
    >>> ax.legend()
    >>> ax.set_ylim(-10, 10)
    >>> ax.set_title("Student t distribution quantile function")
    >>> plt.show()

    The function can be computed for several degrees of freedom at the same
    time by providing a NumPy array or list for `df`:

    >>> stdtrit([1, 2, 3], 0.7)
    array([0.72654253, 0.6172134 , 0.58438973])

    It is possible to calculate the function at several points for several
    different degrees of freedom simultaneously by providing arrays for `df`
    and `p` with shapes compatible for broadcasting. Compute `stdtrit` at
    4 points for 3 degrees of freedom resulting in an array of shape 3x4.

    >>> dfs = np.array([[1], [2], [3]])
    >>> p = np.array([0.2, 0.4, 0.7, 0.8])
    >>> dfs.shape, p.shape
    ((3, 1), (4,))

    >>> stdtrit(dfs, p)
    array([[-1.37638192, -0.3249197 ,  0.72654253,  1.37638192],
           [-1.06066017, -0.28867513,  0.6172134 ,  1.06066017],
           [-0.97847231, -0.27667066,  0.58438973,  0.97847231]])

    The t distribution is also available as `scipy.stats.t`. Calling `stdtrit`
    directly can be much faster than calling the ``ppf`` method of
    `scipy.stats.t`. To get the same results, one must use the following
    parametrization: ``scipy.stats.t(df).ppf(x) = stdtrit(df, x)``.

    >>> from scipy.stats import t
    >>> df, x = 3, 0.5
    >>> stdtrit_result = stdtrit(df, x)  # this can be faster than below
    >>> stats_result = t(df).ppf(x)
    >>> stats_result == stdtrit_result  # test that results are equal
    True
    """)

add_newdoc("struve",
    r"""
    struve(v, x, out=None)

    Struve function.

    Return the value of the Struve function of order `v` at `x`.  The Struve
    function is defined as,

    .. math::
        H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    v : array_like
        Order of the Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    H : scalar or ndarray
        Value of the Struve function of order `v` at `x`.

    Notes
    -----
    Three methods discussed in [1]_ are used to evaluate the Struve function:

    - power series
    - expansion in Bessel functions (if :math:`|z| < |v| + 20`)
    - asymptotic large-z expansion (if :math:`z \geq 0.7v + 12`)

    Rounding errors are estimated based on the largest terms in the sums, and
    the result associated with the smallest error is returned.

    See also
    --------
    modstruve: Modified Struve function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/11

    Examples
    --------
    Calculate the Struve function of order 1 at 2.

    >>> import numpy as np
    >>> from scipy.special import struve
    >>> import matplotlib.pyplot as plt
    >>> struve(1, 2.)
    0.6467637282835622

    Calculate the Struve function at 2 for orders 1, 2 and 3 by providing
    a list for the order parameter `v`.

    >>> struve([1, 2, 3], 2.)
    array([0.64676373, 0.28031806, 0.08363767])

    Calculate the Struve function of order 1 for several points by providing
    an array for `x`.

    >>> points = np.array([2., 5., 8.])
    >>> struve(1, points)
    array([0.64676373, 0.80781195, 0.48811605])

    Compute the Struve function for several orders at several points by
    providing arrays for `v` and `z`. The arrays have to be broadcastable
    to the correct shapes.

    >>> orders = np.array([[1], [2], [3]])
    >>> points.shape, orders.shape
    ((3,), (3, 1))

    >>> struve(orders, points)
    array([[0.64676373, 0.80781195, 0.48811605],
           [0.28031806, 1.56937455, 1.51769363],
           [0.08363767, 1.50872065, 2.98697513]])

    Plot the Struve functions of order 0 to 3 from -10 to 10.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-10., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, struve(i, x), label=f'$H_{i!r}$')
    >>> ax.legend(ncol=2)
    >>> ax.set_xlim(-10, 10)
    >>> ax.set_title(r"Struve functions $H_{\nu}$")
    >>> plt.show()
    """)

add_newdoc("tandg",
    """
    tandg(x, out=None)

    Tangent of angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Tangent at the input.

    See Also
    --------
    sindg, cosdg, cotdg

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    It is more accurate than using tangent directly.

    >>> x = 180 * np.arange(3)
    >>> sc.tandg(x)
    array([0., 0., 0.])
    >>> np.tan(x * np.pi / 180)
    array([ 0.0000000e+00, -1.2246468e-16, -2.4492936e-16])

    """)

add_newdoc(
    "tklmbda",
    r"""
    tklmbda(x, lmbda, out=None)

    Cumulative distribution function of the Tukey lambda distribution.

    Parameters
    ----------
    x, lmbda : array_like
        Parameters
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    cdf : scalar or ndarray
        Value of the Tukey lambda CDF

    See Also
    --------
    scipy.stats.tukeylambda : Tukey lambda distribution

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import tklmbda, expit

    Compute the cumulative distribution function (CDF) of the Tukey lambda
    distribution at several ``x`` values for `lmbda` = -1.5.

    >>> x = np.linspace(-2, 2, 9)
    >>> x
    array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> tklmbda(x, -1.5)
    array([0.34688734, 0.3786554 , 0.41528805, 0.45629737, 0.5       ,
           0.54370263, 0.58471195, 0.6213446 , 0.65311266])

    When `lmbda` is 0, the function is the logistic sigmoid function,
    which is implemented in `scipy.special` as `expit`.

    >>> tklmbda(x, 0)
    array([0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5       ,
           0.62245933, 0.73105858, 0.81757448, 0.88079708])
    >>> expit(x)
    array([0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5       ,
           0.62245933, 0.73105858, 0.81757448, 0.88079708])

    When `lmbda` is 1, the Tukey lambda distribution is uniform on the
    interval [-1, 1], so the CDF increases linearly.

    >>> t = np.linspace(-1, 1, 9)
    >>> tklmbda(t, 1)
    array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])

    In the following, we generate plots for several values of `lmbda`.

    The first figure shows graphs for `lmbda` <= 0.

    >>> styles = ['-', '-.', '--', ':']
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-12, 12, 500)
    >>> for k, lmbda in enumerate([-1.0, -0.5, 0.0]):
    ...     y = tklmbda(x, lmbda)
    ...     ax.plot(x, y, styles[k], label=f'$\lambda$ = {lmbda:-4.1f}')

    >>> ax.set_title('tklmbda(x, $\lambda$)')
    >>> ax.set_label('x')
    >>> ax.legend(framealpha=1, shadow=True)
    >>> ax.grid(True)

    The second figure shows graphs for `lmbda` > 0.  The dots in the
    graphs show the bounds of the support of the distribution.

    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-4.2, 4.2, 500)
    >>> lmbdas = [0.25, 0.5, 1.0, 1.5]
    >>> for k, lmbda in enumerate(lmbdas):
    ...     y = tklmbda(x, lmbda)
    ...     ax.plot(x, y, styles[k], label=f'$\lambda$ = {lmbda}')

    >>> ax.set_prop_cycle(None)
    >>> for lmbda in lmbdas:
    ...     ax.plot([-1/lmbda, 1/lmbda], [0, 1], '.', ms=8)

    >>> ax.set_title('tklmbda(x, $\lambda$)')
    >>> ax.set_xlabel('x')
    >>> ax.legend(framealpha=1, shadow=True)
    >>> ax.grid(True)

    >>> plt.tight_layout()
    >>> plt.show()

    The CDF of the Tukey lambda distribution is also implemented as the
    ``cdf`` method of `scipy.stats.tukeylambda`.  In the following,
    ``tukeylambda.cdf(x, -0.5)`` and ``tklmbda(x, -0.5)`` compute the
    same values:

    >>> from scipy.stats import tukeylambda
    >>> x = np.linspace(-2, 2, 9)

    >>> tukeylambda.cdf(x, -0.5)
    array([0.21995157, 0.27093858, 0.33541677, 0.41328161, 0.5       ,
           0.58671839, 0.66458323, 0.72906142, 0.78004843])

    >>> tklmbda(x, -0.5)
    array([0.21995157, 0.27093858, 0.33541677, 0.41328161, 0.5       ,
           0.58671839, 0.66458323, 0.72906142, 0.78004843])

    The implementation in ``tukeylambda`` also provides location and scale
    parameters, and other methods such as ``pdf()`` (the probability
    density function) and ``ppf()`` (the inverse of the CDF), so for
    working with the Tukey lambda distribution, ``tukeylambda`` is more
    generally useful.  The primary advantage of ``tklmbda`` is that it is
    significantly faster than ``tukeylambda.cdf``.
    """)

add_newdoc("wofz",
    """
    wofz(z, out=None)

    Faddeeva function

    Returns the value of the Faddeeva function for complex argument::

        exp(-z**2) * erfc(-i*z)

    Parameters
    ----------
    z : array_like
        complex argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Value of the Faddeeva function

    See Also
    --------
    dawsn, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-3, 3)
    >>> z = special.wofz(x)

    >>> plt.plot(x, z.real, label='wofz(x).real')
    >>> plt.plot(x, z.imag, label='wofz(x).imag')
    >>> plt.xlabel('$x$')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.show()

    """)

add_newdoc("xlogy",
    """
    xlogy(x, y, out=None)

    Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Parameters
    ----------
    x : array_like
        Multiplier
    y : array_like
        Argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    z : scalar or ndarray
        Computed x*log(y)

    Notes
    -----
    The log function used in the computation is the natural log.

    .. versionadded:: 0.13.0

    Examples
    --------
    We can use this function to calculate the binary logistic loss also
    known as the binary cross entropy. This loss function is used for
    binary classification problems and is defined as:

    .. math::
        L = 1/n * \\sum_{i=0}^n -(y_i*log(y\\_pred_i) + (1-y_i)*log(1-y\\_pred_i))

    We can define the parameters `x` and `y` as y and y_pred respectively.
    y is the array of the actual labels which over here can be either 0 or 1.
    y_pred is the array of the predicted probabilities with respect to
    the positive class (1).

    >>> import numpy as np
    >>> from scipy.special import xlogy
    >>> y = np.array([0, 1, 0, 1, 1, 0])
    >>> y_pred = np.array([0.3, 0.8, 0.4, 0.7, 0.9, 0.2])
    >>> n = len(y)
    >>> loss = -(xlogy(y, y_pred) + xlogy(1 - y, 1 - y_pred)).sum()
    >>> loss /= n
    >>> loss
    0.29597052165495025

    A lower loss is usually better as it indicates that the predictions are
    similar to the actual labels. In this example since our predicted
    probabilties are close to the actual labels, we get an overall loss
    that is reasonably low and appropriate.

    """)

add_newdoc("xlog1py",
    """
    xlog1py(x, y, out=None)

    Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.

    Parameters
    ----------
    x : array_like
        Multiplier
    y : array_like
        Argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    z : scalar or ndarray
        Computed x*log1p(y)

    Notes
    -----

    .. versionadded:: 0.13.0

    Examples
    --------
    This example shows how the function can be used to calculate the log of
    the probability mass function for a geometric discrete random variable.
    The probability mass function of the geometric distribution is defined
    as follows:

    .. math:: f(k) = (1-p)^{k-1} p

    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure
    and :math:`k` is the number of trials to get the first success.

    >>> import numpy as np
    >>> from scipy.special import xlog1py
    >>> p = 0.5
    >>> k = 100
    >>> _pmf = np.power(1 - p, k - 1) * p
    >>> _pmf
    7.888609052210118e-31

    If we take k as a relatively large number the value of the probability
    mass function can become very low. In such cases taking the log of the
    pmf would be more suitable as the log function can change the values
    to a scale that is more appropriate to work with.

    >>> _log_pmf = xlog1py(k - 1, -p) + np.log(p)
    >>> _log_pmf
    -69.31471805599453

    We can confirm that we get a value close to the original pmf value by
    taking the exponential of the log pmf.

    >>> _orig_pmf = np.exp(_log_pmf)
    >>> np.isclose(_pmf, _orig_pmf)
    True

    """)

add_newdoc("y0",
    r"""
    y0(x, out=None)

    Bessel function of the second kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function of the second kind of order 0 at `x`.

    Notes
    -----

    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval a rational approximation :math:`R(x)` is employed to
    compute,

    .. math::

        Y_0(x) = R(x) + \frac{2 \log(x) J_0(x)}{\pi},

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `y0`.

    See also
    --------
    j0: Bessel function of the first kind of order 0
    yv: Bessel function of the first kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import y0
    >>> y0(1.)
    0.08825696421567697

    Calculate at several points:

    >>> import numpy as np
    >>> y0(np.array([0.5, 2., 3.]))
    array([-0.44451873,  0.51037567,  0.37685001])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = y0(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("y1",
    """
    y1(x, out=None)

    Bessel function of the second kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function of the second kind of order 1 at `x`.

    Notes
    -----

    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 25 term Chebyshev expansion is used, and computing
    :math:`J_1` (the Bessel function of the first kind) is required. In the
    second, the asymptotic trigonometric representation is employed using two
    rational functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `y1`.

    See also
    --------
    j1: Bessel function of the first kind of order 1
    yn: Bessel function of the second kind
    yv: Bessel function of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Calculate the function at one point:

    >>> from scipy.special import y1
    >>> y1(1.)
    -0.7812128213002888

    Calculate at several points:

    >>> import numpy as np
    >>> y1(np.array([0.5, 2., 3.]))
    array([-1.47147239, -0.10703243,  0.32467442])

    Plot the function from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> y = y1(x)
    >>> ax.plot(x, y)
    >>> plt.show()

    """)

add_newdoc("yn",
    r"""
    yn(n, x, out=None)

    Bessel function of the second kind of integer order and real argument.

    Parameters
    ----------
    n : array_like
        Order (integer).
    x : array_like
        Argument (float).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function, :math:`Y_n(x)`.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `yn`.

    The function is evaluated by forward recurrence on `n`, starting with
    values computed by the Cephes routines `y0` and `y1`. If `n = 0` or 1,
    the routine for `y0` or `y1` is called directly.

    See also
    --------
    yv : For real order and real or complex argument.
    y0: faster implementation of this function for order 0
    y1: faster implementation of this function for order 1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import yn
    >>> yn(0, 1.)
    0.08825696421567697

    Evaluate the function at one point for different orders.

    >>> yn(0, 1.), yn(1, 1.), yn(2, 1.)
    (0.08825696421567697, -0.7812128213002888, -1.6506826068162546)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> yn([0, 1, 2], 1.)
    array([ 0.08825696, -0.78121282, -1.65068261])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 3., 8.])
    >>> yn(0, points)
    array([-0.44451873,  0.37685001,  0.22352149])

    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    To calculate the orders 0 and 1 for an 1D array:

    >>> orders = np.array([[0], [1]])
    >>> orders.shape
    (2, 1)

    >>> yn(orders, points)
    array([[-0.44451873,  0.37685001,  0.22352149],
           [-1.47147239,  0.32467442, -0.15806046]])

    Plot the functions of order 0 to 3 from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, yn(i, x), label=f'$Y_{i!r}$')
    >>> ax.set_ylim(-3, 1)
    >>> ax.legend()
    >>> plt.show()
    """)

add_newdoc("yv",
    r"""
    yv(v, z, out=None)

    Bessel function of the second kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the Bessel function of the second kind, :math:`Y_v(x)`.

    Notes
    -----
    For positive `v` values, the computation is carried out using the
    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,

    .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).

    For negative `v` values the formula,

    .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)

    is used, where :math:`J_v(z)` is the Bessel function of the first kind,
    computed using the AMOS routine `zbesj`.  Note that the second term is
    exactly zero for integer `v`; to improve accuracy the second term is
    explicitly omitted for `v` values such that `v = floor(v)`.

    See also
    --------
    yve : :math:`Y_v` with leading exponential behavior stripped off.
    y0: faster implementation of this function for order 0
    y1: faster implementation of this function for order 1

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Evaluate the function of order 0 at one point.

    >>> from scipy.special import yv
    >>> yv(0, 1.)
    0.088256964215677

    Evaluate the function at one point for different orders.

    >>> yv(0, 1.), yv(1, 1.), yv(1.5, 1.)
    (0.088256964215677, -0.7812128213002889, -1.102495575160179)

    The evaluation for different orders can be carried out in one call by
    providing a list or NumPy array as argument for the `v` parameter:

    >>> yv([0, 1, 1.5], 1.)
    array([ 0.08825696, -0.78121282, -1.10249558])

    Evaluate the function at several points for order 0 by providing an
    array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 3., 8.])
    >>> yv(0, points)
    array([-0.44451873,  0.37685001,  0.22352149])

    If `z` is an array, the order parameter `v` must be broadcastable to
    the correct shape if different orders shall be computed in one call.
    To calculate the orders 0 and 1 for an 1D array:

    >>> orders = np.array([[0], [1]])
    >>> orders.shape
    (2, 1)

    >>> yv(orders, points)
    array([[-0.44451873,  0.37685001,  0.22352149],
           [-1.47147239,  0.32467442, -0.15806046]])

    Plot the functions of order 0 to 3 from 0 to 10.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0., 10., 1000)
    >>> for i in range(4):
    ...     ax.plot(x, yv(i, x), label=f'$Y_{i!r}$')
    >>> ax.set_ylim(-3, 1)
    >>> ax.legend()
    >>> plt.show()

    """)

add_newdoc("yve",
    r"""
    yve(v, z, out=None)

    Exponentially scaled Bessel function of the second kind of real order.

    Returns the exponentially scaled Bessel function of the second
    kind of real order `v` at complex `z`::

        yve(v, z) = yv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    Y : scalar or ndarray
        Value of the exponentially scaled Bessel function.

    See Also
    --------
    yv: Unscaled Bessel function of the second kind of real order.

    Notes
    -----
    For positive `v` values, the computation is carried out using the
    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,

    .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).

    For negative `v` values the formula,

    .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)

    is used, where :math:`J_v(z)` is the Bessel function of the first kind,
    computed using the AMOS routine `zbesj`.  Note that the second term is
    exactly zero for integer `v`; to improve accuracy the second term is
    explicitly omitted for `v` values such that `v = floor(v)`.

    Exponentially scaled Bessel functions are useful for large `z`:
    for these, the unscaled Bessel functions can easily under-or overflow.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compare the output of `yv` and `yve` for large complex arguments for `z`
    by computing their values for order ``v=1`` at ``z=1000j``. We see that
    `yv` returns nan but `yve` returns a finite number:

    >>> import numpy as np
    >>> from scipy.special import yv, yve
    >>> v = 1
    >>> z = 1000j
    >>> yv(v, z), yve(v, z)
    ((nan+nanj), (-0.012610930256928629+7.721967686709076e-19j))

    For real arguments for `z`, `yve` returns the same as `yv` up to
    floating point errors.

    >>> v, z = 1, 1000
    >>> yv(v, z), yve(v, z)
    (-0.02478433129235178, -0.02478433129235179)

    The function can be evaluated for several orders at the same time by
    providing a list or NumPy array for `v`:

    >>> yve([1, 2, 3], 1j)
    array([-0.20791042+0.14096627j,  0.38053618-0.04993878j,
           0.00815531-1.66311097j])

    In the same way, the function can be evaluated at several points in one
    call by providing a list or NumPy array for `z`:

    >>> yve(1, np.array([1j, 2j, 3j]))
    array([-0.20791042+0.14096627j, -0.21526929+0.01205044j,
           -0.19682671+0.00127278j])

    It is also possible to evaluate several orders at several points
    at the same time by providing arrays for `v` and `z` with
    broadcasting compatible shapes. Compute `yve` for two different orders
    `v` and three points `z` resulting in a 2x3 array.

    >>> v = np.array([[1], [2]])
    >>> z = np.array([3j, 4j, 5j])
    >>> v.shape, z.shape
    ((2, 1), (3,))

    >>> yve(v, z)
    array([[-1.96826713e-01+1.27277544e-03j, -1.78750840e-01+1.45558819e-04j,
            -1.63972267e-01+1.73494110e-05j],
           [1.94960056e-03-1.11782545e-01j,  2.02902325e-04-1.17626501e-01j,
            2.27727687e-05-1.17951906e-01j]])
    """)

add_newdoc("_zeta",
    """
    _zeta(x, q)

    Internal function, Hurwitz zeta.

    """)

add_newdoc("zetac",
    """
    zetac(x, out=None)

    Riemann zeta function minus 1.

    This function is defined as

    .. math:: \\zeta(x) = \\sum_{k=2}^{\\infty} 1 / k^x,

    where ``x > 1``.  For ``x < 1`` the analytic continuation is
    computed. For more information on the Riemann zeta function, see
    [dlmf]_.

    Parameters
    ----------
    x : array_like of float
        Values at which to compute zeta(x) - 1 (must be real).
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of zeta(x) - 1.

    See Also
    --------
    zeta

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import zetac, zeta

    Some special values:

    >>> zetac(2), np.pi**2/6 - 1
    (0.64493406684822641, 0.6449340668482264)

    >>> zetac(-1), -1.0/12 - 1
    (-1.0833333333333333, -1.0833333333333333)

    Compare ``zetac(x)`` to ``zeta(x) - 1`` for large `x`:

    >>> zetac(60), zeta(60) - 1
    (8.673617380119933e-19, 0.0)

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/25

    """)

add_newdoc("_riemann_zeta",
    """
    Internal function, use `zeta` instead.
    """)

add_newdoc("_struve_asymp_large_z",
    """
    _struve_asymp_large_z(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using asymptotic expansion

    Returns
    -------
    v, err
    """)

add_newdoc("_struve_power_series",
    """
    _struve_power_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using power series

    Returns
    -------
    v, err
    """)

add_newdoc("_struve_bessel_series",
    """
    _struve_bessel_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using Bessel function series

    Returns
    -------
    v, err
    """)

add_newdoc("_spherical_jn",
    """
    Internal function, use `spherical_jn` instead.
    """)

add_newdoc("_spherical_jn_d",
    """
    Internal function, use `spherical_jn` instead.
    """)

add_newdoc("_spherical_yn",
    """
    Internal function, use `spherical_yn` instead.
    """)

add_newdoc("_spherical_yn_d",
    """
    Internal function, use `spherical_yn` instead.
    """)

add_newdoc("_spherical_in",
    """
    Internal function, use `spherical_in` instead.
    """)

add_newdoc("_spherical_in_d",
    """
    Internal function, use `spherical_in` instead.
    """)

add_newdoc("_spherical_kn",
    """
    Internal function, use `spherical_kn` instead.
    """)

add_newdoc("_spherical_kn_d",
    """
    Internal function, use `spherical_kn` instead.
    """)

add_newdoc("loggamma",
    r"""
    loggamma(z, out=None)

    Principal branch of the logarithm of the gamma function.

    Defined to be :math:`\log(\Gamma(x))` for :math:`x > 0` and
    extended to the complex plane by analytic continuation. The
    function has a single branch cut on the negative real axis.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    z : array_like
        Values in the complex plane at which to compute ``loggamma``
    out : ndarray, optional
        Output array for computed values of ``loggamma``

    Returns
    -------
    loggamma : scalar or ndarray
        Values of ``loggamma`` at z.

    Notes
    -----
    It is not generally true that :math:`\log\Gamma(z) =
    \log(\Gamma(z))`, though the real parts of the functions do
    agree. The benefit of not defining `loggamma` as
    :math:`\log(\Gamma(z))` is that the latter function has a
    complicated branch cut structure whereas `loggamma` is analytic
    except for on the negative real axis.

    The identities

    .. math::
      \exp(\log\Gamma(z)) &= \Gamma(z) \\
      \log\Gamma(z + 1) &= \log(z) + \log\Gamma(z)

    make `loggamma` useful for working in complex logspace.

    On the real line `loggamma` is related to `gammaln` via
    ``exp(loggamma(x + 0j)) = gammasgn(x)*exp(gammaln(x))``, up to
    rounding error.

    The implementation here is based on [hare1997]_.

    See also
    --------
    gammaln : logarithm of the absolute value of the gamma function
    gammasgn : sign of the gamma function

    References
    ----------
    .. [hare1997] D.E.G. Hare,
      *Computing the Principal Branch of log-Gamma*,
      Journal of Algorithms, Volume 25, Issue 2, November 1997, pages 221-236.
    """)

add_newdoc("_sinpi",
    """
    Internal function, do not use.
    """)

add_newdoc("_cospi",
    """
    Internal function, do not use.
    """)

add_newdoc("owens_t",
    """
    owens_t(h, a, out=None)

    Owen's T Function.

    The function T(h, a) gives the probability of the event
    (X > h and 0 < Y < a * X) where X and Y are independent
    standard normal random variables.

    Parameters
    ----------
    h: array_like
        Input value.
    a: array_like
        Input value.
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    t: scalar or ndarray
        Probability of the event (X > h and 0 < Y < a * X),
        where X and Y are independent standard normal random variables.

    Examples
    --------
    >>> from scipy import special
    >>> a = 3.5
    >>> h = 0.78
    >>> special.owens_t(h, a)
    0.10877216734852274

    References
    ----------
    .. [1] M. Patefield and D. Tandy, "Fast and accurate calculation of
           Owen's T Function", Statistical Software vol. 5, pp. 1-25, 2000.
    """)

add_newdoc("_factorial",
    """
    Internal function, do not use.
    """)

add_newdoc("wright_bessel",
    r"""
    wright_bessel(a, b, x, out=None)

    Wright's generalized Bessel function.

    Wright's generalized Bessel function is an entire function and defined as

    .. math:: \Phi(a, b; x) = \sum_{k=0}^\infty \frac{x^k}{k! \Gamma(a k + b)}

    See also [1].

    Parameters
    ----------
    a : array_like of float
        a >= 0
    b : array_like of float
        b >= 0
    x : array_like of float
        x >= 0
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Value of the Wright's generalized Bessel function

    Notes
    -----
    Due to the compexity of the function with its three parameters, only
    non-negative arguments are implemented.

    Examples
    --------
    >>> from scipy.special import wright_bessel
    >>> a, b, x = 1.5, 1.1, 2.5
    >>> wright_bessel(a, b-1, x)
    4.5314465939443025

    Now, let us verify the relation

    .. math:: \Phi(a, b-1; x) = a x \Phi(a, b+a; x) + (b-1) \Phi(a, b; x)

    >>> a * x * wright_bessel(a, b+a, x) + (b-1) * wright_bessel(a, b, x)
    4.5314465939443025

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 10.46.
           https://dlmf.nist.gov/10.46.E1
    """)


add_newdoc("ndtri_exp",
    r"""
    ndtri_exp(y, out=None)

    Inverse of `log_ndtr` vs x. Allows for greater precision than
    `ndtri` composed with `numpy.exp` for very small values of y and for
    y close to 0.

    Parameters
    ----------
    y : array_like of float
        Function argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Inverse of the log CDF of the standard normal distribution, evaluated
        at y.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.special as sc

    `ndtri_exp` agrees with the naive implementation when the latter does
    not suffer from underflow.

    >>> sc.ndtri_exp(-1)
    -0.33747496376420244
    >>> sc.ndtri(np.exp(-1))
    -0.33747496376420244

    For extreme values of y, the naive approach fails

    >>> sc.ndtri(np.exp(-800))
    -inf
    >>> sc.ndtri(np.exp(-1e-20))
    inf

    whereas `ndtri_exp` is still able to compute the result to high precision.

    >>> sc.ndtri_exp(-800)
    -39.88469483825668
    >>> sc.ndtri_exp(-1e-20)
    9.262340089798409

    See Also
    --------
    log_ndtr, ndtri, ndtr
    """)
