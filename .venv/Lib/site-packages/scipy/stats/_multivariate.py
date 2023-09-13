#
# Author: Joris Vankerschaver 2013
#
import math
import numpy as np
from numpy import asarray_chkfinite, asarray
from numpy.lib import NumpyVersion
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
                           ive, loggamma)
from scipy._lib._util import check_random_state
from scipy.linalg.blas import drot
from scipy.linalg._misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar

__all__ = ['multivariate_normal',
           'matrix_normal',
           'dirichlet',
           'dirichlet_multinomial',
           'wishart',
           'invwishart',
           'multinomial',
           'special_ortho_group',
           'ortho_group',
           'random_correlation',
           'unitary_group',
           'multivariate_t',
           'multivariate_hypergeom',
           'random_table',
           'uniform_direction',
           'vonmises_fisher']

_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


_doc_random_state = """\
seed : {None, int, np.random.RandomState, np.random.Generator}, optional
    Used for drawing random variates.
    If `seed` is `None`, the `~np.random.RandomState` singleton is used.
    If `seed` is an int, a new ``RandomState`` instance is used, seeded
    with seed.
    If `seed` is already a ``RandomState`` or ``Generator`` instance,
    then that object is used.
    Default is `None`.
"""


def _squeeze_output(out):
    """
    Remove single-dimensional entries from array and convert to scalar,
    if necessary.
    """
    out = out.squeeze()
    if out.ndim == 0:
        out = out[()]
    return out


def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


def _pinv_1d(v, eps=1e-5):
    """A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.

    """
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


class _PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.

    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().

    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)

    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().

    """

    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        self._M = np.asarray(M)

        # Compute the symmetric eigendecomposition.
        # Note that eigh takes care of array conversion, chkfinite,
        # and assertion that the matrix is square.
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            msg = "The input matrix must be symmetric positive semidefinite."
            raise ValueError(msg)
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            msg = ("When `allow_singular is False`, the input matrix must be "
                   "symmetric positive definite.")
            raise np.linalg.LinAlgError(msg)
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Save the eigenvector basis, and tolerance for testing support
        self.eps = 1e3*eps
        self.V = u[:, s <= eps]

        # Initialize the eagerly precomputed attributes.
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))

        # Initialize attributes to be lazily computed.
        self._pinv = None

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        residual = np.linalg.norm(x @ self.V, axis=-1)
        in_support = residual < self.eps
        return in_support

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv


class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.
    """
    def __init__(self, seed=None):
        super().__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the Generator object for generating random variates.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):
        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state


class multi_rv_frozen:
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """
    @property
    def random_state(self):
        return self._dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self._dist._random_state = check_random_state(seed)


_mvn_doc_default_callparams = """\
mean : array_like, default: ``[0]``
    Mean of the distribution.
cov : array_like or `Covariance`, default: ``[1]``
    Symmetric positive (semi)definite covariance matrix of the distribution.
allow_singular : bool, default: ``False``
    Whether to allow a singular covariance matrix. This is ignored if `cov` is
    a `Covariance` object.
"""

_mvn_doc_callparams_note = """\
Setting the parameter `mean` to `None` is equivalent to having `mean`
be the zero-vector. The parameter `cov` can be a scalar, in which case
the covariance matrix is the identity times that value, a vector of
diagonal entries for the covariance matrix, a two-dimensional array_like,
or a `Covariance` object.
"""

_mvn_doc_frozen_callparams = ""

_mvn_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

mvn_docdict_params = {
    '_mvn_doc_default_callparams': _mvn_doc_default_callparams,
    '_mvn_doc_callparams_note': _mvn_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

mvn_docdict_noparams = {
    '_mvn_doc_default_callparams': _mvn_doc_frozen_callparams,
    '_mvn_doc_callparams_note': _mvn_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multivariate_normal_gen(multi_rv_generic):
    r"""A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    pdf(x, mean=None, cov=1, allow_singular=False)
        Probability density function.
    logpdf(x, mean=None, cov=1, allow_singular=False)
        Log of the probability density function.
    cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5, lower_limit=None)  # noqa
        Cumulative distribution function.
    logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)
        Log of the cumulative distribution function.
    rvs(mean=None, cov=1, size=1, random_state=None)
        Draw random samples from a multivariate normal distribution.
    entropy()
        Compute the differential entropy of the multivariate normal.

    Parameters
    ----------
    %(_mvn_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvn_doc_callparams_note)s

    The covariance matrix `cov` may be an instance of a subclass of
    `Covariance`, e.g. `scipy.stats.CovViaPrecision`. If so, `allow_singular`
    is ignored.

    Otherwise, `cov` must be a symmetric positive semidefinite
    matrix when `allow_singular` is True; it must be (strictly) positive
    definite when `allow_singular` is False.
    Symmetry is not checked; only the lower triangular portion is used.
    The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \frac{1}{\sqrt{(2 \pi)^k \det \Sigma}}
               \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

    where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
    :math:`k` the rank of :math:`\Sigma`. In case of singular :math:`\Sigma`,
    SciPy extends this definition according to [1]_.

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Multivariate Normal Distribution - Degenerate Case, Wikipedia,
           https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Degenerate_case

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_normal

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)
    >>> plt.show()

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    >>> rv = multivariate_normal(mean=None, cov=1, allow_singular=False)
    >>> # Frozen object with the same methods but holding the given
    >>> # mean and covariance fixed.

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvn_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """Create a frozen multivariate normal distribution.

        See `multivariate_normal_frozen` for more information.
        """
        return multivariate_normal_frozen(mean, cov,
                                          allow_singular=allow_singular,
                                          seed=seed)

    def _process_parameters(self, mean, cov, allow_singular=True):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.
        """
        if isinstance(cov, _covariance.Covariance):
            return self._process_parameters_Covariance(mean, cov)
        else:
            # Before `Covariance` classes were introduced,
            # `multivariate_normal` accepted plain arrays as `cov` and used the
            # following input validation. To avoid disturbing the behavior of
            # `multivariate_normal` when plain arrays are used, we use the
            # original input validation here.
            dim, mean, cov = self._process_parameters_psd(None, mean, cov)
            # After input validation, some methods then processed the arrays
            # with a `_PSD` object and used that to perform computation.
            # To avoid branching statements in each method depending on whether
            # `cov` is an array or `Covariance` object, we always process the
            # array with `_PSD`, and then use wrapper that satisfies the
            # `Covariance` interface, `CovViaPSD`.
            psd = _PSD(cov, allow_singular=allow_singular)
            cov_object = _covariance.CovViaPSD(psd)
            return dim, mean, cov_object

    def _process_parameters_Covariance(self, mean, cov):
        dim = cov.shape[-1]
        mean = np.array([0.]) if mean is None else mean
        message = (f"`cov` represents a covariance matrix in {dim} dimensions,"
                   f"and so `mean` must be broadcastable to shape {(dim,)}")
        try:
            mean = np.broadcast_to(mean, dim)
        except ValueError as e:
            raise ValueError(message) from e
        return dim, mean, cov

    def _process_parameters_psd(self, dim, mean, cov):
        # Try to infer dimensionality
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be "
                                 "a scalar.")

        # Check input sizes and return full arrays for mean and cov if
        # necessary
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)

        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean = mean.reshape(1)
            cov = cov.reshape(1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." %
                             dim)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(cov.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % cov.ndim)

        return dim, mean, cov

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, mean, cov_object):
        """Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        cov_object : Covariance
            An object representing the Covariance matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        log_det_cov, rank = cov_object.log_pdet, cov_object.rank
        dev = x - mean
        if dev.ndim > 1:
            log_det_cov = log_det_cov[..., np.newaxis]
            rank = rank[..., np.newaxis]
        maha = np.sum(np.square(cov_object.whiten(dev)), axis=-1)
        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        """Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = self._logpdf(x, mean, cov_object)
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x-mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        """Multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = np.exp(self._logpdf(x, mean, cov_object))
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x-mean)
            out[out_of_bounds] = 0.0
        return _squeeze_output(out)

    def _cdf(self, x, mean, cov, maxpts, abseps, releps, lower_limit):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        mean : ndarray
            Mean of the distribution
        cov : array_like
            Covariance matrix of the distribution
        maxpts : integer
            The maximum number of points to use for integration
        abseps : float
            Absolute error tolerance
        releps : float
            Relative error tolerance
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.


        .. versionadded:: 1.0.0

        """
        lower = (np.full(mean.shape, -np.inf)
                 if lower_limit is None else lower_limit)
        # In 2d, _mvn.mvnun accepts input in which `lower` bound elements
        # are greater than `x`. Not so in other dimensions. Fix this by
        # ensuring that lower bounds are indeed lower when passed, then
        # set signs of resulting CDF manually.
        b, a = np.broadcast_arrays(x, lower)
        i_swap = b < a
        signs = (-1)**(i_swap.sum(axis=-1))  # odd # of swaps -> negative
        a, b = a.copy(), b.copy()
        a[i_swap], b[i_swap] = b[i_swap], a[i_swap]
        n = x.shape[-1]
        limits = np.concatenate((a, b), axis=-1)

        # mvnun expects 1-d arguments, so process points sequentially
        def func1d(limits):
            return _mvn.mvnun(limits[:n], limits[n:], mean, cov,
                              maxpts, abseps, releps)[0]

        out = np.apply_along_axis(func1d, -1, limits) * signs
        return _squeeze_output(out)

    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
               abseps=1e-5, releps=1e-5, *, lower_limit=None):
        """Log of the multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        cov = cov_object.covariance
        x = self._process_quantiles(x, dim)
        if not maxpts:
            maxpts = 1000000 * dim
        cdf = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        # the log of a negative real is complex, and cdf can be negative
        # if lower limit is greater than upper limit
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        out = np.log(cdf)
        return out

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5, *, lower_limit=None):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        cov = cov_object.covariance
        x = self._process_quantiles(x, dim)
        if not maxpts:
            maxpts = 1000000 * dim
        out = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        return out

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        dim, mean, cov_object = self._process_parameters(mean, cov)
        random_state = self._get_random_state(random_state)

        if isinstance(cov_object, _covariance.CovViaPSD):
            cov = cov_object.covariance
            out = random_state.multivariate_normal(mean, cov, size)
            out = _squeeze_output(out)
        else:
            size = size or tuple()
            if not np.iterable(size):
                size = (size,)
            shape = tuple(size) + (cov_object.shape[-1],)
            x = random_state.normal(size=shape)
            out = mean + cov_object.colorize(x)
        return out

    def entropy(self, mean=None, cov=1):
        """Compute the differential entropy of the multivariate normal.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        dim, mean, cov_object = self._process_parameters(mean, cov)
        return 0.5 * (cov_object.rank * (_LOG_2PI + 1) + cov_object.log_pdet)


multivariate_normal = multivariate_normal_gen()


class multivariate_normal_frozen(multi_rv_frozen):
    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        """Create a frozen multivariate normal distribution.

        Parameters
        ----------
        mean : array_like, default: ``[0]``
            Mean of the distribution.
        cov : array_like, default: ``[1]``
            Symmetric positive (semi)definite covariance matrix of the
            distribution.
        allow_singular : bool, default: ``False``
            Whether to allow a singular covariance matrix.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        maxpts : integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps : float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_normal
        >>> r = multivariate_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self._dist = multivariate_normal_gen(seed)
        self.dim, self.mean, self.cov_object = (
            self._dist._process_parameters(mean, cov, allow_singular))
        self.allow_singular = allow_singular or self.cov_object._allow_singular
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    @property
    def cov(self):
        return self.cov_object.covariance

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.cov_object)
        if np.any(self.cov_object.rank < self.dim):
            out_of_bounds = ~self.cov_object._support_mask(x-self.mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x, *, lower_limit=None):
        cdf = self.cdf(x, lower_limit=lower_limit)
        # the log of a negative real is complex, and cdf can be negative
        # if lower limit is greater than upper limit
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        out = np.log(cdf)
        return out

    def cdf(self, x, *, lower_limit=None):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._cdf(x, self.mean, self.cov_object.covariance,
                              self.maxpts, self.abseps, self.releps,
                              lower_limit)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.cov_object, size, random_state)

    def entropy(self):
        """Computes the differential entropy of the multivariate normal.

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        """
        log_pdet = self.cov_object.log_pdet
        rank = self.cov_object.rank
        return 0.5 * (rank * (_LOG_2PI + 1) + log_pdet)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'logcdf', 'cdf', 'rvs']:
    method = multivariate_normal_gen.__dict__[name]
    method_frozen = multivariate_normal_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             mvn_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvn_docdict_params)

_matnorm_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default: `None`)
rowcov : array_like, optional
    Among-row covariance matrix of the distribution (default: `1`)
colcov : array_like, optional
    Among-column covariance matrix of the distribution (default: `1`)
"""

_matnorm_doc_callparams_note = """\
If `mean` is set to `None` then a matrix of zeros is used for the mean.
The dimensions of this matrix are inferred from the shape of `rowcov` and
`colcov`, if these are provided, or set to `1` if ambiguous.

`rowcov` and `colcov` can be two-dimensional array_likes specifying the
covariance matrices directly. Alternatively, a one-dimensional array will
be be interpreted as the entries of a diagonal matrix, and a scalar or
zero-dimensional array will be interpreted as this value times the
identity matrix.
"""

_matnorm_doc_frozen_callparams = ""

_matnorm_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

matnorm_docdict_params = {
    '_matnorm_doc_default_callparams': _matnorm_doc_default_callparams,
    '_matnorm_doc_callparams_note': _matnorm_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

matnorm_docdict_noparams = {
    '_matnorm_doc_default_callparams': _matnorm_doc_frozen_callparams,
    '_matnorm_doc_callparams_note': _matnorm_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class matrix_normal_gen(multi_rv_generic):
    r"""A matrix normal random variable.

    The `mean` keyword specifies the mean. The `rowcov` keyword specifies the
    among-row covariance matrix. The 'colcov' keyword specifies the
    among-column covariance matrix.

    Methods
    -------
    pdf(X, mean=None, rowcov=1, colcov=1)
        Probability density function.
    logpdf(X, mean=None, rowcov=1, colcov=1)
        Log of the probability density function.
    rvs(mean=None, rowcov=1, colcov=1, size=1, random_state=None)
        Draw random samples.
    entropy(rowcol=1, colcov=1)
        Differential entropy.

    Parameters
    ----------
    %(_matnorm_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_matnorm_doc_callparams_note)s

    The covariance matrices specified by `rowcov` and `colcov` must be
    (symmetric) positive definite. If the samples in `X` are
    :math:`m \times n`, then `rowcov` must be :math:`m \times m` and
    `colcov` must be :math:`n \times n`. `mean` must be the same shape as `X`.

    The probability density function for `matrix_normal` is

    .. math::

        f(X) = (2 \pi)^{-\frac{mn}{2}}|U|^{-\frac{n}{2}} |V|^{-\frac{m}{2}}
               \exp\left( -\frac{1}{2} \mathrm{Tr}\left[ U^{-1} (X-M) V^{-1}
               (X-M)^T \right] \right),

    where :math:`M` is the mean, :math:`U` the among-row covariance matrix,
    :math:`V` the among-column covariance matrix.

    The `allow_singular` behaviour of the `multivariate_normal`
    distribution is not currently supported. Covariance matrices must be
    full rank.

    The `matrix_normal` distribution is closely related to the
    `multivariate_normal` distribution. Specifically, :math:`\mathrm{Vec}(X)`
    (the vector formed by concatenating the columns  of :math:`X`) has a
    multivariate normal distribution with mean :math:`\mathrm{Vec}(M)`
    and covariance :math:`V \otimes U` (where :math:`\otimes` is the Kronecker
    product). Sampling and pdf evaluation are
    :math:`\mathcal{O}(m^3 + n^3 + m^2 n + m n^2)` for the matrix normal, but
    :math:`\mathcal{O}(m^3 n^3)` for the equivalent multivariate normal,
    making this equivalent form algorithmically inefficient.

    .. versionadded:: 0.17.0

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.stats import matrix_normal

    >>> M = np.arange(6).reshape(3,2); M
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> U = np.diag([1,2,3]); U
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    >>> V = 0.3*np.identity(2); V
    array([[ 0.3,  0. ],
           [ 0. ,  0.3]])
    >>> X = M + 0.1; X
    array([[ 0.1,  1.1],
           [ 2.1,  3.1],
           [ 4.1,  5.1]])
    >>> matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
    0.023410202050005054

    >>> # Equivalent multivariate normal
    >>> from scipy.stats import multivariate_normal
    >>> vectorised_X = X.T.flatten()
    >>> equiv_mean = M.T.flatten()
    >>> equiv_cov = np.kron(V,U)
    >>> multivariate_normal.pdf(vectorised_X, mean=equiv_mean, cov=equiv_cov)
    0.023410202050005054

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" matrix normal
    random variable:

    >>> rv = matrix_normal(mean=None, rowcov=1, colcov=1)
    >>> # Frozen object with the same methods but holding the given
    >>> # mean and covariance fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, matnorm_docdict_params)

    def __call__(self, mean=None, rowcov=1, colcov=1, seed=None):
        """Create a frozen matrix normal distribution.

        See `matrix_normal_frozen` for more information.

        """
        return matrix_normal_frozen(mean, rowcov, colcov, seed=seed)

    def _process_parameters(self, mean, rowcov, colcov):
        """
        Infer dimensionality from mean or covariance matrices. Handle
        defaults. Ensure compatible dimensions.
        """

        # Process mean
        if mean is not None:
            mean = np.asarray(mean, dtype=float)
            meanshape = mean.shape
            if len(meanshape) != 2:
                raise ValueError("Array `mean` must be two dimensional.")
            if np.any(meanshape == 0):
                raise ValueError("Array `mean` has invalid shape.")

        # Process among-row covariance
        rowcov = np.asarray(rowcov, dtype=float)
        if rowcov.ndim == 0:
            if mean is not None:
                rowcov = rowcov * np.identity(meanshape[0])
            else:
                rowcov = rowcov * np.identity(1)
        elif rowcov.ndim == 1:
            rowcov = np.diag(rowcov)
        rowshape = rowcov.shape
        if len(rowshape) != 2:
            raise ValueError("`rowcov` must be a scalar or a 2D array.")
        if rowshape[0] != rowshape[1]:
            raise ValueError("Array `rowcov` must be square.")
        if rowshape[0] == 0:
            raise ValueError("Array `rowcov` has invalid shape.")
        numrows = rowshape[0]

        # Process among-column covariance
        colcov = np.asarray(colcov, dtype=float)
        if colcov.ndim == 0:
            if mean is not None:
                colcov = colcov * np.identity(meanshape[1])
            else:
                colcov = colcov * np.identity(1)
        elif colcov.ndim == 1:
            colcov = np.diag(colcov)
        colshape = colcov.shape
        if len(colshape) != 2:
            raise ValueError("`colcov` must be a scalar or a 2D array.")
        if colshape[0] != colshape[1]:
            raise ValueError("Array `colcov` must be square.")
        if colshape[0] == 0:
            raise ValueError("Array `colcov` has invalid shape.")
        numcols = colshape[0]

        # Ensure mean and covariances compatible
        if mean is not None:
            if meanshape[0] != numrows:
                raise ValueError("Arrays `mean` and `rowcov` must have the "
                                 "same number of rows.")
            if meanshape[1] != numcols:
                raise ValueError("Arrays `mean` and `colcov` must have the "
                                 "same number of columns.")
        else:
            mean = np.zeros((numrows, numcols))

        dims = (numrows, numcols)

        return dims, mean, rowcov, colcov

    def _process_quantiles(self, X, dims):
        """
        Adjust quantiles array so that last two axes labels the components of
        each data point.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, :]
        if X.shape[-2:] != dims:
            raise ValueError("The shape of array `X` is not compatible "
                             "with the distribution parameters.")
        return X

    def _logpdf(self, dims, X, mean, row_prec_rt, log_det_rowcov,
                col_prec_rt, log_det_colcov):
        """Log of the matrix normal probability density function.

        Parameters
        ----------
        dims : tuple
            Dimensions of the matrix variates
        X : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        row_prec_rt : ndarray
            A decomposition such that np.dot(row_prec_rt, row_prec_rt.T)
            is the inverse of the among-row covariance matrix
        log_det_rowcov : float
            Logarithm of the determinant of the among-row covariance matrix
        col_prec_rt : ndarray
            A decomposition such that np.dot(col_prec_rt, col_prec_rt.T)
            is the inverse of the among-column covariance matrix
        log_det_colcov : float
            Logarithm of the determinant of the among-column covariance matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        numrows, numcols = dims
        roll_dev = np.moveaxis(X-mean, -1, 0)
        scale_dev = np.tensordot(col_prec_rt.T,
                                 np.dot(roll_dev, row_prec_rt), 1)
        maha = np.sum(np.sum(np.square(scale_dev), axis=-1), axis=0)
        return -0.5 * (numrows*numcols*_LOG_2PI + numcols*log_det_rowcov
                       + numrows*log_det_colcov + maha)

    def logpdf(self, X, mean=None, rowcov=1, colcov=1):
        """Log of the matrix normal probability density function.

        Parameters
        ----------
        X : array_like
            Quantiles, with the last two axes of `X` denoting the components.
        %(_matnorm_doc_default_callparams)s

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at `X`

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        dims, mean, rowcov, colcov = self._process_parameters(mean, rowcov,
                                                              colcov)
        X = self._process_quantiles(X, dims)
        rowpsd = _PSD(rowcov, allow_singular=False)
        colpsd = _PSD(colcov, allow_singular=False)
        out = self._logpdf(dims, X, mean, rowpsd.U, rowpsd.log_pdet, colpsd.U,
                           colpsd.log_pdet)
        return _squeeze_output(out)

    def pdf(self, X, mean=None, rowcov=1, colcov=1):
        """Matrix normal probability density function.

        Parameters
        ----------
        X : array_like
            Quantiles, with the last two axes of `X` denoting the components.
        %(_matnorm_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `X`

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        return np.exp(self.logpdf(X, mean, rowcov, colcov))

    def rvs(self, mean=None, rowcov=1, colcov=1, size=1, random_state=None):
        """Draw random samples from a matrix normal distribution.

        Parameters
        ----------
        %(_matnorm_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `dims`), where `dims` is the
            dimension of the random matrices.

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        size = int(size)
        dims, mean, rowcov, colcov = self._process_parameters(mean, rowcov,
                                                              colcov)
        rowchol = scipy.linalg.cholesky(rowcov, lower=True)
        colchol = scipy.linalg.cholesky(colcov, lower=True)
        random_state = self._get_random_state(random_state)
        # We aren't generating standard normal variates with size=(size,
        # dims[0], dims[1]) directly to ensure random variates remain backwards
        # compatible. See https://github.com/scipy/scipy/pull/12312 for more
        # details.
        std_norm = random_state.standard_normal(
            size=(dims[1], size, dims[0])
        ).transpose(1, 2, 0)
        out = mean + np.einsum('jp,ipq,kq->ijk',
                               rowchol, std_norm, colchol,
                               optimize=True)
        if size == 1:
            out = out.reshape(mean.shape)
        return out

    def entropy(self, rowcov=1, colcov=1):
        """Log of the matrix normal probability density function.

        Parameters
        ----------
        rowcov : array_like, optional
            Among-row covariance matrix of the distribution (default: `1`)
        colcov : array_like, optional
            Among-column covariance matrix of the distribution (default: `1`)

        Returns
        -------
        entropy : float
            Entropy of the distribution

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        dummy_mean = np.zeros((rowcov.shape[0], colcov.shape[0]))
        dims, _, rowcov, colcov = self._process_parameters(dummy_mean,
                                                           rowcov,
                                                           colcov)
        rowpsd = _PSD(rowcov, allow_singular=False)
        colpsd = _PSD(colcov, allow_singular=False)

        return self._entropy(dims, rowpsd.log_pdet, colpsd.log_pdet)

    def _entropy(self, dims, row_cov_logdet, col_cov_logdet):
        n, p = dims
        return (0.5 * n * p * (1 + _LOG_2PI) + 0.5 * p * row_cov_logdet +
                0.5 * n * col_cov_logdet)


matrix_normal = matrix_normal_gen()


class matrix_normal_frozen(multi_rv_frozen):
    """
    Create a frozen matrix normal distribution.

    Parameters
    ----------
    %(_matnorm_doc_default_callparams)s
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is `None` the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import matrix_normal

    >>> distn = matrix_normal(mean=np.zeros((3,3)))
    >>> X = distn.rvs(); X
    array([[-0.02976962,  0.93339138, -0.09663178],
           [ 0.67405524,  0.28250467, -0.93308929],
           [-0.31144782,  0.74535536,  1.30412916]])
    >>> distn.pdf(X)
    2.5160642368346784e-05
    >>> distn.logpdf(X)
    -10.590229595124615
    """

    def __init__(self, mean=None, rowcov=1, colcov=1, seed=None):
        self._dist = matrix_normal_gen(seed)
        self.dims, self.mean, self.rowcov, self.colcov = \
            self._dist._process_parameters(mean, rowcov, colcov)
        self.rowpsd = _PSD(self.rowcov, allow_singular=False)
        self.colpsd = _PSD(self.colcov, allow_singular=False)

    def logpdf(self, X):
        X = self._dist._process_quantiles(X, self.dims)
        out = self._dist._logpdf(self.dims, X, self.mean, self.rowpsd.U,
                                 self.rowpsd.log_pdet, self.colpsd.U,
                                 self.colpsd.log_pdet)
        return _squeeze_output(out)

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.mean, self.rowcov, self.colcov, size,
                              random_state)

    def entropy(self):
        return self._dist._entropy(self.dims, self.rowpsd.log_pdet,
                                   self.colpsd.log_pdet)


# Set frozen generator docstrings from corresponding docstrings in
# matrix_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'entropy']:
    method = matrix_normal_gen.__dict__[name]
    method_frozen = matrix_normal_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             matnorm_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, matnorm_docdict_params)

_dirichlet_doc_default_callparams = """\
alpha : array_like
    The concentration parameters. The number of entries determines the
    dimensionality of the distribution.
"""
_dirichlet_doc_frozen_callparams = ""

_dirichlet_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

dirichlet_docdict_params = {
    '_dirichlet_doc_default_callparams': _dirichlet_doc_default_callparams,
    '_doc_random_state': _doc_random_state
}

dirichlet_docdict_noparams = {
    '_dirichlet_doc_default_callparams': _dirichlet_doc_frozen_callparams,
    '_doc_random_state': _doc_random_state
}


def _dirichlet_check_parameters(alpha):
    alpha = np.asarray(alpha)
    if np.min(alpha) <= 0:
        raise ValueError("All parameters must be greater than 0")
    elif alpha.ndim != 1:
        raise ValueError("Parameter vector 'a' must be one dimensional, "
                         "but a.shape = {}.".format(alpha.shape))
    return alpha


def _dirichlet_check_input(alpha, x):
    x = np.asarray(x)

    if x.shape[0] + 1 != alpha.shape[0] and x.shape[0] != alpha.shape[0]:
        raise ValueError("Vector 'x' must have either the same number "
                         "of entries as, or one entry fewer than, "
                         "parameter vector 'a', but alpha.shape = {} "
                         "and x.shape = {}.".format(alpha.shape, x.shape))

    if x.shape[0] != alpha.shape[0]:
        xk = np.array([1 - np.sum(x, 0)])
        if xk.ndim == 1:
            x = np.append(x, xk)
        elif xk.ndim == 2:
            x = np.vstack((x, xk))
        else:
            raise ValueError("The input must be one dimensional or a two "
                             "dimensional matrix containing the entries.")

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

    if np.max(x) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")

    # Check x_i > 0 or alpha_i > 1
    xeq0 = (x == 0)
    alphalt1 = (alpha < 1)
    if x.shape != alpha.shape:
        alphalt1 = np.repeat(alphalt1, x.shape[-1], axis=-1).reshape(x.shape)
    chk = np.logical_and(xeq0, alphalt1)

    if np.sum(chk):
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "alpha is less than one.")

    if (np.abs(np.sum(x, 0) - 1.0) > 10e-10).any():
        raise ValueError("The input vector 'x' must lie within the normal "
                         "simplex. but np.sum(x, 0) = %s." % np.sum(x, 0))

    return x


def _lnB(alpha):
    r"""Internal helper function to compute the log of the useful quotient.

    .. math::

        B(\alpha) = \frac{\prod_{i=1}{K}\Gamma(\alpha_i)}
                         {\Gamma\left(\sum_{i=1}^{K} \alpha_i \right)}

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s

    Returns
    -------
    B : scalar
        Helper quotient, internal use only

    """
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))


class dirichlet_gen(multi_rv_generic):
    r"""A Dirichlet random variable.

    The ``alpha`` keyword specifies the concentration parameters of the
    distribution.

    .. versionadded:: 0.15.0

    Methods
    -------
    pdf(x, alpha)
        Probability density function.
    logpdf(x, alpha)
        Log of the probability density function.
    rvs(alpha, size=1, random_state=None)
        Draw random samples from a Dirichlet distribution.
    mean(alpha)
        The mean of the Dirichlet distribution
    var(alpha)
        The variance of the Dirichlet distribution
    entropy(alpha)
        Compute the differential entropy of the Dirichlet distribution.

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    Each :math:`\alpha` entry must be positive. The distribution has only
    support on the simplex defined by

    .. math::
        \sum_{i=1}^{K} x_i = 1

    where :math:`0 < x_i < 1`.

    If the quantiles don't lie within the simplex, a ValueError is raised.

    The probability density function for `dirichlet` is

    .. math::

        f(x) = \frac{1}{\mathrm{B}(\boldsymbol\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}

    where

    .. math::

        \mathrm{B}(\boldsymbol\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}
                                     {\Gamma\bigl(\sum_{i=1}^K \alpha_i\bigr)}

    and :math:`\boldsymbol\alpha=(\alpha_1,\ldots,\alpha_K)`, the
    concentration parameters and :math:`K` is the dimension of the space
    where :math:`x` takes values.

    Note that the `dirichlet` interface is somewhat inconsistent.
    The array returned by the rvs function is transposed
    with respect to the format expected by the pdf and logpdf.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import dirichlet

    Generate a dirichlet random variable

    >>> quantiles = np.array([0.2, 0.2, 0.6])  # specify quantiles
    >>> alpha = np.array([0.4, 5, 15])  # specify concentration parameters
    >>> dirichlet.pdf(quantiles, alpha)
    0.2843831684937255

    The same PDF but following a log scale

    >>> dirichlet.logpdf(quantiles, alpha)
    -1.2574327653159187

    Once we specify the dirichlet distribution
    we can then calculate quantities of interest

    >>> dirichlet.mean(alpha)  # get the mean of the distribution
    array([0.01960784, 0.24509804, 0.73529412])
    >>> dirichlet.var(alpha) # get variance
    array([0.00089829, 0.00864603, 0.00909517])
    >>> dirichlet.entropy(alpha)  # calculate the differential entropy
    -4.3280162474082715

    We can also return random samples from the distribution

    >>> dirichlet.rvs(alpha, size=1, random_state=1)
    array([[0.00766178, 0.24670518, 0.74563305]])
    >>> dirichlet.rvs(alpha, size=2, random_state=2)
    array([[0.01639427, 0.1292273 , 0.85437844],
           [0.00156917, 0.19033695, 0.80809388]])

    Alternatively, the object may be called (as a function) to fix
    concentration parameters, returning a "frozen" Dirichlet
    random variable:

    >>> rv = dirichlet(alpha)
    >>> # Frozen object with the same methods but holding the given
    >>> # concentration parameters fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)

    def __call__(self, alpha, seed=None):
        return dirichlet_frozen(alpha, seed=seed)

    def _logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        %(_dirichlet_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        lnB = _lnB(alpha)
        return - lnB + np.sum((xlogy(alpha - 1, x.T)).T, 0)

    def logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`.

        """
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)

        out = self._logpdf(x, alpha)
        return _squeeze_output(out)

    def pdf(self, x, alpha):
        """The Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            The probability density function evaluated at `x`.

        """
        alpha = _dirichlet_check_parameters(alpha)
        x = _dirichlet_check_input(alpha, x)

        out = np.exp(self._logpdf(x, alpha))
        return _squeeze_output(out)

    def mean(self, alpha):
        """Mean of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        mu : ndarray or scalar
            Mean of the Dirichlet distribution.

        """
        alpha = _dirichlet_check_parameters(alpha)

        out = alpha / (np.sum(alpha))
        return _squeeze_output(out)

    def var(self, alpha):
        """Variance of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        v : ndarray or scalar
            Variance of the Dirichlet distribution.

        """

        alpha = _dirichlet_check_parameters(alpha)

        alpha0 = np.sum(alpha)
        out = (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))
        return _squeeze_output(out)

    def entropy(self, alpha):
        """
        Differential entropy of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the Dirichlet distribution

        """

        alpha = _dirichlet_check_parameters(alpha)

        alpha0 = np.sum(alpha)
        lnB = _lnB(alpha)
        K = alpha.shape[0]

        out = lnB + (alpha0 - K) * scipy.special.psi(alpha0) - np.sum(
            (alpha - 1) * scipy.special.psi(alpha))
        return _squeeze_output(out)

    def rvs(self, alpha, size=1, random_state=None):
        """
        Draw random samples from a Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
        size : int, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        """
        alpha = _dirichlet_check_parameters(alpha)
        random_state = self._get_random_state(random_state)
        return random_state.dirichlet(alpha, size=size)


dirichlet = dirichlet_gen()


class dirichlet_frozen(multi_rv_frozen):
    def __init__(self, alpha, seed=None):
        self.alpha = _dirichlet_check_parameters(alpha)
        self._dist = dirichlet_gen(seed)

    def logpdf(self, x):
        return self._dist.logpdf(x, self.alpha)

    def pdf(self, x):
        return self._dist.pdf(x, self.alpha)

    def mean(self):
        return self._dist.mean(self.alpha)

    def var(self):
        return self._dist.var(self.alpha)

    def entropy(self):
        return self._dist.entropy(self.alpha)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.alpha, size, random_state)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'mean', 'var', 'entropy']:
    method = dirichlet_gen.__dict__[name]
    method_frozen = dirichlet_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, dirichlet_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, dirichlet_docdict_params)


_wishart_doc_default_callparams = """\
df : int
    Degrees of freedom, must be greater than or equal to dimension of the
    scale matrix
scale : array_like
    Symmetric positive definite scale matrix of the distribution
"""

_wishart_doc_callparams_note = ""

_wishart_doc_frozen_callparams = ""

_wishart_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

wishart_docdict_params = {
    '_doc_default_callparams': _wishart_doc_default_callparams,
    '_doc_callparams_note': _wishart_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

wishart_docdict_noparams = {
    '_doc_default_callparams': _wishart_doc_frozen_callparams,
    '_doc_callparams_note': _wishart_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class wishart_gen(multi_rv_generic):
    r"""A Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal precision matrix (the inverse of the covariance
    matrix). These arguments must satisfy the relationship
    ``df > scale.ndim - 1``, but see notes on using the `rvs` method with
    ``df < scale.ndim``.

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1, random_state=None)
        Draw random samples from a Wishart distribution.
    entropy()
        Compute the differential entropy of the Wishart distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Raises
    ------
    scipy.linalg.LinAlgError
        If the scale matrix `scale` is not positive definite.

    See Also
    --------
    invwishart, chi2

    Notes
    -----
    %(_doc_callparams_note)s

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported. Symmetry is not checked; only the lower triangular
    portion is used.

    The Wishart distribution is often denoted

    .. math::

        W_p(\nu, \Sigma)

    where :math:`\nu` is the degrees of freedom and :math:`\Sigma` is the
    :math:`p \times p` scale matrix.

    The probability density function for `wishart` has support over positive
    definite matrices :math:`S`; if :math:`S \sim W_p(\nu, \Sigma)`, then
    its PDF is given by:

    .. math::

        f(S) = \frac{|S|^{\frac{\nu - p - 1}{2}}}{2^{ \frac{\nu p}{2} }
               |\Sigma|^\frac{\nu}{2} \Gamma_p \left ( \frac{\nu}{2} \right )}
               \exp\left( -tr(\Sigma^{-1} S) / 2 \right)

    If :math:`S \sim W_p(\nu, \Sigma)` (Wishart) then
    :math:`S^{-1} \sim W_p^{-1}(\nu, \Sigma^{-1})` (inverse Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the Wishart
    distribution :math:`W_1(\nu, 1)` collapses to the :math:`\chi^2(\nu)`
    distribution.

    The algorithm [2]_ implemented by the `rvs` method may
    produce numerically singular matrices with :math:`p - 1 < \nu < p`; the
    user may wish to check for this condition and generate replacement samples
    as necessary.


    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] M.L. Eaton, "Multivariate Statistics: A Vector Space Approach",
           Wiley, 1983.
    .. [2] W.B. Smith and R.R. Hocking, "Algorithm AS 53: Wishart Variate
           Generator", Applied Statistics, vol. 21, pp. 341-345, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import wishart, chi2
    >>> x = np.linspace(1e-5, 8, 100)
    >>> w = wishart.pdf(x, df=3, scale=1); w[:5]
    array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
    >>> c = chi2.pdf(x, 3); c[:5]
    array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
    >>> plt.plot(x, w)
    >>> plt.show()

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" Wishart random
    variable:

    >>> rv = wishart(df=1, scale=1)
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df=None, scale=None, seed=None):
        """Create a frozen Wishart distribution.

        See `wishart_frozen` for more information.
        """
        return wishart_frozen(df, scale, seed)

    def _process_parameters(self, df, scale):
        if scale is None:
            scale = 1.0
        scale = np.asarray(scale, dtype=float)

        if scale.ndim == 0:
            scale = scale[np.newaxis, np.newaxis]
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and not scale.shape[0] == scale.shape[1]:
            raise ValueError("Array 'scale' must be square if it is two"
                             " dimensional, but scale.scale = %s."
                             % str(scale.shape))
        elif scale.ndim > 2:
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)

        dim = scale.shape[0]

        if df is None:
            df = dim
        elif not np.isscalar(df):
            raise ValueError("Degrees of freedom must be a scalar.")
        elif df <= dim - 1:
            raise ValueError("Degrees of freedom must be greater than the "
                             "dimension of scale matrix minus 1.")

        return dim, df, scale

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x * np.eye(dim)[:, :, np.newaxis]
        if x.ndim == 1:
            if dim == 1:
                x = x[np.newaxis, np.newaxis, :]
            else:
                x = np.diag(x)[:, :, np.newaxis]
        elif x.ndim == 2:
            if not x.shape[0] == x.shape[1]:
                raise ValueError("Quantiles must be square if they are two"
                                 " dimensional, but x.shape = %s."
                                 % str(x.shape))
            x = x[:, :, np.newaxis]
        elif x.ndim == 3:
            if not x.shape[0] == x.shape[1]:
                raise ValueError("Quantiles must be square in the first two"
                                 " dimensions if they are three dimensional"
                                 ", but x.shape = %s." % str(x.shape))
        elif x.ndim > 3:
            raise ValueError("Quantiles must be at most two-dimensional with"
                             " an additional dimension for multiple"
                             "components, but x.ndim = %d" % x.ndim)

        # Now we have 3-dim array; should have shape [dim, dim, *]
        if not x.shape[0:2] == (dim, dim):
            raise ValueError('Quantiles have incompatible dimensions: should'
                             ' be {}, got {}.'.format((dim, dim), x.shape[0:2]))

        return x

    def _process_size(self, size):
        size = np.asarray(size)

        if size.ndim == 0:
            size = size[np.newaxis]
        elif size.ndim > 1:
            raise ValueError('Size must be an integer or tuple of integers;'
                             ' thus must have dimension <= 1.'
                             ' Got size.ndim = %s' % str(tuple(size)))
        n = size.prod()
        shape = tuple(size)

        return n, shape

    def _logpdf(self, x, dim, df, scale, log_det_scale, C):
        """Log of the Wishart probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # log determinant of x
        # Note: x has components along the last axis, so that x.T has
        # components alone the 0-th axis. Then since det(A) = det(A'), this
        # gives us a 1-dim vector of determinants

        # Retrieve tr(scale^{-1} x)
        log_det_x = np.empty(x.shape[-1])
        scale_inv_x = np.empty(x.shape)
        tr_scale_inv_x = np.empty(x.shape[-1])
        for i in range(x.shape[-1]):
            _, log_det_x[i] = self._cholesky_logdet(x[:, :, i])
            scale_inv_x[:, :, i] = scipy.linalg.cho_solve((C, True), x[:, :, i])
            tr_scale_inv_x[i] = scale_inv_x[:, :, i].trace()

        # Log PDF
        out = ((0.5 * (df - dim - 1) * log_det_x - 0.5 * tr_scale_inv_x) -
               (0.5 * df * dim * _LOG_2 + 0.5 * df * log_det_scale +
                multigammaln(0.5*df, dim)))

        return out

    def logpdf(self, x, df, scale):
        """Log of the Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)

        # Cholesky decomposition of scale, get log(det(scale))
        C, log_det_scale = self._cholesky_logdet(scale)

        out = self._logpdf(x, dim, df, scale, log_det_scale, C)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """Mean of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.

        """
        return df * scale

    def mean(self, df, scale):
        """Mean of the Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out)

    def _mode(self, dim, df, scale):
        """Mode of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.

        """
        if df >= dim + 1:
            out = (df-dim-1) * scale
        else:
            out = None
        return out

    def mode(self, df, scale):
        """Mode of the Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mode : float or None
            The Mode of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _var(self, dim, df, scale):
        """Variance of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.

        """
        var = scale**2
        diag = scale.diagonal()  # 1 x dim array
        var += np.outer(diag, diag)
        var *= df
        return var

    def var(self, df, scale):
        """Variance of the Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        var : float
            The variance of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._var(dim, df, scale)
        return _squeeze_output(out)

    def _standard_rvs(self, n, shape, dim, df, random_state):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        # Random normal variates for off-diagonal elements
        n_tril = dim * (dim-1) // 2
        covariances = random_state.normal(
            size=n*n_tril).reshape(shape+(n_tril,))

        # Random chi-square variates for diagonal elements
        variances = (np.r_[[random_state.chisquare(df-(i+1)+1, size=n)**0.5
                            for i in range(dim)]].reshape((dim,) +
                                                          shape[::-1]).T)

        # Create the A matri(ces) - lower triangular
        A = np.zeros(shape + (dim, dim))

        # Input the covariances
        size_idx = tuple([slice(None, None, None)]*len(shape))
        tril_idx = np.tril_indices(dim, k=-1)
        A[size_idx + tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(dim)
        A[size_idx + diag_idx] = variances

        return A

    def _rvs(self, n, shape, dim, df, C, random_state):
        """Draw random samples from a Wishart distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triangular.
        %(_doc_random_state)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        # Calculate the matrices A, which are actually lower triangular
        # Cholesky factorizations of a matrix B such that B ~ W(df, I)
        A = self._standard_rvs(n, shape, dim, df, random_state)

        # Calculate SA = C A A' C', where SA ~ W(df, scale)
        # Note: this is the product of a (lower) (lower) (lower)' (lower)'
        #       or, denoting B = AA', it is C B C' where C is the lower
        #       triangular Cholesky factorization of the scale matrix.
        #       this appears to conflict with the instructions in [1]_, which
        #       suggest that it should be D' B D where D is the lower
        #       triangular factorization of the scale matrix. However, it is
        #       meant to refer to the Bartlett (1933) representation of a
        #       Wishart random variate as L A A' L' where L is lower triangular
        #       so it appears that understanding D' to be upper triangular
        #       is either a typo in or misreading of [1]_.
        for index in np.ndindex(shape):
            CA = np.dot(C, A[index])
            A[index] = np.dot(CA, CA.T)

        return A

    def rvs(self, df, scale, size=1, random_state=None):
        """Draw random samples from a Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (`dim`, `dim), where `dim` is
            the dimension of the scale matrix.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # Cholesky decomposition of scale
        C = scipy.linalg.cholesky(scale, lower=True)

        out = self._rvs(n, shape, dim, df, C, random_state)

        return _squeeze_output(out)

    def _entropy(self, dim, df, log_det_scale):
        """Compute the differential entropy of the Wishart.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        log_det_scale : float
            Logarithm of the determinant of the scale matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'entropy' instead.

        """
        return (
            0.5 * (dim+1) * log_det_scale +
            0.5 * dim * (dim+1) * _LOG_2 +
            multigammaln(0.5*df, dim) -
            0.5 * (df - dim - 1) * np.sum(
                [psi(0.5*(df + 1 - (i+1))) for i in range(dim)]
            ) +
            0.5 * df * dim
        )

    def entropy(self, df, scale):
        """Compute the differential entropy of the Wishart.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the Wishart distribution

        Notes
        -----
        %(_doc_callparams_note)s

        """
        dim, df, scale = self._process_parameters(df, scale)
        _, log_det_scale = self._cholesky_logdet(scale)
        return self._entropy(dim, df, log_det_scale)

    def _cholesky_logdet(self, scale):
        """Compute Cholesky decomposition and determine (log(det(scale)).

        Parameters
        ----------
        scale : ndarray
            Scale matrix.

        Returns
        -------
        c_decomp : ndarray
            The Cholesky decomposition of `scale`.
        logdet : scalar
            The log of the determinant of `scale`.

        Notes
        -----
        This computation of ``logdet`` is equivalent to
        ``np.linalg.slogdet(scale)``.  It is ~2x faster though.

        """
        c_decomp = scipy.linalg.cholesky(scale, lower=True)
        logdet = 2 * np.sum(np.log(c_decomp.diagonal()))
        return c_decomp, logdet


wishart = wishart_gen()


class wishart_frozen(multi_rv_frozen):
    """Create a frozen Wishart distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution
    scale : array_like
        Scale matrix of the distribution
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """
    def __init__(self, df, scale, seed=None):
        self._dist = wishart_gen(seed)
        self.dim, self.df, self.scale = self._dist._process_parameters(
            df, scale)
        self.C, self.log_det_scale = self._dist._cholesky_logdet(self.scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)

        out = self._dist._logpdf(x, self.dim, self.df, self.scale,
                                 self.log_det_scale, self.C)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        out = self._dist._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def mode(self):
        out = self._dist._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def var(self):
        out = self._dist._var(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        n, shape = self._dist._process_size(size)
        out = self._dist._rvs(n, shape, self.dim, self.df,
                              self.C, random_state)
        return _squeeze_output(out)

    def entropy(self):
        return self._dist._entropy(self.dim, self.df, self.log_det_scale)


# Set frozen generator docstrings from corresponding docstrings in
# Wishart and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs', 'entropy']:
    method = wishart_gen.__dict__[name]
    method_frozen = wishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)


def _cho_inv_batch(a, check_finite=True):
    """
    Invert the matrices a_i, using a Cholesky factorization of A, where
    a_i resides in the last two dimensions of a and the other indices describe
    the index i.

    Overwrites the data in a.

    Parameters
    ----------
    a : array
        Array of matrices to invert, where the matrices themselves are stored
        in the last two dimensions.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Array of inverses of the matrices ``a_i``.

    See Also
    --------
    scipy.linalg.cholesky : Cholesky factorization of a matrix

    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) < 2 or a1.shape[-2] != a1.shape[-1]:
        raise ValueError('expected square matrix in last two dimensions')

    potrf, potri = get_lapack_funcs(('potrf', 'potri'), (a1,))

    triu_rows, triu_cols = np.triu_indices(a.shape[-2], k=1)
    for index in np.ndindex(a1.shape[:-2]):

        # Cholesky decomposition
        a1[index], info = potrf(a1[index], lower=True, overwrite_a=False,
                                clean=False)
        if info > 0:
            raise LinAlgError("%d-th leading minor not positive definite"
                              % info)
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal'
                             ' potrf' % -info)
        # Inversion
        a1[index], info = potri(a1[index], lower=True, overwrite_c=False)
        if info > 0:
            raise LinAlgError("the inverse could not be computed")
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal'
                             ' potrf' % -info)

        # Make symmetric (dpotri only fills in the lower triangle)
        a1[index][triu_rows, triu_cols] = a1[index][triu_cols, triu_rows]

    return a1


class invwishart_gen(wishart_gen):
    r"""An inverse Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal covariance matrix.

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1, random_state=None)
        Draw random samples from an inverse Wishart distribution.
    entropy(df, scale)
        Differential entropy of the distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Raises
    ------
    scipy.linalg.LinAlgError
        If the scale matrix `scale` is not positive definite.

    See Also
    --------
    wishart

    Notes
    -----
    %(_doc_callparams_note)s

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported. Symmetry is not checked; only the lower triangular
    portion is used.

    The inverse Wishart distribution is often denoted

    .. math::

        W_p^{-1}(\nu, \Psi)

    where :math:`\nu` is the degrees of freedom and :math:`\Psi` is the
    :math:`p \times p` scale matrix.

    The probability density function for `invwishart` has support over positive
    definite matrices :math:`S`; if :math:`S \sim W^{-1}_p(\nu, \Sigma)`,
    then its PDF is given by:

    .. math::

        f(S) = \frac{|\Sigma|^\frac{\nu}{2}}{2^{ \frac{\nu p}{2} }
               |S|^{\frac{\nu + p + 1}{2}} \Gamma_p \left(\frac{\nu}{2} \right)}
               \exp\left( -tr(\Sigma S^{-1}) / 2 \right)

    If :math:`S \sim W_p^{-1}(\nu, \Psi)` (inverse Wishart) then
    :math:`S^{-1} \sim W_p(\nu, \Psi^{-1})` (Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the inverse
    Wishart distribution :math:`W_1(\nu, 1)` collapses to the
    inverse Gamma distribution with parameters shape = :math:`\frac{\nu}{2}`
    and scale = :math:`\frac{1}{2}`.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] M.L. Eaton, "Multivariate Statistics: A Vector Space Approach",
           Wiley, 1983.
    .. [2] M.C. Jones, "Generating Inverse Wishart Matrices", Communications
           in Statistics - Simulation and Computation, vol. 14.2, pp.511-514,
           1985.
    .. [3] Gupta, M. and Srivastava, S. "Parametric Bayesian Estimation of
           Differential Entropy and Relative Entropy". Entropy 12, 818 - 843.
           2010.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import invwishart, invgamma
    >>> x = np.linspace(0.01, 1, 100)
    >>> iw = invwishart.pdf(x, df=6, scale=1)
    >>> iw[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> ig = invgamma.pdf(x, 6/2., scale=1./2)
    >>> ig[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> plt.plot(x, iw)
    >>> plt.show()

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" inverse Wishart
    random variable:

    >>> rv = invwishart(df=1, scale=1)
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df=None, scale=None, seed=None):
        """Create a frozen inverse Wishart distribution.

        See `invwishart_frozen` for more information.

        """
        return invwishart_frozen(df, scale, seed)

    def _logpdf(self, x, dim, df, scale, log_det_scale):
        """Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        log_det_x = np.empty(x.shape[-1])
        x_inv = np.copy(x).T
        if dim > 1:
            _cho_inv_batch(x_inv)  # works in-place
        else:
            x_inv = 1./x_inv
        tr_scale_x_inv = np.empty(x.shape[-1])

        for i in range(x.shape[-1]):
            C, lower = scipy.linalg.cho_factor(x[:, :, i], lower=True)

            log_det_x[i] = 2 * np.sum(np.log(C.diagonal()))

            tr_scale_x_inv[i] = np.dot(scale, x_inv[i]).trace()

        # Log PDF
        out = ((0.5 * df * log_det_scale - 0.5 * tr_scale_x_inv) -
               (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) -
               multigammaln(0.5*df, dim))

        return out

    def logpdf(self, x, df, scale):
        """Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)
        _, log_det_scale = self._cholesky_logdet(scale)
        out = self._logpdf(x, dim, df, scale, log_det_scale)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """Inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """Mean of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.

        """
        if df > dim + 1:
            out = scale / (df - dim - 1)
        else:
            out = None
        return out

    def mean(self, df, scale):
        """Mean of the inverse Wishart distribution.

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus one.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float or None
            The mean of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _mode(self, dim, df, scale):
        """Mode of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.

        """
        return scale / (df + dim + 1)

    def mode(self, df, scale):
        """Mode of the inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mode : float
            The Mode of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out)

    def _var(self, dim, df, scale):
        """Variance of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.

        """
        if df > dim + 3:
            var = (df - dim + 1) * scale**2
            diag = scale.diagonal()  # 1 x dim array
            var += (df - dim - 1) * np.outer(diag, diag)
            var /= (df - dim) * (df - dim - 1)**2 * (df - dim - 3)
        else:
            var = None
        return var

    def var(self, df, scale):
        """Variance of the inverse Wishart distribution.

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus three.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        var : float
            The variance of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._var(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _rvs(self, n, shape, dim, df, C, random_state):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.
        %(_doc_random_state)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        # Get random draws A such that A ~ W(df, I)
        A = super()._standard_rvs(n, shape, dim, df, random_state)

        # Calculate SA = (CA)'^{-1} (CA)^{-1} ~ iW(df, scale)
        eye = np.eye(dim)
        trtrs = get_lapack_funcs(('trtrs'), (A,))

        for index in np.ndindex(A.shape[:-2]):
            # Calculate CA
            CA = np.dot(C, A[index])
            # Get (C A)^{-1} via triangular solver
            if dim > 1:
                CA, info = trtrs(CA, eye, lower=True)
                if info > 0:
                    raise LinAlgError("Singular matrix.")
                if info < 0:
                    raise ValueError('Illegal value in %d-th argument of'
                                     ' internal trtrs' % -info)
            else:
                CA = 1. / CA
            # Get SA
            A[index] = np.dot(CA.T, CA)

        return A

    def rvs(self, df, scale, size=1, random_state=None):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (`dim`, `dim), where `dim` is
            the dimension of the scale matrix.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # Invert the scale
        eye = np.eye(dim)
        L, lower = scipy.linalg.cho_factor(scale, lower=True)
        inv_scale = scipy.linalg.cho_solve((L, lower), eye)
        # Cholesky decomposition of inverted scale
        C = scipy.linalg.cholesky(inv_scale, lower=True)

        out = self._rvs(n, shape, dim, df, C, random_state)

        return _squeeze_output(out)

    def _entropy(self, dim, df, log_det_scale):
        # reference: eq. (17) from ref. 3
        psi_eval_points = [0.5 * (df - dim + i) for i in range(1, dim + 1)]
        psi_eval_points = np.asarray(psi_eval_points)
        return multigammaln(0.5 * df, dim) + 0.5 * dim * df + \
            0.5 * (dim + 1) * (log_det_scale - _LOG_2) - \
            0.5 * (df + dim + 1) * \
            psi(psi_eval_points, out=psi_eval_points).sum()

    def entropy(self, df, scale):
        dim, df, scale = self._process_parameters(df, scale)
        _, log_det_scale = self._cholesky_logdet(scale)
        return self._entropy(dim, df, log_det_scale)


invwishart = invwishart_gen()


class invwishart_frozen(multi_rv_frozen):
    def __init__(self, df, scale, seed=None):
        """Create a frozen inverse Wishart distribution.

        Parameters
        ----------
        df : array_like
            Degrees of freedom of the distribution
        scale : array_like
            Scale matrix of the distribution
        seed : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.

        """
        self._dist = invwishart_gen(seed)
        self.dim, self.df, self.scale = self._dist._process_parameters(
            df, scale
        )

        # Get the determinant via Cholesky factorization
        C, lower = scipy.linalg.cho_factor(self.scale, lower=True)
        self.log_det_scale = 2 * np.sum(np.log(C.diagonal()))

        # Get the inverse using the Cholesky factorization
        eye = np.eye(self.dim)
        self.inv_scale = scipy.linalg.cho_solve((C, lower), eye)

        # Get the Cholesky factorization of the inverse scale
        self.C = scipy.linalg.cholesky(self.inv_scale, lower=True)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.dim, self.df, self.scale,
                                 self.log_det_scale)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        out = self._dist._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def mode(self):
        out = self._dist._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def var(self):
        out = self._dist._var(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def rvs(self, size=1, random_state=None):
        n, shape = self._dist._process_size(size)

        out = self._dist._rvs(n, shape, self.dim, self.df,
                              self.C, random_state)

        return _squeeze_output(out)

    def entropy(self):
        return self._dist._entropy(self.dim, self.df, self.log_det_scale)


# Set frozen generator docstrings from corresponding docstrings in
# inverse Wishart and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs']:
    method = invwishart_gen.__dict__[name]
    method_frozen = wishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)

_multinomial_doc_default_callparams = """\
n : int
    Number of trials
p : array_like
    Probability of a trial falling into each category; should sum to 1
"""

_multinomial_doc_callparams_note = """\
`n` should be a nonnegative integer. Each element of `p` should be in the
interval :math:`[0,1]` and the elements should sum to 1. If they do not sum to
1, the last element of the `p` array is not used and is replaced with the
remaining probability left over from the earlier elements.
"""

_multinomial_doc_frozen_callparams = ""

_multinomial_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

multinomial_docdict_params = {
    '_doc_default_callparams': _multinomial_doc_default_callparams,
    '_doc_callparams_note': _multinomial_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

multinomial_docdict_noparams = {
    '_doc_default_callparams': _multinomial_doc_frozen_callparams,
    '_doc_callparams_note': _multinomial_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multinomial_gen(multi_rv_generic):
    r"""A multinomial random variable.

    Methods
    -------
    pmf(x, n, p)
        Probability mass function.
    logpmf(x, n, p)
        Log of the probability mass function.
    rvs(n, p, size=1, random_state=None)
        Draw random samples from a multinomial distribution.
    entropy(n, p)
        Compute the entropy of the multinomial distribution.
    cov(n, p)
        Compute the covariance matrix of the multinomial distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_callparams_note)s

    The probability mass function for `multinomial` is

    .. math::

        f(x) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k},

    supported on :math:`x=(x_1, \ldots, x_k)` where each :math:`x_i` is a
    nonnegative integer and their sum is :math:`n`.

    .. versionadded:: 0.19.0

    Examples
    --------

    >>> from scipy.stats import multinomial
    >>> rv = multinomial(8, [0.3, 0.2, 0.5])
    >>> rv.pmf([1, 3, 4])
    0.042000000000000072

    The multinomial distribution for :math:`k=2` is identical to the
    corresponding binomial distribution (tiny numerical differences
    notwithstanding):

    >>> from scipy.stats import binom
    >>> multinomial.pmf([3, 4], n=7, p=[0.4, 0.6])
    0.29030399999999973
    >>> binom.pmf(3, 7, 0.4)
    0.29030400000000012

    The functions ``pmf``, ``logpmf``, ``entropy``, and ``cov`` support
    broadcasting, under the convention that the vector parameters (``x`` and
    ``p``) are interpreted as if each row along the last axis is a single
    object. For instance:

    >>> multinomial.pmf([[3, 4], [3, 5]], n=[7, 8], p=[.3, .7])
    array([0.2268945,  0.25412184])

    Here, ``x.shape == (2, 2)``, ``n.shape == (2,)``, and ``p.shape == (2,)``,
    but following the rules mentioned above they behave as if the rows
    ``[3, 4]`` and ``[3, 5]`` in ``x`` and ``[.3, .7]`` in ``p`` were a single
    object, and as if we had ``x.shape = (2,)``, ``n.shape = (2,)``, and
    ``p.shape = ()``. To obtain the individual elements without broadcasting,
    we would do this:

    >>> multinomial.pmf([3, 4], n=7, p=[.3, .7])
    0.2268945
    >>> multinomial.pmf([3, 5], 8, p=[.3, .7])
    0.25412184

    This broadcasting also works for ``cov``, where the output objects are
    square matrices of size ``p.shape[-1]``. For example:

    >>> multinomial.cov([4, 5], [[.3, .7], [.4, .6]])
    array([[[ 0.84, -0.84],
            [-0.84,  0.84]],
           [[ 1.2 , -1.2 ],
            [-1.2 ,  1.2 ]]])

    In this example, ``n.shape == (2,)`` and ``p.shape == (2, 2)``, and
    following the rules above, these broadcast as if ``p.shape == (2,)``.
    Thus the result should also be of shape ``(2,)``, but since each output is
    a :math:`2 \times 2` matrix, the result in fact has shape ``(2, 2, 2)``,
    where ``result[0]`` is equal to ``multinomial.cov(n=4, p=[.3, .7])`` and
    ``result[1]`` is equal to ``multinomial.cov(n=5, p=[.4, .6])``.

    Alternatively, the object may be called (as a function) to fix the `n` and
    `p` parameters, returning a "frozen" multinomial random variable:

    >>> rv = multinomial(n=7, p=[.3, .7])
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    See also
    --------
    scipy.stats.binom : The binomial distribution.
    numpy.random.Generator.multinomial : Sampling from the multinomial distribution.
    scipy.stats.multivariate_hypergeom :
        The multivariate hypergeometric distribution.
    """  # noqa: E501

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = \
            doccer.docformat(self.__doc__, multinomial_docdict_params)

    def __call__(self, n, p, seed=None):
        """Create a frozen multinomial distribution.

        See `multinomial_frozen` for more information.
        """
        return multinomial_frozen(n, p, seed)

    def _process_parameters(self, n, p, eps=1e-15):
        """Returns: n_, p_, npcond.

        n_ and p_ are arrays of the correct shape; npcond is a boolean array
        flagging values out of the domain.
        """
        p = np.array(p, dtype=np.float64, copy=True)
        p_adjusted = 1. - p[..., :-1].sum(axis=-1)
        i_adjusted = np.abs(p_adjusted) > eps
        p[i_adjusted, -1] = p_adjusted[i_adjusted]

        # true for bad p
        pcond = np.any(p < 0, axis=-1)
        pcond |= np.any(p > 1, axis=-1)

        n = np.array(n, dtype=np.int_, copy=True)

        # true for bad n
        ncond = n < 0

        return n, p, ncond | pcond

    def _process_quantiles(self, x, n, p):
        """Returns: x_, xcond.

        x_ is an int array; xcond is a boolean array flagging values out of the
        domain.
        """
        xx = np.asarray(x, dtype=np.int_)

        if xx.ndim == 0:
            raise ValueError("x must be an array.")

        if xx.size != 0 and not xx.shape[-1] == p.shape[-1]:
            raise ValueError("Size of each quantile should be size of p: "
                             "received %d, but expected %d." %
                             (xx.shape[-1], p.shape[-1]))

        # true for x out of the domain
        cond = np.any(xx != x, axis=-1)
        cond |= np.any(xx < 0, axis=-1)
        cond = cond | (np.sum(xx, axis=-1) != n)

        return xx, cond

    def _checkresult(self, result, cond, bad_value):
        result = np.asarray(result)

        if cond.ndim != 0:
            result[cond] = bad_value
        elif cond:
            if result.ndim == 0:
                return bad_value
            result[...] = bad_value
        return result

    def _logpmf(self, x, n, p):
        return gammaln(n+1) + np.sum(xlogy(x, p) - gammaln(x+1), axis=-1)

    def logpmf(self, x, n, p):
        """Log of the Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)
        x, xcond = self._process_quantiles(x, n, p)

        result = self._logpmf(x, n, p)

        # replace values for which x was out of the domain; broadcast
        # xcond to the right shape
        xcond_ = xcond | np.zeros(npcond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, -np.inf)

        # replace values bad for n or p; broadcast npcond to the right shape
        npcond_ = npcond | np.zeros(xcond.shape, dtype=np.bool_)
        return self._checkresult(result, npcond_, np.nan)

    def pmf(self, x, n, p):
        """Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        return np.exp(self.logpmf(x, n, p))

    def mean(self, n, p):
        """Mean of the Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        n, p, npcond = self._process_parameters(n, p)
        result = n[..., np.newaxis]*p
        return self._checkresult(result, npcond, np.nan)

    def cov(self, n, p):
        """Covariance matrix of the multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        cov : ndarray
            The covariance matrix of the distribution
        """
        n, p, npcond = self._process_parameters(n, p)

        nn = n[..., np.newaxis, np.newaxis]
        result = nn * np.einsum('...j,...k->...jk', -p, p)

        # change the diagonal
        for i in range(p.shape[-1]):
            result[..., i, i] += n*p[..., i]

        return self._checkresult(result, npcond, np.nan)

    def entropy(self, n, p):
        r"""Compute the entropy of the multinomial distribution.

        The entropy is computed using this expression:

        .. math::

            f(x) = - \log n! - n\sum_{i=1}^k p_i \log p_i +
            \sum_{i=1}^k \sum_{x=0}^n \binom n x p_i^x(1-p_i)^{n-x} \log x!

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the Multinomial distribution

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)

        x = np.r_[1:np.max(n)+1]

        term1 = n*np.sum(entr(p), axis=-1)
        term1 -= gammaln(n+1)

        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,)*new_axes_needed

        term2 = np.sum(binom.pmf(x, n, p)*gammaln(x+1),
                       axis=(-1, -1-new_axes_needed))

        return self._checkresult(term1 + term2, npcond, np.nan)

    def rvs(self, n, p, size=None, random_state=None):
        """Draw random samples from a Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of shape (`size`, `len(p)`)

        Notes
        -----
        %(_doc_callparams_note)s
        """
        n, p, npcond = self._process_parameters(n, p)
        random_state = self._get_random_state(random_state)
        return random_state.multinomial(n, p, size)


multinomial = multinomial_gen()


class multinomial_frozen(multi_rv_frozen):
    r"""Create a frozen Multinomial distribution.

    Parameters
    ----------
    n : int
        number of trials
    p: array_like
        probability of a trial falling into each category; should sum to 1
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    """
    def __init__(self, n, p, seed=None):
        self._dist = multinomial_gen(seed)
        self.n, self.p, self.npcond = self._dist._process_parameters(n, p)

        # monkey patch self._dist
        def _process_parameters(n, p):
            return self.n, self.p, self.npcond

        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, self.n, self.p)

    def pmf(self, x):
        return self._dist.pmf(x, self.n, self.p)

    def mean(self):
        return self._dist.mean(self.n, self.p)

    def cov(self):
        return self._dist.cov(self.n, self.p)

    def entropy(self):
        return self._dist.entropy(self.n, self.p)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.n, self.p, size, random_state)


# Set frozen generator docstrings from corresponding docstrings in
# multinomial and fill in default strings in class docstrings
for name in ['logpmf', 'pmf', 'mean', 'cov', 'rvs']:
    method = multinomial_gen.__dict__[name]
    method_frozen = multinomial_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, multinomial_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__,
                                      multinomial_docdict_params)


class special_ortho_group_gen(multi_rv_generic):
    r"""A Special Orthogonal matrix (SO(N)) random variable.

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(N)) with a determinant of +1.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from SO(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is wrapping the random_rot code from the MDP Toolkit,
    https://github.com/mdp-toolkit/mdp-toolkit

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(N)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization

    See also the similar `ortho_group`. For a random rotation in three
    dimensions, see `scipy.spatial.transform.Rotation.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import special_ortho_group
    >>> x = special_ortho_group.rvs(3)

    >>> np.dot(x, x.T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    >>> import scipy.linalg
    >>> scipy.linalg.det(x)
    1.0

    This generates one random matrix from SO(3). It is orthogonal and
    has a determinant of 1.

    Alternatively, the object may be called (as a function) to fix the `dim`
    parameter, returning a "frozen" special_ortho_group random variable:

    >>> rv = special_ortho_group(5)
    >>> # Frozen object with the same methods but holding the
    >>> # dimension parameter fixed.

    See Also
    --------
    ortho_group, scipy.spatial.transform.Rotation.random

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen SO(N) distribution.

        See `special_ortho_group_frozen` for more information.
        """
        return special_ortho_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
            raise ValueError("""Dimension of rotation must be specified,
                                and must be a scalar greater than 1.""")

        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from SO(N).

        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        random_state = self._get_random_state(random_state)

        size = int(size)
        size = (size,) if size > 1 else ()

        dim = self._process_parameters(dim)

        # H represents a (dim, dim) matrix, while D represents the diagonal of
        # a (dim, dim) diagonal matrix. The algorithm that follows is
        # broadcasted on the leading shape in `size` to vectorize along
        # samples.
        H = np.empty(size + (dim, dim))
        H[..., :, :] = np.eye(dim)
        D = np.empty(size + (dim,))

        for n in range(dim-1):

            # x is a vector with length dim-n, xrow and xcol are views of it as
            # a row vector and column vector respectively. It's important they
            # are views and not copies because we are going to modify x
            # in-place.
            x = random_state.normal(size=size + (dim-n,))
            xrow = x[..., None, :]
            xcol = x[..., :, None]

            # This is the squared norm of x, without vectorization it would be
            # dot(x, x), to have proper broadcasting we use matmul and squeeze
            # out (convert to scalar) the resulting 1x1 matrix
            norm2 = np.matmul(xrow, xcol).squeeze((-2, -1))

            x0 = x[..., 0].copy()
            D[..., n] = np.where(x0 != 0, np.sign(x0), 1)
            x[..., 0] += D[..., n]*np.sqrt(norm2)

            # In renormalizing x we have to append an additional axis with
            # [..., None] to broadcast the scalar against the vector x
            x /= np.sqrt((norm2 - x0**2 + x[..., 0]**2) / 2.)[..., None]

            # Householder transformation, without vectorization the RHS can be
            # written as outer(H @ x, x) (apart from the slicing)
            H[..., :, n:] -= np.matmul(H[..., :, n:], xcol) * xrow

        D[..., -1] = (-1)**(dim-1)*D[..., :-1].prod(axis=-1)

        # Without vectorization this could be written as H = diag(D) @ H,
        # left-multiplication by a diagonal matrix amounts to multiplying each
        # row of H by an element of the diagonal, so we add a dummy axis for
        # the column index
        H *= D[..., :, None]
        return H


special_ortho_group = special_ortho_group_gen()


class special_ortho_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen SO(N) distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import special_ortho_group
        >>> g = special_ortho_group(5)
        >>> x = g.rvs()

        """
        self._dist = special_ortho_group_gen(seed)
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.dim, size, random_state)


class ortho_group_gen(multi_rv_generic):
    r"""An Orthogonal matrix (O(N)) random variable.

    Return a random orthogonal matrix, drawn from the O(N) Haar
    distribution (the only uniform distribution on O(N)).

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from O(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is closely related to `special_ortho_group`.

    Some care is taken to avoid numerical error, as per the paper by Mezzadri.

    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
           compact groups", :arXiv:`math-ph/0609050v2`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> x = ortho_group.rvs(3)

    >>> np.dot(x, x.T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    >>> import scipy.linalg
    >>> np.fabs(scipy.linalg.det(x))
    1.0

    This generates one random matrix from O(3). It is orthogonal and
    has a determinant of +1 or -1.

    Alternatively, the object may be called (as a function) to fix the `dim`
    parameter, returning a "frozen" ortho_group random variable:

    >>> rv = ortho_group(5)
    >>> # Frozen object with the same methods but holding the
    >>> # dimension parameter fixed.

    See Also
    --------
    special_ortho_group
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen O(N) distribution.

        See `ortho_group_frozen` for more information.
        """
        return ortho_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
            raise ValueError("Dimension of rotation must be specified,"
                             "and must be a scalar greater than 1.")

        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from O(N).

        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        random_state = self._get_random_state(random_state)

        size = int(size)
        if size > 1 and NumpyVersion(np.__version__) < '1.22.0':
            return np.array([self.rvs(dim, size=1, random_state=random_state)
                             for i in range(size)])

        dim = self._process_parameters(dim)

        size = (size,) if size > 1 else ()
        z = random_state.normal(size=size + (dim, dim))
        q, r = np.linalg.qr(z)
        # The last two dimensions are the rows and columns of R matrices.
        # Extract the diagonals. Note that this eliminates a dimension.
        d = r.diagonal(offset=0, axis1=-2, axis2=-1)
        # Add back a dimension for proper broadcasting: we're dividing
        # each row of each R matrix by the diagonal of the R matrix.
        q *= (d/abs(d))[..., np.newaxis, :]  # to broadcast properly
        return q


ortho_group = ortho_group_gen()


class ortho_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen O(N) distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import ortho_group
        >>> g = ortho_group(5)
        >>> x = g.rvs()

        """
        self._dist = ortho_group_gen(seed)
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.dim, size, random_state)


class random_correlation_gen(multi_rv_generic):
    r"""A random correlation matrix.

    Return a random correlation matrix, given a vector of eigenvalues.

    The `eigs` keyword specifies the eigenvalues of the correlation matrix,
    and implies the dimension.

    Methods
    -------
    rvs(eigs=None, random_state=None)
        Draw random correlation matrices, all with eigenvalues eigs.

    Parameters
    ----------
    eigs : 1d ndarray
        Eigenvalues of correlation matrix
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance
        then that instance is used.
    tol : float, optional
        Tolerance for input parameter checks
    diag_tol : float, optional
        Tolerance for deviation of the diagonal of the resulting
        matrix. Default: 1e-7

    Raises
    ------
    RuntimeError
        Floating point error prevented generating a valid correlation
        matrix.

    Returns
    -------
    rvs : ndarray or scalar
        Random size N-dimensional matrices, dimension (size, dim, dim),
        each having eigenvalues eigs.

    Notes
    -----

    Generates a random correlation matrix following a numerically stable
    algorithm spelled out by Davies & Higham. This algorithm uses a single O(N)
    similarity transformation to construct a symmetric positive semi-definite
    matrix, and applies a series of Givens rotations to scale it to have ones
    on the diagonal.

    References
    ----------

    .. [1] Davies, Philip I; Higham, Nicholas J; "Numerically stable generation
           of correlation matrices and their factors", BIT 2000, Vol. 40,
           No. 4, pp. 640 651

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import random_correlation
    >>> rng = np.random.default_rng()
    >>> x = random_correlation.rvs((.5, .8, 1.2, 1.5), random_state=rng)
    >>> x
    array([[ 1.        , -0.02423399,  0.03130519,  0.4946965 ],
           [-0.02423399,  1.        ,  0.20334736,  0.04039817],
           [ 0.03130519,  0.20334736,  1.        ,  0.02694275],
           [ 0.4946965 ,  0.04039817,  0.02694275,  1.        ]])
    >>> import scipy.linalg
    >>> e, v = scipy.linalg.eigh(x)
    >>> e
    array([ 0.5,  0.8,  1.2,  1.5])

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, eigs, seed=None, tol=1e-13, diag_tol=1e-7):
        """Create a frozen random correlation matrix.

        See `random_correlation_frozen` for more information.
        """
        return random_correlation_frozen(eigs, seed=seed, tol=tol,
                                         diag_tol=diag_tol)

    def _process_parameters(self, eigs, tol):
        eigs = np.asarray(eigs, dtype=float)
        dim = eigs.size

        if eigs.ndim != 1 or eigs.shape[0] != dim or dim <= 1:
            raise ValueError("Array 'eigs' must be a vector of length "
                             "greater than 1.")

        if np.fabs(np.sum(eigs) - dim) > tol:
            raise ValueError("Sum of eigenvalues must equal dimensionality.")

        for x in eigs:
            if x < -tol:
                raise ValueError("All eigenvalues must be non-negative.")

        return dim, eigs

    def _givens_to_1(self, aii, ajj, aij):
        """Computes a 2x2 Givens matrix to put 1's on the diagonal.

        The input matrix is a 2x2 symmetric matrix M = [ aii aij ; aij ajj ].

        The output matrix g is a 2x2 anti-symmetric matrix of the form
        [ c s ; -s c ];  the elements c and s are returned.

        Applying the output matrix to the input matrix (as b=g.T M g)
        results in a matrix with bii=1, provided tr(M) - det(M) >= 1
        and floating point issues do not occur. Otherwise, some other
        valid rotation is returned. When tr(M)==2, also bjj=1.

        """
        aiid = aii - 1.
        ajjd = ajj - 1.

        if ajjd == 0:
            # ajj==1, so swap aii and ajj to avoid division by zero
            return 0., 1.

        dd = math.sqrt(max(aij**2 - aiid*ajjd, 0))

        # The choice of t should be chosen to avoid cancellation [1]
        t = (aij + math.copysign(dd, aij)) / ajjd
        c = 1. / math.sqrt(1. + t*t)
        if c == 0:
            # Underflow
            s = 1.0
        else:
            s = c*t
        return c, s

    def _to_corr(self, m):
        """
        Given a psd matrix m, rotate to put one's on the diagonal, turning it
        into a correlation matrix.  This also requires the trace equal the
        dimensionality. Note: modifies input matrix
        """
        # Check requirements for in-place Givens
        if not (m.flags.c_contiguous and m.dtype == np.float64 and
                m.shape[0] == m.shape[1]):
            raise ValueError()

        d = m.shape[0]
        for i in range(d-1):
            if m[i, i] == 1:
                continue
            elif m[i, i] > 1:
                for j in range(i+1, d):
                    if m[j, j] < 1:
                        break
            else:
                for j in range(i+1, d):
                    if m[j, j] > 1:
                        break

            c, s = self._givens_to_1(m[i, i], m[j, j], m[i, j])

            # Use BLAS to apply Givens rotations in-place. Equivalent to:
            # g = np.eye(d)
            # g[i, i] = g[j,j] = c
            # g[j, i] = -s; g[i, j] = s
            # m = np.dot(g.T, np.dot(m, g))
            mv = m.ravel()
            drot(mv, mv, c, -s, n=d,
                 offx=i*d, incx=1, offy=j*d, incy=1,
                 overwrite_x=True, overwrite_y=True)
            drot(mv, mv, c, -s, n=d,
                 offx=i, incx=d, offy=j, incy=d,
                 overwrite_x=True, overwrite_y=True)

        return m

    def rvs(self, eigs, random_state=None, tol=1e-13, diag_tol=1e-7):
        """Draw random correlation matrices.

        Parameters
        ----------
        eigs : 1d ndarray
            Eigenvalues of correlation matrix
        tol : float, optional
            Tolerance for input parameter checks
        diag_tol : float, optional
            Tolerance for deviation of the diagonal of the resulting
            matrix. Default: 1e-7

        Raises
        ------
        RuntimeError
            Floating point error prevented generating a valid correlation
            matrix.

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim),
            each having eigenvalues eigs.

        """
        dim, eigs = self._process_parameters(eigs, tol=tol)

        random_state = self._get_random_state(random_state)

        m = ortho_group.rvs(dim, random_state=random_state)
        m = np.dot(np.dot(m, np.diag(eigs)), m.T)  # Set the trace of m
        m = self._to_corr(m)  # Carefully rotate to unit diagonal

        # Check diagonal
        if abs(m.diagonal() - 1).max() > diag_tol:
            raise RuntimeError("Failed to generate a valid correlation matrix")

        return m


random_correlation = random_correlation_gen()


class random_correlation_frozen(multi_rv_frozen):
    def __init__(self, eigs, seed=None, tol=1e-13, diag_tol=1e-7):
        """Create a frozen random correlation matrix distribution.

        Parameters
        ----------
        eigs : 1d ndarray
            Eigenvalues of correlation matrix
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        tol : float, optional
            Tolerance for input parameter checks
        diag_tol : float, optional
            Tolerance for deviation of the diagonal of the resulting
            matrix. Default: 1e-7

        Raises
        ------
        RuntimeError
            Floating point error prevented generating a valid correlation
            matrix.

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim),
            each having eigenvalues eigs.
        """

        self._dist = random_correlation_gen(seed)
        self.tol = tol
        self.diag_tol = diag_tol
        _, self.eigs = self._dist._process_parameters(eigs, tol=self.tol)

    def rvs(self, random_state=None):
        return self._dist.rvs(self.eigs, random_state=random_state,
                              tol=self.tol, diag_tol=self.diag_tol)


class unitary_group_gen(multi_rv_generic):
    r"""A matrix-valued U(N) random variable.

    Return a random unitary matrix.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from U(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is similar to `ortho_group`.

    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
           compact groups", :arXiv:`math-ph/0609050v2`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(3)

    >>> np.dot(x, x.conj().T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    This generates one random matrix from U(3). The dot product confirms that
    it is unitary up to machine precision.

    Alternatively, the object may be called (as a function) to fix the `dim`
    parameter, return a "frozen" unitary_group random variable:

    >>> rv = unitary_group(5)

    See Also
    --------
    ortho_group

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen (U(N)) n-dimensional unitary matrix distribution.

        See `unitary_group_frozen` for more information.
        """
        return unitary_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
            raise ValueError("Dimension of rotation must be specified,"
                             "and must be a scalar greater than 1.")

        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from U(N).

        Parameters
        ----------
        dim : integer
            Dimension of space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        random_state = self._get_random_state(random_state)

        size = int(size)
        if size > 1 and NumpyVersion(np.__version__) < '1.22.0':
            return np.array([self.rvs(dim, size=1, random_state=random_state)
                             for i in range(size)])

        dim = self._process_parameters(dim)

        size = (size,) if size > 1 else ()
        z = 1/math.sqrt(2)*(random_state.normal(size=size + (dim, dim)) +
                            1j*random_state.normal(size=size + (dim, dim)))
        q, r = np.linalg.qr(z)
        # The last two dimensions are the rows and columns of R matrices.
        # Extract the diagonals. Note that this eliminates a dimension.
        d = r.diagonal(offset=0, axis1=-2, axis2=-1)
        # Add back a dimension for proper broadcasting: we're dividing
        # each row of each R matrix by the diagonal of the R matrix.
        q *= (d/abs(d))[..., np.newaxis, :]  # to broadcast properly
        return q


unitary_group = unitary_group_gen()


class unitary_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen (U(N)) n-dimensional unitary matrix distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import unitary_group
        >>> x = unitary_group(3)
        >>> x.rvs()

        """
        self._dist = unitary_group_gen(seed)
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.dim, size, random_state)


_mvt_doc_default_callparams = """\
loc : array_like, optional
    Location of the distribution. (default ``0``)
shape : array_like, optional
    Positive semidefinite matrix of the distribution. (default ``1``)
df : float, optional
    Degrees of freedom of the distribution; must be greater than zero.
    If ``np.inf`` then results are multivariate normal. The default is ``1``.
allow_singular : bool, optional
    Whether to allow a singular matrix. (default ``False``)
"""

_mvt_doc_callparams_note = """\
Setting the parameter `loc` to ``None`` is equivalent to having `loc`
be the zero-vector. The parameter `shape` can be a scalar, in which case
the shape matrix is the identity times that value, a vector of
diagonal entries for the shape matrix, or a two-dimensional array_like.
"""

_mvt_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

mvt_docdict_params = {
    '_mvt_doc_default_callparams': _mvt_doc_default_callparams,
    '_mvt_doc_callparams_note': _mvt_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

mvt_docdict_noparams = {
    '_mvt_doc_default_callparams': "",
    '_mvt_doc_callparams_note': _mvt_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multivariate_t_gen(multi_rv_generic):
    r"""A multivariate t-distributed random variable.

    The `loc` parameter specifies the location. The `shape` parameter specifies
    the positive semidefinite shape matrix. The `df` parameter specifies the
    degrees of freedom.

    In addition to calling the methods below, the object itself may be called
    as a function to fix the location, shape matrix, and degrees of freedom
    parameters, returning a "frozen" multivariate t-distribution random.

    Methods
    -------
    pdf(x, loc=None, shape=1, df=1, allow_singular=False)
        Probability density function.
    logpdf(x, loc=None, shape=1, df=1, allow_singular=False)
        Log of the probability density function.
    cdf(x, loc=None, shape=1, df=1, allow_singular=False, *,
        maxpts=None, lower_limit=None, random_state=None)
        Cumulative distribution function.
    rvs(loc=None, shape=1, df=1, size=1, random_state=None)
        Draw random samples from a multivariate t-distribution.
    entropy(loc=None, shape=1, df=1)
        Differential entropy of a multivariate t-distribution.

    Parameters
    ----------
    %(_mvt_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvt_doc_callparams_note)s
    The matrix `shape` must be a (symmetric) positive semidefinite matrix. The
    determinant and inverse of `shape` are computed as the pseudo-determinant
    and pseudo-inverse, respectively, so that `shape` does not need to have
    full rank.

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = \frac{\Gamma((\nu + p)/2)}{\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}|\Sigma|^{1/2}}
               \left[1 + \frac{1}{\nu} (\mathbf{x} - \boldsymbol{\mu})^{\top}
               \boldsymbol{\Sigma}^{-1}
               (\mathbf{x} - \boldsymbol{\mu}) \right]^{-(\nu + p)/2},

    where :math:`p` is the dimension of :math:`\mathbf{x}`,
    :math:`\boldsymbol{\mu}` is the :math:`p`-dimensional location,
    :math:`\boldsymbol{\Sigma}` the :math:`p \times p`-dimensional shape
    matrix, and :math:`\nu` is the degrees of freedom.

    .. versionadded:: 1.6.0

    References
    ----------
    [1]     Arellano-Valle et al. "Shannon Entropy and Mutual Information for
            Multivariate Skew-Elliptical Distributions". Scandinavian Journal
            of Statistics. Vol. 40, issue 1.

    Examples
    --------
    The object may be called (as a function) to fix the `loc`, `shape`,
    `df`, and `allow_singular` parameters, returning a "frozen"
    multivariate_t random variable:

    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> rv = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2)
    >>> # Frozen object with the same methods but holding the given location,
    >>> # scale, and degrees of freedom fixed.

    Create a contour plot of the PDF.

    >>> import matplotlib.pyplot as plt
    >>> x, y = np.mgrid[-1:3:.01, -2:1.5:.01]
    >>> pos = np.dstack((x, y))
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.set_aspect('equal')
    >>> plt.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.

        """
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        """Create a frozen multivariate t-distribution.

        See `multivariate_t_frozen` for parameters.
        """
        if df == np.inf:
            return multivariate_normal_frozen(mean=loc, cov=shape,
                                              allow_singular=allow_singular,
                                              seed=seed)
        return multivariate_t_frozen(loc=loc, shape=shape, df=df,
                                     allow_singular=allow_singular, seed=seed)

    def pdf(self, x, loc=None, shape=1, df=1, allow_singular=False):
        """Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : Probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.pdf(x, loc, shape, df)
        0.00075713

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape, allow_singular=allow_singular)
        logpdf = self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df,
                              dim, shape_info.rank)
        return np.exp(logpdf)

    def logpdf(self, x, loc=None, shape=1, df=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.logpdf(x, loc, shape, df)
        -7.1859802

        See Also
        --------
        pdf : Probability density function.

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape)
        return self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim,
                            shape_info.rank)

    def _logpdf(self, x, loc, prec_U, log_pdet, df, dim, rank):
        """Utility method `pdf`, `logpdf` for parameters.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability density
            function.
        loc : ndarray
            Location of the distribution.
        prec_U : ndarray
            A decomposition such that `np.dot(prec_U, prec_U.T)` is the inverse
            of the shape matrix.
        log_pdet : float
            Logarithm of the determinant of the shape matrix.
        df : float
            Degrees of freedom of the distribution.
        dim : int
            Dimension of the quantiles x.
        rank : int
            Rank of the shape matrix.

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly; use 'logpdf' instead.

        """
        if df == np.inf:
            return multivariate_normal._logpdf(x, loc, prec_U, log_pdet, rank)

        dev = x - loc
        maha = np.square(np.dot(dev, prec_U)).sum(axis=-1)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1./df) * maha)

        return _squeeze_output(A - B - C - D + E)

    def _cdf(self, x, loc, shape, df, dim, maxpts=None, lower_limit=None,
             random_state=None):

        # All of this -  random state validation, maxpts, apply_along_axis,
        # etc. needs to go in this private method unless we want
        # frozen distribution's `cdf` method to duplicate it or call `cdf`,
        # which would require re-processing parameters
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        if not maxpts:
            maxpts = 1000 * dim

        x = self._process_quantiles(x, dim)
        lower_limit = (np.full(loc.shape, -np.inf)
                       if lower_limit is None else lower_limit)

        # remove the mean
        x, lower_limit = x - loc, lower_limit - loc

        b, a = np.broadcast_arrays(x, lower_limit)
        i_swap = b < a
        signs = (-1)**(i_swap.sum(axis=-1))  # odd # of swaps -> negative
        a, b = a.copy(), b.copy()
        a[i_swap], b[i_swap] = b[i_swap], a[i_swap]
        n = x.shape[-1]
        limits = np.concatenate((a, b), axis=-1)

        def func1d(limits):
            a, b = limits[:n], limits[n:]
            return _qmvt(maxpts, df, shape, a, b, rng)[0]

        res = np.apply_along_axis(func1d, -1, limits) * signs
        # Fixing the output shape for existing distributions is a separate
        # issue. For now, let's keep this consistent with pdf.
        return _squeeze_output(res)

    def cdf(self, x, loc=None, shape=1, df=1, allow_singular=False, *,
            maxpts=None, lower_limit=None, random_state=None):
        """Multivariate t-distribution cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        %(_mvt_doc_default_callparams)s
        maxpts : int, optional
            Maximum number of points to use for integration. The default is
            1000 times the number of dimensions.
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.
        %(_doc_random_state)s

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.cdf(x, loc, shape, df)
        0.64798491

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        shape = _PSD(shape, allow_singular=allow_singular)._M

        return self._cdf(x, loc, shape, df, dim, maxpts,
                         lower_limit, random_state)

    def _entropy(self, dim, df=1, shape=1):
        if df == np.inf:
            return multivariate_normal(None, cov=shape).entropy()

        shape_info = _PSD(shape)
        halfsum = 0.5 * (dim + df)
        half_df = 0.5 * df
        return (-gammaln(halfsum) + gammaln(half_df)
                + 0.5 * dim * np.log(df * np.pi) + halfsum
                * (psi(halfsum) - psi(half_df))
                + 0.5 * shape_info.log_pdet)

    def entropy(self, loc=None, shape=1, df=1):
        """Calculate the differential entropy of a multivariate
        t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        h : float
            Differential entropy

        """
        dim, loc, shape, df = self._process_parameters(None, shape, df)
        return self._entropy(dim, df, shape)

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `P`), where `P` is the
            dimension of the random variable.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.rvs(loc, shape, df)
        array([[0.93477495, 3.00408716]])

        """
        # For implementation details, see equation (3):
        #
        #    Hofert, "On Sampling from the Multivariatet Distribution", 2013
        #     http://rjournal.github.io/archive/2013-2/hofert.pdf
        #
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        if np.isinf(df):
            x = np.ones(size)
        else:
            x = rng.chisquare(df, size=size) / df

        z = rng.multivariate_normal(np.zeros(dim), shape, size=size)
        samples = loc + z / np.sqrt(x)[..., None]
        return _squeeze_output(samples)

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_parameters(self, loc, shape, df):
        """
        Infer dimensionality from location array and shape matrix, handle
        defaults, and ensure compatible dimensions.
        """
        if loc is None and shape is None:
            loc = np.asarray(0, dtype=float)
            shape = np.asarray(1, dtype=float)
            dim = 1
        elif loc is None:
            shape = np.asarray(shape, dtype=float)
            if shape.ndim < 2:
                dim = 1
            else:
                dim = shape.shape[0]
            loc = np.zeros(dim)
        elif shape is None:
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
            shape = np.eye(dim)
        else:
            shape = np.asarray(shape, dtype=float)
            loc = np.asarray(loc, dtype=float)
            dim = loc.size

        if dim == 1:
            loc = loc.reshape(1)
            shape = shape.reshape(1, 1)

        if loc.ndim != 1 or loc.shape[0] != dim:
            raise ValueError("Array 'loc' must be a vector of length %d." %
                             dim)
        if shape.ndim == 0:
            shape = shape * np.eye(dim)
        elif shape.ndim == 1:
            shape = np.diag(shape)
        elif shape.ndim == 2 and shape.shape != (dim, dim):
            rows, cols = shape.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(shape.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'loc' is a vector of length %d.")
                msg = msg % (str(shape.shape), len(loc))
            raise ValueError(msg)
        elif shape.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % shape.ndim)

        # Process degrees of freedom.
        if df is None:
            df = 1
        elif df <= 0:
            raise ValueError("'df' must be greater than zero.")
        elif np.isnan(df):
            raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")

        return dim, loc, shape, df


class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        """Create a frozen multivariate t distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Examples
        --------
        >>> import numpy as np
        >>> loc = np.zeros(3)
        >>> shape = np.eye(3)
        >>> df = 10
        >>> dist = multivariate_t(loc, shape, df)
        >>> dist.rvs()
        array([[ 0.81412036, -1.53612361,  0.42199647]])
        >>> dist.pdf([1, 1, 1])
        array([0.01237803])

        """
        self._dist = multivariate_t_gen(seed)
        dim, loc, shape, df = self._dist._process_parameters(loc, shape, df)
        self.dim, self.loc, self.shape, self.df = dim, loc, shape, df
        self.shape_info = _PSD(shape, allow_singular=allow_singular)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.loc, U, log_pdet, self.df, self.dim,
                                  self.shape_info.rank)

    def cdf(self, x, *, maxpts=None, lower_limit=None, random_state=None):
        x = self._dist._process_quantiles(x, self.dim)
        return self._dist._cdf(x, self.loc, self.shape, self.df, self.dim,
                               maxpts, lower_limit, random_state)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(loc=self.loc,
                              shape=self.shape,
                              df=self.df,
                              size=size,
                              random_state=random_state)

    def entropy(self):
        return self._dist._entropy(self.dim, self.df, self.shape)


multivariate_t = multivariate_t_gen()


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_t_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'cdf', 'entropy']:
    method = multivariate_t_gen.__dict__[name]
    method_frozen = multivariate_t_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             mvt_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_params)


_mhg_doc_default_callparams = """\
m : array_like
    The number of each type of object in the population.
    That is, :math:`m[i]` is the number of objects of
    type :math:`i`.
n : array_like
    The number of samples taken from the population.
"""

_mhg_doc_callparams_note = """\
`m` must be an array of positive integers. If the quantile
:math:`i` contains values out of the range :math:`[0, m_i]`
where :math:`m_i` is the number of objects of type :math:`i`
in the population or if the parameters are inconsistent with one
another (e.g. ``x.sum() != n``), methods return the appropriate
value (e.g. ``0`` for ``pmf``). If `m` or `n` contain negative
values, the result will contain ``nan`` there.
"""

_mhg_doc_frozen_callparams = ""

_mhg_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

mhg_docdict_params = {
    '_doc_default_callparams': _mhg_doc_default_callparams,
    '_doc_callparams_note': _mhg_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

mhg_docdict_noparams = {
    '_doc_default_callparams': _mhg_doc_frozen_callparams,
    '_doc_callparams_note': _mhg_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multivariate_hypergeom_gen(multi_rv_generic):
    r"""A multivariate hypergeometric random variable.

    Methods
    -------
    pmf(x, m, n)
        Probability mass function.
    logpmf(x, m, n)
        Log of the probability mass function.
    rvs(m, n, size=1, random_state=None)
        Draw random samples from a multivariate hypergeometric
        distribution.
    mean(m, n)
        Mean of the multivariate hypergeometric distribution.
    var(m, n)
        Variance of the multivariate hypergeometric distribution.
    cov(m, n)
        Compute the covariance matrix of the multivariate
        hypergeometric distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_callparams_note)s

    The probability mass function for `multivariate_hypergeom` is

    .. math::

        P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{\binom{m_1}{x_1}
        \binom{m_2}{x_2} \cdots \binom{m_k}{x_k}}{\binom{M}{n}}, \\ \quad
        (x_1, x_2, \ldots, x_k) \in \mathbb{N}^k \text{ with }
        \sum_{i=1}^k x_i = n

    where :math:`m_i` are the number of objects of type :math:`i`, :math:`M`
    is the total number of objects in the population (sum of all the
    :math:`m_i`), and :math:`n` is the size of the sample to be taken
    from the population.

    .. versionadded:: 1.6.0

    Examples
    --------
    To evaluate the probability mass function of the multivariate
    hypergeometric distribution, with a dichotomous population of size
    :math:`10` and :math:`20`, at a sample of size :math:`12` with
    :math:`8` objects of the first type and :math:`4` objects of the
    second type, use:

    >>> from scipy.stats import multivariate_hypergeom
    >>> multivariate_hypergeom.pmf(x=[8, 4], m=[10, 20], n=12)
    0.0025207176631464523

    The `multivariate_hypergeom` distribution is identical to the
    corresponding `hypergeom` distribution (tiny numerical differences
    notwithstanding) when only two types (good and bad) of objects
    are present in the population as in the example above. Consider
    another example for a comparison with the hypergeometric distribution:

    >>> from scipy.stats import hypergeom
    >>> multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
    0.4395604395604395
    >>> hypergeom.pmf(k=3, M=15, n=4, N=10)
    0.43956043956044005

    The functions ``pmf``, ``logpmf``, ``mean``, ``var``, ``cov``, and ``rvs``
    support broadcasting, under the convention that the vector parameters
    (``x``, ``m``, and ``n``) are interpreted as if each row along the last
    axis is a single object. For instance, we can combine the previous two
    calls to `multivariate_hypergeom` as

    >>> multivariate_hypergeom.pmf(x=[[8, 4], [3, 1]], m=[[10, 20], [10, 5]],
    ...                            n=[12, 4])
    array([0.00252072, 0.43956044])

    This broadcasting also works for ``cov``, where the output objects are
    square matrices of size ``m.shape[-1]``. For example:

    >>> multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
    array([[[ 1.05, -1.05],
            [-1.05,  1.05]],
           [[ 1.56, -1.56],
            [-1.56,  1.56]]])

    That is, ``result[0]`` is equal to
    ``multivariate_hypergeom.cov(m=[7, 9], n=8)`` and ``result[1]`` is equal
    to ``multivariate_hypergeom.cov(m=[10, 15], n=12)``.

    Alternatively, the object may be called (as a function) to fix the `m`
    and `n` parameters, returning a "frozen" multivariate hypergeometric
    random variable.

    >>> rv = multivariate_hypergeom(m=[10, 20], n=12)
    >>> rv.pmf(x=[8, 4])
    0.0025207176631464523

    See Also
    --------
    scipy.stats.hypergeom : The hypergeometric distribution.
    scipy.stats.multinomial : The multinomial distribution.

    References
    ----------
    .. [1] The Multivariate Hypergeometric Distribution,
           http://www.randomservices.org/random/urn/MultiHypergeometric.html
    .. [2] Thomas J. Sargent and John Stachurski, 2020,
           Multivariate Hypergeometric Distribution
           https://python.quantecon.org/_downloads/pdf/multi_hyper.pdf
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mhg_docdict_params)

    def __call__(self, m, n, seed=None):
        """Create a frozen multivariate_hypergeom distribution.

        See `multivariate_hypergeom_frozen` for more information.
        """
        return multivariate_hypergeom_frozen(m, n, seed=seed)

    def _process_parameters(self, m, n):
        m = np.asarray(m)
        n = np.asarray(n)
        if m.size == 0:
            m = m.astype(int)
        if n.size == 0:
            n = n.astype(int)
        if not np.issubdtype(m.dtype, np.integer):
            raise TypeError("'m' must an array of integers.")
        if not np.issubdtype(n.dtype, np.integer):
            raise TypeError("'n' must an array of integers.")
        if m.ndim == 0:
            raise ValueError("'m' must be an array with"
                             " at least one dimension.")

        # check for empty arrays
        if m.size != 0:
            n = n[..., np.newaxis]

        m, n = np.broadcast_arrays(m, n)

        # check for empty arrays
        if m.size != 0:
            n = n[..., 0]

        mcond = m < 0

        M = m.sum(axis=-1)

        ncond = (n < 0) | (n > M)
        return M, m, n, mcond, ncond, np.any(mcond, axis=-1) | ncond

    def _process_quantiles(self, x, M, m, n):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError("'x' must an array of integers.")
        if x.ndim == 0:
            raise ValueError("'x' must be an array with"
                             " at least one dimension.")
        if not x.shape[-1] == m.shape[-1]:
            raise ValueError(f"Size of each quantile must be size of 'm': "
                             f"received {x.shape[-1]}, "
                             f"but expected {m.shape[-1]}.")

        # check for empty arrays
        if m.size != 0:
            n = n[..., np.newaxis]
            M = M[..., np.newaxis]

        x, m, n, M = np.broadcast_arrays(x, m, n, M)

        # check for empty arrays
        if m.size != 0:
            n, M = n[..., 0], M[..., 0]

        xcond = (x < 0) | (x > m)
        return (x, M, m, n, xcond,
                np.any(xcond, axis=-1) | (x.sum(axis=-1) != n))

    def _checkresult(self, result, cond, bad_value):
        result = np.asarray(result)
        if cond.ndim != 0:
            result[cond] = bad_value
        elif cond:
            return bad_value
        if result.ndim == 0:
            return result[()]
        return result

    def _logpmf(self, x, M, m, n, mxcond, ncond):
        # This equation of the pmf comes from the relation,
        # n combine r = beta(n+1, 1) / beta(r+1, n-r+1)
        num = np.zeros_like(m, dtype=np.float_)
        den = np.zeros_like(n, dtype=np.float_)
        m, x = m[~mxcond], x[~mxcond]
        M, n = M[~ncond], n[~ncond]
        num[~mxcond] = (betaln(m+1, 1) - betaln(x+1, m-x+1))
        den[~ncond] = (betaln(M+1, 1) - betaln(n+1, M-n+1))
        num[mxcond] = np.nan
        den[ncond] = np.nan
        num = num.sum(axis=-1)
        return num - den

    def logpmf(self, x, m, n):
        """Log of the multivariate hypergeometric probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        M, m, n, mcond, ncond, mncond = self._process_parameters(m, n)
        (x, M, m, n, xcond,
         xcond_reduced) = self._process_quantiles(x, M, m, n)
        mxcond = mcond | xcond
        ncond = ncond | np.zeros(n.shape, dtype=np.bool_)

        result = self._logpmf(x, M, m, n, mxcond, ncond)

        # replace values for which x was out of the domain; broadcast
        # xcond to the right shape
        xcond_ = xcond_reduced | np.zeros(mncond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, -np.inf)

        # replace values bad for n or m; broadcast
        # mncond to the right shape
        mncond_ = mncond | np.zeros(xcond_reduced.shape, dtype=np.bool_)
        return self._checkresult(result, mncond_, np.nan)

    def pmf(self, x, m, n):
        """Multivariate hypergeometric probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        out = np.exp(self.logpmf(x, m, n))
        return out

    def mean(self, m, n):
        """Mean of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : array_like or scalar
            The mean of the distribution
        """
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        # check for empty arrays
        if m.size != 0:
            M, n = M[..., np.newaxis], n[..., np.newaxis]
        cond = (M == 0)
        M = np.ma.masked_array(M, mask=cond)
        mu = n*(m/M)
        if m.size != 0:
            mncond = (mncond[..., np.newaxis] |
                      np.zeros(mu.shape, dtype=np.bool_))
        return self._checkresult(mu, mncond, np.nan)

    def var(self, m, n):
        """Variance of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        array_like
            The variances of the components of the distribution.  This is
            the diagonal of the covariance matrix of the distribution
        """
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        # check for empty arrays
        if m.size != 0:
            M, n = M[..., np.newaxis], n[..., np.newaxis]
        cond = (M == 0) & (M-1 == 0)
        M = np.ma.masked_array(M, mask=cond)
        output = n * m/M * (M-m)/M * (M-n)/(M-1)
        if m.size != 0:
            mncond = (mncond[..., np.newaxis] |
                      np.zeros(output.shape, dtype=np.bool_))
        return self._checkresult(output, mncond, np.nan)

    def cov(self, m, n):
        """Covariance matrix of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        cov : array_like
            The covariance matrix of the distribution
        """
        # see [1]_ for the formula and [2]_ for implementation
        # cov( x_i,x_j ) = -n * (M-n)/(M-1) * (K_i*K_j) / (M**2)
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        # check for empty arrays
        if m.size != 0:
            M = M[..., np.newaxis, np.newaxis]
            n = n[..., np.newaxis, np.newaxis]
        cond = (M == 0) & (M-1 == 0)
        M = np.ma.masked_array(M, mask=cond)
        output = (-n * (M-n)/(M-1) *
                  np.einsum("...i,...j->...ij", m, m) / (M**2))
        # check for empty arrays
        if m.size != 0:
            M, n = M[..., 0, 0], n[..., 0, 0]
            cond = cond[..., 0, 0]
        dim = m.shape[-1]
        # diagonal entries need to be computed differently
        for i in range(dim):
            output[..., i, i] = (n * (M-n) * m[..., i]*(M-m[..., i]))
            output[..., i, i] = output[..., i, i] / (M-1)
            output[..., i, i] = output[..., i, i] / (M**2)
        if m.size != 0:
            mncond = (mncond[..., np.newaxis, np.newaxis] |
                      np.zeros(output.shape, dtype=np.bool_))
        return self._checkresult(output, mncond, np.nan)

    def rvs(self, m, n, size=None, random_state=None):
        """Draw random samples from a multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw. Default is ``None``, in which case a
            single variate is returned as an array with shape ``m.shape``.
        %(_doc_random_state)s

        Returns
        -------
        rvs : array_like
            Random variates of shape ``size`` or ``m.shape``
            (if ``size=None``).

        Notes
        -----
        %(_doc_callparams_note)s

        Also note that NumPy's `multivariate_hypergeometric` sampler is not
        used as it doesn't support broadcasting.
        """
        M, m, n, _, _, _ = self._process_parameters(m, n)

        random_state = self._get_random_state(random_state)

        if size is not None and isinstance(size, int):
            size = (size, )

        if size is None:
            rvs = np.empty(m.shape, dtype=m.dtype)
        else:
            rvs = np.empty(size + (m.shape[-1], ), dtype=m.dtype)
        rem = M

        # This sampler has been taken from numpy gh-13794
        # https://github.com/numpy/numpy/pull/13794
        for c in range(m.shape[-1] - 1):
            rem = rem - m[..., c]
            n0mask = n == 0
            rvs[..., c] = (~n0mask *
                           random_state.hypergeometric(m[..., c],
                                                       rem + n0mask,
                                                       n + n0mask,
                                                       size=size))
            n = n - rvs[..., c]
        rvs[..., m.shape[-1] - 1] = n

        return rvs


multivariate_hypergeom = multivariate_hypergeom_gen()


class multivariate_hypergeom_frozen(multi_rv_frozen):
    def __init__(self, m, n, seed=None):
        self._dist = multivariate_hypergeom_gen(seed)
        (self.M, self.m, self.n,
         self.mcond, self.ncond,
         self.mncond) = self._dist._process_parameters(m, n)

        # monkey patch self._dist
        def _process_parameters(m, n):
            return (self.M, self.m, self.n,
                    self.mcond, self.ncond,
                    self.mncond)
        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, self.m, self.n)

    def pmf(self, x):
        return self._dist.pmf(x, self.m, self.n)

    def mean(self):
        return self._dist.mean(self.m, self.n)

    def var(self):
        return self._dist.var(self.m, self.n)

    def cov(self):
        return self._dist.cov(self.m, self.n)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.m, self.n,
                              size=size,
                              random_state=random_state)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_hypergeom and fill in default strings in class docstrings
for name in ['logpmf', 'pmf', 'mean', 'var', 'cov', 'rvs']:
    method = multivariate_hypergeom_gen.__dict__[name]
    method_frozen = multivariate_hypergeom_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, mhg_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__,
                                      mhg_docdict_params)


class random_table_gen(multi_rv_generic):
    r"""Contingency tables from independent samples with fixed marginal sums.

    This is the distribution of random tables with given row and column vector
    sums. This distribution represents the set of random tables under the null
    hypothesis that rows and columns are independent. It is used in hypothesis
    tests of independence.

    Because of assumed independence, the expected frequency of each table
    element can be computed from the row and column sums, so that the
    distribution is completely determined by these two vectors.

    Methods
    -------
    logpmf(x)
        Log-probability of table `x` to occur in the distribution.
    pmf(x)
        Probability of table `x` to occur in the distribution.
    mean(row, col)
        Mean table.
    rvs(row, col, size=None, method=None, random_state=None)
        Draw random tables with given row and column vector sums.

    Parameters
    ----------
    %(_doc_row_col)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_row_col_note)s

    Random elements from the distribution are generated either with Boyett's
    [1]_ or Patefield's algorithm [2]_. Boyett's algorithm has
    O(N) time and space complexity, where N is the total sum of entries in the
    table. Patefield's algorithm has O(K x log(N)) time complexity, where K is
    the number of cells in the table and requires only a small constant work
    space. By default, the `rvs` method selects the fastest algorithm based on
    the input, but you can specify the algorithm with the keyword `method`.
    Allowed values are "boyett" and "patefield".

    .. versionadded:: 1.10.0

    Examples
    --------
    >>> from scipy.stats import random_table

    >>> row = [1, 5]
    >>> col = [2, 3, 1]
    >>> random_table.mean(row, col)
    array([[0.33333333, 0.5       , 0.16666667],
           [1.66666667, 2.5       , 0.83333333]])

    Alternatively, the object may be called (as a function) to fix the row
    and column vector sums, returning a "frozen" distribution.

    >>> dist = random_table(row, col)
    >>> dist.rvs(random_state=123)
    array([[1., 0., 0.],
           [1., 3., 1.]])

    References
    ----------
    .. [1] J. Boyett, AS 144 Appl. Statist. 28 (1979) 329-332
    .. [2] W.M. Patefield, AS 159 Appl. Statist. 30 (1981) 91-97
    """

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, row, col, *, seed=None):
        """Create a frozen distribution of tables with given marginals.

        See `random_table_frozen` for more information.
        """
        return random_table_frozen(row, col, seed=seed)

    def logpmf(self, x, row, col):
        """Log-probability of table to occur in the distribution.

        Parameters
        ----------
        %(_doc_x)s
        %(_doc_row_col)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`.

        Notes
        -----
        %(_doc_row_col_note)s

        If row and column marginals of `x` do not match `row` and `col`,
        negative infinity is returned.

        Examples
        --------
        >>> from scipy.stats import random_table
        >>> import numpy as np

        >>> x = [[1, 5, 1], [2, 3, 1]]
        >>> row = np.sum(x, axis=1)
        >>> col = np.sum(x, axis=0)
        >>> random_table.logpmf(x, row, col)
        -1.6306401200847027

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.logpmf(x)
        -1.6306401200847027
        """
        r, c, n = self._process_parameters(row, col)
        x = np.asarray(x)

        if x.ndim < 2:
            raise ValueError("`x` must be at least two-dimensional")

        dtype_is_int = np.issubdtype(x.dtype, np.integer)
        with np.errstate(invalid='ignore'):
            if not dtype_is_int and not np.all(x.astype(int) == x):
                raise ValueError("`x` must contain only integral values")

        # x does not contain NaN if we arrive here
        if np.any(x < 0):
            raise ValueError("`x` must contain only non-negative values")

        r2 = np.sum(x, axis=-1)
        c2 = np.sum(x, axis=-2)

        if r2.shape[-1] != len(r):
            raise ValueError("shape of `x` must agree with `row`")

        if c2.shape[-1] != len(c):
            raise ValueError("shape of `x` must agree with `col`")

        res = np.empty(x.shape[:-2])

        mask = np.all(r2 == r, axis=-1) & np.all(c2 == c, axis=-1)

        def lnfac(x):
            return gammaln(x + 1)

        res[mask] = (np.sum(lnfac(r), axis=-1) + np.sum(lnfac(c), axis=-1)
                     - lnfac(n) - np.sum(lnfac(x[mask]), axis=(-1, -2)))
        res[~mask] = -np.inf

        return res[()]

    def pmf(self, x, row, col):
        """Probability of table to occur in the distribution.

        Parameters
        ----------
        %(_doc_x)s
        %(_doc_row_col)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability mass function evaluated at `x`.

        Notes
        -----
        %(_doc_row_col_note)s

        If row and column marginals of `x` do not match `row` and `col`,
        zero is returned.

        Examples
        --------
        >>> from scipy.stats import random_table
        >>> import numpy as np

        >>> x = [[1, 5, 1], [2, 3, 1]]
        >>> row = np.sum(x, axis=1)
        >>> col = np.sum(x, axis=0)
        >>> random_table.pmf(x, row, col)
        0.19580419580419592

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.pmf(x)
        0.19580419580419592
        """
        return np.exp(self.logpmf(x, row, col))

    def mean(self, row, col):
        """Mean of distribution of conditional tables.
        %(_doc_mean_params)s

        Returns
        -------
        mean: ndarray
            Mean of the distribution.

        Notes
        -----
        %(_doc_row_col_note)s

        Examples
        --------
        >>> from scipy.stats import random_table

        >>> row = [1, 5]
        >>> col = [2, 3, 1]
        >>> random_table.mean(row, col)
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.mean()
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])
        """
        r, c, n = self._process_parameters(row, col)
        return np.outer(r, c) / n

    def rvs(self, row, col, *, size=None, method=None, random_state=None):
        """Draw random tables with fixed column and row marginals.

        Parameters
        ----------
        %(_doc_row_col)s
        size : integer, optional
            Number of samples to draw (default 1).
        method : str, optional
            Which method to use, "boyett" or "patefield". If None (default),
            selects the fastest method for this input.
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random 2D tables of shape (`size`, `len(row)`, `len(col)`).

        Notes
        -----
        %(_doc_row_col_note)s

        Examples
        --------
        >>> from scipy.stats import random_table

        >>> row = [1, 5]
        >>> col = [2, 3, 1]
        >>> random_table.rvs(row, col, random_state=123)
        array([[1., 0., 0.],
               [1., 3., 1.]])

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.rvs(random_state=123)
        array([[1., 0., 0.],
               [1., 3., 1.]])
        """
        r, c, n = self._process_parameters(row, col)
        size, shape = self._process_size_shape(size, r, c)

        random_state = self._get_random_state(random_state)
        meth = self._process_rvs_method(method, r, c, n)

        return meth(r, c, n, size, random_state).reshape(shape)

    @staticmethod
    def _process_parameters(row, col):
        """
        Check that row and column vectors are one-dimensional, that they do
        not contain negative or non-integer entries, and that the sums over
        both vectors are equal.
        """
        r = np.array(row, dtype=np.int64, copy=True)
        c = np.array(col, dtype=np.int64, copy=True)

        if np.ndim(r) != 1:
            raise ValueError("`row` must be one-dimensional")
        if np.ndim(c) != 1:
            raise ValueError("`col` must be one-dimensional")

        if np.any(r < 0):
            raise ValueError("each element of `row` must be non-negative")
        if np.any(c < 0):
            raise ValueError("each element of `col` must be non-negative")

        n = np.sum(r)
        if n != np.sum(c):
            raise ValueError("sums over `row` and `col` must be equal")

        if not np.all(r == np.asarray(row)):
            raise ValueError("each element of `row` must be an integer")
        if not np.all(c == np.asarray(col)):
            raise ValueError("each element of `col` must be an integer")

        return r, c, n

    @staticmethod
    def _process_size_shape(size, r, c):
        """
        Compute the number of samples to be drawn and the shape of the output
        """
        shape = (len(r), len(c))

        if size is None:
            return 1, shape

        size = np.atleast_1d(size)
        if not np.issubdtype(size.dtype, np.integer) or np.any(size < 0):
            raise ValueError("`size` must be a non-negative integer or `None`")

        return np.prod(size), tuple(size) + shape

    @classmethod
    def _process_rvs_method(cls, method, r, c, n):
        known_methods = {
            None: cls._rvs_select(r, c, n),
            "boyett": cls._rvs_boyett,
            "patefield": cls._rvs_patefield,
        }
        try:
            return known_methods[method]
        except KeyError:
            raise ValueError(f"'{method}' not recognized, "
                             f"must be one of {set(known_methods)}")

    @classmethod
    def _rvs_select(cls, r, c, n):
        fac = 1.0  # benchmarks show that this value is about 1
        k = len(r) * len(c)  # number of cells
        # n + 1 guards against failure if n == 0
        if n > fac * np.log(n + 1) * k:
            return cls._rvs_patefield
        return cls._rvs_boyett

    @staticmethod
    def _rvs_boyett(row, col, ntot, size, random_state):
        return _rcont.rvs_rcont1(row, col, ntot, size, random_state)

    @staticmethod
    def _rvs_patefield(row, col, ntot, size, random_state):
        return _rcont.rvs_rcont2(row, col, ntot, size, random_state)


random_table = random_table_gen()


class random_table_frozen(multi_rv_frozen):
    def __init__(self, row, col, *, seed=None):
        self._dist = random_table_gen(seed)
        self._params = self._dist._process_parameters(row, col)

        # monkey patch self._dist
        def _process_parameters(r, c):
            return self._params
        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, None, None)

    def pmf(self, x):
        return self._dist.pmf(x, None, None)

    def mean(self):
        return self._dist.mean(None, None)

    def rvs(self, size=None, method=None, random_state=None):
        # optimisations are possible here
        return self._dist.rvs(None, None, size=size, method=method,
                              random_state=random_state)


_ctab_doc_row_col = """\
row : array_like
    Sum of table entries in each row.
col : array_like
    Sum of table entries in each column."""

_ctab_doc_x = """\
x : array-like
   Two-dimensional table of non-negative integers, or a
   multi-dimensional array with the last two dimensions
   corresponding with the tables."""

_ctab_doc_row_col_note = """\
The row and column vectors must be one-dimensional, not empty,
and each sum up to the same value. They cannot contain negative
or noninteger entries."""

_ctab_doc_mean_params = f"""
Parameters
----------
{_ctab_doc_row_col}"""

_ctab_doc_row_col_note_frozen = """\
See class definition for a detailed description of parameters."""

_ctab_docdict = {
    "_doc_random_state": _doc_random_state,
    "_doc_row_col": _ctab_doc_row_col,
    "_doc_x": _ctab_doc_x,
    "_doc_mean_params": _ctab_doc_mean_params,
    "_doc_row_col_note": _ctab_doc_row_col_note,
}

_ctab_docdict_frozen = _ctab_docdict.copy()
_ctab_docdict_frozen.update({
    "_doc_row_col": "",
    "_doc_mean_params": "",
    "_doc_row_col_note": _ctab_doc_row_col_note_frozen,
})


def _docfill(obj, docdict, template=None):
    obj.__doc__ = doccer.docformat(template or obj.__doc__, docdict)


# Set frozen generator docstrings from corresponding docstrings in
# random_table and fill in default strings in class docstrings
_docfill(random_table_gen, _ctab_docdict)
for name in ['logpmf', 'pmf', 'mean', 'rvs']:
    method = random_table_gen.__dict__[name]
    method_frozen = random_table_frozen.__dict__[name]
    _docfill(method_frozen, _ctab_docdict_frozen, method.__doc__)
    _docfill(method, _ctab_docdict)


class uniform_direction_gen(multi_rv_generic):
    r"""A vector-valued uniform direction.

    Return a random direction (unit vector). The `dim` keyword specifies
    the dimensionality of the space.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random directions.

    Parameters
    ----------
    dim : scalar
        Dimension of directions.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This distribution generates unit vectors uniformly distributed on
    the surface of a hypersphere. These can be interpreted as random
    directions.
    For example, if `dim` is 3, 3D vectors from the surface of :math:`S^2`
    will be sampled.

    References
    ----------
    .. [1] Marsaglia, G. (1972). "Choosing a Point from the Surface of a
           Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import uniform_direction
    >>> x = uniform_direction.rvs(3)
    >>> np.linalg.norm(x)
    1.

    This generates one random direction, a vector on the surface of
    :math:`S^2`.

    Alternatively, the object may be called (as a function) to return a frozen
    distribution with fixed `dim` parameter. Here,
    we create a `uniform_direction` with ``dim=3`` and draw 5 observations.
    The samples are then arranged in an array of shape 5x3.

    >>> rng = np.random.default_rng()
    >>> uniform_sphere_dist = uniform_direction(3)
    >>> unit_vectors = uniform_sphere_dist.rvs(5, random_state=rng)
    >>> unit_vectors
    array([[ 0.56688642, -0.1332634 , -0.81294566],
           [-0.427126  , -0.74779278,  0.50830044],
           [ 0.3793989 ,  0.92346629,  0.05715323],
           [ 0.36428383, -0.92449076, -0.11231259],
           [-0.27733285,  0.94410968, -0.17816678]])
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen n-dimensional uniform direction distribution.

        See `uniform_direction` for more information.
        """
        return uniform_direction_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim < 1 or dim != int(dim):
            raise ValueError("Dimension of vector must be specified, "
                             "and must be an integer greater than 0.")

        return int(dim)

    def rvs(self, dim, size=None, random_state=None):
        """Draw random samples from S(N-1).

        Parameters
        ----------
        dim : integer
            Dimension of space (N).
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            Pseudorandom number generator state used to generate resamples.

            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

        Returns
        -------
        rvs : ndarray
            Random direction vectors

        """
        random_state = self._get_random_state(random_state)
        if size is None:
            size = np.array([], dtype=int)
        size = np.atleast_1d(size)

        dim = self._process_parameters(dim)

        samples = _sample_uniform_direction(dim, size, random_state)
        return samples


uniform_direction = uniform_direction_gen()


class uniform_direction_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen n-dimensional uniform direction distribution.

        Parameters
        ----------
        dim : int
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import uniform_direction
        >>> x = uniform_direction(3)
        >>> x.rvs()

        """
        self._dist = uniform_direction_gen(seed)
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=None, random_state=None):
        return self._dist.rvs(self.dim, size, random_state)


def _sample_uniform_direction(dim, size, random_state):
    """
    Private method to generate uniform directions
    Reference: Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    """
    samples_shape = np.append(size, dim)
    samples = random_state.standard_normal(samples_shape)
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples


_dirichlet_mn_doc_default_callparams = """\
alpha : array_like
    The concentration parameters. The number of entries along the last axis
    determines the dimensionality of the distribution. Each entry must be
    strictly positive.
n : int or array_like
    The number of trials. Each element must be a strictly positive integer.
"""

_dirichlet_mn_doc_frozen_callparams = ""

_dirichlet_mn_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

dirichlet_mn_docdict_params = {
    '_dirichlet_mn_doc_default_callparams': _dirichlet_mn_doc_default_callparams,  # noqa
    '_doc_random_state': _doc_random_state
}

dirichlet_mn_docdict_noparams = {
    '_dirichlet_mn_doc_default_callparams': _dirichlet_mn_doc_frozen_callparams, # noqa
    '_doc_random_state': _doc_random_state
}


def _dirichlet_multinomial_check_parameters(alpha, n, x=None):

    alpha = np.asarray(alpha)
    n = np.asarray(n)

    if x is not None:
        # Ensure that `x` and `alpha` are arrays. If the shapes are
        # incompatible, NumPy will raise an appropriate error.
        try:
            x, alpha = np.broadcast_arrays(x, alpha)
        except ValueError as e:
            msg = "`x` and `alpha` must be broadcastable."
            raise ValueError(msg) from e

        x_int = np.floor(x)
        if np.any(x < 0) or np.any(x != x_int):
            raise ValueError("`x` must contain only non-negative integers.")
        x = x_int

    if np.any(alpha <= 0):
        raise ValueError("`alpha` must contain only positive values.")

    n_int = np.floor(n)
    if np.any(n <= 0) or np.any(n != n_int):
        raise ValueError("`n` must be a positive integer.")
    n = n_int

    sum_alpha = np.sum(alpha, axis=-1)
    sum_alpha, n = np.broadcast_arrays(sum_alpha, n)

    return (alpha, sum_alpha, n) if x is None else (alpha, sum_alpha, n, x)


class dirichlet_multinomial_gen(multi_rv_generic):
    r"""A Dirichlet multinomial random variable.

    The Dirichlet multinomial distribution is a compound probability
    distribution: it is the multinomial distribution with number of trials
    `n` and class probabilities ``p`` randomly sampled from a Dirichlet
    distribution with concentration parameters ``alpha``.

    Methods
    -------
    logpmf(x, alpha, n):
        Log of the probability mass function.
    pmf(x, alpha, n):
        Probability mass function.
    mean(alpha, n):
        Mean of the Dirichlet multinomial distribution.
    var(alpha, n):
        Variance of the Dirichlet multinomial distribution.
    cov(alpha, n):
        The covariance of the Dirichlet multinomial distribution.

    Parameters
    ----------
    %(_dirichlet_mn_doc_default_callparams)s
    %(_doc_random_state)s

    See Also
    --------
    scipy.stats.dirichlet : The dirichlet distribution.
    scipy.stats.multinomial : The multinomial distribution.

    References
    ----------
    .. [1] Dirichlet-multinomial distribution, Wikipedia,
           https://www.wikipedia.org/wiki/Dirichlet-multinomial_distribution

    Examples
    --------
    >>> from scipy.stats import dirichlet_multinomial

    Get the PMF

    >>> n = 6  # number of trials
    >>> alpha = [3, 4, 5]  # concentration parameters
    >>> x = [1, 2, 3]  # counts
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    0.08484162895927604

    If the sum of category counts does not equal the number of trials,
    the probability mass is zero.

    >>> dirichlet_multinomial.pmf(x, alpha, n=7)
    0.0

    Get the log of the PMF

    >>> dirichlet_multinomial.logpmf(x, alpha, n)
    -2.4669689491013327

    Get the mean

    >>> dirichlet_multinomial.mean(alpha, n)
    array([1.5, 2. , 2.5])

    Get the variance

    >>> dirichlet_multinomial.var(alpha, n)
    array([1.55769231, 1.84615385, 2.01923077])

    Get the covariance

    >>> dirichlet_multinomial.cov(alpha, n)
    array([[ 1.55769231, -0.69230769, -0.86538462],
           [-0.69230769,  1.84615385, -1.15384615],
           [-0.86538462, -1.15384615,  2.01923077]])

    Alternatively, the object may be called (as a function) to fix the
    `alpha` and `n` parameters, returning a "frozen" Dirichlet multinomial
    random variable.

    >>> dm = dirichlet_multinomial(alpha, n)
    >>> dm.pmf(x)
    0.08484162895927579

    All methods are fully vectorized. Each element of `x` and `alpha` is
    a vector (along the last axis), each element of `n` is an
    integer (scalar), and the result is computed element-wise.

    >>> x = [[1, 2, 3], [4, 5, 6]]
    >>> alpha = [[1, 2, 3], [4, 5, 6]]
    >>> n = [6, 15]
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    array([0.06493506, 0.02626937])

    >>> dirichlet_multinomial.cov(alpha, n).shape  # both covariance matrices
    (2, 3, 3)

    Broadcasting according to standard NumPy conventions is supported. Here,
    we have four sets of concentration parameters (each a two element vector)
    for each of three numbers of trials (each a scalar).

    >>> alpha = [[3, 4], [4, 5], [5, 6], [6, 7]]
    >>> n = [[6], [7], [8]]
    >>> dirichlet_multinomial.mean(alpha, n).shape
    (3, 4, 2)

    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__,
                                        dirichlet_mn_docdict_params)

    def __call__(self, alpha, n, seed=None):
        return dirichlet_multinomial_frozen(alpha, n, seed=seed)

    def logpmf(self, x, alpha, n):
        """The log of the probability mass function.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Log of the probability mass function.

        """

        a, Sa, n, x = _dirichlet_multinomial_check_parameters(alpha, n, x)

        out = np.asarray(loggamma(Sa) + loggamma(n + 1) - loggamma(n + Sa))
        out += (loggamma(x + a) - (loggamma(a) + loggamma(x + 1))).sum(axis=-1)
        np.place(out, n != x.sum(axis=-1), -np.inf)
        return out[()]

    def pmf(self, x, alpha, n):
        """Probability mass function for a Dirichlet multinomial distribution.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Probability mass function.

        """
        return np.exp(self.logpmf(x, alpha, n))

    def mean(self, alpha, n):
        """Mean of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray
            Mean of a Dirichlet multinomial distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = n[..., np.newaxis], Sa[..., np.newaxis]
        return n * a / Sa

    def var(self, alpha, n):
        """The variance of the Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: array_like
            The variances of the components of the distribution. This is
            the diagonal of the covariance matrix of the distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = n[..., np.newaxis], Sa[..., np.newaxis]
        return n * a / Sa * (1 - a/Sa) * (n + Sa) / (1 + Sa)

    def cov(self, alpha, n):
        """Covariance matrix of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out : array_like
            The covariance matrix of the distribution.

        """
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        var = dirichlet_multinomial.var(a, n)

        n, Sa = n[..., np.newaxis, np.newaxis], Sa[..., np.newaxis, np.newaxis]
        aiaj = a[..., :, np.newaxis] * a[..., np.newaxis, :]
        cov = -n * aiaj / Sa ** 2 * (n + Sa) / (1 + Sa)

        ii = np.arange(cov.shape[-1])
        cov[..., ii, ii] = var
        return cov


dirichlet_multinomial = dirichlet_multinomial_gen()


class dirichlet_multinomial_frozen(multi_rv_frozen):
    def __init__(self, alpha, n, seed=None):
        alpha, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        self.alpha = alpha
        self.n = n
        self._dist = dirichlet_multinomial_gen(seed)

    def logpmf(self, x):
        return self._dist.logpmf(x, self.alpha, self.n)

    def pmf(self, x):
        return self._dist.pmf(x, self.alpha, self.n)

    def mean(self):
        return self._dist.mean(self.alpha, self.n)

    def var(self):
        return self._dist.var(self.alpha, self.n)

    def cov(self):
        return self._dist.cov(self.alpha, self.n)


# Set frozen generator docstrings from corresponding docstrings in
# dirichlet_multinomial and fill in default strings in class docstrings.
for name in ['logpmf', 'pmf', 'mean', 'var', 'cov']:
    method = dirichlet_multinomial_gen.__dict__[name]
    method_frozen = dirichlet_multinomial_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, dirichlet_mn_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__,
                                      dirichlet_mn_docdict_params)


class vonmises_fisher_gen(multi_rv_generic):
    r"""A von Mises-Fisher variable.

    The `mu` keyword specifies the mean direction vector. The `kappa` keyword
    specifies the concentration parameter.

    Methods
    -------
    pdf(x, mu=None, kappa=1)
        Probability density function.
    logpdf(x, mu=None, kappa=1)
        Log of the probability density function.
    rvs(mu=None, kappa=1, size=1, random_state=None)
        Draw random samples from a von Mises-Fisher distribution.
    entropy(mu=None, kappa=1)
        Compute the differential entropy of the von Mises-Fisher distribution.
    fit(data)
        Fit a von Mises-Fisher distribution to data.

    Parameters
    ----------
    mu : array_like
        Mean direction of the distribution. Must be a one-dimensional unit
        vector of norm 1.
    kappa : float
        Concentration parameter. Must be positive.
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    See Also
    --------
    scipy.stats.vonmises : Von-Mises Fisher distribution in 2D on a circle
    uniform_direction : uniform distribution on the surface of a hypersphere

    Notes
    -----
    The von Mises-Fisher distribution is a directional distribution on the
    surface of the unit hypersphere. The probability density
    function of a unit vector :math:`\mathbf{x}` is

    .. math::

        f(\mathbf{x}) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)}
               \exp\left(\kappa \mathbf{\mu}^T\mathbf{x}\right),

    where :math:`\mathbf{\mu}` is the mean direction, :math:`\kappa` the
    concentration parameter, :math:`d` the dimension and :math:`I` the
    modified Bessel function of the first kind. As :math:`\mu` represents
    a direction, it must be a unit vector or in other words, a point
    on the hypersphere: :math:`\mathbf{\mu}\in S^{d-1}`. :math:`\kappa` is a
    concentration parameter, which means that it must be positive
    (:math:`\kappa>0`) and that the distribution becomes more narrow with
    increasing :math:`\kappa`. In that sense, the reciprocal value
    :math:`1/\kappa` resembles the variance parameter of the normal
    distribution.

    The von Mises-Fisher distribution often serves as an analogue of the
    normal distribution on the sphere. Intuitively, for unit vectors, a
    useful distance measure is given by the angle :math:`\alpha` between
    them. This is exactly what the scalar product
    :math:`\mathbf{\mu}^T\mathbf{x}=\cos(\alpha)` in the
    von Mises-Fisher probability density function describes: the angle
    between the mean direction :math:`\mathbf{\mu}` and the vector
    :math:`\mathbf{x}`. The larger the angle between them, the smaller the
    probability to observe :math:`\mathbf{x}` for this particular mean
    direction :math:`\mathbf{\mu}`.

    In dimensions 2 and 3, specialized algorithms are used for fast sampling
    [2]_, [3]_. For dimenions of 4 or higher the rejection sampling algorithm
    described in [4]_ is utilized. This implementation is partially based on
    the geomstats package [5]_, [6]_.

    .. versionadded:: 1.11

    References
    ----------
    .. [1] Von Mises-Fisher distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    .. [2] Mardia, K., and Jupp, P. Directional statistics. Wiley, 2000.
    .. [3] J. Wenzel. Numerically stable sampling of the von Mises Fisher
           distribution on S2.
           https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    .. [4] Wood, A. Simulation of the von mises fisher distribution.
           Communications in statistics-simulation and computation 23,
           1 (1994), 157-164. https://doi.org/10.1080/03610919408813161
    .. [5] geomstats, Github. MIT License. Accessed: 06.01.2023.
           https://github.com/geomstats/geomstats
    .. [6] Miolane, N. et al. Geomstats:  A Python Package for Riemannian
           Geometry in Machine Learning. Journal of Machine Learning Research
           21 (2020). http://jmlr.org/papers/v21/19-027.html

    Examples
    --------
    **Visualization of the probability density**

    Plot the probability density in three dimensions for increasing
    concentration parameter. The density is calculated by the ``pdf``
    method.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import vonmises_fisher
    >>> from matplotlib.colors import Normalize
    >>> n_grid = 100
    >>> u = np.linspace(0, np.pi, n_grid)
    >>> v = np.linspace(0, 2 * np.pi, n_grid)
    >>> u_grid, v_grid = np.meshgrid(u, v)
    >>> vertices = np.stack([np.cos(v_grid) * np.sin(u_grid),
    ...                      np.sin(v_grid) * np.sin(u_grid),
    ...                      np.cos(u_grid)],
    ...                     axis=2)
    >>> x = np.outer(np.cos(v), np.sin(u))
    >>> y = np.outer(np.sin(v), np.sin(u))
    >>> z = np.outer(np.ones_like(u), np.cos(u))
    >>> def plot_vmf_density(ax, x, y, z, vertices, mu, kappa):
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     pdf_values = vmf.pdf(vertices)
    ...     pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1,
    ...                     facecolors=plt.cm.viridis(pdfnorm(pdf_values)),
    ...                     linewidth=0)
    ...     ax.set_aspect('equal')
    ...     ax.view_init(azim=-130, elev=0)
    ...     ax.axis('off')
    ...     ax.set_title(rf"$\kappa={kappa}$")
    >>> fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4),
    ...                          subplot_kw={"projection": "3d"})
    >>> left, middle, right = axes
    >>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
    >>> plot_vmf_density(left, x, y, z, vertices, mu, 5)
    >>> plot_vmf_density(middle, x, y, z, vertices, mu, 20)
    >>> plot_vmf_density(right, x, y, z, vertices, mu, 100)
    >>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, wspace=0.)
    >>> plt.show()

    As we increase the concentration parameter, the points are getting more
    clustered together around the mean direction.

    **Sampling**

    Draw 5 samples from the distribution using the ``rvs`` method resulting
    in a 5x3 array.

    >>> rng = np.random.default_rng()
    >>> mu = np.array([0, 0, 1])
    >>> samples = vonmises_fisher(mu, 20).rvs(5, random_state=rng)
    >>> samples
    array([[ 0.3884594 , -0.32482588,  0.86231516],
           [ 0.00611366, -0.09878289,  0.99509023],
           [-0.04154772, -0.01637135,  0.99900239],
           [-0.14613735,  0.12553507,  0.98126695],
           [-0.04429884, -0.23474054,  0.97104814]])

    These samples are unit vectors on the sphere :math:`S^2`. To verify,
    let us calculate their euclidean norms:

    >>> np.linalg.norm(samples, axis=1)
    array([1., 1., 1., 1., 1.])

    Plot 20 observations drawn from the von Mises-Fisher distribution for
    increasing concentration parameter :math:`\kappa`. The red dot highlights
    the mean direction :math:`\mu`.

    >>> def plot_vmf_samples(ax, x, y, z, mu, kappa):
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     samples = vmf.rvs(20)
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
    ...                     alpha=0.2)
    ...     ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5)
    ...     ax.scatter(mu[0], mu[1], mu[2], c='r', s=30)
    ...     ax.set_aspect('equal')
    ...     ax.view_init(azim=-130, elev=0)
    ...     ax.axis('off')
    ...     ax.set_title(rf"$\kappa={kappa}$")
    >>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
    >>> fig, axes = plt.subplots(nrows=1, ncols=3,
    ...                          subplot_kw={"projection": "3d"},
    ...                          figsize=(9, 4))
    >>> left, middle, right = axes
    >>> plot_vmf_samples(left, x, y, z, mu, 5)
    >>> plot_vmf_samples(middle, x, y, z, mu, 20)
    >>> plot_vmf_samples(right, x, y, z, mu, 100)
    >>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0,
    ...                     right=1.0, wspace=0.)
    >>> plt.show()

    The plots show that with increasing concentration :math:`\kappa` the
    resulting samples are centered more closely around the mean direction.

    **Fitting the distribution parameters**

    The distribution can be fitted to data using the ``fit`` method returning
    the estimated parameters. As a toy example let's fit the distribution to
    samples drawn from a known von Mises-Fisher distribution.

    >>> mu, kappa = np.array([0, 0, 1]), 20
    >>> samples = vonmises_fisher(mu, kappa).rvs(1000, random_state=rng)
    >>> mu_fit, kappa_fit = vonmises_fisher.fit(samples)
    >>> mu_fit, kappa_fit
    (array([0.01126519, 0.01044501, 0.99988199]), 19.306398751730995)

    We see that the estimated parameters `mu_fit` and `kappa_fit` are
    very close to the ground truth parameters.

    """
    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, mu=None, kappa=1, seed=None):
        """Create a frozen von Mises-Fisher distribution.

        See `vonmises_fisher_frozen` for more information.
        """
        return vonmises_fisher_frozen(mu, kappa, seed=seed)

    def _process_parameters(self, mu, kappa):
        """
        Infer dimensionality from mu and ensure that mu is a one-dimensional
        unit vector and kappa positive.
        """
        mu = np.asarray(mu)
        if mu.ndim > 1:
            raise ValueError("'mu' must have one-dimensional shape.")
        if not np.allclose(np.linalg.norm(mu), 1.):
            raise ValueError("'mu' must be a unit vector of norm 1.")
        if not mu.size > 1:
            raise ValueError("'mu' must have at least two entries.")
        kappa_error_msg = "'kappa' must be a positive scalar."
        if not np.isscalar(kappa) or kappa < 0:
            raise ValueError(kappa_error_msg)
        if float(kappa) == 0.:
            raise ValueError("For 'kappa=0' the von Mises-Fisher distribution "
                             "becomes the uniform distribution on the sphere "
                             "surface. Consider using "
                             "'scipy.stats.uniform_direction' instead.")
        dim = mu.size

        return dim, mu, kappa

    def _check_data_vs_dist(self, x, dim):
        if x.shape[-1] != dim:
            raise ValueError("The dimensionality of the last axis of 'x' must "
                             "match the dimensionality of the "
                             "von Mises Fisher distribution.")
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)

    def _log_norm_factor(self, dim, kappa):
        # normalization factor is given by
        # c = kappa**(dim/2-1)/((2*pi)**(dim/2)*I[dim/2-1](kappa))
        #   = kappa**(dim/2-1)*exp(-kappa) /
        #     ((2*pi)**(dim/2)*I[dim/2-1](kappa)*exp(-kappa)
        #   = kappa**(dim/2-1)*exp(-kappa) /
        #     ((2*pi)**(dim/2)*ive[dim/2-1](kappa)
        # Then the log is given by
        # log c = 1/2*(dim -1)*log(kappa) - kappa - -1/2*dim*ln(2*pi) -
        #         ive[dim/2-1](kappa)
        halfdim = 0.5 * dim
        return (0.5 * (dim - 2)*np.log(kappa) - halfdim * _LOG_2PI -
                np.log(ive(halfdim - 1, kappa)) - kappa)

    def _logpdf(self, x, dim, mu, kappa):
        """Log of the von Mises-Fisher probability density function.

        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        x = np.asarray(x)
        self._check_data_vs_dist(x, dim)
        dotproducts = np.einsum('i,...i->...', mu, x)
        return self._log_norm_factor(dim, kappa) + kappa * dotproducts

    def logpdf(self, x, mu=None, kappa=1):
        """Log of the von Mises-Fisher probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.
        mu : array_like, default: None
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float, default: 1
            Concentration parameter. Must be positive.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of the probability density function evaluated at `x`.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        return self._logpdf(x, dim, mu, kappa)

    def pdf(self, x, mu=None, kappa=1):
        """Von Mises-Fisher probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.
        mu : array_like
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float
            Concentration parameter. Must be positive.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        return np.exp(self._logpdf(x, dim, mu, kappa))

    def _rvs_2d(self, mu, kappa, size, random_state):
        """
        In 2D, the von Mises-Fisher distribution reduces to the
        von Mises distribution which can be efficiently sampled by numpy.
        This method is much faster than the general rejection
        sampling based algorithm.

        """
        mean_angle = np.arctan2(mu[1], mu[0])
        angle_samples = random_state.vonmises(mean_angle, kappa, size=size)
        samples = np.stack([np.cos(angle_samples), np.sin(angle_samples)],
                           axis=-1)
        return samples

    def _rvs_3d(self, kappa, size, random_state):
        """
        Generate samples from a von Mises-Fisher distribution
        with mu = [1, 0, 0] and kappa. Samples then have to be
        rotated towards the desired mean direction mu.
        This method is much faster than the general rejection
        sampling based algorithm.
        Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

        """
        if size is None:
            sample_size = 1
        else:
            sample_size = size

        # compute x coordinate acc. to equation from section 3.1
        x = random_state.random(sample_size)
        x = 1. + np.log(x + (1. - x) * np.exp(-2 * kappa))/kappa

        # (y, z) are random 2D vectors that only have to be
        # normalized accordingly. Then (x, y z) follow a VMF distribution
        temp = np.sqrt(1. - np.square(x))
        uniformcircle = _sample_uniform_direction(2, sample_size, random_state)
        samples = np.stack([x, temp * uniformcircle[..., 0],
                            temp * uniformcircle[..., 1]],
                           axis=-1)
        if size is None:
            samples = np.squeeze(samples)
        return samples

    def _rejection_sampling(self, dim, kappa, size, random_state):
        """
        Generate samples from a n-dimensional von Mises-Fisher distribution
        with mu = [1, 0, ..., 0] and kappa via rejection sampling.
        Samples then have to be rotated towards the desired mean direction mu.
        Reference: https://doi.org/10.1080/03610919408813161
        """
        dim_minus_one = dim - 1
        # calculate number of requested samples
        if size is not None:
            if not np.iterable(size):
                size = (size, )
            n_samples = math.prod(size)
        else:
            n_samples = 1
        # calculate envelope for rejection sampler (eq. 4)
        sqrt = np.sqrt(4 * kappa ** 2. + dim_minus_one ** 2)
        envelop_param = (-2 * kappa + sqrt) / dim_minus_one
        if envelop_param == 0:
            # the regular formula suffers from loss of precision for high
            # kappa. This can only be detected by checking for 0 here.
            # Workaround: expansion for sqrt variable
            # https://www.wolframalpha.com/input?i=sqrt%284*x%5E2%2Bd%5E2%29
            # e = (-2 * k + sqrt(k**2 + d**2)) / d
            #   ~ (-2 * k + 2 * k + d**2/(4 * k) - d**4/(64 * k**3)) / d
            #   = d/(4 * k) - d**3/(64 * k**3)
            envelop_param = (dim_minus_one/4 * kappa**-1.
                             - dim_minus_one**3/64 * kappa**-3.)
        # reference step 0
        node = (1. - envelop_param) / (1. + envelop_param)
        # t = ln(1 - ((1-x)/(1+x))**2)
        #   = ln(4 * x / (1+x)**2)
        #   = ln(4) + ln(x) - 2*log1p(x)
        correction = (kappa * node + dim_minus_one
                      * (np.log(4) + np.log(envelop_param)
                      - 2 * np.log1p(envelop_param)))
        n_accepted = 0
        x = np.zeros((n_samples, ))
        halfdim = 0.5 * dim_minus_one
        # main loop
        while n_accepted < n_samples:
            # generate candidates acc. to reference step 1
            sym_beta = random_state.beta(halfdim, halfdim,
                                         size=n_samples - n_accepted)
            coord_x = (1 - (1 + envelop_param) * sym_beta) / (
                1 - (1 - envelop_param) * sym_beta)
            # accept or reject: reference step 2
            # reformulation for numerical stability:
            # t = ln(1 - (1-x)/(1+x) * y)
            #   = ln((1 + x - y +x*y)/(1 +x))
            accept_tol = random_state.random(n_samples - n_accepted)
            criterion = (
                kappa * coord_x
                + dim_minus_one * (np.log((1 + envelop_param - coord_x
                + coord_x * envelop_param) / (1 + envelop_param)))
                - correction) > np.log(accept_tol)
            accepted_iter = np.sum(criterion)
            x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
            n_accepted += accepted_iter
        # concatenate x and remaining coordinates: step 3
        coord_rest = _sample_uniform_direction(dim_minus_one, n_accepted,
                                               random_state)
        coord_rest = np.einsum(
            '...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
        samples = np.concatenate([x[..., None], coord_rest], axis=1)
        # reshape output to (size, dim)
        if size is not None:
            samples = samples.reshape(size + (dim, ))
        else:
            samples = np.squeeze(samples)
        return samples

    def _rotate_samples(self, samples, mu, dim):
        """A QR decomposition is used to find the rotation that maps the
        north pole (1, 0,...,0) to the vector mu. This rotation is then
        applied to all samples.

        Parameters
        ----------
        samples: array_like, shape = [..., n]
        mu : array-like, shape=[n, ]
            Point to parametrise the rotation.

        Returns
        -------
        samples : rotated samples

        """
        base_point = np.zeros((dim, ))
        base_point[0] = 1.
        embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])
        rotmatrix, _ = np.linalg.qr(np.transpose(embedded))
        if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], mu):
            rotsign = 1
        else:
            rotsign = -1

        # apply rotation
        samples = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign
        return samples

    def _rvs(self, dim, mu, kappa, size, random_state):
        if dim == 2:
            samples = self._rvs_2d(mu, kappa, size, random_state)
        elif dim == 3:
            samples = self._rvs_3d(kappa, size, random_state)
        else:
            samples = self._rejection_sampling(dim, kappa, size,
                                               random_state)

        if dim != 2:
            samples = self._rotate_samples(samples, mu, dim)
        return samples

    def rvs(self, mu=None, kappa=1, size=1, random_state=None):
        """Draw random samples from a von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array_like
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float
            Concentration parameter. Must be positive.
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, np.random.RandomState, np.random.Generator},
                        optional
            Used for drawing random variates.
            If `seed` is `None`, the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is `None`.

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`, `N`), where `N` is the
            dimension of the distribution.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        random_state = self._get_random_state(random_state)
        samples = self._rvs(dim, mu, kappa, size, random_state)
        return samples

    def _entropy(self, dim, kappa):
        halfdim = 0.5 * dim
        return (-self._log_norm_factor(dim, kappa) - kappa *
                ive(halfdim, kappa) / ive(halfdim - 1, kappa))

    def entropy(self, mu=None, kappa=1):
        """Compute the differential entropy of the von Mises-Fisher
        distribution.

        Parameters
        ----------
        mu : array_like, default: None
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float, default: 1
            Concentration parameter. Must be positive.

        Returns
        -------
        h : scalar
            Entropy of the von Mises-Fisher distribution.

        """
        dim, _, kappa = self._process_parameters(mu, kappa)
        return self._entropy(dim, kappa)

    def fit(self, x):
        """Fit the von Mises-Fisher distribution to data.

        Parameters
        ----------
        x : array-like
            Data the distribution is fitted to. Must be two dimensional.
            The second axis of `x` must be unit vectors of norm 1 and
            determine the dimensionality of the fitted
            von Mises-Fisher distribution.

        Returns
        -------
        mu : ndarray
            Estimated mean direction.
        kappa : float
            Estimated concentration parameter.

        """
        # validate input data
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("'x' must be two dimensional.")
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)
        dim = x.shape[-1]

        # mu is simply the directional mean
        dirstats = directional_stats(x)
        mu = dirstats.mean_direction
        r = dirstats.mean_resultant_length

        # kappa is the solution to the equation:
        # r = I[dim/2](kappa) / I[dim/2 -1](kappa)
        #   = I[dim/2](kappa) * exp(-kappa) / I[dim/2 -1](kappa) * exp(-kappa)
        #   = ive(dim/2, kappa) / ive(dim/2 -1, kappa)

        halfdim = 0.5 * dim

        def solve_for_kappa(kappa):
            bessel_vals = ive([halfdim, halfdim - 1], kappa)
            return bessel_vals[0]/bessel_vals[1] - r

        root_res = root_scalar(solve_for_kappa, method="brentq",
                               bracket=(1e-8, 1e9))
        kappa = root_res.root
        return mu, kappa


vonmises_fisher = vonmises_fisher_gen()


class vonmises_fisher_frozen(multi_rv_frozen):
    def __init__(self, mu=None, kappa=1, seed=None):
        """Create a frozen von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array_like, default: None
            Mean direction of the distribution.
        kappa : float, default: 1
            Concentration parameter. Must be positive.
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        """
        self._dist = vonmises_fisher_gen(seed)
        self.dim, self.mu, self.kappa = (
            self._dist._process_parameters(mu, kappa)
        )

    def logpdf(self, x):
        """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of probability density function evaluated at `x`.

        """
        return self._dist._logpdf(x, self.dim, self.mu, self.kappa)

    def pdf(self, x):
        """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.

        """
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """Draw random variates from the Von Mises-Fisher distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the distribution.

        """
        random_state = self._dist._get_random_state(random_state)
        return self._dist._rvs(self.dim, self.mu, self.kappa, size,
                               random_state)

    def entropy(self):
        """
        Calculate the differential entropy of the von Mises-Fisher
        distribution.

        Returns
        -------
        h: float
            Entropy of the Von Mises-Fisher distribution.

        """
        return self._dist._entropy(self.dim, self.kappa)
