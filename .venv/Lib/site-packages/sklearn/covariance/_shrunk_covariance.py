"""
Covariance estimators using shrinkage.

Shrinkage corresponds to regularising `cov` using a convex combination:
shrunk_cov = (1-shrinkage)*cov + shrinkage*structured_estimate.

"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD 3 clause

# avoid division truncation
import warnings
from numbers import Integral, Real

import numpy as np

from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance


def _ledoit_wolf(X, *, assume_centered, block_size):
    """Estimate the shrunk Ledoit-Wolf covariance matrix."""
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X**2).mean()), 0.0
    n_features = X.shape[1]

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, block_size=block_size
    )
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp_cov)) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage


def _oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(X.shape) == 2 and X.shape[1] == 1:
        # for only one feature, the result is the same whatever the shrinkage
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X**2).mean()), 0.0

    n_samples, n_features = X.shape

    emp_cov = empirical_covariance(X, assume_centered=assume_centered)

    # The shrinkage is defined as:
    # shrinkage = min(
    # trace(S @ S.T) + trace(S)**2) / ((n + 1) (trace(S @ S.T) - trace(S)**2 / p), 1
    # )
    # where n and p are n_samples and n_features, respectively (cf. Eq. 23 in [1]).
    # The factor 2 / p is omitted since it does not impact the value of the estimator
    # for large p.

    # Instead of computing trace(S)**2, we can compute the average of the squared
    # elements of S that is equal to trace(S)**2 / p**2.
    # See the definition of the Frobenius norm:
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    alpha = np.mean(emp_cov**2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu**2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    # The shrunk covariance is defined as:
    # (1 - shrinkage) * S + shrinkage * F (cf. Eq. 4 in [1])
    # where S is the empirical covariance and F is the shrinkage target defined as
    # F = trace(S) / n_features * np.identity(n_features) (cf. Eq. 3 in [1])
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage


###############################################################################
# Public API
# ShrunkCovariance estimator


@validate_params(
    {
        "emp_cov": ["array-like"],
        "shrinkage": [Interval(Real, 0, 1, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def shrunk_covariance(emp_cov, shrinkage=0.1):
    """Calculate covariance matrices shrunk on the diagonal.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    emp_cov : array-like of shape (..., n_features, n_features)
        Covariance matrices to be shrunk, at least 2D ndarray.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Returns
    -------
    shrunk_cov : ndarray of shape (..., n_features, n_features)
        Shrunk covariance matrices.

    Notes
    -----
    The regularized (shrunk) covariance is given by::

        (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where `mu = trace(cov) / n_features`.
    """
    emp_cov = check_array(emp_cov, allow_nd=True)
    n_features = emp_cov.shape[-1]

    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mu = np.trace(emp_cov, axis1=-2, axis2=-1) / n_features
    mu = np.expand_dims(mu, axis=tuple(range(mu.ndim, emp_cov.ndim)))
    shrunk_cov += shrinkage * mu * np.eye(n_features)

    return shrunk_cov


class ShrunkCovariance(EmpiricalCovariance):
    """Covariance estimator with shrinkage.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.

    Notes
    -----
    The regularized covariance is given by:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import ShrunkCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=500)
    >>> cov = ShrunkCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7387..., 0.2536...],
           [0.2536..., 0.4110...]])
    >>> cov.location_
    array([0.0622..., 0.0193...])
    """

    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "shrinkage": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, *, store_precision=True, assume_centered=False, shrinkage=0.1):
        super().__init__(
            store_precision=store_precision, assume_centered=assume_centered
        )
        self.shrinkage = shrinkage

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        # Not calling the parent object to fit, to avoid a potential
        # matrix inversion when setting the precision
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        covariance = shrunk_covariance(covariance, self.shrinkage)
        self._set_covariance(covariance)

        return self


# Ledoit-Wolf estimator


@validate_params(
    {
        "X": ["array-like"],
        "assume_centered": ["boolean"],
        "block_size": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    """
    X = check_array(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X**2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = 0.0  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.0  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits :]))
        delta_ += np.sum(np.dot(X.T[rows], X[:, block_size * n_splits :]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits :], X2[:, cols]))
        delta_ += np.sum(np.dot(X.T[block_size * n_splits :], X[:, cols]) ** 2)
    delta_ += np.sum(
        np.dot(X.T[block_size * n_splits :], X[:, block_size * n_splits :]) ** 2
    )
    delta_ /= n_samples**2
    beta_ += np.sum(
        np.dot(X2.T[block_size * n_splits :], X2[:, block_size * n_splits :])
    )
    # use delta_ to compute beta
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage


@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
def ledoit_wolf(X, *, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    """
    estimator = LedoitWolf(
        assume_centered=assume_centered,
        block_size=block_size,
        store_precision=False,
    ).fit(X)

    return estimator.covariance_, estimator.shrinkage_


class LedoitWolf(EmpiricalCovariance):
    """LedoitWolf Estimator.

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import LedoitWolf
    >>> real_cov = np.array([[.4, .2],
    ...                      [.2, .8]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=50)
    >>> cov = LedoitWolf().fit(X)
    >>> cov.covariance_
    array([[0.4406..., 0.1616...],
           [0.1616..., 0.8022...]])
    >>> cov.location_
    array([ 0.0595... , -0.0075...])
    """

    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "block_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, *, store_precision=True, assume_centered=False, block_size=1000):
        super().__init__(
            store_precision=store_precision, assume_centered=assume_centered
        )
        self.block_size = block_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Not calling the parent object to fit, to avoid computing the
        # covariance matrix (and potentially the precision)
        X = self._validate_data(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance, shrinkage = _ledoit_wolf(
            X - self.location_, assume_centered=True, block_size=self.block_size
        )
        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self


# OAS estimator
@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
def oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage as proposed in [1]_.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : array-like of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`
    """
    estimator = OAS(
        assume_centered=assume_centered,
    ).fit(X)
    return estimator.covariance_, estimator.shrinkage_


class OAS(EmpiricalCovariance):
    """Oracle Approximating Shrinkage Estimator as proposed in [1]_.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
      coefficient in the convex combination used for the computation
      of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    EmpiricalCovariance : Maximum likelihood covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with cross-validated
        choice of the l1 penalty.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    ShrunkCovariance : Covariance estimator with shrinkage.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import OAS
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> oas = OAS().fit(X)
    >>> oas.covariance_
    array([[0.7533..., 0.2763...],
           [0.2763..., 0.3964...]])
    >>> oas.precision_
    array([[ 1.7833..., -1.2431... ],
           [-1.2431...,  3.3889...]])
    >>> oas.shrinkage_
    0.0195...
    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Oracle Approximating Shrinkage covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        # Not calling the parent object to fit, to avoid computing the
        # covariance matrix (and potentially the precision)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        covariance, shrinkage = _oas(X - self.location_, assume_centered=True)
        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self
