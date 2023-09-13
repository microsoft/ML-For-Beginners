"""
This file contains preprocessing tools based on polynomials.
"""
import collections
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import comb

from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    _check_sample_weight,
    check_is_fitted,
)
from ._csr_polynomial_expansion import (
    _calc_expanded_nnz,
    _calc_total_nnz,
    _csr_polynomial_expansion,
)

__all__ = [
    "PolynomialFeatures",
    "SplineTransformer",
]


def _create_expansion(X, interaction_only, deg, n_features, cumulative_size=0):
    """Helper function for creating and appending sparse expansion matrices"""

    total_nnz = _calc_total_nnz(X.indptr, interaction_only, deg)
    expanded_col = _calc_expanded_nnz(n_features, interaction_only, deg)

    if expanded_col == 0:
        return None
    # This only checks whether each block needs 64bit integers upon
    # expansion. We prefer to keep int32 indexing where we can,
    # since currently SciPy's CSR construction downcasts when possible,
    # so we prefer to avoid an unnecessary cast. The dtype may still
    # change in the concatenation process if needed.
    # See: https://github.com/scipy/scipy/issues/16569
    max_indices = expanded_col - 1
    max_indptr = total_nnz
    max_int32 = np.iinfo(np.int32).max
    needs_int64 = max(max_indices, max_indptr) > max_int32
    index_dtype = np.int64 if needs_int64 else np.int32

    # This is a pretty specific bug that is hard to work around by a user,
    # hence we do not detail the entire bug and all possible avoidance
    # mechnasisms. Instead we recommend upgrading scipy or shrinking their data.
    cumulative_size += expanded_col
    if (
        sp_version < parse_version("1.8.0")
        and cumulative_size - 1 > max_int32
        and not needs_int64
    ):
        raise ValueError(
            "In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
            " sometimes produces negative columns when the output shape contains"
            " `n_cols` too large to be represented by a 32bit signed"
            " integer. To avoid this error, either use a version"
            " of scipy `>=1.8.0` or alter the `PolynomialFeatures`"
            " transformer to produce fewer than 2^31 output features."
        )

    # Result of the expansion, modified in place by the
    # `_csr_polynomial_expansion` routine.
    expanded_data = np.empty(shape=total_nnz, dtype=X.data.dtype)
    expanded_indices = np.empty(shape=total_nnz, dtype=index_dtype)
    expanded_indptr = np.empty(shape=X.indptr.shape[0], dtype=index_dtype)
    _csr_polynomial_expansion(
        X.data,
        X.indices,
        X.indptr,
        X.shape[1],
        expanded_data,
        expanded_indices,
        expanded_indptr,
        interaction_only,
        deg,
    )
    return sparse.csr_matrix(
        (expanded_data, expanded_indices, expanded_indptr),
        shape=(X.indptr.shape[0] - 1, expanded_col),
        dtype=X.dtype,
    )


class PolynomialFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Read more in the :ref:`User Guide <polynomial_features>`.

    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        If a single int is given, it specifies the maximal degree of the
        polynomial features. If a tuple `(min_degree, max_degree)` is passed,
        then `min_degree` is the minimum and `max_degree` is the maximum
        polynomial degree of the generated features. Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.

    interaction_only : bool, default=False
        If `True`, only interaction features are produced: features that are
        products of at most `degree` *distinct* input features, i.e. terms with
        power of 2 or higher of the same input feature are excluded:

            - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
            - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.

    include_bias : bool, default=True
        If `True` (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. `'F'` order is faster to
        compute, but may slow down subsequent estimators.

        .. versionadded:: 0.21

    Attributes
    ----------
    powers_ : ndarray of shape (`n_output_features_`, `n_features_in_`)
        `powers_[i, j]` is the exponent of the jth input in the ith output.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.

    See Also
    --------
    SplineTransformer : Transformer that generates univariate B-spline bases
        for features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.

    See :ref:`examples/linear_model/plot_polynomial_interpolation.py
    <sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])
    >>> poly = PolynomialFeatures(interaction_only=True)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.],
           [ 1.,  2.,  3.,  6.],
           [ 1.,  4.,  5., 20.]])
    """

    _parameter_constraints: dict = {
        "degree": [Interval(Integral, 0, None, closed="left"), "array-like"],
        "interaction_only": ["boolean"],
        "include_bias": ["boolean"],
        "order": [StrOptions({"C", "F"})],
    }

    def __init__(
        self, degree=2, *, interaction_only=False, include_bias=True, order="C"
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    @staticmethod
    def _combinations(
        n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        comb = combinations if interaction_only else combinations_w_r
        start = max(1, min_degree)
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        if include_bias:
            iter = chain(comb(range(n_features), 0), iter)
        return iter

    @staticmethod
    def _num_combinations(
        n_features, min_degree, max_degree, interaction_only, include_bias
    ):
        """Calculate number of terms in polynomial expansion

        This should be equivalent to counting the number of terms returned by
        _combinations(...) but much faster.
        """

        if interaction_only:
            combinations = sum(
                [
                    comb(n_features, i, exact=True)
                    for i in range(max(1, min_degree), min(max_degree, n_features) + 1)
                ]
            )
        else:
            combinations = comb(n_features + max_degree, max_degree, exact=True) - 1
            if min_degree > 0:
                d = min_degree - 1
                combinations -= comb(n_features + d, d, exact=True) - 1

        if include_bias:
            combinations += 1

        return combinations

    @property
    def powers_(self):
        """Exponent for each of the inputs in the output."""
        check_is_fitted(self)

        combinations = self._combinations(
            n_features=self.n_features_in_,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        return np.vstack(
            [np.bincount(c, minlength=self.n_features_in_) for c in combinations]
        )

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features is None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        powers = self.powers_
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    (
                        "%s^%d" % (input_features[ind], exp)
                        if exp != 1
                        else input_features[ind]
                    )
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.asarray(feature_names, dtype=object)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        _, n_features = self._validate_data(X, accept_sparse=True).shape

        if isinstance(self.degree, Integral):
            if self.degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting degree to zero and include_bias to False would result in"
                    " an empty output array."
                )

            self._min_degree = 0
            self._max_degree = self.degree
        elif (
            isinstance(self.degree, collections.abc.Iterable) and len(self.degree) == 2
        ):
            self._min_degree, self._max_degree = self.degree
            if not (
                isinstance(self._min_degree, Integral)
                and isinstance(self._max_degree, Integral)
                and self._min_degree >= 0
                and self._min_degree <= self._max_degree
            ):
                raise ValueError(
                    "degree=(min_degree, max_degree) must "
                    "be non-negative integers that fulfil "
                    "min_degree <= max_degree, got "
                    f"{self.degree}."
                )
            elif self._max_degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting both min_degree and max_degree to zero and include_bias to"
                    " False would result in an empty output array."
                )
        else:
            raise ValueError(
                "degree must be a non-negative int or tuple "
                "(min_degree, max_degree), got "
                f"{self.degree}."
            )

        self.n_output_features_ = self._num_combinations(
            n_features=n_features,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        if self.n_output_features_ > np.iinfo(np.intp).max:
            msg = (
                "The output that would result from the current configuration would"
                f" have {self.n_output_features_} features which is too large to be"
                f" indexed by {np.intp().dtype.name}. Please change some or all of the"
                " following:\n- The number of features in the input, currently"
                f" {n_features=}\n- The range of degrees to calculate, currently"
                f" [{self._min_degree}, {self._max_degree}]\n- Whether to include only"
                f" interaction terms, currently {self.interaction_only}\n- Whether to"
                f" include a bias term, currently {self.include_bias}."
            )
            if (
                np.intp == np.int32
                and self.n_output_features_ <= np.iinfo(np.int64).max
            ):  # pragma: nocover
                msg += (
                    "\nNote that the current Python runtime has a limited 32 bit "
                    "address space and that this configuration would have been "
                    "admissible if run on a 64 bit Python runtime."
                )
            raise ValueError(msg)
        # We also record the number of output features for
        # _max_degree = 0
        self._n_out_full = self._num_combinations(
            n_features=n_features,
            min_degree=0,
            max_degree=self._max_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        return self

    def transform(self, X):
        """Transform data to polynomial features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform, row by row.

            Prefer CSR over CSC for sparse input (for speed), but CSC is
            required if the degree is 4 or higher. If the degree is less than
            4 and the input format is CSC, it will be converted to CSR, have
            its polynomial features generated, then converted back to CSC.

            If the degree is 2 or 3, the method described in "Leveraging
            Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
            Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
            used, which is much faster than the method used on CSC input. For
            this reason, a CSC input will be converted to CSR, and the output
            will be converted back to CSC prior to being returned, hence the
            preference of CSR.

        Returns
        -------
        XP : {ndarray, sparse matrix} of shape (n_samples, NP)
            The matrix of features, where `NP` is the number of polynomial
            features generated from the combination of inputs. If a sparse
            matrix is provided, it will be converted into a sparse
            `csr_matrix`.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X, order="F", dtype=FLOAT_DTYPES, reset=False, accept_sparse=("csr", "csc")
        )

        n_samples, n_features = X.shape
        max_int32 = np.iinfo(np.int32).max
        if sparse.isspmatrix_csr(X):
            if self._max_degree > 3:
                return self.transform(X.tocsc()).tocsr()
            to_stack = []
            if self.include_bias:
                to_stack.append(
                    sparse.csr_matrix(np.ones(shape=(n_samples, 1), dtype=X.dtype))
                )
            if self._min_degree <= 1 and self._max_degree > 0:
                to_stack.append(X)

            cumulative_size = sum(mat.shape[1] for mat in to_stack)
            for deg in range(max(2, self._min_degree), self._max_degree + 1):
                expanded = _create_expansion(
                    X=X,
                    interaction_only=self.interaction_only,
                    deg=deg,
                    n_features=n_features,
                    cumulative_size=cumulative_size,
                )
                if expanded is not None:
                    to_stack.append(expanded)
                    cumulative_size += expanded.shape[1]
            if len(to_stack) == 0:
                # edge case: deal with empty matrix
                XP = sparse.csr_matrix((n_samples, 0), dtype=X.dtype)
            else:
                # `scipy.sparse.hstack` breaks in scipy<1.9.2
                # when `n_output_features_ > max_int32`
                all_int32 = all(mat.indices.dtype == np.int32 for mat in to_stack)
                if (
                    sp_version < parse_version("1.9.2")
                    and self.n_output_features_ > max_int32
                    and all_int32
                ):
                    raise ValueError(  # pragma: no cover
                        "In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`"
                        " produces negative columns when:\n1. The output shape contains"
                        " `n_cols` too large to be represented by a 32bit signed"
                        " integer.\n2. All sub-matrices to be stacked have indices of"
                        " dtype `np.int32`.\nTo avoid this error, either use a version"
                        " of scipy `>=1.9.2` or alter the `PolynomialFeatures`"
                        " transformer to produce fewer than 2^31 output features"
                    )
                XP = sparse.hstack(to_stack, dtype=X.dtype, format="csr")
        elif sparse.isspmatrix_csc(X) and self._max_degree < 4:
            return self.transform(X.tocsr()).tocsc()
        elif sparse.isspmatrix(X):
            combinations = self._combinations(
                n_features=n_features,
                min_degree=self._min_degree,
                max_degree=self._max_degree,
                interaction_only=self.interaction_only,
                include_bias=self.include_bias,
            )
            columns = []
            for combi in combinations:
                if combi:
                    out_col = 1
                    for col_idx in combi:
                        out_col = X[:, col_idx].multiply(out_col)
                    columns.append(out_col)
                else:
                    bias = sparse.csc_matrix(np.ones((X.shape[0], 1)))
                    columns.append(bias)
            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
        else:
            # Do as if _min_degree = 0 and cut down array after the
            # computation, i.e. use _n_out_full instead of n_output_features_.
            XP = np.empty(
                shape=(n_samples, self._n_out_full), dtype=X.dtype, order=self.order
            )

            # What follows is a faster implementation of:
            # for i, comb in enumerate(combinations):
            #     XP[:, i] = X[:, comb].prod(1)
            # This implementation uses two optimisations.
            # First one is broadcasting,
            # multiply ([X1, ..., Xn], X1) -> [X1 X1, ..., Xn X1]
            # multiply ([X2, ..., Xn], X2) -> [X2 X2, ..., Xn X2]
            # ...
            # multiply ([X[:, start:end], X[:, start]) -> ...
            # Second optimisation happens for degrees >= 3.
            # Xi^3 is computed reusing previous computation:
            # Xi^3 = Xi^2 * Xi.

            # degree 0 term
            if self.include_bias:
                XP[:, 0] = 1
                current_col = 1
            else:
                current_col = 0

            if self._max_degree == 0:
                return XP

            # degree 1 term
            XP[:, current_col : current_col + n_features] = X
            index = list(range(current_col, current_col + n_features))
            current_col += n_features
            index.append(current_col)

            # loop over degree >= 2 terms
            for _ in range(2, self._max_degree + 1):
                new_index = []
                end = index[-1]
                for feature_idx in range(n_features):
                    start = index[feature_idx]
                    new_index.append(current_col)
                    if self.interaction_only:
                        start += index[feature_idx + 1] - index[feature_idx]
                    next_col = current_col + end - start
                    if next_col <= current_col:
                        break
                    # XP[:, start:end] are terms of degree d - 1
                    # that exclude feature #feature_idx.
                    np.multiply(
                        XP[:, start:end],
                        X[:, feature_idx : feature_idx + 1],
                        out=XP[:, current_col:next_col],
                        casting="no",
                    )
                    current_col = next_col

                new_index.append(current_col)
                index = new_index

            if self._min_degree > 1:
                n_XP, n_Xout = self._n_out_full, self.n_output_features_
                if self.include_bias:
                    Xout = np.empty(
                        shape=(n_samples, n_Xout), dtype=XP.dtype, order=self.order
                    )
                    Xout[:, 0] = 1
                    Xout[:, 1:] = XP[:, n_XP - n_Xout + 1 :]
                else:
                    Xout = XP[:, n_XP - n_Xout :].copy()
                XP = Xout
        return XP


class SplineTransformer(TransformerMixin, BaseEstimator):
    """Generate univariate B-spline bases for features.

    Generate a new feature matrix consisting of
    `n_splines=n_knots + degree - 1` (`n_knots - 1` for
    `extrapolation="periodic"`) spline basis functions
    (B-splines) of polynomial order=`degree` for each feature.

    Read more in the :ref:`User Guide <spline_transformer>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots of the splines if `knots` equals one of
        {'uniform', 'quantile'}. Must be larger or equal 2. Ignored if `knots`
        is array-like.

    degree : int, default=3
        The polynomial degree of the spline basis. Must be a non-negative
        integer.

    knots : {'uniform', 'quantile'} or array-like of shape \
        (n_knots, n_features), default='uniform'
        Set knot positions such that first knot <= features <= last knot.

        - If 'uniform', `n_knots` number of knots are distributed uniformly
          from min to max values of the features.
        - If 'quantile', they are distributed uniformly along the quantiles of
          the features.
        - If an array-like is given, it directly specifies the sorted knot
          positions including the boundary knots. Note that, internally,
          `degree` number of knots are added before the first knot, the same
          after the last knot.

    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'}, \
        default='constant'
        If 'error', values outside the min and max values of the training
        features raises a `ValueError`. If 'constant', the value of the
        splines at minimum and maximum value of the features is used as
        constant extrapolation. If 'linear', a linear extrapolation is used.
        If 'continue', the splines are extrapolated as is, i.e. option
        `extrapolate=True` in :class:`scipy.interpolate.BSpline`. If
        'periodic', periodic splines with a periodicity equal to the distance
        between the first and last knot are used. Periodic splines enforce
        equal function values and derivatives at the first and last knot.
        For example, this makes it possible to avoid introducing an arbitrary
        jump between Dec 31st and Jan 1st in spline features derived from a
        naturally periodic "day-of-year" input feature. In this case it is
        recommended to manually set the knot values to control the period.

    include_bias : bool, default=True
        If False, then the last spline element inside the data range
        of a feature is dropped. As B-splines sum to one over the spline basis
        functions for each data point, they implicitly include a bias term,
        i.e. a column of ones. It acts as an intercept term in a linear models.

    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. `'F'` order is faster to compute, but
        may slow down subsequent estimators.

    sparse_output : bool, default=False
        Will return sparse CSR matrix if set True else will return an array. This
        option is only available with `scipy>=1.8`.

        .. versionadded:: 1.2

    Attributes
    ----------
    bsplines_ : list of shape (n_features,)
        List of BSplines objects, one for each feature.

    n_features_in_ : int
        The total number of input features.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_out_ : int
        The total number of output features, which is computed as
        `n_features * n_splines`, where `n_splines` is
        the number of bases elements of the B-splines,
        `n_knots + degree - 1` for non-periodic splines and
        `n_knots - 1` for periodic ones.
        If `include_bias=False`, then it is only
        `n_features * (n_splines - 1)`.

    See Also
    --------
    KBinsDiscretizer : Transformer that bins continuous data into intervals.

    PolynomialFeatures : Transformer that generates polynomial and interaction
        features.

    Notes
    -----
    High degrees and a high number of knots can cause overfitting.

    See :ref:`examples/linear_model/plot_polynomial_interpolation.py
    <sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import SplineTransformer
    >>> X = np.arange(6).reshape(6, 1)
    >>> spline = SplineTransformer(degree=2, n_knots=3)
    >>> spline.fit_transform(X)
    array([[0.5 , 0.5 , 0.  , 0.  ],
           [0.18, 0.74, 0.08, 0.  ],
           [0.02, 0.66, 0.32, 0.  ],
           [0.  , 0.32, 0.66, 0.02],
           [0.  , 0.08, 0.74, 0.18],
           [0.  , 0.  , 0.5 , 0.5 ]])
    """

    _parameter_constraints: dict = {
        "n_knots": [Interval(Integral, 2, None, closed="left")],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "knots": [StrOptions({"uniform", "quantile"}), "array-like"],
        "extrapolation": [
            StrOptions({"error", "constant", "linear", "continue", "periodic"})
        ],
        "include_bias": ["boolean"],
        "order": [StrOptions({"C", "F"})],
        "sparse_output": ["boolean"],
    }

    def __init__(
        self,
        n_knots=5,
        degree=3,
        *,
        knots="uniform",
        extrapolation="constant",
        include_bias=True,
        order="C",
        sparse_output=False,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.sparse_output = sparse_output

    @staticmethod
    def _get_base_knot_positions(X, n_knots=10, knots="uniform", sample_weight=None):
        """Calculate base knot positions.

        Base knots such that first knot <= feature <= last knot. For the
        B-spline construction with scipy.interpolate.BSpline, 2*degree knots
        beyond the base interval are added.

        Returns
        -------
        knots : ndarray of shape (n_knots, n_features), dtype=np.float64
            Knot positions (points) of base interval.
        """
        if knots == "quantile":
            percentiles = 100 * np.linspace(
                start=0, stop=1, num=n_knots, dtype=np.float64
            )

            if sample_weight is None:
                knots = np.percentile(X, percentiles, axis=0)
            else:
                knots = np.array(
                    [
                        _weighted_percentile(X, sample_weight, percentile)
                        for percentile in percentiles
                    ]
                )

        else:
            # knots == 'uniform':
            # Note that the variable `knots` has already been validated and
            # `else` is therefore safe.
            # Disregard observations with zero weight.
            mask = slice(None, None, 1) if sample_weight is None else sample_weight > 0
            x_min = np.amin(X[mask], axis=0)
            x_max = np.amax(X[mask], axis=0)

            knots = np.linspace(
                start=x_min,
                stop=x_max,
                num=n_knots,
                endpoint=True,
                dtype=np.float64,
            )

        return knots

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        n_splines = self.bsplines_[0].c.shape[1]

        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for i in range(self.n_features_in_):
            for j in range(n_splines - 1 + self.include_bias):
                feature_names.append(f"{input_features[i]}_sp_{j}")
        return np.asarray(feature_names, dtype=object)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute knot positions of splines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default = None
            Individual weights for each sample. Used to calculate quantiles if
            `knots="quantile"`. For `knots="uniform"`, zero weighted
            observations are ignored for finding the min and max of `X`.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._validate_data(
            X,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_2d=True,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        _, n_features = X.shape

        if isinstance(self.knots, str):
            base_knots = self._get_base_knot_positions(
                X, n_knots=self.n_knots, knots=self.knots, sample_weight=sample_weight
            )
        else:
            base_knots = check_array(self.knots, dtype=np.float64)
            if base_knots.shape[0] < 2:
                raise ValueError("Number of knots, knots.shape[0], must be >= 2.")
            elif base_knots.shape[1] != n_features:
                raise ValueError("knots.shape[1] == n_features is violated.")
            elif not np.all(np.diff(base_knots, axis=0) > 0):
                raise ValueError("knots must be sorted without duplicates.")

        if self.sparse_output and sp_version < parse_version("1.8.0"):
            raise ValueError(
                "Option sparse_output=True is only available with scipy>=1.8.0, "
                f"but here scipy=={sp_version} is used."
            )

        # number of knots for base interval
        n_knots = base_knots.shape[0]

        if self.extrapolation == "periodic" and n_knots <= self.degree:
            raise ValueError(
                "Periodic splines require degree < n_knots. Got n_knots="
                f"{n_knots} and degree={self.degree}."
            )

        # number of splines basis functions
        if self.extrapolation != "periodic":
            n_splines = n_knots + self.degree - 1
        else:
            # periodic splines have self.degree less degrees of freedom
            n_splines = n_knots - 1

        degree = self.degree
        n_out = n_features * n_splines
        # We have to add degree number of knots below, and degree number knots
        # above the base knots in order to make the spline basis complete.
        if self.extrapolation == "periodic":
            # For periodic splines the spacing of the first / last degree knots
            # needs to be a continuation of the spacing of the last / first
            # base knots.
            period = base_knots[-1] - base_knots[0]
            knots = np.r_[
                base_knots[-(degree + 1) : -1] - period,
                base_knots,
                base_knots[1 : (degree + 1)] + period,
            ]

        else:
            # Eilers & Marx in "Flexible smoothing with B-splines and
            # penalties" https://doi.org/10.1214/ss/1038425655 advice
            # against repeating first and last knot several times, which
            # would have inferior behaviour at boundaries if combined with
            # a penalty (hence P-Spline). We follow this advice even if our
            # splines are unpenalized. Meaning we do not:
            # knots = np.r_[
            #     np.tile(base_knots.min(axis=0), reps=[degree, 1]),
            #     base_knots,
            #     np.tile(base_knots.max(axis=0), reps=[degree, 1])
            # ]
            # Instead, we reuse the distance of the 2 fist/last knots.
            dist_min = base_knots[1] - base_knots[0]
            dist_max = base_knots[-1] - base_knots[-2]

            knots = np.r_[
                np.linspace(
                    base_knots[0] - degree * dist_min,
                    base_knots[0] - dist_min,
                    num=degree,
                ),
                base_knots,
                np.linspace(
                    base_knots[-1] + dist_max,
                    base_knots[-1] + degree * dist_max,
                    num=degree,
                ),
            ]

        # With a diagonal coefficient matrix, we get back the spline basis
        # elements, i.e. the design matrix of the spline.
        # Note, BSpline appreciates C-contiguous float64 arrays as c=coef.
        coef = np.eye(n_splines, dtype=np.float64)
        if self.extrapolation == "periodic":
            coef = np.concatenate((coef, coef[:degree, :]))

        extrapolate = self.extrapolation in ["periodic", "continue"]

        bsplines = [
            BSpline.construct_fast(
                knots[:, i], coef, self.degree, extrapolate=extrapolate
            )
            for i in range(n_features)
        ]
        self.bsplines_ = bsplines

        self.n_features_out_ = n_out - n_features * (1 - self.include_bias)
        return self

    def transform(self, X):
        """Transform each feature data to B-splines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        XBS : {ndarray, sparse matrix} of shape (n_samples, n_features * n_splines)
            The matrix of features, where n_splines is the number of bases
            elements of the B-splines, n_knots + degree - 1.
        """
        check_is_fitted(self)

        X = self._validate_data(X, reset=False, accept_sparse=False, ensure_2d=True)

        n_samples, n_features = X.shape
        n_splines = self.bsplines_[0].c.shape[1]
        degree = self.degree

        # TODO: Remove this condition, once scipy 1.10 is the minimum version.
        #       Only scipy => 1.10 supports design_matrix(.., extrapolate=..).
        #       The default (implicit in scipy < 1.10) is extrapolate=False.
        scipy_1_10 = sp_version >= parse_version("1.10.0")
        # Note: self.bsplines_[0].extrapolate is True for extrapolation in
        # ["periodic", "continue"]
        if scipy_1_10:
            use_sparse = self.sparse_output
            kwargs_extrapolate = {"extrapolate": self.bsplines_[0].extrapolate}
        else:
            use_sparse = self.sparse_output and not self.bsplines_[0].extrapolate
            kwargs_extrapolate = dict()

        # Note that scipy BSpline returns float64 arrays and converts input
        # x=X[:, i] to c-contiguous float64.
        n_out = self.n_features_out_ + n_features * (1 - self.include_bias)
        if X.dtype in FLOAT_DTYPES:
            dtype = X.dtype
        else:
            dtype = np.float64
        if use_sparse:
            output_list = []
        else:
            XBS = np.zeros((n_samples, n_out), dtype=dtype, order=self.order)

        for i in range(n_features):
            spl = self.bsplines_[i]

            if self.extrapolation in ("continue", "error", "periodic"):
                if self.extrapolation == "periodic":
                    # With periodic extrapolation we map x to the segment
                    # [spl.t[k], spl.t[n]].
                    # This is equivalent to BSpline(.., extrapolate="periodic")
                    # for scipy>=1.0.0.
                    n = spl.t.size - spl.k - 1
                    # Assign to new array to avoid inplace operation
                    x = spl.t[spl.k] + (X[:, i] - spl.t[spl.k]) % (
                        spl.t[n] - spl.t[spl.k]
                    )
                else:
                    x = X[:, i]

                if use_sparse:
                    XBS_sparse = BSpline.design_matrix(
                        x, spl.t, spl.k, **kwargs_extrapolate
                    )
                    if self.extrapolation == "periodic":
                        # See the construction of coef in fit. We need to add the last
                        # degree spline basis function to the first degree ones and
                        # then drop the last ones.
                        # Note: See comment about SparseEfficiencyWarning below.
                        XBS_sparse = XBS_sparse.tolil()
                        XBS_sparse[:, :degree] += XBS_sparse[:, -degree:]
                        XBS_sparse = XBS_sparse[:, :-degree]
                else:
                    XBS[:, (i * n_splines) : ((i + 1) * n_splines)] = spl(x)
            else:  # extrapolation in ("constant", "linear")
                xmin, xmax = spl.t[degree], spl.t[-degree - 1]
                # spline values at boundaries
                f_min, f_max = spl(xmin), spl(xmax)
                mask = (xmin <= X[:, i]) & (X[:, i] <= xmax)
                if use_sparse:
                    mask_inv = ~mask
                    x = X[:, i].copy()
                    # Set some arbitrary values outside boundary that will be reassigned
                    # later.
                    x[mask_inv] = spl.t[self.degree]
                    XBS_sparse = BSpline.design_matrix(x, spl.t, spl.k)
                    # Note: Without converting to lil_matrix we would get:
                    # scipy.sparse._base.SparseEfficiencyWarning: Changing the sparsity
                    # structure of a csr_matrix is expensive. lil_matrix is more
                    # efficient.
                    if np.any(mask_inv):
                        XBS_sparse = XBS_sparse.tolil()
                        XBS_sparse[mask_inv, :] = 0
                else:
                    XBS[mask, (i * n_splines) : ((i + 1) * n_splines)] = spl(X[mask, i])

            # Note for extrapolation:
            # 'continue' is already returned as is by scipy BSplines
            if self.extrapolation == "error":
                # BSpline with extrapolate=False does not raise an error, but
                # outputs np.nan.
                if (use_sparse and np.any(np.isnan(XBS_sparse.data))) or (
                    not use_sparse
                    and np.any(
                        np.isnan(XBS[:, (i * n_splines) : ((i + 1) * n_splines)])
                    )
                ):
                    raise ValueError(
                        "X contains values beyond the limits of the knots."
                    )
            elif self.extrapolation == "constant":
                # Set all values beyond xmin and xmax to the value of the
                # spline basis functions at those two positions.
                # Only the first degree and last degree number of splines
                # have non-zero values at the boundaries.

                mask = X[:, i] < xmin
                if np.any(mask):
                    if use_sparse:
                        # Note: See comment about SparseEfficiencyWarning above.
                        XBS_sparse = XBS_sparse.tolil()
                        XBS_sparse[mask, :degree] = f_min[:degree]

                    else:
                        XBS[mask, (i * n_splines) : (i * n_splines + degree)] = f_min[
                            :degree
                        ]

                mask = X[:, i] > xmax
                if np.any(mask):
                    if use_sparse:
                        # Note: See comment about SparseEfficiencyWarning above.
                        XBS_sparse = XBS_sparse.tolil()
                        XBS_sparse[mask, -degree:] = f_max[-degree:]
                    else:
                        XBS[
                            mask,
                            ((i + 1) * n_splines - degree) : ((i + 1) * n_splines),
                        ] = f_max[-degree:]

            elif self.extrapolation == "linear":
                # Continue the degree first and degree last spline bases
                # linearly beyond the boundaries, with slope = derivative at
                # the boundary.
                # Note that all others have derivative = value = 0 at the
                # boundaries.

                # spline derivatives = slopes at boundaries
                fp_min, fp_max = spl(xmin, nu=1), spl(xmax, nu=1)
                # Compute the linear continuation.
                if degree <= 1:
                    # For degree=1, the derivative of 2nd spline is not zero at
                    # boundary. For degree=0 it is the same as 'constant'.
                    degree += 1
                for j in range(degree):
                    mask = X[:, i] < xmin
                    if np.any(mask):
                        linear_extr = f_min[j] + (X[mask, i] - xmin) * fp_min[j]
                        if use_sparse:
                            # Note: See comment about SparseEfficiencyWarning above.
                            XBS_sparse = XBS_sparse.tolil()
                            XBS_sparse[mask, j] = linear_extr
                        else:
                            XBS[mask, i * n_splines + j] = linear_extr

                    mask = X[:, i] > xmax
                    if np.any(mask):
                        k = n_splines - 1 - j
                        linear_extr = f_max[k] + (X[mask, i] - xmax) * fp_max[k]
                        if use_sparse:
                            # Note: See comment about SparseEfficiencyWarning above.
                            XBS_sparse = XBS_sparse.tolil()
                            XBS_sparse[mask, k : k + 1] = linear_extr[:, None]
                        else:
                            XBS[mask, i * n_splines + k] = linear_extr

            if use_sparse:
                if not sparse.isspmatrix_csr(XBS_sparse):
                    XBS_sparse = XBS_sparse.tocsr()
                output_list.append(XBS_sparse)

        if use_sparse:
            # TODO: Remove this conditional error when the minimum supported version of
            # SciPy is 1.9.2
            # `scipy.sparse.hstack` breaks in scipy<1.9.2
            # when `n_features_out_ > max_int32`
            max_int32 = np.iinfo(np.int32).max
            all_int32 = True
            for mat in output_list:
                all_int32 &= mat.indices.dtype == np.int32
            if (
                sp_version < parse_version("1.9.2")
                and self.n_features_out_ > max_int32
                and all_int32
            ):
                raise ValueError(
                    "In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`"
                    " produces negative columns when:\n1. The output shape contains"
                    " `n_cols` too large to be represented by a 32bit signed"
                    " integer.\n. All sub-matrices to be stacked have indices of"
                    " dtype `np.int32`.\nTo avoid this error, either use a version"
                    " of scipy `>=1.9.2` or alter the `SplineTransformer`"
                    " transformer to produce fewer than 2^31 output features"
                )
            XBS = sparse.hstack(output_list, format="csr")
        elif self.sparse_output:
            # TODO: Remove ones scipy 1.10 is the minimum version. See comments above.
            XBS = sparse.csr_matrix(XBS)

        if self.include_bias:
            return XBS
        else:
            # We throw away one spline basis per feature.
            # We chose the last one.
            indices = [j for j in range(XBS.shape[1]) if (j + 1) % n_splines != 0]
            return XBS[:, indices]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimators_pickle": (
                    "Current Scipy implementation of _bsplines does not"
                    "support const memory views."
                ),
            }
        }
