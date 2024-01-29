"""
The :mod:`sklearn.utils.multiclass` module includes utilities to handle
multiclass/multioutput target in classifiers.
"""

# Author: Arnaud Joly, Joel Nothman, Hamzeh Alsalhi
#
# License: BSD 3 clause
import warnings
from collections.abc import Sequence
from itertools import chain

import numpy as np
from scipy.sparse import issparse

from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array


def _unique_multiclass(y):
    xp, is_array_api_compliant = get_namespace(y)
    if hasattr(y, "__array__") or is_array_api_compliant:
        return xp.unique_values(xp.asarray(y))
    else:
        return set(y)


def _unique_indicator(y):
    xp, _ = get_namespace(y)
    return xp.arange(
        check_array(y, input_name="y", accept_sparse=["csr", "csc", "coo"]).shape[1]
    )


_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,
    "multiclass": _unique_multiclass,
    "multilabel-indicator": _unique_indicator,
}


def unique_labels(*ys):
    """Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes
        Label values.

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    xp, is_array_api_compliant = get_namespace(*ys)
    if not ys:
        raise ValueError("No argument has been passed.")
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Check consistency for the indicator format
    if (
        label_type == "multilabel-indicator"
        and len(
            set(
                check_array(y, accept_sparse=["csr", "csc", "coo"]).shape[1] for y in ys
            )
        )
        > 1
    ):
        raise ValueError(
            "Multi-label binary indicator input with different numbers of labels"
        )

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    if is_array_api_compliant:
        # array_api does not allow for mixed dtypes
        unique_ys = xp.concat([_unique_labels(y) for y in ys])
        return xp.unique_values(unique_ys)

    ys_labels = set(chain.from_iterable((i for i in _unique_labels(y)) for y in ys))
    # Check that we don't mix string type with number type
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    return xp.asarray(sorted(ys_labels))


def _is_integral_float(y):
    xp, is_array_api_compliant = get_namespace(y)
    return xp.isdtype(y.dtype, "real floating") and bool(
        xp.all(xp.astype((xp.astype(y, xp.int64)), y.dtype) == y)
    )


def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    xp, is_array_api_compliant = get_namespace(y)
    if hasattr(y, "__array__") or isinstance(y, Sequence) or is_array_api_compliant:
        # DeprecationWarning will be replaced by ValueError, see NEP 34
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        check_y_kwargs = dict(
            accept_sparse=True,
            allow_nd=True,
            force_all_finite=False,
            ensure_2d=False,
            ensure_min_samples=0,
            ensure_min_features=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            try:
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (VisibleDeprecationWarning, ValueError) as e:
                if str(e).startswith("Complex data not supported"):
                    raise

                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = check_array(y, dtype=object, **check_y_kwargs)

    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if y.format in ("dok", "lil"):
            y = y.tocsr()
        labels = xp.unique_values(y.data)
        return (
            len(y.data) == 0
            or (labels.size == 1 or (labels.size == 2) and (0 in labels))
            and (y.dtype.kind in "biu" or _is_integral_float(labels))  # bool, int, uint
        )
    else:
        labels = xp.unique_values(y)

        return labels.shape[0] < 3 and (
            xp.isdtype(y.dtype, ("bool", "signed integer", "unsigned integer"))
            or _is_integral_float(labels)
        )


def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        Target values.
    """
    y_type = type_of_target(y, input_name="y")
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError(
            f"Unknown label type: {y_type}. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a "
            "regression target with continuous values."
        )


def type_of_target(y, input_name=""):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : {array-like, sparse matrix}
        Target values. If a sparse matrix, `y` is expected to be a
        CSR/CSC matrix.

    input_name : str, default=""
        The data name used to construct the error message.

        .. versionadded:: 1.1.0

    Returns
    -------
    target_type : str
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    xp, is_array_api_compliant = get_namespace(y)
    valid = (
        (isinstance(y, Sequence) or issparse(y) or hasattr(y, "__array__"))
        and not isinstance(y, str)
        or is_array_api_compliant
    )

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), got %r" % y
        )

    sparse_pandas = y.__class__.__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return "multilabel-indicator"

    # DeprecationWarning will be replaced by ValueError, see NEP 34
    # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    # We therefore catch both deprecation (NumPy < 1.24) warning and
    # value error (NumPy >= 1.24).
    check_y_kwargs = dict(
        accept_sparse=True,
        allow_nd=True,
        force_all_finite=False,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_min_features=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", VisibleDeprecationWarning)
        if not issparse(y):
            try:
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (VisibleDeprecationWarning, ValueError) as e:
                if str(e).startswith("Complex data not supported"):
                    raise

                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = check_array(y, dtype=object, **check_y_kwargs)

    # The old sequence of sequences format
    try:
        first_row = y[[0], :] if issparse(y) else y[0]
        if (
            not hasattr(first_row, "__array__")
            and isinstance(first_row, Sequence)
            and not isinstance(first_row, str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim not in (1, 2):
        # Number of dimension greater than 2: [[[1, 2]]]
        return "unknown"
    if not min(y.shape):
        # Empty ndarray: []/[[]]
        if y.ndim == 1:
            # 1-D empty array: []
            return "binary"  # []
        # 2-D empty array: [[]]
        return "unknown"
    if not issparse(y) and y.dtype == object and not isinstance(y.flat[0], str):
        # [obj_1] and not ["label_1"]
        return "unknown"

    # Check if multioutput
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # Check float and contains non-integer float values
    if xp.isdtype(y.dtype, "real floating"):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        data = y.data if issparse(y) else y
        if xp.any(data != xp.astype(data, int)):
            _assert_all_finite(data, input_name=input_name)
            return "continuous" + suffix

    # Check multiclass
    if issparse(first_row):
        first_row = first_row.data
    if xp.unique_values(y).shape[0] > 2 or (y.ndim == 2 and len(first_row) > 1):
        # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        return "multiclass" + suffix
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]


def _check_partial_fit_first_call(clf, classes=None):
    """Private helper function for factorizing common classes param logic.

    Estimators that implement the ``partial_fit`` API need to be provided with
    the list of possible classes at the first call to partial_fit.

    Subsequent calls to partial_fit should check that ``classes`` is still
    consistent with a previous value of ``clf.classes_`` when provided.

    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.

    """
    if getattr(clf, "classes_", None) is None and classes is None:
        raise ValueError("classes must be passed on the first call to partial_fit.")

    elif classes is not None:
        if getattr(clf, "classes_", None) is not None:
            if not np.array_equal(clf.classes_, unique_labels(classes)):
                raise ValueError(
                    "`classes=%r` is not the same as on last call "
                    "to partial_fit, was: %r" % (classes, clf.classes_)
                )

        else:
            # This is the first call to partial_fit
            clf.classes_ = unique_labels(classes)
            return True

    # classes is None and clf.classes_ has already previously been set:
    # nothing to do
    return False


def class_distribution(y, sample_weight=None):
    """Compute class priors from multioutput-multiclass target data.

    Parameters
    ----------
    y : {array-like, sparse matrix} of size (n_samples, n_outputs)
        The labels for each example.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    classes : list of size n_outputs of ndarray of size (n_classes,)
        List of classes for each column.

    n_classes : list of int of size n_outputs
        Number of classes in each column.

    class_prior : list of size n_outputs of ndarray of size (n_classes,)
        Class distribution of each column.
    """
    classes = []
    n_classes = []
    class_prior = []

    n_samples, n_outputs = y.shape
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if issparse(y):
        y = y.tocsc()
        y_nnz = np.diff(y.indptr)

        for k in range(n_outputs):
            col_nonzero = y.indices[y.indptr[k] : y.indptr[k + 1]]
            # separate sample weights for zero and non-zero elements
            if sample_weight is not None:
                nz_samp_weight = sample_weight[col_nonzero]
                zeros_samp_weight_sum = np.sum(sample_weight) - np.sum(nz_samp_weight)
            else:
                nz_samp_weight = None
                zeros_samp_weight_sum = y.shape[0] - y_nnz[k]

            classes_k, y_k = np.unique(
                y.data[y.indptr[k] : y.indptr[k + 1]], return_inverse=True
            )
            class_prior_k = np.bincount(y_k, weights=nz_samp_weight)

            # An explicit zero was found, combine its weight with the weight
            # of the implicit zeros
            if 0 in classes_k:
                class_prior_k[classes_k == 0] += zeros_samp_weight_sum

            # If an there is an implicit zero and it is not in classes and
            # class_prior, make an entry for it
            if 0 not in classes_k and y_nnz[k] < y.shape[0]:
                classes_k = np.insert(classes_k, 0, 0)
                class_prior_k = np.insert(class_prior_k, 0, zeros_samp_weight_sum)

            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])
            class_prior.append(class_prior_k / class_prior_k.sum())
    else:
        for k in range(n_outputs):
            classes_k, y_k = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])
            class_prior_k = np.bincount(y_k, weights=sample_weight)
            class_prior.append(class_prior_k / class_prior_k.sum())

    return (classes, n_classes, class_prior)


def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking OvR decision function from OvO.

    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.

    confidences : array-like of shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.

    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``.
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    # Monotonically transform the sum_of_confidences to (-1/3, 1/3)
    # and add it with votes. The monotonic transformation  is
    # f: x -> x / (3 * (|x| + 1)), it uses 1/3 instead of 1/2
    # to ensure that we won't reach the limits and change vote order.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    transformed_confidences = sum_of_confidences / (
        3 * (np.abs(sum_of_confidences) + 1)
    )
    return votes + transformed_confidences
