import itertools
import re
import warnings
from collections import defaultdict
from math import floor, log10

import numpy as np
import pytest
import threadpoolctl
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from sklearn.metrics import euclidean_distances
from sklearn.metrics._pairwise_distances_reduction import (
    ArgKmin,
    ArgKminClassMode,
    BaseDistancesReductionDispatcher,
    RadiusNeighbors,
    sqeuclidean_row_norms,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    create_memmap_backed_data,
)

# Common supported metric between scipy.spatial.distance.cdist
# and BaseDistanceReductionDispatcher.
# This allows constructing tests to check consistency of results
# of concrete BaseDistanceReductionDispatcher on some metrics using APIs
# from scipy and numpy.
CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "euclidean",
    "minkowski",
    "seuclidean",
]


def _get_metric_params_list(metric: str, n_features: int, seed: int = 1):
    """Return list of dummy DistanceMetric kwargs for tests."""

    # Distinguishing on cases not to compute unneeded datastructures.
    rng = np.random.RandomState(seed)

    if metric == "minkowski":
        minkowski_kwargs = [
            dict(p=1.5),
            dict(p=2),
            dict(p=3),
            dict(p=np.inf),
            dict(p=3, w=rng.rand(n_features)),
        ]

        return minkowski_kwargs

    if metric == "seuclidean":
        return [dict(V=rng.rand(n_features))]

    # Case of: "euclidean", "manhattan", "chebyshev", "haversine" or any other metric.
    # In those cases, no kwargs is needed.
    return [{}]


def assert_argkmin_results_equality(ref_dist, dist, ref_indices, indices, rtol=1e-7):
    assert_array_equal(
        ref_indices,
        indices,
        err_msg="Query vectors have different neighbors' indices",
    )
    assert_allclose(
        ref_dist,
        dist,
        err_msg="Query vectors have different neighbors' distances",
        rtol=rtol,
    )


def relative_rounding(scalar, n_significant_digits):
    """Round a scalar to a number of significant digits relatively to its value."""
    if scalar == 0:
        return 0.0
    magnitude = int(floor(log10(abs(scalar)))) + 1
    return round(scalar, n_significant_digits - magnitude)


def test_relative_rounding():
    assert relative_rounding(0, 1) == 0.0
    assert relative_rounding(0, 10) == 0.0
    assert relative_rounding(0, 123456) == 0.0

    assert relative_rounding(123456789, 0) == 0
    assert relative_rounding(123456789, 2) == 120000000
    assert relative_rounding(123456789, 3) == 123000000
    assert relative_rounding(123456789, 10) == 123456789
    assert relative_rounding(123456789, 20) == 123456789

    assert relative_rounding(1.23456789, 2) == 1.2
    assert relative_rounding(1.23456789, 3) == 1.23
    assert relative_rounding(1.23456789, 10) == 1.23456789

    assert relative_rounding(123.456789, 3) == 123.0
    assert relative_rounding(123.456789, 9) == 123.456789
    assert relative_rounding(123.456789, 10) == 123.456789


def assert_argkmin_results_quasi_equality(
    ref_dist,
    dist,
    ref_indices,
    indices,
    rtol=1e-4,
):
    """Assert that argkmin results are valid up to:
      - relative tolerance on computed distance values
      - permutations of indices for distances values that differ up to
        a precision level

    To be used for testing neighbors queries on float32 datasets: we
    accept neighbors rank swaps only if they are caused by small
    rounding errors on the distance computations.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    n_significant_digits = -(int(floor(log10(abs(rtol)))) + 1)

    assert (
        ref_dist.shape == dist.shape == ref_indices.shape == indices.shape
    ), "Arrays of results have various shapes."

    n_queries, n_neighbors = ref_dist.shape

    # Asserting equality results one row at a time
    for query_idx in range(n_queries):
        ref_dist_row = ref_dist[query_idx]
        dist_row = dist[query_idx]

        assert is_sorted(
            ref_dist_row
        ), f"Reference distances aren't sorted on row {query_idx}"
        assert is_sorted(dist_row), f"Distances aren't sorted on row {query_idx}"

        assert_allclose(ref_dist_row, dist_row, rtol=rtol)

        ref_indices_row = ref_indices[query_idx]
        indices_row = indices[query_idx]

        # Grouping indices by distances using sets on a rounded distances up
        # to a given number of decimals of significant digits derived from rtol.
        reference_neighbors_groups = defaultdict(set)
        effective_neighbors_groups = defaultdict(set)

        for neighbor_rank in range(n_neighbors):
            rounded_dist = relative_rounding(
                ref_dist_row[neighbor_rank],
                n_significant_digits=n_significant_digits,
            )
            reference_neighbors_groups[rounded_dist].add(ref_indices_row[neighbor_rank])
            effective_neighbors_groups[rounded_dist].add(indices_row[neighbor_rank])

        # Asserting equality of groups (sets) for each distance
        msg = (
            f"Neighbors indices for query {query_idx} are not matching "
            f"when rounding distances at {n_significant_digits} significant digits "
            f"derived from rtol={rtol:.1e}"
        )
        for rounded_distance in reference_neighbors_groups.keys():
            assert (
                reference_neighbors_groups[rounded_distance]
                == effective_neighbors_groups[rounded_distance]
            ), msg


def assert_radius_neighbors_results_equality(
    ref_dist, dist, ref_indices, indices, radius
):
    # We get arrays of arrays and we need to check for individual pairs
    for i in range(ref_dist.shape[0]):
        assert (ref_dist[i] <= radius).all()
        assert_array_equal(
            ref_indices[i],
            indices[i],
            err_msg=f"Query vector #{i} has different neighbors' indices",
        )
        assert_allclose(
            ref_dist[i],
            dist[i],
            err_msg=f"Query vector #{i} has different neighbors' distances",
            rtol=1e-7,
        )


def assert_radius_neighbors_results_quasi_equality(
    ref_dist,
    dist,
    ref_indices,
    indices,
    radius,
    rtol=1e-4,
):
    """Assert that radius neighborhood results are valid up to:
      - relative tolerance on computed distance values
      - permutations of indices for distances values that differ up to
        a precision level
      - missing or extra last elements if their distance is
        close to the radius

    To be used for testing neighbors queries on float32 datasets: we
    accept neighbors rank swaps only if they are caused by small
    rounding errors on the distance computations.

    Input arrays must be sorted w.r.t distances.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    n_significant_digits = -(int(floor(log10(abs(rtol)))) + 1)

    assert (
        len(ref_dist) == len(dist) == len(ref_indices) == len(indices)
    ), "Arrays of results have various lengths."

    n_queries = len(ref_dist)

    # Asserting equality of results one vector at a time
    for query_idx in range(n_queries):
        ref_dist_row = ref_dist[query_idx]
        dist_row = dist[query_idx]

        assert is_sorted(
            ref_dist_row
        ), f"Reference distances aren't sorted on row {query_idx}"
        assert is_sorted(dist_row), f"Distances aren't sorted on row {query_idx}"

        # Vectors' lengths might be different due to small
        # numerical differences of distance w.r.t the `radius` threshold.
        largest_row = ref_dist_row if len(ref_dist_row) > len(dist_row) else dist_row

        # For the longest distances vector, we check that last extra elements
        # that aren't present in the other vector are all in: [radius ± rtol]
        min_length = min(len(ref_dist_row), len(dist_row))
        last_extra_elements = largest_row[min_length:]
        if last_extra_elements.size > 0:
            assert np.all(radius - rtol <= last_extra_elements <= radius + rtol), (
                f"The last extra elements ({last_extra_elements}) aren't in [radius ±"
                f" rtol]=[{radius} ± {rtol}]"
            )

        # We truncate the neighbors results list on the smallest length to
        # be able to compare them, ignoring the elements checked above.
        ref_dist_row = ref_dist_row[:min_length]
        dist_row = dist_row[:min_length]

        assert_allclose(ref_dist_row, dist_row, rtol=rtol)

        ref_indices_row = ref_indices[query_idx]
        indices_row = indices[query_idx]

        # Grouping indices by distances using sets on a rounded distances up
        # to a given number of significant digits derived from rtol.
        reference_neighbors_groups = defaultdict(set)
        effective_neighbors_groups = defaultdict(set)

        for neighbor_rank in range(min_length):
            rounded_dist = relative_rounding(
                ref_dist_row[neighbor_rank],
                n_significant_digits=n_significant_digits,
            )
            reference_neighbors_groups[rounded_dist].add(ref_indices_row[neighbor_rank])
            effective_neighbors_groups[rounded_dist].add(indices_row[neighbor_rank])

        # Asserting equality of groups (sets) for each distance
        msg = (
            f"Neighbors indices for query {query_idx} are not matching "
            f"when rounding distances at {n_significant_digits} significant digits "
            f"derived from rtol={rtol:.1e}"
        )
        for rounded_distance in reference_neighbors_groups.keys():
            assert (
                reference_neighbors_groups[rounded_distance]
                == effective_neighbors_groups[rounded_distance]
            ), msg


ASSERT_RESULT = {
    # In the case of 64bit, we test for exact equality of the results rankings
    # and standard tolerance levels for the computed distance values.
    #
    # XXX: Note that in the future we might be interested in using quasi equality
    # checks also for float64 data (with a larger number of significant digits)
    # as the tests could be unstable because of numerically tied distances on
    # some datasets (e.g. uniform grids).
    (ArgKmin, np.float64): assert_argkmin_results_equality,
    (
        RadiusNeighbors,
        np.float64,
    ): assert_radius_neighbors_results_equality,
    # In the case of 32bit, indices can be permuted due to small difference
    # in the computations of their associated distances, hence we test equality of
    # results up to valid permutations.
    (ArgKmin, np.float32): assert_argkmin_results_quasi_equality,
    (
        RadiusNeighbors,
        np.float32,
    ): assert_radius_neighbors_results_quasi_equality,
}


def test_assert_argkmin_results_quasi_equality():
    rtol = 1e-7
    eps = 1e-7
    _1m = 1.0 - eps
    _1p = 1.0 + eps

    _6_1m = 6.1 - eps
    _6_1p = 6.1 + eps

    ref_dist = np.array(
        [
            [1.2, 2.5, _6_1m, 6.1, _6_1p],
            [_1m, _1m, 1, _1p, _1p],
        ]
    )
    ref_indices = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
    )

    # Sanity check: compare the reference results to themselves.
    assert_argkmin_results_quasi_equality(
        ref_dist, ref_dist, ref_indices, ref_indices, rtol
    )

    # Apply valid permutation on indices: the last 3 points are
    # all very close to one another so we accept any permutation
    # on their rankings.
    assert_argkmin_results_quasi_equality(
        np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
        np.array([[1.2, 2.5, 6.1, 6.1, 6.1]]),
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 4, 5, 3]]),
        rtol=rtol,
    )
    # All points are have close distances so any ranking permutation
    # is valid for this query result.
    assert_argkmin_results_quasi_equality(
        np.array([[_1m, _1m, 1, _1p, _1p]]),
        np.array([[_1m, _1m, 1, _1p, _1p]]),
        np.array([[6, 7, 8, 9, 10]]),
        np.array([[6, 9, 7, 8, 10]]),
        rtol=rtol,
    )

    # Apply invalid permutation on indices: permuting the ranks
    # of the 2 nearest neighbors is invalid because the distance
    # values are too different.
    msg = "Neighbors indices for query 0 are not matching"
    with pytest.raises(AssertionError, match=msg):
        assert_argkmin_results_quasi_equality(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 3, 4, 5]]),
            rtol=rtol,
        )

    # Indices aren't properly sorted w.r.t their distances
    msg = "Neighbors indices for query 0 are not matching"
    with pytest.raises(AssertionError, match=msg):
        assert_argkmin_results_quasi_equality(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 4, 5, 3]]),
            rtol=rtol,
        )

    # Distances aren't properly sorted
    msg = "Distances aren't sorted on row 0"
    with pytest.raises(AssertionError, match=msg):
        assert_argkmin_results_quasi_equality(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[2.5, 1.2, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 4, 5, 3]]),
            rtol=rtol,
        )


def test_assert_radius_neighbors_results_quasi_equality():
    rtol = 1e-7
    eps = 1e-7
    _1m = 1.0 - eps
    _1p = 1.0 + eps

    _6_1m = 6.1 - eps
    _6_1p = 6.1 + eps

    ref_dist = [
        np.array([1.2, 2.5, _6_1m, 6.1, _6_1p]),
        np.array([_1m, 1, _1p, _1p]),
    ]

    ref_indices = [
        np.array([1, 2, 3, 4, 5]),
        np.array([6, 7, 8, 9]),
    ]

    # Sanity check: compare the reference results to themselves.
    assert_radius_neighbors_results_quasi_equality(
        ref_dist,
        ref_dist,
        ref_indices,
        ref_indices,
        radius=6.1,
        rtol=rtol,
    )

    # Apply valid permutation on indices
    assert_radius_neighbors_results_quasi_equality(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
        np.array([np.array([1, 2, 3, 4, 5])]),
        np.array([np.array([1, 2, 4, 5, 3])]),
        radius=6.1,
        rtol=rtol,
    )
    assert_radius_neighbors_results_quasi_equality(
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),
        np.array([np.array([6, 7, 8, 9, 10])]),
        np.array([np.array([6, 9, 7, 8, 10])]),
        radius=6.1,
        rtol=rtol,
    )

    # Apply invalid permutation on indices
    msg = "Neighbors indices for query 0 are not matching"
    with pytest.raises(AssertionError, match=msg):
        assert_radius_neighbors_results_quasi_equality(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 3, 4, 5])]),
            radius=6.1,
            rtol=rtol,
        )

    # Having extra last elements is valid if they are in: [radius ± rtol]
    assert_radius_neighbors_results_quasi_equality(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
        np.array([np.array([1.2, 2.5, _6_1m, 6.1])]),
        np.array([np.array([1, 2, 3, 4, 5])]),
        np.array([np.array([1, 2, 3, 4])]),
        radius=6.1,
        rtol=rtol,
    )

    # Having extra last elements is invalid if they are lesser than radius - rtol
    msg = re.escape(
        "The last extra elements ([6.]) aren't in [radius ± rtol]=[6.1 ± 1e-07]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_radius_neighbors_results_quasi_equality(
            np.array([np.array([1.2, 2.5, 6])]),
            np.array([np.array([1.2, 2.5])]),
            np.array([np.array([1, 2, 3])]),
            np.array([np.array([1, 2])]),
            radius=6.1,
            rtol=rtol,
        )

    # Indices aren't properly sorted w.r.t their distances
    msg = "Neighbors indices for query 0 are not matching"
    with pytest.raises(AssertionError, match=msg):
        assert_radius_neighbors_results_quasi_equality(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 4, 5, 3])]),
            radius=6.1,
            rtol=rtol,
        )

    # Distances aren't properly sorted
    msg = "Distances aren't sorted on row 0"
    with pytest.raises(AssertionError, match=msg):
        assert_radius_neighbors_results_quasi_equality(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 4, 5, 3])]),
            radius=6.1,
            rtol=rtol,
        )


def test_pairwise_distances_reduction_is_usable_for():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    X_csr = csr_matrix(X)
    Y_csr = csr_matrix(Y)
    metric = "manhattan"

    # Must be usable for all possible pair of {dense, sparse} datasets
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y_csr, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y_csr, metric)

    assert BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float64), Y.astype(np.float64), metric
    )

    assert BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float32), Y.astype(np.float32), metric
    )

    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.int64), Y.astype(np.int64), metric
    )

    assert not BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric="pyfunc")
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X.astype(np.float32), Y, metric
    )
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        X, Y.astype(np.int32), metric
    )

    # F-ordered arrays are not supported
    assert not BaseDistancesReductionDispatcher.is_usable_for(
        np.asfortranarray(X), Y, metric
    )

    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric="euclidean")
    assert BaseDistancesReductionDispatcher.is_usable_for(
        X, Y_csr, metric="sqeuclidean"
    )

    assert BaseDistancesReductionDispatcher.is_usable_for(
        X_csr, Y_csr, metric="sqeuclidean"
    )
    assert BaseDistancesReductionDispatcher.is_usable_for(
        X_csr, Y_csr, metric="euclidean"
    )

    # CSR matrices without non-zeros elements aren't currently supported
    # TODO: support CSR matrices without non-zeros elements
    X_csr_0_nnz = csr_matrix(X * 0)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_0_nnz, Y, metric)

    # CSR matrices with int64 indices and indptr (e.g. large nnz, or large n_features)
    # aren't supported as of now.
    # See: https://github.com/scikit-learn/scikit-learn/issues/23653
    # TODO: support CSR matrices with int64 indices and indptr
    X_csr_int64 = csr_matrix(X)
    X_csr_int64.indices = X_csr_int64.indices.astype(np.int64)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_int64, Y, metric)


def test_argkmin_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    k = 5
    metric = "euclidean"

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    with pytest.raises(ValueError, match=msg):
        ArgKmin.compute(X=X.astype(np.float32), Y=Y, k=k, metric=metric)

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    with pytest.raises(ValueError, match=msg):
        ArgKmin.compute(X=X, Y=Y.astype(np.int32), k=k, metric=metric)

    with pytest.raises(ValueError, match="k == -1, must be >= 1."):
        ArgKmin.compute(X=X, Y=Y, k=-1, metric=metric)

    with pytest.raises(ValueError, match="k == 0, must be >= 1."):
        ArgKmin.compute(X=X, Y=Y, k=0, metric=metric)

    with pytest.raises(ValueError, match="Unrecognized metric"):
        ArgKmin.compute(X=X, Y=Y, k=k, metric="wrong metric")

    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        ArgKmin.compute(X=np.array([1.0, 2.0]), Y=Y, k=k, metric=metric)

    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        ArgKmin.compute(X=np.asfortranarray(X), Y=Y, k=k, metric=metric)

    # A UserWarning must be raised in this case.
    unused_metric_kwargs = {"p": 3}

    message = r"Some metric_kwargs have been passed \({'p': 3}\) but"

    with pytest.warns(UserWarning, match=message):
        ArgKmin.compute(
            X=X, Y=Y, k=k, metric=metric, metric_kwargs=unused_metric_kwargs
        )

    # A UserWarning must be raised in this case.
    metric_kwargs = {
        "p": 3,  # unused
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }

    message = r"Some metric_kwargs have been passed \({'p': 3, 'Y_norm_squared'"

    with pytest.warns(UserWarning, match=message):
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)

    # No user warning must be raised in this case.
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)

    # No user warning must be raised in this case.
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        ArgKmin.compute(X=X, Y=Y, k=k, metric=metric, metric_kwargs=metric_kwargs)


def test_argkmin_classmode_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    k = 5
    metric = "manhattan"

    weights = "uniform"
    labels = rng.randint(low=0, high=10, size=100)
    unique_labels = np.unique(labels)

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    with pytest.raises(ValueError, match=msg):
        ArgKminClassMode.compute(
            X=X.astype(np.float32),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    with pytest.raises(ValueError, match=msg):
        ArgKminClassMode.compute(
            X=X,
            Y=Y.astype(np.int32),
            k=k,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    with pytest.raises(ValueError, match="k == -1, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=-1,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    with pytest.raises(ValueError, match="k == 0, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=0,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    with pytest.raises(ValueError, match="Unrecognized metric"):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=k,
            metric="wrong metric",
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        ArgKminClassMode.compute(
            X=np.array([1.0, 2.0]),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        ArgKminClassMode.compute(
            X=np.asfortranarray(X),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            labels=labels,
            unique_labels=unique_labels,
        )

    non_existent_weights_strategy = "non_existent_weights_strategy"
    message = (
        "Only the 'uniform' or 'distance' weights options are supported at this time. "
        f"Got: weights='{non_existent_weights_strategy}'."
    )
    with pytest.raises(ValueError, match=message):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=k,
            metric=metric,
            weights=non_existent_weights_strategy,
            labels=labels,
            unique_labels=unique_labels,
        )

    # TODO: introduce assertions on UserWarnings once the Euclidean specialisation
    # of ArgKminClassMode is supported.


def test_radius_neighbors_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = "euclidean"

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        RadiusNeighbors.compute(
            X=X.astype(np.float32), Y=Y, radius=radius, metric=metric
        )

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        RadiusNeighbors.compute(X=X, Y=Y.astype(np.int32), radius=radius, metric=metric)

    with pytest.raises(ValueError, match="radius == -1.0, must be >= 0."):
        RadiusNeighbors.compute(X=X, Y=Y, radius=-1, metric=metric)

    with pytest.raises(ValueError, match="Unrecognized metric"):
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric="wrong metric")

    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        RadiusNeighbors.compute(
            X=np.array([1.0, 2.0]), Y=Y, radius=radius, metric=metric
        )

    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        RadiusNeighbors.compute(
            X=np.asfortranarray(X), Y=Y, radius=radius, metric=metric
        )

    unused_metric_kwargs = {"p": 3}

    # A UserWarning must be raised in this case.
    message = r"Some metric_kwargs have been passed \({'p': 3}\) but"

    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=unused_metric_kwargs
        )

    # A UserWarning must be raised in this case.
    metric_kwargs = {
        "p": 3,  # unused
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }

    message = r"Some metric_kwargs have been passed \({'p': 3, 'Y_norm_squared'"

    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )

    # No user warning must be raised in this case.
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
        "Y_norm_squared": sqeuclidean_row_norms(Y, num_threads=2),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )

    # No user warning must be raised in this case.
    metric_kwargs = {
        "X_norm_squared": sqeuclidean_row_norms(X, num_threads=2),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        RadiusNeighbors.compute(
            X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs
        )


@pytest.mark.parametrize(
    "n_samples_X, n_samples_Y", [(100, 100), (500, 100), (100, 500)]
)
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_chunk_size_agnosticism(
    global_random_seed,
    Dispatcher,
    n_samples_X,
    n_samples_Y,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the chunk size."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Scaling the radius slightly with the numbers of dimensions
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=256,  # default
        metric="manhattan",
        return_distance=True,
        **compute_parameters,
    )

    dist, indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=41,
        metric="manhattan",
        return_distance=True,
        **compute_parameters,
    )

    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist, ref_indices, indices, **check_parameters
    )


@pytest.mark.parametrize(
    "n_samples_X, n_samples_Y", [(100, 100), (500, 100), (100, 500)]
)
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_n_threads_agnosticism(
    global_random_seed,
    Dispatcher,
    n_samples_X,
    n_samples_Y,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the number of threads."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Scaling the radius slightly with the numbers of dimensions
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=25,  # make sure we use multiple threads
        return_distance=True,
        **compute_parameters,
    )

    with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
        dist, indices = Dispatcher.compute(
            X,
            Y,
            parameter,
            chunk_size=25,
            return_distance=True,
            **compute_parameters,
        )

    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist, ref_indices, indices, **check_parameters
    )


@pytest.mark.parametrize(
    "Dispatcher, dtype",
    [
        (ArgKmin, np.float64),
        (RadiusNeighbors, np.float32),
        (ArgKmin, np.float32),
        (RadiusNeighbors, np.float64),
    ],
)
def test_format_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
):
    """Check that results do not depend on the format (dense, sparse) of the input."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples, n_features = 100, 100

    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_matrix(X)
    Y_csr = csr_matrix(Y)

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Scaling the radius slightly with the numbers of dimensions
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    dist_dense, indices_dense = Dispatcher.compute(
        X,
        Y,
        parameter,
        chunk_size=50,
        return_distance=True,
        **compute_parameters,
    )

    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        if _X is X and _Y is Y:
            continue
        dist, indices = Dispatcher.compute(
            _X,
            _Y,
            parameter,
            chunk_size=50,
            return_distance=True,
            **compute_parameters,
        )
        ASSERT_RESULT[(Dispatcher, dtype)](
            dist_dense,
            dist,
            indices_dense,
            indices,
            **check_parameters,
        )


@pytest.mark.parametrize(
    "n_samples_X, n_samples_Y", [(100, 100), (100, 500), (500, 100)]
)
@pytest.mark.parametrize(
    "metric",
    ["euclidean", "minkowski", "manhattan", "infinity", "seuclidean", "haversine"],
)
@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_strategies_consistency(
    global_random_seed,
    Dispatcher,
    metric,
    n_samples_X,
    n_samples_Y,
    dtype,
    n_features=10,
):
    """Check that the results do not depend on the strategy used."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread

    # Haversine distance only accepts 2D data
    if metric == "haversine":
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Scaling the radius slightly with the numbers of dimensions
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    dist_par_X, indices_par_X = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        # Taking the first
        metric_kwargs=_get_metric_params_list(
            metric, n_features, seed=global_random_seed
        )[0],
        # To be sure to use parallelization
        chunk_size=n_samples_X // 4,
        strategy="parallel_on_X",
        return_distance=True,
        **compute_parameters,
    )

    dist_par_Y, indices_par_Y = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        # Taking the first
        metric_kwargs=_get_metric_params_list(
            metric, n_features, seed=global_random_seed
        )[0],
        # To be sure to use parallelization
        chunk_size=n_samples_Y // 4,
        strategy="parallel_on_Y",
        return_distance=True,
        **compute_parameters,
    )

    ASSERT_RESULT[(Dispatcher, dtype)](
        dist_par_X, dist_par_Y, indices_par_X, indices_par_Y, **check_parameters
    )


# "Concrete Dispatchers"-specific tests


@pytest.mark.parametrize("n_features", [50, 500])
@pytest.mark.parametrize("translation", [0, 1e6])
@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pairwise_distances_argkmin(
    global_random_seed,
    n_features,
    translation,
    metric,
    strategy,
    dtype,
    n_samples=100,
    k=10,
):
    # TODO: can we easily fix this discrepancy?
    edge_cases = [
        (np.float32, "chebyshev", 1000000.0),
        (np.float32, "cityblock", 1000000.0),
    ]
    if (dtype, metric, translation) in edge_cases:
        pytest.xfail("Numerical differences lead to small differences in results.")

    rng = np.random.RandomState(global_random_seed)
    spread = 1000
    X = translation + rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_matrix(X)
    Y_csr = csr_matrix(Y)

    # Haversine distance only accepts 2D data
    if metric == "haversine":
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])

    metric_kwargs = _get_metric_params_list(metric, n_features)[0]

    # Reference for argkmin results
    if metric == "euclidean":
        # Compare to scikit-learn GEMM optimized implementation
        dist_matrix = euclidean_distances(X, Y)
    else:
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)
    # Taking argkmin (indices of the k smallest values)
    argkmin_indices_ref = np.argsort(dist_matrix, axis=1)[:, :k]
    # Getting the associated distances
    argkmin_distances_ref = np.zeros(argkmin_indices_ref.shape, dtype=np.float64)
    for row_idx in range(argkmin_indices_ref.shape[0]):
        argkmin_distances_ref[row_idx] = dist_matrix[
            row_idx, argkmin_indices_ref[row_idx]
        ]

    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        argkmin_distances, argkmin_indices = ArgKmin.compute(
            _X,
            _Y,
            k,
            metric=metric,
            metric_kwargs=metric_kwargs,
            return_distance=True,
            # So as to have more than a chunk, forcing parallelism.
            chunk_size=n_samples // 4,
            strategy=strategy,
        )

        ASSERT_RESULT[(ArgKmin, dtype)](
            argkmin_distances,
            argkmin_distances_ref,
            argkmin_indices,
            argkmin_indices_ref,
        )


@pytest.mark.parametrize("n_features", [50, 500])
@pytest.mark.parametrize("translation", [0, 1e6])
@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pairwise_distances_radius_neighbors(
    global_random_seed,
    n_features,
    translation,
    metric,
    strategy,
    dtype,
    n_samples=100,
):
    rng = np.random.RandomState(global_random_seed)
    spread = 1000
    radius = spread * np.log(n_features)
    X = translation + rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread

    metric_kwargs = _get_metric_params_list(
        metric, n_features, seed=global_random_seed
    )[0]

    # Reference for argkmin results
    if metric == "euclidean":
        # Compare to scikit-learn GEMM optimized implementation
        dist_matrix = euclidean_distances(X, Y)
    else:
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)

    # Getting the neighbors for a given radius
    neigh_indices_ref = []
    neigh_distances_ref = []

    for row in dist_matrix:
        ind = np.arange(row.shape[0])[row <= radius]
        dist = row[ind]

        sort = np.argsort(dist)
        ind, dist = ind[sort], dist[sort]

        neigh_indices_ref.append(ind)
        neigh_distances_ref.append(dist)

    neigh_distances, neigh_indices = RadiusNeighbors.compute(
        X,
        Y,
        radius,
        metric=metric,
        metric_kwargs=metric_kwargs,
        return_distance=True,
        # So as to have more than a chunk, forcing parallelism.
        chunk_size=n_samples // 4,
        strategy=strategy,
        sort_results=True,
    )

    ASSERT_RESULT[(RadiusNeighbors, dtype)](
        neigh_distances, neigh_distances_ref, neigh_indices, neigh_indices_ref, radius
    )


@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("metric", ["manhattan", "euclidean"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_memmap_backed_data(
    metric,
    Dispatcher,
    dtype,
):
    """Check that the results do not depend on the datasets writability."""
    rng = np.random.RandomState(0)
    spread = 100
    n_samples, n_features = 128, 10
    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread

    # Create read only datasets
    X_mm, Y_mm = create_memmap_backed_data([X, Y])

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Scaling the radius slightly with the numbers of dimensions
        radius = 10 ** np.log(n_features)
        parameter = radius
        check_parameters = {"radius": radius}
        compute_parameters = {"sort_results": True}

    ref_dist, ref_indices = Dispatcher.compute(
        X,
        Y,
        parameter,
        metric=metric,
        return_distance=True,
        **compute_parameters,
    )

    dist_mm, indices_mm = Dispatcher.compute(
        X_mm,
        Y_mm,
        parameter,
        metric=metric,
        return_distance=True,
        **compute_parameters,
    )

    ASSERT_RESULT[(Dispatcher, dtype)](
        ref_dist, dist_mm, ref_indices, indices_mm, **check_parameters
    )


@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_features", [5, 10, 100])
@pytest.mark.parametrize("num_threads", [1, 2, 8])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sqeuclidean_row_norms(
    global_random_seed,
    n_samples,
    n_features,
    num_threads,
    dtype,
):
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    X = rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_matrix(X)

    sq_row_norm_reference = np.linalg.norm(X, axis=1) ** 2
    sq_row_norm = sqeuclidean_row_norms(X, num_threads=num_threads)

    sq_row_norm_csr = sqeuclidean_row_norms(X_csr, num_threads=num_threads)

    assert_allclose(sq_row_norm_reference, sq_row_norm)
    assert_allclose(sq_row_norm_reference, sq_row_norm_csr)

    with pytest.raises(ValueError):
        X = np.asfortranarray(X)
        sqeuclidean_row_norms(X, num_threads=num_threads)


def test_argkmin_classmode_strategy_consistent():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    k = 5
    metric = "manhattan"

    weights = "uniform"
    labels = rng.randint(low=0, high=10, size=100)
    unique_labels = np.unique(labels)
    results_X = ArgKminClassMode.compute(
        X=X,
        Y=Y,
        k=k,
        metric=metric,
        weights=weights,
        labels=labels,
        unique_labels=unique_labels,
        strategy="parallel_on_X",
    )
    results_Y = ArgKminClassMode.compute(
        X=X,
        Y=Y,
        k=k,
        metric=metric,
        weights=weights,
        labels=labels,
        unique_labels=unique_labels,
        strategy="parallel_on_Y",
    )
    assert_array_equal(results_X, results_Y)
