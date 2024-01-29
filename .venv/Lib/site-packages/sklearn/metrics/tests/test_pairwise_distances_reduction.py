import itertools
import re
import warnings
from functools import partial

import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist

from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
    ArgKmin,
    ArgKminClassMode,
    BaseDistancesReductionDispatcher,
    RadiusNeighbors,
    RadiusNeighborsClassMode,
    sqeuclidean_row_norms,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    create_memmap_backed_data,
)
from sklearn.utils.fixes import CSR_CONTAINERS

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


def assert_same_distances_for_common_neighbors(
    query_idx,
    dist_row_a,
    dist_row_b,
    indices_row_a,
    indices_row_b,
    rtol,
    atol,
):
    """Check that the distances of common neighbors are equal up to tolerance.

    This does not check if there are missing neighbors in either result set.
    Missingness is handled by assert_no_missing_neighbors.
    """
    # Compute a mapping from indices to distances for each result set and
    # check that the computed neighbors with matching indices are within
    # the expected distance tolerance.
    indices_to_dist_a = dict(zip(indices_row_a, dist_row_a))
    indices_to_dist_b = dict(zip(indices_row_b, dist_row_b))

    common_indices = set(indices_row_a).intersection(set(indices_row_b))
    for idx in common_indices:
        dist_a = indices_to_dist_a[idx]
        dist_b = indices_to_dist_b[idx]
        try:
            assert_allclose(dist_a, dist_b, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Wrap exception to provide more context while also including
            # the original exception with the computed absolute and
            # relative differences.
            raise AssertionError(
                f"Query vector with index {query_idx} lead to different distances"
                f" for common neighbor with index {idx}:"
                f" dist_a={dist_a} vs dist_b={dist_b} (with atol={atol} and"
                f" rtol={rtol})"
            ) from e


def assert_no_missing_neighbors(
    query_idx,
    dist_row_a,
    dist_row_b,
    indices_row_a,
    indices_row_b,
    threshold,
):
    """Compare the indices of neighbors in two results sets.

    Any neighbor index with a distance below the precision threshold should
    match one in the other result set. We ignore the last few neighbors beyond
    the threshold as those can typically be missing due to rounding errors.

    For radius queries, the threshold is just the radius minus the expected
    precision level.

    For k-NN queries, it is the maximum distance to the k-th neighbor minus the
    expected precision level.
    """
    mask_a = dist_row_a < threshold
    mask_b = dist_row_b < threshold
    missing_from_b = np.setdiff1d(indices_row_a[mask_a], indices_row_b)
    missing_from_a = np.setdiff1d(indices_row_b[mask_b], indices_row_a)
    if len(missing_from_a) > 0 or len(missing_from_b) > 0:
        raise AssertionError(
            f"Query vector with index {query_idx} lead to mismatched result indices:\n"
            f"neighbors in b missing from a: {missing_from_a}\n"
            f"neighbors in a missing from b: {missing_from_b}\n"
            f"dist_row_a={dist_row_a}\n"
            f"dist_row_b={dist_row_b}\n"
            f"indices_row_a={indices_row_a}\n"
            f"indices_row_b={indices_row_b}\n"
        )


def assert_compatible_argkmin_results(
    neighbors_dists_a,
    neighbors_dists_b,
    neighbors_indices_a,
    neighbors_indices_b,
    rtol=1e-5,
    atol=1e-6,
):
    """Assert that argkmin results are valid up to rounding errors.

    This function asserts that the results of argkmin queries are valid up to:
    - rounding error tolerance on distance values;
    - permutations of indices for distances values that differ up to the
      expected precision level.

    Furthermore, the distances must be sorted.

    To be used for testing neighbors queries on float32 datasets: we accept
    neighbors rank swaps only if they are caused by small rounding errors on
    the distance computations.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    assert (
        neighbors_dists_a.shape
        == neighbors_dists_b.shape
        == neighbors_indices_a.shape
        == neighbors_indices_b.shape
    ), "Arrays of results have incompatible shapes."

    n_queries, _ = neighbors_dists_a.shape

    # Asserting equality results one row at a time
    for query_idx in range(n_queries):
        dist_row_a = neighbors_dists_a[query_idx]
        dist_row_b = neighbors_dists_b[query_idx]
        indices_row_a = neighbors_indices_a[query_idx]
        indices_row_b = neighbors_indices_b[query_idx]

        assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
        assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"

        assert_same_distances_for_common_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            rtol,
            atol,
        )

        # Check that any neighbor with distances below the rounding error
        # threshold have matching indices. The threshold is the distance to the
        # k-th neighbors minus the expected precision level:
        #
        # (1 - rtol) * dist_k - atol
        #
        # Where dist_k is defined as the maximum distance to the kth-neighbor
        # among the two result sets. This way of defining the threshold is
        # stricter than taking the minimum of the two.
        threshold = (1 - rtol) * np.maximum(
            np.max(dist_row_a), np.max(dist_row_b)
        ) - atol
        assert_no_missing_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            threshold,
        )


def _non_trivial_radius(
    *,
    X=None,
    Y=None,
    metric=None,
    precomputed_dists=None,
    expected_n_neighbors=10,
    n_subsampled_queries=10,
    **metric_kwargs,
):
    # Find a non-trivial radius using a small subsample of the pairwise
    # distances between X and Y: we want to return around expected_n_neighbors
    # on average. Yielding too many results would make the test slow (because
    # checking the results is expensive for large result sets), yielding 0 most
    # of the time would make the test useless.
    assert (
        precomputed_dists is not None or metric is not None
    ), "Either metric or precomputed_dists must be provided."

    if precomputed_dists is None:
        assert X is not None
        assert Y is not None
        sampled_dists = pairwise_distances(X, Y, metric=metric, **metric_kwargs)
    else:
        sampled_dists = precomputed_dists[:n_subsampled_queries].copy()
    sampled_dists.sort(axis=1)
    return sampled_dists[:, expected_n_neighbors].mean()


def assert_compatible_radius_results(
    neighbors_dists_a,
    neighbors_dists_b,
    neighbors_indices_a,
    neighbors_indices_b,
    radius,
    check_sorted=True,
    rtol=1e-5,
    atol=1e-6,
):
    """Assert that radius neighborhood results are valid up to:

      - relative and absolute tolerance on computed distance values
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

    assert (
        len(neighbors_dists_a)
        == len(neighbors_dists_b)
        == len(neighbors_indices_a)
        == len(neighbors_indices_b)
    )

    n_queries = len(neighbors_dists_a)

    # Asserting equality of results one vector at a time
    for query_idx in range(n_queries):
        dist_row_a = neighbors_dists_a[query_idx]
        dist_row_b = neighbors_dists_b[query_idx]
        indices_row_a = neighbors_indices_a[query_idx]
        indices_row_b = neighbors_indices_b[query_idx]

        if check_sorted:
            assert is_sorted(dist_row_a), f"Distances aren't sorted on row {query_idx}"
            assert is_sorted(dist_row_b), f"Distances aren't sorted on row {query_idx}"

        assert len(dist_row_a) == len(indices_row_a)
        assert len(dist_row_b) == len(indices_row_b)

        # Check that all distances are within the requested radius
        if len(dist_row_a) > 0:
            max_dist_a = np.max(dist_row_a)
            assert max_dist_a <= radius, (
                f"Largest returned distance {max_dist_a} not within requested"
                f" radius {radius} on row {query_idx}"
            )
        if len(dist_row_b) > 0:
            max_dist_b = np.max(dist_row_b)
            assert max_dist_b <= radius, (
                f"Largest returned distance {max_dist_b} not within requested"
                f" radius {radius} on row {query_idx}"
            )

        assert_same_distances_for_common_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            rtol,
            atol,
        )

        threshold = (1 - rtol) * radius - atol
        assert_no_missing_neighbors(
            query_idx,
            dist_row_a,
            dist_row_b,
            indices_row_a,
            indices_row_b,
            threshold,
        )


FLOAT32_TOLS = {
    "atol": 1e-7,
    "rtol": 1e-5,
}
FLOAT64_TOLS = {
    "atol": 1e-9,
    "rtol": 1e-7,
}
ASSERT_RESULT = {
    (ArgKmin, np.float64): partial(assert_compatible_argkmin_results, **FLOAT64_TOLS),
    (ArgKmin, np.float32): partial(assert_compatible_argkmin_results, **FLOAT32_TOLS),
    (
        RadiusNeighbors,
        np.float64,
    ): partial(assert_compatible_radius_results, **FLOAT64_TOLS),
    (
        RadiusNeighbors,
        np.float32,
    ): partial(assert_compatible_radius_results, **FLOAT32_TOLS),
}


def test_assert_compatible_argkmin_results():
    atol = 1e-7
    rtol = 0.0
    tols = dict(atol=atol, rtol=rtol)

    eps = atol / 3
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
    assert_compatible_argkmin_results(
        ref_dist, ref_dist, ref_indices, ref_indices, rtol
    )

    # Apply valid permutation on indices: the last 3 points are all very close
    # to one another so we accept any permutation on their rankings.
    assert_compatible_argkmin_results(
        np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
        np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 5, 4, 3]]),
        **tols,
    )

    # The last few indices do not necessarily have to match because of the rounding
    # errors on the distances: there could be tied results at the boundary.
    assert_compatible_argkmin_results(
        np.array([[1.2, 2.5, 3.0, 6.1, _6_1p]]),
        np.array([[1.2, 2.5, 3.0, _6_1m, 6.1]]),
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 3, 6, 7]]),
        **tols,
    )

    # All points have close distances so any ranking permutation
    # is valid for this query result.
    assert_compatible_argkmin_results(
        np.array([[_1m, 1, _1p, _1p, _1p]]),
        np.array([[1, 1, 1, 1, _1p]]),
        np.array([[7, 6, 8, 10, 9]]),
        np.array([[6, 9, 7, 8, 10]]),
        **tols,
    )

    # They could also be nearly truncation of very large nearly tied result
    # sets hence all indices can also be distinct in this case:
    assert_compatible_argkmin_results(
        np.array([[_1m, 1, _1p, _1p, _1p]]),
        np.array([[_1m, 1, 1, 1, _1p]]),
        np.array([[34, 30, 8, 12, 24]]),
        np.array([[42, 1, 21, 13, 3]]),
        **tols,
    )

    # Apply invalid permutation on indices: permuting the ranks of the 2
    # nearest neighbors is invalid because the distance values are too
    # different.
    msg = re.escape(
        "Query vector with index 0 lead to different distances for common neighbor with"
        " index 1: dist_a=1.2 vs dist_b=2.5"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 3, 4, 5]]),
            **tols,
        )

    # Detect missing indices within the expected precision level, even when the
    # distances match exactly.
    msg = re.escape(
        "neighbors in b missing from a: [12]\nneighbors in a missing from b: [1]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[12, 2, 4, 11, 3]]),
            **tols,
        )

    # Detect missing indices outside the expected precision level.
    msg = re.escape(
        "neighbors in b missing from a: []\nneighbors in a missing from b: [3]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[_1m, 1.0, _6_1m, 6.1, _6_1p]]),
            np.array([[1.0, 1.0, _6_1m, 6.1, 7]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 4, 5, 12]]),
            **tols,
        )

    # Detect missing indices outside the expected precision level, in the other
    # direction:
    msg = re.escape(
        "neighbors in b missing from a: [5]\nneighbors in a missing from b: []"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[_1m, 1.0, _6_1m, 6.1, 7]]),
            np.array([[1.0, 1.0, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 12]]),
            np.array([[2, 1, 5, 3, 4]]),
            **tols,
        )

    # Distances aren't properly sorted
    msg = "Distances aren't sorted on row 0"
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_argkmin_results(
            np.array([[1.2, 2.5, _6_1m, 6.1, _6_1p]]),
            np.array([[2.5, 1.2, _6_1m, 6.1, _6_1p]]),
            np.array([[1, 2, 3, 4, 5]]),
            np.array([[2, 1, 4, 5, 3]]),
            **tols,
        )


@pytest.mark.parametrize("check_sorted", [True, False])
def test_assert_compatible_radius_results(check_sorted):
    atol = 1e-7
    rtol = 0.0
    tols = dict(atol=atol, rtol=rtol)

    eps = atol / 3
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
    assert_compatible_radius_results(
        ref_dist,
        ref_dist,
        ref_indices,
        ref_indices,
        radius=7.0,
        check_sorted=check_sorted,
        **tols,
    )

    # Apply valid permutation on indices
    assert_compatible_radius_results(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
        np.array([np.array([1, 2, 3, 4, 5])]),
        np.array([np.array([1, 2, 4, 5, 3])]),
        radius=7.0,
        check_sorted=check_sorted,
        **tols,
    )
    assert_compatible_radius_results(
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),
        np.array([np.array([_1m, _1m, 1, _1p, _1p])]),
        np.array([np.array([6, 7, 8, 9, 10])]),
        np.array([np.array([6, 9, 7, 8, 10])]),
        radius=7.0,
        check_sorted=check_sorted,
        **tols,
    )

    # Apply invalid permutation on indices
    msg = re.escape(
        "Query vector with index 0 lead to different distances for common neighbor with"
        " index 1: dist_a=1.2 vs dist_b=2.5"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 3, 4, 5])]),
            radius=7.0,
            check_sorted=check_sorted,
            **tols,
        )

    # Having extra last or missing elements is valid if they are in the
    # tolerated rounding error range: [(1 - rtol) * radius - atol, radius]
    assert_compatible_radius_results(
        np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p, _6_1p])]),
        np.array([np.array([1.2, 2.5, _6_1m, 6.1])]),
        np.array([np.array([1, 2, 3, 4, 5, 7])]),
        np.array([np.array([1, 2, 3, 6])]),
        radius=_6_1p,
        check_sorted=check_sorted,
        **tols,
    )

    # Any discrepancy outside the tolerated rounding error range is invalid and
    # indicates a missing neighbor in one of the result sets.
    msg = re.escape(
        "Query vector with index 0 lead to mismatched result indices:\nneighbors in b"
        " missing from a: []\nneighbors in a missing from b: [3]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, 6])]),
            np.array([np.array([1.2, 2.5])]),
            np.array([np.array([1, 2, 3])]),
            np.array([np.array([1, 2])]),
            radius=6.1,
            check_sorted=check_sorted,
            **tols,
        )
    msg = re.escape(
        "Query vector with index 0 lead to mismatched result indices:\nneighbors in b"
        " missing from a: [4]\nneighbors in a missing from b: [2]"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.1, 2.5])]),
            np.array([np.array([1.2, 2, 2.5])]),
            np.array([np.array([1, 2, 3])]),
            np.array([np.array([1, 4, 3])]),
            radius=6.1,
            check_sorted=check_sorted,
            **tols,
        )

    # Radius upper bound is strictly checked
    msg = re.escape(
        "Largest returned distance 6.100000033333333 not within requested radius 6.1 on"
        " row 0"
    )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 4, 5, 3])]),
            radius=6.1,
            check_sorted=check_sorted,
            **tols,
        )
    with pytest.raises(AssertionError, match=msg):
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, 6.1])]),
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 4, 5, 3])]),
            radius=6.1,
            check_sorted=check_sorted,
            **tols,
        )

    if check_sorted:
        # Distances aren't properly sorted
        msg = "Distances aren't sorted on row 0"
        with pytest.raises(AssertionError, match=msg):
            assert_compatible_radius_results(
                np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
                np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]),
                np.array([np.array([1, 2, 3, 4, 5])]),
                np.array([np.array([2, 1, 4, 5, 3])]),
                radius=_6_1p,
                check_sorted=True,
                **tols,
            )
    else:
        assert_compatible_radius_results(
            np.array([np.array([1.2, 2.5, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([2.5, 1.2, _6_1m, 6.1, _6_1p])]),
            np.array([np.array([1, 2, 3, 4, 5])]),
            np.array([np.array([2, 1, 4, 5, 3])]),
            radius=_6_1p,
            check_sorted=False,
            **tols,
        )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_reduction_is_usable_for(csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    X_csr = csr_container(X)
    Y_csr = csr_container(Y)
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
    X_csr_0_nnz = csr_container(X * 0)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_0_nnz, Y, metric)

    # CSR matrices with int64 indices and indptr (e.g. large nnz, or large n_features)
    # aren't supported as of now.
    # See: https://github.com/scikit-learn/scikit-learn/issues/23653
    # TODO: support CSR matrices with int64 indices and indptr
    X_csr_int64 = csr_container(X)
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
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)

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
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
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
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    with pytest.raises(ValueError, match="k == -1, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=-1,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    with pytest.raises(ValueError, match="k == 0, must be >= 1."):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=0,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    with pytest.raises(ValueError, match="Unrecognized metric"):
        ArgKminClassMode.compute(
            X=X,
            Y=Y,
            k=k,
            metric="wrong metric",
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
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
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
        )

    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        ArgKminClassMode.compute(
            X=np.asfortranarray(X),
            Y=Y,
            k=k,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
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
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
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


def test_radius_neighbors_classmode_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = "manhattan"
    weights = "uniform"
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float32 and Y.dtype=float64"
    )
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(
            X=X.astype(np.float32),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    msg = (
        "Only float64 or float32 datasets pairs are supported at this time, "
        "got: X.dtype=float64 and Y.dtype=int32"
    )
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y.astype(np.int32),
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    with pytest.raises(ValueError, match="radius == -1.0, must be >= 0."):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=-1,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    with pytest.raises(ValueError, match="Unrecognized metric"):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=-1,
            metric="wrong_metric",
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    with pytest.raises(
        ValueError, match=r"Buffer has wrong number of dimensions \(expected 2, got 1\)"
    ):
        RadiusNeighborsClassMode.compute(
            X=np.array([1.0, 2.0]),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    with pytest.raises(ValueError, match="ndarray is not C-contiguous"):
        RadiusNeighborsClassMode.compute(
            X=np.asfortranarray(X),
            Y=Y,
            radius=radius,
            metric=metric,
            weights=weights,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )

    non_existent_weights_strategy = "non_existent_weights_strategy"
    msg = (
        "Only the 'uniform' or 'distance' weights options are supported at this time. "
        f"Got: weights='{non_existent_weights_strategy}'."
    )
    with pytest.raises(ValueError, match=msg):
        RadiusNeighborsClassMode.compute(
            X=X,
            Y=Y,
            radius=radius,
            metric="wrong_metric",
            weights=non_existent_weights_strategy,
            Y_labels=Y_labels,
            unique_Y_labels=unique_Y_labels,
            outlier_label=None,
        )


@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_chunk_size_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the chunk size."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")
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


@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_n_threads_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
    n_features=100,
):
    """Check that results do not depend on the number of threads."""
    rng = np.random.RandomState(global_random_seed)
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(dtype) * spread

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")
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
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_format_agnosticism(
    global_random_seed,
    Dispatcher,
    dtype,
    csr_container,
):
    """Check that results do not depend on the format (dense, sparse) of the input."""
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples, n_features = 100, 100

    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    Y = rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_container(X)
    Y_csr = csr_container(Y)

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        # Adjusting the radius to ensure that the expected results is neither
        # trivially empty nor too large.
        radius = _non_trivial_radius(X=X, Y=Y, metric="euclidean")
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


@pytest.mark.parametrize("Dispatcher", [ArgKmin, RadiusNeighbors])
def test_strategies_consistency(
    global_random_seed,
    global_dtype,
    Dispatcher,
    n_features=10,
):
    """Check that the results do not depend on the strategy used."""
    rng = np.random.RandomState(global_random_seed)
    metric = rng.choice(
        np.array(
            [
                "euclidean",
                "minkowski",
                "manhattan",
                "haversine",
            ],
            dtype=object,
        )
    )
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(global_dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(global_dtype) * spread

    # Haversine distance only accepts 2D data
    if metric == "haversine":
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])

    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric=metric)
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

    ASSERT_RESULT[(Dispatcher, global_dtype)](
        dist_par_X, dist_par_Y, indices_par_X, indices_par_Y, **check_parameters
    )


# "Concrete Dispatchers"-specific tests


@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_argkmin(
    global_random_seed,
    metric,
    strategy,
    dtype,
    csr_container,
    n_queries=5,
    n_samples=100,
    k=10,
):
    rng = np.random.RandomState(global_random_seed)
    n_features = rng.choice([50, 500])
    translation = rng.choice([0, 1e6])
    spread = 1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_container(X)
    Y_csr = csr_container(Y)

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


@pytest.mark.parametrize("metric", CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize("strategy", ("parallel_on_X", "parallel_on_Y"))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pairwise_distances_radius_neighbors(
    global_random_seed,
    metric,
    strategy,
    dtype,
    n_queries=5,
    n_samples=100,
):
    rng = np.random.RandomState(global_random_seed)
    n_features = rng.choice([50, 500])
    translation = rng.choice([0, 1e6])
    spread = 1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread
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

    radius = _non_trivial_radius(precomputed_dists=dist_matrix)

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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sqeuclidean_row_norms(
    global_random_seed,
    dtype,
    csr_container,
):
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples = rng.choice([97, 100, 101, 1000])
    n_features = rng.choice([5, 10, 100])
    num_threads = rng.choice([1, 2, 8])
    X = rng.rand(n_samples, n_features).astype(dtype) * spread

    X_csr = csr_container(X)

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
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    results_X = ArgKminClassMode.compute(
        X=X,
        Y=Y,
        k=k,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        strategy="parallel_on_X",
    )
    results_Y = ArgKminClassMode.compute(
        X=X,
        Y=Y,
        k=k,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        strategy="parallel_on_Y",
    )
    assert_array_equal(results_X, results_Y)


@pytest.mark.parametrize("outlier_label", [None, 0, 3, 6, 9])
def test_radius_neighbors_classmode_strategy_consistent(outlier_label):
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = "manhattan"

    weights = "uniform"
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    results_X = RadiusNeighborsClassMode.compute(
        X=X,
        Y=Y,
        radius=radius,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        outlier_label=outlier_label,
        strategy="parallel_on_X",
    )
    results_Y = RadiusNeighborsClassMode.compute(
        X=X,
        Y=Y,
        radius=radius,
        metric=metric,
        weights=weights,
        Y_labels=Y_labels,
        unique_Y_labels=unique_Y_labels,
        outlier_label=outlier_label,
        strategy="parallel_on_Y",
    )
    assert_allclose(results_X, results_Y)
