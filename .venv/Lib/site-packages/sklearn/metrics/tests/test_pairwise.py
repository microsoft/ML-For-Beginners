import warnings
from types import GeneratorType

import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
    cdist,
    cityblock,
    cosine,
    minkowski,
    pdist,
    squareform,
)

try:
    from scipy.spatial.distance import wminkowski
except ImportError:
    # In scipy 1.6.0, wminkowski is deprecated and minkowski
    # should be used instead.
    from scipy.spatial.distance import minkowski as wminkowski

import pytest

from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    PAIRWISE_BOOLEAN_FUNCTIONS,
    PAIRWISE_DISTANCE_FUNCTIONS,
    PAIRWISE_KERNEL_FUNCTIONS,
    _euclidean_distances_upcast,
    additive_chi2_kernel,
    check_paired_arrays,
    check_pairwise_arrays,
    chi2_kernel,
    cosine_distances,
    cosine_similarity,
    euclidean_distances,
    haversine_distances,
    laplacian_kernel,
    linear_kernel,
    manhattan_distances,
    nan_euclidean_distances,
    paired_cosine_distances,
    paired_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
    pairwise_distances,
    pairwise_distances_argmin,
    pairwise_distances_argmin_min,
    pairwise_distances_chunked,
    pairwise_kernels,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.parallel import Parallel, delayed


def test_pairwise_distances_for_dense_data(global_dtype):
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)

    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, metric="euclidean")
    S2 = euclidean_distances(X)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Euclidean distance, with Y != X.
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X, Y, metric="euclidean")
    S2 = euclidean_distances(X, Y)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Check to ensure NaNs work with pairwise_distances.
    X_masked = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y_masked = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    X_masked[0, 0] = np.nan
    Y_masked[0, 0] = np.nan
    S_masked = pairwise_distances(X_masked, Y_masked, metric="nan_euclidean")
    S2_masked = nan_euclidean_distances(X_masked, Y_masked)
    assert_allclose(S_masked, S2_masked)
    assert S_masked.dtype == S2_masked.dtype == global_dtype

    # Test with tuples as X and Y
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    S2 = pairwise_distances(X_tuples, Y_tuples, metric="euclidean")
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # Test haversine distance
    # The data should be valid latitude and longitude
    # haversine converts to float64 currently so we don't check dtypes.
    X = rng.random_sample((5, 2)).astype(global_dtype, copy=False)
    X[:, 0] = (X[:, 0] - 0.5) * 2 * np.pi / 2
    X[:, 1] = (X[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, metric="haversine")
    S2 = haversine_distances(X)
    assert_allclose(S, S2)

    # Test haversine distance, with Y != X
    Y = rng.random_sample((2, 2)).astype(global_dtype, copy=False)
    Y[:, 0] = (Y[:, 0] - 0.5) * 2 * np.pi / 2
    Y[:, 1] = (Y[:, 1] - 0.5) * 2 * np.pi
    S = pairwise_distances(X, Y, metric="haversine")
    S2 = haversine_distances(X, Y)
    assert_allclose(S, S2)

    # "cityblock" uses scikit-learn metric, cityblock (function) is
    # scipy.spatial.
    # The metric functions from scipy converts to float64 so we don't check the dtypes.
    S = pairwise_distances(X, metric="cityblock")
    S2 = pairwise_distances(X, metric=cityblock)
    assert S.shape[0] == S.shape[1]
    assert S.shape[0] == X.shape[0]
    assert_allclose(S, S2)

    # The manhattan metric should be equivalent to cityblock.
    S = pairwise_distances(X, Y, metric="manhattan")
    S2 = pairwise_distances(X, Y, metric=cityblock)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)

    # Test cosine as a string metric versus cosine callable
    # The string "cosine" uses sklearn.metric,
    # while the function cosine is scipy.spatial
    S = pairwise_distances(X, Y, metric="cosine")
    S2 = pairwise_distances(X, Y, metric=cosine)
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    assert_allclose(S, S2)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("bsr_container", BSR_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_for_sparse_data(
    coo_container, csc_container, bsr_container, csr_container, global_dtype
):
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)

    # Test with sparse X and Y,
    # currently only supported for Euclidean, L1 and cosine.
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)

    S = pairwise_distances(X_sparse, Y_sparse, metric="euclidean")
    S2 = euclidean_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    S = pairwise_distances(X_sparse, Y_sparse, metric="cosine")
    S2 = cosine_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    S = pairwise_distances(X_sparse, csc_container(Y), metric="manhattan")
    S2 = manhattan_distances(bsr_container(X), coo_container(Y))
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO Fix manhattan_distances to preserve dtype.
        # currently pairwise_distances uses manhattan_distances but converts the result
        # back to the input dtype
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    S2 = manhattan_distances(X, Y)
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO Fix manhattan_distances to preserve dtype.
        # currently pairwise_distances uses manhattan_distances but converts the result
        # back to the input dtype
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    # Test with scipy.spatial.distance metric, with a kwd
    kwds = {"p": 2.0}
    S = pairwise_distances(X, Y, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, Y, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # same with Y = None
    kwds = {"p": 2.0}
    S = pairwise_distances(X, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # Test that scipy distance metrics throw an error if sparse matrix given
    with pytest.raises(TypeError):
        pairwise_distances(X_sparse, metric="minkowski")
    with pytest.raises(TypeError):
        pairwise_distances(X, Y_sparse, metric="minkowski")


@pytest.mark.parametrize("metric", PAIRWISE_BOOLEAN_FUNCTIONS)
def test_pairwise_boolean_distance(metric):
    # test that we convert to boolean arrays for boolean distances
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    Y = X.copy()
    Y[0, 0] = 1 - Y[0, 0]

    # ignore conversion to boolean in pairwise_distances
    with ignore_warnings(category=DataConversionWarning):
        for Z in [Y, None]:
            res = pairwise_distances(X, Z, metric=metric)
            np.nan_to_num(res, nan=0, posinf=0, neginf=0, copy=False)
            assert np.sum(res != 0) == 0

    # non-boolean arrays are converted to boolean for boolean
    # distance metrics with a data conversion warning
    msg = "Data was converted to boolean for metric %s" % metric
    with pytest.warns(DataConversionWarning, match=msg):
        pairwise_distances(X, metric=metric)

    # Check that the warning is raised if X is boolean by Y is not boolean:
    with pytest.warns(DataConversionWarning, match=msg):
        pairwise_distances(X.astype(bool), Y=Y, metric=metric)

    # Check that no warning is raised if X is already boolean and Y is None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        pairwise_distances(X.astype(bool), metric=metric)


def test_no_data_conversion_warning():
    # No warnings issued if metric is not a boolean distance function
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        pairwise_distances(X, metric="minkowski")


@pytest.mark.parametrize("func", [pairwise_distances, pairwise_kernels])
def test_pairwise_precomputed(func):
    # Test correct shape
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), metric="precomputed")
    # with two args
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), np.zeros((4, 4)), metric="precomputed")
    # even if shape[1] agrees (although thus second arg is spurious)
    with pytest.raises(ValueError, match=".* shape .*"):
        func(np.zeros((5, 3)), np.zeros((4, 3)), metric="precomputed")

    # Test not copied (if appropriate dtype)
    S = np.zeros((5, 5))
    S2 = func(S, metric="precomputed")
    assert S is S2
    # with two args
    S = np.zeros((5, 3))
    S2 = func(S, np.zeros((3, 3)), metric="precomputed")
    assert S is S2

    # Test always returns float dtype
    S = func(np.array([[1]], dtype="int"), metric="precomputed")
    assert "f" == S.dtype.kind

    # Test converts list to array-like
    S = func([[1.0]], metric="precomputed")
    assert isinstance(S, np.ndarray)


def test_pairwise_precomputed_non_negative():
    # Test non-negative values
    with pytest.raises(ValueError, match=".* non-negative values.*"):
        pairwise_distances(np.full((5, 5), -1), metric="precomputed")


_minkowski_kwds = {"w": np.arange(1, 5).astype("double", copy=False), "p": 1}
_wminkowski_kwds = {"w": np.arange(1, 5).astype("double", copy=False), "p": 1}


def callable_rbf_kernel(x, y, **kwds):
    # Callable version of pairwise.rbf_kernel.
    K = rbf_kernel(np.atleast_2d(x), np.atleast_2d(y), **kwds)
    # unpack the output since this is a scalar packed in a 0-dim array
    return K.item()


@pytest.mark.parametrize(
    "func, metric, kwds",
    [
        (pairwise_distances, "euclidean", {}),
        pytest.param(
            pairwise_distances,
            minkowski,
            _minkowski_kwds,
        ),
        pytest.param(
            pairwise_distances,
            "minkowski",
            _minkowski_kwds,
        ),
        pytest.param(
            pairwise_distances,
            wminkowski,
            _wminkowski_kwds,
            marks=pytest.mark.skipif(
                sp_version >= parse_version("1.6.0"),
                reason="wminkowski is now minkowski and it has been already tested.",
            ),
        ),
        pytest.param(
            pairwise_distances,
            "wminkowski",
            _wminkowski_kwds,
            marks=pytest.mark.skipif(
                sp_version >= parse_version("1.6.0"),
                reason="wminkowski is now minkowski and it has been already tested.",
            ),
        ),
        (pairwise_kernels, "polynomial", {"degree": 1}),
        (pairwise_kernels, callable_rbf_kernel, {"gamma": 0.1}),
    ],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, int])
def test_pairwise_parallel(func, metric, kwds, dtype):
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample((5, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((3, 4)), dtype=dtype)

    S = func(X, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)

    S = func(X, Y, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, Y, metric=metric, n_jobs=2, **kwds)
    assert_allclose(S, S2)


def test_pairwise_callable_nonstrict_metric():
    # paired_distances should allow callable metric where metric(x, x) != 0
    # Knowing that the callable is a strict metric would allow the diagonal to
    # be left uncalculated and set to 0.
    assert pairwise_distances([[1.0]], metric=lambda x, y: 5)[0, 0] == 5


# Test with all metrics that should be in PAIRWISE_KERNEL_FUNCTIONS.
@pytest.mark.parametrize(
    "metric",
    ["rbf", "laplacian", "sigmoid", "polynomial", "linear", "chi2", "additive_chi2"],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_kernels(metric, csr_container):
    # Test the pairwise_kernels helper function.

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    function = PAIRWISE_KERNEL_FUNCTIONS[metric]
    # Test with Y=None
    K1 = pairwise_kernels(X, metric=metric)
    K2 = function(X)
    assert_allclose(K1, K2)
    # Test with Y=Y
    K1 = pairwise_kernels(X, Y=Y, metric=metric)
    K2 = function(X, Y=Y)
    assert_allclose(K1, K2)
    # Test with tuples as X and Y
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    K2 = pairwise_kernels(X_tuples, Y_tuples, metric=metric)
    assert_allclose(K1, K2)

    # Test with sparse X and Y
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)
    if metric in ["chi2", "additive_chi2"]:
        # these don't support sparse matrices yet
        return
    K1 = pairwise_kernels(X_sparse, Y=Y_sparse, metric=metric)
    assert_allclose(K1, K2)


def test_pairwise_kernels_callable():
    # Test the pairwise_kernels helper function
    # with a callable function, with given keywords.
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))

    metric = callable_rbf_kernel
    kwds = {"gamma": 0.1}
    K1 = pairwise_kernels(X, Y=Y, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=Y, **kwds)
    assert_allclose(K1, K2)

    # callable function, X=Y
    K1 = pairwise_kernels(X, Y=X, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=X, **kwds)
    assert_allclose(K1, K2)


def test_pairwise_kernels_filter_param():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    K = rbf_kernel(X, Y, gamma=0.1)
    params = {"gamma": 0.1, "blabla": ":)"}
    K2 = pairwise_kernels(X, Y, metric="rbf", filter_params=True, **params)
    assert_allclose(K, K2)

    with pytest.raises(TypeError):
        pairwise_kernels(X, Y, metric="rbf", **params)


@pytest.mark.parametrize("metric, func", PAIRED_DISTANCES.items())
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_paired_distances(metric, func, csr_container):
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4))
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((5, 4))

    S = paired_distances(X, Y, metric=metric)
    S2 = func(X, Y)
    assert_allclose(S, S2)
    S3 = func(csr_container(X), csr_container(Y))
    assert_allclose(S, S3)
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        # Check the pairwise_distances implementation
        # gives the same value
        distances = PAIRWISE_DISTANCE_FUNCTIONS[metric](X, Y)
        distances = np.diag(distances)
        assert_allclose(distances, S)


def test_paired_distances_callable(global_dtype):
    # Test the paired_distance helper function
    # with the callable implementation
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)

    S = paired_distances(X, Y, metric="manhattan")
    S2 = paired_distances(X, Y, metric=lambda x, y: np.abs(x - y).sum(axis=0))
    assert_allclose(S, S2)

    # Test that a value error is raised when the lengths of X and Y should not
    # differ
    Y = rng.random_sample((3, 4))
    with pytest.raises(ValueError):
        paired_distances(X, Y)


@pytest.mark.parametrize("dok_container", DOK_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_argmin_min(dok_container, csr_container, global_dtype):
    # Check pairwise minimum distances computation for any metric
    X = np.asarray([[0], [1]], dtype=global_dtype)
    Y = np.asarray([[-2], [3]], dtype=global_dtype)

    Xsp = dok_container(X)
    Ysp = csr_container(Y, dtype=global_dtype)

    expected_idx = [0, 1]
    expected_vals = [2, 2]
    expected_vals_sq = [4, 4]

    # euclidean metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="euclidean")
    idx2 = pairwise_distances_argmin(X, Y, metric="euclidean")
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    # sparse matrix case
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="euclidean")
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="euclidean")
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)
    # We don't want np.matrix here
    assert type(idxsp) == np.ndarray
    assert type(valssp) == np.ndarray

    # Squared Euclidean metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="sqeuclidean")
    idx2, vals2 = pairwise_distances_argmin_min(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )
    idx3 = pairwise_distances_argmin(X, Y, metric="sqeuclidean")
    idx4 = pairwise_distances_argmin(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )

    assert_allclose(vals, expected_vals_sq)
    assert_allclose(vals2, expected_vals_sq)

    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(idx3, expected_idx)
    assert_allclose(idx4, expected_idx)

    # Non-euclidean scikit-learn metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="manhattan")
    idx2 = pairwise_distances_argmin(X, Y, metric="manhattan")
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    # sparse matrix case
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="manhattan")
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="manhattan")
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)

    # Non-euclidean Scipy distance (callable)
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric=minkowski, metric_kwargs={"p": 2}
    )
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # Non-euclidean Scipy distance (string)
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric="minkowski", metric_kwargs={"p": 2}
    )
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # Compare with naive implementation
    rng = np.random.RandomState(0)
    X = rng.randn(97, 149)
    Y = rng.randn(111, 149)

    dist = pairwise_distances(X, Y, metric="manhattan")
    dist_orig_ind = dist.argmin(axis=0)
    dist_orig_val = dist[dist_orig_ind, range(len(dist_orig_ind))]

    dist_chunked_ind, dist_chunked_val = pairwise_distances_argmin_min(
        X, Y, axis=0, metric="manhattan"
    )
    assert_allclose(dist_orig_ind, dist_chunked_ind, rtol=1e-7)
    assert_allclose(dist_orig_val, dist_chunked_val, rtol=1e-7)

    # Changing the axis and permuting datasets must give the same results
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, Y, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(Y, X, axis=1)

    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    argmin_0, dist_0 = pairwise_distances_argmin_min(X, X, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(X, X, axis=1)

    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    # Changing the axis and permuting datasets must give the same results
    argmin_0 = pairwise_distances_argmin(X, Y, axis=0)
    argmin_1 = pairwise_distances_argmin(Y, X, axis=1)

    assert_array_equal(argmin_0, argmin_1)

    argmin_0 = pairwise_distances_argmin(X, X, axis=0)
    argmin_1 = pairwise_distances_argmin(X, X, axis=1)

    assert_array_equal(argmin_0, argmin_1)

    # F-contiguous arrays must be supported and must return identical results.
    argmin_C_contiguous = pairwise_distances_argmin(X, Y)
    argmin_F_contiguous = pairwise_distances_argmin(
        np.asfortranarray(X), np.asfortranarray(Y)
    )

    assert_array_equal(argmin_C_contiguous, argmin_F_contiguous)


def _reduce_func(dist, start):
    return dist[:, :100]


def test_pairwise_distances_chunked_reduce(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((400, 4)).astype(global_dtype, copy=False)
    # Reduced Euclidean distance
    S = pairwise_distances(X)[:, :100]
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=_reduce_func, working_memory=2**-16
    )
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert S_chunks[0].dtype == X.dtype

    # atol is for diagonal where S is explicitly zeroed on the diagonal
    assert_allclose(np.vstack(S_chunks), S, atol=1e-7)


def test_pairwise_distances_chunked_reduce_none(global_dtype):
    # check that the reduce func is allowed to return None
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=lambda dist, start: None, working_memory=2**-16
    )
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert all(chunk is None for chunk in S_chunks)


@pytest.mark.parametrize(
    "good_reduce",
    [
        lambda D, start: list(D),
        lambda D, start: np.array(D),
        lambda D, start: (list(D), list(D)),
    ]
    + [
        lambda D, start, scipy_csr_type=scipy_csr_type: scipy_csr_type(D)
        for scipy_csr_type in CSR_CONTAINERS
    ]
    + [
        lambda D, start, scipy_dok_type=scipy_dok_type: (
            scipy_dok_type(D),
            np.array(D),
            list(D),
        )
        for scipy_dok_type in DOK_CONTAINERS
    ],
)
def test_pairwise_distances_chunked_reduce_valid(good_reduce):
    X = np.arange(10).reshape(-1, 1)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=good_reduce, working_memory=64
    )
    next(S_chunks)


@pytest.mark.parametrize(
    ("bad_reduce", "err_type", "message"),
    [
        (
            lambda D, s: np.concatenate([D, D[-1:]]),
            ValueError,
            r"length 11\..* input: 10\.",
        ),
        (
            lambda D, s: (D, np.concatenate([D, D[-1:]])),
            ValueError,
            r"length \(10, 11\)\..* input: 10\.",
        ),
        (lambda D, s: (D[:9], D), ValueError, r"length \(9, 10\)\..* input: 10\."),
        (
            lambda D, s: 7,
            TypeError,
            r"returned 7\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (7, 8),
            TypeError,
            r"returned \(7, 8\)\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (np.arange(10), 9),
            TypeError,
            r", 9\)\. Expected sequence\(s\) of length 10\.",
        ),
    ],
)
def test_pairwise_distances_chunked_reduce_invalid(
    global_dtype, bad_reduce, err_type, message
):
    X = np.arange(10).reshape(-1, 1).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=bad_reduce, working_memory=64
    )
    with pytest.raises(err_type, match=message):
        next(S_chunks)


def check_pairwise_distances_chunked(X, Y, working_memory, metric="euclidean"):
    gen = pairwise_distances_chunked(X, Y, working_memory=working_memory, metric=metric)
    assert isinstance(gen, GeneratorType)
    blockwise_distances = list(gen)
    Y = X if Y is None else Y
    min_block_mib = len(Y) * 8 * 2**-20

    for block in blockwise_distances:
        memory_used = block.nbytes
        assert memory_used <= max(working_memory, min_block_mib) * 2**20

    blockwise_distances = np.vstack(blockwise_distances)
    S = pairwise_distances(X, Y, metric=metric)
    assert_allclose(blockwise_distances, S, atol=1e-7)


@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_pairwise_distances_chunked_diagonal(metric, global_dtype):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    chunks = list(pairwise_distances_chunked(X, working_memory=1, metric=metric))
    assert len(chunks) > 1
    assert_allclose(np.diag(np.vstack(chunks)), 0, rtol=1e-10)


@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_parallel_pairwise_distances_diagonal(metric, global_dtype):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    distances = pairwise_distances(X, metric=metric, n_jobs=2)
    assert_allclose(np.diag(distances), 0, atol=1e-10)


@ignore_warnings
def test_pairwise_distances_chunked(global_dtype):
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)
    # Euclidean distance should be equivalent to calling the function.
    X = rng.random_sample((200, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, None, working_memory=1, metric="euclidean")
    # Test small amounts of memory
    for power in range(-16, 0):
        check_pairwise_distances_chunked(
            X, None, working_memory=2**power, metric="euclidean"
        )
    # X as list
    check_pairwise_distances_chunked(
        X.tolist(), None, working_memory=1, metric="euclidean"
    )
    # Euclidean distance, with Y != X.
    Y = rng.random_sample((100, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="euclidean")
    check_pairwise_distances_chunked(
        X.tolist(), Y.tolist(), working_memory=1, metric="euclidean"
    )
    # absurdly large working_memory
    check_pairwise_distances_chunked(X, Y, working_memory=10000, metric="euclidean")
    # "cityblock" uses scikit-learn metric, cityblock (function) is
    # scipy.spatial.
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="cityblock")

    # Test precomputed returns all at once
    D = pairwise_distances(X)
    gen = pairwise_distances_chunked(D, working_memory=2**-16, metric="precomputed")
    assert isinstance(gen, GeneratorType)
    assert next(gen) is D
    with pytest.raises(StopIteration):
        next(gen)


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_known_result(x_array_constr, y_array_constr):
    # Check the pairwise Euclidean distances computation on known result
    X = x_array_constr([[0]])
    Y = y_array_constr([[1], [2]])
    D = euclidean_distances(X, Y)
    assert_allclose(D, [[1.0, 2.0]])


@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_with_norms(global_dtype, y_array_constr):
    # check that we still get the right answers with {X,Y}_norm_squared
    # and that we get a wrong answer with wrong {X,Y}_norm_squared
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10)).astype(global_dtype, copy=False)
    Y = rng.random_sample((20, 10)).astype(global_dtype, copy=False)

    # norms will only be used if their dtype is float64
    X_norm_sq = (X.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)

    Y = y_array_constr(Y)

    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)

    # check we get the wrong answer with wrong {X,Y}_norm_squared
    wrong_D = euclidean_distances(
        X,
        Y,
        X_norm_squared=np.zeros_like(X_norm_sq),
        Y_norm_squared=np.zeros_like(Y_norm_sq),
    )
    with pytest.raises(AssertionError):
        assert_allclose(wrong_D, D1)


@pytest.mark.parametrize("symmetric", [True, False])
def test_euclidean_distances_float32_norms(global_random_seed, symmetric):
    # Non-regression test for #27621
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((10, 10))
    Y = X if symmetric else rng.random_sample((20, 10))
    X_norm_sq = (X.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)


def test_euclidean_distances_norm_shapes():
    # Check all accepted shapes for the norms or appropriate error messages.
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10))
    Y = rng.random_sample((20, 10))

    X_norm_squared = (X**2).sum(axis=1)
    Y_norm_squared = (Y**2).sum(axis=1)

    D1 = euclidean_distances(
        X, Y, X_norm_squared=X_norm_squared, Y_norm_squared=Y_norm_squared
    )
    D2 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(-1, 1),
        Y_norm_squared=Y_norm_squared.reshape(-1, 1),
    )
    D3 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(1, -1),
        Y_norm_squared=Y_norm_squared.reshape(1, -1),
    )

    assert_allclose(D2, D1)
    assert_allclose(D3, D1)

    with pytest.raises(ValueError, match="Incompatible dimensions for X"):
        euclidean_distances(X, Y, X_norm_squared=X_norm_squared[:5])
    with pytest.raises(ValueError, match="Incompatible dimensions for Y"):
        euclidean_distances(X, Y, Y_norm_squared=Y_norm_squared[:5])


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances(global_dtype, x_array_constr, y_array_constr):
    # check that euclidean distances gives same result as scipy cdist
    # when X and Y != X are provided
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    X[X < 0.8] = 0
    Y = rng.random_sample((10, 10)).astype(global_dtype, copy=False)
    Y[Y < 0.8] = 0

    expected = cdist(X, Y)

    X = x_array_constr(X)
    Y = y_array_constr(Y)
    distances = euclidean_distances(X, Y)

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)
    assert distances.dtype == global_dtype


@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_sym(global_dtype, x_array_constr):
    # check that euclidean distances gives same result as scipy pdist
    # when only X is provided
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    X[X < 0.8] = 0

    expected = squareform(pdist(X))

    X = x_array_constr(X)
    distances = euclidean_distances(X)

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)
    assert distances.dtype == global_dtype


@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_upcast(batch_size, x_array_constr, y_array_constr):
    # check batches handling when Y != X (#13910)
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    X[X < 0.8] = 0
    Y = rng.random_sample((10, 10)).astype(np.float32)
    Y[Y < 0.8] = 0

    expected = cdist(X, Y)

    X = x_array_constr(X)
    Y = y_array_constr(Y)
    distances = _euclidean_distances_upcast(X, Y=Y, batch_size=batch_size)
    distances = np.sqrt(np.maximum(distances, 0))

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_upcast_sym(batch_size, x_array_constr):
    # check batches handling when X is Y (#13910)
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    X[X < 0.8] = 0

    expected = squareform(pdist(X))

    X = x_array_constr(X)
    distances = _euclidean_distances_upcast(X, Y=X, batch_size=batch_size)
    distances = np.sqrt(np.maximum(distances, 0))

    # the default rtol=1e-7 is too close to the float32 precision
    # and fails due to rounding errors.
    assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "dtype, eps, rtol",
    [
        (np.float32, 1e-4, 1e-5),
        pytest.param(
            np.float64,
            1e-8,
            0.99,
            marks=pytest.mark.xfail(reason="failing due to lack of precision"),
        ),
    ],
)
@pytest.mark.parametrize("dim", [1, 1000000])
def test_euclidean_distances_extreme_values(dtype, eps, rtol, dim):
    # check that euclidean distances is correct with float32 input thanks to
    # upcasting. On float64 there are still precision issues.
    X = np.array([[1.0] * dim], dtype=dtype)
    Y = np.array([[1.0 + eps] * dim], dtype=dtype)

    distances = euclidean_distances(X, Y)
    expected = cdist(X, Y)

    assert_allclose(distances, expected, rtol=1e-5)


@pytest.mark.parametrize("squared", [True, False])
def test_nan_euclidean_distances_equal_to_euclidean_distance(squared):
    # with no nan values
    rng = np.random.RandomState(1337)
    X = rng.randn(3, 4)
    Y = rng.randn(4, 4)

    normal_distance = euclidean_distances(X, Y=Y, squared=squared)
    nan_distance = nan_euclidean_distances(X, Y=Y, squared=squared)
    assert_allclose(normal_distance, nan_distance)


@pytest.mark.parametrize("X", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]])])
@pytest.mark.parametrize("Y", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]]), None])
def test_nan_euclidean_distances_infinite_values(X, Y):
    with pytest.raises(ValueError) as excinfo:
        nan_euclidean_distances(X, Y=Y)

    exp_msg = "Input contains infinity or a value too large for dtype('float64')."
    assert exp_msg == str(excinfo.value)


@pytest.mark.parametrize(
    "X, X_diag, missing_value",
    [
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[np.nan, 1], [1, np.nan]]), np.nan, np.nan),
        (np.array([[np.nan, 1], [np.nan, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, np.nan], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), -1),
        (np.array([[0, 1], [1, -1]]), np.sqrt(2), -1),
        (np.array([[-1, 1], [1, -1]]), np.nan, -1),
        (np.array([[-1, 1], [-1, 0]]), np.sqrt(2), -1),
        (np.array([[0, -1], [1, -1]]), np.sqrt(2), -1),
    ],
)
def test_nan_euclidean_distances_2x2(X, X_diag, missing_value):
    exp_dist = np.array([[0.0, X_diag], [X_diag, 0]])

    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)

    dist_sq = nan_euclidean_distances(X, squared=True, missing_values=missing_value)
    assert_allclose(exp_dist**2, dist_sq)

    dist_two = nan_euclidean_distances(X, X, missing_values=missing_value)
    assert_allclose(exp_dist, dist_two)

    dist_two_copy = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist_two_copy)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_complete_nan(missing_value):
    X = np.array([[missing_value, missing_value], [0, 1]])

    exp_dist = np.array([[np.nan, np.nan], [np.nan, 0]])

    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)

    dist = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_not_trival(missing_value):
    X = np.array(
        [
            [1.0, missing_value, 3.0, 4.0, 2.0],
            [missing_value, 4.0, 6.0, 1.0, missing_value],
            [3.0, missing_value, missing_value, missing_value, 1.0],
        ]
    )

    Y = np.array(
        [
            [missing_value, 7.0, 7.0, missing_value, 2.0],
            [missing_value, missing_value, 5.0, 4.0, 7.0],
            [missing_value, missing_value, missing_value, 4.0, 5.0],
        ]
    )

    # Check for symmetry
    D1 = nan_euclidean_distances(X, Y, missing_values=missing_value)
    D2 = nan_euclidean_distances(Y, X, missing_values=missing_value)

    assert_almost_equal(D1, D2.T)

    # Check with explicit formula and squared=True
    assert_allclose(
        nan_euclidean_distances(
            X[:1], Y[:1], squared=True, missing_values=missing_value
        ),
        [[5.0 / 2.0 * ((7 - 3) ** 2 + (2 - 2) ** 2)]],
    )

    # Check with explicit formula and squared=False
    assert_allclose(
        nan_euclidean_distances(
            X[1:2], Y[1:2], squared=False, missing_values=missing_value
        ),
        [[np.sqrt(5.0 / 2.0 * ((6 - 5) ** 2 + (1 - 4) ** 2))]],
    )

    # Check when Y = X is explicitly passed
    D3 = nan_euclidean_distances(X, missing_values=missing_value)
    D4 = nan_euclidean_distances(X, X, missing_values=missing_value)
    D5 = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(D3, D4)
    assert_allclose(D4, D5)

    # Check copy = True against copy = False
    D6 = nan_euclidean_distances(X, Y, copy=True)
    D7 = nan_euclidean_distances(X, Y, copy=False)
    assert_allclose(D6, D7)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_one_feature_match_positive(missing_value):
    # First feature is the only feature that is non-nan and in both
    # samples. The result of `nan_euclidean_distances` with squared=True
    # should be non-negative. The non-squared version should all be close to 0.
    X = np.array(
        [
            [-122.27, 648.0, missing_value, 37.85],
            [-122.27, missing_value, 2.34701493, missing_value],
        ]
    )

    dist_squared = nan_euclidean_distances(
        X, missing_values=missing_value, squared=True
    )
    assert np.all(dist_squared >= 0)

    dist = nan_euclidean_distances(X, missing_values=missing_value, squared=False)
    assert_allclose(dist, 0.0)


def test_cosine_distances():
    # Check the pairwise Cosine distances computation
    rng = np.random.RandomState(1337)
    x = np.abs(rng.rand(910))
    XA = np.vstack([x, x])
    D = cosine_distances(XA)
    assert_allclose(D, [[0.0, 0.0], [0.0, 0.0]], atol=1e-10)
    # check that all elements are in [0, 2]
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)
    # check that diagonal elements are equal to 0
    assert_allclose(D[np.diag_indices_from(D)], [0.0, 0.0])

    XB = np.vstack([x, -x])
    D2 = cosine_distances(XB)
    # check that all elements are in [0, 2]
    assert np.all(D2 >= 0.0)
    assert np.all(D2 <= 2.0)
    # check that diagonal elements are equal to 0 and non diagonal to 2
    assert_allclose(D2, [[0.0, 2.0], [2.0, 0.0]])

    # check large random matrix
    X = np.abs(rng.rand(1000, 5000))
    D = cosine_distances(X)
    # check that diagonal elements are equal to 0
    assert_allclose(D[np.diag_indices_from(D)], [0.0] * D.shape[0])
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)


def test_haversine_distances():
    # Check haversine distance with distances computation
    def slow_haversine_distances(x, y):
        diff_lat = y[0] - x[0]
        diff_lon = y[1] - x[1]
        a = np.sin(diff_lat / 2) ** 2 + (
            np.cos(x[0]) * np.cos(y[0]) * np.sin(diff_lon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return c

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 2))
    Y = rng.random_sample((10, 2))
    D1 = np.array([[slow_haversine_distances(x, y) for y in Y] for x in X])
    D2 = haversine_distances(X, Y)
    assert_allclose(D1, D2)
    # Test haversine distance does not accept X where n_feature != 2
    X = rng.random_sample((10, 3))
    err_msg = "Haversine distance only valid in 2 dimensions"
    with pytest.raises(ValueError, match=err_msg):
        haversine_distances(X)


# Paired distances


def test_paired_euclidean_distances():
    # Check the paired Euclidean distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_euclidean_distances(X, Y)
    assert_allclose(D, [1.0, 2.0])


def test_paired_manhattan_distances():
    # Check the paired manhattan distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_manhattan_distances(X, Y)
    assert_allclose(D, [1.0, 2.0])


def test_paired_cosine_distances():
    # Check the paired manhattan distances computation
    X = [[0], [0]]
    Y = [[1], [2]]
    D = paired_cosine_distances(X, Y)
    assert_allclose(D, [0.5, 0.5])


def test_chi_square_kernel():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((10, 4))
    K_add = additive_chi2_kernel(X, Y)
    gamma = 0.1
    K = chi2_kernel(X, Y, gamma=gamma)
    assert K.dtype == float
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            chi2 = -np.sum((x - y) ** 2 / (x + y))
            chi2_exp = np.exp(gamma * chi2)
            assert_almost_equal(K_add[i, j], chi2)
            assert_almost_equal(K[i, j], chi2_exp)

    # check diagonal is ones for data with itself
    K = chi2_kernel(Y)
    assert_array_equal(np.diag(K), 1)
    # check off-diagonal is < 1 but > 0:
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)
    # check that float32 is preserved
    X = rng.random_sample((5, 4)).astype(np.float32)
    Y = rng.random_sample((10, 4)).astype(np.float32)
    K = chi2_kernel(X, Y)
    assert K.dtype == np.float32

    # check integer type gets converted,
    # check that zeros are handled
    X = rng.random_sample((10, 4)).astype(np.int32)
    K = chi2_kernel(X, X)
    assert np.isfinite(K).all()
    assert K.dtype == float

    # check that kernel of similar things is greater than dissimilar ones
    X = [[0.3, 0.7], [1.0, 0]]
    Y = [[0, 1], [0.9, 0.1]]
    K = chi2_kernel(X, Y)
    assert K[0, 0] > K[0, 1]
    assert K[1, 1] > K[1, 0]

    # test negative input
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]], [[-1, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[-1, -1]])

    # different n_features in X and Y
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[0.2, 0.2, 0.6]])


@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
def test_kernel_symmetry(kernel):
    # Valid kernels should be symmetric
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = kernel(X, X)
    assert_allclose(K, K.T, 15)


@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_kernel_sparse(kernel, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    X_sparse = csr_container(X)
    K = kernel(X, X)
    K2 = kernel(X_sparse, X_sparse)
    assert_allclose(K, K2)


def test_linear_kernel():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = linear_kernel(X, X)
    # the diagonal elements of a linear kernel are their squared norm
    assert_allclose(K.flat[::6], [linalg.norm(x) ** 2 for x in X])


def test_rbf_kernel():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = rbf_kernel(X, X)
    # the diagonal elements of a rbf kernel are 1
    assert_allclose(K.flat[::6], np.ones(5))


def test_laplacian_kernel():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    K = laplacian_kernel(X, X)
    # the diagonal elements of a laplacian kernel are 1
    assert_allclose(np.diag(K), np.ones(5))

    # off-diagonal elements are < 1 but > 0:
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)


@pytest.mark.parametrize(
    "metric, pairwise_func",
    [("linear", linear_kernel), ("cosine", cosine_similarity)],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_similarity_sparse_output(metric, pairwise_func, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)

    # should be sparse
    K1 = pairwise_func(Xcsr, Ycsr, dense_output=False)
    assert issparse(K1)

    # should be dense, and equal to K1
    K2 = pairwise_func(X, Y, dense_output=True)
    assert not issparse(K2)
    assert_allclose(K1.toarray(), K2)

    # show the kernel output equal to the sparse.toarray()
    K3 = pairwise_kernels(X, Y=Y, metric=metric)
    assert_allclose(K1.toarray(), K3)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cosine_similarity(csr_container):
    # Test the cosine_similarity.

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)

    for X_, Y_ in ((X, None), (X, Y), (Xcsr, None), (Xcsr, Ycsr)):
        # Test that the cosine is kernel is equal to a linear kernel when data
        # has been previously normalized by L2-norm.
        K1 = pairwise_kernels(X_, Y=Y_, metric="cosine")
        X_ = normalize(X_)
        if Y_ is not None:
            Y_ = normalize(Y_)
        K2 = pairwise_kernels(X_, Y=Y_, metric="linear")
        assert_allclose(K1, K2)


def test_check_dense_matrices():
    # Ensure that pairwise array check works for dense matrices.
    # Check that if XB is None, XB is returned as reference to XA
    XA = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert XA_checked is XB_checked
    assert_array_equal(XA, XA_checked)


def test_check_XB_returned():
    # Ensure that if XA and XB are given correctly, they return as equal.
    # Check that if XB is not None, it is returned equal.
    # Note that the second dimension of XB is the same as XA.
    XA = np.resize(np.arange(40), (5, 8))
    XB = np.resize(np.arange(32), (4, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)

    XB = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_paired_arrays(XA, XB)
    assert_array_equal(XA, XA_checked)
    assert_array_equal(XB, XB_checked)


def test_check_different_dimensions():
    # Ensure an error is raised if the dimensions are different.
    XA = np.resize(np.arange(45), (5, 9))
    XB = np.resize(np.arange(32), (4, 8))
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)

    XB = np.resize(np.arange(4 * 9), (4, 9))
    with pytest.raises(ValueError):
        check_paired_arrays(XA, XB)


def test_check_invalid_dimensions():
    # Ensure an error is raised on 1D input arrays.
    # The modified tests are not 1D. In the old test, the array was internally
    # converted to 2D anyways
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)
    XA = np.arange(45).reshape(9, 5)
    XB = np.arange(32).reshape(4, 8)
    with pytest.raises(ValueError):
        check_pairwise_arrays(XA, XB)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_check_sparse_arrays(csr_container):
    # Ensures that checks return valid sparse matrices.
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_sparse = csr_container(XA)
    XB = rng.random_sample((5, 4))
    XB_sparse = csr_container(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_sparse, XB_sparse)
    # compare their difference because testing csr matrices for
    # equality with '==' does not work as expected.
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XB_checked)
    assert abs(XB_sparse - XB_checked).sum() == 0

    XA_checked, XA_2_checked = check_pairwise_arrays(XA_sparse, XA_sparse)
    assert issparse(XA_checked)
    assert abs(XA_sparse - XA_checked).sum() == 0
    assert issparse(XA_2_checked)
    assert abs(XA_2_checked - XA_checked).sum() == 0


def tuplify(X):
    # Turns a numpy matrix (any n-dimensional array) into tuples.
    s = X.shape
    if len(s) > 1:
        # Tuplify each sub-array in the input.
        return tuple(tuplify(row) for row in X)
    else:
        # Single dimension input, just return tuple of contents.
        return tuple(r for r in X)


def test_check_tuple_input():
    # Ensures that checks return valid tuples.
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_tuples = tuplify(XA)
    XB = rng.random_sample((5, 4))
    XB_tuples = tuplify(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_tuples, XB_tuples)
    assert_array_equal(XA_tuples, XA_checked)
    assert_array_equal(XB_tuples, XB_checked)


def test_check_preserve_type():
    # Ensures that type float32 is preserved.
    XA = np.resize(np.arange(40), (5, 8)).astype(np.float32)
    XB = np.resize(np.arange(40), (5, 8)).astype(np.float32)

    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert XA_checked.dtype == np.float32

    # both float32
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert XA_checked.dtype == np.float32
    assert XB_checked.dtype == np.float32

    # mismatched A
    XA_checked, XB_checked = check_pairwise_arrays(XA.astype(float), XB)
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float

    # mismatched B
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB.astype(float))
    assert XA_checked.dtype == float
    assert XB_checked.dtype == float


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
@pytest.mark.parametrize(
    "dist_function", [pairwise_distances, pairwise_distances_chunked]
)
def test_pairwise_distances_data_derived_params(n_jobs, metric, dist_function):
    # check that pairwise_distances give the same result in sequential and
    # parallel, when metric has data-derived parameters.
    with config_context(working_memory=0.1):  # to have more than 1 chunk
        rng = np.random.RandomState(0)
        X = rng.random_sample((100, 10))

        expected_dist = squareform(pdist(X, metric=metric))
        dist = np.vstack(tuple(dist_function(X, metric=metric, n_jobs=n_jobs)))

        assert_allclose(dist, expected_dist)


@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
def test_pairwise_distances_data_derived_params_error(metric):
    # check that pairwise_distances raises an error when Y is passed but
    # metric has data-derived params that are not provided by the user.
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    Y = rng.random_sample((100, 10))

    with pytest.raises(
        ValueError,
        match=rf"The '(V|VI)' parameter is required for the " rf"{metric} metric",
    ):
        pairwise_distances(X, Y, metric=metric)


@pytest.mark.parametrize(
    "metric",
    [
        "braycurtis",
        "canberra",
        "chebyshev",
        "correlation",
        "hamming",
        "mahalanobis",
        "minkowski",
        "seuclidean",
        "sqeuclidean",
        "cityblock",
        "cosine",
        "euclidean",
    ],
)
@pytest.mark.parametrize("y_is_x", [True, False], ids=["Y is X", "Y is not X"])
def test_numeric_pairwise_distances_datatypes(metric, global_dtype, y_is_x):
    # Check that pairwise distances gives the same result as pdist and cdist
    # regardless of input datatype when using any scipy metric for comparing
    # numeric vectors
    #
    # This test is necessary because pairwise_distances used to throw an
    # error when using metric='seuclidean' and the input data was not
    # of type np.float64 (#15730)

    rng = np.random.RandomState(0)

    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)

    params = {}
    if y_is_x:
        Y = X
        expected_dist = squareform(pdist(X, metric=metric))
    else:
        Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
        expected_dist = cdist(X, Y, metric=metric)
        # precompute parameters for seuclidean & mahalanobis when x is not y
        if metric == "seuclidean":
            params = {"V": np.var(np.vstack([X, Y]), axis=0, ddof=1, dtype=np.float64)}
        elif metric == "mahalanobis":
            params = {"VI": np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T}

    dist = pairwise_distances(X, Y, metric=metric, **params)

    assert_allclose(dist, expected_dist)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_manhattan_readonly_dataset(csr_container):
    # Non-regression test for: https://github.com/scikit-learn/scikit-learn/issues/7981
    matrices1 = [csr_container(np.ones((5, 5)))]
    matrices2 = [csr_container(np.ones((5, 5)))]
    # Joblib memory maps datasets which makes them read-only.
    # The following call was reporting as failing in #7981, but this must pass.
    Parallel(n_jobs=2, max_nbytes=0)(
        delayed(manhattan_distances)(m1, m2) for m1, m2 in zip(matrices1, matrices2)
    )
