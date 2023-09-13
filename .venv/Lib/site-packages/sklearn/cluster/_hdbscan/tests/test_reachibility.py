import numpy as np
import pytest

from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
)


def test_mutual_reachability_graph_error_sparse_format():
    """Check that we raise an error if the sparse format is not CSR."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, "sparse_csc")

    err_msg = "Only sparse CSR matrices are supported"
    with pytest.raises(ValueError, match=err_msg):
        mutual_reachability_graph(X)


@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
def test_mutual_reachability_graph_inplace(array_type):
    """Check that the operation is happening inplace."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)

    mr_graph = mutual_reachability_graph(X)

    assert id(mr_graph) == id(X)


def test_mutual_reachability_graph_equivalence_dense_sparse():
    """Check that we get the same results for dense and sparse implementation."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 5)
    X_dense = X.T @ X
    X_sparse = _convert_container(X_dense, "sparse_csr")

    mr_graph_dense = mutual_reachability_graph(X_dense, min_samples=3)
    mr_graph_sparse = mutual_reachability_graph(X_sparse, min_samples=3)

    assert_allclose(mr_graph_dense, mr_graph_sparse.A)


@pytest.mark.parametrize("array_type", ["array", "sparse_csr"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mutual_reachability_graph_preserve_dtype(array_type, dtype):
    """Check that the computation preserve dtype thanks to fused types."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = (X.T @ X).astype(dtype)
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, array_type)

    assert X.dtype == dtype
    mr_graph = mutual_reachability_graph(X)
    assert mr_graph.dtype == dtype
