import functools
import warnings
from typing import Any, List

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.exceptions import DataDimensionalityWarning
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    _gaussian_random_matrix,
    _sparse_random_matrix,
    johnson_lindenstrauss_min_dim,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

all_sparse_random_matrix: List[Any] = [_sparse_random_matrix]
all_dense_random_matrix: List[Any] = [_gaussian_random_matrix]
all_random_matrix = all_sparse_random_matrix + all_dense_random_matrix

all_SparseRandomProjection: List[Any] = [SparseRandomProjection]
all_DenseRandomProjection: List[Any] = [GaussianRandomProjection]
all_RandomProjection = all_SparseRandomProjection + all_DenseRandomProjection


# Make some random data with uniformly located non zero entries with
# Gaussian distributed values
def make_sparse_random_data(n_samples, n_features, n_nonzeros, random_state=0):
    rng = np.random.RandomState(random_state)
    data_coo = sp.coo_matrix(
        (
            rng.randn(n_nonzeros),
            (
                rng.randint(n_samples, size=n_nonzeros),
                rng.randint(n_features, size=n_nonzeros),
            ),
        ),
        shape=(n_samples, n_features),
    )
    return data_coo.toarray(), data_coo.tocsr()


def densify(matrix):
    if not sp.issparse(matrix):
        return matrix
    else:
        return matrix.toarray()


n_samples, n_features = (10, 1000)
n_nonzeros = int(n_samples * n_features / 100.0)
data, data_csr = make_sparse_random_data(n_samples, n_features, n_nonzeros)


###############################################################################
# test on JL lemma
###############################################################################


@pytest.mark.parametrize(
    "n_samples, eps",
    [
        ([100, 110], [0.9, 1.1]),
        ([90, 100], [0.1, 0.0]),
        ([50, -40], [0.1, 0.2]),
    ],
)
def test_invalid_jl_domain(n_samples, eps):
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(n_samples, eps=eps)


def test_input_size_jl_min_dim():
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(3 * [100], eps=2 * [0.9])

    johnson_lindenstrauss_min_dim(
        np.random.randint(1, 10, size=(10, 10)), eps=np.full((10, 10), 0.5)
    )


###############################################################################
# tests random matrix generation
###############################################################################
def check_input_size_random_matrix(random_matrix):
    inputs = [(0, 0), (-1, 1), (1, -1), (1, 0), (-1, 0)]
    for n_components, n_features in inputs:
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features)


def check_size_generated(random_matrix):
    inputs = [(1, 5), (5, 1), (5, 5), (1, 1)]
    for n_components, n_features in inputs:
        assert random_matrix(n_components, n_features).shape == (
            n_components,
            n_features,
        )


def check_zero_mean_and_unit_norm(random_matrix):
    # All random matrix should produce a transformation matrix
    # with zero mean and unit norm for each columns

    A = densify(random_matrix(10000, 1, random_state=0))

    assert_array_almost_equal(0, np.mean(A), 3)
    assert_array_almost_equal(1.0, np.linalg.norm(A), 1)


def check_input_with_sparse_random_matrix(random_matrix):
    n_components, n_features = 5, 10

    for density in [-1.0, 0.0, 1.1]:
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features, density=density)


@pytest.mark.parametrize("random_matrix", all_random_matrix)
def test_basic_property_of_random_matrix(random_matrix):
    # Check basic properties of random matrix generation
    check_input_size_random_matrix(random_matrix)
    check_size_generated(random_matrix)
    check_zero_mean_and_unit_norm(random_matrix)


@pytest.mark.parametrize("random_matrix", all_sparse_random_matrix)
def test_basic_property_of_sparse_random_matrix(random_matrix):
    check_input_with_sparse_random_matrix(random_matrix)

    random_matrix_dense = functools.partial(random_matrix, density=1.0)

    check_zero_mean_and_unit_norm(random_matrix_dense)


def test_gaussian_random_matrix():
    # Check some statical properties of Gaussian random matrix
    # Check that the random matrix follow the proper distribution.
    # Let's say that each element of a_{ij} of A is taken from
    #   a_ij ~ N(0.0, 1 / n_components).
    #
    n_components = 100
    n_features = 1000
    A = _gaussian_random_matrix(n_components, n_features, random_state=0)

    assert_array_almost_equal(0.0, np.mean(A), 2)
    assert_array_almost_equal(np.var(A, ddof=1), 1 / n_components, 1)


def test_sparse_random_matrix():
    # Check some statical properties of sparse random matrix
    n_components = 100
    n_features = 500

    for density in [0.3, 1.0]:
        s = 1 / density

        A = _sparse_random_matrix(
            n_components, n_features, density=density, random_state=0
        )
        A = densify(A)

        # Check possible values
        values = np.unique(A)
        assert np.sqrt(s) / np.sqrt(n_components) in values
        assert -np.sqrt(s) / np.sqrt(n_components) in values

        if density == 1.0:
            assert np.size(values) == 2
        else:
            assert 0.0 in values
            assert np.size(values) == 3

        # Check that the random matrix follow the proper distribution.
        # Let's say that each element of a_{ij} of A is taken from
        #
        # - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
        # -  0                              with probability 1 - 1 / s
        # - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s
        #
        assert_almost_equal(np.mean(A == 0.0), 1 - 1 / s, decimal=2)
        assert_almost_equal(
            np.mean(A == np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2
        )
        assert_almost_equal(
            np.mean(A == -np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2
        )

        assert_almost_equal(np.var(A == 0.0, ddof=1), (1 - 1 / s) * 1 / s, decimal=2)
        assert_almost_equal(
            np.var(A == np.sqrt(s) / np.sqrt(n_components), ddof=1),
            (1 - 1 / (2 * s)) * 1 / (2 * s),
            decimal=2,
        )
        assert_almost_equal(
            np.var(A == -np.sqrt(s) / np.sqrt(n_components), ddof=1),
            (1 - 1 / (2 * s)) * 1 / (2 * s),
            decimal=2,
        )


###############################################################################
# tests on random projection transformer
###############################################################################


def test_random_projection_transformer_invalid_input():
    n_components = "auto"
    fit_data = [[0, 1, 2]]
    for RandomProjection in all_RandomProjection:
        with pytest.raises(ValueError):
            RandomProjection(n_components=n_components).fit(fit_data)


def test_try_to_transform_before_fit():
    for RandomProjection in all_RandomProjection:
        with pytest.raises(ValueError):
            RandomProjection(n_components="auto").transform(data)


def test_too_many_samples_to_find_a_safe_embedding():
    data, _ = make_sparse_random_data(1000, 100, 1000)

    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", eps=0.1)
        expected_msg = (
            "eps=0.100000 and n_samples=1000 lead to a target dimension"
            " of 5920 which is larger than the original space with"
            " n_features=100"
        )
        with pytest.raises(ValueError, match=expected_msg):
            rp.fit(data)


def test_random_projection_embedding_quality():
    data, _ = make_sparse_random_data(8, 5000, 15000)
    eps = 0.2

    original_distances = euclidean_distances(data, squared=True)
    original_distances = original_distances.ravel()
    non_identical = original_distances != 0.0

    # remove 0 distances to avoid division by 0
    original_distances = original_distances[non_identical]

    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", eps=eps, random_state=0)
        projected = rp.fit_transform(data)

        projected_distances = euclidean_distances(projected, squared=True)
        projected_distances = projected_distances.ravel()

        # remove 0 distances to avoid division by 0
        projected_distances = projected_distances[non_identical]

        distances_ratio = projected_distances / original_distances

        # check that the automatically tuned values for the density respect the
        # contract for eps: pairwise distances are preserved according to the
        # Johnson-Lindenstrauss lemma
        assert distances_ratio.max() < 1 + eps
        assert 1 - eps < distances_ratio.min()


def test_SparseRandomProj_output_representation():
    for SparseRandomProj in all_SparseRandomProjection:
        # when using sparse input, the projected data can be forced to be a
        # dense numpy array
        rp = SparseRandomProj(n_components=10, dense_output=True, random_state=0)
        rp.fit(data)
        assert isinstance(rp.transform(data), np.ndarray)

        sparse_data = sp.csr_matrix(data)
        assert isinstance(rp.transform(sparse_data), np.ndarray)

        # the output can be left to a sparse matrix instead
        rp = SparseRandomProj(n_components=10, dense_output=False, random_state=0)
        rp = rp.fit(data)
        # output for dense input will stay dense:
        assert isinstance(rp.transform(data), np.ndarray)

        # output for sparse output will be sparse:
        assert sp.issparse(rp.transform(sparse_data))


def test_correct_RandomProjection_dimensions_embedding():
    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components="auto", random_state=0, eps=0.5).fit(data)

        # the number of components is adjusted from the shape of the training
        # set
        assert rp.n_components == "auto"
        assert rp.n_components_ == 110

        if RandomProjection in all_SparseRandomProjection:
            assert rp.density == "auto"
            assert_almost_equal(rp.density_, 0.03, 2)

        assert rp.components_.shape == (110, n_features)

        projected_1 = rp.transform(data)
        assert projected_1.shape == (n_samples, 110)

        # once the RP is 'fitted' the projection is always the same
        projected_2 = rp.transform(data)
        assert_array_equal(projected_1, projected_2)

        # fit transform with same random seed will lead to the same results
        rp2 = RandomProjection(random_state=0, eps=0.5)
        projected_3 = rp2.fit_transform(data)
        assert_array_equal(projected_1, projected_3)

        # Try to transform with an input X of size different from fitted.
        with pytest.raises(ValueError):
            rp.transform(data[:, 1:5])

        # it is also possible to fix the number of components and the density
        # level
        if RandomProjection in all_SparseRandomProjection:
            rp = RandomProjection(n_components=100, density=0.001, random_state=0)
            projected = rp.fit_transform(data)
            assert projected.shape == (n_samples, 100)
            assert rp.components_.shape == (100, n_features)
            assert rp.components_.nnz < 115  # close to 1% density
            assert 85 < rp.components_.nnz  # close to 1% density


def test_warning_n_components_greater_than_n_features():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for RandomProjection in all_RandomProjection:
        with pytest.warns(DataDimensionalityWarning):
            RandomProjection(n_components=n_features + 1).fit(data)


def test_works_with_sparse_data():
    n_features = 20
    data, _ = make_sparse_random_data(5, n_features, int(n_features / 4))

    for RandomProjection in all_RandomProjection:
        rp_dense = RandomProjection(n_components=3, random_state=1).fit(data)
        rp_sparse = RandomProjection(n_components=3, random_state=1).fit(
            sp.csr_matrix(data)
        )
        assert_array_almost_equal(
            densify(rp_dense.components_), densify(rp_sparse.components_)
        )


def test_johnson_lindenstrauss_min_dim():
    """Test Johnson-Lindenstrauss for small eps.

    Regression test for #17111: before #19374, 32-bit systems would fail.
    """
    assert johnson_lindenstrauss_min_dim(100, eps=1e-5) == 368416070986


@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
def test_random_projection_feature_names_out(random_projection_cls):
    random_projection = random_projection_cls(n_components=2)
    random_projection.fit(data)
    names_out = random_projection.get_feature_names_out()
    class_name_lower = random_projection_cls.__name__.lower()
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(random_projection.n_components_)],
        dtype=object,
    )

    assert_array_equal(names_out, expected_names_out)


@pytest.mark.parametrize("n_samples", (2, 9, 10, 11, 1000))
@pytest.mark.parametrize("n_features", (2, 9, 10, 11, 1000))
@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
@pytest.mark.parametrize("compute_inverse_components", [True, False])
def test_inverse_transform(
    n_samples,
    n_features,
    random_projection_cls,
    compute_inverse_components,
    global_random_seed,
):
    n_components = 10

    random_projection = random_projection_cls(
        n_components=n_components,
        compute_inverse_components=compute_inverse_components,
        random_state=global_random_seed,
    )

    X_dense, X_csr = make_sparse_random_data(
        n_samples,
        n_features,
        n_samples * n_features // 100 + 1,
        random_state=global_random_seed,
    )

    for X in [X_dense, X_csr]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The number of components is higher than the number of features"
                ),
                category=DataDimensionalityWarning,
            )
            projected = random_projection.fit_transform(X)

        if compute_inverse_components:
            assert hasattr(random_projection, "inverse_components_")
            inv_components = random_projection.inverse_components_
            assert inv_components.shape == (n_features, n_components)

        projected_back = random_projection.inverse_transform(projected)
        assert projected_back.shape == X.shape

        projected_again = random_projection.transform(projected_back)
        if hasattr(projected, "toarray"):
            projected = projected.toarray()
        assert_allclose(projected, projected_again, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
@pytest.mark.parametrize(
    "input_dtype, expected_dtype",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_random_projection_dtype_match(
    random_projection_cls, input_dtype, expected_dtype
):
    # Verify output matrix dtype
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp = random_projection_cls(random_state=0)
    transformed = rp.fit_transform(X.astype(input_dtype))

    assert rp.components_.dtype == expected_dtype
    assert transformed.dtype == expected_dtype


@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
def test_random_projection_numerical_consistency(random_projection_cls):
    # Verify numerical consistency among np.float32 and np.float64
    atol = 1e-5
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp_32 = random_projection_cls(random_state=0)
    rp_64 = random_projection_cls(random_state=0)

    projection_32 = rp_32.fit_transform(X.astype(np.float32))
    projection_64 = rp_64.fit_transform(X.astype(np.float64))

    assert_allclose(projection_64, projection_32, atol=atol)

    assert_allclose_dense_sparse(rp_32.components_, rp_64.components_)
