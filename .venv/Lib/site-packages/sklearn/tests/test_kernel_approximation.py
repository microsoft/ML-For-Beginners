import re

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    PolynomialCountSketch,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.metrics.pairwise import (
    chi2_kernel,
    kernel_metrics,
    polynomial_kernel,
    rbf_kernel,
)
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= X.sum(axis=1)[:, np.newaxis]
Y /= Y.sum(axis=1)[:, np.newaxis]


@pytest.mark.parametrize("gamma", [0.1, 1, 2.5])
@pytest.mark.parametrize("degree, n_components", [(1, 500), (2, 500), (3, 5000)])
@pytest.mark.parametrize("coef0", [0, 2.5])
def test_polynomial_count_sketch(gamma, degree, coef0, n_components):
    # test that PolynomialCountSketch approximates polynomial
    # kernel on random data

    # compute exact kernel
    kernel = polynomial_kernel(X, Y, gamma=gamma, degree=degree, coef0=coef0)

    # approximate kernel mapping
    ps_transform = PolynomialCountSketch(
        n_components=n_components,
        gamma=gamma,
        coef0=coef0,
        degree=degree,
        random_state=42,
    )
    X_trans = ps_transform.fit_transform(X)
    Y_trans = ps_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.05  # close to unbiased
    np.abs(error, out=error)
    assert np.max(error) <= 0.1  # nothing too far off
    assert np.mean(error) <= 0.05  # mean is fairly close


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("coef0", [0, 2.5])
def test_polynomial_count_sketch_dense_sparse(gamma, degree, coef0, csr_container):
    """Check that PolynomialCountSketch results are the same for dense and sparse
    input.
    """
    ps_dense = PolynomialCountSketch(
        n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42
    )
    Xt_dense = ps_dense.fit_transform(X)
    Yt_dense = ps_dense.transform(Y)

    ps_sparse = PolynomialCountSketch(
        n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42
    )
    Xt_sparse = ps_sparse.fit_transform(csr_container(X))
    Yt_sparse = ps_sparse.transform(csr_container(Y))

    assert_allclose(Xt_dense, Xt_sparse)
    assert_allclose(Yt_dense, Yt_sparse)


def _linear_kernel(X, Y):
    return np.dot(X, Y.T)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_additive_chi2_sampler(csr_container):
    # test that AdditiveChi2Sampler approximates kernel on random data

    # compute exact kernel
    # abbreviations for easier formula
    X_ = X[:, np.newaxis, :]
    Y_ = Y[np.newaxis, :, :]

    large_kernel = 2 * X_ * Y_ / (X_ + Y_)

    # reduce to n_samples_x x n_samples_y by summing over features
    kernel = large_kernel.sum(axis=2)

    # approximate kernel mapping
    transform = AdditiveChi2Sampler(sample_steps=3)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)

    assert_array_almost_equal(kernel, kernel_approx, 1)

    X_sp_trans = transform.fit_transform(csr_container(X))
    Y_sp_trans = transform.transform(csr_container(Y))

    assert_array_equal(X_trans, X_sp_trans.toarray())
    assert_array_equal(Y_trans, Y_sp_trans.toarray())

    # test error is raised on negative input
    Y_neg = Y.copy()
    Y_neg[0, 0] = -1
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        transform.fit(Y_neg)


@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
@pytest.mark.parametrize("sample_steps", range(1, 4))
def test_additive_chi2_sampler_sample_steps(method, sample_steps):
    """Check that the input sample step doesn't raise an error
    and that sample interval doesn't change after fit.
    """
    transformer = AdditiveChi2Sampler(sample_steps=sample_steps)
    getattr(transformer, method)(X)

    sample_interval = 0.5
    transformer = AdditiveChi2Sampler(
        sample_steps=sample_steps,
        sample_interval=sample_interval,
    )
    getattr(transformer, method)(X)
    assert transformer.sample_interval == sample_interval


# TODO(1.5): remove
def test_additive_chi2_sampler_future_warnings():
    """Check that we raise a FutureWarning when accessing to `sample_interval_`."""
    transformer = AdditiveChi2Sampler()
    transformer.fit(X)
    msg = re.escape(
        "The ``sample_interval_`` attribute was deprecated in version 1.3 and "
        "will be removed 1.5."
    )
    with pytest.warns(FutureWarning, match=msg):
        assert transformer.sample_interval_ is not None


@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
def test_additive_chi2_sampler_wrong_sample_steps(method):
    """Check that we raise a ValueError on invalid sample_steps"""
    transformer = AdditiveChi2Sampler(sample_steps=4)
    msg = re.escape(
        "If sample_steps is not in [1, 2, 3], you need to provide sample_interval"
    )
    with pytest.raises(ValueError, match=msg):
        getattr(transformer, method)(X)


def test_skewed_chi2_sampler():
    # test that RBFSampler approximates kernel on random data

    # compute exact kernel
    c = 0.03
    # set on negative component but greater than c to ensure that the kernel
    # approximation is valid on the group (-c; +\infty) endowed with the skewed
    # multiplication.
    Y[0, 0] = -c / 2.0

    # abbreviations for easier formula
    X_c = (X + c)[:, np.newaxis, :]
    Y_c = (Y + c)[np.newaxis, :, :]

    # we do it in log-space in the hope that it's more stable
    # this array is n_samples_x x n_samples_y big x n_features
    log_kernel = (
        (np.log(X_c) / 2.0) + (np.log(Y_c) / 2.0) + np.log(2.0) - np.log(X_c + Y_c)
    )
    # reduce to n_samples_x x n_samples_y by summing over features in log-space
    kernel = np.exp(log_kernel.sum(axis=2))

    # approximate kernel mapping
    transform = SkewedChi2Sampler(skewedness=c, n_components=1000, random_state=42)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)
    assert_array_almost_equal(kernel, kernel_approx, 1)
    assert np.isfinite(kernel).all(), "NaNs found in the Gram matrix"
    assert np.isfinite(kernel_approx).all(), "NaNs found in the approximate Gram matrix"

    # test error is raised on when inputs contains values smaller than -c
    Y_neg = Y.copy()
    Y_neg[0, 0] = -c * 2.0
    msg = "X may not contain entries smaller than -skewedness"
    with pytest.raises(ValueError, match=msg):
        transform.transform(Y_neg)


def test_additive_chi2_sampler_exceptions():
    """Ensures correct error message"""
    transformer = AdditiveChi2Sampler()
    X_neg = X.copy()
    X_neg[0, 0] = -1
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.fit"):
        transformer.fit(X_neg)
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.transform"):
        transformer.fit(X)
        transformer.transform(X_neg)


def test_rbf_sampler():
    # test that RBFSampler approximates kernel on random data
    # compute exact kernel
    gamma = 10.0
    kernel = rbf_kernel(X, Y, gamma=gamma)

    # approximate kernel mapping
    rbf_transform = RBFSampler(gamma=gamma, n_components=1000, random_state=42)
    X_trans = rbf_transform.fit_transform(X)
    Y_trans = rbf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.01  # close to unbiased
    np.abs(error, out=error)
    assert np.max(error) <= 0.1  # nothing too far off
    assert np.mean(error) <= 0.05  # mean is fairly close


def test_rbf_sampler_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    rbf = RBFSampler()

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    rbf.fit(X)

    assert rbf.random_offset_.dtype == global_dtype
    assert rbf.random_weights_.dtype == global_dtype


def test_rbf_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    rbf32 = RBFSampler(random_state=42)
    X32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    rbf32.fit(X32)

    rbf64 = RBFSampler(random_state=42)
    X64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    rbf64.fit(X64)

    assert_allclose(rbf32.random_offset_, rbf64.random_offset_)
    assert_allclose(rbf32.random_weights_, rbf64.random_weights_)


def test_rbf_sampler_gamma_scale():
    """Check the inner value computed when `gamma='scale'`."""
    X, y = [[0.0], [1.0]], [0, 1]
    rbf = RBFSampler(gamma="scale")
    rbf.fit(X, y)
    assert rbf._gamma == pytest.approx(4)


def test_skewed_chi2_sampler_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    skewed_chi2_sampler = SkewedChi2Sampler()

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    skewed_chi2_sampler.fit(X)

    assert skewed_chi2_sampler.random_offset_.dtype == global_dtype
    assert skewed_chi2_sampler.random_weights_.dtype == global_dtype


def test_skewed_chi2_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    skewed_chi2_sampler_32 = SkewedChi2Sampler(random_state=42)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    skewed_chi2_sampler_32.fit(X_32)

    skewed_chi2_sampler_64 = SkewedChi2Sampler(random_state=42)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    skewed_chi2_sampler_64.fit(X_64)

    assert_allclose(
        skewed_chi2_sampler_32.random_offset_, skewed_chi2_sampler_64.random_offset_
    )
    assert_allclose(
        skewed_chi2_sampler_32.random_weights_, skewed_chi2_sampler_64.random_weights_
    )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_input_validation(csr_container):
    # Regression test: kernel approx. transformers should work on lists
    # No assertions; the old versions would simply crash
    X = [[1, 2], [3, 4], [5, 6]]
    AdditiveChi2Sampler().fit(X).transform(X)
    SkewedChi2Sampler().fit(X).transform(X)
    RBFSampler().fit(X).transform(X)

    X = csr_container(X)
    RBFSampler().fit(X).transform(X)


def test_nystroem_approximation():
    # some basic tests
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 4))

    # With n_components = n_samples this is exact
    X_transformed = Nystroem(n_components=X.shape[0]).fit_transform(X)
    K = rbf_kernel(X)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    trans = Nystroem(n_components=2, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    assert X_transformed.shape == (X.shape[0], 2)

    # test callable kernel
    trans = Nystroem(n_components=2, kernel=_linear_kernel, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    assert X_transformed.shape == (X.shape[0], 2)

    # test that available kernels fit and transform
    kernels_available = kernel_metrics()
    for kern in kernels_available:
        trans = Nystroem(n_components=2, kernel=kern, random_state=rnd)
        X_transformed = trans.fit(X).transform(X)
        assert X_transformed.shape == (X.shape[0], 2)


def test_nystroem_default_parameters():
    rnd = np.random.RandomState(42)
    X = rnd.uniform(size=(10, 4))

    # rbf kernel should behave as gamma=None by default
    # aka gamma = 1 / n_features
    nystroem = Nystroem(n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = rbf_kernel(X, gamma=None)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)

    # chi2 kernel should behave as gamma=1 by default
    nystroem = Nystroem(kernel="chi2", n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = chi2_kernel(X, gamma=1)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)


def test_nystroem_singular_kernel():
    # test that nystroem works with singular kernel matrix
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    X = np.vstack([X] * 2)  # duplicate samples

    gamma = 100
    N = Nystroem(gamma=gamma, n_components=X.shape[0]).fit(X)
    X_transformed = N.transform(X)

    K = rbf_kernel(X, gamma=gamma)

    assert_array_almost_equal(K, np.dot(X_transformed, X_transformed.T))
    assert np.all(np.isfinite(Y))


def test_nystroem_poly_kernel_params():
    # Non-regression: Nystroem should pass other parameters beside gamma.
    rnd = np.random.RandomState(37)
    X = rnd.uniform(size=(10, 4))

    K = polynomial_kernel(X, degree=3.1, coef0=0.1)
    nystroem = Nystroem(
        kernel="polynomial", n_components=X.shape[0], degree=3.1, coef0=0.1
    )
    X_transformed = nystroem.fit_transform(X)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)


def test_nystroem_callable():
    # Test Nystroem on a callable.
    rnd = np.random.RandomState(42)
    n_samples = 10
    X = rnd.uniform(size=(n_samples, 4))

    def logging_histogram_kernel(x, y, log):
        """Histogram kernel that writes to a log."""
        log.append(1)
        return np.minimum(x, y).sum()

    kernel_log = []
    X = list(X)  # test input validation
    Nystroem(
        kernel=logging_histogram_kernel,
        n_components=(n_samples - 1),
        kernel_params={"log": kernel_log},
    ).fit(X)
    assert len(kernel_log) == n_samples * (n_samples - 1) / 2

    # if degree, gamma or coef0 is passed, we raise a ValueError
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    for param in params:
        ny = Nystroem(kernel=_linear_kernel, n_components=(n_samples - 1), **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(X)


def test_nystroem_precomputed_kernel():
    # Non-regression: test Nystroem on precomputed kernel.
    # PR - 14706
    rnd = np.random.RandomState(12)
    X = rnd.uniform(size=(10, 4))

    K = polynomial_kernel(X, degree=2, coef0=0.1)
    nystroem = Nystroem(kernel="precomputed", n_components=X.shape[0])
    X_transformed = nystroem.fit_transform(K)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    # if degree, gamma or coef0 is passed, we raise a ValueError
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    for param in params:
        ny = Nystroem(kernel="precomputed", n_components=X.shape[0], **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(K)


def test_nystroem_component_indices():
    """Check that `component_indices_` corresponds to the subset of
    training points used to construct the feature map.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20474
    """
    X, _ = make_classification(n_samples=100, n_features=20)
    feature_map_nystroem = Nystroem(
        n_components=10,
        random_state=0,
    )
    feature_map_nystroem.fit(X)
    assert feature_map_nystroem.component_indices_.shape == (10,)


@pytest.mark.parametrize(
    "Estimator", [PolynomialCountSketch, RBFSampler, SkewedChi2Sampler, Nystroem]
)
def test_get_feature_names_out(Estimator):
    """Check get_feature_names_out"""
    est = Estimator().fit(X)
    X_trans = est.transform(X)

    names_out = est.get_feature_names_out()
    class_name = Estimator.__name__.lower()
    expected_names = [f"{class_name}{i}" for i in range(X_trans.shape[1])]
    assert_array_equal(names_out, expected_names)


def test_additivechi2sampler_get_feature_names_out():
    """Check get_feature_names_out for AdditiveChi2Sampler."""
    rng = np.random.RandomState(0)
    X = rng.random_sample(size=(300, 3))

    chi2_sampler = AdditiveChi2Sampler(sample_steps=3).fit(X)
    input_names = ["f0", "f1", "f2"]
    suffixes = [
        "f0_sqrt",
        "f1_sqrt",
        "f2_sqrt",
        "f0_cos1",
        "f1_cos1",
        "f2_cos1",
        "f0_sin1",
        "f1_sin1",
        "f2_sin1",
        "f0_cos2",
        "f1_cos2",
        "f2_cos2",
        "f0_sin2",
        "f1_sin2",
        "f2_sin2",
    ]

    names_out = chi2_sampler.get_feature_names_out(input_features=input_names)
    expected_names = [f"additivechi2sampler_{suffix}" for suffix in suffixes]
    assert_array_equal(names_out, expected_names)
