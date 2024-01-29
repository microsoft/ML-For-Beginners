"""
Test functions for multivariate normal distributions.

"""
import pickle

from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal, assert_equal,
                           assert_array_less, assert_)
import pytest
from pytest import raises as assert_raises

from .test_continuous_basic import check_distribution_rvs

import numpy
import numpy as np

import scipy.linalg
from scipy.stats._multivariate import (_PSD,
                                       _lnB,
                                       multivariate_normal_frozen)
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
                         matrix_normal, special_ortho_group, ortho_group,
                         random_correlation, unitary_group, dirichlet,
                         beta, wishart, multinomial, invwishart, chi2,
                         invgamma, norm, uniform, ks_2samp, kstest, binom,
                         hypergeom, multivariate_t, cauchy, normaltest,
                         random_table, uniform_direction, vonmises_fisher,
                         dirichlet_multinomial, vonmises)

from scipy.stats import _covariance, Covariance
from scipy import stats

from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version

from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv

from unittest.mock import patch


def assert_close(res, ref, *args, **kwargs):
    res, ref = np.asarray(res), np.asarray(ref)
    assert_allclose(res, ref, *args, **kwargs)
    assert_equal(res.shape, ref.shape)


class TestCovariance:

    def test_input_validation(self):

        message = "The input `precision` must be a square, two-dimensional..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.ones(2))

        message = "`precision.shape` must equal `covariance.shape`."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaPrecision(np.eye(3), covariance=np.eye(2))

        message = "The input `diagonal` must be a one-dimensional array..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaDiagonal("alpaca")

        message = "The input `cholesky` must be a square, two-dimensional..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaCholesky(np.ones(2))

        message = "The input `eigenvalues` must be a one-dimensional..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(("alpaca", np.eye(2)))

        message = "The input `eigenvectors` must be a square..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition((np.ones(2), "alpaca"))

        message = "The shapes of `eigenvalues` and `eigenvectors` must be..."
        with pytest.raises(ValueError, match=message):
            _covariance.CovViaEigendecomposition(([1, 2, 3], np.eye(2)))

    _covariance_preprocessing = {"Diagonal": np.diag,
                                 "Precision": np.linalg.inv,
                                 "Cholesky": np.linalg.cholesky,
                                 "Eigendecomposition": np.linalg.eigh,
                                 "PSD": lambda x:
                                     _PSD(x, allow_singular=True)}
    _all_covariance_types = np.array(list(_covariance_preprocessing))
    _matrices = {"diagonal full rank": np.diag([1, 2, 3]),
                 "general full rank": [[5, 1, 3], [1, 6, 4], [3, 4, 7]],
                 "diagonal singular": np.diag([1, 0, 3]),
                 "general singular": [[5, -1, 0], [-1, 5, 0], [0, 0, 0]]}
    _cov_types = {"diagonal full rank": _all_covariance_types,
                  "general full rank": _all_covariance_types[1:],
                  "diagonal singular": _all_covariance_types[[0, -2, -1]],
                  "general singular": _all_covariance_types[-2:]}

    @pytest.mark.parametrize("cov_type_name", _all_covariance_types[:-1])
    def test_factories(self, cov_type_name):
        A = np.diag([1, 2, 3])
        x = [-4, 2, 5]

        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        preprocessing = self._covariance_preprocessing[cov_type_name]
        factory = getattr(Covariance, f"from_{cov_type_name.lower()}")

        res = factory(preprocessing(A))
        ref = cov_type(preprocessing(A))
        assert type(res) == type(ref)
        assert_allclose(res.whiten(x), ref.whiten(x))

    @pytest.mark.parametrize("matrix_type", list(_matrices))
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    def test_covariance(self, matrix_type, cov_type_name):
        message = (f"CovVia{cov_type_name} does not support {matrix_type} "
                   "matrices")
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)

        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        preprocessing = self._covariance_preprocessing[cov_type_name]

        psd = _PSD(A, allow_singular=True)

        # test properties
        cov_object = cov_type(preprocessing(A))
        assert_close(cov_object.log_pdet, psd.log_pdet)
        assert_equal(cov_object.rank, psd.rank)
        assert_equal(cov_object.shape, np.asarray(A).shape)
        assert_close(cov_object.covariance, np.asarray(A))

        # test whitening/coloring 1D x
        rng = np.random.default_rng(5292808890472453840)
        x = rng.random(size=3)
        res = cov_object.whiten(x)
        ref = x @ psd.U
        # res != ref in general; but res @ res == ref @ ref
        assert_close(res @ res, ref @ ref)
        if hasattr(cov_object, "_colorize") and "singular" not in matrix_type:
            # CovViaPSD does not have _colorize
            assert_close(cov_object.colorize(res), x)

        # test whitening/coloring 3D x
        x = rng.random(size=(2, 4, 3))
        res = cov_object.whiten(x)
        ref = x @ psd.U
        assert_close((res**2).sum(axis=-1), (ref**2).sum(axis=-1))
        if hasattr(cov_object, "_colorize") and "singular" not in matrix_type:
            assert_close(cov_object.colorize(res), x)

        # gh-19197 reported that multivariate normal `rvs` produced incorrect
        # results when a singular Covariance object was produce using
        # `from_eigenvalues`. This was due to an issue in `colorize` with
        # singular covariance matrices. Check this edge case, which is skipped
        # in the previous tests.
        if hasattr(cov_object, "_colorize"):
            res = cov_object.colorize(np.eye(len(A)))
            assert_close(res.T @ res, A)

    @pytest.mark.parametrize("size", [None, tuple(), 1, (2, 4, 3)])
    @pytest.mark.parametrize("matrix_type", list(_matrices))
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    def test_mvn_with_covariance(self, size, matrix_type, cov_type_name):
        message = (f"CovVia{cov_type_name} does not support {matrix_type} "
                   "matrices")
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)

        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        preprocessing = self._covariance_preprocessing[cov_type_name]

        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)

        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)
        rng = np.random.default_rng(5292808890472453840)
        x1 = mvn.rvs(mean, cov_object, size=size, random_state=rng)
        rng = np.random.default_rng(5292808890472453840)
        x2 = mvn(mean, cov_object, seed=rng).rvs(size=size)
        if isinstance(cov_object, _covariance.CovViaPSD):
            assert_close(x1, np.squeeze(x))  # for backward compatibility
            assert_close(x2, np.squeeze(x))
        else:
            assert_equal(x1.shape, x.shape)
            assert_equal(x2.shape, x.shape)
            assert_close(x2, x1)

        assert_close(mvn.pdf(x, mean, cov_object), dist0.pdf(x))
        assert_close(dist1.pdf(x), dist0.pdf(x))
        assert_close(mvn.logpdf(x, mean, cov_object), dist0.logpdf(x))
        assert_close(dist1.logpdf(x), dist0.logpdf(x))
        assert_close(mvn.entropy(mean, cov_object), dist0.entropy())
        assert_close(dist1.entropy(), dist0.entropy())

    @pytest.mark.parametrize("size", [tuple(), (2, 4, 3)])
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    def test_mvn_with_covariance_cdf(self, size, cov_type_name):
        # This is split from the test above because it's slow to be running
        # with all matrix types, and there's no need because _mvn.mvnun
        # does the calculation. All Covariance needs to do is pass is
        # provide the `covariance` attribute.
        matrix_type = "diagonal full rank"
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        preprocessing = self._covariance_preprocessing[cov_type_name]

        mean = [0.1, 0.2, 0.3]
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)

        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)

        assert_close(mvn.cdf(x, mean, cov_object), dist0.cdf(x))
        assert_close(dist1.cdf(x), dist0.cdf(x))
        assert_close(mvn.logcdf(x, mean, cov_object), dist0.logcdf(x))
        assert_close(dist1.logcdf(x), dist0.logcdf(x))

    def test_covariance_instantiation(self):
        message = "The `Covariance` class cannot be instantiated directly."
        with pytest.raises(NotImplementedError, match=message):
            Covariance()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # matrix not PSD
    def test_gh9942(self):
        # Originally there was a mistake in the `multivariate_normal_frozen`
        # `rvs` method that caused all covariance objects to be processed as
        # a `_CovViaPSD`. Ensure that this is resolved.
        A = np.diag([1, 2, -1e-8])
        n = A.shape[0]
        mean = np.zeros(n)

        # Error if the matrix is processed as a `_CovViaPSD`
        with pytest.raises(ValueError, match="The input matrix must be..."):
            multivariate_normal(mean, A).rvs()

        # No error if it is provided as a `CovViaEigendecomposition`
        seed = 3562050283508273023
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        cov = Covariance.from_eigendecomposition(np.linalg.eigh(A))
        rv = multivariate_normal(mean, cov)
        res = rv.rvs(random_state=rng1)
        ref = multivariate_normal.rvs(mean, cov, random_state=rng2)
        assert_equal(res, ref)

    def test_gh19197(self):
        # gh-19197 reported that multivariate normal `rvs` produced incorrect
        # results when a singular Covariance object was produce using
        # `from_eigenvalues`. Check that this specific issue is resolved;
        # a more general test is included in `test_covariance`.
        mean = np.ones(2)
        cov = Covariance.from_eigendecomposition((np.zeros(2), np.eye(2)))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert_equal(rvs, mean)

        cov = scipy.stats.Covariance.from_eigendecomposition(
            (np.array([1., 0.]), np.array([[1., 0.], [0., 400.]])))
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        rvs = dist.rvs(size=None)
        assert rvs[0] != mean[0]
        assert rvs[1] == mean[1]


def _random_covariance(dim, evals, rng, singular=False):
    # Generates random covariance matrix with dimensionality `dim` and
    # eigenvalues `evals` using provided Generator `rng`. Randomly sets
    # some evals to zero if `singular` is True.
    A = rng.random((dim, dim))
    A = A @ A.T
    _, v = np.linalg.eigh(A)
    if singular:
        zero_eigs = rng.normal(size=dim) > 0
        evals[zero_eigs] = 0
    cov = v @ np.diag(evals) @ v.T
    return cov


def _sample_orthonormal_matrix(n):
    M = np.random.randn(n, n)
    u, s, v = scipy.linalg.svd(M)
    return u


class TestMultivariateNormal:
    def test_input_shape(self):
        mu = np.arange(3)
        cov = np.identity(2)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1, 2), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1, 2), mu, cov)

    def test_scalar_values(self):
        np.random.seed(1234)

        # When evaluated on scalar data, the pdf should return a scalar
        x, mean, cov = 1.5, 1.7, 2.5
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)

        # When evaluated on a single vector, the pdf should return a scalar
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))  # Diagonal values for cov. matrix
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)

        # When evaluated on scalar data, the cdf should return a scalar
        x, mean, cov = 1.5, 1.7, 2.5
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

        # When evaluated on a single vector, the cdf should return a scalar
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))  # Diagonal values for cov. matrix
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

    def test_logpdf(self):
        # Check that the log of the pdf is in fact the logpdf
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logpdf(x, mean, cov)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logpdf_default_values(self):
        # Check that the log of the pdf is in fact the logpdf
        # with default parameters Mean=None and cov = 1
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logpdf(x)
        d2 = multivariate_normal.pdf(x)
        # check whether default values are being used
        d3 = multivariate_normal.logpdf(x, None, 1)
        d4 = multivariate_normal.pdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_logcdf(self):
        # Check that the log of the cdf is in fact the logcdf
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logcdf(x, mean, cov)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    def test_logcdf_default_values(self):
        # Check that the log of the cdf is in fact the logcdf
        # with default parameters Mean=None and cov = 1
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logcdf(x)
        d2 = multivariate_normal.cdf(x)
        # check whether default values are being used
        d3 = multivariate_normal.logcdf(x, None, 1)
        d4 = multivariate_normal.cdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    def test_rank(self):
        # Check that the rank is detected correctly.
        np.random.seed(1234)
        n = 4
        mean = np.random.randn(n)
        for expected_rank in range(1, n + 1):
            s = np.random.randn(n, expected_rank)
            cov = np.dot(s, s.T)
            distn = multivariate_normal(mean, cov, allow_singular=True)
            assert_equal(distn.cov_object.rank, expected_rank)

    def test_degenerate_distributions(self):

        for n in range(1, 5):
            z = np.random.randn(n)
            for k in range(1, n):
                # Sample a small covariance matrix.
                s = np.random.randn(k, k)
                cov_kk = np.dot(s, s.T)

                # Embed the small covariance matrix into a larger singular one.
                cov_nn = np.zeros((n, n))
                cov_nn[:k, :k] = cov_kk

                # Embed part of the vector in the same way
                x = np.zeros(n)
                x[:k] = z[:k]

                # Define a rotation of the larger low rank matrix.
                u = _sample_orthonormal_matrix(n)
                cov_rr = np.dot(u, np.dot(cov_nn, u.T))
                y = np.dot(u, x)

                # Check some identities.
                distn_kk = multivariate_normal(np.zeros(k), cov_kk,
                                               allow_singular=True)
                distn_nn = multivariate_normal(np.zeros(n), cov_nn,
                                               allow_singular=True)
                distn_rr = multivariate_normal(np.zeros(n), cov_rr,
                                               allow_singular=True)
                assert_equal(distn_kk.cov_object.rank, k)
                assert_equal(distn_nn.cov_object.rank, k)
                assert_equal(distn_rr.cov_object.rank, k)
                pdf_kk = distn_kk.pdf(x[:k])
                pdf_nn = distn_nn.pdf(x)
                pdf_rr = distn_rr.pdf(y)
                assert_allclose(pdf_kk, pdf_nn)
                assert_allclose(pdf_kk, pdf_rr)
                logpdf_kk = distn_kk.logpdf(x[:k])
                logpdf_nn = distn_nn.logpdf(x)
                logpdf_rr = distn_rr.logpdf(y)
                assert_allclose(logpdf_kk, logpdf_nn)
                assert_allclose(logpdf_kk, logpdf_rr)

                # Add an orthogonal component and find the density
                y_orth = y + u[:, -1]
                pdf_rr_orth = distn_rr.pdf(y_orth)
                logpdf_rr_orth = distn_rr.logpdf(y_orth)

                # Ensure that this has zero probability
                assert_equal(pdf_rr_orth, 0.0)
                assert_equal(logpdf_rr_orth, -np.inf)

    def test_degenerate_array(self):
        # Test that we can generate arrays of random variate from a degenerate
        # multivariate normal, and that the pdf for these samples is non-zero
        # (i.e. samples from the distribution lie on the subspace)
        k = 10
        for n in range(2, 6):
            for r in range(1, n):
                mn = np.zeros(n)
                u = _sample_orthonormal_matrix(n)[:, :r]
                vr = np.dot(u, u.T)
                X = multivariate_normal.rvs(mean=mn, cov=vr, size=k)

                pdf = multivariate_normal.pdf(X, mean=mn, cov=vr,
                                              allow_singular=True)
                assert_equal(pdf.size, k)
                assert np.all(pdf > 0.0)

                logpdf = multivariate_normal.logpdf(X, mean=mn, cov=vr,
                                                    allow_singular=True)
                assert_equal(logpdf.size, k)
                assert np.all(logpdf > -np.inf)

    def test_large_pseudo_determinant(self):
        # Check that large pseudo-determinants are handled appropriately.

        # Construct a singular diagonal covariance matrix
        # whose pseudo determinant overflows double precision.
        large_total_log = 1000.0
        npos = 100
        nzero = 2
        large_entry = np.exp(large_total_log / npos)
        n = npos + nzero
        cov = np.zeros((n, n), dtype=float)
        np.fill_diagonal(cov, large_entry)
        cov[-nzero:, -nzero:] = 0

        # Check some determinants.
        assert_equal(scipy.linalg.det(cov), 0)
        assert_equal(scipy.linalg.det(cov[:npos, :npos]), np.inf)
        assert_allclose(np.linalg.slogdet(cov[:npos, :npos]),
                        (1, large_total_log))

        # Check the pseudo-determinant.
        psd = _PSD(cov)
        assert_allclose(psd.log_pdet, large_total_log)

    def test_broadcasting(self):
        np.random.seed(1234)
        n = 4

        # Construct a random covariance matrix.
        data = np.random.randn(n, n)
        cov = np.dot(data, data.T)
        mean = np.random.randn(n)

        # Construct an ndarray which can be interpreted as
        # a 2x3 array whose elements are random data vectors.
        X = np.random.randn(2, 3, n)

        # Check that multiple data points can be evaluated at once.
        desired_pdf = multivariate_normal.pdf(X, mean, cov)
        desired_cdf = multivariate_normal.cdf(X, mean, cov)
        for i in range(2):
            for j in range(3):
                actual = multivariate_normal.pdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_pdf[i,j])
                # Repeat for cdf
                actual = multivariate_normal.cdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_cdf[i,j], rtol=1e-3)

    def test_normal_1D(self):
        # The probability density function for a 1D normal variable should
        # agree with the standard normal distribution in scipy.stats.distributions
        x = np.linspace(0, 2, 10)
        mean, cov = 1.2, 0.9
        scale = cov**0.5
        d1 = norm.pdf(x, mean, scale)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, d2)
        # The same should hold for the cumulative distribution function
        d1 = norm.cdf(x, mean, scale)
        d2 = multivariate_normal.cdf(x, mean, cov)
        assert_allclose(d1, d2)

    def test_marginalization(self):
        # Integrating out one of the variables of a 2D Gaussian should
        # yield a 1D Gaussian
        mean = np.array([2.5, 3.5])
        cov = np.array([[.5, 0.2], [0.2, .6]])
        n = 2 ** 8 + 1  # Number of samples
        delta = 6 / (n - 1)  # Grid spacing

        v = np.linspace(0, 6, n)
        xv, yv = np.meshgrid(v, v)
        pos = np.empty((n, n, 2))
        pos[:, :, 0] = xv
        pos[:, :, 1] = yv
        pdf = multivariate_normal.pdf(pos, mean, cov)

        # Marginalize over x and y axis
        margin_x = romb(pdf, delta, axis=0)
        margin_y = romb(pdf, delta, axis=1)

        # Compare with standard normal distribution
        gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)
        gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)
        assert_allclose(margin_x, gauss_x, rtol=1e-2, atol=1e-2)
        assert_allclose(margin_y, gauss_y, rtol=1e-2, atol=1e-2)

    def test_frozen(self):
        # The frozen distribution should agree with the regular one
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        norm_frozen = multivariate_normal(mean, cov)
        assert_allclose(norm_frozen.pdf(x), multivariate_normal.pdf(x, mean, cov))
        assert_allclose(norm_frozen.logpdf(x),
                        multivariate_normal.logpdf(x, mean, cov))
        assert_allclose(norm_frozen.cdf(x), multivariate_normal.cdf(x, mean, cov))
        assert_allclose(norm_frozen.logcdf(x),
                        multivariate_normal.logcdf(x, mean, cov))

    @pytest.mark.parametrize(
        'covariance',
        [
            np.eye(2),
            Covariance.from_diagonal([1, 1]),
        ]
    )
    def test_frozen_multivariate_normal_exposes_attributes(self, covariance):
        mean = np.ones((2,))
        cov_should_be = np.eye(2)
        norm_frozen = multivariate_normal(mean, covariance)
        assert np.allclose(norm_frozen.mean, mean)
        assert np.allclose(norm_frozen.cov, cov_should_be)

    def test_pseudodet_pinv(self):
        # Make sure that pseudo-inverse and pseudo-det agree on cutoff

        # Assemble random covariance matrix with large and small eigenvalues
        np.random.seed(1234)
        n = 7
        x = np.random.randn(n, n)
        cov = np.dot(x, x.T)
        s, u = scipy.linalg.eigh(cov)
        s = np.full(n, 0.5)
        s[0] = 1.0
        s[-1] = 1e-7
        cov = np.dot(u, np.dot(np.diag(s), u.T))

        # Set cond so that the lowest eigenvalue is below the cutoff
        cond = 1e-5
        psd = _PSD(cov, cond=cond)
        psd_pinv = _PSD(psd.pinv, cond=cond)

        # Check that the log pseudo-determinant agrees with the sum
        # of the logs of all but the smallest eigenvalue
        assert_allclose(psd.log_pdet, np.sum(np.log(s[:-1])))
        # Check that the pseudo-determinant of the pseudo-inverse
        # agrees with 1 / pseudo-determinant
        assert_allclose(-psd.log_pdet, psd_pinv.log_pdet)

    def test_exception_nonsquare_cov(self):
        cov = [[1, 2, 3], [4, 5, 6]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_nonfinite_cov(self):
        cov_nan = [[1, 0], [0, np.nan]]
        assert_raises(ValueError, _PSD, cov_nan)
        cov_inf = [[1, 0], [0, np.inf]]
        assert_raises(ValueError, _PSD, cov_inf)

    def test_exception_non_psd_cov(self):
        cov = [[1, 0], [0, -1]]
        assert_raises(ValueError, _PSD, cov)

    def test_exception_singular_cov(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.ones((5, 5))
        e = np.linalg.LinAlgError
        assert_raises(e, multivariate_normal, mean, cov)
        assert_raises(e, multivariate_normal.pdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logpdf, x, mean, cov)
        assert_raises(e, multivariate_normal.cdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logcdf, x, mean, cov)

        # Message used to be "singular matrix", but this is more accurate.
        # See gh-15508
        cov = [[1., 0.], [1., 1.]]
        msg = "When `allow_singular is False`, the input matrix"
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            multivariate_normal(cov=cov)

    def test_R_values(self):
        # Compare the multivariate pdf with some values precomputed
        # in R version 3.0.1 (2013-05-16) on Mac OS X 10.6.

        # The values below were generated by the following R-script:
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > z <- x + cos(y)
        # > mu <- c(1, 3, 2)
        # > Sigma <- matrix(c(1,2,0,2,5,0.5,0,0.5,3), 3, 3)
        # > r_pdf <- dmnorm(cbind(x,y,z), mu, Sigma)
        r_pdf = np.array([0.0002214706, 0.0013819953, 0.0049138692,
                          0.0103803050, 0.0140250800])

        x = np.linspace(0, 2, 5)
        y = 3 * x - 2
        z = x + np.cos(y)
        r = np.array([x, y, z]).T

        mean = np.array([1, 3, 2], 'd')
        cov = np.array([[1, 2, 0], [2, 5, .5], [0, .5, 3]], 'd')

        pdf = multivariate_normal.pdf(r, mean, cov)
        assert_allclose(pdf, r_pdf, atol=1e-10)

        # Compare the multivariate cdf with some values precomputed
        # in R version 3.3.2 (2016-10-31) on Debian GNU/Linux.

        # The values below were generated by the following R-script:
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > z <- x + cos(y)
        # > mu <- c(1, 3, 2)
        # > Sigma <- matrix(c(1,2,0,2,5,0.5,0,0.5,3), 3, 3)
        # > r_cdf <- pmnorm(cbind(x,y,z), mu, Sigma)
        r_cdf = np.array([0.0017866215, 0.0267142892, 0.0857098761,
                          0.1063242573, 0.2501068509])

        cdf = multivariate_normal.cdf(r, mean, cov)
        assert_allclose(cdf, r_cdf, atol=2e-5)

        # Also test bivariate cdf with some values precomputed
        # in R version 3.3.2 (2016-10-31) on Debian GNU/Linux.

        # The values below were generated by the following R-script:
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > mu <- c(1, 3)
        # > Sigma <- matrix(c(1,2,2,5), 2, 2)
        # > r_cdf2 <- pmnorm(cbind(x,y), mu, Sigma)
        r_cdf2 = np.array([0.01262147, 0.05838989, 0.18389571,
                           0.40696599, 0.66470577])

        r2 = np.array([x, y]).T

        mean2 = np.array([1, 3], 'd')
        cov2 = np.array([[1, 2], [2, 5]], 'd')

        cdf2 = multivariate_normal.cdf(r2, mean2, cov2)
        assert_allclose(cdf2, r_cdf2, atol=1e-5)

    def test_multivariate_normal_rvs_zero_covariance(self):
        mean = np.zeros(2)
        covariance = np.zeros((2, 2))
        model = multivariate_normal(mean, covariance, allow_singular=True)
        sample = model.rvs()
        assert_equal(sample, [0, 0])

    def test_rvs_shape(self):
        # Check that rvs parses the mean and covariance correctly, and returns
        # an array of the right shape
        N = 300
        d = 4
        sample = multivariate_normal.rvs(mean=np.zeros(d), cov=1, size=N)
        assert_equal(sample.shape, (N, d))

        sample = multivariate_normal.rvs(mean=None,
                                         cov=np.array([[2, .1], [.1, 1]]),
                                         size=N)
        assert_equal(sample.shape, (N, 2))

        u = multivariate_normal(mean=0, cov=1)
        sample = u.rvs(N)
        assert_equal(sample.shape, (N, ))

    def test_large_sample(self):
        # Generate large sample and compare sample mean and sample covariance
        # with mean and covariance matrix.

        np.random.seed(2846)

        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        size = 5000

        sample = multivariate_normal.rvs(mean, cov, size)

        assert_allclose(numpy.cov(sample.T), cov, rtol=1e-1)
        assert_allclose(sample.mean(0), mean, rtol=1e-1)

    def test_entropy(self):
        np.random.seed(2846)

        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)

        rv = multivariate_normal(mean, cov)

        # Check that frozen distribution agrees with entropy function
        assert_almost_equal(rv.entropy(), multivariate_normal.entropy(mean, cov))
        # Compare entropy with manually computed expression involving
        # the sum of the logs of the eigenvalues of the covariance matrix
        eigs = np.linalg.eig(cov)[0]
        desired = 1 / 2 * (n * (np.log(2 * np.pi) + 1) + np.sum(np.log(eigs)))
        assert_almost_equal(desired, rv.entropy())

    def test_lnB(self):
        alpha = np.array([1, 1, 1])
        desired = .5  # e^lnB = 1/2 for [1, 1, 1]

        assert_almost_equal(np.exp(_lnB(alpha)), desired)

    def test_cdf_with_lower_limit_arrays(self):
        # test CDF with lower limit in several dimensions
        rng = np.random.default_rng(2408071309372769818)
        mean = [0, 0]
        cov = np.eye(2)
        a = rng.random((4, 3, 2))*6 - 3
        b = rng.random((4, 3, 2))*6 - 3

        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)

        cdf2a = multivariate_normal.cdf(b, mean, cov)
        cdf2b = multivariate_normal.cdf(a, mean, cov)
        ab1 = np.concatenate((a[..., 0:1], b[..., 1:2]), axis=-1)
        ab2 = np.concatenate((a[..., 1:2], b[..., 0:1]), axis=-1)
        cdf2ab1 = multivariate_normal.cdf(ab1, mean, cov)
        cdf2ab2 = multivariate_normal.cdf(ab2, mean, cov)
        cdf2 = cdf2a + cdf2b - cdf2ab1 - cdf2ab2

        assert_allclose(cdf1, cdf2)

    def test_cdf_with_lower_limit_consistency(self):
        # check that multivariate normal CDF functions are consistent
        rng = np.random.default_rng(2408071309372769818)
        mean = rng.random(3)
        cov = rng.random((3, 3))
        cov = cov @ cov.T
        a = rng.random((2, 3))*6 - 3
        b = rng.random((2, 3))*6 - 3

        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        cdf2 = multivariate_normal(mean, cov).cdf(b, lower_limit=a)
        cdf3 = np.exp(multivariate_normal.logcdf(b, mean, cov, lower_limit=a))
        cdf4 = np.exp(multivariate_normal(mean, cov).logcdf(b, lower_limit=a))

        assert_allclose(cdf2, cdf1, rtol=1e-4)
        assert_allclose(cdf3, cdf1, rtol=1e-4)
        assert_allclose(cdf4, cdf1, rtol=1e-4)

    def test_cdf_signs(self):
        # check that sign of output is correct when np.any(lower > x)
        mean = np.zeros(3)
        cov = np.eye(3)
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        # when odd number of elements of b < a, output is negative
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        assert_allclose(cdf, cdf[0]*expected_signs)

    def test_mean_cov(self):
        # test the interaction between a Covariance object and mean
        P = np.diag(1 / np.array([1, 2, 3]))
        cov_object = _covariance.CovViaPrecision(P)

        message = "`cov` represents a covariance matrix in 3 dimensions..."
        with pytest.raises(ValueError, match=message):
            multivariate_normal.entropy([0, 0], cov_object)

        with pytest.raises(ValueError, match=message):
            multivariate_normal([0, 0], cov_object)

        x = [0.5, 0.5, 0.5]
        ref = multivariate_normal.pdf(x, [0, 0, 0], cov_object)
        assert_equal(multivariate_normal.pdf(x, cov=cov_object), ref)

        ref = multivariate_normal.pdf(x, [1, 1, 1], cov_object)
        assert_equal(multivariate_normal.pdf(x, 1, cov=cov_object), ref)

    def test_fit_wrong_fit_data_shape(self):
        data = [1, 3]
        error_msg = "`x` must be two-dimensional."
        with pytest.raises(ValueError, match=error_msg):
            multivariate_normal.fit(data)

    @pytest.mark.parametrize('dim', (3, 5))
    def test_fit_correctness(self, dim):
        rng = np.random.default_rng(4385269356937404)
        x = rng.random((100, dim))
        mean_est, cov_est = multivariate_normal.fit(x)
        mean_ref, cov_ref = np.mean(x, axis=0), np.cov(x.T, ddof=0)
        assert_allclose(mean_est, mean_ref, atol=1e-15)
        assert_allclose(cov_est, cov_ref, rtol=1e-15)

    def test_fit_both_parameters_fixed(self):
        data = np.full((2, 1), 3)
        mean_fixed = 1.
        cov_fixed = np.atleast_2d(1.)
        mean, cov = multivariate_normal.fit(data, fix_mean=mean_fixed,
                                            fix_cov=cov_fixed)
        assert_equal(mean, mean_fixed)
        assert_equal(cov, cov_fixed)

    @pytest.mark.parametrize('fix_mean', [np.zeros((2, 2)),
                                          np.zeros((3, ))])
    def test_fit_fix_mean_input_validation(self, fix_mean):
        msg = ("`fix_mean` must be a one-dimensional array the same "
                "length as the dimensionality of the vectors `x`.")
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(2), fix_mean=fix_mean)

    @pytest.mark.parametrize('fix_cov', [np.zeros((2, )),
                                         np.zeros((3, 2)),
                                         np.zeros((4, 4))])
    def test_fit_fix_cov_input_validation_dimension(self, fix_cov):
        msg = ("`fix_cov` must be a two-dimensional square array "
                "of same side length as the dimensionality of the "
                "vectors `x`.")
        with pytest.raises(ValueError, match=msg):
            multivariate_normal.fit(np.eye(3), fix_cov=fix_cov)
    
    def test_fit_fix_cov_not_positive_semidefinite(self):
        error_msg = "`fix_cov` must be symmetric positive semidefinite."
        with pytest.raises(ValueError, match=error_msg):
            fix_cov = np.array([[1., 0.], [0., -1.]])
            multivariate_normal.fit(np.eye(2), fix_cov=fix_cov)
    
    def test_fit_fix_mean(self):
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100,
                                          random_state=rng)
        mean_free, cov_free = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free,
                                               cov=cov_free).sum()
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_mean=loc)
        assert_equal(mean_fix, loc)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix,
                                              cov=cov_fix).sum()
        # test that fixed parameters result in lower likelihood than free
        # parameters
        assert logp_fix < logp_free
        # test that a small perturbation of the resulting parameters
        # has lower likelihood than the estimated parameters
        A = rng.random((3, 3))
        m = 1e-8 * np.dot(A, A.T)
        cov_perturbed = cov_fix + m
        logp_perturbed = (multivariate_normal.logpdf(samples,
                                                     mean=mean_fix,
                                                     cov=cov_perturbed)
                                                     ).sum()
        assert logp_perturbed < logp_fix


    def test_fit_fix_cov(self):
        rng = np.random.default_rng(4385269356937404)
        loc = rng.random(3)
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        samples = multivariate_normal.rvs(mean=loc, cov=cov,
                                          size=100, random_state=rng)
        mean_free, cov_free = multivariate_normal.fit(samples)
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free,
                                               cov=cov_free).sum()
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_cov=cov)
        assert_equal(mean_fix, np.mean(samples, axis=0))
        assert_equal(cov_fix, cov)
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix,
                                              cov=cov_fix).sum()
        # test that fixed parameters result in lower likelihood than free
        # parameters
        assert logp_fix < logp_free
        # test that a small perturbation of the resulting parameters
        # has lower likelihood than the estimated parameters
        mean_perturbed = mean_fix + 1e-8 * rng.random(3)
        logp_perturbed = (multivariate_normal.logpdf(samples,
                                                     mean=mean_perturbed,
                                                     cov=cov_fix)
                                                     ).sum()
        assert logp_perturbed < logp_fix


class TestMatrixNormal:

    def test_bad_input(self):
        # Check that bad inputs raise errors
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)

        # Incorrect dimensions
        assert_raises(ValueError, matrix_normal, np.zeros((5,4,3)))
        assert_raises(ValueError, matrix_normal, M, np.zeros(10), V)
        assert_raises(ValueError, matrix_normal, M, U, np.zeros(10))
        assert_raises(ValueError, matrix_normal, M, U, U)
        assert_raises(ValueError, matrix_normal, M, V, V)
        assert_raises(ValueError, matrix_normal, M.T, U, V)

        e = np.linalg.LinAlgError
        # Singular covariance for the rvs method of a non-frozen instance
        assert_raises(e, matrix_normal.rvs,
                      M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal.rvs,
                      M, np.ones((num_rows, num_rows)), V)
        # Singular covariance for a frozen instance
        assert_raises(e, matrix_normal, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal, M, np.ones((num_rows, num_rows)), V)

    def test_default_inputs(self):
        # Check that default argument handling works
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        Z = np.zeros((num_rows, num_cols))
        Zr = np.zeros((num_rows, 1))
        Zc = np.zeros((1, num_cols))
        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)
        I1 = np.identity(1)

        assert_equal(matrix_normal.rvs(mean=M, rowcov=U, colcov=V).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U).shape,
                     (num_rows, 1))
        assert_equal(matrix_normal.rvs(colcov=V).shape,
                     (1, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, colcov=V).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U, colcov=V).shape,
                     (num_rows, num_cols))

        assert_equal(matrix_normal(mean=M).rowcov, Ir)
        assert_equal(matrix_normal(mean=M).colcov, Ic)
        assert_equal(matrix_normal(rowcov=U).mean, Zr)
        assert_equal(matrix_normal(rowcov=U).colcov, I1)
        assert_equal(matrix_normal(colcov=V).mean, Zc)
        assert_equal(matrix_normal(colcov=V).rowcov, I1)
        assert_equal(matrix_normal(mean=M, rowcov=U).colcov, Ic)
        assert_equal(matrix_normal(mean=M, colcov=V).rowcov, Ir)
        assert_equal(matrix_normal(rowcov=U, colcov=V).mean, Z)

    def test_covariance_expansion(self):
        # Check that covariance can be specified with scalar or vector
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows, num_cols), 0.3)
        Uv = np.full(num_rows, 0.2)
        Us = 0.2
        Vv = np.full(num_cols, 0.1)
        Vs = 0.1

        Ir = np.identity(num_rows)
        Ic = np.identity(num_cols)

        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).rowcov,
                     0.2*Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).colcov,
                     0.1*Ic)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).rowcov,
                     0.2*Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).colcov,
                     0.1*Ic)

    def test_frozen_matrix_normal(self):
        for i in range(1,5):
            for j in range(1,5):
                M = np.full((i,j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i,i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j,j), 0.3)

                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)

                rvs1 = frozen.rvs(random_state=1234)
                rvs2 = matrix_normal.rvs(mean=M, rowcov=U, colcov=V,
                                         random_state=1234)
                assert_equal(rvs1, rvs2)

                X = frozen.rvs(random_state=1234)

                pdf1 = frozen.pdf(X)
                pdf2 = matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(pdf1, pdf2)

                logpdf1 = frozen.logpdf(X)
                logpdf2 = matrix_normal.logpdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(logpdf1, logpdf2)

    def test_matches_multivariate(self):
        # Check that the pdfs match those obtained by vectorising and
        # treating as a multivariate normal.
        for i in range(1,5):
            for j in range(1,5):
                M = np.full((i,j), 0.3)
                U = 0.5 * np.identity(i) + np.full((i,i), 0.5)
                V = 0.7 * np.identity(j) + np.full((j,j), 0.3)

                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                X = frozen.rvs(random_state=1234)
                pdf1 = frozen.pdf(X)
                logpdf1 = frozen.logpdf(X)
                entropy1 = frozen.entropy()

                vecX = X.T.flatten()
                vecM = M.T.flatten()
                cov = np.kron(V,U)
                pdf2 = multivariate_normal.pdf(vecX, mean=vecM, cov=cov)
                logpdf2 = multivariate_normal.logpdf(vecX, mean=vecM, cov=cov)
                entropy2 = multivariate_normal.entropy(mean=vecM, cov=cov)

                assert_allclose(pdf1, pdf2, rtol=1E-10)
                assert_allclose(logpdf1, logpdf2, rtol=1E-10)
                assert_allclose(entropy1, entropy2)

    def test_array_input(self):
        # Check array of inputs has the same output as the separate entries.
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 10

        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X1 = frozen.rvs(size=N, random_state=1234)
        X2 = frozen.rvs(size=N, random_state=4321)
        X = np.concatenate((X1[np.newaxis,:,:,:],X2[np.newaxis,:,:,:]), axis=0)
        assert_equal(X.shape, (2, N, num_rows, num_cols))

        array_logpdf = frozen.logpdf(X)
        assert_equal(array_logpdf.shape, (2, N))
        for i in range(2):
            for j in range(N):
                separate_logpdf = matrix_normal.logpdf(X[i,j], mean=M,
                                                       rowcov=U, colcov=V)
                assert_allclose(separate_logpdf, array_logpdf[i,j], 1E-10)

    def test_moments(self):
        # Check that the sample moments match the parameters
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)
        N = 1000

        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
        X = frozen.rvs(size=N, random_state=1234)

        sample_mean = np.mean(X,axis=0)
        assert_allclose(sample_mean, M, atol=0.1)

        sample_colcov = np.cov(X.reshape(N*num_rows,num_cols).T)
        assert_allclose(sample_colcov, V, atol=0.1)

        sample_rowcov = np.cov(np.swapaxes(X,1,2).reshape(
                                                        N*num_cols,num_rows).T)
        assert_allclose(sample_rowcov, U, atol=0.1)

    def test_samples(self):
        # Regression test to ensure that we always generate the same stream of
        # random variates.
        actual = matrix_normal.rvs(
            mean=np.array([[1, 2], [3, 4]]),
            rowcov=np.array([[4, -1], [-1, 2]]),
            colcov=np.array([[5, 1], [1, 10]]),
            random_state=np.random.default_rng(0),
            size=2
        )
        expected = np.array(
            [[[1.56228264238181, -1.24136424071189],
              [2.46865788392114, 6.22964440489445]],
             [[3.86405716144353, 10.73714311429529],
              [2.59428444080606, 5.79987854490876]]]
        )
        assert_allclose(actual, expected)


class TestDirichlet:

    def test_frozen_dirichlet(self):
        np.random.seed(2846)

        n = np.random.randint(1, 32)
        alpha = np.random.uniform(10e-10, 100, n)

        d = dirichlet(alpha)

        assert_equal(d.var(), dirichlet.var(alpha))
        assert_equal(d.mean(), dirichlet.mean(alpha))
        assert_equal(d.entropy(), dirichlet.entropy(alpha))
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(10e-10, 100, n)
            x /= np.sum(x)
            assert_equal(d.pdf(x[:-1]), dirichlet.pdf(x[:-1], alpha))
            assert_equal(d.logpdf(x[:-1]), dirichlet.logpdf(x[:-1], alpha))

    def test_numpy_rvs_shape_compatibility(self):
        np.random.seed(2846)
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.random.dirichlet(alpha, size=7)
        assert_equal(x.shape, (7, 3))
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)
        dirichlet.pdf(x.T, alpha)
        dirichlet.pdf(x.T[:-1], alpha)
        dirichlet.logpdf(x.T, alpha)
        dirichlet.logpdf(x.T[:-1], alpha)

    def test_alpha_with_zeros(self):
        np.random.seed(2846)
        alpha = [1.0, 0.0, 3.0]
        # don't pass invalid alpha to np.random.dirichlet
        x = np.random.dirichlet(np.maximum(1e-9, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_with_negative_entries(self):
        np.random.seed(2846)
        alpha = [1.0, -2.0, 3.0]
        # don't pass invalid alpha to np.random.dirichlet
        x = np.random.dirichlet(np.maximum(1e-9, alpha), size=7).T
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_zeros(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)
        alpha = np.array([1.0, 1.0, 1.0, 1.0])
        assert_almost_equal(dirichlet.pdf(x, alpha), 6)
        assert_almost_equal(dirichlet.logpdf(x, alpha), np.log(6))

    def test_data_with_zeros_and_small_alpha(self):
        alpha = np.array([1.0, 0.5, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_negative_entries(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, -0.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_too_large_entries(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 1.1, 0.3, 0.7])
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_too_deep_c(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((2, 7, 7), 1 / 14)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_too_deep(self):
        alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.full((2, 2, 7), 1 / 4)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_correct_depth(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 3)
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)

    def test_non_simplex_data(self):
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.full((3, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_short(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((2, 7), 1 / 2)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_vector_too_long(self):
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.full((5, 7), 1 / 5)
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_mean_var_cov(self):
        # Reference values calculated by hand and confirmed with Mathematica, e.g.
        # `Covariance[DirichletDistribution[{ 1, 0.8, 0.2, 10^-300}]]`
        alpha = np.array([1., 0.8, 0.2])
        d = dirichlet(alpha)

        expected_mean = [0.5, 0.4, 0.1]
        expected_var = [1. / 12., 0.08, 0.03]
        expected_cov = [
                [ 1. / 12, -1. / 15, -1. / 60],
                [-1. / 15,  2. / 25, -1. / 75],
                [-1. / 60, -1. / 75,  3. / 100],
        ]

        assert_array_almost_equal(d.mean(), expected_mean)
        assert_array_almost_equal(d.var(), expected_var)
        assert_array_almost_equal(d.cov(), expected_cov)

    def test_scalar_values(self):
        alpha = np.array([0.2])
        d = dirichlet(alpha)

        # For alpha of length 1, mean and var should be scalar instead of array
        assert_equal(d.mean().ndim, 0)
        assert_equal(d.var().ndim, 0)

        assert_equal(d.pdf([1.]).ndim, 0)
        assert_equal(d.logpdf([1.]).ndim, 0)

    def test_K_and_K_minus_1_calls_equal(self):
        # Test that calls with K and K-1 entries yield the same results.

        np.random.seed(2846)

        n = np.random.randint(1, 32)
        alpha = np.random.uniform(10e-10, 100, n)

        d = dirichlet(alpha)
        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(10e-10, 100, n)
            x /= np.sum(x)
            assert_almost_equal(d.pdf(x[:-1]), d.pdf(x))

    def test_multiple_entry_calls(self):
        # Test that calls with multiple x vectors as matrix work
        np.random.seed(2846)

        n = np.random.randint(1, 32)
        alpha = np.random.uniform(10e-10, 100, n)
        d = dirichlet(alpha)

        num_tests = 10
        num_multiple = 5
        xm = None
        for i in range(num_tests):
            for m in range(num_multiple):
                x = np.random.uniform(10e-10, 100, n)
                x /= np.sum(x)
                if xm is not None:
                    xm = np.vstack((xm, x))
                else:
                    xm = x
            rm = d.pdf(xm.T)
            rs = None
            for xs in xm:
                r = d.pdf(xs)
                if rs is not None:
                    rs = np.append(rs, r)
                else:
                    rs = r
            assert_array_almost_equal(rm, rs)

    def test_2D_dirichlet_is_beta(self):
        np.random.seed(2846)

        alpha = np.random.uniform(10e-10, 100, 2)
        d = dirichlet(alpha)
        b = beta(alpha[0], alpha[1])

        num_tests = 10
        for i in range(num_tests):
            x = np.random.uniform(10e-10, 100, 2)
            x /= np.sum(x)
            assert_almost_equal(b.pdf(x), d.pdf([x]))

        assert_almost_equal(b.mean(), d.mean()[0])
        assert_almost_equal(b.var(), d.var()[0])


def test_multivariate_normal_dimensions_mismatch():
    # Regression test for GH #3493. Check that setting up a PDF with a mean of
    # length M and a covariance matrix of size (N, N), where M != N, raises a
    # ValueError with an informative error message.
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0]])

    assert_raises(ValueError, multivariate_normal, mu, sigma)

    # A simple check that the right error message was passed along. Checking
    # that the entire message is there, word for word, would be somewhat
    # fragile, so we just check for the leading part.
    try:
        multivariate_normal(mu, sigma)
    except ValueError as e:
        msg = "Dimension mismatch"
        assert_equal(str(e)[:len(msg)], msg)


class TestWishart:
    def test_scale_dimensions(self):
        # Test that we can call the Wishart with various scale dimensions

        # Test case: dim=1, scale=1
        true_scale = np.array(1, ndmin=2)
        scales = [
            1,                    # scalar
            [1],                  # iterable
            np.array(1),          # 0-dim
            np.r_[1],             # 1-dim
            np.array(1, ndmin=2)  # 2-dim
        ]
        for scale in scales:
            w = wishart(1, scale)
            assert_equal(w.scale, true_scale)
            assert_equal(w.scale.shape, true_scale.shape)

        # Test case: dim=2, scale=[[1,0]
        #                          [0,2]
        true_scale = np.array([[1,0],
                               [0,2]])
        scales = [
            [1,2],             # iterable
            np.r_[1,2],        # 1-dim
            np.array([[1,0],   # 2-dim
                      [0,2]])
        ]
        for scale in scales:
            w = wishart(2, scale)
            assert_equal(w.scale, true_scale)
            assert_equal(w.scale.shape, true_scale.shape)

        # We cannot call with a df < dim - 1
        assert_raises(ValueError, wishart, 1, np.eye(2))

        # But we can call with dim - 1 < df < dim
        wishart(1.1, np.eye(2))  # no error
        # see gh-5562

        # We cannot call with a 3-dimension array
        scale = np.array(1, ndmin=3)
        assert_raises(ValueError, wishart, 1, scale)

    def test_quantile_dimensions(self):
        # Test that we can call the Wishart rvs with various quantile dimensions

        # If dim == 1, consider x.shape = [1,1,1]
        X = [
            1,                      # scalar
            [1],                    # iterable
            np.array(1),            # 0-dim
            np.r_[1],               # 1-dim
            np.array(1, ndmin=2),   # 2-dim
            np.array([1], ndmin=3)  # 3-dim
        ]

        w = wishart(1,1)
        density = w.pdf(np.array(1, ndmin=3))
        for x in X:
            assert_equal(w.pdf(x), density)

        # If dim == 1, consider x.shape = [1,1,*]
        X = [
            [1,2,3],                     # iterable
            np.r_[1,2,3],                # 1-dim
            np.array([1,2,3], ndmin=3)   # 3-dim
        ]

        w = wishart(1,1)
        density = w.pdf(np.array([1,2,3], ndmin=3))
        for x in X:
            assert_equal(w.pdf(x), density)

        # If dim == 2, consider x.shape = [2,2,1]
        # where x[:,:,*] = np.eye(1)*2
        X = [
            2,                    # scalar
            [2,2],                # iterable
            np.array(2),          # 0-dim
            np.r_[2,2],           # 1-dim
            np.array([[2,0],
                      [0,2]]),    # 2-dim
            np.array([[2,0],
                      [0,2]])[:,:,np.newaxis]  # 3-dim
        ]

        w = wishart(2,np.eye(2))
        density = w.pdf(np.array([[2,0],
                                  [0,2]])[:,:,np.newaxis])
        for x in X:
            assert_equal(w.pdf(x), density)

    def test_frozen(self):
        # Test that the frozen and non-frozen Wishart gives the same answers

        # Construct an arbitrary positive definite scale matrix
        dim = 4
        scale = np.diag(np.arange(dim)+1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim-1) // 2)
        scale = np.dot(scale.T, scale)

        # Construct a collection of positive definite matrices to test the PDF
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim)+(i+1)**2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim-1) // 2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T

        # Construct a 1D and 2D set of parameters
        parameters = [
            (10, 1, np.linspace(0.1, 10, 5)),  # 1D case
            (10, scale, X)
        ]

        for (df, scale, x) in parameters:
            w = wishart(df, scale)
            assert_equal(w.var(), wishart.var(df, scale))
            assert_equal(w.mean(), wishart.mean(df, scale))
            assert_equal(w.mode(), wishart.mode(df, scale))
            assert_equal(w.entropy(), wishart.entropy(df, scale))
            assert_equal(w.pdf(x), wishart.pdf(x, df, scale))

    def test_wishart_2D_rvs(self):
        dim = 3
        df = 10

        # Construct a simple non-diagonal positive definite matrix
        scale = np.eye(dim)
        scale[0,1] = 0.5
        scale[1,0] = 0.5

        # Construct frozen Wishart random variables
        w = wishart(df, scale)

        # Get the generated random variables from a known seed
        np.random.seed(248042)
        w_rvs = wishart.rvs(df, scale)
        np.random.seed(248042)
        frozen_w_rvs = w.rvs()

        # Manually calculate what it should be, based on the Bartlett (1933)
        # decomposition of a Wishart into D A A' D', where D is the Cholesky
        # factorization of the scale matrix and A is the lower triangular matrix
        # with the square root of chi^2 variates on the diagonal and N(0,1)
        # variates in the lower triangle.
        np.random.seed(248042)
        covariances = np.random.normal(size=3)
        variances = np.r_[
            np.random.chisquare(df),
            np.random.chisquare(df-1),
            np.random.chisquare(df-2),
        ]**0.5

        # Construct the lower-triangular A matrix
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances

        # Wishart random variate
        D = np.linalg.cholesky(scale)
        DA = D.dot(A)
        manual_w_rvs = np.dot(DA, DA.T)

        # Test for equality
        assert_allclose(w_rvs, manual_w_rvs)
        assert_allclose(frozen_w_rvs, manual_w_rvs)

    def test_1D_is_chisquared(self):
        # The 1-dimensional Wishart with an identity scale matrix is just a
        # chi-squared distribution.
        # Test variance, mean, entropy, pdf
        # Kolgomorov-Smirnov test for rvs
        np.random.seed(482974)

        sn = 500
        dim = 1
        scale = np.eye(dim)

        df_range = np.arange(1, 10, 2, dtype=float)
        X = np.linspace(0.1,10,num=10)
        for df in df_range:
            w = wishart(df, scale)
            c = chi2(df)

            # Statistics
            assert_allclose(w.var(), c.var())
            assert_allclose(w.mean(), c.mean())
            assert_allclose(w.entropy(), c.entropy())

            # PDF
            assert_allclose(w.pdf(X), c.pdf(X))

            # rvs
            rvs = w.rvs(size=sn)
            args = (df,)
            alpha = 0.01
            check_distribution_rvs('chi2', args, alpha, rvs)

    def test_is_scaled_chisquared(self):
        # The 2-dimensional Wishart with an arbitrary scale matrix can be
        # transformed to a scaled chi-squared distribution.
        # For :math:`S \sim W_p(V,n)` and :math:`\lambda \in \mathbb{R}^p` we have
        # :math:`\lambda' S \lambda \sim \lambda' V \lambda \times \chi^2(n)`
        np.random.seed(482974)

        sn = 500
        df = 10
        dim = 4
        # Construct an arbitrary positive definite matrix
        scale = np.diag(np.arange(4)+1)
        scale[np.tril_indices(4, k=-1)] = np.arange(6)
        scale = np.dot(scale.T, scale)
        # Use :math:`\lambda = [1, \dots, 1]'`
        lamda = np.ones((dim,1))
        sigma_lamda = lamda.T.dot(scale).dot(lamda).squeeze()
        w = wishart(df, sigma_lamda)
        c = chi2(df, scale=sigma_lamda)

        # Statistics
        assert_allclose(w.var(), c.var())
        assert_allclose(w.mean(), c.mean())
        assert_allclose(w.entropy(), c.entropy())

        # PDF
        X = np.linspace(0.1,10,num=10)
        assert_allclose(w.pdf(X), c.pdf(X))

        # rvs
        rvs = w.rvs(size=sn)
        args = (df,0,sigma_lamda)
        alpha = 0.01
        check_distribution_rvs('chi2', args, alpha, rvs)

class TestMultinomial:
    def test_logpmf(self):
        vals1 = multinomial.logpmf((3,4), 7, (0.3, 0.7))
        assert_allclose(vals1, -1.483270127243324, rtol=1e-8)

        vals2 = multinomial.logpmf([3, 4], 0, [.3, .7])
        assert vals2 == -np.inf

        vals3 = multinomial.logpmf([0, 0], 0, [.3, .7])
        assert vals3 == 0

        vals4 = multinomial.logpmf([3, 4], 0, [-2, 3])
        assert_allclose(vals4, np.nan, rtol=1e-8)

    def test_reduces_binomial(self):
        # test that the multinomial pmf reduces to the binomial pmf in the 2d
        # case
        val1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        val2 = binom.logpmf(3, 7, 0.3)
        assert_allclose(val1, val2, rtol=1e-8)

        val1 = multinomial.pmf((6, 8), 14, (0.1, 0.9))
        val2 = binom.pmf(6, 14, 0.1)
        assert_allclose(val1, val2, rtol=1e-8)

    def test_R(self):
        # test against the values produced by this R code
        # (https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Multinom.html)
        # X <- t(as.matrix(expand.grid(0:3, 0:3))); X <- X[, colSums(X) <= 3]
        # X <- rbind(X, 3:3 - colSums(X)); dimnames(X) <- list(letters[1:3], NULL)
        # X
        # apply(X, 2, function(x) dmultinom(x, prob = c(1,2,5)))

        n, p = 3, [1./8, 2./8, 5./8]
        r_vals = {(0, 0, 3): 0.244140625, (1, 0, 2): 0.146484375,
                  (2, 0, 1): 0.029296875, (3, 0, 0): 0.001953125,
                  (0, 1, 2): 0.292968750, (1, 1, 1): 0.117187500,
                  (2, 1, 0): 0.011718750, (0, 2, 1): 0.117187500,
                  (1, 2, 0): 0.023437500, (0, 3, 0): 0.015625000}
        for x in r_vals:
            assert_allclose(multinomial.pmf(x, n, p), r_vals[x], atol=1e-14)

    @pytest.mark.parametrize("n", [0, 3])
    def test_rvs_np(self, n):
        # test that .rvs agrees w/numpy
        sc_rvs = multinomial.rvs(n, [1/4.]*3, size=7, random_state=123)
        rndm = np.random.RandomState(123)
        np_rvs = rndm.multinomial(n, [1/4.]*3, size=7)
        assert_equal(sc_rvs, np_rvs)

    def test_pmf(self):
        vals0 = multinomial.pmf((5,), 5, (1,))
        assert_allclose(vals0, 1, rtol=1e-8)

        vals1 = multinomial.pmf((3,4), 7, (.3, .7))
        assert_allclose(vals1, .22689449999999994, rtol=1e-8)

        vals2 = multinomial.pmf([[[3,5],[0,8]], [[-1, 9], [1, 1]]], 8,
                                (.1, .9))
        assert_allclose(vals2, [[.03306744, .43046721], [0, 0]], rtol=1e-8)

        x = np.empty((0,2), dtype=np.float64)
        vals3 = multinomial.pmf(x, 4, (.3, .7))
        assert_equal(vals3, np.empty([], dtype=np.float64))

        vals4 = multinomial.pmf([1,2], 4, (.3, .7))
        assert_allclose(vals4, 0, rtol=1e-8)

        vals5 = multinomial.pmf([3, 3, 0], 6, [2/3.0, 1/3.0, 0])
        assert_allclose(vals5, 0.219478737997, rtol=1e-8)

        vals5 = multinomial.pmf([0, 0, 0], 0, [2/3.0, 1/3.0, 0])
        assert vals5 == 1

        vals6 = multinomial.pmf([2, 1, 0], 0, [2/3.0, 1/3.0, 0])
        assert vals6 == 0

    def test_pmf_broadcasting(self):
        vals0 = multinomial.pmf([1, 2], 3, [[.1, .9], [.2, .8]])
        assert_allclose(vals0, [.243, .384], rtol=1e-8)

        vals1 = multinomial.pmf([1, 2], [3, 4], [.1, .9])
        assert_allclose(vals1, [.243, 0], rtol=1e-8)

        vals2 = multinomial.pmf([[[1, 2], [1, 1]]], 3, [.1, .9])
        assert_allclose(vals2, [[.243, 0]], rtol=1e-8)

        vals3 = multinomial.pmf([1, 2], [[[3], [4]]], [.1, .9])
        assert_allclose(vals3, [[[.243], [0]]], rtol=1e-8)

        vals4 = multinomial.pmf([[1, 2], [1,1]], [[[[3]]]], [.1, .9])
        assert_allclose(vals4, [[[[.243, 0]]]], rtol=1e-8)

    @pytest.mark.parametrize("n", [0, 5])
    def test_cov(self, n):
        cov1 = multinomial.cov(n, (.2, .3, .5))
        cov2 = [[n*.2*.8, -n*.2*.3, -n*.2*.5],
                [-n*.3*.2, n*.3*.7, -n*.3*.5],
                [-n*.5*.2, -n*.5*.3, n*.5*.5]]
        assert_allclose(cov1, cov2, rtol=1e-8)

    def test_cov_broadcasting(self):
        cov1 = multinomial.cov(5, [[.1, .9], [.2, .8]])
        cov2 = [[[.45, -.45],[-.45, .45]], [[.8, -.8], [-.8, .8]]]
        assert_allclose(cov1, cov2, rtol=1e-8)

        cov3 = multinomial.cov([4, 5], [.1, .9])
        cov4 = [[[.36, -.36], [-.36, .36]], [[.45, -.45], [-.45, .45]]]
        assert_allclose(cov3, cov4, rtol=1e-8)

        cov5 = multinomial.cov([4, 5], [[.3, .7], [.4, .6]])
        cov6 = [[[4*.3*.7, -4*.3*.7], [-4*.3*.7, 4*.3*.7]],
                [[5*.4*.6, -5*.4*.6], [-5*.4*.6, 5*.4*.6]]]
        assert_allclose(cov5, cov6, rtol=1e-8)

    @pytest.mark.parametrize("n", [0, 2])
    def test_entropy(self, n):
        # this is equivalent to a binomial distribution with n=2, so the
        # entropy .77899774929 is easily computed "by hand"
        ent0 = multinomial.entropy(n, [.2, .8])
        assert_allclose(ent0, binom.entropy(n, .2), rtol=1e-8)

    def test_entropy_broadcasting(self):
        ent0 = multinomial.entropy([2, 3], [.2, .3])
        assert_allclose(ent0, [binom.entropy(2, .2), binom.entropy(3, .2)],
                        rtol=1e-8)

        ent1 = multinomial.entropy([7, 8], [[.3, .7], [.4, .6]])
        assert_allclose(ent1, [binom.entropy(7, .3), binom.entropy(8, .4)],
                        rtol=1e-8)

        ent2 = multinomial.entropy([[7], [8]], [[.3, .7], [.4, .6]])
        assert_allclose(ent2,
                        [[binom.entropy(7, .3), binom.entropy(7, .4)],
                         [binom.entropy(8, .3), binom.entropy(8, .4)]],
                        rtol=1e-8)

    @pytest.mark.parametrize("n", [0, 5])
    def test_mean(self, n):
        mean1 = multinomial.mean(n, [.2, .8])
        assert_allclose(mean1, [n*.2, n*.8], rtol=1e-8)

    def test_mean_broadcasting(self):
        mean1 = multinomial.mean([5, 6], [.2, .8])
        assert_allclose(mean1, [[5*.2, 5*.8], [6*.2, 6*.8]], rtol=1e-8)

    def test_frozen(self):
        # The frozen distribution should agree with the regular one
        np.random.seed(1234)
        n = 12
        pvals = (.1, .2, .3, .4)
        x = [[0,0,0,12],[0,0,1,11],[0,1,1,10],[1,1,1,9],[1,1,2,8]]
        x = np.asarray(x, dtype=np.float64)
        mn_frozen = multinomial(n, pvals)
        assert_allclose(mn_frozen.pmf(x), multinomial.pmf(x, n, pvals))
        assert_allclose(mn_frozen.logpmf(x), multinomial.logpmf(x, n, pvals))
        assert_allclose(mn_frozen.entropy(), multinomial.entropy(n, pvals))

    def test_gh_11860(self):
        # gh-11860 reported cases in which the adjustments made by multinomial
        # to the last element of `p` can cause `nan`s even when the input is
        # essentially valid. Check that a pathological case returns a finite,
        # nonzero result. (This would fail in main before the PR.)
        n = 88
        rng = np.random.default_rng(8879715917488330089)
        p = rng.random(n)
        p[-1] = 1e-30
        p /= np.sum(p)
        x = np.ones(n)
        logpmf = multinomial.logpmf(x, n, p)
        assert np.isfinite(logpmf)

class TestInvwishart:
    def test_frozen(self):
        # Test that the frozen and non-frozen inverse Wishart gives the same
        # answers

        # Construct an arbitrary positive definite scale matrix
        dim = 4
        scale = np.diag(np.arange(dim)+1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim*(dim-1)/2)
        scale = np.dot(scale.T, scale)

        # Construct a collection of positive definite matrices to test the PDF
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim)+(i+1)**2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim*(dim-1)/2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T

        # Construct a 1D and 2D set of parameters
        parameters = [
            (10, 1, np.linspace(0.1, 10, 5)),  # 1D case
            (10, scale, X)
        ]

        for (df, scale, x) in parameters:
            iw = invwishart(df, scale)
            assert_equal(iw.var(), invwishart.var(df, scale))
            assert_equal(iw.mean(), invwishart.mean(df, scale))
            assert_equal(iw.mode(), invwishart.mode(df, scale))
            assert_allclose(iw.pdf(x), invwishart.pdf(x, df, scale))

    def test_1D_is_invgamma(self):
        # The 1-dimensional inverse Wishart with an identity scale matrix is
        # just an inverse gamma distribution.
        # Test variance, mean, pdf, entropy
        # Kolgomorov-Smirnov test for rvs
        np.random.seed(482974)

        sn = 500
        dim = 1
        scale = np.eye(dim)

        df_range = np.arange(5, 20, 2, dtype=float)
        X = np.linspace(0.1,10,num=10)
        for df in df_range:
            iw = invwishart(df, scale)
            ig = invgamma(df/2, scale=1./2)

            # Statistics
            assert_allclose(iw.var(), ig.var())
            assert_allclose(iw.mean(), ig.mean())

            # PDF
            assert_allclose(iw.pdf(X), ig.pdf(X))

            # rvs
            rvs = iw.rvs(size=sn)
            args = (df/2, 0, 1./2)
            alpha = 0.01
            check_distribution_rvs('invgamma', args, alpha, rvs)

            # entropy
            assert_allclose(iw.entropy(), ig.entropy())

    def test_invwishart_2D_rvs(self):
        dim = 3
        df = 10

        # Construct a simple non-diagonal positive definite matrix
        scale = np.eye(dim)
        scale[0,1] = 0.5
        scale[1,0] = 0.5

        # Construct frozen inverse-Wishart random variables
        iw = invwishart(df, scale)

        # Get the generated random variables from a known seed
        np.random.seed(608072)
        iw_rvs = invwishart.rvs(df, scale)
        np.random.seed(608072)
        frozen_iw_rvs = iw.rvs()

        # Manually calculate what it should be, based on the decomposition in
        # https://arxiv.org/abs/2310.15884 of an invers-Wishart into L L',
        # where L A = D, D is the Cholesky factorization of the scale matrix,
        # and A is the lower triangular matrix with the square root of chi^2
        # variates on the diagonal and N(0,1) variates in the lower triangle.
        # the diagonal chi^2 variates in this A are reversed compared to those
        # in the Bartlett decomposition A for Wishart rvs.
        np.random.seed(608072)
        covariances = np.random.normal(size=3)
        variances = np.r_[
            np.random.chisquare(df-2),
            np.random.chisquare(df-1),
            np.random.chisquare(df),
        ]**0.5

        # Construct the lower-triangular A matrix
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances

        # inverse-Wishart random variate
        D = np.linalg.cholesky(scale)
        L = np.linalg.solve(A.T, D.T).T
        manual_iw_rvs = np.dot(L, L.T)

        # Test for equality
        assert_allclose(iw_rvs, manual_iw_rvs)
        assert_allclose(frozen_iw_rvs, manual_iw_rvs)

    def test_sample_mean(self):
        """Test that sample mean consistent with known mean."""
        # Construct an arbitrary positive definite scale matrix
        df = 10
        sample_size = 20_000
        for dim in [1, 5]:
            scale = np.diag(np.arange(dim) + 1)
            scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
            scale = np.dot(scale.T, scale)

            dist = invwishart(df, scale)
            Xmean_exp = dist.mean()
            Xvar_exp = dist.var()
            Xmean_std = (Xvar_exp / sample_size)**0.5  # asymptotic SE of mean estimate

            X = dist.rvs(size=sample_size, random_state=1234)
            Xmean_est = X.mean(axis=0)

            ntests = dim*(dim + 1)//2
            fail_rate = 0.01 / ntests  # correct for multiple tests
            max_diff = norm.ppf(1 - fail_rate / 2)
            assert np.allclose(
                (Xmean_est - Xmean_exp) / Xmean_std,
                0,
                atol=max_diff,
            )

    def test_logpdf_4x4(self):
        """Regression test for gh-8844."""
        X = np.array([[2, 1, 0, 0.5],
                      [1, 2, 0.5, 0.5],
                      [0, 0.5, 3, 1],
                      [0.5, 0.5, 1, 2]])
        Psi = np.array([[9, 7, 3, 1],
                        [7, 9, 5, 1],
                        [3, 5, 8, 2],
                        [1, 1, 2, 9]])
        nu = 6
        prob = invwishart.logpdf(X, nu, Psi)
        # Explicit calculation from the formula on wikipedia.
        p = X.shape[0]
        sig, logdetX = np.linalg.slogdet(X)
        sig, logdetPsi = np.linalg.slogdet(Psi)
        M = np.linalg.solve(X, Psi)
        expected = ((nu/2)*logdetPsi
                    - (nu*p/2)*np.log(2)
                    - multigammaln(nu/2, p)
                    - (nu + p + 1)/2*logdetX
                    - 0.5*M.trace())
        assert_allclose(prob, expected)


class TestSpecialOrthoGroup:
    def test_reproducibility(self):
        np.random.seed(514)
        x = special_ortho_group.rvs(3)
        expected = np.array([[-0.99394515, -0.04527879, 0.10011432],
                             [0.04821555, -0.99846897, 0.02711042],
                             [0.09873351, 0.03177334, 0.99460653]])
        assert_array_almost_equal(x, expected)

        random_state = np.random.RandomState(seed=514)
        x = special_ortho_group.rvs(3, random_state=random_state)
        assert_array_almost_equal(x, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, special_ortho_group.rvs, None)
        assert_raises(ValueError, special_ortho_group.rvs, (2, 2))
        assert_raises(ValueError, special_ortho_group.rvs, 1)
        assert_raises(ValueError, special_ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = special_ortho_group(dim)

        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = special_ortho_group.rvs(dim, random_state=1234)

        assert_equal(rvs1, rvs2)

    def test_det_and_ortho(self):
        xs = [special_ortho_group.rvs(dim)
              for dim in range(2,12)
              for i in range(3)]

        # Test that determinants are always +1
        dets = [np.linalg.det(x) for x in xs]
        assert_allclose(dets, [1.]*30, rtol=1e-13)

        # Test that these are orthogonal matrices
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T),
                                      np.eye(x.shape[0]))

    def test_haar(self):
        # Test that the distribution is constant under rotation
        # Every column should have the same distribution
        # Additionally, the distribution should be invariant under another rotation

        # Generate samples
        dim = 5
        samples = 1000  # Not too many, or the test takes too long
        ks_prob = .05
        np.random.seed(514)
        xs = special_ortho_group.rvs(dim, size=samples)

        # Dot a few rows (0, 1, 2) with unit vectors (0, 2, 4, 3),
        #   effectively picking off entries in the matrices of xs.
        #   These projections should all have the same distribution,
        #     establishing rotational invariance. We use the two-sided
        #     KS test to confirm this.
        #   We could instead test that angles between random vectors
        #     are uniformly distributed, but the below is sufficient.
        #   It is not feasible to consider all pairs, so pick a few.
        els = ((0,0), (0,2), (1,4), (2,3))
        #proj = {(er, ec): [x[er][ec] for x in xs] for er, ec in els}
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob]*len(pairs), ks_tests)


class TestOrthoGroup:
    def test_reproducibility(self):
        seed = 514
        np.random.seed(seed)
        x = ortho_group.rvs(3)
        x2 = ortho_group.rvs(3, random_state=seed)
        # Note this matrix has det -1, distinguishing O(N) from SO(N)
        assert_almost_equal(np.linalg.det(x), -1)
        expected = np.array([[0.381686, -0.090374, 0.919863],
                             [0.905794, -0.161537, -0.391718],
                             [-0.183993, -0.98272, -0.020204]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, ortho_group.rvs, None)
        assert_raises(ValueError, ortho_group.rvs, (2, 2))
        assert_raises(ValueError, ortho_group.rvs, 1)
        assert_raises(ValueError, ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = ortho_group(dim)
        frozen_seed = ortho_group(dim, seed=1234)

        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = ortho_group.rvs(dim, random_state=1234)
        rvs3 = frozen_seed.rvs(size=1)

        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_det_and_ortho(self):
        xs = [[ortho_group.rvs(dim)
               for i in range(10)]
              for dim in range(2,12)]

        # Test that abs determinants are always +1
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)

        # Test that these are orthogonal matrices
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T),
                                          np.eye(x.shape[0]))

    @pytest.mark.parametrize("dim", [2, 5, 10, 20])
    def test_det_distribution_gh18272(self, dim):
        # Test that positive and negative determinants are equally likely.
        rng = np.random.default_rng(6796248956179332344)
        dist = ortho_group(dim=dim)
        rvs = dist.rvs(size=5000, random_state=rng)
        dets = scipy.linalg.det(rvs)
        k = np.sum(dets > 0)
        n = len(dets)
        res = stats.binomtest(k, n)
        low, high = res.proportion_ci(confidence_level=0.95)
        assert low < 0.5 < high

    def test_haar(self):
        # Test that the distribution is constant under rotation
        # Every column should have the same distribution
        # Additionally, the distribution should be invariant under another rotation

        # Generate samples
        dim = 5
        samples = 1000  # Not too many, or the test takes too long
        ks_prob = .05
        np.random.seed(518)  # Note that the test is sensitive to seed too
        xs = ortho_group.rvs(dim, size=samples)

        # Dot a few rows (0, 1, 2) with unit vectors (0, 2, 4, 3),
        #   effectively picking off entries in the matrices of xs.
        #   These projections should all have the same distribution,
        #     establishing rotational invariance. We use the two-sided
        #     KS test to confirm this.
        #   We could instead test that angles between random vectors
        #     are uniformly distributed, but the below is sufficient.
        #   It is not feasible to consider all pairs, so pick a few.
        els = ((0,0), (0,2), (1,4), (2,3))
        #proj = {(er, ec): [x[er][ec] for x in xs] for er, ec in els}
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob]*len(pairs), ks_tests)

    @pytest.mark.slow
    def test_pairwise_distances(self):
        # Test that the distribution of pairwise distances is close to correct.
        np.random.seed(514)

        def random_ortho(dim):
            u, _s, v = np.linalg.svd(np.random.normal(size=(dim, dim)))
            return np.dot(u, v)

        for dim in range(2, 6):
            def generate_test_statistics(rvs, N=1000, eps=1e-10):
                stats = np.array([
                    np.sum((rvs(dim=dim) - rvs(dim=dim))**2)
                    for _ in range(N)
                ])
                # Add a bit of noise to account for numeric accuracy.
                stats += np.random.uniform(-eps, eps, size=stats.shape)
                return stats

            expected = generate_test_statistics(random_ortho)
            actual = generate_test_statistics(scipy.stats.ortho_group.rvs)

            _D, p = scipy.stats.ks_2samp(expected, actual)

            assert_array_less(.05, p)


class TestRandomCorrelation:
    def test_reproducibility(self):
        np.random.seed(514)
        eigs = (.5, .8, 1.2, 1.5)
        x = random_correlation.rvs(eigs)
        x2 = random_correlation.rvs(eigs, random_state=514)
        expected = np.array([[1., -0.184851, 0.109017, -0.227494],
                             [-0.184851, 1., 0.231236, 0.326669],
                             [0.109017, 0.231236, 1., -0.178912],
                             [-0.227494, 0.326669, -0.178912, 1.]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_eigs(self):
        assert_raises(ValueError, random_correlation.rvs, None)
        assert_raises(ValueError, random_correlation.rvs, 'test')
        assert_raises(ValueError, random_correlation.rvs, 2.5)
        assert_raises(ValueError, random_correlation.rvs, [2.5])
        assert_raises(ValueError, random_correlation.rvs, [[1,2],[3,4]])
        assert_raises(ValueError, random_correlation.rvs, [2.5, -.5])
        assert_raises(ValueError, random_correlation.rvs, [1, 2, .1])

    def test_frozen_matrix(self):
        eigs = (.5, .8, 1.2, 1.5)
        frozen = random_correlation(eigs)
        frozen_seed = random_correlation(eigs, seed=514)

        rvs1 = random_correlation.rvs(eigs, random_state=514)
        rvs2 = frozen.rvs(random_state=514)
        rvs3 = frozen_seed.rvs()

        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_definition(self):
        # Test the definition of a correlation matrix in several dimensions:
        #
        # 1. Det is product of eigenvalues (and positive by construction
        #    in examples)
        # 2. 1's on diagonal
        # 3. Matrix is symmetric

        def norm(i, e):
            return i*e/sum(e)

        np.random.seed(123)

        eigs = [norm(i, np.random.uniform(size=i)) for i in range(2, 6)]
        eigs.append([4,0,0,0])

        ones = [[1.]*len(e) for e in eigs]
        xs = [random_correlation.rvs(e) for e in eigs]

        # Test that determinants are products of eigenvalues
        #   These are positive by construction
        # Could also test that the eigenvalues themselves are correct,
        #   but this seems sufficient.
        dets = [np.fabs(np.linalg.det(x)) for x in xs]
        dets_known = [np.prod(e) for e in eigs]
        assert_allclose(dets, dets_known, rtol=1e-13, atol=1e-13)

        # Test for 1's on the diagonal
        diags = [np.diag(x) for x in xs]
        for a, b in zip(diags, ones):
            assert_allclose(a, b, rtol=1e-13)

        # Correlation matrices are symmetric
        for x in xs:
            assert_allclose(x, x.T, rtol=1e-13)

    def test_to_corr(self):
        # Check some corner cases in to_corr

        # ajj == 1
        m = np.array([[0.1, 0], [0, 1]], dtype=float)
        m = random_correlation._to_corr(m)
        assert_allclose(m, np.array([[1, 0], [0, 0.1]]))

        # Floating point overflow; fails to compute the correct
        # rotation, but should still produce some valid rotation
        # rather than infs/nans
        with np.errstate(over='ignore'):
            g = np.array([[0, 1], [-1, 0]])

            m0 = np.array([[1e300, 0], [0, np.nextafter(1, 0)]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))

            m0 = np.array([[0.9, 1e300], [1e300, 1.1]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))

        # Zero discriminant; should set the first diag entry to 1
        m0 = np.array([[2, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0,0], 1)

        # Slightly negative discriminant; should be approx correct still
        m0 = np.array([[2 + 1e-7, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0,0], 1)


class TestUniformDirection:
    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("size", [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        # test that samples have correct shape and norm 1
        rng = np.random.default_rng(2777937887058094419)
        uniform_direction_dist = uniform_direction(dim, seed=rng)
        samples = uniform_direction_dist.rvs(size)
        mean, cov = np.zeros(dim), np.eye(dim)
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.)

    @pytest.mark.parametrize("dim", [None, 0, (2, 2), 2.5])
    def test_invalid_dim(self, dim):
        message = ("Dimension of vector must be specified, "
                   "and must be an integer greater than 0.")
        with pytest.raises(ValueError, match=message):
            uniform_direction.rvs(dim)

    def test_frozen_distribution(self):
        dim = 5
        frozen = uniform_direction(dim)
        frozen_seed = uniform_direction(dim, seed=514)

        rvs1 = frozen.rvs(random_state=514)
        rvs2 = uniform_direction.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs()

        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    @pytest.mark.parametrize("dim", [2, 5, 8])
    def test_uniform(self, dim):
        rng = np.random.default_rng(1036978481269651776)
        spherical_dist = uniform_direction(dim, seed=rng)
        # generate random, orthogonal vectors
        v1, v2 = spherical_dist.rvs(size=2)
        v2 -= v1 @ v2 * v1
        v2 /= np.linalg.norm(v2)
        assert_allclose(v1 @ v2, 0, atol=1e-14)  # orthogonal
        # generate data and project onto orthogonal vectors
        samples = spherical_dist.rvs(size=10000)
        s1 = samples @ v1
        s2 = samples @ v2
        angles = np.arctan2(s1, s2)
        # test that angles follow a uniform distribution
        # normalize angles to range [0, 1]
        angles += np.pi
        angles /= 2*np.pi
        # perform KS test
        uniform_dist = uniform()
        kstest_result = kstest(angles, uniform_dist.cdf)
        assert kstest_result.pvalue > 0.05


class TestUnitaryGroup:
    def test_reproducibility(self):
        np.random.seed(514)
        x = unitary_group.rvs(3)
        x2 = unitary_group.rvs(3, random_state=514)

        expected = np.array(
            [[0.308771+0.360312j, 0.044021+0.622082j, 0.160327+0.600173j],
             [0.732757+0.297107j, 0.076692-0.4614j, -0.394349+0.022613j],
             [-0.148844+0.357037j, -0.284602-0.557949j, 0.607051+0.299257j]]
        )

        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, unitary_group.rvs, None)
        assert_raises(ValueError, unitary_group.rvs, (2, 2))
        assert_raises(ValueError, unitary_group.rvs, 1)
        assert_raises(ValueError, unitary_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = unitary_group(dim)
        frozen_seed = unitary_group(dim, seed=514)

        rvs1 = frozen.rvs(random_state=514)
        rvs2 = unitary_group.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs(size=1)

        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_unitarity(self):
        xs = [unitary_group.rvs(dim)
              for dim in range(2,12)
              for i in range(3)]

        # Test that these are unitary matrices
        for x in xs:
            assert_allclose(np.dot(x, x.conj().T), np.eye(x.shape[0]), atol=1e-15)

    def test_haar(self):
        # Test that the eigenvalues, which lie on the unit circle in
        # the complex plane, are uncorrelated.

        # Generate samples
        dim = 5
        samples = 1000  # Not too many, or the test takes too long
        np.random.seed(514)  # Note that the test is sensitive to seed too
        xs = unitary_group.rvs(dim, size=samples)

        # The angles "x" of the eigenvalues should be uniformly distributed
        # Overall this seems to be a necessary but weak test of the distribution.
        eigs = np.vstack([scipy.linalg.eigvals(x) for x in xs])
        x = np.arctan2(eigs.imag, eigs.real)
        res = kstest(x.ravel(), uniform(-np.pi, 2*np.pi).cdf)
        assert_(res.pvalue > 0.05)


class TestMultivariateT:

    # These tests were created by running vpa(mvtpdf(...)) in MATLAB. The
    # function takes no `mu` parameter. The tests were run as
    #
    # >> ans = vpa(mvtpdf(x - mu, shape, df));
    #
    PDF_TESTS = [(
        # x
        [
            [1, 2],
            [4, 1],
            [2, 1],
            [2, 4],
            [1, 4],
            [4, 1],
            [3, 2],
            [3, 3],
            [4, 4],
            [5, 1],
        ],
        # loc
        [0, 0],
        # shape
        [
            [1, 0],
            [0, 1]
        ],
        # df
        4,
        # ans
        [
            0.013972450422333741737457302178882,
            0.0010998721906793330026219646100571,
            0.013972450422333741737457302178882,
            0.00073682844024025606101402363634634,
            0.0010998721906793330026219646100571,
            0.0010998721906793330026219646100571,
            0.0020732579600816823488240725481546,
            0.00095660371505271429414668515889275,
            0.00021831953784896498569831346792114,
            0.00037725616140301147447000396084604
        ]

    ), (
        # x
        [
            [0.9718, 0.1298, 0.8134],
            [0.4922, 0.5522, 0.7185],
            [0.3010, 0.1491, 0.5008],
            [0.5971, 0.2585, 0.8940],
            [0.5434, 0.5287, 0.9507],
        ],
        # loc
        [-1, 1, 50],
        # shape
        [
            [1.0000, 0.5000, 0.2500],
            [0.5000, 1.0000, -0.1000],
            [0.2500, -0.1000, 1.0000],
        ],
        # df
        8,
        # ans
        [
            0.00000000000000069609279697467772867405511133763,
            0.00000000000000073700739052207366474839369535934,
            0.00000000000000069522909962669171512174435447027,
            0.00000000000000074212293557998314091880208889767,
            0.00000000000000077039675154022118593323030449058,
        ]
    )]

    @pytest.mark.parametrize("x, loc, shape, df, ans", PDF_TESTS)
    def test_pdf_correctness(self, x, loc, shape, df, ans):
        dist = multivariate_t(loc, shape, df, seed=0)
        val = dist.pdf(x)
        assert_array_almost_equal(val, ans)

    @pytest.mark.parametrize("x, loc, shape, df, ans", PDF_TESTS)
    def test_logpdf_correct(self, x, loc, shape, df, ans):
        dist = multivariate_t(loc, shape, df, seed=0)
        val1 = dist.pdf(x)
        val2 = dist.logpdf(x)
        assert_array_almost_equal(np.log(val1), val2)

    # https://github.com/scipy/scipy/issues/10042#issuecomment-576795195
    def test_mvt_with_df_one_is_cauchy(self):
        x = [9, 7, 4, 1, -3, 9, 0, -3, -1, 3]
        val = multivariate_t.pdf(x, df=1)
        ans = cauchy.pdf(x)
        assert_array_almost_equal(val, ans)

    def test_mvt_with_high_df_is_approx_normal(self):
        # `normaltest` returns the chi-squared statistic and the associated
        # p-value. The null hypothesis is that `x` came from a normal
        # distribution, so a low p-value represents rejecting the null, i.e.
        # that it is unlikely that `x` came a normal distribution.
        P_VAL_MIN = 0.1

        dist = multivariate_t(0, 1, df=100000, seed=1)
        samples = dist.rvs(size=100000)
        _, p = normaltest(samples)
        assert (p > P_VAL_MIN)

        dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000,
                              seed=42)
        samples = dist.rvs(size=100000)
        _, p = normaltest(samples)
        assert ((p > P_VAL_MIN).all())

    @patch('scipy.stats.multivariate_normal._logpdf')
    def test_mvt_with_inf_df_calls_normal(self, mock):
        dist = multivariate_t(0, 1, df=np.inf, seed=7)
        assert isinstance(dist, multivariate_normal_frozen)
        multivariate_t.pdf(0, df=np.inf)
        assert mock.call_count == 1
        multivariate_t.logpdf(0, df=np.inf)
        assert mock.call_count == 2

    def test_shape_correctness(self):
        # pdf and logpdf should return scalar when the
        # number of samples in x is one.
        dim = 4
        loc = np.zeros(dim)
        shape = np.eye(dim)
        df = 4.5
        x = np.zeros(dim)
        res = multivariate_t(loc, shape, df).pdf(x)
        assert np.isscalar(res)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert np.isscalar(res)

        # pdf() and logpdf() should return probabilities of shape
        # (n_samples,) when x has n_samples.
        n_samples = 7
        x = np.random.random((n_samples, dim))
        res = multivariate_t(loc, shape, df).pdf(x)
        assert (res.shape == (n_samples,))
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert (res.shape == (n_samples,))

        # rvs() should return scalar unless a size argument is applied.
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs()
        assert np.isscalar(res)

        # rvs() should return vector of shape (size,) if size argument
        # is applied.
        size = 7
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs(size=size)
        assert (res.shape == (size,))

    def test_default_arguments(self):
        dist = multivariate_t()
        assert_equal(dist.loc, [0])
        assert_equal(dist.shape, [[1]])
        assert (dist.df == 1)

    DEFAULT_ARGS_TESTS = [
        (None, None, None, 0, 1, 1),
        (None, None, 7, 0, 1, 7),
        (None, [[7, 0], [0, 7]], None, [0, 0], [[7, 0], [0, 7]], 1),
        (None, [[7, 0], [0, 7]], 7, [0, 0], [[7, 0], [0, 7]], 7),
        ([7, 7], None, None, [7, 7], [[1, 0], [0, 1]], 1),
        ([7, 7], None, 7, [7, 7], [[1, 0], [0, 1]], 7),
        ([7, 7], [[7, 0], [0, 7]], None, [7, 7], [[7, 0], [0, 7]], 1),
        ([7, 7], [[7, 0], [0, 7]], 7, [7, 7], [[7, 0], [0, 7]], 7)
    ]

    @pytest.mark.parametrize("loc, shape, df, loc_ans, shape_ans, df_ans",
                             DEFAULT_ARGS_TESTS)
    def test_default_args(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        dist = multivariate_t(loc=loc, shape=shape, df=df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert (dist.df == df_ans)

    ARGS_SHAPES_TESTS = [
        (-1, 2, 3, [-1], [[2]], 3),
        ([-1], [2], 3, [-1], [[2]], 3),
        (np.array([-1]), np.array([2]), 3, [-1], [[2]], 3)
    ]

    @pytest.mark.parametrize("loc, shape, df, loc_ans, shape_ans, df_ans",
                             ARGS_SHAPES_TESTS)
    def test_scalar_list_and_ndarray_arguments(self, loc, shape, df, loc_ans,
                                               shape_ans, df_ans):
        dist = multivariate_t(loc, shape, df)
        assert_equal(dist.loc, loc_ans)
        assert_equal(dist.shape, shape_ans)
        assert_equal(dist.df, df_ans)

    def test_argument_error_handling(self):
        # `loc` should be a one-dimensional vector.
        loc = [[1, 1]]
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc))

        # `shape` should be scalar or square matrix.
        shape = [[1, 1], [2, 2], [3, 3]]
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape))

        # `df` should be greater than zero.
        loc = np.zeros(2)
        shape = np.eye(2)
        df = -1
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape, df=df))
        df = 0
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape, df=df))

    def test_reproducibility(self):
        rng = np.random.RandomState(4)
        loc = rng.uniform(size=3)
        shape = np.eye(3)
        dist1 = multivariate_t(loc, shape, df=3, seed=2)
        dist2 = multivariate_t(loc, shape, df=3, seed=2)
        samples1 = dist1.rvs(size=10)
        samples2 = dist2.rvs(size=10)
        assert_equal(samples1, samples2)

    def test_allow_singular(self):
        # Make shape singular and verify error was raised.
        args = dict(loc=[0,0], shape=[[0,0],[0,1]], df=1, allow_singular=False)
        assert_raises(np.linalg.LinAlgError, multivariate_t, **args)

    @pytest.mark.parametrize("size", [(10, 3), (5, 6, 4, 3)])
    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    @pytest.mark.parametrize("df", [1., 2., np.inf])
    def test_rvs(self, size, dim, df):
        dist = multivariate_t(np.zeros(dim), np.eye(dim), df)
        rvs = dist.rvs(size=size)
        assert rvs.shape == size + (dim, )

    def test_cdf_signs(self):
        # check that sign of output is correct when np.any(lower > x)
        mean = np.zeros(3)
        cov = np.eye(3)
        df = 10
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        # when odd number of elements of b < a, output is negative
        expected_signs = np.array([1, -1, -1, 1])
        cdf = multivariate_normal.cdf(b, mean, cov, df, lower_limit=a)
        assert_allclose(cdf, cdf[0]*expected_signs)

    @pytest.mark.parametrize('dim', [1, 2, 5, 10])
    def test_cdf_against_multivariate_normal(self, dim):
        # Check accuracy against MVN randomly-generated cases
        self.cdf_against_mvn_test(dim)

    @pytest.mark.parametrize('dim', [3, 6, 9])
    def test_cdf_against_multivariate_normal_singular(self, dim):
        # Check accuracy against MVN for randomly-generated singular cases
        self.cdf_against_mvn_test(3, True)

    def cdf_against_mvn_test(self, dim, singular=False):
        # Check for accuracy in the limit that df -> oo and MVT -> MVN
        rng = np.random.default_rng(413722918996573)
        n = 3

        w = 10**rng.uniform(-2, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular)

        mean = 10**rng.uniform(-1, 2, size=dim) * np.sign(rng.normal(size=dim))
        a = -10**rng.uniform(-1, 2, size=(n, dim)) + mean
        b = 10**rng.uniform(-1, 2, size=(n, dim)) + mean

        res = stats.multivariate_t.cdf(b, mean, cov, df=10000, lower_limit=a,
                                       allow_singular=True, random_state=rng)
        ref = stats.multivariate_normal.cdf(b, mean, cov, allow_singular=True,
                                            lower_limit=a)
        assert_allclose(res, ref, atol=5e-4)

    def test_cdf_against_univariate_t(self):
        rng = np.random.default_rng(413722918996573)
        cov = 2
        mean = 0
        x = rng.normal(size=10, scale=np.sqrt(cov))
        df = 3

        res = stats.multivariate_t.cdf(x, mean, cov, df, lower_limit=-np.inf,
                                       random_state=rng)
        ref = stats.t.cdf(x, df, mean, np.sqrt(cov))
        incorrect = stats.norm.cdf(x, mean, np.sqrt(cov))

        assert_allclose(res, ref, atol=5e-4)  # close to t
        assert np.all(np.abs(res - incorrect) > 1e-3)  # not close to normal

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    @pytest.mark.parametrize("seed", [3363958638, 7891119608, 3887698049,
                                      5013150848, 1495033423, 6170824608])
    @pytest.mark.parametrize("singular", [False, True])
    def test_cdf_against_qsimvtv(self, dim, seed, singular):
        if singular and seed != 3363958638:
            pytest.skip('Agreement with qsimvtv is not great in singular case')
        rng = np.random.default_rng(seed)
        w = 10**rng.uniform(-2, 2, size=dim)
        cov = _random_covariance(dim, w, rng, singular)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5

        # no lower limit
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng,
                                       allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, np.inf*a, b - mean, rng)[0]
        assert_allclose(res, ref, atol=2e-4, rtol=1e-3)

        # with lower limit
        res = stats.multivariate_t.cdf(b, mean, cov, df, lower_limit=a,
                                       random_state=rng, allow_singular=True)
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, a - mean, b - mean, rng)[0]
        assert_allclose(res, ref, atol=1e-4, rtol=1e-3)

    def test_cdf_against_generic_integrators(self):
        # Compare result against generic numerical integrators
        dim = 3
        rng = np.random.default_rng(41372291899657)
        w = 10 ** rng.uniform(-1, 1, size=dim)
        cov = _random_covariance(dim, w, rng, singular=True)
        mean = rng.random(dim)
        a = -rng.random(dim)
        b = rng.random(dim)
        df = rng.random() * 5

        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng,
                                       lower_limit=a)

        def integrand(x):
            return stats.multivariate_t.pdf(x.T, mean, cov, df)

        ref = qmc_quad(integrand, a, b, qrng=stats.qmc.Halton(d=dim, seed=rng))
        assert_allclose(res, ref.integral, rtol=1e-3)

        def integrand(*zyx):
            return stats.multivariate_t.pdf(zyx[::-1], mean, cov, df)

        ref = tplquad(integrand, a[0], b[0], a[1], b[1], a[2], b[2])
        assert_allclose(res, ref[0], rtol=1e-3)

    def test_against_matlab(self):
        # Test against matlab mvtcdf:
        # C = [6.21786909  0.2333667 7.95506077;
        #      0.2333667 29.67390923 16.53946426;
        #      7.95506077 16.53946426 19.17725252]
        # df = 1.9559939787727658
        # mvtcdf([0, 0, 0], C, df)  % 0.2523
        rng = np.random.default_rng(2967390923)
        cov = np.array([[ 6.21786909,  0.2333667 ,  7.95506077],
                        [ 0.2333667 , 29.67390923, 16.53946426],
                        [ 7.95506077, 16.53946426, 19.17725252]])
        df = 1.9559939787727658
        dist = stats.multivariate_t(shape=cov, df=df)
        res = dist.cdf([0, 0, 0], random_state=rng)
        ref = 0.2523
        assert_allclose(res, ref, rtol=1e-3)

    def test_frozen(self):
        seed = 4137229573
        rng = np.random.default_rng(seed)
        loc = rng.uniform(size=3)
        x = rng.uniform(size=3) + loc
        shape = np.eye(3)
        df = rng.random()
        args = (loc, shape, df)

        rng_frozen = np.random.default_rng(seed)
        rng_unfrozen = np.random.default_rng(seed)
        dist = stats.multivariate_t(*args, seed=rng_frozen)
        assert_equal(dist.cdf(x),
                     multivariate_t.cdf(x, *args, random_state=rng_unfrozen))

    def test_vectorized(self):
        dim = 4
        n = (2, 3)
        rng = np.random.default_rng(413722918996573)
        A = rng.random(size=(dim, dim))
        cov = A @ A.T
        mean = rng.random(dim)
        x = rng.random(n + (dim,))
        df = rng.random() * 5

        res = stats.multivariate_t.cdf(x, mean, cov, df, random_state=rng)

        def _cdf_1d(x):
            return _qsimvtv(10000, df, cov, -np.inf*x, x-mean, rng)[0]

        ref = np.apply_along_axis(_cdf_1d, -1, x)
        assert_allclose(res, ref, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("dim", (3, 7))
    def test_against_analytical(self, dim):
        rng = np.random.default_rng(413722918996573)
        A = scipy.linalg.toeplitz(c=[1] + [0.5] * (dim - 1))
        res = stats.multivariate_t(shape=A).cdf([0] * dim, random_state=rng)
        ref = 1 / (dim + 1)
        assert_allclose(res, ref, rtol=5e-5)

    def test_entropy_inf_df(self):
        cov = np.eye(3, 3)
        df = np.inf
        mvt_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mvn_entropy = stats.multivariate_normal.entropy(None, cov)
        assert mvt_entropy == mvn_entropy

    @pytest.mark.parametrize("df", [1, 10, 100])
    def test_entropy_1d(self, df):
        mvt_entropy = stats.multivariate_t.entropy(shape=1., df=df)
        t_entropy = stats.t.entropy(df=df)
        assert_allclose(mvt_entropy, t_entropy, rtol=1e-13)

    # entropy reference values were computed via numerical integration
    #
    # def integrand(x, y, mvt):
    #     vec = np.array([x, y])
    #     return mvt.logpdf(vec) * mvt.pdf(vec)

    # def multivariate_t_entropy_quad_2d(df, cov):
    #     dim = cov.shape[0]
    #     loc = np.zeros((dim, ))
    #     mvt = stats.multivariate_t(loc, cov, df)
    #     limit = 100
    #     return -integrate.dblquad(integrand, -limit, limit, -limit, limit,
    #                               args=(mvt, ))[0]

    @pytest.mark.parametrize("df, cov, ref, tol",
                             [(10, np.eye(2, 2), 3.0378770664093313, 1e-14),
                              (100, np.array([[0.5, 1], [1, 10]]),
                               3.55102424550609, 1e-8)])
    def test_entropy_vs_numerical_integration(self, df, cov, ref, tol):
        loc = np.zeros((2, ))
        mvt = stats.multivariate_t(loc, cov, df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    @pytest.mark.parametrize(
        "df, dim, ref, tol",
        [
            (10, 1, 1.5212624929756808, 1e-15),
            (100, 1, 1.4289633653182439, 1e-13),
            (500, 1, 1.420939531869349, 1e-14),
            (1e20, 1, 1.4189385332046727, 1e-15),
            (1e100, 1, 1.4189385332046727, 1e-15),
            (10, 10, 15.069150450832911, 1e-15),
            (1000, 10, 14.19936546446673, 1e-13),
            (1e20, 10, 14.189385332046728, 1e-15),
            (1e100, 10, 14.189385332046728, 1e-15),
            (10, 100, 148.28902883192654, 1e-15),
            (1000, 100, 141.99155538003762, 1e-14),
            (1e20, 100, 141.8938533204673, 1e-15),
            (1e100, 100, 141.8938533204673, 1e-15),
        ]
    )
    def test_extreme_entropy(self, df, dim, ref, tol):
        # Reference values were calculated with mpmath:
        # from mpmath import mp
        # mp.dps = 500
        #
        # def mul_t_mpmath_entropy(dim, df=1):
        #     dim = mp.mpf(dim)
        #     df = mp.mpf(df)
        #     halfsum = (dim + df)/2
        #     half_df = df/2
        #
        #     return float(
        #         -mp.loggamma(halfsum) + mp.loggamma(half_df)
        #         + dim / 2 * mp.log(df * mp.pi)
        #         + halfsum * (mp.digamma(halfsum) - mp.digamma(half_df))
        #         + 0.0
        #     )
        mvt = stats.multivariate_t(shape=np.eye(dim), df=df)
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    def test_entropy_with_covariance(self):
        # Generated using np.randn(5, 5) and then rounding
        # to two decimal places
        _A = np.array([
            [1.42, 0.09, -0.49, 0.17, 0.74],
            [-1.13, -0.01,  0.71, 0.4, -0.56],
            [1.07, 0.44, -0.28, -0.44, 0.29],
            [-1.5, -0.94, -0.67, 0.73, -1.1],
            [0.17, -0.08, 1.46, -0.32, 1.36]
        ])
        # Set cov to be a symmetric positive semi-definite matrix
        cov = _A @ _A.T

        # Test the asymptotic case. For large degrees of freedom
        # the entropy approaches the multivariate normal entropy.
        df = 1e20
        mul_t_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mul_norm_entropy = multivariate_normal(None, cov=cov).entropy()
        assert_allclose(mul_t_entropy, mul_norm_entropy, rtol=1e-15)

        # Test the regular case. For a dim of 5 the threshold comes out
        # to be approximately 766.45. So using slightly
        # different dfs on each site of the threshold, the entropies
        # are being compared.
        df1 = 765
        df2 = 768
        _entropy1 = stats.multivariate_t.entropy(shape=cov, df=df1)
        _entropy2 = stats.multivariate_t.entropy(shape=cov, df=df2)
        assert_allclose(_entropy1, _entropy2, rtol=1e-5)


class TestMultivariateHypergeom:
    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            # Ground truth value from R dmvhyper
            ([3, 4], [5, 10], 7, -1.119814),
            # test for `n=0`
            ([3, 4], [5, 10], 0, -np.inf),
            # test for `x < 0`
            ([-3, 4], [5, 10], 7, -np.inf),
            # test for `m < 0` (RuntimeWarning issue)
            ([3, 4], [-5, 10], 7, np.nan),
            # test for all `m < 0` and `x.sum() != n`
            ([[1, 2], [3, 4]], [[-4, -6], [-5, -10]],
             [3, 7], [np.nan, np.nan]),
            # test for `x < 0` and `m < 0` (RuntimeWarning issue)
            ([-3, 4], [-5, 10], 1, np.nan),
            # test for `x > m`
            ([1, 11], [10, 1], 12, np.nan),
            # test for `m < 0` (RuntimeWarning issue)
            ([1, 11], [10, -1], 12, np.nan),
            # test for `n < 0`
            ([3, 4], [5, 10], -7, np.nan),
            # test for `x.sum() != n`
            ([3, 3], [5, 10], 7, -np.inf)
        ]
    )
    def test_logpmf(self, x, m, n, expected):
        vals = multivariate_hypergeom.logpmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-6)

    def test_reduces_hypergeom(self):
        # test that the multivariate_hypergeom pmf reduces to the
        # hypergeom pmf in the 2d case.
        val1 = multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
        val2 = hypergeom.pmf(k=3, M=15, n=4, N=10)
        assert_allclose(val1, val2, rtol=1e-8)

        val1 = multivariate_hypergeom.pmf(x=[7, 3], m=[15, 10], n=10)
        val2 = hypergeom.pmf(k=7, M=25, n=10, N=15)
        assert_allclose(val1, val2, rtol=1e-8)

    def test_rvs(self):
        # test if `rvs` is unbiased and large sample size converges
        # to the true mean.
        rv = multivariate_hypergeom(m=[3, 5], n=4)
        rvs = rv.rvs(size=1000, random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=1e-2)

    def test_rvs_broadcasting(self):
        rv = multivariate_hypergeom(m=[[3, 5], [5, 10]], n=[4, 9])
        rvs = rv.rvs(size=(1000, 2), random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=1e-2)

    @pytest.mark.parametrize('m, n', (
        ([0, 0, 20, 0, 0], 5), ([0, 0, 0, 0, 0], 0),
        ([0, 0], 0), ([0], 0)
    ))
    def test_rvs_gh16171(self, m, n):
        res = multivariate_hypergeom.rvs(m, n)
        m = np.asarray(m)
        res_ex = m.copy()
        res_ex[m != 0] = n
        assert_equal(res, res_ex)

    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            ([5], [5], 5, 1),
            ([3, 4], [5, 10], 7, 0.3263403),
            # Ground truth value from R dmvhyper
            ([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]],
             [5, 10], [[8, 8], [8, 2]],
             [[0.3916084, 0.006993007], [0, 0.4761905]]),
            # test with empty arrays.
            (np.array([], dtype=int), np.array([], dtype=int), 0, []),
            ([1, 2], [4, 5], 5, 0),
            # Ground truth value from R dmvhyper
            ([3, 3, 0], [5, 6, 7], 6, 0.01077354)
        ]
    )
    def test_pmf(self, x, m, n, expected):
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-7)

    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            ([3, 4], [[5, 10], [10, 15]], 7, [0.3263403, 0.3407531]),
            ([[1], [2]], [[3], [4]], [1, 3], [1., 0.]),
            ([[[1], [2]]], [[3], [4]], [1, 3], [[1., 0.]]),
            ([[1], [2]], [[[[3]]]], [1, 3], [[[1., 0.]]])
        ]
    )
    def test_pmf_broadcasting(self, x, m, n, expected):
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-7)

    def test_cov(self):
        cov1 = multivariate_hypergeom.cov(m=[3, 7, 10], n=12)
        cov2 = [[0.64421053, -0.26526316, -0.37894737],
                [-0.26526316, 1.14947368, -0.88421053],
                [-0.37894737, -0.88421053, 1.26315789]]
        assert_allclose(cov1, cov2, rtol=1e-8)

    def test_cov_broadcasting(self):
        cov1 = multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
        cov2 = [[[1.05, -1.05], [-1.05, 1.05]],
                [[1.56, -1.56], [-1.56, 1.56]]]
        assert_allclose(cov1, cov2, rtol=1e-8)

        cov3 = multivariate_hypergeom.cov(m=[[4], [5]], n=[4, 5])
        cov4 = [[[0.]], [[0.]]]
        assert_allclose(cov3, cov4, rtol=1e-8)

        cov5 = multivariate_hypergeom.cov(m=[7, 9], n=[8, 12])
        cov6 = [[[1.05, -1.05], [-1.05, 1.05]],
                [[0.7875, -0.7875], [-0.7875, 0.7875]]]
        assert_allclose(cov5, cov6, rtol=1e-8)

    def test_var(self):
        # test with hypergeom
        var0 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var1 = hypergeom.var(M=15, n=4, N=10)
        assert_allclose(var0, var1, rtol=1e-8)

    def test_var_broadcasting(self):
        var0 = multivariate_hypergeom.var(m=[10, 5], n=[4, 8])
        var1 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var2 = multivariate_hypergeom.var(m=[10, 5], n=8)
        assert_allclose(var0[0], var1, rtol=1e-8)
        assert_allclose(var0[1], var2, rtol=1e-8)

        var3 = multivariate_hypergeom.var(m=[[10, 5], [10, 14]], n=[4, 8])
        var4 = [[0.6984127, 0.6984127], [1.352657, 1.352657]]
        assert_allclose(var3, var4, rtol=1e-8)

        var5 = multivariate_hypergeom.var(m=[[5], [10]], n=[5, 10])
        var6 = [[0.], [0.]]
        assert_allclose(var5, var6, rtol=1e-8)

    def test_mean(self):
        # test with hypergeom
        mean0 = multivariate_hypergeom.mean(m=[10, 5], n=4)
        mean1 = hypergeom.mean(M=15, n=4, N=10)
        assert_allclose(mean0[0], mean1, rtol=1e-8)

        mean2 = multivariate_hypergeom.mean(m=[12, 8], n=10)
        mean3 = [12.*10./20., 8.*10./20.]
        assert_allclose(mean2, mean3, rtol=1e-8)

    def test_mean_broadcasting(self):
        mean0 = multivariate_hypergeom.mean(m=[[3, 5], [10, 5]], n=[4, 8])
        mean1 = [[3.*4./8., 5.*4./8.], [10.*8./15., 5.*8./15.]]
        assert_allclose(mean0, mean1, rtol=1e-8)

    def test_mean_edge_cases(self):
        mean0 = multivariate_hypergeom.mean(m=[0, 0, 0], n=0)
        assert_equal(mean0, [0., 0., 0.])

        mean1 = multivariate_hypergeom.mean(m=[1, 0, 0], n=2)
        assert_equal(mean1, [np.nan, np.nan, np.nan])

        mean2 = multivariate_hypergeom.mean(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(mean2, [[np.nan, np.nan, np.nan], [1., 0., 1.]],
                        rtol=1e-17)

        mean3 = multivariate_hypergeom.mean(m=np.array([], dtype=int), n=0)
        assert_equal(mean3, [])
        assert_(mean3.shape == (0, ))

    def test_var_edge_cases(self):
        var0 = multivariate_hypergeom.var(m=[0, 0, 0], n=0)
        assert_allclose(var0, [0., 0., 0.], rtol=1e-16)

        var1 = multivariate_hypergeom.var(m=[1, 0, 0], n=2)
        assert_equal(var1, [np.nan, np.nan, np.nan])

        var2 = multivariate_hypergeom.var(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(var2, [[np.nan, np.nan, np.nan], [0., 0., 0.]],
                        rtol=1e-17)

        var3 = multivariate_hypergeom.var(m=np.array([], dtype=int), n=0)
        assert_equal(var3, [])
        assert_(var3.shape == (0, ))

    def test_cov_edge_cases(self):
        cov0 = multivariate_hypergeom.cov(m=[1, 0, 0], n=1)
        cov1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        assert_allclose(cov0, cov1, rtol=1e-17)

        cov3 = multivariate_hypergeom.cov(m=[0, 0, 0], n=0)
        cov4 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        assert_equal(cov3, cov4)

        cov5 = multivariate_hypergeom.cov(m=np.array([], dtype=int), n=0)
        cov6 = np.array([], dtype=np.float64).reshape(0, 0)
        assert_allclose(cov5, cov6, rtol=1e-17)
        assert_(cov5.shape == (0, 0))

    def test_frozen(self):
        # The frozen distribution should agree with the regular one
        np.random.seed(1234)
        n = 12
        m = [7, 9, 11, 13]
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10],
             [1, 1, 1, 9], [1, 1, 2, 8]]
        x = np.asarray(x, dtype=int)
        mhg_frozen = multivariate_hypergeom(m, n)
        assert_allclose(mhg_frozen.pmf(x),
                        multivariate_hypergeom.pmf(x, m, n))
        assert_allclose(mhg_frozen.logpmf(x),
                        multivariate_hypergeom.logpmf(x, m, n))
        assert_allclose(mhg_frozen.var(), multivariate_hypergeom.var(m, n))
        assert_allclose(mhg_frozen.cov(), multivariate_hypergeom.cov(m, n))

    def test_invalid_params(self):
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, 10, 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, [10], 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, [5, 4], [10], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5.5, 4.5],
                      [10, 15], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4],
                      [10.5, 15.5], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4],
                      [10, 15], 5.5)


class TestRandomTable:
    def get_rng(self):
        return np.random.default_rng(628174795866951638)

    def test_process_parameters(self):
        message = "`row` must be one-dimensional"
        with pytest.raises(ValueError, match=message):
            random_table([[1, 2]], [1, 2])

        message = "`col` must be one-dimensional"
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [[1, 2]])

        message = "each element of `row` must be non-negative"
        with pytest.raises(ValueError, match=message):
            random_table([1, -1], [1, 2])

        message = "each element of `col` must be non-negative"
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, -2])

        message = "sums over `row` and `col` must be equal"
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1, 0])

        message = "each element of `row` must be an integer"
        with pytest.raises(ValueError, match=message):
            random_table([2.1, 2.1], [1, 1, 2])

        message = "each element of `col` must be an integer"
        with pytest.raises(ValueError, match=message):
            random_table([1, 2], [1.1, 1.1, 1])

        row = [1, 3]
        col = [2, 1, 1]
        r, c, n = random_table._process_parameters([1, 3], [2, 1, 1])
        assert_equal(row, r)
        assert_equal(col, c)
        assert n == np.sum(row)

    @pytest.mark.parametrize("scale,method",
                             ((1, "boyett"), (100, "patefield")))
    def test_process_rvs_method_on_None(self, scale, method):
        row = np.array([1, 3]) * scale
        col = np.array([2, 1, 1]) * scale

        ct = random_table
        expected = ct.rvs(row, col, method=method, random_state=1)
        got = ct.rvs(row, col, method=None, random_state=1)

        assert_equal(expected, got)

    def test_process_rvs_method_bad_argument(self):
        row = [1, 3]
        col = [2, 1, 1]

        # order of items in set is random, so cannot check that
        message = "'foo' not recognized, must be one of"
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, method="foo")

    @pytest.mark.parametrize('frozen', (True, False))
    @pytest.mark.parametrize('log', (True, False))
    def test_pmf_logpmf(self, frozen, log):
        # The pmf is tested through random sample generation
        # with Boyett's algorithm, whose implementation is simple
        # enough to verify manually for correctness.
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000,
                               method="boyett", random_state=rng)

        obj = random_table(row, col) if frozen else random_table
        method = getattr(obj, "logpmf" if log else "pmf")
        if not frozen:
            original_method = method

            def method(x):
                return original_method(x, row, col)
        pmf = (lambda x: np.exp(method(x))) if log else method

        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)

        # rough accuracy check
        p = pmf(unique_rvs)
        assert_allclose(p * len(rvs), counts, rtol=0.1)

        # accept any iterable
        p2 = pmf(list(unique_rvs[0]))
        assert_equal(p2, p[0])

        # accept high-dimensional input and 2d input
        rvs_nd = rvs.reshape((10, 100) + rvs.shape[1:])
        p = pmf(rvs_nd)
        assert p.shape == (10, 100)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                pij = p[i, j]
                rvij = rvs_nd[i, j]
                qij = pmf(rvij)
                assert_equal(pij, qij)

        # probability is zero if column marginal does not match
        x = [[0, 1, 1], [2, 1, 3]]
        assert_equal(np.sum(x, axis=-1), row)
        p = pmf(x)
        assert p == 0

        # probability is zero if row marginal does not match
        x = [[0, 1, 2], [1, 2, 2]]
        assert_equal(np.sum(x, axis=-2), col)
        p = pmf(x)
        assert p == 0

        # response to invalid inputs
        message = "`x` must be at least two-dimensional"
        with pytest.raises(ValueError, match=message):
            pmf([1])

        message = "`x` must contain only integral values"
        with pytest.raises(ValueError, match=message):
            pmf([[1.1]])

        message = "`x` must contain only integral values"
        with pytest.raises(ValueError, match=message):
            pmf([[np.nan]])

        message = "`x` must contain only non-negative values"
        with pytest.raises(ValueError, match=message):
            pmf([[-1]])

        message = "shape of `x` must agree with `row`"
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2, 3]])

        message = "shape of `x` must agree with `col`"
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2],
                 [3, 4]])

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_mean(self, method):
        # test if `rvs` is unbiased and large sample size converges
        # to the true mean.
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs = random_table.rvs(row, col, size=1000, method=method,
                               random_state=rng)
        mean = random_table.mean(row, col)
        assert_equal(np.sum(mean), np.sum(row))
        assert_allclose(rvs.mean(0), mean, atol=0.05)
        assert_equal(rvs.sum(axis=-1), np.broadcast_to(row, (1000, 2)))
        assert_equal(rvs.sum(axis=-2), np.broadcast_to(col, (1000, 3)))

    def test_rvs_cov(self):
        # test if `rvs` generated with patefield and boyett algorithms
        # produce approximately the same covariance matrix
        rng = self.get_rng()
        row = [2, 6]
        col = [1, 3, 4]
        rvs1 = random_table.rvs(row, col, size=10000, method="boyett",
                                random_state=rng)
        rvs2 = random_table.rvs(row, col, size=10000, method="patefield",
                                random_state=rng)
        cov1 = np.var(rvs1, axis=0)
        cov2 = np.var(rvs2, axis=0)
        assert_allclose(cov1, cov2, atol=0.02)

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_size(self, method):
        row = [2, 6]
        col = [1, 3, 4]

        # test size `None`
        rv = random_table.rvs(row, col, method=method,
                              random_state=self.get_rng())
        assert rv.shape == (2, 3)

        # test size 1
        rv2 = random_table.rvs(row, col, size=1, method=method,
                               random_state=self.get_rng())
        assert rv2.shape == (1, 2, 3)
        assert_equal(rv, rv2[0])

        # test size 0
        rv3 = random_table.rvs(row, col, size=0, method=method,
                               random_state=self.get_rng())
        assert rv3.shape == (0, 2, 3)

        # test other valid size
        rv4 = random_table.rvs(row, col, size=20, method=method,
                               random_state=self.get_rng())
        assert rv4.shape == (20, 2, 3)

        rv5 = random_table.rvs(row, col, size=(4, 5), method=method,
                               random_state=self.get_rng())
        assert rv5.shape == (4, 5, 2, 3)

        assert_allclose(rv5.reshape(20, 2, 3), rv4, rtol=1e-15)

        # test invalid size
        message = "`size` must be a non-negative integer or `None`"
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=-1, method=method,
                             random_state=self.get_rng())

        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=np.nan, method=method,
                             random_state=self.get_rng())

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_method(self, method):
        # This test assumes that pmf is correct and checks that random samples
        # follow this probability distribution. This seems like a circular
        # argument, since pmf is checked in test_pmf_logpmf with random samples
        # generated with the rvs method. This test is not redundant, because
        # test_pmf_logpmf intentionally uses rvs generation with Boyett only,
        # but here we test both Boyett and Patefield.
        row = [2, 6]
        col = [1, 3, 4]

        ct = random_table
        rvs = ct.rvs(row, col, size=100000, method=method,
                     random_state=self.get_rng())

        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)

        # generated frequencies should match expected frequencies
        p = ct.pmf(unique_rvs, row, col)
        assert_allclose(p * len(rvs), counts, rtol=0.02)

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_with_zeros_in_col_row(self, method):
        row = [0, 1, 0]
        col = [1, 0, 0, 0]
        d = random_table(row, col)
        rv = d.rvs(1000, method=method, random_state=self.get_rng())
        expected = np.zeros((1000, len(row), len(col)))
        expected[...] = [[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0]]
        assert_equal(rv, expected)

    @pytest.mark.parametrize("method", (None, "boyett", "patefield"))
    @pytest.mark.parametrize("col", ([], [0]))
    @pytest.mark.parametrize("row", ([], [0]))
    def test_rvs_with_edge_cases(self, method, row, col):
        d = random_table(row, col)
        rv = d.rvs(10, method=method, random_state=self.get_rng())
        expected = np.zeros((10, len(row), len(col)))
        assert_equal(rv, expected)

    @pytest.mark.parametrize('v', (1, 2))
    def test_rvs_rcont(self, v):
        # This test checks the internal low-level interface.
        # It is implicitly also checked by the other test_rvs* calls.
        import scipy.stats._rcont as _rcont

        row = np.array([1, 3], dtype=np.int64)
        col = np.array([2, 1, 1], dtype=np.int64)

        rvs = getattr(_rcont, f"rvs_rcont{v}")

        ntot = np.sum(row)
        result = rvs(row, col, ntot, 1, self.get_rng())

        assert result.shape == (1, len(row), len(col))
        assert np.sum(result) == ntot

    def test_frozen(self):
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())

        sample = d.rvs()

        expected = random_table.mean(row, col)
        assert_equal(expected, d.mean())

        expected = random_table.pmf(sample, row, col)
        assert_equal(expected, d.pmf(sample))

        expected = random_table.logpmf(sample, row, col)
        assert_equal(expected, d.logpmf(sample))

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_frozen(self, method):
        row = [2, 6]
        col = [1, 3, 4]
        d = random_table(row, col, seed=self.get_rng())

        expected = random_table.rvs(row, col, size=10, method=method,
                                    random_state=self.get_rng())
        got = d.rvs(size=10, method=method)
        assert_equal(expected, got)


def check_pickling(distfn, args):
    # check that a distribution instance pickles and unpickles
    # pay special attention to the random_state property

    # save the random_state (restore later)
    rndm = distfn.random_state

    distfn.random_state = 1234
    distfn.rvs(*args, size=8)
    s = pickle.dumps(distfn)
    r0 = distfn.rvs(*args, size=8)

    unpickled = pickle.loads(s)
    r1 = unpickled.rvs(*args, size=8)
    assert_equal(r0, r1)

    # restore the random_state
    distfn.random_state = rndm


def test_random_state_property():
    scale = np.eye(3)
    scale[0, 1] = 0.5
    scale[1, 0] = 0.5
    dists = [
        [multivariate_normal, ()],
        [dirichlet, (np.array([1.]), )],
        [wishart, (10, scale)],
        [invwishart, (10, scale)],
        [multinomial, (5, [0.5, 0.4, 0.1])],
        [ortho_group, (2,)],
        [special_ortho_group, (2,)]
    ]
    for distfn, args in dists:
        check_random_state_property(distfn, args)
        check_pickling(distfn, args)


class TestVonMises_Fisher:
    @pytest.mark.parametrize("dim", [2, 3, 4, 6])
    @pytest.mark.parametrize("size", [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        # test that samples have correct shape and norm 1
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim, ), 1/np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, 1, seed=rng)
        samples = vmf_dist.rvs(size)
        mean, cov = np.zeros(dim), np.eye(dim)
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.)

    @pytest.mark.parametrize("dim", [5, 8])
    @pytest.mark.parametrize("kappa", [1e15, 1e20, 1e30])
    def test_sampling_high_concentration(self, dim, kappa):
        # test that no warnings are encountered for high values
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim, ), 1/np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa, seed=rng)
        vmf_dist.rvs(10)

    def test_two_dimensional_mu(self):
        mu = np.ones((2, 2))
        msg = "'mu' must have one-dimensional shape."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_wrong_norm_mu(self):
        mu = np.ones((2, ))
        msg = "'mu' must be a unit vector of norm 1."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_one_entry_mu(self):
        mu = np.ones((1, ))
        msg = "'mu' must have at least two entries."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    @pytest.mark.parametrize("kappa", [-1, (5, 3)])
    def test_kappa_validation(self, kappa):
        msg = "'kappa' must be a positive scalar."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)

    @pytest.mark.parametrize("kappa", [0, 0.])
    def test_kappa_zero(self, kappa):
        msg = ("For 'kappa=0' the von Mises-Fisher distribution "
               "becomes the uniform distribution on the sphere "
               "surface. Consider using 'scipy.stats.uniform_direction' "
               "instead.")
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)


    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_invalid_shapes_pdf_logpdf(self, method):
        x = np.array([1., 0., 0])
        msg = ("The dimensionality of the last axis of 'x' must "
               "match the dimensionality of the von Mises Fisher "
               "distribution.")
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_unnormalized_input(self, method):
        x = np.array([0.5, 0.])
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    # Expected values of the vonmises-fisher logPDF were computed via mpmath
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def logpdf_mpmath(x, mu, kappa):
    #     dim = mu.size
    #     halfdim = mp.mpf(0.5 * dim)
    #     kappa = mp.mpf(kappa)
    #     const = (kappa**(halfdim - mp.one)/((2*mp.pi)**halfdim * \
    #              mp.besseli(halfdim -mp.one, kappa)))
    #     return float(const * mp.exp(kappa*mp.fdot(x, mu)))

    @pytest.mark.parametrize('x, mu, kappa, reference',
                             [(np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               1e-4, 0.0795854295583605),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               1e-4, 0.07957747141331854),
                              (np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               100, 15.915494309189533),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               100, 5.920684802611232e-43),
                              (np.array([1., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.]),
                               2000, 5.930499050746588e-07),
                              (np.array([1., 0., 0]), np.array([1., 0., 0.]),
                               2000, 318.3098861837907),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([1., 0., 0., 0., 0.]),
                               2000, 101371.86957712633),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.,
                                         0, 0.]),
                               2000, 0.00018886808182653578),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.8), np.sqrt(0.2), 0.,
                                         0, 0.]),
                               2000, 2.0255393314603194e-87)])
    def test_pdf_accuracy(self, x, mu, kappa, reference):
        pdf = vonmises_fisher(mu, kappa).pdf(x)
        assert_allclose(pdf, reference, rtol=1e-13)

    # Expected values of the vonmises-fisher logPDF were computed via mpmath
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def logpdf_mpmath(x, mu, kappa):
    #     dim = mu.size
    #     halfdim = mp.mpf(0.5 * dim)
    #     kappa = mp.mpf(kappa)
    #     two = mp.mpf(2.)
    #     const = (kappa**(halfdim - mp.one)/((two*mp.pi)**halfdim * \
    #              mp.besseli(halfdim - mp.one, kappa)))
    #     return float(mp.log(const * mp.exp(kappa*mp.fdot(x, mu))))

    @pytest.mark.parametrize('x, mu, kappa, reference',
                             [(np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               1e-4, -2.5309242486359573),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               1e-4, -2.5310242486359575),
                              (np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               100, 2.767293119578746),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               100, -97.23270688042125),
                              (np.array([1., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.]),
                               2000, -14.337987284534103),
                              (np.array([1., 0., 0]), np.array([1., 0., 0.]),
                               2000, 5.763025393132737),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([1., 0., 0., 0., 0.]),
                               2000, 11.526550911307156),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.,
                                         0, 0.]),
                               2000, -8.574461766359684),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.8), np.sqrt(0.2), 0.,
                                         0, 0.]),
                               2000, -199.61906708886113)])
    def test_logpdf_accuracy(self, x, mu, kappa, reference):
        logpdf = vonmises_fisher(mu, kappa).logpdf(x)
        assert_allclose(logpdf, reference, rtol=1e-14)

    # Expected values of the vonmises-fisher entropy were computed via mpmath
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def entropy_mpmath(dim, kappa):
    #     mu = np.full((dim, ), 1/np.sqrt(dim))
    #     kappa = mp.mpf(kappa)
    #     halfdim = mp.mpf(0.5 * dim)
    #     logconstant = (mp.log(kappa**(halfdim - mp.one)
    #                    /((2*mp.pi)**halfdim
    #                    * mp.besseli(halfdim -mp.one, kappa)))
    #     return float(-logconstant - kappa * mp.besseli(halfdim, kappa)/
    #             mp.besseli(halfdim -1, kappa))

    @pytest.mark.parametrize('dim, kappa, reference',
                             [(3, 1e-4, 2.531024245302624),
                              (3, 100, -1.7672931195787458),
                              (5, 5000, -11.359032310024453),
                              (8, 1, 3.4189526482545527)])
    def test_entropy_accuracy(self, dim, kappa, reference):
        mu = np.full((dim, ), 1/np.sqrt(dim))
        entropy = vonmises_fisher(mu, kappa).entropy()
        assert_allclose(entropy, reference, rtol=2e-14)

    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_broadcasting(self, method):
        # test that pdf and logpdf values are correctly broadcasted
        testshape = (2, 2)
        rng = np.random.default_rng(2777937887058094419)
        x = uniform_direction(3).rvs(testshape, random_state=rng)
        mu = np.full((3, ), 1/np.sqrt(3))
        kappa = 5
        result_all = method(x, mu, kappa)
        assert result_all.shape == testshape
        for i in range(testshape[0]):
            for j in range(testshape[1]):
                current_val = method(x[i, j, :], mu, kappa)
                assert_allclose(current_val, result_all[i, j], rtol=1e-15)

    def test_vs_vonmises_2d(self):
        # test that in 2D, von Mises-Fisher yields the same results
        # as the von Mises distribution
        rng = np.random.default_rng(2777937887058094419)
        mu = np.array([0, 1])
        mu_angle = np.arctan2(mu[1], mu[0])
        kappa = 20
        vmf = vonmises_fisher(mu, kappa)
        vonmises_dist = vonmises(loc=mu_angle, kappa=kappa)
        vectors = uniform_direction(2).rvs(10, random_state=rng)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        assert_allclose(vonmises_dist.entropy(), vmf.entropy())
        assert_allclose(vonmises_dist.pdf(angles), vmf.pdf(vectors))
        assert_allclose(vonmises_dist.logpdf(angles), vmf.logpdf(vectors))

    @pytest.mark.parametrize("dim", [2, 3, 6])
    @pytest.mark.parametrize("kappa, mu_tol, kappa_tol",
                             [(1, 5e-2, 5e-2),
                              (10, 1e-2, 1e-2),
                              (100, 5e-3, 2e-2),
                              (1000, 1e-3, 2e-2)])
    def test_fit_accuracy(self, dim, kappa, mu_tol, kappa_tol):
        mu = np.full((dim, ), 1/np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa)
        rng = np.random.default_rng(2777937887058094419)
        n_samples = 10000
        samples = vmf_dist.rvs(n_samples, random_state=rng)
        mu_fit, kappa_fit = vonmises_fisher.fit(samples)
        angular_error = np.arccos(mu.dot(mu_fit))
        assert_allclose(angular_error, 0., atol=mu_tol, rtol=0)
        assert_allclose(kappa, kappa_fit, rtol=kappa_tol)

    def test_fit_error_one_dimensional_data(self):
        x = np.zeros((3, ))
        msg = "'x' must be two dimensional."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_fit_error_unnormalized_data(self):
        x = np.ones((3, 3))
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_frozen_distribution(self):
        mu = np.array([0, 0, 1])
        kappa = 5
        frozen = vonmises_fisher(mu, kappa)
        frozen_seed = vonmises_fisher(mu, kappa, seed=514)

        rvs1 = frozen.rvs(random_state=514)
        rvs2 = vonmises_fisher.rvs(mu, kappa, random_state=514)
        rvs3 = frozen_seed.rvs()

        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)


class TestDirichletMultinomial:
    @classmethod
    def get_params(self, m):
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, size=2)
        x = rng.integers(1, 20, size=(m, 2))
        n = x.sum(axis=-1)
        return rng, m, alpha, n, x

    def test_frozen(self):
        rng = np.random.default_rng(28469824356873456)

        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)

        d = dirichlet_multinomial(alpha, n)
        assert_equal(d.logpmf(x), dirichlet_multinomial.logpmf(x, alpha, n))
        assert_equal(d.pmf(x), dirichlet_multinomial.pmf(x, alpha, n))
        assert_equal(d.mean(), dirichlet_multinomial.mean(alpha, n))
        assert_equal(d.var(), dirichlet_multinomial.var(alpha, n))
        assert_equal(d.cov(), dirichlet_multinomial.cov(alpha, n))

    def test_pmf_logpmf_against_R(self):
        # # Compare PMF against R's extraDistr ddirmnon
        # # library(extraDistr)
        # # options(digits=16)
        # ddirmnom(c(1, 2, 3), 6, c(3, 4, 5))
        x = np.array([1, 2, 3])
        n = np.sum(x)
        alpha = np.array([3, 4, 5])
        res = dirichlet_multinomial.pmf(x, alpha, n)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 0.08484162895927638
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))
        assert res.shape == logres.shape == ()

        # library(extraDistr)
        # options(digits=16)
        # ddirmnom(c(4, 3, 2, 0, 2, 3, 5, 7, 4, 7), 37,
        #          c(45.01025314, 21.98739582, 15.14851365, 80.21588671,
        #            52.84935481, 25.20905262, 53.85373737, 4.88568118,
        #            89.06440654, 20.11359466))
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        res = dirichlet_multinomial(alpha, n).pmf(x)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 3.65409306285992e-16
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))

    def test_pmf_logpmf_support(self):
        # when the sum of the category counts does not equal the number of
        # trials, the PMF is zero
        rng, m, alpha, n, x = self.get_params(1)
        n += 1
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x), 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x), -np.inf)

        rng, m, alpha, n, x = self.get_params(10)
        i = rng.random(size=10) > 0.5
        x[i] = np.round(x[i] * 2)  # sum of these x does not equal n
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x)[i], 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x)[i], -np.inf)
        assert np.all(dirichlet_multinomial(alpha, n).pmf(x)[~i] > 0)
        assert np.all(dirichlet_multinomial(alpha, n).logpmf(x)[~i] > -np.inf)

    def test_dimensionality_one(self):
        # if the dimensionality is one, there is only one possible outcome
        n = 6  # number of trials
        alpha = [10]  # concentration parameters
        x = np.asarray([n])  # counts
        dist = dirichlet_multinomial(alpha, n)

        assert_equal(dist.pmf(x), 1)
        assert_equal(dist.pmf(x+1), 0)
        assert_equal(dist.logpmf(x), 0)
        assert_equal(dist.logpmf(x+1), -np.inf)
        assert_equal(dist.mean(), n)
        assert_equal(dist.var(), 0)
        assert_equal(dist.cov(), 0)

    @pytest.mark.parametrize('method_name', ['pmf', 'logpmf'])
    def test_against_betabinom_pmf(self, method_name):
        rng, m, alpha, n, x = self.get_params(100)

        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)

        res = method(x)
        ref = ref_method(x.T[0])
        assert_allclose(res, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var'])
    def test_against_betabinom_moments(self, method_name):
        rng, m, alpha, n, x = self.get_params(100)

        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)

        res = method()[:, 0]
        ref = ref_method()
        assert_allclose(res, ref)

    def test_moments(self):
        message = 'Needs NumPy 1.22.0 for multinomial broadcasting'
        if Version(np.__version__) < Version("1.22.0"):
            pytest.skip(reason=message)

        rng = np.random.default_rng(28469824356873456)
        dim = 5
        n = rng.integers(1, 100)
        alpha = rng.random(size=dim) * 10
        dist = dirichlet_multinomial(alpha, n)

        # Generate a random sample from the distribution using NumPy
        m = 100000
        p = rng.dirichlet(alpha, size=m)
        x = rng.multinomial(n, p, size=m)

        assert_allclose(dist.mean(), np.mean(x, axis=0), rtol=5e-3)
        assert_allclose(dist.var(), np.var(x, axis=0), rtol=1e-2)
        assert dist.mean().shape == dist.var().shape == (dim,)

        cov = dist.cov()
        assert cov.shape == (dim, dim)
        assert_allclose(cov, np.cov(x.T), rtol=2e-2)
        assert_equal(np.diag(cov), dist.var())
        assert np.all(scipy.linalg.eigh(cov)[0] > 0)  # positive definite

    def test_input_validation(self):
        # valid inputs
        x0 = np.array([1, 2, 3])
        n0 = np.sum(x0)
        alpha0 = np.array([3, 4, 5])

        text = "`x` must contain only non-negative integers."
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, -1, 3], alpha0, n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, 2.1, 3], alpha0, n0)

        text = "`alpha` must contain only positive values."
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, 0, 4], n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, -1, 4], n0)

        text = "`n` must be a positive integer."
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 49.1)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 0)

        x = np.array([1, 2, 3, 4])
        alpha = np.array([3, 4, 5])
        text = "`x` and `alpha` must be broadcastable."
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x, alpha, x.sum())

    @pytest.mark.parametrize('method', ['pmf', 'logpmf'])
    def test_broadcasting_pmf(self, method):
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        x = np.array([[1, 2, 3], [2, 2, 3]]).reshape((2, 1, 1, 3))
        method = getattr(dirichlet_multinomial, method)
        res = method(x, alpha, n)
        assert res.shape == (2, 3, 4)
        for i in range(len(x)):
            for j in range(len(n)):
                for k in range(len(alpha)):
                    res_ijk = res[i, j, k]
                    ref = method(x[i].squeeze(), alpha[k].squeeze(), n[j].squeeze())
                    assert_allclose(res_ijk, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var', 'cov'])
    def test_broadcasting_moments(self, method_name):
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        method = getattr(dirichlet_multinomial, method_name)
        res = method(alpha, n)
        assert res.shape == (3, 4, 3) if method_name != 'cov' else (3, 4, 3, 3)
        for j in range(len(n)):
            for k in range(len(alpha)):
                res_ijk = res[j, k]
                ref = method(alpha[k].squeeze(), n[j].squeeze())
                assert_allclose(res_ijk, ref)
