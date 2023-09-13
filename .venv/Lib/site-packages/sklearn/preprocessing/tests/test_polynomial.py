import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
from sklearn.preprocessing._csr_polynomial_expansion import (
    _calc_expanded_nnz,
    _calc_total_nnz,
    _get_sizeof_LARGEST_INT_t,
)
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import parse_version, sp_version


@pytest.mark.parametrize("est", (PolynomialFeatures, SplineTransformer))
def test_polynomial_and_spline_array_order(est):
    """Test that output array has the given order."""
    X = np.arange(10).reshape(5, 2)

    def is_c_contiguous(a):
        return np.isfortran(a.T)

    assert is_c_contiguous(est().fit_transform(X))
    assert is_c_contiguous(est(order="C").fit_transform(X))
    assert np.isfortran(est(order="F").fit_transform(X))


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"knots": [[1]]}, r"Number of knots, knots.shape\[0\], must be >= 2."),
        ({"knots": [[1, 1], [2, 2]]}, r"knots.shape\[1\] == n_features is violated"),
        ({"knots": [[1], [0]]}, "knots must be sorted without duplicates."),
    ],
)
def test_spline_transformer_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in SplineTransformer."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        SplineTransformer(**params).fit(X)


@pytest.mark.parametrize("extrapolation", ["continue", "periodic"])
def test_spline_transformer_integer_knots(extrapolation):
    """Test that SplineTransformer accepts integer value knot positions."""
    X = np.arange(20).reshape(10, 2)
    knots = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
    _ = SplineTransformer(
        degree=3, knots=knots, extrapolation=extrapolation
    ).fit_transform(X)


def test_spline_transformer_feature_names():
    """Test that SplineTransformer generates correct features name."""
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=True).fit(X)
    feature_names = splt.get_feature_names_out()
    assert_array_equal(
        feature_names,
        [
            "x0_sp_0",
            "x0_sp_1",
            "x0_sp_2",
            "x0_sp_3",
            "x0_sp_4",
            "x1_sp_0",
            "x1_sp_1",
            "x1_sp_2",
            "x1_sp_3",
            "x1_sp_4",
        ],
    )

    splt = SplineTransformer(n_knots=3, degree=3, include_bias=False).fit(X)
    feature_names = splt.get_feature_names_out(["a", "b"])
    assert_array_equal(
        feature_names,
        [
            "a_sp_0",
            "a_sp_1",
            "a_sp_2",
            "a_sp_3",
            "b_sp_0",
            "b_sp_1",
            "b_sp_2",
            "b_sp_3",
        ],
    )


@pytest.mark.parametrize(
    "extrapolation",
    ["constant", "linear", "continue", "periodic"],
)
@pytest.mark.parametrize("degree", [2, 3])
def test_split_transform_feature_names_extrapolation_degree(extrapolation, degree):
    """Test feature names are correct for different extrapolations and degree.

    Non-regression test for gh-25292.
    """
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(degree=degree, extrapolation=extrapolation).fit(X)
    feature_names = splt.get_feature_names_out(["a", "b"])
    assert len(feature_names) == splt.n_features_out_

    X_trans = splt.transform(X)
    assert X_trans.shape[1] == len(feature_names)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("n_knots", range(3, 5))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize("extrapolation", ["constant", "periodic"])
def test_spline_transformer_unity_decomposition(degree, n_knots, knots, extrapolation):
    """Test that B-splines are indeed a decomposition of unity.

    Splines basis functions must sum up to 1 per row, if we stay in between boundaries.
    """
    X = np.linspace(0, 1, 100)[:, None]
    # make the boundaries 0 and 1 part of X_train, for sure.
    X_train = np.r_[[[0]], X[::2, :], [[1]]]
    X_test = X[1::2, :]

    if extrapolation == "periodic":
        n_knots = n_knots + degree  # periodic splines require degree < n_knots

    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots=knots,
        include_bias=True,
        extrapolation=extrapolation,
    )
    splt.fit(X_train)
    for X in [X_train, X_test]:
        assert_allclose(np.sum(splt.transform(X), axis=1), 1)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_linear_regression(bias, intercept):
    """Test that B-splines fit a sinusodial curve pretty well."""
    X = np.linspace(0, 10, 100)[:, None]
    y = np.sin(X[:, 0]) + 2  # +2 to avoid the value 0 in assert_allclose
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=15,
                    degree=3,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict(X), y, rtol=1e-3)


@pytest.mark.parametrize(
    ["knots", "n_knots", "sample_weight", "expected_knots"],
    [
        ("uniform", 3, None, np.array([[0, 2], [3, 8], [6, 14]])),
        (
            "uniform",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [4, 8], [6, 14]]),
        ),
        ("uniform", 4, None, np.array([[0, 2], [2, 6], [4, 10], [6, 14]])),
        ("quantile", 3, None, np.array([[0, 2], [3, 3], [6, 14]])),
        (
            "quantile",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [5, 8], [6, 14]]),
        ),
    ],
)
def test_spline_transformer_get_base_knot_positions(
    knots, n_knots, sample_weight, expected_knots
):
    """Check the behaviour to find knot positions with and without sample_weight."""
    X = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]])
    base_knots = SplineTransformer._get_base_knot_positions(
        X=X, knots=knots, n_knots=n_knots, sample_weight=sample_weight
    )
    assert_allclose(base_knots, expected_knots)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_periodic_linear_regression(bias, intercept):
    """Test that B-splines fit a periodic curve pretty well."""

    # "+ 3" to avoid the value 0 in assert_allclose
    def f(x):
        return np.sin(2 * np.pi * x) - np.sin(8 * np.pi * x) + 3

    X = np.linspace(0, 1, 101)[:, None]
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=20,
                    degree=3,
                    include_bias=bias,
                    extrapolation="periodic",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, f(X[:, 0]))

    # Generate larger array to check periodic extrapolation
    X_ = np.linspace(-1, 2, 301)[:, None]
    predictions = pipe.predict(X_)
    assert_allclose(predictions, f(X_[:, 0]), atol=0.01, rtol=0.01)
    assert_allclose(predictions[0:100], predictions[100:200], rtol=1e-3)


def test_spline_transformer_periodic_spline_backport():
    """Test that the backport of extrapolate="periodic" works correctly"""
    X = np.linspace(-2, 3.5, 10)[:, None]
    degree = 2

    # Use periodic extrapolation backport in SplineTransformer
    transformer = SplineTransformer(
        degree=degree, extrapolation="periodic", knots=[[-1.0], [0.0], [1.0]]
    )
    Xt = transformer.fit_transform(X)

    # Use periodic extrapolation in BSpline
    coef = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    spl = BSpline(np.arange(-3, 4), coef, degree, "periodic")
    Xspl = spl(X[:, 0])
    assert_allclose(Xt, Xspl)


def test_spline_transformer_periodic_splines_periodicity():
    """Test if shifted knots result in the same transformation up to permutation."""
    X = np.linspace(0, 10, 101)[:, None]

    transformer_1 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )

    transformer_2 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[1.0], [3.0], [4.0], [5.0], [8.0], [9.0]],
    )

    Xt_1 = transformer_1.fit_transform(X)
    Xt_2 = transformer_2.fit_transform(X)

    assert_allclose(Xt_1, Xt_2[:, [4, 0, 1, 2, 3]])


@pytest.mark.parametrize("degree", [3, 5])
def test_spline_transformer_periodic_splines_smoothness(degree):
    """Test that spline transformation is smooth at first / last knot."""
    X = np.linspace(-2, 10, 10_000)[:, None]

    transformer = SplineTransformer(
        degree=degree,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )
    Xt = transformer.fit_transform(X)

    delta = (X.max() - X.min()) / len(X)
    tol = 10 * delta

    dXt = Xt
    # We expect splines of degree `degree` to be (`degree`-1) times
    # continuously differentiable. I.e. for d = 0, ..., `degree` - 1 the d-th
    # derivative should be continuous. This is the case if the (d+1)-th
    # numerical derivative is reasonably small (smaller than `tol` in absolute
    # value). We thus compute d-th numeric derivatives for d = 1, ..., `degree`
    # and compare them to `tol`.
    #
    # Note that the 0-th derivative is the function itself, such that we are
    # also checking its continuity.
    for d in range(1, degree + 1):
        # Check continuity of the (d-1)-th derivative
        diff = np.diff(dXt, axis=0)
        assert np.abs(diff).max() < tol
        # Compute d-th numeric derivative
        dXt = diff / delta

    # As degree `degree` splines are not `degree` times continuously
    # differentiable at the knots, the `degree + 1`-th numeric derivative
    # should have spikes at the knots.
    diff = np.diff(dXt, axis=0)
    assert np.abs(diff).max() > 1


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_transformer_extrapolation(bias, intercept, degree):
    """Test that B-spline extrapolation works correctly."""
    # we use a straight line for that
    X = np.linspace(-1, 1, 100)[:, None]
    y = X.squeeze()

    # 'constant'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-1, 1])

    # 'linear'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="linear",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-10, 5])

    # 'error'
    splt = SplineTransformer(
        n_knots=4, degree=degree, include_bias=bias, extrapolation="error"
    )
    splt.fit(X)
    msg = "X contains values beyond the limits of the knots"
    with pytest.raises(ValueError, match=msg):
        splt.transform([[-10]])
    with pytest.raises(ValueError, match=msg):
        splt.transform([[5]])


def test_spline_transformer_kbindiscretizer():
    """Test that a B-spline of degree=0 is equivalent to KBinsDiscretizer."""
    rng = np.random.RandomState(97531)
    X = rng.randn(200).reshape(200, 1)
    n_bins = 5
    n_knots = n_bins + 1

    splt = SplineTransformer(
        n_knots=n_knots, degree=0, knots="quantile", include_bias=True
    )
    splines = splt.fit_transform(X)

    kbd = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")
    kbins = kbd.fit_transform(X)

    # Though they should be exactly equal, we test approximately with high
    # accuracy.
    assert_allclose(splines, kbins, rtol=1e-13)


@pytest.mark.skipif(
    sp_version < parse_version("1.8.0"),
    reason="The option `sparse_output` is available as of scipy 1.8.0",
)
@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("include_bias", [False, True])
def test_spline_transformer_sparse_output(
    degree, knots, extrapolation, include_bias, global_random_seed
):
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(200).reshape(40, 5)

    splt_dense = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=False,
    )
    splt_sparse = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=True,
    )

    splt_dense.fit(X)
    splt_sparse.fit(X)

    assert sparse.isspmatrix_csr(splt_sparse.transform(X))
    assert_allclose(splt_dense.transform(X), splt_sparse.transform(X).toarray())

    # extrapolation regime
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    X_extra = np.r_[
        np.linspace(X_min - 5, X_min, 10), np.linspace(X_max, X_max + 5, 10)
    ]
    if extrapolation == "error":
        msg = "X contains values beyond the limits of the knots"
        with pytest.raises(ValueError, match=msg):
            splt_dense.transform(X_extra)
        msg = "Out of bounds"
        with pytest.raises(ValueError, match=msg):
            splt_sparse.transform(X_extra)
    else:
        assert_allclose(
            splt_dense.transform(X_extra), splt_sparse.transform(X_extra).toarray()
        )


@pytest.mark.skipif(
    sp_version >= parse_version("1.8.0"),
    reason="The option `sparse_output` is available as of scipy 1.8.0",
)
def test_spline_transformer_sparse_output_raise_error_for_old_scipy():
    """Test that SplineTransformer with sparse=True raises for scipy<1.8.0."""
    X = [[1], [2]]
    with pytest.raises(ValueError, match="scipy>=1.8.0"):
        SplineTransformer(sparse_output=True).fit(X)


@pytest.mark.parametrize("n_knots", [5, 10])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("sparse_output", [False, True])
def test_spline_transformer_n_features_out(
    n_knots, include_bias, degree, extrapolation, sparse_output
):
    """Test that transform results in n_features_out_ features."""
    if sparse_output and sp_version < parse_version("1.8.0"):
        pytest.skip("The option `sparse_output` is available as of scipy 1.8.0")

    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        extrapolation=extrapolation,
        sparse_output=sparse_output,
    )
    X = np.linspace(0, 1, 10)[:, None]
    splt.fit(X)

    assert splt.transform(X).shape[1] == splt.n_features_out_


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"degree": (-1, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (0, 1.5)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (3, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (1, 2, 3)}, r"int or tuple \(min_degree, max_degree\)"),
    ],
)
def test_polynomial_features_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in PolynomialFeatures."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        PolynomialFeatures(**params).fit(X)


@pytest.fixture()
def single_feature_degree3():
    X = np.arange(6)[:, np.newaxis]
    P = np.hstack([np.ones_like(X), X, X**2, X**3])
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1]),
        (3, False, True, [1]),
        ((2, 3), True, False, [0, 2, 3]),
        ((2, 3), False, False, [2, 3]),
        ((2, 3), True, True, [0]),
        ((2, 3), False, True, []),
    ],
)
@pytest.mark.parametrize(
    "sparse_X",
    [False, sparse.csr_matrix, sparse.csc_matrix],
)
def test_polynomial_features_one_feature(
    single_feature_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    sparse_X,
):
    """Test PolynomialFeatures on single feature up to degree 3."""
    X, P = single_feature_degree3
    if sparse_X:
        X = sparse_X(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if sparse_X:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


@pytest.fixture()
def two_features_degree3():
    X = np.arange(6).reshape((3, 2))
    x1 = X[:, :1]
    x2 = X[:, 1:]
    P = np.hstack(
        [
            x1**0 * x2**0,  # 0
            x1**1 * x2**0,  # 1
            x1**0 * x2**1,  # 2
            x1**2 * x2**0,  # 3
            x1**1 * x2**1,  # 4
            x1**0 * x2**2,  # 5
            x1**3 * x2**0,  # 6
            x1**2 * x2**1,  # 7
            x1**1 * x2**2,  # 8
            x1**0 * x2**3,  # 9
        ]
    )
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (2, True, False, slice(0, 6)),
        (2, False, False, slice(1, 6)),
        (2, True, True, [0, 1, 2, 4]),
        (2, False, True, [1, 2, 4]),
        ((2, 2), True, False, [0, 3, 4, 5]),
        ((2, 2), False, False, [3, 4, 5]),
        ((2, 2), True, True, [0, 4]),
        ((2, 2), False, True, [4]),
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1, 2, 4]),
        (3, False, True, [1, 2, 4]),
        ((2, 3), True, False, [0, 3, 4, 5, 6, 7, 8, 9]),
        ((2, 3), False, False, slice(3, None)),
        ((2, 3), True, True, [0, 4]),
        ((2, 3), False, True, [4]),
        ((3, 3), True, False, [0, 6, 7, 8, 9]),
        ((3, 3), False, False, [6, 7, 8, 9]),
        ((3, 3), True, True, [0]),
        ((3, 3), False, True, []),  # would need 3 input features
    ],
)
@pytest.mark.parametrize(
    "sparse_X",
    [False, sparse.csr_matrix, sparse.csc_matrix],
)
def test_polynomial_features_two_features(
    two_features_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    sparse_X,
):
    """Test PolynomialFeatures on 2 features up to degree 3."""
    X, P = two_features_degree3
    if sparse_X:
        X = sparse_X(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if sparse_X:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


def test_polynomial_feature_names():
    X = np.arange(30).reshape(10, 3)
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out()
    assert_array_equal(
        ["1", "x0", "x1", "x2", "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2"],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(
        [
            "a",
            "b",
            "c",
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=(2, 3), include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(
        [
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    assert_array_equal(["1", "a b c"], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]

    # test some unicode
    poly = PolynomialFeatures(degree=1, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out(["\u0001F40D", "\u262e", "\u05d0"])
    assert_array_equal(["1", "\u0001F40D", "\u262e", "\u05d0"], feature_names)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
        (4, False, False, np.float64),
        (4, False, True, np.float64),
    ],
)
def test_polynomial_features_csc_X(deg, include_bias, interaction_only, dtype):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csc = sparse.csc_matrix(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csc = est.fit_transform(X_csc.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert sparse.isspmatrix_csc(Xt_csc)
    assert Xt_csc.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csc.A, Xt_dense)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
def test_polynomial_features_csr_X(deg, include_bias, interaction_only, dtype):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csr = sparse.csr_matrix(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype, copy=False))

    assert sparse.isspmatrix_csr(Xt_csr)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize("n_features", [1, 4, 5])
@pytest.mark.parametrize(
    "min_degree, max_degree", [(0, 1), (0, 2), (1, 3), (0, 4), (3, 4)]
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_num_combinations(
    n_features,
    min_degree,
    max_degree,
    interaction_only,
    include_bias,
):
    """
    Test that n_output_features_ is calculated correctly.
    """
    x = sparse.csr_matrix(([1], ([0], [n_features - 1])))
    est = PolynomialFeatures(
        degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    est.fit(x)
    num_combos = est.n_output_features_

    combos = PolynomialFeatures._combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    assert num_combos == sum([1 for _ in combos])


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
def test_polynomial_features_csr_X_floats(deg, include_bias, interaction_only, dtype):
    X_csr = sparse_random(1000, 10, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert sparse.isspmatrix_csr(Xt_csr)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize(
    ["zero_row_index", "deg", "interaction_only"],
    [
        (0, 2, True),
        (1, 2, True),
        (2, 2, True),
        (0, 3, True),
        (1, 3, True),
        (2, 3, True),
        (0, 2, False),
        (1, 2, False),
        (2, 2, False),
        (0, 3, False),
        (1, 3, False),
        (2, 3, False),
    ],
)
def test_polynomial_features_csr_X_zero_row(zero_row_index, deg, interaction_only):
    X_csr = sparse_random(3, 10, 1.0, random_state=0).tocsr()
    X_csr[zero_row_index, :] = 0.0
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, include_bias=False, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.isspmatrix_csr(Xt_csr)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


# This degree should always be one more than the highest degree supported by
# _csr_expansion.
@pytest.mark.parametrize(
    ["include_bias", "interaction_only"],
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_polynomial_features_csr_X_degree_4(include_bias, interaction_only):
    X_csr = sparse_random(1000, 10, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(
        4, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.isspmatrix_csr(X_csr)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize(
    ["deg", "dim", "interaction_only"],
    [
        (2, 1, True),
        (2, 2, True),
        (3, 1, True),
        (3, 2, True),
        (3, 3, True),
        (2, 1, False),
        (2, 2, False),
        (3, 1, False),
        (3, 2, False),
        (3, 3, False),
    ],
)
def test_polynomial_features_csr_X_dim_edges(deg, dim, interaction_only):
    X_csr = sparse_random(1000, dim, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert sparse.isspmatrix_csr(Xt_csr)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_csr_polynomial_expansion_index_overflow_non_regression(
    interaction_only, include_bias
):
    """Check the automatic index dtype promotion to `np.int64` when needed.

    This ensures that sufficiently large input configurations get
    properly promoted to use `np.int64` for index and indptr representation
    while preserving data integrity. Non-regression test for gh-16803.

    Note that this is only possible for Python runtimes with a 64 bit address
    space. On 32 bit platforms, a `ValueError` is raised instead.
    """

    def degree_2_calc(d, i, j):
        if interaction_only:
            return d * i - (i**2 + 3 * i) // 2 - 1 + j
        else:
            return d * i - (i**2 + i) // 2 + j

    n_samples = 13
    n_features = 120001
    data_dtype = np.float32
    data = np.arange(1, 5, dtype=np.int64)
    row = np.array([n_samples - 2, n_samples - 2, n_samples - 1, n_samples - 1])
    # An int64 dtype is required to avoid overflow error on Windows within the
    # `degree_2_calc` function.
    col = np.array(
        [n_features - 2, n_features - 1, n_features - 2, n_features - 1], dtype=np.int64
    )
    X = sparse.csr_matrix(
        (data, (row, col)),
        shape=(n_samples, n_features),
        dtype=data_dtype,
    )
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=2
    )

    # Calculate the number of combinations a-priori, and if needed check for
    # the correct ValueError and terminate the test early.
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=2,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return
    X_trans = pf.fit_transform(X)
    row_nonzero, col_nonzero = X_trans.nonzero()
    n_degree_1_features_out = n_features + include_bias
    max_degree_2_idx = (
        degree_2_calc(n_features, col[int(not interaction_only)], col[1])
        + n_degree_1_features_out
    )

    # Account for bias of all samples except last one which will be handled
    # separately since there are distinct data values before it
    data_target = [1] * (n_samples - 2) if include_bias else []
    col_nonzero_target = [0] * (n_samples - 2) if include_bias else []

    for i in range(2):
        x = data[2 * i]
        y = data[2 * i + 1]
        x_idx = col[2 * i]
        y_idx = col[2 * i + 1]
        if include_bias:
            data_target.append(1)
            col_nonzero_target.append(0)
        data_target.extend([x, y])
        col_nonzero_target.extend(
            [x_idx + int(include_bias), y_idx + int(include_bias)]
        )
        if not interaction_only:
            data_target.extend([x * x, x * y, y * y])
            col_nonzero_target.extend(
                [
                    degree_2_calc(n_features, x_idx, x_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, y_idx, y_idx) + n_degree_1_features_out,
                ]
            )
        else:
            data_target.extend([x * y])
            col_nonzero_target.append(
                degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out
            )

    nnz_per_row = int(include_bias) + 3 + 2 * int(not interaction_only)

    assert pf.n_output_features_ == max_degree_2_idx + 1
    assert X_trans.dtype == data_dtype
    assert X_trans.shape == (n_samples, max_degree_2_idx + 1)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == np.int64
    # Ensure that dtype promotion was actually required:
    assert X_trans.indices.max() > np.iinfo(np.int32).max

    row_nonzero_target = list(range(n_samples - 2)) if include_bias else []
    row_nonzero_target.extend(
        [n_samples - 2] * nnz_per_row + [n_samples - 1] * nnz_per_row
    )

    assert_allclose(X_trans.data, data_target)
    assert_array_equal(row_nonzero, row_nonzero_target)
    assert_array_equal(col_nonzero, col_nonzero_target)


@pytest.mark.parametrize(
    "degree, n_features",
    [
        # Needs promotion to int64 when interaction_only=False
        (2, 65535),
        (3, 2344),
        # This guarantees that the intermediate operation when calculating
        # output columns would overflow a C-long, hence checks that python-
        # longs are being used.
        (2, int(np.sqrt(np.iinfo(np.int64).max) + 1)),
        (3, 65535),
        # This case tests the second clause of the overflow check which
        # takes into account the value of `n_features` itself.
        (2, int(np.sqrt(np.iinfo(np.int64).max))),
    ],
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_csr_polynomial_expansion_index_overflow(
    degree, n_features, interaction_only, include_bias
):
    """Tests known edge-cases to the dtype promotion strategy and custom
    Cython code, including a current bug in the upstream
    `scipy.sparse.hstack`.
    """
    data = [1.0]
    row = [0]
    col = [n_features - 1]

    # First degree index
    expected_indices = [
        n_features - 1 + int(include_bias),
    ]
    # Second degree index
    expected_indices.append(n_features * (n_features + 1) // 2 + expected_indices[0])
    # Third degree index
    expected_indices.append(
        n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1]
    )

    X = sparse.csr_matrix((data, (row, col)))
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=degree
    )

    # Calculate the number of combinations a-priori, and if needed check for
    # the correct ValueError and terminate the test early.
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=degree,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return

    # In SciPy < 1.8, a bug occurs when an intermediate matrix in
    # `to_stack` in `hstack` fits within int32 however would require int64 when
    # combined with all previous matrices in `to_stack`.
    if sp_version < parse_version("1.8.0"):
        has_bug = False
        max_int32 = np.iinfo(np.int32).max
        cumulative_size = n_features + include_bias
        for deg in range(2, degree + 1):
            max_indptr = _calc_total_nnz(X.indptr, interaction_only, deg)
            max_indices = _calc_expanded_nnz(n_features, interaction_only, deg) - 1
            cumulative_size += max_indices + 1
            needs_int64 = max(max_indices, max_indptr) > max_int32
            has_bug |= not needs_int64 and cumulative_size > max_int32
        if has_bug:
            msg = r"In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
            with pytest.raises(ValueError, match=msg):
                X_trans = pf.fit_transform(X)
            return

    # When `n_features>=65535`, `scipy.sparse.hstack` may not use the right
    # dtype for representing indices and indptr if `n_features` is still
    # small enough so that each block matrix's indices and indptr arrays
    # can be represented with `np.int32`. We test `n_features==65535`
    # since it is guaranteed to run into this bug.
    if (
        sp_version < parse_version("1.9.2")
        and n_features == 65535
        and degree == 2
        and not interaction_only
    ):  # pragma: no cover
        msg = r"In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`"
        with pytest.raises(ValueError, match=msg):
            X_trans = pf.fit_transform(X)
        return
    X_trans = pf.fit_transform(X)

    expected_dtype = np.int64 if num_combinations > np.iinfo(np.int32).max else np.int32
    # Terms higher than first degree
    non_bias_terms = 1 + (degree - 1) * int(not interaction_only)
    expected_nnz = int(include_bias) + non_bias_terms
    assert X_trans.dtype == X.dtype
    assert X_trans.shape == (1, pf.n_output_features_)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == expected_dtype
    assert X_trans.nnz == expected_nnz

    if include_bias:
        assert X_trans[0, 0] == pytest.approx(1.0)
    for idx in range(non_bias_terms):
        assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)

    offset = interaction_only * n_features
    if degree == 3:
        offset *= 1 + n_features
    assert pf.n_output_features_ == expected_indices[degree - 1] + 1 - offset


@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_csr_polynomial_expansion_too_large_to_index(interaction_only, include_bias):
    n_features = np.iinfo(np.int64).max // 2
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    X = sparse.csr_matrix((data, (row, col)))
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=(2, 2)
    )
    msg = (
        r"The output that would result from the current configuration would have \d*"
        r" features which is too large to be indexed"
    )
    with pytest.raises(ValueError, match=msg):
        pf.fit(X)
    with pytest.raises(ValueError, match=msg):
        pf.fit_transform(X)


def test_polynomial_features_behaviour_on_zero_degree():
    """Check that PolynomialFeatures raises error when degree=0 and include_bias=False,
    and output a single constant column when include_bias=True
    """
    X = np.ones((10, 2))
    poly = PolynomialFeatures(degree=0, include_bias=False)
    err_msg = (
        "Setting degree to zero and include_bias to False would result in"
        " an empty output array."
    )
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    poly = PolynomialFeatures(degree=(0, 0), include_bias=False)
    err_msg = (
        "Setting both min_degree and max_degree to zero and include_bias to"
        " False would result in an empty output array."
    )
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    for _X in [X, sparse.csr_matrix(X), sparse.csc_matrix(X)]:
        poly = PolynomialFeatures(degree=0, include_bias=True)
        output = poly.fit_transform(_X)
        # convert to dense array if needed
        if sparse.issparse(output):
            output = output.toarray()
        assert_array_equal(output, np.ones((X.shape[0], 1)))


def test_sizeof_LARGEST_INT_t():
    # On Windows, scikit-learn is typically compiled with MSVC that
    # does not support int128 arithmetic (at the time of writing):
    # https://stackoverflow.com/a/6761962/163740
    if sys.platform == "win32" or (
        sys.maxsize <= 2**32 and sys.platform != "emscripten"
    ):
        expected_size = 8
    else:
        expected_size = 16

    assert _get_sizeof_LARGEST_INT_t() == expected_size


@pytest.mark.xfail(
    sys.platform == "win32",
    reason=(
        "On Windows, scikit-learn is typically compiled with MSVC that does not support"
        " int128 arithmetic (at the time of writing)"
    ),
    run=True,
)
def test_csr_polynomial_expansion_windows_fail():
    # Minimum needed to ensure integer overflow occurs while guaranteeing an
    # int64-indexable output.
    n_features = int(np.iinfo(np.int64).max ** (1 / 3) + 3)
    data = [1.0]
    row = [0]
    col = [n_features - 1]

    # First degree index
    expected_indices = [
        n_features - 1,
    ]
    # Second degree index
    expected_indices.append(
        int(n_features * (n_features + 1) // 2 + expected_indices[0])
    )
    # Third degree index
    expected_indices.append(
        int(n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1])
    )

    X = sparse.csr_matrix((data, (row, col)))
    pf = PolynomialFeatures(interaction_only=False, include_bias=False, degree=3)
    if sys.maxsize <= 2**32:
        msg = (
            r"The output that would result from the current configuration would"
            r" have \d*"
            r" features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit_transform(X)
    else:
        X_trans = pf.fit_transform(X)
        for idx in range(3):
            assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)
