import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
)

X = [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]


@pytest.mark.parametrize(
    "strategy, expected, sample_weight",
    [
        ("uniform", [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]], None),
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], None),
        ("quantile", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], None),
        (
            "quantile",
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
            [1, 1, 2, 1],
        ),
        (
            "quantile",
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
        (
            "quantile",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            [0, 1, 1, 1],
        ),
        (
            "kmeans",
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [1, 0, 3, 1],
        ),
        (
            "kmeans",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [1, 1, 1, 1],
        ),
    ],
)
# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
def test_fit_transform(strategy, expected, sample_weight):
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)
    est.fit(X, sample_weight=sample_weight)
    assert_array_equal(expected, est.transform(X))


def test_valid_n_bins():
    KBinsDiscretizer(n_bins=2).fit_transform(X)
    KBinsDiscretizer(n_bins=np.array([2])[0]).fit_transform(X)
    assert KBinsDiscretizer(n_bins=2).fit(X).n_bins_.dtype == np.dtype(int)


@pytest.mark.parametrize("strategy", ["uniform"])
def test_kbinsdiscretizer_wrong_strategy_with_weights(strategy):
    """Check that we raise an error when the wrong strategy is used."""
    sample_weight = np.ones(shape=(len(X)))
    est = KBinsDiscretizer(n_bins=3, strategy=strategy)
    err_msg = (
        "`sample_weight` was provided but it cannot be used with strategy='uniform'."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, sample_weight=sample_weight)


def test_invalid_n_bins_array():
    # Bad shape
    n_bins = np.full((2, 4), 2.0)
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Incorrect number of features
    n_bins = [1, 2, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Bad bin values
    n_bins = [1, 2, 2, 1]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 3. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Float bin values
    n_bins = [2.1, 2, 2.1, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 2. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)


@pytest.mark.parametrize(
    "strategy, expected, sample_weight",
    [
        ("uniform", [[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]], None),
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2]], None),
        ("quantile", [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]], None),
        (
            "quantile",
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
            [1, 1, 3, 1],
        ),
        (
            "quantile",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            [0, 1, 3, 1],
        ),
        # (
        #     "quantile",
        #     [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
        #     [1, 1, 1, 1],
        # ),
        #
        # TODO: This test case above aims to test if the case where an array of
        #       ones passed in sample_weight parameter is equal to the case when
        #       sample_weight is None.
        #       Unfortunately, the behavior of `_weighted_percentile` when
        #       `sample_weight = [1, 1, 1, 1]` are currently not equivalent.
        #       This problem has been addressed in issue :
        #       https://github.com/scikit-learn/scikit-learn/issues/17370
        (
            "kmeans",
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 2]],
            [1, 0, 3, 1],
        ),
    ],
)
# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
def test_fit_transform_n_bins_array(strategy, expected, sample_weight):
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3], encode="ordinal", strategy=strategy
    ).fit(X, sample_weight=sample_weight)
    assert_array_equal(expected, est.transform(X))

    # test the shape of bin_edges_
    n_features = np.array(X).shape[1]
    assert est.bin_edges_.shape == (n_features,)
    for bin_edges, n_bins in zip(est.bin_edges_, est.n_bins_):
        assert bin_edges.shape == (n_bins + 1,)


@pytest.mark.filterwarnings("ignore: Bins whose width are too small")
def test_kbinsdiscretizer_effect_sample_weight():
    """Check the impact of `sample_weight` one computed quantiles."""
    X = np.array([[-2], [-1], [1], [3], [500], [1000]])
    # add a large number of bins such that each sample with a non-null weight
    # will be used as bin edge
    est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    est.fit(X, sample_weight=[1, 1, 1, 1, 0, 0])
    assert_allclose(est.bin_edges_[0], [-2, -1, 1, 3])
    assert_allclose(est.transform(X), [[0.0], [1.0], [2.0], [2.0], [2.0], [2.0]])


# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
@pytest.mark.parametrize("strategy", ["kmeans", "quantile"])
def test_kbinsdiscretizer_no_mutating_sample_weight(strategy):
    """Make sure that `sample_weight` is not changed in place."""
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)
    sample_weight = np.array([1, 3, 1, 2], dtype=np.float64)
    sample_weight_copy = np.copy(sample_weight)
    est.fit(X, sample_weight=sample_weight)
    assert_allclose(sample_weight, sample_weight_copy)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_same_min_max(strategy):
    warnings.simplefilter("always")
    X = np.array([[1, -2], [1, -1], [1, 0], [1, 1]])
    est = KBinsDiscretizer(strategy=strategy, n_bins=3, encode="ordinal")
    warning_message = "Feature 0 is constant and will be replaced with 0."
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X)
    assert est.n_bins_[0] == 1
    # replace the feature with zeros
    Xt = est.transform(X)
    assert_array_equal(Xt[:, 0], np.zeros(X.shape[0]))


def test_transform_1d_behavior():
    X = np.arange(4)
    est = KBinsDiscretizer(n_bins=2)
    with pytest.raises(ValueError):
        est.fit(X)

    est = KBinsDiscretizer(n_bins=2)
    est.fit(X.reshape(-1, 1))
    with pytest.raises(ValueError):
        est.transform(X)


@pytest.mark.parametrize("i", range(1, 9))
def test_numeric_stability(i):
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    Xt_expected = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)

    # Test up to discretizing nano units
    X = X_init / 10**i
    Xt = KBinsDiscretizer(n_bins=2, encode="ordinal").fit_transform(X)
    assert_array_equal(Xt_expected, Xt)


def test_encode_options():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="ordinal").fit(X)
    Xt_1 = est.transform(X)
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot-dense").fit(X)
    Xt_2 = est.transform(X)
    assert not sp.issparse(Xt_2)
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=False
        ).fit_transform(Xt_1),
        Xt_2,
    )
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot").fit(X)
    Xt_3 = est.transform(X)
    assert sp.issparse(Xt_3)
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=True
        )
        .fit_transform(Xt_1)
        .toarray(),
        Xt_3.toarray(),
    )


@pytest.mark.parametrize(
    "strategy, expected_2bins, expected_3bins, expected_5bins",
    [
        ("uniform", [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 2, 2], [0, 0, 1, 1, 4, 4]),
        ("kmeans", [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 3, 4]),
        ("quantile", [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 4]),
    ],
)
# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
def test_nonuniform_strategies(
    strategy, expected_2bins, expected_3bins, expected_5bins
):
    X = np.array([0, 0.5, 2, 3, 9, 10]).reshape(-1, 1)

    # with 2 bins
    est = KBinsDiscretizer(n_bins=2, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_2bins, Xt.ravel())

    # with 3 bins
    est = KBinsDiscretizer(n_bins=3, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_3bins, Xt.ravel())

    # with 5 bins
    est = KBinsDiscretizer(n_bins=5, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_5bins, Xt.ravel())


@pytest.mark.parametrize(
    "strategy, expected_inv",
    [
        (
            "uniform",
            [
                [-1.5, 2.0, -3.5, -0.5],
                [-0.5, 3.0, -2.5, -0.5],
                [0.5, 4.0, -1.5, 0.5],
                [0.5, 4.0, -1.5, 1.5],
            ],
        ),
        (
            "kmeans",
            [
                [-1.375, 2.125, -3.375, -0.5625],
                [-1.375, 2.125, -3.375, -0.5625],
                [-0.125, 3.375, -2.125, 0.5625],
                [0.75, 4.25, -1.25, 1.625],
            ],
        ),
        (
            "quantile",
            [
                [-1.5, 2.0, -3.5, -0.75],
                [-0.5, 3.0, -2.5, 0.0],
                [0.5, 4.0, -1.5, 1.25],
                [0.5, 4.0, -1.5, 1.25],
            ],
        ),
    ],
)
# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_inverse_transform(strategy, encode, expected_inv):
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy, encode=encode)
    Xt = kbd.fit_transform(X)
    Xinv = kbd.inverse_transform(Xt)
    assert_array_almost_equal(expected_inv, Xinv)


# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_transform_outside_fit_range(strategy):
    X = np.array([0, 1, 2, 3])[:, None]
    kbd = KBinsDiscretizer(n_bins=4, strategy=strategy, encode="ordinal")
    kbd.fit(X)

    X2 = np.array([-2, 5])[:, None]
    X2t = kbd.transform(X2)
    assert_array_equal(X2t.max(axis=0) + 1, kbd.n_bins_)
    assert_array_equal(X2t.min(axis=0), [0])


def test_overwrite():
    X = np.array([0, 1, 2, 3])[:, None]
    X_before = X.copy()

    est = KBinsDiscretizer(n_bins=3, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(X, X_before)

    Xt_before = Xt.copy()
    Xinv = est.inverse_transform(Xt)
    assert_array_equal(Xt, Xt_before)
    assert_array_equal(Xinv, np.array([[0.5], [1.5], [2.5], [2.5]]))


@pytest.mark.parametrize(
    "strategy, expected_bin_edges", [("quantile", [0, 1, 3]), ("kmeans", [0, 1.5, 3])]
)
def test_redundant_bins(strategy, expected_bin_edges):
    X = [[0], [0], [0], [0], [3], [3]]
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy)
    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)
    assert_array_almost_equal(kbd.bin_edges_[0], expected_bin_edges)


def test_percentile_numeric_stability():
    X = np.array([0.05, 0.05, 0.95]).reshape(-1, 1)
    bin_edges = np.array([0.05, 0.23, 0.41, 0.59, 0.77, 0.95])
    Xt = np.array([0, 0, 4]).reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)

    assert_array_almost_equal(kbd.bin_edges_[0], bin_edges)
    assert_array_almost_equal(kbd.transform(X), Xt)


@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("out_dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_consistent_dtype(in_dtype, out_dtype, encode):
    X_input = np.array(X, dtype=in_dtype)
    kbd = KBinsDiscretizer(n_bins=3, encode=encode, dtype=out_dtype)
    kbd.fit(X_input)

    # test output dtype
    if out_dtype is not None:
        expected_dtype = out_dtype
    elif out_dtype is None and X_input.dtype == np.float16:
        # wrong numeric input dtype are cast in np.float64
        expected_dtype = np.float64
    else:
        expected_dtype = X_input.dtype
    Xt = kbd.transform(X_input)
    assert Xt.dtype == expected_dtype


@pytest.mark.parametrize("input_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_32_equal_64(input_dtype, encode):
    # TODO this check is redundant with common checks and can be removed
    #  once #16290 is merged
    X_input = np.array(X, dtype=input_dtype)

    # 32 bit output
    kbd_32 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float32)
    kbd_32.fit(X_input)
    Xt_32 = kbd_32.transform(X_input)

    # 64 bit output
    kbd_64 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float64)
    kbd_64.fit(X_input)
    Xt_64 = kbd_64.transform(X_input)

    assert_allclose_dense_sparse(Xt_32, Xt_64)


def test_kbinsdiscretizer_subsample_default():
    # Since the size of X is small (< 2e5), subsampling will not take place.
    X = np.array([-2, 1.5, -4, -1]).reshape(-1, 1)
    kbd_default = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    kbd_default.fit(X)

    kbd_without_subsampling = clone(kbd_default)
    kbd_without_subsampling.set_params(subsample=None)
    kbd_without_subsampling.fit(X)

    for bin_kbd_default, bin_kbd_with_subsampling in zip(
        kbd_default.bin_edges_[0], kbd_without_subsampling.bin_edges_[0]
    ):
        np.testing.assert_allclose(bin_kbd_default, bin_kbd_with_subsampling)
    assert kbd_default.bin_edges_.shape == kbd_without_subsampling.bin_edges_.shape


@pytest.mark.parametrize(
    "encode, expected_names",
    [
        (
            "onehot",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        (
            "onehot-dense",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        ("ordinal", [f"feat{col_id}" for col_id in range(3)]),
    ],
)
def test_kbinsdiscrtizer_get_feature_names_out(encode, expected_names):
    """Check get_feature_names_out for different settings.
    Non-regression test for #22731
    """
    X = [[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]]

    kbd = KBinsDiscretizer(n_bins=4, encode=encode).fit(X)
    Xt = kbd.transform(X)

    input_features = [f"feat{i}" for i in range(3)]
    output_names = kbd.get_feature_names_out(input_features)
    assert Xt.shape[1] == output_names.shape[0]

    assert_array_equal(output_names, expected_names)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_kbinsdiscretizer_subsample(strategy, global_random_seed):
    # Check that the bin edges are almost the same when subsampling is used.
    X = np.random.RandomState(global_random_seed).random_sample((100000, 1)) + 1

    kbd_subsampling = KBinsDiscretizer(
        strategy=strategy, subsample=50000, random_state=global_random_seed
    )
    kbd_subsampling.fit(X)

    kbd_no_subsampling = clone(kbd_subsampling)
    kbd_no_subsampling.set_params(subsample=None)
    kbd_no_subsampling.fit(X)

    # We use a large tolerance because we can't expect the bin edges to be exactly the
    # same when subsampling is used.
    assert_allclose(
        kbd_subsampling.bin_edges_[0], kbd_no_subsampling.bin_edges_[0], rtol=1e-2
    )


# TODO(1.5) remove this test
@pytest.mark.parametrize("strategy", ["uniform", "kmeans"])
def test_kbd_subsample_warning(strategy):
    # Check the future warning for the change of default of subsample
    X = np.random.RandomState(0).random_sample((100, 1))

    kbd = KBinsDiscretizer(strategy=strategy, random_state=0)
    with pytest.warns(FutureWarning, match="subsample=200_000 will be used by default"):
        kbd.fit(X)
