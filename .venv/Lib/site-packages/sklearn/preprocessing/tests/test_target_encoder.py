import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    TargetEncoder,
)


def _encode_target(X_ordinal, y_int, n_categories, smooth):
    """Simple Python implementation of target encoding."""
    cur_encodings = np.zeros(n_categories, dtype=np.float64)
    y_mean = np.mean(y_int)

    if smooth == "auto":
        y_variance = np.var(y_int)
        for c in range(n_categories):
            y_subset = y_int[X_ordinal == c]
            n_i = y_subset.shape[0]

            if n_i == 0:
                cur_encodings[c] = y_mean
                continue

            y_subset_variance = np.var(y_subset)
            m = y_subset_variance / y_variance
            lambda_ = n_i / (n_i + m)

            cur_encodings[c] = lambda_ * np.mean(y_subset) + (1 - lambda_) * y_mean
        return cur_encodings
    else:  # float
        for c in range(n_categories):
            y_subset = y_int[X_ordinal == c]
            current_sum = np.sum(y_subset) + y_mean * smooth
            current_cnt = y_subset.shape[0] + smooth
            cur_encodings[c] = current_sum / current_cnt
        return cur_encodings


@pytest.mark.parametrize(
    "categories, unknown_value",
    [
        ([np.array([0, 1, 2], dtype=np.int64)], 4),
        ([np.array([1.0, 3.0, np.nan], dtype=np.float64)], 6.0),
        ([np.array(["cat", "dog", "snake"], dtype=object)], "bear"),
        ("auto", 3),
    ],
)
@pytest.mark.parametrize("smooth", [5.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary", "continuous"])
def test_encoding(categories, unknown_value, global_random_seed, smooth, target_type):
    """Check encoding for binary and continuous targets."""

    X_train_array = np.array([[0] * 20 + [1] * 30 + [2] * 40], dtype=np.int64).T
    X_test_array = np.array([[0, 1, 2]], dtype=np.int64).T
    n_categories = 3
    n_samples = X_train_array.shape[0]

    if categories == "auto":
        X_train = X_train_array
    else:
        X_train = categories[0][X_train_array]

    if categories == "auto":
        X_test = X_test_array
    else:
        X_test = categories[0][X_test_array]
    X_test = np.concatenate((X_test, [[unknown_value]]))

    rng = np.random.RandomState(global_random_seed)

    if target_type == "binary":
        y_int = rng.randint(low=0, high=2, size=n_samples)
        target_names = np.array(["cat", "dog"], dtype=object)
        y_train = target_names[y_int]
        cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    else:  # target_type == continuous
        y_int = rng.uniform(low=-10, high=20, size=n_samples)
        y_train = y_int
        cv = KFold(n_splits=3, random_state=0, shuffle=True)

    shuffled_idx = rng.permutation(n_samples)
    X_train_array = X_train_array[shuffled_idx]
    X_train = X_train[shuffled_idx]
    y_train = y_train[shuffled_idx]
    y_int = y_int[shuffled_idx]

    # Get encodings for cv splits to validate `fit_transform`
    expected_X_fit_transform = np.empty_like(X_train_array, dtype=np.float64)

    for train_idx, test_idx in cv.split(X_train_array, y_train):
        X_, y_ = X_train_array[train_idx, 0], y_int[train_idx]
        cur_encodings = _encode_target(X_, y_, n_categories, smooth)
        expected_X_fit_transform[test_idx, 0] = cur_encodings[
            X_train_array[test_idx, 0]
        ]

    target_encoder = TargetEncoder(
        smooth=smooth, categories=categories, cv=3, random_state=0
    )

    X_fit_transform = target_encoder.fit_transform(X_train, y_train)

    assert target_encoder.target_type_ == target_type
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    assert len(target_encoder.encodings_) == 1

    # compute encodings for all data to validate `transform`
    y_mean = np.mean(y_int)
    expected_encodings = _encode_target(
        X_train_array[:, 0], y_int, n_categories, smooth
    )
    assert_allclose(target_encoder.encodings_[0], expected_encodings)
    assert target_encoder.target_mean_ == pytest.approx(y_mean)

    # Transform on test data, the last value is unknown is it is encoded as the target
    # mean
    expected_X_test_transform = np.concatenate(
        (expected_encodings, np.array([y_mean]))
    ).reshape(-1, 1)

    X_test_transform = target_encoder.transform(X_test)
    assert_allclose(X_test_transform, expected_X_test_transform)


@pytest.mark.parametrize(
    "X, categories",
    [
        (
            np.array([[0] * 10 + [1] * 10 + [3]], dtype=np.int64).T,  # 3 is unknown
            [[0, 1, 2]],
        ),
        (
            np.array(
                [["cat"] * 10 + ["dog"] * 10 + ["snake"]], dtype=object
            ).T,  # snake is unknown
            [["dog", "cat", "cow"]],
        ),
    ],
)
@pytest.mark.parametrize("smooth", [4.0, "auto"])
def test_custom_categories(X, categories, smooth):
    """Custom categories with unknown categories that are not in training data."""
    rng = np.random.RandomState(0)
    y = rng.uniform(low=-10, high=20, size=X.shape[0])
    enc = TargetEncoder(categories=categories, smooth=smooth, random_state=0).fit(X, y)

    # The last element is unknown and encoded as the mean
    y_mean = y.mean()
    X_trans = enc.transform(X[-1:])
    assert X_trans[0, 0] == pytest.approx(y_mean)

    assert len(enc.encodings_) == 1
    # custom category that is not in training data
    assert enc.encodings_[0][-1] == pytest.approx(y_mean)


@pytest.mark.parametrize(
    "y, msg",
    [
        ([1, 2, 0, 1], "Found input variables with inconsistent"),
        (
            np.array([[1, 2, 0], [1, 2, 3]]).T,
            "Target type was inferred to be 'multiclass-multioutput'",
        ),
        (["cat", "dog", "bear"], "Target type was inferred to be 'multiclass'"),
    ],
)
def test_errors(y, msg):
    """Check invalidate input."""
    X = np.array([[1, 0, 1]]).T

    enc = TargetEncoder()
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X, y)


def test_use_regression_target():
    """Custom target_type to avoid inferring the target type."""
    X = np.array([[0, 1, 0, 1, 0, 1]]).T

    # XXX: When multiclass is supported, then the following `y`
    # is considered a multiclass problem and `TargetEncoder` will not error.
    # type_of_target would be 'multiclass'
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    enc = TargetEncoder()
    msg = "Target type was inferred to be 'multiclass'"
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X, y)

    enc = TargetEncoder(target_type="continuous")
    enc.fit_transform(X, y)
    assert enc.target_type_ == "continuous"


def test_feature_names_out_set_output():
    """Check TargetEncoder works with set_output."""
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"] * 10, "B": [1, 2] * 10})
    y = [1, 2] * 10

    enc_default = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_default.set_output(transform="default")
    enc_pandas = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_pandas.set_output(transform="pandas")

    X_default = enc_default.fit_transform(X_df, y)
    X_pandas = enc_pandas.fit_transform(X_df, y)

    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(enc_pandas.get_feature_names_out(), ["A", "B"])
    assert_array_equal(enc_pandas.get_feature_names_out(), X_pandas.columns)


@pytest.mark.parametrize("to_pandas", [True, False])
@pytest.mark.parametrize("smooth", [1.0, "auto"])
@pytest.mark.parametrize("target_type", ["binary-ints", "binary-str", "continuous"])
def test_multiple_features_quick(to_pandas, smooth, target_type):
    """Check target encoder with multiple features."""
    X_ordinal = np.array(
        [[1, 1], [0, 1], [1, 1], [2, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64
    )
    if target_type == "binary-str":
        y_train = np.array(["a", "b", "a", "a", "b", "b", "a", "b"])
        y_integer = LabelEncoder().fit_transform(y_train)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    elif target_type == "binary-ints":
        y_train = np.array([3, 4, 3, 3, 3, 4, 4, 4])
        y_integer = LabelEncoder().fit_transform(y_train)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    else:
        y_train = np.array([3.0, 5.1, 2.4, 3.5, 4.1, 5.5, 10.3, 7.3], dtype=np.float32)
        y_integer = y_train
        cv = KFold(2, random_state=0, shuffle=True)
    y_mean = np.mean(y_integer)
    categories = [[0, 1, 2], [0, 1]]

    X_test = np.array(
        [
            [0, 1],
            [3, 0],  # 3 is unknown
            [1, 10],  # 10 is unknown
        ],
        dtype=np.int64,
    )

    if to_pandas:
        pd = pytest.importorskip("pandas")
        # convert second feature to an object
        X_train = pd.DataFrame(
            {
                "feat0": X_ordinal[:, 0],
                "feat1": np.array(["cat", "dog"], dtype=object)[X_ordinal[:, 1]],
            }
        )
        # "snake" is unknown
        X_test = pd.DataFrame({"feat0": X_test[:, 0], "feat1": ["dog", "cat", "snake"]})
    else:
        X_train = X_ordinal

    # manually compute encoding for fit_transform
    expected_X_fit_transform = np.empty_like(X_ordinal, dtype=np.float64)
    for f_idx, cats in enumerate(categories):
        for train_idx, test_idx in cv.split(X_ordinal, y_integer):
            X_, y_ = X_ordinal[train_idx, f_idx], y_integer[train_idx]
            current_encoding = _encode_target(X_, y_, len(cats), smooth)
            expected_X_fit_transform[test_idx, f_idx] = current_encoding[
                X_ordinal[test_idx, f_idx]
            ]

    # manually compute encoding for transform
    expected_encodings = []
    for f_idx, cats in enumerate(categories):
        current_encoding = _encode_target(
            X_ordinal[:, f_idx], y_integer, len(cats), smooth
        )
        expected_encodings.append(current_encoding)

    expected_X_test_transform = np.array(
        [
            [expected_encodings[0][0], expected_encodings[1][1]],
            [y_mean, expected_encodings[1][0]],
            [expected_encodings[0][1], y_mean],
        ],
        dtype=np.float64,
    )

    enc = TargetEncoder(smooth=smooth, cv=2, random_state=0)
    X_fit_transform = enc.fit_transform(X_train, y_train)
    assert_allclose(X_fit_transform, expected_X_fit_transform)

    assert len(enc.encodings_) == 2
    for i in range(2):
        assert_allclose(enc.encodings_[i], expected_encodings[i])

    X_test_transform = enc.transform(X_test)
    assert_allclose(X_test_transform, expected_X_test_transform)


@pytest.mark.parametrize(
    "y, y_mean",
    [
        (np.array([3.4] * 20), 3.4),
        (np.array([0] * 20), 0),
        (np.array(["a"] * 20, dtype=object), 0),
    ],
    ids=["continuous", "binary", "binary-string"],
)
@pytest.mark.parametrize("smooth", ["auto", 4.0, 0.0])
def test_constant_target_and_feature(y, y_mean, smooth):
    """Check edge case where feature and target is constant."""
    X = np.array([[1] * 20]).T
    n_samples = X.shape[0]

    enc = TargetEncoder(cv=2, smooth=smooth, random_state=0)
    X_trans = enc.fit_transform(X, y)
    assert_allclose(X_trans, np.repeat([[y_mean]], n_samples, axis=0))
    assert enc.encodings_[0][0] == pytest.approx(y_mean)
    assert enc.target_mean_ == pytest.approx(y_mean)

    X_test = np.array([[1], [0]])
    X_test_trans = enc.transform(X_test)
    assert_allclose(X_test_trans, np.repeat([[y_mean]], 2, axis=0))


def test_fit_transform_not_associated_with_y_if_ordinal_categorical_is_not(
    global_random_seed,
):
    cardinality = 30  # not too large, otherwise we need a very large n_samples
    n_samples = 3000
    rng = np.random.RandomState(global_random_seed)
    y_train = rng.normal(size=n_samples)
    X_train = rng.randint(0, cardinality, size=n_samples).reshape(-1, 1)

    # Sort by y_train to attempt to cause a leak
    y_sorted_indices = y_train.argsort()
    y_train = y_train[y_sorted_indices]
    X_train = X_train[y_sorted_indices]

    target_encoder = TargetEncoder(shuffle=True, random_state=global_random_seed)
    X_encoded_train_shuffled = target_encoder.fit_transform(X_train, y_train)

    target_encoder = TargetEncoder(shuffle=False)
    X_encoded_train_no_shuffled = target_encoder.fit_transform(X_train, y_train)

    # Check that no information about y_train has leaked into X_train:
    regressor = RandomForestRegressor(
        n_estimators=10, min_samples_leaf=20, random_state=global_random_seed
    )

    # It's impossible to learn a good predictive model on the training set when
    # using the original representation X_train or the target encoded
    # representation with shuffled inner CV. For the latter, no information
    # about y_train has inadvertently leaked into the prior used to generate
    # `X_encoded_train_shuffled`:
    cv = ShuffleSplit(n_splits=50, random_state=global_random_seed)
    assert cross_val_score(regressor, X_train, y_train, cv=cv).mean() < 0.1
    assert (
        cross_val_score(regressor, X_encoded_train_shuffled, y_train, cv=cv).mean()
        < 0.1
    )

    # Without the inner CV shuffling, a lot of information about y_train goes into the
    # the per-fold y_train.mean() priors: shrinkage is no longer effective in this
    # case and would no longer be able to prevent downstream over-fitting.
    assert (
        cross_val_score(regressor, X_encoded_train_no_shuffled, y_train, cv=cv).mean()
        > 0.5
    )


def test_smooth_zero():
    """Check edge case with zero smoothing and cv does not contain category."""
    X = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    y = np.array([2.1, 4.3, 1.2, 3.1, 1.0, 9.0, 10.3, 14.2, 13.3, 15.0])

    enc = TargetEncoder(smooth=0.0, shuffle=False, cv=2)
    X_trans = enc.fit_transform(X, y)

    # With cv = 2, category 0 does not exist in the second half, thus
    # it will be encoded as the mean of the second half
    assert_allclose(X_trans[0], np.mean(y[5:]))

    # category 1 does nto exist in the first half, thus it will be encoded as
    # the mean of the first half
    assert_allclose(X_trans[-1], np.mean(y[:5]))


@pytest.mark.parametrize("smooth", [0.0, 1e3, "auto"])
def test_invariance_of_encoding_under_label_permutation(smooth, global_random_seed):
    # Check that the encoding does not depend on the integer of the value of
    # the integer labels. This is quite of a trivial property but it is helpful
    # to understand the following test.
    rng = np.random.RandomState(global_random_seed)

    # Random y and informative categorical X to make the test non-trivial when
    # using smoothing.
    y = rng.normal(size=1000)
    n_categories = 30
    X = KBinsDiscretizer(n_bins=n_categories, encode="ordinal").fit_transform(
        y.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=global_random_seed
    )

    # Shuffle the labels to make sure that the encoding is invariant to the
    # permutation of the labels
    permutated_labels = rng.permutation(n_categories)
    X_train_permuted = permutated_labels[X_train.astype(np.int32)]
    X_test_permuted = permutated_labels[X_test.astype(np.int32)]

    target_encoder = TargetEncoder(smooth=smooth, random_state=global_random_seed)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    X_train_permuted_encoded = target_encoder.fit_transform(X_train_permuted, y_train)
    X_test_permuted_encoded = target_encoder.transform(X_test_permuted)

    assert_allclose(X_train_encoded, X_train_permuted_encoded)
    assert_allclose(X_test_encoded, X_test_permuted_encoded)


# TODO(1.5) remove warning filter when kbd's subsample default is changed
@pytest.mark.filterwarnings("ignore:In version 1.5 onwards, subsample=200_000")
@pytest.mark.parametrize("smooth", [0.0, "auto"])
def test_target_encoding_for_linear_regression(smooth, global_random_seed):
    # Check some expected statistical properties when fitting a linear
    # regression model on target encoded features depending on there relation
    # with that target.

    # In this test, we use the Ridge class with the "lsqr" solver and a little
    # bit of regularization to implement a linear regression model that
    # converges quickly for large `n_samples` and robustly in case of
    # correlated features. Since we will fit this model on a mean centered
    # target, we do not need to fit an intercept and this will help simplify
    # the analysis with respect to the expected coefficients.
    linear_regression = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False)

    # Construct a random target variable. We need a large number of samples for
    # this test to be stable across all values of the random seed.
    n_samples = 50_000
    rng = np.random.RandomState(global_random_seed)
    y = rng.randn(n_samples)

    # Generate a single informative ordinal feature with medium cardinality.
    # Inject some irreducible noise to make it harder for a multivariate model
    # to identify the informative feature from other pure noise features.
    noise = 0.8 * rng.randn(n_samples)
    n_categories = 100
    X_informative = KBinsDiscretizer(
        n_bins=n_categories,
        encode="ordinal",
        strategy="uniform",
        random_state=rng,
    ).fit_transform((y + noise).reshape(-1, 1))

    # Let's permute the labels to hide the fact that this feature is
    # informative to naive linear regression model trained on the raw ordinal
    # values. As highlighted in the previous test, the target encoding should be
    # invariant to such a permutation.
    permutated_labels = rng.permutation(n_categories)
    X_informative = permutated_labels[X_informative.astype(np.int32)]

    # Generate a shuffled copy of the informative feature to destroy the
    # relationship with the target.
    X_shuffled = rng.permutation(X_informative)

    # Also include a very high cardinality categorical feature that is by
    # itself independent of the target variable: target encoding such a feature
    # without internal cross-validation should cause catastrophic overfitting
    # for the downstream regressor, even with shrinkage. This kind of features
    # typically represents near unique idenfiers of samples. In general they
    # should be removed from a machine learning datasets but here we want to
    # study the ability of the default behavior of TargetEncoder to mitigate
    # them automatically.
    X_near_unique_categories = rng.choice(
        int(0.9 * n_samples), size=n_samples, replace=True
    ).reshape(-1, 1)

    # Assemble the dataset and do a train-test split:
    X = np.concatenate(
        [X_informative, X_shuffled, X_near_unique_categories],
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Let's first check that a linear regression model trained on the raw
    # features underfits because of the meaning-less ordinal encoding of the
    # labels.
    raw_model = linear_regression.fit(X_train, y_train)
    assert raw_model.score(X_train, y_train) < 0.1
    assert raw_model.score(X_test, y_test) < 0.1

    # Now do the same with target encoding using the internal CV mechanism
    # implemented when using fit_transform.
    model_with_cv = make_pipeline(
        TargetEncoder(smooth=smooth, random_state=rng), linear_regression
    ).fit(X_train, y_train)

    # This model should be able to fit the data well and also generalise to the
    # test data (assuming that the binning is fine-grained enough). The R2
    # scores are not perfect because of the noise injected during the
    # generation of the unique informative feature.
    coef = model_with_cv[-1].coef_
    assert model_with_cv.score(X_train, y_train) > 0.5, coef
    assert model_with_cv.score(X_test, y_test) > 0.5, coef

    # The target encoder recovers the linear relationship with slope 1 between
    # the target encoded unique informative predictor and the target. Since the
    # target encoding of the 2 other features is not informative thanks to the
    # use of internal cross-validation, the multivariate linear regressor
    # assigns a coef of 1 to the first feature and 0 to the other 2.
    assert coef[0] == pytest.approx(1, abs=1e-2)
    assert (np.abs(coef[1:]) < 0.2).all()

    # Let's now disable the internal cross-validation by calling fit and then
    # transform separately on the training set:
    target_encoder = TargetEncoder(smooth=smooth, random_state=rng).fit(
        X_train, y_train
    )
    X_enc_no_cv_train = target_encoder.transform(X_train)
    X_enc_no_cv_test = target_encoder.transform(X_test)
    model_no_cv = linear_regression.fit(X_enc_no_cv_train, y_train)

    # The linear regression model should always overfit because it assigns
    # too much weight to the extremely high cardinality feature relatively to
    # the informative feature. Note that this is the case even when using
    # the empirical Bayes smoothing which is not enough to prevent such
    # overfitting alone.
    coef = model_no_cv.coef_
    assert model_no_cv.score(X_enc_no_cv_train, y_train) > 0.7, coef
    assert model_no_cv.score(X_enc_no_cv_test, y_test) < 0.5, coef

    # The model overfits because it assigns too much weight to the high
    # cardinality yet non-informative feature instead of the lower
    # cardinality yet informative feature:
    assert abs(coef[0]) < abs(coef[2])
