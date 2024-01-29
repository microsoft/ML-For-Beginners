import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version

from imblearn.ensemble import BalancedRandomForestClassifier

sklearn_version = parse_version(sklearn.__version__)


@pytest.fixture
def imbalanced_dataset():
    return make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94],
        class_sep=0.8,
        random_state=0,
    )


def test_balanced_random_forest_error_warning_warm_start(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        n_estimators=5, sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf.fit(*imbalanced_dataset)

    with pytest.raises(ValueError, match="must be larger or equal to"):
        brf.set_params(warm_start=True, n_estimators=2)
        brf.fit(*imbalanced_dataset)

    brf.set_params(n_estimators=10)
    brf.fit(*imbalanced_dataset)

    with pytest.warns(UserWarning, match="Warm-start fitting without"):
        brf.fit(*imbalanced_dataset)


def test_balanced_random_forest(imbalanced_dataset):
    n_estimators = 10
    brf = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(*imbalanced_dataset)

    assert len(brf.samplers_) == n_estimators
    assert len(brf.estimators_) == n_estimators
    assert len(brf.pipelines_) == n_estimators
    assert len(brf.feature_importances_) == imbalanced_dataset[0].shape[1]


def test_balanced_random_forest_attributes(imbalanced_dataset):
    X, y = imbalanced_dataset
    n_estimators = 10
    brf = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(X, y)

    for idx in range(n_estimators):
        X_res, y_res = brf.samplers_[idx].fit_resample(X, y)
        X_res_2, y_res_2 = (
            brf.pipelines_[idx].named_steps["randomundersampler"].fit_resample(X, y)
        )
        assert_allclose(X_res, X_res_2)
        assert_array_equal(y_res, y_res_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict(X)
        assert_array_equal(y_pred, y_pred_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict_proba(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict_proba(X)
        assert_array_equal(y_pred, y_pred_2)


def test_balanced_random_forest_sample_weight(imbalanced_dataset):
    rng = np.random.RandomState(42)
    X, y = imbalanced_dataset
    sample_weight = rng.rand(y.shape[0])
    brf = BalancedRandomForestClassifier(
        n_estimators=5,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(X, y, sample_weight)


@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob(imbalanced_dataset):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )
    est = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=0,
        n_estimators=1000,
        min_samples_leaf=2,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    est.fit(X_train, y_train)
    test_score = est.score(X_test, y_test)

    assert abs(test_score - est.oob_score_) < 0.1

    # Check warning if not enough estimators
    est = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=0,
        n_estimators=1,
        bootstrap=True,
        sampling_strategy="all",
        replacement=True,
    )
    with pytest.warns(UserWarning) and np.errstate(divide="ignore", invalid="ignore"):
        est.fit(X, y)


def test_balanced_random_forest_grid_search(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False
    )
    grid = GridSearchCV(brf, {"n_estimators": (1, 2), "max_depth": (1, 2)}, cv=3)
    grid.fit(*imbalanced_dataset)


def test_little_tree_with_small_max_samples():
    rng = np.random.RandomState(1)

    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0

    # First fit with no restriction on max samples
    est1 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    # Second fit with max samples restricted to just 2
    est2 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    est1.fit(X, y)
    est2.fit(X, y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


def test_balanced_random_forest_pruning(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf.fit(*imbalanced_dataset)
    n_nodes_no_pruning = brf.estimators_[0].tree_.node_count

    brf_pruned = BalancedRandomForestClassifier(
        ccp_alpha=0.015, sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf_pruned.fit(*imbalanced_dataset)
    n_nodes_pruning = brf_pruned.estimators_[0].tree_.node_count

    assert n_nodes_no_pruning > n_nodes_pruning


@pytest.mark.parametrize("ratio", [0.5, 0.1])
@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob_binomial(ratio):
    # Regression test for #655: check that the oob score is closed to 0.5
    # a binomial experiment.
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = np.arange(n_samples).reshape(-1, 1)
    y = rng.binomial(1, ratio, size=n_samples)

    erf = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=42,
        sampling_strategy="not minority",
        replacement=False,
        bootstrap=True,
    )
    erf.fit(X, y)
    assert np.abs(erf.oob_score_ - 0.5) < 0.1


def test_balanced_bagging_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    X, y = load_iris(return_X_y=True)
    estimator = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False
    ).fit(X, y)
    with pytest.warns(FutureWarning, match="`n_features_` was deprecated"):
        estimator.n_features_


# TODO: remove in 0.13
def test_balanced_random_forest_change_behaviour(imbalanced_dataset):
    """Check that we raise a change of behaviour for the parameters `sampling_strategy`
    and `replacement`.
    """
    estimator = BalancedRandomForestClassifier(sampling_strategy="all", bootstrap=False)
    with pytest.warns(FutureWarning, match="The default of `replacement`"):
        estimator.fit(*imbalanced_dataset)
    estimator = BalancedRandomForestClassifier(replacement=True, bootstrap=False)
    with pytest.warns(FutureWarning, match="The default of `sampling_strategy`"):
        estimator.fit(*imbalanced_dataset)
    estimator = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True
    )
    with pytest.warns(FutureWarning, match="The default of `bootstrap`"):
        estimator.fit(*imbalanced_dataset)


@pytest.mark.skipif(
    parse_version(sklearn_version.base_version) < parse_version("1.4"),
    reason="scikit-learn should be >= 1.4",
)
def test_missing_values_is_resilient():
    """Check that forest can deal with missing values and has decent performance."""

    rng = np.random.RandomState(0)
    n_samples, n_features = 1000, 10
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )

    # Create dataset with missing values
    X_missing = X.copy()
    X_missing[rng.choice([False, True], size=X.shape, p=[0.95, 0.05])] = np.nan
    assert np.isnan(X_missing).any()

    X_missing_train, X_missing_test, y_train, y_test = train_test_split(
        X_missing, y, random_state=0
    )

    # Train forest with missing values
    forest_with_missing = BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=rng,
        n_estimators=50,
    )
    forest_with_missing.fit(X_missing_train, y_train)
    score_with_missing = forest_with_missing.score(X_missing_test, y_test)

    # Train forest without missing values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    forest = BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=rng,
        n_estimators=50,
    )
    forest.fit(X_train, y_train)
    score_without_missing = forest.score(X_test, y_test)

    # Score is still 80 percent of the forest's score that had no missing values
    assert score_with_missing >= 0.80 * score_without_missing


@pytest.mark.skipif(
    parse_version(sklearn_version.base_version) < parse_version("1.4"),
    reason="scikit-learn should be >= 1.4",
)
def test_missing_value_is_predictive():
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    rng = np.random.RandomState(0)
    n_samples = 300

    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)

    # Create a predictive feature using `y` and with some noise
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()

    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature

    (
        X_predictive_train,
        X_predictive_test,
        X_non_predictive_train,
        X_non_predictive_test,
        y_train,
        y_test,
    ) = train_test_split(X_predictive, X_non_predictive, y, random_state=0)
    forest_predictive = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False, random_state=0
    ).fit(X_predictive_train, y_train)
    forest_non_predictive = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False, random_state=0
    ).fit(X_non_predictive_train, y_train)

    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)

    assert predictive_test_score >= 0.75
    assert predictive_test_score >= forest_non_predictive.score(
        X_non_predictive_test, y_test
    )
