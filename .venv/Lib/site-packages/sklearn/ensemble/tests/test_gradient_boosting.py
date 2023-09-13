"""
Testing for the gradient boosting module (sklearn.ensemble.gradient_boosting).
"""
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.special import expit

from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal,
    skip_if_32bit,
)

GRADIENT_BOOSTING_ESTIMATORS = [GradientBoostingClassifier, GradientBoostingRegressor]

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also make regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=4, n_informative=8, noise=10, random_state=7
)
y_reg = scale(y_reg)

rng = np.random.RandomState(0)
# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@pytest.mark.parametrize("loss", ("log_loss", "exponential"))
def test_classification_toy(loss, global_random_seed):
    # Check classification on a toy dataset.
    clf = GradientBoostingClassifier(
        loss=loss, n_estimators=10, random_state=global_random_seed
    )

    with pytest.raises(ValueError):
        clf.predict(T)

    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf.estimators_)

    log_loss_decrease = clf.train_score_[:-1] - clf.train_score_[1:]
    assert np.any(log_loss_decrease >= 0.0)

    leaves = clf.apply(X)
    assert leaves.shape == (6, 10, 1)


@pytest.mark.parametrize("loss", ("log_loss", "exponential"))
def test_classification_synthetic(loss, global_random_seed):
    # Test GradientBoostingClassifier on synthetic dataset used by
    # Hastie et al. in ESLII - Figure 10.9
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=global_random_seed)

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # Increasing the number of trees should decrease the test error
    common_params = {
        "max_depth": 1,
        "learning_rate": 1.0,
        "loss": loss,
        "random_state": global_random_seed,
    }
    gbrt_100_stumps = GradientBoostingClassifier(n_estimators=100, **common_params)
    gbrt_100_stumps.fit(X_train, y_train)

    gbrt_200_stumps = GradientBoostingClassifier(n_estimators=200, **common_params)
    gbrt_200_stumps.fit(X_train, y_train)

    assert gbrt_100_stumps.score(X_test, y_test) < gbrt_200_stumps.score(X_test, y_test)

    # Decision stumps are better suited for this dataset with a large number of
    # estimators.
    common_params = {
        "n_estimators": 200,
        "learning_rate": 1.0,
        "loss": loss,
        "random_state": global_random_seed,
    }
    gbrt_stumps = GradientBoostingClassifier(max_depth=1, **common_params)
    gbrt_stumps.fit(X_train, y_train)

    gbrt_10_nodes = GradientBoostingClassifier(max_leaf_nodes=10, **common_params)
    gbrt_10_nodes.fit(X_train, y_train)

    assert gbrt_stumps.score(X_test, y_test) > gbrt_10_nodes.score(X_test, y_test)


@pytest.mark.parametrize("loss", ("squared_error", "absolute_error", "huber"))
@pytest.mark.parametrize("subsample", (1.0, 0.5))
def test_regression_dataset(loss, subsample, global_random_seed):
    # Check consistency on regression dataset with least squares
    # and least absolute deviation.
    ones = np.ones(len(y_reg))
    last_y_pred = None
    for sample_weight in [None, ones, 2 * ones]:
        # learning_rate, max_depth and n_estimators were adjusted to get a mode
        # that is accurate enough to reach a low MSE on the training set while
        # keeping the resource used to execute this test low enough.
        reg = GradientBoostingRegressor(
            n_estimators=30,
            loss=loss,
            max_depth=4,
            subsample=subsample,
            min_samples_split=2,
            random_state=global_random_seed,
            learning_rate=0.5,
        )

        reg.fit(X_reg, y_reg, sample_weight=sample_weight)
        leaves = reg.apply(X_reg)
        assert leaves.shape == (100, 30)

        y_pred = reg.predict(X_reg)
        mse = mean_squared_error(y_reg, y_pred)
        assert mse < 0.05

        if last_y_pred is not None:
            # FIXME: We temporarily bypass this test. This is due to the fact
            # that GBRT with and without `sample_weight` do not use the same
            # implementation of the median during the initialization with the
            # `DummyRegressor`. In the future, we should make sure that both
            # implementations should be the same. See PR #17377 for more.
            # assert_allclose(last_y_pred, y_pred)
            pass

        last_y_pred = y_pred


@pytest.mark.parametrize("subsample", (1.0, 0.5))
@pytest.mark.parametrize("sample_weight", (None, 1))
def test_iris(subsample, sample_weight, global_random_seed):
    if sample_weight == 1:
        sample_weight = np.ones(len(iris.target))
    # Check consistency on dataset iris.
    clf = GradientBoostingClassifier(
        n_estimators=100,
        loss="log_loss",
        random_state=global_random_seed,
        subsample=subsample,
    )
    clf.fit(iris.data, iris.target, sample_weight=sample_weight)
    score = clf.score(iris.data, iris.target)
    assert score > 0.9

    leaves = clf.apply(iris.data)
    assert leaves.shape == (150, 100, 3)


def test_regression_synthetic(global_random_seed):
    # Test on synthetic regression datasets used in Leo Breiman,
    # `Bagging Predictors?. Machine Learning 24(2): 123-140 (1996).
    random_state = check_random_state(global_random_seed)
    regression_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_split": 2,
        "learning_rate": 0.1,
        "loss": "squared_error",
        "random_state": global_random_seed,
    }

    # Friedman1
    X, y = datasets.make_friedman1(n_samples=1200, random_state=random_state, noise=1.0)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 6.5

    # Friedman2
    X, y = datasets.make_friedman2(n_samples=1200, random_state=random_state)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 2500.0

    # Friedman3
    X, y = datasets.make_friedman3(n_samples=1200, random_state=random_state)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 0.025


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (GradientBoostingRegressor, X_reg, y_reg),
        (GradientBoostingClassifier, iris.data, iris.target),
    ],
)
def test_feature_importances(GradientBoosting, X, y):
    # smoke test to check that the gradient boosting expose an attribute
    # feature_importances_
    gbdt = GradientBoosting()
    assert not hasattr(gbdt, "feature_importances_")
    gbdt.fit(X, y)
    assert hasattr(gbdt, "feature_importances_")


def test_probability_log(global_random_seed):
    # Predict probabilities.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=global_random_seed)

    with pytest.raises(ValueError):
        clf.predict_proba(T)

    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)

    # check if probabilities are in [0, 1].
    y_proba = clf.predict_proba(T)
    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)

    # derive predictions from probabilities
    y_pred = clf.classes_.take(y_proba.argmax(axis=1), axis=0)
    assert_array_equal(y_pred, true_result)


def test_single_class_with_sample_weight():
    sample_weight = [0, 0, 0, 1, 1, 1]
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    msg = (
        "y contains 1 class after sample_weight trimmed classes with "
        "zero weights, while a minimum of 2 classes are required."
    )
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y, sample_weight=sample_weight)


def test_check_inputs_predict_stages():
    # check that predict_stages through an error if the type of X is not
    # supported
    x, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    x_sparse_csc = csc_matrix(x)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(x, y)
    score = np.zeros((y.shape)).reshape(-1, 1)
    err_msg = "When X is a sparse matrix, a CSR format is expected"
    with pytest.raises(ValueError, match=err_msg):
        predict_stages(clf.estimators_, x_sparse_csc, clf.learning_rate, score)
    x_fortran = np.asfortranarray(x)
    with pytest.raises(ValueError, match="X should be C-ordered np.ndarray"):
        predict_stages(clf.estimators_, x_fortran, clf.learning_rate, score)


def test_max_feature_regression(global_random_seed):
    # Test to make sure random state is set properly.
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=global_random_seed)

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    gbrt = GradientBoostingClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_depth=2,
        learning_rate=0.1,
        max_features=2,
        random_state=global_random_seed,
    )
    gbrt.fit(X_train, y_train)
    log_loss = gbrt._loss(y_test, gbrt.decision_function(X_test))
    assert log_loss < 0.5, "GB failed with deviance %.4f" % log_loss


def test_feature_importance_regression(
    fetch_california_housing_fxt, global_random_seed
):
    """Test that Gini importance is calculated correctly.

    This test follows the example from [1]_ (pg. 373).

    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements
       of statistical learning. New York: Springer series in statistics.
    """
    california = fetch_california_housing_fxt()
    X, y = california.data, california.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=global_random_seed
    )

    reg = GradientBoostingRegressor(
        loss="huber",
        learning_rate=0.1,
        max_leaf_nodes=6,
        n_estimators=100,
        random_state=global_random_seed,
    )
    reg.fit(X_train, y_train)
    sorted_idx = np.argsort(reg.feature_importances_)[::-1]
    sorted_features = [california.feature_names[s] for s in sorted_idx]

    # The most important feature is the median income by far.
    assert sorted_features[0] == "MedInc"

    # The three subsequent features are the following. Their relative ordering
    # might change a bit depending on the randomness of the trees and the
    # train / test split.
    assert set(sorted_features[1:4]) == {"Longitude", "AveOccup", "Latitude"}


def test_max_features():
    # Test if max features is set properly for floats and str.
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
    _, n_features = X.shape

    X_train = X[:2000]
    y_train = y[:2000]

    gbrt = GradientBoostingClassifier(n_estimators=1, max_features=None)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == n_features

    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=None)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == n_features

    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.3)
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(n_features * 0.3)

    gbrt = GradientBoostingRegressor(n_estimators=1, max_features="sqrt")
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(np.sqrt(n_features))

    gbrt = GradientBoostingRegressor(n_estimators=1, max_features="log2")
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == int(np.log2(n_features))

    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.01 / X.shape[1])
    gbrt.fit(X_train, y_train)
    assert gbrt.max_features_ == 1


def test_staged_predict():
    # Test whether staged decision function eventually gives
    # the same prediction.
    X, y = datasets.make_friedman1(n_samples=1200, random_state=1, noise=1.0)
    X_train, y_train = X[:200], y[:200]
    X_test = X[200:]
    clf = GradientBoostingRegressor()
    # test raise ValueError if not fitted
    with pytest.raises(ValueError):
        np.fromiter(clf.staged_predict(X_test), dtype=np.float64)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # test if prediction for last stage equals ``predict``
    for y in clf.staged_predict(X_test):
        assert y.shape == y_pred.shape

    assert_array_almost_equal(y_pred, y)


def test_staged_predict_proba():
    # Test whether staged predict proba eventually gives
    # the same prediction.
    X, y = datasets.make_hastie_10_2(n_samples=1200, random_state=1)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]
    clf = GradientBoostingClassifier(n_estimators=20)
    # test raise NotFittedError if not
    with pytest.raises(NotFittedError):
        np.fromiter(clf.staged_predict_proba(X_test), dtype=np.float64)

    clf.fit(X_train, y_train)

    # test if prediction for last stage equals ``predict``
    for y_pred in clf.staged_predict(X_test):
        assert y_test.shape == y_pred.shape

    assert_array_equal(clf.predict(X_test), y_pred)

    # test if prediction for last stage equals ``predict_proba``
    for staged_proba in clf.staged_predict_proba(X_test):
        assert y_test.shape[0] == staged_proba.shape[0]
        assert 2 == staged_proba.shape[1]

    assert_array_almost_equal(clf.predict_proba(X_test), staged_proba)


@pytest.mark.parametrize("Estimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_staged_functions_defensive(Estimator, global_random_seed):
    # test that staged_functions make defensive copies
    rng = np.random.RandomState(global_random_seed)
    X = rng.uniform(size=(10, 3))
    y = (4 * X[:, 0]).astype(int) + 1  # don't predict zeros
    estimator = Estimator()
    estimator.fit(X, y)
    for func in ["predict", "decision_function", "predict_proba"]:
        staged_func = getattr(estimator, "staged_" + func, None)
        if staged_func is None:
            # regressor has no staged_predict_proba
            continue
        with warnings.catch_warnings(record=True):
            staged_result = list(staged_func(X))
        staged_result[1][:] = 0
        assert np.all(staged_result[0] != 0)


def test_serialization():
    # Check model serialization.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)

    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    serialized_clf = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    clf = None
    clf = pickle.loads(serialized_clf)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)


def test_degenerate_targets():
    # Check if we can fit even though all targets are equal.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # classifier should raise exception
    with pytest.raises(ValueError):
        clf.fit(X, np.ones(len(X)))

    clf = GradientBoostingRegressor(n_estimators=100, random_state=1)
    clf.fit(X, np.ones(len(X)))
    clf.predict([rng.rand(2)])
    assert_array_equal(np.ones((1,), dtype=np.float64), clf.predict([rng.rand(2)]))


def test_quantile_loss(global_random_seed):
    # Check if quantile loss with alpha=0.5 equals absolute_error.
    clf_quantile = GradientBoostingRegressor(
        n_estimators=100,
        loss="quantile",
        max_depth=4,
        alpha=0.5,
        random_state=global_random_seed,
    )

    clf_quantile.fit(X_reg, y_reg)
    y_quantile = clf_quantile.predict(X_reg)

    clf_ae = GradientBoostingRegressor(
        n_estimators=100,
        loss="absolute_error",
        max_depth=4,
        random_state=global_random_seed,
    )

    clf_ae.fit(X_reg, y_reg)
    y_ae = clf_ae.predict(X_reg)
    assert_allclose(y_quantile, y_ae)


def test_symbol_labels():
    # Test with non-integer class labels.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    symbol_y = tosequence(map(str, y))

    clf.fit(X, symbol_y)
    assert_array_equal(clf.predict(T), tosequence(map(str, true_result)))
    assert 100 == len(clf.estimators_)


def test_float_class_labels():
    # Test with float class labels.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    float_y = np.asarray(y, dtype=np.float32)

    clf.fit(X, float_y)
    assert_array_equal(clf.predict(T), np.asarray(true_result, dtype=np.float32))
    assert 100 == len(clf.estimators_)


def test_shape_y():
    # Test with float class labels.
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    y_ = np.asarray(y, dtype=np.int32)
    y_ = y_[:, np.newaxis]

    # This will raise a DataConversionWarning that we want to
    # "always" raise, elsewhere the warnings gets ignored in the
    # later tests, and the tests that check for this warning fail
    warn_msg = (
        "A column-vector y was passed when a 1d array was expected. "
        "Please change the shape of y to \\(n_samples, \\), for "
        "example using ravel()."
    )
    with pytest.warns(DataConversionWarning, match=warn_msg):
        clf.fit(X, y_)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)


def test_mem_layout():
    # Test with different memory layouts of X and y
    X_ = np.asfortranarray(X)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X_, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)

    X_ = np.ascontiguousarray(X)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X_, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)

    y_ = np.asarray(y, dtype=np.int32)
    y_ = np.ascontiguousarray(y_)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y_)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)

    y_ = np.asarray(y, dtype=np.int32)
    y_ = np.asfortranarray(y_)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y_)
    assert_array_equal(clf.predict(T), true_result)
    assert 100 == len(clf.estimators_)


@pytest.mark.parametrize("GradientBoostingEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_oob_improvement(GradientBoostingEstimator):
    # Test if oob improvement has correct shape and regression test.
    estimator = GradientBoostingEstimator(
        n_estimators=100, random_state=1, subsample=0.5
    )
    estimator.fit(X, y)
    assert estimator.oob_improvement_.shape[0] == 100
    # hard-coded regression test - change if modification in OOB computation
    assert_array_almost_equal(
        estimator.oob_improvement_[:5],
        np.array([0.19, 0.15, 0.12, -0.11, 0.11]),
        decimal=2,
    )


@pytest.mark.parametrize("GradientBoostingEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_oob_scores(GradientBoostingEstimator):
    # Test if oob scores has correct shape and regression test.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    estimator = GradientBoostingEstimator(
        n_estimators=100, random_state=1, subsample=0.5
    )
    estimator.fit(X, y)
    assert estimator.oob_scores_.shape[0] == 100
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    estimator = GradientBoostingEstimator(
        n_estimators=100,
        random_state=1,
        subsample=0.5,
        n_iter_no_change=5,
    )
    estimator.fit(X, y)
    assert estimator.oob_scores_.shape[0] < 100
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)


@pytest.mark.parametrize(
    "GradientBoostingEstimator, oob_attribute",
    [
        (GradientBoostingClassifier, "oob_improvement_"),
        (GradientBoostingClassifier, "oob_scores_"),
        (GradientBoostingClassifier, "oob_score_"),
        (GradientBoostingRegressor, "oob_improvement_"),
        (GradientBoostingRegressor, "oob_scores_"),
        (GradientBoostingRegressor, "oob_score_"),
    ],
)
def test_oob_attributes_error(GradientBoostingEstimator, oob_attribute):
    """
    Check that we raise an AttributeError when the OOB statistics were not computed.
    """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    estimator = GradientBoostingEstimator(
        n_estimators=100,
        random_state=1,
        subsample=1.0,
    )
    estimator.fit(X, y)
    with pytest.raises(AttributeError):
        estimator.oob_attribute


def test_oob_multilcass_iris():
    # Check OOB improvement on multi-class dataset.
    estimator = GradientBoostingClassifier(
        n_estimators=100, loss="log_loss", random_state=1, subsample=0.5
    )
    estimator.fit(iris.data, iris.target)
    score = estimator.score(iris.data, iris.target)
    assert score > 0.9
    assert estimator.oob_improvement_.shape[0] == estimator.n_estimators
    assert estimator.oob_scores_.shape[0] == estimator.n_estimators
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    estimator = GradientBoostingClassifier(
        n_estimators=100,
        loss="log_loss",
        random_state=1,
        subsample=0.5,
        n_iter_no_change=5,
    )
    estimator.fit(iris.data, iris.target)
    score = estimator.score(iris.data, iris.target)
    assert estimator.oob_improvement_.shape[0] < estimator.n_estimators
    assert estimator.oob_scores_.shape[0] < estimator.n_estimators
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    # hard-coded regression test - change if modification in OOB computation
    # FIXME: the following snippet does not yield the same results on 32 bits
    # assert_array_almost_equal(estimator.oob_improvement_[:5],
    #                           np.array([12.68, 10.45, 8.18, 6.43, 5.13]),
    #                           decimal=2)


def test_verbose_output():
    # Check verbose=1 does not cause error.
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    clf = GradientBoostingClassifier(
        n_estimators=100, random_state=1, verbose=1, subsample=0.8
    )
    clf.fit(X, y)
    verbose_output = sys.stdout
    sys.stdout = old_stdout

    # check output
    verbose_output.seek(0)
    header = verbose_output.readline().rstrip()
    # with OOB
    true_header = " ".join(["%10s"] + ["%16s"] * 3) % (
        "Iter",
        "Train Loss",
        "OOB Improve",
        "Remaining Time",
    )
    assert true_header == header

    n_lines = sum(1 for l in verbose_output.readlines())
    # one for 1-10 and then 9 for 20-100
    assert 10 + 9 == n_lines


def test_more_verbose_output():
    # Check verbose=2 does not cause error.
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1, verbose=2)
    clf.fit(X, y)
    verbose_output = sys.stdout
    sys.stdout = old_stdout

    # check output
    verbose_output.seek(0)
    header = verbose_output.readline().rstrip()
    # no OOB
    true_header = " ".join(["%10s"] + ["%16s"] * 2) % (
        "Iter",
        "Train Loss",
        "Remaining Time",
    )
    assert true_header == header

    n_lines = sum(1 for l in verbose_output.readlines())
    # 100 lines for n_estimators==100
    assert 100 == n_lines


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start(Cls, global_random_seed):
    # Test if warm start equals fit.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    est = Cls(n_estimators=200, max_depth=1, random_state=global_random_seed)
    est.fit(X, y)

    est_ws = Cls(
        n_estimators=100, max_depth=1, warm_start=True, random_state=global_random_seed
    )
    est_ws.fit(X, y)
    est_ws.set_params(n_estimators=200)
    est_ws.fit(X, y)

    if Cls is GradientBoostingRegressor:
        assert_allclose(est_ws.predict(X), est.predict(X))
    else:
        # Random state is preserved and hence predict_proba must also be
        # same
        assert_array_equal(est_ws.predict(X), est.predict(X))
        assert_allclose(est_ws.predict_proba(X), est.predict_proba(X))


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_n_estimators(Cls, global_random_seed):
    # Test if warm start equals fit - set n_estimators.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    est = Cls(n_estimators=300, max_depth=1, random_state=global_random_seed)
    est.fit(X, y)

    est_ws = Cls(
        n_estimators=100, max_depth=1, warm_start=True, random_state=global_random_seed
    )
    est_ws.fit(X, y)
    est_ws.set_params(n_estimators=300)
    est_ws.fit(X, y)

    assert_allclose(est_ws.predict(X), est.predict(X))


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_max_depth(Cls):
    # Test if possible to fit trees of different depth in ensemble.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=110, max_depth=2)
    est.fit(X, y)

    # last 10 trees have different depth
    assert est.estimators_[0, 0].max_depth == 1
    for i in range(1, 11):
        assert est.estimators_[-i, 0].max_depth == 2


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_clear(Cls):
    # Test if fit clears state.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1)
    est.fit(X, y)

    est_2 = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est_2.fit(X, y)  # inits state
    est_2.set_params(warm_start=False)
    est_2.fit(X, y)  # clears old state and equals est

    assert_array_almost_equal(est_2.predict(X), est.predict(X))


@pytest.mark.parametrize("GradientBoosting", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_state_oob_scores(GradientBoosting):
    """
    Check that the states of the OOB scores are cleared when used with `warm_start`.
    """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    n_estimators = 100
    estimator = GradientBoosting(
        n_estimators=n_estimators,
        max_depth=1,
        subsample=0.5,
        warm_start=True,
        random_state=1,
    )
    estimator.fit(X, y)
    oob_scores, oob_score = estimator.oob_scores_, estimator.oob_score_
    assert len(oob_scores) == n_estimators
    assert oob_scores[-1] == pytest.approx(oob_score)

    n_more_estimators = 200
    estimator.set_params(n_estimators=n_more_estimators).fit(X, y)
    assert len(estimator.oob_scores_) == n_more_estimators
    assert_allclose(estimator.oob_scores_[:n_estimators], oob_scores)

    estimator.set_params(n_estimators=n_estimators, warm_start=False).fit(X, y)
    assert estimator.oob_scores_ is not oob_scores
    assert estimator.oob_score_ is not oob_score
    assert_allclose(estimator.oob_scores_, oob_scores)
    assert estimator.oob_score_ == pytest.approx(oob_score)
    assert oob_scores[-1] == pytest.approx(oob_score)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_smaller_n_estimators(Cls):
    # Test if warm start with smaller n_estimators raises error
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=99)
    with pytest.raises(ValueError):
        est.fit(X, y)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_equal_n_estimators(Cls):
    # Test if warm start with equal n_estimators does nothing
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1)
    est.fit(X, y)

    est2 = clone(est)
    est2.set_params(n_estimators=est.n_estimators, warm_start=True)
    est2.fit(X, y)

    assert_array_almost_equal(est2.predict(X), est.predict(X))


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_oob_switch(Cls):
    # Test if oob can be turned on during warm start.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=110, subsample=0.5)
    est.fit(X, y)

    assert_array_equal(est.oob_improvement_[:100], np.zeros(100))
    assert_array_equal(est.oob_scores_[:100], np.zeros(100))

    # the last 10 are not zeros
    assert (est.oob_improvement_[-10:] != 0.0).all()
    assert (est.oob_scores_[-10:] != 0.0).all()

    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_oob(Cls):
    # Test if warm start OOB equals fit.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=200, max_depth=1, subsample=0.5, random_state=1)
    est.fit(X, y)

    est_ws = Cls(
        n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True
    )
    est_ws.fit(X, y)
    est_ws.set_params(n_estimators=200)
    est_ws.fit(X, y)

    assert_array_almost_equal(est_ws.oob_improvement_[:100], est.oob_improvement_[:100])
    assert_array_almost_equal(est_ws.oob_scores_[:100], est.oob_scores_[:100])
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)
    assert est_ws.oob_scores_[-1] == pytest.approx(est_ws.oob_score_)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_sparse(Cls):
    # Test that all sparse matrix types are supported
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    sparse_matrix_type = [csr_matrix, csc_matrix, coo_matrix]
    est_dense = Cls(
        n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True
    )
    est_dense.fit(X, y)
    est_dense.predict(X)
    est_dense.set_params(n_estimators=200)
    est_dense.fit(X, y)
    y_pred_dense = est_dense.predict(X)

    for sparse_constructor in sparse_matrix_type:
        X_sparse = sparse_constructor(X)

        est_sparse = Cls(
            n_estimators=100,
            max_depth=1,
            subsample=0.5,
            random_state=1,
            warm_start=True,
        )
        est_sparse.fit(X_sparse, y)
        est_sparse.predict(X)
        est_sparse.set_params(n_estimators=200)
        est_sparse.fit(X_sparse, y)
        y_pred_sparse = est_sparse.predict(X)

        assert_array_almost_equal(
            est_dense.oob_improvement_[:100], est_sparse.oob_improvement_[:100]
        )
        assert est_dense.oob_scores_[-1] == pytest.approx(est_dense.oob_score_)
        assert_array_almost_equal(
            est_dense.oob_scores_[:100], est_sparse.oob_scores_[:100]
        )
        assert est_sparse.oob_scores_[-1] == pytest.approx(est_sparse.oob_score_)
        assert_array_almost_equal(y_pred_dense, y_pred_sparse)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_fortran(Cls, global_random_seed):
    # Test that feeding a X in Fortran-ordered is giving the same results as
    # in C-ordered
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    est_c = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)
    est_fortran = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)

    est_c.fit(X, y)
    est_c.set_params(n_estimators=11)
    est_c.fit(X, y)

    X_fortran = np.asfortranarray(X)
    est_fortran.fit(X_fortran, y)
    est_fortran.set_params(n_estimators=11)
    est_fortran.fit(X_fortran, y)

    assert_allclose(est_c.predict(X), est_fortran.predict(X))


def early_stopping_monitor(i, est, locals):
    """Returns True on the 10th iteration."""
    if i == 9:
        return True
    else:
        return False


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_monitor_early_stopping(Cls):
    # Test if monitor return value works.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    est = Cls(n_estimators=20, max_depth=1, random_state=1, subsample=0.5)
    est.fit(X, y, monitor=early_stopping_monitor)
    assert est.n_estimators == 20  # this is not altered
    assert est.estimators_.shape[0] == 10
    assert est.train_score_.shape[0] == 10
    assert est.oob_improvement_.shape[0] == 10
    assert est.oob_scores_.shape[0] == 10
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    # try refit
    est.set_params(n_estimators=30)
    est.fit(X, y)
    assert est.n_estimators == 30
    assert est.estimators_.shape[0] == 30
    assert est.train_score_.shape[0] == 30
    assert est.oob_improvement_.shape[0] == 30
    assert est.oob_scores_.shape[0] == 30
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    est = Cls(
        n_estimators=20, max_depth=1, random_state=1, subsample=0.5, warm_start=True
    )
    est.fit(X, y, monitor=early_stopping_monitor)
    assert est.n_estimators == 20
    assert est.estimators_.shape[0] == 10
    assert est.train_score_.shape[0] == 10
    assert est.oob_improvement_.shape[0] == 10
    assert est.oob_scores_.shape[0] == 10
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    # try refit
    est.set_params(n_estimators=30, warm_start=False)
    est.fit(X, y)
    assert est.n_estimators == 30
    assert est.train_score_.shape[0] == 30
    assert est.estimators_.shape[0] == 30
    assert est.oob_improvement_.shape[0] == 30
    assert est.oob_scores_.shape[0] == 30
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)


def test_complete_classification():
    # Test greedy trees with max_depth + 1 leafs.
    from sklearn.tree._tree import TREE_LEAF

    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    k = 4

    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=None, random_state=1, max_leaf_nodes=k + 1
    )
    est.fit(X, y)

    tree = est.estimators_[0, 0].tree_
    assert tree.max_depth == k
    assert tree.children_left[tree.children_left == TREE_LEAF].shape[0] == k + 1


def test_complete_regression():
    # Test greedy trees with max_depth + 1 leafs.
    from sklearn.tree._tree import TREE_LEAF

    k = 4

    est = GradientBoostingRegressor(
        n_estimators=20, max_depth=None, random_state=1, max_leaf_nodes=k + 1
    )
    est.fit(X_reg, y_reg)

    tree = est.estimators_[-1, 0].tree_
    assert tree.children_left[tree.children_left == TREE_LEAF].shape[0] == k + 1


def test_zero_estimator_reg(global_random_seed):
    # Test if init='zero' works for regression by checking that it is better
    # than a simple baseline.

    baseline = DummyRegressor(strategy="mean").fit(X_reg, y_reg)
    mse_baseline = mean_squared_error(baseline.predict(X_reg), y_reg)
    est = GradientBoostingRegressor(
        n_estimators=5,
        max_depth=1,
        random_state=global_random_seed,
        init="zero",
        learning_rate=0.5,
    )
    est.fit(X_reg, y_reg)
    y_pred = est.predict(X_reg)
    mse_gbdt = mean_squared_error(y_reg, y_pred)
    assert mse_gbdt < mse_baseline


def test_zero_estimator_clf(global_random_seed):
    # Test if init='zero' works for classification.
    X = iris.data
    y = np.array(iris.target)

    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=1, random_state=global_random_seed, init="zero"
    )
    est.fit(X, y)

    assert est.score(X, y) > 0.96

    # binary clf
    mask = y != 0
    y[mask] = 1
    y[~mask] = 0
    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=1, random_state=global_random_seed, init="zero"
    )
    est.fit(X, y)
    assert est.score(X, y) > 0.96


@pytest.mark.parametrize("GBEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_max_leaf_nodes_max_depth(GBEstimator):
    # Test precedence of max_leaf_nodes over max_depth.
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    k = 4

    est = GBEstimator(max_depth=1, max_leaf_nodes=k).fit(X, y)
    tree = est.estimators_[0, 0].tree_
    assert tree.max_depth == 1

    est = GBEstimator(max_depth=1).fit(X, y)
    tree = est.estimators_[0, 0].tree_
    assert tree.max_depth == 1


@pytest.mark.parametrize("GBEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_min_impurity_decrease(GBEstimator):
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    est = GBEstimator(min_impurity_decrease=0.1)
    est.fit(X, y)
    for tree in est.estimators_.flat:
        # Simply check if the parameter is passed on correctly. Tree tests
        # will suffice for the actual working of this param
        assert tree.min_impurity_decrease == 0.1


def test_warm_start_wo_nestimators_change():
    # Test if warm_start does nothing if n_estimators is not changed.
    # Regression test for #3513.
    clf = GradientBoostingClassifier(n_estimators=10, warm_start=True)
    clf.fit([[0, 1], [2, 3]], [0, 1])
    assert clf.estimators_.shape[0] == 10
    clf.fit([[0, 1], [2, 3]], [0, 1])
    assert clf.estimators_.shape[0] == 10


def test_probability_exponential(global_random_seed):
    # Predict probabilities.
    clf = GradientBoostingClassifier(
        loss="exponential", n_estimators=100, random_state=global_random_seed
    )

    with pytest.raises(ValueError):
        clf.predict_proba(T)

    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)

    # check if probabilities are in [0, 1].
    y_proba = clf.predict_proba(T)
    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)
    score = clf.decision_function(T).ravel()
    assert_allclose(y_proba[:, 1], expit(2 * score))

    # derive predictions from probabilities
    y_pred = clf.classes_.take(y_proba.argmax(axis=1), axis=0)
    assert_array_equal(y_pred, true_result)


def test_non_uniform_weights_toy_edge_case_reg():
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1]
    for loss in ("huber", "squared_error", "absolute_error", "quantile"):
        gb = GradientBoostingRegressor(learning_rate=1.0, n_estimators=2, loss=loss)
        gb.fit(X, y, sample_weight=sample_weight)
        assert gb.predict([[1, 0]])[0] > 0.5


def test_non_uniform_weights_toy_edge_case_clf():
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1]
    for loss in ("log_loss", "exponential"):
        gb = GradientBoostingClassifier(n_estimators=5, loss=loss)
        gb.fit(X, y, sample_weight=sample_weight)
        assert_array_equal(gb.predict([[1, 0]]), [1])


@skip_if_32bit
@pytest.mark.parametrize(
    "EstimatorClass", (GradientBoostingClassifier, GradientBoostingRegressor)
)
@pytest.mark.parametrize("sparse_matrix", (csr_matrix, csc_matrix, coo_matrix))
def test_sparse_input(EstimatorClass, sparse_matrix):
    y, X = datasets.make_multilabel_classification(
        random_state=0, n_samples=50, n_features=1, n_classes=20
    )
    y = y[:, 0]
    X_sparse = sparse_matrix(X)

    dense = EstimatorClass(
        n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-7
    ).fit(X, y)
    sparse = EstimatorClass(
        n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-7
    ).fit(X_sparse, y)

    assert_array_almost_equal(sparse.apply(X), dense.apply(X))
    assert_array_almost_equal(sparse.predict(X), dense.predict(X))
    assert_array_almost_equal(sparse.feature_importances_, dense.feature_importances_)

    assert_array_almost_equal(sparse.predict(X_sparse), dense.predict(X))
    assert_array_almost_equal(dense.predict(X_sparse), sparse.predict(X))

    if issubclass(EstimatorClass, GradientBoostingClassifier):
        assert_array_almost_equal(sparse.predict_proba(X), dense.predict_proba(X))
        assert_array_almost_equal(
            sparse.predict_log_proba(X), dense.predict_log_proba(X)
        )

        assert_array_almost_equal(
            sparse.decision_function(X_sparse), sparse.decision_function(X)
        )
        assert_array_almost_equal(
            dense.decision_function(X_sparse), sparse.decision_function(X)
        )
        for res_sparse, res in zip(
            sparse.staged_decision_function(X_sparse),
            sparse.staged_decision_function(X),
        ):
            assert_array_almost_equal(res_sparse, res)


@pytest.mark.parametrize(
    "GradientBoostingEstimator", [GradientBoostingClassifier, GradientBoostingRegressor]
)
def test_gradient_boosting_early_stopping(GradientBoostingEstimator):
    # Check if early stopping works as expected, that is empirically check that the
    # number of trained estimators is increasing when the tolerance decreases.

    X, y = make_classification(n_samples=1000, random_state=0)
    n_estimators = 1000

    gb_large_tol = GradientBoostingEstimator(
        n_estimators=n_estimators,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        tol=1e-1,
    )

    gb_small_tol = GradientBoostingEstimator(
        n_estimators=n_estimators,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        tol=1e-3,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gb_large_tol.fit(X_train, y_train)
    gb_small_tol.fit(X_train, y_train)

    assert gb_large_tol.n_estimators_ < gb_small_tol.n_estimators_ < n_estimators

    assert gb_large_tol.score(X_test, y_test) > 0.7
    assert gb_small_tol.score(X_test, y_test) > 0.7


def test_gradient_boosting_without_early_stopping():
    # When early stopping is not used, the number of trained estimators
    # must be the one specified.
    X, y = make_classification(n_samples=1000, random_state=0)

    gbc = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbc.fit(X, y)
    gbr = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbr.fit(X, y)

    # The number of trained estimators must be the one specified.
    assert gbc.n_estimators_ == 50
    assert gbr.n_estimators_ == 30


def test_gradient_boosting_validation_fraction():
    X, y = make_classification(n_samples=1000, random_state=0)

    gbc = GradientBoostingClassifier(
        n_estimators=100,
        n_iter_no_change=10,
        validation_fraction=0.1,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gbc2 = clone(gbc).set_params(validation_fraction=0.3)
    gbc3 = clone(gbc).set_params(n_iter_no_change=20)

    gbr = GradientBoostingRegressor(
        n_estimators=100,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        validation_fraction=0.1,
        random_state=42,
    )
    gbr2 = clone(gbr).set_params(validation_fraction=0.3)
    gbr3 = clone(gbr).set_params(n_iter_no_change=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Check if validation_fraction has an effect
    gbc.fit(X_train, y_train)
    gbc2.fit(X_train, y_train)
    assert gbc.n_estimators_ != gbc2.n_estimators_

    gbr.fit(X_train, y_train)
    gbr2.fit(X_train, y_train)
    assert gbr.n_estimators_ != gbr2.n_estimators_

    # Check if n_estimators_ increase monotonically with n_iter_no_change
    # Set validation
    gbc3.fit(X_train, y_train)
    gbr3.fit(X_train, y_train)
    assert gbr.n_estimators_ < gbr3.n_estimators_
    assert gbc.n_estimators_ < gbc3.n_estimators_


def test_early_stopping_stratified():
    # Make sure data splitting for early stopping is stratified
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 0, 1]

    gbc = GradientBoostingClassifier(n_iter_no_change=5)
    with pytest.raises(
        ValueError, match="The least populated class in y has only 1 member"
    ):
        gbc.fit(X, y)


def _make_multiclass():
    return make_classification(n_classes=3, n_clusters_per_class=1)


@pytest.mark.parametrize(
    "gb, dataset_maker, init_estimator",
    [
        (GradientBoostingClassifier, make_classification, DummyClassifier),
        (GradientBoostingClassifier, _make_multiclass, DummyClassifier),
        (GradientBoostingRegressor, make_regression, DummyRegressor),
    ],
    ids=["binary classification", "multiclass classification", "regression"],
)
def test_gradient_boosting_with_init(
    gb, dataset_maker, init_estimator, global_random_seed
):
    # Check that GradientBoostingRegressor works when init is a sklearn
    # estimator.
    # Check that an error is raised if trying to fit with sample weight but
    # initial estimator does not support sample weight

    X, y = dataset_maker()
    sample_weight = np.random.RandomState(global_random_seed).rand(100)

    # init supports sample weights
    init_est = init_estimator()
    gb(init=init_est).fit(X, y, sample_weight=sample_weight)

    # init does not support sample weights
    init_est = NoSampleWeightWrapper(init_estimator())
    gb(init=init_est).fit(X, y)  # ok no sample weights
    with pytest.raises(ValueError, match="estimator.*does not support sample weights"):
        gb(init=init_est).fit(X, y, sample_weight=sample_weight)


def test_gradient_boosting_with_init_pipeline():
    # Check that the init estimator can be a pipeline (see issue #13466)

    X, y = make_regression(random_state=0)
    init = make_pipeline(LinearRegression())
    gb = GradientBoostingRegressor(init=init)
    gb.fit(X, y)  # pipeline without sample_weight works fine

    with pytest.raises(
        ValueError,
        match="The initial estimator Pipeline does not support sample weights",
    ):
        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))

    # Passing sample_weight to a pipeline raises a ValueError. This test makes
    # sure we make the distinction between ValueError raised by a pipeline that
    # was passed sample_weight, and a InvalidParameterError raised by a regular
    # estimator whose input checking failed.
    invalid_nu = 1.5
    err_msg = (
        "The 'nu' parameter of NuSVR must be a float in the"
        f" range (0.0, 1.0]. Got {invalid_nu} instead."
    )
    with pytest.raises(InvalidParameterError, match=re.escape(err_msg)):
        # Note that NuSVR properly supports sample_weight
        init = NuSVR(gamma="auto", nu=invalid_nu)
        gb = GradientBoostingRegressor(init=init)
        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))


def test_early_stopping_n_classes():
    # when doing early stopping (_, , y_train, _ = train_test_split(X, y))
    # there might be classes in y that are missing in y_train. As the init
    # estimator will be trained on y_train, we need to raise an error if this
    # happens.

    X = [[1]] * 10
    y = [0, 0] + [1] * 8  # only 2 negative class over 10 samples
    gb = GradientBoostingClassifier(
        n_iter_no_change=5, random_state=0, validation_fraction=0.8
    )
    with pytest.raises(
        ValueError, match="The training data after the early stopping split"
    ):
        gb.fit(X, y)

    # No error if we let training data be big enough
    gb = GradientBoostingClassifier(
        n_iter_no_change=5, random_state=0, validation_fraction=0.4
    )


def test_gbr_degenerate_feature_importances():
    # growing an ensemble of single node trees. See #13620
    X = np.zeros((10, 10))
    y = np.ones((10,))
    gbr = GradientBoostingRegressor().fit(X, y)
    assert_array_equal(gbr.feature_importances_, np.zeros(10, dtype=np.float64))
