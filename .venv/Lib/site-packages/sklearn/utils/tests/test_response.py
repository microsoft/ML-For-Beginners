import numpy as np
import pytest

from sklearn.datasets import (
    load_iris,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import _MockEstimatorOnOffPrediction
from sklearn.utils._response import _get_response_values, _get_response_values_binary
from sklearn.utils._testing import assert_allclose, assert_array_equal

X, y = load_iris(return_X_y=True)
# scale the data to avoid ConvergenceWarning with LogisticRegression
X = scale(X, copy=False)
X_binary, y_binary = X[:100], y[:100]


@pytest.mark.parametrize(
    "response_method", ["decision_function", "predict_proba", "predict_log_proba"]
)
def test_get_response_values_regressor_error(response_method):
    """Check the error message with regressor an not supported response
    method."""
    my_estimator = _MockEstimatorOnOffPrediction(response_methods=[response_method])
    X = "mocking_data", "mocking_target"
    err_msg = f"{my_estimator.__class__.__name__} should either be a classifier"
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(my_estimator, X, response_method=response_method)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_regressor(return_response_method_used):
    """Check the behaviour of `_get_response_values` with regressor."""
    X, y = make_regression(n_samples=10, random_state=0)
    regressor = LinearRegression().fit(X, y)
    results = _get_response_values(
        regressor,
        X,
        response_method="predict",
        return_response_method_used=return_response_method_used,
    )
    assert_array_equal(results[0], regressor.predict(X))
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == "predict"


@pytest.mark.parametrize(
    "response_method",
    ["predict", "decision_function", ["decision_function", "predict"]],
)
@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_outlier_detection(
    response_method, return_response_method_used
):
    """Check the behaviour of `_get_response_values` with outlier detector."""
    X, y = make_classification(n_samples=50, random_state=0)
    outlier_detector = IsolationForest(random_state=0).fit(X, y)
    results = _get_response_values(
        outlier_detector,
        X,
        response_method=response_method,
        return_response_method_used=return_response_method_used,
    )
    chosen_response_method = (
        response_method[0] if isinstance(response_method, list) else response_method
    )
    prediction_method = getattr(outlier_detector, chosen_response_method)
    assert_array_equal(results[0], prediction_method(X))
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == chosen_response_method


@pytest.mark.parametrize(
    "response_method",
    ["predict_proba", "decision_function", "predict", "predict_log_proba"],
)
def test_get_response_values_classifier_unknown_pos_label(response_method):
    """Check that `_get_response_values` raises the proper error message with
    classifier."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=0)
    classifier = LogisticRegression().fit(X, y)

    # provide a `pos_label` which is not in `y`
    err_msg = r"pos_label=whatever is not a valid label: It should be one of \[0 1\]"
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(
            classifier,
            X,
            response_method=response_method,
            pos_label="whatever",
        )


@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_classifier_inconsistent_y_pred_for_binary_proba(
    response_method,
):
    """Check that `_get_response_values` will raise an error when `y_pred` has a
    single class with `predict_proba`."""
    X, y_two_class = make_classification(n_samples=10, n_classes=2, random_state=0)
    y_single_class = np.zeros_like(y_two_class)
    classifier = DecisionTreeClassifier().fit(X, y_single_class)

    err_msg = (
        r"Got predict_proba of shape \(10, 1\), but need classifier with "
        r"two classes"
    )
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(classifier, X, response_method=response_method)


@pytest.mark.parametrize("return_response_method_used", [True, False])
def test_get_response_values_binary_classifier_decision_function(
    return_response_method_used,
):
    """Check the behaviour of `_get_response_values` with `decision_function`
    and binary classifier."""
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    classifier = LogisticRegression().fit(X, y)
    response_method = "decision_function"

    # default `pos_label`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X))
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == "decision_function"

    # when forcing `pos_label=classifier.classes_[0]`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], classifier.decision_function(X) * -1)
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == "decision_function"


@pytest.mark.parametrize("return_response_method_used", [True, False])
@pytest.mark.parametrize("response_method", ["predict_proba", "predict_log_proba"])
def test_get_response_values_binary_classifier_predict_proba(
    return_response_method_used, response_method
):
    """Check that `_get_response_values` with `predict_proba` and binary
    classifier."""
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=0,
    )
    classifier = LogisticRegression().fit(X, y)

    # default `pos_label`
    results = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=None,
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(results[0], getattr(classifier, response_method)(X)[:, 1])
    assert results[1] == 1
    if return_response_method_used:
        assert len(results) == 3
        assert results[2] == response_method
    else:
        assert len(results) == 2

    # when forcing `pos_label=classifier.classes_[0]`
    y_pred, pos_label, *_ = _get_response_values(
        classifier,
        X,
        response_method=response_method,
        pos_label=classifier.classes_[0],
        return_response_method_used=return_response_method_used,
    )
    assert_allclose(y_pred, getattr(classifier, response_method)(X)[:, 0])
    assert pos_label == 0


@pytest.mark.parametrize(
    "estimator, X, y, err_msg, params",
    [
        (
            DecisionTreeRegressor(),
            X_binary,
            y_binary,
            "Expected 'estimator' to be a binary classifier",
            {"response_method": "auto"},
        ),
        (
            DecisionTreeClassifier(),
            X_binary,
            y_binary,
            r"pos_label=unknown is not a valid label: It should be one of \[0 1\]",
            {"response_method": "auto", "pos_label": "unknown"},
        ),
        (
            DecisionTreeClassifier(),
            X,
            y,
            "be a binary classifier. Got 3 classes instead.",
            {"response_method": "predict_proba"},
        ),
    ],
)
def test_get_response_error(estimator, X, y, err_msg, params):
    """Check that we raise the proper error messages in _get_response_values_binary."""

    estimator.fit(X, y)
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values_binary(estimator, X, **params)


def test_get_response_predict_proba():
    """Check the behaviour of `_get_response_values_binary` using `predict_proba`."""
    classifier = DecisionTreeClassifier().fit(X_binary, y_binary)
    y_proba, pos_label = _get_response_values_binary(
        classifier, X_binary, response_method="predict_proba"
    )
    assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1

    y_proba, pos_label = _get_response_values_binary(
        classifier, X_binary, response_method="predict_proba", pos_label=0
    )
    assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 0])
    assert pos_label == 0


def test_get_response_decision_function():
    """Check the behaviour of `_get_response_values_binary` using decision_function."""
    classifier = LogisticRegression().fit(X_binary, y_binary)
    y_score, pos_label = _get_response_values_binary(
        classifier, X_binary, response_method="decision_function"
    )
    assert_allclose(y_score, classifier.decision_function(X_binary))
    assert pos_label == 1

    y_score, pos_label = _get_response_values_binary(
        classifier, X_binary, response_method="decision_function", pos_label=0
    )
    assert_allclose(y_score, classifier.decision_function(X_binary) * -1)
    assert pos_label == 0


@pytest.mark.parametrize(
    "estimator, response_method",
    [
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_proba"),
        (DecisionTreeClassifier(max_depth=2, random_state=0), "predict_log_proba"),
        (LogisticRegression(), "decision_function"),
    ],
)
def test_get_response_values_multiclass(estimator, response_method):
    """Check that we can call `_get_response_values` with a multiclass estimator.
    It should return the predictions untouched.
    """
    estimator.fit(X, y)
    predictions, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )

    assert pos_label is None
    assert predictions.shape == (X.shape[0], len(estimator.classes_))
    if response_method == "predict_proba":
        assert np.logical_and(predictions >= 0, predictions <= 1).all()
    elif response_method == "predict_log_proba":
        assert (predictions <= 0.0).all()


def test_get_response_values_with_response_list():
    """Check the behaviour of passing a list of responses to `_get_response_values`."""
    classifier = LogisticRegression().fit(X_binary, y_binary)

    # it should use `predict_proba`
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["predict_proba", "decision_function"],
        return_response_method_used=True,
    )
    assert_allclose(y_pred, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1
    assert response_method == "predict_proba"

    # it should use `decision_function`
    y_pred, pos_label, response_method = _get_response_values(
        classifier,
        X_binary,
        response_method=["decision_function", "predict_proba"],
        return_response_method_used=True,
    )
    assert_allclose(y_pred, classifier.decision_function(X_binary))
    assert pos_label == 1
    assert response_method == "decision_function"


@pytest.mark.parametrize(
    "response_method", ["predict_proba", "decision_function", "predict"]
)
def test_get_response_values_multilabel_indicator(response_method):
    X, Y = make_multilabel_classification(random_state=0)
    estimator = ClassifierChain(LogisticRegression()).fit(X, Y)

    y_pred, pos_label = _get_response_values(
        estimator, X, response_method=response_method
    )
    assert pos_label is None
    assert y_pred.shape == Y.shape

    if response_method == "predict_proba":
        assert np.logical_and(y_pred >= 0, y_pred <= 1).all()
    elif response_method == "decision_function":
        # values returned by `decision_function` are not bounded in [0, 1]
        assert (y_pred < 0).sum() > 0
        assert (y_pred > 1).sum() > 0
    else:  # response_method == "predict"
        assert np.logical_or(y_pred == 0, y_pred == 1).all()
