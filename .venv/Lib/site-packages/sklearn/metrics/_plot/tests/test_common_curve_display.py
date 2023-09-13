import numpy as np
import pytest

from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_classifier(pyplot, data, data_binary, Display):
    """Check that a proper error is raised when only binary classification is
    supported."""
    X, y = data
    X_binary, y_binary = data_binary
    clf = DecisionTreeClassifier().fit(X, y)

    # Case 1: multiclass classifier with multiclass target
    msg = "Expected 'estimator' to be a binary classifier. Got 3 classes instead."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)

    # Case 2: multiclass classifier with binary target
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X_binary, y_binary)

    # Case 3: binary classifier with multiclass target
    clf = DecisionTreeClassifier().fit(X_binary, y_binary)
    msg = "The target y is not binary. Got multiclass type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_regression(pyplot, data_binary, Display):
    """Check that we raise an error with regressor."""

    # Case 1: regressor
    X, y = data_binary
    regressor = DecisionTreeRegressor().fit(X, y)

    msg = "Expected 'estimator' to be a binary classifier. Got DecisionTreeRegressor"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(regressor, X, y)

    # Case 2: regression target
    classifier = DecisionTreeClassifier().fit(X, y)
    # Force `y_true` to be seen as a regression problem
    y = y + 0.5
    msg = "The target y is not binary. Got continuous type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y)
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, regressor.fit(X, y).predict(X))


@pytest.mark.parametrize(
    "response_method, msg",
    [
        (
            "predict_proba",
            "MyClassifier has none of the following attributes: predict_proba.",
        ),
        (
            "decision_function",
            "MyClassifier has none of the following attributes: decision_function.",
        ),
        (
            "auto",
            (
                "MyClassifier has none of the following attributes: predict_proba,"
                " decision_function."
            ),
        ),
        (
            "bad_method",
            "MyClassifier has none of the following attributes: bad_method.",
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_error_no_response(
    pyplot,
    data_binary,
    response_method,
    msg,
    Display,
):
    """Check that a proper error is raised when the response method requested
    is not defined for the given trained classifier."""
    X, y = data_binary

    class MyClassifier(ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(AttributeError, match=msg):
        Display.from_estimator(clf, X, y, response_method=response_method)


@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_display_curve_estimator_name_multiple_calls(
    pyplot,
    data_binary,
    Display,
    constructor_name,
):
    """Check that passing `name` when calling `plot` will overwrite the original name
    in the legend."""
    X, y = data_binary
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    if constructor_name == "from_estimator":
        disp = Display.from_estimator(clf, X, y, name=clf_name)
    else:
        disp = Display.from_predictions(y, y_pred, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_not_fitted_errors(pyplot, data_binary, clf, Display):
    """Check that a proper error is raised when the classifier is not
    fitted."""
    X, y = data_binary
    # clone since we parametrize the test and the classifier will be fitted
    # when testing the second and subsequent plotting function
    model = clone(clf)
    with pytest.raises(NotFittedError):
        Display.from_estimator(model, X, y)
    model.fit(X, y)
    disp = Display.from_estimator(model, X, y)
    assert model.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == model.__class__.__name__


@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_n_samples_consistency(pyplot, data_binary, Display):
    """Check the error raised when `y_pred` or `sample_weight` have inconsistent
    length."""
    X, y = data_binary
    classifier = DecisionTreeClassifier().fit(X, y)

    msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X[:-2], y)
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y[:-2])
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y, sample_weight=np.ones(X.shape[0] - 2))


@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_error_pos_label(pyplot, data_binary, Display):
    """Check consistence of error message when `pos_label` should be specified."""
    X, y = data_binary
    y = y + 10

    classifier = DecisionTreeClassifier().fit(X, y)
    y_pred = classifier.predict_proba(X)[:, -1]
    msg = r"y_true takes value in {10, 11} and pos_label is not specified"
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, y_pred)
