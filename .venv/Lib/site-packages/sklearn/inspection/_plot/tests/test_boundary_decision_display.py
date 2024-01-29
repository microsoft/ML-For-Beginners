import warnings

import numpy as np
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_classification,
    make_multilabel_classification,
)
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
)

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)


X, y = make_classification(
    n_informative=1,
    n_redundant=1,
    n_clusters_per_class=1,
    n_features=2,
    random_state=42,
)


def load_iris_2d_scaled():
    X, y = load_iris(return_X_y=True)
    X = scale(X)[:, :2]
    return X, y


@pytest.fixture(scope="module")
def fitted_clf():
    return LogisticRegression().fit(X, y)


def test_input_data_dimension(pyplot):
    """Check that we raise an error when `X` does not have exactly 2 features."""
    X, y = make_classification(n_samples=10, n_features=4, random_state=0)

    clf = LogisticRegression().fit(X, y)
    msg = "n_features must be equal to 2. Got 4 instead."
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(estimator=clf, X=X)


def test_check_boundary_response_method_error():
    """Check that we raise an error for the cases not supported by
    `_check_boundary_response_method`.
    """

    class MultiLabelClassifier:
        classes_ = [np.array([0, 1]), np.array([0, 1])]

    err_msg = "Multi-label and multi-output multi-class classifiers are not supported"
    with pytest.raises(ValueError, match=err_msg):
        _check_boundary_response_method(MultiLabelClassifier(), "predict", None)

    class MulticlassClassifier:
        classes_ = [0, 1, 2]

    err_msg = "Multiclass classifiers are only supported when `response_method` is"
    for response_method in ("predict_proba", "decision_function"):
        with pytest.raises(ValueError, match=err_msg):
            _check_boundary_response_method(
                MulticlassClassifier(), response_method, None
            )


@pytest.mark.parametrize(
    "estimator, response_method, class_of_interest, expected_prediction_method",
    [
        (DecisionTreeRegressor(), "predict", None, "predict"),
        (DecisionTreeRegressor(), "auto", None, "predict"),
        (LogisticRegression().fit(*load_iris_2d_scaled()), "predict", None, "predict"),
        (LogisticRegression().fit(*load_iris_2d_scaled()), "auto", None, "predict"),
        (
            LogisticRegression().fit(*load_iris_2d_scaled()),
            "predict_proba",
            0,
            "predict_proba",
        ),
        (
            LogisticRegression().fit(*load_iris_2d_scaled()),
            "decision_function",
            0,
            "decision_function",
        ),
        (
            LogisticRegression().fit(X, y),
            "auto",
            None,
            ["decision_function", "predict_proba", "predict"],
        ),
        (LogisticRegression().fit(X, y), "predict", None, "predict"),
        (
            LogisticRegression().fit(X, y),
            ["predict_proba", "decision_function"],
            None,
            ["predict_proba", "decision_function"],
        ),
    ],
)
def test_check_boundary_response_method(
    estimator, response_method, class_of_interest, expected_prediction_method
):
    """Check the behaviour of `_check_boundary_response_method` for the supported
    cases.
    """
    prediction_method = _check_boundary_response_method(
        estimator, response_method, class_of_interest
    )
    assert prediction_method == expected_prediction_method


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_multiclass_error(pyplot, response_method):
    """Check multiclass errors."""
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    lr = LogisticRegression().fit(X, y)

    msg = (
        "Multiclass classifiers are only supported when `response_method` is 'predict'"
        " or 'auto'"
    )
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(lr, X, response_method=response_method)


@pytest.mark.parametrize("response_method", ["auto", "predict"])
def test_multiclass(pyplot, response_method):
    """Check multiclass gives expected results."""
    grid_resolution = 10
    eps = 1.0
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    lr = LogisticRegression(random_state=0).fit(X, y)

    disp = DecisionBoundaryDisplay.from_estimator(
        lr, X, response_method=response_method, grid_resolution=grid_resolution, eps=1.0
    )

    x0_min, x0_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    x1_min, x1_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )
    response = lr.predict(np.c_[xx0.ravel(), xx1.ravel()])
    assert_allclose(disp.response, response.reshape(xx0.shape))
    assert_allclose(disp.xx0, xx0)
    assert_allclose(disp.xx1, xx1)


@pytest.mark.parametrize(
    "kwargs, error_msg",
    [
        (
            {"plot_method": "hello_world"},
            r"plot_method must be one of contourf, contour, pcolormesh. Got hello_world"
            r" instead.",
        ),
        (
            {"grid_resolution": 1},
            r"grid_resolution must be greater than 1. Got 1 instead",
        ),
        (
            {"grid_resolution": -1},
            r"grid_resolution must be greater than 1. Got -1 instead",
        ),
        ({"eps": -1.1}, r"eps must be greater than or equal to 0. Got -1.1 instead"),
    ],
)
def test_input_validation_errors(pyplot, kwargs, error_msg, fitted_clf):
    """Check input validation from_estimator."""
    with pytest.raises(ValueError, match=error_msg):
        DecisionBoundaryDisplay.from_estimator(fitted_clf, X, **kwargs)


def test_display_plot_input_error(pyplot, fitted_clf):
    """Check input validation for `plot`."""
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, X, grid_resolution=5)

    with pytest.raises(ValueError, match="plot_method must be 'contourf'"):
        disp.plot(plot_method="hello_world")


@pytest.mark.parametrize(
    "response_method", ["auto", "predict", "predict_proba", "decision_function"]
)
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_classifier(
    pyplot, fitted_clf, response_method, plot_method
):
    """Check that decision boundary is correct."""
    fig, ax = pyplot.subplots()
    eps = 2.0
    disp = DecisionBoundaryDisplay.from_estimator(
        fitted_clf,
        X,
        grid_resolution=5,
        response_method=response_method,
        plot_method=plot_method,
        eps=eps,
        ax=ax,
    )
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    assert disp.ax_ == ax
    assert disp.figure_ == fig

    x0, x1 = X[:, 0], X[:, 1]

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    assert disp.xx0.min() == pytest.approx(x0_min)
    assert disp.xx0.max() == pytest.approx(x0_max)
    assert disp.xx1.min() == pytest.approx(x1_min)
    assert disp.xx1.max() == pytest.approx(x1_max)

    fig2, ax2 = pyplot.subplots()
    # change plotting method for second plot
    disp.plot(plot_method="pcolormesh", ax=ax2, shading="auto")
    assert isinstance(disp.surface_, pyplot.matplotlib.collections.QuadMesh)
    assert disp.ax_ == ax2
    assert disp.figure_ == fig2


@pytest.mark.parametrize("response_method", ["auto", "predict", "decision_function"])
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_outlier_detector(
    pyplot, response_method, plot_method
):
    """Check that decision boundary is correct for outlier detector."""
    fig, ax = pyplot.subplots()
    eps = 2.0
    outlier_detector = IsolationForest(random_state=0).fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(
        outlier_detector,
        X,
        grid_resolution=5,
        response_method=response_method,
        plot_method=plot_method,
        eps=eps,
        ax=ax,
    )
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    assert disp.ax_ == ax
    assert disp.figure_ == fig

    x0, x1 = X[:, 0], X[:, 1]

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    assert disp.xx0.min() == pytest.approx(x0_min)
    assert disp.xx0.max() == pytest.approx(x0_max)
    assert disp.xx1.min() == pytest.approx(x1_min)
    assert disp.xx1.max() == pytest.approx(x1_max)


@pytest.mark.parametrize("response_method", ["auto", "predict"])
@pytest.mark.parametrize("plot_method", ["contourf", "contour"])
def test_decision_boundary_display_regressor(pyplot, response_method, plot_method):
    """Check that we can display the decision boundary for a regressor."""
    X, y = load_diabetes(return_X_y=True)
    X = X[:, :2]
    tree = DecisionTreeRegressor().fit(X, y)
    fig, ax = pyplot.subplots()
    eps = 2.0
    disp = DecisionBoundaryDisplay.from_estimator(
        tree,
        X,
        response_method=response_method,
        ax=ax,
        eps=eps,
        plot_method=plot_method,
    )
    assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
    assert disp.ax_ == ax
    assert disp.figure_ == fig

    x0, x1 = X[:, 0], X[:, 1]

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    assert disp.xx0.min() == pytest.approx(x0_min)
    assert disp.xx0.max() == pytest.approx(x0_max)
    assert disp.xx1.min() == pytest.approx(x1_min)
    assert disp.xx1.max() == pytest.approx(x1_max)

    fig2, ax2 = pyplot.subplots()
    # change plotting method for second plot
    disp.plot(plot_method="pcolormesh", ax=ax2, shading="auto")
    assert isinstance(disp.surface_, pyplot.matplotlib.collections.QuadMesh)
    assert disp.ax_ == ax2
    assert disp.figure_ == fig2


@pytest.mark.parametrize(
    "response_method, msg",
    [
        (
            "predict_proba",
            "MyClassifier has none of the following attributes: predict_proba",
        ),
        (
            "decision_function",
            "MyClassifier has none of the following attributes: decision_function",
        ),
        (
            "auto",
            (
                "MyClassifier has none of the following attributes: decision_function, "
                "predict_proba, predict"
            ),
        ),
        (
            "bad_method",
            "MyClassifier has none of the following attributes: bad_method",
        ),
    ],
)
def test_error_bad_response(pyplot, response_method, msg):
    """Check errors for bad response."""

    class MyClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.fitted_ = True
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(AttributeError, match=msg):
        DecisionBoundaryDisplay.from_estimator(clf, X, response_method=response_method)


@pytest.mark.parametrize("response_method", ["auto", "predict", "predict_proba"])
def test_multilabel_classifier_error(pyplot, response_method):
    """Check that multilabel classifier raises correct error."""
    X, y = make_multilabel_classification(random_state=0)
    X = X[:, :2]
    tree = DecisionTreeClassifier().fit(X, y)

    msg = "Multi-label and multi-output multi-class classifiers are not supported"
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(
            tree,
            X,
            response_method=response_method,
        )


@pytest.mark.parametrize("response_method", ["auto", "predict", "predict_proba"])
def test_multi_output_multi_class_classifier_error(pyplot, response_method):
    """Check that multi-output multi-class classifier raises correct error."""
    X = np.asarray([[0, 1], [1, 2]])
    y = np.asarray([["tree", "cat"], ["cat", "tree"]])
    tree = DecisionTreeClassifier().fit(X, y)

    msg = "Multi-label and multi-output multi-class classifiers are not supported"
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(
            tree,
            X,
            response_method=response_method,
        )


def test_multioutput_regressor_error(pyplot):
    """Check that multioutput regressor raises correct error."""
    X = np.asarray([[0, 1], [1, 2]])
    y = np.asarray([[0, 1], [4, 1]])
    tree = DecisionTreeRegressor().fit(X, y)
    with pytest.raises(ValueError, match="Multi-output regressors are not supported"):
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method="predict")


@pytest.mark.parametrize(
    "response_method",
    ["predict_proba", "decision_function", ["predict_proba", "predict"]],
)
def test_regressor_unsupported_response(pyplot, response_method):
    """Check that we can display the decision boundary for a regressor."""
    X, y = load_diabetes(return_X_y=True)
    X = X[:, :2]
    tree = DecisionTreeRegressor().fit(X, y)
    err_msg = "should either be a classifier to be used with response_method"
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method=response_method)


@pytest.mark.filterwarnings(
    # We expect to raise the following warning because the classifier is fit on a
    # NumPy array
    "ignore:X has feature names, but LogisticRegression was fitted without"
)
def test_dataframe_labels_used(pyplot, fitted_clf):
    """Check that column names are used for pandas."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(X, columns=["col_x", "col_y"])

    # pandas column names are used by default
    _, ax = pyplot.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, df, ax=ax)
    assert ax.get_xlabel() == "col_x"
    assert ax.get_ylabel() == "col_y"

    # second call to plot will have the names
    fig, ax = pyplot.subplots()
    disp.plot(ax=ax)
    assert ax.get_xlabel() == "col_x"
    assert ax.get_ylabel() == "col_y"

    # axes with a label will not get overridden
    fig, ax = pyplot.subplots()
    ax.set(xlabel="hello", ylabel="world")
    disp.plot(ax=ax)
    assert ax.get_xlabel() == "hello"
    assert ax.get_ylabel() == "world"

    # labels get overridden only if provided to the `plot` method
    disp.plot(ax=ax, xlabel="overwritten_x", ylabel="overwritten_y")
    assert ax.get_xlabel() == "overwritten_x"
    assert ax.get_ylabel() == "overwritten_y"

    # labels do not get inferred if provided to `from_estimator`
    _, ax = pyplot.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(
        fitted_clf, df, ax=ax, xlabel="overwritten_x", ylabel="overwritten_y"
    )
    assert ax.get_xlabel() == "overwritten_x"
    assert ax.get_ylabel() == "overwritten_y"


def test_string_target(pyplot):
    """Check that decision boundary works with classifiers trained on string labels."""
    iris = load_iris()
    X = iris.data[:, [0, 1]]

    # Use strings as target
    y = iris.target_names[iris.target]
    log_reg = LogisticRegression().fit(X, y)

    # Does not raise
    DecisionBoundaryDisplay.from_estimator(
        log_reg,
        X,
        grid_resolution=5,
        response_method="predict",
    )


def test_dataframe_support(pyplot):
    """Check that passing a dataframe at fit and to the Display does not
    raise warnings.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23311
    """
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(X, columns=["col_x", "col_y"])
    estimator = LogisticRegression().fit(df, y)

    with warnings.catch_warnings():
        # no warnings linked to feature names validation should be raised
        warnings.simplefilter("error", UserWarning)
        DecisionBoundaryDisplay.from_estimator(estimator, df, response_method="predict")


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_class_of_interest_binary(pyplot, response_method):
    """Check the behaviour of passing `class_of_interest` for plotting the output of
    `predict_proba` and `decision_function` in the binary case.
    """
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    assert_array_equal(np.unique(y), [0, 1])

    estimator = LogisticRegression().fit(X, y)
    # We will check that `class_of_interest=None` is equivalent to
    # `class_of_interest=estimator.classes_[1]`
    disp_default = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=None,
    )
    disp_class_1 = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=estimator.classes_[1],
    )

    assert_allclose(disp_default.response, disp_class_1.response)

    # we can check that `_get_response_values` modifies the response when targeting
    # the other class, i.e. 1 - p(y=1|x) for `predict_proba` and -decision_function
    # for `decision_function`.
    disp_class_0 = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=estimator.classes_[0],
    )

    if response_method == "predict_proba":
        assert_allclose(disp_default.response, 1 - disp_class_0.response)
    else:
        assert response_method == "decision_function"
        assert_allclose(disp_default.response, -disp_class_0.response)


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_class_of_interest_multiclass(pyplot, response_method):
    """Check the behaviour of passing `class_of_interest` for plotting the output of
    `predict_proba` and `decision_function` in the multiclass case.
    """
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target  # the target are numerical labels
    class_of_interest_idx = 2

    estimator = LogisticRegression().fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=class_of_interest_idx,
    )

    # we will check that we plot the expected values as response
    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)

    # make the same test but this time using target as strings
    y = iris.target_names[iris.target]
    estimator = LogisticRegression().fit(X, y)

    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method=response_method,
        class_of_interest=iris.target_names[class_of_interest_idx],
    )

    grid = np.concatenate([disp.xx0.reshape(-1, 1), disp.xx1.reshape(-1, 1)], axis=1)
    response = getattr(estimator, response_method)(grid)[:, class_of_interest_idx]
    assert_allclose(response.reshape(*disp.response.shape), disp.response)

    # check that we raise an error for unknown labels
    # this test should already be handled in `_get_response_values` but we can have this
    # test here as well
    err_msg = "class_of_interest=2 is not a valid label: It should be one of"
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X,
            response_method=response_method,
            class_of_interest=class_of_interest_idx,
        )

    # TODO: remove this test when we handle multiclass with class_of_interest=None
    # by showing the max of the decision function or the max of the predicted
    # probabilities.
    err_msg = "Multiclass classifiers are only supported"
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X,
            response_method=response_method,
            class_of_interest=None,
        )


def test_subclass_named_constructors_return_type_is_subclass(pyplot):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    clf = LogisticRegression().fit(X, y)

    class SubclassOfDisplay(DecisionBoundaryDisplay):
        pass

    curve = SubclassOfDisplay.from_estimator(estimator=clf, X=X)

    assert isinstance(curve, SubclassOfDisplay)
