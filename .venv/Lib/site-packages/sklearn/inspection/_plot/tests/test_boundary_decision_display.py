import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
    load_iris,
    make_classification,
    make_multilabel_classification,
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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


def test_check_boundary_response_method_auto():
    """Check _check_boundary_response_method behavior with 'auto'."""

    class A:
        def decision_function(self):
            pass

    a_inst = A()
    method = _check_boundary_response_method(a_inst, "auto")
    assert method == a_inst.decision_function

    class B:
        def predict_proba(self):
            pass

    b_inst = B()
    method = _check_boundary_response_method(b_inst, "auto")
    assert method == b_inst.predict_proba

    class C:
        def predict_proba(self):
            pass

        def decision_function(self):
            pass

    c_inst = C()
    method = _check_boundary_response_method(c_inst, "auto")
    assert method == c_inst.decision_function

    class D:
        def predict(self):
            pass

    d_inst = D()
    method = _check_boundary_response_method(d_inst, "auto")
    assert method == d_inst.predict


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_multiclass_error(pyplot, response_method):
    """Check multiclass errors."""
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    lr = LogisticRegression().fit(X, y)

    msg = (
        "Multiclass classifiers are only supported when response_method is 'predict' or"
        " 'auto'"
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
def test_decision_boundary_display(pyplot, fitted_clf, response_method, plot_method):
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

    with pytest.raises(ValueError, match=msg):
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
        DecisionBoundaryDisplay.from_estimator(tree, X)


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
