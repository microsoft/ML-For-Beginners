import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    LearningCurveDisplay,
    ValidationCurveDisplay,
    learning_curve,
    validation_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal


@pytest.fixture
def data():
    return shuffle(*load_iris(return_X_y=True), random_state=0)


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"std_display_style": "invalid"}, ValueError, "Unknown std_display_style:"),
        ({"score_type": "invalid"}, ValueError, "Unknown score_type:"),
    ],
)
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_parameters_validation(
    pyplot, data, params, err_type, err_msg, CurveDisplay, specific_params
):
    """Check that we raise a proper error when passing invalid parameters."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    with pytest.raises(err_type, match=err_msg):
        CurveDisplay.from_estimator(estimator, X, y, **specific_params, **params)


def test_learning_curve_display_default_usage(pyplot, data):
    """Check the default usage of the LearningCurveDisplay class."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    train_sizes = [0.3, 0.6, 0.9]
    display = LearningCurveDisplay.from_estimator(
        estimator, X, y, train_sizes=train_sizes
    )

    import matplotlib as mpl

    assert display.errorbar_ is None

    assert isinstance(display.lines_, list)
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)

    assert isinstance(display.fill_between_, list)
    for fill in display.fill_between_:
        assert isinstance(fill, mpl.collections.PolyCollection)
        assert fill.get_alpha() == 0.5

    assert display.score_name == "Score"
    assert display.ax_.get_xlabel() == "Number of samples in the training set"
    assert display.ax_.get_ylabel() == "Score"

    _, legend_labels = display.ax_.get_legend_handles_labels()
    assert legend_labels == ["Train", "Test"]

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes
    )

    assert_array_equal(display.train_sizes, train_sizes_abs)
    assert_allclose(display.train_scores, train_scores)
    assert_allclose(display.test_scores, test_scores)


def test_validation_curve_display_default_usage(pyplot, data):
    """Check the default usage of the ValidationCurveDisplay class."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    param_name, param_range = "max_depth", [1, 3, 5]
    display = ValidationCurveDisplay.from_estimator(
        estimator, X, y, param_name=param_name, param_range=param_range
    )

    import matplotlib as mpl

    assert display.errorbar_ is None

    assert isinstance(display.lines_, list)
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)

    assert isinstance(display.fill_between_, list)
    for fill in display.fill_between_:
        assert isinstance(fill, mpl.collections.PolyCollection)
        assert fill.get_alpha() == 0.5

    assert display.score_name == "Score"
    assert display.ax_.get_xlabel() == f"{param_name}"
    assert display.ax_.get_ylabel() == "Score"

    _, legend_labels = display.ax_.get_legend_handles_labels()
    assert legend_labels == ["Train", "Test"]

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range
    )

    assert display.param_range == param_range
    assert_array_equal(display.param_range, param_range)
    assert_allclose(display.train_scores, train_scores)
    assert_allclose(display.test_scores, test_scores)


@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_negate_score(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the `negate_score` parameter calling `from_estimator` and
    `plot`.
    """
    X, y = data
    estimator = DecisionTreeClassifier(max_depth=1, random_state=0)

    negate_score = False
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )

    positive_scores = display.lines_[0].get_data()[1]
    assert (positive_scores >= 0).all()
    assert display.ax_.get_ylabel() == "Score"

    negate_score = True
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )

    negative_scores = display.lines_[0].get_data()[1]
    assert (negative_scores <= 0).all()
    assert_allclose(negative_scores, -positive_scores)
    assert display.ax_.get_ylabel() == "Negative score"

    negate_score = False
    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, negate_score=negate_score
    )
    assert display.ax_.get_ylabel() == "Score"
    display.plot(negate_score=not negate_score)
    assert display.ax_.get_ylabel() == "Score"
    assert (display.lines_[0].get_data()[1] < 0).all()


@pytest.mark.parametrize(
    "score_name, ylabel", [(None, "Score"), ("Accuracy", "Accuracy")]
)
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_score_name(
    pyplot, data, score_name, ylabel, CurveDisplay, specific_params
):
    """Check that we can overwrite the default score name shown on the y-axis."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, score_name=score_name
    )

    assert display.ax_.get_ylabel() == ylabel
    X, y = data
    estimator = DecisionTreeClassifier(max_depth=1, random_state=0)

    display = CurveDisplay.from_estimator(
        estimator, X, y, **specific_params, score_name=score_name
    )

    assert display.score_name == ylabel


@pytest.mark.parametrize("std_display_style", (None, "errorbar"))
def test_learning_curve_display_score_type(pyplot, data, std_display_style):
    """Check the behaviour of setting the `score_type` parameter."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    train_sizes = [0.3, 0.6, 0.9]
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes
    )

    score_type = "train"
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train"]

    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    assert_array_equal(x_data, train_sizes_abs)
    assert_allclose(y_data, train_scores.mean(axis=1))

    score_type = "test"
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Test"]

    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    assert_array_equal(x_data, train_sizes_abs)
    assert_allclose(y_data, test_scores.mean(axis=1))

    score_type = "both"
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train", "Test"]

    if std_display_style is None:
        assert len(display.lines_) == 2
        assert display.errorbar_ is None
        x_data_train, y_data_train = display.lines_[0].get_data()
        x_data_test, y_data_test = display.lines_[1].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 2
        x_data_train, y_data_train = display.errorbar_[0].lines[0].get_data()
        x_data_test, y_data_test = display.errorbar_[1].lines[0].get_data()

    assert_array_equal(x_data_train, train_sizes_abs)
    assert_allclose(y_data_train, train_scores.mean(axis=1))
    assert_array_equal(x_data_test, train_sizes_abs)
    assert_allclose(y_data_test, test_scores.mean(axis=1))


@pytest.mark.parametrize("std_display_style", (None, "errorbar"))
def test_validation_curve_display_score_type(pyplot, data, std_display_style):
    """Check the behaviour of setting the `score_type` parameter."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    param_name, param_range = "max_depth", [1, 3, 5]
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range
    )

    score_type = "train"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train"]

    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    assert_array_equal(x_data, param_range)
    assert_allclose(y_data, train_scores.mean(axis=1))

    score_type = "test"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Test"]

    if std_display_style is None:
        assert len(display.lines_) == 1
        assert display.errorbar_ is None
        x_data, y_data = display.lines_[0].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 1
        x_data, y_data = display.errorbar_[0].lines[0].get_data()

    assert_array_equal(x_data, param_range)
    assert_allclose(y_data, test_scores.mean(axis=1))

    score_type = "both"
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        score_type=score_type,
        std_display_style=std_display_style,
    )

    _, legend_label = display.ax_.get_legend_handles_labels()
    assert legend_label == ["Train", "Test"]

    if std_display_style is None:
        assert len(display.lines_) == 2
        assert display.errorbar_ is None
        x_data_train, y_data_train = display.lines_[0].get_data()
        x_data_test, y_data_test = display.lines_[1].get_data()
    else:
        assert display.lines_ is None
        assert len(display.errorbar_) == 2
        x_data_train, y_data_train = display.errorbar_[0].lines[0].get_data()
        x_data_test, y_data_test = display.errorbar_[1].lines[0].get_data()

    assert_array_equal(x_data_train, param_range)
    assert_allclose(y_data_train, train_scores.mean(axis=1))
    assert_array_equal(x_data_test, param_range)
    assert_allclose(y_data_test, test_scores.mean(axis=1))


@pytest.mark.parametrize(
    "CurveDisplay, specific_params, expected_xscale",
    [
        (
            ValidationCurveDisplay,
            {"param_name": "max_depth", "param_range": np.arange(1, 5)},
            "linear",
        ),
        (LearningCurveDisplay, {"train_sizes": np.linspace(0.1, 0.9, num=5)}, "linear"),
        (
            ValidationCurveDisplay,
            {
                "param_name": "max_depth",
                "param_range": np.round(np.logspace(0, 2, num=5)).astype(np.int64),
            },
            "log",
        ),
        (LearningCurveDisplay, {"train_sizes": np.logspace(-1, 0, num=5)}, "log"),
    ],
)
def test_curve_display_xscale_auto(
    pyplot, data, CurveDisplay, specific_params, expected_xscale
):
    """Check the behaviour of the x-axis scaling depending on the data provided."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    display = CurveDisplay.from_estimator(estimator, X, y, **specific_params)
    assert display.ax_.get_xscale() == expected_xscale


@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_std_display_style(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the parameter `std_display_style`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    import matplotlib as mpl

    std_display_style = None
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
    )

    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
    assert display.errorbar_ is None
    assert display.fill_between_ is None
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2

    std_display_style = "fill_between"
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
    )

    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
    assert display.errorbar_ is None
    assert len(display.fill_between_) == 2
    for fill_between in display.fill_between_:
        assert isinstance(fill_between, mpl.collections.PolyCollection)
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2

    std_display_style = "errorbar"
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
    )

    assert display.lines_ is None
    assert len(display.errorbar_) == 2
    for errorbar in display.errorbar_:
        assert isinstance(errorbar, mpl.container.ErrorbarContainer)
    assert display.fill_between_ is None
    _, legend_label = display.ax_.get_legend_handles_labels()
    assert len(legend_label) == 2


@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
def test_curve_display_plot_kwargs(pyplot, data, CurveDisplay, specific_params):
    """Check the behaviour of the different plotting keyword arguments: `line_kw`,
    `fill_between_kw`, and `errorbar_kw`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    std_display_style = "fill_between"
    line_kw = {"color": "red"}
    fill_between_kw = {"color": "red", "alpha": 1.0}
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
        line_kw=line_kw,
        fill_between_kw=fill_between_kw,
    )

    assert display.lines_[0].get_color() == "red"
    assert_allclose(
        display.fill_between_[0].get_facecolor(),
        [[1.0, 0.0, 0.0, 1.0]],  # trust me, it's red
    )

    std_display_style = "errorbar"
    errorbar_kw = {"color": "red"}
    display = CurveDisplay.from_estimator(
        estimator,
        X,
        y,
        **specific_params,
        std_display_style=std_display_style,
        errorbar_kw=errorbar_kw,
    )

    assert display.errorbar_[0].lines[0].get_color() == "red"


# TODO(1.5): to be removed
def test_learning_curve_display_deprecate_log_scale(data, pyplot):
    """Check that we warn for the deprecated parameter `log_scale`."""
    X, y = data
    estimator = DecisionTreeClassifier(random_state=0)

    with pytest.warns(FutureWarning, match="`log_scale` parameter is deprecated"):
        display = LearningCurveDisplay.from_estimator(
            estimator, X, y, train_sizes=[0.3, 0.6, 0.9], log_scale=True
        )

    assert display.ax_.get_xscale() == "log"
    assert display.ax_.get_yscale() == "linear"

    with pytest.warns(FutureWarning, match="`log_scale` parameter is deprecated"):
        display = LearningCurveDisplay.from_estimator(
            estimator, X, y, train_sizes=[0.3, 0.6, 0.9], log_scale=False
        )

    assert display.ax_.get_xscale() == "linear"
    assert display.ax_.get_yscale() == "linear"
