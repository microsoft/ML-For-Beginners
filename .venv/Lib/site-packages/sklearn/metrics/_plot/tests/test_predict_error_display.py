import pytest
from numpy.testing import assert_allclose

from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay

X, y = load_diabetes(return_X_y=True)


@pytest.fixture
def regressor_fitted():
    return Ridge().fit(X, y)


@pytest.mark.parametrize(
    "regressor, params, err_type, err_msg",
    [
        (
            Ridge().fit(X, y),
            {"subsample": -1},
            ValueError,
            "When an integer, subsample=-1 should be",
        ),
        (
            Ridge().fit(X, y),
            {"subsample": 20.0},
            ValueError,
            "When a floating-point, subsample=20.0 should be",
        ),
        (
            Ridge().fit(X, y),
            {"subsample": -20.0},
            ValueError,
            "When a floating-point, subsample=-20.0 should be",
        ),
        (
            Ridge().fit(X, y),
            {"kind": "xxx"},
            ValueError,
            "`kind` must be one of",
        ),
    ],
)
@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
def test_prediction_error_display_raise_error(
    pyplot, class_method, regressor, params, err_type, err_msg
):
    """Check that we raise the proper error when making the parameters
    # validation."""
    with pytest.raises(err_type, match=err_msg):
        if class_method == "from_estimator":
            PredictionErrorDisplay.from_estimator(regressor, X, y, **params)
        else:
            y_pred = regressor.predict(X)
            PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, **params)


def test_from_estimator_not_fitted(pyplot):
    """Check that we raise a `NotFittedError` when the passed regressor is not
    fit."""
    regressor = Ridge()
    with pytest.raises(NotFittedError, match="is not fitted yet."):
        PredictionErrorDisplay.from_estimator(regressor, X, y)


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("kind", ["actual_vs_predicted", "residual_vs_predicted"])
def test_prediction_error_display(pyplot, regressor_fitted, class_method, kind):
    """Check the default behaviour of the display."""
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, kind=kind
        )
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, kind=kind
        )

    if kind == "actual_vs_predicted":
        assert_allclose(display.line_.get_xdata(), display.line_.get_ydata())
        assert display.ax_.get_xlabel() == "Predicted values"
        assert display.ax_.get_ylabel() == "Actual values"
        assert display.line_ is not None
    else:
        assert display.ax_.get_xlabel() == "Predicted values"
        assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"
        assert display.line_ is not None

    assert display.ax_.get_legend() is None


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize(
    "subsample, expected_size",
    [(5, 5), (0.1, int(X.shape[0] * 0.1)), (None, X.shape[0])],
)
def test_plot_prediction_error_subsample(
    pyplot, regressor_fitted, class_method, subsample, expected_size
):
    """Check the behaviour of `subsample`."""
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, subsample=subsample
        )
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, subsample=subsample
        )
    assert len(display.scatter_.get_offsets()) == expected_size


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
def test_plot_prediction_error_ax(pyplot, regressor_fitted, class_method):
    """Check that we can pass an axis to the display."""
    _, ax = pyplot.subplots()
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y, ax=ax)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, ax=ax
        )
    assert display.ax_ is ax


@pytest.mark.parametrize("class_method", ["from_estimator", "from_predictions"])
def test_prediction_error_custom_artist(pyplot, regressor_fitted, class_method):
    """Check that we can tune the style of the lines."""
    extra_params = {
        "kind": "actual_vs_predicted",
        "scatter_kwargs": {"color": "red"},
        "line_kwargs": {"color": "black"},
    }
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(
            regressor_fitted, X, y, **extra_params
        )
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, **extra_params
        )

    assert display.line_.get_color() == "black"
    assert_allclose(display.scatter_.get_edgecolor(), [[1.0, 0.0, 0.0, 0.8]])

    # create a display with the default values
    if class_method == "from_estimator":
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred)
    pyplot.close("all")

    display.plot(**extra_params)
    assert display.line_.get_color() == "black"
    assert_allclose(display.scatter_.get_edgecolor(), [[1.0, 0.0, 0.0, 0.8]])
