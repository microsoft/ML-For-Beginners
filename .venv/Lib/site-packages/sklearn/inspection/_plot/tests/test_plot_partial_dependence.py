import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles

from sklearn.compose import make_column_transformer
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    (
        "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
        "matplotlib.*"
    ),
)


@pytest.fixture(scope="module")
def diabetes():
    # diabetes dataset, subsampled for speed
    data = load_diabetes()
    data.data = data.data[:50]
    data.target = data.target[:50]
    return data


@pytest.fixture(scope="module")
def clf_diabetes(diabetes):
    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    clf.fit(diabetes.data, diabetes.target)
    return clf


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("grid_resolution", [10, 20])
def test_plot_partial_dependence(grid_resolution, pyplot, clf_diabetes, diabetes):
    # Test partial dependence plot function.
    # Use columns 0 & 2 as 1 is not quantitative (sex)
    feature_names = diabetes.feature_names
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2, (0, 2)],
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        contour_kw={"cmap": "jet"},
    )
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert disp.figure_ is fig
    assert len(axs) == 4

    assert disp.bounding_ax_ is not None
    assert disp.axes_.shape == (1, 3)
    assert disp.lines_.shape == (1, 3)
    assert disp.contours_.shape == (1, 3)
    assert disp.deciles_vlines_.shape == (1, 3)
    assert disp.deciles_hlines_.shape == (1, 3)

    assert disp.lines_[0, 2] is None
    assert disp.contours_[0, 0] is None
    assert disp.contours_[0, 1] is None

    # deciles lines: always show on xaxis, only show on yaxis if 2-way PDP
    for i in range(3):
        assert disp.deciles_vlines_[0, i] is not None
    assert disp.deciles_hlines_[0, 0] is None
    assert disp.deciles_hlines_[0, 1] is None
    assert disp.deciles_hlines_[0, 2] is not None

    assert disp.features == [(0,), (2,), (0, 2)]
    assert np.all(disp.feature_names == feature_names)
    assert len(disp.deciles) == 2
    for i in [0, 2]:
        assert_allclose(
            disp.deciles[i],
            mquantiles(diabetes.data[:, i], prob=np.arange(0.1, 1.0, 0.1)),
        )

    single_feature_positions = [(0, (0, 0)), (2, (0, 1))]
    expected_ylabels = ["Partial dependence", ""]

    for i, (feat_col, pos) in enumerate(single_feature_positions):
        ax = disp.axes_[pos]
        assert ax.get_ylabel() == expected_ylabels[i]
        assert ax.get_xlabel() == diabetes.feature_names[feat_col]

        line = disp.lines_[pos]

        avg_preds = disp.pd_results[i]
        assert avg_preds.average.shape == (1, grid_resolution)
        target_idx = disp.target_idx

        line_data = line.get_data()
        assert_allclose(line_data[0], avg_preds["grid_values"][0])
        assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # two feature position
    ax = disp.axes_[0, 2]
    coutour = disp.contours_[0, 2]
    assert coutour.get_cmap().name == "jet"
    assert ax.get_xlabel() == diabetes.feature_names[0]
    assert ax.get_ylabel() == diabetes.feature_names[2]


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "kind, centered, subsample, shape",
    [
        ("average", False, None, (1, 3)),
        ("individual", False, None, (1, 3, 50)),
        ("both", False, None, (1, 3, 51)),
        ("individual", False, 20, (1, 3, 20)),
        ("both", False, 20, (1, 3, 21)),
        ("individual", False, 0.5, (1, 3, 25)),
        ("both", False, 0.5, (1, 3, 26)),
        ("average", True, None, (1, 3)),
        ("individual", True, None, (1, 3, 50)),
        ("both", True, None, (1, 3, 51)),
        ("individual", True, 20, (1, 3, 20)),
        ("both", True, 20, (1, 3, 21)),
    ],
)
def test_plot_partial_dependence_kind(
    pyplot,
    kind,
    centered,
    subsample,
    shape,
    clf_diabetes,
    diabetes,
):
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1, 2],
        kind=kind,
        centered=centered,
        subsample=subsample,
    )

    assert disp.axes_.shape == (1, 3)
    assert disp.lines_.shape == shape
    assert disp.contours_.shape == (1, 3)

    assert disp.contours_[0, 0] is None
    assert disp.contours_[0, 1] is None
    assert disp.contours_[0, 2] is None

    if centered:
        assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])
    else:
        assert all([ln._y[0] != 0.0 for ln in disp.lines_.ravel() if ln is not None])


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "input_type, feature_names_type",
    [
        ("dataframe", None),
        ("dataframe", "list"),
        ("list", "list"),
        ("array", "list"),
        ("dataframe", "array"),
        ("list", "array"),
        ("array", "array"),
        ("dataframe", "series"),
        ("list", "series"),
        ("array", "series"),
        ("dataframe", "index"),
        ("list", "index"),
        ("array", "index"),
    ],
)
def test_plot_partial_dependence_str_features(
    pyplot,
    clf_diabetes,
    diabetes,
    input_type,
    feature_names_type,
):
    if input_type == "dataframe":
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    elif input_type == "list":
        X = diabetes.data.tolist()
    else:
        X = diabetes.data

    if feature_names_type is None:
        feature_names = None
    else:
        feature_names = _convert_container(diabetes.feature_names, feature_names_type)

    grid_resolution = 25
    # check with str features and array feature names and single column
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        X,
        [("age", "bmi"), "bmi"],
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        n_cols=1,
        line_kw={"alpha": 0.8},
    )
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert len(axs) == 3

    assert disp.figure_ is fig
    assert disp.axes_.shape == (2, 1)
    assert disp.lines_.shape == (2, 1)
    assert disp.contours_.shape == (2, 1)
    assert disp.deciles_vlines_.shape == (2, 1)
    assert disp.deciles_hlines_.shape == (2, 1)

    assert disp.lines_[0, 0] is None
    assert disp.deciles_vlines_[0, 0] is not None
    assert disp.deciles_hlines_[0, 0] is not None
    assert disp.contours_[1, 0] is None
    assert disp.deciles_hlines_[1, 0] is None
    assert disp.deciles_vlines_[1, 0] is not None

    # line
    ax = disp.axes_[1, 0]
    assert ax.get_xlabel() == "bmi"
    assert ax.get_ylabel() == "Partial dependence"

    line = disp.lines_[1, 0]
    avg_preds = disp.pd_results[1]
    target_idx = disp.target_idx
    assert line.get_alpha() == 0.8

    line_data = line.get_data()
    assert_allclose(line_data[0], avg_preds["grid_values"][0])
    assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # contour
    ax = disp.axes_[0, 0]
    assert ax.get_xlabel() == "age"
    assert ax.get_ylabel() == "bmi"


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_custom_axes(pyplot, clf_diabetes, diabetes):
    grid_resolution = 25
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", ("age", "bmi")],
        grid_resolution=grid_resolution,
        feature_names=diabetes.feature_names,
        ax=[ax1, ax2],
    )
    assert fig is disp.figure_
    assert disp.bounding_ax_ is None
    assert disp.axes_.shape == (2,)
    assert disp.axes_[0] is ax1
    assert disp.axes_[1] is ax2

    ax = disp.axes_[0]
    assert ax.get_xlabel() == "age"
    assert ax.get_ylabel() == "Partial dependence"

    line = disp.lines_[0]
    avg_preds = disp.pd_results[0]
    target_idx = disp.target_idx

    line_data = line.get_data()
    assert_allclose(line_data[0], avg_preds["grid_values"][0])
    assert_allclose(line_data[1], avg_preds.average[target_idx].ravel())

    # contour
    ax = disp.axes_[1]
    assert ax.get_xlabel() == "age"
    assert ax.get_ylabel() == "bmi"


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "kind, lines", [("average", 1), ("individual", 50), ("both", 51)]
)
def test_plot_partial_dependence_passing_numpy_axes(
    pyplot, clf_diabetes, diabetes, kind, lines
):
    grid_resolution = 25
    feature_names = diabetes.feature_names
    disp1 = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
    )
    assert disp1.axes_.shape == (1, 2)
    assert disp1.axes_[0, 0].get_ylabel() == "Partial dependence"
    assert disp1.axes_[0, 1].get_ylabel() == ""
    assert len(disp1.axes_[0, 0].get_lines()) == lines
    assert len(disp1.axes_[0, 1].get_lines()) == lines

    lr = LinearRegression()
    lr.fit(diabetes.data, diabetes.target)

    disp2 = PartialDependenceDisplay.from_estimator(
        lr,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        ax=disp1.axes_,
    )

    assert np.all(disp1.axes_ == disp2.axes_)
    assert len(disp2.axes_[0, 0].get_lines()) == 2 * lines
    assert len(disp2.axes_[0, 1].get_lines()) == 2 * lines


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("nrows, ncols", [(2, 2), (3, 1)])
def test_plot_partial_dependence_incorrent_num_axes(
    pyplot, clf_diabetes, diabetes, nrows, ncols
):
    grid_resolution = 5
    fig, axes = pyplot.subplots(nrows, ncols)
    axes_formats = [list(axes.ravel()), tuple(axes.ravel()), axes]

    msg = "Expected ax to have 2 axes, got {}".format(nrows * ncols)

    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        grid_resolution=grid_resolution,
        feature_names=diabetes.feature_names,
    )

    for ax_format in axes_formats:
        with pytest.raises(ValueError, match=msg):
            PartialDependenceDisplay.from_estimator(
                clf_diabetes,
                diabetes.data,
                ["age", "bmi"],
                grid_resolution=grid_resolution,
                feature_names=diabetes.feature_names,
                ax=ax_format,
            )

        # with axes object
        with pytest.raises(ValueError, match=msg):
            disp.plot(ax=ax_format)


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_with_same_axes(pyplot, clf_diabetes, diabetes):
    # The first call to plot_partial_dependence will create two new axes to
    # place in the space of the passed in axes, which results in a total of
    # three axes in the figure.
    # Currently the API does not allow for the second call to
    # plot_partial_dependence to use the same axes again, because it will
    # create two new axes in the space resulting in five axes. To get the
    # expected behavior one needs to pass the generated axes into the second
    # call:
    # disp1 = plot_partial_dependence(...)
    # disp2 = plot_partial_dependence(..., ax=disp1.axes_)

    grid_resolution = 25
    fig, ax = pyplot.subplots()
    PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        grid_resolution=grid_resolution,
        feature_names=diabetes.feature_names,
        ax=ax,
    )

    msg = (
        "The ax was already used in another plot function, please set "
        "ax=display.axes_ instead"
    )

    with pytest.raises(ValueError, match=msg):
        PartialDependenceDisplay.from_estimator(
            clf_diabetes,
            diabetes.data,
            ["age", "bmi"],
            grid_resolution=grid_resolution,
            feature_names=diabetes.feature_names,
            ax=ax,
        )


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_feature_name_reuse(pyplot, clf_diabetes, diabetes):
    # second call to plot does not change the feature names from the first
    # call

    feature_names = diabetes.feature_names
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        grid_resolution=10,
        feature_names=feature_names,
    )

    PartialDependenceDisplay.from_estimator(
        clf_diabetes, diabetes.data, [0, 1], grid_resolution=10, ax=disp.axes_
    )

    for i, ax in enumerate(disp.axes_.ravel()):
        assert ax.get_xlabel() == feature_names[i]


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_multiclass(pyplot):
    grid_resolution = 25
    clf_int = GradientBoostingClassifier(n_estimators=10, random_state=1)
    iris = load_iris()

    # Test partial dependence plot function on multi-class input.
    clf_int.fit(iris.data, iris.target)
    disp_target_0 = PartialDependenceDisplay.from_estimator(
        clf_int, iris.data, [0, 3], target=0, grid_resolution=grid_resolution
    )
    assert disp_target_0.figure_ is pyplot.gcf()
    assert disp_target_0.axes_.shape == (1, 2)
    assert disp_target_0.lines_.shape == (1, 2)
    assert disp_target_0.contours_.shape == (1, 2)
    assert disp_target_0.deciles_vlines_.shape == (1, 2)
    assert disp_target_0.deciles_hlines_.shape == (1, 2)
    assert all(c is None for c in disp_target_0.contours_.flat)
    assert disp_target_0.target_idx == 0

    # now with symbol labels
    target = iris.target_names[iris.target]
    clf_symbol = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf_symbol.fit(iris.data, target)
    disp_symbol = PartialDependenceDisplay.from_estimator(
        clf_symbol, iris.data, [0, 3], target="setosa", grid_resolution=grid_resolution
    )
    assert disp_symbol.figure_ is pyplot.gcf()
    assert disp_symbol.axes_.shape == (1, 2)
    assert disp_symbol.lines_.shape == (1, 2)
    assert disp_symbol.contours_.shape == (1, 2)
    assert disp_symbol.deciles_vlines_.shape == (1, 2)
    assert disp_symbol.deciles_hlines_.shape == (1, 2)
    assert all(c is None for c in disp_symbol.contours_.flat)
    assert disp_symbol.target_idx == 0

    for int_result, symbol_result in zip(
        disp_target_0.pd_results, disp_symbol.pd_results
    ):
        assert_allclose(int_result.average, symbol_result.average)
        assert_allclose(int_result["grid_values"], symbol_result["grid_values"])

    # check that the pd plots are different for another target
    disp_target_1 = PartialDependenceDisplay.from_estimator(
        clf_int, iris.data, [0, 3], target=1, grid_resolution=grid_resolution
    )
    target_0_data_y = disp_target_0.lines_[0, 0].get_data()[1]
    target_1_data_y = disp_target_1.lines_[0, 0].get_data()[1]
    assert any(target_0_data_y != target_1_data_y)


multioutput_regression_data = make_regression(n_samples=50, n_targets=2, random_state=0)


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize("target", [0, 1])
def test_plot_partial_dependence_multioutput(pyplot, target):
    # Test partial dependence plot function on multi-output input.
    X, y = multioutput_regression_data
    clf = LinearRegression().fit(X, y)

    grid_resolution = 25
    disp = PartialDependenceDisplay.from_estimator(
        clf, X, [0, 1], target=target, grid_resolution=grid_resolution
    )
    fig = pyplot.gcf()
    axs = fig.get_axes()
    assert len(axs) == 3
    assert disp.target_idx == target
    assert disp.bounding_ax_ is not None

    positions = [(0, 0), (0, 1)]
    expected_label = ["Partial dependence", ""]

    for i, pos in enumerate(positions):
        ax = disp.axes_[pos]
        assert ax.get_ylabel() == expected_label[i]
        assert ax.get_xlabel() == f"x{i}"


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
def test_plot_partial_dependence_dataframe(pyplot, clf_diabetes, diabetes):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    grid_resolution = 25

    PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        df,
        ["bp", "s1"],
        grid_resolution=grid_resolution,
        feature_names=df.columns.tolist(),
    )


dummy_classification_data = make_classification(random_state=0)


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "data, params, err_msg",
    [
        (
            multioutput_regression_data,
            {"target": None, "features": [0]},
            "target must be specified for multi-output",
        ),
        (
            multioutput_regression_data,
            {"target": -1, "features": [0]},
            r"target must be in \[0, n_tasks\]",
        ),
        (
            multioutput_regression_data,
            {"target": 100, "features": [0]},
            r"target must be in \[0, n_tasks\]",
        ),
        (
            dummy_classification_data,
            {"features": ["foobar"], "feature_names": None},
            "Feature 'foobar' not in feature_names",
        ),
        (
            dummy_classification_data,
            {"features": ["foobar"], "feature_names": ["abcd", "def"]},
            "Feature 'foobar' not in feature_names",
        ),
        (
            dummy_classification_data,
            {"features": [(1, 2, 3)]},
            "Each entry in features must be either an int, ",
        ),
        (
            dummy_classification_data,
            {"features": [1, {}]},
            "Each entry in features must be either an int, ",
        ),
        (
            dummy_classification_data,
            {"features": [tuple()]},
            "Each entry in features must be either an int, ",
        ),
        (
            dummy_classification_data,
            {"features": [123], "feature_names": ["blahblah"]},
            "All entries of features must be less than ",
        ),
        (
            dummy_classification_data,
            {"features": [0, 1, 2], "feature_names": ["a", "b", "a"]},
            "feature_names should not contain duplicates",
        ),
        (
            dummy_classification_data,
            {"features": [1, 2], "kind": ["both"]},
            "When `kind` is provided as a list of strings, it should contain",
        ),
        (
            dummy_classification_data,
            {"features": [1], "subsample": -1},
            "When an integer, subsample=-1 should be positive.",
        ),
        (
            dummy_classification_data,
            {"features": [1], "subsample": 1.2},
            r"When a floating-point, subsample=1.2 should be in the \(0, 1\) range",
        ),
        (
            dummy_classification_data,
            {"features": [1, 2], "categorical_features": [1.0, 2.0]},
            "Expected `categorical_features` to be an array-like of boolean,",
        ),
        (
            dummy_classification_data,
            {"features": [(1, 2)], "categorical_features": [2]},
            "Two-way partial dependence plots are not supported for pairs",
        ),
        (
            dummy_classification_data,
            {"features": [1], "categorical_features": [1], "kind": "individual"},
            "It is not possible to display individual effects",
        ),
    ],
)
def test_plot_partial_dependence_error(pyplot, data, params, err_msg):
    X, y = data
    estimator = LinearRegression().fit(X, y)

    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(estimator, X, **params)


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"target": 4, "features": [0]}, "target not in est.classes_, got 4"),
        ({"target": None, "features": [0]}, "target must be specified for multi-class"),
        (
            {"target": 1, "features": [4.5]},
            "Each entry in features must be either an int,",
        ),
    ],
)
def test_plot_partial_dependence_multiclass_error(pyplot, params, err_msg):
    iris = load_iris()
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf.fit(iris.data, iris.target)

    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(clf, iris.data, **params)


def test_plot_partial_dependence_does_not_override_ylabel(
    pyplot, clf_diabetes, diabetes
):
    # Non-regression test to be sure to not override the ylabel if it has been
    # See https://github.com/scikit-learn/scikit-learn/issues/15772
    _, axes = pyplot.subplots(1, 2)
    axes[0].set_ylabel("Hello world")
    PartialDependenceDisplay.from_estimator(
        clf_diabetes, diabetes.data, [0, 1], ax=axes
    )

    assert axes[0].get_ylabel() == "Hello world"
    assert axes[1].get_ylabel() == "Partial dependence"


@pytest.mark.parametrize(
    "categorical_features, array_type",
    [
        (["col_A", "col_C"], "dataframe"),
        ([0, 2], "array"),
        ([True, False, True], "array"),
    ],
)
def test_plot_partial_dependence_with_categorical(
    pyplot, categorical_features, array_type
):
    X = [[1, 1, "A"], [2, 0, "C"], [3, 2, "B"]]
    column_name = ["col_A", "col_B", "col_C"]
    X = _convert_container(X, array_type, columns_name=column_name)
    y = np.array([1.2, 0.5, 0.45]).T

    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)

    # single feature
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=["col_C"],
        feature_names=column_name,
        categorical_features=categorical_features,
    )

    assert disp.figure_ is pyplot.gcf()
    assert disp.bars_.shape == (1, 1)
    assert disp.bars_[0][0] is not None
    assert disp.lines_.shape == (1, 1)
    assert disp.lines_[0][0] is None
    assert disp.contours_.shape == (1, 1)
    assert disp.contours_[0][0] is None
    assert disp.deciles_vlines_.shape == (1, 1)
    assert disp.deciles_vlines_[0][0] is None
    assert disp.deciles_hlines_.shape == (1, 1)
    assert disp.deciles_hlines_[0][0] is None
    assert disp.axes_[0, 0].get_legend() is None

    # interaction between two features
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[("col_A", "col_C")],
        feature_names=column_name,
        categorical_features=categorical_features,
    )

    assert disp.figure_ is pyplot.gcf()
    assert disp.bars_.shape == (1, 1)
    assert disp.bars_[0][0] is None
    assert disp.lines_.shape == (1, 1)
    assert disp.lines_[0][0] is None
    assert disp.contours_.shape == (1, 1)
    assert disp.contours_[0][0] is None
    assert disp.deciles_vlines_.shape == (1, 1)
    assert disp.deciles_vlines_[0][0] is None
    assert disp.deciles_hlines_.shape == (1, 1)
    assert disp.deciles_hlines_[0][0] is None
    assert disp.axes_[0, 0].get_legend() is None


def test_plot_partial_dependence_legend(pyplot):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "col_A": ["A", "B", "C"],
            "col_B": [1, 0, 2],
            "col_C": ["C", "B", "A"],
        }
    )
    y = np.array([1.2, 0.5, 0.45]).T

    categorical_features = ["col_A", "col_C"]
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)

    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=["col_B", "col_C"],
        categorical_features=categorical_features,
        kind=["both", "average"],
    )

    legend_text = disp.axes_[0, 0].get_legend().get_texts()
    assert len(legend_text) == 1
    assert legend_text[0].get_text() == "average"
    assert disp.axes_[0, 1].get_legend() is None


@pytest.mark.parametrize(
    "kind, expected_shape",
    [("average", (1, 2)), ("individual", (1, 2, 20)), ("both", (1, 2, 21))],
)
def test_plot_partial_dependence_subsampling(
    pyplot, clf_diabetes, diabetes, kind, expected_shape
):
    # check that the subsampling is properly working
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/pull/18359
    matplotlib = pytest.importorskip("matplotlib")
    grid_resolution = 25
    feature_names = diabetes.feature_names

    disp1 = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        ["age", "bmi"],
        kind=kind,
        grid_resolution=grid_resolution,
        feature_names=feature_names,
        subsample=20,
        random_state=0,
    )

    assert disp1.lines_.shape == expected_shape
    assert all(
        [isinstance(line, matplotlib.lines.Line2D) for line in disp1.lines_.ravel()]
    )


@pytest.mark.parametrize(
    "kind, line_kw, label",
    [
        ("individual", {}, None),
        ("individual", {"label": "xxx"}, None),
        ("average", {}, None),
        ("average", {"label": "xxx"}, "xxx"),
        ("both", {}, "average"),
        ("both", {"label": "xxx"}, "xxx"),
    ],
)
def test_partial_dependence_overwrite_labels(
    pyplot,
    clf_diabetes,
    diabetes,
    kind,
    line_kw,
    label,
):
    """Test that make sure that we can overwrite the label of the PDP plot"""
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2],
        grid_resolution=25,
        feature_names=diabetes.feature_names,
        kind=kind,
        line_kw=line_kw,
    )

    for ax in disp.axes_.ravel():
        if label is None:
            assert ax.get_legend() is None
        else:
            legend_text = ax.get_legend().get_texts()
            assert len(legend_text) == 1
            assert legend_text[0].get_text() == label


@pytest.mark.parametrize(
    "categorical_features, array_type",
    [
        (["col_A", "col_C"], "dataframe"),
        ([0, 2], "array"),
        ([True, False, True], "array"),
    ],
)
def test_grid_resolution_with_categorical(pyplot, categorical_features, array_type):
    """Check that we raise a ValueError when the grid_resolution is too small
    respect to the number of categories in the categorical features targeted.
    """
    X = [["A", 1, "A"], ["B", 0, "C"], ["C", 2, "B"]]
    column_name = ["col_A", "col_B", "col_C"]
    X = _convert_container(X, array_type, columns_name=column_name)
    y = np.array([1.2, 0.5, 0.45]).T

    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)

    err_msg = (
        "resolution of the computed grid is less than the minimum number of categories"
    )
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=["col_C"],
            feature_names=column_name,
            categorical_features=categorical_features,
            grid_resolution=2,
        )


@pytest.mark.parametrize("kind", ["individual", "average", "both"])
@pytest.mark.parametrize("centered", [True, False])
def test_partial_dependence_plot_limits_one_way(
    pyplot, clf_diabetes, diabetes, kind, centered
):
    """Check that the PD limit on the plots are properly set on one-way plots."""
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=(0, 1),
        kind=kind,
        grid_resolution=25,
        feature_names=diabetes.feature_names,
    )

    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        if "average" in pd:
            pd["average"][...] = range_pd[1]
            pd["average"][0, 0] = range_pd[0]
        if "individual" in pd:
            pd["individual"][...] = range_pd[1]
            pd["individual"][0, 0, 0] = range_pd[0]

    disp.plot(centered=centered)
    # check that we anchor to zero x-axis when centering
    y_lim = range_pd - range_pd[0] if centered else range_pd
    padding = 0.05 * (y_lim[1] - y_lim[0])
    y_lim[0] -= padding
    y_lim[1] += padding
    for ax in disp.axes_.ravel():
        assert_allclose(ax.get_ylim(), y_lim)


@pytest.mark.parametrize("centered", [True, False])
def test_partial_dependence_plot_limits_two_way(
    pyplot, clf_diabetes, diabetes, centered
):
    """Check that the PD limit on the plots are properly set on two-way plots."""
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[(0, 1)],
        kind="average",
        grid_resolution=25,
        feature_names=diabetes.feature_names,
    )

    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        pd["average"][...] = range_pd[1]
        pd["average"][0, 0] = range_pd[0]

    disp.plot(centered=centered)
    contours = disp.contours_[0, 0]
    levels = range_pd - range_pd[0] if centered else range_pd

    padding = 0.05 * (levels[1] - levels[0])
    levels[0] -= padding
    levels[1] += padding
    expect_levels = np.linspace(*levels, num=8)
    assert_allclose(contours.levels, expect_levels)


def test_partial_dependence_kind_list(
    pyplot,
    clf_diabetes,
    diabetes,
):
    """Check that we can provide a list of strings to kind parameter."""
    matplotlib = pytest.importorskip("matplotlib")

    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[0, 2, (1, 2)],
        grid_resolution=20,
        kind=["both", "both", "average"],
    )

    for idx in [0, 1]:
        assert all(
            [
                isinstance(line, matplotlib.lines.Line2D)
                for line in disp.lines_[0, idx].ravel()
            ]
        )
        assert disp.contours_[0, idx] is None

    assert disp.contours_[0, 2] is not None
    assert all([line is None for line in disp.lines_[0, 2].ravel()])


@pytest.mark.parametrize(
    "features, kind",
    [
        ([0, 2, (1, 2)], "individual"),
        ([0, 2, (1, 2)], "both"),
        ([(0, 1), (0, 2), (1, 2)], "individual"),
        ([(0, 1), (0, 2), (1, 2)], "both"),
        ([0, 2, (1, 2)], ["individual", "individual", "individual"]),
        ([0, 2, (1, 2)], ["both", "both", "both"]),
    ],
)
def test_partial_dependence_kind_error(
    pyplot,
    clf_diabetes,
    diabetes,
    features,
    kind,
):
    """Check that we raise an informative error when 2-way PD is requested
    together with 1-way PD/ICE"""
    warn_msg = (
        "ICE plot cannot be rendered for 2-way feature interactions. 2-way "
        "feature interactions mandates PD plots using the 'average' kind"
    )
    with pytest.raises(ValueError, match=warn_msg):
        PartialDependenceDisplay.from_estimator(
            clf_diabetes,
            diabetes.data,
            features=features,
            grid_resolution=20,
            kind=kind,
        )


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "line_kw, pd_line_kw, ice_lines_kw, expected_colors",
    [
        ({"color": "r"}, {"color": "g"}, {"color": "b"}, ("g", "b")),
        (None, {"color": "g"}, {"color": "b"}, ("g", "b")),
        ({"color": "r"}, None, {"color": "b"}, ("r", "b")),
        ({"color": "r"}, {"color": "g"}, None, ("g", "r")),
        ({"color": "r"}, None, None, ("r", "r")),
        ({"color": "r"}, {"linestyle": "--"}, {"linestyle": "-."}, ("r", "r")),
    ],
)
def test_plot_partial_dependence_lines_kw(
    pyplot,
    clf_diabetes,
    diabetes,
    line_kw,
    pd_line_kw,
    ice_lines_kw,
    expected_colors,
):
    """Check that passing `pd_line_kw` and `ice_lines_kw` will act on the
    specific lines in the plot.
    """

    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 2],
        grid_resolution=20,
        feature_names=diabetes.feature_names,
        n_cols=2,
        kind="both",
        line_kw=line_kw,
        pd_line_kw=pd_line_kw,
        ice_lines_kw=ice_lines_kw,
    )

    line = disp.lines_[0, 0, -1]
    assert line.get_color() == expected_colors[0]
    if pd_line_kw is not None and "linestyle" in pd_line_kw:
        assert line.get_linestyle() == pd_line_kw["linestyle"]
    else:
        assert line.get_linestyle() == "--"

    line = disp.lines_[0, 0, 0]
    assert line.get_color() == expected_colors[1]
    if ice_lines_kw is not None and "linestyle" in ice_lines_kw:
        assert line.get_linestyle() == ice_lines_kw["linestyle"]
    else:
        assert line.get_linestyle() == "-"


def test_partial_dependence_display_wrong_len_kind(
    pyplot,
    clf_diabetes,
    diabetes,
):
    """Check that we raise an error when `kind` is a list with a wrong length.

    This case can only be triggered using the `PartialDependenceDisplay.from_estimator`
    method.
    """
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        features=[0, 2],
        grid_resolution=20,
        kind="average",  # len(kind) != len(features)
    )

    # alter `kind` to be a list with a length different from length of `features`
    disp.kind = ["average"]
    err_msg = (
        r"When `kind` is provided as a list of strings, it should contain as many"
        r" elements as `features`. `kind` contains 1 element\(s\) and `features`"
        r" contains 2 element\(s\)."
    )
    with pytest.raises(ValueError, match=err_msg):
        disp.plot()


@pytest.mark.parametrize(
    "kind",
    ["individual", "both", "average", ["average", "both"], ["individual", "both"]],
)
def test_partial_dependence_display_kind_centered_interaction(
    pyplot,
    kind,
    clf_diabetes,
    diabetes,
):
    """Check that we properly center ICE and PD when passing kind as a string and as a
    list."""
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        kind=kind,
        centered=True,
        subsample=5,
    )

    assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])


def test_partial_dependence_display_with_constant_sample_weight(
    pyplot,
    clf_diabetes,
    diabetes,
):
    """Check that the utilization of a constant sample weight maintains the
    standard behavior.
    """
    disp = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        kind="average",
        method="brute",
    )

    sample_weight = np.ones_like(diabetes.target)
    disp_sw = PartialDependenceDisplay.from_estimator(
        clf_diabetes,
        diabetes.data,
        [0, 1],
        sample_weight=sample_weight,
        kind="average",
        method="brute",
    )

    assert np.array_equal(
        disp.pd_results[0]["average"], disp_sw.pd_results[0]["average"]
    )
