"""
Testing for the partial dependence module.
"""
import warnings

import numpy as np
import pytest

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
    _grid_from_X,
    _partial_dependence_brute,
    _partial_dependence_recursion,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    scale,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]


# (X, y), n_targets  <-- as expected in the output of partial_dep()
binary_classification_data = (make_classification(n_samples=50, random_state=0), 1)
multiclass_classification_data = (
    make_classification(
        n_samples=50, n_classes=3, n_clusters_per_class=1, random_state=0
    ),
    3,
)
regression_data = (make_regression(n_samples=50, random_state=0), 1)
multioutput_regression_data = (
    make_regression(n_samples=50, n_targets=2, random_state=0),
    2,
)

# iris
iris = load_iris()


@pytest.mark.parametrize(
    "Estimator, method, data",
    [
        (GradientBoostingClassifier, "auto", binary_classification_data),
        (GradientBoostingClassifier, "auto", multiclass_classification_data),
        (GradientBoostingClassifier, "brute", binary_classification_data),
        (GradientBoostingClassifier, "brute", multiclass_classification_data),
        (GradientBoostingRegressor, "auto", regression_data),
        (GradientBoostingRegressor, "brute", regression_data),
        (DecisionTreeRegressor, "brute", regression_data),
        (LinearRegression, "brute", regression_data),
        (LinearRegression, "brute", multioutput_regression_data),
        (LogisticRegression, "brute", binary_classification_data),
        (LogisticRegression, "brute", multiclass_classification_data),
        (MultiTaskLasso, "brute", multioutput_regression_data),
    ],
)
@pytest.mark.parametrize("grid_resolution", (5, 10))
@pytest.mark.parametrize("features", ([1], [1, 2]))
@pytest.mark.parametrize("kind", ("average", "individual", "both"))
def test_output_shape(Estimator, method, data, grid_resolution, features, kind):
    # Check that partial_dependence has consistent output shape for different
    # kinds of estimators:
    # - classifiers with binary and multiclass settings
    # - regressors
    # - multi-task regressors

    est = Estimator()
    if hasattr(est, "n_estimators"):
        est.set_params(n_estimators=2)  # speed-up computations

    # n_target corresponds to the number of classes (1 for binary classif) or
    # the number of tasks / outputs in multi task settings. It's equal to 1 for
    # classical regression_data.
    (X, y), n_targets = data
    n_instances = X.shape[0]

    est.fit(X, y)
    result = partial_dependence(
        est,
        X=X,
        features=features,
        method=method,
        kind=kind,
        grid_resolution=grid_resolution,
    )
    pdp, axes = result, result["grid_values"]

    expected_pdp_shape = (n_targets, *[grid_resolution for _ in range(len(features))])
    expected_ice_shape = (
        n_targets,
        n_instances,
        *[grid_resolution for _ in range(len(features))],
    )
    if kind == "average":
        assert pdp.average.shape == expected_pdp_shape
    elif kind == "individual":
        assert pdp.individual.shape == expected_ice_shape
    else:  # 'both'
        assert pdp.average.shape == expected_pdp_shape
        assert pdp.individual.shape == expected_ice_shape

    expected_axes_shape = (len(features), grid_resolution)
    assert axes is not None
    assert np.asarray(axes).shape == expected_axes_shape


def test_grid_from_X():
    # tests for _grid_from_X: sanity check for output, and for shapes.

    # Make sure that the grid is a cartesian product of the input (it will use
    # the unique values instead of the percentiles)
    percentiles = (0.05, 0.95)
    grid_resolution = 100
    is_categorical = [False, False]
    X = np.asarray([[1, 2], [3, 4]])
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution)
    assert_array_equal(grid, [[1, 2], [1, 4], [3, 2], [3, 4]])
    assert_array_equal(axes, X.T)

    # test shapes of returned objects depending on the number of unique values
    # for a feature.
    rng = np.random.RandomState(0)
    grid_resolution = 15

    # n_unique_values > grid_resolution
    X = rng.normal(size=(20, 2))
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    assert grid.shape == (grid_resolution * grid_resolution, X.shape[1])
    assert np.asarray(axes).shape == (2, grid_resolution)

    # n_unique_values < grid_resolution, will use actual values
    n_unique_values = 12
    X[n_unique_values - 1 :, 0] = 12345
    rng.shuffle(X)  # just to make sure the order is irrelevant
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    assert grid.shape == (n_unique_values * grid_resolution, X.shape[1])
    # axes is a list of arrays of different shapes
    assert axes[0].shape == (n_unique_values,)
    assert axes[1].shape == (grid_resolution,)


@pytest.mark.parametrize(
    "grid_resolution",
    [
        2,  # since n_categories > 2, we should not use quantiles resampling
        100,
    ],
)
def test_grid_from_X_with_categorical(grid_resolution):
    """Check that `_grid_from_X` always sample from categories and does not
    depend from the percentiles.
    """
    pd = pytest.importorskip("pandas")
    percentiles = (0.05, 0.95)
    is_categorical = [True]
    X = pd.DataFrame({"cat_feature": ["A", "B", "C", "A", "B", "D", "E"]})
    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    assert grid.shape == (5, X.shape[1])
    assert axes[0].shape == (5,)


@pytest.mark.parametrize("grid_resolution", [3, 100])
def test_grid_from_X_heterogeneous_type(grid_resolution):
    """Check that `_grid_from_X` always sample from categories and does not
    depend from the percentiles.
    """
    pd = pytest.importorskip("pandas")
    percentiles = (0.05, 0.95)
    is_categorical = [True, False]
    X = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "A", "B", "D", "E", "A", "B", "D"],
            "num": [1, 1, 1, 2, 5, 6, 6, 6, 6, 8],
        }
    )
    nunique = X.nunique()

    grid, axes = _grid_from_X(
        X, percentiles, is_categorical, grid_resolution=grid_resolution
    )
    if grid_resolution == 3:
        assert grid.shape == (15, 2)
        assert axes[0].shape[0] == nunique["num"]
        assert axes[1].shape[0] == grid_resolution
    else:
        assert grid.shape == (25, 2)
        assert axes[0].shape[0] == nunique["cat"]
        assert axes[1].shape[0] == nunique["cat"]


@pytest.mark.parametrize(
    "grid_resolution, percentiles, err_msg",
    [
        (2, (0, 0.0001), "percentiles are too close"),
        (100, (1, 2, 3, 4), "'percentiles' must be a sequence of 2 elements"),
        (100, 12345, "'percentiles' must be a sequence of 2 elements"),
        (100, (-1, 0.95), r"'percentiles' values must be in \[0, 1\]"),
        (100, (0.05, 2), r"'percentiles' values must be in \[0, 1\]"),
        (100, (0.9, 0.1), r"percentiles\[0\] must be strictly less than"),
        (1, (0.05, 0.95), "'grid_resolution' must be strictly greater than 1"),
    ],
)
def test_grid_from_X_error(grid_resolution, percentiles, err_msg):
    X = np.asarray([[1, 2], [3, 4]])
    is_categorical = [False]
    with pytest.raises(ValueError, match=err_msg):
        _grid_from_X(X, percentiles, is_categorical, grid_resolution)


@pytest.mark.parametrize("target_feature", range(5))
@pytest.mark.parametrize(
    "est, method",
    [
        (LinearRegression(), "brute"),
        (GradientBoostingRegressor(random_state=0), "brute"),
        (GradientBoostingRegressor(random_state=0), "recursion"),
        (HistGradientBoostingRegressor(random_state=0), "brute"),
        (HistGradientBoostingRegressor(random_state=0), "recursion"),
    ],
)
def test_partial_dependence_helpers(est, method, target_feature):
    # Check that what is returned by _partial_dependence_brute or
    # _partial_dependence_recursion is equivalent to manually setting a target
    # feature to a given value, and computing the average prediction over all
    # samples.
    # This also checks that the brute and recursion methods give the same
    # output.
    # Note that even on the trainset, the brute and the recursion methods
    # aren't always strictly equivalent, in particular when the slow method
    # generates unrealistic samples that have low mass in the joint
    # distribution of the input features, and when some of the features are
    # dependent. Hence the high tolerance on the checks.

    X, y = make_regression(random_state=0, n_features=5, n_informative=5)
    # The 'init' estimator for GBDT (here the average prediction) isn't taken
    # into account with the recursion method, for technical reasons. We set
    # the mean to 0 to that this 'bug' doesn't have any effect.
    y = y - y.mean()
    est.fit(X, y)

    # target feature will be set to .5 and then to 123
    features = np.array([target_feature], dtype=np.int32)
    grid = np.array([[0.5], [123]])

    if method == "brute":
        pdp, predictions = _partial_dependence_brute(
            est, grid, features, X, response_method="auto"
        )
    else:
        pdp = _partial_dependence_recursion(est, grid, features)

    mean_predictions = []
    for val in (0.5, 123):
        X_ = X.copy()
        X_[:, target_feature] = val
        mean_predictions.append(est.predict(X_).mean())

    pdp = pdp[0]  # (shape is (1, 2) so make it (2,))

    # allow for greater margin for error with recursion method
    rtol = 1e-1 if method == "recursion" else 1e-3
    assert np.allclose(pdp, mean_predictions, rtol=rtol)


@pytest.mark.parametrize("seed", range(1))
def test_recursion_decision_tree_vs_forest_and_gbdt(seed):
    # Make sure that the recursion method gives the same results on a
    # DecisionTreeRegressor and a GradientBoostingRegressor or a
    # RandomForestRegressor with 1 tree and equivalent parameters.

    rng = np.random.RandomState(seed)

    # Purely random dataset to avoid correlated features
    n_samples = 1000
    n_features = 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples) * 10

    # The 'init' estimator for GBDT (here the average prediction) isn't taken
    # into account with the recursion method, for technical reasons. We set
    # the mean to 0 to that this 'bug' doesn't have any effect.
    y = y - y.mean()

    # set max_depth not too high to avoid splits with same gain but different
    # features
    max_depth = 5

    tree_seed = 0
    forest = RandomForestRegressor(
        n_estimators=1,
        max_features=None,
        bootstrap=False,
        max_depth=max_depth,
        random_state=tree_seed,
    )
    # The forest will use ensemble.base._set_random_states to set the
    # random_state of the tree sub-estimator. We simulate this here to have
    # equivalent estimators.
    equiv_random_state = check_random_state(tree_seed).randint(np.iinfo(np.int32).max)
    gbdt = GradientBoostingRegressor(
        n_estimators=1,
        learning_rate=1,
        criterion="squared_error",
        max_depth=max_depth,
        random_state=equiv_random_state,
    )
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=equiv_random_state)

    forest.fit(X, y)
    gbdt.fit(X, y)
    tree.fit(X, y)

    # sanity check: if the trees aren't the same, the PD values won't be equal
    try:
        assert_is_subtree(tree.tree_, gbdt[0, 0].tree_)
        assert_is_subtree(tree.tree_, forest[0].tree_)
    except AssertionError:
        # For some reason the trees aren't exactly equal on 32bits, so the PDs
        # cannot be equal either. See
        # https://github.com/scikit-learn/scikit-learn/issues/8853
        assert _IS_32BIT, "this should only fail on 32 bit platforms"
        return

    grid = rng.randn(50).reshape(-1, 1)
    for f in range(n_features):
        features = np.array([f], dtype=np.int32)

        pdp_forest = _partial_dependence_recursion(forest, grid, features)
        pdp_gbdt = _partial_dependence_recursion(gbdt, grid, features)
        pdp_tree = _partial_dependence_recursion(tree, grid, features)

        np.testing.assert_allclose(pdp_gbdt, pdp_tree)
        np.testing.assert_allclose(pdp_forest, pdp_tree)


@pytest.mark.parametrize(
    "est",
    (
        GradientBoostingClassifier(random_state=0),
        HistGradientBoostingClassifier(random_state=0),
    ),
)
@pytest.mark.parametrize("target_feature", (0, 1, 2, 3, 4, 5))
def test_recursion_decision_function(est, target_feature):
    # Make sure the recursion method (implicitly uses decision_function) has
    # the same result as using brute method with
    # response_method=decision_function

    X, y = make_classification(n_classes=2, n_clusters_per_class=1, random_state=1)
    assert np.mean(y) == 0.5  # make sure the init estimator predicts 0 anyway

    est.fit(X, y)

    preds_1 = partial_dependence(
        est,
        X,
        [target_feature],
        response_method="decision_function",
        method="recursion",
        kind="average",
    )
    preds_2 = partial_dependence(
        est,
        X,
        [target_feature],
        response_method="decision_function",
        method="brute",
        kind="average",
    )

    assert_allclose(preds_1["average"], preds_2["average"], atol=1e-7)


@pytest.mark.parametrize(
    "est",
    (
        LinearRegression(),
        GradientBoostingRegressor(random_state=0),
        HistGradientBoostingRegressor(
            random_state=0, min_samples_leaf=1, max_leaf_nodes=None, max_iter=1
        ),
        DecisionTreeRegressor(random_state=0),
    ),
)
@pytest.mark.parametrize("power", (1, 2))
def test_partial_dependence_easy_target(est, power):
    # If the target y only depends on one feature in an obvious way (linear or
    # quadratic) then the partial dependence for that feature should reflect
    # it.
    # We here fit a linear regression_data model (with polynomial features if
    # needed) and compute r_squared to check that the partial dependence
    # correctly reflects the target.

    rng = np.random.RandomState(0)
    n_samples = 200
    target_variable = 2
    X = rng.normal(size=(n_samples, 5))
    y = X[:, target_variable] ** power

    est.fit(X, y)

    pdp = partial_dependence(
        est, features=[target_variable], X=X, grid_resolution=1000, kind="average"
    )

    new_X = pdp["grid_values"][0].reshape(-1, 1)
    new_y = pdp["average"][0]
    # add polynomial features if needed
    new_X = PolynomialFeatures(degree=power).fit_transform(new_X)

    lr = LinearRegression().fit(new_X, new_y)
    r2 = r2_score(new_y, lr.predict(new_X))

    assert r2 > 0.99


@pytest.mark.parametrize(
    "Estimator",
    (
        sklearn.tree.DecisionTreeClassifier,
        sklearn.tree.ExtraTreeClassifier,
        sklearn.ensemble.ExtraTreesClassifier,
        sklearn.neighbors.KNeighborsClassifier,
        sklearn.neighbors.RadiusNeighborsClassifier,
        sklearn.ensemble.RandomForestClassifier,
    ),
)
def test_multiclass_multioutput(Estimator):
    # Make sure error is raised for multiclass-multioutput classifiers

    # make multiclass-multioutput dataset
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=0)
    y = np.array([y, y]).T

    est = Estimator()
    est.fit(X, y)

    with pytest.raises(
        ValueError, match="Multiclass-multioutput estimators are not supported"
    ):
        partial_dependence(est, X, [0])


class NoPredictProbaNoDecisionFunction(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        # simulate that we have some classes
        self.classes_ = [0, 1]
        return self


@pytest.mark.filterwarnings("ignore:A Bunch will be returned")
@pytest.mark.parametrize(
    "estimator, params, err_msg",
    [
        (
            KMeans(random_state=0, n_init="auto"),
            {"features": [0]},
            "'estimator' must be a fitted regressor or classifier",
        ),
        (
            LinearRegression(),
            {"features": [0], "response_method": "predict_proba"},
            "The response_method parameter is ignored for regressors",
        ),
        (
            GradientBoostingClassifier(random_state=0),
            {
                "features": [0],
                "response_method": "predict_proba",
                "method": "recursion",
            },
            "'recursion' method, the response_method must be 'decision_function'",
        ),
        (
            GradientBoostingClassifier(random_state=0),
            {"features": [0], "response_method": "predict_proba", "method": "auto"},
            "'recursion' method, the response_method must be 'decision_function'",
        ),
        (
            LinearRegression(),
            {"features": [0], "method": "recursion", "kind": "individual"},
            "The 'recursion' method only applies when 'kind' is set to 'average'",
        ),
        (
            LinearRegression(),
            {"features": [0], "method": "recursion", "kind": "both"},
            "The 'recursion' method only applies when 'kind' is set to 'average'",
        ),
        (
            LinearRegression(),
            {"features": [0], "method": "recursion"},
            "Only the following estimators support the 'recursion' method:",
        ),
    ],
)
def test_partial_dependence_error(estimator, params, err_msg):
    X, y = make_classification(random_state=0)
    estimator.fit(X, y)

    with pytest.raises(ValueError, match=err_msg):
        partial_dependence(estimator, X, **params)


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
@pytest.mark.parametrize("features", [-1, 10000])
def test_partial_dependence_unknown_feature_indices(estimator, features):
    X, y = make_classification(random_state=0)
    estimator.fit(X, y)

    err_msg = "all features must be in"
    with pytest.raises(ValueError, match=err_msg):
        partial_dependence(estimator, X, [features])


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
def test_partial_dependence_unknown_feature_string(estimator):
    pd = pytest.importorskip("pandas")
    X, y = make_classification(random_state=0)
    df = pd.DataFrame(X)
    estimator.fit(df, y)

    features = ["random"]
    err_msg = "A given column is not a column of the dataframe"
    with pytest.raises(ValueError, match=err_msg):
        partial_dependence(estimator, df, features)


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), GradientBoostingClassifier(random_state=0)]
)
def test_partial_dependence_X_list(estimator):
    # check that array-like objects are accepted
    X, y = make_classification(random_state=0)
    estimator.fit(X, y)
    partial_dependence(estimator, list(X), [0], kind="average")


def test_warning_recursion_non_constant_init():
    # make sure that passing a non-constant init parameter to a GBDT and using
    # recursion method yields a warning.

    gbc = GradientBoostingClassifier(init=DummyClassifier(), random_state=0)
    gbc.fit(X, y)

    with pytest.warns(
        UserWarning, match="Using recursion method with a non-constant init predictor"
    ):
        partial_dependence(gbc, X, [0], method="recursion", kind="average")

    with pytest.warns(
        UserWarning, match="Using recursion method with a non-constant init predictor"
    ):
        partial_dependence(gbc, X, [0], method="recursion", kind="average")


def test_partial_dependence_sample_weight_of_fitted_estimator():
    # Test near perfect correlation between partial dependence and diagonal
    # when sample weights emphasize y = x predictions
    # non-regression test for #13193
    # TODO: extend to HistGradientBoosting once sample_weight is supported
    N = 1000
    rng = np.random.RandomState(123456)
    mask = rng.randint(2, size=N, dtype=bool)

    x = rng.rand(N)
    # set y = x on mask and y = -x outside
    y = x.copy()
    y[~mask] = -y[~mask]
    X = np.c_[mask, x]
    # sample weights to emphasize data points where y = x
    sample_weight = np.ones(N)
    sample_weight[mask] = 1000.0

    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    clf.fit(X, y, sample_weight=sample_weight)

    pdp = partial_dependence(clf, X, features=[1], kind="average")

    assert np.corrcoef(pdp["average"], pdp["grid_values"])[0, 1] > 0.99


def test_hist_gbdt_sw_not_supported():
    # TODO: remove/fix when PDP supports HGBT with sample weights
    clf = HistGradientBoostingRegressor(random_state=1)
    clf.fit(X, y, sample_weight=np.ones(len(X)))

    with pytest.raises(
        NotImplementedError, match="does not support partial dependence"
    ):
        partial_dependence(clf, X, features=[1])


def test_partial_dependence_pipeline():
    # check that the partial dependence support pipeline
    iris = load_iris()

    scaler = StandardScaler()
    clf = DummyClassifier(random_state=42)
    pipe = make_pipeline(scaler, clf)

    clf.fit(scaler.fit_transform(iris.data), iris.target)
    pipe.fit(iris.data, iris.target)

    features = 0
    pdp_pipe = partial_dependence(
        pipe, iris.data, features=[features], grid_resolution=10, kind="average"
    )
    pdp_clf = partial_dependence(
        clf,
        scaler.transform(iris.data),
        features=[features],
        grid_resolution=10,
        kind="average",
    )
    assert_allclose(pdp_pipe["average"], pdp_clf["average"])
    assert_allclose(
        pdp_pipe["grid_values"][0],
        pdp_clf["grid_values"][0] * scaler.scale_[features] + scaler.mean_[features],
    )


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=1000, random_state=0),
        GradientBoostingClassifier(random_state=0, n_estimators=5),
    ],
    ids=["estimator-brute", "estimator-recursion"],
)
@pytest.mark.parametrize(
    "preprocessor",
    [
        None,
        make_column_transformer(
            (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
            (RobustScaler(), [iris.feature_names[i] for i in (1, 3)]),
        ),
        make_column_transformer(
            (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
            remainder="passthrough",
        ),
    ],
    ids=["None", "column-transformer", "column-transformer-passthrough"],
)
@pytest.mark.parametrize(
    "features",
    [[0, 2], [iris.feature_names[i] for i in (0, 2)]],
    ids=["features-integer", "features-string"],
)
def test_partial_dependence_dataframe(estimator, preprocessor, features):
    # check that the partial dependence support dataframe and pipeline
    # including a column transformer
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(scale(iris.data), columns=iris.feature_names)

    pipe = make_pipeline(preprocessor, estimator)
    pipe.fit(df, iris.target)
    pdp_pipe = partial_dependence(
        pipe, df, features=features, grid_resolution=10, kind="average"
    )

    # the column transformer will reorder the column when transforming
    # we mixed the index to be sure that we are computing the partial
    # dependence of the right columns
    if preprocessor is not None:
        X_proc = clone(preprocessor).fit_transform(df)
        features_clf = [0, 1]
    else:
        X_proc = df
        features_clf = [0, 2]

    clf = clone(estimator).fit(X_proc, iris.target)
    pdp_clf = partial_dependence(
        clf,
        X_proc,
        features=features_clf,
        method="brute",
        grid_resolution=10,
        kind="average",
    )

    assert_allclose(pdp_pipe["average"], pdp_clf["average"])
    if preprocessor is not None:
        scaler = preprocessor.named_transformers_["standardscaler"]
        assert_allclose(
            pdp_pipe["grid_values"][1],
            pdp_clf["grid_values"][1] * scaler.scale_[1] + scaler.mean_[1],
        )
    else:
        assert_allclose(pdp_pipe["grid_values"][1], pdp_clf["grid_values"][1])


@pytest.mark.parametrize(
    "features, expected_pd_shape",
    [
        (0, (3, 10)),
        (iris.feature_names[0], (3, 10)),
        ([0, 2], (3, 10, 10)),
        ([iris.feature_names[i] for i in (0, 2)], (3, 10, 10)),
        ([True, False, True, False], (3, 10, 10)),
    ],
    ids=["scalar-int", "scalar-str", "list-int", "list-str", "mask"],
)
def test_partial_dependence_feature_type(features, expected_pd_shape):
    # check all possible features type supported in PDP
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    preprocessor = make_column_transformer(
        (StandardScaler(), [iris.feature_names[i] for i in (0, 2)]),
        (RobustScaler(), [iris.feature_names[i] for i in (1, 3)]),
    )
    pipe = make_pipeline(
        preprocessor, LogisticRegression(max_iter=1000, random_state=0)
    )
    pipe.fit(df, iris.target)
    pdp_pipe = partial_dependence(
        pipe, df, features=features, grid_resolution=10, kind="average"
    )
    assert pdp_pipe["average"].shape == expected_pd_shape
    assert len(pdp_pipe["grid_values"]) == len(pdp_pipe["average"].shape) - 1


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        LogisticRegression(),
        GradientBoostingRegressor(),
        GradientBoostingClassifier(),
    ],
)
def test_partial_dependence_unfitted(estimator):
    X = iris.data
    preprocessor = make_column_transformer(
        (StandardScaler(), [0, 2]), (RobustScaler(), [1, 3])
    )
    pipe = make_pipeline(preprocessor, estimator)
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        partial_dependence(pipe, X, features=[0, 2], grid_resolution=10)
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        partial_dependence(estimator, X, features=[0, 2], grid_resolution=10)


@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),
        (LogisticRegression, binary_classification_data),
    ],
)
def test_kind_average_and_average_of_individual(Estimator, data):
    est = Estimator()
    (X, y), n_targets = data
    est.fit(X, y)

    pdp_avg = partial_dependence(est, X=X, features=[1, 2], kind="average")
    pdp_ind = partial_dependence(est, X=X, features=[1, 2], kind="individual")
    avg_ind = np.mean(pdp_ind["individual"], axis=1)
    assert_allclose(avg_ind, pdp_avg["average"])


@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),
        (LogisticRegression, binary_classification_data),
    ],
)
def test_partial_dependence_kind_individual_ignores_sample_weight(Estimator, data):
    """Check that `sample_weight` does not have any effect on reported ICE."""
    est = Estimator()
    (X, y), n_targets = data
    sample_weight = np.arange(X.shape[0])
    est.fit(X, y)

    pdp_nsw = partial_dependence(est, X=X, features=[1, 2], kind="individual")
    pdp_sw = partial_dependence(
        est, X=X, features=[1, 2], kind="individual", sample_weight=sample_weight
    )
    assert_allclose(pdp_nsw["individual"], pdp_sw["individual"])
    assert_allclose(pdp_nsw["grid_values"], pdp_sw["grid_values"])


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        LogisticRegression(),
        RandomForestRegressor(),
        GradientBoostingClassifier(),
    ],
)
@pytest.mark.parametrize("non_null_weight_idx", [0, 1, -1])
def test_partial_dependence_non_null_weight_idx(estimator, non_null_weight_idx):
    """Check that if we pass a `sample_weight` of zeros with only one index with
    sample weight equals one, then the average `partial_dependence` with this
    `sample_weight` is equal to the individual `partial_dependence` of the
    corresponding index.
    """
    X, y = iris.data, iris.target
    preprocessor = make_column_transformer(
        (StandardScaler(), [0, 2]), (RobustScaler(), [1, 3])
    )
    pipe = make_pipeline(preprocessor, estimator).fit(X, y)

    sample_weight = np.zeros_like(y)
    sample_weight[non_null_weight_idx] = 1
    pdp_sw = partial_dependence(
        pipe,
        X,
        [2, 3],
        kind="average",
        sample_weight=sample_weight,
        grid_resolution=10,
    )
    pdp_ind = partial_dependence(pipe, X, [2, 3], kind="individual", grid_resolution=10)
    output_dim = 1 if is_regressor(pipe) else len(np.unique(y))
    for i in range(output_dim):
        assert_allclose(
            pdp_ind["individual"][i][non_null_weight_idx],
            pdp_sw["average"][i],
        )


@pytest.mark.parametrize(
    "Estimator, data",
    [
        (LinearRegression, multioutput_regression_data),
        (LogisticRegression, binary_classification_data),
    ],
)
def test_partial_dependence_equivalence_equal_sample_weight(Estimator, data):
    """Check that `sample_weight=None` is equivalent to having equal weights."""

    est = Estimator()
    (X, y), n_targets = data
    est.fit(X, y)

    sample_weight, params = None, {"X": X, "features": [1, 2], "kind": "average"}
    pdp_sw_none = partial_dependence(est, **params, sample_weight=sample_weight)
    sample_weight = np.ones(len(y))
    pdp_sw_unit = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none["average"], pdp_sw_unit["average"])
    sample_weight = 2 * np.ones(len(y))
    pdp_sw_doubling = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none["average"], pdp_sw_doubling["average"])


def test_partial_dependence_sample_weight_size_error():
    """Check that we raise an error when the size of `sample_weight` is not
    consistent with `X` and `y`.
    """
    est = LogisticRegression()
    (X, y), n_targets = binary_classification_data
    sample_weight = np.ones_like(y)
    est.fit(X, y)

    with pytest.raises(ValueError, match="sample_weight.shape =="):
        partial_dependence(
            est, X, features=[0], sample_weight=sample_weight[1:], grid_resolution=10
        )


def test_partial_dependence_sample_weight_with_recursion():
    """Check that we raise an error when `sample_weight` is provided with
    `"recursion"` method.
    """
    est = RandomForestRegressor()
    (X, y), n_targets = regression_data
    sample_weight = np.ones_like(y)
    est.fit(X, y, sample_weight=sample_weight)

    with pytest.raises(ValueError, match="'recursion' method can only be applied when"):
        partial_dependence(
            est, X, features=[0], method="recursion", sample_weight=sample_weight
        )


# TODO(1.5): Remove when bunch values is deprecated in 1.5
def test_partial_dependence_bunch_values_deprecated():
    """Test that deprecation warning is raised when values is accessed."""

    est = LogisticRegression()
    (X, y), _ = binary_classification_data
    est.fit(X, y)

    pdp_avg = partial_dependence(est, X=X, features=[1, 2], kind="average")

    msg = (
        "Key: 'values', is deprecated in 1.3 and will be "
        "removed in 1.5. Please use 'grid_values' instead"
    )

    with warnings.catch_warnings():
        # Does not raise warnings with "grid_values"
        warnings.simplefilter("error", FutureWarning)
        grid_values = pdp_avg["grid_values"]

    with pytest.warns(FutureWarning, match=msg):
        # Warns for "values"
        values = pdp_avg["values"]

    # "values" and "grid_values" are the same object
    assert values is grid_values


def test_mixed_type_categorical():
    """Check that we raise a proper error when a column has mixed types and
    the sorting of `np.unique` will fail."""
    X = np.array(["A", "B", "C", np.nan], dtype=object).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])

    from sklearn.preprocessing import OrdinalEncoder

    clf = make_pipeline(
        OrdinalEncoder(encoded_missing_value=-1),
        LogisticRegression(),
    ).fit(X, y)
    with pytest.raises(ValueError, match="The column #0 contains mixed data types"):
        partial_dependence(clf, X, features=[0])
