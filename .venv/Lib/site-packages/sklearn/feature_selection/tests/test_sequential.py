import numpy as np
import pytest
import scipy
from numpy.testing import assert_array_equal

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_bad_n_features_to_select():
    n_features = 5
    X, y = make_regression(n_features=n_features)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features)
    with pytest.raises(ValueError, match="n_features_to_select must be < n_features"):
        sfs.fit(X, y)


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("n_features_to_select", (1, 5, 9, "auto"))
def test_n_features_to_select(direction, n_features_to_select):
    # Make sure n_features_to_select is respected

    n_features = 10
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)

    if n_features_to_select == "auto":
        n_features_to_select = n_features // 2

    assert sfs.get_support(indices=True).shape[0] == n_features_to_select
    assert sfs.n_features_to_select_ == n_features_to_select
    assert sfs.transform(X).shape[1] == n_features_to_select


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_auto(direction):
    """Check the behaviour of `n_features_to_select="auto"` with different
    values for the parameter `tol`.
    """

    n_features = 10
    tol = 1e-3
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=tol,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)

    max_features_to_select = n_features - 1

    assert sfs.get_support(indices=True).shape[0] <= max_features_to_select
    assert sfs.n_features_to_select_ <= max_features_to_select
    assert sfs.transform(X).shape[1] <= max_features_to_select
    assert sfs.get_support(indices=True).shape[0] == sfs.n_features_to_select_


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_n_features_to_select_stopping_criterion(direction):
    """Check the behaviour stopping criterion for feature selection
    depending on the values of `n_features_to_select` and `tol`.

    When `direction` is `'forward'`, select a new features at random
    among those not currently selected in selector.support_,
    build a new version of the data that includes all the features
    in selector.support_ + this newly selected feature.
    And check that the cross-validation score of the model trained on
    this new dataset variant is lower than the model with
    the selected forward selected features or at least does not improve
    by more than the tol margin.

    When `direction` is `'backward'`, instead of adding a new feature
    to selector.support_, try to remove one of those selected features at random
    And check that the cross-validation score is either decreasing or
    not improving by more than the tol margin.
    """

    X, y = make_regression(n_features=50, n_informative=10, random_state=0)

    tol = 1e-3

    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=tol,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    selected_X = sfs.transform(X)

    rng = np.random.RandomState(0)

    added_candidates = list(set(range(X.shape[1])) - set(sfs.get_support(indices=True)))
    added_X = np.hstack(
        [
            selected_X,
            (X[:, rng.choice(added_candidates)])[:, np.newaxis],
        ]
    )

    removed_candidate = rng.choice(list(range(sfs.n_features_to_select_)))
    removed_X = np.delete(selected_X, removed_candidate, axis=1)

    plain_cv_score = cross_val_score(LinearRegression(), X, y, cv=2).mean()
    sfs_cv_score = cross_val_score(LinearRegression(), selected_X, y, cv=2).mean()
    added_cv_score = cross_val_score(LinearRegression(), added_X, y, cv=2).mean()
    removed_cv_score = cross_val_score(LinearRegression(), removed_X, y, cv=2).mean()

    assert sfs_cv_score >= plain_cv_score

    if direction == "forward":
        assert (sfs_cv_score - added_cv_score) <= tol
        assert (sfs_cv_score - removed_cv_score) >= tol
    else:
        assert (added_cv_score - sfs_cv_score) <= tol
        assert (removed_cv_score - sfs_cv_score) <= tol


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected",
    (
        (0.1, 1),
        (1.0, 10),
        (0.5, 5),
    ),
)
def test_n_features_to_select_float(direction, n_features_to_select, expected):
    # Test passing a float as n_features_to_select
    X, y = make_regression(n_features=10)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    assert sfs.n_features_to_select_ == expected


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize(
    "n_features_to_select, expected_selected_features",
    [
        (2, [0, 2]),  # f1 is dropped since it has no predictive power
        (1, [2]),  # f2 is more predictive than f0 so it's kept
    ],
)
def test_sanity(seed, direction, n_features_to_select, expected_selected_features):
    # Basic sanity check: 3 features, only f0 and f2 are correlated with the
    # target, f2 having a stronger correlation than f0. We expect f1 to be
    # dropped, and f2 to always be selected.

    rng = np.random.RandomState(seed)
    n_samples = 100
    X = rng.randn(n_samples, 3)
    y = 3 * X[:, 0] - 10 * X[:, 2]

    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=2,
    )
    sfs.fit(X, y)
    assert_array_equal(sfs.get_support(indices=True), expected_selected_features)


def test_sparse_support():
    # Make sure sparse data is supported

    X, y = make_regression(n_features=10)
    X = scipy.sparse.csr_matrix(X)
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", cv=2
    )
    sfs.fit(X, y)
    sfs.transform(X)


def test_nan_support():
    # Make sure nans are OK if the underlying estimator supports nans

    rng = np.random.RandomState(0)
    n_samples, n_features = 40, 4
    X, y = make_regression(n_samples, n_features, random_state=0)
    nan_mask = rng.randint(0, 2, size=(n_samples, n_features), dtype=bool)
    X[nan_mask] = np.nan
    sfs = SequentialFeatureSelector(
        HistGradientBoostingRegressor(), n_features_to_select="auto", cv=2
    )
    sfs.fit(X, y)
    sfs.transform(X)

    with pytest.raises(ValueError, match="Input X contains NaN"):
        # LinearRegression does not support nans
        SequentialFeatureSelector(
            LinearRegression(), n_features_to_select="auto", cv=2
        ).fit(X, y)


def test_pipeline_support():
    # Make sure that pipelines can be passed into SFS and that SFS can be
    # passed into a pipeline

    n_samples, n_features = 50, 3
    X, y = make_regression(n_samples, n_features, random_state=0)

    # pipeline in SFS
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    sfs = SequentialFeatureSelector(pipe, n_features_to_select="auto", cv=2)
    sfs.fit(X, y)
    sfs.transform(X)

    # SFS in pipeline
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select="auto", cv=2
    )
    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(X, y)
    pipe.transform(X)


@pytest.mark.parametrize("n_features_to_select", (2, 3))
def test_unsupervised_model_fit(n_features_to_select):
    # Make sure that models without classification labels are not being
    # validated

    X, y = make_blobs(n_features=4)
    sfs = SequentialFeatureSelector(
        KMeans(n_init=1),
        n_features_to_select=n_features_to_select,
    )
    sfs.fit(X)
    assert sfs.transform(X).shape[1] == n_features_to_select


@pytest.mark.parametrize("y", ("no_validation", 1j, 99.9, np.nan, 3))
def test_no_y_validation_model_fit(y):
    # Make sure that other non-conventional y labels are not accepted

    X, clusters = make_blobs(n_features=6)
    sfs = SequentialFeatureSelector(
        KMeans(),
        n_features_to_select=3,
    )

    with pytest.raises((TypeError, ValueError)):
        sfs.fit(X, y)


def test_forward_neg_tol_error():
    """Check that we raise an error when tol<0 and direction='forward'"""
    X, y = make_regression(n_features=10, random_state=0)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        direction="forward",
        tol=-1e-3,
    )

    with pytest.raises(ValueError, match="tol must be positive"):
        sfs.fit(X, y)


def test_backward_neg_tol():
    """Check that SequentialFeatureSelector works negative tol

    non-regression test for #25525
    """
    X, y = make_regression(n_features=10, random_state=0)
    lr = LinearRegression()
    initial_score = lr.fit(X, y).score(X, y)

    sfs = SequentialFeatureSelector(
        lr,
        n_features_to_select="auto",
        direction="backward",
        tol=-1e-3,
    )
    Xr = sfs.fit_transform(X, y)
    new_score = lr.fit(Xr, y).score(Xr, y)

    assert 0 < sfs.get_support().sum() < X.shape[1]
    assert new_score < initial_score


def test_cv_generator_support():
    """Check that no exception raised when cv is generator

    non-regression test for #25957
    """
    X, y = make_classification(random_state=0)

    groups = np.zeros_like(y, dtype=int)
    groups[y.size // 2 :] = 1

    cv = LeaveOneGroupOut()
    splits = cv.split(X, y, groups=groups)

    knc = KNeighborsClassifier(n_neighbors=5)

    sfs = SequentialFeatureSelector(knc, n_features_to_select=5, cv=splits)
    sfs.fit(X, y)
