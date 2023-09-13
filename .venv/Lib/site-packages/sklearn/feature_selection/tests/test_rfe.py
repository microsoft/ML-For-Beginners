"""
Testing Recursive feature elimination
"""

from operator import attrgetter

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings


class MockClassifier:
    """
    Dummy classifier to test recursive feature elimination
    """

    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, y):
        assert len(X) == len(y)
        self.coef_ = np.ones(X.shape[1], dtype=np.float64)
        return self

    def predict(self, T):
        return T.shape[0]

    predict_proba = predict
    decision_function = predict
    transform = predict

    def score(self, X=None, y=None):
        return 0.0

    def get_params(self, deep=True):
        return {"foo_param": self.foo_param}

    def set_params(self, **params):
        return self

    def _more_tags(self):
        return {"allow_nan": True}


def test_rfe_features_importance():
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    clf = RandomForestClassifier(n_estimators=20, random_state=generator, max_depth=2)
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    assert len(rfe.ranking_) == X.shape[1]

    clf_svc = SVC(kernel="linear")
    rfe_svc = RFE(estimator=clf_svc, n_features_to_select=4, step=0.1)
    rfe_svc.fit(X, y)

    # Check if the supports are equal
    assert_array_equal(rfe.get_support(), rfe_svc.get_support())


def test_rfe():
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    X_sparse = sparse.csr_matrix(X)
    y = iris.target

    # dense model
    clf = SVC(kernel="linear")
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    X_r = rfe.transform(X)
    clf.fit(X_r, y)
    assert len(rfe.ranking_) == X.shape[1]

    # sparse model
    clf_sparse = SVC(kernel="linear")
    rfe_sparse = RFE(estimator=clf_sparse, n_features_to_select=4, step=0.1)
    rfe_sparse.fit(X_sparse, y)
    X_r_sparse = rfe_sparse.transform(X_sparse)

    assert X_r.shape == iris.data.shape
    assert_array_almost_equal(X_r[:10], iris.data[:10])

    assert_array_almost_equal(rfe.predict(X), clf.predict(iris.data))
    assert rfe.score(X, y) == clf.score(iris.data, iris.target)
    assert_array_almost_equal(X_r, X_r_sparse.toarray())


def test_RFE_fit_score_params():
    # Make sure RFE passes the metadata down to fit and score methods of the
    # underlying estimator
    class TestEstimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y, prop=None):
            if prop is None:
                raise ValueError("fit: prop cannot be None")
            self.svc_ = SVC(kernel="linear").fit(X, y)
            self.coef_ = self.svc_.coef_
            return self

        def score(self, X, y, prop=None):
            if prop is None:
                raise ValueError("score: prop cannot be None")
            return self.svc_.score(X, y)

    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="fit: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y)
    with pytest.raises(ValueError, match="score: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y)

    RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y, prop="foo")


def test_rfe_percent_n_features():
    # test that the results are the same
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target
    # there are 10 features in the data. We select 40%.
    clf = SVC(kernel="linear")
    rfe_num = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe_num.fit(X, y)

    rfe_perc = RFE(estimator=clf, n_features_to_select=0.4, step=0.1)
    rfe_perc.fit(X, y)

    assert_array_equal(rfe_perc.ranking_, rfe_num.ranking_)
    assert_array_equal(rfe_perc.support_, rfe_num.support_)


def test_rfe_mockclassifier():
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # dense model
    clf = MockClassifier()
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    X_r = rfe.transform(X)
    clf.fit(X_r, y)
    assert len(rfe.ranking_) == X.shape[1]
    assert X_r.shape == iris.data.shape


def test_rfecv():
    generator = check_random_state(0)
    iris = load_iris()
    # Add some irrelevant features. Random seed is set to make sure that
    # irrelevant features are always irrelevant.
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Test using the score function
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1)
    rfecv.fit(X, y)
    # non-regression test for missing worst feature:

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]

    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)

    # All the noisy variable were filtered out
    assert_array_equal(X_r, iris.data)

    # same in sparse
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=1)
    X_sparse = sparse.csr_matrix(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # Test using a customized loss function
    scoring = make_scorer(zero_one_loss, greater_is_better=False)
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scoring)
    ignore_warnings(rfecv.fit)(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    # Test using a scorer
    scorer = get_scorer("accuracy")
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scorer)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    # Test fix on cv_results_
    def test_scorer(estimator, X, y):
        return 1.0

    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=test_scorer)
    rfecv.fit(X, y)

    # In the event of cross validation score ties, the expected behavior of
    # RFECV is to return the FEWEST features that maximize the CV score.
    # Because test_scorer always returns 1.0 in this example, RFECV should
    # reduce the dimensionality to a single feature (i.e. n_features_ = 1)
    assert rfecv.n_features_ == 1

    # Same as the first two tests, but with step=2
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=2)
    rfecv.fit(X, y)

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == 6

    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=2)
    X_sparse = sparse.csr_matrix(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # Verifying that steps < 1 don't blow up.
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=0.2)
    X_sparse = sparse.csr_matrix(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)


def test_rfecv_mockclassifier():
    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Test using the score function
    rfecv = RFECV(estimator=MockClassifier(), step=1)
    rfecv.fit(X, y)
    # non-regression test for missing worst feature:

    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]

    assert len(rfecv.ranking_) == X.shape[1]


def test_rfecv_verbose_output():
    # Check verbose=1 is producing an output.
    import sys
    from io import StringIO

    sys.stdout = StringIO()

    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)

    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, verbose=1)
    rfecv.fit(X, y)

    verbose_output = sys.stdout
    verbose_output.seek(0)
    assert len(verbose_output.readline()) > 0


def test_rfecv_cv_results_size(global_random_seed):
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # Non-regression test for varying combinations of step and
    # min_features_to_select.
    for step, min_features_to_select in [[2, 1], [2, 2], [3, 3]]:
        rfecv = RFECV(
            estimator=MockClassifier(),
            step=step,
            min_features_to_select=min_features_to_select,
        )
        rfecv.fit(X, y)

        score_len = np.ceil((X.shape[1] - min_features_to_select) / step) + 1

        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == score_len

        assert len(rfecv.ranking_) == X.shape[1]
        assert rfecv.n_features_ >= min_features_to_select


def test_rfe_estimator_tags():
    rfe = RFE(SVC(kernel="linear"))
    assert rfe._estimator_type == "classifier"
    # make sure that cross-validation is stratified
    iris = load_iris()
    score = cross_val_score(rfe, iris.data, iris.target)
    assert score.min() > 0.7


def test_rfe_min_step(global_random_seed):
    n_features = 10
    X, y = make_friedman1(
        n_samples=50, n_features=n_features, random_state=global_random_seed
    )
    n_samples, n_features = X.shape
    estimator = SVR(kernel="linear")

    # Test when floor(step * n_features) <= 0
    selector = RFE(estimator, step=0.01)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2

    # Test when step is between (0,1) and floor(step * n_features) > 0
    selector = RFE(estimator, step=0.20)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2

    # Test when step is an integer
    selector = RFE(estimator, step=5)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2


def test_number_of_subsets_of_features(global_random_seed):
    # In RFE, 'number_of_subsets_of_features'
    # = the number of iterations in '_fit'
    # = max(ranking_)
    # = 1 + (n_features + step - n_features_to_select - 1) // step
    # After optimization #4534, this number
    # = 1 + np.ceil((n_features - n_features_to_select) / float(step))
    # This test case is to test their equivalence, refer to #4534 and #3824

    def formula1(n_features, n_features_to_select, step):
        return 1 + ((n_features + step - n_features_to_select - 1) // step)

    def formula2(n_features, n_features_to_select, step):
        return 1 + np.ceil((n_features - n_features_to_select) / float(step))

    # RFE
    # Case 1, n_features - n_features_to_select is divisible by step
    # Case 2, n_features - n_features_to_select is not divisible by step
    n_features_list = [11, 11]
    n_features_to_select_list = [3, 3]
    step_list = [2, 3]
    for n_features, n_features_to_select, step in zip(
        n_features_list, n_features_to_select_list, step_list
    ):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfe = RFE(
            estimator=SVC(kernel="linear"),
            n_features_to_select=n_features_to_select,
            step=step,
        )
        rfe.fit(X, y)
        # this number also equals to the maximum of ranking_
        assert np.max(rfe.ranking_) == formula1(n_features, n_features_to_select, step)
        assert np.max(rfe.ranking_) == formula2(n_features, n_features_to_select, step)

    # In RFECV, 'fit' calls 'RFE._fit'
    # 'number_of_subsets_of_features' of RFE
    # = the size of each score in 'cv_results_' of RFECV
    # = the number of iterations of the for loop before optimization #4534

    # RFECV, n_features_to_select = 1
    # Case 1, n_features - 1 is divisible by step
    # Case 2, n_features - 1 is not divisible by step

    n_features_to_select = 1
    n_features_list = [11, 10]
    step_list = [2, 2]
    for n_features, step in zip(n_features_list, step_list):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfecv = RFECV(estimator=SVC(kernel="linear"), step=step)
        rfecv.fit(X, y)

        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == formula1(
                n_features, n_features_to_select, step
            )
            assert len(rfecv.cv_results_[key]) == formula2(
                n_features, n_features_to_select, step
            )


def test_rfe_cv_n_jobs(global_random_seed):
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    rfecv = RFECV(estimator=SVC(kernel="linear"))
    rfecv.fit(X, y)
    rfecv_ranking = rfecv.ranking_

    rfecv_cv_results_ = rfecv.cv_results_

    rfecv.set_params(n_jobs=2)
    rfecv.fit(X, y)
    assert_array_almost_equal(rfecv.ranking_, rfecv_ranking)

    assert rfecv_cv_results_.keys() == rfecv.cv_results_.keys()
    for key in rfecv_cv_results_.keys():
        assert rfecv_cv_results_[key] == pytest.approx(rfecv.cv_results_[key])


def test_rfe_cv_groups():
    generator = check_random_state(0)
    iris = load_iris()
    number_groups = 4
    groups = np.floor(np.linspace(0, number_groups, len(iris.target)))
    X = iris.data
    y = (iris.target > 0).astype(int)

    est_groups = RFECV(
        estimator=RandomForestClassifier(random_state=generator),
        step=1,
        scoring="accuracy",
        cv=GroupKFold(n_splits=2),
    )
    est_groups.fit(X, y, groups=groups)
    assert est_groups.n_features_ > 0


@pytest.mark.parametrize(
    "importance_getter", [attrgetter("regressor_.coef_"), "regressor_.coef_"]
)
@pytest.mark.parametrize("selector, expected_n_features", [(RFE, 5), (RFECV, 4)])
def test_rfe_wrapped_estimator(importance_getter, selector, expected_n_features):
    # Non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/15312
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = LinearSVR(dual="auto", random_state=0)

    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    selector = selector(log_estimator, importance_getter=importance_getter)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == expected_n_features


@pytest.mark.parametrize(
    "importance_getter, err_type",
    [
        ("auto", ValueError),
        ("random", AttributeError),
        (lambda x: x.importance, AttributeError),
    ],
)
@pytest.mark.parametrize("Selector", [RFE, RFECV])
def test_rfe_importance_getter_validation(importance_getter, err_type, Selector):
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=42)
    estimator = LinearSVR(dual="auto")
    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    with pytest.raises(err_type):
        model = Selector(log_estimator, importance_getter=importance_getter)
        model.fit(X, y)


@pytest.mark.parametrize("cv", [None, 5])
def test_rfe_allow_nan_inf_in_x(cv):
    iris = load_iris()
    X = iris.data
    y = iris.target

    # add nan and inf value to X
    X[0][0] = np.nan
    X[0][1] = np.inf

    clf = MockClassifier()
    if cv is not None:
        rfe = RFECV(estimator=clf, cv=cv)
    else:
        rfe = RFE(estimator=clf)
    rfe.fit(X, y)
    rfe.transform(X)


def test_w_pipeline_2d_coef_():
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    data, y = load_iris(return_X_y=True)
    sfm = RFE(
        pipeline,
        n_features_to_select=2,
        importance_getter="named_steps.logisticregression.coef_",
    )

    sfm.fit(data, y)
    assert sfm.transform(data).shape[1] == 2


def test_rfecv_std_and_mean(global_random_seed):
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    rfecv = RFECV(estimator=SVC(kernel="linear"))
    rfecv.fit(X, y)
    n_split_keys = len(rfecv.cv_results_) - 2
    split_keys = [f"split{i}_test_score" for i in range(n_split_keys)]

    cv_scores = np.asarray([rfecv.cv_results_[key] for key in split_keys])
    expected_mean = np.mean(cv_scores, axis=0)
    expected_std = np.std(cv_scores, axis=0)

    assert_allclose(rfecv.cv_results_["mean_test_score"], expected_mean)
    assert_allclose(rfecv.cv_results_["std_test_score"], expected_std)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
def test_multioutput(ClsRFE):
    X = np.random.normal(size=(10, 3))
    y = np.random.randint(2, size=(10, 2))
    clf = RandomForestClassifier(n_estimators=5)
    rfe_test = ClsRFE(clf)
    rfe_test.fit(X, y)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
@pytest.mark.parametrize("PLSEstimator", [CCA, PLSCanonical, PLSRegression])
def test_rfe_pls(ClsRFE, PLSEstimator):
    """Check the behaviour of RFE with PLS estimators.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12410
    """
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = PLSEstimator(n_components=1)
    selector = ClsRFE(estimator, step=1).fit(X, y)
    assert selector.score(X, y) > 0.5
