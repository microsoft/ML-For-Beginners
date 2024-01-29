import html
import locale
import re
from contextlib import closing
from io import StringIO
from unittest.mock import patch

import pytest

from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._estimator_html_repr import (
    _get_css_style,
    _get_visual_block,
    _HTMLDocumentationLinkMixin,
    _write_label_html,
    estimator_html_repr,
)
from sklearn.utils.fixes import parse_version


@pytest.mark.parametrize("checked", [True, False])
def test_write_label_html(checked):
    # Test checking logic and labeling
    name = "LogisticRegression"
    tool_tip = "hello-world"

    with closing(StringIO()) as out:
        _write_label_html(out, name, tool_tip, checked=checked)
        html_label = out.getvalue()

        p = (
            r'<label for="sk-estimator-id-[0-9]*"'
            r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
            r"LogisticRegression"
        )
        re_compiled = re.compile(p)
        assert re_compiled.search(html_label)

        assert html_label.startswith('<div class="sk-label-container">')
        assert "<pre>hello-world</pre>" in html_label
        if checked:
            assert "checked>" in html_label


@pytest.mark.parametrize("est", ["passthrough", "drop", None])
def test_get_visual_block_single_str_none(est):
    # Test estimators that are represented by strings
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == str(est)
    assert est_html_info.name_details == str(est)


def test_get_visual_block_single_estimator():
    est = LogisticRegression(C=10.0)
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == est.__class__.__name__
    assert est_html_info.name_details == str(est)


def test_get_visual_block_pipeline():
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("do_nothing", "passthrough"),
            ("do_nothing_more", None),
            ("classifier", LogisticRegression()),
        ]
    )
    est_html_info = _get_visual_block(pipe)
    assert est_html_info.kind == "serial"
    assert est_html_info.estimators == tuple(step[1] for step in pipe.steps)
    assert est_html_info.names == [
        "imputer: SimpleImputer",
        "do_nothing: passthrough",
        "do_nothing_more: passthrough",
        "classifier: LogisticRegression",
    ]
    assert est_html_info.name_details == [str(est) for _, est in pipe.steps]


def test_get_visual_block_feature_union():
    f_union = FeatureUnion([("pca", PCA()), ("svd", TruncatedSVD())])
    est_html_info = _get_visual_block(f_union)
    assert est_html_info.kind == "parallel"
    assert est_html_info.names == ("pca", "svd")
    assert est_html_info.estimators == tuple(
        trans[1] for trans in f_union.transformer_list
    )
    assert est_html_info.name_details == (None, None)


def test_get_visual_block_voting():
    clf = VotingClassifier(
        [("log_reg", LogisticRegression()), ("mlp", MLPClassifier())]
    )
    est_html_info = _get_visual_block(clf)
    assert est_html_info.kind == "parallel"
    assert est_html_info.estimators == tuple(trans[1] for trans in clf.estimators)
    assert est_html_info.names == ("log_reg", "mlp")
    assert est_html_info.name_details == (None, None)


def test_get_visual_block_column_transformer():
    ct = ColumnTransformer(
        [("pca", PCA(), ["num1", "num2"]), ("svd", TruncatedSVD, [0, 3])]
    )
    est_html_info = _get_visual_block(ct)
    assert est_html_info.kind == "parallel"
    assert est_html_info.estimators == tuple(trans[1] for trans in ct.transformers)
    assert est_html_info.names == ("pca", "svd")
    assert est_html_info.name_details == (["num1", "num2"], [0, 3])


def test_estimator_html_repr_pipeline():
    num_trans = Pipeline(
        steps=[("pass", "passthrough"), ("imputer", SimpleImputer(strategy="median"))]
    )

    cat_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", missing_values="empty")),
            ("one-hot", OneHotEncoder(drop="first")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", num_trans, ["a", "b", "c", "d", "e"]),
            ("cat", cat_trans, [0, 1, 2, 3]),
        ]
    )

    feat_u = FeatureUnion(
        [
            ("pca", PCA(n_components=1)),
            (
                "tsvd",
                Pipeline(
                    [
                        ("first", TruncatedSVD(n_components=3)),
                        ("select", SelectPercentile()),
                    ]
                ),
            ),
        ]
    )

    clf = VotingClassifier(
        [
            ("lr", LogisticRegression(solver="lbfgs", random_state=1)),
            ("mlp", MLPClassifier(alpha=0.001)),
        ]
    )

    pipe = Pipeline(
        [("preprocessor", preprocess), ("feat_u", feat_u), ("classifier", clf)]
    )
    html_output = estimator_html_repr(pipe)

    # top level estimators show estimator with changes
    assert html.escape(str(pipe)) in html_output
    for _, est in pipe.steps:
        assert (
            '<div class="sk-toggleable__content "><pre>' + html.escape(str(est))
        ) in html_output

    # low level estimators do not show changes
    with config_context(print_changed_only=True):
        assert html.escape(str(num_trans["pass"])) in html_output
        assert "passthrough</label>" in html_output
        assert html.escape(str(num_trans["imputer"])) in html_output

        for _, _, cols in preprocess.transformers:
            assert f"<pre>{html.escape(str(cols))}</pre>" in html_output

        # feature union
        for name, _ in feat_u.transformer_list:
            assert f"<label>{html.escape(name)}</label>" in html_output

        pca = feat_u.transformer_list[0][1]
        assert f"<pre>{html.escape(str(pca))}</pre>" in html_output

        tsvd = feat_u.transformer_list[1][1]
        first = tsvd["first"]
        select = tsvd["select"]
        assert f"<pre>{html.escape(str(first))}</pre>" in html_output
        assert f"<pre>{html.escape(str(select))}</pre>" in html_output

        # voting classifier
        for name, est in clf.estimators:
            assert f"<label>{html.escape(name)}</label>" in html_output
            assert f"<pre>{html.escape(str(est))}</pre>" in html_output

    # verify that prefers-color-scheme is implemented
    assert "prefers-color-scheme" in html_output


@pytest.mark.parametrize("final_estimator", [None, LinearSVC()])
def test_stacking_classifier(final_estimator):
    estimators = [
        ("mlp", MLPClassifier(alpha=0.001)),
        ("tree", DecisionTreeClassifier()),
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    html_output = estimator_html_repr(clf)

    assert html.escape(str(clf)) in html_output
    # If final_estimator's default changes from LogisticRegression
    # this should be updated
    if final_estimator is None:
        assert "LogisticRegression(" in html_output
    else:
        assert final_estimator.__class__.__name__ in html_output


@pytest.mark.parametrize("final_estimator", [None, LinearSVR()])
def test_stacking_regressor(final_estimator):
    reg = StackingRegressor(
        estimators=[("svr", LinearSVR())], final_estimator=final_estimator
    )
    html_output = estimator_html_repr(reg)

    assert html.escape(str(reg.estimators[0][0])) in html_output
    p = (
        r'<label for="sk-estimator-id-[0-9]*"'
        r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
        r"&nbsp;LinearSVR"
    )
    re_compiled = re.compile(p)
    assert re_compiled.search(html_output)

    if final_estimator is None:
        p = (
            r'<label for="sk-estimator-id-[0-9]*"'
            r' class="sk-toggleable__label (fitted)? sk-toggleable__label-arrow ">'
            r"&nbsp;RidgeCV"
        )
        re_compiled = re.compile(p)
        assert re_compiled.search(html_output)
    else:
        assert html.escape(final_estimator.__class__.__name__) in html_output


def test_birch_duck_typing_meta():
    # Test duck typing meta estimators with Birch
    birch = Birch(n_clusters=AgglomerativeClustering(n_clusters=3))
    html_output = estimator_html_repr(birch)

    # inner estimators do not show changes
    with config_context(print_changed_only=True):
        assert f"<pre>{html.escape(str(birch.n_clusters))}" in html_output
        assert "AgglomerativeClustering</label>" in html_output

    # outer estimator contains all changes
    assert f"<pre>{html.escape(str(birch))}" in html_output


def test_ovo_classifier_duck_typing_meta():
    # Test duck typing metaestimators with OVO
    ovo = OneVsOneClassifier(LinearSVC(penalty="l1"))
    html_output = estimator_html_repr(ovo)

    # inner estimators do not show changes
    with config_context(print_changed_only=True):
        assert f"<pre>{html.escape(str(ovo.estimator))}" in html_output
        # regex to match the start of the tag
        p = (
            r'<label for="sk-estimator-id-[0-9]*" '
            r'class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;LinearSVC'
        )
        re_compiled = re.compile(p)
        assert re_compiled.search(html_output)

    # outer estimator
    assert f"<pre>{html.escape(str(ovo))}" in html_output


def test_duck_typing_nested_estimator():
    # Test duck typing metaestimators with random search
    kernel_ridge = KernelRidge(kernel=ExpSineSquared())
    param_distributions = {"alpha": [1, 2]}

    kernel_ridge_tuned = RandomizedSearchCV(
        kernel_ridge,
        param_distributions=param_distributions,
    )
    html_output = estimator_html_repr(kernel_ridge_tuned)
    assert "estimator: KernelRidge</label>" in html_output


@pytest.mark.parametrize("print_changed_only", [True, False])
def test_one_estimator_print_change_only(print_changed_only):
    pca = PCA(n_components=10)

    with config_context(print_changed_only=print_changed_only):
        pca_repr = html.escape(str(pca))
        html_output = estimator_html_repr(pca)
        assert pca_repr in html_output


def test_fallback_exists():
    """Check that repr fallback is in the HTML."""
    pca = PCA(n_components=10)
    html_output = estimator_html_repr(pca)

    assert (
        f'<div class="sk-text-repr-fallback"><pre>{html.escape(str(pca))}'
        in html_output
    )


def test_show_arrow_pipeline():
    """Show arrow in pipeline for top level in pipeline"""
    pipe = Pipeline([("scale", StandardScaler()), ("log_Reg", LogisticRegression())])

    html_output = estimator_html_repr(pipe)
    assert (
        'class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;&nbsp;Pipeline'
        in html_output
    )


def test_invalid_parameters_in_stacking():
    """Invalidate stacking configuration uses default repr.

    Non-regression test for #24009.
    """
    stacker = StackingClassifier(estimators=[])

    html_output = estimator_html_repr(stacker)
    assert html.escape(str(stacker)) in html_output


def test_estimator_get_params_return_cls():
    """Check HTML repr works where a value in get_params is a class."""

    class MyEstimator:
        def get_params(self, deep=False):
            return {"inner_cls": LogisticRegression}

    est = MyEstimator()
    assert "MyEstimator" in estimator_html_repr(est)


def test_estimator_html_repr_unfitted_vs_fitted():
    """Check that we have the information that the estimator is fitted or not in the
    HTML representation.
    """

    class MyEstimator(BaseEstimator):
        def fit(self, X, y):
            self.fitted_ = True
            return self

    X, y = load_iris(return_X_y=True)
    estimator = MyEstimator()
    assert "<span>Not fitted</span>" in estimator_html_repr(estimator)
    estimator.fit(X, y)
    assert "<span>Fitted</span>" in estimator_html_repr(estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), slice(0, 3))),
            LogisticRegression(),
        ),
    ],
)
def test_estimator_html_repr_fitted_icon(estimator):
    """Check that we are showing the fitted status icon only once."""
    pattern = '<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span>'
    assert estimator_html_repr(estimator).count(pattern) == 1
    X, y = load_iris(return_X_y=True)
    estimator.fit(X, y)
    pattern = '<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span>'
    assert estimator_html_repr(estimator).count(pattern) == 1


@pytest.mark.parametrize("mock_version", ["1.3.0.dev0", "1.3.0"])
def test_html_documentation_link_mixin_sklearn(mock_version):
    """Check the behaviour of the `_HTMLDocumentationLinkMixin` class for scikit-learn
    default.
    """

    # mock the `__version__` where the mixin is located
    with patch("sklearn.utils._estimator_html_repr.__version__", mock_version):
        mixin = _HTMLDocumentationLinkMixin()

        assert mixin._doc_link_module == "sklearn"
        sklearn_version = parse_version(mock_version)
        # we need to parse the version manually to be sure that this test is passing in
        # other branches than `main` (that is "dev").
        if sklearn_version.dev is None:
            version = f"{sklearn_version.major}.{sklearn_version.minor}"
        else:
            version = "dev"
        assert (
            mixin._doc_link_template
            == f"https://scikit-learn.org/{version}/modules/generated/"
            "{estimator_module}.{estimator_name}.html"
        )
        assert (
            mixin._get_doc_link()
            == f"https://scikit-learn.org/{version}/modules/generated/"
            "sklearn.utils._HTMLDocumentationLinkMixin.html"
        )


@pytest.mark.parametrize(
    "module_path,expected_module",
    [
        ("prefix.mymodule", "prefix.mymodule"),
        ("prefix._mymodule", "prefix"),
        ("prefix.mypackage._mymodule", "prefix.mypackage"),
        ("prefix.mypackage._mymodule.submodule", "prefix.mypackage"),
        ("prefix.mypackage.mymodule.submodule", "prefix.mypackage.mymodule.submodule"),
    ],
)
def test_html_documentation_link_mixin_get_doc_link(module_path, expected_module):
    """Check the behaviour of the `_get_doc_link` with various parameter."""

    class FooBar(_HTMLDocumentationLinkMixin):
        pass

    FooBar.__module__ = module_path
    est = FooBar()
    # if we set `_doc_link`, then we expect to infer a module and name for the estimator
    est._doc_link_module = "prefix"
    est._doc_link_template = (
        "https://website.com/{estimator_module}.{estimator_name}.html"
    )
    assert est._get_doc_link() == f"https://website.com/{expected_module}.FooBar.html"


def test_html_documentation_link_mixin_get_doc_link_out_of_library():
    """Check the behaviour of the `_get_doc_link` with various parameter."""
    mixin = _HTMLDocumentationLinkMixin()

    # if the `_doc_link_module` does not refer to the root module of the estimator
    # (here the mixin), then we should return an empty string.
    mixin._doc_link_module = "xxx"
    assert mixin._get_doc_link() == ""


def test_html_documentation_link_mixin_doc_link_url_param_generator():
    mixin = _HTMLDocumentationLinkMixin()
    # we can bypass the generation by providing our own callable
    mixin._doc_link_template = (
        "https://website.com/{my_own_variable}.{another_variable}.html"
    )

    def url_param_generator(estimator):
        return {
            "my_own_variable": "value_1",
            "another_variable": "value_2",
        }

    mixin._doc_link_url_param_generator = url_param_generator

    assert mixin._get_doc_link() == "https://website.com/value_1.value_2.html"


@pytest.fixture
def set_non_utf8_locale():
    """Pytest fixture to set non utf-8 locale during the test.

    The locale is set to the original one after the test has run.
    """
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
    except locale.Error:
        pytest.skip("'C' locale is not available on this OS")

    yield

    # Resets the locale to the original one. Python calls setlocale(LC_TYPE, "")
    # at startup according to
    # https://docs.python.org/3/library/locale.html#background-details-hints-tips-and-caveats.
    # This assumes that no other locale changes have been made. For some reason,
    # on some platforms, trying to restore locale with something like
    # locale.setlocale(locale.LC_CTYPE, locale.getlocale()) raises a
    # locale.Error: unsupported locale setting
    locale.setlocale(locale.LC_CTYPE, "")


def test_non_utf8_locale(set_non_utf8_locale):
    """Checks that utf8 encoding is used when reading the CSS file.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/27725
    """
    _get_css_style()
