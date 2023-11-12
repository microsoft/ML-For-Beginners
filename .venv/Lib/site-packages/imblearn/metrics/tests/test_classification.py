# coding: utf-8
"""Testing the metric for classification with imbalanced dataset"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from functools import partial

import numpy as np
import pytest
from sklearn import datasets, svm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    assert_no_warnings,
)
from sklearn.utils.validation import check_random_state

from imblearn.metrics import (
    classification_report_imbalanced,
    geometric_mean_score,
    macro_averaged_mean_absolute_error,
    make_index_balanced_accuracy,
    sensitivity_score,
    sensitivity_specificity_support,
    specificity_score,
)

RND_SEED = 42
R_TOL = 1e-2

###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC
    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]

    return y_true, y_pred, probas_pred


###############################################################################
# Tests


def test_sensitivity_specificity_score_binary():
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    sen, spe, sup = sensitivity_specificity_support(y_true, y_pred, average=None)
    assert_allclose(sen, [0.88, 0.68], rtol=R_TOL)
    assert_allclose(spe, [0.68, 0.88], rtol=R_TOL)
    assert_array_equal(sup, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs in ({}, {"average": "binary"}):
        sen = assert_no_warnings(sensitivity_score, y_true, y_pred, **kwargs)
        assert sen == pytest.approx(0.68, rel=R_TOL)

        spe = assert_no_warnings(specificity_score, y_true, y_pred, **kwargs)
        assert spe == pytest.approx(0.88, rel=R_TOL)


@pytest.mark.filterwarnings("ignore:Specificity is ill-defined")
@pytest.mark.parametrize(
    "y_pred, expected_sensitivity, expected_specificity",
    [(([1, 1], [1, 1]), 1.0, 0.0), (([-1, -1], [-1, -1]), 0.0, 0.0)],
)
def test_sensitivity_specificity_f_binary_single_class(
    y_pred, expected_sensitivity, expected_specificity
):
    # Such a case may occur with non-stratified cross-validation
    assert sensitivity_score(*y_pred) == expected_sensitivity
    assert specificity_score(*y_pred) == expected_specificity


@pytest.mark.parametrize(
    "average, expected_specificty",
    [
        (None, [1.0, 0.67, 1.0, 1.0, 1.0]),
        ("macro", np.mean([1.0, 0.67, 1.0, 1.0, 1.0])),
        ("micro", 15 / 16),
    ],
)
def test_sensitivity_specificity_extra_labels(average, expected_specificty):
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]

    actual = specificity_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average)
    assert_allclose(expected_specificty, actual, rtol=R_TOL)


def test_sensitivity_specificity_ignored_labels():
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]

    specificity_13 = partial(specificity_score, y_true, y_pred, labels=[1, 3])
    specificity_all = partial(specificity_score, y_true, y_pred, labels=None)

    assert_allclose([1.0, 0.33], specificity_13(average=None), rtol=R_TOL)
    assert_allclose(np.mean([1.0, 0.33]), specificity_13(average="macro"), rtol=R_TOL)
    assert_allclose(
        np.average([1.0, 0.33], weights=[2.0, 1.0]),
        specificity_13(average="weighted"),
        rtol=R_TOL,
    )
    assert_allclose(3.0 / (3.0 + 2.0), specificity_13(average="micro"), rtol=R_TOL)

    # ensure the above were meaningful tests:
    for each in ["macro", "weighted", "micro"]:
        assert specificity_13(average=each) != specificity_all(average=each)


def test_sensitivity_specificity_error_multilabels():
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))

    with pytest.raises(ValueError):
        sensitivity_score(y_true_bin, y_pred_bin)


def test_sensitivity_specificity_support_errors():
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad pos_label
    with pytest.raises(ValueError):
        sensitivity_specificity_support(y_true, y_pred, pos_label=2, average="binary")

    # Bad average option
    with pytest.raises(ValueError):
        sensitivity_specificity_support([0, 1, 2], [1, 2, 0], average="mega")


def test_sensitivity_specificity_unused_pos_label():
    # but average != 'binary'; even if data is binary
    msg = r"use labels=\[pos_label\] to specify a single"
    with pytest.warns(UserWarning, match=msg):
        sensitivity_specificity_support(
            [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        )


def test_geometric_mean_support_binary():
    y_true, y_pred, _ = make_prediction(binary=True)

    # compute the geometric mean for the binary problem
    geo_mean = geometric_mean_score(y_true, y_pred)

    assert_allclose(geo_mean, 0.77, rtol=R_TOL)


@pytest.mark.filterwarnings("ignore:Recall is ill-defined")
@pytest.mark.parametrize(
    "y_true, y_pred, correction, expected_gmean",
    [
        ([0, 0, 1, 1], [0, 0, 1, 1], 0.0, 1.0),
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.0, 0.0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.001, 1.0),
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.001, 0.001),
        ([0, 0, 1, 1], [0, 1, 1, 0], 0.001, 0.5),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 2, 1, 0, 0, 1],
            0.001,
            (0.001**2) ** (1 / 3),
        ),
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], 0.001, 1),
        ([0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], 0.001, (0.5 * 0.75) ** 0.5),
    ],
)
def test_geometric_mean_multiclass(y_true, y_pred, correction, expected_gmean):
    gmean = geometric_mean_score(y_true, y_pred, correction=correction)
    assert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.filterwarnings("ignore:Recall is ill-defined")
@pytest.mark.parametrize(
    "y_true, y_pred, average, expected_gmean",
    [
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "macro", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "micro", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], "weighted", 0.471),
        ([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], None, [0.8660254, 0.0, 0.0]),
    ],
)
def test_geometric_mean_average(y_true, y_pred, average, expected_gmean):
    gmean = geometric_mean_score(y_true, y_pred, average=average)
    assert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.parametrize(
    "y_true, y_pred, sample_weight, average, expected_gmean",
    [
        ([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 0, 1], None, "multiclass", 0.707),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 1, 1, 0, 0, 1],
            [1, 2, 1, 1, 2, 1],
            "multiclass",
            0.707,
        ),
        (
            [0, 1, 2, 0, 1, 2],
            [0, 1, 1, 0, 0, 1],
            [1, 2, 1, 1, 2, 1],
            "weighted",
            0.333,
        ),
    ],
)
def test_geometric_mean_sample_weight(
    y_true, y_pred, sample_weight, average, expected_gmean
):
    gmean = geometric_mean_score(
        y_true,
        y_pred,
        labels=[0, 1],
        sample_weight=sample_weight,
        average=average,
    )
    assert gmean == pytest.approx(expected_gmean, rel=R_TOL)


@pytest.mark.parametrize(
    "average, expected_gmean",
    [
        ("multiclass", 0.41),
        (None, [0.85, 0.29, 0.7]),
        ("macro", 0.68),
        ("weighted", 0.65),
    ],
)
def test_geometric_mean_score_prediction(average, expected_gmean):
    y_true, y_pred, _ = make_prediction(binary=False)

    gmean = geometric_mean_score(y_true, y_pred, average=average)
    assert gmean == pytest.approx(expected_gmean, rel=R_TOL)


def test_iba_geo_mean_binary():
    y_true, y_pred, _ = make_prediction(binary=True)

    iba_gmean = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        geometric_mean_score
    )
    iba = iba_gmean(y_true, y_pred)

    assert_allclose(iba, 0.5948, rtol=R_TOL)


def _format_report(report):
    return " ".join(report.split())


def test_classification_report_imbalanced_multiclass():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = (
        "pre rec spe f1 geo iba sup setosa 0.83 0.79 0.92 "
        "0.81 0.85 0.72 24 versicolor 0.33 0.10 0.86 0.15 "
        "0.29 0.08 31 virginica 0.42 0.90 0.55 0.57 0.70 "
        "0.51 20 avg / total 0.51 0.53 0.80 0.47 0.58 0.40 75"
    )

    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
    )
    assert _format_report(report) == expected_report
    # print classification report with label detection
    expected_report = (
        "pre rec spe f1 geo iba sup 0 0.83 0.79 0.92 0.81 "
        "0.85 0.72 24 1 0.33 0.10 0.86 0.15 0.29 0.08 31 "
        "2 0.42 0.90 0.55 0.57 0.70 0.51 20 avg / total "
        "0.51 0.53 0.80 0.47 0.58 0.40 75"
    )

    report = classification_report_imbalanced(y_true, y_pred)
    assert _format_report(report) == expected_report


def test_classification_report_imbalanced_multiclass_with_digits():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = (
        "pre rec spe f1 geo iba sup setosa 0.82609 0.79167 "
        "0.92157 0.80851 0.85415 0.72010 24 versicolor "
        "0.33333 0.09677 0.86364 0.15000 0.28910 0.07717 "
        "31 virginica 0.41860 0.90000 0.54545 0.57143 0.70065 "
        "0.50831 20 avg / total 0.51375 0.53333 0.79733 "
        "0.47310 0.57966 0.39788 75"
    )
    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        digits=5,
    )
    assert _format_report(report) == expected_report
    # print classification report with label detection
    expected_report = (
        "pre rec spe f1 geo iba sup 0 0.83 0.79 0.92 0.81 "
        "0.85 0.72 24 1 0.33 0.10 0.86 0.15 0.29 0.08 31 "
        "2 0.42 0.90 0.55 0.57 0.70 0.51 20 avg / total 0.51 "
        "0.53 0.80 0.47 0.58 0.40 75"
    )
    report = classification_report_imbalanced(y_true, y_pred)
    assert _format_report(report) == expected_report


def test_classification_report_imbalanced_multiclass_with_string_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    y_true = np.array(["blue", "green", "red"])[y_true]
    y_pred = np.array(["blue", "green", "red"])[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue 0.83 0.79 0.92 0.81 "
        "0.85 0.72 24 green 0.33 0.10 0.86 0.15 0.29 0.08 31 "
        "red 0.42 0.90 0.55 0.57 0.70 0.51 20 avg / total "
        "0.51 0.53 0.80 0.47 0.58 0.40 75"
    )
    report = classification_report_imbalanced(y_true, y_pred)
    assert _format_report(report) == expected_report

    expected_report = (
        "pre rec spe f1 geo iba sup a 0.83 0.79 0.92 0.81 0.85 "
        "0.72 24 b 0.33 0.10 0.86 0.15 0.29 0.08 31 c 0.42 "
        "0.90 0.55 0.57 0.70 0.51 20 avg / total 0.51 0.53 "
        "0.80 0.47 0.58 0.40 75"
    )
    report = classification_report_imbalanced(
        y_true, y_pred, target_names=["a", "b", "c"]
    )
    assert _format_report(report) == expected_report


def test_classification_report_imbalanced_multiclass_with_unicode_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue¢ 0.83 0.79 0.92 0.81 "
        "0.85 0.72 24 green¢ 0.33 0.10 0.86 0.15 0.29 0.08 31 "
        "red¢ 0.42 0.90 0.55 0.57 0.70 0.51 20 avg / total "
        "0.51 0.53 0.80 0.47 0.58 0.40 75"
    )
    report = classification_report_imbalanced(y_true, y_pred)
    assert _format_report(report) == expected_report


def test_classification_report_imbalanced_multiclass_with_long_string_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue", "green" * 5, "red"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = (
        "pre rec spe f1 geo iba sup blue 0.83 0.79 0.92 0.81 "
        "0.85 0.72 24 greengreengreengreengreen 0.33 0.10 "
        "0.86 0.15 0.29 0.08 31 red 0.42 0.90 0.55 0.57 0.70 "
        "0.51 20 avg / total 0.51 0.53 0.80 0.47 0.58 0.40 75"
    )

    report = classification_report_imbalanced(y_true, y_pred)
    assert _format_report(report) == expected_report


@pytest.mark.parametrize(
    "score, expected_score",
    [
        (accuracy_score, 0.54756),
        (jaccard_score, 0.33176),
        (precision_score, 0.65025),
        (recall_score, 0.41616),
    ],
)
def test_iba_sklearn_metrics(score, expected_score):
    y_true, y_pred, _ = make_prediction(binary=True)

    score_iba = make_index_balanced_accuracy(alpha=0.5, squared=True)(score)
    score = score_iba(y_true, y_pred)
    assert score == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "score_loss",
    [average_precision_score, brier_score_loss, cohen_kappa_score, roc_auc_score],
)
def test_iba_error_y_score_prob_error(score_loss):
    y_true, y_pred, _ = make_prediction(binary=True)

    aps = make_index_balanced_accuracy(alpha=0.5, squared=True)(score_loss)
    with pytest.raises(AttributeError):
        aps(y_true, y_pred)


def test_classification_report_imbalanced_dict_with_target_names():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        output_dict=True,
    )
    outer_keys = set(report.keys())
    inner_keys = set(report["setosa"].keys())

    expected_outer_keys = {
        "setosa",
        "versicolor",
        "virginica",
        "avg_pre",
        "avg_rec",
        "avg_spe",
        "avg_f1",
        "avg_geo",
        "avg_iba",
        "total_support",
    }
    expected_inner_keys = {"spe", "f1", "sup", "rec", "geo", "iba", "pre"}

    assert outer_keys == expected_outer_keys
    assert inner_keys == expected_inner_keys


def test_classification_report_imbalanced_dict_without_target_names():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
    print(iris.target_names)
    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        output_dict=True,
    )
    print(report.keys())
    outer_keys = set(report.keys())
    inner_keys = set(report["0"].keys())

    expected_outer_keys = {
        "0",
        "1",
        "2",
        "avg_pre",
        "avg_rec",
        "avg_spe",
        "avg_f1",
        "avg_geo",
        "avg_iba",
        "total_support",
    }
    expected_inner_keys = {"spe", "f1", "sup", "rec", "geo", "iba", "pre"}

    assert outer_keys == expected_outer_keys
    assert inner_keys == expected_inner_keys


@pytest.mark.parametrize(
    "y_true, y_pred, expected_ma_mae",
    [
        ([1, 1, 1, 2, 2, 2], [1, 2, 1, 2, 1, 2], 0.333),
        ([1, 1, 1, 1, 1, 2], [1, 2, 1, 2, 1, 2], 0.2),
        ([1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 3, 1, 2, 1, 1, 2, 3, 3], 0.555),
        ([1, 1, 1, 1, 1, 1, 2, 3, 3], [1, 3, 1, 2, 1, 1, 2, 3, 3], 0.166),
    ],
)
def test_macro_averaged_mean_absolute_error(y_true, y_pred, expected_ma_mae):
    ma_mae = macro_averaged_mean_absolute_error(y_true, y_pred)
    assert ma_mae == pytest.approx(expected_ma_mae, rel=R_TOL)


def test_macro_averaged_mean_absolute_error_sample_weight():
    y_true = [1, 1, 1, 2, 2, 2]
    y_pred = [1, 2, 1, 2, 1, 2]

    ma_mae_no_weights = macro_averaged_mean_absolute_error(y_true, y_pred)

    sample_weight = [1, 1, 1, 1, 1, 1]
    ma_mae_unit_weights = macro_averaged_mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    )

    assert ma_mae_unit_weights == pytest.approx(ma_mae_no_weights)
