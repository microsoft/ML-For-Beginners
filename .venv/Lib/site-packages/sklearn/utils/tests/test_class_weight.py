import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS


def test_compute_class_weight():
    # Test (and demo) compute_class_weight.
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)

    cw = compute_class_weight("balanced", classes=classes, y=y)
    # total effect of samples is preserved
    class_counts = np.bincount(y)[2:]
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert cw[0] < cw[1] < cw[2]


@pytest.mark.parametrize(
    "y_type, class_weight, classes, err_msg",
    [
        (
            "numeric",
            "balanced",
            np.arange(4),
            "classes should have valid labels that are in y",
        ),
        # Non-regression for https://github.com/scikit-learn/scikit-learn/issues/8312
        (
            "numeric",
            {"label_not_present": 1.0},
            np.arange(4),
            r"The classes, \[0, 1, 2, 3\], are not in class_weight",
        ),
        (
            "numeric",
            "balanced",
            np.arange(2),
            "classes should include all valid labels",
        ),
        (
            "numeric",
            {0: 1.0, 1: 2.0},
            np.arange(2),
            "classes should include all valid labels",
        ),
        (
            "string",
            {"dogs": 3, "cat": 2},
            np.array(["dog", "cat"]),
            r"The classes, \['dog'\], are not in class_weight",
        ),
    ],
)
def test_compute_class_weight_not_present(y_type, class_weight, classes, err_msg):
    # Raise error when y does not contain all class labels
    y = (
        np.asarray([0, 0, 0, 1, 1, 2])
        if y_type == "numeric"
        else np.asarray(["dog", "cat", "dog"])
    )

    print(y)
    with pytest.raises(ValueError, match=err_msg):
        compute_class_weight(class_weight, classes=classes, y=y)


def test_compute_class_weight_dict():
    classes = np.arange(3)
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}
    y = np.asarray([0, 0, 1, 2])
    cw = compute_class_weight(class_weights, classes=classes, y=y)

    # When the user specifies class weights, compute_class_weights should just
    # return them.
    assert_array_almost_equal(np.asarray([1.0, 2.0, 3.0]), cw)

    # When a class weight is specified that isn't in classes, the weight is ignored
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0, 4: 1.5}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([1.0, 2.0, 3.0], cw)

    class_weights = {-1: 5.0, 0: 4.0, 1: 2.0, 2: 3.0}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([4.0, 2.0, 3.0], cw)


def test_compute_class_weight_invariance():
    # Test that results with class_weight="balanced" is invariant wrt
    # class imbalance if the number of samples is identical.
    # The test uses a balanced two class dataset with 100 datapoints.
    # It creates three versions, one where class 1 is duplicated
    # resulting in 150 points of class 1 and 50 of class 0,
    # one where there are 50 points in class 1 and 150 in class 0,
    # and one where there are 100 points of each class (this one is balanced
    # again).
    # With balancing class weights, all three should give the same model.
    X, y = make_blobs(centers=2, random_state=0)
    # create dataset where class 1 is duplicated twice
    X_1 = np.vstack([X] + [X[y == 1]] * 2)
    y_1 = np.hstack([y] + [y[y == 1]] * 2)
    # create dataset where class 0 is duplicated twice
    X_0 = np.vstack([X] + [X[y == 0]] * 2)
    y_0 = np.hstack([y] + [y[y == 0]] * 2)
    # duplicate everything
    X_ = np.vstack([X] * 2)
    y_ = np.hstack([y] * 2)
    # results should be identical
    logreg1 = LogisticRegression(class_weight="balanced").fit(X_1, y_1)
    logreg0 = LogisticRegression(class_weight="balanced").fit(X_0, y_0)
    logreg = LogisticRegression(class_weight="balanced").fit(X_, y_)
    assert_array_almost_equal(logreg1.coef_, logreg0.coef_)
    assert_array_almost_equal(logreg.coef_, logreg0.coef_)


def test_compute_class_weight_balanced_negative():
    # Test compute_class_weight when labels are negative
    # Test with balanced class labels.
    classes = np.array([-2, -1, 0])
    y = np.asarray([-1, -1, 0, 0, -2, -2])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    assert len(cw) == len(classes)
    assert_array_almost_equal(cw, np.array([1.0, 1.0, 1.0]))

    # Test with unbalanced class labels.
    y = np.asarray([-1, 0, 0, -2, -2, -2])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    assert len(cw) == len(classes)
    class_counts = np.bincount(y + 2)
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0 / 3, 2.0, 1.0])


def test_compute_class_weight_balanced_unordered():
    # Test compute_class_weight when classes are unordered
    classes = np.array([1, 0, 3])
    y = np.asarray([1, 0, 0, 3, 3, 3])

    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_counts = np.bincount(y)[classes]
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0, 1.0, 2.0 / 3])


def test_compute_class_weight_default():
    # Test for the case where no weight is given for a present class.
    # Current behaviour is to assign the unweighted classes a weight of 1.
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    classes_len = len(classes)

    # Test for non specified weights
    cw = compute_class_weight(None, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, np.ones(3))

    # Tests for partly specified weights
    cw = compute_class_weight({2: 1.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 1.0])

    cw = compute_class_weight({2: 1.5, 4: 0.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 0.5])


def test_compute_sample_weight():
    # Test (and demo) compute_sample_weight.
    # Test with balanced classes
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with user-defined weights
    sample_weight = compute_sample_weight({1: 2, 2: 1}, y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

    # Test with column vector of balanced classes
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with unbalanced classes
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight("balanced", y)
    expected_balanced = np.array(
        [0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 2.3333]
    )
    assert_array_almost_equal(sample_weight, expected_balanced, decimal=4)

    # Test with `None` weights
    sample_weight = compute_sample_weight(None, y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with multi-output of balanced classes
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with multi-output with user-defined weights
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight([{1: 2, 2: 1}, {0: 1, 1: 2}], y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    # Test with multi-output of unbalanced classes
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [3, -1]])
    sample_weight = compute_sample_weight("balanced", y)
    assert_array_almost_equal(sample_weight, expected_balanced**2, decimal=3)


def test_compute_sample_weight_with_subsample():
    # Test compute_sample_weight with subsamples specified.
    # Test with balanced classes and all samples present
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with column vector of balanced classes and all samples present
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test with a subsample
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y, indices=range(4))
    assert_array_almost_equal(sample_weight, [2.0 / 3, 2.0 / 3, 2.0 / 3, 2.0, 2.0, 2.0])

    # Test with a bootstrap subsample
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight("balanced", y, indices=[0, 1, 1, 2, 2, 3])
    expected_balanced = np.asarray([0.6, 0.6, 0.6, 3.0, 3.0, 3.0])
    assert_array_almost_equal(sample_weight, expected_balanced)

    # Test with a bootstrap subsample for multi-output
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight("balanced", y, indices=[0, 1, 1, 2, 2, 3])
    assert_array_almost_equal(sample_weight, expected_balanced**2)

    # Test with a missing class
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    # Test with a missing class for multi-output
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [2, 2]])
    sample_weight = compute_sample_weight("balanced", y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "y_type, class_weight, indices, err_msg",
    [
        (
            "single-output",
            {1: 2, 2: 1},
            range(4),
            "The only valid class_weight for subsampling is 'balanced'.",
        ),
        (
            "multi-output",
            {1: 2, 2: 1},
            None,
            "For multi-output, class_weight should be a list of dicts, or the string",
        ),
        (
            "multi-output",
            [{1: 2, 2: 1}],
            None,
            r"Got 1 element\(s\) while having 2 outputs",
        ),
    ],
)
def test_compute_sample_weight_errors(y_type, class_weight, indices, err_msg):
    # Test compute_sample_weight raises errors expected.
    # Invalid preset string
    y_single_output = np.asarray([1, 1, 1, 2, 2, 2])
    y_multi_output = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])

    y = y_single_output if y_type == "single-output" else y_multi_output
    with pytest.raises(ValueError, match=err_msg):
        compute_sample_weight(class_weight, y, indices=indices)


def test_compute_sample_weight_more_than_32():
    # Non-regression smoke test for #12146
    y = np.arange(50)  # more than 32 distinct classes
    indices = np.arange(50)  # use subsampling
    weight = compute_sample_weight("balanced", y, indices=indices)
    assert_array_almost_equal(weight, np.ones(y.shape[0]))


def test_class_weight_does_not_contains_more_classes():
    """Check that class_weight can contain more labels than in y.

    Non-regression test for #22413
    """
    tree = DecisionTreeClassifier(class_weight={0: 1, 1: 10, 2: 20})

    # Does not raise
    tree.fit([[0, 0, 1], [1, 0, 1], [1, 2, 0]], [0, 0, 1])


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_compute_sample_weight_sparse(csc_container):
    """Check that we can compute weight for sparse `y`."""
    y = csc_container(np.asarray([[0], [1], [1]]))
    sample_weight = compute_sample_weight("balanced", y)
    assert_allclose(sample_weight, [1.5, 0.75, 0.75])
