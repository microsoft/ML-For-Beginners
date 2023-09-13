import numpy as np
import pytest

from sklearn.utils._plotting import _interval_max_min_ratio, _validate_score_name


def metric():
    pass  # pragma: no cover


def neg_metric():
    pass  # pragma: no cover


@pytest.mark.parametrize(
    "score_name, scoring, negate_score, expected_score_name",
    [
        ("accuracy", None, False, "accuracy"),  # do not transform the name
        (None, "accuracy", False, "Accuracy"),  # capitalize the name
        (None, "accuracy", True, "Negative accuracy"),  # add "Negative"
        (None, "neg_mean_absolute_error", False, "Negative mean absolute error"),
        (None, "neg_mean_absolute_error", True, "Mean absolute error"),  # remove "neg_"
        ("MAE", "neg_mean_absolute_error", True, "MAE"),  # keep score_name
        (None, None, False, "Score"),  # default name
        (None, None, True, "Negative score"),  # default name but negated
        ("Some metric", metric, False, "Some metric"),  # do not transform the name
        ("Some metric", metric, True, "Some metric"),  # do not transform the name
        (None, metric, False, "Metric"),  # default name
        (None, metric, True, "Negative metric"),  # default name but negated
        ("Some metric", neg_metric, False, "Some metric"),  # do not transform the name
        ("Some metric", neg_metric, True, "Some metric"),  # do not transform the name
        (None, neg_metric, False, "Negative metric"),  # default name
        (None, neg_metric, True, "Metric"),  # default name but negated
    ],
)
def test_validate_score_name(score_name, scoring, negate_score, expected_score_name):
    """Check that we return the right score name."""
    assert (
        _validate_score_name(score_name, scoring, negate_score) == expected_score_name
    )


# In the following test, we check the value of the max to min ratio
# for parameter value intervals to check that using a decision threshold
# of 5. is a good heuristic to decide between linear and log scales on
# common ranges of parameter values.
@pytest.mark.parametrize(
    "data, lower_bound, upper_bound",
    [
        # Such a range could be clearly displayed with either log scale or linear
        # scale.
        (np.geomspace(0.1, 1, 5), 5, 6),
        # Checking that the ratio is still positive on a negative log scale.
        (-np.geomspace(0.1, 1, 10), 7, 8),
        # Evenly spaced parameter values lead to a ratio of 1.
        (np.linspace(0, 1, 5), 0.9, 1.1),
        # This is not exactly spaced on a log scale but we will benefit from treating
        # it as such for visualization.
        ([1, 2, 5, 10, 20, 50], 20, 40),
    ],
)
def test_inverval_max_min_ratio(data, lower_bound, upper_bound):
    assert lower_bound < _interval_max_min_ratio(data) < upper_bound
