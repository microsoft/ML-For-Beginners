import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn._loss.link import (
    _LINKS,
    HalfLogitLink,
    Interval,
    MultinomialLogit,
    _inclusive_low_high,
)

LINK_FUNCTIONS = list(_LINKS.values())


def test_interval_raises():
    """Test that interval with low > high raises ValueError."""
    with pytest.raises(
        ValueError, match="One must have low <= high; got low=1, high=0."
    ):
        Interval(1, 0, False, False)


@pytest.mark.parametrize(
    "interval",
    [
        Interval(0, 1, False, False),
        Interval(0, 1, False, True),
        Interval(0, 1, True, False),
        Interval(0, 1, True, True),
        Interval(-np.inf, np.inf, False, False),
        Interval(-np.inf, np.inf, False, True),
        Interval(-np.inf, np.inf, True, False),
        Interval(-np.inf, np.inf, True, True),
        Interval(-10, -1, False, False),
        Interval(-10, -1, False, True),
        Interval(-10, -1, True, False),
        Interval(-10, -1, True, True),
    ],
)
def test_is_in_range(interval):
    # make sure low and high are always within the interval, used for linspace
    low, high = _inclusive_low_high(interval)

    x = np.linspace(low, high, num=10)
    assert interval.includes(x)

    # x contains lower bound
    assert interval.includes(np.r_[x, interval.low]) == interval.low_inclusive

    # x contains upper bound
    assert interval.includes(np.r_[x, interval.high]) == interval.high_inclusive

    # x contains upper and lower bound
    assert interval.includes(np.r_[x, interval.low, interval.high]) == (
        interval.low_inclusive and interval.high_inclusive
    )


@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_inverse_identity(link, global_random_seed):
    # Test that link of inverse gives identity.
    rng = np.random.RandomState(global_random_seed)
    link = link()
    n_samples, n_classes = 100, None
    # The values for `raw_prediction` are limited from -20 to 20 because in the
    # class `LogitLink` the term `expit(x)` comes very close to 1 for large
    # positive x and therefore loses precision.
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    elif isinstance(link, HalfLogitLink):
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples))
    else:
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples))

    assert_allclose(link.link(link.inverse(raw_prediction)), raw_prediction)
    y_pred = link.inverse(raw_prediction)
    assert_allclose(link.inverse(link.link(y_pred)), y_pred)


@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_out_argument(link):
    # Test that out argument gets assigned the result.
    rng = np.random.RandomState(42)
    link = link()
    n_samples, n_classes = 100, None
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.normal(loc=0, scale=10, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    else:
        # So far, the valid interval of raw_prediction is (-inf, inf) and
        # we do not need to distinguish.
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples))

    y_pred = link.inverse(raw_prediction, out=None)
    out = np.empty_like(raw_prediction)
    y_pred_2 = link.inverse(raw_prediction, out=out)
    assert_allclose(y_pred, out)
    assert_array_equal(out, y_pred_2)
    assert np.shares_memory(out, y_pred_2)

    out = np.empty_like(y_pred)
    raw_prediction_2 = link.link(y_pred, out=out)
    assert_allclose(raw_prediction, out)
    assert_array_equal(out, raw_prediction_2)
    assert np.shares_memory(out, raw_prediction_2)
