"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(request):
    """
    Fixture for generating intervals of types from a start value and a shift
    value that can be added to start to generate an endpoint
    """
    return request.param


class TestOverlaps:
    def test_overlaps_self(self, start_shift, closed):
        start, shift = start_shift
        interval = Interval(start, start + shift, closed)
        assert interval.overlaps(interval)

    def test_overlaps_nested(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + 3 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # nested intervals should always overlap
        assert interval1.overlaps(interval2)

    def test_overlaps_disjoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)

        # disjoint intervals should never overlap
        assert not interval1.overlaps(interval2)

    def test_overlaps_endpoint(self, start_shift, closed, other_closed):
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # overlap if shared endpoint is closed for both (overlap at a point)
        result = interval1.overlaps(interval2)
        expected = interval1.closed_right and interval2.closed_left
        assert result == expected

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, other):
        interval = Interval(0, 1)
        msg = f"`other` must be an Interval, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval.overlaps(other)


class TestContains:
    def test_contains_interval(self, inclusive_endpoints_fixture):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(0, 1, inclusive_endpoints_fixture)
        assert interval1 in interval1
        assert interval2 in interval2
        assert interval2 in interval1
        assert interval1 not in interval2 or inclusive_endpoints_fixture == "both"

    def test_contains_infinite_length(self):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(float("-inf"), float("inf"), "neither")
        assert interval1 in interval2
        assert interval2 not in interval1

    def test_contains_zero_length(self):
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(-1, -1, "both")
        interval3 = Interval(0.5, 0.5, "both")
        assert interval2 not in interval1
        assert interval3 in interval1
        assert interval2 not in interval3 and interval3 not in interval2
        assert interval1 not in interval2 and interval1 not in interval3

    @pytest.mark.parametrize(
        "type1",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    @pytest.mark.parametrize(
        "type2",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    def test_contains_mixed_types(self, type1, type2):
        interval1 = Interval(*type1)
        interval2 = Interval(*type2)
        if type1 == type2:
            assert interval1 in interval2
        else:
            msg = "^'<=' not supported between instances of"
            with pytest.raises(TypeError, match=msg):
                interval1 in interval2
