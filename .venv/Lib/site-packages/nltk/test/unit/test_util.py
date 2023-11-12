import pytest

from nltk.util import everygrams


@pytest.fixture
def everygram_input():
    """Form test data for tests."""
    return iter(["a", "b", "c"])


def test_everygrams_without_padding(everygram_input):
    expected_output = [
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
        ("b",),
        ("b", "c"),
        ("c",),
    ]
    output = list(everygrams(everygram_input))
    assert output == expected_output


def test_everygrams_max_len(everygram_input):
    expected_output = [
        ("a",),
        ("a", "b"),
        ("b",),
        ("b", "c"),
        ("c",),
    ]
    output = list(everygrams(everygram_input, max_len=2))
    assert output == expected_output


def test_everygrams_min_len(everygram_input):
    expected_output = [
        ("a", "b"),
        ("a", "b", "c"),
        ("b", "c"),
    ]
    output = list(everygrams(everygram_input, min_len=2))
    assert output == expected_output


def test_everygrams_pad_right(everygram_input):
    expected_output = [
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
        ("b",),
        ("b", "c"),
        ("b", "c", None),
        ("c",),
        ("c", None),
        ("c", None, None),
        (None,),
        (None, None),
        (None,),
    ]
    output = list(everygrams(everygram_input, max_len=3, pad_right=True))
    assert output == expected_output


def test_everygrams_pad_left(everygram_input):
    expected_output = [
        (None,),
        (None, None),
        (None, None, "a"),
        (None,),
        (None, "a"),
        (None, "a", "b"),
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
        ("b",),
        ("b", "c"),
        ("c",),
    ]
    output = list(everygrams(everygram_input, max_len=3, pad_left=True))
    assert output == expected_output
