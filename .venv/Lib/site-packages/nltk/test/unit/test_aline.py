"""
Test Aline algorithm for aligning phonetic sequences
"""
from nltk.metrics import aline


def test_aline():
    result = aline.align("θin", "tenwis")
    expected = [[("θ", "t"), ("i", "e"), ("n", "n")]]

    assert result == expected

    result = aline.align("jo", "ʒə")
    expected = [[("j", "ʒ"), ("o", "ə")]]

    assert result == expected

    result = aline.align("pematesiweni", "pematesewen")
    expected = [
        [
            ("p", "p"),
            ("e", "e"),
            ("m", "m"),
            ("a", "a"),
            ("t", "t"),
            ("e", "e"),
            ("s", "s"),
            ("i", "e"),
            ("w", "w"),
            ("e", "e"),
            ("n", "n"),
        ]
    ]

    assert result == expected

    result = aline.align("tuwθ", "dentis")
    expected = [[("t", "t"), ("u", "i"), ("w", "-"), ("θ", "s")]]

    assert result == expected


def test_aline_delta():
    """
    Test aline for computing the difference between two segments
    """
    assert aline.delta("p", "q") == 20.0
    assert aline.delta("a", "A") == 0.0
