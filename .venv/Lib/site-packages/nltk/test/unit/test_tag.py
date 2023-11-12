def test_basic():
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize

    result = pos_tag(word_tokenize("John's big idea isn't all that bad."))
    assert result == [
        ("John", "NNP"),
        ("'s", "POS"),
        ("big", "JJ"),
        ("idea", "NN"),
        ("is", "VBZ"),
        ("n't", "RB"),
        ("all", "PDT"),
        ("that", "DT"),
        ("bad", "JJ"),
        (".", "."),
    ]


def setup_module(module):
    import pytest

    pytest.importorskip("numpy")
