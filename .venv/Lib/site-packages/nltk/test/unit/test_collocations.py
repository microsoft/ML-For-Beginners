from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

## Test bigram counters with discontinuous bigrams and repeated words

_EPSILON = 1e-8
SENT = "this this is is a a test test".split()


def close_enough(x, y):
    """Verify that two sequences of n-gram association values are within
    _EPSILON of each other.
    """

    return all(abs(x1[1] - y1[1]) <= _EPSILON for x1, y1 in zip(x, y))


def test_bigram2():
    b = BigramCollocationFinder.from_words(SENT)

    assert sorted(b.ngram_fd.items()) == [
        (("a", "a"), 1),
        (("a", "test"), 1),
        (("is", "a"), 1),
        (("is", "is"), 1),
        (("test", "test"), 1),
        (("this", "is"), 1),
        (("this", "this"), 1),
    ]
    assert sorted(b.word_fd.items()) == [("a", 2), ("is", 2), ("test", 2), ("this", 2)]

    assert len(SENT) == sum(b.word_fd.values()) == sum(b.ngram_fd.values()) + 1
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        [
            (("a", "a"), 1.0),
            (("a", "test"), 1.0),
            (("is", "a"), 1.0),
            (("is", "is"), 1.0),
            (("test", "test"), 1.0),
            (("this", "is"), 1.0),
            (("this", "this"), 1.0),
        ],
    )


def test_bigram3():
    b = BigramCollocationFinder.from_words(SENT, window_size=3)
    assert sorted(b.ngram_fd.items()) == sorted(
        [
            (("a", "test"), 3),
            (("is", "a"), 3),
            (("this", "is"), 3),
            (("a", "a"), 1),
            (("is", "is"), 1),
            (("test", "test"), 1),
            (("this", "this"), 1),
        ]
    )

    assert sorted(b.word_fd.items()) == sorted(
        [("a", 2), ("is", 2), ("test", 2), ("this", 2)]
    )

    assert (
        len(SENT) == sum(b.word_fd.values()) == (sum(b.ngram_fd.values()) + 2 + 1) / 2.0
    )
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        sorted(
            [
                (("a", "test"), 1.584962500721156),
                (("is", "a"), 1.584962500721156),
                (("this", "is"), 1.584962500721156),
                (("a", "a"), 0.0),
                (("is", "is"), 0.0),
                (("test", "test"), 0.0),
                (("this", "this"), 0.0),
            ]
        ),
    )


def test_bigram5():
    b = BigramCollocationFinder.from_words(SENT, window_size=5)
    assert sorted(b.ngram_fd.items()) == sorted(
        [
            (("a", "test"), 4),
            (("is", "a"), 4),
            (("this", "is"), 4),
            (("is", "test"), 3),
            (("this", "a"), 3),
            (("a", "a"), 1),
            (("is", "is"), 1),
            (("test", "test"), 1),
            (("this", "this"), 1),
        ]
    )
    assert sorted(b.word_fd.items()) == sorted(
        [("a", 2), ("is", 2), ("test", 2), ("this", 2)]
    )
    n_word_fd = sum(b.word_fd.values())
    n_ngram_fd = (sum(b.ngram_fd.values()) + 4 + 3 + 2 + 1) / 4.0
    assert len(SENT) == n_word_fd == n_ngram_fd
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        sorted(
            [
                (("a", "test"), 1.0),
                (("is", "a"), 1.0),
                (("this", "is"), 1.0),
                (("is", "test"), 0.5849625007211562),
                (("this", "a"), 0.5849625007211562),
                (("a", "a"), -1.0),
                (("is", "is"), -1.0),
                (("test", "test"), -1.0),
                (("this", "this"), -1.0),
            ]
        ),
    )
