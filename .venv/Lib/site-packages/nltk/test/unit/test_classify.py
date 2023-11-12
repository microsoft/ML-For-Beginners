"""
Unit tests for nltk.classify. See also: nltk/test/classify.doctest
"""
import pytest

from nltk import classify

TRAIN = [
    (dict(a=1, b=1, c=1), "y"),
    (dict(a=1, b=1, c=1), "x"),
    (dict(a=1, b=1, c=0), "y"),
    (dict(a=0, b=1, c=1), "x"),
    (dict(a=0, b=1, c=1), "y"),
    (dict(a=0, b=0, c=1), "y"),
    (dict(a=0, b=1, c=0), "x"),
    (dict(a=0, b=0, c=0), "x"),
    (dict(a=0, b=1, c=1), "y"),
]

TEST = [
    (dict(a=1, b=0, c=1)),  # unseen
    (dict(a=1, b=0, c=0)),  # unseen
    (dict(a=0, b=1, c=1)),  # seen 3 times, labels=y,y,x
    (dict(a=0, b=1, c=0)),  # seen 1 time, label=x
]

RESULTS = [(0.16, 0.84), (0.46, 0.54), (0.41, 0.59), (0.76, 0.24)]


def assert_classifier_correct(algorithm):
    try:
        classifier = classify.MaxentClassifier.train(
            TRAIN, algorithm, trace=0, max_iter=1000
        )
    except (LookupError, AttributeError) as e:
        pytest.skip(str(e))

    for (px, py), featureset in zip(RESULTS, TEST):
        pdist = classifier.prob_classify(featureset)
        assert abs(pdist.prob("x") - px) < 1e-2, (pdist.prob("x"), px)
        assert abs(pdist.prob("y") - py) < 1e-2, (pdist.prob("y"), py)


def test_megam():
    assert_classifier_correct("MEGAM")


def test_tadm():
    assert_classifier_correct("TADM")
