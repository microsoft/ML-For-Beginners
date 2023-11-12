import pytest

from nltk.tag import hmm


def _wikipedia_example_hmm():
    # Example from wikipedia
    # (https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm)

    states = ["rain", "no rain"]
    symbols = ["umbrella", "no umbrella"]

    A = [[0.7, 0.3], [0.3, 0.7]]  # transition probabilities
    B = [[0.9, 0.1], [0.2, 0.8]]  # emission probabilities
    pi = [0.5, 0.5]  # initial probabilities

    seq = ["umbrella", "umbrella", "no umbrella", "umbrella", "umbrella"]
    seq = list(zip(seq, [None] * len(seq)))

    model = hmm._create_hmm_tagger(states, symbols, A, B, pi)
    return model, states, symbols, seq


def test_forward_probability():
    from numpy.testing import assert_array_almost_equal

    # example from p. 385, Huang et al
    model, states, symbols = hmm._market_hmm_example()
    seq = [("up", None), ("up", None)]
    expected = [[0.35, 0.02, 0.09], [0.1792, 0.0085, 0.0357]]

    fp = 2 ** model._forward_probability(seq)

    assert_array_almost_equal(fp, expected)


def test_forward_probability2():
    from numpy.testing import assert_array_almost_equal

    model, states, symbols, seq = _wikipedia_example_hmm()
    fp = 2 ** model._forward_probability(seq)

    # examples in wikipedia are normalized
    fp = (fp.T / fp.sum(axis=1)).T

    wikipedia_results = [
        [0.8182, 0.1818],
        [0.8834, 0.1166],
        [0.1907, 0.8093],
        [0.7308, 0.2692],
        [0.8673, 0.1327],
    ]

    assert_array_almost_equal(wikipedia_results, fp, 4)


def test_backward_probability():
    from numpy.testing import assert_array_almost_equal

    model, states, symbols, seq = _wikipedia_example_hmm()

    bp = 2 ** model._backward_probability(seq)
    # examples in wikipedia are normalized

    bp = (bp.T / bp.sum(axis=1)).T

    wikipedia_results = [
        # Forward-backward algorithm doesn't need b0_5,
        # so .backward_probability doesn't compute it.
        # [0.6469, 0.3531],
        [0.5923, 0.4077],
        [0.3763, 0.6237],
        [0.6533, 0.3467],
        [0.6273, 0.3727],
        [0.5, 0.5],
    ]

    assert_array_almost_equal(wikipedia_results, bp, 4)


def setup_module(module):
    pytest.importorskip("numpy")
