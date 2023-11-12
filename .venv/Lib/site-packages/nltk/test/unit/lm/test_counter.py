# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import unittest

import pytest

from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams


class TestNgramCounter:
    """Tests for NgramCounter that only involve lookup, no modification."""

    @classmethod
    def setup_class(self):
        text = [list("abcd"), list("egdbe")]
        self.trigram_counter = NgramCounter(
            everygrams(sent, max_len=3) for sent in text
        )
        self.bigram_counter = NgramCounter(everygrams(sent, max_len=2) for sent in text)
        self.case = unittest.TestCase()

    def test_N(self):
        assert self.bigram_counter.N() == 16
        assert self.trigram_counter.N() == 21

    def test_counter_len_changes_with_lookup(self):
        assert len(self.bigram_counter) == 2
        self.bigram_counter[50]
        assert len(self.bigram_counter) == 3

    def test_ngram_order_access_unigrams(self):
        assert self.bigram_counter[1] == self.bigram_counter.unigrams

    def test_ngram_conditional_freqdist(self):
        case = unittest.TestCase()
        expected_trigram_contexts = [
            ("a", "b"),
            ("b", "c"),
            ("e", "g"),
            ("g", "d"),
            ("d", "b"),
        ]
        expected_bigram_contexts = [("a",), ("b",), ("d",), ("e",), ("c",), ("g",)]

        bigrams = self.trigram_counter[2]
        trigrams = self.trigram_counter[3]

        self.case.assertCountEqual(expected_bigram_contexts, bigrams.conditions())
        self.case.assertCountEqual(expected_trigram_contexts, trigrams.conditions())

    def test_bigram_counts_seen_ngrams(self):
        assert self.bigram_counter[["a"]]["b"] == 1
        assert self.bigram_counter[["b"]]["c"] == 1

    def test_bigram_counts_unseen_ngrams(self):
        assert self.bigram_counter[["b"]]["z"] == 0

    def test_unigram_counts_seen_words(self):
        assert self.bigram_counter["b"] == 2

    def test_unigram_counts_completely_unseen_words(self):
        assert self.bigram_counter["z"] == 0


class TestNgramCounterTraining:
    @classmethod
    def setup_class(self):
        self.counter = NgramCounter()
        self.case = unittest.TestCase()

    @pytest.mark.parametrize("case", ["", [], None])
    def test_empty_inputs(self, case):
        test = NgramCounter(case)
        assert 2 not in test
        assert test[1] == FreqDist()

    def test_train_on_unigrams(self):
        words = list("abcd")
        counter = NgramCounter([[(w,) for w in words]])

        assert not counter[3]
        assert not counter[2]
        self.case.assertCountEqual(words, counter[1].keys())

    def test_train_on_illegal_sentences(self):
        str_sent = ["Check", "this", "out", "!"]
        list_sent = [["Check", "this"], ["this", "out"], ["out", "!"]]

        with pytest.raises(TypeError):
            NgramCounter([str_sent])

        with pytest.raises(TypeError):
            NgramCounter([list_sent])

    def test_train_on_bigrams(self):
        bigram_sent = [("a", "b"), ("c", "d")]
        counter = NgramCounter([bigram_sent])
        assert not bool(counter[3])

    def test_train_on_mix(self):
        mixed_sent = [("a", "b"), ("c", "d"), ("e", "f", "g"), ("h",)]
        counter = NgramCounter([mixed_sent])
        unigrams = ["h"]
        bigram_contexts = [("a",), ("c",)]
        trigram_contexts = [("e", "f")]

        self.case.assertCountEqual(unigrams, counter[1].keys())
        self.case.assertCountEqual(bigram_contexts, counter[2].keys())
        self.case.assertCountEqual(trigram_contexts, counter[3].keys())
