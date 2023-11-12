# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import unittest
from collections import Counter
from timeit import timeit

from nltk.lm import Vocabulary


class NgramModelVocabularyTests(unittest.TestCase):
    """tests Vocabulary Class"""

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocabulary(
            ["z", "a", "b", "c", "f", "d", "e", "g", "a", "d", "b", "e", "w"],
            unk_cutoff=2,
        )

    def test_truthiness(self):
        self.assertTrue(self.vocab)

    def test_cutoff_value_set_correctly(self):
        self.assertEqual(self.vocab.cutoff, 2)

    def test_unable_to_change_cutoff(self):
        with self.assertRaises(AttributeError):
            self.vocab.cutoff = 3

    def test_cutoff_setter_checks_value(self):
        with self.assertRaises(ValueError) as exc_info:
            Vocabulary("abc", unk_cutoff=0)
        expected_error_msg = "Cutoff value cannot be less than 1. Got: 0"
        self.assertEqual(expected_error_msg, str(exc_info.exception))

    def test_counts_set_correctly(self):
        self.assertEqual(self.vocab.counts["a"], 2)
        self.assertEqual(self.vocab.counts["b"], 2)
        self.assertEqual(self.vocab.counts["c"], 1)

    def test_membership_check_respects_cutoff(self):
        # a was seen 2 times, so it should be considered part of the vocabulary
        self.assertTrue("a" in self.vocab)
        # "c" was seen once, it shouldn't be considered part of the vocab
        self.assertFalse("c" in self.vocab)
        # "z" was never seen at all, also shouldn't be considered in the vocab
        self.assertFalse("z" in self.vocab)

    def test_vocab_len_respects_cutoff(self):
        # Vocab size is the number of unique tokens that occur at least as often
        # as the cutoff value, plus 1 to account for unknown words.
        self.assertEqual(5, len(self.vocab))

    def test_vocab_iter_respects_cutoff(self):
        vocab_counts = ["a", "b", "c", "d", "e", "f", "g", "w", "z"]
        vocab_items = ["a", "b", "d", "e", "<UNK>"]

        self.assertCountEqual(vocab_counts, list(self.vocab.counts.keys()))
        self.assertCountEqual(vocab_items, list(self.vocab))

    def test_update_empty_vocab(self):
        empty = Vocabulary(unk_cutoff=2)
        self.assertEqual(len(empty), 0)
        self.assertFalse(empty)
        self.assertIn(empty.unk_label, empty)

        empty.update(list("abcde"))
        self.assertIn(empty.unk_label, empty)

    def test_lookup(self):
        self.assertEqual(self.vocab.lookup("a"), "a")
        self.assertEqual(self.vocab.lookup("c"), "<UNK>")

    def test_lookup_iterables(self):
        self.assertEqual(self.vocab.lookup(["a", "b"]), ("a", "b"))
        self.assertEqual(self.vocab.lookup(("a", "b")), ("a", "b"))
        self.assertEqual(self.vocab.lookup(("a", "c")), ("a", "<UNK>"))
        self.assertEqual(
            self.vocab.lookup(map(str, range(3))), ("<UNK>", "<UNK>", "<UNK>")
        )

    def test_lookup_empty_iterables(self):
        self.assertEqual(self.vocab.lookup(()), ())
        self.assertEqual(self.vocab.lookup([]), ())
        self.assertEqual(self.vocab.lookup(iter([])), ())
        self.assertEqual(self.vocab.lookup(n for n in range(0, 0)), ())

    def test_lookup_recursive(self):
        self.assertEqual(
            self.vocab.lookup([["a", "b"], ["a", "c"]]), (("a", "b"), ("a", "<UNK>"))
        )
        self.assertEqual(self.vocab.lookup([["a", "b"], "c"]), (("a", "b"), "<UNK>"))
        self.assertEqual(self.vocab.lookup([[[[["a", "b"]]]]]), ((((("a", "b"),),),),))

    def test_lookup_None(self):
        with self.assertRaises(TypeError):
            self.vocab.lookup(None)
        with self.assertRaises(TypeError):
            list(self.vocab.lookup([None, None]))

    def test_lookup_int(self):
        with self.assertRaises(TypeError):
            self.vocab.lookup(1)
        with self.assertRaises(TypeError):
            list(self.vocab.lookup([1, 2]))

    def test_lookup_empty_str(self):
        self.assertEqual(self.vocab.lookup(""), "<UNK>")

    def test_eqality(self):
        v1 = Vocabulary(["a", "b", "c"], unk_cutoff=1)
        v2 = Vocabulary(["a", "b", "c"], unk_cutoff=1)
        v3 = Vocabulary(["a", "b", "c"], unk_cutoff=1, unk_label="blah")
        v4 = Vocabulary(["a", "b"], unk_cutoff=1)

        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertNotEqual(v1, v4)

    def test_str(self):
        self.assertEqual(
            str(self.vocab), "<Vocabulary with cutoff=2 unk_label='<UNK>' and 5 items>"
        )

    def test_creation_with_counter(self):
        self.assertEqual(
            self.vocab,
            Vocabulary(
                Counter(
                    ["z", "a", "b", "c", "f", "d", "e", "g", "a", "d", "b", "e", "w"]
                ),
                unk_cutoff=2,
            ),
        )

    @unittest.skip(
        reason="Test is known to be flaky as it compares (runtime) performance."
    )
    def test_len_is_constant(self):
        # Given an obviously small and an obviously large vocabulary.
        small_vocab = Vocabulary("abcde")
        from nltk.corpus.europarl_raw import english

        large_vocab = Vocabulary(english.words())

        # If we time calling `len` on them.
        small_vocab_len_time = timeit("len(small_vocab)", globals=locals())
        large_vocab_len_time = timeit("len(large_vocab)", globals=locals())

        # The timing should be the same order of magnitude.
        self.assertAlmostEqual(small_vocab_len_time, large_vocab_len_time, places=1)
