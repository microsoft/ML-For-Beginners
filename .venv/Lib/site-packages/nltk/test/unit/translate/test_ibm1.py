"""
Tests for IBM Model 1 training methods
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel, IBMModel1
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel1(unittest.TestCase):
    def test_set_uniform_translation_probabilities(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model1 = IBMModel1(corpus, 0)

        # act
        model1.set_uniform_probabilities(corpus)

        # assert
        # expected_prob = 1.0 / (target vocab size + 1)
        self.assertEqual(model1.translation_table["ham"]["eier"], 1.0 / 3)
        self.assertEqual(model1.translation_table["eggs"][None], 1.0 / 3)

    def test_set_uniform_translation_probabilities_of_non_domain_values(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model1 = IBMModel1(corpus, 0)

        # act
        model1.set_uniform_probabilities(corpus)

        # assert
        # examine target words that are not in the training data domain
        self.assertEqual(model1.translation_table["parrot"]["eier"], IBMModel.MIN_PROB)

    def test_prob_t_a_given_s(self):
        # arrange
        src_sentence = ["ich", "esse", "ja", "gern", "räucherschinken"]
        trg_sentence = ["i", "love", "to", "eat", "smoked", "ham"]
        corpus = [AlignedSent(trg_sentence, src_sentence)]
        alignment_info = AlignmentInfo(
            (0, 1, 4, 0, 2, 5, 5),
            [None] + src_sentence,
            ["UNUSED"] + trg_sentence,
            None,
        )

        translation_table = defaultdict(lambda: defaultdict(float))
        translation_table["i"]["ich"] = 0.98
        translation_table["love"]["gern"] = 0.98
        translation_table["to"][None] = 0.98
        translation_table["eat"]["esse"] = 0.98
        translation_table["smoked"]["räucherschinken"] = 0.98
        translation_table["ham"]["räucherschinken"] = 0.98

        model1 = IBMModel1(corpus, 0)
        model1.translation_table = translation_table

        # act
        probability = model1.prob_t_a_given_s(alignment_info)

        # assert
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        expected_probability = lexical_translation
        self.assertEqual(round(probability, 4), round(expected_probability, 4))
