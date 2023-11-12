"""
Tests for IBM Model 2 training methods
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel, IBMModel2
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel2(unittest.TestCase):
    def test_set_uniform_alignment_probabilities(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model2 = IBMModel2(corpus, 0)

        # act
        model2.set_uniform_probabilities(corpus)

        # assert
        # expected_prob = 1.0 / (length of source sentence + 1)
        self.assertEqual(model2.alignment_table[0][1][3][2], 1.0 / 4)
        self.assertEqual(model2.alignment_table[2][4][2][4], 1.0 / 3)

    def test_set_uniform_alignment_probabilities_of_non_domain_values(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model2 = IBMModel2(corpus, 0)

        # act
        model2.set_uniform_probabilities(corpus)

        # assert
        # examine i and j values that are not in the training data domain
        self.assertEqual(model2.alignment_table[99][1][3][2], IBMModel.MIN_PROB)
        self.assertEqual(model2.alignment_table[2][99][2][4], IBMModel.MIN_PROB)

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

        alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        )
        alignment_table[0][3][5][6] = 0.97  # None -> to
        alignment_table[1][1][5][6] = 0.97  # ich -> i
        alignment_table[2][4][5][6] = 0.97  # esse -> eat
        alignment_table[4][2][5][6] = 0.97  # gern -> love
        alignment_table[5][5][5][6] = 0.96  # räucherschinken -> smoked
        alignment_table[5][6][5][6] = 0.96  # räucherschinken -> ham

        model2 = IBMModel2(corpus, 0)
        model2.translation_table = translation_table
        model2.alignment_table = alignment_table

        # act
        probability = model2.prob_t_a_given_s(alignment_info)

        # assert
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        alignment = 0.97 * 0.97 * 0.97 * 0.97 * 0.96 * 0.96
        expected_probability = lexical_translation * alignment
        self.assertEqual(round(probability, 4), round(expected_probability, 4))
