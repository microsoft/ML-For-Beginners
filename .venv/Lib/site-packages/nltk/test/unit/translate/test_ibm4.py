"""
Tests for IBM Model 4 training methods
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel, IBMModel4
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel4(unittest.TestCase):
    def test_set_uniform_distortion_probabilities_of_max_displacements(self):
        # arrange
        src_classes = {"schinken": 0, "eier": 0, "spam": 1}
        trg_classes = {"ham": 0, "eggs": 1, "spam": 2}
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model4 = IBMModel4(corpus, 0, src_classes, trg_classes)

        # act
        model4.set_uniform_probabilities(corpus)

        # assert
        # number of displacement values =
        #     2 *(number of words in longest target sentence - 1)
        expected_prob = 1.0 / (2 * (4 - 1))

        # examine the boundary values for (displacement, src_class, trg_class)
        self.assertEqual(model4.head_distortion_table[3][0][0], expected_prob)
        self.assertEqual(model4.head_distortion_table[-3][1][2], expected_prob)
        self.assertEqual(model4.non_head_distortion_table[3][0], expected_prob)
        self.assertEqual(model4.non_head_distortion_table[-3][2], expected_prob)

    def test_set_uniform_distortion_probabilities_of_non_domain_values(self):
        # arrange
        src_classes = {"schinken": 0, "eier": 0, "spam": 1}
        trg_classes = {"ham": 0, "eggs": 1, "spam": 2}
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model4 = IBMModel4(corpus, 0, src_classes, trg_classes)

        # act
        model4.set_uniform_probabilities(corpus)

        # assert
        # examine displacement values that are not in the training data domain
        self.assertEqual(model4.head_distortion_table[4][0][0], IBMModel.MIN_PROB)
        self.assertEqual(model4.head_distortion_table[100][1][2], IBMModel.MIN_PROB)
        self.assertEqual(model4.non_head_distortion_table[4][0], IBMModel.MIN_PROB)
        self.assertEqual(model4.non_head_distortion_table[100][2], IBMModel.MIN_PROB)

    def test_prob_t_a_given_s(self):
        # arrange
        src_sentence = ["ich", "esse", "ja", "gern", "räucherschinken"]
        trg_sentence = ["i", "love", "to", "eat", "smoked", "ham"]
        src_classes = {"räucherschinken": 0, "ja": 1, "ich": 2, "esse": 3, "gern": 4}
        trg_classes = {"ham": 0, "smoked": 1, "i": 3, "love": 4, "to": 2, "eat": 4}
        corpus = [AlignedSent(trg_sentence, src_sentence)]
        alignment_info = AlignmentInfo(
            (0, 1, 4, 0, 2, 5, 5),
            [None] + src_sentence,
            ["UNUSED"] + trg_sentence,
            [[3], [1], [4], [], [2], [5, 6]],
        )

        head_distortion_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        head_distortion_table[1][None][3] = 0.97  # None, i
        head_distortion_table[3][2][4] = 0.97  # ich, eat
        head_distortion_table[-2][3][4] = 0.97  # esse, love
        head_distortion_table[3][4][1] = 0.97  # gern, smoked

        non_head_distortion_table = defaultdict(lambda: defaultdict(float))
        non_head_distortion_table[1][0] = 0.96  # ham

        translation_table = defaultdict(lambda: defaultdict(float))
        translation_table["i"]["ich"] = 0.98
        translation_table["love"]["gern"] = 0.98
        translation_table["to"][None] = 0.98
        translation_table["eat"]["esse"] = 0.98
        translation_table["smoked"]["räucherschinken"] = 0.98
        translation_table["ham"]["räucherschinken"] = 0.98

        fertility_table = defaultdict(lambda: defaultdict(float))
        fertility_table[1]["ich"] = 0.99
        fertility_table[1]["esse"] = 0.99
        fertility_table[0]["ja"] = 0.99
        fertility_table[1]["gern"] = 0.99
        fertility_table[2]["räucherschinken"] = 0.999
        fertility_table[1][None] = 0.99

        probabilities = {
            "p1": 0.167,
            "translation_table": translation_table,
            "head_distortion_table": head_distortion_table,
            "non_head_distortion_table": non_head_distortion_table,
            "fertility_table": fertility_table,
            "alignment_table": None,
        }

        model4 = IBMModel4(corpus, 0, src_classes, trg_classes, probabilities)

        # act
        probability = model4.prob_t_a_given_s(alignment_info)

        # assert
        null_generation = 5 * pow(0.167, 1) * pow(0.833, 4)
        fertility = 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 2 * 0.999
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        distortion = 0.97 * 0.97 * 1 * 0.97 * 0.97 * 0.96
        expected_probability = (
            null_generation * fertility * lexical_translation * distortion
        )
        self.assertEqual(round(probability, 4), round(expected_probability, 4))
