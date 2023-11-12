"""
Tests for IBM Model 3 training methods
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel, IBMModel3
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel3(unittest.TestCase):
    def test_set_uniform_distortion_probabilities(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model3 = IBMModel3(corpus, 0)

        # act
        model3.set_uniform_probabilities(corpus)

        # assert
        # expected_prob = 1.0 / length of target sentence
        self.assertEqual(model3.distortion_table[1][0][3][2], 1.0 / 2)
        self.assertEqual(model3.distortion_table[4][2][2][4], 1.0 / 4)

    def test_set_uniform_distortion_probabilities_of_non_domain_values(self):
        # arrange
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model3 = IBMModel3(corpus, 0)

        # act
        model3.set_uniform_probabilities(corpus)

        # assert
        # examine i and j values that are not in the training data domain
        self.assertEqual(model3.distortion_table[0][0][3][2], IBMModel.MIN_PROB)
        self.assertEqual(model3.distortion_table[9][2][2][4], IBMModel.MIN_PROB)
        self.assertEqual(model3.distortion_table[2][9][2][4], IBMModel.MIN_PROB)

    def test_prob_t_a_given_s(self):
        # arrange
        src_sentence = ["ich", "esse", "ja", "gern", "räucherschinken"]
        trg_sentence = ["i", "love", "to", "eat", "smoked", "ham"]
        corpus = [AlignedSent(trg_sentence, src_sentence)]
        alignment_info = AlignmentInfo(
            (0, 1, 4, 0, 2, 5, 5),
            [None] + src_sentence,
            ["UNUSED"] + trg_sentence,
            [[3], [1], [4], [], [2], [5, 6]],
        )

        distortion_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        )
        distortion_table[1][1][5][6] = 0.97  # i -> ich
        distortion_table[2][4][5][6] = 0.97  # love -> gern
        distortion_table[3][0][5][6] = 0.97  # to -> NULL
        distortion_table[4][2][5][6] = 0.97  # eat -> esse
        distortion_table[5][5][5][6] = 0.97  # smoked -> räucherschinken
        distortion_table[6][5][5][6] = 0.97  # ham -> räucherschinken

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
            "distortion_table": distortion_table,
            "fertility_table": fertility_table,
            "alignment_table": None,
        }

        model3 = IBMModel3(corpus, 0, probabilities)

        # act
        probability = model3.prob_t_a_given_s(alignment_info)

        # assert
        null_generation = 5 * pow(0.167, 1) * pow(0.833, 4)
        fertility = 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 2 * 0.999
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        distortion = 0.97 * 0.97 * 0.97 * 0.97 * 0.97 * 0.97
        expected_probability = (
            null_generation * fertility * lexical_translation * distortion
        )
        self.assertEqual(round(probability, 4), round(expected_probability, 4))
