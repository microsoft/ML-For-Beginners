"""
Tests for IBM Model 5 training methods
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel, IBMModel4, IBMModel5
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel5(unittest.TestCase):
    def test_set_uniform_vacancy_probabilities_of_max_displacements(self):
        # arrange
        src_classes = {"schinken": 0, "eier": 0, "spam": 1}
        trg_classes = {"ham": 0, "eggs": 1, "spam": 2}
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model5 = IBMModel5(corpus, 0, src_classes, trg_classes)

        # act
        model5.set_uniform_probabilities(corpus)

        # assert
        # number of vacancy difference values =
        #     2 * number of words in longest target sentence
        expected_prob = 1.0 / (2 * 4)

        # examine the boundary values for (dv, max_v, trg_class)
        self.assertEqual(model5.head_vacancy_table[4][4][0], expected_prob)
        self.assertEqual(model5.head_vacancy_table[-3][1][2], expected_prob)
        self.assertEqual(model5.non_head_vacancy_table[4][4][0], expected_prob)
        self.assertEqual(model5.non_head_vacancy_table[-3][1][2], expected_prob)

    def test_set_uniform_vacancy_probabilities_of_non_domain_values(self):
        # arrange
        src_classes = {"schinken": 0, "eier": 0, "spam": 1}
        trg_classes = {"ham": 0, "eggs": 1, "spam": 2}
        corpus = [
            AlignedSent(["ham", "eggs"], ["schinken", "schinken", "eier"]),
            AlignedSent(["spam", "spam", "spam", "spam"], ["spam", "spam"]),
        ]
        model5 = IBMModel5(corpus, 0, src_classes, trg_classes)

        # act
        model5.set_uniform_probabilities(corpus)

        # assert
        # examine dv and max_v values that are not in the training data domain
        self.assertEqual(model5.head_vacancy_table[5][4][0], IBMModel.MIN_PROB)
        self.assertEqual(model5.head_vacancy_table[-4][1][2], IBMModel.MIN_PROB)
        self.assertEqual(model5.head_vacancy_table[4][0][0], IBMModel.MIN_PROB)
        self.assertEqual(model5.non_head_vacancy_table[5][4][0], IBMModel.MIN_PROB)
        self.assertEqual(model5.non_head_vacancy_table[-4][1][2], IBMModel.MIN_PROB)

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

        head_vacancy_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        head_vacancy_table[1 - 0][6][3] = 0.97  # ich -> i
        head_vacancy_table[3 - 0][5][4] = 0.97  # esse -> eat
        head_vacancy_table[1 - 2][4][4] = 0.97  # gern -> love
        head_vacancy_table[2 - 0][2][1] = 0.97  # räucherschinken -> smoked

        non_head_vacancy_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        non_head_vacancy_table[1 - 0][1][0] = 0.96  # räucherschinken -> ham

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
            "fertility_table": fertility_table,
            "head_vacancy_table": head_vacancy_table,
            "non_head_vacancy_table": non_head_vacancy_table,
            "head_distortion_table": None,
            "non_head_distortion_table": None,
            "alignment_table": None,
        }

        model5 = IBMModel5(corpus, 0, src_classes, trg_classes, probabilities)

        # act
        probability = model5.prob_t_a_given_s(alignment_info)

        # assert
        null_generation = 5 * pow(0.167, 1) * pow(0.833, 4)
        fertility = 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 1 * 0.99 * 2 * 0.999
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        vacancy = 0.97 * 0.97 * 1 * 0.97 * 0.97 * 0.96
        expected_probability = (
            null_generation * fertility * lexical_translation * vacancy
        )
        self.assertEqual(round(probability, 4), round(expected_probability, 4))

    def test_prune(self):
        # arrange
        alignment_infos = [
            AlignmentInfo((1, 1), None, None, None),
            AlignmentInfo((1, 2), None, None, None),
            AlignmentInfo((2, 1), None, None, None),
            AlignmentInfo((2, 2), None, None, None),
            AlignmentInfo((0, 0), None, None, None),
        ]
        min_factor = IBMModel5.MIN_SCORE_FACTOR
        best_score = 0.9
        scores = {
            (1, 1): min(min_factor * 1.5, 1) * best_score,  # above threshold
            (1, 2): best_score,
            (2, 1): min_factor * best_score,  # at threshold
            (2, 2): min_factor * best_score * 0.5,  # low score
            (0, 0): min(min_factor * 1.1, 1) * 1.2,  # above threshold
        }
        corpus = [AlignedSent(["a"], ["b"])]
        original_prob_function = IBMModel4.model4_prob_t_a_given_s
        # mock static method
        IBMModel4.model4_prob_t_a_given_s = staticmethod(
            lambda a, model: scores[a.alignment]
        )
        model5 = IBMModel5(corpus, 0, None, None)

        # act
        pruned_alignments = model5.prune(alignment_infos)

        # assert
        self.assertEqual(len(pruned_alignments), 3)

        # restore static method
        IBMModel4.model4_prob_t_a_given_s = original_prob_function
