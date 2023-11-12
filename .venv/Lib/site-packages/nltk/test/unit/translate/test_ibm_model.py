"""
Tests for common methods of IBM translation models
"""

import unittest
from collections import defaultdict

from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo


class TestIBMModel(unittest.TestCase):
    __TEST_SRC_SENTENCE = ["j'", "aime", "bien", "jambon"]
    __TEST_TRG_SENTENCE = ["i", "love", "ham"]

    def test_vocabularies_are_initialized(self):
        parallel_corpora = [
            AlignedSent(["one", "two", "three", "four"], ["un", "deux", "trois"]),
            AlignedSent(["five", "one", "six"], ["quatre", "cinq", "six"]),
            AlignedSent([], ["sept"]),
        ]

        ibm_model = IBMModel(parallel_corpora)
        self.assertEqual(len(ibm_model.src_vocab), 8)
        self.assertEqual(len(ibm_model.trg_vocab), 6)

    def test_vocabularies_are_initialized_even_with_empty_corpora(self):
        parallel_corpora = []

        ibm_model = IBMModel(parallel_corpora)
        self.assertEqual(len(ibm_model.src_vocab), 1)  # addition of NULL token
        self.assertEqual(len(ibm_model.trg_vocab), 0)

    def test_best_model2_alignment(self):
        # arrange
        sentence_pair = AlignedSent(
            TestIBMModel.__TEST_TRG_SENTENCE, TestIBMModel.__TEST_SRC_SENTENCE
        )
        # None and 'bien' have zero fertility
        translation_table = {
            "i": {"j'": 0.9, "aime": 0.05, "bien": 0.02, "jambon": 0.03, None: 0},
            "love": {"j'": 0.05, "aime": 0.9, "bien": 0.01, "jambon": 0.01, None: 0.03},
            "ham": {"j'": 0, "aime": 0.01, "bien": 0, "jambon": 0.99, None: 0},
        }
        alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.2)))
        )

        ibm_model = IBMModel([])
        ibm_model.translation_table = translation_table
        ibm_model.alignment_table = alignment_table

        # act
        a_info = ibm_model.best_model2_alignment(sentence_pair)

        # assert
        self.assertEqual(a_info.alignment[1:], (1, 2, 4))  # 0th element unused
        self.assertEqual(a_info.cepts, [[], [1], [2], [], [3]])

    def test_best_model2_alignment_does_not_change_pegged_alignment(self):
        # arrange
        sentence_pair = AlignedSent(
            TestIBMModel.__TEST_TRG_SENTENCE, TestIBMModel.__TEST_SRC_SENTENCE
        )
        translation_table = {
            "i": {"j'": 0.9, "aime": 0.05, "bien": 0.02, "jambon": 0.03, None: 0},
            "love": {"j'": 0.05, "aime": 0.9, "bien": 0.01, "jambon": 0.01, None: 0.03},
            "ham": {"j'": 0, "aime": 0.01, "bien": 0, "jambon": 0.99, None: 0},
        }
        alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.2)))
        )

        ibm_model = IBMModel([])
        ibm_model.translation_table = translation_table
        ibm_model.alignment_table = alignment_table

        # act: force 'love' to be pegged to 'jambon'
        a_info = ibm_model.best_model2_alignment(sentence_pair, 2, 4)
        # assert
        self.assertEqual(a_info.alignment[1:], (1, 4, 4))
        self.assertEqual(a_info.cepts, [[], [1], [], [], [2, 3]])

    def test_best_model2_alignment_handles_fertile_words(self):
        # arrange
        sentence_pair = AlignedSent(
            ["i", "really", ",", "really", "love", "ham"],
            TestIBMModel.__TEST_SRC_SENTENCE,
        )
        # 'bien' produces 2 target words: 'really' and another 'really'
        translation_table = {
            "i": {"j'": 0.9, "aime": 0.05, "bien": 0.02, "jambon": 0.03, None: 0},
            "really": {"j'": 0, "aime": 0, "bien": 0.9, "jambon": 0.01, None: 0.09},
            ",": {"j'": 0, "aime": 0, "bien": 0.3, "jambon": 0, None: 0.7},
            "love": {"j'": 0.05, "aime": 0.9, "bien": 0.01, "jambon": 0.01, None: 0.03},
            "ham": {"j'": 0, "aime": 0.01, "bien": 0, "jambon": 0.99, None: 0},
        }
        alignment_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.2)))
        )

        ibm_model = IBMModel([])
        ibm_model.translation_table = translation_table
        ibm_model.alignment_table = alignment_table

        # act
        a_info = ibm_model.best_model2_alignment(sentence_pair)

        # assert
        self.assertEqual(a_info.alignment[1:], (1, 3, 0, 3, 2, 4))
        self.assertEqual(a_info.cepts, [[3], [1], [5], [2, 4], [6]])

    def test_best_model2_alignment_handles_empty_src_sentence(self):
        # arrange
        sentence_pair = AlignedSent(TestIBMModel.__TEST_TRG_SENTENCE, [])
        ibm_model = IBMModel([])

        # act
        a_info = ibm_model.best_model2_alignment(sentence_pair)

        # assert
        self.assertEqual(a_info.alignment[1:], (0, 0, 0))
        self.assertEqual(a_info.cepts, [[1, 2, 3]])

    def test_best_model2_alignment_handles_empty_trg_sentence(self):
        # arrange
        sentence_pair = AlignedSent([], TestIBMModel.__TEST_SRC_SENTENCE)
        ibm_model = IBMModel([])

        # act
        a_info = ibm_model.best_model2_alignment(sentence_pair)

        # assert
        self.assertEqual(a_info.alignment[1:], ())
        self.assertEqual(a_info.cepts, [[], [], [], [], []])

    def test_neighboring_finds_neighbor_alignments(self):
        # arrange
        a_info = AlignmentInfo(
            (0, 3, 2),
            (None, "des", "œufs", "verts"),
            ("UNUSED", "green", "eggs"),
            [[], [], [2], [1]],
        )
        ibm_model = IBMModel([])

        # act
        neighbors = ibm_model.neighboring(a_info)

        # assert
        neighbor_alignments = set()
        for neighbor in neighbors:
            neighbor_alignments.add(neighbor.alignment)
        expected_alignments = {
            # moves
            (0, 0, 2),
            (0, 1, 2),
            (0, 2, 2),
            (0, 3, 0),
            (0, 3, 1),
            (0, 3, 3),
            # swaps
            (0, 2, 3),
            # original alignment
            (0, 3, 2),
        }
        self.assertEqual(neighbor_alignments, expected_alignments)

    def test_neighboring_sets_neighbor_alignment_info(self):
        # arrange
        a_info = AlignmentInfo(
            (0, 3, 2),
            (None, "des", "œufs", "verts"),
            ("UNUSED", "green", "eggs"),
            [[], [], [2], [1]],
        )
        ibm_model = IBMModel([])

        # act
        neighbors = ibm_model.neighboring(a_info)

        # assert: select a few particular alignments
        for neighbor in neighbors:
            if neighbor.alignment == (0, 2, 2):
                moved_alignment = neighbor
            elif neighbor.alignment == (0, 3, 2):
                swapped_alignment = neighbor

        self.assertEqual(moved_alignment.cepts, [[], [], [1, 2], []])
        self.assertEqual(swapped_alignment.cepts, [[], [], [2], [1]])

    def test_neighboring_returns_neighbors_with_pegged_alignment(self):
        # arrange
        a_info = AlignmentInfo(
            (0, 3, 2),
            (None, "des", "œufs", "verts"),
            ("UNUSED", "green", "eggs"),
            [[], [], [2], [1]],
        )
        ibm_model = IBMModel([])

        # act: peg 'eggs' to align with 'œufs'
        neighbors = ibm_model.neighboring(a_info, 2)

        # assert
        neighbor_alignments = set()
        for neighbor in neighbors:
            neighbor_alignments.add(neighbor.alignment)
        expected_alignments = {
            # moves
            (0, 0, 2),
            (0, 1, 2),
            (0, 2, 2),
            # no swaps
            # original alignment
            (0, 3, 2),
        }
        self.assertEqual(neighbor_alignments, expected_alignments)

    def test_hillclimb(self):
        # arrange
        initial_alignment = AlignmentInfo((0, 3, 2), None, None, None)

        def neighboring_mock(a, j):
            if a.alignment == (0, 3, 2):
                return {
                    AlignmentInfo((0, 2, 2), None, None, None),
                    AlignmentInfo((0, 1, 1), None, None, None),
                }
            elif a.alignment == (0, 2, 2):
                return {
                    AlignmentInfo((0, 3, 3), None, None, None),
                    AlignmentInfo((0, 4, 4), None, None, None),
                }
            return set()

        def prob_t_a_given_s_mock(a):
            prob_values = {
                (0, 3, 2): 0.5,
                (0, 2, 2): 0.6,
                (0, 1, 1): 0.4,
                (0, 3, 3): 0.6,
                (0, 4, 4): 0.7,
            }
            return prob_values.get(a.alignment, 0.01)

        ibm_model = IBMModel([])
        ibm_model.neighboring = neighboring_mock
        ibm_model.prob_t_a_given_s = prob_t_a_given_s_mock

        # act
        best_alignment = ibm_model.hillclimb(initial_alignment)

        # assert: hill climbing goes from (0, 3, 2) -> (0, 2, 2) -> (0, 4, 4)
        self.assertEqual(best_alignment.alignment, (0, 4, 4))

    def test_sample(self):
        # arrange
        sentence_pair = AlignedSent(
            TestIBMModel.__TEST_TRG_SENTENCE, TestIBMModel.__TEST_SRC_SENTENCE
        )
        ibm_model = IBMModel([])
        ibm_model.prob_t_a_given_s = lambda x: 0.001

        # act
        samples, best_alignment = ibm_model.sample(sentence_pair)

        # assert
        self.assertEqual(len(samples), 61)
