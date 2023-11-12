# Natural Language Toolkit: IBM Model Core
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Common methods and classes for all IBM models. See ``IBMModel1``,
``IBMModel2``, ``IBMModel3``, ``IBMModel4``, and ``IBMModel5``
for specific implementations.

The IBM models are a series of generative models that learn lexical
translation probabilities, p(target language word|source language word),
given a sentence-aligned parallel corpus.

The models increase in sophistication from model 1 to 5. Typically, the
output of lower models is used to seed the higher models. All models
use the Expectation-Maximization (EM) algorithm to learn various
probability tables.

Words in a sentence are one-indexed. The first word of a sentence has
position 1, not 0. Index 0 is reserved in the source sentence for the
NULL token. The concept of position does not apply to NULL, but it is
indexed at 0 by convention.

Each target word is aligned to exactly one source word or the NULL
token.

References:
Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.

Peter E Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and
Robert L. Mercer. 1993. The Mathematics of Statistical Machine
Translation: Parameter Estimation. Computational Linguistics, 19 (2),
263-311.
"""

from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil


def longest_target_sentence_length(sentence_aligned_corpus):
    """
    :param sentence_aligned_corpus: Parallel corpus under consideration
    :type sentence_aligned_corpus: list(AlignedSent)
    :return: Number of words in the longest target language sentence
        of ``sentence_aligned_corpus``
    """
    max_m = 0
    for aligned_sentence in sentence_aligned_corpus:
        m = len(aligned_sentence.words)
        max_m = max(m, max_m)
    return max_m


class IBMModel:
    """
    Abstract base class for all IBM models
    """

    # Avoid division by zero and precision errors by imposing a minimum
    # value for probabilities. Note that this approach is theoretically
    # incorrect, since it may create probabilities that sum to more
    # than 1. In practice, the contribution of probabilities with MIN_PROB
    # is tiny enough that the value of MIN_PROB can be treated as zero.
    MIN_PROB = 1.0e-12  # GIZA++ is more liberal and uses 1.0e-7

    def __init__(self, sentence_aligned_corpus):
        self.init_vocab(sentence_aligned_corpus)
        self.reset_probabilities()

    def reset_probabilities(self):
        self.translation_table = defaultdict(
            lambda: defaultdict(lambda: IBMModel.MIN_PROB)
        )
        """
        dict[str][str]: float. Probability(target word | source word).
        Values accessed as ``translation_table[target_word][source_word]``.
        """

        self.alignment_table = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: IBMModel.MIN_PROB))
            )
        )
        """
        dict[int][int][int][int]: float. Probability(i | j,l,m).
        Values accessed as ``alignment_table[i][j][l][m]``.
        Used in model 2 and hill climbing in models 3 and above
        """

        self.fertility_table = defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))
        """
        dict[int][str]: float. Probability(fertility | source word).
        Values accessed as ``fertility_table[fertility][source_word]``.
        Used in model 3 and higher.
        """

        self.p1 = 0.5
        """
        Probability that a generated word requires another target word
        that is aligned to NULL.
        Used in model 3 and higher.
        """

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        """
        Initialize probability tables to a uniform distribution

        Derived classes should implement this accordingly.
        """
        pass

    def init_vocab(self, sentence_aligned_corpus):
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in sentence_aligned_corpus:
            trg_vocab.update(aligned_sentence.words)
            src_vocab.update(aligned_sentence.mots)
        # Add the NULL token
        src_vocab.add(None)

        self.src_vocab = src_vocab
        """
        set(str): All source language words used in training
        """

        self.trg_vocab = trg_vocab
        """
        set(str): All target language words used in training
        """

    def sample(self, sentence_pair):
        """
        Sample the most probable alignments from the entire alignment
        space

        First, determine the best alignment according to IBM Model 2.
        With this initial alignment, use hill climbing to determine the
        best alignment according to a higher IBM Model. Add this
        alignment and its neighbors to the sample set. Repeat this
        process with other initial alignments obtained by pegging an
        alignment point.

        Hill climbing may be stuck in a local maxima, hence the pegging
        and trying out of different alignments.

        :param sentence_pair: Source and target language sentence pair
            to generate a sample of alignments from
        :type sentence_pair: AlignedSent

        :return: A set of best alignments represented by their ``AlignmentInfo``
            and the best alignment of the set for convenience
        :rtype: set(AlignmentInfo), AlignmentInfo
        """
        sampled_alignments = set()
        l = len(sentence_pair.mots)
        m = len(sentence_pair.words)

        # Start from the best model 2 alignment
        initial_alignment = self.best_model2_alignment(sentence_pair)
        potential_alignment = self.hillclimb(initial_alignment)
        sampled_alignments.update(self.neighboring(potential_alignment))
        best_alignment = potential_alignment

        # Start from other model 2 alignments,
        # with the constraint that j is aligned (pegged) to i
        for j in range(1, m + 1):
            for i in range(0, l + 1):
                initial_alignment = self.best_model2_alignment(sentence_pair, j, i)
                potential_alignment = self.hillclimb(initial_alignment, j)
                neighbors = self.neighboring(potential_alignment, j)
                sampled_alignments.update(neighbors)
                if potential_alignment.score > best_alignment.score:
                    best_alignment = potential_alignment

        return sampled_alignments, best_alignment

    def best_model2_alignment(self, sentence_pair, j_pegged=None, i_pegged=0):
        """
        Finds the best alignment according to IBM Model 2

        Used as a starting point for hill climbing in Models 3 and
        above, because it is easier to compute than the best alignments
        in higher models

        :param sentence_pair: Source and target language sentence pair
            to be word-aligned
        :type sentence_pair: AlignedSent

        :param j_pegged: If specified, the alignment point of j_pegged
            will be fixed to i_pegged
        :type j_pegged: int

        :param i_pegged: Alignment point to j_pegged
        :type i_pegged: int
        """
        src_sentence = [None] + sentence_pair.mots
        trg_sentence = ["UNUSED"] + sentence_pair.words  # 1-indexed

        l = len(src_sentence) - 1  # exclude NULL
        m = len(trg_sentence) - 1

        alignment = [0] * (m + 1)  # init all alignments to NULL
        cepts = [[] for i in range(l + 1)]  # init all cepts to empty list

        for j in range(1, m + 1):
            if j == j_pegged:
                # use the pegged alignment instead of searching for best one
                best_i = i_pegged
            else:
                best_i = 0
                max_alignment_prob = IBMModel.MIN_PROB
                t = trg_sentence[j]

                for i in range(0, l + 1):
                    s = src_sentence[i]
                    alignment_prob = (
                        self.translation_table[t][s] * self.alignment_table[i][j][l][m]
                    )

                    if alignment_prob >= max_alignment_prob:
                        max_alignment_prob = alignment_prob
                        best_i = i

            alignment[j] = best_i
            cepts[best_i].append(j)

        return AlignmentInfo(
            tuple(alignment), tuple(src_sentence), tuple(trg_sentence), cepts
        )

    def hillclimb(self, alignment_info, j_pegged=None):
        """
        Starting from the alignment in ``alignment_info``, look at
        neighboring alignments iteratively for the best one

        There is no guarantee that the best alignment in the alignment
        space will be found, because the algorithm might be stuck in a
        local maximum.

        :param j_pegged: If specified, the search will be constrained to
            alignments where ``j_pegged`` remains unchanged
        :type j_pegged: int

        :return: The best alignment found from hill climbing
        :rtype: AlignmentInfo
        """
        alignment = alignment_info  # alias with shorter name
        max_probability = self.prob_t_a_given_s(alignment)

        while True:
            old_alignment = alignment
            for neighbor_alignment in self.neighboring(alignment, j_pegged):
                neighbor_probability = self.prob_t_a_given_s(neighbor_alignment)

                if neighbor_probability > max_probability:
                    alignment = neighbor_alignment
                    max_probability = neighbor_probability

            if alignment == old_alignment:
                # Until there are no better alignments
                break

        alignment.score = max_probability
        return alignment

    def neighboring(self, alignment_info, j_pegged=None):
        """
        Determine the neighbors of ``alignment_info``, obtained by
        moving or swapping one alignment point

        :param j_pegged: If specified, neighbors that have a different
            alignment point from j_pegged will not be considered
        :type j_pegged: int

        :return: A set neighboring alignments represented by their
            ``AlignmentInfo``
        :rtype: set(AlignmentInfo)
        """
        neighbors = set()

        l = len(alignment_info.src_sentence) - 1  # exclude NULL
        m = len(alignment_info.trg_sentence) - 1
        original_alignment = alignment_info.alignment
        original_cepts = alignment_info.cepts

        for j in range(1, m + 1):
            if j != j_pegged:
                # Add alignments that differ by one alignment point
                for i in range(0, l + 1):
                    new_alignment = list(original_alignment)
                    new_cepts = deepcopy(original_cepts)
                    old_i = original_alignment[j]

                    # update alignment
                    new_alignment[j] = i

                    # update cepts
                    insort_left(new_cepts[i], j)
                    new_cepts[old_i].remove(j)

                    new_alignment_info = AlignmentInfo(
                        tuple(new_alignment),
                        alignment_info.src_sentence,
                        alignment_info.trg_sentence,
                        new_cepts,
                    )
                    neighbors.add(new_alignment_info)

        for j in range(1, m + 1):
            if j != j_pegged:
                # Add alignments that have two alignment points swapped
                for other_j in range(1, m + 1):
                    if other_j != j_pegged and other_j != j:
                        new_alignment = list(original_alignment)
                        new_cepts = deepcopy(original_cepts)
                        other_i = original_alignment[other_j]
                        i = original_alignment[j]

                        # update alignments
                        new_alignment[j] = other_i
                        new_alignment[other_j] = i

                        # update cepts
                        new_cepts[other_i].remove(other_j)
                        insort_left(new_cepts[other_i], j)
                        new_cepts[i].remove(j)
                        insort_left(new_cepts[i], other_j)

                        new_alignment_info = AlignmentInfo(
                            tuple(new_alignment),
                            alignment_info.src_sentence,
                            alignment_info.trg_sentence,
                            new_cepts,
                        )
                        neighbors.add(new_alignment_info)

        return neighbors

    def maximize_lexical_translation_probabilities(self, counts):
        for t, src_words in counts.t_given_s.items():
            for s in src_words:
                estimate = counts.t_given_s[t][s] / counts.any_t_given_s[s]
                self.translation_table[t][s] = max(estimate, IBMModel.MIN_PROB)

    def maximize_fertility_probabilities(self, counts):
        for phi, src_words in counts.fertility.items():
            for s in src_words:
                estimate = counts.fertility[phi][s] / counts.fertility_for_any_phi[s]
                self.fertility_table[phi][s] = max(estimate, IBMModel.MIN_PROB)

    def maximize_null_generation_probabilities(self, counts):
        p1_estimate = counts.p1 / (counts.p1 + counts.p0)
        p1_estimate = max(p1_estimate, IBMModel.MIN_PROB)
        # Clip p1 if it is too large, because p0 = 1 - p1 should not be
        # smaller than MIN_PROB
        self.p1 = min(p1_estimate, 1 - IBMModel.MIN_PROB)

    def prob_of_alignments(self, alignments):
        probability = 0
        for alignment_info in alignments:
            probability += self.prob_t_a_given_s(alignment_info)
        return probability

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence

        All required information is assumed to be in ``alignment_info``
        and self.

        Derived classes should override this method
        """
        return 0.0


class AlignmentInfo:
    """
    Helper data object for training IBM Models 3 and up

    Read-only. For a source sentence and its counterpart in the target
    language, this class holds information about the sentence pair's
    alignment, cepts, and fertility.

    Warning: Alignments are one-indexed here, in contrast to
    nltk.translate.Alignment and AlignedSent, which are zero-indexed
    This class is not meant to be used outside of IBM models.
    """

    def __init__(self, alignment, src_sentence, trg_sentence, cepts):
        if not isinstance(alignment, tuple):
            raise TypeError(
                "The alignment must be a tuple because it is used "
                "to uniquely identify AlignmentInfo objects."
            )

        self.alignment = alignment
        """
        tuple(int): Alignment function. ``alignment[j]`` is the position
        in the source sentence that is aligned to the position j in the
        target sentence.
        """

        self.src_sentence = src_sentence
        """
        tuple(str): Source sentence referred to by this object.
        Should include NULL token (None) in index 0.
        """

        self.trg_sentence = trg_sentence
        """
        tuple(str): Target sentence referred to by this object.
        Should have a dummy element in index 0 so that the first word
        starts from index 1.
        """

        self.cepts = cepts
        """
        list(list(int)): The positions of the target words, in
        ascending order, aligned to a source word position. For example,
        cepts[4] = (2, 3, 7) means that words in positions 2, 3 and 7
        of the target sentence are aligned to the word in position 4 of
        the source sentence
        """

        self.score = None
        """
        float: Optional. Probability of alignment, as defined by the
        IBM model that assesses this alignment
        """

    def fertility_of_i(self, i):
        """
        Fertility of word in position ``i`` of the source sentence
        """
        return len(self.cepts[i])

    def is_head_word(self, j):
        """
        :return: Whether the word in position ``j`` of the target
            sentence is a head word
        """
        i = self.alignment[j]
        return self.cepts[i][0] == j

    def center_of_cept(self, i):
        """
        :return: The ceiling of the average positions of the words in
            the tablet of cept ``i``, or 0 if ``i`` is None
        """
        if i is None:
            return 0

        average_position = sum(self.cepts[i]) / len(self.cepts[i])
        return int(ceil(average_position))

    def previous_cept(self, j):
        """
        :return: The previous cept of ``j``, or None if ``j`` belongs to
            the first cept
        """
        i = self.alignment[j]
        if i == 0:
            raise ValueError(
                "Words aligned to NULL cannot have a previous "
                "cept because NULL has no position"
            )
        previous_cept = i - 1
        while previous_cept > 0 and self.fertility_of_i(previous_cept) == 0:
            previous_cept -= 1

        if previous_cept <= 0:
            previous_cept = None
        return previous_cept

    def previous_in_tablet(self, j):
        """
        :return: The position of the previous word that is in the same
            tablet as ``j``, or None if ``j`` is the first word of the
            tablet
        """
        i = self.alignment[j]
        tablet_position = self.cepts[i].index(j)
        if tablet_position == 0:
            return None
        return self.cepts[i][tablet_position - 1]

    def zero_indexed_alignment(self):
        """
        :return: Zero-indexed alignment, suitable for use in external
            ``nltk.translate`` modules like ``nltk.translate.Alignment``
        :rtype: list(tuple)
        """
        zero_indexed_alignment = []
        for j in range(1, len(self.trg_sentence)):
            i = self.alignment[j] - 1
            if i < 0:
                i = None  # alignment to NULL token
            zero_indexed_alignment.append((j - 1, i))
        return zero_indexed_alignment

    def __eq__(self, other):
        return self.alignment == other.alignment

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.alignment)


class Counts:
    """
    Data object to store counts of various parameters during training
    """

    def __init__(self):
        self.t_given_s = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.any_t_given_s = defaultdict(lambda: 0.0)
        self.p0 = 0.0
        self.p1 = 0.0
        self.fertility = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.fertility_for_any_phi = defaultdict(lambda: 0.0)

    def update_lexical_translation(self, count, alignment_info, j):
        i = alignment_info.alignment[j]
        t = alignment_info.trg_sentence[j]
        s = alignment_info.src_sentence[i]
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count

    def update_null_generation(self, count, alignment_info):
        m = len(alignment_info.trg_sentence) - 1
        fertility_of_null = alignment_info.fertility_of_i(0)
        self.p1 += fertility_of_null * count
        self.p0 += (m - 2 * fertility_of_null) * count

    def update_fertility(self, count, alignment_info):
        for i in range(0, len(alignment_info.src_sentence)):
            s = alignment_info.src_sentence[i]
            phi = alignment_info.fertility_of_i(i)
            self.fertility[phi][s] += count
            self.fertility_for_any_phi[s] += count
