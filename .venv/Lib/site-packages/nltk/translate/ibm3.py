# Natural Language Toolkit: IBM Model 3
#
# Copyright (C) 2001-2013 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Translation model that considers how a word can be aligned to
multiple words in another language.

IBM Model 3 improves on Model 2 by directly modeling the phenomenon
where a word in one language may be translated into zero or more words
in another. This is expressed by the fertility probability,
n(phi | source word).

If a source word translates into more than one word, it is possible to
generate sentences that have the same alignment in multiple ways. This
is modeled by a distortion step. The distortion probability, d(j|i,l,m),
predicts a target word position, given its aligned source word's
position. The distortion probability replaces the alignment probability
of Model 2.

The fertility probability is not applicable for NULL. Target words that
align to NULL are assumed to be distributed uniformly in the target
sentence. The existence of these words is modeled by p1, the probability
that a target word produced by a real source word requires another
target word that is produced by NULL.

The EM algorithm used in Model 3 is:

:E step: In the training data, collect counts, weighted by prior
         probabilities.

         - (a) count how many times a source language word is translated
               into a target language word
         - (b) count how many times a particular position in the target
               sentence is aligned to a particular position in the source
               sentence
         - (c) count how many times a source word is aligned to phi number
               of target words
         - (d) count how many times NULL is aligned to a target word

:M step: Estimate new probabilities based on the counts from the E step

Because there are too many possible alignments, only the most probable
ones are considered. First, the best alignment is determined using prior
probabilities. Then, a hill climbing approach is used to find other good
candidates.

Notations
---------

:i: Position in the source sentence
     Valid values are 0 (for NULL), 1, 2, ..., length of source sentence
:j: Position in the target sentence
     Valid values are 1, 2, ..., length of target sentence
:l: Number of words in the source sentence, excluding NULL
:m: Number of words in the target sentence
:s: A word in the source language
:t: A word in the target language
:phi: Fertility, the number of target words produced by a source word
:p1: Probability that a target word produced by a source word is
     accompanied by another target word that is aligned to NULL
:p0: 1 - p1

References
----------

Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.

Peter E Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and
Robert L. Mercer. 1993. The Mathematics of Statistical Machine
Translation: Parameter Estimation. Computational Linguistics, 19 (2),
263-311.
"""

import warnings
from collections import defaultdict
from math import factorial

from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel2
from nltk.translate.ibm_model import Counts


class IBMModel3(IBMModel):
    """
    Translation model that considers how a word can be aligned to
    multiple words in another language

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'war', 'ja', 'groß'], ['the', 'house', 'was', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['ein', 'haus', 'ist', 'klein'], ['a', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))
    >>> bitext.append(AlignedSent(['ich', 'fasse', 'das', 'buch', 'zusammen'], ['i', 'summarize', 'the', 'book']))
    >>> bitext.append(AlignedSent(['fasse', 'zusammen'], ['summarize']))

    >>> ibm3 = IBMModel3(bitext, 5)

    >>> print(round(ibm3.translation_table['buch']['book'], 3))
    1.0
    >>> print(round(ibm3.translation_table['das']['book'], 3))
    0.0
    >>> print(round(ibm3.translation_table['ja'][None], 3))
    1.0

    >>> print(round(ibm3.distortion_table[1][1][2][2], 3))
    1.0
    >>> print(round(ibm3.distortion_table[1][2][2][2], 3))
    0.0
    >>> print(round(ibm3.distortion_table[2][2][4][5], 3))
    0.75

    >>> print(round(ibm3.fertility_table[2]['summarize'], 3))
    1.0
    >>> print(round(ibm3.fertility_table[1]['book'], 3))
    1.0

    >>> print(round(ibm3.p1, 3))
    0.054

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, None), (4, 3)])

    """

    def __init__(self, sentence_aligned_corpus, iterations, probability_tables=None):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model, a distortion model, a fertility model, and a
        model for generating NULL-aligned words.

        Translation direction is from ``AlignedSent.mots`` to
        ``AlignedSent.words``.

        :param sentence_aligned_corpus: Sentence-aligned parallel corpus
        :type sentence_aligned_corpus: list(AlignedSent)

        :param iterations: Number of iterations to run training algorithm
        :type iterations: int

        :param probability_tables: Optional. Use this to pass in custom
            probability values. If not specified, probabilities will be
            set to a uniform distribution, or some other sensible value.
            If specified, all the following entries must be present:
            ``translation_table``, ``alignment_table``,
            ``fertility_table``, ``p1``, ``distortion_table``.
            See ``IBMModel`` for the type and purpose of these tables.
        :type probability_tables: dict[str]: object
        """
        super().__init__(sentence_aligned_corpus)
        self.reset_probabilities()

        if probability_tables is None:
            # Get translation and alignment probabilities from IBM Model 2
            ibm2 = IBMModel2(sentence_aligned_corpus, iterations)
            self.translation_table = ibm2.translation_table
            self.alignment_table = ibm2.alignment_table
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables["translation_table"]
            self.alignment_table = probability_tables["alignment_table"]
            self.fertility_table = probability_tables["fertility_table"]
            self.p1 = probability_tables["p1"]
            self.distortion_table = probability_tables["distortion_table"]

        for n in range(0, iterations):
            self.train(sentence_aligned_corpus)

    def reset_probabilities(self):
        super().reset_probabilities()
        self.distortion_table = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))
            )
        )
        """
        dict[int][int][int][int]: float. Probability(j | i,l,m).
        Values accessed as ``distortion_table[j][i][l][m]``.
        """

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        # d(j | i,l,m) = 1 / m for all i, j, l, m
        l_m_combinations = set()
        for aligned_sentence in sentence_aligned_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            if (l, m) not in l_m_combinations:
                l_m_combinations.add((l, m))
                initial_prob = 1 / m
                if initial_prob < IBMModel.MIN_PROB:
                    warnings.warn(
                        "A target sentence is too long ("
                        + str(m)
                        + " words). Results may be less accurate."
                    )
                for j in range(1, m + 1):
                    for i in range(0, l + 1):
                        self.distortion_table[j][i][l][m] = initial_prob

        # simple initialization, taken from GIZA++
        self.fertility_table[0] = defaultdict(lambda: 0.2)
        self.fertility_table[1] = defaultdict(lambda: 0.65)
        self.fertility_table[2] = defaultdict(lambda: 0.1)
        self.fertility_table[3] = defaultdict(lambda: 0.04)
        MAX_FERTILITY = 10
        initial_fert_prob = 0.01 / (MAX_FERTILITY - 4)
        for phi in range(4, MAX_FERTILITY):
            self.fertility_table[phi] = defaultdict(lambda: initial_fert_prob)

        self.p1 = 0.5

    def train(self, parallel_corpus):
        counts = Model3Counts()
        for aligned_sentence in parallel_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)

            # Sample the alignment space
            sampled_alignments, best_alignment = self.sample(aligned_sentence)
            # Record the most probable alignment
            aligned_sentence.alignment = Alignment(
                best_alignment.zero_indexed_alignment()
            )

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_of_alignments(sampled_alignments)

            # E step (b): Collect counts
            for alignment_info in sampled_alignments:
                count = self.prob_t_a_given_s(alignment_info)
                normalized_count = count / total_count

                for j in range(1, m + 1):
                    counts.update_lexical_translation(
                        normalized_count, alignment_info, j
                    )
                    counts.update_distortion(normalized_count, alignment_info, j, l, m)

                counts.update_null_generation(normalized_count, alignment_info)
                counts.update_fertility(normalized_count, alignment_info)

        # M step: Update probabilities with maximum likelihood estimates
        # If any probability is less than MIN_PROB, clamp it to MIN_PROB
        existing_alignment_table = self.alignment_table
        self.reset_probabilities()
        self.alignment_table = existing_alignment_table  # don't retrain

        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_distortion_probabilities(counts)
        self.maximize_fertility_probabilities(counts)
        self.maximize_null_generation_probabilities(counts)

    def maximize_distortion_probabilities(self, counts):
        MIN_PROB = IBMModel.MIN_PROB
        for j, i_s in counts.distortion.items():
            for i, src_sentence_lengths in i_s.items():
                for l, trg_sentence_lengths in src_sentence_lengths.items():
                    for m in trg_sentence_lengths:
                        estimate = (
                            counts.distortion[j][i][l][m]
                            / counts.distortion_for_any_j[i][l][m]
                        )
                        self.distortion_table[j][i][l][m] = max(estimate, MIN_PROB)

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        src_sentence = alignment_info.src_sentence
        trg_sentence = alignment_info.trg_sentence
        l = len(src_sentence) - 1  # exclude NULL
        m = len(trg_sentence) - 1
        p1 = self.p1
        p0 = 1 - p1

        probability = 1.0
        MIN_PROB = IBMModel.MIN_PROB

        # Combine NULL insertion probability
        null_fertility = alignment_info.fertility_of_i(0)
        probability *= pow(p1, null_fertility) * pow(p0, m - 2 * null_fertility)
        if probability < MIN_PROB:
            return MIN_PROB

        # Compute combination (m - null_fertility) choose null_fertility
        for i in range(1, null_fertility + 1):
            probability *= (m - null_fertility - i + 1) / i
            if probability < MIN_PROB:
                return MIN_PROB

        # Combine fertility probabilities
        for i in range(1, l + 1):
            fertility = alignment_info.fertility_of_i(i)
            probability *= (
                factorial(fertility) * self.fertility_table[fertility][src_sentence[i]]
            )
            if probability < MIN_PROB:
                return MIN_PROB

        # Combine lexical and distortion probabilities
        for j in range(1, m + 1):
            t = trg_sentence[j]
            i = alignment_info.alignment[j]
            s = src_sentence[i]

            probability *= (
                self.translation_table[t][s] * self.distortion_table[j][i][l][m]
            )
            if probability < MIN_PROB:
                return MIN_PROB

        return probability


class Model3Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for distortion.
    """

    def __init__(self):
        super().__init__()
        self.distortion = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        )
        self.distortion_for_any_j = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
        )

    def update_distortion(self, count, alignment_info, j, l, m):
        i = alignment_info.alignment[j]
        self.distortion[j][i][l][m] += count
        self.distortion_for_any_j[i][l][m] += count
