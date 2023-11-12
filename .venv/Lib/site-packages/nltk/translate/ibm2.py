# Natural Language Toolkit: IBM Model 2
#
# Copyright (C) 2001-2013 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Lexical translation model that considers word order.

IBM Model 2 improves on Model 1 by accounting for word order.
An alignment probability is introduced, a(i | j,l,m), which predicts
a source word position, given its aligned target word's position.

The EM algorithm used in Model 2 is:

:E step: In the training data, collect counts, weighted by prior
         probabilities.

         - (a) count how many times a source language word is translated
               into a target language word
         - (b) count how many times a particular position in the source
               sentence is aligned to a particular position in the target
               sentence

:M step: Estimate new probabilities based on the counts from the E step

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

from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts


class IBMModel2(IBMModel):
    """
    Lexical translation model that considers word order

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))

    >>> ibm2 = IBMModel2(bitext, 5)

    >>> print(round(ibm2.translation_table['buch']['book'], 3))
    1.0
    >>> print(round(ibm2.translation_table['das']['book'], 3))
    0.0
    >>> print(round(ibm2.translation_table['buch'][None], 3))
    0.0
    >>> print(round(ibm2.translation_table['ja'][None], 3))
    0.0

    >>> print(round(ibm2.alignment_table[1][1][2][2], 3))
    0.939
    >>> print(round(ibm2.alignment_table[1][2][2][2], 3))
    0.0
    >>> print(round(ibm2.alignment_table[2][2][4][5], 3))
    1.0

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, 2), (4, 3)])

    """

    def __init__(self, sentence_aligned_corpus, iterations, probability_tables=None):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model and an alignment model.

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
            ``translation_table``, ``alignment_table``.
            See ``IBMModel`` for the type and purpose of these tables.
        :type probability_tables: dict[str]: object
        """
        super().__init__(sentence_aligned_corpus)

        if probability_tables is None:
            # Get translation probabilities from IBM Model 1
            # Run more iterations of training for Model 1, since it is
            # faster than Model 2
            ibm1 = IBMModel1(sentence_aligned_corpus, 2 * iterations)
            self.translation_table = ibm1.translation_table
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables["translation_table"]
            self.alignment_table = probability_tables["alignment_table"]

        for n in range(0, iterations):
            self.train(sentence_aligned_corpus)

        self.align_all(sentence_aligned_corpus)

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        # a(i | j,l,m) = 1 / (l+1) for all i, j, l, m
        l_m_combinations = set()
        for aligned_sentence in sentence_aligned_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            if (l, m) not in l_m_combinations:
                l_m_combinations.add((l, m))
                initial_prob = 1 / (l + 1)
                if initial_prob < IBMModel.MIN_PROB:
                    warnings.warn(
                        "A source sentence is too long ("
                        + str(l)
                        + " words). Results may be less accurate."
                    )

                for i in range(0, l + 1):
                    for j in range(1, m + 1):
                        self.alignment_table[i][j][l][m] = initial_prob

    def train(self, parallel_corpus):
        counts = Model2Counts()
        for aligned_sentence in parallel_corpus:
            src_sentence = [None] + aligned_sentence.mots
            trg_sentence = ["UNUSED"] + aligned_sentence.words  # 1-indexed
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_all_alignments(src_sentence, trg_sentence)

            # E step (b): Collect counts
            for j in range(1, m + 1):
                t = trg_sentence[j]
                for i in range(0, l + 1):
                    s = src_sentence[i]
                    count = self.prob_alignment_point(i, j, src_sentence, trg_sentence)
                    normalized_count = count / total_count[t]

                    counts.update_lexical_translation(normalized_count, s, t)
                    counts.update_alignment(normalized_count, i, j, l, m)

        # M step: Update probabilities with maximum likelihood estimates
        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_alignment_probabilities(counts)

    def maximize_alignment_probabilities(self, counts):
        MIN_PROB = IBMModel.MIN_PROB
        for i, j_s in counts.alignment.items():
            for j, src_sentence_lengths in j_s.items():
                for l, trg_sentence_lengths in src_sentence_lengths.items():
                    for m in trg_sentence_lengths:
                        estimate = (
                            counts.alignment[i][j][l][m]
                            / counts.alignment_for_any_i[j][l][m]
                        )
                        self.alignment_table[i][j][l][m] = max(estimate, MIN_PROB)

    def prob_all_alignments(self, src_sentence, trg_sentence):
        """
        Computes the probability of all possible word alignments,
        expressed as a marginal distribution over target words t

        Each entry in the return value represents the contribution to
        the total alignment probability by the target word t.

        To obtain probability(alignment | src_sentence, trg_sentence),
        simply sum the entries in the return value.

        :return: Probability of t for all s in ``src_sentence``
        :rtype: dict(str): float
        """
        alignment_prob_for_t = defaultdict(lambda: 0.0)
        for j in range(1, len(trg_sentence)):
            t = trg_sentence[j]
            for i in range(0, len(src_sentence)):
                alignment_prob_for_t[t] += self.prob_alignment_point(
                    i, j, src_sentence, trg_sentence
                )
        return alignment_prob_for_t

    def prob_alignment_point(self, i, j, src_sentence, trg_sentence):
        """
        Probability that position j in ``trg_sentence`` is aligned to
        position i in the ``src_sentence``
        """
        l = len(src_sentence) - 1
        m = len(trg_sentence) - 1
        s = src_sentence[i]
        t = trg_sentence[j]
        return self.translation_table[t][s] * self.alignment_table[i][j][l][m]

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        prob = 1.0
        l = len(alignment_info.src_sentence) - 1
        m = len(alignment_info.trg_sentence) - 1

        for j, i in enumerate(alignment_info.alignment):
            if j == 0:
                continue  # skip the dummy zeroeth element
            trg_word = alignment_info.trg_sentence[j]
            src_word = alignment_info.src_sentence[i]
            prob *= (
                self.translation_table[trg_word][src_word]
                * self.alignment_table[i][j][l][m]
            )

        return max(prob, IBMModel.MIN_PROB)

    def align_all(self, parallel_corpus):
        for sentence_pair in parallel_corpus:
            self.align(sentence_pair)

    def align(self, sentence_pair):
        """
        Determines the best word alignment for one sentence pair from
        the corpus that the model was trained on.

        The best alignment will be set in ``sentence_pair`` when the
        method returns. In contrast with the internal implementation of
        IBM models, the word indices in the ``Alignment`` are zero-
        indexed, not one-indexed.

        :param sentence_pair: A sentence in the source language and its
            counterpart sentence in the target language
        :type sentence_pair: AlignedSent
        """
        best_alignment = []

        l = len(sentence_pair.mots)
        m = len(sentence_pair.words)

        for j, trg_word in enumerate(sentence_pair.words):
            # Initialize trg_word to align with the NULL token
            best_prob = (
                self.translation_table[trg_word][None]
                * self.alignment_table[0][j + 1][l][m]
            )
            best_prob = max(best_prob, IBMModel.MIN_PROB)
            best_alignment_point = None
            for i, src_word in enumerate(sentence_pair.mots):
                align_prob = (
                    self.translation_table[trg_word][src_word]
                    * self.alignment_table[i + 1][j + 1][l][m]
                )
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_alignment_point = i

            best_alignment.append((j, best_alignment_point))

        sentence_pair.alignment = Alignment(best_alignment)


class Model2Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for alignment.
    """

    def __init__(self):
        super().__init__()
        self.alignment = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        )
        self.alignment_for_any_i = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0))
        )

    def update_lexical_translation(self, count, s, t):
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count

    def update_alignment(self, count, i, j, l, m):
        self.alignment[i][j][l][m] += count
        self.alignment_for_any_i[j][l][m] += count
