# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
#         Manu Joseph <manujosephv@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Smoothing algorithms for language modeling.

According to Chen & Goodman 1995 these should work with both Backoff and
Interpolation.
"""
from operator import methodcaller

from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist


def _count_values_gt_zero(distribution):
    """Count values that are greater than zero in a distribution.

    Assumes distribution is either a mapping with counts as values or
    an instance of `nltk.ConditionalFreqDist`.
    """
    as_count = (
        methodcaller("N")
        if isinstance(distribution, ConditionalFreqDist)
        else lambda count: count
    )
    # We explicitly check that values are > 0 to guard against negative counts.
    return sum(
        1 for dist_or_count in distribution.values() if as_count(dist_or_count) > 0
    )


class WittenBell(Smoothing):
    """Witten-Bell smoothing."""

    def __init__(self, vocabulary, counter, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)

    def alpha_gamma(self, word, context):
        alpha = self.counts[context].freq(word)
        gamma = self._gamma(context)
        return (1.0 - gamma) * alpha, gamma

    def _gamma(self, context):
        n_plus = _count_values_gt_zero(self.counts[context])
        return n_plus / (n_plus + self.counts[context].N())

    def unigram_score(self, word):
        return self.counts.unigrams.freq(word)


class AbsoluteDiscounting(Smoothing):
    """Smoothing with absolute discount."""

    def __init__(self, vocabulary, counter, discount=0.75, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.discount = discount

    def alpha_gamma(self, word, context):
        alpha = (
            max(self.counts[context][word] - self.discount, 0)
            / self.counts[context].N()
        )
        gamma = self._gamma(context)
        return alpha, gamma

    def _gamma(self, context):
        n_plus = _count_values_gt_zero(self.counts[context])
        return (self.discount * n_plus) / self.counts[context].N()

    def unigram_score(self, word):
        return self.counts.unigrams.freq(word)


class KneserNey(Smoothing):
    """Kneser-Ney Smoothing.

    This is an extension of smoothing with a discount.

    Resources:
    - https://pages.ucsd.edu/~rlevy/lign256/winter2008/kneser_ney_mini_example.pdf
    - https://www.youtube.com/watch?v=ody1ysUTD7o
    - https://medium.com/@dennyc/a-simple-numerical-example-for-kneser-ney-smoothing-nlp-4600addf38b8
    - https://www.cl.uni-heidelberg.de/courses/ss15/smt/scribe6.pdf
    - https://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf
    """

    def __init__(self, vocabulary, counter, order, discount=0.1, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.discount = discount
        self._order = order

    def unigram_score(self, word):
        word_continuation_count, total_count = self._continuation_counts(word)
        return word_continuation_count / total_count

    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        word_continuation_count, total_count = (
            (prefix_counts[word], prefix_counts.N())
            if len(context) + 1 == self._order
            else self._continuation_counts(word, context)
        )
        alpha = max(word_continuation_count - self.discount, 0.0) / total_count
        gamma = self.discount * _count_values_gt_zero(prefix_counts) / total_count
        return alpha, gamma

    def _continuation_counts(self, word, context=tuple()):
        """Count continuations that end with context and word.

        Continuations track unique ngram "types", regardless of how many
        instances were observed for each "type".
        This is different than raw ngram counts which track number of instances.
        """
        higher_order_ngrams_with_context = (
            counts
            for prefix_ngram, counts in self.counts[len(context) + 2].items()
            if prefix_ngram[1:] == context
        )
        higher_order_ngrams_with_word_count, total = 0, 0
        for counts in higher_order_ngrams_with_context:
            higher_order_ngrams_with_word_count += int(counts[word] > 0)
            total += _count_values_gt_zero(counts)
        return higher_order_ngrams_with_word_count, total
