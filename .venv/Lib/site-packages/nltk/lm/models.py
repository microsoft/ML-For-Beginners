# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
#         Manu Joseph <manujosephv@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Language Models"""

from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell


class MLE(LanguageModel):
    """Class for providing MLE ngram model scores.

    Inherits initialization from BaseNgramModel.
    """

    def unmasked_score(self, word, context=None):
        """Returns the MLE score for a word given a context.

        Args:
        - word is expected to be a string
        - context is expected to be something reasonably convertible to a tuple
        """
        return self.context_counts(context).freq(word)


class Lidstone(LanguageModel):
    """Provides Lidstone-smoothed scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def unmasked_score(self, word, context=None):
        """Add-one smoothing: Lidstone or Laplace.

        To see what kind, look at `gamma` attribute on the class.

        """
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        return (word_count + self.gamma) / (norm_count + len(self.vocab) * self.gamma)


class Laplace(Lidstone):
    """Implements Laplace (add one) smoothing.

    Initialization identical to BaseNgramModel because gamma is always 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class StupidBackoff(LanguageModel):
    """Provides StupidBackoff scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a parameter alpha with which we scale the lower order probabilities.
    Note that this is not a true probability distribution as scores for ngrams
    of the same order do not sum up to unity.
    """

    def __init__(self, alpha=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def unmasked_score(self, word, context=None):
        if not context:
            # Base recursion
            return self.counts.unigrams.freq(word)
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        if word_count > 0:
            return word_count / norm_count
        else:
            return self.alpha * self.unmasked_score(word, context[1:])


class InterpolatedLanguageModel(LanguageModel):
    """Logic common to all interpolated language models.

    The idea to abstract this comes from Chen & Goodman 1995.
    Do not instantiate this class directly!
    """

    def __init__(self, smoothing_cls, order, **kwargs):
        params = kwargs.pop("params", {})
        super().__init__(order, **kwargs)
        self.estimator = smoothing_cls(self.vocab, self.counts, **params)

    def unmasked_score(self, word, context=None):
        if not context:
            # The base recursion case: no context, we only have a unigram.
            return self.estimator.unigram_score(word)
        if not self.counts[context]:
            # It can also happen that we have no data for this context.
            # In that case we defer to the lower-order ngram.
            # This is the same as setting alpha to 0 and gamma to 1.
            alpha, gamma = 0, 1
        else:
            alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])


class WittenBellInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""

    def __init__(self, order, **kwargs):
        super().__init__(WittenBell, order, **kwargs)


class AbsoluteDiscountingInterpolated(InterpolatedLanguageModel):
    """Interpolated version of smoothing with absolute discount."""

    def __init__(self, order, discount=0.75, **kwargs):
        super().__init__(
            AbsoluteDiscounting, order, params={"discount": discount}, **kwargs
        )


class KneserNeyInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Kneser-Ney smoothing."""

    def __init__(self, order, discount=0.1, **kwargs):
        if not (0 <= discount <= 1):
            raise ValueError(
                "Discount must be between 0 and 1 for probabilities to sum to unity."
            )
        super().__init__(
            KneserNey, order, params={"discount": discount, "order": order}, **kwargs
        )
