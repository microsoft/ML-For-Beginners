# Natural Language Toolkit
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
Language Model Counter
----------------------
"""

from collections import defaultdict
from collections.abc import Sequence

from nltk.probability import ConditionalFreqDist, FreqDist


class NgramCounter:
    """Class for counting ngrams.

    Will count any ngram sequence you give it ;)

    First we need to make sure we are feeding the counter sentences of ngrams.

    >>> text = [["a", "b", "c", "d"], ["a", "c", "d", "c"]]
    >>> from nltk.util import ngrams
    >>> text_bigrams = [ngrams(sent, 2) for sent in text]
    >>> text_unigrams = [ngrams(sent, 1) for sent in text]

    The counting itself is very simple.

    >>> from nltk.lm import NgramCounter
    >>> ngram_counts = NgramCounter(text_bigrams + text_unigrams)

    You can conveniently access ngram counts using standard python dictionary notation.
    String keys will give you unigram counts.

    >>> ngram_counts['a']
    2
    >>> ngram_counts['aliens']
    0

    If you want to access counts for higher order ngrams, use a list or a tuple.
    These are treated as "context" keys, so what you get is a frequency distribution
    over all continuations after the given context.

    >>> sorted(ngram_counts[['a']].items())
    [('b', 1), ('c', 1)]
    >>> sorted(ngram_counts[('a',)].items())
    [('b', 1), ('c', 1)]

    This is equivalent to specifying explicitly the order of the ngram (in this case
    2 for bigram) and indexing on the context.

    >>> ngram_counts[2][('a',)] is ngram_counts[['a']]
    True

    Note that the keys in `ConditionalFreqDist` cannot be lists, only tuples!
    It is generally advisable to use the less verbose and more flexible square
    bracket notation.

    To get the count of the full ngram "a b", do this:

    >>> ngram_counts[['a']]['b']
    1

    Specifying the ngram order as a number can be useful for accessing all ngrams
    in that order.

    >>> ngram_counts[2]
    <ConditionalFreqDist with 4 conditions>

    The keys of this `ConditionalFreqDist` are the contexts we discussed earlier.
    Unigrams can also be accessed with a human-friendly alias.

    >>> ngram_counts.unigrams is ngram_counts[1]
    True

    Similarly to `collections.Counter`, you can update counts after initialization.

    >>> ngram_counts['e']
    0
    >>> ngram_counts.update([ngrams(["d", "e", "f"], 1)])
    >>> ngram_counts['e']
    1

    """

    def __init__(self, ngram_text=None):
        """Creates a new NgramCounter.

        If `ngram_text` is specified, counts ngrams from it, otherwise waits for
        `update` method to be called explicitly.

        :param ngram_text: Optional text containing sentences of ngrams, as for `update` method.
        :type ngram_text: Iterable(Iterable(tuple(str))) or None

        """
        self._counts = defaultdict(ConditionalFreqDist)
        self._counts[1] = self.unigrams = FreqDist()

        if ngram_text:
            self.update(ngram_text)

    def update(self, ngram_text):
        """Updates ngram counts from `ngram_text`.

        Expects `ngram_text` to be a sequence of sentences (sequences).
        Each sentence consists of ngrams as tuples of strings.

        :param Iterable(Iterable(tuple(str))) ngram_text: Text containing sentences of ngrams.
        :raises TypeError: if the ngrams are not tuples.

        """

        for sent in ngram_text:
            for ngram in sent:
                if not isinstance(ngram, tuple):
                    raise TypeError(
                        "Ngram <{}> isn't a tuple, " "but {}".format(ngram, type(ngram))
                    )

                ngram_order = len(ngram)
                if ngram_order == 1:
                    self.unigrams[ngram[0]] += 1
                    continue

                context, word = ngram[:-1], ngram[-1]
                self[ngram_order][context][word] += 1

    def N(self):
        """Returns grand total number of ngrams stored.

        This includes ngrams from all orders, so some duplication is expected.
        :rtype: int

        >>> from nltk.lm import NgramCounter
        >>> counts = NgramCounter([[("a", "b"), ("c",), ("d", "e")]])
        >>> counts.N()
        3

        """
        return sum(val.N() for val in self._counts.values())

    def __getitem__(self, item):
        """User-friendly access to ngram counts."""
        if isinstance(item, int):
            return self._counts[item]
        elif isinstance(item, str):
            return self._counts.__getitem__(1)[item]
        elif isinstance(item, Sequence):
            return self._counts.__getitem__(len(item) + 1)[tuple(item)]

    def __str__(self):
        return "<{} with {} ngram orders and {} ngrams>".format(
            self.__class__.__name__, len(self._counts), self.N()
        )

    def __len__(self):
        return self._counts.__len__()

    def __contains__(self, item):
        return item in self._counts
