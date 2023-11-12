# Natural Language Toolkit: Collocations and Association Measures
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Joel Nothman <jnothman@student.usyd.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#
"""
Tools to identify collocations --- words that often appear consecutively
--- within corpora. They may also be used to find other associations between
word occurrences.
See Manning and Schutze ch. 5 at https://nlp.stanford.edu/fsnlp/promo/colloc.pdf
and the Text::NSP Perl package at http://ngram.sourceforge.net

Finding collocations requires first calculating the frequencies of words and
their appearance in the context of other words. Often the collection of words
will then requiring filtering to only retain useful content terms. Each ngram
of words may then be scored according to some association measure, in order
to determine the relative likelihood of each ngram being a collocation.

The ``BigramCollocationFinder`` and ``TrigramCollocationFinder`` classes provide
these functionalities, dependent on being provided a function which scores a
ngram given appropriate frequency counts. A number of standard association
measures are provided in bigram_measures and trigram_measures.
"""

# Possible TODOs:
# - consider the distinction between f(x,_) and f(x) and whether our
#   approximation is good enough for fragmented data, and mention it
# - add a n-gram collocation finder with measures which only utilise n-gram
#   and unigram counts (raw_freq, pmi, student_t)

import itertools as _itertools

# these two unused imports are referenced in collocations.doctest
from nltk.metrics import (
    BigramAssocMeasures,
    ContingencyMeasures,
    QuadgramAssocMeasures,
    TrigramAssocMeasures,
)
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams


class AbstractCollocationFinder:
    """
    An abstract base class for collocation finders whose purpose is to
    collect collocation candidate frequencies, filter and rank them.

    As a minimum, collocation finders require the frequencies of each
    word in a corpus, and the joint frequency of word tuples. This data
    should be provided through nltk.probability.FreqDist objects or an
    identical interface.
    """

    def __init__(self, word_fd, ngram_fd):
        self.word_fd = word_fd
        self.N = word_fd.N()
        self.ngram_fd = ngram_fd

    @classmethod
    def _build_new_documents(
        cls, documents, window_size, pad_left=False, pad_right=False, pad_symbol=None
    ):
        """
        Pad the document with the place holder according to the window_size
        """
        padding = (pad_symbol,) * (window_size - 1)
        if pad_right:
            return _itertools.chain.from_iterable(
                _itertools.chain(doc, padding) for doc in documents
            )
        if pad_left:
            return _itertools.chain.from_iterable(
                _itertools.chain(padding, doc) for doc in documents
            )

    @classmethod
    def from_documents(cls, documents):
        """Constructs a collocation finder given a collection of documents,
        each of which is a list (or iterable) of tokens.
        """
        # return cls.from_words(_itertools.chain(*documents))
        return cls.from_words(
            cls._build_new_documents(documents, cls.default_ws, pad_right=True)
        )

    @staticmethod
    def _ngram_freqdist(words, n):
        return FreqDist(tuple(words[i : i + n]) for i in range(len(words) - 1))

    def _apply_filter(self, fn=lambda ngram, freq: False):
        """Generic filter removes ngrams from the frequency distribution
        if the function returns True when passed an ngram tuple.
        """
        tmp_ngram = FreqDist()
        for ngram, freq in self.ngram_fd.items():
            if not fn(ngram, freq):
                tmp_ngram[ngram] = freq
        self.ngram_fd = tmp_ngram

    def apply_freq_filter(self, min_freq):
        """Removes candidate ngrams which have frequency less than min_freq."""
        self._apply_filter(lambda ng, freq: freq < min_freq)

    def apply_ngram_filter(self, fn):
        """Removes candidate ngrams (w1, w2, ...) where fn(w1, w2, ...)
        evaluates to True.
        """
        self._apply_filter(lambda ng, f: fn(*ng))

    def apply_word_filter(self, fn):
        """Removes candidate ngrams (w1, w2, ...) where any of (fn(w1), fn(w2),
        ...) evaluates to True.
        """
        self._apply_filter(lambda ng, f: any(fn(w) for w in ng))

    def _score_ngrams(self, score_fn):
        """Generates of (ngram, score) pairs as determined by the scoring
        function provided.
        """
        for tup in self.ngram_fd:
            score = self.score_ngram(score_fn, *tup)
            if score is not None:
                yield tup, score

    def score_ngrams(self, score_fn):
        """Returns a sequence of (ngram, score) pairs ordered from highest to
        lowest score, as determined by the scoring function provided.
        """
        return sorted(self._score_ngrams(score_fn), key=lambda t: (-t[1], t[0]))

    def nbest(self, score_fn, n):
        """Returns the top n ngrams when scored by the given function."""
        return [p for p, s in self.score_ngrams(score_fn)[:n]]

    def above_score(self, score_fn, min_score):
        """Returns a sequence of ngrams, ordered by decreasing score, whose
        scores each exceed the given minimum score.
        """
        for ngram, score in self.score_ngrams(score_fn):
            if score > min_score:
                yield ngram
            else:
                break


class BigramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of bigram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """

    default_ws = 2

    def __init__(self, word_fd, bigram_fd, window_size=2):
        """Construct a BigramCollocationFinder, given FreqDists for
        appearances of words and (possibly non-contiguous) bigrams.
        """
        AbstractCollocationFinder.__init__(self, word_fd, bigram_fd)
        self.window_size = window_size

    @classmethod
    def from_words(cls, words, window_size=2):
        """Construct a BigramCollocationFinder for all bigrams in the given
        sequence.  When window_size > 2, count non-contiguous bigrams, in the
        style of Church and Hanks's (1990) association ratio.
        """
        wfd = FreqDist()
        bfd = FreqDist()

        if window_size < 2:
            raise ValueError("Specify window_size at least 2")

        for window in ngrams(words, window_size, pad_right=True):
            w1 = window[0]
            if w1 is None:
                continue
            wfd[w1] += 1
            for w2 in window[1:]:
                if w2 is not None:
                    bfd[(w1, w2)] += 1
        return cls(wfd, bfd, window_size=window_size)

    def score_ngram(self, score_fn, w1, w2):
        """Returns the score for a given bigram using the given scoring
        function.  Following Church and Hanks (1990), counts are scaled by
        a factor of 1/(window_size - 1).
        """
        n_all = self.N
        n_ii = self.ngram_fd[(w1, w2)] / (self.window_size - 1.0)
        if not n_ii:
            return
        n_ix = self.word_fd[w1]
        n_xi = self.word_fd[w2]
        return score_fn(n_ii, (n_ix, n_xi), n_all)


class TrigramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of trigram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """

    default_ws = 3

    def __init__(self, word_fd, bigram_fd, wildcard_fd, trigram_fd):
        """Construct a TrigramCollocationFinder, given FreqDists for
        appearances of words, bigrams, two words with any word between them,
        and trigrams.
        """
        AbstractCollocationFinder.__init__(self, word_fd, trigram_fd)
        self.wildcard_fd = wildcard_fd
        self.bigram_fd = bigram_fd

    @classmethod
    def from_words(cls, words, window_size=3):
        """Construct a TrigramCollocationFinder for all trigrams in the given
        sequence.
        """
        if window_size < 3:
            raise ValueError("Specify window_size at least 3")

        wfd = FreqDist()
        wildfd = FreqDist()
        bfd = FreqDist()
        tfd = FreqDist()
        for window in ngrams(words, window_size, pad_right=True):
            w1 = window[0]
            if w1 is None:
                continue
            for w2, w3 in _itertools.combinations(window[1:], 2):
                wfd[w1] += 1
                if w2 is None:
                    continue
                bfd[(w1, w2)] += 1
                if w3 is None:
                    continue
                wildfd[(w1, w3)] += 1
                tfd[(w1, w2, w3)] += 1
        return cls(wfd, bfd, wildfd, tfd)

    def bigram_finder(self):
        """Constructs a bigram collocation finder with the bigram and unigram
        data from this finder. Note that this does not include any filtering
        applied to this finder.
        """
        return BigramCollocationFinder(self.word_fd, self.bigram_fd)

    def score_ngram(self, score_fn, w1, w2, w3):
        """Returns the score for a given trigram using the given scoring
        function.
        """
        n_all = self.N
        n_iii = self.ngram_fd[(w1, w2, w3)]
        if not n_iii:
            return
        n_iix = self.bigram_fd[(w1, w2)]
        n_ixi = self.wildcard_fd[(w1, w3)]
        n_xii = self.bigram_fd[(w2, w3)]
        n_ixx = self.word_fd[w1]
        n_xix = self.word_fd[w2]
        n_xxi = self.word_fd[w3]
        return score_fn(n_iii, (n_iix, n_ixi, n_xii), (n_ixx, n_xix, n_xxi), n_all)


class QuadgramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of quadgram collocations or other association measures.
    It is often useful to use from_words() rather than constructing an instance directly.
    """

    default_ws = 4

    def __init__(self, word_fd, quadgram_fd, ii, iii, ixi, ixxi, iixi, ixii):
        """Construct a QuadgramCollocationFinder, given FreqDists for appearances of words,
        bigrams, trigrams, two words with one word and two words between them, three words
        with a word between them in both variations.
        """
        AbstractCollocationFinder.__init__(self, word_fd, quadgram_fd)
        self.iii = iii
        self.ii = ii
        self.ixi = ixi
        self.ixxi = ixxi
        self.iixi = iixi
        self.ixii = ixii

    @classmethod
    def from_words(cls, words, window_size=4):
        if window_size < 4:
            raise ValueError("Specify window_size at least 4")
        ixxx = FreqDist()
        iiii = FreqDist()
        ii = FreqDist()
        iii = FreqDist()
        ixi = FreqDist()
        ixxi = FreqDist()
        iixi = FreqDist()
        ixii = FreqDist()

        for window in ngrams(words, window_size, pad_right=True):
            w1 = window[0]
            if w1 is None:
                continue
            for w2, w3, w4 in _itertools.combinations(window[1:], 3):
                ixxx[w1] += 1
                if w2 is None:
                    continue
                ii[(w1, w2)] += 1
                if w3 is None:
                    continue
                iii[(w1, w2, w3)] += 1
                ixi[(w1, w3)] += 1
                if w4 is None:
                    continue
                iiii[(w1, w2, w3, w4)] += 1
                ixxi[(w1, w4)] += 1
                ixii[(w1, w3, w4)] += 1
                iixi[(w1, w2, w4)] += 1

        return cls(ixxx, iiii, ii, iii, ixi, ixxi, iixi, ixii)

    def score_ngram(self, score_fn, w1, w2, w3, w4):
        n_all = self.N
        n_iiii = self.ngram_fd[(w1, w2, w3, w4)]
        if not n_iiii:
            return
        n_iiix = self.iii[(w1, w2, w3)]
        n_xiii = self.iii[(w2, w3, w4)]
        n_iixi = self.iixi[(w1, w2, w4)]
        n_ixii = self.ixii[(w1, w3, w4)]

        n_iixx = self.ii[(w1, w2)]
        n_xxii = self.ii[(w3, w4)]
        n_xiix = self.ii[(w2, w3)]
        n_ixix = self.ixi[(w1, w3)]
        n_ixxi = self.ixxi[(w1, w4)]
        n_xixi = self.ixi[(w2, w4)]

        n_ixxx = self.word_fd[w1]
        n_xixx = self.word_fd[w2]
        n_xxix = self.word_fd[w3]
        n_xxxi = self.word_fd[w4]
        return score_fn(
            n_iiii,
            (n_iiix, n_iixi, n_ixii, n_xiii),
            (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix),
            (n_ixxx, n_xixx, n_xxix, n_xxxi),
            n_all,
        )


def demo(scorer=None, compare_scorer=None):
    """Finds bigram collocations in the files of the WebText corpus."""
    from nltk.metrics import (
        BigramAssocMeasures,
        ranks_from_scores,
        spearman_correlation,
    )

    if scorer is None:
        scorer = BigramAssocMeasures.likelihood_ratio
    if compare_scorer is None:
        compare_scorer = BigramAssocMeasures.raw_freq

    from nltk.corpus import stopwords, webtext

    ignored_words = stopwords.words("english")
    word_filter = lambda w: len(w) < 3 or w.lower() in ignored_words

    for file in webtext.fileids():
        words = [word.lower() for word in webtext.words(file)]

        cf = BigramCollocationFinder.from_words(words)
        cf.apply_freq_filter(3)
        cf.apply_word_filter(word_filter)

        corr = spearman_correlation(
            ranks_from_scores(cf.score_ngrams(scorer)),
            ranks_from_scores(cf.score_ngrams(compare_scorer)),
        )
        print(file)
        print("\t", [" ".join(tup) for tup in cf.nbest(scorer, 15)])
        print(f"\t Correlation to {compare_scorer.__name__}: {corr:0.4f}")


# Slows down loading too much
# bigram_measures = BigramAssocMeasures()
# trigram_measures = TrigramAssocMeasures()

if __name__ == "__main__":
    import sys

    from nltk.metrics import BigramAssocMeasures

    try:
        scorer = eval("BigramAssocMeasures." + sys.argv[1])
    except IndexError:
        scorer = None
    try:
        compare_scorer = eval("BigramAssocMeasures." + sys.argv[2])
    except IndexError:
        compare_scorer = None

    demo(scorer, compare_scorer)

__all__ = [
    "BigramCollocationFinder",
    "TrigramCollocationFinder",
    "QuadgramCollocationFinder",
]
