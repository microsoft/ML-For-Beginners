"""
@file
@brief Overloads :epkg:`TfidfVectorizer` and :epkg:`CountVectorizer`.
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    from sklearn.feature_extraction.text import _VectorizerMixin as VectorizerMixin
except ImportError:  # pragma: no cover
    # scikit-learn < 0.23
    from sklearn.feature_extraction.text import VectorizerMixin


class NGramsMixin(VectorizerMixin):
    """
    Overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to get tuples instead of string in member `vocabulary_
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
    of `TfidfVectorizer` or :epkg:`CountVectorizer`.
    It contains the list of n-grams used to process documents.
    See :class:`TraceableCountVectorizer` and :class:`TraceableTfidfVectorizer`
    for example.
    """

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        if tokens is not None:
            new_tokens = []
            for token in tokens:
                val = (token,) if isinstance(token, str) else token
                if not isinstance(val, tuple):
                    raise TypeError(f"Unexpected type {type(val)}:{val!r} for a token.")
                if any(map(lambda x: not isinstance(x, str), val)):
                    raise TypeError(
                        f"Unexpected type {val!r}, one part of a "
                        f"token is not a string."
                    )
                new_tokens.append(val)
            tokens = new_tokens

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append

            def space_join(tokens):
                new_tokens = []
                for token in tokens:
                    if isinstance(token, str):
                        new_tokens.append(token)
                    elif isinstance(token, tuple):
                        new_tokens.extend(token)
                    else:
                        raise TypeError(  # pragma: no cover
                            f"Unable to build a n-grams out of {tokens}."
                        )
                return tuple(new_tokens)

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i : i + n]))
        return tokens

    @staticmethod
    def _fix_vocabulary(expected, new_voc):
        update = {}
        for w, wid in new_voc.items():
            if not isinstance(w, tuple):
                raise TypeError(f"Tuple is expected for a token not {type(w)}.")
            s = " ".join(w)
            if s in expected:
                if expected[s] != wid:
                    update[w] = wid
        if update:
            new_voc.update(update)
        duplicates = {}
        for w, wid in new_voc.items():
            if wid not in duplicates:
                duplicates[wid] = {w}
            else:
                duplicates[wid].add(w)
        dups = {k: v for k, v in duplicates.items() if len(v) > 1}
        return update, dups


class TraceableCountVectorizer(CountVectorizer, NGramsMixin):
    """
    Inherits from :class:`NGramsMixin` which overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to keep more information about n-grams but still produces the same
    outputs than `CountVectorizer`.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.feature_extraction.text import CountVectorizer
        from mlinsights.mlmodel.sklearn_text import TraceableCountVectorizer
        from pprint import pformat

        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "Is this the first document?",
            "",
        ]).reshape((4, ))

        print('CountVectorizer from scikit-learn')
        mod1 = CountVectorizer(ngram_range=(1, 2))
        mod1.fit(corpus)
        print(mod1.transform(corpus).todense()[:2])
        print(pformat(mod1.vocabulary_)[:100])

        print('TraceableCountVectorizer from scikit-learn')
        mod2 = TraceableCountVectorizer(ngram_range=(1, 2))
        mod2.fit(corpus)
        print(mod2.transform(corpus).todense()[:2])
        print(pformat(mod2.vocabulary_)[:100])

    A weirder example with
    @see cl TraceableTfidfVectorizer shows more differences.

    The class is training an instance of CountVectorizer on the
    same data. This is used to update the vocabulary to match
    the same columns as the one obtained with scikit-learn.
    scikit-learn cannot distinguish between bi gram ("a b", "c") and
    ("a", "b c"). Therefore, there are merged into the same
    column by scikit-learn. This class, even if it is able to distinguish
    between them, keeps the same ambiguity.
    """

    def _word_ngrams(self, tokens, stop_words=None):
        return NGramsMixin._word_ngrams(self, tokens=tokens, stop_words=stop_words)

    def fit(self, X, y=None):
        # scikit-learn implements fit_transform and fit calls it.
        new_self = CountVectorizer(**self.get_params())
        new_self._word_ngrams = self._word_ngrams
        new_self.fit(X, y=y)
        for k, v in new_self.__dict__.items():
            if k.endswith("_") and not k.endswith("__"):
                setattr(self, k, v)
        same = CountVectorizer(**self.get_params())
        same.fit(X, y=y)
        self.same_ = same
        if self.stop_words != same.stop_words:
            raise AssertionError(
                f"Different stop_words {self.stop_words} != {same.stop_words}."
            )
        update, dups = self._fix_vocabulary(same.vocabulary_, self.vocabulary_)
        self.updated_vocabulary_ = update
        self.duplicated_vocabulary_ = dups
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class TraceableTfidfVectorizer(TfidfVectorizer, NGramsMixin):
    """
    Inherits from :class:`NGramsMixin` which overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to keep more information about n-grams but still produces the same
    outputs than `TfidfVectorizer`.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from mlinsights.mlmodel.sklearn_text import TraceableTfidfVectorizer
        from pprint import pformat

        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "Is this the first document?",
            "",
        ]).reshape((4, ))

        print('TfidfVectorizer from scikit-learn')
        mod1 = TfidfVectorizer(ngram_range=(1, 2),
                               token_pattern="[a-zA-Z ]{1,4}")
        mod1.fit(corpus)
        print(mod1.transform(corpus).todense()[:2])
        print(pformat(mod1.vocabulary_)[:100])

        print('TraceableTfidfVectorizer from scikit-learn')
        mod2 = TraceableTfidfVectorizer(ngram_range=(1, 2),
                                       token_pattern="[a-zA-Z ]{1,4}")
        mod2.fit(corpus)
        print(mod2.transform(corpus).todense()[:2])
        print(pformat(mod2.vocabulary_)[:100])

    The class is training an instance of TfidfVectorizer on the
    same data. This is used to update the vocabulary to match
    the same columns as the one obtained with scikit-learn.
    scikit-learn cannot distinguish between bi gram ("a b", "c") and
    ("a", "b c"). Therefore, there are merged into the same
    column by scikit-learn. This class, even if it is able to distinguish
    between them, keeps the same ambiguity."""

    def _word_ngrams(self, tokens, stop_words=None):
        return NGramsMixin._word_ngrams(self, tokens=tokens, stop_words=stop_words)

    def fit(self, X, y=None):
        super().fit(X, y=y)
        same = TfidfVectorizer(**self.get_params())
        same.fit(X, y=y)
        self.same_ = same
        if self.stop_words != same.stop_words:
            raise AssertionError(
                f"Different stop_words {self.stop_words} != {same.stop_words}."
            )
        update, dups = self._fix_vocabulary(same.vocabulary_, self.vocabulary_)
        self.updated_vocabulary_ = update
        self.duplicated_vocabulary_ = dups
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
