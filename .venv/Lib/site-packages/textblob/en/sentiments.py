# -*- coding: utf-8 -*-
"""Sentiment analysis implementations.

.. versionadded:: 0.5.0
"""
from __future__ import absolute_import
from collections import namedtuple

import nltk

from textblob.en import sentiment as pattern_sentiment
from textblob.tokenizers import word_tokenize
from textblob.decorators import requires_nltk_corpus
from textblob.base import BaseSentimentAnalyzer, DISCRETE, CONTINUOUS


class PatternAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer that uses the same implementation as the
    pattern library. Returns results as a named tuple of the form:

    ``Sentiment(polarity, subjectivity, [assessments])``

    where [assessments] is a list of the assessed tokens and their
    polarity and subjectivity scores
    """
    kind = CONTINUOUS
    # This is only here for backwards-compatibility.
    # The return type is actually determined upon calling analyze()
    RETURN_TYPE = namedtuple('Sentiment', ['polarity', 'subjectivity'])

    def analyze(self, text, keep_assessments=False):
        """Return the sentiment as a named tuple of the form:
        ``Sentiment(polarity, subjectivity, [assessments])``.
        """
        #: Return type declaration
        if keep_assessments:
            Sentiment = namedtuple('Sentiment', ['polarity', 'subjectivity', 'assessments'])
            assessments = pattern_sentiment(text).assessments
            polarity, subjectivity = pattern_sentiment(text)
            return Sentiment(polarity, subjectivity, assessments)

        else:
            Sentiment = namedtuple('Sentiment', ['polarity', 'subjectivity'])
            return Sentiment(*pattern_sentiment(text))


def _default_feature_extractor(words):
    """Default feature extractor for the NaiveBayesAnalyzer."""
    return dict(((word, True) for word in words))


class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    """Naive Bayes analyzer that is trained on a dataset of movie reviews.
    Returns results as a named tuple of the form:
    ``Sentiment(classification, p_pos, p_neg)``

    :param callable feature_extractor: Function that returns a dictionary of
        features, given a list of words.
    """

    kind = DISCRETE
    #: Return type declaration
    RETURN_TYPE = namedtuple('Sentiment', ['classification', 'p_pos', 'p_neg'])

    def __init__(self, feature_extractor=_default_feature_extractor):
        super(NaiveBayesAnalyzer, self).__init__()
        self._classifier = None
        self.feature_extractor = feature_extractor

    @requires_nltk_corpus
    def train(self):
        """Train the Naive Bayes classifier on the movie review corpus."""
        super(NaiveBayesAnalyzer, self).train()
        neg_ids = nltk.corpus.movie_reviews.fileids('neg')
        pos_ids = nltk.corpus.movie_reviews.fileids('pos')
        neg_feats = [(self.feature_extractor(
            nltk.corpus.movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]
        pos_feats = [(self.feature_extractor(
            nltk.corpus.movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]
        train_data = neg_feats + pos_feats
        self._classifier = nltk.classify.NaiveBayesClassifier.train(train_data)

    def analyze(self, text):
        """Return the sentiment as a named tuple of the form:
        ``Sentiment(classification, p_pos, p_neg)``
        """
        # Lazily train the classifier
        super(NaiveBayesAnalyzer, self).analyze(text)
        tokens = word_tokenize(text, include_punc=False)
        filtered = (t.lower() for t in tokens if len(t) >= 3)
        feats = self.feature_extractor(filtered)
        prob_dist = self._classifier.prob_classify(feats)
        return self.RETURN_TYPE(
            classification=prob_dist.max(),
            p_pos=prob_dist.prob('pos'),
            p_neg=prob_dist.prob("neg")
        )
