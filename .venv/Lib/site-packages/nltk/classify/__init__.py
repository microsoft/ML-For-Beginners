# Natural Language Toolkit: Classifiers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes and interfaces for labeling tokens with category labels (or
"class labels").  Typically, labels are represented with strings
(such as ``'health'`` or ``'sports'``).  Classifiers can be used to
perform a wide range of classification tasks.  For example,
classifiers can be used...

- to classify documents by topic
- to classify ambiguous words by which word sense is intended
- to classify acoustic signals by which phoneme they represent
- to classify sentences by their author

Features
========
In order to decide which category label is appropriate for a given
token, classifiers examine one or more 'features' of the token.  These
"features" are typically chosen by hand, and indicate which aspects
of the token are relevant to the classification decision.  For
example, a document classifier might use a separate feature for each
word, recording how often that word occurred in the document.

Featuresets
===========
The features describing a token are encoded using a "featureset",
which is a dictionary that maps from "feature names" to "feature
values".  Feature names are unique strings that indicate what aspect
of the token is encoded by the feature.  Examples include
``'prevword'``, for a feature whose value is the previous word; and
``'contains-word(library)'`` for a feature that is true when a document
contains the word ``'library'``.  Feature values are typically
booleans, numbers, or strings, depending on which feature they
describe.

Featuresets are typically constructed using a "feature detector"
(also known as a "feature extractor").  A feature detector is a
function that takes a token (and sometimes information about its
context) as its input, and returns a featureset describing that token.
For example, the following feature detector converts a document
(stored as a list of words) to a featureset describing the set of
words included in the document:

    >>> # Define a feature detector function.
    >>> def document_features(document):
    ...     return dict([('contains-word(%s)' % w, True) for w in document])

Feature detectors are typically applied to each token before it is fed
to the classifier:

    >>> # Classify each Gutenberg document.
    >>> from nltk.corpus import gutenberg
    >>> for fileid in gutenberg.fileids(): # doctest: +SKIP
    ...     doc = gutenberg.words(fileid) # doctest: +SKIP
    ...     print(fileid, classifier.classify(document_features(doc))) # doctest: +SKIP

The parameters that a feature detector expects will vary, depending on
the task and the needs of the feature detector.  For example, a
feature detector for word sense disambiguation (WSD) might take as its
input a sentence, and the index of a word that should be classified,
and return a featureset for that word.  The following feature detector
for WSD includes features describing the left and right contexts of
the target word:

    >>> def wsd_features(sentence, index):
    ...     featureset = {}
    ...     for i in range(max(0, index-3), index):
    ...         featureset['left-context(%s)' % sentence[i]] = True
    ...     for i in range(index, max(index+3, len(sentence))):
    ...         featureset['right-context(%s)' % sentence[i]] = True
    ...     return featureset

Training Classifiers
====================
Most classifiers are built by training them on a list of hand-labeled
examples, known as the "training set".  Training sets are represented
as lists of ``(featuredict, label)`` tuples.
"""

from nltk.classify.api import ClassifierI, MultiClassifierI
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.maxent import (
    BinaryMaxentFeatureEncoding,
    ConditionalExponentialClassifier,
    MaxentClassifier,
    TypedMaxentFeatureEncoding,
)
from nltk.classify.megam import call_megam, config_megam
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.senna import Senna
from nltk.classify.textcat import TextCat
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.weka import WekaClassifier, config_weka
