#
# Natural Language Toolkit: Sentiment Analyzer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Pierpaolo Pantone <24alsecondo@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A SentimentAnalyzer is a tool to implement and facilitate Sentiment Analysis tasks
using NLTK features and classifiers, especially for teaching and demonstrative
purposes.
"""

import sys
from collections import defaultdict

from nltk.classify.util import accuracy as eval_accuracy
from nltk.classify.util import apply_features
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import f_measure as eval_f_measure
from nltk.metrics import precision as eval_precision
from nltk.metrics import recall as eval_recall
from nltk.probability import FreqDist


class SentimentAnalyzer:
    """
    A Sentiment Analysis tool based on machine learning approaches.
    """

    def __init__(self, classifier=None):
        self.feat_extractors = defaultdict(list)
        self.classifier = classifier

    def all_words(self, documents, labeled=None):
        """
        Return all words/tokens from the documents (with duplicates).

        :param documents: a list of (words, label) tuples.
        :param labeled: if `True`, assume that each document is represented by a
            (words, label) tuple: (list(str), str). If `False`, each document is
            considered as being a simple list of strings: list(str).
        :rtype: list(str)
        :return: A list of all words/tokens in `documents`.
        """
        all_words = []
        if labeled is None:
            labeled = documents and isinstance(documents[0], tuple)
        if labeled:
            for words, _sentiment in documents:
                all_words.extend(words)
        elif not labeled:
            for words in documents:
                all_words.extend(words)
        return all_words

    def apply_features(self, documents, labeled=None):
        """
        Apply all feature extractor functions to the documents. This is a wrapper
        around `nltk.classify.util.apply_features`.

        If `labeled=False`, return featuresets as:
            [feature_func(doc) for doc in documents]
        If `labeled=True`, return featuresets as:
            [(feature_func(tok), label) for (tok, label) in toks]

        :param documents: a list of documents. `If labeled=True`, the method expects
            a list of (words, label) tuples.
        :rtype: LazyMap
        """
        return apply_features(self.extract_features, documents, labeled)

    def unigram_word_feats(self, words, top_n=None, min_freq=0):
        """
        Return most common top_n word features.

        :param words: a list of words/tokens.
        :param top_n: number of best words/tokens to use, sorted by frequency.
        :rtype: list(str)
        :return: A list of `top_n` words/tokens (with no duplicates) sorted by
            frequency.
        """
        # Stopwords are not removed
        unigram_feats_freqs = FreqDist(word for word in words)
        return [
            w
            for w, f in unigram_feats_freqs.most_common(top_n)
            if unigram_feats_freqs[w] > min_freq
        ]

    def bigram_collocation_feats(
        self, documents, top_n=None, min_freq=3, assoc_measure=BigramAssocMeasures.pmi
    ):
        """
        Return `top_n` bigram features (using `assoc_measure`).
        Note that this method is based on bigram collocations measures, and not
        on simple bigram frequency.

        :param documents: a list (or iterable) of tokens.
        :param top_n: number of best words/tokens to use, sorted by association
            measure.
        :param assoc_measure: bigram association measure to use as score function.
        :param min_freq: the minimum number of occurrencies of bigrams to take
            into consideration.

        :return: `top_n` ngrams scored by the given association measure.
        """
        finder = BigramCollocationFinder.from_documents(documents)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(assoc_measure, top_n)

    def classify(self, instance):
        """
        Classify a single instance applying the features that have already been
        stored in the SentimentAnalyzer.

        :param instance: a list (or iterable) of tokens.
        :return: the classification result given by applying the classifier.
        """
        instance_feats = self.apply_features([instance], labeled=False)
        return self.classifier.classify(instance_feats[0])

    def add_feat_extractor(self, function, **kwargs):
        """
        Add a new function to extract features from a document. This function will
        be used in extract_features().
        Important: in this step our kwargs are only representing additional parameters,
        and NOT the document we have to parse. The document will always be the first
        parameter in the parameter list, and it will be added in the extract_features()
        function.

        :param function: the extractor function to add to the list of feature extractors.
        :param kwargs: additional parameters required by the `function` function.
        """
        self.feat_extractors[function].append(kwargs)

    def extract_features(self, document):
        """
        Apply extractor functions (and their parameters) to the present document.
        We pass `document` as the first parameter of the extractor functions.
        If we want to use the same extractor function multiple times, we have to
        add it to the extractors with `add_feat_extractor` using multiple sets of
        parameters (one for each call of the extractor function).

        :param document: the document that will be passed as argument to the
            feature extractor functions.
        :return: A dictionary of populated features extracted from the document.
        :rtype: dict
        """
        all_features = {}
        for extractor in self.feat_extractors:
            for param_set in self.feat_extractors[extractor]:
                feats = extractor(document, **param_set)
            all_features.update(feats)
        return all_features

    def train(self, trainer, training_set, save_classifier=None, **kwargs):
        """
        Train classifier on the training set, optionally saving the output in the
        file specified by `save_classifier`.
        Additional arguments depend on the specific trainer used. For example,
        a MaxentClassifier can use `max_iter` parameter to specify the number
        of iterations, while a NaiveBayesClassifier cannot.

        :param trainer: `train` method of a classifier.
            E.g.: NaiveBayesClassifier.train
        :param training_set: the training set to be passed as argument to the
            classifier `train` method.
        :param save_classifier: the filename of the file where the classifier
            will be stored (optional).
        :param kwargs: additional parameters that will be passed as arguments to
            the classifier `train` function.
        :return: A classifier instance trained on the training set.
        :rtype:
        """
        print("Training classifier")
        self.classifier = trainer(training_set, **kwargs)
        if save_classifier:
            self.save_file(self.classifier, save_classifier)

        return self.classifier

    def save_file(self, content, filename):
        """
        Store `content` in `filename`. Can be used to store a SentimentAnalyzer.
        """
        print("Saving", filename, file=sys.stderr)
        with open(filename, "wb") as storage_file:
            import pickle

            # The protocol=2 parameter is for python2 compatibility
            pickle.dump(content, storage_file, protocol=2)

    def evaluate(
        self,
        test_set,
        classifier=None,
        accuracy=True,
        f_measure=True,
        precision=True,
        recall=True,
        verbose=False,
    ):
        """
        Evaluate and print classifier performance on the test set.

        :param test_set: A list of (tokens, label) tuples to use as gold set.
        :param classifier: a classifier instance (previously trained).
        :param accuracy: if `True`, evaluate classifier accuracy.
        :param f_measure: if `True`, evaluate classifier f_measure.
        :param precision: if `True`, evaluate classifier precision.
        :param recall: if `True`, evaluate classifier recall.
        :return: evaluation results.
        :rtype: dict(str): float
        """
        if classifier is None:
            classifier = self.classifier
        print(f"Evaluating {type(classifier).__name__} results...")
        metrics_results = {}
        if accuracy:
            accuracy_score = eval_accuracy(classifier, test_set)
            metrics_results["Accuracy"] = accuracy_score

        gold_results = defaultdict(set)
        test_results = defaultdict(set)
        labels = set()
        for i, (feats, label) in enumerate(test_set):
            labels.add(label)
            gold_results[label].add(i)
            observed = classifier.classify(feats)
            test_results[observed].add(i)

        for label in labels:
            if precision:
                precision_score = eval_precision(
                    gold_results[label], test_results[label]
                )
                metrics_results[f"Precision [{label}]"] = precision_score
            if recall:
                recall_score = eval_recall(gold_results[label], test_results[label])
                metrics_results[f"Recall [{label}]"] = recall_score
            if f_measure:
                f_measure_score = eval_f_measure(
                    gold_results[label], test_results[label]
                )
                metrics_results[f"F-measure [{label}]"] = f_measure_score

        # Print evaluation results (in alphabetical order)
        if verbose:
            for result in sorted(metrics_results):
                print(f"{result}: {metrics_results[result]}")

        return metrics_results
