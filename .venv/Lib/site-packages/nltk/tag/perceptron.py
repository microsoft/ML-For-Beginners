# This module is a port of the Textblob Averaged Perceptron Tagger
# Author: Matthew Honnibal <honnibal+gh@gmail.com>,
#         Long Duong <longdt219@gmail.com> (NLTK port)
# URL: <https://github.com/sloria/textblob-aptagger>
#      <https://www.nltk.org/>
# Copyright 2013 Matthew Honnibal
# NLTK modifications Copyright 2015 The NLTK Project
#
# This module is provided under the terms of the MIT License.

import logging
import pickle
import random
from collections import defaultdict

from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI

try:
    import numpy as np
except ImportError:
    pass

PICKLE = "averaged_perceptron_tagger.pickle"


@jsontags.register_tag
class AveragedPerceptron:

    """An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
    """

    json_tag = "nltk.tag.perceptron.AveragedPerceptron"

    def __init__(self, weights=None):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = weights if weights else {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def _softmax(self, scores):
        s = np.fromiter(scores.values(), dtype=float)
        exps = np.exp(s)
        return exps / np.sum(exps)

    def predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        # compute the confidence
        conf = max(self._softmax(scores)) if return_conf == True else None

        return best_label, conf

    def update(self, truth, guess, features):
        """Update the feature weights."""

        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        """Average weights from all iterations."""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        """Save the pickled model weights."""
        with open(path, "wb") as fout:
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        """Load the pickled model weights."""
        self.weights = load(path)

    def encode_json_obj(self):
        return self.weights

    @classmethod
    def decode_json_obj(cls, obj):
        return cls(obj)


@jsontags.register_tag
class PerceptronTagger(TaggerI):

    """
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
    https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

    >>> from nltk.tag.perceptron import PerceptronTagger

    Train the model

    >>> tagger = PerceptronTagger(load=False)

    >>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
    ... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

    >>> tagger.tag(['today','is','a','beautiful','day'])
    [('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]

    Use the pretrain model (the default constructor)

    >>> pretrain = PerceptronTagger()

    >>> pretrain.tag('The quick brown fox jumps over the lazy dog'.split())
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

    >>> pretrain.tag("The red cat".split())
    [('The', 'DT'), ('red', 'JJ'), ('cat', 'NN')]
    """

    json_tag = "nltk.tag.sequential.PerceptronTagger"

    START = ["-START-", "-START2-"]
    END = ["-END-", "-END2-"]

    def __init__(self, load=True):
        """
        :param load: Load the pickled model upon instantiation.
        """
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            AP_MODEL_LOC = "file:" + str(
                find("taggers/averaged_perceptron_tagger/" + PICKLE)
            )
            self.load(AP_MODEL_LOC)

    def tag(self, tokens, return_conf=False, use_tagdict=True):
        """
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        """
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag, conf = (
                (self.tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
            )
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, conf = self.model.predict(features, return_conf)
            output.append((word, tag, conf) if return_conf == True else (word, tag))

            prev2 = prev
            prev = tag

        return output

    def train(self, sentences, save_loc=None, nr_iter=5):
        """Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        """
        # We'd like to allow ``sentences`` to be either a list or an iterator,
        # the latter being especially important for a large training dataset.
        # Because ``self._make_tagdict(sentences)`` runs regardless, we make
        # it populate ``self._sentences`` (a list) with all the sentences.
        # This saves the overheard of just iterating through ``sentences`` to
        # get the list by ``sentences = list(sentences)``.

        self._sentences = list()  # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for sentence in self._sentences:
                words, tags = zip(*sentence)

                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess, _ = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(self._sentences)
            logging.info(f"Iter {iter_}: {c}/{n}={_pc(c, n)}")

        # We don't need the training sentences anymore, and we don't want to
        # waste space on them when we pickle the trained tagger.
        self._sentences = None

        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            with open(save_loc, "wb") as fout:
                # changed protocol from -1 to 2 to make pickling Python 2 compatible
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)

    def load(self, loc):
        """
        :param loc: Load a pickled model at location.
        :type loc: str
        """

        self.model.weights, self.tagdict, self.classes = load(loc)
        self.model.classes = self.classes

    def encode_json_obj(self):
        return self.model.weights, self.tagdict, list(self.classes)

    @classmethod
    def decode_json_obj(cls, obj):
        tagger = cls(load=False)
        tagger.model.weights, tagger.tagdict, tagger.classes = obj
        tagger.classes = set(tagger.classes)
        tagger.model.classes = tagger.classes
        return tagger

    def normalize(self, word):
        """
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        """
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word and word[0].isdigit():
            return "!DIGITS"
        return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        """

        def add(name, *args):
            features[" ".join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add("bias")
        add("i suffix", word[-3:])
        add("i pref1", word[0] if word else "")
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1])
        add("i-1 suffix", context[i - 1][-3:])
        add("i-2 word", context[i - 2])
        add("i+1 word", context[i + 1])
        add("i+1 suffix", context[i + 1][-3:])
        add("i+2 word", context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        """
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        """
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            self._sentences.append(sentence)
            for word, tag in sentence:
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
                self.tagdict[word] = tag


def _pc(n, d):
    return (n / d) * 100


def _load_data_conll_format(filename):
    print("Read from file: ", filename)
    with open(filename, "rb") as fin:
        sentences = []
        sentence = []
        for line in fin.readlines():
            line = line.strip()
            # print line
            if len(line) == 0:
                sentences.append(sentence)
                sentence = []
                continue
            tokens = line.split("\t")
            word = tokens[1]
            tag = tokens[4]
            sentence.append((word, tag))
        return sentences


def _get_pretrain_model():
    # Train and test on English part of ConLL data (WSJ part of Penn Treebank)
    # Train: section 2-11
    # Test : section 23
    tagger = PerceptronTagger()
    training = _load_data_conll_format("english_ptb_train.conll")
    testing = _load_data_conll_format("english_ptb_test.conll")
    print("Size of training and testing (sentence)", len(training), len(testing))
    # Train and save the model
    tagger.train(training, PICKLE)
    print("Accuracy : ", tagger.accuracy(testing))


if __name__ == "__main__":
    # _get_pretrain_model()
    pass
