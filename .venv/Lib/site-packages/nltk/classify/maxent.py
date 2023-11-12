# Natural Language Toolkit: Maximum Entropy Classifiers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Dmitry Chichkov <dchichkov@gmail.com> (TypedMaxentFeatureEncoding)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A classifier model based on maximum entropy modeling framework.  This
framework considers all of the probability distributions that are
empirically consistent with the training data; and chooses the
distribution with the highest entropy.  A probability distribution is
"empirically consistent" with a set of training data if its estimated
frequency with which a class and a feature vector value co-occur is
equal to the actual frequency in the data.

Terminology: 'feature'
======================
The term *feature* is usually used to refer to some property of an
unlabeled token.  For example, when performing word sense
disambiguation, we might define a ``'prevword'`` feature whose value is
the word preceding the target word.  However, in the context of
maxent modeling, the term *feature* is typically used to refer to a
property of a "labeled" token.  In order to prevent confusion, we
will introduce two distinct terms to disambiguate these two different
concepts:

  - An "input-feature" is a property of an unlabeled token.
  - A "joint-feature" is a property of a labeled token.

In the rest of the ``nltk.classify`` module, the term "features" is
used to refer to what we will call "input-features" in this module.

In literature that describes and discusses maximum entropy models,
input-features are typically called "contexts", and joint-features
are simply referred to as "features".

Converting Input-Features to Joint-Features
-------------------------------------------
In maximum entropy models, joint-features are required to have numeric
values.  Typically, each input-feature ``input_feat`` is mapped to a
set of joint-features of the form:

|   joint_feat(token, label) = { 1 if input_feat(token) == feat_val
|                              {      and label == some_label
|                              {
|                              { 0 otherwise

For all values of ``feat_val`` and ``some_label``.  This mapping is
performed by classes that implement the ``MaxentFeatureEncodingI``
interface.
"""
try:
    import numpy
except ImportError:
    pass

import os
import tempfile
from collections import defaultdict

from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict

__docformat__ = "epytext en"

######################################################################
# { Classifier Model
######################################################################


class MaxentClassifier(ClassifierI):
    """
    A maximum entropy classifier (also known as a "conditional
    exponential classifier").  This classifier is parameterized by a
    set of "weights", which are used to combine the joint-features
    that are generated from a featureset by an "encoding".  In
    particular, the encoding maps each ``(featureset, label)`` pair to
    a vector.  The probability of each label is then computed using
    the following equation::

                                dotprod(weights, encode(fs,label))
      prob(fs|label) = ---------------------------------------------------
                       sum(dotprod(weights, encode(fs,l)) for l in labels)

    Where ``dotprod`` is the dot product::

      dotprod(a,b) = sum(x*y for (x,y) in zip(a,b))
    """

    def __init__(self, encoding, weights, logarithmic=True):
        """
        Construct a new maxent classifier model.  Typically, new
        classifier models are created using the ``train()`` method.

        :type encoding: MaxentFeatureEncodingI
        :param encoding: An encoding that is used to convert the
            featuresets that are given to the ``classify`` method into
            joint-feature vectors, which are used by the maxent
            classifier model.

        :type weights: list of float
        :param weights:  The feature weight vector for this classifier.

        :type logarithmic: bool
        :param logarithmic: If false, then use non-logarithmic weights.
        """
        self._encoding = encoding
        self._weights = weights
        self._logarithmic = logarithmic
        # self._logarithmic = False
        assert encoding.length() == len(weights)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights):
        """
        Set the feature weight vector for this classifier.
        :param new_weights: The new feature weight vector.
        :type new_weights: list of float
        """
        self._weights = new_weights
        assert self._encoding.length() == len(new_weights)

    def weights(self):
        """
        :return: The feature weight vector for this classifier.
        :rtype: list of float
        """
        return self._weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        prob_dict = {}
        for label in self._encoding.labels():
            feature_vector = self._encoding.encode(featureset, label)

            if self._logarithmic:
                total = 0.0
                for (f_id, f_val) in feature_vector:
                    total += self._weights[f_id] * f_val
                prob_dict[label] = total

            else:
                prod = 1.0
                for (f_id, f_val) in feature_vector:
                    prod *= self._weights[f_id] ** f_val
                prob_dict[label] = prod

        # Normalize the dictionary to give a probability distribution
        return DictionaryProbDist(prob_dict, log=self._logarithmic, normalize=True)

    def explain(self, featureset, columns=4):
        """
        Print a table showing the effect of each of the features in
        the given feature set, and how they combine to determine the
        probabilities of each label for that featureset.
        """
        descr_width = 50
        TEMPLATE = "  %-" + str(descr_width - 2) + "s%s%8.3f"

        pdist = self.prob_classify(featureset)
        labels = sorted(pdist.samples(), key=pdist.prob, reverse=True)
        labels = labels[:columns]
        print(
            "  Feature".ljust(descr_width)
            + "".join("%8s" % (("%s" % l)[:7]) for l in labels)
        )
        print("  " + "-" * (descr_width - 2 + 8 * len(labels)))
        sums = defaultdict(int)
        for i, label in enumerate(labels):
            feature_vector = self._encoding.encode(featureset, label)
            feature_vector.sort(
                key=lambda fid__: abs(self._weights[fid__[0]]), reverse=True
            )
            for (f_id, f_val) in feature_vector:
                if self._logarithmic:
                    score = self._weights[f_id] * f_val
                else:
                    score = self._weights[f_id] ** f_val
                descr = self._encoding.describe(f_id)
                descr = descr.split(" and label is ")[0]  # hack
                descr += " (%s)" % f_val  # hack
                if len(descr) > 47:
                    descr = descr[:44] + "..."
                print(TEMPLATE % (descr, i * 8 * " ", score))
                sums[label] += score
        print("  " + "-" * (descr_width - 1 + 8 * len(labels)))
        print(
            "  TOTAL:".ljust(descr_width) + "".join("%8.3f" % sums[l] for l in labels)
        )
        print(
            "  PROBS:".ljust(descr_width)
            + "".join("%8.3f" % pdist.prob(l) for l in labels)
        )

    def most_informative_features(self, n=10):
        """
        Generates the ranked list of informative features from most to least.
        """
        if hasattr(self, "_most_informative_features"):
            return self._most_informative_features[:n]
        else:
            self._most_informative_features = sorted(
                list(range(len(self._weights))),
                key=lambda fid: abs(self._weights[fid]),
                reverse=True,
            )
            return self._most_informative_features[:n]

    def show_most_informative_features(self, n=10, show="all"):
        """
        :param show: all, neg, or pos (for negative-only or positive-only)
        :type show: str
        :param n: The no. of top features
        :type n: int
        """
        # Use None the full list of ranked features.
        fids = self.most_informative_features(None)
        if show == "pos":
            fids = [fid for fid in fids if self._weights[fid] > 0]
        elif show == "neg":
            fids = [fid for fid in fids if self._weights[fid] < 0]
        for fid in fids[:n]:
            print(f"{self._weights[fid]:8.3f} {self._encoding.describe(fid)}")

    def __repr__(self):
        return "<ConditionalExponentialClassifier: %d labels, %d features>" % (
            len(self._encoding.labels()),
            self._encoding.length(),
        )

    #: A list of the algorithm names that are accepted for the
    #: ``train()`` method's ``algorithm`` parameter.
    ALGORITHMS = ["GIS", "IIS", "MEGAM", "TADM"]

    @classmethod
    def train(
        cls,
        train_toks,
        algorithm=None,
        trace=3,
        encoding=None,
        labels=None,
        gaussian_prior_sigma=0,
        **cutoffs,
    ):
        """
        Train a new maxent classifier based on the given corpus of
        training samples.  This classifier will have its weights
        chosen to maximize entropy while remaining empirically
        consistent with the training corpus.

        :rtype: MaxentClassifier
        :return: The new maxent classifier

        :type train_toks: list
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a featureset,
            and the second of which is a classification label.

        :type algorithm: str
        :param algorithm: A case-insensitive string, specifying which
            algorithm should be used to train the classifier.  The
            following algorithms are currently available.

            - Iterative Scaling Methods: Generalized Iterative Scaling (``'GIS'``),
              Improved Iterative Scaling (``'IIS'``)
            - External Libraries (requiring megam):
              LM-BFGS algorithm, with training performed by Megam (``'megam'``)

            The default algorithm is ``'IIS'``.

        :type trace: int
        :param trace: The level of diagnostic tracing output to produce.
            Higher values produce more verbose output.
        :type encoding: MaxentFeatureEncodingI
        :param encoding: A feature encoding, used to convert featuresets
            into feature vectors.  If none is specified, then a
            ``BinaryMaxentFeatureEncoding`` will be built based on the
            features that are attested in the training corpus.
        :type labels: list(str)
        :param labels: The set of possible labels.  If none is given, then
            the set of all labels attested in the training data will be
            used instead.
        :param gaussian_prior_sigma: The sigma value for a gaussian
            prior on model weights.  Currently, this is supported by
            ``megam``. For other algorithms, its value is ignored.
        :param cutoffs: Arguments specifying various conditions under
            which the training should be halted.  (Some of the cutoff
            conditions are not supported by some algorithms.)

            - ``max_iter=v``: Terminate after ``v`` iterations.
            - ``min_ll=v``: Terminate after the negative average
              log-likelihood drops under ``v``.
            - ``min_lldelta=v``: Terminate if a single iteration improves
              log likelihood by less than ``v``.
        """
        if algorithm is None:
            algorithm = "iis"
        for key in cutoffs:
            if key not in (
                "max_iter",
                "min_ll",
                "min_lldelta",
                "max_acc",
                "min_accdelta",
                "count_cutoff",
                "norm",
                "explicit",
                "bernoulli",
            ):
                raise TypeError("Unexpected keyword arg %r" % key)
        algorithm = algorithm.lower()
        if algorithm == "iis":
            return train_maxent_classifier_with_iis(
                train_toks, trace, encoding, labels, **cutoffs
            )
        elif algorithm == "gis":
            return train_maxent_classifier_with_gis(
                train_toks, trace, encoding, labels, **cutoffs
            )
        elif algorithm == "megam":
            return train_maxent_classifier_with_megam(
                train_toks, trace, encoding, labels, gaussian_prior_sigma, **cutoffs
            )
        elif algorithm == "tadm":
            kwargs = cutoffs
            kwargs["trace"] = trace
            kwargs["encoding"] = encoding
            kwargs["labels"] = labels
            kwargs["gaussian_prior_sigma"] = gaussian_prior_sigma
            return TadmMaxentClassifier.train(train_toks, **kwargs)
        else:
            raise ValueError("Unknown algorithm %s" % algorithm)


#: Alias for MaxentClassifier.
ConditionalExponentialClassifier = MaxentClassifier


######################################################################
# { Feature Encodings
######################################################################


class MaxentFeatureEncodingI:
    """
    A mapping that converts a set of input-feature values to a vector
    of joint-feature values, given a label.  This conversion is
    necessary to translate featuresets into a format that can be used
    by maximum entropy models.

    The set of joint-features used by a given encoding is fixed, and
    each index in the generated joint-feature vectors corresponds to a
    single joint-feature.  The length of the generated joint-feature
    vectors is therefore constant (for a given encoding).

    Because the joint-feature vectors generated by
    ``MaxentFeatureEncodingI`` are typically very sparse, they are
    represented as a list of ``(index, value)`` tuples, specifying the
    value of each non-zero joint-feature.

    Feature encodings are generally created using the ``train()``
    method, which generates an appropriate encoding based on the
    input-feature values and labels that are present in a given
    corpus.
    """

    def encode(self, featureset, label):
        """
        Given a (featureset, label) pair, return the corresponding
        vector of joint-feature values.  This vector is represented as
        a list of ``(index, value)`` tuples, specifying the value of
        each non-zero joint-feature.

        :type featureset: dict
        :rtype: list(tuple(int, int))
        """
        raise NotImplementedError()

    def length(self):
        """
        :return: The size of the fixed-length joint-feature vectors
            that are generated by this encoding.
        :rtype: int
        """
        raise NotImplementedError()

    def labels(self):
        """
        :return: A list of the \"known labels\" -- i.e., all labels
            ``l`` such that ``self.encode(fs,l)`` can be a nonzero
            joint-feature vector for some value of ``fs``.
        :rtype: list
        """
        raise NotImplementedError()

    def describe(self, fid):
        """
        :return: A string describing the value of the joint-feature
            whose index in the generated feature vectors is ``fid``.
        :rtype: str
        """
        raise NotImplementedError()

    def train(cls, train_toks):
        """
        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.

        :type train_toks: list(tuple(dict, str))
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a feature dictionary,
            and the second of which is a classification label.
        """
        raise NotImplementedError()


class FunctionBackedMaxentFeatureEncoding(MaxentFeatureEncodingI):
    """
    A feature encoding that calls a user-supplied function to map a
    given featureset/label pair to a sparse joint-feature vector.
    """

    def __init__(self, func, length, labels):
        """
        Construct a new feature encoding based on the given function.

        :type func: (callable)
        :param func: A function that takes two arguments, a featureset
             and a label, and returns the sparse joint feature vector
             that encodes them::

                 func(featureset, label) -> feature_vector

             This sparse joint feature vector (``feature_vector``) is a
             list of ``(index,value)`` tuples.

        :type length: int
        :param length: The size of the fixed-length joint-feature
            vectors that are generated by this encoding.

        :type labels: list
        :param labels: A list of the \"known labels\" for this
            encoding -- i.e., all labels ``l`` such that
            ``self.encode(fs,l)`` can be a nonzero joint-feature vector
            for some value of ``fs``.
        """
        self._length = length
        self._func = func
        self._labels = labels

    def encode(self, featureset, label):
        return self._func(featureset, label)

    def length(self):
        return self._length

    def labels(self):
        return self._labels

    def describe(self, fid):
        return "no description available"


class BinaryMaxentFeatureEncoding(MaxentFeatureEncodingI):
    """
    A feature encoding that generates vectors containing a binary
    joint-features of the form:

    |  joint_feat(fs, l) = { 1 if (fs[fname] == fval) and (l == label)
    |                      {
    |                      { 0 otherwise

    Where ``fname`` is the name of an input-feature, ``fval`` is a value
    for that input-feature, and ``label`` is a label.

    Typically, these features are constructed based on a training
    corpus, using the ``train()`` method.  This method will create one
    feature for each combination of ``fname``, ``fval``, and ``label``
    that occurs at least once in the training corpus.

    The ``unseen_features`` parameter can be used to add "unseen-value
    features", which are used whenever an input feature has a value
    that was not encountered in the training corpus.  These features
    have the form:

    |  joint_feat(fs, l) = { 1 if is_unseen(fname, fs[fname])
    |                      {      and l == label
    |                      {
    |                      { 0 otherwise

    Where ``is_unseen(fname, fval)`` is true if the encoding does not
    contain any joint features that are true when ``fs[fname]==fval``.

    The ``alwayson_features`` parameter can be used to add "always-on
    features", which have the form::

    |  joint_feat(fs, l) = { 1 if (l == label)
    |                      {
    |                      { 0 otherwise

    These always-on features allow the maxent model to directly model
    the prior probabilities of each label.
    """

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        """
        :param labels: A list of the \"known labels\" for this encoding.

        :param mapping: A dictionary mapping from ``(fname,fval,label)``
            tuples to corresponding joint-feature indexes.  These
            indexes must be the set of integers from 0...len(mapping).
            If ``mapping[fname,fval,label]=id``, then
            ``self.encode(..., fname:fval, ..., label)[id]`` is 1;
            otherwise, it is 0.

        :param unseen_features: If true, then include unseen value
           features in the generated joint-feature vectors.

        :param alwayson_features: If true, then include always-on
           features in the generated joint-feature vectors.
        """
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError(
                "Mapping values must be exactly the "
                "set of integers from 0...len(mapping)"
            )

        self._labels = list(labels)
        """A list of attested labels."""

        self._mapping = mapping
        """dict mapping from (fname,fval,label) -> fid"""

        self._length = len(mapping)
        """The length of generated joint feature vectors."""

        self._alwayson = None
        """dict mapping from label -> fid"""

        self._unseen = None
        """dict mapping from fname -> fid"""

        if alwayson_features:
            self._alwayson = {
                label: i + self._length for (i, label) in enumerate(labels)
            }
            self._length += len(self._alwayson)

        if unseen_features:
            fnames = {fname for (fname, fval, label) in mapping}
            self._unseen = {fname: i + self._length for (i, fname) in enumerate(fnames)}
            self._length += len(fnames)

    def encode(self, featureset, label):
        # Inherit docs.
        encoding = []

        # Convert input-features to joint-features:
        for fname, fval in featureset.items():
            # Known feature name & value:
            if (fname, fval, label) in self._mapping:
                encoding.append((self._mapping[fname, fval, label], 1))

            # Otherwise, we might want to fire an "unseen-value feature".
            elif self._unseen:
                # Have we seen this fname/fval combination with any label?
                for label2 in self._labels:
                    if (fname, fval, label2) in self._mapping:
                        break  # we've seen this fname/fval combo
                # We haven't -- fire the unseen-value feature
                else:
                    if fname in self._unseen:
                        encoding.append((self._unseen[fname], 1))

        # Add always-on features:
        if self._alwayson and label in self._alwayson:
            encoding.append((self._alwayson[label], 1))

        return encoding

    def describe(self, f_id):
        # Inherit docs.
        if not isinstance(f_id, int):
            raise TypeError("describe() expected an int")
        try:
            self._inv_mapping
        except AttributeError:
            self._inv_mapping = [-1] * len(self._mapping)
            for (info, i) in self._mapping.items():
                self._inv_mapping[i] = info

        if f_id < len(self._mapping):
            (fname, fval, label) = self._inv_mapping[f_id]
            return f"{fname}=={fval!r} and label is {label!r}"
        elif self._alwayson and f_id in self._alwayson.values():
            for (label, f_id2) in self._alwayson.items():
                if f_id == f_id2:
                    return "label is %r" % label
        elif self._unseen and f_id in self._unseen.values():
            for (fname, f_id2) in self._unseen.items():
                if f_id == f_id2:
                    return "%s is unseen" % fname
        else:
            raise ValueError("Bad feature id")

    def labels(self):
        # Inherit docs.
        return self._labels

    def length(self):
        # Inherit docs.
        return self._length

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        """
        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.  See the class description
        ``BinaryMaxentFeatureEncoding`` for a description of the
        joint-features that will be included in this encoding.

        :type train_toks: list(tuple(dict, str))
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a feature dictionary,
            and the second of which is a classification label.

        :type count_cutoff: int
        :param count_cutoff: A cutoff value that is used to discard
            rare joint-features.  If a joint-feature's value is 1
            fewer than ``count_cutoff`` times in the training corpus,
            then that joint-feature is not included in the generated
            encoding.

        :type labels: list
        :param labels: A list of labels that should be used by the
            classifier.  If not specified, then the set of labels
            attested in ``train_toks`` will be used.

        :param options: Extra parameters for the constructor, such as
            ``unseen_features`` and ``alwayson_features``.
        """
        mapping = {}  # maps (fname, fval, label) -> fid
        seen_labels = set()  # The set of labels we've encountered
        count = defaultdict(int)  # maps (fname, fval) -> count

        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError("Unexpected label %s" % label)
            seen_labels.add(label)

            # Record each of the features.
            for (fname, fval) in tok.items():

                # If a count cutoff is given, then only add a joint
                # feature once the corresponding (fname, fval, label)
                # tuple exceeds that cutoff.
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)

        if labels is None:
            labels = seen_labels
        return cls(labels, mapping, **options)


class GISEncoding(BinaryMaxentFeatureEncoding):
    """
    A binary feature encoding which adds one new joint-feature to the
    joint-features defined by ``BinaryMaxentFeatureEncoding``: a
    correction feature, whose value is chosen to ensure that the
    sparse vector always sums to a constant non-negative number.  This
    new feature is used to ensure two preconditions for the GIS
    training algorithm:

      - At least one feature vector index must be nonzero for every
        token.
      - The feature vector must sum to a constant non-negative number
        for every token.
    """

    def __init__(
        self, labels, mapping, unseen_features=False, alwayson_features=False, C=None
    ):
        """
        :param C: The correction constant.  The value of the correction
            feature is based on this value.  In particular, its value is
            ``C - sum([v for (f,v) in encoding])``.
        :seealso: ``BinaryMaxentFeatureEncoding.__init__``
        """
        BinaryMaxentFeatureEncoding.__init__(
            self, labels, mapping, unseen_features, alwayson_features
        )
        if C is None:
            C = len({fname for (fname, fval, label) in mapping}) + 1
        self._C = C

    @property
    def C(self):
        """The non-negative constant that all encoded feature vectors
        will sum to."""
        return self._C

    def encode(self, featureset, label):
        # Get the basic encoding.
        encoding = BinaryMaxentFeatureEncoding.encode(self, featureset, label)
        base_length = BinaryMaxentFeatureEncoding.length(self)

        # Add a correction feature.
        total = sum(v for (f, v) in encoding)
        if total >= self._C:
            raise ValueError("Correction feature is not high enough!")
        encoding.append((base_length, self._C - total))

        # Return the result
        return encoding

    def length(self):
        return BinaryMaxentFeatureEncoding.length(self) + 1

    def describe(self, f_id):
        if f_id == BinaryMaxentFeatureEncoding.length(self):
            return "Correction feature (%s)" % self._C
        else:
            return BinaryMaxentFeatureEncoding.describe(self, f_id)


class TadmEventMaxentFeatureEncoding(BinaryMaxentFeatureEncoding):
    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        self._mapping = OrderedDict(mapping)
        self._label_mapping = OrderedDict()
        BinaryMaxentFeatureEncoding.__init__(
            self, labels, self._mapping, unseen_features, alwayson_features
        )

    def encode(self, featureset, label):
        encoding = []
        for feature, value in featureset.items():
            if (feature, label) not in self._mapping:
                self._mapping[(feature, label)] = len(self._mapping)
            if value not in self._label_mapping:
                if not isinstance(value, int):
                    self._label_mapping[value] = len(self._label_mapping)
                else:
                    self._label_mapping[value] = value
            encoding.append(
                (self._mapping[(feature, label)], self._label_mapping[value])
            )
        return encoding

    def labels(self):
        return self._labels

    def describe(self, fid):
        for (feature, label) in self._mapping:
            if self._mapping[(feature, label)] == fid:
                return (feature, label)

    def length(self):
        return len(self._mapping)

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        mapping = OrderedDict()
        if not labels:
            labels = []

        # This gets read twice, so compute the values in case it's lazy.
        train_toks = list(train_toks)

        for (featureset, label) in train_toks:
            if label not in labels:
                labels.append(label)

        for (featureset, label) in train_toks:
            for label in labels:
                for feature in featureset:
                    if (feature, label) not in mapping:
                        mapping[(feature, label)] = len(mapping)

        return cls(labels, mapping, **options)


class TypedMaxentFeatureEncoding(MaxentFeatureEncodingI):
    """
    A feature encoding that generates vectors containing integer,
    float and binary joint-features of the form:

    Binary (for string and boolean features):

    |  joint_feat(fs, l) = { 1 if (fs[fname] == fval) and (l == label)
    |                      {
    |                      { 0 otherwise

    Value (for integer and float features):

    |  joint_feat(fs, l) = { fval if     (fs[fname] == type(fval))
    |                      {         and (l == label)
    |                      {
    |                      { not encoded otherwise

    Where ``fname`` is the name of an input-feature, ``fval`` is a value
    for that input-feature, and ``label`` is a label.

    Typically, these features are constructed based on a training
    corpus, using the ``train()`` method.

    For string and boolean features [type(fval) not in (int, float)]
    this method will create one feature for each combination of
    ``fname``, ``fval``, and ``label`` that occurs at least once in the
    training corpus.

    For integer and float features [type(fval) in (int, float)] this
    method will create one feature for each combination of ``fname``
    and ``label`` that occurs at least once in the training corpus.

    For binary features the ``unseen_features`` parameter can be used
    to add "unseen-value features", which are used whenever an input
    feature has a value that was not encountered in the training
    corpus.  These features have the form:

    |  joint_feat(fs, l) = { 1 if is_unseen(fname, fs[fname])
    |                      {      and l == label
    |                      {
    |                      { 0 otherwise

    Where ``is_unseen(fname, fval)`` is true if the encoding does not
    contain any joint features that are true when ``fs[fname]==fval``.

    The ``alwayson_features`` parameter can be used to add "always-on
    features", which have the form:

    |  joint_feat(fs, l) = { 1 if (l == label)
    |                      {
    |                      { 0 otherwise

    These always-on features allow the maxent model to directly model
    the prior probabilities of each label.
    """

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        """
        :param labels: A list of the \"known labels\" for this encoding.

        :param mapping: A dictionary mapping from ``(fname,fval,label)``
            tuples to corresponding joint-feature indexes.  These
            indexes must be the set of integers from 0...len(mapping).
            If ``mapping[fname,fval,label]=id``, then
            ``self.encode({..., fname:fval, ...``, label)[id]} is 1;
            otherwise, it is 0.

        :param unseen_features: If true, then include unseen value
           features in the generated joint-feature vectors.

        :param alwayson_features: If true, then include always-on
           features in the generated joint-feature vectors.
        """
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError(
                "Mapping values must be exactly the "
                "set of integers from 0...len(mapping)"
            )

        self._labels = list(labels)
        """A list of attested labels."""

        self._mapping = mapping
        """dict mapping from (fname,fval,label) -> fid"""

        self._length = len(mapping)
        """The length of generated joint feature vectors."""

        self._alwayson = None
        """dict mapping from label -> fid"""

        self._unseen = None
        """dict mapping from fname -> fid"""

        if alwayson_features:
            self._alwayson = {
                label: i + self._length for (i, label) in enumerate(labels)
            }
            self._length += len(self._alwayson)

        if unseen_features:
            fnames = {fname for (fname, fval, label) in mapping}
            self._unseen = {fname: i + self._length for (i, fname) in enumerate(fnames)}
            self._length += len(fnames)

    def encode(self, featureset, label):
        # Inherit docs.
        encoding = []

        # Convert input-features to joint-features:
        for fname, fval in featureset.items():
            if isinstance(fval, (int, float)):
                # Known feature name & value:
                if (fname, type(fval), label) in self._mapping:
                    encoding.append((self._mapping[fname, type(fval), label], fval))
            else:
                # Known feature name & value:
                if (fname, fval, label) in self._mapping:
                    encoding.append((self._mapping[fname, fval, label], 1))

                # Otherwise, we might want to fire an "unseen-value feature".
                elif self._unseen:
                    # Have we seen this fname/fval combination with any label?
                    for label2 in self._labels:
                        if (fname, fval, label2) in self._mapping:
                            break  # we've seen this fname/fval combo
                    # We haven't -- fire the unseen-value feature
                    else:
                        if fname in self._unseen:
                            encoding.append((self._unseen[fname], 1))

        # Add always-on features:
        if self._alwayson and label in self._alwayson:
            encoding.append((self._alwayson[label], 1))

        return encoding

    def describe(self, f_id):
        # Inherit docs.
        if not isinstance(f_id, int):
            raise TypeError("describe() expected an int")
        try:
            self._inv_mapping
        except AttributeError:
            self._inv_mapping = [-1] * len(self._mapping)
            for (info, i) in self._mapping.items():
                self._inv_mapping[i] = info

        if f_id < len(self._mapping):
            (fname, fval, label) = self._inv_mapping[f_id]
            return f"{fname}=={fval!r} and label is {label!r}"
        elif self._alwayson and f_id in self._alwayson.values():
            for (label, f_id2) in self._alwayson.items():
                if f_id == f_id2:
                    return "label is %r" % label
        elif self._unseen and f_id in self._unseen.values():
            for (fname, f_id2) in self._unseen.items():
                if f_id == f_id2:
                    return "%s is unseen" % fname
        else:
            raise ValueError("Bad feature id")

    def labels(self):
        # Inherit docs.
        return self._labels

    def length(self):
        # Inherit docs.
        return self._length

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        """
        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.  See the class description
        ``TypedMaxentFeatureEncoding`` for a description of the
        joint-features that will be included in this encoding.

        Note: recognized feature values types are (int, float), over
        types are interpreted as regular binary features.

        :type train_toks: list(tuple(dict, str))
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a feature dictionary,
            and the second of which is a classification label.

        :type count_cutoff: int
        :param count_cutoff: A cutoff value that is used to discard
            rare joint-features.  If a joint-feature's value is 1
            fewer than ``count_cutoff`` times in the training corpus,
            then that joint-feature is not included in the generated
            encoding.

        :type labels: list
        :param labels: A list of labels that should be used by the
            classifier.  If not specified, then the set of labels
            attested in ``train_toks`` will be used.

        :param options: Extra parameters for the constructor, such as
            ``unseen_features`` and ``alwayson_features``.
        """
        mapping = {}  # maps (fname, fval, label) -> fid
        seen_labels = set()  # The set of labels we've encountered
        count = defaultdict(int)  # maps (fname, fval) -> count

        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError("Unexpected label %s" % label)
            seen_labels.add(label)

            # Record each of the features.
            for (fname, fval) in tok.items():
                if type(fval) in (int, float):
                    fval = type(fval)
                # If a count cutoff is given, then only add a joint
                # feature once the corresponding (fname, fval, label)
                # tuple exceeds that cutoff.
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)

        if labels is None:
            labels = seen_labels
        return cls(labels, mapping, **options)


######################################################################
# { Classifier Trainer: Generalized Iterative Scaling
######################################################################


def train_maxent_classifier_with_gis(
    train_toks, trace=3, encoding=None, labels=None, **cutoffs
):
    """
    Train a new ``ConditionalExponentialClassifier``, using the given
    training samples, using the Generalized Iterative Scaling
    algorithm.  This ``ConditionalExponentialClassifier`` will encode
    the model that maximizes entropy from all the models that are
    empirically consistent with ``train_toks``.

    :see: ``train_maxent_classifier()`` for parameter descriptions.
    """
    cutoffs.setdefault("max_iter", 100)
    cutoffchecker = CutoffChecker(cutoffs)

    # Construct an encoding from the training data.
    if encoding is None:
        encoding = GISEncoding.train(train_toks, labels=labels)

    if not hasattr(encoding, "C"):
        raise TypeError(
            "The GIS algorithm requires an encoding that "
            "defines C (e.g., GISEncoding)."
        )

    # Cinv is the inverse of the sum of each joint feature vector.
    # This controls the learning rate: higher Cinv (or lower C) gives
    # faster learning.
    Cinv = 1.0 / encoding.C

    # Count how many times each feature occurs in the training data.
    empirical_fcount = calculate_empirical_fcount(train_toks, encoding)

    # Check for any features that are not attested in train_toks.
    unattested = set(numpy.nonzero(empirical_fcount == 0)[0])

    # Build the classifier.  Start with weight=0 for each attested
    # feature, and weight=-infinity for each unattested feature.
    weights = numpy.zeros(len(empirical_fcount), "d")
    for fid in unattested:
        weights[fid] = numpy.NINF
    classifier = ConditionalExponentialClassifier(encoding, weights)

    # Take the log of the empirical fcount.
    log_empirical_fcount = numpy.log2(empirical_fcount)
    del empirical_fcount

    if trace > 0:
        print("  ==> Training (%d iterations)" % cutoffs["max_iter"])
    if trace > 2:
        print()
        print("      Iteration    Log Likelihood    Accuracy")
        print("      ---------------------------------------")

    # Train the classifier.
    try:
        while True:
            if trace > 2:
                ll = cutoffchecker.ll or log_likelihood(classifier, train_toks)
                acc = cutoffchecker.acc or accuracy(classifier, train_toks)
                iternum = cutoffchecker.iter
                print("     %9d    %14.5f    %9.3f" % (iternum, ll, acc))

            # Use the model to estimate the number of times each
            # feature should occur in the training data.
            estimated_fcount = calculate_estimated_fcount(
                classifier, train_toks, encoding
            )

            # Take the log of estimated fcount (avoid taking log(0).)
            for fid in unattested:
                estimated_fcount[fid] += 1
            log_estimated_fcount = numpy.log2(estimated_fcount)
            del estimated_fcount

            # Update the classifier weights
            weights = classifier.weights()
            weights += (log_empirical_fcount - log_estimated_fcount) * Cinv
            classifier.set_weights(weights)

            # Check the log-likelihood & accuracy cutoffs.
            if cutoffchecker.check(classifier, train_toks):
                break

    except KeyboardInterrupt:
        print("      Training stopped: keyboard interrupt")
    except:
        raise

    if trace > 2:
        ll = log_likelihood(classifier, train_toks)
        acc = accuracy(classifier, train_toks)
        print(f"         Final    {ll:14.5f}    {acc:9.3f}")

    # Return the classifier.
    return classifier


def calculate_empirical_fcount(train_toks, encoding):
    fcount = numpy.zeros(encoding.length(), "d")

    for tok, label in train_toks:
        for (index, val) in encoding.encode(tok, label):
            fcount[index] += val

    return fcount


def calculate_estimated_fcount(classifier, train_toks, encoding):
    fcount = numpy.zeros(encoding.length(), "d")

    for tok, label in train_toks:
        pdist = classifier.prob_classify(tok)
        for label in pdist.samples():
            prob = pdist.prob(label)
            for (fid, fval) in encoding.encode(tok, label):
                fcount[fid] += prob * fval

    return fcount


######################################################################
# { Classifier Trainer: Improved Iterative Scaling
######################################################################


def train_maxent_classifier_with_iis(
    train_toks, trace=3, encoding=None, labels=None, **cutoffs
):
    """
    Train a new ``ConditionalExponentialClassifier``, using the given
    training samples, using the Improved Iterative Scaling algorithm.
    This ``ConditionalExponentialClassifier`` will encode the model
    that maximizes entropy from all the models that are empirically
    consistent with ``train_toks``.

    :see: ``train_maxent_classifier()`` for parameter descriptions.
    """
    cutoffs.setdefault("max_iter", 100)
    cutoffchecker = CutoffChecker(cutoffs)

    # Construct an encoding from the training data.
    if encoding is None:
        encoding = BinaryMaxentFeatureEncoding.train(train_toks, labels=labels)

    # Count how many times each feature occurs in the training data.
    empirical_ffreq = calculate_empirical_fcount(train_toks, encoding) / len(train_toks)

    # Find the nf map, and related variables nfarray and nfident.
    # nf is the sum of the features for a given labeled text.
    # nfmap compresses this sparse set of values to a dense list.
    # nfarray performs the reverse operation.  nfident is
    # nfarray multiplied by an identity matrix.
    nfmap = calculate_nfmap(train_toks, encoding)
    nfarray = numpy.array(sorted(nfmap, key=nfmap.__getitem__), "d")
    nftranspose = numpy.reshape(nfarray, (len(nfarray), 1))

    # Check for any features that are not attested in train_toks.
    unattested = set(numpy.nonzero(empirical_ffreq == 0)[0])

    # Build the classifier.  Start with weight=0 for each attested
    # feature, and weight=-infinity for each unattested feature.
    weights = numpy.zeros(len(empirical_ffreq), "d")
    for fid in unattested:
        weights[fid] = numpy.NINF
    classifier = ConditionalExponentialClassifier(encoding, weights)

    if trace > 0:
        print("  ==> Training (%d iterations)" % cutoffs["max_iter"])
    if trace > 2:
        print()
        print("      Iteration    Log Likelihood    Accuracy")
        print("      ---------------------------------------")

    # Train the classifier.
    try:
        while True:
            if trace > 2:
                ll = cutoffchecker.ll or log_likelihood(classifier, train_toks)
                acc = cutoffchecker.acc or accuracy(classifier, train_toks)
                iternum = cutoffchecker.iter
                print("     %9d    %14.5f    %9.3f" % (iternum, ll, acc))

            # Calculate the deltas for this iteration, using Newton's method.
            deltas = calculate_deltas(
                train_toks,
                classifier,
                unattested,
                empirical_ffreq,
                nfmap,
                nfarray,
                nftranspose,
                encoding,
            )

            # Use the deltas to update our weights.
            weights = classifier.weights()
            weights += deltas
            classifier.set_weights(weights)

            # Check the log-likelihood & accuracy cutoffs.
            if cutoffchecker.check(classifier, train_toks):
                break

    except KeyboardInterrupt:
        print("      Training stopped: keyboard interrupt")
    except:
        raise

    if trace > 2:
        ll = log_likelihood(classifier, train_toks)
        acc = accuracy(classifier, train_toks)
        print(f"         Final    {ll:14.5f}    {acc:9.3f}")

    # Return the classifier.
    return classifier


def calculate_nfmap(train_toks, encoding):
    """
    Construct a map that can be used to compress ``nf`` (which is
    typically sparse).

    *nf(feature_vector)* is the sum of the feature values for
    *feature_vector*.

    This represents the number of features that are active for a
    given labeled text.  This method finds all values of *nf(t)*
    that are attested for at least one token in the given list of
    training tokens; and constructs a dictionary mapping these
    attested values to a continuous range *0...N*.  For example,
    if the only values of *nf()* that were attested were 3, 5, and
    7, then ``_nfmap`` might return the dictionary ``{3:0, 5:1, 7:2}``.

    :return: A map that can be used to compress ``nf`` to a dense
        vector.
    :rtype: dict(int -> int)
    """
    # Map from nf to indices.  This allows us to use smaller arrays.
    nfset = set()
    for tok, _ in train_toks:
        for label in encoding.labels():
            nfset.add(sum(val for (id, val) in encoding.encode(tok, label)))
    return {nf: i for (i, nf) in enumerate(nfset)}


def calculate_deltas(
    train_toks,
    classifier,
    unattested,
    ffreq_empirical,
    nfmap,
    nfarray,
    nftranspose,
    encoding,
):
    r"""
    Calculate the update values for the classifier weights for
    this iteration of IIS.  These update weights are the value of
    ``delta`` that solves the equation::

      ffreq_empirical[i]
             =
      SUM[fs,l] (classifier.prob_classify(fs).prob(l) *
                 feature_vector(fs,l)[i] *
                 exp(delta[i] * nf(feature_vector(fs,l))))

    Where:
        - *(fs,l)* is a (featureset, label) tuple from ``train_toks``
        - *feature_vector(fs,l)* = ``encoding.encode(fs,l)``
        - *nf(vector)* = ``sum([val for (id,val) in vector])``

    This method uses Newton's method to solve this equation for
    *delta[i]*.  In particular, it starts with a guess of
    ``delta[i]`` = 1; and iteratively updates ``delta`` with:

    | delta[i] -= (ffreq_empirical[i] - sum1[i])/(-sum2[i])

    until convergence, where *sum1* and *sum2* are defined as:

    |    sum1[i](delta) = SUM[fs,l] f[i](fs,l,delta)
    |    sum2[i](delta) = SUM[fs,l] (f[i](fs,l,delta).nf(feature_vector(fs,l)))
    |    f[i](fs,l,delta) = (classifier.prob_classify(fs).prob(l) .
    |                        feature_vector(fs,l)[i] .
    |                        exp(delta[i] . nf(feature_vector(fs,l))))

    Note that *sum1* and *sum2* depend on ``delta``; so they need
    to be re-computed each iteration.

    The variables ``nfmap``, ``nfarray``, and ``nftranspose`` are
    used to generate a dense encoding for *nf(ltext)*.  This
    allows ``_deltas`` to calculate *sum1* and *sum2* using
    matrices, which yields a significant performance improvement.

    :param train_toks: The set of training tokens.
    :type train_toks: list(tuple(dict, str))
    :param classifier: The current classifier.
    :type classifier: ClassifierI
    :param ffreq_empirical: An array containing the empirical
        frequency for each feature.  The *i*\ th element of this
        array is the empirical frequency for feature *i*.
    :type ffreq_empirical: sequence of float
    :param unattested: An array that is 1 for features that are
        not attested in the training data; and 0 for features that
        are attested.  In other words, ``unattested[i]==0`` iff
        ``ffreq_empirical[i]==0``.
    :type unattested: sequence of int
    :param nfmap: A map that can be used to compress ``nf`` to a dense
        vector.
    :type nfmap: dict(int -> int)
    :param nfarray: An array that can be used to uncompress ``nf``
        from a dense vector.
    :type nfarray: array(float)
    :param nftranspose: The transpose of ``nfarray``
    :type nftranspose: array(float)
    """
    # These parameters control when we decide that we've
    # converged.  It probably should be possible to set these
    # manually, via keyword arguments to train.
    NEWTON_CONVERGE = 1e-12
    MAX_NEWTON = 300

    deltas = numpy.ones(encoding.length(), "d")

    # Precompute the A matrix:
    # A[nf][id] = sum ( p(fs) * p(label|fs) * f(fs,label) )
    # over all label,fs s.t. num_features[label,fs]=nf
    A = numpy.zeros((len(nfmap), encoding.length()), "d")

    for tok, label in train_toks:
        dist = classifier.prob_classify(tok)

        for label in encoding.labels():
            # Generate the feature vector
            feature_vector = encoding.encode(tok, label)
            # Find the number of active features
            nf = sum(val for (id, val) in feature_vector)
            # Update the A matrix
            for (id, val) in feature_vector:
                A[nfmap[nf], id] += dist.prob(label) * val
    A /= len(train_toks)

    # Iteratively solve for delta.  Use the following variables:
    #   - nf_delta[x][y] = nfarray[x] * delta[y]
    #   - exp_nf_delta[x][y] = exp(nf[x] * delta[y])
    #   - nf_exp_nf_delta[x][y] = nf[x] * exp(nf[x] * delta[y])
    #   - sum1[i][nf] = sum p(fs)p(label|fs)f[i](label,fs)
    #                       exp(delta[i]nf)
    #   - sum2[i][nf] = sum p(fs)p(label|fs)f[i](label,fs)
    #                       nf exp(delta[i]nf)
    for rangenum in range(MAX_NEWTON):
        nf_delta = numpy.outer(nfarray, deltas)
        exp_nf_delta = 2**nf_delta
        nf_exp_nf_delta = nftranspose * exp_nf_delta
        sum1 = numpy.sum(exp_nf_delta * A, axis=0)
        sum2 = numpy.sum(nf_exp_nf_delta * A, axis=0)

        # Avoid division by zero.
        for fid in unattested:
            sum2[fid] += 1

        # Update the deltas.
        deltas -= (ffreq_empirical - sum1) / -sum2

        # We can stop once we converge.
        n_error = numpy.sum(abs(ffreq_empirical - sum1)) / numpy.sum(abs(deltas))
        if n_error < NEWTON_CONVERGE:
            return deltas

    return deltas


######################################################################
# { Classifier Trainer: megam
######################################################################

# [xx] possible extension: add support for using implicit file format;
# this would need to put requirements on what encoding is used.  But
# we may need this for other maxent classifier trainers that require
# implicit formats anyway.
def train_maxent_classifier_with_megam(
    train_toks, trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, **kwargs
):
    """
    Train a new ``ConditionalExponentialClassifier``, using the given
    training samples, using the external ``megam`` library.  This
    ``ConditionalExponentialClassifier`` will encode the model that
    maximizes entropy from all the models that are empirically
    consistent with ``train_toks``.

    :see: ``train_maxent_classifier()`` for parameter descriptions.
    :see: ``nltk.classify.megam``
    """

    explicit = True
    bernoulli = True
    if "explicit" in kwargs:
        explicit = kwargs["explicit"]
    if "bernoulli" in kwargs:
        bernoulli = kwargs["bernoulli"]

    # Construct an encoding from the training data.
    if encoding is None:
        # Count cutoff can also be controlled by megam with the -minfc
        # option. Not sure where the best place for it is.
        count_cutoff = kwargs.get("count_cutoff", 0)
        encoding = BinaryMaxentFeatureEncoding.train(
            train_toks, count_cutoff, labels=labels, alwayson_features=True
        )
    elif labels is not None:
        raise ValueError("Specify encoding or labels, not both")

    # Write a training file for megam.
    try:
        fd, trainfile_name = tempfile.mkstemp(prefix="nltk-")
        with open(trainfile_name, "w") as trainfile:
            write_megam_file(
                train_toks, encoding, trainfile, explicit=explicit, bernoulli=bernoulli
            )
        os.close(fd)
    except (OSError, ValueError) as e:
        raise ValueError("Error while creating megam training file: %s" % e) from e

    # Run megam on the training file.
    options = []
    options += ["-nobias", "-repeat", "10"]
    if explicit:
        options += ["-explicit"]
    if not bernoulli:
        options += ["-fvals"]
    if gaussian_prior_sigma:
        # Lambda is just the precision of the Gaussian prior, i.e. it's the
        # inverse variance, so the parameter conversion is 1.0/sigma**2.
        # See https://users.umiacs.umd.edu/~hal/docs/daume04cg-bfgs.pdf
        inv_variance = 1.0 / gaussian_prior_sigma**2
    else:
        inv_variance = 0
    options += ["-lambda", "%.2f" % inv_variance, "-tune"]
    if trace < 3:
        options += ["-quiet"]
    if "max_iter" in kwargs:
        options += ["-maxi", "%s" % kwargs["max_iter"]]
    if "ll_delta" in kwargs:
        # [xx] this is actually a perplexity delta, not a log
        # likelihood delta
        options += ["-dpp", "%s" % abs(kwargs["ll_delta"])]
    if hasattr(encoding, "cost"):
        options += ["-multilabel"]  # each possible la
    options += ["multiclass", trainfile_name]
    stdout = call_megam(options)
    # print('./megam_i686.opt ', ' '.join(options))
    # Delete the training file
    try:
        os.remove(trainfile_name)
    except OSError as e:
        print(f"Warning: unable to delete {trainfile_name}: {e}")

    # Parse the generated weight vector.
    weights = parse_megam_weights(stdout, encoding.length(), explicit)

    # Convert from base-e to base-2 weights.
    weights *= numpy.log2(numpy.e)

    # Build the classifier
    return MaxentClassifier(encoding, weights)


######################################################################
# { Classifier Trainer: tadm
######################################################################


class TadmMaxentClassifier(MaxentClassifier):
    @classmethod
    def train(cls, train_toks, **kwargs):
        algorithm = kwargs.get("algorithm", "tao_lmvm")
        trace = kwargs.get("trace", 3)
        encoding = kwargs.get("encoding", None)
        labels = kwargs.get("labels", None)
        sigma = kwargs.get("gaussian_prior_sigma", 0)
        count_cutoff = kwargs.get("count_cutoff", 0)
        max_iter = kwargs.get("max_iter")
        ll_delta = kwargs.get("min_lldelta")

        # Construct an encoding from the training data.
        if not encoding:
            encoding = TadmEventMaxentFeatureEncoding.train(
                train_toks, count_cutoff, labels=labels
            )

        trainfile_fd, trainfile_name = tempfile.mkstemp(
            prefix="nltk-tadm-events-", suffix=".gz"
        )
        weightfile_fd, weightfile_name = tempfile.mkstemp(prefix="nltk-tadm-weights-")

        trainfile = gzip_open_unicode(trainfile_name, "w")
        write_tadm_file(train_toks, encoding, trainfile)
        trainfile.close()

        options = []
        options.extend(["-monitor"])
        options.extend(["-method", algorithm])
        if sigma:
            options.extend(["-l2", "%.6f" % sigma**2])
        if max_iter:
            options.extend(["-max_it", "%d" % max_iter])
        if ll_delta:
            options.extend(["-fatol", "%.6f" % abs(ll_delta)])
        options.extend(["-events_in", trainfile_name])
        options.extend(["-params_out", weightfile_name])
        if trace < 3:
            options.extend(["2>&1"])
        else:
            options.extend(["-summary"])

        call_tadm(options)

        with open(weightfile_name) as weightfile:
            weights = parse_tadm_weights(weightfile)

        os.remove(trainfile_name)
        os.remove(weightfile_name)

        # Convert from base-e to base-2 weights.
        weights *= numpy.log2(numpy.e)

        # Build the classifier
        return cls(encoding, weights)


######################################################################
# { Demo
######################################################################
def demo():
    from nltk.classify.util import names_demo

    classifier = names_demo(MaxentClassifier.train)


if __name__ == "__main__":
    demo()
