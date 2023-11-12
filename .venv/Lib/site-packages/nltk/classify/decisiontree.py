# Natural Language Toolkit: Decision Tree Classifiers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A classifier model that decides which label to assign to a token on
the basis of a tree structure, where branches correspond to conditions
on feature values, and leaves correspond to label assignments.
"""

from collections import defaultdict

from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy


class DecisionTreeClassifier(ClassifierI):
    def __init__(self, label, feature_name=None, decisions=None, default=None):
        """
        :param label: The most likely label for tokens that reach
            this node in the decision tree.  If this decision tree
            has no children, then this label will be assigned to
            any token that reaches this decision tree.
        :param feature_name: The name of the feature that this
            decision tree selects for.
        :param decisions: A dictionary mapping from feature values
            for the feature identified by ``feature_name`` to
            child decision trees.
        :param default: The child that will be used if the value of
            feature ``feature_name`` does not match any of the keys in
            ``decisions``.  This is used when constructing binary
            decision trees.
        """
        self._label = label
        self._fname = feature_name
        self._decisions = decisions
        self._default = default

    def labels(self):
        labels = [self._label]
        if self._decisions is not None:
            for dt in self._decisions.values():
                labels.extend(dt.labels())
        if self._default is not None:
            labels.extend(self._default.labels())
        return list(set(labels))

    def classify(self, featureset):
        # Decision leaf:
        if self._fname is None:
            return self._label

        # Decision tree:
        fval = featureset.get(self._fname)
        if fval in self._decisions:
            return self._decisions[fval].classify(featureset)
        elif self._default is not None:
            return self._default.classify(featureset)
        else:
            return self._label

    def error(self, labeled_featuresets):
        errors = 0
        for featureset, label in labeled_featuresets:
            if self.classify(featureset) != label:
                errors += 1
        return errors / len(labeled_featuresets)

    def pretty_format(self, width=70, prefix="", depth=4):
        """
        Return a string containing a pretty-printed version of this
        decision tree.  Each line in this string corresponds to a
        single decision tree node or leaf, and indentation is used to
        display the structure of the decision tree.
        """
        # [xx] display default!!
        if self._fname is None:
            n = width - len(prefix) - 15
            return "{}{} {}\n".format(prefix, "." * n, self._label)
        s = ""
        for i, (fval, result) in enumerate(
            sorted(
                self._decisions.items(),
                key=lambda item: (item[0] in [None, False, True], str(item[0]).lower()),
            )
        ):
            hdr = f"{prefix}{self._fname}={fval}? "
            n = width - 15 - len(hdr)
            s += "{}{} {}\n".format(hdr, "." * (n), result._label)
            if result._fname is not None and depth > 1:
                s += result.pretty_format(width, prefix + "  ", depth - 1)
        if self._default is not None:
            n = width - len(prefix) - 21
            s += "{}else: {} {}\n".format(prefix, "." * n, self._default._label)
            if self._default._fname is not None and depth > 1:
                s += self._default.pretty_format(width, prefix + "  ", depth - 1)
        return s

    def pseudocode(self, prefix="", depth=4):
        """
        Return a string representation of this decision tree that
        expresses the decisions it makes as a nested set of pseudocode
        if statements.
        """
        if self._fname is None:
            return f"{prefix}return {self._label!r}\n"
        s = ""
        for (fval, result) in sorted(
            self._decisions.items(),
            key=lambda item: (item[0] in [None, False, True], str(item[0]).lower()),
        ):
            s += f"{prefix}if {self._fname} == {fval!r}: "
            if result._fname is not None and depth > 1:
                s += "\n" + result.pseudocode(prefix + "  ", depth - 1)
            else:
                s += f"return {result._label!r}\n"
        if self._default is not None:
            if len(self._decisions) == 1:
                s += "{}if {} != {!r}: ".format(
                    prefix, self._fname, list(self._decisions.keys())[0]
                )
            else:
                s += f"{prefix}else: "
            if self._default._fname is not None and depth > 1:
                s += "\n" + self._default.pseudocode(prefix + "  ", depth - 1)
            else:
                s += f"return {self._default._label!r}\n"
        return s

    def __str__(self):
        return self.pretty_format()

    @staticmethod
    def train(
        labeled_featuresets,
        entropy_cutoff=0.05,
        depth_cutoff=100,
        support_cutoff=10,
        binary=False,
        feature_values=None,
        verbose=False,
    ):
        """
        :param binary: If true, then treat all feature/value pairs as
            individual binary features, rather than using a single n-way
            branch for each feature.
        """
        # Collect a list of all feature names.
        feature_names = set()
        for featureset, label in labeled_featuresets:
            for fname in featureset:
                feature_names.add(fname)

        # Collect a list of the values each feature can take.
        if feature_values is None and binary:
            feature_values = defaultdict(set)
            for featureset, label in labeled_featuresets:
                for fname, fval in featureset.items():
                    feature_values[fname].add(fval)

        # Start with a stump.
        if not binary:
            tree = DecisionTreeClassifier.best_stump(
                feature_names, labeled_featuresets, verbose
            )
        else:
            tree = DecisionTreeClassifier.best_binary_stump(
                feature_names, labeled_featuresets, feature_values, verbose
            )

        # Refine the stump.
        tree.refine(
            labeled_featuresets,
            entropy_cutoff,
            depth_cutoff - 1,
            support_cutoff,
            binary,
            feature_values,
            verbose,
        )

        # Return it
        return tree

    @staticmethod
    def leaf(labeled_featuresets):
        label = FreqDist(label for (featureset, label) in labeled_featuresets).max()
        return DecisionTreeClassifier(label)

    @staticmethod
    def stump(feature_name, labeled_featuresets):
        label = FreqDist(label for (featureset, label) in labeled_featuresets).max()

        # Find the best label for each value.
        freqs = defaultdict(FreqDist)  # freq(label|value)
        for featureset, label in labeled_featuresets:
            feature_value = featureset.get(feature_name)
            freqs[feature_value][label] += 1

        decisions = {val: DecisionTreeClassifier(freqs[val].max()) for val in freqs}
        return DecisionTreeClassifier(label, feature_name, decisions)

    def refine(
        self,
        labeled_featuresets,
        entropy_cutoff,
        depth_cutoff,
        support_cutoff,
        binary=False,
        feature_values=None,
        verbose=False,
    ):
        if len(labeled_featuresets) <= support_cutoff:
            return
        if self._fname is None:
            return
        if depth_cutoff <= 0:
            return
        for fval in self._decisions:
            fval_featuresets = [
                (featureset, label)
                for (featureset, label) in labeled_featuresets
                if featureset.get(self._fname) == fval
            ]

            label_freqs = FreqDist(label for (featureset, label) in fval_featuresets)
            if entropy(MLEProbDist(label_freqs)) > entropy_cutoff:
                self._decisions[fval] = DecisionTreeClassifier.train(
                    fval_featuresets,
                    entropy_cutoff,
                    depth_cutoff,
                    support_cutoff,
                    binary,
                    feature_values,
                    verbose,
                )
        if self._default is not None:
            default_featuresets = [
                (featureset, label)
                for (featureset, label) in labeled_featuresets
                if featureset.get(self._fname) not in self._decisions
            ]
            label_freqs = FreqDist(label for (featureset, label) in default_featuresets)
            if entropy(MLEProbDist(label_freqs)) > entropy_cutoff:
                self._default = DecisionTreeClassifier.train(
                    default_featuresets,
                    entropy_cutoff,
                    depth_cutoff,
                    support_cutoff,
                    binary,
                    feature_values,
                    verbose,
                )

    @staticmethod
    def best_stump(feature_names, labeled_featuresets, verbose=False):
        best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
        best_error = best_stump.error(labeled_featuresets)
        for fname in feature_names:
            stump = DecisionTreeClassifier.stump(fname, labeled_featuresets)
            stump_error = stump.error(labeled_featuresets)
            if stump_error < best_error:
                best_error = stump_error
                best_stump = stump
        if verbose:
            print(
                "best stump for {:6d} toks uses {:20} err={:6.4f}".format(
                    len(labeled_featuresets), best_stump._fname, best_error
                )
            )
        return best_stump

    @staticmethod
    def binary_stump(feature_name, feature_value, labeled_featuresets):
        label = FreqDist(label for (featureset, label) in labeled_featuresets).max()

        # Find the best label for each value.
        pos_fdist = FreqDist()
        neg_fdist = FreqDist()
        for featureset, label in labeled_featuresets:
            if featureset.get(feature_name) == feature_value:
                pos_fdist[label] += 1
            else:
                neg_fdist[label] += 1

        decisions = {}
        default = label
        # But hopefully we have observations!
        if pos_fdist.N() > 0:
            decisions = {feature_value: DecisionTreeClassifier(pos_fdist.max())}
        if neg_fdist.N() > 0:
            default = DecisionTreeClassifier(neg_fdist.max())

        return DecisionTreeClassifier(label, feature_name, decisions, default)

    @staticmethod
    def best_binary_stump(
        feature_names, labeled_featuresets, feature_values, verbose=False
    ):
        best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
        best_error = best_stump.error(labeled_featuresets)
        for fname in feature_names:
            for fval in feature_values[fname]:
                stump = DecisionTreeClassifier.binary_stump(
                    fname, fval, labeled_featuresets
                )
                stump_error = stump.error(labeled_featuresets)
                if stump_error < best_error:
                    best_error = stump_error
                    best_stump = stump
        if verbose:
            if best_stump._decisions:
                descr = "{}={}".format(
                    best_stump._fname, list(best_stump._decisions.keys())[0]
                )
            else:
                descr = "(default)"
            print(
                "best stump for {:6d} toks uses {:20} err={:6.4f}".format(
                    len(labeled_featuresets), descr, best_error
                )
            )
        return best_stump


##//////////////////////////////////////////////////////
##  Demo
##//////////////////////////////////////////////////////


def f(x):
    return DecisionTreeClassifier.train(x, binary=True, verbose=True)


def demo():
    from nltk.classify.util import binary_names_demo_features, names_demo

    classifier = names_demo(
        f, binary_names_demo_features  # DecisionTreeClassifier.train,
    )
    print(classifier.pretty_format(depth=7))
    print(classifier.pseudocode(depth=7))


if __name__ == "__main__":
    demo()
