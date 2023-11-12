# Natural Language Toolkit: Minimal Sets
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from collections import defaultdict


class MinimalSet:
    """
    Find contexts where more than one possible target value can
    appear.  E.g. if targets are word-initial letters, and contexts
    are the remainders of words, then we would like to find cases like
    "fat" vs "cat", and "training" vs "draining".  If targets are
    parts-of-speech and contexts are words, then we would like to find
    cases like wind (noun) 'air in rapid motion', vs wind (verb)
    'coil, wrap'.
    """

    def __init__(self, parameters=None):
        """
        Create a new minimal set.

        :param parameters: The (context, target, display) tuples for the item
        :type parameters: list(tuple(str, str, str))
        """
        self._targets = set()  # the contrastive information
        self._contexts = set()  # what we are controlling for
        self._seen = defaultdict(set)  # to record what we have seen
        self._displays = {}  # what we will display

        if parameters:
            for context, target, display in parameters:
                self.add(context, target, display)

    def add(self, context, target, display):
        """
        Add a new item to the minimal set, having the specified
        context, target, and display form.

        :param context: The context in which the item of interest appears
        :type context: str
        :param target: The item of interest
        :type target: str
        :param display: The information to be reported for each item
        :type display: str
        """
        # Store the set of targets that occurred in this context
        self._seen[context].add(target)

        # Keep track of which contexts and targets we have seen
        self._contexts.add(context)
        self._targets.add(target)

        # For a given context and target, store the display form
        self._displays[(context, target)] = display

    def contexts(self, minimum=2):
        """
        Determine which contexts occurred with enough distinct targets.

        :param minimum: the minimum number of distinct target forms
        :type minimum: int
        :rtype: list
        """
        return [c for c in self._contexts if len(self._seen[c]) >= minimum]

    def display(self, context, target, default=""):
        if (context, target) in self._displays:
            return self._displays[(context, target)]
        else:
            return default

    def display_all(self, context):
        result = []
        for target in self._targets:
            x = self.display(context, target)
            if x:
                result.append(x)
        return result

    def targets(self):
        return self._targets
