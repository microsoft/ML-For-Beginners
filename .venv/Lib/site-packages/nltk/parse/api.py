# Natural Language Toolkit: Parser API
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

import itertools

from nltk.internals import overridden


class ParserI:
    """
    A processing class for deriving trees that represent possible
    structures for a sequence of tokens.  These tree structures are
    known as "parses".  Typically, parsers are used to derive syntax
    trees for sentences.  But parsers can also be used to derive other
    kinds of tree structure, such as morphological trees and discourse
    structures.

    Subclasses must define:
      - at least one of: ``parse()``, ``parse_sents()``.

    Subclasses may define:
      - ``grammar()``
    """

    def grammar(self):
        """
        :return: The grammar used by this parser.
        """
        raise NotImplementedError()

    def parse(self, sent, *args, **kwargs):
        """
        :return: An iterator that generates parse trees for the sentence.
            When possible this list is sorted from most likely to least likely.

        :param sent: The sentence to be parsed
        :type sent: list(str)
        :rtype: iter(Tree)
        """
        if overridden(self.parse_sents):
            return next(self.parse_sents([sent], *args, **kwargs))
        elif overridden(self.parse_one):
            return (
                tree
                for tree in [self.parse_one(sent, *args, **kwargs)]
                if tree is not None
            )
        elif overridden(self.parse_all):
            return iter(self.parse_all(sent, *args, **kwargs))
        else:
            raise NotImplementedError()

    def parse_sents(self, sents, *args, **kwargs):
        """
        Apply ``self.parse()`` to each element of ``sents``.
        :rtype: iter(iter(Tree))
        """
        return (self.parse(sent, *args, **kwargs) for sent in sents)

    def parse_all(self, sent, *args, **kwargs):
        """:rtype: list(Tree)"""
        return list(self.parse(sent, *args, **kwargs))

    def parse_one(self, sent, *args, **kwargs):
        """:rtype: Tree or None"""
        return next(self.parse(sent, *args, **kwargs), None)
