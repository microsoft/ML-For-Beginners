# Multi-Word Expression tokenizer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Rob Malouf <rmalouf@mail.sdsu.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Multi-Word Expression Tokenizer

A ``MWETokenizer`` takes a string which has already been divided into tokens and
retokenizes it, merging multi-word expressions into single tokens, using a lexicon
of MWEs:


    >>> from nltk.tokenize import MWETokenizer

    >>> tokenizer = MWETokenizer([('a', 'little'), ('a', 'little', 'bit'), ('a', 'lot')])
    >>> tokenizer.add_mwe(('in', 'spite', 'of'))

    >>> tokenizer.tokenize('Testing testing testing one two three'.split())
    ['Testing', 'testing', 'testing', 'one', 'two', 'three']

    >>> tokenizer.tokenize('This is a test in spite'.split())
    ['This', 'is', 'a', 'test', 'in', 'spite']

    >>> tokenizer.tokenize('In a little or a little bit or a lot in spite of'.split())
    ['In', 'a_little', 'or', 'a_little_bit', 'or', 'a_lot', 'in_spite_of']

"""
from nltk.tokenize.api import TokenizerI
from nltk.util import Trie


class MWETokenizer(TokenizerI):
    """A tokenizer that processes tokenized text and merges multi-word expressions
    into single tokens.
    """

    def __init__(self, mwes=None, separator="_"):
        """Initialize the multi-word tokenizer with a list of expressions and a
        separator

        :type mwes: list(list(str))
        :param mwes: A sequence of multi-word expressions to be merged, where
            each MWE is a sequence of strings.
        :type separator: str
        :param separator: String that should be inserted between words in a multi-word
            expression token. (Default is '_')

        """
        if not mwes:
            mwes = []
        self._mwes = Trie(mwes)
        self._separator = separator

    def add_mwe(self, mwe):
        """Add a multi-word expression to the lexicon (stored as a word trie)

        We use ``util.Trie`` to represent the trie. Its form is a dict of dicts.
        The key True marks the end of a valid MWE.

        :param mwe: The multi-word expression we're adding into the word trie
        :type mwe: tuple(str) or list(str)

        :Example:

        >>> tokenizer = MWETokenizer()
        >>> tokenizer.add_mwe(('a', 'b'))
        >>> tokenizer.add_mwe(('a', 'b', 'c'))
        >>> tokenizer.add_mwe(('a', 'x'))
        >>> expected = {'a': {'x': {True: None}, 'b': {True: None, 'c': {True: None}}}}
        >>> tokenizer._mwes == expected
        True

        """
        self._mwes.insert(mwe)

    def tokenize(self, text):
        """

        :param text: A list containing tokenized text
        :type text: list(str)
        :return: A list of the tokenized text with multi-words merged together
        :rtype: list(str)

        :Example:

        >>> tokenizer = MWETokenizer([('hors', "d'oeuvre")], separator='+')
        >>> tokenizer.tokenize("An hors d'oeuvre tonight, sir?".split())
        ['An', "hors+d'oeuvre", 'tonight,', 'sir?']

        """
        i = 0
        n = len(text)
        result = []

        while i < n:
            if text[i] in self._mwes:
                # possible MWE match
                j = i
                trie = self._mwes
                last_match = -1
                while j < n and text[j] in trie:  # and len(trie[text[j]]) > 0 :
                    trie = trie[text[j]]
                    j = j + 1
                    if Trie.LEAF in trie:
                        last_match = j
                else:
                    if last_match > -1:
                        j = last_match

                    if Trie.LEAF in trie or last_match > -1:
                        # success!
                        result.append(self._separator.join(text[i:j]))
                        i = j
                    else:
                        # no match, so backtrack
                        result.append(text[i])
                        i += 1
            else:
                result.append(text[i])
                i += 1
        return result
