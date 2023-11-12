# -*- coding: utf-8 -*-
'''Various tokenizer implementations.

.. versionadded:: 0.4.0
'''
from __future__ import absolute_import
from itertools import chain

import nltk

from textblob.utils import strip_punc
from textblob.base import BaseTokenizer
from textblob.decorators import requires_nltk_corpus


class WordTokenizer(BaseTokenizer):
    """NLTK's recommended word tokenizer (currently the TreeBankTokenizer).
    Uses regular expressions to tokenize text. Assumes text has already been
    segmented into sentences.

    Performs the following steps:

    * split standard contractions, e.g. don't -> do n't
    * split commas and single quotes
    * separate periods that appear at the end of line
    """

    def tokenize(self, text, include_punc=True):
        '''Return a list of word tokens.

        :param text: string of text.
        :param include_punc: (optional) whether to include punctuation as separate tokens. Default to True.
        '''
        tokens = nltk.tokenize.word_tokenize(text)
        if include_punc:
            return tokens
        else:
            # Return each word token
            # Strips punctuation unless the word comes from a contraction
            # e.g. "Let's" => ["Let", "'s"]
            # e.g. "Can't" => ["Ca", "n't"]
            # e.g. "home." => ['home']
            return [word if word.startswith("'") else strip_punc(word, all=False)
                    for word in tokens if strip_punc(word, all=False)]


class SentenceTokenizer(BaseTokenizer):
    """NLTK's sentence tokenizer (currently PunktSentenceTokenizer).
    Uses an unsupervised algorithm to build a model for abbreviation words,
    collocations, and words that start sentences,
    then uses that to find sentence boundaries.
    """

    @requires_nltk_corpus
    def tokenize(self, text):
        '''Return a list of sentences.'''
        return nltk.tokenize.sent_tokenize(text)


#: Convenience function for tokenizing sentences
sent_tokenize = SentenceTokenizer().itokenize

_word_tokenizer = WordTokenizer()  # Singleton word tokenizer
def word_tokenize(text, include_punc=True, *args, **kwargs):
    """Convenience function for tokenizing text into words.

    NOTE: NLTK's word tokenizer expects sentences as input, so the text will be
    tokenized to sentences before being tokenized to words.
    """
    words = chain.from_iterable(
        _word_tokenizer.itokenize(sentence, include_punc=include_punc,
                                *args, **kwargs)
        for sentence in sent_tokenize(text))
    return words
