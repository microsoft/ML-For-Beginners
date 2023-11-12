# -*- coding: utf-8 -*-
"""Wordnet interface. Contains classes for creating Synsets and Lemmas
directly.

.. versionadded:: 0.7.0

"""
import nltk

#: wordnet module from nltk
wordnet = nltk.corpus.wordnet
#: Synset constructor
Synset = nltk.corpus.wordnet.synset
#: Lemma constructor
Lemma = nltk.corpus.wordnet.lemma
# Part of speech constants
VERB, NOUN, ADJ, ADV = wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV
