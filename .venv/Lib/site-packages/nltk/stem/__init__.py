# Natural Language Toolkit: Stemmers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
#         Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
NLTK Stemmers

Interfaces used to remove morphological affixes from words, leaving
only the word stem.  Stemming algorithms aim to remove those affixes
required for eg. grammatical role, tense, derivational morphology
leaving only the stem of the word.  This is a difficult problem due to
irregular words (eg. common verbs in English), complicated
morphological rules, and part-of-speech and sense ambiguities
(eg. ``ceil-`` is not the stem of ``ceiling``).

StemmerI defines a standard interface for stemmers.
"""

from nltk.stem.api import StemmerI
from nltk.stem.arlstem import ARLSTem
from nltk.stem.arlstem2 import ARLSTem2
from nltk.stem.cistem import Cistem
from nltk.stem.isri import ISRIStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.regexp import RegexpStemmer
from nltk.stem.rslp import RSLPStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
