# Natural Language Toolkit: Tagset Mapping
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Nathan Schneider <nathan@cmu.edu>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Interface for converting POS tags from various treebanks
to the universal tagset of Petrov, Das, & McDonald.

The tagset consists of the following 12 coarse tags:

VERB - verbs (all tenses and modes)
NOUN - nouns (common and proper)
PRON - pronouns
ADJ - adjectives
ADV - adverbs
ADP - adpositions (prepositions and postpositions)
CONJ - conjunctions
DET - determiners
NUM - cardinal numbers
PRT - particles or other function words
X - other: foreign words, typos, abbreviations
. - punctuation

@see: https://arxiv.org/abs/1104.2086 and https://code.google.com/p/universal-pos-tags/

"""

from collections import defaultdict
from os.path import join

from nltk.data import load

_UNIVERSAL_DATA = "taggers/universal_tagset"
_UNIVERSAL_TAGS = (
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
)

# _MAPPINGS = defaultdict(lambda: defaultdict(dict))
# the mapping between tagset T1 and T2 returns UNK if applied to an unrecognized tag
_MAPPINGS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "UNK")))


def _load_universal_map(fileid):
    contents = load(join(_UNIVERSAL_DATA, fileid + ".map"), format="text")

    # When mapping to the Universal Tagset,
    # map unknown inputs to 'X' not 'UNK'
    _MAPPINGS[fileid]["universal"].default_factory = lambda: "X"

    for line in contents.splitlines():
        line = line.strip()
        if line == "":
            continue
        fine, coarse = line.split("\t")

        assert coarse in _UNIVERSAL_TAGS, f"Unexpected coarse tag: {coarse}"
        assert (
            fine not in _MAPPINGS[fileid]["universal"]
        ), f"Multiple entries for original tag: {fine}"

        _MAPPINGS[fileid]["universal"][fine] = coarse


def tagset_mapping(source, target):
    """
    Retrieve the mapping dictionary between tagsets.

    >>> tagset_mapping('ru-rnc', 'universal') == {'!': '.', 'A': 'ADJ', 'C': 'CONJ', 'AD': 'ADV',\
            'NN': 'NOUN', 'VG': 'VERB', 'COMP': 'CONJ', 'NC': 'NUM', 'VP': 'VERB', 'P': 'ADP',\
            'IJ': 'X', 'V': 'VERB', 'Z': 'X', 'VI': 'VERB', 'YES_NO_SENT': 'X', 'PTCL': 'PRT'}
    True
    """

    if source not in _MAPPINGS or target not in _MAPPINGS[source]:
        if target == "universal":
            _load_universal_map(source)
            # Added the new Russian National Corpus mappings because the
            # Russian model for nltk.pos_tag() uses it.
            _MAPPINGS["ru-rnc-new"]["universal"] = {
                "A": "ADJ",
                "A-PRO": "PRON",
                "ADV": "ADV",
                "ADV-PRO": "PRON",
                "ANUM": "ADJ",
                "CONJ": "CONJ",
                "INTJ": "X",
                "NONLEX": ".",
                "NUM": "NUM",
                "PARENTH": "PRT",
                "PART": "PRT",
                "PR": "ADP",
                "PRAEDIC": "PRT",
                "PRAEDIC-PRO": "PRON",
                "S": "NOUN",
                "S-PRO": "PRON",
                "V": "VERB",
            }

    return _MAPPINGS[source][target]


def map_tag(source, target, source_tag):
    """
    Maps the tag from the source tagset to the target tagset.

    >>> map_tag('en-ptb', 'universal', 'VBZ')
    'VERB'
    >>> map_tag('en-ptb', 'universal', 'VBP')
    'VERB'
    >>> map_tag('en-ptb', 'universal', '``')
    '.'
    """

    # we need a systematic approach to naming
    if target == "universal":
        if source == "wsj":
            source = "en-ptb"
        if source == "brown":
            source = "en-brown"

    return tagset_mapping(source, target)[source_tag]
