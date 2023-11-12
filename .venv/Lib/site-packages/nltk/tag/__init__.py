# Natural Language Toolkit: Taggers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
NLTK Taggers

This package contains classes and interfaces for part-of-speech
tagging, or simply "tagging".

A "tag" is a case-sensitive string that specifies some property of a token,
such as its part of speech.  Tagged tokens are encoded as tuples
``(tag, token)``.  For example, the following tagged token combines
the word ``'fly'`` with a noun part of speech tag (``'NN'``):

    >>> tagged_tok = ('fly', 'NN')

An off-the-shelf tagger is available for English. It uses the Penn Treebank tagset:

    >>> from nltk import pos_tag, word_tokenize
    >>> pos_tag(word_tokenize("John's big idea isn't all that bad.")) # doctest: +NORMALIZE_WHITESPACE
    [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
    ("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]

A Russian tagger is also available if you specify lang="rus". It uses
the Russian National Corpus tagset:

    >>> pos_tag(word_tokenize("Илья оторопел и дважды перечитал бумажку."), lang='rus')    # doctest: +SKIP
    [('Илья', 'S'), ('оторопел', 'V'), ('и', 'CONJ'), ('дважды', 'ADV'), ('перечитал', 'V'),
    ('бумажку', 'S'), ('.', 'NONLEX')]

This package defines several taggers, which take a list of tokens,
assign a tag to each one, and return the resulting list of tagged tokens.
Most of the taggers are built automatically based on a training corpus.
For example, the unigram tagger tags each word *w* by checking what
the most frequent tag for *w* was in a training corpus:

    >>> from nltk.corpus import brown
    >>> from nltk.tag import UnigramTagger
    >>> tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
    >>> sent = ['Mitchell', 'decried', 'the', 'high', 'rate', 'of', 'unemployment']
    >>> for word, tag in tagger.tag(sent):
    ...     print(word, '->', tag)
    Mitchell -> NP
    decried -> None
    the -> AT
    high -> JJ
    rate -> NN
    of -> IN
    unemployment -> None

Note that words that the tagger has not seen during training receive a tag
of ``None``.

We evaluate a tagger on data that was not seen during training:

    >>> round(tagger.accuracy(brown.tagged_sents(categories='news')[500:600]), 3)
    0.735

For more information, please consult chapter 5 of the NLTK Book.

isort:skip_file
"""

from nltk.tag.api import TaggerI
from nltk.tag.util import str2tuple, tuple2str, untag
from nltk.tag.sequential import (
    SequentialBackoffTagger,
    ContextTagger,
    DefaultTagger,
    NgramTagger,
    UnigramTagger,
    BigramTagger,
    TrigramTagger,
    AffixTagger,
    RegexpTagger,
    ClassifierBasedTagger,
    ClassifierBasedPOSTagger,
)
from nltk.tag.brill import BrillTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.tnt import TnT
from nltk.tag.hunpos import HunposTagger
from nltk.tag.stanford import StanfordTagger, StanfordPOSTagger, StanfordNERTagger
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.tag.senna import SennaTagger, SennaChunkTagger, SennaNERTagger
from nltk.tag.mapping import tagset_mapping, map_tag
from nltk.tag.crf import CRFTagger
from nltk.tag.perceptron import PerceptronTagger

from nltk.data import load, find

RUS_PICKLE = (
    "taggers/averaged_perceptron_tagger_ru/averaged_perceptron_tagger_ru.pickle"
)


def _get_tagger(lang=None):
    if lang == "rus":
        tagger = PerceptronTagger(False)
        ap_russian_model_loc = "file:" + str(find(RUS_PICKLE))
        tagger.load(ap_russian_model_loc)
    else:
        tagger = PerceptronTagger()
    return tagger


def _pos_tag(tokens, tagset=None, tagger=None, lang=None):
    # Currently only supports English and Russian.
    if lang not in ["eng", "rus"]:
        raise NotImplementedError(
            "Currently, NLTK pos_tag only supports English and Russian "
            "(i.e. lang='eng' or lang='rus')"
        )
    # Throws Error if tokens is of string type
    elif isinstance(tokens, str):
        raise TypeError("tokens: expected a list of strings, got a string")

    else:
        tagged_tokens = tagger.tag(tokens)
        if tagset:  # Maps to the specified tagset.
            if lang == "eng":
                tagged_tokens = [
                    (token, map_tag("en-ptb", tagset, tag))
                    for (token, tag) in tagged_tokens
                ]
            elif lang == "rus":
                # Note that the new Russian pos tags from the model contains suffixes,
                # see https://github.com/nltk/nltk/issues/2151#issuecomment-430709018
                tagged_tokens = [
                    (token, map_tag("ru-rnc-new", tagset, tag.partition("=")[0]))
                    for (token, tag) in tagged_tokens
                ]
        return tagged_tokens


def pos_tag(tokens, tagset=None, lang="eng"):
    """
    Use NLTK's currently recommended part of speech tagger to
    tag the given list of tokens.

        >>> from nltk.tag import pos_tag
        >>> from nltk.tokenize import word_tokenize
        >>> pos_tag(word_tokenize("John's big idea isn't all that bad.")) # doctest: +NORMALIZE_WHITESPACE
        [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
        ("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]
        >>> pos_tag(word_tokenize("John's big idea isn't all that bad."), tagset='universal') # doctest: +NORMALIZE_WHITESPACE
        [('John', 'NOUN'), ("'s", 'PRT'), ('big', 'ADJ'), ('idea', 'NOUN'), ('is', 'VERB'),
        ("n't", 'ADV'), ('all', 'DET'), ('that', 'DET'), ('bad', 'ADJ'), ('.', '.')]

    NB. Use `pos_tag_sents()` for efficient tagging of more than one sentence.

    :param tokens: Sequence of tokens to be tagged
    :type tokens: list(str)
    :param tagset: the tagset to be used, e.g. universal, wsj, brown
    :type tagset: str
    :param lang: the ISO 639 code of the language, e.g. 'eng' for English, 'rus' for Russian
    :type lang: str
    :return: The tagged tokens
    :rtype: list(tuple(str, str))
    """
    tagger = _get_tagger(lang)
    return _pos_tag(tokens, tagset, tagger, lang)


def pos_tag_sents(sentences, tagset=None, lang="eng"):
    """
    Use NLTK's currently recommended part of speech tagger to tag the
    given list of sentences, each consisting of a list of tokens.

    :param sentences: List of sentences to be tagged
    :type sentences: list(list(str))
    :param tagset: the tagset to be used, e.g. universal, wsj, brown
    :type tagset: str
    :param lang: the ISO 639 code of the language, e.g. 'eng' for English, 'rus' for Russian
    :type lang: str
    :return: The list of tagged sentences
    :rtype: list(list(tuple(str, str)))
    """
    tagger = _get_tagger(lang)
    return [_pos_tag(sent, tagset, tagger, lang) for sent in sentences]
