# -*- coding: utf-8 -*-
'''This file is based on pattern.en. See the bundled NOTICE file for
license information.
'''
from __future__ import absolute_import
import os

from textblob._text import (Parser as _Parser, Sentiment as _Sentiment, Lexicon,
    WORD, POS, CHUNK, PNP, PENN, UNIVERSAL, Spelling)

from textblob.compat import text_type, unicode

try:
    MODULE = os.path.dirname(os.path.abspath(__file__))
except:
    MODULE = ""

spelling = Spelling(
        path = os.path.join(MODULE, "en-spelling.txt")
)

#--- ENGLISH PARSER --------------------------------------------------------------------------------

def find_lemmata(tokens):
    """ Annotates the tokens with lemmata for plural nouns and conjugated verbs,
        where each token is a [word, part-of-speech] list.
    """
    for token in tokens:
        word, pos, lemma = token[0], token[1], token[0]
        # cats => cat
        if pos == "NNS":
            lemma = singularize(word)
        # sat => sit
        if pos.startswith(("VB", "MD")):
            lemma = conjugate(word, INFINITIVE) or word
        token.append(lemma.lower())
    return tokens

class Parser(_Parser):

    def find_lemmata(self, tokens, **kwargs):
        return find_lemmata(tokens)

    def find_tags(self, tokens, **kwargs):
        if kwargs.get("tagset") in (PENN, None):
            kwargs.setdefault("map", lambda token, tag: (token, tag))
        if kwargs.get("tagset") == UNIVERSAL:
            kwargs.setdefault("map", lambda token, tag: penntreebank2universal(token, tag))
        return _Parser.find_tags(self, tokens, **kwargs)

class Sentiment(_Sentiment):

    def load(self, path=None):
        _Sentiment.load(self, path)
        # Map "terrible" to adverb "terribly" (+1% accuracy)
        if not path:
            for w, pos in list(dict.items(self)):
                if "JJ" in pos:
                    if w.endswith("y"):
                        w = w[:-1] + "i"
                    if w.endswith("le"):
                        w = w[:-2]
                    p, s, i = pos["JJ"]
                    self.annotate(w + "ly", "RB", p, s, i)


lexicon = Lexicon(
        path = os.path.join(MODULE, "en-lexicon.txt"),
  morphology = os.path.join(MODULE, "en-morphology.txt"),
     context = os.path.join(MODULE, "en-context.txt"),
    entities = os.path.join(MODULE, "en-entities.txt"),
    language = "en"
)
parser = Parser(
     lexicon = lexicon,
     default = ("NN", "NNP", "CD"),
    language = "en"
)

sentiment = Sentiment(
        path = os.path.join(MODULE, "en-sentiment.xml"),
      synset = "wordnet_id",
   negations = ("no", "not", "n't", "never"),
   modifiers = ("RB",),
   modifier  = lambda w: w.endswith("ly"),
   tokenizer = parser.find_tokens,
    language = "en"
)


def tokenize(s, *args, **kwargs):
    """ Returns a list of sentences, where punctuation marks have been split from words.
    """
    return parser.find_tokens(text_type(s), *args, **kwargs)

def parse(s, *args, **kwargs):
    """ Returns a tagged Unicode string.
    """
    return parser.parse(unicode(s), *args, **kwargs)

def parsetree(s, *args, **kwargs):
    """ Returns a parsed Text from the given string.
    """
    return Text(parse(unicode(s), *args, **kwargs))

def split(s, token=[WORD, POS, CHUNK, PNP]):
    """ Returns a parsed Text from the given parsed string.
    """
    return Text(text_type(s), token)

def tag(s, tokenize=True, encoding="utf-8"):
    """ Returns a list of (token, tag)-tuples from the given string.
    """
    tags = []
    for sentence in parse(s, tokenize, True, False, False, False, encoding).split():
        for token in sentence:
            tags.append((token[0], token[1]))
    return tags

def suggest(w):
    """ Returns a list of (word, confidence)-tuples of spelling corrections.
    """
    return spelling.suggest(w)

def polarity(s, **kwargs):
    """ Returns the sentence polarity (positive/negative) between -1.0 and 1.0.
    """
    return sentiment(unicode(s), **kwargs)[0]

def subjectivity(s, **kwargs):
    """ Returns the sentence subjectivity (objective/subjective) between 0.0 and 1.0.
    """
    return sentiment(unicode(s), **kwargs)[1]

def positive(s, threshold=0.1, **kwargs):
    """ Returns True if the given sentence has a positive sentiment (polarity >= threshold).
    """
    return polarity(unicode(s), **kwargs) >= threshold

