# Natural Language Toolkit: SentiWordNet
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Christopher Potts <cgpotts@stanford.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
An NLTK interface for SentiWordNet

SentiWordNet is a lexical resource for opinion mining.
SentiWordNet assigns to each synset of WordNet three
sentiment scores: positivity, negativity, and objectivity.

For details about SentiWordNet see:
http://sentiwordnet.isti.cnr.it/

    >>> from nltk.corpus import sentiwordnet as swn
    >>> print(swn.senti_synset('breakdown.n.03'))
    <breakdown.n.03: PosScore=0.0 NegScore=0.25>
    >>> list(swn.senti_synsets('slow'))
    [SentiSynset('decelerate.v.01'), SentiSynset('slow.v.02'),\
 SentiSynset('slow.v.03'), SentiSynset('slow.a.01'),\
 SentiSynset('slow.a.02'), SentiSynset('dense.s.04'),\
 SentiSynset('slow.a.04'), SentiSynset('boring.s.01'),\
 SentiSynset('dull.s.08'), SentiSynset('slowly.r.01'),\
 SentiSynset('behind.r.03')]
    >>> happy = swn.senti_synsets('happy', 'a')
    >>> happy0 = list(happy)[0]
    >>> happy0.pos_score()
    0.875
    >>> happy0.neg_score()
    0.0
    >>> happy0.obj_score()
    0.125
"""

import re

from nltk.corpus.reader import CorpusReader


class SentiWordNetCorpusReader(CorpusReader):
    def __init__(self, root, fileids, encoding="utf-8"):
        """
        Construct a new SentiWordNet Corpus Reader, using data from
        the specified file.
        """
        super().__init__(root, fileids, encoding=encoding)
        if len(self._fileids) != 1:
            raise ValueError("Exactly one file must be specified")
        self._db = {}
        self._parse_src_file()

    def _parse_src_file(self):
        lines = self.open(self._fileids[0]).read().splitlines()
        lines = filter((lambda x: not re.search(r"^\s*#", x)), lines)
        for i, line in enumerate(lines):
            fields = [field.strip() for field in re.split(r"\t+", line)]
            try:
                pos, offset, pos_score, neg_score, synset_terms, gloss = fields
            except BaseException as e:
                raise ValueError(f"Line {i} formatted incorrectly: {line}\n") from e
            if pos and offset:
                offset = int(offset)
                self._db[(pos, offset)] = (float(pos_score), float(neg_score))

    def senti_synset(self, *vals):
        from nltk.corpus import wordnet as wn

        if tuple(vals) in self._db:
            pos_score, neg_score = self._db[tuple(vals)]
            pos, offset = vals
            if pos == "s":
                pos = "a"
            synset = wn.synset_from_pos_and_offset(pos, offset)
            return SentiSynset(pos_score, neg_score, synset)
        else:
            synset = wn.synset(vals[0])
            pos = synset.pos()
            if pos == "s":
                pos = "a"
            offset = synset.offset()
            if (pos, offset) in self._db:
                pos_score, neg_score = self._db[(pos, offset)]
                return SentiSynset(pos_score, neg_score, synset)
            else:
                return None

    def senti_synsets(self, string, pos=None):
        from nltk.corpus import wordnet as wn

        sentis = []
        synset_list = wn.synsets(string, pos)
        for synset in synset_list:
            sentis.append(self.senti_synset(synset.name()))
        sentis = filter(lambda x: x, sentis)
        return sentis

    def all_senti_synsets(self):
        from nltk.corpus import wordnet as wn

        for key, fields in self._db.items():
            pos, offset = key
            pos_score, neg_score = fields
            synset = wn.synset_from_pos_and_offset(pos, offset)
            yield SentiSynset(pos_score, neg_score, synset)


class SentiSynset:
    def __init__(self, pos_score, neg_score, synset):
        self._pos_score = pos_score
        self._neg_score = neg_score
        self._obj_score = 1.0 - (self._pos_score + self._neg_score)
        self.synset = synset

    def pos_score(self):
        return self._pos_score

    def neg_score(self):
        return self._neg_score

    def obj_score(self):
        return self._obj_score

    def __str__(self):
        """Prints just the Pos/Neg scores for now."""
        s = "<"
        s += self.synset.name() + ": "
        s += "PosScore=%s " % self._pos_score
        s += "NegScore=%s" % self._neg_score
        s += ">"
        return s

    def __repr__(self):
        return "Senti" + repr(self.synset)
