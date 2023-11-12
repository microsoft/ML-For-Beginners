# Natural Language Toolkit: NPS Chat Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re
import textwrap

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation


class NPSChatCorpusReader(XMLCorpusReader):
    def __init__(self, root, fileids, wrap_etree=False, tagset=None):
        XMLCorpusReader.__init__(self, root, fileids, wrap_etree)
        self._tagset = tagset

    def xml_posts(self, fileids=None):
        if self._wrap_etree:
            return concat(
                [
                    XMLCorpusView(fileid, "Session/Posts/Post", self._wrap_elt)
                    for fileid in self.abspaths(fileids)
                ]
            )
        else:
            return concat(
                [
                    XMLCorpusView(fileid, "Session/Posts/Post")
                    for fileid in self.abspaths(fileids)
                ]
            )

    def posts(self, fileids=None):
        return concat(
            [
                XMLCorpusView(
                    fileid, "Session/Posts/Post/terminals", self._elt_to_words
                )
                for fileid in self.abspaths(fileids)
            ]
        )

    def tagged_posts(self, fileids=None, tagset=None):
        def reader(elt, handler):
            return self._elt_to_tagged_words(elt, handler, tagset)

        return concat(
            [
                XMLCorpusView(fileid, "Session/Posts/Post/terminals", reader)
                for fileid in self.abspaths(fileids)
            ]
        )

    def words(self, fileids=None):
        return LazyConcatenation(self.posts(fileids))

    def tagged_words(self, fileids=None, tagset=None):
        return LazyConcatenation(self.tagged_posts(fileids, tagset))

    def _wrap_elt(self, elt, handler):
        return ElementWrapper(elt)

    def _elt_to_words(self, elt, handler):
        return [self._simplify_username(t.attrib["word"]) for t in elt.findall("t")]

    def _elt_to_tagged_words(self, elt, handler, tagset=None):
        tagged_post = [
            (self._simplify_username(t.attrib["word"]), t.attrib["pos"])
            for t in elt.findall("t")
        ]
        if tagset and tagset != self._tagset:
            tagged_post = [
                (w, map_tag(self._tagset, tagset, t)) for (w, t) in tagged_post
            ]
        return tagged_post

    @staticmethod
    def _simplify_username(word):
        if "User" in word:
            word = "U" + word.split("User", 1)[1]
        elif isinstance(word, bytes):
            word = word.decode("ascii")
        return word
