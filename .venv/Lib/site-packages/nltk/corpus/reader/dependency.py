# Natural Language Toolkit: Dependency Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Kepa Sarasola <kepa.sarasola@ehu.es>
#         Iker Manterola <returntothehangar@hotmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.parse import DependencyGraph
from nltk.tokenize import *


class DependencyCorpusReader(SyntaxCorpusReader):
    def __init__(
        self,
        root,
        fileids,
        encoding="utf8",
        word_tokenizer=TabTokenizer(),
        sent_tokenizer=RegexpTokenizer("\n", gaps=True),
        para_block_reader=read_blankline_block,
    ):
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)

    #########################################################

    def words(self, fileids=None):
        return concat(
            [
                DependencyCorpusView(fileid, False, False, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def tagged_words(self, fileids=None):
        return concat(
            [
                DependencyCorpusView(fileid, True, False, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def sents(self, fileids=None):
        return concat(
            [
                DependencyCorpusView(fileid, False, True, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def tagged_sents(self, fileids=None):
        return concat(
            [
                DependencyCorpusView(fileid, True, True, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def parsed_sents(self, fileids=None):
        sents = concat(
            [
                DependencyCorpusView(fileid, False, True, True, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )
        return [DependencyGraph(sent) for sent in sents]


class DependencyCorpusView(StreamBackedCorpusView):
    _DOCSTART = "-DOCSTART- -DOCSTART- O\n"  # dokumentu hasiera definitzen da

    def __init__(
        self,
        corpus_file,
        tagged,
        group_by_sent,
        dependencies,
        chunk_types=None,
        encoding="utf8",
    ):
        self._tagged = tagged
        self._dependencies = dependencies
        self._group_by_sent = group_by_sent
        self._chunk_types = chunk_types
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        # Read the next sentence.
        sent = read_blankline_block(stream)[0].strip()
        # Strip off the docstart marker, if present.
        if sent.startswith(self._DOCSTART):
            sent = sent[len(self._DOCSTART) :].lstrip()

        # extract word and tag from any of the formats
        if not self._dependencies:
            lines = [line.split("\t") for line in sent.split("\n")]
            if len(lines[0]) == 3 or len(lines[0]) == 4:
                sent = [(line[0], line[1]) for line in lines]
            elif len(lines[0]) == 10:
                sent = [(line[1], line[4]) for line in lines]
            else:
                raise ValueError("Unexpected number of fields in dependency tree file")

            # discard tags if they weren't requested
            if not self._tagged:
                sent = [word for (word, tag) in sent]

        # Return the result.
        if self._group_by_sent:
            return [sent]
        else:
            return list(sent)
