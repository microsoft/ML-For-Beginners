# Natural Language Toolkit: Aligned Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# Author: Steven Bird <stevenbird1@gmail.com>
# For license information, see LICENSE.TXT

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import (
    StreamBackedCorpusView,
    concat,
    read_alignedsent_block,
)
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.translate import AlignedSent, Alignment


class AlignedCorpusReader(CorpusReader):
    """
    Reader for corpora of word-aligned sentences.  Tokens are assumed
    to be separated by whitespace.  Sentences begin on separate lines.
    """

    def __init__(
        self,
        root,
        fileids,
        sep="/",
        word_tokenizer=WhitespaceTokenizer(),
        sent_tokenizer=RegexpTokenizer("\n", gaps=True),
        alignedsent_block_reader=read_alignedsent_block,
        encoding="latin1",
    ):
        """
        Construct a new Aligned Corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/...path to corpus.../'
            >>> reader = AlignedCorpusReader(root, '.*', '.txt') # doctest: +SKIP

        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._sep = sep
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._alignedsent_block_reader = alignedsent_block_reader

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                AlignedSentCorpusView(
                    fileid,
                    enc,
                    False,
                    False,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._alignedsent_block_reader,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        return concat(
            [
                AlignedSentCorpusView(
                    fileid,
                    enc,
                    False,
                    True,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._alignedsent_block_reader,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def aligned_sents(self, fileids=None):
        """
        :return: the given file(s) as a list of AlignedSent objects.
        :rtype: list(AlignedSent)
        """
        return concat(
            [
                AlignedSentCorpusView(
                    fileid,
                    enc,
                    True,
                    True,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._alignedsent_block_reader,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )


class AlignedSentCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for aligned sentences.
    ``AlignedSentCorpusView`` objects are typically created by
    ``AlignedCorpusReader`` (not directly by nltk users).
    """

    def __init__(
        self,
        corpus_file,
        encoding,
        aligned,
        group_by_sent,
        word_tokenizer,
        sent_tokenizer,
        alignedsent_block_reader,
    ):
        self._aligned = aligned
        self._group_by_sent = group_by_sent
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._alignedsent_block_reader = alignedsent_block_reader
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        block = [
            self._word_tokenizer.tokenize(sent_str)
            for alignedsent_str in self._alignedsent_block_reader(stream)
            for sent_str in self._sent_tokenizer.tokenize(alignedsent_str)
        ]
        if self._aligned:
            block[2] = Alignment.fromstring(
                " ".join(block[2])
            )  # kludge; we shouldn't have tokenized the alignment string
            block = [AlignedSent(*block)]
        elif self._group_by_sent:
            block = [block[0]]
        else:
            block = block[0]

        return block
