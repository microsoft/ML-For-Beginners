# Natural Language Toolkit: Plaintext Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
#         Nitin Madnani <nmadnani@umiacs.umd.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A reader for corpora that consist of plaintext documents.
"""

import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *


class PlaintextCorpusReader(CorpusReader):
    """
    Reader for corpora that consist of plaintext documents.  Paragraphs
    are assumed to be split using blank lines.  Sentences and words can
    be tokenized using the default tokenizers, or by custom tokenizers
    specified as parameters to the constructor.

    This corpus reader can be customized (e.g., to skip preface
    sections of specific document formats) by creating a subclass and
    overriding the ``CorpusView`` class variable.
    """

    CorpusView = StreamBackedCorpusView
    """The corpus view class used by this reader.  Subclasses of
       ``PlaintextCorpusReader`` may specify alternative corpus view
       classes (e.g., to skip the preface sections of documents.)"""

    def __init__(
        self,
        root,
        fileids,
        word_tokenizer=WordPunctTokenizer(),
        sent_tokenizer=nltk.data.LazyLoader("tokenizers/punkt/english.pickle"),
        para_block_reader=read_blankline_block,
        encoding="utf8",
    ):
        r"""
        Construct a new plaintext corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/usr/local/share/nltk_data/corpora/webtext/'
            >>> reader = PlaintextCorpusReader(root, '.*\.txt') # doctest: +SKIP

        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param word_tokenizer: Tokenizer for breaking sentences or
            paragraphs into words.
        :param sent_tokenizer: Tokenizer for breaking paragraphs
            into words.
        :param para_block_reader: The block reader used to divide the
            corpus into paragraph blocks.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                self.CorpusView(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        if self._sent_tokenizer is None:
            raise ValueError("No sentence tokenizer for this corpus")

        return concat(
            [
                self.CorpusView(path, self._read_sent_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        if self._sent_tokenizer is None:
            raise ValueError("No sentence tokenizer for this corpus")

        return concat(
            [
                self.CorpusView(path, self._read_para_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            words.extend(self._word_tokenizer.tokenize(stream.readline()))
        return words

    def _read_sent_block(self, stream):
        sents = []
        for para in self._para_block_reader(stream):
            sents.extend(
                [
                    self._word_tokenizer.tokenize(sent)
                    for sent in self._sent_tokenizer.tokenize(para)
                ]
            )
        return sents

    def _read_para_block(self, stream):
        paras = []
        for para in self._para_block_reader(stream):
            paras.append(
                [
                    self._word_tokenizer.tokenize(sent)
                    for sent in self._sent_tokenizer.tokenize(para)
                ]
            )
        return paras


class CategorizedPlaintextCorpusReader(CategorizedCorpusReader, PlaintextCorpusReader):
    """
    A reader for plaintext corpora whose documents are divided into
    categories based on their file identifiers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``PlaintextCorpusReader`` constructor.
        """
        CategorizedCorpusReader.__init__(self, kwargs)
        PlaintextCorpusReader.__init__(self, *args, **kwargs)


# FIXME: Is there a better way? How to not hardcode this?
#       Possibly, add a language kwargs to CategorizedPlaintextCorpusReader to
#       override the `sent_tokenizer`.
class PortugueseCategorizedPlaintextCorpusReader(CategorizedPlaintextCorpusReader):
    def __init__(self, *args, **kwargs):
        CategorizedCorpusReader.__init__(self, kwargs)
        kwargs["sent_tokenizer"] = nltk.data.LazyLoader(
            "tokenizers/punkt/portuguese.pickle"
        )
        PlaintextCorpusReader.__init__(self, *args, **kwargs)


class EuroparlCorpusReader(PlaintextCorpusReader):

    """
    Reader for Europarl corpora that consist of plaintext documents.
    Documents are divided into chapters instead of paragraphs as
    for regular plaintext documents. Chapters are separated using blank
    lines. Everything is inherited from ``PlaintextCorpusReader`` except
    that:

    - Since the corpus is pre-processed and pre-tokenized, the
      word tokenizer should just split the line at whitespaces.
    - For the same reason, the sentence tokenizer should just
      split the paragraph at line breaks.
    - There is a new 'chapters()' method that returns chapters instead
      instead of paragraphs.
    - The 'paras()' method inherited from PlaintextCorpusReader is
      made non-functional to remove any confusion between chapters
      and paragraphs for Europarl.
    """

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            words.extend(stream.readline().split())
        return words

    def _read_sent_block(self, stream):
        sents = []
        for para in self._para_block_reader(stream):
            sents.extend([sent.split() for sent in para.splitlines()])
        return sents

    def _read_para_block(self, stream):
        paras = []
        for para in self._para_block_reader(stream):
            paras.append([sent.split() for sent in para.splitlines()])
        return paras

    def chapters(self, fileids=None):
        """
        :return: the given file(s) as a list of
            chapters, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        return concat(
            [
                self.CorpusView(fileid, self._read_para_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def paras(self, fileids=None):
        raise NotImplementedError(
            "The Europarl corpus reader does not support paragraphs. Please use chapters() instead."
        )
