# Natural Language Toolkit: Tagged Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Jacob Perkins <japerk@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A reader for corpora whose documents contain part-of-speech-tagged words.
"""

import os

from nltk.corpus.reader.api import *
from nltk.corpus.reader.timit import read_timit_block
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
from nltk.tokenize import *


class TaggedCorpusReader(CorpusReader):
    """
    Reader for simple part-of-speech tagged corpora.  Paragraphs are
    assumed to be split using blank lines.  Sentences and words can be
    tokenized using the default tokenizers, or by custom tokenizers
    specified as parameters to the constructor.  Words are parsed
    using ``nltk.tag.str2tuple``.  By default, ``'/'`` is used as the
    separator.  I.e., words should have the form::

       word1/tag1 word2/tag2 word3/tag3 ...

    But custom separators may be specified as parameters to the
    constructor.  Part of speech tags are case-normalized to upper
    case.
    """

    def __init__(
        self,
        root,
        fileids,
        sep="/",
        word_tokenizer=WhitespaceTokenizer(),
        sent_tokenizer=RegexpTokenizer("\n", gaps=True),
        para_block_reader=read_blankline_block,
        encoding="utf8",
        tagset=None,
    ):
        """
        Construct a new Tagged Corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/...path to corpus.../'
            >>> reader = TaggedCorpusReader(root, '.*', '.txt') # doctest: +SKIP

        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._sep = sep
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader
        self._tagset = tagset

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                TaggedCorpusView(
                    fileid,
                    enc,
                    False,
                    False,
                    False,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    None,
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
                TaggedCorpusView(
                    fileid,
                    enc,
                    False,
                    True,
                    False,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    None,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        return concat(
            [
                TaggedCorpusView(
                    fileid,
                    enc,
                    False,
                    True,
                    True,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    None,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_words(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat(
            [
                TaggedCorpusView(
                    fileid,
                    enc,
                    True,
                    False,
                    False,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    tag_mapping_function,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_sents(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.

        :rtype: list(list(tuple(str,str)))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat(
            [
                TaggedCorpusView(
                    fileid,
                    enc,
                    True,
                    True,
                    False,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    tag_mapping_function,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tagged_paras(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of ``(word,tag)`` tuples.
        :rtype: list(list(list(tuple(str,str))))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat(
            [
                TaggedCorpusView(
                    fileid,
                    enc,
                    True,
                    True,
                    True,
                    self._sep,
                    self._word_tokenizer,
                    self._sent_tokenizer,
                    self._para_block_reader,
                    tag_mapping_function,
                )
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )


class CategorizedTaggedCorpusReader(CategorizedCorpusReader, TaggedCorpusReader):
    """
    A reader for part-of-speech tagged corpora whose documents are
    divided into categories based on their file identifiers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``TaggedCorpusReader``.
        """
        CategorizedCorpusReader.__init__(self, kwargs)
        TaggedCorpusReader.__init__(self, *args, **kwargs)

    def tagged_words(self, fileids=None, categories=None, tagset=None):
        return super().tagged_words(self._resolve(fileids, categories), tagset)

    def tagged_sents(self, fileids=None, categories=None, tagset=None):
        return super().tagged_sents(self._resolve(fileids, categories), tagset)

    def tagged_paras(self, fileids=None, categories=None, tagset=None):
        return super().tagged_paras(self._resolve(fileids, categories), tagset)


class TaggedCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for tagged documents.  It can be
    customized via flags to divide the tagged corpus documents up by
    sentence or paragraph, and to include or omit part of speech tags.
    ``TaggedCorpusView`` objects are typically created by
    ``TaggedCorpusReader`` (not directly by nltk users).
    """

    def __init__(
        self,
        corpus_file,
        encoding,
        tagged,
        group_by_sent,
        group_by_para,
        sep,
        word_tokenizer,
        sent_tokenizer,
        para_block_reader,
        tag_mapping_function=None,
    ):
        self._tagged = tagged
        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        self._sep = sep
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader
        self._tag_mapping_function = tag_mapping_function
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        """Reads one paragraph at a time."""
        block = []
        for para_str in self._para_block_reader(stream):
            para = []
            for sent_str in self._sent_tokenizer.tokenize(para_str):
                sent = [
                    str2tuple(s, self._sep)
                    for s in self._word_tokenizer.tokenize(sent_str)
                ]
                if self._tag_mapping_function:
                    sent = [(w, self._tag_mapping_function(t)) for (w, t) in sent]
                if not self._tagged:
                    sent = [w for (w, t) in sent]
                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)
            if self._group_by_para:
                block.append(para)
            else:
                block.extend(para)
        return block


# needs to implement simplified tags
class MacMorphoCorpusReader(TaggedCorpusReader):
    """
    A corpus reader for the MAC_MORPHO corpus.  Each line contains a
    single tagged word, using '_' as a separator.  Sentence boundaries
    are based on the end-sentence tag ('_.').  Paragraph information
    is not included in the corpus, so each paragraph returned by
    ``self.paras()`` and ``self.tagged_paras()`` contains a single
    sentence.
    """

    def __init__(self, root, fileids, encoding="utf8", tagset=None):
        TaggedCorpusReader.__init__(
            self,
            root,
            fileids,
            sep="_",
            word_tokenizer=LineTokenizer(),
            sent_tokenizer=RegexpTokenizer(".*\n"),
            para_block_reader=self._read_block,
            encoding=encoding,
            tagset=tagset,
        )

    def _read_block(self, stream):
        return read_regexp_block(stream, r".*", r".*_\.")


class TimitTaggedCorpusReader(TaggedCorpusReader):
    """
    A corpus reader for tagged sentences that are included in the TIMIT corpus.
    """

    def __init__(self, *args, **kwargs):
        TaggedCorpusReader.__init__(
            self, para_block_reader=read_timit_block, *args, **kwargs
        )

    def paras(self):
        raise NotImplementedError("use sents() instead")

    def tagged_paras(self):
        raise NotImplementedError("use tagged_sents() instead")
