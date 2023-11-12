# Natural Language Toolkit: Comparative Sentence Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Pierpaolo Pantone <24alsecondo@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
CorpusReader for the Comparative Sentence Dataset.

- Comparative Sentence Dataset information -

Annotated by: Nitin Jindal and Bing Liu, 2006.
              Department of Computer Sicence
              University of Illinois at Chicago

Contact: Nitin Jindal, njindal@cs.uic.edu
         Bing Liu, liub@cs.uic.edu (https://www.cs.uic.edu/~liub)

Distributed with permission.

Related papers:

- Nitin Jindal and Bing Liu. "Identifying Comparative Sentences in Text Documents".
   Proceedings of the ACM SIGIR International Conference on Information Retrieval
   (SIGIR-06), 2006.

- Nitin Jindal and Bing Liu. "Mining Comprative Sentences and Relations".
   Proceedings of Twenty First National Conference on Artificial Intelligence
   (AAAI-2006), 2006.

- Murthy Ganapathibhotla and Bing Liu. "Mining Opinions in Comparative Sentences".
    Proceedings of the 22nd International Conference on Computational Linguistics
    (Coling-2008), Manchester, 18-22 August, 2008.
"""
import re

from nltk.corpus.reader.api import *
from nltk.tokenize import *

# Regular expressions for dataset components
STARS = re.compile(r"^\*+$")
COMPARISON = re.compile(r"<cs-[1234]>")
CLOSE_COMPARISON = re.compile(r"</cs-[1234]>")
GRAD_COMPARISON = re.compile(r"<cs-[123]>")
NON_GRAD_COMPARISON = re.compile(r"<cs-4>")
ENTITIES_FEATS = re.compile(r"(\d)_((?:[\.\w\s/-](?!\d_))+)")
KEYWORD = re.compile(r"\(([^\(]*)\)$")


class Comparison:
    """
    A Comparison represents a comparative sentence and its constituents.
    """

    def __init__(
        self,
        text=None,
        comp_type=None,
        entity_1=None,
        entity_2=None,
        feature=None,
        keyword=None,
    ):
        """
        :param text: a string (optionally tokenized) containing a comparison.
        :param comp_type: an integer defining the type of comparison expressed.
            Values can be: 1 (Non-equal gradable), 2 (Equative), 3 (Superlative),
            4 (Non-gradable).
        :param entity_1: the first entity considered in the comparison relation.
        :param entity_2: the second entity considered in the comparison relation.
        :param feature: the feature considered in the comparison relation.
        :param keyword: the word or phrase which is used for that comparative relation.
        """
        self.text = text
        self.comp_type = comp_type
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.feature = feature
        self.keyword = keyword

    def __repr__(self):
        return (
            'Comparison(text="{}", comp_type={}, entity_1="{}", entity_2="{}", '
            'feature="{}", keyword="{}")'
        ).format(
            self.text,
            self.comp_type,
            self.entity_1,
            self.entity_2,
            self.feature,
            self.keyword,
        )


class ComparativeSentencesCorpusReader(CorpusReader):
    """
    Reader for the Comparative Sentence Dataset by Jindal and Liu (2006).

        >>> from nltk.corpus import comparative_sentences
        >>> comparison = comparative_sentences.comparisons()[0]
        >>> comparison.text # doctest: +NORMALIZE_WHITESPACE
        ['its', 'fast-forward', 'and', 'rewind', 'work', 'much', 'more', 'smoothly',
        'and', 'consistently', 'than', 'those', 'of', 'other', 'models', 'i', "'ve",
        'had', '.']
        >>> comparison.entity_2
        'models'
        >>> (comparison.feature, comparison.keyword)
        ('rewind', 'more')
        >>> len(comparative_sentences.comparisons())
        853
    """

    CorpusView = StreamBackedCorpusView

    def __init__(
        self,
        root,
        fileids,
        word_tokenizer=WhitespaceTokenizer(),
        sent_tokenizer=None,
        encoding="utf8",
    ):
        """
        :param root: The root directory for this corpus.
        :param fileids: a list or regexp specifying the fileids in this corpus.
        :param word_tokenizer: tokenizer for breaking sentences or paragraphs
            into words. Default: `WhitespaceTokenizer`
        :param sent_tokenizer: tokenizer for breaking paragraphs into sentences.
        :param encoding: the encoding that should be used to read the corpus.
        """

        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._readme = "README.txt"

    def comparisons(self, fileids=None):
        """
        Return all comparisons in the corpus.

        :param fileids: a list or regexp specifying the ids of the files whose
            comparisons have to be returned.
        :return: the given file(s) as a list of Comparison objects.
        :rtype: list(Comparison)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat(
            [
                self.CorpusView(path, self._read_comparison_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def keywords(self, fileids=None):
        """
        Return a set of all keywords used in the corpus.

        :param fileids: a list or regexp specifying the ids of the files whose
            keywords have to be returned.
        :return: the set of keywords and comparative phrases used in the corpus.
        :rtype: set(str)
        """
        all_keywords = concat(
            [
                self.CorpusView(path, self._read_keyword_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

        keywords_set = {keyword.lower() for keyword in all_keywords if keyword}
        return keywords_set

    def keywords_readme(self):
        """
        Return the list of words and constituents considered as clues of a
        comparison (from listOfkeywords.txt).
        """
        keywords = []
        with self.open("listOfkeywords.txt") as fp:
            raw_text = fp.read()
        for line in raw_text.split("\n"):
            if not line or line.startswith("//"):
                continue
            keywords.append(line.strip())
        return keywords

    def sents(self, fileids=None):
        """
        Return all sentences in the corpus.

        :param fileids: a list or regexp specifying the ids of the files whose
            sentences have to be returned.
        :return: all sentences of the corpus as lists of tokens (or as plain
            strings, if no word tokenizer is specified).
        :rtype: list(list(str)) or list(str)
        """
        return concat(
            [
                self.CorpusView(path, self._read_sent_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def words(self, fileids=None):
        """
        Return all words and punctuation symbols in the corpus.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return concat(
            [
                self.CorpusView(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def _read_comparison_block(self, stream):
        while True:
            line = stream.readline()
            if not line:
                return []  # end of file.
            comparison_tags = re.findall(COMPARISON, line)
            if comparison_tags:
                grad_comparisons = re.findall(GRAD_COMPARISON, line)
                non_grad_comparisons = re.findall(NON_GRAD_COMPARISON, line)
                # Advance to the next line (it contains the comparative sentence)
                comparison_text = stream.readline().strip()
                if self._word_tokenizer:
                    comparison_text = self._word_tokenizer.tokenize(comparison_text)
                # Skip the next line (it contains closing comparison tags)
                stream.readline()
                # If gradable comparisons are found, create Comparison instances
                # and populate their fields
                comparison_bundle = []
                if grad_comparisons:
                    # Each comparison tag has its own relations on a separate line
                    for comp in grad_comparisons:
                        comp_type = int(re.match(r"<cs-(\d)>", comp).group(1))
                        comparison = Comparison(
                            text=comparison_text, comp_type=comp_type
                        )
                        line = stream.readline()
                        entities_feats = ENTITIES_FEATS.findall(line)
                        if entities_feats:
                            for (code, entity_feat) in entities_feats:
                                if code == "1":
                                    comparison.entity_1 = entity_feat.strip()
                                elif code == "2":
                                    comparison.entity_2 = entity_feat.strip()
                                elif code == "3":
                                    comparison.feature = entity_feat.strip()
                        keyword = KEYWORD.findall(line)
                        if keyword:
                            comparison.keyword = keyword[0]
                        comparison_bundle.append(comparison)
                # If non-gradable comparisons are found, create a simple Comparison
                # instance for each one
                if non_grad_comparisons:
                    for comp in non_grad_comparisons:
                        # comp_type in this case should always be 4.
                        comp_type = int(re.match(r"<cs-(\d)>", comp).group(1))
                        comparison = Comparison(
                            text=comparison_text, comp_type=comp_type
                        )
                        comparison_bundle.append(comparison)
                # Flatten the list of comparisons before returning them
                # return concat([comparison_bundle])
                return comparison_bundle

    def _read_keyword_block(self, stream):
        keywords = []
        for comparison in self._read_comparison_block(stream):
            keywords.append(comparison.keyword)
        return keywords

    def _read_sent_block(self, stream):
        while True:
            line = stream.readline()
            if re.match(STARS, line):
                while True:
                    line = stream.readline()
                    if re.match(STARS, line):
                        break
                continue
            if (
                not re.findall(COMPARISON, line)
                and not ENTITIES_FEATS.findall(line)
                and not re.findall(CLOSE_COMPARISON, line)
            ):
                if self._sent_tokenizer:
                    return [
                        self._word_tokenizer.tokenize(sent)
                        for sent in self._sent_tokenizer.tokenize(line)
                    ]
                else:
                    return [self._word_tokenizer.tokenize(line)]

    def _read_word_block(self, stream):
        words = []
        for sent in self._read_sent_block(stream):
            words.extend(sent)
        return words
