# Natural Language Toolkit: Opinion Lexicon Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Pierpaolo Pantone <24alsecondo@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
CorpusReader for the Opinion Lexicon.

Opinion Lexicon information
===========================

Authors: Minqing Hu and Bing Liu, 2004.
    Department of Computer Science
    University of Illinois at Chicago

Contact: Bing Liu, liub@cs.uic.edu
        https://www.cs.uic.edu/~liub

Distributed with permission.

Related papers:

- Minqing Hu and Bing Liu. "Mining and summarizing customer reviews".
    Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery
    & Data Mining (KDD-04), Aug 22-25, 2004, Seattle, Washington, USA.

- Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing and
    Comparing Opinions on the Web". Proceedings of the 14th International World
    Wide Web conference (WWW-2005), May 10-14, 2005, Chiba, Japan.
"""

from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *


class IgnoreReadmeCorpusView(StreamBackedCorpusView):
    """
    This CorpusView is used to skip the initial readme block of the corpus.
    """

    def __init__(self, *args, **kwargs):
        StreamBackedCorpusView.__init__(self, *args, **kwargs)
        # open self._stream
        self._open()
        # skip the readme block
        read_blankline_block(self._stream)
        # Set the initial position to the current stream position
        self._filepos = [self._stream.tell()]


class OpinionLexiconCorpusReader(WordListCorpusReader):
    """
    Reader for Liu and Hu opinion lexicon.  Blank lines and readme are ignored.

        >>> from nltk.corpus import opinion_lexicon
        >>> opinion_lexicon.words()
        ['2-faced', '2-faces', 'abnormal', 'abolish', ...]

    The OpinionLexiconCorpusReader provides shortcuts to retrieve positive/negative
    words:

        >>> opinion_lexicon.negative()
        ['2-faced', '2-faces', 'abnormal', 'abolish', ...]

    Note that words from `words()` method are sorted by file id, not alphabetically:

        >>> opinion_lexicon.words()[0:10] # doctest: +NORMALIZE_WHITESPACE
        ['2-faced', '2-faces', 'abnormal', 'abolish', 'abominable', 'abominably',
        'abominate', 'abomination', 'abort', 'aborted']
        >>> sorted(opinion_lexicon.words())[0:10] # doctest: +NORMALIZE_WHITESPACE
        ['2-faced', '2-faces', 'a+', 'abnormal', 'abolish', 'abominable', 'abominably',
        'abominate', 'abomination', 'abort']
    """

    CorpusView = IgnoreReadmeCorpusView

    def words(self, fileids=None):
        """
        Return all words in the opinion lexicon. Note that these words are not
        sorted in alphabetical order.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat(
            [
                self.CorpusView(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )

    def positive(self):
        """
        Return all positive words in alphabetical order.

        :return: a list of positive words.
        :rtype: list(str)
        """
        return self.words("positive-words.txt")

    def negative(self):
        """
        Return all negative words in alphabetical order.

        :return: a list of negative words.
        :rtype: list(str)
        """
        return self.words("negative-words.txt")

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            line = stream.readline()
            if not line:
                continue
            words.append(line.strip())
        return words
