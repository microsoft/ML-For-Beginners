# Natural Language Toolkit: String Category Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Read tuples from a corpus consisting of categorized strings.
For example, from the question classification corpus:

NUM:dist How far is it from Denver to Aspen ?
LOC:city What county is Modesto , California in ?
HUM:desc Who was Galileo ?
DESC:def What is an atom ?
NUM:date When did Hawaii become a state ?
"""

from nltk.corpus.reader.api import *

# based on PPAttachmentCorpusReader
from nltk.corpus.reader.util import *


# [xx] Should the order of the tuple be reversed -- in most other places
# in nltk, we use the form (data, tag) -- e.g., tagged words and
# labeled texts for classifiers.
class StringCategoryCorpusReader(CorpusReader):
    def __init__(self, root, fileids, delimiter=" ", encoding="utf8"):
        """
        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param delimiter: Field delimiter
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._delimiter = delimiter

    def tuples(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat(
            [
                StreamBackedCorpusView(fileid, self._read_tuple_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def _read_tuple_block(self, stream):
        line = stream.readline().strip()
        if line:
            return [tuple(line.split(self._delimiter, 1))]
        else:
            return []
