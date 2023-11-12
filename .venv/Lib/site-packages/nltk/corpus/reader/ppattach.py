# Natural Language Toolkit: PP Attachment Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Read lines from the Prepositional Phrase Attachment Corpus.

The PP Attachment Corpus contains several files having the format:

sentence_id verb noun1 preposition noun2 attachment

For example:

42960 gives authority to administration V
46742 gives inventors of microchip N

The PP attachment is to the verb phrase (V) or noun phrase (N), i.e.:

(VP gives (NP authority) (PP to administration))
(VP gives (NP inventors (PP of microchip)))

The corpus contains the following files:

training:   training set
devset:     development test set, used for algorithm development.
test:       test set, used to report results
bitstrings: word classes derived from Mutual Information Clustering for the Wall Street Journal.

Ratnaparkhi, Adwait (1994). A Maximum Entropy Model for Prepositional
Phrase Attachment.  Proceedings of the ARPA Human Language Technology
Conference.  [http://www.cis.upenn.edu/~adwait/papers/hlt94.ps]

The PP Attachment Corpus is distributed with NLTK with the permission
of the author.
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *


class PPAttachment:
    def __init__(self, sent, verb, noun1, prep, noun2, attachment):
        self.sent = sent
        self.verb = verb
        self.noun1 = noun1
        self.prep = prep
        self.noun2 = noun2
        self.attachment = attachment

    def __repr__(self):
        return (
            "PPAttachment(sent=%r, verb=%r, noun1=%r, prep=%r, "
            "noun2=%r, attachment=%r)"
            % (self.sent, self.verb, self.noun1, self.prep, self.noun2, self.attachment)
        )


class PPAttachmentCorpusReader(CorpusReader):
    """
    sentence_id verb noun1 preposition noun2 attachment
    """

    def attachments(self, fileids):
        return concat(
            [
                StreamBackedCorpusView(fileid, self._read_obj_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def tuples(self, fileids):
        return concat(
            [
                StreamBackedCorpusView(fileid, self._read_tuple_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def _read_tuple_block(self, stream):
        line = stream.readline()
        if line:
            return [tuple(line.split())]
        else:
            return []

    def _read_obj_block(self, stream):
        line = stream.readline()
        if line:
            return [PPAttachment(*line.split())]
        else:
            return []
