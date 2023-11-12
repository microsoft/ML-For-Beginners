# Natural Language Toolkit: Switchboard Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
import re

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple


class SwitchboardTurn(list):
    """
    A specialized list object used to encode switchboard utterances.
    The elements of the list are the words in the utterance; and two
    attributes, ``speaker`` and ``id``, are provided to retrieve the
    spearker identifier and utterance id.  Note that utterance ids
    are only unique within a given discourse.
    """

    def __init__(self, words, speaker, id):
        list.__init__(self, words)
        self.speaker = speaker
        self.id = int(id)

    def __repr__(self):
        if len(self) == 0:
            text = ""
        elif isinstance(self[0], tuple):
            text = " ".join("%s/%s" % w for w in self)
        else:
            text = " ".join(self)
        return f"<{self.speaker}.{self.id}: {text!r}>"


class SwitchboardCorpusReader(CorpusReader):
    _FILES = ["tagged"]
    # Use the "tagged" file even for non-tagged data methods, since
    # it's tokenized.

    def __init__(self, root, tagset=None):
        CorpusReader.__init__(self, root, self._FILES)
        self._tagset = tagset

    def words(self):
        return StreamBackedCorpusView(self.abspath("tagged"), self._words_block_reader)

    def tagged_words(self, tagset=None):
        def tagged_words_block_reader(stream):
            return self._tagged_words_block_reader(stream, tagset)

        return StreamBackedCorpusView(self.abspath("tagged"), tagged_words_block_reader)

    def turns(self):
        return StreamBackedCorpusView(self.abspath("tagged"), self._turns_block_reader)

    def tagged_turns(self, tagset=None):
        def tagged_turns_block_reader(stream):
            return self._tagged_turns_block_reader(stream, tagset)

        return StreamBackedCorpusView(self.abspath("tagged"), tagged_turns_block_reader)

    def discourses(self):
        return StreamBackedCorpusView(
            self.abspath("tagged"), self._discourses_block_reader
        )

    def tagged_discourses(self, tagset=False):
        def tagged_discourses_block_reader(stream):
            return self._tagged_discourses_block_reader(stream, tagset)

        return StreamBackedCorpusView(
            self.abspath("tagged"), tagged_discourses_block_reader
        )

    def _discourses_block_reader(self, stream):
        # returns at most 1 discourse.  (The other methods depend on this.)
        return [
            [
                self._parse_utterance(u, include_tag=False)
                for b in read_blankline_block(stream)
                for u in b.split("\n")
                if u.strip()
            ]
        ]

    def _tagged_discourses_block_reader(self, stream, tagset=None):
        # returns at most 1 discourse.  (The other methods depend on this.)
        return [
            [
                self._parse_utterance(u, include_tag=True, tagset=tagset)
                for b in read_blankline_block(stream)
                for u in b.split("\n")
                if u.strip()
            ]
        ]

    def _turns_block_reader(self, stream):
        return self._discourses_block_reader(stream)[0]

    def _tagged_turns_block_reader(self, stream, tagset=None):
        return self._tagged_discourses_block_reader(stream, tagset)[0]

    def _words_block_reader(self, stream):
        return sum(self._discourses_block_reader(stream)[0], [])

    def _tagged_words_block_reader(self, stream, tagset=None):
        return sum(self._tagged_discourses_block_reader(stream, tagset)[0], [])

    _UTTERANCE_RE = re.compile(r"(\w+)\.(\d+)\:\s*(.*)")
    _SEP = "/"

    def _parse_utterance(self, utterance, include_tag, tagset=None):
        m = self._UTTERANCE_RE.match(utterance)
        if m is None:
            raise ValueError("Bad utterance %r" % utterance)
        speaker, id, text = m.groups()
        words = [str2tuple(s, self._SEP) for s in text.split()]
        if not include_tag:
            words = [w for (w, t) in words]
        elif tagset and tagset != self._tagset:
            words = [(w, map_tag(self._tagset, tagset, t)) for (w, t) in words]
        return SwitchboardTurn(words, speaker, id)
