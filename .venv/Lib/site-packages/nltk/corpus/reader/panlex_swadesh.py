# Natural Language Toolkit: Word List Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


import re
from collections import defaultdict, namedtuple

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize

PanlexLanguage = namedtuple(
    "PanlexLanguage",
    [
        "panlex_uid",  # (1) PanLex UID
        "iso639",  # (2) ISO 639 language code
        "iso639_type",  # (3) ISO 639 language type, see README
        "script",  # (4) normal scripts of expressions
        "name",  # (5) PanLex default name
        "langvar_uid",  # (6) UID of the language variety in which the default name is an expression
    ],
)


class PanlexSwadeshCorpusReader(WordListCorpusReader):
    """
    This is a class to read the PanLex Swadesh list from

    David Kamholz, Jonathan Pool, and Susan M. Colowick (2014).
    PanLex: Building a Resource for Panlingual Lexical Translation.
    In LREC. http://www.lrec-conf.org/proceedings/lrec2014/pdf/1029_Paper.pdf

    License: CC0 1.0 Universal
    https://creativecommons.org/publicdomain/zero/1.0/legalcode
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Find the swadesh size using the fileids' path.
        self.swadesh_size = re.match(r"swadesh([0-9].*)\/", self.fileids()[0]).group(1)
        self._languages = {lang.panlex_uid: lang for lang in self.get_languages()}
        self._macro_langauges = self.get_macrolanguages()

    def license(self):
        return "CC0 1.0 Universal"

    def language_codes(self):
        return self._languages.keys()

    def get_languages(self):
        for line in self.raw(f"langs{self.swadesh_size}.txt").split("\n"):
            if not line.strip():  # Skip empty lines.
                continue
            yield PanlexLanguage(*line.strip().split("\t"))

    def get_macrolanguages(self):
        macro_langauges = defaultdict(list)
        for lang in self._languages.values():
            macro_langauges[lang.iso639].append(lang.panlex_uid)
        return macro_langauges

    def words_by_lang(self, lang_code):
        """
        :return: a list of list(str)
        """
        fileid = f"swadesh{self.swadesh_size}/{lang_code}.txt"
        return [concept.split("\t") for concept in self.words(fileid)]

    def words_by_iso639(self, iso63_code):
        """
        :return: a list of list(str)
        """
        fileids = [
            f"swadesh{self.swadesh_size}/{lang_code}.txt"
            for lang_code in self._macro_langauges[iso63_code]
        ]
        return [
            concept.split("\t") for fileid in fileids for concept in self.words(fileid)
        ]

    def entries(self, fileids=None):
        """
        :return: a tuple of words for the specified fileids.
        """
        if not fileids:
            fileids = self.fileids()

        wordlists = [self.words(f) for f in fileids]
        return list(zip(*wordlists))
