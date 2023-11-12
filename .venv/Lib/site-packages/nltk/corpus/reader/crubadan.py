# Natural Language Toolkit: An Crubadan N-grams Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Avital Pekker <avital.pekker@utoronto.ca>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
An NLTK interface for the n-gram statistics gathered from
the corpora for each language using An Crubadan.

There are multiple potential applications for the data but
this reader was created with the goal of using it in the
context of language identification.

For details about An Crubadan, this data, and its potential uses, see:
http://borel.slu.edu/crubadan/index.html
"""

import re
from os import path

from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist


class CrubadanCorpusReader(CorpusReader):
    """
    A corpus reader used to access language An Crubadan n-gram files.
    """

    _LANG_MAPPER_FILE = "table.txt"
    _all_lang_freq = {}

    def __init__(self, root, fileids, encoding="utf8", tagset=None):
        super().__init__(root, fileids, encoding="utf8")
        self._lang_mapping_data = []
        self._load_lang_mapping_data()

    def lang_freq(self, lang):
        """Return n-gram FreqDist for a specific language
        given ISO 639-3 language code"""

        if lang not in self._all_lang_freq:
            self._all_lang_freq[lang] = self._load_lang_ngrams(lang)

        return self._all_lang_freq[lang]

    def langs(self):
        """Return a list of supported languages as ISO 639-3 codes"""
        return [row[1] for row in self._lang_mapping_data]

    def iso_to_crubadan(self, lang):
        """Return internal Crubadan code based on ISO 639-3 code"""
        for i in self._lang_mapping_data:
            if i[1].lower() == lang.lower():
                return i[0]

    def crubadan_to_iso(self, lang):
        """Return ISO 639-3 code given internal Crubadan code"""
        for i in self._lang_mapping_data:
            if i[0].lower() == lang.lower():
                return i[1]

    def _load_lang_mapping_data(self):
        """Load language mappings between codes and description from table.txt"""
        if isinstance(self.root, ZipFilePathPointer):
            raise RuntimeError(
                "Please install the 'crubadan' corpus first, use nltk.download()"
            )

        mapper_file = path.join(self.root, self._LANG_MAPPER_FILE)
        if self._LANG_MAPPER_FILE not in self.fileids():
            raise RuntimeError("Could not find language mapper file: " + mapper_file)

        with open(mapper_file, encoding="utf-8") as raw:
            strip_raw = raw.read().strip()

            self._lang_mapping_data = [row.split("\t") for row in strip_raw.split("\n")]

    def _load_lang_ngrams(self, lang):
        """Load single n-gram language file given the ISO 639-3 language code
        and return its FreqDist"""

        if lang not in self.langs():
            raise RuntimeError("Unsupported language.")

        crubadan_code = self.iso_to_crubadan(lang)
        ngram_file = path.join(self.root, crubadan_code + "-3grams.txt")

        if not path.isfile(ngram_file):
            raise RuntimeError("No N-gram file found for requested language.")

        counts = FreqDist()
        with open(ngram_file, encoding="utf-8") as f:
            for line in f:
                data = line.split(" ")

                ngram = data[1].strip("\n")
                freq = int(data[0])

                counts[ngram] = freq

        return counts
