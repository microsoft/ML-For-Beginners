# Natural Language Toolkit: Europarl Corpus Readers
#
# Copyright (C) 2001-2023 NLTK Project
# Author:  Nitin Madnani <nmadnani@umiacs.umd.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re

from nltk.corpus.reader import *
from nltk.corpus.util import LazyCorpusLoader

# Create a new corpus reader instance for each European language
danish: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/danish", EuroparlCorpusReader, r"ep-.*\.da", encoding="utf-8"
)

dutch: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/dutch", EuroparlCorpusReader, r"ep-.*\.nl", encoding="utf-8"
)

english: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/english", EuroparlCorpusReader, r"ep-.*\.en", encoding="utf-8"
)

finnish: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/finnish", EuroparlCorpusReader, r"ep-.*\.fi", encoding="utf-8"
)

french: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/french", EuroparlCorpusReader, r"ep-.*\.fr", encoding="utf-8"
)

german: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/german", EuroparlCorpusReader, r"ep-.*\.de", encoding="utf-8"
)

greek: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/greek", EuroparlCorpusReader, r"ep-.*\.el", encoding="utf-8"
)

italian: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/italian", EuroparlCorpusReader, r"ep-.*\.it", encoding="utf-8"
)

portuguese: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/portuguese", EuroparlCorpusReader, r"ep-.*\.pt", encoding="utf-8"
)

spanish: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/spanish", EuroparlCorpusReader, r"ep-.*\.es", encoding="utf-8"
)

swedish: EuroparlCorpusReader = LazyCorpusLoader(
    "europarl_raw/swedish", EuroparlCorpusReader, r"ep-.*\.sv", encoding="utf-8"
)
