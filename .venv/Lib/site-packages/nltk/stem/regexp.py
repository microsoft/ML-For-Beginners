# Natural Language Toolkit: Stemmers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
#         Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
import re

from nltk.stem.api import StemmerI


class RegexpStemmer(StemmerI):
    """
    A stemmer that uses regular expressions to identify morphological
    affixes.  Any substrings that match the regular expressions will
    be removed.

        >>> from nltk.stem import RegexpStemmer
        >>> st = RegexpStemmer('ing$|s$|e$|able$', min=4)
        >>> st.stem('cars')
        'car'
        >>> st.stem('mass')
        'mas'
        >>> st.stem('was')
        'was'
        >>> st.stem('bee')
        'bee'
        >>> st.stem('compute')
        'comput'
        >>> st.stem('advisable')
        'advis'

    :type regexp: str or regexp
    :param regexp: The regular expression that should be used to
        identify morphological affixes.
    :type min: int
    :param min: The minimum length of string to stem
    """

    def __init__(self, regexp, min=0):

        if not hasattr(regexp, "pattern"):
            regexp = re.compile(regexp)
        self._regexp = regexp
        self._min = min

    def stem(self, word):
        if len(word) < self._min:
            return word
        else:
            return self._regexp.sub("", word)

    def __repr__(self):
        return f"<RegexpStemmer: {self._regexp.pattern!r}>"
