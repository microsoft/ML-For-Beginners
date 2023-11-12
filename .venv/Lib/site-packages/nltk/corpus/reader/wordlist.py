# Natural Language Toolkit: Word List Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import line_tokenize


class WordListCorpusReader(CorpusReader):
    """
    List of words, one per line.  Blank lines are ignored.
    """

    def words(self, fileids=None, ignore_lines_startswith="\n"):
        return [
            line
            for line in line_tokenize(self.raw(fileids))
            if not line.startswith(ignore_lines_startswith)
        ]


class SwadeshCorpusReader(WordListCorpusReader):
    def entries(self, fileids=None):
        """
        :return: a tuple of words for the specified fileids.
        """
        if not fileids:
            fileids = self.fileids()

        wordlists = [self.words(f) for f in fileids]
        return list(zip(*wordlists))


class NonbreakingPrefixesCorpusReader(WordListCorpusReader):
    """
    This is a class to read the nonbreaking prefixes textfiles from the
    Moses Machine Translation toolkit. These lists are used in the Python port
    of the Moses' word tokenizer.
    """

    available_langs = {
        "catalan": "ca",
        "czech": "cs",
        "german": "de",
        "greek": "el",
        "english": "en",
        "spanish": "es",
        "finnish": "fi",
        "french": "fr",
        "hungarian": "hu",
        "icelandic": "is",
        "italian": "it",
        "latvian": "lv",
        "dutch": "nl",
        "polish": "pl",
        "portuguese": "pt",
        "romanian": "ro",
        "russian": "ru",
        "slovak": "sk",
        "slovenian": "sl",
        "swedish": "sv",
        "tamil": "ta",
    }
    # Also, add the lang IDs as the keys.
    available_langs.update({v: v for v in available_langs.values()})

    def words(self, lang=None, fileids=None, ignore_lines_startswith="#"):
        """
        This module returns a list of nonbreaking prefixes for the specified
        language(s).

        >>> from nltk.corpus import nonbreaking_prefixes as nbp
        >>> nbp.words('en')[:10] == [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J']
        True
        >>> nbp.words('ta')[:5] == [u'\u0b85', u'\u0b86', u'\u0b87', u'\u0b88', u'\u0b89']
        True

        :return: a list words for the specified language(s).
        """
        # If *lang* in list of languages available, allocate apt fileid.
        # Otherwise, the function returns non-breaking prefixes for
        # all languages when fileids==None.
        if lang in self.available_langs:
            lang = self.available_langs[lang]
            fileids = ["nonbreaking_prefix." + lang]
        return [
            line
            for line in line_tokenize(self.raw(fileids))
            if not line.startswith(ignore_lines_startswith)
        ]


class UnicharsCorpusReader(WordListCorpusReader):
    """
    This class is used to read lists of characters from the Perl Unicode
    Properties (see https://perldoc.perl.org/perluniprops.html).
    The files in the perluniprop.zip are extracted using the Unicode::Tussle
    module from https://search.cpan.org/~bdfoy/Unicode-Tussle-1.11/lib/Unicode/Tussle.pm
    """

    # These are categories similar to the Perl Unicode Properties
    available_categories = [
        "Close_Punctuation",
        "Currency_Symbol",
        "IsAlnum",
        "IsAlpha",
        "IsLower",
        "IsN",
        "IsSc",
        "IsSo",
        "IsUpper",
        "Line_Separator",
        "Number",
        "Open_Punctuation",
        "Punctuation",
        "Separator",
        "Symbol",
    ]

    def chars(self, category=None, fileids=None):
        """
        This module returns a list of characters from  the Perl Unicode Properties.
        They are very useful when porting Perl tokenizers to Python.

        >>> from nltk.corpus import perluniprops as pup
        >>> pup.chars('Open_Punctuation')[:5] == [u'(', u'[', u'{', u'\u0f3a', u'\u0f3c']
        True
        >>> pup.chars('Currency_Symbol')[:5] == [u'$', u'\xa2', u'\xa3', u'\xa4', u'\xa5']
        True
        >>> pup.available_categories
        ['Close_Punctuation', 'Currency_Symbol', 'IsAlnum', 'IsAlpha', 'IsLower', 'IsN', 'IsSc', 'IsSo', 'IsUpper', 'Line_Separator', 'Number', 'Open_Punctuation', 'Punctuation', 'Separator', 'Symbol']

        :return: a list of characters given the specific unicode character category
        """
        if category in self.available_categories:
            fileids = [category + ".txt"]
        return list(self.raw(fileids).strip())


class MWAPPDBCorpusReader(WordListCorpusReader):
    """
    This class is used to read the list of word pairs from the subset of lexical
    pairs of The Paraphrase Database (PPDB) XXXL used in the Monolingual Word
    Alignment (MWA) algorithm described in Sultan et al. (2014a, 2014b, 2015):

     - http://acl2014.org/acl2014/Q14/pdf/Q14-1017
     - https://www.aclweb.org/anthology/S14-2039
     - https://www.aclweb.org/anthology/S15-2027

    The original source of the full PPDB corpus can be found on
    https://www.cis.upenn.edu/~ccb/ppdb/

    :return: a list of tuples of similar lexical terms.
    """

    mwa_ppdb_xxxl_file = "ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs"

    def entries(self, fileids=mwa_ppdb_xxxl_file):
        """
        :return: a tuple of synonym word pairs.
        """
        return [tuple(line.split("\t")) for line in line_tokenize(self.raw(fileids))]
