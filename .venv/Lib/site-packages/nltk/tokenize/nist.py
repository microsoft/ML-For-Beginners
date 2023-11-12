# Natural Language Toolkit: Python port of the mteval-v14.pl tokenizer.
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Liling Tan (ported from ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14.pl)
# Contributors: Ozan Caglayan, Wiktor Stribizew
#
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

"""
This is a NLTK port of the tokenizer used in the NIST BLEU evaluation script,
https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L926
which was also ported into Python in
https://github.com/lium-lst/nmtpy/blob/master/nmtpy/metrics/mtevalbleu.py#L162
"""


import io
import re

from nltk.corpus import perluniprops
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import xml_unescape


class NISTTokenizer(TokenizerI):
    """
    This NIST tokenizer is sentence-based instead of the original
    paragraph-based tokenization from mteval-14.pl; The sentence-based
    tokenization is consistent with the other tokenizers available in NLTK.

    >>> from nltk.tokenize.nist import NISTTokenizer
    >>> nist = NISTTokenizer()
    >>> s = "Good muffins cost $3.88 in New York."
    >>> expected_lower = [u'good', u'muffins', u'cost', u'$', u'3.88', u'in', u'new', u'york', u'.']
    >>> expected_cased = [u'Good', u'muffins', u'cost', u'$', u'3.88', u'in', u'New', u'York', u'.']
    >>> nist.tokenize(s, lowercase=False) == expected_cased
    True
    >>> nist.tokenize(s, lowercase=True) == expected_lower  # Lowercased.
    True

    The international_tokenize() is the preferred function when tokenizing
    non-european text, e.g.

    >>> from nltk.tokenize.nist import NISTTokenizer
    >>> nist = NISTTokenizer()

    # Input strings.
    >>> albb = u'Alibaba Group Holding Limited (Chinese: 阿里巴巴集团控股 有限公司) us a Chinese e-commerce company...'
    >>> amz = u'Amazon.com, Inc. (/ˈæməzɒn/) is an American electronic commerce...'
    >>> rkt = u'Rakuten, Inc. (楽天株式会社 Rakuten Kabushiki-gaisha) is a Japanese electronic commerce and Internet company based in Tokyo.'

    # Expected tokens.
    >>> expected_albb = [u'Alibaba', u'Group', u'Holding', u'Limited', u'(', u'Chinese', u':', u'\u963f\u91cc\u5df4\u5df4\u96c6\u56e2\u63a7\u80a1', u'\u6709\u9650\u516c\u53f8', u')']
    >>> expected_amz = [u'Amazon', u'.', u'com', u',', u'Inc', u'.', u'(', u'/', u'\u02c8\xe6', u'm']
    >>> expected_rkt = [u'Rakuten', u',', u'Inc', u'.', u'(', u'\u697d\u5929\u682a\u5f0f\u4f1a\u793e', u'Rakuten', u'Kabushiki', u'-', u'gaisha']

    >>> nist.international_tokenize(albb)[:10] == expected_albb
    True
    >>> nist.international_tokenize(amz)[:10] == expected_amz
    True
    >>> nist.international_tokenize(rkt)[:10] == expected_rkt
    True

    # Doctest for patching issue #1926
    >>> sent = u'this is a foo\u2604sentence.'
    >>> expected_sent = [u'this', u'is', u'a', u'foo', u'\u2604', u'sentence', u'.']
    >>> nist.international_tokenize(sent) == expected_sent
    True
    """

    # Strip "skipped" tags
    STRIP_SKIP = re.compile("<skipped>"), ""
    #  Strip end-of-line hyphenation and join lines
    STRIP_EOL_HYPHEN = re.compile("\u2028"), " "
    # Tokenize punctuation.
    PUNCT = re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), " \\1 "
    # Tokenize period and comma unless preceded by a digit.
    PERIOD_COMMA_PRECEED = re.compile(r"([^0-9])([\.,])"), "\\1 \\2 "
    # Tokenize period and comma unless followed by a digit.
    PERIOD_COMMA_FOLLOW = re.compile(r"([\.,])([^0-9])"), " \\1 \\2"
    # Tokenize dash when preceded by a digit
    DASH_PRECEED_DIGIT = re.compile("([0-9])(-)"), "\\1 \\2 "

    LANG_DEPENDENT_REGEXES = [
        PUNCT,
        PERIOD_COMMA_PRECEED,
        PERIOD_COMMA_FOLLOW,
        DASH_PRECEED_DIGIT,
    ]

    # Perluniprops characters used in NIST tokenizer.
    pup_number = str("".join(set(perluniprops.chars("Number"))))  # i.e. \p{N}
    pup_punct = str("".join(set(perluniprops.chars("Punctuation"))))  # i.e. \p{P}
    pup_symbol = str("".join(set(perluniprops.chars("Symbol"))))  # i.e. \p{S}

    # Python regexes needs to escape some special symbols, see
    # see https://stackoverflow.com/q/45670950/610569
    number_regex = re.sub(r"[]^\\-]", r"\\\g<0>", pup_number)
    punct_regex = re.sub(r"[]^\\-]", r"\\\g<0>", pup_punct)
    symbol_regex = re.sub(r"[]^\\-]", r"\\\g<0>", pup_symbol)

    # Note: In the original perl implementation, \p{Z} and \p{Zl} were used to
    #       (i) strip trailing and heading spaces  and
    #       (ii) de-deuplicate spaces.
    #       In Python, this would do: ' '.join(str.strip().split())
    # Thus, the next two lines were commented out.
    # Line_Separator = str(''.join(perluniprops.chars('Line_Separator'))) # i.e. \p{Zl}
    # Separator = str(''.join(perluniprops.chars('Separator'))) # i.e. \p{Z}

    # Pads non-ascii strings with space.
    NONASCII = re.compile("([\x00-\x7f]+)"), r" \1 "
    #  Tokenize any punctuation unless followed AND preceded by a digit.
    PUNCT_1 = (
        re.compile(f"([{number_regex}])([{punct_regex}])"),
        "\\1 \\2 ",
    )
    PUNCT_2 = (
        re.compile(f"([{punct_regex}])([{number_regex}])"),
        " \\1 \\2",
    )
    # Tokenize symbols
    SYMBOLS = re.compile(f"([{symbol_regex}])"), " \\1 "

    INTERNATIONAL_REGEXES = [NONASCII, PUNCT_1, PUNCT_2, SYMBOLS]

    def lang_independent_sub(self, text):
        """Performs the language independent string substituitions."""
        # It's a strange order of regexes.
        # It'll be better to unescape after STRIP_EOL_HYPHEN
        # but let's keep it close to the original NIST implementation.
        regexp, substitution = self.STRIP_SKIP
        text = regexp.sub(substitution, text)
        text = xml_unescape(text)
        regexp, substitution = self.STRIP_EOL_HYPHEN
        text = regexp.sub(substitution, text)
        return text

    def tokenize(self, text, lowercase=False, western_lang=True, return_str=False):
        text = str(text)
        # Language independent regex.
        text = self.lang_independent_sub(text)
        # Language dependent regex.
        if western_lang:
            # Pad string with whitespace.
            text = " " + text + " "
            if lowercase:
                text = text.lower()
            for regexp, substitution in self.LANG_DEPENDENT_REGEXES:
                text = regexp.sub(substitution, text)
        # Remove contiguous whitespaces.
        text = " ".join(text.split())
        # Finally, strips heading and trailing spaces
        # and converts output string into unicode.
        text = str(text.strip())
        return text if return_str else text.split()

    def international_tokenize(
        self, text, lowercase=False, split_non_ascii=True, return_str=False
    ):
        text = str(text)
        # Different from the 'normal' tokenize(), STRIP_EOL_HYPHEN is applied
        # first before unescaping.
        regexp, substitution = self.STRIP_SKIP
        text = regexp.sub(substitution, text)
        regexp, substitution = self.STRIP_EOL_HYPHEN
        text = regexp.sub(substitution, text)
        text = xml_unescape(text)

        if lowercase:
            text = text.lower()

        for regexp, substitution in self.INTERNATIONAL_REGEXES:
            text = regexp.sub(substitution, text)

        # Make sure that there's only one space only between words.
        # Strip leading and trailing spaces.
        text = " ".join(text.strip().split())
        return text if return_str else text.split()
