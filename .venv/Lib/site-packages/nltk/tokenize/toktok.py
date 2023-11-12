# Natural Language Toolkit: Python port of the tok-tok.pl tokenizer.
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Jon Dehdari
# Contributors: Liling Tan, Selcuk Ayguney, ikegami, Martijn Pieters
#
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

"""
The tok-tok tokenizer is a simple, general tokenizer, where the input has one
sentence per line; thus only final period is tokenized.

Tok-tok has been tested on, and gives reasonably good results for English,
Persian, Russian, Czech, French, German, Vietnamese, Tajik, and a few others.
The input should be in UTF-8 encoding.

Reference:
Jon Dehdari. 2014. A Neurophysiologically-Inspired Statistical Language
Model (Doctoral dissertation). Columbus, OH, USA: The Ohio State University.
"""

import re

from nltk.tokenize.api import TokenizerI


class ToktokTokenizer(TokenizerI):
    """
    This is a Python port of the tok-tok.pl from
    https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl

    >>> toktok = ToktokTokenizer()
    >>> text = u'Is 9.5 or 525,600 my favorite number?'
    >>> print(toktok.tokenize(text, return_str=True))
    Is 9.5 or 525,600 my favorite number ?
    >>> text = u'The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things'
    >>> print(toktok.tokenize(text, return_str=True))
    The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things
    >>> text = u'\xa1This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf'
    >>> expected = u'\xa1 This , is a sentence with weird \xbb symbols \u2026 appearing everywhere \xbf'
    >>> assert toktok.tokenize(text, return_str=True) == expected
    >>> toktok.tokenize(text) == [u'\xa1', u'This', u',', u'is', u'a', u'sentence', u'with', u'weird', u'\xbb', u'symbols', u'\u2026', u'appearing', u'everywhere', u'\xbf']
    True
    """

    # Replace non-breaking spaces with normal spaces.
    NON_BREAKING = re.compile("\u00A0"), " "

    # Pad some funky punctuation.
    FUNKY_PUNCT_1 = re.compile(r'([،;؛¿!"\])}»›”؟¡%٪°±©®।॥…])'), r" \1 "
    # Pad more funky punctuation.
    FUNKY_PUNCT_2 = re.compile(r"([({\[“‘„‚«‹「『])"), r" \1 "
    # Pad En dash and em dash
    EN_EM_DASHES = re.compile("([–—])"), r" \1 "

    # Replace problematic character with numeric character reference.
    AMPERCENT = re.compile("& "), "&amp; "
    TAB = re.compile("\t"), " &#9; "
    PIPE = re.compile(r"\|"), " &#124; "

    # Pad numbers with commas to keep them from further tokenization.
    COMMA_IN_NUM = re.compile(r"(?<!,)([,،])(?![,\d])"), r" \1 "

    # Just pad problematic (often neurotic) hyphen/single quote, etc.
    PROB_SINGLE_QUOTES = re.compile(r"(['’`])"), r" \1 "
    # Group ` ` stupid quotes ' ' into a single token.
    STUPID_QUOTES_1 = re.compile(r" ` ` "), r" `` "
    STUPID_QUOTES_2 = re.compile(r" ' ' "), r" '' "

    # Don't tokenize period unless it ends the line and that it isn't
    # preceded by another period, e.g.
    # "something ..." -> "something ..."
    # "something." -> "something ."
    FINAL_PERIOD_1 = re.compile(r"(?<!\.)\.$"), r" ."
    # Don't tokenize period unless it ends the line eg.
    # " ... stuff." ->  "... stuff ."
    FINAL_PERIOD_2 = re.compile(r"""(?<!\.)\.\s*(["'’»›”]) *$"""), r" . \1"

    # Treat continuous commas as fake German,Czech, etc.: „
    MULTI_COMMAS = re.compile(r"(,{2,})"), r" \1 "
    # Treat continuous dashes as fake en-dash, etc.
    MULTI_DASHES = re.compile(r"(-{2,})"), r" \1 "
    # Treat multiple periods as a thing (eg. ellipsis)
    MULTI_DOTS = re.compile(r"(\.{2,})"), r" \1 "

    # This is the \p{Open_Punctuation} from Perl's perluniprops
    # see https://perldoc.perl.org/perluniprops.html
    OPEN_PUNCT = str(
        "([{\u0f3a\u0f3c\u169b\u201a\u201e\u2045\u207d"
        "\u208d\u2329\u2768\u276a\u276c\u276e\u2770\u2772"
        "\u2774\u27c5\u27e6\u27e8\u27ea\u27ec\u27ee\u2983"
        "\u2985\u2987\u2989\u298b\u298d\u298f\u2991\u2993"
        "\u2995\u2997\u29d8\u29da\u29fc\u2e22\u2e24\u2e26"
        "\u2e28\u3008\u300a\u300c\u300e\u3010\u3014\u3016"
        "\u3018\u301a\u301d\ufd3e\ufe17\ufe35\ufe37\ufe39"
        "\ufe3b\ufe3d\ufe3f\ufe41\ufe43\ufe47\ufe59\ufe5b"
        "\ufe5d\uff08\uff3b\uff5b\uff5f\uff62"
    )
    # This is the \p{Close_Punctuation} from Perl's perluniprops
    CLOSE_PUNCT = str(
        ")]}\u0f3b\u0f3d\u169c\u2046\u207e\u208e\u232a"
        "\u2769\u276b\u276d\u276f\u2771\u2773\u2775\u27c6"
        "\u27e7\u27e9\u27eb\u27ed\u27ef\u2984\u2986\u2988"
        "\u298a\u298c\u298e\u2990\u2992\u2994\u2996\u2998"
        "\u29d9\u29db\u29fd\u2e23\u2e25\u2e27\u2e29\u3009"
        "\u300b\u300d\u300f\u3011\u3015\u3017\u3019\u301b"
        "\u301e\u301f\ufd3f\ufe18\ufe36\ufe38\ufe3a\ufe3c"
        "\ufe3e\ufe40\ufe42\ufe44\ufe48\ufe5a\ufe5c\ufe5e"
        "\uff09\uff3d\uff5d\uff60\uff63"
    )
    # This is the \p{Close_Punctuation} from Perl's perluniprops
    CURRENCY_SYM = str(
        "$\xa2\xa3\xa4\xa5\u058f\u060b\u09f2\u09f3\u09fb"
        "\u0af1\u0bf9\u0e3f\u17db\u20a0\u20a1\u20a2\u20a3"
        "\u20a4\u20a5\u20a6\u20a7\u20a8\u20a9\u20aa\u20ab"
        "\u20ac\u20ad\u20ae\u20af\u20b0\u20b1\u20b2\u20b3"
        "\u20b4\u20b5\u20b6\u20b7\u20b8\u20b9\u20ba\ua838"
        "\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6"
    )

    # Pad spaces after opening punctuations.
    OPEN_PUNCT_RE = re.compile(f"([{OPEN_PUNCT}])"), r"\1 "
    # Pad spaces before closing punctuations.
    CLOSE_PUNCT_RE = re.compile(f"([{CLOSE_PUNCT}])"), r"\1 "
    # Pad spaces after currency symbols.
    CURRENCY_SYM_RE = re.compile(f"([{CURRENCY_SYM}])"), r"\1 "

    # Use for tokenizing URL-unfriendly characters: [:/?#]
    URL_FOE_1 = re.compile(r":(?!//)"), r" : "  # in perl s{:(?!//)}{ : }g;
    URL_FOE_2 = re.compile(r"\?(?!\S)"), r" ? "  # in perl s{\?(?!\S)}{ ? }g;
    # in perl: m{://} or m{\S+\.\S+/\S+} or s{/}{ / }g;
    URL_FOE_3 = re.compile(r"(:\/\/)[\S+\.\S+\/\S+][\/]"), " / "
    URL_FOE_4 = re.compile(r" /"), r" / "  # s{ /}{ / }g;

    # Left/Right strip, i.e. remove heading/trailing spaces.
    # These strip regexes should NOT be used,
    # instead use str.lstrip(), str.rstrip() or str.strip()
    # (They are kept for reference purposes to the original toktok.pl code)
    LSTRIP = re.compile(r"^ +"), ""
    RSTRIP = re.compile(r"\s+$"), "\n"
    # Merge multiple spaces.
    ONE_SPACE = re.compile(r" {2,}"), " "

    TOKTOK_REGEXES = [
        NON_BREAKING,
        FUNKY_PUNCT_1,
        URL_FOE_1,
        URL_FOE_2,
        URL_FOE_3,
        URL_FOE_4,
        AMPERCENT,
        TAB,
        PIPE,
        OPEN_PUNCT_RE,
        CLOSE_PUNCT_RE,
        MULTI_COMMAS,
        COMMA_IN_NUM,
        FINAL_PERIOD_2,
        PROB_SINGLE_QUOTES,
        STUPID_QUOTES_1,
        STUPID_QUOTES_2,
        CURRENCY_SYM_RE,
        EN_EM_DASHES,
        MULTI_DASHES,
        MULTI_DOTS,
        FINAL_PERIOD_1,
        FINAL_PERIOD_2,
        ONE_SPACE,
    ]

    def tokenize(self, text, return_str=False):
        text = str(text)  # Converts input string into unicode.
        for regexp, substitution in self.TOKTOK_REGEXES:
            text = regexp.sub(substitution, text)
        # Finally, strips heading and trailing spaces
        # and converts output string into unicode.
        text = str(text.strip())
        return text if return_str else text.split()
