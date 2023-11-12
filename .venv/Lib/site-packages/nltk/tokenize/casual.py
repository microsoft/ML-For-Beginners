#
# Natural Language Toolkit: Twitter Tokenizer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Christopher Potts <cgpotts@stanford.edu>
#         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
#         Pierpaolo Pantone <> (modifications)
#         Tom Aarsen <> (modifications)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#


"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new
domains and tasks. The basic logic is this:

1. The tuple REGEXPS defines a list of regular expression
   strings.

2. The REGEXPS strings are put, in order, into a compiled
   regular expression object called WORD_RE, under the TweetTokenizer
   class.

3. The tokenization is done by WORD_RE.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   TweetTokenizer.

4. When instantiating Tokenizer objects, there are several options:
    * preserve_case. By default, it is set to True. If it is set to
      False, then the tokenizer will downcase everything except for
      emoticons.
    * reduce_len. By default, it is set to False. It specifies whether
      to replace repeated character sequences of length 3 or greater
      with sequences of length 3.
    * strip_handles. By default, it is set to False. It specifies
      whether to remove Twitter handles of text used in the
      `tokenize` method.
    * match_phone_numbers. By default, it is set to True. It indicates
      whether the `tokenize` method should look for phone numbers.
"""


######################################################################

import html
from typing import List

import regex  # https://github.com/nltk/nltk/issues/2409

from nltk.tokenize.api import TokenizerI

######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most importantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# ToDo: Update with https://en.wikipedia.org/wiki/List_of_emoticons ?

# This particular element is used in a couple ways, so we define it
# with a name:
EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      </?3                       # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715

URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

# emoji flag sequence
# https://en.wikipedia.org/wiki/Regional_indicator_symbol
# For regex simplicity, include all possible enclosed letter pairs,
# not the ISO subset of two-letter regional indicator symbols.
# See https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Current_codes
# Future regional flag support may be handled with the regex for
# U+1F3F4 🏴 followed by emoji tag sequences:
# r'\U0001F3F4[\U000E0000-\U000E007E]{5}\U000E007F'
FLAGS = r"""
  (?:
    [\U0001F1E6-\U0001F1FF]{2}  # all enclosed letter pairs
    |
    # English flag
    \U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006e\U000E0067\U000E007F
    |
    # Scottish flag
    \U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F
    |
    # For Wales? Why Richard, it profit a man nothing to give his soul for the whole world … but for Wales!
    \U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F
  )
"""

# Regex for recognizing phone numbers:
PHONE_REGEX = r"""
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )"""

# The components of the tokenizer:
REGEXPS = (
    URLS,
    # ASCII Emoticons
    EMOTICONS,
    # HTML tags:
    r"""<[^>\s]+>""",
    # ASCII Arrows
    r"""[\-]+>|<[\-]+""",
    # Twitter username:
    r"""(?:@[\w_]+)""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # email addresses
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
    # Zero-Width-Joiner and Skin tone modifier emojis
    """.(?:
        [\U0001F3FB-\U0001F3FF]?(?:\u200d.[\U0001F3FB-\U0001F3FF]?)+
        |
        [\U0001F3FB-\U0001F3FF]
    )""",
    # flags
    FLAGS,
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

# Take the main components and add a phone regex as the second parameter
REGEXPS_PHONE = (REGEXPS[0], PHONE_REGEX, *REGEXPS[1:])

######################################################################
# TweetTokenizer.WORD_RE and TweetTokenizer.PHONE_WORD_RE represent
# the core tokenizing regexes. They are compiled lazily.

# WORD_RE performs poorly on these patterns:
HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

# The emoticon string gets its own regex so that we can preserve case for
# them as needed:
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)

# These are for regularizing HTML entities to Unicode:
ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")

# For stripping away handles from a tweet:
HANDLES_RE = regex.compile(
    r"(?<![A-Za-z0-9_!@#\$%&*])@"
    r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"
)


######################################################################
# Functions for converting html entities
######################################################################


def _str_to_unicode(text, encoding=None, errors="strict"):
    if encoding is None:
        encoding = "utf-8"
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text


def _replace_html_entities(text, keep=(), remove_illegal=True, encoding="utf-8"):
    """
    Remove entities from text by converting them to their
    corresponding unicode character.

    :param text: a unicode string or a byte string encoded in the given
    `encoding` (which defaults to 'utf-8').

    :param list keep:  list of entity names which should not be replaced.\
    This supports both numeric entities (``&#nnnn;`` and ``&#hhhh;``)
    and named entities (such as ``&nbsp;`` or ``&gt;``).

    :param bool remove_illegal: If `True`, entities that can't be converted are\
    removed. Otherwise, entities that can't be converted are kept "as
    is".

    :returns: A unicode string with the entities removed.

    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py

        >>> from nltk.tokenize.casual import _replace_html_entities
        >>> _replace_html_entities(b'Price: &pound;100')
        'Price: \\xa3100'
        >>> print(_replace_html_entities(b'Price: &pound;100'))
        Price: £100
        >>>
    """

    def _convert_entity(match):
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                # Numeric character references in the 80-9F range are typically
                # interpreted by browsers as representing the characters mapped
                # to bytes 80-9F in the Windows-1252 encoding. For more info
                # see: https://en.wikipedia.org/wiki/ISO/IEC_8859-1#Similar_character_sets
                if 0x80 <= number <= 0x9F:
                    return bytes((number,)).decode("cp1252")
            except ValueError:
                number = None
        else:
            if entity_body in keep:
                return match.group(0)
            number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass

        return "" if remove_illegal else match.group(0)

    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))


######################################################################


class TweetTokenizer(TokenizerI):
    r"""
    Tokenizer for tweets.

        >>> from nltk.tokenize import TweetTokenizer
        >>> tknzr = TweetTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0) # doctest: +NORMALIZE_WHITESPACE
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->',
         '<--']

    Examples using `strip_handles` and `reduce_len parameters`:

        >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
        >>> tknzr.tokenize(s1)
        [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    """

    # Values used to lazily compile WORD_RE and PHONE_WORD_RE,
    # which are the core tokenizing regexes.
    _WORD_RE = None
    _PHONE_WORD_RE = None

    ######################################################################

    def __init__(
        self,
        preserve_case=True,
        reduce_len=False,
        strip_handles=False,
        match_phone_numbers=True,
    ):
        """
        Create a `TweetTokenizer` instance with settings for use in the `tokenize` method.

        :param preserve_case: Flag indicating whether to preserve the casing (capitalisation)
            of text used in the `tokenize` method. Defaults to True.
        :type preserve_case: bool
        :param reduce_len: Flag indicating whether to replace repeated character sequences
            of length 3 or greater with sequences of length 3. Defaults to False.
        :type reduce_len: bool
        :param strip_handles: Flag indicating whether to remove Twitter handles of text used
            in the `tokenize` method. Defaults to False.
        :type strip_handles: bool
        :param match_phone_numbers: Flag indicating whether the `tokenize` method should look
            for phone numbers. Defaults to True.
        :type match_phone_numbers: bool
        """
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles
        self.match_phone_numbers = match_phone_numbers

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.

        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; joining this list returns\
        the original string if `preserve_case=False`.
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove username handles
        if self.strip_handles:
            text = remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # Recognise phone numbers during tokenization
        if self.match_phone_numbers:
            words = self.PHONE_WORD_RE.findall(safe_text)
        else:
            words = self.WORD_RE.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(
                map((lambda x: x if EMOTICON_RE.search(x) else x.lower()), words)
            )
        return words

    @property
    def WORD_RE(self) -> "regex.Pattern":
        """Core TweetTokenizer regex"""
        # Compiles the regex for this and all future instantiations of TweetTokenizer.
        if not type(self)._WORD_RE:
            type(self)._WORD_RE = regex.compile(
                f"({'|'.join(REGEXPS)})",
                regex.VERBOSE | regex.I | regex.UNICODE,
            )
        return type(self)._WORD_RE

    @property
    def PHONE_WORD_RE(self) -> "regex.Pattern":
        """Secondary core TweetTokenizer regex"""
        # Compiles the regex for this and all future instantiations of TweetTokenizer.
        if not type(self)._PHONE_WORD_RE:
            type(self)._PHONE_WORD_RE = regex.compile(
                f"({'|'.join(REGEXPS_PHONE)})",
                regex.VERBOSE | regex.I | regex.UNICODE,
            )
        return type(self)._PHONE_WORD_RE


######################################################################
# Normalization Functions
######################################################################


def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


def remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    # Substitute handles with ' ' to ensure that text on either side of removed handles are tokenized correctly
    return HANDLES_RE.sub(" ", text)


######################################################################
# Tokenization Function
######################################################################


def casual_tokenize(
    text,
    preserve_case=True,
    reduce_len=False,
    strip_handles=False,
    match_phone_numbers=True,
):
    """
    Convenience function for wrapping the tokenizer.
    """
    return TweetTokenizer(
        preserve_case=preserve_case,
        reduce_len=reduce_len,
        strip_handles=strip_handles,
        match_phone_numbers=match_phone_numbers,
    ).tokenize(text)


###############################################################################
