# Natural Language Toolkit: Punkt sentence tokenizer
#
# Copyright (C) 2001-2023 NLTK Project
# Algorithm: Kiss & Strunk (2006)
# Author: Willy <willy@csse.unimelb.edu.au> (original Python port)
#         Steven Bird <stevenbird1@gmail.com> (additions)
#         Edward Loper <edloper@gmail.com> (rewrite)
#         Joel Nothman <jnothman@student.usyd.edu.au> (almost rewrite)
#         Arthur Darcet <arthur@darcet.fr> (fixes)
#         Tom Aarsen <> (tackle ReDoS & performance issues)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

r"""
Punkt Sentence Tokenizer

This tokenizer divides a text into a list of sentences
by using an unsupervised algorithm to build a model for abbreviation
words, collocations, and words that start sentences.  It must be
trained on a large collection of plaintext in the target language
before it can be used.

The NLTK data package includes a pre-trained Punkt tokenizer for
English.

    >>> import nltk.data
    >>> text = '''
    ... Punkt knows that the periods in Mr. Smith and Johann S. Bach
    ... do not mark sentence boundaries.  And sometimes sentences
    ... can start with non-capitalized words.  i is a good variable
    ... name.
    ... '''
    >>> sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    >>> print('\n-----\n'.join(sent_detector.tokenize(text.strip())))
    Punkt knows that the periods in Mr. Smith and Johann S. Bach
    do not mark sentence boundaries.
    -----
    And sometimes sentences
    can start with non-capitalized words.
    -----
    i is a good variable
    name.

(Note that whitespace from the original text, including newlines, is
retained in the output.)

Punctuation following sentences is also included by default
(from NLTK 3.0 onwards). It can be excluded with the realign_boundaries
flag.

    >>> text = '''
    ... (How does it deal with this parenthesis?)  "It should be part of the
    ... previous sentence." "(And the same with this one.)" ('And this one!')
    ... "('(And (this)) '?)" [(and this. )]
    ... '''
    >>> print('\n-----\n'.join(
    ...     sent_detector.tokenize(text.strip())))
    (How does it deal with this parenthesis?)
    -----
    "It should be part of the
    previous sentence."
    -----
    "(And the same with this one.)"
    -----
    ('And this one!')
    -----
    "('(And (this)) '?)"
    -----
    [(and this. )]
    >>> print('\n-----\n'.join(
    ...     sent_detector.tokenize(text.strip(), realign_boundaries=False)))
    (How does it deal with this parenthesis?
    -----
    )  "It should be part of the
    previous sentence.
    -----
    " "(And the same with this one.
    -----
    )" ('And this one!
    -----
    ')
    "('(And (this)) '?
    -----
    )" [(and this.
    -----
    )]

However, Punkt is designed to learn parameters (a list of abbreviations, etc.)
unsupervised from a corpus similar to the target domain. The pre-packaged models
may therefore be unsuitable: use ``PunktSentenceTokenizer(text)`` to learn
parameters from the given text.

:class:`.PunktTrainer` learns parameters such as a list of abbreviations
(without supervision) from portions of text. Using a ``PunktTrainer`` directly
allows for incremental training and modification of the hyper-parameters used
to decide what is considered an abbreviation, etc.

The algorithm for this tokenizer is described in::

  Kiss, Tibor and Strunk, Jan (2006): Unsupervised Multilingual Sentence
    Boundary Detection.  Computational Linguistics 32: 485-525.
"""

# TODO: Make orthographic heuristic less susceptible to overtraining
# TODO: Frequent sentence starters optionally exclude always-capitalised words
# FIXME: Problem with ending string with e.g. '!!!' -> '!! !'

import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union

from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI

######################################################################
# { Orthographic Context Constants
######################################################################
# The following constants are used to describe the orthographic
# contexts in which a word can occur.  BEG=beginning, MID=middle,
# UNK=unknown, UC=uppercase, LC=lowercase, NC=no case.

_ORTHO_BEG_UC = 1 << 1
"""Orthographic context: beginning of a sentence with upper case."""

_ORTHO_MID_UC = 1 << 2
"""Orthographic context: middle of a sentence with upper case."""

_ORTHO_UNK_UC = 1 << 3
"""Orthographic context: unknown position in a sentence with upper case."""

_ORTHO_BEG_LC = 1 << 4
"""Orthographic context: beginning of a sentence with lower case."""

_ORTHO_MID_LC = 1 << 5
"""Orthographic context: middle of a sentence with lower case."""

_ORTHO_UNK_LC = 1 << 6
"""Orthographic context: unknown position in a sentence with lower case."""

_ORTHO_UC = _ORTHO_BEG_UC + _ORTHO_MID_UC + _ORTHO_UNK_UC
"""Orthographic context: occurs with upper case."""

_ORTHO_LC = _ORTHO_BEG_LC + _ORTHO_MID_LC + _ORTHO_UNK_LC
"""Orthographic context: occurs with lower case."""

_ORTHO_MAP = {
    ("initial", "upper"): _ORTHO_BEG_UC,
    ("internal", "upper"): _ORTHO_MID_UC,
    ("unknown", "upper"): _ORTHO_UNK_UC,
    ("initial", "lower"): _ORTHO_BEG_LC,
    ("internal", "lower"): _ORTHO_MID_LC,
    ("unknown", "lower"): _ORTHO_UNK_LC,
}
"""A map from context position and first-letter case to the
appropriate orthographic context flag."""

# } (end orthographic context constants)
######################################################################

######################################################################
# { Decision reasons for debugging
######################################################################

REASON_DEFAULT_DECISION = "default decision"
REASON_KNOWN_COLLOCATION = "known collocation (both words)"
REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC = "abbreviation + orthographic heuristic"
REASON_ABBR_WITH_SENTENCE_STARTER = "abbreviation + frequent sentence starter"
REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC = (
    "initial + special orthographic heuristic"
)


# } (end decision reasons for debugging)
######################################################################

######################################################################
# { Language-dependent variables
######################################################################


class PunktLanguageVars:
    """
    Stores variables, mostly regular expressions, which may be
    language-dependent for correct application of the algorithm.
    An extension of this class may modify its properties to suit
    a language other than English; an instance can then be passed
    as an argument to PunktSentenceTokenizer and PunktTrainer
    constructors.
    """

    __slots__ = ("_re_period_context", "_re_word_tokenizer")

    def __getstate__(self):
        # All modifications to the class are performed by inheritance.
        # Non-default parameters to be pickled must be defined in the inherited
        # class.
        return 1

    def __setstate__(self, state):
        return 1

    sent_end_chars = (".", "?", "!")
    """Characters which are candidates for sentence boundaries"""

    @property
    def _re_sent_end_chars(self):
        return "[%s]" % re.escape("".join(self.sent_end_chars))

    internal_punctuation = ",:;"  # might want to extend this..
    """sentence internal punctuation, which indicates an abbreviation if
    preceded by a period-final token."""

    re_boundary_realignment = re.compile(r'["\')\]}]+?(?:\s+|(?=--)|$)', re.MULTILINE)
    """Used to realign punctuation that should be included in a sentence
    although it follows the period (or ?, !)."""

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    """Excludes some characters from starting word tokens"""

    @property
    def _re_non_word_chars(self):
        return r"(?:[)\";}\]\*:@\'\({\[%s])" % re.escape(
            "".join(set(self.sent_end_chars) - {"."})
        )

    """Characters that cannot appear within words"""

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""
    """Format of a regular expression to split punctuation from words,
    excluding period."""

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        """Tokenize a string to split off punctuation other than periods"""
        return self._word_tokenizer_re().findall(s)

    _period_context_fmt = r"""
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""
    """Format of a regular expression to find contexts including possible
    sentence boundaries. Matches token which the possible sentence boundary
    ends, and matches the following token within a lookahead expression."""

    def period_context_re(self):
        """Compiles and returns a regular expression to find contexts
        including possible sentence boundaries."""
        try:
            return self._re_period_context
        except:
            self._re_period_context = re.compile(
                self._period_context_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "SentEndChars": self._re_sent_end_chars,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_period_context


_re_non_punct = re.compile(r"[^\W\d]", re.UNICODE)
"""Matches token types that are not merely punctuation. (Types for
numeric tokens are changed to ##number## and hence contain alpha.)"""


# }
######################################################################


# ////////////////////////////////////////////////////////////
# { Helper Functions
# ////////////////////////////////////////////////////////////


def _pair_iter(iterator):
    """
    Yields pairs of tokens from the given iterator such that each input
    token will appear as the first element in a yielded tuple. The last
    pair will have None as its second element.
    """
    iterator = iter(iterator)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    for el in iterator:
        yield (prev, el)
        prev = el
    yield (prev, None)


######################################################################
# { Punkt Parameters
######################################################################


class PunktParameters:
    """Stores data used to perform sentence boundary detection with Punkt."""

    def __init__(self):
        self.abbrev_types = set()
        """A set of word types for known abbreviations."""

        self.collocations = set()
        """A set of word type tuples for known common collocations
        where the first word ends in a period.  E.g., ('S.', 'Bach')
        is a common collocation in a text that discusses 'Johann
        S. Bach'.  These count as negative evidence for sentence
        boundaries."""

        self.sent_starters = set()
        """A set of word types for words that often appear at the
        beginning of sentences."""

        self.ortho_context = defaultdict(int)
        """A dictionary mapping word types to the set of orthographic
        contexts that word type appears in.  Contexts are represented
        by adding orthographic context flags: ..."""

    def clear_abbrevs(self):
        self.abbrev_types = set()

    def clear_collocations(self):
        self.collocations = set()

    def clear_sent_starters(self):
        self.sent_starters = set()

    def clear_ortho_context(self):
        self.ortho_context = defaultdict(int)

    def add_ortho_context(self, typ, flag):
        self.ortho_context[typ] |= flag

    def _debug_ortho_context(self, typ):
        context = self.ortho_context[typ]
        if context & _ORTHO_BEG_UC:
            yield "BEG-UC"
        if context & _ORTHO_MID_UC:
            yield "MID-UC"
        if context & _ORTHO_UNK_UC:
            yield "UNK-UC"
        if context & _ORTHO_BEG_LC:
            yield "BEG-LC"
        if context & _ORTHO_MID_LC:
            yield "MID-LC"
        if context & _ORTHO_UNK_LC:
            yield "UNK-LC"


######################################################################
# { PunktToken
######################################################################


class PunktToken:
    """Stores a token of text with annotations produced during
    sentence boundary detection."""

    _properties = ["parastart", "linestart", "sentbreak", "abbr", "ellipsis"]
    __slots__ = ["tok", "type", "period_final"] + _properties

    def __init__(self, tok, **params):
        self.tok = tok
        self.type = self._get_type(tok)
        self.period_final = tok.endswith(".")

        for prop in self._properties:
            setattr(self, prop, None)
        for k in params:
            setattr(self, k, params[k])

    # ////////////////////////////////////////////////////////////
    # { Regular expressions for properties
    # ////////////////////////////////////////////////////////////
    # Note: [A-Za-z] is approximated by [^\W\d] in the general case.
    _RE_ELLIPSIS = re.compile(r"\.\.+$")
    _RE_NUMERIC = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")
    _RE_INITIAL = re.compile(r"[^\W\d]\.$", re.UNICODE)
    _RE_ALPHA = re.compile(r"[^\W\d]+$", re.UNICODE)

    # ////////////////////////////////////////////////////////////
    # { Derived properties
    # ////////////////////////////////////////////////////////////

    def _get_type(self, tok):
        """Returns a case-normalized representation of the token."""
        return self._RE_NUMERIC.sub("##number##", tok.lower())

    @property
    def type_no_period(self):
        """
        The type with its final period removed if it has one.
        """
        if len(self.type) > 1 and self.type[-1] == ".":
            return self.type[:-1]
        return self.type

    @property
    def type_no_sentperiod(self):
        """
        The type with its final period removed if it is marked as a
        sentence break.
        """
        if self.sentbreak:
            return self.type_no_period
        return self.type

    @property
    def first_upper(self):
        """True if the token's first character is uppercase."""
        return self.tok[0].isupper()

    @property
    def first_lower(self):
        """True if the token's first character is lowercase."""
        return self.tok[0].islower()

    @property
    def first_case(self):
        if self.first_lower:
            return "lower"
        if self.first_upper:
            return "upper"
        return "none"

    @property
    def is_ellipsis(self):
        """True if the token text is that of an ellipsis."""
        return self._RE_ELLIPSIS.match(self.tok)

    @property
    def is_number(self):
        """True if the token text is that of a number."""
        return self.type.startswith("##number##")

    @property
    def is_initial(self):
        """True if the token text is that of an initial."""
        return self._RE_INITIAL.match(self.tok)

    @property
    def is_alpha(self):
        """True if the token text is all alphabetic."""
        return self._RE_ALPHA.match(self.tok)

    @property
    def is_non_punct(self):
        """True if the token is either a number or is alphabetic."""
        return _re_non_punct.search(self.type)

    # ////////////////////////////////////////////////////////////
    # { String representation
    # ////////////////////////////////////////////////////////////

    def __repr__(self):
        """
        A string representation of the token that can reproduce it
        with eval(), which lists all the token's non-default
        annotations.
        """
        typestr = " type=%s," % repr(self.type) if self.type != self.tok else ""

        propvals = ", ".join(
            f"{p}={repr(getattr(self, p))}"
            for p in self._properties
            if getattr(self, p)
        )

        return "{}({},{} {})".format(
            self.__class__.__name__,
            repr(self.tok),
            typestr,
            propvals,
        )

    def __str__(self):
        """
        A string representation akin to that used by Kiss and Strunk.
        """
        res = self.tok
        if self.abbr:
            res += "<A>"
        if self.ellipsis:
            res += "<E>"
        if self.sentbreak:
            res += "<S>"
        return res


######################################################################
# { Punkt base class
######################################################################


class PunktBaseClass:
    """
    Includes common components of PunktTrainer and PunktSentenceTokenizer.
    """

    def __init__(self, lang_vars=None, token_cls=PunktToken, params=None):
        if lang_vars is None:
            lang_vars = PunktLanguageVars()
        if params is None:
            params = PunktParameters()
        self._params = params
        self._lang_vars = lang_vars
        self._Token = token_cls
        """The collection of parameters that determines the behavior
        of the punkt tokenizer."""

    # ////////////////////////////////////////////////////////////
    # { Word tokenization
    # ////////////////////////////////////////////////////////////

    def _tokenize_words(self, plaintext):
        """
        Divide the given text into tokens, using the punkt word
        segmentation regular expression, and generate the resulting list
        of tokens augmented as three-tuples with two boolean values for whether
        the given token occurs at the start of a paragraph or a new line,
        respectively.
        """
        parastart = False
        for line in plaintext.split("\n"):
            if line.strip():
                line_toks = iter(self._lang_vars.word_tokenize(line))

                try:
                    tok = next(line_toks)
                except StopIteration:
                    continue

                yield self._Token(tok, parastart=parastart, linestart=True)
                parastart = False

                for tok in line_toks:
                    yield self._Token(tok)
            else:
                parastart = True

    # ////////////////////////////////////////////////////////////
    # { Annotation Procedures
    # ////////////////////////////////////////////////////////////

    def _annotate_first_pass(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        """
        Perform the first pass of annotation, which makes decisions
        based purely based on the word type of each word:

          - '?', '!', and '.' are marked as sentence breaks.
          - sequences of two or more periods are marked as ellipsis.
          - any word ending in '.' that's a known abbreviation is
            marked as an abbreviation.
          - any other word ending in '.' is marked as a sentence break.

        Return these annotations as a tuple of three sets:

          - sentbreak_toks: The indices of all sentence breaks.
          - abbrev_toks: The indices of all abbreviations.
          - ellipsis_toks: The indices of all ellipsis marks.
        """
        for aug_tok in tokens:
            self._first_pass_annotation(aug_tok)
            yield aug_tok

    def _first_pass_annotation(self, aug_tok: PunktToken) -> None:
        """
        Performs type-based annotation on a single token.
        """

        tok = aug_tok.tok

        if tok in self._lang_vars.sent_end_chars:
            aug_tok.sentbreak = True
        elif aug_tok.is_ellipsis:
            aug_tok.ellipsis = True
        elif aug_tok.period_final and not tok.endswith(".."):
            if (
                tok[:-1].lower() in self._params.abbrev_types
                or tok[:-1].lower().split("-")[-1] in self._params.abbrev_types
            ):

                aug_tok.abbr = True
            else:
                aug_tok.sentbreak = True

        return


######################################################################
# { Punkt Trainer
######################################################################


class PunktTrainer(PunktBaseClass):
    """Learns parameters used in Punkt sentence boundary detection."""

    def __init__(
        self, train_text=None, verbose=False, lang_vars=None, token_cls=PunktToken
    ):

        PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)

        self._type_fdist = FreqDist()
        """A frequency distribution giving the frequency of each
        case-normalized token type in the training data."""

        self._num_period_toks = 0
        """The number of words ending in period in the training data."""

        self._collocation_fdist = FreqDist()
        """A frequency distribution giving the frequency of all
        bigrams in the training data where the first word ends in a
        period.  Bigrams are encoded as tuples of word types.
        Especially common collocations are extracted from this
        frequency distribution, and stored in
        ``_params``.``collocations <PunktParameters.collocations>``."""

        self._sent_starter_fdist = FreqDist()
        """A frequency distribution giving the frequency of all words
        that occur at the training data at the beginning of a sentence
        (after the first pass of annotation).  Especially common
        sentence starters are extracted from this frequency
        distribution, and stored in ``_params.sent_starters``.
        """

        self._sentbreak_count = 0
        """The total number of sentence breaks identified in training, used for
        calculating the frequent sentence starter heuristic."""

        self._finalized = True
        """A flag as to whether the training has been finalized by finding
        collocations and sentence starters, or whether finalize_training()
        still needs to be called."""

        if train_text:
            self.train(train_text, verbose, finalize=True)

    def get_params(self):
        """
        Calculates and returns parameters for sentence boundary detection as
        derived from training."""
        if not self._finalized:
            self.finalize_training()
        return self._params

    # ////////////////////////////////////////////////////////////
    # { Customization Variables
    # ////////////////////////////////////////////////////////////

    ABBREV = 0.3
    """cut-off value whether a 'token' is an abbreviation"""

    IGNORE_ABBREV_PENALTY = False
    """allows the disabling of the abbreviation penalty heuristic, which
    exponentially disadvantages words that are found at times without a
    final period."""

    ABBREV_BACKOFF = 5
    """upper cut-off for Mikheev's(2002) abbreviation detection algorithm"""

    COLLOCATION = 7.88
    """minimal log-likelihood value that two tokens need to be considered
    as a collocation"""

    SENT_STARTER = 30
    """minimal log-likelihood value that a token requires to be considered
    as a frequent sentence starter"""

    INCLUDE_ALL_COLLOCS = False
    """this includes as potential collocations all word pairs where the first
    word ends in a period. It may be useful in corpora where there is a lot
    of variation that makes abbreviations like Mr difficult to identify."""

    INCLUDE_ABBREV_COLLOCS = False
    """this includes as potential collocations all word pairs where the first
    word is an abbreviation. Such collocations override the orthographic
    heuristic, but not the sentence starter heuristic. This is overridden by
    INCLUDE_ALL_COLLOCS, and if both are false, only collocations with initials
    and ordinals are considered."""
    """"""

    MIN_COLLOC_FREQ = 1
    """this sets a minimum bound on the number of times a bigram needs to
    appear before it can be considered a collocation, in addition to log
    likelihood statistics. This is useful when INCLUDE_ALL_COLLOCS is True."""

    # ////////////////////////////////////////////////////////////
    # { Training..
    # ////////////////////////////////////////////////////////////

    def train(self, text, verbose=False, finalize=True):
        """
        Collects training data from a given text. If finalize is True, it
        will determine all the parameters for sentence boundary detection. If
        not, this will be delayed until get_params() or finalize_training() is
        called. If verbose is True, abbreviations found will be listed.
        """
        # Break the text into tokens; record which token indices correspond to
        # line starts and paragraph starts; and determine their types.
        self._train_tokens(self._tokenize_words(text), verbose)
        if finalize:
            self.finalize_training(verbose)

    def train_tokens(self, tokens, verbose=False, finalize=True):
        """
        Collects training data from a given list of tokens.
        """
        self._train_tokens((self._Token(t) for t in tokens), verbose)
        if finalize:
            self.finalize_training(verbose)

    def _train_tokens(self, tokens, verbose):
        self._finalized = False

        # Ensure tokens are a list
        tokens = list(tokens)

        # Find the frequency of each case-normalized type.  (Don't
        # strip off final periods.)  Also keep track of the number of
        # tokens that end in periods.
        for aug_tok in tokens:
            self._type_fdist[aug_tok.type] += 1
            if aug_tok.period_final:
                self._num_period_toks += 1

        # Look for new abbreviations, and for types that no longer are
        unique_types = self._unique_types(tokens)
        for abbr, score, is_add in self._reclassify_abbrev_types(unique_types):
            if score >= self.ABBREV:
                if is_add:
                    self._params.abbrev_types.add(abbr)
                    if verbose:
                        print(f"  Abbreviation: [{score:6.4f}] {abbr}")
            else:
                if not is_add:
                    self._params.abbrev_types.remove(abbr)
                    if verbose:
                        print(f"  Removed abbreviation: [{score:6.4f}] {abbr}")

        # Make a preliminary pass through the document, marking likely
        # sentence breaks, abbreviations, and ellipsis tokens.
        tokens = list(self._annotate_first_pass(tokens))

        # Check what contexts each word type can appear in, given the
        # case of its first letter.
        self._get_orthography_data(tokens)

        # We need total number of sentence breaks to find sentence starters
        self._sentbreak_count += self._get_sentbreak_count(tokens)

        # The remaining heuristics relate to pairs of tokens where the first
        # ends in a period.
        for aug_tok1, aug_tok2 in _pair_iter(tokens):
            if not aug_tok1.period_final or not aug_tok2:
                continue

            # Is the first token a rare abbreviation?
            if self._is_rare_abbrev_type(aug_tok1, aug_tok2):
                self._params.abbrev_types.add(aug_tok1.type_no_period)
                if verbose:
                    print("  Rare Abbrev: %s" % aug_tok1.type)

            # Does second token have a high likelihood of starting a sentence?
            if self._is_potential_sent_starter(aug_tok2, aug_tok1):
                self._sent_starter_fdist[aug_tok2.type] += 1

            # Is this bigram a potential collocation?
            if self._is_potential_collocation(aug_tok1, aug_tok2):
                self._collocation_fdist[
                    (aug_tok1.type_no_period, aug_tok2.type_no_sentperiod)
                ] += 1

    def _unique_types(self, tokens):
        return {aug_tok.type for aug_tok in tokens}

    def finalize_training(self, verbose=False):
        """
        Uses data that has been gathered in training to determine likely
        collocations and sentence starters.
        """
        self._params.clear_sent_starters()
        for typ, log_likelihood in self._find_sent_starters():
            self._params.sent_starters.add(typ)
            if verbose:
                print(f"  Sent Starter: [{log_likelihood:6.4f}] {typ!r}")

        self._params.clear_collocations()
        for (typ1, typ2), log_likelihood in self._find_collocations():
            self._params.collocations.add((typ1, typ2))
            if verbose:
                print(f"  Collocation: [{log_likelihood:6.4f}] {typ1!r}+{typ2!r}")

        self._finalized = True

    # ////////////////////////////////////////////////////////////
    # { Overhead reduction
    # ////////////////////////////////////////////////////////////

    def freq_threshold(
        self, ortho_thresh=2, type_thresh=2, colloc_thres=2, sentstart_thresh=2
    ):
        """
        Allows memory use to be reduced after much training by removing data
        about rare tokens that are unlikely to have a statistical effect with
        further training. Entries occurring above the given thresholds will be
        retained.
        """
        if ortho_thresh > 1:
            old_oc = self._params.ortho_context
            self._params.clear_ortho_context()
            for tok in self._type_fdist:
                count = self._type_fdist[tok]
                if count >= ortho_thresh:
                    self._params.ortho_context[tok] = old_oc[tok]

        self._type_fdist = self._freq_threshold(self._type_fdist, type_thresh)
        self._collocation_fdist = self._freq_threshold(
            self._collocation_fdist, colloc_thres
        )
        self._sent_starter_fdist = self._freq_threshold(
            self._sent_starter_fdist, sentstart_thresh
        )

    def _freq_threshold(self, fdist, threshold):
        """
        Returns a FreqDist containing only data with counts below a given
        threshold, as well as a mapping (None -> count_removed).
        """
        # We assume that there is more data below the threshold than above it
        # and so create a new FreqDist rather than working in place.
        res = FreqDist()
        num_removed = 0
        for tok in fdist:
            count = fdist[tok]
            if count < threshold:
                num_removed += 1
            else:
                res[tok] += count
        res[None] += num_removed
        return res

    # ////////////////////////////////////////////////////////////
    # { Orthographic data
    # ////////////////////////////////////////////////////////////

    def _get_orthography_data(self, tokens):
        """
        Collect information about whether each token type occurs
        with different case patterns (i) overall, (ii) at
        sentence-initial positions, and (iii) at sentence-internal
        positions.
        """
        # 'initial' or 'internal' or 'unknown'
        context = "internal"
        tokens = list(tokens)

        for aug_tok in tokens:
            # If we encounter a paragraph break, then it's a good sign
            # that it's a sentence break.  But err on the side of
            # caution (by not positing a sentence break) if we just
            # saw an abbreviation.
            if aug_tok.parastart and context != "unknown":
                context = "initial"

            # If we're at the beginning of a line, then we can't decide
            # between 'internal' and 'initial'.
            if aug_tok.linestart and context == "internal":
                context = "unknown"

            # Find the case-normalized type of the token.  If it's a
            # sentence-final token, strip off the period.
            typ = aug_tok.type_no_sentperiod

            # Update the orthographic context table.
            flag = _ORTHO_MAP.get((context, aug_tok.first_case), 0)
            if flag:
                self._params.add_ortho_context(typ, flag)

            # Decide whether the next word is at a sentence boundary.
            if aug_tok.sentbreak:
                if not (aug_tok.is_number or aug_tok.is_initial):
                    context = "initial"
                else:
                    context = "unknown"
            elif aug_tok.ellipsis or aug_tok.abbr:
                context = "unknown"
            else:
                context = "internal"

    # ////////////////////////////////////////////////////////////
    # { Abbreviations
    # ////////////////////////////////////////////////////////////

    def _reclassify_abbrev_types(self, types):
        """
        (Re)classifies each given token if
          - it is period-final and not a known abbreviation; or
          - it is not period-final and is otherwise a known abbreviation
        by checking whether its previous classification still holds according
        to the heuristics of section 3.
        Yields triples (abbr, score, is_add) where abbr is the type in question,
        score is its log-likelihood with penalties applied, and is_add specifies
        whether the present type is a candidate for inclusion or exclusion as an
        abbreviation, such that:
          - (is_add and score >= 0.3)    suggests a new abbreviation; and
          - (not is_add and score < 0.3) suggests excluding an abbreviation.
        """
        # (While one could recalculate abbreviations from all .-final tokens at
        # every iteration, in cases requiring efficiency, the number of tokens
        # in the present training document will be much less.)

        for typ in types:
            # Check some basic conditions, to rule out words that are
            # clearly not abbrev_types.
            if not _re_non_punct.search(typ) or typ == "##number##":
                continue

            if typ.endswith("."):
                if typ in self._params.abbrev_types:
                    continue
                typ = typ[:-1]
                is_add = True
            else:
                if typ not in self._params.abbrev_types:
                    continue
                is_add = False

            # Count how many periods & nonperiods are in the
            # candidate.
            num_periods = typ.count(".") + 1
            num_nonperiods = len(typ) - num_periods + 1

            # Let <a> be the candidate without the period, and <b>
            # be the period.  Find a log likelihood ratio that
            # indicates whether <ab> occurs as a single unit (high
            # value of log_likelihood), or as two independent units <a> and
            # <b> (low value of log_likelihood).
            count_with_period = self._type_fdist[typ + "."]
            count_without_period = self._type_fdist[typ]
            log_likelihood = self._dunning_log_likelihood(
                count_with_period + count_without_period,
                self._num_period_toks,
                count_with_period,
                self._type_fdist.N(),
            )

            # Apply three scaling factors to 'tweak' the basic log
            # likelihood ratio:
            #   F_length: long word -> less likely to be an abbrev
            #   F_periods: more periods -> more likely to be an abbrev
            #   F_penalty: penalize occurrences w/o a period
            f_length = math.exp(-num_nonperiods)
            f_periods = num_periods
            f_penalty = int(self.IGNORE_ABBREV_PENALTY) or math.pow(
                num_nonperiods, -count_without_period
            )
            score = log_likelihood * f_length * f_periods * f_penalty

            yield typ, score, is_add

    def find_abbrev_types(self):
        """
        Recalculates abbreviations given type frequencies, despite no prior
        determination of abbreviations.
        This fails to include abbreviations otherwise found as "rare".
        """
        self._params.clear_abbrevs()
        tokens = (typ for typ in self._type_fdist if typ and typ.endswith("."))
        for abbr, score, _is_add in self._reclassify_abbrev_types(tokens):
            if score >= self.ABBREV:
                self._params.abbrev_types.add(abbr)

    # This function combines the work done by the original code's
    # functions `count_orthography_context`, `get_orthography_count`,
    # and `get_rare_abbreviations`.
    def _is_rare_abbrev_type(self, cur_tok, next_tok):
        """
        A word type is counted as a rare abbreviation if...
          - it's not already marked as an abbreviation
          - it occurs fewer than ABBREV_BACKOFF times
          - either it is followed by a sentence-internal punctuation
            mark, *or* it is followed by a lower-case word that
            sometimes appears with upper case, but never occurs with
            lower case at the beginning of sentences.
        """
        if cur_tok.abbr or not cur_tok.sentbreak:
            return False

        # Find the case-normalized type of the token.  If it's
        # a sentence-final token, strip off the period.
        typ = cur_tok.type_no_sentperiod

        # Proceed only if the type hasn't been categorized as an
        # abbreviation already, and is sufficiently rare...
        count = self._type_fdist[typ] + self._type_fdist[typ[:-1]]
        if typ in self._params.abbrev_types or count >= self.ABBREV_BACKOFF:
            return False

        # Record this token as an abbreviation if the next
        # token is a sentence-internal punctuation mark.
        # [XX] :1 or check the whole thing??
        if next_tok.tok[:1] in self._lang_vars.internal_punctuation:
            return True

        # Record this type as an abbreviation if the next
        # token...  (i) starts with a lower case letter,
        # (ii) sometimes occurs with an uppercase letter,
        # and (iii) never occus with an uppercase letter
        # sentence-internally.
        # [xx] should the check for (ii) be modified??
        if next_tok.first_lower:
            typ2 = next_tok.type_no_sentperiod
            typ2ortho_context = self._params.ortho_context[typ2]
            if (typ2ortho_context & _ORTHO_BEG_UC) and not (
                typ2ortho_context & _ORTHO_MID_UC
            ):
                return True

    # ////////////////////////////////////////////////////////////
    # { Log Likelihoods
    # ////////////////////////////////////////////////////////////

    # helper for _reclassify_abbrev_types:
    @staticmethod
    def _dunning_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that calculates the modified Dunning log-likelihood
        ratio scores for abbreviation candidates.  The details of how
        this works is available in the paper.
        """
        p1 = count_b / N
        p2 = 0.99

        null_hypo = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
        alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)

        likelihood = null_hypo - alt_hypo

        return -2.0 * likelihood

    @staticmethod
    def _col_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that will just compute log-likelihood estimate, in
        the original paper it's described in algorithm 6 and 7.

        This *should* be the original Dunning log-likelihood values,
        unlike the previous log_l function where it used modified
        Dunning log-likelihood values
        """
        p = count_b / N
        p1 = count_ab / count_a
        try:
            p2 = (count_b - count_ab) / (N - count_a)
        except ZeroDivisionError:
            p2 = 1

        try:
            summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)
        except ValueError:
            summand1 = 0

        try:
            summand2 = (count_b - count_ab) * math.log(p) + (
                N - count_a - count_b + count_ab
            ) * math.log(1.0 - p)
        except ValueError:
            summand2 = 0

        if count_a == count_ab or p1 <= 0 or p1 >= 1:
            summand3 = 0
        else:
            summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(
                1.0 - p1
            )

        if count_b == count_ab or p2 <= 0 or p2 >= 1:
            summand4 = 0
        else:
            summand4 = (count_b - count_ab) * math.log(p2) + (
                N - count_a - count_b + count_ab
            ) * math.log(1.0 - p2)

        likelihood = summand1 + summand2 - summand3 - summand4

        return -2.0 * likelihood

    # ////////////////////////////////////////////////////////////
    # { Collocation Finder
    # ////////////////////////////////////////////////////////////

    def _is_potential_collocation(self, aug_tok1, aug_tok2):
        """
        Returns True if the pair of tokens may form a collocation given
        log-likelihood statistics.
        """
        return (
            (
                self.INCLUDE_ALL_COLLOCS
                or (self.INCLUDE_ABBREV_COLLOCS and aug_tok1.abbr)
                or (aug_tok1.sentbreak and (aug_tok1.is_number or aug_tok1.is_initial))
            )
            and aug_tok1.is_non_punct
            and aug_tok2.is_non_punct
        )

    def _find_collocations(self):
        """
        Generates likely collocations and their log-likelihood.
        """
        for types in self._collocation_fdist:
            try:
                typ1, typ2 = types
            except TypeError:
                # types may be None after calling freq_threshold()
                continue
            if typ2 in self._params.sent_starters:
                continue

            col_count = self._collocation_fdist[types]
            typ1_count = self._type_fdist[typ1] + self._type_fdist[typ1 + "."]
            typ2_count = self._type_fdist[typ2] + self._type_fdist[typ2 + "."]
            if (
                typ1_count > 1
                and typ2_count > 1
                and self.MIN_COLLOC_FREQ < col_count <= min(typ1_count, typ2_count)
            ):

                log_likelihood = self._col_log_likelihood(
                    typ1_count, typ2_count, col_count, self._type_fdist.N()
                )
                # Filter out the not-so-collocative
                if log_likelihood >= self.COLLOCATION and (
                    self._type_fdist.N() / typ1_count > typ2_count / col_count
                ):
                    yield (typ1, typ2), log_likelihood

    # ////////////////////////////////////////////////////////////
    # { Sentence-Starter Finder
    # ////////////////////////////////////////////////////////////

    def _is_potential_sent_starter(self, cur_tok, prev_tok):
        """
        Returns True given a token and the token that precedes it if it
        seems clear that the token is beginning a sentence.
        """
        # If a token (i) is preceded by a sentece break that is
        # not a potential ordinal number or initial, and (ii) is
        # alphabetic, then it is a a sentence-starter.
        return (
            prev_tok.sentbreak
            and not (prev_tok.is_number or prev_tok.is_initial)
            and cur_tok.is_alpha
        )

    def _find_sent_starters(self):
        """
        Uses collocation heuristics for each candidate token to
        determine if it frequently starts sentences.
        """
        for typ in self._sent_starter_fdist:
            if not typ:
                continue

            typ_at_break_count = self._sent_starter_fdist[typ]
            typ_count = self._type_fdist[typ] + self._type_fdist[typ + "."]
            if typ_count < typ_at_break_count:
                # needed after freq_threshold
                continue

            log_likelihood = self._col_log_likelihood(
                self._sentbreak_count,
                typ_count,
                typ_at_break_count,
                self._type_fdist.N(),
            )

            if (
                log_likelihood >= self.SENT_STARTER
                and self._type_fdist.N() / self._sentbreak_count
                > typ_count / typ_at_break_count
            ):
                yield typ, log_likelihood

    def _get_sentbreak_count(self, tokens):
        """
        Returns the number of sentence breaks marked in a given set of
        augmented tokens.
        """
        return sum(1 for aug_tok in tokens if aug_tok.sentbreak)


######################################################################
# { Punkt Sentence Tokenizer
######################################################################


class PunktSentenceTokenizer(PunktBaseClass, TokenizerI):
    """
    A sentence tokenizer which uses an unsupervised algorithm to build
    a model for abbreviation words, collocations, and words that start
    sentences; and then uses that model to find sentence boundaries.
    This approach has been shown to work well for many European
    languages.
    """

    def __init__(
        self, train_text=None, verbose=False, lang_vars=None, token_cls=PunktToken
    ):
        """
        train_text can either be the sole training text for this sentence
        boundary detector, or can be a PunktParameters object.
        """
        PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)

        if train_text:
            self._params = self.train(train_text, verbose)

    def train(self, train_text, verbose=False):
        """
        Derives parameters from a given training text, or uses the parameters
        given. Repeated calls to this method destroy previous parameters. For
        incremental training, instantiate a separate PunktTrainer instance.
        """
        if not isinstance(train_text, str):
            return train_text
        return PunktTrainer(
            train_text, lang_vars=self._lang_vars, token_cls=self._Token
        ).get_params()

    # ////////////////////////////////////////////////////////////
    # { Tokenization
    # ////////////////////////////////////////////////////////////

    def tokenize(self, text: str, realign_boundaries: bool = True) -> List[str]:
        """
        Given a text, returns a list of the sentences in that text.
        """
        return list(self.sentences_from_text(text, realign_boundaries))

    def debug_decisions(self, text: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies candidate periods as sentence breaks, yielding a dict for
        each that may be used to understand why the decision was made.

        See format_debug_decision() to help make this output readable.
        """

        for match, decision_text in self._match_potential_end_contexts(text):
            tokens = self._tokenize_words(decision_text)
            tokens = list(self._annotate_first_pass(tokens))
            while tokens and not tokens[0].tok.endswith(self._lang_vars.sent_end_chars):
                tokens.pop(0)
            yield {
                "period_index": match.end() - 1,
                "text": decision_text,
                "type1": tokens[0].type,
                "type2": tokens[1].type,
                "type1_in_abbrs": bool(tokens[0].abbr),
                "type1_is_initial": bool(tokens[0].is_initial),
                "type2_is_sent_starter": tokens[1].type_no_sentperiod
                in self._params.sent_starters,
                "type2_ortho_heuristic": self._ortho_heuristic(tokens[1]),
                "type2_ortho_contexts": set(
                    self._params._debug_ortho_context(tokens[1].type_no_sentperiod)
                ),
                "collocation": (
                    tokens[0].type_no_sentperiod,
                    tokens[1].type_no_sentperiod,
                )
                in self._params.collocations,
                "reason": self._second_pass_annotation(tokens[0], tokens[1])
                or REASON_DEFAULT_DECISION,
                "break_decision": tokens[0].sentbreak,
            }

    def span_tokenize(
        self, text: str, realign_boundaries: bool = True
    ) -> Iterator[Tuple[int, int]]:
        """
        Given a text, generates (start, end) spans of sentences
        in the text.
        """
        slices = self._slices_from_text(text)
        if realign_boundaries:
            slices = self._realign_boundaries(text, slices)
        for sentence in slices:
            yield (sentence.start, sentence.stop)

    def sentences_from_text(
        self, text: str, realign_boundaries: bool = True
    ) -> List[str]:
        """
        Given a text, generates the sentences in that text by only
        testing candidate sentence breaks. If realign_boundaries is
        True, includes in the sentence closing punctuation that
        follows the period.
        """
        return [text[s:e] for s, e in self.span_tokenize(text, realign_boundaries)]

    def _get_last_whitespace_index(self, text: str) -> int:
        """
        Given a text, find the index of the *last* occurrence of *any*
        whitespace character, i.e. " ", "\n", "\t", "\r", etc.
        If none is found, return 0.
        """
        for i in range(len(text) - 1, -1, -1):
            if text[i] in string.whitespace:
                return i
        return 0

    def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[Match, str]]:
        """
        Given a text, find the matches of potential sentence breaks,
        alongside the contexts surrounding these sentence breaks.

        Since the fix for the ReDOS discovered in issue #2866, we no longer match
        the word before a potential end of sentence token. Instead, we use a separate
        regex for this. As a consequence, `finditer`'s desire to find non-overlapping
        matches no longer aids us in finding the single longest match.
        Where previously, we could use::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +SKIP
            [<re.Match object; span=(9, 18), match='acting!!!'>]

        Now we have to find the word before (i.e. 'acting') separately, and `finditer`
        returns::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +NORMALIZE_WHITESPACE
            [<re.Match object; span=(15, 16), match='!'>,
            <re.Match object; span=(16, 17), match='!'>,
            <re.Match object; span=(17, 18), match='!'>]

        So, we need to find the word before the match from right to left, and then manually remove
        the overlaps. That is what this method does::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._match_potential_end_contexts(text))
            [(<re.Match object; span=(17, 18), match='!'>, 'acting!!! I')]

        :param text: String of one or more sentences
        :type text: str
        :return: Generator of match-context tuples.
        :rtype: Iterator[Tuple[Match, str]]
        """
        previous_slice = slice(0, 0)
        previous_match = None
        for match in self._lang_vars.period_context_re().finditer(text):

            # Get the slice of the previous word
            before_text = text[previous_slice.stop : match.start()]
            index_after_last_space = self._get_last_whitespace_index(before_text)
            if index_after_last_space:
                # + 1 to exclude the space itself
                index_after_last_space += previous_slice.stop + 1
            else:
                index_after_last_space = previous_slice.start
            prev_word_slice = slice(index_after_last_space, match.start())

            # If the previous slice does not overlap with this slice, then
            # we can yield the previous match and slice. If there is an overlap,
            # then we do not yield the previous match and slice.
            if previous_match and previous_slice.stop <= prev_word_slice.start:
                yield (
                    previous_match,
                    text[previous_slice]
                    + previous_match.group()
                    + previous_match.group("after_tok"),
                )
            previous_match = match
            previous_slice = prev_word_slice

        # Yield the last match and context, if it exists
        if previous_match:
            yield (
                previous_match,
                text[previous_slice]
                + previous_match.group()
                + previous_match.group("after_tok"),
            )

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        last_break = 0
        for match, context in self._match_potential_end_contexts(text):
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                if match.group("next_tok"):
                    # next sentence starts after whitespace
                    last_break = match.start("next_tok")
                else:
                    # next sentence starts at following punctuation
                    last_break = match.end()
        # The last sentence should not contain trailing whitespace.
        yield slice(last_break, len(text.rstrip()))

    def _realign_boundaries(
        self, text: str, slices: Iterator[slice]
    ) -> Iterator[slice]:
        """
        Attempts to realign punctuation that falls after the period but
        should otherwise be included in the same sentence.

        For example: "(Sent1.) Sent2." will otherwise be split as::

            ["(Sent1.", ") Sent1."].

        This method will produce::

            ["(Sent1.)", "Sent2."].
        """
        realign = 0
        for sentence1, sentence2 in _pair_iter(slices):
            sentence1 = slice(sentence1.start + realign, sentence1.stop)
            if not sentence2:
                if text[sentence1]:
                    yield sentence1
                continue

            m = self._lang_vars.re_boundary_realignment.match(text[sentence2])
            if m:
                yield slice(sentence1.start, sentence2.start + len(m.group(0).rstrip()))
                realign = m.end()
            else:
                realign = 0
                if text[sentence1]:
                    yield sentence1

    def text_contains_sentbreak(self, text: str) -> bool:
        """
        Returns True if the given text includes a sentence break.
        """
        found = False  # used to ignore last token
        for tok in self._annotate_tokens(self._tokenize_words(text)):
            if found:
                return True
            if tok.sentbreak:
                found = True
        return False

    def sentences_from_text_legacy(self, text: str) -> Iterator[str]:
        """
        Given a text, generates the sentences in that text. Annotates all
        tokens, rather than just those with possible sentence breaks. Should
        produce the same results as ``sentences_from_text``.
        """
        tokens = self._annotate_tokens(self._tokenize_words(text))
        return self._build_sentence_list(text, tokens)

    def sentences_from_tokens(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        """
        Given a sequence of tokens, generates lists of tokens, each list
        corresponding to a sentence.
        """
        tokens = iter(self._annotate_tokens(self._Token(t) for t in tokens))
        sentence = []
        for aug_tok in tokens:
            sentence.append(aug_tok.tok)
            if aug_tok.sentbreak:
                yield sentence
                sentence = []
        if sentence:
            yield sentence

    def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Given a set of tokens augmented with markers for line-start and
        paragraph-start, returns an iterator through those tokens with full
        annotation including predicted sentence breaks.
        """
        # Make a preliminary pass through the document, marking likely
        # sentence breaks, abbreviations, and ellipsis tokens.
        tokens = self._annotate_first_pass(tokens)

        # Make a second pass through the document, using token context
        # information to change our preliminary decisions about where
        # sentence breaks, abbreviations, and ellipsis occurs.
        tokens = self._annotate_second_pass(tokens)

        ## [XX] TESTING
        # tokens = list(tokens)
        # self.dump(tokens)

        return tokens

    def _build_sentence_list(
        self, text: str, tokens: Iterator[PunktToken]
    ) -> Iterator[str]:
        """
        Given the original text and the list of augmented word tokens,
        construct and return a tokenized list of sentence strings.
        """
        # Most of the work here is making sure that we put the right
        # pieces of whitespace back in all the right places.

        # Our position in the source text, used to keep track of which
        # whitespace to add:
        pos = 0

        # A regular expression that finds pieces of whitespace:
        white_space_regexp = re.compile(r"\s*")

        sentence = ""
        for aug_tok in tokens:
            tok = aug_tok.tok

            # Find the whitespace before this token, and update pos.
            white_space = white_space_regexp.match(text, pos).group()
            pos += len(white_space)

            # Some of the rules used by the punkt word tokenizer
            # strip whitespace out of the text, resulting in tokens
            # that contain whitespace in the source text.  If our
            # token doesn't match, see if adding whitespace helps.
            # If so, then use the version with whitespace.
            if text[pos : pos + len(tok)] != tok:
                pat = r"\s*".join(re.escape(c) for c in tok)
                m = re.compile(pat).match(text, pos)
                if m:
                    tok = m.group()

            # Move our position pointer to the end of the token.
            assert text[pos : pos + len(tok)] == tok
            pos += len(tok)

            # Add this token.  If it's not at the beginning of the
            # sentence, then include any whitespace that separated it
            # from the previous token.
            if sentence:
                sentence += white_space
            sentence += tok

            # If we're at a sentence break, then start a new sentence.
            if aug_tok.sentbreak:
                yield sentence
                sentence = ""

        # If the last sentence is empty, discard it.
        if sentence:
            yield sentence

    # [XX] TESTING
    def dump(self, tokens: Iterator[PunktToken]) -> None:
        print("writing to /tmp/punkt.new...")
        with open("/tmp/punkt.new", "w") as outfile:
            for aug_tok in tokens:
                if aug_tok.parastart:
                    outfile.write("\n\n")
                elif aug_tok.linestart:
                    outfile.write("\n")
                else:
                    outfile.write(" ")

                outfile.write(str(aug_tok))

    # ////////////////////////////////////////////////////////////
    # { Customization Variables
    # ////////////////////////////////////////////////////////////

    PUNCTUATION = tuple(";:,.!?")

    # ////////////////////////////////////////////////////////////
    # { Annotation Procedures
    # ////////////////////////////////////////////////////////////

    def _annotate_second_pass(
        self, tokens: Iterator[PunktToken]
    ) -> Iterator[PunktToken]:
        """
        Performs a token-based classification (section 4) over the given
        tokens, making use of the orthographic heuristic (4.1.1), collocation
        heuristic (4.1.2) and frequent sentence starter heuristic (4.1.3).
        """
        for token1, token2 in _pair_iter(tokens):
            self._second_pass_annotation(token1, token2)
            yield token1

    def _second_pass_annotation(
        self, aug_tok1: PunktToken, aug_tok2: Optional[PunktToken]
    ) -> Optional[str]:
        """
        Performs token-based classification over a pair of contiguous tokens
        updating the first.
        """
        # Is it the last token? We can't do anything then.
        if not aug_tok2:
            return

        if not aug_tok1.period_final:
            # We only care about words ending in periods.
            return
        typ = aug_tok1.type_no_period
        next_typ = aug_tok2.type_no_sentperiod
        tok_is_initial = aug_tok1.is_initial

        # [4.1.2. Collocation Heuristic] If there's a
        # collocation between the word before and after the
        # period, then label tok as an abbreviation and NOT
        # a sentence break. Note that collocations with
        # frequent sentence starters as their second word are
        # excluded in training.
        if (typ, next_typ) in self._params.collocations:
            aug_tok1.sentbreak = False
            aug_tok1.abbr = True
            return REASON_KNOWN_COLLOCATION

        # [4.2. Token-Based Reclassification of Abbreviations] If
        # the token is an abbreviation or an ellipsis, then decide
        # whether we should *also* classify it as a sentbreak.
        if (aug_tok1.abbr or aug_tok1.ellipsis) and (not tok_is_initial):
            # [4.1.1. Orthographic Heuristic] Check if there's
            # orthogrpahic evidence about whether the next word
            # starts a sentence or not.
            is_sent_starter = self._ortho_heuristic(aug_tok2)
            if is_sent_starter == True:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC

            # [4.1.3. Frequent Sentence Starter Heruistic] If the
            # next word is capitalized, and is a member of the
            # frequent-sentence-starters list, then label tok as a
            # sentence break.
            if aug_tok2.first_upper and next_typ in self._params.sent_starters:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_SENTENCE_STARTER

        # [4.3. Token-Based Detection of Initials and Ordinals]
        # Check if any initials or ordinals tokens that are marked
        # as sentbreaks should be reclassified as abbreviations.
        if tok_is_initial or typ == "##number##":

            # [4.1.1. Orthographic Heuristic] Check if there's
            # orthogrpahic evidence about whether the next word
            # starts a sentence or not.
            is_sent_starter = self._ortho_heuristic(aug_tok2)

            if is_sent_starter == False:
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                if tok_is_initial:
                    return REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC
                return REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC

            # Special heuristic for initials: if orthogrpahic
            # heuristic is unknown, and next word is always
            # capitalized, then mark as abbrev (eg: J. Bach).
            if (
                is_sent_starter == "unknown"
                and tok_is_initial
                and aug_tok2.first_upper
                and not (self._params.ortho_context[next_typ] & _ORTHO_LC)
            ):
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                return REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC

        return

    def _ortho_heuristic(self, aug_tok: PunktToken) -> Union[bool, str]:
        """
        Decide whether the given token is the first token in a sentence.
        """
        # Sentences don't start with punctuation marks:
        if aug_tok.tok in self.PUNCTUATION:
            return False

        ortho_context = self._params.ortho_context[aug_tok.type_no_sentperiod]

        # If the word is capitalized, occurs at least once with a
        # lower case first letter, and never occurs with an upper case
        # first letter sentence-internally, then it's a sentence starter.
        if (
            aug_tok.first_upper
            and (ortho_context & _ORTHO_LC)
            and not (ortho_context & _ORTHO_MID_UC)
        ):
            return True

        # If the word is lower case, and either (a) we've seen it used
        # with upper case, or (b) we've never seen it used
        # sentence-initially with lower case, then it's not a sentence
        # starter.
        if aug_tok.first_lower and (
            (ortho_context & _ORTHO_UC) or not (ortho_context & _ORTHO_BEG_LC)
        ):
            return False

        # Otherwise, we're not sure.
        return "unknown"


DEBUG_DECISION_FMT = """Text: {text!r} (at offset {period_index})
Sentence break? {break_decision} ({reason})
Collocation? {collocation}
{type1!r}:
    known abbreviation: {type1_in_abbrs}
    is initial: {type1_is_initial}
{type2!r}:
    known sentence starter: {type2_is_sent_starter}
    orthographic heuristic suggests is a sentence starter? {type2_ortho_heuristic}
    orthographic contexts in training: {type2_ortho_contexts}
"""


def format_debug_decision(d):
    return DEBUG_DECISION_FMT.format(**d)


def demo(text, tok_cls=PunktSentenceTokenizer, train_cls=PunktTrainer):
    """Builds a punkt model and applies it to the same text"""
    cleanup = (
        lambda s: re.compile(r"(?:\r|^\s+)", re.MULTILINE).sub("", s).replace("\n", " ")
    )
    trainer = train_cls()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(text)
    sbd = tok_cls(trainer.get_params())
    for sentence in sbd.sentences_from_text(text):
        print(cleanup(sentence))
