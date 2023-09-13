"""Completion for IPython.

This module started as fork of the rlcompleter module in the Python standard
library.  The original enhancements made to rlcompleter have been sent
upstream and were accepted as of Python 2.3,

This module now support a wide variety of completion mechanism both available
for normal classic Python code, as well as completer for IPython specific
Syntax like magics.

Latex and Unicode completion
============================

IPython and compatible frontends not only can complete your code, but can help
you to input a wide range of characters. In particular we allow you to insert
a unicode character using the tab completion mechanism.

Forward latex/unicode completion
--------------------------------

Forward completion allows you to easily type a unicode character using its latex
name, or unicode long description. To do so type a backslash follow by the
relevant name and press tab:


Using latex completion:

.. code::

    \\alpha<tab>
    α

or using unicode completion:


.. code::

    \\GREEK SMALL LETTER ALPHA<tab>
    α


Only valid Python identifiers will complete. Combining characters (like arrow or
dots) are also available, unlike latex they need to be put after the their
counterpart that is to say, ``F\\\\vec<tab>`` is correct, not ``\\\\vec<tab>F``.

Some browsers are known to display combining characters incorrectly.

Backward latex completion
-------------------------

It is sometime challenging to know how to type a character, if you are using
IPython, or any compatible frontend you can prepend backslash to the character
and press :kbd:`Tab` to expand it to its latex form.

.. code::

    \\α<tab>
    \\alpha


Both forward and backward completions can be deactivated by setting the
:std:configtrait:`Completer.backslash_combining_completions` option to
``False``.


Experimental
============

Starting with IPython 6.0, this module can make use of the Jedi library to
generate completions both using static analysis of the code, and dynamically
inspecting multiple namespaces. Jedi is an autocompletion and static analysis
for Python. The APIs attached to this new mechanism is unstable and will
raise unless use in an :any:`provisionalcompleter` context manager.

You will find that the following are experimental:

    - :any:`provisionalcompleter`
    - :any:`IPCompleter.completions`
    - :any:`Completion`
    - :any:`rectify_completions`

.. note::

    better name for :any:`rectify_completions` ?

We welcome any feedback on these new API, and we also encourage you to try this
module in debug mode (start IPython with ``--Completer.debug=True``) in order
to have extra logging information if :any:`jedi` is crashing, or if current
IPython completer pending deprecations are returning results not yet handled
by :any:`jedi`

Using Jedi for tab completion allow snippets like the following to work without
having to execute any code:

   >>> myvar = ['hello', 42]
   ... myvar[1].bi<tab>

Tab completion will be able to infer that ``myvar[1]`` is a real number without
executing almost any code unlike the deprecated :any:`IPCompleter.greedy`
option.

Be sure to update :any:`jedi` to the latest stable version or to try the
current development version to get better completions.

Matchers
========

All completions routines are implemented using unified *Matchers* API.
The matchers API is provisional and subject to change without notice.

The built-in matchers include:

- :any:`IPCompleter.dict_key_matcher`:  dictionary key completions,
- :any:`IPCompleter.magic_matcher`: completions for magics,
- :any:`IPCompleter.unicode_name_matcher`,
  :any:`IPCompleter.fwd_unicode_matcher`
  and :any:`IPCompleter.latex_name_matcher`: see `Forward latex/unicode completion`_,
- :any:`back_unicode_name_matcher` and :any:`back_latex_name_matcher`: see `Backward latex completion`_,
- :any:`IPCompleter.file_matcher`: paths to files and directories,
- :any:`IPCompleter.python_func_kw_matcher` - function keywords,
- :any:`IPCompleter.python_matches` - globals and attributes (v1 API),
- ``IPCompleter.jedi_matcher`` - static analysis with Jedi,
- :any:`IPCompleter.custom_completer_matcher` - pluggable completer with a default
  implementation in :any:`InteractiveShell` which uses IPython hooks system
  (`complete_command`) with string dispatch (including regular expressions).
  Differently to other matchers, ``custom_completer_matcher`` will not suppress
  Jedi results to match behaviour in earlier IPython versions.

Custom matchers can be added by appending to ``IPCompleter.custom_matchers`` list.

Matcher API
-----------

Simplifying some details, the ``Matcher`` interface can described as

.. code-block::

    MatcherAPIv1 = Callable[[str], list[str]]
    MatcherAPIv2 = Callable[[CompletionContext], SimpleMatcherResult]

    Matcher = MatcherAPIv1 | MatcherAPIv2

The ``MatcherAPIv1`` reflects the matcher API as available prior to IPython 8.6.0
and remains supported as a simplest way for generating completions. This is also
currently the only API supported by the IPython hooks system `complete_command`.

To distinguish between matcher versions ``matcher_api_version`` attribute is used.
More precisely, the API allows to omit ``matcher_api_version`` for v1 Matchers,
and requires a literal ``2`` for v2 Matchers.

Once the API stabilises future versions may relax the requirement for specifying
``matcher_api_version`` by switching to :any:`functools.singledispatch`, therefore
please do not rely on the presence of ``matcher_api_version`` for any purposes.

Suppression of competing matchers
---------------------------------

By default results from all matchers are combined, in the order determined by
their priority. Matchers can request to suppress results from subsequent
matchers by setting ``suppress`` to ``True`` in the ``MatcherResult``.

When multiple matchers simultaneously request surpression, the results from of
the matcher with higher priority will be returned.

Sometimes it is desirable to suppress most but not all other matchers;
this can be achieved by adding a set of identifiers of matchers which
should not be suppressed to ``MatcherResult`` under ``do_not_suppress`` key.

The suppression behaviour can is user-configurable via
:std:configtrait:`IPCompleter.suppress_competing_matchers`.
"""


# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
#
# Some of this code originated from rlcompleter in the Python standard library
# Copyright (C) 2001 Python Software Foundation, www.python.org

from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
    Iterable,
    Iterator,
    List,
    Tuple,
    Union,
    Any,
    Sequence,
    Dict,
    Optional,
    TYPE_CHECKING,
    Set,
    Sized,
    TypeVar,
    Literal,
)

from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
    Bool,
    Enum,
    Int,
    List as ListTrait,
    Unicode,
    Dict as DictTrait,
    Union as UnionTrait,
    observe,
)
from traitlets.config.configurable import Configurable

import __main__

# skip module docstests
__skip_doctest__ = True


try:
    import jedi
    jedi.settings.case_insensitive_completion = False
    import jedi.api.helpers
    import jedi.api.classes
    JEDI_INSTALLED = True
except ImportError:
    JEDI_INSTALLED = False


if TYPE_CHECKING or GENERATING_DOCUMENTATION and sys.version_info >= (3, 11):
    from typing import cast
    from typing_extensions import TypedDict, NotRequired, Protocol, TypeAlias, TypeGuard
else:
    from typing import Generic

    def cast(type_, obj):
        """Workaround for `TypeError: MatcherAPIv2() takes no arguments`"""
        return obj

    # do not require on runtime
    NotRequired = Tuple  # requires Python >=3.11
    TypedDict = Dict  # by extension of `NotRequired` requires 3.11 too
    Protocol = object  # requires Python >=3.8
    TypeAlias = Any  # requires Python >=3.10
    TypeGuard = Generic  # requires Python >=3.10
if GENERATING_DOCUMENTATION:
    from typing import TypedDict

# -----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# ranges where we have most of the valid unicode names. We could be more finer
# grained but is it worth it for performance  While unicode have character in the
# range 0, 0x110000, we seem to have name for about 10% of those. (131808 as I
# write this). With below range we cover them all, with a density of ~67%
# biggest next gap we consider only adds up about 1% density and there are 600
# gaps that would need hard coding.
_UNICODE_RANGES = [(32, 0x323B0), (0xE0001, 0xE01F0)]

# Public API
__all__ = ["Completer", "IPCompleter"]

if sys.platform == 'win32':
    PROTECTABLES = ' '
else:
    PROTECTABLES = ' ()[]{}?=\\|;:\'#*"^&'

# Protect against returning an enormous number of completions which the frontend
# may have trouble processing.
MATCHES_LIMIT = 500

# Completion type reported when no type can be inferred.
_UNKNOWN_TYPE = "<unknown>"

# sentinel value to signal lack of a match
not_found = object()

class ProvisionalCompleterWarning(FutureWarning):
    """
    Exception raise by an experimental feature in this module.

    Wrap code in :any:`provisionalcompleter` context manager if you
    are certain you want to use an unstable feature.
    """
    pass

warnings.filterwarnings('error', category=ProvisionalCompleterWarning)


@skip_doctest
@contextmanager
def provisionalcompleter(action='ignore'):
    """
    This context manager has to be used in any place where unstable completer
    behavior and API may be called.

    >>> with provisionalcompleter():
    ...     completer.do_experimental_things() # works

    >>> completer.do_experimental_things() # raises.

    .. note::

        Unstable

        By using this context manager you agree that the API in use may change
        without warning, and that you won't complain if they do so.

        You also understand that, if the API is not to your liking, you should report
        a bug to explain your use case upstream.

        We'll be happy to get your feedback, feature requests, and improvements on
        any of the unstable APIs!
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action, category=ProvisionalCompleterWarning)
        yield


def has_open_quotes(s):
    """Return whether a string has open quotes.

    This simply counts whether the number of quote characters of either type in
    the string is odd.

    Returns
    -------
    If there is an open quote, the quote character is returned.  Else, return
    False.
    """
    # We check " first, then ', so complex cases with nested quotes will get
    # the " to take precedence.
    if s.count('"') % 2:
        return '"'
    elif s.count("'") % 2:
        return "'"
    else:
        return False


def protect_filename(s, protectables=PROTECTABLES):
    """Escape a string to protect certain characters."""
    if set(s) & set(protectables):
        if sys.platform == "win32":
            return '"' + s + '"'
        else:
            return "".join(("\\" + c if c in protectables else c) for c in s)
    else:
        return s


def expand_user(path:str) -> Tuple[str, bool, str]:
    """Expand ``~``-style usernames in strings.

    This is similar to :func:`os.path.expanduser`, but it computes and returns
    extra information that will be useful if the input was being used in
    computing completions, and you wish to return the completions with the
    original '~' instead of its expanded value.

    Parameters
    ----------
    path : str
        String to be expanded.  If no ~ is present, the output is the same as the
        input.

    Returns
    -------
    newpath : str
        Result of ~ expansion in the input path.
    tilde_expand : bool
        Whether any expansion was performed or not.
    tilde_val : str
        The value that ~ was replaced with.
    """
    # Default values
    tilde_expand = False
    tilde_val = ''
    newpath = path

    if path.startswith('~'):
        tilde_expand = True
        rest = len(path)-1
        newpath = os.path.expanduser(path)
        if rest:
            tilde_val = newpath[:-rest]
        else:
            tilde_val = newpath

    return newpath, tilde_expand, tilde_val


def compress_user(path:str, tilde_expand:bool, tilde_val:str) -> str:
    """Does the opposite of expand_user, with its outputs.
    """
    if tilde_expand:
        return path.replace(tilde_val, '~')
    else:
        return path


def completions_sorting_key(word):
    """key for sorting completions

    This does several things:

    - Demote any completions starting with underscores to the end
    - Insert any %magic and %%cellmagic completions in the alphabetical order
      by their name
    """
    prio1, prio2 = 0, 0

    if word.startswith('__'):
        prio1 = 2
    elif word.startswith('_'):
        prio1 = 1

    if word.endswith('='):
        prio1 = -1

    if word.startswith('%%'):
        # If there's another % in there, this is something else, so leave it alone
        if not "%" in word[2:]:
            word = word[2:]
            prio2 = 2
    elif word.startswith('%'):
        if not "%" in word[1:]:
            word = word[1:]
            prio2 = 1

    return prio1, word, prio2


class _FakeJediCompletion:
    """
    This is a workaround to communicate to the UI that Jedi has crashed and to
    report a bug. Will be used only id :any:`IPCompleter.debug` is set to true.

    Added in IPython 6.0 so should likely be removed for 7.0

    """

    def __init__(self, name):

        self.name = name
        self.complete = name
        self.type = 'crashed'
        self.name_with_symbols = name
        self.signature = ""
        self._origin = "fake"
        self.text = "crashed"

    def __repr__(self):
        return '<Fake completion object jedi has crashed>'


_JediCompletionLike = Union["jedi.api.Completion", _FakeJediCompletion]


class Completion:
    """
    Completion object used and returned by IPython completers.

    .. warning::

        Unstable

        This function is unstable, API may change without warning.
        It will also raise unless use in proper context manager.

    This act as a middle ground :any:`Completion` object between the
    :any:`jedi.api.classes.Completion` object and the Prompt Toolkit completion
    object. While Jedi need a lot of information about evaluator and how the
    code should be ran/inspected, PromptToolkit (and other frontend) mostly
    need user facing information.

    - Which range should be replaced replaced by what.
    - Some metadata (like completion type), or meta information to displayed to
      the use user.

    For debugging purpose we can also store the origin of the completion (``jedi``,
    ``IPython.python_matches``, ``IPython.magics_matches``...).
    """

    __slots__ = ['start', 'end', 'text', 'type', 'signature', '_origin']

    def __init__(
        self,
        start: int,
        end: int,
        text: str,
        *,
        type: Optional[str] = None,
        _origin="",
        signature="",
    ) -> None:
        warnings.warn(
            "``Completion`` is a provisional API (as of IPython 6.0). "
            "It may change without warnings. "
            "Use in corresponding context manager.",
            category=ProvisionalCompleterWarning,
            stacklevel=2,
        )

        self.start = start
        self.end = end
        self.text = text
        self.type = type
        self.signature = signature
        self._origin = _origin

    def __repr__(self):
        return '<Completion start=%s end=%s text=%r type=%r, signature=%r,>' % \
                (self.start, self.end, self.text, self.type or '?', self.signature or '?')

    def __eq__(self, other) -> bool:
        """
        Equality and hash do not hash the type (as some completer may not be
        able to infer the type), but are use to (partially) de-duplicate
        completion.

        Completely de-duplicating completion is a bit tricker that just
        comparing as it depends on surrounding text, which Completions are not
        aware of.
        """
        return self.start == other.start and \
            self.end == other.end and \
            self.text == other.text

    def __hash__(self):
        return hash((self.start, self.end, self.text))


class SimpleCompletion:
    """Completion item to be included in the dictionary returned by new-style Matcher (API v2).

    .. warning::

        Provisional

        This class is used to describe the currently supported attributes of
        simple completion items, and any additional implementation details
        should not be relied on. Additional attributes may be included in
        future versions, and meaning of text disambiguated from the current
        dual meaning of "text to insert" and "text to used as a label".
    """

    __slots__ = ["text", "type"]

    def __init__(self, text: str, *, type: Optional[str] = None):
        self.text = text
        self.type = type

    def __repr__(self):
        return f"<SimpleCompletion text={self.text!r} type={self.type!r}>"


class _MatcherResultBase(TypedDict):
    """Definition of dictionary to be returned by new-style Matcher (API v2)."""

    #: Suffix of the provided ``CompletionContext.token``, if not given defaults to full token.
    matched_fragment: NotRequired[str]

    #: Whether to suppress results from all other matchers (True), some
    #: matchers (set of identifiers) or none (False); default is False.
    suppress: NotRequired[Union[bool, Set[str]]]

    #: Identifiers of matchers which should NOT be suppressed when this matcher
    #: requests to suppress all other matchers; defaults to an empty set.
    do_not_suppress: NotRequired[Set[str]]

    #: Are completions already ordered and should be left as-is? default is False.
    ordered: NotRequired[bool]


@sphinx_options(show_inherited_members=True, exclude_inherited_from=["dict"])
class SimpleMatcherResult(_MatcherResultBase, TypedDict):
    """Result of new-style completion matcher."""

    # note: TypedDict is added again to the inheritance chain
    # in order to get __orig_bases__ for documentation

    #: List of candidate completions
    completions: Sequence[SimpleCompletion] | Iterator[SimpleCompletion]


class _JediMatcherResult(_MatcherResultBase):
    """Matching result returned by Jedi (will be processed differently)"""

    #: list of candidate completions
    completions: Iterator[_JediCompletionLike]


AnyMatcherCompletion = Union[_JediCompletionLike, SimpleCompletion]
AnyCompletion = TypeVar("AnyCompletion", AnyMatcherCompletion, Completion)


@dataclass
class CompletionContext:
    """Completion context provided as an argument to matchers in the Matcher API v2."""

    # rationale: many legacy matchers relied on completer state (`self.text_until_cursor`)
    # which was not explicitly visible as an argument of the matcher, making any refactor
    # prone to errors; by explicitly passing `cursor_position` we can decouple the matchers
    # from the completer, and make substituting them in sub-classes easier.

    #: Relevant fragment of code directly preceding the cursor.
    #: The extraction of token is implemented via splitter heuristic
    #: (following readline behaviour for legacy reasons), which is user configurable
    #: (by switching the greedy mode).
    token: str

    #: The full available content of the editor or buffer
    full_text: str

    #: Cursor position in the line (the same for ``full_text`` and ``text``).
    cursor_position: int

    #: Cursor line in ``full_text``.
    cursor_line: int

    #: The maximum number of completions that will be used downstream.
    #: Matchers can use this information to abort early.
    #: The built-in Jedi matcher is currently excepted from this limit.
    # If not given, return all possible completions.
    limit: Optional[int]

    @cached_property
    def text_until_cursor(self) -> str:
        return self.line_with_cursor[: self.cursor_position]

    @cached_property
    def line_with_cursor(self) -> str:
        return self.full_text.split("\n")[self.cursor_line]


#: Matcher results for API v2.
MatcherResult = Union[SimpleMatcherResult, _JediMatcherResult]


class _MatcherAPIv1Base(Protocol):
    def __call__(self, text: str) -> List[str]:
        """Call signature."""
        ...

    #: Used to construct the default matcher identifier
    __qualname__: str


class _MatcherAPIv1Total(_MatcherAPIv1Base, Protocol):
    #: API version
    matcher_api_version: Optional[Literal[1]]

    def __call__(self, text: str) -> List[str]:
        """Call signature."""
        ...


#: Protocol describing Matcher API v1.
MatcherAPIv1: TypeAlias = Union[_MatcherAPIv1Base, _MatcherAPIv1Total]


class MatcherAPIv2(Protocol):
    """Protocol describing Matcher API v2."""

    #: API version
    matcher_api_version: Literal[2] = 2

    def __call__(self, context: CompletionContext) -> MatcherResult:
        """Call signature."""
        ...

    #: Used to construct the default matcher identifier
    __qualname__: str


Matcher: TypeAlias = Union[MatcherAPIv1, MatcherAPIv2]


def _is_matcher_v1(matcher: Matcher) -> TypeGuard[MatcherAPIv1]:
    api_version = _get_matcher_api_version(matcher)
    return api_version == 1


def _is_matcher_v2(matcher: Matcher) -> TypeGuard[MatcherAPIv2]:
    api_version = _get_matcher_api_version(matcher)
    return api_version == 2


def _is_sizable(value: Any) -> TypeGuard[Sized]:
    """Determines whether objects is sizable"""
    return hasattr(value, "__len__")


def _is_iterator(value: Any) -> TypeGuard[Iterator]:
    """Determines whether objects is sizable"""
    return hasattr(value, "__next__")


def has_any_completions(result: MatcherResult) -> bool:
    """Check if any result includes any completions."""
    completions = result["completions"]
    if _is_sizable(completions):
        return len(completions) != 0
    if _is_iterator(completions):
        try:
            old_iterator = completions
            first = next(old_iterator)
            result["completions"] = cast(
                Iterator[SimpleCompletion],
                itertools.chain([first], old_iterator),
            )
            return True
        except StopIteration:
            return False
    raise ValueError(
        "Completions returned by matcher need to be an Iterator or a Sizable"
    )


def completion_matcher(
    *,
    priority: Optional[float] = None,
    identifier: Optional[str] = None,
    api_version: int = 1,
):
    """Adds attributes describing the matcher.

    Parameters
    ----------
    priority : Optional[float]
        The priority of the matcher, determines the order of execution of matchers.
        Higher priority means that the matcher will be executed first. Defaults to 0.
    identifier : Optional[str]
        identifier of the matcher allowing users to modify the behaviour via traitlets,
        and also used to for debugging (will be passed as ``origin`` with the completions).

        Defaults to matcher function's ``__qualname__`` (for example,
        ``IPCompleter.file_matcher`` for the built-in matched defined
        as a ``file_matcher`` method of the ``IPCompleter`` class).
    api_version: Optional[int]
        version of the Matcher API used by this matcher.
        Currently supported values are 1 and 2.
        Defaults to 1.
    """

    def wrapper(func: Matcher):
        func.matcher_priority = priority or 0  # type: ignore
        func.matcher_identifier = identifier or func.__qualname__  # type: ignore
        func.matcher_api_version = api_version  # type: ignore
        if TYPE_CHECKING:
            if api_version == 1:
                func = cast(MatcherAPIv1, func)
            elif api_version == 2:
                func = cast(MatcherAPIv2, func)
        return func

    return wrapper


def _get_matcher_priority(matcher: Matcher):
    return getattr(matcher, "matcher_priority", 0)


def _get_matcher_id(matcher: Matcher):
    return getattr(matcher, "matcher_identifier", matcher.__qualname__)


def _get_matcher_api_version(matcher):
    return getattr(matcher, "matcher_api_version", 1)


context_matcher = partial(completion_matcher, api_version=2)


_IC = Iterable[Completion]


def _deduplicate_completions(text: str, completions: _IC)-> _IC:
    """
    Deduplicate a set of completions.

    .. warning::

        Unstable

        This function is unstable, API may change without warning.

    Parameters
    ----------
    text : str
        text that should be completed.
    completions : Iterator[Completion]
        iterator over the completions to deduplicate

    Yields
    ------
    `Completions` objects
    Completions coming from multiple sources, may be different but end up having
    the same effect when applied to ``text``. If this is the case, this will
    consider completions as equal and only emit the first encountered.
    Not folded in `completions()` yet for debugging purpose, and to detect when
    the IPython completer does return things that Jedi does not, but should be
    at some point.
    """
    completions = list(completions)
    if not completions:
        return

    new_start = min(c.start for c in completions)
    new_end = max(c.end for c in completions)

    seen = set()
    for c in completions:
        new_text = text[new_start:c.start] + c.text + text[c.end:new_end]
        if new_text not in seen:
            yield c
            seen.add(new_text)


def rectify_completions(text: str, completions: _IC, *, _debug: bool = False) -> _IC:
    """
    Rectify a set of completions to all have the same ``start`` and ``end``

    .. warning::

        Unstable

        This function is unstable, API may change without warning.
        It will also raise unless use in proper context manager.

    Parameters
    ----------
    text : str
        text that should be completed.
    completions : Iterator[Completion]
        iterator over the completions to rectify
    _debug : bool
        Log failed completion

    Notes
    -----
    :any:`jedi.api.classes.Completion` s returned by Jedi may not have the same start and end, though
    the Jupyter Protocol requires them to behave like so. This will readjust
    the completion to have the same ``start`` and ``end`` by padding both
    extremities with surrounding text.

    During stabilisation should support a ``_debug`` option to log which
    completion are return by the IPython completer and not found in Jedi in
    order to make upstream bug report.
    """
    warnings.warn("`rectify_completions` is a provisional API (as of IPython 6.0). "
                 "It may change without warnings. "
                 "Use in corresponding context manager.",
                  category=ProvisionalCompleterWarning, stacklevel=2)

    completions = list(completions)
    if not completions:
        return
    starts = (c.start for c in completions)
    ends = (c.end for c in completions)

    new_start = min(starts)
    new_end = max(ends)

    seen_jedi = set()
    seen_python_matches = set()
    for c in completions:
        new_text = text[new_start:c.start] + c.text + text[c.end:new_end]
        if c._origin == 'jedi':
            seen_jedi.add(new_text)
        elif c._origin == 'IPCompleter.python_matches':
            seen_python_matches.add(new_text)
        yield Completion(new_start, new_end, new_text, type=c.type, _origin=c._origin, signature=c.signature)
    diff = seen_python_matches.difference(seen_jedi)
    if diff and _debug:
        print('IPython.python matches have extras:', diff)


if sys.platform == 'win32':
    DELIMS = ' \t\n`!@#$^&*()=+[{]}|;\'",<>?'
else:
    DELIMS = ' \t\n`!@#$^&*()=+[{]}\\|;:\'",<>?'

GREEDY_DELIMS = ' =\r\n'


class CompletionSplitter(object):
    """An object to split an input line in a manner similar to readline.

    By having our own implementation, we can expose readline-like completion in
    a uniform manner to all frontends.  This object only needs to be given the
    line of text to be split and the cursor position on said line, and it
    returns the 'word' to be completed on at the cursor after splitting the
    entire line.

    What characters are used as splitting delimiters can be controlled by
    setting the ``delims`` attribute (this is a property that internally
    automatically builds the necessary regular expression)"""

    # Private interface

    # A string of delimiter characters.  The default value makes sense for
    # IPython's most typical usage patterns.
    _delims = DELIMS

    # The expression (a normal string) to be compiled into a regular expression
    # for actual splitting.  We store it as an attribute mostly for ease of
    # debugging, since this type of code can be so tricky to debug.
    _delim_expr = None

    # The regular expression that does the actual splitting
    _delim_re = None

    def __init__(self, delims=None):
        delims = CompletionSplitter._delims if delims is None else delims
        self.delims = delims

    @property
    def delims(self):
        """Return the string of delimiter characters."""
        return self._delims

    @delims.setter
    def delims(self, delims):
        """Set the delimiters for line splitting."""
        expr = '[' + ''.join('\\'+ c for c in delims) + ']'
        self._delim_re = re.compile(expr)
        self._delims = delims
        self._delim_expr = expr

    def split_line(self, line, cursor_pos=None):
        """Split a line of text with a cursor at the given position.
        """
        l = line if cursor_pos is None else line[:cursor_pos]
        return self._delim_re.split(l)[-1]



class Completer(Configurable):

    greedy = Bool(
        False,
        help="""Activate greedy completion.

        .. deprecated:: 8.8
            Use :std:configtrait:`Completer.evaluation` and :std:configtrait:`Completer.auto_close_dict_keys` instead.

        When enabled in IPython 8.8 or newer, changes configuration as follows:

        - ``Completer.evaluation = 'unsafe'``
        - ``Completer.auto_close_dict_keys = True``
        """,
    ).tag(config=True)

    evaluation = Enum(
        ("forbidden", "minimal", "limited", "unsafe", "dangerous"),
        default_value="limited",
        help="""Policy for code evaluation under completion.

        Successive options allow to enable more eager evaluation for better
        completion suggestions, including for nested dictionaries, nested lists,
        or even results of function calls.
        Setting ``unsafe`` or higher can lead to evaluation of arbitrary user
        code on :kbd:`Tab` with potentially unwanted or dangerous side effects.

        Allowed values are:

        - ``forbidden``: no evaluation of code is permitted,
        - ``minimal``: evaluation of literals and access to built-in namespace;
          no item/attribute evaluationm no access to locals/globals,
          no evaluation of any operations or comparisons.
        - ``limited``: access to all namespaces, evaluation of hard-coded methods
          (for example: :any:`dict.keys`, :any:`object.__getattr__`,
          :any:`object.__getitem__`) on allow-listed objects (for example:
          :any:`dict`, :any:`list`, :any:`tuple`, ``pandas.Series``),
        - ``unsafe``: evaluation of all methods and function calls but not of
          syntax with side-effects like `del x`,
        - ``dangerous``: completely arbitrary evaluation.
        """,
    ).tag(config=True)

    use_jedi = Bool(default_value=JEDI_INSTALLED,
                    help="Experimental: Use Jedi to generate autocompletions. "
                    "Default to True if jedi is installed.").tag(config=True)

    jedi_compute_type_timeout = Int(default_value=400,
        help="""Experimental: restrict time (in milliseconds) during which Jedi can compute types.
        Set to 0 to stop computing types. Non-zero value lower than 100ms may hurt
        performance by preventing jedi to build its cache.
        """).tag(config=True)

    debug = Bool(default_value=False,
                 help='Enable debug for the Completer. Mostly print extra '
                      'information for experimental jedi integration.')\
                      .tag(config=True)

    backslash_combining_completions = Bool(True,
        help="Enable unicode completions, e.g. \\alpha<tab> . "
             "Includes completion of latex commands, unicode names, and expanding "
             "unicode characters back to latex commands.").tag(config=True)

    auto_close_dict_keys = Bool(
        False,
        help="""
        Enable auto-closing dictionary keys.

        When enabled string keys will be suffixed with a final quote
        (matching the opening quote), tuple keys will also receive a
        separating comma if needed, and keys which are final will
        receive a closing bracket (``]``).
        """,
    ).tag(config=True)

    def __init__(self, namespace=None, global_namespace=None, **kwargs):
        """Create a new completer for the command line.

        Completer(namespace=ns, global_namespace=ns2) -> completer instance.

        If unspecified, the default namespace where completions are performed
        is __main__ (technically, __main__.__dict__). Namespaces should be
        given as dictionaries.

        An optional second namespace can be given.  This allows the completer
        to handle cases where both the local and global scopes need to be
        distinguished.
        """

        # Don't bind to namespace quite yet, but flag whether the user wants a
        # specific namespace or to use __main__.__dict__. This will allow us
        # to bind to __main__.__dict__ at completion time, not now.
        if namespace is None:
            self.use_main_ns = True
        else:
            self.use_main_ns = False
            self.namespace = namespace

        # The global namespace, if given, can be bound directly
        if global_namespace is None:
            self.global_namespace = {}
        else:
            self.global_namespace = global_namespace

        self.custom_matchers = []

        super(Completer, self).__init__(**kwargs)

    def complete(self, text, state):
        """Return the next possible completion for 'text'.

        This is called successively with state == 0, 1, 2, ... until it
        returns None.  The completion should begin with 'text'.

        """
        if self.use_main_ns:
            self.namespace = __main__.__dict__

        if state == 0:
            if "." in text:
                self.matches = self.attr_matches(text)
            else:
                self.matches = self.global_matches(text)
        try:
            return self.matches[state]
        except IndexError:
            return None

    def global_matches(self, text):
        """Compute matches when text is a simple name.

        Return a list of all keywords, built-in functions and names currently
        defined in self.namespace or self.global_namespace that match.

        """
        matches = []
        match_append = matches.append
        n = len(text)
        for lst in [
            keyword.kwlist,
            builtin_mod.__dict__.keys(),
            list(self.namespace.keys()),
            list(self.global_namespace.keys()),
        ]:
            for word in lst:
                if word[:n] == text and word != "__builtins__":
                    match_append(word)

        snake_case_re = re.compile(r"[^_]+(_[^_]+)+?\Z")
        for lst in [list(self.namespace.keys()), list(self.global_namespace.keys())]:
            shortened = {
                "_".join([sub[0] for sub in word.split("_")]): word
                for word in lst
                if snake_case_re.match(word)
            }
            for word in shortened.keys():
                if word[:n] == text and word != "__builtins__":
                    match_append(shortened[word])
        return matches

    def attr_matches(self, text):
        """Compute matches when text contains a dot.

        Assuming the text is of the form NAME.NAME....[NAME], and is
        evaluatable in self.namespace or self.global_namespace, it will be
        evaluated and its attributes (as revealed by dir()) are used as
        possible completions.  (For class instances, class members are
        also considered.)

        WARNING: this can still invoke arbitrary C code, if an object
        with a __getattr__ hook is evaluated.

        """
        m2 = re.match(r"(.+)\.(\w*)$", self.line_buffer)
        if not m2:
            return []
        expr, attr = m2.group(1, 2)

        obj = self._evaluate_expr(expr)

        if obj is not_found:
            return []

        if self.limit_to__all__ and hasattr(obj, '__all__'):
            words = get__all__entries(obj)
        else:
            words = dir2(obj)

        try:
            words = generics.complete_object(obj, words)
        except TryNext:
            pass
        except AssertionError:
            raise
        except Exception:
            # Silence errors from completion function
            pass
        # Build match list to return
        n = len(attr)

        # Note: ideally we would just return words here and the prefix
        # reconciliator would know that we intend to append to rather than
        # replace the input text; this requires refactoring to return range
        # which ought to be replaced (as does jedi).
        tokens = _parse_tokens(expr)
        rev_tokens = reversed(tokens)
        skip_over = {tokenize.ENDMARKER, tokenize.NEWLINE}
        name_turn = True

        parts = []
        for token in rev_tokens:
            if token.type in skip_over:
                continue
            if token.type == tokenize.NAME and name_turn:
                parts.append(token.string)
                name_turn = False
            elif token.type == tokenize.OP and token.string == "." and not name_turn:
                parts.append(token.string)
                name_turn = True
            else:
                # short-circuit if not empty nor name token
                break

        prefix_after_space = "".join(reversed(parts))

        return ["%s.%s" % (prefix_after_space, w) for w in words if w[:n] == attr]

    def _evaluate_expr(self, expr):
        obj = not_found
        done = False
        while not done and expr:
            try:
                obj = guarded_eval(
                    expr,
                    EvaluationContext(
                        globals=self.global_namespace,
                        locals=self.namespace,
                        evaluation=self.evaluation,
                    ),
                )
                done = True
            except Exception as e:
                if self.debug:
                    print("Evaluation exception", e)
                # trim the expression to remove any invalid prefix
                # e.g. user starts `(d[`, so we get `expr = '(d'`,
                # where parenthesis is not closed.
                # TODO: make this faster by reusing parts of the computation?
                expr = expr[1:]
        return obj

def get__all__entries(obj):
    """returns the strings in the __all__ attribute"""
    try:
        words = getattr(obj, '__all__')
    except:
        return []

    return [w for w in words if isinstance(w, str)]


class _DictKeyState(enum.Flag):
    """Represent state of the key match in context of other possible matches.

    - given `d1 = {'a': 1}` completion on `d1['<tab>` will yield `{'a': END_OF_ITEM}` as there is no tuple.
    - given `d2 = {('a', 'b'): 1}`: `d2['a', '<tab>` will yield `{'b': END_OF_TUPLE}` as there is no tuple members to add beyond `'b'`.
    - given `d3 = {('a', 'b'): 1}`: `d3['<tab>` will yield `{'a': IN_TUPLE}` as `'a'` can be added.
    - given `d4 = {'a': 1, ('a', 'b'): 2}`: `d4['<tab>` will yield `{'a': END_OF_ITEM & END_OF_TUPLE}`
    """

    BASELINE = 0
    END_OF_ITEM = enum.auto()
    END_OF_TUPLE = enum.auto()
    IN_TUPLE = enum.auto()


def _parse_tokens(c):
    """Parse tokens even if there is an error."""
    tokens = []
    token_generator = tokenize.generate_tokens(iter(c.splitlines()).__next__)
    while True:
        try:
            tokens.append(next(token_generator))
        except tokenize.TokenError:
            return tokens
        except StopIteration:
            return tokens


def _match_number_in_dict_key_prefix(prefix: str) -> Union[str, None]:
    """Match any valid Python numeric literal in a prefix of dictionary keys.

    References:
    - https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals
    - https://docs.python.org/3/library/tokenize.html
    """
    if prefix[-1].isspace():
        # if user typed a space we do not have anything to complete
        # even if there was a valid number token before
        return None
    tokens = _parse_tokens(prefix)
    rev_tokens = reversed(tokens)
    skip_over = {tokenize.ENDMARKER, tokenize.NEWLINE}
    number = None
    for token in rev_tokens:
        if token.type in skip_over:
            continue
        if number is None:
            if token.type == tokenize.NUMBER:
                number = token.string
                continue
            else:
                # we did not match a number
                return None
        if token.type == tokenize.OP:
            if token.string == ",":
                break
            if token.string in {"+", "-"}:
                number = token.string + number
        else:
            return None
    return number


_INT_FORMATS = {
    "0b": bin,
    "0o": oct,
    "0x": hex,
}


def match_dict_keys(
    keys: List[Union[str, bytes, Tuple[Union[str, bytes], ...]]],
    prefix: str,
    delims: str,
    extra_prefix: Optional[Tuple[Union[str, bytes], ...]] = None,
) -> Tuple[str, int, Dict[str, _DictKeyState]]:
    """Used by dict_key_matches, matching the prefix to a list of keys

    Parameters
    ----------
    keys
        list of keys in dictionary currently being completed.
    prefix
        Part of the text already typed by the user. E.g. `mydict[b'fo`
    delims
        String of delimiters to consider when finding the current key.
    extra_prefix : optional
        Part of the text already typed in multi-key index cases. E.g. for
        `mydict['foo', "bar", 'b`, this would be `('foo', 'bar')`.

    Returns
    -------
    A tuple of three elements: ``quote``, ``token_start``, ``matched``, with
    ``quote`` being the quote that need to be used to close current string.
    ``token_start`` the position where the replacement should start occurring,
    ``matches`` a dictionary of replacement/completion keys on keys and values
        indicating whether the state.
    """
    prefix_tuple = extra_prefix if extra_prefix else ()

    prefix_tuple_size = sum(
        [
            # for pandas, do not count slices as taking space
            not isinstance(k, slice)
            for k in prefix_tuple
        ]
    )
    text_serializable_types = (str, bytes, int, float, slice)

    def filter_prefix_tuple(key):
        # Reject too short keys
        if len(key) <= prefix_tuple_size:
            return False
        # Reject keys which cannot be serialised to text
        for k in key:
            if not isinstance(k, text_serializable_types):
                return False
        # Reject keys that do not match the prefix
        for k, pt in zip(key, prefix_tuple):
            if k != pt and not isinstance(pt, slice):
                return False
        # All checks passed!
        return True

    filtered_key_is_final: Dict[
        Union[str, bytes, int, float], _DictKeyState
    ] = defaultdict(lambda: _DictKeyState.BASELINE)

    for k in keys:
        # If at least one of the matches is not final, mark as undetermined.
        # This can happen with `d = {111: 'b', (111, 222): 'a'}` where
        # `111` appears final on first match but is not final on the second.

        if isinstance(k, tuple):
            if filter_prefix_tuple(k):
                key_fragment = k[prefix_tuple_size]
                filtered_key_is_final[key_fragment] |= (
                    _DictKeyState.END_OF_TUPLE
                    if len(k) == prefix_tuple_size + 1
                    else _DictKeyState.IN_TUPLE
                )
        elif prefix_tuple_size > 0:
            # we are completing a tuple but this key is not a tuple,
            # so we should ignore it
            pass
        else:
            if isinstance(k, text_serializable_types):
                filtered_key_is_final[k] |= _DictKeyState.END_OF_ITEM

    filtered_keys = filtered_key_is_final.keys()

    if not prefix:
        return "", 0, {repr(k): v for k, v in filtered_key_is_final.items()}

    quote_match = re.search("(?:\"|')", prefix)
    is_user_prefix_numeric = False

    if quote_match:
        quote = quote_match.group()
        valid_prefix = prefix + quote
        try:
            prefix_str = literal_eval(valid_prefix)
        except Exception:
            return "", 0, {}
    else:
        # If it does not look like a string, let's assume
        # we are dealing with a number or variable.
        number_match = _match_number_in_dict_key_prefix(prefix)

        # We do not want the key matcher to suggest variable names so we yield:
        if number_match is None:
            # The alternative would be to assume that user forgort the quote
            # and if the substring matches, suggest adding it at the start.
            return "", 0, {}

        prefix_str = number_match
        is_user_prefix_numeric = True
        quote = ""

    pattern = '[^' + ''.join('\\' + c for c in delims) + ']*$'
    token_match = re.search(pattern, prefix, re.UNICODE)
    assert token_match is not None # silence mypy
    token_start = token_match.start()
    token_prefix = token_match.group()

    matched: Dict[str, _DictKeyState] = {}

    str_key: Union[str, bytes]

    for key in filtered_keys:
        if isinstance(key, (int, float)):
            # User typed a number but this key is not a number.
            if not is_user_prefix_numeric:
                continue
            str_key = str(key)
            if isinstance(key, int):
                int_base = prefix_str[:2].lower()
                # if user typed integer using binary/oct/hex notation:
                if int_base in _INT_FORMATS:
                    int_format = _INT_FORMATS[int_base]
                    str_key = int_format(key)
        else:
            # User typed a string but this key is a number.
            if is_user_prefix_numeric:
                continue
            str_key = key
        try:
            if not str_key.startswith(prefix_str):
                continue
        except (AttributeError, TypeError, UnicodeError) as e:
            # Python 3+ TypeError on b'a'.startswith('a') or vice-versa
            continue

        # reformat remainder of key to begin with prefix
        rem = str_key[len(prefix_str) :]
        # force repr wrapped in '
        rem_repr = repr(rem + '"') if isinstance(rem, str) else repr(rem + b'"')
        rem_repr = rem_repr[1 + rem_repr.index("'"):-2]
        if quote == '"':
            # The entered prefix is quoted with ",
            # but the match is quoted with '.
            # A contained " hence needs escaping for comparison:
            rem_repr = rem_repr.replace('"', '\\"')

        # then reinsert prefix from start of token
        match = "%s%s" % (token_prefix, rem_repr)

        matched[match] = filtered_key_is_final[key]
    return quote, token_start, matched


def cursor_to_position(text:str, line:int, column:int)->int:
    """
    Convert the (line,column) position of the cursor in text to an offset in a
    string.

    Parameters
    ----------
    text : str
        The text in which to calculate the cursor offset
    line : int
        Line of the cursor; 0-indexed
    column : int
        Column of the cursor 0-indexed

    Returns
    -------
    Position of the cursor in ``text``, 0-indexed.

    See Also
    --------
    position_to_cursor : reciprocal of this function

    """
    lines = text.split('\n')
    assert line <= len(lines), '{} <= {}'.format(str(line), str(len(lines)))

    return sum(len(l) + 1 for l in lines[:line]) + column

def position_to_cursor(text:str, offset:int)->Tuple[int, int]:
    """
    Convert the position of the cursor in text (0 indexed) to a line
    number(0-indexed) and a column number (0-indexed) pair

    Position should be a valid position in ``text``.

    Parameters
    ----------
    text : str
        The text in which to calculate the cursor offset
    offset : int
        Position of the cursor in ``text``, 0-indexed.

    Returns
    -------
    (line, column) : (int, int)
        Line of the cursor; 0-indexed, column of the cursor 0-indexed

    See Also
    --------
    cursor_to_position : reciprocal of this function

    """

    assert 0 <= offset <= len(text) , "0 <= %s <= %s" % (offset , len(text))

    before = text[:offset]
    blines = before.split('\n')  # ! splitnes trim trailing \n
    line = before.count('\n')
    col = len(blines[-1])
    return line, col


def _safe_isinstance(obj, module, class_name, *attrs):
    """Checks if obj is an instance of module.class_name if loaded
    """
    if module in sys.modules:
        m = sys.modules[module]
        for attr in [class_name, *attrs]:
            m = getattr(m, attr)
        return isinstance(obj, m)


@context_matcher()
def back_unicode_name_matcher(context: CompletionContext):
    """Match Unicode characters back to Unicode name

    Same as :any:`back_unicode_name_matches`, but adopted to new Matcher API.
    """
    fragment, matches = back_unicode_name_matches(context.text_until_cursor)
    return _convert_matcher_v1_result_to_v2(
        matches, type="unicode", fragment=fragment, suppress_if_matches=True
    )


def back_unicode_name_matches(text: str) -> Tuple[str, Sequence[str]]:
    """Match Unicode characters back to Unicode name

    This does  ``☃`` -> ``\\snowman``

    Note that snowman is not a valid python3 combining character but will be expanded.
    Though it will not recombine back to the snowman character by the completion machinery.

    This will not either back-complete standard sequences like \\n, \\b ...

    .. deprecated:: 8.6
        You can use :meth:`back_unicode_name_matcher` instead.

    Returns
    =======

    Return a tuple with two elements:

    - The Unicode character that was matched (preceded with a backslash), or
        empty string,
    - a sequence (of 1), name for the match Unicode character, preceded by
        backslash, or empty if no match.
    """
    if len(text)<2:
        return '', ()
    maybe_slash = text[-2]
    if maybe_slash != '\\':
        return '', ()

    char = text[-1]
    # no expand on quote for completion in strings.
    # nor backcomplete standard ascii keys
    if char in string.ascii_letters or char in ('"',"'"):
        return '', ()
    try :
        unic = unicodedata.name(char)
        return '\\'+char,('\\'+unic,)
    except KeyError:
        pass
    return '', ()


@context_matcher()
def back_latex_name_matcher(context: CompletionContext):
    """Match latex characters back to unicode name

    Same as :any:`back_latex_name_matches`, but adopted to new Matcher API.
    """
    fragment, matches = back_latex_name_matches(context.text_until_cursor)
    return _convert_matcher_v1_result_to_v2(
        matches, type="latex", fragment=fragment, suppress_if_matches=True
    )


def back_latex_name_matches(text: str) -> Tuple[str, Sequence[str]]:
    """Match latex characters back to unicode name

    This does ``\\ℵ`` -> ``\\aleph``

    .. deprecated:: 8.6
        You can use :meth:`back_latex_name_matcher` instead.
    """
    if len(text)<2:
        return '', ()
    maybe_slash = text[-2]
    if maybe_slash != '\\':
        return '', ()


    char = text[-1]
    # no expand on quote for completion in strings.
    # nor backcomplete standard ascii keys
    if char in string.ascii_letters or char in ('"',"'"):
        return '', ()
    try :
        latex = reverse_latex_symbol[char]
        # '\\' replace the \ as well
        return '\\'+char,[latex]
    except KeyError:
        pass
    return '', ()


def _formatparamchildren(parameter) -> str:
    """
    Get parameter name and value from Jedi Private API

    Jedi does not expose a simple way to get `param=value` from its API.

    Parameters
    ----------
    parameter
        Jedi's function `Param`

    Returns
    -------
    A string like 'a', 'b=1', '*args', '**kwargs'

    """
    description = parameter.description
    if not description.startswith('param '):
        raise ValueError('Jedi function parameter description have change format.'
                         'Expected "param ...", found %r".' % description)
    return description[6:]

def _make_signature(completion)-> str:
    """
    Make the signature from a jedi completion

    Parameters
    ----------
    completion : jedi.Completion
        object does not complete a function type

    Returns
    -------
    a string consisting of the function signature, with the parenthesis but
    without the function name. example:
    `(a, *args, b=1, **kwargs)`

    """

    # it looks like this might work on jedi 0.17
    if hasattr(completion, 'get_signatures'):
        signatures = completion.get_signatures()
        if not signatures:
            return  '(?)'

        c0 = completion.get_signatures()[0]
        return '('+c0.to_string().split('(', maxsplit=1)[1]

    return '(%s)'% ', '.join([f for f in (_formatparamchildren(p) for signature in completion.get_signatures()
                                          for p in signature.defined_names()) if f])


_CompleteResult = Dict[str, MatcherResult]


DICT_MATCHER_REGEX = re.compile(
    r"""(?x)
(  # match dict-referring - or any get item object - expression
    .+
)
\[   # open bracket
\s*  # and optional whitespace
# Capture any number of serializable objects (e.g. "a", "b", 'c')
# and slices
((?:(?:
    (?: # closed string
        [uUbB]?  # string prefix (r not handled)
        (?:
            '(?:[^']|(?<!\\)\\')*'
        |
            "(?:[^"]|(?<!\\)\\")*"
        )
    )
    |
        # capture integers and slices
        (?:[-+]?\d+)?(?::(?:[-+]?\d+)?){0,2}
    |
        # integer in bin/hex/oct notation
        0[bBxXoO]_?(?:\w|\d)+
    )
    \s*,\s*
)*)
((?:
    (?: # unclosed string
        [uUbB]?  # string prefix (r not handled)
        (?:
            '(?:[^']|(?<!\\)\\')*
            |
            "(?:[^"]|(?<!\\)\\")*
        )
    )
    |
        # unfinished integer
        (?:[-+]?\d+)
    |
        # integer in bin/hex/oct notation
        0[bBxXoO]_?(?:\w|\d)+
    )
)?
$
"""
)


def _convert_matcher_v1_result_to_v2(
    matches: Sequence[str],
    type: str,
    fragment: Optional[str] = None,
    suppress_if_matches: bool = False,
) -> SimpleMatcherResult:
    """Utility to help with transition"""
    result = {
        "completions": [SimpleCompletion(text=match, type=type) for match in matches],
        "suppress": (True if matches else False) if suppress_if_matches else False,
    }
    if fragment is not None:
        result["matched_fragment"] = fragment
    return cast(SimpleMatcherResult, result)


class IPCompleter(Completer):
    """Extension of the completer class with IPython-specific features"""

    @observe('greedy')
    def _greedy_changed(self, change):
        """update the splitter and readline delims when greedy is changed"""
        if change["new"]:
            self.evaluation = "unsafe"
            self.auto_close_dict_keys = True
            self.splitter.delims = GREEDY_DELIMS
        else:
            self.evaluation = "limited"
            self.auto_close_dict_keys = False
            self.splitter.delims = DELIMS

    dict_keys_only = Bool(
        False,
        help="""
        Whether to show dict key matches only.

        (disables all matchers except for `IPCompleter.dict_key_matcher`).
        """,
    )

    suppress_competing_matchers = UnionTrait(
        [Bool(allow_none=True), DictTrait(Bool(None, allow_none=True))],
        default_value=None,
        help="""
        Whether to suppress completions from other *Matchers*.

        When set to ``None`` (default) the matchers will attempt to auto-detect
        whether suppression of other matchers is desirable. For example, at
        the beginning of a line followed by `%` we expect a magic completion
        to be the only applicable option, and after ``my_dict['`` we usually
        expect a completion with an existing dictionary key.

        If you want to disable this heuristic and see completions from all matchers,
        set ``IPCompleter.suppress_competing_matchers = False``.
        To disable the heuristic for specific matchers provide a dictionary mapping:
        ``IPCompleter.suppress_competing_matchers = {'IPCompleter.dict_key_matcher': False}``.

        Set ``IPCompleter.suppress_competing_matchers = True`` to limit
        completions to the set of matchers with the highest priority;
        this is equivalent to ``IPCompleter.merge_completions`` and
        can be beneficial for performance, but will sometimes omit relevant
        candidates from matchers further down the priority list.
        """,
    ).tag(config=True)

    merge_completions = Bool(
        True,
        help="""Whether to merge completion results into a single list

        If False, only the completion results from the first non-empty
        completer will be returned.

        As of version 8.6.0, setting the value to ``False`` is an alias for:
        ``IPCompleter.suppress_competing_matchers = True.``.
        """,
    ).tag(config=True)

    disable_matchers = ListTrait(
        Unicode(),
        help="""List of matchers to disable.

        The list should contain matcher identifiers (see :any:`completion_matcher`).
        """,
    ).tag(config=True)

    omit__names = Enum(
        (0, 1, 2),
        default_value=2,
        help="""Instruct the completer to omit private method names

        Specifically, when completing on ``object.<tab>``.

        When 2 [default]: all names that start with '_' will be excluded.

        When 1: all 'magic' names (``__foo__``) will be excluded.

        When 0: nothing will be excluded.
        """
    ).tag(config=True)
    limit_to__all__ = Bool(False,
        help="""
        DEPRECATED as of version 5.0.

        Instruct the completer to use __all__ for the completion

        Specifically, when completing on ``object.<tab>``.

        When True: only those names in obj.__all__ will be included.

        When False [default]: the __all__ attribute is ignored
        """,
    ).tag(config=True)

    profile_completions = Bool(
        default_value=False,
        help="If True, emit profiling data for completion subsystem using cProfile."
    ).tag(config=True)

    profiler_output_dir = Unicode(
        default_value=".completion_profiles",
        help="Template for path at which to output profile data for completions."
    ).tag(config=True)

    @observe('limit_to__all__')
    def _limit_to_all_changed(self, change):
        warnings.warn('`IPython.core.IPCompleter.limit_to__all__` configuration '
            'value has been deprecated since IPython 5.0, will be made to have '
            'no effects and then removed in future version of IPython.',
            UserWarning)

    def __init__(
        self, shell=None, namespace=None, global_namespace=None, config=None, **kwargs
    ):
        """IPCompleter() -> completer

        Return a completer object.

        Parameters
        ----------
        shell
            a pointer to the ipython shell itself.  This is needed
            because this completer knows about magic functions, and those can
            only be accessed via the ipython instance.
        namespace : dict, optional
            an optional dict where completions are performed.
        global_namespace : dict, optional
            secondary optional dict for completions, to
            handle cases (such as IPython embedded inside functions) where
            both Python scopes are visible.
        config : Config
            traitlet's config object
        **kwargs
            passed to super class unmodified.
        """

        self.magic_escape = ESC_MAGIC
        self.splitter = CompletionSplitter()

        # _greedy_changed() depends on splitter and readline being defined:
        super().__init__(
            namespace=namespace,
            global_namespace=global_namespace,
            config=config,
            **kwargs,
        )

        # List where completion matches will be stored
        self.matches = []
        self.shell = shell
        # Regexp to split filenames with spaces in them
        self.space_name_re = re.compile(r'([^\\] )')
        # Hold a local ref. to glob.glob for speed
        self.glob = glob.glob

        # Determine if we are running on 'dumb' terminals, like (X)Emacs
        # buffers, to avoid completion problems.
        term = os.environ.get('TERM','xterm')
        self.dumb_terminal = term in ['dumb','emacs']

        # Special handling of backslashes needed in win32 platforms
        if sys.platform == "win32":
            self.clean_glob = self._clean_glob_win32
        else:
            self.clean_glob = self._clean_glob

        #regexp to parse docstring for function signature
        self.docstring_sig_re = re.compile(r'^[\w|\s.]+\(([^)]*)\).*')
        self.docstring_kwd_re = re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
        #use this if positional argument name is also needed
        #= re.compile(r'[\s|\[]*(\w+)(?:\s*=?\s*.*)')

        self.magic_arg_matchers = [
            self.magic_config_matcher,
            self.magic_color_matcher,
        ]

        # This is set externally by InteractiveShell
        self.custom_completers = None

        # This is a list of names of unicode characters that can be completed
        # into their corresponding unicode value. The list is large, so we
        # lazily initialize it on first use. Consuming code should access this
        # attribute through the `@unicode_names` property.
        self._unicode_names = None

        self._backslash_combining_matchers = [
            self.latex_name_matcher,
            self.unicode_name_matcher,
            back_latex_name_matcher,
            back_unicode_name_matcher,
            self.fwd_unicode_matcher,
        ]

        if not self.backslash_combining_completions:
            for matcher in self._backslash_combining_matchers:
                self.disable_matchers.append(_get_matcher_id(matcher))

        if not self.merge_completions:
            self.suppress_competing_matchers = True

    @property
    def matchers(self) -> List[Matcher]:
        """All active matcher routines for completion"""
        if self.dict_keys_only:
            return [self.dict_key_matcher]

        if self.use_jedi:
            return [
                *self.custom_matchers,
                *self._backslash_combining_matchers,
                *self.magic_arg_matchers,
                self.custom_completer_matcher,
                self.magic_matcher,
                self._jedi_matcher,
                self.dict_key_matcher,
                self.file_matcher,
            ]
        else:
            return [
                *self.custom_matchers,
                *self._backslash_combining_matchers,
                *self.magic_arg_matchers,
                self.custom_completer_matcher,
                self.dict_key_matcher,
                # TODO: convert python_matches to v2 API
                self.magic_matcher,
                self.python_matches,
                self.file_matcher,
                self.python_func_kw_matcher,
            ]

    def all_completions(self, text:str) -> List[str]:
        """
        Wrapper around the completion methods for the benefit of emacs.
        """
        prefix = text.rpartition('.')[0]
        with provisionalcompleter():
            return ['.'.join([prefix, c.text]) if prefix and self.use_jedi else c.text
                    for c in self.completions(text, len(text))]

        return self.complete(text)[1]

    def _clean_glob(self, text:str):
        return self.glob("%s*" % text)

    def _clean_glob_win32(self, text:str):
        return [f.replace("\\","/")
                for f in self.glob("%s*" % text)]

    @context_matcher()
    def file_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Same as :any:`file_matches`, but adopted to new Matcher API."""
        matches = self.file_matches(context.token)
        # TODO: add a heuristic for suppressing (e.g. if it has OS-specific delimiter,
        #  starts with `/home/`, `C:\`, etc)
        return _convert_matcher_v1_result_to_v2(matches, type="path")

    def file_matches(self, text: str) -> List[str]:
        """Match filenames, expanding ~USER type strings.

        Most of the seemingly convoluted logic in this completer is an
        attempt to handle filenames with spaces in them.  And yet it's not
        quite perfect, because Python's readline doesn't expose all of the
        GNU readline details needed for this to be done correctly.

        For a filename with a space in it, the printed completions will be
        only the parts after what's already been typed (instead of the
        full completions, as is normally done).  I don't think with the
        current (as of Python 2.3) Python readline it's possible to do
        better.

        .. deprecated:: 8.6
            You can use :meth:`file_matcher` instead.
        """

        # chars that require escaping with backslash - i.e. chars
        # that readline treats incorrectly as delimiters, but we
        # don't want to treat as delimiters in filename matching
        # when escaped with backslash
        if text.startswith('!'):
            text = text[1:]
            text_prefix = u'!'
        else:
            text_prefix = u''

        text_until_cursor = self.text_until_cursor
        # track strings with open quotes
        open_quotes = has_open_quotes(text_until_cursor)

        if '(' in text_until_cursor or '[' in text_until_cursor:
            lsplit = text
        else:
            try:
                # arg_split ~ shlex.split, but with unicode bugs fixed by us
                lsplit = arg_split(text_until_cursor)[-1]
            except ValueError:
                # typically an unmatched ", or backslash without escaped char.
                if open_quotes:
                    lsplit = text_until_cursor.split(open_quotes)[-1]
                else:
                    return []
            except IndexError:
                # tab pressed on empty line
                lsplit = ""

        if not open_quotes and lsplit != protect_filename(lsplit):
            # if protectables are found, do matching on the whole escaped name
            has_protectables = True
            text0,text = text,lsplit
        else:
            has_protectables = False
            text = os.path.expanduser(text)

        if text == "":
            return [text_prefix + protect_filename(f) for f in self.glob("*")]

        # Compute the matches from the filesystem
        if sys.platform == 'win32':
            m0 = self.clean_glob(text)
        else:
            m0 = self.clean_glob(text.replace('\\', ''))

        if has_protectables:
            # If we had protectables, we need to revert our changes to the
            # beginning of filename so that we don't double-write the part
            # of the filename we have so far
            len_lsplit = len(lsplit)
            matches = [text_prefix + text0 +
                       protect_filename(f[len_lsplit:]) for f in m0]
        else:
            if open_quotes:
                # if we have a string with an open quote, we don't need to
                # protect the names beyond the quote (and we _shouldn't_, as
                # it would cause bugs when the filesystem call is made).
                matches = m0 if sys.platform == "win32" else\
                    [protect_filename(f, open_quotes) for f in m0]
            else:
                matches = [text_prefix +
                           protect_filename(f) for f in m0]

        # Mark directories in input list by appending '/' to their names.
        return [x+'/' if os.path.isdir(x) else x for x in matches]

    @context_matcher()
    def magic_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match magics."""
        text = context.token
        matches = self.magic_matches(text)
        result = _convert_matcher_v1_result_to_v2(matches, type="magic")
        is_magic_prefix = len(text) > 0 and text[0] == "%"
        result["suppress"] = is_magic_prefix and bool(result["completions"])
        return result

    def magic_matches(self, text: str):
        """Match magics.

        .. deprecated:: 8.6
            You can use :meth:`magic_matcher` instead.
        """
        # Get all shell magics now rather than statically, so magics loaded at
        # runtime show up too.
        lsm = self.shell.magics_manager.lsmagic()
        line_magics = lsm['line']
        cell_magics = lsm['cell']
        pre = self.magic_escape
        pre2 = pre+pre

        explicit_magic = text.startswith(pre)

        # Completion logic:
        # - user gives %%: only do cell magics
        # - user gives %: do both line and cell magics
        # - no prefix: do both
        # In other words, line magics are skipped if the user gives %% explicitly
        #
        # We also exclude magics that match any currently visible names:
        # https://github.com/ipython/ipython/issues/4877, unless the user has
        # typed a %:
        # https://github.com/ipython/ipython/issues/10754
        bare_text = text.lstrip(pre)
        global_matches = self.global_matches(bare_text)
        if not explicit_magic:
            def matches(magic):
                """
                Filter magics, in particular remove magics that match
                a name present in global namespace.
                """
                return ( magic.startswith(bare_text) and
                         magic not in global_matches )
        else:
            def matches(magic):
                return magic.startswith(bare_text)

        comp = [ pre2+m for m in cell_magics if matches(m)]
        if not text.startswith(pre2):
            comp += [ pre+m for m in line_magics if matches(m)]

        return comp

    @context_matcher()
    def magic_config_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match class names and attributes for %config magic."""
        # NOTE: uses `line_buffer` equivalent for compatibility
        matches = self.magic_config_matches(context.line_with_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type="param")

    def magic_config_matches(self, text: str) -> List[str]:
        """Match class names and attributes for %config magic.

        .. deprecated:: 8.6
            You can use :meth:`magic_config_matcher` instead.
        """
        texts = text.strip().split()

        if len(texts) > 0 and (texts[0] == 'config' or texts[0] == '%config'):
            # get all configuration classes
            classes = sorted(set([ c for c in self.shell.configurables
                                   if c.__class__.class_traits(config=True)
                                   ]), key=lambda x: x.__class__.__name__)
            classnames = [ c.__class__.__name__ for c in classes ]

            # return all classnames if config or %config is given
            if len(texts) == 1:
                return classnames

            # match classname
            classname_texts = texts[1].split('.')
            classname = classname_texts[0]
            classname_matches = [ c for c in classnames
                                  if c.startswith(classname) ]

            # return matched classes or the matched class with attributes
            if texts[1].find('.') < 0:
                return classname_matches
            elif len(classname_matches) == 1 and \
                            classname_matches[0] == classname:
                cls = classes[classnames.index(classname)].__class__
                help = cls.class_get_help()
                # strip leading '--' from cl-args:
                help = re.sub(re.compile(r'^--', re.MULTILINE), '', help)
                return [ attr.split('=')[0]
                         for attr in help.strip().splitlines()
                         if attr.startswith(texts[1]) ]
        return []

    @context_matcher()
    def magic_color_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match color schemes for %colors magic."""
        # NOTE: uses `line_buffer` equivalent for compatibility
        matches = self.magic_color_matches(context.line_with_cursor)
        return _convert_matcher_v1_result_to_v2(matches, type="param")

    def magic_color_matches(self, text: str) -> List[str]:
        """Match color schemes for %colors magic.

        .. deprecated:: 8.6
            You can use :meth:`magic_color_matcher` instead.
        """
        texts = text.split()
        if text.endswith(' '):
            # .split() strips off the trailing whitespace. Add '' back
            # so that: '%colors ' -> ['%colors', '']
            texts.append('')

        if len(texts) == 2 and (texts[0] == 'colors' or texts[0] == '%colors'):
            prefix = texts[1]
            return [ color for color in InspectColors.keys()
                     if color.startswith(prefix) ]
        return []

    @context_matcher(identifier="IPCompleter.jedi_matcher")
    def _jedi_matcher(self, context: CompletionContext) -> _JediMatcherResult:
        matches = self._jedi_matches(
            cursor_column=context.cursor_position,
            cursor_line=context.cursor_line,
            text=context.full_text,
        )
        return {
            "completions": matches,
            # static analysis should not suppress other matchers
            "suppress": False,
        }

    def _jedi_matches(
        self, cursor_column: int, cursor_line: int, text: str
    ) -> Iterator[_JediCompletionLike]:
        """
        Return a list of :any:`jedi.api.Completion`s object from a ``text`` and
        cursor position.

        Parameters
        ----------
        cursor_column : int
            column position of the cursor in ``text``, 0-indexed.
        cursor_line : int
            line position of the cursor in ``text``, 0-indexed
        text : str
            text to complete

        Notes
        -----
        If ``IPCompleter.debug`` is ``True`` may return a :any:`_FakeJediCompletion`
        object containing a string with the Jedi debug information attached.

        .. deprecated:: 8.6
            You can use :meth:`_jedi_matcher` instead.
        """
        namespaces = [self.namespace]
        if self.global_namespace is not None:
            namespaces.append(self.global_namespace)

        completion_filter = lambda x:x
        offset = cursor_to_position(text, cursor_line, cursor_column)
        # filter output if we are completing for object members
        if offset:
            pre = text[offset-1]
            if pre == '.':
                if self.omit__names == 2:
                    completion_filter = lambda c:not c.name.startswith('_')
                elif self.omit__names == 1:
                    completion_filter = lambda c:not (c.name.startswith('__') and c.name.endswith('__'))
                elif self.omit__names == 0:
                    completion_filter = lambda x:x
                else:
                    raise ValueError("Don't understand self.omit__names == {}".format(self.omit__names))

        interpreter = jedi.Interpreter(text[:offset], namespaces)
        try_jedi = True

        try:
            # find the first token in the current tree -- if it is a ' or " then we are in a string
            completing_string = False
            try:
                first_child = next(c for c in interpreter._get_module().tree_node.children if hasattr(c, 'value'))
            except StopIteration:
                pass
            else:
                # note the value may be ', ", or it may also be ''' or """, or
                # in some cases, """what/you/typed..., but all of these are
                # strings.
                completing_string = len(first_child.value) > 0 and first_child.value[0] in {"'", '"'}

            # if we are in a string jedi is likely not the right candidate for
            # now. Skip it.
            try_jedi = not completing_string
        except Exception as e:
            # many of things can go wrong, we are using private API just don't crash.
            if self.debug:
                print("Error detecting if completing a non-finished string :", e, '|')

        if not try_jedi:
            return iter([])
        try:
            return filter(completion_filter, interpreter.complete(column=cursor_column, line=cursor_line + 1))
        except Exception as e:
            if self.debug:
                return iter(
                    [
                        _FakeJediCompletion(
                            'Oops Jedi has crashed, please report a bug with the following:\n"""\n%s\ns"""'
                            % (e)
                        )
                    ]
                )
            else:
                return iter([])

    @completion_matcher(api_version=1)
    def python_matches(self, text: str) -> Iterable[str]:
        """Match attributes or global python names"""
        if "." in text:
            try:
                matches = self.attr_matches(text)
                if text.endswith('.') and self.omit__names:
                    if self.omit__names == 1:
                        # true if txt is _not_ a __ name, false otherwise:
                        no__name = (lambda txt:
                                    re.match(r'.*\.__.*?__',txt) is None)
                    else:
                        # true if txt is _not_ a _ name, false otherwise:
                        no__name = (lambda txt:
                                    re.match(r'\._.*?',txt[txt.rindex('.'):]) is None)
                    matches = filter(no__name, matches)
            except NameError:
                # catches <undefined attributes>.<tab>
                matches = []
        else:
            matches = self.global_matches(text)
        return matches

    def _default_arguments_from_docstring(self, doc):
        """Parse the first line of docstring for call signature.

        Docstring should be of the form 'min(iterable[, key=func])\n'.
        It can also parse cython docstring of the form
        'Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)'.
        """
        if doc is None:
            return []

        #care only the firstline
        line = doc.lstrip().splitlines()[0]

        #p = re.compile(r'^[\w|\s.]+\(([^)]*)\).*')
        #'min(iterable[, key=func])\n' -> 'iterable[, key=func]'
        sig = self.docstring_sig_re.search(line)
        if sig is None:
            return []
        # iterable[, key=func]' -> ['iterable[' ,' key=func]']
        sig = sig.groups()[0].split(',')
        ret = []
        for s in sig:
            #re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
            ret += self.docstring_kwd_re.findall(s)
        return ret

    def _default_arguments(self, obj):
        """Return the list of default arguments of obj if it is callable,
        or empty list otherwise."""
        call_obj = obj
        ret = []
        if inspect.isbuiltin(obj):
            pass
        elif not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            if inspect.isclass(obj):
                #for cython embedsignature=True the constructor docstring
                #belongs to the object itself not __init__
                ret += self._default_arguments_from_docstring(
                            getattr(obj, '__doc__', ''))
                # for classes, check for __init__,__new__
                call_obj = (getattr(obj, '__init__', None) or
                       getattr(obj, '__new__', None))
            # for all others, check if they are __call__able
            elif hasattr(obj, '__call__'):
                call_obj = obj.__call__
        ret += self._default_arguments_from_docstring(
                 getattr(call_obj, '__doc__', ''))

        _keeps = (inspect.Parameter.KEYWORD_ONLY,
                  inspect.Parameter.POSITIONAL_OR_KEYWORD)

        try:
            sig = inspect.signature(obj)
            ret.extend(k for k, v in sig.parameters.items() if
                       v.kind in _keeps)
        except ValueError:
            pass

        return list(set(ret))

    @context_matcher()
    def python_func_kw_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match named parameters (kwargs) of the last open function."""
        matches = self.python_func_kw_matches(context.token)
        return _convert_matcher_v1_result_to_v2(matches, type="param")

    def python_func_kw_matches(self, text):
        """Match named parameters (kwargs) of the last open function.

        .. deprecated:: 8.6
            You can use :meth:`python_func_kw_matcher` instead.
        """

        if "." in text: # a parameter cannot be dotted
            return []
        try: regexp = self.__funcParamsRegex
        except AttributeError:
            regexp = self.__funcParamsRegex = re.compile(r'''
                '.*?(?<!\\)' |    # single quoted strings or
                ".*?(?<!\\)" |    # double quoted strings or
                \w+          |    # identifier
                \S                # other characters
                ''', re.VERBOSE | re.DOTALL)
        # 1. find the nearest identifier that comes before an unclosed
        # parenthesis before the cursor
        # e.g. for "foo (1+bar(x), pa<cursor>,a=1)", the candidate is "foo"
        tokens = regexp.findall(self.text_until_cursor)
        iterTokens = reversed(tokens); openPar = 0

        for token in iterTokens:
            if token == ')':
                openPar -= 1
            elif token == '(':
                openPar += 1
                if openPar > 0:
                    # found the last unclosed parenthesis
                    break
        else:
            return []
        # 2. Concatenate dotted names ("foo.bar" for "foo.bar(x, pa" )
        ids = []
        isId = re.compile(r'\w+$').match

        while True:
            try:
                ids.append(next(iterTokens))
                if not isId(ids[-1]):
                    ids.pop(); break
                if not next(iterTokens) == '.':
                    break
            except StopIteration:
                break

        # Find all named arguments already assigned to, as to avoid suggesting
        # them again
        usedNamedArgs = set()
        par_level = -1
        for token, next_token in zip(tokens, tokens[1:]):
            if token == '(':
                par_level += 1
            elif token == ')':
                par_level -= 1

            if par_level != 0:
                continue

            if next_token != '=':
                continue

            usedNamedArgs.add(token)

        argMatches = []
        try:
            callableObj = '.'.join(ids[::-1])
            namedArgs = self._default_arguments(eval(callableObj,
                                                    self.namespace))

            # Remove used named arguments from the list, no need to show twice
            for namedArg in set(namedArgs) - usedNamedArgs:
                if namedArg.startswith(text):
                    argMatches.append("%s=" %namedArg)
        except:
            pass

        return argMatches

    @staticmethod
    def _get_keys(obj: Any) -> List[Any]:
        # Objects can define their own completions by defining an
        # _ipy_key_completions_() method.
        method = get_real_method(obj, '_ipython_key_completions_')
        if method is not None:
            return method()

        # Special case some common in-memory dict-like types
        if isinstance(obj, dict) or _safe_isinstance(obj, "pandas", "DataFrame"):
            try:
                return list(obj.keys())
            except Exception:
                return []
        elif _safe_isinstance(obj, "pandas", "core", "indexing", "_LocIndexer"):
            try:
                return list(obj.obj.keys())
            except Exception:
                return []
        elif _safe_isinstance(obj, 'numpy', 'ndarray') or\
             _safe_isinstance(obj, 'numpy', 'void'):
            return obj.dtype.names or []
        return []

    @context_matcher()
    def dict_key_matcher(self, context: CompletionContext) -> SimpleMatcherResult:
        """Match string keys in a dictionary, after e.g. ``foo[``."""
        matches = self.dict_key_matches(context.token)
        return _convert_matcher_v1_result_to_v2(
            matches, type="dict key", suppress_if_matches=True
        )

    def dict_key_matches(self, text: str) -> List[str]:
        """Match string keys in a dictionary, after e.g. ``foo[``.

        .. deprecated:: 8.6
            You can use :meth:`dict_key_matcher` instead.
        """

        # Short-circuit on closed dictionary (regular expression would
        # not match anyway, but would take quite a while).
        if self.text_until_cursor.strip().endswith("]"):
            return []

        match = DICT_MATCHER_REGEX.search(self.text_until_cursor)

        if match is None:
            return []

        expr, prior_tuple_keys, key_prefix = match.groups()

        obj = self._evaluate_expr(expr)

        if obj is not_found:
            return []

        keys = self._get_keys(obj)
        if not keys:
            return keys

        tuple_prefix = guarded_eval(
            prior_tuple_keys,
            EvaluationContext(
                globals=self.global_namespace,
                locals=self.namespace,
                evaluation=self.evaluation,
                in_subscript=True,
            ),
        )

        closing_quote, token_offset, matches = match_dict_keys(
            keys, key_prefix, self.splitter.delims, extra_prefix=tuple_prefix
        )
        if not matches:
            return []

        # get the cursor position of
        # - the text being completed
        # - the start of the key text
        # - the start of the completion
        text_start = len(self.text_until_cursor) - len(text)
        if key_prefix:
            key_start = match.start(3)
            completion_start = key_start + token_offset
        else:
            key_start = completion_start = match.end()

        # grab the leading prefix, to make sure all completions start with `text`
        if text_start > key_start:
            leading = ''
        else:
            leading = text[text_start:completion_start]

        # append closing quote and bracket as appropriate
        # this is *not* appropriate if the opening quote or bracket is outside
        # the text given to this method, e.g. `d["""a\nt
        can_close_quote = False
        can_close_bracket = False

        continuation = self.line_buffer[len(self.text_until_cursor) :].strip()

        if continuation.startswith(closing_quote):
            # do not close if already closed, e.g. `d['a<tab>'`
            continuation = continuation[len(closing_quote) :]
        else:
            can_close_quote = True

        continuation = continuation.strip()

        # e.g. `pandas.DataFrame` has different tuple indexer behaviour,
        # handling it is out of scope, so let's avoid appending suffixes.
        has_known_tuple_handling = isinstance(obj, dict)

        can_close_bracket = (
            not continuation.startswith("]") and self.auto_close_dict_keys
        )
        can_close_tuple_item = (
            not continuation.startswith(",")
            and has_known_tuple_handling
            and self.auto_close_dict_keys
        )
        can_close_quote = can_close_quote and self.auto_close_dict_keys

        # fast path if closing qoute should be appended but not suffix is allowed
        if not can_close_quote and not can_close_bracket and closing_quote:
            return [leading + k for k in matches]

        results = []

        end_of_tuple_or_item = _DictKeyState.END_OF_TUPLE | _DictKeyState.END_OF_ITEM

        for k, state_flag in matches.items():
            result = leading + k
            if can_close_quote and closing_quote:
                result += closing_quote

            if state_flag == end_of_tuple_or_item:
                # We do not know which suffix to add,
                # e.g. both tuple item and string
                # match this item.
                pass

            if state_flag in end_of_tuple_or_item and can_close_bracket:
                result += "]"
            if state_flag == _DictKeyState.IN_TUPLE and can_close_tuple_item:
                result += ", "
            results.append(result)
        return results

    @context_matcher()
    def unicode_name_matcher(self, context: CompletionContext):
        """Same as :any:`unicode_name_matches`, but adopted to new Matcher API."""
        fragment, matches = self.unicode_name_matches(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(
            matches, type="unicode", fragment=fragment, suppress_if_matches=True
        )

    @staticmethod
    def unicode_name_matches(text: str) -> Tuple[str, List[str]]:
        """Match Latex-like syntax for unicode characters base
        on the name of the character.

        This does  ``\\GREEK SMALL LETTER ETA`` -> ``η``

        Works only on valid python 3 identifier, or on combining characters that
        will combine to form a valid identifier.
        """
        slashpos = text.rfind('\\')
        if slashpos > -1:
            s = text[slashpos+1:]
            try :
                unic = unicodedata.lookup(s)
                # allow combining chars
                if ('a'+unic).isidentifier():
                    return '\\'+s,[unic]
            except KeyError:
                pass
        return '', []

    @context_matcher()
    def latex_name_matcher(self, context: CompletionContext):
        """Match Latex syntax for unicode characters.

        This does both ``\\alp`` -> ``\\alpha`` and ``\\alpha`` -> ``α``
        """
        fragment, matches = self.latex_matches(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(
            matches, type="latex", fragment=fragment, suppress_if_matches=True
        )

    def latex_matches(self, text: str) -> Tuple[str, Sequence[str]]:
        """Match Latex syntax for unicode characters.

        This does both ``\\alp`` -> ``\\alpha`` and ``\\alpha`` -> ``α``

        .. deprecated:: 8.6
            You can use :meth:`latex_name_matcher` instead.
        """
        slashpos = text.rfind('\\')
        if slashpos > -1:
            s = text[slashpos:]
            if s in latex_symbols:
                # Try to complete a full latex symbol to unicode
                # \\alpha -> α
                return s, [latex_symbols[s]]
            else:
                # If a user has partially typed a latex symbol, give them
                # a full list of options \al -> [\aleph, \alpha]
                matches = [k for k in latex_symbols if k.startswith(s)]
                if matches:
                    return s, matches
        return '', ()

    @context_matcher()
    def custom_completer_matcher(self, context):
        """Dispatch custom completer.

        If a match is found, suppresses all other matchers except for Jedi.
        """
        matches = self.dispatch_custom_completer(context.token) or []
        result = _convert_matcher_v1_result_to_v2(
            matches, type=_UNKNOWN_TYPE, suppress_if_matches=True
        )
        result["ordered"] = True
        result["do_not_suppress"] = {_get_matcher_id(self._jedi_matcher)}
        return result

    def dispatch_custom_completer(self, text):
        """
        .. deprecated:: 8.6
            You can use :meth:`custom_completer_matcher` instead.
        """
        if not self.custom_completers:
            return

        line = self.line_buffer
        if not line.strip():
            return None

        # Create a little structure to pass all the relevant information about
        # the current completion to any custom completer.
        event = SimpleNamespace()
        event.line = line
        event.symbol = text
        cmd = line.split(None,1)[0]
        event.command = cmd
        event.text_until_cursor = self.text_until_cursor

        # for foo etc, try also to find completer for %foo
        if not cmd.startswith(self.magic_escape):
            try_magic = self.custom_completers.s_matches(
                self.magic_escape + cmd)
        else:
            try_magic = []

        for c in itertools.chain(self.custom_completers.s_matches(cmd),
                 try_magic,
                 self.custom_completers.flat_matches(self.text_until_cursor)):
            try:
                res = c(event)
                if res:
                    # first, try case sensitive match
                    withcase = [r for r in res if r.startswith(text)]
                    if withcase:
                        return withcase
                    # if none, then case insensitive ones are ok too
                    text_low = text.lower()
                    return [r for r in res if r.lower().startswith(text_low)]
            except TryNext:
                pass
            except KeyboardInterrupt:
                """
                If custom completer take too long,
                let keyboard interrupt abort and return nothing.
                """
                break

        return None

    def completions(self, text: str, offset: int)->Iterator[Completion]:
        """
        Returns an iterator over the possible completions

        .. warning::

            Unstable

            This function is unstable, API may change without warning.
            It will also raise unless use in proper context manager.

        Parameters
        ----------
        text : str
            Full text of the current input, multi line string.
        offset : int
            Integer representing the position of the cursor in ``text``. Offset
            is 0-based indexed.

        Yields
        ------
        Completion

        Notes
        -----
        The cursor on a text can either be seen as being "in between"
        characters or "On" a character depending on the interface visible to
        the user. For consistency the cursor being on "in between" characters X
        and Y is equivalent to the cursor being "on" character Y, that is to say
        the character the cursor is on is considered as being after the cursor.

        Combining characters may span more that one position in the
        text.

        .. note::

            If ``IPCompleter.debug`` is :any:`True` will yield a ``--jedi/ipython--``
            fake Completion token to distinguish completion returned by Jedi
            and usual IPython completion.

        .. note::

            Completions are not completely deduplicated yet. If identical
            completions are coming from different sources this function does not
            ensure that each completion object will only be present once.
        """
        warnings.warn("_complete is a provisional API (as of IPython 6.0). "
                      "It may change without warnings. "
                      "Use in corresponding context manager.",
                      category=ProvisionalCompleterWarning, stacklevel=2)

        seen = set()
        profiler:Optional[cProfile.Profile]
        try:
            if self.profile_completions:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            else:
                profiler = None

            for c in self._completions(text, offset, _timeout=self.jedi_compute_type_timeout/1000):
                if c and (c in seen):
                    continue
                yield c
                seen.add(c)
        except KeyboardInterrupt:
            """if completions take too long and users send keyboard interrupt,
            do not crash and return ASAP. """
            pass
        finally:
            if profiler is not None:
                profiler.disable()
                ensure_dir_exists(self.profiler_output_dir)
                output_path = os.path.join(self.profiler_output_dir, str(uuid.uuid4()))
                print("Writing profiler output to", output_path)
                profiler.dump_stats(output_path)

    def _completions(self, full_text: str, offset: int, *, _timeout) -> Iterator[Completion]:
        """
        Core completion module.Same signature as :any:`completions`, with the
        extra `timeout` parameter (in seconds).

        Computing jedi's completion ``.type`` can be quite expensive (it is a
        lazy property) and can require some warm-up, more warm up than just
        computing the ``name`` of a completion. The warm-up can be :

            - Long warm-up the first time a module is encountered after
            install/update: actually build parse/inference tree.

            - first time the module is encountered in a session: load tree from
            disk.

        We don't want to block completions for tens of seconds so we give the
        completer a "budget" of ``_timeout`` seconds per invocation to compute
        completions types, the completions that have not yet been computed will
        be marked as "unknown" an will have a chance to be computed next round
        are things get cached.

        Keep in mind that Jedi is not the only thing treating the completion so
        keep the timeout short-ish as if we take more than 0.3 second we still
        have lots of processing to do.

        """
        deadline = time.monotonic() + _timeout

        before = full_text[:offset]
        cursor_line, cursor_column = position_to_cursor(full_text, offset)

        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)

        def is_non_jedi_result(
            result: MatcherResult, identifier: str
        ) -> TypeGuard[SimpleMatcherResult]:
            return identifier != jedi_matcher_id

        results = self._complete(
            full_text=full_text, cursor_line=cursor_line, cursor_pos=cursor_column
        )

        non_jedi_results: Dict[str, SimpleMatcherResult] = {
            identifier: result
            for identifier, result in results.items()
            if is_non_jedi_result(result, identifier)
        }

        jedi_matches = (
            cast(_JediMatcherResult, results[jedi_matcher_id])["completions"]
            if jedi_matcher_id in results
            else ()
        )

        iter_jm = iter(jedi_matches)
        if _timeout:
            for jm in iter_jm:
                try:
                    type_ = jm.type
                except Exception:
                    if self.debug:
                        print("Error in Jedi getting type of ", jm)
                    type_ = None
                delta = len(jm.name_with_symbols) - len(jm.complete)
                if type_ == 'function':
                    signature = _make_signature(jm)
                else:
                    signature = ''
                yield Completion(start=offset - delta,
                                 end=offset,
                                 text=jm.name_with_symbols,
                                 type=type_,
                                 signature=signature,
                                 _origin='jedi')

                if time.monotonic() > deadline:
                    break

        for jm in iter_jm:
            delta = len(jm.name_with_symbols) - len(jm.complete)
            yield Completion(
                start=offset - delta,
                end=offset,
                text=jm.name_with_symbols,
                type=_UNKNOWN_TYPE,  # don't compute type for speed
                _origin="jedi",
                signature="",
            )

        # TODO:
        # Suppress this, right now just for debug.
        if jedi_matches and non_jedi_results and self.debug:
            some_start_offset = before.rfind(
                next(iter(non_jedi_results.values()))["matched_fragment"]
            )
            yield Completion(
                start=some_start_offset,
                end=offset,
                text="--jedi/ipython--",
                _origin="debug",
                type="none",
                signature="",
            )

        ordered: List[Completion] = []
        sortable: List[Completion] = []

        for origin, result in non_jedi_results.items():
            matched_text = result["matched_fragment"]
            start_offset = before.rfind(matched_text)
            is_ordered = result.get("ordered", False)
            container = ordered if is_ordered else sortable

            # I'm unsure if this is always true, so let's assert and see if it
            # crash
            assert before.endswith(matched_text)

            for simple_completion in result["completions"]:
                completion = Completion(
                    start=start_offset,
                    end=offset,
                    text=simple_completion.text,
                    _origin=origin,
                    signature="",
                    type=simple_completion.type or _UNKNOWN_TYPE,
                )
                container.append(completion)

        yield from list(self._deduplicate(ordered + self._sort(sortable)))[
            :MATCHES_LIMIT
        ]

    def complete(self, text=None, line_buffer=None, cursor_pos=None) -> Tuple[str, Sequence[str]]:
        """Find completions for the given text and line context.

        Note that both the text and the line_buffer are optional, but at least
        one of them must be given.

        Parameters
        ----------
        text : string, optional
            Text to perform the completion on.  If not given, the line buffer
            is split using the instance's CompletionSplitter object.
        line_buffer : string, optional
            If not given, the completer attempts to obtain the current line
            buffer via readline.  This keyword allows clients which are
            requesting for text completions in non-readline contexts to inform
            the completer of the entire text.
        cursor_pos : int, optional
            Index of the cursor in the full line buffer.  Should be provided by
            remote frontends where kernel has no access to frontend state.

        Returns
        -------
        Tuple of two items:
        text : str
            Text that was actually used in the completion.
        matches : list
            A list of completion matches.

        Notes
        -----
            This API is likely to be deprecated and replaced by
            :any:`IPCompleter.completions` in the future.

        """
        warnings.warn('`Completer.complete` is pending deprecation since '
                'IPython 6.0 and will be replaced by `Completer.completions`.',
                      PendingDeprecationWarning)
        # potential todo, FOLD the 3rd throw away argument of _complete
        # into the first 2 one.
        # TODO: Q: does the above refer to jedi completions (i.e. 0-indexed?)
        # TODO: should we deprecate now, or does it stay?

        results = self._complete(
            line_buffer=line_buffer, cursor_pos=cursor_pos, text=text, cursor_line=0
        )

        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)

        return self._arrange_and_extract(
            results,
            # TODO: can we confirm that excluding Jedi here was a deliberate choice in previous version?
            skip_matchers={jedi_matcher_id},
            # this API does not support different start/end positions (fragments of token).
            abort_if_offset_changes=True,
        )

    def _arrange_and_extract(
        self,
        results: Dict[str, MatcherResult],
        skip_matchers: Set[str],
        abort_if_offset_changes: bool,
    ):
        sortable: List[AnyMatcherCompletion] = []
        ordered: List[AnyMatcherCompletion] = []
        most_recent_fragment = None
        for identifier, result in results.items():
            if identifier in skip_matchers:
                continue
            if not result["completions"]:
                continue
            if not most_recent_fragment:
                most_recent_fragment = result["matched_fragment"]
            if (
                abort_if_offset_changes
                and result["matched_fragment"] != most_recent_fragment
            ):
                break
            if result.get("ordered", False):
                ordered.extend(result["completions"])
            else:
                sortable.extend(result["completions"])

        if not most_recent_fragment:
            most_recent_fragment = ""  # to satisfy typechecker (and just in case)

        return most_recent_fragment, [
            m.text for m in self._deduplicate(ordered + self._sort(sortable))
        ]

    def _complete(self, *, cursor_line, cursor_pos, line_buffer=None, text=None,
                  full_text=None) -> _CompleteResult:
        """
        Like complete but can also returns raw jedi completions as well as the
        origin of the completion text. This could (and should) be made much
        cleaner but that will be simpler once we drop the old (and stateful)
        :any:`complete` API.

        With current provisional API, cursor_pos act both (depending on the
        caller) as the offset in the ``text`` or ``line_buffer``, or as the
        ``column`` when passing multiline strings this could/should be renamed
        but would add extra noise.

        Parameters
        ----------
        cursor_line
            Index of the line the cursor is on. 0 indexed.
        cursor_pos
            Position of the cursor in the current line/line_buffer/text. 0
            indexed.
        line_buffer : optional, str
            The current line the cursor is in, this is mostly due to legacy
            reason that readline could only give a us the single current line.
            Prefer `full_text`.
        text : str
            The current "token" the cursor is in, mostly also for historical
            reasons. as the completer would trigger only after the current line
            was parsed.
        full_text : str
            Full text of the current cell.

        Returns
        -------
        An ordered dictionary where keys are identifiers of completion
        matchers and values are ``MatcherResult``s.
        """

        # if the cursor position isn't given, the only sane assumption we can
        # make is that it's at the end of the line (the common case)
        if cursor_pos is None:
            cursor_pos = len(line_buffer) if text is None else len(text)

        if self.use_main_ns:
            self.namespace = __main__.__dict__

        # if text is either None or an empty string, rely on the line buffer
        if (not line_buffer) and full_text:
            line_buffer = full_text.split('\n')[cursor_line]
        if not text:  # issue #11508: check line_buffer before calling split_line
            text = (
                self.splitter.split_line(line_buffer, cursor_pos) if line_buffer else ""
            )

        # If no line buffer is given, assume the input text is all there was
        if line_buffer is None:
            line_buffer = text

        # deprecated - do not use `line_buffer` in new code.
        self.line_buffer = line_buffer
        self.text_until_cursor = self.line_buffer[:cursor_pos]

        if not full_text:
            full_text = line_buffer

        context = CompletionContext(
            full_text=full_text,
            cursor_position=cursor_pos,
            cursor_line=cursor_line,
            token=text,
            limit=MATCHES_LIMIT,
        )

        # Start with a clean slate of completions
        results: Dict[str, MatcherResult] = {}

        jedi_matcher_id = _get_matcher_id(self._jedi_matcher)

        suppressed_matchers: Set[str] = set()

        matchers = {
            _get_matcher_id(matcher): matcher
            for matcher in sorted(
                self.matchers, key=_get_matcher_priority, reverse=True
            )
        }

        for matcher_id, matcher in matchers.items():
            matcher_id = _get_matcher_id(matcher)

            if matcher_id in self.disable_matchers:
                continue

            if matcher_id in results:
                warnings.warn(f"Duplicate matcher ID: {matcher_id}.")

            if matcher_id in suppressed_matchers:
                continue

            result: MatcherResult
            try:
                if _is_matcher_v1(matcher):
                    result = _convert_matcher_v1_result_to_v2(
                        matcher(text), type=_UNKNOWN_TYPE
                    )
                elif _is_matcher_v2(matcher):
                    result = matcher(context)
                else:
                    api_version = _get_matcher_api_version(matcher)
                    raise ValueError(f"Unsupported API version {api_version}")
            except:
                # Show the ugly traceback if the matcher causes an
                # exception, but do NOT crash the kernel!
                sys.excepthook(*sys.exc_info())
                continue

            # set default value for matched fragment if suffix was not selected.
            result["matched_fragment"] = result.get("matched_fragment", context.token)

            if not suppressed_matchers:
                suppression_recommended: Union[bool, Set[str]] = result.get(
                    "suppress", False
                )

                suppression_config = (
                    self.suppress_competing_matchers.get(matcher_id, None)
                    if isinstance(self.suppress_competing_matchers, dict)
                    else self.suppress_competing_matchers
                )
                should_suppress = (
                    (suppression_config is True)
                    or (suppression_recommended and (suppression_config is not False))
                ) and has_any_completions(result)

                if should_suppress:
                    suppression_exceptions: Set[str] = result.get(
                        "do_not_suppress", set()
                    )
                    if isinstance(suppression_recommended, Iterable):
                        to_suppress = set(suppression_recommended)
                    else:
                        to_suppress = set(matchers)
                    suppressed_matchers = to_suppress - suppression_exceptions

                    new_results = {}
                    for previous_matcher_id, previous_result in results.items():
                        if previous_matcher_id not in suppressed_matchers:
                            new_results[previous_matcher_id] = previous_result
                    results = new_results

            results[matcher_id] = result

        _, matches = self._arrange_and_extract(
            results,
            # TODO Jedi completions non included in legacy stateful API; was this deliberate or omission?
            #  if it was omission, we can remove the filtering step, otherwise remove this comment.
            skip_matchers={jedi_matcher_id},
            abort_if_offset_changes=False,
        )

        # populate legacy stateful API
        self.matches = matches

        return results

    @staticmethod
    def _deduplicate(
        matches: Sequence[AnyCompletion],
    ) -> Iterable[AnyCompletion]:
        filtered_matches: Dict[str, AnyCompletion] = {}
        for match in matches:
            text = match.text
            if (
                text not in filtered_matches
                or filtered_matches[text].type == _UNKNOWN_TYPE
            ):
                filtered_matches[text] = match

        return filtered_matches.values()

    @staticmethod
    def _sort(matches: Sequence[AnyCompletion]):
        return sorted(matches, key=lambda x: completions_sorting_key(x.text))

    @context_matcher()
    def fwd_unicode_matcher(self, context: CompletionContext):
        """Same as :any:`fwd_unicode_match`, but adopted to new Matcher API."""
        # TODO: use `context.limit` to terminate early once we matched the maximum
        #  number that will be used downstream; can be added as an optional to
        #  `fwd_unicode_match(text: str, limit: int = None)` or we could re-implement here.
        fragment, matches = self.fwd_unicode_match(context.text_until_cursor)
        return _convert_matcher_v1_result_to_v2(
            matches, type="unicode", fragment=fragment, suppress_if_matches=True
        )

    def fwd_unicode_match(self, text: str) -> Tuple[str, Sequence[str]]:
        """
        Forward match a string starting with a backslash with a list of
        potential Unicode completions.

        Will compute list of Unicode character names on first call and cache it.

        .. deprecated:: 8.6
            You can use :meth:`fwd_unicode_matcher` instead.

        Returns
        -------
        At tuple with:
            - matched text (empty if no matches)
            - list of potential completions, empty tuple  otherwise)
        """
        # TODO: self.unicode_names is here a list we traverse each time with ~100k elements.
        # We could do a faster match using a Trie.

        # Using pygtrie the following seem to work:

        #     s = PrefixSet()

        #     for c in range(0,0x10FFFF + 1):
        #         try:
        #             s.add(unicodedata.name(chr(c)))
        #         except ValueError:
        #             pass
        #     [''.join(k) for k in s.iter(prefix)]

        # But need to be timed and adds an extra dependency.

        slashpos = text.rfind('\\')
        # if text starts with slash
        if slashpos > -1:
            # PERF: It's important that we don't access self._unicode_names
            # until we're inside this if-block. _unicode_names is lazily
            # initialized, and it takes a user-noticeable amount of time to
            # initialize it, so we don't want to initialize it unless we're
            # actually going to use it.
            s = text[slashpos + 1 :]
            sup = s.upper()
            candidates = [x for x in self.unicode_names if x.startswith(sup)]
            if candidates:
                return s, candidates
            candidates = [x for x in self.unicode_names if sup in x]
            if candidates:
                return s, candidates
            splitsup = sup.split(" ")
            candidates = [
                x for x in self.unicode_names if all(u in x for u in splitsup)
            ]
            if candidates:
                return s, candidates

            return "", ()

        # if text does not start with slash
        else:
            return '', ()

    @property
    def unicode_names(self) -> List[str]:
        """List of names of unicode code points that can be completed.

        The list is lazily initialized on first access.
        """
        if self._unicode_names is None:
            names = []
            for c in range(0,0x10FFFF + 1):
                try:
                    names.append(unicodedata.name(chr(c)))
                except ValueError:
                    pass
            self._unicode_names = _unicode_name_compute(_UNICODE_RANGES)

        return self._unicode_names

def _unicode_name_compute(ranges:List[Tuple[int,int]]) -> List[str]:
    names = []
    for start,stop in ranges:
        for c in range(start, stop) :
            try:
                names.append(unicodedata.name(chr(c)))
            except ValueError:
                pass
    return names
