"""
This is a python implementation of wcwidth() and wcswidth().

https://github.com/jquast/wcwidth

from Markus Kuhn's C code, retrieved from:

    http://www.cl.cam.ac.uk/~mgk25/ucs/wcwidth.c

This is an implementation of wcwidth() and wcswidth() (defined in
IEEE Std 1002.1-2001) for Unicode.

http://www.opengroup.org/onlinepubs/007904975/functions/wcwidth.html
http://www.opengroup.org/onlinepubs/007904975/functions/wcswidth.html

In fixed-width output devices, Latin characters all occupy a single
"cell" position of equal width, whereas ideographic CJK characters
occupy two such cells. Interoperability between terminal-line
applications and (teletype-style) character terminals using the
UTF-8 encoding requires agreement on which character should advance
the cursor by how many cell positions. No established formal
standards exist at present on which Unicode character shall occupy
how many cell positions on character terminals. These routines are
a first attempt of defining such behavior based on simple rules
applied to data provided by the Unicode Consortium.

For some graphical characters, the Unicode standard explicitly
defines a character-cell width via the definition of the East Asian
FullWidth (F), Wide (W), Half-width (H), and Narrow (Na) classes.
In all these cases, there is no ambiguity about which width a
terminal shall use. For characters in the East Asian Ambiguous (A)
class, the width choice depends purely on a preference of backward
compatibility with either historic CJK or Western practice.
Choosing single-width for these characters is easy to justify as
the appropriate long-term solution, as the CJK practice of
displaying these characters as double-width comes from historic
implementation simplicity (8-bit encoded characters were displayed
single-width and 16-bit ones double-width, even for Greek,
Cyrillic, etc.) and not any typographic considerations.

Much less clear is the choice of width for the Not East Asian
(Neutral) class. Existing practice does not dictate a width for any
of these characters. It would nevertheless make sense
typographically to allocate two character cells to characters such
as for instance EM SPACE or VOLUME INTEGRAL, which cannot be
represented adequately with a single-width glyph. The following
routines at present merely assign a single-cell width to all
neutral characters, in the interest of simplicity. This is not
entirely satisfactory and should be reconsidered before
establishing a formal standard in this area. At the moment, the
decision which Not East Asian (Neutral) characters should be
represented by double-width glyphs cannot yet be answered by
applying a simple rule from the Unicode database content. Setting
up a proper standard for the behavior of UTF-8 character terminals
will require a careful analysis not only of each Unicode character,
but also of each presentation form, something the author of these
routines has avoided to do so far.

http://www.unicode.org/unicode/reports/tr11/

Latest version: http://www.cl.cam.ac.uk/~mgk25/ucs/wcwidth.c
"""
from __future__ import division

# std imports
import os
import sys
import warnings

# local
from .table_wide import WIDE_EASTASIAN
from .table_zero import ZERO_WIDTH
from .unicode_versions import list_versions

try:
    # std imports
    from functools import lru_cache
except ImportError:
    # lru_cache was added in Python 3.2
    # 3rd party
    from backports.functools_lru_cache import lru_cache

# global cache
_UNICODE_CMPTABLE = None
_PY3 = (sys.version_info[0] >= 3)


# NOTE: created by hand, there isn't anything identifiable other than
# general Cf category code to identify these, and some characters in Cf
# category code are of non-zero width.
# Also includes some Cc, Mn, Zl, and Zp characters
ZERO_WIDTH_CF = set([
    0,       # Null (Cc)
    0x034F,  # Combining grapheme joiner (Mn)
    0x200B,  # Zero width space
    0x200C,  # Zero width non-joiner
    0x200D,  # Zero width joiner
    0x200E,  # Left-to-right mark
    0x200F,  # Right-to-left mark
    0x2028,  # Line separator (Zl)
    0x2029,  # Paragraph separator (Zp)
    0x202A,  # Left-to-right embedding
    0x202B,  # Right-to-left embedding
    0x202C,  # Pop directional formatting
    0x202D,  # Left-to-right override
    0x202E,  # Right-to-left override
    0x2060,  # Word joiner
    0x2061,  # Function application
    0x2062,  # Invisible times
    0x2063,  # Invisible separator
])


def _bisearch(ucs, table):
    """
    Auxiliary function for binary search in interval table.

    :arg int ucs: Ordinal value of unicode character.
    :arg list table: List of starting and ending ranges of ordinal values,
        in form of ``[(start, end), ...]``.
    :rtype: int
    :returns: 1 if ordinal value ucs is found within lookup table, else 0.
    """
    lbound = 0
    ubound = len(table) - 1

    if ucs < table[0][0] or ucs > table[ubound][1]:
        return 0
    while ubound >= lbound:
        mid = (lbound + ubound) // 2
        if ucs > table[mid][1]:
            lbound = mid + 1
        elif ucs < table[mid][0]:
            ubound = mid - 1
        else:
            return 1

    return 0


@lru_cache(maxsize=1000)
def wcwidth(wc, unicode_version='auto'):
    r"""
    Given one Unicode character, return its printable length on a terminal.

    :param str wc: A single Unicode character.
    :param str unicode_version: A Unicode version number, such as
        ``'6.0.0'``, the list of available version levels may be
        listed by pairing function :func:`list_versions`.

        Any version string may be specified without error -- the nearest
        matching version is selected.  When ``latest`` (default), the
        highest Unicode version level is used.
    :return: The width, in cells, necessary to display the character of
        Unicode string character, ``wc``.  Returns 0 if the ``wc`` argument has
        no printable effect on a terminal (such as NUL '\0'), -1 if ``wc`` is
        not printable, or has an indeterminate effect on the terminal, such as
        a control character.  Otherwise, the number of column positions the
        character occupies on a graphic terminal (1 or 2) is returned.
    :rtype: int

    The following have a column width of -1:

        - C0 control characters (U+001 through U+01F).

        - C1 control characters and DEL (U+07F through U+0A0).

    The following have a column width of 0:

    - Non-spacing and enclosing combining characters (general
      category code Mn or Me in the Unicode database).

    - NULL (``U+0000``).

    - COMBINING GRAPHEME JOINER (``U+034F``).

    - ZERO WIDTH SPACE (``U+200B``) *through*
      RIGHT-TO-LEFT MARK (``U+200F``).

    - LINE SEPARATOR (``U+2028``) *and*
      PARAGRAPH SEPARATOR (``U+2029``).

    - LEFT-TO-RIGHT EMBEDDING (``U+202A``) *through*
      RIGHT-TO-LEFT OVERRIDE (``U+202E``).

    - WORD JOINER (``U+2060``) *through*
      INVISIBLE SEPARATOR (``U+2063``).

    The following have a column width of 1:

    - SOFT HYPHEN (``U+00AD``).

    - All remaining characters, including all printable ISO 8859-1
      and WGL4 characters, Unicode control characters, etc.

    The following have a column width of 2:

        - Spacing characters in the East Asian Wide (W) or East Asian
          Full-width (F) category as defined in Unicode Technical
          Report #11 have a column width of 2.

         - Some kinds of Emoji or symbols.
    """
    # NOTE: created by hand, there isn't anything identifiable other than
    # general Cf category code to identify these, and some characters in Cf
    # category code are of non-zero width.
    ucs = ord(wc)
    if ucs in ZERO_WIDTH_CF:
        return 0

    # C0/C1 control characters
    if ucs < 32 or 0x07F <= ucs < 0x0A0:
        return -1

    _unicode_version = _wcmatch_version(unicode_version)

    # combining characters with zero width
    if _bisearch(ucs, ZERO_WIDTH[_unicode_version]):
        return 0

    # "Wide AastAsian" (and emojis)
    return 1 + _bisearch(ucs, WIDE_EASTASIAN[_unicode_version])


def wcswidth(pwcs, n=None, unicode_version='auto'):
    """
    Given a unicode string, return its printable length on a terminal.

    :param str pwcs: Measure width of given unicode string.
    :param int n: When ``n`` is None (default), return the length of the
        entire string, otherwise width the first ``n`` characters specified.
    :param str unicode_version: An explicit definition of the unicode version
        level to use for determination, may be ``auto`` (default), which uses
        the Environment Variable, ``UNICODE_VERSION`` if defined, or the latest
        available unicode version, otherwise.
    :rtype: int
    :returns: The width, in cells, necessary to display the first ``n``
        characters of the unicode string ``pwcs``.  Returns ``-1`` if
        a non-printable character is encountered.
    """
    # pylint: disable=C0103
    #         Invalid argument name "n"

    end = len(pwcs) if n is None else n
    idx = slice(0, end)
    width = 0
    for char in pwcs[idx]:
        wcw = wcwidth(char, unicode_version)
        if wcw < 0:
            return -1
        width += wcw
    return width


@lru_cache(maxsize=128)
def _wcversion_value(ver_string):
    """
    Integer-mapped value of given dotted version string.

    :param str ver_string: Unicode version string, of form ``n.n.n``.
    :rtype: tuple(int)
    :returns: tuple of digit tuples, ``tuple(int, [...])``.
    """
    retval = tuple(map(int, (ver_string.split('.'))))
    return retval


@lru_cache(maxsize=8)
def _wcmatch_version(given_version):
    """
    Return nearest matching supported Unicode version level.

    If an exact match is not determined, the nearest lowest version level is
    returned after a warning is emitted.  For example, given supported levels
    ``4.1.0`` and ``5.0.0``, and a version string of ``4.9.9``, then ``4.1.0``
    is selected and returned:

    >>> _wcmatch_version('4.9.9')
    '4.1.0'
    >>> _wcmatch_version('8.0')
    '8.0.0'
    >>> _wcmatch_version('1')
    '4.1.0'

    :param str given_version: given version for compare, may be ``auto``
        (default), to select Unicode Version from Environment Variable,
        ``UNICODE_VERSION``. If the environment variable is not set, then the
        latest is used.
    :rtype: str
    :returns: unicode string, or non-unicode ``str`` type for python 2
        when given ``version`` is also type ``str``.
    """
    # Design note: the choice to return the same type that is given certainly
    # complicates it for python 2 str-type, but allows us to define an api that
    # to use 'string-type', for unicode version level definitions, so all of our
    # example code works with all versions of python. That, along with the
    # string-to-numeric and comparisons of earliest, latest, matching, or
    # nearest, greatly complicates this function.
    _return_str = not _PY3 and isinstance(given_version, str)

    if _return_str:
        unicode_versions = [ucs.encode() for ucs in list_versions()]
    else:
        unicode_versions = list_versions()
    latest_version = unicode_versions[-1]

    if given_version in (u'auto', 'auto'):
        given_version = os.environ.get(
            'UNICODE_VERSION',
            'latest' if not _return_str else latest_version.encode())

    if given_version in (u'latest', 'latest'):
        # default match, when given as 'latest', use the most latest unicode
        # version specification level supported.
        return latest_version if not _return_str else latest_version.encode()

    if given_version in unicode_versions:
        # exact match, downstream has specified an explicit matching version
        # matching any value of list_versions().
        return given_version if not _return_str else given_version.encode()

    # The user's version is not supported by ours. We return the newest unicode
    # version level that we support below their given value.
    try:
        cmp_given = _wcversion_value(given_version)

    except ValueError:
        # submitted value raises ValueError in int(), warn and use latest.
        warnings.warn("UNICODE_VERSION value, {given_version!r}, is invalid. "
                      "Value should be in form of `integer[.]+', the latest "
                      "supported unicode version {latest_version!r} has been "
                      "inferred.".format(given_version=given_version,
                                         latest_version=latest_version))
        return latest_version if not _return_str else latest_version.encode()

    # given version is less than any available version, return earliest
    # version.
    earliest_version = unicode_versions[0]
    cmp_earliest_version = _wcversion_value(earliest_version)

    if cmp_given <= cmp_earliest_version:
        # this probably isn't what you wanted, the oldest wcwidth.c you will
        # find in the wild is likely version 5 or 6, which we both support,
        # but it's better than not saying anything at all.
        warnings.warn("UNICODE_VERSION value, {given_version!r}, is lower "
                      "than any available unicode version. Returning lowest "
                      "version level, {earliest_version!r}".format(
                          given_version=given_version,
                          earliest_version=earliest_version))
        return earliest_version if not _return_str else earliest_version.encode()

    # create list of versions which are less than our equal to given version,
    # and return the tail value, which is the highest level we may support,
    # or the latest value we support, when completely unmatched or higher
    # than any supported version.
    #
    # function will never complete, always returns.
    for idx, unicode_version in enumerate(unicode_versions):
        # look ahead to next value
        try:
            cmp_next_version = _wcversion_value(unicode_versions[idx + 1])
        except IndexError:
            # at end of list, return latest version
            return latest_version if not _return_str else latest_version.encode()

        # Maybe our given version has less parts, as in tuple(8, 0), than the
        # next compare version tuple(8, 0, 0). Test for an exact match by
        # comparison of only the leading dotted piece(s): (8, 0) == (8, 0).
        if cmp_given == cmp_next_version[:len(cmp_given)]:
            return unicode_versions[idx + 1]

        # Or, if any next value is greater than our given support level
        # version, return the current value in index.  Even though it must
        # be less than the given value, its our closest possible match. That
        # is, 4.1 is returned for given 4.9.9, where 4.1 and 5.0 are available.
        if cmp_next_version > cmp_given:
            return unicode_version
    assert False, ("Code path unreachable", given_version, unicode_versions)
