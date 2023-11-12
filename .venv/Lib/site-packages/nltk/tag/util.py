# Natural Language Toolkit: Tagger Utilities
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


def str2tuple(s, sep="/"):
    """
    Given the string representation of a tagged token, return the
    corresponding tuple representation.  The rightmost occurrence of
    *sep* in *s* will be used to divide *s* into a word string and
    a tag string.  If *sep* does not occur in *s*, return (s, None).

        >>> from nltk.tag.util import str2tuple
        >>> str2tuple('fly/NN')
        ('fly', 'NN')

    :type s: str
    :param s: The string representation of a tagged token.
    :type sep: str
    :param sep: The separator string used to separate word strings
        from tags.
    """
    loc = s.rfind(sep)
    if loc >= 0:
        return (s[:loc], s[loc + len(sep) :].upper())
    else:
        return (s, None)


def tuple2str(tagged_token, sep="/"):
    """
    Given the tuple representation of a tagged token, return the
    corresponding string representation.  This representation is
    formed by concatenating the token's word string, followed by the
    separator, followed by the token's tag.  (If the tag is None,
    then just return the bare word string.)

        >>> from nltk.tag.util import tuple2str
        >>> tagged_token = ('fly', 'NN')
        >>> tuple2str(tagged_token)
        'fly/NN'

    :type tagged_token: tuple(str, str)
    :param tagged_token: The tuple representation of a tagged token.
    :type sep: str
    :param sep: The separator string used to separate word strings
        from tags.
    """
    word, tag = tagged_token
    if tag is None:
        return word
    else:
        assert sep not in tag, "tag may not contain sep!"
        return f"{word}{sep}{tag}"


def untag(tagged_sentence):
    """
    Given a tagged sentence, return an untagged version of that
    sentence.  I.e., return a list containing the first element
    of each tuple in *tagged_sentence*.

        >>> from nltk.tag.util import untag
        >>> untag([('John', 'NNP'), ('saw', 'VBD'), ('Mary', 'NNP')])
        ['John', 'saw', 'Mary']

    """
    return [w for (w, t) in tagged_sentence]
