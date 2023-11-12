# -*- coding: utf-8 -*-
import re
import string

PUNCTUATION_REGEX = re.compile('[{0}]'.format(re.escape(string.punctuation)))


def strip_punc(s, all=False):
    """Removes punctuation from a string.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    if all:
        return PUNCTUATION_REGEX.sub('', s.strip())
    else:
        return s.strip().strip(string.punctuation)


def lowerstrip(s, all=False):
    """Makes text all lowercase and strips punctuation and whitespace.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    return strip_punc(s.lower().strip(), all=all)


def tree2str(tree, concat=' '):
    """Convert a nltk.tree.Tree to a string.

    For example:
        (NP a/DT beautiful/JJ new/JJ dashboard/NN) -> "a beautiful dashboard"
    """
    return concat.join([word for (word, tag) in tree])


def filter_insignificant(chunk, tag_suffixes=('DT', 'CC', 'PRP$', 'PRP')):
    """Filter out insignificant (word, tag) tuples from a chunk of text."""
    good = []
    for word, tag in chunk:
        ok = True
        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break
        if ok:
            good.append((word, tag))
    return good


def is_filelike(obj):
    """Return whether ``obj`` is a file-like object."""
    return hasattr(obj, 'read')
