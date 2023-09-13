# encoding: utf-8
"""Utilities for working with data structures like lists, dicts and tuples.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------


def uniq_stable(elems):
    """uniq_stable(elems) -> list

    Return from an iterable, a list of all the unique elements in the input,
    but maintaining the order in which they first appear.

    Note: All elements in the input must be hashable for this routine
    to work, as it internally uses a set for efficiency reasons.
    """
    seen = set()
    return [x for x in elems if x not in seen and not seen.add(x)]


def chop(seq, size):
    """Chop a sequence into chunks of the given size."""
    return [seq[i:i+size] for i in range(0,len(seq),size)]


