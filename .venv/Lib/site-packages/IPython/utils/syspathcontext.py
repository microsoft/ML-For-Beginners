# encoding: utf-8
"""
Context managers for adding things to sys.path temporarily.

Authors:

* Brian Granger
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

import sys
import warnings


class appended_to_syspath(object):
    """
    Deprecated since IPython 8.1, no replacements.

    A context for appending a directory to sys.path for a second."""

    def __init__(self, dir):
        warnings.warn(
            "`appended_to_syspath` is deprecated since IPython 8.1, and has no replacements",
            DeprecationWarning,
            stacklevel=2,
        )
        self.dir = dir

    def __enter__(self):
        if self.dir not in sys.path:
            sys.path.append(self.dir)
            self.added = True
        else:
            self.added = False

    def __exit__(self, type, value, traceback):
        if self.added:
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
        # Returning False causes any exceptions to be re-raised.
        return False

class prepended_to_syspath(object):
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, dir):
        self.dir = dir

    def __enter__(self):
        if self.dir not in sys.path:
            sys.path.insert(0,self.dir)
            self.added = True
        else:
            self.added = False

    def __exit__(self, type, value, traceback):
        if self.added:
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
        # Returning False causes any exceptions to be re-raised.
        return False
