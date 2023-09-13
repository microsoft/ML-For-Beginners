"""Compiler tools with improved interactive support.

Provides compilation machinery similar to codeop, but with caching support so
we can provide interactive tracebacks.

Authors
-------
* Robert Kern
* Fernando Perez
* Thomas Kluyver
"""

# Note: though it might be more natural to name this module 'compiler', that
# name is in the stdlib and name collisions with the stdlib tend to produce
# weird problems (often with third-party tools).

#-----------------------------------------------------------------------------
#  Copyright (C) 2010-2011 The IPython Development Team.
#
#  Distributed under the terms of the BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib imports
import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# Roughly equal to PyCF_MASK | PyCF_MASK_OBSOLETE as defined in pythonrun.h,
# this is used as a bitmask to extract future-related code flags.
PyCF_MASK = functools.reduce(operator.or_,
                             (getattr(__future__, fname).compiler_flag
                              for fname in __future__.all_feature_names))

#-----------------------------------------------------------------------------
# Local utilities
#-----------------------------------------------------------------------------

def code_name(code, number=0):
    """ Compute a (probably) unique name for code for caching.

    This now expects code to be unicode.
    """
    hash_digest = hashlib.sha1(code.encode("utf-8")).hexdigest()
    # Include the number and 12 characters of the hash in the name.  It's
    # pretty much impossible that in a single session we'll have collisions
    # even with truncated hashes, and the full one makes tracebacks too long
    return '<ipython-input-{0}-{1}>'.format(number, hash_digest[:12])

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

class CachingCompiler(codeop.Compile):
    """A compiler that caches code compiled from interactive statements.
    """

    def __init__(self):
        codeop.Compile.__init__(self)

        # Caching a dictionary { filename: execution_count } for nicely
        # rendered tracebacks. The filename corresponds to the filename
        # argument used for the builtins.compile function.
        self._filename_map = {}

    def ast_parse(self, source, filename='<unknown>', symbol='exec'):
        """Parse code to an AST with the current compiler flags active.

        Arguments are exactly the same as ast.parse (in the standard library),
        and are passed to the built-in compile function."""
        return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)

    def reset_compiler_flags(self):
        """Reset compiler flags to default state."""
        # This value is copied from codeop.Compile.__init__, so if that ever
        # changes, it will need to be updated.
        self.flags = codeop.PyCF_DONT_IMPLY_DEDENT

    @property
    def compiler_flags(self):
        """Flags currently active in the compilation process.
        """
        return self.flags

    def get_code_name(self, raw_code, transformed_code, number):
        """Compute filename given the code, and the cell number.

        Parameters
        ----------
        raw_code : str
            The raw cell code.
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.

        Returns
        -------
        The computed filename.
        """
        return code_name(transformed_code, number)

    def format_code_name(self, name):
        """Return a user-friendly label and name for a code block.

        Parameters
        ----------
        name : str
            The name for the code block returned from get_code_name

        Returns
        -------
        A (label, name) pair that can be used in tracebacks, or None if the default formatting should be used.
        """
        if name in self._filename_map:
            return "Cell", "In[%s]" % self._filename_map[name]

    def cache(self, transformed_code, number=0, raw_code=None):
        """Make a name for a block of code, and cache the code.

        Parameters
        ----------
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.
        raw_code : str
            The raw code before transformation, if None, set to `transformed_code`.

        Returns
        -------
        The name of the cached code (as a string). Pass this as the filename
        argument to compilation, so that tracebacks are correctly hooked up.
        """
        if raw_code is None:
            raw_code = transformed_code

        name = self.get_code_name(raw_code, transformed_code, number)

        # Save the execution count
        self._filename_map[name] = number

        # Since Python 2.5, setting mtime to `None` means the lines will
        # never be removed by `linecache.checkcache`.  This means all the
        # monkeypatching has *never* been necessary, since this code was
        # only added in 2010, at which point IPython had already stopped
        # supporting Python 2.4.
        #
        # Note that `linecache.clearcache` and `linecache.updatecache` may
        # still remove our code from the cache, but those show explicit
        # intent, and we should not try to interfere.  Normally the former
        # is never called except when out of memory, and the latter is only
        # called for lines *not* in the cache.
        entry = (
            len(transformed_code),
            None,
            [line + "\n" for line in transformed_code.splitlines()],
            name,
        )
        linecache.cache[name] = entry
        return name

    @contextmanager
    def extra_flags(self, flags):
        ## bits that we'll set to 1
        turn_on_bits = ~self.flags & flags


        self.flags = self.flags | flags
        try:
            yield
        finally:
            # turn off only the bits we turned on so that something like
            # __future__ that set flags stays.
            self.flags &= ~turn_on_bits


def check_linecache_ipython(*args):
    """Deprecated since IPython 8.6.  Call linecache.checkcache() directly.

    It was already not necessary to call this function directly.  If no
    CachingCompiler had been created, this function would fail badly.  If
    an instance had been created, this function would've been monkeypatched
    into place.

    As of IPython 8.6, the monkeypatching has gone away entirely.  But there
    were still internal callers of this function, so maybe external callers
    also existed?
    """
    import warnings

    warnings.warn(
        "Deprecated Since IPython 8.6, Just call linecache.checkcache() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    linecache.checkcache()
