# coding: utf-8
"""
Utilities for dealing with text encodings
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2012  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import sys
import locale
import warnings

# to deal with the possibility of sys.std* not being a stream at all
def get_stream_enc(stream, default=None):
    """Return the given stream's encoding or a default.

    There are cases where ``sys.std*`` might not actually be a stream, so
    check for the encoding attribute prior to returning it, and return
    a default if it doesn't exist or evaluates as False. ``default``
    is None if not provided.
    """
    if not hasattr(stream, 'encoding') or not stream.encoding:
        return default
    else:
        return stream.encoding

# Less conservative replacement for sys.getdefaultencoding, that will try
# to match the environment.
# Defined here as central function, so if we find better choices, we
# won't need to make changes all over IPython.
def getdefaultencoding(prefer_stream=True):
    """Return IPython's guess for the default encoding for bytes as text.

    If prefer_stream is True (default), asks for stdin.encoding first,
    to match the calling Terminal, but that is often None for subprocesses.

    Then fall back on locale.getpreferredencoding(),
    which should be a sensible platform default (that respects LANG environment),
    and finally to sys.getdefaultencoding() which is the most conservative option,
    and usually UTF8 as of Python 3.
    """
    enc = None
    if prefer_stream:
        enc = get_stream_enc(sys.stdin)
    if not enc or enc=='ascii':
        try:
            # There are reports of getpreferredencoding raising errors
            # in some cases, which may well be fixed, but let's be conservative here.
            enc = locale.getpreferredencoding()
        except Exception:
            pass
    enc = enc or sys.getdefaultencoding()
    # On windows `cp0` can be returned to indicate that there is no code page.
    # Since cp0 is an invalid encoding return instead cp1252 which is the
    # Western European default.
    if enc == 'cp0':
        warnings.warn(
            "Invalid code page cp0 detected - using cp1252 instead."
            "If cp1252 is incorrect please ensure a valid code page "
            "is defined for the process.", RuntimeWarning)
        return 'cp1252'
    return enc

DEFAULT_ENCODING = getdefaultencoding()
