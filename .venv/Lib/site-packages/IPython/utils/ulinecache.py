"""
This module has been deprecated since IPython 6.0.

Wrapper around linecache which decodes files to unicode according to PEP 263.
"""
import functools
import linecache
from warnings import warn

getline = linecache.getline

# getlines has to be looked up at runtime, because doctests monkeypatch it.
@functools.wraps(linecache.getlines)
def getlines(filename, module_globals=None):
    """
    Deprecated since IPython 6.0
    """
    warn(("`IPython.utils.ulinecache.getlines` is deprecated since"
          " IPython 6.0 and will be removed in future versions."),
         DeprecationWarning, stacklevel=2)
    return linecache.getlines(filename, module_globals=module_globals)
