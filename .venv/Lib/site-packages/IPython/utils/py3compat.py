# coding: utf-8
"""Compatibility tricks for Python 3. Mainly to do with unicode.

This file is deprecated and will be removed in a future version.
"""
import platform
import builtins as builtin_mod

from .encoding import DEFAULT_ENCODING


def decode(s, encoding=None):
    encoding = encoding or DEFAULT_ENCODING
    return s.decode(encoding, "replace")


def encode(u, encoding=None):
    encoding = encoding or DEFAULT_ENCODING
    return u.encode(encoding, "replace")


def cast_unicode(s, encoding=None):
    if isinstance(s, bytes):
        return decode(s, encoding)
    return s


def safe_unicode(e):
    """unicode(e) with various fallbacks. Used for exceptions, which may not be
    safe to call unicode() on.
    """
    try:
        return str(e)
    except UnicodeError:
        pass

    try:
        return repr(e)
    except UnicodeError:
        pass

    return "Unrecoverably corrupt evalue"


# keep reference to builtin_mod because the kernel overrides that value
# to forward requests to a frontend.
def input(prompt=""):
    return builtin_mod.input(prompt)


def execfile(fname, glob, loc=None, compiler=None):
    loc = loc if (loc is not None) else glob
    with open(fname, "rb") as f:
        compiler = compiler or compile
        exec(compiler(f.read(), fname, "exec"), glob, loc)


PYPY = platform.python_implementation() == "PyPy"

# Cython still rely on that as a Dec 28 2019
# See https://github.com/cython/cython/pull/3291 and
# https://github.com/ipython/ipython/issues/12068
def no_code(x, encoding=None):
    return x


unicode_to_str = cast_bytes_py2 = no_code
