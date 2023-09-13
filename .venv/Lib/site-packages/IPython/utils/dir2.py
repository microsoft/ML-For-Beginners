# encoding: utf-8
"""A fancy version of Python's builtin :func:`dir` function.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import inspect
import types


def safe_hasattr(obj, attr):
    """In recent versions of Python, hasattr() only catches AttributeError.
    This catches all errors.
    """
    try:
        getattr(obj, attr)
        return True
    except:
        return False


def dir2(obj):
    """dir2(obj) -> list of strings

    Extended version of the Python builtin dir(), which does a few extra
    checks.

    This version is guaranteed to return only a list of true strings, whereas
    dir() returns anything that objects inject into themselves, even if they
    are later not really valid for attribute access (many extension libraries
    have such bugs).
    """

    # Start building the attribute list via dir(), and then complete it
    # with a few extra special-purpose calls.

    try:
        words = set(dir(obj))
    except Exception:
        # TypeError: dir(obj) does not return a list
        words = set()

    if safe_hasattr(obj, '__class__'):
        words |= set(dir(obj.__class__))

    # filter out non-string attributes which may be stuffed by dir() calls
    # and poor coding in third-party modules

    words = [w for w in words if isinstance(w, str)]
    return sorted(words)


def get_real_method(obj, name):
    """Like getattr, but with a few extra sanity checks:

    - If obj is a class, ignore everything except class methods
    - Check if obj is a proxy that claims to have all attributes
    - Catch attribute access failing with any exception
    - Check that the attribute is a callable object

    Returns the method or None.
    """
    try:
        canary = getattr(obj, '_ipython_canary_method_should_not_exist_', None)
    except Exception:
        return None

    if canary is not None:
        # It claimed to have an attribute it should never have
        return None

    try:
        m = getattr(obj, name, None)
    except Exception:
        return None

    if inspect.isclass(obj) and not isinstance(m, types.MethodType):
        return None

    if callable(m):
        return m

    return None
