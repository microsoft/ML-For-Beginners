# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import inspect
import os
import sys


def evaluate(code, path=__file__, mode="eval"):
    # Setting file path here to avoid breaking here if users have set
    # "break on exception raised" setting. This code can potentially run
    # in user process and is indistinguishable if the path is not set.
    # We use the path internally to skip exception inside the debugger.
    expr = compile(code, path, "eval")
    return eval(expr, {}, sys.modules)


class Observable(object):
    """An object with change notifications."""

    observers = ()  # used when attributes are set before __init__ is invoked

    def __init__(self):
        self.observers = []

    def __setattr__(self, name, value):
        try:
            return super().__setattr__(name, value)
        finally:
            for ob in self.observers:
                ob(self, name)


class Env(dict):
    """A dict for environment variables."""

    @staticmethod
    def snapshot():
        """Returns a snapshot of the current environment."""
        return Env(os.environ)

    def copy(self, updated_from=None):
        result = Env(self)
        if updated_from is not None:
            result.update(updated_from)
        return result

    def prepend_to(self, key, entry):
        """Prepends a new entry to a PATH-style environment variable, creating
        it if it doesn't exist already.
        """
        try:
            tail = os.path.pathsep + self[key]
        except KeyError:
            tail = ""
        self[key] = entry + tail


def force_str(s, encoding, errors="strict"):
    """Converts s to str, using the provided encoding. If s is already str,
    it is returned as is.
    """
    return s.decode(encoding, errors) if isinstance(s, bytes) else str(s)


def force_bytes(s, encoding, errors="strict"):
    """Converts s to bytes, using the provided encoding. If s is already bytes,
    it is returned as is.

    If errors="strict" and s is bytes, its encoding is verified by decoding it;
    UnicodeError is raised if it cannot be decoded.
    """
    if isinstance(s, str):
        return s.encode(encoding, errors)
    else:
        s = bytes(s)
        if errors == "strict":
            # Return value ignored - invoked solely for verification.
            s.decode(encoding, errors)
        return s


def force_ascii(s, errors="strict"):
    """Same as force_bytes(s, "ascii", errors)"""
    return force_bytes(s, "ascii", errors)


def force_utf8(s, errors="strict"):
    """Same as force_bytes(s, "utf8", errors)"""
    return force_bytes(s, "utf8", errors)


def nameof(obj, quote=False):
    """Returns the most descriptive name of a Python module, class, or function,
    as a Unicode string

    If quote=True, name is quoted with repr().

    Best-effort, but guaranteed to not fail - always returns something.
    """

    try:
        name = obj.__qualname__
    except Exception:
        try:
            name = obj.__name__
        except Exception:
            # Fall back to raw repr(), and skip quoting.
            try:
                name = repr(obj)
            except Exception:
                return "<unknown>"
            else:
                quote = False

    if quote:
        try:
            name = repr(name)
        except Exception:
            pass

    return force_str(name, "utf-8", "replace")


def srcnameof(obj):
    """Returns the most descriptive name of a Python module, class, or function,
    including source information (filename and linenumber), if available.

    Best-effort, but guaranteed to not fail - always returns something.
    """

    name = nameof(obj, quote=True)

    # Get the source information if possible.
    try:
        src_file = inspect.getsourcefile(obj)
    except Exception:
        pass
    else:
        name += f" (file {src_file!r}"
        try:
            _, src_lineno = inspect.getsourcelines(obj)
        except Exception:
            pass
        else:
            name += f", line {src_lineno}"
        name += ")"

    return name


def hide_debugpy_internals():
    """Returns True if the caller should hide something from debugpy."""
    return "DEBUGPY_TRACE_DEBUGPY" not in os.environ


def hide_thread_from_debugger(thread):
    """Disables tracing for the given thread if DEBUGPY_TRACE_DEBUGPY is not set.
    DEBUGPY_TRACE_DEBUGPY is used to debug debugpy with debugpy
    """
    if hide_debugpy_internals():
        thread.pydev_do_not_trace = True
        thread.is_pydev_daemon_thread = True
