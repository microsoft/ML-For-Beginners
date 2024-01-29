# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback

import debugpy
from debugpy.common import json, timestamp, util


LEVELS = ("debug", "info", "warning", "error")
"""Logging levels, lowest to highest importance.
"""

log_dir = os.getenv("DEBUGPY_LOG_DIR")
"""If not None, debugger logs its activity to a file named debugpy.*-<pid>.log
in the specified directory, where <pid> is the return value of os.getpid().
"""

timestamp_format = "09.3f"
"""Format spec used for timestamps. Can be changed to dial precision up or down.
"""

_lock = threading.RLock()
_tls = threading.local()
_files = {}  # filename -> LogFile
_levels = set()  # combined for all log files


def _update_levels():
    global _levels
    _levels = frozenset(level for file in _files.values() for level in file.levels)


class LogFile(object):
    def __init__(self, filename, file, levels=LEVELS, close_file=True):
        info("Also logging to {0}.", json.repr(filename))
        self.filename = filename
        self.file = file
        self.close_file = close_file
        self._levels = frozenset(levels)

        with _lock:
            _files[self.filename] = self
            _update_levels()
            info(
                "{0} {1}\n{2} {3} ({4}-bit)\ndebugpy {5}",
                platform.platform(),
                platform.machine(),
                platform.python_implementation(),
                platform.python_version(),
                64 if sys.maxsize > 2**32 else 32,
                debugpy.__version__,
                _to_files=[self],
            )

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value):
        with _lock:
            self._levels = frozenset(LEVELS if value is all else value)
            _update_levels()

    def write(self, level, output):
        if level in self.levels:
            try:
                self.file.write(output)
                self.file.flush()
            except Exception:  # pragma: no cover
                pass

    def close(self):
        with _lock:
            del _files[self.filename]
            _update_levels()
        info("Not logging to {0} anymore.", json.repr(self.filename))

        if self.close_file:
            try:
                self.file.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NoLog(object):
    file = filename = None

    __bool__ = __nonzero__ = lambda self: False

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Used to inject a newline into stderr if logging there, to clean up the output
# when it's intermixed with regular prints from other sources.
def newline(level="info"):
    with _lock:
        stderr.write(level, "\n")


def write(level, text, _to_files=all):
    assert level in LEVELS

    t = timestamp.current()
    format_string = "{0}+{1:" + timestamp_format + "}: "
    prefix = format_string.format(level[0].upper(), t)

    text = getattr(_tls, "prefix", "") + text
    indent = "\n" + (" " * len(prefix))
    output = indent.join(text.split("\n"))
    output = prefix + output + "\n\n"

    with _lock:
        if _to_files is all:
            _to_files = _files.values()
        for file in _to_files:
            file.write(level, output)

    return text


def write_format(level, format_string, *args, **kwargs):
    # Don't spend cycles doing expensive formatting if we don't have to. Errors are
    # always formatted, so that error() can return the text even if it's not logged.
    if level != "error" and level not in _levels:
        return

    try:
        text = format_string.format(*args, **kwargs)
    except Exception:  # pragma: no cover
        reraise_exception()

    return write(level, text, kwargs.pop("_to_files", all))


debug = functools.partial(write_format, "debug")
info = functools.partial(write_format, "info")
warning = functools.partial(write_format, "warning")


def error(*args, **kwargs):
    """Logs an error.

    Returns the output wrapped in AssertionError. Thus, the following::

        raise log.error(s, ...)

    has the same effect as::

        log.error(...)
        assert False, (s.format(...))
    """
    return AssertionError(write_format("error", *args, **kwargs))


def _exception(format_string="", *args, **kwargs):
    level = kwargs.pop("level", "error")
    exc_info = kwargs.pop("exc_info", sys.exc_info())

    if format_string:
        format_string += "\n\n"
    format_string += "{exception}\nStack where logged:\n{stack}"

    exception = "".join(traceback.format_exception(*exc_info))

    f = inspect.currentframe()
    f = f.f_back if f else f  # don't log this frame
    try:
        stack = "".join(traceback.format_stack(f))
    finally:
        del f  # avoid cycles

    write_format(
        level, format_string, *args, exception=exception, stack=stack, **kwargs
    )


def swallow_exception(format_string="", *args, **kwargs):
    """Logs an exception with full traceback.

    If format_string is specified, it is formatted with format(*args, **kwargs), and
    prepended to the exception traceback on a separate line.

    If exc_info is specified, the exception it describes will be logged. Otherwise,
    sys.exc_info() - i.e. the exception being handled currently - will be logged.

    If level is specified, the exception will be logged as a message of that level.
    The default is "error".
    """

    _exception(format_string, *args, **kwargs)


def reraise_exception(format_string="", *args, **kwargs):
    """Like swallow_exception(), but re-raises the current exception after logging it."""

    assert "exc_info" not in kwargs
    _exception(format_string, *args, **kwargs)
    raise


def to_file(filename=None, prefix=None, levels=LEVELS):
    """Starts logging all messages at the specified levels to the designated file.

    Either filename or prefix must be specified, but not both.

    If filename is specified, it designates the log file directly.

    If prefix is specified, the log file is automatically created in options.log_dir,
    with filename computed as prefix + os.getpid(). If log_dir is None, no log file
    is created, and the function returns immediately.

    If the file with the specified or computed name is already being used as a log
    file, it is not overwritten, but its levels are updated as specified.

    The function returns an object with a close() method. When the object is closed,
    logs are not written into that file anymore. Alternatively, the returned object
    can be used in a with-statement:

        with log.to_file("some.log"):
            # now also logging to some.log
        # not logging to some.log anymore
    """

    assert (filename is not None) ^ (prefix is not None)

    if filename is None:
        if log_dir is None:
            return NoLog()
        try:
            os.makedirs(log_dir)
        except OSError:  # pragma: no cover
            pass
        filename = f"{log_dir}/{prefix}-{os.getpid()}.log"

    file = _files.get(filename)
    if file is None:
        file = LogFile(filename, io.open(filename, "w", encoding="utf-8"), levels)
    else:
        file.levels = levels
    return file


@contextlib.contextmanager
def prefixed(format_string, *args, **kwargs):
    """Adds a prefix to all messages logged from the current thread for the duration
    of the context manager.
    """
    prefix = format_string.format(*args, **kwargs)
    old_prefix = getattr(_tls, "prefix", "")
    _tls.prefix = prefix + old_prefix
    try:
        yield
    finally:
        _tls.prefix = old_prefix


def get_environment_description(header):
    import sysconfig
    import site  # noqa

    result = [header, "\n\n"]

    def report(s, *args, **kwargs):
        result.append(s.format(*args, **kwargs))

    def report_paths(get_paths, label=None):
        prefix = f"    {label or get_paths}: "

        expr = None
        if not callable(get_paths):
            expr = get_paths
            get_paths = lambda: util.evaluate(expr)
        try:
            paths = get_paths()
        except AttributeError:
            report("{0}<missing>\n", prefix)
            return
        except Exception:  # pragma: no cover
            swallow_exception(
                "Error evaluating {0}",
                repr(expr) if expr else util.srcnameof(get_paths),
                level="info",
            )
            return

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        for p in sorted(paths):
            report("{0}{1}", prefix, p)
            if p is not None:
                rp = os.path.realpath(p)
                if p != rp:
                    report("({0})", rp)
            report("\n")

            prefix = " " * len(prefix)

    report("System paths:\n")
    report_paths("sys.executable")
    report_paths("sys.prefix")
    report_paths("sys.base_prefix")
    report_paths("sys.real_prefix")
    report_paths("site.getsitepackages()")
    report_paths("site.getusersitepackages()")

    site_packages = [
        p
        for p in sys.path
        if os.path.exists(p) and os.path.basename(p) == "site-packages"
    ]
    report_paths(lambda: site_packages, "sys.path (site-packages)")

    for name in sysconfig.get_path_names():
        expr = "sysconfig.get_path({0!r})".format(name)
        report_paths(expr)

    report_paths("os.__file__")
    report_paths("threading.__file__")
    report_paths("debugpy.__file__")
    report("\n")

    importlib_metadata = None
    try:
        import importlib_metadata
    except ImportError:  # pragma: no cover
        try:
            from importlib import metadata as importlib_metadata
        except ImportError:
            pass
    if importlib_metadata is None:  # pragma: no cover
        report("Cannot enumerate installed packages - missing importlib_metadata.")
    else:
        report("Installed packages:\n")
        try:
            for pkg in importlib_metadata.distributions():
                report("    {0}=={1}\n", pkg.name, pkg.version)
        except Exception:  # pragma: no cover
            swallow_exception(
                "Error while enumerating installed packages.", level="info"
            )

    return "".join(result).rstrip("\n")


def describe_environment(header):
    info("{0}", get_environment_description(header))


stderr = LogFile(
    "<stderr>",
    sys.stderr,
    levels=os.getenv("DEBUGPY_LOG_STDERR", "warning error").split(),
    close_file=False,
)


@atexit.register
def _close_files():
    for file in tuple(_files.values()):
        file.close()


# The following are helper shortcuts for printf debugging. They must never be used
# in production code.


def _repr(value):  # pragma: no cover
    warning("$REPR {0!r}", value)


def _vars(*names):  # pragma: no cover
    locals = inspect.currentframe().f_back.f_locals
    if names:
        locals = {name: locals[name] for name in names if name in locals}
    warning("$VARS {0!r}", locals)


def _stack():  # pragma: no cover
    stack = "\n".join(traceback.format_stack())
    warning("$STACK:\n\n{0}", stack)


def _threads():  # pragma: no cover
    output = "\n".join([str(t) for t in threading.enumerate()])
    warning("$THREADS:\n\n{0}", output)
