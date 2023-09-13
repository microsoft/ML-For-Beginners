from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
    ForkSafeLock
from contextlib import contextmanager
import traceback
import os
import sys


class _LoggingGlobals(object):
    _warn_once_map = {}
    _debug_stream_filename = None
    _debug_stream = NULL
    _debug_stream_initialized = False
    _initialize_lock = ForkSafeLock()


def initialize_debug_stream(reinitialize=False):
    '''
    :param bool reinitialize:
        Reinitialize is used to update the debug stream after a fork (thus, if it wasn't
        initialized, we don't need to do anything, just wait for the first regular log call
        to initialize).
    '''
    if reinitialize:
        if not _LoggingGlobals._debug_stream_initialized:
            return
    else:
        if _LoggingGlobals._debug_stream_initialized:
            return

    with _LoggingGlobals._initialize_lock:
        # Initialization is done lazilly, so, it's possible that multiple threads try to initialize
        # logging.

        # Check initial conditions again after obtaining the lock.
        if reinitialize:
            if not _LoggingGlobals._debug_stream_initialized:
                return
        else:
            if _LoggingGlobals._debug_stream_initialized:
                return

        _LoggingGlobals._debug_stream_initialized = True

        # Note: we cannot initialize with sys.stderr because when forking we may end up logging things in 'os' calls.
        _LoggingGlobals._debug_stream = NULL
        _LoggingGlobals._debug_stream_filename = None

        if not DebugInfoHolder.PYDEVD_DEBUG_FILE:
            _LoggingGlobals._debug_stream = sys.stderr
        else:
            # Add pid to the filename.
            try:
                target_file = DebugInfoHolder.PYDEVD_DEBUG_FILE
                debug_file = _compute_filename_with_pid(target_file)
                _LoggingGlobals._debug_stream = open(debug_file, 'w')
                _LoggingGlobals._debug_stream_filename = debug_file
            except Exception:
                _LoggingGlobals._debug_stream = sys.stderr
                # Don't fail when trying to setup logging, just show the exception.
                traceback.print_exc()


def _compute_filename_with_pid(target_file, pid=None):
    # Note: used in tests.
    dirname = os.path.dirname(target_file)
    basename = os.path.basename(target_file)
    try:
        os.makedirs(dirname)
    except Exception:
        pass  # Ignore error if it already exists.

    name, ext = os.path.splitext(basename)
    if pid is None:
        pid = os.getpid()
    return os.path.join(dirname, '%s.%s%s' % (name, pid, ext))


def log_to(log_file:str, log_level:int=3) -> None:
    with _LoggingGlobals._initialize_lock:
        # Can be set directly.
        DebugInfoHolder.DEBUG_TRACE_LEVEL = log_level

        if DebugInfoHolder.PYDEVD_DEBUG_FILE != log_file:
            # Note that we don't need to reset it unless it actually changed
            # (would be the case where it's set as an env var in a new process
            # and a subprocess initializes logging to the same value).
            _LoggingGlobals._debug_stream = NULL
            _LoggingGlobals._debug_stream_filename = None

            DebugInfoHolder.PYDEVD_DEBUG_FILE = log_file

            _LoggingGlobals._debug_stream_initialized = False


def list_log_files(pydevd_debug_file):
    log_files = []
    dirname = os.path.dirname(pydevd_debug_file)
    basename = os.path.basename(pydevd_debug_file)
    if os.path.isdir(dirname):
        name, ext = os.path.splitext(basename)
        for f in os.listdir(dirname):
            if f.startswith(name) and f.endswith(ext):
                log_files.append(os.path.join(dirname, f))
    return log_files


@contextmanager
def log_context(trace_level, stream):
    '''
    To be used to temporarily change the logging settings.
    '''
    with _LoggingGlobals._initialize_lock:
        original_trace_level = DebugInfoHolder.DEBUG_TRACE_LEVEL
        original_debug_stream = _LoggingGlobals._debug_stream
        original_pydevd_debug_file = DebugInfoHolder.PYDEVD_DEBUG_FILE
        original_debug_stream_filename = _LoggingGlobals._debug_stream_filename
        original_initialized = _LoggingGlobals._debug_stream_initialized

        DebugInfoHolder.DEBUG_TRACE_LEVEL = trace_level
        _LoggingGlobals._debug_stream = stream
        _LoggingGlobals._debug_stream_initialized = True
    try:
        yield
    finally:
        with _LoggingGlobals._initialize_lock:
            DebugInfoHolder.DEBUG_TRACE_LEVEL = original_trace_level
            _LoggingGlobals._debug_stream = original_debug_stream
            DebugInfoHolder.PYDEVD_DEBUG_FILE = original_pydevd_debug_file
            _LoggingGlobals._debug_stream_filename = original_debug_stream_filename
            _LoggingGlobals._debug_stream_initialized = original_initialized


import time
_last_log_time = time.time()

# Set to True to show pid in each logged message (usually the file has it, but sometimes it's handy).
_LOG_PID = False


def _pydevd_log(level, msg, *args):
    '''
    Levels are:

    0 most serious warnings/errors (always printed)
    1 warnings/significant events
    2 informational trace
    3 verbose mode
    '''
    if level <= DebugInfoHolder.DEBUG_TRACE_LEVEL:
        # yes, we can have errors printing if the console of the program has been finished (and we're still trying to print something)
        try:
            try:
                if args:
                    msg = msg % args
            except:
                msg = '%s - %s' % (msg, args)

            if LOG_TIME:
                global _last_log_time
                new_log_time = time.time()
                time_diff = new_log_time - _last_log_time
                _last_log_time = new_log_time
                msg = '%.2fs - %s\n' % (time_diff, msg,)
            else:
                msg = '%s\n' % (msg,)

            if _LOG_PID:
                msg = '<%s> - %s\n' % (os.getpid(), msg,)

            try:
                try:
                    initialize_debug_stream()  # Do it as late as possible
                    _LoggingGlobals._debug_stream.write(msg)
                except TypeError:
                    if isinstance(msg, bytes):
                        # Depending on the StringIO flavor, it may only accept unicode.
                        msg = msg.decode('utf-8', 'replace')
                        _LoggingGlobals._debug_stream.write(msg)
            except UnicodeEncodeError:
                # When writing to the stream it's possible that the string can't be represented
                # in the encoding expected (in this case, convert it to the stream encoding
                # or ascii if we can't find one suitable using a suitable replace).
                encoding = getattr(_LoggingGlobals._debug_stream, 'encoding', 'ascii')
                msg = msg.encode(encoding, 'backslashreplace')
                msg = msg.decode(encoding)
                _LoggingGlobals._debug_stream.write(msg)

            _LoggingGlobals._debug_stream.flush()
        except:
            pass
        return True


def _pydevd_log_exception(msg='', *args):
    if msg or args:
        _pydevd_log(0, msg, *args)
    try:
        initialize_debug_stream()  # Do it as late as possible
        traceback.print_exc(file=_LoggingGlobals._debug_stream)
        _LoggingGlobals._debug_stream.flush()
    except:
        raise


def verbose(msg, *args):
    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
        _pydevd_log(3, msg, *args)


def debug(msg, *args):
    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
        _pydevd_log(2, msg, *args)


def info(msg, *args):
    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
        _pydevd_log(1, msg, *args)


warn = info


def critical(msg, *args):
    _pydevd_log(0, msg, *args)


def exception(msg='', *args):
    try:
        _pydevd_log_exception(msg, *args)
    except:
        pass  # Should never fail (even at interpreter shutdown).


error = exception


def error_once(msg, *args):
    try:
        if args:
            message = msg % args
        else:
            message = str(msg)
    except:
        message = '%s - %s' % (msg, args)

    if message not in _LoggingGlobals._warn_once_map:
        _LoggingGlobals._warn_once_map[message] = True
        critical(message)


def exception_once(msg, *args):
    try:
        if args:
            message = msg % args
        else:
            message = str(msg)
    except:
        message = '%s - %s' % (msg, args)

    if message not in _LoggingGlobals._warn_once_map:
        _LoggingGlobals._warn_once_map[message] = True
        exception(message)


def debug_once(msg, *args):
    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
        error_once(msg, *args)


def show_compile_cython_command_line():
    if SHOW_COMPILE_CYTHON_COMMAND_LINE:
        dirname = os.path.dirname(os.path.dirname(__file__))
        error_once("warning: Debugger speedups using cython not found. Run '\"%s\" \"%s\" build_ext --inplace' to build.",
            sys.executable, os.path.join(dirname, 'setup_pydevd_cython.py'))

