'''
This module holds the constants used for specifying the states of the debugger.
'''
from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager

STATE_RUN = 1
STATE_SUSPEND = 2

PYTHON_SUSPEND = 1
DJANGO_SUSPEND = 2
JINJA2_SUSPEND = 3

int_types = (int,)

# types does not include a MethodWrapperType
try:
    MethodWrapperType = type([].__str__)
except:
    MethodWrapperType = None

import sys  # Note: the sys import must be here anyways (others depend on it)

# Preload codecs to avoid imports to them later on which can potentially halt the debugger.
import codecs as _codecs
for _codec in ["ascii", "utf8", "utf-8", "latin1", "latin-1", "idna"]:
    _codecs.lookup(_codec)


class DebugInfoHolder:
    # we have to put it here because it can be set through the command line (so, the
    # already imported references would not have it).

    # General information
    DEBUG_TRACE_LEVEL = 0  # 0 = critical, 1 = info, 2 = debug, 3 = verbose

    PYDEVD_DEBUG_FILE = None


# Any filename that starts with these strings is not traced nor shown to the user.
# In Python 3.7 "<frozen ..." appears multiple times during import and should be ignored for the user.
# In PyPy "<builtin> ..." can appear and should be ignored for the user.
# <attrs is used internally by attrs
# <__array_function__ is used by numpy
IGNORE_BASENAMES_STARTING_WITH = ('<frozen ', '<builtin', '<attrs', '<__array_function__')

# Note: <string> has special heuristics to know whether it should be traced or not (it's part of
# user code when it's the <string> used in python -c and part of the library otherwise).

# Any filename that starts with these strings is considered user (project) code. Note
# that files for which we have a source mapping are also considered as a part of the project.
USER_CODE_BASENAMES_STARTING_WITH = ('<ipython',)

# Any filename that starts with these strings is considered library code (note: checked after USER_CODE_BASENAMES_STARTING_WITH).
LIBRARY_CODE_BASENAMES_STARTING_WITH = ('<',)

IS_CPYTHON = platform.python_implementation() == 'CPython'

# Hold a reference to the original _getframe (because psyco will change that as soon as it's imported)
IS_IRONPYTHON = sys.platform == 'cli'
try:
    get_frame = sys._getframe
    if IS_IRONPYTHON:

        def get_frame():
            try:
                return sys._getframe()
            except ValueError:
                pass

except AttributeError:

    def get_frame():
        raise AssertionError('sys._getframe not available (possible causes: enable -X:Frames on IronPython?)')

# Used to determine the maximum size of each variable passed to eclipse -- having a big value here may make
# the communication slower -- as the variables are being gathered lazily in the latest version of eclipse,
# this value was raised from 200 to 1000.
MAXIMUM_VARIABLE_REPRESENTATION_SIZE = 1000
# Prefix for saving functions return values in locals
RETURN_VALUES_DICT = '__pydevd_ret_val_dict'
GENERATED_LEN_ATTR_NAME = 'len()'

import os

from _pydevd_bundle import pydevd_vm_type

# Constant detects when running on Jython/windows properly later on.
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform in ('linux', 'linux2')
IS_MAC = sys.platform == 'darwin'
IS_WASM = sys.platform == 'emscripten' or sys.platform == 'wasi'

IS_64BIT_PROCESS = sys.maxsize > (2 ** 32)

IS_JYTHON = pydevd_vm_type.get_vm_type() == pydevd_vm_type.PydevdVmType.JYTHON

IS_PYPY = platform.python_implementation() == 'PyPy'

if IS_JYTHON:
    import java.lang.System  # @UnresolvedImport
    IS_WINDOWS = java.lang.System.getProperty("os.name").lower().startswith("windows")

USE_CUSTOM_SYS_CURRENT_FRAMES = not hasattr(sys, '_current_frames') or IS_PYPY
USE_CUSTOM_SYS_CURRENT_FRAMES_MAP = USE_CUSTOM_SYS_CURRENT_FRAMES and (IS_PYPY or IS_IRONPYTHON)

if USE_CUSTOM_SYS_CURRENT_FRAMES:

    # Some versions of Jython don't have it (but we can provide a replacement)
    if IS_JYTHON:
        from java.lang import NoSuchFieldException
        from org.python.core import ThreadStateMapping
        try:
            cachedThreadState = ThreadStateMapping.getDeclaredField('globalThreadStates')  # Dev version
        except NoSuchFieldException:
            cachedThreadState = ThreadStateMapping.getDeclaredField('cachedThreadState')  # Release Jython 2.7.0
        cachedThreadState.accessible = True
        thread_states = cachedThreadState.get(ThreadStateMapping)

        def _current_frames():
            as_array = thread_states.entrySet().toArray()
            ret = {}
            for thread_to_state in as_array:
                thread = thread_to_state.getKey()
                if thread is None:
                    continue
                thread_state = thread_to_state.getValue()
                if thread_state is None:
                    continue

                frame = thread_state.frame
                if frame is None:
                    continue

                ret[thread.getId()] = frame
            return ret

    elif USE_CUSTOM_SYS_CURRENT_FRAMES_MAP:
        constructed_tid_to_last_frame = {}

        # IronPython doesn't have it. Let's use our workaround...
        def _current_frames():
            return constructed_tid_to_last_frame

    else:
        raise RuntimeError('Unable to proceed (sys._current_frames not available in this Python implementation).')
else:
    _current_frames = sys._current_frames

IS_PYTHON_STACKLESS = "stackless" in sys.version.lower()
CYTHON_SUPPORTED = False

python_implementation = platform.python_implementation()
if python_implementation == 'CPython':
    # Only available for CPython!
    CYTHON_SUPPORTED = True

#=======================================================================================================================
# Python 3?
#=======================================================================================================================
IS_PY36_OR_GREATER = sys.version_info >= (3, 6)
IS_PY37_OR_GREATER = sys.version_info >= (3, 7)
IS_PY38_OR_GREATER = sys.version_info >= (3, 8)
IS_PY39_OR_GREATER = sys.version_info >= (3, 9)
IS_PY310_OR_GREATER = sys.version_info >= (3, 10)
IS_PY311_OR_GREATER = sys.version_info >= (3, 11)


def version_str(v):
    return '.'.join((str(x) for x in v[:3])) + ''.join((str(x) for x in v[3:]))


PY_VERSION_STR = version_str(sys.version_info)
try:
    PY_IMPL_VERSION_STR = version_str(sys.implementation.version)
except AttributeError:
    PY_IMPL_VERSION_STR = ''

try:
    PY_IMPL_NAME = sys.implementation.name
except AttributeError:
    PY_IMPL_NAME = ''

ENV_TRUE_LOWER_VALUES = ('yes', 'true', '1')
ENV_FALSE_LOWER_VALUES = ('no', 'false', '0')


def is_true_in_env(env_key):
    if isinstance(env_key, tuple):
        # If a tuple, return True if any of those ends up being true.
        for v in env_key:
            if is_true_in_env(v):
                return True
        return False
    else:
        return os.getenv(env_key, '').lower() in ENV_TRUE_LOWER_VALUES


def as_float_in_env(env_key, default):
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        raise RuntimeError(
            'Error: expected the env variable: %s to be set to a float value. Found: %s' % (
                env_key, value))


def as_int_in_env(env_key, default):
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        raise RuntimeError(
            'Error: expected the env variable: %s to be set to a int value. Found: %s' % (
                env_key, value))


# If true in env, use gevent mode.
SUPPORT_GEVENT = is_true_in_env('GEVENT_SUPPORT')

# Opt-in support to show gevent paused greenlets. False by default because if too many greenlets are
# paused the UI can slow-down (i.e.: if 1000 greenlets are paused, each one would be shown separate
# as a different thread, but if the UI isn't optimized for that the experience is lacking...).
GEVENT_SHOW_PAUSED_GREENLETS = is_true_in_env('GEVENT_SHOW_PAUSED_GREENLETS')

DISABLE_FILE_VALIDATION = is_true_in_env('PYDEVD_DISABLE_FILE_VALIDATION')

GEVENT_SUPPORT_NOT_SET_MSG = os.getenv(
    'GEVENT_SUPPORT_NOT_SET_MSG',
    'It seems that the gevent monkey-patching is being used.\n'
    'Please set an environment variable with:\n'
    'GEVENT_SUPPORT=True\n'
    'to enable gevent support in the debugger.'
)

USE_LIB_COPY = SUPPORT_GEVENT

INTERACTIVE_MODE_AVAILABLE = sys.platform in ('darwin', 'win32') or os.getenv('DISPLAY') is not None

# If true in env, forces cython to be used (raises error if not available).
# If false in env, disables it.
# If not specified, uses default heuristic to determine if it should be loaded.
USE_CYTHON_FLAG = os.getenv('PYDEVD_USE_CYTHON')

if USE_CYTHON_FLAG is not None:
    USE_CYTHON_FLAG = USE_CYTHON_FLAG.lower()
    if USE_CYTHON_FLAG not in ENV_TRUE_LOWER_VALUES and USE_CYTHON_FLAG not in ENV_FALSE_LOWER_VALUES:
        raise RuntimeError('Unexpected value for PYDEVD_USE_CYTHON: %s (enable with one of: %s, disable with one of: %s)' % (
            USE_CYTHON_FLAG, ENV_TRUE_LOWER_VALUES, ENV_FALSE_LOWER_VALUES))

else:
    if not CYTHON_SUPPORTED:
        USE_CYTHON_FLAG = 'no'

# If true in env, forces frame eval to be used (raises error if not available).
# If false in env, disables it.
# If not specified, uses default heuristic to determine if it should be loaded.
PYDEVD_USE_FRAME_EVAL = os.getenv('PYDEVD_USE_FRAME_EVAL', '').lower()

# Values used to determine how much container items will be shown.
# PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS:
#     - Defines how many items will appear initially expanded after which a 'more...' will appear.
#
# PYDEVD_CONTAINER_BUCKET_SIZE
#    - Defines the size of each bucket inside the 'more...' item
#        i.e.: a bucket with size == 2 would show items such as:
#            - [2:4]
#            - [4:6]
#            ...
#
# PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS
#    - Defines the maximum number of items for dicts and sets.
#
PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS = as_int_in_env('PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS', 100)
PYDEVD_CONTAINER_BUCKET_SIZE = as_int_in_env('PYDEVD_CONTAINER_BUCKET_SIZE', 1000)
PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS = as_int_in_env('PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS', 500)
PYDEVD_CONTAINER_NUMPY_MAX_ITEMS = as_int_in_env('PYDEVD_CONTAINER_NUMPY_MAX_ITEMS', 500)

PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING = is_true_in_env('PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING')

# If specified in PYDEVD_IPYTHON_CONTEXT it must be a string with the basename
# and then the name of 2 methods in which the evaluate is done.
PYDEVD_IPYTHON_CONTEXT = ('interactiveshell.py', 'run_code', 'run_ast_nodes')
_ipython_ctx = os.getenv('PYDEVD_IPYTHON_CONTEXT')
if _ipython_ctx:
    PYDEVD_IPYTHON_CONTEXT = tuple(x.strip() for x in _ipython_ctx.split(','))
    assert len(PYDEVD_IPYTHON_CONTEXT) == 3, 'Invalid PYDEVD_IPYTHON_CONTEXT: %s' % (_ipython_ctx,)

# Use to disable loading the lib to set tracing to all threads (default is using heuristics based on where we're running).
LOAD_NATIVE_LIB_FLAG = os.getenv('PYDEVD_LOAD_NATIVE_LIB', '').lower()

LOG_TIME = os.getenv('PYDEVD_LOG_TIME', 'true').lower() in ENV_TRUE_LOWER_VALUES

SHOW_COMPILE_CYTHON_COMMAND_LINE = is_true_in_env('PYDEVD_SHOW_COMPILE_CYTHON_COMMAND_LINE')

LOAD_VALUES_ASYNC = is_true_in_env('PYDEVD_LOAD_VALUES_ASYNC')
DEFAULT_VALUE = "__pydevd_value_async"
ASYNC_EVAL_TIMEOUT_SEC = 60
NEXT_VALUE_SEPARATOR = "__pydev_val__"
BUILTINS_MODULE_NAME = 'builtins'

# Pandas customization.
PANDAS_MAX_ROWS = as_int_in_env('PYDEVD_PANDAS_MAX_ROWS', 60)
PANDAS_MAX_COLS = as_int_in_env('PYDEVD_PANDAS_MAX_COLS', 10)
PANDAS_MAX_COLWIDTH = as_int_in_env('PYDEVD_PANDAS_MAX_COLWIDTH', 50)

# If getting an attribute or computing some value is too slow, let the user know if the given timeout elapses.
PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT = as_float_in_env('PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT', 0.50)

# This timeout is used to track the time to send a message saying that the evaluation
# is taking too long and possible mitigations.
PYDEVD_WARN_EVALUATION_TIMEOUT = as_float_in_env('PYDEVD_WARN_EVALUATION_TIMEOUT', 3.)

# If True in env shows a thread dump when the evaluation times out.
PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT = is_true_in_env('PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT')

# This timeout is used only when the mode that all threads are stopped/resumed at once is used
# (i.e.: multi_threads_single_notification)
#
# In this mode, if some evaluation doesn't finish until this timeout, we notify the user
# and then resume all threads until the evaluation finishes.
#
# A negative value will disable the timeout and a value of 0 will automatically run all threads
# (without any notification) when the evaluation is started and pause all threads when the
# evaluation is finished. A positive value will run run all threads after the timeout
# elapses.
PYDEVD_UNBLOCK_THREADS_TIMEOUT = as_float_in_env('PYDEVD_UNBLOCK_THREADS_TIMEOUT', -1.)

# Timeout to interrupt a thread (so, if some evaluation doesn't finish until this
# timeout, the thread doing the evaluation is interrupted).
# A value <= 0 means this is disabled.
# See: _pydevd_bundle.pydevd_timeout.create_interrupt_this_thread_callback for details
# on how the thread interruption works (there are some caveats related to it).
PYDEVD_INTERRUPT_THREAD_TIMEOUT = as_float_in_env('PYDEVD_INTERRUPT_THREAD_TIMEOUT', -1)

# If PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS is set to False, the patching to hide pydevd threads won't be applied.
PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS = os.getenv('PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS', 'true').lower() in ENV_TRUE_LOWER_VALUES

EXCEPTION_TYPE_UNHANDLED = 'UNHANDLED'
EXCEPTION_TYPE_USER_UNHANDLED = 'USER_UNHANDLED'
EXCEPTION_TYPE_HANDLED = 'HANDLED'

SHOW_DEBUG_INFO_ENV = is_true_in_env(('PYCHARM_DEBUG', 'PYDEV_DEBUG', 'PYDEVD_DEBUG'))

if SHOW_DEBUG_INFO_ENV:
    # show debug info before the debugger start
    DebugInfoHolder.DEBUG_TRACE_LEVEL = 3

DebugInfoHolder.PYDEVD_DEBUG_FILE = os.getenv('PYDEVD_DEBUG_FILE')


def protect_libraries_from_patching():
    """
    In this function we delete some modules from `sys.modules` dictionary and import them again inside
      `_pydev_saved_modules` in order to save their original copies there. After that we can use these
      saved modules within the debugger to protect them from patching by external libraries (e.g. gevent).
    """
    patched = ['threading', 'thread', '_thread', 'time', 'socket', 'queue', 'select',
               'xmlrpclib', 'SimpleXMLRPCServer', 'BaseHTTPServer', 'SocketServer',
               'xmlrpc.client', 'xmlrpc.server', 'http.server', 'socketserver']

    for name in patched:
        try:
            __import__(name)
        except:
            pass

    patched_modules = dict([(k, v) for k, v in sys.modules.items()
                            if k in patched])

    for name in patched_modules:
        del sys.modules[name]

    # import for side effects
    import _pydev_bundle._pydev_saved_modules

    for name in patched_modules:
        sys.modules[name] = patched_modules[name]


if USE_LIB_COPY:
    protect_libraries_from_patching()

from _pydev_bundle._pydev_saved_modules import thread, threading

_fork_safe_locks = []

if IS_JYTHON:

    def ForkSafeLock(rlock=False):
        if rlock:
            return threading.RLock()
        else:
            return threading.Lock()

else:

    class ForkSafeLock(object):
        '''
        A lock which is fork-safe (when a fork is done, `pydevd_constants.after_fork()`
        should be called to reset the locks in the new process to avoid deadlocks
        from a lock which was locked during the fork).

        Note:
            Unlike `threading.Lock` this class is not completely atomic, so, doing:

            lock = ForkSafeLock()
            with lock:
                ...

            is different than using `threading.Lock` directly because the tracing may
            find an additional function call on `__enter__` and on `__exit__`, so, it's
            not recommended to use this in all places, only where the forking may be important
            (so, for instance, the locks on PyDB should not be changed to this lock because
            of that -- and those should all be collected in the new process because PyDB itself
            should be completely cleared anyways).

            It's possible to overcome this limitation by using `ForkSafeLock.acquire` and
            `ForkSafeLock.release` instead of the context manager (as acquire/release are
            bound to the original implementation, whereas __enter__/__exit__ is not due to Python
            limitations).
        '''

        def __init__(self, rlock=False):
            self._rlock = rlock
            self._init()
            _fork_safe_locks.append(weakref.ref(self))

        def __enter__(self):
            return self._lock.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._lock.__exit__(exc_type, exc_val, exc_tb)

        def _init(self):
            if self._rlock:
                self._lock = threading.RLock()
            else:
                self._lock = thread.allocate_lock()

            self.acquire = self._lock.acquire
            self.release = self._lock.release
            _fork_safe_locks.append(weakref.ref(self))


def after_fork():
    '''
    Must be called after a fork operation (will reset the ForkSafeLock).
    '''
    global _fork_safe_locks
    locks = _fork_safe_locks[:]
    _fork_safe_locks = []
    for lock in locks:
        lock = lock()
        if lock is not None:
            lock._init()


_thread_id_lock = ForkSafeLock()
thread_get_ident = thread.get_ident


def as_str(s):
    assert isinstance(s, str)
    return s


@contextmanager
def filter_all_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


def silence_warnings_decorator(func):

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with filter_all_warnings():
            return func(*args, **kwargs)

    return new_func


def sorted_dict_repr(d):
    s = sorted(d.items(), key=lambda x:str(x[0]))
    return '{' + ', '.join(('%r: %r' % x) for x in s) + '}'


def iter_chars(b):
    # In Python 2, we can iterate bytes or str with individual characters, but Python 3 onwards
    # changed that behavior so that when iterating bytes we actually get ints!
    if isinstance(b, bytes):
        # i.e.: do something as struct.unpack('3c', b)
        return iter(struct.unpack(str(len(b)) + 'c', b))
    return iter(b)


if IS_JYTHON:

    def NO_FTRACE(frame, event, arg):
        return None

else:
    _curr_trace = sys.gettrace()

    # Set a temporary trace which does nothing for us to test (otherwise setting frame.f_trace has no
    # effect).
    def _temp_trace(frame, event, arg):
        return None

    sys.settrace(_temp_trace)

    def _check_ftrace_set_none():
        '''
        Will throw an error when executing a line event
        '''
        sys._getframe().f_trace = None
        _line_event = 1
        _line_event = 2

    try:
        _check_ftrace_set_none()

        def NO_FTRACE(frame, event, arg):
            frame.f_trace = None
            return None

    except TypeError:

        def NO_FTRACE(frame, event, arg):
            # In Python <= 2.6 and <= 3.4, if we're tracing a method, frame.f_trace may not be set
            # to None, it must always be set to a tracing function.
            # See: tests_python.test_tracing_gotchas.test_tracing_gotchas
            #
            # Note: Python 2.7 sometimes works and sometimes it doesn't depending on the minor
            # version because of https://bugs.python.org/issue20041 (although bug reports didn't
            # include the minor version, so, mark for any Python 2.7 as I'm not completely sure
            # the fix in later 2.7 versions is the same one we're dealing with).
            return None

    sys.settrace(_curr_trace)


#=======================================================================================================================
# get_pid
#=======================================================================================================================
def get_pid():
    try:
        return os.getpid()
    except AttributeError:
        try:
            # Jython does not have it!
            import java.lang.management.ManagementFactory  # @UnresolvedImport -- just for jython
            pid = java.lang.management.ManagementFactory.getRuntimeMXBean().getName()
            return pid.replace('@', '_')
        except:
            # ok, no pid available (will be unable to debug multiple processes)
            return '000001'


def clear_cached_thread_id(thread):
    with _thread_id_lock:
        try:
            if thread.__pydevd_id__ != 'console_main':
                # The console_main is a special thread id used in the console and its id should never be reset
                # (otherwise we may no longer be able to get its variables -- see: https://www.brainwy.com/tracker/PyDev/776).
                del thread.__pydevd_id__
        except AttributeError:
            pass


# Don't let threads be collected (so that id(thread) is guaranteed to be unique).
_thread_id_to_thread_found = {}


def _get_or_compute_thread_id_with_lock(thread, is_current_thread):
    with _thread_id_lock:
        # We do a new check with the lock in place just to be sure that nothing changed
        tid = getattr(thread, '__pydevd_id__', None)
        if tid is not None:
            return tid

        _thread_id_to_thread_found[id(thread)] = thread

        # Note: don't use thread.ident because a new thread may have the
        # same id from an old thread.
        pid = get_pid()
        tid = 'pid_%s_id_%s' % (pid, id(thread))

        thread.__pydevd_id__ = tid

    return tid


def get_current_thread_id(thread):
    '''
    Note: the difference from get_current_thread_id to get_thread_id is that
    for the current thread we can get the thread id while the thread.ident
    is still not set in the Thread instance.
    '''
    try:
        # Fast path without getting lock.
        tid = thread.__pydevd_id__
        if tid is None:
            # Fix for https://www.brainwy.com/tracker/PyDev/645
            # if __pydevd_id__ is None, recalculate it... also, use an heuristic
            # that gives us always the same id for the thread (using thread.ident or id(thread)).
            raise AttributeError()
    except AttributeError:
        tid = _get_or_compute_thread_id_with_lock(thread, is_current_thread=True)

    return tid


def get_thread_id(thread):
    try:
        # Fast path without getting lock.
        tid = thread.__pydevd_id__
        if tid is None:
            # Fix for https://www.brainwy.com/tracker/PyDev/645
            # if __pydevd_id__ is None, recalculate it... also, use an heuristic
            # that gives us always the same id for the thread (using thread.ident or id(thread)).
            raise AttributeError()
    except AttributeError:
        tid = _get_or_compute_thread_id_with_lock(thread, is_current_thread=False)

    return tid


def set_thread_id(thread, thread_id):
    with _thread_id_lock:
        thread.__pydevd_id__ = thread_id


#=======================================================================================================================
# Null
#=======================================================================================================================
class Null:
    """
    Gotten from: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/68205
    """

    def __init__(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return self

    def __getattr__(self, mname):
        if len(mname) > 4 and mname[:2] == '__' and mname[-2:] == '__':
            # Don't pretend to implement special method names.
            raise AttributeError(mname)
        return self

    def __setattr__(self, name, value):
        return self

    def __delattr__(self, name):
        return self

    def __repr__(self):
        return "<Null>"

    def __str__(self):
        return "Null"

    def __len__(self):
        return 0

    def __getitem__(self):
        return self

    def __setitem__(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def __nonzero__(self):
        return 0

    def __iter__(self):
        return iter(())


# Default instance
NULL = Null()


class KeyifyList(object):

    def __init__(self, inner, key):
        self.inner = inner
        self.key = key

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, k):
        return self.key(self.inner[k])


def call_only_once(func):
    '''
    To be used as a decorator

    @call_only_once
    def func():
        print 'Calling func only this time'

    Actually, in PyDev it must be called as:

    func = call_only_once(func) to support older versions of Python.
    '''

    def new_func(*args, **kwargs):
        if not new_func._called:
            new_func._called = True
            return func(*args, **kwargs)

    new_func._called = False
    return new_func


# Protocol where each line is a new message (text is quoted to prevent new lines).
# payload is xml
QUOTED_LINE_PROTOCOL = 'quoted-line'
ARGUMENT_QUOTED_LINE_PROTOCOL = 'protocol-quoted-line'

# Uses http protocol to provide a new message.
# i.e.: Content-Length:xxx\r\n\r\npayload
# payload is xml
HTTP_PROTOCOL = 'http'
ARGUMENT_HTTP_PROTOCOL = 'protocol-http'

# Message is sent without any header.
# payload is json
JSON_PROTOCOL = 'json'
ARGUMENT_JSON_PROTOCOL = 'json-dap'

# Same header as the HTTP_PROTOCOL
# payload is json
HTTP_JSON_PROTOCOL = 'http_json'
ARGUMENT_HTTP_JSON_PROTOCOL = 'json-dap-http'

ARGUMENT_PPID = 'ppid'


class _GlobalSettings:
    protocol = QUOTED_LINE_PROTOCOL


def set_protocol(protocol):
    expected = (HTTP_PROTOCOL, QUOTED_LINE_PROTOCOL, JSON_PROTOCOL, HTTP_JSON_PROTOCOL)
    assert protocol in expected, 'Protocol (%s) should be one of: %s' % (
        protocol, expected)

    _GlobalSettings.protocol = protocol


def get_protocol():
    return _GlobalSettings.protocol


def is_json_protocol():
    return _GlobalSettings.protocol in (JSON_PROTOCOL, HTTP_JSON_PROTOCOL)


class GlobalDebuggerHolder:
    '''
        Holder for the global debugger.
    '''
    global_dbg = None  # Note: don't rename (the name is used in our attach to process)


def get_global_debugger():
    return GlobalDebuggerHolder.global_dbg


GetGlobalDebugger = get_global_debugger  # Backward-compatibility


def set_global_debugger(dbg):
    GlobalDebuggerHolder.global_dbg = dbg


if __name__ == '__main__':
    if Null():
        sys.stdout.write('here\n')

