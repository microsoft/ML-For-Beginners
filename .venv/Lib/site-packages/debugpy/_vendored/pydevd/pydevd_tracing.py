from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
    IS_LINUX, IS_MAC, DebugInfoHolder, LOAD_NATIVE_LIB_FLAG, \
    ENV_FALSE_LOWER_VALUES, ForkSafeLock
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback

_original_settrace = sys.settrace


class TracingFunctionHolder:
    '''This class exists just to keep some variables (so that we don't keep them in the global namespace).
    '''
    _original_tracing = None
    _warn = True
    _traceback_limit = 1
    _warnings_shown = {}


def get_exception_traceback_str():
    exc_info = sys.exc_info()
    s = StringIO()
    traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=s)
    return s.getvalue()


def _get_stack_str(frame):

    msg = '\nIf this is needed, please check: ' + \
          '\nhttp://pydev.blogspot.com/2007/06/why-cant-pydev-debugger-work-with.html' + \
          '\nto see how to restore the debug tracing back correctly.\n'

    if TracingFunctionHolder._traceback_limit:
        s = StringIO()
        s.write('Call Location:\n')
        traceback.print_stack(f=frame, limit=TracingFunctionHolder._traceback_limit, file=s)
        msg = msg + s.getvalue()

    return msg


def _internal_set_trace(tracing_func):
    if TracingFunctionHolder._warn:
        frame = get_frame()
        if frame is not None and frame.f_back is not None:
            filename = os.path.splitext(frame.f_back.f_code.co_filename.lower())[0]
            if filename.endswith('threadpool') and 'gevent' in filename:
                if tracing_func is None:
                    pydev_log.debug('Disabled internal sys.settrace from gevent threadpool.')
                    return

            elif not filename.endswith(
                    (
                        'threading',
                        'pydevd_tracing',
                    )
                ):

                message = \
                '\nPYDEV DEBUGGER WARNING:' + \
                '\nsys.settrace() should not be used when the debugger is being used.' + \
                '\nThis may cause the debugger to stop working correctly.' + \
                '%s' % _get_stack_str(frame.f_back)

                if message not in TracingFunctionHolder._warnings_shown:
                    # only warn about each message once...
                    TracingFunctionHolder._warnings_shown[message] = 1
                    sys.stderr.write('%s\n' % (message,))
                    sys.stderr.flush()

    if TracingFunctionHolder._original_tracing:
        TracingFunctionHolder._original_tracing(tracing_func)


_last_tracing_func_thread_local = threading.local()


def SetTrace(tracing_func):
    _last_tracing_func_thread_local.tracing_func = tracing_func

    if tracing_func is not None:
        if set_trace_to_threads(tracing_func, thread_idents=[thread.get_ident()], create_dummy_thread=False) == 0:
            # If we can use our own tracer instead of the one from sys.settrace, do it (the reason
            # is that this is faster than the Python version because we don't call
            # PyFrame_FastToLocalsWithError and PyFrame_LocalsToFast at each event!
            # (the difference can be huge when checking line events on frames as the
            # time increases based on the number of local variables in the scope)
            # See: InternalCallTrampoline (on the C side) for details.
            return

    # If it didn't work (or if it was None), use the Python version.
    set_trace = TracingFunctionHolder._original_tracing or sys.settrace
    set_trace(tracing_func)


def reapply_settrace():
    try:
        tracing_func = _last_tracing_func_thread_local.tracing_func
    except AttributeError:
        return
    else:
        SetTrace(tracing_func)


def replace_sys_set_trace_func():
    if TracingFunctionHolder._original_tracing is None:
        TracingFunctionHolder._original_tracing = sys.settrace
        sys.settrace = _internal_set_trace


def restore_sys_set_trace_func():
    if TracingFunctionHolder._original_tracing is not None:
        sys.settrace = TracingFunctionHolder._original_tracing
        TracingFunctionHolder._original_tracing = None


_lock = ForkSafeLock()


def _load_python_helper_lib():
    try:
        # If it's already loaded, just return it.
        return _load_python_helper_lib.__lib__
    except AttributeError:
        pass
    with _lock:
        try:
            return _load_python_helper_lib.__lib__
        except AttributeError:
            pass

        lib = _load_python_helper_lib_uncached()
        _load_python_helper_lib.__lib__ = lib
        return lib


def get_python_helper_lib_filename():
    # Note: we have an independent (and similar -- but not equal) version of this method in
    # `add_code_to_python_process.py` which should be kept synchronized with this one (we do a copy
    # because the `pydevd_attach_to_process` is mostly independent and shouldn't be imported in the
    # debugger -- the only situation where it's imported is if the user actually does an attach to
    # process, through `attach_pydevd.py`, but this should usually be called from the IDE directly
    # and not from the debugger).
    libdir = os.path.join(os.path.dirname(__file__), 'pydevd_attach_to_process')

    arch = ''
    if IS_WINDOWS:
        # prefer not using platform.machine() when possible (it's a bit heavyweight as it may
        # spawn a subprocess).
        arch = os.environ.get("PROCESSOR_ARCHITEW6432", os.environ.get('PROCESSOR_ARCHITECTURE', ''))

    if not arch:
        arch = platform.machine()
        if not arch:
            pydev_log.info('platform.machine() did not return valid value.')  # This shouldn't happen...
            return None

    if IS_WINDOWS:
        extension = '.dll'
        suffix_64 = 'amd64'
        suffix_32 = 'x86'

    elif IS_LINUX:
        extension = '.so'
        suffix_64 = 'amd64'
        suffix_32 = 'x86'

    elif IS_MAC:
        extension = '.dylib'
        suffix_64 = 'x86_64'
        suffix_32 = 'x86'

    else:
        pydev_log.info('Unable to set trace to all threads in platform: %s', sys.platform)
        return None

    if arch.lower() not in ('amd64', 'x86', 'x86_64', 'i386', 'x86'):
        # We don't support this processor by default. Still, let's support the case where the
        # user manually compiled it himself with some heuristics.
        #
        # Ideally the user would provide a library in the format: "attach_<arch>.<extension>"
        # based on the way it's currently compiled -- see:
        # - windows/compile_windows.bat
        # - linux_and_mac/compile_linux.sh
        # - linux_and_mac/compile_mac.sh

        try:
            found = [name for name in os.listdir(libdir) if name.startswith('attach_') and name.endswith(extension)]
        except:
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                # There is no need to show this unless debug tracing is enabled.
                pydev_log.exception('Error listing dir: %s', libdir)
            return None

        expected_name = 'attach_' + arch + extension
        expected_name_linux = 'attach_linux_' + arch + extension

        filename = None
        if expected_name in found:  # Heuristic: user compiled with "attach_<arch>.<extension>"
            filename = os.path.join(libdir, expected_name)

        elif IS_LINUX and expected_name_linux in found:  # Heuristic: user compiled with "attach_linux_<arch>.<extension>"
            filename = os.path.join(libdir, expected_name_linux)

        elif len(found) == 1:  # Heuristic: user removed all libraries and just left his own lib.
            filename = os.path.join(libdir, found[0])

        else:  # Heuristic: there's one additional library which doesn't seem to be our own. Find the odd one.
            filtered = [name for name in found if not name.endswith((suffix_64 + extension, suffix_32 + extension))]
            if len(filtered) == 1:  # If more than one is available we can't be sure...
                filename = os.path.join(libdir, found[0])

        if filename is None:
            pydev_log.info(
                'Unable to set trace to all threads in arch: %s (did not find a %s lib in %s).',
                arch, expected_name, libdir

            )
            return None

        pydev_log.info('Using %s lib in arch: %s.', filename, arch)

    else:
        # Happy path for which we have pre-compiled binaries.
        if IS_64BIT_PROCESS:
            suffix = suffix_64
        else:
            suffix = suffix_32

        if IS_WINDOWS or IS_MAC:  # just the extension changes
            prefix = 'attach_'
        elif IS_LINUX:  #
            prefix = 'attach_linux_'  # historically it has a different name
        else:
            pydev_log.info('Unable to set trace to all threads in platform: %s', sys.platform)
            return None

        filename = os.path.join(libdir, '%s%s%s' % (prefix, suffix, extension))

    if not os.path.exists(filename):
        pydev_log.critical('Expected: %s to exist.', filename)
        return None

    return filename


def _load_python_helper_lib_uncached():
    if (not IS_CPYTHON or sys.version_info[:2] > (3, 11)
            or hasattr(sys, 'gettotalrefcount') or LOAD_NATIVE_LIB_FLAG in ENV_FALSE_LOWER_VALUES):
        pydev_log.info('Helper lib to set tracing to all threads not loaded.')
        return None

    try:
        filename = get_python_helper_lib_filename()
        if filename is None:
            return None
        # Load as pydll so that we don't release the gil.
        lib = ctypes.pydll.LoadLibrary(filename)
        pydev_log.info('Successfully Loaded helper lib to set tracing to all threads.')
        return lib
    except:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
            # Only show message if tracing is on (we don't have pre-compiled
            # binaries for all architectures -- i.e.: ARM).
            pydev_log.exception('Error loading: %s', filename)
        return None


def set_trace_to_threads(tracing_func, thread_idents=None, create_dummy_thread=True):
    assert tracing_func is not None

    ret = 0

    # Note: use sys._current_frames() keys to get the thread ids because it'll return
    # thread ids created in C/C++ where there's user code running, unlike the APIs
    # from the threading module which see only threads created through it (unless
    # a call for threading.current_thread() was previously done in that thread,
    # in which case a dummy thread would've been created for it).
    if thread_idents is None:
        thread_idents = set(sys._current_frames().keys())

        for t in threading.enumerate():
            # PY-44778: ignore pydevd threads and also add any thread that wasn't found on
            # sys._current_frames() as some existing threads may not appear in
            # sys._current_frames() but may be available through the `threading` module.
            if getattr(t, 'pydev_do_not_trace', False):
                thread_idents.discard(t.ident)
            else:
                thread_idents.add(t.ident)

    curr_ident = thread.get_ident()
    curr_thread = threading._active.get(curr_ident)

    if curr_ident in thread_idents and len(thread_idents) != 1:
        # The current thread must be updated first (because we need to set
        # the reference to `curr_thread`).
        thread_idents = list(thread_idents)
        thread_idents.remove(curr_ident)
        thread_idents.insert(0, curr_ident)

    for thread_ident in thread_idents:
        # If that thread is not available in the threading module we also need to create a
        # dummy thread for it (otherwise it'll be invisible to the debugger).
        if create_dummy_thread:
            if thread_ident not in threading._active:

                class _DummyThread(threading._DummyThread):

                    def _set_ident(self):
                        # Note: Hack to set the thread ident that we want.
                        self._ident = thread_ident

                t = _DummyThread()
                # Reset to the base class (don't expose our own version of the class).
                t.__class__ = threading._DummyThread

                if thread_ident == curr_ident:
                    curr_thread = t

                with threading._active_limbo_lock:
                    # On Py2 it'll put in active getting the current indent, not using the
                    # ident that was set, so, we have to update it (should be harmless on Py3
                    # so, do it always).
                    threading._active[thread_ident] = t
                    threading._active[curr_ident] = curr_thread

                    if t.ident != thread_ident:
                        # Check if it actually worked.
                        pydev_log.critical('pydevd: creation of _DummyThread with fixed thread ident did not succeed.')

        # Some (ptvsd) tests failed because of this, so, leave it always disabled for now.
        # show_debug_info = 1 if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1 else 0
        show_debug_info = 0

        # Hack to increase _Py_TracingPossible.
        # See comments on py_custom_pyeval_settrace.hpp
        proceed = thread.allocate_lock()
        proceed.acquire()

        def dummy_trace(frame, event, arg):
            return dummy_trace

        def increase_tracing_count():
            set_trace = TracingFunctionHolder._original_tracing or sys.settrace
            set_trace(dummy_trace)
            proceed.release()

        start_new_thread = pydev_monkey.get_original_start_new_thread(thread)
        start_new_thread(increase_tracing_count, ())
        proceed.acquire()  # Only proceed after the release() is done.
        proceed = None

        # Note: The set_trace_func is not really used anymore in the C side.
        set_trace_func = TracingFunctionHolder._original_tracing or sys.settrace

        lib = _load_python_helper_lib()
        if lib is None:  # This is the case if it's not CPython.
            pydev_log.info('Unable to load helper lib to set tracing to all threads (unsupported python vm).')
            ret = -1
        else:
            try:
                result = lib.AttachDebuggerTracing(
                    ctypes.c_int(show_debug_info),
                    ctypes.py_object(set_trace_func),
                    ctypes.py_object(tracing_func),
                    ctypes.c_uint(thread_ident),
                    ctypes.py_object(None),
                )
            except:
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                    # There is no need to show this unless debug tracing is enabled.
                    pydev_log.exception('Error attaching debugger tracing')
                ret = -1
            else:
                if result != 0:
                    pydev_log.info('Unable to set tracing for existing thread. Result: %s', result)
                    ret = result

    return ret

