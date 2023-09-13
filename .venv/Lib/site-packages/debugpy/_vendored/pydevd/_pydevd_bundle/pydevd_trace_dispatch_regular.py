from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, NO_FTRACE,
    USE_CUSTOM_SYS_CURRENT_FRAMES_MAP, ForkSafeLock)
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER

# IFDEF CYTHON
# from cpython.object cimport PyObject
# from cpython.ref cimport Py_INCREF, Py_XDECREF
# ELSE
from _pydevd_bundle.pydevd_frame import PyDBFrame, is_unhandled_exception
# ENDIF

# IFDEF CYTHON
# cdef dict _global_notify_skipped_step_in
# cython_inline_constant: CMD_STEP_INTO = 107
# cython_inline_constant: CMD_STEP_INTO_MY_CODE = 144
# cython_inline_constant: CMD_STEP_RETURN = 109
# cython_inline_constant: CMD_STEP_RETURN_MY_CODE = 160
# ELSE
# Note: those are now inlined on cython.
CMD_STEP_INTO = 107
CMD_STEP_INTO_MY_CODE = 144
CMD_STEP_RETURN = 109
CMD_STEP_RETURN_MY_CODE = 160
# ENDIF

# Cache where we should keep that we completely skipped entering some context.
# It needs to be invalidated when:
# - Breakpoints are changed
# It can be used when running regularly (without step over/step in/step return)
global_cache_skips = {}
global_cache_frame_skips = {}

_global_notify_skipped_step_in = False
_global_notify_skipped_step_in_lock = ForkSafeLock()


def notify_skipped_step_in_because_of_filters(py_db, frame):
    global _global_notify_skipped_step_in

    with _global_notify_skipped_step_in_lock:
        if _global_notify_skipped_step_in:
            # Check with lock in place (callers should actually have checked
            # before without the lock in place due to performance).
            return
        _global_notify_skipped_step_in = True
        py_db.notify_skipped_step_in_because_of_filters(frame)

# IFDEF CYTHON
# cdef class SafeCallWrapper:
#     cdef method_object
#     def __init__(self, method_object):
#         self.method_object = method_object
#     def  __call__(self, *args):
#         #Cannot use 'self' once inside the delegate call since we are borrowing the self reference f_trace field
#         #in the frame, and that reference might get destroyed by set trace on frame and parents
#         cdef PyObject* method_obj = <PyObject*> self.method_object
#         Py_INCREF(<object>method_obj)
#         ret = (<object>method_obj)(*args)
#         Py_XDECREF (method_obj)
#         return SafeCallWrapper(ret) if ret is not None else None
#     def  get_method_object(self):
#         return self.method_object
# ELSE
# ENDIF


def fix_top_level_trace_and_get_trace_func(py_db, frame):
    # IFDEF CYTHON
    # cdef str filename;
    # cdef str name;
    # cdef tuple args;
    # ENDIF

    # Note: this is always the first entry-point in the tracing for any thread.
    # After entering here we'll set a new tracing function for this thread
    # where more information is cached (and will also setup the tracing for
    # frames where we should deal with unhandled exceptions).
    thread = None
    # Cache the frame which should be traced to deal with unhandled exceptions.
    # (i.e.: thread entry-points).

    f_unhandled = frame
    # print('called at', f_unhandled.f_code.co_name, f_unhandled.f_code.co_filename, f_unhandled.f_code.co_firstlineno)
    force_only_unhandled_tracer = False
    while f_unhandled is not None:
        # name = splitext(basename(f_unhandled.f_code.co_filename))[0]

        name = f_unhandled.f_code.co_filename
        # basename
        i = name.rfind('/')
        j = name.rfind('\\')
        if j > i:
            i = j
        if i >= 0:
            name = name[i + 1:]
        # remove ext
        i = name.rfind('.')
        if i >= 0:
            name = name[:i]

        if name == 'threading':
            if f_unhandled.f_code.co_name in ('__bootstrap', '_bootstrap'):
                # We need __bootstrap_inner, not __bootstrap.
                return None, False

            elif f_unhandled.f_code.co_name in ('__bootstrap_inner', '_bootstrap_inner'):
                # Note: be careful not to use threading.currentThread to avoid creating a dummy thread.
                t = f_unhandled.f_locals.get('self')
                force_only_unhandled_tracer = True
                if t is not None and isinstance(t, threading.Thread):
                    thread = t
                    break

        elif name == 'pydev_monkey':
            if f_unhandled.f_code.co_name == '__call__':
                force_only_unhandled_tracer = True
                break

        elif name == 'pydevd':
            if f_unhandled.f_code.co_name in ('run', 'main'):
                # We need to get to _exec
                return None, False

            if f_unhandled.f_code.co_name == '_exec':
                force_only_unhandled_tracer = True
                break

        elif name == 'pydevd_tracing':
            return None, False

        elif f_unhandled.f_back is None:
            break

        f_unhandled = f_unhandled.f_back

    if thread is None:
        # Important: don't call threadingCurrentThread if we're in the threading module
        # to avoid creating dummy threads.
        if py_db.threading_get_ident is not None:
            thread = py_db.threading_active.get(py_db.threading_get_ident())
            if thread is None:
                return None, False
        else:
            # Jython does not have threading.get_ident().
            thread = py_db.threading_current_thread()

    if getattr(thread, 'pydev_do_not_trace', None):
        py_db.disable_tracing()
        return None, False

    try:
        additional_info = thread.additional_info
        if additional_info is None:
            raise AttributeError()
    except:
        additional_info = py_db.set_additional_thread_info(thread)

    # print('enter thread tracer', thread, get_current_thread_id(thread))
    args = (py_db, thread, additional_info, global_cache_skips, global_cache_frame_skips)

    if f_unhandled is not None:
        if f_unhandled.f_back is None and not force_only_unhandled_tracer:
            # Happens when we attach to a running program (cannot reuse instance because it's mutable).
            top_level_thread_tracer = TopLevelThreadTracerNoBackFrame(ThreadTracer(args), args)
            additional_info.top_level_thread_tracer_no_back_frames.append(top_level_thread_tracer)  # Hack for cython to keep it alive while the thread is alive (just the method in the SetTrace is not enough).
        else:
            top_level_thread_tracer = additional_info.top_level_thread_tracer_unhandled
            if top_level_thread_tracer is None:
                # Stop in some internal place to report about unhandled exceptions
                top_level_thread_tracer = TopLevelThreadTracerOnlyUnhandledExceptions(args)
                additional_info.top_level_thread_tracer_unhandled = top_level_thread_tracer  # Hack for cython to keep it alive while the thread is alive (just the method in the SetTrace is not enough).

        # print(' --> found to trace unhandled', f_unhandled.f_code.co_name, f_unhandled.f_code.co_filename, f_unhandled.f_code.co_firstlineno)
        f_trace = top_level_thread_tracer.get_trace_dispatch_func()
        # IFDEF CYTHON
        # f_trace = SafeCallWrapper(f_trace)
        # ENDIF
        f_unhandled.f_trace = f_trace

        if frame is f_unhandled:
            return f_trace, False

    thread_tracer = additional_info.thread_tracer
    if thread_tracer is None or thread_tracer._args[0] is not py_db:
        thread_tracer = ThreadTracer(args)
        additional_info.thread_tracer = thread_tracer

# IFDEF CYTHON
#     return SafeCallWrapper(thread_tracer), True
# ELSE
    return thread_tracer, True
# ENDIF


def trace_dispatch(py_db, frame, event, arg):
    thread_trace_func, apply_to_settrace = py_db.fix_top_level_trace_and_get_trace_func(py_db, frame)
    if thread_trace_func is None:
        return None if event == 'call' else NO_FTRACE
    if apply_to_settrace:
        py_db.enable_tracing(thread_trace_func)
    return thread_trace_func(frame, event, arg)


# IFDEF CYTHON
# cdef class TopLevelThreadTracerOnlyUnhandledExceptions:
#     cdef public tuple _args;
#     def __init__(self, tuple args):
#         self._args = args
# ELSE
class TopLevelThreadTracerOnlyUnhandledExceptions(object):

    def __init__(self, args):
        self._args = args
# ENDIF

    def trace_unhandled_exceptions(self, frame, event, arg):
        # Note that we ignore the frame as this tracing method should only be put in topmost frames already.
        # print('trace_unhandled_exceptions', event, frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno)
        if event == 'exception' and arg is not None:
            py_db, t, additional_info = self._args[0:3]
            if arg is not None:
                if not additional_info.suspended_at_unhandled:
                    additional_info.suspended_at_unhandled = True

                    py_db.stop_on_unhandled_exception(py_db, t, additional_info, arg)

        # No need to reset frame.f_trace to keep the same trace function.
        return self.trace_unhandled_exceptions

    def get_trace_dispatch_func(self):
        return self.trace_unhandled_exceptions


# IFDEF CYTHON
# cdef class TopLevelThreadTracerNoBackFrame:
#
#     cdef public object _frame_trace_dispatch;
#     cdef public tuple _args;
#     cdef public object try_except_infos;
#     cdef public object _last_exc_arg;
#     cdef public set _raise_lines;
#     cdef public int _last_raise_line;
#
#     def __init__(self, frame_trace_dispatch, tuple args):
#         self._frame_trace_dispatch = frame_trace_dispatch
#         self._args = args
#         self.try_except_infos = None
#         self._last_exc_arg = None
#         self._raise_lines = set()
#         self._last_raise_line = -1
# ELSE
class TopLevelThreadTracerNoBackFrame(object):
    '''
    This tracer is pretty special in that it's dealing with a frame without f_back (i.e.: top frame
    on remote attach or QThread).

    This means that we have to carefully inspect exceptions to discover whether the exception will
    be unhandled or not (if we're dealing with an unhandled exception we need to stop as unhandled,
    otherwise we need to use the regular tracer -- unfortunately the debugger has little info to
    work with in the tracing -- see: https://bugs.python.org/issue34099, so, we inspect bytecode to
    determine if some exception will be traced or not... note that if this is not available -- such
    as on Jython -- we consider any top-level exception to be unnhandled).
    '''

    def __init__(self, frame_trace_dispatch, args):
        self._frame_trace_dispatch = frame_trace_dispatch
        self._args = args
        self.try_except_infos = None
        self._last_exc_arg = None
        self._raise_lines = set()
        self._last_raise_line = -1
# ENDIF

    def trace_dispatch_and_unhandled_exceptions(self, frame, event, arg):
        # DEBUG = 'code_to_debug' in frame.f_code.co_filename
        # if DEBUG: print('trace_dispatch_and_unhandled_exceptions: %s %s %s %s %s %s' % (event, frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno, self._frame_trace_dispatch, frame.f_lineno))
        frame_trace_dispatch = self._frame_trace_dispatch
        if frame_trace_dispatch is not None:
            self._frame_trace_dispatch = frame_trace_dispatch(frame, event, arg)

        if event == 'exception':
            self._last_exc_arg = arg
            self._raise_lines.add(frame.f_lineno)
            self._last_raise_line = frame.f_lineno

        elif event == 'return' and self._last_exc_arg is not None:
            # For unhandled exceptions we actually track the return when at the topmost level.
            try:
                py_db, t, additional_info = self._args[0:3]
                if not additional_info.suspended_at_unhandled:  # Note: only check it here, don't set.
                    if is_unhandled_exception(self, py_db, frame, self._last_raise_line, self._raise_lines):
                        py_db.stop_on_unhandled_exception(py_db, t, additional_info, self._last_exc_arg)
            finally:
                # Remove reference to exception after handling it.
                self._last_exc_arg = None

        ret = self.trace_dispatch_and_unhandled_exceptions

        # Need to reset (the call to _frame_trace_dispatch may have changed it).
        # IFDEF CYTHON
        # frame.f_trace = SafeCallWrapper(ret)
        # ELSE
        frame.f_trace = ret
        # ENDIF
        return ret

    def get_trace_dispatch_func(self):
        return self.trace_dispatch_and_unhandled_exceptions


# IFDEF CYTHON
# cdef class ThreadTracer:
#     cdef public tuple _args;
#     def __init__(self, tuple args):
#         self._args = args
# ELSE
class ThreadTracer(object):

    def __init__(self, args):
        self._args = args
# ENDIF

    def __call__(self, frame, event, arg):
        ''' This is the callback used when we enter some context in the debugger.

        We also decorate the thread we are in with info about the debugging.
        The attributes added are:
            pydev_state
            pydev_step_stop
            pydev_step_cmd
            pydev_notify_kill

        :param PyDB py_db:
            This is the global debugger (this method should actually be added as a method to it).
        '''
        # IFDEF CYTHON
        # cdef str filename;
        # cdef str base;
        # cdef int pydev_step_cmd;
        # cdef object frame_cache_key;
        # cdef dict cache_skips;
        # cdef bint is_stepping;
        # cdef tuple abs_path_canonical_path_and_base;
        # cdef PyDBAdditionalThreadInfo additional_info;
        # ENDIF

        # DEBUG = 'code_to_debug' in frame.f_code.co_filename
        # if DEBUG: print('ENTER: trace_dispatch: %s %s %s %s' % (frame.f_code.co_filename, frame.f_lineno, event, frame.f_code.co_name))
        py_db, t, additional_info, cache_skips, frame_skips_cache = self._args
        if additional_info.is_tracing:
            return None if event == 'call' else NO_FTRACE  # we don't wan't to trace code invoked from pydevd_frame.trace_dispatch

        additional_info.is_tracing += 1
        try:
            pydev_step_cmd = additional_info.pydev_step_cmd
            is_stepping = pydev_step_cmd != -1
            if py_db.pydb_disposed:
                return None if event == 'call' else NO_FTRACE

            # if thread is not alive, cancel trace_dispatch processing
            if not is_thread_alive(t):
                py_db.notify_thread_not_alive(get_current_thread_id(t))
                return None if event == 'call' else NO_FTRACE

            # Note: it's important that the context name is also given because we may hit something once
            # in the global context and another in the local context.
            frame_cache_key = frame.f_code
            if frame_cache_key in cache_skips:
                if not is_stepping:
                    # if DEBUG: print('skipped: trace_dispatch (cache hit)', frame_cache_key, frame.f_lineno, event, frame.f_code.co_name)
                    return None if event == 'call' else NO_FTRACE
                else:
                    # When stepping we can't take into account caching based on the breakpoints (only global filtering).
                    if cache_skips.get(frame_cache_key) == 1:

                        if additional_info.pydev_original_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE) and not _global_notify_skipped_step_in:
                            notify_skipped_step_in_because_of_filters(py_db, frame)

                        back_frame = frame.f_back
                        if back_frame is not None and pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                            back_frame_cache_key = back_frame.f_code
                            if cache_skips.get(back_frame_cache_key) == 1:
                                # if DEBUG: print('skipped: trace_dispatch (cache hit: 1)', frame_cache_key, frame.f_lineno, event, frame.f_code.co_name)
                                return None if event == 'call' else NO_FTRACE
                        else:
                            # if DEBUG: print('skipped: trace_dispatch (cache hit: 2)', frame_cache_key, frame.f_lineno, event, frame.f_code.co_name)
                            return None if event == 'call' else NO_FTRACE

            try:
                # Make fast path faster!
                abs_path_canonical_path_and_base = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
            except:
                abs_path_canonical_path_and_base = get_abs_path_real_path_and_base_from_frame(frame)

            file_type = py_db.get_file_type(frame, abs_path_canonical_path_and_base)  # we don't want to debug threading or anything related to pydevd

            if file_type is not None:
                if file_type == 1:  # inlining LIB_FILE = 1
                    if not py_db.in_project_scope(frame, abs_path_canonical_path_and_base[0]):
                        # if DEBUG: print('skipped: trace_dispatch (not in scope)', abs_path_canonical_path_and_base[2], frame.f_lineno, event, frame.f_code.co_name, file_type)
                        cache_skips[frame_cache_key] = 1
                        return None if event == 'call' else NO_FTRACE
                else:
                    # if DEBUG: print('skipped: trace_dispatch', abs_path_canonical_path_and_base[2], frame.f_lineno, event, frame.f_code.co_name, file_type)
                    cache_skips[frame_cache_key] = 1
                    return None if event == 'call' else NO_FTRACE

            if py_db.is_files_filter_enabled:
                if py_db.apply_files_filter(frame, abs_path_canonical_path_and_base[0], False):
                    cache_skips[frame_cache_key] = 1

                    if is_stepping and additional_info.pydev_original_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE) and not _global_notify_skipped_step_in:
                        notify_skipped_step_in_because_of_filters(py_db, frame)

                    # A little gotcha, sometimes when we're stepping in we have to stop in a
                    # return event showing the back frame as the current frame, so, we need
                    # to check not only the current frame but the back frame too.
                    back_frame = frame.f_back
                    if back_frame is not None and pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                        if py_db.apply_files_filter(back_frame, back_frame.f_code.co_filename, False):
                            back_frame_cache_key = back_frame.f_code
                            cache_skips[back_frame_cache_key] = 1
                            # if DEBUG: print('skipped: trace_dispatch (filtered out: 1)', frame_cache_key, frame.f_lineno, event, frame.f_code.co_name)
                            return None if event == 'call' else NO_FTRACE
                    else:
                        # if DEBUG: print('skipped: trace_dispatch (filtered out: 2)', frame_cache_key, frame.f_lineno, event, frame.f_code.co_name)
                        return None if event == 'call' else NO_FTRACE

            # if DEBUG: print('trace_dispatch', filename, frame.f_lineno, event, frame.f_code.co_name, file_type)

            # Just create PyDBFrame directly (removed support for Python versions < 2.5, which required keeping a weak
            # reference to the frame).
            ret = PyDBFrame(
                (
                    py_db, abs_path_canonical_path_and_base, additional_info, t, frame_skips_cache, frame_cache_key,
                )
            ).trace_dispatch(frame, event, arg)
            if ret is None:
                # 1 means skipped because of filters.
                # 2 means skipped because no breakpoints were hit.
                cache_skips[frame_cache_key] = 2
                return None if event == 'call' else NO_FTRACE

            # IFDEF CYTHON
            # frame.f_trace = SafeCallWrapper(ret)  # Make sure we keep the returned tracer.
            # ELSE
            frame.f_trace = ret  # Make sure we keep the returned tracer.
            # ENDIF
            return ret

        except SystemExit:
            return None if event == 'call' else NO_FTRACE

        except Exception:
            if py_db.pydb_disposed:
                return None if event == 'call' else NO_FTRACE  # Don't log errors when we're shutting down.
            # Log it
            try:
                if pydev_log_exception is not None:
                    # This can actually happen during the interpreter shutdown in Python 2.7
                    pydev_log_exception()
            except:
                # Error logging? We're really in the interpreter shutdown...
                # (https://github.com/fabioz/PyDev.Debugger/issues/8)
                pass
            return None if event == 'call' else NO_FTRACE
        finally:
            additional_info.is_tracing -= 1


if USE_CUSTOM_SYS_CURRENT_FRAMES_MAP:
    # This is far from ideal, as we'll leak frames (we'll always have the last created frame, not really
    # the last topmost frame saved -- this should be Ok for our usage, but it may leak frames and things
    # may live longer... as IronPython is garbage-collected, things should live longer anyways, so, it
    # shouldn't be an issue as big as it's in CPython -- it may still be annoying, but this should
    # be a reasonable workaround until IronPython itself is able to provide that functionality).
    #
    # See: https://github.com/IronLanguages/main/issues/1630
    from _pydevd_bundle.pydevd_constants import constructed_tid_to_last_frame

    _original_call = ThreadTracer.__call__

    def __call__(self, frame, event, arg):
        constructed_tid_to_last_frame[self._args[1].ident] = frame
        return _original_call(self, frame, event, arg)

    ThreadTracer.__call__ = __call__
