from __future__ import print_function
from _pydev_bundle._pydev_saved_modules import threading, thread
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
import dis
import sys
from _pydevd_frame_eval.pydevd_frame_tracing import update_globals_dict, dummy_tracing_holder
from _pydevd_frame_eval.pydevd_modify_bytecode import DebugHelper, insert_pydevd_breaks
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_trace_dispatch import fix_top_level_trace_and_get_trace_func

from _pydevd_bundle.pydevd_additional_thread_info import _set_additional_thread_info_lock
from _pydevd_bundle.pydevd_cython cimport PyDBAdditionalThreadInfo
from pydevd_tracing import SetTrace

_get_ident = threading.get_ident  # Note this is py3 only, if py2 needed to be supported, _get_ident would be needed.
_thread_local_info = threading.local()
_thread_active = threading._active

def clear_thread_local_info():
    global _thread_local_info
    _thread_local_info = threading.local()


cdef class ThreadInfo:

    cdef public PyDBAdditionalThreadInfo additional_info
    cdef public bint is_pydevd_thread
    cdef public int inside_frame_eval
    cdef public bint fully_initialized
    cdef public object thread_trace_func
    cdef bint _can_create_dummy_thread

    # Note: whenever get_func_code_info is called, this value is reset (we're using
    # it as a thread-local value info).
    # If True the debugger should not go into trace mode even if the new
    # code for a function is None and there are breakpoints.
    cdef public bint force_stay_in_untraced_mode

    cdef initialize(self, PyFrameObject * frame_obj):
        # Places that create a ThreadInfo should verify that
        # a current Python frame is being executed!
        assert frame_obj != NULL

        self.additional_info = None
        self.is_pydevd_thread = False
        self.inside_frame_eval = 0
        self.fully_initialized = False
        self.thread_trace_func = None

        # Get the root (if it's not a Thread initialized from the threading
        # module, create the dummy thread entry so that we can debug it --
        # otherwise, we have to wait for the threading module itself to
        # create the Thread entry).
        while frame_obj.f_back != NULL:
            frame_obj = frame_obj.f_back

        basename = <str> frame_obj.f_code.co_filename
        i = basename.rfind('/')
        j = basename.rfind('\\')
        if j > i:
            i = j
        if i >= 0:
            basename = basename[i + 1:]
        # remove ext
        i = basename.rfind('.')
        if i >= 0:
            basename = basename[:i]

        co_name = <str> frame_obj.f_code.co_name

        # In these cases we cannot create a dummy thread (an actual
        # thread will be created later or tracing will already be set).
        if basename == 'threading' and co_name in ('__bootstrap', '_bootstrap', '__bootstrap_inner', '_bootstrap_inner'):
            self._can_create_dummy_thread = False
        elif basename == 'pydev_monkey' and co_name == '__call__':
            self._can_create_dummy_thread = False
        elif basename == 'pydevd' and co_name in ('run', 'main', '_exec'):
            self._can_create_dummy_thread = False
        elif basename == 'pydevd_tracing':
            self._can_create_dummy_thread = False
        else:
            self._can_create_dummy_thread = True

        # print('Can create dummy thread for thread started in: %s %s' % (basename, co_name))

    cdef initialize_if_possible(self):
        # Don't call threading.currentThread because if we're too early in the process
        # we may create a dummy thread.
        self.inside_frame_eval += 1

        try:
            thread_ident = _get_ident()
            t = _thread_active.get(thread_ident)
            if t is None:
                if self._can_create_dummy_thread:
                    # Initialize the dummy thread and set the tracing (both are needed to
                    # actually stop on breakpoints).
                    t = threading.current_thread()
                    SetTrace(dummy_trace_dispatch)
                else:
                    return  # Cannot initialize until thread becomes active.

            if getattr(t, 'is_pydev_daemon_thread', False):
                self.is_pydevd_thread = True
                self.fully_initialized = True
            else:
                try:
                    additional_info = t.additional_info
                    if additional_info is None:
                        raise AttributeError()
                except:
                    with _set_additional_thread_info_lock:
                        # If it's not there, set it within a lock to avoid any racing
                        # conditions.
                        additional_info = getattr(thread, 'additional_info', None)
                        if additional_info is None:
                            additional_info = PyDBAdditionalThreadInfo()
                        t.additional_info = additional_info
                self.additional_info = additional_info
                self.fully_initialized = True
        finally:
            self.inside_frame_eval -= 1


cdef class FuncCodeInfo:

    cdef public str co_filename
    cdef public str co_name
    cdef public str canonical_normalized_filename
    cdef bint always_skip_code
    cdef public bint breakpoint_found
    cdef public object new_code

    # When breakpoints_mtime != PyDb.mtime the validity of breakpoints have
    # to be re-evaluated (if invalid a new FuncCodeInfo must be created and
    # tracing can't be disabled for the related frames).
    cdef public int breakpoints_mtime

    def __init__(self):
        self.co_filename = ''
        self.canonical_normalized_filename = ''
        self.always_skip_code = False

        # If breakpoints are found but new_code is None,
        # this means we weren't able to actually add the code
        # where needed, so, fallback to tracing.
        self.breakpoint_found = False
        self.new_code = None
        self.breakpoints_mtime = -1


def dummy_trace_dispatch(frame, str event, arg):
    if event == 'call':
        if frame.f_trace is not None:
            return frame.f_trace(frame, event, arg)
    return None


def get_thread_info_py() -> ThreadInfo:
    return get_thread_info(PyEval_GetFrame())


cdef ThreadInfo get_thread_info(PyFrameObject * frame_obj):
    '''
    Provides thread-related info.

    May return None if the thread is still not active.
    '''
    cdef ThreadInfo thread_info
    try:
        # Note: changing to a `dict[thread.ident] = thread_info` had almost no
        # effect in the performance.
        thread_info = _thread_local_info.thread_info
    except:
        if frame_obj == NULL:
            return None
        thread_info = ThreadInfo()
        thread_info.initialize(frame_obj)
        thread_info.inside_frame_eval += 1
        try:
            _thread_local_info.thread_info = thread_info

            # Note: _code_extra_index is not actually thread-related,
            # but this is a good point to initialize it.
            global _code_extra_index
            if _code_extra_index == -1:
                _code_extra_index = <int> _PyEval_RequestCodeExtraIndex(release_co_extra)

            thread_info.initialize_if_possible()
        finally:
            thread_info.inside_frame_eval -= 1

    return thread_info


def decref_py(obj):
    '''
    Helper to be called from Python.
    '''
    Py_DECREF(obj)


def get_func_code_info_py(thread_info, frame, code_obj) -> FuncCodeInfo:
    '''
    Helper to be called from Python.
    '''
    return get_func_code_info(<ThreadInfo> thread_info, <PyFrameObject *> frame, <PyCodeObject *> code_obj)


cdef int _code_extra_index = -1

cdef FuncCodeInfo get_func_code_info(ThreadInfo thread_info, PyFrameObject * frame_obj, PyCodeObject * code_obj):
    '''
    Provides code-object related info.

    Stores the gathered info in a cache in the code object itself. Note that
    multiple threads can get the same info.

    get_thread_info() *must* be called at least once before get_func_code_info()
    to initialize _code_extra_index.

    '''
    # f_code = <object> code_obj
    # DEBUG = f_code.co_filename.endswith('_debugger_case_multiprocessing.py')
    # if DEBUG:
    #     print('get_func_code_info', f_code.co_name, f_code.co_filename)

    cdef object main_debugger = GlobalDebuggerHolder.global_dbg
    thread_info.force_stay_in_untraced_mode = False  # This is an output value of the function.

    cdef PyObject * extra
    _PyCode_GetExtra(<PyObject *> code_obj, _code_extra_index, & extra)
    if extra is not NULL:
        extra_obj = <PyObject *> extra
        if extra_obj is not NULL:
            func_code_info_obj = <FuncCodeInfo> extra_obj
            if func_code_info_obj.breakpoints_mtime == main_debugger.mtime:
                # if DEBUG:
                #     print('get_func_code_info: matched mtime', f_code.co_name, f_code.co_filename)

                return func_code_info_obj

    cdef str co_filename = <str> code_obj.co_filename
    cdef str co_name = <str> code_obj.co_name
    cdef dict cache_file_type
    cdef tuple cache_file_type_key

    func_code_info = FuncCodeInfo()
    func_code_info.breakpoints_mtime = main_debugger.mtime

    func_code_info.co_filename = co_filename
    func_code_info.co_name = co_name

    if not func_code_info.always_skip_code:
        try:
            abs_path_real_path_and_base = NORM_PATHS_AND_BASE_CONTAINER[co_filename]
        except:
            abs_path_real_path_and_base = get_abs_path_real_path_and_base_from_frame(<object>frame_obj)

        func_code_info.canonical_normalized_filename = abs_path_real_path_and_base[1]

        cache_file_type = main_debugger.get_cache_file_type()
        # Note: this cache key must be the same from PyDB.get_file_type() -- see it for comments
        # on the cache.
        cache_file_type_key = (frame_obj.f_code.co_firstlineno, abs_path_real_path_and_base[0], <object>frame_obj.f_code)
        try:
            file_type = cache_file_type[cache_file_type_key]  # Make it faster
        except:
            file_type = main_debugger.get_file_type(<object>frame_obj, abs_path_real_path_and_base)  # we don't want to debug anything related to pydevd

        if file_type is not None:
            func_code_info.always_skip_code = True

    if not func_code_info.always_skip_code:
        if main_debugger is not None:

            breakpoints: dict = main_debugger.breakpoints.get(func_code_info.canonical_normalized_filename)
            function_breakpoint: object = main_debugger.function_breakpoint_name_to_breakpoint.get(func_code_info.co_name)
            # print('\n---')
            # print(main_debugger.breakpoints)
            # print(func_code_info.canonical_normalized_filename)
            # print(main_debugger.breakpoints.get(func_code_info.canonical_normalized_filename))
            code_obj_py: object = <object> code_obj
            cached_code_obj_info: object = _cache.get(code_obj_py)
            if cached_code_obj_info:
                # The cache is for new code objects, so, in this case it's already
                # using the new code and we can't change it as this is a generator!
                # There's still a catch though: even though we don't replace the code,
                # we may not want to go into tracing mode (as would usually happen
                # when the new_code is None).
                func_code_info.new_code = None
                breakpoint_found, thread_info.force_stay_in_untraced_mode = \
                    cached_code_obj_info.compute_force_stay_in_untraced_mode(breakpoints)
                func_code_info.breakpoint_found = breakpoint_found

            elif function_breakpoint:
                # Go directly into tracing mode
                func_code_info.breakpoint_found = True
                func_code_info.new_code = None
                
            elif breakpoints:
                # if DEBUG:
                #    print('found breakpoints', code_obj_py.co_name, breakpoints)

                # Note: new_code can be None if unable to generate.
                # It should automatically put the new code object in the cache.
                breakpoint_found, func_code_info.new_code = generate_code_with_breakpoints(code_obj_py, breakpoints)
                func_code_info.breakpoint_found = breakpoint_found

    Py_INCREF(func_code_info)
    _PyCode_SetExtra(<PyObject *> code_obj, _code_extra_index, <PyObject *> func_code_info)

    return func_code_info


cdef class _CodeLineInfo:

    cdef public dict line_to_offset
    cdef public int first_line
    cdef public int last_line

    def __init__(self, dict line_to_offset,  int first_line,  int last_line):
        self.line_to_offset = line_to_offset
        self.first_line = first_line
        self.last_line = last_line


# Note: this method has a version in pure-python too.
def _get_code_line_info(code_obj):
    line_to_offset: dict = {}
    first_line: int = None
    last_line: int = None

    cdef int offset
    cdef int line

    for offset, line in dis.findlinestarts(code_obj):
        line_to_offset[line] = offset

    if line_to_offset:
        first_line = min(line_to_offset)
        last_line = max(line_to_offset)
    return _CodeLineInfo(line_to_offset, first_line, last_line)


# Note: this is a cache where the key is the code objects we create ourselves so that
# we always return the same code object for generators.
# (so, we don't have a cache from the old code to the new info -- that's actually
# handled by the cython side in `FuncCodeInfo get_func_code_info` by providing the
# same code info if the debugger mtime is still the same).
_cache: dict = {}

def get_cached_code_obj_info_py(code_obj_py):
    '''
    :return _CacheValue:
    :note: on cython use _cache.get(code_obj_py) directly.
    '''
    return _cache.get(code_obj_py)


cdef class _CacheValue(object):

    cdef public object code_obj_py
    cdef public _CodeLineInfo code_line_info
    cdef public set breakpoints_hit_at_lines
    cdef public set code_lines_as_set

    def __init__(self, object code_obj_py, _CodeLineInfo code_line_info, set breakpoints_hit_at_lines):
        '''
        :param code_obj_py:
        :param _CodeLineInfo code_line_info:
        :param set[int] breakpoints_hit_at_lines:
        '''
        self.code_obj_py = code_obj_py
        self.code_line_info = code_line_info
        self.breakpoints_hit_at_lines = breakpoints_hit_at_lines
        self.code_lines_as_set = set(code_line_info.line_to_offset)

    cpdef compute_force_stay_in_untraced_mode(self, breakpoints):
        '''
        :param breakpoints:
            set(breakpoint_lines) or dict(breakpoint_line->breakpoint info)
        :return tuple(breakpoint_found, force_stay_in_untraced_mode)
        '''
        cdef bint force_stay_in_untraced_mode
        cdef bint breakpoint_found
        cdef set target_breakpoints

        force_stay_in_untraced_mode = False

        target_breakpoints = self.code_lines_as_set.intersection(breakpoints)
        breakpoint_found = bool(target_breakpoints)

        if not breakpoint_found:
            force_stay_in_untraced_mode = True
        else:
            force_stay_in_untraced_mode = self.breakpoints_hit_at_lines.issuperset(set(breakpoints))

        return breakpoint_found, force_stay_in_untraced_mode

def generate_code_with_breakpoints_py(object code_obj_py, dict breakpoints):
    return generate_code_with_breakpoints(code_obj_py, breakpoints)

# DEBUG = True
# debug_helper = DebugHelper()

cdef generate_code_with_breakpoints(object code_obj_py, dict breakpoints):
    '''
    :param breakpoints:
        dict where the keys are the breakpoint lines.
    :return tuple(breakpoint_found, new_code)
    '''
    # The cache is needed for generator functions, because after each yield a new frame
    # is created but the former code object is used (so, check if code_to_modify is
    # already there and if not cache based on the new code generated).

    cdef bint success
    cdef int breakpoint_line
    cdef bint breakpoint_found
    cdef _CacheValue cache_value
    cdef set breakpoints_hit_at_lines
    cdef dict line_to_offset

    assert code_obj_py not in _cache, 'If a code object is cached, that same code object must be reused.'

#     if DEBUG:
#         initial_code_obj_py = code_obj_py

    code_line_info = _get_code_line_info(code_obj_py)

    success = True

    breakpoints_hit_at_lines = set()
    line_to_offset = code_line_info.line_to_offset

    for breakpoint_line in breakpoints:
        if breakpoint_line in line_to_offset:
            breakpoints_hit_at_lines.add(breakpoint_line)

    if breakpoints_hit_at_lines:
        success, new_code = insert_pydevd_breaks(
            code_obj_py,
            breakpoints_hit_at_lines,
            code_line_info
        )

        if not success:
            code_obj_py = None
        else:
            code_obj_py = new_code

    breakpoint_found = bool(breakpoints_hit_at_lines)
    if breakpoint_found and success:
#         if DEBUG:
#             op_number = debug_helper.write_dis(
#                 'inserting code, breaks at: %s' % (list(breakpoints),),
#                 initial_code_obj_py
#             )
#
#             debug_helper.write_dis(
#                 'after inserting code, breaks at: %s' % (list(breakpoints,)),
#                 code_obj_py,
#                 op_number=op_number,
#             )

        cache_value = _CacheValue(code_obj_py, code_line_info, breakpoints_hit_at_lines)
        _cache[code_obj_py] = cache_value

    return breakpoint_found, code_obj_py

import sys

cdef bint IS_PY_39_OWNARDS = sys.version_info[:2] >= (3, 9)

def frame_eval_func():
    cdef PyThreadState *state = PyThreadState_Get()
    if IS_PY_39_OWNARDS:
        state.interp.eval_frame = <_PyFrameEvalFunction *> get_bytecode_while_frame_eval_39
    else:
        state.interp.eval_frame = <_PyFrameEvalFunction *> get_bytecode_while_frame_eval_38
    dummy_tracing_holder.set_trace_func(dummy_trace_dispatch)


def stop_frame_eval():
    cdef PyThreadState *state = PyThreadState_Get()
    state.interp.eval_frame = _PyEval_EvalFrameDefault

# During the build we'll generate 2 versions of the code below so that we're compatible with
# Python 3.9, which receives a "PyThreadState* tstate" as the first parameter and Python 3.6-3.8
# which doesn't.
### TEMPLATE_START
cdef PyObject * get_bytecode_while_frame_eval(PyFrameObject * frame_obj, int exc):
    '''
    This function makes the actual evaluation and changes the bytecode to a version
    where programmatic breakpoints are added.
    '''
    if GlobalDebuggerHolder is None or _thread_local_info is None or exc:
        # Sometimes during process shutdown these global variables become None
        return CALL_EvalFrameDefault

    # co_filename: str = <str>frame_obj.f_code.co_filename
    # if co_filename.endswith('threading.py'):
    #     return CALL_EvalFrameDefault

    cdef ThreadInfo thread_info
    cdef int STATE_SUSPEND = 2
    cdef int CMD_STEP_INTO = 107
    cdef int CMD_STEP_OVER = 108
    cdef int CMD_STEP_OVER_MY_CODE = 159
    cdef int CMD_STEP_INTO_MY_CODE = 144
    cdef int CMD_STEP_INTO_COROUTINE = 206
    cdef int CMD_SMART_STEP_INTO = 128
    cdef bint can_skip = True
    try:
        thread_info = _thread_local_info.thread_info
    except:
        thread_info = get_thread_info(frame_obj)
        if thread_info is None:
            return CALL_EvalFrameDefault

    if thread_info.inside_frame_eval:
        return CALL_EvalFrameDefault

    if not thread_info.fully_initialized:
        thread_info.initialize_if_possible()
        if not thread_info.fully_initialized:
            return CALL_EvalFrameDefault

    # Can only get additional_info when fully initialized.
    cdef PyDBAdditionalThreadInfo additional_info = thread_info.additional_info
    if thread_info.is_pydevd_thread or additional_info.is_tracing:
        # Make sure that we don't trace pydevd threads or inside our own calls.
        return CALL_EvalFrameDefault

    # frame = <object> frame_obj
    # DEBUG = frame.f_code.co_filename.endswith('_debugger_case_tracing.py')
    # if DEBUG:
    #     print('get_bytecode_while_frame_eval', frame.f_lineno, frame.f_code.co_name, frame.f_code.co_filename)

    thread_info.inside_frame_eval += 1
    additional_info.is_tracing = True
    try:
        main_debugger: object = GlobalDebuggerHolder.global_dbg
        if main_debugger is None:
            return CALL_EvalFrameDefault
        frame = <object> frame_obj

        if thread_info.thread_trace_func is None:
            trace_func, apply_to_global = fix_top_level_trace_and_get_trace_func(main_debugger, frame)
            if apply_to_global:
                thread_info.thread_trace_func = trace_func

        if additional_info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE, CMD_SMART_STEP_INTO) or \
                main_debugger.break_on_caught_exceptions or \
                main_debugger.break_on_user_uncaught_exceptions or \
                main_debugger.has_plugin_exception_breaks or \
                main_debugger.signature_factory or \
                additional_info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE) and main_debugger.show_return_values and frame.f_back is additional_info.pydev_step_stop:

            # if DEBUG:
            #     print('get_bytecode_while_frame_eval enabled trace')
            if thread_info.thread_trace_func is not None:
                frame.f_trace = thread_info.thread_trace_func
            else:
                frame.f_trace = <object> main_debugger.trace_dispatch
        else:
            func_code_info: FuncCodeInfo = get_func_code_info(thread_info, frame_obj, frame_obj.f_code)
            # if DEBUG:
            #     print('get_bytecode_while_frame_eval always skip', func_code_info.always_skip_code)
            if not func_code_info.always_skip_code:

                if main_debugger.has_plugin_line_breaks or main_debugger.has_plugin_exception_breaks:
                    can_skip = main_debugger.plugin.can_skip(main_debugger, <object> frame_obj)

                    if not can_skip:
                        # if DEBUG:
                        #     print('get_bytecode_while_frame_eval not can_skip')
                        if thread_info.thread_trace_func is not None:
                            frame.f_trace = thread_info.thread_trace_func
                        else:
                            frame.f_trace = <object> main_debugger.trace_dispatch

                if can_skip and func_code_info.breakpoint_found:
                    # if DEBUG:
                    #     print('get_bytecode_while_frame_eval new_code', func_code_info.new_code)
                    if not thread_info.force_stay_in_untraced_mode:
                        # If breakpoints are found but new_code is None,
                        # this means we weren't able to actually add the code
                        # where needed, so, fallback to tracing.
                        if func_code_info.new_code is None:
                            if thread_info.thread_trace_func is not None:
                                frame.f_trace = thread_info.thread_trace_func
                            else:
                                frame.f_trace = <object> main_debugger.trace_dispatch
                        else:
                            # print('Using frame eval break for', <object> frame_obj.f_code.co_name)
                            update_globals_dict(<object> frame_obj.f_globals)
                            Py_INCREF(func_code_info.new_code)
                            old = <object> frame_obj.f_code
                            frame_obj.f_code = <PyCodeObject *> func_code_info.new_code
                            Py_DECREF(old)
                    else:
                        # When we're forcing to stay in traced mode we need to
                        # update the globals dict (because this means that we're reusing
                        # a previous code which had breakpoints added in a new frame).
                        update_globals_dict(<object> frame_obj.f_globals)

    finally:
        thread_info.inside_frame_eval -= 1
        additional_info.is_tracing = False

    return CALL_EvalFrameDefault
### TEMPLATE_END
