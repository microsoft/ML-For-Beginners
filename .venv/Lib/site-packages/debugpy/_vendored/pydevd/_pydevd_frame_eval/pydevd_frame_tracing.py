import sys

from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_comm import get_global_debugger
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info


class DummyTracingHolder:
    dummy_trace_func = None

    def set_trace_func(self, trace_func):
        self.dummy_trace_func = trace_func


dummy_tracing_holder = DummyTracingHolder()


def update_globals_dict(globals_dict):
    new_globals = {'_pydev_stop_at_break': _pydev_stop_at_break}
    globals_dict.update(new_globals)


def _get_line_for_frame(frame):
    # it's absolutely necessary to reset tracing function for frame in order to get the real line number
    tracing_func = frame.f_trace
    frame.f_trace = None
    line = frame.f_lineno
    frame.f_trace = tracing_func
    return line


def _pydev_stop_at_break(line):
    frame = sys._getframe(1)
    # print('pydevd SET TRACING at ', line, 'curr line', frame.f_lineno)
    t = threading.current_thread()
    try:
        additional_info = t.additional_info
    except:
        additional_info = set_additional_thread_info(t)

    if additional_info.is_tracing:
        return

    additional_info.is_tracing += 1
    try:
        py_db = get_global_debugger()
        if py_db is None:
            return

        pydev_log.debug("Setting f_trace due to frame eval mode in file: %s on line %s", frame.f_code.co_filename, line)
        additional_info.trace_suspend_type = 'frame_eval'

        pydevd_frame_eval_cython_wrapper = sys.modules['_pydevd_frame_eval.pydevd_frame_eval_cython_wrapper']
        thread_info = pydevd_frame_eval_cython_wrapper.get_thread_info_py()
        if thread_info.thread_trace_func is not None:
            frame.f_trace = thread_info.thread_trace_func
        else:
            frame.f_trace = py_db.get_thread_local_trace_func()
    finally:
        additional_info.is_tracing -= 1


def _pydev_needs_stop_at_break(line):
    '''
    We separate the functionality into 2 functions so that we can generate a bytecode which
    generates a spurious line change so that we can do:

    if _pydev_needs_stop_at_break():
        # Set line to line -1
        _pydev_stop_at_break()
        # then, proceed to go to the current line
        # (which will then trigger a line event).
    '''
    t = threading.current_thread()
    try:
        additional_info = t.additional_info
    except:
        additional_info = set_additional_thread_info(t)

    if additional_info.is_tracing:
        return False

    additional_info.is_tracing += 1
    try:
        frame = sys._getframe(1)
        # print('pydev needs stop at break?', line, 'curr line', frame.f_lineno, 'curr trace', frame.f_trace)
        if frame.f_trace is not None:
            # i.e.: this frame is already being traced, thus, we don't need to use programmatic breakpoints.
            return False

        py_db = get_global_debugger()
        if py_db is None:
            return False

        try:
            abs_path_real_path_and_base = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
        except:
            abs_path_real_path_and_base = get_abs_path_real_path_and_base_from_frame(frame)
        canonical_normalized_filename = abs_path_real_path_and_base[1]

        try:
            python_breakpoint = py_db.breakpoints[canonical_normalized_filename][line]
        except:
            # print("Couldn't find breakpoint in the file %s on line %s" % (frame.f_code.co_filename, line))
            # Could be KeyError if line is not there or TypeError if breakpoints_for_file is None.
            # Note: using catch-all exception for performance reasons (if the user adds a breakpoint
            # and then removes it after hitting it once, this method added for the programmatic
            # breakpoint will keep on being called and one of those exceptions will always be raised
            # here).
            return False

        if python_breakpoint:
            # print('YES')
            return True

    finally:
        additional_info.is_tracing -= 1

    return False

