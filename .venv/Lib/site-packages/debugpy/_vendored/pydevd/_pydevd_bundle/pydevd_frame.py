import linecache
import os.path
import re

from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
    EXCEPTION_TYPE_HANDLED, EXCEPTION_TYPE_USER_UNHANDLED, PYDEVD_IPYTHON_CONTEXT)
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
try:
    from _pydevd_bundle.pydevd_bytecode_utils import get_smart_step_into_variant_from_frame_offset
except ImportError:

    def get_smart_step_into_variant_from_frame_offset(*args, **kwargs):
        return None

# IFDEF CYTHON
# cython_inline_constant: CMD_STEP_INTO = 107
# cython_inline_constant: CMD_STEP_INTO_MY_CODE = 144
# cython_inline_constant: CMD_STEP_RETURN = 109
# cython_inline_constant: CMD_STEP_RETURN_MY_CODE = 160
# cython_inline_constant: CMD_STEP_OVER = 108
# cython_inline_constant: CMD_STEP_OVER_MY_CODE = 159
# cython_inline_constant: CMD_STEP_CAUGHT_EXCEPTION = 137
# cython_inline_constant: CMD_SET_BREAK = 111
# cython_inline_constant: CMD_SMART_STEP_INTO = 128
# cython_inline_constant: CMD_STEP_INTO_COROUTINE = 206
# cython_inline_constant: STATE_RUN = 1
# cython_inline_constant: STATE_SUSPEND = 2
# ELSE
# Note: those are now inlined on cython.
CMD_STEP_INTO = 107
CMD_STEP_INTO_MY_CODE = 144
CMD_STEP_RETURN = 109
CMD_STEP_RETURN_MY_CODE = 160
CMD_STEP_OVER = 108
CMD_STEP_OVER_MY_CODE = 159
CMD_STEP_CAUGHT_EXCEPTION = 137
CMD_SET_BREAK = 111
CMD_SMART_STEP_INTO = 128
CMD_STEP_INTO_COROUTINE = 206
STATE_RUN = 1
STATE_SUSPEND = 2
# ENDIF

basename = os.path.basename

IGNORE_EXCEPTION_TAG = re.compile('[^#]*#.*@IgnoreException')
DEBUG_START = ('pydevd.py', 'run')
DEBUG_START_PY3K = ('_pydev_execfile.py', 'execfile')
TRACE_PROPERTY = 'pydevd_traceproperty.py'

import dis

try:
    StopAsyncIteration
except NameError:
    StopAsyncIteration = StopIteration


# IFDEF CYTHON
# cdef is_unhandled_exception(container_obj, py_db, frame, int last_raise_line, set raise_lines):
# ELSE
def is_unhandled_exception(container_obj, py_db, frame, last_raise_line, raise_lines):
# ENDIF
    if frame.f_lineno in raise_lines:
        return True

    else:
        try_except_infos = container_obj.try_except_infos
        if try_except_infos is None:
            container_obj.try_except_infos = try_except_infos = py_db.collect_try_except_info(frame.f_code)

        if not try_except_infos:
            # Consider the last exception as unhandled because there's no try..except in it.
            return True
        else:
            # Now, consider only the try..except for the raise
            valid_try_except_infos = []
            for try_except_info in try_except_infos:
                if try_except_info.is_line_in_try_block(last_raise_line):
                    valid_try_except_infos.append(try_except_info)

            if not valid_try_except_infos:
                return True

            else:
                # Note: check all, not only the "valid" ones to cover the case
                # in "tests_python.test_tracing_on_top_level.raise_unhandled10"
                # where one try..except is inside the other with only a raise
                # and it's gotten in the except line.
                for try_except_info in try_except_infos:
                    if try_except_info.is_line_in_except_block(frame.f_lineno):
                        if (
                                frame.f_lineno == try_except_info.except_line or
                                frame.f_lineno in try_except_info.raise_lines_in_except
                            ):
                            # In a raise inside a try..except block or some except which doesn't
                            # match the raised exception.
                            return True
    return False


# IFDEF CYTHON
# cdef class _TryExceptContainerObj:
#     cdef public list try_except_infos;
#     def __init__(self):
#         self.try_except_infos = None
# ELSE
class _TryExceptContainerObj(object):
    '''
    A dumb container object just to containe the try..except info when needed. Meant to be
    persisent among multiple PyDBFrames to the same code object.
    '''
    try_except_infos = None
# ENDIF


#=======================================================================================================================
# PyDBFrame
#=======================================================================================================================
# IFDEF CYTHON
# cdef class PyDBFrame:
# ELSE
class PyDBFrame:
    '''This makes the tracing for a given frame, so, the trace_dispatch
    is used initially when we enter into a new context ('call') and then
    is reused for the entire context.
    '''
# ENDIF

    # Note: class (and not instance) attributes.

    # Same thing in the main debugger but only considering the file contents, while the one in the main debugger
    # considers the user input (so, the actual result must be a join of both).
    filename_to_lines_where_exceptions_are_ignored = {}
    filename_to_stat_info = {}

    # IFDEF CYTHON
    # cdef tuple _args
    # cdef int should_skip
    # cdef object exc_info
    # def __init__(self, tuple args):
        # self._args = args # In the cython version we don't need to pass the frame
        # self.should_skip = -1  # On cythonized version, put in instance.
        # self.exc_info = ()
    # ELSE
    should_skip = -1  # Default value in class (put in instance on set).
    exc_info = ()  # Default value in class (put in instance on set).

    def __init__(self, args):
        # args = main_debugger, abs_path_canonical_path_and_base, base, info, t, frame
        # yeap, much faster than putting in self and then getting it from self later on
        self._args = args
    # ENDIF

    def set_suspend(self, *args, **kwargs):
        self._args[0].set_suspend(*args, **kwargs)

    def do_wait_suspend(self, *args, **kwargs):
        self._args[0].do_wait_suspend(*args, **kwargs)

    # IFDEF CYTHON
    # def trace_exception(self, frame, str event, arg):
    #     cdef bint should_stop;
    #     cdef tuple exc_info;
    # ELSE
    def trace_exception(self, frame, event, arg):
    # ENDIF
        if event == 'exception':
            should_stop, frame = self._should_stop_on_exception(frame, event, arg)

            if should_stop:
                if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                    return self.trace_dispatch

        elif event == 'return':
            exc_info = self.exc_info
            if exc_info and arg is None:
                frame_skips_cache, frame_cache_key = self._args[4], self._args[5]
                custom_key = (frame_cache_key, 'try_exc_info')
                container_obj = frame_skips_cache.get(custom_key)
                if container_obj is None:
                    container_obj = frame_skips_cache[custom_key] = _TryExceptContainerObj()
                if is_unhandled_exception(container_obj, self._args[0], frame, exc_info[1], exc_info[2]) and \
                        self.handle_user_exception(frame):
                    return self.trace_dispatch

        return self.trace_exception

    # IFDEF CYTHON
    # cdef _should_stop_on_exception(self, frame, str event, arg):
    #     cdef PyDBAdditionalThreadInfo info;
    #     cdef bint should_stop;
    #     cdef bint was_just_raised;
    #     cdef list check_excs;
    # ELSE
    def _should_stop_on_exception(self, frame, event, arg):
    # ENDIF

        # main_debugger, _filename, info, _thread = self._args
        main_debugger = self._args[0]
        info = self._args[2]
        should_stop = False

        # STATE_SUSPEND = 2
        if info.pydev_state != 2:  # and breakpoint is not None:
            exception, value, trace = arg

            if trace is not None and hasattr(trace, 'tb_next'):
                # on jython trace is None on the first event and it may not have a tb_next.

                should_stop = False
                exception_breakpoint = None
                try:
                    if main_debugger.plugin is not None:
                        result = main_debugger.plugin.exception_break(main_debugger, self, frame, self._args, arg)
                        if result:
                            should_stop, frame = result
                except:
                    pydev_log.exception()

                if not should_stop:
                    # Apply checks that don't need the exception breakpoint (where we shouldn't ever stop).
                    if exception == SystemExit and main_debugger.ignore_system_exit_code(value):
                        pass

                    elif exception in (GeneratorExit, StopIteration, StopAsyncIteration):
                        # These exceptions are control-flow related (they work as a generator
                        # pause), so, we shouldn't stop on them.
                        pass

                    elif ignore_exception_trace(trace):
                        pass

                    else:
                        was_just_raised = trace.tb_next is None

                        # It was not handled by any plugin, lets check exception breakpoints.
                        check_excs = []

                        # Note: check user unhandled before regular exceptions.
                        exc_break_user = main_debugger.get_exception_breakpoint(
                            exception, main_debugger.break_on_user_uncaught_exceptions)
                        if exc_break_user is not None:
                            check_excs.append((exc_break_user, True))

                        exc_break_caught = main_debugger.get_exception_breakpoint(
                            exception, main_debugger.break_on_caught_exceptions)
                        if exc_break_caught is not None:
                            check_excs.append((exc_break_caught, False))

                        for exc_break, is_user_uncaught in check_excs:
                            # Initially mark that it should stop and then go into exclusions.
                            should_stop = True

                            if main_debugger.exclude_exception_by_filter(exc_break, trace):
                                pydev_log.debug("Ignore exception %s in library %s -- (%s)" % (exception, frame.f_code.co_filename, frame.f_code.co_name))
                                should_stop = False

                            elif exc_break.condition is not None and \
                                    not main_debugger.handle_breakpoint_condition(info, exc_break, frame):
                                should_stop = False

                            elif is_user_uncaught:
                                # Note: we don't stop here, we just collect the exc_info to use later on...
                                should_stop = False
                                if not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, True) \
                                        and (frame.f_back is None or main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)):
                                    # User uncaught means that we're currently in user code but the code
                                    # up the stack is library code.
                                    exc_info = self.exc_info
                                    if not exc_info:
                                        exc_info = (arg, frame.f_lineno, set([frame.f_lineno]))
                                    else:
                                        lines = exc_info[2]
                                        lines.add(frame.f_lineno)
                                        exc_info = (arg, frame.f_lineno, lines)
                                    self.exc_info = exc_info
                            else:
                                # I.e.: these are only checked if we're not dealing with user uncaught exceptions.
                                if exc_break.notify_on_first_raise_only and main_debugger.skip_on_exceptions_thrown_in_same_context \
                                        and not was_just_raised and not just_raised(trace.tb_next):
                                    # In this case we never stop if it was just raised, so, to know if it was the first we
                                    # need to check if we're in the 2nd method.
                                    should_stop = False  # I.e.: we stop only when we're at the caller of a method that throws an exception

                                elif exc_break.notify_on_first_raise_only and not main_debugger.skip_on_exceptions_thrown_in_same_context \
                                        and not was_just_raised:
                                    should_stop = False  # I.e.: we stop only when it was just raised

                                elif was_just_raised and main_debugger.skip_on_exceptions_thrown_in_same_context:
                                    # Option: Don't break if an exception is caught in the same function from which it is thrown
                                    should_stop = False

                            if should_stop:
                                exception_breakpoint = exc_break
                                try:
                                    info.pydev_message = exc_break.qname
                                except:
                                    info.pydev_message = exc_break.qname.encode('utf-8')
                                break

                if should_stop:
                    # Always add exception to frame (must remove later after we proceed).
                    add_exception_to_frame(frame, (exception, value, trace))

                    if exception_breakpoint is not None and exception_breakpoint.expression is not None:
                        main_debugger.handle_breakpoint_expression(exception_breakpoint, info, frame)

        return should_stop, frame

    def handle_user_exception(self, frame):
        exc_info = self.exc_info
        if exc_info:
            return self._handle_exception(frame, 'exception', exc_info[0], EXCEPTION_TYPE_USER_UNHANDLED)
        return False

    # IFDEF CYTHON
    # cdef _handle_exception(self, frame, str event, arg, str exception_type):
    #     cdef bint stopped;
    #     cdef tuple abs_real_path_and_base;
    #     cdef str absolute_filename;
    #     cdef str canonical_normalized_filename;
    #     cdef dict filename_to_lines_where_exceptions_are_ignored;
    #     cdef dict lines_ignored;
    #     cdef dict frame_id_to_frame;
    #     cdef dict merged;
    #     cdef object trace_obj;
    #     cdef object main_debugger;
    # ELSE
    def _handle_exception(self, frame, event, arg, exception_type):
    # ENDIF
        stopped = False
        try:
            # print('_handle_exception', frame.f_lineno, frame.f_code.co_name)

            # We have 3 things in arg: exception type, description, traceback object
            trace_obj = arg[2]
            main_debugger = self._args[0]

            initial_trace_obj = trace_obj
            if trace_obj.tb_next is None and trace_obj.tb_frame is frame:
                # I.e.: tb_next should be only None in the context it was thrown (trace_obj.tb_frame is frame is just a double check).
                pass
            else:
                # Get the trace_obj from where the exception was raised...
                while trace_obj.tb_next is not None:
                    trace_obj = trace_obj.tb_next

            if main_debugger.ignore_exceptions_thrown_in_lines_with_ignore_exception:
                for check_trace_obj in (initial_trace_obj, trace_obj):
                    abs_real_path_and_base = get_abs_path_real_path_and_base_from_frame(check_trace_obj.tb_frame)
                    absolute_filename = abs_real_path_and_base[0]
                    canonical_normalized_filename = abs_real_path_and_base[1]

                    filename_to_lines_where_exceptions_are_ignored = self.filename_to_lines_where_exceptions_are_ignored

                    lines_ignored = filename_to_lines_where_exceptions_are_ignored.get(canonical_normalized_filename)
                    if lines_ignored is None:
                        lines_ignored = filename_to_lines_where_exceptions_are_ignored[canonical_normalized_filename] = {}

                    try:
                        curr_stat = os.stat(absolute_filename)
                        curr_stat = (curr_stat.st_size, curr_stat.st_mtime)
                    except:
                        curr_stat = None

                    last_stat = self.filename_to_stat_info.get(absolute_filename)
                    if last_stat != curr_stat:
                        self.filename_to_stat_info[absolute_filename] = curr_stat
                        lines_ignored.clear()
                        try:
                            linecache.checkcache(absolute_filename)
                        except:
                            pydev_log.exception('Error in linecache.checkcache(%r)', absolute_filename)

                    from_user_input = main_debugger.filename_to_lines_where_exceptions_are_ignored.get(canonical_normalized_filename)
                    if from_user_input:
                        merged = {}
                        merged.update(lines_ignored)
                        # Override what we have with the related entries that the user entered
                        merged.update(from_user_input)
                    else:
                        merged = lines_ignored

                    exc_lineno = check_trace_obj.tb_lineno

                    # print ('lines ignored', lines_ignored)
                    # print ('user input', from_user_input)
                    # print ('merged', merged, 'curr', exc_lineno)

                    if exc_lineno not in merged:  # Note: check on merged but update lines_ignored.
                        try:
                            line = linecache.getline(absolute_filename, exc_lineno, check_trace_obj.tb_frame.f_globals)
                        except:
                            pydev_log.exception('Error in linecache.getline(%r, %s, f_globals)', absolute_filename, exc_lineno)
                            line = ''

                        if IGNORE_EXCEPTION_TAG.match(line) is not None:
                            lines_ignored[exc_lineno] = 1
                            return False
                        else:
                            # Put in the cache saying not to ignore
                            lines_ignored[exc_lineno] = 0
                    else:
                        # Ok, dict has it already cached, so, let's check it...
                        if merged.get(exc_lineno, 0):
                            return False

            thread = self._args[3]

            try:
                frame_id_to_frame = {}
                frame_id_to_frame[id(frame)] = frame
                f = trace_obj.tb_frame
                while f is not None:
                    frame_id_to_frame[id(f)] = f
                    f = f.f_back
                f = None

                stopped = True
                main_debugger.send_caught_exception_stack(thread, arg, id(frame))
                try:
                    self.set_suspend(thread, CMD_STEP_CAUGHT_EXCEPTION)
                    self.do_wait_suspend(thread, frame, event, arg, exception_type=exception_type)
                finally:
                    main_debugger.send_caught_exception_stack_proceeded(thread)
            except:
                pydev_log.exception()

            main_debugger.set_trace_for_frame_and_parents(frame)
        finally:
            # Make sure the user cannot see the '__exception__' we added after we leave the suspend state.
            remove_exception_from_frame(frame)
            # Clear some local variables...
            frame = None
            trace_obj = None
            initial_trace_obj = None
            check_trace_obj = None
            f = None
            frame_id_to_frame = None
            main_debugger = None
            thread = None

        return stopped

    # IFDEF CYTHON
    # cdef get_func_name(self, frame):
    #     cdef str func_name
    # ELSE
    def get_func_name(self, frame):
    # ENDIF
        code_obj = frame.f_code
        func_name = code_obj.co_name
        try:
            cls_name = get_clsname_for_code(code_obj, frame)
            if cls_name is not None:
                return "%s.%s" % (cls_name, func_name)
            else:
                return func_name
        except:
            pydev_log.exception()
            return func_name

    # IFDEF CYTHON
    # cdef _show_return_values(self, frame, arg):
    # ELSE
    def _show_return_values(self, frame, arg):
    # ENDIF
        try:
            try:
                f_locals_back = getattr(frame.f_back, "f_locals", None)
                if f_locals_back is not None:
                    return_values_dict = f_locals_back.get(RETURN_VALUES_DICT, None)
                    if return_values_dict is None:
                        return_values_dict = {}
                        f_locals_back[RETURN_VALUES_DICT] = return_values_dict
                    name = self.get_func_name(frame)
                    return_values_dict[name] = arg
            except:
                pydev_log.exception()
        finally:
            f_locals_back = None

    # IFDEF CYTHON
    # cdef _remove_return_values(self, main_debugger, frame):
    # ELSE
    def _remove_return_values(self, main_debugger, frame):
    # ENDIF
        try:
            try:
                # Showing return values was turned off, we should remove them from locals dict.
                # The values can be in the current frame or in the back one
                frame.f_locals.pop(RETURN_VALUES_DICT, None)

                f_locals_back = getattr(frame.f_back, "f_locals", None)
                if f_locals_back is not None:
                    f_locals_back.pop(RETURN_VALUES_DICT, None)
            except:
                pydev_log.exception()
        finally:
            f_locals_back = None

    # IFDEF CYTHON
    # cdef _get_unfiltered_back_frame(self, main_debugger, frame):
    # ELSE
    def _get_unfiltered_back_frame(self, main_debugger, frame):
    # ENDIF
        f = frame.f_back
        while f is not None:
            if not main_debugger.is_files_filter_enabled:
                return f

            else:
                if main_debugger.apply_files_filter(f, f.f_code.co_filename, False):
                    f = f.f_back

                else:
                    return f

        return f

    # IFDEF CYTHON
    # cdef _is_same_frame(self, target_frame, current_frame):
    #     cdef PyDBAdditionalThreadInfo info;
    # ELSE
    def _is_same_frame(self, target_frame, current_frame):
    # ENDIF
        if target_frame is current_frame:
            return True

        info = self._args[2]
        if info.pydev_use_scoped_step_frame:
            # If using scoped step we don't check the target, we just need to check
            # if the current matches the same heuristic where the target was defined.
            if target_frame is not None and current_frame is not None:
                if target_frame.f_code.co_filename == current_frame.f_code.co_filename:
                    # The co_name may be different (it may include the line number), but
                    # the filename must still be the same.
                    f = current_frame.f_back
                    if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                        f = f.f_back
                        if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                            return True

        return False

    # IFDEF CYTHON
    # cpdef trace_dispatch(self, frame, str event, arg):
    #     cdef tuple abs_path_canonical_path_and_base;
    #     cdef bint is_exception_event;
    #     cdef bint has_exception_breakpoints;
    #     cdef bint can_skip;
    #     cdef bint stop;
    #     cdef bint stop_on_plugin_breakpoint;
    #     cdef PyDBAdditionalThreadInfo info;
    #     cdef int step_cmd;
    #     cdef int line;
    #     cdef bint is_line;
    #     cdef bint is_call;
    #     cdef bint is_return;
    #     cdef bint should_stop;
    #     cdef dict breakpoints_for_file;
    #     cdef dict stop_info;
    #     cdef str curr_func_name;
    #     cdef dict frame_skips_cache;
    #     cdef object frame_cache_key;
    #     cdef tuple line_cache_key;
    #     cdef int breakpoints_in_line_cache;
    #     cdef int breakpoints_in_frame_cache;
    #     cdef bint has_breakpoint_in_frame;
    #     cdef bint is_coroutine_or_generator;
    #     cdef int bp_line;
    #     cdef object bp;
    #     cdef int pydev_smart_parent_offset
    #     cdef int pydev_smart_child_offset
    #     cdef tuple pydev_smart_step_into_variants
    # ELSE
    def trace_dispatch(self, frame, event, arg):
    # ENDIF
        # Note: this is a big function because most of the logic related to hitting a breakpoint and
        # stepping is contained in it. Ideally this could be split among multiple functions, but the
        # problem in this case is that in pure-python function calls are expensive and even more so
        # when tracing is on (because each function call will get an additional tracing call). We
        # try to address this by using the info.is_tracing for the fastest possible return, but the
        # cost is still high (maybe we could use code-generation in the future and make the code
        # generation be better split among what each part does).

        try:
            # DEBUG = '_debugger_case_generator.py' in frame.f_code.co_filename
            main_debugger, abs_path_canonical_path_and_base, info, thread, frame_skips_cache, frame_cache_key = self._args
            # if DEBUG: print('frame trace_dispatch %s %s %s %s %s %s, stop: %s' % (frame.f_lineno, frame.f_code.co_name, frame.f_code.co_filename, event, constant_to_str(info.pydev_step_cmd), arg, info.pydev_step_stop))
            info.is_tracing += 1

            # TODO: This shouldn't be needed. The fact that frame.f_lineno
            # is None seems like a bug in Python 3.11.
            # Reported in: https://github.com/python/cpython/issues/94485
            line = frame.f_lineno or 0  # Workaround or case where frame.f_lineno is None
            line_cache_key = (frame_cache_key, line)

            if main_debugger.pydb_disposed:
                return None if event == 'call' else NO_FTRACE

            plugin_manager = main_debugger.plugin
            has_exception_breakpoints = (
                main_debugger.break_on_caught_exceptions
                or main_debugger.break_on_user_uncaught_exceptions
                or main_debugger.has_plugin_exception_breaks)

            stop_frame = info.pydev_step_stop
            step_cmd = info.pydev_step_cmd
            function_breakpoint_on_call_event = None

            if frame.f_code.co_flags & 0xa0:  # 0xa0 ==  CO_GENERATOR = 0x20 | CO_COROUTINE = 0x80
                # Dealing with coroutines and generators:
                # When in a coroutine we change the perceived event to the debugger because
                # a call, StopIteration exception and return are usually just pausing/unpausing it.
                if event == 'line':
                    is_line = True
                    is_call = False
                    is_return = False
                    is_exception_event = False

                elif event == 'return':
                    is_line = False
                    is_call = False
                    is_return = True
                    is_exception_event = False

                    returns_cache_key = (frame_cache_key, 'returns')
                    return_lines = frame_skips_cache.get(returns_cache_key)
                    if return_lines is None:
                        # Note: we're collecting the return lines by inspecting the bytecode as
                        # there are multiple returns and multiple stop iterations when awaiting and
                        # it doesn't give any clear indication when a coroutine or generator is
                        # finishing or just pausing.
                        return_lines = set()
                        for x in main_debugger.collect_return_info(frame.f_code):
                            # Note: cython does not support closures in cpdefs (so we can't use
                            # a list comprehension).
                            return_lines.add(x.return_line)

                        frame_skips_cache[returns_cache_key] = return_lines

                    if line not in return_lines:
                        # Not really a return (coroutine/generator paused).
                        return self.trace_dispatch
                    else:
                        if self.exc_info:
                            self.handle_user_exception(frame)
                            return self.trace_dispatch

                        # Tricky handling: usually when we're on a frame which is about to exit
                        # we set the step mode to step into, but in this case we'd end up in the
                        # asyncio internal machinery, which is not what we want, so, we just
                        # ask the stop frame to be a level up.
                        #
                        # Note that there's an issue here which we may want to fix in the future: if
                        # the back frame is a frame which is filtered, we won't stop properly.
                        # Solving this may not be trivial as we'd need to put a scope in the step
                        # in, but we may have to do it anyways to have a step in which doesn't end
                        # up in asyncio).
                        #
                        # Note2: we don't revert to a step in if we're doing scoped stepping
                        # (because on scoped stepping we're always receiving a call/line/return
                        # event for each line in ipython, so, we can't revert to step in on return
                        # as the return shouldn't mean that we've actually completed executing a
                        # frame in this case).
                        if stop_frame is frame and not info.pydev_use_scoped_step_frame:
                            if step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
                                f = self._get_unfiltered_back_frame(main_debugger, frame)
                                if f is not None:
                                    info.pydev_step_cmd = CMD_STEP_INTO_COROUTINE
                                    info.pydev_step_stop = f
                                else:
                                    if step_cmd == CMD_STEP_OVER:
                                        info.pydev_step_cmd = CMD_STEP_INTO
                                        info.pydev_step_stop = None

                                    elif step_cmd == CMD_STEP_OVER_MY_CODE:
                                        info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE
                                        info.pydev_step_stop = None

                            elif step_cmd == CMD_STEP_INTO_COROUTINE:
                                # We're exiting this one, so, mark the new coroutine context.
                                f = self._get_unfiltered_back_frame(main_debugger, frame)
                                if f is not None:
                                    info.pydev_step_stop = f
                                else:
                                    info.pydev_step_cmd = CMD_STEP_INTO
                                    info.pydev_step_stop = None

                elif event == 'exception':
                    breakpoints_for_file = None
                    if has_exception_breakpoints:
                        should_stop, frame = self._should_stop_on_exception(frame, event, arg)
                        if should_stop:
                            if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                                return self.trace_dispatch

                    return self.trace_dispatch
                else:
                    # event == 'call' or event == 'c_XXX'
                    return self.trace_dispatch

            else:  # Not coroutine nor generator
                if event == 'line':
                    is_line = True
                    is_call = False
                    is_return = False
                    is_exception_event = False

                elif event == 'return':
                    is_line = False
                    is_return = True
                    is_call = False
                    is_exception_event = False

                    # If we are in single step mode and something causes us to exit the current frame, we need to make sure we break
                    # eventually.  Force the step mode to step into and the step stop frame to None.
                    # I.e.: F6 in the end of a function should stop in the next possible position (instead of forcing the user
                    # to make a step in or step over at that location).
                    # Note: this is especially troublesome when we're skipping code with the
                    # @DontTrace comment.
                    if (
                            stop_frame is frame and
                            not info.pydev_use_scoped_step_frame and is_return and
                            step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_STEP_OVER_MY_CODE, CMD_STEP_RETURN_MY_CODE, CMD_SMART_STEP_INTO)
                        ):

                        if step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_SMART_STEP_INTO):
                            info.pydev_step_cmd = CMD_STEP_INTO
                        else:
                            info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE
                        info.pydev_step_stop = None

                    if self.exc_info:
                        if self.handle_user_exception(frame):
                            return self.trace_dispatch

                elif event == 'call':
                    is_line = False
                    is_call = True
                    is_return = False
                    is_exception_event = False
                    if frame.f_code.co_firstlineno == frame.f_lineno:  # Check line to deal with async/await.
                        function_breakpoint_on_call_event = main_debugger.function_breakpoint_name_to_breakpoint.get(frame.f_code.co_name)

                elif event == 'exception':
                    is_exception_event = True
                    breakpoints_for_file = None
                    if has_exception_breakpoints:
                        should_stop, frame = self._should_stop_on_exception(frame, event, arg)
                        if should_stop:
                            if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                                return self.trace_dispatch
                    is_line = False
                    is_return = False
                    is_call = False

                else:
                    # Unexpected: just keep the same trace func (i.e.: event == 'c_XXX').
                    return self.trace_dispatch

            if not is_exception_event:
                breakpoints_for_file = main_debugger.breakpoints.get(abs_path_canonical_path_and_base[1])

                can_skip = False

                if info.pydev_state == 1:  # STATE_RUN = 1
                    # we can skip if:
                    # - we have no stop marked
                    # - we should make a step return/step over and we're not in the current frame
                    # - we're stepping into a coroutine context and we're not in that context
                    if step_cmd == -1:
                        can_skip = True

                    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_STEP_OVER_MY_CODE, CMD_STEP_RETURN_MY_CODE) and not self._is_same_frame(stop_frame, frame):
                        can_skip = True

                    elif step_cmd == CMD_SMART_STEP_INTO and (
                            stop_frame is not None and
                            stop_frame is not frame and
                            stop_frame is not frame.f_back and
                            (frame.f_back is None or stop_frame is not frame.f_back.f_back)):
                        can_skip = True

                    elif step_cmd == CMD_STEP_INTO_MY_CODE:
                        if (
                            main_debugger.apply_files_filter(frame, frame.f_code.co_filename, True)
                            and (frame.f_back is None or main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True))
                            ):
                                can_skip = True

                    elif step_cmd == CMD_STEP_INTO_COROUTINE:
                        f = frame
                        while f is not None:
                            if self._is_same_frame(stop_frame, f):
                                break
                            f = f.f_back
                        else:
                            can_skip = True

                    if can_skip:
                        if plugin_manager is not None and (
                                main_debugger.has_plugin_line_breaks or main_debugger.has_plugin_exception_breaks):
                            can_skip = plugin_manager.can_skip(main_debugger, frame)

                        if can_skip and main_debugger.show_return_values and info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE) and self._is_same_frame(stop_frame, frame.f_back):
                            # trace function for showing return values after step over
                            can_skip = False

                # Let's check to see if we are in a function that has a breakpoint. If we don't have a breakpoint,
                # we will return nothing for the next trace
                # also, after we hit a breakpoint and go to some other debugging state, we have to force the set trace anyway,
                # so, that's why the additional checks are there.

                if function_breakpoint_on_call_event:
                    pass  # Do nothing here (just keep on going as we can't skip it).

                elif not breakpoints_for_file:
                    if can_skip:
                        if has_exception_breakpoints:
                            return self.trace_exception
                        else:
                            return None if is_call else NO_FTRACE

                else:
                    # When cached, 0 means we don't have a breakpoint and 1 means we have.
                    if can_skip:
                        breakpoints_in_line_cache = frame_skips_cache.get(line_cache_key, -1)
                        if breakpoints_in_line_cache == 0:
                            return self.trace_dispatch

                    breakpoints_in_frame_cache = frame_skips_cache.get(frame_cache_key, -1)
                    if breakpoints_in_frame_cache != -1:
                        # Gotten from cache.
                        has_breakpoint_in_frame = breakpoints_in_frame_cache == 1

                    else:
                        has_breakpoint_in_frame = False

                        try:
                            func_lines = set()
                            for offset_and_lineno in dis.findlinestarts(frame.f_code):
                                func_lines.add(offset_and_lineno[1])
                        except:
                            # This is a fallback for implementations where we can't get the function
                            # lines -- i.e.: jython (in this case clients need to provide the function
                            # name to decide on the skip or we won't be able to skip the function
                            # completely).

                            # Checks the breakpoint to see if there is a context match in some function.
                            curr_func_name = frame.f_code.co_name

                            # global context is set with an empty name
                            if curr_func_name in ('?', '<module>', '<lambda>'):
                                curr_func_name = ''

                            for bp in breakpoints_for_file.values():
                                # will match either global or some function
                                if bp.func_name in ('None', curr_func_name):
                                    has_breakpoint_in_frame = True
                                    break
                        else:
                            for bp_line in breakpoints_for_file:  # iterate on keys
                                if bp_line in func_lines:
                                    has_breakpoint_in_frame = True
                                    break

                        # Cache the value (1 or 0 or -1 for default because of cython).
                        if has_breakpoint_in_frame:
                            frame_skips_cache[frame_cache_key] = 1
                        else:
                            frame_skips_cache[frame_cache_key] = 0

                    if can_skip and not has_breakpoint_in_frame:
                        if has_exception_breakpoints:
                            return self.trace_exception
                        else:
                            return None if is_call else NO_FTRACE

            # We may have hit a breakpoint or we are already in step mode. Either way, let's check what we should do in this frame
            # if DEBUG: print('NOT skipped: %s %s %s %s' % (frame.f_lineno, frame.f_code.co_name, event, frame.__class__.__name__))

            try:
                stop_on_plugin_breakpoint = False
                # return is not taken into account for breakpoint hit because we'd have a double-hit in this case
                # (one for the line and the other for the return).

                stop_info = {}
                breakpoint = None
                stop = False
                stop_reason = CMD_SET_BREAK
                bp_type = None

                if function_breakpoint_on_call_event:
                    breakpoint = function_breakpoint_on_call_event
                    stop = True
                    new_frame = frame
                    stop_reason = CMD_SET_FUNCTION_BREAK

                elif is_line and info.pydev_state != STATE_SUSPEND and breakpoints_for_file is not None and line in breakpoints_for_file:
                    breakpoint = breakpoints_for_file[line]
                    new_frame = frame
                    stop = True

                elif plugin_manager is not None and main_debugger.has_plugin_line_breaks:
                    result = plugin_manager.get_breakpoint(main_debugger, self, frame, event, self._args)
                    if result:
                        stop_on_plugin_breakpoint, breakpoint, new_frame, bp_type = result

                if breakpoint:
                    # ok, hit breakpoint, now, we have to discover if it is a conditional breakpoint
                    # lets do the conditional stuff here
                    if breakpoint.expression is not None:
                        main_debugger.handle_breakpoint_expression(breakpoint, info, new_frame)

                    if stop or stop_on_plugin_breakpoint:
                        eval_result = False
                        if breakpoint.has_condition:
                            eval_result = main_debugger.handle_breakpoint_condition(info, breakpoint, new_frame)
                            if not eval_result:
                                stop = False
                                stop_on_plugin_breakpoint = False

                    if is_call and (frame.f_code.co_name in ('<lambda>', '<module>') or (line == 1 and frame.f_code.co_name.startswith('<cell'))):
                        # If we find a call for a module, it means that the module is being imported/executed for the
                        # first time. In this case we have to ignore this hit as it may later duplicated by a
                        # line event at the same place (so, if there's a module with a print() in the first line
                        # the user will hit that line twice, which is not what we want).
                        #
                        # For lambda, as it only has a single statement, it's not interesting to trace
                        # its call and later its line event as they're usually in the same line.
                        #
                        # For ipython, <cell xxx> may be executed having each line compiled as a new
                        # module, so it's the same case as <module>.

                        return self.trace_dispatch

                    # Handle logpoint (on a logpoint we should never stop).
                    if (stop or stop_on_plugin_breakpoint) and breakpoint.is_logpoint:
                        stop = False
                        stop_on_plugin_breakpoint = False

                        if info.pydev_message is not None and len(info.pydev_message) > 0:
                            cmd = main_debugger.cmd_factory.make_io_message(info.pydev_message + os.linesep, '1')
                            main_debugger.writer.add_command(cmd)

                if main_debugger.show_return_values:
                    if is_return and (
                            (info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO) and (self._is_same_frame(stop_frame, frame.f_back))) or
                            (info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE) and (self._is_same_frame(stop_frame, frame))) or
                            (info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_COROUTINE)) or
                            (
                                info.pydev_step_cmd == CMD_STEP_INTO_MY_CODE
                                and frame.f_back is not None
                                and not main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)
                            )
                        ):
                        self._show_return_values(frame, arg)

                elif main_debugger.remove_return_values_flag:
                    try:
                        self._remove_return_values(main_debugger, frame)
                    finally:
                        main_debugger.remove_return_values_flag = False

                if stop:
                    self.set_suspend(
                        thread,
                        stop_reason,
                        suspend_other_threads=breakpoint and breakpoint.suspend_policy == "ALL",
                    )

                elif stop_on_plugin_breakpoint and plugin_manager is not None:
                    result = plugin_manager.suspend(main_debugger, thread, frame, bp_type)
                    if result:
                        frame = result

                # if thread has a suspend flag, we suspend with a busy wait
                if info.pydev_state == STATE_SUSPEND:
                    self.do_wait_suspend(thread, frame, event, arg)
                    return self.trace_dispatch
                else:
                    if not breakpoint and is_line:
                        # No stop from anyone and no breakpoint found in line (cache that).
                        frame_skips_cache[line_cache_key] = 0

            except:
                # Unfortunately Python itself stops the tracing when it originates from
                # the tracing function, so, we can't do much about it (just let the user know).
                exc = sys.exc_info()[0]
                cmd = main_debugger.cmd_factory.make_console_message(
                    '%s raised from within the callback set in sys.settrace.\nDebugging will be disabled for this thread (%s).\n' % (exc, thread,))
                main_debugger.writer.add_command(cmd)
                if not issubclass(exc, (KeyboardInterrupt, SystemExit)):
                    pydev_log.exception()

                raise

            # step handling. We stop when we hit the right frame
            try:
                should_skip = 0
                if pydevd_dont_trace.should_trace_hook is not None:
                    if self.should_skip == -1:
                        # I.e.: cache the result on self.should_skip (no need to evaluate the same frame multiple times).
                        # Note that on a code reload, we won't re-evaluate this because in practice, the frame.f_code
                        # Which will be handled by this frame is read-only, so, we can cache it safely.
                        if not pydevd_dont_trace.should_trace_hook(frame, abs_path_canonical_path_and_base[0]):
                            # -1, 0, 1 to be Cython-friendly
                            should_skip = self.should_skip = 1
                        else:
                            should_skip = self.should_skip = 0
                    else:
                        should_skip = self.should_skip

                plugin_stop = False
                if should_skip:
                    stop = False

                elif step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE):
                    force_check_project_scope = step_cmd == CMD_STEP_INTO_MY_CODE
                    if is_line:
                        if not info.pydev_use_scoped_step_frame:
                            if force_check_project_scope or main_debugger.is_files_filter_enabled:
                                stop = not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, force_check_project_scope)
                            else:
                                stop = True
                        else:
                            if force_check_project_scope or main_debugger.is_files_filter_enabled:
                                # Make sure we check the filtering inside ipython calls too...
                                if not not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, force_check_project_scope):
                                    return None if is_call else NO_FTRACE

                            # We can only stop inside the ipython call.
                            filename = frame.f_code.co_filename
                            if filename.endswith('.pyc'):
                                filename = filename[:-1]

                            if not filename.endswith(PYDEVD_IPYTHON_CONTEXT[0]):
                                f = frame.f_back
                                while f is not None:
                                    if f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                                        f2 = f.f_back
                                        if f2 is not None and f2.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                                            pydev_log.debug('Stop inside ipython call')
                                            stop = True
                                            break
                                    f = f.f_back

                                del f

                            if not stop:
                                # In scoped mode if step in didn't work in this context it won't work
                                # afterwards anyways.
                                return None if is_call else NO_FTRACE

                    elif is_return and frame.f_back is not None and not info.pydev_use_scoped_step_frame:
                        if main_debugger.get_file_type(frame.f_back) == main_debugger.PYDEV_FILE:
                            stop = False
                        else:
                            if force_check_project_scope or main_debugger.is_files_filter_enabled:
                                stop = not main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, force_check_project_scope)
                                if stop:
                                    # Prevent stopping in a return to the same location we were initially
                                    # (i.e.: double-stop at the same place due to some filtering).
                                    if info.step_in_initial_location == (frame.f_back, frame.f_back.f_lineno):
                                        stop = False
                            else:
                                stop = True
                    else:
                        stop = False

                    if stop:
                        if step_cmd == CMD_STEP_INTO_COROUTINE:
                            # i.e.: Check if we're stepping into the proper context.
                            f = frame
                            while f is not None:
                                if self._is_same_frame(stop_frame, f):
                                    break
                                f = f.f_back
                            else:
                                stop = False

                    if plugin_manager is not None:
                        result = plugin_manager.cmd_step_into(main_debugger, frame, event, self._args, stop_info, stop)
                        if result:
                            stop, plugin_stop = result

                elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE):
                    # Note: when dealing with a step over my code it's the same as a step over (the
                    # difference is that when we return from a frame in one we go to regular step
                    # into and in the other we go to a step into my code).
                    stop = self._is_same_frame(stop_frame, frame) and is_line
                    # Note: don't stop on a return for step over, only for line events
                    # i.e.: don't stop in: (stop_frame is frame.f_back and is_return) as we'd stop twice in that line.

                    if plugin_manager is not None:
                        result = plugin_manager.cmd_step_over(main_debugger, frame, event, self._args, stop_info, stop)
                        if result:
                            stop, plugin_stop = result

                elif step_cmd == CMD_SMART_STEP_INTO:
                    stop = False
                    back = frame.f_back
                    if self._is_same_frame(stop_frame, frame) and is_return:
                        # We're exiting the smart step into initial frame (so, we probably didn't find our target).
                        stop = True

                    elif self._is_same_frame(stop_frame, back) and is_line:
                        if info.pydev_smart_child_offset != -1:
                            # i.e.: in this case, we're not interested in the pause in the parent, rather
                            # we're interested in the pause in the child (when the parent is at the proper place).
                            stop = False

                        else:
                            pydev_smart_parent_offset = info.pydev_smart_parent_offset

                            pydev_smart_step_into_variants = info.pydev_smart_step_into_variants
                            if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                                # Preferred mode (when the smart step into variants are available
                                # and the offset is set).
                                stop = get_smart_step_into_variant_from_frame_offset(back.f_lasti, pydev_smart_step_into_variants) is \
                                       get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)

                            else:
                                # Only the name/line is available, so, check that.
                                curr_func_name = frame.f_code.co_name

                                # global context is set with an empty name
                                if curr_func_name in ('?', '<module>') or curr_func_name is None:
                                    curr_func_name = ''
                                if curr_func_name == info.pydev_func_name and stop_frame.f_lineno == info.pydev_next_line:
                                    stop = True

                        if not stop:
                            # In smart step into, if we didn't hit it in this frame once, that'll
                            # not be the case next time either, so, disable tracing for this frame.
                            return None if is_call else NO_FTRACE

                    elif back is not None and self._is_same_frame(stop_frame, back.f_back) and is_line:
                        # Ok, we have to track 2 stops at this point, the parent and the child offset.
                        # This happens when handling a step into which targets a function inside a list comprehension
                        # or generator (in which case an intermediary frame is created due to an internal function call).
                        pydev_smart_parent_offset = info.pydev_smart_parent_offset
                        pydev_smart_child_offset = info.pydev_smart_child_offset
                        # print('matched back frame', pydev_smart_parent_offset, pydev_smart_child_offset)
                        # print('parent f_lasti', back.f_back.f_lasti)
                        # print('child f_lasti', back.f_lasti)
                        stop = False
                        if pydev_smart_child_offset >= 0 and pydev_smart_child_offset >= 0:
                            pydev_smart_step_into_variants = info.pydev_smart_step_into_variants

                            if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                                # Note that we don't really check the parent offset, only the offset of
                                # the child (because this is a generator, the parent may have moved forward
                                # already -- and that's ok, so, we just check that the parent frame
                                # matches in this case).
                                smart_step_into_variant = get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)
                                # print('matched parent offset', pydev_smart_parent_offset)
                                # Ok, now, check the child variant
                                children_variants = smart_step_into_variant.children_variants
                                stop = children_variants and (
                                    get_smart_step_into_variant_from_frame_offset(back.f_lasti, children_variants) is \
                                    get_smart_step_into_variant_from_frame_offset(pydev_smart_child_offset, children_variants)
                                )
                                # print('stop at child', stop)

                        if not stop:
                            # In smart step into, if we didn't hit it in this frame once, that'll
                            # not be the case next time either, so, disable tracing for this frame.
                            return None if is_call else NO_FTRACE

                elif step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                    stop = is_return and self._is_same_frame(stop_frame, frame)

                else:
                    stop = False

                if stop and step_cmd != -1 and is_return and hasattr(frame, "f_back"):
                    f_code = getattr(frame.f_back, 'f_code', None)
                    if f_code is not None:
                        if main_debugger.get_file_type(frame.f_back) == main_debugger.PYDEV_FILE:
                            stop = False

                if plugin_stop:
                    stopped_on_plugin = plugin_manager.stop(main_debugger, frame, event, self._args, stop_info, arg, step_cmd)
                elif stop:
                    if is_line:
                        self.set_suspend(thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
                        self.do_wait_suspend(thread, frame, event, arg)
                    elif is_return:  # return event
                        back = frame.f_back
                        if back is not None:
                            # When we get to the pydevd run function, the debugging has actually finished for the main thread
                            # (note that it can still go on for other threads, but for this one, we just make it finish)
                            # So, just setting it to None should be OK
                            back_absolute_filename, _, base = get_abs_path_real_path_and_base_from_frame(back)
                            if (base, back.f_code.co_name) in (DEBUG_START, DEBUG_START_PY3K):
                                back = None

                            elif base == TRACE_PROPERTY:
                                # We dont want to trace the return event of pydevd_traceproperty (custom property for debugging)
                                # if we're in a return, we want it to appear to the user in the previous frame!
                                return None if is_call else NO_FTRACE

                            elif pydevd_dont_trace.should_trace_hook is not None:
                                if not pydevd_dont_trace.should_trace_hook(back, back_absolute_filename):
                                    # In this case, we'll have to skip the previous one because it shouldn't be traced.
                                    # Also, we have to reset the tracing, because if the parent's parent (or some
                                    # other parent) has to be traced and it's not currently, we wouldn't stop where
                                    # we should anymore (so, a step in/over/return may not stop anywhere if no parent is traced).
                                    # Related test: _debugger_case17a.py
                                    main_debugger.set_trace_for_frame_and_parents(back)
                                    return None if is_call else NO_FTRACE

                        if back is not None:
                            # if we're in a return, we want it to appear to the user in the previous frame!
                            self.set_suspend(thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
                            self.do_wait_suspend(thread, back, event, arg)
                        else:
                            # in jython we may not have a back frame
                            info.pydev_step_stop = None
                            info.pydev_original_step_cmd = -1
                            info.pydev_step_cmd = -1
                            info.pydev_state = STATE_RUN

                # if we are quitting, let's stop the tracing
                if main_debugger.quitting:
                    return None if is_call else NO_FTRACE

                return self.trace_dispatch
            except:
                # Unfortunately Python itself stops the tracing when it originates from
                # the tracing function, so, we can't do much about it (just let the user know).
                exc = sys.exc_info()[0]
                cmd = main_debugger.cmd_factory.make_console_message(
                    '%s raised from within the callback set in sys.settrace.\nDebugging will be disabled for this thread (%s).\n' % (exc, thread,))
                main_debugger.writer.add_command(cmd)
                if not issubclass(exc, (KeyboardInterrupt, SystemExit)):
                    pydev_log.exception()
                raise

        finally:
            info.is_tracing -= 1

        # end trace_dispatch
