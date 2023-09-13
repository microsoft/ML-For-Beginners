import sys
import bisect
import types

from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
    InternalSetNextStatementThread, internal_reload_code,
    InternalGetVariable, InternalGetArray, InternalLoadFullValue,
    internal_get_description, internal_get_frame, internal_evaluate_expression, InternalConsoleExec,
    internal_get_variable_json, internal_change_variable, internal_change_variable_json,
    internal_evaluate_expression_json, internal_set_expression_json, internal_get_exception_details_json,
    internal_step_in_thread, internal_smart_step_into)
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
    CMD_STEP_INTO_MY_CODE, CMD_STOP_ON_START, CMD_SMART_STEP_INTO)
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
    HTTP_JSON_PROTOCOL, JSON_PROTOCOL, DebugInfoHolder, IS_WINDOWS)
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize

try:
    import dis
except ImportError:

    def _get_code_lines(code):
        raise NotImplementedError

else:

    def _get_code_lines(code):
        if not isinstance(code, types.CodeType):
            path = code
            with tokenize.open(path) as f:
                src = f.read()
            code = compile(src, path, 'exec', 0, dont_inherit=True)
            return _get_code_lines(code)

        def iterate():
            # First, get all line starts for this code object. This does not include
            # bodies of nested class and function definitions, as they have their
            # own objects.
            for _, lineno in dis.findlinestarts(code):
                yield lineno

            # For nested class and function definitions, their respective code objects
            # are constants referenced by this object.
            for const in code.co_consts:
                if isinstance(const, types.CodeType) and const.co_filename == code.co_filename:
                    for lineno in _get_code_lines(const):
                        yield lineno

        return iterate()


class PyDevdAPI(object):

    class VariablePresentation(object):

        def __init__(self, special='group', function='group', class_='group', protected='inline'):
            self._presentation = {
                DAPGrouper.SCOPE_SPECIAL_VARS: special,
                DAPGrouper.SCOPE_FUNCTION_VARS: function,
                DAPGrouper.SCOPE_CLASS_VARS: class_,
                DAPGrouper.SCOPE_PROTECTED_VARS: protected,
            }

        def get_presentation(self, scope):
            return self._presentation[scope]

    def run(self, py_db):
        py_db.ready_to_run = True

    def notify_initialize(self, py_db):
        py_db.on_initialize()

    def notify_configuration_done(self, py_db):
        py_db.on_configuration_done()

    def notify_disconnect(self, py_db):
        py_db.on_disconnect()

    def set_protocol(self, py_db, seq, protocol):
        set_protocol(protocol.strip())
        if get_protocol() in (HTTP_JSON_PROTOCOL, JSON_PROTOCOL):
            cmd_factory_class = NetCommandFactoryJson
        else:
            cmd_factory_class = NetCommandFactory

        if not isinstance(py_db.cmd_factory, cmd_factory_class):
            py_db.cmd_factory = cmd_factory_class()

        return py_db.cmd_factory.make_protocol_set_message(seq)

    def set_ide_os_and_breakpoints_by(self, py_db, seq, ide_os, breakpoints_by):
        '''
        :param ide_os: 'WINDOWS' or 'UNIX'
        :param breakpoints_by: 'ID' or 'LINE'
        '''
        if breakpoints_by == 'ID':
            py_db._set_breakpoints_with_id = True
        else:
            py_db._set_breakpoints_with_id = False

        self.set_ide_os(ide_os)

        return py_db.cmd_factory.make_version_message(seq)

    def set_ide_os(self, ide_os):
        '''
        :param ide_os: 'WINDOWS' or 'UNIX'
        '''
        pydevd_file_utils.set_ide_os(ide_os)

    def set_gui_event_loop(self, py_db, gui_event_loop):
        py_db._gui_event_loop = gui_event_loop

    def send_error_message(self, py_db, msg):
        cmd = py_db.cmd_factory.make_warning_message('pydevd: %s\n' % (msg,))
        py_db.writer.add_command(cmd)

    def set_show_return_values(self, py_db, show_return_values):
        if show_return_values:
            py_db.show_return_values = True
        else:
            if py_db.show_return_values:
                # We should remove saved return values
                py_db.remove_return_values_flag = True
            py_db.show_return_values = False
        pydev_log.debug("Show return values: %s", py_db.show_return_values)

    def list_threads(self, py_db, seq):
        # Response is the command with the list of threads to be added to the writer thread.
        return py_db.cmd_factory.make_list_threads_message(py_db, seq)

    def request_suspend_thread(self, py_db, thread_id='*'):
        # Yes, thread suspend is done at this point, not through an internal command.
        threads = []
        suspend_all = thread_id.strip() == '*'
        if suspend_all:
            threads = pydevd_utils.get_non_pydevd_threads()

        elif thread_id.startswith('__frame__:'):
            sys.stderr.write("Can't suspend tasklet: %s\n" % (thread_id,))

        else:
            threads = [pydevd_find_thread_by_id(thread_id)]

        for t in threads:
            if t is None:
                continue
            py_db.set_suspend(
                t,
                CMD_THREAD_SUSPEND,
                suspend_other_threads=suspend_all,
                is_pause=True,
            )
            # Break here (even if it's suspend all) as py_db.set_suspend will
            # take care of suspending other threads.
            break

    def set_enable_thread_notifications(self, py_db, enable):
        '''
        When disabled, no thread notifications (for creation/removal) will be
        issued until it's re-enabled.

        Note that when it's re-enabled, a creation notification will be sent for
        all existing threads even if it was previously sent (this is meant to
        be used on disconnect/reconnect).
        '''
        py_db.set_enable_thread_notifications(enable)

    def request_disconnect(self, py_db, resume_threads):
        self.set_enable_thread_notifications(py_db, False)
        self.remove_all_breakpoints(py_db, '*')
        self.remove_all_exception_breakpoints(py_db)
        self.notify_disconnect(py_db)

        if resume_threads:
            self.request_resume_thread(thread_id='*')

    def request_resume_thread(self, thread_id):
        resume_threads(thread_id)

    def request_completions(self, py_db, seq, thread_id, frame_id, act_tok, line=-1, column=-1):
        py_db.post_method_as_internal_command(
            thread_id, internal_get_completions, seq, thread_id, frame_id, act_tok, line=line, column=column)

    def request_stack(self, py_db, seq, thread_id, fmt=None, timeout=.5, start_frame=0, levels=0):
        # If it's already suspended, get it right away.
        internal_get_thread_stack = InternalGetThreadStack(
            seq, thread_id, py_db, set_additional_thread_info, fmt=fmt, timeout=timeout, start_frame=start_frame, levels=levels)
        if internal_get_thread_stack.can_be_executed_by(get_current_thread_id(threading.current_thread())):
            internal_get_thread_stack.do_it(py_db)
        else:
            py_db.post_internal_command(internal_get_thread_stack, '*')

    def request_exception_info_json(self, py_db, request, thread_id, thread, max_frames):
        py_db.post_method_as_internal_command(
            thread_id,
            internal_get_exception_details_json,
            request,
            thread_id,
            thread,
            max_frames,
            set_additional_thread_info=set_additional_thread_info,
            iter_visible_frames_info=py_db.cmd_factory._iter_visible_frames_info,
        )

    def request_step(self, py_db, thread_id, step_cmd_id):
        t = pydevd_find_thread_by_id(thread_id)
        if t:
            py_db.post_method_as_internal_command(
                thread_id,
                internal_step_in_thread,
                thread_id,
                step_cmd_id,
                set_additional_thread_info=set_additional_thread_info,
            )
        elif thread_id.startswith('__frame__:'):
            sys.stderr.write("Can't make tasklet step command: %s\n" % (thread_id,))

    def request_smart_step_into(self, py_db, seq, thread_id, offset, child_offset):
        t = pydevd_find_thread_by_id(thread_id)
        if t:
            py_db.post_method_as_internal_command(
                thread_id, internal_smart_step_into, thread_id, offset, child_offset, set_additional_thread_info=set_additional_thread_info)
        elif thread_id.startswith('__frame__:'):
            sys.stderr.write("Can't set next statement in tasklet: %s\n" % (thread_id,))

    def request_smart_step_into_by_func_name(self, py_db, seq, thread_id, line, func_name):
        # Same thing as set next, just with a different cmd id.
        self.request_set_next(py_db, seq, thread_id, CMD_SMART_STEP_INTO, None, line, func_name)

    def request_set_next(self, py_db, seq, thread_id, set_next_cmd_id, original_filename, line, func_name):
        '''
        set_next_cmd_id may actually be one of:

        CMD_RUN_TO_LINE
        CMD_SET_NEXT_STATEMENT

        CMD_SMART_STEP_INTO -- note: request_smart_step_into is preferred if it's possible
                               to work with bytecode offset.

        :param Optional[str] original_filename:
            If available, the filename may be source translated, otherwise no translation will take
            place (the set next just needs the line afterwards as it executes locally, but for
            the Jupyter integration, the source mapping may change the actual lines and not only
            the filename).
        '''
        t = pydevd_find_thread_by_id(thread_id)
        if t:
            if original_filename is not None:
                translated_filename = self.filename_to_server(original_filename)  # Apply user path mapping.
                pydev_log.debug('Set next (after path translation) in: %s line: %s', translated_filename, line)
                func_name = self.to_str(func_name)

                assert translated_filename.__class__ == str  # i.e.: bytes on py2 and str on py3
                assert func_name.__class__ == str  # i.e.: bytes on py2 and str on py3

                # Apply source mapping (i.e.: ipython).
                _source_mapped_filename, new_line, multi_mapping_applied = py_db.source_mapping.map_to_server(
                    translated_filename, line)
                if multi_mapping_applied:
                    pydev_log.debug('Set next (after source mapping) in: %s line: %s', translated_filename, line)
                    line = new_line

            int_cmd = InternalSetNextStatementThread(thread_id, set_next_cmd_id, line, func_name, seq=seq)
            py_db.post_internal_command(int_cmd, thread_id)
        elif thread_id.startswith('__frame__:'):
            sys.stderr.write("Can't set next statement in tasklet: %s\n" % (thread_id,))

    def request_reload_code(self, py_db, seq, module_name, filename):
        '''
        :param seq: if -1 no message will be sent back when the reload is done.

        Note: either module_name or filename may be None (but not both at the same time).
        '''
        thread_id = '*'  # Any thread
        # Note: not going for the main thread because in this case it'd only do the load
        # when we stopped on a breakpoint.
        py_db.post_method_as_internal_command(
            thread_id, internal_reload_code, seq, module_name, filename)

    def request_change_variable(self, py_db, seq, thread_id, frame_id, scope, attr, value):
        '''
        :param scope: 'FRAME' or 'GLOBAL'
        '''
        py_db.post_method_as_internal_command(
            thread_id, internal_change_variable, seq, thread_id, frame_id, scope, attr, value)

    def request_get_variable(self, py_db, seq, thread_id, frame_id, scope, attrs):
        '''
        :param scope: 'FRAME' or 'GLOBAL'
        '''
        int_cmd = InternalGetVariable(seq, thread_id, frame_id, scope, attrs)
        py_db.post_internal_command(int_cmd, thread_id)

    def request_get_array(self, py_db, seq, roffset, coffset, rows, cols, fmt, thread_id, frame_id, scope, attrs):
        int_cmd = InternalGetArray(seq, roffset, coffset, rows, cols, fmt, thread_id, frame_id, scope, attrs)
        py_db.post_internal_command(int_cmd, thread_id)

    def request_load_full_value(self, py_db, seq, thread_id, frame_id, vars):
        int_cmd = InternalLoadFullValue(seq, thread_id, frame_id, vars)
        py_db.post_internal_command(int_cmd, thread_id)

    def request_get_description(self, py_db, seq, thread_id, frame_id, expression):
        py_db.post_method_as_internal_command(
            thread_id, internal_get_description, seq, thread_id, frame_id, expression)

    def request_get_frame(self, py_db, seq, thread_id, frame_id):
        py_db.post_method_as_internal_command(
            thread_id, internal_get_frame, seq, thread_id, frame_id)

    def to_str(self, s):
        '''
        -- in py3 raises an error if it's not str already.
        '''
        if s.__class__ != str:
            raise AssertionError('Expected to have str on Python 3. Found: %s (%s)' % (s, s.__class__))
        return s

    def filename_to_str(self, filename):
        '''
        -- in py3 raises an error if it's not str already.
        '''
        if filename.__class__ != str:
            raise AssertionError('Expected to have str on Python 3. Found: %s (%s)' % (filename, filename.__class__))
        return filename

    def filename_to_server(self, filename):
        filename = self.filename_to_str(filename)
        filename = pydevd_file_utils.map_file_to_server(filename)
        return filename

    class _DummyFrame(object):
        '''
        Dummy frame to be used with PyDB.apply_files_filter (as we don't really have the
        related frame as breakpoints are added before execution).
        '''

        class _DummyCode(object):

            def __init__(self, filename):
                self.co_firstlineno = 1
                self.co_filename = filename
                self.co_name = 'invalid func name '

        def __init__(self, filename):
            self.f_code = self._DummyCode(filename)
            self.f_globals = {}

    ADD_BREAKPOINT_NO_ERROR = 0
    ADD_BREAKPOINT_FILE_NOT_FOUND = 1
    ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS = 2

    # This means that the breakpoint couldn't be fully validated (more runtime
    # information may be needed).
    ADD_BREAKPOINT_LAZY_VALIDATION = 3
    ADD_BREAKPOINT_INVALID_LINE = 4

    class _AddBreakpointResult(object):

        # :see: ADD_BREAKPOINT_NO_ERROR = 0
        # :see: ADD_BREAKPOINT_FILE_NOT_FOUND = 1
        # :see: ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS = 2
        # :see: ADD_BREAKPOINT_LAZY_VALIDATION = 3
        # :see: ADD_BREAKPOINT_INVALID_LINE = 4

        __slots__ = ['error_code', 'breakpoint_id', 'translated_filename', 'translated_line', 'original_line']

        def __init__(self, breakpoint_id, translated_filename, translated_line, original_line):
            self.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR
            self.breakpoint_id = breakpoint_id
            self.translated_filename = translated_filename
            self.translated_line = translated_line
            self.original_line = original_line

    def add_breakpoint(
            self, py_db, original_filename, breakpoint_type, breakpoint_id, line, condition, func_name,
            expression, suspend_policy, hit_condition, is_logpoint, adjust_line=False, on_changed_breakpoint_state=None):
        '''
        :param str original_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function and its final value will be available in the returned _AddBreakpointResult.

        :param str breakpoint_type:
            One of: 'python-line', 'django-line', 'jinja2-line'.

        :param int breakpoint_id:

        :param int line:
            Note: it's possible that a new line was actually used. If that's the case its
            final value will be available in the returned _AddBreakpointResult.

        :param condition:
            Either None or the condition to activate the breakpoint.

        :param str func_name:
            If "None" (str), may hit in any context.
            Empty string will hit only top level.
            Any other value must match the scope of the method to be matched.

        :param str expression:
            None or the expression to be evaluated.

        :param suspend_policy:
            Either "NONE" (to suspend only the current thread when the breakpoint is hit) or
            "ALL" (to suspend all threads when a breakpoint is hit).

        :param str hit_condition:
            An expression where `@HIT@` will be replaced by the number of hits.
            i.e.: `@HIT@ == x` or `@HIT@ >= x`

        :param bool is_logpoint:
            If True and an expression is passed, pydevd will create an io message command with the
            result of the evaluation.

        :param bool adjust_line:
            If True, the breakpoint line should be adjusted if the current line doesn't really
            match an executable line (if possible).

        :param callable on_changed_breakpoint_state:
            This is called when something changed internally on the breakpoint after it was initially
            added (for instance, template file_to_line_to_breakpoints could be signaled as invalid initially and later
            when the related template is loaded, if the line is valid it could be marked as valid).

            The signature for the callback should be:
                on_changed_breakpoint_state(breakpoint_id: int, add_breakpoint_result: _AddBreakpointResult)

                Note that the add_breakpoint_result should not be modified by the callback (the
                implementation may internally reuse the same instance multiple times).

        :return _AddBreakpointResult:
        '''
        assert original_filename.__class__ == str, 'Expected str, found: %s' % (original_filename.__class__,)  # i.e.: bytes on py2 and str on py3

        original_filename_normalized = pydevd_file_utils.normcase_from_client(original_filename)

        pydev_log.debug('Request for breakpoint in: %s line: %s', original_filename, line)
        original_line = line
        # Parameters to reapply breakpoint.
        api_add_breakpoint_params = (original_filename, breakpoint_type, breakpoint_id, line, condition, func_name,
            expression, suspend_policy, hit_condition, is_logpoint)

        translated_filename = self.filename_to_server(original_filename)  # Apply user path mapping.
        pydev_log.debug('Breakpoint (after path translation) in: %s line: %s', translated_filename, line)
        func_name = self.to_str(func_name)

        assert translated_filename.__class__ == str  # i.e.: bytes on py2 and str on py3
        assert func_name.__class__ == str  # i.e.: bytes on py2 and str on py3

        # Apply source mapping (i.e.: ipython).
        source_mapped_filename, new_line, multi_mapping_applied = py_db.source_mapping.map_to_server(
            translated_filename, line)

        if multi_mapping_applied:
            pydev_log.debug('Breakpoint (after source mapping) in: %s line: %s', source_mapped_filename, new_line)
            # Note that source mapping is internal and does not change the resulting filename nor line
            # (we want the outside world to see the line in the original file and not in the ipython
            # cell, otherwise the editor wouldn't be correct as the returned line is the line to
            # which the breakpoint will be moved in the editor).
            result = self._AddBreakpointResult(breakpoint_id, original_filename, line, original_line)

            # If a multi-mapping was applied, consider it the canonical / source mapped version (translated to ipython cell).
            translated_absolute_filename = source_mapped_filename
            canonical_normalized_filename = pydevd_file_utils.normcase(source_mapped_filename)
            line = new_line

        else:
            translated_absolute_filename = pydevd_file_utils.absolute_path(translated_filename)
            canonical_normalized_filename = pydevd_file_utils.canonical_normalized_path(translated_filename)

            if adjust_line and not translated_absolute_filename.startswith('<'):
                # Validate file_to_line_to_breakpoints and adjust their positions.
                try:
                    lines = sorted(_get_code_lines(translated_absolute_filename))
                except Exception:
                    pass
                else:
                    if line not in lines:
                        # Adjust to the first preceding valid line.
                        idx = bisect.bisect_left(lines, line)
                        if idx > 0:
                            line = lines[idx - 1]

            result = self._AddBreakpointResult(breakpoint_id, original_filename, line, original_line)

        py_db.api_received_breakpoints[(original_filename_normalized, breakpoint_id)] = (canonical_normalized_filename, api_add_breakpoint_params)

        if not translated_absolute_filename.startswith('<'):
            # Note: if a mapping pointed to a file starting with '<', don't validate.

            if not pydevd_file_utils.exists(translated_absolute_filename):
                result.error_code = self.ADD_BREAKPOINT_FILE_NOT_FOUND
                return result

            if (
                    py_db.is_files_filter_enabled and
                    not py_db.get_require_module_for_filters() and
                    py_db.apply_files_filter(self._DummyFrame(translated_absolute_filename), translated_absolute_filename, False)
                ):
                # Note that if `get_require_module_for_filters()` returns False, we don't do this check.
                # This is because we don't have the module name given a file at this point (in
                # runtime it's gotten from the frame.f_globals).
                # An option could be calculate it based on the filename and current sys.path,
                # but on some occasions that may be wrong (for instance with `__main__` or if
                # the user dynamically changes the PYTHONPATH).

                # Note: depending on the use-case, filters may be changed, so, keep on going and add the
                # breakpoint even with the error code.
                result.error_code = self.ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS

        if breakpoint_type == 'python-line':
            added_breakpoint = LineBreakpoint(
                breakpoint_id, line, condition, func_name, expression, suspend_policy, hit_condition=hit_condition, is_logpoint=is_logpoint)

            file_to_line_to_breakpoints = py_db.breakpoints
            file_to_id_to_breakpoint = py_db.file_to_id_to_line_breakpoint
            supported_type = True

        else:
            add_plugin_breakpoint_result = None
            plugin = py_db.get_plugin_lazy_init()
            if plugin is not None:
                add_plugin_breakpoint_result = plugin.add_breakpoint(
                    'add_line_breakpoint', py_db, breakpoint_type, canonical_normalized_filename,
                    breakpoint_id, line, condition, expression, func_name, hit_condition=hit_condition, is_logpoint=is_logpoint,
                    add_breakpoint_result=result, on_changed_breakpoint_state=on_changed_breakpoint_state)

            if add_plugin_breakpoint_result is not None:
                supported_type = True
                added_breakpoint, file_to_line_to_breakpoints = add_plugin_breakpoint_result
                file_to_id_to_breakpoint = py_db.file_to_id_to_plugin_breakpoint
            else:
                supported_type = False

        if not supported_type:
            raise NameError(breakpoint_type)

        pydev_log.debug('Added breakpoint:%s - line:%s - func_name:%s\n', canonical_normalized_filename, line, func_name)

        if canonical_normalized_filename in file_to_id_to_breakpoint:
            id_to_pybreakpoint = file_to_id_to_breakpoint[canonical_normalized_filename]
        else:
            id_to_pybreakpoint = file_to_id_to_breakpoint[canonical_normalized_filename] = {}

        id_to_pybreakpoint[breakpoint_id] = added_breakpoint
        py_db.consolidate_breakpoints(canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
        if py_db.plugin is not None:
            py_db.has_plugin_line_breaks = py_db.plugin.has_line_breaks()
            py_db.plugin.after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)

        py_db.on_breakpoints_changed()
        return result

    def reapply_breakpoints(self, py_db):
        '''
        Reapplies all the received breakpoints as they were received by the API (so, new
        translations are applied).
        '''
        pydev_log.debug('Reapplying breakpoints.')
        values = list(py_db.api_received_breakpoints.values())  # Create a copy with items to reapply.
        self.remove_all_breakpoints(py_db, '*')
        for val in values:
            _new_filename, api_add_breakpoint_params = val
            self.add_breakpoint(py_db, *api_add_breakpoint_params)

    def remove_all_breakpoints(self, py_db, received_filename):
        '''
        Removes all the breakpoints from a given file or from all files if received_filename == '*'.

        :param str received_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function.
        '''
        assert received_filename.__class__ == str  # i.e.: bytes on py2 and str on py3
        changed = False
        lst = [
            py_db.file_to_id_to_line_breakpoint,
            py_db.file_to_id_to_plugin_breakpoint,
            py_db.breakpoints
        ]
        if hasattr(py_db, 'django_breakpoints'):
            lst.append(py_db.django_breakpoints)

        if hasattr(py_db, 'jinja2_breakpoints'):
            lst.append(py_db.jinja2_breakpoints)

        if received_filename == '*':
            py_db.api_received_breakpoints.clear()

            for file_to_id_to_breakpoint in lst:
                if file_to_id_to_breakpoint:
                    file_to_id_to_breakpoint.clear()
                    changed = True

        else:
            received_filename_normalized = pydevd_file_utils.normcase_from_client(received_filename)
            items = list(py_db.api_received_breakpoints.items())  # Create a copy to remove items.
            translated_filenames = []
            for key, val in items:
                original_filename_normalized, _breakpoint_id = key
                if original_filename_normalized == received_filename_normalized:
                    canonical_normalized_filename, _api_add_breakpoint_params = val
                    # Note: there can be actually 1:N mappings due to source mapping (i.e.: ipython).
                    translated_filenames.append(canonical_normalized_filename)
                    del py_db.api_received_breakpoints[key]

            for canonical_normalized_filename in translated_filenames:
                for file_to_id_to_breakpoint in lst:
                    if canonical_normalized_filename in file_to_id_to_breakpoint:
                        file_to_id_to_breakpoint.pop(canonical_normalized_filename, None)
                        changed = True

        if changed:
            py_db.on_breakpoints_changed(removed=True)

    def remove_breakpoint(self, py_db, received_filename, breakpoint_type, breakpoint_id):
        '''
        :param str received_filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function.

        :param str breakpoint_type:
            One of: 'python-line', 'django-line', 'jinja2-line'.

        :param int breakpoint_id:
        '''
        received_filename_normalized = pydevd_file_utils.normcase_from_client(received_filename)
        for key, val in list(py_db.api_received_breakpoints.items()):
            original_filename_normalized, existing_breakpoint_id = key
            _new_filename, _api_add_breakpoint_params = val
            if received_filename_normalized == original_filename_normalized and existing_breakpoint_id == breakpoint_id:
                del py_db.api_received_breakpoints[key]
                break
        else:
            pydev_log.info(
                'Did not find breakpoint to remove: %s (breakpoint id: %s)', received_filename, breakpoint_id)

        file_to_id_to_breakpoint = None
        received_filename = self.filename_to_server(received_filename)
        canonical_normalized_filename = pydevd_file_utils.canonical_normalized_path(received_filename)

        if breakpoint_type == 'python-line':
            file_to_line_to_breakpoints = py_db.breakpoints
            file_to_id_to_breakpoint = py_db.file_to_id_to_line_breakpoint

        elif py_db.plugin is not None:
            result = py_db.plugin.get_breakpoints(py_db, breakpoint_type)
            if result is not None:
                file_to_id_to_breakpoint = py_db.file_to_id_to_plugin_breakpoint
                file_to_line_to_breakpoints = result

        if file_to_id_to_breakpoint is None:
            pydev_log.critical('Error removing breakpoint. Cannot handle breakpoint of type %s', breakpoint_type)

        else:
            try:
                id_to_pybreakpoint = file_to_id_to_breakpoint.get(canonical_normalized_filename, {})
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                    existing = id_to_pybreakpoint[breakpoint_id]
                    pydev_log.info('Removed breakpoint:%s - line:%s - func_name:%s (id: %s)\n' % (
                        canonical_normalized_filename, existing.line, existing.func_name, breakpoint_id))

                del id_to_pybreakpoint[breakpoint_id]
                py_db.consolidate_breakpoints(canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)
                if py_db.plugin is not None:
                    py_db.has_plugin_line_breaks = py_db.plugin.has_line_breaks()
                    py_db.plugin.after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints)

            except KeyError:
                pydev_log.info("Error removing breakpoint: Breakpoint id not found: %s id: %s. Available ids: %s\n",
                    canonical_normalized_filename, breakpoint_id, list(id_to_pybreakpoint))

        py_db.on_breakpoints_changed(removed=True)

    def set_function_breakpoints(self, py_db, function_breakpoints):
        function_breakpoint_name_to_breakpoint = {}
        for function_breakpoint in function_breakpoints:
            function_breakpoint_name_to_breakpoint[function_breakpoint.func_name] = function_breakpoint

        py_db.function_breakpoint_name_to_breakpoint = function_breakpoint_name_to_breakpoint
        py_db.on_breakpoints_changed()

    def request_exec_or_evaluate(
            self, py_db, seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result):
        py_db.post_method_as_internal_command(
            thread_id, internal_evaluate_expression,
            seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result)

    def request_exec_or_evaluate_json(
            self, py_db, request, thread_id):
        py_db.post_method_as_internal_command(
            thread_id, internal_evaluate_expression_json, request, thread_id)

    def request_set_expression_json(self, py_db, request, thread_id):
        py_db.post_method_as_internal_command(
            thread_id, internal_set_expression_json, request, thread_id)

    def request_console_exec(self, py_db, seq, thread_id, frame_id, expression):
        int_cmd = InternalConsoleExec(seq, thread_id, frame_id, expression)
        py_db.post_internal_command(int_cmd, thread_id)

    def request_load_source(self, py_db, seq, filename):
        '''
        :param str filename:
            Note: must be sent as it was received in the protocol. It may be translated in this
            function.
        '''
        try:
            filename = self.filename_to_server(filename)
            assert filename.__class__ == str  # i.e.: bytes on py2 and str on py3

            with tokenize.open(filename) as stream:
                source = stream.read()
            cmd = py_db.cmd_factory.make_load_source_message(seq, source)
        except:
            cmd = py_db.cmd_factory.make_error_message(seq, get_exception_traceback_str())

        py_db.writer.add_command(cmd)

    def get_decompiled_source_from_frame_id(self, py_db, frame_id):
        '''
        :param py_db:
        :param frame_id:
        :throws Exception:
            If unable to get the frame in the currently paused frames or if some error happened
            when decompiling.
        '''
        variable = py_db.suspended_frames_manager.get_variable(int(frame_id))
        frame = variable.value

        # Check if it's in the linecache first.
        lines = (linecache.getline(frame.f_code.co_filename, i) for i in itertools.count(1))
        lines = itertools.takewhile(bool, lines)  # empty lines are '\n', EOF is ''

        source = ''.join(lines)
        if not source:
            source = code_to_bytecode_representation(frame.f_code)

        return source

    def request_load_source_from_frame_id(self, py_db, seq, frame_id):
        try:
            source = self.get_decompiled_source_from_frame_id(py_db, frame_id)
            cmd = py_db.cmd_factory.make_load_source_from_frame_id_message(seq, source)
        except:
            cmd = py_db.cmd_factory.make_error_message(seq, get_exception_traceback_str())

        py_db.writer.add_command(cmd)

    def add_python_exception_breakpoint(
            self,
            py_db,
            exception,
            condition,
            expression,
            notify_on_handled_exceptions,
            notify_on_unhandled_exceptions,
            notify_on_user_unhandled_exceptions,
            notify_on_first_raise_only,
            ignore_libraries,
        ):
        exception_breakpoint = py_db.add_break_on_exception(
            exception,
            condition=condition,
            expression=expression,
            notify_on_handled_exceptions=notify_on_handled_exceptions,
            notify_on_unhandled_exceptions=notify_on_unhandled_exceptions,
            notify_on_user_unhandled_exceptions=notify_on_user_unhandled_exceptions,
            notify_on_first_raise_only=notify_on_first_raise_only,
            ignore_libraries=ignore_libraries,
        )

        if exception_breakpoint is not None:
            py_db.on_breakpoints_changed()

    def add_plugins_exception_breakpoint(self, py_db, breakpoint_type, exception):
        supported_type = False
        plugin = py_db.get_plugin_lazy_init()
        if plugin is not None:
            supported_type = plugin.add_breakpoint('add_exception_breakpoint', py_db, breakpoint_type, exception)

        if supported_type:
            py_db.has_plugin_exception_breaks = py_db.plugin.has_exception_breaks()
            py_db.on_breakpoints_changed()
        else:
            raise NameError(breakpoint_type)

    def remove_python_exception_breakpoint(self, py_db, exception):
        try:
            cp = py_db.break_on_uncaught_exceptions.copy()
            cp.pop(exception, None)
            py_db.break_on_uncaught_exceptions = cp

            cp = py_db.break_on_caught_exceptions.copy()
            cp.pop(exception, None)
            py_db.break_on_caught_exceptions = cp

            cp = py_db.break_on_user_uncaught_exceptions.copy()
            cp.pop(exception, None)
            py_db.break_on_user_uncaught_exceptions = cp
        except:
            pydev_log.exception("Error while removing exception %s", sys.exc_info()[0])

        py_db.on_breakpoints_changed(removed=True)

    def remove_plugins_exception_breakpoint(self, py_db, exception_type, exception):
        # I.e.: no need to initialize lazy (if we didn't have it in the first place, we can't remove
        # anything from it anyways).
        plugin = py_db.plugin
        if plugin is None:
            return

        supported_type = plugin.remove_exception_breakpoint(py_db, exception_type, exception)

        if supported_type:
            py_db.has_plugin_exception_breaks = py_db.plugin.has_exception_breaks()
        else:
            pydev_log.info('No exception of type: %s was previously registered.', exception_type)

        py_db.on_breakpoints_changed(removed=True)

    def remove_all_exception_breakpoints(self, py_db):
        py_db.break_on_uncaught_exceptions = {}
        py_db.break_on_caught_exceptions = {}
        py_db.break_on_user_uncaught_exceptions = {}

        plugin = py_db.plugin
        if plugin is not None:
            plugin.remove_all_exception_breakpoints(py_db)
        py_db.on_breakpoints_changed(removed=True)

    def set_project_roots(self, py_db, project_roots):
        '''
        :param str project_roots:
        '''
        py_db.set_project_roots(project_roots)

    def set_stepping_resumes_all_threads(self, py_db, stepping_resumes_all_threads):
        py_db.stepping_resumes_all_threads = stepping_resumes_all_threads

    # Add it to the namespace so that it's available as PyDevdAPI.ExcludeFilter
    from _pydevd_bundle.pydevd_filtering import ExcludeFilter  # noqa

    def set_exclude_filters(self, py_db, exclude_filters):
        '''
        :param list(PyDevdAPI.ExcludeFilter) exclude_filters:
        '''
        py_db.set_exclude_filters(exclude_filters)

    def set_use_libraries_filter(self, py_db, use_libraries_filter):
        py_db.set_use_libraries_filter(use_libraries_filter)

    def request_get_variable_json(self, py_db, request, thread_id):
        '''
        :param VariablesRequest request:
        '''
        py_db.post_method_as_internal_command(
            thread_id, internal_get_variable_json, request)

    def request_change_variable_json(self, py_db, request, thread_id):
        '''
        :param SetVariableRequest request:
        '''
        py_db.post_method_as_internal_command(
            thread_id, internal_change_variable_json, request)

    def set_dont_trace_start_end_patterns(self, py_db, start_patterns, end_patterns):
        # Note: start/end patterns normalized internally.
        start_patterns = tuple(pydevd_file_utils.normcase(x) for x in start_patterns)
        end_patterns = tuple(pydevd_file_utils.normcase(x) for x in end_patterns)

        # After it's set the first time, we can still change it, but we need to reset the
        # related caches.
        reset_caches = False
        dont_trace_start_end_patterns_previously_set = \
            py_db.dont_trace_external_files.__name__ == 'custom_dont_trace_external_files'

        if not dont_trace_start_end_patterns_previously_set and not start_patterns and not end_patterns:
            # If it wasn't set previously and start and end patterns are empty we don't need to do anything.
            return

        if not py_db.is_cache_file_type_empty():
            # i.e.: custom function set in set_dont_trace_start_end_patterns.
            if dont_trace_start_end_patterns_previously_set:
                reset_caches = py_db.dont_trace_external_files.start_patterns != start_patterns or \
                    py_db.dont_trace_external_files.end_patterns != end_patterns

            else:
                reset_caches = True

        def custom_dont_trace_external_files(abs_path):
            normalized_abs_path = pydevd_file_utils.normcase(abs_path)
            return normalized_abs_path.startswith(start_patterns) or normalized_abs_path.endswith(end_patterns)

        custom_dont_trace_external_files.start_patterns = start_patterns
        custom_dont_trace_external_files.end_patterns = end_patterns
        py_db.dont_trace_external_files = custom_dont_trace_external_files

        if reset_caches:
            py_db.clear_dont_trace_start_end_patterns_caches()

    def stop_on_entry(self):
        main_thread = pydevd_utils.get_main_thread()
        if main_thread is None:
            pydev_log.critical('Could not find main thread while setting Stop on Entry.')
        else:
            info = set_additional_thread_info(main_thread)
            info.pydev_original_step_cmd = CMD_STOP_ON_START
            info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE

    def set_ignore_system_exit_codes(self, py_db, ignore_system_exit_codes):
        py_db.set_ignore_system_exit_codes(ignore_system_exit_codes)

    SourceMappingEntry = pydevd_source_mapping.SourceMappingEntry

    def set_source_mapping(self, py_db, source_filename, mapping):
        '''
        :param str source_filename:
            The filename for the source mapping (bytes on py2 and str on py3).
            This filename will be made absolute in this function.

        :param list(SourceMappingEntry) mapping:
            A list with the source mapping entries to be applied to the given filename.

        :return str:
            An error message if it was not possible to set the mapping or an empty string if
            everything is ok.
        '''
        source_filename = self.filename_to_server(source_filename)
        absolute_source_filename = pydevd_file_utils.absolute_path(source_filename)
        for map_entry in mapping:
            map_entry.source_filename = absolute_source_filename
        error_msg = py_db.source_mapping.set_source_mapping(absolute_source_filename, mapping)
        if error_msg:
            return error_msg

        self.reapply_breakpoints(py_db)
        return ''

    def set_variable_presentation(self, py_db, variable_presentation):
        assert isinstance(variable_presentation, self.VariablePresentation)
        py_db.variable_presentation = variable_presentation

    def get_ppid(self):
        '''
        Provides the parent pid (even for older versions of Python on Windows).
        '''
        ppid = None

        try:
            ppid = os.getppid()
        except AttributeError:
            pass

        if ppid is None and IS_WINDOWS:
            ppid = self._get_windows_ppid()

        return ppid

    def _get_windows_ppid(self):
        this_pid = os.getpid()
        for ppid, pid in _list_ppid_and_pid():
            if pid == this_pid:
                return ppid

        return None

    def _terminate_child_processes_windows(self, dont_terminate_child_pids):
        this_pid = os.getpid()
        for _ in range(50):  # Try this at most 50 times before giving up.

            # Note: we can't kill the process itself with taskkill, so, we
            # list immediate children, kill that tree and then exit this process.

            children_pids = []
            for ppid, pid in _list_ppid_and_pid():
                if ppid == this_pid:
                    if pid not in dont_terminate_child_pids:
                        children_pids.append(pid)

            if not children_pids:
                break
            else:
                for pid in children_pids:
                    self._call(
                        ['taskkill', '/F', '/PID', str(pid), '/T'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                del children_pids[:]

    def _terminate_child_processes_linux_and_mac(self, dont_terminate_child_pids):
        this_pid = os.getpid()

        def list_children_and_stop_forking(initial_pid, stop=True):
            children_pids = []
            if stop:
                # Ask to stop forking (shouldn't be called for this process, only subprocesses).
                self._call(
                    ['kill', '-STOP', str(initial_pid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            list_popen = self._popen(
                ['pgrep', '-P', str(initial_pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if list_popen is not None:
                stdout, _ = list_popen.communicate()
                for line in stdout.splitlines():
                    line = line.decode('ascii').strip()
                    if line:
                        pid = str(line)
                        if pid in dont_terminate_child_pids:
                            continue
                        children_pids.append(pid)
                        # Recursively get children.
                        children_pids.extend(list_children_and_stop_forking(pid))
            return children_pids

        previously_found = set()

        for _ in range(50):  # Try this at most 50 times before giving up.

            children_pids = list_children_and_stop_forking(this_pid, stop=False)
            found_new = False

            for pid in children_pids:
                if pid not in previously_found:
                    found_new = True
                    previously_found.add(pid)
                    self._call(
                        ['kill', '-KILL', str(pid)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

            if not found_new:
                break

    def _popen(self, cmdline, **kwargs):
        try:
            return subprocess.Popen(cmdline, **kwargs)
        except:
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                pydev_log.exception('Error running: %s' % (' '.join(cmdline)))
            return None

    def _call(self, cmdline, **kwargs):
        try:
            subprocess.check_call(cmdline, **kwargs)
        except:
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                pydev_log.exception('Error running: %s' % (' '.join(cmdline)))

    def set_terminate_child_processes(self, py_db, terminate_child_processes):
        py_db.terminate_child_processes = terminate_child_processes

    def set_terminate_keyboard_interrupt(self, py_db, terminate_keyboard_interrupt):
        py_db.terminate_keyboard_interrupt = terminate_keyboard_interrupt

    def terminate_process(self, py_db):
        '''
        Terminates the current process (and child processes if the option to also terminate
        child processes is enabled).
        '''
        try:
            if py_db.terminate_child_processes:
                pydev_log.debug('Terminating child processes.')
                if IS_WINDOWS:
                    self._terminate_child_processes_windows(py_db.dont_terminate_child_pids)
                else:
                    self._terminate_child_processes_linux_and_mac(py_db.dont_terminate_child_pids)
        finally:
            pydev_log.debug('Exiting process (os._exit(0)).')
            os._exit(0)

    def _terminate_if_commands_processed(self, py_db):
        py_db.dispose_and_kill_all_pydevd_threads()
        self.terminate_process(py_db)

    def request_terminate_process(self, py_db):
        if py_db.terminate_keyboard_interrupt:
            if not py_db.keyboard_interrupt_requested:
                py_db.keyboard_interrupt_requested = True
                interrupt_main_thread()
                return

        # We mark with a terminate_requested to avoid that paused threads start running
        # (we should terminate as is without letting any paused thread run).
        py_db.terminate_requested = True
        run_as_pydevd_daemon_thread(py_db, self._terminate_if_commands_processed, py_db)

    def setup_auto_reload_watcher(self, py_db, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns):
        py_db.setup_auto_reload_watcher(enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns)


def _list_ppid_and_pid():
    _TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_uint32),
                    ("cntUsage", ctypes.c_uint32),
                    ("th32ProcessID", ctypes.c_uint32),
                    ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", ctypes.c_uint32),
                    ("cntThreads", ctypes.c_uint32),
                    ("th32ParentProcessID", ctypes.c_uint32),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_uint32),
                    ("szExeFile", ctypes.c_char * 260)]

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    ppid_and_pids = []
    try:
        process_entry = PROCESSENTRY32()
        process_entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32First(ctypes.c_void_p(snapshot), ctypes.byref(process_entry)):
            pydev_log.critical('Process32First failed (getting process from CreateToolhelp32Snapshot).')
        else:
            while True:
                ppid_and_pids.append((process_entry.th32ParentProcessID, process_entry.th32ProcessID))
                if not kernel32.Process32Next(ctypes.c_void_p(snapshot), ctypes.byref(process_entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)

    return ppid_and_pids
