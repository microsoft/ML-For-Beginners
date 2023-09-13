from functools import partial
import itertools
import os
import sys
import socket as socket_module

from _pydev_bundle._pydev_imports_tipper import TYPE_IMPORT, TYPE_CLASS, TYPE_FUNCTION, TYPE_ATTR, \
    TYPE_BUILTIN, TYPE_PARAM
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle._debug_adapter import pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import ModuleEvent, ModuleEventBody, Module, \
    OutputEventBody, OutputEvent, ContinuedEventBody, ExitedEventBody, \
    ExitedEvent
from _pydevd_bundle.pydevd_comm_constants import CMD_THREAD_CREATE, CMD_RETURN, CMD_MODULE_EVENT, \
    CMD_WRITE_TO_CONSOLE, CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, \
    CMD_STEP_RETURN, CMD_STEP_CAUGHT_EXCEPTION, CMD_ADD_EXCEPTION_BREAK, CMD_SET_BREAK, \
    CMD_SET_NEXT_STATEMENT, CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION, \
    CMD_THREAD_RESUME_SINGLE_NOTIFICATION, CMD_THREAD_KILL, CMD_STOP_ON_START, CMD_INPUT_REQUESTED, \
    CMD_EXIT, CMD_STEP_INTO_COROUTINE, CMD_STEP_RETURN_MY_CODE, CMD_SMART_STEP_INTO, \
    CMD_SET_FUNCTION_BREAK
from _pydevd_bundle.pydevd_constants import get_thread_id, ForkSafeLock, DebugInfoHolder
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_utils import get_non_pydevd_threads
import pydevd_file_utils
from _pydevd_bundle.pydevd_comm import build_exception_info_response
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle import pydevd_frame_utils, pydevd_constants, pydevd_utils
import linecache
from io import StringIO
from _pydev_bundle import pydev_log


class ModulesManager(object):

    def __init__(self):
        self._lock = ForkSafeLock()
        self._modules = {}
        self._next_id = partial(next, itertools.count(0))

    def track_module(self, filename_in_utf8, module_name, frame):
        '''
        :return list(NetCommand):
            Returns a list with the module events to be sent.
        '''
        if filename_in_utf8 in self._modules:
            return []

        module_events = []
        with self._lock:
            # Must check again after getting the lock.
            if filename_in_utf8 in self._modules:
                return

            try:
                version = str(frame.f_globals.get('__version__', ''))
            except:
                version = '<unknown>'

            try:
                package_name = str(frame.f_globals.get('__package__', ''))
            except:
                package_name = '<unknown>'

            module_id = self._next_id()

            module = Module(module_id, module_name, filename_in_utf8)
            if version:
                module.version = version

            if package_name:
                # Note: package doesn't appear in the docs but seems to be expected?
                module.kwargs['package'] = package_name

            module_event = ModuleEvent(ModuleEventBody('new', module))

            module_events.append(NetCommand(CMD_MODULE_EVENT, 0, module_event, is_json=True))

            self._modules[filename_in_utf8] = module.to_dict()
        return module_events

    def get_modules_info(self):
        '''
        :return list(Module)
        '''
        with self._lock:
            return list(self._modules.values())


class NetCommandFactoryJson(NetCommandFactory):
    '''
    Factory for commands which will provide messages as json (they should be
    similar to the debug adapter where possible, although some differences
    are currently Ok).

    Note that it currently overrides the xml version so that messages
    can be done one at a time (any message not overridden will currently
    use the xml version) -- after having all messages handled, it should
    no longer use NetCommandFactory as the base class.
    '''

    def __init__(self):
        NetCommandFactory.__init__(self)
        self.modules_manager = ModulesManager()

    @overrides(NetCommandFactory.make_version_message)
    def make_version_message(self, seq):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_protocol_set_message)
    def make_protocol_set_message(self, seq):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_thread_created_message)
    def make_thread_created_message(self, thread):

        # Note: the thread id for the debug adapter must be an int
        # (make the actual id from get_thread_id respect that later on).
        msg = pydevd_schema.ThreadEvent(
            pydevd_schema.ThreadEventBody('started', get_thread_id(thread)),
        )

        return NetCommand(CMD_THREAD_CREATE, 0, msg, is_json=True)

    @overrides(NetCommandFactory.make_custom_frame_created_message)
    def make_custom_frame_created_message(self, frame_id, frame_description):
        self._additional_thread_id_to_thread_name[frame_id] = frame_description
        msg = pydevd_schema.ThreadEvent(
            pydevd_schema.ThreadEventBody('started', frame_id),
        )

        return NetCommand(CMD_THREAD_CREATE, 0, msg, is_json=True)

    @overrides(NetCommandFactory.make_thread_killed_message)
    def make_thread_killed_message(self, tid):
        self._additional_thread_id_to_thread_name.pop(tid, None)
        msg = pydevd_schema.ThreadEvent(
            pydevd_schema.ThreadEventBody('exited', tid),
        )

        return NetCommand(CMD_THREAD_KILL, 0, msg, is_json=True)

    @overrides(NetCommandFactory.make_list_threads_message)
    def make_list_threads_message(self, py_db, seq):
        threads = []
        for thread in get_non_pydevd_threads():
            if is_thread_alive(thread):
                thread_id = get_thread_id(thread)

                # Notify that it's created (no-op if we already notified before).
                py_db.notify_thread_created(thread_id, thread)

                thread_schema = pydevd_schema.Thread(id=thread_id, name=thread.name)
                threads.append(thread_schema.to_dict())

        for thread_id, thread_name in list(self._additional_thread_id_to_thread_name.items()):
            thread_schema = pydevd_schema.Thread(id=thread_id, name=thread_name)
            threads.append(thread_schema.to_dict())

        body = pydevd_schema.ThreadsResponseBody(threads)
        response = pydevd_schema.ThreadsResponse(
            request_seq=seq, success=True, command='threads', body=body)

        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    @overrides(NetCommandFactory.make_get_completions_message)
    def make_get_completions_message(self, seq, completions, qualifier, start):
        COMPLETION_TYPE_LOOK_UP = {
            TYPE_IMPORT: pydevd_schema.CompletionItemType.MODULE,
            TYPE_CLASS: pydevd_schema.CompletionItemType.CLASS,
            TYPE_FUNCTION: pydevd_schema.CompletionItemType.FUNCTION,
            TYPE_ATTR: pydevd_schema.CompletionItemType.FIELD,
            TYPE_BUILTIN: pydevd_schema.CompletionItemType.KEYWORD,
            TYPE_PARAM: pydevd_schema.CompletionItemType.VARIABLE,
        }

        qualifier = qualifier.lower()
        qualifier_len = len(qualifier)
        targets = []
        for completion in completions:
            label = completion[0]
            if label.lower().startswith(qualifier):
                completion = pydevd_schema.CompletionItem(
                    label=label, type=COMPLETION_TYPE_LOOK_UP[completion[3]], start=start, length=qualifier_len)
                targets.append(completion.to_dict())

        body = pydevd_schema.CompletionsResponseBody(targets)
        response = pydevd_schema.CompletionsResponse(
            request_seq=seq, success=True, command='completions', body=body)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def _format_frame_name(self, fmt, initial_name, module_name, line, path):
        if fmt is None:
            return initial_name
        frame_name = initial_name
        if fmt.get('module', False):
            if module_name:
                if initial_name == '<module>':
                    frame_name = module_name
                else:
                    frame_name = '%s.%s' % (module_name, initial_name)
            else:
                basename = os.path.basename(path)
                basename = basename[0:-3] if basename.lower().endswith('.py') else basename
                if initial_name == '<module>':
                    frame_name = '%s in %s' % (initial_name, basename)
                else:
                    frame_name = '%s.%s' % (basename, initial_name)

        if fmt.get('line', False):
            frame_name = '%s : %d' % (frame_name, line)

        return frame_name

    @overrides(NetCommandFactory.make_get_thread_stack_message)
    def make_get_thread_stack_message(self, py_db, seq, thread_id, topmost_frame, fmt, must_be_suspended=False, start_frame=0, levels=0):
        frames = []
        module_events = []

        try:
            # : :type suspended_frames_manager: SuspendedFramesManager
            suspended_frames_manager = py_db.suspended_frames_manager
            frames_list = suspended_frames_manager.get_frames_list(thread_id)
            if frames_list is None:
                # Could not find stack of suspended frame...
                if must_be_suspended:
                    return None
                else:
                    frames_list = pydevd_frame_utils.create_frames_list_from_frame(topmost_frame)

            for frame_id, frame, method_name, original_filename, filename_in_utf8, lineno, applied_mapping, show_as_current_frame, line_col_info in self._iter_visible_frames_info(
                    py_db, frames_list, flatten_chained=True
                ):

                try:
                    module_name = str(frame.f_globals.get('__name__', ''))
                except:
                    module_name = '<unknown>'

                module_events.extend(self.modules_manager.track_module(filename_in_utf8, module_name, frame))

                presentation_hint = None
                if not getattr(frame, 'IS_PLUGIN_FRAME', False):  # Never filter out plugin frames!
                    if py_db.is_files_filter_enabled and py_db.apply_files_filter(frame, original_filename, False):
                        continue

                    if not py_db.in_project_scope(frame):
                        presentation_hint = 'subtle'

                formatted_name = self._format_frame_name(fmt, method_name, module_name, lineno, filename_in_utf8)
                if show_as_current_frame:
                    formatted_name += ' (Current frame)'
                source_reference = pydevd_file_utils.get_client_filename_source_reference(filename_in_utf8)

                if not source_reference and not applied_mapping and not os.path.exists(original_filename):
                    if getattr(frame.f_code, 'co_lines', None) or getattr(frame.f_code, 'co_lnotab', None):
                        # Create a source-reference to be used where we provide the source by decompiling the code.
                        # Note: When the time comes to retrieve the source reference in this case, we'll
                        # check the linecache first (see: get_decompiled_source_from_frame_id).
                        source_reference = pydevd_file_utils.create_source_reference_for_frame_id(frame_id, original_filename)
                    else:
                        # Check if someone added a source reference to the linecache (Python attrs does this).
                        if linecache.getline(original_filename, 1):
                            source_reference = pydevd_file_utils.create_source_reference_for_linecache(
                                original_filename)

                column = 1
                endcol = None
                if line_col_info is not None:
                    try:
                        line_text = linecache.getline(original_filename, lineno)
                    except:
                        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
                            pydev_log.exception('Unable to get line from linecache for file: %s', original_filename)
                    else:
                        if line_text:
                            colno, endcolno = line_col_info.map_columns_to_line(line_text)
                            column = colno + 1
                            if line_col_info.lineno == line_col_info.end_lineno:
                                endcol = endcolno + 1

                frames.append(pydevd_schema.StackFrame(
                    frame_id, formatted_name, lineno, column=column, endColumn=endcol, source={
                        'path': filename_in_utf8,
                        'sourceReference': source_reference,
                    },
                    presentationHint=presentation_hint).to_dict())
        finally:
            topmost_frame = None

        for module_event in module_events:
            py_db.writer.add_command(module_event)

        total_frames = len(frames)
        stack_frames = frames
        if bool(levels):
            start = start_frame
            end = min(start + levels, total_frames)
            stack_frames = frames[start:end]

        response = pydevd_schema.StackTraceResponse(
            request_seq=seq,
            success=True,
            command='stackTrace',
            body=pydevd_schema.StackTraceResponseBody(stackFrames=stack_frames, totalFrames=total_frames))
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    @overrides(NetCommandFactory.make_warning_message)
    def make_warning_message(self, msg):
        category = 'important'
        body = OutputEventBody(msg, category)
        event = OutputEvent(body)
        return NetCommand(CMD_WRITE_TO_CONSOLE, 0, event, is_json=True)

    @overrides(NetCommandFactory.make_io_message)
    def make_io_message(self, msg, ctx):
        category = 'stdout' if int(ctx) == 1 else 'stderr'
        body = OutputEventBody(msg, category)
        event = OutputEvent(body)
        return NetCommand(CMD_WRITE_TO_CONSOLE, 0, event, is_json=True)

    @overrides(NetCommandFactory.make_console_message)
    def make_console_message(self, msg):
        category = 'console'
        body = OutputEventBody(msg, category)
        event = OutputEvent(body)
        return NetCommand(CMD_WRITE_TO_CONSOLE, 0, event, is_json=True)

    _STEP_REASONS = set([
        CMD_STEP_INTO,
        CMD_STEP_INTO_MY_CODE,
        CMD_STEP_OVER,
        CMD_STEP_OVER_MY_CODE,
        CMD_STEP_RETURN,
        CMD_STEP_RETURN_MY_CODE,
        CMD_STEP_INTO_MY_CODE,
        CMD_STOP_ON_START,
        CMD_STEP_INTO_COROUTINE,
        CMD_SMART_STEP_INTO,
    ])
    _EXCEPTION_REASONS = set([
        CMD_STEP_CAUGHT_EXCEPTION,
        CMD_ADD_EXCEPTION_BREAK,
    ])

    @overrides(NetCommandFactory.make_thread_suspend_single_notification)
    def make_thread_suspend_single_notification(self, py_db, thread_id, thread, stop_reason):
        exc_desc = None
        exc_name = None
        info = set_additional_thread_info(thread)

        preserve_focus_hint = False
        if stop_reason in self._STEP_REASONS:
            if info.pydev_original_step_cmd == CMD_STOP_ON_START:

                # Just to make sure that's not set as the original reason anymore.
                info.pydev_original_step_cmd = -1
                stop_reason = 'entry'
            else:
                stop_reason = 'step'
        elif stop_reason in self._EXCEPTION_REASONS:
            stop_reason = 'exception'
        elif stop_reason == CMD_SET_BREAK:
            stop_reason = 'breakpoint'
        elif stop_reason == CMD_SET_FUNCTION_BREAK:
            stop_reason = 'function breakpoint'
        elif stop_reason == CMD_SET_NEXT_STATEMENT:
            stop_reason = 'goto'
        else:
            stop_reason = 'pause'
            preserve_focus_hint = True

        if stop_reason == 'exception':
            exception_info_response = build_exception_info_response(
                py_db, thread_id, thread, -1, set_additional_thread_info, self._iter_visible_frames_info, max_frames=-1)
            exception_info_response

            exc_name = exception_info_response.body.exceptionId
            exc_desc = exception_info_response.body.description

        body = pydevd_schema.StoppedEventBody(
            reason=stop_reason,
            description=exc_desc,
            threadId=thread_id,
            text=exc_name,
            allThreadsStopped=True,
            preserveFocusHint=preserve_focus_hint,
        )
        event = pydevd_schema.StoppedEvent(body)
        return NetCommand(CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION, 0, event, is_json=True)

    @overrides(NetCommandFactory.make_thread_resume_single_notification)
    def make_thread_resume_single_notification(self, thread_id):
        body = ContinuedEventBody(threadId=thread_id, allThreadsContinued=True)
        event = pydevd_schema.ContinuedEvent(body)
        return NetCommand(CMD_THREAD_RESUME_SINGLE_NOTIFICATION, 0, event, is_json=True)

    @overrides(NetCommandFactory.make_set_next_stmnt_status_message)
    def make_set_next_stmnt_status_message(self, seq, is_success, exception_msg):
        response = pydevd_schema.GotoResponse(
            request_seq=int(seq),
            success=is_success,
            command='goto',
            body={},
            message=(None if is_success else exception_msg))
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    @overrides(NetCommandFactory.make_send_curr_exception_trace_message)
    def make_send_curr_exception_trace_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_send_curr_exception_trace_proceeded_message)
    def make_send_curr_exception_trace_proceeded_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_send_breakpoint_exception_message)
    def make_send_breakpoint_exception_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_process_created_message)
    def make_process_created_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_process_about_to_be_replaced_message)
    def make_process_about_to_be_replaced_message(self):
        event = ExitedEvent(ExitedEventBody(-1, pydevdReason="processReplaced"))

        cmd = NetCommand(CMD_RETURN, 0, event, is_json=True)

        def after_send(socket):
            socket.setsockopt(socket_module.IPPROTO_TCP, socket_module.TCP_NODELAY, 1)

        cmd.call_after_send(after_send)
        return cmd

    @overrides(NetCommandFactory.make_thread_suspend_message)
    def make_thread_suspend_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_thread_run_message)
    def make_thread_run_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_reloaded_code_message)
    def make_reloaded_code_message(self, *args, **kwargs):
        return NULL_NET_COMMAND  # Not a part of the debug adapter protocol

    @overrides(NetCommandFactory.make_input_requested_message)
    def make_input_requested_message(self, started):
        event = pydevd_schema.PydevdInputRequestedEvent(body={})
        return NetCommand(CMD_INPUT_REQUESTED, 0, event, is_json=True)

    @overrides(NetCommandFactory.make_skipped_step_in_because_of_filters)
    def make_skipped_step_in_because_of_filters(self, py_db, frame):
        msg = 'Frame skipped from debugging during step-in.'
        if py_db.get_use_libraries_filter():
            msg += ('\nNote: may have been skipped because of "justMyCode" option (default == true). '
                    'Try setting \"justMyCode\": false in the debug configuration (e.g., launch.json).\n')
        return self.make_warning_message(msg)

    @overrides(NetCommandFactory.make_evaluation_timeout_msg)
    def make_evaluation_timeout_msg(self, py_db, expression, curr_thread):
        msg = '''Evaluating: %s did not finish after %.2f seconds.
This may mean a number of things:
- This evaluation is really slow and this is expected.
    In this case it's possible to silence this error by raising the timeout, setting the
    PYDEVD_WARN_EVALUATION_TIMEOUT environment variable to a bigger value.

- The evaluation may need other threads running while it's running:
    In this case, it's possible to set the PYDEVD_UNBLOCK_THREADS_TIMEOUT
    environment variable so that if after a given timeout an evaluation doesn't finish,
    other threads are unblocked or you can manually resume all threads.

    Alternatively, it's also possible to skip breaking on a particular thread by setting a
    `pydev_do_not_trace = True` attribute in the related threading.Thread instance
    (if some thread should always be running and no breakpoints are expected to be hit in it).

- The evaluation is deadlocked:
    In this case you may set the PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT
    environment variable to true so that a thread dump is shown along with this message and
    optionally, set the PYDEVD_INTERRUPT_THREAD_TIMEOUT to some value so that the debugger
    tries to interrupt the evaluation (if possible) when this happens.
''' % (expression, pydevd_constants.PYDEVD_WARN_EVALUATION_TIMEOUT)

        if pydevd_constants.PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT:
            stream = StringIO()
            pydevd_utils.dump_threads(stream, show_pydevd_threads=False)
            msg += '\n\n%s\n' % stream.getvalue()
        return self.make_warning_message(msg)

    @overrides(NetCommandFactory.make_exit_command)
    def make_exit_command(self, py_db):
        event = pydevd_schema.TerminatedEvent(pydevd_schema.TerminatedEventBody())
        return NetCommand(CMD_EXIT, 0, event, is_json=True)
