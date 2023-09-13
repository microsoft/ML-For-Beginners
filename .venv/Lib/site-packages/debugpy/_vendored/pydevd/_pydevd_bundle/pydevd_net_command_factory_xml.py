import json

from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle._pydev_saved_modules import thread
from _pydevd_bundle import pydevd_xml, pydevd_frame_utils, pydevd_constants, pydevd_utils
from _pydevd_bundle.pydevd_comm_constants import (
    CMD_THREAD_CREATE, CMD_THREAD_KILL, CMD_THREAD_SUSPEND, CMD_THREAD_RUN, CMD_GET_VARIABLE,
    CMD_EVALUATE_EXPRESSION, CMD_GET_FRAME, CMD_WRITE_TO_CONSOLE, CMD_GET_COMPLETIONS,
    CMD_LOAD_SOURCE, CMD_SET_NEXT_STATEMENT, CMD_EXIT, CMD_GET_FILE_CONTENTS,
    CMD_EVALUATE_CONSOLE_EXPRESSION, CMD_RUN_CUSTOM_OPERATION,
    CMD_GET_BREAKPOINT_EXCEPTION, CMD_SEND_CURR_EXCEPTION_TRACE,
    CMD_SEND_CURR_EXCEPTION_TRACE_PROCEEDED, CMD_SHOW_CONSOLE, CMD_GET_ARRAY,
    CMD_INPUT_REQUESTED, CMD_GET_DESCRIPTION, CMD_PROCESS_CREATED,
    CMD_SHOW_CYTHON_WARNING, CMD_LOAD_FULL_VALUE, CMD_GET_THREAD_STACK,
    CMD_GET_EXCEPTION_DETAILS, CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION,
    CMD_THREAD_RESUME_SINGLE_NOTIFICATION,
    CMD_GET_NEXT_STATEMENT_TARGETS, CMD_VERSION,
    CMD_RETURN, CMD_SET_PROTOCOL, CMD_ERROR, MAX_IO_MSG_SIZE, VERSION_STRING,
    CMD_RELOAD_CODE, CMD_LOAD_SOURCE_FROM_FRAME_ID)
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, get_thread_id,
    get_global_debugger, GetGlobalDebugger, set_global_debugger)  # Keep for backward compatibility @UnusedImport
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND, NULL_EXIT_COMMAND
from _pydevd_bundle.pydevd_utils import quote_smart as quote, get_non_pydevd_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
import pydevd_file_utils
from pydevd_tracing import get_exception_traceback_str
from _pydev_bundle._pydev_completer import completions_to_xml
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame_utils import FramesList
from io import StringIO


#=======================================================================================================================
# NetCommandFactory
#=======================================================================================================================
class NetCommandFactory(object):

    def __init__(self):
        self._additional_thread_id_to_thread_name = {}

    def _thread_to_xml(self, thread):
        """ thread information as XML """
        name = pydevd_xml.make_valid_xml_value(thread.name)
        cmd_text = '<thread name="%s" id="%s" />' % (quote(name), get_thread_id(thread))
        return cmd_text

    def make_error_message(self, seq, text):
        cmd = NetCommand(CMD_ERROR, seq, text)
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 2:
            pydev_log.error("Error: %s" % (text,))
        return cmd

    def make_protocol_set_message(self, seq):
        return NetCommand(CMD_SET_PROTOCOL, seq, '')

    def make_thread_created_message(self, thread):
        cmdText = "<xml>" + self._thread_to_xml(thread) + "</xml>"
        return NetCommand(CMD_THREAD_CREATE, 0, cmdText)

    def make_process_created_message(self):
        cmdText = '<process/>'
        return NetCommand(CMD_PROCESS_CREATED, 0, cmdText)

    def make_process_about_to_be_replaced_message(self):
        return NULL_NET_COMMAND

    def make_show_cython_warning_message(self):
        try:
            return NetCommand(CMD_SHOW_CYTHON_WARNING, 0, '')
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_custom_frame_created_message(self, frame_id, frame_description):
        self._additional_thread_id_to_thread_name[frame_id] = frame_description
        frame_description = pydevd_xml.make_valid_xml_value(frame_description)
        return NetCommand(CMD_THREAD_CREATE, 0, '<xml><thread name="%s" id="%s"/></xml>' % (frame_description, frame_id))

    def make_list_threads_message(self, py_db, seq):
        """ returns thread listing as XML """
        try:
            threads = get_non_pydevd_threads()
            cmd_text = ["<xml>"]
            append = cmd_text.append
            for thread in threads:
                if is_thread_alive(thread):
                    append(self._thread_to_xml(thread))

            for thread_id, thread_name in list(self._additional_thread_id_to_thread_name.items()):
                name = pydevd_xml.make_valid_xml_value(thread_name)
                append('<thread name="%s" id="%s" />' % (quote(name), thread_id))

            append("</xml>")
            return NetCommand(CMD_RETURN, seq, ''.join(cmd_text))
        except:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_thread_stack_message(self, py_db, seq, thread_id, topmost_frame, fmt, must_be_suspended=False, start_frame=0, levels=0):
        """
        Returns thread stack as XML.

        :param must_be_suspended: If True and the thread is not suspended, returns None.
        """
        try:
            # If frame is None, the return is an empty frame list.
            cmd_text = ['<xml><thread id="%s">' % (thread_id,)]

            if topmost_frame is not None:
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

                    cmd_text.append(self.make_thread_stack_str(py_db, frames_list))
                finally:
                    topmost_frame = None
            cmd_text.append('</thread></xml>')
            return NetCommand(CMD_GET_THREAD_STACK, seq, ''.join(cmd_text))
        except:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_variable_changed_message(self, seq, payload):
        # notify debugger that value was changed successfully
        return NetCommand(CMD_RETURN, seq, payload)

    def make_warning_message(self, msg):
        return self.make_io_message(msg, 2)

    def make_console_message(self, msg):
        return self.make_io_message(msg, 2)

    def make_io_message(self, msg, ctx):
        '''
        @param msg: the message to pass to the debug server
        @param ctx: 1 for stdio 2 for stderr
        '''
        try:
            msg = pydevd_constants.as_str(msg)

            if len(msg) > MAX_IO_MSG_SIZE:
                msg = msg[0:MAX_IO_MSG_SIZE]
                msg += '...'

            msg = pydevd_xml.make_valid_xml_value(quote(msg, '/>_= '))
            return NetCommand(str(CMD_WRITE_TO_CONSOLE), 0, '<xml><io s="%s" ctx="%s"/></xml>' % (msg, ctx))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_version_message(self, seq):
        try:
            return NetCommand(CMD_VERSION, seq, VERSION_STRING)
        except:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_thread_killed_message(self, tid):
        self._additional_thread_id_to_thread_name.pop(tid, None)
        try:
            return NetCommand(CMD_THREAD_KILL, 0, str(tid))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def _iter_visible_frames_info(self, py_db, frames_list, flatten_chained=False):
        assert frames_list.__class__ == FramesList
        is_chained = False
        while True:
            for frame in frames_list:
                show_as_current_frame = frame is frames_list.current_frame
                if frame.f_code is None:
                    pydev_log.info('Frame without f_code: %s', frame)
                    continue  # IronPython sometimes does not have it!

                method_name = frame.f_code.co_name  # method name (if in method) or ? if global
                if method_name is None:
                    pydev_log.info('Frame without co_name: %s', frame)
                    continue  # IronPython sometimes does not have it!

                if is_chained:
                    method_name = '[Chained Exc: %s] %s' % (frames_list.exc_desc, method_name)

                abs_path_real_path_and_base = get_abs_path_real_path_and_base_from_frame(frame)
                if py_db.get_file_type(frame, abs_path_real_path_and_base) == py_db.PYDEV_FILE:
                    # Skip pydevd files.
                    frame = frame.f_back
                    continue

                frame_id = id(frame)
                lineno = frames_list.frame_id_to_lineno.get(frame_id, frame.f_lineno)
                line_col_info = frames_list.frame_id_to_line_col_info.get(frame_id)

                filename_in_utf8, lineno, changed = py_db.source_mapping.map_to_client(abs_path_real_path_and_base[0], lineno)
                new_filename_in_utf8, applied_mapping = pydevd_file_utils.map_file_to_client(filename_in_utf8)
                applied_mapping = applied_mapping or changed

                yield frame_id, frame, method_name, abs_path_real_path_and_base[0], new_filename_in_utf8, lineno, applied_mapping, show_as_current_frame, line_col_info

            if not flatten_chained:
                break

            frames_list = frames_list.chained_frames_list
            if frames_list is None or len(frames_list) == 0:
                break
            is_chained = True

    def make_thread_stack_str(self, py_db, frames_list):
        assert frames_list.__class__ == FramesList
        make_valid_xml_value = pydevd_xml.make_valid_xml_value
        cmd_text_list = []
        append = cmd_text_list.append

        try:
            for frame_id, frame, method_name, _original_filename, filename_in_utf8, lineno, _applied_mapping, _show_as_current_frame, line_col_info in self._iter_visible_frames_info(
                    py_db, frames_list, flatten_chained=True
                ):

                # print("file is ", filename_in_utf8)
                # print("line is ", lineno)

                # Note: variables are all gotten 'on-demand'.
                append('<frame id="%s" name="%s" ' % (frame_id , make_valid_xml_value(method_name)))
                append('file="%s" line="%s">' % (quote(make_valid_xml_value(filename_in_utf8), '/>_= \t'), lineno))
                append("</frame>")
        except:
            pydev_log.exception()

        return ''.join(cmd_text_list)

    def make_thread_suspend_str(
        self,
        py_db,
        thread_id,
        frames_list,
        stop_reason=None,
        message=None,
        suspend_type="trace",
        ):
        """
        :return tuple(str,str):
            Returns tuple(thread_suspended_str, thread_stack_str).

            i.e.:
            (
                '''
                    <xml>
                        <thread id="id" stop_reason="reason">
                            <frame id="id" name="functionName " file="file" line="line">
                            </frame>
                        </thread>
                    </xml>
                '''
                ,
                '''
                <frame id="id" name="functionName " file="file" line="line">
                </frame>
                '''
            )
        """
        assert frames_list.__class__ == FramesList
        make_valid_xml_value = pydevd_xml.make_valid_xml_value
        cmd_text_list = []
        append = cmd_text_list.append

        cmd_text_list.append('<xml>')
        if message:
            message = make_valid_xml_value(message)

        append('<thread id="%s"' % (thread_id,))
        if stop_reason is not None:
            append(' stop_reason="%s"' % (stop_reason,))
        if message is not None:
            append(' message="%s"' % (message,))
        if suspend_type is not None:
            append(' suspend_type="%s"' % (suspend_type,))
        append('>')
        thread_stack_str = self.make_thread_stack_str(py_db, frames_list)
        append(thread_stack_str)
        append("</thread></xml>")

        return ''.join(cmd_text_list), thread_stack_str

    def make_thread_suspend_message(self, py_db, thread_id, frames_list, stop_reason, message, suspend_type):
        try:
            thread_suspend_str, thread_stack_str = self.make_thread_suspend_str(
                py_db, thread_id, frames_list, stop_reason, message, suspend_type)
            cmd = NetCommand(CMD_THREAD_SUSPEND, 0, thread_suspend_str)
            cmd.thread_stack_str = thread_stack_str
            cmd.thread_suspend_str = thread_suspend_str
            return cmd
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_thread_suspend_single_notification(self, py_db, thread_id, thread, stop_reason):
        try:
            return NetCommand(CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION, 0, json.dumps(
                {'thread_id': thread_id, 'stop_reason':stop_reason}))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_thread_resume_single_notification(self, thread_id):
        try:
            return NetCommand(CMD_THREAD_RESUME_SINGLE_NOTIFICATION, 0, json.dumps(
                {'thread_id': thread_id}))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_thread_run_message(self, thread_id, reason):
        try:
            return NetCommand(CMD_THREAD_RUN, 0, "%s\t%s" % (thread_id, reason))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_get_variable_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_VARIABLE, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_array_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_ARRAY, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_description_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_DESCRIPTION, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_frame_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_FRAME, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_evaluate_expression_message(self, seq, payload):
        try:
            return NetCommand(CMD_EVALUATE_EXPRESSION, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_completions_message(self, seq, completions, qualifier, start):
        try:
            payload = completions_to_xml(completions)
            return NetCommand(CMD_GET_COMPLETIONS, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_file_contents(self, seq, payload):
        try:
            return NetCommand(CMD_GET_FILE_CONTENTS, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_reloaded_code_message(self, seq, reloaded_ok):
        try:
            return NetCommand(CMD_RELOAD_CODE, seq, '<xml><reloaded ok="%s"></reloaded></xml>' % reloaded_ok)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_send_breakpoint_exception_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_BREAKPOINT_EXCEPTION, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def _make_send_curr_exception_trace_str(self, py_db, thread_id, exc_type, exc_desc, trace_obj):
        frames_list = pydevd_frame_utils.create_frames_list_from_traceback(trace_obj, None, exc_type, exc_desc)

        exc_type = pydevd_xml.make_valid_xml_value(str(exc_type)).replace('\t', '  ') or 'exception: type unknown'
        exc_desc = pydevd_xml.make_valid_xml_value(str(exc_desc)).replace('\t', '  ') or 'exception: no description'

        thread_suspend_str, thread_stack_str = self.make_thread_suspend_str(
            py_db, thread_id, frames_list, CMD_SEND_CURR_EXCEPTION_TRACE, '')
        return exc_type, exc_desc, thread_suspend_str, thread_stack_str

    def make_send_curr_exception_trace_message(self, py_db, seq, thread_id, curr_frame_id, exc_type, exc_desc, trace_obj):
        try:
            exc_type, exc_desc, thread_suspend_str, _thread_stack_str = self._make_send_curr_exception_trace_str(
                py_db, thread_id, exc_type, exc_desc, trace_obj)
            payload = str(curr_frame_id) + '\t' + exc_type + "\t" + exc_desc + "\t" + thread_suspend_str
            return NetCommand(CMD_SEND_CURR_EXCEPTION_TRACE, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_exception_details_message(self, py_db, seq, thread_id, topmost_frame):
        """Returns exception details as XML """
        try:
            # If the debugger is not suspended, just return the thread and its id.
            cmd_text = ['<xml><thread id="%s" ' % (thread_id,)]

            if topmost_frame is not None:
                try:
                    frame = topmost_frame
                    topmost_frame = None
                    while frame is not None:
                        if frame.f_code.co_name == 'do_wait_suspend' and frame.f_code.co_filename.endswith('pydevd.py'):
                            arg = frame.f_locals.get('arg', None)
                            if arg is not None:
                                exc_type, exc_desc, _thread_suspend_str, thread_stack_str = self._make_send_curr_exception_trace_str(
                                    py_db, thread_id, *arg)
                                cmd_text.append('exc_type="%s" ' % (exc_type,))
                                cmd_text.append('exc_desc="%s" ' % (exc_desc,))
                                cmd_text.append('>')
                                cmd_text.append(thread_stack_str)
                                break
                        frame = frame.f_back
                    else:
                        cmd_text.append('>')
                finally:
                    frame = None
            cmd_text.append('</thread></xml>')
            return NetCommand(CMD_GET_EXCEPTION_DETAILS, seq, ''.join(cmd_text))
        except:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_send_curr_exception_trace_proceeded_message(self, seq, thread_id):
        try:
            return NetCommand(CMD_SEND_CURR_EXCEPTION_TRACE_PROCEEDED, 0, str(thread_id))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_send_console_message(self, seq, payload):
        try:
            return NetCommand(CMD_EVALUATE_CONSOLE_EXPRESSION, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_custom_operation_message(self, seq, payload):
        try:
            return NetCommand(CMD_RUN_CUSTOM_OPERATION, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_load_source_message(self, seq, source):
        return NetCommand(CMD_LOAD_SOURCE, seq, source)

    def make_load_source_from_frame_id_message(self, seq, source):
        return NetCommand(CMD_LOAD_SOURCE_FROM_FRAME_ID, seq, source)

    def make_show_console_message(self, py_db, thread_id, frame):
        try:
            frames_list = pydevd_frame_utils.create_frames_list_from_frame(frame)
            thread_suspended_str, _thread_stack_str = self.make_thread_suspend_str(
                py_db, thread_id, frames_list, CMD_SHOW_CONSOLE, '')
            return NetCommand(CMD_SHOW_CONSOLE, 0, thread_suspended_str)
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_input_requested_message(self, started):
        try:
            return NetCommand(CMD_INPUT_REQUESTED, 0, str(started))
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_set_next_stmnt_status_message(self, seq, is_success, exception_msg):
        try:
            message = str(is_success) + '\t' + exception_msg
            return NetCommand(CMD_SET_NEXT_STATEMENT, int(seq), message)
        except:
            return self.make_error_message(0, get_exception_traceback_str())

    def make_load_full_value_message(self, seq, payload):
        try:
            return NetCommand(CMD_LOAD_FULL_VALUE, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_get_next_statement_targets_message(self, seq, payload):
        try:
            return NetCommand(CMD_GET_NEXT_STATEMENT_TARGETS, seq, payload)
        except Exception:
            return self.make_error_message(seq, get_exception_traceback_str())

    def make_skipped_step_in_because_of_filters(self, py_db, frame):
        return NULL_NET_COMMAND  # Not a part of the xml protocol

    def make_evaluation_timeout_msg(self, py_db, expression, thread):
        msg = '''pydevd: Evaluating: %s did not finish after %.2f seconds.
This may mean a number of things:
- This evaluation is really slow and this is expected.
    In this case it's possible to silence this error by raising the timeout, setting the
    PYDEVD_WARN_EVALUATION_TIMEOUT environment variable to a bigger value.

- The evaluation may need other threads running while it's running:
    In this case, you may need to manually let other paused threads continue.

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

    def make_exit_command(self, py_db):
        return NULL_EXIT_COMMAND
