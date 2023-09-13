# coding: utf-8
from collections import namedtuple
import json
from os.path import normcase
import os.path
import sys
import time

import pytest

from _pydev_bundle.pydev_localhost import get_socket_name
from _pydevd_bundle._debug_adapter import pydevd_schema, pydevd_base_schema
from _pydevd_bundle._debug_adapter.pydevd_base_schema import from_json
from _pydevd_bundle._debug_adapter.pydevd_schema import (ThreadEvent, ModuleEvent, OutputEvent,
    ExceptionOptions, Response, StoppedEvent, ContinuedEvent, ProcessEvent, InitializeRequest,
    InitializeRequestArguments, TerminateArguments, TerminateRequest, TerminatedEvent,
    FunctionBreakpoint, SetFunctionBreakpointsRequest, SetFunctionBreakpointsArguments,
    BreakpointEvent, InitializedEvent)
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding
from _pydevd_bundle.pydevd_constants import (int_types, IS_64BIT_PROCESS,
    PY_VERSION_STR, PY_IMPL_VERSION_STR, PY_IMPL_NAME, IS_PY36_OR_GREATER,
    IS_PYPY, GENERATED_LEN_ATTR_NAME, IS_WINDOWS, IS_LINUX, IS_MAC, IS_PY38_OR_GREATER,
    IS_PY311_OR_GREATER)
from tests_python import debugger_unittest
from tests_python.debug_constants import TEST_CHERRYPY, TEST_DJANGO, TEST_FLASK, \
    IS_CPYTHON, TEST_GEVENT, TEST_CYTHON, TODO_PY311
from tests_python.debugger_unittest import (IS_JYTHON, IS_APPVEYOR, overrides,
    get_free_port, wait_for_condition)
from _pydevd_bundle.pydevd_utils import DAPGrouper
import pydevd_file_utils
from _pydevd_bundle import pydevd_constants

pytest_plugins = [
    str('tests_python.debugger_fixtures'),
]

_JsonHit = namedtuple('_JsonHit', 'thread_id, frame_id, stack_trace_response')

pytestmark = pytest.mark.skipif(IS_JYTHON, reason='Single notification is not OK in Jython (investigate).')

# Note: in reality must be < int32, but as it's created sequentially this should be
# a reasonable number for tests.
MAX_EXPECTED_ID = 10000


class _MessageWithMark(object):

    def __init__(self, msg):
        self.msg = msg
        self.marked = False


class JsonFacade(object):

    def __init__(self, writer):
        self.writer = writer
        writer.reader_thread.accept_xml_messages = False
        self._all_json_messages_found = []
        self._sent_launch_or_attach = False

    def mark_messages(self, expected_class, accept_message=lambda obj:True):
        ret = []
        for message_with_mark in self._all_json_messages_found:
            if not message_with_mark.marked:
                if isinstance(message_with_mark.msg, expected_class):
                    if accept_message(message_with_mark.msg):
                        message_with_mark.marked = True
                        ret.append(message_with_mark.msg)
        return ret

    def wait_for_json_message(self, expected_class, accept_message=lambda obj:True):

        def accept_json_message(msg):
            if msg.startswith('{'):
                decoded_msg = from_json(msg)

                self._all_json_messages_found.append(_MessageWithMark(decoded_msg))

                if isinstance(decoded_msg, expected_class):
                    if accept_message(decoded_msg):
                        return True
            return False

        msg = self.writer.wait_for_message(accept_json_message, unquote_msg=False, expect_xml=False)
        return from_json(msg)

    def wait_for_response(self, request, response_class=None):
        if response_class is None:
            response_class = pydevd_base_schema.get_response_class(request)

        def accept_message(response):
            if isinstance(request, dict):
                if response.request_seq == request['seq']:
                    return True
            else:
                if response.request_seq == request.seq:
                    return True
            return False

        return self.wait_for_json_message((response_class, Response), accept_message)

    def write_request(self, request):
        seq = self.writer.next_seq()
        if isinstance(request, dict):
            request['seq'] = seq
            self.writer.write_with_content_len(json.dumps(request))
        else:
            request.seq = seq
            self.writer.write_with_content_len(request.to_json())
        return request

    def write_make_initial_run(self):
        if not self._sent_launch_or_attach:
            self._auto_write_launch()

        configuration_done_request = self.write_request(pydevd_schema.ConfigurationDoneRequest())
        return self.wait_for_response(configuration_done_request)

    def write_list_threads(self):
        return self.wait_for_response(self.write_request(pydevd_schema.ThreadsRequest()))

    def wait_for_terminated(self):
        return self.wait_for_json_message(TerminatedEvent)

    def wait_for_thread_stopped(self, reason='breakpoint', line=None, file=None, name=None, preserve_focus_hint=None):
        '''
        :param file:
            utf-8 bytes encoded file or unicode
        '''
        stopped_event = self.wait_for_json_message(StoppedEvent)
        assert stopped_event.body.reason == reason
        if preserve_focus_hint is not None:
            assert stopped_event.body.preserveFocusHint == preserve_focus_hint
        json_hit = self.get_stack_as_json_hit(stopped_event.body.threadId)
        if file is not None:
            path = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']

            if not path.endswith(file):
                # pytest may give a lowercase tempdir, so, also check with
                # the real case if possible
                file = pydevd_file_utils.get_path_with_real_case(file)
                if not path.endswith(file):
                    raise AssertionError('Expected path: %s to end with: %s' % (path, file))
        if name is not None:
            assert json_hit.stack_trace_response.body.stackFrames[0]['name'] == name
        if line is not None:
            found_line = json_hit.stack_trace_response.body.stackFrames[0]['line']
            path = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']
            if not isinstance(line, (tuple, list)):
                line = [line]
            assert found_line in line, 'Expect to break at line: %s. Found: %s (file: %s)' % (line, found_line, path)
        return json_hit

    def write_set_function_breakpoints(
        self, function_names):
        function_breakpoints = [FunctionBreakpoint(name,) for name in function_names]
        arguments = SetFunctionBreakpointsArguments(function_breakpoints)
        request = SetFunctionBreakpointsRequest(arguments)
        response = self.wait_for_response(self.write_request(request))
        assert response.success

    def write_set_breakpoints(
            self,
            lines,
            filename=None,
            line_to_info=None,
            success=True,
            verified=True,
            send_launch_if_needed=True,
            expected_lines_in_response=None,
        ):
        '''
        Adds a breakpoint.
        '''
        if send_launch_if_needed and not self._sent_launch_or_attach:
            self._auto_write_launch()

        if isinstance(lines, int):
            lines = [lines]

        if line_to_info is None:
            line_to_info = {}

        if filename is None:
            filename = self.writer.get_main_filename()

        if isinstance(filename, bytes):
            filename = filename.decode(file_system_encoding)  # file is in the filesystem encoding but protocol needs it in utf-8
            filename = filename.encode('utf-8')

        source = pydevd_schema.Source(path=filename)
        breakpoints = []
        for line in lines:
            condition = None
            hit_condition = None
            log_message = None

            if line in line_to_info:
                line_info = line_to_info.get(line)
                condition = line_info.get('condition')
                hit_condition = line_info.get('hit_condition')
                log_message = line_info.get('log_message')

            breakpoints.append(pydevd_schema.SourceBreakpoint(
                line, condition=condition, hitCondition=hit_condition, logMessage=log_message).to_dict())

        arguments = pydevd_schema.SetBreakpointsArguments(source, breakpoints)
        request = pydevd_schema.SetBreakpointsRequest(arguments)

        # : :type response: SetBreakpointsResponse
        response = self.wait_for_response(self.write_request(request))
        body = response.body

        assert response.success == success

        if success:
            # : :type body: SetBreakpointsResponseBody
            assert len(body.breakpoints) == len(lines)
            lines_in_response = [b['line'] for b in body.breakpoints]

            if expected_lines_in_response is None:
                expected_lines_in_response = lines
            assert set(lines_in_response) == set(expected_lines_in_response)

            for b in body.breakpoints:
                if isinstance(verified, dict):
                    if b['verified'] != verified[b['id']]:
                        raise AssertionError('Expected verified breakpoint to be: %s. Found: %s.\nBreakpoint: %s' % (
                            verified, verified[b['id']], b))

                elif b['verified'] != verified:
                    raise AssertionError('Expected verified breakpoint to be: %s. Found: %s.\nBreakpoint: %s' % (
                        verified, b['verified'], b))
        return response

    def write_set_exception_breakpoints(self, filters=None, exception_options=None):
        '''
        :param list(str) filters:
            A list with 'raised' or 'uncaught' entries.

        :param list(ExceptionOptions) exception_options:

        '''
        filters = filters or []
        assert set(filters).issubset(set(('raised', 'uncaught', 'userUnhandled')))

        exception_options = exception_options or []
        exception_options = [exception_option.to_dict() for exception_option in exception_options]

        arguments = pydevd_schema.SetExceptionBreakpointsArguments(filters=filters, exceptionOptions=exception_options)
        request = pydevd_schema.SetExceptionBreakpointsRequest(arguments)
        # : :type response: SetExceptionBreakpointsResponse
        response = self.wait_for_response(self.write_request(request))
        assert response.success

    def reset_sent_launch_or_attach(self):
        self._sent_launch_or_attach = False

    def _write_launch_or_attach(self, command, **arguments):
        assert not self._sent_launch_or_attach
        self._sent_launch_or_attach = True
        arguments['noDebug'] = False
        request = {'type': 'request', 'command': command, 'arguments': arguments, 'seq':-1}
        self.wait_for_response(self.write_request(request))

    def _auto_write_launch(self):
        self.write_launch(variablePresentation={
            "all": "hide",
            "protected": "inline",
        })

    def write_launch(self, **arguments):
        return self._write_launch_or_attach('launch', **arguments)

    def write_attach(self, **arguments):
        return self._write_launch_or_attach('attach', **arguments)

    def write_disconnect(self, wait_for_response=True, terminate_debugee=False):
        assert self._sent_launch_or_attach
        self._sent_launch_or_attach = False
        arguments = pydevd_schema.DisconnectArguments(terminateDebuggee=terminate_debugee)
        request = pydevd_schema.DisconnectRequest(arguments=arguments)
        self.write_request(request)
        if wait_for_response:
            self.wait_for_response(request)

    def get_stack_as_json_hit(self, thread_id, no_stack_frame=False):
        stack_trace_request = self.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=thread_id)))

        # : :type stack_trace_response: StackTraceResponse
        # : :type stack_trace_response_body: StackTraceResponseBody
        # : :type stack_frame: StackFrame
        stack_trace_response = self.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        if no_stack_frame:
            assert len(stack_trace_response_body.stackFrames) == 0
            frame_id = None
        else:
            assert len(stack_trace_response_body.stackFrames) > 0
            for stack_frame in stack_trace_response_body.stackFrames:
                assert stack_frame['id'] < MAX_EXPECTED_ID

            stack_frame = next(iter(stack_trace_response_body.stackFrames))
            frame_id = stack_frame['id']

        return _JsonHit(
            thread_id=thread_id, frame_id=frame_id, stack_trace_response=stack_trace_response)

    def get_variables_response(self, variables_reference, fmt=None, success=True):
        assert variables_reference < MAX_EXPECTED_ID
        variables_request = self.write_request(
            pydevd_schema.VariablesRequest(pydevd_schema.VariablesArguments(variables_reference, format=fmt)))
        variables_response = self.wait_for_response(variables_request)
        assert variables_response.success == success
        return variables_response

    def filter_return_variables(self, variables):
        ret = []
        for variable in variables:
            if variable['name'].startswith('(return)'):
                ret.append(variable)
        return ret

    def pop_variables_reference(self, lst):
        '''
        Modifies dicts in-place to remove the variablesReference and returns those (in the same order
        in which they were received).
        '''
        references = []
        for dct in lst:
            reference = dct.pop('variablesReference', None)
            if reference is not None:
                assert isinstance(reference, int_types)
                assert reference < MAX_EXPECTED_ID
            references.append(reference)
        return references

    def wait_for_continued_event(self):
        assert self.wait_for_json_message(ContinuedEvent).body.allThreadsContinued

    def write_continue(self, wait_for_response=True):
        continue_request = self.write_request(
            pydevd_schema.ContinueRequest(pydevd_schema.ContinueArguments('*')))

        if wait_for_response:
            # The continued event is received before the response.
            self.wait_for_continued_event()

            continue_response = self.wait_for_response(continue_request)
            assert continue_response.body.allThreadsContinued

    def write_pause(self):
        pause_request = self.write_request(
            pydevd_schema.PauseRequest(pydevd_schema.PauseArguments('*')))
        pause_response = self.wait_for_response(pause_request)
        assert pause_response.success

    def write_step_in(self, thread_id, target_id=None):
        arguments = pydevd_schema.StepInArguments(threadId=thread_id, targetId=target_id)
        self.wait_for_response(self.write_request(pydevd_schema.StepInRequest(arguments)))

    def write_step_next(self, thread_id, wait_for_response=True):
        next_request = self.write_request(
            pydevd_schema.NextRequest(pydevd_schema.NextArguments(thread_id)))
        if wait_for_response:
            self.wait_for_response(next_request)

    def write_step_out(self, thread_id, wait_for_response=True):
        stepout_request = self.write_request(
            pydevd_schema.StepOutRequest(pydevd_schema.StepOutArguments(thread_id)))
        if wait_for_response:
            self.wait_for_response(stepout_request)

    def write_set_variable(self, frame_variables_reference, name, value, success=True):
        set_variable_request = self.write_request(
            pydevd_schema.SetVariableRequest(pydevd_schema.SetVariableArguments(
                frame_variables_reference, name, value,
        )))
        set_variable_response = self.wait_for_response(set_variable_request)
        if set_variable_response.success != success:
            raise AssertionError(
                'Expected %s. Found: %s\nResponse: %s\n' % (
                    success, set_variable_response.success, set_variable_response.to_json()))
        return set_variable_response

    def get_name_to_scope(self, frame_id):
        scopes_request = self.write_request(pydevd_schema.ScopesRequest(
            pydevd_schema.ScopesArguments(frame_id)))

        scopes_response = self.wait_for_response(scopes_request)

        scopes = scopes_response.body.scopes
        name_to_scopes = dict((scope['name'], pydevd_schema.Scope(**scope)) for scope in scopes)

        assert len(scopes) == 2
        assert sorted(name_to_scopes.keys()) == ['Globals', 'Locals']
        assert not name_to_scopes['Locals'].expensive
        assert name_to_scopes['Locals'].presentationHint == 'locals'

        return name_to_scopes

    def get_step_in_targets(self, frame_id):
        request = self.write_request(pydevd_schema.StepInTargetsRequest(
            pydevd_schema.StepInTargetsArguments(frame_id)))

        # : :type response: StepInTargetsResponse
        response = self.wait_for_response(request)

        # : :type body: StepInTargetsResponseBody
        body = response.body
        targets = body.targets
        # : :type targets: List[StepInTarget]
        return targets

    def get_name_to_var(self, variables_reference):
        variables_response = self.get_variables_response(variables_reference)
        return dict((variable['name'], pydevd_schema.Variable(**variable)) for variable in variables_response.body.variables)

    def get_locals_name_to_var(self, frame_id):
        name_to_scope = self.get_name_to_scope(frame_id)

        return self.get_name_to_var(name_to_scope['Locals'].variablesReference)

    def get_globals_name_to_var(self, frame_id):
        name_to_scope = self.get_name_to_scope(frame_id)

        return self.get_name_to_var(name_to_scope['Globals'].variablesReference)

    def get_local_var(self, frame_id, var_name):
        ret = self.get_locals_name_to_var(frame_id)[var_name]
        assert ret.name == var_name
        return ret

    def get_global_var(self, frame_id, var_name):
        ret = self.get_globals_name_to_var(frame_id)[var_name]
        assert ret.name == var_name
        return ret

    def get_var(self, variables_reference, var_name=None, index=None):
        if var_name is not None:
            return self.get_name_to_var(variables_reference)[var_name]
        else:
            assert index is not None, 'Either var_name or index must be passed.'
            variables_response = self.get_variables_response(variables_reference)
            return pydevd_schema.Variable(**variables_response.body.variables[index])

    def write_set_debugger_property(
            self,
            dont_trace_start_patterns=None,
            dont_trace_end_patterns=None,
            multi_threads_single_notification=None,
            success=True
        ):
        dbg_request = self.write_request(
            pydevd_schema.SetDebuggerPropertyRequest(pydevd_schema.SetDebuggerPropertyArguments(
                dontTraceStartPatterns=dont_trace_start_patterns,
                dontTraceEndPatterns=dont_trace_end_patterns,
                multiThreadsSingleNotification=multi_threads_single_notification,
            )))
        response = self.wait_for_response(dbg_request)
        assert response.success == success
        return response

    def write_set_pydevd_source_map(self, source, pydevd_source_maps, success=True):
        dbg_request = self.write_request(
            pydevd_schema.SetPydevdSourceMapRequest(pydevd_schema.SetPydevdSourceMapArguments(
                source=source,
                pydevdSourceMaps=pydevd_source_maps,
            )))
        response = self.wait_for_response(dbg_request)
        assert response.success == success
        return response

    def write_initialize(self, success=True):
        arguments = InitializeRequestArguments(
            adapterID='pydevd_test_case',
        )
        response = self.wait_for_response(self.write_request(InitializeRequest(arguments)))
        assert response.success == success
        if success:
            process_id = response.body.kwargs['pydevd']['processId']
            assert isinstance(process_id, int)
        return response

    def write_authorize(self, access_token, success=True):
        from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdAuthorizeArguments
        from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdAuthorizeRequest
        arguments = PydevdAuthorizeArguments(
            debugServerAccessToken=access_token,
        )
        response = self.wait_for_response(self.write_request(PydevdAuthorizeRequest(arguments)))
        assert response.success == success
        return response

    def evaluate(self, expression, frameId=None, context=None, fmt=None, success=True, wait_for_response=True):
        '''
        :param wait_for_response:
            If True returns the response, otherwise returns the request.

        :returns EvaluateResponse
        '''
        eval_request = self.write_request(
            pydevd_schema.EvaluateRequest(pydevd_schema.EvaluateArguments(
                expression, frameId=frameId, context=context, format=fmt)))
        if wait_for_response:
            eval_response = self.wait_for_response(eval_request)
            assert eval_response.success == success
            return eval_response
        else:
            return eval_request

    def write_terminate(self):
        # Note: this currently terminates promptly, so, no answer is given.
        self.write_request(TerminateRequest(arguments=TerminateArguments()))

    def write_get_source(self, source_reference, success=True):
        response = self.wait_for_response(self.write_request(
            pydevd_schema.SourceRequest(pydevd_schema.SourceArguments(source_reference))))
        assert response.success == success
        return response


@pytest.mark.parametrize('scenario', ['basic', 'condition', 'hitCondition'])
def test_case_json_logpoints(case_setup_dap, scenario):
    with case_setup_dap.test_file('_debugger_case_change_breaks.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        break_2 = writer.get_line_index_with_content('break 2')
        break_3 = writer.get_line_index_with_content('break 3')
        if scenario == 'basic':
            json_facade.write_set_breakpoints(
                [break_2, break_3],
                line_to_info={
                    break_2: {'log_message': 'var {repr("_a")} is {_a}'}
            })
        elif scenario == 'condition':
            json_facade.write_set_breakpoints(
                [break_2, break_3],
                line_to_info={
                    break_2: {'log_message': 'var {repr("_a")} is {_a}', 'condition': 'True'}
            })
        elif scenario == 'hitCondition':
            json_facade.write_set_breakpoints(
                [break_2, break_3],
                line_to_info={
                    break_2: {'log_message': 'var {repr("_a")} is {_a}', 'hit_condition': '1'}
            })
        json_facade.write_make_initial_run()

        # Should only print, not stop on logpoints.

        # Just one hit at the end (break 3).
        json_facade.wait_for_thread_stopped(line=break_3)
        json_facade.write_continue()

        def accept_message(output_event):
            msg = output_event.body.output
            ctx = output_event.body.category

            if ctx == 'stdout':
                msg = msg.strip()
                return msg == "var '_a' is 2"

        messages = json_facade.mark_messages(OutputEvent, accept_message)
        if scenario == 'hitCondition':
            assert len(messages) == 1
        else:
            assert len(messages) == 2

        writer.finished_ok = True


def test_case_json_logpoint_and_step_failure_ok(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_hit_count.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        before_loop_line = writer.get_line_index_with_content('before loop line')
        for_line = writer.get_line_index_with_content('for line')
        print_line = writer.get_line_index_with_content('print line')
        json_facade.write_set_breakpoints(
            [before_loop_line, print_line],
            line_to_info={
                print_line: {'log_message': 'var {repr("_a")} is {_a}'}
        })
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=before_loop_line)

        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=for_line)

        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=print_line)

        json_facade.write_continue()

        writer.finished_ok = True


def test_case_json_logpoint_and_step_still_prints(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_hit_count.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        before_loop_line = writer.get_line_index_with_content('before loop line')
        print_line = writer.get_line_index_with_content('print line')
        json_facade.write_set_breakpoints(
            [before_loop_line, print_line],
            line_to_info={
                print_line: {'log_message': 'var {repr("i")} is {i}'}
        })
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=before_loop_line)

        for _i in range(4):
            # I.e.: even when stepping we should have the messages.
            json_facade.write_step_next(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step')

        json_facade.write_continue()

        def accept_last_output_message(output_event):
            return output_event.body.output.startswith("var 'i' is 9")

        json_facade.wait_for_json_message(OutputEvent, accept_last_output_message)

        def accept_message(output_event):
            return output_event.body.output.startswith("var 'i' is ")

        assert len(json_facade.mark_messages(OutputEvent, accept_message)) == 10

        writer.finished_ok = True


def test_case_json_hit_count_and_step(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_hit_count.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        for_line = writer.get_line_index_with_content('for line')
        print_line = writer.get_line_index_with_content('print line')
        json_facade.write_set_breakpoints(
            [print_line],
            line_to_info={
                print_line: {'hit_condition': '5'}
        })
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=print_line)
        i_local_var = json_facade.get_local_var(json_hit.frame_id, 'i')  # : :type i_local_var: pydevd_schema.Variable
        assert i_local_var.value == '4'

        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=for_line)

        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=print_line)

        json_facade.write_continue()

        writer.finished_ok = True


def test_case_json_hit_condition_error(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_hit_count.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        bp = writer.get_line_index_with_content('before loop line')
        json_facade.write_set_breakpoints(
            [bp],
            line_to_info={
                bp: {'condition': 'range.range.range'}
        })
        json_facade.write_make_initial_run()

        def accept_message(msg):
            if msg.body.category == 'important':
                if 'Error while evaluating expression in conditional breakpoint' in msg.body.output:
                    return True
            return False

        json_facade.wait_for_json_message(OutputEvent, accept_message=accept_message)

        # In the dap mode we skip suspending when an error happens in conditional exceptions.
        # json_facade.wait_for_thread_stopped(line=bp)
        # json_facade.write_continue()

        writer.finished_ok = True


def test_case_process_event(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_change_breaks.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        assert len(json_facade.mark_messages(ProcessEvent)) == 1
        json_facade.write_make_initial_run()
        writer.finished_ok = True


def test_case_json_change_breaks(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_change_breaks.py') as writer:
        json_facade = JsonFacade(writer)

        break1_line = writer.get_line_index_with_content('break 1')
        # Note: we can only write breakpoints after the launch is received.
        json_facade.write_set_breakpoints(break1_line, success=False, send_launch_if_needed=False)

        json_facade.write_launch()
        json_facade.write_set_breakpoints(break1_line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(line=break1_line)
        json_facade.write_set_breakpoints([])
        json_facade.write_continue()

        writer.finished_ok = True


def test_case_handled_exception_no_break_on_generator(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_ignore_exceptions.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['raised'])
        json_facade.write_make_initial_run()

        writer.finished_ok = True


def test_case_throw_exc_reason(case_setup_dap):

    def check_test_suceeded_msg(self, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        assert "raise RuntimeError('TEST SUCEEDED')" in stderr
        assert "raise RuntimeError from e" in stderr
        assert "raise Exception('another while handling')" in stderr

    with case_setup_dap.test_file(
            '_debugger_case_raise_with_cause.py',
            EXPECTED_RETURNCODE=1,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content('raise RuntimeError from e'))

        exc_info_request = json_facade.write_request(
            pydevd_schema.ExceptionInfoRequest(pydevd_schema.ExceptionInfoArguments(json_hit.thread_id)))
        exc_info_response = json_facade.wait_for_response(exc_info_request)

        stack_frames = json_hit.stack_trace_response.body.stackFrames
        # Note that the additional context doesn't really appear in the stack
        # frames, only in the details.
        assert [x['name'] for x in stack_frames] == [
            'foobar',
            '<module>',
            '[Chained Exc: another while handling] foobar',
            '[Chained Exc: another while handling] handle',
            '[Chained Exc: TEST SUCEEDED] foobar',
            '[Chained Exc: TEST SUCEEDED] method',
            '[Chained Exc: TEST SUCEEDED] method2',
        ]

        body = exc_info_response.body
        assert body.exceptionId.endswith('RuntimeError')
        assert body.description == 'another while handling'
        assert normcase(body.details.kwargs['source']) == normcase(writer.TEST_FILE)

        # Check that we have all the lines (including the cause/context) in the stack trace.
        import re
        lines_and_names = re.findall(r',\sline\s(\d+),\sin\s(\[Chained Exception\]\s)?([\w|<|>]+)', body.details.stackTrace)
        assert lines_and_names == [
            ('16', '', 'foobar'),
            ('6', '', 'method'),
            ('2', '', 'method2'),
            ('18', '', 'foobar'),
            ('10', '', 'handle'),
            ('20', '', 'foobar'),
            ('23', '', '<module>'),
        ], 'Did not find the expected names in:\n%s' % (body.details.stackTrace,)

        json_facade.write_continue()

        writer.finished_ok = True


def test_case_throw_exc_reason_shown(case_setup_dap):

    def check_test_suceeded_msg(self, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        assert "raise Exception('TEST SUCEEDED') from e" in stderr
        assert "{}['foo']" in stderr
        assert "KeyError: 'foo'" in stderr

    with case_setup_dap.test_file(
            '_debugger_case_raise_with_cause_msg.py',
            EXPECTED_RETURNCODE=1,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content("raise Exception('TEST SUCEEDED') from e"))

        exc_info_request = json_facade.write_request(
            pydevd_schema.ExceptionInfoRequest(pydevd_schema.ExceptionInfoArguments(json_hit.thread_id)))
        exc_info_response = json_facade.wait_for_response(exc_info_request)

        stack_frames = json_hit.stack_trace_response.body.stackFrames
        # Note that the additional context doesn't really appear in the stack
        # frames, only in the details.
        assert [x['name'] for x in stack_frames] == [
            'method',
            '<module>',
            "[Chained Exc: 'foo'] method",
            "[Chained Exc: 'foo'] method2",
        ]

        body = exc_info_response.body
        assert body.exceptionId == 'Exception'
        assert body.description == 'TEST SUCEEDED'
        if IS_PY311_OR_GREATER:
            assert '^^^^' in body.details.stackTrace
        assert normcase(body.details.kwargs['source']) == normcase(writer.TEST_FILE)

        # Check that we have the exception cause in the stack trace.
        assert "KeyError: 'foo'" in body.details.stackTrace

        json_facade.write_continue()

        writer.finished_ok = True


def test_case_handled_exception_breaks(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_exceptions.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['raised'])
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content('raise indexerror line'))
        json_facade.write_continue()

        json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content('reraise on method2'))

        # Clear so that the last one is not hit.
        json_facade.write_set_exception_breakpoints([])
        json_facade.write_continue()

        writer.finished_ok = True


def _check_current_line(json_hit, current_line):
    if not isinstance(current_line, (list, tuple)):
        current_line = (current_line,)
    for frame in json_hit.stack_trace_response.body.stackFrames:
        if '(Current frame)' in frame['name']:
            if frame['line'] not in current_line:
                rep = json.dumps(json_hit.stack_trace_response.body.stackFrames, indent=4)
                raise AssertionError('Expected: %s to be one of: %s\nFrames:\n%s.' % (
                    frame['line'],
                    current_line,
                    rep
                ))

            break
    else:
        rep = json.dumps(json_hit.stack_trace_response.body.stackFrames, indent=4)
        raise AssertionError('Could not find (Current frame) in any frame name in: %s.' % (
            rep))


@pytest.mark.parametrize('stop', [False, True])
def test_case_user_unhandled_exception(case_setup_dap, stop):

    def get_environ(self):
        env = os.environ.copy()

        # Note that we put the working directory in the project roots to check that when expanded
        # the relative file that doesn't exist is still considered a library file.
        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE) + os.pathsep + os.path.abspath('.')
        return env

    if stop:
        target = '_debugger_case_user_unhandled.py'
    else:
        target = '_debugger_case_user_unhandled2.py'
    with case_setup_dap.test_file(target, get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['userUnhandled'])
        json_facade.write_make_initial_run()

        if stop:
            json_hit = json_facade.wait_for_thread_stopped(
                reason='exception', line=writer.get_line_index_with_content('raise here'), file=target)
            _check_current_line(json_hit, writer.get_line_index_with_content('stop here'))

            json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
@pytest.mark.parametrize('stop', [False, True])
def test_case_user_unhandled_exception_coroutine(case_setup_dap, stop):
    if stop:
        target = 'my_code/my_code_coroutine_user_unhandled.py'
    else:
        target = 'my_code/my_code_coroutine_user_unhandled_no_stop.py'
    basename = os.path.basename(target)

    def additional_output_checks(writer, stdout, stderr):
        if stop:
            assert 'raise RuntimeError' in stderr
        else:
            assert 'raise RuntimeError' not in stderr

    with case_setup_dap.test_file(
            target,
            EXPECTED_RETURNCODE=1 if stop else 0,
            additional_output_checks=additional_output_checks
        ) as writer:
        json_facade = JsonFacade(writer)

        not_my_code_dir = debugger_unittest._get_debugger_test_file('not_my_code')
        json_facade.write_launch(
            rules=[
                {'path': not_my_code_dir, 'include':False},
            ]
        )
        json_facade.write_set_exception_breakpoints(['userUnhandled'])
        json_facade.write_make_initial_run()

        if stop:
            stop_line = writer.get_line_index_with_content('stop here 1')
            current_line = stop_line

            json_hit = json_facade.wait_for_thread_stopped(
                reason='exception', line=stop_line, file=basename)
            _check_current_line(json_hit, current_line)

            json_facade.write_continue()

            current_line = writer.get_line_index_with_content('stop here 2')
            json_hit = json_facade.wait_for_thread_stopped(
                reason='exception', line=stop_line, file=basename)
            _check_current_line(json_hit, current_line)

            json_facade.write_continue()

            current_line = (
                writer.get_line_index_with_content('stop here 3a'),
                writer.get_line_index_with_content('stop here 3b'),
            )

            json_hit = json_facade.wait_for_thread_stopped(
                reason='exception', line=stop_line, file=basename)
            _check_current_line(json_hit, current_line)

            json_facade.write_continue()

        writer.finished_ok = True


def test_case_user_unhandled_exception_dont_stop(case_setup_dap):

    with case_setup_dap.test_file(
            'my_code/my_code_exception_user_unhandled.py',) as writer:
        json_facade = JsonFacade(writer)

        not_my_code_dir = debugger_unittest._get_debugger_test_file('not_my_code')
        json_facade.write_launch(
            debugStdLib=True,
            rules=[
                {'path': not_my_code_dir, 'include':False},
            ]
        )

        json_facade.write_set_exception_breakpoints(['userUnhandled'])
        json_facade.write_make_initial_run()

        writer.finished_ok = True


def test_case_user_unhandled_exception_stop_on_yield(case_setup_dap, pyfile):

    @pyfile
    def case_error_on_yield():

        def on_yield():
            yield
            raise AssertionError()  # raise here

        try:
            for _ in on_yield():  # stop here
                pass
        except:
            print('TEST SUCEEDED!')
            raise

    def get_environ(self):
        env = os.environ.copy()

        # Note that we put the working directory in the project roots to check that when expanded
        # the relative file that doesn't exist is still considered a library file.
        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE) + os.pathsep + os.path.abspath('.')
        return env

    def additional_output_checks(writer, stdout, stderr):
        assert 'raise AssertionError' in stderr

    with case_setup_dap.test_file(
            case_error_on_yield,
            get_environ=get_environ,
            EXPECTED_RETURNCODE=1,
            additional_output_checks=additional_output_checks) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['userUnhandled'])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content('raise here'), file=case_error_on_yield)
        _check_current_line(json_hit, writer.get_line_index_with_content('stop here'))

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.parametrize('target', [
    'absolute',
    'relative',
    ])
@pytest.mark.parametrize('just_my_code', [
    True,
    False,
    ])
def test_case_unhandled_exception_just_my_code(case_setup_dap, target, just_my_code):

    def check_test_suceeded_msg(writer, stdout, stderr):
        # Don't call super (we have an unhandled exception in the stack trace).
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'call_exception_in_exec()' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    def get_environ(self):
        env = os.environ.copy()

        # Note that we put the working directory in the project roots to check that when expanded
        # the relative file that doesn't exist is still considered a library file.
        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE) + os.pathsep + os.path.abspath('.')
        return env

    def update_command_line_args(writer, args):
        ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
        if target == 'absolute':
            if sys.platform == 'win32':
                ret.append('c:/temp/folder/my_filename.pyx')
            else:
                ret.append('/temp/folder/my_filename.pyx')

        elif target == 'relative':
            ret.append('folder/my_filename.pyx')

        else:
            raise AssertionError('Unhandled case: %s' % (target,))
        return args

    target_filename = '_debugger_case_unhandled_just_my_code.py'
    with case_setup_dap.test_file(
            target_filename,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
            update_command_line_args=update_command_line_args,
            get_environ=get_environ,
            EXPECTED_RETURNCODE=1,
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=just_my_code)
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(reason='exception')
        frames = json_hit.stack_trace_response.body.stackFrames
        if just_my_code:
            assert len(frames) == 1
            assert frames[0]['source']['path'].endswith(target_filename)
        else:
            assert len(frames) > 1
            assert frames[0]['source']['path'].endswith('my_filename.pyx')

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Python 3.6 onwards required for test.')
def test_case_stop_async_iteration_exception(case_setup_dap):

    def get_environ(self):
        env = os.environ.copy()
        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE) + os.pathsep + os.path.abspath('.')
        return env

    with case_setup_dap.test_file(
            '_debugger_case_stop_async_iteration.py',
            get_environ=get_environ,
        ) as writer:
        json_facade = JsonFacade(writer)

        # We don't want to hit common library exceptions here.
        json_facade.write_launch(justMyCode=True)

        json_facade.write_set_exception_breakpoints(['raised'])
        json_facade.write_make_initial_run()

        # Just making sure that no exception breakpoints are hit.

        writer.finished_ok = True


@pytest.mark.parametrize('target_file', [
    '_debugger_case_unhandled_exceptions.py',
    '_debugger_case_unhandled_exceptions_custom.py',
    ])
def test_case_unhandled_exception(case_setup_dap, target_file):

    def check_test_suceeded_msg(writer, stdout, stderr):
        # Don't call super (we have an unhandled exception in the stack trace).
        return 'TEST SUCEEDED' in ''.join(stdout) and 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'raise MyError' not in stderr and 'raise Exception' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup_dap.test_file(
            target_file,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
            EXPECTED_RETURNCODE=1,
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        line_in_thread1 = writer.get_line_index_with_content('in thread 1')
        line_in_thread2 = writer.get_line_index_with_content('in thread 2')
        line_in_main = writer.get_line_index_with_content('in main')
        json_facade.wait_for_thread_stopped(
            reason='exception', line=(line_in_thread1, line_in_thread2), file=target_file)
        json_facade.write_continue()

        json_facade.wait_for_thread_stopped(
            reason='exception', line=(line_in_thread1, line_in_thread2), file=target_file)
        json_facade.write_continue()

        json_facade.wait_for_thread_stopped(
            reason='exception', line=line_in_main, file=target_file)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('target_file', [
    '_debugger_case_unhandled_exceptions_generator.py',
    '_debugger_case_unhandled_exceptions_listcomp.py',
    ])
def test_case_unhandled_exception_generator(case_setup_dap, target_file):

    def check_test_suceeded_msg(writer, stdout, stderr):
        # Don't call super (we have an unhandled exception in the stack trace).
        return 'TEST SUCEEDED' in ''.join(stdout) and 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'ZeroDivisionError' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup_dap.test_file(
            target_file,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
            EXPECTED_RETURNCODE=1,
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        line_in_main = writer.get_line_index_with_content('exc line')

        json_hit = json_facade.wait_for_thread_stopped(
            reason='exception', line=line_in_main, file=target_file)
        frames = json_hit.stack_trace_response.body.stackFrames
        json_facade.write_continue()

        if 'generator' in target_file:
            expected_frame_names = ['<genexpr>', 'f', '<module>']
        else:
            expected_frame_names = ['<listcomp>', 'f', '<module>']

        frame_names = [f['name'] for f in frames]
        assert frame_names == expected_frame_names

        writer.finished_ok = True


def test_case_sys_exit_unhandled_exception(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_sysexit.py', EXPECTED_RETURNCODE=1) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        break_line = writer.get_line_index_with_content('sys.exit(1)')
        json_facade.wait_for_thread_stopped(
            reason='exception', line=break_line)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('break_on_system_exit_zero', [True, False])
@pytest.mark.parametrize('target', ['_debugger_case_sysexit_0.py', '_debugger_case_sysexit_none.py'])
def test_case_sys_exit_0_unhandled_exception(case_setup_dap, break_on_system_exit_zero, target):

    with case_setup_dap.test_file(target, EXPECTED_RETURNCODE=0) as writer:
        json_facade = JsonFacade(writer)
        kwargs = {}
        if break_on_system_exit_zero:
            kwargs = {'breakOnSystemExitZero': True}
        json_facade.write_launch(**kwargs)
        json_facade.write_set_exception_breakpoints(['uncaught'])
        json_facade.write_make_initial_run()

        break_line = writer.get_line_index_with_content('sys.exit(')
        if break_on_system_exit_zero:
            json_facade.wait_for_thread_stopped(
                reason='exception', line=break_line)
            json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('break_on_system_exit_zero', [True, False])
def test_case_sys_exit_0_handled_exception(case_setup_dap, break_on_system_exit_zero):

    with case_setup_dap.test_file('_debugger_case_sysexit_0.py', EXPECTED_RETURNCODE=0) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            debugOptions=['BreakOnSystemExitZero'] if break_on_system_exit_zero else [],
        )
        json_facade.write_set_exception_breakpoints(['raised'])
        json_facade.write_make_initial_run()

        break_line = writer.get_line_index_with_content('sys.exit(0)')
        break_main_line = writer.get_line_index_with_content('call_main_line')
        if break_on_system_exit_zero:
            json_facade.wait_for_thread_stopped(
                reason='exception', line=break_line)
            json_facade.write_continue()

            json_facade.wait_for_thread_stopped(
                reason='exception', line=break_main_line)
            json_facade.write_continue()

        writer.finished_ok = True


def test_case_handled_exception_breaks_by_type(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_exceptions.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_exception_breakpoints(exception_options=[
            ExceptionOptions(breakMode='always', path=[
                {'names': ['Python Exceptions']},
                {'names': ['IndexError']},
            ])
        ])
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(
            reason='exception', line=writer.get_line_index_with_content('raise indexerror line'))

        # Deal only with RuntimeErorr now.
        json_facade.write_set_exception_breakpoints(exception_options=[
            ExceptionOptions(breakMode='always', path=[
                {'names': ['Python Exceptions']},
                {'names': ['RuntimeError']},
            ])
        ])

        json_facade.write_continue()

        writer.finished_ok = True


def test_case_json_protocol(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_print.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        break_line = writer.get_line_index_with_content('Break here')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_json_message(ThreadEvent, lambda event: event.body.reason == 'started')

        json_facade.wait_for_thread_stopped(line=break_line)

        # : :type response: ThreadsResponse
        response = json_facade.write_list_threads()
        assert len(response.body.threads) == 1
        assert next(iter(response.body.threads))['name'] == 'MainThread'

        # Removes breakpoints and proceeds running.
        json_facade.write_disconnect()

        writer.finished_ok = True


def test_case_started_exited_threads_protocol(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_thread_started_exited.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        break_line = writer.get_line_index_with_content('Break here')
        json_facade.write_set_breakpoints(break_line)

        json_facade.write_make_initial_run()

        _stopped_event = json_facade.wait_for_json_message(StoppedEvent)
        started_events = json_facade.mark_messages(ThreadEvent, lambda x: x.body.reason == 'started')
        exited_events = json_facade.mark_messages(ThreadEvent, lambda x: x.body.reason == 'exited')
        assert len(started_events) == 4
        assert len(exited_events) == 3  # Main is still running.
        json_facade.write_continue()

        writer.finished_ok = True


def test_case_path_translation_not_skipped(case_setup_dap):
    import site
    sys_folder = None
    if hasattr(site, 'getusersitepackages'):
        sys_folder = site.getusersitepackages()

    if not sys_folder and hasattr(site, 'getsitepackages'):
        sys_folder = site.getsitepackages()

    if not sys_folder:
        sys_folder = sys.prefix

    if isinstance(sys_folder, (list, tuple)):
        sys_folder = next(iter(sys_folder))

    with case_setup_dap.test_file('my_code/my_code.py') as writer:
        json_facade = JsonFacade(writer)

        # We need to set up path mapping to enable source references.
        my_code = debugger_unittest._get_debugger_test_file('my_code')

        json_facade.write_launch(
            justMyCode=False,
            pathMappings=[{
                'localRoot': sys_folder,
                'remoteRoot': my_code,
            }]
        )

        bp_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(
            bp_line,
            filename=os.path.join(sys_folder, 'my_code.py'),
        )
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=bp_line)

        stack_frame = json_hit.stack_trace_response.body.stackFrames[-1]
        assert stack_frame['source']['path'] == os.path.join(sys_folder, 'my_code.py')
        for stack_frame in json_hit.stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0
        json_facade.write_continue()

        writer.finished_ok = True


def test_case_exclude_double_step(case_setup_dap):
    with case_setup_dap.test_file('my_code/my_code_double_step.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            justMyCode=False,  # i.e.: exclude through rules and not my code
            rules=[
                {'path': '**/other_noop.py', 'include':False},
            ]
        )

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=break_line)
        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', file='my_code_double_step.py', line=break_line + 1)

        json_facade.write_continue()
        writer.finished_ok = True


def test_case_update_rules(case_setup_dap):
    with case_setup_dap.test_file('my_code/my_code.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            rules=[
                {'path': '**/other.py', 'include':False},
            ]
        )

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_json_message(ThreadEvent, lambda event: event.body.reason == 'started')

        json_hit = json_facade.wait_for_thread_stopped(line=break_line)
        json_facade.reset_sent_launch_or_attach()
        json_facade.write_launch(
            rules=[
                {'path': '**/other.py', 'include':True},
            ]
        )
        json_facade.write_step_in(json_hit.thread_id)
        # Not how we stoppen in the file that wasn't initially included.
        json_hit = json_facade.wait_for_thread_stopped('step', name='call_me_back1')

        json_facade.reset_sent_launch_or_attach()
        json_facade.write_launch(
            rules=[
                {'path': '**/other.py', 'include':False},
            ]
        )
        json_facade.write_step_in(json_hit.thread_id)
        # Not how we go back to the callback and not to the `call_me_back1` because
        # `call_me_back1` is now excluded again.
        json_hit = json_facade.wait_for_thread_stopped('step', name='callback1')

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.parametrize("custom_setup", [
    'set_exclude_launch_module_full',
    'set_exclude_launch_module_prefix',
    'set_exclude_launch_path_match_filename',
    'set_exclude_launch_path_match_folder',
    'set_just_my_code',
    'set_just_my_code_and_include',
])
def test_case_skipping_filters(case_setup_dap, custom_setup):
    with case_setup_dap.test_file('my_code/my_code.py') as writer:
        json_facade = JsonFacade(writer)

        expect_just_my_code = False
        if custom_setup == 'set_exclude_launch_path_match_filename':
            json_facade.write_launch(
                justMyCode=False,
                rules=[
                    {'path': '**/other.py', 'include':False},
                ]
            )

        elif custom_setup == 'set_exclude_launch_path_match_folder':
            not_my_code_dir = debugger_unittest._get_debugger_test_file('not_my_code')
            json_facade.write_launch(
                debugStdLib=True,
                rules=[
                    {'path': not_my_code_dir, 'include':False},
                ]
            )

            other_filename = os.path.join(not_my_code_dir, 'other.py')
            response = json_facade.write_set_breakpoints(1, filename=other_filename, verified=False)
            assert response.body.breakpoints == [
                {'verified': False, 'id': 0, 'message': 'Breakpoint in file excluded by filters.', 'source': {'path': other_filename}, 'line': 1}]
            # Note: there's actually a use-case where we'd hit that breakpoint even if it was excluded
            # by filters, so, we must actually clear it afterwards (the use-case is that when we're
            # stepping into the context with the breakpoint we wouldn't skip it).
            json_facade.write_set_breakpoints([], filename=other_filename)

            other_filename = os.path.join(not_my_code_dir, 'file_that_does_not_exist.py')
            response = json_facade.write_set_breakpoints(1, filename=other_filename, verified=False)
            assert response.body.breakpoints == [
                {'verified': False, 'id': 1, 'message': 'Breakpoint in file that does not exist.', 'source': {'path': other_filename}, 'line': 1}]

        elif custom_setup == 'set_exclude_launch_module_full':
            json_facade.write_launch(
                debugOptions=['DebugStdLib'],
                rules=[
                    {'module': 'not_my_code.other', 'include':False},
                ]
            )

        elif custom_setup == 'set_exclude_launch_module_prefix':
            json_facade.write_launch(
                debugOptions=['DebugStdLib'],
                rules=[
                    {'module': 'not_my_code', 'include':False},
                ]
            )

        elif custom_setup == 'set_just_my_code':
            expect_just_my_code = True
            writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
            json_facade.write_launch(debugOptions=[])

            not_my_code_dir = debugger_unittest._get_debugger_test_file('not_my_code')
            other_filename = os.path.join(not_my_code_dir, 'other.py')
            response = json_facade.write_set_breakpoints(
                33, filename=other_filename, verified=False, expected_lines_in_response=[14])
            assert response.body.breakpoints == [{
                'verified': False,
                'id': 0,
                'message': 'Breakpoint in file excluded by filters.\nNote: may be excluded because of \"justMyCode\" option (default == true).Try setting \"justMyCode\": false in the debug configuration (e.g., launch.json).\n',
                'source': {'path': other_filename},
                'line': 14
            }]
        elif custom_setup == 'set_just_my_code_and_include':
            expect_just_my_code = True
            # I.e.: nothing in my_code (add it with rule).
            writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('launch')])
            json_facade.write_launch(
                debugOptions=[],
                rules=[
                    {'module': '__main__', 'include':True},
                ]
            )

        else:
            raise AssertionError('Unhandled: %s' % (custom_setup,))

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_json_message(ThreadEvent, lambda event: event.body.reason == 'started')

        json_hit = json_facade.wait_for_thread_stopped(line=break_line)

        json_facade.write_step_in(json_hit.thread_id)

        json_hit = json_facade.wait_for_thread_stopped('step', name='callback1')

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: 'Frame skipped from debugging during step-in.' in output_event.body.output)
        assert len(messages) == 1
        body = next(iter(messages)).body
        found_just_my_code = 'Note: may have been skipped because of \"justMyCode\" option (default == true)' in body.output

        assert found_just_my_code == expect_just_my_code
        assert body.category == 'important'

        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='callback2')

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='callback1')

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='<module>')

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='<module>')

        json_facade.write_step_next(json_hit.thread_id)

        if IS_JYTHON:
            json_facade.write_continue()

        # Check that it's sent only once.
        assert len(json_facade.mark_messages(
            OutputEvent, lambda output_event: 'Frame skipped from debugging during step-in.' in output_event.body.output)) == 0

        writer.finished_ok = True


def test_case_completions_json(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_completions.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        first_hit = None
        for i in range(2):
            json_hit = json_facade.wait_for_thread_stopped()

            json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
            if i == 0:
                first_hit = json_hit

            completions_arguments = pydevd_schema.CompletionsArguments(
                'dict.', 6, frameId=json_hit.frame_id, line=0)
            completions_request = json_facade.write_request(
                pydevd_schema.CompletionsRequest(completions_arguments))

            response = json_facade.wait_for_response(completions_request)
            assert response.success
            labels = [x['label'] for x in response.body.targets]
            assert set(labels).issuperset(set(['__contains__', 'items', 'keys', 'values']))

            completions_arguments = pydevd_schema.CompletionsArguments(
                'dict.item', 10, frameId=json_hit.frame_id)
            completions_request = json_facade.write_request(
                pydevd_schema.CompletionsRequest(completions_arguments))

            response = json_facade.wait_for_response(completions_request)
            assert response.success
            if IS_JYTHON:
                assert response.body.targets == [
                    {'start': 5, 'length': 4, 'type': 'keyword', 'label': 'items'}]
            else:
                assert response.body.targets == [
                    {'start': 5, 'length': 4, 'type': 'function', 'label': 'items'}]

            if i == 1:
                # Check with a previously existing frameId.
                assert first_hit.frame_id != json_hit.frame_id
                completions_arguments = pydevd_schema.CompletionsArguments(
                    'dict.item', 10, frameId=first_hit.frame_id)
                completions_request = json_facade.write_request(
                    pydevd_schema.CompletionsRequest(completions_arguments))

                response = json_facade.wait_for_response(completions_request)
                assert not response.success
                assert response.message == 'Thread to get completions seems to have resumed already.'

                # Check with a never frameId which never existed.
                completions_arguments = pydevd_schema.CompletionsArguments(
                    'dict.item', 10, frameId=99999)
                completions_request = json_facade.write_request(
                    pydevd_schema.CompletionsRequest(completions_arguments))

                response = json_facade.wait_for_response(completions_request)
                assert not response.success
                assert response.message.startswith('Wrong ID sent from the client:')

            json_facade.write_continue()

        writer.finished_ok = True


def test_modules(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break 2 here'))
        json_facade.write_make_initial_run()

        stopped_event = json_facade.wait_for_json_message(StoppedEvent)
        thread_id = stopped_event.body.threadId

        json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=thread_id)))

        json_facade.wait_for_json_message(ModuleEvent)

        # : :type response: ModulesResponse
        # : :type modules_response_body: ModulesResponseBody
        response = json_facade.wait_for_response(json_facade.write_request(
            pydevd_schema.ModulesRequest(pydevd_schema.ModulesArguments())))
        modules_response_body = response.body
        assert len(modules_response_body.modules) == 1
        module = next(iter(modules_response_body.modules))
        assert module['name'] == '__main__'
        assert module['path'].endswith('_debugger_case_local_variables.py')

        json_facade.write_continue()
        writer.finished_ok = True


def test_dict_ordered(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_odict.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = variables_response.body.variables
        for dct in variables_references:
            if dct['name'] == 'odict':
                break
        else:
            raise AssertionError('Expected to find "odict".')
        ref = dct['variablesReference']

        assert isinstance(ref, int_types)
        # : :type variables_response: VariablesResponse

        variables_response = json_facade.get_variables_response(ref)
        assert [(d['name'], d['value']) for d in variables_response.body.variables if (not d['name'].startswith('_OrderedDict')) and (d['name'] not in DAPGrouper.SCOPES_SORTED)] == [
            ('4', "'first'"), ('3', "'second'"), ('2', "'last'"), (GENERATED_LEN_ATTR_NAME, '3')]

        json_facade.write_continue()
        writer.finished_ok = True


def test_dict_contents(case_setup_dap, pyfile):

    @pyfile
    def check():
        dct = {'a': 1, '_b_': 2, '__c__': 3}
        print('TEST SUCEEDED')  # break here

    with case_setup_dap.test_file(check) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = variables_response.body.variables
        for dct in variables_references:
            if dct['name'] == 'dct':
                break
        else:
            raise AssertionError('Expected to find "dct".')
        ref = dct['variablesReference']

        assert isinstance(ref, int_types)
        # : :type variables_response: VariablesResponse

        variables_response = json_facade.get_variables_response(ref)
        variable_names = set(v['name'] for v in variables_response.body.variables)
        for n in ("'a'", "'_b_'", "'__c__'", 'len()'):
            assert n in variable_names

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Putting unicode on frame vars does not work on Jython.')
def test_stack_and_variables_dict(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break 2 here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = json_facade.pop_variables_reference(variables_response.body.variables)
        dict_variable_reference = variables_references[2]
        assert isinstance(dict_variable_reference, int_types)
        # : :type variables_response: VariablesResponse

        expected_unicode = {
            'name': u'\u16A0',
            'value': "'\u16a1'",
            'type': 'str',
            'presentationHint': {'attributes': ['rawString']},
            'evaluateName': u'\u16A0',
        }
        assert variables_response.body.variables == [
            {'name': 'variable_for_test_1', 'value': '10', 'type': 'int', 'evaluateName': 'variable_for_test_1'},
            {'name': 'variable_for_test_2', 'value': '20', 'type': 'int', 'evaluateName': 'variable_for_test_2'},
            {'name': 'variable_for_test_3', 'value': "{'a': 30, 'b': 20}", 'type': 'dict', 'evaluateName': 'variable_for_test_3'},
            expected_unicode
        ]

        variables_response = json_facade.get_variables_response(dict_variable_reference)
        check = [x for x in variables_response.body.variables if x['name'] not in DAPGrouper.SCOPES_SORTED]
        assert check == [
            {'name': "'a'", 'value': '30', 'type': 'int', 'evaluateName': "variable_for_test_3['a']", 'variablesReference': 0 },
            {'name': "'b'", 'value': '20', 'type': 'int', 'evaluateName': "variable_for_test_3['b']", 'variablesReference': 0},
            {'name': GENERATED_LEN_ATTR_NAME, 'value': '2', 'type': 'int', 'evaluateName': 'len(variable_for_test_3)', 'variablesReference': 0, 'presentationHint': {'attributes': ['readOnly']}}
        ]

        json_facade.write_continue()
        writer.finished_ok = True


def test_variables_with_same_name(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_variables_with_same_name.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = json_facade.pop_variables_reference(variables_response.body.variables)
        dict_variable_reference = variables_references[0]
        assert isinstance(dict_variable_reference, int_types)
        # : :type variables_response: VariablesResponse

        assert variables_response.body.variables == [
            {'name': 'td', 'value': "{foo: 'bar', gad: 'zooks', foo: 'bur'}", 'type': 'dict', 'evaluateName': 'td'}
        ]

        dict_variables_response = json_facade.get_variables_response(dict_variable_reference)
        # Note that we don't have the evaluateName because it's not possible to create a key
        # from the user object to actually get its value from the dict in this case.
        variables = dict_variables_response.body.variables[:]

        found_foo = False
        found_foo_with_id = False
        for v in variables:
            if v['name'].startswith('foo'):
                if not found_foo:
                    assert v['name'] == 'foo'
                    found_foo = True
                else:
                    assert v['name'].startswith('foo (id: ')
                    v['name'] = 'foo'
                    found_foo_with_id = True

        assert found_foo
        assert found_foo_with_id

        def compute_key(entry):
            return (entry['name'], entry['value'])

        # Sort because the order may be different on Py2/Py3.
        assert sorted(variables, key=compute_key) == sorted([
            {
                'name': 'foo',
                'value': "'bar'",
                'type': 'str',
                'variablesReference': 0,
                'presentationHint': {'attributes': ['rawString']}
            },

            {
                # 'name': 'foo (id: 2699272929584)', In the code above we changed this
                # to 'name': 'foo' for the comparisson.
                'name': 'foo',
                'value': "'bur'",
                'type': 'str',
                'variablesReference': 0,
                'presentationHint': {'attributes': ['rawString']}
            },

            {
                'name': 'gad',
                'value': "'zooks'",
                'type': 'str',
                'variablesReference': 0,
                'presentationHint': {'attributes': ['rawString']}
            },

            {
                'name': 'len()',
                'value': '3',
                'type': 'int',
                'evaluateName': 'len(td)',
                'variablesReference': 0,
                'presentationHint': {'attributes': ['readOnly']}
            },
        ], key=compute_key)

        json_facade.write_continue()
        writer.finished_ok = True


def test_hasattr_failure(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_hasattr_crash.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        for variable in variables_response.body.variables:
            if variable['evaluateName'] == 'obj':
                break
        else:
            raise AssertionError('Did not find "obj" in %s' % (variables_response.body.variables,))

        evaluate_response = json_facade.evaluate('obj', json_hit.frame_id, context='hover')
        evaluate_response_body = evaluate_response.body.to_dict()
        assert evaluate_response_body['result'] == 'An exception was raised: RuntimeError()'

        json_facade.evaluate('not_there', json_hit.frame_id, context='hover', success=False)
        json_facade.evaluate('not_there', json_hit.frame_id, context='watch', success=False)

        json_facade.write_continue()

        writer.finished_ok = True


def test_getattr_warning(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_warnings.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        for variable in variables_response.body.variables:
            if variable['evaluateName'] == 'obj':
                break
        else:
            raise AssertionError('Did not find "obj" in %s' % (variables_response.body.variables,))

        json_facade.evaluate('obj', json_hit.frame_id, context='hover')
        json_facade.evaluate('not_there', json_hit.frame_id, context='hover', success=False)
        json_facade.evaluate('not_there', json_hit.frame_id, context='watch', success=False)

        json_facade.write_continue()

        # i.e.: the test will fail if anything is printed to stderr!
        writer.finished_ok = True


def test_warning_on_repl(case_setup_dap):

    def additional_output_checks(writer, stdout, stderr):
        assert "WarningCalledOnRepl" in stderr

    with case_setup_dap.test_file(
        '_debugger_case_evaluate.py',
        additional_output_checks=additional_output_checks
        ) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # We want warnings from the in evaluate in the repl (but not hover/watch).
        json_facade.evaluate(
            'import warnings; warnings.warn("WarningCalledOnRepl")', json_hit.frame_id, context='repl')

        json_facade.write_continue()

        writer.finished_ok = True


def test_evaluate_none(case_setup_dap, pyfile):

    @pyfile
    def eval_none():
        print('TEST SUCEEDED')  # break here

    with case_setup_dap.test_file(eval_none) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        evaluate_response = json_facade.evaluate('None', json_hit.frame_id, context='repl')
        assert evaluate_response.body.result is not None
        assert evaluate_response.body.result == ''

        json_facade.write_continue()

        writer.finished_ok = True


def test_evaluate_numpy(case_setup_dap, pyfile):
    try:
        import numpy
    except ImportError:
        pytest.skip('numpy not available')

    @pyfile
    def numpy_small_array_file():
        import numpy

        test_array = numpy.array(2)

        print('TEST SUCEEDED')  # break here

    with case_setup_dap.test_file(numpy_small_array_file) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        for variable in variables_response.body.variables:
            if variable['evaluateName'] == 'test_array':
                break
        else:
            raise AssertionError('Did not find "test_array" in %s' % (variables_response.body.variables,))

        evaluate_response = json_facade.evaluate('test_array', json_hit.frame_id, context='repl')

        variables_response = json_facade.get_variables_response(evaluate_response.body.variablesReference)

        check = [dict([(variable['name'], variable['value'])]) for variable in variables_response.body.variables]
        assert check in (
            [
                {'special variables': ''},
                {'dtype': "dtype('int32')"},
                {'max': '2'},
                {'min': '2'},
                {'shape': '()'},
                {'size': '1'}
            ],
            [
                {'special variables': ''},
                {'dtype': "dtype('int64')"},
                {'max': '2'},
                {'min': '2'},
                {'shape': '()'},
                {'size': '1'}
            ],
        )

        json_facade.write_continue()

        writer.finished_ok = True


def test_evaluate_name_mangling(case_setup_dap, pyfile):

    @pyfile
    def target():

        class SomeObj(object):

            def __init__(self):
                self.__value = 10
                print('here')  # Break here

        SomeObj()

        print('TEST SUCEEDED')

    with case_setup_dap.test_file(target) as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_launch(justMyCode=False)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        # Check eval with a properly indented block
        evaluate_response = json_facade.evaluate(
            'self.__value',
            frameId=json_hit.frame_id,
            context="repl",
        )

        assert evaluate_response.body.result == '10'
        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_no_name_mangling(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        # Check eval with a properly indented block
        evaluate_response = json_facade.evaluate(
            'x = "_"', frameId=json_hit.frame_id, context="repl")
        assert not evaluate_response.body.result

        evaluate_response = json_facade.evaluate(
            'x', frameId=json_hit.frame_id, context="repl")
        assert evaluate_response.body.result == "'_'"

        evaluate_response = json_facade.evaluate(
            'y = "__"', frameId=json_hit.frame_id, context="repl")
        assert not evaluate_response.body.result

        evaluate_response = json_facade.evaluate(
            'y', frameId=json_hit.frame_id, context="repl")
        assert evaluate_response.body.result == "'__'"

        evaluate_response = json_facade.evaluate(
            'None', json_hit.frame_id, context='repl')
        assert not evaluate_response.body.result

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_block_repl(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        # Check eval with a properly indented block
        json_facade.evaluate(
            "for i in range(2):\n  print('var%s' % i)",
            frameId=json_hit.frame_id,
            context="repl",
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'var0' in output_event.body.output)
        assert len(messages) == 1
        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'var1' in output_event.body.output)
        assert len(messages) == 1

        # Check eval with a block that needs to be dedented
        json_facade.evaluate(
            "  for i in range(2):\n    print('foo%s' % i)",
            frameId=json_hit.frame_id,
            context="repl",
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'foo0' in output_event.body.output)
        assert len(messages) == 1
        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'foo1' in output_event.body.output)
        assert len(messages) == 1

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_block_clipboard(case_setup_dap, pyfile):

    @pyfile
    def target():
        MAX_LIMIT = 65538

        class SomeObj(object):

            def __str__(self):
                return var1

            __repr__ = __str__

        var1 = 'a' * 80000
        var2 = 20000
        var3 = SomeObj()

        print('TEST SUCEEDED')  # Break here

    def verify(evaluate_response):
        # : :type evaluate_response: EvaluateResponse
        assert len(evaluate_response.body.result) >= 80000
        assert '...' not in evaluate_response.body.result
        assert set(evaluate_response.body.result).issubset(set(['a', "'"]))

    with case_setup_dap.test_file(target) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        evaluate_response = json_facade.evaluate(
            'var1',
            frameId=json_hit.frame_id,
            context='clipboard',
        )
        verify(evaluate_response)

        evaluate_response = json_facade.evaluate(
            'var2',
            frameId=json_hit.frame_id,
            context='clipboard',
            fmt={'hex': True}
        )
        assert evaluate_response.body.result == "0x4e20"

        evaluate_response = json_facade.evaluate(
            'var3',
            frameId=json_hit.frame_id,
            context='clipboard',
        )
        verify(evaluate_response)

        json_facade.write_continue()
        writer.finished_ok = True


def test_exception_on_dir(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_dir_exception.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = json_facade.pop_variables_reference(variables_response.body.variables)
        variables_response = json_facade.get_variables_response(variables_references[0])
        assert variables_response.body.variables == [
            {'variablesReference': 0, 'type': 'int', 'evaluateName': 'self.__dict__[var1]', 'name': 'var1', 'value': '10'}]

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.parametrize('scenario', [
    'step_in',
    'step_next',
    'step_out',
])
@pytest.mark.parametrize('asyncio', [True, False])
def test_return_value_regular(case_setup_dap, scenario, asyncio):
    with case_setup_dap.test_file('_debugger_case_return_value.py' if not asyncio else '_debugger_case_return_value_asyncio.py') as writer:
        json_facade = JsonFacade(writer)

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_launch(debugOptions=['ShowReturnValue'])
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        if scenario == 'step_next':
            json_facade.write_step_next(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step', name='main', line=break_line + 1)

        elif scenario == 'step_in':
            json_facade.write_step_in(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step', name='method1')

            json_facade.write_step_in(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step', name='main')

        elif scenario == 'step_out':
            json_facade.write_step_in(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step', name='method1')

            json_facade.write_step_out(json_hit.thread_id)
            json_hit = json_facade.wait_for_thread_stopped('step', name='main')

        else:
            raise AssertionError('unhandled scenario: %s' % (scenario,))

        variables_response = json_facade.get_variables_response(json_hit.frame_id)
        return_variables = json_facade.filter_return_variables(variables_response.body.variables)
        assert return_variables == [{
            'name': '(return) method1',
            'value': '1',
            'type': 'int',
            'evaluateName': "__pydevd_ret_val_dict['method1']",
            'presentationHint': {'attributes': ['readOnly']},
            'variablesReference': 0,
        }]

        json_facade.write_continue()
        writer.finished_ok = True


def test_stack_and_variables_set_and_list(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
        variables_response = json_facade.get_variables_response(json_hit.frame_id)

        variables_references = json_facade.pop_variables_reference(variables_response.body.variables)
        expected_set = "{'a'}"
        assert variables_response.body.variables == [
            {'type': 'list', 'evaluateName': 'variable_for_test_1', 'name': 'variable_for_test_1', 'value': "['a', 'b']"},
            {'type': 'set', 'evaluateName': 'variable_for_test_2', 'name': 'variable_for_test_2', 'value': expected_set}
        ]

        variables_response = json_facade.get_variables_response(variables_references[0])
        cleaned_vars = _clear_groups(variables_response.body.variables)
        if IS_PYPY:
            # Functions are not found in PyPy.
            assert cleaned_vars.groups_found == set([DAPGrouper.SCOPE_SPECIAL_VARS])
        else:
            assert cleaned_vars.groups_found == set([DAPGrouper.SCOPE_SPECIAL_VARS, DAPGrouper.SCOPE_FUNCTION_VARS])
        assert cleaned_vars.variables == [{
            u'name': u'0',
            u'type': u'str',
            u'value': u"'a'",
            u'presentationHint': {u'attributes': [u'rawString']},
            u'evaluateName': u'variable_for_test_1[0]',
            u'variablesReference': 0,
        },
        {
            u'name': u'1',
            u'type': u'str',
            u'value': u"'b'",
            u'presentationHint': {u'attributes': [u'rawString']},
            u'evaluateName': u'variable_for_test_1[1]',
            u'variablesReference': 0,
        },
        {
            u'name': GENERATED_LEN_ATTR_NAME,
            u'type': u'int',
            u'value': u'2',
            u'evaluateName': u'len(variable_for_test_1)',
            u'variablesReference': 0,
            u'presentationHint': {'attributes': ['readOnly']},
        }]

        json_facade.write_continue()
        writer.finished_ok = True


_CleanedVars = namedtuple('_CleanedVars', 'variables, groups_found')


def _clear_groups(variables):
    groups_found = set()
    new_variables = []
    for v in variables:
        if v['name'] in DAPGrouper.SCOPES_SORTED:
            groups_found.add(v['name'])
            assert not v['type']
            continue

        else:
            new_variables.append(v)

    return _CleanedVars(new_variables, groups_found)


@pytest.mark.skipif(IS_JYTHON, reason='Putting unicode on frame vars does not work on Jython.')
def test_evaluate_unicode(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break 2 here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        evaluate_response = json_facade.evaluate(u'\u16A0', json_hit.frame_id)

        evaluate_response_body = evaluate_response.body.to_dict()

        assert evaluate_response_body == {
            'result': "'\u16a1'",
            'type': 'str',
            'variablesReference': 0,
            'presentationHint': {'attributes': ['rawString']},
        }

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_exec_unicode(case_setup_dap):

    def get_environ(writer):
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'
        return env

    with case_setup_dap.test_file('_debugger_case_local_variables2.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        writer.write_start_redirect()

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        # Check eval
        json_facade.evaluate(
            "print(u'中')",
            frameId=json_hit.frame_id,
            context="repl",
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: (u'中' in output_event.body.output) and ('pydevd warning' not in output_event.body.output))
        assert len(messages) == 1

        # Check exec
        json_facade.evaluate(
            "a=10;print(u'中')",
            frameId=json_hit.frame_id,
            context="repl",
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: (u'中' in output_event.body.output) and ('pydevd warning' not in output_event.body.output))
        assert len(messages) == 1

        response = json_facade.evaluate(
            "u'中'",
            frameId=json_hit.frame_id,
            context="repl",
        )
        assert response.body.result in ("u'\\u4e2d'", "'\u4e2d'")  # py2 or py3

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: (u'中' in output_event.body.output) and ('pydevd warning' not in output_event.body.output))
        assert len(messages) == 0  # i.e.: we don't print in this case.

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_repl_redirect(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        # Check eval
        json_facade.evaluate(
            "print('var')",
            frameId=json_hit.frame_id,
            context="repl",
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'var' in output_event.body.output)
        assert len(messages) == 1

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_no_double_exec(case_setup_dap, pyfile):

    @pyfile
    def exec_code():

        def print_and_raise():
            print('Something')
            raise RuntimeError()

        print('Break here')
        print('TEST SUCEEDED!')

    with case_setup_dap.test_file(exec_code) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        json_facade.evaluate(
            "print_and_raise()",
            frameId=json_hit.frame_id,
            context="repl",
            success=False,
        )

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: u'Something' in output_event.body.output)
        assert len(messages) == 1

        json_facade.write_continue()
        writer.finished_ok = True


def test_evaluate_variable_references(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import EvaluateRequest
    from _pydevd_bundle._debug_adapter.pydevd_schema import EvaluateArguments
    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        evaluate_response = json_facade.wait_for_response(
            json_facade.write_request(EvaluateRequest(EvaluateArguments('variable_for_test_2', json_hit.frame_id))))

        evaluate_response_body = evaluate_response.body.to_dict()

        variables_reference = json_facade.pop_variables_reference([evaluate_response_body])

        assert evaluate_response_body == {
            'type': 'set',
            'result': "{'a'}",
            'presentationHint': {},
        }
        assert len(variables_reference) == 1
        reference = variables_reference[0]
        assert reference > 0
        variables_response = json_facade.get_variables_response(reference)
        child_variables = variables_response.to_dict()['body']['variables']

        # The name for a reference in a set is the id() of the variable and can change at each run.
        del child_variables[0]['name']

        assert child_variables == [
            {
                'type': 'str',
                'value': "'a'",
                'presentationHint': {'attributes': ['rawString']},
                'variablesReference': 0,
            },
            {
                'name': GENERATED_LEN_ATTR_NAME,
                'type': 'int',
                'value': '1',
                'presentationHint': {'attributes': ['readOnly']},
                'evaluateName': 'len(variable_for_test_2)',
                'variablesReference': 0,
            }
        ]

        json_facade.write_continue()
        writer.finished_ok = True


def test_set_expression(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import SetExpressionRequest
    from _pydevd_bundle._debug_adapter.pydevd_schema import SetExpressionArguments
    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        set_expression_response = json_facade.wait_for_response(
            json_facade.write_request(SetExpressionRequest(
                SetExpressionArguments('bb', '20', frameId=json_hit.frame_id))))
        assert set_expression_response.to_dict()['body'] == {
            'value': '20', 'type': 'int', 'presentationHint': {}, 'variablesReference': 0}

        variables_response = json_facade.get_variables_response(json_hit.frame_id)
        assert {'name': 'bb', 'value': '20', 'type': 'int', 'evaluateName': 'bb', 'variablesReference': 0} in \
            variables_response.to_dict()['body']['variables']

        json_facade.write_continue()
        writer.finished_ok = True


def test_set_expression_failures(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import SetExpressionRequest
    from _pydevd_bundle._debug_adapter.pydevd_schema import SetExpressionArguments

    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)

        set_expression_response = json_facade.wait_for_response(
            json_facade.write_request(SetExpressionRequest(
                SetExpressionArguments('frame_not_there', '10', frameId=0))))
        assert not set_expression_response.success
        assert set_expression_response.message == 'Unable to find thread to set expression.'

        json_facade.write_continue()

        writer.finished_ok = True


def test_get_variable_errors(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_completions.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # First, try with wrong id.
        response = json_facade.get_variables_response(9999, success=False)
        assert response.message == 'Wrong ID sent from the client: 9999'

        first_hit = None
        for i in range(2):
            json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
            if i == 0:
                first_hit = json_hit

            if i == 1:
                # Now, check with a previously existing frameId.
                response = json_facade.get_variables_response(first_hit.frame_id, success=False)
                assert response.message == 'Unable to find thread to evaluate variable reference.'

            json_facade.write_continue(wait_for_response=i == 0)
            if i == 0:
                json_hit = json_facade.wait_for_thread_stopped()

        writer.finished_ok = True


def test_set_variable_failure(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables2.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped()

        # Wrong frame
        set_variable_response = json_facade.write_set_variable(0, 'invalid_reference', 'invalid_reference', success=False)
        assert not set_variable_response.success
        assert set_variable_response.message == 'Unable to find thread to evaluate variable reference.'

        json_facade.write_continue()

        writer.finished_ok = True


def _check_list(json_facade, json_hit):

    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_1')
    assert variable.value == "['a', 'b', self.var1: 11]"

    var0 = json_facade.get_var(variable.variablesReference, '0')

    json_facade.write_set_variable(variable.variablesReference, var0.name, '1')

    # Check that it was actually changed.
    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_1')
    assert variable.value == "[1, 'b', self.var1: 11]"

    var1 = json_facade.get_var(variable.variablesReference, 'var1')

    json_facade.write_set_variable(variable.variablesReference, var1.name, '2')

    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_1')
    assert variable.value == "[1, 'b', self.var1: 2]"


def _check_tuple(json_facade, json_hit):

    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_4')
    assert variable.value == "tuple('a', 1, self.var1: 13)"

    var0 = json_facade.get_var(variable.variablesReference, '0')

    response = json_facade.write_set_variable(variable.variablesReference, var0.name, '1', success=False)
    assert response.message.startswith("Unable to change: ")

    var1 = json_facade.get_var(variable.variablesReference, 'var1')
    json_facade.write_set_variable(variable.variablesReference, var1.name, '2')

    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_4')
    assert variable.value == "tuple('a', 1, self.var1: 2)"


def _check_dict_subclass(json_facade, json_hit):
    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_3')
    assert variable.value == "{in_dct: 20; self.var1: 10}"

    var1 = json_facade.get_var(variable.variablesReference, 'var1')

    json_facade.write_set_variable(variable.variablesReference, var1.name, '2')

    # Check that it was actually changed.
    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_3')
    assert variable.value == "{in_dct: 20; self.var1: 2}"

    var_in_dct = json_facade.get_var(variable.variablesReference, "'in_dct'")

    json_facade.write_set_variable(variable.variablesReference, var_in_dct.name, '5')

    # Check that it was actually changed.
    variable = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_3')
    assert variable.value == "{in_dct: 5; self.var1: 2}"


def _check_set(json_facade, json_hit):
    set_var = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_2')

    assert set_var.value == "set(['a', self.var1: 12])"

    var_in_set = json_facade.get_var(set_var.variablesReference, index=1)
    assert var_in_set.name != 'var1'

    set_variables_response = json_facade.write_set_variable(set_var.variablesReference, var_in_set.name, '1')
    assert set_variables_response.body.type == "int"
    assert set_variables_response.body.value == "1"

    # Check that it was actually changed (which for a set means removing the existing entry
    # and adding a new one).
    set_var = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_2')
    assert set_var.value == "set([1, self.var1: 12])"

    # Check that it can be changed again.
    var_in_set = json_facade.get_var(set_var.variablesReference, index=1)

    # Check that adding a mutable object to the set does not work.
    response = json_facade.write_set_variable(set_var.variablesReference, var_in_set.name, '[22]', success=False)
    assert response.message.startswith('Unable to change: ')

    # Check that it's still the same (the existing entry was not removed).
    assert json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_2').value == "set([1, self.var1: 12])"

    set_variables_response = json_facade.write_set_variable(set_var.variablesReference, var_in_set.name, '(22,)')
    assert set_variables_response.body.type == "tuple"
    assert set_variables_response.body.value == "(22,)"

    # Check that the tuple created can be accessed and is correct in the response.
    var_in_tuple_in_set = json_facade.get_var(set_variables_response.body.variablesReference, '0')
    assert var_in_tuple_in_set.name == '0'
    assert var_in_tuple_in_set.value == '22'

    # Check that we can change the variable in the instance.
    var1 = json_facade.get_var(set_var.variablesReference, 'var1')

    json_facade.write_set_variable(set_var.variablesReference, var1.name, '2')

    # Check that it was actually changed.
    set_var = json_facade.get_local_var(json_hit.frame_id, 'variable_for_test_2')
    assert set_var.value == "set([(22,), self.var1: 2])"


@pytest.mark.parametrize('_check_func', [
    _check_tuple,
    _check_set,
    _check_list,
    _check_dict_subclass,
])
def test_set_variable_multiple_cases(case_setup_dap, _check_func):
    with case_setup_dap.test_file('_debugger_case_local_variables3.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        _check_func(json_facade, json_hit)

        json_facade.write_continue()

        writer.finished_ok = True


def test_get_variables_corner_case(case_setup_dap, pyfile):

    @pyfile
    def case_with_class_as_object():

        class ClassField(object):
            __name__ = 'name?'

            def __hash__(self):
                raise RuntimeError()

        class SomeClass(object):
            __class__ = ClassField()

        some_class = SomeClass()
        print('TEST SUCEEDED')  # Break here

    with case_setup_dap.test_file(case_with_class_as_object) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        set_var = json_facade.get_local_var(json_hit.frame_id, 'some_class')
        assert '__main__.SomeClass' in set_var.value

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Putting unicode on frame vars does not work on Jython.')
def test_stack_and_variables(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_local_variables.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # : :type stack_trace_response: StackTraceResponse
        # : :type stack_trace_response_body: StackTraceResponseBody
        # : :type stack_frame: StackFrame

        # Check stack trace format.
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                threadId=json_hit.thread_id,
                format={'module': True, 'line': True}
        )))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        stack_frame = next(iter(stack_trace_response_body.stackFrames))
        assert stack_frame['name'] == '__main__.Call : 4'

        # Regular stack trace request (no format).
        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
        stack_trace_response = json_hit.stack_trace_response
        stack_trace_response_body = stack_trace_response.body
        assert len(stack_trace_response_body.stackFrames) == 2
        stack_frame = next(iter(stack_trace_response_body.stackFrames))
        assert stack_frame['name'] == 'Call'
        assert stack_frame['source']['path'].endswith('_debugger_case_local_variables.py')

        name_to_scope = json_facade.get_name_to_scope(stack_frame['id'])
        scope = name_to_scope['Locals']
        frame_variables_reference = scope.variablesReference
        assert isinstance(frame_variables_reference, int)

        variables_response = json_facade.get_variables_response(frame_variables_reference)
        # : :type variables_response: VariablesResponse
        assert len(variables_response.body.variables) == 0  # No variables expected here

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step')

        variables_response = json_facade.get_variables_response(frame_variables_reference)
        # : :type variables_response: VariablesResponse
        assert variables_response.body.variables == [{
            'name': 'variable_for_test_1',
            'value': '10',
            'type': 'int',
            'evaluateName': 'variable_for_test_1',
            'variablesReference': 0,
        }]

        # Same thing with hex format
        variables_response = json_facade.get_variables_response(frame_variables_reference, fmt={'hex': True})
        # : :type variables_response: VariablesResponse
        assert variables_response.body.variables == [{
            'name': 'variable_for_test_1',
            'value': '0xa',
            'type': 'int',
            'evaluateName': 'variable_for_test_1',
            'variablesReference': 0,
        }]

        # Note: besides the scope/stack/variables we can also have references when:
        # - setting variable
        #    * If the variable was changed to a container, the new reference should be returned.
        # - evaluate expression
        #    * Currently ptvsd returns a None value in on_setExpression, so, skip this for now.
        # - output
        #    * Currently not handled by ptvsd, so, skip for now.

        # Reference is for parent (in this case the frame).
        # We'll change `variable_for_test_1` from 10 to [1].
        set_variable_response = json_facade.write_set_variable(
            frame_variables_reference, 'variable_for_test_1', '[1]')
        set_variable_response_as_dict = set_variable_response.to_dict()['body']
        if not IS_JYTHON:
            # Not properly changing var on Jython.
            assert isinstance(set_variable_response_as_dict.pop('variablesReference'), int)
            assert set_variable_response_as_dict == {'value': "[1]", 'type': 'list'}

        variables_response = json_facade.get_variables_response(frame_variables_reference)
        # : :type variables_response: VariablesResponse
        variables = variables_response.body.variables
        assert len(variables) == 1
        var_as_dict = next(iter(variables))
        if not IS_JYTHON:
            # Not properly changing var on Jython.
            assert isinstance(var_as_dict.pop('variablesReference'), int)
            assert var_as_dict == {
                'name': 'variable_for_test_1',
                'value': "[1]",
                'type': 'list',
                'evaluateName': 'variable_for_test_1',
            }

        json_facade.write_continue()

        writer.finished_ok = True


def test_hex_variables(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables_hex.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # : :type stack_trace_response: StackTraceResponse
        # : :type stack_trace_response_body: StackTraceResponseBody
        # : :type stack_frame: StackFrame
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        assert len(stack_trace_response_body.stackFrames) == 2
        stack_frame = next(iter(stack_trace_response_body.stackFrames))
        assert stack_frame['name'] == 'Call'
        assert stack_frame['source']['path'].endswith('_debugger_case_local_variables_hex.py')

        name_to_scope = json_facade.get_name_to_scope(stack_frame['id'])

        scope = name_to_scope['Locals']
        frame_variables_reference = scope.variablesReference
        assert isinstance(frame_variables_reference, int)

        fmt = {'hex': True}
        variables_request = json_facade.write_request(
            pydevd_schema.VariablesRequest(pydevd_schema.VariablesArguments(frame_variables_reference, format=fmt)))
        variables_response = json_facade.wait_for_response(variables_request)

        # : :type variables_response: VariablesResponse
        variable_for_test_1, variable_for_test_2, variable_for_test_3, variable_for_test_4 = sorted(list(
            v for v in variables_response.body.variables if v['name'].startswith('variables_for_test')
        ), key=lambda v: v['name'])
        assert variable_for_test_1 == {
            'name': 'variables_for_test_1',
            'value': "0x64",
            'type': 'int',
            'evaluateName': 'variables_for_test_1',
            'variablesReference': 0,
        }

        assert isinstance(variable_for_test_2.pop('variablesReference'), int)
        assert variable_for_test_2 == {
            'name': 'variables_for_test_2',
            'value': "[0x1, 0xa, 0x64]",
            'type': 'list',
            'evaluateName': 'variables_for_test_2'
        }

        assert isinstance(variable_for_test_3.pop('variablesReference'), int)
        assert variable_for_test_3 == {
            'name': 'variables_for_test_3',
            'value': '{0xa: 0xa, 0x64: 0x64, 0x3e8: 0x3e8}',
            'type': 'dict',
            'evaluateName': 'variables_for_test_3'
        }

        assert isinstance(variable_for_test_4.pop('variablesReference'), int)
        assert variable_for_test_4 == {
            'name': 'variables_for_test_4',
            'value': '{(0x1, 0xa, 0x64): (0x2710, 0x186a0, 0x186a0)}',
            'type': 'dict',
            'evaluateName': 'variables_for_test_4'
        }

        json_facade.write_continue()

        writer.finished_ok = True


def test_stopped_event(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_print.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        assert json_hit.thread_id

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not Jython compatible (fails on set variable).')
def test_pause_and_continue(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_pause_continue.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped()

        json_facade.write_continue()

        json_facade.write_pause()

        json_hit = json_facade.wait_for_thread_stopped(reason="pause")

        stack_frame = next(iter(json_hit.stack_trace_response.body.stackFrames))

        name_to_scope = json_facade.get_name_to_scope(stack_frame['id'])
        frame_variables_reference = name_to_scope['Locals'].variablesReference

        set_variable_response = json_facade.write_set_variable(frame_variables_reference, 'loop', 'False')
        set_variable_response_as_dict = set_variable_response.to_dict()['body']
        assert set_variable_response_as_dict == {'value': "False", 'type': 'bool', 'variablesReference': 0}

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('stepping_resumes_all_threads', [False, True])
def test_step_out_multi_threads(case_setup_dap, stepping_resumes_all_threads):
    with case_setup_dap.test_file('_debugger_case_multi_threads_stepping.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(steppingResumesAllThreads=stepping_resumes_all_threads)
        json_facade.write_set_breakpoints([
            writer.get_line_index_with_content('Break thread 1'),
        ])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        response = json_facade.write_list_threads()
        assert len(response.body.threads) == 3

        thread_name_to_id = dict((t['name'], t['id']) for t in response.body.threads)
        assert json_hit.thread_id == thread_name_to_id['thread1']

        if stepping_resumes_all_threads:
            # If we're stepping with multiple threads, we'll exit here.
            json_facade.write_step_out(thread_name_to_id['thread1'])
        else:
            json_facade.write_step_out(thread_name_to_id['thread1'])

            # Timeout is expected... make it shorter.
            writer.reader_thread.set_messages_timeout(2)
            try:
                json_hit = json_facade.wait_for_thread_stopped('step')
                raise AssertionError('Expected timeout!')
            except debugger_unittest.TimeoutError:
                pass

            json_facade.write_step_out(thread_name_to_id['thread2'])
            json_facade.write_step_next(thread_name_to_id['MainThread'])
            json_hit = json_facade.wait_for_thread_stopped('step')
            assert json_hit.thread_id == thread_name_to_id['MainThread']
            json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('stepping_resumes_all_threads', [True, False])
@pytest.mark.parametrize('step_mode', ['step_next', 'step_in'])
def test_step_next_step_in_multi_threads(case_setup_dap, stepping_resumes_all_threads, step_mode):
    with case_setup_dap.test_file('_debugger_case_multi_threads_stepping.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(steppingResumesAllThreads=stepping_resumes_all_threads)
        json_facade.write_set_breakpoints([
            writer.get_line_index_with_content('Break thread 1'),
        ])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        response = json_facade.write_list_threads()
        assert len(response.body.threads) == 3

        thread_name_to_id = dict((t['name'], t['id']) for t in response.body.threads)
        assert json_hit.thread_id == thread_name_to_id['thread1']

        for _i in range(20):
            if step_mode == 'step_next':
                json_facade.write_step_next(thread_name_to_id['thread1'])

            elif step_mode == 'step_in':
                json_facade.write_step_in(thread_name_to_id['thread1'])

            else:
                raise AssertionError('Unexpected step_mode: %s' % (step_mode,))

            json_hit = json_facade.wait_for_thread_stopped('step')
            assert json_hit.thread_id == thread_name_to_id['thread1']
            local_var = json_facade.get_local_var(json_hit.frame_id, '_event2_set')

            # We're stepping in a single thread which depends on events being set in
            # another thread, so, we can only get here if the other thread was also released.
            if local_var.value == 'True':
                if stepping_resumes_all_threads:
                    break
                else:
                    raise AssertionError('Did not expect _event2_set to be set when not resuming other threads on step.')

            time.sleep(.01)
        else:
            if stepping_resumes_all_threads:
                raise AssertionError('Expected _event2_set to be set already.')
            else:
                # That's correct, we should never reach the condition where _event2_set is set if
                # we're not resuming other threads on step.
                pass

        json_facade.write_continue()

        writer.finished_ok = True


def test_stepping(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_stepping.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints([
            writer.get_line_index_with_content('Break here 1'),
            writer.get_line_index_with_content('Break here 2')
        ])
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # Test Step-Over or 'next'
        stack_trace_response = json_hit.stack_trace_response
        for stack_frame in stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0

        stack_frame = next(iter(stack_trace_response.body.stackFrames))
        before_step_over_line = stack_frame['line']

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=before_step_over_line + 1)

        # Test step into or 'stepIn'
        json_facade.write_step_in(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='step_into')

        # Test step return or 'stepOut'
        json_facade.write_continue()
        json_hit = json_facade.wait_for_thread_stopped(name='step_out')

        json_facade.write_step_out(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', name='Call')

        json_facade.write_continue()

        writer.finished_ok = True


def test_evaluate(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_evaluate.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_frame = next(iter(stack_trace_response.body.stackFrames))
        stack_frame_id = stack_frame['id']

        # Check that evaluating variable that does not exist in hover returns success == False.
        json_facade.evaluate(
            'var_does_not_exist', frameId=stack_frame_id, context='hover', success=False)

        # Test evaluate request that results in 'eval'
        eval_response = json_facade.evaluate('var_1', frameId=stack_frame_id, context='repl')
        assert eval_response.body.result == '5'
        assert eval_response.body.type == 'int'

        # Test evaluate request that results in 'exec'
        exec_response = json_facade.evaluate('var_1 = 6', frameId=stack_frame_id, context='repl')
        assert exec_response.body.result == ''

        # Test evaluate request that results in 'exec' but fails
        exec_response = json_facade.evaluate(
            'var_1 = "abc"/6', frameId=stack_frame_id, context='repl', success=False)
        assert 'TypeError' in exec_response.body.result
        assert 'TypeError' in exec_response.message

        # Evaluate without a frameId.

        # Error because 'foo_value' is not set in 'sys'.
        exec_response = json_facade.evaluate('import email;email.foo_value', success=False)
        assert 'AttributeError' in exec_response.body.result
        assert 'AttributeError' in exec_response.message

        # Reading foo_value didn't work, but 'email' should be in the namespace now.
        json_facade.evaluate('email.foo_value=True')

        # Ok, 'foo_value' is now set in 'email' module.
        exec_response = json_facade.evaluate('email.foo_value')

        # We don't actually get variables without a frameId, we can just evaluate and observe side effects
        # (so, the result is always empty -- or an error).
        assert exec_response.body.result == ''

        json_facade.write_continue()

        writer.finished_ok = True


def test_evaluate_failures(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_completions.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # First, try with wrong id.
        exec_request = json_facade.write_request(
            pydevd_schema.EvaluateRequest(pydevd_schema.EvaluateArguments('a = 10', frameId=9999, context='repl')))
        exec_response = json_facade.wait_for_response(exec_request)
        assert exec_response.success == False
        assert exec_response.message == 'Wrong ID sent from the client: 9999'

        first_hit = None
        for i in range(2):
            json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
            if i == 0:
                first_hit = json_hit
                # Check that watch exceptions are shown as string/failure.
                response = json_facade.evaluate(
                    'invalid_var', frameId=first_hit.frame_id, context='watch', success=False)
                assert response.body.result == "NameError: name 'invalid_var' is not defined"
            if i == 1:
                # Now, check with a previously existing frameId.
                exec_request = json_facade.write_request(
                    pydevd_schema.EvaluateRequest(pydevd_schema.EvaluateArguments('a = 10', frameId=first_hit.frame_id, context='repl')))
                exec_response = json_facade.wait_for_response(exec_request)
                assert exec_response.success == False
                assert exec_response.message == 'Unable to find thread for evaluation.'

            json_facade.write_continue(wait_for_response=i == 0)
            if i == 0:
                json_hit = json_facade.wait_for_thread_stopped()

        writer.finished_ok = True


def test_evaluate_exception_trace(case_setup_dap, pyfile):

    @pyfile
    def exception_trace_file():

        class A(object):

            def __init__(self, a):
                pass

        def method():
            A()

        def method2():
            method()

        def method3():
            method2()

        print('TEST SUCEEDED')  # Break here

    with case_setup_dap.test_file(exception_trace_file) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        exec_response = json_facade.evaluate('method3()', json_hit.frame_id, 'repl', success=False)
        assert 'pydevd' not in exec_response.message  # i.e.: don't show pydevd in the trace
        assert 'method3' in exec_response.message
        assert 'method2' in exec_response.message

        exec_response = json_facade.evaluate('method2()', json_hit.frame_id, 'repl', success=False)
        assert 'pydevd' not in exec_response.message
        assert 'method3' not in exec_response.message
        assert 'method2' in exec_response.message

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('max_frames', ['default', 'all', 10])  # -1 = default, 0 = all, 10 = 10 frames
def test_exception_details(case_setup_dap, max_frames):
    with case_setup_dap.test_file('_debugger_case_large_exception_stack.py') as writer:
        json_facade = JsonFacade(writer)

        if max_frames == 'all':
            json_facade.write_launch(maxExceptionStackFrames=0)
            # trace back compresses repeated text
            min_expected_lines = 100
            max_expected_lines = 220
        elif max_frames == 'default':
            json_facade.write_launch()
            # default is all frames
            # trace back compresses repeated text
            min_expected_lines = 100
            max_expected_lines = 220
        else:
            json_facade.write_launch(maxExceptionStackFrames=max_frames)
            min_expected_lines = 10
            max_expected_lines = 22

        json_facade.write_set_exception_breakpoints(['raised'])

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped('exception')

        exc_info_request = json_facade.write_request(
            pydevd_schema.ExceptionInfoRequest(pydevd_schema.ExceptionInfoArguments(json_hit.thread_id)))
        exc_info_response = json_facade.wait_for_response(exc_info_request)

        stack_frames = json_hit.stack_trace_response.body.stackFrames
        assert 100 <= len(stack_frames) <= 104
        assert stack_frames[-1]['name'] == '<module>'
        assert stack_frames[0]['name'] == 'method1'

        body = exc_info_response.body
        assert body.exceptionId.endswith('IndexError')
        assert body.description == 'foo'
        assert normcase(body.details.kwargs['source']) == normcase(writer.TEST_FILE)
        stack_line_count = len(body.details.stackTrace.split('\n'))
        assert  min_expected_lines <= stack_line_count <= max_expected_lines

        json_facade.write_set_exception_breakpoints([])  # Don't stop on reraises.
        json_facade.write_continue()

        writer.finished_ok = True


def test_stack_levels(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_deep_stacks.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()

        # get full stack
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        full_stack_frames = stack_trace_response.body.stackFrames
        total_frames = stack_trace_response.body.totalFrames

        startFrame = 0
        levels = 20
        received_frames = []
        while startFrame < total_frames:
            stack_trace_request = json_facade.write_request(
                pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                    threadId=json_hit.thread_id,
                    startFrame=startFrame,
                    levels=20)))
            stack_trace_response = json_facade.wait_for_response(stack_trace_request)
            received_frames += stack_trace_response.body.stackFrames
            startFrame += levels

        assert full_stack_frames == received_frames

        json_facade.write_continue()

        writer.finished_ok = True


def test_breakpoint_adjustment(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_adjust_breakpoint.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()

        bp_requested = writer.get_line_index_with_content('requested')
        bp_expected = writer.get_line_index_with_content('expected')

        set_bp_request = json_facade.write_request(
            pydevd_schema.SetBreakpointsRequest(pydevd_schema.SetBreakpointsArguments(
                source=pydevd_schema.Source(path=writer.TEST_FILE, sourceReference=0),
                breakpoints=[pydevd_schema.SourceBreakpoint(bp_requested).to_dict()]))
        )
        set_bp_response = json_facade.wait_for_response(set_bp_request)
        assert set_bp_response.success
        assert set_bp_response.body.breakpoints[0]['line'] == bp_expected

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_frame = next(iter(stack_trace_response.body.stackFrames))
        assert stack_frame['line'] == bp_expected

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='No goto on Jython.')
def test_goto(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_set_next_statement.py') as writer:
        json_facade = JsonFacade(writer)

        break_line = writer.get_line_index_with_content('Break here')
        step_line = writer.get_line_index_with_content('Step here')
        json_facade.write_set_breakpoints(break_line)

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_frame = next(iter(stack_trace_response.body.stackFrames))
        assert stack_frame['line'] == break_line

        goto_targets_request = json_facade.write_request(
            pydevd_schema.GotoTargetsRequest(pydevd_schema.GotoTargetsArguments(
                source=pydevd_schema.Source(path=writer.TEST_FILE, sourceReference=0),
                line=step_line)))
        goto_targets_response = json_facade.wait_for_response(goto_targets_request)
        target_id = goto_targets_response.body.targets[0]['id']

        goto_request = json_facade.write_request(
            pydevd_schema.GotoRequest(pydevd_schema.GotoArguments(
                threadId=json_hit.thread_id,
                targetId=12345)))
        goto_response = json_facade.wait_for_response(goto_request)
        assert not goto_response.success

        goto_request = json_facade.write_request(
            pydevd_schema.GotoRequest(pydevd_schema.GotoArguments(
                threadId=json_hit.thread_id,
                targetId=target_id)))
        goto_response = json_facade.wait_for_response(goto_request)

        json_hit = json_facade.wait_for_thread_stopped('goto')

        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=json_hit.thread_id)))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_frame = next(iter(stack_trace_response.body.stackFrames))
        assert stack_frame['line'] == step_line

        json_facade.write_continue()

        # we hit the breakpoint again. Since we moved back
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        writer.finished_ok = True


def _collect_stack_frames_ending_with(json_hit, end_with_pattern):
    stack_trace_response = json_hit.stack_trace_response
    dont_trace_frames = list(frame for frame in stack_trace_response.body.stackFrames
                             if frame['source']['path'].endswith(end_with_pattern))
    return dont_trace_frames


def _check_dont_trace_filtered_out(json_hit):
    assert _collect_stack_frames_ending_with(json_hit, 'dont_trace.py') == []


def _check_dont_trace_not_filtered_out(json_hit):
    assert len(_collect_stack_frames_ending_with(json_hit, 'dont_trace.py')) == 1


@pytest.mark.parametrize('dbg_property', [
    'dont_trace',
    'trace',
    'change_pattern',
    'dont_trace_after_start'
])
def test_set_debugger_property(case_setup_dap, dbg_property):

    kwargs = {}

    with case_setup_dap.test_file('_debugger_case_dont_trace_test.py', **kwargs) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        if dbg_property in ('dont_trace', 'change_pattern', 'dont_trace_after_start'):
            json_facade.write_set_debugger_property([], ['dont_trace.py'] if not IS_WINDOWS else ['Dont_Trace.py'])

        if dbg_property == 'change_pattern':
            json_facade.write_set_debugger_property([], ['something_else.py'])

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        if dbg_property in ('dont_trace', 'dont_trace_after_start'):
            _check_dont_trace_filtered_out(json_hit)

        elif dbg_property in ('change_pattern', 'trace'):
            _check_dont_trace_not_filtered_out(json_hit)

        else:
            raise AssertionError('Unexpected: %s' % (dbg_property,))

        if dbg_property == 'dont_trace_after_start':
            json_facade.write_set_debugger_property([], ['something_else.py'])

        json_facade.write_continue()
        json_hit = json_facade.wait_for_thread_stopped()

        if dbg_property in ('dont_trace',):
            _check_dont_trace_filtered_out(json_hit)

        elif dbg_property in ('change_pattern', 'trace', 'dont_trace_after_start'):
            _check_dont_trace_not_filtered_out(json_hit)

        else:
            raise AssertionError('Unexpected: %s' % (dbg_property,))

        json_facade.write_continue()

        writer.finished_ok = True


def test_source_mapping_errors(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import Source
    from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdSourceMap

    with case_setup_dap.test_file('_debugger_case_source_mapping.py') as writer:
        json_facade = JsonFacade(writer)

        map_to_cell_1_line2 = writer.get_line_index_with_content('map to cEll1, line 2')
        map_to_cell_2_line2 = writer.get_line_index_with_content('map to cEll2, line 2')

        cell1_map = PydevdSourceMap(map_to_cell_1_line2, map_to_cell_1_line2 + 1, Source(path='<cEll1>'), 2)
        cell2_map = PydevdSourceMap(map_to_cell_2_line2, map_to_cell_2_line2 + 1, Source(path='<cEll2>'), 2)
        pydevd_source_maps = [
            cell1_map, cell2_map
        ]

        json_facade.write_set_pydevd_source_map(
            Source(path=writer.TEST_FILE),
            pydevd_source_maps=pydevd_source_maps,
        )
        # This will fail because file mappings must be 1:N, not M:N (i.e.: if there's a mapping from file1.py to <cEll1>,
        # there can be no other mapping from any other file to <cEll1>).
        # This is a limitation to make it easier to remove existing breakpoints when new breakpoints are
        # set to a file (so, any file matching that breakpoint can be removed instead of needing to check
        # which lines are corresponding to that file).
        json_facade.write_set_pydevd_source_map(
            Source(path=os.path.join(os.path.dirname(writer.TEST_FILE), 'foo.py')),
            pydevd_source_maps=pydevd_source_maps,
            success=False,
        )
        json_facade.write_make_initial_run()

        writer.finished_ok = True


@pytest.mark.parametrize(
    'target',
    ['_debugger_case_source_mapping.py', '_debugger_case_source_mapping_and_reference.py']
)
@pytest.mark.parametrize('jmc', [True, False])
def test_source_mapping_base(case_setup_dap, target, jmc):
    from _pydevd_bundle._debug_adapter.pydevd_schema import Source
    from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdSourceMap

    case_setup_dap.check_non_ascii = True

    with case_setup_dap.test_file(target) as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=jmc)

        map_to_cell_1_line2 = writer.get_line_index_with_content('map to cEll1, line 2')
        map_to_cell_2_line2 = writer.get_line_index_with_content('map to cEll2, line 2')

        cell1_map = PydevdSourceMap(map_to_cell_1_line2, map_to_cell_1_line2 + 1, Source(path='<cEll1>'), 2)
        cell2_map = PydevdSourceMap(map_to_cell_2_line2, map_to_cell_2_line2 + 1, Source(path='<cEll2>'), 2)
        pydevd_source_maps = [
            cell1_map, cell2_map, cell2_map,  # The one repeated should be ignored.
        ]

        # Set breakpoints before setting the source map (check that we reapply them).
        json_facade.write_set_breakpoints(map_to_cell_1_line2)

        test_file = writer.TEST_FILE
        if isinstance(test_file, bytes):
            # file is in the filesystem encoding (needed for launch) but protocol needs it in utf-8
            test_file = test_file.decode(file_system_encoding)
            test_file = test_file.encode('utf-8')

        json_facade.write_set_pydevd_source_map(
            Source(path=test_file),
            pydevd_source_maps=pydevd_source_maps,
        )

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=map_to_cell_1_line2, file=os.path.basename(test_file))
        for stack_frame in json_hit.stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0

        # Check that we no longer stop at the cEll1 breakpoint (its mapping should be removed when
        # the new one is added and we should only stop at cEll2).
        json_facade.write_set_breakpoints(map_to_cell_2_line2)
        for stack_frame in json_hit.stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0
        json_facade.write_continue()

        json_hit = json_facade.wait_for_thread_stopped(line=map_to_cell_2_line2, file=os.path.basename(test_file))
        json_facade.write_set_breakpoints([])  # Clears breakpoints
        json_facade.write_continue()

        writer.finished_ok = True


def test_source_mapping_just_my_code(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import Source
    from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdSourceMap

    case_setup_dap.check_non_ascii = True

    with case_setup_dap.test_file('_debugger_case_source_mapping_jmc.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=True)

        map_to_cell_1_line1 = writer.get_line_index_with_content('map to cEll1, line 1')
        map_to_cell_1_line6 = writer.get_line_index_with_content('map to cEll1, line 6')
        map_to_cell_1_line7 = writer.get_line_index_with_content('map to cEll1, line 7')

        cell1_map = PydevdSourceMap(map_to_cell_1_line1, map_to_cell_1_line7, Source(path='<cEll1>'), 1)
        pydevd_source_maps = [cell1_map]

        # Set breakpoints before setting the source map (check that we reapply them).
        json_facade.write_set_breakpoints(map_to_cell_1_line6)

        test_file = writer.TEST_FILE
        if isinstance(test_file, bytes):
            # file is in the filesystem encoding (needed for launch) but protocol needs it in utf-8
            test_file = test_file.decode(file_system_encoding)
            test_file = test_file.encode('utf-8')

        json_facade.write_set_pydevd_source_map(
            Source(path=test_file),
            pydevd_source_maps=pydevd_source_maps,
        )

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=map_to_cell_1_line6, file=os.path.basename(test_file))
        for stack_frame in json_hit.stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0

        # i.e.: Remove the source maps
        json_facade.write_set_pydevd_source_map(
            Source(path=test_file),
            pydevd_source_maps=[],
        )

        json_facade.write_continue()

        writer.finished_ok = True


def test_source_mapping_goto_target(case_setup_dap):
    from _pydevd_bundle._debug_adapter.pydevd_schema import Source
    from _pydevd_bundle._debug_adapter.pydevd_schema import PydevdSourceMap

    def additional_output_checks(writer, stdout, stderr):
        assert 'Skip this print' not in stdout
        assert 'TEST SUCEEDED' in stdout

    with case_setup_dap.test_file('_debugger_case_source_map_goto_target.py', additional_output_checks=additional_output_checks) as writer:
        test_file = writer.TEST_FILE
        if isinstance(test_file, bytes):
            # file is in the filesystem encoding (needed for launch) but protocol needs it in utf-8
            test_file = test_file.decode(file_system_encoding)
            test_file = test_file.encode('utf-8')

        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        map_to_cell_1_line1 = writer.get_line_index_with_content('map to Cell1, line 1')
        map_to_cell_1_line2 = writer.get_line_index_with_content('map to Cell1, line 2')
        map_to_cell_1_line4 = writer.get_line_index_with_content('map to Cell1, line 4')
        map_to_cell_1_line5 = writer.get_line_index_with_content('map to Cell1, line 5')

        cell1_map = PydevdSourceMap(map_to_cell_1_line1, map_to_cell_1_line5, Source(path='<Cell1>'), 1)
        pydevd_source_maps = [cell1_map]
        json_facade.write_set_pydevd_source_map(
            Source(path=test_file),
            pydevd_source_maps=pydevd_source_maps,
        )
        json_facade.write_set_breakpoints(map_to_cell_1_line2)

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=map_to_cell_1_line2, file=os.path.basename(test_file))
        for stack_frame in json_hit.stack_trace_response.body.stackFrames:
            assert stack_frame['source']['sourceReference'] == 0

        goto_targets_request = json_facade.write_request(
            pydevd_schema.GotoTargetsRequest(pydevd_schema.GotoTargetsArguments(
                source=pydevd_schema.Source(path=writer.TEST_FILE, sourceReference=0),
                line=map_to_cell_1_line4)))
        goto_targets_response = json_facade.wait_for_response(goto_targets_request)
        target_id = goto_targets_response.body.targets[0]['id']

        goto_request = json_facade.write_request(
            pydevd_schema.GotoRequest(pydevd_schema.GotoArguments(
                threadId=json_hit.thread_id,
                targetId=target_id)))
        goto_response = json_facade.wait_for_response(goto_request)
        assert goto_response.success

        json_hit = json_facade.wait_for_thread_stopped('goto')

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_CHERRYPY or IS_WINDOWS, reason='No CherryPy available / not ok in Windows.')
def test_process_autoreload_cherrypy(case_setup_multiprocessing_dap, tmpdir):
    '''
    CherryPy does an os.execv(...) which will kill the running process and replace
    it with a new process when a reload takes place, so, it mostly works as
    a new process connection (everything is the same except that the
    existing process is stopped).
    '''
    port = get_free_port()
    # We write a temp file because we'll change it to autoreload later on.
    f = tmpdir.join('_debugger_case_cherrypy.py')

    tmplt = '''
import cherrypy
cherrypy.config.update({
    'engine.autoreload.on': True,
    'checker.on': False,
    'server.socket_port': %(port)s,
})
class HelloWorld(object):

    @cherrypy.expose
    def index(self):
        print('TEST SUCEEDED')
        return "Hello World %(str)s!"  # break here
    @cherrypy.expose('/exit')
    def exit(self):
        cherrypy.engine.exit()

cherrypy.quickstart(HelloWorld())
'''

    f.write(tmplt % dict(port=port, str='INITIAL'))

    file_to_check = str(f)

    def get_environ(writer):
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'
        env["PYTHONPATH"] = str(tmpdir)
        return env

    import threading
    from tests_python.debugger_unittest import AbstractWriterThread
    with case_setup_multiprocessing_dap.test_file(file_to_check, get_environ=get_environ) as writer:

        original_ignore_stderr_line = writer._ignore_stderr_line

        @overrides(writer._ignore_stderr_line)
        def _ignore_stderr_line(line):
            if original_ignore_stderr_line(line):
                return True
            return 'ENGINE ' in line or 'CherryPy Checker' in line or 'has an empty config' in line

        writer._ignore_stderr_line = _ignore_stderr_line

        json_facade = JsonFacade(writer)
        json_facade.write_launch(debugOptions=['DebugStdLib'])

        break1_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break1_line)

        server_socket = writer.server_socket

        secondary_thread_log = []
        secondary_thread_errors = []

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                try:
                    from tests_python.debugger_unittest import ReaderThread
                    expected_connections = 1
                    for _ in range(expected_connections):
                        server_socket.listen(1)
                        self.server_socket = server_socket
                        new_sock, addr = server_socket.accept()

                        reader_thread = ReaderThread(new_sock)
                        reader_thread.name = '  *** Multiprocess Reader Thread'
                        reader_thread.start()

                        writer2 = SecondaryProcessWriterThread()

                        writer2.reader_thread = reader_thread
                        writer2.sock = new_sock

                        writer2.write_version()
                        writer2.write_add_breakpoint(break1_line)
                        writer2.write_make_initial_run()

                    secondary_thread_log.append('Initial run')

                    # Give it some time to startup
                    time.sleep(2)
                    t = writer.create_request_thread('http://127.0.0.1:%s/' % (port,))
                    t.start()

                    secondary_thread_log.append('Waiting for first breakpoint')
                    hit = writer2.wait_for_breakpoint_hit()
                    secondary_thread_log.append('Hit first breakpoint')
                    writer2.write_run_thread(hit.thread_id)

                    contents = t.wait_for_contents()
                    assert 'Hello World NEW!' in contents

                    secondary_thread_log.append('Requesting exit.')
                    t = writer.create_request_thread('http://127.0.0.1:%s/exit' % (port,))
                    t.start()
                except Exception as e:
                    secondary_thread_errors.append('Error from secondary thread: %s' % (e,))
                    raise

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        json_facade.write_make_initial_run()

        # Give it some time to startup
        time.sleep(2)

        t = writer.create_request_thread('http://127.0.0.1:%s/' % (port,))
        t.start()
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        contents = t.wait_for_contents()
        assert 'Hello World INITIAL!' in contents

        # Sleep a bit more to make sure that the initial timestamp was gotten in the
        # CherryPy background thread.
        time.sleep(2)
        f.write(tmplt % dict(port=port, str='NEW'))

        def check_condition():
            return not secondary_process_thread_communication.is_alive()

        def create_msg():
            return 'Expected secondary thread to finish before timeout.\nSecondary thread log:\n%s\nSecondary thread errors:\n%s\n' % (
                '\n'.join(secondary_thread_log), '\n'.join(secondary_thread_errors))

        wait_for_condition(check_condition, msg=create_msg)

        if secondary_thread_errors:
            raise AssertionError('Found errors in secondary thread: %s' % (secondary_thread_errors,))

        writer.finished_ok = True


def test_wait_for_attach_debugpy_mode(case_setup_remote_attach_to_dap):
    host_port = get_socket_name(close=True)

    with case_setup_remote_attach_to_dap.test_file('_debugger_case_wait_for_attach_debugpy_mode.py', host_port[1]) as writer:
        time.sleep(1)  # Give some time for it to pass the first breakpoint and wait in 'wait_for_attach'.
        writer.start_socket_client(*host_port)

        # We don't send initial messages because everything should be pre-configured to
        # the DAP mode already (i.e.: making sure it works).
        json_facade = JsonFacade(writer)
        break2_line = writer.get_line_index_with_content('Break 2')

        json_facade.write_attach()
        # Make sure we also received the initialized in the attach.
        assert len(json_facade.mark_messages(InitializedEvent)) == 1

        json_facade.write_set_breakpoints([break2_line])

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break2_line)
        json_facade.write_continue()
        writer.finished_ok = True


def test_wait_for_attach(case_setup_remote_attach_to_dap):
    host_port = get_socket_name(close=True)

    def check_thread_events(json_facade):
        json_facade.write_list_threads()
        # Check that we have the started thread event (whenever we reconnect).
        started_events = json_facade.mark_messages(ThreadEvent, lambda x: x.body.reason == 'started')
        assert len(started_events) == 1

    def check_process_event(json_facade, start_method):
        if start_method == 'attach':
            json_facade.write_attach()

        elif start_method == 'launch':
            json_facade.write_launch()

        else:
            raise AssertionError('Unexpected: %s' % (start_method,))

        process_events = json_facade.mark_messages(ProcessEvent)
        assert len(process_events) == 1
        assert next(iter(process_events)).body.startMethod == start_method

    with case_setup_remote_attach_to_dap.test_file('_debugger_case_wait_for_attach.py', host_port[1]) as writer:
        writer.TEST_FILE = debugger_unittest._get_debugger_test_file('_debugger_case_wait_for_attach_impl.py')
        time.sleep(1)  # Give some time for it to pass the first breakpoint and wait in 'wait_for_attach'.
        writer.start_socket_client(*host_port)

        json_facade = JsonFacade(writer)
        check_thread_events(json_facade)

        break1_line = writer.get_line_index_with_content('Break 1')
        break2_line = writer.get_line_index_with_content('Break 2')
        break3_line = writer.get_line_index_with_content('Break 3')

        pause1_line = writer.get_line_index_with_content('Pause 1')
        pause2_line = writer.get_line_index_with_content('Pause 2')

        check_process_event(json_facade, start_method='launch')
        json_facade.write_set_breakpoints([break1_line, break2_line, break3_line])
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break2_line)

        # Upon disconnect, all threads should be running again.
        json_facade.write_disconnect()

        # Connect back (socket should remain open).
        writer.start_socket_client(*host_port)
        json_facade = JsonFacade(writer)
        check_thread_events(json_facade)
        check_process_event(json_facade, start_method='attach')
        json_facade.write_set_breakpoints([break1_line, break2_line, break3_line])
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break3_line)

        # Upon disconnect, all threads should be running again.
        json_facade.write_disconnect()

        # Connect back (socket should remain open).
        writer.start_socket_client(*host_port)
        json_facade = JsonFacade(writer)
        check_thread_events(json_facade)
        check_process_event(json_facade, start_method='attach')
        json_facade.write_make_initial_run()

        # Connect back without a disconnect (auto-disconnects previous and connects new client).
        writer.start_socket_client(*host_port)
        json_facade = JsonFacade(writer)
        check_thread_events(json_facade)
        check_process_event(json_facade, start_method='attach')
        json_facade.write_make_initial_run()

        json_facade.write_pause()
        json_hit = json_facade.wait_for_thread_stopped(reason='pause', line=[pause1_line, pause2_line])

        # Change value of 'a' for test to finish.
        json_facade.write_set_variable(json_hit.frame_id, 'a', '10')

        json_facade.write_disconnect()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
def test_wait_for_attach_gevent(case_setup_remote_attach_to_dap):
    host_port = get_socket_name(close=True)

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        return env

    def check_thread_events(json_facade):
        json_facade.write_list_threads()
        # Check that we have the started thread event (whenever we reconnect).
        started_events = json_facade.mark_messages(ThreadEvent, lambda x: x.body.reason == 'started')
        assert len(started_events) == 1

    with case_setup_remote_attach_to_dap.test_file('_debugger_case_gevent.py', host_port[1], additional_args=['remote', 'as-server'], get_environ=get_environ) as writer:
        writer.TEST_FILE = debugger_unittest._get_debugger_test_file('_debugger_case_gevent.py')
        time.sleep(.5)  # Give some time for it to pass the first breakpoint and wait.
        writer.start_socket_client(*host_port)

        json_facade = JsonFacade(writer)
        check_thread_events(json_facade)

        break1_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break1_line)
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break1_line)

        json_facade.write_disconnect()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
@pytest.mark.parametrize('show', [True, False])
def test_gevent_show_paused_greenlets(case_setup_dap, show):

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        if show:
            env['GEVENT_SHOW_PAUSED_GREENLETS'] = 'True'
        else:
            env['GEVENT_SHOW_PAUSED_GREENLETS'] = 'False'
        return env

    with case_setup_dap.test_file('_debugger_case_gevent_simple.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)

        break1_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break1_line)
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break1_line)

        response = json_facade.write_list_threads()
        if show:
            assert len(response.body.threads) > 1

            thread_name_to_id = dict((t['name'], t['id']) for t in response.body.threads)
            assert set(thread_name_to_id.keys()) == set((
                'MainThread',
                'greenlet: <module> - _debugger_case_gevent_simple.py',
                'Greenlet: foo - _debugger_case_gevent_simple.py',
                'Hub: run - hub.py'
            ))

            for tname, tid in thread_name_to_id.items():
                stack = json_facade.get_stack_as_json_hit(
                    tid,
                    no_stack_frame=tname == 'Hub: run - hub.py'
                )
                assert stack

        else:
            assert len(response.body.threads) == 1

        json_facade.write_continue(wait_for_response=False)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
@pytest.mark.skipif(sys.platform == 'win32', reason='tput requires Linux.')
def test_gevent_subprocess_not_python(case_setup_dap):

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        env['CALL_PYTHON_SUB'] = '0'
        return env

    with case_setup_dap.test_file('_debugger_case_gevent_subprocess.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)

        break1_line = writer.get_line_index_with_content("print('TEST SUCEEDED')")
        json_facade.write_set_breakpoints(break1_line)
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(line=break1_line)

        json_facade.write_continue(wait_for_response=False)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
def test_gevent_subprocess_python(case_setup_multiprocessing_dap):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        env['CALL_PYTHON_SUB'] = '1'
        return env

    with case_setup_multiprocessing_dap.test_file(
            '_debugger_case_gevent_subprocess.py',
            get_environ=get_environ,
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()

        break1_line = writer.get_line_index_with_content("print('foo called')")
        json_facade.write_set_breakpoints([break1_line])

        server_socket = writer.server_socket
        secondary_finished_ok = [False]

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                server_socket.listen(1)
                self.server_socket = server_socket
                new_sock, addr = server_socket.accept()

                reader_thread = ReaderThread(new_sock)
                reader_thread.name = '  *** Multiprocess Reader Thread'
                reader_thread.start()

                writer2 = SecondaryProcessWriterThread()
                writer2.reader_thread = reader_thread
                writer2.sock = new_sock
                json_facade2 = JsonFacade(writer2)

                json_facade2.write_set_breakpoints([break1_line, ])
                json_facade2.write_make_initial_run()

                json_facade2.wait_for_thread_stopped()
                json_facade2.write_continue()
                secondary_finished_ok[0] = True

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        time.sleep(.1)

        json_facade.write_make_initial_run()
        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        assert secondary_finished_ok[0]
        writer.finished_ok = True


@pytest.mark.skipif(
    not TEST_GEVENT or IS_WINDOWS,
    reason='Gevent not installed / Sometimes the debugger crashes on Windows as the compiled extensions conflict with gevent.'
)
def test_notify_gevent(case_setup_dap, pyfile):

    def get_environ(writer):
        # I.e.: Make sure that gevent support is disabled
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = ''
        return env

    @pyfile
    def case_gevent():
        from gevent import monkey
        import os
        monkey.patch_all()
        print('TEST SUCEEDED')  # Break here
        os._exit(0)

    def additional_output_checks(writer, stdout, stderr):
        assert 'environment variable' in stderr
        assert 'GEVENT_SUPPORT=True' in stderr

    with case_setup_dap.test_file(
            case_gevent,
            get_environ=get_environ,
            additional_output_checks=additional_output_checks,
            EXPECTED_RETURNCODE='any',
            FORCE_KILL_PROCESS_WHEN_FINISHED_OK=True
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue(wait_for_response=False)

        wait_for_condition(lambda: 'GEVENT_SUPPORT=True' in writer.get_stderr())

        writer.finished_ok = True


def test_ppid(case_setup_dap, pyfile):

    @pyfile
    def case_ppid():
        from pydevd import get_global_debugger
        assert get_global_debugger().get_arg_ppid() == 22
        print('TEST SUCEEDED')

    def update_command_line_args(writer, args):
        ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
        ret.insert(ret.index('--client'), '--ppid')
        ret.insert(ret.index('--client'), '22')
        return ret

    with case_setup_dap.test_file(
            case_ppid,
            update_command_line_args=update_command_line_args,
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        json_facade.write_make_initial_run()

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Flaky on Jython.')
def test_path_translation_and_source_reference(case_setup_dap):

    translated_dir_not_ascii = u'áéíóú汉字'

    def get_file_in_client(writer):
        # Instead of using: test_python/_debugger_case_path_translation.py
        # we'll set the breakpoints at foo/_debugger_case_path_translation.py
        file_in_client = os.path.dirname(os.path.dirname(writer.TEST_FILE))
        return os.path.join(os.path.dirname(file_in_client), translated_dir_not_ascii, '_debugger_case_path_translation.py')

    def get_environ(writer):
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'
        return env

    with case_setup_dap.test_file('_debugger_case_path_translation.py', get_environ=get_environ) as writer:
        file_in_client = get_file_in_client(writer)
        assert 'tests_python' not in file_in_client
        assert translated_dir_not_ascii in file_in_client

        json_facade = JsonFacade(writer)

        bp_line = writer.get_line_index_with_content('break here')
        assert writer.TEST_FILE.endswith('_debugger_case_path_translation.py')
        local_root = os.path.dirname(get_file_in_client(writer))
        json_facade.write_launch(pathMappings=[{
            'localRoot': local_root,
            'remoteRoot': os.path.dirname(writer.TEST_FILE),
        }])
        json_facade.write_set_breakpoints(bp_line, filename=file_in_client)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()

        # : :type stack_trace_response: StackTraceResponse
        # : :type stack_trace_response_body: StackTraceResponseBody
        # : :type stack_frame: StackFrame

        # Check stack trace format.
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                threadId=json_hit.thread_id,
                format={'module': True, 'line': True}
        )))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        stack_frame = stack_trace_response_body.stackFrames[0]
        assert stack_frame['name'] == '__main__.call_this : %s' % (bp_line,)

        path = stack_frame['source']['path']
        file_in_client_unicode = file_in_client

        assert path == file_in_client_unicode
        source_reference = stack_frame['source']['sourceReference']
        assert source_reference == 0  # When it's translated the source reference must be == 0

        stack_frame_not_path_translated = stack_trace_response_body.stackFrames[1]
        if not stack_frame_not_path_translated['name'].startswith(
            'tests_python.resource_path_translation.other.call_me_back1 :'):
            raise AssertionError('Error. Found: >>%s<<.' % (stack_frame_not_path_translated['name'],))

        assert stack_frame_not_path_translated['source']['path'].endswith('other.py')
        source_reference = stack_frame_not_path_translated['source']['sourceReference']
        assert source_reference != 0  # Not translated

        response = json_facade.write_get_source(source_reference)
        assert "def call_me_back1(callback):" in response.body.content

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Flaky on Jython.')
def test_source_reference_no_file(case_setup_dap, tmpdir):

    with case_setup_dap.test_file('_debugger_case_source_reference.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(
            debugOptions=['DebugStdLib'],
            pathMappings=[{
                'localRoot': os.path.dirname(writer.TEST_FILE),
                'remoteRoot': os.path.dirname(writer.TEST_FILE),
        }])

        writer.write_add_breakpoint(writer.get_line_index_with_content('breakpoint'))
        json_facade.write_make_initial_run()

        # First hit is for breakpoint reached via a stack frame that doesn't have source.

        json_hit = json_facade.wait_for_thread_stopped()
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                threadId=json_hit.thread_id,
                format={'module': True, 'line': True}
        )))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        stack_frame = stack_trace_response_body.stackFrames[1]
        assert stack_frame['source']['path'] == '<string>'
        source_reference = stack_frame['source']['sourceReference']
        assert source_reference != 0

        json_facade.write_get_source(source_reference, success=False)

        json_facade.write_continue()

        # First hit is for breakpoint reached via a stack frame that doesn't have source
        # on disk, but which can be retrieved via linecache.

        json_hit = json_facade.wait_for_thread_stopped()
        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                threadId=json_hit.thread_id,
                format={'module': True, 'line': True}
        )))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        stack_frame = stack_trace_response_body.stackFrames[1]
        print(stack_frame['source']['path'])
        assert stack_frame['source']['path'] == '<something>'
        source_reference = stack_frame['source']['sourceReference']
        assert source_reference != 0

        response = json_facade.write_get_source(source_reference)
        assert response.body.content == 'foo()\n'

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_linecache_json_existing_file(case_setup_dap, tmpdir):

    with case_setup_dap.test_file('_debugger_case_linecache_existing_file.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)

        debugger_case_stepping_filename = debugger_unittest._get_debugger_test_file('_debugger_case_stepping.py')
        bp_line = writer.get_line_index_with_content('Break here 1', filename=debugger_case_stepping_filename)
        json_facade.write_set_breakpoints(bp_line, filename=debugger_case_stepping_filename)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        stack_trace_response_body = json_hit.stack_trace_response.body
        for stack_frame in stack_trace_response_body.stackFrames:
            source_reference = stack_frame['source']['sourceReference']
            assert source_reference == 0

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_linecache_json(case_setup_dap, tmpdir):

    with case_setup_dap.test_file('_debugger_case_linecache.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)

        writer.write_add_breakpoint(writer.get_line_index_with_content('breakpoint'))
        json_facade.write_make_initial_run()

        # First hit is for breakpoint reached via a stack frame that doesn't have source.

        json_hit = json_facade.wait_for_thread_stopped()
        stack_trace_response_body = json_hit.stack_trace_response.body
        source_references = []
        for stack_frame in stack_trace_response_body.stackFrames:
            if stack_frame['source']['path'] == '<foo bar>':
                source_reference = stack_frame['source']['sourceReference']
                assert source_reference != 0
                source_references.append(source_reference)

        # Each frame gets its own source reference.
        assert len(set(source_references)) == 2

        for source_reference in source_references:
            response = json_facade.write_get_source(source_reference)
            assert 'def somemethod():' in response.body.content
            assert '    foo()' in response.body.content
            assert '[x for x in range(10)]' in response.body.content

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_show_bytecode_json(case_setup_dap, tmpdir):

    with case_setup_dap.test_file('_debugger_case_show_bytecode.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(justMyCode=False)

        writer.write_add_breakpoint(writer.get_line_index_with_content('breakpoint'))
        json_facade.write_make_initial_run()

        # First hit is for breakpoint reached via a stack frame that doesn't have source.

        json_hit = json_facade.wait_for_thread_stopped()
        stack_trace_response_body = json_hit.stack_trace_response.body
        source_references = []
        for stack_frame in stack_trace_response_body.stackFrames:
            if stack_frame['source']['path'] == '<something>':
                source_reference = stack_frame['source']['sourceReference']
                assert source_reference != 0
                source_references.append(source_reference)

        # Each frame gets its own source reference.
        assert len(set(source_references)) == 2

        for source_reference in source_references:
            response = json_facade.write_get_source(source_reference)
            assert 'MyClass' in response.body.content or 'foo()' in response.body.content

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
@pytest.mark.parametrize("jmc", [False, True])
def test_case_django_no_attribute_exception_breakpoint(case_setup_django_dap, jmc):
    import django  # noqa (may not be there if TEST_DJANGO == False)
    django_version = [int(x) for x in django.get_version().split('.')][:2]

    if django_version < [2, 1]:
        pytest.skip('Template exceptions only supporting Django 2.1 onwards.')

    with case_setup_django_dap.test_file(EXPECTED_RETURNCODE='any') as writer:
        json_facade = JsonFacade(writer)

        if jmc:
            writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
            json_facade.write_launch(debugOptions=['Django'], variablePresentation={
                "all": "hide",
                "protected": "inline",
            })
            json_facade.write_set_exception_breakpoints(['raised'])
        else:
            json_facade.write_launch(debugOptions=['DebugStdLib', 'Django'], variablePresentation={
                "all": "hide",
                "protected": "inline",
            })
            # Don't set to all 'raised' because we'd stop on standard library exceptions here
            # (which is not something we want).
            json_facade.write_set_exception_breakpoints(exception_options=[
                ExceptionOptions(breakMode='always', path=[
                    {'names': ['Python Exceptions']},
                    {'names': ['AssertionError']},
                ])
            ])

        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/template_error')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        json_hit = json_facade.wait_for_thread_stopped('exception', line=7, file='template_error.html')

        stack_trace_request = json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(
                threadId=json_hit.thread_id,
                format={'module': True, 'line': True}
        )))
        stack_trace_response = json_facade.wait_for_response(stack_trace_request)
        stack_trace_response_body = stack_trace_response.body
        stack_frame = next(iter(stack_trace_response_body.stackFrames))
        assert stack_frame['source']['path'].endswith('template_error.html')

        json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
        variables_response = json_facade.get_variables_response(json_hit.frame_id)
        entries = [x for x in variables_response.to_dict()['body']['variables'] if x['name'] == 'entry']
        assert len(entries) == 1
        variables_response = json_facade.get_variables_response(entries[0]['variablesReference'])
        assert variables_response.to_dict()['body']['variables'] == [
            {'name': 'key', 'value': "'v1'", 'type': 'str', 'evaluateName': 'entry.key', 'presentationHint': {'attributes': ['rawString']}, 'variablesReference': 0},
            {'name': 'val', 'value': "'v1'", 'type': 'str', 'evaluateName': 'entry.val', 'presentationHint': {'attributes': ['rawString']}, 'variablesReference': 0}
        ]

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_line_validation(case_setup_django_dap):
    import django  # noqa (may not be there if TEST_DJANGO == False)
    django_version = [int(x) for x in django.get_version().split('.')][:2]

    support_lazy_line_validation = django_version >= [1, 9]

    import django  # noqa (may not be there if TEST_DJANGO == False)

    with case_setup_django_dap.test_file(EXPECTED_RETURNCODE='any') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch(debugOptions=['DebugStdLib', 'Django'])
        template_file = debugger_unittest._get_debugger_test_file(os.path.join(writer.DJANGO_FOLDER, 'my_app', 'templates', 'my_app', 'index.html'))
        file_doesnt_exist = os.path.join(os.path.dirname(template_file), 'this_does_not_exist.html')

        # At this point, breakpoints will still not be verified (that'll happen when we
        # actually load the template).
        if support_lazy_line_validation:
            json_facade.write_set_breakpoints([1, 2, 4], template_file, verified=False)
        else:
            json_facade.write_set_breakpoints([1, 2, 4], template_file, verified=True)

        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        json_facade.wait_for_thread_stopped(line=1)
        breakpoint_events = json_facade.mark_messages(BreakpointEvent)

        found = {}
        for breakpoint_event in breakpoint_events:
            bp = breakpoint_event.body.breakpoint
            found[bp.id] = (bp.verified, bp.line)

        if support_lazy_line_validation:
            # At this point breakpoints were added.
            # id=0 / Line 1 is ok
            # id=1 / Line 2 will be disabled (because line 1 is already taken)
            # id=2 / Line 4 will be moved to line 3
            assert found == {
                0: (True, 1),
                1: (False, 2),
                2: (True, 3),
            }
        else:
            assert found == {}

        # Now, after the template was loaded, when setting the breakpoints we can already
        # know about the template validation.
        if support_lazy_line_validation:
            json_facade.write_set_breakpoints(
                [1, 2, 8], template_file, expected_lines_in_response=set((1, 2, 7)),
                # i.e.: breakpoint id to whether it's verified.
                verified={3: True, 4: False, 5: True})
        else:
            json_facade.write_set_breakpoints(
                [1, 2, 7], template_file, verified=True)

        json_facade.write_continue()
        json_facade.wait_for_thread_stopped(line=7)

        json_facade.write_continue()
        json_facade.wait_for_thread_stopped(line=7)

        # To finish, check that setting on a file that doesn't exist is not verified.
        response = json_facade.write_set_breakpoints([1], file_doesnt_exist, verified=False)
        for bp in response.body.breakpoints:
            assert 'Breakpoint in file that does not exist' in bp['message']

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_FLASK, reason='No flask available')
def test_case_flask_line_validation(case_setup_flask_dap):
    with case_setup_flask_dap.test_file(EXPECTED_RETURNCODE='any') as writer:
        json_facade = JsonFacade(writer)
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('flask1')])
        json_facade.write_launch(debugOptions=['Jinja'])
        json_facade.write_make_initial_run()

        template_file = debugger_unittest._get_debugger_test_file(os.path.join('flask1', 'templates', 'hello.html'))

        # At this point, breakpoints will still not be verified (that'll happen when we
        # actually load the template).
        json_facade.write_set_breakpoints([1, 5, 6, 10], template_file, verified=False)

        writer.write_make_initial_run()

        t = writer.create_request_thread()
        time.sleep(2)  # Give flask some time to get to startup before requesting the page
        t.start()

        json_facade.wait_for_thread_stopped(line=5)
        breakpoint_events = json_facade.mark_messages(BreakpointEvent)

        found = {}
        for breakpoint_event in breakpoint_events:
            bp = breakpoint_event.body.breakpoint
            found[bp.id] = (bp.verified, bp.line)

        # At this point breakpoints were added.
        # id=0 / Line 1 will be disabled
        # id=1 / Line 5 is correct
        # id=2 / Line 6 will be disabled (because line 5 is already taken)
        # id=3 / Line 10 will be moved to line 8
        assert found == {
            0: (False, 1),
            1: (True, 5),
            2: (False, 6),
            3: (True, 8),
        }

        json_facade.write_continue()

        json_facade.wait_for_thread_stopped(line=8)
        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_FLASK, reason='No flask available')
@pytest.mark.parametrize("jmc", [False, True])
def test_case_flask_exceptions(case_setup_flask_dap, jmc):
    with case_setup_flask_dap.test_file(EXPECTED_RETURNCODE='any') as writer:
        json_facade = JsonFacade(writer)

        if jmc:
            ignore_py_exceptions = False
            writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
            json_facade.write_launch(debugOptions=['Jinja'])
            json_facade.write_set_exception_breakpoints(['raised'])
        else:
            ignore_py_exceptions = True
            json_facade.write_launch(debugOptions=['DebugStdLib', 'Jinja'])
            # Don't set to all 'raised' because we'd stop on standard library exceptions here
            # (which is not something we want).
            json_facade.write_set_exception_breakpoints(exception_options=[
                ExceptionOptions(breakMode='always', path=[
                    {'names': ['Python Exceptions']},
                    {'names': ['IndexError']},
                ])
            ])
        json_facade.write_make_initial_run()

        t = writer.create_request_thread('/bad_template')
        time.sleep(2)  # Give flask some time to get to startup before requesting the page
        t.start()

        while True:
            json_hit = json_facade.wait_for_thread_stopped('exception')
            path = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']
            found_line = json_hit.stack_trace_response.body.stackFrames[0]['line']
            if path.endswith('bad.html'):
                assert found_line == 8
                json_facade.write_continue()
                break

            if ignore_py_exceptions and path.endswith('.py'):
                json_facade.write_continue()
                continue

            raise AssertionError('Unexpected thread stop: at %s, %s' % (path, found_line))

        writer.finished_ok = True


@pytest.mark.skipif(IS_APPVEYOR or IS_JYTHON, reason='Flaky on appveyor / Jython encoding issues (needs investigation).')
def test_redirect_output(case_setup_dap):

    def get_environ(writer):
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'
        return env

    with case_setup_dap.test_file('_debugger_case_redirect.py', get_environ=get_environ) as writer:
        original_ignore_stderr_line = writer._ignore_stderr_line

        json_facade = JsonFacade(writer)

        @overrides(writer._ignore_stderr_line)
        def _ignore_stderr_line(line):
            if original_ignore_stderr_line(line):
                return True

            binary_junk = b'\xe8\xF0\x80\x80\x80'
            if sys.version_info[0] >= 3:
                binary_junk = binary_junk.decode('utf-8', 'replace')

            return line.startswith((
                'text',
                'binary',
                'a',
                binary_junk,
            ))

        writer._ignore_stderr_line = _ignore_stderr_line

        # Note: writes to stdout and stderr are now synchronous (so, the order
        # must always be consistent and there's a message for each write).

        expected = [
            'text\n',
            'binary or text\n',
            'ação1\n',
        ]

        if sys.version_info[0] >= 3:
            expected.extend((
                'binary\n',
                'ação2\n'.encode(encoding='latin1').decode('utf-8', 'replace'),
                'ação3\n',
            ))

        binary_junk = '\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\n\n'
        if sys.version_info[0] >= 3:
            binary_junk = "\ufffd\ufffd\ufffd\ufffd\ufffd\n\n"
        expected.append(binary_junk)

        new_expected = [(x, 'stdout') for x in expected]
        new_expected.extend([(x, 'stderr') for x in expected])

        writer.write_start_redirect()

        writer.write_make_initial_run()
        msgs = []
        ignored = []
        while len(msgs) < len(new_expected):
            try:
                output_event = json_facade.wait_for_json_message(OutputEvent)
                output = output_event.body.output
                category = output_event.body.category
                msg = (output, category)
            except Exception:
                for msg in msgs:
                    sys.stderr.write('Found: %s\n' % (msg,))
                for msg in new_expected:
                    sys.stderr.write('Expected: %s\n' % (msg,))
                for msg in ignored:
                    sys.stderr.write('Ignored: %s\n' % (msg,))
                raise
            if msg not in new_expected:
                ignored.append(msg)
                continue
            msgs.append(msg)

        if msgs != new_expected:
            print(msgs)
            print(new_expected)
        assert msgs == new_expected
        writer.finished_ok = True


def test_listen_dap_messages(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_listen_dap_messages.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(debugOptions=['RedirectOutput'],)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        writer.finished_ok = True


def _attach_to_writer_pid(writer):
    import pydevd
    import threading
    import subprocess

    assert writer.process is not None

    def attach():
        attach_pydevd_file = os.path.join(os.path.dirname(pydevd.__file__), 'pydevd_attach_to_process', 'attach_pydevd.py')
        subprocess.call([sys.executable, attach_pydevd_file, '--pid', str(writer.process.pid), '--port', str(writer.port), '--protocol', 'http_json', '--debug-mode', 'debugpy-dap'])

    threading.Thread(target=attach).start()

    wait_for_condition(lambda: writer.finished_initialization)


@pytest.mark.parametrize('reattach', [True, False])
@pytest.mark.skipif(not IS_CPYTHON or IS_MAC, reason='Attach to pid only available in CPython (brittle on Mac).')
def test_attach_to_pid(case_setup_remote, reattach):
    import threading

    with case_setup_remote.test_file('_debugger_case_attach_to_pid_simple.py', wait_for_port=False) as writer:
        time.sleep(1)  # Give it some time to initialize to get to the while loop.
        _attach_to_writer_pid(writer)
        json_facade = JsonFacade(writer)

        bp_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(bp_line)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped(line=bp_line)

        if reattach:
            # This would be the same as a second attach to pid, so, the idea is closing the current
            # connection and then doing a new attach to pid.
            json_facade.write_set_breakpoints([])
            json_facade.write_continue()

            writer.do_kill()  # This will simply close the open sockets without doing anything else.
            time.sleep(1)

            t = threading.Thread(target=writer.start_socket)
            t.start()
            wait_for_condition(lambda: hasattr(writer, 'port'))
            time.sleep(1)
            writer.process = writer.process
            _attach_to_writer_pid(writer)
            wait_for_condition(lambda: hasattr(writer, 'reader_thread'))
            time.sleep(1)

            json_facade = JsonFacade(writer)
            json_facade.write_set_breakpoints(bp_line)
            json_facade.write_make_initial_run()

            json_hit = json_facade.wait_for_thread_stopped(line=bp_line)

        json_facade.write_set_variable(json_hit.frame_id, 'wait', '0')

        json_facade.write_continue()

        writer.finished_ok = True


def test_remote_debugger_basic(case_setup_remote_dap):
    with case_setup_remote_dap.test_file('_debugger_case_remote.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        writer.finished_ok = True


PYDEVD_CUSTOMIZATION_COMMAND_LINE_ARGS = ['', '--use-c-switch']
if hasattr(os, 'posix_spawn'):
    PYDEVD_CUSTOMIZATION_COMMAND_LINE_ARGS.append('--posix-spawn')


@pytest.mark.parametrize('command_line_args', PYDEVD_CUSTOMIZATION_COMMAND_LINE_ARGS)
def test_subprocess_pydevd_customization(case_setup_remote_dap, command_line_args):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread

    with case_setup_remote_dap.test_file(
            '_debugger_case_pydevd_customization.py',
            append_command_line_args=command_line_args if command_line_args else [],
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.writer.write_multi_threads_single_notification(True)
        json_facade.write_launch()

        break1_line = writer.get_line_index_with_content('break 1 here')
        break2_line = writer.get_line_index_with_content('break 2 here')
        json_facade.write_set_breakpoints([break1_line, break2_line])

        server_socket = writer.server_socket

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                expected_connections = 1

                for _ in range(expected_connections):
                    server_socket.listen(1)
                    self.server_socket = server_socket
                    writer.log.append('  *** Multiprocess waiting on server_socket.accept()')
                    new_sock, addr = server_socket.accept()
                    writer.log.append('  *** Multiprocess completed server_socket.accept()')

                    reader_thread = ReaderThread(new_sock)
                    reader_thread.name = '  *** Multiprocess Reader Thread'
                    reader_thread.start()
                    writer.log.append('  *** Multiprocess started ReaderThread')

                    writer2 = SecondaryProcessWriterThread()
                    writer2._WRITE_LOG_PREFIX = '  *** Multiprocess write: '
                    writer2.reader_thread = reader_thread
                    writer2.sock = new_sock
                    json_facade2 = JsonFacade(writer2)
                    json_facade2.writer.write_multi_threads_single_notification(True)

                    json_facade2.write_set_breakpoints([break1_line, break2_line])
                    json_facade2.write_make_initial_run()

                json_facade2.wait_for_thread_stopped()
                json_facade2.write_continue()

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        time.sleep(.1)

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()

        json_facade.write_continue()
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        secondary_process_thread_communication.join(5)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')
        writer.finished_ok = True


def test_subprocess_then_fork(case_setup_multiprocessing_dap):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread

    with case_setup_multiprocessing_dap.test_file('_debugger_case_subprocess_and_fork.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints([break_line])

        server_socket = writer.server_socket

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread

                # Note that we accept 2 connections and then we proceed to receive the breakpoints.
                json_facades = []
                for i in range(2):
                    server_socket.listen(1)
                    self.server_socket = server_socket
                    writer.log.append('  *** Multiprocess %s waiting on server_socket.accept()' % (i,))
                    new_sock, addr = server_socket.accept()
                    writer.log.append('  *** Multiprocess %s completed server_socket.accept()' % (i,))

                    reader_thread = ReaderThread(new_sock)
                    reader_thread.name = '  *** Multiprocess %s Reader Thread' % i
                    reader_thread.start()
                    writer.log.append('  *** Multiprocess %s started ReaderThread' % (i,))

                    writer2 = SecondaryProcessWriterThread()
                    writer2._WRITE_LOG_PREFIX = '  *** Multiprocess %s write: ' % i
                    writer2.reader_thread = reader_thread
                    writer2.sock = new_sock
                    json_facade2 = JsonFacade(writer2)
                    json_facade2.writer.write_multi_threads_single_notification(True)
                    writer.log.append('  *** Multiprocess %s write attachThread' % (i,))
                    json_facade2.write_attach(justMyCode=False)

                    writer.log.append('  *** Multiprocess %s write set breakpoints' % (i,))
                    json_facade2.write_set_breakpoints([break_line])
                    writer.log.append('  *** Multiprocess %s write make initial run' % (i,))
                    json_facade2.write_make_initial_run()
                    json_facades.append(json_facade2)

                for i, json_facade3 in enumerate(json_facades):
                    writer.log.append('  *** Multiprocess %s wait for thread stopped' % (i,))
                    json_facade3.wait_for_thread_stopped(line=break_line)
                    writer.log.append('  *** Multiprocess %s continue' % (i,))
                    json_facade3.write_continue()

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        time.sleep(.1)
        json_facade.write_make_initial_run()

        secondary_process_thread_communication.join(20)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        json_facade.wait_for_thread_stopped(line=break_line)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('apply_multiprocessing_patch', [True])
def test_no_subprocess_patching(case_setup_multiprocessing_dap, apply_multiprocessing_patch):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread

    def update_command_line_args(writer, args):
        ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
        ret.insert(ret.index('--client'), '--multiprocess')
        ret.insert(ret.index('--client'), '--debug-mode')
        ret.insert(ret.index('--client'), 'debugpy-dap')
        ret.insert(ret.index('--client'), '--json-dap-http')

        if apply_multiprocessing_patch:
            ret.append('apply-multiprocessing-patch')
        return ret

    with case_setup_multiprocessing_dap.test_file(
            '_debugger_case_no_subprocess_patching.py',
            update_command_line_args=update_command_line_args
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()

        break1_line = writer.get_line_index_with_content('break 1 here')
        break2_line = writer.get_line_index_with_content('break 2 here')
        json_facade.write_set_breakpoints([break1_line, break2_line])

        server_socket = writer.server_socket

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                expected_connections = 1

                for _ in range(expected_connections):
                    server_socket.listen(1)
                    self.server_socket = server_socket
                    new_sock, addr = server_socket.accept()

                    reader_thread = ReaderThread(new_sock)
                    reader_thread.name = '  *** Multiprocess Reader Thread'
                    reader_thread.start()

                    writer2 = SecondaryProcessWriterThread()
                    writer2.reader_thread = reader_thread
                    writer2.sock = new_sock
                    json_facade2 = JsonFacade(writer2)

                    json_facade2.write_set_breakpoints([break1_line, break2_line])
                    json_facade2.write_make_initial_run()

                json_facade2.wait_for_thread_stopped()
                json_facade2.write_continue()

        if apply_multiprocessing_patch:
            secondary_process_thread_communication = SecondaryProcessThreadCommunication()
            secondary_process_thread_communication.start()
            time.sleep(.1)

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        if apply_multiprocessing_patch:
            secondary_process_thread_communication.join(10)
            if secondary_process_thread_communication.is_alive():
                raise AssertionError('The SecondaryProcessThreadCommunication did not finish')
        writer.finished_ok = True


def test_module_crash(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_module.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        stopped_event = json_facade.wait_for_json_message(StoppedEvent)
        thread_id = stopped_event.body.threadId

        json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=thread_id)))

        module_event = json_facade.wait_for_json_message(ModuleEvent)  # : :type module_event: ModuleEvent
        assert 'MyName' in module_event.body.module.name
        assert 'MyVersion' in module_event.body.module.version
        assert 'MyPackage' in module_event.body.module.kwargs['package']

        json_facade.write_continue()

        writer.finished_ok = True


def test_pydevd_systeminfo(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_print.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        assert json_hit.thread_id

        info_request = json_facade.write_request(
            pydevd_schema.PydevdSystemInfoRequest(
                pydevd_schema.PydevdSystemInfoArguments()
            )
        )
        info_response = json_facade.wait_for_response(info_request)
        body = info_response.to_dict()['body']

        assert body['python']['version'] == PY_VERSION_STR
        assert body['python']['implementation']['name'] == PY_IMPL_NAME
        assert body['python']['implementation']['version'] == PY_IMPL_VERSION_STR
        assert 'description' in body['python']['implementation']

        assert body['platform'] == {'name': sys.platform}

        assert 'pid' in body['process']
        assert 'ppid' in body['process']
        assert body['process']['executable'] == sys.executable
        assert body['process']['bitness'] == 64 if IS_64BIT_PROCESS else 32

        assert 'usingCython' in body['pydevd']
        assert 'usingFrameEval' in body['pydevd']

        use_cython = os.getenv('PYDEVD_USE_CYTHON')
        if use_cython is not None:
            using_cython = use_cython == 'YES'
            assert body['pydevd']['usingCython'] == using_cython
            assert body['pydevd']['usingFrameEval'] == (using_cython and IS_PY36_OR_GREATER and not TODO_PY311)

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('scenario', [
    'terminate_request',
    'terminate_debugee'
])
@pytest.mark.parametrize('check_subprocesses', [
    'no_subprocesses',
    'kill_subprocesses',
    'kill_subprocesses_ignore_pid',
    'dont_kill_subprocesses',
])
def test_terminate(case_setup_dap, scenario, check_subprocesses):
    import psutil

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' not in ''.join(stdout)

    def update_command_line_args(writer, args):
        ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
        if check_subprocesses in ('kill_subprocesses', 'dont_kill_subprocesses'):
            ret.append('check-subprocesses')
        if check_subprocesses in ('kill_subprocesses_ignore_pid',):
            ret.append('check-subprocesses-ignore-pid')
        return ret

    with case_setup_dap.test_file(
        '_debugger_case_terminate.py',
        check_test_suceeded_msg=check_test_suceeded_msg,
        update_command_line_args=update_command_line_args,
        EXPECTED_RETURNCODE='any' if check_subprocesses == 'kill_subprocesses_ignore_pid' else 0,
        ) as writer:
        json_facade = JsonFacade(writer)
        if check_subprocesses == 'dont_kill_subprocesses':
            json_facade.write_launch(terminateChildProcesses=False)

        json_facade.write_make_initial_run()
        response = json_facade.write_initialize()
        pid = response.to_dict()['body']['pydevd']['processId']

        if check_subprocesses in ('kill_subprocesses', 'dont_kill_subprocesses', 'kill_subprocesses_ignore_pid'):
            process_ids_to_check = [pid]
            p = psutil.Process(pid)

            def wait_for_child_processes():
                children = p.children(recursive=True)
                found = len(children)
                if found == 8:
                    process_ids_to_check.extend([x.pid for x in children])
                    return True
                return False

            wait_for_condition(wait_for_child_processes)

        if scenario == 'terminate_request':
            json_facade.write_terminate()
        elif scenario == 'terminate_debugee':
            json_facade.write_disconnect(terminate_debugee=True)
        else:
            raise AssertionError('Unexpected: %s' % (scenario,))
        json_facade.wait_for_terminated()

        if check_subprocesses in ('kill_subprocesses', 'dont_kill_subprocesses', 'kill_subprocesses_ignore_pid'):

            def is_pid_alive(pid):
                # Note: the process may be a zombie process in Linux
                # (althought it's killed it remains in that state
                # because we're monitoring it).
                try:
                    proc = psutil.Process(pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        return False
                except psutil.NoSuchProcess:
                    return False
                return True

            def get_live_pids():
                return [pid for pid in process_ids_to_check if is_pid_alive(pid)]

            if check_subprocesses == 'kill_subprocesses':

                def all_pids_exited():
                    live_pids = get_live_pids()
                    if live_pids:
                        return False

                    return True

                wait_for_condition(all_pids_exited)

            elif check_subprocesses == 'kill_subprocesses_ignore_pid':

                def all_pids_exited():
                    live_pids = get_live_pids()
                    if len(live_pids) == 1:
                        return False

                    return True

                wait_for_condition(all_pids_exited)

                # Now, let's kill the remaining process ourselves.
                for pid in get_live_pids():
                    proc = psutil.Process(pid)
                    proc.kill()

            else:  # 'dont_kill_subprocesses'
                time.sleep(1)

                def only_main_pid_exited():
                    live_pids = get_live_pids()
                    if len(live_pids) == len(process_ids_to_check) - 1:
                        return True

                    return False

                wait_for_condition(only_main_pid_exited)

                # Now, let's kill the remaining processes ourselves.
                for pid in get_live_pids():
                    proc = psutil.Process(pid)
                    proc.kill()

        writer.finished_ok = True


def test_access_token(case_setup_dap):

    def update_command_line_args(self, args):
        args.insert(1, '--json-dap-http')
        args.insert(2, '--access-token')
        args.insert(3, 'bar123')
        args.insert(4, '--client-access-token')
        args.insert(5, 'foo321')
        return args

    with case_setup_dap.test_file('_debugger_case_pause_continue.py', update_command_line_args=update_command_line_args) as writer:
        json_facade = JsonFacade(writer)

        response = json_facade.write_set_debugger_property(multi_threads_single_notification=True, success=False)
        assert response.message == "Client not authenticated."

        response = json_facade.write_authorize(access_token='wrong', success=False)
        assert response.message == "Client not authenticated."

        response = json_facade.write_set_debugger_property(multi_threads_single_notification=True, success=False)
        assert response.message == "Client not authenticated."

        authorize_response = json_facade.write_authorize(access_token='bar123', success=True)
        # : :type authorize_response:PydevdAuthorizeResponse
        assert authorize_response.body.clientAccessToken == 'foo321'

        json_facade.write_set_debugger_property(multi_threads_single_notification=True)
        json_facade.write_launch()

        break_line = writer.get_line_index_with_content('Pause here and change loop to False')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_json_message(ThreadEvent, lambda event: event.body.reason == 'started')
        json_facade.wait_for_thread_stopped(line=break_line)

        # : :type response: ThreadsResponse
        response = json_facade.write_list_threads()
        assert len(response.body.threads) == 1
        assert next(iter(response.body.threads))['name'] == 'MainThread'

        json_facade.write_disconnect()

        response = json_facade.write_authorize(access_token='wrong', success=False)
        assert response.message == "Client not authenticated."

        authorize_response = json_facade.write_authorize(access_token='bar123')
        assert authorize_response.body.clientAccessToken == 'foo321'

        json_facade.write_set_breakpoints(break_line)
        json_hit = json_facade.wait_for_thread_stopped(line=break_line)
        json_facade.write_set_variable(json_hit.frame_id, 'loop', 'False')
        json_facade.write_continue()
        json_facade.wait_for_terminated()

        writer.finished_ok = True


def test_stop_on_entry(case_setup_dap):
    with case_setup_dap.test_file('not_my_code/main_on_entry.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            justMyCode=False,
            stopOnEntry=True,
            rules=[
                {'path': '**/not_my_code/**', 'include':False},
            ]
        )

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(
            'entry',
            file=(
                # We need to match the end with the proper slash.
                'my_code/__init__.py',
                'my_code\\__init__.py'
            )
        )
        json_facade.write_continue()
        writer.finished_ok = True


def test_stop_on_entry2(case_setup_dap):
    with case_setup_dap.test_file('not_my_code/main_on_entry2.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            justMyCode=False,
            stopOnEntry=True,
            showReturnValue=True,
            rules=[
                {'path': '**/main_on_entry2.py', 'include':False},
            ]
        )

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped(
            'entry',
            file='empty_file.py'
        )
        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.parametrize('val', [True, False])
def test_debug_options(case_setup_dap, val):
    with case_setup_dap.test_file('_debugger_case_debug_options.py') as writer:
        json_facade = JsonFacade(writer)
        gui_event_loop = 'matplotlib'
        if val:
            try:
                import PySide2.QtCore
            except ImportError:
                pass
            else:
                gui_event_loop = 'pyside2'
        args = dict(
            justMyCode=val,
            redirectOutput=True,  # Always redirect the output regardless of other values.
            showReturnValue=val,
            breakOnSystemExitZero=val,
            django=val,
            flask=val,
            stopOnEntry=val,
            maxExceptionStackFrames=4 if val else 5,
            guiEventLoop=gui_event_loop,
            clientOS='UNIX' if val else 'WINDOWS'
        )
        json_facade.write_launch(**args)

        json_facade.write_make_initial_run()
        if args['stopOnEntry']:
            json_facade.wait_for_thread_stopped('entry')
            json_facade.write_continue()

        output = json_facade.wait_for_json_message(
            OutputEvent, lambda msg: msg.body.category == 'stdout' and msg.body.output.startswith('{')and msg.body.output.endswith('}'))

        # The values printed are internal values from _pydevd_bundle.pydevd_json_debug_options.DebugOptions,
        # not the parameters we passed.
        translation = {
            'django': 'django_debug',
            'flask': 'flask_debug',
            'justMyCode': 'just_my_code',
            'redirectOutput': 'redirect_output',
            'showReturnValue': 'show_return_value',
            'breakOnSystemExitZero': 'break_system_exit_zero',
            'stopOnEntry': 'stop_on_entry',
            'maxExceptionStackFrames': 'max_exception_stack_frames',
            'guiEventLoop': 'gui_event_loop',
            'clientOS': 'client_os',
        }

        assert json.loads(output.body.output) == dict((translation[key], val) for key, val in args.items())
        json_facade.wait_for_terminated()
        writer.finished_ok = True


def test_gui_event_loop_custom(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_gui_event_loop.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(guiEventLoop='__main__.LoopHolder.gui_loop', redirectOutput=True)
        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()

        json_facade.wait_for_json_message(
            OutputEvent, lambda msg: msg.body.category == 'stdout' and 'gui_loop() called' in msg.body.output)

        json_facade.write_continue()
        json_facade.wait_for_terminated()
        writer.finished_ok = True


def test_gui_event_loop_qt5(case_setup_dap):
    try:
        from PySide2 import QtCore
    except ImportError:
        pytest.skip('PySide2 not available')

    with case_setup_dap.test_file('_debugger_case_gui_event_loop_qt5.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(guiEventLoop='qt5', redirectOutput=True)
        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)

        json_facade.write_make_initial_run()
        json_facade.wait_for_thread_stopped()

        # i.e.: if we don't have the event loop running in this test, this
        # output is not shown (as the QTimer timeout wouldn't be executed).
        for _i in range(3):
            json_facade.wait_for_json_message(
                OutputEvent, lambda msg: msg.body.category == 'stdout' and 'on_timeout() called' in msg.body.output)

        json_facade.write_continue()
        json_facade.wait_for_terminated()
        writer.finished_ok = True


@pytest.mark.parametrize('debug_stdlib', [True, False])
def test_just_my_code_debug_option_deprecated(case_setup_dap, debug_stdlib, debugger_runner_simple):
    from _pydev_bundle import pydev_log
    with case_setup_dap.test_file('_debugger_case_debug_options.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            redirectOutput=True,  # Always redirect the output regardless of other values.
            debugStdLib=debug_stdlib
        )
        json_facade.write_make_initial_run()
        output = json_facade.wait_for_json_message(
            OutputEvent, lambda msg: msg.body.category == 'stdout' and msg.body.output.startswith('{')and msg.body.output.endswith('}'))

        settings = json.loads(output.body.output)
        # Note: the internal attribute is just_my_code.
        assert settings['just_my_code'] == (not debug_stdlib)
        json_facade.wait_for_terminated()

        contents = []
        for f in pydev_log.list_log_files(debugger_runner_simple.pydevd_debug_file):
            if os.path.exists(f):
                with open(f, 'r') as stream:
                    contents.append(stream.read())

        writer.finished_ok = True


def test_send_invalid_messages(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_local_variables.py') as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break 2 here'))
        json_facade.write_make_initial_run()

        stopped_event = json_facade.wait_for_json_message(StoppedEvent)
        thread_id = stopped_event.body.threadId

        json_facade.write_request(
            pydevd_schema.StackTraceRequest(pydevd_schema.StackTraceArguments(threadId=thread_id)))

        # : :type response: ModulesResponse
        # : :type modules_response_body: ModulesResponseBody

        # *** Check that we accept an invalid modules request (i.e.: without arguments).
        response = json_facade.wait_for_response(json_facade.write_request(
            {'type': 'request', 'command': 'modules'}))

        modules_response_body = response.body
        assert len(modules_response_body.modules) == 1
        module = next(iter(modules_response_body.modules))
        assert module['name'] == '__main__'
        assert module['path'].endswith('_debugger_case_local_variables.py')

        # *** Check that we don't fail on request without command.
        request = json_facade.write_request({'type': 'request'})
        response = json_facade.wait_for_response(request, Response)
        assert not response.success
        assert response.command == '<unknown>'

        # *** Check that we don't crash if we can't decode message.
        json_facade.writer.write_with_content_len('invalid json here')

        # *** Check that we get a failure from a completions without arguments.
        response = json_facade.wait_for_response(json_facade.write_request(
            {'type': 'request', 'command': 'completions'}))
        assert not response.success

        json_facade.write_continue()
        writer.finished_ok = True


def test_send_json_message(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_custom_message.py') as writer:
        json_facade = JsonFacade(writer)

        json_facade.write_launch()

        json_facade.write_make_initial_run()

        json_facade.wait_for_json_message(
            OutputEvent, lambda msg: msg.body.category == 'my_category' and msg.body.output == 'some output')

        json_facade.wait_for_json_message(
            OutputEvent, lambda msg: msg.body.category == 'my_category2' and msg.body.output == 'some output 2')

        writer.finished_ok = True


def test_global_scope(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_globals.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('breakpoint here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()

        local_var = json_facade.get_global_var(json_hit.frame_id, 'in_global_scope')
        assert local_var.value == "'in_global_scope_value'"
        json_facade.write_continue()

        writer.finished_ok = True


def _check_inline_var_presentation(json_facade, json_hit, variables_response):
    var_names = [v['name'] for v in variables_response.body.variables]
    assert var_names[:3] == ['SomeClass', 'in_global_scope', '__builtins__']


def _check_hide_var_presentation(json_facade, json_hit, variables_response):
    var_names = [v['name'] for v in variables_response.body.variables]
    assert var_names == ['in_global_scope']


def _check_class_group_special_inline_presentation(json_facade, json_hit, variables_response):
    var_names = [v['name'] for v in variables_response.body.variables]
    assert var_names[:3] == ['class variables', 'in_global_scope', '__builtins__']

    variables_response = json_facade.get_variables_response(variables_response.body.variables[0]['variablesReference'])
    var_names = [v['name'] for v in variables_response.body.variables]
    assert var_names == ['SomeClass']


@pytest.mark.parametrize('var_presentation, check_func', [
    ({"all": "inline"}, _check_inline_var_presentation),
    ({"all": "hide"}, _check_hide_var_presentation),
    ({"class": "group", "special": "inline"}, _check_class_group_special_inline_presentation),
])
def test_variable_presentation(case_setup_dap, var_presentation, check_func):
    with case_setup_dap.test_file('_debugger_case_globals.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(variablePresentation=var_presentation)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('breakpoint here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()
        name_to_scope = json_facade.get_name_to_scope(json_hit.frame_id)

        variables_response = json_facade.get_variables_response(name_to_scope['Globals'].variablesReference)
        check_func(json_facade, json_hit, variables_response)

        json_facade.write_continue()

        writer.finished_ok = True


def test_debugger_case_deadlock_thread_eval(case_setup_dap):

    def get_environ(self):
        env = os.environ.copy()
        env['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '0.5'
        return env

    with case_setup_dap.test_file('_debugger_case_deadlock_thread_eval.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here 1'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()

        # If threads aren't resumed, this will deadlock.
        json_facade.evaluate('processor.process("process in evaluate")', json_hit.frame_id)

        json_facade.write_continue()

        writer.finished_ok = True


def test_debugger_case_breakpoint_on_unblock_thread_eval(case_setup_dap):

    from _pydevd_bundle._debug_adapter.pydevd_schema import EvaluateResponse

    def get_environ(self):
        env = os.environ.copy()
        env['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '0.5'
        return env

    with case_setup_dap.test_file('_debugger_case_deadlock_thread_eval.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        break1 = writer.get_line_index_with_content('Break here 1')
        break2 = writer.get_line_index_with_content('Break here 2')
        json_facade.write_set_breakpoints([break1, break2])

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break1)

        # If threads aren't resumed, this will deadlock.
        evaluate_request = json_facade.evaluate(
            'processor.process("process in evaluate")', json_hit.frame_id, wait_for_response=False)

        # We'll hit another breakpoint during that evaluation.
        json_hit = json_facade.wait_for_thread_stopped(line=break2)
        json_facade.write_set_breakpoints([])
        json_facade.write_continue()

        json_hit = json_facade.wait_for_thread_stopped(line=break1)
        json_facade.write_continue()

        # Check that we got the evaluate responses.
        messages = json_facade.mark_messages(
            EvaluateResponse, lambda evaluate_response: evaluate_response.request_seq == evaluate_request.seq)
        assert len(messages) == 1

        writer.finished_ok = True


def test_debugger_case_unblock_manually(case_setup_dap):

    from _pydevd_bundle._debug_adapter.pydevd_schema import EvaluateResponse

    def get_environ(self):
        env = os.environ.copy()
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0.5'
        return env

    with case_setup_dap.test_file('_debugger_case_deadlock_thread_eval.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()
        break1 = writer.get_line_index_with_content('Break here 1')
        json_facade.write_set_breakpoints([break1])

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break1)

        # If threads aren't resumed, this will deadlock.
        evaluate_request = json_facade.evaluate(
            'processor.process("process in evaluate")', json_hit.frame_id, wait_for_response=False)

        json_facade.wait_for_json_message(
            OutputEvent, lambda output_event: 'did not finish after' in output_event.body.output)

        # User may manually resume it.
        json_facade.write_continue()

        # Check that we got the evaluate responses.
        json_facade.wait_for_json_message(
            EvaluateResponse, lambda evaluate_response: evaluate_response.request_seq == evaluate_request.seq)

        writer.finished_ok = True


def test_debugger_case_deadlock_notify_evaluate_timeout(case_setup_dap, pyfile):

    @pyfile
    def case_slow_evaluate():

        def slow_evaluate():
            import time
            time.sleep(2)

        print('TEST SUCEEDED!')  # Break here

    def get_environ(self):
        env = os.environ.copy()
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0.5'
        return env

    with case_setup_dap.test_file(case_slow_evaluate, get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()

        # If threads aren't resumed, this will deadlock.
        json_facade.evaluate('slow_evaluate()', json_hit.frame_id)

        json_facade.write_continue()

        messages = json_facade.mark_messages(
            OutputEvent, lambda output_event: 'did not finish after' in output_event.body.output)
        assert len(messages) == 1

        writer.finished_ok = True


def test_debugger_case_deadlock_interrupt_thread(case_setup_dap, pyfile):

    @pyfile
    def case_infinite_evaluate():

        def infinite_evaluate():
            import time
            while True:
                time.sleep(.1)

        print('TEST SUCEEDED!')  # Break here

    def get_environ(self):
        env = os.environ.copy()
        env['PYDEVD_INTERRUPT_THREAD_TIMEOUT'] = '0.5'
        return env

    # Sometimes we end up with a different return code on Linux when interrupting (even
    # though we go through completion and print the 'TEST SUCEEDED' msg).
    with case_setup_dap.test_file(
        case_infinite_evaluate, get_environ=get_environ, EXPECTED_RETURNCODE='any') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()

        # If threads aren't resumed, this will deadlock.
        json_facade.evaluate('infinite_evaluate()', json_hit.frame_id, wait_for_response=False)

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.parametrize('launch_through_link', [True, False])
@pytest.mark.parametrize('breakpoints_through_link', [True, False])
def test_debugger_case_symlink(case_setup_dap, tmpdir, launch_through_link, breakpoints_through_link):
    '''
    Test that even if we resolve links internally, externally the contents will be
    related to the version launched.
    '''

    from tests_python.debugger_unittest import _get_debugger_test_file
    original_filename = _get_debugger_test_file('_debugger_case2.py')

    target_link = str(tmpdir.join('resources_link'))
    if pydevd_constants.IS_WINDOWS and not pydevd_constants.IS_PY38_OR_GREATER:
        pytest.skip('Symlink support not available.')

    try:
        os.symlink(os.path.dirname(original_filename), target_link, target_is_directory=True)
    except (OSError, TypeError, AttributeError):
        pytest.skip('Symlink support not available.')

    try:
        target_filename_in_link = os.path.join(target_link, '_debugger_case2.py')

        with case_setup_dap.test_file(target_filename_in_link if launch_through_link else original_filename) as writer:
            json_facade = JsonFacade(writer)
            json_facade.write_launch(justMyCode=False)

            # Note that internally links are resolved to match the breakpoint, so,
            # it doesn't matter if the breakpoint was added as viewed through the
            # link or the real path.
            json_facade.write_set_breakpoints(
                writer.get_line_index_with_content("print('Start Call1')"),
                filename=target_filename_in_link if breakpoints_through_link else original_filename
            )

            json_facade.write_make_initial_run()
            json_hit = json_facade.wait_for_thread_stopped()
            path = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']

            # Regardless of how it was hit, what's shown is what was launched.
            assert path == target_filename_in_link if launch_through_link else original_filename

            json_facade.write_continue()

            writer.finished_ok = True
    finally:
        # We must remove the link, otherwise pytest can end up removing things under that
        # directory when collecting temporary files.
        os.unlink(target_link)


@pytest.mark.skipif(not IS_LINUX, reason='Linux only test.')
def test_debugger_case_sensitive(case_setup_dap, tmpdir):
    path = os.path.abspath(str(tmpdir.join('Path1').join('PaTh2')))
    os.makedirs(path)
    target = os.path.join(path, 'myFile.py')
    with open(target, 'w') as stream:
        stream.write('''
print('current file', __file__) # Break here
print('TEST SUCEEDED')
''')
    assert not os.path.exists(target.lower())
    assert os.path.exists(target)

    def get_environ(self):
        env = os.environ.copy()
        # Force to normalize by doing filename.lower().
        env['PYDEVD_FILENAME_NORMALIZATION'] = 'lower'
        return env

    # Sometimes we end up with a different return code on Linux when interrupting (even
    # though we go through completion and print the 'TEST SUCEEDED' msg).
    with case_setup_dap.test_file(target, get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(writer.get_line_index_with_content('Break here'))

        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped()
        path = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']
        assert path == target

        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(
    not IS_WINDOWS or
    not IS_PY36_OR_GREATER or
    not IS_CPYTHON or
    not TEST_CYTHON or
    TODO_PY311,  # Requires frame-eval mode (still not available for Python 3.11).
    reason='Windows only test and only Python 3.6 onwards.')
def test_native_threads(case_setup_dap, pyfile):

    @pyfile
    def case_native_thread():
        from ctypes import windll, WINFUNCTYPE, c_uint32, c_void_p, c_size_t
        import time

        ThreadProc = WINFUNCTYPE(c_uint32, c_void_p)

        entered_thread = [False]

        @ThreadProc
        def method(_):
            entered_thread[0] = True  # Break here
            return 0

        windll.kernel32.CreateThread(None, c_size_t(0), method, None, c_uint32(0), None)
        while not entered_thread[0]:
            time.sleep(.1)

        print('TEST SUCEEDED')

    with case_setup_dap.test_file(case_native_thread) as writer:
        json_facade = JsonFacade(writer)

        line = writer.get_line_index_with_content('Break here')
        json_facade.write_launch(justMyCode=False)
        json_facade.write_set_breakpoints(line)
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(line=line)

        json_facade.write_continue()
        writer.finished_ok = True


def test_code_reload(case_setup_dap, pyfile):

    @pyfile
    def mod1():
        import mod2
        import time
        finish = False
        for _ in range(50):
            finish = mod2.do_something()
            if finish:
                break
            time.sleep(.1)  # Break 1
        else:
            raise AssertionError('It seems the reload was not done in the available amount of time.')

        print('TEST SUCEEDED')  # Break 2

    @pyfile
    def mod2():

        def do_something():
            return False

    with case_setup_dap.test_file(mod1) as writer:
        json_facade = JsonFacade(writer)

        line1 = writer.get_line_index_with_content('Break 1')
        line2 = writer.get_line_index_with_content('Break 2')
        json_facade.write_launch(justMyCode=False, autoReload={'pollingInterval': 0, 'enable': True})
        json_facade.write_set_breakpoints([line1, line2])
        json_facade.write_make_initial_run()

        # At this point we know that 'do_something' was called at least once.
        json_facade.wait_for_thread_stopped(line=line1)
        json_facade.write_set_breakpoints(line2)

        with open(mod2, 'w') as stream:
            stream.write('''
def do_something():
    return True
''')

        json_facade.write_continue()
        json_facade.wait_for_thread_stopped(line=line2)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_step_into_target_basic(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_smart_step_into.py') as writer:
        json_facade = JsonFacade(writer)

        bp = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints([bp])
        json_facade.write_make_initial_run()

        # At this point we know that 'do_something' was called at least once.
        hit = json_facade.wait_for_thread_stopped(line=bp)

        # : :type step_in_targets: List[StepInTarget]
        step_in_targets = json_facade.get_step_in_targets(hit.frame_id)
        label_to_id = dict((target['label'], target['id']) for target in step_in_targets)
        assert set(label_to_id.keys()) == {'bar', 'foo', 'call_outer'}
        json_facade.write_step_in(hit.thread_id, target_id=label_to_id['foo'])

        on_foo_mark_line = writer.get_line_index_with_content('on foo mark')
        hit = json_facade.wait_for_thread_stopped(reason='step', line=on_foo_mark_line)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_step_into_target_multiple(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_smart_step_into2.py') as writer:
        json_facade = JsonFacade(writer)

        bp = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints([bp])
        json_facade.write_make_initial_run()

        # At this point we know that 'do_something' was called at least once.
        hit = json_facade.wait_for_thread_stopped(line=bp)

        # : :type step_in_targets: List[StepInTarget]
        step_in_targets = json_facade.get_step_in_targets(hit.frame_id)
        label_to_id = dict((target['label'], target['id']) for target in step_in_targets)
        assert set(label_to_id.keys()) == {'foo', 'foo (call 2)', 'foo (call 3)', 'foo (call 4)'}
        json_facade.write_step_in(hit.thread_id, target_id=label_to_id['foo (call 2)'])

        on_foo_mark_line = writer.get_line_index_with_content('on foo mark')
        hit = json_facade.wait_for_thread_stopped(reason='step', line=on_foo_mark_line)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_step_into_target_genexpr(case_setup_dap):
    with case_setup_dap.test_file('_debugger_case_smart_step_into3.py') as writer:
        json_facade = JsonFacade(writer)

        bp = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints([bp])
        json_facade.write_make_initial_run()

        # At this point we know that 'do_something' was called at least once.
        hit = json_facade.wait_for_thread_stopped(line=bp)

        # : :type step_in_targets: List[StepInTarget]
        step_in_targets = json_facade.get_step_in_targets(hit.frame_id)
        label_to_id = dict((target['label'], target['id']) for target in step_in_targets)
        json_facade.write_step_in(hit.thread_id, target_id=label_to_id['foo'])

        on_foo_mark_line = writer.get_line_index_with_content('on foo mark')
        hit = json_facade.wait_for_thread_stopped(reason='step', line=on_foo_mark_line)
        json_facade.write_continue()

        writer.finished_ok = True


def test_function_breakpoints_basic(case_setup_dap, pyfile):

    @pyfile
    def module():

        def do_something():  # break here
            print('TEST SUCEEDED')

        if __name__ == '__main__':
            do_something()

    with case_setup_dap.test_file(module) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        bp = writer.get_line_index_with_content('break here')
        json_facade.write_set_function_breakpoints(['do_something'])
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(
            'function breakpoint', line=bp, preserve_focus_hint=False)
        json_facade.write_continue()

        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Python 3.6 onwards required for test.')
def test_function_breakpoints_async(case_setup_dap):

    with case_setup_dap.test_file('_debugger_case_stop_async_iteration.py') as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)
        bp = writer.get_line_index_with_content('async def gen():')
        json_facade.write_set_function_breakpoints(['gen'])
        json_facade.write_make_initial_run()

        json_facade.wait_for_thread_stopped(
            'function breakpoint', line=bp, preserve_focus_hint=False)
        json_facade.write_continue()

        writer.finished_ok = True


try:
    import pandas
except:
    pandas = None


@pytest.mark.skipif(pandas is None, reason='Pandas not installed.')
def test_pandas(case_setup_dap, pyfile):

    @pyfile
    def pandas_mod():
        import pandas as pd
        import numpy as np

        rows = 5000
        cols = 50

        # i.e.: even with these setting our repr will print at most 300 lines/cols by default.
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        items = rows * cols
        df = pd.DataFrame(np.arange(items).reshape(rows, cols)).applymap(lambda x: 'Test String')
        series = df._series[0]
        styler = df.style

        print('TEST SUCEEDED')  # Break here

    with case_setup_dap.test_file(pandas_mod) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        bp = writer.get_line_index_with_content('Break here')
        json_facade.write_set_breakpoints([bp])

        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        # json_hit = json_facade.get_stack_as_json_hit(json_hit.thread_id)
        name_to_var = json_facade.get_locals_name_to_var(json_hit.frame_id)

        # Check the custom repr(DataFrame)
        assert name_to_var['df'].value.count('\n') <= 63
        assert '...' in name_to_var['df'].value

        evaluate_response = json_facade.evaluate('df', json_hit.frame_id, context='repl')
        evaluate_response_body = evaluate_response.body.to_dict()
        assert '...' not in evaluate_response_body['result']
        assert evaluate_response_body['result'].count('\n') > 4999

        # Check the custom repr(Series)
        assert name_to_var['series'].value.count('\n') <= 60
        assert '...' in name_to_var['series'].value

        # Check custom listing (DataFrame)
        df_variables_response = json_facade.get_variables_response(name_to_var['df'].variablesReference)
        for v in df_variables_response.body.variables:
            if v['name'] == 'T':
                assert v['value'] == "'<transposed dataframe -- debugger:skipped eval>'"
                break
        else:
            raise AssertionError('Did not find variable "T".')

        # Check custom listing (Series)
        df_variables_response = json_facade.get_variables_response(name_to_var['series'].variablesReference)
        for v in df_variables_response.body.variables:
            if v['name'] == 'T':
                assert v['value'] == "'<transposed dataframe -- debugger:skipped eval>'"
                break
        else:
            raise AssertionError('Did not find variable "T".')

        # Check custom listing (Styler)
        df_variables_response = json_facade.get_variables_response(name_to_var['styler'].variablesReference)
        for v in df_variables_response.body.variables:
            if v['name'] == 'data':
                assert v['value'] == "'<Styler data -- debugger:skipped eval>'"
                break
        else:
            raise AssertionError('Did not find variable "data".')

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY38_OR_GREATER, reason='Python 3.8 onwards required for test.')
def test_same_lineno_and_filename(case_setup_dap, pyfile):

    @pyfile
    def target():

        def some_code():
            print('1')  # Break here

        code_obj = compile('''
        func()
        ''', __file__, 'exec')

        code_obj = code_obj.replace(co_name=some_code.__code__.co_name, co_firstlineno=some_code.__code__.co_firstlineno)
        exec(code_obj, {'func': some_code})

        print('TEST SUCEEDED')

    with case_setup_dap.test_file(target) as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'))
        json_facade.write_launch(justMyCode=False)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        json_facade.write_continue()

        if sys.version_info[:2] >= (3, 10):
            # On Python 3.10 we'll stop twice in this specific case
            # because the line actually matches in the caller (so
            # this is correct based on what the debugger is seeing...)
            json_hit = json_facade.wait_for_thread_stopped()
            json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows does not have execvp.')
def test_replace_process(case_setup_multiprocessing_dap):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread
    from _pydevd_bundle._debug_adapter.pydevd_schema import ExitedEvent

    with case_setup_multiprocessing_dap.test_file(
            '_debugger_case_replace_process.py',
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()

        break1_line = writer.get_line_index_with_content("print('In sub')")
        json_facade.write_set_breakpoints([break1_line])

        server_socket = writer.server_socket
        secondary_finished_ok = [False]

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                server_socket.listen(1)
                self.server_socket = server_socket
                new_sock, addr = server_socket.accept()

                reader_thread = ReaderThread(new_sock)
                reader_thread.name = '  *** Multiprocess Reader Thread'
                reader_thread.start()

                writer2 = SecondaryProcessWriterThread()
                writer2.reader_thread = reader_thread
                writer2.sock = new_sock
                json_facade2 = JsonFacade(writer2)

                json_facade2.write_set_breakpoints([break1_line, ])
                json_facade2.write_make_initial_run()

                json_facade2.wait_for_thread_stopped()
                json_facade2.write_continue()
                secondary_finished_ok[0] = True

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        time.sleep(.1)

        json_facade.write_make_initial_run()
        exited_event = json_facade.wait_for_json_message(ExitedEvent)
        assert exited_event.body.kwargs['pydevdReason'] == "processReplaced"

        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        assert secondary_finished_ok[0]
        writer.finished_ok = True


@pytest.mark.parametrize('resolve_symlinks', [True, False])
def test_use_real_path_and_not_links(case_setup_dap, tmpdir, resolve_symlinks):
    dira = tmpdir.join('dira')
    dira.mkdir()

    dirb = tmpdir.join('dirb')
    dirb.mkdir()

    original_file = dira.join('test.py')
    original_file.write('''
print('p1')  # Break here
print('p2')
print('TEST SUCEEDED')
''')

    symlinked_file = dirb.join('testit.py')
    os.symlink(str(original_file), str(symlinked_file))

    # I.e.: we're launching the symlinked file but we're actually
    # working with the original file afterwards.
    with case_setup_dap.test_file(str(symlinked_file)) as writer:
        json_facade = JsonFacade(writer)

        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), filename=str(original_file))
        json_facade.write_launch(justMyCode=False, resolveSymlinks=resolve_symlinks)
        json_facade.write_make_initial_run()

        json_hit = json_facade.wait_for_thread_stopped()
        filename = json_hit.stack_trace_response.body.stackFrames[0]['source']['path']
        if resolve_symlinks:
            assert filename == str(original_file)
        else:
            assert filename == str(symlinked_file)
        json_facade.write_continue()
        writer.finished_ok = True


_TOP_LEVEL_AWAIT_AVAILABLE = False
try:
    from ast import PyCF_ONLY_AST, PyCF_ALLOW_TOP_LEVEL_AWAIT
    _TOP_LEVEL_AWAIT_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not _TOP_LEVEL_AWAIT_AVAILABLE, reason="Top-level await required.")
def test_ipython_stepping_basic(case_setup_dap):

    def get_environ(self):
        env = os.environ.copy()

        # Test setup
        env["SCOPED_STEPPING_TARGET"] = '_debugger_case_scoped_stepping_target.py'

        # Actually setup the debugging
        env["PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING"] = "1"
        env["PYDEVD_IPYTHON_CONTEXT"] = '_debugger_case_scoped_stepping.py, run_code, run_ast_nodes'
        return env

    with case_setup_dap.test_file('_debugger_case_scoped_stepping.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        target_file = debugger_unittest._get_debugger_test_file('_debugger_case_scoped_stepping_target.py')
        break_line = writer.get_line_index_with_content('a = 1', filename=target_file)
        assert break_line == 1
        json_facade.write_set_breakpoints(break_line, filename=target_file)
        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break_line, file='_debugger_case_scoped_stepping_target.py')

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=break_line + 1, file='_debugger_case_scoped_stepping_target.py')

        json_facade.write_step_next(json_hit.thread_id)
        json_hit = json_facade.wait_for_thread_stopped('step', line=break_line + 2, file='_debugger_case_scoped_stepping_target.py')

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not _TOP_LEVEL_AWAIT_AVAILABLE, reason="Top-level await required.")
def test_ipython_stepping_step_in(case_setup_dap):

    def get_environ(self):
        env = os.environ.copy()

        # Test setup
        env["SCOPED_STEPPING_TARGET"] = '_debugger_case_scoped_stepping_target2.py'

        # Actually setup the debugging
        env["PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING"] = "1"
        env["PYDEVD_IPYTHON_CONTEXT"] = '_debugger_case_scoped_stepping.py, run_code, run_ast_nodes'
        return env

    with case_setup_dap.test_file('_debugger_case_scoped_stepping.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=False)

        target_file = debugger_unittest._get_debugger_test_file('_debugger_case_scoped_stepping_target2.py')
        break_line = writer.get_line_index_with_content('break here', filename=target_file)
        json_facade.write_set_breakpoints(break_line, filename=target_file)
        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break_line, file='_debugger_case_scoped_stepping_target2.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('b = 2', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_target2.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('method()  # break here', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_target2.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('c = 3', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_target2.py')

        json_facade.write_continue()
        writer.finished_ok = True


@pytest.mark.skipif(not _TOP_LEVEL_AWAIT_AVAILABLE, reason="Top-level await required.")
def test_ipython_stepping_step_in_justmycode(case_setup_dap):

    def get_environ(self):
        env = os.environ.copy()

        # Test setup
        env["SCOPED_STEPPING_TARGET"] = '_debugger_case_scoped_stepping_print.py'

        # Actually setup the debugging
        env["PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING"] = "1"
        env["PYDEVD_IPYTHON_CONTEXT"] = '_debugger_case_scoped_stepping.py, run_code, run_ast_nodes'
        return env

    with case_setup_dap.test_file('_debugger_case_scoped_stepping.py', get_environ=get_environ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(justMyCode=True)

        target_file = debugger_unittest._get_debugger_test_file('_debugger_case_scoped_stepping_print.py')
        break_line = writer.get_line_index_with_content('break here', filename=target_file)
        json_facade.write_set_breakpoints(break_line, filename=target_file)
        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break_line, file='_debugger_case_scoped_stepping_print.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('pause 1', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_print.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('pause 2', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_print.py')

        json_facade.write_step_in(json_hit.thread_id)
        stop_at = writer.get_line_index_with_content('pause 3', filename=target_file)
        json_hit = json_facade.wait_for_thread_stopped('step', line=stop_at, file='_debugger_case_scoped_stepping_print.py')

        json_facade.write_continue()
        writer.finished_ok = True


def test_logging_api(case_setup_multiprocessing_dap, tmpdir):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread

    log_file = str(tmpdir.join('pydevd_in_test_logging.log'))

    def get_environ(self):
        env = os.environ.copy()
        env["TARGET_LOG_FILE"] = log_file
        return env

    with case_setup_multiprocessing_dap.test_file(
            '_debugger_case_logging.py',
            get_environ=get_environ
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch()

        break1_line = writer.get_line_index_with_content("break on 2nd process")
        json_facade.write_set_breakpoints([break1_line])

        server_socket = writer.server_socket
        secondary_finished_ok = [False]

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                server_socket.listen(1)
                self.server_socket = server_socket
                new_sock, addr = server_socket.accept()

                reader_thread = ReaderThread(new_sock)
                reader_thread.name = '  *** Multiprocess Reader Thread'
                reader_thread.start()

                writer2 = SecondaryProcessWriterThread()
                writer2.reader_thread = reader_thread
                writer2.sock = new_sock
                json_facade2 = JsonFacade(writer2)

                json_facade2.write_set_breakpoints([break1_line, ])
                json_facade2.write_make_initial_run()

                json_facade2.wait_for_thread_stopped()
                json_facade2.write_continue()
                secondary_finished_ok[0] = True

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        time.sleep(.1)

        json_facade.write_make_initial_run()
        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        assert secondary_finished_ok[0]
        writer.finished_ok = True


@pytest.mark.parametrize('soft_kill', [False, True])
def test_soft_terminate(case_setup_dap, pyfile, soft_kill):

    @pyfile
    def target():
        import time
        try:
            while True:
                time.sleep(.2)  # break here
        except KeyboardInterrupt:
            # i.e.: The test succeeds if a keyboard interrupt is received.
            print('TEST SUCEEDED!')
            raise

    def check_test_suceeded_msg(self, stdout, stderr):
        if soft_kill:
            return 'TEST SUCEEDED' in ''.join(stdout)
        else:
            return 'TEST SUCEEDED' not in ''.join(stdout)

    def additional_output_checks(writer, stdout, stderr):
        if soft_kill:
            assert "KeyboardInterrupt" in stderr
        else:
            assert not stderr

    with case_setup_dap.test_file(
            target,
            EXPECTED_RETURNCODE='any',
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
        ) as writer:
        json_facade = JsonFacade(writer)
        json_facade.write_launch(
            onTerminate="KeyboardInterrupt" if soft_kill else "kill",
            justMyCode=False
        )

        break_line = writer.get_line_index_with_content('break here')
        json_facade.write_set_breakpoints(break_line)
        json_facade.write_make_initial_run()
        json_hit = json_facade.wait_for_thread_stopped(line=break_line)

        # Interrupting when inside a breakpoint will actually make the
        # debugger stop working in that thread (because there's no way
        # to keep debugging after an exception exits the tracing).

        json_facade.write_terminate()

        if soft_kill:
            json_facade.wait_for_json_message(
                OutputEvent, lambda output_event: 'raised from within the callback set' in output_event.body.output)

        writer.finished_ok = True


if __name__ == '__main__':
    pytest.main(['-k', 'test_replace_process', '-s'])

