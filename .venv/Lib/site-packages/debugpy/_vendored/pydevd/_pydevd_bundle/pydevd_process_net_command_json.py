import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial

import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
    CompletionsResponseBody, EvaluateResponseBody, ExceptionOptions,
    GotoTargetsResponseBody, ModulesResponseBody, ProcessEventBody,
    ProcessEvent, Scope, ScopesResponseBody, SetExpressionResponseBody,
    SetVariableResponseBody, SourceBreakpoint, SourceResponseBody,
    VariablesResponseBody, SetBreakpointsResponseBody, Response,
    Capabilities, PydevdAuthorizeRequest, Request,
    StepInTargetsResponseBody, SetFunctionBreakpointsResponseBody, BreakpointEvent,
    BreakpointEventBody, InitializedEvent)
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
    CMD_PROCESS_EVENT, CMD_RETURN, CMD_SET_NEXT_STATEMENT, CMD_STEP_INTO,
    CMD_STEP_INTO_MY_CODE, CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, file_system_encoding,
    CMD_STEP_RETURN_MY_CODE, CMD_STEP_RETURN)
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
    PY_IMPL_VERSION_STR, IS_64BIT_PROCESS)
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id


def _convert_rules_to_exclude_filters(rules, on_error):
    exclude_filters = []
    if not isinstance(rules, list):
        on_error('Invalid "rules" (expected list of dicts). Found: %s' % (rules,))

    else:
        directory_exclude_filters = []
        module_exclude_filters = []
        glob_exclude_filters = []

        for rule in rules:
            if not isinstance(rule, dict):
                on_error('Invalid "rules" (expected list of dicts). Found: %s' % (rules,))
                continue

            include = rule.get('include')
            if include is None:
                on_error('Invalid "rule" (expected dict with "include"). Found: %s' % (rule,))
                continue

            path = rule.get('path')
            module = rule.get('module')
            if path is None and module is None:
                on_error('Invalid "rule" (expected dict with "path" or "module"). Found: %s' % (rule,))
                continue

            if path is not None:
                glob_pattern = path
                if '*' not in path and '?' not in path:
                    if os.path.isdir(glob_pattern):
                        # If a directory was specified, add a '/**'
                        # to be consistent with the glob pattern required
                        # by pydevd.
                        if not glob_pattern.endswith('/') and not glob_pattern.endswith('\\'):
                            glob_pattern += '/'
                        glob_pattern += '**'
                    directory_exclude_filters.append(ExcludeFilter(glob_pattern, not include, True))
                else:
                    glob_exclude_filters.append(ExcludeFilter(glob_pattern, not include, True))

            elif module is not None:
                module_exclude_filters.append(ExcludeFilter(module, not include, False))

            else:
                on_error('Internal error: expected path or module to be specified.')

        # Note that we have to sort the directory/module exclude filters so that the biggest
        # paths match first.
        # i.e.: if we have:
        # /sub1/sub2/sub3
        # a rule with /sub1/sub2 would match before a rule only with /sub1.
        directory_exclude_filters = sorted(directory_exclude_filters, key=lambda exclude_filter:-len(exclude_filter.name))
        module_exclude_filters = sorted(module_exclude_filters, key=lambda exclude_filter:-len(exclude_filter.name))
        exclude_filters = directory_exclude_filters + glob_exclude_filters + module_exclude_filters

    return exclude_filters


class IDMap(object):

    def __init__(self):
        self._value_to_key = {}
        self._key_to_value = {}
        self._next_id = partial(next, itertools.count(0))

    def obtain_value(self, key):
        return self._key_to_value[key]

    def obtain_key(self, value):
        try:
            key = self._value_to_key[value]
        except KeyError:
            key = self._next_id()
            self._key_to_value[key] = value
            self._value_to_key[value] = key
        return key


class PyDevJsonCommandProcessor(object):

    def __init__(self, from_json):
        self.from_json = from_json
        self.api = PyDevdAPI()
        self._options = DebugOptions()
        self._next_breakpoint_id = partial(next, itertools.count(0))
        self._goto_targets_map = IDMap()
        self._launch_or_attach_request_done = False

    def process_net_command_json(self, py_db, json_contents, send_response=True):
        '''
        Processes a debug adapter protocol json command.
        '''

        DEBUG = False

        try:
            if isinstance(json_contents, bytes):
                json_contents = json_contents.decode('utf-8')

            request = self.from_json(json_contents, update_ids_from_dap=True)
        except Exception as e:
            try:
                loaded_json = json.loads(json_contents)
                request = Request(loaded_json.get('command', '<unknown>'), loaded_json['seq'])
            except:
                # There's not much we can do in this case...
                pydev_log.exception('Error loading json: %s', json_contents)
                return

            error_msg = str(e)
            if error_msg.startswith("'") and error_msg.endswith("'"):
                error_msg = error_msg[1:-1]

            # This means a failure processing the request (but we were able to load the seq,
            # so, answer with a failure response).
            def on_request(py_db, request):
                error_response = {
                    'type': 'response',
                    'request_seq': request.seq,
                    'success': False,
                    'command': request.command,
                    'message': error_msg,
                }
                return NetCommand(CMD_RETURN, 0, error_response, is_json=True)

        else:
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                pydev_log.info('Process %s: %s\n' % (
                    request.__class__.__name__, json.dumps(request.to_dict(update_ids_to_dap=True), indent=4, sort_keys=True),))

            assert request.type == 'request'
            method_name = 'on_%s_request' % (request.command.lower(),)
            on_request = getattr(self, method_name, None)
            if on_request is None:
                print('Unhandled: %s not available in PyDevJsonCommandProcessor.\n' % (method_name,))
                return

            if DEBUG:
                print('Handled in pydevd: %s (in PyDevJsonCommandProcessor).\n' % (method_name,))

        with py_db._main_lock:
            if request.__class__ == PydevdAuthorizeRequest:
                authorize_request = request  # : :type authorize_request: PydevdAuthorizeRequest
                access_token = authorize_request.arguments.debugServerAccessToken
                py_db.authentication.login(access_token)

            if not py_db.authentication.is_authenticated():
                response = Response(
                    request.seq, success=False, command=request.command, message='Client not authenticated.', body={})
                cmd = NetCommand(CMD_RETURN, 0, response, is_json=True)
                py_db.writer.add_command(cmd)
                return

            cmd = on_request(py_db, request)
            if cmd is not None and send_response:
                py_db.writer.add_command(cmd)

    def on_pydevdauthorize_request(self, py_db, request):
        client_access_token = py_db.authentication.client_access_token
        body = {'clientAccessToken': None}
        if client_access_token:
            body['clientAccessToken'] = client_access_token

        response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_initialize_request(self, py_db, request):
        body = Capabilities(
            # Supported.
            supportsConfigurationDoneRequest=True,
            supportsConditionalBreakpoints=True,
            supportsHitConditionalBreakpoints=True,
            supportsEvaluateForHovers=True,
            supportsSetVariable=True,
            supportsGotoTargetsRequest=True,
            supportsCompletionsRequest=True,
            supportsModulesRequest=True,
            supportsExceptionOptions=True,
            supportsValueFormattingOptions=True,
            supportsExceptionInfoRequest=True,
            supportTerminateDebuggee=True,
            supportsDelayedStackTraceLoading=True,
            supportsLogPoints=True,
            supportsSetExpression=True,
            supportsTerminateRequest=True,
            supportsClipboardContext=True,
            supportsFunctionBreakpoints=True,

            exceptionBreakpointFilters=[
                {'filter': 'raised', 'label': 'Raised Exceptions', 'default': False},
                {'filter': 'uncaught', 'label': 'Uncaught Exceptions', 'default': True},
                {"filter": "userUnhandled", "label": "User Uncaught Exceptions", "default": False},
            ],

            # Not supported.
            supportsStepBack=False,
            supportsRestartFrame=False,
            supportsStepInTargetsRequest=True,
            supportsRestartRequest=False,
            supportsLoadedSourcesRequest=False,
            supportsTerminateThreadsRequest=False,
            supportsDataBreakpoints=False,
            supportsReadMemoryRequest=False,
            supportsDisassembleRequest=False,
            additionalModuleColumns=[],
            completionTriggerCharacters=[],
            supportedChecksumAlgorithms=[],
        ).to_dict()

        # Non-standard capabilities/info below.
        body['supportsDebuggerProperties'] = True

        body['pydevd'] = pydevd_info = {}
        pydevd_info['processId'] = os.getpid()
        self.api.notify_initialize(py_db)
        response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_configurationdone_request(self, py_db, request):
        '''
        :param ConfigurationDoneRequest request:
        '''
        if not self._launch_or_attach_request_done:
            pydev_log.critical('Missing launch request or attach request before configuration done request.')

        self.api.run(py_db)
        self.api.notify_configuration_done(py_db)

        configuration_done_response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, configuration_done_response, is_json=True)

    def on_threads_request(self, py_db, request):
        '''
        :param ThreadsRequest request:
        '''
        return self.api.list_threads(py_db, request.seq)

    def on_terminate_request(self, py_db, request):
        '''
        :param TerminateRequest request:
        '''
        self._request_terminate_process(py_db)
        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def _request_terminate_process(self, py_db):
        self.api.request_terminate_process(py_db)

    def on_completions_request(self, py_db, request):
        '''
        :param CompletionsRequest request:
        '''
        arguments = request.arguments  # : :type arguments: CompletionsArguments
        seq = request.seq
        text = arguments.text
        frame_id = arguments.frameId
        thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
            frame_id)

        if thread_id is None:
            body = CompletionsResponseBody([])
            variables_response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': body,
                    'success': False,
                    'message': 'Thread to get completions seems to have resumed already.'
                })
            return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

        # Note: line and column are 1-based (convert to 0-based for pydevd).
        column = arguments.column - 1

        if arguments.line is None:
            # line is optional
            line = -1
        else:
            line = arguments.line - 1

        self.api.request_completions(py_db, seq, thread_id, frame_id, text, line=line, column=column)

    def _resolve_remote_root(self, local_root, remote_root):
        if remote_root == '.':
            cwd = os.getcwd()
            append_pathsep = local_root.endswith('\\') or local_root.endswith('/')
            return cwd + (os.path.sep if append_pathsep else '')
        return remote_root

    def _set_debug_options(self, py_db, args, start_reason):
        rules = args.get('rules')
        stepping_resumes_all_threads = args.get('steppingResumesAllThreads', True)
        self.api.set_stepping_resumes_all_threads(py_db, stepping_resumes_all_threads)

        terminate_child_processes = args.get('terminateChildProcesses', True)
        self.api.set_terminate_child_processes(py_db, terminate_child_processes)

        terminate_keyboard_interrupt = args.get('onTerminate', 'kill') == 'KeyboardInterrupt'
        self.api.set_terminate_keyboard_interrupt(py_db, terminate_keyboard_interrupt)

        variable_presentation = args.get('variablePresentation', None)
        if isinstance(variable_presentation, dict):

            def get_variable_presentation(setting, default):
                value = variable_presentation.get(setting, default)
                if value not in ('group', 'inline', 'hide'):
                    pydev_log.info(
                        'The value set for "%s" (%s) in the variablePresentation is not valid. Valid values are: "group", "inline", "hide"' % (
                            setting, value,))
                    value = default

                return value

            default = get_variable_presentation('all', 'group')

            special_presentation = get_variable_presentation('special', default)
            function_presentation = get_variable_presentation('function', default)
            class_presentation = get_variable_presentation('class', default)
            protected_presentation = get_variable_presentation('protected', default)

            self.api.set_variable_presentation(py_db, self.api.VariablePresentation(
                special_presentation,
                function_presentation,
                class_presentation,
                protected_presentation
            ))

        exclude_filters = []

        if rules is not None:
            exclude_filters = _convert_rules_to_exclude_filters(
                rules, lambda msg: self.api.send_error_message(py_db, msg))

        self.api.set_exclude_filters(py_db, exclude_filters)

        debug_options = _extract_debug_options(
            args.get('options'),
            args.get('debugOptions'),
        )
        self._options.update_fom_debug_options(debug_options)
        self._options.update_from_args(args)

        self.api.set_use_libraries_filter(py_db, self._options.just_my_code)

        if self._options.client_os:
            self.api.set_ide_os(self._options.client_os)

        path_mappings = []
        for pathMapping in args.get('pathMappings', []):
            localRoot = pathMapping.get('localRoot', '')
            remoteRoot = pathMapping.get('remoteRoot', '')
            remoteRoot = self._resolve_remote_root(localRoot, remoteRoot)
            if (localRoot != '') and (remoteRoot != ''):
                path_mappings.append((localRoot, remoteRoot))

        if bool(path_mappings):
            pydevd_file_utils.setup_client_server_paths(path_mappings)

        resolve_symlinks = args.get('resolveSymlinks', None)
        if resolve_symlinks is not None:
            pydevd_file_utils.set_resolve_symlinks(resolve_symlinks)

        redirecting = args.get("isOutputRedirected")
        if self._options.redirect_output:
            py_db.enable_output_redirection(True, True)
            redirecting = True
        else:
            py_db.enable_output_redirection(False, False)

        py_db.is_output_redirected = redirecting

        self.api.set_show_return_values(py_db, self._options.show_return_value)

        if not self._options.break_system_exit_zero:
            ignore_system_exit_codes = [0, None]
            if self._options.django_debug or self._options.flask_debug:
                ignore_system_exit_codes += [3]

            self.api.set_ignore_system_exit_codes(py_db, ignore_system_exit_codes)

        auto_reload = args.get('autoReload', {})
        if not isinstance(auto_reload, dict):
            pydev_log.info('Expected autoReload to be a dict. Received: %s' % (auto_reload,))
            auto_reload = {}

        enable_auto_reload = auto_reload.get('enable', False)
        watch_dirs = auto_reload.get('watchDirectories')
        if not watch_dirs:
            watch_dirs = []
            # Note: by default this is no longer done because on some cases there are entries in the PYTHONPATH
            # such as the home directory or /python/x64, where the site packages are in /python/x64/libs, so,
            # we only watch the current working directory as well as executed script.
            # check = getattr(sys, 'path', [])[:]
            # # By default only watch directories that are in the project roots /
            # # program dir (if available), sys.argv[0], as well as the current dir (we don't want to
            # # listen to the whole site-packages by default as it can be huge).
            # watch_dirs = [pydevd_file_utils.absolute_path(w) for w in check]
            # watch_dirs = [w for w in watch_dirs if py_db.in_project_roots_filename_uncached(w) and os.path.isdir(w)]

            program = args.get('program')
            if program:
                if os.path.isdir(program):
                    watch_dirs.append(program)
                else:
                    watch_dirs.append(os.path.dirname(program))
            watch_dirs.append(os.path.abspath('.'))

            argv = getattr(sys, 'argv', [])
            if argv:
                f = argv[0]
                if f:  # argv[0] could be None (https://github.com/microsoft/debugpy/issues/987)
                    if os.path.isdir(f):
                        watch_dirs.append(f)
                    else:
                        watch_dirs.append(os.path.dirname(f))

        if not isinstance(watch_dirs, (list, set, tuple)):
            watch_dirs = (watch_dirs,)
        new_watch_dirs = set()
        for w in watch_dirs:
            try:
                new_watch_dirs.add(pydevd_file_utils.get_path_with_real_case(pydevd_file_utils.absolute_path(w)))
            except Exception:
                pydev_log.exception('Error adding watch dir: %s', w)
        watch_dirs = new_watch_dirs

        poll_target_time = auto_reload.get('pollingInterval', 1)
        exclude_patterns = auto_reload.get('exclude', ('**/.git/**', '**/__pycache__/**', '**/node_modules/**', '**/.metadata/**', '**/site-packages/**'))
        include_patterns = auto_reload.get('include', ('**/*.py', '**/*.pyw'))
        self.api.setup_auto_reload_watcher(
            py_db, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns)

        if self._options.stop_on_entry and start_reason == 'launch':
            self.api.stop_on_entry()

        self.api.set_gui_event_loop(py_db, self._options.gui_event_loop)

    def _send_process_event(self, py_db, start_method):
        argv = getattr(sys, 'argv', [])
        if len(argv) > 0:
            name = argv[0]
        else:
            name = ''

        if isinstance(name, bytes):
            name = name.decode(file_system_encoding, 'replace')
            name = name.encode('utf-8')

        body = ProcessEventBody(
            name=name,
            systemProcessId=os.getpid(),
            isLocalProcess=True,
            startMethod=start_method,
        )
        event = ProcessEvent(body)
        py_db.writer.add_command(NetCommand(CMD_PROCESS_EVENT, 0, event, is_json=True))

    def _handle_launch_or_attach_request(self, py_db, request, start_reason):
        self._send_process_event(py_db, start_reason)
        self._launch_or_attach_request_done = True
        self.api.set_enable_thread_notifications(py_db, True)
        self._set_debug_options(py_db, request.arguments.kwargs, start_reason=start_reason)
        response = pydevd_base_schema.build_response(request)

        initialized_event = InitializedEvent()
        py_db.writer.add_command(NetCommand(CMD_RETURN, 0, initialized_event, is_json=True))
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_launch_request(self, py_db, request):
        '''
        :param LaunchRequest request:
        '''
        return self._handle_launch_or_attach_request(py_db, request, start_reason='launch')

    def on_attach_request(self, py_db, request):
        '''
        :param AttachRequest request:
        '''
        return self._handle_launch_or_attach_request(py_db, request, start_reason='attach')

    def on_pause_request(self, py_db, request):
        '''
        :param PauseRequest request:
        '''
        arguments = request.arguments  # : :type arguments: PauseArguments
        thread_id = arguments.threadId

        self.api.request_suspend_thread(py_db, thread_id=thread_id)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_continue_request(self, py_db, request):
        '''
        :param ContinueRequest request:
        '''
        arguments = request.arguments  # : :type arguments: ContinueArguments
        thread_id = arguments.threadId

        def on_resumed():
            body = {'allThreadsContinued': thread_id == '*'}
            response = pydevd_base_schema.build_response(request, kwargs={'body': body})
            cmd = NetCommand(CMD_RETURN, 0, response, is_json=True)
            py_db.writer.add_command(cmd)

        # Only send resumed notification when it has actually resumed!
        # (otherwise the user could send a continue, receive the notification and then
        # request a new pause which would be paused without sending any notification as
        # it didn't really run in the first place).
        py_db.threads_suspended_single_notification.add_on_resumed_callback(on_resumed)
        self.api.request_resume_thread(thread_id)

    def on_next_request(self, py_db, request):
        '''
        :param NextRequest request:
        '''
        arguments = request.arguments  # : :type arguments: NextArguments
        thread_id = arguments.threadId

        if py_db.get_use_libraries_filter():
            step_cmd_id = CMD_STEP_OVER_MY_CODE
        else:
            step_cmd_id = CMD_STEP_OVER

        self.api.request_step(py_db, thread_id, step_cmd_id)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_stepin_request(self, py_db, request):
        '''
        :param StepInRequest request:
        '''
        arguments = request.arguments  # : :type arguments: StepInArguments
        thread_id = arguments.threadId

        target_id = arguments.targetId
        if target_id is not None:
            thread = pydevd_find_thread_by_id(thread_id)
            if thread is None:
                response = Response(
                    request_seq=request.seq,
                    success=False,
                    command=request.command,
                    message='Unable to find thread from thread_id: %s' % (thread_id,),
                    body={},
                )
                return NetCommand(CMD_RETURN, 0, response, is_json=True)

            info = set_additional_thread_info(thread)
            target_id_to_smart_step_into_variant = info.target_id_to_smart_step_into_variant
            if not target_id_to_smart_step_into_variant:
                variables_response = pydevd_base_schema.build_response(
                    request,
                    kwargs={
                        'success': False,
                        'message': 'Unable to step into target (no targets are saved in the thread info).'
                    })
                return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

            variant = target_id_to_smart_step_into_variant.get(target_id)
            if variant is not None:
                parent = variant.parent
                if parent is not None:
                    self.api.request_smart_step_into(py_db, request.seq, thread_id, parent.offset, variant.offset)
                else:
                    self.api.request_smart_step_into(py_db, request.seq, thread_id, variant.offset, -1)
            else:
                variables_response = pydevd_base_schema.build_response(
                    request,
                    kwargs={
                        'success': False,
                        'message': 'Unable to find step into target %s. Available targets: %s' % (
                            target_id, target_id_to_smart_step_into_variant)
                    })
                return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

        else:
            if py_db.get_use_libraries_filter():
                step_cmd_id = CMD_STEP_INTO_MY_CODE
            else:
                step_cmd_id = CMD_STEP_INTO

            self.api.request_step(py_db, thread_id, step_cmd_id)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_stepintargets_request(self, py_db, request):
        '''
        :param StepInTargetsRequest request:
        '''
        frame_id = request.arguments.frameId
        thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
            frame_id)

        if thread_id is None:
            body = StepInTargetsResponseBody([])
            variables_response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': body,
                    'success': False,
                    'message': 'Unable to get thread_id from frame_id (thread to get step in targets seems to have resumed already).'
                })
            return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

        py_db.post_method_as_internal_command(
            thread_id, internal_get_step_in_targets_json, request.seq, thread_id, frame_id, request, set_additional_thread_info)

    def on_stepout_request(self, py_db, request):
        '''
        :param StepOutRequest request:
        '''
        arguments = request.arguments  # : :type arguments: StepOutArguments
        thread_id = arguments.threadId

        if py_db.get_use_libraries_filter():
            step_cmd_id = CMD_STEP_RETURN_MY_CODE
        else:
            step_cmd_id = CMD_STEP_RETURN

        self.api.request_step(py_db, thread_id, step_cmd_id)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def _get_hit_condition_expression(self, hit_condition):
        '''Following hit condition values are supported

        * x or == x when breakpoint is hit x times
        * >= x when breakpoint is hit more than or equal to x times
        * % x when breakpoint is hit multiple of x times

        Returns '@HIT@ == x' where @HIT@ will be replaced by number of hits
        '''
        if not hit_condition:
            return None

        expr = hit_condition.strip()
        try:
            int(expr)
            return '@HIT@ == {}'.format(expr)
        except ValueError:
            pass

        if expr.startswith('%'):
            return '@HIT@ {} == 0'.format(expr)

        if expr.startswith('==') or \
            expr.startswith('>') or \
            expr.startswith('<'):
            return '@HIT@ {}'.format(expr)

        return hit_condition

    def on_disconnect_request(self, py_db, request):
        '''
        :param DisconnectRequest request:
        '''
        if request.arguments.terminateDebuggee:
            self._request_terminate_process(py_db)
            response = pydevd_base_schema.build_response(request)
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

        self._launch_or_attach_request_done = False
        py_db.enable_output_redirection(False, False)
        self.api.request_disconnect(py_db, resume_threads=True)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def _verify_launch_or_attach_done(self, request):
        if not self._launch_or_attach_request_done:
            # Note that to validate the breakpoints we need the launch request to be done already
            # (otherwise the filters wouldn't be set for the breakpoint validation).
            if request.command == 'setFunctionBreakpoints':
                body = SetFunctionBreakpointsResponseBody([])
            else:
                body = SetBreakpointsResponseBody([])
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': body,
                    'success': False,
                    'message': 'Breakpoints may only be set after the launch request is received.'
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_setfunctionbreakpoints_request(self, py_db, request):
        '''
        :param SetFunctionBreakpointsRequest request:
        '''
        response = self._verify_launch_or_attach_done(request)
        if response is not None:
            return response

        arguments = request.arguments  # : :type arguments: SetFunctionBreakpointsArguments
        function_breakpoints = []
        suspend_policy = 'ALL'

        # Not currently covered by the DAP.
        is_logpoint = False
        expression = None

        breakpoints_set = []
        for bp in arguments.breakpoints:
            hit_condition = self._get_hit_condition_expression(bp.get('hitCondition'))
            condition = bp.get('condition')

            function_breakpoints.append(
                FunctionBreakpoint(bp['name'], condition, expression, suspend_policy, hit_condition, is_logpoint))

            # Note: always succeeds.
            breakpoints_set.append(pydevd_schema.Breakpoint(
                    verified=True, id=self._next_breakpoint_id()).to_dict())

        self.api.set_function_breakpoints(py_db, function_breakpoints)

        body = {'breakpoints': breakpoints_set}
        set_breakpoints_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)

    def on_setbreakpoints_request(self, py_db, request):
        '''
        :param SetBreakpointsRequest request:
        '''
        response = self._verify_launch_or_attach_done(request)
        if response is not None:
            return response

        arguments = request.arguments  # : :type arguments: SetBreakpointsArguments
        # TODO: Path is optional here it could be source reference.
        filename = self.api.filename_to_str(arguments.source.path)
        func_name = 'None'

        self.api.remove_all_breakpoints(py_db, filename)

        btype = 'python-line'
        suspend_policy = 'ALL'

        if not filename.lower().endswith('.py'):  # Note: check based on original file, not mapping.
            if self._options.django_debug:
                btype = 'django-line'
            elif self._options.flask_debug:
                btype = 'jinja2-line'

        breakpoints_set = []

        for source_breakpoint in arguments.breakpoints:
            source_breakpoint = SourceBreakpoint(**source_breakpoint)
            line = source_breakpoint.line
            condition = source_breakpoint.condition
            breakpoint_id = self._next_breakpoint_id()

            hit_condition = self._get_hit_condition_expression(source_breakpoint.hitCondition)
            log_message = source_breakpoint.logMessage
            if not log_message:
                is_logpoint = None
                expression = None
            else:
                is_logpoint = True
                expression = convert_dap_log_message_to_expression(log_message)

            on_changed_breakpoint_state = partial(self._on_changed_breakpoint_state, py_db, arguments.source)
            result = self.api.add_breakpoint(
                py_db, filename, btype, breakpoint_id, line, condition, func_name, expression,
                suspend_policy, hit_condition, is_logpoint, adjust_line=True, on_changed_breakpoint_state=on_changed_breakpoint_state)

            bp = self._create_breakpoint_from_add_breakpoint_result(py_db, arguments.source, breakpoint_id, result)
            breakpoints_set.append(bp)

        body = {'breakpoints': breakpoints_set}
        set_breakpoints_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)

    def _on_changed_breakpoint_state(self, py_db, source, breakpoint_id, result):
        bp = self._create_breakpoint_from_add_breakpoint_result(py_db, source, breakpoint_id, result)
        body = BreakpointEventBody(
            reason='changed',
            breakpoint=bp,
        )
        event = BreakpointEvent(body)
        event_id = 0  # Actually ignored in this case
        py_db.writer.add_command(NetCommand(event_id, 0, event, is_json=True))

    def _create_breakpoint_from_add_breakpoint_result(self, py_db, source, breakpoint_id, result):
        error_code = result.error_code

        if error_code:
            if error_code == self.api.ADD_BREAKPOINT_FILE_NOT_FOUND:
                error_msg = 'Breakpoint in file that does not exist.'

            elif error_code == self.api.ADD_BREAKPOINT_FILE_EXCLUDED_BY_FILTERS:
                error_msg = 'Breakpoint in file excluded by filters.'
                if py_db.get_use_libraries_filter():
                    error_msg += ('\nNote: may be excluded because of "justMyCode" option (default == true).'
                                  'Try setting \"justMyCode\": false in the debug configuration (e.g., launch.json).\n')

            elif error_code == self.api.ADD_BREAKPOINT_LAZY_VALIDATION:
                error_msg = 'Waiting for code to be loaded to verify breakpoint.'

            elif error_code == self.api.ADD_BREAKPOINT_INVALID_LINE:
                error_msg = 'Breakpoint added to invalid line.'

            else:
                # Shouldn't get here.
                error_msg = 'Breakpoint not validated (reason unknown -- please report as bug).'

            return pydevd_schema.Breakpoint(
                verified=False, id=breakpoint_id, line=result.translated_line, message=error_msg, source=source).to_dict()
        else:
            return pydevd_schema.Breakpoint(
                verified=True, id=breakpoint_id, line=result.translated_line, source=source).to_dict()

    def on_setexceptionbreakpoints_request(self, py_db, request):
        '''
        :param SetExceptionBreakpointsRequest request:
        '''
        # : :type arguments: SetExceptionBreakpointsArguments
        arguments = request.arguments
        filters = arguments.filters
        exception_options = arguments.exceptionOptions
        self.api.remove_all_exception_breakpoints(py_db)

        # Can't set these in the DAP.
        condition = None
        expression = None
        notify_on_first_raise_only = False

        ignore_libraries = 1 if py_db.get_use_libraries_filter() else 0

        if exception_options:
            break_raised = False
            break_uncaught = False

            for option in exception_options:
                option = ExceptionOptions(**option)
                if not option.path:
                    continue

                # never: never breaks
                #
                # always: always breaks
                #
                # unhandled: breaks when exception unhandled
                #
                # userUnhandled: breaks if the exception is not handled by user code

                notify_on_handled_exceptions = 1 if option.breakMode == 'always' else 0
                notify_on_unhandled_exceptions = 1 if option.breakMode == 'unhandled' else 0
                notify_on_user_unhandled_exceptions = 1 if option.breakMode == 'userUnhandled' else 0
                exception_paths = option.path
                break_raised |= notify_on_handled_exceptions
                break_uncaught |= notify_on_unhandled_exceptions

                exception_names = []
                if len(exception_paths) == 0:
                    continue

                elif len(exception_paths) == 1:
                    if 'Python Exceptions' in exception_paths[0]['names']:
                        exception_names = ['BaseException']

                else:
                    path_iterator = iter(exception_paths)
                    if 'Python Exceptions' in next(path_iterator)['names']:
                        for path in path_iterator:
                            for ex_name in path['names']:
                                exception_names.append(ex_name)

                for exception_name in exception_names:
                    self.api.add_python_exception_breakpoint(
                        py_db,
                        exception_name,
                        condition,
                        expression,
                        notify_on_handled_exceptions,
                        notify_on_unhandled_exceptions,
                        notify_on_user_unhandled_exceptions,
                        notify_on_first_raise_only,
                        ignore_libraries
                    )

        else:
            break_raised = 'raised' in filters
            break_uncaught = 'uncaught' in filters
            break_user = 'userUnhandled' in filters
            if break_raised or break_uncaught or break_user:
                notify_on_handled_exceptions = 1 if break_raised else 0
                notify_on_unhandled_exceptions = 1 if break_uncaught else 0
                notify_on_user_unhandled_exceptions = 1 if break_user else 0
                exception = 'BaseException'

                self.api.add_python_exception_breakpoint(
                    py_db,
                    exception,
                    condition,
                    expression,
                    notify_on_handled_exceptions,
                    notify_on_unhandled_exceptions,
                    notify_on_user_unhandled_exceptions,
                    notify_on_first_raise_only,
                    ignore_libraries
                )

        if break_raised:
            btype = None
            if self._options.django_debug:
                btype = 'django'
            elif self._options.flask_debug:
                btype = 'jinja2'

            if btype:
                self.api.add_plugins_exception_breakpoint(
                    py_db, btype, 'BaseException')  # Note: Exception name could be anything here.

        # Note: no body required on success.
        set_breakpoints_response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, set_breakpoints_response, is_json=True)

    def on_stacktrace_request(self, py_db, request):
        '''
        :param StackTraceRequest request:
        '''
        # : :type stack_trace_arguments: StackTraceArguments
        stack_trace_arguments = request.arguments
        thread_id = stack_trace_arguments.threadId

        if stack_trace_arguments.startFrame:
            start_frame = int(stack_trace_arguments.startFrame)
        else:
            start_frame = 0

        if stack_trace_arguments.levels:
            levels = int(stack_trace_arguments.levels)
        else:
            levels = 0

        fmt = stack_trace_arguments.format
        if hasattr(fmt, 'to_dict'):
            fmt = fmt.to_dict()
        self.api.request_stack(py_db, request.seq, thread_id, fmt=fmt, start_frame=start_frame, levels=levels)

    def on_exceptioninfo_request(self, py_db, request):
        '''
        :param ExceptionInfoRequest request:
        '''
        # : :type exception_into_arguments: ExceptionInfoArguments
        exception_into_arguments = request.arguments
        thread_id = exception_into_arguments.threadId
        max_frames = self._options.max_exception_stack_frames
        thread = pydevd_find_thread_by_id(thread_id)
        if thread is not None:
            self.api.request_exception_info_json(py_db, request, thread_id, thread, max_frames)
        else:
            response = Response(
                request_seq=request.seq,
                success=False,
                command=request.command,
                message='Unable to find thread from thread_id: %s' % (thread_id,),
                body={},
            )
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_scopes_request(self, py_db, request):
        '''
        Scopes are the top-level items which appear for a frame (so, we receive the frame id
        and provide the scopes it has).

        :param ScopesRequest request:
        '''
        frame_id = request.arguments.frameId

        variables_reference = frame_id
        scopes = [
            Scope('Locals', ScopeRequest(int(variables_reference), 'locals'), False, presentationHint='locals'),
            Scope('Globals', ScopeRequest(int(variables_reference), 'globals'), False),
        ]
        body = ScopesResponseBody(scopes)
        scopes_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, scopes_response, is_json=True)

    def on_evaluate_request(self, py_db, request):
        '''
        :param EvaluateRequest request:
        '''
        # : :type arguments: EvaluateArguments
        arguments = request.arguments

        if arguments.frameId is None:
            self.api.request_exec_or_evaluate_json(py_db, request, thread_id='*')
        else:
            thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
                arguments.frameId)

            if thread_id is not None:
                self.api.request_exec_or_evaluate_json(
                    py_db, request, thread_id)
            else:
                body = EvaluateResponseBody('', 0)
                response = pydevd_base_schema.build_response(
                    request,
                    kwargs={
                        'body': body,
                        'success': False,
                        'message': 'Unable to find thread for evaluation.'
                    })
                return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_setexpression_request(self, py_db, request):
        # : :type arguments: SetExpressionArguments
        arguments = request.arguments

        thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
            arguments.frameId)

        if thread_id is not None:
            self.api.request_set_expression_json(py_db, request, thread_id)
        else:
            body = SetExpressionResponseBody('')
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': body,
                    'success': False,
                    'message': 'Unable to find thread to set expression.'
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_variables_request(self, py_db, request):
        '''
        Variables can be asked whenever some place returned a variables reference (so, it
        can be a scope gotten from on_scopes_request, the result of some evaluation, etc.).

        Note that in the DAP the variables reference requires a unique int... the way this works for
        pydevd is that an instance is generated for that specific variable reference and we use its
        id(instance) to identify it to make sure all items are unique (and the actual {id->instance}
        is added to a dict which is only valid while the thread is suspended and later cleared when
        the related thread resumes execution).

        see: SuspendedFramesManager

        :param VariablesRequest request:
        '''
        arguments = request.arguments  # : :type arguments: VariablesArguments
        variables_reference = arguments.variablesReference

        if isinstance(variables_reference, ScopeRequest):
            variables_reference = variables_reference.variable_reference

        thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
            variables_reference)
        if thread_id is not None:
            self.api.request_get_variable_json(py_db, request, thread_id)
        else:
            variables = []
            body = VariablesResponseBody(variables)
            variables_response = pydevd_base_schema.build_response(request, kwargs={
                'body': body,
                'success': False,
                'message': 'Unable to find thread to evaluate variable reference.'
            })
            return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

    def on_setvariable_request(self, py_db, request):
        arguments = request.arguments  # : :type arguments: SetVariableArguments
        variables_reference = arguments.variablesReference

        if isinstance(variables_reference, ScopeRequest):
            variables_reference = variables_reference.variable_reference

        if arguments.name.startswith('(return) '):
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': SetVariableResponseBody(''),
                    'success': False,
                    'message': 'Cannot change return value'
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

        thread_id = py_db.suspended_frames_manager.get_thread_id_for_variable_reference(
            variables_reference)

        if thread_id is not None:
            self.api.request_change_variable_json(py_db, request, thread_id)
        else:
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': SetVariableResponseBody(''),
                    'success': False,
                    'message': 'Unable to find thread to evaluate variable reference.'
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_modules_request(self, py_db, request):
        modules_manager = py_db.cmd_factory.modules_manager  # : :type modules_manager: ModulesManager
        modules_info = modules_manager.get_modules_info()
        body = ModulesResponseBody(modules_info)
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, variables_response, is_json=True)

    def on_source_request(self, py_db, request):
        '''
        :param SourceRequest request:
        '''
        source_reference = request.arguments.sourceReference
        server_filename = None
        content = None

        if source_reference != 0:
            server_filename = pydevd_file_utils.get_server_filename_from_source_reference(source_reference)
            if not server_filename:
                server_filename = pydevd_file_utils.get_source_reference_filename_from_linecache(source_reference)

            if server_filename:
                # Try direct file access first - it's much faster when available.
                try:
                    with open(server_filename, 'r') as stream:
                        content = stream.read()
                except:
                    pass

                if content is None:
                    # File might not exist at all, or we might not have a permission to read it,
                    # but it might also be inside a zipfile, or an IPython cell. In this case,
                    # linecache might still be able to retrieve the source.
                    lines = (linecache.getline(server_filename, i) for i in itertools.count(1))
                    lines = itertools.takewhile(bool, lines)  # empty lines are '\n', EOF is ''

                    # If we didn't get at least one line back, reset it to None so that it's
                    # reported as error below, and not as an empty file.
                    content = ''.join(lines) or None

            if content is None:
                frame_id = pydevd_file_utils.get_frame_id_from_source_reference(source_reference)
                pydev_log.debug('Found frame id: %s for source reference: %s', frame_id, source_reference)
                if frame_id is not None:
                    try:
                        content = self.api.get_decompiled_source_from_frame_id(py_db, frame_id)
                    except Exception:
                        pydev_log.exception('Error getting source for frame id: %s', frame_id)
                        content = None

        body = SourceResponseBody(content or '')
        response_args = {'body': body}

        if content is None:
            if source_reference == 0:
                message = 'Source unavailable'
            elif server_filename:
                message = 'Unable to retrieve source for %s' % (server_filename,)
            else:
                message = 'Invalid sourceReference %d' % (source_reference,)
            response_args.update({'success': False, 'message': message})

        response = pydevd_base_schema.build_response(request, kwargs=response_args)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_gototargets_request(self, py_db, request):
        path = request.arguments.source.path
        line = request.arguments.line
        target_id = self._goto_targets_map.obtain_key((path, line))
        target = {
            'id': target_id,
            'label': '%s:%s' % (path, line),
            'line': line
        }
        body = GotoTargetsResponseBody(targets=[target])
        response_args = {'body': body}
        response = pydevd_base_schema.build_response(request, kwargs=response_args)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_goto_request(self, py_db, request):
        target_id = int(request.arguments.targetId)
        thread_id = request.arguments.threadId
        try:
            path, line = self._goto_targets_map.obtain_value(target_id)
        except KeyError:
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': {},
                    'success': False,
                    'message': 'Unknown goto target id: %d' % (target_id,),
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

        self.api.request_set_next(py_db, request.seq, thread_id, CMD_SET_NEXT_STATEMENT, path, line, '*')
        # See 'NetCommandFactoryJson.make_set_next_stmnt_status_message' for response
        return None

    def on_setdebuggerproperty_request(self, py_db, request):
        args = request.arguments  # : :type args: SetDebuggerPropertyArguments
        if args.ideOS is not None:
            self.api.set_ide_os(args.ideOS)

        if args.dontTraceStartPatterns is not None and args.dontTraceEndPatterns is not None:
            start_patterns = tuple(args.dontTraceStartPatterns)
            end_patterns = tuple(args.dontTraceEndPatterns)
            self.api.set_dont_trace_start_end_patterns(py_db, start_patterns, end_patterns)

        if args.skipSuspendOnBreakpointException is not None:
            py_db.skip_suspend_on_breakpoint_exception = tuple(
                get_exception_class(x) for x in args.skipSuspendOnBreakpointException)

        if args.skipPrintBreakpointException is not None:
            py_db.skip_print_breakpoint_exception = tuple(
                get_exception_class(x) for x in args.skipPrintBreakpointException)

        if args.multiThreadsSingleNotification is not None:
            py_db.multi_threads_single_notification = args.multiThreadsSingleNotification

        # TODO: Support other common settings. Note that not all of these might be relevant to python.
        # JustMyCodeStepping: 0 or 1
        # AllowOutOfProcessSymbols: 0 or 1
        # DisableJITOptimization: 0 or 1
        # InterpreterOptions: 0 or 1
        # StopOnExceptionCrossingManagedBoundary: 0 or 1
        # WarnIfNoUserCodeOnLaunch: 0 or 1
        # EnableStepFiltering: true of false

        response = pydevd_base_schema.build_response(request, kwargs={'body': {}})
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_pydevdsysteminfo_request(self, py_db, request):
        try:
            pid = os.getpid()
        except AttributeError:
            pid = None

        # It's possible to have the ppid reported from args. In this case, use that instead of the
        # real ppid (athough we're using `ppid`, what we want in meaning is the `launcher_pid` --
        # so, if a python process is launched from another python process, consider that process the
        # parent and not any intermediary stubs).

        ppid = py_db.get_arg_ppid() or self.api.get_ppid()

        try:
            impl_desc = platform.python_implementation()
        except AttributeError:
            impl_desc = PY_IMPL_NAME

        py_info = pydevd_schema.PydevdPythonInfo(
            version=PY_VERSION_STR,
            implementation=pydevd_schema.PydevdPythonImplementationInfo(
                name=PY_IMPL_NAME,
                version=PY_IMPL_VERSION_STR,
                description=impl_desc,
            )
        )
        platform_info = pydevd_schema.PydevdPlatformInfo(name=sys.platform)
        process_info = pydevd_schema.PydevdProcessInfo(
            pid=pid,
            ppid=ppid,
            executable=sys.executable,
            bitness=64 if IS_64BIT_PROCESS else 32,
        )
        pydevd_info = pydevd_schema.PydevdInfo(
            usingCython=USING_CYTHON,
            usingFrameEval=USING_FRAME_EVAL,
        )
        body = {
            'python': py_info,
            'platform': platform_info,
            'process': process_info,
            'pydevd': pydevd_info,
        }
        response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

    def on_setpydevdsourcemap_request(self, py_db, request):
        args = request.arguments  # : :type args: SetPydevdSourceMapArguments
        SourceMappingEntry = self.api.SourceMappingEntry

        path = args.source.path
        source_maps = args.pydevdSourceMaps
        # : :type source_map: PydevdSourceMap
        new_mappings = [
            SourceMappingEntry(
                source_map['line'],
                source_map['endLine'],
                source_map['runtimeLine'],
                self.api.filename_to_str(source_map['runtimeSource']['path'])
            ) for source_map in source_maps
        ]

        error_msg = self.api.set_source_mapping(py_db, path, new_mappings)
        if error_msg:
            response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': {},
                    'success': False,
                    'message': error_msg,
                })
            return NetCommand(CMD_RETURN, 0, response, is_json=True)

        response = pydevd_base_schema.build_response(request)
        return NetCommand(CMD_RETURN, 0, response, is_json=True)

