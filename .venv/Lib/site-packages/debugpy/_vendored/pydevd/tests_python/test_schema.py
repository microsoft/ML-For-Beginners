from _pydevd_bundle._debug_adapter.pydevd_schema import InitializeRequest, \
    InitializeRequestArguments, InitializeResponse, Capabilities
from _pydevd_bundle._debug_adapter import pydevd_schema, pydevd_base_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import ThreadsResponse


def test_schema():

    json_msg = '''
{
    "arguments": {
        "adapterID": "pydevd",
        "clientID": "vscode",
        "clientName": "Visual Studio Code",
        "columnsStartAt1": true,
        "linesStartAt1": true,
        "locale": "en-us",
        "pathFormat": "path",
        "supportsRunInTerminalRequest": true,
        "supportsVariablePaging": true,
        "supportsVariableType": true
    },
    "command": "initialize",
    "seq": 1,
    "type": "request"
}'''

    initialize_request = pydevd_base_schema.from_json(json_msg)
    assert initialize_request.__class__ == InitializeRequest
    assert initialize_request.arguments.__class__ == InitializeRequestArguments
    assert initialize_request.arguments.adapterID == 'pydevd'
    assert initialize_request.command == 'initialize'
    assert initialize_request.type == 'request'
    assert initialize_request.seq == 1

    response = pydevd_base_schema.build_response(initialize_request)
    assert response.__class__ == InitializeResponse
    assert response.seq == -1  # Must be set before sending
    assert response.command == 'initialize'
    assert response.type == 'response'
    assert response.body.__class__ == Capabilities

    assert response.to_dict() == {
        "seq":-1,
        "type": "response",
        "request_seq": 1,
        "success": True,
        "command": "initialize",
        "body": {}
    }

    capabilities = response.body  # : :type capabilities: Capabilities
    capabilities.supportsCompletionsRequest = True
    assert response.to_dict() == {
        "seq":-1,
        "type": "response",
        "request_seq": 1,
        "success": True,
        "command": "initialize",
        "body": {'supportsCompletionsRequest':True}
    }

    initialize_event = pydevd_schema.InitializedEvent()
    assert initialize_event.to_dict() == {
        "seq":-1,
        "type": "event",
        "event": "initialized"
    }


def test_schema_translation_frame():
    pydevd_base_schema.BaseSchema.initialize_ids_translation()
    stack_trace_arguments = pydevd_schema.StackTraceArguments(threadId=1)
    stack_trace_request = pydevd_schema.StackTraceRequest(stack_trace_arguments)

    stackFrames = [
        pydevd_schema.StackFrame(id=2 ** 45, name='foo', line=1, column=1).to_dict(),
        pydevd_schema.StackFrame(id=2 ** 46, name='bar', line=1, column=1).to_dict(),
    ]
    body = pydevd_schema.StackTraceResponseBody(stackFrames)
    stack_trace_response = pydevd_base_schema.build_response(stack_trace_request, kwargs=dict(body=body))
    as_dict = stack_trace_response.to_dict(update_ids_to_dap=True)
    assert as_dict == {
        'type': 'response',
        'request_seq':-1,
        'success': True,
        'command': 'stackTrace',
        'body': {'stackFrames': [
            {'id': 1, 'name': 'foo', 'line': 1, 'column': 1, 'source': {}},
            {'id': 2, 'name': 'bar', 'line': 1, 'column': 1, 'source': {}},
        ]},
        'seq':-1}

    reconstructed = pydevd_base_schema.from_dict(as_dict, update_ids_from_dap=True)
    assert reconstructed.to_dict() == {
        'type': 'response',
        'request_seq':-1,
        'success': True,
        'command': 'stackTrace',
        'body': {'stackFrames': [
            {'id': 2 ** 45, 'name': 'foo', 'line': 1, 'column': 1, 'source': {}},
            {'id': 2 ** 46, 'name': 'bar', 'line': 1, 'column': 1, 'source': {}}
        ]},
        'seq':-1
    }


def test_schema_translation_thread():
    from _pydevd_bundle._debug_adapter.pydevd_schema import ThreadsRequest
    pydevd_base_schema.BaseSchema.initialize_ids_translation()

    threads = [
        pydevd_schema.Thread(id=2 ** 45, name='foo').to_dict(),
        pydevd_schema.Thread(id=2 ** 46, name='bar').to_dict(),
    ]
    body = pydevd_schema.ThreadsResponseBody(threads)
    threads_request = ThreadsRequest()
    threads_response = pydevd_base_schema.build_response(threads_request, kwargs=dict(body=body))
    as_dict = threads_response.to_dict(update_ids_to_dap=True)
    assert as_dict == {
        'type': 'response',
        'request_seq':-1,
        'success': True,
        'command': 'threads',
        'body': {'threads': [
            {'id': 1, 'name': 'foo'},
            {'id': 2, 'name': 'bar'},
        ]},
        'seq':-1}

    reconstructed = pydevd_base_schema.from_dict(as_dict, update_ids_from_dap=True)
    assert reconstructed.to_dict() == {
        'type': 'response',
        'request_seq':-1,
        'success': True,
        'command': 'threads',
        'body': {'threads': [
            {'id': 2 ** 45, 'name': 'foo'},
            {'id': 2 ** 46, 'name': 'bar'}
        ]},
        'seq':-1
    }
