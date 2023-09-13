import sys

import pytest

from .utils import TIMEOUT, get_reply, new_kernel

seq = 0

# Skip if debugpy is not available
pytest.importorskip("debugpy")


def wait_for_debug_request(kernel, command, arguments=None, full_reply=False):
    """Carry out a debug request and return the reply content.

    It does not check if the request was successful.
    """
    global seq
    seq += 1

    msg = kernel.session.msg(
        "debug_request",
        {
            "type": "request",
            "seq": seq,
            "command": command,
            "arguments": arguments or {},
        },
    )
    kernel.control_channel.send(msg)
    reply = get_reply(kernel, msg["header"]["msg_id"], channel="control")
    return reply if full_reply else reply["content"]


@pytest.fixture
def kernel():
    with new_kernel() as kc:
        yield kc


@pytest.fixture
def kernel_with_debug(kernel):
    # Initialize
    wait_for_debug_request(
        kernel,
        "initialize",
        {
            "clientID": "test-client",
            "clientName": "testClient",
            "adapterID": "",
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
            "supportsVariablePaging": True,
            "supportsRunInTerminalRequest": True,
            "locale": "en",
        },
    )

    # Attach
    wait_for_debug_request(kernel, "attach")

    try:
        yield kernel
    finally:
        # Detach
        wait_for_debug_request(kernel, "disconnect", {"restart": False, "terminateDebuggee": True})


def test_debug_initialize(kernel):
    reply = wait_for_debug_request(
        kernel,
        "initialize",
        {
            "clientID": "test-client",
            "clientName": "testClient",
            "adapterID": "",
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
            "supportsVariablePaging": True,
            "supportsRunInTerminalRequest": True,
            "locale": "en",
        },
    )
    assert reply["success"]


def test_attach_debug(kernel_with_debug):
    reply = wait_for_debug_request(
        kernel_with_debug, "evaluate", {"expression": "'a' + 'b'", "context": "repl"}
    )
    assert reply["success"]
    assert reply["body"]["result"] == ""


def test_set_breakpoints(kernel_with_debug):
    code = """def f(a, b):
    c = a + b
    return c

f(2, 3)"""

    r = wait_for_debug_request(kernel_with_debug, "dumpCell", {"code": code})
    source = r["body"]["sourcePath"]

    reply = wait_for_debug_request(
        kernel_with_debug,
        "setBreakpoints",
        {
            "breakpoints": [{"line": 2}],
            "source": {"path": source},
            "sourceModified": False,
        },
    )
    assert reply["success"]
    assert len(reply["body"]["breakpoints"]) == 1
    assert reply["body"]["breakpoints"][0]["verified"]
    assert reply["body"]["breakpoints"][0]["source"]["path"] == source

    r = wait_for_debug_request(kernel_with_debug, "debugInfo")
    assert source in map(lambda b: b["source"], r["body"]["breakpoints"])  # noqa

    r = wait_for_debug_request(kernel_with_debug, "configurationDone")
    assert r["success"]


def test_stop_on_breakpoint(kernel_with_debug):
    code = """def f(a, b):
    c = a + b
    return c

f(2, 3)"""

    r = wait_for_debug_request(kernel_with_debug, "dumpCell", {"code": code})
    source = r["body"]["sourcePath"]

    wait_for_debug_request(kernel_with_debug, "debugInfo")

    wait_for_debug_request(
        kernel_with_debug,
        "setBreakpoints",
        {
            "breakpoints": [{"line": 2}],
            "source": {"path": source},
            "sourceModified": False,
        },
    )

    wait_for_debug_request(kernel_with_debug, "configurationDone", full_reply=True)

    kernel_with_debug.execute(code)

    # Wait for stop on breakpoint
    msg: dict = {"msg_type": "", "content": {}}
    while msg.get("msg_type") != "debug_event" or msg["content"].get("event") != "stopped":
        msg = kernel_with_debug.get_iopub_msg(timeout=TIMEOUT)

    assert msg["content"]["body"]["reason"] == "breakpoint"


@pytest.mark.skipif(sys.version_info >= (3, 10), reason="TODO Does not work on Python 3.10")
def test_breakpoint_in_cell_with_leading_empty_lines(kernel_with_debug):
    code = """
def f(a, b):
    c = a + b
    return c

f(2, 3)"""

    r = wait_for_debug_request(kernel_with_debug, "dumpCell", {"code": code})
    source = r["body"]["sourcePath"]

    wait_for_debug_request(kernel_with_debug, "debugInfo")

    wait_for_debug_request(
        kernel_with_debug,
        "setBreakpoints",
        {
            "breakpoints": [{"line": 6}],
            "source": {"path": source},
            "sourceModified": False,
        },
    )

    wait_for_debug_request(kernel_with_debug, "configurationDone", full_reply=True)

    kernel_with_debug.execute(code)

    # Wait for stop on breakpoint
    msg: dict = {"msg_type": "", "content": {}}
    while msg.get("msg_type") != "debug_event" or msg["content"].get("event") != "stopped":
        msg = kernel_with_debug.get_iopub_msg(timeout=TIMEOUT)

    assert msg["content"]["body"]["reason"] == "breakpoint"


def test_rich_inspect_not_at_breakpoint(kernel_with_debug):
    var_name = "text"
    value = "Hello the world"
    code = f"""{var_name}='{value}'
print({var_name})
"""

    msg_id = kernel_with_debug.execute(code)
    get_reply(kernel_with_debug, msg_id)

    r = wait_for_debug_request(kernel_with_debug, "inspectVariables")
    assert var_name in list(map(lambda v: v["name"], r["body"]["variables"]))  # noqa

    reply = wait_for_debug_request(
        kernel_with_debug,
        "richInspectVariables",
        {"variableName": var_name},
    )

    assert reply["body"]["data"] == {"text/plain": f"'{value}'"}


def test_rich_inspect_at_breakpoint(kernel_with_debug):
    code = """def f(a, b):
    c = a + b
    return c

f(2, 3)"""

    r = wait_for_debug_request(kernel_with_debug, "dumpCell", {"code": code})
    source = r["body"]["sourcePath"]

    wait_for_debug_request(
        kernel_with_debug,
        "setBreakpoints",
        {
            "breakpoints": [{"line": 2}],
            "source": {"path": source},
            "sourceModified": False,
        },
    )

    r = wait_for_debug_request(kernel_with_debug, "debugInfo")

    r = wait_for_debug_request(kernel_with_debug, "configurationDone")

    kernel_with_debug.execute(code)

    # Wait for stop on breakpoint
    msg: dict = {"msg_type": "", "content": {}}
    while msg.get("msg_type") != "debug_event" or msg["content"].get("event") != "stopped":
        msg = kernel_with_debug.get_iopub_msg(timeout=TIMEOUT)

    stacks = wait_for_debug_request(kernel_with_debug, "stackTrace", {"threadId": 1})["body"][
        "stackFrames"
    ]

    scopes = wait_for_debug_request(kernel_with_debug, "scopes", {"frameId": stacks[0]["id"]})[
        "body"
    ]["scopes"]

    locals_ = wait_for_debug_request(
        kernel_with_debug,
        "variables",
        {
            "variablesReference": next(filter(lambda s: s["name"] == "Locals", scopes))[
                "variablesReference"
            ]
        },
    )["body"]["variables"]

    reply = wait_for_debug_request(
        kernel_with_debug,
        "richInspectVariables",
        {"variableName": locals_[0]["name"], "frameId": stacks[0]["id"]},
    )

    assert reply["body"]["data"] == {"text/plain": locals_[0]["value"]}


def test_convert_to_long_pathname():
    if sys.platform == "win32":
        from ipykernel.compiler import _convert_to_long_pathname

        _convert_to_long_pathname(__file__)


def test_copy_to_globals(kernel_with_debug):
    local_var_name = "var"
    global_var_name = "var_copy"
    code = f"""from IPython.core.display import HTML
def my_test():
    {local_var_name} = HTML('<p>test content</p>')
    pass
a = 2
my_test()"""

    # Init debugger and set breakpoint
    r = wait_for_debug_request(kernel_with_debug, "dumpCell", {"code": code})
    source = r["body"]["sourcePath"]

    wait_for_debug_request(
        kernel_with_debug,
        "setBreakpoints",
        {
            "breakpoints": [{"line": 4}],
            "source": {"path": source},
            "sourceModified": False,
        },
    )

    wait_for_debug_request(kernel_with_debug, "debugInfo")

    wait_for_debug_request(kernel_with_debug, "configurationDone")

    # Execute code
    kernel_with_debug.execute(code)

    # Wait for stop on breakpoint
    msg: dict = {"msg_type": "", "content": {}}
    while msg.get("msg_type") != "debug_event" or msg["content"].get("event") != "stopped":
        msg = kernel_with_debug.get_iopub_msg(timeout=TIMEOUT)

    stacks = wait_for_debug_request(kernel_with_debug, "stackTrace", {"threadId": 1})["body"][
        "stackFrames"
    ]

    # Get local frame id
    frame_id = stacks[0]["id"]

    # Copy the variable
    wait_for_debug_request(
        kernel_with_debug,
        "copyToGlobals",
        {
            "srcVariableName": local_var_name,
            "dstVariableName": global_var_name,
            "srcFrameId": frame_id,
        },
    )

    # Get the scopes
    scopes = wait_for_debug_request(kernel_with_debug, "scopes", {"frameId": frame_id})["body"][
        "scopes"
    ]

    # Get the local variable
    locals_ = wait_for_debug_request(
        kernel_with_debug,
        "variables",
        {
            "variablesReference": next(filter(lambda s: s["name"] == "Locals", scopes))[
                "variablesReference"
            ]
        },
    )["body"]["variables"]

    local_var = None
    for variable in locals_:
        if local_var_name in variable["evaluateName"]:
            local_var = variable
    assert local_var is not None

    # Get the global variable (copy of the local variable)
    globals_ = wait_for_debug_request(
        kernel_with_debug,
        "variables",
        {
            "variablesReference": next(filter(lambda s: s["name"] == "Globals", scopes))[
                "variablesReference"
            ]
        },
    )["body"]["variables"]

    global_var = None
    for variable in globals_:
        if global_var_name in variable["evaluateName"]:
            global_var = variable
    assert global_var is not None

    # Compare local and global variable
    assert global_var["value"] == local_var["value"] and global_var["type"] == local_var["type"]
