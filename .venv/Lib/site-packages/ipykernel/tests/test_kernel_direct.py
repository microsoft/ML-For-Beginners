"""test the IPython Kernel"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import asyncio
import os
import warnings

import pytest

if os.name == "nt":
    pytest.skip("skipping tests on windows", allow_module_level=True)


async def test_direct_kernel_info_request(kernel):
    reply = await kernel.test_shell_message("kernel_info_request", {})
    assert reply["header"]["msg_type"] == "kernel_info_reply"


async def test_direct_execute_request(kernel):
    reply = await kernel.test_shell_message("execute_request", dict(code="hello", silent=False))
    assert reply["header"]["msg_type"] == "execute_reply"


async def test_direct_execute_request_aborting(kernel):
    kernel._aborting = True
    reply = await kernel.test_shell_message("execute_request", dict(code="hello", silent=False))
    assert reply["header"]["msg_type"] == "execute_reply"
    assert reply["content"]["status"] == "aborted"


async def test_direct_execute_request_error(kernel):
    await kernel.execute_request(None, None, None)


async def test_complete_request(kernel):
    reply = await kernel.test_shell_message("complete_request", dict(code="hello", cursor_pos=0))
    assert reply["header"]["msg_type"] == "complete_reply"


async def test_inspect_request(kernel):
    reply = await kernel.test_shell_message("inspect_request", dict(code="hello", cursor_pos=0))
    assert reply["header"]["msg_type"] == "inspect_reply"


async def test_history_request(kernel):
    reply = await kernel.test_shell_message(
        "history_request", dict(hist_access_type="", output="", raw="")
    )
    assert reply["header"]["msg_type"] == "history_reply"
    reply = await kernel.test_shell_message(
        "history_request", dict(hist_access_type="tail", output="", raw="")
    )
    assert reply["header"]["msg_type"] == "history_reply"
    reply = await kernel.test_shell_message(
        "history_request", dict(hist_access_type="range", output="", raw="")
    )
    assert reply["header"]["msg_type"] == "history_reply"
    reply = await kernel.test_shell_message(
        "history_request", dict(hist_access_type="search", output="", raw="")
    )
    assert reply["header"]["msg_type"] == "history_reply"


async def test_comm_info_request(kernel):
    reply = await kernel.test_shell_message("comm_info_request")
    assert reply["header"]["msg_type"] == "comm_info_reply"


async def test_direct_interrupt_request(kernel):
    reply = await kernel.test_control_message("interrupt_request", {})
    assert reply["header"]["msg_type"] == "interrupt_reply"
    assert reply["content"] == {"status": "ok"}

    # test failure on interrupt request
    def raiseOSError():
        raise OSError("evalue")

    kernel._send_interrupt_children = raiseOSError
    reply = await kernel.test_control_message("interrupt_request", {})
    assert reply["header"]["msg_type"] == "interrupt_reply"
    assert reply["content"]["status"] == "error"
    assert reply["content"]["ename"] == "OSError"
    assert reply["content"]["evalue"] == "evalue"
    assert len(reply["content"]["traceback"]) > 0


async def test_direct_shutdown_request(kernel):
    reply = await kernel.test_shell_message("shutdown_request", dict(restart=False))
    assert reply["header"]["msg_type"] == "shutdown_reply"
    reply = await kernel.test_shell_message("shutdown_request", dict(restart=True))
    assert reply["header"]["msg_type"] == "shutdown_reply"


async def test_is_complete_request(kernel):
    reply = await kernel.test_shell_message("is_complete_request", dict(code="hello"))
    assert reply["header"]["msg_type"] == "is_complete_reply"


async def test_direct_debug_request(kernel):
    reply = await kernel.test_control_message("debug_request", {})
    assert reply["header"]["msg_type"] == "debug_reply"


async def test_deprecated_features(kernel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        header = kernel._parent_header
        assert isinstance(header, dict)
        shell_streams = kernel.shell_streams
        assert len(shell_streams) == 1
        assert shell_streams[0] == kernel.shell_stream
        warnings.simplefilter("ignore", RuntimeWarning)
        kernel.shell_streams = [kernel.shell_stream, kernel.shell_stream]


async def test_process_control(kernel):
    from jupyter_client.session import DELIM

    class FakeMsg:
        def __init__(self, bytes):
            self.bytes = bytes

    await kernel.process_control([FakeMsg(DELIM), 1])
    msg = kernel._prep_msg("does_not_exist")
    await kernel.process_control(msg)


def test_should_handle(kernel):
    msg = kernel.session.msg("debug_request", {})
    kernel.aborted.add(msg["header"]["msg_id"])
    assert not kernel.should_handle(kernel.control_stream, msg, [])


async def test_dispatch_shell(kernel):
    from jupyter_client.session import DELIM

    class FakeMsg:
        def __init__(self, bytes):
            self.bytes = bytes

    await kernel.dispatch_shell([FakeMsg(DELIM), 1])
    msg = kernel._prep_msg("does_not_exist")
    await kernel.dispatch_shell(msg)


async def test_do_one_iteration(kernel):
    kernel.msg_queue = asyncio.Queue()
    await kernel.do_one_iteration()


async def test_publish_debug_event(kernel):
    kernel._publish_debug_event({})


async def test_connect_request(kernel):
    await kernel.connect_request(kernel.shell_stream, "foo", {})


async def test_send_interrupt_children(kernel):
    kernel._send_interrupt_children()


# TODO: this causes deadlock
# async def test_direct_usage_request(kernel):
#     reply = await kernel.test_control_message("usage_request", {})
#     assert reply['header']['msg_type'] == 'usage_reply'
