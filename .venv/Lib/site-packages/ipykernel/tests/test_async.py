"""Test async/await integration"""

import pytest

from .test_message_spec import validate_message
from .utils import TIMEOUT, execute, flush_channels, start_new_kernel

KC = KM = None


def setup_function():
    """start the global kernel (if it isn't running) and return its client"""
    global KM, KC
    KM, KC = start_new_kernel()
    flush_channels(KC)


def teardown_function():
    assert KC is not None
    assert KM is not None
    KC.stop_channels()
    KM.shutdown_kernel(now=True)


def test_async_await():
    flush_channels(KC)
    msg_id, content = execute("import asyncio; await asyncio.sleep(0.1)", KC)
    assert content["status"] == "ok", content


@pytest.mark.parametrize("asynclib", ["asyncio", "trio", "curio"])
def test_async_interrupt(asynclib, request):
    assert KC is not None
    assert KM is not None
    try:
        __import__(asynclib)
    except ImportError:
        pytest.skip("Requires %s" % asynclib)
    request.addfinalizer(lambda: execute("%autoawait asyncio", KC))

    flush_channels(KC)
    msg_id, content = execute("%autoawait " + asynclib, KC)
    assert content["status"] == "ok", content

    flush_channels(KC)
    msg_id = KC.execute(f"print('begin'); import {asynclib}; await {asynclib}.sleep(5)")
    busy = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(busy, "status", msg_id)
    assert busy["content"]["execution_state"] == "busy"
    echo = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(echo, "execute_input")
    stream = KC.get_iopub_msg(timeout=TIMEOUT)
    # wait for the stream output to be sure kernel is in the async block
    validate_message(stream, "stream")
    assert stream["content"]["text"] == "begin\n"

    KM.interrupt_kernel()
    reply = KC.get_shell_msg()["content"]
    assert reply["status"] == "error", reply
    assert reply["ename"] in {"CancelledError", "KeyboardInterrupt"}

    flush_channels(KC)
