"""Test suite for our zeromq-based message specification."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import re
import sys
from queue import Empty

import pytest
from jupyter_client._version import version_info
from jupyter_client.blocking.client import BlockingKernelClient
from packaging.version import Version as V  # noqa
from traitlets import Bool, Dict, Enum, HasTraits, Integer, List, TraitError, Unicode, observe

from .utils import TIMEOUT, execute, flush_channels, get_reply, start_global_kernel

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
KC: BlockingKernelClient = None  # type:ignore


def setup():
    global KC
    KC = start_global_kernel()


# -----------------------------------------------------------------------------
# Message Spec References
# -----------------------------------------------------------------------------


class Reference(HasTraits):

    """
    Base class for message spec specification testing.

    This class is the core of the message specification test.  The
    idea is that child classes implement trait attributes for each
    message keys, so that message keys can be tested against these
    traits using :meth:`check` method.

    """

    def check(self, d):
        """validate a dict against our traits"""
        for key in self.trait_names():
            assert key in d
            # FIXME: always allow None, probably not a good idea
            if d[key] is None:
                continue
            try:
                setattr(self, key, d[key])
            except TraitError as e:
                assert False, str(e)


class Version(Unicode):
    def __init__(self, *args, **kwargs):
        self.min = kwargs.pop("min", None)
        self.max = kwargs.pop("max", None)
        kwargs["default_value"] = self.min
        super().__init__(*args, **kwargs)

    def validate(self, obj, value):
        if self.min and V(value) < V(self.min):
            raise TraitError(f"bad version: {value} < {self.min}")
        if self.max and (V(value) > V(self.max)):
            raise TraitError(f"bad version: {value} > {self.max}")


class RMessage(Reference):
    msg_id = Unicode()
    msg_type = Unicode()
    header = Dict()
    parent_header = Dict()
    content = Dict()

    def check(self, d):
        super().check(d)
        RHeader().check(self.header)
        if self.parent_header:
            RHeader().check(self.parent_header)


class RHeader(Reference):
    msg_id = Unicode()
    msg_type = Unicode()
    session = Unicode()
    username = Unicode()
    version = Version(min="5.0")


mime_pat = re.compile(r"^[\w\-\+\.]+/[\w\-\+\.]+$")


class MimeBundle(Reference):
    metadata = Dict()
    data = Dict()

    @observe("data")
    def _on_data_changed(self, change):
        for k, v in change["new"].items():
            assert mime_pat.match(k)
            assert isinstance(v, str)


# shell replies
class Reply(Reference):
    status = Enum(("ok", "error"), default_value="ok")


class ExecuteReply(Reply):
    execution_count = Integer()

    def check(self, d):
        Reference.check(self, d)
        if d["status"] == "ok":
            ExecuteReplyOkay().check(d)
        elif d["status"] == "error":
            ExecuteReplyError().check(d)
        elif d["status"] == "aborted":
            ExecuteReplyAborted().check(d)


class ExecuteReplyOkay(Reply):
    status = Enum(("ok",))
    user_expressions = Dict()


class ExecuteReplyError(Reply):
    status = Enum(("error",))
    ename = Unicode()
    evalue = Unicode()
    traceback = List(Unicode())


class ExecuteReplyAborted(Reply):
    status = Enum(("aborted",))


class InspectReply(Reply, MimeBundle):
    found = Bool()


class ArgSpec(Reference):
    args = List(Unicode())
    varargs = Unicode()
    varkw = Unicode()
    defaults = List()


class Status(Reference):
    execution_state = Enum(("busy", "idle", "starting"), default_value="busy")


class CompleteReply(Reply):
    matches = List(Unicode())
    cursor_start = Integer()
    cursor_end = Integer()
    status = Unicode()  # type:ignore


class LanguageInfo(Reference):
    name = Unicode("python")
    version = Unicode(sys.version.split()[0])


class KernelInfoReply(Reply):
    protocol_version = Version(min="5.0")
    implementation = Unicode("ipython")
    implementation_version = Version(min="2.1")
    language_info = Dict()
    banner = Unicode()

    def check(self, d):
        Reference.check(self, d)
        LanguageInfo().check(d["language_info"])


class ConnectReply(Reference):
    shell_port = Integer()
    control_port = Integer()
    stdin_port = Integer()
    iopub_port = Integer()
    hb_port = Integer()


class CommInfoReply(Reply):
    comms = Dict()


class IsCompleteReply(Reference):
    status = Enum(("complete", "incomplete", "invalid", "unknown"), default_value="complete")

    def check(self, d):
        Reference.check(self, d)
        if d["status"] == "incomplete":
            IsCompleteReplyIncomplete().check(d)


class IsCompleteReplyIncomplete(Reference):
    indent = Unicode()


# IOPub messages


class ExecuteInput(Reference):
    code = Unicode()
    execution_count = Integer()


class Error(ExecuteReplyError):
    """Errors are the same as ExecuteReply, but without status"""

    status = None  # type:ignore  # no status field


class Stream(Reference):
    name = Enum(("stdout", "stderr"), default_value="stdout")
    text = Unicode()


class DisplayData(MimeBundle):
    pass


class ExecuteResult(MimeBundle):
    execution_count = Integer()


class HistoryReply(Reply):
    history = List(List())


references = {
    "execute_reply": ExecuteReply(),
    "inspect_reply": InspectReply(),
    "status": Status(),
    "complete_reply": CompleteReply(),
    "kernel_info_reply": KernelInfoReply(),
    "connect_reply": ConnectReply(),
    "comm_info_reply": CommInfoReply(),
    "is_complete_reply": IsCompleteReply(),
    "execute_input": ExecuteInput(),
    "execute_result": ExecuteResult(),
    "history_reply": HistoryReply(),
    "error": Error(),
    "stream": Stream(),
    "display_data": DisplayData(),
    "header": RHeader(),
}

# -----------------------------------------------------------------------------
# Specifications of `content` part of the reply messages.
# -----------------------------------------------------------------------------


def validate_message(msg, msg_type=None, parent=None):
    """validate a message

    This is a generator, and must be iterated through to actually
    trigger each test.

    If msg_type and/or parent are given, the msg_type and/or parent msg_id
    are compared with the given values.
    """
    RMessage().check(msg)
    if msg_type:
        assert msg["msg_type"] == msg_type
    if parent:
        assert msg["parent_header"]["msg_id"] == parent
    content = msg["content"]
    ref = references[msg["msg_type"]]
    ref.check(content)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

# Shell channel


def test_execute():
    flush_channels()

    msg_id = KC.execute(code="x=1")
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "execute_reply", msg_id)


def test_execute_silent():
    flush_channels()
    msg_id, reply = execute(code="x=1", silent=True)

    # flush status=idle
    status = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(status, "status", msg_id)
    assert status["content"]["execution_state"] == "idle"

    with pytest.raises(Empty):
        KC.get_iopub_msg(timeout=0.1)

    count = reply["execution_count"]

    msg_id, reply = execute(code="x=2", silent=True)

    # flush status=idle
    status = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(status, "status", msg_id)
    assert status["content"]["execution_state"] == "idle"

    with pytest.raises(Empty):
        KC.get_iopub_msg(timeout=0.1)

    count_2 = reply["execution_count"]
    assert count_2 == count


def test_execute_error():
    flush_channels()

    msg_id, reply = execute(code="1/0")
    assert reply["status"] == "error"
    assert reply["ename"] == "ZeroDivisionError"

    error = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(error, "error", msg_id)


def test_execute_inc():
    """execute request should increment execution_count"""
    flush_channels()

    _, reply = execute(code="x=1")
    count = reply["execution_count"]

    flush_channels()

    _, reply = execute(code="x=2")
    count_2 = reply["execution_count"]
    assert count_2 == count + 1


def test_execute_stop_on_error():
    """execute request should not abort execution queue with stop_on_error False"""
    flush_channels()

    fail = "\n".join(
        [
            # sleep to ensure subsequent message is waiting in the queue to be aborted
            # async sleep to ensure coroutines are processing while this happens
            "import asyncio",
            "await asyncio.sleep(1)",
            "raise ValueError()",
        ]
    )
    KC.execute(code=fail)
    KC.execute(code='print("Hello")')
    KC.execute(code='print("world")')
    reply = KC.get_shell_msg(timeout=TIMEOUT)
    print(reply)
    reply = KC.get_shell_msg(timeout=TIMEOUT)
    assert reply["content"]["status"] == "aborted"
    # second message, too
    reply = KC.get_shell_msg(timeout=TIMEOUT)
    assert reply["content"]["status"] == "aborted"

    flush_channels()

    KC.execute(code=fail, stop_on_error=False)
    KC.execute(code='print("Hello")')
    KC.get_shell_msg(timeout=TIMEOUT)
    reply = KC.get_shell_msg(timeout=TIMEOUT)
    assert reply["content"]["status"] == "ok"


def test_non_execute_stop_on_error():
    """test that non-execute_request's are not aborted after an error"""
    flush_channels()

    fail = "\n".join(
        [
            # sleep to ensure subsequent message is waiting in the queue to be aborted
            "import time",
            "time.sleep(0.5)",
            "raise ValueError",
        ]
    )
    KC.execute(code=fail)
    KC.kernel_info()
    KC.comm_info()
    KC.inspect(code="print")
    reply = KC.get_shell_msg(timeout=TIMEOUT)  # execute
    assert reply["content"]["status"] == "error"
    reply = KC.get_shell_msg(timeout=TIMEOUT)  # kernel_info
    assert reply["content"]["status"] == "ok"
    reply = KC.get_shell_msg(timeout=TIMEOUT)  # comm_info
    assert reply["content"]["status"] == "ok"
    reply = KC.get_shell_msg(timeout=TIMEOUT)  # inspect
    assert reply["content"]["status"] == "ok"


def test_user_expressions():
    flush_channels()

    msg_id, reply = execute(code="x=1", user_expressions=dict(foo="x+1"))
    user_expressions = reply["user_expressions"]
    assert user_expressions == {
        "foo": {
            "status": "ok",
            "data": {"text/plain": "2"},
            "metadata": {},
        }
    }


def test_user_expressions_fail():
    flush_channels()

    msg_id, reply = execute(code="x=0", user_expressions=dict(foo="nosuchname"))
    user_expressions = reply["user_expressions"]
    foo = user_expressions["foo"]
    assert foo["status"] == "error"
    assert foo["ename"] == "NameError"


def test_oinfo():
    flush_channels()

    msg_id = KC.inspect("a")
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "inspect_reply", msg_id)


def test_oinfo_found():
    flush_channels()

    msg_id, reply = execute(code="a=5")

    msg_id = KC.inspect("a")
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "inspect_reply", msg_id)
    content = reply["content"]
    assert content["found"]
    text = content["data"]["text/plain"]
    assert "Type:" in text
    assert "Docstring:" in text


def test_oinfo_detail():
    flush_channels()

    msg_id, reply = execute(code="ip=get_ipython()")

    msg_id = KC.inspect("ip.object_inspect", cursor_pos=10, detail_level=1)
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "inspect_reply", msg_id)
    content = reply["content"]
    assert content["found"]
    text = content["data"]["text/plain"]
    assert "Signature:" in text
    assert "Source:" in text


def test_oinfo_not_found():
    flush_channels()

    msg_id = KC.inspect("dne")
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "inspect_reply", msg_id)
    content = reply["content"]
    assert not content["found"]


def test_complete():
    flush_channels()

    msg_id, reply = execute(code="alpha = albert = 5")

    msg_id = KC.complete("al", 2)
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "complete_reply", msg_id)
    matches = reply["content"]["matches"]
    for name in ("alpha", "albert"):
        assert name in matches


def test_kernel_info_request():
    flush_channels()

    msg_id = KC.kernel_info()
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "kernel_info_reply", msg_id)


def test_connect_request():
    flush_channels()
    msg = KC.session.msg("connect_request")
    KC.shell_channel.send(msg)
    msg_id = msg["header"]["msg_id"]
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "connect_reply", msg_id)


@pytest.mark.skipif(
    version_info < (5, 0),
    reason="earlier Jupyter Client don't have comm_info",
)
def test_comm_info_request():
    flush_channels()
    msg_id = KC.comm_info()
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "comm_info_reply", msg_id)


def test_single_payload():
    """
    We want to test the set_next_input is not triggered several time per cell.
    This is (was ?) mostly due to the fact that `?` in a loop would trigger
    several set_next_input.

    I'm tempted to thing that we actually want to _allow_ multiple
    set_next_input (that's users' choice). But that `?` itself (and ?'s
    transform) should avoid setting multiple set_next_input).
    """
    flush_channels()
    msg_id, reply = execute(
        code="ip = get_ipython()\nfor i in range(3):\n   ip.set_next_input('Hello There')\n"
    )
    payload = reply["payload"]
    next_input_pls = [pl for pl in payload if pl["source"] == "set_next_input"]
    assert len(next_input_pls) == 1


def test_is_complete():
    flush_channels()

    msg_id = KC.is_complete("a = 1")
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "is_complete_reply", msg_id)


def test_history_range():
    flush_channels()

    KC.execute(code="x=1", store_history=True)
    KC.get_shell_msg(timeout=TIMEOUT)

    msg_id = KC.history(hist_access_type="range", raw=True, output=True, start=1, stop=2, session=0)
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "history_reply", msg_id)
    content = reply["content"]
    assert len(content["history"]) == 1


def test_history_tail():
    flush_channels()

    KC.execute(code="x=1", store_history=True)
    KC.get_shell_msg(timeout=TIMEOUT)

    msg_id = KC.history(hist_access_type="tail", raw=True, output=True, n=1, session=0)
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "history_reply", msg_id)
    content = reply["content"]
    assert len(content["history"]) == 1


def test_history_search():
    flush_channels()

    KC.execute(code="x=1", store_history=True)
    KC.get_shell_msg(timeout=TIMEOUT)

    msg_id = KC.history(
        hist_access_type="search", raw=True, output=True, n=1, pattern="*", session=0
    )
    reply = get_reply(KC, msg_id, TIMEOUT)
    validate_message(reply, "history_reply", msg_id)
    content = reply["content"]
    assert len(content["history"]) == 1


# IOPub channel


def test_stream():
    flush_channels()

    msg_id, reply = execute("print('hi')")

    stdout = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(stdout, "stream", msg_id)
    content = stdout["content"]
    assert content["text"] == "hi\n"


def test_display_data():
    flush_channels()

    msg_id, reply = execute("from IPython.display import display; display(1)")

    display = KC.get_iopub_msg(timeout=TIMEOUT)
    validate_message(display, "display_data", parent=msg_id)
    data = display["content"]["data"]
    assert data["text/plain"] == "1"
