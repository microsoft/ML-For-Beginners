"""test the IPython Kernel"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import ast
import os.path
import platform
import signal
import subprocess
import sys
import time
from subprocess import Popen
from tempfile import TemporaryDirectory

import IPython
import psutil
import pytest
from flaky import flaky
from IPython.paths import locate_profile

from .utils import (
    TIMEOUT,
    assemble_output,
    execute,
    flush_channels,
    get_reply,
    kernel,
    new_kernel,
    wait_for_idle,
)


def _check_master(kc, expected=True, stream="stdout"):
    execute(kc=kc, code="import sys")
    flush_channels(kc)
    msg_id, content = execute(kc=kc, code="print(sys.%s._is_master_process())" % stream)
    stdout, stderr = assemble_output(kc.get_iopub_msg)
    assert stdout.strip() == repr(expected)


def _check_status(content):
    """If status=error, show the traceback"""
    if content["status"] == "error":
        assert False, "".join(["\n"] + content["traceback"])


# printing tests


def test_simple_print():
    """simple print statement in kernel"""
    with kernel() as kc:
        msg_id, content = execute(kc=kc, code="print('hi')")
        stdout, stderr = assemble_output(kc.get_iopub_msg)
        assert stdout == "hi\n"
        assert stderr == ""
        _check_master(kc, expected=True)


@pytest.mark.skip(reason="Currently don't capture during test as pytest does its own capturing")
def test_capture_fd():
    """simple print statement in kernel"""
    with kernel() as kc:
        iopub = kc.iopub_channel
        msg_id, content = execute(kc=kc, code="import os; os.system('echo capsys')")
        stdout, stderr = assemble_output(iopub)
        assert stdout == "capsys\n"
        assert stderr == ""
        _check_master(kc, expected=True)


@pytest.mark.skip(reason="Currently don't capture during test as pytest does its own capturing")
def test_subprocess_peek_at_stream_fileno():
    with kernel() as kc:
        iopub = kc.iopub_channel
        msg_id, content = execute(
            kc=kc,
            code="import subprocess, sys; subprocess.run(['python', '-c', 'import os; os.system(\"echo CAP1\"); print(\"CAP2\")'], stderr=sys.stderr)",
        )
        stdout, stderr = assemble_output(iopub)
        assert stdout == "CAP1\nCAP2\n"
        assert stderr == ""
        _check_master(kc, expected=True)


def test_sys_path():
    """test that sys.path doesn't get messed up by default"""
    with kernel() as kc:
        msg_id, content = execute(kc=kc, code="import sys; print(repr(sys.path))")
        stdout, stderr = assemble_output(kc.get_iopub_msg)
    # for error-output on failure
    sys.stderr.write(stderr)

    sys_path = ast.literal_eval(stdout.strip())
    assert "" in sys_path


def test_sys_path_profile_dir():
    """test that sys.path doesn't get messed up when `--profile-dir` is specified"""

    with new_kernel(["--profile-dir", locate_profile("default")]) as kc:
        msg_id, content = execute(kc=kc, code="import sys; print(repr(sys.path))")
        stdout, stderr = assemble_output(kc.get_iopub_msg)
    # for error-output on failure
    sys.stderr.write(stderr)

    sys_path = ast.literal_eval(stdout.strip())
    assert "" in sys_path


@flaky(max_runs=3)
@pytest.mark.skipif(
    sys.platform == "win32" or (sys.platform == "darwin" and sys.version_info >= (3, 8)),
    reason="subprocess prints fail on Windows and MacOS Python 3.8+",
)
def test_subprocess_print():
    """printing from forked mp.Process"""
    with new_kernel() as kc:
        _check_master(kc, expected=True)
        flush_channels(kc)
        np = 5
        code = "\n".join(
            [
                "import time",
                "import multiprocessing as mp",
                "pool = [mp.Process(target=print, args=('hello', i,)) for i in range(%i)]" % np,
                "for p in pool: p.start()",
                "for p in pool: p.join()",
                "time.sleep(0.5),",
            ]
        )

        msg_id, content = execute(kc=kc, code=code)
        stdout, stderr = assemble_output(kc.get_iopub_msg)
        assert stdout.count("hello") == np, stdout
        for n in range(np):
            assert stdout.count(str(n)) == 1, stdout
        assert stderr == ""
        _check_master(kc, expected=True)
        _check_master(kc, expected=True, stream="stderr")


@flaky(max_runs=3)
def test_subprocess_noprint():
    """mp.Process without print doesn't trigger iostream mp_mode"""
    with kernel() as kc:
        np = 5
        code = "\n".join(
            [
                "import multiprocessing as mp",
                "pool = [mp.Process(target=range, args=(i,)) for i in range(%i)]" % np,
                "for p in pool: p.start()",
                "for p in pool: p.join()",
            ]
        )

        msg_id, content = execute(kc=kc, code=code)
        stdout, stderr = assemble_output(kc.get_iopub_msg)
        assert stdout == ""
        assert stderr == ""

        _check_master(kc, expected=True)
        _check_master(kc, expected=True, stream="stderr")


@flaky(max_runs=3)
@pytest.mark.skipif(
    sys.platform == "win32" or (sys.platform == "darwin" and sys.version_info >= (3, 8)),
    reason="subprocess prints fail on Windows and MacOS Python 3.8+",
)
def test_subprocess_error():
    """error in mp.Process doesn't crash"""
    with new_kernel() as kc:
        code = "\n".join(
            [
                "import multiprocessing as mp",
                "p = mp.Process(target=int, args=('hi',))",
                "p.start()",
                "p.join()",
            ]
        )

        msg_id, content = execute(kc=kc, code=code)
        stdout, stderr = assemble_output(kc.get_iopub_msg)
        assert stdout == ""
        assert "ValueError" in stderr

        _check_master(kc, expected=True)
        _check_master(kc, expected=True, stream="stderr")


# raw_input tests


def test_raw_input():
    """test input"""
    with kernel() as kc:
        iopub = kc.iopub_channel

        input_f = "input"
        theprompt = "prompt> "
        code = f'print({input_f}("{theprompt}"))'
        msg_id = kc.execute(code, allow_stdin=True)
        msg = kc.get_stdin_msg(timeout=TIMEOUT)
        assert msg["header"]["msg_type"] == "input_request"
        content = msg["content"]
        assert content["prompt"] == theprompt
        text = "some text"
        kc.input(text)
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "ok"
        stdout, stderr = assemble_output(kc.get_iopub_msg)
        assert stdout == text + "\n"


def test_save_history():
    # Saving history from the kernel with %hist -f was failing because of
    # unicode problems on Python 2.
    with kernel() as kc, TemporaryDirectory() as td:
        file = os.path.join(td, "hist.out")
        execute("a=1", kc=kc)
        wait_for_idle(kc)
        execute('b="abcþ"', kc=kc)
        wait_for_idle(kc)
        _, reply = execute("%hist -f " + file, kc=kc)
        assert reply["status"] == "ok"
        with open(file, encoding="utf-8") as f:
            content = f.read()
        assert "a=1" in content
        assert 'b="abcþ"' in content


def test_smoke_faulthandler():
    faulthadler = pytest.importorskip("faulthandler", reason="this test needs faulthandler")
    with kernel() as kc:
        # Note: faulthandler.register is not available on windows.
        code = "\n".join(
            [
                "import sys",
                "import faulthandler",
                "import signal",
                "faulthandler.enable()",
                'if not sys.platform.startswith("win32"):',
                "    faulthandler.register(signal.SIGTERM)",
            ]
        )
        _, reply = execute(code, kc=kc)
        assert reply["status"] == "ok", reply.get("traceback", "")


def test_help_output():
    """ipython kernel --help-all works"""
    cmd = [sys.executable, "-m", "IPython", "kernel", "--help-all"]
    proc = subprocess.run(cmd, timeout=30, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert b"Traceback" not in proc.stderr
    assert b"Options" in proc.stdout
    assert b"Class" in proc.stdout


def test_is_complete():
    with kernel() as kc:
        # There are more test cases for this in core - here we just check
        # that the kernel exposes the interface correctly.
        kc.is_complete("2+2")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "complete"

        # SyntaxError
        kc.is_complete("raise = 2")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "invalid"

        kc.is_complete("a = [1,\n2,")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "incomplete"
        assert reply["content"]["indent"] == ""

        # Cell magic ends on two blank lines for console UIs
        kc.is_complete("%%timeit\na\n\n")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "complete"


@pytest.mark.skipif(sys.platform != "win32", reason="only run on Windows")
def test_complete():
    with kernel() as kc:
        execute("a = 1", kc=kc)
        wait_for_idle(kc)
        cell = "import IPython\nb = a."
        kc.complete(cell)
        reply = kc.get_shell_msg(timeout=TIMEOUT)

    c = reply["content"]
    assert c["status"] == "ok"
    start = cell.find("a.")
    end = start + 2
    assert c["cursor_end"] == cell.find("a.") + 2
    assert c["cursor_start"] <= end

    # there are many right answers for cursor_start,
    # so verify application of the completion
    # rather than the value of cursor_start

    matches = c["matches"]
    assert matches
    for m in matches:
        completed = cell[: c["cursor_start"]] + m
        assert completed.startswith(cell)


def test_matplotlib_inline_on_import():
    pytest.importorskip("matplotlib", reason="this test requires matplotlib")
    with kernel() as kc:
        cell = "\n".join(
            ["import matplotlib, matplotlib.pyplot as plt", "backend = matplotlib.get_backend()"]
        )
        _, reply = execute(cell, user_expressions={"backend": "backend"}, kc=kc)
        _check_status(reply)
        backend_bundle = reply["user_expressions"]["backend"]
        _check_status(backend_bundle)
        assert "backend_inline" in backend_bundle["data"]["text/plain"]


def test_message_order():
    N = 100  # number of messages to test
    with kernel() as kc:
        _, reply = execute("a = 1", kc=kc)
        _check_status(reply)
        offset = reply["execution_count"] + 1
        cell = "a += 1\na"
        msg_ids = []
        # submit N executions as fast as we can
        for _ in range(N):
            msg_ids.append(kc.execute(cell))
        # check message-handling order
        for i, msg_id in enumerate(msg_ids, offset):
            reply = kc.get_shell_msg(timeout=TIMEOUT)
            _check_status(reply["content"])
            assert reply["content"]["execution_count"] == i
            assert reply["parent_header"]["msg_id"] == msg_id


@pytest.mark.skipif(
    sys.platform.startswith("linux") or sys.platform.startswith("darwin"),
    reason="test only on windows",
)
def test_unc_paths():
    with kernel() as kc, TemporaryDirectory() as td:
        drive_file_path = os.path.join(td, "unc.txt")
        with open(drive_file_path, "w+") as f:
            f.write("# UNC test")
        unc_root = "\\\\localhost\\C$"
        file_path = os.path.splitdrive(os.path.dirname(drive_file_path))[1]
        unc_file_path = os.path.join(unc_root, file_path[1:])

        kc.execute(f"cd {unc_file_path:s}")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "ok"
        out, err = assemble_output(kc.get_iopub_msg)
        assert unc_file_path in out

        flush_channels(kc)
        kc.execute(code="ls")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "ok"
        out, err = assemble_output(kc.get_iopub_msg)
        assert "unc.txt" in out

        kc.execute(code="cd")
        reply = kc.get_shell_msg(timeout=TIMEOUT)
        assert reply["content"]["status"] == "ok"


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="does not work on PyPy",
)
def test_shutdown():
    """Kernel exits after polite shutdown_request"""
    with new_kernel() as kc:
        km = kc.parent
        execute("a = 1", kc=kc)
        wait_for_idle(kc)
        kc.shutdown()
        for _ in range(300):  # 30s timeout
            if km.is_alive():
                time.sleep(0.1)
            else:
                break
        assert not km.is_alive()


def test_interrupt_during_input():
    """
    The kernel exits after being interrupted while waiting in input().

    input() appears to have issues other functions don't, and it needs to be
    interruptible in order for pdb to be interruptible.
    """
    with new_kernel() as kc:
        km = kc.parent
        msg_id = kc.execute("input()")
        time.sleep(1)  # Make sure it's actually waiting for input.
        km.interrupt_kernel()
        from .test_message_spec import validate_message

        # If we failed to interrupt interrupt, this will timeout:
        reply = get_reply(kc, msg_id, TIMEOUT)
        validate_message(reply, "execute_reply", msg_id)


@pytest.mark.skipif(os.name == "nt", reason="Message based interrupt not supported on Windows")
def test_interrupt_with_message():
    with new_kernel() as kc:
        km = kc.parent
        km.kernel_spec.interrupt_mode = "message"
        msg_id = kc.execute("input()")
        time.sleep(1)  # Make sure it's actually waiting for input.
        km.interrupt_kernel()
        from .test_message_spec import validate_message

        # If we failed to interrupt interrupt, this will timeout:
        reply = get_reply(kc, msg_id, TIMEOUT)
        validate_message(reply, "execute_reply", msg_id)


@pytest.mark.skipif(
    "__pypy__" in sys.builtin_module_names,
    reason="fails on pypy",
)
def test_interrupt_during_pdb_set_trace():
    """
    The kernel exits after being interrupted while waiting in pdb.set_trace().

    Merely testing input() isn't enough, pdb has its own issues that need
    to be handled in addition.

    This test will fail with versions of IPython < 7.14.0.
    """
    with new_kernel() as kc:
        km = kc.parent
        msg_id = kc.execute("import pdb; pdb.set_trace()")
        msg_id2 = kc.execute("3 + 4")
        time.sleep(1)  # Make sure it's actually waiting for input.
        km.interrupt_kernel()
        from .test_message_spec import validate_message

        # If we failed to interrupt interrupt, this will timeout:
        reply = get_reply(kc, msg_id, TIMEOUT)
        validate_message(reply, "execute_reply", msg_id)
        # If we failed to interrupt interrupt, this will timeout:
        reply = get_reply(kc, msg_id2, TIMEOUT)
        validate_message(reply, "execute_reply", msg_id2)


def test_control_thread_priority():
    N = 5
    with new_kernel() as kc:
        msg_id = kc.execute("pass")
        get_reply(kc, msg_id)

        sleep_msg_id = kc.execute("import asyncio; await asyncio.sleep(2)")

        # submit N shell messages
        shell_msg_ids = []
        for i in range(N):
            shell_msg_ids.append(kc.execute(f"i = {i}"))

        # ensure all shell messages have arrived at the kernel before any control messages
        time.sleep(0.5)
        # at this point, shell messages should be waiting in msg_queue,
        # rather than zmq while the kernel is still in the middle of processing
        # the first execution

        # now send N control messages
        control_msg_ids = []
        for _ in range(N):
            msg = kc.session.msg("kernel_info_request", {})
            kc.control_channel.send(msg)
            control_msg_ids.append(msg["header"]["msg_id"])

        # finally, collect the replies on both channels for comparison
        get_reply(kc, sleep_msg_id)
        shell_replies = []
        for msg_id in shell_msg_ids:
            shell_replies.append(get_reply(kc, msg_id))

        control_replies = []
        for msg_id in control_msg_ids:
            control_replies.append(get_reply(kc, msg_id, channel="control"))

    # verify that all control messages were handled before all shell messages
    shell_dates = [msg["header"]["date"] for msg in shell_replies]
    control_dates = [msg["header"]["date"] for msg in control_replies]
    # comparing first to last ought to be enough, since queues preserve order
    # use <= in case of very-fast handling and/or low resolution timers
    assert control_dates[-1] <= shell_dates[0]


def _child():
    print("in child", os.getpid())

    def _print_and_exit(sig, frame):
        print(f"Received signal {sig}")
        # take some time so retries are triggered
        time.sleep(0.5)
        sys.exit(-sig)

    signal.signal(signal.SIGTERM, _print_and_exit)
    time.sleep(30)


def _start_children():
    ip = IPython.get_ipython()  # type:ignore[attr-defined]
    ns = ip.user_ns

    cmd = [sys.executable, "-c", f"from {__name__} import _child; _child()"]
    child_pg = Popen(cmd, start_new_session=False)
    child_newpg = Popen(cmd, start_new_session=True)
    ns["pid"] = os.getpid()
    ns["child_pg"] = child_pg.pid
    ns["child_newpg"] = child_newpg.pid
    # give them time to start up and register signal handlers
    time.sleep(1)


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="does not work on PyPy",
)
@pytest.mark.skipif(
    sys.platform.lower() == "linux",
    reason="Stalls on linux",
)
def test_shutdown_subprocesses():
    """Kernel exits after polite shutdown_request"""
    with new_kernel() as kc:
        km = kc.parent
        msg_id, reply = execute(
            f"from {__name__} import _start_children\n_start_children()",
            kc=kc,
            user_expressions={
                "pid": "pid",
                "child_pg": "child_pg",
                "child_newpg": "child_newpg",
            },
        )
        print(reply)
        expressions = reply["user_expressions"]
        kernel_process = psutil.Process(int(expressions["pid"]["data"]["text/plain"]))
        child_pg = psutil.Process(int(expressions["child_pg"]["data"]["text/plain"]))
        child_newpg = psutil.Process(int(expressions["child_newpg"]["data"]["text/plain"]))
        wait_for_idle(kc)

        kc.shutdown()
        for _ in range(300):  # 30s timeout
            if km.is_alive():
                time.sleep(0.1)
            else:
                break
        assert not km.is_alive()
        assert not kernel_process.is_running()
        # child in the process group shut down
        assert not child_pg.is_running()
        # child outside the process group was not shut down (unix only)
        if os.name != "nt":
            assert child_newpg.is_running()
        try:
            child_newpg.terminate()
        except psutil.NoSuchProcess:
            pass
