"""Test IO capturing functionality"""

import io
import os
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from unittest import mock

import pytest
import zmq
from jupyter_client.session import Session

from ipykernel.iostream import MASTER, BackgroundSocket, IOPubThread, OutStream


@pytest.fixture
def ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.destroy()


@pytest.fixture
def iopub_thread(ctx):
    with ctx.socket(zmq.PUB) as pub:
        thread = IOPubThread(pub)
        thread.start()

        yield thread
        thread.stop()
        thread.close()


def test_io_api(iopub_thread):
    """Test that wrapped stdout has the same API as a normal TextIO object"""
    session = Session()
    stream = OutStream(session, iopub_thread, "stdout")

    assert stream.errors is None
    assert not stream.isatty()
    with pytest.raises(io.UnsupportedOperation):
        stream.detach()
    with pytest.raises(io.UnsupportedOperation):
        next(stream)
    with pytest.raises(io.UnsupportedOperation):
        stream.read()
    with pytest.raises(io.UnsupportedOperation):
        stream.readline()
    with pytest.raises(io.UnsupportedOperation):
        stream.seek(0)
    with pytest.raises(io.UnsupportedOperation):
        stream.tell()
    with pytest.raises(TypeError):
        stream.write(b"")  # type:ignore


def test_io_isatty(iopub_thread):
    session = Session()
    stream = OutStream(session, iopub_thread, "stdout", isatty=True)
    assert stream.isatty()


def test_io_thread(iopub_thread):
    thread = iopub_thread
    thread._setup_pipe_in()
    msg = [thread._pipe_uuid, b"a"]
    thread._handle_pipe_msg(msg)
    ctx1, pipe = thread._setup_pipe_out()
    pipe.close()
    thread._pipe_in.close()
    thread._check_mp_mode = lambda: MASTER
    thread._really_send([b"hi"])
    ctx1.destroy()
    thread.close()
    thread.close()
    thread._really_send(None)


def test_background_socket(iopub_thread):
    sock = BackgroundSocket(iopub_thread)
    assert sock.__class__ == BackgroundSocket
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sock.linger = 101
        assert iopub_thread.socket.linger == 101
    assert sock.io_thread == iopub_thread
    sock.send(b"hi")


def test_outstream(iopub_thread):
    session = Session()
    pub = iopub_thread.socket
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stream = OutStream(session, pub, "stdout")
        stream.close()
        stream = OutStream(session, iopub_thread, "stdout", pipe=object())
        stream.close()

        stream = OutStream(session, iopub_thread, "stdout", watchfd=False)
        stream.close()

    stream = OutStream(session, iopub_thread, "stdout", isatty=True, echo=io.StringIO())

    with stream:
        with pytest.raises(io.UnsupportedOperation):
            stream.fileno()
        stream._watch_pipe_fd()
        stream.flush()
        stream.write("hi")
        stream.writelines(["ab", "cd"])
        assert stream.writable()


async def test_event_pipe_gc(iopub_thread):
    session = Session(key=b'abc')
    stream = OutStream(
        session,
        iopub_thread,
        "stdout",
        isatty=True,
        watchfd=False,
    )
    save_stdout = sys.stdout
    assert iopub_thread._event_pipes == {}
    with stream, mock.patch.object(sys, "stdout", stream), ThreadPoolExecutor(1) as pool:
        pool.submit(print, "x").result()
        pool_thread = pool.submit(threading.current_thread).result()
        assert list(iopub_thread._event_pipes) == [pool_thread]

    # run gc once in the iopub thread
    f: Future = Future()

    async def run_gc():
        try:
            await iopub_thread._event_pipe_gc()
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(None)

    iopub_thread.io_loop.add_callback(run_gc)
    # wait for call to finish in iopub thread
    f.result()
    assert iopub_thread._event_pipes == {}


def subprocess_test_echo_watch():
    # handshake Pub subscription
    session = Session(key=b'abc')

    # use PUSH socket to avoid subscription issues
    with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as pub:
        pub.connect(os.environ["IOPUB_URL"])
        iopub_thread = IOPubThread(pub)
        iopub_thread.start()
        stdout_fd = sys.stdout.fileno()
        sys.stdout.flush()
        stream = OutStream(
            session,
            iopub_thread,
            "stdout",
            isatty=True,
            echo=sys.stdout,
            watchfd="force",
        )
        save_stdout = sys.stdout
        with stream, mock.patch.object(sys, "stdout", stream):
            # write to low-level FD
            os.write(stdout_fd, b"fd\n")
            # print (writes to stream)
            print("print\n", end="")
            sys.stdout.flush()
            # write to unwrapped __stdout__ (should also go to original FD)
            sys.__stdout__.write("__stdout__\n")
            sys.__stdout__.flush()
            # write to original sys.stdout (should be the same as __stdout__)
            save_stdout.write("stdout\n")
            save_stdout.flush()
            # is there another way to flush on the FD?
            fd_file = os.fdopen(stdout_fd, "w")
            fd_file.flush()
            # we don't have a sync flush on _reading_ from the watched pipe
            time.sleep(1)
            stream.flush()
        iopub_thread.stop()
        iopub_thread.close()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows")
def test_echo_watch(ctx):
    """Test echo on underlying FD while capturing the same FD

    Test runs in a subprocess to avoid messing with pytest output capturing.
    """
    s = ctx.socket(zmq.PULL)
    port = s.bind_to_random_port("tcp://127.0.0.1")
    url = f"tcp://127.0.0.1:{port}"
    session = Session(key=b'abc')
    messages = []
    stdout_chunks = []
    with s:
        env = dict(os.environ)
        env["IOPUB_URL"] = url
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTEST_CURRENT_TEST", None)
        p = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {__name__}; {__name__}.subprocess_test_echo_watch()",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(f"{p.stdout=}")
        print(f"{p.stderr}=", file=sys.stderr)
        assert p.returncode == 0
        while s.poll(timeout=100):
            ident, msg = session.recv(s)
            assert msg is not None  # for type narrowing
            if msg["header"]["msg_type"] == "stream" and msg["content"]["name"] == "stdout":
                stdout_chunks.append(msg["content"]["text"])

    # check outputs
    # use sets of lines to ignore ordering issues with
    # async flush and watchfd thread

    # Check the stream output forwarded over zmq
    zmq_stdout = "".join(stdout_chunks)
    assert set(zmq_stdout.strip().splitlines()) == {
        "fd",
        "print",
        "stdout",
        "__stdout__",
    }

    # Check what was written to the process stdout (kernel terminal)
    # just check that each output source went to the terminal
    assert set(p.stdout.strip().splitlines()) == {
        "fd",
        "print",
        "stdout",
        "__stdout__",
    }
