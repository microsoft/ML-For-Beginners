"""test embed_kernel"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from subprocess import PIPE, Popen

import pytest
from flaky import flaky
from jupyter_client.blocking.client import BlockingKernelClient
from jupyter_core import paths

from ipykernel.embed import IPKernelApp, embed_kernel  # type:ignore[attr-defined]

SETUP_TIMEOUT = 60
TIMEOUT = 15


if os.name == "nt":
    pytest.skip("skipping tests on windows", allow_module_level=True)


@contextmanager
def setup_kernel(cmd):
    """start an embedded kernel in a subprocess, and wait for it to be ready

    Returns
    -------
    kernel_manager: connected KernelManager instance
    """

    def connection_file_ready(connection_file):
        """Check if connection_file is a readable json file."""
        if not os.path.exists(connection_file):
            return False
        try:
            with open(connection_file) as f:
                json.load(f)
            return True
        except ValueError:
            return False

    kernel = Popen([sys.executable, "-c", cmd], stdout=PIPE, stderr=PIPE, encoding="utf-8")
    try:
        connection_file = os.path.join(
            paths.jupyter_runtime_dir(),
            "kernel-%i.json" % kernel.pid,
        )
        # wait for connection file to exist, timeout after 5s
        tic = time.time()
        while (
            not connection_file_ready(connection_file)
            and kernel.poll() is None
            and time.time() < tic + SETUP_TIMEOUT
        ):
            time.sleep(0.1)

        # Wait 100ms for the writing to finish
        time.sleep(0.1)

        if kernel.poll() is not None:
            o, e = kernel.communicate()
            raise OSError("Kernel failed to start:\n%s" % e)

        if not os.path.exists(connection_file):
            if kernel.poll() is None:
                kernel.terminate()
            raise OSError("Connection file %r never arrived" % connection_file)

        client = BlockingKernelClient(connection_file=connection_file)
        client.load_connection_file()
        client.start_channels()
        client.wait_for_ready()
        try:
            yield client
        finally:
            client.stop_channels()
    finally:
        kernel.terminate()
        kernel.wait()
        # Make sure all the fds get closed.
        for attr in ["stdout", "stderr", "stdin"]:
            fid = getattr(kernel, attr)
            if fid:
                fid.close()


@flaky(max_runs=3)
def test_embed_kernel_basic():
    """IPython.embed_kernel() is basically functional"""
    cmd = "\n".join(
        [
            "from IPython import embed_kernel",
            "def go():",
            "    a=5",
            '    b="hi there"',
            "    embed_kernel()",
            "go()",
            "",
        ]
    )

    with setup_kernel(cmd) as client:
        # oinfo a (int)
        client.inspect("a")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert content["found"]

        client.execute("c=a*2")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert content["status"] == "ok"

        # oinfo c (should be 10)
        client.inspect("c")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert content["found"]
        text = content["data"]["text/plain"]
        assert "10" in text


@flaky(max_runs=3)
def test_embed_kernel_namespace():
    """IPython.embed_kernel() inherits calling namespace"""
    cmd = "\n".join(
        [
            "from IPython import embed_kernel",
            "def go():",
            "    a=5",
            '    b="hi there"',
            "    embed_kernel()",
            "go()",
            "",
        ]
    )

    with setup_kernel(cmd) as client:
        # oinfo a (int)
        client.inspect("a")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert content["found"]
        text = content["data"]["text/plain"]
        assert "5" in text

        # oinfo b (str)
        client.inspect("b")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert content["found"]
        text = content["data"]["text/plain"]
        assert "hi there" in text

        # oinfo c (undefined)
        client.inspect("c")
        msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg["content"]
        assert not content["found"]


@flaky(max_runs=3)
def test_embed_kernel_reentrant():
    """IPython.embed_kernel() can be called multiple times"""
    cmd = "\n".join(
        [
            "from IPython import embed_kernel",
            "count = 0",
            "def go():",
            "    global count",
            "    embed_kernel()",
            "    count = count + 1",
            "",
            "while True:    go()",
            "",
        ]
    )

    with setup_kernel(cmd) as client:
        for i in range(5):
            client.inspect("count")
            msg = client.get_shell_msg(timeout=TIMEOUT)
            content = msg["content"]
            assert content["found"]
            text = content["data"]["text/plain"]
            assert str(i) in text

            # exit from embed_kernel
            client.execute("get_ipython().exit_now = True")
            msg = client.get_shell_msg(timeout=TIMEOUT)
            time.sleep(0.2)


def test_embed_kernel_func():
    from types import ModuleType

    module = ModuleType("test")

    def trigger_stop():
        time.sleep(1)
        app = IPKernelApp.instance()
        app.io_loop.add_callback(app.io_loop.stop)
        IPKernelApp.clear_instance()

    thread = threading.Thread(target=trigger_stop)
    thread.start()

    embed_kernel(module, outstream_class=None)
