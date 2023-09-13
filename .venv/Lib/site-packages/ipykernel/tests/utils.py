"""utilities for testing IPython kernels"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import atexit
import os
import sys
from contextlib import contextmanager
from queue import Empty
from subprocess import STDOUT
from tempfile import TemporaryDirectory
from time import time

from jupyter_client import manager
from jupyter_client.blocking.client import BlockingKernelClient

STARTUP_TIMEOUT = 60
TIMEOUT = 100

KM: manager.KernelManager = None  # type:ignore
KC: BlockingKernelClient = None  # type:ignore


def start_new_kernel(**kwargs):
    """start a new kernel, and return its Manager and Client

    Integrates with our output capturing for tests.
    """
    kwargs["stderr"] = STDOUT
    try:
        import nose

        kwargs["stdout"] = nose.iptest_stdstreams_fileno()
    except (ImportError, AttributeError):
        pass
    return manager.start_new_kernel(startup_timeout=STARTUP_TIMEOUT, **kwargs)


def flush_channels(kc=None):
    """flush any messages waiting on the queue"""
    from .test_message_spec import validate_message

    if kc is None:
        kc = KC
    for get_msg in (kc.get_shell_msg, kc.get_iopub_msg):
        while True:
            try:
                msg = get_msg(timeout=0.1)
            except Empty:
                break
            else:
                validate_message(msg)


def get_reply(kc, msg_id, timeout=TIMEOUT, channel="shell"):
    t0 = time()
    while True:
        get_msg = getattr(kc, f"get_{channel}_msg")
        reply = get_msg(timeout=timeout)
        if reply["parent_header"]["msg_id"] == msg_id:
            break
        # Allow debugging ignored replies
        print(f"Ignoring reply not to {msg_id}: {reply}")
        t1 = time()
        timeout -= t1 - t0
        t0 = t1
    return reply


def execute(code="", kc=None, **kwargs):
    """wrapper for doing common steps for validating an execution request"""
    from .test_message_spec import validate_message

    if kc is None:
        kc = KC
    msg_id = kc.execute(code=code, **kwargs)
    reply = get_reply(kc, msg_id, TIMEOUT)
    validate_message(reply, "execute_reply", msg_id)
    busy = kc.get_iopub_msg(timeout=TIMEOUT)
    validate_message(busy, "status", msg_id)
    assert busy["content"]["execution_state"] == "busy"

    if not kwargs.get("silent"):
        execute_input = kc.get_iopub_msg(timeout=TIMEOUT)
        validate_message(execute_input, "execute_input", msg_id)
        assert execute_input["content"]["code"] == code

    # show tracebacks if present for debugging
    if reply["content"].get("traceback"):
        print("\n".join(reply["content"]["traceback"]), file=sys.stderr)

    return msg_id, reply["content"]


def start_global_kernel():
    """start the global kernel (if it isn't running) and return its client"""
    global KM, KC
    if KM is None:
        KM, KC = start_new_kernel()
        atexit.register(stop_global_kernel)
    else:
        flush_channels(KC)
    return KC


@contextmanager
def kernel():
    """Context manager for the global kernel instance

    Should be used for most kernel tests

    Returns
    -------
    kernel_client: connected KernelClient instance
    """
    yield start_global_kernel()


def uses_kernel(test_f):
    """Decorator for tests that use the global kernel"""

    def wrapped_test():
        with kernel() as kc:
            test_f(kc)

    wrapped_test.__doc__ = test_f.__doc__
    wrapped_test.__name__ = test_f.__name__
    return wrapped_test


def stop_global_kernel():
    """Stop the global shared kernel instance, if it exists"""
    global KM, KC
    KC.stop_channels()
    KC = None  # type:ignore
    if KM is None:
        return
    KM.shutdown_kernel(now=True)
    KM = None  # type:ignore


def new_kernel(argv=None):
    """Context manager for a new kernel in a subprocess

    Should only be used for tests where the kernel must not be re-used.

    Returns
    -------
    kernel_client: connected KernelClient instance
    """
    kwargs = {"stderr": STDOUT}
    try:
        import nose

        kwargs["stdout"] = nose.iptest_stdstreams_fileno()
    except (ImportError, AttributeError):
        pass
    if argv is not None:
        kwargs["extra_arguments"] = argv
    return manager.run_kernel(**kwargs)


def assemble_output(get_msg):
    """assemble stdout/err from an execution"""
    stdout = ""
    stderr = ""
    while True:
        msg = get_msg(timeout=1)
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content["execution_state"] == "idle":
            # idle message signals end of output
            break
        elif msg["msg_type"] == "stream":
            if content["name"] == "stdout":
                stdout += content["text"]
            elif content["name"] == "stderr":
                stderr += content["text"]
            else:
                raise KeyError("bad stream: %r" % content["name"])
        else:
            # other output, ignored
            pass
    return stdout, stderr


def wait_for_idle(kc):
    while True:
        msg = kc.get_iopub_msg(timeout=1)
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content["execution_state"] == "idle":
            break


class TemporaryWorkingDirectory(TemporaryDirectory):
    """
    Creates a temporary directory and sets the cwd to that directory.
    Automatically reverts to previous cwd upon cleanup.
    Usage example:

        with TemporaryWorkingDirectory() as tmpdir:
            ...
    """

    def __enter__(self):
        self.old_wd = os.getcwd()
        os.chdir(self.name)
        return super().__enter__()

    def __exit__(self, exc, value, tb):
        os.chdir(self.old_wd)
        return super().__exit__(exc, value, tb)
