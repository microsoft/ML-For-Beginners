# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import codecs
import os
import pydevd
import socket
import sys
import threading

import debugpy
from debugpy import adapter
from debugpy.common import json, log, sockets
from _pydevd_bundle.pydevd_constants import get_global_debugger
from pydevd_file_utils import absolute_path
from debugpy.common.util import hide_debugpy_internals

_tls = threading.local()

# TODO: "gevent", if possible.
_config = {
    "qt": "none",
    "subProcess": True,
    "python": sys.executable,
    "pythonEnv": {},
}

_config_valid_values = {
    # If property is not listed here, any value is considered valid, so long as
    # its type matches that of the default value in _config.
    "qt": ["auto", "none", "pyside", "pyside2", "pyqt4", "pyqt5"],
}

# This must be a global to prevent it from being garbage collected and triggering
# https://bugs.python.org/issue37380.
_adapter_process = None


def _settrace(*args, **kwargs):
    log.debug("pydevd.settrace(*{0!r}, **{1!r})", args, kwargs)
    # The stdin in notification is not acted upon in debugpy, so, disable it.
    kwargs.setdefault("notify_stdin", False)
    try:
        return pydevd.settrace(*args, **kwargs)
    except Exception:
        raise
    else:
        _settrace.called = True


_settrace.called = False


def ensure_logging():
    """Starts logging to log.log_dir, if it hasn't already been done."""
    if ensure_logging.ensured:
        return
    ensure_logging.ensured = True
    log.to_file(prefix="debugpy.server")
    log.describe_environment("Initial environment:")
    if log.log_dir is not None:
        pydevd.log_to(log.log_dir + "/debugpy.pydevd.log")


ensure_logging.ensured = False


def log_to(path):
    if ensure_logging.ensured:
        raise RuntimeError("logging has already begun")

    log.debug("log_to{0!r}", (path,))
    if path is sys.stderr:
        log.stderr.levels |= set(log.LEVELS)
    else:
        log.log_dir = path


def configure(properties=None, **kwargs):
    if _settrace.called:
        raise RuntimeError("debug adapter is already running")

    ensure_logging()
    log.debug("configure{0!r}", (properties, kwargs))

    if properties is None:
        properties = kwargs
    else:
        properties = dict(properties)
        properties.update(kwargs)

    for k, v in properties.items():
        if k not in _config:
            raise ValueError("Unknown property {0!r}".format(k))
        expected_type = type(_config[k])
        if type(v) is not expected_type:
            raise ValueError("{0!r} must be a {1}".format(k, expected_type.__name__))
        valid_values = _config_valid_values.get(k)
        if (valid_values is not None) and (v not in valid_values):
            raise ValueError("{0!r} must be one of: {1!r}".format(k, valid_values))
        _config[k] = v


def _starts_debugging(func):
    def debug(address, **kwargs):
        if _settrace.called:
            raise RuntimeError("this process already has a debug adapter")

        try:
            _, port = address
        except Exception:
            port = address
            address = ("127.0.0.1", port)
        try:
            port.__index__()  # ensure it's int-like
        except Exception:
            raise ValueError("expected port or (host, port)")
        if not (0 <= port < 2 ** 16):
            raise ValueError("invalid port number")

        ensure_logging()
        log.debug("{0}({1!r}, **{2!r})", func.__name__, address, kwargs)
        log.info("Initial debug configuration: {0}", json.repr(_config))

        qt_mode = _config.get("qt", "none")
        if qt_mode != "none":
            pydevd.enable_qt_support(qt_mode)

        settrace_kwargs = {
            "suspend": False,
            "patch_multiprocessing": _config.get("subProcess", True),
        }

        if hide_debugpy_internals():
            debugpy_path = os.path.dirname(absolute_path(debugpy.__file__))
            settrace_kwargs["dont_trace_start_patterns"] = (debugpy_path,)
            settrace_kwargs["dont_trace_end_patterns"] = (str("debugpy_launcher.py"),)

        try:
            return func(address, settrace_kwargs, **kwargs)
        except Exception:
            log.reraise_exception("{0}() failed:", func.__name__, level="info")

    return debug


@_starts_debugging
def listen(address, settrace_kwargs, in_process_debug_adapter=False):
    # Errors below are logged with level="info", because the caller might be catching
    # and handling exceptions, and we don't want to spam their stderr unnecessarily.

    if in_process_debug_adapter:
        host, port = address
        log.info("Listening: pydevd without debugpy adapter: {0}:{1}", host, port)
        settrace_kwargs['patch_multiprocessing'] = False
        _settrace(
            host=host,
            port=port,
            wait_for_ready_to_run=False,
            block_until_connected=False,
            **settrace_kwargs
        )
        return

    import subprocess

    server_access_token = codecs.encode(os.urandom(32), "hex").decode("ascii")

    try:
        endpoints_listener = sockets.create_server("127.0.0.1", 0, timeout=10)
    except Exception as exc:
        log.swallow_exception("Can't listen for adapter endpoints:")
        raise RuntimeError("can't listen for adapter endpoints: " + str(exc))

    try:
        endpoints_host, endpoints_port = endpoints_listener.getsockname()
        log.info(
            "Waiting for adapter endpoints on {0}:{1}...",
            endpoints_host,
            endpoints_port,
        )

        host, port = address
        adapter_args = [
            _config.get("python", sys.executable),
            os.path.dirname(adapter.__file__),
            "--for-server",
            str(endpoints_port),
            "--host",
            host,
            "--port",
            str(port),
            "--server-access-token",
            server_access_token,
        ]
        if log.log_dir is not None:
            adapter_args += ["--log-dir", log.log_dir]
        log.info("debugpy.listen() spawning adapter: {0}", json.repr(adapter_args))

        # On Windows, detach the adapter from our console, if any, so that it doesn't
        # receive Ctrl+C from it, and doesn't keep it open once we exit.
        creationflags = 0
        if sys.platform == "win32":
            creationflags |= 0x08000000  # CREATE_NO_WINDOW
            creationflags |= 0x00000200  # CREATE_NEW_PROCESS_GROUP

        # On embedded applications, environment variables might not contain
        # Python environment settings.
        python_env = _config.get("pythonEnv")
        if not bool(python_env):
            python_env = None

        # Adapter will outlive this process, so we shouldn't wait for it. However, we
        # need to ensure that the Popen instance for it doesn't get garbage-collected
        # by holding a reference to it in a non-local variable, to avoid triggering
        # https://bugs.python.org/issue37380.
        try:
            global _adapter_process
            _adapter_process = subprocess.Popen(
                adapter_args, close_fds=True, creationflags=creationflags, env=python_env
            )
            if os.name == "posix":
                # It's going to fork again to daemonize, so we need to wait on it to
                # clean it up properly.
                _adapter_process.wait()
            else:
                # Suppress misleading warning about child process still being alive when
                # this process exits (https://bugs.python.org/issue38890).
                _adapter_process.returncode = 0
                pydevd.add_dont_terminate_child_pid(_adapter_process.pid)
        except Exception as exc:
            log.swallow_exception("Error spawning debug adapter:", level="info")
            raise RuntimeError("error spawning debug adapter: " + str(exc))

        try:
            sock, _ = endpoints_listener.accept()
            try:
                sock.settimeout(None)
                sock_io = sock.makefile("rb", 0)
                try:
                    endpoints = json.loads(sock_io.read().decode("utf-8"))
                finally:
                    sock_io.close()
            finally:
                sockets.close_socket(sock)
        except socket.timeout:
            log.swallow_exception(
                "Timed out waiting for adapter to connect:", level="info"
            )
            raise RuntimeError("timed out waiting for adapter to connect")
        except Exception as exc:
            log.swallow_exception("Error retrieving adapter endpoints:", level="info")
            raise RuntimeError("error retrieving adapter endpoints: " + str(exc))

    finally:
        endpoints_listener.close()

    log.info("Endpoints received from adapter: {0}", json.repr(endpoints))

    if "error" in endpoints:
        raise RuntimeError(str(endpoints["error"]))

    try:
        server_host = str(endpoints["server"]["host"])
        server_port = int(endpoints["server"]["port"])
        client_host = str(endpoints["client"]["host"])
        client_port = int(endpoints["client"]["port"])
    except Exception as exc:
        log.swallow_exception(
            "Error parsing adapter endpoints:\n{0}\n",
            json.repr(endpoints),
            level="info",
        )
        raise RuntimeError("error parsing adapter endpoints: " + str(exc))
    log.info(
        "Adapter is accepting incoming client connections on {0}:{1}",
        client_host,
        client_port,
    )

    _settrace(
        host=server_host,
        port=server_port,
        wait_for_ready_to_run=False,
        block_until_connected=True,
        access_token=server_access_token,
        **settrace_kwargs
    )
    log.info("pydevd is connected to adapter at {0}:{1}", server_host, server_port)
    return client_host, client_port


@_starts_debugging
def connect(address, settrace_kwargs, access_token=None):
    host, port = address
    _settrace(host=host, port=port, client_access_token=access_token, **settrace_kwargs)


class wait_for_client:
    def __call__(self):
        ensure_logging()
        log.debug("wait_for_client()")

        pydb = get_global_debugger()
        if pydb is None:
            raise RuntimeError("listen() or connect() must be called first")

        cancel_event = threading.Event()
        self.cancel = cancel_event.set
        pydevd._wait_for_attach(cancel=cancel_event)

    @staticmethod
    def cancel():
        raise RuntimeError("wait_for_client() must be called first")


wait_for_client = wait_for_client()


def is_client_connected():
    return pydevd._is_attached()


def breakpoint():
    ensure_logging()
    if not is_client_connected():
        log.info("breakpoint() ignored - debugger not attached")
        return
    log.debug("breakpoint()")

    # Get the first frame in the stack that's not an internal frame.
    pydb = get_global_debugger()
    stop_at_frame = sys._getframe().f_back
    while (
        stop_at_frame is not None
        and pydb.get_file_type(stop_at_frame) == pydb.PYDEV_FILE
    ):
        stop_at_frame = stop_at_frame.f_back

    _settrace(
        suspend=True,
        trace_only_current_thread=True,
        patch_multiprocessing=False,
        stop_at_frame=stop_at_frame,
    )
    stop_at_frame = None


def debug_this_thread():
    ensure_logging()
    log.debug("debug_this_thread()")

    _settrace(suspend=False)


def trace_this_thread(should_trace):
    ensure_logging()
    log.debug("trace_this_thread({0!r})", should_trace)

    pydb = get_global_debugger()
    if should_trace:
        pydb.enable_tracing()
    else:
        pydb.disable_tracing()
