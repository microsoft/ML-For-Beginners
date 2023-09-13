# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

from __future__ import annotations

import functools
import typing

from debugpy import _version


# Expose debugpy.server API from subpackage, but do not actually import it unless
# and until a member is invoked - we don't want the server package loaded in the
# adapter, the tests, or setup.py.

# Docstrings for public API members must be formatted according to PEP 8 - no more
# than 72 characters per line! - and must be readable when retrieved via help().


Endpoint = typing.Tuple[str, int]


def _api(cancelable=False):
    def apply(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            from debugpy.server import api

            wrapped = getattr(api, f.__name__)
            return wrapped(*args, **kwargs)

        if cancelable:

            def cancel(*args, **kwargs):
                from debugpy.server import api

                wrapped = getattr(api, f.__name__)
                return wrapped.cancel(*args, **kwargs)

            wrapper.cancel = cancel

        return wrapper

    return apply


@_api()
def log_to(__path: str) -> None:
    """Generate detailed debugpy logs in the specified directory.

    The directory must already exist. Several log files are generated,
    one for every process involved in the debug session.
    """


@_api()
def configure(__properties: dict[str, typing.Any] | None = None, **kwargs) -> None:
    """Sets debug configuration properties that cannot be set in the
    "attach" request, because they must be applied as early as possible
    in the process being debugged.

    For example, a "launch" configuration with subprocess debugging
    disabled can be defined entirely in JSON::

        {
            "request": "launch",
            "subProcess": false,
            ...
        }

    But the same cannot be done with "attach", because "subProcess"
    must be known at the point debugpy starts tracing execution. Thus,
    it is not available in JSON, and must be omitted::

        {
            "request": "attach",
            ...
        }

    and set from within the debugged process instead::

        debugpy.configure(subProcess=False)
        debugpy.listen(...)

    Properties to set can be passed either as a single dict argument,
    or as separate keyword arguments::

        debugpy.configure({"subProcess": False})
    """


@_api()
def listen(
    __endpoint: Endpoint | int, *, in_process_debug_adapter: bool = False
) -> Endpoint:
    """Starts a debug adapter debugging this process, that listens for
    incoming socket connections from clients on the specified address.

    `__endpoint` must be either a (host, port) tuple as defined by the
    standard `socket` module for the `AF_INET` address family, or a port
    number. If only the port is specified, host is "127.0.0.1".

    `in_process_debug_adapter`: by default a separate python process is
    spawned and used to communicate with the client as the debug adapter.
    By setting the value of `in_process_debug_adapter` to True a new 
    python process is not spawned. Note: the con of setting 
    `in_process_debug_adapter` to True is that subprocesses won't be 
    automatically debugged.
        
    Returns the interface and the port on which the debug adapter is
    actually listening, in the same format as `__endpoint`. This may be
    different from address if port was 0 in the latter, in which case
    the adapter will pick some unused ephemeral port to listen on.

    This function does't wait for a client to connect to the debug
    adapter that it starts. Use `wait_for_client` to block execution
    until the client connects.
    """


@_api()
def connect(__endpoint: Endpoint | int, *, access_token: str | None = None) -> Endpoint:
    """Tells an existing debug adapter instance that is listening on the
    specified address to debug this process.

    `__endpoint` must be either a (host, port) tuple as defined by the
    standard `socket` module for the `AF_INET` address family, or a port
    number. If only the port is specified, host is "127.0.0.1".

    `access_token` must be the same value that was passed to the adapter
    via the `--server-access-token` command-line switch.

    This function does't wait for a client to connect to the debug
    adapter that it connects to. Use `wait_for_client` to block
    execution until the client connects.
    """


@_api(cancelable=True)
def wait_for_client() -> None:
    """If there is a client connected to the debug adapter that is
    debugging this process, returns immediately. Otherwise, blocks
    until a client connects to the adapter.

    While this function is waiting, it can be canceled by calling
    `wait_for_client.cancel()` from another thread.
    """


@_api()
def is_client_connected() -> bool:
    """True if a client is connected to the debug adapter that is
    debugging this process.
    """


@_api()
def breakpoint() -> None:
    """If a client is connected to the debug adapter that is debugging
    this process, pauses execution of all threads, and simulates a
    breakpoint being hit at the line following the call.

    It is also registered as the default handler for builtins.breakpoint().
    """


@_api()
def debug_this_thread() -> None:
    """Makes the debugger aware of the current thread.

    Must be called on any background thread that is started by means
    other than the usual Python APIs (i.e. the "threading" module),
    in order for breakpoints to work on that thread.
    """


@_api()
def trace_this_thread(__should_trace: bool):
    """Tells the debug adapter to enable or disable tracing on the
    current thread.

    When the thread is traced, the debug adapter can detect breakpoints
    being hit, but execution is slower, especially in functions that
    have any breakpoints set in them. Disabling tracing when breakpoints
    are not anticipated to be hit can improve performance. It can also
    be used to skip breakpoints on a particular thread.

    Tracing is automatically disabled for all threads when there is no
    client connected to the debug adapter.
    """


__version__: str = _version.get_versions()["version"]
