"""zmq device functions"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from ._cffi import ffi
from ._cffi import lib as C
from .socket import Socket
from .utils import _retry_sys_call


def device(device_type, frontend, backend):
    return proxy(frontend, backend)


def proxy(frontend, backend, capture=None):
    if isinstance(capture, Socket):
        capture = capture._zmq_socket
    else:
        capture = ffi.NULL

    _retry_sys_call(C.zmq_proxy, frontend._zmq_socket, backend._zmq_socket, capture)


def proxy_steerable(frontend, backend, capture=None, control=None):
    """proxy_steerable(frontend, backend, capture, control)

    Start a zeromq proxy with control flow.

    .. versionadded:: libzmq-4.1
    .. versionadded:: 18.0

    Parameters
    ----------
    frontend : Socket
        The Socket instance for the incoming traffic.
    backend : Socket
        The Socket instance for the outbound traffic.
    capture : Socket (optional)
        The Socket instance for capturing traffic.
    control : Socket (optional)
        The Socket instance for control flow.
    """
    if isinstance(capture, Socket):
        capture = capture._zmq_socket
    else:
        capture = ffi.NULL

    if isinstance(control, Socket):
        control = control._zmq_socket
    else:
        control = ffi.NULL

    _retry_sys_call(
        C.zmq_proxy_steerable,
        frontend._zmq_socket,
        backend._zmq_socket,
        capture,
        control,
    )


__all__ = ['device', 'proxy', 'proxy_steerable']
