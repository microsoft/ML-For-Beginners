"""zmq poll function"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

try:
    from time import monotonic
except ImportError:
    from time import clock as monotonic

import warnings

from zmq.error import InterruptedSystemCall, _check_rc

from ._cffi import ffi
from ._cffi import lib as C


def _make_zmq_pollitem(socket, flags):
    zmq_socket = socket._zmq_socket
    zmq_pollitem = ffi.new('zmq_pollitem_t*')
    zmq_pollitem.socket = zmq_socket
    zmq_pollitem.fd = 0
    zmq_pollitem.events = flags
    zmq_pollitem.revents = 0
    return zmq_pollitem[0]


def _make_zmq_pollitem_fromfd(socket_fd, flags):
    zmq_pollitem = ffi.new('zmq_pollitem_t*')
    zmq_pollitem.socket = ffi.NULL
    zmq_pollitem.fd = socket_fd
    zmq_pollitem.events = flags
    zmq_pollitem.revents = 0
    return zmq_pollitem[0]


def zmq_poll(sockets, timeout):
    cffi_pollitem_list = []
    low_level_to_socket_obj = {}
    from zmq import Socket

    for item in sockets:
        if isinstance(item[0], Socket):
            low_level_to_socket_obj[item[0]._zmq_socket] = item
            cffi_pollitem_list.append(_make_zmq_pollitem(item[0], item[1]))
        else:
            if not isinstance(item[0], int):
                # not an FD, get it from fileno()
                item = (item[0].fileno(), item[1])
            low_level_to_socket_obj[item[0]] = item
            cffi_pollitem_list.append(_make_zmq_pollitem_fromfd(item[0], item[1]))
    items = ffi.new('zmq_pollitem_t[]', cffi_pollitem_list)
    list_length = ffi.cast('int', len(cffi_pollitem_list))
    while True:
        c_timeout = ffi.cast('long', timeout)
        start = monotonic()
        rc = C.zmq_poll(items, list_length, c_timeout)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            if timeout > 0:
                ms_passed = int(1000 * (monotonic() - start))
                if ms_passed < 0:
                    # don't allow negative ms_passed,
                    # which can happen on old Python versions without time.monotonic.
                    warnings.warn(
                        "Negative elapsed time for interrupted poll: %s."
                        "  Did the clock change?" % ms_passed,
                        RuntimeWarning,
                    )
                    ms_passed = 0
                timeout = max(0, timeout - ms_passed)
            continue
        else:
            break
    result = []
    for item in items:
        if item.revents > 0:
            if item.socket != ffi.NULL:
                result.append(
                    (
                        low_level_to_socket_obj[item.socket][0],
                        item.revents,
                    )
                )
            else:
                result.append((item.fd, item.revents))
    return result


__all__ = ['zmq_poll']
