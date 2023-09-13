"""Module holding utility and convenience functions for zmq event monitoring."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import struct
from typing import Awaitable, List, Union, overload

import zmq
import zmq.asyncio
from zmq._typing import TypedDict
from zmq.error import _check_version


class _MonitorMessage(TypedDict):
    event: int
    value: int
    endpoint: bytes


def parse_monitor_message(msg: List[bytes]) -> _MonitorMessage:
    """decode zmq_monitor event messages.

    Parameters
    ----------
    msg : list(bytes)
        zmq multipart message that has arrived on a monitor PAIR socket.

        First frame is::

            16 bit event id
            32 bit event value
            no padding

        Second frame is the endpoint as a bytestring

    Returns
    -------
    event : dict
        event description as dict with the keys `event`, `value`, and `endpoint`.
    """
    if len(msg) != 2 or len(msg[0]) != 6:
        raise RuntimeError("Invalid event message format: %s" % msg)
    event_id, value = struct.unpack("=hi", msg[0])
    event: _MonitorMessage = {
        'event': zmq.Event(event_id),
        'value': zmq.Event(value),
        'endpoint': msg[1],
    }
    return event


async def _parse_monitor_msg_async(
    awaitable_msg: Awaitable[List[bytes]],
) -> _MonitorMessage:
    """Like parse_monitor_msg, but awaitable

    Given awaitable message, return awaitable for the parsed monitor message.
    """

    msg = await awaitable_msg
    # 4.0-style event API
    return parse_monitor_message(msg)


@overload
def recv_monitor_message(
    socket: "zmq.asyncio.Socket",
    flags: int = 0,
) -> Awaitable[_MonitorMessage]:
    ...


@overload
def recv_monitor_message(
    socket: zmq.Socket[bytes],
    flags: int = 0,
) -> _MonitorMessage:
    ...


def recv_monitor_message(
    socket: zmq.Socket,
    flags: int = 0,
) -> Union[_MonitorMessage, Awaitable[_MonitorMessage]]:
    """Receive and decode the given raw message from the monitoring socket and return a dict.

    Requires libzmq ≥ 4.0

    The returned dict will have the following entries:
      event     : int, the event id as described in libzmq.zmq_socket_monitor
      value     : int, the event value associated with the event, see libzmq.zmq_socket_monitor
      endpoint  : string, the affected endpoint

    .. versionchanged:: 23.1
        Support for async sockets added.
        When called with a async socket,
        returns an awaitable for the monitor message.

    Parameters
    ----------
    socket : zmq PAIR socket
        The PAIR socket (created by other.get_monitor_socket()) on which to recv the message
    flags : bitfield (int)
        standard zmq recv flags

    Returns
    -------
    event : dict
        event description as dict with the keys `event`, `value`, and `endpoint`.
    """

    _check_version((4, 0), 'libzmq event API')
    # will always return a list
    msg = socket.recv_multipart(flags)

    # transparently handle asyncio socket,
    # returns a Future instead of a dict
    if isinstance(msg, Awaitable):
        return _parse_monitor_msg_async(msg)

    # 4.0-style event API
    return parse_monitor_message(msg)


__all__ = ['parse_monitor_message', 'recv_monitor_message']
