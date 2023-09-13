# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import pytest

import zmq
import zmq.constants


def test_constants():
    assert zmq.POLLIN is zmq.PollEvent.POLLIN
    assert zmq.PUSH is zmq.SocketType.PUSH
    assert zmq.constants.SUBSCRIBE is zmq.SocketOption.SUBSCRIBE
    assert (
        zmq.RECONNECT_STOP_AFTER_DISCONNECT
        is zmq.constants.ReconnectStop.AFTER_DISCONNECT
    )


def test_socket_options():
    assert zmq.IDENTITY is zmq.SocketOption.ROUTING_ID
    assert zmq.IDENTITY._opt_type is zmq.constants._OptType.bytes
    assert zmq.AFFINITY._opt_type is zmq.constants._OptType.int64
    assert zmq.CURVE_SERVER._opt_type is zmq.constants._OptType.int
    assert zmq.FD._opt_type is zmq.constants._OptType.fd


@pytest.mark.parametrize("event_name", list(zmq.Event.__members__))
def test_event_reprs(event_name):
    event = getattr(zmq.Event, event_name)
    assert event_name in repr(event)
