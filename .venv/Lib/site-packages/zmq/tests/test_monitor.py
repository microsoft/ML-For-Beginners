# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import zmq
import zmq.asyncio
from zmq.tests import require_zmq_4
from zmq.utils.monitor import recv_monitor_message

pytestmark = require_zmq_4
import pytest


@pytest.fixture(params=["zmq", "asyncio"])
def Context(request, event_loop):
    if request.param == "asyncio":
        return zmq.asyncio.Context
    else:
        return zmq.Context


async def test_monitor(context, socket):
    """Test monitoring interface for sockets."""
    s_rep = socket(zmq.REP)
    s_req = socket(zmq.REQ)
    s_req.bind("tcp://127.0.0.1:6666")
    # try monitoring the REP socket
    s_rep.monitor(
        "inproc://monitor.rep",
        zmq.EVENT_CONNECT_DELAYED | zmq.EVENT_CONNECTED | zmq.EVENT_MONITOR_STOPPED,
    )
    # create listening socket for monitor
    s_event = socket(zmq.PAIR)
    s_event.connect("inproc://monitor.rep")
    s_event.linger = 0
    # test receive event for connect event
    s_rep.connect("tcp://127.0.0.1:6666")
    m = recv_monitor_message(s_event)
    if isinstance(context, zmq.asyncio.Context):
        m = await m
    if m['event'] == zmq.EVENT_CONNECT_DELAYED:
        assert m['endpoint'] == b"tcp://127.0.0.1:6666"
        # test receive event for connected event
        m = recv_monitor_message(s_event)
        if isinstance(context, zmq.asyncio.Context):
            m = await m
    assert m['event'] == zmq.EVENT_CONNECTED
    assert m['endpoint'] == b"tcp://127.0.0.1:6666"

    # test monitor can be disabled.
    s_rep.disable_monitor()
    m = recv_monitor_message(s_event)
    if isinstance(context, zmq.asyncio.Context):
        m = await m
    assert m['event'] == zmq.EVENT_MONITOR_STOPPED


async def test_monitor_repeat(context, socket, sockets):
    s = socket(zmq.PULL)
    m = s.get_monitor_socket()
    sockets.append(m)
    m2 = s.get_monitor_socket()
    assert m is m2
    s.disable_monitor()
    evt = recv_monitor_message(m)
    if isinstance(context, zmq.asyncio.Context):
        evt = await evt
    assert evt['event'] == zmq.EVENT_MONITOR_STOPPED
    m.close()
    s.close()


async def test_monitor_connected(context, socket, sockets):
    """Test connected monitoring socket."""
    s_rep = socket(zmq.REP)
    s_req = socket(zmq.REQ)
    s_req.bind("tcp://127.0.0.1:6667")
    # try monitoring the REP socket
    # create listening socket for monitor
    s_event = s_rep.get_monitor_socket()
    s_event.linger = 0
    sockets.append(s_event)
    # test receive event for connect event
    s_rep.connect("tcp://127.0.0.1:6667")
    m = recv_monitor_message(s_event)
    if isinstance(context, zmq.asyncio.Context):
        m = await m
    if m['event'] == zmq.EVENT_CONNECT_DELAYED:
        assert m['endpoint'] == b"tcp://127.0.0.1:6667"
        # test receive event for connected event
        m = recv_monitor_message(s_event)
        if isinstance(context, zmq.asyncio.Context):
            m = await m
    assert m['event'] == zmq.EVENT_CONNECTED
    assert m['endpoint'] == b"tcp://127.0.0.1:6667"
