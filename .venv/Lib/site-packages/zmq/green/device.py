# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import zmq
from zmq.green import Poller


def device(device_type, isocket, osocket):
    """Start a zeromq device (gevent-compatible).

    Unlike the true zmq.device, this does not release the GIL.

    Parameters
    ----------
    device_type : (QUEUE, FORWARDER, STREAMER)
        The type of device to start (ignored).
    isocket : Socket
        The Socket instance for the incoming traffic.
    osocket : Socket
        The Socket instance for the outbound traffic.
    """
    p = Poller()
    if osocket == -1:
        osocket = isocket
    p.register(isocket, zmq.POLLIN)
    p.register(osocket, zmq.POLLIN)

    while True:
        events = dict(p.poll())
        if isocket in events:
            osocket.send_multipart(isocket.recv_multipart())
        if osocket in events:
            isocket.send_multipart(osocket.recv_multipart())
