"""Proxy classes and functions."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice


class ProxyBase:
    """Base class for overriding methods."""

    def __init__(self, in_type, out_type, mon_type=zmq.PUB):
        Device.__init__(self, in_type=in_type, out_type=out_type)
        self.mon_type = mon_type
        self._mon_binds = []
        self._mon_connects = []
        self._mon_sockopts = []

    def bind_mon(self, addr):
        """Enqueue ZMQ address for binding on mon_socket.

        See zmq.Socket.bind for details.
        """
        self._mon_binds.append(addr)

    def bind_mon_to_random_port(self, addr, *args, **kwargs):
        """Enqueue a random port on the given interface for binding on
        mon_socket.

        See zmq.Socket.bind_to_random_port for details.

        .. versionadded:: 18.0
        """
        port = self._reserve_random_port(addr, *args, **kwargs)

        self.bind_mon('%s:%i' % (addr, port))

        return port

    def connect_mon(self, addr):
        """Enqueue ZMQ address for connecting on mon_socket.

        See zmq.Socket.connect for details.
        """
        self._mon_connects.append(addr)

    def setsockopt_mon(self, opt, value):
        """Enqueue setsockopt(opt, value) for mon_socket

        See zmq.Socket.setsockopt for details.
        """
        self._mon_sockopts.append((opt, value))

    def _setup_sockets(self):
        ins, outs = Device._setup_sockets(self)
        ctx = self._context
        mons = ctx.socket(self.mon_type)
        self._sockets.append(mons)

        # set sockopts (must be done first, in case of zmq.IDENTITY)
        for opt, value in self._mon_sockopts:
            mons.setsockopt(opt, value)

        for iface in self._mon_binds:
            mons.bind(iface)

        for iface in self._mon_connects:
            mons.connect(iface)

        return ins, outs, mons

    def run_device(self):
        ins, outs, mons = self._setup_sockets()
        zmq.proxy(ins, outs, mons)


class Proxy(ProxyBase, Device):
    """Threadsafe Proxy object.

    See zmq.devices.Device for most of the spec. This subclass adds a
    <method>_mon version of each <method>_{in|out} method, for configuring the
    monitor socket.

    A Proxy is a 3-socket ZMQ Device that functions just like a
    QUEUE, except each message is also sent out on the monitor socket.

    A PUB socket is the most logical choice for the mon_socket, but it is not required.
    """


class ThreadProxy(ProxyBase, ThreadDevice):
    """Proxy in a Thread. See Proxy for more."""


class ProcessProxy(ProxyBase, ProcessDevice):
    """Proxy in a Process. See Proxy for more."""


__all__ = [
    'Proxy',
    'ThreadProxy',
    'ProcessProxy',
]
