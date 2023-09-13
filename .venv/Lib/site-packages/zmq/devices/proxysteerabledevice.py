"""Classes for running a steerable ZMQ proxy"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy


class ProxySteerableBase:
    """Base class for overriding methods."""

    def __init__(self, in_type, out_type, mon_type=zmq.PUB, ctrl_type=None):
        super().__init__(in_type=in_type, out_type=out_type, mon_type=mon_type)
        self.ctrl_type = ctrl_type
        self._ctrl_binds = []
        self._ctrl_connects = []
        self._ctrl_sockopts = []

    def bind_ctrl(self, addr):
        """Enqueue ZMQ address for binding on ctrl_socket.

        See zmq.Socket.bind for details.
        """
        self._ctrl_binds.append(addr)

    def bind_ctrl_to_random_port(self, addr, *args, **kwargs):
        """Enqueue a random port on the given interface for binding on
        ctrl_socket.

        See zmq.Socket.bind_to_random_port for details.
        """
        port = self._reserve_random_port(addr, *args, **kwargs)

        self.bind_ctrl('%s:%i' % (addr, port))

        return port

    def connect_ctrl(self, addr):
        """Enqueue ZMQ address for connecting on ctrl_socket.

        See zmq.Socket.connect for details.
        """
        self._ctrl_connects.append(addr)

    def setsockopt_ctrl(self, opt, value):
        """Enqueue setsockopt(opt, value) for ctrl_socket

        See zmq.Socket.setsockopt for details.
        """
        self._ctrl_sockopts.append((opt, value))

    def _setup_sockets(self):
        ins, outs, mons = super()._setup_sockets()
        ctx = self._context
        ctrls = ctx.socket(self.ctrl_type)
        self._sockets.append(ctrls)

        for opt, value in self._ctrl_sockopts:
            ctrls.setsockopt(opt, value)

        for iface in self._ctrl_binds:
            ctrls.bind(iface)

        for iface in self._ctrl_connects:
            ctrls.connect(iface)

        return ins, outs, mons, ctrls

    def run_device(self):
        ins, outs, mons, ctrls = self._setup_sockets()
        zmq.proxy_steerable(ins, outs, mons, ctrls)


class ProxySteerable(ProxySteerableBase, Proxy):
    """Class for running a steerable proxy in the background.

    See zmq.devices.Proxy for most of the spec.  If the control socket is not
    NULL, the proxy supports control flow, provided by the socket.

    If PAUSE is received on this socket, the proxy suspends its activities. If
    RESUME is received, it goes on. If TERMINATE is received, it terminates
    smoothly.  If the control socket is NULL, the proxy behave exactly as if
    zmq.devices.Proxy had been used.

    This subclass adds a <method>_ctrl version of each <method>_{in|out}
    method, for configuring the control socket.

    .. versionadded:: libzmq-4.1
    .. versionadded:: 18.0
    """


class ThreadProxySteerable(ProxySteerableBase, ThreadProxy):
    """ProxySteerable in a Thread. See ProxySteerable for details."""


class ProcessProxySteerable(ProxySteerableBase, ProcessProxy):
    """ProxySteerable in a Process. See ProxySteerable for details."""


__all__ = [
    'ProxySteerable',
    'ThreadProxySteerable',
    'ProcessProxySteerable',
]
