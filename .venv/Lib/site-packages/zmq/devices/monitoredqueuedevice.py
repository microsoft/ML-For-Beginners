"""MonitoredQueue classes and functions."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


from zmq import PUB
from zmq.devices.monitoredqueue import monitored_queue
from zmq.devices.proxydevice import ProcessProxy, Proxy, ProxyBase, ThreadProxy


class MonitoredQueueBase(ProxyBase):
    """Base class for overriding methods."""

    _in_prefix = b''
    _out_prefix = b''

    def __init__(
        self, in_type, out_type, mon_type=PUB, in_prefix=b'in', out_prefix=b'out'
    ):
        ProxyBase.__init__(self, in_type=in_type, out_type=out_type, mon_type=mon_type)

        self._in_prefix = in_prefix
        self._out_prefix = out_prefix

    def run_device(self):
        ins, outs, mons = self._setup_sockets()
        monitored_queue(ins, outs, mons, self._in_prefix, self._out_prefix)


class MonitoredQueue(MonitoredQueueBase, Proxy):
    """Class for running monitored_queue in the background.

    See zmq.devices.Device for most of the spec. MonitoredQueue differs from Proxy,
    only in that it adds a ``prefix`` to messages sent on the monitor socket,
    with a different prefix for each direction.

    MQ also supports ROUTER on both sides, which zmq.proxy does not.

    If a message arrives on `in_sock`, it will be prefixed with `in_prefix` on the monitor socket.
    If it arrives on out_sock, it will be prefixed with `out_prefix`.

    A PUB socket is the most logical choice for the mon_socket, but it is not required.
    """


class ThreadMonitoredQueue(MonitoredQueueBase, ThreadProxy):
    """Run zmq.monitored_queue in a background thread.

    See MonitoredQueue and Proxy for details.
    """


class ProcessMonitoredQueue(MonitoredQueueBase, ProcessProxy):
    """Run zmq.monitored_queue in a separate process.

    See MonitoredQueue and Proxy for details.
    """


__all__ = ['MonitoredQueue', 'ThreadMonitoredQueue', 'ProcessMonitoredQueue']
