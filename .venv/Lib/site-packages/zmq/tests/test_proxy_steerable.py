# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import struct
import time

import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest

if PYPY:
    # cleanup of shared Context doesn't work on PyPy
    devices.Device.context_factory = zmq.Context


class TestProxySteerable(BaseZMQTestCase):
    def test_proxy_steerable(self):
        if zmq.zmq_version_info() < (4, 1):
            raise SkipTest("Steerable Proxies only in libzmq >= 4.1")
        dev = devices.ThreadProxySteerable(zmq.PULL, zmq.PUSH, zmq.PUSH, zmq.PAIR)
        iface = 'tcp://127.0.0.1'
        port = dev.bind_in_to_random_port(iface)
        port2 = dev.bind_out_to_random_port(iface)
        port3 = dev.bind_mon_to_random_port(iface)
        port4 = dev.bind_ctrl_to_random_port(iface)
        dev.start()
        time.sleep(0.25)
        msg = b'hello'
        push = self.context.socket(zmq.PUSH)
        push.connect("%s:%i" % (iface, port))
        pull = self.context.socket(zmq.PULL)
        pull.connect("%s:%i" % (iface, port2))
        mon = self.context.socket(zmq.PULL)
        mon.connect("%s:%i" % (iface, port3))
        ctrl = self.context.socket(zmq.PAIR)
        ctrl.connect("%s:%i" % (iface, port4))
        push.send(msg)
        self.sockets.extend([push, pull, mon, ctrl])
        assert msg == self.recv(pull)
        assert msg == self.recv(mon)
        ctrl.send(b'TERMINATE')
        dev.join()

    def test_proxy_steerable_bind_to_random_with_args(self):
        if zmq.zmq_version_info() < (4, 1):
            raise SkipTest("Steerable Proxies only in libzmq >= 4.1")
        dev = devices.ThreadProxySteerable(zmq.PULL, zmq.PUSH, zmq.PUSH, zmq.PAIR)
        iface = 'tcp://127.0.0.1'
        ports = []
        min, max = 5000, 5050
        ports.extend(
            [
                dev.bind_in_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_out_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_mon_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_ctrl_to_random_port(iface, min_port=min, max_port=max),
            ]
        )
        for port in ports:
            if port < min or port > max:
                self.fail('Unexpected port number: %i' % port)

    def test_proxy_steerable_statistics(self):
        if zmq.zmq_version_info() < (4, 3):
            raise SkipTest("STATISTICS only in libzmq >= 4.3")
        dev = devices.ThreadProxySteerable(zmq.PULL, zmq.PUSH, zmq.PUSH, zmq.PAIR)
        iface = 'tcp://127.0.0.1'
        port = dev.bind_in_to_random_port(iface)
        port2 = dev.bind_out_to_random_port(iface)
        port3 = dev.bind_mon_to_random_port(iface)
        port4 = dev.bind_ctrl_to_random_port(iface)
        dev.start()
        time.sleep(0.25)
        msg = b'hello'
        push = self.context.socket(zmq.PUSH)
        push.connect("%s:%i" % (iface, port))
        pull = self.context.socket(zmq.PULL)
        pull.connect("%s:%i" % (iface, port2))
        mon = self.context.socket(zmq.PULL)
        mon.connect("%s:%i" % (iface, port3))
        ctrl = self.context.socket(zmq.PAIR)
        ctrl.connect("%s:%i" % (iface, port4))
        push.send(msg)
        self.sockets.extend([push, pull, mon, ctrl])
        assert msg == self.recv(pull)
        assert msg == self.recv(mon)
        ctrl.send(b'STATISTICS')
        stats = self.recv_multipart(ctrl)
        stats_int = [struct.unpack("=Q", x)[0] for x in stats]
        assert 1 == stats_int[0]
        assert len(msg) == stats_int[1]
        assert 1 == stats_int[6]
        assert len(msg) == stats_int[7]
        ctrl.send(b'TERMINATE')
        dev.join()
