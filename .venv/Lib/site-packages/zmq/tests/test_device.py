# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import time

import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent

if PYPY:
    # cleanup of shared Context doesn't work on PyPy
    devices.Device.context_factory = zmq.Context


class TestDevice(BaseZMQTestCase):
    def test_device_types(self):
        for devtype in (zmq.STREAMER, zmq.FORWARDER, zmq.QUEUE):
            dev = devices.Device(devtype, zmq.PAIR, zmq.PAIR)
            assert dev.device_type == devtype
            del dev

    def test_device_attributes(self):
        dev = devices.Device(zmq.QUEUE, zmq.SUB, zmq.PUB)
        assert dev.in_type == zmq.SUB
        assert dev.out_type == zmq.PUB
        assert dev.device_type == zmq.QUEUE
        assert dev.daemon == True
        del dev

    def test_single_socket_forwarder_connect(self):
        if zmq.zmq_version() in ('4.1.1', '4.0.6'):
            raise SkipTest("libzmq-%s broke single-socket devices" % zmq.zmq_version())
        dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
        req = self.context.socket(zmq.REQ)
        port = req.bind_to_random_port('tcp://127.0.0.1')
        dev.connect_in('tcp://127.0.0.1:%i' % port)
        dev.start()
        time.sleep(0.25)
        msg = b'hello'
        req.send(msg)
        assert msg == self.recv(req)
        del dev
        req.close()
        dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
        req = self.context.socket(zmq.REQ)
        port = req.bind_to_random_port('tcp://127.0.0.1')
        dev.connect_out('tcp://127.0.0.1:%i' % port)
        dev.start()
        time.sleep(0.25)
        msg = b'hello again'
        req.send(msg)
        assert msg == self.recv(req)
        del dev
        req.close()

    def test_single_socket_forwarder_bind(self):
        if zmq.zmq_version() in ('4.1.1', '4.0.6'):
            raise SkipTest("libzmq-%s broke single-socket devices" % zmq.zmq_version())
        dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
        port = dev.bind_in_to_random_port('tcp://127.0.0.1')
        req = self.context.socket(zmq.REQ)
        req.connect('tcp://127.0.0.1:%i' % port)
        dev.start()
        time.sleep(0.25)
        msg = b'hello'
        req.send(msg)
        assert msg == self.recv(req)
        del dev
        req.close()
        dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
        port = dev.bind_in_to_random_port('tcp://127.0.0.1')
        req = self.context.socket(zmq.REQ)
        req.connect('tcp://127.0.0.1:%i' % port)
        dev.start()
        time.sleep(0.25)
        msg = b'hello again'
        req.send(msg)
        assert msg == self.recv(req)
        del dev
        req.close()

    def test_device_bind_to_random_with_args(self):
        dev = devices.ThreadDevice(zmq.PULL, zmq.PUSH, -1)
        iface = 'tcp://127.0.0.1'
        ports = []
        min, max = 5000, 5050
        ports.extend(
            [
                dev.bind_in_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_out_to_random_port(iface, min_port=min, max_port=max),
            ]
        )
        for port in ports:
            if port < min or port > max:
                self.fail('Unexpected port number: %i' % port)

    def test_device_bind_to_random_binderror(self):
        dev = devices.ThreadDevice(zmq.PULL, zmq.PUSH, -1)
        iface = 'tcp://127.0.0.1'
        try:
            for i in range(11):
                dev.bind_in_to_random_port(iface, min_port=10000, max_port=10010)
        except zmq.ZMQBindError as e:
            return
        else:
            self.fail('Should have failed')

    def test_proxy(self):
        if zmq.zmq_version_info() < (3, 2):
            raise SkipTest("Proxies only in libzmq >= 3")
        dev = devices.ThreadProxy(zmq.PULL, zmq.PUSH, zmq.PUSH)
        iface = 'tcp://127.0.0.1'
        port = dev.bind_in_to_random_port(iface)
        port2 = dev.bind_out_to_random_port(iface)
        port3 = dev.bind_mon_to_random_port(iface)
        dev.start()
        time.sleep(0.25)
        msg = b'hello'
        push = self.context.socket(zmq.PUSH)
        push.connect("%s:%i" % (iface, port))
        pull = self.context.socket(zmq.PULL)
        pull.connect("%s:%i" % (iface, port2))
        mon = self.context.socket(zmq.PULL)
        mon.connect("%s:%i" % (iface, port3))
        push.send(msg)
        self.sockets.extend([push, pull, mon])
        assert msg == self.recv(pull)
        assert msg == self.recv(mon)

    def test_proxy_bind_to_random_with_args(self):
        if zmq.zmq_version_info() < (3, 2):
            raise SkipTest("Proxies only in libzmq >= 3")
        dev = devices.ThreadProxy(zmq.PULL, zmq.PUSH, zmq.PUSH)
        iface = 'tcp://127.0.0.1'
        ports = []
        min, max = 5000, 5050
        ports.extend(
            [
                dev.bind_in_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_out_to_random_port(iface, min_port=min, max_port=max),
                dev.bind_mon_to_random_port(iface, min_port=min, max_port=max),
            ]
        )
        for port in ports:
            if port < min or port > max:
                self.fail('Unexpected port number: %i' % port)


if have_gevent:
    import gevent

    import zmq.green

    class TestDeviceGreen(GreenTest, BaseZMQTestCase):
        def test_green_device(self):
            rep = self.context.socket(zmq.REP)
            req = self.context.socket(zmq.REQ)
            self.sockets.extend([req, rep])
            port = rep.bind_to_random_port('tcp://127.0.0.1')
            g = gevent.spawn(zmq.green.device, zmq.QUEUE, rep, rep)
            req.connect('tcp://127.0.0.1:%i' % port)
            req.send(b'hi')
            timeout = gevent.Timeout(3)
            timeout.start()
            receiver = gevent.spawn(req.recv)
            assert receiver.get(2) == b'hi'
            timeout.cancel()
            g.kill(block=True)
