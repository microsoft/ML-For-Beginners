# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import threading
import time

import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase

if PYPY or zmq.zmq_version_info() >= (4, 1):
    # cleanup of shared Context doesn't work on PyPy
    # there also seems to be a bug in cleanup in libzmq-4.1 (zeromq/libzmq#1052)
    devices.Device.context_factory = zmq.Context


class TestMonitoredQueue(BaseZMQTestCase):
    def build_device(self, mon_sub=b"", in_prefix=b'in', out_prefix=b'out'):
        self.device = devices.ThreadMonitoredQueue(
            zmq.PAIR, zmq.PAIR, zmq.PUB, in_prefix, out_prefix
        )
        alice = self.context.socket(zmq.PAIR)
        bob = self.context.socket(zmq.PAIR)
        mon = self.context.socket(zmq.SUB)

        aport = alice.bind_to_random_port('tcp://127.0.0.1')
        bport = bob.bind_to_random_port('tcp://127.0.0.1')
        mport = mon.bind_to_random_port('tcp://127.0.0.1')
        mon.setsockopt(zmq.SUBSCRIBE, mon_sub)

        self.device.connect_in("tcp://127.0.0.1:%i" % aport)
        self.device.connect_out("tcp://127.0.0.1:%i" % bport)
        self.device.connect_mon("tcp://127.0.0.1:%i" % mport)
        self.device.start()
        time.sleep(0.2)
        try:
            # this is currently necessary to ensure no dropped monitor messages
            # see LIBZMQ-248 for more info
            mon.recv_multipart(zmq.NOBLOCK)
        except zmq.ZMQError:
            pass
        self.sockets.extend([alice, bob, mon])
        return alice, bob, mon

    def teardown_device(self):
        # spawn term in a background thread
        for i in range(50):
            # wait for device._context to be populated
            context = getattr(self.device, "_context", None)
            if context is not None:
                break
            time.sleep(0.1)

        if context is not None:
            t = threading.Thread(target=self.device._context.term, daemon=True)
            t.start()

        for socket in self.sockets:
            socket.close()

        if context is not None:
            t.join(timeout=5)

        self.device.join(timeout=5)

    def test_reply(self):
        alice, bob, mon = self.build_device()
        alices = b"hello bob".split()
        alice.send_multipart(alices)
        bobs = self.recv_multipart(bob)
        assert alices == bobs
        bobs = b"hello alice".split()
        bob.send_multipart(bobs)
        alices = self.recv_multipart(alice)
        assert alices == bobs
        self.teardown_device()

    def test_queue(self):
        alice, bob, mon = self.build_device()
        alices = b"hello bob".split()
        alice.send_multipart(alices)
        alices2 = b"hello again".split()
        alice.send_multipart(alices2)
        alices3 = b"hello again and again".split()
        alice.send_multipart(alices3)
        bobs = self.recv_multipart(bob)
        assert alices == bobs
        bobs = self.recv_multipart(bob)
        assert alices2 == bobs
        bobs = self.recv_multipart(bob)
        assert alices3 == bobs
        bobs = b"hello alice".split()
        bob.send_multipart(bobs)
        alices = self.recv_multipart(alice)
        assert alices == bobs
        self.teardown_device()

    def test_monitor(self):
        alice, bob, mon = self.build_device()
        alices = b"hello bob".split()
        alice.send_multipart(alices)
        alices2 = b"hello again".split()
        alice.send_multipart(alices2)
        alices3 = b"hello again and again".split()
        alice.send_multipart(alices3)
        bobs = self.recv_multipart(bob)
        assert alices == bobs
        mons = self.recv_multipart(mon)
        assert [b'in'] + bobs == mons
        bobs = self.recv_multipart(bob)
        assert alices2 == bobs
        bobs = self.recv_multipart(bob)
        assert alices3 == bobs
        mons = self.recv_multipart(mon)
        assert [b'in'] + alices2 == mons
        bobs = b"hello alice".split()
        bob.send_multipart(bobs)
        alices = self.recv_multipart(alice)
        assert alices == bobs
        mons = self.recv_multipart(mon)
        assert [b'in'] + alices3 == mons
        mons = self.recv_multipart(mon)
        assert [b'out'] + bobs == mons
        self.teardown_device()

    def test_prefix(self):
        alice, bob, mon = self.build_device(b"", b'foo', b'bar')
        alices = b"hello bob".split()
        alice.send_multipart(alices)
        alices2 = b"hello again".split()
        alice.send_multipart(alices2)
        alices3 = b"hello again and again".split()
        alice.send_multipart(alices3)
        bobs = self.recv_multipart(bob)
        assert alices == bobs
        mons = self.recv_multipart(mon)
        assert [b'foo'] + bobs == mons
        bobs = self.recv_multipart(bob)
        assert alices2 == bobs
        bobs = self.recv_multipart(bob)
        assert alices3 == bobs
        mons = self.recv_multipart(mon)
        assert [b'foo'] + alices2 == mons
        bobs = b"hello alice".split()
        bob.send_multipart(bobs)
        alices = self.recv_multipart(alice)
        assert alices == bobs
        mons = self.recv_multipart(mon)
        assert [b'foo'] + alices3 == mons
        mons = self.recv_multipart(mon)
        assert [b'bar'] + bobs == mons
        self.teardown_device()

    def test_monitor_subscribe(self):
        alice, bob, mon = self.build_device(b"out")
        alices = b"hello bob".split()
        alice.send_multipart(alices)
        alices2 = b"hello again".split()
        alice.send_multipart(alices2)
        alices3 = b"hello again and again".split()
        alice.send_multipart(alices3)
        bobs = self.recv_multipart(bob)
        assert alices == bobs
        bobs = self.recv_multipart(bob)
        assert alices2 == bobs
        bobs = self.recv_multipart(bob)
        assert alices3 == bobs
        bobs = b"hello alice".split()
        bob.send_multipart(bobs)
        alices = self.recv_multipart(alice)
        assert alices == bobs
        mons = self.recv_multipart(mon)
        assert [b'out'] + bobs == mons
        self.teardown_device()

    def test_router_router(self):
        """test router-router MQ devices"""
        dev = devices.ThreadMonitoredQueue(
            zmq.ROUTER, zmq.ROUTER, zmq.PUB, b'in', b'out'
        )
        self.device = dev
        dev.setsockopt_in(zmq.LINGER, 0)
        dev.setsockopt_out(zmq.LINGER, 0)
        dev.setsockopt_mon(zmq.LINGER, 0)

        porta = dev.bind_in_to_random_port('tcp://127.0.0.1')
        portb = dev.bind_out_to_random_port('tcp://127.0.0.1')
        a = self.context.socket(zmq.DEALER)
        a.identity = b'a'
        b = self.context.socket(zmq.DEALER)
        b.identity = b'b'
        self.sockets.extend([a, b])

        a.connect('tcp://127.0.0.1:%i' % porta)
        b.connect('tcp://127.0.0.1:%i' % portb)
        dev.start()
        time.sleep(1)
        if zmq.zmq_version_info() >= (3, 1, 0):
            # flush erroneous poll state, due to LIBZMQ-280
            ping_msg = [b'ping', b'pong']
            for s in (a, b):
                s.send_multipart(ping_msg)
                try:
                    s.recv(zmq.NOBLOCK)
                except zmq.ZMQError:
                    pass
        msg = [b'hello', b'there']
        a.send_multipart([b'b'] + msg)
        bmsg = self.recv_multipart(b)
        assert bmsg == [b'a'] + msg
        b.send_multipart(bmsg)
        amsg = self.recv_multipart(a)
        assert amsg == [b'b'] + msg
        self.teardown_device()

    def test_default_mq_args(self):
        self.device = dev = devices.ThreadMonitoredQueue(
            zmq.ROUTER, zmq.DEALER, zmq.PUB
        )
        dev.setsockopt_in(zmq.LINGER, 0)
        dev.setsockopt_out(zmq.LINGER, 0)
        dev.setsockopt_mon(zmq.LINGER, 0)
        # this will raise if default args are wrong
        dev.start()
        self.teardown_device()

    def test_mq_check_prefix(self):
        ins = self.context.socket(zmq.ROUTER)
        outs = self.context.socket(zmq.DEALER)
        mons = self.context.socket(zmq.PUB)
        self.sockets.extend([ins, outs, mons])

        ins = 'in'
        outs = 'out'
        self.assertRaises(TypeError, devices.monitoredqueue, ins, outs, mons)
