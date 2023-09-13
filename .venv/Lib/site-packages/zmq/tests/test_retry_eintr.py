# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import signal
import time
from threading import Thread

from pytest import mark

import zmq
from zmq.tests import BaseZMQTestCase, SkipTest

# Partially based on EINTRBaseTest from CPython 3.5 eintr_tester


class TestEINTRSysCall(BaseZMQTestCase):
    """Base class for EINTR tests."""

    # delay for initial signal delivery
    signal_delay = 0.1
    # timeout for tests. Must be > signal_delay
    timeout = 0.25
    timeout_ms = int(timeout * 1e3)

    def alarm(self, t=None):
        """start a timer to fire only once

        like signal.alarm, but with better resolution than integer seconds.
        """
        if not hasattr(signal, 'setitimer'):
            raise SkipTest('EINTR tests require setitimer')
        if t is None:
            t = self.signal_delay
        self.timer_fired = False
        self.orig_handler = signal.signal(signal.SIGALRM, self.stop_timer)
        # signal_period ignored, since only one timer event is allowed to fire
        signal.setitimer(signal.ITIMER_REAL, t, 1000)

    def stop_timer(self, *args):
        self.timer_fired = True
        signal.setitimer(signal.ITIMER_REAL, 0, 0)
        signal.signal(signal.SIGALRM, self.orig_handler)

    @mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason="requires RCVTIMEO")
    def test_retry_recv(self):
        pull = self.socket(zmq.PULL)
        pull.rcvtimeo = self.timeout_ms
        self.alarm()
        self.assertRaises(zmq.Again, pull.recv)
        assert self.timer_fired

    @mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason="requires SNDTIMEO")
    def test_retry_send(self):
        push = self.socket(zmq.PUSH)
        push.sndtimeo = self.timeout_ms
        self.alarm()
        self.assertRaises(zmq.Again, push.send, b'buf')
        assert self.timer_fired

    @mark.flaky(reruns=3)
    def test_retry_poll(self):
        x, y = self.create_bound_pair()
        poller = zmq.Poller()
        poller.register(x, zmq.POLLIN)
        self.alarm()

        def send():
            time.sleep(2 * self.signal_delay)
            y.send(b'ping')

        t = Thread(target=send)
        t.start()
        evts = dict(poller.poll(2 * self.timeout_ms))
        t.join()
        assert x in evts
        assert self.timer_fired
        x.recv()

    def test_retry_term(self):
        push = self.socket(zmq.PUSH)
        push.linger = self.timeout_ms
        push.connect('tcp://127.0.0.1:5555')
        push.send(b'ping')
        time.sleep(0.1)
        self.alarm()
        self.context.destroy()
        assert self.timer_fired
        assert self.context.closed

    def test_retry_getsockopt(self):
        raise SkipTest("TODO: find a way to interrupt getsockopt")

    def test_retry_setsockopt(self):
        raise SkipTest("TODO: find a way to interrupt setsockopt")
