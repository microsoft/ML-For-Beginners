# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase

from pytest import mark

import zmq
from zmq.utils import jsonapi

try:
    import gevent

    from zmq import green as gzmq

    have_gevent = True
except ImportError:
    have_gevent = False


PYPY = platform.python_implementation() == 'PyPy'

# -----------------------------------------------------------------------------
# skip decorators (directly from unittest)
# -----------------------------------------------------------------------------

_id = lambda x: x

skip_pypy = mark.skipif(PYPY, reason="Doesn't work on PyPy")
require_zmq_4 = mark.skipif(zmq.zmq_version_info() < (4,), reason="requires zmq >= 4")

# -----------------------------------------------------------------------------
# Base test class
# -----------------------------------------------------------------------------


def term_context(ctx, timeout):
    """Terminate a context with a timeout"""
    t = Thread(target=ctx.term)
    t.daemon = True
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # reset Context.instance, so the failure to term doesn't corrupt subsequent tests
        zmq.sugar.context.Context._instance = None
        raise RuntimeError(
            "context could not terminate, open sockets likely remain in test"
        )


class BaseZMQTestCase(TestCase):
    green = False
    teardown_timeout = 10
    test_timeout_seconds = int(os.environ.get("ZMQ_TEST_TIMEOUT") or 60)
    sockets: List[zmq.Socket]

    @property
    def _is_pyzmq_test(self):
        return self.__class__.__module__.split(".", 1)[0] == __name__.split(".", 1)[0]

    @property
    def _should_test_timeout(self):
        return (
            self._is_pyzmq_test
            and hasattr(signal, 'SIGALRM')
            and self.test_timeout_seconds
        )

    @property
    def Context(self):
        if self.green:
            return gzmq.Context
        else:
            return zmq.Context

    def socket(self, socket_type):
        s = self.context.socket(socket_type)
        self.sockets.append(s)
        return s

    def _alarm_timeout(self, timeout, *args):
        raise TimeoutError(f"Test did not complete in {timeout} seconds")

    def setUp(self):
        super().setUp()
        if not self._is_pyzmq_test:
            warnings.warn(
                "zmq.tests.BaseZMQTestCase is deprecated in pyzmq 25, we recommend managing your own contexts and sockets.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.green and not have_gevent:
            raise SkipTest("requires gevent")

        self.context = self.Context.instance()
        self.sockets = []
        if self._should_test_timeout:
            # use SIGALRM to avoid test hangs
            signal.signal(
                signal.SIGALRM, partial(self._alarm_timeout, self.test_timeout_seconds)
            )
            signal.alarm(self.test_timeout_seconds)

    def tearDown(self):
        if self._should_test_timeout:
            # cancel the timeout alarm, if there was one
            signal.alarm(0)
        contexts = {self.context}
        while self.sockets:
            sock = self.sockets.pop()
            contexts.add(sock.context)  # in case additional contexts are created
            sock.close(0)
        for ctx in contexts:
            try:
                term_context(ctx, self.teardown_timeout)
            except Exception:
                # reset Context.instance, so the failure to term doesn't corrupt subsequent tests
                zmq.sugar.context.Context._instance = None
                raise

        super().tearDown()

    def create_bound_pair(
        self, type1=zmq.PAIR, type2=zmq.PAIR, interface='tcp://127.0.0.1'
    ):
        """Create a bound socket pair using a random port."""
        s1 = self.context.socket(type1)
        s1.setsockopt(zmq.LINGER, 0)
        port = s1.bind_to_random_port(interface)
        s2 = self.context.socket(type2)
        s2.setsockopt(zmq.LINGER, 0)
        s2.connect(f'{interface}:{port}')
        self.sockets.extend([s1, s2])
        return s1, s2

    def ping_pong(self, s1, s2, msg):
        s1.send(msg)
        msg2 = s2.recv()
        s2.send(msg2)
        msg3 = s1.recv()
        return msg3

    def ping_pong_json(self, s1, s2, o):
        if jsonapi.jsonmod is None:
            raise SkipTest("No json library")
        s1.send_json(o)
        o2 = s2.recv_json()
        s2.send_json(o2)
        o3 = s1.recv_json()
        return o3

    def ping_pong_pyobj(self, s1, s2, o):
        s1.send_pyobj(o)
        o2 = s2.recv_pyobj()
        s2.send_pyobj(o2)
        o3 = s1.recv_pyobj()
        return o3

    def assertRaisesErrno(self, errno, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except zmq.ZMQError as e:
            self.assertEqual(
                e.errno,
                errno,
                "wrong error raised, expected '%s' \
got '%s'"
                % (zmq.ZMQError(errno), zmq.ZMQError(e.errno)),
            )
        else:
            self.fail("Function did not raise any error")

    def _select_recv(self, multipart, socket, **kwargs):
        """call recv[_multipart] in a way that raises if there is nothing to receive"""
        if zmq.zmq_version_info() >= (3, 1, 0):
            # zmq 3.1 has a bug, where poll can return false positives,
            # so we wait a little bit just in case
            # See LIBZMQ-280 on JIRA
            time.sleep(0.1)

        r, w, x = zmq.select([socket], [], [], timeout=kwargs.pop('timeout', 5))
        assert len(r) > 0, "Should have received a message"
        kwargs['flags'] = zmq.DONTWAIT | kwargs.get('flags', 0)

        recv = socket.recv_multipart if multipart else socket.recv
        return recv(**kwargs)

    def recv(self, socket, **kwargs):
        """call recv in a way that raises if there is nothing to receive"""
        return self._select_recv(False, socket, **kwargs)

    def recv_multipart(self, socket, **kwargs):
        """call recv_multipart in a way that raises if there is nothing to receive"""
        return self._select_recv(True, socket, **kwargs)


class PollZMQTestCase(BaseZMQTestCase):
    pass


class GreenTest:
    """Mixin for making green versions of test classes"""

    green = True
    teardown_timeout = 10

    def assertRaisesErrno(self, errno, func, *args, **kwargs):
        if errno == zmq.EAGAIN:
            raise SkipTest("Skipping because we're green.")
        try:
            func(*args, **kwargs)
        except zmq.ZMQError:
            e = sys.exc_info()[1]
            self.assertEqual(
                e.errno,
                errno,
                "wrong error raised, expected '%s' \
got '%s'"
                % (zmq.ZMQError(errno), zmq.ZMQError(e.errno)),
            )
        else:
            self.fail("Function did not raise any error")

    def tearDown(self):
        if self._should_test_timeout:
            # cancel the timeout alarm, if there was one
            signal.alarm(0)
        contexts = {self.context}
        while self.sockets:
            sock = self.sockets.pop()
            contexts.add(sock.context)  # in case additional contexts are created
            sock.close()
        try:
            gevent.joinall(
                [gevent.spawn(ctx.term) for ctx in contexts],
                timeout=self.teardown_timeout,
                raise_error=True,
            )
        except gevent.Timeout:
            raise RuntimeError(
                "context could not terminate, open sockets likely remain in test"
            )

    def skip_green(self):
        raise SkipTest("Skipping because we are green")


def skip_green(f):
    def skipping_test(self, *args, **kwargs):
        if self.green:
            raise SkipTest("Skipping because we are green")
        else:
            return f(self, *args, **kwargs)

    return skipping_test
