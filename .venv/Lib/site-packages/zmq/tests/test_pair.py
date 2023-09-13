# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent

x = b' '


class TestPair(BaseZMQTestCase):
    def test_basic(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)

        msg1 = b'message1'
        msg2 = self.ping_pong(s1, s2, msg1)
        assert msg1 == msg2

    def test_multiple(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)

        for i in range(10):
            msg = i * x
            s1.send(msg)

        for i in range(10):
            msg = i * x
            s2.send(msg)

        for i in range(10):
            msg = s1.recv()
            assert msg == i * x

        for i in range(10):
            msg = s2.recv()
            assert msg == i * x

    def test_json(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        o = dict(a=10, b=list(range(10)))
        self.ping_pong_json(s1, s2, o)

    def test_pyobj(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        o = dict(a=10, b=range(10))
        self.ping_pong_pyobj(s1, s2, o)


if have_gevent:

    class TestReqRepGreen(GreenTest, TestPair):
        pass
