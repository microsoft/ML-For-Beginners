# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import time

import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent


class TestPubSub(BaseZMQTestCase):
    pass

    # We are disabling this test while an issue is being resolved.
    def test_basic(self):
        s1, s2 = self.create_bound_pair(zmq.PUB, zmq.SUB)
        s2.setsockopt(zmq.SUBSCRIBE, b'')
        time.sleep(0.1)
        msg1 = b'message'
        s1.send(msg1)
        msg2 = s2.recv()  # This is blocking!
        assert msg1 == msg2

    def test_topic(self):
        s1, s2 = self.create_bound_pair(zmq.PUB, zmq.SUB)
        s2.setsockopt(zmq.SUBSCRIBE, b'x')
        time.sleep(0.1)
        msg1 = b'message'
        s1.send(msg1)
        self.assertRaisesErrno(zmq.EAGAIN, s2.recv, zmq.NOBLOCK)
        msg1 = b'xmessage'
        s1.send(msg1)
        msg2 = s2.recv()
        assert msg1 == msg2


if have_gevent:

    class TestPubSubGreen(GreenTest, TestPubSub):
        pass
