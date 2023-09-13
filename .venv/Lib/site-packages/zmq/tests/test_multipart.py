# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent


class TestMultipart(BaseZMQTestCase):
    def test_router_dealer(self):
        router, dealer = self.create_bound_pair(zmq.ROUTER, zmq.DEALER)

        msg1 = b'message1'
        dealer.send(msg1)
        self.recv(router)
        more = router.rcvmore
        assert more == True
        msg2 = self.recv(router)
        assert msg1 == msg2
        more = router.rcvmore
        assert more == False

    def test_basic_multipart(self):
        a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        msg = [b'hi', b'there', b'b']
        a.send_multipart(msg)
        recvd = b.recv_multipart()
        assert msg == recvd


if have_gevent:

    class TestMultipartGreen(GreenTest, TestMultipart):
        pass
