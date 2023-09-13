# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import time

import pytest

import zmq
from zmq.tests import BaseZMQTestCase


class TestDraftSockets(BaseZMQTestCase):
    def setUp(self):
        if not zmq.DRAFT_API:
            pytest.skip("draft api unavailable")
        super().setUp()

    def test_client_server(self):
        client, server = self.create_bound_pair(zmq.CLIENT, zmq.SERVER)
        client.send(b'request')
        msg = self.recv(server, copy=False)
        assert msg.routing_id is not None
        server.send(b'reply', routing_id=msg.routing_id)
        reply = self.recv(client)
        assert reply == b'reply'

    def test_radio_dish(self):
        dish, radio = self.create_bound_pair(zmq.DISH, zmq.RADIO)
        dish.rcvtimeo = 250
        group = 'mygroup'
        dish.join(group)
        received_count = 0
        received = set()
        sent = set()
        for i in range(10):
            msg = str(i).encode('ascii')
            sent.add(msg)
            radio.send(msg, group=group)
            try:
                recvd = dish.recv()
            except zmq.Again:
                time.sleep(0.1)
            else:
                received.add(recvd)
                received_count += 1
        # assert that we got *something*
        assert len(received.intersection(sent)) >= 5
