"""Test libzmq security (libzmq >= 3.3.0)"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import contextlib
import os
import time
from threading import Thread

import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85

USER = b"admin"
PASS = b"password"


class TestSecurity(BaseZMQTestCase):
    def setUp(self):
        if zmq.zmq_version_info() < (4, 0):
            raise SkipTest("security is new in libzmq 4.0")
        try:
            zmq.curve_keypair()
        except zmq.ZMQError:
            raise SkipTest("security requires libzmq to be built with CURVE support")
        super().setUp()

    def zap_handler(self):
        socket = self.context.socket(zmq.REP)
        socket.bind("inproc://zeromq.zap.01")
        try:
            msg = self.recv_multipart(socket)

            version, sequence, domain, address, identity, mechanism = msg[:6]
            if mechanism == b'PLAIN':
                username, password = msg[6:]
            elif mechanism == b'CURVE':
                msg[6]

            assert version == b"1.0"
            assert identity == b"IDENT"
            reply = [version, sequence]
            if (
                mechanism == b'CURVE'
                or (mechanism == b'PLAIN' and username == USER and password == PASS)
                or (mechanism == b'NULL')
            ):
                reply.extend(
                    [
                        b"200",
                        b"OK",
                        b"anonymous",
                        b"\5Hello\0\0\0\5World",
                    ]
                )
            else:
                reply.extend(
                    [
                        b"400",
                        b"Invalid username or password",
                        b"",
                        b"",
                    ]
                )
            socket.send_multipart(reply)
        finally:
            socket.close()

    @contextlib.contextmanager
    def zap(self):
        self.start_zap()
        time.sleep(0.5)  # allow time for the Thread to start
        try:
            yield
        finally:
            self.stop_zap()

    def start_zap(self):
        self.zap_thread = Thread(target=self.zap_handler)
        self.zap_thread.start()

    def stop_zap(self):
        self.zap_thread.join()

    def bounce(self, server, client, test_metadata=True):
        msg = [os.urandom(64), os.urandom(64)]
        client.send_multipart(msg)
        frames = self.recv_multipart(server, copy=False)
        recvd = list(map(lambda x: x.bytes, frames))

        try:
            if test_metadata and not PYPY:
                for frame in frames:
                    assert frame.get('User-Id') == 'anonymous'
                    assert frame.get('Hello') == 'World'
                    assert frame['Socket-Type'] == 'DEALER'
        except zmq.ZMQVersionError:
            pass

        assert recvd == msg
        server.send_multipart(recvd)
        msg2 = self.recv_multipart(client)
        assert msg2 == msg

    def test_null(self):
        """test NULL (default) security"""
        server = self.socket(zmq.DEALER)
        client = self.socket(zmq.DEALER)
        assert client.MECHANISM == zmq.NULL
        assert server.mechanism == zmq.NULL
        assert client.plain_server == 0
        assert server.plain_server == 0
        iface = 'tcp://127.0.0.1'
        port = server.bind_to_random_port(iface)
        client.connect("%s:%i" % (iface, port))
        self.bounce(server, client, False)

    def test_plain(self):
        """test PLAIN authentication"""
        server = self.socket(zmq.DEALER)
        server.identity = b'IDENT'
        client = self.socket(zmq.DEALER)
        assert client.plain_username == b''
        assert client.plain_password == b''
        client.plain_username = USER
        client.plain_password = PASS
        assert client.getsockopt(zmq.PLAIN_USERNAME) == USER
        assert client.getsockopt(zmq.PLAIN_PASSWORD) == PASS
        assert client.plain_server == 0
        assert server.plain_server == 0
        server.plain_server = True
        assert server.mechanism == zmq.PLAIN
        assert client.mechanism == zmq.PLAIN

        assert not client.plain_server
        assert server.plain_server

        with self.zap():
            iface = 'tcp://127.0.0.1'
            port = server.bind_to_random_port(iface)
            client.connect("%s:%i" % (iface, port))
            self.bounce(server, client)

    def skip_plain_inauth(self):
        """test PLAIN failed authentication"""
        server = self.socket(zmq.DEALER)
        server.identity = b'IDENT'
        client = self.socket(zmq.DEALER)
        self.sockets.extend([server, client])
        client.plain_username = USER
        client.plain_password = b'incorrect'
        server.plain_server = True
        assert server.mechanism == zmq.PLAIN
        assert client.mechanism == zmq.PLAIN

        with self.zap():
            iface = 'tcp://127.0.0.1'
            port = server.bind_to_random_port(iface)
            client.connect("%s:%i" % (iface, port))
            client.send(b'ping')
            server.rcvtimeo = 250
            self.assertRaisesErrno(zmq.EAGAIN, server.recv)

    def test_keypair(self):
        """test curve_keypair"""
        try:
            public, secret = zmq.curve_keypair()
        except zmq.ZMQError:
            raise SkipTest("CURVE unsupported")

        assert type(secret) == bytes
        assert type(public) == bytes
        assert len(secret) == 40
        assert len(public) == 40

        # verify that it is indeed Z85
        bsecret, bpublic = (z85.decode(key) for key in (public, secret))
        assert type(bsecret) == bytes
        assert type(bpublic) == bytes
        assert len(bsecret) == 32
        assert len(bpublic) == 32

    def test_curve_public(self):
        """test curve_public"""
        try:
            public, secret = zmq.curve_keypair()
        except zmq.ZMQError:
            raise SkipTest("CURVE unsupported")
        if zmq.zmq_version_info() < (4, 2):
            raise SkipTest("curve_public is new in libzmq 4.2")

        derived_public = zmq.curve_public(secret)

        assert type(derived_public) == bytes
        assert len(derived_public) == 40

        # verify that it is indeed Z85
        bpublic = z85.decode(derived_public)
        assert type(bpublic) == bytes
        assert len(bpublic) == 32

        # verify that it is equal to the known public key
        assert derived_public == public

    def test_curve(self):
        """test CURVE encryption"""
        server = self.socket(zmq.DEALER)
        server.identity = b'IDENT'
        client = self.socket(zmq.DEALER)
        self.sockets.extend([server, client])
        try:
            server.curve_server = True
        except zmq.ZMQError as e:
            # will raise EINVAL if no CURVE support
            if e.errno == zmq.EINVAL:
                raise SkipTest("CURVE unsupported")

        server_public, server_secret = zmq.curve_keypair()
        client_public, client_secret = zmq.curve_keypair()

        server.curve_secretkey = server_secret
        server.curve_publickey = server_public
        client.curve_serverkey = server_public
        client.curve_publickey = client_public
        client.curve_secretkey = client_secret

        assert server.mechanism == zmq.CURVE
        assert client.mechanism == zmq.CURVE

        assert server.get(zmq.CURVE_SERVER) == True
        assert client.get(zmq.CURVE_SERVER) == False

        with self.zap():
            iface = 'tcp://127.0.0.1'
            port = server.bind_to_random_port(iface)
            client.connect("%s:%i" % (iface, port))
            self.bounce(server, client)
