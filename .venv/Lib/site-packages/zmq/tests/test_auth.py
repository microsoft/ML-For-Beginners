# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager

import pytest

import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy

try:
    import tornado
except ImportError:
    tornado = None


@pytest.fixture
def Context(event_loop):
    return zmq.asyncio.Context


@pytest.fixture
def create_certs(tmpdir):
    """Create CURVE certificates for a test"""

    # Create temporary CURVE key pairs for this test run. We create all keys in a
    # temp directory and then move them into the appropriate private or public
    # directory.
    base_dir = str(tmpdir.join("certs").mkdir())
    keys_dir = os.path.join(base_dir, "certificates")
    public_keys_dir = os.path.join(base_dir, 'public_keys')
    secret_keys_dir = os.path.join(base_dir, 'private_keys')

    os.mkdir(keys_dir)
    os.mkdir(public_keys_dir)
    os.mkdir(secret_keys_dir)

    server_public_file, server_secret_file = zmq.auth.create_certificates(
        keys_dir, "server"
    )
    client_public_file, client_secret_file = zmq.auth.create_certificates(
        keys_dir, "client"
    )
    for key_file in os.listdir(keys_dir):
        if key_file.endswith(".key"):
            shutil.move(
                os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, '.')
            )

    for key_file in os.listdir(keys_dir):
        if key_file.endswith(".key_secret"):
            shutil.move(
                os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, '.')
            )

    return (public_keys_dir, secret_keys_dir)


def load_certs(secret_keys_dir):
    """Return server and client certificate keys"""
    server_secret_file = os.path.join(secret_keys_dir, "server.key_secret")
    client_secret_file = os.path.join(secret_keys_dir, "client.key_secret")

    server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
    client_public, client_secret = zmq.auth.load_certificate(client_secret_file)

    return server_public, server_secret, client_public, client_secret


@pytest.fixture
def public_keys_dir(create_certs):
    public_keys_dir, secret_keys_dir = create_certs
    return public_keys_dir


@pytest.fixture
def secret_keys_dir(create_certs):
    public_keys_dir, secret_keys_dir = create_certs
    return secret_keys_dir


@pytest.fixture
def certs(secret_keys_dir):
    return load_certs(secret_keys_dir)


@pytest.fixture
async def _async_setup(request, event_loop):
    """pytest doesn't support async setup/teardown"""
    instance = request.instance
    await instance.async_setup()
    yield
    # make sure our teardown runs before the loop closes
    instance.async_teardown()


@pytest.mark.usefixtures("_async_setup")
class AuthTest:
    auth = None

    async def async_setup(self):
        self.context = zmq.asyncio.Context()
        if zmq.zmq_version_info() < (4, 0):
            raise SkipTest("security is new in libzmq 4.0")
        try:
            zmq.curve_keypair()
        except zmq.ZMQError:
            raise SkipTest("security requires libzmq to have curve support")
        # enable debug logging while we run tests
        logging.getLogger('zmq.auth').setLevel(logging.DEBUG)
        self.auth = self.make_auth()
        await self.start_auth()

    def async_teardown(self):
        if self.auth:
            self.auth.stop()
            self.auth = None
        self.context.term()

    def make_auth(self):
        raise NotImplementedError()

    async def start_auth(self):
        self.auth.start()

    async def can_connect(self, server, client, timeout=1000):
        """Check if client can connect to server using tcp transport"""
        result = False
        iface = 'tcp://127.0.0.1'
        port = server.bind_to_random_port(iface)
        client.connect("%s:%i" % (iface, port))
        msg = [b"Hello World"]
        # run poll on server twice
        # to flush spurious events
        await server.poll(100, zmq.POLLOUT)

        if await server.poll(timeout, zmq.POLLOUT):
            try:
                await server.send_multipart(msg, zmq.NOBLOCK)
            except zmq.Again:
                warnings.warn("server set POLLOUT, but cannot send", RuntimeWarning)
                return False
        else:
            return False

        if await client.poll(timeout):
            try:
                rcvd_msg = await client.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                warnings.warn("client set POLLIN, but cannot recv", RuntimeWarning)
            else:
                assert rcvd_msg == msg
                result = True
        return result

    @contextmanager
    def push_pull(self):
        with self.context.socket(zmq.PUSH) as server, self.context.socket(
            zmq.PULL
        ) as client:
            server.linger = 0
            server.sndtimeo = 2000
            client.linger = 0
            client.rcvtimeo = 2000
            yield server, client

    @contextmanager
    def curve_push_pull(self, certs, client_key="ok"):
        server_public, server_secret, client_public, client_secret = certs
        with self.push_pull() as (server, client):
            server.curve_publickey = server_public
            server.curve_secretkey = server_secret
            server.curve_server = True
            if client_key is not None:
                client.curve_publickey = client_public
                client.curve_secretkey = client_secret
                if client_key == "ok":
                    client.curve_serverkey = server_public
                else:
                    private, public = zmq.curve_keypair()
                    client.curve_serverkey = public
            yield (server, client)

    async def test_null(self):
        """threaded auth - NULL"""
        # A default NULL connection should always succeed, and not
        # go through our authentication infrastructure at all.
        self.auth.stop()
        self.auth = None

        # use a new context, so ZAP isn't inherited
        self.context.term()
        self.context = zmq.asyncio.Context()

        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)

        # By setting a domain we switch on authentication for NULL sockets,
        # though no policies are configured yet. The client connection
        # should still be allowed.
        with self.push_pull() as (server, client):
            server.zap_domain = b'global'
            assert await self.can_connect(server, client)

    async def test_deny(self):
        # deny 127.0.0.1, connection should fail
        self.auth.deny('127.0.0.1')
        with pytest.raises(ValueError):
            self.auth.allow("127.0.0.2")
        with self.push_pull() as (server, client):
            # By setting a domain we switch on authentication for NULL sockets,
            # though no policies are configured yet.
            server.zap_domain = b'global'
            assert not await self.can_connect(server, client, timeout=100)

    async def test_allow(self):
        # allow 127.0.0.1, connection should pass
        self.auth.allow('127.0.0.1')
        with pytest.raises(ValueError):
            self.auth.deny("127.0.0.2")
        with self.push_pull() as (server, client):
            # By setting a domain we switch on authentication for NULL sockets,
            # though no policies are configured yet.
            server.zap_domain = b'global'
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize(
        "enabled, password, success",
        [
            (True, "correct", True),
            (False, "correct", False),
            (True, "incorrect", False),
        ],
    )
    async def test_plain(self, enabled, password, success):
        """threaded auth - PLAIN"""

        # Try PLAIN authentication - without configuring server, connection should fail
        with self.push_pull() as (server, client):
            server.plain_server = True
            if password:
                client.plain_username = b'admin'
                client.plain_password = password.encode("ascii")
            if enabled:
                self.auth.configure_plain(domain='*', passwords={'admin': 'correct'})
            if success:
                assert await self.can_connect(server, client)
            else:
                assert not await self.can_connect(server, client, timeout=100)

        # Remove authenticator and check that a normal connection works
        self.auth.stop()
        self.auth = None
        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize(
        "client_key, location, success",
        [
            ('ok', zmq.auth.CURVE_ALLOW_ANY, True),
            ('ok', "public_keys", True),
            ('bad', "public_keys", False),
            (None, "public_keys", False),
        ],
    )
    async def test_curve(self, certs, public_keys_dir, client_key, location, success):
        """threaded auth - CURVE"""
        self.auth.allow('127.0.0.1')

        # Try CURVE authentication - without configuring server, connection should fail
        with self.curve_push_pull(certs, client_key=client_key) as (server, client):
            if location:
                if location == 'public_keys':
                    location = public_keys_dir
                self.auth.configure_curve(domain='*', location=location)
            if success:
                assert await self.can_connect(server, client, timeout=100)
            else:
                assert not await self.can_connect(server, client, timeout=100)

        # Remove authenticator and check that a normal connection works
        self.auth.stop()
        self.auth = None

        # Try connecting using NULL and no authentication enabled, connection should pass
        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize("key", ["ok", "wrong"])
    @pytest.mark.parametrize("async_", [True, False])
    async def test_curve_callback(self, certs, key, async_):
        """threaded auth - CURVE with callback authentication"""
        self.auth.allow('127.0.0.1')
        server_public, server_secret, client_public, client_secret = certs

        class CredentialsProvider:
            def __init__(self):
                if key == "ok":
                    self.client = client_public
                else:
                    self.client = server_public

            def callback(self, domain, key):
                if key == self.client:
                    return True
                else:
                    return False

            async def async_callback(self, domain, key):
                await asyncio.sleep(0.1)
                if key == self.client:
                    return True
                else:
                    return False

        if async_:
            CredentialsProvider.callback = CredentialsProvider.async_callback
        provider = CredentialsProvider()
        self.auth.configure_curve_callback(credentials_provider=provider)
        with self.curve_push_pull(certs) as (server, client):
            if key == "ok":
                assert await self.can_connect(server, client)
            else:
                assert not await self.can_connect(server, client, timeout=200)

    @skip_pypy
    async def test_curve_user_id(self, certs, public_keys_dir):
        """threaded auth - CURVE"""
        self.auth.allow('127.0.0.1')
        server_public, server_secret, client_public, client_secret = certs
        self.auth.configure_curve(domain='*', location=public_keys_dir)
        # reverse server-client relationship, so server is PULL
        with self.push_pull() as (client, server):
            server.curve_publickey = server_public
            server.curve_secretkey = server_secret
            server.curve_server = True

            client.curve_publickey = client_public
            client.curve_secretkey = client_secret
            client.curve_serverkey = server_public

            assert await self.can_connect(client, server)

            # test default user-id map
            await client.send(b'test')
            msg = await server.recv(copy=False)
            assert msg.bytes == b'test'
            try:
                user_id = msg.get('User-Id')
            except zmq.ZMQVersionError:
                pass
            else:
                assert user_id == client_public.decode("utf8")

            # test custom user-id map
            self.auth.curve_user_id = lambda client_key: 'custom'

            with self.context.socket(zmq.PUSH) as client2:
                client2.curve_publickey = client_public
                client2.curve_secretkey = client_secret
                client2.curve_serverkey = server_public
                assert await self.can_connect(client2, server)

                await client2.send(b'test2')
                msg = await server.recv(copy=False)
                assert msg.bytes == b'test2'
                try:
                    user_id = msg.get('User-Id')
                except zmq.ZMQVersionError:
                    pass
                else:
                    assert user_id == 'custom'


class TestThreadAuthentication(AuthTest):
    """Test authentication running in a thread"""

    def make_auth(self):
        from zmq.auth.thread import ThreadAuthenticator

        return ThreadAuthenticator(self.context)


@pytest.mark.skipif(
    sys.platform == 'win32' and sys.version_info < (3, 7),
    reason="flaky event loop cleanup on windows+py36",
)
class TestAsyncioAuthentication(AuthTest):
    """Test authentication running in a thread"""

    def make_auth(self):
        from zmq.auth.asyncio import AsyncioAuthenticator

        return AsyncioAuthenticator(self.context)


async def test_ioloop_authenticator(context, event_loop, io_loop):
    from tornado.ioloop import IOLoop

    with warnings.catch_warnings():
        from zmq.auth.ioloop import IOLoopAuthenticator

    auth = IOLoopAuthenticator(context)
    assert auth.context is context

    loop = IOLoop(make_current=False)
    with pytest.warns(DeprecationWarning):
        auth = IOLoopAuthenticator(io_loop=loop)
