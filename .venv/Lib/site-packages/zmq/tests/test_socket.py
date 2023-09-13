# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock

import pytest
from pytest import mark

import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy

pypy = platform.python_implementation().lower() == 'pypy'
windows = platform.platform().lower().startswith('windows')
on_ci = bool(os.environ.get('CI'))

# polling on windows is slow
POLL_TIMEOUT = 1000 if windows else 100


class TestSocket(BaseZMQTestCase):
    def test_create(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        # Superluminal protocol not yet implemented
        self.assertRaisesErrno(zmq.EPROTONOSUPPORT, s.bind, 'ftl://a')
        self.assertRaisesErrno(zmq.EPROTONOSUPPORT, s.connect, 'ftl://a')
        self.assertRaisesErrno(zmq.EINVAL, s.bind, 'tcp://')
        s.close()
        ctx.term()

    def test_context_manager(self):
        url = 'inproc://a'
        with self.Context() as ctx:
            with ctx.socket(zmq.PUSH) as a:
                a.bind(url)
                with ctx.socket(zmq.PULL) as b:
                    b.connect(url)
                    msg = b'hi'
                    a.send(msg)
                    rcvd = self.recv(b)
                    assert rcvd == msg
                assert b.closed == True
            assert a.closed == True
        assert ctx.closed == True

    def test_connectbind_context_managers(self):
        url = 'inproc://a'
        msg = b'hi'
        with self.Context() as ctx:
            # Test connect() context manager
            with ctx.socket(zmq.PUSH) as a, ctx.socket(zmq.PULL) as b:
                a.bind(url)
                connect_context = b.connect(url)
                assert f'connect={url!r}' in repr(connect_context)
                with connect_context:
                    a.send(msg)
                    rcvd = self.recv(b)
                    assert rcvd == msg
                # b should now be disconnected, so sending and receiving don't work
                with pytest.raises(zmq.Again):
                    a.send(msg, flags=zmq.DONTWAIT)
                with pytest.raises(zmq.Again):
                    b.recv(flags=zmq.DONTWAIT)
                a.unbind(url)
            # Test bind() context manager
            with ctx.socket(zmq.PUSH) as a, ctx.socket(zmq.PULL) as b:
                # unbind() just stops accepting of new connections, so we have to disconnect to test that
                # unbind happened.
                bind_context = a.bind(url)
                assert f'bind={url!r}' in repr(bind_context)
                with bind_context:
                    b.connect(url)
                    a.send(msg)
                    rcvd = self.recv(b)
                    assert rcvd == msg
                    b.disconnect(url)
                b.connect(url)
                # Since a is unbound from url, b is not connected to anything
                with pytest.raises(zmq.Again):
                    a.send(msg, flags=zmq.DONTWAIT)
                with pytest.raises(zmq.Again):
                    b.recv(flags=zmq.DONTWAIT)

    _repr_cls = "zmq.Socket"

    def test_repr(self):
        with self.context.socket(zmq.PUSH) as s:
            assert f'{self._repr_cls}(zmq.PUSH)' in repr(s)
            assert 'closed' not in repr(s)
        assert f'{self._repr_cls}(zmq.PUSH)' in repr(s)
        assert 'closed' in repr(s)

    def test_dir(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        assert 'send' in dir(s)
        assert 'IDENTITY' in dir(s)
        assert 'AFFINITY' in dir(s)
        assert 'FD' in dir(s)
        s.close()
        ctx.term()

    @mark.skipif(mock is None, reason="requires unittest.mock")
    def test_mockable(self):
        s = self.socket(zmq.SUB)
        m = mock.Mock(spec=s)
        s.close()

    def test_bind_unicode(self):
        s = self.socket(zmq.PUB)
        p = s.bind_to_random_port("tcp://*")

    def test_connect_unicode(self):
        s = self.socket(zmq.PUB)
        s.connect("tcp://127.0.0.1:5555")

    def test_bind_to_random_port(self):
        # Check that bind_to_random_port do not hide useful exception
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        # Invalid format
        try:
            s.bind_to_random_port('tcp:*')
        except zmq.ZMQError as e:
            assert e.errno == zmq.EINVAL
        # Invalid protocol
        try:
            s.bind_to_random_port('rand://*')
        except zmq.ZMQError as e:
            assert e.errno == zmq.EPROTONOSUPPORT

        s.close()
        ctx.term()

    def test_bind_connect_addr_error(self):
        with self.socket(zmq.PUSH) as s:
            url = "tcp://1.2.3.4.5:1234567"
            with pytest.raises(zmq.ZMQError) as exc:
                s.bind(url)
            assert url in str(exc.value)

            url = "noproc://no/such/file"
            with pytest.raises(zmq.ZMQError) as exc:
                s.connect(url)
            assert url in str(exc.value)

    def test_identity(self):
        s = self.context.socket(zmq.PULL)
        self.sockets.append(s)
        ident = b'identity\0\0'
        s.identity = ident
        assert s.get(zmq.IDENTITY) == ident

    def test_unicode_sockopts(self):
        """test setting/getting sockopts with unicode strings"""
        topic = "tést"
        p, s = self.create_bound_pair(zmq.PUB, zmq.SUB)
        assert s.send_unicode == s.send_unicode
        assert p.recv_unicode == p.recv_unicode
        self.assertRaises(TypeError, s.setsockopt, zmq.SUBSCRIBE, topic)
        self.assertRaises(TypeError, s.setsockopt, zmq.IDENTITY, topic)
        s.setsockopt_unicode(zmq.IDENTITY, topic, 'utf16')
        self.assertRaises(TypeError, s.setsockopt, zmq.AFFINITY, topic)
        s.setsockopt_unicode(zmq.SUBSCRIBE, topic)
        self.assertRaises(TypeError, s.getsockopt_unicode, zmq.AFFINITY)
        self.assertRaisesErrno(zmq.EINVAL, s.getsockopt_unicode, zmq.SUBSCRIBE)

        identb = s.getsockopt(zmq.IDENTITY)
        identu = identb.decode('utf16')
        identu2 = s.getsockopt_unicode(zmq.IDENTITY, 'utf16')
        assert identu == identu2
        time.sleep(0.1)  # wait for connection/subscription
        p.send_unicode(topic, zmq.SNDMORE)
        p.send_unicode(topic * 2, encoding='latin-1')
        assert topic == s.recv_unicode()
        assert topic * 2 == s.recv_unicode(encoding='latin-1')

    def test_int_sockopts(self):
        "test integer sockopts"
        v = zmq.zmq_version_info()
        if v < (3, 0):
            default_hwm = 0
        else:
            default_hwm = 1000
        p, s = self.create_bound_pair(zmq.PUB, zmq.SUB)
        p.setsockopt(zmq.LINGER, 0)
        assert p.getsockopt(zmq.LINGER) == 0
        p.setsockopt(zmq.LINGER, -1)
        assert p.getsockopt(zmq.LINGER) == -1
        assert p.hwm == default_hwm
        p.hwm = 11
        assert p.hwm == 11
        # p.setsockopt(zmq.EVENTS, zmq.POLLIN)
        assert p.getsockopt(zmq.EVENTS) == zmq.POLLOUT
        self.assertRaisesErrno(zmq.EINVAL, p.setsockopt, zmq.EVENTS, 2**7 - 1)
        assert p.getsockopt(zmq.TYPE) == p.socket_type
        assert p.getsockopt(zmq.TYPE) == zmq.PUB
        assert s.getsockopt(zmq.TYPE) == s.socket_type
        assert s.getsockopt(zmq.TYPE) == zmq.SUB

        # check for overflow / wrong type:
        errors = []
        backref = {}
        constants = zmq.constants
        for name in constants.__all__:
            value = getattr(constants, name)
            if isinstance(value, int):
                backref[value] = name
        for opt in zmq.constants.SocketOption:
            if opt._opt_type not in {
                zmq.constants._OptType.int,
                zmq.constants._OptType.int64,
            }:
                continue
            if opt.name.startswith(
                (
                    'HWM',
                    'ROUTER',
                    'XPUB',
                    'TCP',
                    'FAIL',
                    'REQ_',
                    'CURVE_',
                    'PROBE_ROUTER',
                    'IPC_FILTER',
                    'GSSAPI',
                    'STREAM_',
                    'VMCI_BUFFER_SIZE',
                    'VMCI_BUFFER_MIN_SIZE',
                    'VMCI_BUFFER_MAX_SIZE',
                    'VMCI_CONNECT_TIMEOUT',
                    'BLOCKY',
                    'IN_BATCH_SIZE',
                    'OUT_BATCH_SIZE',
                    'WSS_TRUST_SYSTEM',
                    'ONLY_FIRST_SUBSCRIBE',
                    'PRIORITY',
                    'RECONNECT_STOP',
                )
            ):
                # some sockopts are write-only
                continue
            try:
                n = p.getsockopt(opt)
            except zmq.ZMQError as e:
                errors.append(f"getsockopt({opt!r}) raised {e}.")
            else:
                if n > 2**31:
                    errors.append(
                        f"getsockopt({opt!r}) returned a ridiculous value."
                        " It is probably the wrong type."
                    )
        if errors:
            self.fail('\n'.join([''] + errors))

    def test_bad_sockopts(self):
        """Test that appropriate errors are raised on bad socket options"""
        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        s.setsockopt(zmq.LINGER, 0)
        # unrecognized int sockopts pass through to libzmq, and should raise EINVAL
        self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, 9999, 5)
        self.assertRaisesErrno(zmq.EINVAL, s.getsockopt, 9999)
        # but only int sockopts are allowed through this way, otherwise raise a TypeError
        self.assertRaises(TypeError, s.setsockopt, 9999, b"5")
        # some sockopts are valid in general, but not on every socket:
        self.assertRaisesErrno(zmq.EINVAL, s.setsockopt, zmq.SUBSCRIBE, b'hi')

    def test_sockopt_roundtrip(self):
        "test set/getsockopt roundtrip."
        p = self.context.socket(zmq.PUB)
        self.sockets.append(p)
        p.setsockopt(zmq.LINGER, 11)
        assert p.getsockopt(zmq.LINGER) == 11

    def test_send_unicode(self):
        "test sending unicode objects"
        a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        self.sockets.extend([a, b])
        u = "çπ§"
        self.assertRaises(TypeError, a.send, u, copy=False)
        self.assertRaises(TypeError, a.send, u, copy=True)
        a.send_unicode(u)
        s = b.recv()
        assert s == u.encode('utf8')
        assert s.decode('utf8') == u
        a.send_unicode(u, encoding='utf16')
        s = b.recv_unicode(encoding='utf16')
        assert s == u

    def test_send_multipart_check_type(self):
        "check type on all frames in send_multipart"
        a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        self.sockets.extend([a, b])
        self.assertRaises(TypeError, a.send_multipart, [b'a', 5])
        a.send_multipart([b'b'])
        rcvd = self.recv_multipart(b)
        assert rcvd == [b'b']

    @skip_pypy
    def test_tracker(self):
        "test the MessageTracker object for tracking when zmq is done with a buffer"
        addr = 'tcp://127.0.0.1'
        # get a port:
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        iface = "%s:%i" % (addr, port)
        sock.close()
        time.sleep(0.1)

        a = self.context.socket(zmq.PUSH)
        b = self.context.socket(zmq.PULL)
        self.sockets.extend([a, b])
        a.connect(iface)
        time.sleep(0.1)
        p1 = a.send(b'something', copy=False, track=True)
        assert isinstance(p1, zmq.MessageTracker)
        assert p1 is zmq._FINISHED_TRACKER
        # small message, should start done
        assert p1.done

        # disable zero-copy threshold
        a.copy_threshold = 0

        p2 = a.send_multipart([b'something', b'else'], copy=False, track=True)
        assert isinstance(p2, zmq.MessageTracker)
        assert not p2.done

        b.bind(iface)
        msg = self.recv_multipart(b)
        for i in range(10):
            if p1.done:
                break
            time.sleep(0.1)
        assert p1.done == True
        assert msg == [b'something']
        msg = self.recv_multipart(b)
        for i in range(10):
            if p2.done:
                break
            time.sleep(0.1)
        assert p2.done == True
        assert msg == [b'something', b'else']
        m = zmq.Frame(b"again", copy=False, track=True)
        assert m.tracker.done == False
        p1 = a.send(m, copy=False)
        p2 = a.send(m, copy=False)
        assert m.tracker.done == False
        assert p1.done == False
        assert p2.done == False
        msg = self.recv_multipart(b)
        assert m.tracker.done == False
        assert msg == [b'again']
        msg = self.recv_multipart(b)
        assert m.tracker.done == False
        assert msg == [b'again']
        assert p1.done == False
        assert p2.done == False
        m.tracker
        del m
        for i in range(10):
            if p1.done:
                break
            time.sleep(0.1)
        assert p1.done == True
        assert p2.done == True
        m = zmq.Frame(b'something', track=False)
        self.assertRaises(ValueError, a.send, m, copy=False, track=True)

    def test_close(self):
        ctx = self.Context()
        s = ctx.socket(zmq.PUB)
        s.close()
        self.assertRaisesErrno(zmq.ENOTSOCK, s.bind, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.connect, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.setsockopt, zmq.SUBSCRIBE, b'')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.send, b'asdf')
        self.assertRaisesErrno(zmq.ENOTSOCK, s.recv)
        ctx.term()

    def test_attr(self):
        """set setting/getting sockopts as attributes"""
        s = self.context.socket(zmq.DEALER)
        self.sockets.append(s)
        linger = 10
        s.linger = linger
        assert linger == s.linger
        assert linger == s.getsockopt(zmq.LINGER)
        assert s.fd == s.getsockopt(zmq.FD)

    def test_bad_attr(self):
        s = self.context.socket(zmq.DEALER)
        self.sockets.append(s)
        try:
            s.apple = 'foo'
        except AttributeError:
            pass
        else:
            self.fail("bad setattr should have raised AttributeError")
        try:
            s.apple
        except AttributeError:
            pass
        else:
            self.fail("bad getattr should have raised AttributeError")

    def test_subclass(self):
        """subclasses can assign attributes"""

        class S(zmq.Socket):
            a = None

            def __init__(self, *a, **kw):
                self.a = -1
                super().__init__(*a, **kw)

        s = S(self.context, zmq.REP)
        self.sockets.append(s)
        assert s.a == -1
        s.a = 1
        assert s.a == 1
        a = s.a
        assert a == 1

    def test_recv_multipart(self):
        a, b = self.create_bound_pair()
        msg = b'hi'
        for i in range(3):
            a.send(msg)
        time.sleep(0.1)
        for i in range(3):
            assert self.recv_multipart(b) == [msg]

    def test_close_after_destroy(self):
        """s.close() after ctx.destroy() should be fine"""
        ctx = self.Context()
        s = ctx.socket(zmq.REP)
        ctx.destroy()
        # reaper is not instantaneous
        time.sleep(1e-2)
        s.close()
        assert s.closed

    def test_poll(self):
        a, b = self.create_bound_pair()
        time.time()
        evt = a.poll(POLL_TIMEOUT)
        assert evt == 0
        evt = a.poll(POLL_TIMEOUT, zmq.POLLOUT)
        assert evt == zmq.POLLOUT
        msg = b'hi'
        a.send(msg)
        evt = b.poll(POLL_TIMEOUT)
        assert evt == zmq.POLLIN
        msg2 = self.recv(b)
        evt = b.poll(POLL_TIMEOUT)
        assert evt == 0
        assert msg2 == msg

    def test_ipc_path_max_length(self):
        """IPC_PATH_MAX_LEN is a sensible value"""
        if zmq.IPC_PATH_MAX_LEN == 0:
            raise SkipTest("IPC_PATH_MAX_LEN undefined")

        msg = "Surprising value for IPC_PATH_MAX_LEN: %s" % zmq.IPC_PATH_MAX_LEN
        assert zmq.IPC_PATH_MAX_LEN > 30, msg
        assert zmq.IPC_PATH_MAX_LEN < 1025, msg

    def test_ipc_path_max_length_msg(self):
        if zmq.IPC_PATH_MAX_LEN == 0:
            raise SkipTest("IPC_PATH_MAX_LEN undefined")

        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        try:
            s.bind('ipc://{}'.format('a' * (zmq.IPC_PATH_MAX_LEN + 1)))
        except zmq.ZMQError as e:
            assert str(zmq.IPC_PATH_MAX_LEN) in e.strerror

    @mark.skipif(windows, reason="ipc not supported on Windows.")
    def test_ipc_path_no_such_file_or_directory_message(self):
        """Display the ipc path in case of an ENOENT exception"""
        s = self.context.socket(zmq.PUB)
        self.sockets.append(s)
        invalid_path = '/foo/bar'
        with pytest.raises(zmq.ZMQError) as error:
            s.bind(f'ipc://{invalid_path}')
        assert error.value.errno == errno.ENOENT
        error_message = str(error.value)
        assert invalid_path in error_message
        assert "no such file or directory" in error_message.lower()

    def test_hwm(self):
        zmq3 = zmq.zmq_version_info()[0] >= 3
        for stype in (zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER):
            s = self.context.socket(stype)
            s.hwm = 100
            assert s.hwm == 100
            if zmq3:
                try:
                    assert s.sndhwm == 100
                except AttributeError:
                    pass
                try:
                    assert s.rcvhwm == 100
                except AttributeError:
                    pass
            s.close()

    def test_copy(self):
        s = self.socket(zmq.PUB)
        scopy = copy.copy(s)
        sdcopy = copy.deepcopy(s)
        assert scopy._shadow
        assert sdcopy._shadow
        assert s.underlying == scopy.underlying
        assert s.underlying == sdcopy.underlying
        s.close()

    def test_send_buffer(self):
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        for buffer_type in (memoryview, bytearray):
            rawbytes = str(buffer_type).encode('ascii')
            msg = buffer_type(rawbytes)
            a.send(msg)
            recvd = b.recv()
            assert recvd == rawbytes

    def test_shadow(self):
        p = self.socket(zmq.PUSH)
        p.bind("tcp://127.0.0.1:5555")
        p2 = zmq.Socket.shadow(p.underlying)
        assert p.underlying == p2.underlying
        s = self.socket(zmq.PULL)
        s2 = zmq.Socket.shadow(s)
        assert s2._shadow_obj is s
        assert s.underlying != p.underlying
        assert s2.underlying == s.underlying
        s3 = zmq.Socket(s)
        assert s3._shadow_obj is s
        assert s3.underlying == s.underlying
        s2.connect("tcp://127.0.0.1:5555")
        sent = b'hi'
        p2.send(sent)
        rcvd = self.recv(s2)
        assert rcvd == sent

    def test_shadow_pyczmq(self):
        try:
            from pyczmq import zctx, zsocket
        except Exception:
            raise SkipTest("Requires pyczmq")

        ctx = zctx.new()
        ca = zsocket.new(ctx, zmq.PUSH)
        cb = zsocket.new(ctx, zmq.PULL)
        a = zmq.Socket.shadow(ca)
        b = zmq.Socket.shadow(cb)
        a.bind("inproc://a")
        b.connect("inproc://a")
        a.send(b'hi')
        rcvd = self.recv(b)
        assert rcvd == b'hi'

    def test_subscribe_method(self):
        pub, sub = self.create_bound_pair(zmq.PUB, zmq.SUB)
        sub.subscribe('prefix')
        sub.subscribe = 'c'
        p = zmq.Poller()
        p.register(sub, zmq.POLLIN)
        # wait for subscription handshake
        for i in range(100):
            pub.send(b'canary')
            events = p.poll(250)
            if events:
                break
        self.recv(sub)
        pub.send(b'prefixmessage')
        msg = self.recv(sub)
        assert msg == b'prefixmessage'
        sub.unsubscribe('prefix')
        pub.send(b'prefixmessage')
        events = p.poll(1000)
        assert events == []

    # CI often can't handle how much memory PyPy uses on this test
    @mark.skipif(
        (pypy and on_ci) or (sys.maxsize < 2**32) or (windows),
        reason="only run on 64b and not on CI.",
    )
    @mark.large
    def test_large_send(self):
        c = os.urandom(1)
        N = 2**31 + 1
        try:
            buf = c * N
        except MemoryError as e:
            raise SkipTest("Not enough memory: %s" % e)
        a, b = self.create_bound_pair()
        try:
            a.send(buf, copy=False)
            rcvd = b.recv(copy=False)
        except MemoryError as e:
            raise SkipTest("Not enough memory: %s" % e)
        # sample the front and back of the received message
        # without checking the whole content
        byte = ord(c)
        view = memoryview(rcvd)
        assert len(view) == N
        assert view[0] == byte
        assert view[-1] == byte

    def test_custom_serialize(self):
        a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)

        def serialize(msg):
            frames = []
            frames.extend(msg.get('identities', []))
            content = json.dumps(msg['content']).encode('utf8')
            frames.append(content)
            return frames

        def deserialize(frames):
            identities = frames[:-1]
            content = json.loads(frames[-1].decode('utf8'))
            return {
                'identities': identities,
                'content': content,
            }

        msg = {
            'content': {
                'a': 5,
                'b': 'bee',
            }
        }
        a.send_serialized(msg, serialize)
        recvd = b.recv_serialized(deserialize)
        assert recvd['content'] == msg['content']
        assert recvd['identities']
        # bounce back, tests identities
        b.send_serialized(recvd, serialize)
        r2 = a.recv_serialized(deserialize)
        assert r2['content'] == msg['content']
        assert not r2['identities']


if have_gevent and not windows:
    import gevent

    class TestSocketGreen(GreenTest, TestSocket):
        test_bad_attr = GreenTest.skip_green
        test_close_after_destroy = GreenTest.skip_green
        _repr_cls = "zmq.green.Socket"

        def test_timeout(self):
            a, b = self.create_bound_pair()
            g = gevent.spawn_later(0.5, lambda: a.send(b'hi'))
            timeout = gevent.Timeout(0.1)
            timeout.start()
            self.assertRaises(gevent.Timeout, b.recv)
            g.kill()

        @mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason="requires RCVTIMEO")
        def test_warn_set_timeo(self):
            s = self.context.socket(zmq.REQ)
            with warnings.catch_warnings(record=True) as w:
                s.rcvtimeo = 5
            s.close()
            assert len(w) == 1
            assert w[0].category == UserWarning

        @mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason="requires SNDTIMEO")
        def test_warn_get_timeo(self):
            s = self.context.socket(zmq.REQ)
            with warnings.catch_warnings(record=True) as w:
                s.sndtimeo
            s.close()
            assert len(w) == 1
            assert w[0].category == UserWarning
