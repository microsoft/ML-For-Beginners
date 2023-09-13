import json

from tests_python.debugger_unittest import ReaderThread, IS_JYTHON
import pytest
import socket
from _pydev_bundle import pydev_localhost
from _pydevd_bundle.pydevd_comm import start_client

ABORT_CONNECTION = 'ABORT_CONNECTION'

pytestmark = pytest.mark.skipif(IS_JYTHON, reason='Getting the actual port does not work when the port is == 0.')


class _DummySocket(object):

    def __init__(self):
        self._sock_for_reader_thread = None
        self._sock_for_fixture_test = None
        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._localhost = host = pydev_localhost.get_localhost()
        self._socket_server.bind((host, 0))
        self._socket_server.listen(1)

    def recv(self, *args, **kwargs):
        if self._sock_for_reader_thread is None:
            sock, _addr = self._socket_server.accept()
            self._sock_for_reader_thread = sock
        return self._sock_for_reader_thread.recv(*args, **kwargs)

    def put(self, msg):
        if not isinstance(msg, bytes):
            msg = msg.encode('utf-8')

        if self._sock_for_fixture_test is None:
            self._sock_for_fixture_test = start_client(*self._socket_server.getsockname())

        self._sock_for_fixture_test.sendall(msg)

    def close(self):
        self._socket_server.close()

        if self._sock_for_fixture_test is not None:
            self._sock_for_fixture_test.close()

        if self._sock_for_reader_thread is not None:
            self._sock_for_reader_thread.close()


@pytest.yield_fixture
def _dummy_socket():
    sock = _DummySocket()
    yield sock
    sock.close()


def test_fixture_reader_thread1(_dummy_socket):
    sock = _dummy_socket

    reader_thread = ReaderThread(sock)
    reader_thread.start()

    json_part = json.dumps({'key': 'val'})
    json_part = json_part.replace(':', ':\n')
    msg = json_part

    msg = ('Content-Length: %s\r\n\r\n%s' % (len(msg), msg)).encode('utf-8')
    # Check that receiving 2 messages at a time we're able to properly deal
    # with each one.
    sock.put(msg + msg)

    assert reader_thread.get_next_message('check 1') == json_part
    assert reader_thread.get_next_message('check 2') == json_part


def test_fixture_reader_thread2(_dummy_socket):
    sock = _DummySocket()

    reader_thread = ReaderThread(sock)
    reader_thread.start()

    json_part = json.dumps({'key': 'val'})
    json_part = json_part.replace(':', ':\n')
    msg = json_part

    http = ('Content-Length: %s\r\n\r\n%s' % (len(msg), msg))
    sock.put('msg1\nmsg2\nmsg3\n' + http + http)

    assert reader_thread.get_next_message('check 1') == 'msg1'
    assert reader_thread.get_next_message('check 2') == 'msg2'
    assert reader_thread.get_next_message('check 3') == 'msg3'
    assert reader_thread.get_next_message('check 4') == json_part
    assert reader_thread.get_next_message('check 5') == json_part


def test_fixture_reader_thread3(_dummy_socket):
    sock = _DummySocket()

    reader_thread = ReaderThread(sock)
    reader_thread.start()

    msg = 'aaaaaaabbbbbbbccccccc'
    http = ('Content-Length: %s\r\n\r\n%s' % (len(msg), msg))
    http *= 2
    initial = http
    for i in range(1, len(http)):
        while http:
            sock.put(http[:i])
            http = http[i:]

        assert reader_thread.get_next_message('check 1: %s' % i) == msg
        assert reader_thread.get_next_message('check 2: %s' % i) == msg
        http = initial

