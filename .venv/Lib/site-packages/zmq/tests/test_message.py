# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import copy
import gc
import sys

try:
    from sys import getrefcount
except ImportError:
    grc = None
else:
    grc = getrefcount

import time

import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy

# some useful constants:

x = b'x'

if grc:
    rc0 = grc(x)
    v = memoryview(x)
    view_rc = grc(x) - rc0


def await_gc(obj, rc):
    """wait for refcount on an object to drop to an expected value

    Necessary because of the zero-copy gc thread,
    which can take some time to receive its DECREF message.
    """
    # count refs for this function
    if sys.version_info < (3, 11):
        my_refs = 2
    else:
        my_refs = 1
    for i in range(50):
        # rc + 2 because of the refs in this function
        if grc(obj) <= rc + my_refs:
            return
        time.sleep(0.05)


class TestFrame(BaseZMQTestCase):
    def tearDown(self):
        super().tearDown()
        for i in range(3):
            gc.collect()

    @skip_pypy
    def test_above_30(self):
        """Message above 30 bytes are never copied by 0MQ."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i) * x
            rc = grc(s)
            m = zmq.Frame(s, copy=False)
            assert grc(s) == rc + 2
            del m
            await_gc(s, rc)
            assert grc(s) == rc
            del s

    def test_str(self):
        """Test the str representations of the Frames."""
        for i in range(16):
            s = (2**i) * x
            m = zmq.Frame(s)
            m_str = str(m)
            m_str_b = m_str.encode()
            assert s == m_str_b

    def test_bytes(self):
        """Test the Frame.bytes property."""
        for i in range(1, 16):
            s = (2**i) * x
            m = zmq.Frame(s)
            b = m.bytes
            assert s == m.bytes
            if not PYPY:
                # check that it copies
                assert b is not s
            # check that it copies only once
            assert b is m.bytes

    def test_unicode(self):
        """Test the unicode representations of the Frames."""
        s = 'asdf'
        self.assertRaises(TypeError, zmq.Frame, s)
        for i in range(16):
            s = (2**i) * '§'
            m = zmq.Frame(s.encode('utf8'))
            assert s == m.bytes.decode('utf8')

    def test_len(self):
        """Test the len of the Frames."""
        for i in range(16):
            s = (2**i) * x
            m = zmq.Frame(s)
            assert len(s) == len(m)

    @skip_pypy
    def test_lifecycle1(self):
        """Run through a ref counting cycle with a copy."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i) * x
            rc = rc_0 = grc(s)
            m = zmq.Frame(s, copy=False)
            rc += 2
            assert grc(s) == rc
            m2 = copy.copy(m)
            rc += 1
            assert grc(s) == rc
            # no increase in refcount for accessing buffer
            # which references m2 directly
            buf = m2.buffer
            assert grc(s) == rc

            assert s == str(m).encode()
            assert s == bytes(m2)
            assert s == m.bytes
            assert s == bytes(buf)
            # assert s is str(m)
            # assert s is str(m2)
            del m2
            assert grc(s) == rc
            # buf holds direct reference to m2 which holds
            del buf
            rc -= 1
            assert grc(s) == rc
            del m
            rc -= 2
            await_gc(s, rc)
            assert grc(s) == rc
            assert rc == rc_0
            del s

    @skip_pypy
    def test_lifecycle2(self):
        """Run through a different ref counting cycle with a copy."""
        for i in range(5, 16):  # 32, 64,..., 65536
            s = (2**i) * x
            rc = rc_0 = grc(s)
            m = zmq.Frame(s, copy=False)
            rc += 2
            assert grc(s) == rc
            m2 = copy.copy(m)
            rc += 1
            assert grc(s) == rc
            # no increase in refcount for accessing buffer
            # which references m directly
            buf = m.buffer
            assert grc(s) == rc
            assert s == str(m).encode()
            assert s == bytes(m2)
            assert s == m2.bytes
            assert s == m.bytes
            assert s == bytes(buf)
            # assert s is str(m)
            # assert s is str(m2)
            del buf
            assert grc(s) == rc
            del m
            rc -= 1
            assert grc(s) == rc
            del m2
            rc -= 2
            await_gc(s, rc)
            assert grc(s) == rc
            assert rc == rc_0
            del s

    def test_tracker(self):
        m = zmq.Frame(b'asdf', copy=False, track=True)
        assert not m.tracker.done
        pm = zmq.MessageTracker(m)
        assert not pm.done
        del m
        for i in range(3):
            gc.collect()
        for i in range(10):
            if pm.done:
                break
            time.sleep(0.1)
        assert pm.done

    def test_no_tracker(self):
        m = zmq.Frame(b'asdf', track=False)
        assert m.tracker == None
        m2 = copy.copy(m)
        assert m2.tracker == None
        self.assertRaises(ValueError, zmq.MessageTracker, m)

    def test_multi_tracker(self):
        m = zmq.Frame(b'asdf', copy=False, track=True)
        m2 = zmq.Frame(b'whoda', copy=False, track=True)
        mt = zmq.MessageTracker(m, m2)
        assert not m.tracker.done
        assert not mt.done
        self.assertRaises(zmq.NotDone, mt.wait, 0.1)
        del m
        for i in range(3):
            gc.collect()
        self.assertRaises(zmq.NotDone, mt.wait, 0.1)
        assert not mt.done
        del m2
        for i in range(3):
            gc.collect()
        assert mt.wait(0.1) is None
        assert mt.done

    def test_buffer_in(self):
        """test using a buffer as input"""
        ins = "§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√".encode()
        zmq.Frame(memoryview(ins))

    def test_bad_buffer_in(self):
        """test using a bad object"""
        self.assertRaises(TypeError, zmq.Frame, 5)
        self.assertRaises(TypeError, zmq.Frame, object())

    def test_buffer_out(self):
        """receiving buffered output"""
        ins = "§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√".encode()
        m = zmq.Frame(ins)
        outb = m.buffer
        assert isinstance(outb, memoryview)
        assert outb is m.buffer
        assert m.buffer is m.buffer

    def test_memoryview_shape(self):
        """memoryview shape info"""
        data = "§§¶•ªº˜µ¬˚…∆˙åß∂©œ∑´†≈ç√".encode()
        n = len(data)
        f = zmq.Frame(data)
        view1 = f.buffer
        assert view1.ndim == 1
        assert view1.shape == (n,)
        assert view1.tobytes() == data
        view2 = memoryview(f)
        assert view2.ndim == 1
        assert view2.shape == (n,)
        assert view2.tobytes() == data

    def test_multisend(self):
        """ensure that a message remains intact after multiple sends"""
        a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        s = b"message"
        m = zmq.Frame(s)
        assert s == m.bytes

        a.send(m, copy=False)
        time.sleep(0.1)
        assert s == m.bytes
        a.send(m, copy=False)
        time.sleep(0.1)
        assert s == m.bytes
        a.send(m, copy=True)
        time.sleep(0.1)
        assert s == m.bytes
        a.send(m, copy=True)
        time.sleep(0.1)
        assert s == m.bytes
        for i in range(4):
            r = b.recv()
            assert s == r
        assert s == m.bytes

    def test_memoryview(self):
        """test messages from memoryview"""
        s = b'carrotjuice'
        memoryview(s)
        m = zmq.Frame(s)
        buf = m.buffer
        s2 = buf.tobytes()
        assert s2 == s
        assert m.bytes == s

    def test_noncopying_recv(self):
        """check for clobbering message buffers"""
        null = b'\0' * 64
        sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        for i in range(32):
            # try a few times
            sb.send(null, copy=False)
            m = sa.recv(copy=False)
            mb = m.bytes
            # buf = memoryview(m)
            buf = m.buffer
            del m
            for i in range(5):
                ff = b'\xff' * (40 + i * 10)
                sb.send(ff, copy=False)
                m2 = sa.recv(copy=False)
                b = buf.tobytes()
                assert b == null
                assert mb == null
                assert m2.bytes == ff
                assert type(m2.bytes) is bytes

    def test_noncopying_memoryview(self):
        """test non-copying memmoryview messages"""
        null = b'\0' * 64
        sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        for i in range(32):
            # try a few times
            sb.send(memoryview(null), copy=False)
            m = sa.recv(copy=False)
            buf = memoryview(m)
            for i in range(5):
                ff = b'\xff' * (40 + i * 10)
                sb.send(memoryview(ff), copy=False)
                m2 = sa.recv(copy=False)
                buf2 = memoryview(m2)
                assert buf.tobytes() == null
                assert not buf.readonly
                assert buf2.tobytes() == ff
                assert not buf2.readonly
                assert type(buf) is memoryview

    def test_buffer_numpy(self):
        """test non-copying numpy array messages"""
        try:
            import numpy
            from numpy.testing import assert_array_equal
        except ImportError:
            raise SkipTest("requires numpy")
        rand = numpy.random.randint
        shapes = [rand(2, 5) for i in range(5)]
        a, b = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        dtypes = [int, float, '>i4', 'B']
        for i in range(1, len(shapes) + 1):
            shape = shapes[:i]
            for dt in dtypes:
                A = numpy.empty(shape, dtype=dt)
                a.send(A, copy=False)
                msg = b.recv(copy=False)

                B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
                assert_array_equal(A, B)

            A = numpy.empty(shape, dtype=[('a', int), ('b', float), ('c', 'a32')])
            A['a'] = 1024
            A['b'] = 1e9
            A['c'] = 'hello there'
            a.send(A, copy=False)
            msg = b.recv(copy=False)

            B = numpy.frombuffer(msg, A.dtype).reshape(A.shape)
            assert_array_equal(A, B)

    @skip_pypy
    def test_frame_more(self):
        """test Frame.more attribute"""
        frame = zmq.Frame(b"hello")
        assert not frame.more
        sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        sa.send_multipart([b'hi', b'there'])
        frame = self.recv(sb, copy=False)
        assert frame.more
        if zmq.zmq_version_info()[0] >= 3 and not PYPY:
            assert frame.get(zmq.MORE)
        frame = self.recv(sb, copy=False)
        assert not frame.more
        if zmq.zmq_version_info()[0] >= 3 and not PYPY:
            assert not frame.get(zmq.MORE)
