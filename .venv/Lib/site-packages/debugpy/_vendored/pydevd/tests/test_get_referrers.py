import sys
import threading
import time
import unittest
import pytest
from _pydevd_bundle import pydevd_referrers
from io import StringIO
from tests_python.debugger_unittest import IS_PYPY

try:
    import gc
    gc.get_referrers(unittest)
    has_referrers = True
except NotImplementedError:
    has_referrers = False


# Only do get referrers tests if it's actually available.
@pytest.mark.skipif(not has_referrers or IS_PYPY, reason='gc.get_referrers not implemented')
class Test(unittest.TestCase):

    def test_get_referrers1(self):

        container = []
        contained = [1, 2]
        container.append(0)
        container.append(contained)

        # Ok, we have the contained in this frame and inside the given list (which on turn is in this frame too).
        # we should skip temporary references inside the get_referrer_info.
        result = pydevd_referrers.get_referrer_info(contained)
        assert 'list[1]' in result
        pydevd_referrers.print_referrers(contained, stream=StringIO())

    def test_get_referrers2(self):

        class MyClass(object):

            def __init__(self):
                pass

        contained = [1, 2]
        obj = MyClass()
        obj.contained = contained
        del contained

        # Ok, we have the contained in this frame and inside the given list (which on turn is in this frame too).
        # we should skip temporary references inside the get_referrer_info.
        result = pydevd_referrers.get_referrer_info(obj.contained)
        assert 'found_as="contained"' in result
        assert 'MyClass' in result

    def test_get_referrers3(self):

        class MyClass(object):

            def __init__(self):
                pass

        contained = [1, 2]
        obj = MyClass()
        obj.contained = contained
        del contained

        # Ok, we have the contained in this frame and inside the given list (which on turn is in this frame too).
        # we should skip temporary references inside the get_referrer_info.
        result = pydevd_referrers.get_referrer_info(obj.contained)
        assert 'found_as="contained"' in result
        assert 'MyClass' in result

    def test_get_referrers4(self):

        class MyClass(object):

            def __init__(self):
                pass

        obj = MyClass()
        obj.me = obj

        # Let's see if we detect the cycle...
        result = pydevd_referrers.get_referrer_info(obj)
        assert 'found_as="me"' in result  # Cyclic ref

    def test_get_referrers5(self):
        container = dict(a=[1])

        # Let's see if we detect the cycle...
        result = pydevd_referrers.get_referrer_info(container['a'])
        assert 'test_get_referrers5' not in result  # I.e.: NOT in the current method
        assert 'found_as="a"' in result
        assert 'dict' in result
        assert str(id(container)) in result

    def test_get_referrers6(self):
        import sys
        container = dict(a=[1])

        def should_appear(obj):
            # Let's see if we detect the cycle...
            return pydevd_referrers.get_referrer_info(obj)

        result = should_appear(container['a'])
        if sys.version_info[:2] >= (3, 7):
            # In Python 3.7 the frame is not appearing in gc.get_referrers.
            assert 'should_appear' not in result
        else:
            assert 'should_appear' in result

    def test_get_referrers7(self):

        class MyThread(threading.Thread):

            def run(self):
                # Note: we do that because if we do
                self.frame = sys._getframe()

        t = MyThread()
        t.start()
        while not hasattr(t, 'frame'):
            time.sleep(0.01)

        result = pydevd_referrers.get_referrer_info(t.frame)
        assert 'MyThread' in result

