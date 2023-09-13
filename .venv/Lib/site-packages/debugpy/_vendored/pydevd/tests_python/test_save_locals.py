import inspect
import sys
import unittest

from _pydevd_bundle.pydevd_save_locals import save_locals
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON
import pytest


def use_save_locals(name, value):
    """
    Attempt to set the local of the given name to value, using locals_to_fast.
    """
    frame = inspect.currentframe().f_back
    locals_dict = frame.f_locals
    locals_dict[name] = value

    save_locals(frame)


def check_method(fn):
    """
    A harness for testing methods that attempt to modify the values of locals on the stack.
    """
    x = 1

    # The method 'fn' should attempt to set x = 2 in the current frame.
    fn('x', 2)

    return x



@pytest.mark.skipif(IS_JYTHON or IS_IRONPYTHON, reason='CPython/pypy only')
class TestSetLocals(unittest.TestCase):
    """
    Test setting locals in one function from another function using several approaches.
    """

    def test_set_locals_using_save_locals(self):
        x = check_method(use_save_locals)
        self.assertEqual(x, 2)  # Expected to succeed


    def test_frame_simple_change(self):
        frame = sys._getframe()
        a = 20
        frame.f_locals['a'] = 50
        save_locals(frame)
        self.assertEqual(50, a)


    def test_frame_co_freevars(self):

        outer_var = 20

        def func():
            frame = sys._getframe()
            frame.f_locals['outer_var'] = 50
            save_locals(frame)
            self.assertEqual(50, outer_var)

        func()

    def test_frame_co_cellvars(self):

        def check_co_vars(a):
            frame = sys._getframe()
            def function2():
                print(a)

            assert 'a' in frame.f_code.co_cellvars
            frame = sys._getframe()
            frame.f_locals['a'] = 50
            save_locals(frame)
            self.assertEqual(50, a)

        check_co_vars(1)


    def test_frame_change_in_inner_frame(self):
        def change(f):
            self.assertTrue(f is not sys._getframe())
            f.f_locals['a']= 50
            save_locals(f)


        frame = sys._getframe()
        a = 20
        change(frame)
        self.assertEqual(50, a)


if __name__ == '__main__':
    suite = unittest.TestSuite()
#    suite.addTest(TestSetLocals('test_set_locals_using_dict'))
#    #suite.addTest(Test('testCase10a'))
#    unittest.TextTestRunner(verbosity=3).run(suite)

    suite = unittest.makeSuite(TestSetLocals)
    unittest.TextTestRunner(verbosity=3).run(suite)
