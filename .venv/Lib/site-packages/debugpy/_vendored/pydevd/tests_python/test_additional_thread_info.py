import sys
import os
from _pydev_bundle import pydev_monkey
sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])

import unittest

try:
    import thread
except:
    import _thread as thread  # @UnresolvedImport


#=======================================================================================================================
# TestCase
#=======================================================================================================================
class TestCase(unittest.TestCase):

    def test_start_new_thread(self):
        pydev_monkey.patch_thread_modules()
        try:
            found = {}

            def function(a, b, *args, **kwargs):
                found['a'] = a
                found['b'] = b
                found['args'] = args
                found['kwargs'] = kwargs

            thread.start_new_thread(function, (1, 2, 3, 4), {'d':1, 'e':2})
            import time
            for _i in range(20):
                if len(found) == 4:
                    break
                time.sleep(.1)
            else:
                raise AssertionError('Could not get to condition before 2 seconds')

            self.assertEqual({'a': 1, 'b': 2, 'args': (3, 4), 'kwargs': {'e': 2, 'd': 1}}, found)
        finally:
            pydev_monkey.undo_patch_thread_modules()

    def test_start_new_thread2(self):
        pydev_monkey.patch_thread_modules()
        try:
            found = {}

            class F(object):
                start_new_thread = thread.start_new_thread

                def start_it(self):
                    try:
                        self.start_new_thread(self.function, (1, 2, 3, 4), {'d':1, 'e':2})
                    except:
                        import traceback;traceback.print_exc()

                def function(self, a, b, *args, **kwargs):
                    found['a'] = a
                    found['b'] = b
                    found['args'] = args
                    found['kwargs'] = kwargs

            f = F()
            f.start_it()
            import time
            for _i in range(20):
                if len(found) == 4:
                    break
                time.sleep(.1)
            else:
                raise AssertionError('Could not get to condition before 2 seconds')

            self.assertEqual({'a': 1, 'b': 2, 'args': (3, 4), 'kwargs': {'e': 2, 'd': 1}}, found)
        finally:
            pydev_monkey.undo_patch_thread_modules()


#=======================================================================================================================
# main
#=======================================================================================================================
if __name__ == '__main__':
    unittest.main()
