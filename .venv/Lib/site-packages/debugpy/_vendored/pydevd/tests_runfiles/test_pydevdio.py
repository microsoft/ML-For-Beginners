import sys
import os


import unittest

class Test(unittest.TestCase):

    def test_it(self):
        #make it as if we were executing from the directory above this one (so that we can use jycompletionserver
        #without the need for it being in the pythonpath)
        #(twice the dirname to get the previous level from this file.)
        import test_pydevdio #@UnresolvedImport - importing itself
        ADD_TO_PYTHONPATH = os.path.join(os.path.dirname(os.path.dirname(test_pydevdio.__file__)))
        sys.path.insert(0, ADD_TO_PYTHONPATH)

        try:
            from _pydevd_bundle import pydevd_io
            original = sys.stdout

            try:
                sys.stdout = pydevd_io.IOBuf()
                print('foo')
                print('bar')

                self.assertEqual('foo\nbar\n', sys.stdout.getvalue()) #@UndefinedVariable

                print('ww')
                print('xx')
                self.assertEqual('ww\nxx\n', sys.stdout.getvalue()) #@UndefinedVariable
            finally:
                sys.stdout = original
        finally:
            #remove it to leave it ok for other tests
            sys.path.remove(ADD_TO_PYTHONPATH)

