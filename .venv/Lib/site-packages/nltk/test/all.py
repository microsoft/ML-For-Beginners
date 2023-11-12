"""Test suite that runs all NLTK tests.

This module, `nltk.test.all`, is named as the NLTK ``test_suite`` in the
project's ``setup-eggs.py`` file.  Here, we create a test suite that
runs all of our doctests, and return it for processing by the setuptools
test harness.

"""
import doctest
import os.path
import unittest
from glob import glob


def additional_tests():
    # print("here-000000000000000")
    # print("-----", glob(os.path.join(os.path.dirname(__file__), '*.doctest')))
    dir = os.path.dirname(__file__)
    paths = glob(os.path.join(dir, "*.doctest"))
    files = [os.path.basename(path) for path in paths]
    return unittest.TestSuite([doctest.DocFileSuite(file) for file in files])


# if os.path.split(path)[-1] != 'index.rst'
# skips time-dependent doctest in index.rst
