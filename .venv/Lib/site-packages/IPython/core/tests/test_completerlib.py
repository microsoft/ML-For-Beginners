# -*- coding: utf-8 -*-
"""Tests for completerlib.

"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os
import shutil
import sys
import tempfile
import unittest
from os.path import join

from tempfile import TemporaryDirectory

from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths


class MockEvent(object):
    def __init__(self, line):
        self.line = line

#-----------------------------------------------------------------------------
# Test functions begin
#-----------------------------------------------------------------------------
class Test_magic_run_completer(unittest.TestCase):
    files = [u"aao.py", u"a.py", u"b.py", u"aao.txt"]
    dirs = [u"adir/", "bdir/"]

    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in self.files:
            with open(join(self.BASETESTDIR, fil), "w", encoding="utf-8") as sfile:
                sfile.write("pass\n")
        for d in self.dirs:
            os.mkdir(join(self.BASETESTDIR, d))

        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    def test_1(self):
        """Test magic_run_completer, should match two alternatives
        """
        event = MockEvent(u"%run a")
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"a.py", u"aao.py", u"adir/"})

    def test_2(self):
        """Test magic_run_completer, should match one alternative
        """
        event = MockEvent(u"%run aa")
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"aao.py"})

    def test_3(self):
        """Test magic_run_completer with unterminated " """
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"a.py", u"aao.py", u"adir/"})

    def test_completion_more_args(self):
        event = MockEvent(u'%run a.py ')
        match = set(magic_run_completer(None, event))
        self.assertEqual(match, set(self.files + self.dirs))

    def test_completion_in_dir(self):
        # Github issue #3459
        event = MockEvent(u'%run a.py {}'.format(join(self.BASETESTDIR, 'a')))
        print(repr(event.line))
        match = set(magic_run_completer(None, event))
        # We specifically use replace here rather than normpath, because
        # at one point there were duplicates 'adir' and 'adir/', and normpath
        # would hide the failure for that.
        self.assertEqual(match, {join(self.BASETESTDIR, f).replace('\\','/')
                            for f in (u'a.py', u'aao.py', u'aao.txt', u'adir/')})

class Test_magic_run_completer_nonascii(unittest.TestCase):
    @onlyif_unicode_paths
    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in [u"aaø.py", u"a.py", u"b.py"]:
            with open(join(self.BASETESTDIR, fil), "w", encoding="utf-8") as sfile:
                sfile.write("pass\n")
        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        """Test magic_run_completer, should match two alternatives
        """
        event = MockEvent(u"%run a")
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"a.py", u"aaø.py"})

    @onlyif_unicode_paths
    def test_2(self):
        """Test magic_run_completer, should match one alternative
        """
        event = MockEvent(u"%run aa")
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"aaø.py"})

    @onlyif_unicode_paths
    def test_3(self):
        """Test magic_run_completer with unterminated " """
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u"a.py", u"aaø.py"})

# module_completer:

def test_import_invalid_module():
    """Testing of issue https://github.com/ipython/ipython/issues/1107"""
    invalid_module_names = {'foo-bar', 'foo:bar', '10foo'}
    valid_module_names = {'foobar'}
    with TemporaryDirectory() as tmpdir:
        sys.path.insert( 0, tmpdir )
        for name in invalid_module_names | valid_module_names:
            filename = os.path.join(tmpdir, name + ".py")
            open(filename, "w", encoding="utf-8").close()

        s = set( module_completion('import foo') )
        intersection = s.intersection(invalid_module_names)
        assert intersection == set()

        assert valid_module_names.issubset(s), valid_module_names.intersection(s)


def test_bad_module_all():
    """Test module with invalid __all__

    https://github.com/ipython/ipython/issues/9678
    """
    testsdir = os.path.dirname(__file__)
    sys.path.insert(0, testsdir)
    try:
        results = module_completion("from bad_all import ")
        assert "puppies" in results
        for r in results:
            assert isinstance(r, str)

        # bad_all doesn't contain submodules, but this completion
        # should finish without raising an exception:
        results = module_completion("import bad_all.")
        assert results == []
    finally:
        sys.path.remove(testsdir)


def test_module_without_init():
    """
    Test module without __init__.py.
    
    https://github.com/ipython/ipython/issues/11226
    """
    fake_module_name = "foo"
    with TemporaryDirectory() as tmpdir:
        sys.path.insert(0, tmpdir)
        try:
            os.makedirs(os.path.join(tmpdir, fake_module_name))
            s = try_import(mod=fake_module_name)
            assert s == [], f"for module {fake_module_name}"
        finally:
            sys.path.remove(tmpdir)


def test_valid_exported_submodules():
    """
    Test checking exported (__all__) objects are submodules
    """
    results = module_completion("import os.pa")
    # ensure we get a valid submodule:
    assert "os.path" in results
    # ensure we don't get objects that aren't submodules:
    assert "os.pathconf" not in results
