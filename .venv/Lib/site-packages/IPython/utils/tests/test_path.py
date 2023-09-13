# encoding: utf-8
"""Tests for IPython.utils.path.py"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch

import pytest
from tempfile import TemporaryDirectory

import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
    onlyif_unicode_paths,
    skip_if_not_win32,
    skip_win32,
)
from IPython.testing.tools import make_tempfile
from IPython.utils import path

# Platform-dependent imports
try:
    import winreg as wreg
except ImportError:
    #Fake _winreg module on non-windows platforms
    import types
    wr_name = "winreg"
    sys.modules[wr_name] = types.ModuleType(wr_name)
    try:
        import winreg as wreg
    except ImportError:
        import _winreg as wreg

        #Add entries that needs to be stubbed by the testing code
        (wreg.OpenKey, wreg.QueryValueEx,) = (None, None)

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------
env = os.environ
TMP_TEST_DIR = tempfile.mkdtemp()
HOME_TEST_DIR = join(TMP_TEST_DIR, "home_test_dir")
#
# Setup/teardown functions/decorators
#

def setup_module():
    """Setup testenvironment for the module:

    - Adds dummy home dir tree
    """
    # Do not mask exceptions here.  In particular, catching WindowsError is a
    # problem because that exception is only defined on Windows...
    os.makedirs(os.path.join(HOME_TEST_DIR, 'ipython'))


def teardown_module():
    """Teardown testenvironment for the module:

    - Remove dummy home dir tree
    """
    # Note: we remove the parent test dir, which is the root of all test
    # subdirs we may have created.  Use shutil instead of os.removedirs, so
    # that non-empty directories are all recursively removed.
    shutil.rmtree(TMP_TEST_DIR)


def setup_environment():
    """Setup testenvironment for some functions that are tested
    in this module. In particular this functions stores attributes
    and other things that we need to stub in some test functions.
    This needs to be done on a function level and not module level because
    each testfunction needs a pristine environment.
    """
    global oldstuff, platformstuff
    oldstuff = (env.copy(), os.name, sys.platform, path.get_home_dir, IPython.__file__, os.getcwd())

def teardown_environment():
    """Restore things that were remembered by the setup_environment function
    """
    (oldenv, os.name, sys.platform, path.get_home_dir, IPython.__file__, old_wd) = oldstuff
    os.chdir(old_wd)
    reload(path)

    for key in list(env):
        if key not in oldenv:
            del env[key]
    env.update(oldenv)
    if hasattr(sys, 'frozen'):
        del sys.frozen


# Build decorator that uses the setup_environment/setup_environment
@pytest.fixture
def environment():
    setup_environment()
    yield
    teardown_environment()


with_environment = pytest.mark.usefixtures("environment")


@skip_if_not_win32
@with_environment
def test_get_home_dir_1():
    """Testcase for py2exe logic, un-compressed lib
    """
    unfrozen = path.get_home_dir()
    sys.frozen = True

    #fake filename for IPython.__init__
    IPython.__file__ = abspath(join(HOME_TEST_DIR, "Lib/IPython/__init__.py"))

    home_dir = path.get_home_dir()
    assert home_dir == unfrozen


@skip_if_not_win32
@with_environment
def test_get_home_dir_2():
    """Testcase for py2exe logic, compressed lib
    """
    unfrozen = path.get_home_dir()
    sys.frozen = True
    #fake filename for IPython.__init__
    IPython.__file__ = abspath(join(HOME_TEST_DIR, "Library.zip/IPython/__init__.py")).lower()

    home_dir = path.get_home_dir(True)
    assert home_dir == unfrozen


@skip_win32
@with_environment
def test_get_home_dir_3():
    """get_home_dir() uses $HOME if set"""
    env["HOME"] = HOME_TEST_DIR
    home_dir = path.get_home_dir(True)
    # get_home_dir expands symlinks
    assert home_dir == os.path.realpath(env["HOME"])


@with_environment
def test_get_home_dir_4():
    """get_home_dir() still works if $HOME is not set"""

    if 'HOME' in env: del env['HOME']
    # this should still succeed, but we don't care what the answer is
    home = path.get_home_dir(False)

@skip_win32
@with_environment
def test_get_home_dir_5():
    """raise HomeDirError if $HOME is specified, but not a writable dir"""
    env['HOME'] = abspath(HOME_TEST_DIR+'garbage')
    # set os.name = posix, to prevent My Documents fallback on Windows
    os.name = 'posix'
    pytest.raises(path.HomeDirError, path.get_home_dir, True)

# Should we stub wreg fully so we can run the test on all platforms?
@skip_if_not_win32
@with_environment
def test_get_home_dir_8():
    """Using registry hack for 'My Documents', os=='nt'

    HOMESHARE, HOMEDRIVE, HOMEPATH, USERPROFILE and others are missing.
    """
    os.name = 'nt'
    # Remove from stub environment all keys that may be set
    for key in ['HOME', 'HOMESHARE', 'HOMEDRIVE', 'HOMEPATH', 'USERPROFILE']:
        env.pop(key, None)

    class key:
        def __enter__(self):
            pass
        def Close(self):
            pass
        def __exit__(*args, **kwargs):
            pass

    with patch.object(wreg, 'OpenKey', return_value=key()), \
         patch.object(wreg, 'QueryValueEx', return_value=[abspath(HOME_TEST_DIR)]):
        home_dir = path.get_home_dir()
    assert home_dir == abspath(HOME_TEST_DIR)

@with_environment
def test_get_xdg_dir_0():
    """test_get_xdg_dir_0, check xdg_dir"""
    reload(path)
    path._writable_dir = lambda path: True
    path.get_home_dir = lambda : 'somewhere'
    os.name = "posix"
    sys.platform = "linux2"
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)

    assert path.get_xdg_dir() == os.path.join("somewhere", ".config")


@with_environment
def test_get_xdg_dir_1():
    """test_get_xdg_dir_1, check nonexistent xdg_dir"""
    reload(path)
    path.get_home_dir = lambda : HOME_TEST_DIR
    os.name = "posix"
    sys.platform = "linux2"
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)
    assert path.get_xdg_dir() is None

@with_environment
def test_get_xdg_dir_2():
    """test_get_xdg_dir_2, check xdg_dir default to ~/.config"""
    reload(path)
    path.get_home_dir = lambda : HOME_TEST_DIR
    os.name = "posix"
    sys.platform = "linux2"
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)
    cfgdir=os.path.join(path.get_home_dir(), '.config')
    if not os.path.exists(cfgdir):
        os.makedirs(cfgdir)

    assert path.get_xdg_dir() == cfgdir

@with_environment
def test_get_xdg_dir_3():
    """test_get_xdg_dir_3, check xdg_dir not used on non-posix systems"""
    reload(path)
    path.get_home_dir = lambda : HOME_TEST_DIR
    os.name = "nt"
    sys.platform = "win32"
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)
    cfgdir=os.path.join(path.get_home_dir(), '.config')
    os.makedirs(cfgdir, exist_ok=True)

    assert path.get_xdg_dir() is None

def test_filefind():
    """Various tests for filefind"""
    f = tempfile.NamedTemporaryFile()
    # print 'fname:',f.name
    alt_dirs = paths.get_ipython_dir()
    t = path.filefind(f.name, alt_dirs)
    # print 'found:',t


@dec.skip_if_not_win32
def test_get_long_path_name_win32():
    with TemporaryDirectory() as tmpdir:

        # Make a long path. Expands the path of tmpdir prematurely as it may already have a long
        # path component, so ensure we include the long form of it
        long_path = os.path.join(path.get_long_path_name(tmpdir), 'this is my long path name')
        os.makedirs(long_path)

        # Test to see if the short path evaluates correctly.
        short_path = os.path.join(tmpdir, 'THISIS~1')
        evaluated_path = path.get_long_path_name(short_path)
        assert evaluated_path.lower() == long_path.lower()


@dec.skip_win32
def test_get_long_path_name():
    p = path.get_long_path_name("/usr/local")
    assert p == "/usr/local"


class TestRaiseDeprecation(unittest.TestCase):

    @dec.skip_win32 # can't create not-user-writable dir on win
    @with_environment
    def test_not_writable_ipdir(self):
        tmpdir = tempfile.mkdtemp()
        os.name = "posix"
        env.pop('IPYTHON_DIR', None)
        env.pop('IPYTHONDIR', None)
        env.pop('XDG_CONFIG_HOME', None)
        env['HOME'] = tmpdir
        ipdir = os.path.join(tmpdir, '.ipython')
        os.mkdir(ipdir, 0o555)
        try:
            open(os.path.join(ipdir, "_foo_"), "w", encoding="utf-8").close()
        except IOError:
            pass
        else:
            # I can still write to an unwritable dir,
            # assume I'm root and skip the test
            pytest.skip("I can't create directories that I can't write to")

        with self.assertWarnsRegex(UserWarning, 'is not a writable location'):
            ipdir = paths.get_ipython_dir()
        env.pop('IPYTHON_DIR', None)

@with_environment
def test_get_py_filename():
    os.chdir(TMP_TEST_DIR)
    with make_tempfile("foo.py"):
        assert path.get_py_filename("foo.py") == "foo.py"
        assert path.get_py_filename("foo") == "foo.py"
    with make_tempfile("foo"):
        assert path.get_py_filename("foo") == "foo"
        pytest.raises(IOError, path.get_py_filename, "foo.py")
    pytest.raises(IOError, path.get_py_filename, "foo")
    pytest.raises(IOError, path.get_py_filename, "foo.py")
    true_fn = "foo with spaces.py"
    with make_tempfile(true_fn):
        assert path.get_py_filename("foo with spaces") == true_fn
        assert path.get_py_filename("foo with spaces.py") == true_fn
        pytest.raises(IOError, path.get_py_filename, '"foo with spaces.py"')
        pytest.raises(IOError, path.get_py_filename, "'foo with spaces.py'")

@onlyif_unicode_paths
def test_unicode_in_filename():
    """When a file doesn't exist, the exception raised should be safe to call
    str() on - i.e. in Python 2 it must only have ASCII characters.

    https://github.com/ipython/ipython/issues/875
    """
    try:
        # these calls should not throw unicode encode exceptions
        path.get_py_filename('fooéè.py')
    except IOError as ex:
        str(ex)


class TestShellGlob(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filenames_start_with_a = ['a0', 'a1', 'a2']
        cls.filenames_end_with_b = ['0b', '1b', '2b']
        cls.filenames = cls.filenames_start_with_a + cls.filenames_end_with_b
        cls.tempdir = TemporaryDirectory()
        td = cls.tempdir.name

        with cls.in_tempdir():
            # Create empty files
            for fname in cls.filenames:
                open(os.path.join(td, fname), "w", encoding="utf-8").close()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    @contextmanager
    def in_tempdir(cls):
        save = os.getcwd()
        try:
            os.chdir(cls.tempdir.name)
            yield
        finally:
            os.chdir(save)

    def check_match(self, patterns, matches):
        with self.in_tempdir():
            # glob returns unordered list. that's why sorted is required.
            assert sorted(path.shellglob(patterns)) == sorted(matches)

    def common_cases(self):
        return [
            (['*'], self.filenames),
            (['a*'], self.filenames_start_with_a),
            (['*c'], ['*c']),
            (['*', 'a*', '*b', '*c'], self.filenames
                                      + self.filenames_start_with_a
                                      + self.filenames_end_with_b
                                      + ['*c']),
            (['a[012]'], self.filenames_start_with_a),
        ]

    @skip_win32
    def test_match_posix(self):
        for (patterns, matches) in self.common_cases() + [
                ([r'\*'], ['*']),
                ([r'a\*', 'a*'], ['a*'] + self.filenames_start_with_a),
                ([r'a\[012]'], ['a[012]']),
                ]:
            self.check_match(patterns, matches)

    @skip_if_not_win32
    def test_match_windows(self):
        for (patterns, matches) in self.common_cases() + [
                # In windows, backslash is interpreted as path
                # separator.  Therefore, you can't escape glob
                # using it.
                ([r'a\*', 'a*'], [r'a\*'] + self.filenames_start_with_a),
                ([r'a\[012]'], [r'a\[012]']),
                ]:
            self.check_match(patterns, matches)


@pytest.mark.parametrize(
    "globstr, unescaped_globstr",
    [
        (r"\*\[\!\]\?", "*[!]?"),
        (r"\\*", r"\*"),
        (r"\\\*", r"\*"),
        (r"\\a", r"\a"),
        (r"\a", r"\a"),
    ],
)
def test_unescape_glob(globstr, unescaped_globstr):
    assert path.unescape_glob(globstr) == unescaped_globstr


@onlyif_unicode_paths
def test_ensure_dir_exists():
    with TemporaryDirectory() as td:
        d = os.path.join(td, '∂ir')
        path.ensure_dir_exists(d) # create it
        assert os.path.isdir(d)
        path.ensure_dir_exists(d)  # no-op
        f = os.path.join(td, "ƒile")
        open(f, "w", encoding="utf-8").close()  # touch
        with pytest.raises(IOError):
            path.ensure_dir_exists(f)

class TestLinkOrCopy(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.src = self.dst("src")
        with open(self.src, "w", encoding="utf-8") as f:
            f.write("Hello, world!")

    def tearDown(self):
        self.tempdir.cleanup()

    def dst(self, *args):
        return os.path.join(self.tempdir.name, *args)

    def assert_inode_not_equal(self, a, b):
        assert (
            os.stat(a).st_ino != os.stat(b).st_ino
        ), "%r and %r do reference the same indoes" % (a, b)

    def assert_inode_equal(self, a, b):
        assert (
            os.stat(a).st_ino == os.stat(b).st_ino
        ), "%r and %r do not reference the same indoes" % (a, b)

    def assert_content_equal(self, a, b):
        with open(a, "rb") as a_f:
            with open(b, "rb") as b_f:
                assert a_f.read() == b_f.read()

    @skip_win32
    def test_link_successful(self):
        dst = self.dst("target")
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)

    @skip_win32
    def test_link_into_dir(self):
        dst = self.dst("some_dir")
        os.mkdir(dst)
        path.link_or_copy(self.src, dst)
        expected_dst = self.dst("some_dir", os.path.basename(self.src))
        self.assert_inode_equal(self.src, expected_dst)

    @skip_win32
    def test_target_exists(self):
        dst = self.dst("target")
        open(dst, "w", encoding="utf-8").close()
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)

    @skip_win32
    def test_no_link(self):
        real_link = os.link
        try:
            del os.link
            dst = self.dst("target")
            path.link_or_copy(self.src, dst)
            self.assert_content_equal(self.src, dst)
            self.assert_inode_not_equal(self.src, dst)
        finally:
            os.link = real_link

    @skip_if_not_win32
    def test_windows(self):
        dst = self.dst("target")
        path.link_or_copy(self.src, dst)
        self.assert_content_equal(self.src, dst)

    def test_link_twice(self):
        # Linking the same file twice shouldn't leave duplicates around.
        # See https://github.com/ipython/ipython/issues/6450
        dst = self.dst('target')
        path.link_or_copy(self.src, dst)
        path.link_or_copy(self.src, dst)
        self.assert_inode_equal(self.src, dst)
        assert sorted(os.listdir(self.tempdir.name)) == ["src", "target"]
