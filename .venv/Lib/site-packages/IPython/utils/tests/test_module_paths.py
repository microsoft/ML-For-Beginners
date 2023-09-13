# encoding: utf-8
"""Tests for IPython.utils.module_paths.py"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import shutil
import sys
import tempfile

from pathlib import Path

import IPython.utils.module_paths as mp

TEST_FILE_PATH = Path(__file__).resolve().parent

TMP_TEST_DIR = Path(tempfile.mkdtemp(suffix="with.dot"))
#
# Setup/teardown functions/decorators
#

old_syspath = sys.path

def make_empty_file(fname):
    open(fname, "w", encoding="utf-8").close()


def setup_module():
    """Setup testenvironment for the module:

    """
    # Do not mask exceptions here.  In particular, catching WindowsError is a
    # problem because that exception is only defined on Windows...
    Path(TMP_TEST_DIR / "xmod").mkdir(parents=True)
    Path(TMP_TEST_DIR / "nomod").mkdir(parents=True)
    make_empty_file(TMP_TEST_DIR / "xmod/__init__.py")
    make_empty_file(TMP_TEST_DIR / "xmod/sub.py")
    make_empty_file(TMP_TEST_DIR / "pack.py")
    make_empty_file(TMP_TEST_DIR / "packpyc.pyc")
    sys.path = [str(TMP_TEST_DIR)]

def teardown_module():
    """Teardown testenvironment for the module:

    - Remove tempdir
    - restore sys.path
    """
    # Note: we remove the parent test dir, which is the root of all test
    # subdirs we may have created.  Use shutil instead of os.removedirs, so
    # that non-empty directories are all recursively removed.
    shutil.rmtree(TMP_TEST_DIR)
    sys.path = old_syspath

def test_tempdir():
    """
    Ensure the test are done with a temporary file that have a dot somewhere.
    """
    assert "." in str(TMP_TEST_DIR)


def test_find_mod_1():
    """
    Search for a directory's file path.
    Expected output: a path to that directory's __init__.py file.
    """
    modpath = TMP_TEST_DIR / "xmod" / "__init__.py"
    assert Path(mp.find_mod("xmod")) == modpath

def test_find_mod_2():
    """
    Search for a directory's file path.
    Expected output: a path to that directory's __init__.py file.
    TODO: Confirm why this is a duplicate test.
    """
    modpath = TMP_TEST_DIR / "xmod" / "__init__.py"
    assert Path(mp.find_mod("xmod")) == modpath

def test_find_mod_3():
    """
    Search for a directory + a filename without its .py extension
    Expected output: full path with .py extension.
    """
    modpath = TMP_TEST_DIR / "xmod" / "sub.py"
    assert Path(mp.find_mod("xmod.sub")) == modpath

def test_find_mod_4():
    """
    Search for a filename without its .py extension
    Expected output: full path with .py extension
    """
    modpath = TMP_TEST_DIR / "pack.py"
    assert Path(mp.find_mod("pack")) == modpath

def test_find_mod_5():
    """
    Search for a filename with a .pyc extension
    Expected output: TODO: do we exclude or include .pyc files?
    """
    assert mp.find_mod("packpyc") == None
