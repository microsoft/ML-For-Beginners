# coding: utf-8
"""Tests for the compilerop module.
"""
#-----------------------------------------------------------------------------
#  Copyright (C) 2010-2011 The IPython Development Team.
#
#  Distributed under the terms of the BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib imports
import linecache
import sys

# Our own imports
from IPython.core import compilerop

#-----------------------------------------------------------------------------
# Test functions
#-----------------------------------------------------------------------------

def test_code_name():
    code = 'x=1'
    name = compilerop.code_name(code)
    assert name.startswith("<ipython-input-0")


def test_code_name2():
    code = 'x=1'
    name = compilerop.code_name(code, 9)
    assert name.startswith("<ipython-input-9")


def test_cache():
    """Test the compiler correctly compiles and caches inputs
    """
    cp = compilerop.CachingCompiler()
    ncache = len(linecache.cache)
    cp.cache('x=1')
    assert len(linecache.cache) > ncache

def test_proper_default_encoding():
    # Check we're in a proper Python 2 environment (some imports, such
    # as GTK, can change the default encoding, which can hide bugs.)
    assert sys.getdefaultencoding() == "utf-8"

def test_cache_unicode():
    cp = compilerop.CachingCompiler()
    ncache = len(linecache.cache)
    cp.cache(u"t = 'žćčšđ'")
    assert len(linecache.cache) > ncache

def test_compiler_check_cache():
    """Test the compiler properly manages the cache.
    """
    # Rather simple-minded tests that just exercise the API
    cp = compilerop.CachingCompiler()
    cp.cache('x=1', 99)
    # Ensure now that after clearing the cache, our entries survive
    linecache.checkcache()
    assert any(
        k.startswith("<ipython-input-99") for k in linecache.cache
    ), "Entry for input-99 missing from linecache"
