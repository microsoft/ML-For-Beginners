"""Tests for IPython.utils.importstring."""

#-----------------------------------------------------------------------------
#  Copyright (C) 2013  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import pytest

from IPython.utils.importstring import import_item

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

def test_import_plain():
    "Test simple imports"
    import os

    os2 = import_item("os")
    assert os is os2


def test_import_nested():
    "Test nested imports from the stdlib"
    from os import path

    path2 = import_item("os.path")
    assert path is path2


def test_import_raises():
    "Test that failing imports raise the right exception"
    pytest.raises(ImportError, import_item, "IPython.foobar")
