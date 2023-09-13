# coding: utf-8
"""Test suite for our color utilities.

Authors
-------

* Min RK
"""
#-----------------------------------------------------------------------------
#  Copyright (C) 2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING.txt, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# our own
import sys
from IPython.utils.PyColorize import Parser
import io
import pytest


@pytest.fixture(scope="module", params=("Linux", "NoColor", "LightBG", "Neutral"))
def style(request):
    yield request.param

#-----------------------------------------------------------------------------
# Test functions
#-----------------------------------------------------------------------------

sample = """
def function(arg, *args, kwarg=True, **kwargs):
    '''
    this is docs
    '''
    pass is True
    False == None

    with io.open(ru'unicode', encoding='utf-8'):
        raise ValueError("escape \r sequence")

    print("wěird ünicoðe")

class Bar(Super):

    def __init__(self):
        super(Bar, self).__init__(1**2, 3^4, 5 or 6)
"""


def test_parse_sample(style):
    """and test writing to a buffer"""
    buf = io.StringIO()
    p = Parser(style=style)
    p.format(sample, buf)
    buf.seek(0)
    f1 = buf.read()

    assert "ERROR" not in f1


def test_parse_error(style):
    p = Parser(style=style)
    f1 = p.format(r"\ " if sys.version_info >= (3, 12) else ")", "str")
    if style != "NoColor":
        assert "ERROR" in f1
