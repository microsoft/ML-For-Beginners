# encoding: utf-8
"""Tests for IPython.utils.text"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os
import math
import random

from pathlib import Path

import pytest

from IPython.utils import text

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected, width, row_first, spread",
    (
        (
            "aaaaa  bbbbb  ccccc  ddddd\n",
            80,
            False,
            False,
        ),
        (
            "aaaaa  ccccc\nbbbbb  ddddd\n",
            25,
            False,
            False,
        ),
        (
            "aaaaa  ccccc\nbbbbb  ddddd\n",
            12,
            False,
            False,
        ),
        (
            "aaaaa\nbbbbb\nccccc\nddddd\n",
            10,
            False,
            False,
        ),
        (
            "aaaaa  bbbbb  ccccc  ddddd\n",
            80,
            True,
            False,
        ),
        (
            "aaaaa  bbbbb\nccccc  ddddd\n",
            25,
            True,
            False,
        ),
        (
            "aaaaa  bbbbb\nccccc  ddddd\n",
            12,
            True,
            False,
        ),
        (
            "aaaaa\nbbbbb\nccccc\nddddd\n",
            10,
            True,
            False,
        ),
        (
            "aaaaa      bbbbb      ccccc      ddddd\n",
            40,
            False,
            True,
        ),
        (
            "aaaaa          ccccc\nbbbbb          ddddd\n",
            20,
            False,
            True,
        ),
        (
            "aaaaa  ccccc\nbbbbb  ddddd\n",
            12,
            False,
            True,
        ),
        (
            "aaaaa\nbbbbb\nccccc\nddddd\n",
            10,
            False,
            True,
        ),
    ),
)
def test_columnize(expected, width, row_first, spread):
    """Basic columnize tests."""
    size = 5
    items = [l*size for l in 'abcd']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(
            items, displaywidth=width, row_first=row_first, spread=spread
        )
        assert out == expected


def test_columnize_random():
    """Test with random input to hopefully catch edge case """
    for row_first in [True, False]:
        for nitems in [random.randint(2,70) for i in range(2,20)]:
            displaywidth = random.randint(20,200)
            rand_len = [random.randint(2,displaywidth) for i in range(nitems)]
            items = ['x'*l for l in rand_len]
            with pytest.warns(PendingDeprecationWarning):
                out = text.columnize(
                    items, row_first=row_first, displaywidth=displaywidth
                )
            longer_line = max([len(x) for x in out.split("\n")])
            longer_element = max(rand_len)
            assert longer_line <= displaywidth, (
                f"Columnize displayed something lager than displaywidth : {longer_line}\n"
                f"longer element : {longer_element}\n"
                f"displaywidth : {displaywidth}\n"
                f"number of element : {nitems}\n"
                f"size of each element : {rand_len}\n"
                f"row_first={row_first}\n"
            )


@pytest.mark.parametrize("row_first", [True, False])
def test_columnize_medium(row_first):
    """Test with inputs than shouldn't be wider than 80"""
    size = 40
    items = [l*size for l in 'abc']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(items, row_first=row_first, displaywidth=80)
    assert out == "\n".join(items + [""]), "row_first={0}".format(row_first)


@pytest.mark.parametrize("row_first", [True, False])
def test_columnize_long(row_first):
    """Test columnize with inputs longer than the display window"""
    size = 11
    items = [l*size for l in 'abc']
    with pytest.warns(PendingDeprecationWarning):
        out = text.columnize(items, row_first=row_first, displaywidth=size - 1)
    assert out == "\n".join(items + [""]), "row_first={0}".format(row_first)


def eval_formatter_check(f):
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os, u=u"café", b="café")
    s = f.format("{n} {n//4} {stuff.split()[0]}", **ns)
    assert s == "12 3 hello"
    s = f.format(" ".join(["{n//%i}" % i for i in range(1, 8)]), **ns)
    assert s == "12 6 4 3 2 2 1"
    s = f.format("{[n//i for i in range(1,8)]}", **ns)
    assert s == "[12, 6, 4, 3, 2, 2, 1]"
    s = f.format("{stuff!s}", **ns)
    assert s == ns["stuff"]
    s = f.format("{stuff!r}", **ns)
    assert s == repr(ns["stuff"])

    # Check with unicode:
    s = f.format("{u}", **ns)
    assert s == ns["u"]
    # This decodes in a platform dependent manner, but it shouldn't error out
    s = f.format("{b}", **ns)

    pytest.raises(NameError, f.format, "{dne}", **ns)


def eval_formatter_slicing_check(f):
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os)
    s = f.format(" {stuff.split()[:]} ", **ns)
    assert s == " ['hello', 'there'] "
    s = f.format(" {stuff.split()[::-1]} ", **ns)
    assert s == " ['there', 'hello'] "
    s = f.format("{stuff[::2]}", **ns)
    assert s == ns["stuff"][::2]

    pytest.raises(SyntaxError, f.format, "{n:x}", **ns)

def eval_formatter_no_slicing_check(f):
    ns = dict(n=12, pi=math.pi, stuff="hello there", os=os)

    s = f.format("{n:x} {pi**2:+f}", **ns)
    assert s == "c +9.869604"

    s = f.format("{stuff[slice(1,4)]}", **ns)
    assert s == "ell"

    s = f.format("{a[:]}", a=[1, 2])
    assert s == "[1, 2]"

def test_eval_formatter():
    f = text.EvalFormatter()
    eval_formatter_check(f)
    eval_formatter_no_slicing_check(f)

def test_full_eval_formatter():
    f = text.FullEvalFormatter()
    eval_formatter_check(f)
    eval_formatter_slicing_check(f)

def test_dollar_formatter():
    f = text.DollarFormatter()
    eval_formatter_check(f)
    eval_formatter_slicing_check(f)
    
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os)
    s = f.format("$n", **ns)
    assert s == "12"
    s = f.format("$n.real", **ns)
    assert s == "12"
    s = f.format("$n/{stuff[:5]}", **ns)
    assert s == "12/hello"
    s = f.format("$n $$HOME", **ns)
    assert s == "12 $HOME"
    s = f.format("${foo}", foo="HOME")
    assert s == "$HOME"


def test_strip_email():
    src = """\
        >> >>> def f(x):
        >> ...   return x+1
        >> ... 
        >> >>> zz = f(2.5)"""
    cln = """\
>>> def f(x):
...   return x+1
... 
>>> zz = f(2.5)"""
    assert text.strip_email_quotes(src) == cln


def test_strip_email2():
    src = "> > > list()"
    cln = "list()"
    assert text.strip_email_quotes(src) == cln


def test_LSString():
    lss = text.LSString("abc\ndef")
    assert lss.l == ["abc", "def"]
    assert lss.s == "abc def"
    lss = text.LSString(os.getcwd())
    assert isinstance(lss.p[0], Path)


def test_SList():
    sl = text.SList(["a 11", "b 1", "a 2"])
    assert sl.n == "a 11\nb 1\na 2"
    assert sl.s == "a 11 b 1 a 2"
    assert sl.grep(lambda x: x.startswith("a")) == text.SList(["a 11", "a 2"])
    assert sl.fields(0) == text.SList(["a", "b", "a"])
    assert sl.sort(field=1, nums=True) == text.SList(["b 1", "a 2", "a 11"])
