"""Tests for tokenutil"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import pytest

from IPython.utils.tokenutil import token_at_cursor, line_at_cursor

def expect_token(expected, cell, cursor_pos):
    token = token_at_cursor(cell, cursor_pos)
    offset = 0
    for line in cell.splitlines():
        if offset + len(line) >= cursor_pos:
            break
        else:
            offset += len(line)+1
    column = cursor_pos - offset
    line_with_cursor = "%s|%s" % (line[:column], line[column:])
    assert token == expected, "Expected %r, got %r in: %r (pos %i)" % (
        expected,
        token,
        line_with_cursor,
        cursor_pos,
    )


def test_simple():
    cell = "foo"
    for i in range(len(cell)):
        expect_token("foo", cell, i)

def test_function():
    cell = "foo(a=5, b='10')"
    expected = 'foo'
    # up to `foo(|a=`
    for i in range(cell.find('a=') + 1):
        expect_token("foo", cell, i)
    # find foo after `=`
    for i in [cell.find('=') + 1, cell.rfind('=') + 1]:
        expect_token("foo", cell, i)
    # in between `5,|` and `|b=`
    for i in range(cell.find(','), cell.find('b=')):
        expect_token("foo", cell, i)

def test_multiline():
    cell = '\n'.join([
        'a = 5',
        'b = hello("string", there)'
    ])
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)

def test_multiline_token():
    cell = '\n'.join([
        '"""\n\nxxxxxxxxxx\n\n"""',
        '5, """',
        'docstring',
        'multiline token',
        '""", [',
        '2, 3, "complicated"]',
        'b = hello("string", there)'
    ])
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)

def test_nested_call():
    cell = "foo(bar(a=5), b=10)"
    expected = 'foo'
    start = cell.index('bar') + 1
    for i in range(start, start + 3):
        expect_token(expected, cell, i)
    expected = 'bar'
    start = cell.index('a=')
    for i in range(start, start + 3):
        expect_token(expected, cell, i)
    expected = 'foo'
    start = cell.index(')') + 1
    for i in range(start, len(cell)-1):
        expect_token(expected, cell, i)

def test_attrs():
    cell = "a = obj.attr.subattr"
    expected = 'obj'
    idx = cell.find('obj') + 1
    for i in range(idx, idx + 3):
        expect_token(expected, cell, i)
    idx = cell.find('.attr') + 2
    expected = 'obj.attr'
    for i in range(idx, idx + 4):
        expect_token(expected, cell, i)
    idx = cell.find('.subattr') + 2
    expected = 'obj.attr.subattr'
    for i in range(idx, len(cell)):
        expect_token(expected, cell, i)

def test_line_at_cursor():
    cell = ""
    (line, offset) = line_at_cursor(cell, cursor_pos=11)
    assert line == ""
    assert offset == 0

    # The position after a newline should be the start of the following line.
    cell = "One\nTwo\n"
    (line, offset) = line_at_cursor(cell, cursor_pos=4)
    assert line == "Two\n"
    assert offset == 4

    # The end of a cell should be on the last line
    cell = "pri\npri"
    (line, offset) = line_at_cursor(cell, cursor_pos=7)
    assert line == "pri"
    assert offset == 4


@pytest.mark.parametrize(
    "c, token",
    zip(
        list(range(16, 22)) + list(range(22, 28)),
        ["int"] * (22 - 16) + ["map"] * (28 - 22),
    ),
)
def test_multiline_statement(c, token):
    cell = """a = (1,
    3)

int()
map()
"""
    expect_token(token, cell, c)
