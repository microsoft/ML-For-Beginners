"""Tests for the token-based transformers in IPython.core.inputtransformer2

Line-based transformers are the simpler ones; token-based transformers are
more complex. See test_inputtransformer2_line for tests for line-based
transformations.
"""
import platform
import string
import sys
from textwrap import dedent

import pytest

from IPython.core import inputtransformer2 as ipt2
from IPython.core.inputtransformer2 import _find_assign_op, make_tokens_by_line

MULTILINE_MAGIC = (
    """\
a = f()
%foo \\
bar
g()
""".splitlines(
        keepends=True
    ),
    (2, 0),
    """\
a = f()
get_ipython().run_line_magic('foo', ' bar')
g()
""".splitlines(
        keepends=True
    ),
)

INDENTED_MAGIC = (
    """\
for a in range(5):
    %ls
""".splitlines(
        keepends=True
    ),
    (2, 4),
    """\
for a in range(5):
    get_ipython().run_line_magic('ls', '')
""".splitlines(
        keepends=True
    ),
)

CRLF_MAGIC = (
    ["a = f()\n", "%ls\r\n", "g()\n"],
    (2, 0),
    ["a = f()\n", "get_ipython().run_line_magic('ls', '')\n", "g()\n"],
)

MULTILINE_MAGIC_ASSIGN = (
    """\
a = f()
b = %foo \\
  bar
g()
""".splitlines(
        keepends=True
    ),
    (2, 4),
    """\
a = f()
b = get_ipython().run_line_magic('foo', '   bar')
g()
""".splitlines(
        keepends=True
    ),
)

MULTILINE_SYSTEM_ASSIGN = ("""\
a = f()
b = !foo \\
  bar
g()
""".splitlines(keepends=True), (2, 4), """\
a = f()
b = get_ipython().getoutput('foo    bar')
g()
""".splitlines(keepends=True))

#####

MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT = (
    """\
def test():
  for i in range(1):
    print(i)
  res =! ls
""".splitlines(
        keepends=True
    ),
    (4, 7),
    """\
def test():
  for i in range(1):
    print(i)
  res =get_ipython().getoutput(\' ls\')
""".splitlines(
        keepends=True
    ),
)

######

AUTOCALL_QUOTE = ([",f 1 2 3\n"], (1, 0), ['f("1", "2", "3")\n'])

AUTOCALL_QUOTE2 = ([";f 1 2 3\n"], (1, 0), ['f("1 2 3")\n'])

AUTOCALL_PAREN = (["/f 1 2 3\n"], (1, 0), ["f(1, 2, 3)\n"])

SIMPLE_HELP = (["foo?\n"], (1, 0), ["get_ipython().run_line_magic('pinfo', 'foo')\n"])

DETAILED_HELP = (
    ["foo??\n"],
    (1, 0),
    ["get_ipython().run_line_magic('pinfo2', 'foo')\n"],
)

MAGIC_HELP = (["%foo?\n"], (1, 0), ["get_ipython().run_line_magic('pinfo', '%foo')\n"])

HELP_IN_EXPR = (
    ["a = b + c?\n"],
    (1, 0),
    ["get_ipython().run_line_magic('pinfo', 'c')\n"],
)

HELP_CONTINUED_LINE = (
    """\
a = \\
zip?
""".splitlines(
        keepends=True
    ),
    (1, 0),
    [r"get_ipython().run_line_magic('pinfo', 'zip')" + "\n"],
)

HELP_MULTILINE = (
    """\
(a,
b) = zip?
""".splitlines(
        keepends=True
    ),
    (1, 0),
    [r"get_ipython().run_line_magic('pinfo', 'zip')" + "\n"],
)

HELP_UNICODE = (
    ["π.foo?\n"],
    (1, 0),
    ["get_ipython().run_line_magic('pinfo', 'π.foo')\n"],
)


def null_cleanup_transformer(lines):
    """
    A cleanup transform that returns an empty list.
    """
    return []


def test_check_make_token_by_line_never_ends_empty():
    """
    Check that not sequence of single or double characters ends up leading to en empty list of tokens
    """
    from string import printable

    for c in printable:
        assert make_tokens_by_line(c)[-1] != []
        for k in printable:
            assert make_tokens_by_line(c + k)[-1] != []


def check_find(transformer, case, match=True):
    sample, expected_start, _ = case
    tbl = make_tokens_by_line(sample)
    res = transformer.find(tbl)
    if match:
        # start_line is stored 0-indexed, expected values are 1-indexed
        assert (res.start_line + 1, res.start_col) == expected_start
        return res
    else:
        assert res is None


def check_transform(transformer_cls, case):
    lines, start, expected = case
    transformer = transformer_cls(start)
    assert transformer.transform(lines) == expected


def test_continued_line():
    lines = MULTILINE_MAGIC_ASSIGN[0]
    assert ipt2.find_end_of_continued_line(lines, 1) == 2

    assert ipt2.assemble_continued_line(lines, (1, 5), 2) == "foo    bar"


def test_find_assign_magic():
    check_find(ipt2.MagicAssign, MULTILINE_MAGIC_ASSIGN)
    check_find(ipt2.MagicAssign, MULTILINE_SYSTEM_ASSIGN, match=False)
    check_find(ipt2.MagicAssign, MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT, match=False)


def test_transform_assign_magic():
    check_transform(ipt2.MagicAssign, MULTILINE_MAGIC_ASSIGN)


def test_find_assign_system():
    check_find(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN)
    check_find(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT)
    check_find(ipt2.SystemAssign, (["a =  !ls\n"], (1, 5), None))
    check_find(ipt2.SystemAssign, (["a=!ls\n"], (1, 2), None))
    check_find(ipt2.SystemAssign, MULTILINE_MAGIC_ASSIGN, match=False)


def test_transform_assign_system():
    check_transform(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN)
    check_transform(ipt2.SystemAssign, MULTILINE_SYSTEM_ASSIGN_AFTER_DEDENT)


def test_find_magic_escape():
    check_find(ipt2.EscapedCommand, MULTILINE_MAGIC)
    check_find(ipt2.EscapedCommand, INDENTED_MAGIC)
    check_find(ipt2.EscapedCommand, MULTILINE_MAGIC_ASSIGN, match=False)


def test_transform_magic_escape():
    check_transform(ipt2.EscapedCommand, MULTILINE_MAGIC)
    check_transform(ipt2.EscapedCommand, INDENTED_MAGIC)
    check_transform(ipt2.EscapedCommand, CRLF_MAGIC)


def test_find_autocalls():
    for case in [AUTOCALL_QUOTE, AUTOCALL_QUOTE2, AUTOCALL_PAREN]:
        print("Testing %r" % case[0])
        check_find(ipt2.EscapedCommand, case)


def test_transform_autocall():
    for case in [AUTOCALL_QUOTE, AUTOCALL_QUOTE2, AUTOCALL_PAREN]:
        print("Testing %r" % case[0])
        check_transform(ipt2.EscapedCommand, case)


def test_find_help():
    for case in [SIMPLE_HELP, DETAILED_HELP, MAGIC_HELP, HELP_IN_EXPR]:
        check_find(ipt2.HelpEnd, case)

    tf = check_find(ipt2.HelpEnd, HELP_CONTINUED_LINE)
    assert tf.q_line == 1
    assert tf.q_col == 3

    tf = check_find(ipt2.HelpEnd, HELP_MULTILINE)
    assert tf.q_line == 1
    assert tf.q_col == 8

    # ? in a comment does not trigger help
    check_find(ipt2.HelpEnd, (["foo # bar?\n"], None, None), match=False)
    # Nor in a string
    check_find(ipt2.HelpEnd, (["foo = '''bar?\n"], None, None), match=False)


def test_transform_help():
    tf = ipt2.HelpEnd((1, 0), (1, 9))
    assert tf.transform(HELP_IN_EXPR[0]) == HELP_IN_EXPR[2]

    tf = ipt2.HelpEnd((1, 0), (2, 3))
    assert tf.transform(HELP_CONTINUED_LINE[0]) == HELP_CONTINUED_LINE[2]

    tf = ipt2.HelpEnd((1, 0), (2, 8))
    assert tf.transform(HELP_MULTILINE[0]) == HELP_MULTILINE[2]

    tf = ipt2.HelpEnd((1, 0), (1, 0))
    assert tf.transform(HELP_UNICODE[0]) == HELP_UNICODE[2]


def test_find_assign_op_dedent():
    """
    be careful that empty token like dedent are not counted as parens
    """

    class Tk:
        def __init__(self, s):
            self.string = s

    assert _find_assign_op([Tk(s) for s in ("", "a", "=", "b")]) == 2
    assert (
        _find_assign_op([Tk(s) for s in ("", "(", "a", "=", "b", ")", "=", "5")]) == 6
    )


extra_closing_paren_param = (
    pytest.param("(\n))", "invalid", None)
    if sys.version_info >= (3, 12)
    else pytest.param("(\n))", "incomplete", 0)
)
examples = [
    pytest.param("a = 1", "complete", None),
    pytest.param("for a in range(5):", "incomplete", 4),
    pytest.param("for a in range(5):\n    if a > 0:", "incomplete", 8),
    pytest.param("raise = 2", "invalid", None),
    pytest.param("a = [1,\n2,", "incomplete", 0),
    extra_closing_paren_param,
    pytest.param("\\\r\n", "incomplete", 0),
    pytest.param("a = '''\n   hi", "incomplete", 3),
    pytest.param("def a():\n x=1\n global x", "invalid", None),
    pytest.param(
        "a \\ ",
        "invalid",
        None,
        marks=pytest.mark.xfail(
            reason="Bug in python 3.9.8 – bpo 45738",
            condition=sys.version_info
            in [(3, 9, 8, "final", 0), (3, 11, 0, "alpha", 2)],
            raises=SystemError,
            strict=True,
        ),
    ),  # Nothing allowed after backslash,
    pytest.param("1\\\n+2", "complete", None),
]


@pytest.mark.parametrize("code, expected, number", examples)
def test_check_complete_param(code, expected, number):
    cc = ipt2.TransformerManager().check_complete
    assert cc(code) == (expected, number)


@pytest.mark.xfail(platform.python_implementation() == "PyPy", reason="fail on pypy")
@pytest.mark.xfail(
    reason="Bug in python 3.9.8 – bpo 45738",
    condition=sys.version_info in [(3, 9, 8, "final", 0), (3, 11, 0, "alpha", 2)],
    raises=SystemError,
    strict=True,
)
def test_check_complete():
    cc = ipt2.TransformerManager().check_complete

    example = dedent(
        """
        if True:
            a=1"""
    )

    assert cc(example) == ("incomplete", 4)
    assert cc(example + "\n") == ("complete", None)
    assert cc(example + "\n    ") == ("complete", None)

    # no need to loop on all the letters/numbers.
    short = "12abAB" + string.printable[62:]
    for c in short:
        # test does not raise:
        cc(c)
        for k in short:
            cc(c + k)

    assert cc("def f():\n  x=0\n  \\\n  ") == ("incomplete", 2)


@pytest.mark.xfail(platform.python_implementation() == "PyPy", reason="fail on pypy")
@pytest.mark.parametrize(
    "value, expected",
    [
        ('''def foo():\n    """''', ("incomplete", 4)),
        ("""async with example:\n    pass""", ("incomplete", 4)),
        ("""async with example:\n    pass\n    """, ("complete", None)),
    ],
)
def test_check_complete_II(value, expected):
    """
    Test that multiple line strings are properly handled.

    Separate test function for convenience

    """
    cc = ipt2.TransformerManager().check_complete
    assert cc(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (")", ("invalid", None)),
        ("]", ("invalid", None)),
        ("}", ("invalid", None)),
        (")(", ("invalid", None)),
        ("][", ("invalid", None)),
        ("}{", ("invalid", None)),
        ("]()(", ("invalid", None)),
        ("())(", ("invalid", None)),
        (")[](", ("invalid", None)),
        ("()](", ("invalid", None)),
    ],
)
def test_check_complete_invalidates_sunken_brackets(value, expected):
    """
    Test that a single line with more closing brackets than the opening ones is
    interpreted as invalid
    """
    cc = ipt2.TransformerManager().check_complete
    assert cc(value) == expected


def test_null_cleanup_transformer():
    manager = ipt2.TransformerManager()
    manager.cleanup_transforms.insert(0, null_cleanup_transformer)
    assert manager.transform_cell("") == ""


def test_side_effects_I():
    count = 0

    def counter(lines):
        nonlocal count
        count += 1
        return lines

    counter.has_side_effects = True

    manager = ipt2.TransformerManager()
    manager.cleanup_transforms.insert(0, counter)
    assert manager.check_complete("a=1\n") == ("complete", None)
    assert count == 0


def test_side_effects_II():
    count = 0

    def counter(lines):
        nonlocal count
        count += 1
        return lines

    counter.has_side_effects = True

    manager = ipt2.TransformerManager()
    manager.line_transforms.insert(0, counter)
    assert manager.check_complete("b=1\n") == ("complete", None)
    assert count == 0
