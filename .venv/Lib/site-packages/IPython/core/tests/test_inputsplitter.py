# -*- coding: utf-8 -*-
"""Tests for the inputsplitter module."""


# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import unittest
import pytest
import sys

with pytest.warns(DeprecationWarning, match="inputsplitter"):
    from IPython.core import inputsplitter as isp
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt

#-----------------------------------------------------------------------------
# Semi-complete examples (also used as tests)
#-----------------------------------------------------------------------------

# Note: at the bottom, there's a slightly more complete version of this that
# can be useful during development of code here.

def mini_interactive_loop(input_func):
    """Minimal example of the logic of an interactive interpreter loop.

    This serves as an example, and it is used by the test system with a fake
    raw_input that simulates interactive input."""

    from IPython.core.inputsplitter import InputSplitter

    isp = InputSplitter()
    # In practice, this input loop would be wrapped in an outside loop to read
    # input indefinitely, until some exit/quit command was issued.  Here we
    # only illustrate the basic inner loop.
    while isp.push_accepts_more():
        indent = ' '*isp.get_indent_spaces()
        prompt = '>>> ' + indent
        line = indent + input_func(prompt)
        isp.push(line)

    # Here we just return input so we can use it in a test suite, but a real
    # interpreter would instead send it for execution somewhere.
    src = isp.source_reset()
    #print 'Input source was:\n', src  # dbg
    return src

#-----------------------------------------------------------------------------
# Test utilities, just for local use
#-----------------------------------------------------------------------------


def pseudo_input(lines):
    """Return a function that acts like raw_input but feeds the input list."""
    ilines = iter(lines)
    def raw_in(prompt):
        try:
            return next(ilines)
        except StopIteration:
            return ''
    return raw_in

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
def test_spaces():
    tests = [('', 0),
             (' ', 1),
             ('\n', 0),
             (' \n', 1),
             ('x', 0),
             (' x', 1),
             ('  x',2),
             ('    x',4),
             # Note: tabs are counted as a single whitespace!
             ('\tx', 1),
             ('\t x', 2),
             ]
    with pytest.warns(PendingDeprecationWarning):
        tt.check_pairs(isp.num_ini_spaces, tests)


def test_remove_comments():
    tests = [('text', 'text'),
             ('text # comment', 'text '),
             ('text # comment\n', 'text \n'),
             ('text # comment \n', 'text \n'),
             ('line # c \nline\n','line \nline\n'),
             ('line # c \nline#c2  \nline\nline #c\n\n',
              'line \nline\nline\nline \n\n'),
             ]
    tt.check_pairs(isp.remove_comments, tests)


def test_get_input_encoding():
    encoding = isp.get_input_encoding()
    assert isinstance(encoding, str)
    # simple-minded check that at least encoding a simple string works with the
    # encoding we got.
    assert "test".encode(encoding) == b"test"


class NoInputEncodingTestCase(unittest.TestCase):
    def setUp(self):
        self.old_stdin = sys.stdin
        class X: pass
        fake_stdin = X()
        sys.stdin = fake_stdin

    def test(self):
        # Verify that if sys.stdin has no 'encoding' attribute we do the right
        # thing
        enc = isp.get_input_encoding()
        self.assertEqual(enc, 'ascii')

    def tearDown(self):
        sys.stdin = self.old_stdin


class InputSplitterTestCase(unittest.TestCase):
    def setUp(self):
        self.isp = isp.InputSplitter()

    def test_reset(self):
        isp = self.isp
        isp.push('x=1')
        isp.reset()
        self.assertEqual(isp._buffer, [])
        self.assertEqual(isp.get_indent_spaces(), 0)
        self.assertEqual(isp.source, '')
        self.assertEqual(isp.code, None)
        self.assertEqual(isp._is_complete, False)

    def test_source(self):
        self.isp._store('1')
        self.isp._store('2')
        self.assertEqual(self.isp.source, '1\n2\n')
        self.assertEqual(len(self.isp._buffer)>0, True)
        self.assertEqual(self.isp.source_reset(), '1\n2\n')
        self.assertEqual(self.isp._buffer, [])
        self.assertEqual(self.isp.source, '')

    def test_indent(self):
        isp = self.isp # shorthand
        isp.push('x=1')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_indent2(self):
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        # Blank lines shouldn't change the indent level
        isp.push(' '*2)
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent3(self):
        isp = self.isp
        # When a multiline statement contains parens or multiline strings, we
        # shouldn't get confused.
        isp.push("if 1:")
        isp.push("    x = (1+\n    2)")
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent4(self):
        isp = self.isp
        # whitespace after ':' should not screw up indent level
        isp.push('if 1: \n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\t\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_pass(self):
        isp = self.isp # shorthand
        # should NOT cause dedent
        isp.push('if 1:\n    passes = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     pass')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     pass   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_break(self):
        isp = self.isp # shorthand
        # should NOT cause dedent
        isp.push('while 1:\n    breaks = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     break')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     break   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_continue(self):
        isp = self.isp # shorthand
        # should NOT cause dedent
        isp.push('while 1:\n    continues = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     continue')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     continue   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_raise(self):
        isp = self.isp # shorthand
        # should NOT cause dedent
        isp.push('if 1:\n    raised = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     raise TypeError()')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise      ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_return(self):
        isp = self.isp # shorthand
        # should NOT cause dedent
        isp.push('if 1:\n    returning = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     return 5 + 493')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return      ')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return(0)')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_push(self):
        isp = self.isp
        self.assertEqual(isp.push('x=1'), True)

    def test_push2(self):
        isp = self.isp
        self.assertEqual(isp.push('if 1:'), False)
        for line in ['  x=1', '# a comment', '  y=2']:
            print(line)
            self.assertEqual(isp.push(line), True)

    def test_push3(self):
        isp = self.isp
        isp.push('if True:')
        isp.push('  a = 1')
        self.assertEqual(isp.push('b = [1,'), False)

    def test_push_accepts_more(self):
        isp = self.isp
        isp.push('x=1')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more2(self):
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('  x=1')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more3(self):
        isp = self.isp
        isp.push("x = (2+\n3)")
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more4(self):
        isp = self.isp
        # When a multiline statement contains parens or multiline strings, we
        # shouldn't get confused.
        # FIXME: we should be able to better handle de-dents in statements like
        # multiline strings and multiline expressions (continued with \ or
        # parens).  Right now we aren't handling the indentation tracking quite
        # correctly with this, though in practice it may not be too much of a
        # problem.  We'll need to see.
        isp.push("if 1:")
        isp.push("    x = (2+")
        isp.push("    3)")
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push("    y = 3")
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more5(self):
        isp = self.isp
        isp.push('try:')
        isp.push('    a = 5')
        isp.push('except:')
        isp.push('    raise')
        # We want to be able to add an else: block at this point, so it should
        # wait for a blank line.
        self.assertEqual(isp.push_accepts_more(), True)

    def test_continuation(self):
        isp = self.isp
        isp.push("import os, \\")
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push("sys")
        self.assertEqual(isp.push_accepts_more(), False)

    def test_syntax_error(self):
        isp = self.isp
        # Syntax errors immediately produce a 'ready' block, so the invalid
        # Python can be sent to the kernel for evaluation with possible ipython
        # special-syntax conversion.
        isp.push('run foo')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_unicode(self):
        self.isp.push(u"Pérez")
        self.isp.push(u'\xc3\xa9')
        self.isp.push(u"u'\xc3\xa9'")

    @pytest.mark.xfail(
        reason="Bug in python 3.9.8 – bpo 45738",
        condition=sys.version_info in [(3, 11, 0, "alpha", 2)],
        raises=SystemError,
        strict=True,
    )
    def test_line_continuation(self):
        """ Test issue #2108."""
        isp = self.isp
        # A blank line after a line continuation should not accept more
        isp.push("1 \\\n\n")
        self.assertEqual(isp.push_accepts_more(), False)
        # Whitespace after a \ is a SyntaxError.  The only way to test that
        # here is to test that push doesn't accept more (as with
        # test_syntax_error() above).
        isp.push(r"1 \ ")
        self.assertEqual(isp.push_accepts_more(), False)
        # Even if the line is continuable (c.f. the regular Python
        # interpreter)
        isp.push(r"(1 \ ")
        self.assertEqual(isp.push_accepts_more(), False)

    def test_check_complete(self):
        isp = self.isp
        self.assertEqual(isp.check_complete("a = 1"), ('complete', None))
        self.assertEqual(isp.check_complete("for a in range(5):"), ('incomplete', 4))
        self.assertEqual(isp.check_complete("raise = 2"), ('invalid', None))
        self.assertEqual(isp.check_complete("a = [1,\n2,"), ('incomplete', 0))
        self.assertEqual(isp.check_complete("def a():\n x=1\n global x"), ('invalid', None))

class InteractiveLoopTestCase(unittest.TestCase):
    """Tests for an interactive loop like a python shell.
    """
    def check_ns(self, lines, ns):
        """Validate that the given input lines produce the resulting namespace.

        Note: the input lines are given exactly as they would be typed in an
        auto-indenting environment, as mini_interactive_loop above already does
        auto-indenting and prepends spaces to the input.
        """
        src = mini_interactive_loop(pseudo_input(lines))
        test_ns = {}
        exec(src, test_ns)
        # We can't check that the provided ns is identical to the test_ns,
        # because Python fills test_ns with extra keys (copyright, etc).  But
        # we can check that the given dict is *contained* in test_ns
        for k,v in ns.items():
            self.assertEqual(test_ns[k], v)

    def test_simple(self):
        self.check_ns(['x=1'], dict(x=1))

    def test_simple2(self):
        self.check_ns(['if 1:', 'x=2'], dict(x=2))

    def test_xy(self):
        self.check_ns(['x=1; y=2'], dict(x=1, y=2))

    def test_abc(self):
        self.check_ns(['if 1:','a=1','b=2','c=3'], dict(a=1, b=2, c=3))

    def test_multi(self):
        self.check_ns(['x =(1+','1+','2)'], dict(x=4))


class IPythonInputTestCase(InputSplitterTestCase):
    """By just creating a new class whose .isp is a different instance, we
    re-run the same test battery on the new input splitter.

    In addition, this runs the tests over the syntax and syntax_ml dicts that
    were tested by individual functions, as part of the OO interface.

    It also makes some checks on the raw buffer storage.
    """

    def setUp(self):
        self.isp = isp.IPythonInputSplitter()

    def test_syntax(self):
        """Call all single-line syntax tests from the main object"""
        isp = self.isp
        for example in syntax.values():
            for raw, out_t in example:
                if raw.startswith(' '):
                    continue

                isp.push(raw+'\n')
                out_raw = isp.source_raw
                out = isp.source_reset()
                self.assertEqual(out.rstrip(), out_t,
                        tt.pair_fail_msg.format("inputsplitter",raw, out_t, out))
                self.assertEqual(out_raw.rstrip(), raw.rstrip())

    def test_syntax_multiline(self):
        isp = self.isp
        for example in syntax_ml.values():
            for line_pairs in example:
                out_t_parts = []
                raw_parts = []
                for lraw, out_t_part in line_pairs:
                    if out_t_part is not None:
                        out_t_parts.append(out_t_part)

                    if lraw is not None:
                        isp.push(lraw)
                        raw_parts.append(lraw)

                out_raw = isp.source_raw
                out = isp.source_reset()
                out_t = '\n'.join(out_t_parts).rstrip()
                raw = '\n'.join(raw_parts).rstrip()
                self.assertEqual(out.rstrip(), out_t)
                self.assertEqual(out_raw.rstrip(), raw)

    def test_syntax_multiline_cell(self):
        isp = self.isp
        for example in syntax_ml.values():

            out_t_parts = []
            for line_pairs in example:
                raw = '\n'.join(r for r, _ in line_pairs if r is not None)
                out_t = '\n'.join(t for _,t in line_pairs if t is not None)
                out = isp.transform_cell(raw)
                # Match ignoring trailing whitespace
                self.assertEqual(out.rstrip(), out_t.rstrip())

    def test_cellmagic_preempt(self):
        isp = self.isp
        for raw, name, line, cell in [
            ("%%cellm a\nIn[1]:", u'cellm', u'a', u'In[1]:'),
            ("%%cellm \nline\n>>> hi", u'cellm', u'', u'line\n>>> hi'),
            (">>> %%cellm \nline\n>>> hi", u'cellm', u'', u'line\nhi'),
            ("%%cellm \n>>> hi", u'cellm', u'', u'>>> hi'),
            ("%%cellm \nline1\nline2", u'cellm', u'', u'line1\nline2'),
            ("%%cellm \nline1\\\\\nline2", u'cellm', u'', u'line1\\\\\nline2'),
        ]:
            expected = "get_ipython().run_cell_magic(%r, %r, %r)" % (
                name, line, cell
            )
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())

    def test_multiline_passthrough(self):
        isp = self.isp
        class CommentTransformer(InputTransformer):
            def __init__(self):
                self._lines = []

            def push(self, line):
                self._lines.append(line + '#')

            def reset(self):
                text = '\n'.join(self._lines)
                self._lines = []
                return text

        isp.physical_line_transforms.insert(0, CommentTransformer())

        for raw, expected in [
            ("a=5", "a=5#"),
            ("%ls foo", "get_ipython().run_line_magic(%r, %r)" % (u'ls', u'foo#')),
            ("!ls foo\n%ls bar", "get_ipython().system(%r)\nget_ipython().run_line_magic(%r, %r)" % (
                u'ls foo#', u'ls', u'bar#'
            )),
            ("1\n2\n3\n%ls foo\n4\n5", "1#\n2#\n3#\nget_ipython().run_line_magic(%r, %r)\n4#\n5#" % (u'ls', u'foo#')),
        ]:
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())

#-----------------------------------------------------------------------------
# Main - use as a script, mostly for developer experiments
#-----------------------------------------------------------------------------

if __name__ == '__main__':
    # A simple demo for interactive experimentation.  This code will not get
    # picked up by any test suite.
    from IPython.core.inputsplitter import IPythonInputSplitter

    # configure here the syntax to use, prompt and whether to autoindent
    #isp, start_prompt = InputSplitter(), '>>> '
    isp, start_prompt = IPythonInputSplitter(), 'In> '

    autoindent = True
    #autoindent = False

    try:
        while True:
            prompt = start_prompt
            while isp.push_accepts_more():
                indent = ' '*isp.get_indent_spaces()
                if autoindent:
                    line = indent + input(prompt+indent)
                else:
                    line = input(prompt)
                isp.push(line)
                prompt = '... '

            # Here we just return input so we can use it in a test suite, but a
            # real interpreter would instead send it for execution somewhere.
            #src = isp.source; raise EOFError # dbg
            raw = isp.source_raw
            src = isp.source_reset()
            print('Input source was:\n', src)
            print('Raw source was:\n', raw)
    except EOFError:
        print('Bye')

# Tests for cell magics support

def test_last_blank():
    assert isp.last_blank("") is False
    assert isp.last_blank("abc") is False
    assert isp.last_blank("abc\n") is False
    assert isp.last_blank("abc\na") is False

    assert isp.last_blank("\n") is True
    assert isp.last_blank("\n ") is True
    assert isp.last_blank("abc\n ") is True
    assert isp.last_blank("abc\n\n") is True
    assert isp.last_blank("abc\nd\n\n") is True
    assert isp.last_blank("abc\nd\ne\n\n") is True
    assert isp.last_blank("abc \n \n \n\n") is True


def test_last_two_blanks():
    assert isp.last_two_blanks("") is False
    assert isp.last_two_blanks("abc") is False
    assert isp.last_two_blanks("abc\n") is False
    assert isp.last_two_blanks("abc\n\na") is False
    assert isp.last_two_blanks("abc\n \n") is False
    assert isp.last_two_blanks("abc\n\n") is False

    assert isp.last_two_blanks("\n\n") is True
    assert isp.last_two_blanks("\n\n ") is True
    assert isp.last_two_blanks("\n \n") is True
    assert isp.last_two_blanks("abc\n\n ") is True
    assert isp.last_two_blanks("abc\n\n\n") is True
    assert isp.last_two_blanks("abc\n\n \n") is True
    assert isp.last_two_blanks("abc\n\n \n ") is True
    assert isp.last_two_blanks("abc\n\n \n \n") is True
    assert isp.last_two_blanks("abc\nd\n\n\n") is True
    assert isp.last_two_blanks("abc\nd\ne\nf\n\n\n") is True


class CellMagicsCommon(object):

    def test_whole_cell(self):
        src = "%%cellm line\nbody\n"
        out = self.sp.transform_cell(src)
        ref = "get_ipython().run_cell_magic('cellm', 'line', 'body')\n"
        assert out == ref

    def test_cellmagic_help(self):
        self.sp.push('%%cellm?')
        assert self.sp.push_accepts_more() is False

    def tearDown(self):
        self.sp.reset()


class CellModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=False)

    def test_incremental(self):
        sp = self.sp
        sp.push("%%cellm firstline\n")
        assert sp.push_accepts_more() is True  # 1
        sp.push("line2\n")
        assert sp.push_accepts_more() is True  # 2
        sp.push("\n")
        # This should accept a blank line and carry on until the cell is reset
        assert sp.push_accepts_more() is True  # 3

    def test_no_strip_coding(self):
        src = '\n'.join([
            '%%writefile foo.py',
            '# coding: utf-8',
            'print(u"üñîçø∂é")',
        ])
        out = self.sp.transform_cell(src)
        assert "# coding: utf-8" in out


class LineModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=True)

    def test_incremental(self):
        sp = self.sp
        sp.push("%%cellm line2\n")
        assert sp.push_accepts_more() is True  # 1
        sp.push("\n")
        # In this case, a blank line should end the cell magic
        assert sp.push_accepts_more() is False  # 2


indentation_samples = [
    ('a = 1', 0),
    ('for a in b:', 4),
    ('def f():', 4),
    ('def f(): #comment', 4),
    ('a = ":#not a comment"', 0),
    ('def f():\n    a = 1', 4),
    ('def f():\n    return 1', 0),
    ('for a in b:\n'
     '   if a < 0:'
     '       continue', 3),
    ('a = {', 4),
    ('a = {\n'
     '     1,', 5),
    ('b = """123', 0),
    ('', 0),
    ('def f():\n    pass', 0),
    ('class Bar:\n    def f():\n        pass', 4),
    ('class Bar:\n    def f():\n        raise', 4),
]

def test_find_next_indent():
    for code, exp in indentation_samples:
        res = isp.find_next_indent(code)
        msg = "{!r} != {!r} (expected)\n Code: {!r}".format(res, exp, code)
        assert res == exp, msg
