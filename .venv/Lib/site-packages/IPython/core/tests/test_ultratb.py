# encoding: utf-8
"""Tests for IPython.core.ultratb
"""
import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent

from tempfile import TemporaryDirectory

from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths
from IPython.utils.syspathcontext import prepended_to_syspath

file_1 = """1
2
3
def f():
  1/0
"""

file_2 = """def f():
  1/0
"""


def recursionlimit(frames):
    """
    decorator to set the recursion limit temporarily
    """

    def inner(test_function):
        def wrapper(*args, **kwargs):
            rl = sys.getrecursionlimit()
            sys.setrecursionlimit(frames)
            try:
                return test_function(*args, **kwargs)
            finally:
                sys.setrecursionlimit(rl)

        return wrapper

    return inner


class ChangedPyFileTest(unittest.TestCase):
    def test_changing_py_file(self):
        """Traceback produced if the line where the error occurred is missing?

        https://github.com/ipython/ipython/issues/1456
        """
        with TemporaryDirectory() as td:
            fname = os.path.join(td, "foo.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(file_1)

            with prepended_to_syspath(td):
                ip.run_cell("import foo")

            with tt.AssertPrints("ZeroDivisionError"):
                ip.run_cell("foo.f()")

            # Make the file shorter, so the line of the error is missing.
            with open(fname, "w", encoding="utf-8") as f:
                f.write(file_2)

            # For some reason, this was failing on the *second* call after
            # changing the file, so we call f() twice.
            with tt.AssertNotPrints("Internal Python error", channel='stderr'):
                with tt.AssertPrints("ZeroDivisionError"):
                    ip.run_cell("foo.f()")
                with tt.AssertPrints("ZeroDivisionError"):
                    ip.run_cell("foo.f()")

iso_8859_5_file = u'''# coding: iso-8859-5

def fail():
    """дбИЖ"""
    1/0     # дбИЖ
'''

class NonAsciiTest(unittest.TestCase):
    @onlyif_unicode_paths
    def test_nonascii_path(self):
        # Non-ascii directory name as well.
        with TemporaryDirectory(suffix=u'é') as td:
            fname = os.path.join(td, u"fooé.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(file_1)

            with prepended_to_syspath(td):
                ip.run_cell("import foo")

            with tt.AssertPrints("ZeroDivisionError"):
                ip.run_cell("foo.f()")

    def test_iso8859_5(self):
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'dfghjkl.py')

            with io.open(fname, 'w', encoding='iso-8859-5') as f:
                f.write(iso_8859_5_file)

            with prepended_to_syspath(td):
                ip.run_cell("from dfghjkl import fail")

            with tt.AssertPrints("ZeroDivisionError"):
                with tt.AssertPrints(u'дбИЖ', suppress=False):
                    ip.run_cell('fail()')

    def test_nonascii_msg(self):
        cell = u"raise Exception('é')"
        expected = u"Exception('é')"
        ip.run_cell("%xmode plain")
        with tt.AssertPrints(expected):
            ip.run_cell(cell)

        ip.run_cell("%xmode verbose")
        with tt.AssertPrints(expected):
            ip.run_cell(cell)

        ip.run_cell("%xmode context")
        with tt.AssertPrints(expected):
            ip.run_cell(cell)

        ip.run_cell("%xmode minimal")
        with tt.AssertPrints(u"Exception: é"):
            ip.run_cell(cell)

        # Put this back into Context mode for later tests.
        ip.run_cell("%xmode context")

class NestedGenExprTestCase(unittest.TestCase):
    """
    Regression test for the following issues:
    https://github.com/ipython/ipython/issues/8293
    https://github.com/ipython/ipython/issues/8205
    """
    def test_nested_genexpr(self):
        code = dedent(
            """\
            class SpecificException(Exception):
                pass

            def foo(x):
                raise SpecificException("Success!")

            sum(sum(foo(x) for _ in [0]) for x in [0])
            """
        )
        with tt.AssertPrints('SpecificException: Success!', suppress=False):
            ip.run_cell(code)


indentationerror_file = """if True:
zoon()
"""

class IndentationErrorTest(unittest.TestCase):
    def test_indentationerror_shows_line(self):
        # See issue gh-2398
        with tt.AssertPrints("IndentationError"):
            with tt.AssertPrints("zoon()", suppress=False):
                ip.run_cell(indentationerror_file)

        with TemporaryDirectory() as td:
            fname = os.path.join(td, "foo.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(indentationerror_file)

            with tt.AssertPrints("IndentationError"):
                with tt.AssertPrints("zoon()", suppress=False):
                    ip.magic('run %s' % fname)

se_file_1 = """1
2
7/
"""

se_file_2 = """7/
"""

class SyntaxErrorTest(unittest.TestCase):

    def test_syntaxerror_no_stacktrace_at_compile_time(self):
        syntax_error_at_compile_time = """
def foo():
    ..
"""
        with tt.AssertPrints("SyntaxError"):
            ip.run_cell(syntax_error_at_compile_time)

        with tt.AssertNotPrints("foo()"):
            ip.run_cell(syntax_error_at_compile_time)

    def test_syntaxerror_stacktrace_when_running_compiled_code(self):
        syntax_error_at_runtime = """
def foo():
    eval("..")

def bar():
    foo()

bar()
"""
        with tt.AssertPrints("SyntaxError"):
            ip.run_cell(syntax_error_at_runtime)
        # Assert syntax error during runtime generate stacktrace
        with tt.AssertPrints(["foo()", "bar()"]):
            ip.run_cell(syntax_error_at_runtime)
        del ip.user_ns['bar']
        del ip.user_ns['foo']

    def test_changing_py_file(self):
        with TemporaryDirectory() as td:
            fname = os.path.join(td, "foo.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(se_file_1)

            with tt.AssertPrints(["7/", "SyntaxError"]):
                ip.magic("run " + fname)

            # Modify the file
            with open(fname, "w", encoding="utf-8") as f:
                f.write(se_file_2)

            # The SyntaxError should point to the correct line
            with tt.AssertPrints(["7/", "SyntaxError"]):
                ip.magic("run " + fname)

    def test_non_syntaxerror(self):
        # SyntaxTB may be called with an error other than a SyntaxError
        # See e.g. gh-4361
        try:
            raise ValueError('QWERTY')
        except ValueError:
            with tt.AssertPrints('QWERTY'):
                ip.showsyntaxerror()

import sys

if platform.python_implementation() != "PyPy":
    """
    New 3.9 Pgen Parser does not raise Memory error, except on failed malloc.
    """
    class MemoryErrorTest(unittest.TestCase):
        def test_memoryerror(self):
            memoryerror_code = "(" * 200 + ")" * 200
            ip.run_cell(memoryerror_code)


class Python3ChainedExceptionsTest(unittest.TestCase):
    DIRECT_CAUSE_ERROR_CODE = """
try:
    x = 1 + 2
    print(not_defined_here)
except Exception as e:
    x += 55
    x - 1
    y = {}
    raise KeyError('uh') from e
    """

    EXCEPTION_DURING_HANDLING_CODE = """
try:
    x = 1 + 2
    print(not_defined_here)
except Exception as e:
    x += 55
    x - 1
    y = {}
    raise KeyError('uh')
    """

    SUPPRESS_CHAINING_CODE = """
try:
    1/0
except Exception:
    raise ValueError("Yikes") from None
    """

    def test_direct_cause_error(self):
        with tt.AssertPrints(["KeyError", "NameError", "direct cause"]):
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)

    def test_exception_during_handling_error(self):
        with tt.AssertPrints(["KeyError", "NameError", "During handling"]):
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)

    def test_suppress_exception_chaining(self):
        with tt.AssertNotPrints("ZeroDivisionError"), \
             tt.AssertPrints("ValueError", suppress=False):
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)

    def test_plain_direct_cause_error(self):
        with tt.AssertPrints(["KeyError", "NameError", "direct cause"]):
            ip.run_cell("%xmode Plain")
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)
            ip.run_cell("%xmode Verbose")

    def test_plain_exception_during_handling_error(self):
        with tt.AssertPrints(["KeyError", "NameError", "During handling"]):
            ip.run_cell("%xmode Plain")
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)
            ip.run_cell("%xmode Verbose")

    def test_plain_suppress_exception_chaining(self):
        with tt.AssertNotPrints("ZeroDivisionError"), \
             tt.AssertPrints("ValueError", suppress=False):
            ip.run_cell("%xmode Plain")
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)
            ip.run_cell("%xmode Verbose")


class RecursionTest(unittest.TestCase):
    DEFINITIONS = """
def non_recurs():
    1/0

def r1():
    r1()

def r3a():
    r3b()

def r3b():
    r3c()

def r3c():
    r3a()

def r3o1():
    r3a()

def r3o2():
    r3o1()
"""
    def setUp(self):
        ip.run_cell(self.DEFINITIONS)

    def test_no_recursion(self):
        with tt.AssertNotPrints("skipping similar frames"):
            ip.run_cell("non_recurs()")

    @recursionlimit(200)
    def test_recursion_one_frame(self):
        with tt.AssertPrints(re.compile(
            r"\[\.\.\. skipping similar frames: r1 at line 5 \(\d{2,3} times\)\]")
        ):
            ip.run_cell("r1()")

    @recursionlimit(160)
    def test_recursion_three_frames(self):
        with tt.AssertPrints("[... skipping similar frames: "), \
                tt.AssertPrints(re.compile(r"r3a at line 8 \(\d{2} times\)"), suppress=False), \
                tt.AssertPrints(re.compile(r"r3b at line 11 \(\d{2} times\)"), suppress=False), \
                tt.AssertPrints(re.compile(r"r3c at line 14 \(\d{2} times\)"), suppress=False):
            ip.run_cell("r3o2()")


class PEP678NotesReportingTest(unittest.TestCase):
    ERROR_WITH_NOTE = """
try:
    raise AssertionError("Message")
except Exception as e:
    try:
        e.add_note("This is a PEP-678 note.")
    except AttributeError:  # Python <= 3.10
        e.__notes__ = ("This is a PEP-678 note.",)
    raise
    """

    def test_verbose_reports_notes(self):
        with tt.AssertPrints(["AssertionError", "Message", "This is a PEP-678 note."]):
            ip.run_cell(self.ERROR_WITH_NOTE)

    def test_plain_reports_notes(self):
        with tt.AssertPrints(["AssertionError", "Message", "This is a PEP-678 note."]):
            ip.run_cell("%xmode Plain")
            ip.run_cell(self.ERROR_WITH_NOTE)
            ip.run_cell("%xmode Verbose")


#----------------------------------------------------------------------------

# module testing (minimal)
def test_handlers():
    def spam(c, d_e):
        (d, e) = d_e
        x = c + d
        y = c * d
        foo(x, y)

    def foo(a, b, bar=1):
        eggs(a, b + bar)

    def eggs(f, g, z=globals()):
        h = f + g
        i = f - g
        return h / i

    buff = io.StringIO()

    buff.write('')
    buff.write('*** Before ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        traceback.print_exc(file=buff)

    handler = ColorTB(ostream=buff)
    buff.write('*** ColorTB ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        handler(*sys.exc_info())
    buff.write('')

    handler = VerboseTB(ostream=buff)
    buff.write('*** VerboseTB ***')
    try:
        buff.write(spam(1, (2, 3)))
    except:
        handler(*sys.exc_info())
    buff.write('')
