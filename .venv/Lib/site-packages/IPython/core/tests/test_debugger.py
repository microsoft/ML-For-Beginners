"""Tests for debugging machinery.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import builtins
import os
import sys
import platform

from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch

from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest

#-----------------------------------------------------------------------------
# Helper classes, from CPython's Pdb test suite
#-----------------------------------------------------------------------------

class _FakeInput(object):
    """
    A fake input stream for pdb's interactive debugger.  Whenever a
    line is read, print it (to simulate the user typing it), and then
    return it.  The set of lines to return is specified in the
    constructor; they should not have trailing newlines.
    """
    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self):
        line = next(self.lines)
        print(line)
        return line+'\n'

class PdbTestInput(object):
    """Context manager that makes testing Pdb in doctests easier."""

    def __init__(self, input):
        self.input = input

    def __enter__(self):
        self.real_stdin = sys.stdin
        sys.stdin = _FakeInput(self.input)

    def __exit__(self, *exc):
        sys.stdin = self.real_stdin

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

def test_ipdb_magics():
    '''Test calling some IPython magics from ipdb.

    First, set up some test functions and classes which we can inspect.

    >>> class ExampleClass(object):
    ...    """Docstring for ExampleClass."""
    ...    def __init__(self):
    ...        """Docstring for ExampleClass.__init__"""
    ...        pass
    ...    def __str__(self):
    ...        return "ExampleClass()"

    >>> def example_function(x, y, z="hello"):
    ...     """Docstring for example_function."""
    ...     pass

    >>> old_trace = sys.gettrace()

    Create a function which triggers ipdb.

    >>> def trigger_ipdb():
    ...    a = ExampleClass()
    ...    debugger.Pdb().set_trace()

    >>> with PdbTestInput([
    ...    'pdef example_function',
    ...    'pdoc ExampleClass',
    ...    'up',
    ...    'down',
    ...    'list',
    ...    'pinfo a',
    ...    'll',
    ...    'continue',
    ... ]):
    ...     trigger_ipdb()
    --Return--
    None
    > <doctest ...>(3)trigger_ipdb()
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> pdef example_function
     example_function(x, y, z='hello')
     ipdb> pdoc ExampleClass
    Class docstring:
        Docstring for ExampleClass.
    Init docstring:
        Docstring for ExampleClass.__init__
    ipdb> up
    > <doctest ...>(11)<module>()
          7    'pinfo a',
          8    'll',
          9    'continue',
         10 ]):
    ---> 11     trigger_ipdb()
    <BLANKLINE>
    ipdb> down
    None
    > <doctest ...>(3)trigger_ipdb()
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> list
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> pinfo a
    Type:           ExampleClass
    String form:    ExampleClass()
    Namespace:      Local...
    Docstring:      Docstring for ExampleClass.
    Init docstring: Docstring for ExampleClass.__init__
    ipdb> ll
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> continue
    
    Restore previous trace function, e.g. for coverage.py    
    
    >>> sys.settrace(old_trace)
    '''

def test_ipdb_magics2():
    '''Test ipdb with a very short function.
    
    >>> old_trace = sys.gettrace()

    >>> def bar():
    ...     pass

    Run ipdb.

    >>> with PdbTestInput([
    ...    'continue',
    ... ]):
    ...     debugger.Pdb().runcall(bar)
    > <doctest ...>(2)bar()
          1 def bar():
    ----> 2    pass
    <BLANKLINE>
    ipdb> continue
    
    Restore previous trace function, e.g. for coverage.py    
    
    >>> sys.settrace(old_trace)
    '''

def can_quit():
    '''Test that quit work in ipydb

    >>> old_trace = sys.gettrace()

    >>> def bar():
    ...     pass

    >>> with PdbTestInput([
    ...    'quit',
    ... ]):
    ...     debugger.Pdb().runcall(bar)
    > <doctest ...>(2)bar()
            1 def bar():
    ----> 2    pass
    <BLANKLINE>
    ipdb> quit

    Restore previous trace function, e.g. for coverage.py

    >>> sys.settrace(old_trace)
    '''


def can_exit():
    '''Test that quit work in ipydb

    >>> old_trace = sys.gettrace()

    >>> def bar():
    ...     pass

    >>> with PdbTestInput([
    ...    'exit',
    ... ]):
    ...     debugger.Pdb().runcall(bar)
    > <doctest ...>(2)bar()
            1 def bar():
    ----> 2    pass
    <BLANKLINE>
    ipdb> exit

    Restore previous trace function, e.g. for coverage.py

    >>> sys.settrace(old_trace)
    '''


def test_interruptible_core_debugger():
    """The debugger can be interrupted.

    The presumption is there is some mechanism that causes a KeyboardInterrupt
    (this is implemented in ipykernel).  We want to ensure the
    KeyboardInterrupt cause debugging to cease.
    """
    def raising_input(msg="", called=[0]):
        called[0] += 1
        assert called[0] == 1, "input() should only be called once!"
        raise KeyboardInterrupt()

    tracer_orig = sys.gettrace()
    try:
        with patch.object(builtins, "input", raising_input):
            debugger.InterruptiblePdb().set_trace()
            # The way this test will fail is by set_trace() never exiting,
            # resulting in a timeout by the test runner. The alternative
            # implementation would involve a subprocess, but that adds issues
            # with interrupting subprocesses that are rather complex, so it's
            # simpler just to do it this way.
    finally:
        # restore the original trace function
        sys.settrace(tracer_orig)


@skip_win32
def test_xmode_skip():
    """that xmode skip frames

    Not as a doctest as pytest does not run doctests.
    """
    import pexpect
    env = os.environ.copy()
    env["IPY_TEST_SIMPLE_PROMPT"] = "1"

    child = pexpect.spawn(
        sys.executable, ["-m", "IPython", "--colors=nocolor"], env=env
    )
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.expect("IPython")
    child.expect("\n")
    child.expect_exact("In [1]")

    block = dedent(
        """
    def f():
        __tracebackhide__ = True
        g()

    def g():
        raise ValueError

    f()
    """
    )

    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact("skipping")

    block = dedent(
        """
    def f():
        __tracebackhide__ = True
        g()

    def g():
        from IPython.core.debugger import set_trace
        set_trace()

    f()
    """
    )

    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)

    child.expect("ipdb>")
    child.sendline("w")
    child.expect("hidden")
    child.expect("ipdb>")
    child.sendline("skip_hidden false")
    child.sendline("w")
    child.expect("__traceba")
    child.expect("ipdb>")

    child.close()


skip_decorators_blocks = (
    """
    def helpers_helper():
        pass # should not stop here except breakpoint
    """,
    """
    def helper_1():
        helpers_helper() # should not stop here
    """,
    """
    def helper_2():
        pass # should not stop here
    """,
    """
    def pdb_skipped_decorator2(function):
        def wrapped_fn(*args, **kwargs):
            __debuggerskip__ = True
            helper_2()
            __debuggerskip__ = False
            result = function(*args, **kwargs)
            __debuggerskip__ = True
            helper_2()
            return result
        return wrapped_fn
    """,
    """
    def pdb_skipped_decorator(function):
        def wrapped_fn(*args, **kwargs):
            __debuggerskip__ = True
            helper_1()
            __debuggerskip__ = False
            result = function(*args, **kwargs)
            __debuggerskip__ = True
            helper_2()
            return result
        return wrapped_fn
    """,
    """
    @pdb_skipped_decorator
    @pdb_skipped_decorator2
    def bar(x, y):
        return x * y
    """,
    """import IPython.terminal.debugger as ipdb""",
    """
    def f():
        ipdb.set_trace()
        bar(3, 4)
    """,
    """
    f()
    """,
)


def _decorator_skip_setup():
    import pexpect

    env = os.environ.copy()
    env["IPY_TEST_SIMPLE_PROMPT"] = "1"
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"

    child = pexpect.spawn(
        sys.executable, ["-m", "IPython", "--colors=nocolor"], env=env
    )
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.expect("IPython")
    child.expect("\n")

    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500

    dedented_blocks = [dedent(b).strip() for b in skip_decorators_blocks]
    in_prompt_number = 1
    for cblock in dedented_blocks:
        child.expect_exact(f"In [{in_prompt_number}]:")
        in_prompt_number += 1
        for line in cblock.splitlines():
            child.sendline(line)
            child.expect_exact(line)
        child.sendline("")
    return child


@pytest.mark.skip(reason="recently fail for unknown reason on CI")
@skip_win32
def test_decorator_skip():
    """test that decorator frames can be skipped."""

    child = _decorator_skip_setup()

    child.expect_exact("ipython-input-8")
    child.expect_exact("3     bar(3, 4)")
    child.expect("ipdb>")

    child.expect("ipdb>")
    child.sendline("step")
    child.expect_exact("step")
    child.expect_exact("--Call--")
    child.expect_exact("ipython-input-6")

    child.expect_exact("1 @pdb_skipped_decorator")

    child.sendline("s")
    child.expect_exact("return x * y")

    child.close()


@pytest.mark.skip(reason="recently fail for unknown reason on CI")
@pytest.mark.skipif(platform.python_implementation() == "PyPy", reason="issues on PyPy")
@skip_win32
def test_decorator_skip_disabled():
    """test that decorator frame skipping can be disabled"""

    child = _decorator_skip_setup()

    child.expect_exact("3     bar(3, 4)")

    for input_, expected in [
        ("skip_predicates debuggerskip False", ""),
        ("skip_predicates", "debuggerskip : False"),
        ("step", "---> 2     def wrapped_fn"),
        ("step", "----> 3         __debuggerskip__"),
        ("step", "----> 4         helper_1()"),
        ("step", "---> 1 def helper_1():"),
        ("next", "----> 2     helpers_helper()"),
        ("next", "--Return--"),
        ("next", "----> 5         __debuggerskip__ = False"),
    ]:
        child.expect("ipdb>")
        child.sendline(input_)
        child.expect_exact(input_)
        child.expect_exact(expected)

    child.close()


@pytest.mark.skipif(platform.python_implementation() == "PyPy", reason="issues on PyPy")
@skip_win32
def test_decorator_skip_with_breakpoint():
    """test that decorator frame skipping can be disabled"""

    import pexpect

    env = os.environ.copy()
    env["IPY_TEST_SIMPLE_PROMPT"] = "1"
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"

    child = pexpect.spawn(
        sys.executable, ["-m", "IPython", "--colors=nocolor"], env=env
    )
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500

    child.expect("IPython")
    child.expect("\n")

    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE

    ### we need a filename, so we need to exec the full block with a filename
    with NamedTemporaryFile(suffix=".py", dir=".", delete=True) as tf:
        name = tf.name[:-3].split("/")[-1]
        tf.write("\n".join([dedent(x) for x in skip_decorators_blocks[:-1]]).encode())
        tf.flush()
        codeblock = f"from {name} import f"

        dedented_blocks = [
            codeblock,
            "f()",
        ]

        in_prompt_number = 1
        for cblock in dedented_blocks:
            child.expect_exact(f"In [{in_prompt_number}]:")
            in_prompt_number += 1
            for line in cblock.splitlines():
                child.sendline(line)
                child.expect_exact(line)
            child.sendline("")

        # as the filename does not exists, we'll rely on the filename prompt
        child.expect_exact("47     bar(3, 4)")

        for input_, expected in [
            (f"b {name}.py:3", ""),
            ("step", "1---> 3     pass # should not stop here except"),
            ("step", "---> 38 @pdb_skipped_decorator"),
            ("continue", ""),
        ]:
            child.expect("ipdb>")
            child.sendline(input_)
            child.expect_exact(input_)
            child.expect_exact(expected)

    child.close()


@skip_win32
def test_where_erase_value():
    """Test that `where` does not access f_locals and erase values."""
    import pexpect

    env = os.environ.copy()
    env["IPY_TEST_SIMPLE_PROMPT"] = "1"

    child = pexpect.spawn(
        sys.executable, ["-m", "IPython", "--colors=nocolor"], env=env
    )
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE

    child.expect("IPython")
    child.expect("\n")
    child.expect_exact("In [1]")

    block = dedent(
        """
    def simple_f():
         myvar = 1
         print(myvar)
         1/0
         print(myvar)
    simple_f()    """
    )

    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact("ZeroDivisionError")
    child.expect_exact("In [2]:")

    child.sendline("%debug")

    ##
    child.expect("ipdb>")

    child.sendline("myvar")
    child.expect("1")

    ##
    child.expect("ipdb>")

    child.sendline("myvar = 2")

    ##
    child.expect_exact("ipdb>")

    child.sendline("myvar")

    child.expect_exact("2")

    ##
    child.expect("ipdb>")
    child.sendline("where")

    ##
    child.expect("ipdb>")
    child.sendline("myvar")

    child.expect_exact("2")
    child.expect("ipdb>")

    child.close()
