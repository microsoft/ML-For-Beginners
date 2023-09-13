# -*- coding: utf-8 -*-
"""Tests for the key interactiveshell module.

Historically the main classes in interactiveshell have been under-tested.  This
module should grow as many single-method tests as possible to trap many of the
recurring bugs we seem to encounter with high-level interaction.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock

from os.path import join

from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
    skipif, skip_win32, onlyif_unicode_paths, onlyif_cmds_exist,
)
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------
# This is used by every single test, no point repeating it ad nauseam

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

class DerivedInterrupt(KeyboardInterrupt):
    pass

class InteractiveShellTestCase(unittest.TestCase):
    def test_naked_string_cells(self):
        """Test that cells with only naked strings are fully executed"""
        # First, single-line inputs
        ip.run_cell('"a"\n')
        self.assertEqual(ip.user_ns['_'], 'a')
        # And also multi-line cells
        ip.run_cell('"""a\nb"""\n')
        self.assertEqual(ip.user_ns['_'], 'a\nb')

    def test_run_empty_cell(self):
        """Just make sure we don't get a horrible error with a blank
        cell of input. Yes, I did overlook that."""
        old_xc = ip.execution_count
        res = ip.run_cell('')
        self.assertEqual(ip.execution_count, old_xc)
        self.assertEqual(res.execution_count, None)

    def test_run_cell_multiline(self):
        """Multi-block, multi-line cells must execute correctly.
        """
        src = '\n'.join(["x=1",
                         "y=2",
                         "if 1:",
                         "    x += 1",
                         "    y += 1",])
        res = ip.run_cell(src)
        self.assertEqual(ip.user_ns['x'], 2)
        self.assertEqual(ip.user_ns['y'], 3)
        self.assertEqual(res.success, True)
        self.assertEqual(res.result, None)

    def test_multiline_string_cells(self):
        "Code sprinkled with multiline strings should execute (GH-306)"
        ip.run_cell('tmp=0')
        self.assertEqual(ip.user_ns['tmp'], 0)
        res = ip.run_cell('tmp=1;"""a\nb"""\n')
        self.assertEqual(ip.user_ns['tmp'], 1)
        self.assertEqual(res.success, True)
        self.assertEqual(res.result, "a\nb")

    def test_dont_cache_with_semicolon(self):
        "Ending a line with semicolon should not cache the returned object (GH-307)"
        oldlen = len(ip.user_ns['Out'])
        for cell in ['1;', '1;1;']:
            res = ip.run_cell(cell, store_history=True)
            newlen = len(ip.user_ns['Out'])
            self.assertEqual(oldlen, newlen)
            self.assertIsNone(res.result)
        i = 0
        #also test the default caching behavior
        for cell in ['1', '1;1']:
            ip.run_cell(cell, store_history=True)
            newlen = len(ip.user_ns['Out'])
            i += 1
            self.assertEqual(oldlen+i, newlen)

    def test_syntax_error(self):
        res = ip.run_cell("raise = 3")
        self.assertIsInstance(res.error_before_exec, SyntaxError)

    def test_open_standard_input_stream(self):
        res = ip.run_cell("open(0)")
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_open_standard_output_stream(self):
        res = ip.run_cell("open(1)")
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_open_standard_error_stream(self):
        res = ip.run_cell("open(2)")
        self.assertIsInstance(res.error_in_exec, ValueError)

    def test_In_variable(self):
        "Verify that In variable grows with user input (GH-284)"
        oldlen = len(ip.user_ns['In'])
        ip.run_cell('1;', store_history=True)
        newlen = len(ip.user_ns['In'])
        self.assertEqual(oldlen+1, newlen)
        self.assertEqual(ip.user_ns['In'][-1],'1;')
        
    def test_magic_names_in_string(self):
        ip.run_cell('a = """\n%exit\n"""')
        self.assertEqual(ip.user_ns['a'], '\n%exit\n')
    
    def test_trailing_newline(self):
        """test that running !(command) does not raise a SyntaxError"""
        ip.run_cell('!(true)\n', False)
        ip.run_cell('!(true)\n\n\n', False)
    
    def test_gh_597(self):
        """Pretty-printing lists of objects with non-ascii reprs may cause
        problems."""
        class Spam(object):
            def __repr__(self):
                return "\xe9"*50
        import IPython.core.formatters
        f = IPython.core.formatters.PlainTextFormatter()
        f([Spam(),Spam()])
    

    def test_future_flags(self):
        """Check that future flags are used for parsing code (gh-777)"""
        ip.run_cell('from __future__ import barry_as_FLUFL')
        try:
            ip.run_cell('prfunc_return_val = 1 <> 2')
            assert 'prfunc_return_val' in ip.user_ns
        finally:
            # Reset compiler flags so we don't mess up other tests.
            ip.compile.reset_compiler_flags()

    def test_can_pickle(self):
        "Can we pickle objects defined interactively (GH-29)"
        ip = get_ipython()
        ip.reset()
        ip.run_cell(("class Mylist(list):\n"
                     "    def __init__(self,x=[]):\n"
                     "        list.__init__(self,x)"))
        ip.run_cell("w=Mylist([1,2,3])")
        
        from pickle import dumps
        
        # We need to swap in our main module - this is only necessary
        # inside the test framework, because IPython puts the interactive module
        # in place (but the test framework undoes this).
        _main = sys.modules['__main__']
        sys.modules['__main__'] = ip.user_module
        try:
            res = dumps(ip.user_ns["w"])
        finally:
            sys.modules['__main__'] = _main
        self.assertTrue(isinstance(res, bytes))
        
    def test_global_ns(self):
        "Code in functions must be able to access variables outside them."
        ip = get_ipython()
        ip.run_cell("a = 10")
        ip.run_cell(("def f(x):\n"
                     "    return x + a"))
        ip.run_cell("b = f(12)")
        self.assertEqual(ip.user_ns["b"], 22)

    def test_bad_custom_tb(self):
        """Check that InteractiveShell is protected from bad custom exception handlers"""
        ip.set_custom_exc((IOError,), lambda etype,value,tb: 1/0)
        self.assertEqual(ip.custom_exceptions, (IOError,))
        with tt.AssertPrints("Custom TB Handler failed", channel='stderr'):
            ip.run_cell(u'raise IOError("foo")')
        self.assertEqual(ip.custom_exceptions, ())

    def test_bad_custom_tb_return(self):
        """Check that InteractiveShell is protected from bad return types in custom exception handlers"""
        ip.set_custom_exc((NameError,),lambda etype,value,tb, tb_offset=None: 1)
        self.assertEqual(ip.custom_exceptions, (NameError,))
        with tt.AssertPrints("Custom TB Handler failed", channel='stderr'):
            ip.run_cell(u'a=abracadabra')
        self.assertEqual(ip.custom_exceptions, ())

    def test_drop_by_id(self):
        myvars = {"a":object(), "b":object(), "c": object()}
        ip.push(myvars, interactive=False)
        for name in myvars:
            assert name in ip.user_ns, name
            assert name in ip.user_ns_hidden, name
        ip.user_ns['b'] = 12
        ip.drop_by_id(myvars)
        for name in ["a", "c"]:
            assert name not in ip.user_ns, name
            assert name not in ip.user_ns_hidden, name
        assert ip.user_ns['b'] == 12
        ip.reset()

    def test_var_expand(self):
        ip.user_ns['f'] = u'Ca\xf1o'
        self.assertEqual(ip.var_expand(u'echo $f'), u'echo Ca\xf1o')
        self.assertEqual(ip.var_expand(u'echo {f}'), u'echo Ca\xf1o')
        self.assertEqual(ip.var_expand(u'echo {f[:-1]}'), u'echo Ca\xf1')
        self.assertEqual(ip.var_expand(u'echo {1*2}'), u'echo 2')
        
        self.assertEqual(ip.var_expand(u"grep x | awk '{print $1}'"), u"grep x | awk '{print $1}'")

        ip.user_ns['f'] = b'Ca\xc3\xb1o'
        # This should not raise any exception:
        ip.var_expand(u'echo $f')
   
    def test_var_expand_local(self):
        """Test local variable expansion in !system and %magic calls"""
        # !system
        ip.run_cell(
            "def test():\n"
            '    lvar = "ttt"\n'
            "    ret = !echo {lvar}\n"
            "    return ret[0]\n"
        )
        res = ip.user_ns["test"]()
        self.assertIn("ttt", res)

        # %magic
        ip.run_cell(
            "def makemacro():\n"
            '    macroname = "macro_var_expand_locals"\n'
            "    %macro {macroname} codestr\n"
        )
        ip.user_ns["codestr"] = "str(12)"
        ip.run_cell("makemacro()")
        self.assertIn("macro_var_expand_locals", ip.user_ns)

    def test_var_expand_self(self):
        """Test variable expansion with the name 'self', which was failing.
        
        See https://github.com/ipython/ipython/issues/1878#issuecomment-7698218
        """
        ip.run_cell(
            "class cTest:\n"
            '  classvar="see me"\n'
            "  def test(self):\n"
            "    res = !echo Variable: {self.classvar}\n"
            "    return res[0]\n"
        )
        self.assertIn("see me", ip.user_ns["cTest"]().test())

    def test_bad_var_expand(self):
        """var_expand on invalid formats shouldn't raise"""
        # SyntaxError
        self.assertEqual(ip.var_expand(u"{'a':5}"), u"{'a':5}")
        # NameError
        self.assertEqual(ip.var_expand(u"{asdf}"), u"{asdf}")
        # ZeroDivisionError
        self.assertEqual(ip.var_expand(u"{1/0}"), u"{1/0}")
    
    def test_silent_postexec(self):
        """run_cell(silent=True) doesn't invoke pre/post_run_cell callbacks"""
        pre_explicit = mock.Mock()
        pre_always = mock.Mock()
        post_explicit = mock.Mock()
        post_always = mock.Mock()
        all_mocks = [pre_explicit, pre_always, post_explicit, post_always]
        
        ip.events.register('pre_run_cell', pre_explicit)
        ip.events.register('pre_execute', pre_always)
        ip.events.register('post_run_cell', post_explicit)
        ip.events.register('post_execute', post_always)
        
        try:
            ip.run_cell("1", silent=True)
            assert pre_always.called
            assert not pre_explicit.called
            assert post_always.called
            assert not post_explicit.called
            # double-check that non-silent exec did what we expected
            # silent to avoid
            ip.run_cell("1")
            assert pre_explicit.called
            assert post_explicit.called
            info, = pre_explicit.call_args[0]
            result, = post_explicit.call_args[0]
            self.assertEqual(info, result.info)
            # check that post hooks are always called
            [m.reset_mock() for m in all_mocks]
            ip.run_cell("syntax error")
            assert pre_always.called
            assert pre_explicit.called
            assert post_always.called
            assert post_explicit.called
            info, = pre_explicit.call_args[0]
            result, = post_explicit.call_args[0]
            self.assertEqual(info, result.info)
        finally:
            # remove post-exec
            ip.events.unregister('pre_run_cell', pre_explicit)
            ip.events.unregister('pre_execute', pre_always)
            ip.events.unregister('post_run_cell', post_explicit)
            ip.events.unregister('post_execute', post_always)
    
    def test_silent_noadvance(self):
        """run_cell(silent=True) doesn't advance execution_count"""
        ec = ip.execution_count
        # silent should force store_history=False
        ip.run_cell("1", store_history=True, silent=True)
        
        self.assertEqual(ec, ip.execution_count)
        # double-check that non-silent exec did what we expected
        # silent to avoid
        ip.run_cell("1", store_history=True)
        self.assertEqual(ec+1, ip.execution_count)
    
    def test_silent_nodisplayhook(self):
        """run_cell(silent=True) doesn't trigger displayhook"""
        d = dict(called=False)
        
        trap = ip.display_trap
        save_hook = trap.hook
        
        def failing_hook(*args, **kwargs):
            d['called'] = True
        
        try:
            trap.hook = failing_hook
            res = ip.run_cell("1", silent=True)
            self.assertFalse(d['called'])
            self.assertIsNone(res.result)
            # double-check that non-silent exec did what we expected
            # silent to avoid
            ip.run_cell("1")
            self.assertTrue(d['called'])
        finally:
            trap.hook = save_hook

    def test_ofind_line_magic(self):
        from IPython.core.magic import register_line_magic
        
        @register_line_magic
        def lmagic(line):
            "A line magic"

        # Get info on line magic
        lfind = ip._ofind("lmagic")
        info = OInfo(
            found=True,
            isalias=False,
            ismagic=True,
            namespace="IPython internal",
            obj=lmagic,
            parent=None,
        )
        self.assertEqual(lfind, info)
        
    def test_ofind_cell_magic(self):
        from IPython.core.magic import register_cell_magic
        
        @register_cell_magic
        def cmagic(line, cell):
            "A cell magic"

        # Get info on cell magic
        find = ip._ofind("cmagic")
        info = OInfo(
            found=True,
            isalias=False,
            ismagic=True,
            namespace="IPython internal",
            obj=cmagic,
            parent=None,
        )
        self.assertEqual(find, info)

    def test_ofind_property_with_error(self):
        class A(object):
            @property
            def foo(self):
                raise NotImplementedError()  # pragma: no cover

        a = A()

        found = ip._ofind("a.foo", [("locals", locals())])
        info = OInfo(
            found=True,
            isalias=False,
            ismagic=False,
            namespace="locals",
            obj=A.foo,
            parent=a,
        )
        self.assertEqual(found, info)

    def test_ofind_multiple_attribute_lookups(self):
        class A(object):
            @property
            def foo(self):
                raise NotImplementedError()  # pragma: no cover

        a = A()
        a.a = A()
        a.a.a = A()

        found = ip._ofind("a.a.a.foo", [("locals", locals())])
        info = OInfo(
            found=True,
            isalias=False,
            ismagic=False,
            namespace="locals",
            obj=A.foo,
            parent=a.a.a,
        )
        self.assertEqual(found, info)

    def test_ofind_slotted_attributes(self):
        class A(object):
            __slots__ = ['foo']
            def __init__(self):
                self.foo = 'bar'

        a = A()
        found = ip._ofind("a.foo", [("locals", locals())])
        info = OInfo(
            found=True,
            isalias=False,
            ismagic=False,
            namespace="locals",
            obj=a.foo,
            parent=a,
        )
        self.assertEqual(found, info)

        found = ip._ofind("a.bar", [("locals", locals())])
        expected = OInfo(
            found=False,
            isalias=False,
            ismagic=False,
            namespace=None,
            obj=None,
            parent=a,
        )
        assert found == expected

    def test_ofind_prefers_property_to_instance_level_attribute(self):
        class A(object):
            @property
            def foo(self):
                return 'bar'
        a = A()
        a.__dict__["foo"] = "baz"
        self.assertEqual(a.foo, "bar")
        found = ip._ofind("a.foo", [("locals", locals())])
        self.assertIs(found.obj, A.foo)

    def test_custom_syntaxerror_exception(self):
        called = []
        def my_handler(shell, etype, value, tb, tb_offset=None):
            called.append(etype)
            shell.showtraceback((etype, value, tb), tb_offset=tb_offset)

        ip.set_custom_exc((SyntaxError,), my_handler)
        try:
            ip.run_cell("1f")
            # Check that this was called, and only once.
            self.assertEqual(called, [SyntaxError])
        finally:
            # Reset the custom exception hook
            ip.set_custom_exc((), None)

    def test_custom_exception(self):
        called = []
        def my_handler(shell, etype, value, tb, tb_offset=None):
            called.append(etype)
            shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
        
        ip.set_custom_exc((ValueError,), my_handler)
        try:
            res = ip.run_cell("raise ValueError('test')")
            # Check that this was called, and only once.
            self.assertEqual(called, [ValueError])
            # Check that the error is on the result object
            self.assertIsInstance(res.error_in_exec, ValueError)
        finally:
            # Reset the custom exception hook
            ip.set_custom_exc((), None)
    
    @mock.patch("builtins.print")
    def test_showtraceback_with_surrogates(self, mocked_print):
        values = []

        def mock_print_func(value, sep=" ", end="\n", file=sys.stdout, flush=False):
            values.append(value)
            if value == chr(0xD8FF):
                raise UnicodeEncodeError("utf-8", chr(0xD8FF), 0, 1, "")

        # mock builtins.print
        mocked_print.side_effect = mock_print_func

        # ip._showtraceback() is replaced in globalipapp.py.
        # Call original method to test.
        interactiveshell.InteractiveShell._showtraceback(ip, None, None, chr(0xD8FF))

        self.assertEqual(mocked_print.call_count, 2)
        self.assertEqual(values, [chr(0xD8FF), "\\ud8ff"])

    def test_mktempfile(self):
        filename = ip.mktempfile()
        # Check that we can open the file again on Windows
        with open(filename, "w", encoding="utf-8") as f:
            f.write("abc")

        filename = ip.mktempfile(data="blah")
        with open(filename, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "blah")

    def test_new_main_mod(self):
        # Smoketest to check that this accepts a unicode module name
        name = u'jiefmw'
        mod = ip.new_main_mod(u'%s.py' % name, name)
        self.assertEqual(mod.__name__, name)

    def test_get_exception_only(self):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            msg = ip.get_exception_only()
        self.assertEqual(msg, 'KeyboardInterrupt\n')

        try:
            raise DerivedInterrupt("foo")
        except KeyboardInterrupt:
            msg = ip.get_exception_only()
        self.assertEqual(msg, 'IPython.core.tests.test_interactiveshell.DerivedInterrupt: foo\n')

    def test_inspect_text(self):
        ip.run_cell('a = 5')
        text = ip.object_inspect_text('a')
        self.assertIsInstance(text, str)

    def test_last_execution_result(self):
        """ Check that last execution result gets set correctly (GH-10702) """
        result = ip.run_cell('a = 5; a')
        self.assertTrue(ip.last_execution_succeeded)
        self.assertEqual(ip.last_execution_result.result, 5)

        result = ip.run_cell('a = x_invalid_id_x')
        self.assertFalse(ip.last_execution_succeeded)
        self.assertFalse(ip.last_execution_result.success)
        self.assertIsInstance(ip.last_execution_result.error_in_exec, NameError)

    def test_reset_aliasing(self):
        """ Check that standard posix aliases work after %reset. """
        if os.name != 'posix':
            return

        ip.reset()
        for cmd in ('clear', 'more', 'less', 'man'):
            res = ip.run_cell('%' + cmd)
            self.assertEqual(res.success, True)


class TestSafeExecfileNonAsciiPath(unittest.TestCase):

    @onlyif_unicode_paths
    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        self.TESTDIR = join(self.BASETESTDIR, u"åäö")
        os.mkdir(self.TESTDIR)
        with open(
            join(self.TESTDIR, "åäötestscript.py"), "w", encoding="utf-8"
        ) as sfile:
            sfile.write("pass\n")
        self.oldpath = os.getcwd()
        os.chdir(self.TESTDIR)
        self.fname = u"åäötestscript.py"

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        """Test safe_execfile with non-ascii path
        """
        ip.safe_execfile(self.fname, {}, raise_exceptions=True)

class ExitCodeChecks(tt.TempFileMixin):

    def setUp(self):
        self.system = ip.system_raw

    def test_exit_code_ok(self):
        self.system('exit 0')
        self.assertEqual(ip.user_ns['_exit_code'], 0)

    def test_exit_code_error(self):
        self.system('exit 1')
        self.assertEqual(ip.user_ns['_exit_code'], 1)
    
    @skipif(not hasattr(signal, 'SIGALRM'))
    def test_exit_code_signal(self):
        self.mktmp("import signal, time\n"
                   "signal.setitimer(signal.ITIMER_REAL, 0.1)\n"
                   "time.sleep(1)\n")
        self.system("%s %s" % (sys.executable, self.fname))
        self.assertEqual(ip.user_ns['_exit_code'], -signal.SIGALRM)
    
    @onlyif_cmds_exist("csh")
    def test_exit_code_signal_csh(self):  # pragma: no cover
        SHELL = os.environ.get("SHELL", None)
        os.environ["SHELL"] = find_cmd("csh")
        try:
            self.test_exit_code_signal()
        finally:
            if SHELL is not None:
                os.environ['SHELL'] = SHELL
            else:
                del os.environ['SHELL']


class TestSystemRaw(ExitCodeChecks):

    def setUp(self):
        super().setUp()
        self.system = ip.system_raw

    @onlyif_unicode_paths
    def test_1(self):
        """Test system_raw with non-ascii cmd
        """
        cmd = u'''python -c "'åäö'"   '''
        ip.system_raw(cmd)

    @mock.patch('subprocess.call', side_effect=KeyboardInterrupt)
    @mock.patch('os.system', side_effect=KeyboardInterrupt)
    def test_control_c(self, *mocks):
        try:
            self.system("sleep 1 # wont happen")
        except KeyboardInterrupt:  # pragma: no cove
            self.fail(
                "system call should intercept "
                "keyboard interrupt from subprocess.call"
            )
        self.assertEqual(ip.user_ns["_exit_code"], -signal.SIGINT)


@pytest.mark.parametrize("magic_cmd", ["pip", "conda", "cd"])
def test_magic_warnings(magic_cmd):
    if sys.platform == "win32":
        to_mock = "os.system"
        expected_arg, expected_kwargs = magic_cmd, dict()
    else:
        to_mock = "subprocess.call"
        expected_arg, expected_kwargs = magic_cmd, dict(
            shell=True, executable=os.environ.get("SHELL", None)
        )

    with mock.patch(to_mock, return_value=0) as mock_sub:
        with pytest.warns(Warning, match=r"You executed the system command"):
            ip.system_raw(magic_cmd)
        mock_sub.assert_called_once_with(expected_arg, **expected_kwargs)


# TODO: Exit codes are currently ignored on Windows.
class TestSystemPipedExitCode(ExitCodeChecks):

    def setUp(self):
        super().setUp()
        self.system = ip.system_piped

    @skip_win32
    def test_exit_code_ok(self):
        ExitCodeChecks.test_exit_code_ok(self)

    @skip_win32
    def test_exit_code_error(self):
        ExitCodeChecks.test_exit_code_error(self)

    @skip_win32
    def test_exit_code_signal(self):
        ExitCodeChecks.test_exit_code_signal(self)

class TestModules(tt.TempFileMixin):
    def test_extraneous_loads(self):
        """Test we're not loading modules on startup that we shouldn't.
        """
        self.mktmp("import sys\n"
                   "print('numpy' in sys.modules)\n"
                   "print('ipyparallel' in sys.modules)\n"
                   "print('ipykernel' in sys.modules)\n"
                   )
        out = "False\nFalse\nFalse\n"
        tt.ipexec_validate(self.fname, out)

class Negator(ast.NodeTransformer):
    """Negates all number literals in an AST."""

    def visit_Num(self, node):
        node.n = -node.n
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node

class TestAstTransform(unittest.TestCase):
    def setUp(self):
        self.negator = Negator()
        ip.ast_transformers.append(self.negator)
    
    def tearDown(self):
        ip.ast_transformers.remove(self.negator)

    def test_non_int_const(self):
        with tt.AssertPrints("hello"):
            ip.run_cell('print("hello")')

    def test_run_cell(self):
        with tt.AssertPrints("-34"):
            ip.run_cell("print(12 + 22)")

        # A named reference to a number shouldn't be transformed.
        ip.user_ns["n"] = 55
        with tt.AssertNotPrints("-55"):
            ip.run_cell("print(n)")

    def test_timeit(self):
        called = set()
        def f(x):
            called.add(x)
        ip.push({'f':f})
        
        with tt.AssertPrints("std. dev. of"):
            ip.run_line_magic("timeit", "-n1 f(1)")
        self.assertEqual(called, {-1})
        called.clear()

        with tt.AssertPrints("std. dev. of"):
            ip.run_cell_magic("timeit", "-n1 f(2)", "f(3)")
        self.assertEqual(called, {-2, -3})
    
    def test_time(self):
        called = []
        def f(x):
            called.append(x)
        ip.push({'f':f})
        
        # Test with an expression
        with tt.AssertPrints("Wall time: "):
            ip.run_line_magic("time", "f(5+9)")
        self.assertEqual(called, [-14])
        called[:] = []
        
        # Test with a statement (different code path)
        with tt.AssertPrints("Wall time: "):
            ip.run_line_magic("time", "a = f(-3 + -2)")
        self.assertEqual(called, [5])
    
    def test_macro(self):
        ip.push({'a':10})
        # The AST transformation makes this do a+=-1
        ip.define_macro("amacro", "a+=1\nprint(a)")
        
        with tt.AssertPrints("9"):
            ip.run_cell("amacro")
        with tt.AssertPrints("8"):
            ip.run_cell("amacro")

class TestMiscTransform(unittest.TestCase):


    def test_transform_only_once(self):
        cleanup = 0
        line_t = 0
        def count_cleanup(lines):
            nonlocal cleanup
            cleanup += 1
            return lines

        def count_line_t(lines):
            nonlocal line_t
            line_t += 1
            return lines

        ip.input_transformer_manager.cleanup_transforms.append(count_cleanup)
        ip.input_transformer_manager.line_transforms.append(count_line_t)

        ip.run_cell('1')

        assert cleanup == 1
        assert line_t == 1

class IntegerWrapper(ast.NodeTransformer):
    """Wraps all integers in a call to Integer()"""

    # for Python 3.7 and earlier

    # for Python 3.7 and earlier
    def visit_Num(self, node):
        if isinstance(node.n, int):
            return ast.Call(func=ast.Name(id='Integer', ctx=ast.Load()),
                            args=[node], keywords=[])
        return node

    # For Python 3.8+
    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node


class TestAstTransform2(unittest.TestCase):
    def setUp(self):
        self.intwrapper = IntegerWrapper()
        ip.ast_transformers.append(self.intwrapper)
        
        self.calls = []
        def Integer(*args):
            self.calls.append(args)
            return args
        ip.push({"Integer": Integer})
    
    def tearDown(self):
        ip.ast_transformers.remove(self.intwrapper)
        del ip.user_ns['Integer']
    
    def test_run_cell(self):
        ip.run_cell("n = 2")
        self.assertEqual(self.calls, [(2,)])
        
        # This shouldn't throw an error
        ip.run_cell("o = 2.0")
        self.assertEqual(ip.user_ns['o'], 2.0)

    def test_run_cell_non_int(self):
        ip.run_cell("n = 'a'")
        assert self.calls == []

    def test_timeit(self):
        called = set()
        def f(x):
            called.add(x)
        ip.push({'f':f})

        with tt.AssertPrints("std. dev. of"):
            ip.run_line_magic("timeit", "-n1 f(1)")
        self.assertEqual(called, {(1,)})
        called.clear()

        with tt.AssertPrints("std. dev. of"):
            ip.run_cell_magic("timeit", "-n1 f(2)", "f(3)")
        self.assertEqual(called, {(2,), (3,)})

class ErrorTransformer(ast.NodeTransformer):
    """Throws an error when it sees a number."""

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            raise ValueError("test")
        return node


class TestAstTransformError(unittest.TestCase):
    def test_unregistering(self):
        err_transformer = ErrorTransformer()
        ip.ast_transformers.append(err_transformer)
        
        with self.assertWarnsRegex(UserWarning, "It will be unregistered"):
            ip.run_cell("1 + 2")
        
        # This should have been removed.
        self.assertNotIn(err_transformer, ip.ast_transformers)


class StringRejector(ast.NodeTransformer):
    """Throws an InputRejected when it sees a string literal.

    Used to verify that NodeTransformers can signal that a piece of code should
    not be executed by throwing an InputRejected.
    """
    
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            raise InputRejected("test")
        return node


class TestAstTransformInputRejection(unittest.TestCase):

    def setUp(self):
        self.transformer = StringRejector()
        ip.ast_transformers.append(self.transformer)

    def tearDown(self):
        ip.ast_transformers.remove(self.transformer)

    def test_input_rejection(self):
        """Check that NodeTransformers can reject input."""

        expect_exception_tb = tt.AssertPrints("InputRejected: test")
        expect_no_cell_output = tt.AssertNotPrints("'unsafe'", suppress=False)

        # Run the same check twice to verify that the transformer is not
        # disabled after raising.
        with expect_exception_tb, expect_no_cell_output:
            ip.run_cell("'unsafe'")

        with expect_exception_tb, expect_no_cell_output:
            res = ip.run_cell("'unsafe'")

        self.assertIsInstance(res.error_before_exec, InputRejected)

def test__IPYTHON__():
    # This shouldn't raise a NameError, that's all
    __IPYTHON__


class DummyRepr(object):
    def __repr__(self):
        return "DummyRepr"
    
    def _repr_html_(self):
        return "<b>dummy</b>"
    
    def _repr_javascript_(self):
        return "console.log('hi');", {'key': 'value'}
    

def test_user_variables():
    # enable all formatters
    ip.display_formatter.active_types = ip.display_formatter.format_types
    
    ip.user_ns['dummy'] = d = DummyRepr()
    keys = {'dummy', 'doesnotexist'}
    r = ip.user_expressions({ key:key for key in keys})

    assert keys == set(r.keys())
    dummy = r["dummy"]
    assert {"status", "data", "metadata"} == set(dummy.keys())
    assert dummy["status"] == "ok"
    data = dummy["data"]
    metadata = dummy["metadata"]
    assert data.get("text/html") == d._repr_html_()
    js, jsmd = d._repr_javascript_()
    assert data.get("application/javascript") == js
    assert metadata.get("application/javascript") == jsmd

    dne = r["doesnotexist"]
    assert dne["status"] == "error"
    assert dne["ename"] == "NameError"

    # back to text only
    ip.display_formatter.active_types = ['text/plain']
    
def test_user_expression():
    # enable all formatters
    ip.display_formatter.active_types = ip.display_formatter.format_types
    query = {
        'a' : '1 + 2',
        'b' : '1/0',
    }
    r = ip.user_expressions(query)
    import pprint
    pprint.pprint(r)
    assert set(r.keys()) == set(query.keys())
    a = r["a"]
    assert {"status", "data", "metadata"} == set(a.keys())
    assert a["status"] == "ok"
    data = a["data"]
    metadata = a["metadata"]
    assert data.get("text/plain") == "3"

    b = r["b"]
    assert b["status"] == "error"
    assert b["ename"] == "ZeroDivisionError"

    # back to text only
    ip.display_formatter.active_types = ['text/plain']


class TestSyntaxErrorTransformer(unittest.TestCase):
    """Check that SyntaxError raised by an input transformer is handled by run_cell()"""

    @staticmethod
    def transformer(lines):
        for line in lines:
            pos = line.find('syntaxerror')
            if pos >= 0:
                e = SyntaxError('input contains "syntaxerror"')
                e.text = line
                e.offset = pos + 1
                raise e
        return lines

    def setUp(self):
        ip.input_transformers_post.append(self.transformer)

    def tearDown(self):
        ip.input_transformers_post.remove(self.transformer)

    def test_syntaxerror_input_transformer(self):
        with tt.AssertPrints('1234'):
            ip.run_cell('1234')
        with tt.AssertPrints('SyntaxError: invalid syntax'):
            ip.run_cell('1 2 3')   # plain python syntax error
        with tt.AssertPrints('SyntaxError: input contains "syntaxerror"'):
            ip.run_cell('2345  # syntaxerror')  # input transformer syntax error
        with tt.AssertPrints('3456'):
            ip.run_cell('3456')


class TestWarningSuppression(unittest.TestCase):
    def test_warning_suppression(self):
        ip.run_cell("import warnings")
        try:
            with self.assertWarnsRegex(UserWarning, "asdf"):
                ip.run_cell("warnings.warn('asdf')")
            # Here's the real test -- if we run that again, we should get the
            # warning again. Traditionally, each warning was only issued once per
            # IPython session (approximately), even if the user typed in new and
            # different code that should have also triggered the warning, leading
            # to much confusion.
            with self.assertWarnsRegex(UserWarning, "asdf"):
                ip.run_cell("warnings.warn('asdf')")
        finally:
            ip.run_cell("del warnings")


    def test_deprecation_warning(self):
        ip.run_cell("""
import warnings
def wrn():
    warnings.warn(
        "I AM  A WARNING",
        DeprecationWarning
    )
        """)
        try:
            with self.assertWarnsRegex(DeprecationWarning, "I AM  A WARNING"):
                ip.run_cell("wrn()")
        finally:
            ip.run_cell("del warnings")
            ip.run_cell("del wrn")


class TestImportNoDeprecate(tt.TempFileMixin):

    def setUp(self):
        """Make a valid python temp file."""
        self.mktmp("""
import warnings
def wrn():
    warnings.warn(
        "I AM  A WARNING",
        DeprecationWarning
    )
""")
        super().setUp()

    def test_no_dep(self):
        """
        No deprecation warning should be raised from imported functions
        """
        ip.run_cell("from {} import wrn".format(self.fname))

        with tt.AssertNotPrints("I AM  A WARNING"):
            ip.run_cell("wrn()")
        ip.run_cell("del wrn")


def test_custom_exc_count():
    hook = mock.Mock(return_value=None)
    ip.set_custom_exc((SyntaxError,), hook)
    before = ip.execution_count
    ip.run_cell("def foo()", store_history=True)
    # restore default excepthook
    ip.set_custom_exc((), None)
    assert hook.call_count == 1
    assert ip.execution_count == before + 1


def test_run_cell_async():
    ip.run_cell("import asyncio")
    coro = ip.run_cell_async("await asyncio.sleep(0.01)\n5")
    assert asyncio.iscoroutine(coro)
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(coro)
    assert isinstance(result, interactiveshell.ExecutionResult)
    assert result.result == 5


def test_run_cell_await():
    ip.run_cell("import asyncio")
    result = ip.run_cell("await asyncio.sleep(0.01); 10")
    assert ip.user_ns["_"] == 10


def test_run_cell_asyncio_run():
    ip.run_cell("import asyncio")
    result = ip.run_cell("await asyncio.sleep(0.01); 1")
    assert ip.user_ns["_"] == 1
    result = ip.run_cell("asyncio.run(asyncio.sleep(0.01)); 2")
    assert ip.user_ns["_"] == 2
    result = ip.run_cell("await asyncio.sleep(0.01); 3")
    assert ip.user_ns["_"] == 3


def test_should_run_async():
    assert not ip.should_run_async("a = 5", transformed_cell="a = 5")
    assert ip.should_run_async("await x", transformed_cell="await x")
    assert ip.should_run_async(
        "import asyncio; await asyncio.sleep(1)",
        transformed_cell="import asyncio; await asyncio.sleep(1)",
    )


def test_set_custom_completer():
    num_completers = len(ip.Completer.matchers)

    def foo(*args, **kwargs):
        return "I'm a completer!"

    ip.set_custom_completer(foo, 0)

    # check that we've really added a new completer
    assert len(ip.Completer.matchers) == num_completers + 1

    # check that the first completer is the function we defined
    assert ip.Completer.matchers[0]() == "I'm a completer!"

    # clean up
    ip.Completer.custom_matchers.pop()


class TestShowTracebackAttack(unittest.TestCase):
    """Test that the interactive shell is resilient against the client attack of
    manipulating the showtracebacks method. These attacks shouldn't result in an
    unhandled exception in the kernel."""

    def setUp(self):
        self.orig_showtraceback = interactiveshell.InteractiveShell.showtraceback

    def tearDown(self):
        interactiveshell.InteractiveShell.showtraceback = self.orig_showtraceback

    def test_set_show_tracebacks_none(self):
        """Test the case of the client setting showtracebacks to None"""

        result = ip.run_cell(
            """
            import IPython.core.interactiveshell
            IPython.core.interactiveshell.InteractiveShell.showtraceback = None

            assert False, "This should not raise an exception"
        """
        )
        print(result)

        assert result.result is None
        assert isinstance(result.error_in_exec, TypeError)
        assert str(result.error_in_exec) == "'NoneType' object is not callable"

    def test_set_show_tracebacks_noop(self):
        """Test the case of the client setting showtracebacks to a no op lambda"""

        result = ip.run_cell(
            """
            import IPython.core.interactiveshell
            IPython.core.interactiveshell.InteractiveShell.showtraceback = lambda *args, **kwargs: None

            assert False, "This should not raise an exception"
        """
        )
        print(result)

        assert result.result is None
        assert isinstance(result.error_in_exec, AssertionError)
        assert str(result.error_in_exec) == "This should not raise an exception"
