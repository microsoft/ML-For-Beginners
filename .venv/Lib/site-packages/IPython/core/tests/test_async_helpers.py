"""
Test for async helpers.

Should only trigger on python 3.5+ or will have syntax errors.
"""
from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase

import pytest

from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without

if TYPE_CHECKING:
    from IPython import get_ipython

    ip = get_ipython()


def iprc(x):
    return ip.run_cell(dedent(x)).raise_error()


def iprc_nr(x):
    return ip.run_cell(dedent(x))


class AsyncTest(TestCase):
    def test_should_be_async(self):
        self.assertFalse(_should_be_async("False"))
        self.assertTrue(_should_be_async("await bar()"))
        self.assertTrue(_should_be_async("x = await bar()"))
        self.assertFalse(
            _should_be_async(
                dedent(
                    """
            async def awaitable():
                pass
        """
                )
            )
        )

    def _get_top_level_cases(self):
        # These are test cases that should be valid in a function
        # but invalid outside of a function.
        test_cases = []
        test_cases.append(('basic', "{val}"))

        # Note, in all conditional cases, I use True instead of
        # False so that the peephole optimizer won't optimize away
        # the return, so CPython will see this as a syntax error:
        #
        # while True:
        #    break
        #    return
        #
        # But not this:
        #
        # while False:
        #    return
        #
        # See https://bugs.python.org/issue1875

        test_cases.append(('if', dedent("""
        if True:
            {val}
        """)))

        test_cases.append(('while', dedent("""
        while True:
            {val}
            break
        """)))

        test_cases.append(('try', dedent("""
        try:
            {val}
        except:
            pass
        """)))

        test_cases.append(('except', dedent("""
        try:
            pass
        except:
            {val}
        """)))

        test_cases.append(('finally', dedent("""
        try:
            pass
        except:
            pass
        finally:
            {val}
        """)))

        test_cases.append(('for', dedent("""
        for _ in range(4):
            {val}
        """)))


        test_cases.append(('nested', dedent("""
        if True:
            while True:
                {val}
                break
        """)))

        test_cases.append(('deep-nested', dedent("""
        if True:
            while True:
                break
                for x in range(3):
                    if True:
                        while True:
                            for x in range(3):
                                {val}
        """)))

        return test_cases

    def _get_ry_syntax_errors(self):
        # This is a mix of tests that should be a syntax error if
        # return or yield whether or not they are in a function

        test_cases = []

        test_cases.append(('class', dedent("""
        class V:
            {val}
        """)))

        test_cases.append(('nested-class', dedent("""
        class V:
            class C:
                {val}
        """)))

        return test_cases


    def test_top_level_return_error(self):
        tl_err_test_cases = self._get_top_level_cases()
        tl_err_test_cases.extend(self._get_ry_syntax_errors())

        vals = ('return', 'yield', 'yield from (_ for _ in range(3))',
                dedent('''
                    def f():
                        pass
                    return
                    '''),
                )

        for test_name, test_case in tl_err_test_cases:
            # This example should work if 'pass' is used as the value
            with self.subTest((test_name, 'pass')):
                iprc(test_case.format(val='pass'))

            # It should fail with all the values
            for val in vals:
                with self.subTest((test_name, val)):
                    msg = "Syntax error not raised for %s, %s" % (test_name, val)
                    with self.assertRaises(SyntaxError, msg=msg):
                        iprc(test_case.format(val=val))

    def test_in_func_no_error(self):
        # Test that the implementation of top-level return/yield
        # detection isn't *too* aggressive, and works inside a function
        func_contexts = []

        func_contexts.append(('func', False, dedent("""
        def f():""")))

        func_contexts.append(('method', False, dedent("""
        class MyClass:
            def __init__(self):
        """)))

        func_contexts.append(('async-func', True,  dedent("""
        async def f():""")))

        func_contexts.append(('async-method', True,  dedent("""
        class MyClass:
            async def f(self):""")))

        func_contexts.append(('closure', False, dedent("""
        def f():
            def g():
        """)))

        def nest_case(context, case):
            # Detect indentation
            lines = context.strip().splitlines()
            prefix_len = 0
            for c in lines[-1]:
                if c != ' ':
                    break
                prefix_len += 1

            indented_case = indent(case, ' ' * (prefix_len + 4))
            return context + '\n' + indented_case

        # Gather and run the tests

        # yield is allowed in async functions, starting in Python 3.6,
        # and yield from is not allowed in any version
        vals = ('return', 'yield', 'yield from (_ for _ in range(3))')

        success_tests = zip(self._get_top_level_cases(), repeat(False))
        failure_tests = zip(self._get_ry_syntax_errors(), repeat(True))

        tests = chain(success_tests, failure_tests)

        for context_name, async_func, context in func_contexts:
            for (test_name, test_case), should_fail in tests:
                nested_case = nest_case(context, test_case)

                for val in vals:
                    test_id = (context_name, test_name, val)
                    cell = nested_case.format(val=val)

                    with self.subTest(test_id):
                        if should_fail:
                            msg = ("SyntaxError not raised for %s" %
                                    str(test_id))
                            with self.assertRaises(SyntaxError, msg=msg):
                                iprc(cell)

                                print(cell)
                        else:
                            iprc(cell)

    def test_nonlocal(self):
        # fails if outer scope is not a function scope or if var not defined
        with self.assertRaises(SyntaxError):
            iprc("nonlocal x")
            iprc("""
            x = 1
            def f():
                nonlocal x
                x = 10000
                yield x
            """)
            iprc("""
            def f():
                def g():
                    nonlocal x
                    x = 10000
                    yield x
            """)

        # works if outer scope is a function scope and var exists
        iprc("""
        def f():
            x = 20
            def g():
                nonlocal x
                x = 10000
                yield x
        """)


    def test_execute(self):
        iprc("""
        import asyncio
        await asyncio.sleep(0.001)
        """
        )

    def test_autoawait(self):
        iprc("%autoawait False")
        iprc("%autoawait True")
        iprc("""
        from asyncio import sleep
        await sleep(0.1)
        """
        )

    def test_memory_error(self):
        """
        The pgen parser in 3.8 or before use to raise MemoryError on too many
        nested parens anymore"""

        iprc("(" * 200 + ")" * 200)

    @pytest.mark.xfail(reason="fail on curio 1.6 and before on Python 3.12")
    @pytest.mark.skip(
        reason="skip_without(curio) fails on 3.12 for now even with other skip so must uncond skip"
    )
    # @skip_without("curio")
    def test_autoawait_curio(self):
        iprc("%autoawait curio")

    @skip_without('trio')
    def test_autoawait_trio(self):
        iprc("%autoawait trio")

    @skip_without('trio')
    def test_autoawait_trio_wrong_sleep(self):
        iprc("%autoawait trio")
        res = iprc_nr("""
        import asyncio
        await asyncio.sleep(0)
        """)
        with self.assertRaises(TypeError):
            res.raise_error()

    @skip_without('trio')
    def test_autoawait_asyncio_wrong_sleep(self):
        iprc("%autoawait asyncio")
        res = iprc_nr("""
        import trio
        await trio.sleep(0)
        """)
        with self.assertRaises(RuntimeError):
            res.raise_error()


    def tearDown(self):
        ip.loop_runner = "asyncio"
