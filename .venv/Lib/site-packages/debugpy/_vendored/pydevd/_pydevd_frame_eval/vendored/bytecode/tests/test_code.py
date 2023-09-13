
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
import unittest

from _pydevd_frame_eval.vendored.bytecode import ConcreteBytecode, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.tests import get_code


class CodeTests(unittest.TestCase):
    """Check that bytecode.from_code(code).to_code() returns code."""

    def check(self, source, function=False):
        ref_code = get_code(source, function=function)

        code = ConcreteBytecode.from_code(ref_code).to_code()
        self.assertEqual(code, ref_code)

        code = Bytecode.from_code(ref_code).to_code()
        self.assertEqual(code, ref_code)

        bytecode = Bytecode.from_code(ref_code)
        blocks = ControlFlowGraph.from_bytecode(bytecode)
        code = blocks.to_bytecode().to_code()
        self.assertEqual(code, ref_code)

    def test_loop(self):
        self.check(
            """
            for x in range(1, 10):
                x += 1
                if x == 3:
                    continue
                x -= 1
                if x > 7:
                    break
                x = 0
            print(x)
        """
        )

    def test_varargs(self):
        self.check(
            """
            def func(a, b, *varargs):
                pass
        """,
            function=True,
        )

    def test_kwargs(self):
        self.check(
            """
            def func(a, b, **kwargs):
                pass
        """,
            function=True,
        )

    def test_kwonlyargs(self):
        self.check(
            """
            def func(*, arg, arg2):
                pass
        """,
            function=True,
        )

    # Added because Python 3.10 added some special beahavior with respect to
    # generators in term of stack size
    def test_generator_func(self):
        self.check(
            """
            def func(arg, arg2):
                yield
        """,
            function=True,
        )

    def test_async_func(self):
        self.check(
            """
            async def func(arg, arg2):
                pass
        """,
            function=True,
        )


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
