
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
#!/usr/bin/env python3
import contextlib
import io
import sys
import textwrap
import unittest

from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Bytecode, BasicBlock, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble


class DumpCodeTests(unittest.TestCase):
    maxDiff = 80 * 100

    def check_dump_bytecode(self, code, expected, lineno=None):
        with contextlib.redirect_stdout(io.StringIO()) as stderr:
            if lineno is not None:
                bytecode.dump_bytecode(code, lineno=True)
            else:
                bytecode.dump_bytecode(code)
            output = stderr.getvalue()

        self.assertEqual(output, expected)

    def test_bytecode(self):
        source = """
            def func(test):
                if test == 1:
                    return 1
                elif test == 2:
                    return 2
                return 3
        """
        code = disassemble(source, function=True)

        # without line numbers
        enum_repr = "<Compare.EQ: 2>"
        expected = f"""
    LOAD_FAST 'test'
    LOAD_CONST 1
    COMPARE_OP {enum_repr}
    POP_JUMP_IF_FALSE <label_instr6>
    LOAD_CONST 1
    RETURN_VALUE

label_instr6:
    LOAD_FAST 'test'
    LOAD_CONST 2
    COMPARE_OP {enum_repr}
    POP_JUMP_IF_FALSE <label_instr13>
    LOAD_CONST 2
    RETURN_VALUE

label_instr13:
    LOAD_CONST 3
    RETURN_VALUE

        """[
            1:
        ].rstrip(
            " "
        )
        self.check_dump_bytecode(code, expected)

        # with line numbers
        expected = f"""
    L.  2   0: LOAD_FAST 'test'
            1: LOAD_CONST 1
            2: COMPARE_OP {enum_repr}
            3: POP_JUMP_IF_FALSE <label_instr6>
    L.  3   4: LOAD_CONST 1
            5: RETURN_VALUE

label_instr6:
    L.  4   7: LOAD_FAST 'test'
            8: LOAD_CONST 2
            9: COMPARE_OP {enum_repr}
           10: POP_JUMP_IF_FALSE <label_instr13>
    L.  5  11: LOAD_CONST 2
           12: RETURN_VALUE

label_instr13:
    L.  6  14: LOAD_CONST 3
           15: RETURN_VALUE

        """[
            1:
        ].rstrip(
            " "
        )
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_bytecode_broken_label(self):
        label = Label()
        code = Bytecode([Instr("JUMP_ABSOLUTE", label)])

        expected = "    JUMP_ABSOLUTE <error: unknown label>\n\n"
        self.check_dump_bytecode(code, expected)

    def test_blocks_broken_jump(self):
        block = BasicBlock()
        code = ControlFlowGraph()
        code[0].append(Instr("JUMP_ABSOLUTE", block))

        expected = textwrap.dedent(
            """
            block1:
                JUMP_ABSOLUTE <error: unknown block>

        """
        ).lstrip("\n")
        self.check_dump_bytecode(code, expected)

    def test_bytecode_blocks(self):
        source = """
            def func(test):
                if test == 1:
                    return 1
                elif test == 2:
                    return 2
                return 3
        """
        code = disassemble(source, function=True)
        code = ControlFlowGraph.from_bytecode(code)

        # without line numbers
        enum_repr = "<Compare.EQ: 2>"
        expected = textwrap.dedent(
            f"""
            block1:
                LOAD_FAST 'test'
                LOAD_CONST 1
                COMPARE_OP {enum_repr}
                POP_JUMP_IF_FALSE <block3>
                -> block2

            block2:
                LOAD_CONST 1
                RETURN_VALUE

            block3:
                LOAD_FAST 'test'
                LOAD_CONST 2
                COMPARE_OP {enum_repr}
                POP_JUMP_IF_FALSE <block5>
                -> block4

            block4:
                LOAD_CONST 2
                RETURN_VALUE

            block5:
                LOAD_CONST 3
                RETURN_VALUE

        """
        ).lstrip()
        self.check_dump_bytecode(code, expected)

        # with line numbers
        expected = textwrap.dedent(
            f"""
            block1:
                L.  2   0: LOAD_FAST 'test'
                        1: LOAD_CONST 1
                        2: COMPARE_OP {enum_repr}
                        3: POP_JUMP_IF_FALSE <block3>
                -> block2

            block2:
                L.  3   0: LOAD_CONST 1
                        1: RETURN_VALUE

            block3:
                L.  4   0: LOAD_FAST 'test'
                        1: LOAD_CONST 2
                        2: COMPARE_OP {enum_repr}
                        3: POP_JUMP_IF_FALSE <block5>
                -> block4

            block4:
                L.  5   0: LOAD_CONST 2
                        1: RETURN_VALUE

            block5:
                L.  6   0: LOAD_CONST 3
                        1: RETURN_VALUE

        """
        ).lstrip()
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_concrete_bytecode(self):
        source = """
            def func(test):
                if test == 1:
                    return 1
                elif test == 2:
                    return 2
                return 3
        """
        code = disassemble(source, function=True)
        code = code.to_concrete_bytecode()

        # without line numbers
        expected = f"""
  0    LOAD_FAST 0
  2    LOAD_CONST 1
  4    COMPARE_OP 2
  6    POP_JUMP_IF_FALSE {6 if OFFSET_AS_INSTRUCTION else 12}
  8    LOAD_CONST 1
 10    RETURN_VALUE
 12    LOAD_FAST 0
 14    LOAD_CONST 2
 16    COMPARE_OP 2
 18    POP_JUMP_IF_FALSE {12 if OFFSET_AS_INSTRUCTION else 24}
 20    LOAD_CONST 2
 22    RETURN_VALUE
 24    LOAD_CONST 3
 26    RETURN_VALUE
""".lstrip(
            "\n"
        )
        self.check_dump_bytecode(code, expected)

        # with line numbers
        expected = f"""
L.  2   0: LOAD_FAST 0
        2: LOAD_CONST 1
        4: COMPARE_OP 2
        6: POP_JUMP_IF_FALSE {6 if OFFSET_AS_INSTRUCTION else 12}
L.  3   8: LOAD_CONST 1
       10: RETURN_VALUE
L.  4  12: LOAD_FAST 0
       14: LOAD_CONST 2
       16: COMPARE_OP 2
       18: POP_JUMP_IF_FALSE {12 if OFFSET_AS_INSTRUCTION else 24}
L.  5  20: LOAD_CONST 2
       22: RETURN_VALUE
L.  6  24: LOAD_CONST 3
       26: RETURN_VALUE
""".lstrip(
            "\n"
        )
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_type_validation(self):
        class T:
            first_lineno = 1

        with self.assertRaises(TypeError):
            bytecode.dump_bytecode(T())


class MiscTests(unittest.TestCase):
    def skip_test_version(self):
        import setup

        self.assertEqual(bytecode.__version__, setup.VERSION)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
