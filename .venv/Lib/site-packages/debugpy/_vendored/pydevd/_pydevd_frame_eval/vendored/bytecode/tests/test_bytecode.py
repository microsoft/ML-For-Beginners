
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
#!/usr/bin/env python3
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code


class BytecodeTests(TestCase):
    maxDiff = 80 * 100

    def test_constructor(self):
        code = Bytecode()
        self.assertEqual(code.name, "<module>")
        self.assertEqual(code.filename, "<string>")
        self.assertEqual(code.flags, 0)
        self.assertEqual(code, [])

    def test_invalid_types(self):
        code = Bytecode()
        code.append(123)
        with self.assertRaises(ValueError):
            list(code)
        with self.assertRaises(ValueError):
            code.legalize()
        with self.assertRaises(ValueError):
            Bytecode([123])

    def test_legalize(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend(
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_CONST", 8, lineno=4),
                Instr("STORE_NAME", "y"),
                Label(),
                SetLineno(5),
                Instr("LOAD_CONST", 9, lineno=6),
                Instr("STORE_NAME", "z"),
            ]
        )

        code.legalize()
        self.assertListEqual(
            code,
            [
                Instr("LOAD_CONST", 7, lineno=3),
                Instr("STORE_NAME", "x", lineno=3),
                Instr("LOAD_CONST", 8, lineno=4),
                Instr("STORE_NAME", "y", lineno=4),
                Label(),
                Instr("LOAD_CONST", 9, lineno=5),
                Instr("STORE_NAME", "z", lineno=5),
            ],
        )

    def test_slice(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend(
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                SetLineno(4),
                Instr("LOAD_CONST", 8),
                Instr("STORE_NAME", "y"),
                SetLineno(5),
                Instr("LOAD_CONST", 9),
                Instr("STORE_NAME", "z"),
            ]
        )
        sliced_code = code[:]
        self.assertEqual(code, sliced_code)
        for name in (
            "argcount",
            "posonlyargcount",
            "kwonlyargcount",
            "first_lineno",
            "name",
            "filename",
            "docstring",
            "cellvars",
            "freevars",
            "argnames",
        ):
            self.assertEqual(
                getattr(code, name, None), getattr(sliced_code, name, None)
            )

    def test_copy(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend(
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                SetLineno(4),
                Instr("LOAD_CONST", 8),
                Instr("STORE_NAME", "y"),
                SetLineno(5),
                Instr("LOAD_CONST", 9),
                Instr("STORE_NAME", "z"),
            ]
        )

        copy_code = code.copy()
        self.assertEqual(code, copy_code)
        for name in (
            "argcount",
            "posonlyargcount",
            "kwonlyargcount",
            "first_lineno",
            "name",
            "filename",
            "docstring",
            "cellvars",
            "freevars",
            "argnames",
        ):
            self.assertEqual(getattr(code, name, None), getattr(copy_code, name, None))

    def test_from_code(self):
        code = get_code(
            """
            if test:
                x = 1
            else:
                x = 2
        """
        )
        bytecode = Bytecode.from_code(code)
        label_else = Label()
        label_exit = Label()
        if sys.version_info < (3, 10):
            self.assertEqual(
                bytecode,
                [
                    Instr("LOAD_NAME", "test", lineno=1),
                    Instr("POP_JUMP_IF_FALSE", label_else, lineno=1),
                    Instr("LOAD_CONST", 1, lineno=2),
                    Instr("STORE_NAME", "x", lineno=2),
                    Instr("JUMP_FORWARD", label_exit, lineno=2),
                    label_else,
                    Instr("LOAD_CONST", 2, lineno=4),
                    Instr("STORE_NAME", "x", lineno=4),
                    label_exit,
                    Instr("LOAD_CONST", None, lineno=4),
                    Instr("RETURN_VALUE", lineno=4),
                ],
            )
        # Control flow handling appears to have changed under Python 3.10
        else:
            self.assertEqual(
                bytecode,
                [
                    Instr("LOAD_NAME", "test", lineno=1),
                    Instr("POP_JUMP_IF_FALSE", label_else, lineno=1),
                    Instr("LOAD_CONST", 1, lineno=2),
                    Instr("STORE_NAME", "x", lineno=2),
                    Instr("LOAD_CONST", None, lineno=2),
                    Instr("RETURN_VALUE", lineno=2),
                    label_else,
                    Instr("LOAD_CONST", 2, lineno=4),
                    Instr("STORE_NAME", "x", lineno=4),
                    Instr("LOAD_CONST", None, lineno=4),
                    Instr("RETURN_VALUE", lineno=4),
                ],
            )

    def test_from_code_freevars(self):
        ns = {}
        exec(
            textwrap.dedent(
                """
            def create_func():
                x = 1
                def func():
                    return x
                return func

            func = create_func()
        """
            ),
            ns,
            ns,
        )
        code = ns["func"].__code__

        bytecode = Bytecode.from_code(code)
        self.assertEqual(
            bytecode,
            [
                Instr("LOAD_DEREF", FreeVar("x"), lineno=5),
                Instr("RETURN_VALUE", lineno=5),
            ],
        )

    def test_from_code_load_fast(self):
        code = get_code(
            """
            def func():
                x = 33
                y = x
        """,
            function=True,
        )
        code = Bytecode.from_code(code)
        self.assertEqual(
            code,
            [
                Instr("LOAD_CONST", 33, lineno=2),
                Instr("STORE_FAST", "x", lineno=2),
                Instr("LOAD_FAST", "x", lineno=3),
                Instr("STORE_FAST", "y", lineno=3),
                Instr("LOAD_CONST", None, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
            ],
        )

    def test_setlineno(self):
        # x = 7
        # y = 8
        # z = 9
        code = Bytecode()
        code.first_lineno = 3
        code.extend(
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                SetLineno(4),
                Instr("LOAD_CONST", 8),
                Instr("STORE_NAME", "y"),
                SetLineno(5),
                Instr("LOAD_CONST", 9),
                Instr("STORE_NAME", "z"),
            ]
        )

        concrete = code.to_concrete_bytecode()
        self.assertEqual(concrete.consts, [7, 8, 9])
        self.assertEqual(concrete.names, ["x", "y", "z"])
        self.assertListEqual(
            list(concrete),
            [
                ConcreteInstr("LOAD_CONST", 0, lineno=3),
                ConcreteInstr("STORE_NAME", 0, lineno=3),
                ConcreteInstr("LOAD_CONST", 1, lineno=4),
                ConcreteInstr("STORE_NAME", 1, lineno=4),
                ConcreteInstr("LOAD_CONST", 2, lineno=5),
                ConcreteInstr("STORE_NAME", 2, lineno=5),
            ],
        )

    def test_to_code(self):
        code = Bytecode()
        code.first_lineno = 50
        code.extend(
            [
                Instr("LOAD_NAME", "print"),
                Instr("LOAD_CONST", "%s"),
                Instr("LOAD_GLOBAL", "a"),
                Instr("BINARY_MODULO"),
                Instr("CALL_FUNCTION", 1),
                Instr("RETURN_VALUE"),
            ]
        )
        co = code.to_code()
        # hopefully this is obvious from inspection? :-)
        self.assertEqual(co.co_stacksize, 3)

        co = code.to_code(stacksize=42)
        self.assertEqual(co.co_stacksize, 42)

    def test_negative_size_unary(self):
        opnames = (
            "UNARY_POSITIVE",
            "UNARY_NEGATIVE",
            "UNARY_NOT",
            "UNARY_INVERT",
        )
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr(opname)])
                with self.assertRaises(RuntimeError):
                    code.compute_stacksize()

    def test_negative_size_unary_with_disable_check_of_pre_and_post(self):
        opnames = (
            "UNARY_POSITIVE",
            "UNARY_NEGATIVE",
            "UNARY_NOT",
            "UNARY_INVERT",
        )
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr(opname)])
                co = code.to_code(check_pre_and_post=False)
                self.assertEqual(co.co_stacksize, 0)

    def test_negative_size_binary(self):
        opnames = (
            "BINARY_POWER",
            "BINARY_MULTIPLY",
            "BINARY_MATRIX_MULTIPLY",
            "BINARY_FLOOR_DIVIDE",
            "BINARY_TRUE_DIVIDE",
            "BINARY_MODULO",
            "BINARY_ADD",
            "BINARY_SUBTRACT",
            "BINARY_SUBSCR",
            "BINARY_LSHIFT",
            "BINARY_RSHIFT",
            "BINARY_AND",
            "BINARY_XOR",
            "BINARY_OR",
        )
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr("LOAD_CONST", 1), Instr(opname)])
                with self.assertRaises(RuntimeError):
                    code.compute_stacksize()

    def test_negative_size_binary_with_disable_check_of_pre_and_post(self):
        opnames = (
            "BINARY_POWER",
            "BINARY_MULTIPLY",
            "BINARY_MATRIX_MULTIPLY",
            "BINARY_FLOOR_DIVIDE",
            "BINARY_TRUE_DIVIDE",
            "BINARY_MODULO",
            "BINARY_ADD",
            "BINARY_SUBTRACT",
            "BINARY_SUBSCR",
            "BINARY_LSHIFT",
            "BINARY_RSHIFT",
            "BINARY_AND",
            "BINARY_XOR",
            "BINARY_OR",
        )
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr("LOAD_CONST", 1), Instr(opname)])
                co = code.to_code(check_pre_and_post=False)
                self.assertEqual(co.co_stacksize, 1)

    def test_negative_size_call(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("CALL_FUNCTION", 0)])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_negative_size_unpack(self):
        opnames = (
            "UNPACK_SEQUENCE",
            "UNPACK_EX",
        )
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr(opname, 1)])
                with self.assertRaises(RuntimeError):
                    code.compute_stacksize()

    def test_negative_size_build(self):
        opnames = (
            "BUILD_TUPLE",
            "BUILD_LIST",
            "BUILD_SET",
        )
        if sys.version_info >= (3, 6):
            opnames = (*opnames, "BUILD_STRING")

        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr(opname, 1)])
                with self.assertRaises(RuntimeError):
                    code.compute_stacksize()

    def test_negative_size_build_map(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("LOAD_CONST", 1), Instr("BUILD_MAP", 1)])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_negative_size_build_map_with_disable_check_of_pre_and_post(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("LOAD_CONST", 1), Instr("BUILD_MAP", 1)])
        co = code.to_code(check_pre_and_post=False)
        self.assertEqual(co.co_stacksize, 1)

    @unittest.skipIf(sys.version_info < (3, 6), "Inexistent opcode")
    def test_negative_size_build_const_map(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("LOAD_CONST", ("a",)), Instr("BUILD_CONST_KEY_MAP", 1)])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    @unittest.skipIf(sys.version_info < (3, 6), "Inexistent opcode")
    def test_negative_size_build_const_map_with_disable_check_of_pre_and_post(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("LOAD_CONST", ("a",)), Instr("BUILD_CONST_KEY_MAP", 1)])
        co = code.to_code(check_pre_and_post=False)
        self.assertEqual(co.co_stacksize, 1)

    def test_empty_dup(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("DUP_TOP")])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_not_enough_dup(self):
        code = Bytecode()
        code.first_lineno = 1
        code.extend([Instr("LOAD_CONST", 1), Instr("DUP_TOP_TWO")])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_not_enough_rot(self):
        opnames = ["ROT_TWO", "ROT_THREE"]
        if sys.version_info >= (3, 8):
            opnames.append("ROT_FOUR")
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr("LOAD_CONST", 1), Instr(opname)])
                with self.assertRaises(RuntimeError):
                    code.compute_stacksize()

    def test_not_enough_rot_with_disable_check_of_pre_and_post(self):
        opnames = ["ROT_TWO", "ROT_THREE"]
        if sys.version_info >= (3, 8):
            opnames.append("ROT_FOUR")
        for opname in opnames:
            with self.subTest():
                code = Bytecode()
                code.first_lineno = 1
                code.extend([Instr("LOAD_CONST", 1), Instr(opname)])
                co = code.to_code(check_pre_and_post=False)
                self.assertEqual(co.co_stacksize, 1)

    def test_for_iter_stack_effect_computation(self):
        with self.subTest():
            code = Bytecode()
            code.first_lineno = 1
            lab1 = Label()
            lab2 = Label()
            code.extend(
                [
                    lab1,
                    Instr("FOR_ITER", lab2),
                    Instr("STORE_FAST", "i"),
                    Instr("JUMP_ABSOLUTE", lab1),
                    lab2,
                ]
            )
            with self.assertRaises(RuntimeError):
                # Use compute_stacksize since the code is so broken that conversion
                # to from concrete is actually broken
                code.compute_stacksize(check_pre_and_post=False)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
