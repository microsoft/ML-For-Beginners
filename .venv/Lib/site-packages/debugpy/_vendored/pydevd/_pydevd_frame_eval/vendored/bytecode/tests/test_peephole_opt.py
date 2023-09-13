
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock


class Tests(TestCase):

    maxDiff = 80 * 100

    def optimize_blocks(self, code):
        if isinstance(code, Bytecode):
            code = ControlFlowGraph.from_bytecode(code)
        optimizer = peephole_opt.PeepholeOptimizer()
        optimizer.optimize_cfg(code)
        return code

    def check(self, code, *expected):
        if isinstance(code, Bytecode):
            code = ControlFlowGraph.from_bytecode(code)
        optimizer = peephole_opt.PeepholeOptimizer()
        optimizer.optimize_cfg(code)
        code = code.to_bytecode()

        try:
            self.assertEqual(code, expected)
        except AssertionError:
            print("Optimized code:")
            dump_bytecode(code)

            print("Expected code:")
            for instr in expected:
                print(instr)

            raise

    def check_dont_optimize(self, code):
        code = ControlFlowGraph.from_bytecode(code)
        noopt = code.to_bytecode()

        optim = self.optimize_blocks(code)
        optim = optim.to_bytecode()
        self.assertEqual(optim, noopt)

    def test_unary_op(self):
        def check_unary_op(op, value, result):
            code = Bytecode(
                [Instr("LOAD_CONST", value), Instr(op), Instr("STORE_NAME", "x")]
            )
            self.check(code, Instr("LOAD_CONST", result), Instr("STORE_NAME", "x"))

        check_unary_op("UNARY_POSITIVE", 2, 2)
        check_unary_op("UNARY_NEGATIVE", 3, -3)
        check_unary_op("UNARY_INVERT", 5, -6)

    def test_binary_op(self):
        def check_bin_op(left, op, right, result):
            code = Bytecode(
                [
                    Instr("LOAD_CONST", left),
                    Instr("LOAD_CONST", right),
                    Instr(op),
                    Instr("STORE_NAME", "x"),
                ]
            )
            self.check(code, Instr("LOAD_CONST", result), Instr("STORE_NAME", "x"))

        check_bin_op(10, "BINARY_ADD", 20, 30)
        check_bin_op(5, "BINARY_SUBTRACT", 1, 4)
        check_bin_op(5, "BINARY_MULTIPLY", 3, 15)
        check_bin_op(10, "BINARY_TRUE_DIVIDE", 3, 10 / 3)
        check_bin_op(10, "BINARY_FLOOR_DIVIDE", 3, 3)
        check_bin_op(10, "BINARY_MODULO", 3, 1)
        check_bin_op(2, "BINARY_POWER", 8, 256)
        check_bin_op(1, "BINARY_LSHIFT", 3, 8)
        check_bin_op(16, "BINARY_RSHIFT", 3, 2)
        check_bin_op(10, "BINARY_AND", 3, 2)
        check_bin_op(2, "BINARY_OR", 3, 3)
        check_bin_op(2, "BINARY_XOR", 3, 1)

    def test_combined_unary_bin_ops(self):
        # x = 1 + 3 + 7
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 3),
                Instr("BINARY_ADD"),
                Instr("LOAD_CONST", 7),
                Instr("BINARY_ADD"),
                Instr("STORE_NAME", "x"),
            ]
        )
        self.check(code, Instr("LOAD_CONST", 11), Instr("STORE_NAME", "x"))

        # x = ~(~(5))
        code = Bytecode(
            [
                Instr("LOAD_CONST", 5),
                Instr("UNARY_INVERT"),
                Instr("UNARY_INVERT"),
                Instr("STORE_NAME", "x"),
            ]
        )
        self.check(code, Instr("LOAD_CONST", 5), Instr("STORE_NAME", "x"))

        # "events = [(0, 'call'), (1, 'line'), (-(3), 'call')]"
        code = Bytecode(
            [
                Instr("LOAD_CONST", 0),
                Instr("LOAD_CONST", "call"),
                Instr("BUILD_TUPLE", 2),
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", "line"),
                Instr("BUILD_TUPLE", 2),
                Instr("LOAD_CONST", 3),
                Instr("UNARY_NEGATIVE"),
                Instr("LOAD_CONST", "call"),
                Instr("BUILD_TUPLE", 2),
                Instr("BUILD_LIST", 3),
                Instr("STORE_NAME", "events"),
            ]
        )
        self.check(
            code,
            Instr("LOAD_CONST", (0, "call")),
            Instr("LOAD_CONST", (1, "line")),
            Instr("LOAD_CONST", (-3, "call")),
            Instr("BUILD_LIST", 3),
            Instr("STORE_NAME", "events"),
        )

        # 'x = (1,) + (0,) * 8'
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("BUILD_TUPLE", 1),
                Instr("LOAD_CONST", 0),
                Instr("BUILD_TUPLE", 1),
                Instr("LOAD_CONST", 8),
                Instr("BINARY_MULTIPLY"),
                Instr("BINARY_ADD"),
                Instr("STORE_NAME", "x"),
            ]
        )
        zeros = (0,) * 8
        result = (1,) + zeros
        self.check(code, Instr("LOAD_CONST", result), Instr("STORE_NAME", "x"))

    def test_max_size(self):
        max_size = 3
        with mock.patch.object(peephole_opt, "MAX_SIZE", max_size):
            # optimized binary operation: size <= maximum size
            #
            # (9,) * size
            size = max_size
            result = (9,) * size
            code = Bytecode(
                [
                    Instr("LOAD_CONST", 9),
                    Instr("BUILD_TUPLE", 1),
                    Instr("LOAD_CONST", size),
                    Instr("BINARY_MULTIPLY"),
                    Instr("STORE_NAME", "x"),
                ]
            )
            self.check(code, Instr("LOAD_CONST", result), Instr("STORE_NAME", "x"))

            # don't optimize  binary operation: size > maximum size
            #
            # x = (9,) * size
            size = max_size + 1
            code = Bytecode(
                [
                    Instr("LOAD_CONST", 9),
                    Instr("BUILD_TUPLE", 1),
                    Instr("LOAD_CONST", size),
                    Instr("BINARY_MULTIPLY"),
                    Instr("STORE_NAME", "x"),
                ]
            )
            self.check(
                code,
                Instr("LOAD_CONST", (9,)),
                Instr("LOAD_CONST", size),
                Instr("BINARY_MULTIPLY"),
                Instr("STORE_NAME", "x"),
            )

    def test_bin_op_dont_optimize(self):
        # 1 / 0
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 0),
                Instr("BINARY_TRUE_DIVIDE"),
                Instr("POP_TOP"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

        # 1 // 0
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 0),
                Instr("BINARY_FLOOR_DIVIDE"),
                Instr("POP_TOP"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

        # 1 % 0
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 0),
                Instr("BINARY_MODULO"),
                Instr("POP_TOP"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

        # 1 % 1j
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 1j),
                Instr("BINARY_MODULO"),
                Instr("POP_TOP"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

    def test_build_tuple(self):
        # x = (1, 2, 3)
        code = Bytecode(
            [
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 2),
                Instr("LOAD_CONST", 3),
                Instr("BUILD_TUPLE", 3),
                Instr("STORE_NAME", "x"),
            ]
        )
        self.check(code, Instr("LOAD_CONST", (1, 2, 3)), Instr("STORE_NAME", "x"))

    def test_build_list(self):
        # test = x in [1, 2, 3]
        code = Bytecode(
            [
                Instr("LOAD_NAME", "x"),
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 2),
                Instr("LOAD_CONST", 3),
                Instr("BUILD_LIST", 3),
                Instr("COMPARE_OP", Compare.IN),
                Instr("STORE_NAME", "test"),
            ]
        )

        self.check(
            code,
            Instr("LOAD_NAME", "x"),
            Instr("LOAD_CONST", (1, 2, 3)),
            Instr("COMPARE_OP", Compare.IN),
            Instr("STORE_NAME", "test"),
        )

    def test_build_list_unpack_seq(self):
        for build_list in ("BUILD_TUPLE", "BUILD_LIST"):
            # x, = [a]
            code = Bytecode(
                [
                    Instr("LOAD_NAME", "a"),
                    Instr(build_list, 1),
                    Instr("UNPACK_SEQUENCE", 1),
                    Instr("STORE_NAME", "x"),
                ]
            )
            self.check(code, Instr("LOAD_NAME", "a"), Instr("STORE_NAME", "x"))

            # x, y = [a, b]
            code = Bytecode(
                [
                    Instr("LOAD_NAME", "a"),
                    Instr("LOAD_NAME", "b"),
                    Instr(build_list, 2),
                    Instr("UNPACK_SEQUENCE", 2),
                    Instr("STORE_NAME", "x"),
                    Instr("STORE_NAME", "y"),
                ]
            )
            self.check(
                code,
                Instr("LOAD_NAME", "a"),
                Instr("LOAD_NAME", "b"),
                Instr("ROT_TWO"),
                Instr("STORE_NAME", "x"),
                Instr("STORE_NAME", "y"),
            )

            # x, y, z = [a, b, c]
            code = Bytecode(
                [
                    Instr("LOAD_NAME", "a"),
                    Instr("LOAD_NAME", "b"),
                    Instr("LOAD_NAME", "c"),
                    Instr(build_list, 3),
                    Instr("UNPACK_SEQUENCE", 3),
                    Instr("STORE_NAME", "x"),
                    Instr("STORE_NAME", "y"),
                    Instr("STORE_NAME", "z"),
                ]
            )
            self.check(
                code,
                Instr("LOAD_NAME", "a"),
                Instr("LOAD_NAME", "b"),
                Instr("LOAD_NAME", "c"),
                Instr("ROT_THREE"),
                Instr("ROT_TWO"),
                Instr("STORE_NAME", "x"),
                Instr("STORE_NAME", "y"),
                Instr("STORE_NAME", "z"),
            )

    def test_build_tuple_unpack_seq_const(self):
        # x, y = (3, 4)
        code = Bytecode(
            [
                Instr("LOAD_CONST", 3),
                Instr("LOAD_CONST", 4),
                Instr("BUILD_TUPLE", 2),
                Instr("UNPACK_SEQUENCE", 2),
                Instr("STORE_NAME", "x"),
                Instr("STORE_NAME", "y"),
            ]
        )
        self.check(
            code,
            Instr("LOAD_CONST", (3, 4)),
            Instr("UNPACK_SEQUENCE", 2),
            Instr("STORE_NAME", "x"),
            Instr("STORE_NAME", "y"),
        )

    def test_build_list_unpack_seq_const(self):
        # x, y, z = [3, 4, 5]
        code = Bytecode(
            [
                Instr("LOAD_CONST", 3),
                Instr("LOAD_CONST", 4),
                Instr("LOAD_CONST", 5),
                Instr("BUILD_LIST", 3),
                Instr("UNPACK_SEQUENCE", 3),
                Instr("STORE_NAME", "x"),
                Instr("STORE_NAME", "y"),
                Instr("STORE_NAME", "z"),
            ]
        )
        self.check(
            code,
            Instr("LOAD_CONST", 5),
            Instr("LOAD_CONST", 4),
            Instr("LOAD_CONST", 3),
            Instr("STORE_NAME", "x"),
            Instr("STORE_NAME", "y"),
            Instr("STORE_NAME", "z"),
        )

    def test_build_set(self):
        # test = x in {1, 2, 3}
        code = Bytecode(
            [
                Instr("LOAD_NAME", "x"),
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", 2),
                Instr("LOAD_CONST", 3),
                Instr("BUILD_SET", 3),
                Instr("COMPARE_OP", Compare.IN),
                Instr("STORE_NAME", "test"),
            ]
        )

        self.check(
            code,
            Instr("LOAD_NAME", "x"),
            Instr("LOAD_CONST", frozenset((1, 2, 3))),
            Instr("COMPARE_OP", Compare.IN),
            Instr("STORE_NAME", "test"),
        )

    def test_compare_op_unary_not(self):
        for op, not_op in (
            (Compare.IN, Compare.NOT_IN),  # in => not in
            (Compare.NOT_IN, Compare.IN),  # not in => in
            (Compare.IS, Compare.IS_NOT),  # is => is not
            (Compare.IS_NOT, Compare.IS),  # is not => is
        ):
            code = Bytecode(
                [
                    Instr("LOAD_NAME", "a"),
                    Instr("LOAD_NAME", "b"),
                    Instr("COMPARE_OP", op),
                    Instr("UNARY_NOT"),
                    Instr("STORE_NAME", "x"),
                ]
            )
            self.check(
                code,
                Instr("LOAD_NAME", "a"),
                Instr("LOAD_NAME", "b"),
                Instr("COMPARE_OP", not_op),
                Instr("STORE_NAME", "x"),
            )

        # don't optimize:
        # x = not (a and b is True)
        label_instr5 = Label()
        code = Bytecode(
            [
                Instr("LOAD_NAME", "a"),
                Instr("JUMP_IF_FALSE_OR_POP", label_instr5),
                Instr("LOAD_NAME", "b"),
                Instr("LOAD_CONST", True),
                Instr("COMPARE_OP", Compare.IS),
                label_instr5,
                Instr("UNARY_NOT"),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

    def test_dont_optimize(self):
        # x = 3 < 5
        code = Bytecode(
            [
                Instr("LOAD_CONST", 3),
                Instr("LOAD_CONST", 5),
                Instr("COMPARE_OP", Compare.LT),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        self.check_dont_optimize(code)

        # x = (10, 20, 30)[1:]
        code = Bytecode(
            [
                Instr("LOAD_CONST", (10, 20, 30)),
                Instr("LOAD_CONST", 1),
                Instr("LOAD_CONST", None),
                Instr("BUILD_SLICE", 2),
                Instr("BINARY_SUBSCR"),
                Instr("STORE_NAME", "x"),
            ]
        )
        self.check_dont_optimize(code)

    def test_optimize_code_obj(self):
        # Test optimize() method with a code object
        #
        # x = 3 + 5 => x = 8
        noopt = Bytecode(
            [
                Instr("LOAD_CONST", 3),
                Instr("LOAD_CONST", 5),
                Instr("BINARY_ADD"),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )
        noopt = noopt.to_code()

        optimizer = peephole_opt.PeepholeOptimizer()
        optim = optimizer.optimize(noopt)

        code = Bytecode.from_code(optim)
        self.assertEqual(
            code,
            [
                Instr("LOAD_CONST", 8, lineno=1),
                Instr("STORE_NAME", "x", lineno=1),
                Instr("LOAD_CONST", None, lineno=1),
                Instr("RETURN_VALUE", lineno=1),
            ],
        )

    def test_return_value(self):
        # return+return: remove second return
        #
        #     def func():
        #         return 4
        #         return 5
        code = Bytecode(
            [
                Instr("LOAD_CONST", 4, lineno=2),
                Instr("RETURN_VALUE", lineno=2),
                Instr("LOAD_CONST", 5, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
            ]
        )
        code = ControlFlowGraph.from_bytecode(code)
        self.check(
            code, Instr("LOAD_CONST", 4, lineno=2), Instr("RETURN_VALUE", lineno=2)
        )

        # return+return + return+return: remove second and fourth return
        #
        #     def func():
        #         return 4
        #         return 5
        #         return 6
        #         return 7
        code = Bytecode(
            [
                Instr("LOAD_CONST", 4, lineno=2),
                Instr("RETURN_VALUE", lineno=2),
                Instr("LOAD_CONST", 5, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
                Instr("LOAD_CONST", 6, lineno=4),
                Instr("RETURN_VALUE", lineno=4),
                Instr("LOAD_CONST", 7, lineno=5),
                Instr("RETURN_VALUE", lineno=5),
            ]
        )
        code = ControlFlowGraph.from_bytecode(code)
        self.check(
            code, Instr("LOAD_CONST", 4, lineno=2), Instr("RETURN_VALUE", lineno=2)
        )

        # return + JUMP_ABSOLUTE: remove JUMP_ABSOLUTE
        # while 1:
        #     return 7
        if sys.version_info < (3, 8):
            setup_loop = Label()
            return_label = Label()
            code = Bytecode(
                [
                    setup_loop,
                    Instr("SETUP_LOOP", return_label, lineno=2),
                    Instr("LOAD_CONST", 7, lineno=3),
                    Instr("RETURN_VALUE", lineno=3),
                    Instr("JUMP_ABSOLUTE", setup_loop, lineno=3),
                    Instr("POP_BLOCK", lineno=3),
                    return_label,
                    Instr("LOAD_CONST", None, lineno=3),
                    Instr("RETURN_VALUE", lineno=3),
                ]
            )
            code = ControlFlowGraph.from_bytecode(code)

            end_loop = Label()
            self.check(
                code,
                Instr("SETUP_LOOP", end_loop, lineno=2),
                Instr("LOAD_CONST", 7, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
                end_loop,
                Instr("LOAD_CONST", None, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
            )
        else:
            setup_loop = Label()
            return_label = Label()
            code = Bytecode(
                [
                    setup_loop,
                    Instr("LOAD_CONST", 7, lineno=3),
                    Instr("RETURN_VALUE", lineno=3),
                    Instr("JUMP_ABSOLUTE", setup_loop, lineno=3),
                    Instr("LOAD_CONST", None, lineno=3),
                    Instr("RETURN_VALUE", lineno=3),
                ]
            )
            code = ControlFlowGraph.from_bytecode(code)

            self.check(
                code, Instr("LOAD_CONST", 7, lineno=3), Instr("RETURN_VALUE", lineno=3)
            )

    def test_not_jump_if_false(self):
        # Replace UNARY_NOT+POP_JUMP_IF_FALSE with POP_JUMP_IF_TRUE
        #
        # if not x:
        #     y = 9
        label = Label()
        code = Bytecode(
            [
                Instr("LOAD_NAME", "x"),
                Instr("UNARY_NOT"),
                Instr("POP_JUMP_IF_FALSE", label),
                Instr("LOAD_CONST", 9),
                Instr("STORE_NAME", "y"),
                label,
            ]
        )

        code = self.optimize_blocks(code)
        label = Label()
        self.check(
            code,
            Instr("LOAD_NAME", "x"),
            Instr("POP_JUMP_IF_TRUE", label),
            Instr("LOAD_CONST", 9),
            Instr("STORE_NAME", "y"),
            label,
        )

    def test_unconditional_jump_to_return(self):
        # def func():
        #     if test:
        #         if test2:
        #             x = 10
        #         else:
        #             x = 20
        #     else:
        #         x = 30

        label_instr11 = Label()
        label_instr14 = Label()
        label_instr7 = Label()
        code = Bytecode(
            [
                Instr("LOAD_GLOBAL", "test", lineno=2),
                Instr("POP_JUMP_IF_FALSE", label_instr11, lineno=2),
                Instr("LOAD_GLOBAL", "test2", lineno=3),
                Instr("POP_JUMP_IF_FALSE", label_instr7, lineno=3),
                Instr("LOAD_CONST", 10, lineno=4),
                Instr("STORE_FAST", "x", lineno=4),
                Instr("JUMP_ABSOLUTE", label_instr14, lineno=4),
                label_instr7,
                Instr("LOAD_CONST", 20, lineno=6),
                Instr("STORE_FAST", "x", lineno=6),
                Instr("JUMP_FORWARD", label_instr14, lineno=6),
                label_instr11,
                Instr("LOAD_CONST", 30, lineno=8),
                Instr("STORE_FAST", "x", lineno=8),
                label_instr14,
                Instr("LOAD_CONST", None, lineno=8),
                Instr("RETURN_VALUE", lineno=8),
            ]
        )

        label1 = Label()
        label3 = Label()
        label4 = Label()
        self.check(
            code,
            Instr("LOAD_GLOBAL", "test", lineno=2),
            Instr("POP_JUMP_IF_FALSE", label3, lineno=2),
            Instr("LOAD_GLOBAL", "test2", lineno=3),
            Instr("POP_JUMP_IF_FALSE", label1, lineno=3),
            Instr("LOAD_CONST", 10, lineno=4),
            Instr("STORE_FAST", "x", lineno=4),
            Instr("JUMP_ABSOLUTE", label4, lineno=4),
            label1,
            Instr("LOAD_CONST", 20, lineno=6),
            Instr("STORE_FAST", "x", lineno=6),
            Instr("JUMP_FORWARD", label4, lineno=6),
            label3,
            Instr("LOAD_CONST", 30, lineno=8),
            Instr("STORE_FAST", "x", lineno=8),
            label4,
            Instr("LOAD_CONST", None, lineno=8),
            Instr("RETURN_VALUE", lineno=8),
        )

    def test_unconditional_jumps(self):
        # def func():
        #     if x:
        #         if y:
        #             func()
        label_instr7 = Label()
        code = Bytecode(
            [
                Instr("LOAD_GLOBAL", "x", lineno=2),
                Instr("POP_JUMP_IF_FALSE", label_instr7, lineno=2),
                Instr("LOAD_GLOBAL", "y", lineno=3),
                Instr("POP_JUMP_IF_FALSE", label_instr7, lineno=3),
                Instr("LOAD_GLOBAL", "func", lineno=4),
                Instr("CALL_FUNCTION", 0, lineno=4),
                Instr("POP_TOP", lineno=4),
                label_instr7,
                Instr("LOAD_CONST", None, lineno=4),
                Instr("RETURN_VALUE", lineno=4),
            ]
        )

        label_return = Label()
        self.check(
            code,
            Instr("LOAD_GLOBAL", "x", lineno=2),
            Instr("POP_JUMP_IF_FALSE", label_return, lineno=2),
            Instr("LOAD_GLOBAL", "y", lineno=3),
            Instr("POP_JUMP_IF_FALSE", label_return, lineno=3),
            Instr("LOAD_GLOBAL", "func", lineno=4),
            Instr("CALL_FUNCTION", 0, lineno=4),
            Instr("POP_TOP", lineno=4),
            label_return,
            Instr("LOAD_CONST", None, lineno=4),
            Instr("RETURN_VALUE", lineno=4),
        )

    def test_jump_to_return(self):
        # def func(condition):
        #     return 'yes' if condition else 'no'
        label_instr4 = Label()
        label_instr6 = Label()
        code = Bytecode(
            [
                Instr("LOAD_FAST", "condition"),
                Instr("POP_JUMP_IF_FALSE", label_instr4),
                Instr("LOAD_CONST", "yes"),
                Instr("JUMP_FORWARD", label_instr6),
                label_instr4,
                Instr("LOAD_CONST", "no"),
                label_instr6,
                Instr("RETURN_VALUE"),
            ]
        )

        label = Label()
        self.check(
            code,
            Instr("LOAD_FAST", "condition"),
            Instr("POP_JUMP_IF_FALSE", label),
            Instr("LOAD_CONST", "yes"),
            Instr("RETURN_VALUE"),
            label,
            Instr("LOAD_CONST", "no"),
            Instr("RETURN_VALUE"),
        )

    def test_jump_if_true_to_jump_if_false(self):
        # Replace JUMP_IF_TRUE_OR_POP jumping to POP_JUMP_IF_FALSE <target>
        # with POP_JUMP_IF_TRUE <offset after the second POP_JUMP_IF_FALSE>
        #
        #     if x or y:
        #         z = 1

        label_instr3 = Label()
        label_instr7 = Label()
        code = Bytecode(
            [
                Instr("LOAD_NAME", "x"),
                Instr("JUMP_IF_TRUE_OR_POP", label_instr3),
                Instr("LOAD_NAME", "y"),
                label_instr3,
                Instr("POP_JUMP_IF_FALSE", label_instr7),
                Instr("LOAD_CONST", 1),
                Instr("STORE_NAME", "z"),
                label_instr7,
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )

        label_instr4 = Label()
        label_instr7 = Label()
        self.check(
            code,
            Instr("LOAD_NAME", "x"),
            Instr("POP_JUMP_IF_TRUE", label_instr4),
            Instr("LOAD_NAME", "y"),
            Instr("POP_JUMP_IF_FALSE", label_instr7),
            label_instr4,
            Instr("LOAD_CONST", 1),
            Instr("STORE_NAME", "z"),
            label_instr7,
            Instr("LOAD_CONST", None),
            Instr("RETURN_VALUE"),
        )

    def test_jump_if_false_to_jump_if_false(self):
        # Replace JUMP_IF_FALSE_OR_POP jumping to POP_JUMP_IF_FALSE <label>
        # with POP_JUMP_IF_FALSE <label>
        #
        #     while n > 0 and start > 3:
        #         func()
        if sys.version_info < (3, 8):
            label_instr1 = Label()
            label_instr15 = Label()
            label_instr17 = Label()
            label_instr9 = Label()
            code = Bytecode(
                [
                    Instr("SETUP_LOOP", label_instr17),
                    label_instr1,
                    Instr("LOAD_NAME", "n"),
                    Instr("LOAD_CONST", 0),
                    Instr("COMPARE_OP", Compare.GT),
                    # JUMP_IF_FALSE_OR_POP jumps to POP_JUMP_IF_FALSE
                    # which jumps to label_instr15
                    Instr("JUMP_IF_FALSE_OR_POP", label_instr9),
                    Instr("LOAD_NAME", "start"),
                    Instr("LOAD_CONST", 3),
                    Instr("COMPARE_OP", Compare.GT),
                    label_instr9,
                    Instr("POP_JUMP_IF_FALSE", label_instr15),
                    Instr("LOAD_NAME", "func"),
                    Instr("CALL_FUNCTION", 0),
                    Instr("POP_TOP"),
                    Instr("JUMP_ABSOLUTE", label_instr1),
                    label_instr15,
                    Instr("POP_BLOCK"),
                    label_instr17,
                    Instr("LOAD_CONST", None),
                    Instr("RETURN_VALUE"),
                ]
            )

            label_instr1 = Label()
            label_instr14 = Label()
            label_instr16 = Label()
            self.check(
                code,
                Instr("SETUP_LOOP", label_instr16),
                label_instr1,
                Instr("LOAD_NAME", "n"),
                Instr("LOAD_CONST", 0),
                Instr("COMPARE_OP", Compare.GT),
                Instr("POP_JUMP_IF_FALSE", label_instr14),
                Instr("LOAD_NAME", "start"),
                Instr("LOAD_CONST", 3),
                Instr("COMPARE_OP", Compare.GT),
                Instr("POP_JUMP_IF_FALSE", label_instr14),
                Instr("LOAD_NAME", "func"),
                Instr("CALL_FUNCTION", 0),
                Instr("POP_TOP"),
                Instr("JUMP_ABSOLUTE", label_instr1),
                label_instr14,
                Instr("POP_BLOCK"),
                label_instr16,
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            )
        else:
            label_instr1 = Label()
            label_instr15 = Label()
            label_instr9 = Label()
            code = Bytecode(
                [
                    label_instr1,
                    Instr("LOAD_NAME", "n"),
                    Instr("LOAD_CONST", 0),
                    Instr("COMPARE_OP", Compare.GT),
                    # JUMP_IF_FALSE_OR_POP jumps to POP_JUMP_IF_FALSE
                    # which jumps to label_instr15
                    Instr("JUMP_IF_FALSE_OR_POP", label_instr9),
                    Instr("LOAD_NAME", "start"),
                    Instr("LOAD_CONST", 3),
                    Instr("COMPARE_OP", Compare.GT),
                    label_instr9,
                    Instr("POP_JUMP_IF_FALSE", label_instr15),
                    Instr("LOAD_NAME", "func"),
                    Instr("CALL_FUNCTION", 0),
                    Instr("POP_TOP"),
                    Instr("JUMP_ABSOLUTE", label_instr1),
                    label_instr15,
                    Instr("LOAD_CONST", None),
                    Instr("RETURN_VALUE"),
                ]
            )

            label_instr1 = Label()
            label_instr14 = Label()
            self.check(
                code,
                label_instr1,
                Instr("LOAD_NAME", "n"),
                Instr("LOAD_CONST", 0),
                Instr("COMPARE_OP", Compare.GT),
                Instr("POP_JUMP_IF_FALSE", label_instr14),
                Instr("LOAD_NAME", "start"),
                Instr("LOAD_CONST", 3),
                Instr("COMPARE_OP", Compare.GT),
                Instr("POP_JUMP_IF_FALSE", label_instr14),
                Instr("LOAD_NAME", "func"),
                Instr("CALL_FUNCTION", 0),
                Instr("POP_TOP"),
                Instr("JUMP_ABSOLUTE", label_instr1),
                label_instr14,
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            )

    def test_nop(self):
        code = Bytecode(
            [Instr("LOAD_NAME", "x"), Instr("NOP"), Instr("STORE_NAME", "test")]
        )

        self.check(code, Instr("LOAD_NAME", "x"), Instr("STORE_NAME", "test"))

    def test_dead_code_jump(self):
        label = Label()
        code = Bytecode(
            [
                Instr("LOAD_NAME", "x"),
                Instr("JUMP_ABSOLUTE", label),
                # dead code
                Instr("LOAD_NAME", "y"),
                Instr("STORE_NAME", "test"),
                label,
                Instr("STORE_NAME", "test"),
            ]
        )

        self.check(code, Instr("LOAD_NAME", "x"), Instr("STORE_NAME", "test"))

    def test_uncond_jump_to_uncond_jump(self):
        # Replace JUMP_FORWARD t1 jumping to JUMP_FORWARD t2
        # with JUMP_ABSOLUTE t2

        label = Label()
        label2 = Label()
        label3 = Label()
        label4 = Label()
        code = Bytecode(
            [
                Instr("LOAD_NAME", "test"),
                Instr("POP_JUMP_IF_TRUE", label),
                # redundant jump
                Instr("JUMP_FORWARD", label2),
                label,
                Instr("LOAD_CONST", 1),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_NAME", "test"),
                Instr("POP_JUMP_IF_TRUE", label3),
                label2,
                Instr("JUMP_FORWARD", label4),
                label3,
                Instr("LOAD_CONST", 1),
                Instr("STORE_NAME", "x"),
                label4,
                Instr("LOAD_CONST", None),
                Instr("RETURN_VALUE"),
            ]
        )

        label = Label()
        label3 = Label()
        label4 = Label()
        self.check(
            code,
            Instr("LOAD_NAME", "test"),
            Instr("POP_JUMP_IF_TRUE", label),
            # JUMP_FORWARD label2 was replaced with JUMP_ABSOLUTE label4
            Instr("JUMP_ABSOLUTE", label4),
            label,
            Instr("LOAD_CONST", 1),
            Instr("STORE_NAME", "x"),
            Instr("LOAD_NAME", "test"),
            Instr("POP_JUMP_IF_TRUE", label3),
            Instr("JUMP_FORWARD", label4),
            label3,
            Instr("LOAD_CONST", 1),
            Instr("STORE_NAME", "x"),
            label4,
            Instr("LOAD_CONST", None),
            Instr("RETURN_VALUE"),
        )


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
