
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
#!/usr/bin/env python3
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
    Label,
    Compare,
    SetLineno,
    Instr,
    Bytecode,
    BasicBlock,
    ControlFlowGraph,
)
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase


def disassemble(
    source, *, filename="<string>", function=False, remove_last_return_none=False
):
    code = _disassemble(source, filename=filename, function=function)
    blocks = ControlFlowGraph.from_bytecode(code)
    if remove_last_return_none:
        # drop LOAD_CONST+RETURN_VALUE to only keep 2 instructions,
        # to make unit tests shorter
        block = blocks[-1]
        test = (
            block[-2].name == "LOAD_CONST"
            and block[-2].arg is None
            and block[-1].name == "RETURN_VALUE"
        )
        if not test:
            raise ValueError(
                "unable to find implicit RETURN_VALUE <None>: %s" % block[-2:]
            )
        del block[-2:]
    return blocks


class BlockTests(unittest.TestCase):
    def test_iter_invalid_types(self):
        # Labels are not allowed in basic blocks
        block = BasicBlock()
        block.append(Label())
        with self.assertRaises(ValueError):
            list(block)
        with self.assertRaises(ValueError):
            block.legalize(1)

        # Only one jump allowed and only at the end
        block = BasicBlock()
        block2 = BasicBlock()
        block.extend([Instr("JUMP_ABSOLUTE", block2), Instr("NOP")])
        with self.assertRaises(ValueError):
            list(block)
        with self.assertRaises(ValueError):
            block.legalize(1)

        # jump target must be a BasicBlock
        block = BasicBlock()
        label = Label()
        block.extend([Instr("JUMP_ABSOLUTE", label)])
        with self.assertRaises(ValueError):
            list(block)
        with self.assertRaises(ValueError):
            block.legalize(1)

    def test_slice(self):
        block = BasicBlock([Instr("NOP")])
        next_block = BasicBlock()
        block.next_block = next_block
        self.assertEqual(block, block[:])
        self.assertIs(next_block, block[:].next_block)

    def test_copy(self):
        block = BasicBlock([Instr("NOP")])
        next_block = BasicBlock()
        block.next_block = next_block
        self.assertEqual(block, block.copy())
        self.assertIs(next_block, block.copy().next_block)


class BytecodeBlocksTests(TestCase):
    maxDiff = 80 * 100

    def test_constructor(self):
        code = ControlFlowGraph()
        self.assertEqual(code.name, "<module>")
        self.assertEqual(code.filename, "<string>")
        self.assertEqual(code.flags, 0)
        self.assertBlocksEqual(code, [])

    def test_attr(self):
        source = """
            first_line = 1

            def func(arg1, arg2, *, arg3):
                x = 1
                y = 2
                return arg1
        """
        code = disassemble(source, filename="hello.py", function=True)
        self.assertEqual(code.argcount, 2)
        self.assertEqual(code.filename, "hello.py")
        self.assertEqual(code.first_lineno, 3)
        if sys.version_info > (3, 8):
            self.assertEqual(code.posonlyargcount, 0)
        self.assertEqual(code.kwonlyargcount, 1)
        self.assertEqual(code.name, "func")
        self.assertEqual(code.cellvars, [])

        code.name = "name"
        code.filename = "filename"
        code.flags = 123
        self.assertEqual(code.name, "name")
        self.assertEqual(code.filename, "filename")
        self.assertEqual(code.flags, 123)

        # FIXME: test non-empty cellvars

    def test_add_del_block(self):
        code = ControlFlowGraph()
        code[0].append(Instr("LOAD_CONST", 0))

        block = code.add_block()
        self.assertEqual(len(code), 2)
        self.assertIs(block, code[1])

        code[1].append(Instr("LOAD_CONST", 2))
        self.assertBlocksEqual(code, [Instr("LOAD_CONST", 0)], [Instr("LOAD_CONST", 2)])

        del code[0]
        self.assertBlocksEqual(code, [Instr("LOAD_CONST", 2)])

        del code[0]
        self.assertEqual(len(code), 0)

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

        blocks = ControlFlowGraph.from_bytecode(code)
        self.assertBlocksEqual(
            blocks,
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                SetLineno(4),
                Instr("LOAD_CONST", 8),
                Instr("STORE_NAME", "y"),
                SetLineno(5),
                Instr("LOAD_CONST", 9),
                Instr("STORE_NAME", "z"),
            ],
        )

    def test_legalize(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend(
            [
                Instr("LOAD_CONST", 7),
                Instr("STORE_NAME", "x"),
                Instr("LOAD_CONST", 8, lineno=4),
                Instr("STORE_NAME", "y"),
                SetLineno(5),
                Instr("LOAD_CONST", 9, lineno=6),
                Instr("STORE_NAME", "z"),
            ]
        )

        blocks = ControlFlowGraph.from_bytecode(code)
        blocks.legalize()
        self.assertBlocksEqual(
            blocks,
            [
                Instr("LOAD_CONST", 7, lineno=3),
                Instr("STORE_NAME", "x", lineno=3),
                Instr("LOAD_CONST", 8, lineno=4),
                Instr("STORE_NAME", "y", lineno=4),
                Instr("LOAD_CONST", 9, lineno=5),
                Instr("STORE_NAME", "z", lineno=5),
            ],
        )

    def test_repr(self):
        r = repr(ControlFlowGraph())
        self.assertIn("ControlFlowGraph", r)
        self.assertIn("1", r)

    def test_to_bytecode(self):
        # if test:
        #     x = 2
        # x = 5
        blocks = ControlFlowGraph()
        blocks.add_block()
        blocks.add_block()
        blocks[0].extend(
            [
                Instr("LOAD_NAME", "test", lineno=1),
                Instr("POP_JUMP_IF_FALSE", blocks[2], lineno=1),
            ]
        )

        blocks[1].extend(
            [
                Instr("LOAD_CONST", 5, lineno=2),
                Instr("STORE_NAME", "x", lineno=2),
                Instr("JUMP_FORWARD", blocks[2], lineno=2),
            ]
        )

        blocks[2].extend(
            [
                Instr("LOAD_CONST", 7, lineno=3),
                Instr("STORE_NAME", "x", lineno=3),
                Instr("LOAD_CONST", None, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
            ]
        )

        bytecode = blocks.to_bytecode()
        label = Label()
        self.assertEqual(
            bytecode,
            [
                Instr("LOAD_NAME", "test", lineno=1),
                Instr("POP_JUMP_IF_FALSE", label, lineno=1),
                Instr("LOAD_CONST", 5, lineno=2),
                Instr("STORE_NAME", "x", lineno=2),
                Instr("JUMP_FORWARD", label, lineno=2),
                label,
                Instr("LOAD_CONST", 7, lineno=3),
                Instr("STORE_NAME", "x", lineno=3),
                Instr("LOAD_CONST", None, lineno=3),
                Instr("RETURN_VALUE", lineno=3),
            ],
        )
        # FIXME: test other attributes

    def test_label_at_the_end(self):
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

        cfg = ControlFlowGraph.from_bytecode(code)
        self.assertBlocksEqual(
            cfg,
            [
                Instr("LOAD_NAME", "x"),
                Instr("UNARY_NOT"),
                Instr("POP_JUMP_IF_FALSE", cfg[2]),
            ],
            [Instr("LOAD_CONST", 9), Instr("STORE_NAME", "y")],
            [],
        )

    def test_from_bytecode(self):
        bytecode = Bytecode()
        label = Label()
        bytecode.extend(
            [
                Instr("LOAD_NAME", "test", lineno=1),
                Instr("POP_JUMP_IF_FALSE", label, lineno=1),
                Instr("LOAD_CONST", 5, lineno=2),
                Instr("STORE_NAME", "x", lineno=2),
                Instr("JUMP_FORWARD", label, lineno=2),
                # dead code!
                Instr("LOAD_CONST", 7, lineno=4),
                Instr("STORE_NAME", "x", lineno=4),
                Label(),  # unused label
                label,
                Label(),  # unused label
                Instr("LOAD_CONST", None, lineno=4),
                Instr("RETURN_VALUE", lineno=4),
            ]
        )

        blocks = ControlFlowGraph.from_bytecode(bytecode)
        label2 = blocks[3]
        self.assertBlocksEqual(
            blocks,
            [
                Instr("LOAD_NAME", "test", lineno=1),
                Instr("POP_JUMP_IF_FALSE", label2, lineno=1),
            ],
            [
                Instr("LOAD_CONST", 5, lineno=2),
                Instr("STORE_NAME", "x", lineno=2),
                Instr("JUMP_FORWARD", label2, lineno=2),
            ],
            [Instr("LOAD_CONST", 7, lineno=4), Instr("STORE_NAME", "x", lineno=4)],
            [Instr("LOAD_CONST", None, lineno=4), Instr("RETURN_VALUE", lineno=4)],
        )
        # FIXME: test other attributes

    def test_from_bytecode_loop(self):
        # for x in (1, 2, 3):
        #     if x == 2:
        #         break
        #     continue

        if sys.version_info < (3, 8):
            label_loop_start = Label()
            label_loop_exit = Label()
            label_loop_end = Label()

            code = Bytecode()
            code.extend(
                (
                    Instr("SETUP_LOOP", label_loop_end, lineno=1),
                    Instr("LOAD_CONST", (1, 2, 3), lineno=1),
                    Instr("GET_ITER", lineno=1),
                    label_loop_start,
                    Instr("FOR_ITER", label_loop_exit, lineno=1),
                    Instr("STORE_NAME", "x", lineno=1),
                    Instr("LOAD_NAME", "x", lineno=2),
                    Instr("LOAD_CONST", 2, lineno=2),
                    Instr("COMPARE_OP", Compare.EQ, lineno=2),
                    Instr("POP_JUMP_IF_FALSE", label_loop_start, lineno=2),
                    Instr("BREAK_LOOP", lineno=3),
                    Instr("JUMP_ABSOLUTE", label_loop_start, lineno=4),
                    Instr("JUMP_ABSOLUTE", label_loop_start, lineno=4),
                    label_loop_exit,
                    Instr("POP_BLOCK", lineno=4),
                    label_loop_end,
                    Instr("LOAD_CONST", None, lineno=4),
                    Instr("RETURN_VALUE", lineno=4),
                )
            )
            blocks = ControlFlowGraph.from_bytecode(code)

            expected = [
                [Instr("SETUP_LOOP", blocks[8], lineno=1)],
                [Instr("LOAD_CONST", (1, 2, 3), lineno=1), Instr("GET_ITER", lineno=1)],
                [Instr("FOR_ITER", blocks[7], lineno=1)],
                [
                    Instr("STORE_NAME", "x", lineno=1),
                    Instr("LOAD_NAME", "x", lineno=2),
                    Instr("LOAD_CONST", 2, lineno=2),
                    Instr("COMPARE_OP", Compare.EQ, lineno=2),
                    Instr("POP_JUMP_IF_FALSE", blocks[2], lineno=2),
                ],
                [Instr("BREAK_LOOP", lineno=3)],
                [Instr("JUMP_ABSOLUTE", blocks[2], lineno=4)],
                [Instr("JUMP_ABSOLUTE", blocks[2], lineno=4)],
                [Instr("POP_BLOCK", lineno=4)],
                [Instr("LOAD_CONST", None, lineno=4), Instr("RETURN_VALUE", lineno=4)],
            ]
            self.assertBlocksEqual(blocks, *expected)
        else:
            label_loop_start = Label()
            label_loop_exit = Label()

            code = Bytecode()
            code.extend(
                (
                    Instr("LOAD_CONST", (1, 2, 3), lineno=1),
                    Instr("GET_ITER", lineno=1),
                    label_loop_start,
                    Instr("FOR_ITER", label_loop_exit, lineno=1),
                    Instr("STORE_NAME", "x", lineno=1),
                    Instr("LOAD_NAME", "x", lineno=2),
                    Instr("LOAD_CONST", 2, lineno=2),
                    Instr("COMPARE_OP", Compare.EQ, lineno=2),
                    Instr("POP_JUMP_IF_FALSE", label_loop_start, lineno=2),
                    Instr("JUMP_ABSOLUTE", label_loop_exit, lineno=3),
                    Instr("JUMP_ABSOLUTE", label_loop_start, lineno=4),
                    Instr("JUMP_ABSOLUTE", label_loop_start, lineno=4),
                    label_loop_exit,
                    Instr("LOAD_CONST", None, lineno=4),
                    Instr("RETURN_VALUE", lineno=4),
                )
            )
            blocks = ControlFlowGraph.from_bytecode(code)

            expected = [
                [Instr("LOAD_CONST", (1, 2, 3), lineno=1), Instr("GET_ITER", lineno=1)],
                [Instr("FOR_ITER", blocks[6], lineno=1)],
                [
                    Instr("STORE_NAME", "x", lineno=1),
                    Instr("LOAD_NAME", "x", lineno=2),
                    Instr("LOAD_CONST", 2, lineno=2),
                    Instr("COMPARE_OP", Compare.EQ, lineno=2),
                    Instr("POP_JUMP_IF_FALSE", blocks[1], lineno=2),
                ],
                [Instr("JUMP_ABSOLUTE", blocks[6], lineno=3)],
                [Instr("JUMP_ABSOLUTE", blocks[1], lineno=4)],
                [Instr("JUMP_ABSOLUTE", blocks[1], lineno=4)],
                [Instr("LOAD_CONST", None, lineno=4), Instr("RETURN_VALUE", lineno=4)],
            ]
            self.assertBlocksEqual(blocks, *expected)


class BytecodeBlocksFunctionalTests(TestCase):
    def test_eq(self):
        # compare codes with multiple blocks and labels,
        # Code.__eq__() renumbers labels to get equal labels
        source = "x = 1 if test else 2"
        code1 = disassemble(source)
        code2 = disassemble(source)
        self.assertEqual(code1, code2)

        # Type mismatch
        self.assertFalse(code1 == 1)

        # argnames mismatch
        cfg = ControlFlowGraph()
        cfg.argnames = 10
        self.assertFalse(code1 == cfg)

        # instr mismatch
        cfg = ControlFlowGraph()
        cfg.argnames = code1.argnames
        self.assertFalse(code1 == cfg)

    def check_getitem(self, code):
        # check internal Code block indexes (index by index, index by label)
        for block_index, block in enumerate(code):
            self.assertIs(code[block_index], block)
            self.assertIs(code[block], block)
            self.assertEqual(code.get_block_index(block), block_index)

    def test_delitem(self):
        cfg = ControlFlowGraph()
        b = cfg.add_block()
        del cfg[b]
        self.assertEqual(len(cfg.get_instructions()), 0)

    def sample_code(self):
        code = disassemble("x = 1", remove_last_return_none=True)
        self.assertBlocksEqual(
            code, [Instr("LOAD_CONST", 1, lineno=1), Instr("STORE_NAME", "x", lineno=1)]
        )
        return code

    def test_split_block(self):
        code = self.sample_code()
        code[0].append(Instr("NOP", lineno=1))

        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(
            code,
            [Instr("LOAD_CONST", 1, lineno=1), Instr("STORE_NAME", "x", lineno=1)],
            [Instr("NOP", lineno=1)],
        )
        self.check_getitem(code)

        label2 = code.split_block(code[0], 1)
        self.assertIs(label2, code[1])
        self.assertBlocksEqual(
            code,
            [Instr("LOAD_CONST", 1, lineno=1)],
            [Instr("STORE_NAME", "x", lineno=1)],
            [Instr("NOP", lineno=1)],
        )
        self.check_getitem(code)

        with self.assertRaises(TypeError):
            code.split_block(1, 1)

        with self.assertRaises(ValueError) as e:
            code.split_block(code[0], -2)
        self.assertIn("positive", e.exception.args[0])

    def test_split_block_end(self):
        code = self.sample_code()

        # split at the end of the last block requires to add a new empty block
        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(
            code,
            [Instr("LOAD_CONST", 1, lineno=1), Instr("STORE_NAME", "x", lineno=1)],
            [],
        )
        self.check_getitem(code)

        # split at the end of a block which is not the end doesn't require to
        # add a new block
        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(
            code,
            [Instr("LOAD_CONST", 1, lineno=1), Instr("STORE_NAME", "x", lineno=1)],
            [],
        )

    def test_split_block_dont_split(self):
        code = self.sample_code()

        # FIXME: is it really useful to support that?
        block = code.split_block(code[0], 0)
        self.assertIs(block, code[0])
        self.assertBlocksEqual(
            code, [Instr("LOAD_CONST", 1, lineno=1), Instr("STORE_NAME", "x", lineno=1)]
        )

    def test_split_block_error(self):
        code = self.sample_code()

        with self.assertRaises(ValueError):
            # invalid index
            code.split_block(code[0], 3)

    def test_to_code(self):
        # test resolution of jump labels
        bytecode = ControlFlowGraph()
        bytecode.first_lineno = 3
        bytecode.argcount = 3
        if sys.version_info > (3, 8):
            bytecode.posonlyargcount = 0
        bytecode.kwonlyargcount = 2
        bytecode.name = "func"
        bytecode.filename = "hello.py"
        bytecode.flags = 0x43
        bytecode.argnames = ("arg", "arg2", "arg3", "kwonly", "kwonly2")
        bytecode.docstring = None
        block0 = bytecode[0]
        block1 = bytecode.add_block()
        block2 = bytecode.add_block()
        block0.extend(
            [
                Instr("LOAD_FAST", "x", lineno=4),
                Instr("POP_JUMP_IF_FALSE", block2, lineno=4),
            ]
        )
        block1.extend(
            [Instr("LOAD_FAST", "arg", lineno=5), Instr("STORE_FAST", "x", lineno=5)]
        )
        block2.extend(
            [
                Instr("LOAD_CONST", 3, lineno=6),
                Instr("STORE_FAST", "x", lineno=6),
                Instr("LOAD_FAST", "x", lineno=7),
                Instr("RETURN_VALUE", lineno=7),
            ]
        )

        if OFFSET_AS_INSTRUCTION:
            # The argument of the jump is divided by 2
            expected = (
                b"|\x05" b"r\x04" b"|\x00" b"}\x05" b"d\x01" b"}\x05" b"|\x05" b"S\x00"
            )
        else:
            expected = (
                b"|\x05" b"r\x08" b"|\x00" b"}\x05" b"d\x01" b"}\x05" b"|\x05" b"S\x00"
            )

        code = bytecode.to_code()
        self.assertEqual(code.co_consts, (None, 3))
        self.assertEqual(code.co_argcount, 3)
        if sys.version_info > (3, 8):
            self.assertEqual(code.co_posonlyargcount, 0)
        self.assertEqual(code.co_kwonlyargcount, 2)
        self.assertEqual(code.co_nlocals, 6)
        self.assertEqual(code.co_stacksize, 1)
        # FIXME: don't use hardcoded constants
        self.assertEqual(code.co_flags, 0x43)
        self.assertEqual(code.co_code, expected)
        self.assertEqual(code.co_names, ())
        self.assertEqual(
            code.co_varnames, ("arg", "arg2", "arg3", "kwonly", "kwonly2", "x")
        )
        self.assertEqual(code.co_filename, "hello.py")
        self.assertEqual(code.co_name, "func")
        self.assertEqual(code.co_firstlineno, 3)

        # verify stacksize argument is honored
        explicit_stacksize = code.co_stacksize + 42
        code = bytecode.to_code(stacksize=explicit_stacksize)
        self.assertEqual(code.co_stacksize, explicit_stacksize)

    def test_get_block_index(self):
        blocks = ControlFlowGraph()
        block0 = blocks[0]
        block1 = blocks.add_block()
        block2 = blocks.add_block()
        self.assertEqual(blocks.get_block_index(block0), 0)
        self.assertEqual(blocks.get_block_index(block1), 1)
        self.assertEqual(blocks.get_block_index(block2), 2)

        other_block = BasicBlock()
        self.assertRaises(ValueError, blocks.get_block_index, other_block)


class CFGStacksizeComputationTests(TestCase):
    def check_stack_size(self, func):
        code = func.__code__
        bytecode = Bytecode.from_code(code)
        cfg = ControlFlowGraph.from_bytecode(bytecode)
        self.assertEqual(code.co_stacksize, cfg.compute_stacksize())

    def test_empty_code(self):
        cfg = ControlFlowGraph()
        del cfg[0]
        self.assertEqual(cfg.compute_stacksize(), 0)

    def test_handling_of_set_lineno(self):
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
        self.assertEqual(code.compute_stacksize(), 1)

    def test_invalid_stacksize(self):
        code = Bytecode()
        code.extend([Instr("STORE_NAME", "x")])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_stack_size_computation_and(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            return arg1 and args  # Test JUMP_IF_FALSE_OR_POP

        self.check_stack_size(test)

    def test_stack_size_computation_or(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            return arg1 or args  # Test JUMP_IF_TRUE_OR_POP

        self.check_stack_size(test)

    def test_stack_size_computation_if_else(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            if args:
                return 0
            elif kwargs:
                return 1
            else:
                return 2

        self.check_stack_size(test)

    def test_stack_size_computation_for_loop_continue(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            for k in kwargs:
                if k in args:
                    continue
            else:
                return 1

        self.check_stack_size(test)

    def test_stack_size_computation_while_loop_break(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            while True:
                if arg1:
                    break

        self.check_stack_size(test)

    def test_stack_size_computation_with(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            with open(arg1) as f:
                return f.read()

        self.check_stack_size(test)

    def test_stack_size_computation_try_except(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            try:
                return args[0]
            except Exception:
                return 2

        self.check_stack_size(test)

    def test_stack_size_computation_try_finally(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            try:
                return args[0]
            finally:
                return 2

        self.check_stack_size(test)

    def test_stack_size_computation_try_except_finally(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            try:
                return args[0]
            except Exception:
                return 2
            finally:
                print("Interrupt")

        self.check_stack_size(test)

    def test_stack_size_computation_try_except_else_finally(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            try:
                return args[0]
            except Exception:
                return 2
            else:
                return arg1
            finally:
                print("Interrupt")

        self.check_stack_size(test)

    def test_stack_size_computation_nested_try_except_finally(self):
        def test(arg1, *args, **kwargs):  # pragma: no cover
            k = 1
            try:
                getattr(arg1, k)
            except AttributeError:
                pass
            except Exception:
                try:
                    assert False
                except Exception:
                    return 2
                finally:
                    print("unexpected")
            finally:
                print("attempted to get {}".format(k))

        self.check_stack_size(test)

    def test_stack_size_computation_nested_try_except_else_finally(self):
        def test(*args, **kwargs):
            try:
                v = args[1]
            except IndexError:
                try:
                    w = kwargs["value"]
                except KeyError:
                    return -1
                else:
                    return w
                finally:
                    print("second finally")
            else:
                return v
            finally:
                print("first finally")

        # A direct comparison of the stack depth fails because CPython
        # generate dead code that is used in stack computation.
        cpython_stacksize = test.__code__.co_stacksize
        test.__code__ = Bytecode.from_code(test.__code__).to_code()
        self.assertLessEqual(test.__code__.co_stacksize, cpython_stacksize)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test(1, 4), 4)
            self.assertEqual(stdout.getvalue(), "first finally\n")

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test([], value=3), 3)
            self.assertEqual(stdout.getvalue(), "second finally\nfirst finally\n")

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test([], name=None), -1)
            self.assertEqual(stdout.getvalue(), "second finally\nfirst finally\n")

    def test_stack_size_with_dead_code(self):
        # Simply demonstrate more directly the previously mentioned issue.
        def test(*args):  # pragma: no cover
            return 0
            try:
                a = args[0]
            except IndexError:
                return -1
            else:
                return a

        test.__code__ = Bytecode.from_code(test.__code__).to_code()
        self.assertEqual(test.__code__.co_stacksize, 1)
        self.assertEqual(test(1), 0)

    def test_huge_code_with_numerous_blocks(self):
        def base_func(x):
            pass

        def mk_if_then_else(depth):
            instructions = []
            for i in range(depth):
                label_else = Label()
                instructions.extend(
                    [
                        Instr("LOAD_FAST", "x"),
                        Instr("POP_JUMP_IF_FALSE", label_else),
                        Instr("LOAD_GLOBAL", "f{}".format(i)),
                        Instr("RETURN_VALUE"),
                        label_else,
                    ]
                )
            instructions.extend([Instr("LOAD_CONST", None), Instr("RETURN_VALUE")])
            return instructions

        bytecode = Bytecode(mk_if_then_else(5000))
        bytecode.compute_stacksize()


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
