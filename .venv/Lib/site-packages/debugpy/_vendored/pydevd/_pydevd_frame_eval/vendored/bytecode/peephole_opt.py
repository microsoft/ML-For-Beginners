"""
Peephole optimizer of CPython 3.6 reimplemented in pure Python using
the bytecode module.
"""
import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare

JUMPS_ON_TRUE = frozenset(
    (
        "POP_JUMP_IF_TRUE",
        "JUMP_IF_TRUE_OR_POP",
    )
)

NOT_COMPARE = {
    Compare.IN: Compare.NOT_IN,
    Compare.NOT_IN: Compare.IN,
    Compare.IS: Compare.IS_NOT,
    Compare.IS_NOT: Compare.IS,
}

MAX_SIZE = 20


class ExitUnchanged(Exception):
    """Exception used to skip the peephole optimizer"""

    pass


class PeepholeOptimizer:
    """Python reimplementation of the peephole optimizer.

    Copy of the C comment:

    Perform basic peephole optimizations to components of a code object.
    The consts object should still be in list form to allow new constants
    to be appended.

    To keep the optimizer simple, it bails out (does nothing) for code that
    has a length over 32,700, and does not calculate extended arguments.
    That allows us to avoid overflow and sign issues. Likewise, it bails when
    the lineno table has complex encoding for gaps >= 255. EXTENDED_ARG can
    appear before MAKE_FUNCTION; in this case both opcodes are skipped.
    EXTENDED_ARG preceding any other opcode causes the optimizer to bail.

    Optimizations are restricted to simple transformations occuring within a
    single basic block.  All transformations keep the code size the same or
    smaller.  For those that reduce size, the gaps are initially filled with
    NOPs.  Later those NOPs are removed and the jump addresses retargeted in
    a single pass.  Code offset is adjusted accordingly.
    """

    def __init__(self):
        # bytecode.ControlFlowGraph instance
        self.code = None
        self.const_stack = None
        self.block_index = None
        self.block = None
        # index of the current instruction in self.block instructions
        self.index = None
        # whether we are in a LOAD_CONST sequence
        self.in_consts = False

    def check_result(self, value):
        try:
            size = len(value)
        except TypeError:
            return True
        return size <= MAX_SIZE

    def replace_load_const(self, nconst, instr, result):
        # FIXME: remove temporary computed constants?
        # FIXME: or at least reuse existing constants?

        self.in_consts = True

        load_const = Instr("LOAD_CONST", result, lineno=instr.lineno)
        start = self.index - nconst - 1
        self.block[start : self.index] = (load_const,)
        self.index -= nconst

        if nconst:
            del self.const_stack[-nconst:]
        self.const_stack.append(result)
        self.in_consts = True

    def eval_LOAD_CONST(self, instr):
        self.in_consts = True
        value = instr.arg
        self.const_stack.append(value)
        self.in_consts = True

    def unaryop(self, op, instr):
        try:
            value = self.const_stack[-1]
            result = op(value)
        except IndexError:
            return

        if not self.check_result(result):
            return

        self.replace_load_const(1, instr, result)

    def eval_UNARY_POSITIVE(self, instr):
        return self.unaryop(operator.pos, instr)

    def eval_UNARY_NEGATIVE(self, instr):
        return self.unaryop(operator.neg, instr)

    def eval_UNARY_INVERT(self, instr):
        return self.unaryop(operator.invert, instr)

    def get_next_instr(self, name):
        try:
            next_instr = self.block[self.index]
        except IndexError:
            return None
        if next_instr.name == name:
            return next_instr
        return None

    def eval_UNARY_NOT(self, instr):
        # Note: UNARY_NOT <const> is not optimized

        next_instr = self.get_next_instr("POP_JUMP_IF_FALSE")
        if next_instr is None:
            return None

        # Replace UNARY_NOT+POP_JUMP_IF_FALSE with POP_JUMP_IF_TRUE
        instr.set("POP_JUMP_IF_TRUE", next_instr.arg)
        del self.block[self.index]

    def binop(self, op, instr):
        try:
            left = self.const_stack[-2]
            right = self.const_stack[-1]
        except IndexError:
            return

        try:
            result = op(left, right)
        except Exception:
            return

        if not self.check_result(result):
            return

        self.replace_load_const(2, instr, result)

    def eval_BINARY_ADD(self, instr):
        return self.binop(operator.add, instr)

    def eval_BINARY_SUBTRACT(self, instr):
        return self.binop(operator.sub, instr)

    def eval_BINARY_MULTIPLY(self, instr):
        return self.binop(operator.mul, instr)

    def eval_BINARY_TRUE_DIVIDE(self, instr):
        return self.binop(operator.truediv, instr)

    def eval_BINARY_FLOOR_DIVIDE(self, instr):
        return self.binop(operator.floordiv, instr)

    def eval_BINARY_MODULO(self, instr):
        return self.binop(operator.mod, instr)

    def eval_BINARY_POWER(self, instr):
        return self.binop(operator.pow, instr)

    def eval_BINARY_LSHIFT(self, instr):
        return self.binop(operator.lshift, instr)

    def eval_BINARY_RSHIFT(self, instr):
        return self.binop(operator.rshift, instr)

    def eval_BINARY_AND(self, instr):
        return self.binop(operator.and_, instr)

    def eval_BINARY_OR(self, instr):
        return self.binop(operator.or_, instr)

    def eval_BINARY_XOR(self, instr):
        return self.binop(operator.xor, instr)

    def eval_BINARY_SUBSCR(self, instr):
        return self.binop(operator.getitem, instr)

    def replace_container_of_consts(self, instr, container_type):
        items = self.const_stack[-instr.arg :]
        value = container_type(items)
        self.replace_load_const(instr.arg, instr, value)

    def build_tuple_unpack_seq(self, instr):
        next_instr = self.get_next_instr("UNPACK_SEQUENCE")
        if next_instr is None or next_instr.arg != instr.arg:
            return

        if instr.arg < 1:
            return

        if self.const_stack and instr.arg <= len(self.const_stack):
            nconst = instr.arg
            start = self.index - 1

            # Rewrite LOAD_CONST instructions in the reverse order
            load_consts = self.block[start - nconst : start]
            self.block[start - nconst : start] = reversed(load_consts)

            # Remove BUILD_TUPLE+UNPACK_SEQUENCE
            self.block[start : start + 2] = ()
            self.index -= 2
            self.const_stack.clear()
            return

        if instr.arg == 1:
            # Replace BUILD_TUPLE 1 + UNPACK_SEQUENCE 1 with NOP
            del self.block[self.index - 1 : self.index + 1]
        elif instr.arg == 2:
            # Replace BUILD_TUPLE 2 + UNPACK_SEQUENCE 2 with ROT_TWO
            rot2 = Instr("ROT_TWO", lineno=instr.lineno)
            self.block[self.index - 1 : self.index + 1] = (rot2,)
            self.index -= 1
            self.const_stack.clear()
        elif instr.arg == 3:
            # Replace BUILD_TUPLE 3 + UNPACK_SEQUENCE 3
            # with ROT_THREE + ROT_TWO
            rot3 = Instr("ROT_THREE", lineno=instr.lineno)
            rot2 = Instr("ROT_TWO", lineno=instr.lineno)
            self.block[self.index - 1 : self.index + 1] = (rot3, rot2)
            self.index -= 1
            self.const_stack.clear()

    def build_tuple(self, instr, container_type):
        if instr.arg > len(self.const_stack):
            return

        next_instr = self.get_next_instr("COMPARE_OP")
        if next_instr is None or next_instr.arg not in (Compare.IN, Compare.NOT_IN):
            return

        self.replace_container_of_consts(instr, container_type)
        return True

    def eval_BUILD_TUPLE(self, instr):
        if not instr.arg:
            return

        if instr.arg <= len(self.const_stack):
            self.replace_container_of_consts(instr, tuple)
        else:
            self.build_tuple_unpack_seq(instr)

    def eval_BUILD_LIST(self, instr):
        if not instr.arg:
            return

        if not self.build_tuple(instr, tuple):
            self.build_tuple_unpack_seq(instr)

    def eval_BUILD_SET(self, instr):
        if not instr.arg:
            return

        self.build_tuple(instr, frozenset)

    # Note: BUILD_SLICE is not optimized

    def eval_COMPARE_OP(self, instr):
        # Note: COMPARE_OP: 2 < 3 is not optimized

        try:
            new_arg = NOT_COMPARE[instr.arg]
        except KeyError:
            return

        if self.get_next_instr("UNARY_NOT") is None:
            return

        # not (a is b) -->  a is not b
        # not (a in b) -->  a not in b
        # not (a is not b) -->  a is b
        # not (a not in b) -->  a in b
        instr.arg = new_arg
        self.block[self.index - 1 : self.index + 1] = (instr,)

    def jump_if_or_pop(self, instr):
        # Simplify conditional jump to conditional jump where the
        # result of the first test implies the success of a similar
        # test or the failure of the opposite test.
        #
        # Arises in code like:
        # "if a and b:"
        # "if a or b:"
        # "a and b or c"
        # "(a and b) and c"
        #
        # x:JUMP_IF_FALSE_OR_POP y   y:JUMP_IF_FALSE_OR_POP z
        #    -->  x:JUMP_IF_FALSE_OR_POP z
        #
        # x:JUMP_IF_FALSE_OR_POP y   y:JUMP_IF_TRUE_OR_POP z
        #    -->  x:POP_JUMP_IF_FALSE y+3
        # where y+3 is the instruction following the second test.
        target_block = instr.arg
        try:
            target_instr = target_block[0]
        except IndexError:
            return

        if not target_instr.is_cond_jump():
            self.optimize_jump_to_cond_jump(instr)
            return

        if (target_instr.name in JUMPS_ON_TRUE) == (instr.name in JUMPS_ON_TRUE):
            # The second jump will be taken iff the first is.

            target2 = target_instr.arg
            # The current opcode inherits its target's stack behaviour
            instr.name = target_instr.name
            instr.arg = target2
            self.block[self.index - 1] = instr
            self.index -= 1
        else:
            # The second jump is not taken if the first is (so jump past it),
            # and all conditional jumps pop their argument when they're not
            # taken (so change the first jump to pop its argument when it's
            # taken).
            if instr.name in JUMPS_ON_TRUE:
                name = "POP_JUMP_IF_TRUE"
            else:
                name = "POP_JUMP_IF_FALSE"

            new_label = self.code.split_block(target_block, 1)

            instr.name = name
            instr.arg = new_label
            self.block[self.index - 1] = instr
            self.index -= 1

    def eval_JUMP_IF_FALSE_OR_POP(self, instr):
        self.jump_if_or_pop(instr)

    def eval_JUMP_IF_TRUE_OR_POP(self, instr):
        self.jump_if_or_pop(instr)

    def eval_NOP(self, instr):
        # Remove NOP
        del self.block[self.index - 1]
        self.index -= 1

    def optimize_jump_to_cond_jump(self, instr):
        # Replace jumps to unconditional jumps
        jump_label = instr.arg
        assert isinstance(jump_label, BasicBlock), jump_label

        try:
            target_instr = jump_label[0]
        except IndexError:
            return

        if instr.is_uncond_jump() and target_instr.name == "RETURN_VALUE":
            # Replace JUMP_ABSOLUTE => RETURN_VALUE with RETURN_VALUE
            self.block[self.index - 1] = target_instr

        elif target_instr.is_uncond_jump():
            # Replace JUMP_FORWARD t1 jumping to JUMP_FORWARD t2
            # with JUMP_ABSOLUTE t2
            jump_target2 = target_instr.arg

            name = instr.name
            if instr.name == "JUMP_FORWARD":
                name = "JUMP_ABSOLUTE"
            else:
                # FIXME: reimplement this check
                # if jump_target2 < 0:
                #    # No backward relative jumps
                #    return

                # FIXME: remove this workaround and implement comment code ^^
                if instr.opcode in opcode.hasjrel:
                    return

            instr.name = name
            instr.arg = jump_target2
            self.block[self.index - 1] = instr

    def optimize_jump(self, instr):
        if instr.is_uncond_jump() and self.index == len(self.block):
            # JUMP_ABSOLUTE at the end of a block which points to the
            # following block: remove the jump, link the current block
            # to the following block
            block_index = self.block_index
            target_block = instr.arg
            target_block_index = self.code.get_block_index(target_block)
            if target_block_index == block_index:
                del self.block[self.index - 1]
                self.block.next_block = target_block
                return

        self.optimize_jump_to_cond_jump(instr)

    def iterblock(self, block):
        self.block = block
        self.index = 0
        while self.index < len(block):
            instr = self.block[self.index]
            self.index += 1
            yield instr

    def optimize_block(self, block):
        self.const_stack.clear()
        self.in_consts = False

        for instr in self.iterblock(block):
            if not self.in_consts:
                self.const_stack.clear()
            self.in_consts = False

            meth_name = "eval_%s" % instr.name
            meth = getattr(self, meth_name, None)
            if meth is not None:
                meth(instr)
            elif instr.has_jump():
                self.optimize_jump(instr)

            # Note: Skipping over LOAD_CONST trueconst; POP_JUMP_IF_FALSE
            # <target> is not implemented, since it looks like the optimization
            # is never trigerred in practice. The compiler already optimizes if
            # and while statements.

    def remove_dead_blocks(self):
        # FIXME: remove empty blocks?

        used_blocks = {id(self.code[0])}
        for block in self.code:
            if block.next_block is not None:
                used_blocks.add(id(block.next_block))
            for instr in block:
                if isinstance(instr, Instr) and isinstance(instr.arg, BasicBlock):
                    used_blocks.add(id(instr.arg))

        block_index = 0
        while block_index < len(self.code):
            block = self.code[block_index]
            if id(block) not in used_blocks:
                del self.code[block_index]
            else:
                block_index += 1

        # FIXME: merge following blocks if block1 does not contain any
        # jump and block1.next_block is block2

    def optimize_cfg(self, cfg):
        self.code = cfg
        self.const_stack = []

        self.remove_dead_blocks()

        self.block_index = 0
        while self.block_index < len(self.code):
            block = self.code[self.block_index]
            self.block_index += 1
            self.optimize_block(block)

    def optimize(self, code_obj):
        bytecode = Bytecode.from_code(code_obj)
        cfg = ControlFlowGraph.from_bytecode(bytecode)

        self.optimize_cfg(cfg)

        bytecode = cfg.to_bytecode()
        code = bytecode.to_code()
        return code


# Code transformer for the PEP 511
class CodeTransformer:
    name = "pyopt"

    def code_transformer(self, code, context):
        if sys.flags.verbose:
            print(
                "Optimize %s:%s: %s"
                % (code.co_filename, code.co_firstlineno, code.co_name)
            )
        optimizer = PeepholeOptimizer()
        return optimizer.optimize(code)
