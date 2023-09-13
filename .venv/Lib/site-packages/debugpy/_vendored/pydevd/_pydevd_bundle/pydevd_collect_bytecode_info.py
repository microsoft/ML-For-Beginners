import dis
import inspect
import sys
from collections import namedtuple

from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
                    hasfree, hasjrel, haslocal, hasname, opname)

from io import StringIO


class TryExceptInfo(object):

    def __init__(self, try_line, ignore=False):
        '''
        :param try_line:
        :param ignore:
            Usually we should ignore any block that's not a try..except
            (this can happen for finally blocks, with statements, etc, for
            which we create temporary entries).
        '''
        self.try_line = try_line
        self.ignore = ignore
        self.except_line = -1
        self.except_end_line = -1
        self.raise_lines_in_except = []

        # Note: these may not be available if generated from source instead of bytecode.
        self.except_bytecode_offset = -1
        self.except_end_bytecode_offset = -1

    def is_line_in_try_block(self, line):
        return self.try_line <= line < self.except_line

    def is_line_in_except_block(self, line):
        return self.except_line <= line <= self.except_end_line

    def __str__(self):
        lst = [
            '{try:',
            str(self.try_line),
            ' except ',
            str(self.except_line),
            ' end block ',
            str(self.except_end_line),
        ]
        if self.raise_lines_in_except:
            lst.append(' raises: %s' % (', '.join(str(x) for x in self.raise_lines_in_except),))

        lst.append('}')
        return ''.join(lst)

    __repr__ = __str__


class ReturnInfo(object):

    def __init__(self, return_line):
        self.return_line = return_line

    def __str__(self):
        return '{return: %s}' % (self.return_line,)

    __repr__ = __str__


def _get_line(op_offset_to_line, op_offset, firstlineno, search=False):
    op_offset_original = op_offset
    while op_offset >= 0:
        ret = op_offset_to_line.get(op_offset)
        if ret is not None:
            return ret - firstlineno
        if not search:
            return ret
        else:
            op_offset -= 1
    raise AssertionError('Unable to find line for offset: %s.Info: %s' % (
        op_offset_original, op_offset_to_line))


def debug(s):
    pass


_Instruction = namedtuple('_Instruction', 'opname, opcode, starts_line, argval, is_jump_target, offset, argrepr')


def _iter_as_bytecode_as_instructions_py2(co):
    code = co.co_code
    op_offset_to_line = dict(dis.findlinestarts(co))
    labels = set(dis.findlabels(code))
    bytecode_len = len(code)
    i = 0
    extended_arg = 0
    free = None

    op_to_name = opname

    while i < bytecode_len:
        c = code[i]
        op = ord(c)
        is_jump_target = i in labels

        curr_op_name = op_to_name[op]
        initial_bytecode_offset = i

        i = i + 1
        if op < HAVE_ARGUMENT:
            yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), None, is_jump_target, initial_bytecode_offset, '')

        else:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg

            extended_arg = 0
            i = i + 2
            if op == EXTENDED_ARG:
                extended_arg = oparg * 65536

            if op in hasconst:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_consts[oparg], is_jump_target, initial_bytecode_offset, repr(co.co_consts[oparg]))
            elif op in hasname:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_names[oparg], is_jump_target, initial_bytecode_offset, str(co.co_names[oparg]))
            elif op in hasjrel:
                argval = i + oparg
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), argval, is_jump_target, initial_bytecode_offset, "to " + repr(argval))
            elif op in haslocal:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), co.co_varnames[oparg], is_jump_target, initial_bytecode_offset, str(co.co_varnames[oparg]))
            elif op in hascompare:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), cmp_op[oparg], is_jump_target, initial_bytecode_offset, cmp_op[oparg])
            elif op in hasfree:
                if free is None:
                    free = co.co_cellvars + co.co_freevars
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), free[oparg], is_jump_target, initial_bytecode_offset, str(free[oparg]))
            else:
                yield _Instruction(curr_op_name, op, _get_line(op_offset_to_line, initial_bytecode_offset, 0), oparg, is_jump_target, initial_bytecode_offset, str(oparg))


def iter_instructions(co):
    if sys.version_info[0] < 3:
        iter_in = _iter_as_bytecode_as_instructions_py2(co)
    else:
        iter_in = dis.Bytecode(co)
    iter_in = list(iter_in)

    bytecode_to_instruction = {}
    for instruction in iter_in:
        bytecode_to_instruction[instruction.offset] = instruction

    if iter_in:
        for instruction in iter_in:
            yield instruction


def collect_return_info(co, use_func_first_line=False):
    if not hasattr(co, 'co_lines') and not hasattr(co, 'co_lnotab'):
        return []

    if use_func_first_line:
        firstlineno = co.co_firstlineno
    else:
        firstlineno = 0

    lst = []
    op_offset_to_line = dict(dis.findlinestarts(co))
    for instruction in iter_instructions(co):
        curr_op_name = instruction.opname
        if curr_op_name == 'RETURN_VALUE':
            lst.append(ReturnInfo(_get_line(op_offset_to_line, instruction.offset, firstlineno, search=True)))

    return lst


if sys.version_info[:2] <= (3, 9):

    class _TargetInfo(object):

        def __init__(self, except_end_instruction, jump_if_not_exc_instruction=None):
            self.except_end_instruction = except_end_instruction
            self.jump_if_not_exc_instruction = jump_if_not_exc_instruction

        def __str__(self):
            msg = ['_TargetInfo(']
            msg.append(self.except_end_instruction.opname)
            if self.jump_if_not_exc_instruction:
                msg.append(' - ')
                msg.append(self.jump_if_not_exc_instruction.opname)
                msg.append('(')
                msg.append(str(self.jump_if_not_exc_instruction.argval))
                msg.append(')')
            msg.append(')')
            return ''.join(msg)

    def _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx):
        next_3 = [j_instruction.opname for j_instruction in instructions[exception_end_instruction_index:exception_end_instruction_index + 3]]
        # print('next_3:', [(j_instruction.opname, j_instruction.argval) for j_instruction in instructions[exception_end_instruction_index:exception_end_instruction_index + 3]])
        if next_3 == ['POP_TOP', 'POP_TOP', 'POP_TOP']:  # try..except without checking exception.
            try:
                jump_instruction = instructions[exception_end_instruction_index - 1]
                if jump_instruction.opname not in ('JUMP_FORWARD', 'JUMP_ABSOLUTE'):
                    return None
            except IndexError:
                pass

            if jump_instruction.opname == 'JUMP_ABSOLUTE':
                # On latest versions of Python 3 the interpreter has a go-backwards step,
                # used to show the initial line of a for/while, etc (which is this
                # JUMP_ABSOLUTE)... we're not really interested in it, but rather on where
                # it points to.
                except_end_instruction = instructions[offset_to_instruction_idx[jump_instruction.argval]]
                idx = offset_to_instruction_idx[except_end_instruction.argval]
                # Search for the POP_EXCEPT which should be at the end of the block.
                for pop_except_instruction in reversed(instructions[:idx]):
                    if pop_except_instruction.opname == 'POP_EXCEPT':
                        except_end_instruction = pop_except_instruction
                        return _TargetInfo(except_end_instruction)
                else:
                    return None  # i.e.: Continue outer loop

            else:
                # JUMP_FORWARD
                i = offset_to_instruction_idx[jump_instruction.argval]
                try:
                    # i.e.: the jump is to the instruction after the block finishes (so, we need to
                    # get the previous instruction as that should be the place where the exception
                    # block finishes).
                    except_end_instruction = instructions[i - 1]
                except:
                    pydev_log.critical('Error when computing try..except block end.')
                    return None
                return _TargetInfo(except_end_instruction)

        elif next_3 and next_3[0] == 'DUP_TOP':  # try..except AssertionError.
            iter_in = instructions[exception_end_instruction_index + 1:]
            for j, jump_if_not_exc_instruction in enumerate(iter_in):
                if jump_if_not_exc_instruction.opname == 'JUMP_IF_NOT_EXC_MATCH':
                    # Python 3.9
                    except_end_instruction = instructions[offset_to_instruction_idx[jump_if_not_exc_instruction.argval]]
                    return _TargetInfo(except_end_instruction, jump_if_not_exc_instruction)

                elif jump_if_not_exc_instruction.opname == 'COMPARE_OP' and jump_if_not_exc_instruction.argval == 'exception match':
                    # Python 3.8 and before
                    try:
                        next_instruction = iter_in[j + 1]
                    except:
                        continue
                    if next_instruction.opname == 'POP_JUMP_IF_FALSE':
                        except_end_instruction = instructions[offset_to_instruction_idx[next_instruction.argval]]
                        return _TargetInfo(except_end_instruction, next_instruction)
            else:
                return None  # i.e.: Continue outer loop

        else:
            # i.e.: we're not interested in try..finally statements, only try..except.
            return None

    def collect_try_except_info(co, use_func_first_line=False):
        # We no longer have 'END_FINALLY', so, we need to do things differently in Python 3.9
        if not hasattr(co, 'co_lines') and not hasattr(co, 'co_lnotab'):
            return []

        if use_func_first_line:
            firstlineno = co.co_firstlineno
        else:
            firstlineno = 0

        try_except_info_lst = []

        op_offset_to_line = dict(dis.findlinestarts(co))

        offset_to_instruction_idx = {}

        instructions = list(iter_instructions(co))

        for i, instruction in enumerate(instructions):
            offset_to_instruction_idx[instruction.offset] = i

        for i, instruction in enumerate(instructions):
            curr_op_name = instruction.opname
            if curr_op_name in ('SETUP_FINALLY', 'SETUP_EXCEPT'):  # SETUP_EXCEPT before Python 3.8, SETUP_FINALLY Python 3.8 onwards.
                exception_end_instruction_index = offset_to_instruction_idx[instruction.argval]

                jump_instruction = instructions[exception_end_instruction_index - 1]
                if jump_instruction.opname not in ('JUMP_FORWARD', 'JUMP_ABSOLUTE'):
                    continue

                except_end_instruction = None
                indexes_checked = set()
                indexes_checked.add(exception_end_instruction_index)
                target_info = _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx)
                while target_info is not None:
                    # Handle a try..except..except..except.
                    jump_instruction = target_info.jump_if_not_exc_instruction
                    except_end_instruction = target_info.except_end_instruction

                    if jump_instruction is not None:
                        check_index = offset_to_instruction_idx[jump_instruction.argval]
                        if check_index in indexes_checked:
                            break
                        indexes_checked.add(check_index)
                        target_info = _get_except_target_info(instructions, check_index, offset_to_instruction_idx)
                    else:
                        break

                if except_end_instruction is not None:
                    try_except_info = TryExceptInfo(
                        _get_line(op_offset_to_line, instruction.offset, firstlineno, search=True),
                        ignore=False
                    )
                    try_except_info.except_bytecode_offset = instruction.argval
                    try_except_info.except_line = _get_line(
                        op_offset_to_line,
                        try_except_info.except_bytecode_offset,
                        firstlineno,
                        search=True
                    )

                    try_except_info.except_end_bytecode_offset = except_end_instruction.offset
                    try_except_info.except_end_line = _get_line(op_offset_to_line, except_end_instruction.offset, firstlineno, search=True)
                    try_except_info_lst.append(try_except_info)

                    for raise_instruction in instructions[i:offset_to_instruction_idx[try_except_info.except_end_bytecode_offset]]:
                        if raise_instruction.opname == 'RAISE_VARARGS':
                            if raise_instruction.argval == 0:
                                try_except_info.raise_lines_in_except.append(
                                    _get_line(op_offset_to_line, raise_instruction.offset, firstlineno, search=True))

        return try_except_info_lst

elif sys.version_info[:2] == (3, 10):

    class _TargetInfo(object):

        def __init__(self, except_end_instruction, jump_if_not_exc_instruction=None):
            self.except_end_instruction = except_end_instruction
            self.jump_if_not_exc_instruction = jump_if_not_exc_instruction

        def __str__(self):
            msg = ['_TargetInfo(']
            msg.append(self.except_end_instruction.opname)
            if self.jump_if_not_exc_instruction:
                msg.append(' - ')
                msg.append(self.jump_if_not_exc_instruction.opname)
                msg.append('(')
                msg.append(str(self.jump_if_not_exc_instruction.argval))
                msg.append(')')
            msg.append(')')
            return ''.join(msg)

    def _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx):
        next_3 = [j_instruction.opname for j_instruction in instructions[exception_end_instruction_index:exception_end_instruction_index + 3]]
        # print('next_3:', [(j_instruction.opname, j_instruction.argval) for j_instruction in instructions[exception_end_instruction_index:exception_end_instruction_index + 3]])
        if next_3 == ['POP_TOP', 'POP_TOP', 'POP_TOP']:  # try..except without checking exception.
            # Previously there was a jump which was able to point where the exception would end. This
            # is no longer true, now a bare except doesn't really have any indication in the bytecode
            # where the end would be expected if the exception wasn't raised, so, we just blindly
            # search for a POP_EXCEPT from the current position.
            for pop_except_instruction in instructions[exception_end_instruction_index + 3:]:
                if pop_except_instruction.opname == 'POP_EXCEPT':
                    except_end_instruction = pop_except_instruction
                    return _TargetInfo(except_end_instruction)

        elif next_3 and next_3[0] == 'DUP_TOP':  # try..except AssertionError.
            iter_in = instructions[exception_end_instruction_index + 1:]
            for jump_if_not_exc_instruction in iter_in:
                if jump_if_not_exc_instruction.opname == 'JUMP_IF_NOT_EXC_MATCH':
                    # Python 3.9
                    except_end_instruction = instructions[offset_to_instruction_idx[jump_if_not_exc_instruction.argval]]
                    return _TargetInfo(except_end_instruction, jump_if_not_exc_instruction)
            else:
                return None  # i.e.: Continue outer loop

        else:
            # i.e.: we're not interested in try..finally statements, only try..except.
            return None

    def collect_try_except_info(co, use_func_first_line=False):
        # We no longer have 'END_FINALLY', so, we need to do things differently in Python 3.9
        if not hasattr(co, 'co_lines') and not hasattr(co, 'co_lnotab'):
            return []

        if use_func_first_line:
            firstlineno = co.co_firstlineno
        else:
            firstlineno = 0

        try_except_info_lst = []

        op_offset_to_line = dict(dis.findlinestarts(co))

        offset_to_instruction_idx = {}

        instructions = list(iter_instructions(co))

        for i, instruction in enumerate(instructions):
            offset_to_instruction_idx[instruction.offset] = i

        for i, instruction in enumerate(instructions):
            curr_op_name = instruction.opname
            if curr_op_name == 'SETUP_FINALLY':
                exception_end_instruction_index = offset_to_instruction_idx[instruction.argval]

                jump_instruction = instructions[exception_end_instruction_index]
                if jump_instruction.opname != 'DUP_TOP':
                    continue

                except_end_instruction = None
                indexes_checked = set()
                indexes_checked.add(exception_end_instruction_index)
                target_info = _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx)
                while target_info is not None:
                    # Handle a try..except..except..except.
                    jump_instruction = target_info.jump_if_not_exc_instruction
                    except_end_instruction = target_info.except_end_instruction

                    if jump_instruction is not None:
                        check_index = offset_to_instruction_idx[jump_instruction.argval]
                        if check_index in indexes_checked:
                            break
                        indexes_checked.add(check_index)
                        target_info = _get_except_target_info(instructions, check_index, offset_to_instruction_idx)
                    else:
                        break

                if except_end_instruction is not None:
                    try_except_info = TryExceptInfo(
                        _get_line(op_offset_to_line, instruction.offset, firstlineno, search=True),
                        ignore=False
                    )
                    try_except_info.except_bytecode_offset = instruction.argval
                    try_except_info.except_line = _get_line(
                        op_offset_to_line,
                        try_except_info.except_bytecode_offset,
                        firstlineno,
                        search=True
                    )

                    try_except_info.except_end_bytecode_offset = except_end_instruction.offset

                    # On Python 3.10 the final line of the except end isn't really correct, rather,
                    # it's engineered to be the same line of the except and not the end line of the
                    # block, so, the approach taken is to search for the biggest line between the
                    # except and the end instruction
                    except_end_line = -1
                    start_i = offset_to_instruction_idx[try_except_info.except_bytecode_offset]
                    end_i = offset_to_instruction_idx[except_end_instruction.offset]
                    for instruction in instructions[start_i: end_i + 1]:
                        found_at_line = op_offset_to_line.get(instruction.offset)
                        if found_at_line is not None and found_at_line > except_end_line:
                            except_end_line = found_at_line
                    try_except_info.except_end_line = except_end_line - firstlineno

                    try_except_info_lst.append(try_except_info)

                    for raise_instruction in instructions[i:offset_to_instruction_idx[try_except_info.except_end_bytecode_offset]]:
                        if raise_instruction.opname == 'RAISE_VARARGS':
                            if raise_instruction.argval == 0:
                                try_except_info.raise_lines_in_except.append(
                                    _get_line(op_offset_to_line, raise_instruction.offset, firstlineno, search=True))

        return try_except_info_lst

elif sys.version_info[:2] >= (3, 11):

    def collect_try_except_info(co, use_func_first_line=False):
        '''
        Note: if the filename is available and we can get the source,
        `collect_try_except_info_from_source` is preferred (this is kept as
        a fallback for cases where sources aren't available).
        '''
        return []

import ast as ast_module


class _Visitor(ast_module.NodeVisitor):

    def __init__(self):
        self.try_except_infos = []
        self._stack = []
        self._in_except_stack = []
        self.max_line = -1

    def generic_visit(self, node):
        if hasattr(node, 'lineno'):
            if node.lineno > self.max_line:
                self.max_line = node.lineno
        return ast_module.NodeVisitor.generic_visit(self, node)

    def visit_Try(self, node):
        info = TryExceptInfo(node.lineno, ignore=True)
        self._stack.append(info)
        self.generic_visit(node)
        assert info is self._stack.pop()
        if not info.ignore:
            self.try_except_infos.insert(0, info)

    if sys.version_info[0] < 3:
        visit_TryExcept = visit_Try

    def visit_ExceptHandler(self, node):
        info = self._stack[-1]
        info.ignore = False
        if info.except_line == -1:
            info.except_line = node.lineno
        self._in_except_stack.append(info)
        self.generic_visit(node)
        if hasattr(node, 'end_lineno'):
            info.except_end_line = node.end_lineno
        else:
            info.except_end_line = self.max_line
        self._in_except_stack.pop()

    if sys.version_info[0] >= 3:

        def visit_Raise(self, node):
            for info in self._in_except_stack:
                    if node.exc is None:
                        info.raise_lines_in_except.append(node.lineno)
            self.generic_visit(node)

    else:

        def visit_Raise(self, node):
            for info in self._in_except_stack:
                    if node.type is None and node.tback is None:
                        info.raise_lines_in_except.append(node.lineno)
            self.generic_visit(node)


def collect_try_except_info_from_source(filename):
    with open(filename, 'rb') as stream:
        contents = stream.read()
    return collect_try_except_info_from_contents(contents, filename)


def collect_try_except_info_from_contents(contents, filename='<unknown>'):
    ast = ast_module.parse(contents, filename)
    visitor = _Visitor()
    visitor.visit(ast)
    return visitor.try_except_infos


RESTART_FROM_LOOKAHEAD = object()
SEPARATOR = object()


class _MsgPart(object):

    def __init__(self, line, tok):
        assert line >= 0
        self.line = line
        self.tok = tok

    @classmethod
    def add_to_line_to_contents(cls, obj, line_to_contents, line=None):
        if isinstance(obj, (list, tuple)):
            for o in obj:
                cls.add_to_line_to_contents(o, line_to_contents, line=line)
            return

        if isinstance(obj, str):
            assert line is not None
            line = int(line)
            lst = line_to_contents.setdefault(line, [])
            lst.append(obj)
            return

        if isinstance(obj, _MsgPart):
            if isinstance(obj.tok, (list, tuple)):
                cls.add_to_line_to_contents(obj.tok, line_to_contents, line=obj.line)
                return

            if isinstance(obj.tok, str):
                lst = line_to_contents.setdefault(obj.line, [])
                lst.append(obj.tok)
                return

        raise AssertionError("Unhandled: %" % (obj,))


class _Disassembler(object):

    def __init__(self, co, firstlineno, level=0):
        self.co = co
        self.firstlineno = firstlineno
        self.level = level
        self.instructions = list(iter_instructions(co))
        op_offset_to_line = self.op_offset_to_line = dict(dis.findlinestarts(co))

        # Update offsets so that all offsets have the line index (and update it based on
        # the passed firstlineno).
        line_index = co.co_firstlineno - firstlineno
        for instruction in self.instructions:
            new_line_index = op_offset_to_line.get(instruction.offset)
            if new_line_index is not None:
                line_index = new_line_index - firstlineno
                op_offset_to_line[instruction.offset] = line_index
            else:
                op_offset_to_line[instruction.offset] = line_index

    BIG_LINE_INT = 9999999
    SMALL_LINE_INT = -1

    def min_line(self, *args):
        m = self.BIG_LINE_INT
        for arg in args:
            if isinstance(arg, (list, tuple)):
                m = min(m, self.min_line(*arg))

            elif isinstance(arg, _MsgPart):
                m = min(m, arg.line)

            elif hasattr(arg, 'offset'):
                m = min(m, self.op_offset_to_line[arg.offset])
        return m

    def max_line(self, *args):
        m = self.SMALL_LINE_INT
        for arg in args:
            if isinstance(arg, (list, tuple)):
                m = max(m, self.max_line(*arg))

            elif isinstance(arg, _MsgPart):
                m = max(m, arg.line)

            elif hasattr(arg, 'offset'):
                m = max(m, self.op_offset_to_line[arg.offset])
        return m

    def _lookahead(self):
        '''
        This handles and converts some common constructs from bytecode to actual source code.

        It may change the list of instructions.
        '''
        msg = self._create_msg_part
        found = []
        fullrepr = None

        # Collect all the load instructions
        for next_instruction in self.instructions:
            if next_instruction.opname in ('LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_CONST', 'LOAD_NAME'):
                found.append(next_instruction)
            else:
                break

        if not found:
            return None

        if next_instruction.opname == 'LOAD_ATTR':
            prev_instruction = found[-1]
            # Remove the current LOAD_ATTR
            assert self.instructions.pop(len(found)) is next_instruction

            # Add the LOAD_ATTR to the previous LOAD
            self.instructions[len(found) - 1] = _Instruction(
                prev_instruction.opname,
                prev_instruction.opcode,
                prev_instruction.starts_line,
                prev_instruction.argval,
                False,  # prev_instruction.is_jump_target,
                prev_instruction.offset,
                (
                    msg(prev_instruction),
                    msg(prev_instruction, '.'),
                    msg(next_instruction)
                ),
            )
            return RESTART_FROM_LOOKAHEAD

        if next_instruction.opname in ('CALL_FUNCTION', 'PRECALL'):
            if len(found) == next_instruction.argval + 1:
                force_restart = False
                delta = 0
            else:
                force_restart = True
                if len(found) > next_instruction.argval + 1:
                    delta = len(found) - (next_instruction.argval + 1)
                else:
                    return None  # This is odd

            del_upto = delta + next_instruction.argval + 2  # +2 = NAME / CALL_FUNCTION
            if next_instruction.opname == 'PRECALL':
                del_upto += 1  # Also remove the CALL right after the PRECALL.
            del self.instructions[delta:del_upto]

            found = iter(found[delta:])
            call_func = next(found)
            args = list(found)
            fullrepr = [
                msg(call_func),
                msg(call_func, '('),
            ]
            prev = call_func
            for i, arg in enumerate(args):
                if i > 0:
                    fullrepr.append(msg(prev, ', '))
                prev = arg
                fullrepr.append(msg(arg))

            fullrepr.append(msg(prev, ')'))

            if force_restart:
                self.instructions.insert(delta, _Instruction(
                    call_func.opname,
                    call_func.opcode,
                    call_func.starts_line,
                    call_func.argval,
                    False,  # call_func.is_jump_target,
                    call_func.offset,
                    tuple(fullrepr),
                ))
                return RESTART_FROM_LOOKAHEAD

        elif next_instruction.opname == 'BUILD_TUPLE':

            if len(found) == next_instruction.argval:
                force_restart = False
                delta = 0
            else:
                force_restart = True
                if len(found) > next_instruction.argval:
                    delta = len(found) - (next_instruction.argval)
                else:
                    return None  # This is odd

            del self.instructions[delta:delta + next_instruction.argval + 1]  # +1 = BUILD_TUPLE

            found = iter(found[delta:])

            args = [instruction for instruction in found]
            if args:
                first_instruction = args[0]
            else:
                first_instruction = next_instruction
            prev = first_instruction

            fullrepr = []
            fullrepr.append(msg(prev, '('))
            for i, arg in enumerate(args):
                if i > 0:
                    fullrepr.append(msg(prev, ', '))
                prev = arg
                fullrepr.append(msg(arg))

            fullrepr.append(msg(prev, ')'))

            if force_restart:
                self.instructions.insert(delta, _Instruction(
                    first_instruction.opname,
                    first_instruction.opcode,
                    first_instruction.starts_line,
                    first_instruction.argval,
                    False,  # first_instruction.is_jump_target,
                    first_instruction.offset,
                    tuple(fullrepr),
                ))
                return RESTART_FROM_LOOKAHEAD

        if fullrepr is not None and self.instructions:
            if self.instructions[0].opname == 'POP_TOP':
                self.instructions.pop(0)

            if self.instructions[0].opname in ('STORE_FAST', 'STORE_NAME'):
                next_instruction = self.instructions.pop(0)
                return msg(next_instruction), msg(next_instruction, ' = '), fullrepr

            if self.instructions[0].opname == 'RETURN_VALUE':
                next_instruction = self.instructions.pop(0)
                return msg(next_instruction, 'return ', line=self.min_line(next_instruction, fullrepr)), fullrepr

        return fullrepr

    def _decorate_jump_target(self, instruction, instruction_repr):
        if instruction.is_jump_target:
            return ('|', str(instruction.offset), '|', instruction_repr)

        return instruction_repr

    def _create_msg_part(self, instruction, tok=None, line=None):
        dec = self._decorate_jump_target
        if line is None or line in (self.BIG_LINE_INT, self.SMALL_LINE_INT):
            line = self.op_offset_to_line[instruction.offset]

        argrepr = instruction.argrepr
        if isinstance(argrepr, str) and argrepr.startswith('NULL + '):
            argrepr = argrepr[7:]
        return _MsgPart(
            line, tok if tok is not None else dec(instruction, argrepr))

    def _next_instruction_to_str(self, line_to_contents):
        # indent = ''
        # if self.level > 0:
        #     indent += '    ' * self.level
        # print(indent, 'handle', self.instructions[0])

        if self.instructions:
            ret = self._lookahead()
            if ret:
                return ret

        msg = self._create_msg_part

        instruction = self.instructions.pop(0)

        if instruction.opname in 'RESUME':
            return None

        if instruction.opname in ('LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_CONST', 'LOAD_NAME'):
            next_instruction = self.instructions[0]
            if next_instruction.opname in ('STORE_FAST', 'STORE_NAME'):
                self.instructions.pop(0)
                return (
                    msg(next_instruction),
                    msg(next_instruction, ' = '),
                    msg(instruction))

            if next_instruction.opname == 'RETURN_VALUE':
                self.instructions.pop(0)
                return (msg(instruction, 'return ', line=self.min_line(instruction)), msg(instruction))

            if next_instruction.opname == 'RAISE_VARARGS' and next_instruction.argval == 1:
                self.instructions.pop(0)
                return (msg(instruction, 'raise ', line=self.min_line(instruction)), msg(instruction))

        if instruction.opname == 'LOAD_CONST':
            if inspect.iscode(instruction.argval):

                code_line_to_contents = _Disassembler(
                    instruction.argval, self.firstlineno, self.level + 1
                ).build_line_to_contents()

                for contents in code_line_to_contents.values():
                    contents.insert(0, '    ')
                for line, contents in code_line_to_contents.items():
                    line_to_contents.setdefault(line, []).extend(contents)
                return msg(instruction, 'LOAD_CONST(code)')

        if instruction.opname == 'RAISE_VARARGS':
            if instruction.argval == 0:
                return msg(instruction, 'raise')

        if instruction.opname == 'SETUP_FINALLY':
            return msg(instruction, ('try(', instruction.argrepr, '):'))

        if instruction.argrepr:
            return msg(instruction, (instruction.opname, '(', instruction.argrepr, ')'))

        if instruction.argval:
            return msg(instruction, '%s{%s}' % (instruction.opname, instruction.argval,))

        return msg(instruction, instruction.opname)

    def build_line_to_contents(self):
        # print('----')
        # for instruction in self.instructions:
        #     print(instruction)
        # print('----\n\n')

        line_to_contents = {}

        instructions = self.instructions
        while instructions:
            s = self._next_instruction_to_str(line_to_contents)
            if s is RESTART_FROM_LOOKAHEAD:
                continue
            if s is None:
                continue

            _MsgPart.add_to_line_to_contents(s, line_to_contents)
            m = self.max_line(s)
            if m != self.SMALL_LINE_INT:
                line_to_contents.setdefault(m, []).append(SEPARATOR)
        return line_to_contents

    def disassemble(self):
        line_to_contents = self.build_line_to_contents()
        stream = StringIO()
        last_line = 0
        show_lines = False
        for line, contents in sorted(line_to_contents.items()):
            while last_line < line - 1:
                if show_lines:
                    stream.write('%s.\n' % (last_line + 1,))
                else:
                    stream.write('\n')
                last_line += 1

            if show_lines:
                stream.write('%s. ' % (line,))

            for i, content in enumerate(contents):
                if content == SEPARATOR:
                    if i != len(contents) - 1:
                        stream.write(', ')
                else:
                    stream.write(content)

            stream.write('\n')

            last_line = line

        return stream.getvalue()


def code_to_bytecode_representation(co, use_func_first_line=False):
    '''
    A simple disassemble of bytecode.

    It does not attempt to provide the full Python source code, rather, it provides a low-level
    representation of the bytecode, respecting the lines (so, its target is making the bytecode
    easier to grasp and not providing the original source code).

    Note that it does show jump locations/targets and converts some common bytecode constructs to
    Python code to make it a bit easier to understand.
    '''
    # Reference for bytecodes:
    # https://docs.python.org/3/library/dis.html
    if use_func_first_line:
        firstlineno = co.co_firstlineno
    else:
        firstlineno = 0

    return _Disassembler(co, firstlineno).disassemble()
