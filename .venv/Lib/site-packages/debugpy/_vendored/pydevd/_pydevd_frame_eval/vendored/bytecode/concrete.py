import dis
import inspect
import opcode as _opcode
import struct
import sys
import types

# alias to keep the 'bytecode' variable free
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
    UNSET,
    Instr,
    Label,
    SetLineno,
    FreeVar,
    CellVar,
    Compare,
    const_key,
    _check_arg_int,
)


# - jumps use instruction
# - lineno use bytes (dis.findlinestarts(code))
# - dis displays bytes
OFFSET_AS_INSTRUCTION = sys.version_info >= (3, 10)


def _set_docstring(code, consts):
    if not consts:
        return
    first_const = consts[0]
    if isinstance(first_const, str) or first_const is None:
        code.docstring = first_const


class ConcreteInstr(Instr):
    """Concrete instruction.

    arg must be an integer in the range 0..2147483647.

    It has a read-only size attribute.
    """

    __slots__ = ("_size", "_extended_args", "offset")

    def __init__(self, name, arg=UNSET, *, lineno=None, extended_args=None, offset=None):
        # Allow to remember a potentially meaningless EXTENDED_ARG emitted by
        # Python to properly compute the size and avoid messing up the jump
        # targets
        self._extended_args = extended_args
        self._set(name, arg, lineno)
        self.offset = offset

    def _check_arg(self, name, opcode, arg):
        if opcode >= _opcode.HAVE_ARGUMENT:
            if arg is UNSET:
                raise ValueError("operation %s requires an argument" % name)

            _check_arg_int(name, arg)
        else:
            if arg is not UNSET:
                raise ValueError("operation %s has no argument" % name)

    def _set(self, name, arg, lineno):
        super()._set(name, arg, lineno)
        size = 2
        if arg is not UNSET:
            while arg > 0xFF:
                size += 2
                arg >>= 8
        if self._extended_args is not None:
            size = 2 + 2 * self._extended_args
        self._size = size

    @property
    def size(self):
        return self._size

    def _cmp_key(self, labels=None):
        return (self._lineno, self._name, self._arg)

    def get_jump_target(self, instr_offset):
        if self._opcode in _opcode.hasjrel:
            s = (self._size // 2) if OFFSET_AS_INSTRUCTION else self._size
            return instr_offset + s + self._arg
        if self._opcode in _opcode.hasjabs:
            return self._arg
        return None

    def assemble(self):
        if self._arg is UNSET:
            return bytes((self._opcode, 0))

        arg = self._arg
        b = [self._opcode, arg & 0xFF]
        while arg > 0xFF:
            arg >>= 8
            b[:0] = [_opcode.EXTENDED_ARG, arg & 0xFF]

        if self._extended_args:
            while len(b) < self._size:
                b[:0] = [_opcode.EXTENDED_ARG, 0x00]

        return bytes(b)

    @classmethod
    def disassemble(cls, lineno, code, offset):
        index = 2 * offset if OFFSET_AS_INSTRUCTION else offset
        op = code[index]
        if op >= _opcode.HAVE_ARGUMENT:
            arg = code[index + 1]
        else:
            arg = UNSET
        name = _opcode.opname[op]
        # fabioz: added offset to ConcreteBytecode
        # Need to keep an eye on https://github.com/MatthieuDartiailh/bytecode/issues/48 in
        # case the library decides to add this in some other way.
        return cls(name, arg, lineno=lineno, offset=index)


class ConcreteBytecode(_bytecode._BaseBytecodeList):
    def __init__(self, instructions=(), *, consts=(), names=(), varnames=()):
        super().__init__()
        self.consts = list(consts)
        self.names = list(names)
        self.varnames = list(varnames)
        for instr in instructions:
            self._check_instr(instr)
        self.extend(instructions)

    def __iter__(self):
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr):
        if not isinstance(instr, (ConcreteInstr, SetLineno)):
            raise ValueError(
                "ConcreteBytecode must only contain "
                "ConcreteInstr and SetLineno objects, "
                "but %s was found" % type(instr).__name__
            )

    def _copy_attr_from(self, bytecode):
        super()._copy_attr_from(bytecode)
        if isinstance(bytecode, ConcreteBytecode):
            self.consts = bytecode.consts
            self.names = bytecode.names
            self.varnames = bytecode.varnames

    def __repr__(self):
        return "<ConcreteBytecode instr#=%s>" % len(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        const_keys1 = list(map(const_key, self.consts))
        const_keys2 = list(map(const_key, other.consts))
        if const_keys1 != const_keys2:
            return False

        if self.names != other.names:
            return False
        if self.varnames != other.varnames:
            return False

        return super().__eq__(other)

    @staticmethod
    def from_code(code, *, extended_arg=False):
        line_starts = dict(dis.findlinestarts(code))

        # find block starts
        instructions = []
        offset = 0
        lineno = code.co_firstlineno
        while offset < (len(code.co_code) // (2 if OFFSET_AS_INSTRUCTION else 1)):
            lineno_off = (2 * offset) if OFFSET_AS_INSTRUCTION else offset
            if lineno_off in line_starts:
                lineno = line_starts[lineno_off]

            instr = ConcreteInstr.disassemble(lineno, code.co_code, offset)

            instructions.append(instr)
            offset += (instr.size // 2) if OFFSET_AS_INSTRUCTION else instr.size

        bytecode = ConcreteBytecode()

        # replace jump targets with blocks
        # HINT : in some cases Python generate useless EXTENDED_ARG opcode
        # with a value of zero. Such opcodes do not increases the size of the
        # following opcode the way a normal EXTENDED_ARG does. As a
        # consequence, they need to be tracked manually as otherwise the
        # offsets in jump targets can end up being wrong.
        if not extended_arg:
            # The list is modified in place
            bytecode._remove_extended_args(instructions)

        bytecode.name = code.co_name
        bytecode.filename = code.co_filename
        bytecode.flags = code.co_flags
        bytecode.argcount = code.co_argcount
        if sys.version_info >= (3, 8):
            bytecode.posonlyargcount = code.co_posonlyargcount
        bytecode.kwonlyargcount = code.co_kwonlyargcount
        bytecode.first_lineno = code.co_firstlineno
        bytecode.names = list(code.co_names)
        bytecode.consts = list(code.co_consts)
        bytecode.varnames = list(code.co_varnames)
        bytecode.freevars = list(code.co_freevars)
        bytecode.cellvars = list(code.co_cellvars)
        _set_docstring(bytecode, code.co_consts)

        bytecode[:] = instructions
        return bytecode

    @staticmethod
    def _normalize_lineno(instructions, first_lineno):
        lineno = first_lineno
        for instr in instructions:
            # if instr.lineno is not set, it's inherited from the previous
            # instruction, or from self.first_lineno
            if instr.lineno is not None:
                lineno = instr.lineno

            if isinstance(instr, ConcreteInstr):
                yield (lineno, instr)

    def _assemble_code(self):
        offset = 0
        code_str = []
        linenos = []
        for lineno, instr in self._normalize_lineno(self, self.first_lineno):
            code_str.append(instr.assemble())
            i_size = instr.size
            linenos.append(
                ((offset * 2) if OFFSET_AS_INSTRUCTION else offset, i_size, lineno)
            )
            offset += (i_size // 2) if OFFSET_AS_INSTRUCTION else i_size
        code_str = b"".join(code_str)
        return (code_str, linenos)

    @staticmethod
    def _assemble_lnotab(first_lineno, linenos):
        lnotab = []
        old_offset = 0
        old_lineno = first_lineno
        for offset, _, lineno in linenos:
            dlineno = lineno - old_lineno
            if dlineno == 0:
                continue
            # FIXME: be kind, force monotonic line numbers? add an option?
            if dlineno < 0 and sys.version_info < (3, 6):
                raise ValueError(
                    "negative line number delta is not supported " "on Python < 3.6"
                )
            old_lineno = lineno

            doff = offset - old_offset
            old_offset = offset

            while doff > 255:
                lnotab.append(b"\xff\x00")
                doff -= 255

            while dlineno < -128:
                lnotab.append(struct.pack("Bb", doff, -128))
                doff = 0
                dlineno -= -128

            while dlineno > 127:
                lnotab.append(struct.pack("Bb", doff, 127))
                doff = 0
                dlineno -= 127

            assert 0 <= doff <= 255
            assert -128 <= dlineno <= 127

            lnotab.append(struct.pack("Bb", doff, dlineno))

        return b"".join(lnotab)

    @staticmethod
    def _pack_linetable(doff, dlineno, linetable):

        while dlineno < -127:
            linetable.append(struct.pack("Bb", 0, -127))
            dlineno -= -127

        while dlineno > 127:
            linetable.append(struct.pack("Bb", 0, 127))
            dlineno -= 127

        if doff > 254:
            linetable.append(struct.pack("Bb", 254, dlineno))
            doff -= 254

            while doff > 254:
                linetable.append(b"\xfe\x00")
                doff -= 254
            linetable.append(struct.pack("Bb", doff, 0))

        else:
            linetable.append(struct.pack("Bb", doff, dlineno))

        assert 0 <= doff <= 254
        assert -127 <= dlineno <= 127


    def _assemble_linestable(self, first_lineno, linenos):
        if not linenos:
            return b""

        linetable = []
        old_offset = 0
        
        iter_in = iter(linenos)
        
        offset, i_size, old_lineno = next(iter_in)
        old_dlineno = old_lineno - first_lineno
        for offset, i_size, lineno in iter_in:
            dlineno = lineno - old_lineno
            if dlineno == 0:
                continue
            old_lineno = lineno

            doff = offset - old_offset
            old_offset = offset

            self._pack_linetable(doff, old_dlineno, linetable)
            old_dlineno = dlineno

        # Pack the line of the last instruction.
        doff = offset + i_size - old_offset
        self._pack_linetable(doff, old_dlineno, linetable)

        return b"".join(linetable)

    @staticmethod
    def _remove_extended_args(instructions):
        # replace jump targets with blocks
        # HINT : in some cases Python generate useless EXTENDED_ARG opcode
        # with a value of zero. Such opcodes do not increases the size of the
        # following opcode the way a normal EXTENDED_ARG does. As a
        # consequence, they need to be tracked manually as otherwise the
        # offsets in jump targets can end up being wrong.
        nb_extended_args = 0
        extended_arg = None
        index = 0
        while index < len(instructions):
            instr = instructions[index]

            # Skip SetLineno meta instruction
            if isinstance(instr, SetLineno):
                index += 1
                continue

            if instr.name == "EXTENDED_ARG":
                nb_extended_args += 1
                if extended_arg is not None:
                    extended_arg = (extended_arg << 8) + instr.arg
                else:
                    extended_arg = instr.arg

                del instructions[index]
                continue

            if extended_arg is not None:
                arg = (extended_arg << 8) + instr.arg
                extended_arg = None

                instr = ConcreteInstr(
                    instr.name,
                    arg,
                    lineno=instr.lineno,
                    extended_args=nb_extended_args,
                    offset=instr.offset,
                )
                instructions[index] = instr
                nb_extended_args = 0

            index += 1

        if extended_arg is not None:
            raise ValueError("EXTENDED_ARG at the end of the code")

    def compute_stacksize(self, *, check_pre_and_post=True):
        bytecode = self.to_bytecode()
        cfg = _bytecode.ControlFlowGraph.from_bytecode(bytecode)
        return cfg.compute_stacksize(check_pre_and_post=check_pre_and_post)

    def to_code(self, stacksize=None, *, check_pre_and_post=True):
        code_str, linenos = self._assemble_code()
        lnotab = (
            self._assemble_linestable(self.first_lineno, linenos)
            if sys.version_info >= (3, 10)
            else self._assemble_lnotab(self.first_lineno, linenos)
        )
        nlocals = len(self.varnames)
        if stacksize is None:
            stacksize = self.compute_stacksize(check_pre_and_post=check_pre_and_post)

        if sys.version_info < (3, 8):
            return types.CodeType(
                self.argcount,
                self.kwonlyargcount,
                nlocals,
                stacksize,
                int(self.flags),
                code_str,
                tuple(self.consts),
                tuple(self.names),
                tuple(self.varnames),
                self.filename,
                self.name,
                self.first_lineno,
                lnotab,
                tuple(self.freevars),
                tuple(self.cellvars),
            )
        else:
            return types.CodeType(
                self.argcount,
                self.posonlyargcount,
                self.kwonlyargcount,
                nlocals,
                stacksize,
                int(self.flags),
                code_str,
                tuple(self.consts),
                tuple(self.names),
                tuple(self.varnames),
                self.filename,
                self.name,
                self.first_lineno,
                lnotab,
                tuple(self.freevars),
                tuple(self.cellvars),
            )

    def to_bytecode(self):

        # Copy instruction and remove extended args if any (in-place)
        c_instructions = self[:]
        self._remove_extended_args(c_instructions)

        # find jump targets
        jump_targets = set()
        offset = 0
        for instr in c_instructions:
            if isinstance(instr, SetLineno):
                continue
            target = instr.get_jump_target(offset)
            if target is not None:
                jump_targets.add(target)
            offset += (instr.size // 2) if OFFSET_AS_INSTRUCTION else instr.size

        # create labels
        jumps = []
        instructions = []
        labels = {}
        offset = 0
        ncells = len(self.cellvars)

        for lineno, instr in self._normalize_lineno(c_instructions, self.first_lineno):
            if offset in jump_targets:
                label = Label()
                labels[offset] = label
                instructions.append(label)

            jump_target = instr.get_jump_target(offset)
            size = instr.size

            arg = instr.arg
            # FIXME: better error reporting
            if instr.opcode in _opcode.hasconst:
                arg = self.consts[arg]
            elif instr.opcode in _opcode.haslocal:
                arg = self.varnames[arg]
            elif instr.opcode in _opcode.hasname:
                arg = self.names[arg]
            elif instr.opcode in _opcode.hasfree:
                if arg < ncells:
                    name = self.cellvars[arg]
                    arg = CellVar(name)
                else:
                    name = self.freevars[arg - ncells]
                    arg = FreeVar(name)
            elif instr.opcode in _opcode.hascompare:
                arg = Compare(arg)

            if jump_target is None:
                instr = Instr(instr.name, arg, lineno=lineno, offset=instr.offset)
            else:
                instr_index = len(instructions)
            instructions.append(instr)
            offset += (size // 2) if OFFSET_AS_INSTRUCTION else size

            if jump_target is not None:
                jumps.append((instr_index, jump_target))

        # replace jump targets with labels
        for index, jump_target in jumps:
            instr = instructions[index]
            # FIXME: better error reporting on missing label
            label = labels[jump_target]
            instructions[index] = Instr(instr.name, label, lineno=instr.lineno, offset=instr.offset)

        bytecode = _bytecode.Bytecode()
        bytecode._copy_attr_from(self)

        nargs = bytecode.argcount + bytecode.kwonlyargcount
        if sys.version_info > (3, 8):
            nargs += bytecode.posonlyargcount
        if bytecode.flags & inspect.CO_VARARGS:
            nargs += 1
        if bytecode.flags & inspect.CO_VARKEYWORDS:
            nargs += 1
        bytecode.argnames = self.varnames[:nargs]
        _set_docstring(bytecode, self.consts)

        bytecode.extend(instructions)
        return bytecode


class _ConvertBytecodeToConcrete:

    # Default number of passes of compute_jumps() before giving up.  Refer to
    # assemble_jump_offsets() in compile.c for background.
    _compute_jumps_passes = 10

    def __init__(self, code):
        assert isinstance(code, _bytecode.Bytecode)
        self.bytecode = code

        # temporary variables
        self.instructions = []
        self.jumps = []
        self.labels = {}

        # used to build ConcreteBytecode() object
        self.consts_indices = {}
        self.consts_list = []
        self.names = []
        self.varnames = []

    def add_const(self, value):
        key = const_key(value)
        if key in self.consts_indices:
            return self.consts_indices[key]
        index = len(self.consts_indices)
        self.consts_indices[key] = index
        self.consts_list.append(value)
        return index

    @staticmethod
    def add(names, name):
        try:
            index = names.index(name)
        except ValueError:
            index = len(names)
            names.append(name)
        return index

    def concrete_instructions(self):
        ncells = len(self.bytecode.cellvars)
        lineno = self.bytecode.first_lineno

        for instr in self.bytecode:
            if isinstance(instr, Label):
                self.labels[instr] = len(self.instructions)
                continue

            if isinstance(instr, SetLineno):
                lineno = instr.lineno
                continue

            if isinstance(instr, ConcreteInstr):
                instr = instr.copy()
            else:
                assert isinstance(instr, Instr)

                if instr.lineno is not None:
                    lineno = instr.lineno

                arg = instr.arg
                is_jump = isinstance(arg, Label)
                if is_jump:
                    label = arg
                    # fake value, real value is set in compute_jumps()
                    arg = 0
                elif instr.opcode in _opcode.hasconst:
                    arg = self.add_const(arg)
                elif instr.opcode in _opcode.haslocal:
                    arg = self.add(self.varnames, arg)
                elif instr.opcode in _opcode.hasname:
                    arg = self.add(self.names, arg)
                elif instr.opcode in _opcode.hasfree:
                    if isinstance(arg, CellVar):
                        arg = self.bytecode.cellvars.index(arg.name)
                    else:
                        assert isinstance(arg, FreeVar)
                        arg = ncells + self.bytecode.freevars.index(arg.name)
                elif instr.opcode in _opcode.hascompare:
                    if isinstance(arg, Compare):
                        arg = arg.value

                instr = ConcreteInstr(instr.name, arg, lineno=lineno)
                if is_jump:
                    self.jumps.append((len(self.instructions), label, instr))

            self.instructions.append(instr)

    def compute_jumps(self):
        offsets = []
        offset = 0
        for index, instr in enumerate(self.instructions):
            offsets.append(offset)
            offset += instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
        # needed if a label is at the end
        offsets.append(offset)

        # fix argument of jump instructions: resolve labels
        modified = False
        for index, label, instr in self.jumps:
            target_index = self.labels[label]
            target_offset = offsets[target_index]

            if instr.opcode in _opcode.hasjrel:
                instr_offset = offsets[index]
                target_offset -= instr_offset + (
                    instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
                )

            old_size = instr.size
            # FIXME: better error report if target_offset is negative
            instr.arg = target_offset
            if instr.size != old_size:
                modified = True

        return modified

    def to_concrete_bytecode(self, compute_jumps_passes=None):
        if compute_jumps_passes is None:
            compute_jumps_passes = self._compute_jumps_passes

        first_const = self.bytecode.docstring
        if first_const is not UNSET:
            self.add_const(first_const)

        self.varnames.extend(self.bytecode.argnames)

        self.concrete_instructions()
        for pas in range(0, compute_jumps_passes):
            modified = self.compute_jumps()
            if not modified:
                break
        else:
            raise RuntimeError(
                "compute_jumps() failed to converge after" " %d passes" % (pas + 1)
            )

        concrete = ConcreteBytecode(
            self.instructions,
            consts=self.consts_list.copy(),
            names=self.names,
            varnames=self.varnames,
        )
        concrete._copy_attr_from(self.bytecode)
        return concrete
