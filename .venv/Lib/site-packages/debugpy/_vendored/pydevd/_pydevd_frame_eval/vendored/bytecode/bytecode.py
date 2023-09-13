# alias to keep the 'bytecode' variable free
import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags


class BaseBytecode:
    def __init__(self):
        self.argcount = 0
        if sys.version_info > (3, 8):
            self.posonlyargcount = 0
        self.kwonlyargcount = 0
        self.first_lineno = 1
        self.name = "<module>"
        self.filename = "<string>"
        self.docstring = UNSET
        self.cellvars = []
        # we cannot recreate freevars from instructions because of super()
        # special-case
        self.freevars = []
        self._flags = _bytecode.CompilerFlags(0)

    def _copy_attr_from(self, bytecode):
        self.argcount = bytecode.argcount
        if sys.version_info > (3, 8):
            self.posonlyargcount = bytecode.posonlyargcount
        self.kwonlyargcount = bytecode.kwonlyargcount
        self.flags = bytecode.flags
        self.first_lineno = bytecode.first_lineno
        self.name = bytecode.name
        self.filename = bytecode.filename
        self.docstring = bytecode.docstring
        self.cellvars = list(bytecode.cellvars)
        self.freevars = list(bytecode.freevars)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.argcount != other.argcount:
            return False
        if sys.version_info > (3, 8):
            if self.posonlyargcount != other.posonlyargcount:
                return False
        if self.kwonlyargcount != other.kwonlyargcount:
            return False
        if self.flags != other.flags:
            return False
        if self.first_lineno != other.first_lineno:
            return False
        if self.filename != other.filename:
            return False
        if self.name != other.name:
            return False
        if self.docstring != other.docstring:
            return False
        if self.cellvars != other.cellvars:
            return False
        if self.freevars != other.freevars:
            return False
        if self.compute_stacksize() != other.compute_stacksize():
            return False

        return True

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, value):
        if not isinstance(value, _bytecode.CompilerFlags):
            value = _bytecode.CompilerFlags(value)
        self._flags = value

    def update_flags(self, *, is_async=None):
        self.flags = infer_flags(self, is_async)


class _BaseBytecodeList(BaseBytecode, list):
    """List subclass providing type stable slicing and copying."""

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            value = type(self)(value)
            value._copy_attr_from(self)

        return value

    def copy(self):
        new = type(self)(super().copy())
        new._copy_attr_from(self)
        return new

    def legalize(self):
        """Check that all the element of the list are valid and remove SetLineno."""
        lineno_pos = []
        set_lineno = None
        current_lineno = self.first_lineno

        for pos, instr in enumerate(self):
            if isinstance(instr, SetLineno):
                set_lineno = instr.lineno
                lineno_pos.append(pos)
                continue
            # Filter out Labels
            if not isinstance(instr, Instr):
                continue
            if set_lineno is not None:
                instr.lineno = set_lineno
            elif instr.lineno is None:
                instr.lineno = current_lineno
            else:
                current_lineno = instr.lineno

        for i in reversed(lineno_pos):
            del self[i]

    def __iter__(self):
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr):
        raise NotImplementedError()


class _InstrList(list):
    def _flat(self):
        instructions = []
        labels = {}
        jumps = []

        offset = 0
        for index, instr in enumerate(self):
            if isinstance(instr, Label):
                instructions.append("label_instr%s" % index)
                labels[instr] = offset
            else:
                if isinstance(instr, Instr) and isinstance(instr.arg, Label):
                    target_label = instr.arg
                    instr = _bytecode.ConcreteInstr(instr.name, 0, lineno=instr.lineno)
                    jumps.append((target_label, instr))
                instructions.append(instr)
                offset += 1

        for target_label, instr in jumps:
            instr.arg = labels[target_label]

        return instructions

    def __eq__(self, other):
        if not isinstance(other, _InstrList):
            other = _InstrList(other)

        return self._flat() == other._flat()


class Bytecode(_InstrList, _BaseBytecodeList):
    def __init__(self, instructions=()):
        BaseBytecode.__init__(self)
        self.argnames = []
        for instr in instructions:
            self._check_instr(instr)
        self.extend(instructions)

    def __iter__(self):
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr):
        if not isinstance(instr, (Label, SetLineno, Instr)):
            raise ValueError(
                "Bytecode must only contain Label, "
                "SetLineno, and Instr objects, "
                "but %s was found" % type(instr).__name__
            )

    def _copy_attr_from(self, bytecode):
        super()._copy_attr_from(bytecode)
        if isinstance(bytecode, Bytecode):
            self.argnames = bytecode.argnames

    @staticmethod
    def from_code(code):
        concrete = _bytecode.ConcreteBytecode.from_code(code)
        return concrete.to_bytecode()

    def compute_stacksize(self, *, check_pre_and_post=True):
        cfg = _bytecode.ControlFlowGraph.from_bytecode(self)
        return cfg.compute_stacksize(check_pre_and_post=check_pre_and_post)

    def to_code(
        self, compute_jumps_passes=None, stacksize=None, *, check_pre_and_post=True
    ):
        # Prevent reconverting the concrete bytecode to bytecode and cfg to do the
        # calculation if we need to do it.
        if stacksize is None:
            stacksize = self.compute_stacksize(check_pre_and_post=check_pre_and_post)
        bc = self.to_concrete_bytecode(compute_jumps_passes=compute_jumps_passes)
        return bc.to_code(stacksize=stacksize)

    def to_concrete_bytecode(self, compute_jumps_passes=None):
        converter = _bytecode._ConvertBytecodeToConcrete(self)
        return converter.to_concrete_bytecode(compute_jumps_passes=compute_jumps_passes)
