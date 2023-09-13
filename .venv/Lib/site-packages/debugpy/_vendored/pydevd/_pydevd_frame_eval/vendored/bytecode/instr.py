import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps

from _pydevd_frame_eval.vendored import bytecode as _bytecode


@enum.unique
class Compare(enum.IntEnum):
    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5
    IN = 6
    NOT_IN = 7
    IS = 8
    IS_NOT = 9
    EXC_MATCH = 10


UNSET = object()


def const_key(obj):
    try:
        return _dumps(obj)
    except ValueError:
        # For other types, we use the object identifier as an unique identifier
        # to ensure that they are seen as unequal.
        return (type(obj), id(obj))


def _pushes_back(opname):
    if opname in ["CALL_FINALLY"]:
        # CALL_FINALLY pushes the address of the "finally" block instead of a
        # value, hence we don't treat it as pushing back op
        return False
    return (
        opname.startswith("UNARY_")
        or opname.startswith("GET_")
        # BUILD_XXX_UNPACK have been removed in 3.9
        or opname.startswith("BINARY_")
        or opname.startswith("INPLACE_")
        or opname.startswith("BUILD_")
        or opname.startswith("CALL_")
    ) or opname in (
        "LIST_TO_TUPLE",
        "LIST_EXTEND",
        "SET_UPDATE",
        "DICT_UPDATE",
        "DICT_MERGE",
        "IS_OP",
        "CONTAINS_OP",
        "FORMAT_VALUE",
        "MAKE_FUNCTION",
        "IMPORT_NAME",
        # technically, these three do not push back, but leave the container
        # object on TOS
        "SET_ADD",
        "LIST_APPEND",
        "MAP_ADD",
        "LOAD_ATTR",
    )


def _check_lineno(lineno):
    if not isinstance(lineno, int):
        raise TypeError("lineno must be an int")
    if lineno < 1:
        raise ValueError("invalid lineno")


class SetLineno:
    __slots__ = ("_lineno",)

    def __init__(self, lineno):
        _check_lineno(lineno)
        self._lineno = lineno

    @property
    def lineno(self):
        return self._lineno

    def __eq__(self, other):
        if not isinstance(other, SetLineno):
            return False
        return self._lineno == other._lineno


class Label:
    __slots__ = ()


class _Variable:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self.name)


class CellVar(_Variable):
    __slots__ = ()


class FreeVar(_Variable):
    __slots__ = ()


def _check_arg_int(name, arg):
    if not isinstance(arg, int):
        raise TypeError(
            "operation %s argument must be an int, "
            "got %s" % (name, type(arg).__name__)
        )

    if not (0 <= arg <= 2147483647):
        raise ValueError(
            "operation %s argument must be in " "the range 0..2,147,483,647" % name
        )


if sys.version_info < (3, 8):
    _stack_effects = {
        # NOTE: the entries are all 2-tuples.  Entry[0/False] is non-taken jumps.
        # Entry[1/True] is for taken jumps.
        # opcodes not in dis.stack_effect
        _opcode.opmap["EXTENDED_ARG"]: (0, 0),
        _opcode.opmap["NOP"]: (0, 0),
        # Jump taken/not-taken are different:
        _opcode.opmap["JUMP_IF_TRUE_OR_POP"]: (-1, 0),
        _opcode.opmap["JUMP_IF_FALSE_OR_POP"]: (-1, 0),
        _opcode.opmap["FOR_ITER"]: (1, -1),
        _opcode.opmap["SETUP_WITH"]: (1, 6),
        _opcode.opmap["SETUP_ASYNC_WITH"]: (0, 5),
        _opcode.opmap["SETUP_EXCEPT"]: (0, 6),  # as of 3.7, below for <=3.6
        _opcode.opmap["SETUP_FINALLY"]: (0, 6),  # as of 3.7, below for <=3.6
    }

    # More stack effect values that are unique to the version of Python.
    if sys.version_info < (3, 7):
        _stack_effects.update(
            {
                _opcode.opmap["SETUP_WITH"]: (7, 7),
                _opcode.opmap["SETUP_EXCEPT"]: (6, 9),
                _opcode.opmap["SETUP_FINALLY"]: (6, 9),
            }
        )


class Instr:
    """Abstract instruction."""

    __slots__ = ("_name", "_opcode", "_arg", "_lineno", "offset")

    def __init__(self, name, arg=UNSET, *, lineno=None, offset=None):
        self._set(name, arg, lineno)
        self.offset = offset

    def _check_arg(self, name, opcode, arg):
        if name == "EXTENDED_ARG":
            raise ValueError(
                "only concrete instruction can contain EXTENDED_ARG, "
                "highlevel instruction can represent arbitrary argument without it"
            )

        if opcode >= _opcode.HAVE_ARGUMENT:
            if arg is UNSET:
                raise ValueError("operation %s requires an argument" % name)
        else:
            if arg is not UNSET:
                raise ValueError("operation %s has no argument" % name)

        if self._has_jump(opcode):
            if not isinstance(arg, (Label, _bytecode.BasicBlock)):
                raise TypeError(
                    "operation %s argument type must be "
                    "Label or BasicBlock, got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.hasfree:
            if not isinstance(arg, (CellVar, FreeVar)):
                raise TypeError(
                    "operation %s argument must be CellVar "
                    "or FreeVar, got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.haslocal or opcode in _opcode.hasname:
            if not isinstance(arg, str):
                raise TypeError(
                    "operation %s argument must be a str, "
                    "got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.hasconst:
            if isinstance(arg, Label):
                raise ValueError(
                    "label argument cannot be used " "in %s operation" % name
                )
            if isinstance(arg, _bytecode.BasicBlock):
                raise ValueError(
                    "block argument cannot be used " "in %s operation" % name
                )

        elif opcode in _opcode.hascompare:
            if not isinstance(arg, Compare):
                raise TypeError(
                    "operation %s argument type must be "
                    "Compare, got %s" % (name, type(arg).__name__)
                )

        elif opcode >= _opcode.HAVE_ARGUMENT:
            _check_arg_int(name, arg)

    def _set(self, name, arg, lineno):
        if not isinstance(name, str):
            raise TypeError("operation name must be a str")
        try:
            opcode = _opcode.opmap[name]
        except KeyError:
            raise ValueError("invalid operation name")

        # check lineno
        if lineno is not None:
            _check_lineno(lineno)

        self._check_arg(name, opcode, arg)

        self._name = name
        self._opcode = opcode
        self._arg = arg
        self._lineno = lineno

    def set(self, name, arg=UNSET):
        """Modify the instruction in-place.

        Replace name and arg attributes. Don't modify lineno.
        """
        self._set(name, arg, self._lineno)

    def require_arg(self):
        """Does the instruction require an argument?"""
        return self._opcode >= _opcode.HAVE_ARGUMENT

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._set(name, self._arg, self._lineno)

    @property
    def opcode(self):
        return self._opcode

    @opcode.setter
    def opcode(self, op):
        if not isinstance(op, int):
            raise TypeError("operator code must be an int")
        if 0 <= op <= 255:
            name = _opcode.opname[op]
            valid = name != "<%r>" % op
        else:
            valid = False
        if not valid:
            raise ValueError("invalid operator code")

        self._set(name, self._arg, self._lineno)

    @property
    def arg(self):
        return self._arg

    @arg.setter
    def arg(self, arg):
        self._set(self._name, arg, self._lineno)

    @property
    def lineno(self):
        return self._lineno

    @lineno.setter
    def lineno(self, lineno):
        self._set(self._name, self._arg, lineno)

    def stack_effect(self, jump=None):
        if self._opcode < _opcode.HAVE_ARGUMENT:
            arg = None
        elif not isinstance(self._arg, int) or self._opcode in _opcode.hasconst:
            # Argument is either a non-integer or an integer constant,
            # not oparg.
            arg = 0
        else:
            arg = self._arg

        if sys.version_info < (3, 8):
            effect = _stack_effects.get(self._opcode, None)
            if effect is not None:
                return max(effect) if jump is None else effect[jump]
            return dis.stack_effect(self._opcode, arg)
        else:
            return dis.stack_effect(self._opcode, arg, jump=jump)

    def pre_and_post_stack_effect(self, jump=None):
        _effect = self.stack_effect(jump=jump)

        # To compute pre size and post size to avoid segfault cause by not enough
        # stack element
        _opname = _opcode.opname[self._opcode]
        if _opname.startswith("DUP_TOP"):
            return _effect * -1, _effect * 2
        if _pushes_back(_opname):
            # if the op pushes value back to the stack, then the stack effect given
            # by dis.stack_effect actually equals pre + post effect, therefore we need
            # -1 from the stack effect as a pre condition
            return _effect - 1, 1
        if _opname.startswith("UNPACK_"):
            # Instr(UNPACK_* , n) pops 1 and pushes n
            # _effect = n - 1
            # hence we return -1, _effect + 1
            return -1, _effect + 1
        if _opname == "FOR_ITER" and not jump:
            # Since FOR_ITER needs TOS to be an iterator, which basically means
            # a prerequisite of 1 on the stack
            return -1, 2
        if _opname == "ROT_N":
            return (-self._arg, self._arg)
        return {"ROT_TWO": (-2, 2), "ROT_THREE": (-3, 3), "ROT_FOUR": (-4, 4)}.get(
            _opname, (_effect, 0)
        )

    def copy(self):
        return self.__class__(self._name, self._arg, lineno=self._lineno, offset=self.offset)

    def __repr__(self):
        if self._arg is not UNSET:
            return "<%s arg=%r lineno=%s>" % (self._name, self._arg, self._lineno)
        else:
            return "<%s lineno=%s>" % (self._name, self._lineno)

    def _cmp_key(self, labels=None):
        arg = self._arg
        if self._opcode in _opcode.hasconst:
            arg = const_key(arg)
        elif isinstance(arg, Label) and labels is not None:
            arg = labels[arg]
        return (self._lineno, self._name, arg)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self._cmp_key() == other._cmp_key()

    @staticmethod
    def _has_jump(opcode):
        return opcode in _opcode.hasjrel or opcode in _opcode.hasjabs

    def has_jump(self):
        return self._has_jump(self._opcode)

    def is_cond_jump(self):
        """Is a conditional jump?"""
        # Ex: POP_JUMP_IF_TRUE, JUMP_IF_FALSE_OR_POP
        return "JUMP_IF_" in self._name

    def is_uncond_jump(self):
        """Is an unconditional jump?"""
        return self.name in {"JUMP_FORWARD", "JUMP_ABSOLUTE"}

    def is_final(self):
        if self._name in {
            "RETURN_VALUE",
            "RAISE_VARARGS",
            "RERAISE",
            "BREAK_LOOP",
            "CONTINUE_LOOP",
        }:
            return True
        if self.is_uncond_jump():
            return True
        return False
