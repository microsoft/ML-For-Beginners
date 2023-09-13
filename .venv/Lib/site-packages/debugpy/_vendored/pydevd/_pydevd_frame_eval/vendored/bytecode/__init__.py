__version__ = "0.13.0.dev"

__all__ = [
    "Label",
    "Instr",
    "SetLineno",
    "Bytecode",
    "ConcreteInstr",
    "ConcreteBytecode",
    "ControlFlowGraph",
    "CompilerFlags",
    "Compare",
]

from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import (
    UNSET,
    Label,
    SetLineno,
    Instr,
    CellVar,
    FreeVar,  # noqa
    Compare,
)
from _pydevd_frame_eval.vendored.bytecode.bytecode import (
    BaseBytecode,
    _BaseBytecodeList,
    _InstrList,
    Bytecode,
)  # noqa
from _pydevd_frame_eval.vendored.bytecode.concrete import (
    ConcreteInstr,
    ConcreteBytecode,  # noqa
    # import needed to use it in bytecode.py
    _ConvertBytecodeToConcrete,
)
from _pydevd_frame_eval.vendored.bytecode.cfg import BasicBlock, ControlFlowGraph  # noqa
import sys

def dump_bytecode(bytecode, *, lineno=False, stream=sys.stdout):
    def format_line(index, line):
        nonlocal cur_lineno, prev_lineno
        if lineno:
            if cur_lineno != prev_lineno:
                line = "L.% 3s % 3s: %s" % (cur_lineno, index, line)
                prev_lineno = cur_lineno
            else:
                line = "      % 3s: %s" % (index, line)
        else:
            line = line
        return line

    def format_instr(instr, labels=None):
        text = instr.name
        arg = instr._arg
        if arg is not UNSET:
            if isinstance(arg, Label):
                try:
                    arg = "<%s>" % labels[arg]
                except KeyError:
                    arg = "<error: unknown label>"
            elif isinstance(arg, BasicBlock):
                try:
                    arg = "<%s>" % labels[id(arg)]
                except KeyError:
                    arg = "<error: unknown block>"
            else:
                arg = repr(arg)
            text = "%s %s" % (text, arg)
        return text

    indent = " " * 4

    cur_lineno = bytecode.first_lineno
    prev_lineno = None

    if isinstance(bytecode, ConcreteBytecode):
        offset = 0
        for instr in bytecode:
            fields = []
            if instr.lineno is not None:
                cur_lineno = instr.lineno
            if lineno:
                fields.append(format_instr(instr))
                line = "".join(fields)
                line = format_line(offset, line)
            else:
                fields.append("% 3s    %s" % (offset, format_instr(instr)))
                line = "".join(fields)
            print(line, file=stream)

            offset += instr.size
    elif isinstance(bytecode, Bytecode):
        labels = {}
        for index, instr in enumerate(bytecode):
            if isinstance(instr, Label):
                labels[instr] = "label_instr%s" % index

        for index, instr in enumerate(bytecode):
            if isinstance(instr, Label):
                label = labels[instr]
                line = "%s:" % label
                if index != 0:
                    print(file=stream)
            else:
                if instr.lineno is not None:
                    cur_lineno = instr.lineno
                line = format_instr(instr, labels)
                line = indent + format_line(index, line)
            print(line, file=stream)
        print(file=stream)
    elif isinstance(bytecode, ControlFlowGraph):
        labels = {}
        for block_index, block in enumerate(bytecode, 1):
            labels[id(block)] = "block%s" % block_index

        for block_index, block in enumerate(bytecode, 1):
            print("%s:" % labels[id(block)], file=stream)
            prev_lineno = None
            for index, instr in enumerate(block):
                if instr.lineno is not None:
                    cur_lineno = instr.lineno
                line = format_instr(instr, labels)
                line = indent + format_line(index, line)
                print(line, file=stream)
            if block.next_block is not None:
                print(indent + "-> %s" % labels[id(block.next_block)], file=stream)
            print(file=stream)
    else:
        raise TypeError("unknown bytecode class")
