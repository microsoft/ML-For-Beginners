import sys
import textwrap
import types
import unittest

from _pydevd_frame_eval.vendored.bytecode import (
    UNSET,
    Label,
    Instr,
    ConcreteInstr,
    BasicBlock,  # noqa
    Bytecode,
    ControlFlowGraph,
    ConcreteBytecode,
)


def _format_instr_list(block, labels, lineno):
    instr_list = []
    for instr in block:
        if not isinstance(instr, Label):
            if isinstance(instr, ConcreteInstr):
                cls_name = "ConcreteInstr"
            else:
                cls_name = "Instr"
            arg = instr.arg
            if arg is not UNSET:
                if isinstance(arg, Label):
                    arg = labels[arg]
                elif isinstance(arg, BasicBlock):
                    arg = labels[id(arg)]
                else:
                    arg = repr(arg)
                if lineno:
                    text = "%s(%r, %s, lineno=%s)" % (
                        cls_name,
                        instr.name,
                        arg,
                        instr.lineno,
                    )
                else:
                    text = "%s(%r, %s)" % (cls_name, instr.name, arg)
            else:
                if lineno:
                    text = "%s(%r, lineno=%s)" % (cls_name, instr.name, instr.lineno)
                else:
                    text = "%s(%r)" % (cls_name, instr.name)
        else:
            text = labels[instr]
        instr_list.append(text)
    return "[%s]" % ",\n ".join(instr_list)


def dump_bytecode(code, lineno=False):
    """
    Use this function to write unit tests: copy/paste its output to
    write a self.assertBlocksEqual() check.
    """
    print()

    if isinstance(code, (Bytecode, ConcreteBytecode)):
        is_concrete = isinstance(code, ConcreteBytecode)
        if is_concrete:
            block = list(code)
        else:
            block = code

        indent = " " * 8
        labels = {}
        for index, instr in enumerate(block):
            if isinstance(instr, Label):
                name = "label_instr%s" % index
                labels[instr] = name

        if is_concrete:
            name = "ConcreteBytecode"
            print(indent + "code = %s()" % name)
            if code.argcount:
                print(indent + "code.argcount = %s" % code.argcount)
            if sys.version_info > (3, 8):
                if code.posonlyargcount:
                    print(indent + "code.posonlyargcount = %s" % code.posonlyargcount)
            if code.kwonlyargcount:
                print(indent + "code.kwargonlycount = %s" % code.kwonlyargcount)
            print(indent + "code.flags = %#x" % code.flags)
            if code.consts:
                print(indent + "code.consts = %r" % code.consts)
            if code.names:
                print(indent + "code.names = %r" % code.names)
            if code.varnames:
                print(indent + "code.varnames = %r" % code.varnames)

        for name in sorted(labels.values()):
            print(indent + "%s = Label()" % name)

        if is_concrete:
            text = indent + "code.extend("
            indent = " " * len(text)
        else:
            text = indent + "code = Bytecode("
            indent = " " * len(text)

        lines = _format_instr_list(code, labels, lineno).splitlines()
        last_line = len(lines) - 1
        for index, line in enumerate(lines):
            if index == 0:
                print(text + lines[0])
            elif index == last_line:
                print(indent + line + ")")
            else:
                print(indent + line)

        print()
    else:
        assert isinstance(code, ControlFlowGraph)
        labels = {}
        for block_index, block in enumerate(code):
            labels[id(block)] = "code[%s]" % block_index

        for block_index, block in enumerate(code):
            text = _format_instr_list(block, labels, lineno)
            if block_index != len(code) - 1:
                text += ","
            print(text)
            print()


def get_code(source, *, filename="<string>", function=False):
    source = textwrap.dedent(source).strip()
    code = compile(source, filename, "exec")
    if function:
        sub_code = [
            const for const in code.co_consts if isinstance(const, types.CodeType)
        ]
        if len(sub_code) != 1:
            raise ValueError("unable to find function code")
        code = sub_code[0]
    return code


def disassemble(source, *, filename="<string>", function=False):
    code = get_code(source, filename=filename, function=function)
    return Bytecode.from_code(code)


class TestCase(unittest.TestCase):
    def assertBlocksEqual(self, code, *expected_blocks):
        self.assertEqual(len(code), len(expected_blocks))

        for block1, block2 in zip(code, expected_blocks):
            block_index = code.get_block_index(block1)
            self.assertListEqual(
                list(block1), block2, "Block #%s is different" % block_index
            )
