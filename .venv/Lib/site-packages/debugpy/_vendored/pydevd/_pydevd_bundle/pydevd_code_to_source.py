"""
Decompiler that can be used with the debugger (where statements correctly represent the
line numbers).

Note: this is a work in progress / proof of concept / not ready to be used.
"""

import dis

from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO


class _Stack(object):

    def __init__(self):
        self._contents = []

    def push(self, obj):
        #         print('push', obj)
        self._contents.append(obj)

    def pop(self):
        return self._contents.pop(-1)


INDENT_MARKER = object()
DEDENT_MARKER = object()
_SENTINEL = object()

DEBUG = False


class _Token(object):

    def __init__(self, i_line, instruction=None, tok=_SENTINEL, priority=0, after=None, end_of_line=False):
        '''
        :param i_line:
        :param instruction:
        :param tok:
        :param priority:
        :param after:
        :param end_of_line:
            Marker to signal only after all the other tokens have been written.
        '''
        self.i_line = i_line
        if tok is not _SENTINEL:
            self.tok = tok
        else:
            if instruction is not None:
                if inspect.iscode(instruction.argval):
                    self.tok = ''
                else:
                    self.tok = str(instruction.argval)
            else:
                raise AssertionError('Either the tok or the instruction is needed.')
        self.instruction = instruction
        self.priority = priority
        self.end_of_line = end_of_line
        self._after_tokens = set()
        self._after_handler_tokens = set()
        if after:
            self.mark_after(after)

    def mark_after(self, v):
        if isinstance(v, _Token):
            self._after_tokens.add(v)
        elif isinstance(v, _BaseHandler):
            self._after_handler_tokens.add(v)

        else:
            raise AssertionError('Unhandled: %s' % (v,))

    def get_after_tokens(self):
        ret = self._after_tokens.copy()
        for handler in self._after_handler_tokens:
            ret.update(handler.tokens)
        return ret

    def __repr__(self):
        return 'Token(%s, after: %s)' % (self.tok, self.get_after_tokens())

    __str__ = __repr__


class _Writer(object):

    def __init__(self):
        self.line_to_contents = {}
        self.all_tokens = set()

    def get_line(self, line):
        lst = self.line_to_contents.get(line)
        if lst is None:
            lst = self.line_to_contents[line] = []
        return lst

    def indent(self, line):
        self.get_line(line).append(INDENT_MARKER)

    def dedent(self, line):
        self.get_line(line).append(DEDENT_MARKER)

    def write(self, line, token):
        if token in self.all_tokens:
            return
        self.all_tokens.add(token)
        assert isinstance(token, _Token)
        lst = self.get_line(line)
        lst.append(token)


class _BaseHandler(object):

    def __init__(self, i_line, instruction, stack, writer, disassembler):
        self.i_line = i_line
        self.instruction = instruction
        self.stack = stack
        self.writer = writer
        self.disassembler = disassembler
        self.tokens = []
        self._handle()

    def _write_tokens(self):
        for token in self.tokens:
            self.writer.write(token.i_line, token)

    def _handle(self):
        raise NotImplementedError(self)

    def __repr__(self, *args, **kwargs):
        try:
            return "%s line:%s" % (self.instruction, self.i_line)
        except:
            return object.__repr__(self)

    __str__ = __repr__


_op_name_to_handler = {}


def _register(cls):
    _op_name_to_handler[cls.opname] = cls
    return cls


class _BasePushHandler(_BaseHandler):

    def _handle(self):
        self.stack.push(self)


class _BaseLoadHandler(_BasePushHandler):

    def _handle(self):
        _BasePushHandler._handle(self)
        self.tokens = [_Token(self.i_line, self.instruction)]


@_register
class _LoadBuildClass(_BasePushHandler):
    opname = "LOAD_BUILD_CLASS"


@_register
class _LoadConst(_BaseLoadHandler):
    opname = "LOAD_CONST"


@_register
class _LoadName(_BaseLoadHandler):
    opname = "LOAD_NAME"


@_register
class _LoadGlobal(_BaseLoadHandler):
    opname = "LOAD_GLOBAL"


@_register
class _LoadFast(_BaseLoadHandler):
    opname = "LOAD_FAST"


@_register
class _GetIter(_BaseHandler):
    '''
    Implements TOS = iter(TOS).
    '''
    opname = "GET_ITER"
    iter_target = None

    def _handle(self):
        self.iter_target = self.stack.pop()
        self.tokens.extend(self.iter_target.tokens)
        self.stack.push(self)


@_register
class _ForIter(_BaseHandler):
    '''
    TOS is an iterator. Call its __next__() method. If this yields a new value, push it on the stack
    (leaving the iterator below it). If the iterator indicates it is exhausted TOS is popped, and
    the byte code counter is incremented by delta.
    '''
    opname = "FOR_ITER"

    iter_in = None

    def _handle(self):
        self.iter_in = self.stack.pop()
        self.stack.push(self)

    def store_in_name(self, store_name):
        for_token = _Token(self.i_line, None, 'for ')
        self.tokens.append(for_token)
        prev = for_token

        t_name = _Token(store_name.i_line, store_name.instruction, after=prev)
        self.tokens.append(t_name)
        prev = t_name

        in_token = _Token(store_name.i_line, None, ' in ', after=prev)
        self.tokens.append(in_token)
        prev = in_token

        max_line = store_name.i_line
        if self.iter_in:
            for t in self.iter_in.tokens:
                t.mark_after(prev)
                max_line = max(max_line, t.i_line)
                prev = t
            self.tokens.extend(self.iter_in.tokens)

        colon_token = _Token(self.i_line, None, ':', after=prev)
        self.tokens.append(colon_token)
        prev = for_token

        self._write_tokens()


@_register
class _StoreName(_BaseHandler):
    '''
    Implements name = TOS. namei is the index of name in the attribute co_names of the code object.
    The compiler tries to use STORE_FAST or STORE_GLOBAL if possible.
    '''

    opname = "STORE_NAME"

    def _handle(self):
        v = self.stack.pop()

        if isinstance(v, _ForIter):
            v.store_in_name(self)
        else:
            if not isinstance(v, _MakeFunction) or v.is_lambda:
                line = self.i_line
                for t in v.tokens:
                    line = min(line, t.i_line)

                t_name = _Token(line, self.instruction)
                t_equal = _Token(line, None, '=', after=t_name)

                self.tokens.append(t_name)
                self.tokens.append(t_equal)

                for t in v.tokens:
                    t.mark_after(t_equal)
                self.tokens.extend(v.tokens)

                self._write_tokens()


@_register
class _ReturnValue(_BaseHandler):
    """
    Returns with TOS to the caller of the function.
    """

    opname = "RETURN_VALUE"

    def _handle(self):
        v = self.stack.pop()
        return_token = _Token(self.i_line, None, 'return ', end_of_line=True)
        self.tokens.append(return_token)
        for token in v.tokens:
            token.mark_after(return_token)
        self.tokens.extend(v.tokens)

        self._write_tokens()


@_register
class _CallFunction(_BaseHandler):
    """

    CALL_FUNCTION(argc)

        Calls a callable object with positional arguments. argc indicates the number of positional
        arguments. The top of the stack contains positional arguments, with the right-most argument
        on top. Below the arguments is a callable object to call. CALL_FUNCTION pops all arguments
        and the callable object off the stack, calls the callable object with those arguments, and
        pushes the return value returned by the callable object.

        Changed in version 3.6: This opcode is used only for calls with positional arguments.

    """

    opname = "CALL_FUNCTION"

    def _handle(self):
        args = []
        for _i in range(self.instruction.argval + 1):
            arg = self.stack.pop()
            args.append(arg)
        it = reversed(args)
        name = next(it)
        max_line = name.i_line
        for t in name.tokens:
            self.tokens.append(t)

        tok_open_parens = _Token(name.i_line, None, '(', after=name)
        self.tokens.append(tok_open_parens)

        prev = tok_open_parens
        for i, arg in enumerate(it):
            for t in arg.tokens:
                t.mark_after(name)
                t.mark_after(prev)
                max_line = max(max_line, t.i_line)
                self.tokens.append(t)
            prev = arg

            if i > 0:
                comma_token = _Token(prev.i_line, None, ',', after=prev)
                self.tokens.append(comma_token)
                prev = comma_token

        tok_close_parens = _Token(max_line, None, ')', after=prev)
        self.tokens.append(tok_close_parens)

        self._write_tokens()

        self.stack.push(self)


@_register
class _MakeFunctionPy3(_BaseHandler):
    """
    Pushes a new function object on the stack. From bottom to top, the consumed stack must consist
    of values if the argument carries a specified flag value

        0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order

        0x02 a dictionary of keyword-only parameters' default values

        0x04 an annotation dictionary

        0x08 a tuple containing cells for free variables, making a closure

        the code associated with the function (at TOS1)

        the qualified name of the function (at TOS)
    """

    opname = "MAKE_FUNCTION"
    is_lambda = False

    def _handle(self):
        stack = self.stack
        self.qualified_name = stack.pop()
        self.code = stack.pop()

        default_node = None
        if self.instruction.argval & 0x01:
            default_node = stack.pop()

        is_lambda = self.is_lambda = '<lambda>' in [x.tok for x in self.qualified_name.tokens]

        if not is_lambda:
            def_token = _Token(self.i_line, None, 'def ')
            self.tokens.append(def_token)

        for token in self.qualified_name.tokens:
            self.tokens.append(token)
            if not is_lambda:
                token.mark_after(def_token)
        prev = token

        open_parens_token = _Token(self.i_line, None, '(', after=prev)
        self.tokens.append(open_parens_token)
        prev = open_parens_token

        code = self.code.instruction.argval

        if default_node:
            defaults = ([_SENTINEL] * (len(code.co_varnames) - len(default_node.instruction.argval))) + list(default_node.instruction.argval)
        else:
            defaults = [_SENTINEL] * len(code.co_varnames)

        for i, arg in enumerate(code.co_varnames):
            if i > 0:
                comma_token = _Token(prev.i_line, None, ', ', after=prev)
                self.tokens.append(comma_token)
                prev = comma_token

            arg_token = _Token(self.i_line, None, arg, after=prev)
            self.tokens.append(arg_token)

            default = defaults[i]
            if default is not _SENTINEL:
                eq_token = _Token(default_node.i_line, None, '=', after=prev)
                self.tokens.append(eq_token)
                prev = eq_token

                default_token = _Token(default_node.i_line, None, str(default), after=prev)
                self.tokens.append(default_token)
                prev = default_token

        tok_close_parens = _Token(prev.i_line, None, '):', after=prev)
        self.tokens.append(tok_close_parens)

        self._write_tokens()

        stack.push(self)
        self.writer.indent(prev.i_line + 1)
        self.writer.dedent(max(self.disassembler.merge_code(code)))


_MakeFunction = _MakeFunctionPy3


def _print_after_info(line_contents, stream=None):
    if stream is None:
        stream = sys.stdout
    for token in line_contents:
        after_tokens = token.get_after_tokens()
        if after_tokens:
            s = '%s after: %s\n' % (
                repr(token.tok),
                ('"' + '", "'.join(t.tok for t in token.get_after_tokens()) + '"'))
            stream.write(s)
        else:
            stream.write('%s      (NO REQUISITES)' % repr(token.tok))


def _compose_line_contents(line_contents, previous_line_tokens):
    lst = []
    handled = set()

    add_to_end_of_line = []
    delete_indexes = []
    for i, token in enumerate(line_contents):
        if token.end_of_line:
            add_to_end_of_line.append(token)
            delete_indexes.append(i)
    for i in reversed(delete_indexes):
        del line_contents[i]
    del delete_indexes

    while line_contents:
        added = False
        delete_indexes = []

        for i, token in enumerate(line_contents):
            after_tokens = token.get_after_tokens()
            for after in after_tokens:
                if after not in handled and after not in previous_line_tokens:
                    break
            else:
                added = True
                previous_line_tokens.add(token)
                handled.add(token)
                lst.append(token.tok)
                delete_indexes.append(i)

        for i in reversed(delete_indexes):
            del line_contents[i]

        if not added:
            if add_to_end_of_line:
                line_contents.extend(add_to_end_of_line)
                del add_to_end_of_line[:]
                continue

            # Something is off, let's just add as is.
            for token in line_contents:
                if token not in handled:
                    lst.append(token.tok)

            stream = StringIO()
            _print_after_info(line_contents, stream)
            pydev_log.critical('Error. After markers are not correct:\n%s', stream.getvalue())
            break
    return ''.join(lst)


class _PyCodeToSource(object):

    def __init__(self, co, memo=None):
        if memo is None:
            memo = {}
        self.memo = memo
        self.co = co
        self.instructions = list(iter_instructions(co))
        self.stack = _Stack()
        self.writer = _Writer()

    def _process_next(self, i_line):
        instruction = self.instructions.pop(0)
        handler_class = _op_name_to_handler.get(instruction.opname)
        if handler_class is not None:
            s = handler_class(i_line, instruction, self.stack, self.writer, self)
            if DEBUG:
                print(s)

        else:
            if DEBUG:
                print("UNHANDLED", instruction)

    def build_line_to_contents(self):
        co = self.co

        op_offset_to_line = dict(dis.findlinestarts(co))
        curr_line_index = 0

        instructions = self.instructions
        while instructions:
            instruction = instructions[0]
            new_line_index = op_offset_to_line.get(instruction.offset)
            if new_line_index is not None:
                if new_line_index is not None:
                    curr_line_index = new_line_index

            self._process_next(curr_line_index)
        return self.writer.line_to_contents

    def merge_code(self, code):
        if DEBUG:
            print('merge code ----')
        # for d in dir(code):
        #     if not d.startswith('_'):
        #         print(d, getattr(code, d))
        line_to_contents = _PyCodeToSource(code, self.memo).build_line_to_contents()
        lines = []
        for line, contents in sorted(line_to_contents.items()):
            lines.append(line)
            self.writer.get_line(line).extend(contents)
        if DEBUG:
            print('end merge code ----')
        return lines

    def disassemble(self):
        show_lines = False
        line_to_contents = self.build_line_to_contents()
        stream = StringIO()
        last_line = 0
        indent = ''
        previous_line_tokens = set()
        for i_line, contents in sorted(line_to_contents.items()):
            while last_line < i_line - 1:
                if show_lines:
                    stream.write(u"%s.\n" % (last_line + 1,))
                else:
                    stream.write(u"\n")
                last_line += 1

            line_contents = []
            dedents_found = 0
            for part in contents:
                if part is INDENT_MARKER:
                    if DEBUG:
                        print('found indent', i_line)
                    indent += '    '
                    continue
                if part is DEDENT_MARKER:
                    if DEBUG:
                        print('found dedent', i_line)
                    dedents_found += 1
                    continue
                line_contents.append(part)

            s = indent + _compose_line_contents(line_contents, previous_line_tokens)
            if show_lines:
                stream.write(u"%s. %s\n" % (i_line, s))
            else:
                stream.write(u"%s\n" % s)

            if dedents_found:
                indent = indent[:-(4 * dedents_found)]
            last_line = i_line

        return stream.getvalue()


def code_obj_to_source(co):
    """
    Converts a code object to source code to provide a suitable representation for the compiler when
    the actual source code is not found.

    This is a work in progress / proof of concept / not ready to be used.
    """
    ret = _PyCodeToSource(co).disassemble()
    if DEBUG:
        print(ret)
    return ret
