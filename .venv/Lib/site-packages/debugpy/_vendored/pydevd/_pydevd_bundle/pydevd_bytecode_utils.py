"""
Bytecode analysing utils. Originally added for using in smart step into.

Note: not importable from Python 2.
"""

from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode

from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque

# When True, throws errors on unknown bytecodes, when False, ignore those as if they didn't change the stack.
STRICT_MODE = False

DEBUG = False

_BINARY_OPS = set([opname for opname in dis.opname if opname.startswith('BINARY_')])

_BINARY_OP_MAP = {
    'BINARY_POWER': '__pow__',
    'BINARY_MULTIPLY': '__mul__',
    'BINARY_MATRIX_MULTIPLY': '__matmul__',
    'BINARY_FLOOR_DIVIDE': '__floordiv__',
    'BINARY_TRUE_DIVIDE': '__div__',
    'BINARY_MODULO': '__mod__',
    'BINARY_ADD': '__add__',
    'BINARY_SUBTRACT': '__sub__',
    'BINARY_LSHIFT': '__lshift__',
    'BINARY_RSHIFT': '__rshift__',
    'BINARY_AND': '__and__',
    'BINARY_OR': '__or__',
    'BINARY_XOR': '__xor__',
    'BINARY_SUBSCR': '__getitem__',
    'BINARY_DIVIDE': '__div__'
}

_COMP_OP_MAP = {
    '<': '__lt__',
    '<=': '__le__',
    '==': '__eq__',
    '!=': '__ne__',
    '>': '__gt__',
    '>=': '__ge__',
    'in': '__contains__',
    'not in': '__contains__',
}


class Target(object):
    __slots__ = ['arg', 'lineno', 'offset', 'children_targets']

    def __init__(self, arg, lineno, offset, children_targets=()):
        self.arg = arg
        self.lineno = lineno
        self.offset = offset
        self.children_targets = children_targets

    def __repr__(self):
        ret = []
        for s in self.__slots__:
            ret.append('%s: %s' % (s, getattr(self, s)))
        return 'Target(%s)' % ', '.join(ret)

    __str__ = __repr__


class _TargetIdHashable(object):

    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        if not hasattr(other, 'target'):
            return
        return other.target is self.target

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return id(self.target)


class _StackInterpreter(object):
    '''
    Good reference: https://github.com/python/cpython/blob/fcb55c0037baab6f98f91ee38ce84b6f874f034a/Python/ceval.c
    '''

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self._stack = deque()
        self.function_calls = []
        self.load_attrs = {}
        self.func = set()
        self.func_name_id_to_code_object = {}

    def __str__(self):
        return 'Stack:\nFunction calls:\n%s\nLoad attrs:\n%s\n' % (self.function_calls, list(self.load_attrs.values()))

    def _getname(self, instr):
        if instr.opcode in _opcode.hascompare:
            cmp_op = dis.cmp_op[instr.arg]
            if cmp_op not in ('exception match', 'BAD'):
                return _COMP_OP_MAP.get(cmp_op, cmp_op)
        return instr.arg

    def _getcallname(self, instr):
        if instr.name == 'BINARY_SUBSCR':
            return '__getitem__().__call__'
        if instr.name == 'CALL_FUNCTION':
            # Note: previously a '__call__().__call__' was returned, but this was a bit weird
            # and on Python 3.9 this construct could appear for some internal things where
            # it wouldn't be expected.
            # Note: it'd be what we had in func()().
            return None
        if instr.name == 'MAKE_FUNCTION':
            return '__func__().__call__'
        if instr.name == 'LOAD_ASSERTION_ERROR':
            return 'AssertionError'
        name = self._getname(instr)
        if isinstance(name, CodeType):
            name = name.co_qualname  # Note: only available for Python 3.11
        if isinstance(name, _Variable):
            name = name.name

        if not isinstance(name, str):
            return None
        if name.endswith('>'):  # xxx.<listcomp>, xxx.<lambda>, ...
            return name.split('.')[-1]
        return name

    def _no_stack_change(self, instr):
        pass  # Can be aliased when the instruction does nothing.

    def on_LOAD_GLOBAL(self, instr):
        self._stack.append(instr)

    def on_POP_TOP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            pass  # Ok (in the end of blocks)

    def on_LOAD_ATTR(self, instr):
        self.on_POP_TOP(instr)  # replaces the current top
        self._stack.append(instr)
        self.load_attrs[_TargetIdHashable(instr)] = Target(self._getname(instr), instr.lineno, instr.offset)

    on_LOOKUP_METHOD = on_LOAD_ATTR  # Improvement in PyPy

    def on_LOAD_CONST(self, instr):
        self._stack.append(instr)

    on_LOAD_DEREF = on_LOAD_CONST
    on_LOAD_NAME = on_LOAD_CONST
    on_LOAD_CLOSURE = on_LOAD_CONST
    on_LOAD_CLASSDEREF = on_LOAD_CONST

    # Although it actually changes the stack, it's inconsequential for us as a function call can't
    # really be found there.
    on_IMPORT_NAME = _no_stack_change
    on_IMPORT_FROM = _no_stack_change
    on_IMPORT_STAR = _no_stack_change
    on_SETUP_ANNOTATIONS = _no_stack_change

    def on_STORE_FAST(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            pass  # Ok, we may have a block just with the store

        # Note: it stores in the locals and doesn't put anything in the stack.

    on_STORE_GLOBAL = on_STORE_FAST
    on_STORE_DEREF = on_STORE_FAST
    on_STORE_ATTR = on_STORE_FAST
    on_STORE_NAME = on_STORE_FAST

    on_DELETE_NAME = on_POP_TOP
    on_DELETE_ATTR = on_POP_TOP
    on_DELETE_GLOBAL = on_POP_TOP
    on_DELETE_FAST = on_POP_TOP
    on_DELETE_DEREF = on_POP_TOP

    on_DICT_UPDATE = on_POP_TOP
    on_SET_UPDATE = on_POP_TOP

    on_GEN_START = on_POP_TOP

    def on_NOP(self, instr):
        pass

    def _handle_call_from_instr(self, func_name_instr, func_call_instr):
        self.load_attrs.pop(_TargetIdHashable(func_name_instr), None)
        call_name = self._getcallname(func_name_instr)
        target = None
        if not call_name:
            pass  # Ignore if we can't identify a name
        elif call_name in ('<listcomp>', '<genexpr>', '<setcomp>', '<dictcomp>'):
            code_obj = self.func_name_id_to_code_object[_TargetIdHashable(func_name_instr)]
            if code_obj is not None:
                children_targets = _get_smart_step_into_targets(code_obj)
                if children_targets:
                    # i.e.: we have targets inside of a <listcomp> or <genexpr>.
                    # Note that to actually match this in the debugger we need to do matches on 2 frames,
                    # the one with the <listcomp> and then the actual target inside the <listcomp>.
                    target = Target(call_name, func_name_instr.lineno, func_call_instr.offset, children_targets)
                    self.function_calls.append(
                        target)

        else:
            # Ok, regular call
            target = Target(call_name, func_name_instr.lineno, func_call_instr.offset)
            self.function_calls.append(target)

        if DEBUG and target is not None:
            print('Created target', target)
        self._stack.append(func_call_instr)  # Keep the func call as the result

    def on_COMPARE_OP(self, instr):
        try:
            _right = self._stack.pop()
        except IndexError:
            return
        try:
            _left = self._stack.pop()
        except IndexError:
            return

        cmp_op = dis.cmp_op[instr.arg]
        if cmp_op not in ('exception match', 'BAD'):
            self.function_calls.append(Target(self._getname(instr), instr.lineno, instr.offset))

        self._stack.append(instr)

    def on_IS_OP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return
        try:
            self._stack.pop()
        except IndexError:
            return

    def on_BINARY_SUBSCR(self, instr):
        try:
            _sub = self._stack.pop()
        except IndexError:
            return
        try:
            _container = self._stack.pop()
        except IndexError:
            return
        self.function_calls.append(Target(_BINARY_OP_MAP[instr.name], instr.lineno, instr.offset))
        self._stack.append(instr)

    on_BINARY_MATRIX_MULTIPLY = on_BINARY_SUBSCR
    on_BINARY_POWER = on_BINARY_SUBSCR
    on_BINARY_MULTIPLY = on_BINARY_SUBSCR
    on_BINARY_FLOOR_DIVIDE = on_BINARY_SUBSCR
    on_BINARY_TRUE_DIVIDE = on_BINARY_SUBSCR
    on_BINARY_MODULO = on_BINARY_SUBSCR
    on_BINARY_ADD = on_BINARY_SUBSCR
    on_BINARY_SUBTRACT = on_BINARY_SUBSCR
    on_BINARY_LSHIFT = on_BINARY_SUBSCR
    on_BINARY_RSHIFT = on_BINARY_SUBSCR
    on_BINARY_AND = on_BINARY_SUBSCR
    on_BINARY_OR = on_BINARY_SUBSCR
    on_BINARY_XOR = on_BINARY_SUBSCR

    def on_LOAD_METHOD(self, instr):
        self.on_POP_TOP(instr)  # Remove the previous as we're loading something from it.
        self._stack.append(instr)

    def on_MAKE_FUNCTION(self, instr):
        if not IS_PY311_OR_GREATER:
            # The qualifier name is no longer put in the stack.
            qualname = self._stack.pop()
            code_obj_instr = self._stack.pop()
        else:
            # In 3.11 the code object has a co_qualname which we can use.
            qualname = code_obj_instr = self._stack.pop()

        arg = instr.arg
        if arg & 0x08:
            _func_closure = self._stack.pop()
        if arg & 0x04:
            _func_annotations = self._stack.pop()
        if arg & 0x02:
            _func_kwdefaults = self._stack.pop()
        if arg & 0x01:
            _func_defaults = self._stack.pop()

        call_name = self._getcallname(qualname)
        if call_name in ('<genexpr>', '<listcomp>', '<setcomp>', '<dictcomp>'):
            if isinstance(code_obj_instr.arg, CodeType):
                self.func_name_id_to_code_object[_TargetIdHashable(qualname)] = code_obj_instr.arg
        self._stack.append(qualname)

    def on_LOAD_FAST(self, instr):
        self._stack.append(instr)

    def on_LOAD_ASSERTION_ERROR(self, instr):
        self._stack.append(instr)

    on_LOAD_BUILD_CLASS = on_LOAD_FAST

    def on_CALL_METHOD(self, instr):
        # pop the actual args
        for _ in range(instr.arg):
            self._stack.pop()

        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_PUSH_NULL(self, instr):
        self._stack.append(instr)

    def on_CALL_FUNCTION(self, instr):
        arg = instr.arg

        argc = arg & 0xff  # positional args
        argc += ((arg >> 8) * 2)  # keyword args

        # pop the actual args
        for _ in range(argc):
            try:
                self._stack.pop()
            except IndexError:
                return

        try:
            func_name_instr = self._stack.pop()
        except IndexError:
            return
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_KW(self, instr):
        # names of kw args
        _names_of_kw_args = self._stack.pop()

        # pop the actual args
        arg = instr.arg

        argc = arg & 0xff  # positional args
        argc += ((arg >> 8) * 2)  # keyword args

        for _ in range(argc):
            self._stack.pop()

        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_VAR(self, instr):
        # var name
        _var_arg = self._stack.pop()

        # pop the actual args
        arg = instr.arg

        argc = arg & 0xff  # positional args
        argc += ((arg >> 8) * 2)  # keyword args

        for _ in range(argc):
            self._stack.pop()

        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_VAR_KW(self, instr):
        # names of kw args
        _names_of_kw_args = self._stack.pop()

        arg = instr.arg

        argc = arg & 0xff  # positional args
        argc += ((arg >> 8) * 2)  # keyword args

        # also pop **kwargs
        self._stack.pop()

        # pop the actual args
        for _ in range(argc):
            self._stack.pop()

        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_EX(self, instr):
        if instr.arg & 0x01:
            _kwargs = self._stack.pop()
        _callargs = self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    on_YIELD_VALUE = _no_stack_change
    on_GET_AITER = _no_stack_change
    on_GET_ANEXT = _no_stack_change
    on_END_ASYNC_FOR = _no_stack_change
    on_BEFORE_ASYNC_WITH = _no_stack_change
    on_SETUP_ASYNC_WITH = _no_stack_change
    on_YIELD_FROM = _no_stack_change
    on_SETUP_LOOP = _no_stack_change
    on_FOR_ITER = _no_stack_change
    on_BREAK_LOOP = _no_stack_change
    on_JUMP_ABSOLUTE = _no_stack_change
    on_RERAISE = _no_stack_change
    on_LIST_TO_TUPLE = _no_stack_change
    on_CALL_FINALLY = _no_stack_change
    on_POP_FINALLY = _no_stack_change

    def on_JUMP_IF_FALSE_OR_POP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return

    on_JUMP_IF_TRUE_OR_POP = on_JUMP_IF_FALSE_OR_POP

    def on_JUMP_IF_NOT_EXC_MATCH(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return
        try:
            self._stack.pop()
        except IndexError:
            return

    def on_ROT_TWO(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return

        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return

        self._stack.append(p0)
        self._stack.append(p1)

    def on_ROT_THREE(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return

        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return

        try:
            p2 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            return

        self._stack.append(p0)
        self._stack.append(p1)
        self._stack.append(p2)

    def on_ROT_FOUR(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return

        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return

        try:
            p2 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            return

        try:
            p3 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            self._stack.append(p2)
            return

        self._stack.append(p0)
        self._stack.append(p1)
        self._stack.append(p2)
        self._stack.append(p3)

    def on_BUILD_LIST_FROM_ARG(self, instr):
        self._stack.append(instr)

    def on_BUILD_MAP(self, instr):
        for _i in range(instr.arg):
            self._stack.pop()
            self._stack.pop()
        self._stack.append(instr)

    def on_BUILD_CONST_KEY_MAP(self, instr):
        self.on_POP_TOP(instr)  # keys
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)  # value
        self._stack.append(instr)

    on_RETURN_VALUE = on_POP_TOP
    on_POP_JUMP_IF_FALSE = on_POP_TOP
    on_POP_JUMP_IF_TRUE = on_POP_TOP
    on_DICT_MERGE = on_POP_TOP
    on_LIST_APPEND = on_POP_TOP
    on_SET_ADD = on_POP_TOP
    on_LIST_EXTEND = on_POP_TOP
    on_UNPACK_EX = on_POP_TOP

    # ok: doesn't change the stack (converts top to getiter(top))
    on_GET_ITER = _no_stack_change
    on_GET_AWAITABLE = _no_stack_change
    on_GET_YIELD_FROM_ITER = _no_stack_change

    def on_RETURN_GENERATOR(self, instr):
        self._stack.append(instr)

    on_RETURN_GENERATOR = _no_stack_change
    on_RESUME = _no_stack_change

    def on_MAP_ADD(self, instr):
        self.on_POP_TOP(instr)
        self.on_POP_TOP(instr)

    def on_UNPACK_SEQUENCE(self, instr):
        self._stack.pop()
        for _i in range(instr.arg):
            self._stack.append(instr)

    def on_BUILD_LIST(self, instr):
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)
        self._stack.append(instr)

    on_BUILD_TUPLE = on_BUILD_LIST
    on_BUILD_STRING = on_BUILD_LIST
    on_BUILD_TUPLE_UNPACK_WITH_CALL = on_BUILD_LIST
    on_BUILD_TUPLE_UNPACK = on_BUILD_LIST
    on_BUILD_LIST_UNPACK = on_BUILD_LIST
    on_BUILD_MAP_UNPACK_WITH_CALL = on_BUILD_LIST
    on_BUILD_MAP_UNPACK = on_BUILD_LIST
    on_BUILD_SET = on_BUILD_LIST
    on_BUILD_SET_UNPACK = on_BUILD_LIST

    on_SETUP_FINALLY = _no_stack_change
    on_POP_FINALLY = _no_stack_change
    on_BEGIN_FINALLY = _no_stack_change
    on_END_FINALLY = _no_stack_change

    def on_RAISE_VARARGS(self, instr):
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)

    on_POP_BLOCK = _no_stack_change
    on_JUMP_FORWARD = _no_stack_change
    on_POP_EXCEPT = _no_stack_change
    on_SETUP_EXCEPT = _no_stack_change
    on_WITH_EXCEPT_START = _no_stack_change

    on_END_FINALLY = _no_stack_change
    on_BEGIN_FINALLY = _no_stack_change
    on_SETUP_WITH = _no_stack_change
    on_WITH_CLEANUP_START = _no_stack_change
    on_WITH_CLEANUP_FINISH = _no_stack_change
    on_FORMAT_VALUE = _no_stack_change
    on_EXTENDED_ARG = _no_stack_change

    def on_INPLACE_ADD(self, instr):
        # This would actually pop 2 and leave the value in the stack.
        # In a += 1 it pop `a` and `1` and leave the resulting value
        # for a load. In our case, let's just pop the `1` and leave the `a`
        # instead of leaving the INPLACE_ADD bytecode.
        try:
            self._stack.pop()
        except IndexError:
            pass

    on_INPLACE_POWER = on_INPLACE_ADD
    on_INPLACE_MULTIPLY = on_INPLACE_ADD
    on_INPLACE_MATRIX_MULTIPLY = on_INPLACE_ADD
    on_INPLACE_TRUE_DIVIDE = on_INPLACE_ADD
    on_INPLACE_FLOOR_DIVIDE = on_INPLACE_ADD
    on_INPLACE_MODULO = on_INPLACE_ADD
    on_INPLACE_SUBTRACT = on_INPLACE_ADD
    on_INPLACE_RSHIFT = on_INPLACE_ADD
    on_INPLACE_LSHIFT = on_INPLACE_ADD
    on_INPLACE_AND = on_INPLACE_ADD
    on_INPLACE_OR = on_INPLACE_ADD
    on_INPLACE_XOR = on_INPLACE_ADD

    def on_DUP_TOP(self, instr):
        try:
            i = self._stack[-1]
        except IndexError:
            # ok (in the start of block)
            self._stack.append(instr)
        else:
            self._stack.append(i)

    def on_DUP_TOP_TWO(self, instr):
        if len(self._stack) == 0:
            self._stack.append(instr)
            return

        if len(self._stack) == 1:
            i = self._stack[-1]
            self._stack.append(i)
            self._stack.append(instr)
            return

        i = self._stack[-1]
        j = self._stack[-2]
        self._stack.append(j)
        self._stack.append(i)

    def on_BUILD_SLICE(self, instr):
        for _ in range(instr.arg):
            try:
                self._stack.pop()
            except IndexError:
                pass
        self._stack.append(instr)

    def on_STORE_SUBSCR(self, instr):
        try:
            self._stack.pop()
            self._stack.pop()
            self._stack.pop()
        except IndexError:
            pass

    def on_DELETE_SUBSCR(self, instr):
        try:
            self._stack.pop()
            self._stack.pop()
        except IndexError:
            pass

    # Note: on Python 3 this is only found on interactive mode to print the results of
    # some evaluation.
    on_PRINT_EXPR = on_POP_TOP

    on_UNARY_POSITIVE = _no_stack_change
    on_UNARY_NEGATIVE = _no_stack_change
    on_UNARY_NOT = _no_stack_change
    on_UNARY_INVERT = _no_stack_change

    on_CACHE = _no_stack_change
    on_PRECALL = _no_stack_change


def _get_smart_step_into_targets(code):
    '''
    :return list(Target)
    '''
    b = bytecode.Bytecode.from_code(code)
    cfg = bytecode_cfg.ControlFlowGraph.from_bytecode(b)

    ret = []

    for block in cfg:
        if DEBUG:
            print('\nStart block----')
        stack = _StackInterpreter(block)
        for instr in block:
            try:
                func_name = 'on_%s' % (instr.name,)
                func = getattr(stack, func_name, None)

                if DEBUG:
                    if instr.name != 'CACHE':  # Filter the ones we don't want to see.
                        print('\nWill handle: ', instr, '>>', stack._getname(instr), '<<')
                        print('Current stack:')
                        for entry in stack._stack:
                            print('    arg:', stack._getname(entry), '(', entry, ')')

                if func is None:
                    if STRICT_MODE:
                        raise AssertionError('%s not found.' % (func_name,))
                    else:
                        continue
                func(instr)
            except:
                if STRICT_MODE:
                    raise  # Error in strict mode.
                else:
                    # In non-strict mode, log it (if in verbose mode) and keep on going.
                    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
                        pydev_log.exception('Exception computing step into targets (handled).')

        ret.extend(stack.function_calls)
        # No longer considering attr loads as calls (while in theory sometimes it's possible
        # that something as `some.attr` can turn out to be a property which could be stepped
        # in, it's not that common in practice and can be surprising for users, so, disabling
        # step into from stepping into properties).
        # ret.extend(stack.load_attrs.values())

    return ret


# Note that the offset is unique within the frame (so, we can use it as the target id).
# Also, as the offset is the instruction offset within the frame, it's possible to
# to inspect the parent frame for frame.f_lasti to know where we actually are (as the
# caller name may not always match the new frame name).
class Variant(object):
    __slots__ = ['name', 'is_visited', 'line', 'offset', 'call_order', 'children_variants', 'parent']

    def __init__(self, name, is_visited, line, offset, call_order, children_variants=None):
        self.name = name
        self.is_visited = is_visited
        self.line = line
        self.offset = offset
        self.call_order = call_order
        self.children_variants = children_variants
        self.parent = None
        if children_variants:
            for variant in children_variants:
                variant.parent = self

    def __repr__(self):
        ret = []
        for s in self.__slots__:
            if s == 'parent':
                try:
                    parent = self.parent
                except AttributeError:
                    ret.append('%s: <not set>' % (s,))
                else:
                    if parent is None:
                        ret.append('parent: None')
                    else:
                        ret.append('parent: %s (%s)' % (parent.name, parent.offset))
                continue

            if s == 'children_variants':
                ret.append('children_variants: %s' % (len(self.children_variants) if self.children_variants else 0))
                continue

            try:
                ret.append('%s: %s' % (s, getattr(self, s)))
            except AttributeError:
                ret.append('%s: <not set>' % (s,))
        return 'Variant(%s)' % ', '.join(ret)

    __str__ = __repr__


def _convert_target_to_variant(target, start_line, end_line, call_order_cache, lasti, base):
    name = target.arg
    if not isinstance(name, str):
        return
    if target.lineno > end_line:
        return
    if target.lineno < start_line:
        return

    call_order = call_order_cache.get(name, 0) + 1
    call_order_cache[name] = call_order
    is_visited = target.offset <= lasti

    children_targets = target.children_targets
    children_variants = None
    if children_targets:
        children_variants = [
            _convert_target_to_variant(child, start_line, end_line, call_order_cache, lasti, base)
            for child in target.children_targets]

    return Variant(name, is_visited, target.lineno - base, target.offset, call_order, children_variants)


def calculate_smart_step_into_variants(frame, start_line, end_line, base=0):
    """
    Calculate smart step into variants for the given line range.
    :param frame:
    :type frame: :py:class:`types.FrameType`
    :param start_line:
    :param end_line:
    :return: A list of call names from the first to the last.
    :note: it's guaranteed that the offsets appear in order.
    :raise: :py:class:`RuntimeError` if failed to parse the bytecode or if dis cannot be used.
    """
    variants = []
    code = frame.f_code
    lasti = frame.f_lasti

    call_order_cache = {}
    if DEBUG:
        print('dis.dis:')
        if IS_PY311_OR_GREATER:
            dis.dis(code, show_caches=False)
        else:
            dis.dis(code)

    for target in _get_smart_step_into_targets(code):
        variant = _convert_target_to_variant(target, start_line, end_line, call_order_cache, lasti, base)
        if variant is None:
            continue
        variants.append(variant)

    return variants


def get_smart_step_into_variant_from_frame_offset(frame_f_lasti, variants):
    """
    Given the frame.f_lasti, return the related `Variant`.

    :note: if the offset is found before any variant available or no variants are
           available, None is returned.

    :rtype: Variant|NoneType
    """
    if not variants:
        return None

    i = bisect(KeyifyList(variants, lambda entry:entry.offset), frame_f_lasti)

    if i == 0:
        return None

    else:
        return variants[i - 1]
