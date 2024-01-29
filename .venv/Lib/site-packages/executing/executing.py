"""
MIT License

Copyright (c) 2021 Alex Hall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
    Type, TypeVar, Union, cast

if TYPE_CHECKING:  # pragma: no cover
    from asttokens import ASTTokens, ASTText
    from asttokens.asttokens import ASTTextBase


function_node_types = (ast.FunctionDef, ast.AsyncFunctionDef) # type: Tuple[Type, ...]

cache = lru_cache(maxsize=None)

# Type class used to expand out the definition of AST to include fields added by this library
# It's not actually used for anything other than type checking though!
class EnhancedAST(ast.AST):
    parent = None  # type: EnhancedAST


class Instruction(dis.Instruction):
    lineno = None  # type: int


# Type class used to expand out the definition of AST to include fields added by this library
# It's not actually used for anything other than type checking though!
class EnhancedInstruction(Instruction):
    _copied = None # type: bool



def assert_(condition, message=""):
    # type: (Any, str) -> None
    """
    Like an assert statement, but unaffected by -O
    :param condition: value that is expected to be truthy
    :type message: Any
    """
    if not condition:
        raise AssertionError(str(message))


def get_instructions(co):
    # type: (types.CodeType) -> Iterator[EnhancedInstruction]
    lineno = co.co_firstlineno
    for inst in dis.get_instructions(co):
        inst = cast(EnhancedInstruction, inst)
        lineno = inst.starts_line or lineno
        assert_(lineno)
        inst.lineno = lineno
        yield inst


TESTING = 0


class NotOneValueFound(Exception):
    def __init__(self,msg,values=[]):
        # type: (str, Sequence) -> None
        self.values=values
        super(NotOneValueFound,self).__init__(msg)

T = TypeVar('T')


def only(it):
    # type: (Iterable[T]) -> T
    if isinstance(it, Sized):
        if len(it) != 1:
            raise NotOneValueFound('Expected one value, found %s' % len(it))
        # noinspection PyTypeChecker
        return list(it)[0]

    lst = tuple(islice(it, 2))
    if len(lst) == 0:
        raise NotOneValueFound('Expected one value, found 0')
    if len(lst) > 1:
        raise NotOneValueFound('Expected one value, found several',lst)
    return lst[0]


class Source(object):
    """
    The source code of a single file and associated metadata.

    The main method of interest is the classmethod `executing(frame)`.

    If you want an instance of this class, don't construct it.
    Ideally use the classmethod `for_frame(frame)`.
    If you don't have a frame, use `for_filename(filename [, module_globals])`.
    These methods cache instances by filename, so at most one instance exists per filename.

    Attributes:
        - filename
        - text
        - lines
        - tree: AST parsed from text, or None if text is not valid Python
            All nodes in the tree have an extra `parent` attribute

    Other methods of interest:
        - statements_at_line
        - asttokens
        - code_qualname
    """

    def __init__(self, filename, lines):
        # type: (str, Sequence[str]) -> None
        """
        Don't call this constructor, see the class docstring.
        """

        self.filename = filename
        self.text = ''.join(lines)
        self.lines = [line.rstrip('\r\n') for line in lines]

        self._nodes_by_line = defaultdict(list)
        self.tree = None
        self._qualnames = {}
        self._asttokens = None  # type: Optional[ASTTokens]
        self._asttext = None  # type: Optional[ASTText]

        try:
            self.tree = ast.parse(self.text, filename=filename)
        except (SyntaxError, ValueError):
            pass
        else:
            for node in ast.walk(self.tree):
                for child in ast.iter_child_nodes(node):
                    cast(EnhancedAST, child).parent = cast(EnhancedAST, node)
                for lineno in node_linenos(node):
                    self._nodes_by_line[lineno].append(node)

            visitor = QualnameVisitor()
            visitor.visit(self.tree)
            self._qualnames = visitor.qualnames

    @classmethod
    def for_frame(cls, frame, use_cache=True):
        # type: (types.FrameType, bool) -> "Source"
        """
        Returns the `Source` object corresponding to the file the frame is executing in.
        """
        return cls.for_filename(frame.f_code.co_filename, frame.f_globals or {}, use_cache)

    @classmethod
    def for_filename(
        cls,
        filename,
        module_globals=None,
        use_cache=True,  # noqa no longer used
    ):
        # type: (Union[str, Path], Optional[Dict[str, Any]], bool) -> "Source"
        if isinstance(filename, Path):
            filename = str(filename)

        def get_lines():
            # type: () -> List[str]
            return linecache.getlines(cast(str, filename), module_globals)

        # Save the current linecache entry, then ensure the cache is up to date.
        entry = linecache.cache.get(filename) # type: ignore[attr-defined]
        linecache.checkcache(filename)
        lines = get_lines()
        if entry is not None and not lines:
            # There was an entry, checkcache removed it, and nothing replaced it.
            # This means the file wasn't simply changed (because the `lines` wouldn't be empty)
            # but rather the file was found not to exist, probably because `filename` was fake.
            # Restore the original entry so that we still have something.
            linecache.cache[filename] = entry # type: ignore[attr-defined]
            lines = get_lines()

        return cls._for_filename_and_lines(filename, tuple(lines))

    @classmethod
    def _for_filename_and_lines(cls, filename, lines):
        # type: (str, Sequence[str]) -> "Source"
        source_cache = cls._class_local('__source_cache_with_lines', {}) # type: Dict[Tuple[str, Sequence[str]], Source]
        try:
            return source_cache[(filename, lines)]
        except KeyError:
            pass

        result = source_cache[(filename, lines)] = cls(filename, lines)
        return result

    @classmethod
    def lazycache(cls, frame):
        # type: (types.FrameType) -> None
        linecache.lazycache(frame.f_code.co_filename, frame.f_globals)

    @classmethod
    def executing(cls, frame_or_tb):
        # type: (Union[types.TracebackType, types.FrameType]) -> "Executing"
        """
        Returns an `Executing` object representing the operation
        currently executing in the given frame or traceback object.
        """
        if isinstance(frame_or_tb, types.TracebackType):
            # https://docs.python.org/3/reference/datamodel.html#traceback-objects
            # "tb_lineno gives the line number where the exception occurred;
            #  tb_lasti indicates the precise instruction.
            #  The line number and last instruction in the traceback may differ
            #  from the line number of its frame object
            #  if the exception occurred in a try statement with no matching except clause
            #  or with a finally clause."
            tb = frame_or_tb
            frame = tb.tb_frame
            lineno = tb.tb_lineno
            lasti = tb.tb_lasti
        else:
            frame = frame_or_tb
            lineno = frame.f_lineno
            lasti = frame.f_lasti



        code = frame.f_code
        key = (code, id(code), lasti)
        executing_cache = cls._class_local('__executing_cache', {}) # type: Dict[Tuple[types.CodeType, int, int], Any]

        args = executing_cache.get(key)
        if not args:
            node = stmts = decorator = None
            source = cls.for_frame(frame)
            tree = source.tree
            if tree:
                try:
                    stmts = source.statements_at_line(lineno)
                    if stmts:
                        if is_ipython_cell_code(code):
                            decorator, node = find_node_ipython(frame, lasti, stmts, source)
                        else:
                            node_finder = NodeFinder(frame, stmts, tree, lasti, source)
                            node = node_finder.result
                            decorator = node_finder.decorator
                except Exception:
                    if TESTING:
                        raise

                assert stmts is not None
                if node:
                    new_stmts = {statement_containing_node(node)}
                    assert_(new_stmts <= stmts)
                    stmts = new_stmts

            executing_cache[key] = args = source, node, stmts, decorator

        return Executing(frame, *args)

    @classmethod
    def _class_local(cls, name, default):
        # type: (str, T) -> T
        """
        Returns an attribute directly associated with this class
        (as opposed to subclasses), setting default if necessary
        """
        # classes have a mappingproxy preventing us from using setdefault
        result = cls.__dict__.get(name, default)
        setattr(cls, name, result)
        return result

    @cache
    def statements_at_line(self, lineno):
        # type: (int) -> Set[EnhancedAST]
        """
        Returns the statement nodes overlapping the given line.

        Returns at most one statement unless semicolons are present.

        If the `text` attribute is not valid python, meaning
        `tree` is None, returns an empty set.

        Otherwise, `Source.for_frame(frame).statements_at_line(frame.f_lineno)`
        should return at least one statement.
        """

        return {
            statement_containing_node(node)
            for node in
            self._nodes_by_line[lineno]
        }

    def asttext(self):
        # type: () -> ASTText
        """
        Returns an ASTText object for getting the source of specific AST nodes.

        See http://asttokens.readthedocs.io/en/latest/api-index.html
        """
        from asttokens import ASTText  # must be installed separately

        if self._asttext is None:
            self._asttext = ASTText(self.text, tree=self.tree, filename=self.filename)

        return self._asttext

    def asttokens(self):
        # type: () -> ASTTokens
        """
        Returns an ASTTokens object for getting the source of specific AST nodes.

        See http://asttokens.readthedocs.io/en/latest/api-index.html
        """
        import asttokens  # must be installed separately

        if self._asttokens is None:
            if hasattr(asttokens, 'ASTText'):
                self._asttokens = self.asttext().asttokens
            else:  # pragma: no cover
                self._asttokens = asttokens.ASTTokens(self.text, tree=self.tree, filename=self.filename)
        return self._asttokens

    def _asttext_base(self):
        # type: () -> ASTTextBase
        import asttokens  # must be installed separately

        if hasattr(asttokens, 'ASTText'):
            return self.asttext()
        else:  # pragma: no cover
            return self.asttokens()

    @staticmethod
    def decode_source(source):
        # type: (Union[str, bytes]) -> str
        if isinstance(source, bytes):
            encoding = Source.detect_encoding(source)
            return source.decode(encoding)
        else:
            return source

    @staticmethod
    def detect_encoding(source):
        # type: (bytes) -> str
        return detect_encoding(io.BytesIO(source).readline)[0]

    def code_qualname(self, code):
        # type: (types.CodeType) -> str
        """
        Imitates the __qualname__ attribute of functions for code objects.
        Given:

            - A function `func`
            - A frame `frame` for an execution of `func`, meaning:
                `frame.f_code is func.__code__`

        `Source.for_frame(frame).code_qualname(frame.f_code)`
        will be equal to `func.__qualname__`*. Works for Python 2 as well,
        where of course no `__qualname__` attribute exists.

        Falls back to `code.co_name` if there is no appropriate qualname.

        Based on https://github.com/wbolster/qualname

        (* unless `func` is a lambda
        nested inside another lambda on the same line, in which case
        the outer lambda's qualname will be returned for the codes
        of both lambdas)
        """
        assert_(code.co_filename == self.filename)
        return self._qualnames.get((code.co_name, code.co_firstlineno), code.co_name)


class Executing(object):
    """
    Information about the operation a frame is currently executing.

    Generally you will just want `node`, which is the AST node being executed,
    or None if it's unknown.

    If a decorator is currently being called, then:
        - `node` is a function or class definition
        - `decorator` is the expression in `node.decorator_list` being called
        - `statements == {node}`
    """

    def __init__(self, frame, source, node, stmts, decorator):
        # type: (types.FrameType, Source, EnhancedAST, Set[ast.stmt], Optional[EnhancedAST]) -> None
        self.frame = frame
        self.source = source
        self.node = node
        self.statements = stmts
        self.decorator = decorator

    def code_qualname(self):
        # type: () -> str
        return self.source.code_qualname(self.frame.f_code)

    def text(self):
        # type: () -> str
        return self.source._asttext_base().get_text(self.node)

    def text_range(self):
        # type: () -> Tuple[int, int]
        return self.source._asttext_base().get_text_range(self.node)


class QualnameVisitor(ast.NodeVisitor):
    def __init__(self):
        # type: () -> None
        super(QualnameVisitor, self).__init__()
        self.stack = [] # type: List[str]
        self.qualnames = {} # type: Dict[Tuple[str, int], str]

    def add_qualname(self, node, name=None):
        # type: (ast.AST, Optional[str]) -> None
        name = name or node.name # type: ignore[attr-defined]
        self.stack.append(name)
        if getattr(node, 'decorator_list', ()):
            lineno = node.decorator_list[0].lineno # type: ignore[attr-defined]
        else:
            lineno = node.lineno # type: ignore[attr-defined]
        self.qualnames.setdefault((name, lineno), ".".join(self.stack))

    def visit_FunctionDef(self, node, name=None):
        # type: (ast.AST, Optional[str]) -> None
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)), node
        self.add_qualname(node, name)
        self.stack.append('<locals>')
        children = [] # type: Sequence[ast.AST]
        if isinstance(node, ast.Lambda):
            children = [node.body]
        else:
            children = node.body
        for child in children:
            self.visit(child)
        self.stack.pop()
        self.stack.pop()

        # Find lambdas in the function definition outside the body,
        # e.g. decorators or default arguments
        # Based on iter_child_nodes
        for field, child in ast.iter_fields(node):
            if field == 'body':
                continue
            if isinstance(child, ast.AST):
                self.visit(child)
            elif isinstance(child, list):
                for grandchild in child:
                    if isinstance(grandchild, ast.AST):
                        self.visit(grandchild)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        # type: (ast.AST) -> None
        assert isinstance(node, ast.Lambda)
        self.visit_FunctionDef(node, '<lambda>')

    def visit_ClassDef(self, node):
        # type: (ast.AST) -> None
        assert isinstance(node, ast.ClassDef)
        self.add_qualname(node)
        self.generic_visit(node)
        self.stack.pop()





future_flags = sum(
    getattr(__future__, fname).compiler_flag for fname in __future__.all_feature_names
)


def compile_similar_to(source, matching_code):
    # type: (ast.Module, types.CodeType) -> Any
    return compile(
        source,
        matching_code.co_filename,
        'exec',
        flags=future_flags & matching_code.co_flags,
        dont_inherit=True,
    )


sentinel = 'io8urthglkjdghvljusketgIYRFYUVGHFRTBGVHKGF78678957647698'

def is_rewritten_by_pytest(code):
    # type: (types.CodeType) -> bool
    return any(
        bc.opname != "LOAD_CONST" and isinstance(bc.argval,str) and bc.argval.startswith("@py")
        for bc in get_instructions(code)
    )


class SentinelNodeFinder(object):
    result = None # type: EnhancedAST

    def __init__(self, frame, stmts, tree, lasti, source):
        # type: (types.FrameType, Set[EnhancedAST], ast.Module, int, Source) -> None
        assert_(stmts)
        self.frame = frame
        self.tree = tree
        self.code = code = frame.f_code
        self.is_pytest = is_rewritten_by_pytest(code)

        if self.is_pytest:
            self.ignore_linenos = frozenset(assert_linenos(tree))
        else:
            self.ignore_linenos = frozenset()

        self.decorator = None

        self.instruction = instruction = self.get_actual_current_instruction(lasti)
        op_name = instruction.opname
        extra_filter = lambda e: True # type: Callable[[Any], bool]
        ctx = type(None) # type: Type

        typ = type(None) # type: Type
        if op_name.startswith('CALL_'):
            typ = ast.Call
        elif op_name.startswith(('BINARY_SUBSCR', 'SLICE+')):
            typ = ast.Subscript
            ctx = ast.Load
        elif op_name.startswith('BINARY_'):
            typ = ast.BinOp
            op_type = dict(
                BINARY_POWER=ast.Pow,
                BINARY_MULTIPLY=ast.Mult,
                BINARY_MATRIX_MULTIPLY=getattr(ast, "MatMult", ()),
                BINARY_FLOOR_DIVIDE=ast.FloorDiv,
                BINARY_TRUE_DIVIDE=ast.Div,
                BINARY_MODULO=ast.Mod,
                BINARY_ADD=ast.Add,
                BINARY_SUBTRACT=ast.Sub,
                BINARY_LSHIFT=ast.LShift,
                BINARY_RSHIFT=ast.RShift,
                BINARY_AND=ast.BitAnd,
                BINARY_XOR=ast.BitXor,
                BINARY_OR=ast.BitOr,
            )[op_name]
            extra_filter = lambda e: isinstance(e.op, op_type)
        elif op_name.startswith('UNARY_'):
            typ = ast.UnaryOp
            op_type = dict(
                UNARY_POSITIVE=ast.UAdd,
                UNARY_NEGATIVE=ast.USub,
                UNARY_NOT=ast.Not,
                UNARY_INVERT=ast.Invert,
            )[op_name]
            extra_filter = lambda e: isinstance(e.op, op_type)
        elif op_name in ('LOAD_ATTR', 'LOAD_METHOD', 'LOOKUP_METHOD'):
            typ = ast.Attribute
            ctx = ast.Load
            extra_filter = lambda e: attr_names_match(e.attr, instruction.argval)
        elif op_name in ('LOAD_NAME', 'LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_DEREF', 'LOAD_CLASSDEREF'):
            typ = ast.Name
            ctx = ast.Load
            extra_filter = lambda e: e.id == instruction.argval
        elif op_name in ('COMPARE_OP', 'IS_OP', 'CONTAINS_OP'):
            typ = ast.Compare
            extra_filter = lambda e: len(e.ops) == 1
        elif op_name.startswith(('STORE_SLICE', 'STORE_SUBSCR')):
            ctx = ast.Store
            typ = ast.Subscript
        elif op_name.startswith('STORE_ATTR'):
            ctx = ast.Store
            typ = ast.Attribute
            extra_filter = lambda e: attr_names_match(e.attr, instruction.argval)
        else:
            raise RuntimeError(op_name)

        with lock:
            exprs = {
                cast(EnhancedAST, node)
                for stmt in stmts
                for node in ast.walk(stmt)
                if isinstance(node, typ)
                if isinstance(getattr(node, "ctx", None), ctx)
                if extra_filter(node)
                if statement_containing_node(node) == stmt
            }

            if ctx == ast.Store:
                # No special bytecode tricks here.
                # We can handle multiple assigned attributes with different names,
                # but only one assigned subscript.
                self.result = only(exprs)
                return

            matching = list(self.matching_nodes(exprs))
            if not matching and typ == ast.Call:
                self.find_decorator(stmts)
            else:
                self.result = only(matching)

    def find_decorator(self, stmts):
        # type: (Union[List[EnhancedAST], Set[EnhancedAST]]) -> None
        stmt = only(stmts)
        assert_(isinstance(stmt, (ast.ClassDef, function_node_types)))
        decorators = stmt.decorator_list # type: ignore[attr-defined]
        assert_(decorators)
        line_instructions = [
            inst
            for inst in self.clean_instructions(self.code)
            if inst.lineno == self.frame.f_lineno
        ]
        last_decorator_instruction_index = [
            i
            for i, inst in enumerate(line_instructions)
            if inst.opname == "CALL_FUNCTION"
        ][-1]
        assert_(
            line_instructions[last_decorator_instruction_index + 1].opname.startswith(
                "STORE_"
            )
        )
        decorator_instructions = line_instructions[
            last_decorator_instruction_index
            - len(decorators)
            + 1 : last_decorator_instruction_index
            + 1
        ]
        assert_({inst.opname for inst in decorator_instructions} == {"CALL_FUNCTION"})
        decorator_index = decorator_instructions.index(self.instruction)
        decorator = decorators[::-1][decorator_index]
        self.decorator = decorator
        self.result = stmt

    def clean_instructions(self, code):
        # type: (types.CodeType) -> List[EnhancedInstruction]
        return [
            inst
            for inst in get_instructions(code)
            if inst.opname not in ("EXTENDED_ARG", "NOP")
            if inst.lineno not in self.ignore_linenos
        ]

    def get_original_clean_instructions(self):
        # type: () -> List[EnhancedInstruction]
        result = self.clean_instructions(self.code)

        # pypy sometimes (when is not clear)
        # inserts JUMP_IF_NOT_DEBUG instructions in bytecode
        # If they're not present in our compiled instructions,
        # ignore them in the original bytecode
        if not any(
                inst.opname == "JUMP_IF_NOT_DEBUG"
                for inst in self.compile_instructions()
        ):
            result = [
                inst for inst in result
                if inst.opname != "JUMP_IF_NOT_DEBUG"
            ]

        return result

    def matching_nodes(self, exprs):
        # type: (Set[EnhancedAST]) -> Iterator[EnhancedAST]
        original_instructions = self.get_original_clean_instructions()
        original_index = only(
            i
            for i, inst in enumerate(original_instructions)
            if inst == self.instruction
        )
        for expr_index, expr in enumerate(exprs):
            setter = get_setter(expr)
            assert setter is not None
            # noinspection PyArgumentList
            replacement = ast.BinOp(
                left=expr,
                op=ast.Pow(),
                right=ast.Str(s=sentinel),
            )
            ast.fix_missing_locations(replacement)
            setter(replacement)
            try:
                instructions = self.compile_instructions()
            finally:
                setter(expr)

            if sys.version_info >= (3, 10):
                try:
                    handle_jumps(instructions, original_instructions)
                except Exception:
                    # Give other candidates a chance
                    if TESTING or expr_index < len(exprs) - 1:
                        continue
                    raise

            indices = [
                i
                for i, instruction in enumerate(instructions)
                if instruction.argval == sentinel
            ]

            # There can be several indices when the bytecode is duplicated,
            # as happens in a finally block in 3.9+
            # First we remove the opcodes caused by our modifications
            for index_num, sentinel_index in enumerate(indices):
                # Adjustment for removing sentinel instructions below
                # in past iterations
                sentinel_index -= index_num * 2

                assert_(instructions.pop(sentinel_index).opname == 'LOAD_CONST')
                assert_(instructions.pop(sentinel_index).opname == 'BINARY_POWER')

            # Then we see if any of the instruction indices match
            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2
                new_index = sentinel_index - 1

                if new_index != original_index:
                    continue

                original_inst = original_instructions[original_index]
                new_inst = instructions[new_index]

                # In Python 3.9+, changing 'not x in y' to 'not sentinel_transformation(x in y)'
                # changes a CONTAINS_OP(invert=1) to CONTAINS_OP(invert=0),<sentinel stuff>,UNARY_NOT
                if (
                        original_inst.opname == new_inst.opname in ('CONTAINS_OP', 'IS_OP')
                        and original_inst.arg != new_inst.arg # type: ignore[attr-defined]
                        and (
                        original_instructions[original_index + 1].opname
                        != instructions[new_index + 1].opname == 'UNARY_NOT'
                )):
                    # Remove the difference for the upcoming assert
                    instructions.pop(new_index + 1)

                # Check that the modified instructions don't have anything unexpected
                # 3.10 is a bit too weird to assert this in all cases but things still work
                if sys.version_info < (3, 10):
                    for inst1, inst2 in zip_longest(
                        original_instructions, instructions
                    ):
                        assert_(inst1 and inst2 and opnames_match(inst1, inst2))

                yield expr

    def compile_instructions(self):
        # type: () -> List[EnhancedInstruction]
        module_code = compile_similar_to(self.tree, self.code)
        code = only(self.find_codes(module_code))
        return self.clean_instructions(code)

    def find_codes(self, root_code):
        # type: (types.CodeType) -> list
        checks = [
            attrgetter('co_firstlineno'),
            attrgetter('co_freevars'),
            attrgetter('co_cellvars'),
            lambda c: is_ipython_cell_code_name(c.co_name) or c.co_name,
        ] # type: List[Callable]
        if not self.is_pytest:
            checks += [
                attrgetter('co_names'),
                attrgetter('co_varnames'),
            ]

        def matches(c):
            # type: (types.CodeType) -> bool
            return all(
                f(c) == f(self.code)
                for f in checks
            )

        code_options = []
        if matches(root_code):
            code_options.append(root_code)

        def finder(code):
            # type: (types.CodeType) -> None
            for const in code.co_consts:
                if not inspect.iscode(const):
                    continue

                if matches(const):
                    code_options.append(const)
                finder(const)

        finder(root_code)
        return code_options

    def get_actual_current_instruction(self, lasti):
        # type: (int) -> EnhancedInstruction
        """
        Get the instruction corresponding to the current
        frame offset, skipping EXTENDED_ARG instructions
        """
        # Don't use get_original_clean_instructions
        # because we need the actual instructions including
        # EXTENDED_ARG
        instructions = list(get_instructions(self.code))
        index = only(
            i
            for i, inst in enumerate(instructions)
            if inst.offset == lasti
        )

        while True:
            instruction = instructions[index]
            if instruction.opname != "EXTENDED_ARG":
                return instruction
            index += 1



def non_sentinel_instructions(instructions, start):
    # type: (List[EnhancedInstruction], int) -> Iterator[Tuple[int, EnhancedInstruction]]
    """
    Yields (index, instruction) pairs excluding the basic
    instructions introduced by the sentinel transformation
    """
    skip_power = False
    for i, inst in islice(enumerate(instructions), start, None):
        if inst.argval == sentinel:
            assert_(inst.opname == "LOAD_CONST")
            skip_power = True
            continue
        elif skip_power:
            assert_(inst.opname == "BINARY_POWER")
            skip_power = False
            continue
        yield i, inst


def walk_both_instructions(original_instructions, original_start, instructions, start):
    # type: (List[EnhancedInstruction], int, List[EnhancedInstruction], int) -> Iterator[Tuple[int, EnhancedInstruction, int, EnhancedInstruction]]
    """
    Yields matching indices and instructions from the new and original instructions,
    leaving out changes made by the sentinel transformation.
    """
    original_iter = islice(enumerate(original_instructions), original_start, None)
    new_iter = non_sentinel_instructions(instructions, start)
    inverted_comparison = False
    while True:
        try:
            original_i, original_inst = next(original_iter)
            new_i, new_inst = next(new_iter)
        except StopIteration:
            return
        if (
            inverted_comparison
            and original_inst.opname != new_inst.opname == "UNARY_NOT"
        ):
            new_i, new_inst = next(new_iter)
        inverted_comparison = (
            original_inst.opname == new_inst.opname in ("CONTAINS_OP", "IS_OP")
            and original_inst.arg != new_inst.arg # type: ignore[attr-defined]
        )
        yield original_i, original_inst, new_i, new_inst


def handle_jumps(instructions, original_instructions):
    # type: (List[EnhancedInstruction], List[EnhancedInstruction]) -> None
    """
    Transforms instructions in place until it looks more like original_instructions.
    This is only needed in 3.10+ where optimisations lead to more drastic changes
    after the sentinel transformation.
    Replaces JUMP instructions that aren't also present in original_instructions
    with the sections that they jump to until a raise or return.
    In some other cases duplication found in `original_instructions`
    is replicated in `instructions`.
    """
    while True:
        for original_i, original_inst, new_i, new_inst in walk_both_instructions(
            original_instructions, 0, instructions, 0
        ):
            if opnames_match(original_inst, new_inst):
                continue

            if "JUMP" in new_inst.opname and "JUMP" not in original_inst.opname:
                # Find where the new instruction is jumping to, ignoring
                # instructions which have been copied in previous iterations
                start = only(
                    i
                    for i, inst in enumerate(instructions)
                    if inst.offset == new_inst.argval
                    and not getattr(inst, "_copied", False)
                )
                # Replace the jump instruction with the jumped to section of instructions
                # That section may also be deleted if it's not similarly duplicated
                # in original_instructions
                new_instructions = handle_jump(
                    original_instructions, original_i, instructions, start
                )
                assert new_instructions is not None
                instructions[new_i : new_i + 1] = new_instructions            
            else:
                # Extract a section of original_instructions from original_i to return/raise
                orig_section = []
                for section_inst in original_instructions[original_i:]:
                    orig_section.append(section_inst)
                    if section_inst.opname in ("RETURN_VALUE", "RAISE_VARARGS"):
                        break
                else:
                    # No return/raise - this is just a mismatch we can't handle
                    raise AssertionError

                instructions[new_i:new_i] = only(find_new_matching(orig_section, instructions))

            # instructions has been modified, the for loop can't sensibly continue
            # Restart it from the beginning, checking for other issues
            break

        else:  # No mismatched jumps found, we're done
            return


def find_new_matching(orig_section, instructions):
    # type: (List[EnhancedInstruction], List[EnhancedInstruction]) -> Iterator[List[EnhancedInstruction]]
    """
    Yields sections of `instructions` which match `orig_section`.
    The yielded sections include sentinel instructions, but these
    are ignored when checking for matches.
    """
    for start in range(len(instructions) - len(orig_section)):
        indices, dup_section = zip(
            *islice(
                non_sentinel_instructions(instructions, start),
                len(orig_section),
            )
        )
        if len(dup_section) < len(orig_section):
            return
        if sections_match(orig_section, dup_section):
            yield instructions[start:indices[-1] + 1]


def handle_jump(original_instructions, original_start, instructions, start):
    # type: (List[EnhancedInstruction], int, List[EnhancedInstruction], int) -> Optional[List[EnhancedInstruction]]
    """
    Returns the section of instructions starting at `start` and ending
    with a RETURN_VALUE or RAISE_VARARGS instruction.
    There should be a matching section in original_instructions starting at original_start.
    If that section doesn't appear elsewhere in original_instructions,
    then also delete the returned section of instructions.
    """
    for original_j, original_inst, new_j, new_inst in walk_both_instructions(
        original_instructions, original_start, instructions, start
    ):
        assert_(opnames_match(original_inst, new_inst))
        if original_inst.opname in ("RETURN_VALUE", "RAISE_VARARGS"):
            inlined = deepcopy(instructions[start : new_j + 1])
            for inl in inlined:
                inl._copied = True
            orig_section = original_instructions[original_start : original_j + 1]
            if not check_duplicates(
                original_start, orig_section, original_instructions
            ):
                instructions[start : new_j + 1] = []
            return inlined
    
    return None


def check_duplicates(original_i, orig_section, original_instructions):
    # type: (int, List[EnhancedInstruction], List[EnhancedInstruction]) -> bool
    """
    Returns True if a section of original_instructions starting somewhere other
    than original_i and matching orig_section is found, i.e. orig_section is duplicated.
    """
    for dup_start in range(len(original_instructions)):
        if dup_start == original_i:
            continue
        dup_section = original_instructions[dup_start : dup_start + len(orig_section)]
        if len(dup_section) < len(orig_section):
            return False
        if sections_match(orig_section, dup_section):
            return True
    
    return False

def sections_match(orig_section, dup_section):
    # type: (List[EnhancedInstruction], List[EnhancedInstruction]) -> bool
    """
    Returns True if the given lists of instructions have matching linenos and opnames.
    """
    return all(
        (
            orig_inst.lineno == dup_inst.lineno
            # POP_BLOCKs have been found to have differing linenos in innocent cases
            or "POP_BLOCK" == orig_inst.opname == dup_inst.opname
        )
        and opnames_match(orig_inst, dup_inst)
        for orig_inst, dup_inst in zip(orig_section, dup_section)
    )


def opnames_match(inst1, inst2):
    # type: (Instruction, Instruction) -> bool
    return (
        inst1.opname == inst2.opname
        or "JUMP" in inst1.opname
        and "JUMP" in inst2.opname
        or (inst1.opname == "PRINT_EXPR" and inst2.opname == "POP_TOP")
        or (
            inst1.opname in ("LOAD_METHOD", "LOOKUP_METHOD")
            and inst2.opname == "LOAD_ATTR"
        )
        or (inst1.opname == "CALL_METHOD" and inst2.opname == "CALL_FUNCTION")
    )


def get_setter(node):
    # type: (EnhancedAST) -> Optional[Callable[[ast.AST], None]]
    parent = node.parent
    for name, field in ast.iter_fields(parent):
        if field is node:
            def setter(new_node):
                # type: (ast.AST) -> None
                return setattr(parent, name, new_node)
            return setter
        elif isinstance(field, list):
            for i, item in enumerate(field):
                if item is node:
                    def setter(new_node):
                        # type: (ast.AST) -> None
                        field[i] = new_node

                    return setter
    return None

lock = RLock()


@cache
def statement_containing_node(node):
    # type: (ast.AST) -> EnhancedAST
    while not isinstance(node, ast.stmt):
        node = cast(EnhancedAST, node).parent
    return cast(EnhancedAST, node)


def assert_linenos(tree):
    # type: (ast.AST) -> Iterator[int]
    for node in ast.walk(tree):
        if (
                hasattr(node, 'parent') and
                isinstance(statement_containing_node(node), ast.Assert)
        ):
            for lineno in node_linenos(node):
                yield lineno


def _extract_ipython_statement(stmt):
    # type: (EnhancedAST) -> ast.Module
    # IPython separates each statement in a cell to be executed separately
    # So NodeFinder should only compile one statement at a time or it
    # will find a code mismatch.
    while not isinstance(stmt.parent, ast.Module):
        stmt = stmt.parent
    # use `ast.parse` instead of `ast.Module` for better portability
    # python3.8 changes the signature of `ast.Module`
    # Inspired by https://github.com/pallets/werkzeug/pull/1552/files
    tree = ast.parse("")
    tree.body = [cast(ast.stmt, stmt)]
    ast.copy_location(tree, stmt)
    return tree


def is_ipython_cell_code_name(code_name):
    # type: (str) -> bool
    return bool(re.match(r"(<module>|<cell line: \d+>)$", code_name))


def is_ipython_cell_filename(filename):
    # type: (str) -> bool
    return bool(re.search(r"<ipython-input-|[/\\]ipykernel_\d+[/\\]", filename))


def is_ipython_cell_code(code_obj):
    # type: (types.CodeType) -> bool
    return (
        is_ipython_cell_filename(code_obj.co_filename) and
        is_ipython_cell_code_name(code_obj.co_name)
    )


def find_node_ipython(frame, lasti, stmts, source):
    # type: (types.FrameType, int, Set[EnhancedAST], Source) -> Tuple[Optional[Any], Optional[Any]]
    node = decorator = None
    for stmt in stmts:
        tree = _extract_ipython_statement(stmt)
        try:
            node_finder = NodeFinder(frame, stmts, tree, lasti, source)
            if (node or decorator) and (node_finder.result or node_finder.decorator):
                # Found potential nodes in separate statements,
                # cannot resolve ambiguity, give up here
                return None, None

            node = node_finder.result
            decorator = node_finder.decorator
        except Exception:
            pass
    return decorator, node


def attr_names_match(attr, argval):
    # type: (str, str) -> bool
    """
    Checks that the user-visible attr (from ast) can correspond to
    the argval in the bytecode, i.e. the real attribute fetched internally,
    which may be mangled for private attributes.
    """
    if attr == argval:
        return True
    if not attr.startswith("__"):
        return False
    return bool(re.match(r"^_\w+%s$" % attr, argval))


def node_linenos(node):
    # type: (ast.AST) -> Iterator[int]
    if hasattr(node, "lineno"):
        linenos = [] # type: Sequence[int]
        if hasattr(node, "end_lineno") and isinstance(node, ast.expr):
            assert node.end_lineno is not None # type: ignore[attr-defined]
            linenos = range(node.lineno, node.end_lineno + 1) # type: ignore[attr-defined]
        else:
            linenos = [node.lineno] # type: ignore[attr-defined]
        for lineno in linenos:
            yield lineno


if sys.version_info >= (3, 11):
    from ._position_node_finder import PositionNodeFinder as NodeFinder
else:
    NodeFinder = SentinelNodeFinder

