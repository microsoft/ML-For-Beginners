import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
    Iterator, List, Tuple, Optional, NamedTuple,
    Any, Iterable, Callable, Union,
    Sequence)
from typing import Mapping

import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
    truncate, unique_in_order, line_range,
    frame_and_lineno, iter_stack, collapse_repeated, group_by_key_func,
    cached_property, is_frame, _pygmented_with_ranges, assert_)

RangeInLine = NamedTuple('RangeInLine',
                         [('start', int),
                          ('end', int),
                          ('data', Any)])
RangeInLine.__doc__ = """
Represents a range of characters within one line of source code,
and some associated data.

Typically this will be converted to a pair of markers by markers_from_ranges.
"""

MarkerInLine = NamedTuple('MarkerInLine',
                          [('position', int),
                           ('is_start', bool),
                           ('string', str)])
MarkerInLine.__doc__ = """
A string that is meant to be inserted at a given position in a line of source code.
For example, this could be an ANSI code or the opening or closing of an HTML tag.
is_start should be True if this is the first of a pair such as the opening of an HTML tag.
This will help to sort and insert markers correctly.

Typically this would be created from a RangeInLine by markers_from_ranges.
Then use Line.render to insert the markers correctly.
"""


class BlankLines(Enum):
    """The values are intended to correspond to the following behaviour:
    HIDDEN: blank lines are not shown in the output
    VISIBLE: blank lines are visible in the output
    SINGLE: any consecutive blank lines are shown as a single blank line
            in the output. This option requires the line number to be shown.
            For a single blank line, the corresponding line number is shown.
            Two or more consecutive blank lines are shown as a single blank
            line in the output with a custom string shown instead of a
            specific line number.
    """
    HIDDEN = 1
    VISIBLE = 2
    SINGLE=3

class Variable(
    NamedTuple('_Variable',
               [('name', str),
                ('nodes', Sequence[ast.AST]),
                ('value', Any)])
):
    """
    An expression that appears one or more times in source code and its associated value.
    This will usually be a variable but it can be any expression evaluated by pure_eval.
    - name is the source text of the expression.
    - nodes is a list of equivalent nodes representing the same expression.
    - value is the safely evaluated value of the expression.
    """
    __hash__ = object.__hash__
    __eq__ = object.__eq__


class Source(executing.Source):
    """
    The source code of a single file and associated metadata.

    In addition to the attributes from the base class executing.Source,
    if .tree is not None, meaning this is valid Python code, objects have:
        - pieces: a list of Piece objects
        - tokens_by_lineno: a defaultdict(list) mapping line numbers to lists of tokens.

    Don't construct this class. Get an instance from frame_info.source.
    """

    @cached_property
    def pieces(self) -> List[range]:
        if not self.tree:
            return [
                range(i, i + 1)
                for i in range(1, len(self.lines) + 1)
            ]
        return list(self._clean_pieces())

    @cached_property
    def tokens_by_lineno(self) -> Mapping[int, List[Token]]:
        if not self.tree:
            raise AttributeError("This file doesn't contain valid Python, so .tokens_by_lineno doesn't exist")
        return group_by_key_func(
            self.asttokens().tokens,
            lambda tok: tok.start[0],
        )

    def _clean_pieces(self) -> Iterator[range]:
        pieces = self._raw_split_into_pieces(self.tree, 1, len(self.lines) + 1)
        pieces = [
            (start, end)
            for (start, end) in pieces
            if end > start
        ]

        # Combine overlapping pieces, i.e. consecutive pieces where the end of the first
        # is greater than the start of the second.
        # This can happen when two statements are on the same line separated by a semicolon.
        new_pieces = pieces[:1]
        for (start, end) in pieces[1:]:
            (last_start, last_end) = new_pieces[-1]
            if start < last_end:
                assert start == last_end - 1
                assert ';' in self.lines[start - 1]
                new_pieces[-1] = (last_start, end)
            else:
                new_pieces.append((start, end))
        pieces = new_pieces

        starts = [start for start, end in pieces[1:]]
        ends = [end for start, end in pieces[:-1]]
        if starts != ends:
            joins = list(map(set, zip(starts, ends)))
            mismatches = [s for s in joins if len(s) > 1]
            raise AssertionError("Pieces mismatches: %s" % mismatches)

        def is_blank(i):
            try:
                return not self.lines[i - 1].strip()
            except IndexError:
                return False

        for start, end in pieces:
            while is_blank(start):
                start += 1
            while is_blank(end - 1):
                end -= 1
            if start < end:
                yield range(start, end)

    def _raw_split_into_pieces(
            self,
            stmt: ast.AST,
            start: int,
            end: int,
    ) -> Iterator[Tuple[int, int]]:
        for name, body in ast.iter_fields(stmt):
            if (
                    isinstance(body, list) and body and
                    isinstance(body[0], (ast.stmt, ast.ExceptHandler, getattr(ast, 'match_case', ())))
            ):
                for rang, group in sorted(group_by_key_func(body, self.line_range).items()):
                    sub_stmt = group[0]
                    for inner_start, inner_end in self._raw_split_into_pieces(sub_stmt, *rang):
                        if start < inner_start:
                            yield start, inner_start
                        if inner_start < inner_end:
                            yield inner_start, inner_end
                        start = inner_end

        yield start, end

    def line_range(self, node: ast.AST) -> Tuple[int, int]:
        return line_range(self.asttext(), node)


class Options:
    """
    Configuration for FrameInfo, either in the constructor or the .stack_data classmethod.
    These all determine which Lines and gaps are produced by FrameInfo.lines. 

    before and after are the number of pieces of context to include in a frame
    in addition to the executing piece.

    include_signature is whether to include the function signature as a piece in a frame.

    If a piece (other than the executing piece) has more than max_lines_per_piece lines,
    it will be truncated with a gap in the middle. 
    """
    def __init__(
            self, *,
            before: int = 3,
            after: int = 1,
            include_signature: bool = False,
            max_lines_per_piece: int = 6,
            pygments_formatter=None,
            blank_lines = BlankLines.HIDDEN
    ):
        self.before = before
        self.after = after
        self.include_signature = include_signature
        self.max_lines_per_piece = max_lines_per_piece
        self.pygments_formatter = pygments_formatter
        self.blank_lines = blank_lines

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))


class LineGap(object):
    """
    A singleton representing one or more lines of source code that were skipped
    in FrameInfo.lines.

    LINE_GAP can be created in two ways:
    - by truncating a piece of context that's too long.
    - immediately after the signature piece if Options.include_signature is true
      and the following piece isn't already part of the included pieces. 
    """
    def __repr__(self):
        return "LINE_GAP"


LINE_GAP = LineGap()


class BlankLineRange:
    """
    Records the line number range for blank lines gaps between pieces.
    For a single blank line, begin_lineno == end_lineno.
    """
    def __init__(self, begin_lineno: int, end_lineno: int):
        self.begin_lineno = begin_lineno
        self.end_lineno = end_lineno


class Line(object):
    """
    A single line of source code for a particular stack frame.

    Typically this is obtained from FrameInfo.lines.
    Since that list may also contain LINE_GAP, you should first check
    that this is really a Line before using it.

    Attributes:
        - frame_info
        - lineno: the 1-based line number within the file
        - text: the raw source of this line. For displaying text, see .render() instead.
        - leading_indent: the number of leading spaces that should probably be stripped.
            This attribute is set within FrameInfo.lines. If you construct this class
            directly you should probably set it manually (at least to 0).
        - is_current: whether this is the line currently being executed by the interpreter
            within this frame.
        - tokens: a list of source tokens in this line

    There are several helpers for constructing RangeInLines which can be converted to markers
    using markers_from_ranges which can be passed to .render():
        - token_ranges
        - variable_ranges
        - executing_node_ranges
        - range_from_node
    """
    def __init__(
            self,
            frame_info: 'FrameInfo',
            lineno: int,
    ):
        self.frame_info = frame_info
        self.lineno = lineno
        self.text = frame_info.source.lines[lineno - 1]  # type: str
        self.leading_indent = None  # type: Optional[int]

    def __repr__(self):
        return "<{self.__class__.__name__} {self.lineno} (current={self.is_current}) " \
               "{self.text!r} of {self.frame_info.filename}>".format(self=self)

    @property
    def is_current(self) -> bool:
        """
        Whether this is the line currently being executed by the interpreter
        within this frame.
        """
        return self.lineno == self.frame_info.lineno

    @property
    def tokens(self) -> List[Token]:
        """
        A list of source tokens in this line.
        The tokens are Token objects from asttokens:
        https://asttokens.readthedocs.io/en/latest/api-index.html#asttokens.util.Token
        """
        return self.frame_info.source.tokens_by_lineno[self.lineno]

    @cached_property
    def token_ranges(self) -> List[RangeInLine]:
        """
        A list of RangeInLines for each token in .tokens,
        where range.data is a Token object from asttokens:
        https://asttokens.readthedocs.io/en/latest/api-index.html#asttokens.util.Token
        """
        return [
            RangeInLine(
                token.start[1],
                token.end[1],
                token,
            )
            for token in self.tokens
        ]

    @cached_property
    def variable_ranges(self) -> List[RangeInLine]:
        """
        A list of RangeInLines for each Variable that appears at least partially in this line.
        The data attribute of the range is a pair (variable, node) where node is the particular
        AST node from the list variable.nodes that corresponds to this range.
        """
        return [
            self.range_from_node(node, (variable, node))
            for variable, node in self.frame_info.variables_by_lineno[self.lineno]
        ]

    @cached_property
    def executing_node_ranges(self) -> List[RangeInLine]:
        """
        A list of one or zero RangeInLines for the executing node of this frame.
        The list will have one element if the node can be found and it overlaps this line.
        """
        return self._raw_executing_node_ranges(
            self.frame_info._executing_node_common_indent
        )

    def _raw_executing_node_ranges(self, common_indent=0) -> List[RangeInLine]:
        ex = self.frame_info.executing
        node = ex.node
        if node:
            rang = self.range_from_node(node, ex, common_indent)
            if rang:
                return [rang]
        return []

    def range_from_node(
        self, node: ast.AST, data: Any, common_indent: int = 0
    ) -> Optional[RangeInLine]:
        """
        If the given node overlaps with this line, return a RangeInLine
        with the correct start and end and the given data.
        Otherwise, return None.
        """
        atext = self.frame_info.source.asttext()
        (start, range_start), (end, range_end) = atext.get_text_positions(node, padded=False)

        if not (start <= self.lineno <= end):
            return None

        if start != self.lineno:
            range_start = common_indent

        if end != self.lineno:
            range_end = len(self.text)

        if range_start == range_end == 0:
            # This is an empty line. If it were included, it would result
            # in a value of zero for the common indentation assigned to
            # a block of code.
            return None

        return RangeInLine(range_start, range_end, data)

    def render(
            self,
            markers: Iterable[MarkerInLine] = (),
            *,
            strip_leading_indent: bool = True,
            pygmented: bool = False,
            escape_html: bool = False
    ) -> str:
        """
        Produces a string for display consisting of .text
        with the .strings of each marker inserted at the correct positions.
        If strip_leading_indent is true (the default) then leading spaces
        common to all lines in this frame will be excluded.
        """
        if pygmented and self.frame_info.scope:
            assert_(not markers, ValueError("Cannot use pygmented with markers"))
            start_line, lines = self.frame_info._pygmented_scope_lines
            result = lines[self.lineno - start_line]
            if strip_leading_indent:
                result = result.replace(self.text[:self.leading_indent], "", 1)
            return result

        text = self.text

        # This just makes the loop below simpler
        markers = list(markers) + [MarkerInLine(position=len(text), is_start=False, string='')]

        markers.sort(key=lambda t: t[:2])

        parts = []
        if strip_leading_indent:
            start = self.leading_indent
        else:
            start = 0
        original_start = start

        for marker in markers:
            text_part = text[start:marker.position]
            if escape_html:
                text_part = html.escape(text_part)
            parts.append(text_part)
            parts.append(marker.string)

            # Ensure that start >= leading_indent
            start = max(marker.position, original_start)
        return ''.join(parts)


def markers_from_ranges(
        ranges: Iterable[RangeInLine],
        converter: Callable[[RangeInLine], Optional[Tuple[str, str]]],
) -> List[MarkerInLine]:
    """
    Helper to create MarkerInLines given some RangeInLines.
    converter should be a function accepting a RangeInLine returning
    either None (which is ignored) or a pair of strings which
    are used to create two markers included in the returned list.
    """
    markers = []
    for rang in ranges:
        converted = converter(rang)
        if converted is None:
            continue

        start_string, end_string = converted
        if not (isinstance(start_string, str) and isinstance(end_string, str)):
            raise TypeError("converter should return None or a pair of strings")

        markers += [
            MarkerInLine(position=rang.start, is_start=True, string=start_string),
            MarkerInLine(position=rang.end, is_start=False, string=end_string),
        ]
    return markers


def style_with_executing_node(style, modifier):
    from pygments.styles import get_style_by_name
    if isinstance(style, str):
        style = get_style_by_name(style)

    class NewStyle(style):
        for_executing_node = True

        styles = {
            **style.styles,
            **{
                k.ExecutingNode: v + " " + modifier
                for k, v in style.styles.items()
            }
        }

    return NewStyle


class RepeatedFrames:
    """
    A sequence of consecutive stack frames which shouldn't be displayed because
    the same code and line number were repeated many times in the stack, e.g.
    because of deep recursion.

    Attributes:
        - frames: list of raw frame or traceback objects
        - frame_keys: list of tuples (frame.f_code, lineno) extracted from the frame objects.
                        It's this information from the frames that is used to determine
                        whether two frames should be considered similar (i.e. repeating).
        - description: A string briefly describing frame_keys
    """
    def __init__(
            self,
            frames: List[Union[FrameType, TracebackType]],
            frame_keys: List[Tuple[CodeType, int]],
    ):
        self.frames = frames
        self.frame_keys = frame_keys

    @cached_property
    def description(self) -> str:
        """
        A string briefly describing the repeated frames, e.g.
            my_function at line 10 (100 times)
        """
        counts = sorted(Counter(self.frame_keys).items(),
                        key=lambda item: (-item[1], item[0][0].co_name))
        return ', '.join(
            '{name} at line {lineno} ({count} times)'.format(
                name=Source.for_filename(code.co_filename).code_qualname(code),
                lineno=lineno,
                count=count,
            )
            for (code, lineno), count in counts
        )

    def __repr__(self):
        return '<{self.__class__.__name__} {self.description}>'.format(self=self)


class FrameInfo(object):
    """
    Information about a frame!
    Pass either a frame object or a traceback object,
    and optionally an Options object to configure.

    Or use the classmethod FrameInfo.stack_data() for an iterator of FrameInfo and
    RepeatedFrames objects. 

    Attributes:
        - frame: an actual stack frame object, either frame_or_tb or frame_or_tb.tb_frame
        - options
        - code: frame.f_code
        - source: a Source object
        - filename: a hopefully absolute file path derived from code.co_filename
        - scope: the AST node of the innermost function, class or module being executed
        - lines: a list of Line/LineGap objects to display, determined by options
        - executing: an Executing object from the `executing` library, which has:
            - .node: the AST node being executed in this frame, or None if it's unknown
            - .statements: a set of one or more candidate statements (AST nodes, probably just one)
                currently being executed in this frame.
            - .code_qualname(): the __qualname__ of the function or class being executed,
                or just the code name.

    Properties returning one or more pieces of source code (ranges of lines):
        - scope_pieces: all the pieces in the scope
        - included_pieces: a subset of scope_pieces determined by options
        - executing_piece: the piece currently being executed in this frame

    Properties returning lists of Variable objects:
        - variables: all variables in the scope
        - variables_by_lineno: variables organised into lines
        - variables_in_lines: variables contained within FrameInfo.lines
        - variables_in_executing_piece: variables contained within FrameInfo.executing_piece
    """
    def __init__(
            self,
            frame_or_tb: Union[FrameType, TracebackType],
            options: Optional[Options] = None,
    ):
        self.executing = Source.executing(frame_or_tb)
        frame, self.lineno = frame_and_lineno(frame_or_tb)
        self.frame = frame
        self.code = frame.f_code
        self.options = options or Options()  # type: Options
        self.source = self.executing.source  # type: Source


    def __repr__(self):
        return "{self.__class__.__name__}({self.frame})".format(self=self)

    @classmethod
    def stack_data(
            cls,
            frame_or_tb: Union[FrameType, TracebackType],
            options: Optional[Options] = None,
            *,
            collapse_repeated_frames: bool = True
    ) -> Iterator[Union['FrameInfo', RepeatedFrames]]:
        """
        An iterator of FrameInfo and RepeatedFrames objects representing
        a full traceback or stack. Similar consecutive frames are collapsed into RepeatedFrames
        objects, so always check what type of object has been yielded.

        Pass either a frame object or a traceback object,
        and optionally an Options object to configure.
        """
        stack = list(iter_stack(frame_or_tb))

        # Reverse the stack from a frame so that it's in the same order
        # as the order from a traceback, which is the order of a printed
        # traceback when read top to bottom (most recent call last)
        if is_frame(frame_or_tb):
            stack = stack[::-1]

        def mapper(f):
            return cls(f, options)

        if not collapse_repeated_frames:
            yield from map(mapper, stack)
            return

        def _frame_key(x):
            frame, lineno = frame_and_lineno(x)
            return frame.f_code, lineno

        yield from collapse_repeated(
            stack,
            mapper=mapper,
            collapser=RepeatedFrames,
            key=_frame_key,
        )

    @cached_property
    def scope_pieces(self) -> List[range]:
        """
        All the pieces (ranges of lines) contained in this object's .scope,
        unless there is no .scope (because the source isn't valid Python syntax)
        in which case it returns all the pieces in the source file, each containing one line.
        """
        if not self.scope:
            return self.source.pieces

        scope_start, scope_end = self.source.line_range(self.scope)
        return [
            piece
            for piece in self.source.pieces
            if scope_start <= piece.start and piece.stop <= scope_end
        ]

    @cached_property
    def filename(self) -> str:
        """
        A hopefully absolute file path derived from .code.co_filename,
        the current working directory, and sys.path.
        Code based on ipython.
        """
        result = self.code.co_filename

        if (
                os.path.isabs(result) or
                (
                        result.startswith("<") and
                        result.endswith(">")
                )
        ):
            return result

        # Try to make the filename absolute by trying all
        # sys.path entries (which is also what linecache does)
        # as well as the current working directory
        for dirname in ["."] + list(sys.path):
            try:
                fullname = os.path.join(dirname, result)
                if os.path.isfile(fullname):
                    return os.path.abspath(fullname)
            except Exception:
                # Just in case that sys.path contains very
                # strange entries...
                pass

        return result

    @cached_property
    def executing_piece(self) -> range:
        """
        The piece (range of lines) containing the line currently being executed
        by the interpreter in this frame.
        """
        return only(
            piece
            for piece in self.scope_pieces
            if self.lineno in piece
        )

    @cached_property
    def included_pieces(self) -> List[range]:
        """
        The list of pieces (ranges of lines) to display for this frame.
        Consists of .executing_piece, surrounding context pieces
        determined by .options.before and .options.after,
        and the function signature if a function is being executed and
        .options.include_signature is True (in which case this might not
        be a contiguous range of pieces).
        Always a subset of .scope_pieces.
        """
        scope_pieces = self.scope_pieces
        if not self.scope_pieces:
            return []

        pos = scope_pieces.index(self.executing_piece)
        pieces_start = max(0, pos - self.options.before)
        pieces_end = pos + 1 + self.options.after
        pieces = scope_pieces[pieces_start:pieces_end]

        if (
                self.options.include_signature
                and not self.code.co_name.startswith('<')
                and isinstance(self.scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                and pieces_start > 0
        ):
            pieces.insert(0, scope_pieces[0])

        return pieces

    @cached_property
    def _executing_node_common_indent(self) -> int:
        """
        The common minimal indentation shared by the markers intended
        for an exception node that spans multiple lines.

        Intended to be used only internally.
        """
        indents = []
        lines = [line for line in self.lines if isinstance(line, Line)]

        for line in lines:
            for rang in line._raw_executing_node_ranges():
                begin_text = len(line.text) - len(line.text.lstrip())
                indent = max(rang.start, begin_text)
                indents.append(indent)

        if len(indents) <= 1:
            return 0

        return min(indents[1:])

    @cached_property
    def lines(self) -> List[Union[Line, LineGap, BlankLineRange]]:
        """
        A list of lines to display, determined by options.
        The objects yielded either have type Line, BlankLineRange
        or are the singleton LINE_GAP.
        Always check the type that you're dealing with when iterating.

        LINE_GAP can be created in two ways:
            - by truncating a piece of context that's too long, determined by
                .options.max_lines_per_piece
            - immediately after the signature piece if Options.include_signature is true
              and the following piece isn't already part of the included pieces.

        The Line objects are all within the ranges from .included_pieces.
        """
        pieces = self.included_pieces
        if not pieces:
            return []

        add_empty_lines = self.options.blank_lines in (BlankLines.VISIBLE, BlankLines.SINGLE)
        prev_piece = None
        result = []
        for i, piece in enumerate(pieces):
            if (
                    i == 1
                    and self.scope
                    and pieces[0] == self.scope_pieces[0]
                    and pieces[1] != self.scope_pieces[1]
            ):
                result.append(LINE_GAP)
            elif prev_piece and add_empty_lines and piece.start > prev_piece.stop:
                if self.options.blank_lines == BlankLines.SINGLE:
                    result.append(BlankLineRange(prev_piece.stop, piece.start-1))
                else:  # BlankLines.VISIBLE
                    for lineno in range(prev_piece.stop, piece.start):
                        result.append(Line(self, lineno))

            lines = [Line(self, i) for i in piece]  # type: List[Line]
            if piece != self.executing_piece:
                lines = truncate(
                    lines,
                    max_length=self.options.max_lines_per_piece,
                    middle=[LINE_GAP],
                )
            result.extend(lines)
            prev_piece = piece

        real_lines = [
            line
            for line in result
            if isinstance(line, Line)
        ]

        text = "\n".join(
            line.text
            for line in real_lines
        )
        dedented_lines = dedent(text).splitlines()
        leading_indent = len(real_lines[0].text) - len(dedented_lines[0])
        for line in real_lines:
            line.leading_indent = leading_indent
        return result

    @cached_property
    def scope(self) -> Optional[ast.AST]:
        """
        The AST node of the innermost function, class or module being executed.
        """
        if not self.source.tree or not self.executing.statements:
            return None

        stmt = list(self.executing.statements)[0]
        while True:
            # Get the parent first in case the original statement is already
            # a function definition, e.g. if we're calling a decorator
            # In that case we still want the surrounding scope, not that function
            stmt = stmt.parent
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                return stmt

    @cached_property
    def _pygmented_scope_lines(self) -> Optional[Tuple[int, List[str]]]:
        # noinspection PyUnresolvedReferences
        from pygments.formatters import HtmlFormatter

        formatter = self.options.pygments_formatter
        scope = self.scope
        assert_(formatter, ValueError("Must set a pygments formatter in Options"))
        assert_(scope)

        if isinstance(formatter, HtmlFormatter):
            formatter.nowrap = True

        atext = self.source.asttext()
        node = self.executing.node
        if node and getattr(formatter.style, "for_executing_node", False):
            scope_start = atext.get_text_range(scope)[0]
            start, end = atext.get_text_range(node)
            start -= scope_start
            end -= scope_start
            ranges = [(start, end)]
        else:
            ranges = []

        code = atext.get_text(scope)
        lines = _pygmented_with_ranges(formatter, code, ranges)

        start_line = self.source.line_range(scope)[0]

        return start_line, lines

    @cached_property
    def variables(self) -> List[Variable]:
        """
        All Variable objects whose nodes are contained within .scope
        and whose values could be safely evaluated by pure_eval.
        """
        if not self.scope:
            return []

        evaluator = Evaluator.from_frame(self.frame)
        scope = self.scope
        node_values = [
            pair
            for pair in evaluator.find_expressions(scope)
            if is_expression_interesting(*pair)
        ]  # type: List[Tuple[ast.AST, Any]]

        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(scope.args):
                if not isinstance(node, ast.arg):
                    continue
                name = node.arg
                try:
                    value = evaluator.names[name]
                except KeyError:
                    pass
                else:
                    node_values.append((node, value))

        # Group equivalent nodes together
        def get_text(n):
            if isinstance(n, ast.arg):
                return n.arg
            else:
                return self.source.asttext().get_text(n)

        def normalise_node(n):
            try:
                # Add parens to avoid syntax errors for multiline expressions
                return ast.parse('(' + get_text(n) + ')')
            except Exception:
                return n

        grouped = group_by_key_func(
            node_values,
            lambda nv: ast.dump(normalise_node(nv[0])),
        )

        result = []
        for group in grouped.values():
            nodes, values = zip(*group)
            value = values[0]
            text = get_text(nodes[0])
            if not text:
                continue
            result.append(Variable(text, nodes, value))

        return result

    @cached_property
    def variables_by_lineno(self) -> Mapping[int, List[Tuple[Variable, ast.AST]]]:
        """
        A mapping from 1-based line numbers to lists of pairs:
            - A Variable object
            - A specific AST node from the variable's .nodes list that's
                in the line at that line number.
        """
        result = defaultdict(list)
        for var in self.variables:
            for node in var.nodes:
                for lineno in range(*self.source.line_range(node)):
                    result[lineno].append((var, node))
        return result

    @cached_property
    def variables_in_lines(self) -> List[Variable]:
        """
        A list of Variable objects contained within the lines returned by .lines.
        """
        return unique_in_order(
            var
            for line in self.lines
            if isinstance(line, Line)
            for var, node in self.variables_by_lineno[line.lineno]
        )

    @cached_property
    def variables_in_executing_piece(self) -> List[Variable]:
        """
        A list of Variable objects contained within the lines
        in the range returned by .executing_piece.
        """
        return unique_in_order(
            var
            for lineno in self.executing_piece
            for var, node in self.variables_by_lineno[lineno]
        )
