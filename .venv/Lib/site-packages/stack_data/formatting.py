import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable

from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
                       Variable, RepeatedFrames, BlankLineRange, BlankLines)
from stack_data.utils import assert_


class Formatter:
    def __init__(
            self, *,
            options=None,
            pygmented=False,
            show_executing_node=True,
            pygments_formatter_cls=None,
            pygments_formatter_kwargs=None,
            pygments_style="monokai",
            executing_node_modifier="bg:#005080",
            executing_node_underline="^",
            current_line_indicator="-->",
            line_gap_string="(...)",
            line_number_gap_string=":",
            line_number_format_string="{:4} | ",
            show_variables=False,
            use_code_qualname=True,
            show_linenos=True,
            strip_leading_indent=True,
            html=False,
            chain=True,
            collapse_repeated_frames=True
    ):
        if options is None:
            options = Options()

        if pygmented and not options.pygments_formatter:
            if show_executing_node:
                pygments_style = style_with_executing_node(
                    pygments_style, executing_node_modifier
                )

            if pygments_formatter_cls is None:
                from pygments.formatters.terminal256 import Terminal256Formatter \
                    as pygments_formatter_cls

            options.pygments_formatter = pygments_formatter_cls(
                style=pygments_style,
                **pygments_formatter_kwargs or {},
            )

        self.pygmented = pygmented
        self.show_executing_node = show_executing_node
        assert_(
            len(executing_node_underline) == 1,
            ValueError("executing_node_underline must be a single character"),
        )
        self.executing_node_underline = executing_node_underline
        self.current_line_indicator = current_line_indicator or ""
        self.line_gap_string = line_gap_string
        self.line_number_gap_string = line_number_gap_string
        self.line_number_format_string = line_number_format_string
        self.show_variables = show_variables
        self.show_linenos = show_linenos
        self.use_code_qualname = use_code_qualname
        self.strip_leading_indent = strip_leading_indent
        self.html = html
        self.chain = chain
        self.options = options
        self.collapse_repeated_frames = collapse_repeated_frames
        if not self.show_linenos and self.options.blank_lines == BlankLines.SINGLE:
            raise ValueError(
                "BlankLines.SINGLE option can only be used when show_linenos=True"
            )

    def set_hook(self):
        def excepthook(_etype, evalue, _tb):
            self.print_exception(evalue)

        sys.excepthook = excepthook

    def print_exception(self, e=None, *, file=None):
        self.print_lines(self.format_exception(e), file=file)

    def print_stack(self, frame_or_tb=None, *, file=None):
        if frame_or_tb is None:
            frame_or_tb = inspect.currentframe().f_back

        self.print_lines(self.format_stack(frame_or_tb), file=file)

    def print_lines(self, lines, *, file=None):
        if file is None:
            file = sys.stderr
        for line in lines:
            print(line, file=file, end="")

    def format_exception(self, e=None) -> Iterable[str]:
        if e is None:
            e = sys.exc_info()[1]

        if self.chain:
            if e.__cause__ is not None:
                yield from self.format_exception(e.__cause__)
                yield traceback._cause_message
            elif (e.__context__ is not None
                  and not e.__suppress_context__):
                yield from self.format_exception(e.__context__)
                yield traceback._context_message

        yield 'Traceback (most recent call last):\n'
        yield from self.format_stack(e.__traceback__)
        yield from traceback.format_exception_only(type(e), e)

    def format_stack(self, frame_or_tb=None) -> Iterable[str]:
        if frame_or_tb is None:
            frame_or_tb = inspect.currentframe().f_back

        yield from self.format_stack_data(
            FrameInfo.stack_data(
                frame_or_tb,
                self.options,
                collapse_repeated_frames=self.collapse_repeated_frames,
            )
        )

    def format_stack_data(
            self, stack: Iterable[Union[FrameInfo, RepeatedFrames]]
    ) -> Iterable[str]:
        for item in stack:
            if isinstance(item, FrameInfo):
                yield from self.format_frame(item)
            else:
                yield self.format_repeated_frames(item)

    def format_repeated_frames(self, repeated_frames: RepeatedFrames) -> str:
        return '    [... skipping similar frames: {}]\n'.format(
            repeated_frames.description
        )

    def format_frame(self, frame: Union[FrameInfo, FrameType, TracebackType]) -> Iterable[str]:
        if not isinstance(frame, FrameInfo):
            frame = FrameInfo(frame, self.options)

        yield self.format_frame_header(frame)

        for line in frame.lines:
            if isinstance(line, Line):
                yield self.format_line(line)
            elif isinstance(line, BlankLineRange):
                yield self.format_blank_lines_linenumbers(line)
            else:
                assert_(line is LINE_GAP)
                yield self.line_gap_string + "\n"

        if self.show_variables:
            try:
                yield from self.format_variables(frame)
            except Exception:
                pass

    def format_frame_header(self, frame_info: FrameInfo) -> str:
        return ' File "{frame_info.filename}", line {frame_info.lineno}, in {name}\n'.format(
            frame_info=frame_info,
            name=(
                frame_info.executing.code_qualname()
                if self.use_code_qualname else
                frame_info.code.co_name
            ),
        )

    def format_line(self, line: Line) -> str:
        result = ""
        if self.current_line_indicator:
            if line.is_current:
                result = self.current_line_indicator
            else:
                result = " " * len(self.current_line_indicator)
            result += " "
        else:
            result = "   "

        if self.show_linenos:
            result += self.line_number_format_string.format(line.lineno)

        prefix = result

        result += line.render(
            pygmented=self.pygmented,
            escape_html=self.html,
            strip_leading_indent=self.strip_leading_indent,
        ) + "\n"

        if self.show_executing_node and not self.pygmented:
            for line_range in line.executing_node_ranges:
                start = line_range.start - line.leading_indent
                end = line_range.end - line.leading_indent
                # if end <= start, we have an empty line inside a highlighted
                # block of code. In this case, we need to avoid inserting
                # an extra blank line with no markers present.
                if end > start:
                    result += (
                            " " * (start + len(prefix))
                            + self.executing_node_underline * (end - start)
                            + "\n"
                    )
        return result


    def format_blank_lines_linenumbers(self, blank_line):
        if self.current_line_indicator:
            result = " " * len(self.current_line_indicator) + " "
        else:
            result = "   "
        if blank_line.begin_lineno == blank_line.end_lineno:
            return result + self.line_number_format_string.format(blank_line.begin_lineno) + "\n"
        return result + "   {}\n".format(self.line_number_gap_string)


    def format_variables(self, frame_info: FrameInfo) -> Iterable[str]:
        for var in sorted(frame_info.variables, key=lambda v: v.name):
            try:
                yield self.format_variable(var) + "\n"
            except Exception:
                pass

    def format_variable(self, var: Variable) -> str:
        return "{} = {}".format(
            var.name,
            self.format_variable_value(var.value),
        )

    def format_variable_value(self, value) -> str:
        return repr(value)
