import inspect
import logging
import sys
import traceback
from collections import Counter
from html import escape as escape_html
from types import FrameType, TracebackType
from typing import Union, Iterable, List

from stack_data import (
    style_with_executing_node,
    Options,
    Line,
    FrameInfo,
    Variable,
    RepeatedFrames,
)
from stack_data.utils import some_str

log = logging.getLogger(__name__)


class Serializer:
    def __init__(
        self,
        *,
        options=None,
        pygmented=False,
        show_executing_node=True,
        pygments_formatter_cls=None,
        pygments_formatter_kwargs=None,
        pygments_style="monokai",
        executing_node_modifier="bg:#005080",
        use_code_qualname=True,
        strip_leading_indent=True,
        html=False,
        chain=True,
        collapse_repeated_frames=True,
        show_variables=False,
    ):
        if options is None:
            options = Options()

        if pygmented and not options.pygments_formatter:
            if show_executing_node:
                pygments_style = style_with_executing_node(
                    pygments_style, executing_node_modifier
                )

            if pygments_formatter_cls is None:
                if html:
                    from pygments.formatters.html import (
                        HtmlFormatter as pygments_formatter_cls,
                    )
                else:
                    from pygments.formatters.terminal256 import (
                        Terminal256Formatter as pygments_formatter_cls,
                    )

            options.pygments_formatter = pygments_formatter_cls(
                style=pygments_style,
                **pygments_formatter_kwargs or {},
            )

        self.pygmented = pygmented
        self.use_code_qualname = use_code_qualname
        self.strip_leading_indent = strip_leading_indent
        self.html = html
        self.chain = chain
        self.options = options
        self.collapse_repeated_frames = collapse_repeated_frames
        self.show_variables = show_variables

    def format_exception(self, e=None) -> List[dict]:
        if e is None:
            e = sys.exc_info()[1]

        result = []

        if self.chain:
            if e.__cause__ is not None:
                result = self.format_exception(e.__cause__)
                result[-1]["tail"] = traceback._cause_message.strip()
            elif e.__context__ is not None and not e.__suppress_context__:
                result = self.format_exception(e.__context__)
                result[-1]["tail"] = traceback._context_message.strip()

        result.append(self.format_traceback_part(e))
        return result

    def format_traceback_part(self, e: BaseException) -> dict:
        return dict(
            frames=self.format_stack(e.__traceback__ or sys.exc_info()[2]),
            exception=dict(
                type=type(e).__name__,
                message=some_str(e),
            ),
            tail="",
        )

    def format_stack(self, frame_or_tb=None) -> List[dict]:
        if frame_or_tb is None:
            frame_or_tb = inspect.currentframe().f_back

        return list(
            self.format_stack_data(
                FrameInfo.stack_data(
                    frame_or_tb,
                    self.options,
                    collapse_repeated_frames=self.collapse_repeated_frames,
                )
            )
        )

    def format_stack_data(
        self, stack: Iterable[Union[FrameInfo, RepeatedFrames]]
    ) -> Iterable[dict]:
        for item in stack:
            if isinstance(item, FrameInfo):
                if not self.should_include_frame(item):
                    continue
                yield dict(type="frame", **self.format_frame(item))
            else:
                yield dict(type="repeated_frames", **self.format_repeated_frames(item))

    def format_repeated_frames(self, repeated_frames: RepeatedFrames) -> dict:
        counts = sorted(
            Counter(repeated_frames.frame_keys).items(),
            key=lambda item: (-item[1], item[0][0].co_name),
        )
        return dict(
            frames=[
                dict(
                    name=code.co_name,
                    lineno=lineno,
                    count=count,
                )
                for (code, lineno), count in counts
            ]
        )

    def format_frame(self, frame: Union[FrameInfo, FrameType, TracebackType]) -> dict:
        if not isinstance(frame, FrameInfo):
            frame = FrameInfo(frame, self.options)

        result = dict(
            name=(
                frame.executing.code_qualname()
                if self.use_code_qualname
                else frame.code.co_name
            ),
            filename=frame.filename,
            lineno=frame.lineno,
            lines=list(self.format_lines(frame.lines)),
        )
        if self.show_variables:
            result["variables"] = list(self.format_variables(frame))
        return result

    def format_lines(self, lines):
        for line in lines:
            if isinstance(line, Line):
                yield dict(type="line", **self.format_line(line))
            else:
                yield dict(type="line_gap")

    def format_line(self, line: Line) -> dict:
        return dict(
            is_current=line.is_current,
            lineno=line.lineno,
            text=line.render(
                pygmented=self.pygmented,
                escape_html=self.html,
                strip_leading_indent=self.strip_leading_indent,
            ),
        )

    def format_variables(self, frame_info: FrameInfo) -> Iterable[dict]:
        try:
            for var in sorted(frame_info.variables, key=lambda v: v.name):
                yield self.format_variable(var)
        except Exception:  # pragma: no cover
            log.exception("Error in getting frame variables")

    def format_variable(self, var: Variable) -> dict:
        return dict(
            name=self.format_variable_part(var.name),
            value=self.format_variable_part(self.format_variable_value(var.value)),
        )

    def format_variable_part(self, text):
        if self.html:
            return escape_html(text)
        else:
            return text

    def format_variable_value(self, value) -> str:
        return repr(value)

    def should_include_frame(self, frame_info: FrameInfo) -> bool:
        return True  # pragma: no cover
