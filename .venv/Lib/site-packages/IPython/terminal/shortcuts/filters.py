"""
Filters restricting scope of IPython Terminal shortcuts.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import ast
import re
import signal
import sys
from typing import Callable, Dict, Union

from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.filters import Condition, Filter, emacs_insert_mode, has_completions
from prompt_toolkit.filters import has_focus as has_focus_impl
from prompt_toolkit.filters import (
    Always,
    Never,
    has_selection,
    has_suggestion,
    vi_insert_mode,
    vi_mode,
)
from prompt_toolkit.layout.layout import FocusableElement

from IPython.core.getipython import get_ipython
from IPython.core.guarded_eval import _find_dunder, BINARY_OP_DUNDERS, UNARY_OP_DUNDERS
from IPython.terminal.shortcuts import auto_suggest
from IPython.utils.decorators import undoc


@undoc
@Condition
def cursor_in_leading_ws():
    before = get_app().current_buffer.document.current_line_before_cursor
    return (not before) or before.isspace()


def has_focus(value: FocusableElement):
    """Wrapper around has_focus adding a nice `__name__` to tester function"""
    tester = has_focus_impl(value).func
    tester.__name__ = f"is_focused({value})"
    return Condition(tester)


@undoc
@Condition
def has_line_below() -> bool:
    document = get_app().current_buffer.document
    return document.cursor_position_row < len(document.lines) - 1


@undoc
@Condition
def is_cursor_at_the_end_of_line() -> bool:
    document = get_app().current_buffer.document
    return document.is_cursor_at_the_end_of_line


@undoc
@Condition
def has_line_above() -> bool:
    document = get_app().current_buffer.document
    return document.cursor_position_row != 0


@Condition
def ebivim():
    shell = get_ipython()
    return shell.emacs_bindings_in_vi_insert_mode


@Condition
def supports_suspend():
    return hasattr(signal, "SIGTSTP")


@Condition
def auto_match():
    shell = get_ipython()
    return shell.auto_match


def all_quotes_paired(quote, buf):
    paired = True
    i = 0
    while i < len(buf):
        c = buf[i]
        if c == quote:
            paired = not paired
        elif c == "\\":
            i += 1
        i += 1
    return paired


_preceding_text_cache: Dict[Union[str, Callable], Condition] = {}
_following_text_cache: Dict[Union[str, Callable], Condition] = {}


def preceding_text(pattern: Union[str, Callable]):
    if pattern in _preceding_text_cache:
        return _preceding_text_cache[pattern]

    if callable(pattern):

        def _preceding_text():
            app = get_app()
            before_cursor = app.current_buffer.document.current_line_before_cursor
            # mypy can't infer if(callable): https://github.com/python/mypy/issues/3603
            return bool(pattern(before_cursor))  # type: ignore[operator]

    else:
        m = re.compile(pattern)

        def _preceding_text():
            app = get_app()
            before_cursor = app.current_buffer.document.current_line_before_cursor
            return bool(m.match(before_cursor))

        _preceding_text.__name__ = f"preceding_text({pattern!r})"

    condition = Condition(_preceding_text)
    _preceding_text_cache[pattern] = condition
    return condition


def following_text(pattern):
    try:
        return _following_text_cache[pattern]
    except KeyError:
        pass
    m = re.compile(pattern)

    def _following_text():
        app = get_app()
        return bool(m.match(app.current_buffer.document.current_line_after_cursor))

    _following_text.__name__ = f"following_text({pattern!r})"

    condition = Condition(_following_text)
    _following_text_cache[pattern] = condition
    return condition


@Condition
def not_inside_unclosed_string():
    app = get_app()
    s = app.current_buffer.document.text_before_cursor
    # remove escaped quotes
    s = s.replace('\\"', "").replace("\\'", "")
    # remove triple-quoted string literals
    s = re.sub(r"(?:\"\"\"[\s\S]*\"\"\"|'''[\s\S]*''')", "", s)
    # remove single-quoted string literals
    s = re.sub(r"""(?:"[^"]*["\n]|'[^']*['\n])""", "", s)
    return not ('"' in s or "'" in s)


@Condition
def navigable_suggestions():
    shell = get_ipython()
    return isinstance(shell.auto_suggest, auto_suggest.NavigableAutoSuggestFromHistory)


@Condition
def readline_like_completions():
    shell = get_ipython()
    return shell.display_completions == "readlinelike"


@Condition
def is_windows_os():
    return sys.platform == "win32"


class PassThrough(Filter):
    """A filter allowing to implement pass-through behaviour of keybindings.

    Prompt toolkit key processor dispatches only one event per binding match,
    which means that adding a new shortcut will suppress the old shortcut
    if the keybindings are the same (unless one is filtered out).

    To stop a shortcut binding from suppressing other shortcuts:
    - add the `pass_through` filter to list of filter, and
    - call `pass_through.reply(event)` in the shortcut handler.
    """

    def __init__(self):
        self._is_replying = False

    def reply(self, event: KeyPressEvent):
        self._is_replying = True
        try:
            event.key_processor.reset()
            event.key_processor.feed_multiple(event.key_sequence)
            event.key_processor.process_keys()
        finally:
            self._is_replying = False

    def __call__(self):
        return not self._is_replying


pass_through = PassThrough()

# these one is callable and re-used multiple times hence needs to be
# only defined once beforhand so that transforming back to human-readable
# names works well in the documentation.
default_buffer_focused = has_focus(DEFAULT_BUFFER)

KEYBINDING_FILTERS = {
    "always": Always(),
    # never is used for exposing commands which have no default keybindings
    "never": Never(),
    "has_line_below": has_line_below,
    "has_line_above": has_line_above,
    "is_cursor_at_the_end_of_line": is_cursor_at_the_end_of_line,
    "has_selection": has_selection,
    "has_suggestion": has_suggestion,
    "vi_mode": vi_mode,
    "vi_insert_mode": vi_insert_mode,
    "emacs_insert_mode": emacs_insert_mode,
    # https://github.com/ipython/ipython/pull/12603 argued for inclusion of
    # emacs key bindings with a configurable `emacs_bindings_in_vi_insert_mode`
    # toggle; when the toggle is on user can access keybindigns like `ctrl + e`
    # in vi insert mode. Because some of the emacs bindings involve `escape`
    # followed by another key, e.g. `escape` followed by `f`, prompt-toolkit
    # needs to wait to see if there will be another character typed in before
    # executing pure `escape` keybinding; in vi insert mode `escape` switches to
    # command mode which is common and performance critical action for vi users.
    # To avoid the delay users employ a workaround:
    # https://github.com/ipython/ipython/issues/13443#issuecomment-1032753703
    # which involves switching `emacs_bindings_in_vi_insert_mode` off.
    #
    # For the workaround to work:
    # 1) end users need to toggle `emacs_bindings_in_vi_insert_mode` off
    # 2) all keybindings which would involve `escape` need to respect that
    #    toggle by including either:
    #      - `vi_insert_mode & ebivim` for actions which have emacs keybindings
    #        predefined upstream in prompt-toolkit, or
    #      - `emacs_like_insert_mode` for actions which do not have existing
    #        emacs keybindings predefined upstream (or need overriding of the
    #        upstream bindings to modify behaviour), defined below.
    "emacs_like_insert_mode": (vi_insert_mode & ebivim) | emacs_insert_mode,
    "has_completions": has_completions,
    "insert_mode": vi_insert_mode | emacs_insert_mode,
    "default_buffer_focused": default_buffer_focused,
    "search_buffer_focused": has_focus(SEARCH_BUFFER),
    # `ebivim` stands for emacs bindings in vi insert mode
    "ebivim": ebivim,
    "supports_suspend": supports_suspend,
    "is_windows_os": is_windows_os,
    "auto_match": auto_match,
    "focused_insert": (vi_insert_mode | emacs_insert_mode) & default_buffer_focused,
    "not_inside_unclosed_string": not_inside_unclosed_string,
    "readline_like_completions": readline_like_completions,
    "preceded_by_paired_double_quotes": preceding_text(
        lambda line: all_quotes_paired('"', line)
    ),
    "preceded_by_paired_single_quotes": preceding_text(
        lambda line: all_quotes_paired("'", line)
    ),
    "preceded_by_raw_str_prefix": preceding_text(r".*(r|R)[\"'](-*)$"),
    "preceded_by_two_double_quotes": preceding_text(r'^.*""$'),
    "preceded_by_two_single_quotes": preceding_text(r"^.*''$"),
    "followed_by_closing_paren_or_end": following_text(r"[,)}\]]|$"),
    "preceded_by_opening_round_paren": preceding_text(r".*\($"),
    "preceded_by_opening_bracket": preceding_text(r".*\[$"),
    "preceded_by_opening_brace": preceding_text(r".*\{$"),
    "preceded_by_double_quote": preceding_text('.*"$'),
    "preceded_by_single_quote": preceding_text(r".*'$"),
    "followed_by_closing_round_paren": following_text(r"^\)"),
    "followed_by_closing_bracket": following_text(r"^\]"),
    "followed_by_closing_brace": following_text(r"^\}"),
    "followed_by_double_quote": following_text('^"'),
    "followed_by_single_quote": following_text("^'"),
    "navigable_suggestions": navigable_suggestions,
    "cursor_in_leading_ws": cursor_in_leading_ws,
    "pass_through": pass_through,
}


def eval_node(node: Union[ast.AST, None]):
    if node is None:
        return None
    if isinstance(node, ast.Expression):
        return eval_node(node.body)
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left)
        right = eval_node(node.right)
        dunders = _find_dunder(node.op, BINARY_OP_DUNDERS)
        if dunders:
            return getattr(left, dunders[0])(right)
        raise ValueError(f"Unknown binary operation: {node.op}")
    if isinstance(node, ast.UnaryOp):
        value = eval_node(node.operand)
        dunders = _find_dunder(node.op, UNARY_OP_DUNDERS)
        if dunders:
            return getattr(value, dunders[0])()
        raise ValueError(f"Unknown unary operation: {node.op}")
    if isinstance(node, ast.Name):
        if node.id in KEYBINDING_FILTERS:
            return KEYBINDING_FILTERS[node.id]
        else:
            sep = "\n  - "
            known_filters = sep.join(sorted(KEYBINDING_FILTERS))
            raise NameError(
                f"{node.id} is not a known shortcut filter."
                f" Known filters are: {sep}{known_filters}."
            )
    raise ValueError("Unhandled node", ast.dump(node))


def filter_from_string(code: str):
    expression = ast.parse(code, mode="eval")
    return eval_node(expression)


__all__ = ["KEYBINDING_FILTERS", "filter_from_string"]
