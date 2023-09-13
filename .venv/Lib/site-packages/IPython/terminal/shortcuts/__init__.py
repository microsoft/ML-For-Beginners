"""
Module to define and register Terminal IPython shortcuts with
:mod:`prompt_toolkit`
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import signal
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Any, Optional, List

from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.key_binding.bindings.completion import (
    display_completions_like_readline,
)
from prompt_toolkit.key_binding.vi_state import InputMode, ViState
from prompt_toolkit.filters import Condition

from IPython.core.getipython import get_ipython
from IPython.terminal.shortcuts import auto_match as match
from IPython.terminal.shortcuts import auto_suggest
from IPython.terminal.shortcuts.filters import filter_from_string
from IPython.utils.decorators import undoc

from prompt_toolkit.enums import DEFAULT_BUFFER

__all__ = ["create_ipython_shortcuts"]


@dataclass
class BaseBinding:
    command: Callable[[KeyPressEvent], Any]
    keys: List[str]


@dataclass
class RuntimeBinding(BaseBinding):
    filter: Condition


@dataclass
class Binding(BaseBinding):
    # while filter could be created by referencing variables directly (rather
    # than created from strings), by using strings we ensure that users will
    # be able to create filters in configuration (e.g. JSON) files too, which
    # also benefits the documentation by enforcing human-readable filter names.
    condition: Optional[str] = None

    def __post_init__(self):
        if self.condition:
            self.filter = filter_from_string(self.condition)
        else:
            self.filter = None


def create_identifier(handler: Callable):
    parts = handler.__module__.split(".")
    name = handler.__name__
    package = parts[0]
    if len(parts) > 1:
        final_module = parts[-1]
        return f"{package}:{final_module}.{name}"
    else:
        return f"{package}:{name}"


AUTO_MATCH_BINDINGS = [
    *[
        Binding(
            cmd, [key], "focused_insert & auto_match & followed_by_closing_paren_or_end"
        )
        for key, cmd in match.auto_match_parens.items()
    ],
    *[
        # raw string
        Binding(cmd, [key], "focused_insert & auto_match & preceded_by_raw_str_prefix")
        for key, cmd in match.auto_match_parens_raw_string.items()
    ],
    Binding(
        match.double_quote,
        ['"'],
        "focused_insert"
        " & auto_match"
        " & not_inside_unclosed_string"
        " & preceded_by_paired_double_quotes"
        " & followed_by_closing_paren_or_end",
    ),
    Binding(
        match.single_quote,
        ["'"],
        "focused_insert"
        " & auto_match"
        " & not_inside_unclosed_string"
        " & preceded_by_paired_single_quotes"
        " & followed_by_closing_paren_or_end",
    ),
    Binding(
        match.docstring_double_quotes,
        ['"'],
        "focused_insert"
        " & auto_match"
        " & not_inside_unclosed_string"
        " & preceded_by_two_double_quotes",
    ),
    Binding(
        match.docstring_single_quotes,
        ["'"],
        "focused_insert"
        " & auto_match"
        " & not_inside_unclosed_string"
        " & preceded_by_two_single_quotes",
    ),
    Binding(
        match.skip_over,
        [")"],
        "focused_insert & auto_match & followed_by_closing_round_paren",
    ),
    Binding(
        match.skip_over,
        ["]"],
        "focused_insert & auto_match & followed_by_closing_bracket",
    ),
    Binding(
        match.skip_over,
        ["}"],
        "focused_insert & auto_match & followed_by_closing_brace",
    ),
    Binding(
        match.skip_over, ['"'], "focused_insert & auto_match & followed_by_double_quote"
    ),
    Binding(
        match.skip_over, ["'"], "focused_insert & auto_match & followed_by_single_quote"
    ),
    Binding(
        match.delete_pair,
        ["backspace"],
        "focused_insert"
        " & preceded_by_opening_round_paren"
        " & auto_match"
        " & followed_by_closing_round_paren",
    ),
    Binding(
        match.delete_pair,
        ["backspace"],
        "focused_insert"
        " & preceded_by_opening_bracket"
        " & auto_match"
        " & followed_by_closing_bracket",
    ),
    Binding(
        match.delete_pair,
        ["backspace"],
        "focused_insert"
        " & preceded_by_opening_brace"
        " & auto_match"
        " & followed_by_closing_brace",
    ),
    Binding(
        match.delete_pair,
        ["backspace"],
        "focused_insert"
        " & preceded_by_double_quote"
        " & auto_match"
        " & followed_by_double_quote",
    ),
    Binding(
        match.delete_pair,
        ["backspace"],
        "focused_insert"
        " & preceded_by_single_quote"
        " & auto_match"
        " & followed_by_single_quote",
    ),
]

AUTO_SUGGEST_BINDINGS = [
    # there are two reasons for re-defining bindings defined upstream:
    # 1) prompt-toolkit does not execute autosuggestion bindings in vi mode,
    # 2) prompt-toolkit checks if we are at the end of text, not end of line
    #    hence it does not work in multi-line mode of navigable provider
    Binding(
        auto_suggest.accept_or_jump_to_end,
        ["end"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept_or_jump_to_end,
        ["c-e"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept,
        ["c-f"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept,
        ["right"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept_word,
        ["escape", "f"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept_token,
        ["c-right"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.discard,
        ["escape"],
        # note this one is using `emacs_insert_mode`, not `emacs_like_insert_mode`
        # as in `vi_insert_mode` we do not want `escape` to be shadowed (ever).
        "has_suggestion & default_buffer_focused & emacs_insert_mode",
    ),
    Binding(
        auto_suggest.discard,
        ["delete"],
        "has_suggestion & default_buffer_focused & emacs_insert_mode",
    ),
    Binding(
        auto_suggest.swap_autosuggestion_up,
        ["c-up"],
        "navigable_suggestions"
        " & ~has_line_above"
        " & has_suggestion"
        " & default_buffer_focused",
    ),
    Binding(
        auto_suggest.swap_autosuggestion_down,
        ["c-down"],
        "navigable_suggestions"
        " & ~has_line_below"
        " & has_suggestion"
        " & default_buffer_focused",
    ),
    Binding(
        auto_suggest.up_and_update_hint,
        ["c-up"],
        "has_line_above & navigable_suggestions & default_buffer_focused",
    ),
    Binding(
        auto_suggest.down_and_update_hint,
        ["c-down"],
        "has_line_below & navigable_suggestions & default_buffer_focused",
    ),
    Binding(
        auto_suggest.accept_character,
        ["escape", "right"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept_and_move_cursor_left,
        ["c-left"],
        "has_suggestion & default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.accept_and_keep_cursor,
        ["escape", "down"],
        "has_suggestion & default_buffer_focused & emacs_insert_mode",
    ),
    Binding(
        auto_suggest.backspace_and_resume_hint,
        ["backspace"],
        # no `has_suggestion` here to allow resuming if no suggestion
        "default_buffer_focused & emacs_like_insert_mode",
    ),
    Binding(
        auto_suggest.resume_hinting,
        ["right"],
        "is_cursor_at_the_end_of_line"
        " & default_buffer_focused"
        " & emacs_like_insert_mode"
        " & pass_through",
    ),
]


SIMPLE_CONTROL_BINDINGS = [
    Binding(cmd, [key], "vi_insert_mode & default_buffer_focused & ebivim")
    for key, cmd in {
        "c-a": nc.beginning_of_line,
        "c-b": nc.backward_char,
        "c-k": nc.kill_line,
        "c-w": nc.backward_kill_word,
        "c-y": nc.yank,
        "c-_": nc.undo,
    }.items()
]


ALT_AND_COMOBO_CONTROL_BINDINGS = [
    Binding(cmd, list(keys), "vi_insert_mode & default_buffer_focused & ebivim")
    for keys, cmd in {
        # Control Combos
        ("c-x", "c-e"): nc.edit_and_execute,
        ("c-x", "e"): nc.edit_and_execute,
        # Alt
        ("escape", "b"): nc.backward_word,
        ("escape", "c"): nc.capitalize_word,
        ("escape", "d"): nc.kill_word,
        ("escape", "h"): nc.backward_kill_word,
        ("escape", "l"): nc.downcase_word,
        ("escape", "u"): nc.uppercase_word,
        ("escape", "y"): nc.yank_pop,
        ("escape", "."): nc.yank_last_arg,
    }.items()
]


def add_binding(bindings: KeyBindings, binding: Binding):
    bindings.add(
        *binding.keys,
        **({"filter": binding.filter} if binding.filter is not None else {}),
    )(binding.command)


def create_ipython_shortcuts(shell, skip=None) -> KeyBindings:
    """Set up the prompt_toolkit keyboard shortcuts for IPython.

    Parameters
    ----------
    shell: InteractiveShell
        The current IPython shell Instance
    skip: List[Binding]
        Bindings to skip.

    Returns
    -------
    KeyBindings
        the keybinding instance for prompt toolkit.

    """
    kb = KeyBindings()
    skip = skip or []
    for binding in KEY_BINDINGS:
        skip_this_one = False
        for to_skip in skip:
            if (
                to_skip.command == binding.command
                and to_skip.filter == binding.filter
                and to_skip.keys == binding.keys
            ):
                skip_this_one = True
                break
        if skip_this_one:
            continue
        add_binding(kb, binding)

    def get_input_mode(self):
        app = get_app()
        app.ttimeoutlen = shell.ttimeoutlen
        app.timeoutlen = shell.timeoutlen

        return self._input_mode

    def set_input_mode(self, mode):
        shape = {InputMode.NAVIGATION: 2, InputMode.REPLACE: 4}.get(mode, 6)
        cursor = "\x1b[{} q".format(shape)

        sys.stdout.write(cursor)
        sys.stdout.flush()

        self._input_mode = mode

    if shell.editing_mode == "vi" and shell.modal_cursor:
        ViState._input_mode = InputMode.INSERT  # type: ignore
        ViState.input_mode = property(get_input_mode, set_input_mode)  # type: ignore

    return kb


def reformat_and_execute(event):
    """Reformat code and execute it"""
    shell = get_ipython()
    reformat_text_before_cursor(
        event.current_buffer, event.current_buffer.document, shell
    )
    event.current_buffer.validate_and_handle()


def reformat_text_before_cursor(buffer, document, shell):
    text = buffer.delete_before_cursor(len(document.text[: document.cursor_position]))
    try:
        formatted_text = shell.reformat_handler(text)
        buffer.insert_text(formatted_text)
    except Exception as e:
        buffer.insert_text(text)


def handle_return_or_newline_or_execute(event):
    shell = get_ipython()
    if getattr(shell, "handle_return", None):
        return shell.handle_return(shell)(event)
    else:
        return newline_or_execute_outer(shell)(event)


def newline_or_execute_outer(shell):
    def newline_or_execute(event):
        """When the user presses return, insert a newline or execute the code."""
        b = event.current_buffer
        d = b.document

        if b.complete_state:
            cc = b.complete_state.current_completion
            if cc:
                b.apply_completion(cc)
            else:
                b.cancel_completion()
            return

        # If there's only one line, treat it as if the cursor is at the end.
        # See https://github.com/ipython/ipython/issues/10425
        if d.line_count == 1:
            check_text = d.text
        else:
            check_text = d.text[: d.cursor_position]
        status, indent = shell.check_complete(check_text)

        # if all we have after the cursor is whitespace: reformat current text
        # before cursor
        after_cursor = d.text[d.cursor_position :]
        reformatted = False
        if not after_cursor.strip():
            reformat_text_before_cursor(b, d, shell)
            reformatted = True
        if not (
            d.on_last_line
            or d.cursor_position_row >= d.line_count - d.empty_line_count_at_the_end()
        ):
            if shell.autoindent:
                b.insert_text("\n" + indent)
            else:
                b.insert_text("\n")
            return

        if (status != "incomplete") and b.accept_handler:
            if not reformatted:
                reformat_text_before_cursor(b, d, shell)
            b.validate_and_handle()
        else:
            if shell.autoindent:
                b.insert_text("\n" + indent)
            else:
                b.insert_text("\n")

    return newline_or_execute


def previous_history_or_previous_completion(event):
    """
    Control-P in vi edit mode on readline is history next, unlike default prompt toolkit.

    If completer is open this still select previous completion.
    """
    event.current_buffer.auto_up()


def next_history_or_next_completion(event):
    """
    Control-N in vi edit mode on readline is history previous, unlike default prompt toolkit.

    If completer is open this still select next completion.
    """
    event.current_buffer.auto_down()


def dismiss_completion(event):
    """Dismiss completion"""
    b = event.current_buffer
    if b.complete_state:
        b.cancel_completion()


def reset_buffer(event):
    """Reset buffer"""
    b = event.current_buffer
    if b.complete_state:
        b.cancel_completion()
    else:
        b.reset()


def reset_search_buffer(event):
    """Reset search buffer"""
    if event.current_buffer.document.text:
        event.current_buffer.reset()
    else:
        event.app.layout.focus(DEFAULT_BUFFER)


def suspend_to_bg(event):
    """Suspend to background"""
    event.app.suspend_to_background()


def quit(event):
    """
    Quit application with ``SIGQUIT`` if supported or ``sys.exit`` otherwise.

    On platforms that support SIGQUIT, send SIGQUIT to the current process.
    On other platforms, just exit the process with a message.
    """
    sigquit = getattr(signal, "SIGQUIT", None)
    if sigquit is not None:
        os.kill(0, signal.SIGQUIT)
    else:
        sys.exit("Quit")


def indent_buffer(event):
    """Indent buffer"""
    event.current_buffer.insert_text(" " * 4)


def newline_autoindent(event):
    """Insert a newline after the cursor indented appropriately.

    Fancier version of former ``newline_with_copy_margin`` which should
    compute the correct indentation of the inserted line. That is to say, indent
    by 4 extra space after a function definition, class definition, context
    manager... And dedent by 4 space after ``pass``, ``return``, ``raise ...``.
    """
    shell = get_ipython()
    inputsplitter = shell.input_transformer_manager
    b = event.current_buffer
    d = b.document

    if b.complete_state:
        b.cancel_completion()
    text = d.text[: d.cursor_position] + "\n"
    _, indent = inputsplitter.check_complete(text)
    b.insert_text("\n" + (" " * (indent or 0)), move_cursor=False)


def open_input_in_editor(event):
    """Open code from input in external editor"""
    event.app.current_buffer.open_in_editor()


if sys.platform == "win32":
    from IPython.core.error import TryNext
    from IPython.lib.clipboard import (
        ClipboardEmpty,
        tkinter_clipboard_get,
        win32_clipboard_get,
    )

    @undoc
    def win_paste(event):
        try:
            text = win32_clipboard_get()
        except TryNext:
            try:
                text = tkinter_clipboard_get()
            except (TryNext, ClipboardEmpty):
                return
        except ClipboardEmpty:
            return
        event.current_buffer.insert_text(text.replace("\t", " " * 4))

else:

    @undoc
    def win_paste(event):
        """Stub used on other platforms"""
        pass


KEY_BINDINGS = [
    Binding(
        handle_return_or_newline_or_execute,
        ["enter"],
        "default_buffer_focused & ~has_selection & insert_mode",
    ),
    Binding(
        reformat_and_execute,
        ["escape", "enter"],
        "default_buffer_focused & ~has_selection & insert_mode & ebivim",
    ),
    Binding(quit, ["c-\\"]),
    Binding(
        previous_history_or_previous_completion,
        ["c-p"],
        "vi_insert_mode & default_buffer_focused",
    ),
    Binding(
        next_history_or_next_completion,
        ["c-n"],
        "vi_insert_mode & default_buffer_focused",
    ),
    Binding(dismiss_completion, ["c-g"], "default_buffer_focused & has_completions"),
    Binding(reset_buffer, ["c-c"], "default_buffer_focused"),
    Binding(reset_search_buffer, ["c-c"], "search_buffer_focused"),
    Binding(suspend_to_bg, ["c-z"], "supports_suspend"),
    Binding(
        indent_buffer,
        ["tab"],  # Ctrl+I == Tab
        "default_buffer_focused"
        " & ~has_selection"
        " & insert_mode"
        " & cursor_in_leading_ws",
    ),
    Binding(newline_autoindent, ["c-o"], "default_buffer_focused & emacs_insert_mode"),
    Binding(open_input_in_editor, ["f2"], "default_buffer_focused"),
    *AUTO_MATCH_BINDINGS,
    *AUTO_SUGGEST_BINDINGS,
    Binding(
        display_completions_like_readline,
        ["c-i"],
        "readline_like_completions"
        " & default_buffer_focused"
        " & ~has_selection"
        " & insert_mode"
        " & ~cursor_in_leading_ws",
    ),
    Binding(win_paste, ["c-v"], "default_buffer_focused & ~vi_mode & is_windows_os"),
    *SIMPLE_CONTROL_BINDINGS,
    *ALT_AND_COMOBO_CONTROL_BINDINGS,
]
