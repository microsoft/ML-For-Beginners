import re
import tokenize
from io import StringIO
from typing import Callable, List, Optional, Union, Generator, Tuple
import warnings

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.layout.processors import (
    Processor,
    Transformation,
    TransformationInput,
)

from IPython.core.getipython import get_ipython
from IPython.utils.tokenutil import generate_tokens

from .filters import pass_through


def _get_query(document: Document):
    return document.lines[document.cursor_position_row]


class AppendAutoSuggestionInAnyLine(Processor):
    """
    Append the auto suggestion to lines other than the last (appending to the
    last line is natively supported by the prompt toolkit).
    """

    def __init__(self, style: str = "class:auto-suggestion") -> None:
        self.style = style

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        is_last_line = ti.lineno == ti.document.line_count - 1
        is_active_line = ti.lineno == ti.document.cursor_position_row

        if not is_last_line and is_active_line:
            buffer = ti.buffer_control.buffer

            if buffer.suggestion and ti.document.is_cursor_at_the_end_of_line:
                suggestion = buffer.suggestion.text
            else:
                suggestion = ""

            return Transformation(fragments=ti.fragments + [(self.style, suggestion)])
        else:
            return Transformation(fragments=ti.fragments)


class NavigableAutoSuggestFromHistory(AutoSuggestFromHistory):
    """
    A subclass of AutoSuggestFromHistory that allow navigation to next/previous
    suggestion from history. To do so it remembers the current position, but it
    state need to carefully be cleared on the right events.
    """

    def __init__(
        self,
    ):
        self.skip_lines = 0
        self._connected_apps = []

    def reset_history_position(self, _: Buffer):
        self.skip_lines = 0

    def disconnect(self):
        for pt_app in self._connected_apps:
            text_insert_event = pt_app.default_buffer.on_text_insert
            text_insert_event.remove_handler(self.reset_history_position)

    def connect(self, pt_app: PromptSession):
        self._connected_apps.append(pt_app)
        # note: `on_text_changed` could be used for a bit different behaviour
        # on character deletion (i.e. reseting history position on backspace)
        pt_app.default_buffer.on_text_insert.add_handler(self.reset_history_position)
        pt_app.default_buffer.on_cursor_position_changed.add_handler(self._dismiss)

    def get_suggestion(
        self, buffer: Buffer, document: Document
    ) -> Optional[Suggestion]:
        text = _get_query(document)

        if text.strip():
            for suggestion, _ in self._find_next_match(
                text, self.skip_lines, buffer.history
            ):
                return Suggestion(suggestion)

        return None

    def _dismiss(self, buffer, *args, **kwargs):
        buffer.suggestion = None

    def _find_match(
        self, text: str, skip_lines: float, history: History, previous: bool
    ) -> Generator[Tuple[str, float], None, None]:
        """
        text : str
            Text content to find a match for, the user cursor is most of the
            time at the end of this text.
        skip_lines : float
            number of items to skip in the search, this is used to indicate how
            far in the list the user has navigated by pressing up or down.
            The float type is used as the base value is +inf
        history : History
            prompt_toolkit History instance to fetch previous entries from.
        previous : bool
            Direction of the search, whether we are looking previous match
            (True), or next match (False).

        Yields
        ------
        Tuple with:
        str:
            current suggestion.
        float:
            will actually yield only ints, which is passed back via skip_lines,
            which may be a +inf (float)


        """
        line_number = -1
        for string in reversed(list(history.get_strings())):
            for line in reversed(string.splitlines()):
                line_number += 1
                if not previous and line_number < skip_lines:
                    continue
                # do not return empty suggestions as these
                # close the auto-suggestion overlay (and are useless)
                if line.startswith(text) and len(line) > len(text):
                    yield line[len(text) :], line_number
                if previous and line_number >= skip_lines:
                    return

    def _find_next_match(
        self, text: str, skip_lines: float, history: History
    ) -> Generator[Tuple[str, float], None, None]:
        return self._find_match(text, skip_lines, history, previous=False)

    def _find_previous_match(self, text: str, skip_lines: float, history: History):
        return reversed(
            list(self._find_match(text, skip_lines, history, previous=True))
        )

    def up(self, query: str, other_than: str, history: History) -> None:
        for suggestion, line_number in self._find_next_match(
            query, self.skip_lines, history
        ):
            # if user has history ['very.a', 'very', 'very.b'] and typed 'very'
            # we want to switch from 'very.b' to 'very.a' because a) if the
            # suggestion equals current text, prompt-toolkit aborts suggesting
            # b) user likely would not be interested in 'very' anyways (they
            # already typed it).
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            # no matches found, cycle back to beginning
            self.skip_lines = 0

    def down(self, query: str, other_than: str, history: History) -> None:
        for suggestion, line_number in self._find_previous_match(
            query, self.skip_lines, history
        ):
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            # no matches found, cycle to end
            for suggestion, line_number in self._find_previous_match(
                query, float("Inf"), history
            ):
                if query + suggestion != other_than:
                    self.skip_lines = line_number
                    break


def accept_or_jump_to_end(event: KeyPressEvent):
    """Apply autosuggestion or jump to end of line."""
    buffer = event.current_buffer
    d = buffer.document
    after_cursor = d.text[d.cursor_position :]
    lines = after_cursor.split("\n")
    end_of_current_line = lines[0].strip()
    suggestion = buffer.suggestion
    if (suggestion is not None) and (suggestion.text) and (end_of_current_line == ""):
        buffer.insert_text(suggestion.text)
    else:
        nc.end_of_line(event)


def _deprected_accept_in_vi_insert_mode(event: KeyPressEvent):
    """Accept autosuggestion or jump to end of line.

    .. deprecated:: 8.12
        Use `accept_or_jump_to_end` instead.
    """
    return accept_or_jump_to_end(event)


def accept(event: KeyPressEvent):
    """Accept autosuggestion"""
    buffer = event.current_buffer
    suggestion = buffer.suggestion
    if suggestion:
        buffer.insert_text(suggestion.text)
    else:
        nc.forward_char(event)


def discard(event: KeyPressEvent):
    """Discard autosuggestion"""
    buffer = event.current_buffer
    buffer.suggestion = None


def accept_word(event: KeyPressEvent):
    """Fill partial autosuggestion by word"""
    buffer = event.current_buffer
    suggestion = buffer.suggestion
    if suggestion:
        t = re.split(r"(\S+\s+)", suggestion.text)
        buffer.insert_text(next((x for x in t if x), ""))
    else:
        nc.forward_word(event)


def accept_character(event: KeyPressEvent):
    """Fill partial autosuggestion by character"""
    b = event.current_buffer
    suggestion = b.suggestion
    if suggestion and suggestion.text:
        b.insert_text(suggestion.text[0])


def accept_and_keep_cursor(event: KeyPressEvent):
    """Accept autosuggestion and keep cursor in place"""
    buffer = event.current_buffer
    old_position = buffer.cursor_position
    suggestion = buffer.suggestion
    if suggestion:
        buffer.insert_text(suggestion.text)
        buffer.cursor_position = old_position


def accept_and_move_cursor_left(event: KeyPressEvent):
    """Accept autosuggestion and move cursor left in place"""
    accept_and_keep_cursor(event)
    nc.backward_char(event)


def _update_hint(buffer: Buffer):
    if buffer.auto_suggest:
        suggestion = buffer.auto_suggest.get_suggestion(buffer, buffer.document)
        buffer.suggestion = suggestion


def backspace_and_resume_hint(event: KeyPressEvent):
    """Resume autosuggestions after deleting last character"""
    nc.backward_delete_char(event)
    _update_hint(event.current_buffer)


def resume_hinting(event: KeyPressEvent):
    """Resume autosuggestions"""
    pass_through.reply(event)
    # Order matters: if update happened first and event reply second, the
    # suggestion would be auto-accepted if both actions are bound to same key.
    _update_hint(event.current_buffer)


def up_and_update_hint(event: KeyPressEvent):
    """Go up and update hint"""
    current_buffer = event.current_buffer

    current_buffer.auto_up(count=event.arg)
    _update_hint(current_buffer)


def down_and_update_hint(event: KeyPressEvent):
    """Go down and update hint"""
    current_buffer = event.current_buffer

    current_buffer.auto_down(count=event.arg)
    _update_hint(current_buffer)


def accept_token(event: KeyPressEvent):
    """Fill partial autosuggestion by token"""
    b = event.current_buffer
    suggestion = b.suggestion

    if suggestion:
        prefix = _get_query(b.document)
        text = prefix + suggestion.text

        tokens: List[Optional[str]] = [None, None, None]
        substrings = [""]
        i = 0

        for token in generate_tokens(StringIO(text).readline):
            if token.type == tokenize.NEWLINE:
                index = len(text)
            else:
                index = text.index(token[1], len(substrings[-1]))
            substrings.append(text[:index])
            tokenized_so_far = substrings[-1]
            if tokenized_so_far.startswith(prefix):
                if i == 0 and len(tokenized_so_far) > len(prefix):
                    tokens[0] = tokenized_so_far[len(prefix) :]
                    substrings.append(tokenized_so_far)
                    i += 1
                tokens[i] = token[1]
                if i == 2:
                    break
                i += 1

        if tokens[0]:
            to_insert: str
            insert_text = substrings[-2]
            if tokens[1] and len(tokens[1]) == 1:
                insert_text = substrings[-1]
            to_insert = insert_text[len(prefix) :]
            b.insert_text(to_insert)
            return

    nc.forward_word(event)


Provider = Union[AutoSuggestFromHistory, NavigableAutoSuggestFromHistory, None]


def _swap_autosuggestion(
    buffer: Buffer,
    provider: NavigableAutoSuggestFromHistory,
    direction_method: Callable,
):
    """
    We skip most recent history entry (in either direction) if it equals the
    current autosuggestion because if user cycles when auto-suggestion is shown
    they most likely want something else than what was suggested (otherwise
    they would have accepted the suggestion).
    """
    suggestion = buffer.suggestion
    if not suggestion:
        return

    query = _get_query(buffer.document)
    current = query + suggestion.text

    direction_method(query=query, other_than=current, history=buffer.history)

    new_suggestion = provider.get_suggestion(buffer, buffer.document)
    buffer.suggestion = new_suggestion


def swap_autosuggestion_up(event: KeyPressEvent):
    """Get next autosuggestion from history."""
    shell = get_ipython()
    provider = shell.auto_suggest

    if not isinstance(provider, NavigableAutoSuggestFromHistory):
        return

    return _swap_autosuggestion(
        buffer=event.current_buffer, provider=provider, direction_method=provider.up
    )


def swap_autosuggestion_down(event: KeyPressEvent):
    """Get previous autosuggestion from history."""
    shell = get_ipython()
    provider = shell.auto_suggest

    if not isinstance(provider, NavigableAutoSuggestFromHistory):
        return

    return _swap_autosuggestion(
        buffer=event.current_buffer,
        provider=provider,
        direction_method=provider.down,
    )


def __getattr__(key):
    if key == "accept_in_vi_insert_mode":
        warnings.warn(
            "`accept_in_vi_insert_mode` is deprecated since IPython 8.12 and "
            "renamed to `accept_or_jump_to_end`. Please update your configuration "
            "accordingly",
            DeprecationWarning,
            stacklevel=2,
        )
        return _deprected_accept_in_vi_insert_mode
    raise AttributeError
