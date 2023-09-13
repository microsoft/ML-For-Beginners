"""
Utilities function for keybinding with prompt toolkit.

This will be bound to specific key press and filter modes,
like whether we are in edit mode, and whether the completer is open.
"""
import re
from prompt_toolkit.key_binding import KeyPressEvent


def parenthesis(event: KeyPressEvent):
    """Auto-close parenthesis"""
    event.current_buffer.insert_text("()")
    event.current_buffer.cursor_left()


def brackets(event: KeyPressEvent):
    """Auto-close brackets"""
    event.current_buffer.insert_text("[]")
    event.current_buffer.cursor_left()


def braces(event: KeyPressEvent):
    """Auto-close braces"""
    event.current_buffer.insert_text("{}")
    event.current_buffer.cursor_left()


def double_quote(event: KeyPressEvent):
    """Auto-close double quotes"""
    event.current_buffer.insert_text('""')
    event.current_buffer.cursor_left()


def single_quote(event: KeyPressEvent):
    """Auto-close single quotes"""
    event.current_buffer.insert_text("''")
    event.current_buffer.cursor_left()


def docstring_double_quotes(event: KeyPressEvent):
    """Auto-close docstring (double quotes)"""
    event.current_buffer.insert_text('""""')
    event.current_buffer.cursor_left(3)


def docstring_single_quotes(event: KeyPressEvent):
    """Auto-close docstring (single quotes)"""
    event.current_buffer.insert_text("''''")
    event.current_buffer.cursor_left(3)


def raw_string_parenthesis(event: KeyPressEvent):
    """Auto-close parenthesis in raw strings"""
    matches = re.match(
        r".*(r|R)[\"'](-*)",
        event.current_buffer.document.current_line_before_cursor,
    )
    dashes = matches.group(2) if matches else ""
    event.current_buffer.insert_text("()" + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)


def raw_string_bracket(event: KeyPressEvent):
    """Auto-close bracker in raw strings"""
    matches = re.match(
        r".*(r|R)[\"'](-*)",
        event.current_buffer.document.current_line_before_cursor,
    )
    dashes = matches.group(2) if matches else ""
    event.current_buffer.insert_text("[]" + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)


def raw_string_braces(event: KeyPressEvent):
    """Auto-close braces in raw strings"""
    matches = re.match(
        r".*(r|R)[\"'](-*)",
        event.current_buffer.document.current_line_before_cursor,
    )
    dashes = matches.group(2) if matches else ""
    event.current_buffer.insert_text("{}" + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)


def skip_over(event: KeyPressEvent):
    """Skip over automatically added parenthesis/quote.

    (rather than adding another parenthesis/quote)"""
    event.current_buffer.cursor_right()


def delete_pair(event: KeyPressEvent):
    """Delete auto-closed parenthesis"""
    event.current_buffer.delete()
    event.current_buffer.delete_before_cursor()


auto_match_parens = {"(": parenthesis, "[": brackets, "{": braces}
auto_match_parens_raw_string = {
    "(": raw_string_parenthesis,
    "[": raw_string_bracket,
    "{": raw_string_braces,
}
