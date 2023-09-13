"""Token-related utilities"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from collections import namedtuple
from io import StringIO
from keyword import iskeyword

import tokenize


Token = namedtuple('Token', ['token', 'text', 'start', 'end', 'line'])

def generate_tokens(readline):
    """wrap generate_tokens to catch EOF errors"""
    try:
        for token in tokenize.generate_tokens(readline):
            yield token
    except tokenize.TokenError:
        # catch EOF error
        return


def generate_tokens_catch_errors(readline, extra_errors_to_catch=None):
    default_errors_to_catch = [
        "unterminated string literal",
        "invalid non-printable character",
        "after line continuation character",
    ]
    assert extra_errors_to_catch is None or isinstance(extra_errors_to_catch, list)
    errors_to_catch = default_errors_to_catch + (extra_errors_to_catch or [])

    tokens = []
    try:
        for token in tokenize.generate_tokens(readline):
            tokens.append(token)
            yield token
    except tokenize.TokenError as exc:
        if any(error in exc.args[0] for error in errors_to_catch):
            if tokens:
                start = tokens[-1].start[0], tokens[-1].end[0]
                end = start
                line = tokens[-1].line
            else:
                start = end = (1, 0)
                line = ""
            yield tokenize.TokenInfo(tokenize.ERRORTOKEN, "", start, end, line)
        else:
            # Catch EOF
            raise


def line_at_cursor(cell, cursor_pos=0):
    """Return the line in a cell at a given cursor position

    Used for calling line-based APIs that don't support multi-line input, yet.

    Parameters
    ----------
    cell : str
        multiline block of text
    cursor_pos : integer
        the cursor position

    Returns
    -------
    (line, offset): (string, integer)
        The line with the current cursor, and the character offset of the start of the line.
    """
    offset = 0
    lines = cell.splitlines(True)
    for line in lines:
        next_offset = offset + len(line)
        if not line.endswith('\n'):
            # If the last line doesn't have a trailing newline, treat it as if
            # it does so that the cursor at the end of the line still counts
            # as being on that line.
            next_offset += 1
        if next_offset > cursor_pos:
            break
        offset = next_offset
    else:
        line = ""
    return (line, offset)

def token_at_cursor(cell, cursor_pos=0):
    """Get the token at a given cursor

    Used for introspection.

    Function calls are prioritized, so the token for the callable will be returned
    if the cursor is anywhere inside the call.

    Parameters
    ----------
    cell : unicode
        A block of Python code
    cursor_pos : int
        The location of the cursor in the block where the token should be found
    """
    names = []
    tokens = []
    call_names = []
    
    offsets = {1: 0} # lines start at 1
    for tup in generate_tokens(StringIO(cell).readline):
        
        tok = Token(*tup)
        
        # token, text, start, end, line = tup
        start_line, start_col = tok.start
        end_line, end_col = tok.end
        if end_line + 1 not in offsets:
            # keep track of offsets for each line
            lines = tok.line.splitlines(True)
            for lineno, line in enumerate(lines, start_line + 1):
                if lineno not in offsets:
                    offsets[lineno] = offsets[lineno-1] + len(line)
        
        offset = offsets[start_line]
        # allow '|foo' to find 'foo' at the beginning of a line
        boundary = cursor_pos + 1 if start_col == 0 else cursor_pos
        if offset + start_col >= boundary:
            # current token starts after the cursor,
            # don't consume it
            break
        
        if tok.token == tokenize.NAME and not iskeyword(tok.text):
            if names and tokens and tokens[-1].token == tokenize.OP and tokens[-1].text == '.':
                names[-1] = "%s.%s" % (names[-1], tok.text)
            else:
                names.append(tok.text)
        elif tok.token == tokenize.OP:
            if tok.text == '=' and names:
                # don't inspect the lhs of an assignment
                names.pop(-1)
            if tok.text == '(' and names:
                # if we are inside a function call, inspect the function
                call_names.append(names[-1])
            elif tok.text == ')' and call_names:
                call_names.pop(-1)
        
        tokens.append(tok)
        
        if offsets[end_line] + end_col > cursor_pos:
            # we found the cursor, stop reading
            break
        
    if call_names:
        return call_names[-1]
    elif names:
        return names[-1]
    else:
        return ''
