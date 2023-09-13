"""prompt-toolkit utilities

Everything in this module is a private API,
not to be used outside IPython.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import unicodedata
from wcwidth import wcwidth

from IPython.core.completer import (
    provisionalcompleter, cursor_to_position,
    _deduplicate_completions)
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout

import pygments.lexers as pygments_lexers
import os
import sys
import traceback

_completion_sentinel = object()

def _elide_point(string:str, *, min_elide=30)->str:
    """
    If a string is long enough, and has at least 3 dots,
    replace the middle part with ellipses.

    If a string naming a file is long enough, and has at least 3 slashes,
    replace the middle part with ellipses.

    If three consecutive dots, or two consecutive dots are encountered these are
    replaced by the equivalents HORIZONTAL ELLIPSIS or TWO DOT LEADER unicode
    equivalents
    """
    string = string.replace('...','\N{HORIZONTAL ELLIPSIS}')
    string = string.replace('..','\N{TWO DOT LEADER}')
    if len(string) < min_elide:
        return string

    object_parts = string.split('.')
    file_parts = string.split(os.sep)
    if file_parts[-1] == '':
        file_parts.pop()

    if len(object_parts) > 3:
        return "{}.{}\N{HORIZONTAL ELLIPSIS}{}.{}".format(
            object_parts[0],
            object_parts[1][:1],
            object_parts[-2][-1:],
            object_parts[-1],
        )

    elif len(file_parts) > 3:
        return ("{}" + os.sep + "{}\N{HORIZONTAL ELLIPSIS}{}" + os.sep + "{}").format(
            file_parts[0], file_parts[1][:1], file_parts[-2][-1:], file_parts[-1]
        )

    return string

def _elide_typed(string:str, typed:str, *, min_elide:int=30)->str:
    """
    Elide the middle of a long string if the beginning has already been typed.
    """

    if len(string) < min_elide:
        return string
    cut_how_much = len(typed)-3
    if cut_how_much < 7:
        return string
    if string.startswith(typed) and len(string)> len(typed):
        return f"{string[:3]}\N{HORIZONTAL ELLIPSIS}{string[cut_how_much:]}"
    return string

def _elide(string:str, typed:str, min_elide=30)->str:
    return _elide_typed(
        _elide_point(string, min_elide=min_elide),
        typed, min_elide=min_elide)



def _adjust_completion_text_based_on_context(text, body, offset):
    if text.endswith('=') and len(body) > offset and body[offset] == '=':
        return text[:-1]
    else:
        return text


class IPythonPTCompleter(Completer):
    """Adaptor to provide IPython completions to prompt_toolkit"""
    def __init__(self, ipy_completer=None, shell=None):
        if shell is None and ipy_completer is None:
            raise TypeError("Please pass shell=an InteractiveShell instance.")
        self._ipy_completer = ipy_completer
        self.shell = shell

    @property
    def ipy_completer(self):
        if self._ipy_completer:
            return self._ipy_completer
        else:
            return self.shell.Completer

    def get_completions(self, document, complete_event):
        if not document.current_line.strip():
            return
        # Some bits of our completion system may print stuff (e.g. if a module
        # is imported). This context manager ensures that doesn't interfere with
        # the prompt.

        with patch_stdout(), provisionalcompleter():
            body = document.text
            cursor_row = document.cursor_position_row
            cursor_col = document.cursor_position_col
            cursor_position = document.cursor_position
            offset = cursor_to_position(body, cursor_row, cursor_col)
            try:
                yield from self._get_completions(body, offset, cursor_position, self.ipy_completer)
            except Exception as e:
                try:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_tb)
                except AttributeError:
                    print('Unrecoverable Error in completions')

    @staticmethod
    def _get_completions(body, offset, cursor_position, ipyc):
        """
        Private equivalent of get_completions() use only for unit_testing.
        """
        debug = getattr(ipyc, 'debug', False)
        completions = _deduplicate_completions(
            body, ipyc.completions(body, offset))
        for c in completions:
            if not c.text:
                # Guard against completion machinery giving us an empty string.
                continue
            text = unicodedata.normalize('NFC', c.text)
            # When the first character of the completion has a zero length,
            # then it's probably a decomposed unicode character. E.g. caused by
            # the "\dot" completion. Try to compose again with the previous
            # character.
            if wcwidth(text[0]) == 0:
                if cursor_position + c.start > 0:
                    char_before = body[c.start - 1]
                    fixed_text = unicodedata.normalize(
                        'NFC', char_before + text)

                    # Yield the modified completion instead, if this worked.
                    if wcwidth(text[0:1]) == 1:
                        yield Completion(fixed_text, start_position=c.start - offset - 1)
                        continue

            # TODO: Use Jedi to determine meta_text
            # (Jedi currently has a bug that results in incorrect information.)
            # meta_text = ''
            # yield Completion(m, start_position=start_pos,
            #                  display_meta=meta_text)
            display_text = c.text

            adjusted_text = _adjust_completion_text_based_on_context(c.text, body, offset)
            if c.type == 'function':
                yield Completion(adjusted_text, start_position=c.start - offset, display=_elide(display_text+'()', body[c.start:c.end]), display_meta=c.type+c.signature)
            else:
                yield Completion(adjusted_text, start_position=c.start - offset, display=_elide(display_text,  body[c.start:c.end]), display_meta=c.type)

class IPythonPTLexer(Lexer):
    """
    Wrapper around PythonLexer and BashLexer.
    """
    def __init__(self):
        l = pygments_lexers
        self.python_lexer = PygmentsLexer(l.Python3Lexer)
        self.shell_lexer = PygmentsLexer(l.BashLexer)

        self.magic_lexers = {
            'HTML': PygmentsLexer(l.HtmlLexer),
            'html': PygmentsLexer(l.HtmlLexer),
            'javascript': PygmentsLexer(l.JavascriptLexer),
            'js': PygmentsLexer(l.JavascriptLexer),
            'perl': PygmentsLexer(l.PerlLexer),
            'ruby': PygmentsLexer(l.RubyLexer),
            'latex': PygmentsLexer(l.TexLexer),
        }

    def lex_document(self, document):
        text = document.text.lstrip()

        lexer = self.python_lexer

        if text.startswith('!') or text.startswith('%%bash'):
            lexer = self.shell_lexer

        elif text.startswith('%%'):
            for magic, l in self.magic_lexers.items():
                if text.startswith('%%' + magic):
                    lexer = l
                    break

        return lexer.lex_document(document)
