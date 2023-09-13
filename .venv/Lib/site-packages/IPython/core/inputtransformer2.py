"""Input transformer machinery to support IPython special syntax.

This includes the machinery to recognise and transform ``%magic`` commands,
``!system`` commands, ``help?`` querying, prompt stripping, and so forth.

Added: IPython 7.0. Replaces inputsplitter and inputtransformer which were
deprecated in 7.0.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings

from IPython.utils import tokenutil

_indent_re = re.compile(r'^[ \t]+')

def leading_empty_lines(lines):
    """Remove leading empty lines

    If the leading lines are empty or contain only whitespace, they will be
    removed.
    """
    if not lines:
        return lines
    for i, line in enumerate(lines):
        if line and not line.isspace():
            return lines[i:]
    return lines

def leading_indent(lines):
    """Remove leading indentation.

    If the first line starts with a spaces or tabs, the same whitespace will be
    removed from each following line in the cell.
    """
    if not lines:
        return lines
    m = _indent_re.match(lines[0])
    if not m:
        return lines
    space = m.group(0)
    n = len(space)
    return [l[n:] if l.startswith(space) else l
            for l in lines]

class PromptStripper:
    """Remove matching input prompts from a block of input.

    Parameters
    ----------
    prompt_re : regular expression
        A regular expression matching any input prompt (including continuation,
        e.g. ``...``)
    initial_re : regular expression, optional
        A regular expression matching only the initial prompt, but not continuation.
        If no initial expression is given, prompt_re will be used everywhere.
        Used mainly for plain Python prompts (``>>>``), where the continuation prompt
        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

    Notes
    -----

    If initial_re and prompt_re differ,
    only initial_re will be tested against the first line.
    If any prompt is found on the first two lines,
    prompts will be stripped from the rest of the block.
    """
    def __init__(self, prompt_re, initial_re=None):
        self.prompt_re = prompt_re
        self.initial_re = initial_re or prompt_re

    def _strip(self, lines):
        return [self.prompt_re.sub('', l, count=1) for l in lines]

    def __call__(self, lines):
        if not lines:
            return lines
        if self.initial_re.match(lines[0]) or \
                (len(lines) > 1 and self.prompt_re.match(lines[1])):
            return self._strip(lines)
        return lines

classic_prompt = PromptStripper(
    prompt_re=re.compile(r'^(>>>|\.\.\.)( |$)'),
    initial_re=re.compile(r'^>>>( |$)')
)

ipython_prompt = PromptStripper(
    re.compile(
        r"""
        ^(                         # Match from the beginning of a line, either:

                                   # 1. First-line prompt:
        ((\[nav\]|\[ins\])?\ )?    # Vi editing mode prompt, if it's there
        In\                        # The 'In' of the prompt, with a space
        \[\d+\]:                   # Command index, as displayed in the prompt
        \                          # With a mandatory trailing space

        |                          # ... or ...

                                   # 2. The three dots of the multiline prompt
        \s*                        # All leading whitespace characters
        \.{3,}:                    # The three (or more) dots
        \ ?                        # With an optional trailing space

        )
        """,
        re.VERBOSE,
    )
)


def cell_magic(lines):
    if not lines or not lines[0].startswith('%%'):
        return lines
    if re.match(r'%%\w+\?', lines[0]):
        # This case will be handled by help_end
        return lines
    magic_name, _, first_line = lines[0][2:].rstrip().partition(' ')
    body = ''.join(lines[1:])
    return ['get_ipython().run_cell_magic(%r, %r, %r)\n'
            % (magic_name, first_line, body)]


def _find_assign_op(token_line) -> Optional[int]:
    """Get the index of the first assignment in the line ('=' not inside brackets)

    Note: We don't try to support multiple special assignment (a = b = %foo)
    """
    paren_level = 0
    for i, ti in enumerate(token_line):
        s = ti.string
        if s == '=' and paren_level == 0:
            return i
        if s in {'(','[','{'}:
            paren_level += 1
        elif s in {')', ']', '}'}:
            if paren_level > 0:
                paren_level -= 1
    return None

def find_end_of_continued_line(lines, start_line: int):
    """Find the last line of a line explicitly extended using backslashes.

    Uses 0-indexed line numbers.
    """
    end_line = start_line
    while lines[end_line].endswith('\\\n'):
        end_line += 1
        if end_line >= len(lines):
            break
    return end_line

def assemble_continued_line(lines, start: Tuple[int, int], end_line: int):
    r"""Assemble a single line from multiple continued line pieces

    Continued lines are lines ending in ``\``, and the line following the last
    ``\`` in the block.

    For example, this code continues over multiple lines::

        if (assign_ix is not None) \
             and (len(line) >= assign_ix + 2) \
             and (line[assign_ix+1].string == '%') \
             and (line[assign_ix+2].type == tokenize.NAME):

    This statement contains four continued line pieces.
    Assembling these pieces into a single line would give::

        if (assign_ix is not None) and (len(line) >= assign_ix + 2) and (line[...

    This uses 0-indexed line numbers. *start* is (lineno, colno).

    Used to allow ``%magic`` and ``!system`` commands to be continued over
    multiple lines.
    """
    parts = [lines[start[0]][start[1]:]] + lines[start[0]+1:end_line+1]
    return ' '.join([p.rstrip()[:-1] for p in parts[:-1]]  # Strip backslash+newline
                    + [parts[-1].rstrip()])         # Strip newline from last line

class TokenTransformBase:
    """Base class for transformations which examine tokens.

    Special syntax should not be transformed when it occurs inside strings or
    comments. This is hard to reliably avoid with regexes. The solution is to
    tokenise the code as Python, and recognise the special syntax in the tokens.

    IPython's special syntax is not valid Python syntax, so tokenising may go
    wrong after the special syntax starts. These classes therefore find and
    transform *one* instance of special syntax at a time into regular Python
    syntax. After each transformation, tokens are regenerated to find the next
    piece of special syntax.

    Subclasses need to implement one class method (find)
    and one regular method (transform).

    The priority attribute can select which transformation to apply if multiple
    transformers match in the same place. Lower numbers have higher priority.
    This allows "%magic?" to be turned into a help call rather than a magic call.
    """
    # Lower numbers -> higher priority (for matches in the same location)
    priority = 10

    def sortby(self):
        return self.start_line, self.start_col, self.priority

    def __init__(self, start):
        self.start_line = start[0] - 1   # Shift from 1-index to 0-index
        self.start_col = start[1]

    @classmethod
    def find(cls, tokens_by_line):
        """Find one instance of special syntax in the provided tokens.

        Tokens are grouped into logical lines for convenience,
        so it is easy to e.g. look at the first token of each line.
        *tokens_by_line* is a list of lists of tokenize.TokenInfo objects.

        This should return an instance of its class, pointing to the start
        position it has found, or None if it found no match.
        """
        raise NotImplementedError

    def transform(self, lines: List[str]):
        """Transform one instance of special syntax found by ``find()``

        Takes a list of strings representing physical lines,
        returns a similar list of transformed lines.
        """
        raise NotImplementedError

class MagicAssign(TokenTransformBase):
    """Transformer for assignments from magics (a = %foo)"""
    @classmethod
    def find(cls, tokens_by_line):
        """Find the first magic assignment (a = %foo) in the cell.
        """
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if (assign_ix is not None) \
                    and (len(line) >= assign_ix + 2) \
                    and (line[assign_ix+1].string == '%') \
                    and (line[assign_ix+2].type == tokenize.NAME):
                return cls(line[assign_ix+1].start)

    def transform(self, lines: List[str]):
        """Transform a magic assignment found by the ``find()`` classmethod.
        """
        start_line, start_col = self.start_line, self.start_col
        lhs = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        rhs = assemble_continued_line(lines, (start_line, start_col), end_line)
        assert rhs.startswith('%'), rhs
        magic_name, _, args = rhs[1:].partition(' ')

        lines_before = lines[:start_line]
        call = "get_ipython().run_line_magic({!r}, {!r})".format(magic_name, args)
        new_line = lhs + call + '\n'
        lines_after = lines[end_line+1:]

        return lines_before + [new_line] + lines_after


class SystemAssign(TokenTransformBase):
    """Transformer for assignments from system commands (a = !foo)"""
    @classmethod
    def find_pre_312(cls, tokens_by_line):
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if (assign_ix is not None) \
                    and not line[assign_ix].line.strip().startswith('=') \
                    and (len(line) >= assign_ix + 2) \
                    and (line[assign_ix + 1].type == tokenize.ERRORTOKEN):
                ix = assign_ix + 1

                while ix < len(line) and line[ix].type == tokenize.ERRORTOKEN:
                    if line[ix].string == '!':
                        return cls(line[ix].start)
                    elif not line[ix].string.isspace():
                        break
                    ix += 1

    @classmethod
    def find_post_312(cls, tokens_by_line):
        for line in tokens_by_line:
            assign_ix = _find_assign_op(line)
            if (
                (assign_ix is not None)
                and not line[assign_ix].line.strip().startswith("=")
                and (len(line) >= assign_ix + 2)
                and (line[assign_ix + 1].type == tokenize.OP)
                and (line[assign_ix + 1].string == "!")
            ):
                return cls(line[assign_ix + 1].start)

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first system assignment (a = !foo) in the cell."""
        if sys.version_info < (3, 12):
            return cls.find_pre_312(tokens_by_line)
        return cls.find_post_312(tokens_by_line)

    def transform(self, lines: List[str]):
        """Transform a system assignment found by the ``find()`` classmethod.
        """
        start_line, start_col = self.start_line, self.start_col

        lhs = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        rhs = assemble_continued_line(lines, (start_line, start_col), end_line)
        assert rhs.startswith('!'), rhs
        cmd = rhs[1:]

        lines_before = lines[:start_line]
        call = "get_ipython().getoutput({!r})".format(cmd)
        new_line = lhs + call + '\n'
        lines_after = lines[end_line + 1:]

        return lines_before + [new_line] + lines_after

# The escape sequences that define the syntax transformations IPython will
# apply to user input.  These can NOT be just changed here: many regular
# expressions and other parts of the code may use their hardcoded values, and
# for all intents and purposes they constitute the 'IPython syntax', so they
# should be considered fixed.

ESC_SHELL  = '!'     # Send line to underlying system shell
ESC_SH_CAP = '!!'    # Send line to system shell and capture output
ESC_HELP   = '?'     # Find information about object
ESC_HELP2  = '??'    # Find extra-detailed information about object
ESC_MAGIC  = '%'     # Call magic function
ESC_MAGIC2 = '%%'    # Call cell-magic function
ESC_QUOTE  = ','     # Split args on whitespace, quote each as string and call
ESC_QUOTE2 = ';'     # Quote all args as a single string, call
ESC_PAREN  = '/'     # Call first argument with rest of line as arguments

ESCAPE_SINGLES = {'!', '?', '%', ',', ';', '/'}
ESCAPE_DOUBLES = {'!!', '??'}  # %% (cell magic) is handled separately

def _make_help_call(target, esc):
    """Prepares a pinfo(2)/psearch call from a target name and the escape
    (i.e. ? or ??)"""
    method  = 'pinfo2' if esc == '??' \
                else 'psearch' if '*' in target \
                else 'pinfo'
    arg = " ".join([method, target])
    #Prepare arguments for get_ipython().run_line_magic(magic_name, magic_args)
    t_magic_name, _, t_magic_arg_s = arg.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return "get_ipython().run_line_magic(%r, %r)" % (t_magic_name, t_magic_arg_s)


def _tr_help(content):
    """Translate lines escaped with: ?

    A naked help line should fire the intro help screen (shell.show_usage())
    """
    if not content:
        return 'get_ipython().show_usage()'

    return _make_help_call(content, '?')

def _tr_help2(content):
    """Translate lines escaped with: ??

    A naked help line should fire the intro help screen (shell.show_usage())
    """
    if not content:
        return 'get_ipython().show_usage()'

    return _make_help_call(content, '??')

def _tr_magic(content):
    "Translate lines escaped with a percent sign: %"
    name, _, args = content.partition(' ')
    return 'get_ipython().run_line_magic(%r, %r)' % (name, args)

def _tr_quote(content):
    "Translate lines escaped with a comma: ,"
    name, _, args = content.partition(' ')
    return '%s("%s")' % (name, '", "'.join(args.split()) )

def _tr_quote2(content):
    "Translate lines escaped with a semicolon: ;"
    name, _, args = content.partition(' ')
    return '%s("%s")' % (name, args)

def _tr_paren(content):
    "Translate lines escaped with a slash: /"
    name, _, args = content.partition(' ')
    return '%s(%s)' % (name, ", ".join(args.split()))

tr = { ESC_SHELL  : 'get_ipython().system({!r})'.format,
       ESC_SH_CAP : 'get_ipython().getoutput({!r})'.format,
       ESC_HELP   : _tr_help,
       ESC_HELP2  : _tr_help2,
       ESC_MAGIC  : _tr_magic,
       ESC_QUOTE  : _tr_quote,
       ESC_QUOTE2 : _tr_quote2,
       ESC_PAREN  : _tr_paren }

class EscapedCommand(TokenTransformBase):
    """Transformer for escaped commands like %foo, !foo, or /foo"""
    @classmethod
    def find(cls, tokens_by_line):
        """Find the first escaped command (%foo, !foo, etc.) in the cell.
        """
        for line in tokens_by_line:
            if not line:
                continue
            ix = 0
            ll = len(line)
            while ll > ix and line[ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                ix += 1
            if ix >= ll:
                continue
            if line[ix].string in ESCAPE_SINGLES:
                return cls(line[ix].start)

    def transform(self, lines):
        """Transform an escaped line found by the ``find()`` classmethod.
        """
        start_line, start_col = self.start_line, self.start_col

        indent = lines[start_line][:start_col]
        end_line = find_end_of_continued_line(lines, start_line)
        line = assemble_continued_line(lines, (start_line, start_col), end_line)

        if len(line) > 1 and line[:2] in ESCAPE_DOUBLES:
            escape, content = line[:2], line[2:]
        else:
            escape, content = line[:1], line[1:]

        if escape in tr:
            call = tr[escape](content)
        else:
            call = ''

        lines_before = lines[:start_line]
        new_line = indent + call + '\n'
        lines_after = lines[end_line + 1:]

        return lines_before + [new_line] + lines_after


_help_end_re = re.compile(
    r"""(%{0,2}
    (?!\d)[\w*]+            # Variable name
    (\.(?!\d)[\w*]+|\[-?[0-9]+\])*       # .etc.etc or [0], we only support literal integers.
    )
    (\?\??)$                # ? or ??
    """,
    re.VERBOSE,
)


class HelpEnd(TokenTransformBase):
    """Transformer for help syntax: obj? and obj??"""
    # This needs to be higher priority (lower number) than EscapedCommand so
    # that inspecting magics (%foo?) works.
    priority = 5

    def __init__(self, start, q_locn):
        super().__init__(start)
        self.q_line = q_locn[0] - 1  # Shift from 1-indexed to 0-indexed
        self.q_col = q_locn[1]

    @classmethod
    def find(cls, tokens_by_line):
        """Find the first help command (foo?) in the cell.
        """
        for line in tokens_by_line:
            # Last token is NEWLINE; look at last but one
            if len(line) > 2 and line[-2].string == '?':
                # Find the first token that's not INDENT/DEDENT
                ix = 0
                while line[ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                    ix += 1
                return cls(line[ix].start, line[-2].start)

    def transform(self, lines):
        """Transform a help command found by the ``find()`` classmethod.
        """

        piece = "".join(lines[self.start_line : self.q_line + 1])
        indent, content = piece[: self.start_col], piece[self.start_col :]
        lines_before = lines[: self.start_line]
        lines_after = lines[self.q_line + 1 :]

        m = _help_end_re.search(content)
        if not m:
            raise SyntaxError(content)
        assert m is not None, content
        target = m.group(1)
        esc = m.group(3)


        call = _make_help_call(target, esc)
        new_line = indent + call + '\n'

        return lines_before + [new_line] + lines_after

def make_tokens_by_line(lines:List[str]):
    """Tokenize a series of lines and group tokens by line.

    The tokens for a multiline Python string or expression are grouped as one
    line. All lines except the last lines should keep their line ending ('\\n',
    '\\r\\n') for this to properly work. Use `.splitlines(keeplineending=True)`
    for example when passing block of text to this function.

    """
    # NL tokens are used inside multiline expressions, but also after blank
    # lines or comments. This is intentional - see https://bugs.python.org/issue17061
    # We want to group the former case together but split the latter, so we
    # track parentheses level, similar to the internals of tokenize.

    #   reexported from token on 3.7+
    NEWLINE, NL = tokenize.NEWLINE, tokenize.NL  # type: ignore
    tokens_by_line: List[List[Any]] = [[]]
    if len(lines) > 1 and not lines[0].endswith(("\n", "\r", "\r\n", "\x0b", "\x0c")):
        warnings.warn(
            "`make_tokens_by_line` received a list of lines which do not have lineending markers ('\\n', '\\r', '\\r\\n', '\\x0b', '\\x0c'), behavior will be unspecified",
            stacklevel=2,
        )
    parenlev = 0
    try:
        for token in tokenutil.generate_tokens_catch_errors(
            iter(lines).__next__, extra_errors_to_catch=["expected EOF"]
        ):
            tokens_by_line[-1].append(token)
            if (token.type == NEWLINE) \
                    or ((token.type == NL) and (parenlev <= 0)):
                tokens_by_line.append([])
            elif token.string in {'(', '[', '{'}:
                parenlev += 1
            elif token.string in {')', ']', '}'}:
                if parenlev > 0:
                    parenlev -= 1
    except tokenize.TokenError:
        # Input ended in a multiline string or expression. That's OK for us.
        pass


    if not tokens_by_line[-1]:
        tokens_by_line.pop()


    return tokens_by_line


def has_sunken_brackets(tokens: List[tokenize.TokenInfo]):
    """Check if the depth of brackets in the list of tokens drops below 0"""
    parenlev = 0
    for token in tokens:
        if token.string in {"(", "[", "{"}:
            parenlev += 1
        elif token.string in {")", "]", "}"}:
            parenlev -= 1
            if parenlev < 0:
                return True
    return False


def show_linewise_tokens(s: str):
    """For investigation and debugging"""
    warnings.warn(
        "show_linewise_tokens is deprecated since IPython 8.6",
        DeprecationWarning,
        stacklevel=2,
    )
    if not s.endswith("\n"):
        s += "\n"
    lines = s.splitlines(keepends=True)
    for line in make_tokens_by_line(lines):
        print("Line -------")
        for tokinfo in line:
            print(" ", tokinfo)

# Arbitrary limit to prevent getting stuck in infinite loops
TRANSFORM_LOOP_LIMIT = 500

class TransformerManager:
    """Applies various transformations to a cell or code block.

    The key methods for external use are ``transform_cell()``
    and ``check_complete()``.
    """
    def __init__(self):
        self.cleanup_transforms = [
            leading_empty_lines,
            leading_indent,
            classic_prompt,
            ipython_prompt,
        ]
        self.line_transforms = [
            cell_magic,
        ]
        self.token_transformers = [
            MagicAssign,
            SystemAssign,
            EscapedCommand,
            HelpEnd,
        ]

    def do_one_token_transform(self, lines):
        """Find and run the transform earliest in the code.

        Returns (changed, lines).

        This method is called repeatedly until changed is False, indicating
        that all available transformations are complete.

        The tokens following IPython special syntax might not be valid, so
        the transformed code is retokenised every time to identify the next
        piece of special syntax. Hopefully long code cells are mostly valid
        Python, not using lots of IPython special syntax, so this shouldn't be
        a performance issue.
        """
        tokens_by_line = make_tokens_by_line(lines)
        candidates = []
        for transformer_cls in self.token_transformers:
            transformer = transformer_cls.find(tokens_by_line)
            if transformer:
                candidates.append(transformer)

        if not candidates:
            # Nothing to transform
            return False, lines
        ordered_transformers = sorted(candidates, key=TokenTransformBase.sortby)
        for transformer in ordered_transformers:
            try:
                return True, transformer.transform(lines)
            except SyntaxError:
                pass
        return False, lines

    def do_token_transforms(self, lines):
        for _ in range(TRANSFORM_LOOP_LIMIT):
            changed, lines = self.do_one_token_transform(lines)
            if not changed:
                return lines

        raise RuntimeError("Input transformation still changing after "
                           "%d iterations. Aborting." % TRANSFORM_LOOP_LIMIT)

    def transform_cell(self, cell: str) -> str:
        """Transforms a cell of input code"""
        if not cell.endswith('\n'):
            cell += '\n'  # Ensure the cell has a trailing newline
        lines = cell.splitlines(keepends=True)
        for transform in self.cleanup_transforms + self.line_transforms:
            lines = transform(lines)

        lines = self.do_token_transforms(lines)
        return ''.join(lines)

    def check_complete(self, cell: str):
        """Return whether a block of code is ready to execute, or should be continued

        Parameters
        ----------
        cell : string
            Python input code, which can be multiline.

        Returns
        -------
        status : str
            One of 'complete', 'incomplete', or 'invalid' if source is not a
            prefix of valid code.
        indent_spaces : int or None
            The number of spaces by which to indent the next line of code. If
            status is not 'incomplete', this is None.
        """
        # Remember if the lines ends in a new line.
        ends_with_newline = False
        for character in reversed(cell):
            if character == '\n':
                ends_with_newline = True
                break
            elif character.strip():
                break
            else:
                continue

        if not ends_with_newline:
            # Append an newline for consistent tokenization
            # See https://bugs.python.org/issue33899
            cell += '\n'

        lines = cell.splitlines(keepends=True)

        if not lines:
            return 'complete', None

        for line in reversed(lines):
            if not line.strip():
                continue
            elif line.strip("\n").endswith("\\"):
                return "incomplete", find_last_indent(lines)
            else:
                break

        try:
            for transform in self.cleanup_transforms:
                if not getattr(transform, 'has_side_effects', False):
                    lines = transform(lines)
        except SyntaxError:
            return 'invalid', None

        if lines[0].startswith('%%'):
            # Special case for cell magics - completion marked by blank line
            if lines[-1].strip():
                return 'incomplete', find_last_indent(lines)
            else:
                return 'complete', None

        try:
            for transform in self.line_transforms:
                if not getattr(transform, 'has_side_effects', False):
                    lines = transform(lines)
            lines = self.do_token_transforms(lines)
        except SyntaxError:
            return 'invalid', None

        tokens_by_line = make_tokens_by_line(lines)

        # Bail if we got one line and there are more closing parentheses than
        # the opening ones
        if (
            len(lines) == 1
            and tokens_by_line
            and has_sunken_brackets(tokens_by_line[0])
        ):
            return "invalid", None

        if not tokens_by_line:
            return 'incomplete', find_last_indent(lines)

        if (
            tokens_by_line[-1][-1].type != tokenize.ENDMARKER
            and tokens_by_line[-1][-1].type != tokenize.ERRORTOKEN
        ):
            # We're in a multiline string or expression
            return 'incomplete', find_last_indent(lines)

        newline_types = {tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENDMARKER} # type: ignore

        # Pop the last line which only contains DEDENTs and ENDMARKER
        last_token_line = None
        if {t.type for t in tokens_by_line[-1]} in [
            {tokenize.DEDENT, tokenize.ENDMARKER},
            {tokenize.ENDMARKER}
        ] and len(tokens_by_line) > 1:
            last_token_line = tokens_by_line.pop()

        while tokens_by_line[-1] and tokens_by_line[-1][-1].type in newline_types:
            tokens_by_line[-1].pop()

        if not tokens_by_line[-1]:
            return 'incomplete', find_last_indent(lines)

        if tokens_by_line[-1][-1].string == ':':
            # The last line starts a block (e.g. 'if foo:')
            ix = 0
            while tokens_by_line[-1][ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                ix += 1

            indent = tokens_by_line[-1][ix].start[1]
            return 'incomplete', indent + 4

        if tokens_by_line[-1][0].line.endswith('\\'):
            return 'incomplete', None

        # At this point, our checks think the code is complete (or invalid).
        # We'll use codeop.compile_command to check this with the real parser
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                res = compile_command(''.join(lines), symbol='exec')
        except (SyntaxError, OverflowError, ValueError, TypeError,
                MemoryError, SyntaxWarning):
            return 'invalid', None
        else:
            if res is None:
                return 'incomplete', find_last_indent(lines)

        if last_token_line and last_token_line[0].type == tokenize.DEDENT:
            if ends_with_newline:
                return 'complete', None
            return 'incomplete', find_last_indent(lines)

        # If there's a blank line at the end, assume we're ready to execute
        if not lines[-1].strip():
            return 'complete', None

        return 'complete', None


def find_last_indent(lines):
    m = _indent_re.match(lines[-1])
    if not m:
        return 0
    return len(m.group(0).replace('\t', ' '*4))


class MaybeAsyncCompile(Compile):
    def __init__(self, extra_flags=0):
        super().__init__()
        self.flags |= extra_flags


class MaybeAsyncCommandCompiler(CommandCompiler):
    def __init__(self, extra_flags=0):
        self.compiler = MaybeAsyncCompile(extra_flags=extra_flags)


_extra_flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

compile_command = MaybeAsyncCommandCompiler(extra_flags=_extra_flags)
