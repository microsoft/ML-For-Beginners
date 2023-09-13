"""DEPRECATED: Input handling and transformation machinery.

This module was deprecated in IPython 7.0, in favour of inputtransformer2.

The first class in this module, :class:`InputSplitter`, is designed to tell when
input from a line-oriented frontend is complete and should be executed, and when
the user should be prompted for another line of code instead. The name 'input
splitter' is largely for historical reasons.

A companion, :class:`IPythonInputSplitter`, provides the same functionality but
with full support for the extended IPython syntax (magics, system calls, etc).
The code to actually do these transformations is in :mod:`IPython.core.inputtransformer`.
:class:`IPythonInputSplitter` feeds the raw code to the transformers in order
and stores the results.

For more details, see the class docstrings below.
"""

from warnings import warn

warn('IPython.core.inputsplitter is deprecated since IPython 7 in favor of `IPython.core.inputtransformer2`',
     DeprecationWarning)

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings

from typing import List

from IPython.core.inputtransformer import (leading_indent,
                                           classic_prompt,
                                           ipy_prompt,
                                           cellmagic,
                                           assemble_logical_lines,
                                           help_end,
                                           escaped_commands,
                                           assign_from_magic,
                                           assign_from_system,
                                           assemble_python_lines,
                                           )
from IPython.utils import tokenutil

# These are available in this module for backwards compatibility.
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
                                        ESC_HELP2, ESC_MAGIC, ESC_MAGIC2,
                                        ESC_QUOTE, ESC_QUOTE2, ESC_PAREN, ESC_SEQUENCES)

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------

# FIXME: These are general-purpose utilities that later can be moved to the
# general ward.  Kept here for now because we're being very strict about test
# coverage with this code, and this lets us ensure that we keep 100% coverage
# while developing.

# compiled regexps for autoindent management
dedent_re = re.compile('|'.join([
    r'^\s+raise(\s.*)?$', # raise statement (+ space + other stuff, maybe)
    r'^\s+raise\([^\)]*\).*$', # wacky raise with immediate open paren
    r'^\s+return(\s.*)?$', # normal return (+ space + other stuff, maybe)
    r'^\s+return\([^\)]*\).*$', # wacky return with immediate open paren
    r'^\s+pass\s*$', # pass (optionally followed by trailing spaces)
    r'^\s+break\s*$', # break (optionally followed by trailing spaces)
    r'^\s+continue\s*$', # continue (optionally followed by trailing spaces)
]))
ini_spaces_re = re.compile(r'^([ \t\r\f\v]+)')

# regexp to match pure comment lines so we don't accidentally insert 'if 1:'
# before pure comments
comment_line_re = re.compile(r'^\s*\#')


def num_ini_spaces(s):
    """Return the number of initial spaces in a string.

    Note that tabs are counted as a single space.  For now, we do *not* support
    mixing of tabs and spaces in the user's input.

    Parameters
    ----------
    s : string

    Returns
    -------
    n : int
    """

    ini_spaces = ini_spaces_re.match(s)
    if ini_spaces:
        return ini_spaces.end()
    else:
        return 0

# Fake token types for partial_tokenize:
INCOMPLETE_STRING = tokenize.N_TOKENS
IN_MULTILINE_STATEMENT = tokenize.N_TOKENS + 1

# The 2 classes below have the same API as TokenInfo, but don't try to look up
# a token type name that they won't find.
class IncompleteString:
    type = exact_type = INCOMPLETE_STRING
    def __init__(self, s, start, end, line):
        self.s = s
        self.start = start
        self.end = end
        self.line = line

class InMultilineStatement:
    type = exact_type = IN_MULTILINE_STATEMENT
    def __init__(self, pos, line):
        self.s = ''
        self.start = self.end = pos
        self.line = line

def partial_tokens(s):
    """Iterate over tokens from a possibly-incomplete string of code.

    This adds two special token types: INCOMPLETE_STRING and
    IN_MULTILINE_STATEMENT. These can only occur as the last token yielded, and
    represent the two main ways for code to be incomplete.
    """
    readline = io.StringIO(s).readline
    token = tokenize.TokenInfo(tokenize.NEWLINE, '', (1, 0), (1, 0), '')
    try:
        for token in tokenutil.generate_tokens_catch_errors(readline):
            yield token
    except tokenize.TokenError as e:
        # catch EOF error
        lines = s.splitlines(keepends=True)
        end = len(lines), len(lines[-1])
        if 'multi-line string' in e.args[0]:
            l, c = start = token.end
            s = lines[l-1][c:] + ''.join(lines[l:])
            yield IncompleteString(s, start, end, lines[-1])
        elif 'multi-line statement' in e.args[0]:
            yield InMultilineStatement(end, lines[-1])
        else:
            raise

def find_next_indent(code):
    """Find the number of spaces for the next line of indentation"""
    tokens = list(partial_tokens(code))
    if tokens[-1].type == tokenize.ENDMARKER:
        tokens.pop()
    if not tokens:
        return 0

    while tokens[-1].type in {
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.COMMENT,
        tokenize.ERRORTOKEN,
    }:
        tokens.pop()

    # Starting in Python 3.12, the tokenize module adds implicit newlines at the end
    # of input. We need to remove those if we're in a multiline statement
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        while tokens[-2].type in {tokenize.NL}:
            tokens.pop(-2)


    if tokens[-1].type == INCOMPLETE_STRING:
        # Inside a multiline string
        return 0

    # Find the indents used before
    prev_indents = [0]
    def _add_indent(n):
        if n != prev_indents[-1]:
            prev_indents.append(n)

    tokiter = iter(tokens)
    for tok in tokiter:
        if tok.type in {tokenize.INDENT, tokenize.DEDENT}:
            _add_indent(tok.end[1])
        elif (tok.type == tokenize.NL):
            try:
                _add_indent(next(tokiter).start[1])
            except StopIteration:
                break

    last_indent = prev_indents.pop()

    # If we've just opened a multiline statement (e.g. 'a = ['), indent more
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        if tokens[-2].exact_type in {tokenize.LPAR, tokenize.LSQB, tokenize.LBRACE}:
            return last_indent + 4
        return last_indent

    if tokens[-1].exact_type == tokenize.COLON:
        # Line ends with colon - indent
        return last_indent + 4

    if last_indent:
        # Examine the last line for dedent cues - statements like return or
        # raise which normally end a block of code.
        last_line_starts = 0
        for i, tok in enumerate(tokens):
            if tok.type == tokenize.NEWLINE:
                last_line_starts = i + 1

        last_line_tokens = tokens[last_line_starts:]
        names = [t.string for t in last_line_tokens if t.type == tokenize.NAME]
        if names and names[0] in {'raise', 'return', 'pass', 'break', 'continue'}:
            # Find the most recent indentation less than the current level
            for indent in reversed(prev_indents):
                if indent < last_indent:
                    return indent

    return last_indent


def last_blank(src):
    """Determine if the input source ends in a blank.

    A blank is either a newline or a line consisting of whitespace.

    Parameters
    ----------
    src : string
        A single or multiline string.
    """
    if not src: return False
    ll  = src.splitlines()[-1]
    return (ll == '') or ll.isspace()


last_two_blanks_re = re.compile(r'\n\s*\n\s*$', re.MULTILINE)
last_two_blanks_re2 = re.compile(r'.+\n\s*\n\s+$', re.MULTILINE)

def last_two_blanks(src):
    """Determine if the input source ends in two blanks.

    A blank is either a newline or a line consisting of whitespace.

    Parameters
    ----------
    src : string
        A single or multiline string.
    """
    if not src: return False
    # The logic here is tricky: I couldn't get a regexp to work and pass all
    # the tests, so I took a different approach: split the source by lines,
    # grab the last two and prepend '###\n' as a stand-in for whatever was in
    # the body before the last two lines.  Then, with that structure, it's
    # possible to analyze with two regexps.  Not the most elegant solution, but
    # it works.  If anyone tries to change this logic, make sure to validate
    # the whole test suite first!
    new_src = '\n'.join(['###\n'] + src.splitlines()[-2:])
    return (bool(last_two_blanks_re.match(new_src)) or
            bool(last_two_blanks_re2.match(new_src)) )


def remove_comments(src):
    """Remove all comments from input source.

    Note: comments are NOT recognized inside of strings!

    Parameters
    ----------
    src : string
        A single or multiline input string.

    Returns
    -------
    String with all Python comments removed.
    """

    return re.sub('#.*', '', src)


def get_input_encoding():
    """Return the default standard input encoding.

    If sys.stdin has no encoding, 'ascii' is returned."""
    # There are strange environments for which sys.stdin.encoding is None. We
    # ensure that a valid encoding is returned.
    encoding = getattr(sys.stdin, 'encoding', None)
    if encoding is None:
        encoding = 'ascii'
    return encoding

#-----------------------------------------------------------------------------
# Classes and functions for normal Python syntax handling
#-----------------------------------------------------------------------------

class InputSplitter(object):
    r"""An object that can accumulate lines of Python source before execution.

    This object is designed to be fed python source line-by-line, using
    :meth:`push`. It will return on each push whether the currently pushed
    code could be executed already. In addition, it provides a method called
    :meth:`push_accepts_more` that can be used to query whether more input
    can be pushed into a single interactive block.

    This is a simple example of how an interactive terminal-based client can use
    this tool::

        isp = InputSplitter()
        while isp.push_accepts_more():
            indent = ' '*isp.indent_spaces
            prompt = '>>> ' + indent
            line = indent + raw_input(prompt)
            isp.push(line)
        print 'Input source was:\n', isp.source_reset(),
    """
    # A cache for storing the current indentation
    # The first value stores the most recently processed source input
    # The second value is the number of spaces for the current indentation 
    # If self.source matches the first value, the second value is a valid
    # current indentation. Otherwise, the cache is invalid and the indentation
    # must be recalculated.
    _indent_spaces_cache = None, None
    # String, indicating the default input encoding.  It is computed by default
    # at initialization time via get_input_encoding(), but it can be reset by a
    # client with specific knowledge of the encoding.
    encoding = ''
    # String where the current full source input is stored, properly encoded.
    # Reading this attribute is the normal way of querying the currently pushed
    # source code, that has been properly encoded.
    source = ''
    # Code object corresponding to the current source.  It is automatically
    # synced to the source, so it can be queried at any time to obtain the code
    # object; it will be None if the source doesn't compile to valid Python.
    code = None

    # Private attributes

    # List with lines of input accumulated so far
    _buffer: List[str]
    # Command compiler
    _compile: codeop.CommandCompiler
    # Boolean indicating whether the current block is complete
    _is_complete = None
    # Boolean indicating whether the current block has an unrecoverable syntax error
    _is_invalid = False

    def __init__(self) -> None:
        """Create a new InputSplitter instance."""
        self._buffer = []
        self._compile = codeop.CommandCompiler()
        self.encoding = get_input_encoding()

    def reset(self):
        """Reset the input buffer and associated state."""
        self._buffer[:] = []
        self.source = ''
        self.code = None
        self._is_complete = False
        self._is_invalid = False

    def source_reset(self):
        """Return the input source and perform a full reset.
        """
        out = self.source
        self.reset()
        return out

    def check_complete(self, source):
        """Return whether a block of code is ready to execute, or should be continued

        This is a non-stateful API, and will reset the state of this InputSplitter.

        Parameters
        ----------
        source : string
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
        self.reset()
        try:
            self.push(source)
        except SyntaxError:
            # Transformers in IPythonInputSplitter can raise SyntaxError,
            # which push() will not catch.
            return 'invalid', None
        else:
            if self._is_invalid:
                return 'invalid', None
            elif self.push_accepts_more():
                return 'incomplete', self.get_indent_spaces()
            else:
                return 'complete', None
        finally:
            self.reset()

    def push(self, lines:str) -> bool:
        """Push one or more lines of input.

        This stores the given lines and returns a status code indicating
        whether the code forms a complete Python block or not.

        Any exceptions generated in compilation are swallowed, but if an
        exception was produced, the method returns True.

        Parameters
        ----------
        lines : string
            One or more lines of Python input.

        Returns
        -------
        is_complete : boolean
            True if the current input source (the result of the current input
            plus prior inputs) forms a complete Python execution block.  Note that
            this value is also stored as a private attribute (``_is_complete``), so it
            can be queried at any time.
        """
        assert isinstance(lines, str)
        self._store(lines)
        source = self.source

        # Before calling _compile(), reset the code object to None so that if an
        # exception is raised in compilation, we don't mislead by having
        # inconsistent code/source attributes.
        self.code, self._is_complete = None, None
        self._is_invalid = False

        # Honor termination lines properly
        if source.endswith('\\\n'):
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                self.code = self._compile(source, symbol="exec")
        # Invalid syntax can produce any of a number of different errors from
        # inside the compiler, so we have to catch them all.  Syntax errors
        # immediately produce a 'ready' block, so the invalid Python can be
        # sent to the kernel for evaluation with possible ipython
        # special-syntax conversion.
        except (SyntaxError, OverflowError, ValueError, TypeError,
                MemoryError, SyntaxWarning):
            self._is_complete = True
            self._is_invalid = True
        else:
            # Compilation didn't produce any exceptions (though it may not have
            # given a complete code object)
            self._is_complete = self.code is not None

        return self._is_complete

    def push_accepts_more(self):
        """Return whether a block of interactive input can accept more input.

        This method is meant to be used by line-oriented frontends, who need to
        guess whether a block is complete or not based solely on prior and
        current input lines.  The InputSplitter considers it has a complete
        interactive block and will not accept more input when either:

        * A SyntaxError is raised

        * The code is complete and consists of a single line or a single
          non-compound statement

        * The code is complete and has a blank line at the end

        If the current input produces a syntax error, this method immediately
        returns False but does *not* raise the syntax error exception, as
        typically clients will want to send invalid syntax to an execution
        backend which might convert the invalid syntax into valid Python via
        one of the dynamic IPython mechanisms.
        """

        # With incomplete input, unconditionally accept more
        # A syntax error also sets _is_complete to True - see push()
        if not self._is_complete:
            #print("Not complete")  # debug
            return True
        
        # The user can make any (complete) input execute by leaving a blank line
        last_line = self.source.splitlines()[-1]
        if (not last_line) or last_line.isspace():
            #print("Blank line")  # debug
            return False
        
        # If there's just a single line or AST node, and we're flush left, as is
        # the case after a simple statement such as 'a=1', we want to execute it
        # straight away.
        if self.get_indent_spaces() == 0:
            if len(self.source.splitlines()) <= 1:
                return False
            
            try:
                code_ast = ast.parse("".join(self._buffer))
            except Exception:
                #print("Can't parse AST")  # debug
                return False
            else:
                if len(code_ast.body) == 1 and \
                                    not hasattr(code_ast.body[0], 'body'):
                    #print("Simple statement")  # debug
                    return False

        # General fallback - accept more code
        return True

    def get_indent_spaces(self):
        sourcefor, n = self._indent_spaces_cache
        if sourcefor == self.source:
            return n

        # self.source always has a trailing newline
        n = find_next_indent(self.source[:-1])
        self._indent_spaces_cache = (self.source, n)
        return n

    # Backwards compatibility. I think all code that used .indent_spaces was
    # inside IPython, but we can leave this here until IPython 7 in case any
    # other modules are using it. -TK, November 2017
    indent_spaces = property(get_indent_spaces)

    def _store(self, lines, buffer=None, store='source'):
        """Store one or more lines of input.

        If input lines are not newline-terminated, a newline is automatically
        appended."""

        if buffer is None:
            buffer = self._buffer

        if lines.endswith('\n'):
            buffer.append(lines)
        else:
            buffer.append(lines+'\n')
        setattr(self, store, self._set_source(buffer))

    def _set_source(self, buffer):
        return u''.join(buffer)


class IPythonInputSplitter(InputSplitter):
    """An input splitter that recognizes all of IPython's special syntax."""

    # String with raw, untransformed input.
    source_raw = ''
    
    # Flag to track when a transformer has stored input that it hasn't given
    # back yet.
    transformer_accumulating = False
    
    # Flag to track when assemble_python_lines has stored input that it hasn't
    # given back yet.
    within_python_line = False

    # Private attributes

    # List with lines of raw input accumulated so far.
    _buffer_raw = None

    def __init__(self, line_input_checker=True, physical_line_transforms=None,
                    logical_line_transforms=None, python_line_transforms=None):
        super(IPythonInputSplitter, self).__init__()
        self._buffer_raw = []
        self._validate = True
        
        if physical_line_transforms is not None:
            self.physical_line_transforms = physical_line_transforms
        else:
            self.physical_line_transforms = [
                                             leading_indent(),
                                             classic_prompt(),
                                             ipy_prompt(),
                                             cellmagic(end_on_blank_line=line_input_checker),
                                            ]
        
        self.assemble_logical_lines = assemble_logical_lines()
        if logical_line_transforms is not None:
            self.logical_line_transforms = logical_line_transforms
        else:
            self.logical_line_transforms = [
                                            help_end(),
                                            escaped_commands(),
                                            assign_from_magic(),
                                            assign_from_system(),
                                           ]
        
        self.assemble_python_lines = assemble_python_lines()
        if python_line_transforms is not None:
            self.python_line_transforms = python_line_transforms
        else:
            # We don't use any of these at present
            self.python_line_transforms = []
    
    @property
    def transforms(self):
        "Quick access to all transformers."
        return self.physical_line_transforms + \
            [self.assemble_logical_lines] + self.logical_line_transforms + \
            [self.assemble_python_lines]  + self.python_line_transforms
    
    @property
    def transforms_in_use(self):
        """Transformers, excluding logical line transformers if we're in a
        Python line."""
        t = self.physical_line_transforms[:]
        if not self.within_python_line:
            t += [self.assemble_logical_lines] + self.logical_line_transforms
        return t + [self.assemble_python_lines] + self.python_line_transforms

    def reset(self):
        """Reset the input buffer and associated state."""
        super(IPythonInputSplitter, self).reset()
        self._buffer_raw[:] = []
        self.source_raw = ''
        self.transformer_accumulating = False
        self.within_python_line = False

        for t in self.transforms:
            try:
                t.reset()
            except SyntaxError:
                # Nothing that calls reset() expects to handle transformer
                # errors
                pass
    
    def flush_transformers(self):
        def _flush(transform, outs):
            """yield transformed lines

            always strings, never None

            transform: the current transform
            outs: an iterable of previously transformed inputs.
                 Each may be multiline, which will be passed
                 one line at a time to transform.
            """
            for out in outs:
                for line in out.splitlines():
                    # push one line at a time
                    tmp = transform.push(line)
                    if tmp is not None:
                        yield tmp
            
            # reset the transform
            tmp = transform.reset()
            if tmp is not None:
                yield tmp
        
        out = []
        for t in self.transforms_in_use:
            out = _flush(t, out)
        
        out = list(out)
        if out:
            self._store('\n'.join(out))

    def raw_reset(self):
        """Return raw input only and perform a full reset.
        """
        out = self.source_raw
        self.reset()
        return out
    
    def source_reset(self):
        try:
            self.flush_transformers()
            return self.source
        finally:
            self.reset()

    def push_accepts_more(self):
        if self.transformer_accumulating:
            return True
        else:
            return super(IPythonInputSplitter, self).push_accepts_more()

    def transform_cell(self, cell):
        """Process and translate a cell of input.
        """
        self.reset()
        try:
            self.push(cell)
            self.flush_transformers()
            return self.source
        finally:
            self.reset()

    def push(self, lines:str) -> bool:
        """Push one or more lines of IPython input.

        This stores the given lines and returns a status code indicating
        whether the code forms a complete Python block or not, after processing
        all input lines for special IPython syntax.

        Any exceptions generated in compilation are swallowed, but if an
        exception was produced, the method returns True.

        Parameters
        ----------
        lines : string
            One or more lines of Python input.

        Returns
        -------
        is_complete : boolean
            True if the current input source (the result of the current input
            plus prior inputs) forms a complete Python execution block.  Note that
            this value is also stored as a private attribute (_is_complete), so it
            can be queried at any time.
        """
        assert isinstance(lines, str)
        # We must ensure all input is pure unicode
        # ''.splitlines() --> [], but we need to push the empty line to transformers
        lines_list = lines.splitlines()
        if not lines_list:
            lines_list = ['']

        # Store raw source before applying any transformations to it.  Note
        # that this must be done *after* the reset() call that would otherwise
        # flush the buffer.
        self._store(lines, self._buffer_raw, 'source_raw')

        transformed_lines_list = []
        for line in lines_list:
            transformed = self._transform_line(line)
            if transformed is not None:
                transformed_lines_list.append(transformed)

        if transformed_lines_list:
            transformed_lines = '\n'.join(transformed_lines_list)
            return super(IPythonInputSplitter, self).push(transformed_lines)
        else:
            # Got nothing back from transformers - they must be waiting for
            # more input.
            return False

    def _transform_line(self, line):
        """Push a line of input code through the various transformers.

        Returns any output from the transformers, or None if a transformer
        is accumulating lines.

        Sets self.transformer_accumulating as a side effect.
        """
        def _accumulating(dbg):
            #print(dbg)
            self.transformer_accumulating = True
            return None

        for transformer in self.physical_line_transforms:
            line = transformer.push(line)
            if line is None:
                return _accumulating(transformer)

        if not self.within_python_line:
            line = self.assemble_logical_lines.push(line)
            if line is None:
                return _accumulating('acc logical line')

            for transformer in self.logical_line_transforms:
                line = transformer.push(line)
                if line is None:
                    return _accumulating(transformer)

        line = self.assemble_python_lines.push(line)
        if line is None:
            self.within_python_line = True
            return _accumulating('acc python line')
        else:
            self.within_python_line = False

        for transformer in self.python_line_transforms:
            line = transformer.push(line)
            if line is None:
                return _accumulating(transformer)

        #print("transformers clear") #debug
        self.transformer_accumulating = False
        return line

