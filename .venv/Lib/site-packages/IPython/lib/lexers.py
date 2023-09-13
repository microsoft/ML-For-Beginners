# -*- coding: utf-8 -*-
"""
Defines a variety of Pygments lexers for highlighting IPython code.

This includes:

    IPythonLexer, IPython3Lexer
        Lexers for pure IPython (python + magic/shell commands)

    IPythonPartialTracebackLexer, IPythonTracebackLexer
        Supports 2.x and 3.x via keyword `python3`.  The partial traceback
        lexer reads everything but the Python code appearing in a traceback.
        The full lexer combines the partial lexer with an IPython lexer.

    IPythonConsoleLexer
        A lexer for IPython console sessions, with support for tracebacks.

    IPyLexer
        A friendly lexer which examines the first line of text and from it,
        decides whether to use an IPython lexer or an IPython console lexer.
        This is probably the only lexer that needs to be explicitly added
        to Pygments.

"""
#-----------------------------------------------------------------------------
# Copyright (c) 2013, the IPython Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

# Standard library
import re

# Third party
from pygments.lexers import (
    BashLexer, HtmlLexer, JavascriptLexer, RubyLexer, PerlLexer, PythonLexer,
    Python3Lexer, TexLexer)
from pygments.lexer import (
    Lexer, DelegatingLexer, RegexLexer, do_insertions, bygroups, using,
)
from pygments.token import (
    Generic, Keyword, Literal, Name, Operator, Other, Text, Error,
)
from pygments.util import get_bool_opt

# Local

line_re = re.compile('.*?\n')

__all__ = ['build_ipy_lexer', 'IPython3Lexer', 'IPythonLexer',
           'IPythonPartialTracebackLexer', 'IPythonTracebackLexer',
           'IPythonConsoleLexer', 'IPyLexer']


def build_ipy_lexer(python3):
    """Builds IPython lexers depending on the value of `python3`.

    The lexer inherits from an appropriate Python lexer and then adds
    information about IPython specific keywords (i.e. magic commands,
    shell commands, etc.)

    Parameters
    ----------
    python3 : bool
        If `True`, then build an IPython lexer from a Python 3 lexer.

    """
    # It would be nice to have a single IPython lexer class which takes
    # a boolean `python3`.  But since there are two Python lexer classes,
    # we will also have two IPython lexer classes.
    if python3:
        PyLexer = Python3Lexer
        name = 'IPython3'
        aliases = ['ipython3']
        doc = """IPython3 Lexer"""
    else:
        PyLexer = PythonLexer
        name = 'IPython'
        aliases = ['ipython2', 'ipython']
        doc = """IPython Lexer"""

    ipython_tokens = [
       (r'(?s)(\s*)(%%capture)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%debug)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?is)(\s*)(%%html)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(HtmlLexer))),
        (r'(?s)(\s*)(%%javascript)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))),
        (r'(?s)(\s*)(%%js)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))),
        (r'(?s)(\s*)(%%latex)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(TexLexer))),
        (r'(?s)(\s*)(%%perl)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PerlLexer))),
        (r'(?s)(\s*)(%%prun)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%pypy)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%python)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%python2)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PythonLexer))),
        (r'(?s)(\s*)(%%python3)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(Python3Lexer))),
        (r'(?s)(\s*)(%%ruby)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(RubyLexer))),
        (r'(?s)(\s*)(%%time)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%timeit)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%writefile)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r'(?s)(\s*)(%%file)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))),
        (r"(?s)(\s*)(%%)(\w+)(.*)", bygroups(Text, Operator, Keyword, Text)),
        (r'(?s)(^\s*)(%%!)([^\n]*\n)(.*)', bygroups(Text, Operator, Text, using(BashLexer))),
        (r"(%%?)(\w+)(\?\??)$",  bygroups(Operator, Keyword, Operator)),
        (r"\b(\?\??)(\s*)$",  bygroups(Operator, Text)),
        (r'(%)(sx|sc|system)(.*)(\n)', bygroups(Operator, Keyword,
                                                using(BashLexer), Text)),
        (r'(%)(\w+)(.*\n)', bygroups(Operator, Keyword, Text)),
        (r'^(!!)(.+)(\n)', bygroups(Operator, using(BashLexer), Text)),
        (r'(!)(?!=)(.+)(\n)', bygroups(Operator, using(BashLexer), Text)),
        (r'^(\s*)(\?\??)(\s*%{0,2}[\w\.\*]*)', bygroups(Text, Operator, Text)),
        (r'(\s*%{0,2}[\w\.\*]*)(\?\??)(\s*)$', bygroups(Text, Operator, Text)),
    ]

    tokens = PyLexer.tokens.copy()
    tokens['root'] = ipython_tokens + tokens['root']

    attrs = {'name': name, 'aliases': aliases, 'filenames': [],
             '__doc__': doc, 'tokens': tokens}

    return type(name, (PyLexer,), attrs)


IPython3Lexer = build_ipy_lexer(python3=True)
IPythonLexer = build_ipy_lexer(python3=False)


class IPythonPartialTracebackLexer(RegexLexer):
    """
    Partial lexer for IPython tracebacks.

    Handles all the non-python output.

    """
    name = 'IPython Partial Traceback'

    tokens = {
        'root': [
            # Tracebacks for syntax errors have a different style.
            # For both types of tracebacks, we mark the first line with
            # Generic.Traceback.  For syntax errors, we mark the filename
            # as we mark the filenames for non-syntax tracebacks.
            #
            # These two regexps define how IPythonConsoleLexer finds a
            # traceback.
            #
            ## Non-syntax traceback
            (r'^(\^C)?(-+\n)', bygroups(Error, Generic.Traceback)),
            ## Syntax traceback
            (r'^(  File)(.*)(, line )(\d+\n)',
             bygroups(Generic.Traceback, Name.Namespace,
                      Generic.Traceback, Literal.Number.Integer)),

            # (Exception Identifier)(Whitespace)(Traceback Message)
            (r'(?u)(^[^\d\W]\w*)(\s*)(Traceback.*?\n)',
             bygroups(Name.Exception, Generic.Whitespace, Text)),
            # (Module/Filename)(Text)(Callee)(Function Signature)
            # Better options for callee and function signature?
            (r'(.*)( in )(.*)(\(.*\)\n)',
             bygroups(Name.Namespace, Text, Name.Entity, Name.Tag)),
            # Regular line: (Whitespace)(Line Number)(Python Code)
            (r'(\s*?)(\d+)(.*?\n)',
             bygroups(Generic.Whitespace, Literal.Number.Integer, Other)),
            # Emphasized line: (Arrow)(Line Number)(Python Code)
            # Using Exception token so arrow color matches the Exception.
            (r'(-*>?\s?)(\d+)(.*?\n)',
             bygroups(Name.Exception, Literal.Number.Integer, Other)),
            # (Exception Identifier)(Message)
            (r'(?u)(^[^\d\W]\w*)(:.*?\n)',
             bygroups(Name.Exception, Text)),
            # Tag everything else as Other, will be handled later.
            (r'.*\n', Other),
        ],
    }


class IPythonTracebackLexer(DelegatingLexer):
    """
    IPython traceback lexer.

    For doctests, the tracebacks can be snipped as much as desired with the
    exception to the lines that designate a traceback. For non-syntax error
    tracebacks, this is the line of hyphens. For syntax error tracebacks,
    this is the line which lists the File and line number.

    """
    # The lexer inherits from DelegatingLexer.  The "root" lexer is an
    # appropriate IPython lexer, which depends on the value of the boolean
    # `python3`.  First, we parse with the partial IPython traceback lexer.
    # Then, any code marked with the "Other" token is delegated to the root
    # lexer.
    #
    name = 'IPython Traceback'
    aliases = ['ipythontb']

    def __init__(self, **options):
        """
        A subclass of `DelegatingLexer` which delegates to the appropriate to either IPyLexer,
        IPythonPartialTracebackLexer.
        """
        # note we need a __init__ doc, as otherwise it inherits the doc from the super class
        # which will fail the documentation build as it references section of the pygments docs that
        # do not exists when building IPython's docs.
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3tb']
        else:
            self.aliases = ['ipython2tb', 'ipythontb']

        if self.python3:
            IPyLexer = IPython3Lexer
        else:
            IPyLexer = IPythonLexer

        DelegatingLexer.__init__(self, IPyLexer,
                                 IPythonPartialTracebackLexer, **options)

class IPythonConsoleLexer(Lexer):
    """
    An IPython console lexer for IPython code-blocks and doctests, such as:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: a = 'foo'

            In [2]: a
            Out[2]: 'foo'

            In [3]: print(a)
            foo


    Support is also provided for IPython exceptions:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: raise Exception
            Traceback (most recent call last):
            ...
            Exception

    """
    name = 'IPython console session'
    aliases = ['ipythonconsole']
    mimetypes = ['text/x-ipython-console']

    # The regexps used to determine what is input and what is output.
    # The default prompts for IPython are:
    #
    #    in           = 'In [#]: '
    #    continuation = '   .D.: '
    #    template     = 'Out[#]: '
    #
    # Where '#' is the 'prompt number' or 'execution count' and 'D' 
    # D is a number of dots  matching the width of the execution count 
    #
    in1_regex = r'In \[[0-9]+\]: '
    in2_regex = r'   \.\.+\.: '
    out_regex = r'Out\[[0-9]+\]: '

    #: The regex to determine when a traceback starts.
    ipytb_start = re.compile(r'^(\^C)?(-+\n)|^(  File)(.*)(, line )(\d+\n)')

    def __init__(self, **options):
        """Initialize the IPython console lexer.

        Parameters
        ----------
        python3 : bool
            If `True`, then the console inputs are parsed using a Python 3
            lexer. Otherwise, they are parsed using a Python 2 lexer.
        in1_regex : RegexObject
            The compiled regular expression used to detect the start
            of inputs. Although the IPython configuration setting may have a
            trailing whitespace, do not include it in the regex. If `None`,
            then the default input prompt is assumed.
        in2_regex : RegexObject
            The compiled regular expression used to detect the continuation
            of inputs. Although the IPython configuration setting may have a
            trailing whitespace, do not include it in the regex. If `None`,
            then the default input prompt is assumed.
        out_regex : RegexObject
            The compiled regular expression used to detect outputs. If `None`,
            then the default output prompt is assumed.

        """
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3console']
        else:
            self.aliases = ['ipython2console', 'ipythonconsole']

        in1_regex = options.get('in1_regex', self.in1_regex)
        in2_regex = options.get('in2_regex', self.in2_regex)
        out_regex = options.get('out_regex', self.out_regex)

        # So that we can work with input and output prompts which have been
        # rstrip'd (possibly by editors) we also need rstrip'd variants. If
        # we do not do this, then such prompts will be tagged as 'output'.
        # The reason can't just use the rstrip'd variants instead is because
        # we want any whitespace associated with the prompt to be inserted
        # with the token. This allows formatted code to be modified so as hide
        # the appearance of prompts, with the whitespace included. One example
        # use of this is in copybutton.js from the standard lib Python docs.
        in1_regex_rstrip = in1_regex.rstrip() + '\n'
        in2_regex_rstrip = in2_regex.rstrip() + '\n'
        out_regex_rstrip = out_regex.rstrip() + '\n'

        # Compile and save them all.
        attrs = ['in1_regex', 'in2_regex', 'out_regex',
                 'in1_regex_rstrip', 'in2_regex_rstrip', 'out_regex_rstrip']
        for attr in attrs:
            self.__setattr__(attr, re.compile(locals()[attr]))

        Lexer.__init__(self, **options)

        if self.python3:
            pylexer = IPython3Lexer
            tblexer = IPythonTracebackLexer
        else:
            pylexer = IPythonLexer
            tblexer = IPythonTracebackLexer

        self.pylexer = pylexer(**options)
        self.tblexer = tblexer(**options)

        self.reset()

    def reset(self):
        self.mode = 'output'
        self.index = 0
        self.buffer = u''
        self.insertions = []

    def buffered_tokens(self):
        """
        Generator of unprocessed tokens after doing insertions and before
        changing to a new state.

        """
        if self.mode == 'output':
            tokens = [(0, Generic.Output, self.buffer)]
        elif self.mode == 'input':
            tokens = self.pylexer.get_tokens_unprocessed(self.buffer)
        else: # traceback
            tokens = self.tblexer.get_tokens_unprocessed(self.buffer)

        for i, t, v in do_insertions(self.insertions, tokens):
            # All token indexes are relative to the buffer.
            yield self.index + i, t, v

        # Clear it all
        self.index += len(self.buffer)
        self.buffer = u''
        self.insertions = []

    def get_mci(self, line):
        """
        Parses the line and returns a 3-tuple: (mode, code, insertion).

        `mode` is the next mode (or state) of the lexer, and is always equal
        to 'input', 'output', or 'tb'.

        `code` is a portion of the line that should be added to the buffer
        corresponding to the next mode and eventually lexed by another lexer.
        For example, `code` could be Python code if `mode` were 'input'.

        `insertion` is a 3-tuple (index, token, text) representing an
        unprocessed "token" that will be inserted into the stream of tokens
        that are created from the buffer once we change modes. This is usually
        the input or output prompt.

        In general, the next mode depends on current mode and on the contents
        of `line`.

        """
        # To reduce the number of regex match checks, we have multiple
        # 'if' blocks instead of 'if-elif' blocks.

        # Check for possible end of input
        in2_match = self.in2_regex.match(line)
        in2_match_rstrip = self.in2_regex_rstrip.match(line)
        if (in2_match and in2_match.group().rstrip() == line.rstrip()) or \
           in2_match_rstrip:
            end_input = True
        else:
            end_input = False
        if end_input and self.mode != 'tb':
            # Only look for an end of input when not in tb mode.
            # An ellipsis could appear within the traceback.
            mode = 'output'
            code = u''
            insertion = (0, Generic.Prompt, line)
            return mode, code, insertion

        # Check for output prompt
        out_match = self.out_regex.match(line)
        out_match_rstrip = self.out_regex_rstrip.match(line)
        if out_match or out_match_rstrip:
            mode = 'output'
            if out_match:
                idx = out_match.end()
            else:
                idx = out_match_rstrip.end()
            code = line[idx:]
            # Use the 'heading' token for output.  We cannot use Generic.Error
            # since it would conflict with exceptions.
            insertion = (0, Generic.Heading, line[:idx])
            return mode, code, insertion


        # Check for input or continuation prompt (non stripped version)
        in1_match = self.in1_regex.match(line)
        if in1_match or (in2_match and self.mode != 'tb'):
            # New input or when not in tb, continued input.
            # We do not check for continued input when in tb since it is
            # allowable to replace a long stack with an ellipsis.
            mode = 'input'
            if in1_match:
                idx = in1_match.end()
            else: # in2_match
                idx = in2_match.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return mode, code, insertion

        # Check for input or continuation prompt (stripped version)
        in1_match_rstrip = self.in1_regex_rstrip.match(line)
        if in1_match_rstrip or (in2_match_rstrip and self.mode != 'tb'):
            # New input or when not in tb, continued input.
            # We do not check for continued input when in tb since it is
            # allowable to replace a long stack with an ellipsis.
            mode = 'input'
            if in1_match_rstrip:
                idx = in1_match_rstrip.end()
            else: # in2_match
                idx = in2_match_rstrip.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return mode, code, insertion

        # Check for traceback
        if self.ipytb_start.match(line):
            mode = 'tb'
            code = line
            insertion = None
            return mode, code, insertion

        # All other stuff...
        if self.mode in ('input', 'output'):
            # We assume all other text is output. Multiline input that
            # does not use the continuation marker cannot be detected.
            # For example, the 3 in the following is clearly output:
            #
            #    In [1]: print 3
            #    3
            #
            # But the following second line is part of the input:
            #
            #    In [2]: while True:
            #        print True
            #
            # In both cases, the 2nd line will be 'output'.
            #
            mode = 'output'
        else:
            mode = 'tb'

        code = line
        insertion = None

        return mode, code, insertion

    def get_tokens_unprocessed(self, text):
        self.reset()
        for match in line_re.finditer(text):
            line = match.group()
            mode, code, insertion = self.get_mci(line)

            if mode != self.mode:
                # Yield buffered tokens before transitioning to new mode.
                for token in self.buffered_tokens():
                    yield token
                self.mode = mode

            if insertion:
                self.insertions.append((len(self.buffer), [insertion]))
            self.buffer += code

        for token in self.buffered_tokens():
            yield token

class IPyLexer(Lexer):
    r"""
    Primary lexer for all IPython-like code.

    This is a simple helper lexer.  If the first line of the text begins with
    "In \[[0-9]+\]:", then the entire text is parsed with an IPython console
    lexer. If not, then the entire text is parsed with an IPython lexer.

    The goal is to reduce the number of lexers that are registered
    with Pygments.

    """
    name = 'IPy session'
    aliases = ['ipy']

    def __init__(self, **options):
        """
        Create a new IPyLexer instance which dispatch to either an
        IPythonCOnsoleLexer (if In prompts are present) or and IPythonLexer (if
        In prompts are not present).
        """
        # init docstring is necessary for docs not to fail to build do to parent
        # docs referenceing a section in pygments docs.
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipy3']
        else:
            self.aliases = ['ipy2', 'ipy']

        Lexer.__init__(self, **options)

        self.IPythonLexer = IPythonLexer(**options)
        self.IPythonConsoleLexer = IPythonConsoleLexer(**options)

    def get_tokens_unprocessed(self, text):
        # Search for the input prompt anywhere...this allows code blocks to
        # begin with comments as well.
        if re.match(r'.*(In \[[0-9]+\]:)', text.strip(), re.DOTALL):
            lex = self.IPythonConsoleLexer
        else:
            lex = self.IPythonLexer
        for token in lex.get_tokens_unprocessed(text):
            yield token

