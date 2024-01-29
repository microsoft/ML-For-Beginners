"""DEPRECATED: Input transformer classes to support IPython special syntax.

This module was deprecated in IPython 7.0, in favour of inputtransformer2.

This includes the machinery to recognise and transform ``%magic`` commands,
``!system`` commands, ``help?`` querying, prompt stripping, and so forth.
"""
import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO

from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

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

ESC_SEQUENCES = [ESC_SHELL, ESC_SH_CAP, ESC_HELP ,\
                 ESC_HELP2, ESC_MAGIC, ESC_MAGIC2,\
                 ESC_QUOTE, ESC_QUOTE2, ESC_PAREN ]


class InputTransformer(metaclass=abc.ABCMeta):
    """Abstract base class for line-based input transformers."""
    
    @abc.abstractmethod
    def push(self, line):
        """Send a line of input to the transformer, returning the transformed
        input or None if the transformer is waiting for more input.

        Must be overridden by subclasses.

        Implementations may raise ``SyntaxError`` if the input is invalid. No
        other exceptions may be raised.
        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        """Return, transformed any lines that the transformer has accumulated,
        and reset its internal state.

        Must be overridden by subclasses.
        """
        pass
    
    @classmethod
    def wrap(cls, func):
        """Can be used by subclasses as a decorator, to return a factory that
        will allow instantiation with the decorated object.
        """
        @functools.wraps(func)
        def transformer_factory(**kwargs):
            return cls(func, **kwargs)  # type: ignore [call-arg]

        return transformer_factory

class StatelessInputTransformer(InputTransformer):
    """Wrapper for a stateless input transformer implemented as a function."""
    def __init__(self, func):
        self.func = func
    
    def __repr__(self):
        return "StatelessInputTransformer(func={0!r})".format(self.func)
    
    def push(self, line):
        """Send a line of input to the transformer, returning the
        transformed input."""
        return self.func(line)
    
    def reset(self):
        """No-op - exists for compatibility."""
        pass

class CoroutineInputTransformer(InputTransformer):
    """Wrapper for an input transformer implemented as a coroutine."""
    def __init__(self, coro, **kwargs):
        # Prime it
        self.coro = coro(**kwargs)
        next(self.coro)
    
    def __repr__(self):
        return "CoroutineInputTransformer(coro={0!r})".format(self.coro)
    
    def push(self, line):
        """Send a line of input to the transformer, returning the
        transformed input or None if the transformer is waiting for more
        input.
        """
        return self.coro.send(line)
    
    def reset(self):
        """Return, transformed any lines that the transformer has
        accumulated, and reset its internal state.
        """
        return self.coro.send(None)

class TokenInputTransformer(InputTransformer):
    """Wrapper for a token-based input transformer.
    
    func should accept a list of tokens (5-tuples, see tokenize docs), and
    return an iterable which can be passed to tokenize.untokenize().
    """
    def __init__(self, func):
        self.func = func
        self.buf = []
        self.reset_tokenizer()

    def reset_tokenizer(self):
        it = iter(self.buf)
        self.tokenizer = tokenutil.generate_tokens_catch_errors(it.__next__)

    def push(self, line):
        self.buf.append(line + '\n')
        if all(l.isspace() for l in self.buf):
            return self.reset()

        tokens = []
        stop_at_NL = False
        try:
            for intok in self.tokenizer:
                tokens.append(intok)
                t = intok[0]
                if t == tokenize.NEWLINE or (stop_at_NL and t == tokenize.NL):
                    # Stop before we try to pull a line we don't have yet
                    break
                elif t == tokenize.ERRORTOKEN:
                    stop_at_NL = True
        except TokenError:
            # Multi-line statement - stop and try again with the next line
            self.reset_tokenizer()
            return None
        
        return self.output(tokens)
    
    def output(self, tokens):
        self.buf.clear()
        self.reset_tokenizer()
        return untokenize(self.func(tokens)).rstrip('\n')
    
    def reset(self):
        l = ''.join(self.buf)
        self.buf.clear()
        self.reset_tokenizer()
        if l:
            return l.rstrip('\n')

class assemble_python_lines(TokenInputTransformer):
    def __init__(self):
        super(assemble_python_lines, self).__init__(None)
    
    def output(self, tokens):
        return self.reset()

@CoroutineInputTransformer.wrap
def assemble_logical_lines():
    r"""Join lines following explicit line continuations (\)"""
    line = ''
    while True:
        line = (yield line)
        if not line or line.isspace():
            continue
        
        parts = []
        while line is not None:
            if line.endswith('\\') and (not has_comment(line)):
                parts.append(line[:-1])
                line = (yield None) # Get another line
            else:
                parts.append(line)
                break
        
        # Output
        line = ''.join(parts)

# Utilities
def _make_help_call(target: str, esc: str, lspace: str) -> str:
    """Prepares a pinfo(2)/psearch call from a target name and the escape
    (i.e. ? or ??)"""
    method  = 'pinfo2' if esc == '??' \
                else 'psearch' if '*' in target \
                else 'pinfo'
    arg = " ".join([method, target])
    #Prepare arguments for get_ipython().run_line_magic(magic_name, magic_args)
    t_magic_name, _, t_magic_arg_s = arg.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return "%sget_ipython().run_line_magic(%r, %r)" % (
        lspace,
        t_magic_name,
        t_magic_arg_s,
    )


# These define the transformations for the different escape characters.
def _tr_system(line_info: LineInfo):
    "Translate lines escaped with: !"
    cmd = line_info.line.lstrip().lstrip(ESC_SHELL)
    return '%sget_ipython().system(%r)' % (line_info.pre, cmd)


def _tr_system2(line_info: LineInfo):
    "Translate lines escaped with: !!"
    cmd = line_info.line.lstrip()[2:]
    return '%sget_ipython().getoutput(%r)' % (line_info.pre, cmd)


def _tr_help(line_info: LineInfo):
    "Translate lines escaped with: ?/??"
    # A naked help line should just fire the intro help screen
    if not line_info.line[1:]:
        return 'get_ipython().show_usage()'

    return _make_help_call(line_info.ifun, line_info.esc, line_info.pre)


def _tr_magic(line_info: LineInfo):
    "Translate lines escaped with: %"
    tpl = '%sget_ipython().run_line_magic(%r, %r)'
    if line_info.line.startswith(ESC_MAGIC2):
        return line_info.line
    cmd = ' '.join([line_info.ifun, line_info.the_rest]).strip()
    #Prepare arguments for get_ipython().run_line_magic(magic_name, magic_args)
    t_magic_name, _, t_magic_arg_s = cmd.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return tpl % (line_info.pre, t_magic_name, t_magic_arg_s)


def _tr_quote(line_info: LineInfo):
    "Translate lines escaped with: ,"
    return '%s%s("%s")' % (line_info.pre, line_info.ifun,
                         '", "'.join(line_info.the_rest.split()) )


def _tr_quote2(line_info: LineInfo):
    "Translate lines escaped with: ;"
    return '%s%s("%s")' % (line_info.pre, line_info.ifun,
                           line_info.the_rest)


def _tr_paren(line_info: LineInfo):
    "Translate lines escaped with: /"
    return '%s%s(%s)' % (line_info.pre, line_info.ifun,
                         ", ".join(line_info.the_rest.split()))

tr = { ESC_SHELL  : _tr_system,
       ESC_SH_CAP : _tr_system2,
       ESC_HELP   : _tr_help,
       ESC_HELP2  : _tr_help,
       ESC_MAGIC  : _tr_magic,
       ESC_QUOTE  : _tr_quote,
       ESC_QUOTE2 : _tr_quote2,
       ESC_PAREN  : _tr_paren }

@StatelessInputTransformer.wrap
def escaped_commands(line: str):
    """Transform escaped commands - %magic, !system, ?help + various autocalls."""
    if not line or line.isspace():
        return line
    lineinf = LineInfo(line)
    if lineinf.esc not in tr:
        return line
    
    return tr[lineinf.esc](lineinf)

_initial_space_re = re.compile(r'\s*')

_help_end_re = re.compile(r"""(%{0,2}
                              (?!\d)[\w*]+            # Variable name
                              (\.(?!\d)[\w*]+)*       # .etc.etc
                              )
                              (\?\??)$                # ? or ??
                              """,
                              re.VERBOSE)

# Extra pseudotokens for multiline strings and data structures
_MULTILINE_STRING = object()
_MULTILINE_STRUCTURE = object()

def _line_tokens(line):
    """Helper for has_comment and ends_in_comment_or_string."""
    readline = StringIO(line).readline
    toktypes = set()
    try:
        for t in tokenutil.generate_tokens_catch_errors(readline):
            toktypes.add(t[0])
    except TokenError as e:
        # There are only two cases where a TokenError is raised.
        if 'multi-line string' in e.args[0]:
            toktypes.add(_MULTILINE_STRING)
        else:
            toktypes.add(_MULTILINE_STRUCTURE)
    return toktypes

def has_comment(src):
    """Indicate whether an input line has (i.e. ends in, or is) a comment.

    This uses tokenize, so it can distinguish comments from # inside strings.

    Parameters
    ----------
    src : string
        A single line input string.

    Returns
    -------
    comment : bool
        True if source has a comment.
    """
    return (tokenize.COMMENT in _line_tokens(src))

def ends_in_comment_or_string(src):
    """Indicates whether or not an input line ends in a comment or within
    a multiline string.

    Parameters
    ----------
    src : string
        A single line input string.

    Returns
    -------
    comment : bool
        True if source ends in a comment or multiline string.
    """
    toktypes = _line_tokens(src)
    return (tokenize.COMMENT in toktypes) or (_MULTILINE_STRING in toktypes)
        

@StatelessInputTransformer.wrap
def help_end(line: str):
    """Translate lines with ?/?? at the end"""
    m = _help_end_re.search(line)
    if m is None or ends_in_comment_or_string(line):
        return line
    target = m.group(1)
    esc = m.group(3)
    match = _initial_space_re.match(line)
    assert match is not None
    lspace = match.group(0)

    return _make_help_call(target, esc, lspace)


@CoroutineInputTransformer.wrap
def cellmagic(end_on_blank_line: bool = False):
    """Captures & transforms cell magics.

    After a cell magic is started, this stores up any lines it gets until it is
    reset (sent None).
    """
    tpl = 'get_ipython().run_cell_magic(%r, %r, %r)'
    cellmagic_help_re = re.compile(r'%%\w+\?')
    line = ''
    while True:
        line = (yield line)
        # consume leading empty lines
        while not line:
            line = (yield line)
        
        if not line.startswith(ESC_MAGIC2):
            # This isn't a cell magic, idle waiting for reset then start over
            while line is not None:
                line = (yield line)
            continue
        
        if cellmagic_help_re.match(line):
            # This case will be handled by help_end
            continue
        
        first = line
        body = []
        line = (yield None)
        while (line is not None) and \
                                ((line.strip() != '') or not end_on_blank_line):
            body.append(line)
            line = (yield None)
        
        # Output
        magic_name, _, first = first.partition(' ')
        magic_name = magic_name.lstrip(ESC_MAGIC2)
        line = tpl % (magic_name, first, u'\n'.join(body))


def _strip_prompts(prompt_re, initial_re=None, turnoff_re=None):
    """Remove matching input prompts from a block of input.

    Parameters
    ----------
    prompt_re : regular expression
        A regular expression matching any input prompt (including continuation)
    initial_re : regular expression, optional
        A regular expression matching only the initial prompt, but not continuation.
        If no initial expression is given, prompt_re will be used everywhere.
        Used mainly for plain Python prompts, where the continuation prompt
        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

    Notes
    -----
    If `initial_re` and `prompt_re differ`,
    only `initial_re` will be tested against the first line.
    If any prompt is found on the first two lines,
    prompts will be stripped from the rest of the block.
    """
    if initial_re is None:
        initial_re = prompt_re
    line = ''
    while True:
        line = (yield line)
        
        # First line of cell
        if line is None:
            continue
        out, n1 = initial_re.subn('', line, count=1)
        if turnoff_re and not n1:
            if turnoff_re.match(line):
                # We're in e.g. a cell magic; disable this transformer for
                # the rest of the cell.
                while line is not None:
                    line = (yield line)
                continue

        line = (yield out)
        
        if line is None:
            continue
        # check for any prompt on the second line of the cell,
        # because people often copy from just after the first prompt,
        # so we might not see it in the first line.
        out, n2 = prompt_re.subn('', line, count=1)
        line = (yield out)
        
        if n1 or n2:
            # Found a prompt in the first two lines - check for it in
            # the rest of the cell as well.
            while line is not None:
                line = (yield prompt_re.sub('', line, count=1))
        
        else:
            # Prompts not in input - wait for reset
            while line is not None:
                line = (yield line)

@CoroutineInputTransformer.wrap
def classic_prompt():
    """Strip the >>>/... prompts of the Python interactive shell."""
    # FIXME: non-capturing version (?:...) usable?
    prompt_re = re.compile(r'^(>>>|\.\.\.)( |$)')
    initial_re = re.compile(r'^>>>( |$)')
    # Any %magic/!system is IPython syntax, so we needn't look for >>> prompts
    turnoff_re = re.compile(r'^[%!]')
    return _strip_prompts(prompt_re, initial_re, turnoff_re)

@CoroutineInputTransformer.wrap
def ipy_prompt():
    """Strip IPython's In [1]:/...: prompts."""
    # FIXME: non-capturing version (?:...) usable?
    prompt_re = re.compile(r'^(In \[\d+\]: |\s*\.{3,}: ?)')
    # Disable prompt stripping inside cell magics
    turnoff_re = re.compile(r'^%%')
    return _strip_prompts(prompt_re, turnoff_re=turnoff_re)


@CoroutineInputTransformer.wrap
def leading_indent():
    """Remove leading indentation.

    If the first line starts with a spaces or tabs, the same whitespace will be
    removed from each following line until it is reset.
    """
    space_re = re.compile(r'^[ \t]+')
    line = ''
    while True:
        line = (yield line)
        
        if line is None:
            continue
        
        m = space_re.match(line)
        if m:
            space = m.group(0)
            while line is not None:
                if line.startswith(space):
                    line = line[len(space):]
                line = (yield line)
        else:
            # No leading spaces - wait for reset
            while line is not None:
                line = (yield line)


_assign_pat = \
r'''(?P<lhs>(\s*)
    ([\w\.]+)                # Initial identifier
    (\s*,\s*
        \*?[\w\.]+)*         # Further identifiers for unpacking
    \s*?,?                   # Trailing comma
    )
    \s*=\s*
'''

assign_system_re = re.compile(r'{}!\s*(?P<cmd>.*)'.format(_assign_pat), re.VERBOSE)
assign_system_template = '%s = get_ipython().getoutput(%r)'
@StatelessInputTransformer.wrap
def assign_from_system(line):
    """Transform assignment from system commands (e.g. files = !ls)"""
    m = assign_system_re.match(line)
    if m is None:
        return line
    
    return assign_system_template % m.group('lhs', 'cmd')

assign_magic_re = re.compile(r'{}%\s*(?P<cmd>.*)'.format(_assign_pat), re.VERBOSE)
assign_magic_template = '%s = get_ipython().run_line_magic(%r, %r)'
@StatelessInputTransformer.wrap
def assign_from_magic(line):
    """Transform assignment from magic commands (e.g. a = %who_ls)"""
    m = assign_magic_re.match(line)
    if m is None:
        return line
    #Prepare arguments for get_ipython().run_line_magic(magic_name, magic_args)
    m_lhs, m_cmd = m.group('lhs', 'cmd')
    t_magic_name, _, t_magic_arg_s = m_cmd.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return assign_magic_template % (m_lhs, t_magic_name, t_magic_arg_s)
