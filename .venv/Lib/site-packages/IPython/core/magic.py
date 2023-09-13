# encoding: utf-8
"""Magic functions for InteractiveShell.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2001 Janko Hauser <jhauser@zscout.de> and
#  Copyright (C) 2001 Fernando Perez <fperez@colorado.edu>
#  Copyright (C) 2008 The IPython Development Team

#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

import os
import re
import sys
from getopt import getopt, GetoptError

from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# A dict we'll use for each class that has magics, used as temporary storage to
# pass information between the @line/cell_magic method decorators and the
# @magics_class class decorator, because the method decorators have no
# access to the class when they run.  See for more details:
# http://stackoverflow.com/questions/2366713/can-a-python-decorator-of-an-instance-method-access-the-class

magics = dict(line={}, cell={})

magic_kinds = ('line', 'cell')
magic_spec = ('line', 'cell', 'line_cell')
magic_escapes = dict(line=ESC_MAGIC, cell=ESC_MAGIC2)

#-----------------------------------------------------------------------------
# Utility classes and functions
#-----------------------------------------------------------------------------

class Bunch: pass


def on_off(tag):
    """Return an ON/OFF string for a 1/0 input. Simple utility function."""
    return ['OFF','ON'][tag]


def compress_dhist(dh):
    """Compress a directory history into a new one with at most 20 entries.

    Return a new list made from the first and last 10 elements of dhist after
    removal of duplicates.
    """
    head, tail = dh[:-10], dh[-10:]

    newhead = []
    done = set()
    for h in head:
        if h in done:
            continue
        newhead.append(h)
        done.add(h)

    return newhead + tail


def needs_local_scope(func):
    """Decorator to mark magic functions which need to local scope to run."""
    func.needs_local_scope = True
    return func

#-----------------------------------------------------------------------------
# Class and method decorators for registering magics
#-----------------------------------------------------------------------------

def magics_class(cls):
    """Class decorator for all subclasses of the main Magics class.

    Any class that subclasses Magics *must* also apply this decorator, to
    ensure that all the methods that have been decorated as line/cell magics
    get correctly registered in the class instance.  This is necessary because
    when method decorators run, the class does not exist yet, so they
    temporarily store their information into a module global.  Application of
    this class decorator copies that global data to the class instance and
    clears the global.

    Obviously, this mechanism is not thread-safe, which means that the
    *creation* of subclasses of Magic should only be done in a single-thread
    context.  Instantiation of the classes has no restrictions.  Given that
    these classes are typically created at IPython startup time and before user
    application code becomes active, in practice this should not pose any
    problems.
    """
    cls.registered = True
    cls.magics = dict(line = magics['line'],
                      cell = magics['cell'])
    magics['line'] = {}
    magics['cell'] = {}
    return cls


def record_magic(dct, magic_kind, magic_name, func):
    """Utility function to store a function as a magic of a specific kind.

    Parameters
    ----------
    dct : dict
        A dictionary with 'line' and 'cell' subdicts.
    magic_kind : str
        Kind of magic to be stored.
    magic_name : str
        Key to store the magic as.
    func : function
        Callable object to store.
    """
    if magic_kind == 'line_cell':
        dct['line'][magic_name] = dct['cell'][magic_name] = func
    else:
        dct[magic_kind][magic_name] = func


def validate_type(magic_kind):
    """Ensure that the given magic_kind is valid.

    Check that the given magic_kind is one of the accepted spec types (stored
    in the global `magic_spec`), raise ValueError otherwise.
    """
    if magic_kind not in magic_spec:
        raise ValueError('magic_kind must be one of %s, %s given' %
                         magic_kinds, magic_kind)


# The docstrings for the decorator below will be fairly similar for the two
# types (method and function), so we generate them here once and reuse the
# templates below.
_docstring_template = \
"""Decorate the given {0} as {1} magic.

The decorator can be used with or without arguments, as follows.

i) without arguments: it will create a {1} magic named as the {0} being
decorated::

    @deco
    def foo(...)

will create a {1} magic named `foo`.

ii) with one string argument: which will be used as the actual name of the
resulting magic::

    @deco('bar')
    def foo(...)

will create a {1} magic named `bar`.

To register a class magic use ``Interactiveshell.register_magic(class or instance)``.
"""

# These two are decorator factories.  While they are conceptually very similar,
# there are enough differences in the details that it's simpler to have them
# written as completely standalone functions rather than trying to share code
# and make a single one with convoluted logic.

def _method_magic_marker(magic_kind):
    """Decorator factory for methods in Magics subclasses.
    """

    validate_type(magic_kind)

    # This is a closure to capture the magic_kind.  We could also use a class,
    # but it's overkill for just that one bit of state.
    def magic_deco(arg):
        if callable(arg):
            # "Naked" decorator call (just @foo, no args)
            func = arg
            name = func.__name__
            retval = arg
            record_magic(magics, magic_kind, name, name)
        elif isinstance(arg, str):
            # Decorator called with arguments (@foo('bar'))
            name = arg
            def mark(func, *a, **kw):
                record_magic(magics, magic_kind, name, func.__name__)
                return func
            retval = mark
        else:
            raise TypeError("Decorator can only be called with "
                            "string or function")
        return retval

    # Ensure the resulting decorator has a usable docstring
    magic_deco.__doc__ = _docstring_template.format('method', magic_kind)
    return magic_deco


def _function_magic_marker(magic_kind):
    """Decorator factory for standalone functions.
    """
    validate_type(magic_kind)
    
    # This is a closure to capture the magic_kind.  We could also use a class,
    # but it's overkill for just that one bit of state.
    def magic_deco(arg):
        # Find get_ipython() in the caller's namespace
        caller = sys._getframe(1)
        for ns in ['f_locals', 'f_globals', 'f_builtins']:
            get_ipython = getattr(caller, ns).get('get_ipython')
            if get_ipython is not None:
                break
        else:
            raise NameError('Decorator can only run in context where '
                            '`get_ipython` exists')

        ip = get_ipython()

        if callable(arg):
            # "Naked" decorator call (just @foo, no args)
            func = arg
            name = func.__name__
            ip.register_magic_function(func, magic_kind, name)
            retval = arg
        elif isinstance(arg, str):
            # Decorator called with arguments (@foo('bar'))
            name = arg
            def mark(func, *a, **kw):
                ip.register_magic_function(func, magic_kind, name)
                return func
            retval = mark
        else:
            raise TypeError("Decorator can only be called with "
                             "string or function")
        return retval

    # Ensure the resulting decorator has a usable docstring
    ds = _docstring_template.format('function', magic_kind)

    ds += dedent("""
    Note: this decorator can only be used in a context where IPython is already
    active, so that the `get_ipython()` call succeeds.  You can therefore use
    it in your startup files loaded after IPython initializes, but *not* in the
    IPython configuration file itself, which is executed before IPython is
    fully up and running.  Any file located in the `startup` subdirectory of
    your configuration profile will be OK in this sense.
    """)
    
    magic_deco.__doc__ = ds
    return magic_deco


MAGIC_NO_VAR_EXPAND_ATTR = "_ipython_magic_no_var_expand"
MAGIC_OUTPUT_CAN_BE_SILENCED = "_ipython_magic_output_can_be_silenced"


def no_var_expand(magic_func):
    """Mark a magic function as not needing variable expansion

    By default, IPython interprets `{a}` or `$a` in the line passed to magics
    as variables that should be interpolated from the interactive namespace
    before passing the line to the magic function.
    This is not always desirable, e.g. when the magic executes Python code
    (%timeit, %time, etc.).
    Decorate magics with `@no_var_expand` to opt-out of variable expansion.

    .. versionadded:: 7.3
    """
    setattr(magic_func, MAGIC_NO_VAR_EXPAND_ATTR, True)
    return magic_func


def output_can_be_silenced(magic_func):
    """Mark a magic function so its output may be silenced.

    The output is silenced if the Python code used as a parameter of
    the magic ends in a semicolon, not counting a Python comment that can
    follow it.
    """
    setattr(magic_func, MAGIC_OUTPUT_CAN_BE_SILENCED, True)
    return magic_func

# Create the actual decorators for public use

# These three are used to decorate methods in class definitions
line_magic = _method_magic_marker('line')
cell_magic = _method_magic_marker('cell')
line_cell_magic = _method_magic_marker('line_cell')

# These three decorate standalone functions and perform the decoration
# immediately.  They can only run where get_ipython() works
register_line_magic = _function_magic_marker('line')
register_cell_magic = _function_magic_marker('cell')
register_line_cell_magic = _function_magic_marker('line_cell')

#-----------------------------------------------------------------------------
# Core Magic classes
#-----------------------------------------------------------------------------

class MagicsManager(Configurable):
    """Object that handles all magic-related functionality for IPython.
    """
    # Non-configurable class attributes

    # A two-level dict, first keyed by magic type, then by magic function, and
    # holding the actual callable object as value.  This is the dict used for
    # magic function dispatch
    magics = Dict()
    lazy_magics = Dict(
        help="""
    Mapping from magic names to modules to load.

    This can be used in IPython/IPykernel configuration to declare lazy magics
    that will only be imported/registered on first use.

    For example::

        c.MagicsManager.lazy_magics = {
          "my_magic": "slow.to.import",
          "my_other_magic": "also.slow",
        }

    On first invocation of `%my_magic`, `%%my_magic`, `%%my_other_magic` or
    `%%my_other_magic`, the corresponding module will be loaded as an ipython
    extensions as if you had previously done `%load_ext ipython`.

    Magics names should be without percent(s) as magics can be both cell
    and line magics.

    Lazy loading happen relatively late in execution process, and
    complex extensions that manipulate Python/IPython internal state or global state
    might not support lazy loading.
    """
    ).tag(
        config=True,
    )

    # A registry of the original objects that we've been given holding magics.
    registry = Dict()

    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    auto_magic = Bool(True, help=
        "Automatically call line magics without requiring explicit % prefix"
    ).tag(config=True)
    @observe('auto_magic')
    def _auto_magic_changed(self, change):
        self.shell.automagic = change['new']
    
    _auto_status = [
        'Automagic is OFF, % prefix IS needed for line magics.',
        'Automagic is ON, % prefix IS NOT needed for line magics.']

    user_magics = Instance('IPython.core.magics.UserMagics', allow_none=True)

    def __init__(self, shell=None, config=None, user_magics=None, **traits):

        super(MagicsManager, self).__init__(shell=shell, config=config,
                                           user_magics=user_magics, **traits)
        self.magics = dict(line={}, cell={})
        # Let's add the user_magics to the registry for uniformity, so *all*
        # registered magic containers can be found there.
        self.registry[user_magics.__class__.__name__] = user_magics

    def auto_status(self):
        """Return descriptive string with automagic status."""
        return self._auto_status[self.auto_magic]
    
    def lsmagic(self):
        """Return a dict of currently available magic functions.

        The return dict has the keys 'line' and 'cell', corresponding to the
        two types of magics we support.  Each value is a list of names.
        """
        return self.magics

    def lsmagic_docs(self, brief=False, missing=''):
        """Return dict of documentation of magic functions.

        The return dict has the keys 'line' and 'cell', corresponding to the
        two types of magics we support. Each value is a dict keyed by magic
        name whose value is the function docstring. If a docstring is
        unavailable, the value of `missing` is used instead.

        If brief is True, only the first line of each docstring will be returned.
        """
        docs = {}
        for m_type in self.magics:
            m_docs = {}
            for m_name, m_func in self.magics[m_type].items():
                if m_func.__doc__:
                    if brief:
                        m_docs[m_name] = m_func.__doc__.split('\n', 1)[0]
                    else:
                        m_docs[m_name] = m_func.__doc__.rstrip()
                else:
                    m_docs[m_name] = missing
            docs[m_type] = m_docs
        return docs

    def register_lazy(self, name: str, fully_qualified_name: str):
        """
        Lazily register a magic via an extension.


        Parameters
        ----------
        name : str
            Name of the magic you wish to register.
        fully_qualified_name :
            Fully qualified name of the module/submodule that should be loaded
            as an extensions when the magic is first called.
            It is assumed that loading this extensions will register the given
            magic.
        """

        self.lazy_magics[name] = fully_qualified_name

    def register(self, *magic_objects):
        """Register one or more instances of Magics.

        Take one or more classes or instances of classes that subclass the main
        `core.Magic` class, and register them with IPython to use the magic
        functions they provide.  The registration process will then ensure that
        any methods that have decorated to provide line and/or cell magics will
        be recognized with the `%x`/`%%x` syntax as a line/cell magic
        respectively.

        If classes are given, they will be instantiated with the default
        constructor.  If your classes need a custom constructor, you should
        instanitate them first and pass the instance.

        The provided arguments can be an arbitrary mix of classes and instances.

        Parameters
        ----------
        *magic_objects : one or more classes or instances
        """
        # Start by validating them to ensure they have all had their magic
        # methods registered at the instance level
        for m in magic_objects:
            if not m.registered:
                raise ValueError("Class of magics %r was constructed without "
                                 "the @register_magics class decorator")
            if isinstance(m, type):
                # If we're given an uninstantiated class
                m = m(shell=self.shell)

            # Now that we have an instance, we can register it and update the
            # table of callables
            self.registry[m.__class__.__name__] = m
            for mtype in magic_kinds:
                self.magics[mtype].update(m.magics[mtype])

    def register_function(self, func, magic_kind='line', magic_name=None):
        """Expose a standalone function as magic function for IPython.

        This will create an IPython magic (line, cell or both) from a
        standalone function.  The functions should have the following
        signatures:

        * For line magics: `def f(line)`
        * For cell magics: `def f(line, cell)`
        * For a function that does both: `def f(line, cell=None)`

        In the latter case, the function will be called with `cell==None` when
        invoked as `%f`, and with cell as a string when invoked as `%%f`.

        Parameters
        ----------
        func : callable
            Function to be registered as a magic.
        magic_kind : str
            Kind of magic, one of 'line', 'cell' or 'line_cell'
        magic_name : optional str
            If given, the name the magic will have in the IPython namespace.  By
            default, the name of the function itself is used.
        """

        # Create the new method in the user_magics and register it in the
        # global table
        validate_type(magic_kind)
        magic_name = func.__name__ if magic_name is None else magic_name
        setattr(self.user_magics, magic_name, func)
        record_magic(self.magics, magic_kind, magic_name, func)

    def register_alias(self, alias_name, magic_name, magic_kind='line', magic_params=None):
        """Register an alias to a magic function.

        The alias is an instance of :class:`MagicAlias`, which holds the
        name and kind of the magic it should call. Binding is done at
        call time, so if the underlying magic function is changed the alias
        will call the new function.

        Parameters
        ----------
        alias_name : str
            The name of the magic to be registered.
        magic_name : str
            The name of an existing magic.
        magic_kind : str
            Kind of magic, one of 'line' or 'cell'
        """

        # `validate_type` is too permissive, as it allows 'line_cell'
        # which we do not handle.
        if magic_kind not in magic_kinds:
            raise ValueError('magic_kind must be one of %s, %s given' %
                             magic_kinds, magic_kind)

        alias = MagicAlias(self.shell, magic_name, magic_kind, magic_params)
        setattr(self.user_magics, alias_name, alias)
        record_magic(self.magics, magic_kind, alias_name, alias)

# Key base class that provides the central functionality for magics.


class Magics(Configurable):
    """Base class for implementing magic functions.

    Shell functions which can be reached as %function_name. All magic
    functions should accept a string, which they can parse for their own
    needs. This can make some functions easier to type, eg `%cd ../`
    vs. `%cd("../")`

    Classes providing magic functions need to subclass this class, and they
    MUST:

    - Use the method decorators `@line_magic` and `@cell_magic` to decorate
      individual methods as magic functions, AND

    - Use the class decorator `@magics_class` to ensure that the magic
      methods are properly registered at the instance level upon instance
      initialization.

    See :mod:`magic_functions` for examples of actual implementation classes.
    """
    # Dict holding all command-line options for each magic.
    options_table = None
    # Dict for the mapping of magic names to methods, set by class decorator
    magics = None
    # Flag to check that the class decorator was properly applied
    registered = False
    # Instance of IPython shell
    shell = None

    def __init__(self, shell=None, **kwargs):
        if not(self.__class__.registered):
            raise ValueError('Magics subclass without registration - '
                             'did you forget to apply @magics_class?')
        if shell is not None:
            if hasattr(shell, 'configurables'):
                shell.configurables.append(self)
            if hasattr(shell, 'config'):
                kwargs.setdefault('parent', shell)

        self.shell = shell
        self.options_table = {}
        # The method decorators are run when the instance doesn't exist yet, so
        # they can only record the names of the methods they are supposed to
        # grab.  Only now, that the instance exists, can we create the proper
        # mapping to bound methods.  So we read the info off the original names
        # table and replace each method name by the actual bound method.
        # But we mustn't clobber the *class* mapping, in case of multiple instances.
        class_magics = self.magics
        self.magics = {}
        for mtype in magic_kinds:
            tab = self.magics[mtype] = {}
            cls_tab = class_magics[mtype]
            for magic_name, meth_name in cls_tab.items():
                if isinstance(meth_name, str):
                    # it's a method name, grab it
                    tab[magic_name] = getattr(self, meth_name)
                else:
                    # it's the real thing
                    tab[magic_name] = meth_name
        # Configurable **needs** to be initiated at the end or the config
        # magics get screwed up.
        super(Magics, self).__init__(**kwargs)

    def arg_err(self,func):
        """Print docstring if incorrect arguments were passed"""
        print('Error in arguments:')
        print(oinspect.getdoc(func))

    def format_latex(self, strng):
        """Format a string for latex inclusion."""

        # Characters that need to be escaped for latex:
        escape_re = re.compile(r'(%|_|\$|#|&)',re.MULTILINE)
        # Magic command names as headers:
        cmd_name_re = re.compile(r'^(%s.*?):' % ESC_MAGIC,
                                 re.MULTILINE)
        # Magic commands
        cmd_re = re.compile(r'(?P<cmd>%s.+?\b)(?!\}\}:)' % ESC_MAGIC,
                            re.MULTILINE)
        # Paragraph continue
        par_re = re.compile(r'\\$',re.MULTILINE)

        # The "\n" symbol
        newline_re = re.compile(r'\\n')

        # Now build the string for output:
        #strng = cmd_name_re.sub(r'\n\\texttt{\\textsl{\\large \1}}:',strng)
        strng = cmd_name_re.sub(r'\n\\bigskip\n\\texttt{\\textbf{ \1}}:',
                                strng)
        strng = cmd_re.sub(r'\\texttt{\g<cmd>}',strng)
        strng = par_re.sub(r'\\\\',strng)
        strng = escape_re.sub(r'\\\1',strng)
        strng = newline_re.sub(r'\\textbackslash{}n',strng)
        return strng

    def parse_options(self, arg_str, opt_str, *long_opts, **kw):
        """Parse options passed to an argument string.

        The interface is similar to that of :func:`getopt.getopt`, but it
        returns a :class:`~IPython.utils.struct.Struct` with the options as keys
        and the stripped argument string still as a string.

        arg_str is quoted as a true sys.argv vector by using shlex.split.
        This allows us to easily expand variables, glob files, quote
        arguments, etc.

        Parameters
        ----------
        arg_str : str
            The arguments to parse.
        opt_str : str
            The options specification.
        mode : str, default 'string'
            If given as 'list', the argument string is returned as a list (split
            on whitespace) instead of a string.
        list_all : bool, default False
            Put all option values in lists. Normally only options
            appearing more than once are put in a list.
        posix : bool, default True
            Whether to split the input line in POSIX mode or not, as per the
            conventions outlined in the :mod:`shlex` module from the standard
            library.
        """

        # inject default options at the beginning of the input line
        caller = sys._getframe(1).f_code.co_name
        arg_str = '%s %s' % (self.options_table.get(caller,''),arg_str)

        mode = kw.get('mode','string')
        if mode not in ['string','list']:
            raise ValueError('incorrect mode given: %s' % mode)
        # Get options
        list_all = kw.get('list_all',0)
        posix = kw.get('posix', os.name == 'posix')
        strict = kw.get('strict', True)

        preserve_non_opts = kw.get("preserve_non_opts", False)
        remainder_arg_str = arg_str

        # Check if we have more than one argument to warrant extra processing:
        odict = {}  # Dictionary with options
        args = arg_str.split()
        if len(args) >= 1:
            # If the list of inputs only has 0 or 1 thing in it, there's no
            # need to look for options
            argv = arg_split(arg_str, posix, strict)
            # Do regular option processing
            try:
                opts,args = getopt(argv, opt_str, long_opts)
            except GetoptError as e:
                raise UsageError(
                    '%s ( allowed: "%s" %s)' % (e.msg, opt_str, " ".join(long_opts))
                ) from e
            for o, a in opts:
                if mode == "string" and preserve_non_opts:
                    # remove option-parts from the original args-string and preserve remaining-part.
                    # This relies on the arg_split(...) and getopt(...)'s impl spec, that the parsed options are
                    # returned in the original order.
                    remainder_arg_str = remainder_arg_str.replace(o, "", 1).replace(
                        a, "", 1
                    )
                if o.startswith("--"):
                    o = o[2:]
                else:
                    o = o[1:]
                try:
                    odict[o].append(a)
                except AttributeError:
                    odict[o] = [odict[o],a]
                except KeyError:
                    if list_all:
                        odict[o] = [a]
                    else:
                        odict[o] = a

        # Prepare opts,args for return
        opts = Struct(odict)
        if mode == 'string':
            if preserve_non_opts:
                args = remainder_arg_str.lstrip()
            else:
                args = " ".join(args)

        return opts,args

    def default_option(self, fn, optstr):
        """Make an entry in the options_table for fn, with value optstr"""

        if fn not in self.lsmagic():
            error("%s is not a magic function" % fn)
        self.options_table[fn] = optstr


class MagicAlias(object):
    """An alias to another magic function.

    An alias is determined by its magic name and magic kind. Lookup
    is done at call time, so if the underlying magic changes the alias
    will call the new function.

    Use the :meth:`MagicsManager.register_alias` method or the
    `%alias_magic` magic function to create and register a new alias.
    """
    def __init__(self, shell, magic_name, magic_kind, magic_params=None):
        self.shell = shell
        self.magic_name = magic_name
        self.magic_params = magic_params
        self.magic_kind = magic_kind

        self.pretty_target = '%s%s' % (magic_escapes[self.magic_kind], self.magic_name)
        self.__doc__ = "Alias for `%s`." % self.pretty_target

        self._in_call = False

    def __call__(self, *args, **kwargs):
        """Call the magic alias."""
        fn = self.shell.find_magic(self.magic_name, self.magic_kind)
        if fn is None:
            raise UsageError("Magic `%s` not found." % self.pretty_target)

        # Protect against infinite recursion.
        if self._in_call:
            raise UsageError("Infinite recursion detected; "
                             "magic aliases cannot call themselves.")
        self._in_call = True
        try:
            if self.magic_params:
                args_list = list(args)
                args_list[0] = self.magic_params + " " + args[0]
                args = tuple(args_list)
            return fn(*args, **kwargs)
        finally:
            self._in_call = False
