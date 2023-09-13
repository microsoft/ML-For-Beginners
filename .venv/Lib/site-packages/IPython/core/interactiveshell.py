# -*- coding: utf-8 -*-
"""Main IPython class."""

#-----------------------------------------------------------------------------
#  Copyright (C) 2001 Janko Hauser <jhauser@zscout.de>
#  Copyright (C) 2001-2007 Fernando Perez. <fperez@colorado.edu>
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------


import abc
import ast
import atexit
import bdb
import builtins as builtin_mod
import functools
import inspect
import os
import re
import runpy
import subprocess
import sys
import tempfile
import traceback
import types
import warnings
from ast import stmt
from io import open as io_open
from logging import error
from pathlib import Path
from typing import Callable
from typing import List as ListType, Dict as DictType, Any as AnyType
from typing import Optional, Sequence, Tuple
from warnings import warn

from pickleshare import PickleShareDB
from tempfile import TemporaryDirectory
from traitlets import (
    Any,
    Bool,
    CaselessStrEnum,
    Dict,
    Enum,
    Instance,
    Integer,
    List,
    Type,
    Unicode,
    default,
    observe,
    validate,
)
from traitlets.config.configurable import SingletonConfigurable
from traitlets.utils.importstring import import_item

import IPython.core.hooks
from IPython.core import magic, oinspect, page, prefilter, ultratb
from IPython.core.alias import Alias, AliasManager
from IPython.core.autocall import ExitAutocall
from IPython.core.builtin_trap import BuiltinTrap
from IPython.core.compilerop import CachingCompiler
from IPython.core.debugger import InterruptiblePdb
from IPython.core.display_trap import DisplayTrap
from IPython.core.displayhook import DisplayHook
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import InputRejected, UsageError
from IPython.core.events import EventManager, available_events
from IPython.core.extensions import ExtensionManager
from IPython.core.formatters import DisplayFormatter
from IPython.core.history import HistoryManager
from IPython.core.inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from IPython.core.logger import Logger
from IPython.core.macro import Macro
from IPython.core.payload import PayloadManager
from IPython.core.prefilter import PrefilterManager
from IPython.core.profiledir import ProfileDir
from IPython.core.usage import default_banner
from IPython.display import display
from IPython.paths import get_ipython_dir
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, io, openpy, py3compat
from IPython.utils.decorators import undoc
from IPython.utils.io import ask_yes_no
from IPython.utils.ipstruct import Struct
from IPython.utils.path import ensure_dir_exists, get_home_dir, get_py_filename
from IPython.utils.process import getoutput, system
from IPython.utils.strdispatch import StrDispatch
from IPython.utils.syspathcontext import prepended_to_syspath
from IPython.utils.text import DollarFormatter, LSString, SList, format_screen
from IPython.core.oinspect import OInfo


sphinxify: Optional[Callable]

try:
    import docrepr.sphinxify as sphx

    def sphinxify(oinfo):
        wrapped_docstring = sphx.wrap_main_docstring(oinfo)

        def sphinxify_docstring(docstring):
            with TemporaryDirectory() as dirname:
                return {
                    "text/html": sphx.sphinxify(wrapped_docstring, dirname),
                    "text/plain": docstring,
                }

        return sphinxify_docstring
except ImportError:
    sphinxify = None

if sys.version_info[:2] < (3, 11):
    from exceptiongroup import BaseExceptionGroup

class ProvisionalWarning(DeprecationWarning):
    """
    Warning class for unstable features
    """
    pass

from ast import Module

_assign_nodes = (ast.AugAssign, ast.AnnAssign, ast.Assign)
_single_targets_nodes = (ast.AugAssign, ast.AnnAssign)

#-----------------------------------------------------------------------------
# Await Helpers
#-----------------------------------------------------------------------------

# we still need to run things using the asyncio eventloop, but there is no
# async integration
from .async_helpers import (
    _asyncio_runner,
    _curio_runner,
    _pseudo_sync_runner,
    _should_be_async,
    _trio_runner,
)

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# compiled regexps for autoindent management
dedent_re = re.compile(r'^\s+raise|^\s+return|^\s+pass')

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------


def is_integer_string(s: str):
    """
    Variant of "str.isnumeric()" that allow negative values and other ints.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False
    raise ValueError("Unexpected error")


@undoc
def softspace(file, newvalue):
    """Copied from code.py, to remove the dependency"""

    oldvalue = 0
    try:
        oldvalue = file.softspace
    except AttributeError:
        pass
    try:
        file.softspace = newvalue
    except (AttributeError, TypeError):
        # "attribute-less object" or "read-only attributes"
        pass
    return oldvalue

@undoc
def no_op(*a, **kw):
    pass


class SpaceInInput(Exception): pass


class SeparateUnicode(Unicode):
    r"""A Unicode subclass to validate separate_in, separate_out, etc.

    This is a Unicode based trait that converts '0'->'' and ``'\\n'->'\n'``.
    """

    def validate(self, obj, value):
        if value == '0': value = ''
        value = value.replace('\\n','\n')
        return super(SeparateUnicode, self).validate(obj, value)


@undoc
class DummyMod(object):
    """A dummy module used for IPython's interactive module when
    a namespace must be assigned to the module's __dict__."""
    __spec__ = None


class ExecutionInfo(object):
    """The arguments used for a call to :meth:`InteractiveShell.run_cell`

    Stores information about what is going to happen.
    """
    raw_cell = None
    store_history = False
    silent = False
    shell_futures = True
    cell_id = None

    def __init__(self, raw_cell, store_history, silent, shell_futures, cell_id):
        self.raw_cell = raw_cell
        self.store_history = store_history
        self.silent = silent
        self.shell_futures = shell_futures
        self.cell_id = cell_id

    def __repr__(self):
        name = self.__class__.__qualname__
        raw_cell = (
            (self.raw_cell[:50] + "..") if len(self.raw_cell) > 50 else self.raw_cell
        )
        return (
            '<%s object at %x, raw_cell="%s" store_history=%s silent=%s shell_futures=%s cell_id=%s>'
            % (
                name,
                id(self),
                raw_cell,
                self.store_history,
                self.silent,
                self.shell_futures,
                self.cell_id,
            )
        )


class ExecutionResult(object):
    """The result of a call to :meth:`InteractiveShell.run_cell`

    Stores information about what took place.
    """
    execution_count = None
    error_before_exec = None
    error_in_exec: Optional[BaseException] = None
    info = None
    result = None

    def __init__(self, info):
        self.info = info

    @property
    def success(self):
        return (self.error_before_exec is None) and (self.error_in_exec is None)

    def raise_error(self):
        """Reraises error if `success` is `False`, otherwise does nothing"""
        if self.error_before_exec is not None:
            raise self.error_before_exec
        if self.error_in_exec is not None:
            raise self.error_in_exec

    def __repr__(self):
        name = self.__class__.__qualname__
        return '<%s object at %x, execution_count=%s error_before_exec=%s error_in_exec=%s info=%s result=%s>' %\
                (name, id(self), self.execution_count, self.error_before_exec, self.error_in_exec, repr(self.info), repr(self.result))

@functools.wraps(io_open)
def _modified_open(file, *args, **kwargs):
    if file in {0, 1, 2}:
        raise ValueError(
            f"IPython won't let you open fd={file} by default "
            "as it is likely to crash IPython. If you know what you are doing, "
            "you can use builtins' open."
        )

    return io_open(file, *args, **kwargs)

class InteractiveShell(SingletonConfigurable):
    """An enhanced, interactive shell for Python."""

    _instance = None

    ast_transformers = List([], help=
        """
        A list of ast.NodeTransformer subclass instances, which will be applied
        to user input before code is run.
        """
    ).tag(config=True)

    autocall = Enum((0,1,2), default_value=0, help=
        """
        Make IPython automatically call any callable object even if you didn't
        type explicit parentheses. For example, 'str 43' becomes 'str(43)'
        automatically. The value can be '0' to disable the feature, '1' for
        'smart' autocall, where it is not applied if there are no more
        arguments on the line, and '2' for 'full' autocall, where all callable
        objects are automatically called (even if no arguments are present).
        """
    ).tag(config=True)

    autoindent = Bool(True, help=
        """
        Autoindent IPython code entered interactively.
        """
    ).tag(config=True)

    autoawait = Bool(True, help=
        """
        Automatically run await statement in the top level repl.
        """
    ).tag(config=True)

    loop_runner_map ={
        'asyncio':(_asyncio_runner, True),
        'curio':(_curio_runner, True),
        'trio':(_trio_runner, True),
        'sync': (_pseudo_sync_runner, False)
    }

    loop_runner = Any(default_value="IPython.core.interactiveshell._asyncio_runner",
        allow_none=True,
        help="""Select the loop runner that will be used to execute top-level asynchronous code"""
    ).tag(config=True)

    @default('loop_runner')
    def _default_loop_runner(self):
        return import_item("IPython.core.interactiveshell._asyncio_runner")

    @validate('loop_runner')
    def _import_runner(self, proposal):
        if isinstance(proposal.value, str):
            if proposal.value in self.loop_runner_map:
                runner, autoawait = self.loop_runner_map[proposal.value]
                self.autoawait = autoawait
                return runner
            runner = import_item(proposal.value)
            if not callable(runner):
                raise ValueError('loop_runner must be callable')
            return runner
        if not callable(proposal.value):
            raise ValueError('loop_runner must be callable')
        return proposal.value

    automagic = Bool(True, help=
        """
        Enable magic commands to be called without the leading %.
        """
    ).tag(config=True)

    banner1 = Unicode(default_banner,
        help="""The part of the banner to be printed before the profile"""
    ).tag(config=True)
    banner2 = Unicode('',
        help="""The part of the banner to be printed after the profile"""
    ).tag(config=True)

    cache_size = Integer(1000, help=
        """
        Set the size of the output cache.  The default is 1000, you can
        change it permanently in your config file.  Setting it to 0 completely
        disables the caching system, and the minimum value accepted is 3 (if
        you provide a value less than 3, it is reset to 0 and a warning is
        issued).  This limit is defined because otherwise you'll spend more
        time re-flushing a too small cache than working
        """
    ).tag(config=True)
    color_info = Bool(True, help=
        """
        Use colors for displaying information about objects. Because this
        information is passed through a pager (like 'less'), and some pagers
        get confused with color codes, this capability can be turned off.
        """
    ).tag(config=True)
    colors = CaselessStrEnum(('Neutral', 'NoColor','LightBG','Linux'),
                             default_value='Neutral',
        help="Set the color scheme (NoColor, Neutral, Linux, or LightBG)."
    ).tag(config=True)
    debug = Bool(False).tag(config=True)
    disable_failing_post_execute = Bool(False,
        help="Don't call post-execute functions that have failed in the past."
    ).tag(config=True)
    display_formatter = Instance(DisplayFormatter, allow_none=True)
    displayhook_class = Type(DisplayHook)
    display_pub_class = Type(DisplayPublisher)
    compiler_class = Type(CachingCompiler)
    inspector_class = Type(
        oinspect.Inspector, help="Class to use to instantiate the shell inspector"
    ).tag(config=True)

    sphinxify_docstring = Bool(False, help=
        """
        Enables rich html representation of docstrings. (This requires the
        docrepr module).
        """).tag(config=True)

    @observe("sphinxify_docstring")
    def _sphinxify_docstring_changed(self, change):
        if change['new']:
            warn("`sphinxify_docstring` is provisional since IPython 5.0 and might change in future versions." , ProvisionalWarning)

    enable_html_pager = Bool(False, help=
        """
        (Provisional API) enables html representation in mime bundles sent
        to pagers.
        """).tag(config=True)

    @observe("enable_html_pager")
    def _enable_html_pager_changed(self, change):
        if change['new']:
            warn("`enable_html_pager` is provisional since IPython 5.0 and might change in future versions.", ProvisionalWarning)

    data_pub_class = None

    exit_now = Bool(False)
    exiter = Instance(ExitAutocall)
    @default('exiter')
    def _exiter_default(self):
        return ExitAutocall(self)
    # Monotonically increasing execution counter
    execution_count = Integer(1)
    filename = Unicode("<ipython console>")
    ipython_dir= Unicode('').tag(config=True) # Set to get_ipython_dir() in __init__

    # Used to transform cells before running them, and check whether code is complete
    input_transformer_manager = Instance('IPython.core.inputtransformer2.TransformerManager',
                                         ())

    @property
    def input_transformers_cleanup(self):
        return self.input_transformer_manager.cleanup_transforms

    input_transformers_post = List([],
        help="A list of string input transformers, to be applied after IPython's "
             "own input transformations."
    )

    @property
    def input_splitter(self):
        """Make this available for backward compatibility (pre-7.0 release) with existing code.

        For example, ipykernel ipykernel currently uses
        `shell.input_splitter.check_complete`
        """
        from warnings import warn
        warn("`input_splitter` is deprecated since IPython 7.0, prefer `input_transformer_manager`.",
             DeprecationWarning, stacklevel=2
        )
        return self.input_transformer_manager

    logstart = Bool(False, help=
        """
        Start logging to the default log file in overwrite mode.
        Use `logappend` to specify a log file to **append** logs to.
        """
    ).tag(config=True)
    logfile = Unicode('', help=
        """
        The name of the logfile to use.
        """
    ).tag(config=True)
    logappend = Unicode('', help=
        """
        Start logging to the given file in append mode.
        Use `logfile` to specify a log file to **overwrite** logs to.
        """
    ).tag(config=True)
    object_info_string_level = Enum((0,1,2), default_value=0,
    ).tag(config=True)
    pdb = Bool(False, help=
        """
        Automatically call the pdb debugger after every exception.
        """
    ).tag(config=True)
    display_page = Bool(False,
        help="""If True, anything that would be passed to the pager
        will be displayed as regular output instead."""
    ).tag(config=True)


    show_rewritten_input = Bool(True,
        help="Show rewritten input, e.g. for autocall."
    ).tag(config=True)

    quiet = Bool(False).tag(config=True)

    history_length = Integer(10000,
        help='Total length of command history'
    ).tag(config=True)

    history_load_length = Integer(1000, help=
        """
        The number of saved history entries to be loaded
        into the history buffer at startup.
        """
    ).tag(config=True)

    ast_node_interactivity = Enum(['all', 'last', 'last_expr', 'none', 'last_expr_or_assign'],
                                  default_value='last_expr',
                                  help="""
        'all', 'last', 'last_expr' or 'none', 'last_expr_or_assign' specifying
        which nodes should be run interactively (displaying output from expressions).
        """
    ).tag(config=True)

    warn_venv = Bool(
        True,
        help="Warn if running in a virtual environment with no IPython installed (so IPython from the global environment is used).",
    ).tag(config=True)

    # TODO: this part of prompt management should be moved to the frontends.
    # Use custom TraitTypes that convert '0'->'' and '\\n'->'\n'
    separate_in = SeparateUnicode('\n').tag(config=True)
    separate_out = SeparateUnicode('').tag(config=True)
    separate_out2 = SeparateUnicode('').tag(config=True)
    wildcards_case_sensitive = Bool(True).tag(config=True)
    xmode = CaselessStrEnum(('Context', 'Plain', 'Verbose', 'Minimal'),
                            default_value='Context',
                            help="Switch modes for the IPython exception handlers."
                            ).tag(config=True)

    # Subcomponents of InteractiveShell
    alias_manager = Instance('IPython.core.alias.AliasManager', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    builtin_trap = Instance('IPython.core.builtin_trap.BuiltinTrap', allow_none=True)
    display_trap = Instance('IPython.core.display_trap.DisplayTrap', allow_none=True)
    extension_manager = Instance('IPython.core.extensions.ExtensionManager', allow_none=True)
    payload_manager = Instance('IPython.core.payload.PayloadManager', allow_none=True)
    history_manager = Instance('IPython.core.history.HistoryAccessorBase', allow_none=True)
    magics_manager = Instance('IPython.core.magic.MagicsManager', allow_none=True)

    profile_dir = Instance('IPython.core.application.ProfileDir', allow_none=True)
    @property
    def profile(self):
        if self.profile_dir is not None:
            name = os.path.basename(self.profile_dir.location)
            return name.replace('profile_','')


    # Private interface
    _post_execute = Dict()

    # Tracks any GUI loop loaded for pylab
    pylab_gui_select = None

    last_execution_succeeded = Bool(True, help='Did last executed command succeeded')

    last_execution_result = Instance('IPython.core.interactiveshell.ExecutionResult', help='Result of executing the last command', allow_none=True)

    def __init__(self, ipython_dir=None, profile_dir=None,
                 user_module=None, user_ns=None,
                 custom_exceptions=((), None), **kwargs):
        # This is where traits with a config_key argument are updated
        # from the values on config.
        super(InteractiveShell, self).__init__(**kwargs)
        if 'PromptManager' in self.config:
            warn('As of IPython 5.0 `PromptManager` config will have no effect'
                 ' and has been replaced by TerminalInteractiveShell.prompts_class')
        self.configurables = [self]

        # These are relatively independent and stateless
        self.init_ipython_dir(ipython_dir)
        self.init_profile_dir(profile_dir)
        self.init_instance_attrs()
        self.init_environment()

        # Check if we're in a virtualenv, and set up sys.path.
        self.init_virtualenv()

        # Create namespaces (user_ns, user_global_ns, etc.)
        self.init_create_namespaces(user_module, user_ns)
        # This has to be done after init_create_namespaces because it uses
        # something in self.user_ns, but before init_sys_modules, which
        # is the first thing to modify sys.
        # TODO: When we override sys.stdout and sys.stderr before this class
        # is created, we are saving the overridden ones here. Not sure if this
        # is what we want to do.
        self.save_sys_module_state()
        self.init_sys_modules()

        # While we're trying to have each part of the code directly access what
        # it needs without keeping redundant references to objects, we have too
        # much legacy code that expects ip.db to exist.
        self.db = PickleShareDB(os.path.join(self.profile_dir.location, 'db'))

        self.init_history()
        self.init_encoding()
        self.init_prefilter()

        self.init_syntax_highlighting()
        self.init_hooks()
        self.init_events()
        self.init_pushd_popd_magic()
        self.init_user_ns()
        self.init_logger()
        self.init_builtins()

        # The following was in post_config_initialization
        self.init_inspector()
        self.raw_input_original = input
        self.init_completer()
        # TODO: init_io() needs to happen before init_traceback handlers
        # because the traceback handlers hardcode the stdout/stderr streams.
        # This logic in in debugger.Pdb and should eventually be changed.
        self.init_io()
        self.init_traceback_handlers(custom_exceptions)
        self.init_prompts()
        self.init_display_formatter()
        self.init_display_pub()
        self.init_data_pub()
        self.init_displayhook()
        self.init_magics()
        self.init_alias()
        self.init_logstart()
        self.init_pdb()
        self.init_extension_manager()
        self.init_payload()
        self.events.trigger('shell_initialized', self)
        atexit.register(self.atexit_operations)

        # The trio runner is used for running Trio in the foreground thread. It
        # is different from `_trio_runner(async_fn)` in `async_helpers.py`
        # which calls `trio.run()` for every cell. This runner runs all cells
        # inside a single Trio event loop. If used, it is set from
        # `ipykernel.kernelapp`.
        self.trio_runner = None

    def get_ipython(self):
        """Return the currently running IPython instance."""
        return self

    #-------------------------------------------------------------------------
    # Trait changed handlers
    #-------------------------------------------------------------------------
    @observe('ipython_dir')
    def _ipython_dir_changed(self, change):
        ensure_dir_exists(change['new'])

    def set_autoindent(self,value=None):
        """Set the autoindent flag.

        If called with no arguments, it acts as a toggle."""
        if value is None:
            self.autoindent = not self.autoindent
        else:
            self.autoindent = value

    def set_trio_runner(self, tr):
        self.trio_runner = tr

    #-------------------------------------------------------------------------
    # init_* methods called by __init__
    #-------------------------------------------------------------------------

    def init_ipython_dir(self, ipython_dir):
        if ipython_dir is not None:
            self.ipython_dir = ipython_dir
            return

        self.ipython_dir = get_ipython_dir()

    def init_profile_dir(self, profile_dir):
        if profile_dir is not None:
            self.profile_dir = profile_dir
            return
        self.profile_dir = ProfileDir.create_profile_dir_by_name(
            self.ipython_dir, "default"
        )

    def init_instance_attrs(self):
        self.more = False

        # command compiler
        self.compile = self.compiler_class()

        # Make an empty namespace, which extension writers can rely on both
        # existing and NEVER being used by ipython itself.  This gives them a
        # convenient location for storing additional information and state
        # their extensions may require, without fear of collisions with other
        # ipython names that may develop later.
        self.meta = Struct()

        # Temporary files used for various purposes.  Deleted at exit.
        # The files here are stored with Path from Pathlib
        self.tempfiles = []
        self.tempdirs = []

        # keep track of where we started running (mainly for crash post-mortem)
        # This is not being used anywhere currently.
        self.starting_dir = os.getcwd()

        # Indentation management
        self.indent_current_nsp = 0

        # Dict to track post-execution functions that have been registered
        self._post_execute = {}

    def init_environment(self):
        """Any changes we need to make to the user's environment."""
        pass

    def init_encoding(self):
        # Get system encoding at startup time.  Certain terminals (like Emacs
        # under Win32 have it set to None, and we need to have a known valid
        # encoding to use in the raw_input() method
        try:
            self.stdin_encoding = sys.stdin.encoding or 'ascii'
        except AttributeError:
            self.stdin_encoding = 'ascii'


    @observe('colors')
    def init_syntax_highlighting(self, changes=None):
        # Python source parser/formatter for syntax highlighting
        pyformat = PyColorize.Parser(style=self.colors, parent=self).format
        self.pycolorize = lambda src: pyformat(src,'str')

    def refresh_style(self):
        # No-op here, used in subclass
        pass

    def init_pushd_popd_magic(self):
        # for pushd/popd management
        self.home_dir = get_home_dir()

        self.dir_stack = []

    def init_logger(self):
        self.logger = Logger(self.home_dir, logfname='ipython_log.py',
                             logmode='rotate')

    def init_logstart(self):
        """Initialize logging in case it was requested at the command line.
        """
        if self.logappend:
            self.magic('logstart %s append' % self.logappend)
        elif self.logfile:
            self.magic('logstart %s' % self.logfile)
        elif self.logstart:
            self.magic('logstart')


    def init_builtins(self):
        # A single, static flag that we set to True.  Its presence indicates
        # that an IPython shell has been created, and we make no attempts at
        # removing on exit or representing the existence of more than one
        # IPython at a time.
        builtin_mod.__dict__['__IPYTHON__'] = True
        builtin_mod.__dict__['display'] = display

        self.builtin_trap = BuiltinTrap(shell=self)

    @observe('colors')
    def init_inspector(self, changes=None):
        # Object inspector
        self.inspector = self.inspector_class(
            oinspect.InspectColors,
            PyColorize.ANSICodeColors,
            self.colors,
            self.object_info_string_level,
        )

    def init_io(self):
        # implemented in subclasses, TerminalInteractiveShell does call
        # colorama.init().
        pass

    def init_prompts(self):
        # Set system prompts, so that scripts can decide if they are running
        # interactively.
        sys.ps1 = 'In : '
        sys.ps2 = '...: '
        sys.ps3 = 'Out: '

    def init_display_formatter(self):
        self.display_formatter = DisplayFormatter(parent=self)
        self.configurables.append(self.display_formatter)

    def init_display_pub(self):
        self.display_pub = self.display_pub_class(parent=self, shell=self)
        self.configurables.append(self.display_pub)

    def init_data_pub(self):
        if not self.data_pub_class:
            self.data_pub = None
            return
        self.data_pub = self.data_pub_class(parent=self)
        self.configurables.append(self.data_pub)

    def init_displayhook(self):
        # Initialize displayhook, set in/out prompts and printing system
        self.displayhook = self.displayhook_class(
            parent=self,
            shell=self,
            cache_size=self.cache_size,
        )
        self.configurables.append(self.displayhook)
        # This is a context manager that installs/revmoes the displayhook at
        # the appropriate time.
        self.display_trap = DisplayTrap(hook=self.displayhook)

    @staticmethod
    def get_path_links(p: Path):
        """Gets path links including all symlinks

        Examples
        --------
        In [1]: from IPython.core.interactiveshell import InteractiveShell

        In [2]: import sys, pathlib

        In [3]: paths = InteractiveShell.get_path_links(pathlib.Path(sys.executable))

        In [4]: len(paths) == len(set(paths))
        Out[4]: True

        In [5]: bool(paths)
        Out[5]: True
        """
        paths = [p]
        while p.is_symlink():
            new_path = Path(os.readlink(p))
            if not new_path.is_absolute():
                new_path = p.parent / new_path
            p = new_path
            paths.append(p)
        return paths

    def init_virtualenv(self):
        """Add the current virtualenv to sys.path so the user can import modules from it.
        This isn't perfect: it doesn't use the Python interpreter with which the
        virtualenv was built, and it ignores the --no-site-packages option. A
        warning will appear suggesting the user installs IPython in the
        virtualenv, but for many cases, it probably works well enough.

        Adapted from code snippets online.

        http://blog.ufsoft.org/2009/1/29/ipython-and-virtualenv
        """
        if 'VIRTUAL_ENV' not in os.environ:
            # Not in a virtualenv
            return
        elif os.environ["VIRTUAL_ENV"] == "":
            warn("Virtual env path set to '', please check if this is intended.")
            return

        p = Path(sys.executable)
        p_venv = Path(os.environ["VIRTUAL_ENV"])

        # fallback venv detection:
        # stdlib venv may symlink sys.executable, so we can't use realpath.
        # but others can symlink *to* the venv Python, so we can't just use sys.executable.
        # So we just check every item in the symlink tree (generally <= 3)
        paths = self.get_path_links(p)

        # In Cygwin paths like "c:\..." and '\cygdrive\c\...' are possible
        if p_venv.parts[1] == "cygdrive":
            drive_name = p_venv.parts[2]
            p_venv = (drive_name + ":/") / Path(*p_venv.parts[3:])

        if any(p_venv == p.parents[1] for p in paths):
            # Our exe is inside or has access to the virtualenv, don't need to do anything.
            return

        if sys.platform == "win32":
            virtual_env = str(Path(os.environ["VIRTUAL_ENV"], "Lib", "site-packages"))
        else:
            virtual_env_path = Path(
                os.environ["VIRTUAL_ENV"], "lib", "python{}.{}", "site-packages"
            )
            p_ver = sys.version_info[:2]

            # Predict version from py[thon]-x.x in the $VIRTUAL_ENV
            re_m = re.search(r"\bpy(?:thon)?([23])\.(\d+)\b", os.environ["VIRTUAL_ENV"])
            if re_m:
                predicted_path = Path(str(virtual_env_path).format(*re_m.groups()))
                if predicted_path.exists():
                    p_ver = re_m.groups()

            virtual_env = str(virtual_env_path).format(*p_ver)
        if self.warn_venv:
            warn(
                "Attempting to work in a virtualenv. If you encounter problems, "
                "please install IPython inside the virtualenv."
            )
        import site
        sys.path.insert(0, virtual_env)
        site.addsitedir(virtual_env)

    #-------------------------------------------------------------------------
    # Things related to injections into the sys module
    #-------------------------------------------------------------------------

    def save_sys_module_state(self):
        """Save the state of hooks in the sys module.

        This has to be called after self.user_module is created.
        """
        self._orig_sys_module_state = {'stdin': sys.stdin,
                                       'stdout': sys.stdout,
                                       'stderr': sys.stderr,
                                       'excepthook': sys.excepthook}
        self._orig_sys_modules_main_name = self.user_module.__name__
        self._orig_sys_modules_main_mod = sys.modules.get(self.user_module.__name__)

    def restore_sys_module_state(self):
        """Restore the state of the sys module."""
        try:
            for k, v in self._orig_sys_module_state.items():
                setattr(sys, k, v)
        except AttributeError:
            pass
        # Reset what what done in self.init_sys_modules
        if self._orig_sys_modules_main_mod is not None:
            sys.modules[self._orig_sys_modules_main_name] = self._orig_sys_modules_main_mod

    #-------------------------------------------------------------------------
    # Things related to the banner
    #-------------------------------------------------------------------------

    @property
    def banner(self):
        banner = self.banner1
        if self.profile and self.profile != 'default':
            banner += '\nIPython profile: %s\n' % self.profile
        if self.banner2:
            banner += '\n' + self.banner2
        return banner

    def show_banner(self, banner=None):
        if banner is None:
            banner = self.banner
        sys.stdout.write(banner)

    #-------------------------------------------------------------------------
    # Things related to hooks
    #-------------------------------------------------------------------------

    def init_hooks(self):
        # hooks holds pointers used for user-side customizations
        self.hooks = Struct()

        self.strdispatchers = {}

        # Set all default hooks, defined in the IPython.hooks module.
        hooks = IPython.core.hooks
        for hook_name in hooks.__all__:
            # default hooks have priority 100, i.e. low; user hooks should have
            # 0-100 priority
            self.set_hook(hook_name, getattr(hooks, hook_name), 100)

        if self.display_page:
            self.set_hook('show_in_pager', page.as_hook(page.display_page), 90)

    def set_hook(self, name, hook, priority=50, str_key=None, re_key=None):
        """set_hook(name,hook) -> sets an internal IPython hook.

        IPython exposes some of its internal API as user-modifiable hooks.  By
        adding your function to one of these hooks, you can modify IPython's
        behavior to call at runtime your own routines."""

        # At some point in the future, this should validate the hook before it
        # accepts it.  Probably at least check that the hook takes the number
        # of args it's supposed to.

        f = types.MethodType(hook,self)

        # check if the hook is for strdispatcher first
        if str_key is not None:
            sdp = self.strdispatchers.get(name, StrDispatch())
            sdp.add_s(str_key, f, priority )
            self.strdispatchers[name] = sdp
            return
        if re_key is not None:
            sdp = self.strdispatchers.get(name, StrDispatch())
            sdp.add_re(re.compile(re_key), f, priority )
            self.strdispatchers[name] = sdp
            return

        dp = getattr(self.hooks, name, None)
        if name not in IPython.core.hooks.__all__:
            print("Warning! Hook '%s' is not one of %s" % \
                  (name, IPython.core.hooks.__all__ ))

        if name in IPython.core.hooks.deprecated:
            alternative = IPython.core.hooks.deprecated[name]
            raise ValueError(
                "Hook {} has been deprecated since IPython 5.0. Use {} instead.".format(
                    name, alternative
                )
            )

        if not dp:
            dp = IPython.core.hooks.CommandChainDispatcher()

        try:
            dp.add(f,priority)
        except AttributeError:
            # it was not commandchain, plain old func - replace
            dp = f

        setattr(self.hooks,name, dp)

    #-------------------------------------------------------------------------
    # Things related to events
    #-------------------------------------------------------------------------

    def init_events(self):
        self.events = EventManager(self, available_events)

        self.events.register("pre_execute", self._clear_warning_registry)

    def register_post_execute(self, func):
        """DEPRECATED: Use ip.events.register('post_run_cell', func)

        Register a function for calling after code execution.
        """
        raise ValueError(
            "ip.register_post_execute is deprecated since IPython 1.0, use "
            "ip.events.register('post_run_cell', func) instead."
        )

    def _clear_warning_registry(self):
        # clear the warning registry, so that different code blocks with
        # overlapping line number ranges don't cause spurious suppression of
        # warnings (see gh-6611 for details)
        if "__warningregistry__" in self.user_global_ns:
            del self.user_global_ns["__warningregistry__"]

    #-------------------------------------------------------------------------
    # Things related to the "main" module
    #-------------------------------------------------------------------------

    def new_main_mod(self, filename, modname):
        """Return a new 'main' module object for user code execution.

        ``filename`` should be the path of the script which will be run in the
        module. Requests with the same filename will get the same module, with
        its namespace cleared.

        ``modname`` should be the module name - normally either '__main__' or
        the basename of the file without the extension.

        When scripts are executed via %run, we must keep a reference to their
        __main__ module around so that Python doesn't
        clear it, rendering references to module globals useless.

        This method keeps said reference in a private dict, keyed by the
        absolute path of the script. This way, for multiple executions of the
        same script we only keep one copy of the namespace (the last one),
        thus preventing memory leaks from old references while allowing the
        objects from the last execution to be accessible.
        """
        filename = os.path.abspath(filename)
        try:
            main_mod = self._main_mod_cache[filename]
        except KeyError:
            main_mod = self._main_mod_cache[filename] = types.ModuleType(
                        modname,
                        doc="Module created for script run in IPython")
        else:
            main_mod.__dict__.clear()
            main_mod.__name__ = modname

        main_mod.__file__ = filename
        # It seems pydoc (and perhaps others) needs any module instance to
        # implement a __nonzero__ method
        main_mod.__nonzero__ = lambda : True

        return main_mod

    def clear_main_mod_cache(self):
        """Clear the cache of main modules.

        Mainly for use by utilities like %reset.

        Examples
        --------
        In [15]: import IPython

        In [16]: m = _ip.new_main_mod(IPython.__file__, 'IPython')

        In [17]: len(_ip._main_mod_cache) > 0
        Out[17]: True

        In [18]: _ip.clear_main_mod_cache()

        In [19]: len(_ip._main_mod_cache) == 0
        Out[19]: True
        """
        self._main_mod_cache.clear()

    #-------------------------------------------------------------------------
    # Things related to debugging
    #-------------------------------------------------------------------------

    def init_pdb(self):
        # Set calling of pdb on exceptions
        # self.call_pdb is a property
        self.call_pdb = self.pdb

    def _get_call_pdb(self):
        return self._call_pdb

    def _set_call_pdb(self,val):

        if val not in (0,1,False,True):
            raise ValueError('new call_pdb value must be boolean')

        # store value in instance
        self._call_pdb = val

        # notify the actual exception handlers
        self.InteractiveTB.call_pdb = val

    call_pdb = property(_get_call_pdb,_set_call_pdb,None,
                        'Control auto-activation of pdb at exceptions')

    def debugger(self,force=False):
        """Call the pdb debugger.

        Keywords:

          - force(False): by default, this routine checks the instance call_pdb
            flag and does not actually invoke the debugger if the flag is false.
            The 'force' option forces the debugger to activate even if the flag
            is false.
        """

        if not (force or self.call_pdb):
            return

        if not hasattr(sys,'last_traceback'):
            error('No traceback has been produced, nothing to debug.')
            return

        self.InteractiveTB.debugger(force=True)

    #-------------------------------------------------------------------------
    # Things related to IPython's various namespaces
    #-------------------------------------------------------------------------
    default_user_namespaces = True

    def init_create_namespaces(self, user_module=None, user_ns=None):
        # Create the namespace where the user will operate.  user_ns is
        # normally the only one used, and it is passed to the exec calls as
        # the locals argument.  But we do carry a user_global_ns namespace
        # given as the exec 'globals' argument,  This is useful in embedding
        # situations where the ipython shell opens in a context where the
        # distinction between locals and globals is meaningful.  For
        # non-embedded contexts, it is just the same object as the user_ns dict.

        # FIXME. For some strange reason, __builtins__ is showing up at user
        # level as a dict instead of a module. This is a manual fix, but I
        # should really track down where the problem is coming from. Alex
        # Schmolck reported this problem first.

        # A useful post by Alex Martelli on this topic:
        # Re: inconsistent value from __builtins__
        # Von: Alex Martelli <aleaxit@yahoo.com>
        # Datum: Freitag 01 Oktober 2004 04:45:34 nachmittags/abends
        # Gruppen: comp.lang.python

        # Michael Hohn <hohn@hooknose.lbl.gov> wrote:
        # > >>> print type(builtin_check.get_global_binding('__builtins__'))
        # > <type 'dict'>
        # > >>> print type(__builtins__)
        # > <type 'module'>
        # > Is this difference in return value intentional?

        # Well, it's documented that '__builtins__' can be either a dictionary
        # or a module, and it's been that way for a long time. Whether it's
        # intentional (or sensible), I don't know. In any case, the idea is
        # that if you need to access the built-in namespace directly, you
        # should start with "import __builtin__" (note, no 's') which will
        # definitely give you a module. Yeah, it's somewhat confusing:-(.

        # These routines return a properly built module and dict as needed by
        # the rest of the code, and can also be used by extension writers to
        # generate properly initialized namespaces.
        if (user_ns is not None) or (user_module is not None):
            self.default_user_namespaces = False
        self.user_module, self.user_ns = self.prepare_user_module(user_module, user_ns)

        # A record of hidden variables we have added to the user namespace, so
        # we can list later only variables defined in actual interactive use.
        self.user_ns_hidden = {}

        # Now that FakeModule produces a real module, we've run into a nasty
        # problem: after script execution (via %run), the module where the user
        # code ran is deleted.  Now that this object is a true module (needed
        # so doctest and other tools work correctly), the Python module
        # teardown mechanism runs over it, and sets to None every variable
        # present in that module.  Top-level references to objects from the
        # script survive, because the user_ns is updated with them.  However,
        # calling functions defined in the script that use other things from
        # the script will fail, because the function's closure had references
        # to the original objects, which are now all None.  So we must protect
        # these modules from deletion by keeping a cache.
        #
        # To avoid keeping stale modules around (we only need the one from the
        # last run), we use a dict keyed with the full path to the script, so
        # only the last version of the module is held in the cache.  Note,
        # however, that we must cache the module *namespace contents* (their
        # __dict__).  Because if we try to cache the actual modules, old ones
        # (uncached) could be destroyed while still holding references (such as
        # those held by GUI objects that tend to be long-lived)>
        #
        # The %reset command will flush this cache.  See the cache_main_mod()
        # and clear_main_mod_cache() methods for details on use.

        # This is the cache used for 'main' namespaces
        self._main_mod_cache = {}

        # A table holding all the namespaces IPython deals with, so that
        # introspection facilities can search easily.
        self.ns_table = {'user_global':self.user_module.__dict__,
                         'user_local':self.user_ns,
                         'builtin':builtin_mod.__dict__
                         }

    @property
    def user_global_ns(self):
        return self.user_module.__dict__

    def prepare_user_module(self, user_module=None, user_ns=None):
        """Prepare the module and namespace in which user code will be run.

        When IPython is started normally, both parameters are None: a new module
        is created automatically, and its __dict__ used as the namespace.

        If only user_module is provided, its __dict__ is used as the namespace.
        If only user_ns is provided, a dummy module is created, and user_ns
        becomes the global namespace. If both are provided (as they may be
        when embedding), user_ns is the local namespace, and user_module
        provides the global namespace.

        Parameters
        ----------
        user_module : module, optional
            The current user module in which IPython is being run. If None,
            a clean module will be created.
        user_ns : dict, optional
            A namespace in which to run interactive commands.

        Returns
        -------
        A tuple of user_module and user_ns, each properly initialised.
        """
        if user_module is None and user_ns is not None:
            user_ns.setdefault("__name__", "__main__")
            user_module = DummyMod()
            user_module.__dict__ = user_ns

        if user_module is None:
            user_module = types.ModuleType("__main__",
                doc="Automatically created module for IPython interactive environment")

        # We must ensure that __builtin__ (without the final 's') is always
        # available and pointing to the __builtin__ *module*.  For more details:
        # http://mail.python.org/pipermail/python-dev/2001-April/014068.html
        user_module.__dict__.setdefault('__builtin__', builtin_mod)
        user_module.__dict__.setdefault('__builtins__', builtin_mod)

        if user_ns is None:
            user_ns = user_module.__dict__

        return user_module, user_ns

    def init_sys_modules(self):
        # We need to insert into sys.modules something that looks like a
        # module but which accesses the IPython namespace, for shelve and
        # pickle to work interactively. Normally they rely on getting
        # everything out of __main__, but for embedding purposes each IPython
        # instance has its own private namespace, so we can't go shoving
        # everything into __main__.

        # note, however, that we should only do this for non-embedded
        # ipythons, which really mimic the __main__.__dict__ with their own
        # namespace.  Embedded instances, on the other hand, should not do
        # this because they need to manage the user local/global namespaces
        # only, but they live within a 'normal' __main__ (meaning, they
        # shouldn't overtake the execution environment of the script they're
        # embedded in).

        # This is overridden in the InteractiveShellEmbed subclass to a no-op.
        main_name = self.user_module.__name__
        sys.modules[main_name] = self.user_module

    def init_user_ns(self):
        """Initialize all user-visible namespaces to their minimum defaults.

        Certain history lists are also initialized here, as they effectively
        act as user namespaces.

        Notes
        -----
        All data structures here are only filled in, they are NOT reset by this
        method.  If they were not empty before, data will simply be added to
        them.
        """
        # This function works in two parts: first we put a few things in
        # user_ns, and we sync that contents into user_ns_hidden so that these
        # initial variables aren't shown by %who.  After the sync, we add the
        # rest of what we *do* want the user to see with %who even on a new
        # session (probably nothing, so they really only see their own stuff)

        # The user dict must *always* have a __builtin__ reference to the
        # Python standard __builtin__ namespace,  which must be imported.
        # This is so that certain operations in prompt evaluation can be
        # reliably executed with builtins.  Note that we can NOT use
        # __builtins__ (note the 's'),  because that can either be a dict or a
        # module, and can even mutate at runtime, depending on the context
        # (Python makes no guarantees on it).  In contrast, __builtin__ is
        # always a module object, though it must be explicitly imported.

        # For more details:
        # http://mail.python.org/pipermail/python-dev/2001-April/014068.html
        ns = {}

        # make global variables for user access to the histories
        ns['_ih'] = self.history_manager.input_hist_parsed
        ns['_oh'] = self.history_manager.output_hist
        ns['_dh'] = self.history_manager.dir_hist

        # user aliases to input and output histories.  These shouldn't show up
        # in %who, as they can have very large reprs.
        ns['In']  = self.history_manager.input_hist_parsed
        ns['Out'] = self.history_manager.output_hist

        # Store myself as the public api!!!
        ns['get_ipython'] = self.get_ipython

        ns['exit'] = self.exiter
        ns['quit'] = self.exiter
        ns["open"] = _modified_open

        # Sync what we've added so far to user_ns_hidden so these aren't seen
        # by %who
        self.user_ns_hidden.update(ns)

        # Anything put into ns now would show up in %who.  Think twice before
        # putting anything here, as we really want %who to show the user their
        # stuff, not our variables.

        # Finally, update the real user's namespace
        self.user_ns.update(ns)

    @property
    def all_ns_refs(self):
        """Get a list of references to all the namespace dictionaries in which
        IPython might store a user-created object.

        Note that this does not include the displayhook, which also caches
        objects from the output."""
        return [self.user_ns, self.user_global_ns, self.user_ns_hidden] + \
               [m.__dict__ for m in self._main_mod_cache.values()]

    def reset(self, new_session=True, aggressive=False):
        """Clear all internal namespaces, and attempt to release references to
        user objects.

        If new_session is True, a new history session will be opened.
        """
        # Clear histories
        self.history_manager.reset(new_session)
        # Reset counter used to index all histories
        if new_session:
            self.execution_count = 1

        # Reset last execution result
        self.last_execution_succeeded = True
        self.last_execution_result = None

        # Flush cached output items
        if self.displayhook.do_full_cache:
            self.displayhook.flush()

        # The main execution namespaces must be cleared very carefully,
        # skipping the deletion of the builtin-related keys, because doing so
        # would cause errors in many object's __del__ methods.
        if self.user_ns is not self.user_global_ns:
            self.user_ns.clear()
        ns = self.user_global_ns
        drop_keys = set(ns.keys())
        drop_keys.discard('__builtin__')
        drop_keys.discard('__builtins__')
        drop_keys.discard('__name__')
        for k in drop_keys:
            del ns[k]

        self.user_ns_hidden.clear()

        # Restore the user namespaces to minimal usability
        self.init_user_ns()
        if aggressive and not hasattr(self, "_sys_modules_keys"):
            print("Cannot restore sys.module, no snapshot")
        elif aggressive:
            print("culling sys module...")
            current_keys = set(sys.modules.keys())
            for k in current_keys - self._sys_modules_keys:
                if k.startswith("multiprocessing"):
                    continue
                del sys.modules[k]

        # Restore the default and user aliases
        self.alias_manager.clear_aliases()
        self.alias_manager.init_aliases()

        # Now define aliases that only make sense on the terminal, because they
        # need direct access to the console in a way that we can't emulate in
        # GUI or web frontend
        if os.name == 'posix':
            for cmd in ('clear', 'more', 'less', 'man'):
                if cmd not in self.magics_manager.magics['line']:
                    self.alias_manager.soft_define_alias(cmd, cmd)

        # Flush the private list of module references kept for script
        # execution protection
        self.clear_main_mod_cache()

    def del_var(self, varname, by_name=False):
        """Delete a variable from the various namespaces, so that, as
        far as possible, we're not keeping any hidden references to it.

        Parameters
        ----------
        varname : str
            The name of the variable to delete.
        by_name : bool
            If True, delete variables with the given name in each
            namespace. If False (default), find the variable in the user
            namespace, and delete references to it.
        """
        if varname in ('__builtin__', '__builtins__'):
            raise ValueError("Refusing to delete %s" % varname)

        ns_refs = self.all_ns_refs

        if by_name:                    # Delete by name
            for ns in ns_refs:
                try:
                    del ns[varname]
                except KeyError:
                    pass
        else:                         # Delete by object
            try:
                obj = self.user_ns[varname]
            except KeyError as e:
                raise NameError("name '%s' is not defined" % varname) from e
            # Also check in output history
            ns_refs.append(self.history_manager.output_hist)
            for ns in ns_refs:
                to_delete = [n for n, o in ns.items() if o is obj]
                for name in to_delete:
                    del ns[name]

            # Ensure it is removed from the last execution result
            if self.last_execution_result.result is obj:
                self.last_execution_result = None

            # displayhook keeps extra references, but not in a dictionary
            for name in ('_', '__', '___'):
                if getattr(self.displayhook, name) is obj:
                    setattr(self.displayhook, name, None)

    def reset_selective(self, regex=None):
        """Clear selective variables from internal namespaces based on a
        specified regular expression.

        Parameters
        ----------
        regex : string or compiled pattern, optional
            A regular expression pattern that will be used in searching
            variable names in the users namespaces.
        """
        if regex is not None:
            try:
                m = re.compile(regex)
            except TypeError as e:
                raise TypeError('regex must be a string or compiled pattern') from e
            # Search for keys in each namespace that match the given regex
            # If a match is found, delete the key/value pair.
            for ns in self.all_ns_refs:
                for var in ns:
                    if m.search(var):
                        del ns[var]

    def push(self, variables, interactive=True):
        """Inject a group of variables into the IPython user namespace.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the user's namespace.  If a dict, a
            simple update is done.  If a str, the string is assumed to have
            variable names separated by spaces.  A list/tuple of str can also
            be used to give the variable names.  If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        interactive : bool
            If True (default), the variables will be listed with the ``who``
            magic.
        """
        vdict = None

        # We need a dict of name/value pairs to do namespace updates.
        if isinstance(variables, dict):
            vdict = variables
        elif isinstance(variables, (str, list, tuple)):
            if isinstance(variables, str):
                vlist = variables.split()
            else:
                vlist = variables
            vdict = {}
            cf = sys._getframe(1)
            for name in vlist:
                try:
                    vdict[name] = eval(name, cf.f_globals, cf.f_locals)
                except:
                    print('Could not get variable %s from %s' %
                           (name,cf.f_code.co_name))
        else:
            raise ValueError('variables must be a dict/str/list/tuple')

        # Propagate variables to user namespace
        self.user_ns.update(vdict)

        # And configure interactive visibility
        user_ns_hidden = self.user_ns_hidden
        if interactive:
            for name in vdict:
                user_ns_hidden.pop(name, None)
        else:
            user_ns_hidden.update(vdict)

    def drop_by_id(self, variables):
        """Remove a dict of variables from the user namespace, if they are the
        same as the values in the dictionary.

        This is intended for use by extensions: variables that they've added can
        be taken back out if they are unloaded, without removing any that the
        user has overwritten.

        Parameters
        ----------
        variables : dict
            A dictionary mapping object names (as strings) to the objects.
        """
        for name, obj in variables.items():
            if name in self.user_ns and self.user_ns[name] is obj:
                del self.user_ns[name]
                self.user_ns_hidden.pop(name, None)

    #-------------------------------------------------------------------------
    # Things related to object introspection
    #-------------------------------------------------------------------------
    @staticmethod
    def _find_parts(oname: str) -> Tuple[bool, ListType[str]]:
        """
        Given an object name, return a list of parts of this object name.

        Basically split on docs when using attribute access,
        and extract the value when using square bracket.


        For example foo.bar[3].baz[x] -> foo, bar, 3, baz, x


        Returns
        -------
        parts_ok: bool
            wether we were properly able to parse parts.
        parts: list of str
            extracted parts



        """
        raw_parts = oname.split(".")
        parts = []
        parts_ok = True
        for p in raw_parts:
            if p.endswith("]"):
                var, *indices = p.split("[")
                if not var.isidentifier():
                    parts_ok = False
                    break
                parts.append(var)
                for ind in indices:
                    if ind[-1] != "]" and not is_integer_string(ind[:-1]):
                        parts_ok = False
                        break
                    parts.append(ind[:-1])
                continue

            if not p.isidentifier():
                parts_ok = False
            parts.append(p)

        return parts_ok, parts

    def _ofind(
        self, oname: str, namespaces: Optional[Sequence[Tuple[str, AnyType]]] = None
    ) -> OInfo:
        """Find an object in the available namespaces.


        Returns
        -------
        OInfo with fields:
          - ismagic
          - isalias
          - found
          - obj
          - namespac
          - parent

        Has special code to detect magic functions.
        """
        oname = oname.strip()
        parts_ok, parts = self._find_parts(oname)

        if (
            not oname.startswith(ESC_MAGIC)
            and not oname.startswith(ESC_MAGIC2)
            and not parts_ok
        ):
            return OInfo(
                ismagic=False,
                isalias=False,
                found=False,
                obj=None,
                namespace=None,
                parent=None,
            )

        if namespaces is None:
            # Namespaces to search in:
            # Put them in a list. The order is important so that we
            # find things in the same order that Python finds them.
            namespaces = [ ('Interactive', self.user_ns),
                           ('Interactive (global)', self.user_global_ns),
                           ('Python builtin', builtin_mod.__dict__),
                           ]

        ismagic = False
        isalias = False
        found = False
        ospace = None
        parent = None
        obj = None


        # Look for the given name by splitting it in parts.  If the head is
        # found, then we look for all the remaining parts as members, and only
        # declare success if we can find them all.
        oname_parts = parts
        oname_head, oname_rest = oname_parts[0],oname_parts[1:]
        for nsname,ns in namespaces:
            try:
                obj = ns[oname_head]
            except KeyError:
                continue
            else:
                for idx, part in enumerate(oname_rest):
                    try:
                        parent = obj
                        # The last part is looked up in a special way to avoid
                        # descriptor invocation as it may raise or have side
                        # effects.
                        if idx == len(oname_rest) - 1:
                            obj = self._getattr_property(obj, part)
                        else:
                            if is_integer_string(part):
                                obj = obj[int(part)]
                            else:
                                obj = getattr(obj, part)
                    except:
                        # Blanket except b/c some badly implemented objects
                        # allow __getattr__ to raise exceptions other than
                        # AttributeError, which then crashes IPython.
                        break
                else:
                    # If we finish the for loop (no break), we got all members
                    found = True
                    ospace = nsname
                    break  # namespace loop

        # Try to see if it's magic
        if not found:
            obj = None
            if oname.startswith(ESC_MAGIC2):
                oname = oname.lstrip(ESC_MAGIC2)
                obj = self.find_cell_magic(oname)
            elif oname.startswith(ESC_MAGIC):
                oname = oname.lstrip(ESC_MAGIC)
                obj = self.find_line_magic(oname)
            else:
                # search without prefix, so run? will find %run?
                obj = self.find_line_magic(oname)
                if obj is None:
                    obj = self.find_cell_magic(oname)
            if obj is not None:
                found = True
                ospace = 'IPython internal'
                ismagic = True
                isalias = isinstance(obj, Alias)

        # Last try: special-case some literals like '', [], {}, etc:
        if not found and oname_head in ["''",'""','[]','{}','()']:
            obj = eval(oname_head)
            found = True
            ospace = 'Interactive'

        return OInfo(
            obj=obj,
            found=found,
            parent=parent,
            ismagic=ismagic,
            isalias=isalias,
            namespace=ospace,
        )

    @staticmethod
    def _getattr_property(obj, attrname):
        """Property-aware getattr to use in object finding.

        If attrname represents a property, return it unevaluated (in case it has
        side effects or raises an error.

        """
        if not isinstance(obj, type):
            try:
                # `getattr(type(obj), attrname)` is not guaranteed to return
                # `obj`, but does so for property:
                #
                # property.__get__(self, None, cls) -> self
                #
                # The universal alternative is to traverse the mro manually
                # searching for attrname in class dicts.
                if is_integer_string(attrname):
                    return obj[int(attrname)]
                else:
                    attr = getattr(type(obj), attrname)
            except AttributeError:
                pass
            else:
                # This relies on the fact that data descriptors (with both
                # __get__ & __set__ magic methods) take precedence over
                # instance-level attributes:
                #
                #    class A(object):
                #        @property
                #        def foobar(self): return 123
                #    a = A()
                #    a.__dict__['foobar'] = 345
                #    a.foobar  # == 123
                #
                # So, a property may be returned right away.
                if isinstance(attr, property):
                    return attr

        # Nothing helped, fall back.
        return getattr(obj, attrname)

    def _object_find(self, oname, namespaces=None) -> OInfo:
        """Find an object and return a struct with info about it."""
        return self._ofind(oname, namespaces)

    def _inspect(self, meth, oname, namespaces=None, **kw):
        """Generic interface to the inspector system.

        This function is meant to be called by pdef, pdoc & friends.
        """
        info: OInfo = self._object_find(oname, namespaces)
        if self.sphinxify_docstring:
            if sphinxify is None:
                raise ImportError("Module ``docrepr`` required but missing")
            docformat = sphinxify(self.object_inspect(oname))
        else:
            docformat = None
        if info.found or hasattr(info.parent, oinspect.HOOK_NAME):
            pmethod = getattr(self.inspector, meth)
            # TODO: only apply format_screen to the plain/text repr of the mime
            # bundle.
            formatter = format_screen if info.ismagic else docformat
            if meth == 'pdoc':
                pmethod(info.obj, oname, formatter)
            elif meth == 'pinfo':
                pmethod(
                    info.obj,
                    oname,
                    formatter,
                    info,
                    enable_html_pager=self.enable_html_pager,
                    **kw,
                )
            else:
                pmethod(info.obj, oname)
        else:
            print('Object `%s` not found.' % oname)
            return 'not found'  # so callers can take other action

    def object_inspect(self, oname, detail_level=0):
        """Get object info about oname"""
        with self.builtin_trap:
            info = self._object_find(oname)
            if info.found:
                return self.inspector.info(info.obj, oname, info=info,
                            detail_level=detail_level
                )
            else:
                return oinspect.object_info(name=oname, found=False)

    def object_inspect_text(self, oname, detail_level=0):
        """Get object info as formatted text"""
        return self.object_inspect_mime(oname, detail_level)['text/plain']

    def object_inspect_mime(self, oname, detail_level=0, omit_sections=()):
        """Get object info as a mimebundle of formatted representations.

        A mimebundle is a dictionary, keyed by mime-type.
        It must always have the key `'text/plain'`.
        """
        with self.builtin_trap:
            info = self._object_find(oname)
            if info.found:
                docformat = (
                    sphinxify(self.object_inspect(oname))
                    if self.sphinxify_docstring
                    else None
                )
                return self.inspector._get_info(
                    info.obj,
                    oname,
                    info=info,
                    detail_level=detail_level,
                    formatter=docformat,
                    omit_sections=omit_sections,
                )
            else:
                raise KeyError(oname)

    #-------------------------------------------------------------------------
    # Things related to history management
    #-------------------------------------------------------------------------

    def init_history(self):
        """Sets up the command history, and starts regular autosaves."""
        self.history_manager = HistoryManager(shell=self, parent=self)
        self.configurables.append(self.history_manager)

    #-------------------------------------------------------------------------
    # Things related to exception handling and tracebacks (not debugging)
    #-------------------------------------------------------------------------

    debugger_cls = InterruptiblePdb

    def init_traceback_handlers(self, custom_exceptions):
        # Syntax error handler.
        self.SyntaxTB = ultratb.SyntaxTB(color_scheme='NoColor', parent=self)

        # The interactive one is initialized with an offset, meaning we always
        # want to remove the topmost item in the traceback, which is our own
        # internal code. Valid modes: ['Plain','Context','Verbose','Minimal']
        self.InteractiveTB = ultratb.AutoFormattedTB(mode = 'Plain',
                                                     color_scheme='NoColor',
                                                     tb_offset = 1,
                                   debugger_cls=self.debugger_cls, parent=self)

        # The instance will store a pointer to the system-wide exception hook,
        # so that runtime code (such as magics) can access it.  This is because
        # during the read-eval loop, it may get temporarily overwritten.
        self.sys_excepthook = sys.excepthook

        # and add any custom exception handlers the user may have specified
        self.set_custom_exc(*custom_exceptions)

        # Set the exception mode
        self.InteractiveTB.set_mode(mode=self.xmode)

    def set_custom_exc(self, exc_tuple, handler):
        """set_custom_exc(exc_tuple, handler)

        Set a custom exception handler, which will be called if any of the
        exceptions in exc_tuple occur in the mainloop (specifically, in the
        run_code() method).

        Parameters
        ----------
        exc_tuple : tuple of exception classes
            A *tuple* of exception classes, for which to call the defined
            handler.  It is very important that you use a tuple, and NOT A
            LIST here, because of the way Python's except statement works.  If
            you only want to trap a single exception, use a singleton tuple::

                exc_tuple == (MyCustomException,)

        handler : callable
            handler must have the following signature::

                def my_handler(self, etype, value, tb, tb_offset=None):
                    ...
                    return structured_traceback

            Your handler must return a structured traceback (a list of strings),
            or None.

            This will be made into an instance method (via types.MethodType)
            of IPython itself, and it will be called if any of the exceptions
            listed in the exc_tuple are caught. If the handler is None, an
            internal basic one is used, which just prints basic info.

            To protect IPython from crashes, if your handler ever raises an
            exception or returns an invalid result, it will be immediately
            disabled.

        Notes
        -----
        WARNING: by putting in your own exception handler into IPython's main
        execution loop, you run a very good chance of nasty crashes.  This
        facility should only be used if you really know what you are doing.
        """

        if not isinstance(exc_tuple, tuple):
            raise TypeError("The custom exceptions must be given as a tuple.")

        def dummy_handler(self, etype, value, tb, tb_offset=None):
            print('*** Simple custom exception handler ***')
            print('Exception type :', etype)
            print('Exception value:', value)
            print('Traceback      :', tb)

        def validate_stb(stb):
            """validate structured traceback return type

            return type of CustomTB *should* be a list of strings, but allow
            single strings or None, which are harmless.

            This function will *always* return a list of strings,
            and will raise a TypeError if stb is inappropriate.
            """
            msg = "CustomTB must return list of strings, not %r" % stb
            if stb is None:
                return []
            elif isinstance(stb, str):
                return [stb]
            elif not isinstance(stb, list):
                raise TypeError(msg)
            # it's a list
            for line in stb:
                # check every element
                if not isinstance(line, str):
                    raise TypeError(msg)
            return stb

        if handler is None:
            wrapped = dummy_handler
        else:
            def wrapped(self,etype,value,tb,tb_offset=None):
                """wrap CustomTB handler, to protect IPython from user code

                This makes it harder (but not impossible) for custom exception
                handlers to crash IPython.
                """
                try:
                    stb = handler(self,etype,value,tb,tb_offset=tb_offset)
                    return validate_stb(stb)
                except:
                    # clear custom handler immediately
                    self.set_custom_exc((), None)
                    print("Custom TB Handler failed, unregistering", file=sys.stderr)
                    # show the exception in handler first
                    stb = self.InteractiveTB.structured_traceback(*sys.exc_info())
                    print(self.InteractiveTB.stb2text(stb))
                    print("The original exception:")
                    stb = self.InteractiveTB.structured_traceback(
                                            (etype,value,tb), tb_offset=tb_offset
                    )
                return stb

        self.CustomTB = types.MethodType(wrapped,self)
        self.custom_exceptions = exc_tuple

    def excepthook(self, etype, value, tb):
        """One more defense for GUI apps that call sys.excepthook.

        GUI frameworks like wxPython trap exceptions and call
        sys.excepthook themselves.  I guess this is a feature that
        enables them to keep running after exceptions that would
        otherwise kill their mainloop. This is a bother for IPython
        which expects to catch all of the program exceptions with a try:
        except: statement.

        Normally, IPython sets sys.excepthook to a CrashHandler instance, so if
        any app directly invokes sys.excepthook, it will look to the user like
        IPython crashed.  In order to work around this, we can disable the
        CrashHandler and replace it with this excepthook instead, which prints a
        regular traceback using our InteractiveTB.  In this fashion, apps which
        call sys.excepthook will generate a regular-looking exception from
        IPython, and the CrashHandler will only be triggered by real IPython
        crashes.

        This hook should be used sparingly, only in places which are not likely
        to be true IPython errors.
        """
        self.showtraceback((etype, value, tb), tb_offset=0)

    def _get_exc_info(self, exc_tuple=None):
        """get exc_info from a given tuple, sys.exc_info() or sys.last_type etc.

        Ensures sys.last_type,value,traceback hold the exc_info we found,
        from whichever source.

        raises ValueError if none of these contain any information
        """
        if exc_tuple is None:
            etype, value, tb = sys.exc_info()
        else:
            etype, value, tb = exc_tuple

        if etype is None:
            if hasattr(sys, 'last_type'):
                etype, value, tb = sys.last_type, sys.last_value, \
                                   sys.last_traceback

        if etype is None:
            raise ValueError("No exception to find")

        # Now store the exception info in sys.last_type etc.
        # WARNING: these variables are somewhat deprecated and not
        # necessarily safe to use in a threaded environment, but tools
        # like pdb depend on their existence, so let's set them.  If we
        # find problems in the field, we'll need to revisit their use.
        sys.last_type = etype
        sys.last_value = value
        sys.last_traceback = tb

        return etype, value, tb

    def show_usage_error(self, exc):
        """Show a short message for UsageErrors

        These are special exceptions that shouldn't show a traceback.
        """
        print("UsageError: %s" % exc, file=sys.stderr)

    def get_exception_only(self, exc_tuple=None):
        """
        Return as a string (ending with a newline) the exception that
        just occurred, without any traceback.
        """
        etype, value, tb = self._get_exc_info(exc_tuple)
        msg = traceback.format_exception_only(etype, value)
        return ''.join(msg)

    def showtraceback(self, exc_tuple=None, filename=None, tb_offset=None,
                      exception_only=False, running_compiled_code=False):
        """Display the exception that just occurred.

        If nothing is known about the exception, this is the method which
        should be used throughout the code for presenting user tracebacks,
        rather than directly invoking the InteractiveTB object.

        A specific showsyntaxerror() also exists, but this method can take
        care of calling it if needed, so unless you are explicitly catching a
        SyntaxError exception, don't try to analyze the stack manually and
        simply call this method."""

        try:
            try:
                etype, value, tb = self._get_exc_info(exc_tuple)
            except ValueError:
                print('No traceback available to show.', file=sys.stderr)
                return

            if issubclass(etype, SyntaxError):
                # Though this won't be called by syntax errors in the input
                # line, there may be SyntaxError cases with imported code.
                self.showsyntaxerror(filename, running_compiled_code)
            elif etype is UsageError:
                self.show_usage_error(value)
            else:
                if exception_only:
                    stb = ['An exception has occurred, use %tb to see '
                           'the full traceback.\n']
                    stb.extend(self.InteractiveTB.get_exception_only(etype,
                                                                     value))
                else:

                    def contains_exceptiongroup(val):
                        if val is None:
                            return False
                        return isinstance(
                            val, BaseExceptionGroup
                        ) or contains_exceptiongroup(val.__context__)

                    if contains_exceptiongroup(value):
                        # fall back to native exception formatting until ultratb
                        # supports exception groups
                        traceback.print_exc()
                    else:
                        try:
                            # Exception classes can customise their traceback - we
                            # use this in IPython.parallel for exceptions occurring
                            # in the engines. This should return a list of strings.
                            if hasattr(value, "_render_traceback_"):
                                stb = value._render_traceback_()
                            else:
                                stb = self.InteractiveTB.structured_traceback(
                                    etype, value, tb, tb_offset=tb_offset
                                )

                        except Exception:
                            print(
                                "Unexpected exception formatting exception. Falling back to standard exception"
                            )
                            traceback.print_exc()
                            return None

                        self._showtraceback(etype, value, stb)
                    if self.call_pdb:
                        # drop into debugger
                        self.debugger(force=True)
                    return

                # Actually show the traceback
                self._showtraceback(etype, value, stb)

        except KeyboardInterrupt:
            print('\n' + self.get_exception_only(), file=sys.stderr)

    def _showtraceback(self, etype, evalue, stb: str):
        """Actually show a traceback.

        Subclasses may override this method to put the traceback on a different
        place, like a side channel.
        """
        val = self.InteractiveTB.stb2text(stb)
        try:
            print(val)
        except UnicodeEncodeError:
            print(val.encode("utf-8", "backslashreplace").decode())

    def showsyntaxerror(self, filename=None, running_compiled_code=False):
        """Display the syntax error that just occurred.

        This doesn't display a stack trace because there isn't one.

        If a filename is given, it is stuffed in the exception instead
        of what was there before (because Python's parser always uses
        "<string>" when reading from a string).

        If the syntax error occurred when running a compiled code (i.e. running_compile_code=True),
        longer stack trace will be displayed.
        """
        etype, value, last_traceback = self._get_exc_info()

        if filename and issubclass(etype, SyntaxError):
            try:
                value.filename = filename
            except:
                # Not the format we expect; leave it alone
                pass

        # If the error occurred when executing compiled code, we should provide full stacktrace.
        elist = traceback.extract_tb(last_traceback) if running_compiled_code else []
        stb = self.SyntaxTB.structured_traceback(etype, value, elist)
        self._showtraceback(etype, value, stb)

    # This is overridden in TerminalInteractiveShell to show a message about
    # the %paste magic.
    def showindentationerror(self):
        """Called by _run_cell when there's an IndentationError in code entered
        at the prompt.

        This is overridden in TerminalInteractiveShell to show a message about
        the %paste magic."""
        self.showsyntaxerror()

    @skip_doctest
    def set_next_input(self, s, replace=False):
        """ Sets the 'default' input string for the next command line.

        Example::

            In [1]: _ip.set_next_input("Hello Word")
            In [2]: Hello Word_  # cursor is here
        """
        self.rl_next_input = s

    def _indent_current_str(self):
        """return the current level of indentation as a string"""
        return self.input_splitter.get_indent_spaces() * ' '

    #-------------------------------------------------------------------------
    # Things related to text completion
    #-------------------------------------------------------------------------

    def init_completer(self):
        """Initialize the completion machinery.

        This creates completion machinery that can be used by client code,
        either interactively in-process (typically triggered by the readline
        library), programmatically (such as in test suites) or out-of-process
        (typically over the network by remote frontends).
        """
        from IPython.core.completer import IPCompleter
        from IPython.core.completerlib import (
            cd_completer,
            magic_run_completer,
            module_completer,
            reset_completer,
        )

        self.Completer = IPCompleter(shell=self,
                                     namespace=self.user_ns,
                                     global_namespace=self.user_global_ns,
                                     parent=self,
                                     )
        self.configurables.append(self.Completer)

        # Add custom completers to the basic ones built into IPCompleter
        sdisp = self.strdispatchers.get('complete_command', StrDispatch())
        self.strdispatchers['complete_command'] = sdisp
        self.Completer.custom_completers = sdisp

        self.set_hook('complete_command', module_completer, str_key = 'import')
        self.set_hook('complete_command', module_completer, str_key = 'from')
        self.set_hook('complete_command', module_completer, str_key = '%aimport')
        self.set_hook('complete_command', magic_run_completer, str_key = '%run')
        self.set_hook('complete_command', cd_completer, str_key = '%cd')
        self.set_hook('complete_command', reset_completer, str_key = '%reset')

    @skip_doctest
    def complete(self, text, line=None, cursor_pos=None):
        """Return the completed text and a list of completions.

        Parameters
        ----------
        text : string
            A string of text to be completed on.  It can be given as empty and
            instead a line/position pair are given.  In this case, the
            completer itself will split the line like readline does.
        line : string, optional
            The complete line that text is part of.
        cursor_pos : int, optional
            The position of the cursor on the input line.

        Returns
        -------
        text : string
            The actual text that was completed.
        matches : list
            A sorted list with all possible completions.

        Notes
        -----
        The optional arguments allow the completion to take more context into
        account, and are part of the low-level completion API.

        This is a wrapper around the completion mechanism, similar to what
        readline does at the command line when the TAB key is hit.  By
        exposing it as a method, it can be used by other non-readline
        environments (such as GUIs) for text completion.

        Examples
        --------
        In [1]: x = 'hello'

        In [2]: _ip.complete('x.l')
        Out[2]: ('x.l', ['x.ljust', 'x.lower', 'x.lstrip'])
        """

        # Inject names into __builtin__ so we can complete on the added names.
        with self.builtin_trap:
            return self.Completer.complete(text, line, cursor_pos)

    def set_custom_completer(self, completer, pos=0) -> None:
        """Adds a new custom completer function.

        The position argument (defaults to 0) is the index in the completers
        list where you want the completer to be inserted.

        `completer` should have the following signature::

            def completion(self: Completer, text: string) -> List[str]:
                raise NotImplementedError

        It will be bound to the current Completer instance and pass some text
        and return a list with current completions to suggest to the user.
        """

        newcomp = types.MethodType(completer, self.Completer)
        self.Completer.custom_matchers.insert(pos,newcomp)

    def set_completer_frame(self, frame=None):
        """Set the frame of the completer."""
        if frame:
            self.Completer.namespace = frame.f_locals
            self.Completer.global_namespace = frame.f_globals
        else:
            self.Completer.namespace = self.user_ns
            self.Completer.global_namespace = self.user_global_ns

    #-------------------------------------------------------------------------
    # Things related to magics
    #-------------------------------------------------------------------------

    def init_magics(self):
        from IPython.core import magics as m
        self.magics_manager = magic.MagicsManager(shell=self,
                                   parent=self,
                                   user_magics=m.UserMagics(self))
        self.configurables.append(self.magics_manager)

        # Expose as public API from the magics manager
        self.register_magics = self.magics_manager.register

        self.register_magics(m.AutoMagics, m.BasicMagics, m.CodeMagics,
            m.ConfigMagics, m.DisplayMagics, m.ExecutionMagics,
            m.ExtensionMagics, m.HistoryMagics, m.LoggingMagics,
            m.NamespaceMagics, m.OSMagics, m.PackagingMagics,
            m.PylabMagics, m.ScriptMagics,
        )
        self.register_magics(m.AsyncMagics)

        # Register Magic Aliases
        mman = self.magics_manager
        # FIXME: magic aliases should be defined by the Magics classes
        # or in MagicsManager, not here
        mman.register_alias('ed', 'edit')
        mman.register_alias('hist', 'history')
        mman.register_alias('rep', 'recall')
        mman.register_alias('SVG', 'svg', 'cell')
        mman.register_alias('HTML', 'html', 'cell')
        mman.register_alias('file', 'writefile', 'cell')

        # FIXME: Move the color initialization to the DisplayHook, which
        # should be split into a prompt manager and displayhook. We probably
        # even need a centralize colors management object.
        self.run_line_magic('colors', self.colors)

    # Defined here so that it's included in the documentation
    @functools.wraps(magic.MagicsManager.register_function)
    def register_magic_function(self, func, magic_kind='line', magic_name=None):
        self.magics_manager.register_function(
            func, magic_kind=magic_kind, magic_name=magic_name
        )

    def _find_with_lazy_load(self, /, type_, magic_name: str):
        """
        Try to find a magic potentially lazy-loading it.

        Parameters
        ----------

        type_: "line"|"cell"
            the type of magics we are trying to find/lazy load.
        magic_name: str
            The name of the magic we are trying to find/lazy load


        Note that this may have any side effects
        """
        finder = {"line": self.find_line_magic, "cell": self.find_cell_magic}[type_]
        fn = finder(magic_name)
        if fn is not None:
            return fn
        lazy = self.magics_manager.lazy_magics.get(magic_name)
        if lazy is None:
            return None

        self.run_line_magic("load_ext", lazy)
        res = finder(magic_name)
        return res

    def run_line_magic(self, magic_name: str, line, _stack_depth=1):
        """Execute the given line magic.

        Parameters
        ----------
        magic_name : str
            Name of the desired magic function, without '%' prefix.
        line : str
            The rest of the input line as a single string.
        _stack_depth : int
            If run_line_magic() is called from magic() then _stack_depth=2.
            This is added to ensure backward compatibility for use of 'get_ipython().magic()'
        """
        fn = self._find_with_lazy_load("line", magic_name)
        if fn is None:
            lazy = self.magics_manager.lazy_magics.get(magic_name)
            if lazy:
                self.run_line_magic("load_ext", lazy)
                fn = self.find_line_magic(magic_name)
        if fn is None:
            cm = self.find_cell_magic(magic_name)
            etpl = "Line magic function `%%%s` not found%s."
            extra = '' if cm is None else (' (But cell magic `%%%%%s` exists, '
                                    'did you mean that instead?)' % magic_name )
            raise UsageError(etpl % (magic_name, extra))
        else:
            # Note: this is the distance in the stack to the user's frame.
            # This will need to be updated if the internal calling logic gets
            # refactored, or else we'll be expanding the wrong variables.

            # Determine stack_depth depending on where run_line_magic() has been called
            stack_depth = _stack_depth
            if getattr(fn, magic.MAGIC_NO_VAR_EXPAND_ATTR, False):
                # magic has opted out of var_expand
                magic_arg_s = line
            else:
                magic_arg_s = self.var_expand(line, stack_depth)
            # Put magic args in a list so we can call with f(*a) syntax
            args = [magic_arg_s]
            kwargs = {}
            # Grab local namespace if we need it:
            if getattr(fn, "needs_local_scope", False):
                kwargs['local_ns'] = self.get_local_scope(stack_depth)
            with self.builtin_trap:
                result = fn(*args, **kwargs)

            # The code below prevents the output from being displayed
            # when using magics with decorator @output_can_be_silenced
            # when the last Python token in the expression is a ';'.
            if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):
                if DisplayHook.semicolon_at_end_of_expression(magic_arg_s):
                    return None

            return result

    def get_local_scope(self, stack_depth):
        """Get local scope at given stack depth.

        Parameters
        ----------
        stack_depth : int
            Depth relative to calling frame
        """
        return sys._getframe(stack_depth + 1).f_locals

    def run_cell_magic(self, magic_name, line, cell):
        """Execute the given cell magic.

        Parameters
        ----------
        magic_name : str
            Name of the desired magic function, without '%' prefix.
        line : str
            The rest of the first input line as a single string.
        cell : str
            The body of the cell as a (possibly multiline) string.
        """
        fn = self._find_with_lazy_load("cell", magic_name)
        if fn is None:
            lm = self.find_line_magic(magic_name)
            etpl = "Cell magic `%%{0}` not found{1}."
            extra = '' if lm is None else (' (But line magic `%{0}` exists, '
                            'did you mean that instead?)'.format(magic_name))
            raise UsageError(etpl.format(magic_name, extra))
        elif cell == '':
            message = '%%{0} is a cell magic, but the cell body is empty.'.format(magic_name)
            if self.find_line_magic(magic_name) is not None:
                message += ' Did you mean the line magic %{0} (single %)?'.format(magic_name)
            raise UsageError(message)
        else:
            # Note: this is the distance in the stack to the user's frame.
            # This will need to be updated if the internal calling logic gets
            # refactored, or else we'll be expanding the wrong variables.
            stack_depth = 2
            if getattr(fn, magic.MAGIC_NO_VAR_EXPAND_ATTR, False):
                # magic has opted out of var_expand
                magic_arg_s = line
            else:
                magic_arg_s = self.var_expand(line, stack_depth)
            kwargs = {}
            if getattr(fn, "needs_local_scope", False):
                kwargs['local_ns'] = self.user_ns

            with self.builtin_trap:
                args = (magic_arg_s, cell)
                result = fn(*args, **kwargs)

            # The code below prevents the output from being displayed
            # when using magics with decorator @output_can_be_silenced
            # when the last Python token in the expression is a ';'.
            if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):
                if DisplayHook.semicolon_at_end_of_expression(cell):
                    return None

            return result

    def find_line_magic(self, magic_name):
        """Find and return a line magic by name.

        Returns None if the magic isn't found."""
        return self.magics_manager.magics['line'].get(magic_name)

    def find_cell_magic(self, magic_name):
        """Find and return a cell magic by name.

        Returns None if the magic isn't found."""
        return self.magics_manager.magics['cell'].get(magic_name)

    def find_magic(self, magic_name, magic_kind='line'):
        """Find and return a magic of the given type by name.

        Returns None if the magic isn't found."""
        return self.magics_manager.magics[magic_kind].get(magic_name)

    def magic(self, arg_s):
        """
        DEPRECATED

        Deprecated since IPython 0.13 (warning added in
        8.1), use run_line_magic(magic_name, parameter_s).

        Call a magic function by name.

        Input: a string containing the name of the magic function to call and
        any additional arguments to be passed to the magic.

        magic('name -opt foo bar') is equivalent to typing at the ipython
        prompt:

        In[1]: %name -opt foo bar

        To call a magic without arguments, simply use magic('name').

        This provides a proper Python function to call IPython's magics in any
        valid Python code you can type at the interpreter, including loops and
        compound statements.
        """
        warnings.warn(
            "`magic(...)` is deprecated since IPython 0.13 (warning added in "
            "8.1), use run_line_magic(magic_name, parameter_s).",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: should we issue a loud deprecation warning here?
        magic_name, _, magic_arg_s = arg_s.partition(' ')
        magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
        return self.run_line_magic(magic_name, magic_arg_s, _stack_depth=2)

    #-------------------------------------------------------------------------
    # Things related to macros
    #-------------------------------------------------------------------------

    def define_macro(self, name, themacro):
        """Define a new macro

        Parameters
        ----------
        name : str
            The name of the macro.
        themacro : str or Macro
            The action to do upon invoking the macro.  If a string, a new
            Macro object is created by passing the string to it.
        """

        from IPython.core import macro

        if isinstance(themacro, str):
            themacro = macro.Macro(themacro)
        if not isinstance(themacro, macro.Macro):
            raise ValueError('A macro must be a string or a Macro instance.')
        self.user_ns[name] = themacro

    #-------------------------------------------------------------------------
    # Things related to the running of system commands
    #-------------------------------------------------------------------------

    def system_piped(self, cmd):
        """Call the given cmd in a subprocess, piping stdout/err

        Parameters
        ----------
        cmd : str
            Command to execute (can not end in '&', as background processes are
            not supported.  Should not be a command that expects input
            other than simple text.
        """
        if cmd.rstrip().endswith('&'):
            # this is *far* from a rigorous test
            # We do not support backgrounding processes because we either use
            # pexpect or pipes to read from.  Users can always just call
            # os.system() or use ip.system=ip.system_raw
            # if they really want a background process.
            raise OSError("Background processes not supported.")

        # we explicitly do NOT return the subprocess status code, because
        # a non-None value would trigger :func:`sys.displayhook` calls.
        # Instead, we store the exit_code in user_ns.
        self.user_ns['_exit_code'] = system(self.var_expand(cmd, depth=1))

    def system_raw(self, cmd):
        """Call the given cmd in a subprocess using os.system on Windows or
        subprocess.call using the system shell on other platforms.

        Parameters
        ----------
        cmd : str
            Command to execute.
        """
        cmd = self.var_expand(cmd, depth=1)
        # warn if there is an IPython magic alternative.
        main_cmd = cmd.split()[0]
        has_magic_alternatives = ("pip", "conda", "cd")

        if main_cmd in has_magic_alternatives:
            warnings.warn(
                (
                    "You executed the system command !{0} which may not work "
                    "as expected. Try the IPython magic %{0} instead."
                ).format(main_cmd)
            )

        # protect os.system from UNC paths on Windows, which it can't handle:
        if sys.platform == 'win32':
            from IPython.utils._process_win32 import AvoidUNCPath
            with AvoidUNCPath() as path:
                if path is not None:
                    cmd = '"pushd %s &&"%s' % (path, cmd)
                try:
                    ec = os.system(cmd)
                except KeyboardInterrupt:
                    print('\n' + self.get_exception_only(), file=sys.stderr)
                    ec = -2
        else:
            # For posix the result of the subprocess.call() below is an exit
            # code, which by convention is zero for success, positive for
            # program failure.  Exit codes above 128 are reserved for signals,
            # and the formula for converting a signal to an exit code is usually
            # signal_number+128.  To more easily differentiate between exit
            # codes and signals, ipython uses negative numbers.  For instance
            # since control-c is signal 2 but exit code 130, ipython's
            # _exit_code variable will read -2.  Note that some shells like
            # csh and fish don't follow sh/bash conventions for exit codes.
            executable = os.environ.get('SHELL', None)
            try:
                # Use env shell instead of default /bin/sh
                ec = subprocess.call(cmd, shell=True, executable=executable)
            except KeyboardInterrupt:
                # intercept control-C; a long traceback is not useful here
                print('\n' + self.get_exception_only(), file=sys.stderr)
                ec = 130
            if ec > 128:
                ec = -(ec - 128)

        # We explicitly do NOT return the subprocess status code, because
        # a non-None value would trigger :func:`sys.displayhook` calls.
        # Instead, we store the exit_code in user_ns.  Note the semantics
        # of _exit_code: for control-c, _exit_code == -signal.SIGNIT,
        # but raising SystemExit(_exit_code) will give status 254!
        self.user_ns['_exit_code'] = ec

    # use piped system by default, because it is better behaved
    system = system_piped

    def getoutput(self, cmd, split=True, depth=0):
        """Get output (possibly including stderr) from a subprocess.

        Parameters
        ----------
        cmd : str
            Command to execute (can not end in '&', as background processes are
            not supported.
        split : bool, optional
            If True, split the output into an IPython SList.  Otherwise, an
            IPython LSString is returned.  These are objects similar to normal
            lists and strings, with a few convenience attributes for easier
            manipulation of line-based output.  You can use '?' on them for
            details.
        depth : int, optional
            How many frames above the caller are the local variables which should
            be expanded in the command string? The default (0) assumes that the
            expansion variables are in the stack frame calling this function.
        """
        if cmd.rstrip().endswith('&'):
            # this is *far* from a rigorous test
            raise OSError("Background processes not supported.")
        out = getoutput(self.var_expand(cmd, depth=depth+1))
        if split:
            out = SList(out.splitlines())
        else:
            out = LSString(out)
        return out

    #-------------------------------------------------------------------------
    # Things related to aliases
    #-------------------------------------------------------------------------

    def init_alias(self):
        self.alias_manager = AliasManager(shell=self, parent=self)
        self.configurables.append(self.alias_manager)

    #-------------------------------------------------------------------------
    # Things related to extensions
    #-------------------------------------------------------------------------

    def init_extension_manager(self):
        self.extension_manager = ExtensionManager(shell=self, parent=self)
        self.configurables.append(self.extension_manager)

    #-------------------------------------------------------------------------
    # Things related to payloads
    #-------------------------------------------------------------------------

    def init_payload(self):
        self.payload_manager = PayloadManager(parent=self)
        self.configurables.append(self.payload_manager)

    #-------------------------------------------------------------------------
    # Things related to the prefilter
    #-------------------------------------------------------------------------

    def init_prefilter(self):
        self.prefilter_manager = PrefilterManager(shell=self, parent=self)
        self.configurables.append(self.prefilter_manager)
        # Ultimately this will be refactored in the new interpreter code, but
        # for now, we should expose the main prefilter method (there's legacy
        # code out there that may rely on this).
        self.prefilter = self.prefilter_manager.prefilter_lines

    def auto_rewrite_input(self, cmd):
        """Print to the screen the rewritten form of the user's command.

        This shows visual feedback by rewriting input lines that cause
        automatic calling to kick in, like::

          /f x

        into::

          ------> f(x)

        after the user's input prompt.  This helps the user understand that the
        input line was transformed automatically by IPython.
        """
        if not self.show_rewritten_input:
            return

        # This is overridden in TerminalInteractiveShell to use fancy prompts
        print("------> " + cmd)

    #-------------------------------------------------------------------------
    # Things related to extracting values/expressions from kernel and user_ns
    #-------------------------------------------------------------------------

    def _user_obj_error(self):
        """return simple exception dict

        for use in user_expressions
        """

        etype, evalue, tb = self._get_exc_info()
        stb = self.InteractiveTB.get_exception_only(etype, evalue)

        exc_info = {
            "status": "error",
            "traceback": stb,
            "ename": etype.__name__,
            "evalue": py3compat.safe_unicode(evalue),
        }

        return exc_info

    def _format_user_obj(self, obj):
        """format a user object to display dict

        for use in user_expressions
        """

        data, md = self.display_formatter.format(obj)
        value = {
            'status' : 'ok',
            'data' : data,
            'metadata' : md,
        }
        return value

    def user_expressions(self, expressions):
        """Evaluate a dict of expressions in the user's namespace.

        Parameters
        ----------
        expressions : dict
            A dict with string keys and string values.  The expression values
            should be valid Python expressions, each of which will be evaluated
            in the user namespace.

        Returns
        -------
        A dict, keyed like the input expressions dict, with the rich mime-typed
        display_data of each value.
        """
        out = {}
        user_ns = self.user_ns
        global_ns = self.user_global_ns

        for key, expr in expressions.items():
            try:
                value = self._format_user_obj(eval(expr, global_ns, user_ns))
            except:
                value = self._user_obj_error()
            out[key] = value
        return out

    #-------------------------------------------------------------------------
    # Things related to the running of code
    #-------------------------------------------------------------------------

    def ex(self, cmd):
        """Execute a normal python statement in user namespace."""
        with self.builtin_trap:
            exec(cmd, self.user_global_ns, self.user_ns)

    def ev(self, expr):
        """Evaluate python expression expr in user namespace.

        Returns the result of evaluation
        """
        with self.builtin_trap:
            return eval(expr, self.user_global_ns, self.user_ns)

    def safe_execfile(self, fname, *where, exit_ignore=False, raise_exceptions=False, shell_futures=False):
        """A safe version of the builtin execfile().

        This version will never throw an exception, but instead print
        helpful error messages to the screen.  This only works on pure
        Python files with the .py extension.

        Parameters
        ----------
        fname : string
            The name of the file to be executed.
        *where : tuple
            One or two namespaces, passed to execfile() as (globals,locals).
            If only one is given, it is passed as both.
        exit_ignore : bool (False)
            If True, then silence SystemExit for non-zero status (it is always
            silenced for zero status, as it is so common).
        raise_exceptions : bool (False)
            If True raise exceptions everywhere. Meant for testing.
        shell_futures : bool (False)
            If True, the code will share future statements with the interactive
            shell. It will both be affected by previous __future__ imports, and
            any __future__ imports in the code will affect the shell. If False,
            __future__ imports are not shared in either direction.

        """
        fname = Path(fname).expanduser().resolve()

        # Make sure we can open the file
        try:
            with fname.open("rb"):
                pass
        except:
            warn('Could not open file <%s> for safe execution.' % fname)
            return

        # Find things also in current directory.  This is needed to mimic the
        # behavior of running a script from the system command line, where
        # Python inserts the script's directory into sys.path
        dname = str(fname.parent)

        with prepended_to_syspath(dname), self.builtin_trap:
            try:
                glob, loc = (where + (None, ))[:2]
                py3compat.execfile(
                    fname, glob, loc,
                    self.compile if shell_futures else None)
            except SystemExit as status:
                # If the call was made with 0 or None exit status (sys.exit(0)
                # or sys.exit() ), don't bother showing a traceback, as both of
                # these are considered normal by the OS:
                # > python -c'import sys;sys.exit(0)'; echo $?
                # 0
                # > python -c'import sys;sys.exit()'; echo $?
                # 0
                # For other exit status, we show the exception unless
                # explicitly silenced, but only in short form.
                if status.code:
                    if raise_exceptions:
                        raise
                    if not exit_ignore:
                        self.showtraceback(exception_only=True)
            except:
                if raise_exceptions:
                    raise
                # tb offset is 2 because we wrap execfile
                self.showtraceback(tb_offset=2)

    def safe_execfile_ipy(self, fname, shell_futures=False, raise_exceptions=False):
        """Like safe_execfile, but for .ipy or .ipynb files with IPython syntax.

        Parameters
        ----------
        fname : str
            The name of the file to execute.  The filename must have a
            .ipy or .ipynb extension.
        shell_futures : bool (False)
            If True, the code will share future statements with the interactive
            shell. It will both be affected by previous __future__ imports, and
            any __future__ imports in the code will affect the shell. If False,
            __future__ imports are not shared in either direction.
        raise_exceptions : bool (False)
            If True raise exceptions everywhere.  Meant for testing.
        """
        fname = Path(fname).expanduser().resolve()

        # Make sure we can open the file
        try:
            with fname.open("rb"):
                pass
        except:
            warn('Could not open file <%s> for safe execution.' % fname)
            return

        # Find things also in current directory.  This is needed to mimic the
        # behavior of running a script from the system command line, where
        # Python inserts the script's directory into sys.path
        dname = str(fname.parent)

        def get_cells():
            """generator for sequence of code blocks to run"""
            if fname.suffix == ".ipynb":
                from nbformat import read
                nb = read(fname, as_version=4)
                if not nb.cells:
                    return
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        yield cell.source
            else:
                yield fname.read_text(encoding="utf-8")

        with prepended_to_syspath(dname):
            try:
                for cell in get_cells():
                    result = self.run_cell(cell, silent=True, shell_futures=shell_futures)
                    if raise_exceptions:
                        result.raise_error()
                    elif not result.success:
                        break
            except:
                if raise_exceptions:
                    raise
                self.showtraceback()
                warn('Unknown failure executing file: <%s>' % fname)

    def safe_run_module(self, mod_name, where):
        """A safe version of runpy.run_module().

        This version will never throw an exception, but instead print
        helpful error messages to the screen.

        `SystemExit` exceptions with status code 0 or None are ignored.

        Parameters
        ----------
        mod_name : string
            The name of the module to be executed.
        where : dict
            The globals namespace.
        """
        try:
            try:
                where.update(
                    runpy.run_module(str(mod_name), run_name="__main__",
                                     alter_sys=True)
                            )
            except SystemExit as status:
                if status.code:
                    raise
        except:
            self.showtraceback()
            warn('Unknown failure executing module: <%s>' % mod_name)

    def run_cell(
        self,
        raw_cell,
        store_history=False,
        silent=False,
        shell_futures=True,
        cell_id=None,
    ):
        """Run a complete IPython cell.

        Parameters
        ----------
        raw_cell : str
            The code (including IPython code such as %magic functions) to run.
        store_history : bool
            If True, the raw and translated cell will be stored in IPython's
            history. For user code calling back into IPython's machinery, this
            should be set to False.
        silent : bool
            If True, avoid side-effects, such as implicit displayhooks and
            and logging.  silent=True forces store_history=False.
        shell_futures : bool
            If True, the code will share future statements with the interactive
            shell. It will both be affected by previous __future__ imports, and
            any __future__ imports in the code will affect the shell. If False,
            __future__ imports are not shared in either direction.

        Returns
        -------
        result : :class:`ExecutionResult`
        """
        result = None
        try:
            result = self._run_cell(
                raw_cell, store_history, silent, shell_futures, cell_id
            )
        finally:
            self.events.trigger('post_execute')
            if not silent:
                self.events.trigger('post_run_cell', result)
        return result

    def _run_cell(
        self,
        raw_cell: str,
        store_history: bool,
        silent: bool,
        shell_futures: bool,
        cell_id: str,
    ) -> ExecutionResult:
        """Internal method to run a complete IPython cell."""

        # we need to avoid calling self.transform_cell multiple time on the same thing
        # so we need to store some results:
        preprocessing_exc_tuple = None
        try:
            transformed_cell = self.transform_cell(raw_cell)
        except Exception:
            transformed_cell = raw_cell
            preprocessing_exc_tuple = sys.exc_info()

        assert transformed_cell is not None
        coro = self.run_cell_async(
            raw_cell,
            store_history=store_history,
            silent=silent,
            shell_futures=shell_futures,
            transformed_cell=transformed_cell,
            preprocessing_exc_tuple=preprocessing_exc_tuple,
            cell_id=cell_id,
        )

        # run_cell_async is async, but may not actually need an eventloop.
        # when this is the case, we want to run it using the pseudo_sync_runner
        # so that code can invoke eventloops (for example via the %run , and
        # `%paste` magic.
        if self.trio_runner:
            runner = self.trio_runner
        elif self.should_run_async(
            raw_cell,
            transformed_cell=transformed_cell,
            preprocessing_exc_tuple=preprocessing_exc_tuple,
        ):
            runner = self.loop_runner
        else:
            runner = _pseudo_sync_runner

        try:
            result = runner(coro)
        except BaseException as e:
            info = ExecutionInfo(
                raw_cell, store_history, silent, shell_futures, cell_id
            )
            result = ExecutionResult(info)
            result.error_in_exec = e
            self.showtraceback(running_compiled_code=True)
        finally:
            return result

    def should_run_async(
        self, raw_cell: str, *, transformed_cell=None, preprocessing_exc_tuple=None
    ) -> bool:
        """Return whether a cell should be run asynchronously via a coroutine runner

        Parameters
        ----------
        raw_cell : str
            The code to be executed

        Returns
        -------
        result: bool
            Whether the code needs to be run with a coroutine runner or not
        .. versionadded:: 7.0
        """
        if not self.autoawait:
            return False
        if preprocessing_exc_tuple is not None:
            return False
        assert preprocessing_exc_tuple is None
        if transformed_cell is None:
            warnings.warn(
                "`should_run_async` will not call `transform_cell`"
                " automatically in the future. Please pass the result to"
                " `transformed_cell` argument and any exception that happen"
                " during the"
                "transform in `preprocessing_exc_tuple` in"
                " IPython 7.17 and above.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                cell = self.transform_cell(raw_cell)
            except Exception:
                # any exception during transform will be raised
                # prior to execution
                return False
        else:
            cell = transformed_cell
        return _should_be_async(cell)

    async def run_cell_async(
        self,
        raw_cell: str,
        store_history=False,
        silent=False,
        shell_futures=True,
        *,
        transformed_cell: Optional[str] = None,
        preprocessing_exc_tuple: Optional[AnyType] = None,
        cell_id=None,
    ) -> ExecutionResult:
        """Run a complete IPython cell asynchronously.

        Parameters
        ----------
        raw_cell : str
          The code (including IPython code such as %magic functions) to run.
        store_history : bool
          If True, the raw and translated cell will be stored in IPython's
          history. For user code calling back into IPython's machinery, this
          should be set to False.
        silent : bool
          If True, avoid side-effects, such as implicit displayhooks and
          and logging.  silent=True forces store_history=False.
        shell_futures : bool
          If True, the code will share future statements with the interactive
          shell. It will both be affected by previous __future__ imports, and
          any __future__ imports in the code will affect the shell. If False,
          __future__ imports are not shared in either direction.
        transformed_cell: str
          cell that was passed through transformers
        preprocessing_exc_tuple:
          trace if the transformation failed.

        Returns
        -------
        result : :class:`ExecutionResult`

        .. versionadded:: 7.0
        """
        info = ExecutionInfo(raw_cell, store_history, silent, shell_futures, cell_id)
        result = ExecutionResult(info)

        if (not raw_cell) or raw_cell.isspace():
            self.last_execution_succeeded = True
            self.last_execution_result = result
            return result

        if silent:
            store_history = False

        if store_history:
            result.execution_count = self.execution_count

        def error_before_exec(value):
            if store_history:
                self.execution_count += 1
            result.error_before_exec = value
            self.last_execution_succeeded = False
            self.last_execution_result = result
            return result

        self.events.trigger('pre_execute')
        if not silent:
            self.events.trigger('pre_run_cell', info)

        if transformed_cell is None:
            warnings.warn(
                "`run_cell_async` will not call `transform_cell`"
                " automatically in the future. Please pass the result to"
                " `transformed_cell` argument and any exception that happen"
                " during the"
                "transform in `preprocessing_exc_tuple` in"
                " IPython 7.17 and above.",
                DeprecationWarning,
                stacklevel=2,
            )
            # If any of our input transformation (input_transformer_manager or
            # prefilter_manager) raises an exception, we store it in this variable
            # so that we can display the error after logging the input and storing
            # it in the history.
            try:
                cell = self.transform_cell(raw_cell)
            except Exception:
                preprocessing_exc_tuple = sys.exc_info()
                cell = raw_cell  # cell has to exist so it can be stored/logged
            else:
                preprocessing_exc_tuple = None
        else:
            if preprocessing_exc_tuple is None:
                cell = transformed_cell
            else:
                cell = raw_cell

        # Do NOT store paste/cpaste magic history
        if "get_ipython().run_line_magic(" in cell and "paste" in cell:
            store_history = False

        # Store raw and processed history
        if store_history:
            self.history_manager.store_inputs(self.execution_count, cell, raw_cell)
        if not silent:
            self.logger.log(cell, raw_cell)

        # Display the exception if input processing failed.
        if preprocessing_exc_tuple is not None:
            self.showtraceback(preprocessing_exc_tuple)
            if store_history:
                self.execution_count += 1
            return error_before_exec(preprocessing_exc_tuple[1])

        # Our own compiler remembers the __future__ environment. If we want to
        # run code with a separate __future__ environment, use the default
        # compiler
        compiler = self.compile if shell_futures else self.compiler_class()

        _run_async = False

        with self.builtin_trap:
            cell_name = compiler.cache(cell, self.execution_count, raw_code=raw_cell)

            with self.display_trap:
                # Compile to bytecode
                try:
                    code_ast = compiler.ast_parse(cell, filename=cell_name)
                except self.custom_exceptions as e:
                    etype, value, tb = sys.exc_info()
                    self.CustomTB(etype, value, tb)
                    return error_before_exec(e)
                except IndentationError as e:
                    self.showindentationerror()
                    return error_before_exec(e)
                except (OverflowError, SyntaxError, ValueError, TypeError,
                        MemoryError) as e:
                    self.showsyntaxerror()
                    return error_before_exec(e)

                # Apply AST transformations
                try:
                    code_ast = self.transform_ast(code_ast)
                except InputRejected as e:
                    self.showtraceback()
                    return error_before_exec(e)

                # Give the displayhook a reference to our ExecutionResult so it
                # can fill in the output value.
                self.displayhook.exec_result = result

                # Execute the user code
                interactivity = "none" if silent else self.ast_node_interactivity


                has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
                       interactivity=interactivity, compiler=compiler, result=result)

                self.last_execution_succeeded = not has_raised
                self.last_execution_result = result

                # Reset this so later displayed values do not modify the
                # ExecutionResult
                self.displayhook.exec_result = None

        if store_history:
            # Write output to the database. Does nothing unless
            # history output logging is enabled.
            self.history_manager.store_output(self.execution_count)
            # Each cell is a *single* input, regardless of how many lines it has
            self.execution_count += 1

        return result

    def transform_cell(self, raw_cell):
        """Transform an input cell before parsing it.

        Static transformations, implemented in IPython.core.inputtransformer2,
        deal with things like ``%magic`` and ``!system`` commands.
        These run on all input.
        Dynamic transformations, for things like unescaped magics and the exit
        autocall, depend on the state of the interpreter.
        These only apply to single line inputs.

        These string-based transformations are followed by AST transformations;
        see :meth:`transform_ast`.
        """
        # Static input transformations
        cell = self.input_transformer_manager.transform_cell(raw_cell)

        if len(cell.splitlines()) == 1:
            # Dynamic transformations - only applied for single line commands
            with self.builtin_trap:
                # use prefilter_lines to handle trailing newlines
                # restore trailing newline for ast.parse
                cell = self.prefilter_manager.prefilter_lines(cell) + '\n'

        lines = cell.splitlines(keepends=True)
        for transform in self.input_transformers_post:
            lines = transform(lines)
        cell = ''.join(lines)

        return cell

    def transform_ast(self, node):
        """Apply the AST transformations from self.ast_transformers

        Parameters
        ----------
        node : ast.Node
            The root node to be transformed. Typically called with the ast.Module
            produced by parsing user input.

        Returns
        -------
        An ast.Node corresponding to the node it was called with. Note that it
        may also modify the passed object, so don't rely on references to the
        original AST.
        """
        for transformer in self.ast_transformers:
            try:
                node = transformer.visit(node)
            except InputRejected:
                # User-supplied AST transformers can reject an input by raising
                # an InputRejected.  Short-circuit in this case so that we
                # don't unregister the transform.
                raise
            except Exception as e:
                warn(
                    "AST transformer %r threw an error. It will be unregistered. %s"
                    % (transformer, e)
                )
                self.ast_transformers.remove(transformer)

        if self.ast_transformers:
            ast.fix_missing_locations(node)
        return node

    async def run_ast_nodes(
        self,
        nodelist: ListType[stmt],
        cell_name: str,
        interactivity="last_expr",
        compiler=compile,
        result=None,
    ):
        """Run a sequence of AST nodes. The execution mode depends on the
        interactivity parameter.

        Parameters
        ----------
        nodelist : list
          A sequence of AST nodes to run.
        cell_name : str
          Will be passed to the compiler as the filename of the cell. Typically
          the value returned by ip.compile.cache(cell).
        interactivity : str
          'all', 'last', 'last_expr' , 'last_expr_or_assign' or 'none',
          specifying which nodes should be run interactively (displaying output
          from expressions). 'last_expr' will run the last node interactively
          only if it is an expression (i.e. expressions in loops or other blocks
          are not displayed) 'last_expr_or_assign' will run the last expression
          or the last assignment. Other values for this parameter will raise a
          ValueError.

        compiler : callable
          A function with the same interface as the built-in compile(), to turn
          the AST nodes into code objects. Default is the built-in compile().
        result : ExecutionResult, optional
          An object to store exceptions that occur during execution.

        Returns
        -------
        True if an exception occurred while running code, False if it finished
        running.
        """
        if not nodelist:
            return


        if interactivity == 'last_expr_or_assign':
            if isinstance(nodelist[-1], _assign_nodes):
                asg = nodelist[-1]
                if isinstance(asg, ast.Assign) and len(asg.targets) == 1:
                    target = asg.targets[0]
                elif isinstance(asg, _single_targets_nodes):
                    target = asg.target
                else:
                    target = None
                if isinstance(target, ast.Name):
                    nnode = ast.Expr(ast.Name(target.id, ast.Load()))
                    ast.fix_missing_locations(nnode)
                    nodelist.append(nnode)
            interactivity = 'last_expr'

        _async = False
        if interactivity == 'last_expr':
            if isinstance(nodelist[-1], ast.Expr):
                interactivity = "last"
            else:
                interactivity = "none"

        if interactivity == 'none':
            to_run_exec, to_run_interactive = nodelist, []
        elif interactivity == 'last':
            to_run_exec, to_run_interactive = nodelist[:-1], nodelist[-1:]
        elif interactivity == 'all':
            to_run_exec, to_run_interactive = [], nodelist
        else:
            raise ValueError("Interactivity was %r" % interactivity)

        try:

            def compare(code):
                is_async = inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE
                return is_async

            # refactor that to just change the mod constructor.
            to_run = []
            for node in to_run_exec:
                to_run.append((node, "exec"))

            for node in to_run_interactive:
                to_run.append((node, "single"))

            for node, mode in to_run:
                if mode == "exec":
                    mod = Module([node], [])
                elif mode == "single":
                    mod = ast.Interactive([node])  # type: ignore
                with compiler.extra_flags(
                    getattr(ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0x0)
                    if self.autoawait
                    else 0x0
                ):
                    code = compiler(mod, cell_name, mode)
                    asy = compare(code)
                if await self.run_code(code, result, async_=asy):
                    return True

            # Flush softspace
            if softspace(sys.stdout, 0):
                print()

        except:
            # It's possible to have exceptions raised here, typically by
            # compilation of odd code (such as a naked 'return' outside a
            # function) that did parse but isn't valid. Typically the exception
            # is a SyntaxError, but it's safest just to catch anything and show
            # the user a traceback.

            # We do only one try/except outside the loop to minimize the impact
            # on runtime, and also because if any node in the node list is
            # broken, we should stop execution completely.
            if result:
                result.error_before_exec = sys.exc_info()[1]
            self.showtraceback()
            return True

        return False

    async def run_code(self, code_obj, result=None, *, async_=False):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to display a
        traceback.

        Parameters
        ----------
        code_obj : code object
          A compiled code object, to be executed
        result : ExecutionResult, optional
          An object to store exceptions that occur during execution.
        async_ :  Bool (Experimental)
          Attempt to run top-level asynchronous code in a default loop.

        Returns
        -------
        False : successful execution.
        True : an error occurred.
        """
        # special value to say that anything above is IPython and should be
        # hidden.
        __tracebackhide__ = "__ipython_bottom__"
        # Set our own excepthook in case the user code tries to call it
        # directly, so that the IPython crash handler doesn't get triggered
        old_excepthook, sys.excepthook = sys.excepthook, self.excepthook

        # we save the original sys.excepthook in the instance, in case config
        # code (such as magics) needs access to it.
        self.sys_excepthook = old_excepthook
        outflag = True  # happens in more places, so it's easier as default
        try:
            try:
                if async_:
                    await eval(code_obj, self.user_global_ns, self.user_ns)
                else:
                    exec(code_obj, self.user_global_ns, self.user_ns)
            finally:
                # Reset our crash handler in place
                sys.excepthook = old_excepthook
        except SystemExit as e:
            if result is not None:
                result.error_in_exec = e
            self.showtraceback(exception_only=True)
            warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
        except bdb.BdbQuit:
            etype, value, tb = sys.exc_info()
            if result is not None:
                result.error_in_exec = value
            # the BdbQuit stops here
        except self.custom_exceptions:
            etype, value, tb = sys.exc_info()
            if result is not None:
                result.error_in_exec = value
            self.CustomTB(etype, value, tb)
        except:
            if result is not None:
                result.error_in_exec = sys.exc_info()[1]
            self.showtraceback(running_compiled_code=True)
        else:
            outflag = False
        return outflag

    # For backwards compatibility
    runcode = run_code

    def check_complete(self, code: str) -> Tuple[str, str]:
        """Return whether a block of code is ready to execute, or should be continued

        Parameters
        ----------
        code : string
            Python input code, which can be multiline.

        Returns
        -------
        status : str
            One of 'complete', 'incomplete', or 'invalid' if source is not a
            prefix of valid code.
        indent : str
            When status is 'incomplete', this is some whitespace to insert on
            the next line of the prompt.
        """
        status, nspaces = self.input_transformer_manager.check_complete(code)
        return status, ' ' * (nspaces or 0)

    #-------------------------------------------------------------------------
    # Things related to GUI support and pylab
    #-------------------------------------------------------------------------

    active_eventloop = None

    def enable_gui(self, gui=None):
        raise NotImplementedError('Implement enable_gui in a subclass')

    def enable_matplotlib(self, gui=None):
        """Enable interactive matplotlib and inline figure support.

        This takes the following steps:

        1. select the appropriate eventloop and matplotlib backend
        2. set up matplotlib for interactive use with that backend
        3. configure formatters for inline figure display
        4. enable the selected gui eventloop

        Parameters
        ----------
        gui : optional, string
            If given, dictates the choice of matplotlib GUI backend to use
            (should be one of IPython's supported backends, 'qt', 'osx', 'tk',
            'gtk', 'wx' or 'inline'), otherwise we use the default chosen by
            matplotlib (as dictated by the matplotlib build-time options plus the
            user's matplotlibrc configuration file).  Note that not all backends
            make sense in all contexts, for example a terminal ipython can't
            display figures inline.
        """
        from matplotlib_inline.backend_inline import configure_inline_support

        from IPython.core import pylabtools as pt
        gui, backend = pt.find_gui_and_backend(gui, self.pylab_gui_select)

        if gui != 'inline':
            # If we have our first gui selection, store it
            if self.pylab_gui_select is None:
                self.pylab_gui_select = gui
            # Otherwise if they are different
            elif gui != self.pylab_gui_select:
                print('Warning: Cannot change to a different GUI toolkit: %s.'
                        ' Using %s instead.' % (gui, self.pylab_gui_select))
                gui, backend = pt.find_gui_and_backend(self.pylab_gui_select)

        pt.activate_matplotlib(backend)
        configure_inline_support(self, backend)

        # Now we must activate the gui pylab wants to use, and fix %run to take
        # plot updates into account
        self.enable_gui(gui)
        self.magics_manager.registry['ExecutionMagics'].default_runner = \
            pt.mpl_runner(self.safe_execfile)

        return gui, backend

    def enable_pylab(self, gui=None, import_all=True, welcome_message=False):
        """Activate pylab support at runtime.

        This turns on support for matplotlib, preloads into the interactive
        namespace all of numpy and pylab, and configures IPython to correctly
        interact with the GUI event loop.  The GUI backend to be used can be
        optionally selected with the optional ``gui`` argument.

        This method only adds preloading the namespace to InteractiveShell.enable_matplotlib.

        Parameters
        ----------
        gui : optional, string
            If given, dictates the choice of matplotlib GUI backend to use
            (should be one of IPython's supported backends, 'qt', 'osx', 'tk',
            'gtk', 'wx' or 'inline'), otherwise we use the default chosen by
            matplotlib (as dictated by the matplotlib build-time options plus the
            user's matplotlibrc configuration file).  Note that not all backends
            make sense in all contexts, for example a terminal ipython can't
            display figures inline.
        import_all : optional, bool, default: True
            Whether to do `from numpy import *` and `from pylab import *`
            in addition to module imports.
        welcome_message : deprecated
            This argument is ignored, no welcome message will be displayed.
        """
        from IPython.core.pylabtools import import_pylab

        gui, backend = self.enable_matplotlib(gui)

        # We want to prevent the loading of pylab to pollute the user's
        # namespace as shown by the %who* magics, so we execute the activation
        # code in an empty namespace, and we update *both* user_ns and
        # user_ns_hidden with this information.
        ns = {}
        import_pylab(ns, import_all)
        # warn about clobbered names
        ignored = {"__builtins__"}
        both = set(ns).intersection(self.user_ns).difference(ignored)
        clobbered = [ name for name in both if self.user_ns[name] is not ns[name] ]
        self.user_ns.update(ns)
        self.user_ns_hidden.update(ns)
        return gui, backend, clobbered

    #-------------------------------------------------------------------------
    # Utilities
    #-------------------------------------------------------------------------

    def var_expand(self, cmd, depth=0, formatter=DollarFormatter()):
        """Expand python variables in a string.

        The depth argument indicates how many frames above the caller should
        be walked to look for the local namespace where to expand variables.

        The global namespace for expansion is always the user's interactive
        namespace.
        """
        ns = self.user_ns.copy()
        try:
            frame = sys._getframe(depth+1)
        except ValueError:
            # This is thrown if there aren't that many frames on the stack,
            # e.g. if a script called run_line_magic() directly.
            pass
        else:
            ns.update(frame.f_locals)

        try:
            # We have to use .vformat() here, because 'self' is a valid and common
            # name, and expanding **ns for .format() would make it collide with
            # the 'self' argument of the method.
            cmd = formatter.vformat(cmd, args=[], kwargs=ns)
        except Exception:
            # if formatter couldn't format, just let it go untransformed
            pass
        return cmd

    def mktempfile(self, data=None, prefix='ipython_edit_'):
        """Make a new tempfile and return its filename.

        This makes a call to tempfile.mkstemp (created in a tempfile.mkdtemp),
        but it registers the created filename internally so ipython cleans it up
        at exit time.

        Optional inputs:

          - data(None): if data is given, it gets written out to the temp file
            immediately, and the file is closed again."""

        dir_path = Path(tempfile.mkdtemp(prefix=prefix))
        self.tempdirs.append(dir_path)

        handle, filename = tempfile.mkstemp(".py", prefix, dir=str(dir_path))
        os.close(handle)  # On Windows, there can only be one open handle on a file

        file_path = Path(filename)
        self.tempfiles.append(file_path)

        if data:
            file_path.write_text(data, encoding="utf-8")
        return filename

    def ask_yes_no(self, prompt, default=None, interrupt=None):
        if self.quiet:
            return True
        return ask_yes_no(prompt,default,interrupt)

    def show_usage(self):
        """Show a usage message"""
        page.page(IPython.core.usage.interactive_usage)

    def extract_input_lines(self, range_str, raw=False):
        """Return as a string a set of input history slices.

        Parameters
        ----------
        range_str : str
            The set of slices is given as a string, like "~5/6-~4/2 4:8 9",
            since this function is for use by magic functions which get their
            arguments as strings. The number before the / is the session
            number: ~n goes n back from the current session.

            If empty string is given, returns history of current session
            without the last input.

        raw : bool, optional
            By default, the processed input is used.  If this is true, the raw
            input history is used instead.

        Notes
        -----
        Slices can be described with two notations:

        * ``N:M`` -> standard python form, means including items N...(M-1).
        * ``N-M`` -> include items N..M (closed endpoint).
        """
        lines = self.history_manager.get_range_by_str(range_str, raw=raw)
        text = "\n".join(x for _, _, x in lines)

        # Skip the last line, as it's probably the magic that called this
        if not range_str:
            if "\n" not in text:
                text = ""
            else:
                text = text[: text.rfind("\n")]

        return text

    def find_user_code(self, target, raw=True, py_only=False, skip_encoding_cookie=True, search_ns=False):
        """Get a code string from history, file, url, or a string or macro.

        This is mainly used by magic functions.

        Parameters
        ----------
        target : str
            A string specifying code to retrieve. This will be tried respectively
            as: ranges of input history (see %history for syntax), url,
            corresponding .py file, filename, or an expression evaluating to a
            string or Macro in the user namespace.

            If empty string is given, returns complete history of current
            session, without the last line.

        raw : bool
            If true (default), retrieve raw history. Has no effect on the other
            retrieval mechanisms.

        py_only : bool (default False)
            Only try to fetch python code, do not try alternative methods to decode file
            if unicode fails.

        Returns
        -------
        A string of code.
        ValueError is raised if nothing is found, and TypeError if it evaluates
        to an object of another type. In each case, .args[0] is a printable
        message.
        """
        code = self.extract_input_lines(target, raw=raw)  # Grab history
        if code:
            return code
        try:
            if target.startswith(('http://', 'https://')):
                return openpy.read_py_url(target, skip_encoding_cookie=skip_encoding_cookie)
        except UnicodeDecodeError as e:
            if not py_only :
                # Deferred import
                from urllib.request import urlopen
                response = urlopen(target)
                return response.read().decode('latin1')
            raise ValueError(("'%s' seem to be unreadable.") % target) from e

        potential_target = [target]
        try :
            potential_target.insert(0,get_py_filename(target))
        except IOError:
            pass

        for tgt in potential_target :
            if os.path.isfile(tgt):                        # Read file
                try :
                    return openpy.read_py_file(tgt, skip_encoding_cookie=skip_encoding_cookie)
                except UnicodeDecodeError as e:
                    if not py_only :
                        with io_open(tgt,'r', encoding='latin1') as f :
                            return f.read()
                    raise ValueError(("'%s' seem to be unreadable.") % target) from e
            elif os.path.isdir(os.path.expanduser(tgt)):
                raise ValueError("'%s' is a directory, not a regular file." % target)

        if search_ns:
            # Inspect namespace to load object source
            object_info = self.object_inspect(target, detail_level=1)
            if object_info['found'] and object_info['source']:
                return object_info['source']

        try:                                              # User namespace
            codeobj = eval(target, self.user_ns)
        except Exception as e:
            raise ValueError(("'%s' was not found in history, as a file, url, "
                                "nor in the user namespace.") % target) from e

        if isinstance(codeobj, str):
            return codeobj
        elif isinstance(codeobj, Macro):
            return codeobj.value

        raise TypeError("%s is neither a string nor a macro." % target,
                        codeobj)

    def _atexit_once(self):
        """
        At exist operation that need to be called at most once.
        Second call to this function per instance will do nothing.
        """

        if not getattr(self, "_atexit_once_called", False):
            self._atexit_once_called = True
            # Clear all user namespaces to release all references cleanly.
            self.reset(new_session=False)
            # Close the history session (this stores the end time and line count)
            # this must be *before* the tempfile cleanup, in case of temporary
            # history db
            self.history_manager.end_session()
            self.history_manager = None

    #-------------------------------------------------------------------------
    # Things related to IPython exiting
    #-------------------------------------------------------------------------
    def atexit_operations(self):
        """This will be executed at the time of exit.

        Cleanup operations and saving of persistent data that is done
        unconditionally by IPython should be performed here.

        For things that may depend on startup flags or platform specifics (such
        as having readline or not), register a separate atexit function in the
        code that has the appropriate information, rather than trying to
        clutter
        """
        self._atexit_once()

        # Cleanup all tempfiles and folders left around
        for tfile in self.tempfiles:
            try:
                tfile.unlink()
                self.tempfiles.remove(tfile)
            except FileNotFoundError:
                pass
        del self.tempfiles
        for tdir in self.tempdirs:
            try:
                tdir.rmdir()
                self.tempdirs.remove(tdir)
            except FileNotFoundError:
                pass
        del self.tempdirs

        # Restore user's cursor
        if hasattr(self, "editing_mode") and self.editing_mode == "vi":
            sys.stdout.write("\x1b[0 q")
            sys.stdout.flush()

    def cleanup(self):
        self.restore_sys_module_state()


    # Overridden in terminal subclass to change prompts
    def switch_doctest_mode(self, mode):
        pass


class InteractiveShellABC(metaclass=abc.ABCMeta):
    """An abstract base class for InteractiveShell."""

InteractiveShellABC.register(InteractiveShell)
