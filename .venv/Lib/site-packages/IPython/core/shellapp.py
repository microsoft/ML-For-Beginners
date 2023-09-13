# encoding: utf-8
"""
A mixin for :class:`~IPython.core.application.Application` classes that
launch InteractiveShell instances, load extensions, etc.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import glob
from itertools import chain
import os
import sys

from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
    Unicode, Instance, List, Bool, CaselessStrEnum, observe,
    DottedObjectName,
)
from IPython.terminal import pt_inputhooks

#-----------------------------------------------------------------------------
# Aliases and Flags
#-----------------------------------------------------------------------------

gui_keys = tuple(sorted(pt_inputhooks.backends) + sorted(pt_inputhooks.aliases))

backend_keys = sorted(pylabtools.backends.keys())
backend_keys.insert(0, 'auto')

shell_flags = {}

addflag = lambda *args: shell_flags.update(boolean_flag(*args))
addflag('autoindent', 'InteractiveShell.autoindent',
        'Turn on autoindenting.', 'Turn off autoindenting.'
)
addflag('automagic', 'InteractiveShell.automagic',
        """Turn on the auto calling of magic commands. Type %%magic at the
        IPython  prompt  for  more information.""",
        'Turn off the auto calling of magic commands.'
)
addflag('pdb', 'InteractiveShell.pdb',
    "Enable auto calling the pdb debugger after every exception.",
    "Disable auto calling the pdb debugger after every exception."
)
addflag('pprint', 'PlainTextFormatter.pprint',
    "Enable auto pretty printing of results.",
    "Disable auto pretty printing of results."
)
addflag('color-info', 'InteractiveShell.color_info',
    """IPython can display information about objects via a set of functions,
    and optionally can use colors for this, syntax highlighting
    source code and various other elements. This is on by default, but can cause
    problems with some pagers. If you see such problems, you can disable the
    colours.""",
    "Disable using colors for info related things."
)
addflag('ignore-cwd', 'InteractiveShellApp.ignore_cwd',
        "Exclude the current working directory from sys.path",
        "Include the current working directory in sys.path",
)
nosep_config = Config()
nosep_config.InteractiveShell.separate_in = ''
nosep_config.InteractiveShell.separate_out = ''
nosep_config.InteractiveShell.separate_out2 = ''

shell_flags['nosep']=(nosep_config, "Eliminate all spacing between prompts.")
shell_flags['pylab'] = (
    {'InteractiveShellApp' : {'pylab' : 'auto'}},
    """Pre-load matplotlib and numpy for interactive use with
    the default matplotlib backend."""
)
shell_flags['matplotlib'] = (
    {'InteractiveShellApp' : {'matplotlib' : 'auto'}},
    """Configure matplotlib for interactive use with
    the default matplotlib backend."""
)

# it's possible we don't want short aliases for *all* of these:
shell_aliases = dict(
    autocall='InteractiveShell.autocall',
    colors='InteractiveShell.colors',
    logfile='InteractiveShell.logfile',
    logappend='InteractiveShell.logappend',
    c='InteractiveShellApp.code_to_run',
    m='InteractiveShellApp.module_to_run',
    ext="InteractiveShellApp.extra_extensions",
    gui='InteractiveShellApp.gui',
    pylab='InteractiveShellApp.pylab',
    matplotlib='InteractiveShellApp.matplotlib',
)
shell_aliases['cache-size'] = 'InteractiveShell.cache_size'

#-----------------------------------------------------------------------------
# Main classes and functions
#-----------------------------------------------------------------------------

class InteractiveShellApp(Configurable):
    """A Mixin for applications that start InteractiveShell instances.

    Provides configurables for loading extensions and executing files
    as part of configuring a Shell environment.

    The following methods should be called by the :meth:`initialize` method
    of the subclass:

      - :meth:`init_path`
      - :meth:`init_shell` (to be implemented by the subclass)
      - :meth:`init_gui_pylab`
      - :meth:`init_extensions`
      - :meth:`init_code`
    """
    extensions = List(Unicode(),
        help="A list of dotted module names of IPython extensions to load."
    ).tag(config=True)

    extra_extensions = List(
        DottedObjectName(),
        help="""
        Dotted module name(s) of one or more IPython extensions to load.

        For specifying extra extensions to load on the command-line.

        .. versionadded:: 7.10
        """,
    ).tag(config=True)

    reraise_ipython_extension_failures = Bool(False,
        help="Reraise exceptions encountered loading IPython extensions?",
    ).tag(config=True)

    # Extensions that are always loaded (not configurable)
    default_extensions = List(Unicode(), [u'storemagic']).tag(config=False)

    hide_initial_ns = Bool(True,
        help="""Should variables loaded at startup (by startup files, exec_lines, etc.)
        be hidden from tools like %who?"""
    ).tag(config=True)

    exec_files = List(Unicode(),
        help="""List of files to run at IPython startup."""
    ).tag(config=True)
    exec_PYTHONSTARTUP = Bool(True,
        help="""Run the file referenced by the PYTHONSTARTUP environment
        variable at IPython startup."""
    ).tag(config=True)
    file_to_run = Unicode('',
        help="""A file to be run""").tag(config=True)

    exec_lines = List(Unicode(),
        help="""lines of code to run at IPython startup."""
    ).tag(config=True)
    code_to_run = Unicode('',
        help="Execute the given command string."
    ).tag(config=True)
    module_to_run = Unicode('',
        help="Run the module as a script."
    ).tag(config=True)
    gui = CaselessStrEnum(gui_keys, allow_none=True,
        help="Enable GUI event loop integration with any of {0}.".format(gui_keys)
    ).tag(config=True)
    matplotlib = CaselessStrEnum(backend_keys, allow_none=True,
        help="""Configure matplotlib for interactive use with
        the default matplotlib backend."""
    ).tag(config=True)
    pylab = CaselessStrEnum(backend_keys, allow_none=True,
        help="""Pre-load matplotlib and numpy for interactive use,
        selecting a particular matplotlib backend and loop integration.
        """
    ).tag(config=True)
    pylab_import_all = Bool(True,
        help="""If true, IPython will populate the user namespace with numpy, pylab, etc.
        and an ``import *`` is done from numpy and pylab, when using pylab mode.

        When False, pylab mode should not import any names into the user namespace.
        """
    ).tag(config=True)
    ignore_cwd = Bool(
        False,
        help="""If True, IPython will not add the current working directory to sys.path.
        When False, the current working directory is added to sys.path, allowing imports
        of modules defined in the current directory."""
    ).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC',
                     allow_none=True)
    # whether interact-loop should start
    interact = Bool(True)

    user_ns = Instance(dict, args=None, allow_none=True)
    @observe('user_ns')
    def _user_ns_changed(self, change):
        if self.shell is not None:
            self.shell.user_ns = change['new']
            self.shell.init_user_ns()

    def init_path(self):
        """Add current working directory, '', to sys.path

        Unlike Python's default, we insert before the first `site-packages`
        or `dist-packages` directory,
        so that it is after the standard library.

        .. versionchanged:: 7.2
            Try to insert after the standard library, instead of first.
        .. versionchanged:: 8.0
            Allow optionally not including the current directory in sys.path
        """
        if '' in sys.path or self.ignore_cwd:
            return
        for idx, path in enumerate(sys.path):
            parent, last_part = os.path.split(path)
            if last_part in {'site-packages', 'dist-packages'}:
                break
        else:
            # no site-packages or dist-packages found (?!)
            # back to original behavior of inserting at the front
            idx = 0
        sys.path.insert(idx, '')

    def init_shell(self):
        raise NotImplementedError("Override in subclasses")

    def init_gui_pylab(self):
        """Enable GUI event loop integration, taking pylab into account."""
        enable = False
        shell = self.shell
        if self.pylab:
            enable = lambda key: shell.enable_pylab(key, import_all=self.pylab_import_all)
            key = self.pylab
        elif self.matplotlib:
            enable = shell.enable_matplotlib
            key = self.matplotlib
        elif self.gui:
            enable = shell.enable_gui
            key = self.gui

        if not enable:
            return

        try:
            r = enable(key)
        except ImportError:
            self.log.warning("Eventloop or matplotlib integration failed. Is matplotlib installed?")
            self.shell.showtraceback()
            return
        except Exception:
            self.log.warning("GUI event loop or pylab initialization failed")
            self.shell.showtraceback()
            return

        if isinstance(r, tuple):
            gui, backend = r[:2]
            self.log.info("Enabling GUI event loop integration, "
                      "eventloop=%s, matplotlib=%s", gui, backend)
            if key == "auto":
                print("Using matplotlib backend: %s" % backend)
        else:
            gui = r
            self.log.info("Enabling GUI event loop integration, "
                      "eventloop=%s", gui)

    def init_extensions(self):
        """Load all IPython extensions in IPythonApp.extensions.

        This uses the :meth:`ExtensionManager.load_extensions` to load all
        the extensions listed in ``self.extensions``.
        """
        try:
            self.log.debug("Loading IPython extensions...")
            extensions = (
                self.default_extensions + self.extensions + self.extra_extensions
            )
            for ext in extensions:
                try:
                    self.log.info("Loading IPython extension: %s", ext)
                    self.shell.extension_manager.load_extension(ext)
                except:
                    if self.reraise_ipython_extension_failures:
                        raise
                    msg = ("Error in loading extension: {ext}\n"
                           "Check your config files in {location}".format(
                               ext=ext,
                               location=self.profile_dir.location
                           ))
                    self.log.warning(msg, exc_info=True)
        except:
            if self.reraise_ipython_extension_failures:
                raise
            self.log.warning("Unknown error in loading extensions:", exc_info=True)

    def init_code(self):
        """run the pre-flight code, specified via exec_lines"""
        self._run_startup_files()
        self._run_exec_lines()
        self._run_exec_files()

        # Hide variables defined here from %who etc.
        if self.hide_initial_ns:
            self.shell.user_ns_hidden.update(self.shell.user_ns)

        # command-line execution (ipython -i script.py, ipython -m module)
        # should *not* be excluded from %whos
        self._run_cmd_line_code()
        self._run_module()

        # flush output, so itwon't be attached to the first cell
        sys.stdout.flush()
        sys.stderr.flush()
        self.shell._sys_modules_keys = set(sys.modules.keys())

    def _run_exec_lines(self):
        """Run lines of code in IPythonApp.exec_lines in the user's namespace."""
        if not self.exec_lines:
            return
        try:
            self.log.debug("Running code from IPythonApp.exec_lines...")
            for line in self.exec_lines:
                try:
                    self.log.info("Running code in user namespace: %s" %
                                  line)
                    self.shell.run_cell(line, store_history=False)
                except:
                    self.log.warning("Error in executing line in user "
                                  "namespace: %s" % line)
                    self.shell.showtraceback()
        except:
            self.log.warning("Unknown error in handling IPythonApp.exec_lines:")
            self.shell.showtraceback()

    def _exec_file(self, fname, shell_futures=False):
        try:
            full_filename = filefind(fname, [u'.', self.ipython_dir])
        except IOError:
            self.log.warning("File not found: %r"%fname)
            return
        # Make sure that the running script gets a proper sys.argv as if it
        # were run from a system shell.
        save_argv = sys.argv
        sys.argv = [full_filename] + self.extra_args[1:]
        try:
            if os.path.isfile(full_filename):
                self.log.info("Running file in user namespace: %s" %
                              full_filename)
                # Ensure that __file__ is always defined to match Python
                # behavior.
                with preserve_keys(self.shell.user_ns, '__file__'):
                    self.shell.user_ns['__file__'] = fname
                    if full_filename.endswith('.ipy') or full_filename.endswith('.ipynb'):
                        self.shell.safe_execfile_ipy(full_filename,
                                                     shell_futures=shell_futures)
                    else:
                        # default to python, even without extension
                        self.shell.safe_execfile(full_filename,
                                                 self.shell.user_ns,
                                                 shell_futures=shell_futures,
                                                 raise_exceptions=True)
        finally:
            sys.argv = save_argv

    def _run_startup_files(self):
        """Run files from profile startup directory"""
        startup_dirs = [self.profile_dir.startup_dir] + [
            os.path.join(p, 'startup') for p in chain(ENV_CONFIG_DIRS, SYSTEM_CONFIG_DIRS)
        ]
        startup_files = []

        if self.exec_PYTHONSTARTUP and os.environ.get('PYTHONSTARTUP', False) and \
                not (self.file_to_run or self.code_to_run or self.module_to_run):
            python_startup = os.environ['PYTHONSTARTUP']
            self.log.debug("Running PYTHONSTARTUP file %s...", python_startup)
            try:
                self._exec_file(python_startup)
            except:
                self.log.warning("Unknown error in handling PYTHONSTARTUP file %s:", python_startup)
                self.shell.showtraceback()
        for startup_dir in startup_dirs[::-1]:
            startup_files += glob.glob(os.path.join(startup_dir, '*.py'))
            startup_files += glob.glob(os.path.join(startup_dir, '*.ipy'))
        if not startup_files:
            return

        self.log.debug("Running startup files from %s...", startup_dir)
        try:
            for fname in sorted(startup_files):
                self._exec_file(fname)
        except:
            self.log.warning("Unknown error in handling startup files:")
            self.shell.showtraceback()

    def _run_exec_files(self):
        """Run files from IPythonApp.exec_files"""
        if not self.exec_files:
            return

        self.log.debug("Running files in IPythonApp.exec_files...")
        try:
            for fname in self.exec_files:
                self._exec_file(fname)
        except:
            self.log.warning("Unknown error in handling IPythonApp.exec_files:")
            self.shell.showtraceback()

    def _run_cmd_line_code(self):
        """Run code or file specified at the command-line"""
        if self.code_to_run:
            line = self.code_to_run
            try:
                self.log.info("Running code given at command line (c=): %s" %
                              line)
                self.shell.run_cell(line, store_history=False)
            except:
                self.log.warning("Error in executing line in user namespace: %s" %
                              line)
                self.shell.showtraceback()
                if not self.interact:
                    self.exit(1)

        # Like Python itself, ignore the second if the first of these is present
        elif self.file_to_run:
            fname = self.file_to_run
            if os.path.isdir(fname):
                fname = os.path.join(fname, "__main__.py")
            if not os.path.exists(fname):
                self.log.warning("File '%s' doesn't exist", fname)
                if not self.interact:
                    self.exit(2)
            try:
                self._exec_file(fname, shell_futures=True)
            except:
                self.shell.showtraceback(tb_offset=4)
                if not self.interact:
                    self.exit(1)

    def _run_module(self):
        """Run module specified at the command-line."""
        if self.module_to_run:
            # Make sure that the module gets a proper sys.argv as if it were
            # run using `python -m`.
            save_argv = sys.argv
            sys.argv = [sys.executable] + self.extra_args
            try:
                self.shell.safe_run_module(self.module_to_run,
                                           self.shell.user_ns)
            finally:
                sys.argv = save_argv
