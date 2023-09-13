#!/usr/bin/env python
# encoding: utf-8
"""
The :class:`~traitlets.config.application.Application` object for the command
line :command:`ipython` program.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import logging
import os
import sys
import warnings

from traitlets.config.loader import Config
from traitlets.config.application import boolean_flag, catch_config_error
from IPython.core import release
from IPython.core import usage
from IPython.core.completer import IPCompleter
from IPython.core.crashhandler import CrashHandler
from IPython.core.formatters import PlainTextFormatter
from IPython.core.history import HistoryManager
from IPython.core.application import (
    ProfileDir, BaseIPythonApplication, base_flags, base_aliases
)
from IPython.core.magic import MagicsManager
from IPython.core.magics import (
    ScriptMagics, LoggingMagics
)
from IPython.core.shellapp import (
    InteractiveShellApp, shell_flags, shell_aliases
)
from IPython.extensions.storemagic import StoreMagics
from .interactiveshell import TerminalInteractiveShell
from IPython.paths import get_ipython_dir
from traitlets import (
    Bool, List, default, observe, Type
)

#-----------------------------------------------------------------------------
# Globals, utilities and helpers
#-----------------------------------------------------------------------------

_examples = """
ipython --matplotlib       # enable matplotlib integration
ipython --matplotlib=qt    # enable matplotlib integration with qt4 backend

ipython --log-level=DEBUG  # set logging to DEBUG
ipython --profile=foo      # start with profile foo

ipython profile create foo # create profile foo w/ default config files
ipython help profile       # show the help for the profile subcmd

ipython locate             # print the path to the IPython directory
ipython locate profile foo # print the path to the directory for profile `foo`
"""

#-----------------------------------------------------------------------------
# Crash handler for this application
#-----------------------------------------------------------------------------

class IPAppCrashHandler(CrashHandler):
    """sys.excepthook for IPython itself, leaves a detailed report on disk."""

    def __init__(self, app):
        contact_name = release.author
        contact_email = release.author_email
        bug_tracker = 'https://github.com/ipython/ipython/issues'
        super(IPAppCrashHandler,self).__init__(
            app, contact_name, contact_email, bug_tracker
        )

    def make_report(self,traceback):
        """Return a string containing a crash report."""

        sec_sep = self.section_sep
        # Start with parent report
        report = [super(IPAppCrashHandler, self).make_report(traceback)]
        # Add interactive-specific info we may have
        rpt_add = report.append
        try:
            rpt_add(sec_sep+"History of session input:")
            for line in self.app.shell.user_ns['_ih']:
                rpt_add(line)
            rpt_add('\n*** Last line of input (may not be in above history):\n')
            rpt_add(self.app.shell._last_input_line+'\n')
        except:
            pass

        return ''.join(report)

#-----------------------------------------------------------------------------
# Aliases and Flags
#-----------------------------------------------------------------------------
flags = dict(base_flags)
flags.update(shell_flags)
frontend_flags = {}
addflag = lambda *args: frontend_flags.update(boolean_flag(*args))
addflag('autoedit-syntax', 'TerminalInteractiveShell.autoedit_syntax',
        'Turn on auto editing of files with syntax errors.',
        'Turn off auto editing of files with syntax errors.'
)
addflag('simple-prompt', 'TerminalInteractiveShell.simple_prompt',
        "Force simple minimal prompt using `raw_input`",
        "Use a rich interactive prompt with prompt_toolkit",
)

addflag('banner', 'TerminalIPythonApp.display_banner',
        "Display a banner upon starting IPython.",
        "Don't display a banner upon starting IPython."
)
addflag('confirm-exit', 'TerminalInteractiveShell.confirm_exit',
    """Set to confirm when you try to exit IPython with an EOF (Control-D
    in Unix, Control-Z/Enter in Windows). By typing 'exit' or 'quit',
    you can force a direct exit without any confirmation.""",
    "Don't prompt the user when exiting."
)
addflag('term-title', 'TerminalInteractiveShell.term_title',
    "Enable auto setting the terminal title.",
    "Disable auto setting the terminal title."
)
classic_config = Config()
classic_config.InteractiveShell.cache_size = 0
classic_config.PlainTextFormatter.pprint = False
classic_config.TerminalInteractiveShell.prompts_class='IPython.terminal.prompts.ClassicPrompts'
classic_config.InteractiveShell.separate_in = ''
classic_config.InteractiveShell.separate_out = ''
classic_config.InteractiveShell.separate_out2 = ''
classic_config.InteractiveShell.colors = 'NoColor'
classic_config.InteractiveShell.xmode = 'Plain'

frontend_flags['classic']=(
    classic_config,
    "Gives IPython a similar feel to the classic Python prompt."
)
# # log doesn't make so much sense this way anymore
# paa('--log','-l',
#     action='store_true', dest='InteractiveShell.logstart',
#     help="Start logging to the default log file (./ipython_log.py).")
#
# # quick is harder to implement
frontend_flags['quick']=(
    {'TerminalIPythonApp' : {'quick' : True}},
    "Enable quick startup with no config files."
)

frontend_flags['i'] = (
    {'TerminalIPythonApp' : {'force_interact' : True}},
    """If running code from the command line, become interactive afterwards.
    It is often useful to follow this with `--` to treat remaining flags as
    script arguments.
    """
)
flags.update(frontend_flags)

aliases = dict(base_aliases)
aliases.update(shell_aliases)  # type: ignore[arg-type]

#-----------------------------------------------------------------------------
# Main classes and functions
#-----------------------------------------------------------------------------


class LocateIPythonApp(BaseIPythonApplication):
    description = """print the path to the IPython dir"""
    subcommands = dict(
        profile=('IPython.core.profileapp.ProfileLocate',
            "print the path to an IPython profile directory",
        ),
    )
    def start(self):
        if self.subapp is not None:
            return self.subapp.start()
        else:
            print(self.ipython_dir)


class TerminalIPythonApp(BaseIPythonApplication, InteractiveShellApp):
    name = u'ipython'
    description = usage.cl_usage
    crash_handler_class = IPAppCrashHandler  # typing: ignore[assignment]
    examples = _examples

    flags = flags
    aliases = aliases
    classes = List()

    interactive_shell_class = Type(
        klass=object,   # use default_value otherwise which only allow subclasses.
        default_value=TerminalInteractiveShell,
        help="Class to use to instantiate the TerminalInteractiveShell object. Useful for custom Frontends"
    ).tag(config=True)

    @default('classes')
    def _classes_default(self):
        """This has to be in a method, for TerminalIPythonApp to be available."""
        return [
            InteractiveShellApp, # ShellApp comes before TerminalApp, because
            self.__class__,      # it will also affect subclasses (e.g. QtConsole)
            TerminalInteractiveShell,
            HistoryManager,
            MagicsManager,
            ProfileDir,
            PlainTextFormatter,
            IPCompleter,
            ScriptMagics,
            LoggingMagics,
            StoreMagics,
        ]

    subcommands = dict(
        profile = ("IPython.core.profileapp.ProfileApp",
            "Create and manage IPython profiles."
        ),
        kernel = ("ipykernel.kernelapp.IPKernelApp",
            "Start a kernel without an attached frontend."
        ),
        locate=('IPython.terminal.ipapp.LocateIPythonApp',
            LocateIPythonApp.description
        ),
        history=('IPython.core.historyapp.HistoryApp',
            "Manage the IPython history database."
        ),
    )


    # *do* autocreate requested profile, but don't create the config file.
    auto_create=Bool(True)
    # configurables
    quick = Bool(False,
        help="""Start IPython quickly by skipping the loading of config files."""
    ).tag(config=True)
    @observe('quick')
    def _quick_changed(self, change):
        if change['new']:
            self.load_config_file = lambda *a, **kw: None

    display_banner = Bool(True,
        help="Whether to display a banner upon starting IPython."
    ).tag(config=True)

    # if there is code of files to run from the cmd line, don't interact
    # unless the --i flag (App.force_interact) is true.
    force_interact = Bool(False,
        help="""If a command or file is given via the command-line,
        e.g. 'ipython foo.py', start an interactive shell after executing the
        file or command."""
    ).tag(config=True)
    @observe('force_interact')
    def _force_interact_changed(self, change):
        if change['new']:
            self.interact = True

    @observe('file_to_run', 'code_to_run', 'module_to_run')
    def _file_to_run_changed(self, change):
        new = change['new']
        if new:
            self.something_to_run = True
        if new and not self.force_interact:
                self.interact = False

    # internal, not-configurable
    something_to_run=Bool(False)

    @catch_config_error
    def initialize(self, argv=None):
        """Do actions after construct, but before starting the app."""
        super(TerminalIPythonApp, self).initialize(argv)
        if self.subapp is not None:
            # don't bother initializing further, starting subapp
            return
        # print self.extra_args
        if self.extra_args and not self.something_to_run:
            self.file_to_run = self.extra_args[0]
        self.init_path()
        # create the shell
        self.init_shell()
        # and draw the banner
        self.init_banner()
        # Now a variety of things that happen after the banner is printed.
        self.init_gui_pylab()
        self.init_extensions()
        self.init_code()

    def init_shell(self):
        """initialize the InteractiveShell instance"""
        # Create an InteractiveShell instance.
        # shell.display_banner should always be False for the terminal
        # based app, because we call shell.show_banner() by hand below
        # so the banner shows *before* all extension loading stuff.
        self.shell = self.interactive_shell_class.instance(parent=self,
                        profile_dir=self.profile_dir,
                        ipython_dir=self.ipython_dir, user_ns=self.user_ns)
        self.shell.configurables.append(self)

    def init_banner(self):
        """optionally display the banner"""
        if self.display_banner and self.interact:
            self.shell.show_banner()
        # Make sure there is a space below the banner.
        if self.log_level <= logging.INFO: print()

    def _pylab_changed(self, name, old, new):
        """Replace --pylab='inline' with --pylab='auto'"""
        if new == 'inline':
            warnings.warn("'inline' not available as pylab backend, "
                      "using 'auto' instead.")
            self.pylab = 'auto'

    def start(self):
        if self.subapp is not None:
            return self.subapp.start()
        # perform any prexec steps:
        if self.interact:
            self.log.debug("Starting IPython's mainloop...")
            self.shell.mainloop()
        else:
            self.log.debug("IPython not interactive...")
            self.shell.restore_term_title()
            if not self.shell.last_execution_succeeded:
                sys.exit(1)

def load_default_config(ipython_dir=None):
    """Load the default config file from the default ipython_dir.

    This is useful for embedded shells.
    """
    if ipython_dir is None:
        ipython_dir = get_ipython_dir()

    profile_dir = os.path.join(ipython_dir, 'profile_default')
    app = TerminalIPythonApp()
    app.config_file_paths.append(profile_dir)
    app.load_config_file()
    return app.config

launch_new_instance = TerminalIPythonApp.launch_instance


if __name__ == '__main__':
    launch_new_instance()
