# encoding: utf-8
"""
Utilities for working with terminals.

Authors:

* Brian E. Granger
* Fernando Perez
* Alexander Belchenko (e-mail: bialix AT ukr.net)
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size

# This variable is part of the expected API of the module:
ignore_termtitle = True



if os.name == 'posix':
    def _term_clear():
        os.system('clear')
elif sys.platform == 'win32':
    def _term_clear():
        os.system('cls')
else:
    def _term_clear():
        pass



def toggle_set_term_title(val):
    """Control whether set_term_title is active or not.

    set_term_title() allows writing to the console titlebar.  In embedded
    widgets this can cause problems, so this call can be used to toggle it on
    or off as needed.

    The default state of the module is for the function to be disabled.

    Parameters
    ----------
    val : bool
        If True, set_term_title() actually writes to the terminal (using the
        appropriate platform-specific module).  If False, it is a no-op.
    """
    global ignore_termtitle
    ignore_termtitle = not(val)


def _set_term_title(*args,**kw):
    """Dummy no-op."""
    pass


def _restore_term_title():
    pass


_xterm_term_title_saved = False


def _set_term_title_xterm(title):
    """ Change virtual terminal title in xterm-workalikes """
    global _xterm_term_title_saved
    # Only save the title the first time we set, otherwise restore will only
    # go back one title (probably undoing a %cd title change).
    if not _xterm_term_title_saved:
        # save the current title to the xterm "stack"
        sys.stdout.write("\033[22;0t")
        _xterm_term_title_saved = True
    sys.stdout.write('\033]0;%s\007' % title)


def _restore_term_title_xterm():
    # Make sure the restore has at least one accompanying set.
    global _xterm_term_title_saved
    assert _xterm_term_title_saved
    sys.stdout.write('\033[23;0t') 
    _xterm_term_title_saved = False


if os.name == 'posix':
    TERM = os.environ.get('TERM','')
    if TERM.startswith('xterm'):
        _set_term_title = _set_term_title_xterm
        _restore_term_title = _restore_term_title_xterm
elif sys.platform == 'win32':
    import ctypes

    SetConsoleTitleW = ctypes.windll.kernel32.SetConsoleTitleW
    SetConsoleTitleW.argtypes = [ctypes.c_wchar_p]

    def _set_term_title(title):
        """Set terminal title using ctypes to access the Win32 APIs."""
        SetConsoleTitleW(title)


def set_term_title(title):
    """Set terminal title using the necessary platform-dependent calls."""
    if ignore_termtitle:
        return
    _set_term_title(title)


def restore_term_title():
    """Restore, if possible, terminal title to the original state"""
    if ignore_termtitle:
        return
    _restore_term_title()


def freeze_term_title():
    warnings.warn("This function is deprecated, use toggle_set_term_title()")
    global ignore_termtitle
    ignore_termtitle = True


def get_terminal_size(defaultx=80, defaulty=25):
    return _get_terminal_size((defaultx, defaulty))
