"""Hooks for IPython.

In Python, it is possible to overwrite any method of any object if you really
want to.  But IPython exposes a few 'hooks', methods which are *designed* to
be overwritten by users for customization purposes.  This module defines the
default versions of all such hooks, which get used by IPython if not
overridden by the user.

Hooks are simple functions, but they should be declared with ``self`` as their
first argument, because when activated they are registered into IPython as
instance methods. The self argument will be the IPython running instance
itself, so hooks have full access to the entire IPython object.

If you wish to define a new hook and activate it, you can make an :doc:`extension
</config/extensions/index>` or a :ref:`startup script <startup_files>`. For
example, you could use a startup file like this::

    import os

    def calljed(self,filename, linenum):
        "My editor hook calls the jed editor directly."
        print "Calling my own editor, jed ..."
        if os.system('jed +%d %s' % (linenum,filename)) != 0:
            raise TryNext()

    def load_ipython_extension(ip):
        ip.set_hook('editor', calljed)

"""

#*****************************************************************************
#       Copyright (C) 2005 Fernando Perez. <fperez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#*****************************************************************************

import os
import subprocess
import sys

from .error import TryNext

# List here all the default hooks.  For now it's just the editor functions
# but over time we'll move here all the public API for user-accessible things.

__all__ = [
    "editor",
    "synchronize_with_editor",
    "show_in_pager",
    "pre_prompt_hook",
    "clipboard_get",
]

deprecated = {'pre_run_code_hook': "a callback for the 'pre_execute' or 'pre_run_cell' event",
              'late_startup_hook': "a callback for the 'shell_initialized' event",
              'shutdown_hook': "the atexit module",
             }

def editor(self, filename, linenum=None, wait=True):
    """Open the default editor at the given filename and linenumber.

    This is IPython's default editor hook, you can use it as an example to
    write your own modified one.  To set your own editor function as the
    new editor hook, call ip.set_hook('editor',yourfunc)."""

    # IPython configures a default editor at startup by reading $EDITOR from
    # the environment, and falling back on vi (unix) or notepad (win32).
    editor = self.editor

    # marker for at which line to open the file (for existing objects)
    if linenum is None or editor=='notepad':
        linemark = ''
    else:
        linemark = '+%d' % int(linenum)

    # Enclose in quotes if necessary and legal
    if ' ' in editor and os.path.isfile(editor) and editor[0] != '"':
        editor = '"%s"' % editor

    # Call the actual editor
    proc = subprocess.Popen('%s %s %s' % (editor, linemark, filename),
                            shell=True)
    if wait and proc.wait() != 0:
        raise TryNext()


def synchronize_with_editor(self, filename, linenum, column):
        pass


class CommandChainDispatcher:
    """ Dispatch calls to a chain of commands until some func can handle it

    Usage: instantiate, execute "add" to add commands (with optional
    priority), execute normally via f() calling mechanism.

    """
    def __init__(self,commands=None):
        if commands is None:
            self.chain = []
        else:
            self.chain = commands


    def __call__(self,*args, **kw):
        """ Command chain is called just like normal func.

        This will call all funcs in chain with the same args as were given to
        this function, and return the result of first func that didn't raise
        TryNext"""
        last_exc = TryNext()
        for prio,cmd in self.chain:
            #print "prio",prio,"cmd",cmd #dbg
            try:
                return cmd(*args, **kw)
            except TryNext as exc:
                last_exc = exc
        # if no function will accept it, raise TryNext up to the caller
        raise last_exc

    def __str__(self):
        return str(self.chain)

    def add(self, func, priority=0):
        """ Add a func to the cmd chain with given priority """
        self.chain.append((priority, func))
        self.chain.sort(key=lambda x: x[0])

    def __iter__(self):
        """ Return all objects in chain.

        Handy if the objects are not callable.
        """
        return iter(self.chain)


def show_in_pager(self, data, start, screen_lines):
    """ Run a string through pager """
    # raising TryNext here will use the default paging functionality
    raise TryNext


def pre_prompt_hook(self):
    """ Run before displaying the next prompt

    Use this e.g. to display output from asynchronous operations (in order
    to not mess up text entry)
    """

    return None


def clipboard_get(self):
    """ Get text from the clipboard.
    """
    from ..lib.clipboard import (
        osx_clipboard_get,
        tkinter_clipboard_get,
        win32_clipboard_get,
        wayland_clipboard_get,
    )
    if sys.platform == 'win32':
        chain = [win32_clipboard_get, tkinter_clipboard_get]
    elif sys.platform == 'darwin':
        chain = [osx_clipboard_get, tkinter_clipboard_get]
    else:
        chain = [wayland_clipboard_get, tkinter_clipboard_get]
    dispatcher = CommandChainDispatcher()
    for func in chain:
        dispatcher.add(func)
    text = dispatcher()
    return text
