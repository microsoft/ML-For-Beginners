# encoding: utf-8
"""
Utilities for working with external processes.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import os
import shutil
import sys

if sys.platform == 'win32':
    from ._process_win32 import system, getoutput, arg_split, check_pid
elif sys.platform == 'cli':
    from ._process_cli import system, getoutput, arg_split, check_pid
else:
    from ._process_posix import system, getoutput, arg_split, check_pid

from ._process_common import getoutputerror, get_output_error_code, process_handler


class FindCmdError(Exception):
    pass


def find_cmd(cmd):
    """Find absolute path to executable cmd in a cross platform manner.

    This function tries to determine the full path to a command line program
    using `which` on Unix/Linux/OS X and `win32api` on Windows.  Most of the
    time it will use the version that is first on the users `PATH`.

    Warning, don't use this to find IPython command line programs as there
    is a risk you will find the wrong one.  Instead find those using the
    following code and looking for the application itself::

        import sys
        argv = [sys.executable, '-m', 'IPython']

    Parameters
    ----------
    cmd : str
        The command line program to look for.
    """
    path = shutil.which(cmd)
    if path is None:
        raise FindCmdError('command could not be found: %s' % cmd)
    return path


def abbrev_cwd():
    """ Return abbreviated version of cwd, e.g. d:mydir """
    cwd = os.getcwd().replace('\\','/')
    drivepart = ''
    tail = cwd
    if sys.platform == 'win32':
        if len(cwd) < 4:
            return cwd
        drivepart,tail = os.path.splitdrive(cwd)


    parts = tail.split('/')
    if len(parts) > 2:
        tail = '/'.join(parts[-2:])

    return (drivepart + (
        cwd == '/' and '/' or tail))
