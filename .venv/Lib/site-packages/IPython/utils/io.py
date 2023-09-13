# encoding: utf-8
"""
IO related utilities.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.



import atexit
import os
import sys
import tempfile
from pathlib import Path
from warnings import warn

from IPython.utils.decorators import undoc
from .capture import CapturedIO, capture_output

class Tee(object):
    """A class to duplicate an output stream to stdout/err.

    This works in a manner very similar to the Unix 'tee' command.

    When the object is closed or deleted, it closes the original file given to
    it for duplication.
    """
    # Inspired by:
    # http://mail.python.org/pipermail/python-list/2007-May/442737.html

    def __init__(self, file_or_name, mode="w", channel='stdout'):
        """Construct a new Tee object.

        Parameters
        ----------
        file_or_name : filename or open filehandle (writable)
            File that will be duplicated
        mode : optional, valid mode for open().
            If a filename was give, open with this mode.
        channel : str, one of ['stdout', 'stderr']
        """
        if channel not in ['stdout', 'stderr']:
            raise ValueError('Invalid channel spec %s' % channel)

        if hasattr(file_or_name, 'write') and hasattr(file_or_name, 'seek'):
            self.file = file_or_name
        else:
            encoding = None if "b" in mode else "utf-8"
            self.file = open(file_or_name, mode, encoding=encoding)
        self.channel = channel
        self.ostream = getattr(sys, channel)
        setattr(sys, channel, self)
        self._closed = False

    def close(self):
        """Close the file and restore the channel."""
        self.flush()
        setattr(sys, self.channel, self.ostream)
        self.file.close()
        self._closed = True

    def write(self, data):
        """Write data to both channels."""
        self.file.write(data)
        self.ostream.write(data)
        self.ostream.flush()

    def flush(self):
        """Flush both channels."""
        self.file.flush()
        self.ostream.flush()

    def __del__(self):
        if not self._closed:
            self.close()


def ask_yes_no(prompt, default=None, interrupt=None):
    """Asks a question and returns a boolean (y/n) answer.

    If default is given (one of 'y','n'), it is used if the user input is
    empty. If interrupt is given (one of 'y','n'), it is used if the user
    presses Ctrl-C. Otherwise the question is repeated until an answer is
    given.

    An EOF is treated as the default answer.  If there is no default, an
    exception is raised to prevent infinite loops.

    Valid answers are: y/yes/n/no (match is not case sensitive)."""

    answers = {'y':True,'n':False,'yes':True,'no':False}
    ans = None
    while ans not in answers.keys():
        try:
            ans = input(prompt+' ').lower()
            if not ans:  # response was an empty string
                ans = default
        except KeyboardInterrupt:
            if interrupt:
                ans = interrupt
            print("\r")
        except EOFError:
            if default in answers.keys():
                ans = default
                print()
            else:
                raise

    return answers[ans]


def temp_pyfile(src, ext='.py'):
    """Make a temporary python file, return filename and filehandle.

    Parameters
    ----------
    src : string or list of strings (no need for ending newlines if list)
        Source code to be written to the file.
    ext : optional, string
        Extension for the generated file.

    Returns
    -------
    (filename, open filehandle)
        It is the caller's responsibility to close the open file and unlink it.
    """
    fname = tempfile.mkstemp(ext)[1]
    with open(Path(fname), "w", encoding="utf-8") as f:
        f.write(src)
        f.flush()
    return fname


@undoc
def raw_print(*args, **kw):
    """DEPRECATED: Raw print to sys.__stdout__, otherwise identical interface to print()."""
    warn("IPython.utils.io.raw_print has been deprecated since IPython 7.0", DeprecationWarning, stacklevel=2)

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stdout__)
    sys.__stdout__.flush()

@undoc
def raw_print_err(*args, **kw):
    """DEPRECATED: Raw print to sys.__stderr__, otherwise identical interface to print()."""
    warn("IPython.utils.io.raw_print_err has been deprecated since IPython 7.0", DeprecationWarning, stacklevel=2)

    print(*args, sep=kw.get('sep', ' '), end=kw.get('end', '\n'),
          file=sys.__stderr__)
    sys.__stderr__.flush()
