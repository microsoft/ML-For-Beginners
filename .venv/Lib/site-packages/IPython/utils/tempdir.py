""" This module contains classes - NamedFileInTemporaryDirectory, TemporaryWorkingDirectory.

These classes add extra features such as creating a named file in temporary directory and
creating a context manager for the working directory which is also temporary.
"""

import os as _os
from pathlib import Path
from tempfile import TemporaryDirectory


class NamedFileInTemporaryDirectory(object):
    def __init__(self, filename, mode="w+b", bufsize=-1, add_to_syspath=False, **kwds):
        """
        Open a file named `filename` in a temporary directory.

        This context manager is preferred over `NamedTemporaryFile` in
        stdlib `tempfile` when one needs to reopen the file.

        Arguments `mode` and `bufsize` are passed to `open`.
        Rest of the arguments are passed to `TemporaryDirectory`.

        """
        self._tmpdir = TemporaryDirectory(**kwds)
        path = Path(self._tmpdir.name) / filename
        encoding = None if "b" in mode else "utf-8"
        self.file = open(path, mode, bufsize, encoding=encoding)

    def cleanup(self):
        self.file.close()
        self._tmpdir.cleanup()

    __del__ = cleanup

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.cleanup()


class TemporaryWorkingDirectory(TemporaryDirectory):
    """
    Creates a temporary directory and sets the cwd to that directory.
    Automatically reverts to previous cwd upon cleanup.
    Usage example:

        with TemporaryWorkingDirectory() as tmpdir:
            ...
    """

    def __enter__(self):
        self.old_wd = Path.cwd()
        _os.chdir(self.name)
        return super(TemporaryWorkingDirectory, self).__enter__()

    def __exit__(self, exc, value, tb):
        _os.chdir(self.old_wd)
        return super(TemporaryWorkingDirectory, self).__exit__(exc, value, tb)
