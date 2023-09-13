"""Posix-specific implementation of process utilities.

This file is only meant to be imported by process.py, not by end-users.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2010-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import errno
import os
import subprocess as sp
import sys

import pexpect

# Our own
from ._process_common import getoutput, arg_split
from IPython.utils.encoding import DEFAULT_ENCODING

#-----------------------------------------------------------------------------
# Function definitions
#-----------------------------------------------------------------------------

class ProcessHandler(object):
    """Execute subprocesses under the control of pexpect.
    """
    # Timeout in seconds to wait on each reading of the subprocess' output.
    # This should not be set too low to avoid cpu overusage from our side,
    # since we read in a loop whose period is controlled by this timeout.
    read_timeout = 0.05

    # Timeout to give a process if we receive SIGINT, between sending the
    # SIGINT to the process and forcefully terminating it.
    terminate_timeout = 0.2

    # File object where stdout and stderr of the subprocess will be written
    logfile = None

    # Shell to call for subprocesses to execute
    _sh = None

    @property
    def sh(self):
        if self._sh is None:
            shell_name = os.environ.get("SHELL", "sh")
            self._sh = pexpect.which(shell_name)
            if self._sh is None:
                raise OSError('"{}" shell not found'.format(shell_name))

        return self._sh

    def __init__(self, logfile=None, read_timeout=None, terminate_timeout=None):
        """Arguments are used for pexpect calls."""
        self.read_timeout = (ProcessHandler.read_timeout if read_timeout is
                             None else read_timeout)
        self.terminate_timeout = (ProcessHandler.terminate_timeout if
                                  terminate_timeout is None else
                                  terminate_timeout)
        self.logfile = sys.stdout if logfile is None else logfile

    def getoutput(self, cmd):
        """Run a command and return its stdout/stderr as a string.

        Parameters
        ----------
        cmd : str
            A command to be executed in the system shell.

        Returns
        -------
        output : str
            A string containing the combination of stdout and stderr from the
        subprocess, in whatever order the subprocess originally wrote to its
        file descriptors (so the order of the information in this string is the
        correct order as would be seen if running the command in a terminal).
        """
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

    def getoutput_pexpect(self, cmd):
        """Run a command and return its stdout/stderr as a string.

        Parameters
        ----------
        cmd : str
            A command to be executed in the system shell.

        Returns
        -------
        output : str
            A string containing the combination of stdout and stderr from the
        subprocess, in whatever order the subprocess originally wrote to its
        file descriptors (so the order of the information in this string is the
        correct order as would be seen if running the command in a terminal).
        """
        try:
            return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
        except KeyboardInterrupt:
            print('^C', file=sys.stderr, end='')

    def system(self, cmd):
        """Execute a command in a subshell.

        Parameters
        ----------
        cmd : str
            A command to be executed in the system shell.

        Returns
        -------
        int : child's exitstatus
        """
        # Get likely encoding for the output.
        enc = DEFAULT_ENCODING
        
        # Patterns to match on the output, for pexpect.  We read input and
        # allow either a short timeout or EOF
        patterns = [pexpect.TIMEOUT, pexpect.EOF]
        # the index of the EOF pattern in the list.
        # even though we know it's 1, this call means we don't have to worry if
        # we change the above list, and forget to change this value:
        EOF_index = patterns.index(pexpect.EOF)
        # The size of the output stored so far in the process output buffer.
        # Since pexpect only appends to this buffer, each time we print we
        # record how far we've printed, so that next time we only print *new*
        # content from the buffer.
        out_size = 0
        try:
            # Since we're not really searching the buffer for text patterns, we
            # can set pexpect's search window to be tiny and it won't matter.
            # We only search for the 'patterns' timeout or EOF, which aren't in
            # the text itself.
            #child = pexpect.spawn(pcmd, searchwindowsize=1)
            if hasattr(pexpect, 'spawnb'):
                child = pexpect.spawnb(self.sh, args=['-c', cmd]) # Pexpect-U
            else:
                child = pexpect.spawn(self.sh, args=['-c', cmd])  # Vanilla Pexpect
            flush = sys.stdout.flush
            while True:
                # res is the index of the pattern that caused the match, so we
                # know whether we've finished (if we matched EOF) or not
                res_idx = child.expect_list(patterns, self.read_timeout)
                print(child.before[out_size:].decode(enc, 'replace'), end='')
                flush()
                if res_idx==EOF_index:
                    break
                # Update the pointer to what we've already printed
                out_size = len(child.before)
        except KeyboardInterrupt:
            # We need to send ^C to the process.  The ascii code for '^C' is 3
            # (the character is known as ETX for 'End of Text', see
            # curses.ascii.ETX).
            child.sendline(chr(3))
            # Read and print any more output the program might produce on its
            # way out.
            try:
                out_size = len(child.before)
                child.expect_list(patterns, self.terminate_timeout)
                print(child.before[out_size:].decode(enc, 'replace'), end='')
                sys.stdout.flush()
            except KeyboardInterrupt:
                # Impatient users tend to type it multiple times
                pass
            finally:
                # Ensure the subprocess really is terminated
                child.terminate(force=True)
        # add isalive check, to ensure exitstatus is set:
        child.isalive()

        # We follow the subprocess pattern, returning either the exit status
        # as a positive number, or the terminating signal as a negative
        # number.
        # on Linux, sh returns 128+n for signals terminating child processes on Linux
        # on BSD (OS X), the signal code is set instead
        if child.exitstatus is None:
            # on WIFSIGNALED, pexpect sets signalstatus, leaving exitstatus=None
            if child.signalstatus is None:
                # this condition may never occur,
                # but let's be certain we always return an integer.
                return 0
            return -child.signalstatus
        if child.exitstatus > 128:
            return -(child.exitstatus - 128)
        return child.exitstatus


# Make system() with a functional interface for outside use.  Note that we use
# getoutput() from the _common utils, which is built on top of popen(). Using
# pexpect to get subprocess output produces difficult to parse output, since
# programs think they are talking to a tty and produce highly formatted output
# (ls is a good example) that makes them hard.
system = ProcessHandler().system

def check_pid(pid):
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            # Don't have permission to signal the process - probably means it exists
            return True
        raise
    else:
        return True
