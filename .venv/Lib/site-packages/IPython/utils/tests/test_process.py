"""
Tests for platutils.py
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import sys
import signal
import os
import time
from _thread import interrupt_main  # Py 3
import threading

import pytest

from IPython.utils.process import (find_cmd, FindCmdError, arg_split,
                                   system, getoutput, getoutputerror,
                                   get_output_error_code)
from IPython.utils.capture import capture_output
from IPython.testing import decorators as dec
from IPython.testing import tools as tt

python = os.path.basename(sys.executable)

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------


@dec.skip_win32
def test_find_cmd_ls():
    """Make sure we can find the full path to ls."""
    path = find_cmd("ls")
    assert path.endswith("ls")

    
@dec.skip_if_not_win32
def test_find_cmd_pythonw():
    """Try to find pythonw on Windows."""
    path = find_cmd('pythonw')
    assert path.lower().endswith('pythonw.exe'), path


def test_find_cmd_fail():
    """Make sure that FindCmdError is raised if we can't find the cmd."""
    pytest.raises(FindCmdError, find_cmd, "asdfasdf")


@dec.skip_win32
@pytest.mark.parametrize(
    "argstr, argv",
    [
        ("hi", ["hi"]),
        ("hello there", ["hello", "there"]),
        # \u01ce == \N{LATIN SMALL LETTER A WITH CARON}
        # Do not use \N because the tests crash with syntax error in
        # some cases, for example windows python2.6.
        ("h\u01cello", ["h\u01cello"]),
        ('something "with quotes"', ["something", '"with quotes"']),
    ],
)
def test_arg_split(argstr, argv):
    """Ensure that argument lines are correctly split like in a shell."""
    assert arg_split(argstr) == argv


@dec.skip_if_not_win32
@pytest.mark.parametrize(
    "argstr,argv",
    [
        ("hi", ["hi"]),
        ("hello there", ["hello", "there"]),
        ("h\u01cello", ["h\u01cello"]),
        ('something "with quotes"', ["something", "with quotes"]),
    ],
)
def test_arg_split_win32(argstr, argv):
    """Ensure that argument lines are correctly split like in a shell."""
    assert arg_split(argstr) == argv


class SubProcessTestCase(tt.TempFileMixin):
    def setUp(self):
        """Make a valid python temp file."""
        lines = [ "import sys",
                 "print('on stdout', end='', file=sys.stdout)",
                 "print('on stderr', end='', file=sys.stderr)",
                 "sys.stdout.flush()",
                 "sys.stderr.flush()"]
        self.mktmp('\n'.join(lines))

    def test_system(self):
        status = system(f'{python} "{self.fname}"')
        self.assertEqual(status, 0)

    def test_system_quotes(self):
        status = system('%s -c "import sys"' % python)
        self.assertEqual(status, 0)

    def assert_interrupts(self, command):
        """
        Interrupt a subprocess after a second.
        """
        if threading.main_thread() != threading.current_thread():
            raise pytest.skip("Can't run this test if not in main thread.")

        # Some tests can overwrite SIGINT handler (by using pdb for example),
        # which then breaks this test, so just make sure it's operating
        # normally.
        signal.signal(signal.SIGINT, signal.default_int_handler)

        def interrupt():
            # Wait for subprocess to start:
            time.sleep(0.5)
            interrupt_main()

        threading.Thread(target=interrupt).start()
        start = time.time()
        try:
            result = command()
        except KeyboardInterrupt:
            # Success!
            pass
        end = time.time()
        self.assertTrue(
            end - start < 2, "Process didn't die quickly: %s" % (end - start)
        )
        return result

    def test_system_interrupt(self):
        """
        When interrupted in the way ipykernel interrupts IPython, the
        subprocess is interrupted.
        """
        def command():
            return system('%s -c "import time; time.sleep(5)"' % python)

        status = self.assert_interrupts(command)
        self.assertNotEqual(
            status, 0, f"The process wasn't interrupted. Status: {status}"
        )

    def test_getoutput(self):
        out = getoutput(f'{python} "{self.fname}"')
        # we can't rely on the order the line buffered streams are flushed
        try:
            self.assertEqual(out, 'on stderron stdout')
        except AssertionError:
            self.assertEqual(out, 'on stdouton stderr')

    def test_getoutput_quoted(self):
        out = getoutput('%s -c "print (1)"' % python)
        self.assertEqual(out.strip(), '1')

    #Invalid quoting on windows
    @dec.skip_win32
    def test_getoutput_quoted2(self):
        out = getoutput("%s -c 'print (1)'" % python)
        self.assertEqual(out.strip(), '1')
        out = getoutput("%s -c 'print (\"1\")'" % python)
        self.assertEqual(out.strip(), '1')

    def test_getoutput_error(self):
        out, err = getoutputerror(f'{python} "{self.fname}"')
        self.assertEqual(out, 'on stdout')
        self.assertEqual(err, 'on stderr')

    def test_get_output_error_code(self):
        quiet_exit = '%s -c "import sys; sys.exit(1)"' % python
        out, err, code = get_output_error_code(quiet_exit)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        self.assertEqual(code, 1)
        out, err, code = get_output_error_code(f'{python} "{self.fname}"')
        self.assertEqual(out, 'on stdout')
        self.assertEqual(err, 'on stderr')
        self.assertEqual(code, 0)

        
