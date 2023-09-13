"""Generic testing tools.

Authors
-------
- Fernando Perez <Fernando.Perez@berkeley.edu>
"""


# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

from contextlib import contextmanager
from io import StringIO
from subprocess import Popen, PIPE
from unittest.mock import patch

from traitlets.config.loader import Config
from IPython.utils.process import get_output_error_code
from IPython.utils.text import list_strings
from IPython.utils.io import temp_pyfile, Tee
from IPython.utils import py3compat

from . import decorators as dec
from . import skipdoctest


# The docstring for full_path doctests differently on win32 (different path
# separator) so just skip the doctest there.  The example remains informative.
doctest_deco = skipdoctest.skip_doctest if sys.platform == 'win32' else dec.null_deco

@doctest_deco
def full_path(startPath,files):
    """Make full paths for all the listed files, based on startPath.

    Only the base part of startPath is kept, since this routine is typically
    used with a script's ``__file__`` variable as startPath. The base of startPath
    is then prepended to all the listed files, forming the output list.

    Parameters
    ----------
    startPath : string
      Initial path to use as the base for the results.  This path is split
      using os.path.split() and only its first component is kept.

    files : string or list
      One or more files.

    Examples
    --------

    >>> full_path('/foo/bar.py',['a.txt','b.txt'])
    ['/foo/a.txt', '/foo/b.txt']

    >>> full_path('/foo',['a.txt','b.txt'])
    ['/a.txt', '/b.txt']

    If a single file is given, the output is still a list::

        >>> full_path('/foo','a.txt')
        ['/a.txt']
    """

    files = list_strings(files)
    base = os.path.split(startPath)[0]
    return [ os.path.join(base,f) for f in files ]


def parse_test_output(txt):
    """Parse the output of a test run and return errors, failures.

    Parameters
    ----------
    txt : str
      Text output of a test run, assumed to contain a line of one of the
      following forms::

        'FAILED (errors=1)'
        'FAILED (failures=1)'
        'FAILED (errors=1, failures=1)'

    Returns
    -------
    nerr, nfail
      number of errors and failures.
    """

    err_m = re.search(r'^FAILED \(errors=(\d+)\)', txt, re.MULTILINE)
    if err_m:
        nerr = int(err_m.group(1))
        nfail = 0
        return  nerr, nfail

    fail_m = re.search(r'^FAILED \(failures=(\d+)\)', txt, re.MULTILINE)
    if fail_m:
        nerr = 0
        nfail = int(fail_m.group(1))
        return  nerr, nfail

    both_m = re.search(r'^FAILED \(errors=(\d+), failures=(\d+)\)', txt,
                       re.MULTILINE)
    if both_m:
        nerr = int(both_m.group(1))
        nfail = int(both_m.group(2))
        return  nerr, nfail

    # If the input didn't match any of these forms, assume no error/failures
    return 0, 0


# So nose doesn't think this is a test
parse_test_output.__test__ = False


def default_argv():
    """Return a valid default argv for creating testing instances of ipython"""

    return ['--quick', # so no config file is loaded
            # Other defaults to minimize side effects on stdout
            '--colors=NoColor', '--no-term-title','--no-banner',
            '--autocall=0']


def default_config():
    """Return a config object with good defaults for testing."""
    config = Config()
    config.TerminalInteractiveShell.colors = 'NoColor'
    config.TerminalTerminalInteractiveShell.term_title = False,
    config.TerminalInteractiveShell.autocall = 0
    f = tempfile.NamedTemporaryFile(suffix=u'test_hist.sqlite', delete=False)
    config.HistoryManager.hist_file = Path(f.name)
    f.close()
    config.HistoryManager.db_cache_size = 10000
    return config


def get_ipython_cmd(as_string=False):
    """
    Return appropriate IPython command line name. By default, this will return
    a list that can be used with subprocess.Popen, for example, but passing
    `as_string=True` allows for returning the IPython command as a string.

    Parameters
    ----------
    as_string: bool
        Flag to allow to return the command as a string.
    """
    ipython_cmd = [sys.executable, "-m", "IPython"]

    if as_string:
        ipython_cmd = " ".join(ipython_cmd)

    return ipython_cmd

def ipexec(fname, options=None, commands=()):
    """Utility to call 'ipython filename'.

    Starts IPython with a minimal and safe configuration to make startup as fast
    as possible.

    Note that this starts IPython in a subprocess!

    Parameters
    ----------
    fname : str, Path
      Name of file to be executed (should have .py or .ipy extension).

    options : optional, list
      Extra command-line flags to be passed to IPython.

    commands : optional, list
      Commands to send in on stdin

    Returns
    -------
    ``(stdout, stderr)`` of ipython subprocess.
    """
    __tracebackhide__ = True

    if options is None:
        options = []

    cmdargs = default_argv() + options

    test_dir = os.path.dirname(__file__)

    ipython_cmd = get_ipython_cmd()
    # Absolute path for filename
    full_fname = os.path.join(test_dir, fname)
    full_cmd = ipython_cmd + cmdargs + ['--', full_fname]
    env = os.environ.copy()
    # FIXME: ignore all warnings in ipexec while we have shims
    # should we keep suppressing warnings here, even after removing shims?
    env['PYTHONWARNINGS'] = 'ignore'
    # env.pop('PYTHONWARNINGS', None)  # Avoid extraneous warnings appearing on stderr
    # Prevent coloring under PyCharm ("\x1b[0m" at the end of the stdout)
    env.pop("PYCHARM_HOSTED", None)
    for k, v in env.items():
        # Debug a bizarre failure we've seen on Windows:
        # TypeError: environment can only contain strings
        if not isinstance(v, str):
            print(k, v)
    p = Popen(full_cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, env=env)
    out, err = p.communicate(input=py3compat.encode('\n'.join(commands)) or None)
    out, err = py3compat.decode(out), py3compat.decode(err)
    # `import readline` causes 'ESC[?1034h' to be output sometimes,
    # so strip that out before doing comparisons
    if out:
        out = re.sub(r'\x1b\[[^h]+h', '', out)
    return out, err


def ipexec_validate(fname, expected_out, expected_err='',
                    options=None, commands=()):
    """Utility to call 'ipython filename' and validate output/error.

    This function raises an AssertionError if the validation fails.

    Note that this starts IPython in a subprocess!

    Parameters
    ----------
    fname : str, Path
      Name of the file to be executed (should have .py or .ipy extension).

    expected_out : str
      Expected stdout of the process.

    expected_err : optional, str
      Expected stderr of the process.

    options : optional, list
      Extra command-line flags to be passed to IPython.

    Returns
    -------
    None
    """
    __tracebackhide__ = True

    out, err = ipexec(fname, options, commands)
    #print 'OUT', out  # dbg
    #print 'ERR', err  # dbg
    # If there are any errors, we must check those before stdout, as they may be
    # more informative than simply having an empty stdout.
    if err:
        if expected_err:
            assert "\n".join(err.strip().splitlines()) == "\n".join(
                expected_err.strip().splitlines()
            )
        else:
            raise ValueError('Running file %r produced error: %r' %
                             (fname, err))
    # If no errors or output on stderr was expected, match stdout
    assert "\n".join(out.strip().splitlines()) == "\n".join(
        expected_out.strip().splitlines()
    )


class TempFileMixin(unittest.TestCase):
    """Utility class to create temporary Python/IPython files.

    Meant as a mixin class for test cases."""

    def mktmp(self, src, ext='.py'):
        """Make a valid python temp file."""
        fname = temp_pyfile(src, ext)
        if not hasattr(self, 'tmps'):
            self.tmps=[]
        self.tmps.append(fname)
        self.fname = fname

    def tearDown(self):
        # If the tmpfile wasn't made because of skipped tests, like in
        # win32, there's nothing to cleanup.
        if hasattr(self, 'tmps'):
            for fname in self.tmps:
                # If the tmpfile wasn't made because of skipped tests, like in
                # win32, there's nothing to cleanup.
                try:
                    os.unlink(fname)
                except:
                    # On Windows, even though we close the file, we still can't
                    # delete it.  I have no clue why
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tearDown()


pair_fail_msg = ("Testing {0}\n\n"
                "In:\n"
                "  {1!r}\n"
                "Expected:\n"
                "  {2!r}\n"
                "Got:\n"
                "  {3!r}\n")
def check_pairs(func, pairs):
    """Utility function for the common case of checking a function with a
    sequence of input/output pairs.

    Parameters
    ----------
    func : callable
      The function to be tested. Should accept a single argument.
    pairs : iterable
      A list of (input, expected_output) tuples.

    Returns
    -------
    None. Raises an AssertionError if any output does not match the expected
    value.
    """
    __tracebackhide__ = True

    name = getattr(func, "func_name", getattr(func, "__name__", "<unknown>"))
    for inp, expected in pairs:
        out = func(inp)
        assert out == expected, pair_fail_msg.format(name, inp, expected, out)


MyStringIO = StringIO

_re_type = type(re.compile(r''))

notprinted_msg = """Did not find {0!r} in printed output (on {1}):
-------
{2!s}
-------
"""

class AssertPrints(object):
    """Context manager for testing that code prints certain text.

    Examples
    --------
    >>> with AssertPrints("abc", suppress=False):
    ...     print("abcd")
    ...     print("def")
    ...
    abcd
    def
    """
    def __init__(self, s, channel='stdout', suppress=True):
        self.s = s
        if isinstance(self.s, (str, _re_type)):
            self.s = [self.s]
        self.channel = channel
        self.suppress = suppress

    def __enter__(self):
        self.orig_stream = getattr(sys, self.channel)
        self.buffer = MyStringIO()
        self.tee = Tee(self.buffer, channel=self.channel)
        setattr(sys, self.channel, self.buffer if self.suppress else self.tee)

    def __exit__(self, etype, value, traceback):
        __tracebackhide__ = True

        try:
            if value is not None:
                # If an error was raised, don't check anything else
                return False
            self.tee.flush()
            setattr(sys, self.channel, self.orig_stream)
            printed = self.buffer.getvalue()
            for s in self.s:
                if isinstance(s, _re_type):
                    assert s.search(printed), notprinted_msg.format(s.pattern, self.channel, printed)
                else:
                    assert s in printed, notprinted_msg.format(s, self.channel, printed)
            return False
        finally:
            self.tee.close()

printed_msg = """Found {0!r} in printed output (on {1}):
-------
{2!s}
-------
"""

class AssertNotPrints(AssertPrints):
    """Context manager for checking that certain output *isn't* produced.

    Counterpart of AssertPrints"""
    def __exit__(self, etype, value, traceback):
        __tracebackhide__ = True

        try:
            if value is not None:
                # If an error was raised, don't check anything else
                self.tee.close()
                return False
            self.tee.flush()
            setattr(sys, self.channel, self.orig_stream)
            printed = self.buffer.getvalue()
            for s in self.s:
                if isinstance(s, _re_type):
                    assert not s.search(printed),printed_msg.format(
                        s.pattern, self.channel, printed)
                else:
                    assert s not in printed, printed_msg.format(
                        s, self.channel, printed)
            return False
        finally:
            self.tee.close()

@contextmanager
def mute_warn():
    from IPython.utils import warn
    save_warn = warn.warn
    warn.warn = lambda *a, **kw: None
    try:
        yield
    finally:
        warn.warn = save_warn

@contextmanager
def make_tempfile(name):
    """Create an empty, named, temporary file for the duration of the context."""
    open(name, "w", encoding="utf-8").close()
    try:
        yield
    finally:
        os.unlink(name)

def fake_input(inputs):
    """Temporarily replace the input() function to return the given values

    Use as a context manager:

    with fake_input(['result1', 'result2']):
        ...

    Values are returned in order. If input() is called again after the last value
    was used, EOFError is raised.
    """
    it = iter(inputs)
    def mock_input(prompt=''):
        try:
            return next(it)
        except StopIteration as e:
            raise EOFError('No more inputs given') from e

    return patch('builtins.input', mock_input)

def help_output_test(subcommand=''):
    """test that `ipython [subcommand] -h` works"""
    cmd = get_ipython_cmd() + [subcommand, '-h']
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert "Traceback" not in err
    assert "Options" in out
    assert "--help-all" in out
    return out, err


def help_all_output_test(subcommand=''):
    """test that `ipython [subcommand] --help-all` works"""
    cmd = get_ipython_cmd() + [subcommand, '--help-all']
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert "Traceback" not in err
    assert "Options" in out
    assert "Class" in out
    return out, err

