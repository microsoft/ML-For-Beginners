#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""
Unit test runner, providing new features on top of unittest module:
- colourized output
- parallel run (UNIX only)
- print failures/tracebacks on CTRL+C
- re-run failed tests only (make test-failed)

Invocation examples:
- make test
- make test-failed

Parallel:
- make test-parallel
- make test-process ARGS=--parallel
"""

from __future__ import print_function

import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest


try:
    import ctypes
except ImportError:
    ctypes = None

try:
    import concurrencytest  # pip install concurrencytest
except ImportError:
    concurrencytest = None

import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath


VERBOSITY = 2
FAILED_TESTS_FNAME = '.failed-tests.txt'
NWORKERS = psutil.cpu_count() or 1
USE_COLORS = not CI_TESTING and term_supports_colors()

HERE = os.path.abspath(os.path.dirname(__file__))
loadTestsFromTestCase = unittest.defaultTestLoader.loadTestsFromTestCase


def cprint(msg, color, bold=False, file=None):
    if file is None:
        file = sys.stderr if color == 'red' else sys.stdout
    if USE_COLORS:
        print_color(msg, color, bold=bold, file=file)
    else:
        print(msg, file=file)


class TestLoader:

    testdir = HERE
    skip_files = ['test_memleaks.py']
    if "WHEELHOUSE_UPLOADER_USERNAME" in os.environ:
        skip_files.extend(['test_osx.py', 'test_linux.py', 'test_posix.py'])

    def _get_testmods(self):
        return [os.path.join(self.testdir, x)
                for x in os.listdir(self.testdir)
                if x.startswith('test_') and x.endswith('.py') and
                x not in self.skip_files]

    def _iter_testmod_classes(self):
        """Iterate over all test files in this directory and return
        all TestCase classes in them.
        """
        for path in self._get_testmods():
            mod = import_module_by_path(path)
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and \
                        issubclass(obj, unittest.TestCase):
                    yield obj

    def all(self):
        suite = unittest.TestSuite()
        for obj in self._iter_testmod_classes():
            test = loadTestsFromTestCase(obj)
            suite.addTest(test)
        return suite

    def last_failed(self):
        # ...from previously failed test run
        suite = unittest.TestSuite()
        if not os.path.isfile(FAILED_TESTS_FNAME):
            return suite
        with open(FAILED_TESTS_FNAME, 'rt') as f:
            names = f.read().split()
        for n in names:
            test = unittest.defaultTestLoader.loadTestsFromName(n)
            suite.addTest(test)
        return suite

    def from_name(self, name):
        if name.endswith('.py'):
            name = os.path.splitext(os.path.basename(name))[0]
        return unittest.defaultTestLoader.loadTestsFromName(name)


class ColouredResult(unittest.TextTestResult):

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        cprint("OK", "green")

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        cprint("ERROR", "red", bold=True)

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        cprint("FAIL", "red")

    def addSkip(self, test, reason):
        unittest.TestResult.addSkip(self, test, reason)
        cprint("skipped: %s" % reason.strip(), "brown")

    def printErrorList(self, flavour, errors):
        flavour = hilite(flavour, "red", bold=flavour == 'ERROR')
        super().printErrorList(flavour, errors)


class ColouredTextRunner(unittest.TextTestRunner):
    """
    A coloured text runner which also prints failed tests on KeyboardInterrupt
    and save failed tests in a file so that they can be re-run.
    """
    resultclass = ColouredResult if USE_COLORS else unittest.TextTestResult

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_tnames = set()

    def _makeResult(self):
        # Store result instance so that it can be accessed on
        # KeyboardInterrupt.
        self.result = super()._makeResult()
        return self.result

    def _write_last_failed(self):
        if self.failed_tnames:
            with open(FAILED_TESTS_FNAME, 'wt') as f:
                for tname in self.failed_tnames:
                    f.write(tname + '\n')

    def _save_result(self, result):
        if not result.wasSuccessful():
            for t in result.errors + result.failures:
                tname = t[0].id()
                self.failed_tnames.add(tname)

    def _run(self, suite):
        try:
            result = super().run(suite)
        except (KeyboardInterrupt, SystemExit):
            result = self.runner.result
            result.printErrors()
            raise sys.exit(1)
        else:
            self._save_result(result)
            return result

    def _exit(self, success):
        if success:
            cprint("SUCCESS", "green", bold=True)
            safe_rmpath(FAILED_TESTS_FNAME)
            sys.exit(0)
        else:
            cprint("FAILED", "red", bold=True)
            self._write_last_failed()
            sys.exit(1)

    def run(self, suite):
        result = self._run(suite)
        self._exit(result.wasSuccessful())


class ParallelRunner(ColouredTextRunner):

    @staticmethod
    def _parallelize(suite):
        def fdopen(fd, mode, *kwds):
            stream = orig_fdopen(fd, mode)
            atexit.register(stream.close)
            return stream

        # Monkey patch concurrencytest lib bug (fdopen() stream not closed).
        # https://github.com/cgoldberg/concurrencytest/issues/11
        orig_fdopen = os.fdopen
        concurrencytest.os.fdopen = fdopen
        forker = concurrencytest.fork_for_tests(NWORKERS)
        return concurrencytest.ConcurrentTestSuite(suite, forker)

    @staticmethod
    def _split_suite(suite):
        serial = unittest.TestSuite()
        parallel = unittest.TestSuite()
        for test in suite:
            if test.countTestCases() == 0:
                continue
            elif isinstance(test, unittest.TestSuite):
                test_class = test._tests[0].__class__
            elif isinstance(test, unittest.TestCase):
                test_class = test
            else:
                raise TypeError("can't recognize type %r" % test)

            if getattr(test_class, '_serialrun', False):
                serial.addTest(test)
            else:
                parallel.addTest(test)
        return (serial, parallel)

    def run(self, suite):
        ser_suite, par_suite = self._split_suite(suite)
        par_suite = self._parallelize(par_suite)

        # run parallel
        cprint("starting parallel tests using %s workers" % NWORKERS,
               "green", bold=True)
        t = time.time()
        par = self._run(par_suite)
        par_elapsed = time.time() - t

        # At this point we should have N zombies (the workers), which
        # will disappear with wait().
        orphans = psutil.Process().children()
        gone, alive = psutil.wait_procs(orphans, timeout=1)
        if alive:
            cprint("alive processes %s" % alive, "red")
            reap_children()

        # run serial
        t = time.time()
        ser = self._run(ser_suite)
        ser_elapsed = time.time() - t

        # print
        if not par.wasSuccessful() and ser_suite.countTestCases() > 0:
            par.printErrors()  # print them again at the bottom
        par_fails, par_errs, par_skips = map(len, (par.failures,
                                                   par.errors,
                                                   par.skipped))
        ser_fails, ser_errs, ser_skips = map(len, (ser.failures,
                                                   ser.errors,
                                                   ser.skipped))
        print(textwrap.dedent("""
            +----------+----------+----------+----------+----------+----------+
            |          |    total | failures |   errors |  skipped |     time |
            +----------+----------+----------+----------+----------+----------+
            | parallel |      %3s |      %3s |      %3s |      %3s |    %.2fs |
            +----------+----------+----------+----------+----------+----------+
            | serial   |      %3s |      %3s |      %3s |      %3s |    %.2fs |
            +----------+----------+----------+----------+----------+----------+
            """ % (par.testsRun, par_fails, par_errs, par_skips, par_elapsed,
                   ser.testsRun, ser_fails, ser_errs, ser_skips, ser_elapsed)))
        print("Ran %s tests in %.3fs using %s workers" % (
            par.testsRun + ser.testsRun, par_elapsed + ser_elapsed, NWORKERS))
        ok = par.wasSuccessful() and ser.wasSuccessful()
        self._exit(ok)


def get_runner(parallel=False):
    def warn(msg):
        cprint(msg + " Running serial tests instead.", "red")
    if parallel:
        if psutil.WINDOWS:
            warn("Can't run parallel tests on Windows.")
        elif concurrencytest is None:
            warn("concurrencytest module is not installed.")
        elif NWORKERS == 1:
            warn("Only 1 CPU available.")
        else:
            return ParallelRunner(verbosity=VERBOSITY)
    return ColouredTextRunner(verbosity=VERBOSITY)


# Used by test_*,py modules.
def run_from_name(name):
    if CI_TESTING:
        print_sysinfo()
    suite = TestLoader().from_name(name)
    runner = get_runner()
    runner.run(suite)


def setup():
    psutil._set_debug(True)


def main():
    setup()
    usage = "python3 -m psutil.tests [opts] [test-name]"
    parser = optparse.OptionParser(usage=usage, description="run unit tests")
    parser.add_option("--last-failed",
                      action="store_true", default=False,
                      help="only run last failed tests")
    parser.add_option("--parallel",
                      action="store_true", default=False,
                      help="run tests in parallel")
    opts, args = parser.parse_args()

    if not opts.last_failed:
        safe_rmpath(FAILED_TESTS_FNAME)

    # loader
    loader = TestLoader()
    if args:
        if len(args) > 1:
            parser.print_usage()
            return sys.exit(1)
        else:
            suite = loader.from_name(args[0])
    elif opts.last_failed:
        suite = loader.last_failed()
    else:
        suite = loader.all()

    if CI_TESTING:
        print_sysinfo()
    runner = get_runner(opts.parallel)
    runner.run(suite)


if __name__ == '__main__':
    main()
