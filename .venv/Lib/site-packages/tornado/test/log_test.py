#
# Copyright 2012 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import contextlib
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings

from tornado.escape import utf8
from tornado.log import LogFormatter, define_logging_options, enable_pretty_logging
from tornado.options import OptionParser
from tornado.util import basestring_type


@contextlib.contextmanager
def ignore_bytes_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=BytesWarning)
        yield


class LogFormatterTest(unittest.TestCase):
    # Matches the output of a single logging call (which may be multiple lines
    # if a traceback was included, so we use the DOTALL option)
    LINE_RE = re.compile(
        b"(?s)\x01\\[E [0-9]{6} [0-9]{2}:[0-9]{2}:[0-9]{2} log_test:[0-9]+\\]\x02 (.*)"
    )

    def setUp(self):
        self.formatter = LogFormatter(color=False)
        # Fake color support.  We can't guarantee anything about the $TERM
        # variable when the tests are run, so just patch in some values
        # for testing.  (testing with color off fails to expose some potential
        # encoding issues from the control characters)
        self.formatter._colors = {logging.ERROR: "\u0001"}
        self.formatter._normal = "\u0002"
        # construct a Logger directly to bypass getLogger's caching
        self.logger = logging.Logger("LogFormatterTest")
        self.logger.propagate = False
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, "log.out")
        self.handler = self.make_handler(self.filename)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.handler.close()
        os.unlink(self.filename)
        os.rmdir(self.tempdir)

    def make_handler(self, filename):
        # Base case: default setup without explicit encoding.
        # In python 2, supports arbitrary byte strings and unicode objects
        # that contain only ascii.  In python 3, supports ascii-only unicode
        # strings (but byte strings will be repr'd automatically).
        return logging.FileHandler(filename)

    def get_output(self):
        with open(self.filename, "rb") as f:
            line = f.read().strip()
            m = LogFormatterTest.LINE_RE.match(line)
            if m:
                return m.group(1)
            else:
                raise Exception("output didn't match regex: %r" % line)

    def test_basic_logging(self):
        self.logger.error("foo")
        self.assertEqual(self.get_output(), b"foo")

    def test_bytes_logging(self):
        with ignore_bytes_warning():
            # This will be "\xe9" on python 2 or "b'\xe9'" on python 3
            self.logger.error(b"\xe9")
            self.assertEqual(self.get_output(), utf8(repr(b"\xe9")))

    def test_utf8_logging(self):
        with ignore_bytes_warning():
            self.logger.error("\u00e9".encode("utf8"))
        if issubclass(bytes, basestring_type):
            # on python 2, utf8 byte strings (and by extension ascii byte
            # strings) are passed through as-is.
            self.assertEqual(self.get_output(), utf8("\u00e9"))
        else:
            # on python 3, byte strings always get repr'd even if
            # they're ascii-only, so this degenerates into another
            # copy of test_bytes_logging.
            self.assertEqual(self.get_output(), utf8(repr(utf8("\u00e9"))))

    def test_bytes_exception_logging(self):
        try:
            raise Exception(b"\xe9")
        except Exception:
            self.logger.exception("caught exception")
        # This will be "Exception: \xe9" on python 2 or
        # "Exception: b'\xe9'" on python 3.
        output = self.get_output()
        self.assertRegex(output, rb"Exception.*\\xe9")
        # The traceback contains newlines, which should not have been escaped.
        self.assertNotIn(rb"\n", output)


class UnicodeLogFormatterTest(LogFormatterTest):
    def make_handler(self, filename):
        # Adding an explicit encoding configuration allows non-ascii unicode
        # strings in both python 2 and 3, without changing the behavior
        # for byte strings.
        return logging.FileHandler(filename, encoding="utf8")

    def test_unicode_logging(self):
        self.logger.error("\u00e9")
        self.assertEqual(self.get_output(), utf8("\u00e9"))


class EnablePrettyLoggingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.options = OptionParser()
        define_logging_options(self.options)
        self.logger = logging.Logger("tornado.test.log_test.EnablePrettyLoggingTest")
        self.logger.propagate = False

    def test_log_file(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self.options.log_file_prefix = tmpdir + "/test_log"
            enable_pretty_logging(options=self.options, logger=self.logger)
            self.assertEqual(1, len(self.logger.handlers))
            self.logger.error("hello")
            self.logger.handlers[0].flush()
            filenames = glob.glob(tmpdir + "/test_log*")
            self.assertEqual(1, len(filenames))
            with open(filenames[0]) as f:
                self.assertRegex(f.read(), r"^\[E [^]]*\] hello$")
        finally:
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()
            for filename in glob.glob(tmpdir + "/test_log*"):
                os.unlink(filename)
            os.rmdir(tmpdir)

    def test_log_file_with_timed_rotating(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self.options.log_file_prefix = tmpdir + "/test_log"
            self.options.log_rotate_mode = "time"
            enable_pretty_logging(options=self.options, logger=self.logger)
            self.logger.error("hello")
            self.logger.handlers[0].flush()
            filenames = glob.glob(tmpdir + "/test_log*")
            self.assertEqual(1, len(filenames))
            with open(filenames[0]) as f:
                self.assertRegex(f.read(), r"^\[E [^]]*\] hello$")
        finally:
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()
            for filename in glob.glob(tmpdir + "/test_log*"):
                os.unlink(filename)
            os.rmdir(tmpdir)

    def test_wrong_rotate_mode_value(self):
        try:
            self.options.log_file_prefix = "some_path"
            self.options.log_rotate_mode = "wrong_mode"
            self.assertRaises(
                ValueError,
                enable_pretty_logging,
                options=self.options,
                logger=self.logger,
            )
        finally:
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()


class LoggingOptionTest(unittest.TestCase):
    """Test the ability to enable and disable Tornado's logging hooks."""

    def logs_present(self, statement, args=None):
        # Each test may manipulate and/or parse the options and then logs
        # a line at the 'info' level.  This level is ignored in the
        # logging module by default, but Tornado turns it on by default
        # so it is the easiest way to tell whether tornado's logging hooks
        # ran.
        IMPORT = "from tornado.options import options, parse_command_line"
        LOG_INFO = 'import logging; logging.info("hello")'
        program = ";".join([IMPORT, statement, LOG_INFO])
        proc = subprocess.Popen(
            [sys.executable, "-c", program] + (args or []),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = proc.communicate()
        self.assertEqual(proc.returncode, 0, "process failed: %r" % stdout)
        return b"hello" in stdout

    def test_default(self):
        self.assertFalse(self.logs_present("pass"))

    def test_tornado_default(self):
        self.assertTrue(self.logs_present("parse_command_line()"))

    def test_disable_command_line(self):
        self.assertFalse(self.logs_present("parse_command_line()", ["--logging=none"]))

    def test_disable_command_line_case_insensitive(self):
        self.assertFalse(self.logs_present("parse_command_line()", ["--logging=None"]))

    def test_disable_code_string(self):
        self.assertFalse(
            self.logs_present('options.logging = "none"; parse_command_line()')
        )

    def test_disable_code_none(self):
        self.assertFalse(
            self.logs_present("options.logging = None; parse_command_line()")
        )

    def test_disable_override(self):
        # command line trumps code defaults
        self.assertTrue(
            self.logs_present(
                "options.logging = None; parse_command_line()", ["--logging=info"]
            )
        )
