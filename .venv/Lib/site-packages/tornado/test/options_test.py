import datetime
from io import StringIO
import os
import sys
from unittest import mock
import unittest

from tornado.options import OptionParser, Error
from tornado.util import basestring_type
from tornado.test.util import subTest

import typing

if typing.TYPE_CHECKING:
    from typing import List  # noqa: F401


class Email(object):
    def __init__(self, value):
        if isinstance(value, str) and "@" in value:
            self._value = value
        else:
            raise ValueError()

    @property
    def value(self):
        return self._value


class OptionsTest(unittest.TestCase):
    def test_parse_command_line(self):
        options = OptionParser()
        options.define("port", default=80)
        options.parse_command_line(["main.py", "--port=443"])
        self.assertEqual(options.port, 443)

    def test_parse_config_file(self):
        options = OptionParser()
        options.define("port", default=80)
        options.define("username", default="foo")
        options.define("my_path")
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "options_test.cfg"
        )
        options.parse_config_file(config_path)
        self.assertEqual(options.port, 443)
        self.assertEqual(options.username, "李康")
        self.assertEqual(options.my_path, config_path)

    def test_parse_callbacks(self):
        options = OptionParser()
        self.called = False

        def callback():
            self.called = True

        options.add_parse_callback(callback)

        # non-final parse doesn't run callbacks
        options.parse_command_line(["main.py"], final=False)
        self.assertFalse(self.called)

        # final parse does
        options.parse_command_line(["main.py"])
        self.assertTrue(self.called)

        # callbacks can be run more than once on the same options
        # object if there are multiple final parses
        self.called = False
        options.parse_command_line(["main.py"])
        self.assertTrue(self.called)

    def test_help(self):
        options = OptionParser()
        try:
            orig_stderr = sys.stderr
            sys.stderr = StringIO()
            with self.assertRaises(SystemExit):
                options.parse_command_line(["main.py", "--help"])
            usage = sys.stderr.getvalue()
        finally:
            sys.stderr = orig_stderr
        self.assertIn("Usage:", usage)

    def test_subcommand(self):
        base_options = OptionParser()
        base_options.define("verbose", default=False)
        sub_options = OptionParser()
        sub_options.define("foo", type=str)
        rest = base_options.parse_command_line(
            ["main.py", "--verbose", "subcommand", "--foo=bar"]
        )
        self.assertEqual(rest, ["subcommand", "--foo=bar"])
        self.assertTrue(base_options.verbose)
        rest2 = sub_options.parse_command_line(rest)
        self.assertEqual(rest2, [])
        self.assertEqual(sub_options.foo, "bar")

        # the two option sets are distinct
        try:
            orig_stderr = sys.stderr
            sys.stderr = StringIO()
            with self.assertRaises(Error):
                sub_options.parse_command_line(["subcommand", "--verbose"])
        finally:
            sys.stderr = orig_stderr

    def test_setattr(self):
        options = OptionParser()
        options.define("foo", default=1, type=int)
        options.foo = 2
        self.assertEqual(options.foo, 2)

    def test_setattr_type_check(self):
        # setattr requires that options be the right type and doesn't
        # parse from string formats.
        options = OptionParser()
        options.define("foo", default=1, type=int)
        with self.assertRaises(Error):
            options.foo = "2"

    def test_setattr_with_callback(self):
        values = []  # type: List[int]
        options = OptionParser()
        options.define("foo", default=1, type=int, callback=values.append)
        options.foo = 2
        self.assertEqual(values, [2])

    def _sample_options(self):
        options = OptionParser()
        options.define("a", default=1)
        options.define("b", default=2)
        return options

    def test_iter(self):
        options = self._sample_options()
        # OptionParsers always define 'help'.
        self.assertEqual(set(["a", "b", "help"]), set(iter(options)))

    def test_getitem(self):
        options = self._sample_options()
        self.assertEqual(1, options["a"])

    def test_setitem(self):
        options = OptionParser()
        options.define("foo", default=1, type=int)
        options["foo"] = 2
        self.assertEqual(options["foo"], 2)

    def test_items(self):
        options = self._sample_options()
        # OptionParsers always define 'help'.
        expected = [("a", 1), ("b", 2), ("help", options.help)]
        actual = sorted(options.items())
        self.assertEqual(expected, actual)

    def test_as_dict(self):
        options = self._sample_options()
        expected = {"a": 1, "b": 2, "help": options.help}
        self.assertEqual(expected, options.as_dict())

    def test_group_dict(self):
        options = OptionParser()
        options.define("a", default=1)
        options.define("b", group="b_group", default=2)

        frame = sys._getframe(0)
        this_file = frame.f_code.co_filename
        self.assertEqual(set(["b_group", "", this_file]), options.groups())

        b_group_dict = options.group_dict("b_group")
        self.assertEqual({"b": 2}, b_group_dict)

        self.assertEqual({}, options.group_dict("nonexistent"))

    def test_mock_patch(self):
        # ensure that our setattr hooks don't interfere with mock.patch
        options = OptionParser()
        options.define("foo", default=1)
        options.parse_command_line(["main.py", "--foo=2"])
        self.assertEqual(options.foo, 2)

        with mock.patch.object(options.mockable(), "foo", 3):
            self.assertEqual(options.foo, 3)
        self.assertEqual(options.foo, 2)

        # Try nested patches mixed with explicit sets
        with mock.patch.object(options.mockable(), "foo", 4):
            self.assertEqual(options.foo, 4)
            options.foo = 5
            self.assertEqual(options.foo, 5)
            with mock.patch.object(options.mockable(), "foo", 6):
                self.assertEqual(options.foo, 6)
            self.assertEqual(options.foo, 5)
        self.assertEqual(options.foo, 2)

    def _define_options(self):
        options = OptionParser()
        options.define("str", type=str)
        options.define("basestring", type=basestring_type)
        options.define("int", type=int)
        options.define("float", type=float)
        options.define("datetime", type=datetime.datetime)
        options.define("timedelta", type=datetime.timedelta)
        options.define("email", type=Email)
        options.define("list-of-int", type=int, multiple=True)
        options.define("list-of-str", type=str, multiple=True)
        return options

    def _check_options_values(self, options):
        self.assertEqual(options.str, "asdf")
        self.assertEqual(options.basestring, "qwer")
        self.assertEqual(options.int, 42)
        self.assertEqual(options.float, 1.5)
        self.assertEqual(options.datetime, datetime.datetime(2013, 4, 28, 5, 16))
        self.assertEqual(options.timedelta, datetime.timedelta(seconds=45))
        self.assertEqual(options.email.value, "tornado@web.com")
        self.assertTrue(isinstance(options.email, Email))
        self.assertEqual(options.list_of_int, [1, 2, 3])
        self.assertEqual(options.list_of_str, ["a", "b", "c"])

    def test_types(self):
        options = self._define_options()
        options.parse_command_line(
            [
                "main.py",
                "--str=asdf",
                "--basestring=qwer",
                "--int=42",
                "--float=1.5",
                "--datetime=2013-04-28 05:16",
                "--timedelta=45s",
                "--email=tornado@web.com",
                "--list-of-int=1,2,3",
                "--list-of-str=a,b,c",
            ]
        )
        self._check_options_values(options)

    def test_types_with_conf_file(self):
        for config_file_name in (
            "options_test_types.cfg",
            "options_test_types_str.cfg",
        ):
            options = self._define_options()
            options.parse_config_file(
                os.path.join(os.path.dirname(__file__), config_file_name)
            )
            self._check_options_values(options)

    def test_multiple_string(self):
        options = OptionParser()
        options.define("foo", type=str, multiple=True)
        options.parse_command_line(["main.py", "--foo=a,b,c"])
        self.assertEqual(options.foo, ["a", "b", "c"])

    def test_multiple_int(self):
        options = OptionParser()
        options.define("foo", type=int, multiple=True)
        options.parse_command_line(["main.py", "--foo=1,3,5:7"])
        self.assertEqual(options.foo, [1, 3, 5, 6, 7])

    def test_error_redefine(self):
        options = OptionParser()
        options.define("foo")
        with self.assertRaises(Error) as cm:
            options.define("foo")
        self.assertRegex(str(cm.exception), "Option.*foo.*already defined")

    def test_error_redefine_underscore(self):
        # Ensure that the dash/underscore normalization doesn't
        # interfere with the redefinition error.
        tests = [
            ("foo-bar", "foo-bar"),
            ("foo_bar", "foo_bar"),
            ("foo-bar", "foo_bar"),
            ("foo_bar", "foo-bar"),
        ]
        for a, b in tests:
            with subTest(self, a=a, b=b):
                options = OptionParser()
                options.define(a)
                with self.assertRaises(Error) as cm:
                    options.define(b)
                self.assertRegex(str(cm.exception), "Option.*foo.bar.*already defined")

    def test_dash_underscore_cli(self):
        # Dashes and underscores should be interchangeable.
        for defined_name in ["foo-bar", "foo_bar"]:
            for flag in ["--foo-bar=a", "--foo_bar=a"]:
                options = OptionParser()
                options.define(defined_name)
                options.parse_command_line(["main.py", flag])
                # Attr-style access always uses underscores.
                self.assertEqual(options.foo_bar, "a")
                # Dict-style access allows both.
                self.assertEqual(options["foo-bar"], "a")
                self.assertEqual(options["foo_bar"], "a")

    def test_dash_underscore_file(self):
        # No matter how an option was defined, it can be set with underscores
        # in a config file.
        for defined_name in ["foo-bar", "foo_bar"]:
            options = OptionParser()
            options.define(defined_name)
            options.parse_config_file(
                os.path.join(os.path.dirname(__file__), "options_test.cfg")
            )
            self.assertEqual(options.foo_bar, "a")

    def test_dash_underscore_introspection(self):
        # Original names are preserved in introspection APIs.
        options = OptionParser()
        options.define("with-dash", group="g")
        options.define("with_underscore", group="g")
        all_options = ["help", "with-dash", "with_underscore"]
        self.assertEqual(sorted(options), all_options)
        self.assertEqual(sorted(k for (k, v) in options.items()), all_options)
        self.assertEqual(sorted(options.as_dict().keys()), all_options)

        self.assertEqual(
            sorted(options.group_dict("g")), ["with-dash", "with_underscore"]
        )

        # --help shows CLI-style names with dashes.
        buf = StringIO()
        options.print_help(buf)
        self.assertIn("--with-dash", buf.getvalue())
        self.assertIn("--with-underscore", buf.getvalue())
