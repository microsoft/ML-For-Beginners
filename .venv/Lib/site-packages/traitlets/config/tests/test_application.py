"""
Tests for traitlets.config.application.Application
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import contextlib
import io
import json
import logging
import os
import sys
import typing as t
from io import StringIO
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest
from pytest import mark

from traitlets import Bool, Bytes, Dict, HasTraits, Integer, List, Set, Tuple, Unicode
from traitlets.config.application import Application
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config, KVArgParseConfigLoader
from traitlets.tests.utils import check_help_all_output, check_help_output, get_output_error_code

try:
    from unittest import mock
except ImportError:
    from unittest import mock

pjoin = os.path.join


class Foo(Configurable):

    i = Integer(
        0,
        help="""
    The integer i.

    Details about i.
    """,
    ).tag(config=True)
    j = Integer(1, help="The integer j.").tag(config=True)
    name = Unicode("Brian", help="First name.").tag(config=True)
    la = List([]).tag(config=True)
    li = List(Integer()).tag(config=True)
    fdict = Dict().tag(config=True, multiplicity="+")


class Bar(Configurable):

    b = Integer(0, help="The integer b.").tag(config=True)
    enabled = Bool(True, help="Enable bar.").tag(config=True)
    tb = Tuple(()).tag(config=True, multiplicity="*")
    aset = Set().tag(config=True, multiplicity="+")
    bdict = Dict().tag(config=True)
    idict = Dict(value_trait=Integer()).tag(config=True)
    key_dict = Dict(per_key_traits={"i": Integer(), "b": Bytes()}).tag(config=True)


class MyApp(Application):

    name = Unicode("myapp")
    running = Bool(False, help="Is the app running?").tag(config=True)
    classes = List([Bar, Foo])  # type:ignore
    config_file = Unicode("", help="Load this config file").tag(config=True)

    warn_tpyo = Unicode(
        "yes the name is wrong on purpose",
        config=True,
        help="Should print a warning if `MyApp.warn-typo=...` command is passed",
    )

    aliases: t.Dict[t.Any, t.Any] = {}
    aliases.update(Application.aliases)
    aliases.update(
        {
            ("fooi", "i"): "Foo.i",
            ("j", "fooj"): ("Foo.j", "`j` terse help msg"),
            "name": "Foo.name",
            "la": "Foo.la",
            "li": "Foo.li",
            "tb": "Bar.tb",
            "D": "Bar.bdict",
            "enabled": "Bar.enabled",
            "enable": "Bar.enabled",
            "log-level": "Application.log_level",
        }
    )

    flags: t.Dict[t.Any, t.Any] = {}
    flags.update(Application.flags)
    flags.update(
        {
            ("enable", "e"): ({"Bar": {"enabled": True}}, "Set Bar.enabled to True"),
            ("d", "disable"): ({"Bar": {"enabled": False}}, "Set Bar.enabled to False"),
            "crit": ({"Application": {"log_level": logging.CRITICAL}}, "set level=CRITICAL"),
        }
    )

    def init_foo(self):
        self.foo = Foo(parent=self)

    def init_bar(self):
        self.bar = Bar(parent=self)


def class_to_names(classes):
    return [klass.__name__ for klass in classes]


class TestApplication(TestCase):
    def test_log(self):
        stream = StringIO()
        app = MyApp(log_level=logging.INFO)
        handler = logging.StreamHandler(stream)
        # trigger reconstruction of the log formatter
        app.log_format = "%(message)s"
        app.log_datefmt = "%Y-%m-%d %H:%M"
        app.log.handlers = [handler]
        app.log.info("hello")
        assert "hello" in stream.getvalue()

    def test_no_eval_cli_text(self):
        app = MyApp()
        app.initialize(["--Foo.name=1"])
        app.init_foo()
        assert app.foo.name == "1"

    def test_basic(self):
        app = MyApp()
        self.assertEqual(app.name, "myapp")
        self.assertEqual(app.running, False)
        self.assertEqual(app.classes, [MyApp, Bar, Foo])  # type:ignore
        self.assertEqual(app.config_file, "")

    def test_app_name_set_via_constructor(self):
        app = MyApp(name='set_via_constructor')
        assert app.name == "set_via_constructor"

    def test_mro_discovery(self):
        app = MyApp()

        self.assertSequenceEqual(
            class_to_names(app._classes_with_config_traits()),
            ["Application", "MyApp", "Bar", "Foo"],
        )
        self.assertSequenceEqual(
            class_to_names(app._classes_inc_parents()),
            [
                "Configurable",
                "LoggingConfigurable",
                "SingletonConfigurable",
                "Application",
                "MyApp",
                "Bar",
                "Foo",
            ],
        )

        self.assertSequenceEqual(
            class_to_names(app._classes_with_config_traits([Application])), ["Application"]
        )
        self.assertSequenceEqual(
            class_to_names(app._classes_inc_parents([Application])),
            ["Configurable", "LoggingConfigurable", "SingletonConfigurable", "Application"],
        )

        self.assertSequenceEqual(class_to_names(app._classes_with_config_traits([Foo])), ["Foo"])
        self.assertSequenceEqual(
            class_to_names(app._classes_inc_parents([Bar])), ["Configurable", "Bar"]
        )

        class MyApp2(Application):  # no defined `classes` attr
            pass

        self.assertSequenceEqual(class_to_names(app._classes_with_config_traits([Foo])), ["Foo"])
        self.assertSequenceEqual(
            class_to_names(app._classes_inc_parents([Bar])), ["Configurable", "Bar"]
        )

    def test_config(self):
        app = MyApp()
        app.parse_command_line(
            [
                "--i=10",
                "--Foo.j=10",
                "--enable=False",
                "--log-level=50",
            ]
        )
        config = app.config
        print(config)
        self.assertEqual(config.Foo.i, 10)
        self.assertEqual(config.Foo.j, 10)
        self.assertEqual(config.Bar.enabled, False)
        self.assertEqual(config.MyApp.log_level, 50)

    def test_config_seq_args(self):
        app = MyApp()
        app.parse_command_line(
            "--li 1 --li 3 --la 1 --tb AB 2 --Foo.la=ab --Bar.aset S1 --Bar.aset S2 --Bar.aset S1".split()
        )
        assert app.extra_args == ["2"]
        config = app.config
        assert config.Foo.li == [1, 3]
        assert config.Foo.la == ["1", "ab"]
        assert config.Bar.tb == ("AB",)
        self.assertEqual(config.Bar.aset, {"S1", "S2"})
        app.init_foo()
        assert app.foo.li == [1, 3]
        assert app.foo.la == ["1", "ab"]
        app.init_bar()
        self.assertEqual(app.bar.aset, {"S1", "S2"})
        assert app.bar.tb == ("AB",)

    def test_config_dict_args(self):
        app = MyApp()
        app.parse_command_line(
            "--Foo.fdict a=1 --Foo.fdict b=b --Foo.fdict c=3 "
            "--Bar.bdict k=1 -D=a=b -D 22=33 "
            "--Bar.idict k=1 --Bar.idict b=2 --Bar.idict c=3 ".split()
        )
        fdict = {"a": "1", "b": "b", "c": "3"}
        bdict = {"k": "1", "a": "b", "22": "33"}
        idict = {"k": 1, "b": 2, "c": 3}
        config = app.config
        assert config.Bar.idict == idict
        self.assertDictEqual(config.Foo.fdict, fdict)
        self.assertDictEqual(config.Bar.bdict, bdict)
        app.init_foo()
        self.assertEqual(app.foo.fdict, fdict)
        app.init_bar()
        assert app.bar.idict == idict
        self.assertEqual(app.bar.bdict, bdict)

    def test_config_propagation(self):
        app = MyApp()
        app.parse_command_line(["--i=10", "--Foo.j=10", "--enable=False", "--log-level=50"])
        app.init_foo()
        app.init_bar()
        self.assertEqual(app.foo.i, 10)
        self.assertEqual(app.foo.j, 10)
        self.assertEqual(app.bar.enabled, False)

    def test_cli_priority(self):
        """Test that loading config files does not override CLI options"""
        name = "config.py"

        class TestApp(Application):
            value = Unicode().tag(config=True)
            config_file_loaded = Bool().tag(config=True)
            aliases = {"v": "TestApp.value"}

        app = TestApp()
        with TemporaryDirectory() as td:
            config_file = pjoin(td, name)
            with open(config_file, "w") as f:
                f.writelines(
                    ["c.TestApp.value = 'config file'\n", "c.TestApp.config_file_loaded = True\n"]
                )

            app.parse_command_line(["--v=cli"])
            assert "value" in app.config.TestApp
            assert app.config.TestApp.value == "cli"
            assert app.value == "cli"

            app.load_config_file(name, path=[td])
            assert app.config_file_loaded
            assert app.config.TestApp.value == "cli"
            assert app.value == "cli"

    def test_ipython_cli_priority(self):
        # this test is almost entirely redundant with above,
        # but we can keep it around in case of subtle issues creeping into
        # the exact sequence IPython follows.
        name = "config.py"

        class TestApp(Application):
            value = Unicode().tag(config=True)
            config_file_loaded = Bool().tag(config=True)
            aliases = {"v": ("TestApp.value", "some help")}

        app = TestApp()
        with TemporaryDirectory() as td:
            config_file = pjoin(td, name)
            with open(config_file, "w") as f:
                f.writelines(
                    ["c.TestApp.value = 'config file'\n", "c.TestApp.config_file_loaded = True\n"]
                )
            # follow IPython's config-loading sequence to ensure CLI priority is preserved
            app.parse_command_line(["--v=cli"])
            # this is where IPython makes a mistake:
            # it assumes app.config will not be modified,
            # and storing a reference is storing a copy
            cli_config = app.config
            assert "value" in app.config.TestApp
            assert app.config.TestApp.value == "cli"
            assert app.value == "cli"
            app.load_config_file(name, path=[td])
            assert app.config_file_loaded
            # enforce cl-opts override config file opts:
            # this is where IPython makes a mistake: it assumes
            # that cl_config is a different object, but it isn't.
            app.update_config(cli_config)
            assert app.config.TestApp.value == "cli"
            assert app.value == "cli"

    def test_cli_allow_none(self):
        class App(Application):
            aliases = {"opt": "App.opt"}
            opt = Unicode(allow_none=True, config=True)

        app = App()
        app.parse_command_line(["--opt=None"])
        assert app.opt is None

    def test_flags(self):
        app = MyApp()
        app.parse_command_line(["--disable"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, False)

        app = MyApp()
        app.parse_command_line(["-d"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, False)

        app = MyApp()
        app.parse_command_line(["--enable"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, True)

        app = MyApp()
        app.parse_command_line(["-e"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, True)

    def test_flags_help_msg(self):
        app = MyApp()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            app.print_flag_help()
        hmsg = stdout.getvalue()
        self.assertRegex(hmsg, "(?<!-)-e, --enable\\b")
        self.assertRegex(hmsg, "(?<!-)-d, --disable\\b")
        self.assertIn("Equivalent to: [--Bar.enabled=True]", hmsg)
        self.assertIn("Equivalent to: [--Bar.enabled=False]", hmsg)

    def test_aliases(self):
        app = MyApp()
        app.parse_command_line(["--i=5", "--j=10"])
        app.init_foo()
        self.assertEqual(app.foo.i, 5)
        app.init_foo()
        self.assertEqual(app.foo.j, 10)

        app = MyApp()
        app.parse_command_line(["-i=5", "-j=10"])
        app.init_foo()
        self.assertEqual(app.foo.i, 5)
        app.init_foo()
        self.assertEqual(app.foo.j, 10)

        app = MyApp()
        app.parse_command_line(["--fooi=5", "--fooj=10"])
        app.init_foo()
        self.assertEqual(app.foo.i, 5)
        app.init_foo()
        self.assertEqual(app.foo.j, 10)

    def test_aliases_multiple(self):
        # Test multiple > 2 aliases for the same argument
        class TestMultiAliasApp(Application):
            foo = Integer(config=True)
            aliases = {("f", "bar", "qux"): "TestMultiAliasApp.foo"}

        app = TestMultiAliasApp()
        app.parse_command_line(["-f", "3"])
        self.assertEqual(app.foo, 3)

        app = TestMultiAliasApp()
        app.parse_command_line(["--bar", "4"])
        self.assertEqual(app.foo, 4)

        app = TestMultiAliasApp()
        app.parse_command_line(["--qux", "5"])
        self.assertEqual(app.foo, 5)

    def test_aliases_help_msg(self):
        app = MyApp()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            app.print_alias_help()
        hmsg = stdout.getvalue()
        self.assertRegex(hmsg, "(?<!-)-i, --fooi\\b")
        self.assertRegex(hmsg, "(?<!-)-j, --fooj\\b")
        self.assertIn("Equivalent to: [--Foo.i]", hmsg)
        self.assertIn("Equivalent to: [--Foo.j]", hmsg)
        self.assertIn("Equivalent to: [--Foo.name]", hmsg)

    def test_alias_unrecognized(self):
        """Check ability to override handling for unrecognized aliases"""

        class StrictLoader(KVArgParseConfigLoader):
            def _handle_unrecognized_alias(self, arg):
                self.parser.error("Unrecognized alias: %s" % arg)

        class StrictApplication(Application):
            def _create_loader(self, argv, aliases, flags, classes):
                return StrictLoader(argv, aliases, flags, classes=classes, log=self.log)

        app = StrictApplication()
        app.initialize(["--log-level=20"])  # recognized alias
        assert app.log_level == 20

        app = StrictApplication()
        with pytest.raises(SystemExit, match="2"):
            app.initialize(["--unrecognized=20"])

        # Ideally we would use pytest capsys fixture, but fixtures are incompatible
        # with unittest.TestCase-style classes :(
        # stderr = capsys.readouterr().err
        # assert "Unrecognized alias: unrecognized" in stderr

    def test_flag_clobber(self):
        """test that setting flags doesn't clobber existing settings"""
        app = MyApp()
        app.parse_command_line(["--Bar.b=5", "--disable"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, False)
        self.assertEqual(app.bar.b, 5)
        app.parse_command_line(["--enable", "--Bar.b=10"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, True)
        self.assertEqual(app.bar.b, 10)

    def test_warn_autocorrect(self):
        stream = StringIO()
        app = MyApp(log_level=logging.INFO)
        app.log.handlers = [logging.StreamHandler(stream)]

        cfg = Config()
        cfg.MyApp.warn_typo = "WOOOO"
        app.config = cfg

        self.assertIn("warn_typo", stream.getvalue())
        self.assertIn("warn_tpyo", stream.getvalue())

    def test_flatten_flags(self):
        cfg = Config()
        cfg.MyApp.log_level = logging.WARN
        app = MyApp()
        app.update_config(cfg)
        self.assertEqual(app.log_level, logging.WARN)
        self.assertEqual(app.config.MyApp.log_level, logging.WARN)
        app.initialize(["--crit"])
        self.assertEqual(app.log_level, logging.CRITICAL)
        # this would be app.config.Application.log_level if it failed:
        self.assertEqual(app.config.MyApp.log_level, logging.CRITICAL)

    def test_flatten_aliases(self):
        cfg = Config()
        cfg.MyApp.log_level = logging.WARN
        app = MyApp()
        app.update_config(cfg)
        self.assertEqual(app.log_level, logging.WARN)
        self.assertEqual(app.config.MyApp.log_level, logging.WARN)
        app.initialize(["--log-level", "CRITICAL"])
        self.assertEqual(app.log_level, logging.CRITICAL)
        # this would be app.config.Application.log_level if it failed:
        self.assertEqual(app.config.MyApp.log_level, "CRITICAL")

    def test_extra_args(self):

        app = MyApp()
        app.parse_command_line(["--Bar.b=5", "extra", "args", "--disable"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, False)
        self.assertEqual(app.bar.b, 5)
        self.assertEqual(app.extra_args, ["extra", "args"])

        app = MyApp()
        app.parse_command_line(["--Bar.b=5", "--", "extra", "--disable", "args"])
        app.init_bar()
        self.assertEqual(app.bar.enabled, True)
        self.assertEqual(app.bar.b, 5)
        self.assertEqual(app.extra_args, ["extra", "--disable", "args"])

        app = MyApp()
        app.parse_command_line(["--disable", "--la", "-", "-", "--Bar.b=1", "--", "-", "extra"])
        self.assertEqual(app.extra_args, ["-", "-", "extra"])

    def test_unicode_argv(self):
        app = MyApp()
        app.parse_command_line(["ünîcødé"])

    def test_document_config_option(self):
        app = MyApp()
        app.document_config_options()

    def test_generate_config_file(self):
        app = MyApp()
        assert "The integer b." in app.generate_config_file()

    def test_generate_config_file_classes_to_include(self):
        class NotInConfig(HasTraits):
            from_hidden = Unicode(
                "x",
                help="""From hidden class

            Details about from_hidden.
            """,
            ).tag(config=True)

        class NoTraits(Foo, Bar, NotInConfig):
            pass

        app = MyApp()
        app.classes.append(NoTraits)  # type:ignore

        conf_txt = app.generate_config_file()
        print(conf_txt)
        self.assertIn("The integer b.", conf_txt)
        self.assertIn("# Foo(Configurable)", conf_txt)
        self.assertNotIn("# Configurable", conf_txt)
        self.assertIn("# NoTraits(Foo, Bar)", conf_txt)

        # inherited traits, parent in class list:
        self.assertIn("# c.NoTraits.i", conf_txt)
        self.assertIn("# c.NoTraits.j", conf_txt)
        self.assertIn("# c.NoTraits.n", conf_txt)
        self.assertIn("#  See also: Foo.j", conf_txt)
        self.assertIn("#  See also: Bar.b", conf_txt)
        self.assertEqual(conf_txt.count("Details about i."), 1)

        # inherited traits, parent not in class list:
        self.assertIn("# c.NoTraits.from_hidden", conf_txt)
        self.assertNotIn("#  See also: NotInConfig.", conf_txt)
        self.assertEqual(conf_txt.count("Details about from_hidden."), 1)
        self.assertNotIn("NotInConfig", conf_txt)

    def test_multi_file(self):
        app = MyApp()
        app.log = logging.getLogger()
        name = "config.py"
        with TemporaryDirectory("_1") as td1:
            with open(pjoin(td1, name), "w") as f1:
                f1.write("get_config().MyApp.Bar.b = 1")
            with TemporaryDirectory("_2") as td2:
                with open(pjoin(td2, name), "w") as f2:
                    f2.write("get_config().MyApp.Bar.b = 2")
                app.load_config_file(name, path=[td2, td1])
                app.init_bar()
                self.assertEqual(app.bar.b, 2)
                app.load_config_file(name, path=[td1, td2])
                app.init_bar()
                self.assertEqual(app.bar.b, 1)

    @mark.skipif(not hasattr(TestCase, "assertLogs"), reason="requires TestCase.assertLogs")
    def test_log_collisions(self):
        app = MyApp()
        app.log = logging.getLogger()
        app.log.setLevel(logging.INFO)
        name = "config"
        with TemporaryDirectory("_1") as td:
            with open(pjoin(td, name + ".py"), "w") as f:
                f.write("get_config().Bar.b = 1")
            with open(pjoin(td, name + ".json"), "w") as f:
                json.dump({"Bar": {"b": 2}}, f)
            with self.assertLogs(app.log, logging.WARNING) as captured:
                app.load_config_file(name, path=[td])
                app.init_bar()
        assert app.bar.b == 2
        output = "\n".join(captured.output)
        assert "Collision" in output
        assert "1 ignored, using 2" in output
        assert pjoin(td, name + ".py") in output
        assert pjoin(td, name + ".json") in output

    @mark.skipif(not hasattr(TestCase, "assertLogs"), reason="requires TestCase.assertLogs")
    def test_log_bad_config(self):
        app = MyApp()
        app.log = logging.getLogger()
        name = "config.py"
        with TemporaryDirectory() as td:
            with open(pjoin(td, name), "w") as f:
                f.write("syntax error()")
            with self.assertLogs(app.log, logging.ERROR) as captured:
                app.load_config_file(name, path=[td])
        output = "\n".join(captured.output)
        self.assertIn("SyntaxError", output)

    def test_raise_on_bad_config(self):
        app = MyApp()
        app.raise_config_file_errors = True
        app.log = logging.getLogger()
        name = "config.py"
        with TemporaryDirectory() as td:
            with open(pjoin(td, name), "w") as f:
                f.write("syntax error()")
            with self.assertRaises(SyntaxError):
                app.load_config_file(name, path=[td])

    def test_subcommands_instanciation(self):
        """Try all ways to specify how to create sub-apps."""
        app = Root.instance()
        app.parse_command_line(["sub1"])

        self.assertIsInstance(app.subapp, Sub1)
        # Check parent hierarchy.
        self.assertIs(app.subapp.parent, app)

        Root.clear_instance()
        Sub1.clear_instance()  # Otherwise, replaced spuriously and hierarchy check fails.
        app = Root.instance()

        app.parse_command_line(["sub1", "sub2"])
        self.assertIsInstance(app.subapp, Sub1)
        self.assertIsInstance(app.subapp.subapp, Sub2)
        # Check parent hierarchy.
        self.assertIs(app.subapp.parent, app)
        self.assertIs(app.subapp.subapp.parent, app.subapp)

        Root.clear_instance()
        Sub1.clear_instance()  # Otherwise, replaced spuriously and hierarchy check fails.
        app = Root.instance()

        app.parse_command_line(["sub1", "sub3"])
        self.assertIsInstance(app.subapp, Sub1)
        self.assertIsInstance(app.subapp.subapp, Sub3)
        self.assertTrue(app.subapp.subapp.flag)  # Set by factory.
        # Check parent hierarchy.
        self.assertIs(app.subapp.parent, app)
        self.assertIs(app.subapp.subapp.parent, app.subapp)  # Set by factory.

        Root.clear_instance()
        Sub1.clear_instance()

    def test_loaded_config_files(self):
        app = MyApp()
        app.log = logging.getLogger()
        name = "config.py"
        with TemporaryDirectory("_1") as td1:
            config_file = pjoin(td1, name)
            with open(config_file, "w") as f:
                f.writelines(["c.MyApp.running = True\n"])

            app.load_config_file(name, path=[td1])
            self.assertEqual(len(app.loaded_config_files), 1)
            self.assertEqual(app.loaded_config_files[0], config_file)

            app.start()
            self.assertEqual(app.running, True)

            # emulate an app that allows dynamic updates and update config file
            with open(config_file, "w") as f:
                f.writelines(["c.MyApp.running = False\n"])

            # reload and verify update, and that loaded_configs was not increased
            app.load_config_file(name, path=[td1])
            self.assertEqual(len(app.loaded_config_files), 1)
            self.assertEqual(app.running, False)

            # Attempt to update, ensure error...
            with self.assertRaises(AttributeError):
                app.loaded_config_files = "/foo"  # type:ignore

            # ensure it can't be udpated via append
            app.loaded_config_files.append("/bar")
            self.assertEqual(len(app.loaded_config_files), 1)

            # repeat to ensure no unexpected changes occurred
            app.load_config_file(name, path=[td1])
            self.assertEqual(len(app.loaded_config_files), 1)
            self.assertEqual(app.running, False)


def test_cli_multi_scalar(caplog):
    class App(Application):
        aliases = {"opt": "App.opt"}
        opt = Unicode(config=True)

    app = App(log=logging.getLogger())
    with pytest.raises(SystemExit):
        app.parse_command_line(["--opt", "1", "--opt", "2"])
    record = caplog.get_records("call")[-1]
    message = record.message

    assert "Error loading argument" in message
    assert "App.opt=['1', '2']" in message
    assert "opt only accepts one value" in message
    assert record.levelno == logging.CRITICAL


class Root(Application):
    subcommands = {
        "sub1": ("traitlets.config.tests.test_application.Sub1", "import string"),
    }


class Sub3(Application):
    flag = Bool(False)


class Sub2(Application):
    pass


class Sub1(Application):
    subcommands: dict = {  # type:ignore
        "sub2": (Sub2, "Application class"),
        "sub3": (lambda root: Sub3(parent=root, flag=True), "factory"),
    }


class DeprecatedApp(Application):
    override_called = False
    parent_called = False

    def _config_changed(self, name, old, new):
        self.override_called = True

        def _capture(*args):
            self.parent_called = True

        with mock.patch.object(self.log, "debug", _capture):
            super()._config_changed(name, old, new)


def test_deprecated_notifier():
    app = DeprecatedApp()
    assert not app.override_called
    assert not app.parent_called
    app.config = Config({"A": {"b": "c"}})
    assert app.override_called
    assert app.parent_called


def test_help_output():
    check_help_output(__name__)


def test_help_all_output():
    check_help_all_output(__name__)


def test_show_config_cli():
    out, err, ec = get_output_error_code([sys.executable, "-m", __name__, "--show-config"])
    assert ec == 0
    assert "show_config" not in out


def test_show_config_json_cli():
    out, err, ec = get_output_error_code([sys.executable, "-m", __name__, "--show-config-json"])
    assert ec == 0
    assert "show_config" not in out


def test_show_config(capsys):
    cfg = Config()
    cfg.MyApp.i = 5
    # don't show empty
    cfg.OtherApp

    app = MyApp(config=cfg, show_config=True)
    app.start()
    out, err = capsys.readouterr()
    assert "MyApp" in out
    assert "i = 5" in out
    assert "OtherApp" not in out


def test_show_config_json(capsys):
    cfg = Config()
    cfg.MyApp.i = 5
    cfg.OtherApp

    app = MyApp(config=cfg, show_config_json=True)
    app.start()
    out, err = capsys.readouterr()
    displayed = json.loads(out)
    assert Config(displayed) == cfg


def test_deep_alias():
    from traitlets import Int
    from traitlets.config import Application, Configurable

    class Foo(Configurable):
        val = Int(default_value=5).tag(config=True)

    class Bar(Configurable):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foo = Foo(parent=self)

    class TestApp(Application):
        name = "test"

        aliases = {"val": "Bar.Foo.val"}
        classes = [Foo, Bar]

        def initialize(self, *args, **kwargs):
            super().initialize(*args, **kwargs)
            self.bar = Bar(parent=self)

    app = TestApp()
    app.initialize(["--val=10"])
    assert app.bar.foo.val == 10
    assert len(list(app.emit_alias_help())) > 0


def test_logging_config(tmp_path, capsys):
    """We should be able to configure additional log handlers."""
    log_file = tmp_path / "log_file"
    app = Application(
        logging_config={
            "version": 1,
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "filename": str(log_file),
                },
            },
            "loggers": {
                "Application": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                },
            },
        }
    )
    # the default "console" handler + our new "file" handler
    assert len(app.log.handlers) == 2

    # log a couple of messages
    app.log.info("info")
    app.log.warning("warn")

    # test that log messages get written to the file
    with open(log_file) as log_handle:
        assert log_handle.read() == "info\nwarn\n"

    # test that log messages get written to stderr (default console handler)
    assert capsys.readouterr().err == "[Application] WARNING | warn\n"


def test_get_default_logging_config_pythonw(monkeypatch):
    """Ensure logging is correctly disabled for pythonw usage."""
    monkeypatch.setattr("traitlets.config.application.IS_PYTHONW", True)
    config = Application().get_default_logging_config()
    assert "handlers" not in config
    assert "loggers" not in config

    monkeypatch.setattr("traitlets.config.application.IS_PYTHONW", False)
    config = Application().get_default_logging_config()
    assert "handlers" in config
    assert "loggers" in config


@pytest.fixture
def caplogconfig(monkeypatch):
    """Capture logging config events for DictConfigurator objects.

    This suppresses the event (so the configuration doesn't happen).

    Returns a list of (args, kwargs).
    """
    calls = []

    def _configure(*args, **kwargs):
        nonlocal calls
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "logging.config.DictConfigurator.configure",
        _configure,
    )

    return calls


@pytest.mark.skipif(sys.implementation.name == "pypy", reason="Test does not work on pypy")
def test_logging_teardown_on_error(capsys, caplogconfig):
    """Ensure we don't try to open logs in order to close them (See #722).

    If you try to configure logging handlers whilst Python is shutting down
    you may get traceback.
    """
    # create and destroy an app (without configuring logging)
    # (ensure that the logs were not configured)
    app = Application()
    del app
    assert len(caplogconfig) == 0  # logging was not configured
    out, err = capsys.readouterr()
    assert "Traceback" not in err

    # ensure that the logs would have been configured otherwise
    # (to prevent this test from yielding false-negatives)
    app = Application()
    app._logging_configured = True  # make it look like logging was configured
    del app
    assert len(caplogconfig) == 1  # logging was configured


if __name__ == "__main__":
    # for test_help_output:
    MyApp.launch_instance()
