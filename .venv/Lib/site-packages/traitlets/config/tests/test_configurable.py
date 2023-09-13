"""Tests for traitlets.config.configurable"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import logging
from unittest import TestCase

from pytest import mark

from traitlets.config.application import Application
from traitlets.config.configurable import Configurable, LoggingConfigurable, SingletonConfigurable
from traitlets.config.loader import Config
from traitlets.log import get_logger
from traitlets.traitlets import (
    CaselessStrEnum,
    Dict,
    Enum,
    Float,
    FuzzyEnum,
    Integer,
    List,
    Set,
    Unicode,
    _deprecations_shown,
    validate,
)

from ...tests._warnings import expected_warnings


class MyConfigurable(Configurable):
    a = Integer(1, help="The integer a.").tag(config=True)
    b = Float(1.0, help="The integer b.").tag(config=True)
    c = Unicode("no config")


mc_help = """MyConfigurable(Configurable) options
------------------------------------
--MyConfigurable.a=<Integer>
    The integer a.
    Default: 1
--MyConfigurable.b=<Float>
    The integer b.
    Default: 1.0"""

mc_help_inst = """MyConfigurable(Configurable) options
------------------------------------
--MyConfigurable.a=<Integer>
    The integer a.
    Current: 5
--MyConfigurable.b=<Float>
    The integer b.
    Current: 4.0"""

# On Python 3, the Integer trait is a synonym for Int
mc_help = mc_help.replace("<Integer>", "<Int>")
mc_help_inst = mc_help_inst.replace("<Integer>", "<Int>")


class Foo(Configurable):
    a = Integer(0, help="The integer a.").tag(config=True)
    b = Unicode("nope").tag(config=True)
    flist = List([]).tag(config=True)
    fdict = Dict().tag(config=True)


class Bar(Foo):
    b = Unicode("gotit", help="The string b.").tag(config=False)
    c = Float(help="The string c.").tag(config=True)
    bset = Set([]).tag(config=True, multiplicity="+")
    bset_values = Set([2, 1, 5]).tag(config=True, multiplicity="+")
    bdict = Dict().tag(config=True, multiplicity="+")
    bdict_values = Dict({1: "a", "0": "b", 5: "c"}).tag(config=True, multiplicity="+")


foo_help = """Foo(Configurable) options
-------------------------
--Foo.a=<Int>
    The integer a.
    Default: 0
--Foo.b=<Unicode>
    Default: 'nope'
--Foo.fdict=<key-1>=<value-1>...
    Default: {}
--Foo.flist=<list-item-1>...
    Default: []"""

bar_help = """Bar(Foo) options
----------------
--Bar.a=<Int>
    The integer a.
    Default: 0
--Bar.bdict <key-1>=<value-1>...
    Default: {}
--Bar.bdict_values <key-1>=<value-1>...
    Default: {1: 'a', '0': 'b', 5: 'c'}
--Bar.bset <set-item-1>...
    Default: set()
--Bar.bset_values <set-item-1>...
    Default: {1, 2, 5}
--Bar.c=<Float>
    The string c.
    Default: 0.0
--Bar.fdict=<key-1>=<value-1>...
    Default: {}
--Bar.flist=<list-item-1>...
    Default: []"""


class TestConfigurable(TestCase):
    def test_default(self):
        c1 = Configurable()
        c2 = Configurable(config=c1.config)
        c3 = Configurable(config=c2.config)
        self.assertEqual(c1.config, c2.config)
        self.assertEqual(c2.config, c3.config)

    def test_custom(self):
        config = Config()
        config.foo = "foo"
        config.bar = "bar"
        c1 = Configurable(config=config)
        c2 = Configurable(config=c1.config)
        c3 = Configurable(config=c2.config)
        self.assertEqual(c1.config, config)
        self.assertEqual(c2.config, config)
        self.assertEqual(c3.config, config)
        # Test that copies are not made
        self.assertTrue(c1.config is config)
        self.assertTrue(c2.config is config)
        self.assertTrue(c3.config is config)
        self.assertTrue(c1.config is c2.config)
        self.assertTrue(c2.config is c3.config)

    def test_inheritance(self):
        config = Config()
        config.MyConfigurable.a = 2
        config.MyConfigurable.b = 2.0
        c1 = MyConfigurable(config=config)
        c2 = MyConfigurable(config=c1.config)
        self.assertEqual(c1.a, config.MyConfigurable.a)
        self.assertEqual(c1.b, config.MyConfigurable.b)
        self.assertEqual(c2.a, config.MyConfigurable.a)
        self.assertEqual(c2.b, config.MyConfigurable.b)

    def test_parent(self):
        config = Config()
        config.Foo.a = 10
        config.Foo.b = "wow"
        config.Bar.b = "later"
        config.Bar.c = 100.0
        f = Foo(config=config)
        with expected_warnings(["`b` not recognized"]):
            b = Bar(config=f.config)
        self.assertEqual(f.a, 10)
        self.assertEqual(f.b, "wow")
        self.assertEqual(b.b, "gotit")
        self.assertEqual(b.c, 100.0)

    def test_override1(self):
        config = Config()
        config.MyConfigurable.a = 2
        config.MyConfigurable.b = 2.0
        c = MyConfigurable(a=3, config=config)
        self.assertEqual(c.a, 3)
        self.assertEqual(c.b, config.MyConfigurable.b)
        self.assertEqual(c.c, "no config")

    def test_override2(self):
        config = Config()
        config.Foo.a = 1
        config.Bar.b = "or"  # Up above b is config=False, so this won't do it.
        config.Bar.c = 10.0
        with expected_warnings(["`b` not recognized"]):
            c = Bar(config=config)
        self.assertEqual(c.a, config.Foo.a)
        self.assertEqual(c.b, "gotit")
        self.assertEqual(c.c, config.Bar.c)
        with expected_warnings(["`b` not recognized"]):
            c = Bar(a=2, b="and", c=20.0, config=config)
        self.assertEqual(c.a, 2)
        self.assertEqual(c.b, "and")
        self.assertEqual(c.c, 20.0)

    def test_help(self):
        self.assertEqual(MyConfigurable.class_get_help(), mc_help)
        self.assertEqual(Foo.class_get_help(), foo_help)
        self.assertEqual(Bar.class_get_help(), bar_help)

    def test_help_inst(self):
        inst = MyConfigurable(a=5, b=4)
        self.assertEqual(MyConfigurable.class_get_help(inst), mc_help_inst)

    def test_generated_config_enum_comments(self):
        class MyConf(Configurable):
            an_enum = Enum("Choice1 choice2".split(), help="Many choices.").tag(config=True)

        help_str = "Many choices."
        enum_choices_str = "Choices: any of ['Choice1', 'choice2']"
        rst_choices_str = "MyConf.an_enum : any of ``'Choice1'``|``'choice2'``"
        or_none_str = "or None"

        cls_help = MyConf.class_get_help()

        self.assertIn(help_str, cls_help)
        self.assertIn(enum_choices_str, cls_help)
        self.assertNotIn(or_none_str, cls_help)

        cls_cfg = MyConf.class_config_section()

        self.assertIn(help_str, cls_cfg)
        self.assertIn(enum_choices_str, cls_cfg)
        self.assertNotIn(or_none_str, cls_help)
        # Check order of Help-msg <--> Choices sections
        self.assertGreater(cls_cfg.index(enum_choices_str), cls_cfg.index(help_str))

        rst_help = MyConf.class_config_rst_doc()

        self.assertIn(help_str, rst_help)
        self.assertIn(rst_choices_str, rst_help)
        self.assertNotIn(or_none_str, rst_help)

        class MyConf2(Configurable):
            an_enum = Enum(
                "Choice1 choice2".split(),
                allow_none=True,
                default_value="choice2",
                help="Many choices.",
            ).tag(config=True)

        defaults_str = "Default: 'choice2'"

        cls2_msg = MyConf2.class_get_help()

        self.assertIn(help_str, cls2_msg)
        self.assertIn(enum_choices_str, cls2_msg)
        self.assertIn(or_none_str, cls2_msg)
        self.assertIn(defaults_str, cls2_msg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls2_msg.index(defaults_str), cls2_msg.index(enum_choices_str))

        cls2_cfg = MyConf2.class_config_section()

        self.assertIn(help_str, cls2_cfg)
        self.assertIn(enum_choices_str, cls2_cfg)
        self.assertIn(or_none_str, cls2_cfg)
        self.assertIn(defaults_str, cls2_cfg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls2_cfg.index(defaults_str), cls2_cfg.index(enum_choices_str))

    def test_generated_config_strenum_comments(self):
        help_str = "Many choices."
        defaults_str = "Default: 'choice2'"
        or_none_str = "or None"

        class MyConf3(Configurable):
            an_enum = CaselessStrEnum(
                "Choice1 choice2".split(),
                allow_none=True,
                default_value="choice2",
                help="Many choices.",
            ).tag(config=True)

        enum_choices_str = "Choices: any of ['Choice1', 'choice2'] (case-insensitive)"

        cls3_msg = MyConf3.class_get_help()

        self.assertIn(help_str, cls3_msg)
        self.assertIn(enum_choices_str, cls3_msg)
        self.assertIn(or_none_str, cls3_msg)
        self.assertIn(defaults_str, cls3_msg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls3_msg.index(defaults_str), cls3_msg.index(enum_choices_str))

        cls3_cfg = MyConf3.class_config_section()

        self.assertIn(help_str, cls3_cfg)
        self.assertIn(enum_choices_str, cls3_cfg)
        self.assertIn(or_none_str, cls3_cfg)
        self.assertIn(defaults_str, cls3_cfg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls3_cfg.index(defaults_str), cls3_cfg.index(enum_choices_str))

        class MyConf4(Configurable):
            an_enum = FuzzyEnum(
                "Choice1 choice2".split(),
                allow_none=True,
                default_value="choice2",
                help="Many choices.",
            ).tag(config=True)

        enum_choices_str = "Choices: any case-insensitive prefix of ['Choice1', 'choice2']"

        cls4_msg = MyConf4.class_get_help()

        self.assertIn(help_str, cls4_msg)
        self.assertIn(enum_choices_str, cls4_msg)
        self.assertIn(or_none_str, cls4_msg)
        self.assertIn(defaults_str, cls4_msg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls4_msg.index(defaults_str), cls4_msg.index(enum_choices_str))

        cls4_cfg = MyConf4.class_config_section()

        self.assertIn(help_str, cls4_cfg)
        self.assertIn(enum_choices_str, cls4_cfg)
        self.assertIn(or_none_str, cls4_cfg)
        self.assertIn(defaults_str, cls4_cfg)
        # Check order of Default <--> Choices sections
        self.assertGreater(cls4_cfg.index(defaults_str), cls4_cfg.index(enum_choices_str))


class TestSingletonConfigurable(TestCase):
    def test_instance(self):
        class Foo(SingletonConfigurable):
            pass

        self.assertEqual(Foo.initialized(), False)
        foo = Foo.instance()
        self.assertEqual(Foo.initialized(), True)
        self.assertEqual(foo, Foo.instance())
        self.assertEqual(SingletonConfigurable._instance, None)

    def test_inheritance(self):
        class Bar(SingletonConfigurable):
            pass

        class Bam(Bar):
            pass

        self.assertEqual(Bar.initialized(), False)
        self.assertEqual(Bam.initialized(), False)
        bam = Bam.instance()
        self.assertEqual(Bar.initialized(), True)
        self.assertEqual(Bam.initialized(), True)
        self.assertEqual(bam, Bam._instance)
        self.assertEqual(bam, Bar._instance)
        self.assertEqual(SingletonConfigurable._instance, None)


class TestLoggingConfigurable(TestCase):
    def test_parent_logger(self):
        class Parent(LoggingConfigurable):
            pass

        class Child(LoggingConfigurable):
            pass

        log = get_logger().getChild("TestLoggingConfigurable")

        parent = Parent(log=log)
        child = Child(parent=parent)
        self.assertEqual(parent.log, log)
        self.assertEqual(child.log, log)

        parent = Parent()
        child = Child(parent=parent, log=log)
        self.assertEqual(parent.log, get_logger())
        self.assertEqual(child.log, log)

    def test_parent_not_logging_configurable(self):
        class Parent(Configurable):
            pass

        class Child(LoggingConfigurable):
            pass

        parent = Parent()
        child = Child(parent=parent)
        self.assertEqual(child.log, get_logger())


class MyParent(Configurable):
    pass


class MyParent2(MyParent):
    pass


class TestParentConfigurable(TestCase):
    def test_parent_config(self):
        cfg = Config(
            {
                "MyParent": {
                    "MyConfigurable": {
                        "b": 2.0,
                    }
                }
            }
        )
        parent = MyParent(config=cfg)
        myc = MyConfigurable(parent=parent)
        self.assertEqual(myc.b, parent.config.MyParent.MyConfigurable.b)

    def test_parent_inheritance(self):
        cfg = Config(
            {
                "MyParent": {
                    "MyConfigurable": {
                        "b": 2.0,
                    }
                }
            }
        )
        parent = MyParent2(config=cfg)
        myc = MyConfigurable(parent=parent)
        self.assertEqual(myc.b, parent.config.MyParent.MyConfigurable.b)

    def test_multi_parent(self):
        cfg = Config(
            {
                "MyParent2": {
                    "MyParent": {
                        "MyConfigurable": {
                            "b": 2.0,
                        }
                    },
                    # this one shouldn't count
                    "MyConfigurable": {
                        "b": 3.0,
                    },
                }
            }
        )
        parent2 = MyParent2(config=cfg)
        parent = MyParent(parent=parent2)
        myc = MyConfigurable(parent=parent)
        self.assertEqual(myc.b, parent.config.MyParent2.MyParent.MyConfigurable.b)

    def test_parent_priority(self):
        cfg = Config(
            {
                "MyConfigurable": {
                    "b": 2.0,
                },
                "MyParent": {
                    "MyConfigurable": {
                        "b": 3.0,
                    }
                },
                "MyParent2": {
                    "MyConfigurable": {
                        "b": 4.0,
                    }
                },
            }
        )
        parent = MyParent2(config=cfg)
        myc = MyConfigurable(parent=parent)
        self.assertEqual(myc.b, parent.config.MyParent2.MyConfigurable.b)

    def test_multi_parent_priority(self):
        cfg = Config(
            {
                "MyConfigurable": {
                    "b": 2.0,
                },
                "MyParent": {
                    "MyConfigurable": {
                        "b": 3.0,
                    },
                },
                "MyParent2": {
                    "MyConfigurable": {
                        "b": 4.0,
                    },
                    "MyParent": {
                        "MyConfigurable": {
                            "b": 5.0,
                        },
                    },
                },
            }
        )
        parent2 = MyParent2(config=cfg)
        parent = MyParent2(parent=parent2)
        myc = MyConfigurable(parent=parent)
        self.assertEqual(myc.b, parent.config.MyParent2.MyParent.MyConfigurable.b)


class Containers(Configurable):
    lis = List().tag(config=True)

    def _lis_default(self):
        return [-1]

    s = Set().tag(config=True)

    def _s_default(self):
        return {"a"}

    d = Dict().tag(config=True)

    def _d_default(self):
        return {"a": "b"}


class TestConfigContainers(TestCase):
    def test_extend(self):
        c = Config()
        c.Containers.lis.extend(list(range(5)))
        obj = Containers(config=c)
        self.assertEqual(obj.lis, list(range(-1, 5)))

    def test_insert(self):
        c = Config()
        c.Containers.lis.insert(0, "a")
        c.Containers.lis.insert(1, "b")
        obj = Containers(config=c)
        self.assertEqual(obj.lis, ["a", "b", -1])

    def test_prepend(self):
        c = Config()
        c.Containers.lis.prepend([1, 2])
        c.Containers.lis.prepend([2, 3])
        obj = Containers(config=c)
        self.assertEqual(obj.lis, [2, 3, 1, 2, -1])

    def test_prepend_extend(self):
        c = Config()
        c.Containers.lis.prepend([1, 2])
        c.Containers.lis.extend([2, 3])
        obj = Containers(config=c)
        self.assertEqual(obj.lis, [1, 2, -1, 2, 3])

    def test_append_extend(self):
        c = Config()
        c.Containers.lis.append([1, 2])
        c.Containers.lis.extend([2, 3])
        obj = Containers(config=c)
        self.assertEqual(obj.lis, [-1, [1, 2], 2, 3])

    def test_extend_append(self):
        c = Config()
        c.Containers.lis.extend([2, 3])
        c.Containers.lis.append([1, 2])
        obj = Containers(config=c)
        self.assertEqual(obj.lis, [-1, 2, 3, [1, 2]])

    def test_insert_extend(self):
        c = Config()
        c.Containers.lis.insert(0, 1)
        c.Containers.lis.extend([2, 3])
        obj = Containers(config=c)
        self.assertEqual(obj.lis, [1, -1, 2, 3])

    def test_set_update(self):
        c = Config()
        c.Containers.s.update({0, 1, 2})
        c.Containers.s.update({3})
        obj = Containers(config=c)
        self.assertEqual(obj.s, {"a", 0, 1, 2, 3})

    def test_dict_update(self):
        c = Config()
        c.Containers.d.update({"c": "d"})
        c.Containers.d.update({"e": "f"})
        obj = Containers(config=c)
        self.assertEqual(obj.d, {"a": "b", "c": "d", "e": "f"})

    def test_update_twice(self):
        c = Config()
        c.MyConfigurable.a = 5
        m = MyConfigurable(config=c)
        self.assertEqual(m.a, 5)

        c2 = Config()
        c2.MyConfigurable.a = 10
        m.update_config(c2)
        self.assertEqual(m.a, 10)

        c2.MyConfigurable.a = 15
        m.update_config(c2)
        self.assertEqual(m.a, 15)

    def test_update_self(self):
        """update_config with same config object still triggers config_changed"""
        c = Config()
        c.MyConfigurable.a = 5
        m = MyConfigurable(config=c)
        self.assertEqual(m.a, 5)
        c.MyConfigurable.a = 10
        m.update_config(c)
        self.assertEqual(m.a, 10)

    def test_config_default(self):
        class SomeSingleton(SingletonConfigurable):
            pass

        class DefaultConfigurable(Configurable):
            a = Integer().tag(config=True)

            def _config_default(self):
                if SomeSingleton.initialized():
                    return SomeSingleton.instance().config
                return Config()

        c = Config()
        c.DefaultConfigurable.a = 5

        d1 = DefaultConfigurable()
        self.assertEqual(d1.a, 0)

        single = SomeSingleton.instance(config=c)

        d2 = DefaultConfigurable()
        self.assertIs(d2.config, single.config)
        self.assertEqual(d2.a, 5)

    def test_config_default_deprecated(self):
        """Make sure configurables work even with the deprecations in traitlets"""

        class SomeSingleton(SingletonConfigurable):
            pass

        # reset deprecation limiter
        _deprecations_shown.clear()
        with expected_warnings([]):

            class DefaultConfigurable(Configurable):
                a = Integer(config=True)

                def _config_default(self):
                    if SomeSingleton.initialized():
                        return SomeSingleton.instance().config
                    return Config()

        c = Config()
        c.DefaultConfigurable.a = 5

        d1 = DefaultConfigurable()
        self.assertEqual(d1.a, 0)

        single = SomeSingleton.instance(config=c)

        d2 = DefaultConfigurable()
        self.assertIs(d2.config, single.config)
        self.assertEqual(d2.a, 5)

    def test_kwarg_config_priority(self):
        # a, c set in kwargs
        # a, b set in config
        # verify that:
        # - kwargs are set before config
        # - kwargs have priority over config
        class A(Configurable):
            a = Unicode("default", config=True)
            b = Unicode("default", config=True)
            c = Unicode("default", config=True)
            c_during_config = Unicode("never")

            @validate("b")
            def _record_c(self, proposal):
                # setting b from config records c's value at the time
                self.c_during_config = self.c
                return proposal.value

        cfg = Config()
        cfg.A.a = "a-config"
        cfg.A.b = "b-config"
        obj = A(a="a-kwarg", c="c-kwarg", config=cfg)
        assert obj.a == "a-kwarg"
        assert obj.b == "b-config"
        assert obj.c == "c-kwarg"
        assert obj.c_during_config == "c-kwarg"


class TestLogger(TestCase):
    class A(LoggingConfigurable):
        foo = Integer(config=True)
        bar = Integer(config=True)
        baz = Integer(config=True)

    @mark.skipif(not hasattr(TestCase, "assertLogs"), reason="requires TestCase.assertLogs")
    def test_warn_match(self):
        logger = logging.getLogger("test_warn_match")
        cfg = Config({"A": {"bat": 5}})
        with self.assertLogs(logger, logging.WARNING) as captured:
            TestLogger.A(config=cfg, log=logger)

        output = "\n".join(captured.output)
        self.assertIn("Did you mean one of: `bar, baz`?", output)
        self.assertIn("Config option `bat` not recognized by `A`.", output)

        cfg = Config({"A": {"fool": 5}})
        with self.assertLogs(logger, logging.WARNING) as captured:
            TestLogger.A(config=cfg, log=logger)

        output = "\n".join(captured.output)
        self.assertIn("Config option `fool` not recognized by `A`.", output)
        self.assertIn("Did you mean `foo`?", output)

        cfg = Config({"A": {"totally_wrong": 5}})
        with self.assertLogs(logger, logging.WARNING) as captured:
            TestLogger.A(config=cfg, log=logger)

        output = "\n".join(captured.output)
        self.assertIn("Config option `totally_wrong` not recognized by `A`.", output)
        self.assertNotIn("Did you mean", output)


def test_logger_adapter(caplog, capsys):
    logger = logging.getLogger("Application")
    adapter = logging.LoggerAdapter(logger, {"key": "adapted"})

    app = Application(log=adapter, log_level=logging.INFO)
    app.log_format = "%(key)s %(message)s"
    app.log.info("test message")

    assert "adapted test message" in capsys.readouterr().err
