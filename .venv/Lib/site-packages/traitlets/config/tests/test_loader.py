"""Tests for traitlets.config.loader"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import copy
import os
import pickle
from itertools import chain
from tempfile import mkstemp
from unittest import TestCase

import pytest

from traitlets import Dict, Integer, List, Tuple, Unicode
from traitlets.config import Configurable
from traitlets.config.loader import (
    ArgParseConfigLoader,
    Config,
    JSONFileConfigLoader,
    KeyValueConfigLoader,
    KVArgParseConfigLoader,
    LazyConfigValue,
    PyFileConfigLoader,
)

pyfile = """
c = get_config()
c.a=10
c.b=20
c.Foo.Bar.value=10
c.Foo.Bam.value=list(range(10))
c.D.C.value='hi there'
"""

json1file = """
{
  "version": 1,
  "a": 10,
  "b": 20,
  "Foo": {
    "Bam": {
      "value": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
    },
    "Bar": {
      "value": 10
    }
  },
  "D": {
    "C": {
      "value": "hi there"
    }
  }
}
"""

# should not load
json2file = """
{
  "version": 2
}
"""

import logging

log = logging.getLogger("devnull")
log.setLevel(0)


class TestFileCL(TestCase):
    def _check_conf(self, config):
        self.assertEqual(config.a, 10)
        self.assertEqual(config.b, 20)
        self.assertEqual(config.Foo.Bar.value, 10)
        self.assertEqual(config.Foo.Bam.value, list(range(10)))
        self.assertEqual(config.D.C.value, "hi there")

    def test_python(self):
        fd, fname = mkstemp(".py", prefix="μnïcø∂e")
        f = os.fdopen(fd, "w")
        f.write(pyfile)
        f.close()
        # Unlink the file
        cl = PyFileConfigLoader(fname, log=log)
        config = cl.load_config()
        self._check_conf(config)

    def test_json(self):
        fd, fname = mkstemp(".json", prefix="μnïcø∂e")
        f = os.fdopen(fd, "w")
        f.write(json1file)
        f.close()
        # Unlink the file
        cl = JSONFileConfigLoader(fname, log=log)
        config = cl.load_config()
        self._check_conf(config)

    def test_context_manager(self):

        fd, fname = mkstemp(".json", prefix="μnïcø∂e")
        f = os.fdopen(fd, "w")
        f.write("{}")
        f.close()

        cl = JSONFileConfigLoader(fname, log=log)

        value = "context_manager"

        with cl as c:
            c.MyAttr.value = value

        self.assertEqual(cl.config.MyAttr.value, value)

        # check that another loader does see the change
        _ = JSONFileConfigLoader(fname, log=log)
        self.assertEqual(cl.config.MyAttr.value, value)

    def test_json_context_bad_write(self):
        fd, fname = mkstemp(".json", prefix="μnïcø∂e")
        f = os.fdopen(fd, "w")
        f.write("{}")
        f.close()

        with JSONFileConfigLoader(fname, log=log) as config:
            config.A.b = 1

        with self.assertRaises(TypeError):
            with JSONFileConfigLoader(fname, log=log) as config:
                config.A.cant_json = lambda x: x

        loader = JSONFileConfigLoader(fname, log=log)
        cfg = loader.load_config()
        assert cfg.A.b == 1
        assert "cant_json" not in cfg.A

    def test_collision(self):
        a = Config()
        b = Config()
        self.assertEqual(a.collisions(b), {})
        a.A.trait1 = 1
        b.A.trait2 = 2
        self.assertEqual(a.collisions(b), {})
        b.A.trait1 = 1
        self.assertEqual(a.collisions(b), {})
        b.A.trait1 = 0
        self.assertEqual(
            a.collisions(b),
            {
                "A": {
                    "trait1": "1 ignored, using 0",
                }
            },
        )
        self.assertEqual(
            b.collisions(a),
            {
                "A": {
                    "trait1": "0 ignored, using 1",
                }
            },
        )
        a.A.trait2 = 3
        self.assertEqual(
            b.collisions(a),
            {
                "A": {
                    "trait1": "0 ignored, using 1",
                    "trait2": "2 ignored, using 3",
                }
            },
        )

    def test_v2raise(self):
        fd, fname = mkstemp(".json", prefix="μnïcø∂e")
        f = os.fdopen(fd, "w")
        f.write(json2file)
        f.close()
        # Unlink the file
        cl = JSONFileConfigLoader(fname, log=log)
        with self.assertRaises(ValueError):
            cl.load_config()


def _parse_int_or_str(v):
    try:
        return int(v)
    except Exception:
        return str(v)


class MyLoader1(ArgParseConfigLoader):
    def _add_arguments(self, aliases=None, flags=None, classes=None):
        p = self.parser
        p.add_argument("-f", "--foo", dest="Global.foo", type=str)
        p.add_argument("-b", dest="MyClass.bar", type=int)
        p.add_argument("-n", dest="n", action="store_true")
        p.add_argument("Global.bam", type=str)
        p.add_argument("--list1", action="append", type=_parse_int_or_str)
        p.add_argument("--list2", nargs="+", type=int)


class MyLoader2(ArgParseConfigLoader):
    def _add_arguments(self, aliases=None, flags=None, classes=None):
        subparsers = self.parser.add_subparsers(dest="subparser_name")
        subparser1 = subparsers.add_parser("1")
        subparser1.add_argument("-x", dest="Global.x")
        subparser2 = subparsers.add_parser("2")
        subparser2.add_argument("y")


class TestArgParseCL(TestCase):
    def test_basic(self):
        cl = MyLoader1()
        config = cl.load_config("-f hi -b 10 -n wow".split())
        self.assertEqual(config.Global.foo, "hi")
        self.assertEqual(config.MyClass.bar, 10)
        self.assertEqual(config.n, True)
        self.assertEqual(config.Global.bam, "wow")
        config = cl.load_config(["wow"])
        self.assertEqual(list(config.keys()), ["Global"])
        self.assertEqual(list(config.Global.keys()), ["bam"])
        self.assertEqual(config.Global.bam, "wow")

    def test_add_arguments(self):
        cl = MyLoader2()
        config = cl.load_config("2 frobble".split())
        self.assertEqual(config.subparser_name, "2")
        self.assertEqual(config.y, "frobble")
        config = cl.load_config("1 -x frobble".split())
        self.assertEqual(config.subparser_name, "1")
        self.assertEqual(config.Global.x, "frobble")

    def test_argv(self):
        cl = MyLoader1(argv="-f hi -b 10 -n wow".split())
        config = cl.load_config()
        self.assertEqual(config.Global.foo, "hi")
        self.assertEqual(config.MyClass.bar, 10)
        self.assertEqual(config.n, True)
        self.assertEqual(config.Global.bam, "wow")

    def test_list_args(self):
        cl = MyLoader1()
        config = cl.load_config("--list1 1 wow --list2 1 2 3 --list1 B".split())
        self.assertEqual(list(config.Global.keys()), ["bam"])
        self.assertEqual(config.Global.bam, "wow")
        self.assertEqual(config.list1, [1, "B"])
        self.assertEqual(config.list2, [1, 2, 3])


class C(Configurable):
    str_trait = Unicode(config=True)
    int_trait = Integer(config=True)
    list_trait = List(config=True)
    list_of_ints = List(Integer(), config=True)
    dict_trait = Dict(config=True)
    dict_of_ints = Dict(
        key_trait=Integer(),
        value_trait=Integer(),
        config=True,
    )
    dict_multi = Dict(
        key_trait=Unicode(),
        per_key_traits={
            "int": Integer(),
            "str": Unicode(),
        },
        config=True,
    )


class TestKeyValueCL(TestCase):
    klass = KeyValueConfigLoader

    def test_eval(self):
        cl = self.klass(log=log)
        config = cl.load_config(
            '--C.str_trait=all --C.int_trait=5 --C.list_trait=["hello",5]'.split()
        )
        c = C(config=config)
        assert c.str_trait == "all"
        assert c.int_trait == 5
        assert c.list_trait == ["hello", 5]

    def test_basic(self):
        cl = self.klass(log=log)
        argv = ["--" + s[2:] for s in pyfile.split("\n") if s.startswith("c.")]
        config = cl.load_config(argv)
        assert config.a == "10"
        assert config.b == "20"
        assert config.Foo.Bar.value == "10"
        # non-literal expressions are not evaluated
        self.assertEqual(config.Foo.Bam.value, "list(range(10))")
        self.assertEqual(Unicode().from_string(config.D.C.value), "hi there")

    def test_expanduser(self):
        cl = self.klass(log=log)
        argv = ["--a=~/1/2/3", "--b=~", "--c=~/", '--d="~/"']
        config = cl.load_config(argv)
        u = Unicode()
        self.assertEqual(u.from_string(config.a), os.path.expanduser("~/1/2/3"))
        self.assertEqual(u.from_string(config.b), os.path.expanduser("~"))
        self.assertEqual(u.from_string(config.c), os.path.expanduser("~/"))
        self.assertEqual(u.from_string(config.d), "~/")

    def test_extra_args(self):
        cl = self.klass(log=log)
        config = cl.load_config(["--a=5", "b", "d", "--c=10"])
        self.assertEqual(cl.extra_args, ["b", "d"])
        assert config.a == "5"
        assert config.c == "10"
        config = cl.load_config(["--", "--a=5", "--c=10"])
        self.assertEqual(cl.extra_args, ["--a=5", "--c=10"])

        cl = self.klass(log=log)
        config = cl.load_config(["extra", "--a=2", "--c=1", "--", "-"])
        self.assertEqual(cl.extra_args, ["extra", "-"])

    def test_unicode_args(self):
        cl = self.klass(log=log)
        argv = ["--a=épsîlön"]
        config = cl.load_config(argv)
        print(config, cl.extra_args)
        self.assertEqual(config.a, "épsîlön")

    def test_list_append(self):
        cl = self.klass(log=log)
        argv = ["--C.list_trait", "x", "--C.list_trait", "y"]
        config = cl.load_config(argv)
        assert config.C.list_trait == ["x", "y"]
        c = C(config=config)
        assert c.list_trait == ["x", "y"]

    def test_list_single_item(self):
        cl = self.klass(log=log)
        argv = ["--C.list_trait", "x"]
        config = cl.load_config(argv)
        c = C(config=config)
        assert c.list_trait == ["x"]

    def test_dict(self):
        cl = self.klass(log=log)
        argv = ["--C.dict_trait", "x=5", "--C.dict_trait", "y=10"]
        config = cl.load_config(argv)
        c = C(config=config)
        assert c.dict_trait == {"x": "5", "y": "10"}

    def test_dict_key_traits(self):
        cl = self.klass(log=log)
        argv = ["--C.dict_of_ints", "1=2", "--C.dict_of_ints", "3=4"]
        config = cl.load_config(argv)
        c = C(config=config)
        assert c.dict_of_ints == {1: 2, 3: 4}


class CBase(Configurable):
    a = List().tag(config=True)
    b = List(Integer()).tag(config=True, multiplicity="*")
    c = List().tag(config=True, multiplicity="append")
    adict = Dict().tag(config=True)


class CSub(CBase):
    d = Tuple().tag(config=True)
    e = Tuple().tag(config=True, multiplicity="+")
    bdict = Dict().tag(config=True, multiplicity="*")


class TestArgParseKVCL(TestKeyValueCL):
    klass = KVArgParseConfigLoader  # type:ignore

    def test_no_cast_literals(self):
        cl = self.klass(log=log)  # type:ignore
        # test ipython -c 1 doesn't cast to int
        argv = ["-c", "1"]
        config = cl.load_config(argv, aliases=dict(c="IPython.command_to_run"))
        assert config.IPython.command_to_run == "1"

    def test_int_literals(self):
        cl = self.klass(log=log)  # type:ignore
        # test ipython -c 1 doesn't cast to int
        argv = ["-c", "1"]
        config = cl.load_config(argv, aliases=dict(c="IPython.command_to_run"))
        assert config.IPython.command_to_run == "1"

    def test_unicode_alias(self):
        cl = self.klass(log=log)  # type:ignore
        argv = ["--a=épsîlön"]
        config = cl.load_config(argv, aliases=dict(a="A.a"))
        print(dict(config))
        print(cl.extra_args)
        print(cl.aliases)
        self.assertEqual(config.A.a, "épsîlön")

    def test_expanduser2(self):
        cl = self.klass(log=log)  # type:ignore
        argv = ["-a", "~/1/2/3", "--b", "'~/1/2/3'"]
        config = cl.load_config(argv, aliases=dict(a="A.a", b="A.b"))

        class A(Configurable):
            a = Unicode(config=True)
            b = Unicode(config=True)

        a = A(config=config)
        self.assertEqual(a.a, os.path.expanduser("~/1/2/3"))
        self.assertEqual(a.b, "~/1/2/3")

    def test_eval(self):
        cl = self.klass(log=log)  # type:ignore
        argv = ["-c", "a=5"]
        config = cl.load_config(argv, aliases=dict(c="A.c"))
        self.assertEqual(config.A.c, "a=5")

    def test_seq_traits(self):
        cl = self.klass(log=log, classes=(CBase, CSub))  # type:ignore
        aliases = {"a3": "CBase.c", "a5": "CSub.e"}
        argv = (
            "--CBase.a A --CBase.a 2 --CBase.b 1 --CBase.b 3 --a3 AA --CBase.c BB "
            "--CSub.d 1 --CSub.d BBB --CSub.e 1 --CSub.e=bcd a b c "
        ).split()
        config = cl.load_config(argv, aliases=aliases)
        assert cl.extra_args == ["a", "b", "c"]
        assert config.CBase.a == ["A", "2"]
        assert config.CBase.b == [1, 3]
        self.assertEqual(config.CBase.c, ["AA", "BB"])

        assert config.CSub.d == ("1", "BBB")
        assert config.CSub.e == ("1", "bcd")

    def test_seq_traits_single_empty_string(self):
        cl = self.klass(log=log, classes=(CBase,))  # type:ignore
        aliases = {"seqopt": "CBase.c"}
        argv = ["--seqopt", ""]
        config = cl.load_config(argv, aliases=aliases)
        self.assertEqual(config.CBase.c, [""])

    def test_dict_traits(self):
        cl = self.klass(log=log, classes=(CBase, CSub))  # type:ignore
        aliases = {"D": "CBase.adict", "E": "CSub.bdict"}
        argv = ["-D", "k1=v1", "-D=k2=2", "-D", "k3=v 3", "-E", "k=v", "-E", "22=222"]
        config = cl.load_config(argv, aliases=aliases)
        c = CSub(config=config)
        assert c.adict == {"k1": "v1", "k2": "2", "k3": "v 3"}
        assert c.bdict == {"k": "v", "22": "222"}

    def test_mixed_seq_positional(self):
        aliases = {"c": "Class.trait"}
        cl = self.klass(log=log, aliases=aliases)  # type:ignore
        assignments = [("-c", "1"), ("--Class.trait=2",), ("--c=3",), ("--Class.trait", "4")]
        positionals = ["a", "b", "c"]
        # test with positionals at any index
        for idx in range(len(assignments) + 1):
            argv_parts = assignments[:]
            argv_parts[idx:idx] = (positionals,)  # type:ignore
            argv = list(chain(*argv_parts))

            config = cl.load_config(argv)
            assert config.Class.trait == ["1", "2", "3", "4"]
            assert cl.extra_args == ["a", "b", "c"]

    def test_split_positional(self):
        """Splitting positionals across flags is no longer allowed in traitlets 5"""
        cl = self.klass(log=log)  # type:ignore
        argv = ["a", "--Class.trait=5", "b"]
        with pytest.raises(SystemExit):
            cl.load_config(argv)


class TestConfig(TestCase):
    def test_setget(self):
        c = Config()
        c.a = 10
        self.assertEqual(c.a, 10)
        self.assertEqual("b" in c, False)

    def test_auto_section(self):
        c = Config()
        self.assertNotIn("A", c)
        assert not c._has_section("A")
        A = c.A
        A.foo = "hi there"
        self.assertIn("A", c)
        assert c._has_section("A")
        self.assertEqual(c.A.foo, "hi there")
        del c.A
        self.assertEqual(c.A, Config())

    def test_merge_doesnt_exist(self):
        c1 = Config()
        c2 = Config()
        c2.bar = 10
        c2.Foo.bar = 10
        c1.merge(c2)
        self.assertEqual(c1.Foo.bar, 10)
        self.assertEqual(c1.bar, 10)
        c2.Bar.bar = 10
        c1.merge(c2)
        self.assertEqual(c1.Bar.bar, 10)

    def test_merge_exists(self):
        c1 = Config()
        c2 = Config()
        c1.Foo.bar = 10
        c1.Foo.bam = 30
        c2.Foo.bar = 20
        c2.Foo.wow = 40
        c1.merge(c2)
        self.assertEqual(c1.Foo.bam, 30)
        self.assertEqual(c1.Foo.bar, 20)
        self.assertEqual(c1.Foo.wow, 40)
        c2.Foo.Bam.bam = 10
        c1.merge(c2)
        self.assertEqual(c1.Foo.Bam.bam, 10)

    def test_deepcopy(self):
        c1 = Config()
        c1.Foo.bar = 10
        c1.Foo.bam = 30
        c1.a = "asdf"
        c1.b = range(10)
        c1.Test.logger = logging.Logger("test")
        c1.Test.get_logger = logging.getLogger("test")
        c2 = copy.deepcopy(c1)
        self.assertEqual(c1, c2)
        self.assertTrue(c1 is not c2)
        self.assertTrue(c1.Foo is not c2.Foo)
        self.assertTrue(c1.Test is not c2.Test)
        self.assertTrue(c1.Test.logger is c2.Test.logger)
        self.assertTrue(c1.Test.get_logger is c2.Test.get_logger)

    def test_builtin(self):
        c1 = Config()
        c1.format = "json"

    def test_fromdict(self):
        c1 = Config({"Foo": {"bar": 1}})
        self.assertEqual(c1.Foo.__class__, Config)
        self.assertEqual(c1.Foo.bar, 1)

    def test_fromdictmerge(self):
        c1 = Config()
        c2 = Config({"Foo": {"bar": 1}})
        c1.merge(c2)
        self.assertEqual(c1.Foo.__class__, Config)
        self.assertEqual(c1.Foo.bar, 1)

    def test_fromdictmerge2(self):
        c1 = Config({"Foo": {"baz": 2}})
        c2 = Config({"Foo": {"bar": 1}})
        c1.merge(c2)
        self.assertEqual(c1.Foo.__class__, Config)
        self.assertEqual(c1.Foo.bar, 1)
        self.assertEqual(c1.Foo.baz, 2)
        self.assertNotIn("baz", c2.Foo)

    def test_contains(self):
        c1 = Config({"Foo": {"baz": 2}})
        c2 = Config({"Foo": {"bar": 1}})
        self.assertIn("Foo", c1)
        self.assertIn("Foo.baz", c1)
        self.assertIn("Foo.bar", c2)
        self.assertNotIn("Foo.bar", c1)

    def test_pickle_config(self):
        cfg = Config()
        cfg.Foo.bar = 1
        pcfg = pickle.dumps(cfg)
        cfg2 = pickle.loads(pcfg)
        self.assertEqual(cfg2, cfg)

    def test_getattr_section(self):
        cfg = Config()
        self.assertNotIn("Foo", cfg)
        Foo = cfg.Foo
        assert isinstance(Foo, Config)
        self.assertIn("Foo", cfg)

    def test_getitem_section(self):
        cfg = Config()
        self.assertNotIn("Foo", cfg)
        Foo = cfg["Foo"]
        assert isinstance(Foo, Config)
        self.assertIn("Foo", cfg)

    def test_getattr_not_section(self):
        cfg = Config()
        self.assertNotIn("foo", cfg)
        foo = cfg.foo
        assert isinstance(foo, LazyConfigValue)
        self.assertIn("foo", cfg)

    def test_getattr_private_missing(self):
        cfg = Config()
        self.assertNotIn("_repr_html_", cfg)
        with self.assertRaises(AttributeError):
            _ = cfg._repr_html_
        self.assertNotIn("_repr_html_", cfg)
        self.assertEqual(len(cfg), 0)

    def test_lazy_config_repr(self):
        cfg = Config()
        cfg.Class.lazy.append(1)
        cfg_repr = repr(cfg)
        assert "<LazyConfigValue" in cfg_repr
        assert "extend" in cfg_repr
        assert " [1]}>" in cfg_repr
        assert "value=" not in cfg_repr
        cfg.Class.lazy.get_value([0])
        repr2 = repr(cfg)
        assert repr([0, 1]) in repr2
        assert "value=" in repr2

    def test_getitem_not_section(self):
        cfg = Config()
        self.assertNotIn("foo", cfg)
        foo = cfg["foo"]
        assert isinstance(foo, LazyConfigValue)
        self.assertIn("foo", cfg)

    def test_merge_no_copies(self):
        c = Config()
        c2 = Config()
        c2.Foo.trait = []
        c.merge(c2)
        c2.Foo.trait.append(1)
        self.assertIs(c.Foo, c2.Foo)
        self.assertEqual(c.Foo.trait, [1])
        self.assertEqual(c2.Foo.trait, [1])

    def test_merge_multi_lazy(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        If systemwide overwirte and user append, we want both in the right
        order.
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait = [1]
        c2.Foo.trait.append(2)

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait, [1, 2])

    def test_merge_multi_lazyII(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        If both are lazy we still want a lazy config.
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait.append(1)
        c2.Foo.trait.append(2)

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait._extend, [1, 2])

    def test_merge_multi_lazy_III(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        Prepend should prepend in the right order.
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait = [1]
        c2.Foo.trait.prepend([0])

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait, [0, 1])

    def test_merge_multi_lazy_IV(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        Both prepending should be lazy
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait.prepend([1])
        c2.Foo.trait.prepend([0])

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait._prepend, [0, 1])

    def test_merge_multi_lazy_update_I(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        dict update shoudl be in the right order.
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait = {"a": 1, "z": 26}
        c2.Foo.trait.update({"a": 0, "b": 1})

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait, {"a": 0, "b": 1, "z": 26})

    def test_merge_multi_lazy_update_II(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        Later dict overwrite lazyness
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait.update({"a": 0, "b": 1})
        c2.Foo.trait = {"a": 1, "z": 26}

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait, {"a": 1, "z": 26})

    def test_merge_multi_lazy_update_III(self):
        """
        With multiple config files (systemwide and users), we want compounding.

        Later dict overwrite lazyness
        """
        c1 = Config()
        c2 = Config()

        c1.Foo.trait.update({"a": 0, "b": 1})
        c2.Foo.trait.update({"a": 1, "z": 26})

        c = Config()
        c.merge(c1)
        c.merge(c2)

        self.assertEqual(c.Foo.trait._update, {"a": 1, "z": 26, "b": 1})
