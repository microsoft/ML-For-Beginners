# coding: utf-8
"""Tests for IPython.lib.pretty."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest

import pytest

from IPython.lib import pretty

from io import StringIO


class MyList(object):
    def __init__(self, content):
        self.content = content
    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MyList(...)")
        else:
            with p.group(3, "MyList(", ")"):
                for (i, child) in enumerate(self.content):
                    if i:
                        p.text(",")
                        p.breakable()
                    else:
                        p.breakable("")
                    p.pretty(child)


class MyDict(dict):
    def _repr_pretty_(self, p, cycle):
        p.text("MyDict(...)")

class MyObj(object):
    def somemethod(self):
        pass


class Dummy1(object):
    def _repr_pretty_(self, p, cycle):
        p.text("Dummy1(...)")

class Dummy2(Dummy1):
    _repr_pretty_ = None

class NoModule(object):
    pass

NoModule.__module__ = None

class Breaking(object):
    def _repr_pretty_(self, p, cycle):
        with p.group(4,"TG: ",":"):
            p.text("Breaking(")
            p.break_()
            p.text(")")

class BreakingRepr(object):
    def __repr__(self):
        return "Breaking(\n)"

class BadRepr(object):
    def __repr__(self):
        return 1/0


def test_indentation():
    """Test correct indentation in groups"""
    count = 40
    gotoutput = pretty.pretty(MyList(range(count)))
    expectedoutput = "MyList(\n" + ",\n".join("   %d" % i for i in range(count)) + ")"

    assert gotoutput == expectedoutput


def test_dispatch():
    """
    Test correct dispatching: The _repr_pretty_ method for MyDict
    must be found before the registered printer for dict.
    """
    gotoutput = pretty.pretty(MyDict())
    expectedoutput = "MyDict(...)"

    assert gotoutput == expectedoutput


def test_callability_checking():
    """
    Test that the _repr_pretty_ method is tested for callability and skipped if
    not.
    """
    gotoutput = pretty.pretty(Dummy2())
    expectedoutput = "Dummy1(...)"

    assert gotoutput == expectedoutput


@pytest.mark.parametrize(
    "obj,expected_output",
    zip(
        [
            set(),
            frozenset(),
            set([1]),
            frozenset([1]),
            set([1, 2]),
            frozenset([1, 2]),
            set([-1, -2, -3]),
        ],
        [
            "set()",
            "frozenset()",
            "{1}",
            "frozenset({1})",
            "{1, 2}",
            "frozenset({1, 2})",
            "{-3, -2, -1}",
        ],
    ),
)
def test_sets(obj, expected_output):
    """
    Test that set and frozenset use Python 3 formatting.
    """
    got_output = pretty.pretty(obj)
    assert got_output == expected_output


def test_pprint_heap_allocated_type():
    """
    Test that pprint works for heap allocated types.
    """
    module_name = "xxlimited" if sys.version_info < (3, 10) else "xxlimited_35"
    expected_output = (
        "xxlimited.Null" if sys.version_info < (3, 10, 6) else "xxlimited_35.Null"
    )
    xxlimited = pytest.importorskip(module_name)
    output = pretty.pretty(xxlimited.Null)
    assert output == expected_output


def test_pprint_nomod():
    """
    Test that pprint works for classes with no __module__.
    """
    output = pretty.pretty(NoModule)
    assert output == "NoModule"


def test_pprint_break():
    """
    Test that p.break_ produces expected output
    """
    output = pretty.pretty(Breaking())
    expected = "TG: Breaking(\n    ):"
    assert output == expected

def test_pprint_break_repr():
    """
    Test that p.break_ is used in repr
    """
    output = pretty.pretty([[BreakingRepr()]])
    expected = "[[Breaking(\n  )]]"
    assert output == expected

    output = pretty.pretty([[BreakingRepr()]*2])
    expected = "[[Breaking(\n  ),\n  Breaking(\n  )]]"
    assert output == expected

def test_bad_repr():
    """Don't catch bad repr errors"""
    with pytest.raises(ZeroDivisionError):
        pretty.pretty(BadRepr())

class BadException(Exception):
    def __str__(self):
        return -1

class ReallyBadRepr(object):
    __module__ = 1
    @property
    def __class__(self):
        raise ValueError("I am horrible")

    def __repr__(self):
        raise BadException()

def test_really_bad_repr():
    with pytest.raises(BadException):
        pretty.pretty(ReallyBadRepr())


class SA(object):
    pass

class SB(SA):
    pass

class TestsPretty(unittest.TestCase):

    def test_super_repr(self):
        # "<super: module_name.SA, None>"
        output = pretty.pretty(super(SA))
        self.assertRegex(output, r"<super: \S+.SA, None>")

        # "<super: module_name.SA, <module_name.SB at 0x...>>"
        sb = SB()
        output = pretty.pretty(super(SA, sb))
        self.assertRegex(output, r"<super: \S+.SA,\s+<\S+.SB at 0x\S+>>")


    def test_long_list(self):
        lis = list(range(10000))
        p = pretty.pretty(lis)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...]'])

    def test_long_set(self):
        s = set(range(10000))
        p = pretty.pretty(s)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...}'])

    def test_long_tuple(self):
        tup = tuple(range(10000))
        p = pretty.pretty(tup)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999,', ' ...)'])

    def test_long_dict(self):
        d = { n:n for n in range(10000) }
        p = pretty.pretty(d)
        last2 = p.rsplit('\n', 2)[-2:]
        self.assertEqual(last2, [' 999: 999,', ' ...}'])

    def test_unbound_method(self):
        output = pretty.pretty(MyObj.somemethod)
        self.assertIn('MyObj.somemethod', output)


class MetaClass(type):
    def __new__(cls, name):
        return type.__new__(cls, name, (object,), {'name': name})

    def __repr__(self):
        return "[CUSTOM REPR FOR CLASS %s]" % self.name


ClassWithMeta = MetaClass('ClassWithMeta')


def test_metaclass_repr():
    output = pretty.pretty(ClassWithMeta)
    assert output == "[CUSTOM REPR FOR CLASS ClassWithMeta]"


def test_unicode_repr():
    u = u"üniçodé"
    ustr = u

    class C(object):
        def __repr__(self):
            return ustr

    c = C()
    p = pretty.pretty(c)
    assert p == u
    p = pretty.pretty([c])
    assert p == "[%s]" % u


def test_basic_class():
    def type_pprint_wrapper(obj, p, cycle):
        if obj is MyObj:
            type_pprint_wrapper.called = True
        return pretty._type_pprint(obj, p, cycle)
    type_pprint_wrapper.called = False

    stream = StringIO()
    printer = pretty.RepresentationPrinter(stream)
    printer.type_pprinters[type] = type_pprint_wrapper
    printer.pretty(MyObj)
    printer.flush()
    output = stream.getvalue()

    assert output == "%s.MyObj" % __name__
    assert type_pprint_wrapper.called is True


def test_collections_userlist():
    # Create userlist with cycle
    a = UserList()
    a.append(a)

    cases = [
        (UserList(), "UserList([])"),
        (
            UserList(i for i in range(1000, 1020)),
            "UserList([1000,\n"
            "          1001,\n"
            "          1002,\n"
            "          1003,\n"
            "          1004,\n"
            "          1005,\n"
            "          1006,\n"
            "          1007,\n"
            "          1008,\n"
            "          1009,\n"
            "          1010,\n"
            "          1011,\n"
            "          1012,\n"
            "          1013,\n"
            "          1014,\n"
            "          1015,\n"
            "          1016,\n"
            "          1017,\n"
            "          1018,\n"
            "          1019])",
        ),
        (a, "UserList([UserList(...)])"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


# TODO : pytest.mark.parametrise once nose is gone.
def test_collections_defaultdict():
    # Create defaultdicts with cycles
    a = defaultdict()
    a.default_factory = a
    b = defaultdict(list)
    b['key'] = b

    # Dictionary order cannot be relied on, test against single keys.
    cases = [
        (defaultdict(list), 'defaultdict(list, {})'),
        (defaultdict(list, {'key': '-' * 50}),
         "defaultdict(list,\n"
         "            {'key': '--------------------------------------------------'})"),
        (a, 'defaultdict(defaultdict(...), {})'),
        (b, "defaultdict(list, {'key': defaultdict(...)})"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


# TODO : pytest.mark.parametrise once nose is gone.
def test_collections_ordereddict():
    # Create OrderedDict with cycle
    a = OrderedDict()
    a['key'] = a

    cases = [
        (OrderedDict(), 'OrderedDict()'),
        (OrderedDict((i, i) for i in range(1000, 1010)),
         'OrderedDict([(1000, 1000),\n'
         '             (1001, 1001),\n'
         '             (1002, 1002),\n'
         '             (1003, 1003),\n'
         '             (1004, 1004),\n'
         '             (1005, 1005),\n'
         '             (1006, 1006),\n'
         '             (1007, 1007),\n'
         '             (1008, 1008),\n'
         '             (1009, 1009)])'),
        (a, "OrderedDict([('key', OrderedDict(...))])"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


# TODO : pytest.mark.parametrise once nose is gone.
def test_collections_deque():
    # Create deque with cycle
    a = deque()
    a.append(a)

    cases = [
        (deque(), 'deque([])'),
        (deque(i for i in range(1000, 1020)),
         'deque([1000,\n'
         '       1001,\n'
         '       1002,\n'
         '       1003,\n'
         '       1004,\n'
         '       1005,\n'
         '       1006,\n'
         '       1007,\n'
         '       1008,\n'
         '       1009,\n'
         '       1010,\n'
         '       1011,\n'
         '       1012,\n'
         '       1013,\n'
         '       1014,\n'
         '       1015,\n'
         '       1016,\n'
         '       1017,\n'
         '       1018,\n'
         '       1019])'),
        (a, 'deque([deque(...)])'),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


# TODO : pytest.mark.parametrise once nose is gone.
def test_collections_counter():
    class MyCounter(Counter):
        pass
    cases = [
        (Counter(), 'Counter()'),
        (Counter(a=1), "Counter({'a': 1})"),
        (MyCounter(a=1), "MyCounter({'a': 1})"),
        (Counter(a=1, c=22), "Counter({'c': 22, 'a': 1})"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected

# TODO : pytest.mark.parametrise once nose is gone.
def test_mappingproxy():
    MP = types.MappingProxyType
    underlying_dict = {}
    mp_recursive = MP(underlying_dict)
    underlying_dict[2] = mp_recursive
    underlying_dict[3] = underlying_dict

    cases = [
        (MP({}), "mappingproxy({})"),
        (MP({None: MP({})}), "mappingproxy({None: mappingproxy({})})"),
        (MP({k: k.upper() for k in string.ascii_lowercase}),
         "mappingproxy({'a': 'A',\n"
         "              'b': 'B',\n"
         "              'c': 'C',\n"
         "              'd': 'D',\n"
         "              'e': 'E',\n"
         "              'f': 'F',\n"
         "              'g': 'G',\n"
         "              'h': 'H',\n"
         "              'i': 'I',\n"
         "              'j': 'J',\n"
         "              'k': 'K',\n"
         "              'l': 'L',\n"
         "              'm': 'M',\n"
         "              'n': 'N',\n"
         "              'o': 'O',\n"
         "              'p': 'P',\n"
         "              'q': 'Q',\n"
         "              'r': 'R',\n"
         "              's': 'S',\n"
         "              't': 'T',\n"
         "              'u': 'U',\n"
         "              'v': 'V',\n"
         "              'w': 'W',\n"
         "              'x': 'X',\n"
         "              'y': 'Y',\n"
         "              'z': 'Z'})"),
        (mp_recursive, "mappingproxy({2: {...}, 3: {2: {...}, 3: {...}}})"),
        (underlying_dict,
         "{2: mappingproxy({2: {...}, 3: {...}}), 3: {...}}"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


# TODO : pytest.mark.parametrise once nose is gone.
def test_simplenamespace():
    SN = types.SimpleNamespace

    sn_recursive = SN()
    sn_recursive.first = sn_recursive
    sn_recursive.second = sn_recursive
    cases = [
        (SN(), "namespace()"),
        (SN(x=SN()), "namespace(x=namespace())"),
        (SN(a_long_name=[SN(s=string.ascii_lowercase)]*3, a_short_name=None),
         "namespace(a_long_name=[namespace(s='abcdefghijklmnopqrstuvwxyz'),\n"
         "                       namespace(s='abcdefghijklmnopqrstuvwxyz'),\n"
         "                       namespace(s='abcdefghijklmnopqrstuvwxyz')],\n"
         "          a_short_name=None)"),
        (sn_recursive, "namespace(first=namespace(...), second=namespace(...))"),
    ]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected


def test_pretty_environ():
    dict_repr = pretty.pretty(dict(os.environ))
    # reindent to align with 'environ' prefix
    dict_indented = dict_repr.replace('\n', '\n' + (' ' * len('environ')))
    env_repr = pretty.pretty(os.environ)
    assert env_repr == "environ" + dict_indented


def test_function_pretty():
    "Test pretty print of function"
    # posixpath is a pure python module, its interface is consistent
    # across Python distributions
    import posixpath

    assert pretty.pretty(posixpath.join) == "<function posixpath.join(a, *p)>"

    # custom function
    def meaning_of_life(question=None):
        if question:
            return 42
        return "Don't panic"

    assert "meaning_of_life(question=None)" in pretty.pretty(meaning_of_life)


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class MySet(set):  # Override repr of a basic type
    def __repr__(self):
        return 'mine'

def test_custom_repr():
    """A custom repr should override a pretty printer for a parent type"""
    oc = OrderedCounter("abracadabra")
    assert "OrderedCounter(OrderedDict" in pretty.pretty(oc)

    assert pretty.pretty(MySet()) == "mine"
