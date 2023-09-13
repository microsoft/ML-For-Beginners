# coding: utf-8
import collections
import sys
import re
import pytest
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
import json
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER

try:
    import numpy as np
except ImportError:
    np = None

PY_VER = sys.version_info[0]
assert PY_VER <= 3  # Fix the code when Python 4 comes around.
PY3K = PY_VER == 3


class SafeReprTestBase(object):

    saferepr = SafeRepr()

    def assert_saferepr(self, value, expected):
        safe = self.saferepr(value)

        if len(safe) != len(expected):
            raise AssertionError('Expected:\n%s\nFound:\n%s\n Expected len: %s Found len: %s' % (
                expected, safe, len(expected), len(safe),))
        assert safe == expected
        return safe

    def assert_unchanged(self, value, expected):
        actual = repr(value)

        safe = self.assert_saferepr(value, expected)
        assert safe == actual

    def assert_shortened(self, value, expected):
        actual = repr(value)

        safe = self.assert_saferepr(value, expected)
        assert safe != actual

    def assert_saferepr_regex(self, s, r):
        safe = self.saferepr(s)

        assert re.search(r, safe) is not None
        return safe

    def assert_unchanged_regex(self, value, expected):
        actual = repr(value)

        safe = self.assert_saferepr_regex(value, expected)
        assert safe == actual

    def assert_shortened_regex(self, value, expected):
        actual = repr(value)

        safe = self.assert_saferepr_regex(value, expected)
        assert safe != actual


class TestSafeRepr(SafeReprTestBase):

    def test_collection_types(self):
        colltypes = [t for t, _, _, _ in SafeRepr.collection_types]

        assert colltypes == [
            tuple,
            list,
            frozenset,
            set,
            collections.deque,
        ]

    def test_largest_repr(self):
        # Find the largest possible repr and ensure it is below our arbitrary
        # limit (8KB).
        coll = '-' * (SafeRepr.maxstring_outer * 2)
        for limit in reversed(SafeRepr.maxcollection[1:]):
            coll = [coll] * (limit * 2)
        dcoll = {}
        for i in range(SafeRepr.maxcollection[0]):
            dcoll[str(i) * SafeRepr.maxstring_outer] = coll
        text = self.saferepr(dcoll)
        # try:
        #    text_repr = repr(dcoll)
        # except MemoryError:
        #    print('Memory error raised while creating repr of test data')
        #    text_repr = ''
        # print('len(SafeRepr()(dcoll)) = ' + str(len(text)) +
        #      ', len(repr(coll)) = ' + str(len(text_repr)))

        assert len(text) < 8192


class TestStrings(SafeReprTestBase):

    def test_str_small(self):
        value = 'A' * 5

        self.assert_unchanged(value, "'AAAAA'")
        self.assert_unchanged([value], "['AAAAA']")

    def test_str_large(self):
        value = 'A' * (SafeRepr.maxstring_outer + 10)

        self.assert_shortened(value,
                              "'" + 'A' * 43690 + "..." + 'A' * 21845 + "'")
        self.assert_shortened([value], "['AAAAAAAAAAAAAAAAAAAA...AAAAAAAAAA']")

    def test_str_largest_unchanged(self):
        value = 'A' * (SafeRepr.maxstring_outer)

        self.assert_unchanged(value, "'" + 'A' * 65536 + "'")

    def test_str_smallest_changed(self):
        value = 'A' * (SafeRepr.maxstring_outer + 1)

        self.assert_shortened(value,
                              "'" + 'A' * 43690 + "..." + 'A' * 21845 + "'")

    def test_str_list_largest_unchanged(self):
        value = 'A' * (SafeRepr.maxstring_inner)

        self.assert_unchanged([value], "['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA']")

    def test_str_list_smallest_changed(self):
        value = 'A' * (SafeRepr.maxstring_inner + 1)

        self.assert_shortened([value], "['AAAAAAAAAAAAAAAAAAAA...AAAAAAAAAA']")

    @pytest.mark.skipif(sys.version_info > (3, 0), reason='Py2 specific test')
    def test_unicode_small(self):
        value = u'A' * 5

        self.assert_unchanged(value, "u'AAAAA'")
        self.assert_unchanged([value], "[u'AAAAA']")

    @pytest.mark.skipif(sys.version_info > (3, 0), reason='Py2 specific test')
    def test_unicode_large(self):
        value = u'A' * (SafeRepr.maxstring_outer + 10)

        self.assert_shortened(value,
                              "u'" + 'A' * 43690 + "..." + 'A' * 21845 + "'")
        self.assert_shortened([value], "[u'AAAAAAAAAAAAAAAAAAAA...AAAAAAAAAA']")

    @pytest.mark.skipif(sys.version_info < (3, 0), reason='Py3 specific test')
    def test_bytes_small(self):
        value = b'A' * 5

        self.assert_unchanged(value, "b'AAAAA'")
        self.assert_unchanged([value], "[b'AAAAA']")

    @pytest.mark.skipif(sys.version_info < (3, 0), reason='Py3 specific test')
    def test_bytes_large(self):
        value = b'A' * (SafeRepr.maxstring_outer + 10)

        self.assert_shortened(value,
                              "b'" + 'A' * 43690 + "..." + 'A' * 21845 + "'")
        self.assert_shortened([value], "[b'AAAAAAAAAAAAAAAAAAAA...AAAAAAAAAA']")

    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_bytearray_small(self):
    #    raise NotImplementedError
    #
    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_bytearray_large(self):
    #    raise NotImplementedError


class RawValueTests(SafeReprTestBase):

    def setUp(self):
        super(RawValueTests, self).setUp()
        self.saferepr.raw_value = True

    def test_unicode_raw(self):
        value = u'A\u2000' * 10000
        self.assert_saferepr(value, value)

    def test_bytes_raw(self):
        value = b'A' * 10000
        self.assert_saferepr(value, value.decode('ascii'))

    def test_bytearray_raw(self):
        value = bytearray(b'A' * 5)
        self.assert_saferepr(value, value.decode('ascii'))

# class TestNumbers(SafeReprTestBase):
#
#     @pytest.mark.skip(reason='not written')  # TODO: finish!
#     def test_int(self):
#         raise NotImplementedError
#
#     @pytest.mark.skip(reason='not written')  # TODO: finish!
#     def test_float(self):
#         raise NotImplementedError
#
#     @pytest.mark.skip(reason='not written')  # TODO: finish!
#     def test_complex(self):
#         raise NotImplementedError


class ContainerBase(object):

    CLASS = None
    LEFT = None
    RIGHT = None

    @property
    def info(self):
        try:
            return self._info
        except AttributeError:
            for info in SafeRepr.collection_types:
                ctype, _, _, _ = info
                if self.CLASS is ctype:
                    type(self)._info = info
                    return info
            else:
                raise TypeError('unsupported')

    def _combine(self, items, prefix, suffix, large):
        contents = ', '.join(str(item) for item in items)
        if large:
            contents += ', ...'
        return prefix + contents + suffix

    def combine(self, items, large=False):
        if self.LEFT is None:
            pytest.skip('unsupported')
        return self._combine(items, self.LEFT, self.RIGHT, large=large)

    def combine_nested(self, depth, items, large=False):
        _, _prefix, _suffix, comma = self.info
        prefix = _prefix * (depth + 1)
        if comma:
            suffix = _suffix + ("," + _suffix) * depth
        else:
            suffix = _suffix * (depth + 1)
        # print("ctype = " + ctype.__name__ + ", maxcollection[" +
        #      str(i) + "] == " + str(SafeRepr.maxcollection[i]))
        return self._combine(items, prefix, suffix, large=large)

    def test_large_flat(self):
        c1 = self.CLASS(range(SafeRepr.maxcollection[0] * 2))
        items = range(SafeRepr.maxcollection[0] - 1)
        c1_expect = self.combine(items, large=True)

        self.assert_shortened(c1, c1_expect)

    def test_large_nested(self):
        c1 = self.CLASS(range(SafeRepr.maxcollection[0] * 2))
        c1_items = range(SafeRepr.maxcollection[1] - 1)
        c1_expect = self.combine(c1_items, large=True)

        c2 = self.CLASS(c1 for _ in range(SafeRepr.maxcollection[0] * 2))
        items = (c1_expect for _ in range(SafeRepr.maxcollection[0] - 1))
        c2_expect = self.combine(items, large=True)

        self.assert_shortened(c2, c2_expect)

    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_empty(self):
    #     raise NotImplementedError
    #
    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_subclass(self):
    #     raise NotImplementedError

    def test_boundary(self):
        items1 = range(SafeRepr.maxcollection[0] - 1)
        items2 = range(SafeRepr.maxcollection[0])
        items3 = range(SafeRepr.maxcollection[0] + 1)
        c1 = self.CLASS(items1)
        c2 = self.CLASS(items2)
        c3 = self.CLASS(items3)
        expected1 = self.combine(items1)
        expected2 = self.combine(items2[:-1], large=True)
        expected3 = self.combine(items3[:-2], large=True)

        self.assert_unchanged(c1, expected1)
        self.assert_shortened(c2, expected2)
        self.assert_shortened(c3, expected3)

    def test_nested(self):
        ctype = self.CLASS
        for i in range(1, len(SafeRepr.maxcollection)):
            items1 = range(SafeRepr.maxcollection[i] - 1)
            items2 = range(SafeRepr.maxcollection[i])
            items3 = range(SafeRepr.maxcollection[i] + 1)
            c1 = self.CLASS(items1)
            c2 = self.CLASS(items2)
            c3 = self.CLASS(items3)
            for _j in range(i):
                c1, c2, c3 = ctype((c1,)), ctype((c2,)), ctype((c3,))
            expected1 = self.combine_nested(i, items1)
            expected2 = self.combine_nested(i, items2[:-1], large=True)
            expected3 = self.combine_nested(i, items3[:-2], large=True)

            self.assert_unchanged(c1, expected1)
            self.assert_shortened(c2, expected2)
            self.assert_shortened(c3, expected3)


class TestTuples(ContainerBase, SafeReprTestBase):

    CLASS = tuple
    LEFT = '('
    RIGHT = ')'


class TestLists(ContainerBase, SafeReprTestBase):

    CLASS = list
    LEFT = '['
    RIGHT = ']'

    def test_directly_recursive(self):
        value = [1, 2]
        value.append(value)

        self.assert_unchanged(value, '[1, 2, [...]]')

    def test_indirectly_recursive(self):
        value = [1, 2]
        value.append([value])

        self.assert_unchanged(value, '[1, 2, [[...]]]')


class TestFrozensets(ContainerBase, SafeReprTestBase):

    CLASS = frozenset


class TestSets(ContainerBase, SafeReprTestBase):

    CLASS = set
    if PY_VER != 2:
        LEFT = '{'
        RIGHT = '}'

    def test_nested(self):
        pytest.skip('unsupported')

    def test_large_nested(self):
        pytest.skip('unsupported')


class TestDicts(SafeReprTestBase):

    def test_large_key(self):
        value = {
            'a' * SafeRepr.maxstring_inner * 3: '',
        }

        self.assert_shortened_regex(value, r"{'a+\.\.\.a+': ''}")

    def test_large_value(self):
        value = {
            '': 'a' * SafeRepr.maxstring_inner * 2,
        }

        self.assert_shortened_regex(value, r"{'': 'a+\.\.\.a+'}")

    def test_large_both(self):
        value = {}
        key = 'a' * SafeRepr.maxstring_inner * 2
        value[key] = key

        self.assert_shortened_regex(value, r"{'a+\.\.\.a+': 'a+\.\.\.a+'}")

    def test_nested_value(self):
        d1 = {}
        d1_key = 'a' * SafeRepr.maxstring_inner * 2
        d1[d1_key] = d1_key
        d2 = {d1_key: d1}
        d3 = {d1_key: d2}

        self.assert_shortened_regex(d2, r"{'a+\.\.\.a+': {'a+\.\.\.a+': 'a+\.\.\.a+'}}")  # noqa
        if len(SafeRepr.maxcollection) == 2:
            self.assert_shortened_regex(d3, r"{'a+\.\.\.a+': {'a+\.\.\.a+': {\.\.\.}}}")  # noqa
        else:
            self.assert_shortened_regex(d3, r"{'a+\.\.\.a+': {'a+\.\.\.a+': {'a+\.\.\.a+': 'a+\.\.\.a+'}}}")  # noqa

    def test_empty(self):
        # Ensure empty dicts work
        self.assert_unchanged({}, '{}')

    def test_sorted(self):
        # Ensure dict keys are sorted
        d1 = {}
        d1['c'] = None
        d1['b'] = None
        d1['a'] = None
        if IS_PY36_OR_GREATER:
            self.assert_saferepr(d1, "{'c': None, 'b': None, 'a': None}")
        else:
            self.assert_saferepr(d1, "{'a': None, 'b': None, 'c': None}")

    @pytest.mark.skipif(sys.version_info < (3, 0), reason='Py3 specific test')
    def test_unsortable_keys(self):
        # Ensure dicts with unsortable keys do not crash
        d1 = {}
        for _ in range(100):
            d1[object()] = None
        with pytest.raises(TypeError):
            list(sorted(d1))
        self.saferepr(d1)

    def test_directly_recursive(self):
        value = {1: None}
        value[2] = value

        self.assert_unchanged(value, '{1: None, 2: {...}}')

    def test_indirectly_recursive(self):
        value = {1: None}
        value[2] = {3: value}

        self.assert_unchanged(value, '{1: None, 2: {3: {...}}}')


class TestOtherPythonTypes(SafeReprTestBase):
    # not critical to test:
    #  singletons
    #  <function>
    #  <class>
    #  <iterator>
    #  memoryview
    #  classmethod
    #  staticmethod
    #  property
    #  enumerate
    #  reversed
    #  object
    #  type
    #  super

    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_file(self):
    #     raise NotImplementedError

    def test_range_small(self):
        range_name = range.__name__
        value = range(1, 42)

        self.assert_unchanged(value, '%s(1, 42)' % (range_name,))

    @pytest.mark.skipif(sys.version_info < (3, 0), reason='Py3 specific test')
    def test_range_large_stop_only(self):
        range_name = range.__name__
        stop = SafeRepr.maxcollection[0]
        value = range(stop)

        self.assert_unchanged(value,
                              '%s(0, %s)' % (range_name, stop))

    def test_range_large_with_start(self):
        range_name = range.__name__
        stop = SafeRepr.maxcollection[0] + 1
        value = range(1, stop)

        self.assert_unchanged(value,
                              '%s(1, %s)' % (range_name, stop))

    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_named_struct(self):
    #     # e.g. sys.version_info
    #     raise NotImplementedError
    #
    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # def test_namedtuple(self):
    #     raise NotImplementedError
    #
    # @pytest.mark.skip(reason='not written')  # TODO: finish!
    # @pytest.mark.skipif(sys.version_info < (3, 0), reason='Py3 specific test')
    # def test_SimpleNamespace(self):
    #     raise NotImplementedError


class TestUserDefinedObjects(SafeReprTestBase):

    def test_broken_repr(self):

        class TestClass(object):

            def __repr__(self):
                raise NameError

        value = TestClass()

        with pytest.raises(NameError):
            repr(TestClass())
        self.assert_saferepr(value, object.__repr__(value))

    def test_large(self):

        class TestClass(object):

            def __repr__(self):
                return '<' + 'A' * SafeRepr.maxother_outer * 2 + '>'

        value = TestClass()

        self.assert_shortened_regex(value, r'\<A+\.\.\.A+\>')

    def test_inherit_repr(self):

        class TestClass(dict):
            pass

        value_dict = TestClass()

        class TestClass2(list):
            pass

        value_list = TestClass2()

        self.assert_unchanged(value_dict, '{}')
        self.assert_unchanged(value_list, '[]')

    def test_custom_repr(self):

        class TestClass(dict):

            def __repr__(self):
                return 'MyRepr'

        value1 = TestClass()

        class TestClass2(list):

            def __repr__(self):
                return 'MyRepr'

        value2 = TestClass2()

        self.assert_unchanged(value1, 'MyRepr')
        self.assert_unchanged(value2, 'MyRepr')

    def test_custom_repr_many_items(self):

        class TestClass(list):

            def __init__(self, it=()):
                list.__init__(self, it)

            def __repr__(self):
                return 'MyRepr'

        value1 = TestClass(range(0, 15))
        value2 = TestClass(range(0, 16))
        value3 = TestClass([TestClass(range(0, 10))])
        value4 = TestClass([TestClass(range(0, 11))])

        self.assert_unchanged(value1, 'MyRepr')
        self.assert_shortened(value2, '<TestClass, len() = 16>')
        self.assert_unchanged(value3, 'MyRepr')
        self.assert_shortened(value4, '<TestClass, len() = 1>')

    def test_custom_repr_large_item(self):

        class TestClass(list):

            def __init__(self, it=()):
                list.__init__(self, it)

            def __repr__(self):
                return 'MyRepr'

        value1 = TestClass(['a' * (SafeRepr.maxcollection[1] + 1)])
        value2 = TestClass(['a' * (SafeRepr.maxstring_inner + 1)])

        self.assert_unchanged(value1, 'MyRepr')
        self.assert_shortened(value2, '<TestClass, len() = 1>')


@pytest.mark.skipif(np is None, reason='could not import numpy')
class TestNumpy(SafeReprTestBase):
    # numpy types should all use their native reprs, even arrays
    # exceeding limits.

    def test_int32(self):
        value = np.int32(123)

        self.assert_unchanged(value, repr(value))

    def test_float32(self):
        value = np.float32(123.456)

        self.assert_unchanged(value, repr(value))

    def test_zeros(self):
        value = np.zeros(SafeRepr.maxcollection[0] + 1)

        self.assert_unchanged(value, repr(value))


@pytest.mark.parametrize('params', [
    {'maxother_outer': 20, 'input': "😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄FFFFFFFF", 'output': '😄😄😄😄😄😄😄😄😄😄😄😄😄...FFFFFF'},
    {'maxother_outer': 10, 'input': "😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄😄FFFFFFFF", 'output': '😄😄😄😄😄😄...FFF'},
    {'maxother_outer': 10, 'input': u"������������FFFFFFFF", 'output': u"������...FFF"},

    # Because we can't return bytes, byte-related tests aren't needed (and str works as it should).
])
@pytest.mark.parametrize('use_str', [True, False])
def test_py3_str_slicing(params, use_str):
    # Note: much simpler in python because __repr__ is required to return str
    safe_repr = SafeRepr()
    safe_repr.locale_preferred_encoding = 'ascii'
    safe_repr.sys_stdout_encoding = params.get('sys_stdout_encoding', 'ascii')

    safe_repr.maxother_outer = params['maxother_outer']

    if not use_str:

        class MyObj(object):

            def __repr__(self):
                return params['input']

        safe_repr_input = MyObj()
    else:
        safe_repr_input = params['input']
    expected_output = params['output']
    computed = safe_repr(safe_repr_input)
    expected = repr(expected_output)
    if use_str:
        expected = repr(expected)
    assert repr(computed) == expected

    # Check that we can json-encode the return.
    assert json.dumps(computed)


def test_raw_bytes():
    safe_repr = SafeRepr()
    safe_repr.raw_value = True
    obj = b'\xed\xbd\xbf\xff\xfe\xfa\xfd'
    raw_value_repr = safe_repr(obj)
    assert isinstance(raw_value_repr, str)  # bytes on py2, str on py3
    assert raw_value_repr == obj.decode('latin1')


def test_raw_unicode():
    safe_repr = SafeRepr()
    safe_repr.raw_value = True
    obj = u'\xed\xbd\xbf\xff\xfe\xfa\xfd'
    raw_value_repr = safe_repr(obj)
    assert isinstance(raw_value_repr, str)  # bytes on py2, str on py3
    assert raw_value_repr == obj


def test_no_repr():

    class MyBytes(object):

        def __init__(self, contents):
            self.contents = contents
            self.errored = None

        def __iter__(self):
            return iter(self.contents)

        def decode(self, encoding):
            self.errored = 'decode called'
            raise RuntimeError('Should not be called.')

        def __repr__(self):
            self.errored = '__repr__ called'
            raise RuntimeError('Should not be called.')

        def __getitem__(self, *args):
            return self.contents.__getitem__(*args)

        def __len__(self):
            return len(self.contents)

    safe_repr = SafeRepr()
    safe_repr.string_types = (MyBytes,)
    safe_repr.bytes = MyBytes
    obj = b'f' * (safe_repr.maxstring_outer * 10)
    my_bytes = MyBytes(obj)
    raw_value_repr = safe_repr(my_bytes)
    assert not my_bytes.errored

