"""Tests for the object inspection functionality.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys

from .. import oinspect

from decorator import decorator

from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user


#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------

inspector = None

def setup_module():
    global inspector
    inspector = oinspect.Inspector()


class SourceModuleMainTest:
    __module__ = "__main__"


#-----------------------------------------------------------------------------
# Local utilities
#-----------------------------------------------------------------------------

# WARNING: since this test checks the line number where a function is
# defined, if any code is inserted above, the following line will need to be
# updated.  Do NOT insert any whitespace between the next line and the function
# definition below.
THIS_LINE_NUMBER = 47  # Put here the actual number of this line


def test_find_source_lines():
    assert oinspect.find_source_lines(test_find_source_lines) == THIS_LINE_NUMBER + 3
    assert oinspect.find_source_lines(type) is None
    assert oinspect.find_source_lines(SourceModuleMainTest) is None
    assert oinspect.find_source_lines(SourceModuleMainTest()) is None


def test_getsource():
    assert oinspect.getsource(type) is None
    assert oinspect.getsource(SourceModuleMainTest) is None
    assert oinspect.getsource(SourceModuleMainTest()) is None


def test_inspect_getfile_raises_exception():
    """Check oinspect.find_file/getsource/find_source_lines expectations"""
    with pytest.raises(TypeError):
        inspect.getfile(type)
    with pytest.raises(OSError if sys.version_info >= (3, 10) else TypeError):
        inspect.getfile(SourceModuleMainTest)


# A couple of utilities to ensure these tests work the same from a source or a
# binary install
def pyfile(fname):
    return os.path.normcase(re.sub('.py[co]$', '.py', fname))


def match_pyfiles(f1, f2):
    assert pyfile(f1) == pyfile(f2)


def test_find_file():
    match_pyfiles(oinspect.find_file(test_find_file), os.path.abspath(__file__))
    assert oinspect.find_file(type) is None
    assert oinspect.find_file(SourceModuleMainTest) is None
    assert oinspect.find_file(SourceModuleMainTest()) is None


def test_find_file_decorated1():

    @decorator
    def noop1(f):
        def wrapper(*a, **kw):
            return f(*a, **kw)
        return wrapper

    @noop1
    def f(x):
        "My docstring"

    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == "My docstring"


def test_find_file_decorated2():

    @decorator
    def noop2(f, *a, **kw):
        return f(*a, **kw)

    @noop2
    @noop2
    @noop2
    def f(x):
        "My docstring 2"

    match_pyfiles(oinspect.find_file(f), os.path.abspath(__file__))
    assert f.__doc__ == "My docstring 2"


def test_find_file_magic():
    run = ip.find_line_magic('run')
    assert oinspect.find_file(run) is not None


# A few generic objects we can then inspect in the tests below

class Call(object):
    """This is the class docstring."""

    def __init__(self, x, y=1):
        """This is the constructor docstring."""

    def __call__(self, *a, **kw):
        """This is the call docstring."""

    def method(self, x, z=2):
        """Some method's docstring"""

class HasSignature(object):
    """This is the class docstring."""
    __signature__ = Signature([Parameter('test', Parameter.POSITIONAL_OR_KEYWORD)])

    def __init__(self, *args):
        """This is the init docstring"""


class SimpleClass(object):
    def method(self, x, z=2):
        """Some method's docstring"""


class Awkward(object):
    def __getattr__(self, name):
        raise Exception(name)

class NoBoolCall:
    """
    callable with `__bool__` raising should still be inspect-able.
    """

    def __call__(self):
        """does nothing"""
        pass

    def __bool__(self):
        """just raise NotImplemented"""
        raise NotImplementedError('Must be implemented')


class SerialLiar(object):
    """Attribute accesses always get another copy of the same class.

    unittest.mock.call does something similar, but it's not ideal for testing
    as the failure mode is to eat all your RAM. This gives up after 10k levels.
    """
    def __init__(self, max_fibbing_twig, lies_told=0):
        if lies_told > 10000:
            raise RuntimeError('Nose too long, honesty is the best policy')
        self.max_fibbing_twig = max_fibbing_twig
        self.lies_told = lies_told
        max_fibbing_twig[0] = max(max_fibbing_twig[0], lies_told)

    def __getattr__(self, item):
        return SerialLiar(self.max_fibbing_twig, self.lies_told + 1)

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

def test_info():
    "Check that Inspector.info fills out various fields as expected."
    i = inspector.info(Call, oname="Call")
    assert i["type_name"] == "type"
    expected_class = str(type(type))  # <class 'type'> (Python 3) or <type 'type'>
    assert i["base_class"] == expected_class
    assert re.search(
        "<class 'IPython.core.tests.test_oinspect.Call'( at 0x[0-9a-f]{1,9})?>",
        i["string_form"],
    )
    fname = __file__
    if fname.endswith(".pyc"):
        fname = fname[:-1]
    # case-insensitive comparison needed on some filesystems
    # e.g. Windows:
    assert i["file"].lower() == compress_user(fname).lower()
    assert i["definition"] == None
    assert i["docstring"] == Call.__doc__
    assert i["source"] == None
    assert i["isclass"] is True
    assert i["init_definition"] == "Call(x, y=1)"
    assert i["init_docstring"] == Call.__init__.__doc__

    i = inspector.info(Call, detail_level=1)
    assert i["source"] is not None
    assert i["docstring"] == None

    c = Call(1)
    c.__doc__ = "Modified instance docstring"
    i = inspector.info(c)
    assert i["type_name"] == "Call"
    assert i["docstring"] == "Modified instance docstring"
    assert i["class_docstring"] == Call.__doc__
    assert i["init_docstring"] == Call.__init__.__doc__
    assert i["call_docstring"] == Call.__call__.__doc__


def test_class_signature():
    info = inspector.info(HasSignature, "HasSignature")
    assert info["init_definition"] == "HasSignature(test)"
    assert info["init_docstring"] == HasSignature.__init__.__doc__


def test_info_awkward():
    # Just test that this doesn't throw an error.
    inspector.info(Awkward())

def test_bool_raise():
    inspector.info(NoBoolCall())

def test_info_serialliar():
    fib_tracker = [0]
    inspector.info(SerialLiar(fib_tracker))

    # Nested attribute access should be cut off at 100 levels deep to avoid
    # infinite loops: https://github.com/ipython/ipython/issues/9122
    assert fib_tracker[0] < 9000

def support_function_one(x, y=2, *a, **kw):
    """A simple function."""

def test_calldef_none():
    # We should ignore __call__ for all of these.
    for obj in [support_function_one, SimpleClass().method, any, str.upper]:
        i = inspector.info(obj)
        assert i["call_def"] is None


def f_kwarg(pos, *, kwonly):
    pass

def test_definition_kwonlyargs():
    i = inspector.info(f_kwarg, oname="f_kwarg")  # analysis:ignore
    assert i["definition"] == "f_kwarg(pos, *, kwonly)"


def test_getdoc():
    class A(object):
        """standard docstring"""
        pass

    class B(object):
        """standard docstring"""
        def getdoc(self):
            return "custom docstring"

    class C(object):
        """standard docstring"""
        def getdoc(self):
            return None

    a = A()
    b = B()
    c = C()

    assert oinspect.getdoc(a) == "standard docstring"
    assert oinspect.getdoc(b) == "custom docstring"
    assert oinspect.getdoc(c) == "standard docstring"


def test_empty_property_has_no_source():
    i = inspector.info(property(), detail_level=1)
    assert i["source"] is None


def test_property_sources():
    # A simple adder whose source and signature stays
    # the same across Python distributions
    def simple_add(a, b):
        "Adds two numbers"
        return a + b

    class A(object):
        @property
        def foo(self):
            return 'bar'

        foo = foo.setter(lambda self, v: setattr(self, 'bar', v))

        dname = property(oinspect.getdoc)
        adder = property(simple_add)

    i = inspector.info(A.foo, detail_level=1)
    assert "def foo(self):" in i["source"]
    assert "lambda self, v:" in i["source"]

    i = inspector.info(A.dname, detail_level=1)
    assert "def getdoc(obj)" in i["source"]

    i = inspector.info(A.adder, detail_level=1)
    assert "def simple_add(a, b)" in i["source"]


def test_property_docstring_is_in_info_for_detail_level_0():
    class A(object):
        @property
        def foobar(self):
            """This is `foobar` property."""
            pass

    ip.user_ns["a_obj"] = A()
    assert (
        "This is `foobar` property."
        == ip.object_inspect("a_obj.foobar", detail_level=0)["docstring"]
    )

    ip.user_ns["a_cls"] = A
    assert (
        "This is `foobar` property."
        == ip.object_inspect("a_cls.foobar", detail_level=0)["docstring"]
    )


def test_pdef():
    # See gh-1914
    def foo(): pass
    inspector.pdef(foo, 'foo')


@contextmanager
def cleanup_user_ns(**kwargs):
    """
    On exit delete all the keys that were not in user_ns before entering.

    It does not restore old values !

    Parameters
    ----------

    **kwargs
        used to update ip.user_ns

    """
    try:
        known = set(ip.user_ns.keys())
        ip.user_ns.update(kwargs)
        yield
    finally:
        added = set(ip.user_ns.keys()) - known
        for k in added:
            del ip.user_ns[k]


def test_pinfo_bool_raise():
    """
    Test that bool method is not called on parent.
    """

    class RaiseBool:
        attr = None

        def __bool__(self):
            raise ValueError("pinfo should not access this method")

    raise_bool = RaiseBool()

    with cleanup_user_ns(raise_bool=raise_bool):
        ip._inspect("pinfo", "raise_bool.attr", detail_level=0)


def test_pinfo_getindex():
    def dummy():
        """
        MARKER
        """

    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints("MARKER"):
            ip._inspect("pinfo", "container[0]", detail_level=0)
    assert "container" not in ip.user_ns.keys()


def test_qmark_getindex():
    def dummy():
        """
        MARKER 2
        """

    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints("MARKER 2"):
            ip.run_cell("container[0]?")
    assert "container" not in ip.user_ns.keys()


def test_qmark_getindex_negatif():
    def dummy():
        """
        MARKER 3
        """

    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints("MARKER 3"):
            ip.run_cell("container[-1]?")
    assert "container" not in ip.user_ns.keys()



def test_pinfo_nonascii():
    # See gh-1177
    from . import nonascii2
    ip.user_ns['nonascii2'] = nonascii2
    ip._inspect('pinfo', 'nonascii2', detail_level=1)

def test_pinfo_type():
    """
    type can fail in various edge case, for example `type.__subclass__()`
    """
    ip._inspect('pinfo', 'type')


def test_pinfo_docstring_no_source():
    """Docstring should be included with detail_level=1 if there is no source"""
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=0)
    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'str.format', detail_level=1)


def test_pinfo_no_docstring_if_source():
    """Docstring should not be included with detail_level=1 if source is found"""
    def foo():
        """foo has a docstring"""

    ip.user_ns['foo'] = foo

    with AssertPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=0)
    with AssertPrints('Source:'):
        ip._inspect('pinfo', 'foo', detail_level=1)
    with AssertNotPrints('Docstring:'):
        ip._inspect('pinfo', 'foo', detail_level=1)


def test_pinfo_docstring_if_detail_and_no_source():
    """ Docstring should be displayed if source info not available """
    obj_def = '''class Foo(object):
                  """ This is a docstring for Foo """
                  def bar(self):
                      """ This is a docstring for Foo.bar """
                      pass
              '''

    ip.run_cell(obj_def)
    ip.run_cell('foo = Foo()')

    with AssertNotPrints("Source:"):
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=0)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=1)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=0)

    with AssertNotPrints('Docstring:'):
        with AssertPrints('Source:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=1)


def test_pinfo_docstring_dynamic():
    obj_def = """class Bar:
    __custom_documentations__ = {
     "prop" : "cdoc for prop",
     "non_exist" : "cdoc for non_exist",
    }
    @property
    def prop(self):
        '''
        Docstring for prop
        '''
        return self._prop
    
    @prop.setter
    def prop(self, v):
        self._prop = v
    """
    ip.run_cell(obj_def)

    ip.run_cell("b = Bar()")

    with AssertPrints("Docstring:   cdoc for prop"):
        ip.run_line_magic("pinfo", "b.prop")

    with AssertPrints("Docstring:   cdoc for non_exist"):
        ip.run_line_magic("pinfo", "b.non_exist")

    with AssertPrints("Docstring:   cdoc for prop"):
        ip.run_cell("b.prop?")

    with AssertPrints("Docstring:   cdoc for non_exist"):
        ip.run_cell("b.non_exist?")

    with AssertPrints("Docstring:   <no docstring>"):
        ip.run_cell("b.undefined?")


def test_pinfo_magic():
    with AssertPrints("Docstring:"):
        ip._inspect("pinfo", "lsmagic", detail_level=0)

    with AssertPrints("Source:"):
        ip._inspect("pinfo", "lsmagic", detail_level=1)


def test_init_colors():
    # ensure colors are not present in signature info
    info = inspector.info(HasSignature)
    init_def = info["init_definition"]
    assert "[0m" not in init_def


def test_builtin_init():
    info = inspector.info(list)
    init_def = info['init_definition']
    assert init_def is not None


def test_render_signature_short():
    def short_fun(a=1): pass
    sig = oinspect._render_signature(
        signature(short_fun),
        short_fun.__name__,
    )
    assert sig == "short_fun(a=1)"


def test_render_signature_long():
    from typing import Optional

    def long_function(
        a_really_long_parameter: int,
        and_another_long_one: bool = False,
        let_us_make_sure_this_is_looong: Optional[str] = None,
    ) -> bool: pass

    sig = oinspect._render_signature(
        signature(long_function),
        long_function.__name__,
    )
    expected = """\
long_function(
    a_really_long_parameter: int,
    and_another_long_one: bool = False,
    let_us_make_sure_this_is_looong: Optional[str] = None,
) -> bool\
"""

    assert sig == expected
