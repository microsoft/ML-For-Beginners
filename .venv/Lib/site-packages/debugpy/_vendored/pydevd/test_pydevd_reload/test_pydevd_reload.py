import os  # @NoMove
import sys  # @NoMove
import pytest

from _pydevd_bundle import pydevd_reload
import tempfile
import unittest

SAMPLE_CODE = """
class C:
    def foo(self):
        return 0

    @classmethod
    def bar(cls):
        return (0, 0)

    @staticmethod
    def stomp():
        return (0, 0, 0)

    def unchanged(self):
        return 'unchanged'
"""

from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON


@pytest.mark.skipif(IS_JYTHON or IS_IRONPYTHON, reason='CPython related test')
class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tempdir = None
        self.save_path = None
        self.tempdir = tempfile.mkdtemp()
        self.save_path = list(sys.path)
        sys.path.append(self.tempdir)
        try:
            del sys.modules['x']
        except:
            pass

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        sys.path = self.save_path
        try:
            del sys.modules['x']
        except:
            pass

    def make_mod(self, name="x", repl=None, subst=None, sample=SAMPLE_CODE):
        basedir = self.tempdir
        if '.' in name:
            splitted = name.split('.')
            basedir = os.path.join(self.tempdir, *splitted[:-1])
            name = splitted[-1]
            try:
                os.makedirs(basedir)
            except:
                pass

        fn = os.path.join(basedir, name + ".py")
        f = open(fn, "w")
        if repl is not None and subst is not None:
            sample = sample.replace(repl, subst)
        try:
            f.write(sample)
        finally:
            f.close()

    def test_pydevd_reload(self):

        self.make_mod()
        import x  # @UnresolvedImport

        C = x.C
        COut = C
        Cfoo = C.foo
        Cbar = C.bar
        Cstomp = C.stomp

        def check2(expected):
            C = x.C
            Cfoo = C.foo
            Cbar = C.bar
            Cstomp = C.stomp
            b = C()
            bfoo = b.foo
            self.assertEqual(expected, b.foo())
            self.assertEqual(expected, bfoo())
            self.assertEqual(expected, Cfoo(b))

        def check(expected):
            b = COut()
            bfoo = b.foo
            self.assertEqual(expected, b.foo())
            self.assertEqual(expected, bfoo())
            self.assertEqual(expected, Cfoo(b))
            self.assertEqual((expected, expected), Cbar())
            self.assertEqual((expected, expected, expected), Cstomp())
            check2(expected)

        check(0)

        # modify mod and reload
        count = 0
        while count < 1:
            count += 1
            self.make_mod(repl="0", subst=str(count))
            pydevd_reload.xreload(x)
            check(count)

    def test_pydevd_reload2(self):

        self.make_mod()
        import x  # @UnresolvedImport

        c = x.C()
        cfoo = c.foo
        self.assertEqual(0, c.foo())
        self.assertEqual(0, cfoo())

        self.make_mod(repl="0", subst='1')
        pydevd_reload.xreload(x)
        self.assertEqual(1, c.foo())
        self.assertEqual(1, cfoo())

    def test_pydevd_reload3(self):

        class F:

            def m1(self):
                return 1

        class G:

            def m1(self):
                return 2

        self.assertEqual(F().m1(), 1)
        pydevd_reload.Reload(None)._update(None, None, F, G)
        self.assertEqual(F().m1(), 2)

    def test_pydevd_reload4(self):

        class F:
            pass

        F.m1 = lambda a:None

        class G:
            pass

        G.m1 = lambda a:10

        self.assertEqual(F().m1(), None)
        pydevd_reload.Reload(None)._update(None, None, F, G)
        self.assertEqual(F().m1(), 10)

    def test_if_code_obj_equals(self):

        class F:

            def m1(self):
                return 1

        class G:

            def m1(self):
                return 1

        class H:

            def m1(self):
                return 2

        if hasattr(F.m1, 'func_code'):
            self.assertTrue(pydevd_reload.code_objects_equal(F.m1.func_code, G.m1.func_code))
            self.assertFalse(pydevd_reload.code_objects_equal(F.m1.func_code, H.m1.func_code))
        else:
            self.assertTrue(pydevd_reload.code_objects_equal(F.m1.__code__, G.m1.__code__))
            self.assertFalse(pydevd_reload.code_objects_equal(F.m1.__code__, H.m1.__code__))

    def test_metaclass(self):

        class Meta(type):

            def __init__(cls, name, bases, attrs):
                super(Meta, cls).__init__(name, bases, attrs)

        class F:
            __metaclass__ = Meta

            def m1(self):
                return 1

        class G:
            __metaclass__ = Meta

            def m1(self):
                return 2

        self.assertEqual(F().m1(), 1)
        pydevd_reload.Reload(None)._update(None, None, F, G)
        self.assertEqual(F().m1(), 2)

    def test_change_hierarchy(self):

        class F(object):

            def m1(self):
                return 1

        class B(object):

            def super_call(self):
                return 2

        class G(B):

            def m1(self):
                return self.super_call()

        self.assertEqual(F().m1(), 1)
        old = pydevd_reload.notify_error
        self._called = False

        def on_error(*args):
            self._called = True

        try:
            pydevd_reload.notify_error = on_error
            pydevd_reload.Reload(None)._update(None, None, F, G)
            self.assertTrue(self._called)
        finally:
            pydevd_reload.notify_error = old

    def test_change_hierarchy_old_style(self):

        class F:

            def m1(self):
                return 1

        class B:

            def super_call(self):
                return 2

        class G(B):

            def m1(self):
                return self.super_call()

        self.assertEqual(F().m1(), 1)
        old = pydevd_reload.notify_error
        self._called = False

        def on_error(*args):
            self._called = True

        try:
            pydevd_reload.notify_error = on_error
            pydevd_reload.Reload(None)._update(None, None, F, G)
            self.assertTrue(self._called)
        finally:
            pydevd_reload.notify_error = old

    def test_create_class(self):
        SAMPLE_CODE1 = """
class C:
    def foo(self):
        return 0
"""
        # Creating a new class and using it from old class
        SAMPLE_CODE2 = """
class B:
    pass

class C:
    def foo(self):
        return B
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.C().foo
        self.assertEqual(foo(), 0)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo().__name__, 'B')

    def test_create_class2(self):
        SAMPLE_CODE1 = """
class C(object):
    def foo(self):
        return 0
"""
        # Creating a new class and using it from old class
        SAMPLE_CODE2 = """
class B(object):
    pass

class C(object):
    def foo(self):
        return B
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.C().foo
        self.assertEqual(foo(), 0)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo().__name__, 'B')

    def test_parent_function(self):
        SAMPLE_CODE1 = """
class B(object):
    def foo(self):
        return 0

class C(B):
    def call(self):
        return self.foo()
"""
        # Creating a new class and using it from old class
        SAMPLE_CODE2 = """
class B(object):
    def foo(self):
        return 0
    def bar(self):
        return 'bar'

class C(B):
    def call(self):
        return self.bar()
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        call = x.C().call
        self.assertEqual(call(), 0)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(call(), 'bar')

    def test_update_constant(self):
        SAMPLE_CODE1 = """
CONSTANT = 1

class B(object):
    def foo(self):
        return CONSTANT
"""
        SAMPLE_CODE2 = """
CONSTANT = 2

class B(object):
    def foo(self):
        return CONSTANT
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.B().foo
        self.assertEqual(foo(), 1)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo(), 1)  # Just making it explicit we don't reload constants.

    def test_update_constant_with_custom_code(self):
        SAMPLE_CODE1 = """
CONSTANT = 1

class B(object):
    def foo(self):
        return CONSTANT
"""
        SAMPLE_CODE2 = """
CONSTANT = 2

def __xreload_old_new__(namespace, name, old, new):
    if name == 'CONSTANT':
        namespace[name] = new

class B(object):
    def foo(self):
        return CONSTANT
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.B().foo
        self.assertEqual(foo(), 1)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo(), 2)  # Actually updated it now!

    def test_reload_custom_code_after_changes(self):
        SAMPLE_CODE1 = """
CONSTANT = 1

class B(object):
    def foo(self):
        return CONSTANT
"""
        SAMPLE_CODE2 = """
CONSTANT = 1

def __xreload_after_reload_update__(namespace):
    namespace['CONSTANT'] = 2

class B(object):
    def foo(self):
        return CONSTANT
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.B().foo
        self.assertEqual(foo(), 1)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo(), 2)  # Actually updated it now!

    def test_reload_custom_code_after_changes_in_class(self):
        SAMPLE_CODE1 = """

class B(object):
    CONSTANT = 1

    def foo(self):
        return self.CONSTANT
"""
        SAMPLE_CODE2 = """


class B(object):
    CONSTANT = 1

    @classmethod
    def __xreload_after_reload_update__(cls):
        cls.CONSTANT = 2

    def foo(self):
        return self.CONSTANT
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.B().foo
        self.assertEqual(foo(), 1)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo(), 2)  # Actually updated it now!

    def test_update_constant_with_custom_code2(self):
        SAMPLE_CODE1 = """

class B(object):
    CONSTANT = 1

    def foo(self):
        return self.CONSTANT
"""
        SAMPLE_CODE2 = """


class B(object):

    CONSTANT = 2

    def __xreload_old_new__(cls, name, old, new):
        if name == 'CONSTANT':
            cls.CONSTANT = new
    __xreload_old_new__ = classmethod(__xreload_old_new__)

    def foo(self):
        return self.CONSTANT
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        foo = x.B().foo
        self.assertEqual(foo(), 1)
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        self.assertEqual(foo(), 2)  # Actually updated it now!

    def test_update_with_slots(self):
        SAMPLE_CODE1 = """
class B(object):

    __slots__ = ['bar']

"""
        SAMPLE_CODE2 = """
class B(object):

    __slots__ = ['bar', 'foo']

    def m1(self):
        self.bar = 10
        return 1

"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        B = x.B
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        b = B()
        self.assertEqual(1, b.m1())
        self.assertEqual(10, b.bar)
        self.assertRaises(Exception, setattr, b, 'foo', 20)  # __slots__ can't be updated

    def test_reload_numpy(self):
        SAMPLE_CODE1 = """
import numpy as np
global_numpy = np.array([1, 2, 3])
def method():
    return 1
"""
        SAMPLE_CODE2 = """
import numpy as np
global_numpy = np.array([1, 2, 3, 4])
def method():
    return 2
"""

        self.make_mod(sample=SAMPLE_CODE1)
        import x  # @UnresolvedImport
        assert str(x.global_numpy) == '[1 2 3]'
        self.make_mod(sample=SAMPLE_CODE2)
        pydevd_reload.xreload(x)
        # Note that we don't patch globals (the user could do that in a module,
        # but he'd have to create a custom `__xreload_old_new__` method to
        # do it).
        assert str(x.global_numpy) == '[1 2 3]'

    def test_reload_relative(self):
        MODULE_CODE = """
def add_text(s):
    return s + " module"
"""
        MODULE1_CODE = """
from . import module

def add_more_text(s):
    s = module.add_text(s)
    return s + ' module1'
"""

        MODULE1_CODE_V2 = """
from . import module

def add_more_text(s):
    s = module.add_text(s)
    return s + ' module1V2'
"""

        self.make_mod(sample='', name='package.__init__')
        self.make_mod(sample=MODULE_CODE, name='package.module')
        self.make_mod(sample=MODULE1_CODE, name='package.module1')
        from package import module1  # @UnresolvedImport
        assert module1.add_more_text('1') == '1 module module1'

        self.make_mod(sample=MODULE1_CODE_V2, name='package.module1')
        pydevd_reload.xreload(module1)
        assert module1.add_more_text('1') == '1 module module1V2'
