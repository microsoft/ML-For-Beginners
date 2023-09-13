'''
@author Fabio Zadrozny
'''
from _pydev_bundle import _pydev_imports_tipper
import inspect
import pytest
import sys
import unittest

try:
    import __builtin__  # @UnusedImport
    BUILTIN_MOD = '__builtin__'
except ImportError:
    BUILTIN_MOD = 'builtins'

IS_JYTHON = sys.platform.find('java') != -1

HAS_WX = False


@pytest.mark.skipif(IS_JYTHON, reason='CPython related test')
class TestCPython(unittest.TestCase):

    def p(self, t):
        for a in t:
            sys.stdout.write('%s\n' % (a,))

    def test_imports3(self):
        tip = _pydev_imports_tipper.generate_tip('os')
        ret = self.assert_in('path', tip)
        self.assertEqual('', ret[2])

    def test_imports2(self):
        try:
            tip = _pydev_imports_tipper.generate_tip('OpenGL.GLUT')
            self.assert_in('glutDisplayFunc', tip)
            self.assert_in('glutInitDisplayMode', tip)
        except ImportError:
            pass

    def test_imports4(self):
        try:
            tip = _pydev_imports_tipper.generate_tip('mx.DateTime.mxDateTime.mxDateTime')
            self.assert_in('now', tip)
        except ImportError:
            pass

    def test_imports5(self):
        tip = _pydev_imports_tipper.generate_tip('%s.list' % BUILTIN_MOD)
        s = self.assert_in('sort', tip)
        self.check_args(
            s,
            '(cmp=None, key=None, reverse=False)',
            '(self, object cmp, object key, bool reverse)',
            '(self, cmp: object, key: object, reverse: bool)',
            '(key=None, reverse=False)',
            '(self, key=None, reverse=False)',
            '(self, cmp, key, reverse)',
            '(self, key, reverse)',
        )

    def test_imports2a(self):
        tips = _pydev_imports_tipper.generate_tip('%s.RuntimeError' % BUILTIN_MOD)
        self.assert_in('__doc__', tips)

    def test_imports2b(self):
        try:
            file
        except:
            pass
        else:
            tips = _pydev_imports_tipper.generate_tip('%s' % BUILTIN_MOD)
            t = self.assert_in('file' , tips)
            self.assertTrue('->' in t[1].strip() or 'file' in t[1])

    def test_imports2c(self):
        try:
            file  # file is not available on py 3
        except:
            pass
        else:
            tips = _pydev_imports_tipper.generate_tip('%s.file' % BUILTIN_MOD)
            t = self.assert_in('readlines' , tips)
            self.assertTrue('->' in t[1] or 'sizehint' in t[1])

    def test_imports(self):
        '''
        You can print_ the results to check...
        '''
        if HAS_WX:
            tip = _pydev_imports_tipper.generate_tip('wxPython.wx')
            self.assert_in('wxApp'        , tip)

            tip = _pydev_imports_tipper.generate_tip('wxPython.wx.wxApp')

            try:
                tip = _pydev_imports_tipper.generate_tip('qt')
                self.assert_in('QWidget'        , tip)
                self.assert_in('QDialog'        , tip)

                tip = _pydev_imports_tipper.generate_tip('qt.QWidget')
                self.assert_in('rect'           , tip)
                self.assert_in('rect'           , tip)
                self.assert_in('AltButton'      , tip)

                tip = _pydev_imports_tipper.generate_tip('qt.QWidget.AltButton')
                self.assert_in('__xor__'      , tip)

                tip = _pydev_imports_tipper.generate_tip('qt.QWidget.AltButton.__xor__')
                self.assert_in('__class__'      , tip)
            except ImportError:
                pass

        tip = _pydev_imports_tipper.generate_tip(BUILTIN_MOD)
#        for t in tip[1]:
#            print_ t
        self.assert_in('object'         , tip)
        self.assert_in('tuple'          , tip)
        self.assert_in('list'          , tip)
        self.assert_in('RuntimeError'   , tip)
        self.assert_in('RuntimeWarning' , tip)

        # Remove cmp as it's not available on py 3
        # t = self.assert_in('cmp' , tip)
        # self.check_args(t, '(x, y)', '(object x, object y)', '(x: object, y: object)') #args

        t = self.assert_in('isinstance' , tip)
        self.check_args(
            t,
            '(object, class_or_type_or_tuple)',
            '(object o, type typeinfo)',
            '(o: object, typeinfo: type)',
            '(obj, class_or_tuple)',
            '(obj, klass_or_tuple)',
        )  # args

        t = self.assert_in('compile' , tip)
        self.check_args(
            t,
            '(source, filename, mode)',
            '()',
            '(o: object, name: str, val: object)',
            '(source, filename, mode, flags, dont_inherit, optimize)',
            '(source, filename, mode, flags, dont_inherit)',
            '(source, filename, mode, flags, dont_inherit, optimize, _feature_version=-1)'
        )  # args

        t = self.assert_in('setattr' , tip)
        self.check_args(
            t,
            '(object, name, value)',
            '(object o, str name, object val)',
            '(o: object, name: str, val: object)',
            '(obj, name, value)',
            '(object, name, val)',
        )  # args

        try:
            import compiler
            compiler_module = 'compiler'
        except ImportError:
            try:
                import ast
                compiler_module = 'ast'
            except ImportError:
                compiler_module = None

        if compiler_module is not None:  # Not available in iron python
            tip = _pydev_imports_tipper.generate_tip(compiler_module)
            if compiler_module == 'compiler':
                self.assert_args('parse', '(buf, mode)', tip)
                self.assert_args('walk', '(tree, visitor, walker, verbose)', tip)
                self.assert_in('parseFile'      , tip)
            else:
                self.assert_args('parse', [
                        '(source, filename, mode)',
                        '(source, filename, mode, type_comments=False, feature_version=None)'
                    ], tip
                )
                self.assert_args('walk', '(node)', tip)
            self.assert_in('parse'          , tip)

    def check_args(self, t, *expected):
        for x in expected:
            if x == t[2]:
                return
        self.fail('Found: %s. Expected: %s' % (t[2], expected))

    def assert_args(self, tok, args, tips):
        if not isinstance(args, (list, tuple)):
            args = (args,)

        for a in tips[1]:
            if tok == a[0]:
                for arg in args:
                    if arg == a[2]:
                        return
                raise AssertionError('%s not in %s', a[2], args)

        raise AssertionError('%s not in %s', tok, tips)

    def assert_in(self, tok, tips):
        for a in tips[1]:
            if tok == a[0]:
                return a
        raise AssertionError('%s not in %s' % (tok, tips))

    def test_search(self):
        s = _pydev_imports_tipper.search_definition('inspect.ismodule')
        (f, line, col), foundAs = s
        self.assertTrue(line > 0)

    def test_dot_net_libraries(self):
        if sys.platform == 'cli':
            tip = _pydev_imports_tipper.generate_tip('System.Drawing')
            self.assert_in('Brushes' , tip)

            tip = _pydev_imports_tipper.generate_tip('System.Drawing.Brushes')
            self.assert_in('Aqua' , tip)

    def test_tips_hasattr_failure(self):

        class MyClass(object):

            def __getattribute__(self, attr):
                raise RuntimeError()

        obj = MyClass()

        _pydev_imports_tipper.generate_imports_tip_for_module(obj)

    def test_inspect(self):

        class C(object):

            def metA(self, a, b):
                pass

        obj = C.metA
        if inspect.ismethod (obj):
            pass
#            print_ obj.im_func
#            print_ inspect.getargspec(obj.im_func)
