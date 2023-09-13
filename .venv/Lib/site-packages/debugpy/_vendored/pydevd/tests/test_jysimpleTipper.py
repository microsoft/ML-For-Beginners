import unittest
import os
import sys
import pytest

# Note: ant.jar and junit.jar must be in the PYTHONPATH (see jython_test_deps)

IS_JYTHON = False
if sys.platform.find('java') != -1:
    IS_JYTHON = True
    from _pydev_bundle._pydev_jy_imports_tipper import ismethod
    from _pydev_bundle._pydev_jy_imports_tipper import isclass
    from _pydev_bundle._pydev_jy_imports_tipper import dir_obj
    from _pydev_bundle import _pydev_jy_imports_tipper
    from java.lang.reflect import Method  # @UnresolvedImport
    from java.lang import System  # @UnresolvedImport
    from java.lang import String  # @UnresolvedImport
    from java.lang.System import arraycopy  # @UnresolvedImport
    from java.lang.System import out  # @UnresolvedImport
    import java.lang.String  # @UnresolvedImport
    import org.python.core.PyDictionary  # @UnresolvedImport

__DBG = 0


def dbg(s):
    if __DBG:
        sys.stdout.write('%s\n' % (s,))


@pytest.mark.skipif(not IS_JYTHON, reason='Jython related test')
class TestMod(unittest.TestCase):

    def assert_args(self, tok, args, tips):
        for a in tips:
            if tok == a[0]:
                self.assertEqual(args, a[2])
                return
        raise AssertionError('%s not in %s', tok, tips)

    def assert_in(self, tok, tips):
        self.assertEqual(4, len(tips[0]))
        for a in tips:
            if tok == a[0]:
                return a
        s = ''
        for a in tips:
            s += str(a)
            s += '\n'
        raise AssertionError('%s not in %s' % (tok, s))

    def test_imports1a(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('java.util.HashMap')
        if f is None:
            return  # Not ok with java 9

        assert f.endswith('rt.jar')

    def test_imports1c(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('java.lang.Class')
        if f is None:
            return  # Not ok with java 9
        assert f.endswith('rt.jar')

    def test_imports1b(self):
        try:
            f, tip = _pydev_jy_imports_tipper.generate_tip('__builtin__.m')
            self.fail('err')
        except:
            pass

    def test_imports1(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('junit.framework.TestCase')
        assert f.endswith('junit.jar')
        ret = self.assert_in('assertEquals', tip)
#        self.assertEqual('', ret[2])

    def test_imports2(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('junit.framework')
        assert f.endswith('junit.jar')
        ret = self.assert_in('TestCase', tip)
        self.assertEqual('', ret[2])

    def test_imports2a(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('org.apache.tools.ant')
        assert f.endswith('ant.jar')
        ret = self.assert_in('Task', tip)
        self.assertEqual('', ret[2])

    def test_imports3(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('os')
        assert f.endswith('os.py')
        ret = self.assert_in('path', tip)
        self.assertEqual('', ret[2])

    def test_tip_on_string(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('string')
        self.assert_in('join', tip)
        self.assert_in('uppercase', tip)

    def test_imports(self):
        tip = _pydev_jy_imports_tipper.generate_tip('__builtin__')[1]
        self.assert_in('tuple'          , tip)
        self.assert_in('RuntimeError'   , tip)
        self.assert_in('RuntimeWarning' , tip)

    def test_imports5(self):
        f, tip = _pydev_jy_imports_tipper.generate_tip('java.lang')
        if f is None:
            return  # Not ok with java 9
        assert f.endswith('rt.jar')
        tup = self.assert_in('String' , tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_CLASS), tup[3])

        tip = _pydev_jy_imports_tipper.generate_tip('java')[1]
        tup = self.assert_in('lang' , tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_IMPORT), tup[3])

        tip = _pydev_jy_imports_tipper.generate_tip('java.lang.String')[1]
        tup = self.assert_in('indexOf'          , tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_FUNCTION), tup[3])

        tip = _pydev_jy_imports_tipper.generate_tip('java.lang.String')[1]
        tup = self.assert_in('charAt'          , tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_FUNCTION), tup[3])
        self.assertEqual('(int)', tup[2])

        tup = self.assert_in('format'          , tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_FUNCTION), tup[3])
        self.assertEqual('(string, objectArray)', tup[2])
        self.assertTrue(tup[1].find('[Ljava.lang.Object;') == -1)

        tup = self.assert_in('getBytes', tip)
        self.assertEqual(str(_pydev_jy_imports_tipper.TYPE_FUNCTION), tup[3])
        assert '[B' not in tup[1]
        assert 'byte[]' in tup[1]

        f, tip = _pydev_jy_imports_tipper.generate_tip('__builtin__.str')
        assert f is None or f.endswith('jython.jar')  # Depends on jython version
        self.assert_in('find'          , tip)

        f, tip = _pydev_jy_imports_tipper.generate_tip('__builtin__.dict')
        assert f is None or f.endswith('jython.jar')  # Depends on jython version
        self.assert_in('get'          , tip)


@pytest.mark.skipif(not IS_JYTHON, reason='Jython related test')
class TestSearch(unittest.TestCase):

    def test_search_on_jython(self):
        assert _pydev_jy_imports_tipper.search_definition('os')[0][0].split(os.sep)[-1] in ('javaos.py', 'os.py')
        self.assertEqual(0, _pydev_jy_imports_tipper.search_definition('os')[0][1])

        assert _pydev_jy_imports_tipper.search_definition('os.makedirs')[0][0].split(os.sep)[-1] in ('javaos.py', 'os.py')
        self.assertNotEqual(0, _pydev_jy_imports_tipper.search_definition('os.makedirs')[0][1])

        # print _pydev_jy_imports_tipper.search_definition('os.makedirs')


@pytest.mark.skipif(not IS_JYTHON, reason='Jython related test')
class TestCompl(unittest.TestCase):

    def test_getting_info_on_jython(self):

        dbg('\n\n--------------------------- java')
        assert not ismethod(java)[0]
        assert not isclass(java)
        assert _pydev_jy_imports_tipper.ismodule(java)

        dbg('\n\n--------------------------- java.lang')
        assert not ismethod(java.lang)[0]
        assert not isclass(java.lang)
        assert _pydev_jy_imports_tipper.ismodule(java.lang)

        dbg('\n\n--------------------------- Method')
        assert not ismethod(Method)[0]
        assert isclass(Method)

        dbg('\n\n--------------------------- System')
        assert not ismethod(System)[0]
        assert isclass(System)

        dbg('\n\n--------------------------- String')
        assert not ismethod(System)[0]
        assert isclass(String)
        assert len(dir_obj(String)) > 10

        dbg('\n\n--------------------------- arraycopy')
        isMet = ismethod(arraycopy)
        assert isMet[0]
        assert isMet[1][0].basic_as_str() == "function:arraycopy args=['java.lang.Object', 'int', 'java.lang.Object', 'int', 'int'], varargs=None, kwargs=None, docs:None"
        assert not isclass(arraycopy)

        dbg('\n\n--------------------------- out')
        isMet = ismethod(out)
        assert not isMet[0]
        assert not isclass(out)

        dbg('\n\n--------------------------- out.println')
        isMet = ismethod(out.println)  # @UndefinedVariable
        assert isMet[0]
        assert len(isMet[1]) == 10
        self.assertEqual(isMet[1][0].basic_as_str(), "function:println args=[], varargs=None, kwargs=None, docs:None")
        assert isMet[1][1].basic_as_str() == "function:println args=['long'], varargs=None, kwargs=None, docs:None"
        assert not isclass(out.println)  # @UndefinedVariable

        dbg('\n\n--------------------------- str')
        isMet = ismethod(str)
        # the code below should work, but is failing on jython 22a1
        # assert isMet[0]
        # assert isMet[1][0].basic_as_str() == "function:str args=['org.python.core.PyObject'], varargs=None, kwargs=None, docs:None"
        assert not isclass(str)

        def met1():
            a = 3
            return a

        dbg('\n\n--------------------------- met1')
        isMet = ismethod(met1)
        assert isMet[0]
        assert isMet[1][0].basic_as_str() == "function:met1 args=[], varargs=None, kwargs=None, docs:None"
        assert not isclass(met1)

        def met2(arg1, arg2, *vararg, **kwarg):
            '''docmet2'''

            a = 1
            return a

        dbg('\n\n--------------------------- met2')
        isMet = ismethod(met2)
        assert isMet[0]
        assert isMet[1][0].basic_as_str() == "function:met2 args=['arg1', 'arg2'], varargs=vararg, kwargs=kwarg, docs:docmet2"
        assert not isclass(met2)

# Run for jython in command line:

# On Windows:
# c:/bin/jython2.7.0/bin/jython.exe -Dpython.path=jython_test_deps/ant.jar;jython_test_deps/junit.jar -m py.test tests/test_jysimpleTipper.py

# On Linux (different path separator for jars)
# jython -Dpython.path=jython_test_deps/ant.jar:jython_test_deps/junit.jar -m py.test tests/test_jysimpleTipper.py
