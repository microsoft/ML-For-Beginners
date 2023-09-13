import os.path
import sys

IS_JYTHON = sys.platform.find('java') != -1

try:
    this_file_name = __file__
except NameError:
    # stupid jython. plain old __file__ isnt working for some reason
    import test_runfiles  # @UnresolvedImport - importing the module itself
    this_file_name = test_runfiles.__file__

desired_runfiles_path = os.path.normpath(os.path.dirname(this_file_name) + "/..")
sys.path.insert(0, desired_runfiles_path)

from _pydev_runfiles import pydev_runfiles_unittest
from _pydev_runfiles import pydev_runfiles_xml_rpc
from _pydevd_bundle import pydevd_io

# remove existing pydev_runfiles from modules (if any), so that we can be sure we have the correct version
if 'pydev_runfiles' in sys.modules:
    del sys.modules['pydev_runfiles']
if '_pydev_runfiles.pydev_runfiles' in sys.modules:
    del sys.modules['_pydev_runfiles.pydev_runfiles']

from _pydev_runfiles import pydev_runfiles
import unittest
import tempfile
import re

try:
    set
except:
    from sets import Set as set

# this is an early test because it requires the sys.path changed
orig_syspath = sys.path
a_file = pydev_runfiles.__file__
pydev_runfiles.PydevTestRunner(pydev_runfiles.Configuration(files_or_dirs=[a_file]))
file_dir = os.path.dirname(os.path.dirname(a_file))
assert file_dir in sys.path
sys.path = orig_syspath[:]

# remove it so that we leave it ok for other tests
sys.path.remove(desired_runfiles_path)


class RunfilesTest(unittest.TestCase):

    def _setup_scenario(
        self,
        path,
        include_tests=None,
        tests=None,
        files_to_tests=None,
        exclude_files=None,
        exclude_tests=None,
        include_files=None,
        ):
        self.MyTestRunner = pydev_runfiles.PydevTestRunner(
            pydev_runfiles.Configuration(
                files_or_dirs=path,
                include_tests=include_tests,
                verbosity=1,
                tests=tests,
                files_to_tests=files_to_tests,
                exclude_files=exclude_files,
                exclude_tests=exclude_tests,
                include_files=include_files,
            )
        )
        self.files = self.MyTestRunner.find_import_files()
        self.modules = self.MyTestRunner.find_modules_from_files(self.files)
        self.all_tests = self.MyTestRunner.find_tests_from_modules(self.modules)
        self.filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)

    def setUp(self):
        self.file_dir = [os.path.abspath(os.path.join(desired_runfiles_path, 'tests_runfiles/samples'))]
        self._setup_scenario(self.file_dir, None)

    def test_suite_used(self):
        for suite in self.all_tests + self.filtered_tests:
            self.assertTrue(isinstance(suite, pydev_runfiles_unittest.PydevTestSuite))

    def test_parse_cmdline(self):
        sys.argv = "pydev_runfiles.py ./".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual([sys.argv[1]], configuration.files_or_dirs)
        self.assertEqual(2, configuration.verbosity)  # default value
        self.assertEqual(None, configuration.include_tests)  # default value

        sys.argv = "pydev_runfiles.py ../images c:/temp".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual(sys.argv[1:3], configuration.files_or_dirs)
        self.assertEqual(2, configuration.verbosity)

        sys.argv = "pydev_runfiles.py --verbosity 3 ../junk c:/asdf ".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual(sys.argv[3:], configuration.files_or_dirs)
        self.assertEqual(int(sys.argv[2]), configuration.verbosity)

        sys.argv = "pydev_runfiles.py --include_tests test_def ./".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual([sys.argv[-1]], configuration.files_or_dirs)
        self.assertEqual([sys.argv[2]], configuration.include_tests)

        sys.argv = "pydev_runfiles.py --include_tests Abc.test_def,Mod.test_abc c:/junk/".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual([sys.argv[-1]], configuration.files_or_dirs)
        self.assertEqual(sys.argv[2].split(','), configuration.include_tests)

        sys.argv = ('C:\\eclipse-SDK-3.2-win32\\eclipse\\plugins\\org.python.pydev.debug_1.2.2\\pysrc\\pydev_runfiles.py ' +
                    '--verbosity 1 ' +
                    'C:\\workspace_eclipse\\fronttpa\\tests\\gui_tests\\calendar_popup_control_test.py ').split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual([sys.argv[-1]], configuration.files_or_dirs)
        self.assertEqual(1, configuration.verbosity)

        sys.argv = "pydev_runfiles.py --verbosity 1 --include_tests Mod.test_abc c:/junk/ ./".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual(sys.argv[5:], configuration.files_or_dirs)
        self.assertEqual(int(sys.argv[2]), configuration.verbosity)
        self.assertEqual([sys.argv[4]], configuration.include_tests)

        sys.argv = "pydev_runfiles.py --exclude_files=*.txt,a*.py".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual(['*.txt', 'a*.py'], configuration.exclude_files)

        sys.argv = "pydev_runfiles.py --exclude_tests=*__todo,test*bar".split()
        configuration = pydev_runfiles.parse_cmdline()
        self.assertEqual(['*__todo', 'test*bar'], configuration.exclude_tests)

    def test___adjust_python_path_works_for_directories(self):
        orig_syspath = sys.path
        tempdir = tempfile.gettempdir()
        pydev_runfiles.PydevTestRunner(pydev_runfiles.Configuration(files_or_dirs=[tempdir]))
        self.assertEqual(1, tempdir in sys.path)
        sys.path = orig_syspath[:]

    def test___is_valid_py_file(self):
        isvalid = self.MyTestRunner._PydevTestRunner__is_valid_py_file
        self.assertEqual(1, isvalid("test.py"))
        self.assertEqual(0, isvalid("asdf.pyc"))
        self.assertEqual(0, isvalid("__init__.py"))
        self.assertEqual(0, isvalid("__init__.pyc"))
        self.assertEqual(1, isvalid("asdf asdf.pyw"))

    def test___unixify(self):
        unixify = self.MyTestRunner._PydevTestRunner__unixify
        self.assertEqual("c:/temp/junk/asdf.py", unixify("c:SEPtempSEPjunkSEPasdf.py".replace('SEP', os.sep)))

    def test___importify(self):
        importify = self.MyTestRunner._PydevTestRunner__importify
        self.assertEqual("temp.junk.asdf", importify("temp/junk/asdf.py"))
        self.assertEqual("asdf", importify("asdf.py"))
        self.assertEqual("abc.def.hgi", importify("abc/def/hgi"))

    def test_finding_a_file_from_file_system(self):
        test_file = "simple_test.py"
        self.MyTestRunner.files_or_dirs = [self.file_dir[0] + test_file]
        files = self.MyTestRunner.find_import_files()
        self.assertEqual(1, len(files))
        self.assertEqual(files[0], self.file_dir[0] + test_file)

    def test_finding_files_in_dir_from_file_system(self):
        self.assertEqual(1, len(self.files) > 0)
        for import_file in self.files:
            self.assertEqual(-1, import_file.find(".pyc"))
            self.assertEqual(-1, import_file.find("__init__.py"))
            self.assertEqual(-1, import_file.find("\\"))
            self.assertEqual(-1, import_file.find(".txt"))

    def test___get_module_from_str(self):
        my_importer = self.MyTestRunner._PydevTestRunner__get_module_from_str
        my_os_path = my_importer("os.path", True, 'unused')
        from os import path
        import os.path as path2
        self.assertEqual(path, my_os_path)
        self.assertEqual(path2, my_os_path)
        self.assertNotEqual(__import__("os.path"), my_os_path)
        self.assertNotEqual(__import__("os"), my_os_path)

    def test_finding_modules_from_import_strings(self):
        self.assertEqual(1, len(self.modules) > 0)

    def test_finding_tests_when_no_filter(self):
        # unittest.py will create a TestCase with 0 tests in it
        # since it just imports what is given
        self.assertEqual(1, len(self.all_tests) > 0)
        files_with_tests = [1 for t in self.all_tests if len(t._tests) > 0]
        self.assertNotEqual(len(self.files), len(files_with_tests))

    def count_suite(self, tests=None):
        total = 0
        for t in tests:
            total += t.countTestCases()
        return total

    def test_runfile_imports(self):
        from _pydev_runfiles import pydev_runfiles_coverage
        from _pydev_runfiles import pydev_runfiles_parallel_client
        from _pydev_runfiles import pydev_runfiles_parallel
        import pytest
        from _pydev_runfiles import pydev_runfiles_pytest2
        from _pydev_runfiles import pydev_runfiles_unittest
        from _pydev_runfiles import pydev_runfiles_xml_rpc
        from _pydev_runfiles import pydev_runfiles

    def test___match(self):
        matcher = self.MyTestRunner._PydevTestRunner__match
        self.assertEqual(1, matcher(None, "aname"))
        self.assertEqual(1, matcher([".*"], "aname"))
        self.assertEqual(0, matcher(["^x$"], "aname"))
        self.assertEqual(0, matcher(["abc"], "aname"))
        self.assertEqual(1, matcher(["abc", "123"], "123"))

    def test_finding_tests_from_modules_with_bad_filter_returns_0_tests(self):
        self._setup_scenario(self.file_dir, ["NO_TESTS_ARE_SURE_TO_HAVE_THIS_NAME"])
        self.assertEqual(0, self.count_suite(self.all_tests))

    def test_finding_test_with_unique_name_returns_1_test(self):
        self._setup_scenario(self.file_dir, include_tests=["test_i_am_a_unique_test_name"])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests))

    def test_finding_test_with_non_unique_name(self):
        self._setup_scenario(self.file_dir, include_tests=["test_non_unique_name"])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests) > 2)

    def test_finding_tests_with_regex_filters(self):
        self._setup_scenario(self.file_dir, include_tests=["test_non*"])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests) > 2)

        self._setup_scenario(self.file_dir, ["^$"])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(0, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, exclude_tests=["*"])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(0, self.count_suite(filtered_tests))

    def test_matching_tests(self):
        self._setup_scenario(self.file_dir, None, ['StillYetAnotherSampleTest'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, ['SampleTest.test_xxxxxx1'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, ['SampleTest'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(8, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, ['AnotherSampleTest.todo_not_tested'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(1, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, ['StillYetAnotherSampleTest', 'SampleTest.test_xxxxxx1'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(2, self.count_suite(filtered_tests))

        self._setup_scenario(self.file_dir, None, exclude_tests=['*'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(self.count_suite(filtered_tests), 0)

        self._setup_scenario(self.file_dir, None, exclude_tests=['*a*'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(self.count_suite(filtered_tests), 6)

        self.assertEqual(
            set(self.MyTestRunner.list_test_names(filtered_tests)),
            set(['test_1', 'test_2', 'test_xxxxxx1', 'test_xxxxxx2', 'test_xxxxxx3', 'test_xxxxxx4'])
        )

        self._setup_scenario(self.file_dir, None, exclude_tests=['*a*', '*x*'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        self.assertEqual(self.count_suite(filtered_tests), 2)

        self.assertEqual(
            set(self.MyTestRunner.list_test_names(filtered_tests)),
            set(['test_1', 'test_2'])
        )

        self._setup_scenario(self.file_dir, None, exclude_files=['simple_test.py'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        names = self.MyTestRunner.list_test_names(filtered_tests)
        self.assertTrue('test_xxxxxx1' not in names, 'Found: %s' % (names,))

        self.assertEqual(
            set(['test_abc', 'test_non_unique_name', 'test_non_unique_name', 'test_asdf2', 'test_i_am_a_unique_test_name', 'test_non_unique_name', 'test_blank']),
            set(names)
        )

        self._setup_scenario(self.file_dir, None, include_files=['simple3_test.py'])
        filtered_tests = self.MyTestRunner.filter_tests(self.all_tests)
        names = self.MyTestRunner.list_test_names(filtered_tests)
        self.assertTrue('test_xxxxxx1' not in names, 'Found: %s' % (names,))

        self.assertEqual(
            set(['test_non_unique_name']),
            set(names)
        )

    def test_xml_rpc_communication(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'samples'))
        notifications = []

        class Server:

            def __init__(self, notifications):
                self.notifications = notifications

            def notifyConnected(self):
                # This method is called at the very start (in runfiles.py), and we do not check this here
                raise AssertionError('Should not be called from the run tests.')

            def notifyTestsCollected(self, number_of_tests):
                self.notifications.append(('notifyTestsCollected', number_of_tests))

            def notifyStartTest(self, file, test):
                pass

            def notifyTest(self, cond, captured_output, error_contents, file, test, time):
                try:
                    # I.e.: when marked as Binary in xml-rpc
                    captured_output = captured_output.data
                except:
                    pass
                try:
                    # I.e.: when marked as Binary in xml-rpc
                    error_contents = error_contents.data
                except:
                    pass
                if error_contents:
                    error_contents = error_contents.splitlines()[-1].strip()
                self.notifications.append(('notifyTest', cond, captured_output.strip(), error_contents, file, test))

            def notifyTestRunFinished(self, total_time):
                self.notifications.append(('notifyTestRunFinished',))

        server = Server(notifications)
        pydev_runfiles_xml_rpc.set_server(server)
        simple_test = os.path.join(self.file_dir[0], 'simple_test.py')
        simple_test2 = os.path.join(self.file_dir[0], 'simple2_test.py')
        simpleClass_test = os.path.join(self.file_dir[0], 'simpleClass_test.py')
        simpleModule_test = os.path.join(self.file_dir[0], 'simpleModule_test.py')

        files_to_tests = {}
        files_to_tests.setdefault(simple_test , []).append('SampleTest.test_xxxxxx1')
        files_to_tests.setdefault(simple_test , []).append('SampleTest.test_xxxxxx2')
        files_to_tests.setdefault(simple_test , []).append('SampleTest.test_non_unique_name')
        files_to_tests.setdefault(simple_test2, []).append('YetAnotherSampleTest.test_abc')
        files_to_tests.setdefault(simpleClass_test, []).append('SetUpClassTest.test_blank')
        files_to_tests.setdefault(simpleModule_test, []).append('SetUpModuleTest.test_blank')

        self._setup_scenario(None, files_to_tests=files_to_tests)
        self.MyTestRunner.verbosity = 2

        buf = pydevd_io.start_redirect(keep_original_redirection=False)
        try:
            self.MyTestRunner.run_tests()
            self.assertEqual(8, len(notifications))
            if sys.version_info[:2] <= (2, 6):
                # The setUpClass is not supported in Python 2.6 (thus we have no collection error).
                expected = [
                    ('notifyTest', 'fail', '', 'AssertionError: Fail test 2', simple_test, 'SampleTest.test_xxxxxx1'),
                    ('notifyTest', 'ok', '', '', simple_test2, 'YetAnotherSampleTest.test_abc'),
                    ('notifyTest', 'ok', '', '', simpleClass_test, 'SetUpClassTest.test_blank'),
                    ('notifyTest', 'ok', '', '', simpleModule_test, 'SetUpModuleTest.test_blank'),
                    ('notifyTest', 'ok', '', '', simple_test, 'SampleTest.test_xxxxxx2'),
                    ('notifyTest', 'ok', 'non unique name ran', '', simple_test, 'SampleTest.test_non_unique_name'),
                    ('notifyTestRunFinished',),
                    ('notifyTestsCollected', 6)
                ]
            else:
                expected = [
                        ('notifyTestsCollected', 6),
                        ('notifyTest', 'ok', 'non unique name ran', '', simple_test, 'SampleTest.test_non_unique_name'),
                        ('notifyTest', 'fail', '', 'AssertionError: Fail test 2', simple_test, 'SampleTest.test_xxxxxx1'),
                        ('notifyTest', 'ok', '', '', simple_test, 'SampleTest.test_xxxxxx2'),
                        ('notifyTest', 'ok', '', '', simple_test2, 'YetAnotherSampleTest.test_abc'),
                    ]

                if not IS_JYTHON:
                    if 'samples.simpleClass_test' in str(notifications):
                        expected.append(('notifyTest', 'error', '', 'ValueError: This is an INTENTIONAL value error in setUpClass.',
                                simpleClass_test.replace('/', os.path.sep), 'samples.simpleClass_test.SetUpClassTest <setUpClass>'))
                        expected.append(('notifyTest', 'error', '', 'ValueError: This is an INTENTIONAL value error in setUpModule.',
                                    simpleModule_test.replace('/', os.path.sep), 'samples.simpleModule_test <setUpModule>'))
                    else:
                        expected.append(('notifyTest', 'error', '', 'ValueError: This is an INTENTIONAL value error in setUpClass.',
                                simpleClass_test.replace('/', os.path.sep), 'simpleClass_test.SetUpClassTest <setUpClass>'))
                        expected.append(('notifyTest', 'error', '', 'ValueError: This is an INTENTIONAL value error in setUpModule.',
                                    simpleModule_test.replace('/', os.path.sep), 'simpleModule_test <setUpModule>'))
                else:
                    expected.append(('notifyTest', 'ok', '', '', simpleClass_test, 'SetUpClassTest.test_blank'))
                    expected.append(('notifyTest', 'ok', '', '', simpleModule_test, 'SetUpModuleTest.test_blank'))

                expected.append(('notifyTestRunFinished',))

            expected.sort()
            new_notifications = []
            for notification in expected:
                try:
                    if len(notification) == 6:
                        # Some are binary on Py3.
                        new_notifications.append((
                            notification[0],
                            notification[1],
                            notification[2].encode('latin1'),
                            notification[3].encode('latin1'),
                            notification[4],
                            notification[5],
                        ))
                    else:
                        new_notifications.append(notification)
                except:
                    raise
            expected = new_notifications

            notifications.sort()
            if not IS_JYTHON:
                self.assertEqual(
                    expected,
                    notifications
                )
        finally:
            pydevd_io.end_redirect()
        b = buf.getvalue()
        if sys.version_info[:2] > (2, 6):
            self.assertTrue(b.find('Ran 4 tests in ') != -1, 'Found: ' + b)
        else:
            self.assertTrue(b.find('Ran 6 tests in ') != -1, 'Found: ' + b)
