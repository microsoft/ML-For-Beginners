from __future__ import nested_scopes

import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time


#=======================================================================================================================
# Configuration
#=======================================================================================================================
class Configuration:

    def __init__(
        self,
        files_or_dirs='',
        verbosity=2,
        include_tests=None,
        tests=None,
        port=None,
        files_to_tests=None,
        jobs=1,
        split_jobs='tests',
        coverage_output_dir=None,
        coverage_include=None,
        coverage_output_file=None,
        exclude_files=None,
        exclude_tests=None,
        include_files=None,
        django=False,
        ):
        self.files_or_dirs = files_or_dirs
        self.verbosity = verbosity
        self.include_tests = include_tests
        self.tests = tests
        self.port = port
        self.files_to_tests = files_to_tests
        self.jobs = jobs
        self.split_jobs = split_jobs
        self.django = django

        if include_tests:
            assert isinstance(include_tests, (list, tuple))

        if exclude_files:
            assert isinstance(exclude_files, (list, tuple))

        if exclude_tests:
            assert isinstance(exclude_tests, (list, tuple))

        self.exclude_files = exclude_files
        self.include_files = include_files
        self.exclude_tests = exclude_tests

        self.coverage_output_dir = coverage_output_dir
        self.coverage_include = coverage_include
        self.coverage_output_file = coverage_output_file

    def __str__(self):
        return '''Configuration
 - files_or_dirs: %s
 - verbosity: %s
 - tests: %s
 - port: %s
 - files_to_tests: %s
 - jobs: %s
 - split_jobs: %s

 - include_files: %s
 - include_tests: %s

 - exclude_files: %s
 - exclude_tests: %s

 - coverage_output_dir: %s
 - coverage_include_dir: %s
 - coverage_output_file: %s

 - django: %s
''' % (
        self.files_or_dirs,
        self.verbosity,
        self.tests,
        self.port,
        self.files_to_tests,
        self.jobs,
        self.split_jobs,

        self.include_files,
        self.include_tests,

        self.exclude_files,
        self.exclude_tests,

        self.coverage_output_dir,
        self.coverage_include,
        self.coverage_output_file,

        self.django,
    )


#=======================================================================================================================
# parse_cmdline
#=======================================================================================================================
def parse_cmdline(argv=None):
    """
    Parses command line and returns test directories, verbosity, test filter and test suites

    usage:
        runfiles.py  -v|--verbosity <level>  -t|--tests <Test.test1,Test2>  dirs|files

    Multiprocessing options:
    jobs=number (with the number of jobs to be used to run the tests)
    split_jobs='module'|'tests'
        if == module, a given job will always receive all the tests from a module
        if == tests, the tests will be split independently of their originating module (default)

    --exclude_files  = comma-separated list of patterns with files to exclude (fnmatch style)
    --include_files = comma-separated list of patterns with files to include (fnmatch style)
    --exclude_tests = comma-separated list of patterns with test names to exclude (fnmatch style)

    Note: if --tests is given, --exclude_files, --include_files and --exclude_tests are ignored!
    """
    if argv is None:
        argv = sys.argv

    verbosity = 2
    include_tests = None
    tests = None
    port = None
    jobs = 1
    split_jobs = 'tests'
    files_to_tests = {}
    coverage_output_dir = None
    coverage_include = None
    exclude_files = None
    exclude_tests = None
    include_files = None
    django = False

    from _pydev_bundle._pydev_getopt import gnu_getopt
    optlist, dirs = gnu_getopt(
        argv[1:], "",
        [
            "verbosity=",
            "tests=",

            "port=",
            "config_file=",

            "jobs=",
            "split_jobs=",

            "include_tests=",
            "include_files=",

            "exclude_files=",
            "exclude_tests=",

            "coverage_output_dir=",
            "coverage_include=",

            "django="
        ]
    )

    for opt, value in optlist:
        if opt in ("-v", "--verbosity"):
            verbosity = value

        elif opt in ("-p", "--port"):
            port = int(value)

        elif opt in ("-j", "--jobs"):
            jobs = int(value)

        elif opt in ("-s", "--split_jobs"):
            split_jobs = value
            if split_jobs not in ('module', 'tests'):
                raise AssertionError('Expected split to be either "module" or "tests". Was :%s' % (split_jobs,))

        elif opt in ("-d", "--coverage_output_dir",):
            coverage_output_dir = value.strip()

        elif opt in ("-i", "--coverage_include",):
            coverage_include = value.strip()

        elif opt in ("-I", "--include_tests"):
            include_tests = value.split(',')

        elif opt in ("-E", "--exclude_files"):
            exclude_files = value.split(',')

        elif opt in ("-F", "--include_files"):
            include_files = value.split(',')

        elif opt in ("-e", "--exclude_tests"):
            exclude_tests = value.split(',')

        elif opt in ("-t", "--tests"):
            tests = value.split(',')

        elif opt in ("--django",):
            django = value.strip() in ['true', 'True', '1']

        elif opt in ("-c", "--config_file"):
            config_file = value.strip()
            if os.path.exists(config_file):
                f = open(config_file, 'r')
                try:
                    config_file_contents = f.read()
                finally:
                    f.close()

                if config_file_contents:
                    config_file_contents = config_file_contents.strip()

                if config_file_contents:
                    for line in config_file_contents.splitlines():
                        file_and_test = line.split('|')
                        if len(file_and_test) == 2:
                            file, test = file_and_test
                            if file in files_to_tests:
                                files_to_tests[file].append(test)
                            else:
                                files_to_tests[file] = [test]

            else:
                sys.stderr.write('Could not find config file: %s\n' % (config_file,))

    if type([]) != type(dirs):
        dirs = [dirs]

    ret_dirs = []
    for d in dirs:
        if '|' in d:
            # paths may come from the ide separated by |
            ret_dirs.extend(d.split('|'))
        else:
            ret_dirs.append(d)

    verbosity = int(verbosity)

    if tests:
        if verbosity > 4:
            sys.stdout.write('--tests provided. Ignoring --exclude_files, --exclude_tests and --include_files\n')
        exclude_files = exclude_tests = include_files = None

    config = Configuration(
        ret_dirs,
        verbosity,
        include_tests,
        tests,
        port,
        files_to_tests,
        jobs,
        split_jobs,
        coverage_output_dir,
        coverage_include,
        exclude_files=exclude_files,
        exclude_tests=exclude_tests,
        include_files=include_files,
        django=django,
    )

    if verbosity > 5:
        sys.stdout.write(str(config) + '\n')
    return config


#=======================================================================================================================
# PydevTestRunner
#=======================================================================================================================
class PydevTestRunner(object):
    """ finds and runs a file or directory of files as a unit test """

    __py_extensions = ["*.py", "*.pyw"]
    __exclude_files = ["__init__.*"]

    # Just to check that only this attributes will be written to this file
    __slots__ = [
        'verbosity',  # Always used

        'files_to_tests',  # If this one is given, the ones below are not used

        'files_or_dirs',  # Files or directories received in the command line
        'include_tests',  # The filter used to collect the tests
        'tests',  # Strings with the tests to be run

        'jobs',  # Integer with the number of jobs that should be used to run the test cases
        'split_jobs',  # String with 'tests' or 'module' (how should the jobs be split)

        'configuration',
        'coverage',
    ]

    def __init__(self, configuration):
        self.verbosity = configuration.verbosity

        self.jobs = configuration.jobs
        self.split_jobs = configuration.split_jobs

        files_to_tests = configuration.files_to_tests
        if files_to_tests:
            self.files_to_tests = files_to_tests
            self.files_or_dirs = list(files_to_tests.keys())
            self.tests = None
        else:
            self.files_to_tests = {}
            self.files_or_dirs = configuration.files_or_dirs
            self.tests = configuration.tests

        self.configuration = configuration
        self.__adjust_path()

    def __adjust_path(self):
        """ add the current file or directory to the python path """
        path_to_append = None
        for n in range(len(self.files_or_dirs)):
            dir_name = self.__unixify(self.files_or_dirs[n])
            if os.path.isdir(dir_name):
                if not dir_name.endswith("/"):
                    self.files_or_dirs[n] = dir_name + "/"
                path_to_append = os.path.normpath(dir_name)
            elif os.path.isfile(dir_name):
                path_to_append = os.path.dirname(dir_name)
            else:
                if not os.path.exists(dir_name):
                    block_line = '*' * 120
                    sys.stderr.write('\n%s\n* PyDev test runner error: %s does not exist.\n%s\n' % (block_line, dir_name, block_line))
                    return
                msg = ("unknown type. \n%s\nshould be file or a directory.\n" % (dir_name))
                raise RuntimeError(msg)
        if path_to_append is not None:
            # Add it as the last one (so, first things are resolved against the default dirs and
            # if none resolves, then we try a relative import).
            sys.path.append(path_to_append)

    def __is_valid_py_file(self, fname):
        """ tests that a particular file contains the proper file extension
            and is not in the list of files to exclude """
        is_valid_fname = 0
        for invalid_fname in self.__class__.__exclude_files:
            is_valid_fname += int(not fnmatch.fnmatch(fname, invalid_fname))
        if_valid_ext = 0
        for ext in self.__class__.__py_extensions:
            if_valid_ext += int(fnmatch.fnmatch(fname, ext))
        return is_valid_fname > 0 and if_valid_ext > 0

    def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, "/")

    def __importify(self, s, dir=False):
        """ turns directory separators into dots and removes the ".py*" extension
            so the string can be used as import statement """
        if not dir:
            dirname, fname = os.path.split(s)

            if fname.count('.') > 1:
                # if there's a file named xxx.xx.py, it is not a valid module, so, let's not load it...
                return

            imp_stmt_pieces = [dirname.replace("\\", "/").replace("/", "."), os.path.splitext(fname)[0]]

            if len(imp_stmt_pieces[0]) == 0:
                imp_stmt_pieces = imp_stmt_pieces[1:]

            return ".".join(imp_stmt_pieces)

        else:  # handle dir
            return s.replace("\\", "/").replace("/", ".")

    def __add_files(self, pyfiles, root, files):
        """ if files match, appends them to pyfiles. used by os.path.walk fcn """
        for fname in files:
            if self.__is_valid_py_file(fname):
                name_without_base_dir = self.__unixify(os.path.join(root, fname))
                pyfiles.append(name_without_base_dir)

    def find_import_files(self):
        """ return a list of files to import """
        if self.files_to_tests:
            pyfiles = self.files_to_tests.keys()
        else:
            pyfiles = []

            for base_dir in self.files_or_dirs:
                if os.path.isdir(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        # Note: handling directories that should be excluded from the search because
                        # they don't have __init__.py
                        exclude = {}
                        for d in dirs:
                            for init in ['__init__.py', '__init__.pyo', '__init__.pyc', '__init__.pyw', '__init__$py.class']:
                                if os.path.exists(os.path.join(root, d, init).replace('\\', '/')):
                                    break
                            else:
                                exclude[d] = 1

                        if exclude:
                            new = []
                            for d in dirs:
                                if d not in exclude:
                                    new.append(d)

                            dirs[:] = new

                        self.__add_files(pyfiles, root, files)

                elif os.path.isfile(base_dir):
                    pyfiles.append(base_dir)

        if self.configuration.exclude_files or self.configuration.include_files:
            ret = []
            for f in pyfiles:
                add = True
                basename = os.path.basename(f)
                if self.configuration.include_files:
                    add = False

                    for pat in self.configuration.include_files:
                        if fnmatch.fnmatchcase(basename, pat):
                            add = True
                            break

                if not add:
                    if self.verbosity > 3:
                        sys.stdout.write('Skipped file: %s (did not match any include_files pattern: %s)\n' % (f, self.configuration.include_files))

                elif self.configuration.exclude_files:
                    for pat in self.configuration.exclude_files:
                        if fnmatch.fnmatchcase(basename, pat):
                            if self.verbosity > 3:
                                sys.stdout.write('Skipped file: %s (matched exclude_files pattern: %s)\n' % (f, pat))

                            elif self.verbosity > 2:
                                sys.stdout.write('Skipped file: %s\n' % (f,))

                            add = False
                            break

                if add:
                    if self.verbosity > 3:
                        sys.stdout.write('Adding file: %s for test discovery.\n' % (f,))
                    ret.append(f)

            pyfiles = ret

        return pyfiles

    def __get_module_from_str(self, modname, print_exception, pyfile):
        """ Import the module in the given import path.
            * Returns the "final" module, so importing "coilib40.subject.visu"
            returns the "visu" module, not the "coilib40" as returned by __import__ """
        try:
            mod = __import__(modname)
            for part in modname.split('.')[1:]:
                mod = getattr(mod, part)
            return mod
        except:
            if print_exception:
                from _pydev_runfiles import pydev_runfiles_xml_rpc
                from _pydevd_bundle import pydevd_io
                buf_err = pydevd_io.start_redirect(keep_original_redirection=True, std='stderr')
                buf_out = pydevd_io.start_redirect(keep_original_redirection=True, std='stdout')
                try:
                    import traceback;traceback.print_exc()
                    sys.stderr.write('ERROR: Module: %s could not be imported (file: %s).\n' % (modname, pyfile))
                finally:
                    pydevd_io.end_redirect('stderr')
                    pydevd_io.end_redirect('stdout')

                pydev_runfiles_xml_rpc.notifyTest(
                    'error', buf_out.getvalue(), buf_err.getvalue(), pyfile, modname, 0)

            return None

    def remove_duplicates_keeping_order(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def find_modules_from_files(self, pyfiles):
        """ returns a list of modules given a list of files """
        # let's make sure that the paths we want are in the pythonpath...
        imports = [(s, self.__importify(s)) for s in pyfiles]

        sys_path = [os.path.normpath(path) for path in sys.path]
        sys_path = self.remove_duplicates_keeping_order(sys_path)

        system_paths = []
        for s in sys_path:
            system_paths.append(self.__importify(s, True))

        ret = []
        for pyfile, imp in imports:
            if imp is None:
                continue  # can happen if a file is not a valid module
            choices = []
            for s in system_paths:
                if imp.startswith(s):
                    add = imp[len(s) + 1:]
                    if add:
                        choices.append(add)
                    # sys.stdout.write(' ' + add + ' ')

            if not choices:
                sys.stdout.write('PYTHONPATH not found for file: %s\n' % imp)
            else:
                for i, import_str in enumerate(choices):
                    print_exception = i == len(choices) - 1
                    mod = self.__get_module_from_str(import_str, print_exception, pyfile)
                    if mod is not None:
                        ret.append((pyfile, mod, import_str))
                        break

        return ret

    #===================================================================================================================
    # GetTestCaseNames
    #===================================================================================================================
    class GetTestCaseNames:
        """Yes, we need a class for that (cannot use outer context on jython 2.1)"""

        def __init__(self, accepted_classes, accepted_methods):
            self.accepted_classes = accepted_classes
            self.accepted_methods = accepted_methods

        def __call__(self, testCaseClass):
            """Return a sorted sequence of method names found within testCaseClass"""
            testFnNames = []
            className = testCaseClass.__name__

            if className in self.accepted_classes:
                for attrname in dir(testCaseClass):
                    # If a class is chosen, we select all the 'test' methods'
                    if attrname.startswith('test') and hasattr(getattr(testCaseClass, attrname), '__call__'):
                        testFnNames.append(attrname)

            else:
                for attrname in dir(testCaseClass):
                    # If we have the class+method name, we must do a full check and have an exact match.
                    if className + '.' + attrname in self.accepted_methods:
                        if hasattr(getattr(testCaseClass, attrname), '__call__'):
                            testFnNames.append(attrname)

            # sorted() is not available in jython 2.1
            testFnNames.sort()
            return testFnNames

    def _decorate_test_suite(self, suite, pyfile, module_name):
        import unittest
        if isinstance(suite, unittest.TestSuite):
            add = False
            suite.__pydev_pyfile__ = pyfile
            suite.__pydev_module_name__ = module_name

            for t in suite._tests:
                t.__pydev_pyfile__ = pyfile
                t.__pydev_module_name__ = module_name
                if self._decorate_test_suite(t, pyfile, module_name):
                    add = True

            return add

        elif isinstance(suite, unittest.TestCase):
            return True

        else:
            return False

    def find_tests_from_modules(self, file_and_modules_and_module_name):
        """ returns the unittests given a list of modules """
        # Use our own suite!
        from _pydev_runfiles import pydev_runfiles_unittest
        import unittest
        unittest.TestLoader.suiteClass = pydev_runfiles_unittest.PydevTestSuite
        loader = unittest.TestLoader()

        ret = []
        if self.files_to_tests:
            for pyfile, m, module_name in file_and_modules_and_module_name:
                accepted_classes = {}
                accepted_methods = {}
                tests = self.files_to_tests[pyfile]
                for t in tests:
                    accepted_methods[t] = t

                loader.getTestCaseNames = self.GetTestCaseNames(accepted_classes, accepted_methods)

                suite = loader.loadTestsFromModule(m)
                if self._decorate_test_suite(suite, pyfile, module_name):
                    ret.append(suite)
            return ret

        if self.tests:
            accepted_classes = {}
            accepted_methods = {}

            for t in self.tests:
                splitted = t.split('.')
                if len(splitted) == 1:
                    accepted_classes[t] = t

                elif len(splitted) == 2:
                    accepted_methods[t] = t

            loader.getTestCaseNames = self.GetTestCaseNames(accepted_classes, accepted_methods)

        for pyfile, m, module_name in file_and_modules_and_module_name:
            suite = loader.loadTestsFromModule(m)
            if self._decorate_test_suite(suite, pyfile, module_name):
                ret.append(suite)

        return ret

    def filter_tests(self, test_objs, internal_call=False):
        """ based on a filter name, only return those tests that have
            the test case names that match """
        import unittest
        if not internal_call:
            if not self.configuration.include_tests and not self.tests and not self.configuration.exclude_tests:
                # No need to filter if we have nothing to filter!
                return test_objs

            if self.verbosity > 1:
                if self.configuration.include_tests:
                    sys.stdout.write('Tests to include: %s\n' % (self.configuration.include_tests,))

                if self.tests:
                    sys.stdout.write('Tests to run: %s\n' % (self.tests,))

                if self.configuration.exclude_tests:
                    sys.stdout.write('Tests to exclude: %s\n' % (self.configuration.exclude_tests,))

        test_suite = []
        for test_obj in test_objs:

            if isinstance(test_obj, unittest.TestSuite):
                # Note: keep the suites as they are and just 'fix' the tests (so, don't use the iter_tests).
                if test_obj._tests:
                    test_obj._tests = self.filter_tests(test_obj._tests, True)
                    if test_obj._tests:  # Only add the suite if we still have tests there.
                        test_suite.append(test_obj)

            elif isinstance(test_obj, unittest.TestCase):
                try:
                    testMethodName = test_obj._TestCase__testMethodName
                except AttributeError:
                    # changed in python 2.5
                    testMethodName = test_obj._testMethodName

                add = True
                if self.configuration.exclude_tests:
                    for pat in self.configuration.exclude_tests:
                        if fnmatch.fnmatchcase(testMethodName, pat):
                            if self.verbosity > 3:
                                sys.stdout.write('Skipped test: %s (matched exclude_tests pattern: %s)\n' % (testMethodName, pat))

                            elif self.verbosity > 2:
                                sys.stdout.write('Skipped test: %s\n' % (testMethodName,))

                            add = False
                            break

                if add:
                    if self.__match_tests(self.tests, test_obj, testMethodName):
                        include = True
                        if self.configuration.include_tests:
                            include = False
                            for pat in self.configuration.include_tests:
                                if fnmatch.fnmatchcase(testMethodName, pat):
                                    include = True
                                    break
                        if include:
                            test_suite.append(test_obj)
                        else:
                            if self.verbosity > 3:
                                sys.stdout.write('Skipped test: %s (did not match any include_tests pattern %s)\n' % (
                                    testMethodName, self.configuration.include_tests,))
        return test_suite

    def iter_tests(self, test_objs):
        # Note: not using yield because of Jython 2.1.
        import unittest
        tests = []
        for test_obj in test_objs:
            if isinstance(test_obj, unittest.TestSuite):
                tests.extend(self.iter_tests(test_obj._tests))

            elif isinstance(test_obj, unittest.TestCase):
                tests.append(test_obj)
        return tests

    def list_test_names(self, test_objs):
        names = []
        for tc in self.iter_tests(test_objs):
            try:
                testMethodName = tc._TestCase__testMethodName
            except AttributeError:
                # changed in python 2.5
                testMethodName = tc._testMethodName
            names.append(testMethodName)
        return names

    def __match_tests(self, tests, test_case, test_method_name):
        if not tests:
            return 1

        for t in tests:
            class_and_method = t.split('.')
            if len(class_and_method) == 1:
                # only class name
                if class_and_method[0] == test_case.__class__.__name__:
                    return 1

            elif len(class_and_method) == 2:
                if class_and_method[0] == test_case.__class__.__name__ and class_and_method[1] == test_method_name:
                    return 1

        return 0

    def __match(self, filter_list, name):
        """ returns whether a test name matches the test filter """
        if filter_list is None:
            return 1
        for f in filter_list:
            if re.match(f, name):
                return 1
        return 0

    def run_tests(self, handle_coverage=True):
        """ runs all tests """
        sys.stdout.write("Finding files... ")
        files = self.find_import_files()
        if self.verbosity > 3:
            sys.stdout.write('%s ... done.\n' % (self.files_or_dirs))
        else:
            sys.stdout.write('done.\n')
        sys.stdout.write("Importing test modules ... ")

        if handle_coverage:
            coverage_files, coverage = start_coverage_support(self.configuration)

        file_and_modules_and_module_name = self.find_modules_from_files(files)
        sys.stdout.write("done.\n")

        all_tests = self.find_tests_from_modules(file_and_modules_and_module_name)
        all_tests = self.filter_tests(all_tests)

        from _pydev_runfiles import pydev_runfiles_unittest
        test_suite = pydev_runfiles_unittest.PydevTestSuite(all_tests)
        from _pydev_runfiles import pydev_runfiles_xml_rpc
        pydev_runfiles_xml_rpc.notifyTestsCollected(test_suite.countTestCases())

        start_time = time.time()

        def run_tests():
            executed_in_parallel = False
            if self.jobs > 1:
                from _pydev_runfiles import pydev_runfiles_parallel

                # What may happen is that the number of jobs needed is lower than the number of jobs requested
                # (e.g.: 2 jobs were requested for running 1 test) -- in which case execute_tests_in_parallel will
                # return False and won't run any tests.
                executed_in_parallel = pydev_runfiles_parallel.execute_tests_in_parallel(
                    all_tests, self.jobs, self.split_jobs, self.verbosity, coverage_files, self.configuration.coverage_include)

            if not executed_in_parallel:
                # If in coverage, we don't need to pass anything here (coverage is already enabled for this execution).
                runner = pydev_runfiles_unittest.PydevTextTestRunner(stream=sys.stdout, descriptions=1, verbosity=self.verbosity)
                sys.stdout.write('\n')
                runner.run(test_suite)

        if self.configuration.django:
            get_django_test_suite_runner()(run_tests).run_tests([])
        else:
            run_tests()

        if handle_coverage:
            coverage.stop()
            coverage.save()

        total_time = 'Finished in: %.2f secs.' % (time.time() - start_time,)
        pydev_runfiles_xml_rpc.notifyTestRunFinished(total_time)


DJANGO_TEST_SUITE_RUNNER = None


def get_django_test_suite_runner():
    global DJANGO_TEST_SUITE_RUNNER
    if DJANGO_TEST_SUITE_RUNNER:
        return DJANGO_TEST_SUITE_RUNNER
    try:
        # django >= 1.8
        import django
        from django.test.runner import DiscoverRunner

        class MyDjangoTestSuiteRunner(DiscoverRunner):

            def __init__(self, on_run_suite):
                django.setup()
                DiscoverRunner.__init__(self)
                self.on_run_suite = on_run_suite

            def build_suite(self, *args, **kwargs):
                pass

            def suite_result(self, *args, **kwargs):
                pass

            def run_suite(self, *args, **kwargs):
                self.on_run_suite()

    except:
        # django < 1.8
        try:
            from django.test.simple import DjangoTestSuiteRunner
        except:

            class DjangoTestSuiteRunner:

                def __init__(self):
                    pass

                def run_tests(self, *args, **kwargs):
                    raise AssertionError("Unable to run suite with django.test.runner.DiscoverRunner nor django.test.simple.DjangoTestSuiteRunner because it couldn't be imported.")

        class MyDjangoTestSuiteRunner(DjangoTestSuiteRunner):

            def __init__(self, on_run_suite):
                DjangoTestSuiteRunner.__init__(self)
                self.on_run_suite = on_run_suite

            def build_suite(self, *args, **kwargs):
                pass

            def suite_result(self, *args, **kwargs):
                pass

            def run_suite(self, *args, **kwargs):
                self.on_run_suite()

    DJANGO_TEST_SUITE_RUNNER = MyDjangoTestSuiteRunner
    return DJANGO_TEST_SUITE_RUNNER


#=======================================================================================================================
# main
#=======================================================================================================================
def main(configuration):
    PydevTestRunner(configuration).run_tests()
