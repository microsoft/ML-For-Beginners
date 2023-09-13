'''
Entry point module (keep at root):

Used to run with tests with unittest/pytest/nose.
'''

import os


def main():
    import sys

    # Separate the nose params and the pydev params.
    pydev_params = []
    other_test_framework_params = []
    found_other_test_framework_param = None

    NOSE_PARAMS = '--nose-params'
    PY_TEST_PARAMS = '--py-test-params'

    for arg in sys.argv[1:]:
        if not found_other_test_framework_param and arg != NOSE_PARAMS and arg != PY_TEST_PARAMS:
            pydev_params.append(arg)

        else:
            if not found_other_test_framework_param:
                found_other_test_framework_param = arg
            else:
                other_test_framework_params.append(arg)

    try:
        # Convert to the case stored in the filesystem
        import win32api

        def get_with_filesystem_case(f):
            return win32api.GetLongPathName(win32api.GetShortPathName(f))

    except:

        def get_with_filesystem_case(f):
            return f

    # Here we'll run either with nose or with the pydev_runfiles.
    from _pydev_runfiles import pydev_runfiles
    from _pydev_runfiles import pydev_runfiles_xml_rpc
    from _pydevd_bundle import pydevd_constants
    from pydevd_file_utils import canonical_normalized_path

    DEBUG = 0
    if DEBUG:
        sys.stdout.write('Received parameters: %s\n' % (sys.argv,))
        sys.stdout.write('Params for pydev: %s\n' % (pydev_params,))
        if found_other_test_framework_param:
            sys.stdout.write('Params for test framework: %s, %s\n' % (found_other_test_framework_param, other_test_framework_params))

    try:
        configuration = pydev_runfiles.parse_cmdline([sys.argv[0]] + pydev_params)
    except:
        sys.stderr.write('Command line received: %s\n' % (sys.argv,))
        raise
    pydev_runfiles_xml_rpc.initialize_server(configuration.port)  # Note that if the port is None, a Null server will be initialized.

    NOSE_FRAMEWORK = "nose"
    PY_TEST_FRAMEWORK = "py.test"
    test_framework = None  # Default (pydev)
    try:
        if found_other_test_framework_param:
            if found_other_test_framework_param == NOSE_PARAMS:
                test_framework = NOSE_FRAMEWORK
                import nose

            elif found_other_test_framework_param == PY_TEST_PARAMS:
                test_framework = PY_TEST_FRAMEWORK
                import pytest

            else:
                raise ImportError('Test framework: %s not supported.' % (found_other_test_framework_param,))

        else:
            raise ImportError()

    except ImportError:
        if found_other_test_framework_param:
            raise

        test_framework = None

    # Clear any exception that may be there so that clients don't see it.
    # See: https://sourceforge.net/tracker/?func=detail&aid=3408057&group_id=85796&atid=577329
    if hasattr(sys, 'exc_clear'):
        sys.exc_clear()

    if not test_framework:

        return pydev_runfiles.main(configuration)  # Note: still doesn't return a proper value.

    else:
        # We'll convert the parameters to what nose or py.test expects.
        # The supported parameters are:
        # runfiles.py  --config-file|-t|--tests <Test.test1,Test2>  dirs|files --nose-params xxx yyy zzz
        # (all after --nose-params should be passed directly to nose)

        # In java:
        # --tests = Constants.ATTR_UNITTEST_TESTS
        # --config-file = Constants.ATTR_UNITTEST_CONFIGURATION_FILE

        # The only thing actually handled here are the tests that we want to run, which we'll
        # handle and pass as what the test framework expects.

        py_test_accept_filter = {}
        files_to_tests = configuration.files_to_tests

        if files_to_tests:
            # Handling through the file contents (file where each line is a test)
            files_or_dirs = []
            for file, tests in files_to_tests.items():
                if test_framework == NOSE_FRAMEWORK:
                    for test in tests:
                        files_or_dirs.append(file + ':' + test)

                elif test_framework == PY_TEST_FRAMEWORK:
                    py_test_accept_filter[file] = tests
                    py_test_accept_filter[canonical_normalized_path(file)] = tests
                    files_or_dirs.append(file)

                else:
                    raise AssertionError('Cannot handle test framework: %s at this point.' % (test_framework,))

        else:
            if configuration.tests:
                # Tests passed (works together with the files_or_dirs)
                files_or_dirs = []
                for file in configuration.files_or_dirs:
                    if test_framework == NOSE_FRAMEWORK:
                        for t in configuration.tests:
                            files_or_dirs.append(file + ':' + t)

                    elif test_framework == PY_TEST_FRAMEWORK:
                        py_test_accept_filter[file] = configuration.tests
                        py_test_accept_filter[canonical_normalized_path(file)] = configuration.tests
                        files_or_dirs.append(file)

                    else:
                        raise AssertionError('Cannot handle test framework: %s at this point.' % (test_framework,))
            else:
                # Only files or dirs passed (let it do the test-loading based on those paths)
                files_or_dirs = configuration.files_or_dirs

        argv = other_test_framework_params + files_or_dirs

        if test_framework == NOSE_FRAMEWORK:
            # Nose usage: http://somethingaboutorange.com/mrl/projects/nose/0.11.2/usage.html
            # show_stdout_option = ['-s']
            # processes_option = ['--processes=2']
            argv.insert(0, sys.argv[0])
            if DEBUG:
                sys.stdout.write('Final test framework args: %s\n' % (argv[1:],))

            from _pydev_runfiles import pydev_runfiles_nose
            PYDEV_NOSE_PLUGIN_SINGLETON = pydev_runfiles_nose.start_pydev_nose_plugin_singleton(configuration)
            argv.append('--with-pydevplugin')
            # Return 'not' because it will return 'success' (so, exit == 0 if success)
            return not nose.run(argv=argv, addplugins=[PYDEV_NOSE_PLUGIN_SINGLETON])

        elif test_framework == PY_TEST_FRAMEWORK:

            if '--coverage_output_dir' in pydev_params and '--coverage_include' in pydev_params:
                coverage_output_dir = pydev_params[pydev_params.index('--coverage_output_dir') + 1]
                coverage_include = pydev_params[pydev_params.index('--coverage_include') + 1]
                try:
                    import pytest_cov
                except ImportError:
                    sys.stderr.write('To do a coverage run with pytest the pytest-cov library is needed (i.e.: pip install pytest-cov).\n\n')
                    raise

                argv.insert(0, '--cov-append')
                argv.insert(1, '--cov-report=')
                argv.insert(2, '--cov=%s' % (coverage_include,))

                import time
                os.environ['COVERAGE_FILE'] = os.path.join(coverage_output_dir, '.coverage.%s' % (time.time(),))

            if DEBUG:
                sys.stdout.write('Final test framework args: %s\n' % (argv,))
                sys.stdout.write('py_test_accept_filter: %s\n' % (py_test_accept_filter,))

            def dotted(p):
                # Helper to convert path to have dots instead of slashes
                return os.path.normpath(p).replace(os.sep, "/").replace('/', '.')

            curr_dir = os.path.realpath('.')
            curr_dotted = dotted(curr_dir) + '.'

            # Overcome limitation on py.test:
            # When searching conftest if we have a structure as:
            # /my_package
            # /my_package/conftest.py
            # /my_package/tests
            # /my_package/tests/test_my_package.py
            # The test_my_package won't have access to the conftest contents from the
            # test_my_package.py file unless the working dir is set to /my_package.
            #
            # See related issue (for which we work-around below):
            # https://bitbucket.org/hpk42/pytest/issue/639/conftest-being-loaded-twice-giving

            for path in sys.path:
                path_dotted = dotted(path)
                if curr_dotted.startswith(path_dotted):
                    os.chdir(path)
                    break

            remove = []
            for i in range(len(argv)):
                arg = argv[i]
                # Workaround bug in py.test: if we pass the full path it ends up importing conftest
                # more than once (so, always work with relative paths).
                if os.path.isfile(arg) or os.path.isdir(arg):

                    # Args must be passed with the proper case in the filesystem (otherwise
                    # python itself may not recognize it).
                    arg = get_with_filesystem_case(arg)
                    argv[i] = arg

                    from os.path import relpath
                    try:
                        # May fail if on different drives
                        arg = relpath(arg)
                    except ValueError:
                        pass
                    else:
                        argv[i] = arg
                elif '<unable to get>' in arg:
                    remove.append(i)

            for i in reversed(remove):
                del argv[i]

            # To find our runfile helpers (i.e.: plugin)...
            d = os.path.dirname(__file__)
            if d not in sys.path:
                sys.path.insert(0, d)

            import pickle, zlib, base64

            # Update environment PYTHONPATH so that it finds our plugin if using xdist.
            os.environ['PYTHONPATH'] = os.pathsep.join(sys.path)

            # Set what should be skipped in the plugin through an environment variable
            s = base64.b64encode(zlib.compress(pickle.dumps(py_test_accept_filter)))
            s = s.decode('ascii')  # Must be str in py3.
            os.environ['PYDEV_PYTEST_SKIP'] = s

            # Identifies the main pid (i.e.: if it's not the main pid it has to connect back to the
            # main pid to give xml-rpc notifications).
            os.environ['PYDEV_MAIN_PID'] = str(os.getpid())
            os.environ['PYDEV_PYTEST_SERVER'] = str(configuration.port)

            argv.append('-p')
            argv.append('_pydev_runfiles.pydev_runfiles_pytest2')
            return pytest.main(argv)

        else:
            raise AssertionError('Cannot handle test framework: %s at this point.' % (test_framework,))


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            # The server is not a daemon thread, so, we have to ask for it to be killed!
            from _pydev_runfiles import pydev_runfiles_xml_rpc
            pydev_runfiles_xml_rpc.force_server_kill()
        except:
            pass  # Ignore any errors here

    import sys
    import threading
    if hasattr(sys, '_current_frames') and hasattr(threading, 'enumerate'):
        import time
        import traceback

        class DumpThreads(threading.Thread):

            def run(self):
                time.sleep(10)

                thread_id_to_name = {}
                try:
                    for t in threading.enumerate():
                        thread_id_to_name[t.ident] = '%s  (daemon: %s)' % (t.name, t.daemon)
                except:
                    pass

                stack_trace = [
                    '===============================================================================',
                    'pydev pyunit runner: Threads still found running after tests finished',
                    '================================= Thread Dump =================================']

                for thread_id, stack in sys._current_frames().items():
                    stack_trace.append('\n-------------------------------------------------------------------------------')
                    stack_trace.append(" Thread %s" % thread_id_to_name.get(thread_id, thread_id))
                    stack_trace.append('')

                    if 'self' in stack.f_locals:
                        sys.stderr.write(str(stack.f_locals['self']) + '\n')

                    for filename, lineno, name, line in traceback.extract_stack(stack):
                        stack_trace.append(' File "%s", line %d, in %s' % (filename, lineno, name))
                        if line:
                            stack_trace.append("   %s" % (line.strip()))
                stack_trace.append('\n=============================== END Thread Dump ===============================')
                sys.stderr.write('\n'.join(stack_trace))

        dump_current_frames_thread = DumpThreads()
        dump_current_frames_thread.daemon = True  # Daemon so that this thread doesn't halt it!
        dump_current_frames_thread.start()
