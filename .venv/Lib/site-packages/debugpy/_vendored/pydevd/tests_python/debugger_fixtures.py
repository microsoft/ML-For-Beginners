# coding: utf-8
from contextlib import contextmanager
import os

import pytest

from tests_python import debugger_unittest
from tests_python.debugger_unittest import (get_free_port, overrides, IS_CPYTHON, IS_JYTHON, IS_IRONPYTHON,
    CMD_ADD_DJANGO_EXCEPTION_BREAK, CMD_REMOVE_DJANGO_EXCEPTION_BREAK,
    CMD_ADD_EXCEPTION_BREAK, wait_for_condition, IS_PYPY)
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding

import sys
from _pydevd_bundle.pydevd_constants import IS_WINDOWS


def get_java_location():
    from java.lang import System  # @UnresolvedImport
    jre_dir = System.getProperty("java.home")
    for f in [os.path.join(jre_dir, 'bin', 'java.exe'), os.path.join(jre_dir, 'bin', 'java')]:
        if os.path.exists(f):
            return f
    raise RuntimeError('Unable to find java executable')


def get_jython_jar():
    from java.lang import ClassLoader  # @UnresolvedImport
    cl = ClassLoader.getSystemClassLoader()
    paths = map(lambda url: url.getFile(), cl.getURLs())
    for p in paths:
        if 'jython.jar' in p:
            return p
    raise RuntimeError('Unable to find jython.jar')


class _WriterThreadCaseMSwitch(debugger_unittest.AbstractWriterThread):

    TEST_FILE = 'tests_python.resources._debugger_case_m_switch'
    IS_MODULE = True

    @overrides(debugger_unittest.AbstractWriterThread.get_environ)
    def get_environ(self):
        env = os.environ.copy()
        curr_pythonpath = env.get('PYTHONPATH', '')

        root_dirname = os.path.dirname(os.path.dirname(__file__))

        curr_pythonpath += root_dirname + os.pathsep
        env['PYTHONPATH'] = curr_pythonpath
        return env

    @overrides(debugger_unittest.AbstractWriterThread.get_main_filename)
    def get_main_filename(self):
        return debugger_unittest._get_debugger_test_file('_debugger_case_m_switch.py')


class _WriterThreadCaseModuleWithEntryPoint(_WriterThreadCaseMSwitch):

    TEST_FILE = 'tests_python.resources._debugger_case_module_entry_point:main'
    IS_MODULE = True

    @overrides(_WriterThreadCaseMSwitch.get_main_filename)
    def get_main_filename(self):
        return debugger_unittest._get_debugger_test_file('_debugger_case_module_entry_point.py')


class AbstractWriterThreadCaseFlask(debugger_unittest.AbstractWriterThread):

    FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True
    FLASK_FOLDER = None

    TEST_FILE = 'flask'
    IS_MODULE = True

    def write_add_breakpoint_jinja2(self, line, func, template):
        '''
            @param line: starts at 1
        '''
        assert self.FLASK_FOLDER is not None
        breakpoint_id = self.next_breakpoint_id()
        template_file = debugger_unittest._get_debugger_test_file(os.path.join(self.FLASK_FOLDER, 'templates', template))
        self.write("111\t%s\t%s\t%s\t%s\t%s\t%s\tNone\tNone" % (self.next_seq(), breakpoint_id, 'jinja2-line', template_file, line, func))
        self.log.append('write_add_breakpoint_jinja: %s line: %s func: %s' % (breakpoint_id, line, func))
        return breakpoint_id

    def write_add_exception_breakpoint_jinja2(self, exception='jinja2-Exception'):
        self.write('%s\t%s\t%s\t%s\t%s\t%s' % (CMD_ADD_EXCEPTION_BREAK, self.next_seq(), exception, 2, 0, 0))

    @overrides(debugger_unittest.AbstractWriterThread.get_environ)
    def get_environ(self):
        import platform

        env = os.environ.copy()
        env['FLASK_APP'] = 'app.py'
        env['FLASK_ENV'] = 'development'
        env['FLASK_DEBUG'] = '0'
        if platform.system() != 'Windows':
            locale = 'en_US.utf8' if platform.system() == 'Linux' else 'en_US.UTF-8'
            env.update({
                'LC_ALL': locale,
                'LANG': locale,
            })
        return env

    def get_cwd(self):
        return debugger_unittest._get_debugger_test_file(self.FLASK_FOLDER)

    def get_command_line_args(self):
        assert self.FLASK_FOLDER is not None
        free_port = get_free_port()
        self.flask_port = free_port
        return [
            'flask',
            'run',
             '--no-debugger',
             '--no-reload',
             '--with-threads',
            '--port',
            str(free_port),
        ]

    def _ignore_stderr_line(self, line):
        if debugger_unittest.AbstractWriterThread._ignore_stderr_line(self, line):
            return True

        if 'Running on http:' in line:
            return True

        if 'GET / HTTP/' in line:
            return True

        return False

    def create_request_thread(self, url=''):
        return debugger_unittest.AbstractWriterThread.create_request_thread(
            self, 'http://127.0.0.1:%s%s' % (self.flask_port, url))


class AbstractWriterThreadCaseDjango(debugger_unittest.AbstractWriterThread):

    FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True
    DJANGO_FOLDER = None

    def _ignore_stderr_line(self, line):
        if debugger_unittest.AbstractWriterThread._ignore_stderr_line(self, line):
            return True

        if 'GET /my_app' in line:
            return True

        return False

    def get_command_line_args(self):
        assert self.DJANGO_FOLDER is not None
        free_port = get_free_port()
        self.django_port = free_port
        return [
            debugger_unittest._get_debugger_test_file(os.path.join(self.DJANGO_FOLDER, 'manage.py')),
            'runserver',
            '--noreload',
            '--nothreading',
            str(free_port),
        ]

    def write_add_breakpoint_django(self, line, func, template):
        '''
            @param line: starts at 1
        '''
        assert self.DJANGO_FOLDER is not None
        breakpoint_id = self.next_breakpoint_id()
        template_file = debugger_unittest._get_debugger_test_file(os.path.join(self.DJANGO_FOLDER, 'my_app', 'templates', 'my_app', template))
        self.write("111\t%s\t%s\t%s\t%s\t%s\t%s\tNone\tNone" % (self.next_seq(), breakpoint_id, 'django-line', template_file, line, func))
        self.log.append('write_add_django_breakpoint: %s line: %s func: %s' % (breakpoint_id, line, func))
        return breakpoint_id

    def write_add_exception_breakpoint_django(self, exception='Exception'):
        self.write('%s\t%s\t%s' % (CMD_ADD_DJANGO_EXCEPTION_BREAK, self.next_seq(), exception))

    def write_remove_exception_breakpoint_django(self, exception='Exception'):
        self.write('%s\t%s\t%s' % (CMD_REMOVE_DJANGO_EXCEPTION_BREAK, self.next_seq(), exception))

    def create_request_thread(self, url=''):
        return debugger_unittest.AbstractWriterThread.create_request_thread(
            self, 'http://127.0.0.1:%s/%s' % (self.django_port, url))


class DebuggerRunnerSimple(debugger_unittest.DebuggerRunner):

    def get_command_line(self):
        if IS_JYTHON:
            if sys.executable is not None:
                # i.e.: we're running with the provided jython.exe
                return [sys.executable]
            else:

                return [
                    get_java_location(),
                    '-classpath',
                    get_jython_jar(),
                    'org.python.util.jython'
                ]

        if IS_CPYTHON or IS_PYPY:
            return [sys.executable, '-u']

        if IS_IRONPYTHON:
            return [
                    sys.executable,
                    '-X:Frames'
                ]

        raise RuntimeError('Unable to provide command line')


class DebuggerRunnerRemote(debugger_unittest.DebuggerRunner):

    def get_command_line(self):
        return [sys.executable, '-u']

    def add_command_line_args(self, args, dap=False):
        writer = self.writer

        ret = args + [self.writer.TEST_FILE]
        ret = writer.update_command_line_args(ret)  # Provide a hook for the writer
        return ret


@pytest.fixture
def debugger_runner_simple(tmpdir):
    return DebuggerRunnerSimple(tmpdir)


@pytest.fixture
def debugger_runner_remote(tmpdir):
    return DebuggerRunnerRemote(tmpdir)


@pytest.fixture
def case_setup(tmpdir, debugger_runner_simple):
    runner = debugger_runner_simple

    class WriterThread(debugger_unittest.AbstractWriterThread):
        pass

    class CaseSetup(object):

        check_non_ascii = False
        NON_ASCII_CHARS = u'áéíóú汉字'
        dap = False

        @contextmanager
        def test_file(
                self,
                filename,
                wait_for_port=True,
                wait_for_initialization=True,
                **kwargs
            ):
            import shutil
            filename = debugger_unittest._get_debugger_test_file(filename)
            if self.check_non_ascii:
                basedir = str(tmpdir)
                if isinstance(basedir, bytes):
                    basedir = basedir.decode('utf-8')
                if isinstance(filename, bytes):
                    filename = filename.decode('utf-8')

                new_dir = os.path.join(basedir, self.NON_ASCII_CHARS)
                os.makedirs(new_dir)

                new_filename = os.path.join(new_dir, self.NON_ASCII_CHARS + os.path.basename(filename))
                shutil.copyfile(filename, new_filename)
                filename = new_filename

            WriterThread.TEST_FILE = filename
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with runner.check_case(
                    WriterThread,
                    wait_for_port=wait_for_port,
                    wait_for_initialization=wait_for_initialization,
                    dap=self.dap
                ) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_dap(case_setup):
    case_setup.dap = True
    return case_setup


@pytest.fixture
def case_setup_unhandled_exceptions(case_setup):

    original = case_setup.test_file

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        # Don't call super as we have an expected exception
        if 'ValueError: TEST SUCEEDED' not in stderr:
            raise AssertionError('"ValueError: TEST SUCEEDED" not in stderr.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    def test_file(*args, **kwargs):
        kwargs.setdefault('check_test_suceeded_msg', check_test_suceeded_msg)
        kwargs.setdefault('additional_output_checks', additional_output_checks)
        return original(*args, **kwargs)

    case_setup.test_file = test_file

    return case_setup


@pytest.fixture
def case_setup_remote(debugger_runner_remote):

    class WriterThread(debugger_unittest.AbstractWriterThread):
        pass

    class CaseSetup(object):

        dap = False

        @contextmanager
        def test_file(
                self,
                filename,
                wait_for_port=True,
                access_token=None,
                client_access_token=None,
                append_command_line_args=(),
                **kwargs
            ):

            def update_command_line_args(writer, args):
                ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
                wait_for_condition(lambda: hasattr(writer, 'port'))
                ret.append(str(writer.port))

                if access_token is not None:
                    ret.append('--access-token')
                    ret.append(access_token)
                if client_access_token is not None:
                    ret.append('--client-access-token')
                    ret.append(client_access_token)

                if self.dap:
                    ret.append('--use-dap-mode')

                ret.extend(append_command_line_args)
                return ret

            WriterThread.TEST_FILE = debugger_unittest._get_debugger_test_file(filename)
            WriterThread.update_command_line_args = update_command_line_args
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with debugger_runner_remote.check_case(WriterThread, wait_for_port=wait_for_port) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_remote_dap(case_setup_remote):
    case_setup_remote.dap = True
    return case_setup_remote


@pytest.fixture
def case_setup_remote_attach_to_dap(debugger_runner_remote):
    '''
    The difference from this to case_setup_remote is that this one will connect to a server
    socket started by the debugger and case_setup_remote will create the server socket and wait
    for a connection from the debugger.
    '''

    class WriterThread(debugger_unittest.AbstractWriterThread):

        @overrides(debugger_unittest.AbstractWriterThread.run)
        def run(self):
            # I.e.: don't start socket on start(), rather, the test should call
            # start_socket_client() when needed.
            pass

    class CaseSetup(object):

        dap = True

        @contextmanager
        def test_file(
                self,
                filename,
                port,
                **kwargs
            ):
            additional_args = kwargs.pop('additional_args', [])

            def update_command_line_args(writer, args):
                ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
                ret.append(str(port))
                if self.dap:
                    ret.append('--use-dap-mode')
                ret.extend(additional_args)
                return ret

            WriterThread.TEST_FILE = debugger_unittest._get_debugger_test_file(filename)
            WriterThread.update_command_line_args = update_command_line_args
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with debugger_runner_remote.check_case(WriterThread, wait_for_port=False) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_multiprocessing(debugger_runner_simple):

    class WriterThread(debugger_unittest.AbstractWriterThread):
        pass

    class CaseSetup(object):

        dap = False

        @contextmanager
        def test_file(
                self,
                filename,
                **kwargs
            ):

            def update_command_line_args(writer, args):
                ret = debugger_unittest.AbstractWriterThread.update_command_line_args(writer, args)
                ret.insert(ret.index('--client'), '--multiprocess')
                if self.dap:
                    ret.insert(ret.index('--client'), '--debug-mode')
                    ret.insert(ret.index('--client'), 'debugpy-dap')
                    ret.insert(ret.index('--client'), '--json-dap-http')
                return ret

            WriterThread.update_command_line_args = update_command_line_args
            WriterThread.TEST_FILE = debugger_unittest._get_debugger_test_file(filename)
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with debugger_runner_simple.check_case(WriterThread) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_multiprocessing_dap(case_setup_multiprocessing):
    case_setup_multiprocessing.dap = True
    return case_setup_multiprocessing


@pytest.fixture
def case_setup_m_switch(debugger_runner_simple):

    class WriterThread(_WriterThreadCaseMSwitch):
        pass

    class CaseSetup(object):

        @contextmanager
        def test_file(self, **kwargs):
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)
            with debugger_runner_simple.check_case(WriterThread) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_m_switch_entry_point(debugger_runner_simple):

    runner = debugger_runner_simple

    class WriterThread(_WriterThreadCaseModuleWithEntryPoint):
        pass

    class CaseSetup(object):

        @contextmanager
        def test_file(self, **kwargs):
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)
            with runner.check_case(WriterThread) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_django(debugger_runner_simple):

    class WriterThread(AbstractWriterThreadCaseDjango):
        pass

    class CaseSetup(object):

        dap = False

        @contextmanager
        def test_file(self, **kwargs):
            import django
            version = [int(x) for x in django.get_version().split('.')][:2]
            if version == [1, 7]:
                django_folder = 'my_django_proj_17'
            elif version in ([2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1]):
                django_folder = 'my_django_proj_21'
            else:
                raise AssertionError('Can only check django 1.7 -> 4.1 right now. Found: %s' % (version,))

            WriterThread.DJANGO_FOLDER = django_folder
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with debugger_runner_simple.check_case(WriterThread, dap=self.dap) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_django_dap(case_setup_django):
    case_setup_django.dap = True
    return case_setup_django


@pytest.fixture
def case_setup_flask(debugger_runner_simple):

    class WriterThread(AbstractWriterThreadCaseFlask):
        pass

    class CaseSetup(object):

        dap = False

        @contextmanager
        def test_file(self, **kwargs):
            WriterThread.FLASK_FOLDER = 'flask1'
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            with debugger_runner_simple.check_case(WriterThread, dap=self.dap) as writer:
                yield writer

    return CaseSetup()


@pytest.fixture
def case_setup_flask_dap(case_setup_flask):
    case_setup_flask.dap = True
    return case_setup_flask
