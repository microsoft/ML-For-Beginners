from collections import namedtuple
from contextlib import contextmanager
import json
from urllib.parse import quote, quote_plus, unquote_plus

import re
import socket
import subprocess
import threading
import time
import traceback
from tests_python.debug_constants import *

from _pydev_bundle import pydev_localhost, pydev_log

# Note: copied (don't import because we want it to be independent on the actual code because of backward compatibility).
CMD_RUN = 101
CMD_LIST_THREADS = 102
CMD_THREAD_CREATE = 103
CMD_THREAD_KILL = 104
CMD_THREAD_SUSPEND = 105
CMD_THREAD_RUN = 106
CMD_STEP_INTO = 107
CMD_STEP_OVER = 108
CMD_STEP_RETURN = 109
CMD_GET_VARIABLE = 110
CMD_SET_BREAK = 111
CMD_REMOVE_BREAK = 112
CMD_EVALUATE_EXPRESSION = 113
CMD_GET_FRAME = 114
CMD_EXEC_EXPRESSION = 115
CMD_WRITE_TO_CONSOLE = 116
CMD_CHANGE_VARIABLE = 117
CMD_RUN_TO_LINE = 118
CMD_RELOAD_CODE = 119
CMD_GET_COMPLETIONS = 120

# Note: renumbered (conflicted on merge)
CMD_CONSOLE_EXEC = 121
CMD_ADD_EXCEPTION_BREAK = 122
CMD_REMOVE_EXCEPTION_BREAK = 123
CMD_LOAD_SOURCE = 124
CMD_ADD_DJANGO_EXCEPTION_BREAK = 125
CMD_REMOVE_DJANGO_EXCEPTION_BREAK = 126
CMD_SET_NEXT_STATEMENT = 127
CMD_SMART_STEP_INTO = 128
CMD_EXIT = 129
CMD_SIGNATURE_CALL_TRACE = 130

CMD_SET_PY_EXCEPTION = 131
CMD_GET_FILE_CONTENTS = 132
CMD_SET_PROPERTY_TRACE = 133
# Pydev debug console commands
CMD_EVALUATE_CONSOLE_EXPRESSION = 134
CMD_RUN_CUSTOM_OPERATION = 135
CMD_GET_BREAKPOINT_EXCEPTION = 136
CMD_STEP_CAUGHT_EXCEPTION = 137
CMD_SEND_CURR_EXCEPTION_TRACE = 138
CMD_SEND_CURR_EXCEPTION_TRACE_PROCEEDED = 139
CMD_IGNORE_THROWN_EXCEPTION_AT = 140
CMD_ENABLE_DONT_TRACE = 141
CMD_SHOW_CONSOLE = 142

CMD_GET_ARRAY = 143
CMD_STEP_INTO_MY_CODE = 144
CMD_GET_CONCURRENCY_EVENT = 145
CMD_SHOW_RETURN_VALUES = 146

CMD_GET_THREAD_STACK = 152
CMD_THREAD_DUMP_TO_STDERR = 153  # This is mostly for unit-tests to diagnose errors on ci.
CMD_STOP_ON_START = 154
CMD_GET_EXCEPTION_DETAILS = 155
CMD_PYDEVD_JSON_CONFIG = 156

CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION = 157
CMD_THREAD_RESUME_SINGLE_NOTIFICATION = 158

CMD_STEP_OVER_MY_CODE = 159
CMD_STEP_RETURN_MY_CODE = 160

CMD_SET_PY_EXCEPTION = 161
CMD_SET_PATH_MAPPING_JSON = 162

CMD_GET_SMART_STEP_INTO_VARIANTS = 163  # XXX: PyCharm has 160 for this (we're currently incompatible anyways).

CMD_REDIRECT_OUTPUT = 200
CMD_GET_NEXT_STATEMENT_TARGETS = 201
CMD_SET_PROJECT_ROOTS = 202

CMD_AUTHENTICATE = 205

CMD_VERSION = 501
CMD_RETURN = 502
CMD_SET_PROTOCOL = 503
CMD_ERROR = 901

REASON_CAUGHT_EXCEPTION = CMD_STEP_CAUGHT_EXCEPTION
REASON_UNCAUGHT_EXCEPTION = CMD_ADD_EXCEPTION_BREAK
REASON_STOP_ON_BREAKPOINT = CMD_SET_BREAK
REASON_THREAD_SUSPEND = CMD_THREAD_SUSPEND
REASON_STEP_INTO = CMD_STEP_INTO
REASON_STEP_INTO_MY_CODE = CMD_STEP_INTO_MY_CODE
REASON_STOP_ON_START = CMD_STOP_ON_START
REASON_STEP_RETURN = CMD_STEP_RETURN
REASON_STEP_RETURN_MY_CODE = CMD_STEP_RETURN_MY_CODE
REASON_STEP_OVER = CMD_STEP_OVER
REASON_STEP_OVER_MY_CODE = CMD_STEP_OVER_MY_CODE

# Always True (because otherwise when we do have an error, it's hard to diagnose).
SHOW_WRITES_AND_READS = True
SHOW_OTHER_DEBUG_INFO = True
SHOW_STDOUT = True

import platform

IS_CPYTHON = platform.python_implementation() == 'CPython'
IS_IRONPYTHON = platform.python_implementation() == 'IronPython'
IS_JYTHON = platform.python_implementation() == 'Jython'
IS_PYPY = platform.python_implementation() == 'PyPy'
IS_APPVEYOR = os.environ.get('APPVEYOR', '') in ('True', 'true', '1')

try:
    from thread import start_new_thread
except ImportError:
    from _thread import start_new_thread  # @UnresolvedImport

Hit = namedtuple('Hit', 'thread_id, frame_id, line, suspend_type, name, file')


def overrides(method):
    '''
    Helper to check that one method overrides another (redeclared in unit-tests to avoid importing pydevd).
    '''

    def wrapper(func):
        if func.__name__ != method.__name__:
            msg = "Wrong @override: %r expected, but overwriting %r."
            msg = msg % (func.__name__, method.__name__)
            raise AssertionError(msg)

        if func.__doc__ is None:
            func.__doc__ = method.__doc__

        return func

    return wrapper


TIMEOUT = 20

try:
    TimeoutError = TimeoutError  # @ReservedAssignment
except NameError:

    class TimeoutError(RuntimeError):  # @ReservedAssignment
        pass


def wait_for_condition(condition, msg=None, timeout=TIMEOUT, sleep=.05):
    curtime = time.time()
    while True:
        if condition():
            break
        if time.time() - curtime > timeout:
            error_msg = 'Condition not reached in %s seconds' % (timeout,)
            if msg is not None:
                error_msg += '\n'
                if callable(msg):
                    error_msg += msg()
                else:
                    error_msg += str(msg)

            raise TimeoutError(error_msg)
        time.sleep(sleep)


class IgnoreFailureError(RuntimeError):
    pass


#=======================================================================================================================
# ReaderThread
#=======================================================================================================================
class ReaderThread(threading.Thread):

    MESSAGES_TIMEOUT = 15

    def __init__(self, sock):
        threading.Thread.__init__(self)
        self.name = 'Test Reader Thread'
        try:
            from queue import Queue
        except ImportError:
            from Queue import Queue

        self.daemon = True
        self._buffer = b''
        self.sock = sock
        self._queue = Queue()
        self._kill = False
        self.accept_xml_messages = True
        self.on_message_found = lambda msg: None

    def set_messages_timeout(self, timeout):
        self.MESSAGES_TIMEOUT = timeout

    def get_next_message(self, context_message, timeout=None):
        if timeout is None:
            timeout = self.MESSAGES_TIMEOUT
        try:
            msg = self._queue.get(block=True, timeout=timeout)
            self.on_message_found(msg)
        except:
            raise TimeoutError('No message was written in %s seconds. Error message:\n%s' % (timeout, context_message,))
        else:
            frame = sys._getframe().f_back.f_back
            frame_info = ''
            while frame:
                if not frame.f_code.co_name.startswith('test_'):
                    frame = frame.f_back
                    continue

                if frame.f_code.co_filename.endswith('debugger_unittest.py'):
                    frame = frame.f_back
                    continue

                stack_msg = ' --  File "%s", line %s, in %s\n' % (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                if 'run' == frame.f_code.co_name:
                    frame_info = stack_msg  # Ok, found the writer thread 'run' method (show only that).
                    break
                frame_info += stack_msg
                frame = frame.f_back
                # Just print the first which is not debugger_unittest.py
                break

            frame = None
            sys.stdout.write('Message returned in get_next_message(): %s --  ctx: %s, asked at:\n%s\n' % (unquote_plus(unquote_plus(msg)), context_message, frame_info))

        if not self.accept_xml_messages:
            if '<xml' in msg:
                raise AssertionError('Xml messages disabled. Received: %s' % (msg,))
        return msg

    def _read(self, size):
        while True:
            buffer_len = len(self._buffer)
            if buffer_len == size:
                ret = self._buffer
                self._buffer = b''
                return ret

            if buffer_len > size:
                ret = self._buffer[:size]
                self._buffer = self._buffer[size:]
                return ret

            r = self.sock.recv(max(size - buffer_len, 1024))
            if not r:
                return b''
            self._buffer += r

    def _read_line(self):
        while True:
            i = self._buffer.find(b'\n')
            if i != -1:
                i += 1  # Add the newline to the return
                ret = self._buffer[:i]
                self._buffer = self._buffer[i:]
                return ret
            else:
                r = self.sock.recv(1024)
                if not r:
                    return b''
                self._buffer += r

    def run(self):
        try:
            content_len = -1

            while not self._kill:
                line = self._read_line()

                if not line:
                    break

                if SHOW_WRITES_AND_READS:
                    show_line = line
                    show_line = line.decode('utf-8')

                    print('%s Received %s' % (self.name, show_line,))

                if line.startswith(b'Content-Length:'):
                    content_len = int(line.strip().split(b':', 1)[1])
                    continue

                if content_len != -1:
                    # If we previously received a content length, read until a '\r\n'.
                    if line == b'\r\n':
                        json_contents = self._read(content_len)
                        content_len = -1

                        if len(json_contents) == 0:
                            self.handle_except()
                            return  # Finished communication.

                        msg = json_contents
                        msg = msg.decode('utf-8')
                        print('Test Reader Thread Received %s' % (msg,))
                        self._queue.put(msg)

                    continue
                else:
                    # No content len, regular line-based protocol message (remove trailing new-line).
                    if line.endswith(b'\n\n'):
                        line = line[:-2]

                    elif line.endswith(b'\n'):
                        line = line[:-1]

                    elif line.endswith(b'\r'):
                        line = line[:-1]

                    msg = line
                    msg = msg.decode('utf-8')
                    print('Test Reader Thread Received %s' % (msg,))
                    self._queue.put(msg)

        except:
            pass  # ok, finished it
        finally:
            # When the socket from pydevd is closed the client should shutdown to notify
            # it acknowledged it.
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.sock.close()
            except:
                pass

    def do_kill(self):
        self._kill = True
        if hasattr(self, 'sock'):
            from socket import SHUT_RDWR
            try:
                self.sock.shutdown(SHUT_RDWR)
            except:
                pass
            try:
                self.sock.close()
            except:
                pass
            delattr(self, 'sock')


def read_process(stream, buffer, debug_stream, stream_name, finish):
    while True:
        line = stream.readline()
        if not line:
            break

        line = line.decode('utf-8', errors='replace')

        if SHOW_STDOUT:
            debug_stream.write('%s: %s' % (stream_name, line,))
        buffer.append(line)

        if finish[0]:
            return


def start_in_daemon_thread(target, args):
    t0 = threading.Thread(target=target, args=args)
    t0.daemon = True
    t0.start()


class DebuggerRunner(object):

    def __init__(self, tmpdir):
        if tmpdir is not None:
            self.pydevd_debug_file = os.path.join(str(tmpdir), 'pydevd_debug_file_%s.txt' % (os.getpid(),))
        else:
            self.pydevd_debug_file = None

    def get_command_line(self):
        '''
        Returns the base command line (i.e.: ['python.exe', '-u'])
        '''
        raise NotImplementedError

    def add_command_line_args(self, args, dap=False):
        writer = self.writer
        port = int(writer.port)

        localhost = pydev_localhost.get_localhost()
        ret = [
            writer.get_pydevd_file(),
        ]

        if not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON:
            # i.e.: in frame-eval mode we support native threads, whereas
            # on other cases we need the qt monkeypatch.
            ret += ['--qt-support']

        ret += [
            '--client',
            localhost,
            '--port',
            str(port),
        ]

        if dap:
            ret += ['--debug-mode', 'debugpy-dap']
            ret += ['--json-dap-http']

        if writer.IS_MODULE:
            ret += ['--module']

        ret += ['--file'] + writer.get_command_line_args()
        ret = writer.update_command_line_args(ret)  # Provide a hook for the writer
        return args + ret

    @contextmanager
    def check_case(self, writer_class, wait_for_port=True, wait_for_initialization=True, dap=False):
        try:
            if callable(writer_class):
                writer = writer_class()
            else:
                writer = writer_class
            try:
                writer.start()
                if wait_for_port:
                    wait_for_condition(lambda: hasattr(writer, 'port'))
                self.writer = writer

                args = self.get_command_line()

                args = self.add_command_line_args(args, dap=dap)

                if SHOW_OTHER_DEBUG_INFO:
                    print('executing: %s' % (' '.join(args),))

                with self.run_process(args, writer) as dct_with_stdout_stder:
                    try:
                        if not wait_for_initialization:
                            # The use-case for this is that the debugger can't even start-up in this
                            # scenario, as such, sleep a bit so that the output can be collected.
                            time.sleep(1)
                        elif wait_for_port:
                            wait_for_condition(lambda: writer.finished_initialization)
                    except TimeoutError:
                        sys.stderr.write('Timed out waiting for initialization\n')
                        sys.stderr.write('stdout:\n%s\n\nstderr:\n%s\n' % (
                            ''.join(dct_with_stdout_stder['stdout']),
                            ''.join(dct_with_stdout_stder['stderr']),
                        ))
                        raise
                    finally:
                        writer.get_stdout = lambda: ''.join(dct_with_stdout_stder['stdout'])
                        writer.get_stderr = lambda: ''.join(dct_with_stdout_stder['stderr'])

                    yield writer
            finally:
                writer.do_kill()
                writer.log = []

            stdout = dct_with_stdout_stder['stdout']
            stderr = dct_with_stdout_stder['stderr']
            writer.additional_output_checks(''.join(stdout), ''.join(stderr))
        except IgnoreFailureError:
            sys.stderr.write('Test finished with ignored failure.\n')
            return

    def create_process(self, args, writer):
        env = writer.get_environ() if writer is not None else None
        if env is None:
            env = os.environ.copy()

        if self.pydevd_debug_file:
            env['PYDEVD_DEBUG'] = 'True'
            env['PYDEVD_DEBUG_FILE'] = self.pydevd_debug_file
            print('Logging to: %s' % (self.pydevd_debug_file,))
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=writer.get_cwd() if writer is not None else '.',
            env=env,
        )
        return process

    @contextmanager
    def run_process(self, args, writer):
        process = self.create_process(args, writer)
        writer.process = process
        stdout = []
        stderr = []
        finish = [False]
        dct_with_stdout_stder = {}
        fail_with_message = False

        try:
            start_in_daemon_thread(read_process, (process.stdout, stdout, sys.stdout, 'stdout', finish))
            start_in_daemon_thread(read_process, (process.stderr, stderr, sys.stderr, 'stderr', finish))

            if SHOW_OTHER_DEBUG_INFO:
                print('Both processes started')

            # polls can fail (because the process may finish and the thread still not -- so, we give it some more chances to
            # finish successfully).
            initial_time = time.time()
            shown_intermediate = False
            dumped_threads = False

            dct_with_stdout_stder['stdout'] = stdout
            dct_with_stdout_stder['stderr'] = stderr
            try:
                yield dct_with_stdout_stder
            except:
                fail_with_message = True
                # Let's print the actuayl exception here (it doesn't appear properly on Python 2 and
                # on Python 3 it's hard to find because pytest output is too verbose).
                sys.stderr.write('***********\n')
                sys.stderr.write('***********\n')
                sys.stderr.write('***********\n')
                traceback.print_exc()
                sys.stderr.write('***********\n')
                sys.stderr.write('***********\n')
                sys.stderr.write('***********\n')
                raise

            if not writer.finished_ok:
                self.fail_with_message(
                    "The thread that was doing the tests didn't finish successfully (writer.finished_ok = True not set).",
                    stdout,
                    stderr,
                    writer
                )

            while True:
                if process.poll() is not None:
                    if writer.EXPECTED_RETURNCODE != 'any':
                        expected_returncode = writer.EXPECTED_RETURNCODE
                        if not isinstance(expected_returncode, (list, tuple)):
                            expected_returncode = (expected_returncode,)

                        if process.returncode not in expected_returncode:
                            self.fail_with_message('Expected process.returncode to be %s. Found: %s' % (
                                writer.EXPECTED_RETURNCODE, process.returncode), stdout, stderr, writer)
                    break
                else:
                    if writer is not None:
                        if writer.FORCE_KILL_PROCESS_WHEN_FINISHED_OK:
                            process.kill()
                            continue

                        if not shown_intermediate and (time.time() - initial_time > (TIMEOUT / 3.)):  # 1/3 of timeout
                            print('Warning: writer thread exited and process still did not (%.2f seconds elapsed).' % (time.time() - initial_time,))
                            shown_intermediate = True

                        if time.time() - initial_time > ((TIMEOUT / 3.) * 2.):  # 2/3 of timeout
                            if not dumped_threads:
                                dumped_threads = True
                                # It still didn't finish. Ask for a thread dump
                                # (we'll be able to see it later on the test output stderr).
                                try:
                                    writer.write_dump_threads()
                                except:
                                    traceback.print_exc()

                        if time.time() - initial_time > TIMEOUT:  # timed out
                            process.kill()
                            time.sleep(.2)
                            self.fail_with_message(
                                "The other process should've exited but still didn't (%.2f seconds timeout for process to exit)." % (time.time() - initial_time,),
                                stdout, stderr, writer
                            )
                time.sleep(.2)

            if writer is not None:
                if not writer.FORCE_KILL_PROCESS_WHEN_FINISHED_OK:
                    if stdout is None:
                        self.fail_with_message(
                            "The other process may still be running -- and didn't give any output.", stdout, stderr, writer)

                    check = 0
                    while not writer.check_test_suceeded_msg(stdout, stderr):
                        check += 1
                        if check == 50:
                            self.fail_with_message("TEST SUCEEDED not found.", stdout, stderr, writer)
                        time.sleep(.1)

        except TimeoutError:
            msg = 'TimeoutError'
            try:
                writer.write_dump_threads()
            except:
                msg += ' (note: error trying to dump threads on timeout).'
            time.sleep(.2)
            self.fail_with_message(msg, stdout, stderr, writer)
        except Exception as e:
            if fail_with_message:
                self.fail_with_message(str(e), stdout, stderr, writer)
            else:
                raise
        finally:
            try:
                if process.poll() is None:
                    process.kill()
            except:
                traceback.print_exc()
            finish[0] = True

    def fail_with_message(self, msg, stdout, stderr, writerThread):
        log_contents = ''
        for f in pydev_log.list_log_files(self.pydevd_debug_file):
            if os.path.exists(f):
                with open(f, 'r') as stream:
                    log_contents += '\n-------------------- %s ------------------\n\n' % (f,)
                    log_contents += stream.read()
        msg += ("\n\n===========================\nStdout: \n" + ''.join(stdout) +
            "\n\n===========================\nStderr:" + ''.join(stderr) +
            "\n\n===========================\nWriter Log:\n" + '\n'.join(getattr(writerThread, 'log', [])) +
            "\n\n===========================\nLog:" + log_contents)

        if IS_JYTHON:
            # It seems we have some spurious errors which make Jython tests flaky (on a test run it's
            # not unusual for one test among all the tests to fail with this error on Jython).
            # The usual traceback in this case is:
            #
            # Traceback (most recent call last):
            #   File "/home/travis/build/fabioz/PyDev.Debugger/_pydevd_bundle/pydevd_comm.py", line 287, in _on_run
            #     line = self._read_line()
            #   File "/home/travis/build/fabioz/PyDev.Debugger/_pydevd_bundle/pydevd_comm.py", line 270, in _read_line
            #     r = self.sock.recv(1024)
            #   File "/home/travis/build/fabioz/PyDev.Debugger/_pydevd_bundle/pydevd_comm.py", line 270, in _read_line
            #     r = self.sock.recv(1024)
            #   File "/home/travis/jython/Lib/_socket.py", line 1270, in recv
            #     data, _ = self._get_message(bufsize, "recv")
            #   File "/home/travis/jython/Lib/_socket.py", line 384, in handle_exception
            #     raise _map_exception(jlx)
            # error: [Errno -1] Unmapped exception: java.lang.NullPointerException
            #
            # So, ignore errors in this situation.

            if 'error: [Errno -1] Unmapped exception: java.lang.NullPointerException' in msg:
                raise IgnoreFailureError()
        raise AssertionError(msg)


#=======================================================================================================================
# AbstractWriterThread
#=======================================================================================================================
class AbstractWriterThread(threading.Thread):

    FORCE_KILL_PROCESS_WHEN_FINISHED_OK = False
    IS_MODULE = False
    TEST_FILE = None
    EXPECTED_RETURNCODE = 0

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.process = None  # Set after the process is created.
        self.daemon = True
        self.finished_ok = False
        self.finished_initialization = False
        self._next_breakpoint_id = 0
        self.log = []

    def run(self):
        self.start_socket()

    def check_test_suceeded_msg(self, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stdout)

    def update_command_line_args(self, args):
        return args

    def _ignore_stderr_line(self, line):
        if line.startswith((
            'debugger: ',
            '>>',
            '<<',
            'warning: Debugger speedups',
            'pydev debugger: New process is launching',
            'pydev debugger: To debug that process',
            '*** Multiprocess',
            )):
            return True

        for expected in (
            'PyDev console: using IPython',
            'Attempting to work in a virtualenv. If you encounter problems, please',
            'Unable to create basic Accelerated OpenGL',  # Issue loading qt5
            'Core Image is now using the software OpenGL',  # Issue loading qt5
            'XDG_RUNTIME_DIR not set',  # Issue loading qt5
            ):
            if expected in line:
                return True

        if re.match(r'^(\d+)\t(\d)+', line):
            return True

        if IS_JYTHON:
            for expected in (
                'org.python.netty.util.concurrent.DefaultPromise',
                'org.python.netty.util.concurrent.SingleThreadEventExecutor',
                'Failed to submit a listener notification task. Event loop shut down?',
                'java.util.concurrent.RejectedExecutionException',
                'An event executor terminated with non-empty task',
                'java.lang.UnsupportedOperationException',
                "RuntimeWarning: Parent module '_pydevd_bundle' not found while handling absolute import",
                'from _pydevd_bundle.pydevd_additional_thread_info_regular import _current_frames',
                'from _pydevd_bundle.pydevd_additional_thread_info import _current_frames',
                'import org.python.core as PyCore #@UnresolvedImport',
                'from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info',
                "RuntimeWarning: Parent module '_pydevd_bundle._debug_adapter' not found while handling absolute import",
                'import json',

                # Issues with Jython and Java 9.
                'WARNING: Illegal reflective access by org.python.core.PySystemState',
                'WARNING: Please consider reporting this to the maintainers of org.python.core.PySystemState',
                'WARNING: An illegal reflective access operation has occurred',
                'WARNING: Illegal reflective access by jnr.posix.JavaLibCHelper',
                'WARNING: Please consider reporting this to the maintainers of jnr.posix.JavaLibCHelper',
                'WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations',
                'WARNING: All illegal access operations will be denied in a future release',
                ):
                if expected in line:
                    return True

            if line.strip().startswith('at '):
                return True

        return False

    def additional_output_checks(self, stdout, stderr):
        lines_with_error = []
        for line in stderr.splitlines():
            line = line.strip()
            if not line:
                continue
            if not self._ignore_stderr_line(line):
                lines_with_error.append(line)

        if lines_with_error:
            raise AssertionError('Did not expect to have line(s) in stderr:\n\n%s\n\nFull stderr:\n\n%s' % (
                '\n'.join(lines_with_error), stderr))

    def get_environ(self):
        return None

    def get_pydevd_file(self):
        dirname = os.path.dirname(__file__)
        dirname = os.path.dirname(dirname)
        return os.path.abspath(os.path.join(dirname, 'pydevd.py'))

    def get_pydevconsole_file(self):
        dirname = os.path.dirname(__file__)
        dirname = os.path.dirname(dirname)
        return os.path.abspath(os.path.join(dirname, 'pydevconsole.py'))

    def get_line_index_with_content(self, line_content, filename=None):
        '''
        :return the line index which has the given content (1-based).
        '''
        if filename is None:
            filename = self.TEST_FILE
        with open(filename, 'r') as stream:
            for i_line, line in enumerate(stream):
                if line_content in line:
                    return i_line + 1
        raise AssertionError('Did not find: %s in %s' % (line_content, self.TEST_FILE))

    def get_cwd(self):
        return os.path.dirname(self.get_pydevd_file())

    def get_command_line_args(self):
        return [self.TEST_FILE]

    def do_kill(self):
        if hasattr(self, 'server_socket'):
            self.server_socket.close()
            delattr(self, 'server_socket')

        if hasattr(self, 'reader_thread'):
            # if it's not created, it's not there...
            self.reader_thread.do_kill()
            delattr(self, 'reader_thread')

        if hasattr(self, 'sock'):
            self.sock.close()
            delattr(self, 'sock')

        if hasattr(self, 'port'):
            delattr(self, 'port')

    def write_with_content_len(self, msg):
        self.log.append('write: %s' % (msg,))

        if SHOW_WRITES_AND_READS:
            print('Test Writer Thread Written %s' % (msg,))

        if not hasattr(self, 'sock'):
            print('%s.sock not available when sending: %s' % (self, msg))
            return

        if not isinstance(msg, bytes):
            msg = msg.encode('utf-8')

        self.sock.sendall((u'Content-Length: %s\r\n\r\n' % len(msg)).encode('ascii'))
        self.sock.sendall(msg)

    _WRITE_LOG_PREFIX = 'write: '

    def write(self, s):
        from _pydevd_bundle.pydevd_comm import ID_TO_MEANING
        meaning = ID_TO_MEANING.get(re.search(r'\d+', s).group(), '')
        if meaning:
            meaning += ': '

        self.log.append(self._WRITE_LOG_PREFIX + '%s%s' % (meaning, s,))

        if SHOW_WRITES_AND_READS:
            print('Test Writer Thread Written %s%s' % (meaning, s,))
        msg = s + '\n'

        if not hasattr(self, 'sock'):
            print('%s.sock not available when sending: %s' % (self, msg))
            return

        msg = msg.encode('utf-8')

        self.sock.send(msg)

    def get_next_message(self, context_message, timeout=None):
        return self.reader_thread.get_next_message(context_message, timeout=timeout)

    def start_socket(self, port=None):
        assert not hasattr(self, 'port'), 'Socket already initialized.'
        from _pydev_bundle.pydev_localhost import get_socket_name
        if SHOW_WRITES_AND_READS:
            print('start_socket')

        self._sequence = -1
        if port is None:
            socket_name = get_socket_name(close=True)
        else:
            socket_name = (pydev_localhost.get_localhost(), port)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(socket_name)
        self.port = socket_name[1]
        server_socket.listen(1)
        if SHOW_WRITES_AND_READS:
            print('Waiting in socket.accept()')
        self.server_socket = server_socket
        new_socket, addr = server_socket.accept()
        if SHOW_WRITES_AND_READS:
            print('Test Writer Thread Socket:', new_socket, addr)

        self._set_socket(new_socket)

    def _set_socket(self, new_socket):
        curr_socket = getattr(self, 'sock', None)
        if curr_socket:
            try:
                curr_socket.shutdown(socket.SHUT_WR)
            except:
                pass
            try:
                curr_socket.close()
            except:
                pass

        reader_thread = self.reader_thread = ReaderThread(new_socket)
        self.sock = new_socket
        reader_thread.start()

        # initial command is always the version
        self.write_version()
        self.log.append('start_socket')
        self.finished_initialization = True

    def start_socket_client(self, host, port):
        self._sequence = -1
        if SHOW_WRITES_AND_READS:
            print("Connecting to %s:%s" % (host, port))

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #  Set TCP keepalive on an open socket.
        #  It activates after 1 second (TCP_KEEPIDLE,) of idleness,
        #  then sends a keepalive ping once every 3 seconds (TCP_KEEPINTVL),
        #  and closes the connection after 5 failed ping (TCP_KEEPCNT), or 15 seconds
        try:
            from socket import IPPROTO_TCP, SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT
            s.setsockopt(socket.SOL_SOCKET, SO_KEEPALIVE, 1)
            s.setsockopt(IPPROTO_TCP, TCP_KEEPIDLE, 1)
            s.setsockopt(IPPROTO_TCP, TCP_KEEPINTVL, 3)
            s.setsockopt(IPPROTO_TCP, TCP_KEEPCNT, 5)
        except ImportError:
            pass  # May not be available everywhere.

        # 10 seconds default timeout
        timeout = int(os.environ.get('PYDEVD_CONNECT_TIMEOUT', 10))
        s.settimeout(timeout)
        for _i in range(20):
            try:
                s.connect((host, port))
                break
            except:
                time.sleep(.5)  # We may have to wait a bit more and retry (especially on PyPy).
        s.settimeout(None)  # no timeout after connected
        if SHOW_WRITES_AND_READS:
            print("Connected.")
        self._set_socket(s)
        return s

    def next_breakpoint_id(self):
        self._next_breakpoint_id += 1
        return self._next_breakpoint_id

    def next_seq(self):
        self._sequence += 2
        return self._sequence

    def wait_for_new_thread(self):
        # wait for hit breakpoint
        last = ''
        while not '<xml><thread name="' in last or '<xml><thread name="pydevd.' in last:
            last = self.get_next_message('wait_for_new_thread')

        # we have something like <xml><thread name="MainThread" id="12103472" /></xml>
        splitted = last.split('"')
        thread_id = splitted[3]
        return thread_id

    def wait_for_output(self):
        # Something as:
        # <xml><io s="TEST SUCEEDED%2521" ctx="1"/></xml>
        while True:
            msg = self.get_next_message('wait_output')
            if "<xml><io s=" in msg:
                if 'ctx="1"' in msg:
                    ctx = 'stdout'
                elif 'ctx="2"' in msg:
                    ctx = 'stderr'
                else:
                    raise AssertionError('IO message without ctx.')

                msg = unquote_plus(unquote_plus(msg.split('"')[1]))
                return msg, ctx

    def get_current_stack_hit(self, thread_id, **kwargs):
        self.write_get_thread_stack(thread_id)
        msg = self.wait_for_message(CMD_GET_THREAD_STACK)
        return self._get_stack_as_hit(msg, **kwargs)

    def wait_for_single_notification_as_hit(self, reason=REASON_STOP_ON_BREAKPOINT, **kwargs):
        dct = self.wait_for_json_message(CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION)
        assert dct['stop_reason'] == reason

        line = kwargs.pop('line', None)
        file = kwargs.pop('file', None)
        assert not kwargs, 'Unexpected kwargs: %s' % (kwargs,)

        return self.get_current_stack_hit(dct['thread_id'], line=line, file=file)

    def wait_for_breakpoint_hit(self, reason=REASON_STOP_ON_BREAKPOINT, timeout=None, **kwargs):
        '''
        108 is over
        109 is return
        111 is breakpoint

        :param reason: may be the actual reason (int or string) or a list of reasons.
        '''
        # note: those must be passed in kwargs.
        line = kwargs.get('line')
        file = kwargs.get('file')
        name = kwargs.get('name')

        self.log.append('Start: wait_for_breakpoint_hit')
        # wait for hit breakpoint
        if not isinstance(reason, (list, tuple)):
            reason = (reason,)

        def accept_message(last):
            for r in reason:
                if ('stop_reason="%s"' % (r,)) in last:
                    return True

            return False

        msg = self.wait_for_message(accept_message, timeout=timeout)
        return self._get_stack_as_hit(msg, file, line, name)

    def _get_stack_as_hit(self, msg, file=None, line=None, name=None):
        # we have something like <xml><thread id="12152656" stop_reason="111"><frame id="12453120" name="encode" ...
        if len(msg.thread.frame) == 0:
            frame = msg.thread.frame
        else:
            frame = msg.thread.frame[0]
        thread_id = msg.thread['id']
        frame_id = frame['id']
        suspend_type = msg.thread['suspend_type']
        hit_name = frame['name']
        frame_line = int(frame['line'])
        frame_file = frame['file']

        if file is not None:
            assert frame_file.endswith(file), 'Expected hit to be in file %s, was: %s' % (file, frame_file)

        if line is not None:
            assert line == frame_line, 'Expected hit to be in line %s, was: %s (in file: %s)' % (line, frame_line, frame_file)

        if name is not None:
            if not isinstance(name, (list, tuple, set)):
                assert name == hit_name
            else:
                assert hit_name in name

        self.log.append('End(1): wait_for_breakpoint_hit: %s' % (msg.original_xml,))

        return Hit(
            thread_id=thread_id, frame_id=frame_id, line=frame_line, suspend_type=suspend_type, name=hit_name, file=frame_file)

    def wait_for_get_next_statement_targets(self):
        last = ''
        while not '<xml><line>' in last:
            last = self.get_next_message('wait_for_get_next_statement_targets')

        matches = re.finditer(r"(<line>([0-9]*)<\/line>)", last, re.IGNORECASE)
        lines = []
        for _, match in enumerate(matches):
            try:
                lines.append(int(match.group(2)))
            except ValueError:
                pass
        return set(lines)

    def wait_for_custom_operation(self, expected):
        # wait for custom operation response, the response is double encoded
        expected_encoded = quote(quote_plus(expected))
        last = ''
        while not expected_encoded in last:
            last = self.get_next_message('wait_for_custom_operation. Expected (encoded): %s' % (expected_encoded,))

        return True

    def _is_var_in_last(self, expected, last):
        if expected in last:
            return True

        last = unquote_plus(last)
        if expected in last:
            return True

        # We actually quote 2 times on the backend...
        last = unquote_plus(last)
        if expected in last:
            return True

        return False

    def wait_for_multiple_vars(self, expected_vars):
        if not isinstance(expected_vars, (list, tuple)):
            expected_vars = [expected_vars]

        all_found = []
        ignored = []

        while True:
            try:
                last = self.get_next_message('wait_for_multiple_vars: %s' % (expected_vars,))
            except:
                missing = []
                for v in expected_vars:
                    if v not in all_found:
                        missing.append(v)
                raise ValueError('Not Found:\n%s\nNot found messages: %s\nFound messages: %s\nExpected messages: %s\nIgnored messages:\n%s' % (
                    '\n'.join(str(x) for x in missing), len(missing), len(all_found), len(expected_vars), '\n'.join(str(x) for x in ignored)))

            was_message_used = False
            new_expected = []
            for expected in expected_vars:
                found_expected = False
                if isinstance(expected, (tuple, list)):
                    for e in expected:
                        if self._is_var_in_last(e, last):
                            was_message_used = True
                            found_expected = True
                            all_found.append(expected)
                            break
                else:
                    if self._is_var_in_last(expected, last):
                        was_message_used = True
                        found_expected = True
                        all_found.append(expected)

                if not found_expected:
                    new_expected.append(expected)

            expected_vars = new_expected

            if not expected_vars:
                return True

            if not was_message_used:
                ignored.append(last)

    wait_for_var = wait_for_multiple_vars
    wait_for_vars = wait_for_multiple_vars
    wait_for_evaluation = wait_for_multiple_vars

    def write_make_initial_run(self):
        self.write("101\t%s\t" % self.next_seq())
        self.log.append('write_make_initial_run')

    def write_set_protocol(self, protocol):
        self.write("%s\t%s\t%s" % (CMD_SET_PROTOCOL, self.next_seq(), protocol))

    def write_authenticate(self, access_token, client_access_token):
        msg = "%s\t%s\t%s" % (CMD_AUTHENTICATE, self.next_seq(), access_token)
        self.write(msg)

        self.wait_for_message(lambda msg:client_access_token in msg, expect_xml=False)

    def write_version(self):
        from _pydevd_bundle.pydevd_constants import IS_WINDOWS
        self.write("%s\t%s\t1.0\t%s\tID" % (CMD_VERSION, self.next_seq(), 'WINDOWS' if IS_WINDOWS else 'UNIX'))

    def get_main_filename(self):
        return self.TEST_FILE

    def write_show_return_vars(self, show=1):
        self.write("%s\t%s\tCMD_SHOW_RETURN_VALUES\t%s" % (CMD_SHOW_RETURN_VALUES, self.next_seq(), show))

    def write_add_breakpoint(self, line, func='None', filename=None, hit_condition=None, is_logpoint=False, suspend_policy=None, condition=None):
        '''
        :param line: starts at 1
        :param func: if None, may hit in any context, empty string only top level, otherwise must be method name.
        '''
        if filename is None:
            filename = self.get_main_filename()
        breakpoint_id = self.next_breakpoint_id()
        if hit_condition is None and not is_logpoint and suspend_policy is None and condition is None:
            # Format kept for backward compatibility tests
            self.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\tNone\tNone" % (
                CMD_SET_BREAK, self.next_seq(), breakpoint_id, 'python-line', filename, line, func))
        else:
            # Format: breakpoint_id, type, file, line, func_name, condition, expression, hit_condition, is_logpoint, suspend_policy
            self.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tNone\t%s\t%s\t%s" % (
                CMD_SET_BREAK, self.next_seq(), breakpoint_id, 'python-line', filename, line, func, condition, hit_condition, is_logpoint, suspend_policy))
        self.log.append('write_add_breakpoint: %s line: %s func: %s' % (breakpoint_id, line, func))
        return breakpoint_id

    def write_multi_threads_single_notification(self, multi_threads_single_notification):
        self.write_json_config(dict(
            multi_threads_single_notification=multi_threads_single_notification,
        ))

    def write_suspend_on_breakpoint_exception(self, skip_suspend_on_breakpoint_exception, skip_print_breakpoint_exception):
        self.write_json_config(dict(
            skip_suspend_on_breakpoint_exception=skip_suspend_on_breakpoint_exception,
            skip_print_breakpoint_exception=skip_print_breakpoint_exception
        ))

    def write_json_config(self, config_dict):
        self.write("%s\t%s\t%s" % (CMD_PYDEVD_JSON_CONFIG, self.next_seq(),
            json.dumps(config_dict)
        ))

    def write_stop_on_start(self, stop=True):
        self.write("%s\t%s\t%s" % (CMD_STOP_ON_START, self.next_seq(), stop))

    def write_dump_threads(self):
        self.write("%s\t%s\t" % (CMD_THREAD_DUMP_TO_STDERR, self.next_seq()))

    def write_add_exception_breakpoint(self, exception):
        self.write("%s\t%s\t%s" % (CMD_ADD_EXCEPTION_BREAK, self.next_seq(), exception))
        self.log.append('write_add_exception_breakpoint: %s' % (exception,))

    def write_get_current_exception(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_GET_EXCEPTION_DETAILS, self.next_seq(), thread_id))

    def write_set_py_exception_globals(
            self,
            break_on_uncaught,
            break_on_caught,
            skip_on_exceptions_thrown_in_same_context,
            ignore_exceptions_thrown_in_lines_with_ignore_exception,
            ignore_libraries,
            exceptions=()
        ):
        # Only set the globals, others
        self.write("131\t%s\t%s" % (self.next_seq(), '%s;%s;%s;%s;%s;%s' % (
            'true' if break_on_uncaught else 'false',
            'true' if break_on_caught else 'false',
            'true' if skip_on_exceptions_thrown_in_same_context else 'false',
            'true' if ignore_exceptions_thrown_in_lines_with_ignore_exception else 'false',
            'true' if ignore_libraries else 'false',
            ';'.join(exceptions)
        )))
        self.log.append('write_set_py_exception_globals')

    def write_start_redirect(self):
        self.write("%s\t%s\t%s" % (CMD_REDIRECT_OUTPUT, self.next_seq(), 'STDERR STDOUT'))

    def write_set_project_roots(self, project_roots):
        self.write("%s\t%s\t%s" % (CMD_SET_PROJECT_ROOTS, self.next_seq(), '\t'.join(str(x) for x in project_roots)))

    def write_add_exception_breakpoint_with_policy(
            self, exception, notify_on_handled_exceptions, notify_on_unhandled_exceptions, ignore_libraries):
        self.write("%s\t%s\t%s" % (CMD_ADD_EXCEPTION_BREAK, self.next_seq(), '\t'.join(str(x) for x in [
            exception, notify_on_handled_exceptions, notify_on_unhandled_exceptions, ignore_libraries])))
        self.log.append('write_add_exception_breakpoint: %s' % (exception,))

    def write_remove_exception_breakpoint(self, exception):
        self.write('%s\t%s\t%s' % (CMD_REMOVE_EXCEPTION_BREAK, self.next_seq(), exception))

    def write_remove_breakpoint(self, breakpoint_id):
        self.write("%s\t%s\t%s\t%s\t%s" % (
            CMD_REMOVE_BREAK, self.next_seq(), 'python-line', self.get_main_filename(), breakpoint_id))

    def write_change_variable(self, thread_id, frame_id, varname, value):
        self.write("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            CMD_CHANGE_VARIABLE, self.next_seq(), thread_id, frame_id, 'FRAME', varname, value))

    def write_get_frame(self, thread_id, frame_id):
        self.write("%s\t%s\t%s\t%s\tFRAME" % (CMD_GET_FRAME, self.next_seq(), thread_id, frame_id))
        self.log.append('write_get_frame')

    def write_get_variable(self, thread_id, frame_id, var_attrs):
        self.write("%s\t%s\t%s\t%s\tFRAME\t%s" % (CMD_GET_VARIABLE, self.next_seq(), thread_id, frame_id, var_attrs))

    def write_step_over(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_OVER, self.next_seq(), thread_id,))

    def write_step_in(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_INTO, self.next_seq(), thread_id,))

    def write_step_in_my_code(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_INTO_MY_CODE, self.next_seq(), thread_id,))

    def write_step_return(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_RETURN, self.next_seq(), thread_id,))

    def write_step_return_my_code(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_RETURN_MY_CODE, self.next_seq(), thread_id,))

    def write_step_over_my_code(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_STEP_OVER_MY_CODE, self.next_seq(), thread_id,))

    def write_suspend_thread(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_THREAD_SUSPEND, self.next_seq(), thread_id,))

    def write_reload(self, module_name):
        self.log.append('write_reload')
        self.write("%s\t%s\t%s" % (CMD_RELOAD_CODE, self.next_seq(), module_name,))

    def write_run_thread(self, thread_id):
        self.log.append('write_run_thread')
        self.write("%s\t%s\t%s" % (CMD_THREAD_RUN, self.next_seq(), thread_id,))

    def write_get_thread_stack(self, thread_id):
        self.log.append('write_get_thread_stack')
        self.write("%s\t%s\t%s" % (CMD_GET_THREAD_STACK, self.next_seq(), thread_id,))

    def write_load_source(self, filename):
        self.log.append('write_load_source')
        self.write("%s\t%s\t%s" % (CMD_LOAD_SOURCE, self.next_seq(), filename,))

    def write_load_source_from_frame_id(self, frame_id):
        from _pydevd_bundle.pydevd_comm_constants import CMD_LOAD_SOURCE_FROM_FRAME_ID
        self.log.append('write_load_source_from_frame_id')
        self.write("%s\t%s\t%s" % (CMD_LOAD_SOURCE_FROM_FRAME_ID, self.next_seq(), frame_id,))

    def write_kill_thread(self, thread_id):
        self.write("%s\t%s\t%s" % (CMD_THREAD_KILL, self.next_seq(), thread_id,))

    def write_set_next_statement(self, thread_id, line, func_name):
        self.write("%s\t%s\t%s\t%s\t%s" % (CMD_SET_NEXT_STATEMENT, self.next_seq(), thread_id, line, func_name,))

    def write_smart_step_into(self, thread_id, line, func_name):
        self.write("%s\t%s\t%s\t%s\t%s" % (CMD_SMART_STEP_INTO, self.next_seq(), thread_id, line, func_name,))

    def write_debug_console_expression(self, locator):
        self.write("%s\t%s\t%s" % (CMD_EVALUATE_CONSOLE_EXPRESSION, self.next_seq(), locator))

    def write_custom_operation(self, locator, style, codeOrFile, operation_fn_name):
        self.write("%s\t%s\t%s||%s\t%s\t%s" % (
            CMD_RUN_CUSTOM_OPERATION, self.next_seq(), locator, style, quote_plus(codeOrFile), operation_fn_name))

    def write_evaluate_expression(self, locator, expression):
        self.write("%s\t%s\t%s\t%s\t1" % (CMD_EVALUATE_EXPRESSION, self.next_seq(), locator, expression))

    def write_enable_dont_trace(self, enable):
        if enable:
            enable = 'true'
        else:
            enable = 'false'
        self.write("%s\t%s\t%s" % (CMD_ENABLE_DONT_TRACE, self.next_seq(), enable))

    def write_get_next_statement_targets(self, thread_id, frame_id):
        self.write("201\t%s\t%s\t%s" % (self.next_seq(), thread_id, frame_id))
        self.log.append('write_get_next_statement_targets')

    def write_list_threads(self):
        seq = self.next_seq()
        self.write("%s\t%s\t" % (CMD_LIST_THREADS, seq))
        return seq

    def wait_for_list_threads(self, seq):
        return self.wait_for_message('502')

    def wait_for_get_thread_stack_message(self):
        return self.wait_for_message(CMD_GET_THREAD_STACK)

    def wait_for_curr_exc_stack(self):
        return self.wait_for_message(CMD_SEND_CURR_EXCEPTION_TRACE)

    def wait_for_json_message(self, accept_message, unquote_msg=True, timeout=None):
        last = self.wait_for_message(accept_message, unquote_msg, expect_xml=False, timeout=timeout)
        json_msg = last.split('\t', 2)[-1]  # We have something as: CMD\tSEQ\tJSON
        if isinstance(json_msg, bytes):
            json_msg = json_msg.decode('utf-8')
        try:
            return json.loads(json_msg)
        except:
            traceback.print_exc()
            raise AssertionError('Unable to parse:\n%s\njson:\n%s' % (last, json_msg))

    def wait_for_message(self, accept_message, unquote_msg=True, expect_xml=True, timeout=None, double_unquote=True):
        if isinstance(accept_message, (str, int)):
            msg_starts_with = '%s\t' % (accept_message,)

            def accept_message(msg):
                return msg.startswith(msg_starts_with)

        import untangle
        from io import StringIO
        prev = None
        while True:
            last = self.get_next_message('wait_for_message', timeout=timeout)
            if unquote_msg:
                last = unquote_plus(last)
                if double_unquote:
                    # This is useful if the checking will be done without needing to unpack the
                    # actual xml (in which case we'll be unquoting things inside of attrs --
                    # this could actually make the xml invalid though).
                    last = unquote_plus(last)
            if accept_message(last):
                if expect_xml:
                    # Extract xml and return untangled.
                    xml = ''
                    try:
                        xml = last[last.index('<xml>'):]
                        if isinstance(xml, bytes):
                            xml = xml.decode('utf-8')
                        xml = untangle.parse(StringIO(xml))
                    except:
                        traceback.print_exc()
                        raise AssertionError('Unable to parse:\n%s\nxml:\n%s' % (last, xml))
                    ret = xml.xml
                    ret.original_xml = last
                    return ret
                else:
                    return last
            if prev != last:
                sys.stderr.write('Ignored message: %r\n' % (last,))
                # Uncomment to know where in the stack it was ignored.
                # import traceback
                # traceback.print_stack(limit=7)

            prev = last

    def wait_for_untangled_message(self, accept_message, timeout=None, double_unquote=False):
        import untangle
        from io import StringIO
        prev = None
        while True:
            last = self.get_next_message('wait_for_message', timeout=timeout)
            last = unquote_plus(last)
            if double_unquote:
                last = unquote_plus(last)
            # Extract xml with untangled.
            xml = ''
            try:
                xml = last[last.index('<xml>'):]
            except:
                traceback.print_exc()
                raise AssertionError('Unable to find xml in: %s' % (last,))

            try:
                if isinstance(xml, bytes):
                    xml = xml.decode('utf-8')
                xml = untangle.parse(StringIO(xml))
            except:
                traceback.print_exc()
                raise AssertionError('Unable to parse:\n%s\nxml:\n%s' % (last, xml))
            untangled = xml.xml
            cmd_id = last.split('\t', 1)[0]
            if accept_message(int(cmd_id), untangled):
                return untangled
            if prev != last:
                print('Ignored message: %r' % (last,))

            prev = last

    def get_frame_names(self, thread_id):
        self.write_get_thread_stack(thread_id)
        msg = self.wait_for_message(CMD_GET_THREAD_STACK)
        if msg.thread.frame:
            frame_names = [frame['name'] for frame in msg.thread.frame]
            return frame_names
        return [msg.thread.frame['name']]

    def get_step_into_variants(self, thread_id, frame_id, start_line, end_line):
        self.write("%s\t%s\t%s\t%s\t%s\t%s" % (CMD_GET_SMART_STEP_INTO_VARIANTS, self.next_seq(), thread_id, frame_id, start_line, end_line))
        msg = self.wait_for_message(CMD_GET_SMART_STEP_INTO_VARIANTS)
        if msg.variant:
            variant_info = [
                (variant['name'], variant['isVisited'], variant['line'], variant['callOrder'], variant['offset'], variant['childOffset'])
                for variant in msg.variant
            ]
            return variant_info
        return []

    def wait_for_thread_join(self, main_thread_id):

        def condition():
            return self.get_frame_names(main_thread_id) in (
                ['wait', 'join', '<module>'],
                ['_wait_for_tstate_lock', 'join', '<module>'],
                ['_wait_for_tstate_lock', 'join', '<module>', '_run_code', '_run_module_code', 'run_path'],
            )

        def msg():
            return 'Found stack: %s' % (self.get_frame_names(main_thread_id),)

        wait_for_condition(condition, msg, timeout=5, sleep=.5)

    def create_request_thread(self, full_url):

        class T(threading.Thread):

            def wait_for_contents(self):
                for _ in range(10):
                    if hasattr(self, 'contents'):
                        break
                    time.sleep(.3)
                else:
                    raise AssertionError('Unable to get contents from server. Url: %s' % (full_url,))
                return self.contents

            def run(self):
                try:
                    from urllib.request import urlopen
                except ImportError:
                    from urllib import urlopen
                for _ in range(10):
                    try:
                        stream = urlopen(full_url)
                        contents = stream.read()
                        contents = contents.decode('utf-8')
                        self.contents = contents
                        break
                    except IOError:
                        continue

        t = T()
        t.daemon = True
        return t


def _get_debugger_test_file(filename):
    ret = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
    if not os.path.exists(ret):
        ret = os.path.join(os.path.dirname(__file__), 'resources', filename)
    if not os.path.exists(ret):
        raise AssertionError('Expected: %s to exist.' % (ret,))
    return ret


def get_free_port():
    from _pydev_bundle.pydev_localhost import get_socket_name
    return get_socket_name(close=True)[1]
