from contextlib import contextmanager

import pytest

from _pydev_bundle.pydev_override import overrides
from tests_python.debugger_fixtures import DebuggerRunnerSimple, debugger_runner_simple
from tests_python.debugger_unittest import AbstractWriterThread, SHOW_OTHER_DEBUG_INFO, \
    start_in_daemon_thread, wait_for_condition, IS_JYTHON
from _pydev_bundle.pydev_localhost import get_socket_names, get_socket_name
from _pydev_bundle.pydev_imports import xmlrpclib
from _pydev_bundle.pydev_imports import _queue as queue
from _pydev_bundle.pydev_imports import SimpleXMLRPCServer
import time
import socket

builtin_qualifier = "builtins"


@pytest.fixture
def console_setup(tmpdir):

    server_queue = queue.Queue()

    def notify_finished(more):
        server_queue.put(('notify_finished', more))
        return ''

    class ConsoleRunner(DebuggerRunnerSimple):

        @overrides(DebuggerRunnerSimple.add_command_line_args)
        def add_command_line_args(self, args, dap=False):
            port, client_port = get_socket_names(2, close=True)
            args.extend((
                writer.get_pydevconsole_file(),
                str(port[1]),
                str(client_port[1])
            ))
            self.port = port
            self.client_port = client_port

            server = SimpleXMLRPCServer(client_port)
            server.register_function(notify_finished, "NotifyFinished")
            start_in_daemon_thread(server.serve_forever, [])

            self.proxy = xmlrpclib.ServerProxy("http://%s:%s/" % port)

            return args

    class WriterThread(AbstractWriterThread):

        if IS_JYTHON:
            EXPECTED_RETURNCODE = 'any'

        @overrides(AbstractWriterThread.additional_output_checks)
        def additional_output_checks(self, stdout, stderr):
            print('output found: %s - %s' % (stdout, stderr))

        @overrides(AbstractWriterThread.write_dump_threads)
        def write_dump_threads(self):
            pass  # no-op (may be called on timeout).

        def execute_line(self, command, more=False):
            runner.proxy.execLine(command)
            assert server_queue.get(timeout=5.) == ('notify_finished', more)

        def hello(self):

            def _hello():
                try:
                    msg = runner.proxy.hello('ignored')
                    if msg is not None:
                        if isinstance(msg, (list, tuple)):
                            msg = next(iter(msg))
                        if msg.lower().startswith('hello'):
                            return True
                except:
                    # That's ok, communication still not ready.
                    pass

                return False

            wait_for_condition(_hello)

        def close(self):
            try:
                runner.proxy.close()
            except:
                # Ignore any errors on close.
                pass

        def connect_to_debugger(self, debugger_port):
            runner.proxy.connectToDebugger(debugger_port)

    runner = ConsoleRunner(tmpdir)
    writer = WriterThread()

    class CaseSetup(object):

        @contextmanager
        def check_console(
                self,
                **kwargs
            ):
            for key, value in kwargs.items():
                assert hasattr(WriterThread, key)
                setattr(WriterThread, key, value)

            self.writer = writer

            args = runner.get_command_line()

            args = runner.add_command_line_args(args)

            if SHOW_OTHER_DEBUG_INFO:
                print('executing: %s' % (' '.join(args),))
            try:
                with runner.run_process(args, writer) as dct_with_stdout_stder:
                    writer.get_stdout = lambda: ''.join(dct_with_stdout_stder['stdout'])
                    writer.get_stderr = lambda: ''.join(dct_with_stdout_stder['stderr'])

                    # Make sure communication is setup.
                    writer.hello()
                    yield writer
            finally:
                writer.log = []

            stdout = dct_with_stdout_stder['stdout']
            stderr = dct_with_stdout_stder['stderr']
            writer.additional_output_checks(''.join(stdout), ''.join(stderr))

    return CaseSetup()


def test_console_simple(console_setup):
    with console_setup.check_console() as writer:
        writer.execute_line('a = 10')
        writer.execute_line('print("TEST SUCEEDED")')
        writer.close()
        writer.finished_ok = True


def test_console_debugger_connected(console_setup):

    class _DebuggerWriterThread(AbstractWriterThread):

        FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True

        def __init__(self):
            AbstractWriterThread.__init__(self)
            socket_name = get_socket_name(close=True)
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(socket_name)
            server_socket.listen(1)
            self.port = socket_name[1]
            self.__server_socket = server_socket

        def run(self):
            print('waiting for second process')
            self.sock, addr = self.__server_socket.accept()
            print('accepted second process')

            from tests_python.debugger_unittest import ReaderThread
            self.reader_thread = ReaderThread(self.sock)
            self.reader_thread.start()

            self._sequence = -1
            # initial command is always the version
            self.write_version()
            self.log.append('start_socket')
            self.write_make_initial_run()
            time.sleep(1)

            seq = self.write_list_threads()
            msg = self.wait_for_list_threads(seq)
            assert msg.thread['name'] == 'MainThread'
            assert msg.thread['id'] == 'console_main'

            self.write_get_frame('console_main', '1')
            self.wait_for_vars([
                [
                    '<var name="a" type="int" qualifier="%s" value="int: 10"' % (builtin_qualifier,),
                    '<var name="a" type="int"  value="int',  # jython
                ],
            ])

            self.finished_ok = True

    with console_setup.check_console() as writer:
        writer.execute_line('a = 10')

        debugger_writer_thread = _DebuggerWriterThread()
        debugger_writer_thread.start()
        writer.connect_to_debugger(debugger_writer_thread.port)

        wait_for_condition(lambda: debugger_writer_thread.finished_ok)
        writer.execute_line('print("TEST SUCEEDED")')

        writer.close()
        writer.finished_ok = True

